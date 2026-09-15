"""Fixed two-GPU, native full-frame compression training with resumable updates."""
import argparse
import contextlib
import copy
import json
import math
import os
import subprocess
import time
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .data import DOMAINS, sample, load_indices, sha256
from .evaluate import evaluate, save_report
from .model import ARCHITECTURE, UPSTREAM_COMMIT, Student, teacher, initialize
from .profile import profile


def git_head(path):
    return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()


def verify_config(c):
    if c['architecture']!=ARCHITECTURE or c['world_size']!=2 or c['accumulate']!=4 or c['frames']!=6:
        raise ValueError('Recipe mismatch; no automatic architecture/batch/T changes')
    if c.get('train_mix')!='GoPro6 / DVD1 / BSD3ms24ms1 throughout all 100000 updates':
        raise ValueError('Old sampling config: all updates must use all three datasets; preserve old run for migration')
    if sha256(c['manifest'])!=c['manifest_sha256']: raise ValueError('Manifest changed')
    if sha256(c['teacher_checkpoint'])!=c['teacher_sha256']: raise ValueError('Teacher changed')
    if git_head(c['upstream'])!=UPSTREAM_COMMIT: raise ValueError('Upstream commit changed')
    if git_head(Path(__file__).resolve().parents[1])!=c['source_commit']: raise ValueError('Source commit changed')
    return json.loads(Path(c['manifest']).read_text())


def configure(args):
    p=Path(args.config)
    if p.exists():
        verify_config(json.loads(p.read_text()));print(f'Reusing frozen config: {p}');return
    if git_head(args.upstream)!=UPSTREAM_COMMIT: raise ValueError('Pin official upstream first')
    c={'architecture':ARCHITECTURE,'upstream':str(Path(args.upstream).resolve()),
       'source_commit':git_head(Path(__file__).resolve().parents[1]),
       'manifest':str(Path(args.manifest).resolve()),'manifest_sha256':sha256(args.manifest),
       'teacher_checkpoint':str(Path(args.teacher_checkpoint).resolve()),'teacher_sha256':sha256(args.teacher_checkpoint),
       'frames':6,'world_size':2,'accumulate':4,'global_batch':8,'microbatch_per_gpu':1,
       'total_updates':100000,'distill_updates':80000,'validate_every':5000,'save_every':1000,
       'lr':.0001,'min_lr':.000001,'seed':20260915,'precision':'BF16_AMP',
       'crop_size':0,'resize':False,'spatial_tiling':False,
       'train_mix':'GoPro6 / DVD1 / BSD3ms24ms1 throughout all 100000 updates',
       'target':'full GoPro test1111 RGB8 PSNR >=33; <=500 GFLOPs/native1080p output',
       'output':str(Path(args.output).resolve())}
    manifest=verify_config(c)
    if manifest['frames']!=6: raise ValueError('Manifest T mismatch')
    budget=profile(c['upstream'])
    if not budget['budget_pass']: raise ValueError('FLOPS_BUDGET_FAIL')
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(c,indent=2)+'\n')
    Path(c['output']).mkdir(parents=True,exist_ok=True)
    (Path(c['output'])/'profile_student.json').write_text(json.dumps(budget,indent=2)+'\n')
    print(json.dumps(c,indent=2))


def losses(pred,gt,target=None,kd_weight=.2):
    pred,gt=pred.float(),gt.float()
    supervised=((pred-gt).square()+1e-6).sqrt().mean()
    # Small gradient penalty preserves edges without a learned GAN/perceptual loss.
    edge=(pred[...,1:,:]-pred[...,:-1,:] - gt[...,1:,:]+gt[...,:-1,:]).abs().mean()
    edge+=(pred[...,:,1:]-pred[...,:,:-1] - gt[...,:,1:]+gt[...,:,:-1]).abs().mean()
    kd=pred.new_zeros(())
    if target is not None:
        target=target.float()
        # Do not force a GoPro teacher's incorrect cross-domain predictions.
        with torch.no_grad():
            good=(target-gt).abs().mean(2,keepdim=True)<(pred.detach()-gt).abs().mean(2,keepdim=True)
        kd=((pred-target).abs()*good).mean()
    return supervised+.01*edge+kd_weight*kd, {'supervised':supervised.detach(),'edge':edge.detach(),'kd':kd.detach()}


def atomic_save(value,path):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,path)


def save_state(model,opt,update,best,c,path):
    atomic_save({'architecture':ARCHITECTURE,'model':model.state_dict(),'optimizer':opt.state_dict(),
                 'update':update,'best_val_psnr':best,'config':c,'deployed':False},path)


def setup():
    if int(os.environ.get('WORLD_SIZE','0'))!=2: raise RuntimeError('Exactly two torchrun processes required')
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    torch.set_num_threads(1);torch.cuda.set_device(local)
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16-capable GPU required (A100)')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(hours=2))
    torch.manual_seed(20260915);torch.cuda.manual_seed_all(20260915)
    return rank,torch.device('cuda',local)


def step(ddp,teacher_model,opt,batches,device,lr,kd_weight):
    for pg in opt.param_groups: pg['lr']=lr
    opt.zero_grad(set_to_none=True);values=[]
    for micro,batch in enumerate(batches):
        x=batch['blur'][None].to(device);y=batch['gt'][None].to(device)
        target=None
        if kd_weight:
            with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'): target=teacher_model(x).float()
            if not torch.isfinite(target).all(): raise RuntimeError('Non-finite teacher target')
        ctx=ddp.no_sync() if micro<len(batches)-1 else contextlib.nullcontext()
        with ctx:
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'): pred=ddp(x)
            loss,detail=losses(pred,y,target,kd_weight)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
            (loss/len(batches)).backward()
        values.append(loss.detach())
        del pred,target,x,y,loss,detail
    norm=torch.nn.utils.clip_grad_norm_(ddp.parameters(),1.,error_if_nonfinite=True)
    opt.step()
    value=torch.stack(values).mean();dist.all_reduce(value);value/=2
    return value.item(),float(norm)


def train(args):
    c=json.loads(Path(args.config).read_text());manifest=verify_config(c)
    rank,device=setup();out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
    model=Student(c['upstream'],True)
    init=initialize(model,c['teacher_checkpoint'])
    if rank==0: (out/'initialization.json').write_text(json.dumps(init,indent=2)+'\n')
    model=model.to(device).train();t=teacher(c['upstream'],c['teacher_checkpoint']).to(device)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['lr'],weight_decay=1e-4)
    begin=0;best=-float('inf');latest=out/'latest.pth'
    if latest.exists() and not args.preflight:
        raw=torch.load(latest,map_location=device,weights_only=False)
        if raw['config']!=c: raise ValueError('Resume config differs')
        model.load_state_dict(raw['model'],strict=True);opt.load_state_dict(raw['optimizer'])
        begin=raw['update'];best=raw['best_val_psnr']
    ddp=DDP(model,device_ids=[device.index],broadcast_buffers=False)
    if args.preflight:
        rows=[]
        for d in DOMAINS:
            record=max((r for r in manifest['train']+manifest['val'] if r['domain']==d),key=lambda r:r['height']*r['width'])
            x,y=load_indices(record,range(c['frames']))
            batch={'blur':x,'gt':y}
            torch.cuda.reset_peak_memory_stats(device);started=time.perf_counter()
            value,norm=step(ddp,t,opt,[batch]*4,device,c['lr'],.2)
            torch.cuda.synchronize()
            rows.append({'domain':d,'sequence':record['name'],'shape':list(x.shape),
                'seconds':time.perf_counter()-started,'loss':value,'grad_norm':norm,
                'peak_gib':torch.cuda.max_memory_allocated(device)/1024**3})
            del x,y,batch
        # Verify actual deploy parameter fusion numerically on the current GPU.
        ddp.eval();probe=torch.randn(1,2,3,32,40,device=device)
        with torch.no_grad():
            a=model(probe);deployed=copy.deepcopy(model).deploy();b=deployed(probe)
            diff=(a-b).abs().max().item()
            torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-5)
        report={'status':'PREFLIGHT_PASS','rank':rank,'gpu':torch.cuda.get_device_name(device),
                'config_sha256':sha256(args.config),'rows':rows,'deploy_max_abs':diff}
        (out/f'preflight_rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
        dist.barrier();dist.destroy_process_group();return
    for r in range(2):
        p=json.loads((out/f'preflight_rank{r}.json').read_text())
        if p['config_sha256']!=sha256(args.config) or p['status']!='PREFLIGHT_PASS': raise RuntimeError('Run matching preflight first')
    started=time.perf_counter()
    for update in range(begin,c['total_updates']):
        final=update>=c['distill_updates']
        if final:
            progress=(update-c['distill_updates'])/(c['total_updates']-c['distill_updates'])
            lr=c['min_lr']+(.00002-c['min_lr'])*.5*(1+math.cos(math.pi*progress));kd=0.
        else:
            lr=c['min_lr']+(c['lr']-c['min_lr'])*.5*(1+math.cos(math.pi*update/c['distill_updates']))
            lr*=min(1.,(update+1)/500);kd=.2
        # Keep host RAM bounded: decode each microbatch just before consumption.
        class Batches:
            def __len__(self): return 4
            def __iter__(self):
                for micro in range(4): yield sample(manifest,update,micro,rank)
        value,norm=step(ddp,t,opt,Batches(),device,lr,kd)
        done=update+1
        if rank==0 and (done%50==0 or done==begin+1):
            seconds=(time.perf_counter()-started)/(done-begin)
            info={'update':done,'total':c['total_updates'],'loss':value,'grad_norm':norm,'lr':lr,
                  'seconds_per_update_including_validation':seconds,'eta_hours':seconds*(c['total_updates']-done)/3600,
                  'phase':'mixed_refine' if final else 'mixed_distill'}
            print(json.dumps(info),flush=True)
            with (out/'training.jsonl').open('a') as f:f.write(json.dumps(info)+'\n')
        if done%c['validate_every']==0 or done==c['total_updates']:
            dist.barrier()
            if rank==0:
                # Selection sees only fixed training holdout. Test never used here.
                report=evaluate(model,manifest['val'],device,c['frames'],max_chunks=2)
                save_report(report,out/f'val_{done:06d}.json')
                score=report['summary']['gopro']['psnr_rgb8']
                if score>best:
                    best=score;save_state(model,opt,done,best,c,out/'best_gopro_val.pth')
                save_state(model,opt,done,best,c,latest)
            dist.barrier()
        elif done%c['save_every']==0 and rank==0:
            save_state(model,opt,done,best,c,latest)
    if rank==0:
        (out/'training_complete.json').write_text(json.dumps({'status':'TRAINING_COMPLETE_NOT_YET_ACCEPTED',
            'update':c['total_updates'],'best_student_holdout_psnr':best})+'\n')
    dist.barrier();dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('configure');s.add_argument('--config',required=True);s.add_argument('--upstream',required=True)
    s.add_argument('--teacher-checkpoint',required=True);s.add_argument('--manifest',required=True);s.add_argument('--output',required=True)
    t=sub.add_parser('train');t.add_argument('--config',required=True);t.add_argument('--preflight',action='store_true')
    a=p.parse_args()
    if a.command=='configure': configure(a)
    else: train(a)

if __name__=='__main__': main()
