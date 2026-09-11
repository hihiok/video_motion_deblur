"""One/two-GPU native-frame mixed-domain KD with resumable atomic checkpoints."""
import argparse
from datetime import timedelta
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from .data import Clips, DOMAINS, load_indices, sha256
from .evaluate import evaluate
from .losses import loss_terms
from .model import ShiftModel, load_teacher, initialize_student

STOP=False

def request_stop(*_):
    global STOP
    STOP=True


def atomic_save(value,path):
    path=Path(path);tmp=path.with_suffix('.tmp')
    torch.save(value,tmp);tmp.replace(path)


def worker_init(_):
    torch.set_num_threads(1)


def train(a):
    c=json.loads(Path(a.config).read_text())
    if sha256(c['manifest'])!=c['manifest_sha256'] or sha256(c['teacher_checkpoint'])!=c['teacher_sha256']:
        raise ValueError('Input provenance changed')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    local=int(os.environ.get('LOCAL_RANK',0))
    if world not in (1,2):raise ValueError('Use one or two GPUs per model')
    torch.set_num_threads(2);torch.cuda.set_device(local);device=torch.device('cuda',local)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(hours=4))
    random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    manifest=json.loads(Path(c['manifest']).read_text())
    out=Path(c['output'])/a.variant;out.mkdir(parents=True,exist_ok=True)
    model=ShiftModel(c['upstream'],a.variant,activation_checkpointing=True).to(device)
    teacher=ShiftModel(c['upstream'],'teacher').to(device).eval().requires_grad_(False)
    load_teacher(teacher,c['teacher_checkpoint'])
    transfer=initialize_student(model,teacher)
    opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],betas=(.9,.99),weight_decay=0)
    start=0;best={'gopro':-float('inf'),'balanced':-float('inf')}
    resume=Path(a.resume) if a.resume else out/'latest.pth'
    if resume.exists() and not a.preflight:
        state=torch.load(resume,map_location='cpu',weights_only=False)
        if state['variant']!=a.variant or state['config']!=c:raise ValueError('Resume provenance/config mismatch')
        model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
        start=state['update'];best=state['best']
    elif a.resume and not a.preflight:
        raise FileNotFoundError(a.resume)
    if rank==0:(out/'initialization.json').write_text(json.dumps(transfer,indent=2)+'\n')
    total=c['total_updates']
    if a.preflight:
        if world!=1:raise ValueError('Preflight is single GPU; later DDP uses same per-GPU microbatch')
        reports=[]
        for d in DOMAINS:
            r=max((r for r in manifest['train'] if r['domain']==d),key=lambda r:r['height']*r['width'])
            x,y=load_indices(r,range(c['frames']));x=x[None].to(device);y=y[None,2:-2].to(device)
            torch.cuda.reset_peak_memory_stats();opt.zero_grad(set_to_none=True);t=time.monotonic()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                with torch.no_grad():target=teacher(x)
                pred=model(x);loss,terms=loss_terms(pred,y,target,0,total)
            loss.backward()
            if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('NONFINITE_PREFLIGHT')
            opt.step();torch.cuda.synchronize()
            reports.append(dict(domain=d,sequence=r['name'],shape=list(x.shape),output_shape=list(pred.shape),
                                seconds=time.monotonic()-t,peak_GiB=torch.cuda.max_memory_allocated()/2**30))
            del x,y,pred,target,loss,terms
        (out/'preflight.json').write_text(json.dumps({'status':'PASS','config_sha256':sha256(a.config),'reports':reports},indent=2)+'\n')
        print(json.dumps(reports,indent=2));return
    gate=out/'preflight.json'
    if not gate.exists() or json.loads(gate.read_text()).get('config_sha256')!=sha256(a.config):
        raise ValueError('Run this variant preflight first')
    if world>1:model=DDP(model,device_ids=[local],broadcast_buffers=False)
    raw=model.module if world>1 else model
    accum=c['clips_per_update']//world
    end=total if not a.stop_after else min(total,start+a.stop_after)
    indices=range(start*4+rank,end*4,world)
    dataset=Clips(manifest,total*4,c['frames'],c['seed'])
    loader=iter(DataLoader(dataset,batch_size=1,sampler=indices,num_workers=c['workers'],pin_memory=True,
                           worker_init_fn=worker_init,**({'prefetch_factor':1} if c['workers'] else {})))
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    began=time.monotonic()
    def save(update,filename='latest.pth'):
        if rank==0:
            atomic_save({'model':raw.state_dict(),'optimizer':opt.state_dict(),'update':update,
                         'best':best,'variant':a.variant,'config':c,
                         'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()},out/filename)
    for step in range(start,end):
        progress=max(0,(step-c['warmup_updates'])/max(1,total-c['warmup_updates']))
        lr=c['min_lr']+(c['lr']-c['min_lr'])*.5*(1+math.cos(math.pi*progress))
        lr*=min(1,(step+1)/c['warmup_updates'])
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);meter={}
        for micro in range(accum):
            batch=next(loader);x=batch['blur'].to(device,non_blocking=True);gt=batch['gt'][:,2:-2].to(device,non_blocking=True)
            sync=model.no_sync() if world>1 and micro<accum-1 else nullcontext()
            with sync:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    with torch.no_grad():target=teacher(x)
                    pred=model(x);loss,terms=loss_terms(pred,gt,target,step,total)
                finite=torch.tensor(int(torch.isfinite(loss)),device=device)
                if world>1:dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                if not finite.item():raise RuntimeError('NONFINITE_LOSS; resume last valid checkpoint')
                (loss/accum).backward()
            for k,v in terms.items():meter[k]=meter.get(k,0.)+v.item()/accum
            del x,gt,pred,target,loss,terms,batch
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise RuntimeError('NONFINITE_GRAD; resume last valid checkpoint')
        opt.step();update=step+1
        if update%20==0 or update==start+1:
            if world>1:
                values=torch.tensor(list(meter.values()),device=device)
                dist.all_reduce(values);values/=world
                meter=dict(zip(meter,values.tolist()))
        if rank==0 and (update%20==0 or update==start+1):
            row=dict(update=update,total=total,lr=lr,grad_norm=norm.item(),
                     seconds_per_update=(time.monotonic()-began)/(update-start),**meter)
            print(json.dumps(row),flush=True)
            with open(out/'training.jsonl','a') as f:f.write(json.dumps(row)+'\n')
        if update%c['validate_every']==0 or update==total:
            if world>1:dist.barrier()
            if rank==0:
                result=evaluate(raw,manifest['val'],device,max_windows=c['validation_windows_per_sequence'])
                (out/f'val_{update:06d}.json').write_text(json.dumps(result,indent=2)+'\n')
                metrics=result['summary'];scores={'gopro':metrics['gopro']['psnr'],
                         'balanced':sum(metrics[d]['psnr'] for d in DOMAINS)/3}
                for key,value in scores.items():
                    if value>best[key]:best[key]=value;save(update,f'best_{key}.pth')
                print(json.dumps({'validation':update,'scores':metrics}),flush=True)
            if world>1:dist.barrier()
        stop=torch.tensor(int(STOP),device=device)
        if world>1:dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        if update%c['save_every']==0 or update==end or stop.item():save(update)
        if stop.item():
            save(update)
            if world>1:dist.barrier();dist.destroy_process_group()
            raise SystemExit(75)
    if world>1:dist.barrier();dist.destroy_process_group()
    if STOP:raise SystemExit(75)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--variant',choices=['quality','compact'],required=True)
    p.add_argument('--preflight',action='store_true');p.add_argument('--resume')
    p.add_argument('--stop-after',type=int,help='Optional bounded run; does not alter schedule')
    train(p.parse_args())

if __name__=='__main__':main()
