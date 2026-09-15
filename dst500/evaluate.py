"""FP32 RGB8 mean per-frame PSNR; every native frame, no crop/TTA/temporal discard."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import torch
from PIL import Image
from .data import load_indices, sha256
from .model import load_student, teacher


def rgb8(x):
    return (x.float().clamp(0,1)*255+.5).floor()/255


def psnr(x,y):
    mse=(x.double()-y.double()).square().mean().item()
    return float('inf') if mse==0 else -10*math.log10(mse)


@torch.inference_mode()
def evaluate(model,records,device,frames=6,max_chunks=0,previews=None):
    was_training=model.training;model.eval()
    rows=[]
    for r in records:
        starts=list(range(0,len(r['blur']),frames))
        if max_chunks and len(starts)>max_chunks:
            starts=[starts[i] for i in sorted({round(j*(len(starts)-1)/max(1,max_chunks-1)) for j in range(max_chunks)})]
        for start in starts:
            ids=list(range(start,min(start+frames,len(r['blur']))))
            x,y=load_indices(r,ids)
            raw=model(x[None].to(device))[0].float().cpu()
            if raw.shape!=y.shape or not torch.isfinite(raw).all():
                raise RuntimeError('Output shape/non-finite failure')
            pred=rgb8(raw)
            for j,idx in enumerate(ids):
                dark=x[j].mean()>.06 and pred[j].mean()<x[j].mean()*.5 and (pred[j]<=1/255).float().mean()>.25
                if dark: raise RuntimeError(f'SEVERE_DARK_OUTPUT {r["name"]}:{idx}')
                rows.append({'domain':r['domain'],'sequence':r['name'],'index':idx,
                    'name':Path(r['blur'][idx]).name,'psnr_rgb8':psnr(pred[j],y[j]),
                    'input_psnr_rgb8':psnr(x[j],y[j]),'psnr_float':psnr(raw[j].clamp(0,1),y[j])})
                if previews and idx==0:
                    folder=Path(previews)/r['domain'];folder.mkdir(parents=True,exist_ok=True)
                    panel=torch.cat((rgb8(x[j]),pred[j],y[j]),-1)
                    Image.fromarray((panel.permute(1,2,0).numpy()*255).round().astype('uint8')).save(folder/(r['name'].replace('/','_')+'.png'))
            del raw,pred,x,y
        print(f'evaluated {r["domain"]}/{r["name"]}',flush=True)
    groups=defaultdict(list)
    for row in rows: groups[row['domain']].append(row)
    summary={d:{'frames':len(rs),'sequences':len({v['sequence'] for v in rs}),
                  **{k:sum(v[k] for v in rs)/len(rs) for k in ('psnr_rgb8','input_psnr_rgb8','psnr_float')}} for d,rs in groups.items()}
    model.train(was_training)
    return {'summary':summary,'per_frame':rows,'frames_per_chunk':frames,
            'protocol':'native FP32 TF32 off; RGB8 floor(clip*255+0.5); border=0; mean per-frame; T in T out; reset each chunk',
            'partial':bool(max_chunks)}


def save_report(report,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n')
    rows=report['per_frame']
    if rows:
        with path.with_suffix('.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--checkpoint');p.add_argument('--teacher',action='store_true')
    p.add_argument('--output',required=True);p.add_argument('--split',default='test',choices=['test','val'])
    a=p.parse_args();c=json.loads(Path(a.config).read_text())
    from .run import verify_config
    manifest=verify_config(c)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda')
    if a.teacher: model=teacher(c['upstream'],c['teacher_checkpoint']).to(device)
    else: model,_=load_student(c['upstream'],a.checkpoint,device)
    r=evaluate(model,manifest[a.split],device,c['frames'],previews=str(Path(a.output).with_suffix(''))+'_previews')
    r.update(checkpoint_sha256=sha256(c['teacher_checkpoint'] if a.teacher else a.checkpoint),
             manifest_sha256=c['manifest_sha256'],split=a.split,teacher=a.teacher)
    if not a.teacher and a.split=='test':
        from .profile import profile
        budget=profile(c['upstream'])
        go=r['summary'].get('gopro',{})
        r['profile']=budget
        r['status']='TARGET_MET' if budget['budget_pass'] and go.get('frames')==1111 and go.get('psnr_rgb8',0)>=33 else 'TARGET_NOT_MET'
        # Aux counts are disclosed; never label an incomplete DVD subset full.
        r['dataset_audit']=manifest['audit']
    save_report(r,a.output);print(json.dumps({k:v for k,v in r.items() if k!='per_frame'},indent=2))

if __name__=='__main__': main()
