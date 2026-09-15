"""Full native-frame RGB8/float PSNR; every test frame exactly once."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from shift500.data import load_indices, sha256
from shift500.evaluate import windows, score_frame
from shift500.model import ShiftModel, load_teacher
from .model import ShiftWave, MODEL_ID


def rgb8(x):
    return (x.clamp(0,1)*255).round()/255


@torch.inference_mode()
def evaluate(model, records, device, previews=None):
    model.eval();rows=[]
    for r in records:
        per_frame=[];calls=0
        for indices, valid in windows(len(r['blur']),12,'all_frames'):
            x,gt=load_indices(r,indices)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                raw=model(x[None].to(device))[0,:len(valid)].float().cpu()
            if not torch.isfinite(raw).all():
                raise RuntimeError(f'Nonfinite raw output: {r["name"]}')
            pred=raw.clamp(0,1);gt=gt[2:2+len(valid)];inp=x[2:2+len(valid)]
            if pred.shape!=gt.shape:raise ValueError('Prediction/GT alignment mismatch')
            for j,i in enumerate(valid):
                per_frame.append(dict(index=i,filename=Path(r['blur'][i]).name,
                                      rgb8_psnr=score_frame(rgb8(pred[j]),gt[j]),
                                      float_psnr=score_frame(pred[j],gt[j]),
                                      input_psnr=score_frame(inp[j],gt[j])))
                if previews and i<3:
                    dst=Path(previews)/r['domain']/r['name'];dst.mkdir(parents=True,exist_ok=True)
                    panel=torch.cat((inp[j],pred[j],gt[j]),dim=-1)
                    Image.fromarray((panel.permute(1,2,0).numpy()*255).round().astype(np.uint8)).save(dst/f'{i:06d}.png')
            calls+=1
        if [f['index'] for f in per_frame]!=list(range(len(r['blur']))):
            raise ValueError('Missing/repeated scoring frame')
        rows.append(dict(domain=r['domain'],sequence=r['name'],frames=len(per_frame),windows=calls,per_frame=per_frame))
        print(json.dumps(dict(sequence=r['name'],frames=len(per_frame),rgb8_psnr=sum(f['rgb8_psnr'] for f in per_frame)/len(per_frame))),flush=True)
    summary={}
    for d in sorted({r['domain'] for r in rows}):
        rr=[r for r in rows if r['domain']==d];ff=[f for r in rr for f in r['per_frame']]
        summary[d]=dict(sequences=len(rr),frames=len(ff),**{k:sum(f[k] for f in ff)/len(ff) for k in ('rgb8_psnr','float_psnr','input_psnr')})
        if d=='dvd':summary[d]['complete_official_test']=len(rr)==10
    return dict(protocol='all_frames_16in_12out_replicate_boundaries',
                metric='native full-frame RGB8 PSNR + RGB float PSNR; frame mean; no border crop; no TTA; input RGB /255',
                precision='FP16 autocast' if device.type=='cuda' else 'FP32',
                summary=summary,sequences=rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--model',choices=['student','teacher'],default='student')
    p.add_argument('--checkpoint');p.add_argument('--output',required=True)
    p.add_argument('--domains',nargs='+',choices=['gopro','dvd','bsd'],default=['gopro','dvd','bsd'])
    a=p.parse_args();c=json.loads(Path(a.config).read_text())
    if sha256(c['manifest'])!=c['manifest_sha256']:raise ValueError('Manifest changed')
    device=torch.device('cuda');path=a.checkpoint
    if a.model=='teacher':
        path=c['teacher_checkpoint']
        if sha256(path)!=c['teacher_sha256']:raise ValueError('Teacher changed')
        m=ShiftModel(c['upstream'],'teacher');load_teacher(m,path)
    else:
        if not path:raise ValueError('Student checkpoint required')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state.get('model_id')!=MODEL_ID or state['config']!=c or state['update']!=c['total_updates']:
            raise ValueError('Final evaluation requires this run\'s completed, fixed-iteration checkpoint')
        m=ShiftWave(c['upstream']);m.load_state_dict(state['model'],strict=True)
    records=[r for r in json.loads(Path(c['manifest']).read_text())['test'] if r['domain'] in a.domains]
    if any(not any(r['domain']==d for r in records) for d in a.domains):raise ValueError('Missing requested test domain')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    report=evaluate(m.to(device),records,device,str(out.with_suffix(''))+'_previews')
    report.update(model=a.model,checkpoint_sha256=sha256(path),manifest_sha256=c['manifest_sha256'],model_id=MODEL_ID)
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['summary'],indent=2))

if __name__=='__main__':main()
