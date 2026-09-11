"""Frame-counted native-frame evaluation; fixed chunks, no tiles or ensemble."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from .data import DOMAINS, load_indices, sha256
from .model import ShiftModel, load_teacher


def windows(length, outputs=12, protocol='all_frames'):
    if protocol == 'official_chunks':
        # Exact temporal indexing of upstream one_len protocol; explicitly excludes
        # first/last two frames and incomplete final chunks.
        for start in range(2, length-outputs-1, outputs):
            yield list(range(start-2,start+outputs+2)), list(range(start,start+outputs))
    else:
        for start in range(0,length,outputs):
            valid=list(range(start,min(start+outputs,length)))
            indices=[min(max(i,0),length-1) for i in range(start-2,start+outputs+2)]
            yield indices, valid


def score_frame(pred, gt):
    mse=(pred.double()-gt.double()).square().mean().item()
    return float('inf') if mse==0 else -10*math.log10(mse)


@torch.inference_mode()
def evaluate(model, records, device, outputs=12, protocol='all_frames', max_windows=0, save=None):
    was_training=model.training
    model.eval()
    rows=[]
    for r in records:
        total=0; psnrs=[]; input_psnrs=[]; temporal_num=0.; temporal_n=0
        previous_error=None; previous_gt=None; previous_i=None; per_frame=[]
        ws=list(windows(len(r['blur']),outputs,protocol))
        if max_windows and len(ws)>max_windows:
            ids=np.linspace(0,len(ws)-1,max_windows,dtype=int)
            ws=[ws[i] for i in ids]
        for indices, valid in ws:
            x,y=load_indices(r,indices)
            x=x.unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
                pred=model(x)[0,:len(valid)].float().clamp(0,1).cpu()
            gt=y[2:2+len(valid)]
            if not torch.isfinite(pred).all():
                raise RuntimeError(f'Non-finite prediction: {r["name"]}')
            for j,idx in enumerate(valid):
                psnr=score_frame(pred[j],gt[j]); psnrs.append(psnr)
                base=score_frame(x[0,j+2].float().cpu(),gt[j]); input_psnrs.append(base)
                error=pred[j]-gt[j]
                if previous_error is not None and idx==previous_i+1:
                    if (gt[j]-previous_gt).abs().mean()<=.30:
                        temporal_num+=(error-previous_error).abs().mean().item(); temporal_n+=1
                previous_error,previous_gt,previous_i=error,gt[j],idx
                per_frame.append({'index':idx,'filename':Path(r['blur'][idx]).name,'psnr':psnr,'input_psnr':base})
                if save and total<3:
                    folder=Path(save)/r['domain']/r['name'];folder.mkdir(parents=True,exist_ok=True)
                    panel=torch.cat((x[0,j+2].float().cpu(),pred[j],gt[j]),dim=-1)
                    Image.fromarray((panel.permute(1,2,0).clamp(0,1).numpy()*255).round().astype('uint8')).save(folder/f'{idx:08d}.png')
                total+=1
        if not psnrs:
            raise ValueError(f'No scored frames under {protocol}: {r["name"]}')
        rows.append({'domain':r['domain'],'sequence':r['name'],'frames':len(psnrs),'windows':len(ws),
                     'psnr':sum(psnrs)/len(psnrs),'input_psnr':sum(input_psnrs)/len(input_psnrs),
                     'temporal_sum':temporal_num,'temporal_pairs':temporal_n,'per_frame':per_frame})
    summary={}
    for d in DOMAINS:
        subset=[r for r in rows if r['domain']==d]
        n=sum(r['frames'] for r in subset)
        if not n: continue
        summary[d]={'psnr':sum(r['psnr']*r['frames'] for r in subset)/n,
                    'input_psnr':sum(r['input_psnr']*r['frames'] for r in subset)/n,
                    'frames':n,'sequences':len(subset),
                    'gt_relative_unwarped_temporal_l1':sum(r['temporal_sum'] for r in subset)/max(1,sum(r['temporal_pairs'] for r in subset))}
    model.train(was_training)
    return {'protocol':protocol,'chunk_outputs':outputs,'input_frames_per_call':outputs+4,
            'spatial':'native, pad to multiple of 4 then remove padding',
            'metric':'RGB [0,1] clamp, no rounding, no border crop, mean per-frame PSNR',
            'summary':summary,'sequences':rows}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--variant',choices=['teacher','quality','compact'],required=True)
    p.add_argument('--checkpoint');p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--protocol',choices=['all_frames','official_chunks'],default='all_frames')
    p.add_argument('--max-windows',type=int,default=0);p.add_argument('--outputs',type=int,default=12);p.add_argument('--output',required=True)
    a=p.parse_args();c=json.loads(Path(a.config).read_text());device=torch.device('cuda')
    assert sha256(c['manifest'])==c['manifest_sha256'], 'Manifest changed'
    assert sha256(c['teacher_checkpoint'])==c['teacher_sha256'], 'Teacher changed'
    m=ShiftModel(c['upstream'],a.variant).to(device)
    if a.variant=='teacher': load_teacher(m,c['teacher_checkpoint'])
    else:
        state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        if state['variant']!=a.variant: raise ValueError('Checkpoint variant mismatch')
        m.load_state_dict(state['model'],strict=True)
    report=evaluate(m,json.loads(Path(c['manifest']).read_text())[a.split],device,a.outputs,a.protocol,max_windows=a.max_windows,save=str(Path(a.output).with_suffix(''))+'_previews')
    report.update(split=a.split,variant=a.variant,checkpoint_sha256=sha256(a.checkpoint or c['teacher_checkpoint']),manifest_sha256=c['manifest_sha256'])
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))

if __name__=='__main__': main()
