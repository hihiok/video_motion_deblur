import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from models.nanovsr_unet_fullres_concat_deblur import NanoVSRFullResConcatUNetDeblur

ARCH='NanoVSRFullResConcatUNetDeblur'


def load_model(path,device):
    ck=torch.load(path,map_location='cpu')
    if ck.get('architecture')!=ARCH: raise RuntimeError(f'Unexpected architecture: {ck.get("architecture")}')
    c=ck['model_config']
    m=NanoVSRFullResConcatUNetDeblur(
        c['base_channels'],c['mid_channels'],c['bottleneck_channels'],
        c['encoder_blocks'],c['state_fusion_blocks'],
        c['fullres_blocks'],c['mid_blocks'],c['bottleneck_blocks'],
        c['decoder_channels'],c['decoder_blocks'],False).to(device).eval()
    m.load_state_dict(ck['model'],strict=True)
    return m,ck


def read_video(path):
    cap=cv2.VideoCapture(path)
    if not cap.isOpened(): raise RuntimeError(f'Cannot open {path}')
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0; frames=[]
    while True:
        ok,bgr=cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames: raise RuntimeError('No frames decoded')
    return frames,fps


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--chunk',type=int,default=15); ap.add_argument('--overlap',type=int,default=4); ap.add_argument('--fp16',action='store_true')
    a=ap.parse_args(); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model,ck=load_model(a.checkpoint,dev)
    frames,fps=read_video(a.input); n=len(frames); h,w=frames[0].shape[:2]
    sums=np.zeros((n,h,w,3),np.float32); counts=np.zeros(n,np.float32)
    step=max(1,a.chunk-a.overlap); starts=list(range(0,n,step))
    if starts and starts[-1]+a.chunk<n: starts.append(max(0,n-a.chunk))
    if dev.type=='cuda': torch.cuda.reset_peak_memory_stats(dev)
    with torch.no_grad():
        for s in starts:
            e=min(n,s+a.chunk)
            arr=np.stack(frames[s:e]).astype(np.float32)/255.0
            x=torch.from_numpy(arr).permute(0,3,1,2).unsqueeze(0).to(dev)
            with torch.cuda.amp.autocast(enabled=a.fp16): y=model(x)[0]
            y=y.float().clamp(0,1).permute(0,2,3,1).cpu().numpy()
            for j in range(e-s): sums[s+j]+=y[j]; counts[s+j]+=1
            print(f'chunk={s}:{e}',flush=True)
            if e==n: break
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    vw=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    for i in range(n):
        rgb=np.clip(sums[i]/max(counts[i],1.0)*255.0+0.5,0,255).astype(np.uint8)
        vw.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    vw.release()
    peak=torch.cuda.max_memory_allocated(dev)/(1024**3) if dev.type=='cuda' else 0
    print(f'ARCHITECTURE={ARCH}'); print(f'CHECKPOINT_STEP={ck.get("step")}'); print(f'OUTPUT={out}'); print(f'PEAK_GPU_GIB={peak:.3f}')


if __name__=='__main__': main()
