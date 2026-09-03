import argparse
import torch
from torch.utils.data import DataLoader
from data.gopro_video import GoProVideoDataset
from models.nanovsr_unet_fullres_concat_deblur import NanoVSRFullResConcatUNetDeblur

ARCH='NanoVSRFullResConcatUNetDeblur'


def load_model(path, device):
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


def psnr(a,b):
    mse=(a-b).pow(2).mean(dim=(-3,-2,-1)).clamp_min(1e-12)
    return -10*torch.log10(mse)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gopro-root',required=True); ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--num-frames',type=int,default=15); ap.add_argument('--max-clips',type=int,default=100); ap.add_argument('--center-only',action='store_true'); ap.add_argument('--fp16',action='store_true')
    a=ap.parse_args(); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model,ck=load_model(a.checkpoint,dev)
    ds=GoProVideoDataset(a.gopro_root,'test',a.num_frames,patch_size=None); dl=DataLoader(ds,batch_size=1,shuffle=False,num_workers=1)
    vals=[]
    with torch.no_grad():
        for i,b in enumerate(dl):
            if a.max_clips and i>=a.max_clips: break
            x=b['blur'].to(dev); y=b['sharp'].to(dev)
            with torch.cuda.amp.autocast(enabled=a.fp16): pred=model(x)
            p=psnr(pred.float().clamp(0,1),y.float())[0]
            if a.center_only: vals.append(p[a.num_frames//2].item())
            else: vals.extend(p.cpu().tolist())
            if (i+1)%20==0: print(f'clips={i+1} PSNR_RGB={sum(vals)/len(vals):.4f}',flush=True)
    print(f'ARCHITECTURE={ARCH}')
    print(f'CHECKPOINT_STEP={ck.get("step")}')
    print(f'EVAL_MODE={"CENTER_ONLY" if a.center_only else "ALL_FRAMES"}')
    print(f'NUM_FRAMES={a.num_frames}')
    print(f'PSNR_RGB={sum(vals)/max(1,len(vals)):.4f} dB')


if __name__=='__main__': main()
