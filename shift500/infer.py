"""Native PNG/JPG sequence -> every restored frame and optional streaming MP4."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .data import images, read_rgb
from .evaluate import windows
from .model import ShiftModel


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--fps',type=float,default=30)
    p.add_argument('--mp4',action='store_true');p.add_argument('--cut-threshold',type=float,default=.30)
    a=p.parse_args();c=json.loads(Path(a.config).read_text());state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    m=ShiftModel(c['upstream'],state['variant']).cuda().eval();m.load_state_dict(state['model'],strict=True)
    files=images(Path(a.input));out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if not files:raise ValueError('No input frames')
    if any(out.iterdir()):raise ValueError('Use a new empty output directory')
    cuts=[0];previous=None;shape=None
    for i,path in enumerate(files):
        with Image.open(path) as im:
            if shape and im.size!=shape:raise ValueError('Changing frame resolution')
            shape=im.size;small=np.array(im.convert('RGB').resize((64,36)),dtype=np.float32)/255
        if previous is not None and np.abs(small-previous).mean()>a.cut_threshold:cuts.append(i)
        previous=small
    cuts.append(len(files));writer=None;calls=0
    if a.mp4:
        import cv2
        writer=cv2.VideoWriter(str(out/'output.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),a.fps,shape)
        if not writer.isOpened():raise RuntimeError('MP4 writer unavailable')
    try:
        with torch.inference_mode():
            for lo,hi in zip(cuts,cuts[1:]):
                for indices,valid in windows(hi-lo):
                    x=torch.stack([read_rgb(files[lo+i]) for i in indices])[None].cuda()
                    with torch.autocast('cuda',dtype=torch.float16):pred=m(x)[0,:len(valid)].float().clamp(0,1).cpu()
                    calls+=1
                    for j,idx in enumerate(valid):
                        rgb=(pred[j].permute(1,2,0).numpy()*255).round().astype('uint8')
                        Image.fromarray(rgb).save(out/f'{lo+idx:08d}.png')
                        if writer is not None:writer.write(rgb[...,::-1])
    finally:
        if writer is not None:writer.release()
    report={'frames':len(files),'network_calls':calls,'scene_starts':cuts[:-1],
            'input_frames_per_call':16,'generated_outputs_per_call':12,
            'useful_outputs':len(files),'tail_cost_multiplier':calls*12/len(files),
            'note':'No cross-scene context; last chunks are padded. Tail/discarded-output cost must be included in measured average.'}
    (out/'inference.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
