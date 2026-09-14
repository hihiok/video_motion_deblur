#!/usr/bin/env python3
"""One frozen benchmark method, one whole sequence; receives NO ground truth."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / 'adapters'))
from common import list_frames, load_rgb_float, reflection_indices
from rtf_t6.checkpoint import unwrap_state_dict
from rtf_temporal.posttrain import write_json
from rtf_temporal.data import sha256


def module_file(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def pad4(x, multiple=8, minimum=0):
    h, w = x.shape[-2:]
    ph, pw = max(minimum, (h+multiple-1)//multiple*multiple)-h, max(minimum, (w+multiple-1)//multiple*multiple)-w
    mode = 'reflect' if ph < h and pw < w and min(h,w) > 1 else 'replicate'
    return F.pad(x, (0,pw,0,ph), mode=mode)


def select_state(payload, requested='auto_unique'):
    if requested not in ('auto_unique', 'bare'):
        if not isinstance(payload.get(requested), dict):
            raise ValueError(f'Checkpoint state_key not found: {requested}')
        return unwrap_state_dict(payload[requested]), requested
    containers=[k for k in ('params_ema','params','state_dict','model','model_state_dict','net_g')
                if isinstance(payload.get(k),dict) and payload[k]]
    bare=any(torch.is_tensor(v) for v in payload.values())
    if requested=='bare':
        if not bare or containers: raise ValueError('Checkpoint is not an unambiguous bare state dict')
        return unwrap_state_dict(payload),'bare'
    if len(containers)+int(bare)!=1:
        raise ValueError('Ambiguous checkpoint containers: explicitly set state_key before freezing')
    key=containers[0] if containers else 'bare'
    return unwrap_state_dict(payload if key=='bare' else payload[key]),key


def run(job, frames, output, device):
    kind, options = job['kind'], job.get('inference', {})
    checkpoint = job['checkpoint_snapshot']
    if sha256(checkpoint) != job['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    torch.manual_seed(20260914)
    if job.get('repo'):
        sys.path.insert(0, job['repo'])
    repo = Path(job.get('repo', '.'))
    state_dict, state_key = select_state(torch.load(checkpoint, map_location='cpu', weights_only=False), job.get('state_key','auto_unique'))
    info = {'kind': kind, 'precision': 'fp32', 'tf32': False, 'strict_load': True, 'inference': options,
            'checkpoint_sha256': job['checkpoint_sha256'], 'state_key': state_key, 'torch': torch.__version__}

    def load(ids):
        return torch.from_numpy(np.stack([load_rgb_float(frames[i]) for i in ids])).permute(0,3,1,2).contiguous()

    saved = set()
    def save(index, pred):
        if index in saved:
            raise ValueError('Duplicate output frame')
        if not bool(torch.isfinite(pred).all()):
            raise FloatingPointError(f'Nonfinite raw output at frame {index}; never nan_to_num')
        with Image.open(frames[index]) as im:
            w, h = im.size
        if tuple(pred.shape) != (3,h,w):
            raise ValueError(f'Wrong raw output shape: {pred.shape}, expected {(3,h,w)}')
        array = (pred.detach().float().cpu().permute(1,2,0).numpy().clip(0,1)*255).round().astype(np.uint8)
        Image.fromarray(array).save(output / f'{index:08d}.png')
        saved.add(index)
        if len(saved)%25 == 0:
            print(f'PREDICTED {len(saved)}/{len(frames)}', flush=True)

    n = len(frames)
    if kind in ('rtf_official', 'rtf_temporal'):
        from rtf_temporal.model import TemporalRTFocuser
        from rtf_temporal.upstream_reference import RT_Focuser_Standard
        from rtf_temporal.flow import is_cut
        if kind == 'rtf_temporal':
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model = TemporalRTFocuser(**{**payload['config']['model'], 'activation_checkpointing': False})
            model.load_state_dict(payload['model'], strict=True)
            threshold = payload['config']['train']['cut_threshold']
            info['best_update'] = payload['update']
        else:
            model = RT_Focuser_Standard(); model.load_state_dict(state_dict, strict=True)
        model.to(device).eval()
        state, previous, resets = None, None, []
        for i in range(n):
            x = load([i]).to(device); h,w=x.shape[-2:]
            if kind == 'rtf_temporal':
                current = x[0].permute(1,2,0).cpu().numpy()
                if previous is None or previous.shape != current.shape or is_cut(previous,current,threshold):
                    state=None; resets.append(i)
                pred, state = model.step(x,state); previous=current
            else:
                pred = model(F.pad(x,(0,(-w)%16,0,(-h)%16),mode='replicate'))[..., :h,:w]
            save(i,pred[0])
        info['reset_indices'] = resets
    elif kind == 'shiftnet':
        from shiftnet_infer import load_model_class, pad_video_to_multiple
        cls,_ = load_model_class(repo)
        model=cls(future_frames=2,past_frames=2).to(device).eval()
        model.load_state_dict(state_dict,strict=True)
        length=int(options.get('clip_length',48))
        if length < 1:
            raise ValueError('Invalid clip length')
        # Match official core chunk phase (first core starts at index 2), then
        # explicitly restore skipped boundary/tail frames for the common ALL-frame protocol.
        intervals=[(a,min(a+length,n-2)) for a in range(2,max(2,n-2),length)]
        intervals += [(0,min(2,n))]
        if n > 2:
            intervals.append((max(2,n-2),n))
        for a,b in intervals:
            if b<=a: continue
            ids=reflection_indices(a-2,b+2,n)
            x=pad_video_to_multiple(load(ids)[None].to(device),4)
            pred=model(x.contiguous())
            if pred.ndim==5: pred=pred[0]
            if len(pred)==len(ids): pred=pred[2:-2]
            if len(pred)!=b-a: raise ValueError('Shift-Net output count mismatch')
            h,w=load([a]).shape[-2:]
            for j,i in enumerate(range(a,b)): save(i,pred[j,:,:h,:w])
        info['boundary_policy']='official core chunk phase; additional reflected boundary/tail outputs'
    elif kind == 'dstnet':
        from dstnet_compat import load_dstnet_deblur
        from dstnet_infer import infer_chunk_tiled
        cls,backend=load_dstnet_deblur(repo)
        model=cls(num_feat=64,num_block=15).to(device).eval(); model.load_state_dict(state_dict,strict=True)
        info['dynamic_backend']=backend
        length=int(options.get('clip_length',30)); tile=int(options.get('tile',0))
        for a in range(0,n,length):
            b=min(n,a+length); x=load(range(a,b))[None].to(device)
            array=infer_chunk_tiled(model,x,tile,int(options.get('tile_overlap',64)),False,torch.device(device))
            for j,i in enumerate(range(a,b)): save(i,torch.from_numpy(array[j]).permute(2,0,1))
    elif kind == 'bsstnet':
        from bsstnet_infer import get_bi_flows, pad_spatial, starts_for_size
        from basicsr.archs.BSST_arch import BSST
        from basicsr.archs.RAFT.raft import RAFT
        # RAFT resolves a relative checkpoint path; use a private runtime directory,
        # never overwrite a model repository's files or existing symlinks.
        work=output.parent/'raft_runtime'; (work/'model_zoos').mkdir(parents=True,exist_ok=True)
        target=work/'model_zoos/raft-things.pth'
        raft_path=Path(job['auxiliary_snapshots']['raft']).resolve()
        if not target.exists(): target.symlink_to(raft_path)
        os.chdir(work)
        model=BSST().to(device).eval(); model.load_state_dict(state_dict,strict=True)
        raft=RAFT().to(device).eval(); raft.args.mixed_precision=False
        length=int(options.get('clip_length',48)); overlap=int(options.get('tile_overlap',64))
        for a in range(0,n,length):
            b=min(n,a+length); x=load(range(a,b))[None].to(device); h,w=x.shape[-2:]
            single=b-a==1
            if single: x=torch.cat([x,x],1)
            x,_,_=pad_spatial(x,256,8); fw,bw=get_bi_flows(raft,x)
            accum=torch.zeros_like(x,device='cpu'); weights=torch.zeros((1,1,1,*x.shape[-2:]))
            for y in starts_for_size(x.shape[-2],256,overlap):
                for z in starts_for_size(x.shape[-1],256,overlap):
                    pred=model(x[...,y:y+256,z:z+256],fw[...,y//4:y//4+64,z//4:z//4+64],bw[...,y//4:y//4+64,z//4:z//4+64])
                    if not bool(torch.isfinite(pred).all()): raise FloatingPointError('Nonfinite BSST patch')
                    accum[...,y:y+256,z:z+256]+=pred.float().cpu(); weights[...,y:y+256,z:z+256]+=1
            if (weights==0).any(): raise ValueError('Uncovered BSST pixels')
            pred=accum/weights
            for j,i in enumerate(range(a,b)): save(i,pred[0,j,:,:h,:w])
        info['spatial_policy']='256 patches, fixed overlap, uniform aggregation; no adaptive OOM changes'
    elif kind == 'rvrt':
        official=module_file(repo/'main_test_rvrt.py','official_rvrt_inference')
        model=official.net(upscale=1,clip_size=2,img_size=[2,64,64],window_size=[2,8,8],num_blocks=[1,2,1],
            depths=[2,2,2],embed_dims=[192,192,192],num_heads=[6,6,6],inputconv_groups=[1,3,3,3,3,3],
            deformable_groups=12,attention_heads=12,attention_window=[3,3],cpu_cache_length=100)
        model.load_state_dict(state_dict,strict=True); model.to(device).eval()
        opt=SimpleNamespace(tile=options.get('tile',[30,256,256]),tile_overlap=options.get('tile_overlap',[2,20,20]),
                            scale=1,window_size=[2,8,8],nonblind_denoising=False)
        pred=official.test_video(load(range(n))[None].to(device),model,opt)
        if pred.shape[1]!=n: raise ValueError('RVRT output count mismatch')
        for i in range(n): save(i,pred[0,i])
    elif kind == 'turtle':
        config=yaml.safe_load(Path(job['auxiliary_snapshots']['config']).read_text())
        architecture=options.get('architecture','turtle_t1_arch.py')
        if architecture not in ('turtle_arch.py','turtle_t1_arch.py'): raise ValueError('Unsupported Turtle architecture')
        module=module_file(repo/'basicsr/models/archs'/architecture,'official_turtle_arch')
        model=module.make_model(config).to(device).eval(); model.load_state_dict(state_dict,strict=True)
        info['num_frames_tocache']=config.get('num_frames_tocache',1)
        # Cache follows checkpoint-matched config; no guess based on paper gamma or another variant.
        tile=int(options.get('tile',0)); overlap=int(options.get('tile_overlap',128))
        caches={}; prev=None
        from bsstnet_infer import starts_for_size
        for i in range(n):
            x=load([i]).to(device); h,w=x.shape[-2:]; x=pad4(x,8)
            if prev is None: prev=x
            patches=[(0,0)] if tile==0 else [(y,z) for y in starts_for_size(x.shape[-2],tile,overlap) for z in starts_for_size(x.shape[-1],tile,overlap)]
            accum=torch.zeros_like(x); weights=torch.zeros_like(x)
            for y,z in patches:
                hh=x.shape[-2] if tile==0 else min(tile,x.shape[-2]); ww=x.shape[-1] if tile==0 else min(tile,x.shape[-1])
                k,v=caches.get((y,z),(None,None))
                pair=torch.stack([prev[...,y:y+hh,z:z+ww],x[...,y:y+hh,z:z+ww]],1)
                pred,k,v=model(pair,k,v); caches[(y,z)]=(k,v)
                if not bool(torch.isfinite(pred).all()): raise FloatingPointError('Nonfinite Turtle output')
                accum[...,y:y+hh,z:z+ww]+=pred; weights[...,y:y+hh,z:z+ww]+=1
            save(i,(accum/weights)[0,:,:h,:w]); prev=x
        info['state_policy']='continuous per-sequence per-spatial-tile K/V cache; reset only at new sequence'
    else:
        raise ValueError(f'Unsupported method kind: {kind}')
    if saved != set(range(n)): raise ValueError('Missing prediction indices')
    info['frames']=len(saved)
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',required=True);p.add_argument('--input',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    job=json.loads(Path(a.job).read_text());frames=list_frames(a.input)
    start=time.monotonic()
    with torch.inference_mode(): info=run(job,frames,out,a.device)
    info['seconds']=time.monotonic()-start
    write_json(out.parent/'inference.json',info)


if __name__=='__main__': main()
