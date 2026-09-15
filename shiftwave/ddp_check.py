"""Two actual ranks, two optimizer steps, accumulated gradients, identical weights."""
import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .model import ShiftWave


def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--output',required=True)
    a=p.parse_args();rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=2:raise ValueError('Exactly two ranks required')
    torch.set_num_threads(2);torch.manual_seed(10)
    device=torch.device('cuda',local) if a.device=='cuda' else torch.device('cpu')
    if device.type=='cuda':torch.cuda.set_device(device)
    dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    try:
        net=ShiftWave(a.upstream,activation_checkpointing=True).to(device)
        model=DDP(net,device_ids=[local] if device.type=='cuda' else None,broadcast_buffers=False)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0)
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            for micro in range(2):
                from contextlib import nullcontext
                torch.manual_seed(100+rank*10+step*2+micro)
                x=torch.rand(1,7,3,64,64,device=device)
                with model.no_sync() if micro==0 else nullcontext():
                    loss=(model(x)-x[:,1:-1]).abs().mean()/2
                    loss.backward()
            for name,param in net.named_parameters():
                if param.grad is None or not torch.isfinite(param.grad).all():
                    raise RuntimeError(f'Unused/nonfinite gradient: {name}')
            torch.nn.utils.clip_grad_norm_(net.parameters(),.01);opt.step()
        flat=torch.cat([p.detach().flatten() for p in net.parameters()])
        other=flat.clone();dist.broadcast(other,src=0)
        error=(flat-other).abs().max();dist.all_reduce(error,op=dist.ReduceOp.MAX)
        if error.item()!=0:raise RuntimeError('Rank weights diverged')
        if rank==0:
            out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
            out.write_text(json.dumps(dict(status='PASS',world_size=2,backend=dist.get_backend(),updates=2,
                                           max_rank_parameter_difference=error.item(),precision='FP32; AMP separately checked in training preflight'),indent=2)+'\n')
            print(out.read_text())
    finally:dist.destroy_process_group()

if __name__=='__main__':main()
