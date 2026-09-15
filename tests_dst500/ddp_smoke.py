"""Two CPU ranks exercise the actual accumulation/optimizer path, then resume.

This validates DDP semantics, not A100 memory or CUDA numerical equivalence.
"""
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dst500.model import Student, teacher, initialize
from dst500.run import step, save_state


def main():
    torch.set_num_threads(1);torch.manual_seed(42)
    dist.init_process_group('gloo');rank=dist.get_rank()
    root=Path(os.environ['DST500_TEST_OUTPUT']);root.mkdir(parents=True,exist_ok=True)
    t=teacher(os.environ['DST500_UPSTREAM'])
    if rank==0:torch.save({'params':t.state_dict()},root/'teacher.pth')
    dist.barrier()
    model=Student(os.environ['DST500_UPSTREAM']);initialize(model,root/'teacher.pth')
    ddp=DDP(model,broadcast_buffers=False)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    batches=[{'blur':torch.rand(2,3,32,40),'gt':torch.rand(2,3,32,40)} for _ in range(4)]
    value,norm=step(ddp,t,opt,batches,torch.device('cpu'),1e-4,.2)
    before={k:v.clone() for k,v in model.state_dict().items()}
    if rank==0:save_state(model,opt,1,0.,{},root/'latest.pth')
    dist.barrier();raw=torch.load(root/'latest.pth',weights_only=False)
    model.load_state_dict(raw['model']);opt.load_state_dict(raw['optimizer'])
    for key,val in model.state_dict().items():torch.testing.assert_close(val,before[key],atol=0,rtol=0)
    value2,norm2=step(ddp,t,opt,batches,torch.device('cpu'),1e-4,.2)
    checksum=sum(p.detach().double().sum() for p in model.parameters())
    other=checksum.clone();dist.broadcast(other,0);torch.testing.assert_close(checksum,other,atol=0,rtol=0)
    if rank==0:print(f'DDP accumulation/resume PASS; loss={value:.6f}->{value2:.6f}; gradients={norm:.4f},{norm2:.4f}')
    dist.destroy_process_group()

if __name__=='__main__':main()
