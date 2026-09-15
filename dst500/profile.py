"""Count all dispatched arithmetic, including functional Haar/dynamic filtering.

1 multiplication + 1 addition = 2 FLOPs. Comparisons/nonlinear primitives use
documented nominal counts; memory traffic is not represented by FLOPs.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from .model import Student, teacher


class CounterMode(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.flops = Counter()
        self.memory = Counter()
        self.unknown = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = str(func).split('.')[1].rstrip('_')
        outs = out if isinstance(out,(tuple,list)) else [out]
        n = sum(v.numel() for v in outs if isinstance(v,torch.Tensor))
        if name == 'convolution':
            x,weight,bias,stride,padding,dilation,transposed,output_padding,groups = args
            kernel = math.prod(weight.shape[2:])
            mac = (x.numel()*weight.shape[1]*kernel if transposed
                   else out.numel()*weight.shape[1]*kernel)
            self.flops['convolution'] += 2*mac
            if bias is not None: self.flops['bias'] += out.numel()
        elif name in ('add','sub','rsub','mul','div','pow','sqrt','rsqrt','neg'):
            self.flops[name] += n
        elif name in ('leaky_relu','relu','prelu','_prelu_kernel'):
            self.flops[name] += 2*n
        elif name == 'sigmoid':
            self.flops[name] += 4*n  # neg, exp, add, reciprocal
        elif name in ('sum','mean'):
            self.flops[name] += args[0].numel()
        elif name == 'upsample_bilinear2d':
            self.flops[name] += 7*n
        elif name == 'adaptive_max_pool2d':
            x=args[0]; oh,ow=out[0].shape[-2:]; h,w=x.shape[-2:]
            hs=sum(math.ceil((i+1)*h/oh)-math.floor(i*h/oh) for i in range(oh))
            ws=sum(math.ceil((j+1)*w/ow)-math.floor(j*w/ow) for j in range(ow))
            self.flops['pool_compare'] += math.prod(x.shape[:-2])*(hs*ws-oh*ow)
        elif name in {'view','reshape','_unsafe_view','slice','select','transpose','permute',
                      'cat','stack','split','split_with_sizes','unsqueeze','squeeze','expand',
                      'repeat','clone','detach','alias','copy','empty','empty_like','empty_strided',
                      'zeros','zeros_like','new_zeros','new_empty','ones','ones_like','lift_fresh',
                      'reflection_pad2d','replication_pad2d','constant_pad_nd','im2col','_to_copy'}:
            self.memory[str(func)] += 1
        else:
            self.unknown[str(func)] += 1
        return out


def profile(upstream, variant='student', frames=6, height=1080, width=1920):
    with torch.device('meta'):
        model = Student(upstream,False) if variant=='student' else teacher(upstream)
        # Meta cannot copy RepConv weights. Construct fused form directly; same
        # shapes and arithmetic as numerically verified deploy().
        if variant=='student':
            from .waveshift import RepConv2d
            for shift in model.shifts:
                shift.fusion = RepConv2d(128,64,groups=16,deploy=True)
        model.eval()
        x = torch.zeros(1,frames,3,height,width)
    counter=CounterMode()
    with torch.no_grad(),counter:
        y=model(x)
    if counter.unknown:
        raise RuntimeError(f'Unclassified operations: {counter.unknown}')
    outputs=y.shape[1]
    value=sum(counter.flops.values())/outputs/1e9
    return {'variant':variant,'height':height,'width':width,'input_frames':frames,'output_frames':outputs,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'parameters':sum(p.numel() for p in model.parameters()),
            'gflops_per_output':value,'clip_gflops':sum(counter.flops.values())/1e9,
            'convention':'multiply=1, add=1; 1 MAC=2 FLOPs; nonlinear nominal counts documented',
            'schedule':'T in T out; non-overlapping chunks; final short chunk has no padded time frames',
            'includes':'RGB stem/head, DWT/IWT, both recurrent directions, dynamic filtering, shifts/fusion, HF edge',
            'ops_per_clip':dict(counter.flops),'memory_operations':dict(counter.memory),
            'budget_pass':value<=500 if variant=='student' else None,
            'measured_runtime':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream',required=True)
    p.add_argument('--output',required=True);p.add_argument('--variant',default='student',choices=['student','teacher'])
    p.add_argument('--frames',default=6,type=int);a=p.parse_args()
    r=profile(a.upstream,a.variant,a.frames)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if not isinstance(v,dict)},indent=2))
    if r['budget_pass'] is False: raise SystemExit('FLOPS_BUDGET_FAIL')

if __name__=='__main__': main()
