"""Native meta-tensor accounting; convolution MACs and all scalar arithmetic."""
import argparse
import json
from pathlib import Path
from collections import Counter
import torch
from torch import nn
from shift500.profile import ArithmeticCounter, profile as teacher_profile
from .model import ShiftWave


def profile(upstream, frames=16, height=1080, width=1920):
    with torch.device('meta'):
        model = ShiftWave(upstream).eval()
        x = torch.zeros(1, frames, 3, height, width)
    macs, biases = Counter(), Counter()
    handles = []
    def hook(name):
        def count(m, inputs, output):
            macs[name] += output.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
            if m.bias is not None:
                biases[name] += output.numel()
        return count
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(hook(name)))
    # neg is real Haar arithmetic, not a zero-cost tensor view.
    class CounterWithNeg(ArithmeticCounter):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = super().__torch_dispatch__(func, types, args, kwargs)
            if str(func).split('.')[1] in ('neg', 'rsub'):
                self.counts[str(func).split('.')[1]] += out.numel()
                self.unclassified.pop(str(func), None)
            return out
    counter = CounterWithNeg()
    with torch.no_grad(), counter:
        y = model(x)
    for handle in handles:
        handle.remove()
    memory_only = {'select','replication_pad2d','unsqueeze','view','roll','slice',
                   'zeros_like','copy_','cat','split','pixel_shuffle','alias',
                   'stack','clone','empty_like','empty_strided','_unsafe_view'}
    unsupported = [op for op in counter.unclassified if op.split('.')[1] not in memory_only]
    if unsupported:
        raise RuntimeError(f'Unclassified arithmetic: {unsupported}')
    n = y.shape[1]
    extra = sum(counter.counts.values()) + sum(biases.values())
    return dict(model='shiftwave', parameters_M=sum(p.numel() for p in model.parameters())/1e6,
                input_frames=frames, output_frames=n, height=height, width=width,
                GMAC_per_output=sum(macs.values())/n/1e9,
                conv_GFLOPs_per_output=2*sum(macs.values())/n/1e9,
                arithmetic_GFLOPs_per_output=(2*sum(macs.values())+extra)/n/1e9,
                clip_GFLOPs=(2*sum(macs.values())+extra)/1e9,
                convention='1 MAC=2 FLOPs; scalar activation=1 nominal op; T-4 outputs; all branches; no resize/tiles/TTA; pad to 8',
                scalar_ops=dict(counter.counts), memory_dispatch=dict(counter.unclassified),
                conv_MACs_by_module=dict(macs))


def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    reports={'student':profile(a.upstream), 'teacher':teacher_profile(a.upstream,'teacher')}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(reports,indent=2)+'\n')
    for k,v in reports.items():
        print(k,json.dumps({n:x for n,x in v.items() if not isinstance(x,dict)}))
    if reports['student']['arithmetic_GFLOPs_per_output'] >= 500:
        raise SystemExit('BUDGET_FAIL')

if __name__=='__main__':main()
