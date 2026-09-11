"""Shape-only conv MAC accounting plus actual dispatcher arithmetic accounting."""
import argparse
import json
from collections import Counter
from pathlib import Path
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from .model import ShiftModel, VARIANTS


class ArithmeticCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts = Counter()
        self.unclassified = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = str(func).split('.')[1].rstrip('_')
        outputs = out if isinstance(out, (tuple, list)) else [out]
        n = sum(x.numel() for x in outputs if isinstance(x, torch.Tensor))
        # Scalar special operations counted as one nominal arithmetic operation.
        if name in ('add', 'sub', 'mul', 'div', 'pow', 'sqrt', 'rsqrt', 'sigmoid', 'relu', 'prelu', '_prelu_kernel'):
            self.counts[name] += n
        elif name in ('mean', 'sum'):
            self.counts[name] += args[0].numel()
        elif name == 'upsample_bilinear2d':
            self.counts[name] += n * 7
        elif name in ('convolution',):
            pass  # Counted by Conv2d hooks, once.
        else:
            self.unclassified[str(func)] += 1
        return out


def profile(upstream, variant, frames=16, height=1080, width=1920, widths=None):
    with torch.device('meta'):
        model = ShiftModel(upstream, variant, widths=widths).eval()
        x = torch.zeros(1, frames, 3, height, width)
    macs, biases = Counter(), Counter()
    handles = []
    def hook(name):
        def count(module, inputs, output):
            macs[name] += output.numel() * module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
            if module.bias is not None:
                biases[name] += output.numel()
        return count
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(hook(name)))
    counter = ArithmeticCounter()
    with torch.no_grad(), counter:
        y = model(x)
    for h in handles:
        h.remove()
    memory_only={'select','replication_pad2d','unsqueeze','view','roll','slice',
                 'zeros_like','copy_','cat','split','pixel_shuffle','alias'}
    unsupported=[op for op in counter.unclassified if op.split('.')[1] not in memory_only]
    if unsupported:
        raise RuntimeError(f'Unclassified arithmetic must be reviewed: {unsupported}')
    outputs = y.shape[1]
    conv_flops = 2 * sum(macs.values())
    extra = sum(counter.counts.values()) + sum(biases.values())
    return {'variant': variant, 'widths': model.widths, 'parameters_M': sum(p.numel() for p in model.parameters()) / 1e6,
            'input_frames': frames, 'output_frames': outputs, 'height': height, 'width': width,
            'GMAC_per_output': sum(macs.values()) / outputs / 1e9,
            'conv_GFLOPs_per_output': conv_flops / outputs / 1e9,
            'arithmetic_GFLOPs_per_output': (conv_flops + extra) / outputs / 1e9,
            'clip_GFLOPs': (conv_flops + extra) / 1e9,
            'convention': '1 MAC=2 FLOPs; actual T-4 outputs; overlap included; native resolution; no ensemble',
            'scalar_ops': dict(counter.counts), 'non_arithmetic_or_unclassified_dispatch': dict(counter.unclassified),
            'conv_MACs_by_module': dict(macs)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', required=True)
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--frames', type=int, default=16)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    report = profile(a.upstream, a.variant, a.frames)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if not isinstance(v,dict)}, indent=2))
    if a.variant != 'teacher' and report['arithmetic_GFLOPs_per_output'] > 500:
        raise SystemExit('BUDGET_FAIL: do not train this config')

if __name__ == '__main__':
    main()
