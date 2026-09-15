"""DSTNet+ Base compressed internally; preserve 64-channel wave/PAGF topology.

The upstream architecture remains pinned and is loaded through the existing
exact PyTorch dynamic-convolution adapter. No opaque CUDA extension is needed.
"""
from pathlib import Path
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from adapters.dstnetplus_compat import load_dstnetplus_base, IDynamicDWConv
from .waveshift import GroupedSpatialTemporalShift, EdgeAwareHighFrequency, RepConv2d

UPSTREAM_COMMIT = '54363c15d8b924aa1ae56b8f835c1f0289954e95'
ARCHITECTURE = 'dstplus64_svd_separable_depth13_dyn_g8_projection_g8_waveshift_g16_v1'


class FactorConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwise=nn.Conv2d(64,64,3,padding=1,groups=64,bias=False)
        self.pointwise=nn.Conv2d(64,64,1)

    def forward(self,x):
        return self.pointwise(self.depthwise(x))


class SeparableResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=FactorConv();self.conv2=FactorConv()
        self.act=nn.LeakyReLU(.1)

    def forward(self, x):
        return x+self.conv2(self.act(self.conv1(x)))


class GroupedKernelProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(192,576,1,groups=8)

    def forward(self, x):
        b,c,h,w=x.shape
        # Each channel group sees all three progressive spatial scales.
        x=x.reshape(b,3,8,8,h,w).transpose(1,2).reshape(b,192,h,w)
        return self.conv(x)


class SlimPropagation(nn.Module):
    def __init__(self, original, backward, grad_checkpoint):
        super().__init__()
        self.fusion = original.fusion
        # Preserve progressive 3-scale dynamic filtering, group its expensive
        # kernel-generator convolutions. Dynamic filtering itself is unchanged.
        self.kernel_conv_pixel = IDynamicDWConv(64, 3, 1, 3, 8)
        self.kernel_conv_pixel.tokernel = GroupedKernelProjection()
        self.blocks = nn.Sequential(*[SeparableResidual() for _ in range(13)])
        self.backward = backward
        self.grad_checkpoint = grad_checkpoint

    def step(self, current, state):
        return self.blocks(self.kernel_conv_pixel(self.fusion(current, state)))

    def forward(self, features):
        state = torch.zeros_like(features[:, 0])
        order = range(features.shape[1]-1, -1, -1) if self.backward else range(features.shape[1])
        out = []
        for i in order:
            if self.training and self.grad_checkpoint:
                state = checkpoint(self.step, features[:, i], state, use_reentrant=False)
            else:
                state = self.step(features[:, i], state)
            out.append(state)
        if self.backward:
            out.reverse()
        return torch.stack(out, dim=1)


class Student(nn.Module):
    def __init__(self, upstream, grad_checkpoint=True):
        super().__init__()
        cls, _ = load_dstnetplus_base(upstream)
        self.core = cls(64, 3, 15, nonblind_denoise=False)
        self.core.backward_propagation = SlimPropagation(self.core.backward_propagation, True, grad_checkpoint)
        self.core.forward_propagation = SlimPropagation(self.core.forward_propagation, False, grad_checkpoint)
        self.shifts = nn.Sequential(
            GroupedSpatialTemporalShift(64, 2, False),
            GroupedSpatialTemporalShift(64, 4, True))
        self.edge = EdgeAwareHighFrequency(64)
        # Preserve pretrained HF processing at initialization; activate only the
        # added Laplacian path, not another pair of subband transforms.
        del self.edge.subband_1, self.edge.subband_2, self.edge.subband_scale
        self.grad_checkpoint = grad_checkpoint

    def forward(self, x):
        if x.ndim != 5 or x.shape[2] != 3 or min(x.shape[-2:]) < 16:
            raise ValueError('Expected B,T,3,H,W with H,W >=16')
        b,t,c,h,w = x.shape
        padded = self.core.spatial_padding(x, 2) if h%2 or w%2 else x
        ph,pw = padded.shape[-2:]
        f = self.core.feat_extractor(padded.reshape(b*t,3,ph,pw))
        ll,hf = self.core.wave(f)
        hf = self.core.x_wave_2_conv2(self.core.lrelu(self.core.x_wave_2_conv1(hf)))
        hf = hf + self.edge.edge_scale * self.edge.edge_projection(self.edge.activation(self.edge.laplacian(hf)))
        ll = self.shifts(ll.reshape(b,t,64,ph//2,pw//2))
        ll = self.core.backward_propagation(ll)
        ll = self.core.forward_propagation(ll).reshape(b*t,64,ph//2,pw//2)
        y = self.core.recons(self.core.wave(torch.cat([ll,hf],1), rev=True))
        return (y.reshape(b,t,3,ph,pw)+padded)[...,:h,:w]

    def deploy(self):
        for module in list(self.modules()):
            if isinstance(module, RepConv2d):
                module.switch_to_deploy()
        return self


def teacher(upstream, checkpoint_path=None):
    cls, _ = load_dstnetplus_base(upstream)
    model = cls(64, 3, 15, nonblind_denoise=False)
    if checkpoint_path:
        model.load_state_dict(teacher_state(checkpoint_path), strict=True)
    return model.eval().requires_grad_(False)


def teacher_state(path):
    raw = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('params_ema', 'params', 'state_dict'):
        if isinstance(raw, dict) and key in raw:
            raw = raw[key]
            break
    return {k.removeprefix('module.'):v for k,v in raw.items()}


@torch.no_grad()
def initialize(student, checkpoint_path):
    source = teacher_state(checkpoint_path)
    target = student.state_dict()
    copied, grouped, fresh, factorized = [], [], [], []
    factor_keys=set()
    # Best rank-1 spatial/channel factorization for each input-channel kernel.
    # Both original 3x3 convs retain their bias and inter-convolution activation.
    # Evenly sample 13 of the original 15 blocks, preserving first and last.
    retained=[round(i*14/12) for i in range(13)]
    for direction in ('backward_propagation','forward_propagation'):
        for i,j in enumerate(retained):
            for conv in ('conv1','conv2'):
                old=f'{direction}.resblock_bcakward2d.main.0.{j}.{conv}'
                new=f'core.{direction}.blocks.{i}.{conv}'
                weight=source[old+'.weight'].float().permute(1,0,2,3).reshape(64,64,9)
                u,s,vh=torch.linalg.svd(weight,full_matrices=False)
                root=s[:,0].sqrt()
                target[new+'.depthwise.weight'].copy_((vh[:,0]*root[:,None]).reshape(64,1,3,3))
                target[new+'.pointwise.weight'].copy_((u[:,:,0]*root[:,None]).t().reshape(64,64,1,1))
                target[new+'.pointwise.bias'].copy_(source[old+'.bias'])
                factor_keys.update(new+suffix for suffix in ('.depthwise.weight','.pointwise.weight','.pointwise.bias'))
                factorized.append({'source':old,'target':new,'retained_kernel_energy':float(s[:,0].square().sum()/s.square().sum().clamp_min(1e-30))})
    for key, dst in target.items():
        if key in factor_keys: continue
        source_key=key.removeprefix('core.')
        old = source.get(source_key)
        if '.tokernel.conv.' in source_key:
            old=source.get(source_key.replace('.tokernel.conv.','.tokernel.'))
            if old is not None and old.ndim==4:
                for g in range(8):
                    cols=torch.cat([torch.arange(s*64+g*8,s*64+(g+1)*8) for s in range(3)])
                    dst[g*72:(g+1)*72].copy_(old[g*72:(g+1)*72,cols])
                grouped.append(key)
                continue
        if old is not None and old.shape == dst.shape:
            dst.copy_(old); copied.append(key)
        elif old is not None and old.ndim == dst.ndim == 4 and old.shape[0] == dst.shape[0] and old.shape[1] == dst.shape[1]*8:
            # Dense -> grouped: retain diagonal group slices, exact provenance.
            per_group = old.shape[0]//8
            for g in range(8):
                dst[g*per_group:(g+1)*per_group].copy_(old[g*per_group:(g+1)*per_group,g*dst.shape[1]:(g+1)*dst.shape[1]])
            grouped.append(key)
        else:
            fresh.append(key)
    student.load_state_dict(target, strict=True)
    return {'exact_copy':copied,'group_diagonal_copy':grouped,'svd_factorized':factorized,
            'retained_teacher_blocks':retained,'new_parameters':fresh,
            'note':'SVD minimizes kernel error, not task loss. Fine-tuning/distillation still required.'}


def load_student(upstream, path, device='cpu', deploy=True):
    raw = torch.load(path, map_location='cpu', weights_only=False)
    if raw.get('architecture') != ARCHITECTURE:
        raise ValueError('Student checkpoint architecture mismatch')
    model = Student(upstream, grad_checkpoint=False)
    if raw.get('deployed', False):
        model.deploy()
    model.load_state_dict(raw['model'], strict=True)
    if deploy:
        model.deploy()
    return model.to(device).eval(), raw
