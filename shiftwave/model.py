"""Shift-Net widths preserved; expensive fusion moves to the Haar LL band.

Reference: agent/nanovnr-waveshift-pagf-t6-fullframe-20260907,
network_nanovnr_waveshift_pagf.py (PAGF and edge-aware HF design).
This is an architectural adaptation, not a function-preserving pruning claim.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from shift500.model import load_upstream, checkpoint_state, VARIANTS, Bypass

MODEL_ID = 'shiftwave_ours_s_500g_v1'
REMOVED = ('orb4', 'orb5', 'rorb4', 'rorb5', 'lrelu')
BYPASSED = ('orb3', 'rorb3')


def haar(x):
    """Average-normalized Haar; all four bands retained, no image resize."""
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return (a+b+c+d)*.25, torch.cat(((-a+b-c+d)*.25,
                                                   (-a-b+c+d)*.25,
                                                   (a-b-c+d)*.25), dim=1)


def inverse_haar(ll, hf):
    h, v, d = hf.chunk(3, dim=1)
    pixels = torch.stack((ll-h-v+d, ll+h-v-d, ll-h+v-d, ll+h+v+d), dim=2)
    return F.pixel_shuffle(pixels.flatten(1, 2), 2)


class PAGF(nn.Module):
    """q*k pixel gate; favor the pretrained temporal candidate initially."""
    def __init__(self, channels):
        super().__init__()
        self.qk = nn.Conv2d(channels*2, channels*2, 1)
        self.gate_bias = nn.Parameter(torch.tensor(math.log(9.)))
        self.refine = nn.Conv2d(channels, channels, 3, padding=1)
        self.activation = nn.PReLU(channels)
        self.scale = nn.Parameter(torch.tensor(.01))

    def forward(self, current, temporal):
        q, k = self.qk(torch.cat((current, temporal), dim=1)).chunk(2, dim=1)
        gate = torch.sigmoid(q*k + self.gate_bias)
        x = current*(1-gate) + temporal*gate
        return x + self.scale*self.activation(self.refine(x))


class HighFrequency(nn.Module):
    """Three band-wise branches and a learnable Laplacian residual, no HF shift."""
    def __init__(self, channels):
        super().__init__()
        n = channels*3
        self.subbands = nn.Sequential(nn.Conv2d(n, n, 1, groups=3),
                                     nn.PReLU(n), nn.Conv2d(n, n, 1, groups=3))
        self.laplacian = nn.Conv2d(n, n, 3, padding=1, groups=n, bias=False)
        kernel = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]])
        with torch.no_grad():
            self.laplacian.weight.copy_(kernel[None, None].repeat(n, 1, 1, 1))
        self.edge = nn.Sequential(nn.PReLU(n), nn.Conv2d(n, n, 1, groups=3))
        self.condition = nn.Conv2d(channels, n, 1)
        self.band_scale = nn.Parameter(torch.tensor(.01))
        self.edge_scale = nn.Parameter(torch.tensor(0.))
        self.condition_scale = nn.Parameter(torch.tensor(.01))

    def forward(self, hf, ll_residual):
        # Predict a residual: input RGB already has a direct output skip.
        return (self.band_scale*self.subbands(hf) +
                self.edge_scale*self.edge(self.laplacian(hf)) +
                self.condition_scale*self.condition(ll_residual))


class ShiftWave(nn.Module):
    def __init__(self, upstream, activation_checkpointing=False, training_context=1):
        super().__init__()
        mod = load_upstream(upstream, VARIANTS['teacher'])
        self.net = mod.GShiftNet(future_frames=2, past_frames=2)
        for name in REMOVED:
            delattr(self.net, name)
        for name in BYPASSED:
            setattr(self.net, name, nn.Identity())
        # Preserve widths 14/64 and shapes of every retained pretrained tensor.
        # LL shallow scale: 1/3 repeated groups; LL deep scale: 2/3 groups.
        for level in ('encoder_level1', 'decoder_level1'):
            for suffix in ('_1', '_2'):
                setattr(self.net.stage1, level+suffix, Bypass())
        for level in ('encoder_level2', 'decoder_level2'):
            setattr(self.net.stage1, level+'_2', Bypass())
        self.pagf = PAGF(14)
        self.high_frequency = HighFrequency(14)
        self.training_context = training_context
        self.activation_checkpointing = activation_checkpointing
        if activation_checkpointing:
            for child in self.net.modules():
                if isinstance(child, (mod.TFR_UNet, mod.Encoder_shift_block)):
                    original = child.forward
                    def wrapped(*args, _forward=original, **kwargs):
                        if self.training and torch.is_grad_enabled():
                            return checkpoint(_forward, *args, use_reentrant=False, **kwargs)
                        return _forward(*args, **kwargs)
                    child.forward = wrapped

    def forward(self, x, context=None):
        context = (self.training_context if self.training else 2) if context is None else context
        if context not in (1, 2):
            raise ValueError('Context must be 1 or 2')
        if x.ndim != 5 or x.shape[0] != 1 or x.shape[2] != 3 or x.shape[1] <= 2*context:
            raise ValueError('Expected [1,T,3,H,W], T>2*context')
        if min(x.shape[-2:]) < 64:
            raise ValueError('Minimum spatial dimension is 64 for LL spatial shifts')
        h, w = x.shape[-2:]
        flat = F.pad(x[0], (0, (-w)%8, 0, (-h)%8), mode='replicate')
        features = self.net.feat_extract(flat)
        ll, hf = haar(features)
        spatial, transferred = self.net.stage0(ll)
        temporal = self.net.stage1(transferred)
        s = slice(context, -context)
        chosen = self.pagf(spatial[s], temporal[s])
        residual = self.net.rconcat(torch.cat((ll[s], spatial[s], chosen), dim=1))
        refined = self.net.rorb2(self.net.rorb1(residual)) + residual
        hf_residual = self.high_frequency(hf[s], refined)
        rgb_residual = self.net.conv_last(inverse_haar(refined, hf_residual))
        return (flat[s] + rgb_residual)[None, ..., :h, :w]


def initialize_pretrained(student, path):
    source = checkpoint_state(path)
    target = student.net.state_dict()
    bad = [k for k, v in target.items() if k not in source or source[k].shape != v.shape]
    if bad:
        raise ValueError(f'Pretrained Ours-s tensor mismatch: {bad[:12]}')
    student.net.load_state_dict({k: source[k] for k in target}, strict=True)
    return dict(method='exact_shape_retained_backbone', tensors=len(target),
                pretrained_elements=sum(v.numel() for v in target.values()),
                new_modules=['pagf', 'high_frequency'],
                function_preserving=False)
