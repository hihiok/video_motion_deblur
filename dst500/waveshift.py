"""Adapted from user WaveShift-PAGF reference; no NAF/Nano backbone dependency."""
import torch
from torch import nn
from torch.nn import functional as F

class RepConv2d(nn.Module):
    """Linear 3x3 + 1x1 (+ identity) branches fuseable to one 3x3 conv."""

    def __init__(self, in_channels, out_channels, groups=1, identity=False, deploy=False):
        super().__init__()
        if in_channels % groups or out_channels % groups:
            raise ValueError('RepConv channels must be divisible by groups.')
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.groups = int(groups)
        self.use_identity = bool(identity and in_channels == out_channels)
        self.deploy = bool(deploy)
        if self.deploy:
            self.reparam = nn.Conv2d(
                in_channels, out_channels, 3, 1, 1, groups=groups, bias=True
            )
        else:
            self.branch_3x3 = nn.Conv2d(
                in_channels, out_channels, 3, 1, 1, groups=groups, bias=True
            )
            self.branch_1x1 = nn.Conv2d(
                in_channels, out_channels, 1, 1, 0, groups=groups, bias=True
            )

    def forward(self, x):
        if self.deploy:
            return self.reparam(x)
        y = self.branch_3x3(x) + self.branch_1x1(x)
        if self.use_identity:
            y = y + x
        return y

    def _identity_kernel_bias(self, device, dtype):
        kernel = torch.zeros(
            self.out_channels,
            self.in_channels // self.groups,
            3,
            3,
            device=device,
            dtype=dtype,
        )
        if self.use_identity:
            input_per_group = self.in_channels // self.groups
            for channel in range(self.out_channels):
                kernel[channel, channel % input_per_group, 1, 1] = 1.0
        bias = torch.zeros(self.out_channels, device=device, dtype=dtype)
        return kernel, bias

    def equivalent_kernel_bias(self):
        if self.deploy:
            return self.reparam.weight, self.reparam.bias
        k3 = self.branch_3x3.weight
        b3 = self.branch_3x3.bias
        k1 = F.pad(self.branch_1x1.weight, (1, 1, 1, 1))
        b1 = self.branch_1x1.bias
        kid, bid = self._identity_kernel_bias(k3.device, k3.dtype)
        return k3 + k1 + kid, b3 + b1 + bid

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self.equivalent_kernel_bias()
        reparam = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            3,
            1,
            1,
            groups=self.groups,
            bias=True,
        ).to(device=kernel.device, dtype=kernel.dtype)
        reparam.weight.data.copy_(kernel)
        reparam.bias.data.copy_(bias)
        self.reparam = reparam
        del self.branch_3x3
        del self.branch_1x1
        self.deploy = True

def _shift_group_spatial(x, offsets):
    """Shift each channel with replicate padding; x is B,T,C,H,W."""
    if not offsets:
        return x
    b, t, c, h, w = x.shape
    radius = max(max(abs(dy), abs(dx)) for dy, dx in offsets)
    flat = x.reshape(b * t, c, h, w)
    padded = F.pad(flat, (radius, radius, radius, radius), mode='replicate')
    shifted = []
    for channel in range(c):
        dy, dx = offsets[channel % len(offsets)]
        y0 = radius - dy
        x0 = radius - dx
        shifted.append(padded[:, channel:channel + 1, y0:y0 + h, x0:x0 + w])
    return torch.cat(shifted, dim=1).reshape(b, t, c, h, w)

class GroupedSpatialTemporalShift(nn.Module):
    """Lightweight GSTS adapted to a 12-channel half-resolution sequence.

    One channel group receives the previous frame, one receives the next frame,
    and the rest remains at the current frame. Transported groups are spatially
    shifted in four candidate directions, then fused with the unshifted feature.
    """

    def __init__(self, channels=12, spatial_radius=2, diagonal=False):
        super().__init__()
        if channels < 6:
            raise ValueError('GSTS requires at least 6 channels.')
        self.channels = int(channels)
        self.spatial_radius = int(spatial_radius)
        self.diagonal = bool(diagonal)
        self.prev_channels = channels // 3
        self.next_channels = channels // 3
        self.fusion = RepConv2d(channels * 2, channels, groups=16)
        self.activation = nn.PReLU(channels)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def _offsets(self):
        r = self.spatial_radius
        if self.diagonal:
            return [(-r, -r), (-r, r), (r, -r), (r, r)]
        return [(-r, 0), (r, 0), (0, -r), (0, r)]

    def forward(self, x):
        if x.ndim != 5 or x.size(2) != self.channels:
            raise ValueError(f'Expected B,T,{self.channels},H,W, got {tuple(x.shape)}')
        pc = self.prev_channels
        nc = self.next_channels
        prev = torch.cat([x[:, :1, :pc], x[:, :-1, :pc]], dim=1)
        nxt = torch.cat([x[:, 1:, pc:pc + nc], x[:, -1:, pc:pc + nc]], dim=1)
        prev = _shift_group_spatial(prev, self._offsets())
        nxt = _shift_group_spatial(nxt, list(reversed(self._offsets())))
        stay = x[:, :, pc + nc:]
        shifted = torch.cat([prev, nxt, stay], dim=2)
        b, t, c, h, w = x.shape
        fused = torch.cat([x, shifted], dim=2).reshape(b * t, c * 2, h, w)
        fused = self.activation(self.fusion(fused)).reshape(b, t, c, h, w)
        return x + self.residual_scale * fused

class EdgeAwareHighFrequency(nn.Module):
    """Subband-preserving HF refinement plus learnable Laplacian residual."""

    def __init__(self, channels=12):
        super().__init__()
        hf_channels = channels * 3
        self.hf_channels = hf_channels
        self.subband_1 = nn.Conv2d(hf_channels, hf_channels, 1, groups=3)
        self.activation = nn.PReLU(hf_channels)
        self.subband_2 = nn.Conv2d(hf_channels, hf_channels, 1, groups=3)
        self.subband_scale = nn.Parameter(torch.tensor(0.1))
        self.laplacian = nn.Conv2d(
            hf_channels, hf_channels, 3, 1, 1, groups=hf_channels, bias=False
        )
        kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
        )
        self.laplacian.weight.data.copy_(
            kernel.reshape(1, 1, 3, 3).repeat(hf_channels, 1, 1, 1)
        )
        self.edge_projection = nn.Conv2d(
            hf_channels, hf_channels, 1, groups=3, bias=True
        )
        self.edge_scale = nn.Parameter(torch.zeros(()))

    def forward(self, hf):
        base = self.subband_2(self.activation(self.subband_1(hf)))
        hf = hf + self.subband_scale * base
        edge = self.edge_projection(self.activation(self.laplacian(hf)))
        return hf + self.edge_scale * edge
