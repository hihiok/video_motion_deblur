"""NanoVNR WaveShift-PAGF video deblurring network.

The model keeps the 12-channel RGB NanoVNR/NAFNet backbone, moves recurrent
propagation to the Haar LL band, adds lightweight grouped spatial-temporal
shift (GSTS) before recurrence, uses pixel-attention guided fusion (PAGF) for
state updates, and processes the three Haar high-frequency bands with a small
edge-aware branch.  Reparameterizable convolutions can be fused for inference.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .network_nanovnr_nafnet_rgb import NAFUNetPropagationDefineChannel


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


class HaarWavelet(nn.Module):
    """Fixed, device-agnostic Haar DWT/IWT with LL and three HF bands."""

    def __init__(self, channels):
        super().__init__()
        self.channels = int(channels)
        filters = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, 1.0]],
                [[-1.0, -1.0], [1.0, 1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ]
        ).unsqueeze(1)
        self.register_buffer('filters', filters.repeat(self.channels, 1, 1, 1))

    def decompose(self, x):
        if x.ndim != 4 or x.size(1) != self.channels:
            raise ValueError(f'Expected N,{self.channels},H,W, got {tuple(x.shape)}')
        if x.size(-2) % 2 or x.size(-1) % 2:
            raise ValueError('Haar input must be even; pad before decompose().')
        n, _, h, w = x.shape
        bands = F.conv2d(x, self.filters, stride=2, groups=self.channels) / 4.0
        bands = bands.reshape(n, self.channels, 4, h // 2, w // 2)
        bands = bands.permute(0, 2, 1, 3, 4).contiguous()
        return bands[:, 0], bands[:, 1:].reshape(n, self.channels * 3, h // 2, w // 2)

    def reconstruct(self, ll, hf):
        if ll.size(1) != self.channels or hf.size(1) != self.channels * 3:
            raise ValueError('Invalid LL/HF channels for Haar reconstruction.')
        n, _, h, w = ll.shape
        bands = torch.cat([ll, hf], dim=1).reshape(n, 4, self.channels, h, w)
        bands = bands.permute(0, 2, 1, 3, 4).reshape(n, self.channels * 4, h, w)
        return F.conv_transpose2d(
            bands, self.filters, stride=2, groups=self.channels
        )


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
        self.fusion = RepConv2d(channels * 2, channels)
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


class PAGF(nn.Module):
    """Learnable pixel-attention guided fusion with a stable current-frame prior."""

    def __init__(self, channels=12, initial_history_weight=0.1):
        super().__init__()
        if not 0.0 < initial_history_weight < 1.0:
            raise ValueError('initial_history_weight must be in (0,1).')
        self.channels = int(channels)
        self.qk = nn.Conv2d(channels * 2, channels * 2, 1, 1, 0)
        self.refine = RepConv2d(channels, channels, identity=True)
        self.activation = nn.PReLU(channels)
        initial_logit = math.log(initial_history_weight / (1.0 - initial_history_weight))
        self.gate_bias = nn.Parameter(torch.tensor(initial_logit))
        self.refine_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, current, history, return_gate=False):
        if current.shape != history.shape:
            raise ValueError(f'PAGF shape mismatch: {current.shape} vs {history.shape}')
        q, k = self.qk(torch.cat([current, history], dim=1)).chunk(2, dim=1)
        gate = torch.sigmoid(q * k + self.gate_bias)
        blended = current * (1.0 - gate) + history * gate
        output = blended + self.refine_scale * self.activation(self.refine(blended))
        if return_gate:
            return output, gate
        return output


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


class AdditivePAGFPropagation(nn.Module):
    """PAGF state selection followed by additive NAF U-Net refinement."""

    def __init__(self, channels=12):
        super().__init__()
        self.pagf = PAGF(channels, initial_history_weight=0.1)
        self.naf = NAFUNetPropagationDefineChannel(
            width=12,
            enc_blk_nums=(1, 1, 1),
            middle_blk_num=1,
            dec_blk_nums=(1, 1, 1),
            prop_channels=(24, 32, 48, 72),
        )
        self.update_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, current, history):
        selected = self.pagf(current, history)
        delta = self.naf(current, selected)
        return selected + self.update_scale * delta


class NanoVNRWaveShiftPAGF(nn.Module):
    """Effect-oriented, compact bidirectional NanoVNR deblurring model."""

    def __init__(
        self,
        num_feat=12,
        gsts_blocks=2,
        gsts_radii=(2, 4),
        use_edge_aware=True,
        grad_checkpoint=False,
    ):
        super().__init__()
        if num_feat != 12:
            raise ValueError('This controlled experiment fixes num_feat=12.')
        if int(gsts_blocks) != len(tuple(gsts_radii)):
            raise ValueError('gsts_blocks must equal len(gsts_radii).')
        self.num_feat = int(num_feat)
        self.gsts_blocks = int(gsts_blocks)
        self.gsts_radii = tuple(int(v) for v in gsts_radii)
        self.use_edge_aware = bool(use_edge_aware)
        self.grad_checkpoint = bool(grad_checkpoint)

        self.feat_extract = nn.Conv2d(3, num_feat, 3, 1, 1, bias=True)
        self.haar = HaarWavelet(num_feat)
        self.gsts = nn.ModuleList([
            GroupedSpatialTemporalShift(
                num_feat, spatial_radius=radius, diagonal=bool(index % 2)
            )
            for index, radius in enumerate(self.gsts_radii)
        ])
        self.hf_processor = (
            EdgeAwareHighFrequency(num_feat) if self.use_edge_aware else nn.Identity()
        )
        self.forward_net = AdditivePAGFPropagation(num_feat)
        self.backward_net = AdditivePAGFPropagation(num_feat)
        self.bidirectional_fusion = PAGF(num_feat, initial_history_weight=0.5)
        self.output_fusion = PAGF(num_feat, initial_history_weight=0.25)
        self.reconstruction = RepConv2d(num_feat, 3)
        self.output_scale = nn.Parameter(torch.tensor(0.1))

    @property
    def temporal_radius(self):
        return self.gsts_blocks

    def set_grad_checkpoint(self, enabled=True):
        self.grad_checkpoint = bool(enabled)

    def config_dict(self):
        return {
            'num_feat': self.num_feat,
            'input_channels': 3,
            'haar': True,
            'recurrent_resolution': '1/2',
            'gsts_blocks': self.gsts_blocks,
            'gsts_radii': list(self.gsts_radii),
            'gsts_branch': 'LL_only',
            'pagf': True,
            'additive_recurrence': True,
            'prop_channels': [24, 32, 48, 72],
            'edge_aware_hf': self.use_edge_aware,
            'repconv': True,
            'temporal_radius': self.temporal_radius,
        }

    @classmethod
    def from_config(cls, config, grad_checkpoint=False):
        return cls(
            num_feat=int(config.get('num_feat', 12)),
            gsts_blocks=int(config.get('gsts_blocks', 2)),
            gsts_radii=tuple(config.get('gsts_radii', (2, 4))),
            use_edge_aware=bool(config.get('edge_aware_hf', True)),
            grad_checkpoint=grad_checkpoint,
        )

    def _prop(self, module, current, history):
        if self.training and self.grad_checkpoint:
            return checkpoint(module, current, history, use_reentrant=False)
        return module(current, history)

    @staticmethod
    def _pad_even(features):
        h, w = features.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            features = F.pad(features, (0, pad_w, 0, pad_h), mode='reflect')
        return features

    def forward(self, x, prev_forward_feat=None, core_start=0, core_end=None):
        if x.ndim != 5:
            raise ValueError(f'Expected B,T,3,H,W, got {tuple(x.shape)}')
        b, t, c, h, w = x.shape
        if c != 3:
            raise ValueError(f'RGB model expects C=3, got {c}')
        core_end = t if core_end is None else int(core_end)
        core_start = int(core_start)
        if not 0 <= core_start < core_end <= t:
            raise ValueError(f'Invalid core range [{core_start},{core_end}) for T={t}')

        flat = self.feat_extract(x.reshape(b * t, c, h, w))
        flat = self._pad_even(flat)
        hp, wp = flat.shape[-2:]
        ll, hf = self.haar.decompose(flat)
        ll = ll.reshape(b, t, self.num_feat, hp // 2, wp // 2)
        hf = hf.reshape(b, t, self.num_feat * 3, hp // 2, wp // 2)

        for block in self.gsts:
            ll = block(ll)
        ll = ll[:, core_start:core_end]
        hf = hf[:, core_start:core_end]
        noisy_rgb = x[:, core_start:core_end]
        core_t = core_end - core_start

        hf_flat = hf.reshape(b * core_t, self.num_feat * 3, hp // 2, wp // 2)
        hf_flat = self.hf_processor(hf_flat)

        if prev_forward_feat is None:
            state = torch.zeros_like(ll[:, 0])
        else:
            if prev_forward_feat.shape != ll[:, 0].shape:
                raise ValueError(
                    f'prev_forward_feat shape mismatch: {tuple(prev_forward_feat.shape)} '
                    f'vs {tuple(ll[:, 0].shape)}'
                )
            state = prev_forward_feat
        forward_feats = []
        for index in range(core_t):
            state = self._prop(self.forward_net, ll[:, index], state)
            forward_feats.append(state)
        next_forward_feat = state

        state = torch.zeros_like(ll[:, 0])
        backward_feats = []
        for index in range(core_t - 1, -1, -1):
            state = self._prop(self.backward_net, ll[:, index], state)
            backward_feats.append(state)
        backward_feats.reverse()

        ll_outputs = []
        for index in range(core_t):
            temporal = self.bidirectional_fusion(
                forward_feats[index], backward_feats[index]
            )
            ll_outputs.append(self.output_fusion(ll[:, index], temporal))
        ll_flat = torch.stack(ll_outputs, dim=1).reshape(
            b * core_t, self.num_feat, hp // 2, wp // 2
        )
        reconstructed = self.haar.reconstruct(ll_flat, hf_flat)
        residual = self.reconstruction(reconstructed)[:, :, :h, :w]
        residual = residual.reshape(b, core_t, 3, h, w)
        output = noisy_rgb + self.output_scale * residual
        return output, next_forward_feat

    def switch_to_deploy(self):
        for module in list(self.modules()):
            if isinstance(module, RepConv2d):
                module.switch_to_deploy()
        return self
