"""RT-Focuser backbone with zero-parameter video aggregation.

The frame backbone is a state-dict-compatible reimplementation of the official
MIT-licensed RT-Focuser model by ReaganWu/RT-Focuser.  The video wrapper adds:

* grouped spatial-temporal feature shifts inspired by Shift-Net-s; and
* similarity-gated bidirectional propagation inspired by DSTNet/DSTNet+.

Both temporal operators are parameter-free.  The default video backbone drops
one LD block from encoder2, encoder3 and encoder4, so parameters and per-frame arithmetic
remain below the unmodified RT-Focuser Standard baseline.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    # Official RT-Focuser uses F.interpolate's default nearest mode for MLIA
    # feature and RGB pyramid resizing.  Preserve it for checkpoint fidelity.
    return F.interpolate(x, size=size, mode="nearest")


class ConvBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, 1, 1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch_in, ch_out, 3, 1, 1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, size: tuple[int, int] | None = None) -> torch.Tensor:
        if size is None:
            return self.up(x)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.up[1:](x)


class SN_Module(nn.Module):
    """Official learnable per-channel sharpness-normalization module."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.kernel = nn.Parameter(torch.zeros(channels, 1, kernel_size, kernel_size))
        laplacian = torch.tensor(
            [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]
        )
        self.kernel.data.copy_(laplacian.repeat(channels, 1, 1, 1) / (kernel_size**2))
        self.bn = nn.BatchNorm2d(channels, eps, momentum, affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sharp = F.conv2d(x, self.kernel, padding=self.padding, groups=self.channels)
        return self.bn(sharp)


class Residual(nn.Module):
    def __init__(self, fn: nn.Module, ch_in: int):
        super().__init__()
        self.fn = fn
        self.ch_in = ch_in
        self.denoiser = SN_Module(channels=ch_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x + self.denoiser(x)


class LD_Block(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, depth: int = 1, k: int = 3):
        super().__init__()
        blocks = []
        for _ in range(depth):
            blocks.append(
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(ch_in, ch_in, k, groups=ch_in, padding=k // 2),
                            nn.GELU(),
                            nn.BatchNorm2d(ch_in),
                        ),
                        ch_in=ch_in,
                    ),
                    nn.Conv2d(ch_in, ch_in * 4, 1),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in * 4),
                    nn.Conv2d(ch_in * 4, ch_in, 1),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in),
                )
            )
        self.block = nn.Sequential(*blocks)
        self.up = ConvBlock(ch_in, ch_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.block(x))


class MLIA(nn.Module):
    def __init__(self, in_channels_list: Sequence[int], out_channels: int):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for ch in in_channels_list
            ]
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        hidden = max(out_channels // 4, 1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        projected = [branch(feat) for feat, branch in zip(inputs, self.branches)]
        fused = self.fusion_conv(torch.cat(projected, dim=1))
        return fused * self.channel_attention(fused)


class XFuse_Block(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, 3, 1, 1, groups=2, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, 1),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, 1),
            nn.GELU(),
            nn.BatchNorm2d(ch_out),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_out + 3, ch_out, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_out),
            nn.Conv2d(ch_out, ch_out * 4, 1),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, 1),
            nn.GELU(),
            nn.BatchNorm2d(ch_out),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv2(torch.cat((self.conv(x), skip), dim=1))


class RT_Focuser(nn.Module):
    """Frame model with official module names and configurable LD depths."""

    def __init__(
        self,
        input_channel: int = 3,
        dims: Sequence[int] = (16, 32, 128, 160, 256),
        depths: Sequence[int] = (3, 4, 4, 3, 2),
        kernels: Sequence[int] = (3, 3, 7, 7, 7),
    ):
        super().__init__()
        if not (len(dims) == len(depths) == len(kernels) == 5):
            raise ValueError("dims, depths and kernels must each have length five")
        self.dims = tuple(dims)
        self.depths = tuple(depths)
        self.Maxpool = nn.MaxPool2d(2, 2)
        self.stem = ConvBlock(input_channel, dims[0])
        self.encoder1 = LD_Block(dims[0], dims[0], depths[0], kernels[0])
        self.encoder2 = LD_Block(dims[0], dims[1], depths[1], kernels[1])
        self.encoder3 = LD_Block(dims[1], dims[2], depths[2], kernels[2])
        self.encoder4 = LD_Block(dims[2], dims[3], depths[3], kernels[3])
        self.encoder5 = LD_Block(dims[3], dims[4], depths[4], kernels[4])

        self.Up5 = UpConv(dims[4], dims[3])
        self.Up_conv5 = XFuse_Block(dims[3] * 2, dims[3])
        self.Up4 = UpConv(dims[3], dims[2])
        self.Up_conv4 = XFuse_Block(dims[2] * 2, dims[2])
        self.Up3 = UpConv(dims[2], dims[1])
        self.Up_conv3 = XFuse_Block(dims[1] * 2, dims[1])
        self.Up2 = UpConv(dims[1], dims[0])
        self.Up_conv2 = XFuse_Block(dims[0] * 2, dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], 3, 1, 1, 0)

        sources = [dims[0], dims[1], dims[2], dims[3]]
        self.Msf_8 = MLIA(sources, dims[3])
        self.Msf_4 = MLIA(sources, dims[2])
        self.Msf_2 = MLIA(sources, dims[1])
        self.Msf_1 = MLIA(sources, dims[0])

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x1 = self.encoder1(self.stem(x))
        x2 = self.encoder2(self.Maxpool(x1))
        x3 = self.encoder3(self.Maxpool(x2))
        x4 = self.encoder4(self.Maxpool(x3))
        x5 = self.encoder5(self.Maxpool(x4))
        return x1, x2, x3, x4, x5

    def decode(
        self,
        image: torch.Tensor,
        features: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        x1, x2, x3, x4, x5 = features
        f4 = self.Msf_8([_resize(x1, x4.shape[-2:]), _resize(x2, x4.shape[-2:]),
                         _resize(x3, x4.shape[-2:]), x4])
        f3 = self.Msf_4([_resize(x1, x3.shape[-2:]), _resize(x2, x3.shape[-2:]),
                         x3, _resize(x4, x3.shape[-2:])])
        f2 = self.Msf_2([_resize(x1, x2.shape[-2:]), x2, _resize(x3, x2.shape[-2:]),
                         _resize(x4, x2.shape[-2:])])
        f1 = self.Msf_1([x1, _resize(x2, x1.shape[-2:]), _resize(x3, x1.shape[-2:]),
                         _resize(x4, x1.shape[-2:])])

        o4 = _resize(image, x4.shape[-2:])
        o3 = _resize(image, x3.shape[-2:])
        o2 = _resize(image, x2.shape[-2:])
        d5 = self.Up_conv5(torch.cat((f4, self.Up5(x5, x4.shape[-2:])), dim=1), o4)
        d4 = self.Up_conv4(torch.cat((f3, self.Up4(d5, x3.shape[-2:])), dim=1), o3)
        d3 = self.Up_conv3(torch.cat((f2, self.Up3(d4, x2.shape[-2:])), dim=1), o2)
        d2 = self.Up_conv2(torch.cat((f1, self.Up2(d3, x1.shape[-2:])), dim=1), image)
        return torch.sigmoid(self.Conv_1x1(d2) + image)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(x, self.encode(x))


def RT_Focuser_Standard(
    dims: Sequence[int] = (16, 32, 128, 160, 256),
    depths: Sequence[int] = (3, 4, 4, 3, 2),
    kernels: Sequence[int] = (3, 3, 7, 7, 7),
) -> RT_Focuser:
    return RT_Focuser(dims=dims, depths=depths, kernels=kernels)


def _spatial_translate(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Translate B,T,C,H,W without circular wraparound."""
    b, t, c, h, w = x.shape
    padded = F.pad(x.reshape(b * t, c, h, w), (1, 1, 1, 1), mode="replicate")
    y0 = 1 - dy
    x0 = 1 - dx
    return padded[..., y0 : y0 + h, x0 : x0 + w].reshape(b, t, c, h, w)


class GroupedSpatialTemporalShift(nn.Module):
    """Zero-parameter grouped shift over adjacent frames and nearby pixels."""

    def __init__(self, fold_div: int = 8):
        super().__init__()
        if fold_div < 6:
            raise ValueError("fold_div must be >= 6")
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected B,T,C,H,W, got {tuple(x.shape)}")
        c = x.shape[2]
        fold = c // self.fold_div
        if fold == 0 or x.shape[1] == 1:
            return x
        prev = torch.cat((x[:, :1], x[:, :-1]), dim=1)
        nxt = torch.cat((x[:, 1:], x[:, -1:]), dim=1)
        out = x.clone()
        out[:, :, 0 * fold : 1 * fold] = prev[:, :, 0 * fold : 1 * fold]
        out[:, :, 1 * fold : 2 * fold] = nxt[:, :, 1 * fold : 2 * fold]
        out[:, :, 2 * fold : 3 * fold] = _spatial_translate(
            prev[:, :, 2 * fold : 3 * fold], 0, 1
        )
        out[:, :, 3 * fold : 4 * fold] = _spatial_translate(
            nxt[:, :, 3 * fold : 4 * fold], 0, -1
        )
        out[:, :, 4 * fold : 5 * fold] = _spatial_translate(
            prev[:, :, 4 * fold : 5 * fold], 1, 0
        )
        out[:, :, 5 * fold : 6 * fold] = _spatial_translate(
            nxt[:, :, 5 * fold : 6 * fold], -1, 0
        )
        return out


class SimilarityGatedBidirectionalPropagation(nn.Module):
    """Parameter-free propagation that rejects dissimilar/moving features.

    A high feature difference drives similarity toward zero, leaving the current
    frame unchanged.  Similar regions borrow weak evidence from both directions.
    This makes the operator safer than unconditional temporal averaging.
    """

    def __init__(self, strength: float = 0.25, temperature: float = 0.15):
        super().__init__()
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be within [0,1]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.strength = float(strength)
        self.temperature = float(temperature)

    def _step(self, current: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        diff = (current - history).abs().mean(dim=1, keepdim=True)
        similarity = torch.exp(-diff / self.temperature)
        return current + self.strength * similarity * (history - current)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected B,T,C,H,W, got {tuple(x.shape)}")
        if x.shape[1] == 1 or self.strength == 0:
            return x
        forward = [x[:, 0]]
        for index in range(1, x.shape[1]):
            forward.append(self._step(x[:, index], forward[-1]))
        backward = [x[:, -1]]
        for index in range(x.shape[1] - 2, -1, -1):
            backward.append(self._step(x[:, index], backward[-1]))
        backward.reverse()
        fwd = torch.stack(forward, dim=1)
        bwd = torch.stack(backward, dim=1)
        return x + 0.5 * ((fwd - x) + (bwd - x))


class RTFocuserT6(nn.Module):
    """T-frame RT-Focuser video model; training defaults to T=6."""

    def __init__(
        self,
        dims: Sequence[int] = (16, 32, 128, 160, 256),
        depths: Sequence[int] = (3, 3, 3, 2, 2),
        kernels: Sequence[int] = (3, 3, 7, 7, 7),
        shift_fold_div: int = 8,
        shift_strength: float = 0.5,
        propagation_strength: float = 0.25,
        propagation_temperature: float = 0.15,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        if not 0.0 <= shift_strength <= 1.0:
            raise ValueError("shift_strength must be within [0,1]")
        self.backbone = RT_Focuser(dims=dims, depths=depths, kernels=kernels)
        self.temporal_shift = GroupedSpatialTemporalShift(shift_fold_div)
        self.shift_strength = float(shift_strength)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.bidirectional = SimilarityGatedBidirectionalPropagation(
            propagation_strength, propagation_temperature
        )

    @contextmanager
    def _preserve_batch_norm_running_stats(self):
        """Prevent checkpoint recomputation from updating BN buffers twice."""
        snapshots = []
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.track_running_stats:
                snapshots.append(
                    (
                        module,
                        module.running_mean.detach().clone(),
                        module.running_var.detach().clone(),
                        module.num_batches_tracked.detach().clone(),
                    )
                )
        try:
            yield
        finally:
            with torch.no_grad():
                for module, mean, variance, batches in snapshots:
                    module.running_mean.copy_(mean)
                    module.running_var.copy_(variance)
                    module.num_batches_tracked.copy_(batches)

    def _memory_efficient_call(self, function, *args: torch.Tensor) -> torch.Tensor:
        if not self.training or not self.activation_checkpointing:
            return function(*args)

        def contexts():
            return nullcontext(), self._preserve_batch_norm_running_stats()

        return checkpoint(function, *args, use_reentrant=False, context_fn=contexts)

    @staticmethod
    def _to_4d(x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        return x.reshape(b * t, c, h, w)

    @staticmethod
    def _to_5d(x: torch.Tensor, b: int, t: int) -> torch.Tensor:
        return x.reshape(b, t, *x.shape[1:])

    def _shift(self, x: torch.Tensor) -> torch.Tensor:
        if self.shift_strength == 0.0:
            return x
        shifted = self.temporal_shift(x)
        if self.shift_strength == 1.0:
            return shifted
        return torch.lerp(x, shifted, self.shift_strength)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError(f"Expected B,T,3,H,W, got {tuple(video.shape)}")
        b, t, _, original_h, original_w = video.shape
        pad_h = (-original_h) % 16
        pad_w = (-original_w) % 16
        if pad_h or pad_w:
            flat = self._to_4d(video)
            flat = F.pad(flat, (0, pad_w, 0, pad_h), mode="replicate")
            video = self._to_5d(flat, b, t)
        flat_image = self._to_4d(video)

        x1 = self._memory_efficient_call(
            lambda image: self.backbone.encoder1(self.backbone.stem(image)), flat_image
        )
        x1 = self._shift(self._to_5d(x1, b, t))
        x2 = self._memory_efficient_call(
            self.backbone.encoder2, self.backbone.Maxpool(self._to_4d(x1))
        )
        x2 = self._shift(self._to_5d(x2, b, t))
        x3 = self._memory_efficient_call(
            self.backbone.encoder3, self.backbone.Maxpool(self._to_4d(x2))
        )
        x3 = self._shift(self._to_5d(x3, b, t))
        x4 = self._memory_efficient_call(
            self.backbone.encoder4, self.backbone.Maxpool(self._to_4d(x3))
        )
        x4 = self._shift(self._to_5d(x4, b, t))
        x4 = self.bidirectional(x4)
        x5 = self._memory_efficient_call(
            self.backbone.encoder5, self.backbone.Maxpool(self._to_4d(x4))
        )

        x1_4d, x2_4d, x3_4d, x4_4d = tuple(
            self._to_4d(x) for x in (x1, x2, x3, x4)
        )
        output = self._memory_efficient_call(
            lambda image, f1, f2, f3, f4, f5: self.backbone.decode(
                image, (f1, f2, f3, f4, f5)
            ),
            flat_image,
            x1_4d,
            x2_4d,
            x3_4d,
            x4_4d,
            x5,
        )
        output = self._to_5d(output, b, t)
        return output[..., :original_h, :original_w]
