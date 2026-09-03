import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ConvAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.act = nn.LeakyReLU(0.1, inplace=False)

    def forward(self, x):
        return self.act(self.conv(x))


class ResidualConvBlock(nn.Module):
    """Restoration-friendly residual block without BatchNorm.

    BatchNorm is intentionally avoided because full-frame mixed-dataset training
    uses batch size 1 and variable native resolutions.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.act = nn.LeakyReLU(0.1, inplace=False)

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(x + y)


class FrameEncoder(nn.Module):
    def __init__(self, base_channels=32, mid_channels=48, bottleneck_channels=64):
        super().__init__()
        self.stem = ConvAct(3, base_channels)
        self.enc0 = ResidualConvBlock(base_channels)
        self.down1 = ConvAct(base_channels, mid_channels, stride=2)
        self.enc1 = ResidualConvBlock(mid_channels)
        self.down2 = ConvAct(mid_channels, bottleneck_channels, stride=2)
        self.enc2 = ResidualConvBlock(bottleneck_channels)

    def forward(self, x):
        s0 = self.enc0(self.stem(x))
        s1 = self.enc1(self.down1(s0))
        z = self.enc2(self.down2(s1))
        return s0, s1, z


class TemporalPropagation(nn.Module):
    def __init__(self, channels, num_blocks=6):
        super().__init__()
        self.body = nn.Sequential(*[ResidualConvBlock(channels) for _ in range(num_blocks)])

    def forward(self, x):
        return self.body(x)


class FrameDecoder(nn.Module):
    def __init__(self, base_channels=32, mid_channels=48, bottleneck_channels=64):
        super().__init__()
        self.up1 = ConvAct(bottleneck_channels + mid_channels, mid_channels)
        self.dec1 = ResidualConvBlock(mid_channels)
        self.up0 = ConvAct(mid_channels + base_channels, base_channels)
        self.dec0 = ResidualConvBlock(base_channels)
        self.head = nn.Sequential(
            ConvAct(base_channels, base_channels),
            nn.Conv2d(base_channels, 3, 3, 1, 1, bias=True),
        )

    def forward(self, fused, skip1, skip0):
        x = F.interpolate(fused, size=skip1.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec1(self.up1(torch.cat([x, skip1], dim=1)))
        x = F.interpolate(x, size=skip0.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec0(self.up0(torch.cat([x, skip0], dim=1)))
        return self.head(x)


class NanoVSRUNetDeblur(nn.Module):
    """Full-frame video deblurring model with NanoVSR-style bidirectional recurrence.

    Spatial backbone is a lightweight 3-level U-Net instead of full-resolution
    RepVGG propagation. The recurrent state lives at 1/4 spatial resolution,
    while full/half-resolution encoder skips preserve spatial detail.

    Forward and backward temporal propagation use separate parameters, matching
    the bidirectional design philosophy of NanoVSR. The model is fully
    convolutional and returns the same HxW as the input with no resize/crop.
    """
    def __init__(
        self,
        base_channels=32,
        mid_channels=48,
        bottleneck_channels=64,
        num_temporal_blocks=6,
        grad_checkpoint=False,
    ):
        super().__init__()
        self.base_channels = int(base_channels)
        self.mid_channels = int(mid_channels)
        self.bottleneck_channels = int(bottleneck_channels)
        self.num_temporal_blocks = int(num_temporal_blocks)
        self.grad_checkpoint = bool(grad_checkpoint)

        self.encoder = FrameEncoder(
            self.base_channels, self.mid_channels, self.bottleneck_channels,
        )
        self.forward_net = TemporalPropagation(self.bottleneck_channels, self.num_temporal_blocks)
        self.backward_net = TemporalPropagation(self.bottleneck_channels, self.num_temporal_blocks)
        self.fusion = ConvAct(self.bottleneck_channels * 2, self.bottleneck_channels, kernel_size=1, padding=0)
        self.decoder = FrameDecoder(
            self.base_channels, self.mid_channels, self.bottleneck_channels,
        )

    def set_grad_checkpoint(self, enabled=True):
        self.grad_checkpoint = bool(enabled)

    def config_dict(self):
        return {
            'base_channels': self.base_channels,
            'mid_channels': self.mid_channels,
            'bottleneck_channels': self.bottleneck_channels,
            'num_temporal_blocks': self.num_temporal_blocks,
        }

    def _run_encoder(self, frame):
        if self.training and self.grad_checkpoint:
            return checkpoint(self.encoder, frame, use_reentrant=False)
        return self.encoder(frame)

    def _run_temporal(self, module, x):
        if self.training and self.grad_checkpoint:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def _run_decoder(self, fused, skip1, skip0):
        if self.training and self.grad_checkpoint:
            return checkpoint(self.decoder, fused, skip1, skip0, use_reentrant=False)
        return self.decoder(fused, skip1, skip0)

    def forward(self, x, return_features=False):
        if x.ndim != 5:
            raise ValueError(f'Expected B,T,C,H,W input, got {tuple(x.shape)}')
        b, t, c, h, w = x.shape
        if c != 3:
            raise ValueError(f'Expected RGB input with C=3, got C={c}')

        skip0, skip1, bottlenecks = [], [], []
        for i in range(t):
            s0, s1, z = self._run_encoder(x[:, i])
            skip0.append(s0)
            skip1.append(s1)
            bottlenecks.append(z)

        fwd = []
        state = torch.zeros_like(bottlenecks[0])
        for i in range(t):
            state = self._run_temporal(self.forward_net, bottlenecks[i] + state)
            fwd.append(state)

        bwd = [None] * t
        state = torch.zeros_like(bottlenecks[0])
        for i in range(t - 1, -1, -1):
            state = self._run_temporal(self.backward_net, bottlenecks[i] + state)
            bwd[i] = state

        outs = []
        fused_all = []
        for i in range(t):
            fused = self.fusion(torch.cat([fwd[i], bwd[i]], dim=1))
            residual = self._run_decoder(fused, skip1[i], skip0[i])
            out = torch.clamp(x[:, i] + residual, 0.0, 1.0)
            outs.append(out)
            if return_features:
                fused_all.append(fused)

        out = torch.stack(outs, dim=1)
        if return_features:
            return out, torch.stack(fused_all, dim=1)
        return out
