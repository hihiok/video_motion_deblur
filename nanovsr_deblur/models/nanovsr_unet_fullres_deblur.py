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
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.act = nn.LeakyReLU(0.1, inplace=False)

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(x + y)


class ResidualStack(nn.Module):
    def __init__(self, channels, num_blocks):
        super().__init__()
        self.body = nn.Sequential(*[ResidualConvBlock(channels) for _ in range(int(num_blocks))])

    def forward(self, x):
        return self.body(x)


class FullResPropagationUNet(nn.Module):
    """U-Net recurrent update whose input/output state stays at full HxW resolution.

    The recurrent state itself is never stored at a reduced resolution. Each update
    receives a full-resolution feature/state tensor and returns a full-resolution
    tensor. The U-Net uses internal multi-scale branches only to enlarge receptive
    field; this is not low-resolution recurrent propagation.
    """
    def __init__(
        self,
        base_channels=48,
        mid_channels=64,
        bottleneck_channels=96,
        fullres_blocks=2,
        mid_blocks=2,
        bottleneck_blocks=4,
    ):
        super().__init__()
        self.pre = ResidualStack(base_channels, fullres_blocks)
        self.down1 = ConvAct(base_channels, mid_channels, stride=2)
        self.enc1 = ResidualStack(mid_channels, mid_blocks)
        self.down2 = ConvAct(mid_channels, bottleneck_channels, stride=2)
        self.bottleneck = ResidualStack(bottleneck_channels, bottleneck_blocks)

        self.up1 = ConvAct(bottleneck_channels + mid_channels, mid_channels)
        self.dec1 = ResidualStack(mid_channels, mid_blocks)
        self.up0 = ConvAct(mid_channels + base_channels, base_channels)
        self.dec0 = ResidualStack(base_channels, fullres_blocks)
        self.out = nn.Conv2d(base_channels, base_channels, 3, 1, 1, bias=True)
        self.act = nn.LeakyReLU(0.1, inplace=False)

    def forward(self, x):
        s0 = self.pre(x)
        s1 = self.enc1(self.down1(s0))
        z = self.bottleneck(self.down2(s1))

        y = F.interpolate(z, size=s1.shape[-2:], mode='bilinear', align_corners=False)
        y = self.dec1(self.up1(torch.cat([y, s1], dim=1)))
        y = F.interpolate(y, size=s0.shape[-2:], mode='bilinear', align_corners=False)
        y = self.dec0(self.up0(torch.cat([y, s0], dim=1)))
        return self.act(x + self.out(y))


class NanoVSRFullResUNetDeblur(nn.Module):
    """Quality-first NanoVSR-style deblurring with full-resolution recurrence.

    - RGB frames are mapped to full-resolution feature maps.
    - Forward/backward recurrent states are BxC x H x W at every time step.
    - Each recurrent update is a complete U-Net operating on the full-resolution
      feature/state tensor and returning a full-resolution state.
    - Forward and backward U-Nets use separate parameters.
    - No BatchNorm is used, which is safer for native full-frame batch-1 training.
    - Output resolution exactly matches the input resolution.
    """
    def __init__(
        self,
        base_channels=48,
        mid_channels=64,
        bottleneck_channels=96,
        fullres_blocks=2,
        mid_blocks=2,
        bottleneck_blocks=4,
        grad_checkpoint=False,
    ):
        super().__init__()
        self.base_channels = int(base_channels)
        self.mid_channels = int(mid_channels)
        self.bottleneck_channels = int(bottleneck_channels)
        self.fullres_blocks = int(fullres_blocks)
        self.mid_blocks = int(mid_blocks)
        self.bottleneck_blocks = int(bottleneck_blocks)
        self.grad_checkpoint = bool(grad_checkpoint)

        self.feat_extract = nn.Sequential(
            ConvAct(3, self.base_channels),
            ResidualConvBlock(self.base_channels),
        )
        prop_kwargs = dict(
            base_channels=self.base_channels,
            mid_channels=self.mid_channels,
            bottleneck_channels=self.bottleneck_channels,
            fullres_blocks=self.fullres_blocks,
            mid_blocks=self.mid_blocks,
            bottleneck_blocks=self.bottleneck_blocks,
        )
        self.forward_net = FullResPropagationUNet(**prop_kwargs)
        self.backward_net = FullResPropagationUNet(**prop_kwargs)
        self.fusion = nn.Sequential(
            nn.Conv2d(self.base_channels * 2, self.base_channels, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.1, inplace=False),
            ResidualConvBlock(self.base_channels),
        )
        self.head = nn.Sequential(
            ResidualConvBlock(self.base_channels),
            nn.Conv2d(self.base_channels, 3, 3, 1, 1, bias=True),
        )

    def set_grad_checkpoint(self, enabled=True):
        self.grad_checkpoint = bool(enabled)

    def config_dict(self):
        return {
            'base_channels': self.base_channels,
            'mid_channels': self.mid_channels,
            'bottleneck_channels': self.bottleneck_channels,
            'fullres_blocks': self.fullres_blocks,
            'mid_blocks': self.mid_blocks,
            'bottleneck_blocks': self.bottleneck_blocks,
        }

    def _run(self, module, x):
        if self.training and self.grad_checkpoint:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, x, return_features=False):
        if x.ndim != 5:
            raise ValueError(f'Expected B,T,C,H,W input, got {tuple(x.shape)}')
        b, t, c, h, w = x.shape
        if c != 3:
            raise ValueError(f'Expected RGB input with C=3, got C={c}')

        feats = []
        for i in range(t):
            feats.append(self._run(self.feat_extract, x[:, i]))

        fwd = []
        state = torch.zeros_like(feats[0])
        for i in range(t):
            state = self._run(self.forward_net, feats[i] + state)
            if state.shape[-2:] != (h, w):
                raise RuntimeError('Forward recurrent state is not full resolution.')
            fwd.append(state)

        bwd = [None] * t
        state = torch.zeros_like(feats[0])
        for i in range(t - 1, -1, -1):
            state = self._run(self.backward_net, feats[i] + state)
            if state.shape[-2:] != (h, w):
                raise RuntimeError('Backward recurrent state is not full resolution.')
            bwd[i] = state

        outs = []
        fused_all = []
        for i in range(t):
            fused = self._run(self.fusion, torch.cat([fwd[i], bwd[i]], dim=1))
            residual = self._run(self.head, fused)
            out = torch.clamp(x[:, i] + residual, 0.0, 1.0)
            outs.append(out)
            if return_features:
                fused_all.append(fused)

        out = torch.stack(outs, dim=1)
        if return_features:
            return out, torch.stack(fused_all, dim=1)
        return out
