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


class ImageEncoder(nn.Module):
    """Outer full-resolution CNN encoder before temporal recurrence."""
    def __init__(self, channels=48, num_blocks=2):
        super().__init__()
        self.stem = ConvAct(3, channels)
        self.body = ResidualStack(channels, num_blocks)
        self.tail = ConvAct(channels, channels)

    def forward(self, x):
        return self.tail(self.body(self.stem(x)))


class StateConcatFusion(nn.Module):
    """Fuse current image feature and recurrent state by full-resolution concat."""
    def __init__(self, channels=48, num_blocks=1):
        super().__init__()
        self.reduce = ConvAct(channels * 2, channels, kernel_size=3)
        self.refine = ResidualStack(channels, num_blocks)

    def forward(self, image_feat, hidden):
        if image_feat.shape != hidden.shape:
            raise RuntimeError(
                f'Image feature / hidden mismatch: {tuple(image_feat.shape)} vs {tuple(hidden.shape)}'
            )
        return self.refine(self.reduce(torch.cat([image_feat, hidden], dim=1)))


class FullResPropagationUNet(nn.Module):
    """Recurrent U-Net with full-resolution input/output state.

    Internal downsampling is only for spatial receptive field. The hidden state
    entering and leaving this module remains BxCxHxW.
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


class OutputDecoder(nn.Module):
    """Outer full-resolution CNN decoder after bidirectional temporal fusion."""
    def __init__(self, feat_channels=48, hidden_channels=48, decode_channels=64, num_blocks=2):
        super().__init__()
        in_channels = feat_channels + hidden_channels * 2
        self.in_conv = ConvAct(in_channels, decode_channels)
        self.body = ResidualStack(decode_channels, num_blocks)
        self.to_feat = ConvAct(decode_channels, feat_channels)
        self.refine = ResidualConvBlock(feat_channels)
        self.to_rgb = nn.Conv2d(feat_channels, 3, 3, 1, 1, bias=True)

    def forward(self, image_feat, fwd_hidden, bwd_hidden):
        x = torch.cat([image_feat, fwd_hidden, bwd_hidden], dim=1)
        x = self.body(self.in_conv(x))
        x = self.refine(self.to_feat(x))
        return self.to_rgb(x)


class NanoVSRFullResConcatUNetDeblur(nn.Module):
    """Quality-first full-resolution recurrent U-Net deblurring model.

    Pipeline per frame:
      RGB frame
        -> outer full-resolution CNN ImageEncoder
        -> image feature F_t (Bx48xHxW)

      forward recurrence:
        concat(F_t, H_{t-1}^f) at full resolution
        -> direction-specific fusion CNN
        -> direction-specific full-resolution recurrent U-Net
        -> H_t^f (Bx48xHxW)

      backward recurrence mirrors the same design with independent parameters.

      output:
        concat(F_t, H_t^f, H_t^b) at full resolution
        -> outer CNN OutputDecoder
        -> RGB residual
        -> input + residual

    No BatchNorm is used. All recurrent hidden states remain full resolution.
    """
    def __init__(
        self,
        base_channels=48,
        mid_channels=64,
        bottleneck_channels=96,
        encoder_blocks=2,
        state_fusion_blocks=1,
        fullres_blocks=2,
        mid_blocks=2,
        bottleneck_blocks=4,
        decoder_channels=64,
        decoder_blocks=2,
        grad_checkpoint=False,
    ):
        super().__init__()
        self.base_channels = int(base_channels)
        self.mid_channels = int(mid_channels)
        self.bottleneck_channels = int(bottleneck_channels)
        self.encoder_blocks = int(encoder_blocks)
        self.state_fusion_blocks = int(state_fusion_blocks)
        self.fullres_blocks = int(fullres_blocks)
        self.mid_blocks = int(mid_blocks)
        self.bottleneck_blocks = int(bottleneck_blocks)
        self.decoder_channels = int(decoder_channels)
        self.decoder_blocks = int(decoder_blocks)
        self.grad_checkpoint = bool(grad_checkpoint)

        self.image_encoder = ImageEncoder(self.base_channels, self.encoder_blocks)

        self.forward_state_fusion = StateConcatFusion(
            self.base_channels, self.state_fusion_blocks
        )
        self.backward_state_fusion = StateConcatFusion(
            self.base_channels, self.state_fusion_blocks
        )

        prop_kwargs = dict(
            base_channels=self.base_channels,
            mid_channels=self.mid_channels,
            bottleneck_channels=self.bottleneck_channels,
            fullres_blocks=self.fullres_blocks,
            mid_blocks=self.mid_blocks,
            bottleneck_blocks=self.bottleneck_blocks,
        )
        self.forward_unet = FullResPropagationUNet(**prop_kwargs)
        self.backward_unet = FullResPropagationUNet(**prop_kwargs)

        self.output_decoder = OutputDecoder(
            feat_channels=self.base_channels,
            hidden_channels=self.base_channels,
            decode_channels=self.decoder_channels,
            num_blocks=self.decoder_blocks,
        )

    def set_grad_checkpoint(self, enabled=True):
        self.grad_checkpoint = bool(enabled)

    def config_dict(self):
        return {
            'base_channels': self.base_channels,
            'mid_channels': self.mid_channels,
            'bottleneck_channels': self.bottleneck_channels,
            'encoder_blocks': self.encoder_blocks,
            'state_fusion_blocks': self.state_fusion_blocks,
            'fullres_blocks': self.fullres_blocks,
            'mid_blocks': self.mid_blocks,
            'bottleneck_blocks': self.bottleneck_blocks,
            'decoder_channels': self.decoder_channels,
            'decoder_blocks': self.decoder_blocks,
        }

    def _run(self, module, *inputs):
        if self.training and self.grad_checkpoint:
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    def forward(self, x, return_features=False):
        if x.ndim != 5:
            raise ValueError(f'Expected B,T,C,H,W input, got {tuple(x.shape)}')
        b, t, c, h, w = x.shape
        if c != 3:
            raise ValueError(f'Expected RGB input with C=3, got C={c}')

        image_feats = [self._run(self.image_encoder, x[:, i]) for i in range(t)]

        fwd = []
        state = torch.zeros_like(image_feats[0])
        for i in range(t):
            fused_input = self._run(
                self.forward_state_fusion, image_feats[i], state
            )
            state = self._run(self.forward_unet, fused_input)
            if state.shape != image_feats[i].shape:
                raise RuntimeError('Forward hidden state is not full-resolution feature shape.')
            fwd.append(state)

        bwd = [None] * t
        state = torch.zeros_like(image_feats[0])
        for i in range(t - 1, -1, -1):
            fused_input = self._run(
                self.backward_state_fusion, image_feats[i], state
            )
            state = self._run(self.backward_unet, fused_input)
            if state.shape != image_feats[i].shape:
                raise RuntimeError('Backward hidden state is not full-resolution feature shape.')
            bwd[i] = state

        outs = []
        decoded_feats = []
        for i in range(t):
            residual = self._run(
                self.output_decoder, image_feats[i], fwd[i], bwd[i]
            )
            out = torch.clamp(x[:, i] + residual, 0.0, 1.0)
            outs.append(out)
            if return_features:
                decoded_feats.append(torch.cat([image_feats[i], fwd[i], bwd[i]], dim=1))

        out = torch.stack(outs, dim=1)
        if return_features:
            return out, torch.stack(decoded_feats, dim=1)
        return out
