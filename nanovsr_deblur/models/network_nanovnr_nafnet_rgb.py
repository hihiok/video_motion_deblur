import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelRowLayerNorm(nn.Module):
    """LayerNorm over C and W for every B,H row, matching the supplied model."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.channels = channels
        self.eps = eps

    def forward(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        c = x.size(2)
        w = x.size(3)
        x = F.layer_norm(
            x,
            normalized_shape=(c, w),
            weight=None,
            bias=None,
            eps=self.eps,
        )
        return x.permute(0, 2, 1, 3).contiguous()


class NAFBlockLayerNormCRandWithoutSCA(nn.Module):
    def __init__(self, c, DW_Expand=1, FFN_Expand=1, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv2 = nn.Conv2d(c, dw_channel, 3, 1, 1, bias=True)
        self.prelu1 = nn.PReLU(dw_channel)
        self.conv3 = nn.Conv2d(dw_channel, c, 1, 1, 0, bias=True)

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, bias=True)
        self.prelu2 = nn.PReLU(ffn_channel)
        self.conv5 = nn.Conv2d(ffn_channel, c, 1, 1, 0, bias=True)

        self.norm1 = ChannelRowLayerNorm(c)
        self.norm2 = ChannelRowLayerNorm(c)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv2(x)
        x = self.prelu1(x)
        x = self.conv3(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.prelu2(x)
        x = self.conv5(x)
        return y + x * self.gamma


class NAFUNetPropagationDefineChannel(nn.Module):
    """Exact propagation U-Net from the supplied model.

    cur_feat/prop_feat: [B, 12, H, W]
    concat input:       [B, 24, H, W]
    channels:           24 -> 32 -> 48 -> 72 -> 48 -> 32 -> 24 -> 12
    hidden output:      [B, 12, H, W]
    """

    def __init__(
        self,
        width=12,
        enc_blk_nums=(1, 1, 1),
        middle_blk_num=1,
        dec_blk_nums=(1, 1, 1),
        drop_out_rate=0.0,
        prop_channels=(24, 32, 48, 72),
        **kwargs,
    ):
        super().__init__()
        self.width = width
        self.prop_channels = list(prop_channels)

        # Exact supplied architecture is defined for width=12 and 24ch concat.
        if width != 12 or tuple(prop_channels) != (24, 32, 48, 72):
            raise ValueError(
                'This experiment intentionally fixes the supplied architecture: '
                'width=12, prop_channels=(24,32,48,72).'
            )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        enc_channels = self.prop_channels[:-1]
        middle_ch = self.prop_channels[-1]

        for idx, num in enumerate(enc_blk_nums):
            in_ch = self.prop_channels[idx]
            out_ch = self.prop_channels[idx + 1]
            self.encoders.append(
                nn.Sequential(
                    *[
                        NAFBlockLayerNormCRandWithoutSCA(
                            in_ch, drop_out_rate=drop_out_rate
                        )
                        for _ in range(num)
                    ]
                )
            )
            self.downs.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, bias=True)
            )

        self.middle_blks = nn.Sequential(
            *[
                NAFBlockLayerNormCRandWithoutSCA(
                    middle_ch, drop_out_rate=drop_out_rate
                )
                for _ in range(middle_blk_num)
            ]
        )

        chan = middle_ch
        skip_channels = enc_channels[::-1]
        for num, skip_ch in zip(dec_blk_nums, skip_channels):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(
                        chan,
                        skip_ch * 4,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                    nn.PixelShuffle(2),
                )
            )
            chan = skip_ch
            self.decoders.append(
                nn.Sequential(
                    *[
                        NAFBlockLayerNormCRandWithoutSCA(
                            chan, drop_out_rate=drop_out_rate
                        )
                        for _ in range(num)
                    ]
                )
            )

        self.feature_out = nn.Conv2d(chan, width, 3, 1, 1, bias=True)
        self.padder_size = 2 ** len(self.encoders)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))

    def forward(self, cur_feat, prop_feat):
        _, _, h, w = cur_feat.shape
        x = torch.cat([cur_feat, prop_feat], dim=1)
        x = self.check_image_size(x)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(
            self.decoders, self.ups, encs[::-1]
        ):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.feature_out(x)
        return x[:, :, :h, :w]


class NanoVNRNAFNetRGB(nn.Module):
    """Supplied NanoVNR NAFNet structure with RGB input as the only architecture change.

    Differences from the supplied Python file:
      - input has 3 RGB channels instead of 4 channels
      - feat_extract is Conv2d(3, 12, 3, 1, 1)

    Everything else intentionally follows the supplied model structure.
    """

    def __init__(self, num_feat=12):
        super().__init__()
        if num_feat != 12:
            raise ValueError('Exact supplied architecture requires num_feat=12.')
        self.num_feat = num_feat

        # Only requested architecture change: 4 input channels -> RGB 3 channels.
        self.feat_extract = nn.Conv2d(3, num_feat, 3, 1, 1, bias=True)

        self.forward_net = NAFUNetPropagationDefineChannel()
        self.backward_net = NAFUNetPropagationDefineChannel()
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1, bias=True)

    def config_dict(self):
        return {
            'num_feat': 12,
            'input_channels': 3,
            'prop_channels': [24, 32, 48, 72],
            'enc_blk_nums': [1, 1, 1],
            'middle_blk_num': 1,
            'dec_blk_nums': [1, 1, 1],
        }

    def forward(self, x, prev_forward_feat=None):
        if x.ndim != 5:
            raise ValueError(f'Expected B,T,3,H,W, got {tuple(x.shape)}')
        b, t, c, h, w = x.size()
        if c != 3:
            raise ValueError(f'RGB model expects C=3, got {c}')

        noisy_rgb = x
        x_flat = x.reshape(-1, c, h, w)
        feats = self.feat_extract(x_flat)
        feats = feats.reshape(b, t, -1, h, w)

        forward_feats = []
        if prev_forward_feat is None:
            feat_prop = torch.zeros_like(feats[:, 0, ...])
        else:
            if prev_forward_feat.shape != feats[:, 0, ...].shape:
                raise ValueError(
                    'prev_forward_feat shape mismatch: '
                    f'{tuple(prev_forward_feat.shape)} vs {tuple(feats[:,0,...].shape)}'
                )
            feat_prop = prev_forward_feat

        for i in range(t):
            feat_prop = self.forward_net(feats[:, i, ...], feat_prop)
            forward_feats.append(feat_prop)
        next_forward_feat = feat_prop

        backward_feats = []
        feat_prop = torch.zeros_like(feats[:, 0, ...])
        for i in range(t - 1, -1, -1):
            feat_prop = self.backward_net(feats[:, i, ...], feat_prop)
            backward_feats.insert(0, feat_prop)

        outputs = []
        for i in range(t):
            f_fused = torch.cat([forward_feats[i], backward_feats[i]], dim=1)
            f_fused = self.fusion(f_fused)
            residual = self.conv_last(f_fused)
            out = noisy_rgb[:, i] + residual
            outputs.append(out)

        final_video = torch.stack(outputs, dim=1)
        return final_video, next_forward_feat
