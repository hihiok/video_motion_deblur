import torch
import torch.nn as nn
import torch.nn.functional as F


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.LeakyReLU(0.1, inplace=True)
        if deploy:
            self.reparam = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        else:
            self.b3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels))
            self.b1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(out_channels))
            self.bid = nn.BatchNorm2d(in_channels) if in_channels == out_channels else None

    def forward(self, x):
        if self.deploy:
            return self.act(self.reparam(x))
        y = self.b3(x) + self.b1(x)
        if self.bid is not None:
            y = y + self.bid(x)
        return self.act(y)


class ResidualHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 3, 3, 1, 1),
        )

    def forward(self, x):
        return self.body(x)


class NanoVSRDeblur(nn.Module):
    """1x video deblurring adaptation of NanoVSR.

    Keeps the core ideas of NanoVSR: light per-frame feature extraction,
    bidirectional additive recurrent propagation, and plain convolutions.
    PixelShuffle SR heads are removed and replaced by a 1x residual RGB head.
    """
    def __init__(self, num_feat=48, num_blocks=12, deploy=False):
        super().__init__()
        self.num_feat = num_feat
        self.num_blocks = num_blocks
        self.feat_extract = RepVGGBlock(3, num_feat, deploy=deploy)
        self.forward_net = nn.Sequential(*[RepVGGBlock(num_feat, num_feat, deploy=deploy) for _ in range(num_blocks)])
        self.backward_net = nn.Sequential(*[RepVGGBlock(num_feat, num_feat, deploy=deploy) for _ in range(num_blocks)])
        self.fusion = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.head = ResidualHead(num_feat)

    def forward(self, x, return_features=False):
        b, t, c, h, w = x.shape
        feats = self.feat_extract(x.reshape(b * t, c, h, w)).reshape(b, t, self.num_feat, h, w)

        fwd = []
        state = torch.zeros_like(feats[:, 0])
        for i in range(t):
            state = self.forward_net(feats[:, i] + state)
            fwd.append(state)

        bwd = [None] * t
        state = torch.zeros_like(feats[:, 0])
        for i in range(t - 1, -1, -1):
            state = self.backward_net(feats[:, i] + state)
            bwd[i] = state

        outs, fused_all = [], []
        for i in range(t):
            fused = self.fusion(torch.cat([fwd[i], bwd[i]], dim=1))
            residual = self.head(fused)
            out = torch.clamp(x[:, i] + residual, 0.0, 1.0)
            outs.append(out)
            fused_all.append(fused)
        out = torch.stack(outs, dim=1)
        if return_features:
            return out, torch.stack(fused_all, dim=1)
        return out
