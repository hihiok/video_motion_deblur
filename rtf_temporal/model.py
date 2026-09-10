"""Explicit recurrent state, zero-initialized residual, frozen BN statistics.

Backbone implementation and MIT attribution: rtf_t6/model.py and
third_party/RT_FOCUSER_LICENSE. No blocks are removed from Standard.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from rtf_t6.model import RT_Focuser_Standard, _resize
from rtf_t6.checkpoint import unwrap_state_dict


class GatedHistory(nn.Module):
    def __init__(self, channels=128, hidden=32):
        super().__init__()
        self.compress = nn.Conv2d(channels, hidden, 1)
        self.context = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU())
        self.gate = nn.Conv2d(hidden, hidden, 1)
        self.candidate = nn.Conv2d(hidden, hidden, 1)
        self.project = nn.Conv2d(hidden, channels, 1)
        nn.init.constant_(self.gate.bias, -2.0)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, current, history=None, reset=None):
        compact = self.compress(current)
        if history is None:
            history = torch.zeros_like(compact)
            valid = compact.new_zeros((compact.shape[0], 1, 1, 1))
        else:
            if history.shape != compact.shape:
                raise ValueError('State shape mismatch; reset state on resolution changes')
            valid = compact.new_ones((compact.shape[0], 1, 1, 1))
        if reset is not None:
            valid = valid * (~reset.bool()).reshape(-1, 1, 1, 1)
        # Reset is applied BEFORE any learned operation, so old content cannot leak.
        history = history * valid
        context = self.context(torch.cat((compact, history, (compact - history).abs()), 1))
        delta = valid * torch.sigmoid(self.gate(context)) * torch.tanh(self.candidate(context))
        state = compact + delta
        return current + valid * self.project(delta), state


class TemporalRTFocuser(nn.Module):
    def __init__(self, hidden=32, activation_checkpointing=True):
        super().__init__()
        self.backbone = RT_Focuser_Standard()
        self.temporal = GatedHistory(128, hidden)
        self.activation_checkpointing = activation_checkpointing
        self.train()

    def train(self, mode=True):
        super().train(mode)
        for layer in self.backbone.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()
        return self

    def load_official(self, path):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        state = unwrap_state_dict(payload)
        # No partial loading, depth remapping, or loading old T3/T6 temporal weights.
        self.backbone.load_state_dict(state, strict=True)

    def set_stage(self, adapter_only):
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not adapter_only)
        for parameter in self.temporal.parameters():
            parameter.requires_grad_(True)
        self.train()

    def call(self, fn, *args):
        if self.training and self.activation_checkpointing and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def step(self, image, state=None, reset=None, spatial_only=False):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError('Expected B,3,H,W RGB [0,1]')
        h, w = image.shape[-2:]
        image = F.pad(image, (0, (-w) % 16, 0, (-h) % 16), mode='replicate')
        b = self.backbone
        x1 = self.call(lambda x: b.encoder1(b.stem(x)), image)
        x2 = self.call(b.encoder2, b.Maxpool(x1))
        x3 = self.call(b.encoder3, b.Maxpool(x2))
        x4 = self.call(b.encoder4, b.Maxpool(x3))
        x5 = self.call(b.encoder5, b.Maxpool(x4))

        def fuse(module, size, a, c, d, e):
            return module([_resize(v, size) for v in (a, c, d, e)])

        # Bind module and size in each closure, including during recomputation.
        f4 = self.call(lambda *xs: fuse(b.Msf_8, x4.shape[-2:], *xs), x1, x2, x3, x4)
        d5 = self.call(lambda f, x, rgb: b.Up_conv5(
            torch.cat((f, b.Up5(x)), 1), _resize(rgb, f.shape[-2:])), f4, x5, image)
        f3 = self.call(lambda *xs: fuse(b.Msf_4, x3.shape[-2:], *xs), x1, x2, x3, x4)
        d4 = self.call(lambda f, x, rgb: b.Up_conv4(
            torch.cat((f, b.Up4(x)), 1), _resize(rgb, f.shape[-2:])), f3, d5, image)
        if not spatial_only:
            d4, state = self.temporal(d4, state, reset)
        else:
            state = None
        f2 = self.call(lambda *xs: fuse(b.Msf_2, x2.shape[-2:], *xs), x1, x2, x3, x4)
        d3 = self.call(lambda f, x, rgb: b.Up_conv3(
            torch.cat((f, b.Up3(x)), 1), _resize(rgb, f.shape[-2:])), f2, d4, image)
        f1 = self.call(lambda *xs: fuse(b.Msf_1, x1.shape[-2:], *xs), x1, x2, x3, x4)
        output = self.call(lambda f, x, rgb: torch.sigmoid(b.Conv_1x1(
            b.Up_conv2(torch.cat((f, b.Up2(x)), 1), rgb)) + rgb), f1, d3, image)
        return output[..., :h, :w], state

    def forward(self, video, resets=None, spatial_only=False):
        if video.ndim != 5:
            raise ValueError('Expected B,T,3,H,W')
        state, outputs = None, []
        for t in range(video.shape[1]):
            pred, state = self.step(video[:, t], state,
                                    None if resets is None else resets[:, t], spatial_only)
            outputs.append(pred)
        return torch.stack(outputs, 1)
