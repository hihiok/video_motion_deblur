#!/usr/bin/env python3
"""Inference-only compatibility loader for official DSTNet+ (TPAMI 2025).

This loader keeps the released DSTNetPlus_Final architecture/checkpoint intact,
while avoiding BasicSR package eager imports and the optional CuPy dependency.
The official per-pixel dynamic depthwise convolution is evaluated with an
algebraically equivalent PyTorch unfold implementation.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Registry:
    def register(self, obj=None, **kwargs):
        def deco(cls):
            return cls
        return deco if obj is None else obj


def _namespace(name: str, path: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)]
    module.__file__ = str(path / "__init__.py")
    sys.modules[name] = module
    return module


def make_layer(block, num_blocks: int, **kwargs):
    return nn.Sequential(*(block(**kwargs) for _ in range(num_blocks)))


class ResidualBlockNoBN(nn.Module):
    """Exact state-dict-compatible block used by DSTNet+ propagation."""
    def __init__(self, num_feat=64, pytorch_init=False, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return x + self.conv2(self.lrelu(self.conv1(x)))


class ResBlock(nn.Module):
    """State-dict-compatible copy of basicsr.archs.blocks.ResBlock."""
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, groups=1):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size, stride=1, padding=padding,
            dilation=dilation, groups=groups
        )
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.res_translate = None
        if inplanes != planes or stride != 1:
            self.res_translate = nn.Conv2d(inplanes, planes, 1, stride=stride)

    def forward(self, x):
        residual = x if self.res_translate is None else self.res_translate(x)
        return residual + self.conv2(self.relu(self.conv1(x)))


def dynamic_depthwise_unfold(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    stride=1,
    padding=0,
    dilation=1,
    group_chunk: int = 8,
) -> torch.Tensor:
    """Equivalent implementation of the released CuPy dynamic DW convolution.

    input_tensor: B,C,H,W
    weight: B,G,K,K,Hout,Wout where G == C for DSTNet+ (group_channels=1)
    """
    if input_tensor.ndim != 4 or weight.ndim != 6:
        raise ValueError(f"Unexpected dynamic-conv shapes: {input_tensor.shape}, {weight.shape}")
    stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
    padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
    dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)

    b, c, _, _ = input_tensor.shape
    wb, groups, kh, kw, oh, ow = weight.shape
    if wb != b or c % groups:
        raise ValueError(f"Incompatible dynamic-conv shapes: {input_tensor.shape}, {weight.shape}")
    group_channels = c // groups
    out = input_tensor.new_empty((b, c, oh, ow))
    k2 = kh * kw

    for gs in range(0, groups, group_chunk):
        ge = min(groups, gs + group_chunk)
        cs, ce = gs * group_channels, ge * group_channels
        patches = F.unfold(
            input_tensor[:, cs:ce], kernel_size=(kh, kw),
            dilation=dilation, padding=padding, stride=stride
        ).view(b, ce - cs, k2, oh, ow)
        kernel = weight[:, gs:ge].reshape(b, ge - gs, k2, oh, ow)
        kernel = kernel.repeat_interleave(group_channels, dim=1)
        out[:, cs:ce] = (patches * kernel).sum(dim=2)
    return out


class IDynamicDWConv(nn.Module):
    """State-dict-compatible DSTNet+ progressive dynamic convolution."""
    def __init__(self, channels, kernel_size, group_channels, n_blocks, conv_group):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        self.group_channels = group_channels
        self.groups = channels // group_channels
        self.Block1 = nn.Sequential(*[
            ResBlock(channels, channels, kernel_size=kernel_size, stride=1, groups=conv_group)
            for _ in range(n_blocks)
        ])
        self.Block2 = nn.Sequential(*[
            ResBlock(channels, channels, kernel_size=kernel_size, stride=1, groups=conv_group)
            for _ in range(n_blocks)
        ])
        self.Block3 = nn.Sequential(*[
            ResBlock(channels, channels, kernel_size=kernel_size, stride=1, groups=conv_group)
            for _ in range(n_blocks)
        ])
        self.tokernel = nn.Conv2d(channels * 3, kernel_size**2 * self.groups, 1, 1, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x1 = self.Block1(F.adaptive_max_pool2d(x, output_size=(h // 2, w // 2)))
        x2 = self.Block2(F.adaptive_max_pool2d(x1, output_size=(h // 4, w // 4)))
        x3 = self.Block3(F.adaptive_max_pool2d(x2, output_size=(h // 8, w // 8)))
        x1 = F.interpolate(x1, size=(h, w), mode="bilinear", align_corners=False)
        x2 = F.interpolate(x2, size=(h, w), mode="bilinear", align_corners=False)
        x3 = F.interpolate(x3, size=(h, w), mode="bilinear", align_corners=False)
        weight = self.tokernel(torch.cat([x1, x2, x3], dim=1))
        weight = weight.view(b, self.groups, self.kernel_size, self.kernel_size, h, w)
        return dynamic_depthwise_unfold(
            x, weight, stride=1, padding=(self.kernel_size - 1) // 2
        )


def load_dstnetplus_base(repo: str | Path):
    repo = Path(repo).resolve()
    basicsr_root = repo / "basicsr"
    arch_path = basicsr_root / "archs" / "dstnetplus_deblur_arch.py"
    if not arch_path.is_file():
        raise FileNotFoundError(f"Official DSTNet+ architecture not found: {arch_path}")

    for name in list(sys.modules):
        if name == "basicsr" or name.startswith("basicsr."):
            del sys.modules[name]

    _namespace("basicsr", basicsr_root)
    _namespace("basicsr.archs", basicsr_root / "archs")

    utils = _namespace("basicsr.utils", basicsr_root / "utils")
    registry = types.ModuleType("basicsr.utils.registry")
    registry.ARCH_REGISTRY = _Registry()
    sys.modules["basicsr.utils.registry"] = registry
    utils.registry = registry

    arch_util = types.ModuleType("basicsr.archs.arch_util")
    arch_util.ResidualBlockNoBN = ResidualBlockNoBN
    arch_util.make_layer = make_layer
    sys.modules["basicsr.archs.arch_util"] = arch_util

    prog = types.ModuleType("basicsr.archs.prog_dynconv")
    prog.IDynamicDWConv = IDynamicDWConv
    sys.modules["basicsr.archs.prog_dynconv"] = prog

    name = "basicsr.archs.dstnetplus_deblur_arch"
    spec = importlib.util.spec_from_file_location(name, arch_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load DSTNet+ architecture from {arch_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module.DSTNetPlus_Final, "pytorch_unfold"
