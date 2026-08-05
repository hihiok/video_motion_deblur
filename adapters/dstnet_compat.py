#!/usr/bin/env python3
"""Compatibility loader for DSTNet inference on newer PyTorch environments.

The official DSTNet package eagerly imports all BasicSR modules and its
IDynamicDWConv uses a CuPy runtime-compiled CUDA kernel. For inference this
module:

1. loads only the architecture modules required by ``deblur_arch.py``;
2. exposes the minimal ``basicsr.utils`` API required by architecture files;
3. provides an import-only shim for the unused mmcv ConvModule symbol;
4. uses the official CuPy operator when available;
5. otherwise replaces only ``_idynamic_cuda`` with an equivalent PyTorch
   unfold implementation.

The model architecture and checkpoint parameters are unchanged.
"""
from __future__ import annotations

import importlib
import logging
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


def _namespace(name: str, path: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(path / "__init__.py")
    module.__package__ = name
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def _get_root_logger(log_file=None, log_level=logging.INFO, **kwargs):
    """Minimal inference-only replacement for BasicSR's logger factory."""
    logger = logging.getLogger("basicsr")
    logger.setLevel(log_level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(handler)
    return logger


def _install_mmcv_import_shim() -> None:
    try:
        import mmcv.cnn  # noqa: F401
        return
    except Exception:
        pass

    mmcv = types.ModuleType("mmcv")
    cnn = types.ModuleType("mmcv.cnn")

    class ConvModule(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise RuntimeError(
                "DSTNet attempted to instantiate mmcv.cnn.ConvModule. "
                "The official kpn_pixel.py only imports this symbol and does "
                "not use it in the released Deblur architecture."
            )

    cnn.ConvModule = ConvModule
    mmcv.cnn = cnn
    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.cnn"] = cnn


def _install_cupy_import_shim_if_needed() -> bool:
    """Return True when a fake CuPy module had to be installed."""
    try:
        import cupy  # noqa: F401
        return False
    except Exception:
        pass

    cupy = types.ModuleType("cupy")

    # einops probes optional backends with isinstance(tensor, cupy.ndarray).
    # A dummy type keeps that probe valid while ensuring Torch tensors are not
    # mistaken for CuPy arrays.
    class _FakeCupyArray:
        pass

    class _Util:
        @staticmethod
        def memoize(*args, **kwargs):
            def decorator(function):
                return function
            return decorator

    class _Cuda:
        @staticmethod
        def compile_with_cache(*args, **kwargs):
            raise RuntimeError("CuPy CUDA kernel requested while CuPy is unavailable")

    cupy.ndarray = _FakeCupyArray
    cupy._util = _Util()
    cupy.cuda = _Cuda()
    sys.modules["cupy"] = cupy
    return True


def idynamic_unfold(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    group_chunk: int = 8,
) -> torch.Tensor:
    """Inference-equivalent grouped per-pixel dynamic depthwise convolution."""
    if input_tensor.ndim != 4 or weight.ndim != 6:
        raise ValueError(
            f"Expected input BCHW and weight BGKKHW, got {input_tensor.shape}, {weight.shape}"
        )

    stride_pair = (stride, stride) if isinstance(stride, int) else tuple(stride)
    padding_pair = (padding, padding) if isinstance(padding, int) else tuple(padding)
    dilation_pair = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)

    batch, channels, _, _ = input_tensor.shape
    wb, groups, kernel_h, kernel_w, out_h, out_w = weight.shape
    if wb != batch or channels % groups:
        raise ValueError(
            f"Incompatible dynamic-conv shapes: input={input_tensor.shape}, weight={weight.shape}"
        )
    group_channels = channels // groups
    kernel_elems = kernel_h * kernel_w
    output = input_tensor.new_empty((batch, channels, out_h, out_w))

    for group_start in range(0, groups, group_chunk):
        group_end = min(group_start + group_chunk, groups)
        channel_start = group_start * group_channels
        channel_end = group_end * group_channels
        x_part = input_tensor[:, channel_start:channel_end]
        patches = F.unfold(
            x_part,
            kernel_size=(kernel_h, kernel_w),
            dilation=dilation_pair,
            padding=padding_pair,
            stride=stride_pair,
        )
        patches = patches.view(
            batch,
            channel_end - channel_start,
            kernel_elems,
            out_h,
            out_w,
        )
        kernel = weight[:, group_start:group_end].reshape(
            batch, group_end - group_start, kernel_elems, out_h, out_w
        )
        kernel = kernel.repeat_interleave(group_channels, dim=1)
        output[:, channel_start:channel_end] = (patches * kernel).sum(dim=2)

    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)
    return output


def load_dstnet_deblur(repo: str | Path):
    repo = Path(repo).resolve()
    basicsr_root = repo / "basicsr"
    if not (basicsr_root / "archs" / "deblur_arch.py").is_file():
        raise FileNotFoundError(f"DSTNet architecture not found below {repo}")

    for name in list(sys.modules):
        if name == "basicsr" or name.startswith("basicsr."):
            del sys.modules[name]
    _namespace("basicsr", basicsr_root)
    _namespace("basicsr.archs", basicsr_root / "archs")
    utils_module = _namespace("basicsr.utils", basicsr_root / "utils")
    utils_module.get_root_logger = _get_root_logger

    _install_mmcv_import_shim()
    used_cupy_shim = _install_cupy_import_shim_if_needed()

    deblur_module = importlib.import_module("basicsr.archs.deblur_arch")
    kpn_module = importlib.import_module("basicsr.archs.kpn_pixel")

    if used_cupy_shim:
        kpn_module._idynamic_cuda = idynamic_unfold
        backend = "pytorch_unfold"
    else:
        backend = "official_cupy"

    return deblur_module.Deblur, backend
