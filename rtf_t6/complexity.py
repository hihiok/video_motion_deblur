"""Dependency-free parameter and MAC accounting for the complexity gate."""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def count_conv_linear_macs(model: nn.Module, sample: torch.Tensor) -> int:
    macs = 0
    handles = []

    def hook(module: nn.Module, inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            batch = output.shape[0]
            out_h, out_w = output.shape[-2:]
            kernel_h, kernel_w = module.kernel_size
            per_output = (module.in_channels // module.groups) * kernel_h * kernel_w
            macs += batch * module.out_channels * out_h * out_w * per_output
        elif isinstance(module, nn.Linear):
            macs += output.numel() * module.in_features
        elif module.__class__.__name__ == "SN_Module":
            # SN_Module calls functional conv2d, so it is not covered by the
            # nn.Conv2d hook above.
            macs += output.numel() * int(module.kernel_size) ** 2

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) or module.__class__.__name__ == "SN_Module":
            handles.append(module.register_forward_hook(hook))
    training = model.training
    model.eval()
    with torch.inference_mode():
        model(sample)
    model.train(training)
    for handle in handles:
        handle.remove()
    return int(macs)


def estimate_temporal_ops_per_frame(
    height: int,
    width: int,
    channels: int = 160,
    approximate_ops_per_element: int = 12,
) -> int:
    """Conservative estimate for shift blending plus H/8 bidirectional gating."""
    h = (height + 7) // 8
    w = (width + 7) // 8
    bidirectional = h * w * channels * approximate_ops_per_element
    shift_blends = 3 * (
        height * width * 16
        + ((height + 1) // 2) * ((width + 1) // 2) * 32
        + ((height + 3) // 4) * ((width + 3) // 4) * 128
        + h * w * 160
    )
    return bidirectional + shift_blends


def compare_models(
    baseline: nn.Module,
    candidate: nn.Module,
    height: int = 64,
    width: int = 64,
    clip_length: int = 6,
) -> dict[str, float | int | bool]:
    baseline_sample = torch.zeros(1, 3, height, width)
    candidate_sample = torch.zeros(1, clip_length, 3, height, width)
    baseline_macs = count_conv_linear_macs(baseline, baseline_sample)
    candidate_clip_macs = count_conv_linear_macs(candidate, candidate_sample)
    candidate_frame_macs = candidate_clip_macs / clip_length
    temporal_ops = estimate_temporal_ops_per_frame(height, width)
    baseline_params = count_parameters(baseline)
    candidate_params = count_parameters(candidate)
    return {
        "height": height,
        "width": width,
        "clip_length": clip_length,
        "baseline_parameters": baseline_params,
        "candidate_parameters": candidate_params,
        "parameter_ratio": candidate_params / baseline_params,
        "baseline_conv_macs_per_frame": baseline_macs,
        "candidate_conv_macs_per_frame": candidate_frame_macs,
        "estimated_temporal_ops_per_frame": temporal_ops,
        "candidate_total_estimated_ops_per_frame": candidate_frame_macs + temporal_ops,
        "compute_ratio": (candidate_frame_macs + temporal_ops) / baseline_macs,
        "parameters_pass": candidate_params <= baseline_params,
        "compute_pass": candidate_frame_macs + temporal_ops <= baseline_macs,
    }
