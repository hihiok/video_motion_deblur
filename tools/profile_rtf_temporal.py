#!/usr/bin/env python3
"""Exact Conv/Linear/SN MAC count at requested native size; pointwise ops excluded."""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtf_t6.complexity import count_conv_linear_macs, count_parameters
from rtf_temporal.model import TemporalRTFocuser


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output')
    args = p.parse_args()
    torch.set_num_threads(2)
    model = TemporalRTFocuser(activation_checkpointing=False).to(args.device).eval()
    sample = torch.zeros(1, 2, 3, args.height, args.width, device=args.device)
    total_macs = count_conv_linear_macs(model, sample) // 2
    # Same padded size as candidate, including 1080 -> 1088 where applicable.
    h, w = args.height + (-args.height) % 16, args.width + (-args.width) % 16
    base_macs = count_conv_linear_macs(model.backbone, torch.zeros(1, 3, h, w, device=args.device))
    report = dict(height=args.height, width=args.width, baseline_params=count_parameters(model.backbone),
                  temporal_params=count_parameters(model.temporal), total_params=count_parameters(model),
                  baseline_conv_gmac=base_macs/1e9, total_conv_gmac=total_macs/1e9,
                  added_conv_gmac=(total_macs-base_macs)/1e9,
                  total_conv_gflops_2_per_mac=2*total_macs/1e9,
                  conv_compute_increase_percent=100*(total_macs/base_macs-1),
                  excludes='BN, activation, interpolation, pooling, gate elementwise ops, memory traffic; not latency')
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + '\n')


if __name__ == '__main__':
    main()
