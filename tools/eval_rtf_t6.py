#!/usr/bin/env python3
"""Full-resolution, per-domain evaluation for RT-Focuser-T6."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rtf_t6.checkpoint import unwrap_state_dict
from rtf_t6.datasets import build_domain_sequences, read_rgb
from rtf_t6.inference import infer_spatial_tiles, owned_temporal_range, window_starts
from rtf_t6.losses import psnr, ssim
from rtf_t6.model import RT_Focuser_Standard, RTFocuserT6


class FrameVideoWrapper(nn.Module):
    def __init__(self, frame_model: nn.Module):
        super().__init__()
        self.frame_model = frame_model

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = video.shape
        output = self.frame_model(video.reshape(b * t, c, h, w))
        return output.reshape(b, t, c, h, w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--temporal-overlap", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=384)
    parser.add_argument("--tile-overlap", type=int, default=48)
    parser.add_argument("--max-sequences-per-domain", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--architecture", choices=("t6", "rtfocuser_baseline"), default="t6")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.window != 6:
        raise ValueError("Evaluation protocol is locked to T=6")
    domains, roots = build_domain_sequences(config["datasets"], args.split, args.window)
    device = torch.device(args.device)
    payload = torch.load(Path(args.checkpoint).expanduser().resolve(), map_location="cpu")
    if args.architecture == "t6":
        model = RTFocuserT6(**config.get("model", {}))
        model.load_state_dict(unwrap_state_dict(payload), strict=True)
    else:
        frame_model = RT_Focuser_Standard()
        frame_model.load_state_dict(unwrap_state_dict(payload), strict=True)
        model = FrameVideoWrapper(frame_model)
    model = model.to(device).eval()
    dtype = torch.float16 if args.amp and device.type == "cuda" else None
    report = {"architecture": args.architecture, "split": args.split, "roots": roots, "domains": {}}
    all_domain_psnr = []
    all_domain_ssim = []
    all_domain_temporal = []
    with torch.inference_mode():
        for domain, sequences in sorted(domains.items()):
            if args.max_sequences_per_domain > 0:
                sequences = sequences[: args.max_sequences_per_domain]
            psnr_values = []
            ssim_values = []
            baseline_psnr_values = []
            temporal_errors = []
            sequence_reports = []
            for sequence in sequences:
                starts = window_starts(sequence.length, args.window, args.temporal_overlap)
                sequence_psnr = []
                previous_prediction = None
                previous_gt = None
                frame_count = 0
                for window_index, start in enumerate(starts):
                    end = min(start + args.window, sequence.length)
                    blur = torch.stack([read_rgb(path) for path in sequence.blur[start:end]])
                    gt = torch.stack([read_rgb(path) for path in sequence.gt[start:end]])
                    prediction = infer_spatial_tiles(
                        model, blur.unsqueeze(0).to(device), args.tile_size, args.tile_overlap, dtype
                    )[0].clamp(0, 1).cpu()
                    own_start, own_end = owned_temporal_range(
                        starts, window_index, end, args.window, sequence.length
                    )
                    for global_index in range(own_start, own_end):
                        local = global_index - start
                        current_prediction = prediction[local : local + 1]
                        current_gt = gt[local : local + 1]
                        score_psnr = float(psnr(current_prediction, current_gt))
                        score_ssim = float(ssim(current_prediction, current_gt))
                        psnr_values.append(score_psnr)
                        ssim_values.append(score_ssim)
                        sequence_psnr.append(score_psnr)
                        baseline_psnr_values.append(float(psnr(blur[local : local + 1], current_gt)))
                        if previous_prediction is not None:
                            pred_delta = current_prediction - previous_prediction
                            gt_delta = current_gt - previous_gt
                            temporal_errors.append(float((pred_delta - gt_delta).abs().mean()))
                        previous_prediction = current_prediction
                        previous_gt = current_gt
                        frame_count += 1
                if frame_count != sequence.length:
                    raise RuntimeError(
                        f"Frame ownership mismatch for {domain}/{sequence.name}: {frame_count}/{sequence.length}"
                    )
                sequence_reports.append(
                    {"name": sequence.name, "frames": frame_count, "psnr": float(np.mean(sequence_psnr))}
                )
                print(f"{domain}/{sequence.name}: {sequence_reports[-1]['psnr']:.4f} dB")
            domain_report = {
                "sequences": len(sequences),
                "frames": len(psnr_values),
                "psnr": float(np.mean(psnr_values)),
                "ssim": float(np.mean(ssim_values)),
                "input_psnr": float(np.mean(baseline_psnr_values)),
                "temporal_residual_l1": float(np.mean(temporal_errors)),
                "sequence_metrics": sequence_reports,
            }
            report["domains"][domain] = domain_report
            all_domain_psnr.append(domain_report["psnr"])
            all_domain_ssim.append(domain_report["ssim"])
            all_domain_temporal.append(domain_report["temporal_residual_l1"])
    report["balanced_psnr"] = float(np.mean(all_domain_psnr))
    report["balanced_ssim"] = float(np.mean(all_domain_ssim))
    report["balanced_temporal_residual_l1"] = float(np.mean(all_domain_temporal))
    report["passed"] = True
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("EVALUATION_PASS")


if __name__ == "__main__":
    main()
