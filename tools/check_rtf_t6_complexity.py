#!/usr/bin/env python3
"""Compare RT-Focuser Standard and RT-Focuser-T6 parameter/compute budgets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rtf_t6.checkpoint import load_rtfocuser_pretrained
from rtf_t6.complexity import compare_models
from rtf_t6.model import RT_Focuser_Standard, RTFocuserT6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--probe-size", type=int, default=64)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    baseline = RT_Focuser_Standard()
    candidate = RTFocuserT6(**config.get("model", {}))
    report = compare_models(
        baseline, candidate, args.probe_size, args.probe_size,
        int(config["train"].get("clip_length", 6)),
    )
    scale_256 = (256 / args.probe_size) ** 2
    scale_720p = (1280 * 720) / (args.probe_size**2)
    report["baseline_gmac_256"] = report["baseline_conv_macs_per_frame"] * scale_256 / 1e9
    report["candidate_estimated_gops_256"] = report["candidate_total_estimated_ops_per_frame"] * scale_256 / 1e9
    report["baseline_gmac_720p"] = report["baseline_conv_macs_per_frame"] * scale_720p / 1e9
    report["candidate_estimated_gops_720p"] = report["candidate_total_estimated_ops_per_frame"] * scale_720p / 1e9
    if args.pretrained:
        report["initialization"] = load_rtfocuser_pretrained(candidate, args.pretrained)
    report["passed"] = bool(report["parameters_pass"] and report["compute_pass"])
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("COMPLEXITY_GATE_FAILED")
    print("COMPLEXITY_GATE_PASS")


if __name__ == "__main__":
    main()
