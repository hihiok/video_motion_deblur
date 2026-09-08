#!/usr/bin/env python3
"""Fail-fast audit for GoPro/BSD/DVD paired-frame layouts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rtf_t6.datasets import build_domain_sequences, dataset_summary, read_rgb


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--samples-per-domain", type=int, default=12)
    return parser.parse_args()


def fingerprint(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def main():
    args = arguments()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    clip_length = int(config["train"].get("clip_length", 6))
    train, train_roots = build_domain_sequences(config["datasets"], "train", clip_length)
    val, val_roots = build_domain_sequences(config["datasets"], "val", clip_length)
    report = {
        "passed": True,
        "clip_length": clip_length,
        "train_roots": train_roots,
        "val_roots": val_roots,
        "train": dataset_summary(train),
        "val": dataset_summary(val),
        "domains": {},
    }
    errors = []
    for domain in sorted(train):
        train_pairs = {
            (str(blur.resolve()), str(gt.resolve()))
            for sequence in train[domain]
            for blur, gt in zip(sequence.blur, sequence.gt)
        }
        val_pairs = {
            (str(blur.resolve()), str(gt.resolve()))
            for sequence in val[domain]
            for blur, gt in zip(sequence.blur, sequence.gt)
        }
        overlap = train_pairs & val_pairs
        if overlap:
            errors.append(f"{domain}: {len(overlap)} exact train/val path pairs overlap")
        train_logical = {
            (sequence.name, blur.name, gt.name)
            for sequence in train[domain]
            for blur, gt in zip(sequence.blur, sequence.gt)
        }
        val_logical = {
            (sequence.name, blur.name, gt.name)
            for sequence in val[domain]
            for blur, gt in zip(sequence.blur, sequence.gt)
        }
        logical_overlap = train_logical & val_logical
        if logical_overlap:
            errors.append(f"{domain}: {len(logical_overlap)} logical train/val frame IDs overlap")
        sample_pairs = []
        per_split_limit = max(args.samples_per_domain // 2, 1)
        for split_sequences in (train[domain], val[domain]):
            split_pairs = []
            for sequence in split_sequences:
                for index in sorted({0, sequence.length // 2, sequence.length - 1}):
                    split_pairs.append((sequence, index))
            sample_pairs.extend(split_pairs[:per_split_limit])
        blur_gt_mse = []
        checked = []
        for sequence, index in sample_pairs:
            blur = read_rgb(sequence.blur[index])
            gt = read_rgb(sequence.gt[index])
            if blur.shape != gt.shape:
                errors.append(
                    f"{domain}/{sequence.name}/{index}: shape {tuple(blur.shape)} != {tuple(gt.shape)}"
                )
                continue
            self_mse = torch.mean((gt - gt).square()).item()
            if self_mse != 0.0:
                errors.append(f"{domain}/{sequence.name}/{index}: GT-vs-GT sanity failed")
            mse = torch.mean((blur - gt).square()).item()
            blur_gt_mse.append(mse)
            checked.append(
                {
                    "split": sequence.split,
                    "sequence": sequence.name,
                    "frame": sequence.blur[index].name,
                    "shape": list(blur.shape),
                    "blur_gt_mse": mse,
                    "blur_sha1": fingerprint(sequence.blur[index]),
                    "gt_sha1": fingerprint(sequence.gt[index]),
                }
            )
        if blur_gt_mse and max(blur_gt_mse) <= 1e-12:
            errors.append(f"{domain}: all sampled blur frames are byte/pixel-identical to GT")
        report["domains"][domain] = {
            "train_val_pair_overlap": len(overlap),
            "train_val_logical_id_overlap": len(logical_overlap),
            "sample_blur_gt_mse_min": min(blur_gt_mse) if blur_gt_mse else None,
            "sample_blur_gt_mse_max": max(blur_gt_mse) if blur_gt_mse else None,
            "checked": checked,
        }
    report["errors"] = errors
    report["passed"] = not errors
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("DATA_AUDIT_FAILED")
    print("DATA_AUDIT_PASS")


if __name__ == "__main__":
    main()
