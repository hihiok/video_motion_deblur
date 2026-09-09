#!/usr/bin/env python3
"""Train RT-Focuser-T6 on balanced GoPro, BSD and DVD clips."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rtf_t6.checkpoint import atomic_torch_save, load_rtfocuser_pretrained
from rtf_t6.complexity import compare_models
from rtf_t6.datasets import (
    BalancedMultiDomainClips,
    ValidationClips,
    build_domain_sequences,
    dataset_summary,
)
from rtf_t6.losses import VideoDeblurLoss, psnr, temporal_residual_error
from rtf_t6.model import RT_Focuser_Standard, RTFocuserT6
from rtf_t6.protocol import check_checkpoint_protocol, check_fullframe, clip_length as configured_clip_length


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--stop-after", type=int, default=None,
                        help="Pause at this microbatch without changing the configured LR schedule")
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--validate-every", type=int, default=None)
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def seed_everything(seed: int, rank: int) -> None:
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.state_dict()
        for key, value in self.model.state_dict().items():
            incoming = source[key].detach()
            if value.is_floating_point():
                value.mul_(self.decay).add_(incoming, alpha=1.0 - self.decay)
            else:
                value.copy_(incoming)


def make_loader(dataset, batch_size, workers, rank, world_size, shuffle):
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=shuffle
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        persistent_workers=False,
    ), sampler


def learning_rate(iteration: int, config: dict) -> float:
    base = float(config["lr"])
    minimum = float(config.get("min_lr", 1e-6))
    warmup = int(config.get("warmup_iters", 0))
    total = int(config["total_iters"])
    if warmup and iteration <= warmup:
        return base * iteration / warmup
    progress = (iteration - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * progress))


@torch.inference_mode()
def validate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> dict:
    model.eval()
    domain_psnr = defaultdict(list)
    domain_temporal = defaultdict(list)
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    for batch in loader:
        blur = batch["blur"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp and device.type == "cuda"):
            prediction = model(blur).clamp(0, 1)
        for item, domain in enumerate(batch["domain"]):
            domain_psnr[domain].append(float(psnr(prediction[item], gt[item]).cpu()))
            domain_temporal[domain].append(
                float(temporal_residual_error(prediction[item : item + 1], gt[item : item + 1]).cpu())
            )
        del blur, gt, prediction
    metrics = {"domains": {}}
    for domain in sorted(domain_psnr):
        metrics["domains"][domain] = {
            "psnr": float(np.mean(domain_psnr[domain])),
            "temporal_residual_l1": float(np.mean(domain_temporal[domain])),
            "clips": len(domain_psnr[domain]),
        }
    metrics["balanced_psnr"] = float(
        np.mean([values["psnr"] for values in metrics["domains"].values()])
    )
    metrics["balanced_temporal_residual_l1"] = float(
        np.mean([values["temporal_residual_l1"] for values in metrics["domains"].values()])
    )
    return metrics


def save_checkpoint(path, model, ema, optimizer, scaler, iteration, epoch, best, config):
    atomic_torch_save(
        {
            "iteration": iteration,
            "epoch": epoch,
            "model": unwrap(model).state_dict(),
            "params_ema": ema.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_balanced_psnr": best,
            "config": config,
        },
        path,
    )


def main() -> None:
    args = arguments()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.max_iters is not None:
        config["train"]["total_iters"] = args.max_iters
    if args.samples_per_epoch is not None:
        config["train"]["samples_per_epoch"] = args.samples_per_epoch
    if args.crop_size is not None:
        config["train"]["crop_size"] = args.crop_size
        config.setdefault("validation", {})["crop_size"] = args.crop_size
    if args.validate_every is not None:
        config["train"]["validate_every"] = args.validate_every
    rank, world_size, _, device = setup_distributed()
    seed = int(config.get("seed", 123))
    seed_everything(seed, rank)
    is_main = rank == 0
    train_cfg = config["train"]
    model_cfg = config.get("model", {})
    train_crop_size = int(train_cfg.get("crop_size", 256))
    validation_crop_size = int(config.get("validation", {}).get("crop_size", 256))
    spatial_mode = str(train_cfg.get("spatial_mode", "crop")).lower()
    if spatial_mode not in {"crop", "full_frame"}:
        raise ValueError(f"Unsupported train.spatial_mode: {spatial_mode}")
    if spatial_mode == "full_frame":
        check_fullframe(config)
        if train_crop_size != 0 or validation_crop_size != 0:
            raise ValueError(
                "full_frame mode requires train.crop_size=0 and validation.crop_size=0"
            )
        if int(train_cfg.get("batch_size", 1)) != 1:
            raise ValueError("Native full-frame training is locked to batch_size=1")
    output = Path(args.output or config["output"]).expanduser().resolve()
    if is_main:
        output.mkdir(parents=True, exist_ok=True)
        with open(output / "resolved_config.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

    clip_length = configured_clip_length(config)
    train_domains, train_roots = build_domain_sequences(config["datasets"], "train", clip_length)
    val_domains, val_roots = build_domain_sequences(config["datasets"], "val", clip_length)
    train_dataset = BalancedMultiDomainClips(
        train_domains,
        clip_length=clip_length,
        crop_size=train_crop_size,
        samples_per_epoch=int(train_cfg.get("samples_per_epoch", 10_000)),
        domain_weights=train_cfg.get("domain_weights"),
        seed=seed,
        augment=True,
    )
    val_dataset = ValidationClips(
        val_domains,
        clip_length=clip_length,
        crop_size=validation_crop_size,
        stride=int(config.get("validation", {}).get("stride", clip_length)),
        max_clips_per_domain=int(config.get("validation", {}).get("max_clips_per_domain", 24)),
        seed=seed + 1,
    )
    workers = int(train_cfg.get("workers", 4))
    train_loader, train_sampler = make_loader(
        train_dataset, int(train_cfg.get("batch_size", 1)), workers, rank, world_size, True
    )
    val_loader = None
    if is_main:
        val_loader, _ = make_loader(
            val_dataset, int(config.get("validation", {}).get("batch_size", 1)),
            max(1, workers // 2) if workers > 0 else 0, 0, 1, False
        )

    baseline = RT_Focuser_Standard()
    candidate = RTFocuserT6(**model_cfg)
    complexity = compare_models(baseline, candidate, height=64, width=64, clip_length=clip_length)
    if not complexity["parameters_pass"] or not complexity["compute_pass"]:
        raise RuntimeError(f"Complexity gate failed: {complexity}")
    del baseline

    init_report = None
    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        check_checkpoint_protocol(resume, config, resume=True)
        candidate.load_state_dict(resume["model"], strict=True)
    else:
        pretrained = args.pretrained or config.get("pretrained")
        if not pretrained:
            raise ValueError("An official RT-Focuser checkpoint is required for initialization")
        init_report = load_rtfocuser_pretrained(candidate, pretrained)

    model = candidate.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        betas=tuple(train_cfg.get("betas", (0.9, 0.999))),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    start_iteration = 0
    start_epoch = 0
    best = float("-inf")
    ema = EMA(model, float(train_cfg.get("ema_decay", 0.999)))
    if args.resume:
        optimizer.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume.get("scaler", {}))
        ema.model.load_state_dict(resume.get("params_ema", resume["model"]), strict=True)
        start_iteration = int(resume["iteration"])
        start_epoch = int(resume.get("epoch", 0))
        best = float(resume.get("best_balanced_psnr", best))
        del resume

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False)
    loss_cfg = config.get("loss", {})
    criterion = VideoDeblurLoss(**loss_cfg).to(device)
    total_iters = int(train_cfg["total_iters"])
    accumulation = int(train_cfg.get("gradient_accumulation", 1))
    log_every = int(train_cfg.get("log_every", 100))
    validate_every = int(train_cfg.get("validate_every", 5_000))
    save_every = int(train_cfg.get("save_every", 5_000))
    clip_grad = float(train_cfg.get("clip_grad", 1.0))
    stop_iteration = total_iters if args.stop_after is None else int(args.stop_after)
    if not start_iteration < stop_iteration <= total_iters or stop_iteration % accumulation:
        raise ValueError("Stop iteration must be a future optimizer boundary within total_iters")
    log_path = output / "train_metrics.jsonl"

    if is_main:
        startup = {
            "event": "startup",
            "time": time.time(),
            "device": str(device),
            "world_size": world_size,
            "train_roots": train_roots,
            "val_roots": val_roots,
            "train_data": dataset_summary(train_domains),
            "val_data": dataset_summary(val_domains),
            "complexity": complexity,
            "initialization": init_report,
            "spatial_mode": spatial_mode,
            "train_crop_size": train_crop_size,
            "validation_crop_size": validation_crop_size,
            "clip_length": clip_length,
            "gradient_accumulation": accumulation,
            "iteration_unit": "microbatch",
            "total_optimizer_updates": total_iters // accumulation,
            "frames_per_optimizer_update": clip_length * accumulation * world_size,
        }
        print(json.dumps(startup, indent=2))
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(startup) + "\n")

    iteration = start_iteration
    epoch = start_epoch
    optimizer.zero_grad(set_to_none=True)
    running = defaultdict(float)
    running_count = 0
    log_started = time.perf_counter()
    while iteration < stop_iteration:
        train_dataset.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            if iteration >= stop_iteration:
                break
            iteration += 1
            lr = learning_rate(iteration, train_cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            blur = batch["blur"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            if spatial_mode == "full_frame" and iteration == start_iteration + 1 and is_main:
                first_batch = {
                    "event": "full_frame_first_batch",
                    "shape": list(blur.shape),
                    "crop_applied": False,
                    "resize_applied": False,
                }
                print(json.dumps(first_batch))
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(first_batch) + "\n")
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=amp,
            ):
                prediction = model(blur)
                loss, components = criterion(prediction, gt, iteration)
                scaled_loss = loss / accumulation
            if not all(bool(torch.isfinite(value).all()) for value in components.values()):
                raise FloatingPointError(f"Non-finite loss at iteration {iteration}")
            scaler.scale(scaled_loss).backward()
            if iteration % accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if is_main:
                    ema.update(unwrap(model))

            for key, value in components.items():
                running[key] += float(value.detach())
            running_count += 1
            if is_main and iteration % log_every == 0:
                record = {
                    "event": "train",
                    "iteration": iteration,
                    "epoch": epoch,
                    "lr": lr,
                    "optimizer_updates": iteration // accumulation,
                    "seconds_per_microbatch": (time.perf_counter() - log_started) / running_count,
                    **{key: value / running_count for key, value in running.items()},
                }
                print(json.dumps(record))
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                running.clear()
                running_count = 0
                log_started = time.perf_counter()

            # Do not retain the previous full-frame graph/input during the next
            # forward or EMA validation. Cached allocator blocks remain reusable.
            del blur, gt, prediction, loss, scaled_loss, components, value

            validation_due = iteration % validate_every == 0 or iteration == stop_iteration
            if validation_due:
                if world_size > 1:
                    dist.barrier()
                if is_main:
                    metrics = validate(ema.model, val_loader, device, amp)
                    metrics.update({"event": "validation", "iteration": iteration, "epoch": epoch})
                    print(json.dumps(metrics, indent=2))
                    with open(log_path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(metrics) + "\n")
                    if metrics["balanced_psnr"] > best:
                        best = metrics["balanced_psnr"]
                        save_checkpoint(
                            output / "checkpoints" / "best_balanced_psnr.pth",
                            model, ema, optimizer, scaler, iteration, epoch, best, config,
                        )
                if world_size > 1:
                    value = torch.tensor(best if is_main else 0.0, device=device)
                    dist.broadcast(value, 0)
                    best = float(value)
                    dist.barrier()

            if is_main and (iteration % save_every == 0 or iteration == stop_iteration):
                save_checkpoint(
                    output / "checkpoints" / "latest.pth",
                    model, ema, optimizer, scaler, iteration, epoch, best, config,
                )
        epoch += 1

    if is_main:
        status = "TRAINING_COMPLETE" if iteration == total_iters else "TRAINING_PAUSED"
        print(f"{status} iteration={iteration} best_balanced_psnr={best:.4f}")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
