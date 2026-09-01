#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from generic_restoration.benchmark_utils import git_commit, list_frames, sha256_file, write_json
from generic_restoration.planning import padded_target_size, smallest_8n1_geq


def prepare_condition(
    paths: list[Path], scale: float, dtype: torch.dtype
) -> tuple[torch.Tensor, dict[str, int]]:
    with Image.open(paths[0]) as first:
        width, height = first.size
    target_w, target_h, padded_w, padded_h = padded_target_size(width, height, scale)
    frames = []
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if (target_w, target_h) != rgb.size:
                rgb = rgb.resize((target_w, target_h), Image.Resampling.BICUBIC)
            array = np.asarray(rgb, dtype=np.uint8)
        pad_h = padded_h - target_h
        pad_w = padded_w - target_w
        if pad_h or pad_w:
            array = np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        frames.append(tensor.to(dtype=dtype))
        if index % 25 == 0:
            print(f"FlashVSR input: {index + 1}/{len(paths)}", flush=True)

    required = smallest_8n1_geq(len(frames) + 4)
    frames.extend([frames[-1]] * (required - len(frames)))
    condition = torch.stack(frames, dim=1).unsqueeze(0)
    return condition, {
        "source_width": width,
        "source_height": height,
        "target_width": target_w,
        "target_height": target_h,
        "padded_width": padded_w,
        "padded_height": padded_h,
        "model_input_frames": required,
    }


def init_pipeline(repo: Path, weights: Path):
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "examples" / "WanVSR"))
    from diffsynth import FlashVSRTinyLongPipeline, ModelManager  # type: ignore
    from utils.TCDecoder import build_tcdecoder  # type: ignore
    from utils.utils import Causal_LQ4x_Proj  # type: ignore

    diffusion = weights / "diffusion_pytorch_model_streaming_dmd.safetensors"
    lq_projection = weights / "LQ_proj_in.ckpt"
    decoder = weights / "TCDecoder.ckpt"
    for path in (diffusion, lq_projection, decoder):
        if not path.is_file():
            raise FileNotFoundError(path)

    manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    manager.load_models([str(diffusion)])
    pipe = FlashVSRTinyLongPipeline.from_model_manager(manager, device="cuda")
    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(
        in_dim=3, out_dim=1536, layer_num=1
    ).to("cuda", dtype=torch.bfloat16)
    pipe.denoising_model().LQ_proj_in.load_state_dict(
        torch.load(lq_projection, map_location="cpu"), strict=True
    )
    pipe.denoising_model().LQ_proj_in.to("cuda")

    pipe.TCDecoder = build_tcdecoder(
        new_channels=[512, 256, 128, 128], new_latent_channels=16 + 768
    )
    decoder_loading = pipe.TCDecoder.load_state_dict(
        torch.load(decoder, map_location="cpu"), strict=False
    )
    pipe.to("cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit", "vae"])
    return pipe, {
        "tcdecoder_missing_keys": list(decoder_loading.missing_keys),
        "tcdecoder_unexpected_keys": list(decoder_loading.unexpected_keys),
    }


def save_frames(
    video: torch.Tensor,
    output: Path,
    count: int,
    geometry: dict[str, int],
    normalize_1x: bool,
) -> None:
    if video.ndim != 4:
        raise RuntimeError(f"Unexpected FlashVSR output shape: {tuple(video.shape)}")
    if video.shape[1] < count:
        raise RuntimeError(f"FlashVSR returned only {video.shape[1]} frames for {count} inputs")
    video = video[:, :count, : geometry["target_height"], : geometry["target_width"]]
    if normalize_1x and (
        geometry["target_width"] != geometry["source_width"]
        or geometry["target_height"] != geometry["source_height"]
    ):
        flat = video.permute(1, 0, 2, 3)
        flat = F.interpolate(
            flat,
            size=(geometry["source_height"], geometry["source_width"]),
            mode="area",
        )
        video = flat.permute(1, 0, 2, 3)
    output.mkdir(parents=True, exist_ok=True)
    for old in output.glob("*.png"):
        old.unlink()
    for index in range(count):
        frame = video[:, index].float().add(1).mul(127.5).clamp(0, 255)
        array = frame.round().byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(array, mode="RGB").save(output / f"{index:08d}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="FlashVSR v1.1 long-video benchmark adapter.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input-frames", required=True)
    parser.add_argument("--output-frames", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sparse-ratio", type=float, default=2.0)
    parser.add_argument("--kv-ratio", type=float, default=3.0)
    parser.add_argument("--local-range", type=int, default=11)
    parser.add_argument("--keep-native-size", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    repo = Path(args.repo).resolve()
    weights = Path(args.weights).resolve()
    paths = list_frames(args.input_frames)
    condition, geometry = prepare_condition(paths, args.scale, torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    pipe, loading = init_pipeline(repo, weights)
    topk_ratio = (
        args.sparse_ratio
        * 768
        * 1280
        / (geometry["padded_height"] * geometry["padded_width"])
    )
    with torch.inference_mode():
        video = pipe(
            prompt="",
            negative_prompt="",
            cfg_scale=1.0,
            num_inference_steps=1,
            seed=args.seed,
            LQ_video=condition,
            num_frames=geometry["model_input_frames"],
            height=geometry["padded_height"],
            width=geometry["padded_width"],
            is_full_block=False,
            if_buffer=True,
            topk_ratio=topk_ratio,
            kv_ratio=args.kv_ratio,
            local_range=args.local_range,
            color_fix=True,
        )
    save_frames(
        video,
        Path(args.output_frames).resolve(),
        len(paths),
        geometry,
        normalize_1x=not args.keep_native_size,
    )
    elapsed = time.perf_counter() - started
    metadata = {
        "model": "FlashVSR v1.1 Tiny Long",
        "official_repo": "https://github.com/OpenImagingLab/FlashVSR",
        "official_commit": git_commit(repo),
        "weights_dir": str(weights),
        "weight_sha256": {
            path.name: sha256_file(path)
            for path in sorted(weights.iterdir())
            if path.is_file() and path.suffix in {".ckpt", ".pth", ".safetensors"}
        },
        "loading": loading,
        "input_frame_count": len(paths),
        "output_frame_count": len(list_frames(args.output_frames)),
        "geometry": geometry,
        "scale": args.scale,
        "mode": (
            "experimental direct 1x restoration"
            if args.scale == 1.0
            else "official 4x-oriented inference normalized to source size"
        ),
        "seed": args.seed,
        "topk_ratio": topk_ratio,
        "kv_ratio": args.kv_ratio,
        "local_range": args.local_range,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
    }
    write_json(args.metadata, metadata)
    print(args.metadata)


if __name__ == "__main__":
    main()
