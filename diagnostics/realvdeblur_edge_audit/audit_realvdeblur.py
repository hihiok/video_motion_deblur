#!/usr/bin/env python3
"""Read-only RealVDeblur edge audit. No training, source edits, or output replacement.

Targets the official 63a2eb0 revision. CPU audit is always available. GPU modes
refuse changed model/loader code. Run each mode in a separate process.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import types

import numpy as np
from PIL import Image, ImageDraw

PIN = "63a2eb0ed4a22cb76b796a65d5d3052616352573"
EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(argv, check=True):
    p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and p.returncode:
        raise RuntimeError(f"Command failed: {argv!r}\n{p.stderr}")
    return p


def frames(path):
    p = Path(path)
    return sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() in EXT) if p.is_dir() else []


def rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).copy()


def metrics(a, b):
    if a.shape != b.shape:
        return {"comparable": False, "reason": "shape_mismatch", "a": list(a.shape), "b": list(b.shape)}
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(d * d))
    return {"comparable": True, "mae_8bit": float(np.abs(d).mean()),
            "max_abs_8bit": float(np.abs(d).max()), "p99_abs_8bit": float(np.quantile(np.abs(d), .99)),
            "different_fraction": float(np.mean(d != 0)),
            "psnr_between_images_db_NOT_quality": "inf" if mse == 0 else 10 * math.log10(255 ** 2 / mse)}


def natural_key(path):
    return tuple((1, int(s)) if s.isdigit() else (0, s.lower()) for s in re.split(r"(\d+)", path.name))


def source_state(repo, out):
    """Compare actual importable Python bytes with the pinned Git objects, not just HEAD."""
    repo, out = Path(repo), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    status = command(["git", "-C", str(repo), "status", "--porcelain"], False)
    (out / "git_status.txt").write_text(status.stdout + status.stderr)
    head = command(["git", "-C", str(repo), "rev-parse", "HEAD"], False).stdout.strip()
    test = command(["git", "-C", str(repo), "cat-file", "-e", PIN + "^{commit}"], False)
    result = {"head": head, "reference": PIN, "reference_available": test.returncode == 0,
              "files": [], "unexpected_python": [], "gpu_gate": "BLOCKED"}
    if test.returncode:
        dump(out / "source_integrity.json", result)
        return result
    listing = command(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", PIN]).stdout.splitlines()
    relevant = [p for p in listing if p.endswith(".py") and
                (p == "inference.py" or p.startswith("model/") or p.startswith("utils/"))]
    if not relevant:
        raise RuntimeError("Pinned source inventory is empty")
    for rel in relevant:
        original = subprocess.check_output(["git", "-C", str(repo), "show", f"{PIN}:{rel}"])
        actual = repo / rel
        expected = hashlib.sha256(original).hexdigest()
        current = digest(actual) if actual.is_file() else None
        result["files"].append({"path": rel, "expected_sha256": expected, "actual_sha256": current,
                                "match": expected == current, "symlink": actual.is_symlink()})
    actual_paths = set()
    for base in (repo / "model", repo / "utils"):
        actual_paths.update(str(x.relative_to(repo)) for x in base.rglob("*.py"))
    result["unexpected_python"] = sorted(actual_paths - set(relevant))
    diff = command(["git", "-C", str(repo), "diff", PIN, "--", "inference.py", "model", "utils"], False)
    (out / "local_code_changes_PRIVATE.diff").write_text(diff.stdout, encoding="utf-8")
    ok = all(x["match"] and not x["symlink"] for x in result["files"]) and not result["unexpected_python"]
    result["gpu_gate"] = "PASS" if ok else "BLOCKED_LOCAL_SOURCE_DIFF"
    dump(out / "source_integrity.json", result)
    return result


def inventory(path):
    fs = frames(path)
    info = {"path": str(path), "count": len(fs), "files": [], "errors": [], "sizes": {},
            "lexical_matches_natural": fs == sorted(fs, key=natural_key)}
    for f in fs:
        item = {"name": f.name, "sha256": digest(f), "bytes": f.stat().st_size,
                "symlink": f.is_symlink()}
        try:
            with Image.open(f) as im:
                im.load()
                size = f"{im.width}x{im.height}"
                item.update(size=list(im.size), mode=im.mode, format=im.format)
                info["sizes"][size] = info["sizes"].get(size, 0) + 1
        except Exception as e:
            info["errors"].append({"name": f.name, "error": repr(e)})
        info["files"].append(item)
    p = Path(path)
    info["broken_image_symlinks"] = ([x.name for x in p.iterdir() if x.suffix.lower() in EXT and
                                      x.is_symlink() and not x.exists()] if p.is_dir() else [])
    return info


def comparison_sheet(items, destination, roi=None, zoom=1):
    """No spatial resampling in 1x sheets. 4x is explicitly nearest pixel inspection."""
    items = [(name, Image.fromarray(a.astype(np.uint8))) for name, a in items]
    if not items:
        return
    sizes = {im.size for _, im in items}
    if len(sizes) != 1:
        return
    views = []
    for name, im in items:
        if roi is not None:
            im = im.crop(roi)
        if zoom != 1:
            im = im.resize((im.width * zoom, im.height * zoom), Image.Resampling.NEAREST)
        views.append((name, im))
    w, h = views[0][1].size
    canvas = Image.new("RGB", (w * len(views), h + 36), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (name, im) in enumerate(views):
        canvas.paste(im, (i * w, 36))
        draw.text((i * w + 3, 3), f"{name} | {zoom}x" + (" NEAREST inspection" if zoom != 1 else " native"), fill="black")
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def rois(a, side=160):
    """Input-only, coordinate-labelled candidates, NOT a person/aliasing detector."""
    h, w = a.shape[:2]
    side = min(side, h, w)
    y = a.astype(np.float32).mean(2)
    dy, dx = np.gradient(y)
    energy = np.hypot(dx, dy)
    candidates = []
    for yy in range(0, h - side + 1, max(side // 2, 1)):
        for xx in range(0, w - side + 1, max(side // 2, 1)):
            candidates.append((float(energy[yy:yy + side, xx:xx + side].mean()), (xx, yy, xx + side, yy + side)))
    selected = [((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2)]
    for _, r in sorted(candidates, reverse=True):
        if all(abs(r[0] - q[0]) >= side or abs(r[1] - q[1]) >= side for q in selected):
            selected.append(r)
        if len(selected) == 3:
            break
    return selected


def static_audit(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "static.json").exists():
        raise RuntimeError("This run already has static.json; choose a new --out, do not overwrite evidence")
    state = source_state(args.repo, out)
    base = args.root / "benchmark/outputs/realvdeblur_blackwell"
    paths = {"input": args.root / "benchmark/input_frames", "FP16": base / "frames", "BF16": base / "frames_BF16"}
    inv = {name: inventory(p) for name, p in paths.items()}
    fi = frames(paths["input"])
    if not fi:
        raise RuntimeError("No input images")
    samples = sorted(set(i for center in (1, len(fi) // 4, len(fi) // 2, 3 * len(fi) // 4, len(fi) - 2)
                         for i in (center - 1, center, center + 1) if 0 <= i < len(fi)))
    pairs = []
    for i in samples:
        image = rgb(fi[i])
        items = [("input", image)]
        for label in ("FP16", "BF16"):
            p = paths[label] / fi[i].name
            if p.is_file():
                a = rgb(p)
                pairs.append({"index": i, "name": fi[i].name, "pair": f"input_vs_{label}", **metrics(image, a)})
                if a.shape == image.shape:
                    items.append((label, a))
        if len(items) == 3:
            pairs.append({"index": i, "name": fi[i].name, "pair": "historical_FP16_vs_BF16_NOT_controlled", **metrics(items[1][1], items[2][1])})
        stem = f"{i:06d}_{fi[i].stem}"
        comparison_sheet(items, out / "existing_png" / (stem + "_native.png"))
        for n, box in enumerate(rois(image)):
            for zoom in (1, 4):
                comparison_sheet(items, out / "existing_png" / f"{stem}_ROI{n}_{'_'.join(map(str, box))}_{zoom}x.png", box, zoom)
    weights = args.root / "benchmark/weights/realvdeblur"
    weight_info = []
    for f in (weights / "Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
              weights / "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", weights / "realvdeblur_dmd.safetensors"):
        weight_info.append({"path": str(f), "exists": f.is_file(), "bytes": f.stat().st_size if f.is_file() else None,
                            "sha256": digest(f) if f.is_file() else None})
    video = args.root / "input/xiaobieli38_trimmed.mp4"
    video_info = {"path": str(video), "exists": video.is_file(), "field_order": "NOT_CHECKED"}
    if video.is_file():
        video_info["sha256"] = digest(video)
        if shutil.which("ffprobe"):
            p = command(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=width,height,avg_frame_rate,nb_frames,pix_fmt,field_order", "-of", "json", str(video)], False)
            video_info["ffprobe_stdout"] = p.stdout
            video_info["ffprobe_stderr"] = p.stderr
        try:
            import cv2
            cv2.setNumThreads(1)
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise RuntimeError("Video cannot be opened")
            j, comparisons = 0, []
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                if j in samples:
                    a = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    comparisons.append({"index": j, "decoded_size": [a.shape[1], a.shape[0]], **metrics(rgb(fi[j]), a)})
                j += 1
            cap.release()
            video_info.update(decoded_frames=j, source_vs_canonical= comparisons,
                              decoding_note="Different RGB decoder/color conversions may differ; nonzero MAE alone is not evidence of resize")
        except Exception as e:
            video_info["opencv_check"] = "UNAVAILABLE: " + repr(e)
    missing = {label: sorted(set(f.name for f in fi) - set(f.name for f in frames(paths[label]))) for label in ("FP16", "BF16")}
    result = {"source": state, "inventory": inv, "missing_output_names": missing,
              "weights": weight_info, "video": video_info, "samples": samples, "pair_metrics": pairs,
              "interpretation": "Measurements are diagnostic differences, not deblur quality, and do not prove jagged-edge cause."}
    dump(out / "static.json", result)
    print("STATIC_AUDIT_COMPLETE", out)


def select_gpu(args):
    text = command(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"]).stdout
    rows = []
    for row in csv.reader(io.StringIO(text)):
        idx, uuid, name, total, free, driver = [s.strip() for s in row]
        rows.append(dict(index=int(idx), uuid=uuid, name=name, total_MiB=int(total), free_MiB=int(free), driver=driver))
    if args.allowed_gpus:
        allowed = set(args.allowed_gpus.split(","))
        rows = [r for r in rows if str(r["index"]) in allowed or r["uuid"] in allowed]
    elif os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID"):
        raise RuntimeError("Scheduler detected: pass --allowed-gpus with the actual allocation; do not bypass it")
    if not rows:
        raise RuntimeError("No permitted GPU")
    winner = sorted(rows, key=lambda r: (-r["free_MiB"], r["index"]))[0]
    dump(args.out / f"gpu_{args.mode}_{args.precision}.json", {"inventory": rows, "selected": winner,
         "selection": "largest free MiB, ties smallest index", "logical_device": "cuda:0"})
    if winner["free_MiB"] < args.min_free_mib:
        raise RuntimeError(f"NO_FREE_GPU: selected {winner}, conservative audit gate {args.min_free_mib} MiB, not a model minimum")
    os.environ["CUDA_VISIBLE_DEVICES"] = winner["uuid"]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    print("SELECTED_PHYSICAL_GPU", winner, "LOGICAL cuda:0", flush=True)
    return winner


def gpu_setup(args):
    select_gpu(args)  # MUST precede importing torch or any project module.
    import torch
    torch.set_num_threads(4)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    p = torch.cuda.get_device_properties(0)
    x = torch.ones((128, 128), device="cuda", dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    assert torch.isfinite(y).all()
    del x, y
    # FP32 reference must not silently use TF32. Applied identically in these diagnostic runs.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(args.repo))
    from model.src.realvdeblur_pipeline import RealVDeblurPipeline
    from model.src.wan_video_vae import WanVideoVAE
    from model.src.utils import load_state_dict
    from model.src.lora import GeneralLoRALoader
    from utils.data_load import load_media_list
    module = sys.modules[RealVDeblurPipeline.__module__]
    if not Path(module.__file__).resolve().is_relative_to(args.repo.resolve()):
        raise RuntimeError("Imported an unexpected RealVDeblur checkout")
    info = {"python": sys.executable, "python_version": sys.version, "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "archs": torch.cuda.get_arch_list(),
            "gpu": torch.cuda.get_device_name(0), "compute_capability": [p.major, p.minor],
            "visible_uuid": os.environ["CUDA_VISIBLE_DEVICES"], "module_file": module.__file__,
            "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False}
    dump(args.out / f"environment_{args.mode}_{args.precision}.json", info)
    return torch, RealVDeblurPipeline, WanVideoVAE, load_state_dict, GeneralLoRALoader, load_media_list


def assert_source(args):
    state = source_state(args.repo, args.out / ("source_before_" + args.mode))
    if state["gpu_gate"] != "PASS":
        raise RuntimeError("BLOCKED_LOCAL_SOURCE_DIFF: CPU audit is retained. Return local_code_changes_PRIVATE.diff; do not repair code autonomously")


def save_rgb_fp32(y, path):
    """Same truncation as official exporter; ONLY range conversion arithmetic becomes FP32."""
    import torch
    y = y.detach().float().cpu()
    while y.ndim > 3:
        if y.shape[0] == 1:
            y = y.squeeze(0)
        elif y.shape[1] == 1:
            y = y.squeeze(1)
        else:
            raise ValueError(f"Unexpected RGB tensor {tuple(y.shape)}")
    if y.shape[0] != 3 or not torch.isfinite(y).all():
        raise ValueError("Invalid/nonfinite decoded RGB before uint8 conversion")
    a = ((y.permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).to(torch.uint8).numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path)
    return a


def stats(t):
    import torch
    a = t.detach().float()
    ok = bool(torch.isfinite(a).all())
    return {"shape": list(a.shape), "dtype": str(t.dtype), "finite": ok,
            "min": float(a.min()) if ok else None, "max": float(a.max()) if ok else None,
            "mean": float(a.mean()) if ok else None,
            "clamped_endpoint_fraction": float(((a <= -1) | (a >= 1)).float().mean()) if ok else None}


def capture(args):
    assert_source(args)
    dest = args.out / ("capture_" + args.precision)
    dest.mkdir(exist_ok=False)
    torch, Pipeline, _, load_state, Loader, load_media = gpu_setup(args)
    dtype = getattr(torch, args.precision)
    fi = frames(args.root / "benchmark/input_frames")
    if fi != sorted(fi, key=natural_key):
        raise RuntimeError("FRAME_ORDER_AMBIGUOUS: lexical and numeric orders differ; audit extraction manifest first")
    start = args.clip_start if args.clip_start is not None else max(0, len(fi) // 2 - 12)
    selected = fi[start:start + 24]
    if len(selected) != 24:
        raise RuntimeError("Require 24 contiguous full-resolution frames")
    images, names, h, w = load_media(selected, stride=1)
    original_sizes = {Image.open(p).size for p in selected}
    if original_sizes != {(w, h)} or (w, h) != (1280, 720):
        raise RuntimeError("UNEXPECTED_RESOLUTION: no automatic resize/crop permitted in this audit")
    samples = [10, 11, 12]
    metadata = {"clip_start": start, "clip_frames": 24, "sample_relative": samples,
                "sample_global": [start + i for i in samples], "sample_names": [names[i] for i in samples],
                "input_sha256": [digest(p) for p in selected], "all_names": names, "size_wh": [w, h],
                "dtype": args.precision, "seed": 0, "steps": 1, "twm": True, "window": 21,
                "warning": "24-frame context is NOT equivalent to historical 452-frame inference. Compare paired diagnostic runs only."}
    dump(dest / "clip.json", metadata)
    weights = args.root / "benchmark/weights/realvdeblur"
    pipe = Pipeline(wan_model_dir=weights / "Wan2.1-T2V-1.3B", device="cuda:0", torch_dtype=dtype,
                    enable_twm=True, temporal_window_size=21).eval()
    ckpt = weights / "realvdeblur_dmd.safetensors"
    sd = load_state(str(ckpt), device="cpu")
    mapping = Loader().get_name_dict(sd)
    modules = dict(pipe.dit.named_modules())
    bkeys = [k for k in sd if ".lora_B." in k]
    usedkeys = {k for pair in mapping.values() for k in pair}
    unmatched = sorted(set(mapping) - set(modules))
    orphan = sorted(k for k in sd if (".lora_A." in k or ".lora_B." in k) and k not in usedkeys)
    audit = {"checkpoint": str(ckpt), "b_key_count": len(bkeys), "mapped_target_count": len(mapping),
             "matched_count": len(set(mapping) & set(modules)), "unmatched_targets": unmatched,
             "orphan_lora_keys": orphan, "shape_errors": [], "sample_updates": []}
    for name, (bk, ak) in mapping.items():
        if name not in modules or bk not in sd or ak not in sd:
            audit["shape_errors"].append(name + ": missing target/pair")
            continue
        up, down = sd[bk], sd[ak]
        if up.ndim == 4:
            up, down = up.squeeze(3).squeeze(2), down.squeeze(3).squeeze(2)
        weight = getattr(modules[name], "weight", None)
        if up.ndim != 2 or down.ndim != 2 or up.shape[1] != down.shape[0] or weight is None or tuple(weight.shape[:2]) != (up.shape[0], down.shape[1]):
            audit["shape_errors"].append(name + ": incompatible matrix/weight shape")
    dump(dest / "lora_audit.json", audit)
    if not mapping or len(mapping) != len(bkeys) or unmatched or orphan or audit["shape_errors"]:
        raise RuntimeError("LORA_COVERAGE_FAILED; preserve evidence, do not repair weights")
    keys = sorted(mapping)
    sample_names = sorted(set(keys[i] for i in (0, len(keys) // 2, len(keys) - 1)))
    before = {n: modules[n].weight.detach().cpu().clone() for n in sample_names}
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        loaded = pipe.load_checkpoint(ckpt, alpha=1)
    print(log.getvalue(), end="")
    (dest / "checkpoint_load.log").write_text(log.getvalue())
    audit["embedding_load_status"] = loaded
    match = re.findall(r"(\d+) tensors are updated by LoRA", log.getvalue())
    audit["reported_updated_count"] = int(match[-1]) if match else None
    for n in sample_names:
        after = modules[n].weight.detach().cpu()
        delta = after.float() - before[n].float()
        audit["sample_updates"].append({"target": n, "changed_elements": int((delta != 0).sum()),
                                        "max_abs_update": float(delta.abs().max())})
    dump(dest / "lora_audit.json", audit)
    if not all(loaded.values()) or audit["reported_updated_count"] != len(mapping):
        raise RuntimeError("LORA_LOAD_CONFIRMATION_FAILED")
    if not any(x["changed_elements"] for x in audit["sample_updates"]):
        raise RuntimeError("LORA_SAMPLED_UPDATE_ZERO_REQUIRES_REVIEW")
    del sd, before, modules
    gc.collect()
    original_decode = pipe.vae.decode
    captured = {}
    def decode_hook(self, z, *a, **kw):
        captured["z"] = z[samples].detach().cpu().clone()
        video = original_decode(z, *a, **kw)
        captured["y"] = video[samples].detach().cpu().clone()
        return video
    pipe.vae.decode = types.MethodType(decode_hook, pipe.vae)
    try:
        with torch.inference_mode():
            restored = pipe(blur_video=images, seed=0, height=h, width=w, num_frames=24, num_inference_steps=1)
    finally:
        pipe.vae.decode = original_decode
    if len(restored) != 24 or set(captured) != {"z", "y"}:
        raise RuntimeError("Unexpected pipeline capture count")
    torch.save(captured, dest / "captured_tensors.pt")
    conversion = []
    for i, im in enumerate(restored):
        f = dest / "official_png" / f"{i:06d}.png"
        f.parent.mkdir(exist_ok=True)
        im.save(f)
    for n, i in enumerate(samples):
        stable = save_rgb_fp32(captured["y"][n], dest / "FP32_export_only" / f"{i:06d}.png")
        conversion.append({"relative_index": i, **metrics(rgb(dest / "official_png" / f"{i:06d}.png"), stable)})
    dump(dest / "capture_report.json", {"latents": stats(captured["z"]), "decoded": stats(captured["y"]),
         "export_arithmetic_only_comparison": conversion, "hook": "capture-only, original decode always called, method restored",
         "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024 ** 3})
    print("CAPTURE_COMPLETE", dest)


def vae_tests(args):
    assert_source(args)
    capdir = args.out / ("capture_" + args.precision)
    meta = json.loads((capdir / "clip.json").read_text())
    dest = args.out / ("vae_tests_" + args.precision)
    dest.mkdir(exist_ok=False)
    torch, Pipeline, VAE, load_state, _, _ = gpu_setup(args)
    captured = torch.load(capdir / "captured_tensors.pt", map_location="cpu", weights_only=True)
    z, baseline = captured["z"], captured["y"]
    path = args.root / "benchmark/weights/realvdeblur/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
    report = {"same_latent_sha256": digest(capdir / "captured_tensors.pt"), "source": str(path),
              "same_latent_tests": [], "roundtrip": [], "controlled_decode_gate": "NOT_RUN"}
    # Each VAE is freshly loaded from ORIGINAL checkpoint. Never upcast an already-rounded VAE.
    for precision in dict.fromkeys([args.precision, "float32"]):
        dtype = getattr(torch, precision)
        raw = load_state(str(path), device="cpu")
        inner = raw.get("model_state", raw)
        report.setdefault("checkpoint_dtype_counts", {})[precision] = {str(d): sum(1 for t in inner.values() if torch.is_tensor(t) and t.dtype == d)
            for d in set(t.dtype for t in inner.values() if torch.is_tensor(t))}
        vae = VAE().eval().requires_grad_(False)
        vae.load_state_dict(vae.state_dict_converter().from_civitai(raw), strict=True)
        del raw, inner
        vae.to(device="cuda:0", dtype=dtype)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            y = vae.decode(z.to(device="cpu", dtype=dtype), device="cuda:0", tiled=False).detach().cpu()
            if not torch.isfinite(y).all():
                raise RuntimeError("NONFINITE_VAE_DECODE")
            for n, i in enumerate(meta["sample_relative"]):
                a = save_rgb_fp32(y[n], dest / ("same_latent_" + precision) / f"{i:06d}.png")
                b = save_rgb_fp32(baseline[n], dest / "captured_baseline_FP32_export" / f"{i:06d}.png")
                report["same_latent_tests"].append({"decode_dtype": precision, "sample": i,
                    "raw_rgb_max_abs_vs_capture": float((y[n].float() - baseline[n].float()).abs().max()),
                    **metrics(a, b)})
            if precision == args.precision:
                rawerr = float((y.float() - baseline.float()).abs().max())
                report["native_replay_raw_bitwise_equal"] = bool(torch.equal(y.float(), baseline.float()))
                # Diagnostic tolerance only; discrepancies still need explanation, not a quality criterion.
                report["controlled_decode_gate"] = "PASS" if rawerr <= 1e-6 else "REVIEW_NATIVE_REPLAY_MISMATCH"
            for n, name in enumerate(meta["sample_names"]):
                with Image.open(args.root / "benchmark/input_frames" / name) as im:
                    original = im.convert("RGB")
                    dummy = types.SimpleNamespace(torch_dtype=dtype, device=torch.device("cuda:0"))
                    x = Pipeline.preprocess_image(dummy, original).unsqueeze(2)
                latent = vae.encode(x, device="cuda:0", tiled=False)
                yy = vae.decode(latent, device="cuda:0", tiled=False).detach().cpu()
                a = save_rgb_fp32(yy[0], dest / ("roundtrip_" + precision) / f"{meta['sample_relative'][n]:06d}.png")
                report["roundtrip"].append({"dtype": precision, "input_name": name,
                    "meaning": "VAE reconstruction fidelity to input, NOT deblur PSNR", **metrics(np.asarray(original), a)})
                del latent, yy, x
        report.setdefault("peak_allocated_GiB", {})[precision] = torch.cuda.max_memory_allocated() / 1024 ** 3
        del y, vae
        gc.collect()
        torch.cuda.empty_cache()
        dump(dest / "vae_report.json", report)
    for n, i in enumerate(meta["sample_relative"]):
        image = rgb(args.root / "benchmark/input_frames" / meta["sample_names"][n])
        items = [("input", image), ("native decode", rgb(dest / ("same_latent_" + args.precision) / f"{i:06d}.png")),
                 ("FP32 decode SAME latent", rgb(dest / "same_latent_float32" / f"{i:06d}.png"))]
        comparison_sheet(items, dest / f"decode_{i}_native.png")
        for k, box in enumerate(rois(image)):
            comparison_sheet(items, dest / f"decode_{i}_ROI{k}_{'_'.join(map(str,box))}_1x.png", box)
            comparison_sheet(items, dest / f"decode_{i}_ROI{k}_{'_'.join(map(str,box))}_4x_NEAREST.png", box, 4)
        comparison_sheet([(label, rgb(dest / label / f"{i:06d}.png")) for label in
                          ("roundtrip_" + args.precision, "roundtrip_float32")], dest / f"roundtrip_{i}_native.png")
    print("VAE_TESTS_COMPLETE", dest)


def verify_unchanged(args):
    reference = json.loads((args.out / "static.json").read_text())
    changed = []
    for group in reference["inventory"].values():
        root = Path(group["path"])
        old_names = {x["name"] for x in group["files"]}
        if old_names != {x.name for x in frames(root)}:
            changed.append(str(root) + ": file membership changed")
        for f in group["files"]:
            path = root / f["name"]
            if not path.is_file() or digest(path) != f["sha256"]:
                changed.append(str(path))
    for f in reference["weights"]:
        path = Path(f["path"])
        if f["exists"] and (not path.is_file() or digest(path) != f["sha256"]):
            changed.append(str(path))
    current = source_state(args.repo, args.out / "source_after")
    before = reference["source"]
    if current.get("files") != before.get("files") or current.get("unexpected_python") != before.get("unexpected_python"):
        changed.append("RealVDeblur source changed during audit")
    dump(args.out / "preservation_check.json", {"unchanged": not changed, "changed": changed})
    if changed:
        raise RuntimeError("PRESERVATION_CHECK_FAILED: " + repr(changed))
    print("PRODUCTION_FILES_UNCHANGED")


def summarize(args):
    entries = []
    for p in sorted(args.out.rglob("*.json")):
        entries.append(str(p.relative_to(args.out)))
    report = ["# RealVDeblur edge audit — measurements, not a confirmed diagnosis", "",
              "Status: EVIDENCE_COLLECTED_REVIEW_REQUIRED", "",
              "Do not label sharpness/gradient energy, between-output PSNR, or successful execution as proof of correct edges.",
              "Do not equate a 24-frame rerun to an existing 452-frame sequence.",
              "FP32 decode fixes neither information already lost in latents nor proven upstream aliasing.", "", "## Evidence files"]
    report += ["- " + p for p in entries]
    for precision in ("float16", "bfloat16"):
        p = args.out / ("vae_tests_" + precision) / "vae_report.json"
        if p.is_file():
            d = json.loads(p.read_text())
            report += ["", f"{precision} native replay gate: {d['controlled_decode_gate']}",
                       "If this gate is not PASS, treat the same-latent attribution as inconclusive."]
    a, b = args.out / "capture_float16/clip.json", args.out / "capture_bfloat16/clip.json"
    if a.is_file() and b.is_file():
        da, db = json.loads(a.read_text()), json.loads(b.read_text())
        fields = ("clip_start", "clip_frames", "sample_relative", "input_sha256", "seed", "steps", "twm", "window")
        same = all(da[k] == db[k] for k in fields)
        dump(args.out / "paired_context_check.json", {"matched_controls": same, "fields": fields})
        if same:
            pairs = []
            for i in da["sample_relative"]:
                aa = rgb(args.out / "capture_float16/official_png" / f"{i:06d}.png")
                bb = rgb(args.out / "capture_bfloat16/official_png" / f"{i:06d}.png")
                pairs.append({"relative_index": i, **metrics(aa, bb)})
                comparison_sheet([("FP16 controlled", aa), ("BF16 controlled", bb)], args.out / f"paired_{i}_native.png")
            dump(args.out / "controlled_FP16_BF16_differences.json", pairs)
    failures = sorted(p.name for p in args.out.glob("failure_*.log"))
    if failures:
        report[2] = "Status: PARTIAL_EVIDENCE_WITH_FAILURES"
        report += ["", "Failures retained: " + ", ".join(failures)]
    report += ["", "## Reviewer must answer", "",
        "1. Are jaggies present in native output PNG, or only resized display?",
        "2. Are dimensions, source decoding, filenames, and official source intact?",
        "3. Did every checkpoint LoRA target match? Did sampled weights actually change?",
        "4. Did native VAE replay match captured RGB? What changed with FP32 decode on IDENTICAL latents?",
        "5. Are similar artifacts visible in VAE-only roundtrips?",
        "6. Which observations are facts, hypotheses, or still unresolved?", "",
        "No source code was intentionally modified. No fix is authorized by this audit."]
    (args.out / "MEASUREMENTS_SUMMARY.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("SUMMARY_READY", args.out / "MEASUREMENTS_SUMMARY.md")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["audit", "capture", "vae", "verify", "summary"])
    p.add_argument("--root", type=Path, default=Path("/data/pub1/h00306136/motion_deblur"))
    p.add_argument("--repo", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--precision", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--clip-start", type=int)
    p.add_argument("--allowed-gpus", help="Optional comma-separated scheduler-permitted physical indices or full UUIDs")
    p.add_argument("--min-free-mib", type=int, default=24576)
    args = p.parse_args()
    args.root = args.root.resolve()
    args.repo = (args.repo or args.root / "envs/realvdeblur_repo").resolve()
    args.out = args.out.resolve()
    protected = [args.repo, args.root / "input", args.root / "benchmark/input_frames", args.root / "benchmark/weights",
                 args.root / "benchmark/outputs/realvdeblur_blackwell"]
    if any(args.out == x or args.out.is_relative_to(x) or x.is_relative_to(args.out) for x in protected):
        p.error("--out must be an isolated NEW diagnostic directory, not a production/source ancestor or descendant")
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    try:
        {"audit": static_audit, "capture": capture, "vae": vae_tests, "verify": verify_unchanged, "summary": summarize}[args.mode](args)
    except Exception:
        text = traceback.format_exc()
        (args.out / f"failure_{args.mode}_{args.precision}.log").write_text(text)
        print(text, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
