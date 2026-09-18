#!/usr/bin/env python3
"""Run the unchanged official RealVDeblur CLI; prepare frames and export MP4.
No model implementation, monkey-patching, training, or precision modification.
Python >=3.10. Reuse the user's working torch/Pillow environment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
from fractions import Fraction

PIN = "63a2eb0ed4a22cb76b796a65d5d3052616352573"
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
ROOT = Path("/data/pub1/h00306136/motion_deblur")


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def dump(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_text(cmd, cwd=None, env=None):
    p = subprocess.run([str(x) for x in cmd], cwd=cwd, env=env, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    require(p.returncode == 0, f"Command failed ({p.returncode}): {cmd}\n{p.stdout[-10000:]}")
    return p.stdout


def natural_key(name):
    return tuple((0, int(s)) if s.isdigit() else (1, s.casefold())
                 for s in re.split(r"(\d+)", name))


def list_frames(path: Path):
    require(path.is_dir(), f"INPUT_MISSING: {path}")
    files = sorted((p for p in path.iterdir() if p.is_file() and p.suffix.lower() in EXTS),
                   key=lambda p: (natural_key(p.name), p.name))
    require(bool(files), f"NO_INPUT_IMAGES: {path}")
    names = [p.stem + ".png" for p in files]
    require(len({n.casefold() for n in names}) == len(names), "DUPLICATE_OUTPUT_STEMS")
    require(len({natural_key(p.name) for p in files}) == len(files), "AMBIGUOUS_NATURAL_ORDER")
    return files


def prepare_frames(inp: Path, out: Path):
    from PIL import Image
    files = list_frames(inp)
    stage = out / "_input_png"
    stage.mkdir(exist_ok=False)
    entries, size = [], None
    for i, src in enumerate(files):
        dst = stage / f"{i:08d}.png"
        with Image.open(src) as im:
            require(getattr(im, "n_frames", 1) == 1, f"ANIMATED_INPUT: {src}")
            im.load()
            # Do not silently squeeze high bit depth/HDR or reinterpret grayscale datasets.
            require(im.mode in ("RGB", "RGBA", "L", "LA", "P"), f"UNSUPPORTED_IMAGE_MODE {src}: {im.mode}")
            if size is None:
                size = im.size
            require(im.size == size, f"MIXED_RESOLUTION: {src} {im.size} != {size}")
            require(min(size) >= 16, f"IMAGE_TOO_SMALL: {size}")
            mode = im.mode
            if im.format == "PNG" and mode == "RGB":
                dst.symlink_to(src.resolve())
            else:
                # PNG preserves exactly the RGB pixels decoded by Pillow; no spatial resize.
                im.convert("RGB").save(dst, format="PNG", compress_level=1)
        entries.append({"index": i, "source": str(src.resolve()), "source_name": src.name,
                        "source_sha256": sha256(src), "source_mode": mode,
                        "staged_name": dst.name, "output_name": src.stem + ".png"})
    w, h = size
    ow, oh = w // 16 * 16, h // 16 * 16
    data = {"input": str(inp.resolve()), "count": len(files), "input_wh": [w, h],
            "output_wh": [ow, oh], "resize": False,
            "official_center_crop_ltrb": [(w-ow)//2, (h-oh)//2, (w-ow+1)//2, (h-oh+1)//2],
            "ordering": "natural_numeric_then_lexical; official CLI sees zero-padded PNG names",
            "entries": entries}
    dump(out / "frame_manifest.json", data)
    return data


def parse_fps(value):
    fps = Fraction(str(value).strip())
    require(0 < fps <= 1000, f"INVALID_FPS: {value}")
    return f"{fps.numerator}/{fps.denominator}"


def resolve_fps(inp: Path, explicit=None):
    if explicit is not None:
        return {"value": parse_fps(explicit), "source": "explicit --fps", "assumed": False}
    found = []
    # Only this RGB directory, its Blur parent, and scene 002; never borrow another clip's FPS.
    for d in (inp, inp.parent, inp.parent.parent):
        f = d / "fps.txt"
        if f.is_file():
            found.append((parse_fps(f.read_text().strip()), str(f)))
        f = d / "metadata.json"
        if f.is_file():
            data = json.loads(f.read_text())
            if isinstance(data, dict):
                for k in ("fps", "frame_rate", "framerate"):
                    if k in data:
                        found.append((parse_fps(data[k]), str(f) + ":" + k))
    if found:
        require(len({x[0] for x in found}) == 1, f"CONFLICTING_FPS_METADATA: {found}")
        return {"value": found[0][0], "source": "; ".join(x[1] for x in found), "assumed": False}
    return {"value": "30/1", "source": "ASSUMED_PREVIEW_FPS_NO_TIMING_METADATA", "assumed": True}


def select_gpu(text, allowed=None, inherited=None):
    gpus = []
    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        require(len(row) == 6, f"UNEXPECTED_NVIDIA_SMI_ROW: {row}")
        idx, uid, name, total, free, bus = [s.strip() for s in row]
        gpus.append({"index": int(idx), "uuid": uid, "name": name, "total_mib": int(total),
                     "free_mib": int(free), "pci_bus_id": bus})
    require(bool(gpus), "NO_GPU_FOUND")
    candidates = list(gpus)
    for restriction in (inherited, allowed):
        if restriction is None or restriction == "all":
            continue
        tokens = [x.strip() for x in restriction.split(",") if x.strip()]
        require(tokens and not any(x in ("-1", "none", "void") or x.startswith("MIG-") for x in tokens),
                "NO_ALLOWED_GPU_OR_MIG_REQUIRES_EXPLICIT_HANDLING")
        ids = set()
        for t in tokens:
            matched = [g for g in gpus if str(g["index"]) == t or g["uuid"].startswith(t)]
            require(len(matched) == 1, f"UNKNOWN_OR_AMBIGUOUS_GPU: {t}")
            ids.add(matched[0]["uuid"])
        candidates = [g for g in candidates if g["uuid"] in ids]
    require(bool(candidates), "NO_GPU_WITHIN_ALLOWED_ALLOCATION")
    return max(candidates, key=lambda g: (g["free_mib"], -g["index"])), gpus


def audit_repo(repo: Path, out: Path):
    current = run_text(["git", "rev-parse", "HEAD"], cwd=repo).strip()
    require(current == PIN, f"OFFICIAL_COMMIT_MISMATCH: {current}; do not reset the repo")
    diff = run_text(["git", "diff", PIN, "--", "inference.py", "model", "utils"], cwd=repo)
    (out / "official_code.diff").write_text(diff)
    untracked = run_text(["git", "ls-files", "--others", "--exclude-standard"], cwd=repo)
    custom = [x for x in untracked.splitlines() if x.endswith(".py") and
              (x.startswith("model/") or x.startswith("utils/") or "/" not in x)]
    require(not diff and not custom, f"LOCAL_MODEL_CODE_CHANGED: inspect official_code.diff; untracked={custom}")
    return current


PROBE = r'''
import json, os, sys, torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
assert torch.cuda.is_bf16_supported()
p = torch.cuda.get_device_properties(0)
x = torch.randn(128,128,device='cuda:0',dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
assert torch.isfinite(y).all().item()
f,t = torch.cuda.mem_get_info(0)
print(json.dumps({'python':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,
 'arch_list':torch.cuda.get_arch_list(),'gpu':p.name,'capability':[p.major,p.minor],
 'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'free_mib':f//1024**2,
 'total_mib':t//1024**2,'bf16_matmul':'PASS'}))
'''


def find_ffmpeg(explicit=None):
    candidates = [explicit, os.environ.get("FFMPEG_EXE"), os.environ.get("IMAGEIO_FFMPEG_EXE"),
                  shutil.which("ffmpeg"), str(Path(sys.prefix) / "bin/ffmpeg")]
    try:
        import imageio_ffmpeg
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except (ImportError, RuntimeError):
        pass
    for exe in dict.fromkeys(c for c in candidates if c):
        try:
            text = run_text([exe, "-hide_banner", "-encoders"])
            if re.search(r"\blibx264\b", text):
                return str(Path(exe).resolve()) if Path(exe).exists() else exe
        except (OSError, RuntimeError):
            continue
    raise RuntimeError("FFMPEG_UNAVAILABLE: provide --ffmpeg or local imageio-ffmpeg wheel; MP4 is mandatory")


def validate_pngs(folder, manifest, official_names=False):
    from PIL import Image
    expected = [e["staged_name" if official_names else "output_name"] for e in manifest["entries"]]
    present = {p.name for p in folder.iterdir() if p.is_file()}
    require(present == set(expected), f"OUTPUT_NAMES_OR_COUNT_MISMATCH: {folder}")
    bad, constants = [], []
    for name in expected:
        with Image.open(folder / name) as im:
            im.load()
            require(im.format == "PNG" and im.mode == "RGB", f"NOT_RGB_PNG: {name}")
            require(list(im.size) == manifest["output_wh"], f"OUTPUT_SIZE_MISMATCH: {name}")
            if all(lo == hi for lo, hi in im.getextrema()):
                constants.append(name)
                # Conservatively flag flat outputs, not NaN: uint8 PNG cannot verify latent finiteness.
                bad.append(name)
    report = {"count": len(expected), "output_wh": manifest["output_wh"],
              "constant_frames": constants, "passed": not bad,
              "check_scope": "PNG decode, RGB mode, geometry, exact names/count; not latent-finite or perceptual quality"}
    require(not bad, f"CONSTANT_OUTPUT_FRAMES_REVIEW_REQUIRED: {bad[:10]}")
    return report


def export_mp4(out: Path, manifest, fps, ffmpeg):
    seq = out / "_video_sequence"
    seq.mkdir(exist_ok=True)
    for e in manifest["entries"]:
        link = seq / e["staged_name"]
        target = (out / "frames" / e["output_name"]).resolve()
        if link.is_symlink():
            require(link.resolve() == target, f"STALE_ENCODING_LINK: {link}")
        else:
            require(not link.exists(), f"ENCODING_PATH_EXISTS: {link}")
            link.symlink_to(target)
    partial = out / "output.partial.mp4"
    final = out / "output.mp4"
    require(not final.exists(), "MP4_ALREADY_EXISTS: preserve it or choose another output directory")
    require(not partial.exists(), "PARTIAL_MP4_EXISTS: preserve/rename this run's partial file before retry")
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-n", "-threads", "4", "-framerate", fps["value"],
           "-start_number", "0", "-i", str(seq / "%08d.png"), "-frames:v", str(manifest["count"]),
           "-an", "-c:v", "libx264", "-threads", "4", "-preset", "medium", "-crf", "10",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial)]
    dump(out / "ffmpeg_command.json", cmd)
    with (out / "ffmpeg.log").open("w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
    require(rc == 0, f"MP4_ENCODE_FAILED: rc={rc}; see ffmpeg.log")
    # Decode every frame with the same ffmpeg binary: no ffprobe dependency or assumed nb_frames.
    check = [ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-xerror", "-threads", "4",
             "-i", str(partial), "-map", "0:v:0", "-an", "-f", "framemd5", "-"]
    decoded = run_text(check)
    (out / "mp4_decoded_framemd5.txt").write_text(decoded)
    count = len([x for x in decoded.splitlines() if x.strip() and not x.startswith("#")])
    require(count == manifest["count"], f"MP4_FRAME_COUNT_MISMATCH: {count}")
    w, h = manifest["output_wh"]
    require(re.search(rf"#dimensions\s+0:\s*{w}x{h}\b", decoded) is not None, "MP4_DIMENSION_MISMATCH")
    match = re.search(r"#tb\s+0:\s*(\d+/\d+)", decoded)
    require(match is not None and Fraction(match[1]) == 1 / Fraction(fps["value"]), "MP4_RATE_MISMATCH")
    partial.rename(final)
    return {"path": str(final), "decoded_frames": count, "fps": fps,
            "duration_seconds": float(Fraction(count) / Fraction(fps["value"])),
            "width": w, "height": h, "codec": "libx264", "crf": 10, "pixel_format": "yuv420p",
            "audio": "none: input is image sequence", "sha256": sha256(final), "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT / "envs/realvdeblur_repo")
    parser.add_argument("--input", type=Path, default=ROOT / "input/002/Blur/RGB")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmark/outputs/realvdeblur_002_BF16")
    parser.add_argument("--wan-dir", type=Path, default=ROOT / "benchmark/weights/realvdeblur/Wan2.1-T2V-1.3B")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "benchmark/weights/realvdeblur/realvdeblur_dmd.safetensors")
    parser.add_argument("--fps", help="Explicit playback FPS, e.g. 30 or 30000/1001; never changes inference")
    parser.add_argument("--ffmpeg", help="Existing executable with libx264; never installs packages")
    parser.add_argument("--allowed-gpus", help="Further restrict to physical indices/UUIDs; inherited visibility also respected")
    parser.add_argument("--min-free-gib", type=float, default=8, help="Scheduling guard only, NOT a memory requirement estimate")
    parser.add_argument("--encode-only", action="store_true", help="Use this runner's validated existing frames; do not rerun model")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.encode_only:
        require((out / "frame_manifest.json").is_file(), "MISSING_RUN_MANIFEST")
    else:
        require(not out.exists(), f"OUTPUT_EXISTS: {out}; never overwrite a previous run")
        out.mkdir(parents=True)
    metadata = {"status": "RUNNING", "stage": "preflight", "python": sys.executable,
                "mode": "encode_only" if args.encode_only else "official_inference",
                "dtype": "bfloat16", "seed": 0, "steps": 1, "stride": 1, "TWM": True, "window": 21}
    started = time.monotonic()
    try:
        ffmpeg = find_ffmpeg(args.ffmpeg)
        metadata["ffmpeg"] = ffmpeg
        inp = args.input.resolve()
        fps = resolve_fps(inp, args.fps)
        metadata["fps"] = fps
        if args.encode_only:
            manifest = json.loads((out / "frame_manifest.json").read_text())
            require(manifest["input"] == str(inp), "ENCODE_ONLY_INPUT_DIFFERS_FROM_MANIFEST")
            old = out / "run_metadata.json"
            if old.is_file():
                metadata["prior_run"] = json.loads(old.read_text())
                if args.fps is None and "fps" in metadata["prior_run"]:
                    fps = metadata["prior_run"]["fps"]
                    metadata["fps"] = fps
        else:
            metadata["stage"] = "source_code_audit"
            repo = args.repo.resolve()
            metadata["official_commit"] = audit_repo(repo, out)
            weight_files = [args.wan_dir / "diffusion_pytorch_model.safetensors",
                            args.wan_dir / "Wan2.1_VAE.pth", args.checkpoint]
            require(all(p.is_file() for p in weight_files), "WEIGHTS_MISSING")
            metadata["weights"] = [{"path": str(p.resolve()), "bytes": p.stat().st_size,
                                     "sha256": sha256(p)} for p in weight_files]
            metadata["stage"] = "prepare_input"
            manifest = prepare_frames(inp, out)
            # Do not allocate GPU for enormous sequences without documenting the real input geometry/count.
            print(json.dumps({k: v for k, v in manifest.items() if k != "entries"}, ensure_ascii=False), flush=True)
            metadata["stage"] = "gpu_selection"
            inventory = run_text(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,pci.bus_id",
                                  "--format=csv,noheader,nounits"])
            (out / "gpu_inventory.csv").write_text(inventory)
            inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
            if (os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID")) and inherited is None and args.allowed_gpus is None:
                raise RuntimeError("SCHEDULER_ALLOCATION_UNKNOWN: specify --allowed-gpus without bypassing allocation")
            gpu, _ = select_gpu(inventory, args.allowed_gpus, inherited)
            require(gpu["free_mib"] >= args.min_free_gib * 1024, "NO_SUFFICIENTLY_FREE_ALLOWED_GPU")
            metadata["gpu"] = gpu
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID",
                       PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       OPENBLAS_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4")
            probe = run_text([sys.executable, "-c", PROBE], env=env)
            (out / "environment_probe.log").write_text(probe)
            metadata["environment_probe"] = json.loads(probe.strip().splitlines()[-1])
            raw = out / "_official_frames"
            raw.mkdir()
            cmd = [sys.executable, "-u", "inference.py", "--input", str(out / "_input_png"),
                   "--output", str(raw), "--wan_model_dir", str(args.wan_dir.resolve()),
                   "--checkpoint", str(args.checkpoint.resolve()), "--device", "cuda:0", "--dtype", "bfloat16",
                   "--stride", "1", "--seed", "0", "--num_inference_steps", "1", "--enable_twm", "--temporal_window_size", "21"]
            metadata["command"] = cmd
            metadata["stage"] = "full_inference"
            dump(out / "run_metadata.json", metadata)
            print(f"PHYSICAL_GPU={gpu['index']} UUID={gpu['uuid']} FREE_MiB={gpu['free_mib']}; logical cuda:0", flush=True)
            t = time.monotonic()
            with (out / "inference.log").open("w") as log:
                proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1)
                try:
                    for line in proc.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    rc = proc.wait()
                finally:
                    if proc.poll() is None:
                        proc.terminate()  # Only our own direct child, never another user's job.
                        try:
                            proc.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
            metadata["inference_seconds"] = time.monotonic() - t
            metadata["return_code"] = rc
            require(rc == 0, f"OFFICIAL_INFERENCE_FAILED: rc={rc}; see inference.log; no automatic resize/chunk fallback")
            audit_repo(repo, out)
            logtext = (out / "inference.log").read_text()
            counts = re.findall(r"(\d+) tensors are updated by LoRA", logtext)
            require(counts and int(counts[-1]) > 0, "LORA_ZERO_OR_UNREPORTED: preserve results, do not claim full loading")
            metadata["lora_updated_count"] = int(counts[-1])
            metadata["lora_scope_note"] = "Nonzero update count is not a complete key-by-key coverage audit"
            for label in ("allocated", "reserved"):
                m = re.search(rf"Peak {label} GPU memory:\s*([\d.]+) GiB", logtext)
                metadata[f"peak_{label}_gib"] = float(m[1]) if m else None
            validate_pngs(raw, manifest, official_names=True)
            frames = out / "frames"
            frames.mkdir()
            for e in manifest["entries"]:
                (raw / e["staged_name"]).rename(frames / e["output_name"])
        metadata["stage"] = "validate_png"
        validation = validate_pngs(out / "frames", manifest)
        dump(out / "validation_png.json", validation)
        metadata["stage"] = "encode_mp4"
        dump(out / "run_metadata.json", metadata)
        video = export_mp4(out, manifest, fps, ffmpeg)
        dump(out / "validation_mp4.json", video)
        metadata.update(status="REALVDEBLUR_002_PNG_MP4_COMPLETE", stage="complete",
                        png_count=manifest["count"], output_wh=manifest["output_wh"], mp4=video,
                        human_action_required=False)
    except Exception as exc:
        metadata.update(status="FAILED", error=str(exc), human_action_required=True)
        (out / "traceback.txt").write_text(traceback.format_exc())
        print(traceback.format_exc(), file=sys.stderr)
    finally:
        metadata["elapsed_seconds"] = time.monotonic() - started
        dump(out / "run_metadata.json", metadata)
        (out / "FINAL_REPORT.md").write_text("# RealVDeblur 002 PNG + MP4\n\n```json\n" +
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
        print("STATUS=" + metadata["status"], flush=True)
        print("REPORT=" + str(out / "FINAL_REPORT.md"), flush=True)
    return 0 if metadata["status"].endswith("COMPLETE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
