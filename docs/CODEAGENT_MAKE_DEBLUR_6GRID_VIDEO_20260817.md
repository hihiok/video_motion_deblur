# CODEAGENT_MAKE_DEBLUR_6GRID_VIDEO_20260817

## Goal

Generate one 2x3 comparison MP4 from six already-generated frame streams.

The order is fixed and MUST NOT be changed:

```text
Row 1: Input       | DSTNet | BSSTNet
Row 2: Shift-Net+  | Turtle | RealVDeblur
```

Each tile MUST show its model name at the top-left.

Use the repository script exactly as committed:

```text
scripts/make_deblur_6up_video.py
```

**Do not write, regenerate, or modify the Python script.**
If the script is missing or fails, report the error instead of creating replacement code.

---

## 0. Proxy and SSL configuration

This repository is public, so proxy usernames/passwords MUST NOT be committed into this markdown file.
Use the server's existing secure proxy activation file when available.

First run:

```bash
# Load proxy credentials from the existing local secure config.
# The motion-deblur project handoff records this as the normal proxy source.
if [ -n "${CONDA_PREFIX:-}" ] && [ -f "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh" ]; then
    source "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh"
fi

# Preserve both lower-case and upper-case proxy variables if already loaded.
[ -n "${http_proxy:-}" ]  && export HTTP_PROXY="$http_proxy"
[ -n "${https_proxy:-}" ] && export HTTPS_PROXY="$https_proxy"

# Huawei/internal network may use an intercepting/self-signed certificate.
# Explicitly skip git SSL verification as requested.
git config --global http.sslVerify false

# If proxy variables are loaded, let git use them as well.
if [ -n "${http_proxy:-}" ]; then
    git config --global http.proxy "$http_proxy"
fi
if [ -n "${https_proxy:-}" ]; then
    git config --global https.proxy "$https_proxy"
fi

echo "http_proxy=${http_proxy:+SET}"
echo "https_proxy=${https_proxy:+SET}"
git config --global --get http.sslVerify || true
```

If GitHub access still fails because the secure proxy environment is not loaded, stop and report that the local proxy configuration must be activated. **Do not print or commit proxy passwords.**

---

## 1. Clone or update the prepared GitHub repository

Repository:

```text
https://github.com/hihiok/video_motion_deblur.git
```

Run:

```bash
set -euo pipefail

REPO_URL="https://github.com/hihiok/video_motion_deblur.git"
WORK_ROOT="/mnt/ssd1/z00919662/motion_deblur/codeagent_tools"
REPO_DIR="$WORK_ROOT/video_motion_deblur"

mkdir -p "$WORK_ROOT"

if [ -d "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR"
    git fetch origin main
    git checkout main
    git pull --ff-only origin main
else
    git -c http.sslVerify=false clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

printf 'REPO_HEAD: '
git rev-parse HEAD
printf 'REPO_BRANCH: '
git branch --show-current

test -f scripts/make_deblur_6up_video.py
```

The repository HEAD must contain commit `a5d46b745e4e0e48bf2c3dcf236e22bafb4f667c` or a later commit that includes the same script.

Verify:

```bash
git merge-base --is-ancestor a5d46b745e4e0e48bf2c3dcf236e22bafb4f667c HEAD
```

If this command fails, stop and report `PREPARED_SCRIPT_COMMIT_NOT_PRESENT`.

---

## 2. Fixed source paths

Use exactly these paths:

```bash
INPUT_DIR="/mnt/ssd1/z00919662/motion_deblur/benchmark/input_frames"
DSTNET_DIR="/mnt/ssd1/z00919662/motion_deblur/benchmark/outputs/dstnet_dvd/frames"
BSSTNET_DIR="/mnt/ssd1/z00919662/motion_deblur/benchmark/outputs/bsstnet_dvd/frames"
SHIFTNET_DIR="/mnt/ssd1/z00919662/motion_deblur/runs/shiftnet_only_20260812_v1/frames"
TURTLE_DIR="/mnt/ssd1/z00919662/motion_deblur/runs/turtle/outputs/Turtle_GoPro_simple_320_128_200k_kamran_no_pos/xiaobieli38_trimmed"
REALVDEBLUR_DIR="/mnt/ssd1/z00919662/motion_deblur/benchmark/outputs/realvdeblur_blackwell/frames"

OUT_DIR="/mnt/ssd1/z00919662/motion_deblur/benchmark/compare_6grid_20260817"
mkdir -p "$OUT_DIR"
```

Turtle frames are specifically the files matching:

```text
Frame_xxx_Pred.png
```

The prepared script automatically prefers `Frame_<number>_Pred.png` in the Turtle directory.

---

## 3. Preflight checks

Do not generate any new inference outputs. Only consume the existing frames.

Check all six directories:

```bash
for d in \
    "$INPUT_DIR" \
    "$DSTNET_DIR" \
    "$BSSTNET_DIR" \
    "$SHIFTNET_DIR" \
    "$TURTLE_DIR" \
    "$REALVDEBLUR_DIR"
do
    echo "CHECK_DIR: $d"
    test -d "$d" || { echo "MISSING_DIR: $d"; exit 1; }
done
```

Print frame counts before running:

```bash
python - <<'PY'
import re
from pathlib import Path

items = [
    ("Input", Path("/mnt/ssd1/z00919662/motion_deblur/benchmark/input_frames"), False),
    ("DSTNet", Path("/mnt/ssd1/z00919662/motion_deblur/benchmark/outputs/dstnet_dvd/frames"), False),
    ("BSSTNet", Path("/mnt/ssd1/z00919662/motion_deblur/benchmark/outputs/bsstnet_dvd/frames"), False),
    ("Shift-Net+", Path("/mnt/ssd1/z00919662/motion_deblur/runs/shiftnet_only_20260812_v1/frames"), False),
    ("Turtle", Path("/mnt/ssd1/z00919662/motion_deblur/runs/turtle/outputs/Turtle_GoPro_simple_320_128_200k_kamran_no_pos/xiaobieli38_trimmed"), True),
    ("RealVDeblur", Path("/mnt/ssd1/z00919662/motion_deblur/benchmark/outputs/realvdeblur_blackwell/frames"), False),
]

exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
counts = []
for name, root, turtle in items:
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if turtle:
        pred = [p for p in files if re.fullmatch(r"Frame_\d+_Pred\.png", p.name, flags=re.I)]
        if pred:
            files = pred
    files = sorted(files)
    counts.append(len(files))
    print(f"{name}: {len(files)}")
    if files:
        print(f"  first={files[0].name}")
        print(f"  last ={files[-1].name}")

print(f"MIN_COUNT={min(counts) if counts else 0}")
print(f"MAX_COUNT={max(counts) if counts else 0}")
if not counts or min(counts) == 0:
    raise SystemExit("FAIL: at least one stream has zero frames")
PY
```

A count mismatch is not automatically fatal because the prepared script truncates all streams to the shortest stream, but the mismatch MUST be reported in the final summary.

---

## 4. Python / ffmpeg environment

Check dependencies:

```bash
cd "$REPO_DIR"
python -V
ffmpeg -version | head -n 2 || true
python -c "import cv2, numpy; print('cv2', cv2.__version__); print('numpy', numpy.__version__)"
```

If `cv2` or `numpy` is missing, install only the missing runtime dependency; do not modify repository code:

```bash
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    opencv-python-headless numpy
```

If ffmpeg is unavailable, the prepared script has an OpenCV MP4 fallback. Do not write another encoder script.

---

## 5. Generate the six-grid comparison video

Run the prepared script exactly with these source paths:

```bash
cd "$REPO_DIR"

python scripts/make_deblur_6up_video.py \
    --input-dir "$INPUT_DIR" \
    --dstnet-dir "$DSTNET_DIR" \
    --bsstnet-dir "$BSSTNET_DIR" \
    --shiftnet-dir "$SHIFTNET_DIR" \
    --turtle-dir "$TURTLE_DIR" \
    --realvdeblur-dir "$REALVDEBLUR_DIR" \
    --output-dir "$OUT_DIR" \
    --fps 25 \
    --crf 18
```

For a 1280x720 input stream, the default output is:

```text
Each tile: 640x360
Full 2x3 canvas: 1920x720
```

Expected visual order:

```text
+----------------+----------------+----------------+
| Input          | DSTNet         | BSSTNet        |
+----------------+----------------+----------------+
| Shift-Net+     | Turtle         | RealVDeblur    |
+----------------+----------------+----------------+
```

---

## 6. Validate outputs

Required files:

```bash
test -s "$OUT_DIR/comparison_6grid.mp4"
test -s "$OUT_DIR/preview_first_frame.png"
test -s "$OUT_DIR/run_summary.txt"
```

Inspect video metadata:

```bash
ffprobe -v error \
  -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames \
  -of default=noprint_wrappers=1 \
  "$OUT_DIR/comparison_6grid.mp4" || true
```

Print the generated summary:

```bash
cat "$OUT_DIR/run_summary.txt"
ls -lh \
  "$OUT_DIR/comparison_6grid.mp4" \
  "$OUT_DIR/preview_first_frame.png" \
  "$OUT_DIR/run_summary.txt"
```

CodeAgent should also visually inspect `preview_first_frame.png` if its environment supports image inspection and verify:

1. exactly six tiles are present;
2. first row is `Input | DSTNet | BSSTNet`;
3. second row is `Shift-Net+ | Turtle | RealVDeblur`;
4. each label is visible at the top-left;
5. no tile is obviously blank/corrupt;
6. all tiles show the same temporal scene/frame.

Do not alter the model outputs.

---

## 7. Final report

Return a concise report in this format:

```text
MAKE_DEBLUR_6GRID_VIDEO_20260817
STATUS: PASS / FAIL

REPO_HEAD: <git sha>

VIDEO:
/mnt/ssd1/z00919662/motion_deblur/benchmark/compare_6grid_20260817/comparison_6grid.mp4

PREVIEW:
/mnt/ssd1/z00919662/motion_deblur/benchmark/compare_6grid_20260817/preview_first_frame.png

SUMMARY:
/mnt/ssd1/z00919662/motion_deblur/benchmark/compare_6grid_20260817/run_summary.txt

FRAME_COUNTS:
Input: ...
DSTNet: ...
BSSTNet: ...
Shift-Net+: ...
Turtle: ...
RealVDeblur: ...

USED_FRAMES: ...
OUTPUT_RESOLUTION: ...
FPS: 25
ENCODER: ...
```

If anything fails, report the exact failing command/error. Do not replace the prepared script with CodeAgent-generated code.
