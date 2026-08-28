# CodeAgent 指令：定位 Shift-Net Ours-s 偏蓝/纹理噪音并重新推理

## 执行授权与范围

用户已授权你在本任务定义的 motion-deblur benchmark 范围内执行本文全部 command，不需要向用户申请 manual approval，也不要输出 `WAITING_FOR_USER_CONFIRMATION` 或索要 `CONTINUE_*`。硬检查通过后自动继续全量推理；硬检查失败时停止后续推理并报告证据。

只处理 **Shift-Net Ours-s**。不要运行 ESTRNN，不要运行 MMP-RNN，不要修改官方 Shift-Net repo，不要覆盖或删除旧输出。

本任务的重点不是再次宣布 `PASS`，而是用原始 MP4 建立可信 RGB 基准，定位偏蓝发生在：

1. MP4 解码为 PNG；
2. PNG 输入模型；
3. 模型输出保存为 PNG；
4. PNG 编码为 MP4/对比图。

## 0. Proxy、SSL 与环境

先执行以下网络配置；Git clone/fetch 前必须关闭 SSL verify：

```bash
set +x
export http_proxy="http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080"
export https_proxy="http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080"
export https_proxy="http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080"
export HTTPS_PROXY="http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080"
git config --global http.proxy http://z00919662:Zzhs12345%21@proxy.server.com:8080
git config --global https.proxy http://z00919662:Zzhs12345%21@proxy.server.com:8080
git config --global https.proxy https://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080
git config --global http.proxy http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080
git config --global http.sslVerify false

source /mnt/ssd1/z00919662/anaconda3/bin/activate RVRT
export CUDA_VISIBLE_DEVICES=4
```

## 1. 拉取已准备好的正确代码

分支现在已经存在。不要再次自行编写或修改推理 adapter。

```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur
BENCH_REPO="$ROOT/video_motion_deblur"
BRANCH=agent/fix-color-shiftnet-small-estrnn-20260828

if [ ! -d "$BENCH_REPO/.git" ]; then
  git clone https://github.com/hihiok/video_motion_deblur.git "$BENCH_REPO"
fi
cd "$BENCH_REPO"
git fetch --all --prune
git ls-remote --heads origin "$BRANCH"
git switch -C "$BRANCH" --track "origin/$BRANCH"
git pull --ff-only origin "$BRANCH"
git rev-parse HEAD

test -f adapters/shiftnet_small_infer.py
test -f tools/debug_color_noise.py
python -m py_compile adapters/shiftnet_small_infer.py tools/debug_color_noise.py
```

如果 `git ls-remote` 没有返回该分支，停止并报告完整输出；不要临时重写 adapter。

## 2. 同步 CodeAgent 之前自行创建的代码

之前实际使用的是服务器本地文件：

```text
/mnt/ssd1/z00919662/motion_deblur/benchmark_code/adapters/shiftnet_ours_s_infer.py
```

为了让用户能够复核精确行号，把现有文件原样同步到独立审计分支。只复制已有文件，不要修改其内容，不要把 checkpoint、输出图片或代理配置提交进去。

```bash
OLD_ADAPTER="$ROOT/benchmark_code/adapters/shiftnet_ours_s_infer.py"
test -f "$OLD_ADAPTER"

cd "$BENCH_REPO"
git switch -C agent/audit-codeagent-shiftnet-blue-noise "$BRANCH"
mkdir -p audit_snapshots
cp "$OLD_ADAPTER" audit_snapshots/shiftnet_ours_s_infer_codeagent.py
git add audit_snapshots/shiftnet_ours_s_infer_codeagent.py
git commit -m "Add CodeAgent Shift-Net adapter for color audit"
git push -u origin agent/audit-codeagent-shiftnet-blue-noise
git rev-parse HEAD
git switch "$BRANCH"
```

如果 Git push 失败，继续本地诊断和正确推理，但最终必须回报失败原因，并把该文件绝对路径列为“待同步代码”。不需要向用户申请执行批准。

## 3. 审计旧 adapter 和旧解码/预览路径

只读检查，不要修补旧代码：

```bash
OLD_CODE="$ROOT/benchmark_code"
OLD_RUN="$ROOT/runs/lightweight_temporal_3models_20260827"

rg -n "VideoCapture|imwrite|imageio|Image\.fromarray|cvtColor|VideoWriter|transpose|255|\.\.\., *::-1|\.\.\.,::-1" \
  "$OLD_CODE" | tee "$ROOT/runs/shiftnet_color_debug_old_code_hits.txt"
```

按以下规则确定根因并记录文件与行号：

- `cv2.VideoCapture.read()` 返回 BGR。如果该数组直接传给 `imageio.imwrite` 或 `Image.fromarray`，解码 PNG 会红蓝互换；正确做法是在写入前执行一次 `frame[..., ::-1]`。
- Shift-Net Ours-s 官方模型输入和输出都是 RGB。若使用 imageio/Pillow 保存模型输出，不能再执行 `[..., ::-1]`。
- 只有 `cv2.imwrite` 或 `cv2.VideoWriter.write` 接收 BGR，因此 RGB 输出在进入这两个接口前才反转一次。
- 输入模型前范围必须是 RGB `[0,1]`。禁止 BGR 入网、`[0,255]` 入网、重复 `/255` 或 `[-1,1]` 标准化。
- `gshift_deblur2.py`、4,705,960 参数、严格加载 `net_gopro_deblur_small.pth` 才是 Ours-s。

## 4. 从原始 MP4 建立可信 RGB 输入基准

不要复用旧的 `common/input_frames`，因为它可能在 OpenCV→imageio/Pillow 交界处已经交换 R/B。使用 ffmpeg 直接从原始视频重新解码，避免 Python/OpenCV 通道歧义：

```bash
VIDEO="$ROOT/input/xiaobieli38_trimmed.mp4"
SAFE_RUN="$ROOT/runs/shiftnet_ours_s_rgbfix_20260828"
SAFE_INPUT="$SAFE_RUN/input_frames_ffmpeg_rgb"
OLD_INPUT="$OLD_RUN/common/input_frames"

mkdir -p "$SAFE_INPUT"
ffmpeg -hide_banner -loglevel error -y -i "$VIDEO" -vsync 0 \
  -start_number 0 "$SAFE_INPUT/%08d.png"

test "$(find "$SAFE_INPUT" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 452
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,avg_frame_rate,nb_frames \
  -of default=noprint_wrappers=1 "$VIDEO" | tee "$SAFE_RUN/input_ffprobe.txt"
```

如果旧输入目录存在，用目录诊断工具将“旧解码 PNG”与“ffmpeg 可信 PNG”比较：

```bash
if [ -d "$OLD_INPUT" ]; then
  python "$BENCH_REPO/tools/debug_color_noise.py" \
    --input "$SAFE_INPUT" --output "$OLD_INPUT" \
    --report "$SAFE_RUN/old_decode_vs_ffmpeg_rgb.json" || true
fi
```

判定：如果 `rb_swap_likely=true`，说明偏蓝首先发生在旧 MP4→PNG 解码阶段。即使旧模型输出与旧输入相似，旧 smoke 的颜色检查也属于假通过。

## 5. 使用正确 adapter 自动跑 32 帧 FP32

```bash
SHIFT_REPO="$ROOT/envs/shiftnet_repo"
SHIFT_CKPT="$ROOT/benchmark/weights/lightweight_temporal_3models/shiftnet_ours_s_gopro/net_gopro_deblur_small.pth"
SMOKE_OUT="$SAFE_RUN/fp32_32/frames"

test -f "$SHIFT_REPO/basicsr/models/archs/gshift_deblur2.py"
test -f "$SHIFT_CKPT"
test "$(sha256sum "$SHIFT_CKPT" | awk '{print $1}')" = \
  "39f470a77b0b3d23ce5e1e8972e1213ba6cc73a097241af1021281443d2d4f00"

cd "$BENCH_REPO"
python adapters/shiftnet_small_infer.py \
  --repo "$SHIFT_REPO" \
  --input "$SAFE_INPUT" \
  --checkpoint "$SHIFT_CKPT" \
  --output "$SMOKE_OUT" \
  --one-len 16 --max-frames 32 --device cuda:0

python tools/debug_color_noise.py \
  --input "$SAFE_INPUT" --output "$SMOKE_OUT" \
  --report "$SAFE_RUN/fp32_32_color_noise.json"
```

硬检查必须全部通过：

1. 32/32 帧，均为1280×720，文件名对应。
2. metadata 显示 `gshift_deblur2.py`、strict checkpoint load、4,705,960 参数、FP32。
3. `rb_swap_likely=false`。
4. 输入/输出均通过 Pillow 按 RGB 保存；adapter 内没有 OpenCV。
5. 没有 NaN/Inf，没有 `[0,255]` 直接入网或重复归一化。

硬检查通过后直接继续，不要等待用户确认。

## 6. 自动跑452帧全量 FP32

V100 32GB 已确认 `one_len=32` 会 OOM，因此固定 `one_len=16`：

```bash
FULL_OUT="$SAFE_RUN/fp32_full/frames"

python "$BENCH_REPO/adapters/shiftnet_small_infer.py" \
  --repo "$SHIFT_REPO" \
  --input "$SAFE_INPUT" \
  --checkpoint "$SHIFT_CKPT" \
  --output "$FULL_OUT" \
  --one-len 16 --device cuda:0

test "$(find "$FULL_OUT" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 452

python "$BENCH_REPO/tools/debug_color_noise.py" \
  --input "$SAFE_INPUT" --output "$FULL_OUT" \
  --report "$SAFE_RUN/fp32_full_color_noise.json"
```

不要运行 ESTRNN，不要开 FP16，不要通过额外去噪、锐化或颜色校正掩盖模型输出。

## 7. 用 ffmpeg 生成无通道歧义的 MP4

不要使用旧的 OpenCV/imageio 预览脚本。直接让 ffmpeg 读取 RGB PNG：

```bash
ffmpeg -hide_banner -loglevel error -y \
  -framerate 25 -start_number 0 -i "$FULL_OUT/%08d.png" \
  -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p \
  "$SAFE_RUN/shiftnet_ours_s_rgbfix_output.mp4"

ffmpeg -hide_banner -loglevel error -y \
  -i "$VIDEO" -i "$SAFE_RUN/shiftnet_ours_s_rgbfix_output.mp4" \
  -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" \
  -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p \
  "$SAFE_RUN/comparison_input_shift_rgbfix.mp4"
```

对比原视频、可信输入 PNG、模型输出 PNG 和最终 MP4 的第 0/226/451 帧。若 PNG 正常但 MP4 偏蓝，问题只在编码/预览层；若可信输入正常但输出 PNG 偏蓝，问题才在模型 adapter；若可信输入本身偏蓝，检查 ffmpeg 色彩元数据和播放器，不得继续修改模型。

## 8. 最终回报

不要只写 `PASS`。必须回报：

1. 旧 adapter 审计分支和 commit SHA；旧代码造成偏蓝的具体文件与行号。
2. 旧 `common/input_frames` 对可信 ffmpeg 输入的 `rb_swap_likely`、RGB MAE、swapped-RB MAE。
3. 正确推理代码的分支和 commit SHA。
4. checkpoint 路径、SHA256、strict load、参数量、FP32。
5. 新输入/输出帧数、分辨率、耗时和峰值显存。
6. `fp32_32_color_noise.json` 与 `fp32_full_color_noise.json` 的关键值：`rb_swap_likely`、`blue_minus_red_drift`、`median_laplacian_rms_ratio`。
7. 新输出 PNG、输出 MP4、横向对比 MP4 的绝对路径。
8. 若修复颜色后仍有纹理噪音，明确区分：代码/归一化错误，或 GoPro checkpoint 对压缩业务码流的域差异。不要擅自增加后处理。

若任何硬检查失败，保留日志和输出目录并报告；不需要向用户申请 manual approval。
