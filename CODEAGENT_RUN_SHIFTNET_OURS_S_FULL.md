# CodeAgent 指令：只跑 Shift-Net Ours-s 全量业务码流

## 执行授权

用户已授权你在本任务 benchmark 范围内执行本文全部 command，不需要、也不要向用户索要 manual approval。不要输出 `WAITING_FOR_USER_CONFIRMATION`，不要要求用户回复 `CONTINUE_*`。颜色与结构硬检查通过后自动继续全量；硬检查失败则停止并报告证据。不要运行 ESTRNN。

## 当前结论

- ESTRNN BSD 3ms-24ms 效果不够好，已从本轮移除：不要重跑、不要等待、不要纳入预览。
- 只运行 Shift-Net Ours-s，必须使用官方 `gshift_deblur2.py` 和 `net_gopro_deblur_small.pth`。
- 旧 smoke 的 Shift 输出虽为 32 帧，但尚未提交 R/B 自动诊断证据。用新脚本自动复核，合格后直接全量 452 帧。

## 1. 代理、SSL 与拉取代码

代理凭据已在 conda 激活脚本中，严禁打印或提交明文。

```bash
set +x
source /mnt/ssd1/z00919662/anaconda3/bin/activate RVRT
source "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$https_proxy"
git config --global http.sslVerify false

cd /mnt/ssd1/z00919662/motion_deblur
if [ ! -d video_motion_deblur/.git ]; then
  git clone https://github.com/hihiok/video_motion_deblur.git
fi
cd video_motion_deblur
git fetch origin agent/fix-color-shiftnet-small-estrnn-20260828
git switch -C agent/fix-color-shiftnet-small-estrnn-20260828 \
  --track origin/agent/fix-color-shiftnet-small-estrnn-20260828
git rev-parse HEAD
```

## 2. 固定路径并检查身份

```bash
SHIFT_REPO=/mnt/ssd1/z00919662/motion_deblur/envs/shiftnet_repo
INPUT=/mnt/ssd1/z00919662/motion_deblur/benchmark/input_frames
SHIFT_CKPT=/mnt/ssd1/z00919662/motion_deblur/benchmark/weights/lightweight_temporal_3models/shiftnet_ours_s_gopro/net_gopro_deblur_small.pth
RUN=/mnt/ssd1/z00919662/motion_deblur/runs/shiftnet_ours_s_colorfix_20260828

test -f "$SHIFT_REPO/basicsr/models/archs/gshift_deblur2.py"
test -f "$SHIFT_CKPT"
test -d "$INPUT"
mkdir -p "$RUN"
sha256sum "$SHIFT_CKPT" | tee "$RUN/checkpoint_sha256.txt"
```

必须确认 SHA256 为 smoke 报告中的：

```text
39f470a77b0b3d23ce5e1e8972e1213ba6cc73a097241af1021281443d2d4f00
```

不匹配则停止，不得换 checkpoint 猜测。

## 3. 自动运行 32 帧 FP32 硬检查

使用新目录，不覆盖旧 smoke：

```bash
cd /mnt/ssd1/z00919662/motion_deblur/video_motion_deblur
python adapters/shiftnet_small_infer.py \
  --repo "$SHIFT_REPO" --input "$INPUT" \
  --checkpoint "$SHIFT_CKPT" \
  --output "$RUN/fp32_32" \
  --one-len 32 --max-frames 32 --device cuda:0

python tools/debug_color_noise.py \
  --input "$INPUT" --output "$RUN/fp32_32" \
  --report "$RUN/fp32_32_color_noise.json"
```

硬检查全部满足才继续：

1. 输出 32/32 帧、1280x720、文件名与前 32 个输入一致。
2. metadata 的 `architecture_required` 为 `gshift_deblur2.py`。
3. checkpoint `strict_checkpoint_load=true`，参数量为 4,705,960，dtype 为 float32。
4. `rb_swap_likely=false`；若为 true，工具会返回非零，立即停止。
5. 不能使用 OpenCV BGR 直接入网或直接保存 RGB；新适配器的颜色契约必须是 `file RGB -> tensor RGB [0,1] -> file RGB (Pillow)`。

无需把检查结果发给用户等待确认。全部通过后立即执行下一节。

## 4. 自动运行全量 452 帧

```bash
python adapters/shiftnet_small_infer.py \
  --repo "$SHIFT_REPO" --input "$INPUT" \
  --checkpoint "$SHIFT_CKPT" \
  --output "$RUN/fp32_full" \
  --one-len 48 --device cuda:0

python tools/debug_color_noise.py \
  --input "$INPUT" --output "$RUN/fp32_full" \
  --report "$RUN/fp32_full_color_noise.json"
```

禁止运行 ESTRNN，禁止启用 FP16，禁止通过额外去噪或锐化掩盖纹理问题。

## 5. 生成正确的 RGB 预览并回报

用 Pillow/RGB 生成输入与 Shift 的第 1、中间、最后帧横向对比 JPG；视频预览只允许在写视频前明确执行 RGB->BGR 一次。不要把 VS Code 文件图标当预览图。

最终回报：

- `git rev-parse HEAD`。
- 输入和全量输出均为 452 帧、1280x720。
- checkpoint 路径、SHA256、strict load、参数量、FP32、GPU、峰值显存和总耗时。
- `fp32_32_metadata.json`、`fp32_32_color_noise.json`、`fp32_full_metadata.json`、`fp32_full_color_noise.json` 的绝对路径及关键值。
- RGB 对比 JPG 和 MP4 的绝对路径。
- 明确写出：`rb_swap_likely`、`blue_minus_red_drift`、`median_laplacian_rms_ratio`。

若 CodeAgent 修改任何仓库文件，必须提交到独立分支并 push，回报 commit SHA；不要直接修改 main。
