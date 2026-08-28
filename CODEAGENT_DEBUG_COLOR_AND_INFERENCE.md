# CodeAgent 指令：修复偏蓝与纹理噪音，并重跑 Shift-Net Ours-s / ESTRNN BSD 3ms-24ms

> **已被最新用户决定取代：不要再运行 ESTRNN。** 本文仅保留为历史双模型诊断记录。当前执行入口是 `CODEAGENT_RUN_SHIFTNET_OURS_S_FULL.md`，只运行 Shift-Net Ours-s。

## 执行授权（必须遵守）

用户已授权你在本任务定义的 benchmark 范围内执行本文全部 command，不需要、也不要向用户索要 manual approval。不要在 smoke test 后输出 `WAITING_FOR_USER_CONFIRMATION`，不要要求用户回复 `CONTINUE_*`。硬性检查通过后自动继续全量推理；硬性检查失败时停止后续推理并报告证据，这属于校验失败，不是等待人工授权。授权不包括删除任务目录之外的数据、覆盖旧结果、修改官方模型仓库或泄露凭据。

## 目标与硬约束

在 `hiscv20` 上使用我已提交到 GitHub 的推理脚本，不要现场重写适配器。先诊断旧结果，再用同一业务码流分别重跑：

1. Shift-Net **Ours-s**：官方 `gshift_deblur2.py` + `net_gopro_deblur_small.pth`。
2. ESTRNN：`ESTRNN_C80B15_BSD_3ms24ms.tar`。

硬约束：首轮必须 FP32；checkpoint 必须 strict load；输出必须与输入同数、同尺寸、同文件名；不得覆盖旧结果；颜色检查出现 `rb_swap_likely=true` 时立即停止。

## 0. 网络代理、SSL 与代码同步

代理凭据已经由管理员写入 conda 激活脚本。不得打印、复制或提交代理明文。

```bash
set +x
source /mnt/ssd1/z00919662/anaconda3/bin/activate RVRT
source "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$https_proxy"
git config --global http.sslVerify false
```

获取我提供的分支（若目录已存在就 fetch/reset 到该远端分支；不要改动模型官方 repo）：

```bash
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

## 1. 先定位旧代码的错误

找出产生“偏蓝 + 纹理噪音”的确切命令、adapter、checkpoint 和 dtype，并记录到 `debug_root_cause.md`。重点逐项核对：

- 如果使用 `cv2.imread`，其数组是 BGR；送入 Shift-Net 前必须转 RGB。
- 如果 RGB 模型输出用 `cv2.imwrite` 保存，必须先 `rgb[..., ::-1]`；用 Pillow 保存 RGB 则不要反转。
- ESTRNN 官方 `inference.py` 从 `cv2.imread` 直接入网，因此 **checkpoint 的模型侧通道顺序是 BGR**。文件统一为 RGB 时，必须在模型边界做 `RGB -> BGR -> model -> BGR -> RGB`，且只能各做一次。
- Shift-Net Ours-s 必须加载 `gshift_deblur2.py`。`gshift_deblur1.py` 是 Ours+，不是 Ours-s。
- Shift checkpoint 文件名必须含 `small`，ESTRNN 必须是 `C80B15_BSD_3ms24ms`；两者都必须 strict load，不能 `strict=False`，不能忽略 missing/unexpected keys。
- Shift 输入范围是 RGB `[0,1]`，不做 `[-1,1]` 标准化。ESTRNN 按官方实现使用 BGR `[0,255]` 后 `(x-127.5)/255`，输出严格反变换。
- 首轮禁止 AMP/FP16。错误 checkpoint/架构/归一化或 FP16 数值问题，都可能产生颗粒纹理和伪细节。

对旧输出先跑自动诊断（把路径替换成真实旧目录）：

```bash
python tools/debug_color_noise.py \
  --input /mnt/ssd1/z00919662/motion_deblur/benchmark/input_frames \
  --output /ABS/PATH/TO/OLD_OUTPUT \
  --report /ABS/PATH/TO/OLD_OUTPUT_color_noise.json
```

如果交换 R/B 后的 MAE 明显更低，根因就是通道交换。若两个模型都偏蓝，优先检查公共读写/可视化层；若仅 ESTRNN 偏蓝，优先检查其 BGR 模型边界。

## 2. 确认官方 repo 与 checkpoint

```bash
SHIFT_REPO=/mnt/ssd1/z00919662/motion_deblur/envs/shiftnet_repo
ESTRNN_REPO=/mnt/ssd1/z00919662/motion_deblur/envs/ESTRNN
INPUT=/mnt/ssd1/z00919662/motion_deblur/benchmark/input_frames
SHIFT_CKPT=/mnt/ssd1/z00919662/motion_deblur/benchmark/weights/lightweight_temporal_3models/shiftnet_ours_s_gopro/net_gopro_deblur_small.pth
ESTRNN_CKPT=/mnt/ssd1/z00919662/motion_deblur/benchmark/weights/lightweight_temporal_3models/estrnn_bsd_3ms24ms/ESTRNN_C80B15_BSD_3ms24ms.tar
RUN=/mnt/ssd1/z00919662/motion_deblur/runs/colorfix_20260828

test -f "$SHIFT_REPO/basicsr/models/archs/gshift_deblur2.py"
test -f "$ESTRNN_REPO/model/ESTRNN.py"
test -f "$SHIFT_CKPT"
test -f "$ESTRNN_CKPT"
test -d "$INPUT"
mkdir -p "$RUN"
sha256sum "$SHIFT_CKPT" "$ESTRNN_CKPT" | tee "$RUN/checkpoint_sha256.txt"
```

若路径不同，只允许搜索已有文件并更新变量，不要擅自换模型：

```bash
find /mnt/ssd1/z00919662/motion_deblur -type f \
  \( -name 'net_gopro_deblur_small.pth' -o -name 'ESTRNN_C80B15_BSD_3ms24ms.tar' \) -print
```

## 3. 先跑 32 帧 FP32 验证

```bash
cd /mnt/ssd1/z00919662/motion_deblur/video_motion_deblur

python adapters/shiftnet_small_infer.py \
  --repo "$SHIFT_REPO" --input "$INPUT" \
  --checkpoint "$SHIFT_CKPT" --output "$RUN/shiftnet_ours_s_fp32_32" \
  --one-len 32 --max-frames 32 --device cuda:0

python tools/debug_color_noise.py \
  --input "$INPUT" --output "$RUN/shiftnet_ours_s_fp32_32" \
  --report "$RUN/shiftnet_ours_s_fp32_32_color_noise.json"

python adapters/estrnn_bsd_infer.py \
  --repo "$ESTRNN_REPO" --input "$INPUT" \
  --checkpoint "$ESTRNN_CKPT" --output "$RUN/estrnn_bsd_3ms24ms_fp32_32" \
  --chunk-size 16 --max-frames 32 --device cuda:0

python tools/debug_color_noise.py \
  --input "$INPUT" --output "$RUN/estrnn_bsd_3ms24ms_fp32_32" \
  --report "$RUN/estrnn_bsd_3ms24ms_fp32_32_color_noise.json"
```

验证门槛：两份 metadata 均显示 strict load；Shift architecture 为 `gshift_deblur2.py`；两份 color report 均为 `rb_swap_likely=false`；两个模型输出都必须是 **32/32 帧**且尺寸一致。ESTRNN 首尾使用反射时序 padding，首尾各丢 2 帧的 28/32 结果是旧推理逻辑，判定失败。若纹理 warning 仍触发，先检查旧/新结果的 FP32、checkpoint、归一化与模型参数，不得靠后处理降噪掩盖问题。全部硬检查通过后，无需人工确认，立即执行第 4 节全量推理。

## 4. 通过后用同一业务码流跑全量

```bash
python adapters/shiftnet_small_infer.py \
  --repo "$SHIFT_REPO" --input "$INPUT" \
  --checkpoint "$SHIFT_CKPT" --output "$RUN/shiftnet_ours_s_fp32_full" \
  --one-len 48 --device cuda:0

python adapters/estrnn_bsd_infer.py \
  --repo "$ESTRNN_REPO" --input "$INPUT" \
  --checkpoint "$ESTRNN_CKPT" --output "$RUN/estrnn_bsd_3ms24ms_fp32_full" \
  --chunk-size 16 --device cuda:0

python tools/debug_color_noise.py --input "$INPUT" \
  --output "$RUN/shiftnet_ours_s_fp32_full" \
  --report "$RUN/shiftnet_ours_s_fp32_full_color_noise.json"
python tools/debug_color_noise.py --input "$INPUT" \
  --output "$RUN/estrnn_bsd_3ms24ms_fp32_full" \
  --report "$RUN/estrnn_bsd_3ms24ms_fp32_full_color_noise.json"
```

不要先开 FP16。只有 FP32 视觉、通道检查和纹理检查通过后，才允许单独建立 FP16 输出目录做 A/B；若 FP16 出现噪点，保留 FP32 为正式结果。

## 5. 回报格式

回报以下内容，不要只说“完成”：

1. 旧代码确切根因与对应文件/行号。
2. `git rev-parse HEAD`、GPU、PyTorch/CUDA 版本。
3. 两个 checkpoint 的绝对路径、SHA256、strict load 结果、参数量、dtype。
4. 32 帧和全量的输入/输出帧数、尺寸、耗时、显存峰值。
5. 两份 metadata 与四份 color/noise JSON 路径；明确报告 R/B swap、blue drift、HF ratio。
6. 各抽取第 1、中间、最后帧做输入/Shift/ESTRNN 横向对比图，必须用 Pillow/RGB 生成。
7. 若仍有纹理噪音，提供 FP32 输出与输入的 200% crop，不得擅自修改网络或加去噪。

如果 CodeAgent 修改了任何仓库文件，必须提交到独立分支并 push，回报 commit SHA；不要直接改 main。
