# CodeAgent：RT-Focuser + Shift/DST，T=3 原生整帧训练 v3

## 任务与当前状态

从 GitHub 下载本次已准备好的代码，完成检查与原生整帧 CUDA 预检，通过后自动启动正式训练。
不需要再次询问用户是否开始。不要在服务器写、改、补训练代码。

- 仓库：`https://github.com/hihiok/video_motion_deblur.git`
- 分支：`agent/rtfocuser-shift-dst-t3-fullframe-v3`
- 最低代码 commit：`818d880c1cfd9d0d69e0afba5ea16e5753733ba0`
- 本指令路径：`docs/CODEAGENT_RTFOCUSER_SHIFT_DST_T3_FULLFRAME_20260909.md`
- 前一版：`agent/rtfocuser-shift-dst-t6-fullframe-v2`，commit `fbc7876c604ca37b3348fc7fd137554eeebe0b32`。
- T6 原生 720p 在 V100 32GB 的 DVD 预检 OOM；GoPro 尚未测，不代表通过。
- 旧 crop 训练已停止在约 85000 iteration；完整保留旧运行目录、checkpoint 和日志。
- 旧 `latest.pth` SHA256：`62784e5e3d244dc441445b33c9fb8502afe634283a6c2af112622950ae623e7e`。
- T6 fullframe 没有启动正式训练，本次 T3 从官方 RT-Focuser 权重重新初始化。

本次只运行以下新配置，不能执行旧 T6 指令、重启旧 crop 任务，或覆盖旧目录。

| 设置 | T3 v3 |
|---|---|
| 连续输入/输出 | 3帧输入、3帧输出，全部监督 |
| GoPro/DVD 原生尺寸 | 720×1280，旋转增强后可以是1280×720 |
| BSD 原生尺寸 | 480×640，旋转增强后可以是640×480 |
| 空间处理 | 无 crop、无 resize、无训练 tiling；网络内部必要 padding 后还原原尺寸 |
| 数据采样 | GoPro/BSD/DVD 等概率1:1:1，随机连续3帧 |
| Batch / 累计 | batch=1，gradient_accumulation=4 |
| 显存设置 | AMP、activation checkpointing、GPU EMA |
| CPU | DataLoader 4 workers；OMP/MKL/OpenBLAS各2线程 |
| 学习率 | 5e-5，最小1e-6，warmup 4000 microbatches |
| 训练预算 | 360000 microbatches = 90000次参数更新 = 1080000帧 |
| 验证/保存 | 每10000 microbatches；每域24个原生整帧clip验证 |
| Loss | Charbonnier + FFT + edge + GT帧差匹配 + GT二阶帧差匹配 |
| 时域loss渐入 | 从10000 microbatches开始，在40000 microbatches内渐入 |
| 评估/推理 | window=3、temporal_overlap=2，tile_size=0 |

注意：旧训练器 iteration 按 microbatch 计数。T6 原方案是180000/2=90000次更新，
不是180000次更新。本次360000/4仍为90000次更新，并非训练预算翻倍。
累计4次独立T3片段也不等价于连续T6时域信息。
`RTFocuserT6`、`rtf_t6`、工具文件名及 `--architecture t6` 是保留的兼容标识；实际 T 由新 YAML 决定。

## 1. 代理和 SSL（必须先于 clone/fetch/pip）

**代理内容在服务器 `/mnt/ssd1/z00919662/motion_deblur/proxy.md`。**

由 CodeAgent 在服务器本地读取该文件，使用其中的实际代理配置设置
`http_proxy`、`https_proxy`、`HTTP_PROXY`、`HTTPS_PROXY` 以及 Git HTTP/HTTPS 代理。
文件如果是 Markdown，提取其中有效 shell 配置，不能把整个 Markdown 当作 shell 执行。
不要把用户名、密码、完整带凭据 URL 输出到对话/日志，也不要复制进 GitHub。
禁止 `set -x`、打印全部环境变量或带凭据的 `git config --list`。

按用户指定的公司代理环境跳过 SSL 校验，clone 前执行：

```bash
set +x
export GIT_SSL_NO_VERIFY=true
export CONDA_SSL_VERIFY=false
git config --global http.sslVerify false
```

如必须使用 curl 下载，加 `-k`；pip 下载按实际主机设置 `--trusted-host`。
本任务的数据与官方权重均已在服务器，不应重新下载。不要重装/升级 CUDA PyTorch。
若代理文件不可读或缺失，报告这个具体阻塞，不猜测凭据。

## 2. 获取新分支，避免“指令文件找不到”

在同一个 bash 会话中执行以下代码块。变量若因工具分会话丢失，重新设置本节固定变量。

```bash
set -euo pipefail
export ROOT=/mnt/ssd1/z00919662/motion_deblur
export CODE=$ROOT/benchmark_code_t3_fullframe_v3
export RUN=$ROOT/runs/rtfocuser_shift_dst_t3_gopro_bsd_dvd_fullframe_v3
export CONFIG=$CODE/configs/rtf_t3_gopro_bsd_dvd_fullframe.yaml
export ENV_NAME=deblur_runtime
BRANCH=agent/rtfocuser-shift-dst-t3-fullframe-v3
MIN_COMMIT=818d880c1cfd9d0d69e0afba5ea16e5753733ba0
OLD_RUN=$ROOT/runs/rtfocuser_shift_dst_t6_gopro_bsd_dvd_v1

if [[ ! -e "$CODE" ]]; then
  git clone --single-branch --branch "$BRANCH" \
    https://github.com/hihiok/video_motion_deblur.git "$CODE"
fi
cd "$CODE"
test -d .git
test -z "$(git status --short)"
git fetch origin "refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
git switch --detach "refs/remotes/origin/$BRANCH"
git merge-base --is-ancestor "$MIN_COMMIT" HEAD
test -z "$(git status --short)"
test -s docs/CODEAGENT_RTFOCUSER_SHIFT_DST_T3_FULLFRAME_20260909.md
test -s "$CONFIG"
git rev-parse HEAD
mkdir -p "$RUN/audit"
```

若新目录已存在但不是本仓库，或工作区有本地改动，不要删除、强制checkout或reset。
报告 `git diff`、新文件路径和具体原因，以便中央同步修复。
若文件仍然不存在，报告 fetch 原始错误与实际 HEAD，不要自行重建指令或猜运行命令。

旧目录 `$OLD_RUN` 和 `$ROOT/runs/rtfocuser_shift_dst_t6_gopro_bsd_dvd_fullframe_v2`
均保持原状。无需重复停止已停止的进程；不使用历史 PID 杀进程。

## 3. 激活现有环境，选择空闲 GPU

```bash
source /mnt/ssd1/z00919662/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
python - <<'PY'
import sys, torch, numpy, PIL, yaml, pytest
print('python:', sys.executable)
print('torch/cuda:', torch.__version__, torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA unavailable'
PY
nvidia-smi
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv
```

已验证环境为 `deblur_runtime`，torch2.2.2 / CUDA11.8 / Tesla V100-PCIE-32GB。
环境已具备旧版本12项测试依赖；不要无条件运行 pip upgrade。
如果只缺 pytest/yaml 等纯Python依赖，可以补装缺失依赖，保留现有torch；记录安装内容。

根据实时 `nvidia-smi` 选择真正空闲的卡，不要默认GPU3仍然空闲，不能结束其他用户任务。
设置 `export GPU=<实际空闲卡号>`，再执行：

```bash
: "${GPU:?先根据实时GPU状态设置GPU}"
export CUDA_VISIBLE_DEVICES="$GPU"
python - <<'PY'
import torch
print('Selected GPU:', torch.cuda.get_device_name(0))
print('Free/total GiB:', [round(x / 2**30, 2) for x in torch.cuda.mem_get_info(0)])
PY
export PRETRAINED=$ROOT/envs/RT-Focuser/Pretrained_Weights/GoPro_RT_Focuser_Standard_256.pth
test -s "$PRETRAINED"
sha256sum "$PRETRAINED"
```

必须匹配官方权重 SHA256：
`6bc1310af51b2e47cf631c987ce7da1057aa469c3770d973195b2856014ad3fb`。
启动脚本也会校验；不能换成旧 crop checkpoint。

## 4. 配置、数据与复杂度门禁

```bash
python - <<'PY'
import os, yaml
from rtf_t6.protocol import check_fullframe
c = yaml.safe_load(open(os.environ['CONFIG']))
check_fullframe(c)
t = c['train']
assert t['clip_length'] == 3 and t['gradient_accumulation'] == 4
assert t['batch_size'] == 1 and t['workers'] == 4
assert t['total_iters'] == 360000
assert t['total_iters'] // t['gradient_accumulation'] == 90000
assert c['validation']['stride'] == 3
print('T3_FULLFRAME_CONFIG_PASS')
PY
CUDA_VISIBLE_DEVICES='' python -m pytest -q 2>&1 | tee "$RUN/audit/unit_tests.log"
python tools/audit_rtf_t6_data.py --config "$CONFIG" \
  --output "$RUN/audit/dataset_audit.json" 2>&1 | tee "$RUN/audit/dataset_audit.log"
python tools/check_rtf_t6_complexity.py --config "$CONFIG" --pretrained "$PRETRAINED" \
  --output "$RUN/audit/complexity.json" 2>&1 | tee "$RUN/audit/complexity.log"
```

当前本地 PyTorch2.2.2 CPU 测试结果：18 passed；包括真实训练→暂停保存→恢复更新→整帧评估→7帧目录推理。
这些测试使用小型合成帧/测试权重，不能当作V100原生720p显存验证。

必须 `DATA_AUDIT_PASS`、`COMPLEXITY_GATE_PASS`、`target_coverage=1.0`。
预期参数量 candidate=5710131、baseline=5864531；T3参数不增加。
64×64探针的同口径计算量：candidate=948874368、baseline=962155648 per output frame。
这里是卷积MAC加估算时域算术的工程比较口径，不是完整FLOPs实测。
720p面积外推约213.50G/帧；temporal overlap带来的重复窗口计算未计入，不能当作整段推理的实际单帧开销。

已验证数据布局供核对（由配置自动发现，不移动/复制原始数据）：

| 域 | 训练 | 验证 |
|---|---|---|
| GoPro | `/mnt/ssd1/z00919662/motion_deblur_backup/datasets/GoPro`，22序列/2103帧 | 同根，11序列/1111帧 |
| BSD | `/mnt/ssd1/z00919662/datasets/BSD`，60序列/6000帧 | `/mnt/ssd1/z00919662/RVRT/datasets/BSD`，20序列/3000帧 |
| DVD | `/mnt/ssd1/z00919662/motion_deblur_backup/datasets/DeepVideoDeblurring_Dataset`，66序列/6208帧 | 同根，5序列/500帧 |

若发现不同数量、训练验证泄漏、配对异常或尺寸异常，先核查并报告，不能用“跨域差异”解释数据错误。

## 5. 两轮原生整帧 CUDA 预检

```bash
bash scripts/run_rtf_t3_fullframe_training.sh preflight 2>&1 | tee "$RUN/audit/fullframe_preflight.log"
```

该步骤执行：

1. 每域选择最大画面面积序列，第一轮按BSD→DVD→GoPro连续执行。
2. 每域累计4个T3 microbatch后检查梯度、裁剪、optimizer step、EMA update。
3. 保持optimizer状态和EMA在GPU，测试EMA原生整帧验证。
4. 第二轮训练帧旋转90度，覆盖训练增强产生的H/W交换，然后再做原生尺寸EMA验证。
5. 两轮之间及不同域之间不调用 `empty_cache()`；记录24个microbatch、6次optimizer更新。

结果写入 `$RUN/audit/fullframe_memory_preflight.json`。
全部3域各2轮完整通过、损失/梯度有限、输出尺寸一致，才允许继续。
报告应含每域两轮的 `train_peak_allocated_gib`、`validation_peak_allocated_gib`、
`peak_reserved_gib`、`train_seconds_per_update` 和 `train_seconds_per_microbatch`。

如果 OOM/NaN，立即停止并返回 active_step（域、轮次、microbatch）、错误和现有JSON。
不能进一步降T、crop、resize、tile或自行调整代码/allocator。T3能否装下以此处实测为准。
本版不依赖 `expandable_segments`，不要未经验证追加allocator参数。

## 6. 真实加载器 smoke 与正式启动

```bash
bash scripts/run_rtf_t3_fullframe_training.sh smoke 2>&1 | tee "$RUN/audit/smoke.log"
```

Smoke使用真实三域随机加载器、原生分辨率与4次累计，运行8个microbatch、2次参数更新；
第4/8次运行EMA验证，临时checkpoint写入 `$RUN/smoke/checkpoints/`。
必须完成 `TRAINING_COMPLETE iteration=8`，latest/best存在且所有loss有限。
Smoke临时缩短学习率调度，不用于评价质量，其权重不能用于正式训练。

随后从官方权重重新初始化启动完整预算：

```bash
nohup env ROOT="$ROOT" CODE="$CODE" CONFIG="$CONFIG" RUN="$RUN" \
  ENV_NAME="$ENV_NAME" PRETRAINED="$PRETRAINED" GPU="$GPU" \
  bash "$CODE/scripts/run_rtf_t3_fullframe_training.sh" formal \
  > "$RUN/launcher.log" 2>&1 &
TRAIN_PID=$!
echo "TRAIN_PID=$TRAIN_PID"
ps -fp "$TRAIN_PID"
```

启动器有运行目录锁；会核对预检的代码commit、配置hash、官方权重hash、GPU、torch/CUDA设置。
换GPU、换环境、换commit/配置后必须重新预检。
已有正式训练输出时不会覆盖。不要同时启动两个launcher。

监控至至少100个microbatch，期间每次等待不超过60秒，持续报告状态：

```bash
tail -80 "$RUN/launcher.log"
nvidia-smi
```

第一批日志必须为 `full_frame_first_batch`，shape为 `[1,3,3,H,W]`，无crop/resize。
训练日志同时报告 `iteration`（microbatch）、`optimizer_updates` 和 `seconds_per_microbatch`。
100 microbatches应是25次optimizer更新。估算完成时间使用实测秒数乘剩余microbatch，注明还需验证开销。
首次正式latest通常在10000 microbatches保存；100次时不存在正式checkpoint不属于错误，不要把smoke文件当作正式checkpoint。

## 7. 中断恢复（仅限本次T3正式run）

如果正式run已有有效checkpoint，确认没有同任务进程仍在运行、同一GPU仍可用后：

```bash
nohup env ROOT="$ROOT" CODE="$CODE" CONFIG="$CONFIG" RUN="$RUN" \
  ENV_NAME="$ENV_NAME" PRETRAINED="$PRETRAINED" GPU="$GPU" \
  bash "$CODE/scripts/run_rtf_t3_fullframe_training.sh" resume \
  > "$RUN/resume_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

恢复器会拒绝T6、crop、累计次数或调度不匹配的checkpoint，也拒绝非完整optimizer边界的checkpoint。
不要把旧crop权重、T6预检临时权重或本次smoke权重用于resume。
若已有正式输出但尚无checkpoint，保留日志并报告，由中央处理，不要覆盖后从零静默重跑。

## 8. 完训后的同口径评估

立即报告任务只需到第100个microbatch，不必为了等完训占用会话。
训练完成后执行以下评估；若CodeAgent不能跨会话自动跟进，明确通知用户届时重新发送评估任务。

```bash
for ARCH in rtfocuser_baseline t6; do
  if [[ "$ARCH" == rtfocuser_baseline ]]; then
    CKPT="$PRETRAINED"
  else
    CKPT="$RUN/checkpoints/best_balanced_psnr.pth"
  fi
  python tools/eval_rtf_t6.py --architecture "$ARCH" --config "$CONFIG" \
    --checkpoint "$CKPT" --output "$RUN/eval_${ARCH}_native_t3.json" \
    --window 3 --temporal-overlap 2 --tile-size 0 --tile-overlap 0 --device cuda:0 --amp
done
```

分别报告GoPro/BSD/DVD与balanced的PSNR、SSIM、input PSNR、temporal residual L1。
与旧crop中心验证27.07dB不可直接比较。相同window并不代表每输出帧没有重复计算，需另外测业务推理速度。
训练验证的“每域24clip”也不等价于此处全序列评估。

业务帧目录推理（完训后；先确认该输入确实仍是452帧1280×720业务视频）：

```bash
python tools/infer_rtf_t6.py --config "$CONFIG" \
  --checkpoint "$RUN/checkpoints/best_balanced_psnr.pth" \
  --input "$ROOT/benchmark/input_frames" --output "$RUN/business_t3/frames" \
  --window 3 --temporal-overlap 2 --tile-size 0 --tile-overlap 0 --device cuda:0 --amp
```

输入若缺失/身份不符先报告，不要随意换视频。输出目录必须为空，不覆盖其他运行。
人工检查闪烁、拖影、双边、脸部变形、窗口接缝。输出MP4时沿用源视频真实帧率，不猜fps。

## 必须返回的立即报告

```text
STATUS: IN_PROGRESS / FAILED（formal完训之后才可写COMPLETE）
GITHUB_BRANCH / GIT_COMMIT / WORKTREE_CLEAN
OLD_CROP_RUN_PRESERVED / OLD_T6_FULLFRAME_AUDIT_PRESERVED
ENV_NAME / PYTHON / TORCH / CUDA
GPU / GPU_MODEL
PRETRAINED_PATH / SHA256 / TARGET_COVERAGE
RESOLVED_ROOTS_AND_TRAIN_VAL_SEQUENCE_FRAME_COUNTS_BY_DOMAIN
UNIT_TEST_RESULT / DATA_AUDIT_RESULT
PARAMETERS / COMPUTE_GATE_RESULT / COMPLEXITY_CLIP_LENGTH
T3_FULLFRAME_CONFIG_RESULT / GRADIENT_ACCUMULATION
PREFLIGHT_STATUS / ROUNDS / OPTIMIZER_UPDATES
PREFLIGHT_NATIVE_AND_ROTATED_SHAPES_BY_DOMAIN
PREFLIGHT_TRAIN_VALIDATION_PEAK_MEMORY_BY_DOMAIN_AND_ROUND
PREFLIGHT_SECONDS_PER_MICROBATCH_AND_UPDATE
SMOKE_RESULT / SMOKE_CHECKPOINTS
FORMAL_TRAIN_PID / FIRST_FORMAL_BATCH_SHAPE
CROP_APPLIED / RESIZE_APPLIED
CURRENT_MICROBATCH / OPTIMIZER_UPDATES / LOSS_COMPONENTS / LR
SECONDS_PER_MICROBATCH / ESTIMATED_360K_MICROBATCH_COMPLETION_TIME
LATEST_FORMAL_CHECKPOINT / BEST_FORMAL_CHECKPOINT（尚未到保存点写NOT_YET_SAVED）
SOURCE_CODE_MODIFIED: NO
HUMAN_ACTION_REQUIRED: YES/NO（若YES说明具体原因）
```

如果服务器CodeAgent已经自行修改代码，停止基于未同步版本继续运行，把完整 `git diff` 与未跟踪源码回传，
由中央修改并推送GitHub后再执行；不要让用户只收到一段无法复现的修改说明。
