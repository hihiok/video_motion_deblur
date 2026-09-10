# CodeAgent：新服务器 A100 80GB，T3 整帧训练，单/双卡测速后选择

## 任务与已确认信息

更新现有代码，核对新服务器的数据和官方权重；在空闲A100上完成原生整帧预检，
比较单卡/双卡同训练预算的速度，选择合适方案启动正式训练，监控后返回报告。
用户已授权运行和使用至多两张空闲GPU，无需再询问是否启动。

- 仓库：`https://github.com/hihiok/video_motion_deblur.git`
- 新分支：`agent/rtfocuser-shift-dst-t3-a100-v4`
- 最低代码commit：`e774caae01d28f80b690d94b5de7c76abd8aacf9`
- 完整指令：`docs/CODEAGENT_RTFOCUSER_SHIFT_DST_T3_A100_20260910.md`
- 已有代码：`/data/pub/z00919662/motion_deblur/benchmark_code_t3_fullframe_v3`
- 三域数据父目录：`/data/pub/z00919662/dataset`
- 已有环境：`deblur_runtime`，不假设conda安装路径与旧服务器相同。
- 硬件：8张 NVIDIA A100-SXM4-80GB；最多使用2张实时空闲的完整80GB卡，不占用其他人的任务。

历史V100 32GB预检在DVD C0041的1080×1920帧OOM，C0036也是1080p；
BSD T3通过，但此前没有完成GoPro T3预检。旧文档“DVD全是720p”的说法不正确。
本版保留T3、网络结构、全部数据与checkpointing粒度；利用新硬件重新预检。
**不能把“80GB应该够”当作实测通过。两卡DDP各自处理完整T3片段，不会把两卡显存合为160GB。**

## 训练预算与约束

| 设置 | 单卡 | 双卡DDP |
|---|---|---|
| 连续输入/输出 | T3全部监督 | 每卡T3全部监督 |
| 每卡batch | 1 | 1 |
| 每卡梯度累计 | 4 | 2 |
| 每次参数更新的全局帧数 | 12 | 12 |
| 每rank总microbatches | 360000 | 180000 |
| 总optimizer更新 | 90000 | 90000 |
| 全局训练帧预算 | 1080000 | 1080000 |
| DataLoader workers总数 | 4 | 2×2=4 |
| 验证/保存间隔（每rank microbatches） | 10000 | 5000 |

lr不翻倍，warmup与时域loss渐入按optimizer更新保持相同时间点。
双卡非更新microbatch使用DDP `no_sync()`，仅在累计结束时同步梯度。
日志loss在两卡间取平均；EMA与正式checkpoint由rank0保存；验证由rank0执行，其他rank等待。
保留每卡BatchNorm，不启用SyncBatchNorm，因此不要求单/双卡训练权重逐bit相同。

- 原生整帧：不crop、不resize、不训练tiling，不排除DVD C0036/C0041。
- 允许原有同步翻转/90度旋转/时间反转；网络padding后输出还原原生尺寸。
- GoPro/BSD/DVD等概率采样，AMP、activation checkpointing、GPU EMA、全部loss保持开启。
- 每进程OMP/MKL/OpenBLAS各2线程，不擅自扩大CPU并行度。
- 不改服务器源码；路径差异通过命令行参数生成仓库外的运行配置。
- 保留旧checkpoint和预检日志。本次单/双卡使用各自独立的a100_v4运行目录。

## 1. 代理和SSL（先于fetch）

新服务器默认代理文件：`/data/pub/z00919662/motion_deblur/proxy.md`。
文件尚未远程确认存在；如果代理环境已配置且GitHub可用，直接沿用。
否则读取已迁移的proxy.md，可用 `PROXY_FILE` 指定实际位置。
旧文件在 `/mnt/ssd1/z00919662/motion_deblur/proxy.md`，不要假设旧路径在新机存在。

使用其中有效配置设置 `http_proxy/https_proxy/HTTP_PROXY/HTTPS_PROXY` 及Git代理。
若含Markdown围栏，提取有效shell配置，不source整个Markdown。
禁止 `set -x`，不要把带凭据URL输出到日志、对话或GitHub。
按用户要求跳过SSL校验，clone/fetch前执行：

```bash
set +x
export GIT_SSL_NO_VERIFY=true
export CONDA_SSL_VERIFY=false
git config --global http.sslVerify false
```

curl下载如有需要用 `-k`；pip按实际主机配置 `--trusted-host`。
不重新下载已有数据和权重，不升级已工作的torch/CUDA。
若网络被代理阻断且缺少有效配置，报告需要迁移proxy.md及目标路径，不猜凭据。

## 2. 更新现有代码

先确认没有训练进程仍使用这个代码目录。若有，报告其身份，不在活跃训练中切换源码。
变量在同一bash会话使用；CodeAgent若分会话执行，重新设置固定变量。

```bash
set -euo pipefail
export ROOT=/data/pub/z00919662/motion_deblur
export CODE=$ROOT/benchmark_code_t3_fullframe_v3
export DATASET_BASE=/data/pub/z00919662/dataset
export ENV_NAME=deblur_runtime
RUN_BASE=$ROOT/runs/rtfocuser_shift_dst_t3_gopro_bsd_dvd_a100_v4
RUN1=${RUN_BASE}_1gpu
RUN2=${RUN_BASE}_2gpu
SELECTION=${RUN_BASE}_selection.json
BRANCH=agent/rtfocuser-shift-dst-t3-a100-v4
MIN_COMMIT=e774caae01d28f80b690d94b5de7c76abd8aacf9

cd "$CODE"
test -e .git
test -z "$(git status --short)"
git fetch origin "refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
git switch --detach "refs/remotes/origin/$BRANCH"
git merge-base --is-ancestor "$MIN_COMMIT" HEAD
test -s tools/prepare_rtf_t3_a100.py
test -s scripts/run_rtf_t3_a100_training.sh
test -s docs/CODEAGENT_RTFOCUSER_SHIFT_DST_T3_A100_20260910.md
test -z "$(git status --short)"
git rev-parse HEAD
```

使用本指令，不再执行旧20260909指令或旧 `/mnt/ssd1` 配置。
如果不是Git仓库或工作区有改动，保留文件并回传diff、未跟踪源码、当前HEAD。
不强制reset/checkout、不覆盖已有目录；服务器CodeAgent的源码改动必须同步中央后再用。

## 3. 激活现有环境，选择1或2张空闲卡

```bash
if ! command -v conda >/dev/null 2>&1; then
  for SH in \
    /data/pub/z00919662/anaconda3/etc/profile.d/conda.sh \
    /data/pub/z00919662/miniconda3/etc/profile.d/conda.sh \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    /opt/conda/etc/profile.d/conda.sh; do
    if [[ -r "$SH" ]]; then source "$SH"; break; fi
  done
fi
command -v conda
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
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
nvidia-smi topo -m
```

conda若不在候选位置，查找本机已有安装或 `CONDA_EXE`，不重新安装。
缺pytest/PyYAML等依赖时只补缺失项，不改变torch。不要假设新机torch仍是2.2.2/CUDA11.8。

根据实时结果设置 `GPU_A=<一张空闲A100物理卡号>`。
若另有空闲A100，设置 `GPU_B=<第二张空闲卡号>`，优先选择NVLink互联的两张。
否则 `GPU_B=`，继续单卡流程，不为等第二张卡阻塞训练。
不能默认0/1或旧GPU5空闲；不停止其他用户任务；不把小MIG分区当80GB完整显卡。

```bash
: "${GPU_A:?先选择第一张实时空闲A100}"
GPUS_TO_CHECK="$GPU_A${GPU_B:+,$GPU_B}"
CUDA_VISIBLE_DEVICES="$GPUS_TO_CHECK" python - <<'PY'
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    print(i, p.name, 'free/total GiB:', round(free/2**30,2), round(total/2**30,2))
    assert 'A100' in p.name and total >= 70*2**30
PY
```

## 4. 定位权重并生成单/双卡配置

准备脚本默认查找官方权重：

- `$ROOT/weights/GoPro_RT_Focuser_Standard_256.pth`
- `$ROOT/envs/RT-Focuser/Pretrained_Weights/GoPro_RT_Focuser_Standard_256.pth`
- `$ROOT/envs/rt_focuser_repo/Pretrained_Weights/GoPro_RT_Focuser_Standard_256.pth`
- `$ROOT/benchmark/weights/rt_focuser/GoPro_RT_Focuser_Standard_256.pth`

必须匹配SHA256：`6bc1310af51b2e47cf631c987ce7da1057aa469c3770d973195b2856014ad3fb`。
用户尚未提供新机权重位置。若在其他位置，通过 `--pretrained /实际路径` 传入，不重复复制。
若完全缺失，需要用户迁移到：
`/data/pub/z00919662/motion_deblur/weights/GoPro_RT_Focuser_Standard_256.pth`。
明确报告缺失与目标路径，不能替换为旧crop latest/best或其他模型权重。

默认识别dataset父目录下的 `GoPro/gopro/GOPRO_Large`、`BSD/bsd`、
`DVD/dvd/DeepVideoDeblurring_Dataset`。默认路径合适时：

```bash
PREP_ARGS=()
python tools/prepare_rtf_t3_a100.py --root "$ROOT" --dataset-base "$DATASET_BASE" \
  --run "$RUN1" --gpus 1 "${PREP_ARGS[@]}"
if [[ -n "${GPU_B:-}" ]]; then
  python tools/prepare_rtf_t3_a100.py --root "$ROOT" --dataset-base "$DATASET_BASE" \
    --run "$RUN2" --gpus 2 "${PREP_ARGS[@]}"
fi
export PRETRAINED=$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["pretrained"])' "$RUN1/a100_setup.json")
```

若实际子目录不同，先只读核对，再给相同的 `PREP_ARGS` 数组增加需要的参数，例如
`--gopro-root /实际根 --bsd-root /BSD训练根 --bsd-root /BSD验证根 --dvd-root /DVD实际根`，
以及需要时的 `--pretrained /实际权重路径`。root参数可重复，不改源码或手工改YAML。
如果缺BSD验证集，报告缺失，不把train用作val。

两套路径/数据必须相同，只有GPU数相关预算按表调整。
每个run生成 `runtime_config.yaml`、`a100_setup.json` 和 `audit/`。
准备脚本只解析路径/校验权重，不代表数据审计通过；重复相同配置允许，不覆盖已有不同配置。
若换run，必须同步更新RUN1/RUN2，避免拿旧预检启动新配置。

## 5. 测试、全量尺寸审计、复杂度

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q 2>&1 | tee "$RUN1/audit/unit_tests.log"
python tools/audit_rtf_t6_data.py --config "$RUN1/runtime_config.yaml" \
  --output "$RUN1/audit/dataset_audit.json" 2>&1 | tee "$RUN1/audit/dataset_audit.log"
python tools/check_rtf_t6_complexity.py --config "$RUN1/runtime_config.yaml" --pretrained "$PRETRAINED" \
  --output "$RUN1/audit/complexity.json" 2>&1 | tee "$RUN1/audit/complexity.log"
```

要求全部测试通过（本分支23项）、DATA_AUDIT_PASS、COMPLEXITY_GATE_PASS和权重覆盖率1.0。
中央本地验证：5项不依赖torch的路径/预算/尺寸/选择器测试通过，Python编译和shell语法通过。
中央运行时缺torch且依赖下载未完成，本轮没有本地重跑完整torch测试，也没有实测双卡DDP。
**因此服务器完整测试、双卡预检与smoke不可跳过，不得把此前18 passed当作本轮双卡已验证。**

审计逐一读取所有blur/GT图片头，输出train/val的 `native_resolutions`（分辨率/序列名/帧数），
发现配对尺寸不符或序列内尺寸变化会失败，不修改原图。像素级sanity仍是抽样，不宣称全量像素审计。
原服务器统计供迁移完整性核对：

| 域 | Train | Val |
|---|---|---|
| GoPro | 22序列/2103帧 | 11序列/1111帧 |
| BSD | 60序列/6000帧 | 20序列/3000帧 |
| DVD | 66序列/6208帧 | 5序列/500帧 |

DVD C0036/C0041的1080p帧必须保留。少了序列、被缩成720p或统计减少时先检查迁移。
参数量candidate=5710131、baseline=5864531；64×64 T3探针948874368 vs 962155648。
此为沿用的卷积MAC加估算时域算术口径，未计重叠窗口重复计算，不能当整段推理实测FLOPs。

## 6. 单卡原生预检与短测速

```bash
GPUS="$GPU_A" RUN="$RUN1" CONFIG="$RUN1/runtime_config.yaml" \
  bash scripts/run_rtf_t3_a100_training.sh preflight 2>&1 | tee "$RUN1/audit/fullframe_preflight.log"
GPUS="$GPU_A" RUN="$RUN1" CONFIG="$RUN1/runtime_config.yaml" \
  bash scripts/run_rtf_t3_a100_training.sh benchmark 2>&1 | tee "$RUN1/audit/benchmark.log"
```

预检每域最大原生分辨率，BSD→DVD→GoPro两轮，第二轮旋转90度；GPU EMA/optimizer常驻，
不empty_cache，完整forward/loss/backward/梯度检查/optimizer step/EMA验证。
单卡24个microbatch、6次optimizer更新，全部通过才可测速。
报告在 `$RUN1/audit/fullframe_memory_preflight.json`。

测速使用真实数据加载器，16次更新，舍弃前4次，统计后12次每个12帧更新的耗时。
测速不保存训练权重、不验证精度；结果在 `$RUN1/benchmark/benchmark.json`。
单卡预检失败时停止报告；不能靠双卡DDP把单卡装不下的T3空间拆开。

## 7. 有第二张空闲卡时做真实双卡预检与测速

确保两卡仍然空闲。只有设置了GPU_B才运行；两张卡的预检同时使用DDP，不是只测cuda:0。

```bash
DUAL_OK=0
if [[ -n "${GPU_B:-}" ]]; then
  if GPUS="$GPU_A,$GPU_B" RUN="$RUN2" CONFIG="$RUN2/runtime_config.yaml" \
      bash scripts/run_rtf_t3_a100_training.sh preflight 2>&1 | tee "$RUN2/audit/fullframe_preflight.log"; then
    if GPUS="$GPU_A,$GPU_B" RUN="$RUN2" CONFIG="$RUN2/runtime_config.yaml" \
        bash scripts/run_rtf_t3_a100_training.sh benchmark 2>&1 | tee "$RUN2/audit/benchmark.log"; then
      DUAL_OK=1
    fi
  fi
fi
```

双卡每rank累计2次：两轮共每rank12个microbatch、6次全局optimizer更新。
两个rank分别写 `fullframe_memory_preflight.rank0.json` / `.rank1.json`，必须均PASS。
预检覆盖真实NCCL/DDP反向和双卡显存；测速覆盖真实分布式采样与梯度累计。

若双卡NCCL/显存/数值测试失败，保留两个rank日志并报告原因，允许用已通过的单卡继续。
不自行修源码、改T、crop、resize、tile或变更allocator。检查失败双卡的torchrun子进程已退出，
释放本任务占用后才能继续；不杀别人的任务。

## 8. 根据实测速率选择，smoke后启动正式训练

```bash
SELECT_ARGS=(--single-run "$RUN1" --output "$SELECTION")
if [[ "$DUAL_OK" == 1 ]]; then SELECT_ARGS+=(--dual-run "$RUN2"); fi
python tools/select_rtf_a100_mode.py "${SELECT_ARGS[@]}"
export RUN=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["run"])' "$SELECTION")
export CONFIG=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"])' "$SELECTION")
export GPUS=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["gpus"])' "$SELECTION")
bash scripts/run_rtf_t3_a100_training.sh smoke 2>&1 | tee "$RUN/audit/smoke.log"
```

选择器验证两个benchmark的commit/config hash、权重、数据、预算一致。
双卡每更新实测至少快10%才选双卡；收益不足或第二卡不可用则用单卡。
这是短测速估计，不保证长训练保持同样倍数；后续报告实际速度。

所选模式smoke跑2次optimizer更新：单卡8个microbatch，双卡每rank4个microbatch；
每次更新后进行EMA验证，保存 `$RUN/smoke/checkpoints/latest.pth` 与best。
要求所有loss有限、保存成功。Smoke临时短调度只测链路，不能继承它的权重正式训练。
若选中的双卡smoke失败，不直接开正式训练；可报告并退回单卡，先完成单卡smoke再启动。

确认所选GPU仍然可用，从官方权重重新初始化：

```bash
nohup env ROOT="$ROOT" CODE="$CODE" CONFIG="$CONFIG" RUN="$RUN" \
  ENV_NAME="$ENV_NAME" PRETRAINED="$PRETRAINED" GPUS="$GPUS" \
  bash "$CODE/scripts/run_rtf_t3_a100_training.sh" formal \
  > "$RUN/launcher.log" 2>&1 &
TRAIN_PID=$!
echo "TRAIN_PID=$TRAIN_PID"
ps -fp "$TRAIN_PID"
```

启动器核对每rank预检、commit、配置hash、权重hash、GPU及torch/CUDA；有run锁防止重复启动。
旧V100报告不能替代。已有正式输出不能覆盖，见下一节resume。
监控至少100个每rank microbatch，每次等待不超过60秒，持续查看日志/nvidia-smi。
首批应为 `[1,3,3,H,W]`，无crop/resize；startup的world_size必须为所选1或2。
100 microbatches在单卡是25次更新、双卡是50次全局更新，报告时写清单位。
用“剩余optimizer更新数×实测秒/update”估计耗时，并说明额外验证成本。
首次正式checkpoint通常单卡10000/双卡5000 microbatches保存；未到保存点写NOT_YET_SAVED，
不能把smoke checkpoint当正式checkpoint。

## 9. 恢复与完训评估

恢复仅限所选模式正式run，GPU数量与配置保持匹配。换GPU/环境/commit需重新预检。
确认原进程已退出且GPU可用后：

```bash
nohup env ROOT="$ROOT" CODE="$CODE" CONFIG="$CONFIG" RUN="$RUN" \
  ENV_NAME="$ENV_NAME" PRETRAINED="$PRETRAINED" GPUS="$GPUS" \
  bash "$CODE/scripts/run_rtf_t3_a100_training.sh" resume \
  > "$RUN/resume_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

不从crop/T6/smoke恢复，也不把双卡checkpoint直接套单卡调度。
已有正式输出但没有checkpoint时保留并报告，不静默覆盖重训。

训练完成后用第一张已选择GPU同口径评估baseline与best：

```bash
export CUDA_VISIBLE_DEVICES=${GPUS%%,*}
for ARCH in rtfocuser_baseline t6; do
  if [[ "$ARCH" == rtfocuser_baseline ]]; then CKPT="$PRETRAINED";
  else CKPT="$RUN/checkpoints/best_balanced_psnr.pth"; fi
  python tools/eval_rtf_t6.py --architecture "$ARCH" --config "$CONFIG" \
    --checkpoint "$CKPT" --output "$RUN/eval_${ARCH}_native_t3.json" \
    --window 3 --temporal-overlap 2 --tile-size 0 --tile-overlap 0 --device cuda:0 --amp
done
```

`t6`只是兼容架构标识，实际window为3。报告各域及balanced的PSNR/SSIM/input PSNR/temporal残差。
不能与旧crop验证27.07dB直接比较。新机业务视频路径尚未提供，不为此阻塞当前训练启动。
若CodeAgent不能跨会话等待完训，明确告诉用户届时重新发送评估任务，不声称已安排自动跟进。

## 立即报告

```text
STATUS: IN_PROGRESS / FAILED
GITHUB_BRANCH / GIT_COMMIT / WORKTREE_CLEAN
SERVER / CODE / DATASET_BASE / RUN / CONFIG
ENV_NAME / PYTHON / TORCH / CUDA
GPU_INDICES / GPU_NAMES / FREE_AND_TOTAL_MEMORY_GIB / WORLD_SIZE
PROXY_SOURCE（路径或EXISTING_ENV，不给凭据） / SSL_CONFIGURATION
PRETRAINED_PATH / SHA256 / TARGET_COVERAGE
TRAIN_VAL_ROOTS_COUNTS_BY_DOMAIN / NATIVE_RESOLUTIONS / DVD_1080P_SEQUENCES
UNIT_TEST_RESULT / DATA_AUDIT_RESULT / COMPLEXITY_GATE_RESULT
SINGLE_PREFLIGHT / DUAL_PREFLIGHT_BY_RANK / PEAK_MEMORY_BY_DOMAIN_AND_RANK
SINGLE_SECONDS_PER_UPDATE / DUAL_SECONDS_PER_UPDATE / MEASURED_SPEEDUP
SELECTED_MODE / SELECTION_REASON（双卡失败或不可用也要说明）
CLIP_LENGTH / PER_RANK_ACCUMULATION / GLOBAL_FRAMES_PER_UPDATE
PER_RANK_TOTAL_MICROBATCHES / TOTAL_OPTIMIZER_UPDATES
SMOKE_STATUS / FORMAL_TRAIN_PID / FIRST_BATCH_SHAPE / CROP_APPLIED / RESIZE_APPLIED
CURRENT_PER_RANK_MICROBATCH / OPTIMIZER_UPDATES / LOSS_COMPONENTS / LR / ESTIMATED_COMPLETION
LATEST_FORMAL_CHECKPOINT / BEST_FORMAL_CHECKPOINT
SOURCE_CODE_MODIFIED: NO
HUMAN_ACTION_REQUIRED: YES/NO（YES说明缺失文件、目标路径或具体错误）
```

不能把预检/启动成功写成模型效果提升。CodeAgent若自行改过源码，回传完整diff和未跟踪源码，
由中央在GitHub修复后再继续，不基于未同步版本训练。
