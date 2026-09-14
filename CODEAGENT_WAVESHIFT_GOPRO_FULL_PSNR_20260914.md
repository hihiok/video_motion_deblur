# CodeAgent：WaveShift 150k 完整 GoPro test PSNR

## 任务
直接执行本任务，不训练、不修改模型/权重/源代码，不评测其他模型或数据集。
模型 NanoVNRWaveShiftPAGF / waveshift_edge，step=150000。
数据根目录固定为 /data/pub/z00919662/dataset/GoPro。
本次正式结果为 6进1出 T6_CENTER_RESET_PER_TARGET，不是2进2出或6进2出对比。
不要执行旧 run_nanovnr_waveshift_t6_inference_all.sh（只取首序列且还会跑其他数据集）。
不要使用 infer_video_nanovnr_waveshift_pagf.py 的无限状态传递。

仓库：https://github.com/hihiok/video_motion_deblur
分支：agent/waveshift-gopro-full-psnr-20260914
新增入口：nanovsr_deblur/eval_waveshift_gopro_full.py
作者代码提交：ebcb9f6dbe2cda4ee62165fe8da667948521aad4
以用户启动消息给出的完整最终 SHA 为准；必须记录 git rev-parse HEAD。

## 环境：RVRT
沿用本 WaveShift 已确认的 RVRT 环境。激活命令：
```bash
set -euo pipefail
set +x
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f /mnt/ssd1/z00919662/anaconda3/etc/profile.d/conda.sh ]; then
  source /mnt/ssd1/z00919662/anaconda3/etc/profile.d/conda.sh
else
  echo "BLOCKED: conda activation script not found"
  exit 2
fi
conda activate RVRT
test "${CONDA_DEFAULT_ENV}" = RVRT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
which python
python -c 'import sys, torch, cv2, numpy; print("Python",sys.version); print("torch",torch.__version__); print("CUDA runtime",torch.version.cuda); print("CUDA available",torch.cuda.is_available()); print("OpenCV",cv2.__version__); print("NumPy",numpy.__version__)'
```

如果新服务器只有 deblur_runtime 而没有 RVRT，报告两个环境的实际可用情况和版本，交用户/作者确定；不自行切换，不升级 PyTorch/CUDA，不自动安装依赖。RVRT 是当前 WaveShift 的已确认环境，不从旧 RVRT 交接文件硬编码版本矩阵。

## Proxy 与 SSL（必须在 fetch/clone 前完成）
使用用户已配置的含凭据代理；禁止 set -x、打印代理 URL 或把凭据写入 GitHub commit/日志。GitHub 仓库为公开仓库，因此本文件只加载服务器上的代理配置，不存密码。
```bash
set +x
if [ -f "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh" ]; then
  source "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh"
fi
export http_proxy="${http_proxy:-${HTTP_PROXY:-}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY:-$http_proxy}}"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
test -n "$http_proxy"
test -n "$https_proxy"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$https_proxy"
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=true
```
用户企业代理地址为 proxyhk.huawei.com:8080，HTTP CONNECT 代理 URL scheme 用 http；凭据沿用用户已配置值。如激活后没有代理配置，报告缺少的本地配置位置，不要求把密码提交 GitHub。不要把 Markdown proxy.md 当 shell 脚本直接 source。

## 获取代码
在 /data/pub/z00919662/motion_deblur 下建立独立 checkout waveshift_gopro_eval_20260914，不切换其他训练目录分支，不覆盖已有目录。
```bash
export REPO=/data/pub/z00919662/motion_deblur/waveshift_gopro_eval_20260914
test ! -e "$REPO"
git clone --single-branch --branch agent/waveshift-gopro-full-psnr-20260914 \
  https://github.com/hihiok/video_motion_deblur.git "$REPO"
cd "$REPO"
git rev-parse HEAD
git status --short
```
按用户启动消息的最终 SHA 进行 checkout --detach，并确认 HEAD 精确相等。若目录已存在，检查 origin 和工作区是否干净；可复用同一任务的干净 checkout，fetch 后再按 SHA checkout，不能 reset --hard 或清理未知修改。
正式运行前，工作区必须无源代码修改。

## Checkpoint 定位（只读）
唯一目标是下列历史权重迁移后的同一个文件：
/mnt/ssd1/z00919662/motion_deblur/runs/nanovnr_waveshift_pagf_fullframe_t6_bsd3ms24ms_20260907/amp_recovery_20260907/train/step_0150000.pth

先检查新服务器候选：
/data/pub/z00919662/motion_deblur/runs/nanovnr_waveshift_pagf_fullframe_t6_bsd3ms24ms_20260907/amp_recovery_20260907/train/step_0150000.pth

可在 /data/pub/z00919662/motion_deblur 内用 rg --files -g step_0150000.pth 只读查找。
只能选上述 run / amp_recovery 的权重；不能用其他实验的同名文件。
设置 CHECKPOINT 为实际文件绝对路径，记录 sha256sum。
脚本会检查 architecture、variant=waveshift_edge、step=150000、model_config 与实例化模型完全一致，以及 strict load。
有多个不同内容候选或没有权重时，报告候选路径和 SHA256；缺权重时明确请用户迁移此文件。不要下载任意预训练模型替代。

## GPU
使用一张空闲 GPU。不要使用物理 GPU 6，不中止或占用其他训练任务。
执行以下原样代码，按 UUID 选择无计算进程且空闲显存最大的卡：
```bash
unset CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(python - <<'PY'
import csv
import subprocess

def query(kind, fields):
    return subprocess.check_output(
        ['nvidia-smi', '--query-' + kind + '=' + fields,
         '--format=csv,noheader,nounits'], text=True)

busy = {line.strip() for line in query('compute-apps', 'gpu_uuid').splitlines() if line.strip()}
rows = csv.reader(query('gpu', 'index,uuid,memory.free').splitlines())
candidates = [(int(free.strip()), uuid.strip()) for index, uuid, free in rows
              if index.strip() != '6' and uuid.strip() not in busy]
if not candidates:
    raise SystemExit('BLOCKED: no idle GPU other than physical GPU 6')
print(max(candidates)[1])
PY
)
export CUDA_VISIBLE_DEVICES
printf 'Selected GPU UUID: %s\n' "$CUDA_VISIBLE_DEVICES"
```
选择后再次确认卡仍空闲；脚本内使用 cuda:0（映射到所选UUID）。如 OOM，保留日志返回，不改 FP16、resize、tile 或模型实现。

## 全量执行
在同一个已激活 RVRT、已设置 REPO/CHECKPOINT/CUDA_VISIBLE_DEVICES 的 shell 中执行：
```bash
test -s "$CHECKPOINT"
test -d /data/pub/z00919662/dataset/GoPro/test
cd "$REPO/nanovsr_deblur"
python -m py_compile eval_waveshift_gopro_full.py infer_nanovnr_waveshift_t6_center.py
sha256sum "$CHECKPOINT"
export OUTPUT_DIR="/data/pub/z00919662/motion_deblur/runs/waveshift_gopro_full_psnr_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$(dirname "$OUTPUT_DIR")"
test ! -e "$OUTPUT_DIR"
python -u eval_waveshift_gopro_full.py \
  --dataset-root /data/pub/z00919662/dataset/GoPro \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0 \
  2>&1 | tee "${OUTPUT_DIR}.log"
```
不需要额外人工确认或另跑 smoke；首帧检查属于正式评测。用持久终端任务/会话保持任务运行并跟踪至结束，不因一次命令等待超时而杀进程或重复启动。
只有进程结束且 summary.json.status=PASS、frames=1111、sequences=11 才能报告完成。
若服务器的 split 布局不同、存在双套GT、数量不符，报告实际结构/数量，不能偷偷改 discover 或把部分集写成完整集。
只生成指标与少量无损检查图，不生成全部输出帧/视频。

## 固定评测协议
- test 下所有 11 序列、1111 个目标帧；逐序列独立，输入/GT 按完全相同文件名一一配对。
- 使用 blur，不使用 blur_gamma；GT 为同序列 sharp 或唯一 gt/GT。
- RGB8 PNG，原始整帧，无 resize、crop/shave、Y通道、TTA、后处理平滑。
- B=1、T=6，输出 position=3（0-based），每个目标帧 prev_forward_feat=None，不传递返回状态。
- 序列边缘 reflect padding，不跨序列、不重复计分、不删除边缘目标。
- FP32，AMP/TF32关闭，部署 RepConv；网络内部既有 padding 允许，最终输出尺寸必须等于输入。
- 模型结果先 clamp[0,1]，乘255并按 floor(x+0.5) 转 RGB8；直接与GT像素计算PSNR，不从MP4读回。
- PSNR = 10 log10(255² / RGB全像素MSE)。逐帧求PSNR再对1111帧算术平均，不能先平均MSE或不按帧数加权平均序列分数。
- 同时报告 input RGB8 PSNR、output RGB8 PSNR、gain dB；所有结果属于此新完整集协议。
- GT-vs-GT=inf；配对、shape、raw finite与异常变黑检查。出现异常输出立即停止，summary=FAILED。
- 29.2676 dB、29.3790 dB 是历史子集/不同输出口径，不预设本次应达到这些数值，也不拿旧子集直接排名。

## 交付与报告
输出：
- manifest.json：全部输入/GT配对
- per_frame.csv：1111行逐帧输入/输出PSNR、亮度、暗像素率、raw范围、窗口索引
- per_sequence.csv：每序列帧数和均值
- summary.json：全局指标、模型/环境/版本/耗时/显存/覆盖率
- 每序列首/中/末3张 input | output | GT 无损PNG
- OUTPUT_DIR.log

最后返回：
```text
STATUS:
GITHUB_URL:
GITHUB_HEAD:
SOURCE_CODE_MODIFIED_BY_CODEAGENT: NO
ENVIRONMENT: RVRT
PYTHON / TORCH / CUDA:
GPU_UUID / GPU_NAME:
CHECKPOINT / CHECKPOINT_SHA256:
CHECKPOINT_STEP: 150000
DATASET_ROOT: /data/pub/z00919662/dataset/GoPro
INPUT_VARIANT: blur
SEQUENCES / FRAMES:
INFERENCE_POLICY: T6_CENTER_RESET_PER_TARGET
PSNR_PROTOCOL: RGB8_FULL_FRAME_MEAN_PER_FRAME
INPUT_PSNR_DB:
OUTPUT_PSNR_DB:
GAIN_DB:
DARK_OR_NONFINITE_OUTPUT:
ELAPSED / PEAK_GPU_GIB:
OUTPUT_DIR:
HUMAN_ACTION_REQUIRED:
EXACT_USER_ACTION_IF_ANY:
```
附逐序列表。成功时无需人工操作；检查图可供用户查看。
若源代码已有或被误改，不能静默继续：附 git diff、涉及文件及原始HEAD，交作者同步修正，不自行写补丁。
