# CodeAgent：RT-Focuser 训练后评测与业务 MP4 对比（不做消融）

## 任务与完成条件

直接执行本文件，持续到结果齐全或出现具体阻塞。用户已授权本次评测和业务视频推理，无需再次询问是否开始。

1. 核验已完成训练的 `best_stable.pth`，记录实际保存 update、训练 commit、SHA256，并复制到新的评测目录供只读使用。
2. 用官方 RT-Focuser Standard 权重和训练后的完整时域模型，在官方 GoPro test 全量 11 序列 / 1111 帧上进行相同口径评测。已有 `blur_gamma` 和 `blur` 时分别报告，不混合两种输入。
3. 处理 `/data/pub/z00919662/motion_deblur/input` 目录直接包含的所有 MP4（扩展名大小写不敏感），生成两个模型的原尺寸输出、无损 PNG 和输入/官方/时域模型三栏对比 MP4。
4. 汇报真实结果及 PSNR 差异证据。业务视频没有 sharp GT，不给它编造 PSNR/SSIM，也不能仅凭训练 loss 或 `qualifies: true` 宣称业务闪烁已解决。

**不做消融实验，不继续训练，不关闭训练后模型的时域模块，不改变模型结构、损失、权重或训练数据。** 测试前“官方源码 vs 加载官方权重的零初始化包装器”仅用于核验实现一致性，不测试训练后模型移除模块的表现。

代码已由 Codex 写好。CodeAgent 负责拉取、运行和反馈，不自行修改源码。若确有代码缺陷，保留 traceback、最小复现和已完成结果，反馈给 Codex；若此前已改源码，先提交实际 diff 供同步，不能报告源码未改。

## 固定资源

| 项目 | 值 |
|---|---|
| GitHub | https://github.com/hihiok/video_motion_deblur |
| 本次分支 | `agent/rtfocuser-posttrain-eval-business-v1` |
| 训练代码基准 commit | `85d37edacd03a3feccd030c6518ca752d11cb2f8` |
| 新评测 checkout | `/data/pub/z00919662/motion_deblur/benchmark_code_rtf_posttrain_eval_v1` |
| 训练产出 | `/data/pub/z00919662/motion_deblur/runs/rtfocuser_causal_temporal_finetune_v1` |
| 模型 | 上述目录的 `checkpoints/best_stable.pth`；不可用 latest 偷换 |
| 官方权重 | `/data/pub/z00919662/motion_deblur/weights/GoPro_RT_Focuser_Standard_256.pth` |
| 官方 SHA256 | `6bc1310af51b2e47cf631c987ce7da1057aa469c3770d973195b2856014ad3fb` |
| 业务输入目录 | `/data/pub/z00919662/motion_deblur/input` |
| 环境 | 已验证可用的 `deblur_runtime`，Python 3.9 / torch 2.2.2 / CUDA 11.8 |

训练目录、原数据、原 MP4 只读。不要改动旧训练 checkout 或不相关的 T3/T6 任务。`best_stable` 的 update 可能不是 20000，以 checkpoint 内字段为准。

## 1. 代理、SSL 与代码获取

沿用服务器已授权的代理；代理账号密码只能来自已有环境变量或用户管理的私有 env 文件，不得写进 GitHub、报告或日志。不要开启 `set -x`。网络操作需包含以下代理及 SSL 设置，不修改全局 Git 配置。

```bash
set +x
set -euo pipefail
# 若已配置私有代理文件，DEBLUR_PROXY_ENV 指向该文件；不要在公共文件填密码。
if [ -n "${DEBLUR_PROXY_ENV:-}" ]; then
  source "$DEBLUR_PROXY_ENV"
fi
export http_proxy="${http_proxy:-${HTTP_PROXY:-}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY:-$http_proxy}}"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export GIT_SSL_NO_VERIFY=true
export GIT_CONFIG_COUNT=2
export GIT_CONFIG_KEY_0=http.sslVerify
export GIT_CONFIG_VALUE_0=false
export GIT_CONFIG_KEY_1=http.proxy
export GIT_CONFIG_VALUE_1="$https_proxy"

TASK_BASE=/data/pub/z00919662/motion_deblur
TASK_CODE="$TASK_BASE/benchmark_code_rtf_posttrain_eval_v1"
# 已存在时先检查来源、分支和工作区，不覆盖、不 reset --hard；干净同分支才可 fetch/ff-only。
if [ ! -e "$TASK_CODE" ]; then
  git clone --single-branch --branch agent/rtfocuser-posttrain-eval-business-v1 \
    https://github.com/hihiok/video_motion_deblur.git "$TASK_CODE"
fi
cd "$TASK_CODE"
git branch --show-current
git rev-parse HEAD
git status --porcelain
```

核对 HEAD 等于 Codex 本次交付消息中的 commit，工作区干净且分支正确；不在训练分支上直接运行本任务。记录实际 HEAD。此分支基于训练 commit，只新增评测、测试、上游参考源码及本指令。

## 2. 运行环境与资源

激活现有 `deblur_runtime`。若 `conda activate` 尚不可用，从现有 conda 安装的 `etc/profile.d/conda.sh` 初始化 shell。**不要为这次评测重装/升级 PyTorch、CUDA、NumPy 或整个 requirements。** 现有 torch 2.2.2 已在 A100 上通过训练；本任务不强制改成 2.4。

```bash
conda activate deblur_runtime
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
python -c 'import torch,numpy,cv2; print("torch",torch.__version__,"CUDA",torch.version.cuda,"available",torch.cuda.is_available()); print("numpy",numpy.__version__,"cv2",cv2.__version__); print(cv2.resize(numpy.zeros((8,8,3),numpy.uint8),(4,4)).shape)'
python -c 'import skimage; print("skimage",skimage.__version__)'
ffmpeg -version
ffprobe -version
ffmpeg -hide_banner -h bsf=setts
python tools/eval_rtf_temporal_posttrain.py --help
```

已修复的 OpenCV 是 `opencv-python-headless==4.8.1.78` 配合 NumPy 1.26，不要重新装 `opencv-python` 与之冲突。若仅缺 `scikit-image`，在现有环境补装 Python 3.9 兼容的 `scikit-image==0.22.0`，保留现有 NumPy 1.26。若缺 pytest，仅补 pytest 8.x。先查看依赖计划，避免升级现有可用环境。FFmpeg 需提供 `libx264`、`setts` 和 `-fps_mode`；若现有版本太旧，优先定位服务器已有兼容版本，不擅自修改系统安装。

运行小型 CPU 集成检查（无需真实数据和真实权重）：

```bash
python -m pytest -q tests/test_rtf_posttrain.py
```

选择 1 张当前空闲 A100，资源以实时检查为准，不抢占其他进程。无需 DDP：

```bash
TASK_GPU="$(python tools/select_rtf_temporal_gpus.py --max-gpus 1)"
export CUDA_VISIBLE_DEVICES="$TASK_GPU"
```

若 GPU 暂无空闲，可等待并报告状态，不停止其他用户任务。检查磁盘剩余空间；本任务保存解码 PNG、两套模型 PNG 和预览 PNG，1080p 长视频可占用较多空间，程序会按未压缩体积保守预估。不要自动删除训练结果、数据或用户视频腾空间。

## 3. 定位完整 GoPro test 与执行

首先读取原训练 `manifest.json` 的 `roots.gopro`，并检查实际 GoPro 路径下的 `test`；根路径若是 `train`，使用其父目录。若服务器迁移了数据，允许只读查找实际位置，然后仅修改命令行 `--gopro-root`。禁止把 train 的留出片段当官方 test。

```bash
TASK_RUN="$TASK_BASE/runs/rtfocuser_causal_temporal_finetune_v1"
TASK_GOPRO="$(python - <<'PY'
import json
from pathlib import Path
m = json.loads(Path('/data/pub/z00919662/motion_deblur/runs/rtfocuser_causal_temporal_finetune_v1/manifest.json').read_text())
p = Path(m['roots']['gopro'])
print(p.parent if p.name.lower() in ('train', 'training') else p)
PY
)"
TASK_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TASK_OUT="$TASK_BASE/runs/rtfocuser_posttrain_eval_business_$TASK_STAMP"
TASK_LOG="$TASK_BASE/runs/rtfocuser_posttrain_eval_business_$TASK_STAMP.log"
# 不提前 mkdir TASK_OUT：脚本要求一个尚不存在的新目录。
# 通过持久会话/tmux 或现有任务管理器运行，保持 stdout/stderr 日志并监控到退出。
python -u tools/eval_rtf_temporal_posttrain.py \
  --run "$TASK_RUN" \
  --official "$TASK_BASE/weights/GoPro_RT_Focuser_Standard_256.pth" \
  --gopro-root "$TASK_GOPRO" \
  --input-dir "$TASK_BASE/input" \
  --output "$TASK_OUT" \
  --device cuda:0 \
  --tasks gopro business 2>&1 | tee "$TASK_LOG"
```

`set -o pipefail` 已在开头启用，不能把 tee 成功当评测成功。命令启动后报告实际 PID、GPU、输入文件列表和日志路径；等待任务退出后再给最终结果。若用后台会话，要实际确认进程存在，不只给启动命令。

GoPro 局部失败时，程序仍会继续业务推理，最终返回非零并写 `EVALUATION_PARTIAL_FAILED`。某个 MP4 失败时继续其他 MP4。优先完成可执行部分，保留完整错误；不要把部分失败报告为全部成功。

所有结果写到新目录；不会覆盖既有评测。需要重试时换新目录，可用 `--tasks business` 或 `--tasks gopro` 只运行失败部分，避免重复已完成任务。即使仅跑 business，也保留 `--gopro-root` 参数占位，程序不会访问该数据集。

如果 checkpoint 内 manifest 路径在迁移后不可用，不修改原 manifest 或篡改 SHA；报告具体缺失文件。本任务的严格加载/来源检查失败时不能继续使用未知权重。

## 4. 固定评测口径与诊断

- 官方模型使用仓库新增的 `rtf_temporal/upstream_reference.py`，对应上游 `ReaganWu/RT-Focuser` 的 Git blob `c314f62633ebdbc5c5614cfc42f04e495d69fa57`。官方和训练后完整模型都 strict load；不忽略 missing/unexpected keys。
- 推理是 FP32，关闭 autocast 与 TF32。原尺寸 RGB [0,1]，仅在右侧/底部 replicate pad 到 16 的倍数，随后去掉 padding。没有 crop256、缩放、test-time augmentation、锐化或自动 gamma/曝光修正。
- GoPro 必须全量 11 序列 / 1111 帧，每张 blur 与 sharp 同名、同尺寸，按数字帧号排序。检查与微调训练帧的精确身份/GT 文件内容重叠；**官方 train/test 可以共享采集名前缀，不能仅因前缀相同就宣布泄漏或拒绝官方划分。**
- `blur_gamma` 为上游 dataset.py 使用的输入；本项目通用发现函数优先 `blur`。先以 manifest 实际路径为证据确认本次训练使用哪一种，不凭目录存在作结论。两种输入都完整存在时，各自跑同样的两个模型；这属于数据协议核对，不是模型消融。某种输入完全缺失时明确报告，禁止用另一种冒名替代。
- PSNR 先逐帧计算再平均；SSIM 使用 skimage 的 RGB / data_range=1 / 默认 7×7 窗口。不要直接与训练脚本的 uniform-window SSIM 混为同一实现。时域指标是 sharp-GT 光流对齐后的 GT-relative temporal L1，同时报告有效流比例。遮挡/切镜区域不凭低误差判定稳定。
- 时域 state 在整段视频中连续传递，不每 4 帧重置；首帧、切镜和分辨率变化时重置。官方单帧模型每帧独立。
- 原训练 GoPro 验证只取一条留出序列的中间 32 帧，且使用 BF16；24.90→26.35 dB 是该开发验证口径。新全量 FP32 数字不能直接减去旧验证数字称为本次提升。
- 上游公开验证代码包含随机 256 crop。因此本任务的固定整帧评测与论文 30.67 dB 不能预设为严格同协议。报告差异证据；不以“必须达到 30.67”作为硬门槛，不挑帧、不挑更高分输入合并结果。若完整 `blur_gamma` 上官方仍明显偏低，说明仍有复现差距，并列出未排除原因，不能自动归因于时域模块。

## 5. 业务视频交付

每个 MP4 单独重置状态并完整处理。目录已由用户明确提供；发现文件后直接执行，不再询问哪一条或是否运行。若该目录确实没有直接包含的 MP4，先只读检查一层子目录并汇报实际布局，不能重新声称“业务路径未知”。

每个视频子目录包含：

- `input_frames/`：源视频逐帧解码 PNG；作为对比输入留存。
- `official_frames/`、`best_stable_frames/`：两套原尺寸 RGB PNG。
- `official.mp4`、`best_stable.mp4`：相同 H.264 CRF16 编码参数，两者均保留完整帧序列。
- `comparison.mp4`：INPUT / OFFICIAL / TEMPORAL BEST 三栏预览；单栏缩放至不超过 640×360，仅用于查看，不用于定量指标。
- `business_report.json`：源文件 SHA、分辨率、帧数、时间戳、状态重置帧、编码后帧数及时间偏差。

源 MP4 不修改。默认交付对比视频无音轨。模型 PNG 保持解码尺寸；MP4 若遇奇数宽高，只在右/下补 1 像素满足 yuv420p，报告该事实。不同帧率的视频分别按其时间轴处理；VFR 不强行转固定 FPS；编码后验证相对 PTS 和总时长误差 ≤2ms、帧数一致。变帧率末帧时长也经过检查。若源视频显式标注 HDR/PQ/HLG，脚本会报告需要明确色调映射策略，保留错误而不悄悄转换为 SDR。

如具备视频查看能力，检查文字/边缘、运动物体、平坦背景、切镜附近是否有闪烁、重影、过度平滑或色偏，并记录具体文件与时间点；没有看过视频就注明“尚未人工视觉验收”，不虚构观感。无需额外网络上传业务内容。

## 6. 最终反馈模板

```text
STATUS: EVALUATION_COMPLETE / EVALUATION_PARTIAL_FAILED / EVALUATION_FAILED
HUMAN_ACTION_REQUIRED: YES / NO
GITHUB_REPO / BRANCH / ACTUAL_COMMIT:
WORKTREE_CLEAN / SOURCE_CODE_MODIFIED:
ENV / TORCH / CUDA / GPU:
OFFICIAL_SHA256 / STRICT_LOAD:
BEST_STABLE_SHA256 / ACTUAL_BEST_UPDATE / TRAINING_COMMIT / STRICT_LOAD:
OFFICIAL_IMPLEMENTATION_IDENTITY_MAX_ABS_ERROR:
TRAINING_GOPRO_INPUT_VARIANT_WITH_MANIFEST_EVIDENCE:
GOPRO_TEST_COUNTS_AND_MISSING_VARIANTS:
FULL_TEST_TABLE:
  variant | model(input/official/best_stable) | frames | PSNR | SSIM | temporal L1 | flow valid
PSNR_DISCREPANCY_CONCLUSION:
  已核实原因 / 未排除原因；不得将小验证集数字当官方全量测试。
BUSINESS_INPUTS:
  文件 | 解码尺寸 | 帧数 | FPS/CFR或VFR | 时长
BUSINESS_OUTPUTS_AND_CHECKS:
  每个视频的两套结果、三栏预览路径；帧数、最大PTS误差、总时长误差、重置数。
VISUAL_REVIEW:
  已查看的片段与观察；或明确尚未人工验收。
OUTPUT_ROOT / SUMMARY_JSON / LOG:
NO_ABLATION / NO_RETRAIN / ORIGINAL_RUN_UNMODIFIED: YES / YES / YES
ERROR_AND_REQUIRED_INPUT:
```

说明本次只完成评测和推理，不把“训练完成”“测试通过”“视频导出成功”混为“业务效果验收通过”。所有表格填真实测量值；不要引用 CodeAgent 上一轮训练汇总冒充新测试结果。

## 实现参考

- [上游 RT-Focuser 模型](https://github.com/ReaganWu/RT-Focuser/blob/main/model/rt_focuser_model.py)
- [上游 dataset.py](https://github.com/ReaganWu/RT-Focuser/blob/main/setup/dataset.py)
- [FFmpeg setts 文档](https://ffmpeg.org/ffmpeg-bitstream-filters.html#setts)：本脚本用其指定末帧 packet duration，并回读校验输出时间轴。

随本分支提供的 7 项 CPU 测试覆盖：官方/包装器一致性与非 16 倍数 padding、GoPro 双输入协议与配对失败、精确帧重叠检查、最佳权重快照与 manifest 校验、CFR/VFR 时间戳及末帧时长、完整业务视频输出。CPU 合成测试不替代新服务器上的真实权重和业务视频验收。
