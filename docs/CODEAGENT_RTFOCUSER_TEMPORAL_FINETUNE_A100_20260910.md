# RT-Focuser 预训练权重 + 因果时域模块：三数据集整帧微调

## 任务与代码来源

完整执行本文件，不依赖聊天记忆。用户已授权准备并启动训练，无需在正常步骤后再次请求批准。
先从 GitHub 拉取本分支，再执行已写好的代码。CodeAgent 不得自行实现、修改模型、loss、训练脚本或测试。

- 仓库：https://github.com/hihiok/video_motion_deblur
- 分支：`agent/rtfocuser-causal-temporal-finetune-v1`
- 本指令：https://github.com/hihiok/video_motion_deblur/blob/agent/rtfocuser-causal-temporal-finetune-v1/docs/CODEAGENT_RTFOCUSER_TEMPORAL_FINETUNE_A100_20260910.md
- 原有代码：`/data/pub/z00919662/motion_deblur/benchmark_code_t3_fullframe_v3`
- 本次独立 checkout：`/data/pub/z00919662/motion_deblur/benchmark_code_rtf_temporal_v1`
- 数据根目录：`/data/pub/z00919662/dataset`
- 环境：`deblur_runtime`
- 服务器：8 张 NVIDIA A100-SXM4-80GB，优先选择两张空闲卡，只有一张空闲则单卡。不得占用其他任务的 GPU。
- 输出：`/data/pub/z00919662/motion_deblur/runs/rtfocuser_causal_temporal_finetune_v1`

旧训练、旧 checkpoint、原始数据均保留；本次不授权停止旧任务。所有运行配置、数据清单和日志写在输出目录。

## 代理与 SSL（首次 fetch/clone/download 前执行）

公司网络使用 HTTP CONNECT 代理。优先加载现有 conda 激活脚本或用户私有代理文件，不打印内容。
账号密码只能存在服务器私有环境文件（权限 0600）；不得写入公开 GitHub、日志、截图或 CodeAgent 回传。
私有代理 URL 的格式为 `http://<账号>:<URL编码密码>@proxyhk.huawei.com:8080`，端口的协议是 http。

```bash
set +x
# 先激活已存在的 deblur_runtime，通常会自动加载 activate.d/proxy_env.sh。
# 如果 conda 不在 PATH，从当前用户已有 conda 安装定位 conda.sh；不要硬套旧服务器路径。
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deblur_runtime

if [ -z "${http_proxy:-${https_proxy:-${HTTPS_PROXY:-}}}" ]; then
  if [ -f "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh" ]; then
    source "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh"
  fi
fi
PROXY_URL="${http_proxy:-${https_proxy:-${HTTPS_PROXY:-}}}"
if [ -n "$PROXY_URL" ]; then
  export http_proxy="$PROXY_URL"
  export https_proxy="$PROXY_URL"
  export HTTP_PROXY="$PROXY_URL"
  export HTTPS_PROXY="$PROXY_URL"
  git config --global http.proxy "$PROXY_URL"
  git config --global https.proxy "$PROXY_URL"
fi
unset PROXY_URL

# 用户明确要求跳过 SSL verify；在 clone 前设置。
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=true
# torchrun 本地通信不能走公司代理。
export no_proxy="localhost,127.0.0.1,::1${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"
```

后续 curl 使用 `-k`，wget 使用 `--no-check-certificate`；pip 必要时添加对应的 `--trusted-host`。
不要输出 `env`、`git config --list`、代理 URL 或私有脚本内容。若私有代理尚未配置且直连不通，报告
`HUMAN_ACTION_REQUIRED: YES`，明确请用户在服务器配置代理文件；不要把凭据提交到公开仓库。

## 1. 拉取独立分支

```bash
ROOT=/data/pub/z00919662/motion_deblur
REPO="$ROOT/benchmark_code_rtf_temporal_v1"
RUN="$ROOT/runs/rtfocuser_causal_temporal_finetune_v1"
BRANCH=agent/rtfocuser-causal-temporal-finetune-v1
if [ ! -e "$REPO" ]; then
  git clone --branch "$BRANCH" --single-branch https://github.com/hihiok/video_motion_deblur.git "$REPO"
fi
cd "$REPO"
test "$(git remote get-url origin)" = https://github.com/hihiok/video_motion_deblur.git
test -z "$(git status --porcelain)"
test "$(git branch --show-current)" = "$BRANCH"
git pull --ff-only origin "$BRANCH"
git rev-parse HEAD
```

如目录已有改动、分支不匹配或有本任务正在运行，不执行 reset/clean/强制覆盖，也不重复启动。
回传 `git status --short` 和相关 diff（脱敏），让用户同步给 Codex。环境/路径可通过现有 CLI 解决的继续解决。

## 2. 环境与已有官方权重

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
python -c "import numpy, PIL, yaml, cv2, pytest"
```

要求 PyTorch >=2.4，CUDA 可用，A100 支持 BF16。保留现有 torch/CUDA/conda 环境；不升级 torch、不重装 CUDA。
仅缺 Python 小依赖时安装 `requirements_rtf_temporal.txt` 中缺少的包，不强制升级已有正常依赖。
例如缺包时使用 `python -m pip install -r requirements_rtf_temporal.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org`。
不要因为 CPU 测试机使用不同 torch 小版本而改服务器环境。

官方权重：`GoPro_RT_Focuser_Standard_256.pth`

- SHA256：`6bc1310af51b2e47cf631c987ce7da1057aa469c3770d973195b2856014ad3fb`
- 默认放置：`$ROOT/weights/GoPro_RT_Focuser_Standard_256.pth`
- 官方来源：https://github.com/ReaganWu/RT-Focuser/tree/main/Pretrained_Weights
- 下载地址：https://raw.githubusercontent.com/ReaganWu/RT-Focuser/main/Pretrained_Weights/GoPro_RT_Focuser_Standard_256.pth

先在 `$ROOT/weights`、`$ROOT/envs`、旧 benchmark 的权重目录寻找已存在的同名文件，核验 SHA256。
如缺失，允许下载：

```bash
mkdir -p "$ROOT/weights"
curl -k -fL --retry 3 \
  https://raw.githubusercontent.com/ReaganWu/RT-Focuser/main/Pretrained_Weights/GoPro_RT_Focuser_Standard_256.pth \
  -o "$ROOT/weights/GoPro_RT_Focuser_Standard_256.pth.partial"
echo '6bc1310af51b2e47cf631c987ce7da1057aa469c3770d973195b2856014ad3fb  '"$ROOT/weights/GoPro_RT_Focuser_Standard_256.pth.partial" | sha256sum -c -
# 仅在核验通过且最终文件不存在时移动；不要覆盖不同已有文件。
test ! -e "$ROOT/weights/GoPro_RT_Focuser_Standard_256.pth"
mv "$ROOT/weights/GoPro_RT_Focuser_Standard_256.pth.partial" "$ROOT/weights/GoPro_RT_Focuser_Standard_256.pth"
```

下载被拦截或 SHA 不符时明确报告文件来源、预期 SHA、目标路径以及用户需手动上传什么；不要用随机权重、T3/T6 权重替代。

## 3. 本地功能测试

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python -m pytest -q tests/test_rtf_temporal.py
```

`test_against_upstream_reference` 默认跳过；若服务器已有官方源码，将其路径赋给 `RTF_OFFICIAL_REFERENCE` 后再跑该项。
其余测试在 CPU 上完成，包括真实 Standard 结构、零残差等价、跨帧梯度、切镜重置、冻结 BN、两阶段训练/续训和数据配对。
测试中的 32×32 合成图片仅用于代码校验，正式训练锁定原生整帧。

若准备双卡，额外运行已有双进程集成测试：

```bash
RTF_TEST_DDP=1 python -m pytest -q tests/test_rtf_temporal.py::test_data_audit_and_integration_resume
```

它用 CPU/Gloo 验证两阶段切换、DDP 梯度归约、单/双进程样本预算及输出接近性。
若 Gloo 或 NCCL 通信是服务器环境问题，可改用单卡继续；代码逻辑错误、非有限数值不可跳过。
Codex 当前容器禁止 Gloo socket，故双进程功能与 A100/NCCL 尚待服务器实测，不能宣称已通过。

## 4. 三数据集审计和配置

```bash
python tools/prepare_rtf_temporal.py \
  --root "$ROOT" \
  --dataset-base /data/pub/z00919662/dataset \
  --run "$RUN"
```

如果已有 `runtime_config.yaml`，不要重复准备或覆盖，核验后继续。
默认识别 GoPro/gopro/GOPRO_Large、BSD/bsd、DVD/dvd/DeepVideoDeblurring_Dataset。
如实际目录含额外嵌套或多个 BSD 曝光档，先只读查看目录，再用脚本已有参数指定真实根目录：
`--gopro-root PATH --bsd-root PATH --dvd-root PATH --pretrained PATH`。
BSD 优先使用本项目已准备的 3ms–24ms 配对集；不得把多档混成同一个序列，也不能按排序强配不一致文件名。

审计自动执行：

- 必须有明确的 train 目录；严格同名 blur/GT 配对，帧号连续。
- 扫描全部配对图片尺寸，发现 DVD 1080p 也保留，禁止裁剪、缩小、排除大图。
- 从各数据集官方 train 中按采集组固定留出约 10% 验证；GoPro 同一采集的多个切片不跨集合。
- 检查训练/验证 GT 完整文件哈希交叉重叠；官方 test 不参与优化或 checkpoint 选择。
- 生成 `manifest.json`、`data_audit.json`、`runtime_config.yaml`，记录来源和 SHA256。

出现布局不支持、帧号缺失、尺寸不一致等，报告真实目录结构和错误；不要自行修改数据或写转换代码。
清单生成后不得改动原图。后续需更换数据时新建 run 并重新审计。

## 5. 选择空闲 GPU，执行整帧反向显存预检

```bash
export CUDA_VISIBLE_DEVICES="$(python tools/select_rtf_temporal_gpus.py --max-gpus 2)"
test -n "$CUDA_VISIBLE_DEVICES"
printf 'Selected GPU indices: %s\n' "$CUDA_VISIBLE_DEVICES"
bash scripts/run_rtf_temporal.sh "$RUN/runtime_config.yaml" preflight \
  > "$RUN/preflight.log" 2>&1
```

选择器要求空闲 A100、无计算进程、>=70,000 MiB 可用显存。再次检查所选 GPU 没有被其他任务占用。
双卡通信不通可选一张空闲卡重做预检。没有空闲卡就报告资源等待，不停止其他人的进程。

预检对 GoPro、BSD、DVD 各自最大分辨率序列，分别执行“仅时域模块”和“全网微调”的完整 forward/backward/optimizer step。
每张卡保留一个 T=4 clip，单卡梯度累积两次，双卡各一次；全局每次更新均为 2 clips / 8 帧。
预检会实际建立 Adam 状态，记录 peak allocated/reserved，不把两张 80GB 当成一张 160GB。
预检权重全部丢弃，正式训练重新加载官方权重。

必须出现 `PREFLIGHT_PASS`。OOM 时明确回报失败 domain/shape/GPU/显存；不要改 crop、resize、T、AMP 设置或删数据来绕过。
当前代码已启用分模块 activation checkpointing 和固定 BN；不需要临时魔改。

可在同一张空闲卡记录完整 720p 算力：

```bash
python tools/profile_rtf_temporal.py --height 720 --width 1280 --device cuda:0 \
  --output "$RUN/complexity_720p.json"
```

## 6. 正式训练

```bash
nohup bash scripts/run_rtf_temporal.sh "$RUN/runtime_config.yaml" train \
  > "$RUN/training.log" 2>&1 < /dev/null &
TRAIN_LAUNCH_PID=$!
printf '%s\n' "$TRAIN_LAUNCH_PID" > "$RUN/launcher.pid"
```

训练协议：

| 项目 | 设置 |
|---|---|
| 模型 | 完整 RT-Focuser Standard + 1/4 分辨率、32 通道门控历史残差 |
| 初始化 | 官方权重严格完整加载；新增残差输出零初始化 |
| 输入 | 同序列连续 T=4，原生整帧；仅 pad 到 16 倍数，输出裁回原尺寸 |
| 时域 | 仅历史，无未来帧；clip 内反向传播，clip 间清空状态 |
| 数据 | GoPro/BSD/DVD 按全局样本编号轮流，各占 1/3；clip 内增强一致 |
| Stage 1 | 更新 1–2,000：冻结主干，训练新增模块，初始 LR 1e-4 |
| Stage 2 | 更新 2,001–20,000：全网络微调，主干初始 LR 1e-5、模块 1e-4 |
| BN | 两阶段均固定 running mean/variance；Stage 2 可训练 affine 参数 |
| Loss | MSE + λ×GT-relative 光流对齐时域 L1；λ 在前 2,000 次从 0 升至 0.01 |
| 光流 | GT 半分辨率 Farneback，正确缩放回原分辨率；前后向一致性、越界、光度误差 mask |
| 切镜 | 输入 64×36 预览上的平均差 >0.30 时清空历史、不算跨切镜 temporal loss |
| 训练精度 | BF16 autocast；MSE、warp、时域 loss 使用 FP32 |
| 优化器 | AdamW、分阶段余弦 LR、梯度裁剪 1.0 |
| 保存/验证 | 每 500 更新 latest；每 1,000 更新固定连续片段验证 |

这里的 iteration/update 都指一次 optimizer.step，总计 20,000 次，Stage 2 为 18,000 次；不是旧版 microbatch 口径。
推理无需光流，仅保留 32 通道 1/4 分辨率历史特征。门控没有显式特征对齐，大运动仍可能有拖影，需用结果验证。
本次并非从头训练，也不是旧 Shift/DST T3/T6 架构。

启动后继续观察直到基线验证完成、至少 20 次正式更新且进程仍存活。检查 loss/grad 有限，确认 full-frame/T4、两阶段配置和 GPU 数正确。
根据稳定训练更新耗时估计剩余时间，加上验证耗时；不要引用其他模型速度推算。
不得把训练已启动报告为已完成。若可继续监控，跟进到 Stage 2 首次验证；无需等用户确认再跨阶段。

## 7. 验证与模型选择

训练前同一固定验证片段运行原版空间模型，保存 `baseline.json` 和 input/output/GT 预览。
验证按序列连续传播状态（每序列最多中间连续 32 帧，每 domain 最多 2 序列），不是每四帧重置；这是开发验证，不是全测试集论文分数。

每个域分别比较 PSNR、uniform-window SSIM、GT-relative aligned temporal L1、有效光流 mask 比例。
只有每域 PSNR 下降 <=0.2 dB、每域 temporal L1 不差于原版、每域 mask 有效比例 >=5%，且三域平均 temporal L1 严格改善时，才保存 `best_stable.pth`。
重建 loss 和 temporal loss 数值尺度不同，0.01 是保守初值，不代表改善百分比；日志保留各项原始值。
未出现 best_stable 时报告“尚无满足条件的 checkpoint”，不得将 latest 自动称为更优模型。
本次不用 GAN/VGG loss、不强行提高锐度、不做输出 EMA 平滑；业务码流视觉效果仍需检查细节与拖影。

## 8. 断点续训与故障回传

只对本任务进程发 SIGTERM 可请求在当前更新结束后保存；如用户未要求停止，不主动停止。
恢复时保持 config、manifest、源码 commit 不变；GPU 数可在重新预检后由 2 改 1，样本预算仍一致：

```bash
nohup bash scripts/run_rtf_temporal.sh "$RUN/runtime_config.yaml" train \
  --resume "$RUN/checkpoints/latest.pth" \
  > "$RUN/resume.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RUN/launcher.pid"
```

重启前确认旧 launcher/所有子进程已退出。不要只看旧 pid 文件；核对命令行、启动时间和输出目录，避免 PID 重用。
源码有变化时，不把旧 preflight 当成新代码通过；不要通过修改 checkpoint 的 SHA 字段绕过检查。
如果 CodeAgent 已自行改过源码，停止继续改动并回传脱敏 diff/patch、commit、日志；用户需要将它们发给 Codex 同步修复。

## 9. 训练后的业务序列推理（输入存在时执行）

在 `$ROOT` 的 input/input_frames/benchmark/input_frames 或既有业务推理记录中定位该项目的 452 帧业务序列。
路径不存在或多个候选无法确定时，报告需要用户提供路径；这不阻塞三数据集训练。
确认 source FPS，未知时仅保存 PNG，不猜 25/30 fps。

```bash
# INPUT_FRAMES 是已确认的业务 PNG 目录；FPS 是从原始视频/记录读取的真实帧率。
python tools/infer_rtf_temporal.py --input "$INPUT_FRAMES" \
  --checkpoint "$ROOT/weights/GoPro_RT_Focuser_Standard_256.pth" --official \
  --output "$RUN/business_official" --fps "$FPS"
python tools/infer_rtf_temporal.py --input "$INPUT_FRAMES" \
  --checkpoint "$RUN/checkpoints/best_stable.pth" \
  --output "$RUN/business_temporal" --fps "$FPS"
```

缺少 best_stable 时可用 latest 做明确标注的诊断输出，但不能冒充已改善模型。完整视频内逐帧持续保存状态，仅序列开头、切镜或分辨率变化时重置。
推理是 FP32 参考输出；保存每帧映射和 reset 索引，不进行颜色自动归一化。

## 最终回报格式

```text
STATUS: TRAINING_RUNNING / TRAINING_COMPLETE / BLOCKED / FAILED
HUMAN_ACTION_REQUIRED: NO / YES（YES 时明确要用户做什么）
GITHUB_REPO / BRANCH / COMMIT:
WORKTREE_CLEAN / SOURCE_CODE_MODIFIED:
ENV / TORCH / CUDA:
GPU_INDICES / WORLD_SIZE:
PRETRAINED_PATH / SHA256 / STRICT_LOAD:
TRAIN_VAL_COUNTS_PER_DOMAIN / RESOLUTIONS:
PREFLIGHT_PER_DOMAIN_AND_STAGE / MAX_ALLOCATED_RESERVED:
BASELINE_METRICS:
TRAIN_PID / CURRENT_STAGE / OPTIMIZER_UPDATES / SECONDS_PER_UPDATE / ETA:
LATEST / BEST_STABLE:
PSNR_AND_TEMPORAL_CHANGE_PER_DOMAIN:
BUSINESS_OUTPUTS（如输入可用）:
ERROR_AND_REQUIRED_INPUT（如有）:
```

正常情况下用户只需把任务交给 CodeAgent，不需手动写代码、准备新数据集或下载光流权重。
