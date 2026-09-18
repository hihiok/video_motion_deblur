# CodeAgent：RealVDeblur 官方独立运动模糊视频推理

## 1. 本次范围

下载并对官方独立输入 `example/00_motion.mp4` 做一次完整 RealVDeblur one-step DMD 推理，输出恢复帧、恢复 MP4、输入/输出左右对比 MP4 和预览图。此任务不是训练，不是去模糊模型大比拼。

只处理这一个运动模糊视频。不下载 `01_defocus` / `06_defocus`，不下载网页 `*_blur_ours.mp4` 拼接结果，也不自动追加 `02_motion`–`05_motion`。不要裁掉输入的一半：本样例已经是独立输入。

- 工作根：`/data/pub/z00919662/motion_deblur`
- 代理配置：`/data/pub/z00919662/dataset/proxy.md`
- 输入根：`/data/pub/z00919662/dataset/RealVDeblur_official_motion`
- 输出：工作根 `runs/realvdeblur_official_motion_<时间>_<唯一后缀>/`
- 本任务代码库：`https://github.com/hihiok/video_motion_deblur.git`
- 本任务分支：`agent/realvdeblur-official-motion-demo-20260918`
- 本任务脚本：`tools/realvdeblur_motion/run.sh`
- 本任务离线检查：`tools/realvdeblur_motion/test_media.py`

**模型固定为 RealVDeblur，不是 RealViformer、RVRT、Shift-Net 或用户自己的其他模型。** 本次按当前对话上下文采用 RealVDeblur 官方推理。

## 2. Proxy / SSL：先配置再联网

完整读取 `/data/pub/z00919662/dataset/proxy.md`，使用其中真实的代理地址和凭据。不要直接 source 整个 Markdown；不要把文件内容、代理密码或环境变量全集打印到可共享日志。关闭 shell xtrace，设置 `umask 077`。

执行 proxy.md 中相关 `export http_proxy=...`、`export https_proxy=...`，并同步：

```bash
set +x
umask 077
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

Git 使用实际代理。用户已允许本任务跳过 SSL verification：无可信 CA / 仍报公司证书错误时，Git clone/fetch/pull 使用单次 `git -c http.proxy="$https_proxy" -c http.sslVerify=false ...`；下载脚本设置 `INSECURE=1`，其 curl 会使用 `-k`。不需要永久修改系统或全局 Python SSL。`git http.sslVerify=false` 不会关闭 Python requests 的证书校验。

可信 CA 存在时优先使用它；脚本可设 `CA_BUNDLE=/实际路径.pem`、`INSECURE=0`，pip 可用 `--cert`，Requests 可用 `REQUESTS_CA_BUNDLE`。不存在就不要假定 CA 有效，也不要因寻找 CA 无限重试。本任务允许只在下载进程关闭校验，但要在报告注明。

## 3. 获取本任务脚本：不切换训练工作区

本任务 checkout 建议为：`/data/pub/z00919662/motion_deblur/tools/realvdeblur_motion_demo_task`。
先检查目的目录；不存在时在独立目录 clone 本任务分支，存在时核对 origin/branch/工作区，只有身份一致且干净才 fast-forward pull。不得覆盖、reset、stash 或切换正在训练的目录。

```bash
# 只在目标不存在时执行；代理已按前文设置。
mkdir -p /data/pub/z00919662/motion_deblur/tools
git -c http.proxy="$https_proxy" -c http.sslVerify=false clone \
  --depth 1 --single-branch --branch agent/realvdeblur-official-motion-demo-20260918 \
  https://github.com/hihiok/video_motion_deblur.git \
  /data/pub/z00919662/motion_deblur/tools/realvdeblur_motion_demo_task
```

记录交付代码实际 commit。完整阅读本文件与 run.sh，不自行重写或改脚本。如果已有本地改动，提供完整 `git diff`（先移除凭据），由用户同步给 ChatGPT；不要隐藏改动继续跑。

## 4. 复用可用的官方代码、环境与权重

### 官方代码

真实模型仓库是 `https://github.com/OpenImagingLab/RealVDeblur.git`，固定 commit：

`63a2eb0ed4a22cb76b796a65d5d3052616352573`

不要误 clone `RBJin/RealVDeblur` 网站资源仓库。官方实际样例目录是 `example/`，不是 README 命令中误写的 `examples/`。

先只读查找当前工作根下已有 `envs/realvdeblur_repo` 或其他 RealVDeblur checkout。只有官方身份、commit 和 tracked files 均一致且没有未审查的本地代码，才复用。历史其他服务器路径不是当前已存在的证据。发现旧改动就保留，并提供 diff；不要覆盖。

没有可复用 checkout 时，在新的空目录 `/data/pub/z00919662/motion_deblur/envs/realvdeblur_official_motion_repo` 取官方代码。可按下列方式 sparse checkout，避免下载离焦或其他样例。目标已存在时先核对，不能盲目运行：

```bash
git -c http.proxy="$https_proxy" -c http.sslVerify=false clone \
  --filter=blob:none --no-checkout --single-branch --branch main \
  https://github.com/OpenImagingLab/RealVDeblur.git \
  /data/pub/z00919662/motion_deblur/envs/realvdeblur_official_motion_repo
cd /data/pub/z00919662/motion_deblur/envs/realvdeblur_official_motion_repo
git sparse-checkout init --cone
git -c http.proxy="$https_proxy" -c http.sslVerify=false sparse-checkout set model utils assert
git -c http.proxy="$https_proxy" -c http.sslVerify=false checkout --detach \
  63a2eb0ed4a22cb76b796a65d5d3052616352573
```

### 环境

先通过 `conda env list` 和历史运行记录检查已部署的 RealVDeblur 环境；`deblur_runtime` 只是候选，不要仅凭名称判定可用。记录实际 Python 路径、Torch/CUDA、选中 GPU 能力，并从官方 checkout 执行 `python inference.py --help` 验证依赖。

已有环境通过则复用，不执行 pip -U，不在 base / deblur_runtime 中安装、降级或升级依赖。本任务测试不需要 OpenXLab，也不要使用下载 REDS 的 venv 运行模型。

环境确实缺失或不兼容时，仅创建独立 `realvdeblur_demo` 环境，按该官方 checkout 的 requirements.txt 安装。官方 README 给出的测试组合是 Python 3.10 / PyTorch 2.4.0 / CUDA 11.8；实际GPU和驱动兼容性仍需核验，特别不要把 A100/V100 的 wheel 直接套到 Blackwell。优先复用已有兼容环境，不能为了运行本样例破坏其他任务。如果安装被网络或硬件兼容问题阻塞，报告具体问题，不让 CodeAgent 自行改模型。

### 权重

在当前用户的工作根、已知 weights/checkpoints/benchmark/weights 目录和 Hugging Face 缓存中做有限深度查找，避免扫描整台服务器。必须使用三件套：

```text
Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors
Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
realvdeblur_dmd.safetensors
```

脚本显式传入实际绝对路径，记录SHA256；不会把179MB任务权重当作整个模型。两件 Wan 文件需在同一目录；可以在本任务目录建立指向已校验原件的软链接，不移动或覆盖原件。

确实没有本地权重时，允许仅从官方源下载这三件缺失文件，存到本任务独立 weights 目录。不要重复下载已有文件，不下载11GB文本编码器或整个Wan仓库。官方来源：

- `https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/main`
- `https://huggingface.co/RBJin/RealVDeblur/tree/main`

用已有 `hf download` 或 curl -fL -C - 获取相应文件，代理/SSL按第2节；curl可使用这三个已确认的文件路径：

```text
https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/diffusion_pytorch_model.safetensors
https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth
https://huggingface.co/RBJin/RealVDeblur/resolve/main/realvdeblur_dmd.safetensors
```

下载前检查空间；以平台metadata和可获得的官方哈希校验，记录实际revision。不要把HTML错误页或LFS指针当成权重。若公网下载被阻断，明确告诉用户缺哪个文件、完整下载地址和应上传的绝对路径；不请求聊天中提供token或代理密码。

## 5. 运行：一张空闲卡，完整视频，官方参数

使用 `nvidia-smi` 检查后选择没有其他训练进程且显存充足的一张卡，优先当前机器的空闲 A100 80GB。不得固定抢GPU0，不得停止其他任务。只有没有可用卡时才报告 `BLOCKED_NO_IDLE_GPU`。

A100优先官方默认 `bfloat16`；已验证的环境需要 `float16` 时可显式设定并记录，不在失败后静默切换精度。不要用DataParallel，不启动新训练、不做消融。

运行四项离线检查，它们只验证shell和媒体处理，**不是GPU推理通过的证明**：

```bash
python tools/realvdeblur_motion/test_media.py
```

由CodeAgent根据前一步实际查找结果设置以下变量。下面的尖括号说明不是供用户手工猜路径；应由Agent发现后填写真实绝对路径：

```bash
export MODEL_REPO="<已验证的官方checkout绝对路径>"
export PYTHON_BIN="<可用RealVDeblur环境的bin/python绝对路径>"
export WAN_MODEL_DIR="<两件Wan权重所在的绝对目录>"
export DMD_CHECKPOINT="<realvdeblur_dmd.safetensors绝对路径>"
export GPU="<实际空闲物理GPU索引>"
export DTYPE=bfloat16
# 没有可信CA或证书仍失败时，本任务允许：
export INSECURE=1
bash tools/realvdeblur_motion/run.sh
```

脚本将：

1. 从固定官方commit下载**单独的**00_motion.mp4，并核对大小3254985 bytes和Git blob `9c020c496aa4b456474017b0f4108b0e927188c7`。保留partial文件，绝不覆盖身份不符的现有输入。
2. 使用ffprobe和实际解码检查尺寸、帧数和时间戳，按原CFR帧率处理。输入为VFR时会明确停止而不是伪称保留原时间戳。
3. 解码为RGB PNG以固定输入和输出对比的颜色通道；不先有损重编码视频。如果宽/高不能被16整除，只在右/下添加不足16像素黑色边界，推理后只删除自己添加的边界，**不裁掉原图、不resize**。这一padding策略不是声称复现网页的逐像素输出。
4. 在独立输出目录调用原版 inference.py：`stride=1`、`seed=0`、`num_inference_steps=1`、`enable_twm`、`temporal_window_size=21`、一张GPU。TWM=21是注意力窗口，**不是只输出21帧，也不是让Agent自行切成独立21帧clips**。
5. 核对输入/输出帧名和数量、完整解码，保存恢复PNG、恢复MP4、左右对比MP4以及首/中/末预览。对比左边是输入，右边是RealVDeblur。MP4为CRF16观看副本，RGB PNG是图像检查依据。观看视频不保留音频，报告中已明确标记。

本次直接完成这一段样例，不停在25帧smoke要求人工审批。若OOM，记录帧数、原始尺寸、GPU空闲显存、OOM阶段和日志；不擅自降分辨率、截短、改网络/模型权重、改TWM/精度或做未经核对的跨chunk状态复用。保留失败输出但不能称成功。

## 6. 视觉检查与报告

脚本成功运行只写 `INFERENCE_AND_ENCODING_COMPLETE`。CodeAgent仍需打开生成的首/中/末预览，并查看对比视频：检查黑帧、通道/颜色错位、严重形变、拖影、闪烁和明显chunk边界。没有实际查看就写 `VISUAL_REVIEW: NOT_PERFORMED`，不能冒充通过。遇到可疑结果不擅自改代码，回传日志和预览供用户判断。

这里没有清晰GT，不计算或伪造PSNR/SSIM，不把输入或作者的模型结果当GT。本次只能做视觉效果测试，不能保证与网页展示完全一致。

在本次RUN_DIR保存 `FINAL_REPORT.md`，至少包含：

```text
STATUS: SUCCESS / BLOCKED / FAILED / INFERENCE_COMPLETE_VISUAL_UNREVIEWED
TASK_COMMIT:
MODEL_UPSTREAM_COMMIT:
MODEL_REPO:
PYTHON_BIN:
TORCH_CUDA:
GPU:
INPUT_PATH:
INPUT_SHA256:
INPUT_RESOLUTION / MODEL_RESOLUTION / OUTPUT_RESOLUTION:
INPUT_FRAMES / OUTPUT_FRAMES / MP4_FRAMES:
FPS:
WAN_DIFFUSION_PATH_SHA256:
WAN_VAE_PATH_SHA256:
DMD_PATH_SHA256:
INFERENCE_PARAMETERS:
INFERENCE_WALL_SECONDS_INCLUDING_MODEL_LOAD:
PEAK_GPU_MEMORY: <从官方inference.log获取；缺失则写NOT_MEASURED>
OUTPUT_FRAMES:
OUTPUT_MP4:
COMPARISON_MP4:
PREVIEWS:
VISUAL_REVIEW:
MODEL_CODE_MODIFIED: NO / YES_WITH_DIFF
GT: NOT_PROVIDED
HUMAN_ACTION_REQUIRED: NO / <具体操作>
```

`inference.log`、`input_probe.json`、源文件/权重SHA256以及原始失败日志均保留。不要只报告下载完成或进程启动就声称推理完成。

正常情况下不需要用户提供AK/SK或手工写代码；用户只需要最终观看结果。只有实际遇到缺失文件、网络放行、没有空闲卡或必须人工处理的环境问题时，报告具体人工操作。任何CodeAgent本地代码改动必须用完整diff同步给用户转交ChatGPT，不能隐瞒。

## 7. 已核对来源与验证边界

官方代码与参数来源：
- `https://github.com/OpenImagingLab/RealVDeblur/blob/63a2eb0ed4a22cb76b796a65d5d3052616352573/inference.py`
- `https://github.com/OpenImagingLab/RealVDeblur/blob/63a2eb0ed4a22cb76b796a65d5d3052616352573/utils/data_load.py`
- `https://github.com/OpenImagingLab/RealVDeblur/blob/63a2eb0ed4a22cb76b796a65d5d3052616352573/model/src/realvdeblur_pipeline.py`
- `https://github.com/OpenImagingLab/RealVDeblur/tree/63a2eb0ed4a22cb76b796a65d5d3052616352573/example`

本交付在ChatGPT运行环境中通过4项离线检查（bash语法、缺必填参数失败、16倍数尺寸媒体往返、非16倍数padding后RGB像素无损恢复）。测试用临时合成素材，仅验证前后处理，不是用模型推理。该环境无法下载并解码官方MP4，也未在用户服务器执行GPU推理；正式结果由CodeAgent运行后报告。
