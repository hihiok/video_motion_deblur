# CodeAgent：Shift-Net 压缩，1080p ≤500 GFLOPs，GoPro 目标35 dB，三数据集联合训练

## 目标、边界与已完成的工作

- GitHub：<https://github.com/hihiok/video_motion_deblur>
- 唯一执行分支：`agent/shiftnet-500g-mix-v1`。以本文件和 `shift500/` 为准。
- 新 checkout：`/data/pub/z00919662/motion_deblur/shift500_code_v1`。
- 独立结果目录：`/data/pub/z00919662/motion_deblur/runs/shift500_mix_v1`。
- 数据根：`/data/pub/z00919662/dataset`。环境：`deblur_runtime`。服务器：8×A100-SXM4-80GB。
- 两个模型都训练：quality 主模型、compact 低算力对照。默认顺序训练，每个模型1或2卡；最多占用2张确认空闲的卡。不要停止其他训练或改动原项目。
- 硬目标：1920×1080、每实际输出帧≤500 GFLOPs，GoPro test PSNR目标≥35.0 dB。35 dB尚未实现或验证，不能写成已达到。
- 三域所有训练阶段都参与，优化器每次更新精确采样4个clip：GoPro 2，DVD 1，BSD 1。不是三个域各训一个模型。
- 原生整帧、无random crop、无resize、无训练/推理tile。T=16，输出中间12帧；只有补齐4倍数的边界padding，计算loss前去除。
- 这是非因果窗口模型，需要未来帧；不能称为零延迟流式模型。
- 代码已经由ChatGPT准备。CodeAgent只下载、安装依赖、执行、检查和回报。不要自行写/改Python、shell、网络结构、损失或训练超参数。路径覆盖通过命令行完成，结果写到run目录。
- 如果发现代码问题，记录完整traceback和失败命令；若此前已有本地修改，导出`git diff --binary`、`git status --short`及新增文件清单，交回ChatGPT同步修复。不要自行提交修复，也不要丢弃本地修改。

## 1. Proxy与SSL（在首次git clone之前执行）

代理地址：`proxyhk.huawei.com:8080`。完整账号配置见随附私人执行文件，或已配置的本机proxy环境文件。公共仓库只引用环境变量；禁止把带凭据的运行时文件提交到GitHub，禁止`set -x`或输出完整环境变量。

```bash
# 先加载私人附件中的 http_proxy 配置；本机已有时可直接沿用。
: "${http_proxy:?先加载随附私人执行文件的proxy配置}"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$http_proxy"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$http_proxy"
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=true
```

按用户的公司代理要求跳过git SSL验证。HTTP代理的URL仍使用`http://`；HTTPS流量通过CONNECT。不要将相互覆盖的旧proxy.server.com示例保留为最终配置。需要下载时使用`curl -k`或`gdown --no-check-certificate`。不要更换服务器已有PyTorch/CUDA。

## 2. 获取代码和固定版本的上游

```bash
DEBLUR_ROOT=/data/pub/z00919662/motion_deblur
SHIFT_CODE="$DEBLUR_ROOT/shift500_code_v1"
SHIFT_RUN="$DEBLUR_ROOT/runs/shift500_mix_v1"
SHIFT_UPSTREAM="$DEBLUR_ROOT/envs/Shift-Net-pinned-450a4f2"
mkdir -p "$DEBLUR_ROOT/envs" "$SHIFT_RUN"
if [ ! -d "$SHIFT_CODE/.git" ]; then
  git clone --branch agent/shiftnet-500g-mix-v1 --single-branch \
    https://github.com/hihiok/video_motion_deblur.git "$SHIFT_CODE"
fi
cd "$SHIFT_CODE"
# 若不是期望分支或工作区非空，先报告并同步差异；不要stash/reset覆盖。
test "$(git branch --show-current)" = agent/shiftnet-500g-mix-v1
test -z "$(git status --porcelain)"
git pull --ff-only origin agent/shiftnet-500g-mix-v1
git rev-parse HEAD
if [ ! -d "$SHIFT_UPSTREAM/.git" ]; then
  git clone https://github.com/dasongli1/Shift-Net.git "$SHIFT_UPSTREAM"
fi
git -C "$SHIFT_UPSTREAM" checkout --detach 450a4f246dedccd306aa0bc02d615d797874e1ce
```

上游 `gshift_deblur2.py` SHA256固定为`b2b86200f00d37ce69dfa43e2f4f86696d6f8f46d715e6b131f0de4b120c3d4c`。加载器自动核验，不运行BasicSR setup，不编译自定义CUDA，不修改上游文件。版本不符先fetch该commit后checkout；不要改校验常量。

激活现有环境，先通过`conda info --base`定位conda初始化脚本；不要猜路径。

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deblur_runtime
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements_shift500.txt
export SHIFT500_UPSTREAM="$SHIFT_UPSTREAM"
python -m pytest -q tests_shift500
```

要求Python≥3.10、PyTorch≥2.2、CUDA可用，A100支持BF16。发现环境版本不符先报告，不破坏现有环境。

## 3. 权重与数据准备

Teacher固定为官方 **GoPro Ours-s**；不能用Ours+，不能把文件名当身份验证。加载器严格检查所有实际使用参数。上游存在4个从未执行的UNet，加载时明确排除；没有任意`strict=False`放行。

优先只读查找这些已有目录中的权重：

- `$DEBLUR_ROOT/benchmark/weights/shiftnet/`
- `$DEBLUR_ROOT/weights/shiftnet_ours_s_gopro/`
- `$DEBLUR_ROOT/envs/shiftnet_repo/pretrained_models/`
- `$DEBLUR_ROOT/benchmark_code_t3_fullframe_v3/`内已有权重引用

期望官方文件名`net_gopro_deblur_small.pth`。旧任务可能将small权重另存为`net_gopro_deblur.pth`，必须严格加载确认。若发现多个结构都兼容而来源不同，不能猜：优先官方重新下载，保存来源和SHA256。

官方链接：<https://drive.google.com/file/d/1WnFZRnXN9ZJebMZaAZF6a0c8f3CrLjR_/view>

```bash
# 本地没有已验证官方GoPro-small权重时执行；已有则直接设置SHIFT_TEACHER到它。
mkdir -p "$DEBLUR_ROOT/weights/shiftnet_ours_s_gopro"
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org 'gdown>=5'
gdown --no-check-certificate 1WnFZRnXN9ZJebMZaAZF6a0c8f3CrLjR_ \
  -O "$DEBLUR_ROOT/weights/shiftnet_ours_s_gopro/net_gopro_deblur_small.pth"
export SHIFT_TEACHER="$DEBLUR_ROOT/weights/shiftnet_ours_s_gopro/net_gopro_deblur_small.pth"
```

若下载被阻断且没有本地权重，唯一人工动作是从上面链接下载并上传到该路径；明确报告`BLOCKED_TEACHER_CHECKPOINT`和准确路径。不能改用随机teacher。

准备脚本支持GoPro/GOPRO/GOPRO_Large、DVD/DeepVideoDeblurring_Dataset、BSD名称。BSD若包含多个曝光档，使用明确的3ms-24ms根目录覆盖，避免把同一场景重复纳入多个域。先只读检查目录树确定根；不要解压/复制整个数据集。

```bash
python -m shift500.prepare --upstream "$SHIFT_UPSTREAM" \
  --teacher-checkpoint "$SHIFT_TEACHER" \
  --dataset-base /data/pub/z00919662/dataset --run "$SHIFT_RUN"
```

目录名不同则添加`--gopro-root PATH --dvd-root PATH --bsd-root PATH`，这不算修改代码。必须显式存在train/test分区。支持`split/scene/{blur,gt}`、`split/{blur,gt}/scene`和`{blur,gt}/split/scene`。严格按帧名配对、数值顺序、连续编号、分辨率检查；仅在官方train内部，将同一个GoPro acquisition的多个chunk整体划入内部train或val。官方train/test按完整clip遵循原划分，允许GoPro不同chunk来自同一acquisition，并在split_audit.json记录重叠名称。官方train中留10% acquisition作val，官方test只用于最终评测；完整clip名称跨train/test重复仍阻塞，GT文件SHA256跨split重复仍阻塞。该检查会读数据文件，首次可能较慢。

读取`data_audit.json`核对三域train/val/test数量和分辨率，包括DVD已知的1080p序列。任何缺失、配对错误或曝光混用直接报告，不能静默少训一个域。已有config时复用，不重复prepare覆盖。

## 4. 算力与GPU预检

`profile_{teacher,quality,compact}.json`已由prepare产生。要求两个student的`arithmetic_GFLOPs_per_output≤500`。

统计口径：1MAC=2FLOPs；16输入、12输出；包含所有实际执行卷积、偏置和主要逐元素算术；不将copy/roll/reshape计作浮点运算，但运行速度会受其影响。数字是计算图操作计数，不是A100速度实测。最终report另计窗口尾部丢弃输出导致的实际平均开销。对短场景不承诺每帧固定500G；必须报告。

只读运行`nvidia-smi`和compute-apps查询。只选择无其他用户计算进程、空闲显存至少70GiB的1或2张A100；优先2张。不要预设GPU0/1空闲，不杀任何进程。设置实际物理编号，例如`export CUDA_VISIBLE_DEVICES=2,3`，不是要求必须2、3。

```bash
export SHIFT500_CONFIG="$SHIFT_RUN/config.json"
# CUDA_VISIBLE_DEVICES必须已设置为检查后的物理GPU编号。
# 单步预检、baseline、训练、最终评测由该脚本依次完成。
mkdir -p "$SHIFT_RUN"
nohup bash scripts/run_shift500.sh > "$SHIFT_RUN/launcher.log" 2>&1 &
printf '%s\n' "$!" > "$SHIFT_RUN/launcher.pid"
```

启动前检查同一run目录是否已有活进程；活着就监控，禁止重复启动。
脚本先对两个student各自在GoPro/DVD/BSD中**训练集最大分辨率**的序列执行teacher前向、student前向、反向、optimizer step。记录峰值显存和形状；预检权重丢弃，正式训练重新初始化。
OOM时报告`NATIVE_PREFLIGHT_OOM`和序列/分辨率/显存。不要自动裁图、缩图、缩T或替换模型。DDP不会降低单个clip显存，增加GPU无法修复单卡microbatch OOM。

## 5. 训练过程与选择规则

每个模型180,000 optimizer updates（不是microbatches），全局每步4clips。1卡累积4次；2卡每卡累积2次，保持同样全局batch和采样比例。每步监督48张原分辨率输出帧，全部三域持续参与。

- 初始权重：保留下来的同shape参数从Ours-s复制；改变宽度的参数重新初始化，并输出`initialization.json`。这是缩宽/删重复模块后蒸馏重训，不是无损剪枝，不声称完整继承预训练质量。
- 0–2k：学习率warmup。之后从2e-4按cosine衰减到2e-6。AdamW，无weight decay，grad clip=1，BF16、activation checkpointing。
- 主损失：与官方PSNR目标一致的log-MSE；教师输出L1只在teacher比student更接近GT的像素参与，蒸馏权重随训练减小。
- 20k–40k逐步加入GT-relative帧差损失；不是直接让相邻输出相同，不强行抹平真实运动。该项无光流对齐，指标名称也明确标为unwarped，不能叫flow-aligned flicker。
- 每1k保存可恢复`latest.pth`；每5k在固定val采样评估；保存`best_gopro.pth`和`best_balanced.pth`。
- 最终正式排名用val选出的`best_gopro.pth`在三个test集上评测；不得看test分数继续挑checkpoint。balanced模型只作独立参考，不能把各域不同checkpoint拼成一行。
- quality完成后再训练compact。无需人工批准进入下一模型；任何真实阻塞报告具体原因。

启动后先确认至少20updates日志：loss/梯度有限、三个域配置正确、GPU占用、速度、剩余预计时间。首次5k验证后提供与同协议teacher holdout的PSNR差距。ETA用实际seconds/update计算，不提前保证几小时完成。若loss变差或PSNR未达标如实回报，不擅自改模型。

恢复：确认旧进程已退出，保持config/manifest/teacher不变，重新运行同一脚本，trainer自动读取latest。单独恢复指定模型：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 -m shift500.train \
  --config "$SHIFT500_CONFIG" --variant quality --resume "$SHIFT_RUN/quality/latest.pth"
```

## 6. 验证、推理、完整回报

默认test：所有帧，RGB浮点[0,1]、clamp、无uint8 round、无crop border、逐帧PSNR平均；16输入12输出，包含首尾复制padding，teacher和student完全相同。所有域输出逐序列/逐帧明细与输入PSNR，自动检查排名使用同一帧集合。

论文35.22不能直接当我们固定窗口teacher成绩：官方one_len=96使用100输入96输出，而且丢弃首尾和未满尾块。需要额外论文口径对照时三模型都用：

```bash
# 对teacher示例；student加对应variant和checkpoint，输出另存不能覆盖默认报告。
python -m shift500.evaluate --config "$SHIFT500_CONFIG" --variant teacher \
  --protocol official_chunks --outputs 96 --output "$SHIFT_RUN/official96_teacher.json"
```

100输入整帧显存可能很大；这个附加实验不是训练启动门槛，也不允许将其更高PSNR替换为16帧部署结果。我们的35 dB目标以部署的默认协议为准。

训练完成脚本自动产生`comparison.json`、`comparison.md`、三个域test明细和输入|输出|GT预览。用完整test报告判断：GoPro≥35.0并且实际平均1080p算力≤500才写`TARGET_MET`；否则`TRAINING_COMPLETE_TARGET_NOT_MET`。训练成功不等于目标达成。

业务PNG序列推理（输入路径使用本机已有、用户提供的业务folder；不猜测另外一个项目的目录）：

```bash
python -m shift500.infer --config "$SHIFT500_CONFIG" \
  --checkpoint "$SHIFT_RUN/quality/best_gopro.pth" \
  --input /absolute/path/to/business_png_folder \
  --output "$SHIFT_RUN/business_quality" --fps 30 --mp4
```

所有输入帧都会输出；跨场景重建独立窗口；无时间EMA。原帧率不是30时替换`--fps`。推理输出不含音频。

回报必须包括：STATUS、HUMAN_ACTION_REQUIRED及具体动作（正常启动无需额外人工操作）、GitHub分支/commit、工作区是否干净、Python/PyTorch/CUDA、GPU编号、数据数量/分辨率、teacher来源与SHA256、每模型参数量/GMAC/GFLOPs/输入输出帧数、预检显存、训练update/速度/ETA、三域val/test PSNR及teacher差值、unwarped temporal L1、最佳checkpoint路径、comparison与预览路径。未执行的项写NOT_RUN，不能补论文数值冒充实测。


## GoPro acquisition重叠阻塞的修复与恢复

旧版错误地要求官方GoPro train/test的acquisition名称必须完全不同。用户报告的官方22个train clip、11个test clip中，GOPR0384_11、GOPR0385_11、GOPR0868_11、GOPR0871_11、GOPR0881_11含不同chunk跨官方分区；这不应仅凭名称判成帧泄漏。修复后官方分区保持不动，重叠只作审计。其他域的acquisition检查保留。

CodeAgent现在执行：

1. 沿用本文件第1节和私人附件的proxy/SSL配置，激活已有deblur_runtime；不输出代理凭据。git操作之前仍执行：

```bash
export https_proxy="$http_proxy" HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$http_proxy"
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=true
cd /data/pub/z00919662/motion_deblur/shift500_code_v1
test "$(git branch --show-current)" = agent/shiftnet-500g-mix-v1
test -z "$(git status --porcelain)"
git pull --ff-only origin agent/shiftnet-500g-mix-v1
python -m pytest -q tests_shift500/test_splits.py
```

2. 检查原run目录。此次错误发生在make_manifest返回前，正常情况下还没有config.json；不要删除目录、手工编辑manifest或移动数据。若已经有config.json，先核对来源并回报，不覆盖它。
3. **原样重跑此前失败的`python -m shift500.prepare ...`命令**，复用已经验证的upstream、teacher-checkpoint、dataset-base、run和三个root覆盖参数。无需重新下载权重或重新选数据根。
4. 检查新`split_audit.json`：gopro的official_train_clips应为22、official_test_clips为11，shared acquisitions应与实际发现的5个名称一致；`gt_file_sha256_cross_split_check`为passed。同时检查data_audit.json中GoPro **train+val合计22**、test为11，DVD/BSD都完整；不要误要求内部train单独等于22。
5. 四项回归测试和数据审计通过后，按原第4节选择空闲GPU，执行scripts/run_shift500.sh，继续两个模型的最大整帧预检和正式训练。不要停在prepare成功；发现其他实际阻塞则回报准确错误，不能自行改代码或关闭其他检查。

回报：新commit、工作区状态、四项回归测试、split_audit.json的GoPro部分、三域clip数量，以及预检/正式训练是否已开始。没有新增人工数据整理操作。
