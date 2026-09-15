# DSTNet+ Base → WaveShift 压缩版：双卡执行任务 V1

2026-09-15 更新：按用户最新要求，全部100k更新都混合GoPro/DVD/BSD；取消最后20k仅GoPro采样。
使用本次新commit，不继续沿用旧commit `e54bc164de08d9d9ab788f382787aa7078db6287` 启动新任务。
若旧RUN已生成config或checkpoint，不覆盖它们；尚未开训时可使用新的 `RUN=$ROOT/runs/dstplus_waveshift_500g_all3_v2`。
若旧任务已在训练，先回传当前update、config和checkpoint位置以便作者提供保留优化器状态的迁移，禁止重复启动或擅自从零重训。

## 1. 目标与执行边界

用户授权本任务直接执行，固定使用 **两张空闲 A100 80GB**。不要再次请求开始训练的确认。

- GitHub：https://github.com/hihiok/video_motion_deblur
- 分支：`agent/dstnetplus-waveshift-500g-v1`
- 本文件：`CODEAGENT_DSTPLUS_WAVESHIFT_500GFLOPS_2GPU_V1.md`
- 先获取用户启动消息给出的固定 commit，在独立 checkout 中执行本文件。
- 只运行 `dst500` 与 `scripts/run_dst500_two_gpu.sh`。父分支的 Shift500 四卡/八卡任务不适用于这里。
- 不启动消融，不停止已有训练，不挪用有计算进程的 GPU。
- 作者已经准备全部源代码。CodeAgent 不自行写/改模型、训练、评测代码。
- 如遇代码错误：保留 traceback、日志、`git diff --binary`、修改文件清单与文件本体，回传作者修复后从 GitHub 拉取。
- 不覆盖或丢弃本地已有改动，不执行 `git reset --hard` / `git clean`。

验收必须同时满足：

1. `1920×1080`、每个实际输出帧 **≤500 GFLOPs**，乘法和加法各计1次。
2. 完整官方 GoPro test：11序列、1111帧，RGB8 全帧逐帧 PSNR 均值 **≥33.00 dB**。

当前是待训练候选，不是已经达到33dB的权重。只有 `final_test.json` 同时满足两项，才能报告 `TARGET_MET`。
训练完成但未达标必须报告 `TARGET_NOT_MET`，并将产物发回作者继续分析，禁止把训练成功写成指标达标。

## 2. Proxy 与 SSL（clone / fetch 前完成）

使用用户已经配置的公司代理。凭据从服务器现有配置读取，不回显到日志，不写入 Git 提交。
常见位置：激活环境后的 `$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh`；
或者 `/mnt/ssd1/z00919662/motion_deblur/proxy.md` 中的代理设置。
后者是说明文档，读取其中已授权的代理设置，不把整份 Markdown 当 shell 执行。

环境变量应覆盖以下四种大小写；代理 URI 形式为
`http://<username>:<URL-encoded-password>@proxyhk.huawei.com:8080`。
使用已有 `http_proxy` 或 `https_proxy`，没有时再从上述本机配置读取：

```bash
set +x
export http_proxy="${http_proxy:-${https_proxy:-}}"
test -n "$http_proxy" || { echo 'PROXY_MISSING: read the existing local proxy configuration'; exit 1; }
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$http_proxy"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$https_proxy"
# 用户明确要求：公司 HTTPS 检查环境跳过 SSL verify。
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=true
# DDP 本机通信不要经过公司代理。
export NO_PROXY="localhost,127.0.0.1,::1,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"
```

下载命令已使用 `curl -k`。不要在日志中执行 `env`、`set` 或打印完整 proxy URL。
如代理配置确实不存在，报告准确缺失文件及需要用户补充的动作；不要猜测凭据。

## 3. 环境与路径

服务器默认：

```text
ROOT=/data/pub/z00919662/motion_deblur
DATA=/data/pub/z00919662/dataset
Conda=deblur_runtime
代码=$ROOT/dstplus_waveshift_500g_v1
输出=$ROOT/runs/dstplus_waveshift_500g_v1
teacher=$ROOT/weights/DSTNetPlus_base_gopro.pth
upstream=$ROOT/envs/dstnetplus_500g_upstream
```

克隆作者仓库到上述独立代码目录；如果已存在且干净，fetch 后 checkout 用户给定 commit。
先完整阅读本文件再执行启动命令。不要操作旧的 `benchmark_code_t3_fullframe_v3` 或正在运行的 Shift500 checkout。

```bash
# 使用本机实际 conda 安装，不修改其他项目环境。
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deblur_runtime
python -c 'import sys,torch; print(sys.version); print(torch.__version__,torch.version.cuda)'
python -c 'import numpy,PIL,einops; print("dependencies ready")'
```

支持现有 Python3.9 / torch2.2.2+cu118 或 torch2.4.x。不要为了任务升级 torch/CUDA。
仅在缺少普通依赖时安装 `requirements_dst500.txt` 对应包；若该环境正在被其他进程使用，克隆一个专用环境后补依赖。

### 数据路径检查

- GoPro：指向包含明确 `train` / `test` 的根目录；必须使用 `blur`，不是 `blur_gamma`。
- 自动支持 `split/scene/blur` 和 `split/blur/scene`，GT 可为 `sharp` / `gt` 等；逐帧名字、时序、尺寸严格配对。
- GoPro要求22个官方train clip，以及精确11个test名称/1111帧。
- 官方 GoPro train/test 中 acquisition 名称重合只记录；不会再因 acquisition 标签重合错误停止。
- 按训练 acquisition 分出约10% holdout；测试集不参与损失、选 checkpoint 或早停。
- BSD 必须选择 **3ms-24ms** 专用根目录，禁止把多曝光目录一起当成一个训练集。
- `BSD_ROOT` 的路径需要包含 `3ms24ms`、`3ms-24ms` 或 `3ms_24ms`。如已有纯3ms-24ms数据但根名为BSD，可建清楚命名的符号链接；先核实曝光身份，不重命名伪装。
- DVD / BSD 缺少 test 时仍可训练。已有不完整 test 只评测实际可用帧，完整性写入 audit，不冒充完整榜单。
- 不复制大数据集，不生成全量 teacher 图像缓存；manifest 和 teacher 只做读取/哈希。

## 4. 最终模型与来源

上游真实 DSTNet+ Base，固定 commit：
`sunny2109/DSTNet-plus@54363c15d8b924aa1ae56b8f835c1f0289954e95`。

参考 idea：
https://chatgpt.com/share/6aa0d643-755c-83ec-8563-22b4e44b65ed

核实的 WaveShift 源码：
`hihiok/video_motion_deblur@463c3dd3bd80a4048b77b25811b6b154a08b530f`
中的 `nanovsr_deblur/models/network_nanovnr_waveshift_pagf.py`。

本次保留/修改：

| 部位 | 处理 |
|---|---|
| 特征通道、RGB stem/head | 保留64通道，复制原版权重 |
| Haar / PAGF | 保留原版 DSTNet+ 的 DWT/IWT 和两个传播方向的 PAGF |
| 主干3×3卷积 | 每个方向均匀保留13/15个残差块；每个块仍有两个空间3×3卷积，改为 depthwise+pointwise |
| 权重压缩初始化 | 对原版每个输入通道的3×3卷积核做 rank-1 SVD 分解，初始化上述可分离卷积；记录保留能量 |
| Progressive dynamic conv | 保留三尺度、每尺度3块、逐像素动态深度卷积；生成器分组8，kernel投影分组8，并重排以保留每组的三尺度输入 |
| LL 时空 shift | 参考代码的两层 GSTS，空间半径2/4；融合 RepConv 使用16组并在部署时折叠 |
| 高频边缘 | 原版HF分支后追加可学习 Laplacian depthwise + 分组投影，边缘残差scale从0开始 |
| 长视频状态 | 每个T6 chunk内双向传播，chunk间清零；无无限hidden传递 |

不能从参考的约0.43M Nano主干直接推断GoPro≥33dB。
分享中最新全GoPro结果为26.7738dB，因此这里迁移模块而保留DSTNet+结构、权重来源并蒸馏。

### 算力口径

`reports/dst500/profile_student.json`：**485.98699008 GFLOPs/输出帧 @1080p**。
部署参数602,502个可学习参数，含固定Haar系数为603,526个。
统计包括RGB stem/head、DWT/IWT、动态核生成与实际滤波、双向传播、GSTS融合、HF边缘。
sigmoid按4个标量基本操作、激活按2个名义操作、池化比较也保守计入。
shift/reshape/数据搬运不计浮点运算，但其内存带宽开销不为零。

每次6进6出，所有输出用于评测，因此完整clip FLOPs除以实际6个输出。
最后不足6帧直接短chunk推理，不补无效时间帧。禁止换成6进1出后仍沿用本数字。
本方案与参考的T6中心帧单输出调度不同；它保留所有6个输出，chunk边界的时域上下文不均匀，需检查边界质量。

原版同一计数器为约2835.38GFLOPs。上游注释44.6148G来自THOP的MAC口径；
按像素比例放大约1411G，不能当作严格乘加各计一次的FLOPs。
同时保留用户记录“1411G”和统一计数结果，不能把两种数字混在同列算压缩率。
本地是静态形状执行的运算计数，不是GPU实际延迟/吞吐实测。

## 5. 唯一训练方案（不做消融）

- 两卡DDP，每卡 microbatch=1，累积4次，全局每次参数更新8个clips。
- 每个clip为6张连续原生整帧；所有6张计算loss。**不crop、不resize、不空间tile。**
- 总100,000次参数更新。这是本次压缩配方预算，不是官方600k配方复现，也不是33dB保证。
- 前80k：每8个clips按GoPro6/DVD1/BSD1，原版GoPro Base冻结在线蒸馏。
- 后20k：同一个模型仍按GoPro6/DVD1/BSD1混合精修，仅关闭蒸馏；三个数据集全程参与训练。
- 监督loss：Charbonnier + 0.01×空间梯度L1；蒸馏0.2×RGB L1，仅在teacher逐像素比student更接近GT处作用，避免跨域错误强制传递。
- BF16 AMP，梯度checkpoint、梯度裁剪1.0，AdamW。
- 学习率前80k从1e-4余弦衰减至1e-6，前500次warmup；后20k从2e-5衰减至1e-6。
- 每1000次保存latest；每5000次在固定train holdout位置验证，以GoPro holdout RGB8 PSNR保存best。
- holdout每序列固定两个T6位置，只用于选择；最终1111帧test全部评测。
- 训练holdout没有送入student优化，但官方预训练teacher可能见过这些训练帧，记录这一限制。
- 数据采样由update/micro/rank确定，断点恢复加载模型、AdamW状态与update；不因换物理GPU改变全局batch。
- 50次真实更新后报告seconds/update和预计剩余时间。首次不要凭空承诺训练小时数。

## 6. 启动

确认 `BSD_ROOT` 的准确曝光和目录，然后在激活环境的代码目录执行：

```bash
export ROOT=/data/pub/z00919662/motion_deblur
export DATA=/data/pub/z00919662/dataset
export GOPRO_ROOT="$DATA/GoPro"
export DVD_ROOT="$DATA/DVD"
# 根据实际路径填写，禁止混入其它曝光档。
export BSD_ROOT="$DATA/BSD/3ms24ms"
# GPUS不设置时自动挑选恰好两张空闲80GB卡；已有任务的卡会被排除。
# 可手工指定两张已确认空闲的物理卡，例如 export GPUS=0,1。
export RUN="$ROOT/runs/dstplus_waveshift_500g_v1"
mkdir -p "$RUN"
nohup bash scripts/run_dst500_two_gpu.sh > "$RUN/launcher.log" 2>&1 &
echo $! > "$RUN/launcher.pid"
```

先检查同一RUN的launcher.pid是否仍存活，禁止重复启动。
路径可以通过上述环境变量覆盖；发现服务器实际路径不同，在日志说明映射即可，不改源代码。

启动脚本依次完成：

1. 两张空闲GPU检查、干净代码检查。
2. 固定官方源码、下载或复用Base GoPro权重并strict load。
3. 数据manifest、哈希、原版和压缩模型统一FLOPs检查。
4. 同协议原版teacher全测试集评测。GoPro输入应接近25.6401dB，teacher需≥33dB，否则先报告管线/权重问题。
5. 双卡最大尺寸预检：每个domain选择最大面积训练/holdout序列，两卡各完成4次累积的实际前反向和AdamW更新，包含teacher显存；结果不作为训练checkpoint。
6. 两卡100k训练，断点从latest恢复。
7. best_gopro_val.pth在全部可用三域test评测，GoPro必须完整1111帧。结果写CSV和JSON。
8. 只有PSNR与FLOPs双达标才输出TARGET_MET。

预检OOM或数值异常：保存错误和两个GPU显存信息，报告 `PREFLIGHT_FAILED`；
不要擅自crop、resize、减T、改变精度、删支路、增加卡数或修改代码。无需为每步正常执行另设人工确认。

## 7. 验证状态与产物

作者本地已通过：真实CPU前反向/所有可学习参数梯度有限、SVD初始化、Haar可逆、RepConv折叠等价、
原生奇数尺寸、普通/部署checkpoint strict roundtrip、RGB8已知PSNR、完整模型485.99GFLOPs计数。
双进程Gloo本地测试因当前运行环境禁止其网络设备操作而未能启动；**没有把双卡通信或GPU显存标记为已验证**。
服务器的真实两卡预检为必经步骤；本地没有GoPro训练结果。

主要产物位于RUN：

```text
config.json / manifest.json
profile_student.json / profile_teacher.json
teacher_test.json / teacher_test.csv
preflight_rank0.json / preflight_rank1.json
initialization.json
training.jsonl / launcher.log / launcher.pid
latest.pth / best_gopro_val.pth
val_*.json / val_*.csv
training_complete.json
final_test.json / final_test.csv / final_test_previews/
```

首次启动后应回报：真实Git HEAD、两张物理GPU/UUID、环境、数据帧数、teacher SHA、统一FLOPs、预检峰值显存，
以及50次更新后的吞吐和ETA。继续跟踪长期任务，不要把“成功启动”报成“训练完成”。

最终回报：

```text
STATUS: TARGET_MET / TARGET_NOT_MET / PREFLIGHT_FAILED / BLOCKED
GITHUB_URL / BRANCH / HEAD:
SOURCE_CODE_MODIFIED_BY_CODEAGENT: NO（若YES，附diff和文件供作者同步）
GPU_PHYSICAL_IDS / GPU_UUIDS: 必须恰好两张
ENV / PYTHON / TORCH / CUDA:
TEACHER_CHECKPOINT / SHA256:
STUDENT_CHECKPOINT / SHA256 / UPDATE:
INPUT_OUTPUT_POLICY: T6_IN_T6_OUT_NONOVERLAP_RESET
STUDENT_PARAMETERS / GFLOPS_1080P_PER_OUTPUT:
GOPRO_SEQUENCES / FRAMES / INPUT_PSNR_RGB8 / OUTPUT_PSNR_RGB8:
DVD / BSD_AVAILABLE_TEST_COUNTS_AND_METRICS:
PEAK_GPU_GIB_RANK0 / RANK1 / SECONDS_PER_UPDATE:
OUTPUT_DIR:
HUMAN_ACTION_REQUIRED: YES / NO
EXACT_USER_ACTION_IF_ANY:
```

若没有需要用户介入的问题，明确 `HUMAN_ACTION_REQUIRED: NO`。
若未达33dB，至少回传final_test.json、per-frame CSV、所有val指标、training.jsonl、config.json和initialization.json。
无需用户手工重写代码；作者依据回传结果在GitHub更新。
