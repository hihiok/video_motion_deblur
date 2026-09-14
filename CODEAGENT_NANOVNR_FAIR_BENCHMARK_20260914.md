# NanoVNR：三域完整测试集 PSNR benchmark 与 BSD GT 恢复

## 目标和边界

仓库：https://github.com/hihiok/video_motion_deblur.git

本次分支：`agent/nanovnr-fair-benchmark-20260914`。
基于已核验的 `21b6bb669bb8e67eb26c30597493d9fe521ef113`。
使用本消息给出的新提交号固定 checkout；不要拿旧分支 HEAD 代替。

只测试，不训练、不改网络、不改权重、不继续筛选 checkpoint。不执行此前的训练 MD。
之前“前两个视频”的结果仅是样本检查；本次默认 **三个 test 下全部视频的全部原生帧**。
不允许 SKIP_BSD、max-clips、首100帧截断、空间裁剪/缩放/tiling、颜色交换、亮度拟合、GT引导对齐。
使用 125k Nano checkpoint，参数 414923；旧结果不能代填。

交付代码：
- `benchmark_v2/bsd_gt.py`：原始 Sharp 源清单、严格匹配、原文件复制。
- `benchmark_v2/data.py`：冻结全量测试帧 manifest、每帧 SHA256、可选训练/测试交叉检查。
- `benchmark_v2/evaluate.py`：Nano / Shift-Net Ours-s 推理、统一指标、外部PNG评分和对比。
- `run_nanovnr_fair_benchmark.sh`：不联网的执行入口。
- `tests/test_benchmark_v2.py`：单元/合成端到端测试。

禁止 CodeAgent 重写这些代码。若发现 bug，回传报错和涉及的源码/diff。
允许创建纯数据 JSON 配置、来源映射和报告；这不等于允许修改 Python/SH。
旧 `run_nanovnr_test_first2.sh` 的本地 SKIP_BSD 改动仅归档同步，不用于本次测试。

## 1. 先区分三种比较，不能混为一张“论文SOTA榜”

A. **统一实测（默认）**：同一个冻结 manifest、完整原生帧、同一RGB8指标、相同数值精度；各模型使用预先固定并公开的时域协议。Nano 是 chunk15/forward carry，Shift Ours-s 是 one_len16/前后各2帧 halo。这是现有 checkpoint 的实际效果对比，不是等训练数据或等时域窗口的架构对照。

B. **严格同上下文比较（已提供，按用户需要运行，不自动加做）**：`CONTEXT=16`。每个目标帧都接收完全相同的16个输入，过去8帧+当前+未来7帧，序列边界相同 reflection，每个目标重新清空状态，仅对同一目标帧评分。Nano取输入index8；Shift返回去掉首尾各2帧后的12帧，取输出index6。每帧只评分一次。该模式重算较多，MAC/最终输出帧不能沿用原顺序推理的251.6GFLOPs。禁止混进A表。

C. **论文复现**：按具体论文的原始数据版本/官方权重/边界和尾帧规则/精度/量化规则/聚合方式运行作者脚本。论文值单独标 `PAPER_REFERENCE`，不填入A/B的 measured 列。不能只因为同叫“GoPro test”就直接作严格比较。

核对的官方来源（2026-09-14）：
- https://github.com/JingyunLiang/RVRT ：其README列 GoPro 11视频1111帧、DVD 10视频1000帧，并指向特定最终评估脚本。这里只作为数量参考，不是所有发布版本的强制统一定义。
- https://github.com/dasongli1/Shift-Net/blob/main/inference/test_deblur_small.py ：官方输入/输出RGB，clamp浮点后与GT评分；跳过首尾各2帧，按one_len的整块数处理，剩余尾段可能不计分。默认one_len96，不等于本次完整帧16窗口协议。
- https://github.com/zzh-tech/ESTRNN ：BSD曝光档分别为1ms8ms、2ms16ms、3ms24ms；不能将它们混作一个BSD分数。

Nano训练来源是三域混合，而现成GoPro专用权重不等于三域混合训练。表格必须列 training_data。
125k权重以前已依据GoPro test子集选择，因此本次GoPro不能宣称“从未用于模型选择的严格holdout”。本次冻结权重，禁止进一步依test调参。

## 2. 代理 / SSL / Git安全

执行前 `set +x`，不要打印凭据，不使用 `bash -x`、`env`、`git config --list`。
优先在服务器加载已有私有代理配置；不要把代理密码提交GitHub。

```bash
set +x
if [ -f "${CONDA_PREFIX:-/nonexistent}/etc/conda/activate.d/proxy_env.sh" ]; then
  source "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh" >/dev/null 2>&1
fi
set +x
PROXY_URL="${https_proxy:-${HTTPS_PROXY:-${http_proxy:-}}}"
if [ -n "$PROXY_URL" ]; then
  export http_proxy="$PROXY_URL"
  export https_proxy="$PROXY_URL"
  export HTTP_PROXY="$PROXY_URL"
  export HTTPS_PROXY="$PROXY_URL"
  git config --global http.proxy "$PROXY_URL"
  git config --global https.proxy "$PROXY_URL"
fi
# 用户要求的公司HTTPS检查环境配置；会关闭Git证书校验，仅在该环境使用。
git config --global http.sslVerify false
```

如果代理未配置，读取 `/mnt/ssd1/z00919662/motion_deblur/proxy.md` 中的既有本地配置；MD不是shell脚本，不要整个source。使用其中的proxyhk.huawei.com代理，不猜账号/密码，不写入报告。

用Git克隆/拉取，不再使用旧脚本的raw.githubusercontent.com/curl下载。
在独立checkout操作；旧训练工作区、checkpoint、视频保持原样。禁止reset --hard或git clean破坏现有工作区。

```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur
CODE="$ROOT/benchmark_nanovnr_fair_20260914"
BRANCH=agent/nanovnr-fair-benchmark-20260914
# 不存在才clone；存在时先检查git status并只做fetch/安全checkout。
git clone --single-branch -b "$BRANCH" https://github.com/hihiok/video_motion_deblur.git "$CODE"
cd "$CODE"
git rev-parse HEAD
# 将HEAD与用户消息中的最终提交号对照。
```

激活已有 `deblur_runtime`，不重建/升级环境、不下载训练数据。使用一个可用GPU，CPU线程限制1。

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deblur_runtime
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m compileall -q benchmark_v2
bash -n run_nanovnr_fair_benchmark.sh
python -m unittest discover -s tests -p 'test_benchmark_v2.py' -v
```

## 3. BSD恢复：MP4确认来源，GT从原始Sharp复制

已知：
- BSD整理根：`/mnt/ssd1/z00919662/datasets/BSD`
- 测试仅用直接的 `BSD/test`，训练根不改。
- `gt(color_is_purple)` 曾被移除，不允许用颜色错误GT。
- `gt_mp4`等MP4可作为视频身份/人工颜色参照，**不得抽帧替代benchmark GT**。
- 最新用户明确授权访问原始目录中的Sharp并复制恢复到 `BSD/test/gt`。原始目录只用于追溯和复制，不作为额外测试样本来源。

先在上述BSD根和已有数据处理日志中定位 mp4 folder、实际原始 `.../test/<seq>/Sharp[/RGB]` 与对应 `Blur[/RGB]`。
原始数据也可能在另一已知的BSD解压根；找到后作为额外 `--raw-root` 显式传入。
只做限定目录查找，不全盘扫描；不要扫描不相关业务数据。

准备独立记录目录：

```bash
WORK="$ROOT/runs/nanovnr_fair_prepare_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$WORK"
BSD=/mnt/ssd1/z00919662/datasets/BSD
# 下面两个变量由本地目录/原处理日志确定，不要原样使用占位符。
MP4_ROOT=<实际使用且颜色正确的gt_mp4或mp4目录>
RAW_ROOT=<包含原始BSD test/Sharp与Blur的目录>
python -m benchmark_v2.bsd_gt inventory \
  --bsd-root "$BSD" --raw-root "$RAW_ROOT" --mp4-root "$MP4_ROOT" \
  --out "$WORK/bsd_sources.json"
python -m benchmark_v2.bsd_gt plan \
  --inventory "$WORK/bsd_sources.json" --out "$WORK/bsd_restore_plan.json"
```

自动匹配条件：
1. MP4文件名与整理后视频目录有唯一身份对应。
2. 整理后每张blur与原始Blur逐帧RGB像素完全一致；支持基于唯一连续匹配确定重命名前的帧区间。
3. 原始Blur与Sharp按文件stem明确对应，不按排序强行配对。
4. 原始源必须在test split；所有选中BSD序列是同一个明确曝光档。
5. 记录源文件/MP4 SHA256、原始帧区间、源视频完整长度、局部视频是否为截断子序列。

目录名和MP4名不一致时，允许创建纯数据JSON显式映射，再重跑plan到新文件；**不能修改Python**。
示例（值必须来自本机实际文件，不能照抄数字）：

```json
{
  "Scene000": {
    "mp4": "/实际目录/对应视频.mp4",
    "source_sharp_dir": "/实际原始BSD_3ms24ms/test/真实ID/Sharp/RGB"
  }
}
```

```bash
python -m benchmark_v2.bsd_gt plan --inventory "$WORK/bsd_sources.json" \
  --mapping "$WORK/bsd_mapping.json" --out "$WORK/bsd_restore_plan_v2.json"
```

mapping只能缩小源身份，不能绕过逐帧像素核对。
若源路径本身不带曝光档，映射可添加 `exposure` 和 `exposure_evidence`（实际来源清单/处理日志文件）；必须人工审阅该证据内容，不能只凭文件存在。
若blur本身来自有损MP4/JPEG转码，无法与原始Blur逐像素一致：停止自动复制并报告 `SOURCE_ALIGNMENT_BLOCKED`。这也是不宜直接宣称官方benchmark的重要信号；不要通过搜索最大PSNR选GT或swap通道来“修好”。

计划全部 `READY_TO_COPY` 后，本次授权允许直接执行：

```bash
python -m benchmark_v2.bsd_gt apply --plan "$WORK/bsd_restore_plan.json"
```

apply先核对所有源hash，写独立暂存目录，校验每份拷贝，最终发布到 `BSD/test/gt`。原始文件不动、不转码、不改色。
已有GT不同就停止，不覆盖；已有相同完整GT只校验。`restore_provenance.json`在新GT目录保留。
有歧义则报告具体缺少的映射/源路径，不能声称HUMAN_ACTION=None并跳过BSD。
本次不改BSD/train、不自动重训。若发现训练GT也有颜色问题，独立报告，不现场改训练数据。

## 4. 冻结三个完整测试集

使用以下实际根：
- GoPro `/mnt/ssd1/z00919662/motion_deblur/datasets/GoPro/test`
- DVD `/mnt/ssd1/z00919662/motion_deblur/datasets/DVD/test`
- BSD `/mnt/ssd1/z00919662/datasets/BSD/test`

写 `$WORK/datasets.json`，BSD exposure从恢复计划读出，不猜：

```json
{
  "datasets": {
    "GoPro": {
      "test_root": "/mnt/ssd1/z00919662/motion_deblur/datasets/GoPro/test",
      "train_root": "/mnt/ssd1/z00919662/motion_deblur/datasets/GoPro/train",
      "expected_counts": {"videos": 11, "frames": 1111}
    },
    "DVD": {
      "test_root": "/mnt/ssd1/z00919662/motion_deblur/datasets/DVD/test",
      "train_root": "/mnt/ssd1/z00919662/motion_deblur/datasets/DVD/train",
      "expected_counts": {"videos": 10, "frames": 1000}
    },
    "BSD": {
      "test_root": "/mnt/ssd1/z00919662/datasets/BSD/test",
      "train_root": "/mnt/ssd1/z00919662/datasets/BSD/train",
      "exposure": "必须替换为来源已证实的1ms8ms或2ms16ms或3ms24ms"
    }
  }
}
```

这里读取train只用于重复帧/序列交叉检查，不推理、不训练，不将train混入test。
若训练目录组织形式不兼容，只报告缺失的交叉检查，允许移除配置中的train_root继续输出明确标记`NOT_AUDITED`的实测结果，禁止宣称已证实无泄漏。
重复原图会触发review，不能随意删测试样本消除冲突。
GoPro不同chunk共享acquisition前缀不是自动泄漏；代码不会截短视频ID来作错误的前缀拦截。

数量不符或恢复计划显示原视频被截短：仍可输出完整“本地test”的实测，但标`CUSTOM_SUBSET/LOCAL_RELEASE`，与论文分开。
不要擅自增加原始测试视频、更换曝光档、删难样本或假造缺帧。数量符合也不等于源版本已经认证；默认`paper_comparable=false`。
若要正式论文复现，还需对应论文官方列表/原始帧与当前manifest逐项核对。

## 5. 默认执行：全三域、固定125k、FP32

固定权重：
`/mnt/ssd1/z00919662/motion_deblur/runs/nanovnr_nafnet_rgb_fullframe_bsd_train_test_20260904/train/step_0125000.pth`

报告完整64位SHA256，不能只报`b0a1c536...c90af`。
网络源码Git blob必须为 `de3b8032940bd96413fc685ce15389de3e899a90`。
新分支已包含它；无须curl下载，不接受本地未经审查的改动。

```bash
export CONFIG="$WORK/datasets.json"
export BENCH="$ROOT/runs/nanovnr_fair_all_$(date +%Y%m%d_%H%M%S)_$$"
export GPU=<实际空闲GPU编号>
export CONTEXT=0
export PRECISION=fp32
bash run_nanovnr_fair_benchmark.sh
```

默认不做TTA、x8 self ensemble或任何GT相关后处理。FP32前向、TF32关闭、指标FP64累积。
Nano按T15不重叠顺序推理，同一视频forward hidden续传，每个新视频重置，尾段全部处理。
这是已固定的部署协议，不是全序列双向或同T的论文协议。
若OOM不偷偷降T、切图或改FP16，报告失败场景及显存。新协议必须整轮另开目录，不能前半FP32后半FP16。
代码按完整视频保存断点；相同CONFIG/BENCH重跑时会校验hash并跳过完成视频。中断视频从其开头重算，避免丢失时域状态。

输出：
- `manifest.json`：冻结视频、每帧LQ/GT路径和SHA。
- `nano/run.json`：模型源码/权重SHA、环境、推理协议、训练数据、选模历史。
- `nano/sequences/<dataset>/<sequence>/pred/*.png`：完整原生无损预测。
- 同目录 `per_frame.csv`、`sequence.json`、首中末Input|Output|GT PNG。
- `nano/summary.json`、`nano/summary.csv`：完整结果。
- 未完成时只有 `summary.partial.json`，不能当完整榜。

主指标是RGB8 PSNR：预测clamp[0,1]、统一round到8bit、与原始GT计算全RGB MSE，再算每帧PSNR，最后全测试帧算术平均。
同时输出未量化浮点PSNR；只能float对float、RGB8对RGB8，不能混比。
同时报告按视频等权的macro mean；主表采用按帧等权，不能平均PSNR和整体MSE转换PSNR混用。
不裁边、不作亮度缩放或几何配准。GT=GT的PSNR应为inf，不人为截为120dB。

## 6. 其他方法：先复用已验证的Shift-Net Ours-s

代码内置官方 `GShiftNet(future_frames=2,past_frames=2)` 的加载适配，不实现/修改其网络。
在服务器已有仓库/日志中找到 `gshift_deblur2.py` 和确实可用的Ours-s权重。
不能将文件名net_gopro_deblur.pth直接当作small；必须依据结构、严格state加载、实际参数量和之前验证日志确认。
记录官方checkout完整commit和本地改动。若没有可核验的权重/环境，则完成Nano全三域并报告比较模型缺少何物；不能拿论文分数冒充本次测量。
不要为此任务重训其他模型或下载大型数据集。

先写 `$WORK/shift_metadata.json`，例如：

```json
{
  "training_data": ["GoPro train，须依据权重来源核验"],
  "checkpoint_origin": "官方Shift-Net Ours-s，写真实权重来源与已核验SHA256",
  "checkpoint_selection": "Published fixed checkpoint; no retuning on current tests",
  "source_code_commit": "实际官方checkout完整commit"
}
```

默认固定同一个经过确认的GoPro Ours-s checkpoint跨三域测试，BSD/DVD栏标cross-dataset，不能说它也在三域混合训练过。
若另外跑DVD专用权重，作为独立checkpoint行，不在一个method行中暗中换权重。

```bash
export SHIFT_REPO=<已验证官方Shift-Net checkout绝对路径>
export SHIFT_CHECKPOINT=<已验证Ours-s权重绝对路径>
export SHIFT_METADATA="$WORK/shift_metadata.json"
# CONFIG、BENCH保持与上一步相同，Nano会校验并resume跳过已完成的视频。
bash run_nanovnr_fair_benchmark.sh
```

本适配对每个视频首尾reflection padding、处理全部尾段、原生输出。官方脚本不一定这样，因此不是其论文数字复现。
相同FP32/RGB8/manifest下生成 `comparison.json`，里面保留不同的时域与训练信息。

需要严格同上下文时，另设 `CONTEXT=16` 和新的BENCH，对Nano与Shift一起跑；不复用部署协议结果。
不需要为了“公平”强迫所有模型去改内部结构或训练；严格同训练条件需要独立重训项目，本任务不做。

## 7. 其他已验证方法的统一评分

已有RT-Focuser/BSSTNet等官方推理脚本可继续使用，但必须读取同一manifest对应原始图像，输出一张对应每个目标帧的原生RGB PNG。
仅将输出排布成：`<prediction-root>/<dataset>/<sequence>/<input_frame_stem>.png`。
不允许从MP4回抽PNG评分。官方脚本改名输出时，必须使用它的明确frame mapping，不按目录序号猜。
不要自行实现新teacher/网络/adapter。现成脚本的GT不得用于推理调参或对齐。

元数据JSON必须包含 `method, training_data, checkpoint_sha256, protocol, precision, source_code_commit`。

```bash
python -m benchmark_v2.evaluate index --manifest "$BENCH/manifest.json" \
  --prediction-root <原生PNG结果根目录> --metadata <该方法元数据JSON> \
  --out "$WORK/other_index.json"
python -m benchmark_v2.evaluate score --manifest "$BENCH/manifest.json" \
  --index "$WORK/other_index.json" --out "$WORK/other_score.json"
python -m benchmark_v2.evaluate compare \
  --reports "$BENCH/nano/summary.json" "$WORK/other_score.json" \
  --out "$WORK/other_comparison.json"
```

缺帧/重复帧/不同manifest/不同精度/不同主指标拒绝出共同榜。不自动取交集掩盖某模型漏测。
若某方法的论文只评中间帧，应另建其明确目标帧列表的论文复现，不改本次全帧榜。

## 8. 人工事项和报告

常规流程：CodeAgent自行定位已有源数据、核对映射、通过后复制GT并全量评估，无需用户反复提供已知路径。
仅出现源身份/曝光档不清、原始Sharp缺失、有损blur无法验证、已有GT冲突或权重缺失等真正阻塞时，写出具体人工事项。
人工视觉检查不能写成已完成：用户最终查看恢复的GT和模型Input|Output|GT预览，尤其颜色、人脸、运动边缘与chunk边界。
不要因平均PSNR提高就声称所有帧变好或时域稳定。

最终报告必须包括：

```text
STATUS: COMPLETE / PARTIAL / BLOCKED
HUMAN_ACTION_REQUIRED:
EXACT_USER_ACTION:
GITHUB_BRANCH:
GITHUB_COMMIT:
WORKTREE_DIFF_OR_NONE:
MODEL_BLOB:
CHECKPOINT_SHA256:
CHECKPOINT_STEP: 125000
BSD_SOURCE_MP4_ROOT:
BSD_ORIGINAL_SHARP_ROOTS:
BSD_RESTORE_PLAN_SHA256:
BSD_EXPOSURE:
BSD_SOURCE_ALIGNMENT:
BSD_COPIED_FILES:
BSD_EXISTING_DATA_OVERWRITTEN: NO
BSD_TRAIN_MODIFIED: NO
DATASET_MANIFEST_SHA256:
TEST_COUNTS_BY_DATASET_AND_VIDEO:
OFFICIAL_RELEASE_VERIFIED: YES / NO + evidence
LOCAL_SUBSET_OR_RELEASE_WARNINGS:
TRAIN_TEST_OVERLAP_AUDIT:
KNOWN_GOPRO_TEST_BASED_CHECKPOINT_SELECTION: YES
PRECISION:
TEMPORAL_PROTOCOL_PER_METHOD:
PRIMARY_RGB8_FRAME_MEAN:
Method | Training data | GoPro | DVD | BSD(exposure) | Native-compute/latency notes
SECONDARY_FLOAT_FRAME_MEAN:
VIDEO_MACRO_MEAN:
PER_VIDEO_AND_PER_FRAME_FILES:
BASELINE_UNAVAILABLE_REASONS:
PAPER_REFERENCE_SEPARATE_FROM_MEASURED: YES
OUTPUT_ROOTS:
VISUAL_REVIEW_REQUIRED: YES
```

不得填写未运行的结果，不把“本地全量”冒充“官方全量”，不在测试中重新挑125k以外权重。
