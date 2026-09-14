# CodeAgent：三个数据集统一 PSNR benchmark（不做消融、不重新训练）

## 目标与边界

用户要比较已训练完成的 RT-Focuser 时域模型与其他方法在 GoPro、BSD、DVD 上的 PSNR，同时保留业务 MP4 对比任务。直接完成可执行任务，缺少某模型的资源时记录具体缺失并继续其他方法。

代码已经写好。CodeAgent 只负责定位资源、填写本地运行 YAML、运行、检查和汇报；不要自行改源码、模型结构、时域缓存长度、指标公式或加载规则。填本地路径/权重来源/官方输出映射属于配置工作，可以自主完成。

**不做消融，不关闭训练后时域模块，不新增训练。** 对比只用当前 `best_stable.pth`，不在 test 上挑 latest 或其他 checkpoint。下面的两类结果必须分开：

1. **统一实测表**：所有方法使用固定的同一套完整测试帧、原尺寸 RGB8 PNG、同一个 PSNR 代码。它回答“这些已训练好的系统，在相同输入上的效果如何”。必须保留训练条件与因果性标签。
2. **官方协议复现记录**：使用各方法官方脚本/配置，记录其实际帧清单、精度、量化、边界、聚合规则及论文参考值。它用于发现适配器或口径差异，不能把其日志分数塞入统一表。

当前模型是 GoPro 官方预训练后再做 GoPro+BSD+DVD 三域联合微调。直接与仅在单域训练的模型比较，**不能据此宣称同训练条件下的架构优势**。本轮不补做单域训练；在表里单列 `joint_finetuned`。BSD 上使用 GoPro 权重的方法列 `cross_domain_pretrained`，不是 BSD 专训成绩。未知训练来源列 `unknown_training`。

## 仓库与路径

| 项目 | 值 |
|---|---|
| 仓库 | https://github.com/hihiok/video_motion_deblur |
| 本次分支 | `agent/rtfocuser-fair-benchmark-three-datasets-v1` |
| 本次代码 commit | 使用 Codex 最终交付消息中的完整 SHA，拉取后核对 |
| 新 checkout | `/data/pub/z00919662/motion_deblur/benchmark_code_fair_v1` |
| 主入口 | `tools/benchmark_three_datasets.py` |
| 方法入口 | `tools/infer_fair_video.py` |
| 原训练 run | `/data/pub/z00919662/motion_deblur/runs/rtfocuser_causal_temporal_finetune_v1` |
| 业务输入目录 | `/data/pub/z00919662/motion_deblur/input` |
| 业务完整指令 | `docs/CODEAGENT_RTF_POSTTRAIN_EVAL_BUSINESS_V1.md` |

本分支包含上一项业务评测代码。业务任务若尚未运行，可以在本 checkout 执行业务 MD 中的 Python 入口，使用本次 checkout/commit 作为评测代码身份；无需再 clone 旧评测分支。若业务任务已经完成，不重复运行；统一 benchmark 的 RGB8 PSNR 与该 MD 的 FP32 float PSNR是不同指标口径，不能混用。

## 1. 代理、SSL、环境与代码

沿用已授权代理。交接中代理可能在 `$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh`，优先正常激活环境让它生效，或使用已有私有 `DEBLUR_PROXY_ENV` 文件。绝不把账号密码写入 GitHub、报告或日志，不开 `set -x`。

```bash
set +x
set -euo pipefail
if [ -n "${DEBLUR_PROXY_ENV:-}" ]; then source "$DEBLUR_PROXY_ENV"; fi
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
TASK_CODE="$TASK_BASE/benchmark_code_fair_v1"
if [ ! -e "$TASK_CODE" ]; then
  git clone --single-branch --branch agent/rtfocuser-fair-benchmark-three-datasets-v1 \
    https://github.com/hihiok/video_motion_deblur.git "$TASK_CODE"
fi
cd "$TASK_CODE"
git branch --show-current
git rev-parse HEAD
git status --porcelain
conda activate deblur_runtime
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
python tools/benchmark_three_datasets.py --help
python -m pytest -q tests/test_fair_benchmark.py tests/test_rtf_posttrain.py
```

checkout 已存在时核对来源、分支和干净状态，仅允许干净同分支做 fetch/ff-only，不能覆盖用户改动。核对本次 commit；不要更新仍被其他运行任务使用的源码。

保持现有 Python 3.9 / torch 2.2.2 / CUDA 11.8 / NumPy 1.26 / `opencv-python-headless==4.8.1.78`。不要整环境升级。缺少 `scikit-image` 时只补 Python3.9 兼容的 0.22.0；测试用 pytest 8.x。不同方法可以在 YAML 的 `python` 字段指定各自已经可用的环境解释器，模型之间通过独立子进程隔离 BasicSR/CUDA 扩展导入；指标由统一 benchmark 环境计算。

其他模型若缺依赖，优先复用服务器已有环境；按官方说明在独立环境修复。禁用 stub、零张量占位、自行删层、非 strict load 等“能跑就行”的替代。不要安装损坏 deblur_runtime 的旧 torch/mmcv 全家桶。

## 2. 先生成本地配置，查清资源

```bash
TASK_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TASK_CFG="$TASK_BASE/runs/fair_benchmark_config_$TASK_STAMP.yaml"
TASK_OUT="$TASK_BASE/runs/fair_benchmark_$TASK_STAMP"
python tools/benchmark_three_datasets.py init --base "$TASK_BASE" --config "$TASK_CFG"
```

生成器从训练 manifest 和 `weights/`、`benchmark/weights/`、`envs/` 查找资源。**生成的路径是待核验配置，不代表权重身份已经确认。** 遇到同名文件多个候选时不得按最高 test PSNR 选择；查来源、配置、SHA，固定一个有证据的 checkpoint。

在 `$TASK_CFG` 中只修改实际路径、数据布局、明确的训练来源、匹配 checkpoint 的配置文件和现有解释器路径。所有修改必须在 `prepare` 前完成。未知权重来源不能靠改文字“变成已核实”。checkpoint 同时含 params/params_ema 等多个容器时，必须依据官方加载代码在该 job 显式填写 `state_key`（例如 `params`），不能自动选其中得分更高的版本；默认 `auto_unique` 会拒绝歧义。

默认实现的模型种类：

| kind | GoPro | DVD | BSD 3ms24ms |
|---|---|---|---|
| `rtf_official` | 官方 GoPro 权重 | 同一 GoPro 权重，跨域 | 同一 GoPro 权重，跨域 |
| `rtf_temporal` | 当前三域联合 best_stable | 同一 best_stable | 同一 best_stable |
| `shiftnet` | 官方 Shift-Net+ GoPro | 官方 Shift-Net+ DVD | 默认 GoPro 权重，跨域 |
| `dstnet` | 官方 GOPRO.pth | 官方 DVD.pth | BSD.pth 必须核实曝光档与训练来源 |
| `bsstnet` | BSST_gopro.pth | BSST_dvd.pth | 默认 GoPro 权重，跨域 |
| `rvrt` | 官方 005 GoPro | 官方 004 DVD | 默认 005，跨域 |
| `turtle` | 匹配 GoPro 配置的权重 | 默认无已验证 DVD 专用权重，记缺失 | 必须匹配 BSD 权重、曝光档及其配置 |

缺少真实模型仓库时，可从配置中 `repo_url` 获取官方源码到独立目录，记录 commit。权重优先使用已有本地文件，允许从官方 README 指定的公开下载来源补齐；链接需要账号或不可达时记录具体资源/错误，继续可跑模型。不能用其他任务权重冒充，不能在某模型失败后写 0 dB 入榜。

特别注意：

- DSTNet 当前公开 BSD 测试 YAML 指向 **BSD_1ms8ms**。仅凭文件叫 `BSD.pth` 无法确认它适用于 3ms24ms 专训对比。未核实就保留 `training_domains: [unknown]`，汇报来源不明。
- Turtle 的 t0/t1 结构及 YAML 必须与权重匹配，`num_frames_tocache` 按该配置保留。不能把论文中的 γ=5、历史项目参数或 GoPro 配置直接套到 BSD 权重上。严格加载能验证参数形状，但不能替代非参数缓存设置的来源核查。
- RVRT 使用真实官方去模糊网络和官方 `test_video` 函数，记录 CUDA op 是否正常；不可使用历史 stub。
- `rtf_temporal` 的权重必须与原训练 run 中 audited `best_stable.pth` 一致。程序记录真正 best_update，不假定为 20000。

## 3. 三个数据集的固定测试口径

| 数据集 | 测试数据 | 主表帧数 | 空间处理 |
|---|---|---:|---|
| GoPro | 官方 11 条 test 序列；视频 benchmark 默认明确使用 `blur` | 1111 | 原图尺寸，通常 720×1280 |
| BSD | **3ms24ms / RGB / test**，20 条×150 帧 | 3000 | 480×640 |
| DVD | 官方元数据中列出的 10 条 test 序列 | 1000 | 按官方逐序列尺寸，保留横/竖屏 |

GoPro `blur` 与 RT-Focuser 上游 image loader 的 `blur_gamma` 是需要标明的协议差异。统一三数据集表默认用视频方法配置里的 `blur`。若现有数据仅有 `blur_gamma`，可以在第一次 freeze 前把 GoPro `lq_dir` 和 `variant` 都设为 `blur_gamma`，对所有方法使用同一输入，并给表明确标注该变体；不能只给某个方法切换，也不能看分数后挑更高的一版。业务评测任务已有双变体诊断，不在这里扩展模型消融。

支持两种明确布局：

```text
sequence_modalities: test/<sequence>/<blur-or-Blur>[/RGB]/*.png
                     test/<sequence>/<sharp-or-Sharp>[/RGB]/*.png
split_modalities:    test/<blur>/<sequence>/*.png
                     test/<gt>/<sequence>/*.png
```

`lq_dir`、`gt_dir` 字段明确指定实际目录名，大小写查找不敏感。BSD 多一层 RGB 已直接支持，不需要重新建立扁平视图。DVD 若 GT 目录叫 `GT`，配置 `gt_dir: GT`。禁止 train/val 自动回退、截取凑数、为通过元数据检查而改序列名或把 1080p 缩到 720p。不同版本的原始 DVD 数据不能无说明当标准 DVD10。

**DVD 先审计**：已报告的训练+留出集合是 66 序列 / 6208 帧，而常用 DVD 数据划分训练为 61 序列 / 5708 帧，test 为 10 / 1000。当前差异不是泄漏的既定结论，但必须核对。程序会用全部三域 train **和用于选 best 的 val** 的 GT 像素哈希，与测试集核对；同内容重新保存 PNG 也能识别。亦检查相同 domain/sequence/original-frame 身份。

若发现重叠，该测试域上的 `rtf_temporal` 标记 `TEST_NOT_INDEPENDENT:OVERLAP`，不推理出正式成绩，不通过删测试帧来“修复”。其他无阻塞方法与数据域照常完成。原 checkpoint/manifest 不改，本轮不擅自重训；报告重叠序列和训练路径。如果源 GT 已迁移不可读，记 `UNVERIFIABLE`，不能假装独立。哈希检查无法排除所有改尺寸/近重复采集内容，还需检查 DVD 序列来源和切分规则。

GoPro 官方 train/test 可能有共同采集名前缀，**不能仅因前缀相同就判泄漏**；新代码核对的是精确帧身份和像素内容。

## 4. 冻结并运行

```bash
python tools/benchmark_three_datasets.py prepare --config "$TASK_CFG" --output "$TASK_OUT"
```

准备后检查：

- `frozen.json` / `frozen.sha256`：固定测试帧、SHA、路径、方法参数、权重快照、源码 commit、来源标签。
- `overlap_audit.json`：训练及选模数据与每个测试域的重叠。
- `training_audit/`：官方权重 SHA、原 manifest SHA、best_update。
- 每个 job 的 `READY` / `UNAVAILABLE` 和具体原因。

检查实际内容，不以 prepare 返回码或“目录存在”推断三域齐全。缺某域 test 时仍准备其他域，报告 `dataset_errors`。新目录建立后不改 frozen 文件；需要纠正路径/权重/协议，建立新配置和新输出目录。不能在已有成绩基础上调整参数再当同一次预注册评测。

默认 1 张空闲 A100，不抢业务推理正在使用的 GPU。可先完成业务再跑 benchmark；两项独立任务也可各用一张经实时确认空闲的卡，但不要同时改同一个运行目录。

```bash
TASK_GPU="$(python tools/select_rtf_temporal_gpus.py --max-gpus 1)"
export CUDA_VISIBLE_DEVICES="$TASK_GPU"
# 在 tmux/持久任务会话中运行，保留日志、退出码并监控。
set +e
python -u tools/benchmark_three_datasets.py run --output "$TASK_OUT" --phase smoke --device cuda:0 \
  2>&1 | tee "$TASK_OUT/smoke.log"
TASK_SMOKE_RC=${PIPESTATUS[0]}
set -e
```

smoke 对每个 READY job 的前两条**完整序列**进行推理；不是裁图或只推两帧。检查 `jobs/<job>/<sequence>/inference.log`、输入/输出/GT、帧号与尺寸，以及有无 NaN、颜色错误、输出几乎恒定、过度平滑。代码会在量化前拒绝 NaN/Inf，不使用旧 adapter 的 `nan_to_num` 掩盖异常。

对比官方脚本时按下一节添加 anchor。有具体错误的 job 不重复空跑；先记录 traceback。其他通过的 job 继续全量，可用 `--only job_id ...` 指定。无需停下等用户再次确认启动。

```bash
set +e
python -u tools/benchmark_three_datasets.py run --output "$TASK_OUT" --phase full --device cuda:0 \
  2>&1 | tee "$TASK_OUT/full.log"
TASK_FULL_RC=${PIPESTATUS[0]}
python tools/benchmark_three_datasets.py score --output "$TASK_OUT" \
  2>&1 | tee "$TASK_OUT/score.log"
TASK_SCORE_RC=${PIPESTATUS[0]}
set -e
```

完成过的序列会核对 job 指纹与输出 SHA 后跳过，smoke 结果会直接复用。半途失败的目录移到 `failed_attempts/` 留存，再重试该序列；不会覆盖原 checkpoint 或原数据。OOM 时不要自动缩图、减时域窗口、换 FP16 或改变 tile；记录实际失败。确需另一个资源配置，必须新 job / 新 freeze，并标注该配置，不在同一行混合不同推理条件。

实际使用多张 GPU 时，为不同 job 指定不同 `--only` 集合；不可让两个进程写同一个 job，且最终所有 job 完成后单独运行 score。

## 5. 官方输出 anchor：避免再次出现 harness 分数异常

统一推理明确使用 FP32 / TF32 off / 原尺寸。各方法保留不同的输入信息与结构：

| 方法 | 本实现固定策略 |
|---|---|
| RT-Focuser | 单帧；仅 pad16 后去 padding |
| 时域 RT-Focuser | 连续 state；序列开头及原训练 cut 规则重置；不每4帧重置 |
| Shift-Net+ | 48 个核心输出帧；核心 chunk 相位与官方从第2帧开始一致；额外补全首尾及不足窗口尾段 |
| DSTNet | 固定30帧段，默认整帧；不使用旧业务脚本的4帧+动态缩tile兜底 |
| BSSTNet | 固定48帧段；256空间块/64重叠/均匀聚合；RAFT与模型 FP32 |
| RVRT | 官方 test_video；默认 tile=[30,256,256]，overlap=[2,20,20]，记录这一资源配置 |
| Turtle | 匹配权重的 t0/t1 与配置；默认320块/128重叠；整序列连续的逐块 K/V 缓存 |

这些是明确登记的统一实测设置，不声称与每篇论文的原始执行方式完全一致。图像完整尺寸不等于所有模型都一次吃整帧：BSSTNet 等方法需要空间块，但最终必须覆盖每个原始像素。

对准备引用其论文成绩或信任适配器一致性的外部模型，优先用同一 checkpoint、同一上游 commit 的官方推理代码在相同前两条测试序列跑出输出。官方脚本所需 YAML 路径修改写到独立 runtime 配置，保持原仓库源码不变。以下为官方入口定位，实际参数以已检出的源码/帮助为准：

- Shift-Net+：`inference/test_deblur.py --default_data GOPRO|DVD --one_len 48 --save_image`。该脚本的输出重新从0编号，**输出000对应的GT并不是原始帧0**，需按原代码 `begin_frames=2` 和 chunk 相位建立映射；它还跳过不足完整窗口的尾段。
- DSTNet：`basicsr/test.py -opt options/test/Deblur/test_Deblur_GOPRO.yml`，其他域使用相应配置；官方结果需按 `merge_full.py` 的真实帧含义合并，禁止按文件排序硬 zip GT。
- BSSTNet：`basicsr/test.py` / `scripts/dist_test.sh`，配置 `options/test/BSST/gopro_BSST.yml` 或 `dvd_BSST.yml`；独立 runtime YAML 中开启保存输出并指向相同真实测试数据。
- RVRT：`main_test_rvrt.py --task 004_RVRT_videodeblurring_DVD_16frames` 或 005 GoPro，显式 `--folder_lq --folder_gt --tile --tile_overlap --save_result`；记录其逐序列平均与统一逐帧平均的区别。
- Turtle：仓库 `basicsr/inference.py` 中对应去模糊配置/权重/模型类型；按 checkpoint 对应的 t0/t1 和历史缓存处理。脚本若只提供硬编码演示入口，不让 CodeAgent自行改网络或猜参数：先提供已有官方推理产物及其来源，无法取得就明确“官方 anchor 未验证”。

官方代码缺依赖或没有适用权重时，不造 reference。统一表的可用实测仍保留，但 `official_reference=NOT_CHECKED`，不得声称已复现论文。可视化检查不能替代数值 anchor，数值接近论文也不能替代帧对应关系。

官方输出映射是一个本地 JSON 数据文件，至少覆盖2条序列、32个真实对应帧。它可以在 smoke 后追加，不改变 frozen 权重或输入。示意结构（字段必须填实际证据）：

```json
{
  "checkpoint_sha256": "与该job完全一致的真实SHA",
  "repo_commit": "真实上游commit",
  "command": ["实际执行的解释器", "实际官方脚本", "实际参数"],
  "protocol_note": "官方的精度、窗口、padding、保存量化和跳帧规则",
  "rows": [
    {"sequence": "真实序列名", "original_frame_id": "原始GT帧stem", "official_prediction": "/绝对路径/官方对应输出.png"}
  ]
}
```

```bash
python tools/benchmark_three_datasets.py anchor \
  --output "$TASK_OUT" --job shiftnet_gopro --mapping /绝对路径/shiftnet_gopro_anchor.json
```

原生形状与对应帧必须相同；不允许 resize、移位、色彩拟合、挑高分帧使其通过。代码参考阈值：最大像素差≤2/255，逐帧PSNR绝对差的均值≤0.05dB。不同精度/边界/空间融合导致不通过时，先报告具体协议差异；不能一概称模型损坏。anchor 不通过的结果留作诊断，不混进统一报告的有效结果区。没有 anchor 的实测必须显示其验证状态。

## 6. PSNR 计算与最终交付

统一主指标为原尺寸 RGB8 PNG：所有预测在保存前检查 finite，然后 clip[0,1]、乘255、round、转uint8一次。输入/GT来自同一 canonical PNG，JPEG 数据源只解码一次转换为无损 PNG，不用重新编码 JPEG/MP4 算分。

`PSNR(frame) = 10*log10(255^2 / mean((prediction-GT)^2))`，RGB三通道所有像素均参与，MSE用float64。主表是**逐帧dB算术平均**；额外给出“逐序列平均后再平均”，不可用全数据集平均MSE换算值冒充这两种。GT-vs-GT 为 inf，JSON 用字符串 `inf` 保存，不伪造120dB。SSIM统一 skimage / RGB / data_range255 / win7，作为辅助列。

没有 spatial crop、Y通道、自动对齐、边界扣除、随机 crop256、自增强、锐化或颜色补偿。所有方法的完整首尾帧均入主表；缺一帧即该单元不出正式全量分数。

交付目录：

- `benchmark_report.md`、`benchmark_table.csv`、`benchmark_report.json`：三个数据集分表，训练条件、因果性和状态可见。
- `scores/<job>/per_frame.csv`、`per_sequence.json`、`metrics.json`：可复核每帧/每序列分数。
- `predictions/<job>/<sequence>/*.png`：统一命名输出，原始帧 ID 保存在 frozen manifest。
- `jobs/`：严格加载、推理策略、运行日志、已完成帧 SHA；`anchors/` 为官方输出与来源快照。
- `overlap_audit.json` 与 `training_audit/`：尤其报告 DVD 是否出现训练/选模与测试重叠。
- `run_smoke_status.json`、`run_full_status.json` 与总日志。

只报告真实实测结果，不从论文表抄数字填缺失项。分别报告三域，不用帧数加权的总PSNR掩盖某一域退化；本代码不输出三域混合排名。未知来源、不同训练条件、非因果方法都保留标签；不得将训练域/未来帧信息差异解释成架构的净收益。

最终反馈至少包含：

```text
STATUS: BENCHMARK_COMPLETE / BENCHMARK_PARTIAL / BLOCKED
REPO / BRANCH / COMMIT / SOURCE_CODE_MODIFIED:
DATASET_COUNTS_VARIANTS_RESOLUTIONS:
  GoPro / BSD_3ms24ms / DVD：实际序列、帧数、输入类型、是否符合官方名单。
DVD_SPLIT_AUDIT:
  66训练+留出序列与官方61训练的差异来源；精确重叠/未发现/不可验证及证据。
WEIGHT_MATRIX:
  method × dataset：checkpoint SHA、训练来源、same/cross/joint/unknown、实际参数、因果性。
UNIFIED_RGB8_PSNR_TABLE:
  三个数据集分别报告 input、official RT、temporal RT、各对照方法。
OFFICIAL_REFERENCE_STATUS:
  每个外部方法的 anchor PASS / MISMATCH / NOT_CHECKED；官方协议结果单列。
MISSING_FAILED_EXCLUDED:
  具体缺失资源/错误，不使用0分或小样本分数冒充全量。
BUSINESS_MP4_OUTPUTS:
  是否完成输入目录全部视频、两个原尺寸结果和三栏对比视频的路径。
OUTPUT_ROOT / REPORT_PATHS / LOG_PATHS:
NO_ABLATION / NO_RETRAIN / TRAINING_RUN_UNMODIFIED: YES / YES / YES
HUMAN_ACTION_REQUIRED / EXACT_REQUIRED_INPUT:
```

测试在 Codex 侧只覆盖合成数据和 CPU 下的真实 RT-Focuser 推理，不代表其他方法的真实权重与 CUDA 扩展已在新服务器验证。以本次服务器日志为准。

协议来源及上游文件 SHA 见 `fair_benchmark/meta/SOURCES.md`。其中 GoPro/DVD 测试名单来自 BSSTNet 作者仓库，BSD曝光与测试帧长来自原始 ESTRNN/BSD 作者仓库；不是根据本次模型得分挑选的名单。
