# CodeAgent：按官方Shift-Net-s输入方式恢复训练（crop256、T13）

## 本次用户授权与执行目标

用户在2026-09-14明确允许训练输入与原版Shift-Net-s一致。本文件替代旧版的原生整帧训练要求、T16训练要求、1080p整帧反向预检及Python>=3.10限制。只以本文件为本次恢复执行入口。

- GitHub：<https://github.com/hihiok/video_motion_deblur>
- 分支：`agent/shiftnet-500g-mix-v1`
- checkout：`/data/pub/z00919662/motion_deblur/shift500_code_v1`
- 已审计旧run：`/data/pub/z00919662/motion_deblur/runs/shift500_mix_v1`
- **新run**：`/data/pub/z00919662/motion_deblur/runs/shift500_crop256_t13_v2`
- 环境：沿用`deblur_runtime`，Python **3.9.23**、torch **2.2.2**、CUDA **11.8**；这些版本允许使用，不升级Python/PyTorch/CUDA。
- Teacher：已下载的官方GoPro Ours-s，SHA256 `39f470a77b0b3d23ce5e1e8972e1213ba6cc73a097241af1021281443d2d4f00`。从旧config取实际路径，不重新下载或猜位置。
- quality和compact各180k optimizer updates，三域GoPro/DVD/BSD=2/1/1 clips per update。继续PSNR约35dB和1080p每输出帧<=500GFLOPs目标，未训练结果不得写为已达标。
- CodeAgent只执行现成代码。禁止自行修改源码/参数来绕过检查；有修改先导出diff及新增文件回传ChatGPT。本次不需要用户手工改代码、移动数据或重装环境。

## 原版输入依据与本轮口径

固定上游commit `450a4f246dedccd306aa0bc02d615d797874e1ce`：

1. `options/gopro_deblur_small.yml`：n_sequence=13，patch_size=256，gt_size=256。
2. `basicsr/data/video_image_dataset.py`：读取同一clip内连续13帧，将时序帧串在通道维后统一get_patch，GT使用同一crop；再做一致的水平/垂直翻转和90度旋转。不是给每帧独立随机裁剪，也不是缩放整帧到256。
3. `basicsr/models/archs/gshift_deblur2.py`：训练make_model使用默认past_frames=future_frames=1；`image_restoration1_model.py`的GT取`[1:-1]`，因此训练13输入、11输出。
4. `inference/test_deblur_small.py`：完整画面推理；前后各2帧上下文，one_len是输出帧数，README示例one_len=96，因此100输入96输出。窗口长度可按显存选择。

本次训练遵循上述**输入、裁剪、空间增强和GT索引方式**。我们仍使用压缩student、三域混合、teacher蒸馏和原先损失/180k计划，不声称完整复现原版300k、8卡、GoPro单域训练配方。

默认部署仍使用原分辨率16输入12输出，teacher/student评测同一协议；不因训练crop把1080p算力除以分辨率比例后冒充部署算力。quality **0.643947M、475.933GFLOPs**，compact **0.448255M、389.390GFLOPs**，1MAC=2FLOPs。尾部不足12输出的额外计算仍单独统计。两个模型参数和实际部署计算图不因训练输入修改而变化。

## 1. Proxy、SSL、pull与环境

沿用私人附件中的公司代理配置，或本机已设置的代理环境。不要输出凭据，也不要set -x。每个新shell先加载原来的proxy配置；下面的检查不需要重新提供密码。

```bash
: "${http_proxy:?先加载之前私人附件的proxy环境}"
export https_proxy="$http_proxy" HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$http_proxy"
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=true
cd /data/pub/z00919662/motion_deblur/shift500_code_v1
test "$(git branch --show-current)" = agent/shiftnet-500g-mix-v1
test -z "$(git status --porcelain)"
git pull --ff-only origin agent/shiftnet-500g-mix-v1
git rev-parse HEAD
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deblur_runtime
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export SHIFT500_UPSTREAM=/data/pub/z00919662/motion_deblur/envs/Shift-Net-pinned-450a4f2
```

若上游实际路径不同，从旧config.json读取其upstream值设置SHIFT500_UPSTREAM；不要修改上游文件或source哈希校验。不需要新依赖，复用先前已通过测试的环境。

```bash
python -m pytest -q tests_shift500
```

新回归覆盖：13帧同位置crop/GT配对、空间增强同步且时序不打乱、13->11与原版输出逐元素一致、两个student梯度有限、eval仍16->12、旧整帧config不能误通过新训练gate、配置迁移保留数据划分/旧文件、Python3.9语法。服务器DDP测试必须实际通过；不能把跳过写成通过。

## 2. 从已审计配置迁移到新的run

```bash
DEBLUR_ROOT=/data/pub/z00919662/motion_deblur
SHIFT_OLD_RUN="$DEBLUR_ROOT/runs/shift500_mix_v1"
SHIFT_RUN="$DEBLUR_ROOT/runs/shift500_crop256_t13_v2"
python -m shift500.prepare_crop \
  --from-config "$SHIFT_OLD_RUN/config.json" \
  --run "$SHIFT_RUN"
export SHIFT500_CONFIG="$SHIFT_RUN/config.json"
```

迁移程序验证旧manifest和teacher文件哈希，完整保留train/val/test分配及GoPro重叠审计，只更新训练输入协议和新输出目录，重新核算部署预算。旧config、manifest、OOM报告和日志不被覆盖；不载入旧run checkpoint。

确认新config：frames=13、training_context=1、crop_size=256、training_spatial_mode=paired_random_crop、evaluation_outputs=12。新manifest保存旧manifest的parent SHA256。三域clip数应保持：GoPro train+val22/test11，DVD66/5，BSD60/20。已完成的迁移不要重复运行；若新config已存在且正确，复用该config。若迁移半途失败留有部分文件，报告后使用另一个新的run目录重试，不删除旧产物。

## 3. 选择空闲GPU，执行预检和训练

先用nvidia-smi及compute-apps查询实际状态；即使此前8卡空闲也要重新查看。选择1或2张空闲A100，优先2张；禁止停止其他进程。将实际物理编号赋给CUDA_VISIBLE_DEVICES。每个模型最多2卡，默认quality完成后自动训练compact。

启动前确认新run无活进程，防止重复启动。然后：

```bash
: "${CUDA_VISIBLE_DEVICES:?先设置已确认空闲GPU的实际编号}"
nohup bash scripts/run_shift500.sh > "$SHIFT_RUN/launcher.log" 2>&1 &
printf '%s\n' "$!" > "$SHIFT_RUN/launcher.pid"
```

预检行为已经更新：

- 三个域各选择**原始分辨率最大的训练序列**，从连续13帧及对应GT裁相同256x256区域，在CPU完成裁剪后才上传GPU。
- 实际GPU训练input应为`[1,13,3,256,256]`，pred和GT应为`[1,11,3,256,256]`。执行teacher前向、student前向、反向和optimizer step；预检训练权重随后丢弃。
- 单独选manifest中最大原分辨率形状，使用16帧合成输入验证teacher/student的**无梯度整帧推理**显存，输出12帧。此步只检查形状和内存，不计算PSNR；仍没有任何整帧反向。
- 报告包含原始source_resolution、crop_box、实际input/output shape，以及crop训练和native inference各自peak_GiB。旧NATIVE_PREFLIGHT_OOM不再是这个新crop训练的gate。

两种student预检通过后，脚本自动进入teacher holdout参考、quality训练/测试、compact训练/测试和比较报告。不要停在预检成功等待用户确认。若仍有OOM，标明发生在crop训练还是native inference，附完整traceback；禁止静默变更T、crop尺寸、网络或评测窗口。

运行中：1卡累积4clips、2卡每卡累积2clips；每optimizer update精确覆盖GoPro2/DVD1/BSD1，共44张监督输出crop。每20步记录速度，每1k保存latest，每5k验证并保存best_gopro/best_balanced。val/test仍原分辨率、无crop；最终排名使用同一best_gopro checkpoint跑三个test域，另保存逐帧明细及输入|输出|GT预览。

## 4. 回报与恢复

回报STATUS、HUMAN_ACTION_REQUIRED、新commit/工作区是否干净、实际Python/PyTorch/CUDA、teacher SHA256、三域数量、训练crop/T/输出帧数、GPU编号、两种预检显存、当前update/速度/ETA。NOT_RUN必须如实填写；35dB尚未达到不能提前确认。

正常人工动作：用户把本任务交给CodeAgent即可，没有新增人工数据准备。旧完整任务书中的checkpoint恢复和推理命令仍可参考，但所有config/output都应指向**新crop run**，训练只能使用新crop配置。

来源：<https://github.com/dasongli1/Shift-Net/blob/450a4f246dedccd306aa0bc02d615d797874e1ce/options/gopro_deblur_small.yml>；<https://github.com/dasongli1/Shift-Net/blob/450a4f246dedccd306aa0bc02d615d797874e1ce/basicsr/data/video_image_dataset.py>；<https://github.com/dasongli1/Shift-Net/blob/450a4f246dedccd306aa0bc02d615d797874e1ce/basicsr/models/image_restoration1_model.py>。
