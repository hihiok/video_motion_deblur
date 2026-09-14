# CodeAgent：压缩 Shift-Net-s，按官方实际训练配方执行 V3

## 授权、优先级与目标

用户明确要求：尽量与 Shift-Net-s 一致，只压缩网络并增加 GoPro/DVD/BSD 联合训练。本文件替代旧 MIX V1、CROP V2 的训练任务。源码由 ChatGPT 修改并发布；CodeAgent 拉取、测试、运行及报告，不自行更改网络/超参数或绕过检查，不需要用户再手工改代码。

- 仓库：https://github.com/hihiok/video_motion_deblur
- 分支：`agent/shiftnet-500g-mix-v1`
- checkout：`/data/pub/z00919662/motion_deblur/shift500_code_v1`
- 已审计来源 run：`/data/pub/z00919662/motion_deblur/runs/shift500_mix_v1`
- **新 run**：`/data/pub/z00919662/motion_deblur/runs/shift500_official_recipe_v3`
- 沿用 `deblur_runtime`，允许 Python 3.9.23 / torch 2.2.2 / CUDA 11.8；不重装环境。
- quality、compact 分别训练，各自覆盖三个域；每个模型 300000 次训练迭代。目标仍为 GoPro test >=35 dB、1080p 每有效输出帧 <=500 GFLOPs。未实测不得宣布达标。
- 旧配置、旧 checkpoint、日志、OOM 报告全部保留。**V3 从随机初始化重新训练，不能接续 V1/V2 权重或优化器。** 相同 V3 配置的中断才可恢复。

## 已核对的官方执行配方

固定上游 commit：`450a4f246dedccd306aa0bc02d615d797874e1ce`。

| 项目 | V3 行为与上游依据 |
|---|---|
| 时间输入 | 同一视频连续 13 帧；训练前后各 1 帧上下文，监督中间 11 帧 |
| 空间输入 | 同一个 256×256 随机 crop 同步用于全部输入帧/GT；CPU 裁好再传 GPU；不 resize |
| 数据增强 | 同步水平翻转、垂直翻转、90°旋转；无时间倒序 |
| 帧范围 | 与官方 `n_frames_per_video=100` 一致，每个训练视频目录取前至多 100 帧，在其中遍历合法 13 帧窗口；不得跨视频目录 |
| 初始化 | 随机初始化，seed=10；不从 teacher 转移任何权重 |
| 实际损失 | **L1 mean，权重 1**；没有蒸馏、额外时域损失、log-MSE |
| 实际优化器 | **AdamW**，lr=4e-4，betas=(0.9,0.99)，weight_decay=0 |
| LR | CosineAnnealingLR 对应闭式余弦；T_max=300000，eta_min=1e-7；无 warmup。第 1 次迭代 lr=4e-4 |
| 精度与裁剪 | FP16 autocast + GradScaler，输入 half；unscale 后 clip_grad_norm=0.01 |
| batch | 每卡 microbatch=1 clip；全局每次迭代 8 clips，共 88 张输出 crop |
| 迭代数 | 每模型 300000；按原版 AMP 语义，溢出时 scaler 跳过优化器更新并调整 scale，日志记录 amp_skipped |

注意：官方 YAML 的 `pixel_opt: PSNRLoss` 被实例化但未参与 `optimize_parameters`，真正调用的是 `Loss2(opt['loss_type'])`，而 `loss_type=1*L1`。同样，YAML 写 `optim_g.type=Adam`，实际分支构造的是 `torch.optim.AdamW`。**以执行代码为依据，不按 YAML 名称误改。**

有意保留的差异：

1. 网络压缩：quality 0.643947M / 475.933 GFLOPs，compact 0.448255M / 389.390 GFLOPs。以上是 1080p、16 输入/12 输出、1 MAC=2 FLOPs 的既有实测计数，不是 crop 算力。尾部浪费计入最终有效输出预算。
2. 三域混合：全局 8 clips 中 GoPro4/DVD2/BSD2。各域内部对合法起点做确定性 epoch shuffle，不按视频等概率采样；每个域的每个合法窗口在一轮内恰好一次。
3. 默认仍用 1 或 2 张空闲卡：单卡累积 8 次，双卡各累积 4 次；全局 batch 与原版 8 卡×1 一致。激活 checkpointing 用于节省显存。浮点数值不承诺逐位复现 8 卡训练。
4. 每 1000 次迭代保存 latest，便于恢复；原版更稀疏。无需训练中验证或 best 选择，最终统一评测第 300000 次迭代 checkpoint。
5. 原分辨率推理仍按 16 输入/12 输出运行，以控制 1080p 显存；官方 README 示例 one_len=96（100 输入/96 输出）。两者时序窗口、边界与评分帧集合不同，**本工程 PSNR 不能直接当成论文 35.22 的严格复现**。teacher/student 用同一全帧协议、同一测试帧列表比较，保留逐帧数据。

数据划分：原 manifest 的 train+内部 val 合并回官方 train，保留 test 原样。应为 GoPro 22/11、DVD66/5、BSD60/20（train/test 视频数）。不再扣除内部 10% holdout。原来 5 个 GoPro 跨 split 共享 acquisition 仍只记录，不移动官方 chunk；完整 clip 重叠和跨 split GT 哈希审计保留。全源帧数与实际前 100 帧可用窗口数分别写入 `data_audit.json`。

## 1. 环境、Proxy、SSL 与拉取

先检查本项目旧 V1/V2 是否已有活跃进程。若有，记录 PID、命令、run、update；仅对确认属于本次 Shift500 的旧任务做温和 SIGTERM，等待保存退出，并确保其旧 launcher 不会继续启动下一模型。保留日志及 checkpoint，不使用 kill -9，不停止其他项目。不得在训练仍使用 checkout 时 pull。若无法确认进程归属，报告具体 PID/命令，不猜测终止。

沿用已有公司 proxy 配置，凭据来自环境激活脚本或之前私人附件；不输出密码、不使用 set -x。

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deblur_runtime
: "${http_proxy:?先加载已经配置好的私人proxy环境}"
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
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export SHIFT500_UPSTREAM=/data/pub/z00919662/motion_deblur/envs/Shift-Net-pinned-450a4f2
python -m pytest -q tests_shift500
```

若上游实际路径不同，从旧 config 的 upstream 字段设置环境变量。不修改上游哈希或放宽测试。作者本地 CPU 回归 20 passed、1 skipped（沙箱无法运行 Gloo socket）；服务器 DDP 测试须实际通过。GPU FP16/显存检查必须在服务器执行，不能将 CPU 测试称为已经完成训练。

## 2. 迁移为全新 V3 run

```bash
DEBLUR_ROOT=/data/pub/z00919662/motion_deblur
SHIFT_OLD_RUN="$DEBLUR_ROOT/runs/shift500_mix_v1"
SHIFT_RUN="$DEBLUR_ROOT/runs/shift500_official_recipe_v3"
python -m shift500.prepare_official \
  --from-config "$SHIFT_OLD_RUN/config.json" \
  --run "$SHIFT_RUN"
export SHIFT500_CONFIG="$SHIFT_RUN/config.json"
```

来源也可以是已成功准备的 V2 config，但新 run 必须为空且与旧 run 独立。已经成功迁移的新 config 不重复创建，检查 recipe 为 `shiftnet_official_recipe_crop256_t13_v3` 后复用。半途失败留有文件时保留原目录和错误报告，使用另一个新的 V3 run 名重试，不覆盖。

迁移检查旧 manifest/teacher SHA，合并仅来自官方 train 的内部 holdout，重算两模型部署预算，重写配置到新 run。沿用现有 teacher 文件：SHA256 `39f470a77b0b3d23ce5e1e8972e1213ba6cc73a097241af1021281443d2d4f00`。Teacher 仅作最终比较基线，**不进入学生初始化、训练或训练显存预检**。

## 3. 自动执行预检、两个模型训练及最终三域评测

重新用 nvidia-smi 和 compute-apps 检查实际空闲卡。选 1 或 2 张空闲 A100，优先 2 张，设置 CUDA_VISIBLE_DEVICES 为实际物理编号。确认新 run 没有活跃任务，防止重复启动。

```bash
: "${CUDA_VISIBLE_DEVICES:?设置1或2张已确认空闲卡的编号}"
nohup bash scripts/run_shift500.sh > "$SHIFT_RUN/launcher.log" 2>&1 &
printf '%s\n' "$!" > "$SHIFT_RUN/launcher.pid"
```

脚本自动顺序完成：quality/compact 预检 → quality 300k及三个test域 → compact 300k及三个test域 → teacher 三域同协议参考 → comparison。预检通过后不要暂停等待用户确认。

预检分别检查：

- 每域最大源分辨率序列，实际训练 tensor `[1,13,3,256,256]`，pred/GT `[1,11,3,256,256]`；L1、FP16 GradScaler、unscale、clip0.01、有效 optimizer step。允许有界 scale 回退，但必须真正完成有限梯度更新才 PASS。
- 最大原始分辨率的学生无梯度整帧推理，16→12；检查输出形状、有限性、显存。没有整帧反向要求。
- 记录 source_resolution、crop_box、训练/推理 peak_GiB、AMP scale。预检权重丢弃；训练进程重新用 seed10 随机初始化。

训练每20步记录 loss、lr、grad_norm、amp_scale、amp_skipped/amp_skipped_total、速度；每1k原子保存 model/optimizer/scaler/update/config。最终使用同一 `latest.pth`（update=300000）跑三个 test 域，**不使用旧 best_gopro/best_balanced**。不得利用 test 选 checkpoint。35dB 由最终真实 PSNR 判断，失败如实写 TARGET_NOT_MET。

中断恢复：新 shell 重做环境/proxy、设置相同 SHIFT500_CONFIG 和已确认空闲 GPU，然后重跑 launcher；训练自动恢复同配置的 V3 latest。若 OOM 或非有限数持续出现，记录阶段、tensor shape、完整 traceback，不擅自减少 T、crop、batch 或模型宽深，也不恢复旧蒸馏配方。

## 4. 回报

回报 STATUS、HUMAN_ACTION_REQUIRED、Git commit/工作区、Python/Torch/CUDA、GPU、recipe、三域 train/test 与可用窗口数、参数/GFLOPs、两模型 crop/native 预检、当前迭代、速度/ETA、AMP skipped、最终 PSNR及逐帧报告路径。未执行填 NOT_RUN。无需用户手工准备数据或改文件；具体无法自动处理的阻塞需附证据。

官方来源：

- [GoPro Ours-s 配置](https://github.com/dasongli1/Shift-Net/blob/450a4f246dedccd306aa0bc02d615d797874e1ce/options/gopro_deblur_small.yml)
- [视频数据加载与裁剪](https://github.com/dasongli1/Shift-Net/blob/450a4f246dedccd306aa0bc02d615d797874e1ce/basicsr/data/video_image_dataset.py)
- [实际 optimizer/loss/AMP 训练入口](https://github.com/dasongli1/Shift-Net/blob/450a4f246dedccd306aa0bc02d615d797874e1ce/basicsr/models/image_restoration1_model.py)
- [Loss2 实现](https://github.com/dasongli1/Shift-Net/blob/450a4f246dedccd306aa0bc02d615d797874e1ce/basicsr/loss/__init__.py)
