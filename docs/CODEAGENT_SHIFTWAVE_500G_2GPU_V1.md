# CodeAgent：Shift-Net Ours-s + Haar / LL-GSTS / PAGF / HF，双卡训练

本任务独立于旧 Shift500 quality/compact 和 DSTNet 压缩任务。本文件是本分支的唯一训练入口；不要执行继承目录里旧任务书的停训练、加到4/8卡、quality/compact两轮训练等指令。

## 等比例更新（V2）

用户最新要求：三个域严格平均，GoPro:DVD:BSD=1:1:1。每次更新各2 clips，全局batch6；两张卡每卡依次读取三个域各1 clip，累积3次。已新增3项等比例及checkpoint迁移测试并通过。120k更新、LR、loss和网络保持原值。新的每次更新样本数由8变6，因此不承诺与旧配方优化轨迹等价。

旧配置是4:2:2，不能直接复用：使用本文件新的equal checkout/run。若旧ShiftWave没有启动，直接从官方预训练初始化。若旧ShiftWave已在训练，只对确认属于该ShiftWave run的launcher及trainer做温和停止，等待latest保存（不影响旧Shift500或其他任务）。在新的checkout里使用下列迁移命令，保留权重/optimizer/scaler/update，从该进度继续等比例采样；此前已完成的更新仍属于旧比例，报告中必须注明。

```bash
python -m shiftwave.prepare \
  --from-config /data/pub/z00919662/motion_deblur/runs/shiftwave_500g_2gpu_v1/config.json \
  --run /data/pub/z00919662/motion_deblur/runs/shiftwave_500g_equal_2gpu_v2 \
  --continue-checkpoint /data/pub/z00919662/motion_deblur/runs/shiftwave_500g_2gpu_v1/student/latest.pth
```

此命令在完成下面环境/proxy及下载步骤后执行，只接受尚未完成的ShiftWave checkpoint，不接受quality/compact权重。迁移成功后跳过第2节prepare创建，直接复用新config。原run及checkpoint全部保留；禁止直接改旧config以绕过resume校验。

## 目标和授权

- 仓库：https://github.com/hihiok/video_motion_deblur
- 分支：`agent/shiftnet-waveshift-pagf-500g-v1`
- 新 checkout：`/data/pub/z00919662/motion_deblur/shiftwave_500g_equal_code_v2`
- 新 run：`/data/pub/z00919662/motion_deblur/runs/shiftwave_500g_equal_2gpu_v2`
- 环境：复用 `deblur_runtime`，兼容 Python 3.9 / PyTorch 2.2.2 / CUDA 11.8；不替换现有 torch。
- **只用两张确认空闲的 A100-80GB 做 DDP 训练。** 不停止其他任务，不复用旧任务的 checkout 或输出目录。
- 只训练 student 一种架构，无消融；teacher 固定参数，仅推理，不训练第二个模型。
- 目标：1080p 每有效输出帧严格 `<500 GFLOPs`，完整 GoPro test RGB8 PSNR `>=33.0 dB`。
- 用户给出的 Ours-s `1490G` 是 **FLOPs**，不解释为 MACs。当前 pinned 实现按16输入/12输出计数为3248.662 GFLOPs，两者差异尚未查明；不要改写用户数据或声称完成1490G原协议复现。新模型用完整前向的独立计数作为验收证据。
- **33dB尚未实测。** 不能因为 loss 下降、蒸馏或 FLOPs 通过就标记 TARGET_MET。最终未达标，保留所有数据并报告 `TARGET_NOT_MET`。
- ChatGPT 已完成代码；CodeAgent 负责下载、验证、运行、汇报，不自行写/改模型、训练、评测代码或降低门槛。不需要等待人工确认才开始已授权的训练。

## 设计与已验证范围

参考用户分享：https://chatgpt.com/share/6aa0d643-755c-83ec-8563-22b4e44b65ed

已核对分享中最终方向：GSTS放LL，高频不做shift。代码来源：
https://github.com/hihiok/video_motion_deblur/blob/agent/nanovnr-waveshift-pagf-t6-fullframe-20260907/nanovsr_deblur/models/network_nanovnr_waveshift_pagf.py

结构：RGB stem14 → feature Haar → LL14（Shift-Net空间编码与GSTS时空融合 → PAGF → 空间恢复）与 HF42（按三个子带分组卷积、可学习Laplacian、LL条件）→ inverse Haar → RGB residual。

- 保持原版宽度14/64，保留两个前置和两个后置空间UNet。
- LL浅层重复GSTS保留1/3，LL深层保留2/3；上游源文件不修改。
- 高频分支预测残差，不直接做固定锐化；edge_scale从0开始。
- 复用原版所有保留张量的精确形状权重；PAGF/HF新模块初始化。该转换**不是函数等价裁剪**，不能直接使用官方权重宣称已有33dB。
- 保持Shift-Net非因果窗口方式，没有引入NanoVSR的跨窗口循环状态，不将两个框架的状态机制混用。
- 本版不依赖可重参数化部署转换；训练和部署结构一致，444.41G不依赖未验证的卷积融合。

已完成CPU检查：14项测试通过（6项新模型/评测测试、8项共享数据/配方回归），包括Haar可逆与梯度、保留权重加载、全部参数参与反向、activation checkpointing梯度一致、奇数尺寸padding/裁除、RGB8逐帧配对、非有限输出拒绝、联合验收门槛。

形状计数结果：参数 **2.329256M**；1080p、16输入/12输出，每输出帧 **444.410869 GFLOPs**。已计所有卷积、bias、Haar与非线性标量操作（非线性按名义1 op记）；shift、张量搬运不记浮点运算，速度/带宽要另测。每MAC按2个FLOPs。报告：`reports/shiftwave/profile_1080p.json`。

作者环境无CUDA，Gloo socket被运行环境限制，因此未完成真正双卡/NCCL、FP16或80GB显存验证；服务器launcher会自动执行，必须通过才长训。不能把CPU结果当作GPU验证。

## 1. 环境、Proxy、SSL及下载

凭据沿用用户给定的私人代理配置/环境激活脚本；私人指令附件包含可直接执行的设置。本公开仓库只引用环境变量，不存密码。不要 `set -x` 或打印代理值。用户已要求本项目Git跳过SSL校验，按以下设置执行，无需再次询问。

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deblur_runtime
# 激活脚本若没有代理，先执行私人附件的Proxy段。
: "${http_proxy:?先加载用户已有的私人proxy配置}"
export https_proxy="$http_proxy" HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$http_proxy"
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=true
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
DEBLUR_ROOT=/data/pub/z00919662/motion_deblur
SHIFTWAVE_CODE="$DEBLUR_ROOT/shiftwave_500g_equal_code_v2"
SHIFTWAVE_RUN="$DEBLUR_ROOT/runs/shiftwave_500g_equal_2gpu_v2"
if [ ! -e "$SHIFTWAVE_CODE" ]; then
    git clone --branch agent/shiftnet-waveshift-pagf-500g-v1 --single-branch \
        https://github.com/hihiok/video_motion_deblur.git "$SHIFTWAVE_CODE"
else
    # 先检查没有进程正在使用这个新checkout；活跃时不得pull。
    test -d "$SHIFTWAVE_CODE/.git"
    test "$(git -C "$SHIFTWAVE_CODE" branch --show-current)" = agent/shiftnet-waveshift-pagf-500g-v1
    test -z "$(git -C "$SHIFTWAVE_CODE" status --porcelain)"
    git -C "$SHIFTWAVE_CODE" pull --ff-only origin agent/shiftnet-waveshift-pagf-500g-v1
fi
cd "$SHIFTWAVE_CODE"
git rev-parse HEAD
python -c 'import torch,numpy,PIL; print(torch.__version__,torch.version.cuda,torch.cuda.is_available())'
```

只在缺少轻量依赖时安装（不要无条件重装已有环境）：

```bash
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org 'numpy<2' 'Pillow>=9' 'pytest>=7'
```

不安装其他模型库，不重建conda环境，不下载新的大数据集。已有数据在 `/data/pub/z00919662/dataset`，从已成功审计的配置读取实际配对路径。

## 2. 复用已审计数据，创建独立run

优先源配置：`/data/pub/z00919662/motion_deblur/runs/shift500_official_recipe_v3/config.json`。若不存在，查旧 `runs/shift500_mix_v1/config.json`，只读其 manifest / teacher / upstream 路径；两个都不存在时报告缺失路径，不虚构manifest或绕过审计。

上游固定commit：`450a4f246dedccd306aa0bc02d615d797874e1ce`。模型加载会校验 `gshift_deblur2.py` 的SHA256，不能自行补丁。若旧config内上游目录不存在，允许克隆到新的独立目录并传 `--upstream`，不要覆盖任何已有上游代码。

```bash
SHIFTWAVE_SOURCE="$DEBLUR_ROOT/runs/shift500_official_recipe_v3/config.json"
if [ ! -f "$SHIFTWAVE_SOURCE" ]; then
    SHIFTWAVE_SOURCE="$DEBLUR_ROOT/runs/shift500_mix_v1/config.json"
fi
test -f "$SHIFTWAVE_SOURCE"
export SHIFT500_UPSTREAM=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["upstream"])' "$SHIFTWAVE_SOURCE")
# 如果上述上游不存在：在新目录clone官方仓库、checkout固定commit，再更新本shell变量即可。
python -m pytest -q tests_shiftwave
if [ ! -f "$SHIFTWAVE_RUN/config.json" ]; then
    python -m shiftwave.prepare --from-config "$SHIFTWAVE_SOURCE" \
        --run "$SHIFTWAVE_RUN" --upstream "$SHIFT500_UPSTREAM"
fi
export SHIFTWAVE_CONFIG="$SHIFTWAVE_RUN/config.json"
```

prepare会检查：来源manifest/teacher哈希；官方train/test划分及既有跨split GT哈希审计；GoPro train22、test11且完整1111帧；13帧训练可用窗口；学生模型完整FLOPs；GoPro实际序列末尾重复/丢弃输出的额外算力。GoPro在train/test共享acquisition名字但chunk不同的官方split已被原审计允许，不能再次错误拦截或改动官方split。

此前内部holdout合回官方train，无训练中test选模。每条训练序列取前至多100帧用于窗口采样，**test必须使用完整1111帧**，不限制前100。继承数据集路径与blur配对，不自动替换成blur_gamma。

DVD如只有5/10个test序列，可继续已授权训练并评测现有序列，但报告标记不完整，不能称为完整DVD榜单；不会因此阻止完整GoPro目标验收。

已经完整准备的config可复用；半成品目录保留错误记录，另取新run后缀重做，不删除旧结果或覆盖源配置。

## 3. 固定训练配方

| 项目 | 执行值 |
|---|---|
| 输入 | 连续13帧，所有帧与GT共用一个256×256随机crop，不resize |
| 监督 | 中间11帧；预测与GT同尺寸 |
| 数据 | GoPro2 / DVD2 / BSD2，每个全局batch共6 clips，严格1:1:1 |
| 双卡 | 每卡microbatch1，各累积3次；DDP2进程，FP16 GradScaler |
| 初始化 | 官方Ours-s预训练权重中的所有保留张量，严格shape加载 |
| Teacher | 同一个官方Ours-s，全尺寸256 crop、同一13帧输入，取中间11帧；eval/no_grad |
| 学生loss | L1(GT) + 0.05×Haar高频L1(GT) + λ×L1(teacher output) |
| 蒸馏 | λ从0.1余弦降到0；第100000次之后关闭teacher |
| 迭代 | 固定120000次；最后20000次只有GT损失；不声称是作者官方300k复现 |
| 优化器 | AdamW，lr1e-4→1e-7余弦，betas0.9/0.99，weight_decay0 |
| 梯度 | unscale后clip_norm0.01，activation checkpointing开启 |
| checkpoint | 每1000保存latest；每20000保留里程碑；原子写入，含optimizer/scaler/update/config |
| test选模 | 禁止；使用固定第120000次的latest |
| 推理 | 原分辨率整帧，16输入/12输出，空间padding到8倍数后裁回，不tile/TTA/resize |

本任务沿用最近Shift500已采用的crop256/T13输入策略；Haar是网络内部可逆分解，未丢弃高频。不要自行改整帧反向或减小帧长/改变学习率来躲OOM。CPU线程按命令限制，workers每rank2。

## 4. 两卡自动检查和启动

先记录现有训练PID/命令/占卡情况，**保留所有其他任务**。下面自动选两张没有计算进程、利用率<=10%、空闲显存>=60000MiB的A100。没有两张空闲卡就报告 `TWO_IDLE_GPUS_UNAVAILABLE`，不抢占或擅自改单卡。编号不可写死为0/1或4/5。

```bash
export CUDA_VISIBLE_DEVICES=$(python -m shiftwave.gpus)
python -m shiftwave.gpus --validate "$CUDA_VISIBLE_DEVICES"
nohup bash scripts/run_shiftwave_2gpu.sh > "$SHIFTWAVE_RUN/launcher.log" 2>&1 &
printf '%s\n' "$!" > "$SHIFTWAVE_RUN/launcher.pid"
```

launcher持有run文件锁，防止重复启动。自动依次执行：

1. 两卡真实DDP梯度累积/两次更新，检查所有参数有有限梯度、两rank权重完全一致；合成测试权重丢弃。
2. 单卡分别检查三域最大源图的256 crop，真实teacher+student FP16前反向及有效optimizer step；允许GradScaler有界退档。检查原生最大分辨率（至少1080p）16→12推理；测试权重丢弃。
3. 真实双卡训练2次更新并保存，然后恢复同一checkpoint，接续到120000。不是重复训练两轮。
4. 学生在GoPro/DVD/BSD现有全部test帧上统一评测，保存每帧RGB8和float PSNR及输入/输出/GT预览。
5. Teacher在完整GoPro test同协议评测，作为蒸馏基线参考；其失败不抹掉学生已完成的结果，附日志。
6. 写入 `FINAL_REPORT.json`，同时检查完整GoPro1111帧、RGB8 PSNR>=33、1080p名义和GoPro实际有效输出两种FLOPs都<500。

预检通过直接继续训练，不人为停下来等确认。若OOM、非有限数或校验失败，保留完整错误、checkpoint与日志，明确阶段、GPU显存、tensor shape；不要绕过门槛。没有真正测出的字段填 `NOT_RUN`。

中断恢复：重新加载环境/proxy，复用同一config和该run的latest，确认没有活动launcher并选两张空闲卡后重跑启动命令。DDP预检会重新执行；生产checkpoint不被预检覆盖。旧Shift500 quality/compact权重不能作为本模型resume。

速度：前200次更新后按recent_seconds_per_update估算剩余时间；前100k有teacher开销，后20k无teacher，分别报告。不要沿用旧quality模型1.92s/update做承诺。

## 5. 必须回报与代码同步

回报：STATUS、HUMAN_ACTION_REQUIRED、Git commit与工作区是否干净、环境版本、两张物理GPU编号、数据audit、444.41G及有效输出tail预算、DDP/AMP/1080p预检、teacher权重SHA与初始化张量数、update/120000、loss/kd_weight/AMP skipped、双卡速度/ETA、checkpoint和日志路径。完成后附完整GoPro RGB8/float PSNR、DVD完整性、BSD指标、TARGET_MET或TARGET_NOT_MET。

如果GPU资源/数据/权限等确实需要用户操作，明确写要人工做什么及原因，不能只报HUMAN_ACTION_REQUIRED: YES。

如果发现服务器代码已被CodeAgent改动：不reset、不覆盖，保存 `git status --short`、`git diff --binary` 和所有新增源码/修改文件清单到run的同步目录；日志中不带代理密码。能推送时提交到**新的repair分支**供ChatGPT核对，不能覆盖本任务发布分支；无法同步时明确请用户把diff及新增源码发回本聊天。未同步的代码不能声称与发布commit一致。

## 来源

- Shift-Net官方源码及训练配置：https://github.com/dasongli1/Shift-Net/tree/450a4f246dedccd306aa0bc02d615d797874e1ce
- 本任务在官方预训练张量上做架构压缩和蒸馏，不承诺复制论文35.22dB。
