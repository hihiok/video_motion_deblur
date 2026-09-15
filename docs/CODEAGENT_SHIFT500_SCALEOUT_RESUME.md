# CodeAgent：Shift500 V3 加卡提速，接续当前 checkpoint

用户已授权加卡并要求避免低利用率。直接执行本文件，替代旧任务书“每模型最多2卡”限制。本次只改变执行方式，V3 config、网络、训练数据、global batch8、loss、LR及300k计划均不变。不要重建run或从零训练。

- GitHub：https://github.com/hihiok/video_motion_deblur
- 分支：`agent/shiftnet-500g-mix-v1`
- checkout：`/data/pub/z00919662/motion_deblur/shift500_code_v1`
- 现有run：`/data/pub/z00919662/motion_deblur/runs/shift500_official_recipe_v3`
- 环境：`deblur_runtime`，Python3.9.23/torch2.2.2/CUDA11.8继续使用。
- 用户报告：quality约40300/300000，GPU4/5，1.92s/update，AMP稳定。执行时读取最新实际进度，不将40300写死。
- CodeAgent只拉取作者代码、运行命令、检查和回报。不得自行改源码/训练config。发现本地已有修改，导出脱敏diff与新增文件供用户转交ChatGPT，再处理；不要reset覆盖修改。
- 不创建cron，不修改OpenCode/gateway配置，不等待训练全部完成才回报。

## 1. 先确认资源，再温和保存退出

只读检查nvidia-smi、compute-apps、GPU拓扑、CPU核数/负载、内存、run/launcher.pid、进程树及quality/training.jsonl。识别现有训练rank0、其他rank、torchrun和launcher的PID与所属run。不要只凭PID文件判断进程归属。

确认除当前GPU4/5外至少还有2张可用A100，优先争取总计8张；4卡或8卡必须全部属于当前任务或已核实空闲，禁止停止其他任务。先记下旧CUDA_VISIBLE_DEVICES。没有额外资源时保留当前训练并如实报告，不先停止训练。

对**确认属于此run的训练rank0 Python进程**发送SIGTERM（不是先杀torchrun或launcher）。旧train.py会在更新边界全rank同步停止，原子保存latest并退出75；旧launcher有pipefail，应随之退出而不会启动compact。分次检查保存和退出，等待期间保持状态更新。不得kill -9；若无法正常退出，报告具体PID/日志，不能拉取代码或启动第二个任务。

退出后确认：

- 旧launcher、torchrun和全部训练rank已经结束；没有残留进程会自动启动下一模型。
- quality/latest.pth的update不低于停止前已落盘checkpoint，模型、optimizer、scaler存在且有限，variant/config一致；记录实际update及SHA256。
- 保存旧launcher.log、launcher.pid的备份到独立本次加卡目录或用时间戳副本；不删除旧日志、checkpoint、监控配置。

## 2. 环境、Proxy、SSL、pull和测试

公司proxy沿用已有环境激活脚本或私人proxy附件，包含用户已提供的账号和URL编码密码。代理凭据不得打印或提交到公共GitHub。每个新shell先激活环境并加载已有proxy，再执行下列配置；不需要用户重新输入凭据。

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deblur_runtime
: "${http_proxy:?加载现有proxy环境或私人proxy附件}"
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
export SHIFT_RUN=/data/pub/z00919662/motion_deblur/runs/shift500_official_recipe_v3
export SHIFT500_CONFIG="$SHIFT_RUN/config.json"
export SHIFT500_UPSTREAM="$(python -c 'import json,os; print(json.load(open(os.environ["SHIFT500_CONFIG"]))["upstream"])')"
python -m pytest -q tests_shift500
```

作者本地验证：24项通过、1项因Gloo socket限制跳过；新增检查覆盖换卡后的样本集合、AdamW接续数学一致性、选卡规则，以及真实trainer短测不改正式文件。服务器必须补做实际多卡/FP16短测，本地CPU测试不代表GPU吞吐验证。

全局样本顺序按update索引保存：换卡后相同下一update仍为原来8个clip；模型、AdamW状态、GradScaler、AMP skip计数和LR进度均接续。新运行选项另存runtime.json/checkpoint.execution，不修改原config，因而旧V3 checkpoint直接兼容。DDP归约顺序不同允许微小浮点误差，不声称逐位相同。

## 3. 实测选卡与执行参数（短测权重全部丢弃）

确认现在候选GPU全部空闲。设置SHIFT_SCALE_GPUS为4或8个实际物理编号，**原训练卡放在最前两个**以便2卡对照；不能把示例当作已验证空闲。例如全部8卡均可用时为 `4,5,0,1,2,3,6,7`，只用4卡时可以是 `4,5,0,1`。

```bash
: "${SHIFT_SCALE_GPUS:?设置4或8张实际已验证空闲GPU编号，原两卡在前}"
SHIFT_TRIAL_DIR="$SHIFT_RUN/scaleout_$(date +%Y%m%d_%H%M%S)"
python -m shift500.scaleout \
  --config "$SHIFT500_CONFIG" \
  --resume "$SHIFT_RUN/quality/latest.pth" \
  --variant quality \
  --gpus "$SHIFT_SCALE_GPUS" \
  --output "$SHIFT_TRIAL_DIR"
```

脚本已由作者编写，不需要CodeAgent写benchmark代码。它会：

1. 复制并校验一份不可变resume_snapshot.pth，每个短测从完全相同的checkpoint、optimizer/scaler和样本起点开始。
2. 比较2卡对照，以及4/8卡的activation checkpointing开/关。每次20步预热+80步计时，真实数据、完整前反向及optimizer step。试验结果不写入正式training.jsonl、latest.pth或初始化记录，不计入正式训练进度。
3. 每2秒采样各张候选GPU利用率和显存，只统计计时区间；报告最慢rank的data wait比例、峰值显存、实际s/update及AMP skip。
4. 若最快加卡配置等待数据超过10%，或平均GPU利用率低于70%，再测一次每rank4个单线程DataLoader worker。默认每rank2个，prefetch=2；最多8卡×4=32个单线程worker。若CPU资源不允许本次候选卡数×4个worker，开始前就回报资源限制，保留现有训练，不无上限增worker。
5. 默认选平均利用率至少60%、最慢卡平均至少40%的更快4卡配置。只有8卡比最佳4卡再快至少30%、平均利用率>=70%、最慢卡平均利用率>=50%，才选8卡。候选必须计时区间AMP无新增skip（预热时允许正常GradScaler回退并报告）、PyTorch峰值显存低于单卡总容量90%；加卡结果至少比当次2卡对照快15%。阈值用于避免浪费卡，不是宣称所有情况下能达到这些利用率。
6. 写trials.json、每次日志/GPU采样、selected.json及不含凭据的selected_env.sh。短测OOM仅作该候选失败，不改变模型或crop；其余配置继续测。

每个短测最长15分钟；正常短测总计通常是数分钟到十几分钟，不能一直等待不报告。若子进程SIGTERM后仍不退出，停止后续短测并报告其进程组，只处理本脚本创建的子进程，不停止其他任务。

没有selected_env.sh时，先确认所有短测已退出。若仅因加卡收益不足或候选OOM而选型失败，**恢复原两卡继续已有训练**，附trials.json向ChatGPT报告瓶颈；不要让正式训练长时间空等。若发现配置/checkpoint被其他进程改变或身份不一致，先解决重复任务，不能盲目重启。

## 4. 立刻接续正式训练

```bash
source "$SHIFT_TRIAL_DIR/selected_env.sh"
```

再次检查所选GPU没有新进程占用，确认旧launcher和短测均已退出。将旧launcher.log和launcher.pid按时间戳备份后执行：

```bash
nohup bash scripts/run_shift500.sh > "$SHIFT_RUN/launcher.log" 2>&1 &
printf '%s\n' "$!" > "$SHIFT_RUN/launcher.pid"
```

新launcher会使用所选卡数/worker/checkpointing设置完成预检，自动从正式quality/latest.pth恢复，随后quality训练/评测→compact训练/评测→teacher比较及报告。保留双模型顺序流程。关闭activation checkpointing只省去反向重算，state_dict和网络计算目标不变；显存以GPU预检和短测为准。

- 4卡：每rank microbatch1、累积2次；8卡：每rank microbatch1、累积1次。
- 两者每update仍是GoPro4/DVD2/BSD2，总8clips；不增加batch、lr或裁剪尺寸来“刷利用率”。
- 新日志有recent_seconds_per_update，用稳定区间速度估计剩余时间，不用刚启动的加载时间或旧进程累计速度。
- 恢复后观察至少200次更新，确认update从暂停点继续、loss/AMP正常，GPU实际利用率和短测一致。达到下一个正式1k checkpoint后确认模型/optimizer/scaler与新commit正确保存，随后回报并结束本轮；launcher继续后台运行。
- 若运行速度明显低于短测或长期低利用率，收集各GPU采样、data wait、CPU/IO现象，先报告，不自行改源码。所有试验和正式执行均无需新建定时监控。

## 5. 回报内容

STATUS、HUMAN_ACTION_REQUIRED、commit/工作区、停止前后与恢复update、checkpoint SHA、GPU选择、global batch/accumulation、checkpointing、worker数、2/4/8卡对照s/update、每卡平均利用率、峰值显存、data wait、相对2卡加速比、正式恢复200步后的速度/AMP、质量模型剩余ETA及输出目录。compact耗时尚未实测，单独标为估计。

300k是对齐官方的迭代计划，不是35dB的充分条件，也不能据此声称缩短训练必然达不到35dB。除将本任务交给CodeAgent外，正常无需用户手工改代码或停止进程；实际权限/资源阻塞需说明具体人工动作。
