# Shift-Net三域压缩实验设计

目标：1080p每输出帧≤500 GFLOPs，GoPro test约35dB，同时训练GoPro、DVD、BSD。质量目标待实测。

## 两个模型

两者都从官方GShiftNet small结构派生，保留空间特征提取、GSTS融合、重建残差三段。高分辨率重复UNet从前3+后3减少到前1+后1；时域核心64→32通道。quality在半分辨率各保留1组GSTS、四分之一分辨率各保留2组；compact的四分之一分辨率再降至各1组。每组仍有4轮双向channel shift和spatial shift，CAB门控保留。

quality空间通道9、UNet增量4；compact空间通道8、增量3。改变宽度也会改变shift通道分组与位移分配，因此不能称为与原版等价的剪枝。未实际执行的orb4/orb5/rorb4/rorb5从部署模型移除，不影响teacher输出。改变宽度的参数重新初始化，只复制精确shape匹配的参数，训练时通过冻结的官方GoPro-small教师进行输出蒸馏。

不训练BSSTNet压缩版的原因：当前更优先解决算力口径、原版可复现性及混合数据训练闭环。BSSTNet的稀疏注意力/传播结构压到该预算需要另一个大幅结构改动实验。本轮两个对照保持相同Shift-Net家族，便于判断参数/深度损失。

实测计算图统计（不是GPU时间实测）：

| 模型 | 部署参数M | 1080p GMAC/输出帧 | 1080p GFLOPs/输出帧 | GoPro PSNR |
|---|---:|---:|---:|---|
| quality | 0.643947 | 231.059 | 475.933 | 待训练，目标35 |
| compact | 0.448255 | 188.877 | 389.390 | 待训练，目标35 |
| 同协议官方small教师 | 4.219075 | 1591.925 | 3248.662 | 待在本机权重/测试集验证 |

教师与用户旧表1490G不同：这里统一1MAC=2FLOPs，16输入12输出，并计主要标量算术。旧表可能混合MAC/FLOP与其他窗口长度，不能直接拼接。官方配置确实使用256 crop、n_sequence=13；本轮遵循用户要求使用原生整帧，故训练协议也不同。

## 训练与评估

180k updates/模型，每步4clips，GoPro/DVD/BSD=2/1/1，T16→12原生完整帧，全部阶段持续混合。log-MSE、质量掩码教师L1、GT相对帧差约束；无感知/GAN损失，PSNR优先。20k以后轻量时域项渐增；详细公式与权重在losses.py。

10%官方train acquisition留出验证；test不参与训练、蒸馏、选checkpoint。按val GoPro选一个checkpoint，使用同一个模型跑三个测试域；balanced另存参考。数据root、manifest、teacher文件固定哈希，运行结果与checkpoint全在源码checkout外。

## 算力口径

报告文件见reports/shift500。1MAC=2FLOPs，16输入12输出；包含重复halo计算，所有执行卷积/偏置/主要标量算术，memory copy不算FLOPs。每种参数量为实际部署模块参数数，不包含teacher或已删除的未执行模块（包括未使用的lrelu）。算力是meta计算图计数，非真实硬件FPS。

默认推理窗口最后不足12输出时仍计算12输出，丢弃部分成本由report/infer单独计入；极短镜头的平均每帧成本可能超过满块预算。quality刻意保留余量。不得用输入帧数作除数，也不能把GMAC直接写成同数值GFLOPs。FP16/BF16只改变数值格式，不把FLOPs减半。

官方Shift-Net README采用one_len=96，论文与本项目T16协议不同，不能直接声称已经复现35.22；同时有raw源码teacher适配器逐元素一致测试，防止RGB/时间索引问题。尚未下载官方checkpoint或运行GPU训练，本地验证只是代码级检查。

## 验证与限制

本地tests_shift500覆盖teacher随机权重与未包装上游的逐元素一致、两个student前反向和checkpoint重算梯度一致、完整帧覆盖、官方chunk索引、严格数据配对、GT指标与预算。本地7项测试通过，1项Gloo/DDP测试因环境禁止socket跳过；服务器必须补做。真实A100最大整帧反向预检由CodeAgent执行，随后开始训练。PSNR、显存、速度尚无服务器实测结果。

Proxy/SSL：CodeAgent应使用随附私人配置导出的http_proxy/https_proxy/HTTP_PROXY/HTTPS_PROXY；git clone前执行`git config --global http.proxy "$http_proxy"`、`git config --global https.proxy "$http_proxy"`和`git config --global http.sslVerify false`。完整执行步骤在CODEAGENT_SHIFTNET_500G_MIX_20260911.md。

来源：
- 官方代码/训练配置：<https://github.com/dasongli1/Shift-Net>，固定commit 450a4f246dedccd306aa0bc02d615d797874e1ce。
- CVPR 2023论文：<https://openaccess.thecvf.com/content/CVPR2023/html/Li_A_Simple_Baseline_for_Video_Restoration_With_Grouped_Spatial-Temporal_Shift_CVPR_2023_paper.html>。
