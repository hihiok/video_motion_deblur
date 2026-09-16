# CodeAgent：从 REDS 作者官方 Hugging Face 源继续下载（替代 OpenDataLab 空仓库）

## 1. 目标、来源与边界

本任务直接继续，不再要求用户选择来源。停止重试 `OpenDataLab/REDS`：用户最新报告已确认，该仓库只有 3 个文件、1087 bytes，`raw/REDS.tar.gz.00` 是空目录占位包；这不是 SSL、登录或下载不完整造成的。本任务不修改 OpenXLab SDK 的 `rprint` bug。

数据来源已由 REDS 官方网页交叉确认：
- 官方网页：https://seungjunnah.github.io/Datasets/reds.html
- 作者维护的公开数据仓库：https://huggingface.co/datasets/snah/REDS/tree/main
- 固定快照：`62dc25d16e6f43d2214f1b365023abda86f7a0ae`
- 官方网页于 2025-03-03 宣布增加 Hugging Face 下载，替代旧 SNU CVLab 服务器；不要使用过期的 SNU 链接。

延续上次“完整下载”的范围：下载固定快照中 **全部 snah/REDS 文件**，不只下载 README，不只下载四个核心包。网页当前列出 17 个 ZIP 和 2 个元文件，约 110 GB 压缩包；真实文件大小及哈希以下载脚本取得的官方 API 清单为准。

重点检查普通视频去模糊的 `train_blur.zip`、`train_sharp.zip`、`val_blur.zip`、`val_sharp.zip`，但完整仓库内的 comp / jpeg / bicubic 版本也应下载并独立保留，不能混入普通 blur。`test_blur.zip` 有公开下载，完整分辨率 `test_sharp.zip` 未公开提供；不能拿 bicubic sharp 或 val sharp 当 test GT。

不额外下载另一仓库 `snah/REDS_orig` 的 120fps 源帧，也不下载 `REDS_VTSR`；这些不是当前普通 REDS 仓库的一部分。这不影响本任务对标准 REDS 仓库的全量覆盖。

仅负责数据获取与校验，不启动训练，不改现有模型、代码、采样比例、GoPro/DVD/BSD，不采用 REDS4 重划分，不将 val 合入 train。

## 2. 复用现有路径，保留旧记录

```text
工作根：       /data/pub/z00919662/dataset
代理说明：     /data/pub/z00919662/dataset/proxy.md
代码目录：     /data/pub/z00919662/dataset/tools/reds_openxlab_repo
复用独立 venv：/data/pub/z00919662/dataset/.venvs/reds_openxlab
新原包目录：   /data/pub/z00919662/dataset/downloads/REDS_official/snah_REDS
解压目录：     /data/pub/z00919662/dataset/REDS
新报告目录：   /data/pub/z00919662/dataset/reports/reds_official_hf
GitHub 仓库：  https://github.com/hihiok/video_motion_deblur.git
分支：         agent/reds-openxlab-download-20260916
新下载代码：   tools/reds_official/download.py
离线测试：     tools/reds_official/test_download.py
```

保留旧 `downloads/REDS/OpenDataLab___REDS/`、空的 `REDS/REDS/` 及 `reports/reds_openxlab/FINAL_REDS_DOWNLOAD_REPORT.md`，不删除、不移动、不覆盖来掩盖旧来源问题。新 ZIP 依据包内原生路径直接解压到 REDS 根；不要因为旧的空嵌套目录，再额外人为增加一层 REDS。

## 3. Proxy 与 SSL（所有网络操作之前）

完整读取 `/data/pub/z00919662/dataset/proxy.md`，按其中实际命令配置代理。它是 Markdown，不能未经检查直接 `source`；也不要 `cat` 到对外日志或报告中。不要在 GitHub 文档、脚本、shell 历史截图中复制用户名、密码、AK/SK、Token。

```bash
set +x
umask 077
# 先在当前 shell 执行 proxy.md 内实际的 export http_proxy=... / https_proxy=...。
# 凭据以服务器文件为准，本公开任务书不重复其内容。
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
```

有可信公司 CA 时传 `--ca-bundle <实际存在的CA路径>`。无可用 CA 或仍然出现证书错误时，用户已授权此下载任务使用 `--insecure`；脚本会对自己的每次 Requests 调用显式传 `verify=False`，包括元数据、ZIP Range 预检、下载和续传，不污染系统 Python、SDK 或其他进程。

关闭 TLS 校验会失去服务器身份认证保护，只对本任务使用，不永久关闭系统校验；官方哈希有助于检查文件一致性，但不能取代可信 TLS 的来源认证。不得修改全局 sitecustomize、PYTHONPATH，或声称 `git http.sslVerify=false` 同时修好了 Python HTTPS。

Git 拉取前，复用 proxy.md 已配置的 Git proxy；仅本次 Git 命令允许跳过校验：

```bash
ROOT=/data/pub/z00919662/dataset
REPO="$ROOT/tools/reds_openxlab_repo"
ENV="$ROOT/.venvs/reds_openxlab"
RAW="$ROOT/downloads/REDS_official/snah_REDS"
OUT="$ROOT/REDS"
REPORT="$ROOT/reports/reds_official_hf"

cd "$REPO"
git status --short
# 核对 origin 是 hihiok/video_motion_deblur；不得把含认证信息的 remote 输出到报告。
# 分支应为 agent/reds-openxlab-download-20260916，工作区应干净。
git -c http.sslVerify=false pull --ff-only origin agent/reds-openxlab-download-20260916
git rev-parse HEAD
```

已有未提交改动时先报告并同步 `git diff`，不要 reset、强行覆盖或改动正在训练的 checkout。不要固定回旧提交 `70da6b580276acbba8acdf41ee8dcf05daf6f3f1`，它没有新脚本。

## 4. 环境与测试

复用已验证成功的独立 Python 3.9.23 venv。不要重新登录 OpenXLab，不要重复创建环境，不在 base / deblur_runtime 安装、升级或卸载任何依赖。新脚本直接使用 Requests，不依赖 OpenXLab / Hugging Face SDK / hf_xet / git-lfs / GPU，也无需 OpenXLab AK/SK。

```bash
mkdir -p "$RAW" "$OUT" "$REPORT"
"$ENV/bin/python" -c 'import sys,requests; print(sys.version); print(requests.__version__)'
"$ENV/bin/python" -c 'from PIL import Image; print("Pillow OK")'
```

只有 Pillow 不存在时才向这个 venv 安装固定兼容版本：

```bash
"$ENV/bin/python" -m pip install 'Pillow==10.4.0'
# 若 pip 仅因公司证书失败且没有可信 CA，允许本次命令：
# "$ENV/bin/python" -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org 'Pillow==10.4.0'
```

现有 Requests 2.28.2 可复用，不要因下载任务去升级它。若 Requests 缺失或环境与报告不一致，先报告 ENVIRONMENT_MISMATCH，不随意修改 base。

```bash
"$ENV/bin/python" "$REPO/tools/reds_official/test_download.py"
```

交付时本地 28 项离线测试通过；CodeAgent 必须在服务器再跑一次。离线测试不表示服务器能访问远端，也不代表全量数据已下载。

## 5. 先做真实文件预检，再直接全量执行

保留当前 shell 的代理设置，使用一个任务进程、串行下载和串行解压，不用 GPU，不占满 CPU。不要再请求“是否切换来源”的确认。

```bash
# 缺可信 CA 或已知证书错误时：
TLS=(--insecure)
# 确认可信 CA 文件存在且可用时改为：
# TLS=(--ca-bundle "$HOME/.config/ssl/org-ca-bundle.pem")

COMMON=(--raw "$RAW" --output "$OUT" --report-dir "$REPORT" "${TLS[@]}")

"$ENV/bin/python" -u "$REPO/tools/reds_official/download.py" \
  "${COMMON[@]}" --plan-only
```

预检自动取得固定快照的完整大小与官方 LFS SHA256 / Git blob SHA1，拒绝缺少核心包、无官方哈希或空占位包；再对每个 ZIP 读取少量 Range 数据和 central directory，验证路径安全、实际图片成员存在，计算真正解压字节数和所需磁盘空间（另留 10 GiB）。

仅当预检退出 0 且状态为 `PLAN_ONLY` 时继续；PLAN_ONLY 不是 DOWNLOAD_SUCCESS。检查 `preflight.json` 和 `manifest.json`。

```bash
set -o pipefail
LOG="$REPORT/run_$(date +%Y%m%d_%H%M%S).log"
"$ENV/bin/python" -u "$REPO/tools/reds_official/download.py" \
  "${COMMON[@]}" 2>&1 | tee "$LOG"
```

允许通过 CodeAgent 自身的持久终端会话保持长下载运行；不得只启动后台进程就报告 SUCCESS。必须等待实际完成，或明确报告 RUNNING/PARTIAL 与可恢复命令。

断网后可以重跑同一命令，脚本保留 `.part` 和身份 sidecar，通过 HTTP Range 续传；已完成包必须与官方哈希一致才跳过。若服务器忽略 Range，拒绝把完整文件追加到部分文件末尾，也不删除部分文件。若哈希失败，保留现场，不自行清空重下。

通过原始 Hugging Face resolve 链接自动获取新 CDN/Xet 签名 URL，不保存、硬编码、打印临时签名 URL。无需部署任何 Hugging Face SDK；也不要把 `HF_HUB_DISABLE_SSL_VERIFICATION` 当作通用 SSL 修复。

## 6. 解压、配对与成功条件

脚本对每个 ZIP 逐成员读取，验证 ZIP CRC、安全路径及文件类型；拒绝路径穿越和软链接。已有目标文件只有大小和 CRC 与 ZIP 一致才跳过，否则保留原件并停止。保存所有下载原包。

解压后，普通 train/val 的 blur 与 sharp 必须逐序列、逐帧名一一对应，不能只抽查 3 个视频就把全量配对标成 PASS。当前标准 REDS 的验收预期为：

- train：240 个视频，每个 100 帧，即 24,000 blur + 24,000 sharp。
- val：30 个视频，每个 100 帧，即 3,000 blur + 3,000 sharp。
- 普通 blur/sharp 分辨率：宽 1280、高 720。

脚本对所有普通配对 PNG 检查头部尺寸，并在 train/val 各抽至少 3 个分散序列的首、中、末帧，用 Pillow 真正解码 blur 和 sharp。包内目录为准，不把 val 重命名为 test，不把同名但不同 split 的序列混用。

test sharp 标记 `NOT_PROVIDED_BY_SOURCE`，不计算其 PSNR，不用别的版本伪造 GT。超分、MPEG、JPEG 变体保持独立；全包覆盖与普通去模糊配对结果分别报告。

只有所有源文件哈希验证、所有 ZIP CRC/解压和普通 train/val 全量配对均通过，脚本才输出 SUCCESS。单个文件下载成功、预检成功、目录存在、下载进程启动都不构成成功。

## 7. 阻塞与人工操作

本次选源已确定，不需要人工 AK/SK 或选择来源；先按公开、匿名 HTTPS 下载执行。

- 407/ProxyError：检查本地 proxy.md；不得泄露凭据。
- SSL_ERROR：使用可信 CA 或任务局部 `--insecure`，不要反复重装库。
- 403/401：可能是访问策略或代理/CDN 阻断，不自动认定需要 HF Token。记录阻断主机名和 HTTP 状态，不记录 URL 签名。不能索要 OpenXLab AK/SK。
- `RANGE_NOT_SUPPORTED` / `CONTENT_RANGE_MISMATCH`：保留部分文件和报告；不要自行改续传实现或退回空 OpenDataLab 源。
- `DISK_SPACE_ERROR`：报告需要和可用字节数，让用户提供容量/明确清理范围；不删除已有数据，不擅自把全仓下载改成四个包。
- 下载代码/数据结构不符：报告真实 code commit、阶段、脱敏错误。不得自行写/改代码；已经改过则输出完整 `git diff`，由用户传回 ChatGPT 同步。

只有实际触发需要人工权限、网络放行、磁盘处理或代码修复时才输出 HUMAN_ACTION_REQUIRED: YES，并明确事项。正常成功为 NO。

最终报告：`/data/pub/z00919662/dataset/reports/reds_official_hf/FINAL_REDS_OFFICIAL_REPORT.md`。
另保留 `manifest.json`、`preflight.json`、`report.json`、真实脱敏运行日志。回报 Git commit、HF snapshot、文件总数、压缩/解压大小、官方哈希检查、ZIP CRC、train/val 视频/帧数与配对结果、解码抽查、test GT 可用性和人工事项。
