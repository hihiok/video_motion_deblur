# CodeAgent：下载 OpenDataLab/REDS，并修复下载进程的代理 / SSL 问题

## 1. 本次任务与边界

本次目标是 **REDS，不是 BSD**。数据仓库固定为 `OpenDataLab/REDS`。
用户要求通过 OpenXLab 下载完整仓库；不得只下载 README 后报告成功。
不下载 BSD，不启动训练，不改模型、训练配置或已有 GoPro / DVD / BSD 数据。
不要擅自缩减成仅几个 REDS 文件；空间不足时保留已下载文件并报告。

- 工作根：`/data/pub/z00919662/dataset`
- 代理说明：`/data/pub/z00919662/dataset/proxy.md`
- 原始下载根：`/data/pub/z00919662/dataset/downloads/REDS`
- 解压目标根：`/data/pub/z00919662/dataset/REDS`
- 独立环境：`/data/pub/z00919662/dataset/.venvs/reds_openxlab`
- 独立代码目录建议：`/data/pub/z00919662/dataset/tools/reds_openxlab_repo`
- 报告目录：`/data/pub/z00919662/dataset/reports/reds_openxlab`
- GitHub：`https://github.com/hihiok/video_motion_deblur`
- 分支：`agent/reds-openxlab-download-20260916`
- 下载包装器：`tools/reds_openxlab/cli.py`
- 离线测试：`tools/reds_openxlab/test_cli.py`

使用 GitHub 上已准备好的代码，不自行重写或修改包装器。遇到代码问题时，提供脱敏报错与 commit；若已作本地改动，提交 `git diff` 供用户转交，不隐藏改动、不带改动继续执行。

## 2. 已知错误：不要把 SSL 错误误判为账号或数据集错误

此前日志显示 `openxlab login` 在向 `openapi.openxlab.org.cn` 发起 HTTPS 请求时出现：

```text
SSLCertVerificationError: self-signed certificate in certificate chain
```

它首先证明证书校验失败，不能据此判断 AK/SK 正确或错误，也不能判断 REDS 是否可访问。
此外，原先在 base 安装 openxlab 后，requests 被降至 2.28.2，出现依赖冲突。
本次不要继续安装、升级、卸载或修复 base 中的软件包，也不要改 deblur_runtime。

## 3. Proxy 与 SSL：先配置，再 clone / 安装 / 登录

先完整读取服务器上的 `/data/pub/z00919662/dataset/proxy.md`，只执行其中与本任务相关的代理设置。
这是 Markdown，不要未经检查就 `source proxy.md`。
代理地址、用户名与 URL-encoded 密码以该文件为准；不要使用旧服务器路径或编造代理。
若文件缺失或无法读取，报告 `BLOCKED_PROXY_CONFIG`，不要索取聊天中的明文密码。

确保当前 shell 已配置 `http_proxy`、`https_proxy`，并同步大写变量：

```bash
set +x
umask 077
# 先执行 proxy.md 中真实的 export http_proxy=... / export https_proxy=...。
# 不要在报告、截图、日志、Git 或本 md 中复制凭据。
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

不要输出 `env`、代理变量值、完整 Git 配置或 AK/SK。
不要用 `set -x`；敏感配置只留在服务器受限权限的配置文件中。

优先用现有可信公司 CA bundle（proxy.md 指定路径，或检查 `$HOME/.config/ssl/org-ca-bundle.pem`）；不存在就不要假定已经配置成功。
用户允许本下载任务跳过 SSL verification；无可用 CA 或仍报证书错误时，使用后文包装器的 `--insecure`，而不是无休止寻找证书。
关闭校验会失去服务器身份认证保护，因此只用于本次明确授权的下载进程，不能扩展为系统级长期关闭。

如需 clone / pull，本次可以用单次命令禁用 Git SSL 校验；无需污染全局配置：

```bash
# 先按 proxy.md 配置好代理，然后在新的、空的目标目录中执行。
git -c http.proxy="$https_proxy" -c http.sslVerify=false clone \
  --depth 1 --single-branch --branch agent/reds-openxlab-download-20260916 \
  https://github.com/hihiok/video_motion_deblur.git \
  /data/pub/z00919662/dataset/tools/reds_openxlab_repo
```

已有该目录时先检查 remote、branch、`git status --short`，干净且身份一致才 fast-forward pull；禁止 reset、覆盖或切换正在训练的代码目录。
记录实际 commit。Git 的 `http.sslVerify=false` 只处理 Git，不能当作已经修好 Python requests。

## 4. 独立下载环境

先检查是否已有本任务专用环境，可以复用经过确认的环境；不要反复创建。
优先借用现有 `deblur_runtime` 的 Python 创建一个不共享 site-packages 的 venv，**这不会向 deblur_runtime 安装包**：

```bash
ROOT=/data/pub/z00919662/dataset
ENV="$ROOT/.venvs/reds_openxlab"
REPO="$ROOT/tools/reds_openxlab_repo"
RAW="$ROOT/downloads/REDS"
OUT="$ROOT/REDS"
REPORT="$ROOT/reports/reds_openxlab"
mkdir -p "$ROOT/.venvs" "$RAW" "$OUT" "$REPORT"

# 仅在 ENV 不存在且确认 deblur_runtime 可用时执行：
conda run -n deblur_runtime python -m venv "$ENV"
source "$ENV/bin/activate"
python --version
command -v python
```

如果 deblur_runtime 不存在或没有 venv 支持，可以使用已存在的 Python 3.9–3.11 创建隔离 venv；或按 proxy.md 建独立 conda Python 3.10 环境。不得为此修改 base 的 requests/setuptools 等依赖。
不要盲目用 base Python 3.12 与旧 SDK 的依赖组合；创建失败时保留真实错误，不冒充已激活环境。

仅在确认 Python 指向本下载环境后安装：

```bash
python -m pip install 'openxlab==0.1.3'
python -m pip check
python "$REPO/tools/reds_openxlab/test_cli.py"
```

无需重复执行 `pip install openxlab` 与 `pip install -U openxlab`。
安装失败时检查代理和证书；有可信 CA 可给 pip 使用 `--cert`。
若只有 PyPI 证书校验失败，允许本次安装命令加：

```bash
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org 'openxlab==0.1.3'
```

使用内部镜像时只按实际镜像主机处理；不要添加无关 trusted-host。
pip 的 trusted-host 不会自动关闭 OpenXLab requests 的 SSL 校验。
若环境已被其他任务使用或存在冲突，不得就地强行降级，报告环境冲突。

## 5. 包装器与登录

包装器调用已安装 OpenXLab 的原生 CLI 入口，只在该进程内设置 Requests 的 TLS 行为。
它不修改 SDK 源码、不写 sitecustomize、不更改 PYTHONPATH 或系统 SSL 设置；默认保留 SSL 警告。
当前测试是离线单元测试，**不是本服务器登录或实际下载成功的证明**。

在当前 shell 定义下面数组。无可信 CA / 已遇到证书失败时：

```bash
OXL=("$ENV/bin/python" "$REPO/tools/reds_openxlab/cli.py" --insecure --)
```

若确认存在可用可信 CA，则使用这一行替代，路径以实际检查为准：

```bash
OXL=("$ENV/bin/python" "$REPO/tools/reds_openxlab/cli.py" --ca-bundle "$HOME/.config/ssl/org-ca-bundle.pem" --)
```

先复用当前用户已有有效登录，不输出认证配置内容。尝试：

```bash
"${OXL[@]}" dataset info --dataset-repo OpenDataLab/REDS
```

若返回未登录/登录过期，则执行：

```bash
"${OXL[@]}" login
```

登录需要 AK/SK 时，用户在服务器的交互终端输入，不要让用户发到聊天或写进命令参数。
CodeAgent 没有可安全移交的交互终端时，输出当前环境路径和上述完整登录命令，由用户在自己的终端执行后继续。
这时状态为 `WAITING_FOR_USER_AUTH`，`HUMAN_ACTION_REQUIRED: OPENXLAB_AK_SK_IN_TERMINAL`，不能报成功，也不是数据下载失败。
不猜测、搜索、生成或记录 AK/SK。允许 SDK 按正常流程写入用户私有认证配置；检查该配置权限，不删除已有账号配置。
如果网页要求接受使用条款/申请访问，由用户人工完成；不要代替用户接受新条款。
登录后重新运行 info 与 ls，真正验证该账号能访问该仓库。

## 6. 查看完整清单、检查空间，再下载

```bash
"${OXL[@]}" dataset info --dataset-repo OpenDataLab/REDS
"${OXL[@]}" dataset ls --dataset-repo OpenDataLab/REDS
"${OXL[@]}" dataset get --help

df -h "$ROOT"
df -i "$ROOT"
```

如 ls 需要额外参数或分页，以当前 CLI help 为准获取完整清单，不把截断列表当全量。
保存脱敏后的仓库文件清单、文件大小、可获取的官方校验值。
根据实际清单和 archive header 估算空间；同时留出原包、解压数据与工作余量。
不使用未经确认的固定容量估算；不要删除其他数据腾空间。

可用 README 作为小文件网络测试，但其成功不等于完整数据已下载：

```bash
"${OXL[@]}" dataset download \
  --dataset-repo OpenDataLab/REDS \
  --source-path /README.md \
  --target-path "$REPORT"
```

README 路径若不存在，只能按真实仓库列表选择已有小文件；禁止猜造数据文件名或签名 URL。
真正完整下载命令是：

```bash
"${OXL[@]}" dataset get \
  --dataset-repo OpenDataLab/REDS \
  --target-path "$RAW"
```

普通官方 CLI 等价命令为 `openxlab dataset get --dataset-repo OpenDataLab/REDS --target-path "$RAW"`；本任务经包装器执行，避免 SSL 修复只对 login 生效。
SDK 可能在目标路径下加仓库子目录，检查并记录实际落盘路径，不假定目录结构。
保持已下载文件及 SDK 缓存。续传/已完成文件跳过以 SDK 的实际行为为准，不编造 `--resume` 参数或保证字节级续传。
遇到暂时性网络错误最多额外重试 3 次、逐步退避；遇到 AUTH / ACCESS / SSL / 空间错误先解决分类问题，不无限重试。
仅凭进程退出码不能判定成功：检查 CLI 错误输出，并将下载结果与完整仓库清单逐项核对。
不擅自切换第三方镜像；OpenXLab 不可用则报告具体阻塞与已下载进度。

## 7. 校验与解压：保留 REDS 原生结构

下载后对所有原包做对应格式的完整性检查（例如 zip 用 `unzip -t`，7z 用 `7z t`，tar 先遍历检查）。
有官方 checksum 就比对；本地 SHA256 只作为本地指纹，不能冒充官方一致性校验。
分卷包必须齐全才能解压；不覆盖旧数据，不删除唯一原包。解压前检查路径穿越/绝对路径/危险链接及磁盘余量；遇到不安全条目停止。
使用系统已有解压工具，不为下载数据另写大型处理程序；原包保留在 RAW，解压到 OUT。
并发解压最多 2 个，不使用 GPU，不占满 CPU。

以仓库实际内容及 README 为准，记录各子集实际目录。
视频去模糊重点核对 `train_blur` 与 `train_sharp`、`val_blur` 与 `val_sharp`；目录名称可以有外层嵌套，不能据此误判缺失。
`*_blur_comp` 是另一个退化版本，`*_bicubic` 是低分辨率版本，必须分开记录，不与普通 blur 混用。
下载范围仍是完整 OpenXLab 仓库，不因上述训练重点而自动跳过其他仓库文件。

保留官方 train / val / test 划分，不能将 val 改名为 test；不改成 BSD 的目录模板。
不要采用 REDS4 等重划分方案，不把验证或测试数据混进训练，不修改现有三数据集采样比例。
逐序列检查普通去模糊 train / val 的 blur 与 sharp 帧名集合、数量及相对帧对应关系；抽查至少各 3 个序列图片能解码、尺寸匹配。
保留原分辨率、原帧名与顺序，不 resize / crop / 重编码。
若来源未提供 test sharp GT，报告 `NOT_PROVIDED_BY_SOURCE`，不能伪造 GT 或拿 val 充当 test；但来源清单明确有而本地缺失的文件必须报未下载完整。
数据量、分辨率均实际统计，不将网上常见数字直接写成测量结果。

## 8. 交付与状态

报告保存到 `$REPORT/FINAL_REDS_DOWNLOAD_REPORT.md`。报告若作为后续 CodeAgent 交接文档，必须写清 proxy.md 路径、SSL_METHOD 和实际复现命令，但不能复制凭据。
记录代码 commit、Python/OpenXLab 版本、目标路径、原生目录、来源清单对账、校验依据、未完成事项与必要人工操作。
若仍在下载，报告 `IN_PROGRESS` 与真实可核验进度，不提前报告完成；不要只给“已启动”就当任务完成。

```text
STATUS: SUCCESS / IN_PROGRESS / WAITING_FOR_USER_AUTH / BLOCKED
DATASET: REDS
DATASET_REPO: OpenDataLab/REDS
DOWNLOAD_SCOPE: FULL_REPOSITORY
PROXY_CONFIG: /data/pub/z00919662/dataset/proxy.md
SSL_METHOD: CA_BUNDLE / VERIFY_DISABLED
CODE_COMMIT:
ENV_PATH:
PYTHON_VERSION:
OPENXLAB_VERSION:
LOGIN_AND_ACCESS: PASS / FAIL / NOT_TESTED
RAW_ROOT: /data/pub/z00919662/dataset/downloads/REDS
EXTRACTED_ROOT: /data/pub/z00919662/dataset/REDS
ACTUAL_DOWNLOAD_SUBDIRECTORY:
REPO_FILE_COVERAGE: <matched / expected; or UNKNOWN with reason>
ARCHIVE_INTEGRITY: PASS / FAIL / NOT_APPLICABLE / NOT_TESTED
OFFICIAL_CHECKSUM: PASS / FAIL / NOT_PROVIDED_BY_SOURCE / NOT_TESTED
TRAIN_BLUR_DIR:
TRAIN_SHARP_DIR:
VAL_BLUR_DIR:
VAL_SHARP_DIR:
TRAIN_SEQUENCES:
VAL_SEQUENCES:
TRAIN_BLUR_FRAMES:
TRAIN_SHARP_FRAMES:
VAL_BLUR_FRAMES:
VAL_SHARP_FRAMES:
TEST_DATA_AVAILABLE:
TEST_GT_STATUS:
OTHER_VARIANTS:
PAIRING_CHECK: PASS / FAIL / NOT_TESTED
IMAGE_DECODE_AND_SIZE_CHECK: PASS / FAIL / NOT_TESTED
RAW_SIZE:
EXTRACTED_SIZE:
HUMAN_ACTION_REQUIRED: NO / <specific action>
ERROR_CLASS: NONE / PROXY / SSL / AUTH / ACCESS_DENIED / DISK / NETWORK / INTEGRITY / ENV / CODE
REPORT_PATH:
```

`SUCCESS` 要求仓库清单对账完成、原包完整性检查通过、所需解压完成、普通去模糊训练/验证配对检查通过；无法获取完整清单时不能宣称整库完整。
只完成 README、登录、下载启动、部分文件或抽查，不得报告 `SUCCESS`。

## 9. 参考依据

- OpenXLab 官方 CLI 文档（get 整库、download 单文件、target-path）：https://openxlab.org.cn/docs/developers/数据集/数据集CLI（命令行工具）.html
- OpenXLab 下载说明：https://openxlab.org.cn/docs/en/datasets/下载数据集.html
- SDK 发布与版本：https://pypi.org/project/openxlab/
- Requests SSL / CA / proxy：https://requests.readthedocs.io/en/latest/user/advanced/
- REDS 作者页面与数据类型：https://seungjunnah.github.io/Datasets/reds.html

这些资料用于解释参数和校验类型，不表示已验证用户账号权限或服务器下载能力。
