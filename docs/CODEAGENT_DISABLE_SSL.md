# CodeAgent: disable SSL verification for all download/install steps

## Purpose

The server is behind an internal HTTPS-inspection proxy and repeatedly raises `CERTIFICATE_VERIFY_FAILED` / SSL handshake errors. For this benchmark task, do **not** stop because of SSL certificate verification. Re-run downloads and package installs with certificate verification disabled.

This guide intentionally disables certificate verification. Use it only on the trusted internal server/network.

## Security constraints

- Never write proxy usernames, passwords, tokens, cookies, or full authenticated proxy URLs into this repository, logs, Markdown files, shell history, or commits.
- Reuse the proxy variables already configured by the server environment.
- Do not print `http_proxy`, `https_proxy`, `HTTP_PROXY`, or `HTTPS_PROXY` values.
- A download is accepted only after checking file type, nontrivial size, and SHA256. Disabling SSL does not mean accepting HTML error pages, Git-LFS pointers, or incomplete checkpoints.

## 1. Apply global no-SSL settings once

Run from the same Linux user that launches CodeAgent:

```bash
set -e

# Conda: disable certificate validation for all future commands by this user.
conda config --set ssl_verify false

# Git/Git-LFS: disable HTTPS certificate validation.
git config --global http.sslVerify false
git config --global http.version HTTP/1.1

# pip: trust all hosts used by this project.
python -m pip config set global.trusted-host \
  "pypi.org files.pythonhosted.org download.pytorch.org download.openmmlab.com huggingface.co cdn-lfs.huggingface.co cas-bridge.xethub.hf.co github.com raw.githubusercontent.com objects.githubusercontent.com"

# Do not fail because pip checks a newer version.
python -m pip config set global.disable-pip-version-check true

# Confirm only the non-secret settings.
conda config --show ssl_verify
git config --global --get http.sslVerify
python -m pip config get global.trusted-host
```

Expected values:

```text
ssl_verify: False
false
```

## 2. Export no-SSL variables before every CodeAgent run

```bash
export GIT_SSL_NO_VERIFY=true
export PYTHONHTTPSVERIFY=0
export SSL_NO_VERIFY=1
export HF_HUB_DISABLE_XET=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org download.pytorch.org download.openmmlab.com huggingface.co cdn-lfs.huggingface.co cas-bridge.xethub.hf.co github.com raw.githubusercontent.com objects.githubusercontent.com"
```

`HF_HUB_DISABLE_XET=1` avoids Xet transfer issues through the proxy. For large Hugging Face files, prefer `curl -k` or `wget --no-check-certificate` instead of relying on the Hugging Face CLI.

## 3. Disable SSL inside Python `requests` / `httpx`

Some tools such as `gdown` and some Hugging Face clients use Python HTTP libraries and ignore pip/conda settings. Create an isolated `sitecustomize.py` and inject it only into download commands.

```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur
NO_SSL_DIR="$ROOT/benchmark/no_ssl_python"
mkdir -p "$NO_SSL_DIR"

cat > "$NO_SSL_DIR/sitecustomize.py" <<'PY'
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

try:
    import requests.sessions

    _old_request = requests.sessions.Session.request
    if not getattr(_old_request, "_codeagent_no_ssl", False):
        def _request_no_ssl(self, method, url, **kwargs):
            kwargs["verify"] = False
            return _old_request(self, method, url, **kwargs)

        _request_no_ssl._codeagent_no_ssl = True
        requests.sessions.Session.request = _request_no_ssl
except Exception:
    pass

try:
    import httpx

    _old_client_init = httpx.Client.__init__
    if not getattr(_old_client_init, "_codeagent_no_ssl", False):
        def _client_init_no_ssl(self, *args, **kwargs):
            kwargs["verify"] = False
            return _old_client_init(self, *args, **kwargs)

        _client_init_no_ssl._codeagent_no_ssl = True
        httpx.Client.__init__ = _client_init_no_ssl

    _old_async_init = httpx.AsyncClient.__init__
    if not getattr(_old_async_init, "_codeagent_no_ssl", False):
        def _async_init_no_ssl(self, *args, **kwargs):
            kwargs["verify"] = False
            return _old_async_init(self, *args, **kwargs)

        _async_init_no_ssl._codeagent_no_ssl = True
        httpx.AsyncClient.__init__ = _async_init_no_ssl
except Exception:
    pass
PY

export PYTHONPATH="$NO_SSL_DIR${PYTHONPATH:+:$PYTHONPATH}"
```

Test the injection without printing proxy information:

```bash
python - <<'PY'
import ssl
print("default HTTPS context:", ssl._create_default_https_context.__name__)
try:
    import requests
    print("requests import: OK")
except Exception as exc:
    print("requests import:", repr(exc))
PY
```

## 4. Command rules

### Conda

Use the permanent config above, and add `-k` to create/install commands when possible:

```bash
conda create -n bsstnet python=3.8 -y -k
conda install -n <env> <packages> -y -k
```

Do not abandon an environment setup solely because of `CERTIFICATE_VERIFY_FAILED`.

### pip

Always include trusted hosts for non-PyPI wheel indexes:

```bash
conda run -n <env> env PYTHONPATH="$NO_SSL_DIR" \
  python -m pip install <packages> \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org \
  --trusted-host download.pytorch.org \
  --trusted-host download.openmmlab.com
```

For BSSTNet official versions:

```bash
conda run -n bsstnet env PYTHONPATH="$NO_SSL_DIR" \
  python -m pip install \
  torch==1.9.1+cu111 \
  torchvision==0.10.1+cu111 \
  torchaudio==0.9.1 \
  -f https://download.pytorch.org/whl/torch_stable.html \
  --trusted-host download.pytorch.org \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org

conda run -n bsstnet env PYTHONPATH="$NO_SSL_DIR" \
  python -m pip install \
  mmcv-full==1.7.1 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html \
  --trusted-host download.openmmlab.com \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org

conda run -n bsstnet env PYTHONPATH="$NO_SSL_DIR" \
  python -m pip install \
  -r /mnt/ssd1/z00919662/motion_deblur/envs/bsstnet_repo/requirements.txt \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org
```

### git and Git-LFS

```bash
GIT_SSL_NO_VERIFY=true git clone <repository-url> <destination>

cd <repository>
GIT_SSL_NO_VERIFY=true git lfs pull
```

If Git-LFS still fails, obtain the direct object URL and use `curl -k -L -C -`.

### curl

Always use `-k`, redirects, retries, and resume support:

```bash
curl -k -L \
  --retry 20 \
  --retry-delay 5 \
  --retry-all-errors \
  -C - \
  -o <output-file> \
  '<url>'
```

### wget

```bash
wget --no-check-certificate \
  --continue \
  --tries=20 \
  --timeout=60 \
  -O <output-file> \
  '<url>'
```

### gdown / Google Drive

Run `gdown` with the Python no-SSL injection:

```bash
NO_SSL_DIR=/mnt/ssd1/z00919662/motion_deblur/benchmark/no_ssl_python

conda run -n <env> env PYTHONPATH="$NO_SSL_DIR" \
  python -m gdown '<google-drive-url-or-id>' -O <output-file>
```

For a folder:

```bash
conda run -n <env> env PYTHONPATH="$NO_SSL_DIR" \
  python -m gdown --folder '<google-drive-folder-url>' -O <output-directory>
```

If Google Drive returns a permission/virus-confirmation HTML page, do not treat that as an SSL error and do not rename it to `.pth`.

## 5. RealVDeblur large Wan2.1 model

Expected target:

```text
/mnt/ssd1/z00919662/motion_deblur/benchmark/weights/realvdeblur/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors
```

Download without SSL checking and with resume:

```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur
OUT="$ROOT/benchmark/weights/realvdeblur/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
mkdir -p "$(dirname "$OUT")"

curl -k -L \
  --retry 20 \
  --retry-delay 5 \
  --retry-all-errors \
  -C - \
  -o "$OUT" \
  'https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/diffusion_pytorch_model.safetensors?download=true'
```

Do not redownload if a valid file already exists.

## 6. Mandatory validation after every download

```bash
FILE=<downloaded-file>

ls -lh "$FILE"
file "$FILE"
sha256sum "$FILE"
```

Reject the file if any of the following is true:

- zero bytes or unexpectedly small;
- `file` reports HTML/XML/text for a `.pth` or `.safetensors` checkpoint;
- the first line contains `version https://git-lfs.github.com/spec/v1`;
- the file contains an HTTP error page, Google Drive confirmation page, or proxy login page;
- a resumed download is larger than the expected remote object;
- safetensors header validation fails.

For a `.safetensors` file:

```bash
conda run -n deblur_runtime python - <<'PY'
from pathlib import Path
from safetensors import safe_open

path = Path("/path/to/file.safetensors")
with safe_open(str(path), framework="pt", device="cpu") as handle:
    keys = list(handle.keys())
print("safetensors OK; tensors:", len(keys))
print("first keys:", keys[:5])
PY
```

For a PyTorch `.pth` file:

```bash
conda run -n deblur_runtime python - <<'PY'
from pathlib import Path
import torch

path = Path("/path/to/file.pth")
obj = torch.load(path, map_location="cpu")
print("torch checkpoint OK; type:", type(obj))
PY
```

## 7. CodeAgent execution policy

CodeAgent must follow this order:

1. Export the variables in section 2.
2. Create and export the Python no-SSL injection in section 3.
3. Apply the global settings in section 1.
4. Retry the original failed download/install command with the corresponding no-SSL method.
5. Do not change model versions merely because SSL failed.
6. Do not rebuild a working environment merely because SSL failed.
7. Validate every downloaded artifact using section 6.
8. Continue to the next model when one model has a non-SSL failure; record the exact error and log path.

Treat these messages as SSL errors and retry without verification:

```text
CERTIFICATE_VERIFY_FAILED
certificate verify failed
self signed certificate in certificate chain
unable to get local issuer certificate
SSL: CERTIFICATE_VERIFY_FAILED
TLS certificate verification failed
```

Do not treat these as SSL errors:

```text
HTTP 401 / 403
Google Drive access denied
quota exceeded
file not found
checkpoint key mismatch
CUDA OOM
binary ABI mismatch
missing CUDA operator
```

## 8. Full smoke-run command after applying no-SSL setup

```bash
cd /mnt/ssd1/z00919662/motion_deblur/benchmark_code
git pull --ff-only

ROOT=/mnt/ssd1/z00919662/motion_deblur \
SOURCE_ENV=turtle_joint_py222 \
RUNTIME_ENV=deblur_runtime \
REAL_ENV=deblur_runtime \
SHIFT_ENV=deblur_runtime \
DST_ENV=deblur_runtime \
BSST_ENV=bsstnet \
SKIP_REALVDEBLUR=0 \
GPU=0 \
PYTHONPATH="/mnt/ssd1/z00919662/motion_deblur/benchmark/no_ssl_python${PYTHONPATH:+:$PYTHONPATH}" \
bash scripts/recover_after_codeagent.sh
```

## Optional cleanup after the benchmark

Only after all packages and weights are downloaded:

```bash
conda config --set ssl_verify true
git config --global http.sslVerify true
python -m pip config unset global.trusted-host || true
unset GIT_SSL_NO_VERIFY PYTHONHTTPSVERIFY SSL_NO_VERIFY HF_HUB_DISABLE_XET PIP_TRUSTED_HOST
```
