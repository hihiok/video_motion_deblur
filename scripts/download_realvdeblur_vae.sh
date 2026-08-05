#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/ssd1/z00919662/motion_deblur}"
TARGET="$ROOT/benchmark/weights/realvdeblur/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
EXPECTED_SHA256="38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
OFFICIAL_URL="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth?download=true"
mkdir -p "$(dirname "$TARGET")"

verify() {
  [[ -s "$TARGET" ]] || return 1
  local actual size
  size=$(stat -c '%s' "$TARGET")
  # Official file is 508 MB. Reject HTML, Xet pointer files and partial files.
  (( size > 500000000 )) || {
    echo "VAE file is too small ($size bytes); deleting invalid/partial file" >&2
    rm -f "$TARGET"
    return 1
  }
  actual=$(sha256sum "$TARGET" | awk '{print $1}')
  [[ "$actual" == "$EXPECTED_SHA256" ]] || {
    echo "VAE SHA256 mismatch: $actual" >&2
    rm -f "$TARGET"
    return 1
  }
  echo "Verified Wan2.1_VAE.pth: $actual"
}

if verify; then
  exit 0
fi

curl_flags=(-L --fail --retry 8 --retry-all-errors --connect-timeout 30 --continue-at -)
if [[ "${ALLOW_INSECURE_SSL:-0}" == "1" ]]; then
  echo "WARNING: TLS certificate verification disabled for this download." >&2
  curl_flags+=(-k)
fi

if command -v curl >/dev/null 2>&1; then
  echo "Downloading official VAE with curl..."
  if curl "${curl_flags[@]}" "$OFFICIAL_URL" -o "$TARGET" && verify; then
    exit 0
  fi
fi

# Fallback to huggingface_hub while disabling the optional Xet transport.
if command -v hf >/dev/null 2>&1; then
  echo "curl failed; retrying with Hugging Face CLI and HTTP fallback..."
  rm -f "$TARGET"
  HF_HUB_DISABLE_XET=1 hf download Wan-AI/Wan2.1-T2V-1.3B Wan2.1_VAE.pth \
    --local-dir "$(dirname "$TARGET")" || true
  verify && exit 0
fi

# Optional user-supplied mirror endpoint, for example an internal approved mirror.
if [[ -n "${HF_MIRROR_ENDPOINT:-}" ]]; then
  echo "Trying configured mirror endpoint: $HF_MIRROR_ENDPOINT"
  rm -f "$TARGET"
  mirror_url="${HF_MIRROR_ENDPOINT%/}/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth"
  curl "${curl_flags[@]}" "$mirror_url" -o "$TARGET" || true
  verify && exit 0
fi

echo "Failed to download the official Wan2.1 VAE." >&2
echo "Place the exact file at: $TARGET" >&2
echo "Required SHA256: $EXPECTED_SHA256" >&2
exit 1
