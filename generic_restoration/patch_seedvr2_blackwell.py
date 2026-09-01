#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ATTENTION_IMPORT = "from flash_attn import flash_attn_varlen_func"
ATTENTION_FALLBACK = '''try:
    from flash_attn import flash_attn_varlen_func
    SEEDVR_ATTN_BACKEND = "flash_attn"
except Exception:
    SEEDVR_ATTN_BACKEND = "torch_sdpa_varlen"

    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
                              max_seqlen_q, max_seqlen_k, **kwargs):
        """Inference-only, non-causal varlen fallback for Blackwell setup rescue."""
        del max_seqlen_q, max_seqlen_k, kwargs
        outputs = []
        for index in range(cu_seqlens_q.numel() - 1):
            q0, q1 = int(cu_seqlens_q[index]), int(cu_seqlens_q[index + 1])
            k0, k1 = int(cu_seqlens_k[index]), int(cu_seqlens_k[index + 1])
            query = q[q0:q1].transpose(0, 1).unsqueeze(0)
            key = k[k0:k1].transpose(0, 1).unsqueeze(0)
            value = v[k0:k1].transpose(0, 1).unsqueeze(0)
            result = torch.nn.functional.scaled_dot_product_attention(query, key, value)
            outputs.append(result.squeeze(0).transpose(0, 1))
        return torch.cat(outputs, dim=0)
'''


OLD_SAVE = '''                if sample.shape[0] == 1:
                    mediapy.write_image(filename, sample.squeeze(0))
                else:
                    mediapy.write_video(
                        filename, sample, fps=save_fps
                    )'''


NEW_SAVE = '''                output_folder = os.path.join(tgt_path, os.path.splitext(os.path.basename(path))[0])
                os.makedirs(output_folder, exist_ok=True)
                for frame_index, frame in enumerate(sample):
                    mediapy.write_image(
                        os.path.join(output_folder, f"{frame_index:08d}.png"), frame
                    )'''


def replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"Expected patch context not found in {path}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply pinned, auditable SeedVR2 inference patches.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--apex-fallback", action="store_true")
    parser.add_argument("--sdpa-fallback", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    inference = repo / "projects" / "inference_seedvr2_3b.py"
    attention = repo / "models" / "dit_v2" / "attention.py"
    config = repo / "configs_3b" / "main.yaml"
    changes = []
    if replace_once(inference, "video_list = os.listdir(video_path)", "video_list = sorted(os.listdir(video_path))"):
        changes.append("deterministic_input_sort")
    if replace_once(inference, OLD_SAVE, NEW_SAVE):
        changes.append("lossless_png_output")

    if args.sdpa_fallback and replace_once(attention, ATTENTION_IMPORT, ATTENTION_FALLBACK):
        changes.append("torch_sdpa_varlen_fallback")

    if args.apex_fallback:
        text = config.read_text(encoding="utf-8")
        updated = text.replace("fusedrms", "rms").replace("fusedln", "layer")
        if updated != text:
            config.write_text(updated, encoding="utf-8")
            changes.append("torch_native_norm_fallback")

    marker = {
        "patch": "seedvr2_blackwell_business_v1",
        "apex_fallback": args.apex_fallback,
        "sdpa_fallback": args.sdpa_fallback,
        "changes_this_run": changes,
    }
    (repo / ".business_benchmark_patch.json").write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()
