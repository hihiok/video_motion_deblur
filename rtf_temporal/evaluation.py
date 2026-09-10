"""Continuous-state full-frame evaluation against a fixed spatial baseline."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rtf_t6.datasets import read_rgb
from rtf_t6.losses import ssim
from .data import DOMAINS
from .flow import is_cut, pair_flow, warp


@torch.inference_mode()
def evaluate(model, manifest, cfg, device, spatial_only=False, preview_dir=None):
    model.eval()
    result = {'domains': {}, 'protocol': 'native RGB [0,1]; frame-mean PSNR; uniform-window SSIM; contiguous state'}
    dtype = torch.bfloat16
    for domain in DOMAINS:
        records = sorted((r for r in manifest['val'] if r['domain'] == domain), key=lambda r: r['name'])
        records = records[:cfg['validation']['max_sequences_per_domain']]
        psnrs, ssims, numer, denom, possible, frame_ids = [], [], 0., 0., 0., []
        for record in records:
            state, previous = None, None
            length = min(cfg['validation']['frames_per_sequence'], len(record['blur']))
            start = (len(record['blur']) - length) // 2
            frame_ids.append({'sequence': record['name'], 'start': start, 'length': length})
            for index in range(start, start + length):
                inp = read_rgb(Path(record['blur'][index])).unsqueeze(0).to(device)
                gt = read_rgb(Path(record['gt'][index])).unsqueeze(0).to(device)
                inp_np = inp[0].permute(1, 2, 0).cpu().numpy()
                gt_np = gt[0].permute(1, 2, 0).cpu().numpy()
                cut = previous is None or is_cut(previous[2], inp_np, cfg['train']['cut_threshold'])
                reset = torch.tensor([cut], device=device)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == 'cuda'):
                    pred, state = model.step(inp, state, reset, spatial_only=spatial_only)
                pred = pred.float()
                mse = (pred - gt).square().mean().item()
                psnrs.append(-10 * np.log10(max(mse, 1e-12)))
                ssims.append(float(ssim(pred, gt)))
                if previous is not None and not cut:
                    flow, mask = pair_flow(previous[3], gt_np)
                    flow, mask = flow[None].to(device), mask[None].to(device)
                    diff = pred - warp(previous[0], flow) - gt + warp(previous[1], flow)
                    numer += float((mask * diff.abs()).sum())
                    denom += float(mask.sum()) * 3
                    possible += mask.numel() * 3
                if preview_dir is not None and index == start + length // 2:
                    dest = Path(preview_dir)
                    dest.mkdir(parents=True, exist_ok=True)
                    panel = torch.cat((inp[0], pred[0], gt[0]), -1).permute(1, 2, 0).cpu().numpy()
                    img = Image.fromarray((panel.clip(0, 1) * 255).round().astype(np.uint8))
                    img.thumbnail((2400, 900))
                    img.save(dest / f'{domain}_{record["name"]}.jpg')
                previous = pred, gt, inp_np, gt_np
        result['domains'][domain] = dict(psnr=float(np.mean(psnrs)),
            ssim_uniform=float(np.mean(ssims)), aligned_temporal_l1=numer / max(denom, 1),
            valid_flow_fraction=denom / max(possible, 1), frames=len(psnrs), selections=frame_ids)
    result['balanced_psnr'] = float(np.mean([m['psnr'] for m in result['domains'].values()]))
    result['balanced_aligned_temporal_l1'] = float(np.mean([m['aligned_temporal_l1'] for m in result['domains'].values()]))
    return result


def qualifies(metrics, baseline, cfg):
    val = cfg['validation']
    return all(metrics['domains'][d]['psnr'] >= baseline['domains'][d]['psnr'] - val['max_psnr_drop_db_per_domain']
               and metrics['domains'][d]['aligned_temporal_l1'] <= baseline['domains'][d]['aligned_temporal_l1']
               and metrics['domains'][d]['valid_flow_fraction'] >= val['minimum_valid_flow_fraction']
               for d in DOMAINS)
