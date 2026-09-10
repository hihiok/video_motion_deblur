"""Training-only fixed Farneback flow on sharp GT; no downloaded flow weights.

Flow convention: flow(current, previous) gives offsets to sample previous.
OpenCV computation runs in dataset workers, capped at one thread each.
"""
import cv2
import numpy as np
import torch
from torch.nn import functional as F


FLOW_VERSION = 'farneback-half-v1'


def is_cut(previous, current, threshold=0.30):
    """Cheap conservative cut heuristic on RGB [0,1], shared train/inference."""
    a = cv2.resize(previous, (64, 36), interpolation=cv2.INTER_AREA)
    b = cv2.resize(current, (64, 36), interpolation=cv2.INTER_AREA)
    return bool(np.abs(a - b).mean() > threshold)


def pair_flow(previous, current):
    cv2.setNumThreads(1)
    h, w = current.shape[:2]
    size = (max(16, w // 2), max(16, h // 2))
    def gray(x):
        return cv2.resize(cv2.cvtColor((x * 255).clip(0, 255).astype(np.uint8),
                                      cv2.COLOR_RGB2GRAY), size, interpolation=cv2.INTER_AREA)
    a, b = gray(previous), gray(current)
    backward = cv2.calcOpticalFlowFarneback(b, a, None, .5, 4, 25, 5, 7, 1.5, 0)
    forward = cv2.calcOpticalFlowFarneback(a, b, None, .5, 4, 25, 5, 7, 1.5, 0)
    def full(flow):
        out = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        out[..., 0] *= w / size[0]
        out[..., 1] *= h / size[1]
        return out
    backward, forward = full(backward), full(forward)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = xx + backward[..., 0], yy + backward[..., 1]
    fw = cv2.remap(forward, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    error = np.square(backward + fw).sum(-1)
    bound = .01 * (np.square(backward).sum(-1) + np.square(fw).sum(-1)) + .5
    warped = cv2.remap(previous, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = ((mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1) &
            (error < bound) & (np.abs(current - warped).mean(-1) < .15))
    return (torch.from_numpy(backward.copy()).permute(2, 0, 1),
            torch.from_numpy(mask.copy()).unsqueeze(0).float())


def warp(previous, flow):
    _, _, h, w = previous.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=previous.device),
                            torch.arange(w, device=previous.device), indexing='ij')
    x = xx.float()[None] + flow[:, 0].float()
    y = yy.float()[None] + flow[:, 1].float()
    grid = torch.stack((2 * (x + .5) / w - 1, 2 * (y + .5) / h - 1), -1)
    return F.grid_sample(previous.float(), grid, mode='bilinear',
                         padding_mode='zeros', align_corners=False)


def aligned_error(pred, gt, flow, mask):
    """GT-relative temporal L1, normalized by valid channel-pixels only."""
    numer, denom = pred.new_zeros((), dtype=torch.float32), pred.new_zeros((), dtype=torch.float32)
    for t in range(1, pred.shape[1]):
        residual = (pred[:, t].float() - warp(pred[:, t - 1], flow[:, t - 1]) -
                    gt[:, t].float() + warp(gt[:, t - 1], flow[:, t - 1]))
        numer = numer + (mask[:, t - 1] * residual.abs()).sum()
        denom = denom + mask[:, t - 1].sum() * pred.shape[2]
    return numer / denom.clamp_min(1), numer.detach(), denom.detach()
