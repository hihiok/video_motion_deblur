"""Spatial and motion-preserving temporal objectives for video deblurring."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x.square() + eps * eps).mean()


def fft_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_fft = torch.fft.rfft2(prediction.float(), norm="ortho")
    target_fft = torch.fft.rfft2(target.float(), norm="ortho")
    return charbonnier(torch.view_as_real(pred_fft - target_fft))


def spatial_gradient(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return dx, dy


def edge_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx, pred_dy = spatial_gradient(prediction)
    target_dx, target_dy = spatial_gradient(target)
    return 0.5 * (charbonnier(pred_dx - target_dx) + charbonnier(pred_dy - target_dy))


def temporal_difference_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match GT motion instead of suppressing all change between frames."""
    pred_delta = prediction[:, 1:] - prediction[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    return charbonnier(pred_delta - target_delta)


def temporal_acceleration_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape[1] < 3:
        return prediction.new_zeros(())
    pred_acc = prediction[:, 2:] - 2 * prediction[:, 1:-1] + prediction[:, :-2]
    target_acc = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
    return charbonnier(pred_acc - target_acc)


class VideoDeblurLoss(nn.Module):
    def __init__(
        self,
        pixel_weight: float = 1.0,
        fft_weight: float = 0.05,
        edge_weight: float = 0.1,
        temporal_weight: float = 0.15,
        acceleration_weight: float = 0.03,
        temporal_start_iter: int = 5_000,
        temporal_ramp_iters: int = 20_000,
    ):
        super().__init__()
        self.pixel_weight = pixel_weight
        self.fft_weight = fft_weight
        self.edge_weight = edge_weight
        self.temporal_weight = temporal_weight
        self.acceleration_weight = acceleration_weight
        self.temporal_start_iter = temporal_start_iter
        self.temporal_ramp_iters = temporal_ramp_iters

    def temporal_scale(self, iteration: int) -> float:
        if iteration < self.temporal_start_iter:
            return 0.0
        if self.temporal_ramp_iters <= 0:
            return 1.0
        return min((iteration - self.temporal_start_iter + 1) / self.temporal_ramp_iters, 1.0)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        iteration: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if prediction.shape != target.shape or prediction.ndim != 5:
            raise ValueError(f"Expected matching B,T,C,H,W tensors, got {prediction.shape}, {target.shape}")
        components = {
            "pixel": charbonnier(prediction - target),
            "fft": fft_loss(prediction, target),
            "edge": edge_loss(prediction, target),
            "temporal": temporal_difference_loss(prediction, target),
            "acceleration": temporal_acceleration_loss(prediction, target),
        }
        scale = self.temporal_scale(iteration)
        total = (
            self.pixel_weight * components["pixel"]
            + self.fft_weight * components["fft"]
            + self.edge_weight * components["edge"]
            + scale * self.temporal_weight * components["temporal"]
            + scale * self.acceleration_weight * components["acceleration"]
        )
        components["temporal_scale"] = prediction.new_tensor(scale)
        components["total"] = total
        return total, components


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(prediction.float(), target.float())
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def ssim(prediction: torch.Tensor, target: torch.Tensor, window: int = 11) -> torch.Tensor:
    """RGB SSIM with a uniform local window, averaged over all pixels/channels."""
    prediction = prediction.float()
    target = target.float()
    padding = window // 2
    mu_x = F.avg_pool2d(prediction, window, 1, padding)
    mu_y = F.avg_pool2d(target, window, 1, padding)
    sigma_x = F.avg_pool2d(prediction.square(), window, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target.square(), window, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, window, 1, padding) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(1e-12)
    return score.mean()


def temporal_residual_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_delta = prediction[:, 1:].float() - prediction[:, :-1].float()
    target_delta = target[:, 1:].float() - target[:, :-1].float()
    return (pred_delta - target_delta).abs().mean()
