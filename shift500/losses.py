"""Supervised restoration, quality-masked distillation, GT-relative temporal loss."""
import torch
from torch.nn import functional as F


def loss_terms(pred, gt, teacher, step, total):
    pred,gt,teacher=pred.float(),gt.float(),teacher.float()
    error=(pred-gt).square().mean(dim=(2,3,4))
    # Scale is compatible with upstream 0.5 PSNRLoss (log MSE objective).
    reconstruction=(5./torch.log(torch.tensor(10.,device=pred.device)))*torch.log(error+1e-8).mean()
    # Teacher contributes only where closer to GT than the current prediction.
    teacher_error=(teacher-gt).abs().mean(2,keepdim=True)
    student_error=(pred.detach()-gt).abs().mean(2,keepdim=True)
    weight=(teacher_error <= student_error).float()
    kd=((pred-teacher).abs()*weight).sum()/(3*weight.sum()).clamp_min(1)
    # Match GT frame differences; do NOT penalize actual object movement.
    # No flow; name explicitly reflects unwarped GT-relative residual differences.
    temporal=((pred[:,1:]-pred[:,:-1])-(gt[:,1:]-gt[:,:-1])).abs().mean()
    # GT-scene cuts excluded from temporal regularization.
    cut=(gt[:,1:]-gt[:,:-1]).abs().mean((2,3,4))>.30
    residual=((pred[:,1:]-pred[:,:-1])-(gt[:,1:]-gt[:,:-1])).abs().mean((2,3,4))
    temporal=(residual*(~cut)).sum()/(~cut).sum().clamp_min(1)
    ramp=min(1.,max(0.,(step-20000)/20000))
    kd_weight=2.*(1.-.75*step/max(1,total))
    # Small relative to log-MSE gradients; avoid optimizing appearance by blurring.
    loss=reconstruction+kd_weight*kd+.25*ramp*temporal
    return loss, {'loss':loss.detach(), 'log_mse':reconstruction.detach(), 'kd_l1':kd.detach(),
                  'gt_relative_temporal_l1':temporal.detach(), 'mse':error.mean().detach()}
