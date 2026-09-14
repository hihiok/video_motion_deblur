"""Actual upstream Loss2('1*L1') objective (cri_pix PSNRLoss is unused)."""
from torch.nn import functional as F


def loss_terms(pred, gt):
    pred,gt=pred.float(),gt.float()
    loss=F.l1_loss(pred,gt,reduction='mean')
    return loss, {'loss':loss.detach(), 'l1':loss.detach(),
                  'mse':(pred.detach()-gt).square().mean()}
