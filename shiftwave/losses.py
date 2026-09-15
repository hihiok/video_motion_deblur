"""GT controls reconstruction; frozen teacher is a small decaying auxiliary."""
import torch.nn.functional as F
from .model import haar


def loss_terms(pred, gt, teacher=None, kd_weight=0., hf_weight=.05):
    pred,gt=pred.float(),gt.float()
    l1=F.l1_loss(pred,gt)
    p=pred.flatten(0,1);g=gt.flatten(0,1)
    hp=haar(p)[1];hg=haar(g)[1]
    hf=F.l1_loss(hp,hg)
    kd=pred.new_zeros(()) if teacher is None else F.l1_loss(pred,teacher.detach().float())
    loss=l1+hf_weight*hf+kd_weight*kd
    return loss, {'loss':loss.detach(),'l1':l1.detach(),'hf':hf.detach(),'kd':kd.detach(),
                  'mse':(pred.detach()-gt).square().mean()}
