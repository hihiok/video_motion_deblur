import torch

from rtf_t6.losses import VideoDeblurLoss, temporal_difference_loss


def test_temporal_difference_loss_does_not_penalize_correct_motion():
    target = torch.rand(1, 6, 3, 8, 8)
    assert temporal_difference_loss(target, target).item() < 0.0011


def test_temporal_loss_ramps_in_after_start():
    criterion = VideoDeblurLoss(temporal_start_iter=10, temporal_ramp_iters=10)
    assert criterion.temporal_scale(9) == 0.0
    assert 0.0 < criterion.temporal_scale(10) < 1.0
    assert criterion.temporal_scale(19) == 1.0
