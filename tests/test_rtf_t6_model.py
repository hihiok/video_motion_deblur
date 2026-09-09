from pathlib import Path

import torch

from rtf_t6.checkpoint import load_rtfocuser_pretrained
from rtf_t6.complexity import compare_models, count_parameters
from rtf_t6.model import (
    GroupedSpatialTemporalShift,
    RT_Focuser_Standard,
    RTFocuserT6,
)


SMALL_DIMS = (8, 16, 24, 32, 48)
SMALL_KERNELS = (3, 3, 3, 3, 3)


def test_video_shape_and_t1_matches_frame_backbone():
    model = RTFocuserT6(
        dims=SMALL_DIMS,
        depths=(1, 1, 1, 1, 1),
        kernels=SMALL_KERNELS,
    ).eval()
    image = torch.rand(1, 1, 3, 32, 48)
    with torch.inference_mode():
        video_output = model(image)
        frame_output = model.backbone(image[:, 0]).unsqueeze(1)
    assert video_output.shape == image.shape
    torch.testing.assert_close(video_output, frame_output, rtol=1e-5, atol=1e-6)


def test_video_model_pads_and_crops_odd_resolution():
    model = RTFocuserT6(
        dims=SMALL_DIMS,
        depths=(1, 1, 1, 1, 1),
        kernels=SMALL_KERNELS,
    ).eval()
    image = torch.rand(1, 2, 3, 31, 47)
    with torch.inference_mode():
        output = model(image)
    assert output.shape == image.shape


def test_temporal_shift_replicates_boundaries_without_wraparound():
    shift = GroupedSpatialTemporalShift(fold_div=8)
    x = torch.zeros(1, 3, 8, 4, 4)
    x[:, 0] = 1
    x[:, 1] = 2
    x[:, 2] = 9
    y = shift(x)
    assert torch.all(y[:, 0, 0] == 1)
    assert torch.all(y[:, 2, 1] == 9)
    assert torch.all(y[:, 1, 0] == 1)
    assert torch.all(y[:, 1, 1] == 9)


def test_pretrained_loader_maps_frame_keys_to_video_backbone(tmp_path: Path):
    frame = RT_Focuser_Standard(
        dims=SMALL_DIMS,
        depths=(2, 2, 2, 2, 1),
        kernels=SMALL_KERNELS,
    )
    checkpoint = tmp_path / "frame.pth"
    torch.save(frame.state_dict(), checkpoint)
    video = RTFocuserT6(
        dims=SMALL_DIMS,
        depths=(1, 1, 1, 1, 1),
        kernels=SMALL_KERNELS,
    )
    report = load_rtfocuser_pretrained(video, checkpoint, min_target_coverage=0.99)
    assert report["target_coverage"] == 1.0
    assert report["dropped_source_keys"]


def test_default_candidate_is_below_frame_baseline_budget():
    baseline = RT_Focuser_Standard()
    candidate = RTFocuserT6()
    assert count_parameters(candidate) < count_parameters(baseline)
    report = compare_models(baseline, candidate, height=32, width=32, clip_length=6)
    assert report["parameters_pass"]
    assert report["compute_pass"]


def test_activation_checkpointing_preserves_output_gradients_and_bn_buffers():
    reference = RTFocuserT6(
        dims=SMALL_DIMS,
        depths=(1, 1, 1, 1, 1),
        kernels=SMALL_KERNELS,
        activation_checkpointing=False,
    ).train()
    checkpointed = RTFocuserT6(
        dims=SMALL_DIMS,
        depths=(1, 1, 1, 1, 1),
        kernels=SMALL_KERNELS,
        activation_checkpointing=True,
    ).train()
    checkpointed.load_state_dict(reference.state_dict())
    video = torch.rand(1, 2, 3, 32, 32)

    reference_output = reference(video)
    reference_output.mean().backward()
    checkpointed_output = checkpointed(video)
    checkpointed_output.mean().backward()

    torch.testing.assert_close(checkpointed_output, reference_output, rtol=1e-5, atol=1e-6)
    for (_, reference_parameter), (_, checkpointed_parameter) in zip(
        reference.named_parameters(), checkpointed.named_parameters()
    ):
        if reference_parameter.grad is None or checkpointed_parameter.grad is None:
            assert reference_parameter.grad is checkpointed_parameter.grad
        else:
            torch.testing.assert_close(
                checkpointed_parameter.grad,
                reference_parameter.grad,
                rtol=2e-4,
                atol=2e-6,
            )
    reference_buffers = dict(reference.named_buffers())
    checkpointed_buffers = dict(checkpointed.named_buffers())
    for name in reference_buffers:
        if "running_" in name or "num_batches_tracked" in name:
            torch.testing.assert_close(checkpointed_buffers[name], reference_buffers[name])
