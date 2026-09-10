import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from rtf_t6.model import RT_Focuser_Standard
from rtf_temporal.model import TemporalRTFocuser
from rtf_temporal.flow import aligned_error, warp, pair_flow, is_cut
from rtf_temporal.data import RankSampleIndices, make_manifest, sha256, sequence_group


torch.set_num_threads(1)


def test_initial_identity_official_weights_and_bn(tmp_path):
    torch.manual_seed(5)
    base = RT_Focuser_Standard().eval()
    path = tmp_path / 'official.pth'
    torch.save(base.state_dict(), path)
    model = TemporalRTFocuser().eval()
    model.load_official(path)
    video = torch.rand(1, 3, 3, 32, 48)
    with torch.no_grad():
        actual = model(video)
        expected = torch.stack([base(x) for x in video.unbind(1)], 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    model.train()
    assert all(not m.training for m in model.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d))


def test_against_upstream_reference():
    ref = os.environ.get('RTF_OFFICIAL_REFERENCE')
    if not ref:
        pytest.skip('Set RTF_OFFICIAL_REFERENCE to downloaded official model source')
    spec = importlib.util.spec_from_file_location('upstream', ref)
    upstream = importlib.util.module_from_spec(spec); spec.loader.exec_module(upstream)
    official = upstream.RT_Focuser_Standard().eval()
    model = TemporalRTFocuser().eval()
    model.backbone.load_state_dict(official.state_dict(), strict=True)
    x = torch.rand(1, 3, 32, 48)
    with torch.no_grad():
        torch.testing.assert_close(model.step(x)[0], official(x), rtol=0, atol=0)


def test_recurrence_reset_causality_and_odd_resolution():
    model = TemporalRTFocuser().eval()
    torch.nn.init.normal_(model.temporal.project.weight, std=.3)
    video = torch.rand(1, 4, 3, 33, 49)
    with torch.no_grad():
        full = model(video)
        state, parts = None, []
        for t in range(4):
            pred, state = model.step(video[:, t], state)
            parts.append(pred)
        torch.testing.assert_close(full, torch.stack(parts, 1), rtol=0, atol=0)
        altered = video.clone(); altered[:, 2:] = torch.rand_like(altered[:, 2:])
        torch.testing.assert_close(full[:, :2], model(altered)[:, :2], rtol=0, atol=0)
        cold, _ = model.step(video[:, -1])
        reset, _ = model.step(video[:, -1], state, torch.tensor([True]))
        torch.testing.assert_close(cold, reset, rtol=0, atol=0)
    assert full.shape == video.shape


def test_history_gradient_and_freeze_stage():
    model = TemporalRTFocuser(activation_checkpointing=True)
    model.set_stage(True)
    original = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    torch.nn.init.normal_(model.temporal.project.weight, std=.1)
    x = torch.rand(1, 3, 3, 32, 32, requires_grad=True)
    model(x)[:, -1].square().mean().backward()
    assert x.grad[:, 0].abs().sum() > 0
    assert model.temporal.project.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.parameters())
    for k, v in model.backbone.state_dict().items():
        torch.testing.assert_close(v, original[k], rtol=0, atol=0)
    model.set_stage(False)
    model.zero_grad(set_to_none=True)
    model(torch.rand(1, 2, 3, 32, 32)).square().mean().backward()
    assert model.backbone.Conv_1x1.weight.grad.abs().sum() > 0


def test_flow_direction_and_gt_relative_loss():
    old = torch.arange(20.).reshape(1, 1, 4, 5)
    flow = torch.zeros(1, 2, 4, 5); flow[:, 0] = 1
    shifted = warp(old, flow)
    torch.testing.assert_close(shifted[..., :-1], old[..., 1:])
    gt = torch.rand(1, 3, 3, 16, 16)
    flows = torch.zeros(1, 2, 2, 16, 16)
    masks = torch.ones(1, 2, 1, 16, 16)
    assert aligned_error(gt, gt, flows, masks)[0] == 0
    # A fixed reconstruction bias should not be treated as temporal flicker.
    assert aligned_error(gt + .1, gt, flows, masks)[0] < 1e-6
    pred = gt.clone(); pred[:, 1] += .1
    assert aligned_error(pred, gt, flows, masks)[0] > .09
    assert aligned_error(pred, gt, flows, masks * 0)[0] == 0
    image = np.random.default_rng(1).random((32, 32, 3), dtype=np.float32)
    f, m = pair_flow(image, image)
    assert f.shape == (2, 32, 32) and m.mean() > .5
    assert is_cut(np.zeros_like(image), np.ones_like(image))


def test_one_two_gpu_samples_and_resume():
    one = list(RankSampleIndices(0, 10, 0, 1))
    two = sorted(list(RankSampleIndices(0, 10, 0, 2)) + list(RankSampleIndices(0, 10, 1, 2)))
    assert one == two == list(range(20))
    assert list(RankSampleIndices(4, 10, 0, 1)) == one[8:]
    assert sequence_group('gopro', 'GOPR0372_07_01') == sequence_group('gopro', 'GOPR0372_07_00')


def miniature_data(tmp_path):
    roots = {}
    rng = np.random.default_rng(27)
    for domain in ('gopro', 'bsd', 'dvd'):
        roots[domain] = str(tmp_path / domain)
        for seq in range(4):
            image = rng.integers(20, 230, (32, 32, 3), dtype=np.uint8)
            for frame in range(4):
                for role in ('blur', 'sharp'):
                    directory = tmp_path / domain / 'train' / f'seq{seq}' / role
                    directory.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(image if role == 'sharp' else (image // 2 + 30)).save(directory / f'{frame:04d}.png')
    return roots


def test_data_audit_and_integration_resume(tmp_path):
    roots = miniature_data(tmp_path)
    manifest = make_manifest(roots)
    assert len(manifest['train']) == 9 and len(manifest['val']) == 3
    run = tmp_path / 'run'; run.mkdir()
    mp = run / 'manifest.json'; mp.write_text(json.dumps(manifest))
    pretrained = tmp_path / 'base.pth'; torch.save(RT_Focuser_Standard().state_dict(), pretrained)
    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo / 'configs/rtf_temporal_finetune.yaml').read_text())
    cfg.update(output=str(run), manifest=str(mp), manifest_sha256=sha256(mp),
               pretrained=str(pretrained), pretrained_sha256=sha256(pretrained))
    cfg['train'].update(warmup_updates=1, total_updates=3, workers_per_rank=0, log_every=1,
                        save_every=1, validate_every=3, cpu_threads_per_rank=1)
    cfg['validation'].update(frames_per_sequence=4, max_sequences_per_domain=1)
    cp = run / 'runtime_config.yaml'; cp.write_text(yaml.safe_dump(cfg))
    command = [sys.executable, str(repo / 'tools/train_rtf_temporal.py'), '--config', str(cp), '--cpu-test']
    def execute(args):
        result = subprocess.run(command + args, cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert result.returncode == 0, result.stdout
        return result.stdout
    execute(['--mode', 'preflight'])
    assert json.loads((run / 'preflight_1gpu.json').read_text())['status'] == 'CPU_TEST_ONLY'
    assert 'TRAINING_PAUSED' in execute(['--stop-after', '1'])
    latest = run / 'checkpoints/latest.pth'
    assert 'TRAINING_COMPLETE' in execute(['--resume', str(latest)])
    saved = torch.load(latest, map_location='cpu', weights_only=False)
    assert saved['update'] == 3 and saved['global_clips_consumed'] == 6
    assert (run / 'baseline.json').exists() and (run / 'validation_000003.json').exists()
    if not os.environ.get('RTF_TEST_DDP'):
        return  # DDP runs separately on the server; this environment forbids Gloo sockets.
    # Two-rank Gloo exercises the same reducer and stage rewrap logic as NCCL.
    ddp_run = tmp_path / 'ddp'; ddp_run.mkdir()
    cfg['output'] = str(ddp_run)
    ddp_cfg = ddp_run / 'runtime_config.yaml'; ddp_cfg.write_text(yaml.safe_dump(cfg))
    result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
        '--nproc_per_node=2', str(repo / 'tools/train_rtf_temporal.py'), '--config', str(ddp_cfg),
        '--cpu-test'], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=180)
    assert result.returncode == 0, result.stdout
    other = torch.load(ddp_run / 'checkpoints/latest.pth', map_location='cpu', weights_only=False)
    assert other['update'] == 3
    for key in saved['model']:
        torch.testing.assert_close(saved['model'][key], other['model'][key], rtol=1e-4, atol=2e-5)


def test_non_contiguous_frames_rejected(tmp_path):
    roots = miniature_data(tmp_path)
    for role in ('blur', 'sharp'):
        path = Path(roots['gopro']) / 'train/seq0' / role / '0003.png'
        path.rename(path.with_name('0004.png'))
    with pytest.raises(ValueError, match='Non-contiguous'):
        make_manifest(roots)
