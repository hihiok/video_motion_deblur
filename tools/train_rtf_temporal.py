#!/usr/bin/env python3
"""Two-stage native-frame fine-tuning, exact sample resume, one/two GPU DDP."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import signal
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtf_t6.checkpoint import atomic_torch_save
from rtf_temporal.data import DOMAINS, TrainingClips, RankSampleIndices, load_record, sha256
from rtf_temporal.flow import aligned_error, FLOW_VERSION
from rtf_temporal.model import TemporalRTFocuser
from rtf_temporal.evaluation import evaluate, qualifies


def git_commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip()


def loss_fn(pred, batch, update, cfg):
    mse = (pred.float() - batch['gt'].float()).square().mean()
    temporal, _, _ = aligned_error(pred, batch['gt'], batch['flow'], batch['mask'])
    weight = cfg['temporal_weight'] * min(1., update / cfg['temporal_ramp_updates'])
    total = mse + weight * temporal
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('Non-finite loss; no optimizer step performed')
    return total, torch.stack((mse.detach(), temporal.detach(), total.detach(), batch['mask'].mean()))


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def optimizer_for(model, cfg):
    return torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': cfg['backbone_lr'], 'base_lr': cfg['backbone_lr']},
        {'params': model.temporal.parameters(), 'lr': cfg['adapter_lr'], 'base_lr': cfg['adapter_lr']}],
        weight_decay=cfg['weight_decay'])


def wrap(model, world, device):
    return DDP(model, device_ids=[device.index] if device.type == 'cuda' else None,
               broadcast_buffers=False) if world > 1 else model


def sync(world):
    if world > 1:
        dist.barrier()


def fingerprint(config_path, config, world, device):
    return dict(config_sha256=sha256(config_path), manifest_sha256=config['manifest_sha256'],
                pretrained_sha256=config['pretrained_sha256'], git_commit=git_commit(),
                world_size=world, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                torch_version=torch.__version__, device_type=device.type, flow_version=FLOW_VERSION,
                device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU')


def preflight(args, config, manifest, rank, world, device):
    tcfg, records = config['train'], []
    accumulation = tcfg['clips_per_update'] // world
    for adapter_only in (True, False):
        model = TemporalRTFocuser(**config['model']).to(device)
        model.load_official(config['pretrained'])
        model.set_stage(adapter_only)
        optimizer = optimizer_for(model, tcfg)
        wrapped = wrap(model, world, device)
        for d in DOMAINS:
            record = max((r for r in manifest['train'] + manifest['val'] if r['domain'] == d),
                         key=lambda r: (r['height'] * r['width'], r['name']))
            sample = load_record(record, 0, tcfg['clip_length'], cut_threshold=tcfg['cut_threshold'])
            batch = to_device({k: v[None] if torch.is_tensor(v) else v for k, v in sample.items()}, device)
            # Exercise recurrent graph even if the selected clip has a cut.
            batch['resets'][:, 1:] = False
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            start = time.perf_counter()
            for micro in range(accumulation):
                ctx = wrapped.no_sync() if world > 1 and micro < accumulation - 1 else nullcontext()
                with ctx:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                        pred = wrapped(batch['blur'], batch['resets'])
                    loss, _ = loss_fn(pred, batch, tcfg['temporal_ramp_updates'], tcfg)
                    (loss / accumulation).backward()
                del pred, loss
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg['clip_grad'], error_if_nonfinite=True)
            optimizer.step()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            item = dict(domain=d, adapter_only=adapter_only, shape=list(batch['blur'].shape),
                        sequence=record['name'], rank=rank, seconds=time.perf_counter() - start,
                        grad_norm=float(norm), peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None,
                        peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30 if device.type == 'cuda' else None)
            records.append(item)
            print(json.dumps({'event': 'preflight_case', **item}), flush=True)
            del batch, sample
        del wrapped, model, optimizer
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, records)
        records = [item for group in gathered for item in group]
    if rank == 0:
        report = {'status': 'PREFLIGHT_PASS' if device.type == 'cuda' else 'CPU_TEST_ONLY',
                  'identity': fingerprint(args.config, config, world, device), 'cases': records,
                  'weights_discarded': True}
        (Path(config['output']) / f'preflight_{world}gpu.json').write_text(json.dumps(report, indent=2) + '\n')
        print(report['status'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=('preflight', 'train'), default='train')
    parser.add_argument('--resume')
    parser.add_argument('--stop-after', type=int, help='Absolute update boundary; preserve full schedule')
    parser.add_argument('--cpu-test', action='store_true', help='Integration tests only, cannot issue GPU gate')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    tcfg = config['train']
    if config['protocol'] != 'rtfocuser_causal_temporal_finetune_v1' or tcfg['crop_size'] != 0 or tcfg['resize']:
        raise ValueError('Protocol/full-frame configuration mismatch')
    if tcfg['clip_length'] != 4 or tcfg['clips_per_update'] != 2 or tcfg['precision'] != 'bf16':
        raise ValueError('v1 protocol requires T4, two global clips/update and BF16')
    for pathkey in ('manifest', 'pretrained'):
        if sha256(config[pathkey]) != config[pathkey + '_sha256']:
            raise ValueError(f'{pathkey} SHA256 mismatch')
    manifest = json.loads(Path(config['manifest']).read_text())
    world, rank, local = (int(os.environ.get(k, default)) for k, default in
                           (('WORLD_SIZE', 1), ('RANK', 0), ('LOCAL_RANK', 0)))
    if world not in (1, 2):
        raise ValueError('Only 1/2 GPUs supported by this protocol')
    if args.cpu_test:
        device = torch.device('cpu')
    else:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA BF16 required; activate deblur_runtime on A100')
        torch.cuda.set_device(local)
        device = torch.device('cuda', local)
    torch.set_num_threads(tcfg['cpu_threads_per_rank'])
    torch.manual_seed(config['seed'])
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(hours=2))
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'preflight':
        preflight(args, config, manifest, rank, world, device)
        if world > 1:
            dist.destroy_process_group()
        return
    if not args.cpu_test:
        gate = json.loads((output / f'preflight_{world}gpu.json').read_text())
        if gate['status'] != 'PREFLIGHT_PASS' or gate['identity'] != fingerprint(args.config, config, world, device):
            raise ValueError('Missing or stale full-frame preflight; rerun with current code/config/GPU count')

    start_update, best = 0, float('inf')
    if not args.resume and (output / 'checkpoints/latest.pth').exists():
        raise FileExistsError('Existing training: pass --resume, never restart over it')
    model = TemporalRTFocuser(**config['model']).to(device)
    model.load_official(config['pretrained'])
    optimizer = optimizer_for(model, tcfg)
    baseline = None
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved['config_sha256'] != sha256(args.config) or saved['git_commit'] != git_commit():
            raise ValueError('Resume requires same code/config/manifest, no old T3/T6 checkpoints')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        start_update, best, baseline = saved['update'], saved['best_temporal'], saved['baseline']
        del saved
    stop_after = args.stop_after or tcfg['total_updates']
    if not start_update < stop_after <= tcfg['total_updates']:
        raise ValueError('stop-after must be greater than saved update and <= total_updates')
    # Compute baseline BEFORE any weight updates. Other ranks wait in a long-timeout barrier.
    if baseline is None:
        if rank == 0:
            baseline = evaluate(model, manifest, config, device, spatial_only=True,
                                preview_dir=output / 'previews/baseline')
            (output / 'baseline.json').write_text(json.dumps(baseline, indent=2) + '\n')
            print(json.dumps({'event': 'baseline', **baseline}), flush=True)
        if world > 1:
            obj = [baseline]
            dist.broadcast_object_list(obj, 0)
            baseline = obj[0]
    adapter_only = start_update < tcfg['warmup_updates']
    model.set_stage(adapter_only)
    wrapped = wrap(model, world, device)
    accumulation = tcfg['clips_per_update'] // world
    dataset = TrainingClips(manifest, tcfg['total_updates'] * 2, config['seed'],
                            tcfg['clip_length'], tcfg['cut_threshold'])
    loader = DataLoader(dataset, batch_size=1,
        sampler=RankSampleIndices(start_update, stop_after, rank, world),
        num_workers=tcfg['workers_per_rank'], pin_memory=device.type == 'cuda',
        persistent_workers=False, **({'prefetch_factor': 1} if tcfg['workers_per_rank'] else {}))
    iterator = iter(loader)
    stop_requested = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop_requested.__setitem__(0, True))

    def save(name, update):
        atomic_torch_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), update=update,
            best_temporal=best, baseline=baseline, config=config, config_sha256=sha256(args.config),
            git_commit=git_commit(), global_clips_consumed=update * 2), output / 'checkpoints' / name)

    metrics_path = output / 'train_metrics.jsonl'
    started, running, count = time.perf_counter(), torch.zeros(4, device=device), 0
    if rank == 0:
        print(json.dumps({'event': 'training_start', 'start_update': start_update, 'world_size': world,
                          'frames_per_update': 8, 'full_frame': True, 'config': config}), flush=True)
    for update in range(start_update + 1, stop_after + 1):
        new_adapter_only = update <= tcfg['warmup_updates']
        if new_adapter_only != adapter_only:
            sync(world)
            del wrapped
            model.set_stage(new_adapter_only)
            wrapped = wrap(model, world, device)
            adapter_only = new_adapter_only
            if rank == 0:
                print(json.dumps({'event': 'joint_finetune', 'update': update}), flush=True)
        model.train()
        for group in optimizer.param_groups:
            stage_progress = ((update - 1) / max(tcfg['warmup_updates'], 1) if adapter_only else
                              (update - tcfg['warmup_updates'] - 1) / max(tcfg['total_updates'] - tcfg['warmup_updates'], 1))
            ratio = tcfg['min_lr_ratio'] + (1 - tcfg['min_lr_ratio']) * .5 * (1 + math.cos(math.pi * stage_progress))
            group['lr'] = group['base_lr'] * ratio
        optimizer.zero_grad(set_to_none=True)
        for micro in range(accumulation):
            batch = to_device(next(iterator), device)
            ctx = wrapped.no_sync() if world > 1 and micro < accumulation - 1 else nullcontext()
            with ctx:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    pred = wrapped(batch['blur'], batch['resets'])
                loss, values = loss_fn(pred, batch, update, tcfg)
                (loss / accumulation).backward()
            running += values
            count += 1
            del batch, pred, loss, values
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg['clip_grad'], error_if_nonfinite=True)
        optimizer.step()
        stop = torch.tensor(int(stop_requested[0]), device=device)
        if world > 1:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        stopping = bool(stop)
        if update % tcfg['log_every'] == 0 or update == stop_after or stopping:
            if world > 1:
                dist.all_reduce(running)
            record = dict(event='train', update=update, stage='adapter' if adapter_only else 'joint',
                mse=float(running[0] / (count * world)), temporal_l1=float(running[1] / (count * world)),
                loss=float(running[2] / (count * world)), valid_flow_fraction=float(running[3] / (count * world)),
                grad_norm=float(norm), seconds_per_update=(time.perf_counter() - started) / (count / accumulation),
                backbone_lr=optimizer.param_groups[0]['lr'], adapter_lr=optimizer.param_groups[1]['lr'])
            if rank == 0:
                print(json.dumps(record), flush=True)
                with metrics_path.open('a') as handle:
                    handle.write(json.dumps(record) + '\n')
            running.zero_(); count = 0; started = time.perf_counter()
        if not stopping and (update % tcfg['validate_every'] == 0 or update == stop_after):
            sync(world)
            if rank == 0:
                metrics = evaluate(model, manifest, config, device, preview_dir=output / f'previews/update_{update:06d}')
                accepted = qualifies(metrics, baseline, config)
                record = dict(event='validation', update=update, qualifies=accepted, **metrics)
                (output / f'validation_{update:06d}.json').write_text(json.dumps(record, indent=2) + '\n')
                with metrics_path.open('a') as handle:
                    handle.write(json.dumps(record) + '\n')
                if accepted and metrics['balanced_aligned_temporal_l1'] < best and metrics['balanced_aligned_temporal_l1'] < baseline['balanced_aligned_temporal_l1']:
                    best = metrics['balanced_aligned_temporal_l1']
                    save('best_stable.pth', update)
                print(json.dumps(record), flush=True)
            sync(world)
            started = time.perf_counter()
        if rank == 0 and (update % tcfg['save_every'] == 0 or update == stop_after or stopping):
            save('latest.pth', update)
        sync(world)
        if stopping:
            break
    if rank == 0:
        print(json.dumps({'status': 'TRAINING_COMPLETE' if update == tcfg['total_updates'] else 'TRAINING_PAUSED',
                          'update': update, 'best_stable_available': (output / 'checkpoints/best_stable.pth').exists()}), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
