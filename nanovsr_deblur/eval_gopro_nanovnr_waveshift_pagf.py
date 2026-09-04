import argparse

import torch
from torch.utils.data import DataLoader

from data.gopro_video import GoProVideoDataset
from models.network_nanovnr_waveshift_pagf import NanoVNRWaveShiftPAGF


ARCHITECTURE = 'NanoVNRWaveShiftPAGF'


def psnr_per_frame(a, b):
    mse = (a - b).pow(2).mean(dim=(-3, -2, -1)).clamp_min(1e-12)
    return -10.0 * torch.log10(mse)


def load_model(checkpoint, device, deploy_reparam=False):
    data = torch.load(checkpoint, map_location='cpu')
    if data.get('architecture') != ARCHITECTURE:
        raise RuntimeError(f'Unexpected architecture: {data.get("architecture")}')
    config = data.get('model_config')
    if not isinstance(config, dict):
        raise RuntimeError('Checkpoint is missing model_config.')
    model = NanoVNRWaveShiftPAGF.from_config(config).to(device).eval()
    model.load_state_dict(data['model'], strict=True)
    if deploy_reparam:
        model.switch_to_deploy()
    return model, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gopro-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num-frames', type=int, default=15)
    parser.add_argument('--max-clips', type=int, default=100)
    parser.add_argument('--center-only', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--deploy-reparam', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, checkpoint_data = load_model(
        args.checkpoint, device, deploy_reparam=args.deploy_reparam
    )
    dataset = GoProVideoDataset(
        args.gopro_root,
        split='test',
        num_frames=args.num_frames,
        patch_size=None,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    output_values = []
    input_values = []

    with torch.no_grad():
        for index, batch in enumerate(loader):
            if args.max_clips and index >= args.max_clips:
                break
            blur = batch['blur'].to(device, non_blocking=True)
            sharp = batch['sharp'].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.fp16):
                prediction, _ = model(blur)
            output_psnr = psnr_per_frame(prediction.float().clamp(0, 1), sharp.float())[0]
            input_psnr = psnr_per_frame(blur.float().clamp(0, 1), sharp.float())[0]
            if args.center_only:
                center = args.num_frames // 2
                output_values.append(output_psnr[center].item())
                input_values.append(input_psnr[center].item())
            else:
                output_values.extend(output_psnr.cpu().tolist())
                input_values.extend(input_psnr.cpu().tolist())
            if (index + 1) % 20 == 0:
                print(
                    f'clips={index + 1} OUTPUT_PSNR_RGB='
                    f'{sum(output_values) / len(output_values):.4f}',
                    flush=True,
                )

    mode = 'CENTER_ONLY' if args.center_only else 'ALL_FRAMES'
    clips = min(len(dataset), args.max_clips) if args.max_clips else len(dataset)
    output_mean = sum(output_values) / max(1, len(output_values))
    input_mean = sum(input_values) / max(1, len(input_values))
    print(f'ARCHITECTURE={checkpoint_data.get("architecture")}')
    print(f'VARIANT={checkpoint_data.get("variant")}')
    print(f'CHECKPOINT_STEP={checkpoint_data.get("step")}')
    print(f'EVAL_MODE={mode}')
    print(f'NUM_FRAMES={args.num_frames}')
    print(f'CLIPS={clips}')
    print(f'DEPLOY_REPARAM={args.deploy_reparam}')
    print(f'INPUT_PSNR_RGB={input_mean:.4f} dB')
    print(f'OUTPUT_PSNR_RGB={output_mean:.4f} dB')
    print(f'GAIN_VS_BLUR_INPUT={output_mean - input_mean:+.4f} dB')


if __name__ == '__main__':
    main()
