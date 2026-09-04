"""Compare T=7/15/30 on exactly matched GoPro center-frame targets."""

import argparse

import torch

from data.gopro_video import GoProVideoDataset
from eval_gopro_nanovnr_waveshift_pagf import load_model, psnr_per_frame


def target_index(dataset, context):
    center_offset = context // 2
    result = {}
    for dataset_index, (sequence, start, _) in enumerate(dataset.samples):
        result[(sequence, start + center_offset)] = dataset_index
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gopro-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--contexts', type=int, nargs='+', default=(7, 15, 30))
    parser.add_argument('--max-targets', type=int, default=100)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--deploy-reparam', action='store_true')
    args = parser.parse_args()
    contexts = tuple(args.contexts)
    if len(set(contexts)) != len(contexts) or any(value < 1 for value in contexts):
        raise ValueError('--contexts must contain unique positive integers.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, checkpoint_data = load_model(
        args.checkpoint, device, deploy_reparam=args.deploy_reparam
    )
    datasets = {
        context: GoProVideoDataset(
            args.gopro_root,
            split='test',
            num_frames=context,
            patch_size=None,
        )
        for context in contexts
    }
    indices = {
        context: target_index(dataset, context)
        for context, dataset in datasets.items()
    }
    shared_targets = set.intersection(
        *(set(index_map) for index_map in indices.values())
    )
    shared_targets = sorted(shared_targets)
    if args.max_targets:
        shared_targets = shared_targets[:args.max_targets]
    if not shared_targets:
        raise RuntimeError('No matched center-frame targets across requested contexts.')

    results = {}
    with torch.no_grad():
        for context in contexts:
            output_values = []
            input_values = []
            center = context // 2
            dataset = datasets[context]
            for target in shared_targets:
                sample = dataset[indices[context][target]]
                blur = sample['blur'].unsqueeze(0).to(device)
                sharp = sample['sharp'].unsqueeze(0).to(device)
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    prediction, _ = model(blur)
                output_values.append(
                    psnr_per_frame(
                        prediction[:, center].float().clamp(0, 1),
                        sharp[:, center].float(),
                    )[0].item()
                )
                input_values.append(
                    psnr_per_frame(
                        blur[:, center].float(), sharp[:, center].float()
                    )[0].item()
                )
            results[context] = (
                sum(output_values) / len(output_values),
                sum(input_values) / len(input_values),
            )

    print(f'ARCHITECTURE={checkpoint_data.get("architecture")}')
    print(f'VARIANT={checkpoint_data.get("variant")}')
    print(f'CHECKPOINT_STEP={checkpoint_data.get("step")}')
    print(f'MATCHED_TARGETS={len(shared_targets)}')
    print('TARGET_KEY=sequence,absolute_center_frame_index')
    for context in contexts:
        output_psnr, input_psnr = results[context]
        print(f'CENTER_T{context}_INPUT_PSNR={input_psnr:.4f} dB')
        print(f'CENTER_T{context}_OUTPUT_PSNR={output_psnr:.4f} dB')
    for previous, current in zip(contexts, contexts[1:]):
        gain = results[current][0] - results[previous][0]
        print(f'CENTER_CONTEXT_GAIN_T{current}_VS_T{previous}={gain:+.4f} dB')


if __name__ == '__main__':
    main()
