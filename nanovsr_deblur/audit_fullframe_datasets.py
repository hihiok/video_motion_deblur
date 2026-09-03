import argparse
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from data.mixed_deblur import build_mixed_dataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gopro-root', required=True)
    ap.add_argument('--dvd-root', required=True)
    ap.add_argument('--bsd-root', required=True)
    return ap.parse_args()


def component_resolution(component):
    if not component.samples:
        return None
    _, _, pairs = component.samples[0]
    blur_path, gt_path = pairs[0]
    with Image.open(blur_path) as b, Image.open(gt_path) as g:
        bw, bh = b.size
        gw, gh = g.size
    if (bw, bh) != (gw, gh):
        raise RuntimeError(
            f'Native blur/GT resolution mismatch: {blur_path}={bw}x{bh}, '
            f'{gt_path}={gw}x{gh}'
        )
    return int(bw), int(bh)


def main():
    args = parse_args()
    roots = {
        'GoPro': args.gopro_root,
        'DVD': args.dvd_root,
        'BSD': args.bsd_root,
    }
    for family, root in roots.items():
        print(f'{family}_ROOT={Path(root).resolve()}')

    for t in (7, 30):
        print(f'\n=== FULL_FRAME_AUDIT T={t} ===')
        dataset, _, audit = build_mixed_dataset(
            roots,
            split='train',
            num_frames=t,
            patch_size=None,
            train=True,
        )
        windows = defaultdict(int)
        for row in audit:
            windows[row['family']] += row['windows']
        print('FAMILY_WINDOWS=' + ', '.join(f'{k}:{v}' for k, v in sorted(windows.items())))
        print(f'TOTAL_WINDOWS={len(dataset)}')

        resolutions = defaultdict(Counter)
        for component in dataset.datasets:
            wh = component_resolution(component)
            if wh is not None:
                resolutions[component.family][wh] += 1
        for family in sorted(resolutions):
            text = ', '.join(
                f'{w}x{h}:components={count}'
                for (w, h), count in sorted(resolutions[family].items())
            )
            print(f'{family}_NATIVE_RESOLUTIONS={text}')

    print('\nFULL_FRAME_AUDIT_STATUS=PASS')
    print('RANDOM_CROP=NO')
    print('RESIZE=NO')


if __name__ == '__main__':
    main()
