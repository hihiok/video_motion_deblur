import argparse
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from data.mixed_deblur import (
    VideoPairWindowDataset,
    build_mixed_dataset,
    discover_pair_roots,
)


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


def assert_under(path, allowed_root):
    path = Path(path).resolve()
    allowed_root = Path(allowed_root).resolve()
    try:
        path.relative_to(allowed_root)
    except ValueError as exc:
        raise RuntimeError(
            f'Path escaped allowed BSD split root: path={path} allowed={allowed_root}'
        ) from exc


def audit_bsd_split(bsd_root, split, t_values=(7, 30)):
    bsd_root = Path(bsd_root).resolve()
    allowed = (bsd_root / split).resolve()
    if not allowed.is_dir():
        raise RuntimeError(f'Missing required BSD split directory: {allowed}')

    pairs = discover_pair_roots(
        bsd_root,
        split=split,
        root_split_only=True,
    )
    if not pairs:
        raise RuntimeError(
            f'No BSD blur/GT pairs found strictly under {allowed}. '
            'Nested BSD/<config>/<split> directories are intentionally forbidden.'
        )

    print(f'\n=== BSD_STRICT_SPLIT split={split} ===')
    print(f'BSD_ALLOWED_ROOT={allowed}')
    print('BSD_ROOT_SPLIT_ONLY=YES')
    for blur_root, gt_root in pairs:
        assert_under(blur_root, allowed)
        assert_under(gt_root, allowed)
        print(f'BSD_PAIR blur={Path(blur_root).resolve()} gt={Path(gt_root).resolve()}')

    for t in t_values:
        total_windows = 0
        resolutions = Counter()
        for blur_root, gt_root in pairs:
            ds = VideoPairWindowDataset(
                family='BSD',
                blur_root=blur_root,
                gt_root=gt_root,
                num_frames=t,
                patch_size=None,
                train=(split == 'train'),
            )
            total_windows += len(ds)
            wh = component_resolution(ds)
            if wh is not None:
                resolutions[wh] += 1
        res_text = ', '.join(
            f'{w}x{h}:components={count}'
            for (w, h), count in sorted(resolutions.items())
        )
        print(f'BSD_{split.upper()}_T{t}_WINDOWS={total_windows}')
        print(f'BSD_{split.upper()}_T{t}_RESOLUTIONS={res_text}')


def main():
    args = parse_args()
    roots = {
        'GoPro': args.gopro_root,
        'DVD': args.dvd_root,
        'BSD': args.bsd_root,
    }
    for family, root in roots.items():
        print(f'{family}_ROOT={Path(root).resolve()}')

    # Mixed training audit: BSD is automatically strict-root-split because the
    # family name is BSD inside build_mixed_dataset().
    for t in (7, 30):
        print(f'\n=== FULL_FRAME_TRAIN_AUDIT T={t} ===')
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
            if row['family'].lower() == 'bsd':
                if not row.get('strict_root_split', False):
                    raise RuntimeError('BSD audit row is not marked strict_root_split.')
                assert_under(row['blur_root'], Path(args.bsd_root) / 'train')
                assert_under(row['gt_root'], Path(args.bsd_root) / 'train')

        print(
            'FAMILY_WINDOWS=' + ', '.join(
                f'{k}:{v}' for k, v in sorted(windows.items())
            )
        )
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

    # Explicitly audit both allowed BSD splits. No other BSD directory is eligible.
    audit_bsd_split(args.bsd_root, 'train')
    audit_bsd_split(args.bsd_root, 'test')

    print('\nFULL_FRAME_AUDIT_STATUS=PASS')
    print('RANDOM_CROP=NO')
    print('RESIZE=NO')
    print('BSD_ALLOWED_SPLITS=train,test')
    print('BSD_NESTED_CONFIG_SPLITS=FORBIDDEN')


if __name__ == '__main__':
    main()
