import argparse
from pathlib import Path

from data.mixed_deblur import build_mixed_dataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gopro-root', required=True)
    ap.add_argument('--dvd-root', required=True)
    ap.add_argument('--bsd-root', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--patch-size', type=int, default=256)
    return ap.parse_args()


def main():
    args = parse_args()
    roots = {
        'GoPro': args.gopro_root,
        'DVD': args.dvd_root,
        'BSD': args.bsd_root,
    }
    for name, p in roots.items():
        print(f'{name}_ROOT={Path(p).resolve()}')

    for t in (7, 30):
        print(f'\n=== AUDIT T={t} ===')
        try:
            ds, _, audit = build_mixed_dataset(
                roots, split=args.split, num_frames=t,
                patch_size=args.patch_size, train=True,
            )
        except Exception as e:
            print(f'AUDIT_T{t}_FAIL: {type(e).__name__}: {e}')
            continue
        print(f'TOTAL_WINDOWS={len(ds)}')
        totals = {}
        for row in audit:
            print(
                f"{row['family']}: blur={row['blur_root']} gt={row['gt_root']} "
                f"sequences={row['sequences']} windows={row['windows']}"
            )
            totals[row['family']] = totals.get(row['family'], 0) + row['windows']
        print('FAMILY_WINDOWS=' + ', '.join(f'{k}:{v}' for k, v in sorted(totals.items())))


if __name__ == '__main__':
    main()
