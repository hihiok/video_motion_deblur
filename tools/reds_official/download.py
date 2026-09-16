#!/usr/bin/env python3
"""Fetch the author's complete snah/REDS snapshot; never use OpenDataLab stubs.

Python >=3.9, requests and Pillow. No Hugging Face SDK/token/GPU required.
Metadata, ZIP central directories and disk capacity are checked before bulk I/O.
Partial files, conflicting originals and prior OpenXLab downloads are preserved.
"""
import argparse
import collections
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import sys
import time
from urllib.parse import quote, urlsplit
import uuid
import zipfile
import zlib

import requests

REPO = 'snah/REDS'
REVISION = '62dc25d16e6f43d2214f1b365023abda86f7a0ae'
SITE = 'https://seungjunnah.github.io/Datasets/reds.html'
HF = 'https://huggingface.co'
CORE = ('train_blur.zip', 'train_sharp.zip', 'val_blur.zip', 'val_sharp.zip')
GIB = 1024 ** 3
CHUNK = 1024 ** 2


class Blocked(RuntimeError):
    """A safe-to-log error without credentials or signed URLs."""


def check(condition, message):
    if not condition:
        raise Blocked(message)


def save_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def safe_parts(name):
    p = PurePosixPath(name)
    check(bool(name) and not p.is_absolute() and '\\' not in name and ':' not in name,
          'UNSAFE_ARCHIVE_PATH')
    check(all(v not in ('', '.', '..') for v in name.rstrip('/').split('/')),
          'UNSAFE_ARCHIVE_PATH')
    return p.parts


def safe_target(root, name):
    root = Path(root)
    check(not root.is_symlink(), 'SYMLINK_OUTPUT_ROOT')
    target = root.joinpath(*safe_parts(name))
    for p in (target, *target.parents):
        check(not p.is_symlink(), 'SYMLINK_OUTPUT_PATH')
    return target


class Client:
    def __init__(self, verify=True):
        self.session = requests.Session()
        # Use env proxies, but never implicitly read/send netrc credentials.
        self.session.auth = lambda request: request
        self.session.headers.update({'Accept-Encoding': 'identity',
                                     'User-Agent': 'REDS-official-downloader/1'})
        self.verify = verify

    def get(self, url, headers=None, stream=False):
        for attempt in range(4):
            try:
                r = self.session.get(url, headers=headers, stream=stream,
                                     timeout=(20, 90), verify=self.verify)
                check(urlsplit(r.url).scheme == 'https', 'NON_HTTPS_REDIRECT')
                if r.status_code in (408, 429, 500, 502, 503, 504):
                    r.close()
                    if attempt < 3:
                        time.sleep(2 ** attempt)
                        continue
                if r.status_code in (401, 403, 407):
                    code = r.status_code
                    r.close()
                    raise Blocked('HTTP_%d_ACCESS_OR_PROXY_BLOCK: check host policy; do not print signed URL' % code)
                if r.status_code not in (200, 206):
                    code = r.status_code
                    r.close()
                    raise Blocked('HTTP_%d' % code)
                return r
            except requests.exceptions.SSLError:
                raise Blocked('SSL_ERROR: supply trusted --ca-bundle or task-scoped --insecure') from None
            except requests.exceptions.ProxyError:
                raise Blocked('PROXY_ERROR: check local proxy.md; no credentials in report') from None
            except requests.exceptions.RequestException:
                if attempt == 3:
                    raise Blocked('NETWORK_ERROR: bounded retries exhausted') from None
                time.sleep(2 ** attempt)
        raise Blocked('NETWORK_ERROR')

    def json(self, url):
        with self.get(url) as r:
            try:
                return r.json()
            except ValueError:
                raise Blocked('INVALID_METADATA_JSON') from None


def resolve_url(name):
    # Fetch a fresh origin redirect on retries; do not persist expiring Xet/CDN URLs.
    return '%s/datasets/%s/resolve/%s/%s?download=true&reds_request=%s' % (
        HF, REPO, REVISION, quote(name, safe='/'), uuid.uuid4().hex)


def get_manifest(client):
    data = client.json('%s/api/datasets/%s/revision/%s?blobs=true' % (HF, REPO, REVISION))
    check(data.get('sha') == REVISION, 'REVISION_MISMATCH')
    files = []
    for source in data.get('siblings', []):
        name = source.get('rfilename', '')
        safe_parts(name)
        # Current author snapshot has only ZIP archives + these two small files.
        check(name.endswith('.zip') or name in ('README.md', '.gitattributes'),
              'UNEXPECTED_SOURCE_FILE: review pinned snapshot')
        size = source.get('size')
        check(isinstance(size, int) and size > 0, 'MISSING_FILE_SIZE')
        lfs = source.get('lfs') or {}
        if lfs:
            algorithm, digest = 'sha256', lfs.get('sha256')
            check(lfs.get('size') == size, 'LFS_SIZE_MISMATCH')
        else:
            algorithm, digest = 'git-sha1', source.get('blobId')
        check(isinstance(digest, str) and re.fullmatch(
            '[0-9a-f]{%d}' % (64 if algorithm == 'sha256' else 40), digest), 'MISSING_OFFICIAL_HASH')
        if name.endswith('.zip'):
            check(size > 1024 ** 2, 'SOURCE_STUB_ARCHIVE')
        files.append({'name': name, 'size': size, 'algorithm': algorithm, 'digest': digest})
    check(len({x['name'] for x in files}) == len(files), 'DUPLICATE_SOURCE_FILE')
    check(set(CORE).issubset(x['name'] for x in files), 'MISSING_CORE_ARCHIVE')
    check(sum(x['size'] for x in files) > GIB, 'SOURCE_EMPTY_OR_STUB')
    # Download small core validation packs first, but do not omit other repo files.
    priority = {n: i for i, n in enumerate(('README.md', '.gitattributes', 'val_blur.zip',
                                         'val_sharp.zip', 'train_blur.zip', 'train_sharp.zip'))}
    return sorted(files, key=lambda x: (priority.get(x['name'], 99), x['name']))


def content_range(response, start, total, end=None):
    m = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
    check(response.status_code == 206 and m is not None, 'RANGE_NOT_SUPPORTED: partial preserved')
    a, b, size = map(int, m.groups())
    check(a == start and a <= b < size and size == total and (end is None or b == end),
          'CONTENT_RANGE_MISMATCH: partial preserved')
    return b - a + 1


class RemoteZip(io.RawIOBase):
    """Only small, strict HTTP Range reads; never download a full ZIP during plan."""
    def __init__(self, client, item):
        self.client, self.item, self.pos = client, item, 0

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        base = (0, self.pos, self.item['size'])[whence]
        self.pos = base + offset
        if self.pos < 0:
            raise ValueError('negative seek')
        return self.pos

    def read(self, n=-1):
        size = self.item['size']
        n = max(0, size - self.pos) if n is None or n < 0 else min(n, max(0, size - self.pos))
        check(n <= 32 * CHUNK, 'REMOTE_ZIP_READ_TOO_LARGE')
        if n == 0:
            return b''
        end = self.pos + n - 1
        with self.client.get(resolve_url(self.item['name']), stream=True,
                             headers={'Range': 'bytes=%d-%d' % (self.pos, end)}) as r:
            content_range(r, self.pos, size, end)
            data = r.raw.read(n + 1)
        check(len(data) == n, 'TRUNCATED_REMOTE_ZIP_RANGE')
        self.pos += n
        return data


def zip_inventory(zf, output=None):
    seen, size, images, additional = set(), 0, 0, 0
    for info in zf.infolist():
        safe_parts(info.filename)
        check(info.filename not in seen, 'DUPLICATE_ZIP_MEMBER')
        seen.add(info.filename)
        mode = stat.S_IFMT(info.external_attr >> 16)
        check(mode in (0, stat.S_IFREG, stat.S_IFDIR), 'UNSAFE_ZIP_FILE_TYPE')
        check(not info.flag_bits & 1, 'ENCRYPTED_ZIP_NOT_ALLOWED')
        if not info.is_dir():
            size += info.file_size
            images += info.filename.lower().endswith(('.png', '.jpg', '.jpeg'))
            target = safe_target(output, info.filename) if output is not None else None
            if target is not None and target.exists():
                check(target.is_file() and target.stat().st_size == info.file_size,
                      'EXISTING_EXTRACT_SIZE_CONFLICT: original preserved')
            else:
                additional += info.file_size
    check(images > 0, 'EMPTY_ZIP_NO_IMAGES')
    return {'uncompressed_bytes': size, 'additional_extract_bytes': additional,
            'members': len(seen), 'images': images}


def hash_ok(path, item):
    if not path.is_file() or path.stat().st_size != item['size']:
        return False
    h = hashlib.sha256() if item['algorithm'] == 'sha256' else hashlib.sha1()
    if item['algorithm'] == 'git-sha1':
        h.update(('blob %d\0' % item['size']).encode('ascii'))
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * CHUNK), b''):
            h.update(chunk)
    return h.hexdigest() == item['digest']


def download_one(client, item, raw):
    final = safe_target(raw, item['name'])
    final.parent.mkdir(parents=True, exist_ok=True)
    part = final.with_name(final.name + '.part')
    sidecar = final.with_name(final.name + '.part.json')
    check(not part.is_symlink() and not sidecar.is_symlink(), 'SYMLINK_PARTIAL')
    if final.exists():
        check(hash_ok(final, item), 'EXISTING_ARCHIVE_CONFLICT: retained ' + item['name'])
        print('HASH_OK_SKIP ' + item['name'], flush=True)
        return
    if part.exists():
        check(sidecar.is_file() and json.loads(sidecar.read_text()) == item,
              'UNKNOWN_PARTIAL_OR_REVISION: preserved')
    else:
        save_json(sidecar, item)
    expected = item['size']
    for attempt in range(4):
        offset = part.stat().st_size if part.exists() else 0
        check(offset <= expected, 'PARTIAL_TOO_LARGE: preserved')
        if offset == expected:
            break
        try:
            headers = {'Range': 'bytes=%d-' % offset} if offset else {}
            with client.get(resolve_url(item['name']), headers=headers, stream=True) as r:
                if offset or r.status_code == 206:
                    span = content_range(r, offset, expected)
                else:
                    check(r.status_code == 200, 'UNEXPECTED_DOWNLOAD_STATUS')
                    span = expected
                check(r.headers.get('Content-Encoding', 'identity') == 'identity', 'ENCODED_RANGE_RESPONSE')
                supplied = r.headers.get('Content-Length')
                check(supplied is None or int(supplied) == span, 'CONTENT_LENGTH_MISMATCH')
                print('DOWNLOAD %s offset=%d total=%d' % (item['name'], offset, expected), flush=True)
                last, received = time.monotonic(), 0
                with part.open('ab') as f:
                    for chunk in r.iter_content(CHUNK):
                        if not chunk:
                            continue
                        check(received + len(chunk) <= span, 'RESPONSE_OVERSIZE: partial preserved')
                        f.write(chunk)
                        received += len(chunk)
                        if time.monotonic() - last > 60:
                            print('PROGRESS %s %.2f%%' % (item['name'], 100 * (offset + received) / expected), flush=True)
                            last = time.monotonic()
                    f.flush()
                    os.fsync(f.fileno())
                if received == span and part.stat().st_size == expected:
                    break
        except (requests.exceptions.RequestException, OSError) as exc:
            if isinstance(exc, OSError) and getattr(exc, 'errno', None) == 28:
                raise Blocked('DISK_SPACE_ERROR: partial preserved') from None
            if attempt == 3:
                raise Blocked('DOWNLOAD_INTERRUPTED: partial preserved; rerun same command') from None
        if attempt < 3:
            time.sleep(2 ** attempt)
    check(hash_ok(part, item), 'INCOMPLETE_OR_HASH_MISMATCH: partial preserved ' + item['name'])
    os.replace(part, final)
    sidecar.unlink(missing_ok=True)
    print('OFFICIAL_HASH_OK ' + item['name'], flush=True)


def file_crc(path):
    value = 0
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(CHUNK), b''):
            value = zlib.crc32(chunk, value)
    return value & 0xffffffff


def extract_one(archive, out):
    with zipfile.ZipFile(archive) as zf:
        inventory = zip_inventory(zf)
        for info in zf.infolist():
            target = safe_target(out, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                check(target.is_file() and target.stat().st_size == info.file_size
                      and file_crc(target) == info.CRC, 'EXTRACT_CONFLICT: original preserved')
                continue
            temp = target.with_name(target.name + '.reds-extract-part')
            check(not temp.is_symlink(), 'SYMLINK_EXTRACT_TEMP')
            # Only our own temp for this member is overwritten; committed files never are.
            with zf.open(info) as src, temp.open('wb') as dst:
                shutil.copyfileobj(src, dst, CHUNK)  # ZipExtFile checks CRC at EOF.
            check(temp.stat().st_size == info.file_size, 'TRUNCATED_EXTRACT')
            os.replace(temp, target)
    return inventory


def frame_index(archive, out):
    result = {}
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith('.png'):
                continue
            p = PurePosixPath(info.filename)
            check(len(p.parts) >= 2, 'UNEXPECTED_FRAME_LAYOUT')
            key = (p.parts[-2], p.name)
            check(key not in result, 'DUPLICATE_FRAME_KEY')
            result[key] = safe_target(out, info.filename)
    check(bool(result), 'MISSING_PNG_FRAMES')
    return result


def png_size(path):
    with path.open('rb') as f:
        h = f.read(24)
    check(len(h) == 24 and h[:8] == b'\x89PNG\r\n\x1a\n' and h[12:16] == b'IHDR', 'BAD_PNG_HEADER')
    return struct.unpack('>II', h[16:24])


def audit_pairs(raw, out, expected=None):
    from PIL import Image
    expected = expected or {'train': (240, 100), 'val': (30, 100)}
    results = {}
    for split, (scene_count, frame_count) in expected.items():
        blur = frame_index(raw / (split + '_blur.zip'), out)
        sharp = frame_index(raw / (split + '_sharp.zip'), out)
        check(blur.keys() == sharp.keys(), 'PAIRING_MISMATCH_' + split)
        counts = collections.Counter(s for s, _ in blur)
        check(len(counts) == scene_count and set(counts.values()) == {frame_count},
              'SEQUENCE_OR_FRAME_COUNT_MISMATCH_' + split)
        sizes = set()
        for k, path in blur.items():
            shape = png_size(path)
            check(shape == png_size(sharp[k]), 'PAIR_RESOLUTION_MISMATCH')
            sizes.add(shape)
        check(sizes == {(1280, 720)}, 'UNEXPECTED_RESOLUTION_' + split)
        scenes = sorted(counts)
        samples = sorted({scenes[0], scenes[len(scenes) // 2], scenes[-1]})
        decoded = 0
        for scene in samples:
            frames = sorted(k for k in blur if k[0] == scene)
            for key in {frames[0], frames[len(frames) // 2], frames[-1]}:
                for p in (blur[key], sharp[key]):
                    with Image.open(p) as im:
                        im.load()
                    decoded += 1
        results[split] = {'videos': len(counts), 'blur_frames': len(blur), 'sharp_frames': len(sharp),
                          'resolution_WH': [1280, 720], 'decoded_samples': decoded, 'pairing': 'PASS'}
    return results


def check_space(raw, out, files, uncompressed, reserve):
    remaining = 0
    for item in files:
        p = safe_target(raw, item['name'])
        part = p.with_name(p.name + '.part')
        check(not part.is_symlink(), 'SYMLINK_PARTIAL')
        have = p.stat().st_size if p.is_file() else (part.stat().st_size if part.is_file() else 0)
        remaining += max(0, item['size'] - have)
    same_device = raw.stat().st_dev == out.stat().st_dev
    free_raw, free_out = shutil.disk_usage(raw).free, shutil.disk_usage(out).free
    # Matching existing sizes reduce reservation; CRC is still checked during extraction.
    need_raw = remaining + reserve + (uncompressed if same_device else 0)
    need_out = uncompressed + reserve
    result = {'remaining_download_bytes': remaining, 'uncompressed_bytes': uncompressed,
              'reserve_bytes': reserve, 'raw_free_bytes': free_raw, 'out_free_bytes': free_out,
              'required_raw_bytes': need_raw, 'required_out_bytes': need_out,
              'same_filesystem': same_device, 'existing_extracts_verified_later_by_crc': True}
    check(free_raw >= need_raw and (same_device or free_out >= need_out),
          'DISK_SPACE_ERROR: need raw=%d out=%d; free raw=%d out=%d; no files deleted' %
          (need_raw, need_out, free_raw, free_out))
    return result


def run(args, report):
    raw, out = Path(args.raw), Path(args.output)
    for p in (raw, out):
        check(not p.is_symlink(), 'SYMLINK_ROOT')
        p.mkdir(parents=True, exist_ok=True)
    check(raw.resolve() != out.resolve(), 'RAW_AND_OUTPUT_MUST_DIFFER')
    # Locks held for the whole function (both original download and shared output).
    locks = []
    try:
        for directory in (raw, out):
            handle = safe_target(directory, '.reds-official.lock').open('a')
            locks.append(handle)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise Blocked('ALREADY_RUNNING: do not launch a second copy') from None
        verify = False if args.insecure else (args.ca_bundle or True)
        if args.ca_bundle:
            check(Path(args.ca_bundle).is_file(), 'CA_BUNDLE_NOT_FOUND')
        client = Client(verify)
        report['stage'] = 'SOURCE_MANIFEST'
        files = get_manifest(client)
        report['file_count'] = len(files)
        report['compressed_bytes'] = sum(x['size'] for x in files)
        save_json(Path(args.report_dir) / 'manifest.json', {'repository': REPO, 'revision': REVISION, 'files': files})
        inventory, total, additional = {}, 0, 0
        report['stage'] = 'REMOTE_ZIP_PREFLIGHT'
        for item in files:
            if item['name'].endswith('.zip'):
                print('ZIP_PREFLIGHT ' + item['name'], flush=True)
                with zipfile.ZipFile(RemoteZip(client, item)) as zf:
                    inv = zip_inventory(zf, out)
                inventory[item['name']] = inv
                total += inv['uncompressed_bytes']
                additional += inv['additional_extract_bytes']
        report['archive_inventory'] = inventory
        report['disk_plan'] = check_space(raw, out, files, additional, int(args.reserve_gib * GIB))
        save_json(Path(args.report_dir) / 'preflight.json', report)
        print('PREFLIGHT_PASS files=%d compressed=%.2f_GiB unpacked=%.2f_GiB' %
              (len(files), report['compressed_bytes'] / GIB, total / GIB), flush=True)
        if args.plan_only:
            report.update(status='PLAN_ONLY', stage='PREFLIGHT_COMPLETE', human_action_required='NO')
            return
        # Fail before large downloads if sample decoding dependency is unavailable.
        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            raise Blocked('PILLOW_REQUIRED_IN_ISOLATED_VENV') from None
        report['stage'], report['downloaded_hash_verified'] = 'DOWNLOAD', []
        for item in files:
            download_one(client, item, raw)
            report['downloaded_hash_verified'].append(item['name'])
        report['stage'], report['extracted_crc_verified'] = 'EXTRACT', []
        for item in files:
            if item['name'].endswith('.zip'):
                extract_one(raw / item['name'], out)
                report['extracted_crc_verified'].append(item['name'])
                print('EXTRACT_CRC_OK ' + item['name'], flush=True)
        report['stage'] = 'PAIR_AUDIT'
        report['pairs'] = audit_pairs(raw, out)
        report.update(status='SUCCESS', stage='COMPLETE', human_action_required='NO',
                      test_sharp='NOT_PROVIDED_BY_SOURCE',
                      coverage='ALL_FILES_IN_PINNED_snah/REDS; excludes separate snah/REDS_orig')
    finally:
        for handle in locks:
            handle.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--report-dir', required=True)
    tls = p.add_mutually_exclusive_group()
    tls.add_argument('--insecure', action='store_true', help='Disable TLS verification for this process only')
    tls.add_argument('--ca-bundle')
    p.add_argument('--plan-only', action='store_true', help='Metadata/range/space check; no full dataset download')
    p.add_argument('--reserve-gib', type=float, default=10)
    args = p.parse_args(argv)
    check(args.reserve_gib >= 1, 'RESERVE_MUST_BE_AT_LEAST_1_GIB')
    os.umask(0o077)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {'status': 'RUNNING', 'source': REPO, 'source_revision': REVISION,
              'source_page': SITE, 'raw_root': args.raw, 'output_root': args.output,
              'ssl_method': 'VERIFY_DISABLED' if args.insecure else ('CA_BUNDLE' if args.ca_bundle else 'DEFAULT_VERIFY'),
              'human_action_required': 'UNDETERMINED'}
    code = 0
    try:
        run(args, report)
    except Blocked as exc:
        report.update(status='BLOCKED', reason=str(exc), human_action_required='SEE_REASON')
        code = 2
    except KeyboardInterrupt:
        report.update(status='INTERRUPTED', reason='User/process interruption; partial files preserved',
                      human_action_required='RESTART_SAME_COMMAND')
        code = 130
    except Exception as exc:
        # Avoid serializing exception text, request objects, tokens, proxy URLs, etc.
        report.update(status='BLOCKED', reason='UNEXPECTED_' + type(exc).__name__,
                      human_action_required='SEND_SANITIZED_REPORT_AND_COMMIT; DO_NOT_EDIT_CODE')
        code = 2
    report['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    save_json(report_dir / 'report.json', report)
    (report_dir / 'FINAL_REDS_OFFICIAL_REPORT.md').write_text(
        '# REDS 官方源下载报告\n\n```json\n' + json.dumps(report, ensure_ascii=False, indent=2) + '\n```\n', encoding='utf-8')
    print('STATUS: ' + report['status'], flush=True)
    if 'reason' in report:
        print('REASON: ' + report['reason'], flush=True)
    print('HUMAN_ACTION_REQUIRED: ' + report['human_action_required'], flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
