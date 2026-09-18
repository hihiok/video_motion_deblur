#!/usr/bin/env python3
"""Offline shell/media checks only. No download, credentials, CUDA, or model inference.

The media tests place a COPY of the padded input in a temporary fixture output
folder to exercise encoding/cropping. That fixture is NOT a restored model result.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('run.sh')

def execute(args, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=True, **kwargs)

class RunnerTests(unittest.TestCase):
    def test_bash_syntax(self):
        execute(['bash', '-n', str(SCRIPT)])

    def test_requires_explicit_model_paths(self):
        env = dict(os.environ)
        for name in ('MODEL_REPO', 'PYTHON_BIN', 'WAN_MODEL_DIR', 'DMD_CHECKPOINT', 'GPU'):
            env.pop(name, None)
        proc = subprocess.run(['bash', str(SCRIPT)], env=env, text=True, capture_output=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('MODEL_REPO', proc.stderr)

    def media_roundtrip(self, width, height):
        for tool in ('ffmpeg', 'ffprobe', 'git'):
            if not shutil.which(tool):
                self.skipTest(f'{tool} not installed')
        with tempfile.TemporaryDirectory(prefix='rvd-media-test-') as temp:
            root = Path(temp)
            run = root / 'run'
            run.mkdir()
            repo = root / 'fixture_repo'
            execute(['git', 'init', '-q', str(repo)])
            source = root / 'fixture_not_official_sample.mp4'
            execute(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
                     f'testsrc2=size={width}x{height}:rate=5:duration=1',
                     '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p', str(source)])
            text = SCRIPT.read_text()
            prep = text.split('STAGE=decode\n', 1)[1].split('STAGE=inference\n', 1)[0]
            post = text.split('STAGE=verify_and_encode\n', 1)[1].split('cat > "$RUN_DIR/STATUS.txt"', 1)[0]
            # Deliberately do NOT run production model code. This tests media only.
            fixture = '\ncp "$MODEL_INPUT"/*.png "$RUN_DIR/frames_model/"\n'
            env = dict(os.environ, INPUT=str(source), RUN_DIR=str(run),
                       MODEL_REPO=str(repo), PYTHON_BIN=sys.executable)
            execute(['bash', '-c', 'set -Eeuo pipefail\n' + prep + fixture + post], env=env)
            self.assertEqual(len(list((run / 'frames').glob('*.png'))), 5)
            for folder in ('input_frames', 'frames'):
                md5 = execute(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1',
                               '-framerate', '5', '-i', str(run / folder / '%08d.png'),
                               '-pix_fmt', 'rgb24', '-f', 'md5', '-']).stdout
                if folder == 'input_frames':
                    original_md5 = md5
                else:
                    self.assertEqual(md5, original_md5, 'Padding/unpadding changed original RGB pixels')
            self.assertEqual(len(list((run / 'previews').glob('*.png'))), 3)
            for name in ('output_00_motion.mp4', 'input_vs_realvdeblur.mp4'):
                self.assertTrue((run / name).stat().st_size > 0)

    def test_divisible_size_media_roundtrip(self):
        self.media_roundtrip(64, 48)

    def test_padding_preserves_original_pixels(self):
        self.media_roundtrip(66, 50)

if __name__ == '__main__':
    unittest.main(verbosity=2)
