from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg is required")
class StreamRoundTripTests(unittest.TestCase):
    def test_prepare_and_finalize_preserve_count_and_size(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generic_restoration_test_") as temporary:
            root = Path(temporary)
            source_frames = root / "source_frames"
            source_frames.mkdir()
            for index in range(7):
                image = Image.new("RGB", (64, 48), (index * 20, 80, 160))
                image.save(source_frames / f"{index:08d}.png")
            source_video = root / "source.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-framerate",
                    "7",
                    "-i",
                    str(source_frames / "%08d.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(source_video),
                ],
                check=True,
            )
            canonical = root / "canonical"
            subprocess.run(
                [
                    "python",
                    "-m",
                    "generic_restoration.prepare_stream",
                    "--input-video",
                    str(source_video),
                    "--work-dir",
                    str(canonical),
                    "--smoke-frames",
                    "5",
                ],
                check=True,
            )
            manifest = json.loads((canonical / "manifest.json").read_text())
            self.assertEqual(manifest["frame_count"], 7)
            self.assertEqual((manifest["width"], manifest["height"]), (64, 48))
            smoke = json.loads((canonical / "manifest_smoke.json").read_text())
            self.assertEqual(smoke["frame_count"], 5)

            output_video = root / "output.mp4"
            report = root / "report.json"
            subprocess.run(
                [
                    "python",
                    "-m",
                    "generic_restoration.finalize_video",
                    "--manifest",
                    str(canonical / "manifest.json"),
                    "--frames",
                    str(canonical / "frames_full"),
                    "--output",
                    str(output_video),
                    "--report",
                    str(report),
                ],
                check=True,
            )
            checked = json.loads(report.read_text())
            self.assertEqual(checked["status"], "PASS")
            self.assertEqual(checked["frame_count"], 7)
            self.assertEqual(checked["encoded_frame_count"], 7)
            self.assertTrue(output_video.is_file())


if __name__ == "__main__":
    unittest.main()
