import importlib.util
from pathlib import Path
import tempfile
import unittest
from PIL import Image

SPEC = importlib.util.spec_from_file_location("runner", Path(__file__).with_name("run_realvdeblur_sequence.py"))
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)

GPU_CSV = """0, GPU-zero, Blackwell, 97280, 9700, 00000000:01:00.0
4, GPU-four, Blackwell, 97280, 83074, 00000000:05:00.0
6, GPU-six, Blackwell, 97280, 97230, 00000000:07:00.0
"""


class RunnerTests(unittest.TestCase):
    def test_natural_order(self):
        self.assertEqual(sorted(["10.png", "2.png", "1.png"], key=r.natural_key),
                         ["1.png", "2.png", "10.png"])

    def test_numeric_gpu_selection(self):
        gpu, all_gpus = r.select_gpu(GPU_CSV)
        self.assertEqual(gpu["index"], 6)
        self.assertEqual(len(all_gpus), 3)

    def test_scheduler_intersection(self):
        gpu, _ = r.select_gpu(GPU_CSV, allowed="4,6", inherited="0,4")
        self.assertEqual(gpu["uuid"], "GPU-four")

    def test_uuid_restriction(self):
        gpu, _ = r.select_gpu(GPU_CSV, inherited="GPU-zero,GPU-four")
        self.assertEqual(gpu["index"], 4)

    def test_empty_allocation_blocks(self):
        with self.assertRaises(RuntimeError):
            r.select_gpu(GPU_CSV, inherited="")
        with self.assertRaises(RuntimeError):
            r.select_gpu(GPU_CSV, inherited="0", allowed="6")

    def test_fps_valid(self):
        self.assertEqual(r.parse_fps("30000/1001"), "30000/1001")
        self.assertEqual(r.parse_fps(25), "25/1")
        for v in ("0", "-1", "1001"):
            with self.assertRaises(RuntimeError):
                r.parse_fps(v)

    def test_fps_metadata_and_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp)/"002"
            inp = scene/"Blur/RGB"
            inp.mkdir(parents=True)
            self.assertTrue(r.resolve_fps(inp)["assumed"])
            (scene/"fps.txt").write_text("25")
            f = r.resolve_fps(inp)
            self.assertEqual(f["value"], "25/1")
            self.assertFalse(f["assumed"])
            (inp/"metadata.json").write_text('{"fps":30}')
            with self.assertRaises(RuntimeError):
                r.resolve_fps(inp)
            self.assertEqual(r.resolve_fps(inp, "60")["value"], "60/1")

    def test_duplicate_stem_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            Image.new("RGB",(32,32)).save(p/"1.jpg")
            Image.new("RGB",(32,32)).save(p/"1.png")
            with self.assertRaises(RuntimeError):
                r.list_frames(p)

    def test_prepare_jpeg_real_png_no_resize(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp, out = Path(tmp)/"in", Path(tmp)/"out"
            inp.mkdir(); out.mkdir()
            im=Image.new("RGB",(35,33),(12,40,70))
            im.putpixel((1,1),(200,5,10))
            im.save(inp/"10.jpg"); im.save(inp/"2.png")
            hashes = [r.sha256(p) for p in sorted(inp.iterdir())]
            m = r.prepare_frames(inp,out)
            self.assertEqual([e["source_name"] for e in m["entries"]],["2.png","10.jpg"])
            self.assertEqual(m["output_wh"],[32,32])
            self.assertEqual(m["official_center_crop_ltrb"],[1,0,2,1])
            with Image.open(out/"_input_png/00000001.png") as staged, Image.open(inp/"10.jpg") as source:
                self.assertEqual(staged.format,"PNG")
                self.assertEqual(staged.size,(35,33))
                self.assertEqual(staged.tobytes(),source.convert("RGB").tobytes())
            self.assertEqual(hashes,[r.sha256(p) for p in sorted(inp.iterdir())])

    def test_mixed_resolution_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp,out = Path(tmp)/"in", Path(tmp)/"out"
            inp.mkdir(); out.mkdir()
            Image.new("RGB",(32,32)).save(inp/"0.png")
            Image.new("RGB",(48,32)).save(inp/"1.png")
            with self.assertRaises(RuntimeError):
                r.prepare_frames(inp,out)

    def test_real_mp4_encode_decode_and_preserve(self):
        try:
            ffmpeg=r.find_ffmpeg()
        except RuntimeError:
            self.skipTest("ffmpeg libx264 unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            inp,out=Path(tmp)/"in",Path(tmp)/"out"
            inp.mkdir(); out.mkdir()
            for n in (1,2,10):
                im=Image.new("RGB",(32,32),(n*5,80,150))
                im.putpixel((n,7),(250,100,20))
                im.save(inp/f"{n}.png")
            m=r.prepare_frames(inp,out)
            frames=out/"frames";frames.mkdir()
            for e in m["entries"]:
                with Image.open(e["source"]) as im:
                    im.save(frames/e["output_name"])
            self.assertTrue(r.validate_pngs(frames,m)["passed"])
            hashes = [r.sha256(p) for p in sorted(frames.iterdir())]
            video=r.export_mp4(out,m,{"value":"30000/1001","source":"test","assumed":False},ffmpeg)
            self.assertTrue(video["passed"])
            self.assertEqual(video["decoded_frames"],3)
            self.assertEqual(hashes,[r.sha256(p) for p in sorted(frames.iterdir())])
            with self.assertRaises(RuntimeError):
                r.export_mp4(out,m,{"value":"30/1"},ffmpeg)


if __name__ == '__main__':
    unittest.main()
