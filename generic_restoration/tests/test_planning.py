from __future__ import annotations

import unittest

from generic_restoration.planning import (
    padded_target_size,
    plan_seed_chunks,
    smallest_8n1_geq,
    tile_starts,
)


class PlanningTests(unittest.TestCase):
    def test_flashvsr_keeps_all_452_frames(self) -> None:
        model_frames = smallest_8n1_geq(452 + 4)
        self.assertEqual(model_frames, 457)
        self.assertEqual(model_frames % 8, 1)
        self.assertGreaterEqual(model_frames - 4, 452)

    def test_flashvsr_uses_padding_not_center_crop(self) -> None:
        self.assertEqual(padded_target_size(1280, 720, 1.0), (1280, 720, 1280, 768))

    def test_realviformer_tiles_cover_full_axis(self) -> None:
        starts = tile_starts(1280, 256, 48)
        covered = set()
        for start in starts:
            covered.update(range(start, min(start + 256, 1280)))
        self.assertEqual(len(covered), 1280)
        self.assertEqual(starts[-1], 1280 - 256)

    def test_seed_chunks_have_contiguous_cores(self) -> None:
        chunks = plan_seed_chunks(452, 49, 4)
        cursor = 0
        for chunk in chunks:
            self.assertEqual(chunk["core_start"], cursor)
            self.assertLessEqual(chunk["clip_start"], chunk["core_start"])
            self.assertGreaterEqual(chunk["clip_end"], chunk["core_end"])
            self.assertEqual(chunk["keep_end"] - chunk["keep_start"], chunk["core_end"] - chunk["core_start"])
            cursor = int(chunk["core_end"])
        self.assertEqual(cursor, 452)


if __name__ == "__main__":
    unittest.main()
