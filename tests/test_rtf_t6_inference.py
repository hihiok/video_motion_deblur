from rtf_t6.inference import owned_temporal_range, window_starts
import pytest


@pytest.mark.parametrize('window,overlap', [(3, 2), (6, 4)])
def test_owned_ranges_cover_every_frame_once(window, overlap):
    for total in range(1, 30):
        starts = window_starts(total, window=window, overlap=overlap)
        owned = []
        for index, start in enumerate(starts):
            end = min(start + window, total)
            left, right = owned_temporal_range(starts, index, end, window, total)
            owned.extend(range(left, right))
        assert owned == list(range(total))
