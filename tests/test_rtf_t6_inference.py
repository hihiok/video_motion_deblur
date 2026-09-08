from rtf_t6.inference import owned_temporal_range, window_starts


def test_owned_ranges_cover_every_frame_once():
    for total in range(1, 30):
        starts = window_starts(total, window=6, overlap=4)
        owned = []
        for index, start in enumerate(starts):
            end = min(start + 6, total)
            left, right = owned_temporal_range(starts, index, end, 6, total)
            owned.extend(range(left, right))
        assert owned == list(range(total))
