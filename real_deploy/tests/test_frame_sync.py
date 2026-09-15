"""
Verifies frame_sync.py's timestamp pairing - pure list math, no video file
or camera needed. Run: py tests/test_frame_sync.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame_sync import measured_fps, pair_timestamps


def _series(fps, count, start=0.0):
    step = 1000.0 / fps
    return [start + i * step for i in range(count)]


def test_identical_streams_pair_one_to_one():
    ts = _series(20, 50)
    pairs = pair_timestamps(ts, ts, max_skew_ms=40)
    assert len(pairs) == 50
    assert all(p.index_a == p.index_b for p in pairs)
    assert all(abs(p.skew_ms) < 1e-9 for p in pairs)


def test_mismatched_frame_rates_stay_bounded():
    """The real ITC case: 20 fps vs 12.5 fps. Every emitted pair must be
    within tolerance - that is the whole point of pairing by timestamp."""
    ts_a = _series(20, 200)
    ts_b = _series(12.5, 125)
    pairs = pair_timestamps(ts_a, ts_b, max_skew_ms=40)
    assert pairs, "expected some pairs"
    assert all(abs(p.skew_ms) <= 40 for p in pairs)
    # Cannot pair more than the slower stream has frames.
    assert len(pairs) <= len(ts_b)


def test_naive_one_for_one_pairing_would_have_drifted():
    """Regression guard for the bug this module exists to fix: reading one
    frame from each source per loop iteration (capture_infer.py's approach)
    drifts without bound when the rates differ.

    Uses the ACTUAL 2026-08-18 ITC recording lengths - 258.4s at 20 fps and
    12.5 fps - so the number this asserts is the real field error, not a
    made-up one. Naive pairing ends up over 90 SECONDS apart.
    """
    ts_a = _series(20, 5167)
    ts_b = _series(12.5, 3101)
    naive_worst = max(abs(ts_a[i] - ts_b[i]) for i in range(min(len(ts_a), len(ts_b))))
    assert naive_worst > 90_000, f"expected ~93s of naive drift, got {naive_worst}ms"

    synced_worst = max(abs(p.skew_ms) for p in pair_timestamps(ts_a, ts_b, max_skew_ms=40))
    assert synced_worst <= 40
    assert synced_worst < naive_worst / 1000


def test_constant_clock_offset_is_corrected():
    ts_a = _series(20, 60)
    skew = 4300.0  # the .101/.102 filename difference
    ts_b = [t - skew for t in ts_a]

    assert not pair_timestamps(ts_a, ts_b, max_skew_ms=40), "uncorrected skew should pair nothing"

    pairs = pair_timestamps(ts_a, ts_b, max_skew_ms=40, offset_ms=skew)
    assert len(pairs) == 60
    assert all(abs(p.skew_ms - skew) < 1e-6 for p in pairs)


def test_pairs_beyond_tolerance_are_dropped_not_forced():
    """A dropped frame is a visible gap; a badly-paired frame is an
    invisible wrong answer. Never emit the latter."""
    ts_a = [0.0, 1000.0, 2000.0]
    ts_b = [0.0, 5000.0]
    pairs = pair_timestamps(ts_a, ts_b, max_skew_ms=40)
    assert len(pairs) == 1
    assert pairs[0].index_a == 0 and pairs[0].index_b == 0


def test_no_b_frame_is_reused_across_pairs():
    """One physical exposure must not be triangulated as two separate
    instants - that would fabricate motion that never happened."""
    ts_a = _series(50, 100)
    ts_b = _series(10, 20)
    pairs = pair_timestamps(ts_a, ts_b, max_skew_ms=40)
    used = [p.index_b for p in pairs]
    assert len(used) == len(set(used))


def test_empty_input_pairs_nothing():
    assert pair_timestamps([], _series(20, 10), max_skew_ms=40) == []
    assert pair_timestamps(_series(20, 10), [], max_skew_ms=40) == []


def test_measured_fps_recovers_the_real_rate():
    """Guards the actual field bug: cam .102's container claims 25 fps while
    its timestamps say 12.5."""
    assert abs(measured_fps(_series(12.5, 100)) - 12.5) < 0.01
    assert abs(measured_fps(_series(20, 100)) - 20.0) < 0.01
    assert measured_fps([1.0]) == 0.0


def test_invalid_tolerance_is_rejected():
    try:
        pair_timestamps([0.0], [0.0], max_skew_ms=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-positive max_skew_ms")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
