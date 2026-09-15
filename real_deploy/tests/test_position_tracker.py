"""
Verifies position_tracker.py's smoothing/staleness/arrival logic - pure
math, no camera or hardware needed. Run: py tests/test_position_tracker.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from position_tracker import PositionTracker, distance_m, has_arrived


def test_first_measurement_is_taken_as_is():
    tracker = PositionTracker(alpha=0.3)
    est = tracker.update(np.array([1.0, 2.0, 3.0]), confidence=0.9, now_s=0.0)
    assert np.allclose(est.position, [1.0, 2.0, 3.0])
    assert est.confidence == 0.9
    assert est.age_s == 0.0
    assert not est.stale


def test_smoothing_converges_toward_a_steady_measurement():
    tracker = PositionTracker(alpha=0.3)
    target = np.array([5.0, 0.0, 0.0])
    est = None
    for i in range(50):
        est = tracker.update(target, confidence=0.9, now_s=float(i))
    assert np.allclose(est.position, target, atol=1e-6), "should converge exactly onto a constant measurement"


def test_smoothing_damps_single_frame_noise():
    # Same scenario that caused the simulation's hunting bug: a target that
    # jitters around a true value tick to tick. Checking a single final
    # smoothed sample against a tight tolerance would be a flaky test (the
    # smoothed signal still has *some* variance, so any one draw can land
    # anywhere in that distribution's tail) - instead this checks that the
    # smoothed trajectory's spread (std dev, over its steady-state tail) is
    # meaningfully smaller than the raw per-frame noise, which is the
    # actual claim "smoothing damps noise" makes.
    rng = np.random.default_rng(42)
    true_pos = np.array([2.0, 1.0, 0.5])
    tracker = PositionTracker(alpha=0.2)
    smoothed_trail = []
    for i in range(300):
        noisy = true_pos + rng.normal(scale=0.3, size=3)  # +-30cm noise, deliberately large
        est = tracker.update(noisy, confidence=0.8, now_s=float(i) * 0.1)
        if i >= 100:  # skip the initial transient, only look at steady state
            smoothed_trail.append(est.position)
    smoothed_std = float(np.std(np.array(smoothed_trail), axis=0).mean())
    assert smoothed_std < 0.15, f"expected smoothed steady-state std well under the 0.3m raw noise, got {smoothed_std:.3f}m"


def test_missing_measurement_keeps_last_smoothed_value():
    tracker = PositionTracker(alpha=0.3)
    tracker.update(np.array([1.0, 1.0, 1.0]), confidence=0.9, now_s=0.0)
    est = tracker.update(None, confidence=0.0, now_s=0.5)  # occluded this tick
    assert np.allclose(est.position, [1.0, 1.0, 1.0]), "a missed tick shouldn't reset or drift the estimate"


def test_goes_stale_after_no_updates():
    tracker = PositionTracker(alpha=0.3, stale_after_s=1.0)
    tracker.update(np.array([0.0, 0.0, 0.0]), confidence=0.9, now_s=0.0)
    fresh = tracker.update(None, confidence=0.0, now_s=0.5)
    stale = tracker.update(None, confidence=0.0, now_s=2.0)
    assert not fresh.stale, "0.5s since last update should still be fresh (stale_after_s=1.0)"
    assert stale.stale, "2.0s since last update should be stale (stale_after_s=1.0)"


def test_never_observed_has_no_position():
    tracker = PositionTracker()
    est = tracker.update(None, confidence=0.0, now_s=5.0)
    assert est.position is None
    assert est.stale, "no measurement ever received should count as stale (age is infinite)"


def test_reset_discards_history():
    tracker = PositionTracker(alpha=0.3)
    tracker.update(np.array([10.0, 10.0, 10.0]), confidence=0.9, now_s=0.0)
    tracker.reset()
    est = tracker.update(np.array([0.0, 0.0, 0.0]), confidence=0.9, now_s=1.0)
    assert np.allclose(est.position, [0.0, 0.0, 0.0]), "post-reset, the first update should be taken as-is, not blended with old history"


def test_has_arrived_true_within_tolerance():
    assert has_arrived(np.array([1.0, 1.0, 1.0]), np.array([1.02, 1.0, 1.0]), tolerance_m=0.05)


def test_has_arrived_false_outside_tolerance():
    assert not has_arrived(np.array([1.0, 1.0, 1.0]), np.array([1.5, 1.0, 1.0]), tolerance_m=0.05)


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
