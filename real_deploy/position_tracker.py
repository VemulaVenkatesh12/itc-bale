"""
Turns a per-frame stream of (possibly missing, possibly noisy) triangulated
3D detections into a stable position estimate a control loop could actually
act on - this is the piece that replaces BOTH the simulation's dead-
reckoning (frontend/src/live/controller.ts's driveToward - there is no
"assume a constant velocity and integrate time" for real hardware with no
encoders) AND naively trusting every raw triangulated frame (which is
exactly what caused the crane to hunt back and forth chasing noisy vision
in the simulation - see project memory itc-crane-vision-position-hunting-
bug and itc-crane-full-batch-completion-fix for the two times that bit).

## Why an exponential moving average (EMA), not a Kalman filter

A Kalman filter is the textbook-correct answer, but it needs a MOTION
MODEL (how fast/how the hook actually accelerates) that nobody has
measured yet on real hardware. An EMA needs one number (`alpha`) that
directly trades off responsiveness vs. noise rejection, is trivial to
reason about and tune by hand, and is the right STARTING point. Swap in a
Kalman filter later once real motion characteristics (from real testing)
justify the extra complexity - the PositionTracker interface below
(`update()` -> `PositionEstimate`) is deliberately generic enough that a
Kalman-based implementation could be substituted without changing any
caller.

## Staleness, not just smoothing

A smoothed position from 10 seconds ago (e.g. the hook is currently
occluded, or off-frame in both cameras) is not a position a control loop
should act on - `PositionEstimate.stale` is the explicit "don't trust this"
signal, separate from confidence, since a stale-but-was-once-confident
estimate is a different failure mode than a fresh-but-noisy one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PositionEstimate:
    position: np.ndarray | None  # smoothed (x, y, z), or None if never observed at all
    confidence: float  # confidence of the most recent measurement folded in (not smoothed itself)
    age_s: float  # seconds since the last measurement was folded in
    stale: bool  # age_s > the tracker's stale_after_s - see PositionTracker


class PositionTracker:
    def __init__(self, alpha: float = 0.3, stale_after_s: float = 1.0):
        """`alpha`: weight given to each new measurement (0-1). Higher =
        trusts new measurements more (less smoothing, more responsive to
        real movement, more sensitive to noise); lower = smoother but
        laggier. 0.3 is a reasonable starting point for a target moving at
        walking pace or slower relative to the tick rate; retune once real
        detection-noise magnitude and tick rate are known.

        `stale_after_s`: how long without a confident measurement before
        `PositionEstimate.stale` goes True. Should be a few multiples of
        the expected detection tick interval, not a hair-trigger - a
        single missed frame from normal detection noise shouldn't count as
        "lost the target." """
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._stale_after_s = stale_after_s
        self._smoothed: np.ndarray | None = None
        self._confidence = 0.0
        self._last_update_time: float | None = None

    def update(self, measurement: np.ndarray | None, confidence: float, now_s: float) -> PositionEstimate:
        """Call once per tick. `measurement` is this tick's triangulated
        (x, y, z), or None if nothing matched this tick (occlusion, low
        confidence filtered out upstream, etc. - see capture_infer.py's
        matching). `now_s` is a monotonic clock reading (e.g.
        time.monotonic()) - the caller's responsibility, not this class's,
        since real deployments and tests both need control over what
        "now" means."""
        if measurement is not None:
            if self._smoothed is None:
                self._smoothed = np.array(measurement, dtype=np.float64)
            else:
                self._smoothed = self._alpha * np.array(measurement, dtype=np.float64) + (1 - self._alpha) * self._smoothed
            self._confidence = confidence
            self._last_update_time = now_s

        age = math.inf if self._last_update_time is None else now_s - self._last_update_time
        return PositionEstimate(
            position=None if self._smoothed is None else self._smoothed.copy(),
            confidence=self._confidence,
            age_s=age,
            stale=age > self._stale_after_s,
        )

    def reset(self) -> None:
        """Discard the current estimate entirely (e.g. after a long gap or
        a phase change where stale history shouldn't bias the new
        estimate) - the next update() starts fresh rather than blending
        into old data."""
        self._smoothed = None
        self._confidence = 0.0
        self._last_update_time = None


def distance_m(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def has_arrived(current: np.ndarray, target: np.ndarray, tolerance_m: float) -> bool:
    """`tolerance_m` should be set with real numbers in hand, not guessed:
    at minimum, above your calibration's reprojection error converted to
    real-world units (see calibrate_extrinsics.py's printed reprojection
    error) and above the PositionTracker's own smoothing lag at your
    tick rate - an arrival tolerance tighter than your own measurement
    noise floor will never reliably trigger, the same failure mode that
    made the simulation's ARRIVE_EPS=0.02 (2cm) unreachable against noisy
    live-tracked targets (see itc-crane-full-batch-completion-fix)."""
    return distance_m(current, target) <= tolerance_m
