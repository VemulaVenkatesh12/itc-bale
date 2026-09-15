"""
Pairs frames from two independent cameras by TIMESTAMP rather than by loop
iteration - the fix for the synchronization limitation `capture_infer.py`'s
docstring flags as unimplemented.

## Why this is needed, with real numbers

`capture_infer.py` reads one frame from each source per loop iteration and
treats them as simultaneous. That silently assumes both cameras run at the
same rate. The 2026-08-18 ITC recordings do not:

  cam .101   2560x1440   20.0 fps   5167 frames over 258.4s
  cam .102   1920x1080   12.5 fps   3101 frames over 258.4s
             (its container CLAIMS 25 fps - the container is wrong)

Read one-for-one, the two streams drift apart by a full second every ~1.7
seconds of footage, so by the end of a 4-minute clip the "simultaneous" pair
is over two minutes apart. Triangulating that pair produces a confident,
precise, completely wrong 3D point - a bale position computed from where the
hook was two minutes ago. Static bales hide the error; a moving hook is
where it becomes dangerous.

The two cameras' clocks also disagree (the .101/.102 filenames differ by
4.3s, while .102's burned-in OSD is ~7-8s off its own filename start), so
`offset_ms` exists to correct a measured constant skew. Measure it - film
one visible event in both views and diff the timestamps - rather than
guessing.

## What this does NOT fix

Pairing by timestamp bounds the error; it cannot remove it. Two cameras that
are not hardware-triggered together still expose at different instants, so a
pair is at best `max_skew_ms` apart. At 12.5 fps the best achievable pairing
is ~40ms, during which a hook moving 0.5 m/s travels 2cm. That is the
accuracy floor of this rig, and it is a genlock/trigger-cable problem, not a
software one.

The pairing core (`pair_timestamps`) is deliberately pure - a function over
two lists of timestamps - so it is testable without any video file or
camera. See tests/test_frame_sync.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass
class FramePair:
    index_a: int
    index_b: int
    t_a_ms: float
    t_b_ms: float

    @property
    def skew_ms(self) -> float:
        """Signed: positive means camera A's frame is LATER than camera B's."""
        return self.t_a_ms - self.t_b_ms


def pair_timestamps(
    ts_a: list[float],
    ts_b: list[float],
    max_skew_ms: float = 40.0,
    offset_ms: float = 0.0,
) -> list[FramePair]:
    """Greedy two-pointer merge: for each frame of the SLOWER stream, take
    the nearest-in-time frame of the other, and drop the pair entirely if
    even that nearest match is worse than `max_skew_ms`.

    Dropping is the important part. The alternative - pairing whatever is
    closest no matter how far off - is what makes a desynchronized rig look
    like it is working: it always yields a pair, so it always yields a 3D
    point, and nothing in the output distinguishes a 5ms pair from a 5000ms
    one. A dropped frame is a visible gap; a badly-paired frame is an
    invisible wrong answer.

    `offset_ms` is ADDED to every `ts_b` before comparison, to cancel a known
    constant clock skew between the two recordings.
    """
    if max_skew_ms <= 0:
        raise ValueError(f"max_skew_ms must be positive, got {max_skew_ms}")

    pairs: list[FramePair] = []
    j = 0
    claimed_b: set[int] = set()

    for i, ta in enumerate(ts_a):
        # Advance j while the NEXT b-frame is a strictly better match than the current.
        while j + 1 < len(ts_b) and abs((ts_b[j + 1] + offset_ms) - ta) <= abs((ts_b[j] + offset_ms) - ta):
            j += 1
        if j >= len(ts_b):
            break
        skew = abs((ts_b[j] + offset_ms) - ta)
        if skew > max_skew_ms or j in claimed_b:
            continue
        claimed_b.add(j)
        pairs.append(FramePair(index_a=i, index_b=j, t_a_ms=ta, t_b_ms=ts_b[j]))

    return pairs


def read_timestamps(source: str) -> list[float]:
    """Decodes `source` start to finish, collecting each frame's presentation
    timestamp in ms.

    This walks the whole file because CAP_PROP_POS_MSEC is only trustworthy
    after an actual decoded read - seeking to sample it is exactly how the
    fps of these recordings gets misread in the first place (the container's
    declared rate is not the real one).
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"could not open source: {source}")
    timestamps: list[float] = []
    try:
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    finally:
        cap.release()
    return timestamps


def measured_fps(timestamps: list[float]) -> float:
    """Real frame rate implied by the timestamps - use this, never
    CAP_PROP_FPS, which reports the container's (here wrong) claim."""
    if len(timestamps) < 2:
        return 0.0
    span_s = (timestamps[-1] - timestamps[0]) / 1000.0
    return (len(timestamps) - 1) / span_s if span_s > 0 else 0.0


class SyncedFrameReader:
    """Iterates timestamp-matched frame pairs from two sources.

    Both sources are indexed first (one full decode pass each) so pairing
    decisions are made with the whole timeline known, then frames are
    re-read in pair order. That costs an extra pass but makes the pairing
    correct at the start of the stream rather than only after the two
    pointers happen to converge.
    """

    def __init__(self, source_a: str, source_b: str, max_skew_ms: float = 40.0, offset_ms: float = 0.0):
        self.source_a = source_a
        self.source_b = source_b
        self.ts_a = read_timestamps(source_a)
        self.ts_b = read_timestamps(source_b)
        self.pairs = pair_timestamps(self.ts_a, self.ts_b, max_skew_ms, offset_ms)

    def describe(self) -> str:
        return (
            f"  A: {len(self.ts_a)} frames, {measured_fps(self.ts_a):.1f} fps measured\n"
            f"  B: {len(self.ts_b)} frames, {measured_fps(self.ts_b):.1f} fps measured\n"
            f"  paired: {len(self.pairs)} "
            f"(dropped {len(self.ts_a) - len(self.pairs)} unmatched A-frames)"
        )

    def __iter__(self):
        cap_a = cv2.VideoCapture(self.source_a)
        cap_b = cv2.VideoCapture(self.source_b)
        try:
            next_a = next_b = 0
            frame_a = frame_b = None
            for pair in self.pairs:
                while next_a <= pair.index_a:
                    ok, frame_a = cap_a.read()
                    if not ok:
                        return
                    next_a += 1
                while next_b <= pair.index_b:
                    ok, frame_b = cap_b.read()
                    if not ok:
                        return
                    next_b += 1
                yield pair, frame_a, frame_b
        finally:
            cap_a.release()
            cap_b.release()
