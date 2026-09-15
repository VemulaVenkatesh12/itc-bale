# Real-hardware deployment (Milestone 1: vision only, no movement)

This directory is the real-world counterpart to the simulation in
`backend/`/`frontend/` - it does NOT touch or replace any simulation code,
and it does not talk to any relay, PLC, or actuator. It exists to answer
one question honestly: **can this vision pipeline produce trustworthy real
3D positions from two real cameras**, before anything is ever allowed to
act on those positions and move real machinery.

## Why this exists (read this before treating any of it as "done")

The simulation's crane controller (`backend/app/live_controller.py`) got
its approach targeting to reliably converge by homing on each bale's
*known, exact position* from the simulation's own data (`BALE_GRID`) - see
project memory `itc-crane-full-batch-completion-fix`. That only works
because the simulation secretly already knows the truth. **A real system
has no such ground truth - only whatever these two cameras and this model
can recover**, and recovering it reliably (despite real detection noise,
with no position encoders - see below) is a genuinely different, harder
problem than anything fixed in the simulation. This directory is where
that problem actually gets solved, not where the simulation's shortcut
gets copy-pasted onto real hardware.

Two decisions already made (2026-08-16, discussed directly): actuation
will go through a **Modbus PLC**, and there are **no plans for axis
encoders** - position feedback is vision-only. That second one is the hard
constraint that shapes everything here: a single top-down camera cannot
recover height/depth without a known-height assumption, so real 3D
position requires **two calibrated cameras triangulated together** (not
"one drives control, one is just for display" like the simulation's
Camera_Top/Camera_Back split - see `triangulate.py`'s docstring).

## Pipeline

```
calibrate_intrinsics.py     -> per-camera lens calibration (run once per camera)
calibrate_extrinsics.py     -> per-camera real-world pose (run once per camera, after intrinsics)
                                both write CameraCalibration JSON (see calibration.py)
frame_sync.py               -> pairs frames from the two cameras by TIMESTAMP, not by loop
                                iteration - required because the two cameras do not run at
                                the same rate (see "Frame rate" below)
capture_infer.py            -> two-camera capture + real-model inference + cross-camera
                                matching + triangulation -> live 3D positions, PRINTED ONLY
mark_reference_position.py  -> physically park the crane head somewhere (e.g. "home"),
                                triangulate + median-average N samples, save it by name
track_reference.py          -> live crane_hook position (SMOOTHED via position_tracker.py,
                                not raw per-frame) + distance/arrived-or-not relative to a
                                named reference - e.g. "has it actually returned home"
planner3d.py                -> real metric 3D pick planning -> SF-8DR relay timeline. The
                                real-world counterpart to backend/app/planner.py, which
                                plans in image pixels
run_pipeline.py             -> END-TO-END DRY RUN tying all of the above together:
                                two sources -> synced frames -> detect -> match ->
                                triangulate -> survey/cluster -> plan -> relay timeline
                                JSON. Actuates NOTHING.

extract_training_frames.py  -> cuts a labeling-ready frame set from recorded footage
autolabel_frames.py         -> draft COCO annotations from an existing checkpoint
```

`example_plan.json` is a committed sample of what `planner3d.py` emits for a
3x3 truck stack - useful for seeing the output shape without running the
whole chain.

`mark_reference_position.py` + `track_reference.py` are the direct answer
to "use vision to decide where the crane head is, and mark a designated
home position to start/end the flow at": the crane's position now comes
from a real triangulated (and smoothed - see `position_tracker.py`)
measurement, and "home" is a real point measured through the same
pipeline, not a hardcoded constant like the simulation's craneRef.position
reset to `(0, 5, 0)`. Still print-only - nothing here fires a relay yet.

**Why smoothing (`position_tracker.py`), not raw per-frame trust:** this is
the exact mechanism that fixed the simulation's crane-hunting bug
(`itc-crane-vision-position-hunting-bug`/`itc-crane-full-batch-completion-
fix`) - trusting a single noisy detection to decide "have I arrived" causes
oscillation. A real system has no ground-truth fallback the way the
simulation did, so real noise has to be handled with real filtering (a
simple exponential moving average here, with an explicit staleness signal
so an old position never gets silently treated as current) rather than
either extreme.

Plus `DATA_COLLECTION.md` (how to capture and label real bale/hook footage
so it plugs into the existing `training/finetune.py`) and two test files -
run these first, and after any change to the files they cover:

```
py tests/test_triangulate.py        # triangulate.py, calibration.py - 6/6 passing
py tests/test_position_tracker.py   # position_tracker.py - 9/9 passing
py tests/test_frame_sync.py         # frame_sync.py - 9/9 passing
py tests/test_planner3d.py          # planner3d.py - 16/16 passing
```

## Frame rate: the two cameras do NOT run at the same rate

Measured from the 2026-08-18 recordings, not from container metadata:

| Camera | Resolution | Container claims | ACTUAL |
|---|---|---|---|
| .101 | 2560x1440 | 20 fps | 20.0 fps (clean, 3 dropped frames) |
| .102 | 1920x1080 | 25 fps | **12.5 fps** (+247 further dropped frames) |

Reading one frame from each source per loop iteration - what
`capture_infer.py` does - therefore drifts the two streams apart by a full
second every ~1.7s of footage, reaching **93 seconds apart** by the end of a
258s clip. Triangulating that pair yields a confident, precise, completely
wrong 3D point. `frame_sync.py` pairs by timestamp instead and DROPS pairs
that cannot be matched within tolerance, because a dropped frame is a
visible gap while a badly-paired frame is an invisible wrong answer.

The cameras' clocks also disagree (the two filenames differ by 4.3s; .102's
burned-in OSD is ~7-8s off its own filename start), so `--offset-ms` exists
to cancel a *measured* constant skew. Measure it against an event visible in
both views; do not guess it.

The geometry and the filtering/staleness/arrival logic are both verified
correct against synthetic ground truth - that's what's actually proven
right now, independent of any real camera. `capture_infer.py` and
`mark_reference_position.py` have also been smoke-tested end-to-end (model
loads, detects, matches, triangulates/captures, exits cleanly) **against a
single existing video fed in as both "cameras," which is a structural
check only** - no camera hardware was available this session:
- `capture_infer.py`'s smoke test printed positions at literally planetary
  distances - two views of the exact same footage have zero real
  parallax, a degenerate case for triangulation, not a bug.
- `mark_reference_position.py`'s smoke test correctly refused to save a
  reference position at all (0/3 usable samples) because that test
  footage has no crane hook in frame - the right behavior, not a failure.

Neither smoke test proves anything about real-world accuracy; the unit
tests are what actually validate the math and logic.

## Getting to a real Milestone 1

1. Mount both real cameras where they'll actually live.
2. `calibrate_intrinsics.py` for each (checkerboard photos, per-camera lens
   properties).
3. `calibrate_extrinsics.py` for each (measured real-world reference
   points, click their pixel locations) - **this is the step that actually
   defines your shared world frame**, do it carefully; a sloppy extrinsic
   calibration is the most likely source of bad 3D positions later, not a
   bug in the triangulation code.
4. Collect and label real footage - `DATA_COLLECTION.md`.
5. Fine-tune on it - reuses `training/finetune.py`.
6. `capture_infer.py` with the real calibrations, real cameras (or
   recorded real video first), and the real-data checkpoint. Watch the
   printed positions against something you can independently measure (a
   bale at a known spot) before trusting it for anything.
7. `mark_reference_position.py --name home` with the crane physically
   parked home (and any other named waypoints your flow needs - e.g. a
   drop point, if it's not more naturally derived some other way).
8. `track_reference.py --name home` while moving the crane around, then
   parking it back home - confirm the reported distance actually drops to
   ~0 and `ARRIVED` prints when it should, and that a real move away from
   home is reflected promptly (tune `--alpha`/`--tolerance-m` against what
   you observe, per this script's own docstring, before trusting either).

## What's explicitly NOT here yet

- **No Modbus/PLC driver.** Given the decision to use Modbus, that's the
  natural next piece after Milestone 1 is validated - but it needs your
  actual PLC's register map / addressing scheme to write correctly, which
  I don't have. Bring the PLC's documentation (or the vendor/model) when
  you're ready for this step.
- **No closed-loop control - the ACTING half specifically.** `planner3d.py`
  and `run_pipeline.py` now produce a full relay timeline from real 3D
  positions, but it is still an OPEN-LOOP plan: durations computed from
  distances up front, exactly like the simulation. Nothing feeds
  `position_tracker.py`'s smoothed position back to decide when to actually
  CUT a relay. `track_reference.py` remains the observing half and stops
  there on purpose. Turning "arrived: False, 0.8m short" into a live relay
  command needs a control law that accounts for real-world timing/latency
  (this is not the simulation's tick-rate-aware setpoint scheme, which
  assumed a known, constant crane speed) - a separate design conversation
  once the observing half is proven reliable on real hardware.

- **No measured speeds.** Every value in `planner3d.SpeedConfig3D` is a
  placeholder. Time the real crane over a known distance at each speed
  before any emitted timeline means anything in seconds - the plan's
  structure is correct, its durations are currently fiction.
- **No safety system.** e-stop, interlocks, exclusion zones - required
  before any of this is allowed to move real machinery. Not something to
  design in a coding session; get a controls/safety engineer to review the
  actual mechanical/electrical installation before Milestone 2 ("full pick
  end-to-end") is even attempted.
