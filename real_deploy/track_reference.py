"""
Live crane_hook position tracking relative to a named reference position
(see mark_reference_position.py) - e.g. "has the crane actually arrived
home", to start/end a real pick-and-place flow on a REAL measurement
instead of the simulation's craneRef.position reset to a hardcoded
(0, 5, 0). Prints only - still no relay/actuator output; this is the
observation half of closed-loop control, proven correct on its own before
anything is allowed to act on it.

## Usage

```
py track_reference.py --name home \
    --cam1-calibration calib_out/top_calibration.json --source1 0 \
    --cam2-calibration calib_out/back_calibration.json --source2 1 \
    --checkpoint ../training/output_real/checkpoint_best_ema.pth \
    --reference-positions reference_positions.json \
    --tolerance-m 0.05
```

## Setting --tolerance-m and --alpha for real use

Don't leave these at the defaults for anything that matters - see
position_tracker.py's has_arrived()/PositionTracker docstrings. As a
starting point: --tolerance-m should sit comfortably above both your
calibration's reprojection error (printed by calibrate_extrinsics.py, in
pixels - convert to real-world meters using roughly (reprojection_error_px
/ focal_length_px) * distance_to_target_m) and the smoothed tracker's own
residual noise at your real detection rate (mark_reference_position.py's
printed std dev across samples is a reasonable proxy for this).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from calibration import CameraCalibration
from capture_infer import _detect, _load_model, _open_source
from mark_reference_position import _best_hook_pixel
from position_tracker import PositionTracker, distance_m, has_arrived
from triangulate import epipolar_distance_px, reprojection_error_px, triangulate_point

REPROJECTION_ACCEPT_THRESHOLD_PX = 15.0


def run(
    cam1_cal: CameraCalibration,
    source1: str,
    cam2_cal: CameraCalibration,
    source2: str,
    checkpoint: str,
    resolution: int,
    threshold: float,
    reference_point,
    reference_name: str,
    tolerance_m: float,
    alpha: float,
    stale_after_s: float,
    max_frames: int | None,
) -> None:
    print(f"loading model from {checkpoint}")
    model = _load_model(checkpoint, resolution)
    cap1, cap2 = _open_source(source1), _open_source(source2)
    if not cap1.isOpened():
        raise RuntimeError(f"could not open source1: {source1}")
    if not cap2.isOpened():
        raise RuntimeError(f"could not open source2: {source2}")

    tracker = PositionTracker(alpha=alpha, stale_after_s=stale_after_s)
    print(f"tracking crane_hook relative to '{reference_name}' = {reference_point} "
          f"(tolerance={tolerance_m}m) - Ctrl+C to stop\n")

    frame_i = 0
    try:
        while True:
            ok1, frame1 = cap1.read()
            ok2, frame2 = cap2.read()
            if not ok1 or not ok2:
                print("end of stream (or a camera dropped) - stopping")
                break
            frame_i += 1
            if max_frames is not None and frame_i > max_frames:
                print(f"reached --max-frames ({max_frames}) - stopping")
                break

            px1 = _best_hook_pixel(_detect(model, frame1, threshold))
            px2 = _best_hook_pixel(_detect(model, frame2, threshold))

            measurement, confidence = None, 0.0
            if px1 is not None and px2 is not None:
                epi = epipolar_distance_px(cam1_cal, px1, cam2_cal, px2)
                err = reprojection_error_px(cam1_cal, px1, cam2_cal, px2)
                if epi <= REPROJECTION_ACCEPT_THRESHOLD_PX and err <= REPROJECTION_ACCEPT_THRESHOLD_PX:
                    measurement = triangulate_point(cam1_cal, px1, cam2_cal, px2)
                    confidence = 1.0  # class Detection2D doesn't carry a combined score here; refine if needed

            est = tracker.update(measurement, confidence, now_s=time.monotonic())

            if est.position is None:
                print(f"[frame {frame_i}] hook never observed yet")
            elif est.stale:
                print(f"[frame {frame_i}] STALE (last seen {est.age_s:.1f}s ago) - last known "
                      f"({est.position[0]:+.3f}, {est.position[1]:+.3f}, {est.position[2]:+.3f})")
            else:
                dist = distance_m(est.position, reference_point)
                arrived = has_arrived(est.position, reference_point, tolerance_m)
                print(f"[frame {frame_i}] hook=({est.position[0]:+.3f}, {est.position[1]:+.3f}, "
                      f"{est.position[2]:+.3f}) dist_to_{reference_name}={dist:.3f}m "
                      f"{'ARRIVED' if arrived else ''}")
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        cap1.release()
        cap2.release()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help='Which named reference to track against, e.g. "home"')
    p.add_argument("--reference-positions", required=True, help="Path to the JSON written by mark_reference_position.py")
    p.add_argument("--cam1-calibration", required=True)
    p.add_argument("--source1", required=True)
    p.add_argument("--cam2-calibration", required=True)
    p.add_argument("--source2", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--resolution", type=int, default=640)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tolerance-m", type=float, default=0.05, help="See this script's docstring before trusting the default")
    p.add_argument("--alpha", type=float, default=0.3, help="PositionTracker smoothing factor - see position_tracker.py")
    p.add_argument("--stale-after-s", type=float, default=1.0)
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()

    references = json.loads(Path(args.reference_positions).read_text())
    if args.name not in references:
        raise ValueError(f"'{args.name}' not found in {args.reference_positions} - "
                          f"known names: {list(references.keys())}. Run mark_reference_position.py first.")
    reference_point = references[args.name]

    cam1_cal = CameraCalibration.load(args.cam1_calibration)
    cam2_cal = CameraCalibration.load(args.cam2_calibration)

    run(
        cam1_cal, args.source1, cam2_cal, args.source2, args.checkpoint, args.resolution, args.threshold,
        reference_point, args.name, args.tolerance_m, args.alpha, args.stale_after_s, args.max_frames,
    )


if __name__ == "__main__":
    main()
