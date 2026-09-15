"""
Marks a NAMED reference position (e.g. "home", "drop_point") in the real
world frame by triangulating the crane hook's real position while it's
physically parked there. This is how "home" gets defined for a
vision-only, no-encoder system: it has to be MEASURED through the exact
same pipeline that will later have to recognize it again, not assumed or
hand-typed - there's no encoder to just read a "home" limit switch from.

## Usage

1. Physically move the crane head to the position you want to name (e.g.
   fully home/idle) and hold it stationary there.
2. Run, with both cameras live and pointed at it:
   ```
   py mark_reference_position.py --name home \
       --cam1-calibration calib_out/top_calibration.json --source1 0 \
       --cam2-calibration calib_out/back_calibration.json --source2 1 \
       --checkpoint ../training/output_real/checkpoint_best_ema.pth \
       --out reference_positions.json
   ```
3. It captures --samples confident hook detections (default 30, skipping
   frames where the hook isn't matched in both views), triangulates each,
   and saves the MEDIAN 3D position under --name - median rather than mean
   so a handful of bad/occluded frames during capture can't drag the
   result off, the same reasoning as a real survey measurement throwing
   out outliers rather than averaging them in.
4. Repeat for every named position your flow needs (e.g. also "drop_point"
   if that's not more naturally derived some other way) - each call
   appends to/overwrites just its own key in --out, existing named
   positions are preserved.

track_reference.py loads this file to report real-time distance from the
live tracked position to a named reference - e.g. "has the crane actually
returned home" replacing the simulation's craneRef.position reset to a
hardcoded (0, 5, 0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from calibration import CameraCalibration
from capture_infer import _detect, _load_model, _open_source
from triangulate import epipolar_distance_px, reprojection_error_px, triangulate_point

CLASS_CRANE_HOOK = 1


def _best_hook_pixel(dets: list) -> tuple[float, float] | None:
    hooks = [d for d in dets if d.class_id == CLASS_CRANE_HOOK]
    if not hooks:
        return None
    return max(hooks, key=lambda d: d.confidence).center_px


def capture_samples(
    cam1_cal: CameraCalibration,
    source1: str,
    cam2_cal: CameraCalibration,
    source2: str,
    checkpoint: str,
    resolution: int,
    threshold: float,
    samples: int,
    max_attempts: int,
    reprojection_threshold_px: float,
) -> list[np.ndarray]:
    print(f"loading model from {checkpoint}")
    model = _load_model(checkpoint, resolution)

    cap1, cap2 = _open_source(source1), _open_source(source2)
    if not cap1.isOpened():
        raise RuntimeError(f"could not open source1: {source1}")
    if not cap2.isOpened():
        raise RuntimeError(f"could not open source2: {source2}")

    print(f"capturing {samples} confident crane_hook samples (up to {max_attempts} frame attempts)...")
    points: list[np.ndarray] = []
    attempts = 0
    try:
        while len(points) < samples and attempts < max_attempts:
            ok1, frame1 = cap1.read()
            ok2, frame2 = cap2.read()
            if not ok1 or not ok2:
                print("source ended before enough samples were captured - is the crane hook actually visible?")
                break
            attempts += 1

            px1 = _best_hook_pixel(_detect(model, frame1, threshold))
            px2 = _best_hook_pixel(_detect(model, frame2, threshold))
            if px1 is None or px2 is None:
                continue

            epi = epipolar_distance_px(cam1_cal, px1, cam2_cal, px2)
            if epi > reprojection_threshold_px:
                continue
            err = reprojection_error_px(cam1_cal, px1, cam2_cal, px2)
            if err > reprojection_threshold_px:
                continue

            point = triangulate_point(cam1_cal, px1, cam2_cal, px2)
            points.append(point)
            print(f"  sample {len(points)}/{samples}: ({point[0]:+.3f}, {point[1]:+.3f}, {point[2]:+.3f}) "
                  f"reproj_err={err:.1f}px")
    finally:
        cap1.release()
        cap2.release()

    return points


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help='Name for this reference position, e.g. "home"')
    p.add_argument("--cam1-calibration", required=True)
    p.add_argument("--source1", required=True)
    p.add_argument("--cam2-calibration", required=True)
    p.add_argument("--source2", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--resolution", type=int, default=640)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--max-attempts", type=int, default=300, help="Give up after this many frames if not enough confident samples are found")
    p.add_argument("--reprojection-threshold-px", type=float, default=15.0)
    p.add_argument("--out", required=True, help="Path to the reference-positions JSON (created if missing, existing names preserved)")
    args = p.parse_args()

    cam1_cal = CameraCalibration.load(args.cam1_calibration)
    cam2_cal = CameraCalibration.load(args.cam2_calibration)

    points = capture_samples(
        cam1_cal, args.source1, cam2_cal, args.source2, args.checkpoint, args.resolution, args.threshold,
        args.samples, args.max_attempts, args.reprojection_threshold_px,
    )
    if len(points) < max(3, args.samples // 3):
        raise RuntimeError(
            f"only captured {len(points)}/{args.samples} usable samples - too few to trust. "
            "Check the crane hook is actually in both cameras' view and not occluded."
        )

    median_point = np.median(np.array(points), axis=0)
    spread = np.std(np.array(points), axis=0)
    print(f"\n{args.name}: median=({median_point[0]:+.3f}, {median_point[1]:+.3f}, {median_point[2]:+.3f}) "
          f"std=({spread[0]:.3f}, {spread[1]:.3f}, {spread[2]:.3f}) over {len(points)} samples")
    if float(spread.max()) > 0.05:
        print("WARNING: over 5cm of spread across samples - the hook may not have been fully "
              "stationary, or the calibration/detections are noisier than expected. Consider re-running.")

    out_path = Path(args.out)
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    existing[args.name] = median_point.tolist()
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"saved '{args.name}' to {args.out}")


if __name__ == "__main__":
    main()
