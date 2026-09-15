"""
Per-camera extrinsic calibration (where the camera actually is and which
way it's pointing, in a world frame YOU define and share across both
cameras) - run once per camera AFTER calibrate_intrinsics.py, and again any
time a camera gets physically bumped/remounted.

## Define your world frame once, use it for both cameras

Pick an origin and axes on the physical setup and write them down - e.g.
"origin at the truck bed's near-left corner post, X along the bed's long
axis toward the far end, Y across the bed width, Z straight up". It doesn't
matter what you pick as long as it's the SAME frame for every point you
measure and for both cameras' calibrations - that shared frame is the
entire point of calibrating two cameras at all (see triangulate.py's
docstring).

## Measure real-world reference points

Pick 6+ points you can both physically measure (tape measure / laser
measure from your chosen origin) AND see clearly in the camera's image -
e.g. corners of the truck bed frame, marks taped to the floor, corners of a
fixture. Prefer points that are NOT all on one flat plane (mix some at bed
height with some at floor height, say) - a fully planar point set makes
solvePnP's depth estimate along the camera's viewing axis less reliable.

Write them to a JSON file, e.g. `real_deploy/calib_points/top_points.json`:
```json
[
  {"label": "bed_near_left",  "world": [0.0, 0.0, 1.0]},
  {"label": "bed_near_right", "world": [0.0, 2.2, 1.0]},
  {"label": "bed_far_left",   "world": [5.5, 0.0, 1.0]},
  {"label": "bed_far_right",  "world": [5.5, 2.2, 1.0]},
  {"label": "floor_left",     "world": [0.0, 0.0, 0.0]},
  {"label": "floor_right",    "world": [0.0, 2.2, 0.0]}
]
```

## Mark where each point is in the camera's image

```
py calibrate_extrinsics.py --intrinsics calib_out/top_intrinsics.json \
    --points calib_points/top_points.json --image calib_images/top_ref.jpg \
    --click --out calib_out/top_calibration.json
```
`--click` opens the image in a window; click each labeled point IN THE
ORDER PRINTED (same order as the JSON file) - the label is printed to the
console before each click so you know which one you're marking. Press 'u'
to undo the last click, 'q' once all points are marked (or after clicking
all of them, it closes automatically).

Without `--click`, pixel coordinates must already be filled into the
points JSON (`"pixel": [u, v]` per entry) - useful for scripting this or
reusing points you marked previously.

Writes the FULL calibration (intrinsics + this camera's now-known pose) to
`--out`, overwriting the intrinsics-only file - this is what triangulate.py
and capture_infer.py actually load.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibration import CameraCalibration


def _click_points(image_path: str, labels: list[str]) -> list[tuple[float, float]]:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"could not read image: {image_path}")
    clicked: list[tuple[float, float]] = []
    window = "calibrate_extrinsics - click each point in order, 'u' to undo, 'q' to finish"

    def _redraw():
        display = img.copy()
        for i, (x, y) in enumerate(clicked):
            cv2.circle(display, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(display, labels[i], (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if len(clicked) < len(labels):
            cv2.setWindowTitle(window, f"{window} - next: {labels[len(clicked)]}")
        cv2.imshow(window, display)

    def _on_click(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < len(labels):
            clicked.append((float(x), float(y)))
            print(f"  {labels[len(clicked) - 1]} -> ({x}, {y})")
            _redraw()

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _on_click)
    print(f"click: {labels[0]}")
    _redraw()
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("u") and clicked:
            removed = clicked.pop()
            print(f"  undid {labels[len(clicked)]} (was {removed})")
            _redraw()
        elif key == ord("q") or len(clicked) == len(labels):
            break
    cv2.destroyWindow(window)
    if len(clicked) != len(labels):
        raise ValueError(f"only clicked {len(clicked)}/{len(labels)} points - re-run and click all of them")
    return clicked


def calibrate_extrinsics(
    intrinsics: CameraCalibration, world_points: np.ndarray, pixel_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (rotation 3x3, translation (3,), mean reprojection error px)."""
    if len(world_points) < 4:
        raise ValueError(f"solvePnP needs at least 4 points, got {len(world_points)}")

    ok, rvec, tvec = cv2.solvePnP(
        world_points.astype(np.float64),
        pixel_points.astype(np.float64),
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed to converge - check the world/pixel correspondences are correct")

    rotation, _ = cv2.Rodrigues(rvec)
    translation = tvec.flatten()

    reprojected, _ = cv2.projectPoints(
        world_points, rvec, tvec, intrinsics.camera_matrix, intrinsics.dist_coeffs
    )
    errors = np.linalg.norm(reprojected.reshape(-1, 2) - pixel_points, axis=1)
    return rotation, translation, float(errors.mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--intrinsics", required=True, help="Path to the intrinsics-only JSON from calibrate_intrinsics.py")
    p.add_argument("--points", required=True, help="Path to the world-points JSON (see this script's docstring)")
    p.add_argument("--image", required=True, help="A photo from this camera showing all the reference points")
    p.add_argument("--click", action="store_true", help="Interactively click each point's pixel location in --image")
    p.add_argument("--out", required=True, help="Output path for the full calibration JSON")
    args = p.parse_args()

    intrinsics = CameraCalibration.load(args.intrinsics)
    points_data = json.loads(Path(args.points).read_text())
    labels = [pt["label"] for pt in points_data]
    world_points = np.array([pt["world"] for pt in points_data], dtype=np.float64)

    if args.click:
        pixel_points = np.array(_click_points(args.image, labels), dtype=np.float64)
    else:
        missing = [pt["label"] for pt in points_data if "pixel" not in pt]
        if missing:
            raise ValueError(f"--points is missing \"pixel\" for: {missing} - pass --click or fill them in by hand")
        pixel_points = np.array([pt["pixel"] for pt in points_data], dtype=np.float64)

    rotation, translation, reproj_err = calibrate_extrinsics(intrinsics, world_points, pixel_points)

    print(f"\nmean reprojection error: {reproj_err:.2f}px "
          f"({'good' if reproj_err < 3 else 'usable' if reproj_err < 8 else 'CHECK YOUR MEASUREMENTS - too high'})")

    cal = CameraCalibration(
        name=intrinsics.name,
        image_width=intrinsics.image_width,
        image_height=intrinsics.image_height,
        camera_matrix=intrinsics.camera_matrix,
        dist_coeffs=intrinsics.dist_coeffs,
        rotation=rotation,
        translation=translation,
    )
    print(f"camera center in world coordinates: {cal.camera_center_world} "
          "(sanity-check this against a tape measure)")
    cal.save(args.out)
    print(f"\nsaved full calibration to {args.out}")


if __name__ == "__main__":
    main()
