"""
Per-camera intrinsic calibration (focal length, principal point, lens
distortion) from a folder of checkerboard photos - standard OpenCV
chessboard calibration, run ONCE per physical camera (re-run only if the
lens/focus/zoom changes; does not depend on where the camera is mounted -
that's calibrate_extrinsics.py, run second).

## How to capture the checkerboard photos

1. Print a checkerboard pattern (e.g. 9x6 internal corners, any square
   size - a standard letter/A4-page pattern works fine; OpenCV's own docs
   have a printable one). Mount it on something rigid and FLAT (a clipboard
   or foam board - a checkerboard that can flex gives bad calibrations).
2. With the real camera you're calibrating, take 15-25 photos of the board
   at different positions, angles, and distances, covering the corners and
   edges of the frame (not just the center) - tilt it, move it close and
   far, into each corner of the image. Save them into one folder, e.g.
   `real_deploy/calib_images/top/*.jpg`.
3. Run:
   ```
   py calibrate_intrinsics.py --name top --images "calib_images/top/*.jpg" \
       --board-cols 9 --board-rows 6 --square-size-m 0.025 \
       --out calib_out/top_intrinsics.json
   ```
   `--square-size-m` only matters if you want the REPROJECTION ERROR
   reported in real units - the intrinsics themselves (K, distortion) don't
   depend on it, but get it right anyway, it's free.
4. Check the printed reprojection error: under ~0.5px is good, under 1px is
   usable, more than that means retake photos (usually: not enough
   coverage of the frame edges, or a checkerboard that wasn't flat).

Produces a CameraCalibration JSON with a placeholder identity
rotation/zero translation (see CameraCalibration.intrinsics_only) -
calibrate_extrinsics.py loads this file and fills in the real pose.
"""

from __future__ import annotations

import argparse
import glob

import cv2
import numpy as np

from calibration import CameraCalibration


def calibrate_intrinsics(
    image_paths: list[str], board_cols: int, board_rows: int, square_size_m: float
) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    """Returns (camera_matrix, dist_coeffs, image_width, image_height, mean_reprojection_error_px)."""
    if not image_paths:
        raise ValueError("no calibration images found - check --images matches your file glob")

    board_size = (board_cols, board_rows)
    # Real-world coordinates of the checkerboard's internal corners, in the
    # board's own flat plane (Z=0) - the SAME for every photo, since it's
    # the same physical board; only its pose relative to the camera differs
    # shot to shot, which is exactly what solvePnP-under-the-hood in
    # calibrateCamera solves for per image.
    object_points_template = np.zeros((board_cols * board_rows, 3), dtype=np.float32)
    object_points_template[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2) * square_size_m

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    found_count = 0

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  skip (unreadable): {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(gray, board_size)
        if not found:
            print(f"  no checkerboard found: {path}")
            continue

        # Sub-pixel corner refinement - the raw findChessboardCorners
        # result is only accurate to the nearest pixel, which is not
        # precise enough for a good calibration.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        object_points.append(object_points_template)
        image_points.append(corners)
        found_count += 1
        print(f"  ok: {path}")

    if found_count < 5:
        raise ValueError(
            f"only found the checkerboard in {found_count}/{len(image_paths)} images - "
            "need at least 5-10 good detections for a usable calibration (15-25 recommended)"
        )
    assert image_size is not None

    reprojection_error, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    return camera_matrix, dist_coeffs.flatten(), image_size[0], image_size[1], reprojection_error


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help='Camera name, e.g. "top" or "back" - stored in the output JSON')
    p.add_argument("--images", required=True, help='Glob pattern for checkerboard photos, e.g. "calib_images/top/*.jpg"')
    p.add_argument("--board-cols", type=int, default=9, help="Internal corner count along the board's longer side")
    p.add_argument("--board-rows", type=int, default=6, help="Internal corner count along the board's shorter side")
    p.add_argument("--square-size-m", type=float, default=0.025, help="Physical size of one checkerboard square, in meters")
    p.add_argument("--out", required=True, help="Output path for the calibration JSON")
    args = p.parse_args()

    image_paths = sorted(glob.glob(args.images))
    print(f"found {len(image_paths)} candidate images")

    camera_matrix, dist_coeffs, width, height, reproj_err = calibrate_intrinsics(
        image_paths, args.board_cols, args.board_rows, args.square_size_m
    )

    print(f"\nimage size: {width}x{height}")
    print(f"mean reprojection error: {reproj_err:.3f}px "
          f"({'good' if reproj_err < 0.5 else 'usable' if reproj_err < 1.0 else 'RETAKE PHOTOS - too high'})")
    print(f"camera_matrix:\n{camera_matrix}")
    print(f"dist_coeffs: {dist_coeffs}")

    cal = CameraCalibration.intrinsics_only(args.name, width, height, camera_matrix, dist_coeffs)
    cal.save(args.out)
    print(f"\nsaved to {args.out} - run calibrate_extrinsics.py next to fill in this camera's real-world pose")


if __name__ == "__main__":
    main()
