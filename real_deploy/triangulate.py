"""
Turns matched 2D pixel detections from two calibrated cameras into a real
3D world point - this is the piece that replaces BOTH the simulation's
"secretly already knows the exact position" shortcut (backend/app/
live_controller.py's BALE_GRID) and a single top-down camera's inherent
inability to recover height/depth on its own. Two independently calibrated
views of the same real point over-determine its position; you don't need a
tight stereo baseline or synchronized-at-the-hardware-level cameras for
this to work, just two calibrations in the SAME world frame (see
calibration.py's docstring) and a correctly-matched pair of pixels.

Pure geometry - no camera, network, or model code here, which is what
makes it testable without any real hardware (see tests/test_triangulate.py:
sets up synthetic cameras with known ground-truth pose, projects a known
3D point into "pixels" through each, and checks triangulation recovers the
original point).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from calibration import CameraCalibration


def _undistort_to_pixel(cal: CameraCalibration, pixel: tuple[float, float]) -> np.ndarray:
    """Removes lens distortion from a raw detected pixel, returned in the
    SAME (still-pixel, not normalized) scale cv2.triangulatePoints needs to
    match against a K-based projection matrix - see cv2.undistortPoints'
    P= argument, which re-applies K after undistorting into normalized
    coordinates specifically so the output stays in pixel units."""
    pts = np.array([[pixel]], dtype=np.float64)  # shape (1,1,2), what undistortPoints expects
    undistorted = cv2.undistortPoints(pts, cal.camera_matrix, cal.dist_coeffs, P=cal.camera_matrix)
    return undistorted.reshape(2)


def triangulate_point(
    cam1: CameraCalibration, pixel1: tuple[float, float], cam2: CameraCalibration, pixel2: tuple[float, float]
) -> np.ndarray:
    """The 3D world point (x, y, z) that projects to `pixel1` in cam1 and
    `pixel2` in cam2, given both cameras' full calibration (intrinsics +
    extrinsics - see CameraCalibration). Both pixels must be the SAME real
    point seen from each camera (see reprojection_error_px below to sanity
    check a match before trusting it, and epipolar_distance_px in
    capture_infer.py for matching un-correlated detections in the first
    place)."""
    u1 = _undistort_to_pixel(cam1, pixel1)
    u2 = _undistort_to_pixel(cam2, pixel2)
    point_4d = cv2.triangulatePoints(
        cam1.projection_matrix, cam2.projection_matrix, u1.reshape(2, 1), u2.reshape(2, 1)
    )
    point_3d = (point_4d[:3] / point_4d[3]).flatten()
    return point_3d


def reproject_point(cal: CameraCalibration, point_3d: np.ndarray) -> np.ndarray:
    """Where a 3D world point WOULD land in this camera's image, distortion
    included - the forward direction of triangulate_point, used to check
    how well a triangulated point (or a proposed match) actually agrees
    with what a camera saw."""
    rvec, _ = cv2.Rodrigues(cal.rotation)
    projected, _ = cv2.projectPoints(
        point_3d.reshape(1, 1, 3), rvec, cal.translation.reshape(3, 1), cal.camera_matrix, cal.dist_coeffs
    )
    return projected.reshape(2)


def reprojection_error_px(
    cam1: CameraCalibration, pixel1: tuple[float, float], cam2: CameraCalibration, pixel2: tuple[float, float]
) -> float:
    """Triangulates then reprojects into both cameras, returning the worse
    of the two pixel errors - a large value means either the calibration is
    off or (more likely day-to-day) pixel1/pixel2 aren't actually the same
    real point (a bad cross-camera match - see capture_infer.py). Use this
    to reject bad matches rather than silently trusting every triangulation."""
    point_3d = triangulate_point(cam1, pixel1, cam2, pixel2)
    err1 = np.linalg.norm(reproject_point(cam1, point_3d) - np.array(pixel1))
    err2 = np.linalg.norm(reproject_point(cam2, point_3d) - np.array(pixel2))
    return float(max(err1, err2))


def fundamental_matrix(cam1: CameraCalibration, cam2: CameraCalibration) -> np.ndarray:
    """F such that pixel2^T @ F @ pixel1 == 0 for a genuine match (in
    undistorted-pixel coordinates) - derived from both cameras' calibrated
    pose, not estimated from point correspondences (the usual textbook use)
    since we already have real calibration. Used by
    capture_infer.py's epipolar matching to score candidate cross-camera
    detection pairs before ever triangulating them."""
    # Relative pose of cam2 w.r.t. cam1: X_cam2 = R_rel @ X_cam1 + t_rel
    r_rel = cam2.rotation @ cam1.rotation.T
    t_rel = cam2.translation - r_rel @ cam1.translation
    t_cross = np.array(
        [
            [0.0, -t_rel[2], t_rel[1]],
            [t_rel[2], 0.0, -t_rel[0]],
            [-t_rel[1], t_rel[0], 0.0],
        ]
    )
    essential = t_cross @ r_rel
    k1_inv = np.linalg.inv(cam1.camera_matrix)
    k2_inv_t = np.linalg.inv(cam2.camera_matrix).T
    return k2_inv_t @ essential @ k1_inv


def epipolar_distance_px(
    cam1: CameraCalibration, pixel1: tuple[float, float], cam2: CameraCalibration, pixel2: tuple[float, float]
) -> float:
    """Perpendicular distance, in cam2's pixels, from pixel2 to the
    epipolar line that a genuine match to pixel1 must lie on. Cheap
    (no triangulation) first-pass filter for candidate cross-camera
    matches before the costlier reprojection_error_px check."""
    f = fundamental_matrix(cam1, cam2)
    p1_h = np.array([pixel1[0], pixel1[1], 1.0])
    line = f @ p1_h  # [a, b, c] of the line a*u + b*v + c = 0 in cam2's image
    denom = math.hypot(line[0], line[1])
    if denom < 1e-9:
        return float("inf")
    p2_h = np.array([pixel2[0], pixel2[1], 1.0])
    return float(abs(line @ p2_h) / denom)
