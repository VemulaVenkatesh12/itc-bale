"""
Verifies triangulate.py's geometry is actually correct, using synthetic
cameras with KNOWN ground-truth pose - no real camera or PLC needed to run
this. This is the test that should be trusted BEFORE ever pointing this
code at real footage: if triangulation can't recover a point it was told
the exact camera geometry for, it certainly won't work with real noisy
detections and an imperfectly-surveyed real calibration.

Run: py -m pytest real_deploy/tests/test_triangulate.py -v
 (or just: py real_deploy/tests/test_triangulate.py)
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration import CameraCalibration
from triangulate import (
    epipolar_distance_px,
    reprojection_error_px,
    triangulate_point,
)

IMAGE_W, IMAGE_H = 640, 480
K = np.array([[800.0, 0.0, IMAGE_W / 2], [0.0, 800.0, IMAGE_H / 2], [0.0, 0.0, 1.0]])


def _project(cal: CameraCalibration, point_3d: np.ndarray) -> tuple[float, float]:
    """Ground-truth forward projection (independent of triangulate.py's own
    reproject_point, so this test doesn't just check the module agrees with
    itself) - what a real camera with this exact calibration would actually
    see a point at, distortion included."""
    rvec, _ = cv2.Rodrigues(cal.rotation)
    px, _ = cv2.projectPoints(point_3d.reshape(1, 1, 3), rvec, cal.translation, cal.camera_matrix, cal.dist_coeffs)
    return tuple(px.reshape(2))


def _two_cameras(baseline_m: float = 1.0, with_distortion: bool = False) -> tuple[CameraCalibration, CameraCalibration]:
    """cam1 at world origin looking down +Z; cam2 offset `baseline_m` along
    world X, same orientation - a simple but non-degenerate two-view rig,
    analogous to two real cameras surveyed into a shared world frame
    (see calibration.py's docstring for the convention)."""
    dist = np.array([-0.08, 0.01, 0.0, 0.0, 0.0]) if with_distortion else np.zeros(5)
    cam1 = CameraCalibration(
        name="cam1", image_width=IMAGE_W, image_height=IMAGE_H,
        camera_matrix=K, dist_coeffs=dist, rotation=np.eye(3), translation=np.zeros(3),
    )
    # X_cam2 = R @ X_world + t, R=I here, so cam2's own world position is at
    # -t = (baseline_m, 0, 0): camera center = -R^T @ t (see
    # CameraCalibration.camera_center_world).
    cam2 = CameraCalibration(
        name="cam2", image_width=IMAGE_W, image_height=IMAGE_H,
        camera_matrix=K, dist_coeffs=dist, rotation=np.eye(3), translation=np.array([-baseline_m, 0.0, 0.0]),
    )
    return cam1, cam2


def test_recovers_known_point_no_distortion():
    cam1, cam2 = _two_cameras(with_distortion=False)
    true_point = np.array([0.3, 0.2, 5.0])
    px1 = _project(cam1, true_point)
    px2 = _project(cam2, true_point)

    recovered = triangulate_point(cam1, px1, cam2, px2)

    assert np.allclose(recovered, true_point, atol=1e-6), f"expected {true_point}, got {recovered}"


def test_recovers_known_point_with_lens_distortion():
    # The real cameras this eventually points at WILL have lens distortion;
    # if undistortion in triangulate.py were wrong or missing, this is the
    # test that would catch it (the no-distortion test above would still
    # pass even with a broken undistort step, since undistorting a point
    # from a zero-distortion camera is a no-op).
    cam1, cam2 = _two_cameras(with_distortion=True)
    true_point = np.array([-0.15, 0.4, 4.2])
    px1 = _project(cam1, true_point)
    px2 = _project(cam2, true_point)

    recovered = triangulate_point(cam1, px1, cam2, px2)

    assert np.allclose(recovered, true_point, atol=1e-4), f"expected {true_point}, got {recovered}"


def test_reprojection_error_near_zero_for_a_genuine_match():
    cam1, cam2 = _two_cameras()
    true_point = np.array([0.1, -0.3, 6.0])
    px1 = _project(cam1, true_point)
    px2 = _project(cam2, true_point)

    err = reprojection_error_px(cam1, px1, cam2, px2)

    assert err < 0.01, f"expected ~0px reprojection error for a real match, got {err}"


def test_reprojection_error_large_for_a_bad_match():
    # pixel2 here comes from a DIFFERENT 3D point than pixel1 - simulates
    # accidentally pairing detections of two different bales across
    # cameras. This is the check capture_infer.py should use to refuse a
    # triangulation rather than silently reporting a nonsense 3D position.
    cam1, cam2 = _two_cameras()
    px1 = _project(cam1, np.array([0.1, -0.3, 6.0]))
    px2_wrong = _project(cam2, np.array([-0.8, 0.5, 3.0]))

    err = reprojection_error_px(cam1, px1, cam2, px2_wrong)

    assert err > 5.0, f"expected a large reprojection error for a mismatched pair, got {err}"


def test_epipolar_distance_zero_for_a_genuine_match():
    cam1, cam2 = _two_cameras()
    true_point = np.array([0.5, 0.1, 3.5])
    px1 = _project(cam1, true_point)
    px2 = _project(cam2, true_point)

    dist = epipolar_distance_px(cam1, px1, cam2, px2)

    assert dist < 0.01, f"expected ~0px epipolar distance for a real match, got {dist}"


def test_epipolar_distance_large_for_an_unrelated_point():
    cam1, cam2 = _two_cameras()
    px1 = _project(cam1, np.array([0.5, 0.1, 3.5]))
    px2_wrong = _project(cam2, np.array([-0.5, 0.6, 8.0]))

    dist = epipolar_distance_px(cam1, px1, cam2, px2_wrong)

    assert dist > 5.0, f"expected a large epipolar distance for an unrelated point, got {dist}"


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
