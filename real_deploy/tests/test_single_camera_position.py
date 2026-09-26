"""
Verifies single_camera_position.py's ray/known-height-plane geometry is
correct, using a synthetic camera with KNOWN ground-truth pose - same
approach as test_triangulate.py. Also checks the failure mode: giving the
WRONG assumed height produces a WRONG but predictable X/Y error, which is
the whole point of this module's docstring warning (accuracy is only as
good as the height assumption).

Run: py -m pytest real_deploy/tests/test_single_camera_position.py -v
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration import CameraCalibration
from single_camera_position import estimate_position_at_height

IMAGE_W, IMAGE_H = 1280, 720
K = np.array([[900.0, 0.0, IMAGE_W / 2], [0.0, 900.0, IMAGE_H / 2], [0.0, 0.0, 1.0]])


def _project(cal: CameraCalibration, point_3d: np.ndarray) -> tuple[float, float]:
    rvec, _ = cv2.Rodrigues(cal.rotation)
    px, _ = cv2.projectPoints(point_3d.reshape(1, 1, 3), rvec, cal.translation, cal.camera_matrix, cal.dist_coeffs)
    return tuple(px.reshape(2))


def _overhead_camera(tilt_deg: float = 35.0, height_m: float = 6.0) -> CameraCalibration:
    """A cam101-like rig: mounted high, tilted down at the scene (not
    straight overhead - matches the real gantry mount, and a pure
    straight-down camera would be a degenerate edge case for this test,
    not a realistic one)."""
    # Rotate about the world X axis so the camera's +Z (forward) axis tips
    # downward from horizontal by `tilt_deg` - rvec angle 90 degrees gives a
    # purely horizontal forward (0,1,0); adding tilt_deg tips it down into
    # -Z, verified numerically (90->[0,1,0], 135->[0,.707,-.707], etc).
    rvec = np.array([np.deg2rad(90.0 + tilt_deg), 0.0, 0.0])
    rotation, _ = cv2.Rodrigues(rvec)
    camera_center_world = np.array([0.0, -3.0, height_m])
    translation = -rotation @ camera_center_world  # t = -R @ C
    return CameraCalibration(
        name="cam101_synthetic", image_width=IMAGE_W, image_height=IMAGE_H,
        camera_matrix=K, dist_coeffs=np.zeros(5),
        rotation=rotation, translation=translation,
    )


def test_recovers_point_at_correct_height():
    cal = _overhead_camera()
    truth = np.array([1.2, 2.4, 1.0])  # e.g. a bale-top height of 1.0m
    pixel = _project(cal, truth)

    recovered = estimate_position_at_height(cal, pixel, assumed_z_m=1.0)

    assert np.allclose(recovered, truth, atol=1e-6), f"expected {truth}, got {recovered}"


def test_wrong_height_assumption_produces_bounded_xy_error():
    cal = _overhead_camera()
    truth = np.array([1.2, 2.4, 1.0])
    pixel = _project(cal, truth)

    # Assume the bale is 0.3m taller than it really is (e.g. wrong layer).
    recovered = estimate_position_at_height(cal, pixel, assumed_z_m=1.3)

    xy_error = np.linalg.norm(recovered[:2] - truth[:2])
    # Should be wrong (this is the whole point of the module's warning)...
    assert xy_error > 0.05
    # ...but not wildly wrong for a 0.3m height error at a ~35 degree tilt -
    # sanity bound so a real regression (e.g. a sign flip) would fail loudly.
    assert xy_error < 1.0


def test_multiple_points_same_camera():
    cal = _overhead_camera()
    for truth in [
        np.array([0.0, 0.0, 1.0]),
        np.array([-2.0, 1.5, 1.0]),
        np.array([3.0, -1.0, 0.8]),
    ]:
        pixel = _project(cal, truth)
        recovered = estimate_position_at_height(cal, pixel, assumed_z_m=truth[2])
        assert np.allclose(recovered, truth, atol=1e-6), f"expected {truth}, got {recovered}"


if __name__ == "__main__":
    test_recovers_point_at_correct_height()
    test_wrong_height_assumption_produces_bounded_xy_error()
    test_multiple_points_same_camera()
    print("all tests passed")
