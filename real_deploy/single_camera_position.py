"""
Single-camera position estimate via ray/known-height-plane intersection -
the cam101-ONLY replacement for triangulate.py's two-camera triangulation
(see decision: 2026-09-15, cam102 cannot currently see the same 6 extrinsic
reference points as cam101 and a re-aim wasn't done, so real automation
runs on cam101 alone).

## Why this exists, and what it costs vs. two-camera triangulation

A single camera has NO way to recover depth on its own - a pixel is a full
3D ray (see triangulate.py's module docstring), not a point. Two cameras
resolve this by intersecting two independent rays. With only one camera,
the only way to pin down a unique 3D point is to ADD an assumption:
"this pixel's real-world Z (height) is approximately H" - then the ray has
exactly one intersection with the Z=H plane, and that intersection is your
X, Y estimate.

This project's own scope decision (crane only works the TOP TWO LAYERS of
a load, see measurements.md section 0 and backend/app/pick_order.py's
max_layers) makes "assume a per-layer height" a reasonable assumption, not
an arbitrary one - a bale's top surface at a given layer sits at a roughly
known height once bale height and truck-bed floor height are measured
(Phase 4). But it IS still an assumption, and every meter it's wrong by
translates roughly 1:1 into X/Y position error at cam101's viewing angle -
worse the shallower the ray-to-plane angle, better the steeper (cam101's
overhead-ish mount is the good case here, not the bad one, which is WHY
cam101-only is viable at all - a near-horizontal camera would be far worse
for this technique).

Compare to triangulate.py: two independently calibrated views
over-determine the point with no height assumption needed at all. That
redundancy is also a safety cross-check (see backend/app/main.py's
pick_order_cross_view_status) - with one camera, there is no independent
way to catch this camera being occluded, mis-aimed, or drifted after a
bump. That safety property is genuinely lost, not just accuracy - Phase 10
(safety-engineer review) should know this before sign-off.

Pure geometry, testable without hardware the same way triangulate.py is
(synthetic camera with known pose, project a known point down to a known
Z-plane, check this recovers it) - see tests/test_single_camera_position.py.
"""

from __future__ import annotations

import numpy as np

from calibration import CameraCalibration
from triangulate import _undistort_to_pixel  # reuse the same distortion-removal convention


def estimate_position_at_height(
    cal: CameraCalibration, pixel: tuple[float, float], assumed_z_m: float
) -> np.ndarray:
    """The world point (x, y, assumed_z_m) whose ray from `cal`'s camera
    center passes through `pixel` (distortion-corrected first, same as
    triangulate.py). Degenerates (ZeroDivisionError-ish -> raises) if the
    ray is exactly parallel to the Z=assumed_z_m plane, which in practice
    only happens for a pixel exactly on the camera's own horizon line."""
    u = _undistort_to_pixel(cal, pixel)  # ideal pixel coords, distortion removed

    # Ray direction in camera space: K^-1 @ [u, v, 1]^T (not unit length,
    # doesn't need to be - the plane-intersection scale factor s below
    # absorbs whatever length this is).
    k_inv = np.linalg.inv(cal.camera_matrix)
    ray_camera = k_inv @ np.array([u[0], u[1], 1.0])

    # Camera center and ray direction, both rotated into WORLD coordinates.
    # X_camera = R @ X_world + t  =>  X_world = R^T @ (X_camera - t), and a
    # direction (no translation) transforms as just R^T @ dir.
    camera_center = cal.camera_center_world  # already -R^T @ t, see calibration.py
    ray_world = cal.rotation.T @ ray_camera

    if abs(ray_world[2]) < 1e-9:
        raise ValueError(
            f"ray for pixel {pixel} is parallel to the Z={assumed_z_m} plane - "
            "cannot solve for an intersection (this pixel is on the camera's own horizon)"
        )

    s = (assumed_z_m - camera_center[2]) / ray_world[2]
    if s <= 0:
        raise ValueError(
            f"pixel {pixel} at assumed height {assumed_z_m}m solves to a point BEHIND the "
            "camera (s<=0) - assumed_z_m is likely wrong for this pixel, or the calibration is"
        )
    point_world = camera_center + s * ray_world
    return point_world
