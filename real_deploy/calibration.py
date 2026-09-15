"""
Camera calibration data model shared by calibrate_intrinsics.py,
calibrate_extrinsics.py, triangulate.py, and capture_infer.py.

Nothing in this file talks to a camera or a PLC - it's the plain-data
representation of "what a calibrated camera is" (intrinsics + distortion +
pose in a shared world frame), plus (de)serialization so the two
calibration scripts (run once, offline, per camera) and the live capture
service (run continuously) can be decoupled: calibrate once, save JSON,
load it wherever it's needed.

Convention (standard OpenCV / computer-vision convention, NOT the
simulation's Three.js/Blender axis conventions in backend/app/
live_controller.py - this is a real-world coordinate frame you define
yourself when you place the extrinsic calibration targets, e.g. "origin at
the truck bed's near-left corner, X along the bed's long axis, Z up"):

    X_camera = R @ X_world + t

R, t map a WORLD point into that camera's own coordinate frame. The 3x4
projection matrix P = K @ [R | t] then maps a world point to homogeneous
image pixel coordinates: [u, v, w]^T = P @ [X, Y, Z, 1]^T, pixel = (u/w, v/w).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CameraCalibration:
    name: str  # e.g. "top", "back" - matches the simulation's CAMERA_TOP/CAMERA_BACK naming
    image_width: int
    image_height: int
    camera_matrix: np.ndarray  # 3x3 intrinsics K
    dist_coeffs: np.ndarray  # (N,) distortion coefficients, OpenCV's k1,k2,p1,p2[,k3...] order
    rotation: np.ndarray  # 3x3 R, world -> camera
    translation: np.ndarray  # (3,) t, world -> camera, same units as your world frame (meters recommended)

    @property
    def projection_matrix(self) -> np.ndarray:
        """3x4 P = K @ [R | t] - what triangulate.py actually needs."""
        rt = np.hstack([self.rotation, self.translation.reshape(3, 1)])
        return self.camera_matrix @ rt

    @property
    def camera_center_world(self) -> np.ndarray:
        """This camera's own position in world coordinates (C = -R^T @ t) -
        useful for sanity-checking a calibration against a tape-measure
        distance, not used by triangulation itself."""
        return -self.rotation.T @ self.translation

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraCalibration":
        return cls(
            name=d["name"],
            image_width=d["image_width"],
            image_height=d["image_height"],
            camera_matrix=np.array(d["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(d["dist_coeffs"], dtype=np.float64),
            rotation=np.array(d["rotation"], dtype=np.float64),
            translation=np.array(d["translation"], dtype=np.float64),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CameraCalibration":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def intrinsics_only(
        cls, name: str, image_width: int, image_height: int, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
    ) -> "CameraCalibration":
        """Intermediate object produced by calibrate_intrinsics.py, before
        extrinsics are known - rotation/translation are identity/zero
        placeholders, NOT a valid pose. calibrate_extrinsics.py loads one of
        these and fills in the real rotation/translation; anything that
        needs a real projection_matrix (triangulate.py, capture_infer.py)
        must be given a calibration that's been through both steps."""
        return cls(
            name=name,
            image_width=image_width,
            image_height=image_height,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            rotation=np.eye(3),
            translation=np.zeros(3),
        )
