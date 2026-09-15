"""
Single-reference pixel-to-metre scale (2026-09-12): the operator gave ONE
real distance - hook to the currently-targeted bale - read off a live
camera frame ("hook to targeted bale is about 1.3m right now"). This turns
that single number into a pixels-per-metre ratio, so horizontal_servo's
pixel offsets can be reported as real-world metres too.

This is NOT the 6-point calibration (see PLANT_VISIT_CHECKLIST.md) - it is
one scale factor assumed constant across the traced POC zone (one rough
depth plane: left truck + bed roller). It will drift for anything much
nearer/farther from the camera than the reference pair was. Good enough for
direction + rough distance in this bounded zone; not a substitute for real
per-axis calibration if precise stop-distances are ever needed.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class ScaleReference:
    cam_id: str
    pixels_per_metre: float
    ref_distance_m: float
    ref_pixel_distance: float
    hook_px: tuple[float, float]
    target_px: tuple[float, float]
    set_at: str


_state: dict[str, ScaleReference] = {}


def set_reference(
    cam_id: str,
    hook_px: tuple[float, float],
    target_px: tuple[float, float],
    distance_m: float,
) -> ScaleReference:
    if distance_m <= 0:
        raise ValueError("distance_m must be > 0")
    pixel_distance = math.hypot(target_px[0] - hook_px[0], target_px[1] - hook_px[1])
    if pixel_distance <= 1:
        raise ValueError("hook and target are the same point in this frame - can't derive a scale from it")
    ref = ScaleReference(
        cam_id=cam_id,
        pixels_per_metre=pixel_distance / distance_m,
        ref_distance_m=distance_m,
        ref_pixel_distance=pixel_distance,
        hook_px=hook_px,
        target_px=target_px,
        set_at=time.strftime("%H:%M:%S"),
    )
    _state[cam_id] = ref
    return ref


def get_reference(cam_id: str) -> Optional[ScaleReference]:
    return _state.get(cam_id)


def px_to_m(cam_id: str, pixel_distance: float) -> Optional[float]:
    ref = _state.get(cam_id)
    if ref is None:
        return None
    return pixel_distance / ref.pixels_per_metre
