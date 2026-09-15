"""
Conveyor congestion gate - requirement 3 of the 2026-09-11 production
requirement: "if the conveyor/drop area already has bales piled up, stop
issuing pick signals; resume once the count there drops."

Deliberately independent of calibration - it only needs to know which
region of a camera's PIXELS is the conveyor belt, then counts how many bale
detections currently sit there. That's a fixed screen-space fact for a
camera that isn't moving, unlike a bale's real-world position (which needs
the calibration this repo doesn't have yet). This is why it can ship today
while requirement 2 (real pick/place with measured speed) still can't.

Nothing here fires a relay - it only answers "is it safe to send the next
pick signal right now", for whatever eventually issues that signal
(currently nothing; this is the gate ready and waiting for it).
"""
from __future__ import annotations

import numpy as np

from .schemas import Detection

CLASS_BALE = 0

# One polygon (pixel corners, clockwise or counter-clockwise, doesn't
# matter) per camera, tracing the conveyor belt SURFACE where a dropped bale
# actually lands - not the whole belt structure/frame. Eyeballed against a
# 2026-09-11 cam101 snapshot (backend/app is the ground truth for what the
# camera currently sees; re-trace this if the camera is re-aimed - see
# PLANT_VISIT_CHECKLIST.md Stage 2 - since these are raw screen pixels, not
# world coordinates, and do not survive a camera move).
CONVEYOR_ROI_PX: dict[str, list[tuple[float, float]]] = {
    "cam101": [(625, 70), (745, 65), (1155, 450), (860, 400)],
}

# How many bales sitting in the ROI counts as "full" - i.e. the belt hasn't
# cleared the last drop(s) yet. 2 is a conservative starting default (one
# bale mid-transit is normal; two+ means it's backing up); tune once real
# belt speed/spacing is known.
DEFAULT_FULL_THRESHOLD = 2


def _point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    pts = np.array(poly, dtype=np.float32).reshape((-1, 1, 2))
    import cv2
    return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0


def count_bales_on_conveyor(detections: list[Detection], cam_id: str) -> int:
    """How many bale detections have their centre inside this camera's
    conveyor ROI right now. Returns 0 (not an error) for a camera with no
    configured ROI - callers should treat that as "gate not available here",
    not "definitely empty"."""
    poly = CONVEYOR_ROI_PX.get(cam_id)
    if not poly:
        return 0
    return sum(
        1 for d in detections
        if d.class_id == CLASS_BALE and _point_in_polygon(d.center.x, d.center.y, poly)
    )


def conveyor_status(detections: list[Detection], cam_id: str, full_threshold: int = DEFAULT_FULL_THRESHOLD) -> dict:
    """{"count", "full", "ok_to_pick", "roi_configured"} - `ok_to_pick` is
    the actual answer requirement 3 asks for: False means hold every pick
    signal until this flips back to True on a later call."""
    roi_configured = cam_id in CONVEYOR_ROI_PX
    count = count_bales_on_conveyor(detections, cam_id)
    full = roi_configured and count >= full_threshold
    return {
        "roi_configured": roi_configured,
        "count": count,
        "full_threshold": full_threshold,
        "full": full,
        "ok_to_pick": (not full) if roi_configured else None,
    }
