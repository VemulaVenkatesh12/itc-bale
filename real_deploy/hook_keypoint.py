"""
Contact-tip point for the crane hook, derived geometrically from its
detected bounding box.

## Why this is an offset, not a keypoint model

The proper way to get the hook's contact point is a keypoint head on the
detector. RF-DETR does ship one (`RFDETRKeypointPreview`), but only as an
XLarge model - there is no keypoint Nano - and it does not fit the 6 GB
GTX 1660 SUPER this project trains on. So instead of a learned keypoint we
use the fact that, across the 873 hand-verified hook frames in
`hook_front_vision_verified.jsonl`, the contact tip sits at a very stable
spot *inside* the detected box:

    x = left + 0.500 * width      (dead centre, horizontally)
    y = top  + 0.824 * height     (~82 % of the way down)

Residual error of that fixed offset, measured against the 873 verified
points (median box ~205 x 130 px in a 2560 x 1440 frame):

    typical (median):  x +/- 12.8 px,  y +/- 8.5 px
    90th percentile:   x +/- 66.6 px

The vertical estimate is tight; the horizontal one has a longer tail (on
some frames the ram is genuinely angled off-centre). For closed-loop crane
approach - where the hook is re-detected every control tick and the
vertical axis comes from the mast encoder, not vision - this is accurate
enough. Swap in a real keypoint head here if a >=12 GB GPU becomes
available; the verified points are already in the COCO annotations for it.
"""

from __future__ import annotations

# Fraction of the box width / height at which the contact tip sits.
# Medians over the 873 verified frames; see module docstring.
CONTACT_TIP_X_FRAC = 0.500
CONTACT_TIP_Y_FRAC = 0.824


def contact_tip_px(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    """Given a crane_hook box (x1,y1)-(x2,y2) in pixels, return the estimated
    contact-tip point (x, y) in the same pixel coordinates.

    Use this in place of the plain box centre for `class_id == crane_hook`
    when feeding detections into triangulation / position tracking.
    """
    w = x2 - x1
    h = y2 - y1
    return (x1 + CONTACT_TIP_X_FRAC * w, y1 + CONTACT_TIP_Y_FRAC * h)
