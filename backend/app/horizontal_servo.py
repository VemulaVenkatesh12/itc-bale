"""
Closed-loop horizontal (E/W/N/S) visual servo decision - compares the
hook's CURRENT pixel position to a target bale's pixel position and says
which way to nudge, refreshed every tick, instead of computing a real
distance/duration up front. This is what removes the need for metric
calibration on the horizontal axes specifically - see the 2026-09-11
conversation that led to this.

What this still does NOT solve, on purpose: the vertical/hoist stroke (the
actual pierce into the bale). Two points can align in a 2D image while
sitting at completely different real depths - pixel alignment here says
nothing about whether the spike is at the right height. That phase stays
under direct human direction; this module only ever proposes E/W/N/S.

DISPLAY-ONLY, same as pick_order.py and conveyor_gate.py: this computes and
returns a suggested nudge direction, it never calls crane_remote itself.
Two more things stand between this and firing anything for real:

1. The hook must be seen with real confidence (>= HOOK_POSITION_CONFIDENCE,
   same bar live_controller.py already uses for trusting a hook fix - not
   loosened here). Below that, this returns "hook not confidently visible"
   and suggests nothing, rather than guess.
2. The mapping from "target is left of hook in the image" to "which relay
   is that" is NOT assumed - it depends on how this specific camera is
   mounted relative to the crane's real axes, and has not been empirically
   established yet. See PIXEL_TO_RELAY_MAP below: until it's filled in from
   a real short test (fire a known relay briefly, see which way the hook
   moves in frame, record the sign), this reports the image-space direction
   only ("target is LEFT of hook"), not a relay name - so nothing downstream
   can act on it as if it were calibrated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from . import scale_calibration
from .live_controller import HOOK_POSITION_CONFIDENCE, best_hook_detection
from .schemas import Detection

# image-axis -> relay name, once empirically confirmed. None = not yet known;
# compute_servo_decision then reports direction in image terms only.
# Fill in after the mapping test, e.g.:
#   PIXEL_TO_RELAY_MAP = {"dx_positive": "east_1s", "dx_negative": "west_1s",
#                         "dy_positive": "south_1s", "dy_negative": "north_1s"}
PIXEL_TO_RELAY_MAP: Optional[dict[str, str]] = None

DEADZONE_PX = 25.0  # close enough in the image to call it "aligned"


@dataclass
class ServoDecision:
    status: str  # "no_hook" | "aligned" | "nudge"
    hook_px: Optional[tuple[float, float]] = None
    target_px: Optional[tuple[float, float]] = None
    dx: Optional[float] = None
    dy: Optional[float] = None
    image_direction: Optional[str] = None   # e.g. "target is LEFT and BELOW the hook"
    suggested_relay: Optional[str] = None   # only set once PIXEL_TO_RELAY_MAP is filled in
    hook_confidence: Optional[float] = None
    distance_m: Optional[float] = None      # only set once scale_calibration has a reference for this camera


def compute_servo_decision(
    detections: list[Detection], target_px: tuple[float, float], cam_id: Optional[str] = None,
) -> ServoDecision:
    hook = best_hook_detection(detections)  # already gates on HOOK_POSITION_CONFIDENCE
    if hook is None:
        return ServoDecision(status="no_hook")

    hx, hy = hook.center.x, hook.center.y
    tx, ty = target_px
    dx, dy = tx - hx, ty - hy
    distance_m = None
    if cam_id is not None:
        distance_m = scale_calibration.px_to_m(cam_id, math.hypot(dx, dy))

    if abs(dx) < DEADZONE_PX and abs(dy) < DEADZONE_PX:
        return ServoDecision(status="aligned", hook_px=(hx, hy), target_px=target_px,
                              dx=dx, dy=dy, hook_confidence=hook.confidence, distance_m=distance_m)

    horiz = "RIGHT" if dx > 0 else "LEFT"
    vert = "BELOW" if dy > 0 else "ABOVE"
    parts = []
    if abs(dx) >= DEADZONE_PX:
        parts.append(horiz)
    if abs(dy) >= DEADZONE_PX:
        parts.append(vert)
    image_direction = f"target is {' and '.join(parts)} the hook (dx={dx:+.0f}px, dy={dy:+.0f}px)"
    if distance_m is not None:
        image_direction += f" (~{distance_m:.2f}m away)"

    suggested_relay = None
    if PIXEL_TO_RELAY_MAP is not None:
        # Dominant axis first - same "one axis at a time, nearest first"
        # spirit as compute_tick, just picking whichever error is larger.
        if abs(dx) >= abs(dy):
            suggested_relay = PIXEL_TO_RELAY_MAP["dx_positive" if dx > 0 else "dx_negative"]
        else:
            suggested_relay = PIXEL_TO_RELAY_MAP["dy_positive" if dy > 0 else "dy_negative"]

    return ServoDecision(status="nudge", hook_px=(hx, hy), target_px=target_px, dx=dx, dy=dy,
                          image_direction=image_direction, suggested_relay=suggested_relay,
                          hook_confidence=hook.confidence, distance_m=distance_m)
