"""
Vision-based safety boundary - the senior engineer's suggestion (2026-09-12):
instead of computing precise real-world positions (needs calibration, still
blocked), define a KEEP-IN region directly in camera pixels and stop the
crane the instant the hook is seen crossing its edge. Same idea as a
mechanical limit switch on a rail, done with vision instead.

Why this needs NO calibration, when precise positioning does: this only
ever asks "is this pixel inside that pixel region" - a fact the image
answers directly. It never converts anything to real metres, so the
single-camera scale ambiguity that blocks the rest of this project doesn't
apply here. That's also why its only output is STOP, never a direction -
stopping is safe regardless of what the real distances turn out to be;
telling the crane to move a computed distance is not.

This is a SAFETY NET, not a targeting system - it does not replace
pick_order/horizontal_servo, it runs alongside them (or alongside manual
control) and only ever intervenes to stop something already in motion from
going somewhere it shouldn't.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from . import crane_remote, model_service
from .live_controller import HOOK_POSITION_CONFIDENCE, best_hook_detection
from .pick_order import SCOPE_ROI_PX
from .schemas import Detection

# One KEEP-IN polygon per camera - the hook is expected to stay INSIDE this
# region. Reuses pick_order.SCOPE_ROI_PX (2026-09-12 POC scope: left truck +
# conveyor only, right-side truck and background stacks excluded) so the
# safety boundary and "what's in play for picking" always agree - retrace
# both together if this camera is re-aimed, not just one.
BOUNDARY_POLYGON_PX: dict[str, list[tuple[float, float]]] = SCOPE_ROI_PX

# Inside this many pixels of the edge counts as "near_edge" - an early
# warning before an actual violation, so a monitor/log can flag it before
# a stop is ever needed.
NEAR_EDGE_MARGIN_PX = 40.0

# Don't fire stop_all more than once per this many seconds while a violation
# is continuously true - crane_remote.stop_all() is idempotent/safe to call
# repeatedly, this just avoids hammering the Modbus bridge needlessly.
STOP_RETRIGGER_S = 3.0


def _point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> float:
    """cv2.pointPolygonTest's signed distance: >0 inside, 0 on edge, <0 outside."""
    pts = np.array(poly, dtype=np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(pts, (float(x), float(y)), True)


@dataclass
class BoundaryStatus:
    status: str  # "no_hook" | "inside" | "near_edge" | "outside"
    hook_px: Optional[tuple[float, float]] = None
    margin_px: Optional[float] = None  # signed distance to the boundary edge
    hook_confidence: Optional[float] = None
    boundary_configured: bool = True


def check_boundary(detections: list[Detection], cam_id: str) -> BoundaryStatus:
    poly = BOUNDARY_POLYGON_PX.get(cam_id)
    if not poly:
        return BoundaryStatus(status="no_hook", boundary_configured=False)

    hook = best_hook_detection(detections)  # gated on HOOK_POSITION_CONFIDENCE - no guessing
    if hook is None:
        return BoundaryStatus(status="no_hook")

    hx, hy = hook.center.x, hook.center.y
    margin = _point_in_polygon(hx, hy, poly)
    if margin < 0:
        status = "outside"
    elif margin < NEAR_EDGE_MARGIN_PX:
        status = "near_edge"
    else:
        status = "inside"
    return BoundaryStatus(status=status, hook_px=(hx, hy), margin_px=margin, hook_confidence=hook.confidence)


# --- background watchdog -----------------------------------------------------
# Continuously watches the configured cameras and calls crane_remote.stop_all()
# the instant a confident boundary violation is seen. This is the ACTIVE half
# of the safety net - check_boundary() above is passive (just answers the
# question); this loop is what actually intervenes.

@dataclass
class _WatchdogState:
    enabled: bool = True
    last_status: dict[str, BoundaryStatus] = field(default_factory=dict)
    last_stop_ts: dict[str, float] = field(default_factory=dict)
    violation_count: int = 0
    log: list[str] = field(default_factory=list)

    def note(self, line: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {line}")
        self.log[:] = self.log[-200:]
        print(f"[boundary_guard] {line}")


_state = _WatchdogState()
_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()

WATCHDOG_CAMERAS = ["cam101"]
WATCHDOG_INTERVAL_S = 1.0
WATCHDOG_THRESHOLD = 0.3


def _watchdog_loop() -> None:
    from . import camera_feed  # local import - avoids a startup-order cycle with main.py

    _state.note("watchdog started")
    while not _stop_event.wait(WATCHDOG_INTERVAL_S):
        if not _state.enabled:
            continue
        for cam_id in WATCHDOG_CAMERAS:
            if not camera_feed.has_camera(cam_id):
                continue
            jpeg = camera_feed.get_snapshot(cam_id)
            if jpeg is None:
                continue
            img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            try:
                detections = model_service.detect_all(img, threshold=WATCHDOG_THRESHOLD)
            except FileNotFoundError:
                continue

            status = check_boundary(detections, cam_id)
            _state.last_status[cam_id] = status

            if status.status == "outside":
                now = time.monotonic()
                last = _state.last_stop_ts.get(cam_id, 0.0)
                if now - last >= STOP_RETRIGGER_S:
                    _state.last_stop_ts[cam_id] = now
                    _state.violation_count += 1
                    _state.note(
                        f"{cam_id}: BOUNDARY VIOLATION at {status.hook_px} "
                        f"(margin {status.margin_px:.0f}px) - calling crane_remote.stop_all()"
                    )
                    crane_remote.stop_all()
            elif status.status == "near_edge":
                _state.note(f"{cam_id}: near edge (margin {status.margin_px:.0f}px)")


def start() -> None:
    """Called from FastAPI startup - idempotent, safe to call more than once."""
    global _thread
    if _thread and _thread.is_alive():
        return
    _stop_event.clear()
    _thread = threading.Thread(target=_watchdog_loop, name="boundary-guard", daemon=True)
    _thread.start()


def stop() -> None:
    _stop_event.set()


def get_status() -> dict:
    return {
        "enabled": _state.enabled,
        "violation_count": _state.violation_count,
        "cameras": {
            cam_id: {
                "status": s.status,
                "hook_px": s.hook_px,
                "margin_px": s.margin_px,
                "hook_confidence": s.hook_confidence,
                "boundary_configured": s.boundary_configured,
            }
            for cam_id, s in _state.last_status.items()
        },
        "log": _state.log[-20:],
    }


def set_enabled(enabled: bool) -> None:
    _state.enabled = enabled
    _state.note(f"watchdog {'enabled' if enabled else 'disabled'}")
