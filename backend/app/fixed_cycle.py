"""
Fixed, single-target pick-and-drop cycle for the top-2-layers POC.

NOT a general vision-driven controller (see main.py's /api/control_tick for
that - and its own docstring on why real position feedback is required for
that path). This is a hardcoded, TIMED sequence of relay pulses: move
horizontal, hoist down, hoist up, move horizontal, hoist down, hoist up,
return home. It assumes the crane starts each run from the same known HOME
position (hoist fully up, parked at the reference corner) - if it doesn't,
every duration below is wrong and the cycle will not go where you expect.

Timing sources (see real_deploy/calib_out/rail_calibration.json):
  - E-W / N-S speeds and hoist speed: real, computed from tape-measured
    rail spans + remote-measured full-travel durations.
  - HOME_TO_PICK_S / PICK_TO_DROP_S / DROP_HOIST_DOWN_S: PLACEHOLDERS.
    These were never empirically timed (WiFi to the crane AP was unstable
    all session - see chat). Run one live supervised trial and update these
    three constants before trusting this for an unattended demo.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from . import camera_feed, crane_remote, model_service, pick_order
from .config import DEFAULT_THRESHOLD

UP, DOWN, EAST, WEST, SOUTH, NORTH = 0, 1, 2, 3, 4, 5

PICK_CAM_ID = "cam101"
MAX_LAYERS = 2  # POC scope: only ever targets the top 2 layers


def check_bale_present(cam_id: str = PICK_CAM_ID, max_layers: int = MAX_LAYERS) -> dict:
    """Vision gate, NOT a position solver (no working camera calibration -
    see calib_out/ and the session's failed checkerboard attempts). Confirms
    a real bale is actually detected within the top `max_layers` layers of
    the current camera frame before the fixed sequence commits to hoisting
    down. Returns {"present": bool, "count": int, "reason": str|None}."""
    jpeg = camera_feed.get_snapshot(cam_id)
    if jpeg is None:
        return {"present": False, "count": 0, "reason": f"no live frame from {cam_id}"}
    img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"present": False, "count": 0, "reason": "could not decode camera frame"}

    bales = model_service.detect_bales(img, threshold=DEFAULT_THRESHOLD)
    bales = [b for b in bales if pick_order.in_scope(b, cam_id)]
    if not bales:
        return {"present": False, "count": 0, "reason": "no bale detected in scope"}

    layers = pick_order.group_into_layers_px(bales)
    top_bales = [b for layer in layers[:max_layers] for b in layer]
    if not top_bales:
        return {"present": False, "count": 0, "reason": f"bales detected but none within top {max_layers} layers"}
    return {"present": True, "count": len(top_bales), "reason": None}

# --- Calibrated (real) ---
HOIST_FULL_UP_S = 86.0     # bottom -> top, or top -> bottom, full travel
HOIST_FULL_DOWN_S = 86.0
EW_FULL_SPAN_S = 102.0
NS_FULL_SPAN_S = 150.0

# No position sensor exists, so "fixed position" can only be guaranteed by
# running each axis into its physical limit switch - the motor stalls
# harmlessly against the limit regardless of where it started. 1.3x the
# calibrated full-span duration as overrun margin, so this reaches the limit
# even starting from the far opposite end.
HOME_MARGIN = 1.3
HOME_EW_S = EW_FULL_SPAN_S * HOME_MARGIN     # WEST - PICK: verify this is
                                              # actually the truck-bed side,
                                              # not yet confirmed which end.
HOME_NS_S = NS_FULL_SPAN_S * HOME_MARGIN     # NORTH - direction NOT yet
                                              # confirmed against the pick
                                              # side either; check on first
                                              # live run and flip to SOUTH
                                              # here if it homes the wrong way.
HOME_HOIST_S = HOIST_FULL_UP_S * HOME_MARGIN

# --- PLACEHOLDER - must be live-tuned once the crane WiFi link is stable.
# Best current guess only: short nudge from the pick spot to the conveyor
# input, based on today's manual jog tests (NOT a measured value).
HOME_TO_PICK_S = 3.0        # WEST, from HOME corner to over the truck bed
PICK_TO_DROP_S = 2.0        # EAST, truck bed -> conveyor input (short, adjacent)
DROP_HOIST_DOWN_S = 60.0    # hoist down at drop point - conveyor height unknown,
                             # this is a fraction of full 86s, UNVERIFIED

Step = tuple[int, float, str]  # (channel, duration_s, label)

SEQUENCE: list[Step] = [
    # --- Homing: run into physical limits, guarantees a known start
    # regardless of wherever the crane currently is. No vision/position
    # feedback needed for this part - it's mechanically self-correcting.
    (WEST, HOME_EW_S, "home: run to WEST limit"),
    (NORTH, HOME_NS_S, "home: run to NORTH limit (verify this is the pick side)"),
    (UP, HOME_HOIST_S, "home: run hoist to UP limit"),
    # --- Fixed timed offsets from HOME - placeholders, see note above.
    (WEST, HOME_TO_PICK_S, "move to pick position (over truck bed top layer)"),
    (DOWN, HOIST_FULL_DOWN_S, "hoist down onto bale (~173cm mark)"),
    (UP, HOIST_FULL_UP_S, "hoist up with bale"),
    (EAST, PICK_TO_DROP_S, "move to drop position (conveyor input)"),
    (DOWN, DROP_HOIST_DOWN_S, "hoist down to release onto conveyor"),
    (UP, HOIST_FULL_UP_S, "hoist up clear"),
    (EAST, HOME_TO_PICK_S - PICK_TO_DROP_S if HOME_TO_PICK_S > PICK_TO_DROP_S else 0.0,
     "return toward home"),
]


@dataclass
class CycleState:
    running: bool = False
    stop_requested: bool = False
    step_index: int = -1
    step_label: str = ""
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


_state = CycleState()


def get_status() -> dict:
    with _state.lock:
        return {
            "running": _state.running,
            "step_index": _state.step_index,
            "step_label": _state.step_label,
            "total_steps": len(SEQUENCE),
            "error": _state.error,
            "started_at": _state.started_at,
            "finished_at": _state.finished_at,
        }


def _run():
    with _state.lock:
        _state.running = True
        _state.stop_requested = False
        _state.error = None
        _state.started_at = time.time()
        _state.finished_at = None

    try:
        for i, (ch, dur, label) in enumerate(SEQUENCE):
            with _state.lock:
                if _state.stop_requested:
                    break
                _state.step_index = i
                _state.step_label = label

            # Vision gate right before the pick hoist-down - refuse to
            # blindly lower onto empty space. Matched by label (not a raw
            # index) so this stays correct if SEQUENCE is reordered. This is
            # a PRESENCE check only, not a position fix (see
            # check_bale_present docstring) - it does not steer the fixed
            # timings below.
            if label == "hoist down onto bale (~173cm mark)":
                check = check_bale_present()
                if not check["present"]:
                    with _state.lock:
                        _state.error = f"vision check failed before pick: {check['reason']}"
                    break

            if dur <= 0:
                continue
            crane_remote.manual_channel(ch, True)
            # Poll stop_requested in small slices so STOP is responsive
            # instead of blocking for the whole step duration.
            slept = 0.0
            while slept < dur:
                with _state.lock:
                    if _state.stop_requested:
                        break
                time.sleep(min(0.25, dur - slept))
                slept += 0.25
            crane_remote.manual_channel(ch, False)
            with _state.lock:
                if _state.stop_requested:
                    break
    except Exception as e:  # noqa: BLE001 - surface any failure to the UI, never crash silently
        with _state.lock:
            _state.error = str(e)
    finally:
        crane_remote.stop_all()
        with _state.lock:
            _state.running = False
            _state.finished_at = time.time()


def start_cycle() -> dict:
    """Kick off the fixed cycle on a background thread. Refuses to start if
    the physical crane remote isn't reachable (never blind-fires into a dead
    link) or if a cycle is already running."""
    with _state.lock:
        if _state.running:
            return {"ok": False, "reason": "already running"}
    remote_status = crane_remote.status()
    if remote_status is None:
        return {"ok": False, "reason": "crane remote unreachable - check WiFi link before starting"}
    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True}


def stop_cycle() -> dict:
    with _state.lock:
        _state.stop_requested = True
    crane_remote.stop_all()
    return {"ok": True}
