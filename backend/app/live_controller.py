"""
Python port of the live-3D closed-loop crane controller
(frontend/src/live/controller.ts) plus the scene geometry it depends on
(frontend/src/live/factoryLayout.ts). This is the "brain" behind
POST /api/control_tick: given the crane's self-reported current pose and
this tick's fresh camera frame, it re-runs detection, re-confirms the
target bale against LIVE vision every tick (not just once at target lock),
and decides which relay(s) should be energized right now plus the setpoint
they're driving toward.

Two-tier control, matching how real vision-guided industrial automation
actually splits the work: this (the vision/PLC scan, ticking at whatever
rate RF-DETR inference allows - see /api/control_tick) decides relay
engagement AND the current setpoint each scan; the frontend's local "drive"
loop (controller.ts) respects that setpoint with a tight 60fps loop between
scans, the way a real VFD/servo with its own encoder handles precise
stopping between PLC scans. A pure "backend just names a relay" design
would overshoot badly at our ~1.3Hz vision tick rate (up to ~1.4m of travel
between ticks at high speed) - the setpoint is what avoids that.

Stateless by design: nothing here is remembered between calls. The
frontend echoes back its own last-known crane position/phase/
elapsed-in-phase time each tick, the way a real controller reads fresh
encoder feedback each scan rather than trusting its own memory of where
the actuator "should" be by now.

Geometry constants below are hand-mirrored from
frontend/src/live/factoryLayout.ts - there's no shared source of truth
across the language boundary, so keep them in sync by hand if that file
changes. The bale-grid PRNG (mulberry32) and the Camera_Top unprojection
math were verified bit-exact against real three.js/JS output (node,
against the actual `three` package) before this was written; if
factoryLayout.ts's numbers change, this file needs the same edit.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Geometry (mirrors frontend/src/live/factoryLayout.ts)
# ---------------------------------------------------------------------------

_ORIGIN_BX = 9.0
_ORIGIN_BY = 2.75


def _blender_to_three(bx: float, by: float, bz: float) -> tuple[float, float, float]:
    return (bx - _ORIGIN_BX, bz, -(by - _ORIGIN_BY))


TRUCK_BED_TOP_Y = _blender_to_three(9.0, 2.75, 1.0)[1]
TRUCK_BED_MAX_Z = _ORIGIN_BY  # open/back end - see factoryLayout.ts's TRUCK.bedMaxZ

BALE_SIZE = (0.85, 0.6, 0.85)  # x, y (height), z

CRANE_HOIST_DOWN_Y = 1.4
CRANE_HOIST_UP_Y = 5.0
CRANE_SPIKE_REACH = BALE_SIZE[2]
CRANE_HOOK_SIZE_Z = 0.24
CRANE_HOOK_HALF_Z = CRANE_HOOK_SIZE_Z / 2

CONVEYOR_DROP_POINT = _blender_to_three(11.4, 1.3, 1.4)  # (x, y, z)

CAMERA_TOP_POSITION = _blender_to_three(10.2, 2.75, 9.5)
CAMERA_TOP_LOOKAT = _blender_to_three(10.2, 2.75, 0)
CAMERA_TOP_UP = (0.0, 0.0, -1.0)  # matches CameraFeed.tsx's TOP_CAMERA_UP
CAMERA_TOP_LENS_MM = 16.0
SENSOR_WIDTH_MM = 36.0
# frontend/src/live/CameraFeed.tsx's CameraFeed default panelWidth/panelHeight,
# used (unchanged) for the "top" instance - this is the aspect the camera's own
# fov was actually set up with, independent of any particular captured frame's
# dimensions (which should match it exactly, but the *camera* doesn't care).
CAMERA_TOP_PANEL_ASPECT = 320 / 180


def _horizontal_fov_deg(lens_mm: float, sensor_mm: float = SENSOR_WIDTH_MM) -> float:
    return math.degrees(2 * math.atan(sensor_mm / (2 * lens_mm)))


def _vertical_fov_deg(h_fov_deg: float, aspect: float) -> float:
    h_half_rad = math.radians(h_fov_deg) / 2
    return math.degrees(2 * math.atan(math.tan(h_half_rad) / aspect))


CAMERA_TOP_VFOV_DEG = _vertical_fov_deg(_horizontal_fov_deg(CAMERA_TOP_LENS_MM), CAMERA_TOP_PANEL_ASPECT)

Vec3 = tuple[float, float, float]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(a: Vec3) -> Vec3:
    length = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    return (a[0] / length, a[1] / length, a[2] / length)


# Confidence bar a crane_hook detection must clear before it's trusted as a
# POSITION MEASUREMENT (used to drive the crane), separate from and stricter
# than DEFAULT_THRESHOLD (which just gates what's worth drawing/returning at
# all). A wrong vision fix is worse than falling back to the self-reported
# pose for one tick, so this errs conservative.
HOOK_POSITION_CONFIDENCE = 0.5


def best_hook_detection(detections):
    """Highest-confidence crane_hook (class_id==1, see model_service.
    CLASS_CRANE_HOOK) detection at or above HOOK_POSITION_CONFIDENCE, or None
    if the hook wasn't confidently seen this tick - e.g. self-occluded by the
    mast pole from near-straight-down, which happens roughly 2/3 of the time
    from Camera_Top (see training/README.md's visibility-rate note). Accepts
    anything with .class_id/.confidence (schemas.Detection in practice)."""
    candidates = [d for d in detections if d.class_id == 1 and d.confidence >= HOOK_POSITION_CONFIDENCE]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.confidence)


def unproject_pixel_to_ground(
    px: float, py: float, frame_w: int, frame_h: int, plane_y: float = TRUCK_BED_TOP_Y
) -> Optional[tuple[float, float]]:
    """Pixel (origin top-left, matching Detection.bbox convention) -> world
    (x, z) on the ground plane, through the fixed Camera_Top pose. A direct
    port of coords.ts's unprojectToGroundPlane (THREE.Raycaster-based) -
    verified bit-exact (to fp noise) against real three.js output for a
    battery of test pixels before this was written."""
    ndc_x = (px / frame_w) * 2 - 1
    ndc_y = -((py / frame_h) * 2 - 1)

    z_axis = _norm(_sub(CAMERA_TOP_POSITION, CAMERA_TOP_LOOKAT))
    x_axis = _norm(_cross(CAMERA_TOP_UP, z_axis))
    y_axis = _cross(z_axis, x_axis)
    forward = (-z_axis[0], -z_axis[1], -z_axis[2])

    tan_fov_y = math.tan(math.radians(CAMERA_TOP_VFOV_DEG) / 2)
    tan_fov_x = tan_fov_y * CAMERA_TOP_PANEL_ASPECT

    direction = _norm((
        x_axis[0] * ndc_x * tan_fov_x + y_axis[0] * ndc_y * tan_fov_y + forward[0],
        x_axis[1] * ndc_x * tan_fov_x + y_axis[1] * ndc_y * tan_fov_y + forward[1],
        x_axis[2] * ndc_x * tan_fov_x + y_axis[2] * ndc_y * tan_fov_y + forward[2],
    ))
    if abs(direction[1]) < 1e-9:
        return None
    t = (plane_y - CAMERA_TOP_POSITION[1]) / direction[1]
    return (CAMERA_TOP_POSITION[0] + direction[0] * t, CAMERA_TOP_POSITION[2] + direction[2] * t)


# ---------------------------------------------------------------------------
# Bale grid (mirrors factoryLayout.ts's buildBaleGrid(1) exactly - same
# mulberry32 PRNG, same seed, verified bit-exact against the real JS output)
# ---------------------------------------------------------------------------

_MASK32 = 0xFFFFFFFF


def _make_mulberry32(seed: int):
    state = seed & _MASK32

    def rand() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & _MASK32
        a = state
        t = ((a ^ (a >> 15)) * (1 | a)) & _MASK32
        old_t = t
        prod2 = ((t ^ (t >> 7)) * (61 | t)) & _MASK32
        t = ((old_t + prod2) & _MASK32) ^ old_t
        t &= _MASK32
        return ((t ^ (t >> 14)) & _MASK32) / 4294967296.0

    return rand


def _build_bale_grid(seed: int = 1) -> list[dict]:
    rand = _make_mulberry32(seed)
    bales: list[dict] = []
    bh = BALE_SIZE[1]
    col_xs = [-0.45, 0.45]
    row_count = 6
    row_spacing = 0.92
    row_start_z = TRUCK_BED_MAX_Z - 0.6
    bid = 0

    layer0_y = TRUCK_BED_TOP_Y + bh / 2
    for row in range(row_count):
        for cx in col_xs:
            jx = (rand() - 0.5) * 0.05
            jz = (rand() - 0.5) * 0.05
            bales.append({"id": bid, "x": cx + jx, "y": layer0_y, "z": row_start_z - row * row_spacing + jz, "layer": 0})
            bid += 1

    layer1_y = layer0_y + bh
    for row in range(4):
        for cx in col_xs:
            jx = (rand() - 0.5) * 0.05
            jz = (rand() - 0.5) * 0.05
            bales.append({"id": bid, "x": cx + jx, "y": layer1_y, "z": row_start_z - row * row_spacing + jz, "layer": 1})
            bid += 1

    return bales


BALE_GRID: list[dict] = _build_bale_grid(1)

# ---------------------------------------------------------------------------
# Control law (mirrors controller.ts)
# ---------------------------------------------------------------------------

ARRIVE_EPS = 0.02
CONFIRM_RADIUS_M = 0.7  # how close a live top-camera detection must be to a bale to "confirm" it
STALL_FALLBACK_S = 4.0  # see pick_next_target's allow_blocked docstring
LONG_STALL_FALLBACK_S = 10.0  # see pick_next_target's require_confirmation docstring
PIERCE_DWELL_S = 0.8
PUSH_OFF_S = 0.8
PIERCE_DEPTH_M = 0.2  # how far the spike sinks into a bale from its OWN top surface
HOOK_HALF_Z = CRANE_HOOK_HALF_Z
APPROACH_CLEARANCE_M = 0.3
_MAX_CHAIN = 12  # bounded chain-through for zero-wait phase transitions (mirrors controller.ts's guard<20)


def _hoist_down_y_for(bale: dict) -> float:
    bale_top_y = bale["y"] + BALE_SIZE[1] / 2
    return bale_top_y - PIERCE_DEPTH_M


def _insert_z_for(z: float) -> float:
    return z + BALE_SIZE[2] / 2 + HOOK_HALF_Z


def _standoff_z_for(z: float) -> float:
    return _insert_z_for(z) + CRANE_SPIKE_REACH + APPROACH_CLEARANCE_M


# Row spacing in the bale grid (see _build_bale_grid's row_spacing) is 0.92m,
# but the standoff distance in front of a target - insertZ's half-bale-depth
# + hookHalfZ + the full spike reach (one bale-depth) + clearance - is ~1.7m,
# i.e. MORE than one row's spacing. So the standoff position for any row
# isn't clear air: it lands back on top of whatever's still sitting in the
# same column, one row closer to the open end (verified: with these
# constants the crane's hook+spike danger zone overlaps ~0.6m into that
# row's own footprint while it lowers the hoist for the row behind it).
# A real fixed-spike crane has this exact constraint - it has to clear a
# column front-to-back for the same geometric reason. pick_next_target below
# refuses to target a bale while a still-unpicked bale occupies that "in the
# way" slot, rather than letting the crane bulldoze through it (this was a
# real, reproduced bug: picking skipped a momentarily-unconfirmed front bale
# and went straight for the one behind it, clipping the front one on the way
# down - see itc-crane-column-blocking-bug in project memory).
_COLUMN_TOLERANCE_M = 0.5
_ROW_TOLERANCE_M = 0.3  # layers share row_start_z/row_spacing (see _build_bale_grid) - this is jitter-only slack


def _is_blocked(candidate: dict, unpicked: list[dict]) -> bool:
    for other in unpicked:
        if other["id"] == candidate["id"]:
            continue
        if abs(other["x"] - candidate["x"]) > _COLUMN_TOLERANCE_M:
            continue  # different column - spike can't reach it
        if other["layer"] == candidate["layer"] and other["z"] > candidate["z"]:
            return True  # still there, nearer the open end -> in the way of the approach
        if other["layer"] > candidate["layer"] and abs(other["z"] - candidate["z"]) < _ROW_TOLERANCE_M:
            return True  # still there, directly above -> in the way of the vertical descent
    return False


# Exactly one relay energized at a time, always the "1s" (low-speed) tier -
# no combining a direction relay with a "2s" speed-boost relay, and no
# combining relays across axes (see return_home below for the other place
# that used to do the latter). Observed live: the simulated relay panel
# was lighting up two relays simultaneously (e.g. "WEST 1S" + "E/W 2S"
# together for a long move) - real industrial relay panels/PLC outputs are
# not assumed safe to energize two coils on the same axis (or two axes)
# at once, so this reflects that constraint even though nothing here
# fires a real relay yet. The "2s" high-speed tier and multi-axis "return
# home" are dropped for now, not redesigned - if variable speed is wanted
# back later, it needs a real mechanism (e.g. sequential timed pulses on
# a single relay) rather than energizing two coils together.
def _axis_relays(axis: str, remaining: float) -> list[str]:
    if abs(remaining) < ARRIVE_EPS:
        return []
    direction = 1 if remaining > 0 else -1
    if axis == "x":
        return ["east_1s" if direction > 0 else "west_1s"]
    return ["south_1s" if direction > 0 else "north_1s"]


def _hoist_relays(remaining: float) -> list[str]:
    if abs(remaining) < ARRIVE_EPS:
        return []
    return ["up_1s"] if remaining > 0 else ["down_1s"]


def pick_next_target(
    picked_ids: set[int],
    live_bale_detections: list[tuple[float, float, float]],
    prev_x: float,
    prev_z: float,
    allow_blocked: bool = False,
    require_confirmation: bool = True,
) -> Optional[dict]:
    """Confirm-radius match against live world-space bale detections,
    topmost layer first, back-to-front tie-break - port of controller.ts's
    pickNextTarget. Re-run every tick (not just once at target lock) so the
    live detections keep refreshing what "confirmed" means.

    `allow_blocked` is the bounded escape hatch compute_tick reaches for
    after STALL_FALLBACK_S of nothing confirmable: normally a candidate with
    a still-unpicked bale in its approach/descent path (see _is_blocked) is
    refused outright rather than risk clipping it, but the flip side is that
    if vision just never re-confirms the one bale that WOULD clear the jam
    (observed: a freshly-exposed front bale drifting a few cm after its
    neighbor comes out, just enough to sit outside CONFIRM_RADIUS_M), the
    cycle stalls forever despite the target bed data knowing exactly what's
    unpicked and where. Falling back to the pre-blocking behavior after a
    timeout accepts the small risk of a clipped neighbor over an infinite,
    silent hang - the same tradeoff a real system makes with a "confirm
    timeout" override.

    `require_confirmation=False` is a second, longer-timeout escape hatch
    (see compute_tick's LONG_STALL_FALLBACK_S) for a harder case than
    allow_blocked covers: a bale that's been physically knocked off the
    truck bed can end up somewhere the detector just never confidently
    recognizes as a bale again (reproduced live: confidence stuck around
    0.1-0.17, well under DEFAULT_THRESHOLD) - no amount of waiting fixes
    that, since it isn't detection noise, it's a genuine detection gap. The
    bale grid still knows it's unpicked, so this targets it on that alone."""
    unpicked = [b for b in BALE_GRID if b["id"] not in picked_ids]
    if not unpicked:
        return None

    def _is_confirmed(b: dict) -> bool:
        if not require_confirmation:
            return True
        return any(math.hypot(d[0] - b["x"], d[1] - b["z"]) < CONFIRM_RADIUS_M for d in live_bale_detections)

    confirmed = [b for b in unpicked if _is_confirmed(b) and (allow_blocked or not _is_blocked(b, unpicked))]
    if not confirmed:
        return None

    max_layer = max(b["layer"] for b in confirmed)
    top_row = [b for b in confirmed if b["layer"] == max_layer]

    def _compare(a: dict, b: dict) -> int:
        dist_a = TRUCK_BED_MAX_Z - a["z"]
        dist_b = TRUCK_BED_MAX_Z - b["z"]
        if abs(dist_a - dist_b) > 0.3:
            return -1 if dist_a < dist_b else 1
        da = math.hypot(a["x"] - prev_x, a["z"] - prev_z)
        db = math.hypot(b["x"] - prev_x, b["z"] - prev_z)
        return -1 if da < db else (1 if da > db else 0)

    top_row.sort(key=functools.cmp_to_key(_compare))
    return top_row[0]


# Approach/pierce waypoints below are computed from each target bale's
# NOMINAL (x, z) - its known, ground-truth-in-this-sim grid position - not
# from re-homing on live per-tick detections of it. An earlier version tried
# live-tracking the target the whole way (re-running _live_xz_for-style
# nearest-detection matching in "cross"/"descend"/"pierce"), even with the
# match clamped to a small radius around nominal - but a detection can be
# persistently (not just randomly jittering) offset from the bale's true
# center by an amount that never satisfies ARRIVE_EPS (0.02), so the crane
# chased a target that never settled and never arrived (reproduced live:
# stuck orbiting ~0.3m short of centered on the actual bale). Worse, in
# "pierce" specifically a detection biased too far AWAY from the crane could
# drive the spike past the bale's true far face into whatever still sat in
# the row behind it - a real, reproduced collision. Live detections still
# matter for target CONFIRMATION (pick_next_target, above) and for the
# overlay/telemetry - just not for homing in on the approach waypoints.


@dataclass
class TickState:
    phase: str
    crane_x: float
    crane_y: float
    crane_z: float
    target_bale_id: Optional[int]
    elapsed_in_phase_s: float
    picked_ids: set[int] = field(default_factory=set)


@dataclass
class TickDecision:
    phase: str
    active_relays: list[str]
    target_bale_id: Optional[int]
    target_x: float
    target_y: float
    target_z: float
    mark_picked_bale_id: Optional[int]
    log: Optional[str]


def compute_tick(state: TickState, live_bale_detections: list[tuple[float, float, float]]) -> TickDecision:
    phase = state.phase
    elapsed = state.elapsed_in_phase_s
    cx, cy, cz = state.crane_x, state.crane_y, state.crane_z
    target_bale_id = state.target_bale_id
    log: Optional[str] = None

    def resolve_bale(bid: Optional[int]) -> Optional[dict]:
        if bid is None:
            return None
        return next((b for b in BALE_GRID if b["id"] == bid), None)

    for _ in range(_MAX_CHAIN):
        if phase in ("idle", "select_target"):
            unpicked_count = len(BALE_GRID) - len(state.picked_ids)
            if unpicked_count == 0:
                phase, target_bale_id = "return_home", None
                log = log or "all bales picked - returning home"
                continue
            target = pick_next_target(state.picked_ids, live_bale_detections, cx, cz)
            if target is not None:
                log = f"picking bale #{target['id']}"
            elif elapsed >= LONG_STALL_FALLBACK_S:
                # Nothing at all confirmable for a long stretch - not just a
                # blocked-but-visible bale (STALL_FALLBACK_S below already
                # covers that), but NO bale confirmable, period. Usually
                # means one got knocked off the bed into a pose/lighting the
                # detector doesn't recognize (see pick_next_target's
                # require_confirmation docstring) - target it on the grid's
                # own bookkeeping since vision isn't going to help.
                target = pick_next_target(
                    state.picked_ids, live_bale_detections, cx, cz, allow_blocked=True, require_confirmation=False
                )
                if target is not None:
                    log = (
                        f"picking bale #{target['id']} - vision hasn't confirmed ANY target in "
                        f"{LONG_STALL_FALLBACK_S:.0f}s, proceeding on its last-known position"
                    )
            elif elapsed >= STALL_FALLBACK_S:
                target = pick_next_target(state.picked_ids, live_bale_detections, cx, cz, allow_blocked=True)
                if target is not None:
                    log = (
                        f"picking bale #{target['id']} - vision hasn't confirmed a clear "
                        f"target in {STALL_FALLBACK_S:.0f}s, proceeding despite a blocking neighbor"
                    )
            if target is None:
                return TickDecision("select_target", [], None, cx, cy, cz, None, log)
            phase, target_bale_id = "cross", target["id"]
            elapsed = 0.0
            continue

        # return_home/done are the only phases below that don't need a
        # target bale (return_home's own transition, above, deliberately
        # clears target_bale_id to None) - resolving one for them would
        # always fail and bounce straight back to select_target, discarding
        # the "all picked" transition before it could ever run (reproduced
        # live: stuck re-logging "all bales picked - returning home" forever
        # once genuinely all 20 were picked, without the crane ever actually
        # heading home).
        if phase not in ("return_home", "done"):
            target_bale = resolve_bale(target_bale_id)
            if target_bale is None:
                # stale/unknown target (e.g. a request racing a reset) - bail
                # to a safe re-selection state rather than raise.
                return TickDecision("select_target", [], None, cx, cy, cz, None, log)

        if phase == "cross":
            target_x = target_bale["x"]  # nominal - see the comment after pick_next_target, above
            relays = _axis_relays("x", target_x - cx)
            if relays:
                return TickDecision("cross", relays, target_bale_id, target_x, cy, cz, None, log)
            phase, elapsed = "descend", 0.0
            continue

        if phase == "descend":
            target_z = _standoff_z_for(target_bale["z"])
            relays = _axis_relays("z", target_z - cz)
            if relays:
                return TickDecision("descend", relays, target_bale_id, cx, cy, target_z, None, log)
            phase, elapsed = "hoist_down_to_bale", 0.0
            continue

        if phase == "hoist_down_to_bale":
            target_y = _hoist_down_y_for(target_bale)
            relays = _hoist_relays(target_y - cy)
            if relays:
                return TickDecision("hoist_down_to_bale", relays, target_bale_id, cx, target_y, cz, None, log)
            phase, elapsed = "pierce", 0.0
            continue

        if phase == "pierce":
            # Nominal z, same reasoning as "cross"/"descend" above: a
            # persistent (not just jittering) live-detection offset in
            # either direction would otherwise either stall convergence
            # (never satisfying ARRIVE_EPS against a target that keeps
            # reading a bit short) or - the more dangerous direction - drive
            # the spike past the bale's true far face into whatever's still
            # sitting in the row behind it (reproduced live). The bale's own
            # grid position is ground truth in this sim, so there's nothing
            # for live tracking to correct here.
            target_z = _insert_z_for(target_bale["z"])
            relays = _axis_relays("z", target_z - cz)
            if relays:
                return TickDecision("pierce", relays, target_bale_id, cx, cy, target_z, None, log)
            phase, elapsed = "pierce_settle", 0.0
            continue

        if phase == "pierce_settle":
            if elapsed >= PIERCE_DWELL_S:
                phase, elapsed = "hoist_up_clear", 0.0
                continue
            return TickDecision("pierce_settle", [], target_bale_id, cx, cy, cz, None, log)

        if phase == "hoist_up_clear":
            relays = _hoist_relays(CRANE_HOIST_UP_Y - cy)
            if relays:
                return TickDecision("hoist_up_clear", relays, target_bale_id, cx, CRANE_HOIST_UP_Y, cz, None, log)
            phase, elapsed = "drop_cross", 0.0
            continue

        if phase == "drop_cross":
            target_x = CONVEYOR_DROP_POINT[0]
            relays = _axis_relays("x", target_x - cx)
            if relays:
                return TickDecision("drop_cross", relays, target_bale_id, target_x, cy, cz, None, log)
            phase, elapsed = "drop_descend", 0.0
            continue

        if phase == "drop_descend":
            target_z = CONVEYOR_DROP_POINT[2]
            relays = _axis_relays("z", target_z - cz)
            if relays:
                return TickDecision("drop_descend", relays, target_bale_id, cx, cy, target_z, None, log)
            phase, elapsed = "hoist_down_to_drop", 0.0
            continue

        if phase == "hoist_down_to_drop":
            relays = _hoist_relays(CRANE_HOIST_DOWN_Y - cy)
            if relays:
                return TickDecision("hoist_down_to_drop", relays, target_bale_id, cx, CRANE_HOIST_DOWN_Y, cz, None, log)
            phase, elapsed = "push_off", 0.0
            continue

        if phase == "push_off":
            if elapsed >= PUSH_OFF_S:
                return TickDecision(
                    "hoist_up_after_drop", [], target_bale_id, cx, cy, cz,
                    target_bale_id, f"bale #{target_bale_id} dropped",
                )
            return TickDecision("push_off", ["push"], target_bale_id, cx, cy, cz, None, log)

        if phase == "hoist_up_after_drop":
            relays = _hoist_relays(CRANE_HOIST_UP_Y - cy)
            if relays:
                return TickDecision("hoist_up_after_drop", relays, target_bale_id, cx, CRANE_HOIST_UP_Y, cz, None, log)
            phase, target_bale_id, elapsed = "select_target", None, 0.0
            continue

        if phase == "return_home":
            # X then Z, never both - see _axis_relays' module comment. z is
            # held at its current value (cz, not 0.0) while x is still
            # moving, so the frontend's drive loop doesn't also move z just
            # because a nonzero target was sent for it.
            x_relays = _axis_relays("x", -cx)
            if x_relays:
                return TickDecision("return_home", x_relays, None, 0.0, cy, cz, None, log)
            z_relays = _axis_relays("z", -cz)
            if z_relays:
                return TickDecision("return_home", z_relays, None, 0.0, cy, 0.0, None, log)
            return TickDecision("done", [], None, cx, cy, cz, None, log or "cycle complete")

        if phase == "done":
            return TickDecision("done", [], None, cx, cy, cz, None, None)

        # unknown phase (e.g. stale client) - fail safe
        return TickDecision("select_target", [], None, cx, cy, cz, None, log)

    # guard tripped (shouldn't happen in practice) - hold in place inertly
    # rather than risk an infinite loop.
    return TickDecision(phase, [], target_bale_id, cx, cy, cz, None, "control loop guard tripped")
