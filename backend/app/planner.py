"""
Core domain logic: turns raw bale detections into

  1. a pick order (topmost-in-stack first, nearest-neighbour tie-break —
     this doubles as the shortest-path heuristic between consecutive picks)
  2. a full relay on/off timeline compatible with the SF-8DR wiring layout
  3. crane position keyframes for frontend animation

No real position feedback exists in this POC (no laser ToF, no PLC) — the
full ITC proposal's closed-loop Δx/Δy control is future scope. Here motion
is *timed* (open-loop): distance / calibrated px-per-second speed = duration.
This mirrors exactly what you asked for: "how long the switch should be on
and when to release is the main logic."

Obstacle avoidance: the hook is only ever moved horizontally while fully
hoisted ("clear"), and horizontal travel is routed through a transit row
above the whole working area (`_TimelineBuilder.safe_move`) instead of a
direct diagonal — so it never sweeps across the remaining stack at
bale-height. Concretely, every move is: retreat straight up to the transit
row -> cross at the transit row -> descend onto the target. Once at the
transit row, crossing is a straight shortest-path line — there's nothing
left to hit up there. See `safe_move` docstring for the exact three-leg
breakdown.

Drop point: every bale is dropped at the *same* fixed point (the marked
drop area's center), not spread across a grid. This matches a conveyor
belt drop target — the belt carries each bale away before the next one
arrives, so there's no need (and no realism benefit) to spatially
de-conflict drop positions the way we would for a static storage area.

Relay naming matches the SF-8DR terminal legend:
  north_1s / south_1s / ns_2s   (terminals 15,16,17)
  east_1s  / west_1s  / ew_2s   (terminals 11,12,13)
  up_1s    / up_2s              (terminals 6,7)
  down_1s  / down_2s            (terminals 8,9)
  push                          (ASSUMPTION: mapped to spare relay R1 /
                                  terminal 21 — a hydraulic push-off
                                  actuator, NOT a grip actuator. The spike
                                  is a fixed prong: bales are impaled by the
                                  crane's own `DOWN` stroke and held purely
                                  by friction while lifted — no relay fires
                                  for grabbing or holding. `push` fires only
                                  once, briefly, at the drop point to eject
                                  a bale that's stuck on the spike onto the
                                  conveyor. Confirm/remap against real
                                  wiring before driving actual hardware.)

Safety: EAST/WEST, NORTH/SOUTH and UP/DOWN are opposing pairs and must
never be energized simultaneously (this mirrors the "software interlocking
grid" called out in the ITC proposal). Because each axis move picks exactly
one direction relay per move, this is true by construction here;
`verify_interlocks` re-checks the generated timeline defensively.
"""
from __future__ import annotations

import math
from typing import Optional

from .schemas import (
    CraneKeyframe,
    Detection,
    OrientationConfig,
    PickCycle,
    Point,
    Rect,
    RelayEvent,
    SimulateRequest,
    SimulationResult,
    SpeedConfig,
)

Y_TOLERANCE_PX = 30  # bales within this many px of the topmost bbox top are treated as "same layer"


def point_in_rect(p: Point, r: Rect) -> bool:
    rn = r.normalized()
    return rn.x1 <= p.x <= rn.x2 and rn.y1 <= p.y <= rn.y2


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def filter_detections_in_area(detections: list[Detection], area: Rect) -> list[Detection]:
    return [d for d in detections if point_in_rect(d.center, area)]


def order_bales(detections: list[Detection], start: Point) -> list[Detection]:
    """Topmost-bbox-first (proxy for 'physically highest in the stack, must
    come off before what's under it'), nearest-to-current-position as
    tie-break within the same layer band."""
    remaining = list(detections)
    order: list[Detection] = []
    pos = start
    while remaining:
        min_y = min(d.bbox.y1 for d in remaining)
        band = [d for d in remaining if d.bbox.y1 <= min_y + Y_TOLERANCE_PX]
        nxt = min(band, key=lambda d: _dist(pos, d.center))
        order.append(nxt)
        remaining.remove(nxt)
        pos = nxt.center
    return order


def _axis_events(
    distance_px: float,
    positive_relay: str,
    negative_relay: str,
    shared_2s_relay: str,
    axis: str,
    low_px_s: float,
    high_px_s: float,
    min_dist_high: float,
    creep_dist: float,
) -> tuple[list[RelayEvent], float]:
    """Relay events (times relative to move start = 0) + move duration (s)."""
    if abs(distance_px) < 0.5:
        return [], 0.0
    relay_name = positive_relay if distance_px > 0 else negative_relay
    dist = abs(distance_px)

    if dist < min_dist_high or high_px_s <= low_px_s:
        total = dist / low_px_s
        return [RelayEvent(relay=relay_name, axis=axis, phase="", on_at_ms=0, off_at_ms=round(total * 1000))], total

    creep = min(creep_dist, dist)
    high_dist = dist - creep
    t_high = high_dist / high_px_s
    t_creep = creep / low_px_s
    total = t_high + t_creep

    events = [RelayEvent(relay=relay_name, axis=axis, phase="", on_at_ms=0, off_at_ms=round(total * 1000))]
    if t_high > 0:
        events.append(
            RelayEvent(relay=shared_2s_relay, axis=axis, phase="", on_at_ms=0, off_at_ms=round(t_high * 1000))
        )
    return events, total


def _hoist_events(
    direction: str, total_s: float, low_relay: str, high_relay: str, creep_s: float
) -> tuple[list[RelayEvent], float]:
    if total_s <= 0:
        return [], 0.0
    creep = min(creep_s, total_s)
    high_part = total_s - creep
    events = [RelayEvent(relay=low_relay, axis="hoist", phase="", on_at_ms=0, off_at_ms=round(total_s * 1000))]
    if high_part > 0:
        events.append(
            RelayEvent(relay=high_relay, axis="hoist", phase="", on_at_ms=0, off_at_ms=round(high_part * 1000))
        )
    return events, total_s


class _TimelineBuilder:
    def __init__(self, speed: SpeedConfig, orientation: OrientationConfig, start: Point):
        self.speed = speed
        self.orientation = orientation
        self.pos = start
        self.t_ms = 0
        self.hoist = 1.0
        self.thrust = 0.0
        self.spiked = False
        self.relay_timeline: list[RelayEvent] = []
        self.keyframes: list[CraneKeyframe] = [
            CraneKeyframe(
                t_ms=0, x=start.x, y=start.y, hoist=1.0, thrust=0.0,
                spiked=False, active_bale_id=None, phase="home",
            )
        ]

    def _settle(self) -> None:
        self.t_ms += round(self.speed.settle_s * 1000)

    def move_to(self, target: Point, phase: str, bale_id: Optional[int]) -> None:
        s = self.speed
        o = self.orientation
        dx = target.x - self.pos.x
        dy = target.y - self.pos.y

        ew_pos_relay, ew_neg_relay = ("west_1s", "east_1s") if o.flip_ew else ("east_1s", "west_1s")
        ns_pos_relay, ns_neg_relay = ("north_1s", "south_1s") if o.flip_ns else ("south_1s", "north_1s")

        ew_events, ew_dur = _axis_events(
            dx, ew_pos_relay, ew_neg_relay, "ew_2s", "ew", s.ew_low_px_s, s.ew_high_px_s,
            s.high_speed_min_distance_px, s.creep_distance_px,
        )
        ns_events, ns_dur = _axis_events(
            dy, ns_pos_relay, ns_neg_relay, "ns_2s", "ns", s.ns_low_px_s, s.ns_high_px_s,
            s.high_speed_min_distance_px, s.creep_distance_px,
        )
        duration_s = max(ew_dur, ns_dur)
        if duration_s <= 0:
            self.pos = target
            return

        for ev in ew_events + ns_events:
            self.relay_timeline.append(
                ev.model_copy(update={
                    "phase": phase,
                    "on_at_ms": self.t_ms + ev.on_at_ms,
                    "off_at_ms": self.t_ms + ev.off_at_ms,
                })
            )
        self.t_ms += round(duration_s * 1000)
        self.pos = target
        self.keyframes.append(
            CraneKeyframe(
                t_ms=self.t_ms, x=self.pos.x, y=self.pos.y, hoist=self.hoist, thrust=self.thrust,
                spiked=self.spiked, active_bale_id=bale_id, phase=phase,
            )
        )
        self._settle()

    def safe_move(self, target: Point, phase: str, bale_id: Optional[int], transit_y: float) -> None:
        """Move to `target` without ever sweeping diagonally across the
        working area at bale height. Three axis-pure legs:

          1. retreat straight to the transit row (N/S only) — this is the
             "move up/clear first" the whole way, not just a hoist bump.
          2. cross at the transit row (E/W only) — nothing else is up
             there, so a direct line *is* the shortest safe path.
          3. descend onto the target (N/S only).

        Any leg with zero distance is a no-op (handled by move_to itself),
        so this degrades gracefully to a single straight move when no
        detour is actually needed. Caller is responsible for making sure
        the hook is already fully hoisted (`self.hoist == 1.0`) before
        calling this — that's what actually keeps it above the stack.
        """
        self.move_to(Point(x=self.pos.x, y=transit_y), f"{phase}_retreat", bale_id)
        self.move_to(Point(x=target.x, y=transit_y), f"{phase}_cross", bale_id)
        self.move_to(target, f"{phase}_descend", bale_id)

    def hoist_move(self, direction: str, total_s: float, phase: str, bale_id: Optional[int]) -> None:
        s = self.speed
        low_relay, high_relay = ("up_1s", "up_2s") if direction == "up" else ("down_1s", "down_2s")
        events, dur = _hoist_events(direction, total_s, low_relay, high_relay, s.hoist_creep_s)
        if dur <= 0:
            return
        for ev in events:
            self.relay_timeline.append(
                ev.model_copy(update={
                    "phase": phase,
                    "on_at_ms": self.t_ms + ev.on_at_ms,
                    "off_at_ms": self.t_ms + ev.off_at_ms,
                })
            )
        self.t_ms += round(dur * 1000)
        self.hoist = 1.0 if direction == "up" else 0.0
        self.keyframes.append(
            CraneKeyframe(
                t_ms=self.t_ms, x=self.pos.x, y=self.pos.y, hoist=self.hoist, thrust=self.thrust,
                spiked=self.spiked, active_bale_id=bale_id, phase=phase,
            )
        )
        self._settle()

    def pierce(self, phase: str, bale_id: int) -> None:
        """The actual grab: a horizontal stab (parallel to ground, NOT the
        vertical hoist) that drives the fixed spike into the bale. `hoist`
        by this point has only brought the hook down to roughly top-layer
        height — this is a distinct axis (`thrust`) doing the piercing, and
        no relay fires for it (mechanical, same as the resulting hold)."""
        self.t_ms += round(self.speed.pierce_dwell_s * 1000)
        self.spiked = True
        self.thrust = 1.0
        self.keyframes.append(
            CraneKeyframe(
                t_ms=self.t_ms, x=self.pos.x, y=self.pos.y, hoist=self.hoist, thrust=self.thrust,
                spiked=True, active_bale_id=bale_id, phase=phase,
            )
        )
        self._settle()

    def push_off(self, phase: str, bale_id: Optional[int]) -> None:
        """Hydraulic push relay pulse: ejects a bale that's stuck on the
        spike onto the conveyor. This is the only relay event in the whole
        grab/carry/release cycle — the carry itself needs none. Also the
        horizontal counterpart to `pierce`: `thrust` retracts back to 0 as
        the bale comes off the spike."""
        on_at = self.t_ms
        self.t_ms += round(self.speed.push_off_s * 1000)
        self.relay_timeline.append(
            RelayEvent(relay="push", axis="push", phase=phase, on_at_ms=on_at, off_at_ms=self.t_ms)
        )
        self.spiked = False
        self.thrust = 0.0
        self.keyframes.append(
            CraneKeyframe(
                t_ms=self.t_ms, x=self.pos.x, y=self.pos.y, hoist=self.hoist, thrust=self.thrust,
                spiked=False, active_bale_id=None, phase=phase,
            )
        )
        self._settle()


def verify_interlocks(relay_timeline: list[RelayEvent]) -> list[str]:
    """Defensive re-check: opposing relays must never overlap in time."""
    warnings: list[str] = []
    opposing_pairs = [("east_1s", "west_1s"), ("north_1s", "south_1s"), ("up_1s", "down_1s")]
    for a, b in opposing_pairs:
        a_events = [e for e in relay_timeline if e.relay == a]
        b_events = [e for e in relay_timeline if e.relay == b]
        for ea in a_events:
            for eb in b_events:
                if ea.on_at_ms < eb.off_at_ms and eb.on_at_ms < ea.off_at_ms:
                    warnings.append(
                        f"INTERLOCK VIOLATION: {a} and {b} overlap "
                        f"({ea.on_at_ms}-{ea.off_at_ms}ms vs {eb.on_at_ms}-{eb.off_at_ms}ms)"
                    )
    return warnings


def plan_simulation(all_detections: list[Detection], req: SimulateRequest, image_w: int, image_h: int) -> SimulationResult:
    warnings: list[str] = []
    detection_area = req.detection_area.normalized()
    drop_area = req.drop_area.normalized()

    in_area = filter_detections_in_area(all_detections, detection_area)
    if not in_area:
        warnings.append("No bales detected inside the marked detection area.")

    crane_home = req.crane_home or Point(
        x=(detection_area.x1 + detection_area.x2) / 2,
        y=max(0.0, detection_area.y1 - 40.0),
    )

    order = order_bales(in_area, crane_home)
    if req.max_bales is not None:
        order = order[: req.max_bales]

    # Fixed single drop point — the drop area represents a moving conveyor,
    # which carries each bale away before the next one lands, so there's no
    # previous-bale-in-the-way to dodge (unlike a static storage grid).
    drop_point = drop_area.center

    # A y-row above everything (pick pile, drop area, and home) that
    # `safe_move` transits through so horizontal travel never crosses the
    # remaining stack at bale height.
    transit_margin = 30.0
    transit_y = max(0.0, min(detection_area.y1, drop_area.y1, crane_home.y) - transit_margin)

    tb = _TimelineBuilder(req.speed, req.orientation, crane_home)
    pick_cycles: list[PickCycle] = []

    for idx, det in enumerate(order, start=1):
        tag = f"#{idx}"
        trip_start_ms = tb.t_ms

        # The previous trip ends lowered (for release), so re-lift before
        # any horizontal travel — skipping this was the bug that let the
        # empty hook drag through the remaining stack between picks.
        if tb.hoist < 1.0:
            tb.hoist_move("up", req.speed.hoist_clear_s, f"lift_clear{tag}", None)

        tb.safe_move(det.center, f"travel_to_bale{tag}", det.id, transit_y)
        tb.hoist_move("down", req.speed.hoist_lower_s, f"lower_to_bale{tag}", det.id)
        tb.pierce(f"pierce{tag}", det.id)
        tb.hoist_move("up", req.speed.hoist_clear_s, f"lift_with_load{tag}", det.id)
        tb.safe_move(drop_point, f"travel_to_drop{tag}", det.id, transit_y)
        tb.hoist_move("down", req.speed.hoist_lower_s, f"lower_to_drop{tag}", det.id)
        tb.push_off(f"push_off{tag}", det.id)

        pick_cycles.append(
            PickCycle(
                order=idx, bale_id=det.id, bbox=det.bbox, confidence=det.confidence,
                pick_point=det.center, drop_point=drop_point,
                t_start_ms=trip_start_ms, t_end_ms=tb.t_ms,
            )
        )

    if pick_cycles:
        if tb.hoist < 1.0:
            tb.hoist_move("up", req.speed.hoist_clear_s, "return_lift", None)
        tb.safe_move(crane_home, "return_home", None, transit_y)

    warnings.extend(verify_interlocks(tb.relay_timeline))

    return SimulationResult(
        image_width=image_w,
        image_height=image_h,
        detections=in_area,
        pick_cycles=pick_cycles,
        relay_timeline=tb.relay_timeline,
        keyframes=tb.keyframes,
        total_duration_ms=tb.t_ms,
        crane_home=crane_home,
        warnings=warnings,
    )
