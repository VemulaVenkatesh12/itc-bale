"""
Turns real triangulated 3D bale positions into an SF-8DR relay on/off
timeline. This is the real-world counterpart to `backend/app/planner.py`,
which plans in IMAGE PIXELS because the simulation only ever had a 2D frame.

It does not import from `backend/` - `real_deploy/` stays standalone, per
this directory's README.

## World frame convention (this MUST match your extrinsic calibration)

    +X = EAST   (trolley travel)      -X = WEST
    +Y = NORTH  (bridge travel)       -Y = SOUTH
    +Z = UP     (hoist)               metres, Z=0 at conveyor/floor datum

`calibrate_extrinsics.py` is what physically defines this frame, by way of
the real-world reference points you measure and enter. If you set that frame
up with a different axis convention, every relay this module emits drives
the crane in the wrong direction - so fix the convention at calibration
time, not here. There is no way for this module to detect the mistake.

## Three things this does BETTER than the 2D planner, because Z is real

1. **Pick order is genuinely "highest first."** `planner.py` orders by
   topmost bounding-box edge (`bbox.y1`) as a *proxy* for "physically
   highest in the stack, must come off before what's under it" - a proxy
   that fails whenever a distant bale sits higher in the image than a near
   bale that is actually taller. Here the stack height IS measured, so
   ordering is by real Z.

2. **Hoist duration is computed, not configured.** `planner.py`'s known
   assumption #4 is that `hoist_clear_s`/`hoist_lower_s` are fixed constants
   "because there's no Z-depth signal in a 2D image." Triangulation is
   exactly that missing signal, so hoist moves are distance-based like every
   other axis.

3. **Obstacle avoidance is a real height, not a fake image row.**
   `planner.py::safe_move` detours through a "transit row" in IMAGE Y
   because it had no third axis to lift into - a 2D stand-in for lifting.
   Here the hook genuinely rises to a transit Z above the tallest bale, and
   X/Y then travel simultaneously (they are independent relay pairs, and
   nothing is up there to hit).

## What is deliberately unchanged from planner.py

The relay names and SF-8DR terminal mapping, the two-speed 1S/2S pattern,
the single fixed conveyor drop point, the push-off-only-at-drop cycle, and
the interlock re-check. Those are wiring and mechanical facts about this
plant; they do not become different because the positions got better. In
particular `push` -> R1 is still the same UNCONFIRMED assumption flagged in
the root README - confirm it against real wiring before energizing anything.

## Still open-loop

The timeline is durations computed from distances, exactly like the
simulation. Real closed-loop control - feeding `position_tracker.py`'s
smoothed position back to decide when to CUT a relay rather than trusting a
precomputed duration - is the ACTING half that `real_deploy/README.md`
deliberately excludes until the observing half is proven on real hardware.
Everything here is a plan to be reviewed, not a command stream to be run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

# Opposing relays that must never be energized together - the "software
# interlocking grid" from the ITC proposal, same pairs as planner.py.
OPPOSING_PAIRS = [("east_1s", "west_1s"), ("north_1s", "south_1s"), ("up_1s", "down_1s")]


@dataclass
class SpeedConfig3D:
    """All speeds in metres/second, all distances in metres - the whole point
    of the real pipeline is that these are physical units, not pixels.

    Every default here is a PLACEHOLDER. Measure them on the real crane
    (time a known travel distance at each speed) before the numbers mean
    anything - a timeline built on guessed speeds is precise and wrong.
    """
    ew_low_m_s: float = 0.15
    ew_high_m_s: float = 0.45
    ns_low_m_s: float = 0.15
    ns_high_m_s: float = 0.45
    hoist_low_m_s: float = 0.10
    hoist_high_m_s: float = 0.25

    high_speed_min_distance_m: float = 0.75  # below this, never bother with 2S
    creep_distance_m: float = 0.25           # final approach on 1S only, for precision
    hoist_creep_m: float = 0.15

    settle_s: float = 0.4        # mechanical settle dwell after each move
    pierce_dwell_s: float = 1.0  # spike seating - mechanical, fires no relay
    push_off_s: float = 1.5      # the single `push` relay pulse at the drop

    clearance_m: float = 0.35    # how far above the tallest bale the transit height sits
    approach_m: float = 0.10     # stop this far above a bale before the piercing stroke


@dataclass
class Bale3D:
    id: int
    position: np.ndarray  # (x, y, z) metres, world frame
    confidence: float = 1.0


@dataclass
class RelayEvent3D:
    relay: str
    axis: str
    phase: str
    on_at_ms: int
    off_at_ms: int


@dataclass
class PickCycle3D:
    order: int
    bale_id: int
    pick_point: list[float]
    drop_point: list[float]
    t_start_ms: int
    t_end_ms: int


@dataclass
class Plan3D:
    relay_timeline: list[RelayEvent3D]
    pick_cycles: list[PickCycle3D]
    total_duration_ms: int
    transit_z: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "relay_timeline": [asdict(e) for e in self.relay_timeline],
            "pick_cycles": [asdict(c) for c in self.pick_cycles],
            "total_duration_ms": self.total_duration_ms,
            "transit_z": self.transit_z,
            "warnings": self.warnings,
        }


def group_into_layers(bales: list[Bale3D], layer_band_m: float = 0.5) -> list[list[Bale3D]]:
    """Cluster bales into physical layers by real Z, tallest layer first.

    Greedy from the top: the highest remaining bale defines a layer, and
    every bale within `layer_band_m` below it joins that layer; repeat on
    what's left. `layer_band_m` must be larger than triangulation noise plus
    stacking slop (a few cm) but smaller than one bale height, so a genuine
    next-layer-down bale is never absorbed into the layer above it.
    """
    remaining = sorted(bales, key=lambda b: float(b.position[2]), reverse=True)
    layers: list[list[Bale3D]] = []
    while remaining:
        top_z = float(remaining[0].position[2])
        layer = [b for b in remaining if float(b.position[2]) >= top_z - layer_band_m]
        layers.append(layer)
        layer_ids = {id(b) for b in layer}
        remaining = [b for b in remaining if id(b) not in layer_ids]
    return layers


def filter_top_layers(
    bales: list[Bale3D], max_layers: int | None, layer_band_m: float = 0.5
) -> tuple[list[Bale3D], int]:
    """Keep only bales in the top `max_layers` physical layers; return
    (kept_bales, n_layers_seen).

    This is the "crane services the TOP TWO LAYERS ONLY" rule: the lower
    layer(s) are left on the truck bed and are not pick candidates. Pass
    `max_layers=None` to disable (keep everything).
    """
    layers = group_into_layers(bales, layer_band_m)
    if max_layers is None or max_layers >= len(layers):
        return list(bales), len(layers)
    kept = [b for layer in layers[:max_layers] for b in layer]
    return kept, len(layers)


def order_bales_by_height(bales: list[Bale3D], start: np.ndarray, z_tolerance_m: float = 0.20) -> list[Bale3D]:
    """Highest real Z first; nearest-to-current-position as tie-break within
    a same-layer band.

    The band exists because bales in one physical layer are never at exactly
    equal Z - measurement noise and real stacking slop spread them by a few
    centimetres. Ordering on raw Z alone would therefore zig-zag across the
    stack picking a marginally-higher bale on the far side over an adjacent
    one, which is both slower and, since every pick is a detour through
    transit height, needlessly hard on the machine. `z_tolerance_m` should be
    comfortably larger than your triangulation noise but smaller than one
    bale height.
    """
    remaining = list(bales)
    order: list[Bale3D] = []
    pos = np.asarray(start, dtype=np.float64)

    while remaining:
        max_z = max(b.position[2] for b in remaining)
        band = [b for b in remaining if b.position[2] >= max_z - z_tolerance_m]
        nxt = min(band, key=lambda b: float(np.linalg.norm(b.position[:2] - pos[:2])))
        order.append(nxt)
        remaining.remove(nxt)
        pos = nxt.position
    return order


def _axis_events(
    distance_m: float, positive_relay: str, negative_relay: str, shared_2s_relay: str | None,
    axis: str, low_m_s: float, high_m_s: float, min_dist_high_m: float, creep_m: float,
) -> tuple[list[RelayEvent3D], float]:
    """Relay events (times relative to move start = 0) plus the move duration.

    Mirrors planner.py::_axis_events: the direction relay is held for the
    WHOLE move, and the shared 2S high-speed line is additionally energized
    only for the initial fast portion, dropping to 1S-only for the final
    creep. That is how real two-speed contactors are driven.
    """
    if abs(distance_m) < 1e-3:  # sub-millimetre: not a move
        return [], 0.0

    relay_name = positive_relay if distance_m > 0 else negative_relay
    dist = abs(distance_m)

    if dist < min_dist_high_m or high_m_s <= low_m_s or shared_2s_relay is None:
        total = dist / low_m_s
        return [RelayEvent3D(relay_name, axis, "", 0, round(total * 1000))], total

    creep = min(creep_m, dist)
    t_high = (dist - creep) / high_m_s
    t_creep = creep / low_m_s
    total = t_high + t_creep

    events = [RelayEvent3D(relay_name, axis, "", 0, round(total * 1000))]
    if t_high > 0:
        events.append(RelayEvent3D(shared_2s_relay, axis, "", 0, round(t_high * 1000)))
    return events, total


class _Builder:
    def __init__(self, speed: SpeedConfig3D, start: np.ndarray):
        self.s = speed
        self.pos = np.array(start, dtype=np.float64)
        self.t_ms = 0
        self.timeline: list[RelayEvent3D] = []

    def _settle(self) -> None:
        self.t_ms += round(self.s.settle_s * 1000)

    def _emit(self, events: list[RelayEvent3D], phase: str, duration_s: float) -> None:
        for ev in events:
            self.timeline.append(RelayEvent3D(
                relay=ev.relay, axis=ev.axis, phase=phase,
                on_at_ms=self.t_ms + ev.on_at_ms, off_at_ms=self.t_ms + ev.off_at_ms,
            ))
        self.t_ms += round(duration_s * 1000)
        self._settle()

    def move_xy(self, target_xy: np.ndarray, phase: str) -> None:
        """X and Y travel SIMULTANEOUSLY - they are independent relay pairs
        (terminals 11/12/13 vs 15/16/17), so the move takes as long as the
        slower axis, not the sum. Only ever called at transit height."""
        s = self.s
        dx = float(target_xy[0] - self.pos[0])
        dy = float(target_xy[1] - self.pos[1])

        ew_events, ew_dur = _axis_events(
            dx, "east_1s", "west_1s", "ew_2s", "ew",
            s.ew_low_m_s, s.ew_high_m_s, s.high_speed_min_distance_m, s.creep_distance_m)
        ns_events, ns_dur = _axis_events(
            dy, "north_1s", "south_1s", "ns_2s", "ns",
            s.ns_low_m_s, s.ns_high_m_s, s.high_speed_min_distance_m, s.creep_distance_m)

        duration = max(ew_dur, ns_dur)
        if duration <= 0:
            return
        self._emit(ew_events + ns_events, phase, duration)
        self.pos[0], self.pos[1] = target_xy[0], target_xy[1]

    def move_z(self, target_z: float, phase: str) -> None:
        """Hoist to an absolute height. Distance-based, which the 2D planner
        could not do (its known assumption #4)."""
        s = self.s
        dz = float(target_z - self.pos[2])
        events, duration = _axis_events(
            dz, "up_1s", "down_1s", None, "hoist",
            s.hoist_low_m_s, s.hoist_high_m_s, s.high_speed_min_distance_m, s.hoist_creep_m)
        if duration <= 0:
            return
        # Hoist has its own dedicated 2S lines (6/7 up, 8/9 down) rather than
        # one shared line, so the high-speed relay is direction-specific.
        if dz != 0 and abs(dz) >= s.high_speed_min_distance_m and s.hoist_high_m_s > s.hoist_low_m_s:
            creep = min(s.hoist_creep_m, abs(dz))
            t_high = (abs(dz) - creep) / s.hoist_high_m_s
            if t_high > 0:
                events.append(RelayEvent3D(
                    "up_2s" if dz > 0 else "down_2s", "hoist", "", 0, round(t_high * 1000)))
        self._emit(events, phase, duration)
        self.pos[2] = target_z

    def safe_move(self, target: np.ndarray, phase: str, transit_z: float) -> None:
        """Lift to transit height -> travel in XY -> descend onto target.

        The lift is what actually provides obstacle avoidance: horizontal
        travel only ever happens above the tallest bale. This replaces the
        2D planner's three axis-pure legs through an image "transit row",
        which was a stand-in for exactly this lift.
        """
        if self.pos[2] < transit_z:
            self.move_z(transit_z, f"{phase}_lift")
        self.move_xy(np.asarray(target[:2], dtype=np.float64), f"{phase}_cross")
        if target[2] < transit_z:
            self.move_z(float(target[2]), f"{phase}_descend")

    def pierce(self, target_z: float, phase: str) -> None:
        """The DOWN stroke that impales the bale, then a mechanical settle
        dwell. The dwell fires no relay - the spike is a fixed prong and the
        bale is held by friction, so nothing is energized for grip or carry.
        """
        self.move_z(target_z, f"{phase}_stroke")
        self.t_ms += round(self.s.pierce_dwell_s * 1000)
        self._settle()

    def push_off(self, phase: str) -> None:
        """The one relay pulse in the entire grab/carry/release cycle."""
        on_at = self.t_ms
        self.t_ms += round(self.s.push_off_s * 1000)
        self.timeline.append(RelayEvent3D("push", "push", phase, on_at, self.t_ms))
        self._settle()


def verify_interlocks(timeline: list[RelayEvent3D]) -> list[str]:
    """Defensive re-check that opposing relays never overlap in time.

    Should be redundant - each move picks exactly one direction relay, so
    violations are impossible by construction - which is precisely why it is
    worth checking: if this ever fires, the bug is upstream in the builder,
    and catching it here is cheaper than catching it on a real gantry.
    """
    warnings: list[str] = []
    for a, b in OPPOSING_PAIRS:
        for ea in [e for e in timeline if e.relay == a]:
            for eb in [e for e in timeline if e.relay == b]:
                if ea.on_at_ms < eb.off_at_ms and eb.on_at_ms < ea.off_at_ms:
                    warnings.append(
                        f"INTERLOCK VIOLATION: {a} and {b} overlap "
                        f"({ea.on_at_ms}-{ea.off_at_ms}ms vs {eb.on_at_ms}-{eb.off_at_ms}ms)")
    return warnings


def plan_pick_sequence(
    bales: list[Bale3D],
    drop_point: np.ndarray,
    home: np.ndarray,
    speed: SpeedConfig3D | None = None,
    max_bales: int | None = None,
    max_layers: int | None = None,
    layer_band_m: float = 0.5,
) -> Plan3D:
    """Full pick-and-place plan from real 3D positions.

    `home` should come from `mark_reference_position.py --name home` - a real
    measured point through the same pipeline - not a hardcoded constant.
    `drop_point` is the conveyor: every bale goes to the SAME point, because
    the belt carries each one away before the next lands.

    `max_layers` enforces the "crane services the TOP N LAYERS ONLY" rule
    (2 at this plant): bales in lower layers are dropped from the candidate
    set and left on the bed. `max_bales` still applies on top, after the
    layer filter and the height ordering.
    """
    speed = speed or SpeedConfig3D()
    warnings: list[str] = []

    if not bales:
        warnings.append("No bales given - nothing to plan.")
        return Plan3D([], [], 0, 0.0, warnings)

    home = np.asarray(home, dtype=np.float64)
    drop_point = np.asarray(drop_point, dtype=np.float64)

    candidates, n_layers = filter_top_layers(bales, max_layers, layer_band_m)
    if max_layers is not None and n_layers > max_layers:
        warnings.append(
            f"Top-{max_layers}-layer filter: {len(candidates)} of {len(bales)} bales are "
            f"pick candidates; {len(bales) - len(candidates)} in {n_layers - max_layers} "
            f"lower layer(s) left on the bed."
        )

    order = order_bales_by_height(candidates, home)
    if max_bales is not None:
        order = order[:max_bales]

    # Transit height clears the TALLEST bale, not the current one - the hook
    # crosses over bales that have not been picked yet.
    transit_z = max(max(b.position[2] for b in bales), float(home[2]), float(drop_point[2])) + speed.clearance_m

    b = _Builder(speed, home)
    cycles: list[PickCycle3D] = []

    for idx, bale in enumerate(order, start=1):
        tag = f"#{idx}"
        t_start = b.t_ms

        approach = np.array([bale.position[0], bale.position[1], bale.position[2] + speed.approach_m])
        b.safe_move(approach, f"travel_to_bale{tag}", transit_z)
        b.pierce(float(bale.position[2]), f"pierce{tag}")
        b.move_z(transit_z, f"lift_with_load{tag}")
        b.safe_move(drop_point, f"travel_to_drop{tag}", transit_z)
        b.push_off(f"push_off{tag}")

        cycles.append(PickCycle3D(
            order=idx, bale_id=bale.id,
            pick_point=[float(v) for v in bale.position],
            drop_point=[float(v) for v in drop_point],
            t_start_ms=t_start, t_end_ms=b.t_ms,
        ))

    if cycles:
        b.safe_move(home, "return_home", transit_z)

    warnings.extend(verify_interlocks(b.timeline))
    return Plan3D(b.timeline, cycles, b.t_ms, transit_z, warnings)
