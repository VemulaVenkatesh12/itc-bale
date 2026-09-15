"""
Verifies planner3d.py's ordering, timing and SAFETY invariants against
known geometry - pure math, no camera or crane needed.
Run: py tests/test_planner3d.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner3d import (
    Bale3D,
    SpeedConfig3D,
    filter_top_layers,
    group_into_layers,
    order_bales_by_height,
    plan_pick_sequence,
    verify_interlocks,
)

HOME = np.array([0.0, 0.0, 3.0])
DROP = np.array([5.0, 0.0, 1.0])


def _stack():
    return [
        Bale3D(id=1, position=np.array([1.0, 1.0, 0.5])),
        Bale3D(id=2, position=np.array([1.0, 2.0, 2.0])),  # tallest
        Bale3D(id=3, position=np.array([2.0, 1.0, 1.2])),
    ]


def test_ordering_is_by_real_height_not_proximity():
    """The 2D planner could only approximate this with bbox.y1. Bale 2 is
    farther from home but physically highest, so it must come off first."""
    order = order_bales_by_height(_stack(), HOME)
    assert [b.id for b in order] == [2, 3, 1]


def test_same_layer_ties_break_by_distance():
    bales = [
        Bale3D(id=1, position=np.array([9.0, 0.0, 2.00])),
        Bale3D(id=2, position=np.array([1.0, 0.0, 1.95])),  # within z tolerance, much nearer
    ]
    order = order_bales_by_height(bales, HOME, z_tolerance_m=0.20)
    assert [b.id for b in order] == [2, 1]


def test_z_tolerance_does_not_swallow_a_real_layer():
    bales = [
        Bale3D(id=1, position=np.array([9.0, 0.0, 2.0])),
        Bale3D(id=2, position=np.array([1.0, 0.0, 1.0])),  # a full layer lower
    ]
    order = order_bales_by_height(bales, HOME, z_tolerance_m=0.20)
    assert [b.id for b in order] == [1, 2]


def test_horizontal_travel_only_ever_happens_at_transit_height():
    """THE safety invariant: the hook must never sweep sideways at bale
    height through the remaining stack. Every horizontal relay event must
    therefore belong to a `_cross` phase, which the builder only emits after
    lifting to transit height."""
    plan = plan_pick_sequence(_stack(), DROP, HOME)
    horizontal = [e for e in plan.relay_timeline if e.axis in ("ew", "ns")]
    assert horizontal, "expected horizontal moves"
    offenders = [e for e in horizontal if not e.phase.endswith("_cross")]
    assert not offenders, f"horizontal movement outside transit: {offenders[:3]}"


def test_transit_height_clears_the_tallest_bale():
    speed = SpeedConfig3D()
    bales = _stack()
    plan = plan_pick_sequence(bales, DROP, HOME, speed)
    tallest = max(b.position[2] for b in bales)
    assert plan.transit_z >= tallest + speed.clearance_m - 1e-9
    assert plan.transit_z >= HOME[2]


def test_generated_plan_never_violates_interlocks():
    plan = plan_pick_sequence(_stack(), DROP, HOME)
    assert verify_interlocks(plan.relay_timeline) == []
    assert plan.warnings == []


def test_opposing_relays_are_caught_when_they_do_overlap():
    """Guards the guard - verify_interlocks must actually detect a violation,
    not just always return clean."""
    from planner3d import RelayEvent3D
    bad = [
        RelayEvent3D("east_1s", "ew", "x", 0, 1000),
        RelayEvent3D("west_1s", "ew", "x", 500, 1500),
    ]
    assert len(verify_interlocks(bad)) == 1


def test_hoist_duration_scales_with_distance():
    """The 2D planner's known assumption #4 was a FIXED hoist time because a
    2D image has no Z. Triangulation removes that limitation."""
    near = [Bale3D(id=1, position=np.array([1.0, 0.0, 2.9]))]
    far = [Bale3D(id=1, position=np.array([1.0, 0.0, 0.1]))]
    t_near = plan_pick_sequence(near, DROP, HOME).total_duration_ms
    t_far = plan_pick_sequence(far, DROP, HOME).total_duration_ms
    assert t_far > t_near, f"deeper pick should take longer ({t_far} vs {t_near})"


def test_push_fires_exactly_once_per_bale_and_only_at_the_drop():
    bales = _stack()
    plan = plan_pick_sequence(bales, DROP, HOME)
    pushes = [e for e in plan.relay_timeline if e.relay == "push"]
    assert len(pushes) == len(bales)
    assert all(e.phase.startswith("push_off") for e in pushes)


def test_nothing_is_energized_during_the_carry():
    """The spike is a fixed prong - bales are held by friction, so no relay
    fires for grip or hold. Between a pierce and its lift there must be no
    relay activity other than the hoist itself."""
    plan = plan_pick_sequence([_stack()[0]], DROP, HOME)
    grip_like = [e for e in plan.relay_timeline if e.relay not in
                 ("east_1s", "west_1s", "ew_2s", "north_1s", "south_1s", "ns_2s",
                  "up_1s", "up_2s", "down_1s", "down_2s", "push")]
    assert not grip_like, f"unexpected relay: {grip_like}"


def test_two_speed_pattern_holds_direction_relay_for_the_whole_move():
    """1S is energized for the entire move; 2S only for the initial fast
    portion, dropping out for the final creep."""
    far = [Bale3D(id=1, position=np.array([20.0, 0.0, 1.0]))]
    plan = plan_pick_sequence(far, DROP, HOME)
    crosses = [e for e in plan.relay_timeline if e.phase.endswith("_cross") and e.axis == "ew"]
    ones = [e for e in crosses if e.relay == "east_1s"]
    twos = [e for e in crosses if e.relay == "ew_2s"]
    assert ones and twos
    assert twos[0].on_at_ms == ones[0].on_at_ms
    assert twos[0].off_at_ms < ones[0].off_at_ms, "2S must drop out before 1S for the creep"


def test_short_move_skips_high_speed_entirely():
    speed = SpeedConfig3D()
    tiny = [Bale3D(id=1, position=np.array([0.1, 0.0, 3.0 - speed.clearance_m]))]
    plan = plan_pick_sequence(tiny, DROP, HOME, speed)
    early = [e for e in plan.relay_timeline if e.phase == "travel_to_bale#1_cross"]
    assert not [e for e in early if e.relay == "ew_2s"], "2S should not fire for a sub-threshold move"


def test_cycles_are_sequential_and_non_overlapping():
    plan = plan_pick_sequence(_stack(), DROP, HOME)
    assert len(plan.pick_cycles) == 3
    for prev, nxt in zip(plan.pick_cycles, plan.pick_cycles[1:]):
        assert prev.t_end_ms <= nxt.t_start_ms
    assert plan.pick_cycles[-1].t_end_ms <= plan.total_duration_ms


def test_max_bales_limits_the_plan():
    plan = plan_pick_sequence(_stack(), DROP, HOME, max_bales=2)
    assert len(plan.pick_cycles) == 2


def _three_layer_load():
    """3 across x 2 deep x 3 high, layer tops at Z = 2.4 / 1.6 / 0.8."""
    bales = []
    bid = 1
    for z in (2.4, 1.6, 0.8):
        for x in (1.0, 1.8, 2.6):
            for y in (1.0, 1.8):
                bales.append(Bale3D(id=bid, position=np.array([x, y, z + 0.03 * (bid % 3)])))
                bid += 1
    return bales


def test_group_into_layers_finds_three_layers():
    layers = group_into_layers(_three_layer_load(), layer_band_m=0.5)
    assert [len(l) for l in layers] == [6, 6, 6]
    assert min(b.position[2] for b in layers[0]) > max(b.position[2] for b in layers[1])


def test_filter_top_layers_keeps_only_the_top_two():
    kept, n_layers = filter_top_layers(_three_layer_load(), max_layers=2, layer_band_m=0.5)
    assert n_layers == 3
    assert len(kept) == 12
    assert all(b.position[2] > 1.0 for b in kept)  # nothing from the Z~0.8 bottom layer


def test_filter_top_layers_disabled_keeps_everything():
    kept, n_layers = filter_top_layers(_three_layer_load(), max_layers=None)
    assert len(kept) == 18


def test_max_layers_limits_the_plan_and_warns():
    load = _three_layer_load()
    plan = plan_pick_sequence(load, DROP, HOME, max_layers=2)
    assert len(plan.pick_cycles) == 12
    assert any("Top-2-layer" in w for w in plan.warnings)
    picked_z = [c.pick_point[2] for c in plan.pick_cycles]
    assert all(z > 1.0 for z in picked_z)


def test_max_layers_none_plans_the_whole_load():
    plan = plan_pick_sequence(_three_layer_load(), DROP, HOME, max_layers=None)
    assert len(plan.pick_cycles) == 18
    assert not any("layer" in w for w in plan.warnings)


def test_empty_input_warns_and_plans_nothing():
    plan = plan_pick_sequence([], DROP, HOME)
    assert plan.relay_timeline == []
    assert plan.total_duration_ms == 0
    assert any("No bales" in w for w in plan.warnings)


def test_plan_serializes_to_plain_json_types():
    import json
    plan = plan_pick_sequence(_stack(), DROP, HOME)
    json.dumps(plan.to_dict())  # must not raise on numpy types


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
