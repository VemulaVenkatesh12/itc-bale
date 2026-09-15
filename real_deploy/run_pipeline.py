"""
END-TO-END DRY RUN: two real cameras -> time-synced frames -> detection ->
cross-camera match -> triangulation -> real 3D bale/hook positions -> pick
plan -> SF-8DR relay timeline (written to JSON).

This is the piece that finally connects the two halves of the POC: the
simulation's planner (`backend/app/planner.py`) which knew how to emit a
relay timeline but only understood image pixels, and `real_deploy/`'s vision
stack which recovers real metric 3D but deliberately stopped at observing.

**Nothing here actuates anything.** No relay, no PLC, no Modbus write. The
output is a JSON plan for a human to review. That boundary is not an
oversight - `README.md` lists a PLC driver, a closed-loop control law, and a
safety system as the three things that must exist before any of this is
allowed to move real machinery, and none of them exist yet.

## The survey-then-plan structure, and why it is not a live control loop

Bales do not move. So rather than replanning every frame against whatever
that frame happened to detect, this SURVEYS for a window of frames,
accumulates every triangulated bale observation, clusters them, and takes
the MEDIAN of each cluster as that bale's position. One noisy frame then
cannot move a bale by half a metre in the plan; an outlier is outvoted
rather than averaged in (the same reason `mark_reference_position.py` takes
a median rather than a mean).

The hook is tracked separately through `position_tracker.py`, because unlike
the bales it does move, and its smoothed position is what tells the planner
where the crane actually is right now.

## Usage

    py run_pipeline.py \\
        --cam1-calibration calib_out/cam101.json --source1 ../192.168.1.101_....mp4 \\
        --cam2-calibration calib_out/cam102.json --source2 ../192.168.1.102_....mp4 \\
        --checkpoint ../training/output_real/checkpoint_best_ema.pth \\
        --reference-positions reference_positions.json \\
        --out plan.json

`--offset-ms` corrects a measured constant clock skew between the two
sources - see frame_sync.py. Measure it; do not guess it.

## Prerequisites this will refuse to run without

Both calibration JSONs must carry EXTRINSICS in a shared world frame
(`calibrate_extrinsics.py`), and `reference_positions.json` must contain
real measured `home` and `drop_point` entries
(`mark_reference_position.py`). Those are physical-measurement steps that
cannot be faked in software, and a plan built on invented values would be
confidently wrong in metres.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from calibration import CameraCalibration
from capture_infer import CLASS_NAMES, _detect, _load_model, _match_detections
from frame_sync import SyncedFrameReader
from planner3d import Bale3D, SpeedConfig3D, plan_pick_sequence
from position_tracker import PositionTracker

BALE_CLASS_ID = 0
HOOK_CLASS_ID = 1


def _is_placeholder_pose(cal: CameraCalibration) -> bool:
    """True if this calibration never went through calibrate_extrinsics.py.

    `CameraCalibration.intrinsics_only` fills rotation/translation with an
    identity/zero PLACEHOLDER rather than leaving them unset, so a missing
    extrinsic calibration cannot be caught with a None check - it has to be
    recognized by that exact signature.
    """
    return bool(
        np.allclose(cal.rotation, np.eye(3), atol=1e-9)
        and np.allclose(cal.translation, np.zeros(3), atol=1e-9)
    )


def cluster_observations(
    points: list[np.ndarray], radius_m: float, min_observations: int
) -> tuple[list[np.ndarray], int]:
    """Greedy spatial clustering of repeated observations of static bales.

    Returns (median positions, number of clusters rejected for being too
    sparse). A cluster seen only once or twice across the whole survey is far
    more likely to be a cross-camera mismatch than a real bale, so
    `min_observations` discards it rather than planning a pick against it -
    a spurious bale means the crane drives somewhere empty and stabs at
    nothing.
    """
    clusters: list[list[np.ndarray]] = []
    for p in points:
        for c in clusters:
            if np.linalg.norm(np.median(np.array(c), axis=0) - p) <= radius_m:
                c.append(p)
                break
        else:
            clusters.append([p])

    kept = [np.median(np.array(c), axis=0) for c in clusters if len(c) >= min_observations]
    rejected = len(clusters) - len(kept)
    return kept, rejected


def survey(reader, model, cam1_cal, cam2_cal, threshold, max_frames, tracker) -> tuple[list[np.ndarray], dict]:
    """Runs the vision chain over synced frame pairs, collecting every
    triangulated bale observation and folding hook sightings into `tracker`."""
    bale_points: list[np.ndarray] = []
    stats = defaultdict(int)

    for n, (pair, frame_a, frame_b) in enumerate(reader, start=1):
        if max_frames is not None and n > max_frames:
            break
        stats["frames"] += 1

        dets1 = _detect(model, frame_a, threshold)
        dets2 = _detect(model, frame_b, threshold)
        matches = _match_detections(cam1_cal, dets1, cam2_cal, dets2)
        stats["matched"] += len(matches)

        hook_this_frame = None
        for m in matches:
            if m.class_id == BALE_CLASS_ID:
                bale_points.append(m.world_point)
                stats["bale_obs"] += 1
            elif m.class_id == HOOK_CLASS_ID:
                # Keep the most confident hook match if several came back.
                if hook_this_frame is None or m.confidence > hook_this_frame.confidence:
                    hook_this_frame = m

        now_s = pair.t_a_ms / 1000.0
        if hook_this_frame is not None:
            stats["hook_obs"] += 1
            tracker.update(hook_this_frame.world_point, hook_this_frame.confidence, now_s)
        else:
            tracker.update(None, 0.0, now_s)

        if n % 25 == 0:
            print(f"  [{n}] skew={pair.skew_ms:+.0f}ms  bale_obs={stats['bale_obs']}  hook_obs={stats['hook_obs']}")

    return bale_points, dict(stats)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cam1-calibration", required=True)
    p.add_argument("--source1", required=True)
    p.add_argument("--cam2-calibration", required=True)
    p.add_argument("--source2", required=True)
    p.add_argument("--checkpoint", required=True, help="A REAL-data-trained checkpoint, not the simulation's")
    p.add_argument("--reference-positions", required=True, help="JSON from mark_reference_position.py")
    p.add_argument("--out", required=True, help="Where to write the relay-timeline plan")
    p.add_argument("--resolution", type=int, default=640)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-skew-ms", type=float, default=40.0)
    p.add_argument("--offset-ms", type=float, default=0.0, help="Measured clock skew of source2 vs source1")
    p.add_argument("--survey-frames", type=int, default=200)
    p.add_argument("--cluster-radius-m", type=float, default=0.40)
    p.add_argument("--min-observations", type=int, default=3)
    p.add_argument("--max-bales", type=int, default=None)
    p.add_argument("--max-layers", type=int, default=2,
                   help="Plan only the top N physical layers (this plant services the top 2). "
                        "Pass 0 or a negative value to disable the layer filter.")
    p.add_argument("--layer-band-m", type=float, default=0.5,
                   help="Z spread treated as one layer - larger than triangulation noise, "
                        "smaller than one bale height.")
    args = p.parse_args()

    cam1_cal = CameraCalibration.load(args.cam1_calibration)
    cam2_cal = CameraCalibration.load(args.cam2_calibration)
    for cal, path in ((cam1_cal, args.cam1_calibration), (cam2_cal, args.cam2_calibration)):
        if _is_placeholder_pose(cal):
            raise SystemExit(
                f"{path} still has calibrate_intrinsics.py's PLACEHOLDER pose "
                "(identity rotation, zero translation) - run calibrate_extrinsics.py first.\n"
                "Without a shared world frame there is nothing to triangulate INTO: two cameras "
                "both sitting at the world origin have zero baseline, which is the degenerate case "
                "that made an earlier smoke test print positions at planetary distances."
            )

    references = json.loads(Path(args.reference_positions).read_text())
    missing = [k for k in ("home", "drop_point") if k not in references]
    if missing:
        raise SystemExit(
            f"reference positions missing: {missing}\n"
            "Park the crane at each and run mark_reference_position.py --name <name>. "
            "These must be REAL measured points - inventing them puts the plan metres off."
        )
    home = np.array(references["home"], dtype=np.float64)
    drop_point = np.array(references["drop_point"], dtype=np.float64)

    print("indexing both sources (full decode pass each, to read real timestamps)...")
    reader = SyncedFrameReader(args.source1, args.source2, args.max_skew_ms, args.offset_ms)
    print(reader.describe())
    if not reader.pairs:
        raise SystemExit(
            "no frame pairs within tolerance - the two sources do not overlap in time.\n"
            "Check --offset-ms against a real event visible in both views."
        )

    print(f"\nloading model: {args.checkpoint}")
    model = _load_model(args.checkpoint, args.resolution)

    tracker = PositionTracker(alpha=0.3, stale_after_s=1.0)
    print(f"\nsurveying up to {args.survey_frames} synced frame pairs...")
    bale_points, stats = survey(
        reader, model, cam1_cal, cam2_cal, args.threshold, args.survey_frames, tracker)

    positions, rejected = cluster_observations(bale_points, args.cluster_radius_m, args.min_observations)
    print(f"\n{stats.get('bale_obs', 0)} bale observations -> {len(positions)} bales "
          f"({rejected} sparse clusters rejected as probable mismatches)")

    hook = tracker.update(None, 0.0, now_s=1e9)  # force a staleness read without folding in data
    if hook.position is None:
        print("WARNING: the hook was never observed - planning from the measured `home` instead.")
        start = home
    elif hook.stale:
        print("WARNING: hook estimate is stale - planning from the measured `home` instead.")
        start = home
    else:
        start = hook.position
        print(f"hook last seen at ({start[0]:+.3f}, {start[1]:+.3f}, {start[2]:+.3f}) m")

    if not positions:
        raise SystemExit("no bales survived clustering - nothing to plan. Check detection threshold/calibration.")

    bales = [Bale3D(id=i, position=pos) for i, pos in enumerate(positions, start=1)]
    max_layers = args.max_layers if args.max_layers and args.max_layers > 0 else None
    plan = plan_pick_sequence(
        bales, drop_point, start, SpeedConfig3D(),
        max_bales=args.max_bales, max_layers=max_layers, layer_band_m=args.layer_band_m,
    )

    out = plan.to_dict()
    out["provenance"] = {
        "source1": args.source1, "source2": args.source2, "checkpoint": args.checkpoint,
        "frames_surveyed": stats.get("frames", 0), "bale_observations": stats.get("bale_obs", 0),
        "hook_observations": stats.get("hook_obs", 0), "sparse_clusters_rejected": rejected,
        "planned_from": "hook" if start is not home else "home_reference",
    }
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"\n{len(plan.pick_cycles)} picks, {len(plan.relay_timeline)} relay events, "
          f"{plan.total_duration_ms / 1000:.1f}s total")
    for w in plan.warnings:
        print(f"  WARNING: {w}")
    print(f"plan written to {args.out}")
    print("\nDRY RUN - no relay, PLC or actuator was touched. Review this plan before")
    print("it is ever fed to hardware; speeds in SpeedConfig3D are placeholders until measured.")


if __name__ == "__main__":
    main()
