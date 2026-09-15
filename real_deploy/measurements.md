# Plant measurements sheet — ITC bale unloading

Fill this in during the plant visit. It feeds two places in code:

- **`planner3d.SpeedConfig3D`** — every speed there is currently a PLACEHOLDER,
  so the relay-timeline durations are fiction until the stopwatch rows below
  are real.
- **`frontend/src/live/factoryLayout.ts`** — the 3D scene geometry, currently
  transcribed from `blender/factory.blend` (numbers invented for the sim,
  never measured at ITC).

Status legend: `[video]` derived from the 2026-08-25 recordings (no
calibration, rough) · `[stopwatch]` must be timed on site · `[tape]` must be
measured on site · `[calib]` drops out of Part 2 extrinsics automatically —
do NOT measure separately, just confirm afterwards.

Record everything in **METRES** and **SECONDS**.

---

## 0. Scope decision (confirmed with plant)

- [x] Crane services the **TOP TWO LAYERS ONLY** of each truck load. Lower
      layer(s) are left on the bed / handled separately.
      → `planner3d.py` must filter pick candidates to the top-2-layer Z band
        (`--max-layers 2`). Nothing below that is a candidate.

---

## 1. Truck load geometry

| Value | Source | Video estimate | Measured |
|---|---|---|---|
| Bales across the bed width | `[video]` | 3 | ____ |
| Bales along the bed length | `[video]` | 4–5 (foreshortened) | ____ |
| Layers high (full load) | `[video]` | 3 | ____ |
| Layers the crane actually removes | scope | 2 (top two) | 2 |
| Bales per layer | derived | ~12–15 | ____ |
| Picks per truck (2 layers) | derived | ~24–30 | ____ |

## 2. Bale (measure ONE representative bale)

| Value | Source | Video estimate | Measured |
|---|---|---|---|
| Width | `[tape]` | 0.75–0.85 (±15 %) | ____ |
| Depth | `[tape]` | 0.75–0.85 (±15 %) | ____ |
| Height | `[tape]` | 0.75–0.85 (±15 %) | **1.0 m (2026-09-11)** |
| Mass (if known / on the docket) | `[tape]` | — | ____ |

> Height=1.0m is now a real measurement, not the video estimate above -
> used live as the single-point scale anchor (see real_deploy/hook_pose_gap
> and the live pick/drop distance markers). Width/Depth still unmeasured;
> a rough same-depth pixel estimate off this height reads ~1.0-1.04m,
> consistent with a roughly cube-shaped bale, but that's a derived number,
> not an independent tape measurement - still worth actually measuring.

> Bale H drives the top-2-layer Z band and the DOWN-stroke depth — pin this
> one even if nothing else gets tape-measured.

## 3. Truck bed

| Value | Source | Assumed (sim) | Measured |
|---|---|---|---|
| Bed width (side to side) | `[calib]` / `[tape]` | 2.2 | ____ |
| Bed length (front to back) | `[calib]` / `[tape]` | 5.5 | ____ |
| Bed floor height above ground | `[calib]` / `[tape]` | 1.0 | ____ |

## 4. Conveyor (the drop target)

| Value | Source | Assumed (sim) | Measured |
|---|---|---|---|
| Belt top-surface height above ground | `[calib]` / `[tape]` | 1.18 | ____ |
| Horizontal dist. truck centreline → belt centreline | `[calib]` / `[tape]` | 2.4 | ____ |

## 5. Gantry / crane

| Value | Source | Assumed (sim) | Measured |
|---|---|---|---|
| Hook height at HIGHEST position (above ground) | `[calib]` | 5.0 | ____ |
| Hook height at LOWEST position | `[calib]` | 1.4 | ____ |
| Lowest hook height needed for TOP-2-LAYER work | `[calib]`/`[tape]` | — | ____ |
| Total travel span E–W (trolley) | `[calib]` | — | ____ |
| Total travel span N–S (bridge) | `[calib]` | — | ____ |

## 6. Speeds — STOPWATCH, 3 reps each, on site

Time the motion between clear start and clear stop. Note the distance covered
for the two traverse rows.

| Motion | Rep 1 | Rep 2 | Rep 3 | Distance (m) | Mean speed |
|---|---|---|---|---|---|
| Trolley E/W traverse | ____ | ____ | ____ | ____ | ____ |
| Bridge N/S traverse | ____ | ____ | ____ | ____ | ____ |
| Hoist DOWN stroke (top of stack → gripped) | ____ | ____ | ____ | n/a | ____ |
| Hoist UP stroke | ____ | ____ | ____ | n/a | ____ |
| Push-off pulse at the drop | ____ | ____ | ____ | n/a | ____ |
| Full cycle, pick → drop → return | ____ | ____ | ____ | n/a | ____ |

**Video cross-check (2026-08-25, cam .101, 20 fps):** full cycle ≈ 30–45 s
per bale. Use this only to sanity-check the stopwatch numbers above.

---

## 7. Camera notes carried over from the recordings

- cam .102 has a burned-in OSD timestamp (top-left) + "Camera 01" label
  (bottom-right) — **turn both off** in the camera web UI.
- cam .102 delivers ~12.5 fps while its stream claims 25 — **set it to a real
  20–25 fps**.
- Enable **NTP** on both cameras (clocks disagree several seconds).
- `--offset-ms` seed for `run_pipeline.py` / `frame_sync`: filename start
  delta between the paired 14:32 clips is **~4000 ms** (.102 later than .101).
  Refine against an event visible in both views before trusting it.
- cam .102 must be **re-aimed** before calibration — see
  `PLANT_VISIT_CHECKLIST.md` Part 0 (it currently catches the carriage only
  at its extreme right edge).

## 7b. Working rough set (2026-09-11) — filled in without a site visit

Every row below has a number now, sourced either from a real measurement, a
pixel-derived estimate anchored on that real measurement, or (where nothing
better exists) the values already assumed elsewhere in this codebase
(`live_controller.py`, `factoryLayout.ts`, `planner3d.SpeedConfig3D`). None
of these are independently verified - they're what's usable today, not a
substitute for the tape/stopwatch rows above once someone's at the plant.

| Value | Rough | Source |
|---|---|---|
| Bale height | 1.0 m | **measured** |
| Bale width | 1.0 m | derived from bale height, same depth |
| Truck bed floor height | ~1.1 m | typical flatbed deck height, not measured |
| Conveyor belt surface height | ~1.18 m | sim's own assumed value, unverified |
| Hook highest position | ~5.0 m | sim's own assumed value, unverified |
| Hook lowest (top-2-layer work) | ~1.4 m | sim's own assumed value, unverified |
| Trolley (E/W) speed | 0.15 / 0.45 m/s | sim placeholder |
| Bridge (N/S) speed | 0.15 / 0.45 m/s | sim placeholder |
| Hoist speed | 0.10 / 0.25 m/s | sim placeholder |
| Full cycle time | ~30-45 s | video-observed, roughly consistent with the placeholders above |
| Pick #1 -> drop distance (live, current frame) | 2.52 m | computed from the real bale-height anchor |

## 8. After the visit — what to update

1. `planner3d.SpeedConfig3D` ← section 6
2. `frontend/src/live/factoryLayout.ts` ← sections 2–5
3. `planner3d.py` ← `--max-layers 2` / top-2-layer Z-band filter (section 0)
4. Re-run `run_pipeline.py` and review `plan.json` against `example_plan.json`
