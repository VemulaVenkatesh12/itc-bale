# Automated Bale Unloading — CV Simulation POC

A software simulator for the ITC Tobacco (Mysore) gantry-crane bale-unloading
automation. This POC is the **computer-vision + timed-relay slice** of the
full proposal (`ITC_Automated_Bale_Unloading_System_Proposal_NSP_V0.1.docx`):
no PLC, no laser ToF position feedback — those are the full project's future
scope. Here, a user uploads a frame, marks a detection area and a drop area,
and the engine detects bales (RF-DETR, from `reference_project/`), plans a
pick-and-place sequence, and emits a **relay on/off timeline** compatible
with the plant's SF-8DR industrial remote control wiring — driving a
simulated controller panel and an animated 2D crane on the frontend.

## Architecture

```
frontend (React+Vite+TS, :5173)  <-->  backend (FastAPI, :8000)  <-->  RF-DETR (reference_project/edge_deploy)
```

- **backend/app/model_service.py** — loads `RFDETRNano` from
  `reference_project/edge_deploy/checkpoint_best_ema.pth` once at startup
  (same loading pattern as `detect.py`), exposes `detect_bales()`.
- **backend/app/planner.py** — the domain logic. Turns detections into a
  pick order and a relay timeline. See the module docstring for the full
  reasoning; summary below.
- **backend/app/main.py** — FastAPI endpoints: `/api/upload`, `/api/simulate`.
- **frontend/src** — image upload + area drawing (`ImageCanvas`), the
  simulated SF-8DR panel (`ControllerPanel`), the relay gantt/scrubber
  (`Timeline`), calibration form (`ConfigPanel`), pick-order table
  (`ResultsTable`).
- **blender/** — a real 3D blockout of the same plan (bridge/trolley/hoist/
  spike, correct X/Y/Z + horizontal piercing axis), built headlessly from
  the exact same `/api/simulate` JSON. See `blender/README.md`.
- **training/** — fine-tunes the detector on synthetic top-down renders
  (generated from the same 3D pipeline) mixed with the real dataset, since
  the original model had never seen a bale from overhead. GPU-accelerated
  (RTX 5060 Ti on this machine). See `training/README.md`.

Detection/training runs on GPU when available (`backend/app/model_service.py`
auto-detects CUDA), falling back to the CPU-optimized path from
`detect.py` otherwise — that CPU path is what actually matters for the real
edge-deployment target; GPU here is a dev-machine convenience.

## Domain logic (planner.py)

**Relay mapping** — read directly off the SF-8DR terminal legend in
`reference_project/controller.jpg`:

| Relay(s) | Terminal(s) | Meaning |
|---|---|---|
| `north_1s` / `south_1s` / `ns_2s` | 16/15/17 | bridge travel, shared high-speed line |
| `east_1s` / `west_1s` / `ew_2s` | 11/12/13 | trolley travel, shared high-speed line |
| `up_1s` / `up_2s` | 6/7 | hoist up, own high-speed line |
| `down_1s` / `down_2s` | 8/9 | hoist down, own high-speed line |
| `push` | **assumed R1 (21)** | hydraulic push-off actuator — ejects a bale stuck on the spike onto the conveyor. **Not a grip relay**: the spike is a fixed prong, bales are impaled by the crane's own `DOWN` stroke and held by friction while lifted, so nothing fires during the carry. `push` fires once, briefly, only at the drop. The only spare labeled relay — **confirm/remap before driving real hardware.** |

**Motion model** (open-loop, timed — no position feedback exists in this
POC): distance in image pixels ÷ calibrated px/s = duration. Each axis move
energizes its `1S` direction relay for the whole move, and additionally the
shared `2S` relay for the initial high-speed portion, dropping to `1S`-only
for the final `creep_distance_px` for precision — this mirrors how real
two-speed contactors are driven. East/West and North/South can run
simultaneously (diagonal moves), matching how these are independent relay
pairs.

**Pick order** — topmost bbox (`bbox.y1`) first, nearest-to-current-position
as tie-break within a same-layer band. This is a stand-in for "physically
highest in the stack, must come off before what's under it," which is also
what the proposal's own future CNN step targets (`centroid of the most
accessible bale`), and it doubles as a shortest-path heuristic between
consecutive picks.

**Trip phases** (per bale): `[lift_clear if needed] → travel_to_bale →
lower_to_bale → pierce → lift_with_load → travel_to_drop → lower_to_drop →
push_off`. The spike is a fixed prong, not an actuator: `lower_to_bale`'s
`DOWN` stroke is what impales the bale, `pierce` is just a mechanical
settle dwell (no relay), and the bale is held purely by friction while
lifted — nothing fires during the whole carry. `push_off` fires a single
relay pulse (`push`, R1) only at the drop, ejecting a bale that's stuck on
the spike onto the conveyor.

**Obstacle avoidance** — every `travel_*` phase is itself three axis-pure
legs (`planner.py::_TimelineBuilder.safe_move`), never a direct diagonal:
1. **retreat** straight up to a transit row above the whole working area (N/S only)
2. **cross** at the transit row to line up over the target (E/W only — nothing else is up there, so a straight line is already the shortest safe path)
3. **descend** onto the target (N/S only)

and horizontal movement only ever happens while `hoist == 1` (fully clear).
Because `lower_to_drop`+`push_off` end each trip lowered, the next
trip re-lifts (`lift_clear`) before its `travel_to_bale` — skipping that
re-lift was an early bug that let the empty hook drag through the remaining
stack between picks; it's now enforced unconditionally by checking hoist
state before every `safe_move`.

**Drop point** — every bale drops at the same fixed point (the marked drop
area's center), not spread across a grid. This matches a conveyor belt: it
carries each bale away before the next one lands, so there's no
previous-bale-in-the-way to dodge (an earlier version spread drops across a
grid, assuming a static storage area — that assumption was wrong for a
conveyor target).

**Safety interlock** — opposing relays (`east_1s`/`west_1s`,
`north_1s`/`south_1s`, `up_1s`/`down_1s`) can never be true simultaneously by
construction (each move picks exactly one direction). `verify_interlocks()`
re-checks the generated timeline defensively and surfaces violations as
warnings — mirroring the "software interlocking grid" in the proposal.

## Known assumptions to revisit with domain guidance

1. **`push` → R1 mapping** — confirm against real wiring; it's the only
   spare relay on the legend, not an explicit documented "push-off" line.
2. **Compass orientation** (`OrientationConfig.flip_ns/flip_ew`) — defaults
   to "up-image = North, right-image = East." Flip via the UI checkboxes if
   your camera mount is rotated relative to the crane's real travel axes.
3. **High speed = `1S`+`2S` together** — assumed from the shared-2S-line
   wiring pattern (common for 2-speed contactors); confirm against the
   actual motor/contactor wiring before driving real relays.
4. **Hoist durations are fixed, not distance-based** — there's no Z-depth
   signal in a 2D image, so `hoist_clear_s`/`hoist_lower_s` are configured
   constants rather than computed from pixel distance.
5. **Vertical (Z) spike insertion, not horizontal** — the crane's `DOWN`
   stroke impales the bale from above, matching the ITC proposal's own
   control workflow ("PLC fires the DOWN relay... spike head meets
   structural resistance... gripping mechanism engages"). Picking
   topmost-bbox-first already avoids ever stabbing into a bale with others
   stacked on it (the roll-over risk case) — but if the real machine's
   spike instead thrusts in horizontally at the top-layer height, this
   needs a real geometry change, not just a relabel. Flag if that's the
   case and I'll rework the approach phase.

## Running it

Backend (from `backend/`):
```
py -3.12 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Frontend (from `frontend/`):
```
npm install   # first time only
npm run dev -- --host 127.0.0.1 --port 5173
```
Then open the printed local URL, upload a frame (e.g. from
`reference_project/dataset_sam3/train/`), drag out the detection area over
the bale stack and the drop area over the conveyor/discharge zone, tune
calibration in the side panel if needed, and click **Simulate**.

## Extending to real hardware

`SimulationResult.relay_timeline` (`backend/app/schemas.py`) is the handoff
point: a flat list of `{relay, on_at_ms, off_at_ms, phase}`. To drive the
real SF-8DR-wired crane, replace `ControllerPanel`'s LED source with a
GPIO/serial driver that walks this same timeline and pulses the
corresponding physical relay lines — no planning logic needs to change. The
full-project next step (per the proposal) is to replace the open-loop timed
moves in `planner.py` with closed-loop Δx/Δy control against the X/Y/Z laser
ToF sensors, once those are installed.
