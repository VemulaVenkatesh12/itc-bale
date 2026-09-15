# 3D (Blender)

Two things live here, built at different times for different purposes:

1. **`scene.blend`** — a headless, one-shot animated blockout of the
   pick-and-place *cycle*, generated straight from the same
   `/api/simulate` timeline the 2D web app uses. Good for validating the
   planner's motion logic in 3D. Built by `build_scene.py` / `generate.py`,
   documented below.
2. **`factory.blend`** — a live-built factory *scene*: truck bed loaded
   with bales textured from real photo crops, one conveyor (raised
   platform), a top camera and a back/ground camera (placed
   per the ITC proposal's camera-zone description), and a static crane
   rig. Built interactively through a live MCP connection to a running
   Blender GUI, not headlessly. This is the base the real-time reactive
   controller (next phase) will run against — the two synthetic camera
   feeds are what that controller reads. See "Live MCP-driven factory
   scene" below.

## Requirements

- The backend running (`../README.md`).
- Blender 5.x. This was built/tested against `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`.
  Pass `--blender <path>` to `generate.py` if yours lives elsewhere.

## Live MCP-driven factory scene (`factory.blend`)

This uses [ahujasid/blender-mcp](https://github.com/ahujasid/blender-mcp) —
fetched directly from GitHub, **not the PyPI `blender-mcp` package**, which
phones prompt text (and, per its config, screenshots) to an undisclosed
third-party Supabase project by default. The real upstream project has the
same telemetry hook but its actual credentials file is deliberately
gitignored upstream ("local config secrets"), so it's inert when run from a
plain clone; `BLENDER_MCP_DISABLE_TELEMETRY=true` is set as well, belt and
braces. If you're setting this up on a new machine:

```
git clone https://github.com/ahujasid/blender-mcp.git blender/_blender_mcp_src
cp blender/_blender_mcp_src/addon.py "%APPDATA%\Blender Foundation\Blender\5.2\scripts\addons\blender_mcp.py"
cd blender/_blender_mcp_src && py -3.12 -m pip install -e .

# launch Blender's GUI with the addon enabled + server started (see _mcp_startup.py)
"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --python ..\_mcp_startup.py

# register the MCP server (local scope = machine-specific, not committed)
claude mcp add -e BLENDER_MCP_DISABLE_TELEMETRY=true -- blender py -3.12 -m blender_mcp.server
```

Then run `/mcp` in Claude Code (or restart) to load the `blender` tools.
Blender's GUI has to stay running for the connection to work — it's a live
socket server (`localhost:9876`), not a one-shot script.

**Real bale textures**: `crop_bales.py` (plain Python + Pillow, not run
inside Blender) crops each detected bale's bbox out of the source image
into `crops/bale_<id>.png`, using the cached `_last_sim.json`. Those get
loaded as Blender image textures and mapped onto each bale cube, so the
factory scene shows the *actual* photographed bales, not flat color.

```
py crop_bales.py
```

UV mapping matters here: **every face's UVs are set to the full 0–1 image
square**, not `smart_project`. `smart_project` splits a cube into 6 UV
islands and packs them into different regions of the *same* source image,
so each face ends up showing a distorted fragment instead of the whole
photo — looks wrong and, worse, isn't something a bale detector trained on
real photos will recognize. Full-face mapping means every face shows the
complete, undistorted crop (minor squash to fit a square face, nothing
like the fragment distortion).

**Layout** (matches `reference_project/dataset_sam3/train/frame_0040.jpg`,
the clearest overview shot of the real setup — check that frame if the
geometry below needs revisiting):
- Truck's long axis runs along **Y** (depth, away from the open/pick end)
  — same axis the crane's long-travel rail runs along, i.e. **Y = N/S**.
  Cross-travel (trolley) is **X = E/W**, a short reach from over the truck
  to the conveyor beside it. This matches `planner.py`'s relay convention,
  which the real-time controller (next phase) will reuse directly.
- **Conveyor is a raised platform** (deck + stair access), not a belt or
  roller table — matches `frame_0000.jpg`'s machine. Positioned right
  beside the truck (small X offset), running parallel to it, deck at
  bale-drop height (~1.1m) — not a distant separate structure. (An earlier
  pass added a large cylindrical drum on top guessing at the real
  feeder mechanism — turned out wrong, removed; deck + stairs only now.)
- **Truck bed sits ~0.7m off the ground** on wheels + a simple chassis
  rail per side — was originally modeled sitting almost on the ground.
- **Crane rail support posts** have generous clearance from the truck/
  conveyor (`Y_MARGIN=3.5m` beyond the truck's ends, `X_MARGIN=1.2m`
  beyond the trolley's travel extremes in `build_scene`'s live-session
  equivalent) — tighter margins put posts directly in both cameras'
  sightlines, occluding bales.
- **`Camera_Top`**: TRUE overhead, centered above both the truck and the
  conveyor, well above the crane rails (`RAIL_HEIGHT + 4m`), looking
  straight down — gives a clean (X, Y) read of every bale. An earlier
  45°-from-the-back version had the crane's own rail-support posts sitting
  directly in its sightline; going straight overhead sidesteps that.
- **`Camera_Back`**: ground level, at the truck's open (= "back" in
  standard trucking terms — cab end is the front) end, a *level* view (not
  tilted down into the pile) — gives a clean (X, Z) read of lateral
  position and stack height, complementing `Camera_Top`'s (X, Y). This is
  roughly `frame_0040.jpg`'s own vantage point.

## Headless animated blockout (`scene.blend`)

### Usage

```
cd blender
py generate.py --image ..\reference_project\dataset_sam3\train\frame_0000.jpg ^
    --detection-area 850,250,1280,650 --drop-area 50,550,350,680 --open
```

This uploads the image, runs detection + planning, then invokes Blender
headlessly to build `scene.blend` (add `--out` to change the path). `--open`
launches Blender's GUI on the result once it's built. Scrub the timeline at
the bottom of Blender's window, or hit spacebar to play.

Useful flags: `--crane-home x,y`, `--threshold`, `--max-bales N` (handy for
a quick 2-3-bale scene instead of a full cycle), `--bale-width-m` (real bale
width in meters — everything else scales off this).

## What you're looking at

- **Blue**: the gantry — two rail posts (static), the bridge (moves along
  Y/N-S), the trolley (rides the bridge, moves along X/E-W), the mast+hook
  (vertical Z/hoist), and the spike (thin cylinder, thrusts along local -Y
  when piercing).
- **Tan boxes**: detected bales, sized to their real detected footprint.
  The ones scheduled this cycle are a slightly different shade.
- **Dark box**: the conveyor at the drop point.
- **White marker**: crane home/parking position.

Grip has no dedicated relay/animation of its own — the bale simply starts
following the hook's exact position from the moment `thrust` reaches 1.0
(the pierce) until `push_off` fires and it settles at the drop point. That
mirrors the real mechanism: a fixed spike, no gripper actuator.

## Coordinate mapping / assumptions (`build_scene.py` header has the full detail)

- World X = image X (E/W), World Y = -image Y (N/S), World Z = hoist scaled
  onto `[Z_GRAB, Z_CLEAR]` meters.
- The spike's thrust direction is a **fixed local axis** (-Y on the hook
  rig), independent of travel direction — confirm this matches your real
  spike's mounted orientation.
- Scale is derived from the average detected bale width vs. `--bale-width-m`
  (default 1.0m) — adjust if real bales are a different size.
- This is a blockout: primitives + flat colors, meant to validate that the
  pick-and-place motion makes physical sense in 3D, not a final visual.

## Regenerating without re-running detection

`generate.py` caches the raw simulate() response at `_last_sim.json`. To
rebuild the scene from that (e.g. after tweaking `build_scene.py`) without
re-uploading/re-detecting:

```
"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --python build_scene.py -- --sim _last_sim.json --out scene.blend
```

## Rendering specific frames headlessly (no GUI)

```
"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -b scene.blend -o "//renders/frame_" -F PNG -f 1,150,400
```

`-f` takes a comma-separated frame list; frame numbers are `t_ms/1000*24 + 1`
(24fps, 1-based). Frame `scene.frame_end` is the end of the whole cycle.
