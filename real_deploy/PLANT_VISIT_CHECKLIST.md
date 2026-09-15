# Plant visit checklist - camera calibration (POC stage 5)

Everything in `real_deploy/` downstream of detection is blocked on this.
The code is written and unit-tested (`tests/test_triangulate.py` 6/6); what
is missing is measurements only a person standing in the plant can take.

Do the three parts IN ORDER. Part 0 first - calibrating cameras that are
about to be moved wastes the trip.

---

## Part 0 - Re-aim camera .102  (do this BEFORE calibrating)

**Why:** triangulation needs BOTH cameras to see the SAME point at the SAME
time. Evidence from the 2026-08-18 recording that they currently do not:

  - the hook detector found the hook in 186/186 cam .101 frames
  - and in 0 usable cam .102 frames - every cam .102 "hit" was glare at the
    frame edge

Camera .102 only catches the gantry carriage at its extreme right edge, cut
off, in the worst part of its fisheye distortion.

**Target:** both cameras should see the whole volume the hook travels
through - over the truck bed, across to the conveyor, and at both its
highest and lowest positions. They must view it from genuinely different
angles (real parallax); two cameras looking from nearly the same direction
triangulate badly.

**Also fix while you are up there:**
- [ ] Set camera .102 to a real 20-25 fps. It currently delivers 12.5 fps
      while its stream claims 25 - see README "Frame rate".
- [ ] Enable NTP time sync on BOTH cameras (their clocks disagree by ~4-8s).
- [ ] Turn OFF the burned-in timestamp/"Camera 01" overlay on .102.
- [ ] Record a fresh 5-10 min clip of a full unload AFTER re-aiming.

---

## Part 1 - Intrinsics  (lens properties; once per camera)

Independent of where the camera is mounted. Re-run only if the lens, focus
or zoom changes.

**Bring:**
- [ ] A printed checkerboard, 9x6 internal corners, glued to something
      RIGID and FLAT (foam board or a clipboard - a bending board gives a
      bad calibration)
- [ ] A ruler, to measure one square precisely (needed for `--square-size-m`)

**Do, for EACH camera:**
- [ ] Have someone walk the board around in front of the camera while you
      record the stream: near and far, tilted at various angles, and into
      every corner of the frame - not just the middle.
- [ ] Extract 15-25 stills from that recording into one folder per camera.

**Then run:**
```
py calibrate_intrinsics.py --name cam101 --images "calib_images/cam101/*.jpg" \
    --board-cols 9 --board-rows 6 --square-size-m 0.025 \
    --out calib_out/cam101_intrinsics.json
```

- [ ] Check the printed reprojection error: **under 0.5px good, under 1px
      usable, above 1px retake** (usually means poor frame-edge coverage or
      a board that flexed).

Note: camera .102 has heavy fisheye distortion, and this script uses
OpenCV's standard pinhole+radial model (`cv2.calibrateCamera`), not
`cv2.fisheye`. If .102's reprojection error will not come below ~1px, that
model is the likely reason - flag it and the script needs a fisheye path.

---

## Part 2 - Extrinsics  (where each camera is; the shared world frame)

**This is the step that defines your world frame, and the most likely
source of bad 3D positions later. Take your time here.**

**Decide the frame ONCE, write it down, use it for BOTH cameras:**

    origin = <a permanent, findable point - e.g. near-left base of the gantry post>
    +X = EAST   (trolley travel)
    +Y = NORTH  (bridge travel)
    +Z = UP
    units = METRES

The axes MUST match `planner3d.py`'s convention above, or every relay it
emits drives the crane the wrong way. Nothing in software can detect that
mistake.

**Bring:**
- [ ] Tape measure or laser distance meter
- [ ] Chalk / tape to mark points
- [ ] Someone who can safely access the floor area

**Do:**
- [ ] Pick **6 or more** points that are BOTH physically measurable from
      your origin AND clearly visible in the camera image - truck bed frame
      corners, taped floor marks, fixture corners.
- [ ] **Do not put them all on one flat plane.** Mix floor-height and
      bed-height points; a fully planar set makes the depth estimate
      unreliable.
- [ ] Measure X, Y, Z of each, in metres, from the origin. Record in a table
      on paper first.
- [ ] Grab one still from each camera showing all the points.

**Write `calib_points/cam101_points.json`:**
```json
[
  {"label": "bed_near_left",  "world": [0.0, 0.0, 1.0]},
  {"label": "bed_near_right", "world": [0.0, 2.2, 1.0]},
  {"label": "bed_far_left",   "world": [5.5, 0.0, 1.0]},
  {"label": "bed_far_right",  "world": [5.5, 2.2, 1.0]},
  {"label": "floor_left",     "world": [0.0, 0.0, 0.0]},
  {"label": "floor_right",    "world": [0.0, 2.2, 0.0]}
]
```

**Then run, per camera:**
```
py calibrate_extrinsics.py --intrinsics calib_out/cam101_intrinsics.json \
    --points calib_points/cam101_points.json --image calib_images/cam101_ref.jpg \
    --click --out calib_out/cam101_calibration.json
```
`--click` opens the image; click each labelled point in the SAME order.

- [ ] Check the printed reprojection error again.
- [ ] Sanity-check `camera_center_world` against a tape-measure distance to
      the real camera. If it says the camera is 40m away and it is 6m away,
      the calibration is wrong - do not proceed.

---

## Part 3 - Named reference positions

With both calibrations done:

- [ ] Park the crane at its home position, run:
      `py mark_reference_position.py --name home ... --out reference_positions.json`
- [ ] Park it over the conveyor drop point, run the same with `--name drop_point`

---

## After the visit - what unblocks

With `cam101_calibration.json`, `cam102_calibration.json` and
`reference_positions.json` in hand, the whole chain runs:

```
py run_pipeline.py --cam1-calibration ... --cam2-calibration ... \
    --source1 <video1> --source2 <video2> \
    --checkpoint ../training/output_real/checkpoint_best_ema.pth \
    --reference-positions reference_positions.json --out plan.json
```

That produces real 3D bale positions and an SF-8DR relay timeline. It
actuates nothing - it is a plan for review.

Still out of scope after this: the Modbus/PLC driver (needs the PLC's
register map), closed-loop control, and the safety system.

---

## Part 4 - Measurements for the 3D scene  (cheap, do it in the same visit)

Separate from calibration, and much less work. The live 3D view and the
Blender scene currently use geometry transcribed from `blender/factory.blend`
- numbers that were INVENTED for the simulation, never measured at ITC. See
`frontend/src/live/factoryLayout.ts`.

Measuring these ~8 values makes the 3D view match the real plant. They do
NOT require calibration, a checkerboard, or a shared world frame - just a
tape measure and five minutes.

Record in METRES:

**Truck**
- [ ] Bed width (across, side to side): ................ m   (assumed 2.2)
- [ ] Bed length (front to back): ..................... m   (assumed 5.5)
- [ ] Bed floor height above ground: .................. m   (assumed 1.0)

**Bale** (measure one representative bale)
- [ ] Width: ..... m   Depth: ..... m   Height: ..... m
- [ ] How many bales fit across the bed width? ........
- [ ] How many layers high is a typical load? ........

**Conveyor**
- [ ] Belt top surface height above ground: ........... m   (assumed 1.18)
- [ ] Horizontal distance, truck centreline to belt centreline: ..... m  (assumed 2.4)

**Gantry / crane**
- [ ] Height of the hook at its HIGHEST position (above ground): ..... m  (assumed 5.0)
- [ ] Height of the hook at its LOWEST position: ..................... m  (assumed 1.4)
- [ ] Total travel span East-West (trolley): ......................... m
- [ ] Total travel span North-South (bridge): ........................ m

**Speeds** - time these if at all possible; every value in
`planner3d.SpeedConfig3D` is currently a PLACEHOLDER, so the relay
timeline's durations are fiction until these are real:
- [ ] Seconds for the trolley (E/W) to cross a measured distance: ..... s over ..... m
- [ ] Same for the bridge (N/S): ...................................... s over ..... m
- [ ] Seconds for a full hoist DOWN stroke: ........................... s
- [ ] Seconds for a full hoist UP stroke: ............................. s
- [ ] Duration of the push-off pulse at the drop: ..................... s

With these, `factoryLayout.ts` and `SpeedConfig3D` get replaced with real
numbers, and both the 3D view and the emitted relay timings become
meaningful rather than illustrative.
