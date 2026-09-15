"""
Builds a 3D blockout .blend of the bale-unloading crane cycle from a
/api/simulate response JSON (the exact same data the 2D web app uses).

Run headlessly through Blender itself, not with plain python:

    blender --background --python build_scene.py -- --sim sim.json --out scene.blend

Everything after the bare `--` is this script's own argv; Blender ignores it.

Coordinate mapping (documented assumptions — adjust the constants below if
your real geometry differs):
  - World X = image X (East/West, trolley/cross-travel)
  - World Y = -image Y (North/South, bridge/long-travel; sign flipped so
    "up the image" reads as "forward" in Blender's default view)
  - World Z = vertical hoist, mapped from the abstract 0..1 `hoist` value
    onto [Z_GRAB, Z_CLEAR] meters.
  - Spike thrust (piercing axis) is a FIXED local direction on the hook
    rig (local -Y), independent of travel direction — per clarification,
    the bale is impaled by a horizontal stab, not by the vertical hoist.
  - Scale: meters-per-pixel is derived from the average detected bale
    width, assumed to be ~1.0m in reality. Override with --bale-width-m if
    your bales are a different size.

This is a blockout: primitives + flat colors, no textures. It's meant to
validate the 3D motion/geometry (does the pick-and-place cycle actually
make physical sense in 3D?), not to be a final visualization.
"""
import argparse
import json
import math
import sys

import bpy
import mathutils

# ---- Tunable blockout constants (meters) --------------------------------
BALE_HEIGHT_M = 0.6
Z_GRAB = BALE_HEIGHT_M          # hook height when hoist == 0 (at bale/drop level)
Z_CLEAR = 4.0                    # hook height when hoist == 1 (fully lifted/clear)
RAIL_HEIGHT = Z_CLEAR + 1.2      # fixed height of the bridge/rails above the hook's travel range
SPIKE_LENGTH_M = 0.6             # how far the spike stabs forward when thrust == 1
SPIKE_RADIUS_M = 0.05
MAST_RADIUS_M = 0.08
GROUND_MARGIN_M = 2.0
FPS = 24


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser(description="Build a 3D blockout scene from a simulate() response.")
    p.add_argument("--sim", required=True, help="Path to a saved /api/simulate response JSON")
    p.add_argument("--out", required=True, help="Output .blend path")
    p.add_argument("--bale-width-m", type=float, default=1.0, help="Assumed real bale width in meters, used to derive scale")
    return p.parse_args(argv)


def load_sim(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in list(bpy.data.collections):
        bpy.data.collections.remove(coll)


def make_material(name, rgba):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = rgba
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.7
    mat.diffuse_color = rgba
    return mat


def add_box(name, size, location, material, collection):
    # size=2 -> default cube spans -1..+1 (half-extent 1), so setting
    # obj.scale = (sx, sy, sz) below then gives half-extents (sx, sy, sz),
    # i.e. total dimensions (2*sx, 2*sy, 2*sz) matching every call site's
    # "size" tuples, which are all passed as half-dimensions.
    bpy.ops.mesh.primitive_cube_add(size=2, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (size[0], size[1], size[2])
    obj.data.materials.append(material)
    _move_to_collection(obj, collection)
    return obj


def add_cylinder(name, radius, depth, location, material, collection, rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(material)
    _move_to_collection(obj, collection)
    return obj


def _move_to_collection(obj, collection):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    collection.objects.link(obj)


def frame_of(t_ms, fps=FPS):
    return round(t_ms / 1000.0 * fps) + 1  # Blender frames are 1-based


def main():
    args = parse_args()
    sim = load_sim(args.sim)

    image_w = sim["image_width"]
    image_h = sim["image_height"]
    detections = sim["detections"]
    pick_cycles = sim["pick_cycles"]
    keyframes = sim["keyframes"]
    crane_home = sim["crane_home"]

    # ---- scale: meters per source-image pixel, derived from average bale width
    if detections:
        avg_bbox_w_px = sum(d["bbox"]["x2"] - d["bbox"]["x1"] for d in detections) / len(detections)
        scale = args.bale_width_m / avg_bbox_w_px
    else:
        scale = 1.0 / 80.0
    print(f"[build_scene] scale = {scale:.5f} m/px")

    cx_px = image_w / 2.0
    cy_px = image_h / 2.0

    def wx(px_x):
        return (px_x - cx_px) * scale

    def wy(px_y):
        return -(px_y - cy_px) * scale

    def wz(hoist):
        return Z_GRAB + hoist * (Z_CLEAR - Z_GRAB)

    # Every keyframe we insert should be linear (matches the timed, non-eased
    # motion model in planner.py) — set this once instead of touching fcurves
    # after the fact, since Blender's Action/fcurve internals have changed
    # across versions (layered actions in 5.x vs flat fcurves in 4.x).
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"

    clear_scene()
    root = bpy.data.collections.new("BaleCrane")
    bpy.context.scene.collection.children.link(root)

    mat_crane = make_material("Crane", (0.16, 0.32, 0.75, 1))
    mat_bale = make_material("Bale", (0.72, 0.58, 0.35, 1))
    mat_bale_active = make_material("BaleActive", (0.85, 0.65, 0.15, 1))
    mat_ground = make_material("Ground", (0.35, 0.35, 0.38, 1))
    mat_conveyor = make_material("Conveyor", (0.2, 0.2, 0.22, 1))
    mat_spike = make_material("Spike", (0.75, 0.76, 0.78, 1))
    mat_home = make_material("Home", (0.9, 0.9, 0.9, 1))

    # ---- working-area extent (for ground plane sizing) --------------------
    all_x = [kf["x"] for kf in keyframes] + [d["bbox"]["x1"] for d in detections] + [d["bbox"]["x2"] for d in detections]
    all_y = [kf["y"] for kf in keyframes] + [d["bbox"]["y1"] for d in detections] + [d["bbox"]["y2"] for d in detections]
    if pick_cycles:
        all_x += [pick_cycles[0]["drop_point"]["x"]]
        all_y += [pick_cycles[0]["drop_point"]["y"]]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    ground_w = (max_x - min_x) * scale + 2 * GROUND_MARGIN_M
    ground_d = (max_y - min_y) * scale + 2 * GROUND_MARGIN_M
    ground_cx = wx((min_x + max_x) / 2)
    ground_cy = wy((min_y + max_y) / 2)
    add_box("Ground", (ground_w / 2, ground_d / 2, 0.05), (ground_cx, ground_cy, -0.05), mat_ground, root)

    # ---- conveyor block at the drop point ----------------------------------
    if pick_cycles:
        drop = pick_cycles[0]["drop_point"]
        add_box(
            "Conveyor", (1.2, 0.6, 0.35),
            (wx(drop["x"]), wy(drop["y"]), 0.15),
            mat_conveyor, root,
        )

    # ---- rail posts (decorative, static) -----------------------------------
    rail_y = wy((min_y + max_y) / 2)
    for label, px_x in (("A", min_x - GROUND_MARGIN_M / scale * 0.3), ("B", max_x + GROUND_MARGIN_M / scale * 0.3)):
        add_cylinder(
            f"RailPost_{label}", 0.15, RAIL_HEIGHT,
            (wx(px_x), rail_y, RAIL_HEIGHT / 2),
            mat_crane, root,
        )

    # ---- home marker --------------------------------------------------------
    add_cylinder(
        "HomeMarker", 0.2, 0.05,
        (wx(crane_home["x"]), wy(crane_home["y"]), 0.03),
        mat_home, root,
    )

    # ---- crane rig: bridge, trolley, mast, hook, spike ----------------------
    bridge = add_box("Bridge", ((max_x - min_x) * scale / 2 + 1.0, 0.15, 0.15), (0, rail_y, RAIL_HEIGHT), mat_crane, root)
    trolley = add_box("Trolley", (0.3, 0.3, 0.2), (0, rail_y, RAIL_HEIGHT), mat_crane, root)
    mast = add_cylinder("Mast", MAST_RADIUS_M, 1.0, (0, rail_y, RAIL_HEIGHT), mat_crane, root)
    hook = add_box("Hook", (0.15, 0.15, 0.1), (0, rail_y, Z_CLEAR), mat_crane, root)
    spike = add_cylinder(
        "Spike", SPIKE_RADIUS_M, SPIKE_LENGTH_M, (0, 0, 0), mat_spike, root,
        rotation=(math.radians(90), 0, 0),
    )
    spike.parent = hook
    spike.location = (0, 0, 0)

    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start = 1

    for kf in keyframes:
        f = frame_of(kf["t_ms"])
        x, y, z = wx(kf["x"]), wy(kf["y"]), wz(kf["hoist"])

        bridge.location = (0, y, RAIL_HEIGHT)
        bridge.keyframe_insert(data_path="location", frame=f)

        trolley.location = (x, y, RAIL_HEIGHT)
        trolley.keyframe_insert(data_path="location", frame=f)

        hook.location = (x, y, z)
        hook.keyframe_insert(data_path="location", frame=f)

        mast_len = max(0.05, RAIL_HEIGHT - z)
        mast.location = (x, y, RAIL_HEIGHT - mast_len / 2)
        mast.scale = (1, 1, mast_len)
        mast.keyframe_insert(data_path="location", frame=f)
        mast.keyframe_insert(data_path="scale", frame=f)

        # spike thrust: local -Y offset, parented to Hook so it inherits x,y,z
        spike.location = (0, -kf["thrust"] * SPIKE_LENGTH_M / 2, 0)
        spike.keyframe_insert(data_path="location", frame=f)

    # ---- bales ---------------------------------------------------------------
    picked_by_id = {pc["bale_id"]: pc for pc in pick_cycles}
    for det in detections:
        bbox = det["bbox"]
        w = (bbox["x2"] - bbox["x1"]) * scale
        d = (bbox["y2"] - bbox["y1"]) * scale
        rest_x = wx((bbox["x1"] + bbox["x2"]) / 2)
        rest_y = wy((bbox["y1"] + bbox["y2"]) / 2)
        mat = mat_bale_active if det["id"] in picked_by_id else mat_bale
        bale = add_box(
            f"Bale_{det['id']}", (w / 2, d / 2, BALE_HEIGHT_M / 2),
            (rest_x, rest_y, BALE_HEIGHT_M / 2), mat, root,
        )

        cycle = picked_by_id.get(det["id"])
        if cycle is None:
            continue  # detected but not scheduled this cycle (e.g. max_bales limit) — stays put

        carry_kfs = [kf for kf in keyframes if kf.get("active_bale_id") == det["id"] and kf["spiked"]]
        if not carry_kfs:
            continue

        # hold at rest until the pierce moment
        f_start = frame_of(carry_kfs[0]["t_ms"])
        bale.location = (rest_x, rest_y, BALE_HEIGHT_M / 2)
        bale.keyframe_insert(data_path="location", frame=f_start)

        # follow the hook exactly for every keyframe while impaled
        for kf in carry_kfs:
            f = frame_of(kf["t_ms"])
            bale.location = (wx(kf["x"]), wy(kf["y"]), wz(kf["hoist"]))
            bale.keyframe_insert(data_path="location", frame=f)

        # settle at the drop point once pushed off
        drop = cycle["drop_point"]
        f_drop = frame_of(cycle["t_end_ms"])
        bale.location = (wx(drop["x"]), wy(drop["y"]), BALE_HEIGHT_M / 2)
        bale.keyframe_insert(data_path="location", frame=f_drop)

    total_ms = sim["total_duration_ms"]
    scene.frame_end = max(1, frame_of(total_ms))

    # ---- camera + light -------------------------------------------------------
    # Frame the full 3D bounding box (ground footprint AND rail height, not
    # just the footprint) and aim with a proper look-at, so the crane rig
    # above the pile is actually in shot rather than cropped out.
    cam_target_x = wx((min_x + max_x) / 2)
    cam_target_y = wy((min_y + max_y) / 2)
    cam_target_z = RAIL_HEIGHT * 0.35
    scene_span = max(ground_w, ground_d, RAIL_HEIGHT * 1.6)
    cam_dist = scene_span * 1.1 + 4

    cam_location = mathutils.Vector((
        cam_target_x - cam_dist * 0.55,
        cam_target_y - cam_dist * 0.85,
        cam_target_z + cam_dist * 0.5,
    ))
    target = mathutils.Vector((cam_target_x, cam_target_y, cam_target_z))
    direction = target - cam_location

    bpy.ops.object.camera_add(location=cam_location)
    cam = bpy.context.active_object
    cam.name = "MainCamera"
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = 28
    _move_to_collection(cam, root)
    scene.camera = cam

    bpy.ops.object.light_add(type="SUN", location=(cam_target_x, cam_target_y, 15))
    sun = bpy.context.active_object
    sun.data.energy = 3.0
    _move_to_collection(sun, root)

    scene.frame_set(1)

    bpy.ops.wm.save_as_mainfile(filepath=args.out)
    print(f"[build_scene] saved {args.out} ({scene.frame_end} frames @ {FPS}fps = {scene.frame_end / FPS:.1f}s)")


if __name__ == "__main__":
    main()
