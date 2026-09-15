"""
Generates a synthetic bale/hook/spike detection training set: randomized
bale layouts with a crane hook+spike mid-pick, rendered from TWO cameras
per sample (near-nadir top-down, and an oblique/elevated angle) with
pixel-accurate COCO bboxes for all three classes.

Run headlessly:
    blender --background --python generate_synthetic_topdown.py -- --n 50 --out ..\synthetic_topdown\train --seed 0
    blender --background --python generate_synthetic_topdown.py -- --n 10 --out ..\synthetic_topdown\valid --seed 999

Two cameras per sample, not one: from near-straight-down, a vertical spike
foreshortens to almost nothing - there's no meaningful bbox to learn from
that angle. The real reference photos that show the hook/spike clearly
(reference_project/dataset_sam3/train/frame_0005.jpg etc.) are all from an
elevated, angled vantage point, not overhead. So each scene build renders
both: the existing top-down view (still fine for bales) and a second
oblique view (where hook/spike actually have visible extent) - doubling
the value of every scene without doubling scene-construction cost.

Bbox extraction: rather than projecting 3D corners (which ignores
occlusion), each camera's shot is rendered twice — once with real
materials (the saved training image), once with every tracked object
(bales, hook, spike) swapped to a unique flat emission color on a black
background (the "ID matte", discarded after use). Scanning the matte for
each color's pixel extent gives a bbox that matches exactly what's visible
in the real render. See `resolve_visible` for why bbox colors are sampled
from the actual rendered pixels rather than predicted.

Categories: 1=bale, 2=crane_hook, 3=crane_spike.
"""
import argparse
import json
import math
import os
import random
import sys

# Blender's embedded Python has user-site installs disabled, so Pillow
# (used only for a PNG->JPG convert) is vendored locally instead - see
# README.md for the install command.
_DEPS = os.path.join(os.path.dirname(__file__), "_py_deps")
if os.path.isdir(_DEPS) and _DEPS not in sys.path:
    sys.path.insert(0, _DEPS)

import bpy

CAT_BALE, CAT_HOOK, CAT_SPIKE = 1, 2, 3


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50, help="Number of samples to generate")
    p.add_argument("--out", required=True, help="Output directory (images + _annotations.coco.json land here)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--crops-dir", default=os.path.join(os.path.dirname(__file__), "crops"))
    p.add_argument("--res", type=int, default=640)
    p.add_argument("--min-vis-px", type=int, default=150, help="Drop bboxes with fewer visible pixels than this (too occluded to be a useful example)")
    return p.parse_args(argv)


def make_material(name, rgba, emission=False, metallic=0.0, roughness=0.5):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nodes = m.node_tree.nodes
    if emission:
        for n in list(nodes):
            nodes.remove(n)
        emit = nodes.new("ShaderNodeEmission")
        emit.inputs["Color"].default_value = rgba
        emit.inputs["Strength"].default_value = 1.0
        out = nodes.new("ShaderNodeOutputMaterial")
        m.node_tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    else:
        b = nodes.get("Principled BSDF")
        if b:
            b.inputs["Base Color"].default_value = rgba
            b.inputs["Metallic"].default_value = metallic
            b.inputs["Roughness"].default_value = roughness
    return m


def index_to_color(i):
    """Well-separated flat colors for however many objects get tracked
    (bale counts run up to ~35+ once hook/spike are added). A simple
    multiplicative hash mod 255 clusters badly at that count - verified
    real collisions (different indices landing within a few units of each
    other, merging two objects' masks into one). Golden-angle hue spacing
    keeps every pair of indices well separated regardless of how many
    there are."""
    import colorsys
    golden_ratio_conjugate = 0.61803398875
    hue = (i * golden_ratio_conjugate) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (r, g, b, 1.0)


def build_sample_scene(crop_files, crops_dir, rng):
    """Returns (root_collection, tracked) where tracked is a list of
    {"obj": bpy_object, "category_id": int} for every bale + the hook +
    the spike."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in list(bpy.data.collections):
        bpy.data.collections.remove(coll)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)
    for img in list(bpy.data.images):
        if img.name != "Render Result":
            bpy.data.images.remove(img)

    root = bpy.data.collections.new("Sample")
    bpy.context.scene.collection.children.link(root)

    def link(obj):
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        root.objects.link(obj)

    mat_ground = make_material("Ground", (rng.uniform(0.2, 0.4),) * 3 + (1,))
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, -0.05))
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground.scale = (6, 4, 0.05)
    ground.data.materials.append(mat_ground)
    link(ground)

    # randomized bale grid: footprint roughly matching the real truck bed
    BALE_W, BALE_D, BALE_H = 0.85, 0.85, 0.6
    cols_x = rng.randint(2, 3)
    cols_y = rng.randint(4, 7)
    layers = rng.choice([1, 1, 2])  # bias toward single layer, some double

    tracked = []
    bale_tops = []  # (x, y, top_z) for every placed bale, to pick an "active" one later
    idx = 0
    for layer in range(layers):
        z = 0.3 + layer * (BALE_H + 0.05) + BALE_H / 2
        for cy in range(cols_y):
            if layer == 1 and rng.random() < 0.35:
                continue  # partial top layer, varied coverage
            for cx in range(cols_x):
                if rng.random() < 0.08:
                    continue  # occasional gaps, more realistic than a perfect grid
                x = (cx - (cols_x - 1) / 2) * (BALE_W + 0.05) + rng.uniform(-0.04, 0.04)
                y = (cy - (cols_y - 1) / 2) * (BALE_D + 0.05) + rng.uniform(-0.04, 0.04)
                rot = math.radians(rng.uniform(-6, 6))

                bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, z))
                bale = bpy.context.active_object
                bale.name = f"Bale_{idx}"
                bale.scale = (BALE_W, BALE_D, BALE_H)
                bale.rotation_euler[2] = rot
                link(bale)

                crop_file = rng.choice(crop_files)
                img = bpy.data.images.load(os.path.join(crops_dir, crop_file), check_existing=True)
                mat = bpy.data.materials.new(f"BaleMat_{idx}")
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                bsdf = nodes.get("Principled BSDF")
                tex_node = nodes.new("ShaderNodeTexImage")
                tex_node.image = img
                mat.node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
                bsdf.inputs["Roughness"].default_value = 0.95
                bale.data.materials.append(mat)

                # full-face UV (same fix as the interactive factory scene)
                import bmesh
                bm = bmesh.new()
                bm.from_mesh(bale.data)
                bm.faces.ensure_lookup_table()
                uv_layer = bm.loops.layers.uv.active or bm.loops.layers.uv.new()
                corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
                for face in bm.faces:
                    for i, loop in enumerate(face.loops):
                        loop[uv_layer].uv = corners[i % 4]
                bm.to_mesh(bale.data)
                bm.free()

                tracked.append({"obj": bale, "category_id": CAT_BALE})
                bale_tops.append((x, y, z + BALE_H / 2))
                idx += 1

    # crane hook + spike, mid-pick over a randomly chosen bale (prefer one
    # of the topmost/most-exposed ones - matches real operation, and
    # matches the real photos where the gripper is always on an exposed
    # top-layer bale, never buried under others)
    if bale_tops:
        max_z = max(t[2] for t in bale_tops)
        candidates = [t for t in bale_tops if t[2] >= max_z - 0.05]
        ax, ay, atop = rng.choice(candidates)

        mat_hook = make_material("HookMat", (0.15, 0.3, 0.7, 1), metallic=0.6, roughness=0.4)
        mat_spike = make_material("SpikeMat", (0.12, 0.12, 0.13, 1), metallic=0.7, roughness=0.35)

        # spike: vertical prong descending into the bale top (matches the
        # real photos - the crane pierces the bale from directly above,
        # not from the side; see frame_0005/0020/0025.jpg)
        spike_len = rng.uniform(0.35, 0.55)
        embed = rng.uniform(0.05, 0.18)  # how far the tip has sunk into the bale
        spike_z = atop - embed + spike_len / 2
        bpy.ops.mesh.primitive_cylinder_add(radius=rng.uniform(0.035, 0.05), depth=spike_len,
                                             location=(ax + rng.uniform(-0.1, 0.1), ay + rng.uniform(-0.1, 0.1), spike_z))
        spike = bpy.context.active_object
        spike.name = "Spike"
        spike.data.materials.append(mat_spike)
        link(spike)
        tracked.append({"obj": spike, "category_id": CAT_SPIKE})

        # hook: the gripper carriage sitting just above the bale, spike
        # passing through/near it
        hook_z = spike_z + spike_len / 2 + rng.uniform(0.05, 0.15)
        bpy.ops.mesh.primitive_cube_add(size=1, location=(ax, ay, hook_z))
        hook = bpy.context.active_object
        hook.name = "Hook"
        hook.scale = (rng.uniform(0.35, 0.5), rng.uniform(0.3, 0.4), rng.uniform(0.12, 0.2))
        hook.data.materials.append(mat_hook)
        link(hook)
        tracked.append({"obj": hook, "category_id": CAT_HOOK})

    return root, tracked


def setup_topdown_camera(tracked, rng):
    xs = [t["obj"].location.x for t in tracked]
    ys = [t["obj"].location.y for t in tracked]
    cx = (min(xs) + max(xs)) / 2 if xs else 0
    cy = (min(ys) + max(ys)) / 2 if ys else 0
    height = rng.uniform(6.0, 9.0)
    tilt = math.radians(rng.uniform(0, 12))       # small deviation off nadir for domain randomization
    azimuth = rng.uniform(0, 2 * math.pi)
    jx = height * math.tan(tilt) * math.cos(azimuth)
    jy = height * math.tan(tilt) * math.sin(azimuth)

    bpy.ops.object.camera_add(location=(cx + jx, cy + jy, height))
    cam = bpy.context.active_object
    cam.name = "TopdownCam"
    cam.data.lens = rng.uniform(14, 20)
    import mathutils
    direction = mathutils.Vector((cx, cy, 0)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return cam


def setup_oblique_camera(tracked, rng):
    """Elevated, angled view - where the hook/spike actually have visible
    extent (matches the real reference photos' vantage point)."""
    xs = [t["obj"].location.x for t in tracked]
    ys = [t["obj"].location.y for t in tracked]
    zs = [t["obj"].location.z for t in tracked]
    cx = (min(xs) + max(xs)) / 2 if xs else 0
    cy = (min(ys) + max(ys)) / 2 if ys else 0
    target_z = (min(zs) + max(zs)) / 2 if zs else 0.5

    dist = rng.uniform(4.0, 7.0)
    azimuth = rng.uniform(0, 2 * math.pi)
    cam_height = rng.uniform(2.5, 4.5)
    cx2 = cx + dist * math.cos(azimuth)
    cy2 = cy + dist * math.sin(azimuth)

    bpy.ops.object.camera_add(location=(cx2, cy2, cam_height))
    cam = bpy.context.active_object
    cam.name = "ObliqueCam"
    cam.data.lens = rng.uniform(24, 38)
    import mathutils
    # aim a bit above the bale-height midpoint, toward the hook, not the ground
    target = mathutils.Vector((cx, cy, target_z + rng.uniform(0.1, 0.4)))
    direction = target - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return cam


def setup_lighting(rng):
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 10))
    sun = bpy.context.active_object
    sun.data.energy = rng.uniform(2.5, 4.5)
    sun.rotation_euler = (math.radians(rng.uniform(20, 50)), 0, rng.uniform(0, 2 * math.pi))


def render_beauty(path, res):
    scene = bpy.context.scene
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"  # keep colors predictable (no AgX/Filmic tone-mapping)
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def render_matte(path, res, tracked):
    scene = bpy.context.scene
    scene.view_settings.view_transform = "Standard"  # exact colors matter here - no tone-mapping curve
    world = bpy.data.worlds[0] if bpy.data.worlds else bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0, 0, 0, 1)
        bg.inputs["Strength"].default_value = 0.0

    saved_materials = {}
    for i, t in enumerate(tracked, start=1):
        obj = t["obj"]
        saved_materials[obj.name] = list(obj.data.materials)
        obj.data.materials.clear()
        mat = make_material(f"__matte_{i}", index_to_color(i), emission=True)
        obj.data.materials.append(mat)

    lights_state = []
    for obj in bpy.data.objects:
        if obj.type == "LIGHT":
            lights_state.append((obj, obj.hide_render))
            obj.hide_render = True

    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)

    for obj, hide in lights_state:
        obj.hide_render = hide
    for t in tracked:
        obj = t["obj"]
        obj.data.materials.clear()
        for m in saved_materials[obj.name]:
            obj.data.materials.append(m)

    bg2 = world.node_tree.nodes.get("Background")
    if bg2:
        bg2.inputs["Strength"].default_value = 1.0


def resolve_visible(scene, cam, tracked, img):
    """For each tracked object, sample the matte's ACTUAL rendered color at
    its projected screen position (rather than predicting what color it
    should be — Blender's color-management pipeline transforms emission
    values before they hit the PNG, so a prediction drifts). A small
    neighborhood is sampled (not a single pixel) so landing exactly on an
    anti-aliased seam doesn't misfire. If two objects' centers both resolve
    to the same color, the farther one's center just leaked the nearer
    (occluding) object's color — only the nearer one is actually visible
    there."""
    import numpy as np
    from bpy_extras.object_utils import world_to_camera_view

    h, w, _ = img.shape
    samples = {}  # obj.name -> (color, depth, category_id, px, py)
    for t in tracked:
        obj, cat = t["obj"], t["category_id"]
        co = world_to_camera_view(scene, cam, obj.matrix_world.translation)
        if co.z <= 0:
            continue  # behind the camera
        px, py = int(co.x * w), int((1.0 - co.y) * h)
        if not (0 <= px < w and 0 <= py < h):
            continue
        x0, x1 = max(0, px - 2), min(w, px + 3)
        y0, y1 = max(0, py - 2), min(h, py + 3)
        patch = img[y0:y1, x0:x1].reshape(-1, 3)
        if patch.size == 0:
            continue
        colors, counts = np.unique(patch, axis=0, return_counts=True)
        color = tuple(int(v) for v in colors[np.argmax(counts)])
        if sum(color) < 10:
            continue  # sampled background - this object's own center is fully hidden
        samples[obj.name] = (color, co.z, cat, px, py)

    winners = {}  # color -> (obj_name, depth, category_id, px, py)
    for name, (color, depth, cat, px, py) in samples.items():
        cur = winners.get(color)
        if cur is None or depth < cur[1]:
            winners[color] = (name, depth, cat, px, py)

    return {name: (color, cat, px, py) for color, (name, depth, cat, px, py) in winners.items()}


def bboxes_from_matte(img, visible, min_vis_px):
    """Small, thin, or partly-occluded objects (the spike, especially) can
    have their sample point land on a pixel that's actually showing a
    DIFFERENT nearby object - not a duplicate-color collision (which
    `resolve_visible` already handles), just an honestly-sampled color that
    happens to belong to someone else because the intended object is mostly
    hidden right at its own center. Guard against this: the resulting bbox
    must actually contain the object's own projected position, or it's
    discarded as an unreliable read for this sample rather than emitted."""
    import numpy as np
    results = {}
    for name, (color, cat, px, py) in visible.items():
        target = np.array(color)
        mask = np.all(np.abs(img.astype(int) - target) <= 6, axis=-1)
        count = int(mask.sum())
        if count < min_vis_px:
            continue
        ys, xs = np.where(mask)
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        if not (x1 <= px < x2 and y1 <= py < y2):
            continue  # sampled color belongs to something else's visible region
        results[name] = {"bbox": [x1, y1, x2 - x1, y2 - y1], "area": count, "category_id": cat}
    return results


def render_and_annotate(cam, tracked, out_path, res, min_vis_px, matte_tmp):
    import numpy as np
    from PIL import Image

    scene = bpy.context.scene
    scene.camera = cam

    png_path = out_path.replace(".jpg", ".png")
    render_beauty(png_path, res)
    Image.open(png_path).convert("RGB").save(out_path, quality=92)
    os.remove(png_path)

    render_matte(matte_tmp, res, tracked)
    matte_img = np.array(Image.open(matte_tmp).convert("RGB"))
    visible = resolve_visible(scene, cam, tracked, matte_img)
    return bboxes_from_matte(matte_img, visible, min_vis_px)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    crop_files = sorted(f for f in os.listdir(args.crops_dir) if f.endswith(".png"))
    if not crop_files:
        raise RuntimeError(f"No crop images found in {args.crops_dir} - run crop_bales.py first")

    images_meta = []
    annotations = []
    ann_id = 1
    image_id = 1

    matte_tmp = os.path.join(args.out, "_matte_tmp.png")

    for sample_i in range(args.n):
        root, tracked = build_sample_scene(crop_files, args.crops_dir, rng)
        top_cam = setup_topdown_camera(tracked, rng)
        oblique_cam = setup_oblique_camera(tracked, rng)
        setup_lighting(rng)

        for cam, tag in ((top_cam, "top"), (oblique_cam, "obl")):
            img_name = f"synth_{sample_i:04d}_{tag}.jpg"
            out_path = os.path.join(args.out, img_name)
            boxes = render_and_annotate(cam, tracked, out_path, args.res, args.min_vis_px, matte_tmp)

            images_meta.append({"id": image_id, "file_name": img_name, "width": args.res, "height": args.res})
            for b in boxes.values():
                annotations.append({
                    "id": ann_id, "image_id": image_id, "category_id": b["category_id"],
                    "bbox": b["bbox"], "area": b["area"], "iscrowd": 0,
                })
                ann_id += 1

            n_bale = sum(1 for b in boxes.values() if b["category_id"] == CAT_BALE)
            n_hook = sum(1 for b in boxes.values() if b["category_id"] == CAT_HOOK)
            n_spike = sum(1 for b in boxes.values() if b["category_id"] == CAT_SPIKE)
            print(f"[{sample_i+1}/{args.n}][{tag}] {img_name}: {n_bale} bales, {n_hook} hook, {n_spike} spike "
                  f"(of {len(tracked)} placed)")
            image_id += 1

    if os.path.exists(matte_tmp):
        os.remove(matte_tmp)

    coco = {
        "images": images_meta,
        "annotations": annotations,
        "categories": [
            {"id": CAT_BALE, "name": "bale", "supercategory": "bale"},
            {"id": CAT_HOOK, "name": "crane_hook", "supercategory": "crane"},
            {"id": CAT_SPIKE, "name": "crane_spike", "supercategory": "crane"},
        ],
    }
    with open(os.path.join(args.out, "_annotations.coco.json"), "w", encoding="utf-8") as f:
        json.dump(coco, f)

    print(f"\nDone: {len(images_meta)} images, {len(annotations)} annotations -> {args.out}")


if __name__ == "__main__":
    main()
