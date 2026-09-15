# Collecting real training data (bale / crane_hook / crane_spike)

You're starting from zero on real footage. This is the single biggest
factor in whether the real system actually works - the simulation's model
was trained on Blender renders, which do not transfer reliably to real
camera images (different lighting, texture, lens distortion, motion blur,
background clutter). There's no shortcut around collecting real data.

The good news: this repo already has a working real-data fine-tuning
pipeline (`training/finetune.py`) and a working real-data example
(`reference_project/dataset_sam3/` - real angled-view bale photos, COCO
format) - the new real dataset just needs to land in the same shape and it
plugs into the existing training script almost unchanged.

## What to capture

Three classes, matching the simulation's: `bale`, `crane_hook`,
`crane_spike`. If your crane's spike isn't visually distinct from the hook
housing in your camera views, it's fine to only label `bale` and
`crane_hook` and drop `crane_spike` - update `class_names` in your copy of
`finetune.py`'s `model.train(...)` call to match whatever you actually
label.

From **both** real cameras (top-down and back/side, matching the sim's
Camera_Top/Camera_Back roles - or whatever two views you actually mount):

- **Empty bed / no bale**: a handful of frames with nothing to detect -
  teaches the model what background looks like.
- **Bales at rest**, varied: different stack positions, different lighting
  (the actual conditions you'll run in - don't shoot indoors under studio
  light if the real deployment is outdoors/mixed lighting), different bale
  orientations if bales aren't always placed identically, partial
  occlusion (bales behind other bales).
- **The hook/spike moving through its full range**: empty (no bale), just
  after insertion, mid-hoist, at the drop point, at every extreme of its
  travel (fully up/down, fully east/west/north/south) - the "vision only"
  milestone specifically needs the hook detected and correctly triangulated
  everywhere it can physically be, not just in a few convenient poses.
- **The hook WITH a bale attached** (mid-pick, mid-carry) - visually
  different from the bare hook and needs its own coverage.

Target roughly 150-300 labeled images per class as a starting point (small
but workable for fine-tuning from the existing checkpoint, which already
has strong bale features from `dataset_sam3` - you're extending it, not
starting the model from scratch). More is better, especially for
`crane_hook`/`crane_spike` since the existing checkpoint has never seen a
real one.

## How to label it

Any tool that exports COCO format works - Roboflow (what `dataset_sam3`'s
`_annotations.coco.json` naming implies was used originally) or CVAT are
both reasonable. Export as:
```
your_dataset/
  train/
    <images>
    _annotations.coco.json
  valid/
    <images>
    _annotations.coco.json
```
(same shape as `reference_project/dataset_sam3/` - copy its structure if
you want a concrete reference.)

## Training

Once you have `your_dataset/{train,valid}/` in that shape, either:

- Copy `training/finetune.py` to something like `training/finetune_real.py`
  and point `DATASET_DIR` at your new folder, `CHECKPOINT_PATH` at
  `reference_project/edge_deploy/checkpoint_best_ema_crane_hook3d_v2.pth`
  (the current multi-class checkpoint - continuing from it rather than the
  original single-class one, so the real fine-tune doesn't have to
  relearn hook/spike detection from nothing), and `OUTPUT_DIR` somewhere
  new (e.g. `training/output_real/`); or
- Run the existing script directly with overrides:
  ```
  cd training
  py finetune.py --dataset-dir ..\your_dataset --output-dir output_real ^
      --checkpoint ..\reference_project\edge_deploy\checkpoint_best_ema_crane_hook3d_v2.pth
  ```

Watch the same reprojection/mAP guidance in `training/README.md` - start
with fewer epochs than the default 40 for a first pass on a small real
dataset, check `training/output_real/checkpoint_best_ema.pth`'s
performance on a few held-out real photos before committing to a long run.

## Using the result

Point `real_deploy/capture_infer.py --checkpoint` at
`training/output_real/checkpoint_best_ema.pth` once you have it. Don't
reuse the simulation's `checkpoint_best_ema_crane_hook3d_v2.pth` for real
inference even though it's technically loadable - it was fine-tuned on
Three.js renders, not real footage, and the earlier `training/README.md`
"Results" table shows exactly this kind of domain gap (0 detections on an
unfamiliar viewpoint before fine-tuning on data from that viewpoint).
