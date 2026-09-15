# Auto-labeled crane_hook increment (2026-09-11)

Source: `real_deploy/hook_pose_gap/collect_frames.py` running against the live
cameras during a real crane movement window (10:36-11:01). All 81 collected
frames were scored with the current checkpoint
(`training/output_real_40ep/checkpoint_best_ema.pth`) at threshold 0.02; the
19 with a crane_hook score >= 0.20 were reviewed **visually, one at a time**
(cropped + boxed, see the 2026-09-11 session) before being labeled - nothing
here is unverified model output.

`annotations.json` - COCO format, single category (`crane_hook`, id 1),
`bbox` in `[x, y, w, h]` absolute pixels, `score` is the model's own
confidence at labeling time (kept for reference, not used as ground truth
weight).

## What's in it

**15 positive images** - the model's own predicted box, human/AI-verified as
genuinely on the crane carriage/hook assembly. Confidence ranged 0.21-0.43 at
labeling time - real detections, just under the 0.3 display threshold this
whole investigation has been about.

**4 hard negatives** (images present, zero annotations - the standard way to
tell a detector "nothing of this class is here"):
- 1 frame where the model boxed a **person** at 0.68 confidence (highest
  score in the whole batch - the most dangerous kind of miss, high
  confidence AND wrong)
- 3 frames (same recurring spot) where it boxed a **fixed conveyor drive
  motor** at 0.24-0.27 - a static piece of machinery, not the crane, that
  visually rhymes with the hook's silhouette from this angle

Both are real, camera-fixed confusions worth teaching the model out - they
will keep recurring at these exact pixel locations on this camera.

## Using this

Small increment (19 images) - meant to nudge the current checkpoint, not
replace it. Merge into a training run via `training/finetune_real.py`,
starting from `output_real_40ep/checkpoint_best_ema.pth` (not from scratch),
and evaluate on these same images plus a held-out set before swapping
anything live. See `real_deploy/hook_pose_gap/README.md` for the fuller
context on why this pose is under-represented in the original dataset.
