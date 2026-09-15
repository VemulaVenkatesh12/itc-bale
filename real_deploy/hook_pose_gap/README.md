# Hook pose gap - training data to collect

Found 2026-09-11: the 40-epoch checkpoint (`training/output_real_40ep/`)
detects the crane hook confidently (0.4-0.6+) when it's **mid-descent over
an open truck bed**, fully visible - that's what the 2026-08-25 recordings
mostly captured and what the dataset is built around.

It scores much lower (~0.05-0.2, below the working 0.3 threshold) when the
hook is **retracted/parked at the top of its travel, squeezed between two
loaded stacks** - only the mounting bracket and gearbox housing are visible,
the actual hook/spike is occluded on both sides. See the frame in this
folder from that investigation (`cam101/2026-09-11T*_parked_occluded.jpg`)
for exactly what this pose looks like.

This isn't a bug to fix in code - it's a real gap in what the model has
seen. `frontend/src/live/PlantCameraFeed.tsx` compensates for it at display
time (lower threshold + dashed "tentative" boxes below 0.3, see
`PLANT_DISPLAY_THRESHOLD`), but the actual fix is more labeled examples of
this pose fine-tuned into the checkpoint.

## What to collect

Frames of the hook in poses **other than** "descending over an open bed":

- retracted/parked at top of travel (any horizontal position)
- partially occluded by a bale stack from either side
- mid-transit between truck and conveyor (not over either)
- close to empty-bed edge cases (last bale, hook near the truck rail)

Both cameras, varied lighting/time of day if possible. Doesn't need to be a
huge set - a few dozen genuinely varied poses is enough to fine-tune against,
per `training/README.md`'s process.

## Collecting

`collect_frames.py` pulls a snapshot from each live camera (via the running
backend's `/api/cameras/<id>/snapshot`) and saves it only if it's visibly
different from the last saved frame for that camera - so pointing this at an
idle crane doesn't fill the folder with near-duplicates.

```
python3 real_deploy/hook_pose_gap/collect_frames.py --every-s 30
# Ctrl-C to stop; run it during a normal shift to sample varied poses over time.
# Or a single grab, e.g. right after seeing an interesting pose live:
python3 real_deploy/hook_pose_gap/collect_frames.py --once
```

Frames land in `cam101/` / `cam102/` as `<ISO-timestamp>.jpg`. Once there's
a useful set:

1. Label crane_hook/crane_spike boxes on the occluded/parked examples (SAM3
   autolabel + manual correction, same as `real_deploy/sam3_autolabel.py` /
   `import_annotations.py` used for the original dataset - see
   `real_deploy/DATA_COLLECTION.md`).
2. Merge into `dataset_real_itc` alongside the existing frames.
3. Re-run `training/finetune_real.py` from the current `output_real_40ep`
   checkpoint (not from scratch) so the model keeps what it already knows
   and just adds coverage of this pose.
