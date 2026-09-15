"""
Generates DRAFT COCO annotations for dataset_real_itc by running the existing
simulation-trained checkpoint over the extracted frames.

These are a labeling head start, NOT ground truth. `DATA_COLLECTION.md` is
explicit that the simulation checkpoint was fine-tuned on Blender/Three.js
renders and that the earlier `training/README.md` results show 0 detections
on an unfamiliar viewpoint - so expect bales to pre-label usably and the
crane hook to pre-label badly or not at all. Every frame still needs a human
pass; this only removes the tedium of drawing the easy boxes from scratch.

Two mappings happen here, both to match `training/finetune_real.py`:

1. **crane_spike -> crane_hook.** The checkpoint predicts 3 classes but the
   real dataset is labeled with 2, because the real end effector has no
   visually separable spike. Spike predictions are therefore relabeled as
   crane_hook rather than dropped (dropping them would leave the hook
   under-covered in the draft).

2. **Per-class overlap suppression.** Because of that merge, a hook and a
   spike prediction on the same physical object become two boxes on one
   thing. Same-class boxes overlapping above --iou are collapsed to the
   highest-confidence one.

Output category ids are 1-indexed (bale=1, crane_hook=2), matching the
schema already used by `dataset_combined`'s own annotation files.

Usage:
    py autolabel_frames.py --dataset-dir ..\\dataset_real_itc --threshold 0.4

Then import dataset_real_itc/{train,valid}/ into Roboflow or CVAT, CORRECT
the boxes, and re-export over these files before training.
"""
import argparse
import json
import os

import cv2
import numpy as np

# Checkpoint class order (see capture_infer.py) -> output category id.
# 2 (crane_spike) folds into crane_hook; see module docstring.
MODEL_CLASS_TO_CATEGORY = {0: 1, 1: 2, 2: 2}
CATEGORIES = [
    {"id": 1, "name": "bale", "supercategory": "bale"},
    {"id": 2, "name": "crane_hook", "supercategory": "crane"},
]
CATEGORY_NAME = {c["id"]: c["name"] for c in CATEGORIES}


def iou(a, b):
    """IoU of two [x, y, w, h] boxes."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def suppress_overlaps(boxes, iou_threshold):
    """Greedy per-class suppression. boxes: [(bbox, score, category_id)], highest score first."""
    kept = []
    for bbox, score, cid in sorted(boxes, key=lambda t: -t[1]):
        if any(c == cid and iou(bbox, kb) > iou_threshold for kb, _, c in kept):
            continue
        kept.append((bbox, score, cid))
    return kept


def annotate_split(model, dataset_dir, split, threshold, iou_threshold, stats):
    split_dir = os.path.join(dataset_dir, split)
    names = sorted(f for f in os.listdir(split_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not names:
        raise SystemExit(f"no images found in {split_dir}")

    images, annotations = [], []
    ann_id = 1

    for image_id, name in enumerate(names, start=1):
        path = os.path.join(split_dir, name)
        frame = cv2.imread(path)
        if frame is None:
            print(f"  WARNING: unreadable, skipped: {name}")
            continue
        height, width = frame.shape[:2]
        images.append({"id": image_id, "file_name": name, "width": width, "height": height})

        result = model.predict(frame, threshold=threshold)
        boxes = []
        for box, conf, cid in zip(result.xyxy, result.confidence, result.class_id):
            category_id = MODEL_CLASS_TO_CATEGORY.get(int(cid))
            if category_id is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(width), x2), min(float(height), y2)
            w, h = x2 - x1, y2 - y1
            if w <= 1 or h <= 1:
                continue
            boxes.append(([round(x1), round(y1), round(w), round(h)], float(conf), category_id))

        kept = suppress_overlaps(boxes, iou_threshold)
        stats["suppressed"] += len(boxes) - len(kept)
        if not kept:
            stats["empty_images"].append(f"{split}/{name}")

        for bbox, score, category_id in kept:
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                # Extra field, ignored by COCO loaders - lets a reviewer sort
                # the draft boxes by how much the model actually believed them.
                "score": round(score, 4),
            })
            stats["per_class"][category_id] = stats["per_class"].get(category_id, 0) + 1
            ann_id += 1

    out_path = os.path.join(split_dir, "_annotations.coco.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"images": images, "annotations": annotations, "categories": CATEGORIES}, fh)

    print(f"{split}: {len(images)} images, {len(annotations)} draft boxes -> {out_path}")
    return len(images)


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default=os.path.join(repo_root, "dataset_real_itc"))
    p.add_argument("--checkpoint", default=os.path.join(
        repo_root, "training", "output_crane_hook3d_v2", "checkpoint_best_ema.pth"))
    p.add_argument("--resolution", type=int, default=640)
    p.add_argument("--threshold", type=float, default=0.4,
                   help="detection confidence floor. Lower than inference default 0.5: for a DRAFT "
                        "it is cheaper to delete a wrong box than to draw a missing one")
    p.add_argument("--iou", type=float, default=0.6, help="same-class overlap above which boxes are merged")
    args = p.parse_args()

    print(f"loading checkpoint: {args.checkpoint}")
    from rfdetr import RFDETRNano
    model = RFDETRNano(resolution=args.resolution, pretrain_weights=args.checkpoint)

    stats = {"per_class": {}, "suppressed": 0, "empty_images": []}
    total = 0
    for split in ("train", "valid"):
        total += annotate_split(model, args.dataset_dir, split, args.threshold, args.iou, stats)

    print("\n--- draft summary ---")
    for cid, name in CATEGORY_NAME.items():
        print(f"  {name}: {stats['per_class'].get(cid, 0)} boxes")
    print(f"  merged/suppressed overlaps: {stats['suppressed']}")
    empty = stats["empty_images"]
    print(f"  images with NO detections: {len(empty)}/{total}"
          f" ({100.0 * len(empty) / max(total, 1):.0f}%) - these need labeling from scratch")
    for name in empty[:10]:
        print(f"    {name}")
    if len(empty) > 10:
        print(f"    ... and {len(empty) - 10} more")
    print("\nDRAFT ONLY - correct these by hand before training. See DATA_COLLECTION.md.")


if __name__ == "__main__":
    main()
