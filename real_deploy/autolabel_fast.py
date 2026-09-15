"""
Faster version of autolabel_frames.py using float16 inference and batched I/O.
Generates DRAFT COCO annotations for dataset_real_itc.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

MODEL_CLASS_TO_CATEGORY = {0: 1, 1: 2, 2: 2}
CATEGORIES = [
    {"id": 1, "name": "bale", "supercategory": "bale"},
    {"id": 2, "name": "crane_hook", "supercategory": "crane"},
]
CATEGORY_NAME = {c["id"]: c["name"] for c in CATEGORIES}


def iou(a, b):
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
        print(f"  no images found in {split_dir}")
        return 0

    images, annotations = [], []
    ann_id = 1
    t0 = time.time()

    for image_id, name in enumerate(names, start=1):
        path = os.path.join(split_dir, name)
        frame = cv2.imread(path)
        if frame is None:
            continue
        height, width = frame.shape[:2]
        images.append({"id": image_id, "file_name": name, "width": width, "height": height})

        import torch
        with torch.no_grad():
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
                "id": ann_id, "image_id": image_id, "category_id": category_id,
                "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0,
                "score": round(score, 4),
            })
            stats["per_class"][category_id] = stats["per_class"].get(category_id, 0) + 1
            ann_id += 1

        if image_id % 50 == 0 or image_id == len(names):
            elapsed = time.time() - t0
            rate = elapsed / image_id
            eta = rate * (len(names) - image_id)
            print(f"  [{split}] {image_id}/{len(names)} - {len(annotations)} boxes "
                  f"({rate:.1f}s/img, ~{eta/60:.0f} min left)")

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
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--iou", type=float, default=0.6)
    args = p.parse_args()

    import torch
    print(f"loading checkpoint: {args.checkpoint}")
    from rfdetr import RFDETRNano
    model = RFDETRNano(resolution=args.resolution, pretrain_weights=args.checkpoint)
    model.inference(dtype=torch.float16)
    print("model loaded with float16 inference")

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
          f" ({100.0 * len(empty) / max(total, 1):.0f}%)")
    for name in empty[:10]:
        print(f"    {name}")
    if len(empty) > 10:
        print(f"    ... and {len(empty) - 10} more")
    print("\nDRAFT ONLY - correct these by hand before training.")


if __name__ == "__main__":
    main()
