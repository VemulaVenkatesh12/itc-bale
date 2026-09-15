"""
Automatically finds the crane hook in the extracted frames, producing draft
`crane_hook` boxes that `import_annotations.py` can merge with the existing
bale labels.

## Why not the model, SAM3 text, or SAM3 templates

All three were tried and all three fail on this object:

  - the existing checkpoint    -> 9 hook boxes across 401 images (it was
                                  trained ~91% on Blender renders and has
                                  never seen a real gantry hook)
  - SAM3 text prompting        -> 0 hits across 22 phrasings; the frame is
                                  full of blue steelwork that any such
                                  phrase matches first
  - SAM3 cross-image templates -> refused by the service outright

## The signal this uses instead

The hook is the only thing in frame that is BOTH blue AND moving:

    blue + static  = the gantry posts and roof steel  (rejected: static)
    moving + dull  = bales, truck, people             (rejected: not blue)
    blue + moving  = the hook assembly                <- what we want

A per-pixel median over all frames of one camera gives a clean plate of the
static structure (the median is used rather than the mean specifically so a
hook that lingers in one spot for a few frames does not smear itself into
the background). Anything that differs from that plate is moving; masking
that by hue keeps only the blue mover.

This is a DRAFT generator - every box it emits is reviewable, and
`--annotate-dir` renders overlays so a human can check them quickly rather
than trusting the count.

Usage:
    py detect_hook_candidates.py --prefix cam101 --out hooks_cam101.json \\
        --annotate-dir ../hook_review
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

# Hue range for the crane's blue, in OpenCV's 0-179 hue scale. Deliberately
# wide: the same paint reads very differently under the plant's mixed
# lighting between the lit conveyor side and the shadowed stack side.
BLUE_LO = np.array([95, 60, 40], dtype=np.uint8)
BLUE_HI = np.array([130, 255, 255], dtype=np.uint8)


def build_background(paths: list[str], sample: int, scale: float) -> np.ndarray:
    """Per-pixel median of a sample of frames = the static structure."""
    step = max(1, len(paths) // sample)
    stack = []
    for p in paths[::step]:
        img = cv2.imread(p)
        if img is None:
            continue
        stack.append(cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    if not stack:
        raise SystemExit("no readable frames to build a background from")
    return np.median(np.array(stack), axis=0).astype(np.uint8)


def _crop_to_end_effector(component: np.ndarray, x: int, y: int, w: int, h: int,
                          widen_ratio: float = 1.4, min_rows: int = 6) -> tuple[int, int, int, int]:
    """Trims a full-assembly component down to just the hook's end piece.

    The moving blue region is the mast PLUS the end effector, so its raw
    bounding box runs from the top of the frame to the foot - a box whose
    size is dominated by how high the ceiling is rather than by the object we
    care about. The end effector is distinguishable by shape alone: it is
    markedly WIDER than the mast above it. So measure the mast's typical
    width from the upper half, then walk up from the bottom while rows stay
    wider than that, and keep only those rows.

    Falls back to the bottom fifth of the component if no clear widening
    exists (e.g. the hook is viewed edge-on, or is occluded) - a slightly
    wrong crop is much better than reverting to a full-height box.
    """
    rows = component[y:y + h, x:x + w]
    widths = rows.sum(axis=1).astype(float)
    if widths.size == 0 or widths.max() == 0:
        return x, y, w, h

    upper = widths[: max(1, len(widths) // 2)]
    mast_width = float(np.median(upper[upper > 0])) if np.any(upper > 0) else 0.0

    cut = None
    if mast_width > 0:
        threshold = mast_width * widen_ratio
        r = len(widths) - 1
        while r >= 0 and widths[r] >= threshold:
            r -= 1
        if len(widths) - 1 - r >= min_rows:
            cut = r + 1

    if cut is None:
        cut = int(len(widths) * 0.8)

    sub = rows[cut:]
    cols = np.where(sub.any(axis=0))[0]
    if cols.size == 0:
        return x, y, w, h
    return x + int(cols[0]), y + cut, int(cols[-1] - cols[0] + 1), int(len(widths) - cut)


def find_hook(img: np.ndarray, background: np.ndarray, scale: float,
              diff_threshold: int, min_area: int) -> tuple[list[int], float] | None:
    """Returns ([x, y, w, h] in FULL-resolution pixels, score) or None."""
    small = cv2.resize(img, (background.shape[1], background.shape[0]), interpolation=cv2.INTER_AREA)

    moving = cv2.cvtColor(cv2.absdiff(small, background), cv2.COLOR_BGR2GRAY)
    moving = (moving > diff_threshold).astype(np.uint8) * 255

    blue = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_BGR2HSV), BLUE_LO, BLUE_HI)

    mask = cv2.bitwise_and(moving, blue)
    # Close first to join the foot to the cylinder across the darker seam
    # between them, then open to drop isolated speckle from compression noise.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None

    best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    area = int(stats[best, cv2.CC_STAT_AREA])
    if area < min_area:
        return None

    x, y, w, h = (int(stats[best, k]) for k in
                  (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    component = labels == best
    x, y, w, h = _crop_to_end_effector(component, x, y, w, h)

    # Fill must be measured over the CROPPED region, not the whole component -
    # using the full component's area against the cropped box's dimensions
    # yields ratios above 1 and silently disables --min-fill.
    cropped_area = int(component[y:y + h, x:x + w].sum())
    if cropped_area < min_area:
        return None

    inv = 1.0 / scale
    box = [round(x * inv), round(y * inv), round(w * inv), round(h * inv)]
    # Score is a coverage ratio, not a probability: how solidly the component
    # fills its own bounding box. A real hook is a chunky solid object; a
    # sprawling thin component that happens to be blue and moving is usually
    # glare travelling along a rail, and scores low.
    return box, float(cropped_area) / max(w * h, 1)


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default=os.path.join(repo_root, "dataset_real_itc"))
    p.add_argument("--prefix", default="cam101", help="Only process frames whose name starts with this")
    p.add_argument("--out", required=True, help="COCO file of draft crane_hook boxes")
    p.add_argument("--annotate-dir", default=None, help="Also write overlay images here for review")
    p.add_argument("--scale", type=float, default=0.4, help="Work at this fraction of full res, for speed")
    p.add_argument("--background-samples", type=int, default=60)
    p.add_argument("--diff-threshold", type=int, default=32)
    p.add_argument("--min-area", type=int, default=400, help="Minimum component area, in SCALED pixels")
    p.add_argument("--min-fill", type=float, default=0.18, help="Reject components sparser than this")
    args = p.parse_args()

    entries = []
    for split in ("train", "valid"):
        d = os.path.join(args.dataset_dir, split)
        for f in sorted(os.listdir(d)):
            if f.startswith(args.prefix) and f.lower().endswith((".jpg", ".jpeg", ".png")):
                entries.append((split, f, os.path.join(d, f)))
    if not entries:
        raise SystemExit(f"no frames matching prefix {args.prefix!r}")

    print(f"{len(entries)} {args.prefix} frames; building static-structure plate...")
    background = build_background([e[2] for e in entries], args.background_samples, args.scale)

    if args.annotate_dir:
        os.makedirs(args.annotate_dir, exist_ok=True)

    images, annotations = [], []
    ann_id = 1
    found = 0

    for image_id, (split, name, path) in enumerate(entries, start=1):
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        images.append({"id": image_id, "file_name": name, "width": w, "height": h})

        hit = find_hook(img, background, args.scale, args.diff_threshold, args.min_area)
        if hit and hit[1] >= args.min_fill:
            box, score = hit
            annotations.append({
                "id": ann_id, "image_id": image_id, "category_id": 1,
                "category_name": "crane_hook", "bbox": box,
                "area": box[2] * box[3], "iscrowd": 0, "score": round(score, 3),
            })
            ann_id += 1
            found += 1
            if args.annotate_dir:
                vis = img.copy()
                cv2.rectangle(vis, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 6)
                cv2.putText(vis, f"fill={score:.2f}", (box[0], max(box[1] - 12, 40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)
                cv2.imwrite(os.path.join(args.annotate_dir, name),
                            cv2.resize(vis, (1280, round(1280 * h / w))))

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"images": images, "annotations": annotations,
                   "categories": [{"id": 1, "name": "crane_hook", "supercategory": "crane"}]}, fh)

    print(f"hook found in {found}/{len(entries)} frames -> {args.out}")
    if args.annotate_dir:
        print(f"review overlays: {args.annotate_dir}")
    print("\nDRAFT - review the overlays before training. Frames with no hook found carry no box, "
          "which is correct when the hook genuinely is not in view and wrong when it was missed.")


if __name__ == "__main__":
    main()
