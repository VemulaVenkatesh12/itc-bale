"""
Imports COCO exported from the SAM3 annotation tool into
dataset_real_itc/{train,valid}/_annotations.coco.json, in the exact shape
training/finetune_real.py expects.

Run this after exporting from the tool. It exists because four things have
to line up between "boxes drawn in a web UI" and "a training run that isn't
quietly wrong", and none of them are automatic:

1. **Category names.** The tool's categories are whatever you typed - e.g.
   "Bale" and "Crane". `finetune_real.py` declares `["bale", "crane_hook"]`,
   and RF-DETR maps class INDEX to name by position, so a mismatch doesn't
   error, it silently trains the wrong label onto the wrong class. Aliases
   below normalize the common spellings; anything unrecognized is reported
   and skipped rather than guessed at.

2. **The train/valid split.** `extract_training_frames.py` split the frames
   by TIME BLOCK precisely so near-identical neighbouring video frames never
   straddle the two sets. This script therefore places each annotation
   according to which folder its image already lives in - it never re-splits.
   If the annotation tool offers to shuffle and split for you, decline.

3. **Integer image ids.** The tool keys annotations by a string UUID; COCO
   (and rfdetr's loader) want integers. Renumbered here.

4. **Coverage.** An export covering 40 of 401 images is a perfectly valid
   COCO file, and training on it would treat the other 361 images as
   containing NO objects - actively teaching the model that bales and hooks
   are background. The coverage report at the end is the thing to actually
   read before training.

Usage:
    py import_annotations.py --export exported.json
    py import_annotations.py --export a.json --export b.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict

# Normalized target classes, in the order finetune_real.py declares them.
# COCO category ids are 1-indexed, matching dataset_combined's own files.
TARGET_CLASSES = ["bale", "crane_hook"]
CATEGORY_ID = {name: i for i, name in enumerate(TARGET_CLASSES, start=1)}

# Spellings seen or plausible from the annotation UI -> canonical class.
ALIASES = {
    "bale": "bale", "bales": "bale", "tobacco bale": "bale",
    "crane": "crane_hook", "crane_hook": "crane_hook", "cranehook": "crane_hook",
    "hook": "crane_hook", "crane hook": "crane_hook", "gripper": "crane_hook",
    "spike": "crane_hook", "crane_spike": "crane_hook",  # folded: see DATA_COLLECTION.md
}

CATEGORIES_OUT = [
    {"id": CATEGORY_ID["bale"], "name": "bale", "supercategory": "bale"},
    {"id": CATEGORY_ID["crane_hook"], "name": "crane_hook", "supercategory": "crane"},
]


def normalize_class(raw: str) -> str | None:
    return ALIASES.get(str(raw).strip().lower())


def load_exports(paths: list[str]) -> tuple[dict[str, list[dict]], Counter, set]:
    """Returns (annotations keyed by image basename, per-class counts, unknown category names)."""
    by_image: dict[str, list[dict]] = defaultdict(list)
    counts: Counter = Counter()
    unknown: set = set()

    for path in paths:
        doc = json.loads(open(path, encoding="utf-8").read())
        # Map the export's own image ids (int or UUID string) to file names.
        id_to_name = {img["id"]: os.path.basename(img["file_name"]) for img in doc.get("images", [])}
        cat_names = {c["id"]: c.get("name", "") for c in doc.get("categories", [])}

        for ann in doc.get("annotations", []):
            raw_name = ann.get("category_name") or cat_names.get(ann.get("category_id"), "")
            cls = normalize_class(raw_name)
            if cls is None:
                unknown.add(raw_name)
                continue
            name = id_to_name.get(ann.get("image_id"))
            if name is None:
                continue
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                counts["_bad_bbox"] += 1
                continue
            by_image[name].append({
                "category_id": CATEGORY_ID[cls],
                # Round to int like dataset_combined's existing files; drop
                # segmentation - RF-DETR trains on boxes, and keeping polygons
                # would triple the file size for no training benefit.
                "bbox": [round(float(v)) for v in bbox],
            })
            counts[cls] += 1
    return by_image, counts, unknown


def load_existing_by_class(split_dir: str, keep_classes: set[str]) -> dict[str, list[dict]]:
    """Reads the annotations already on disk and keeps only the named classes.

    This is what makes "label ONLY the hook by hand" workable: the bale boxes
    already in the file (from autolabel_frames.py) are carried through, while
    the hand-drawn hook boxes come from the export. Without this, importing a
    hook-only export would wipe 7,175 bale boxes and train a model that
    thinks bales are background.
    """
    if not keep_classes:
        return {}
    path = os.path.join(split_dir, "_annotations.coco.json")
    if not os.path.exists(path):
        return {}
    doc = json.loads(open(path, encoding="utf-8").read())
    id_to_name = {img["id"]: os.path.basename(img["file_name"]) for img in doc.get("images", [])}
    cat_names = {c["id"]: c.get("name", "") for c in doc.get("categories", [])}

    kept: dict[str, list[dict]] = defaultdict(list)
    for ann in doc.get("annotations", []):
        cls = normalize_class(cat_names.get(ann.get("category_id"), ""))
        if cls is None or cls not in keep_classes:
            continue
        name = id_to_name.get(ann.get("image_id"))
        if name is None:
            continue
        kept[name].append({"category_id": CATEGORY_ID[cls], "bbox": [round(float(v)) for v in ann["bbox"]]})
    return kept


def build_split(split_dir: str, by_image: dict[str, list[dict]],
                existing: dict[str, list[dict]] | None = None) -> tuple[dict, list[str]]:
    """Builds the COCO doc for one split, plus the list of images left unlabeled."""
    import cv2

    existing = existing or {}
    names = sorted(f for f in os.listdir(split_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    images, annotations, unlabeled = [], [], []
    ann_id = 1

    for image_id, name in enumerate(names, start=1):
        img = cv2.imread(os.path.join(split_dir, name))
        if img is None:
            continue
        h, w = img.shape[:2]
        images.append({"id": image_id, "file_name": name, "width": w, "height": h})

        # Merge per-image AND per-class: a hand-drawn class always wins over
        # the draft for that image, and the draft only fills classes the
        # export is silent about. Merging per-image alone would double every
        # bale on an image where the annotator labelled bales as well as the
        # hook - two boxes on one object, which trains the model to predict
        # duplicates.
        exported = by_image.get(name, [])
        exported_classes = {a["category_id"] for a in exported}
        carried = [a for a in existing.get(name, []) if a["category_id"] not in exported_classes]
        anns = carried + exported
        if not anns:
            unlabeled.append(name)
        for a in anns:
            annotations.append({
                "id": ann_id, "image_id": image_id, "category_id": a["category_id"],
                "bbox": a["bbox"], "area": a["bbox"][2] * a["bbox"][3], "iscrowd": 0,
            })
            ann_id += 1

    return {"images": images, "annotations": annotations, "categories": CATEGORIES_OUT}, unlabeled


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--export", action="append", required=True,
                   help="COCO file exported from the annotation tool (repeat for several)")
    p.add_argument("--dataset-dir", default=os.path.join(repo_root, "dataset_real_itc"))
    p.add_argument("--dry-run", action="store_true", help="Report only; write nothing")
    p.add_argument("--keep-existing", action="append", default=None, metavar="CLASS",
                   help="Carry these classes through from the annotations already on disk "
                        "(e.g. --keep-existing bale, when the export contains only hooks). "
                        "Repeatable. Without it, the export fully replaces what is there.")
    args = p.parse_args()

    keep_classes = set()
    for raw in args.keep_existing or []:
        cls = normalize_class(raw)
        if cls is None:
            raise SystemExit(f"--keep-existing {raw!r} is not a known class; expected one of {TARGET_CLASSES}")
        keep_classes.add(cls)

    by_image, counts, unknown = load_exports(args.export)
    print(f"read {len(args.export)} export file(s): "
          f"{sum(v for k, v in counts.items() if not k.startswith('_'))} boxes "
          f"across {len(by_image)} images")
    for cls in TARGET_CLASSES:
        print(f"  {cls}: {counts.get(cls, 0)}")
    if counts.get("_bad_bbox"):
        print(f"  skipped {counts['_bad_bbox']} zero/negative-sized boxes")
    if unknown:
        print(f"  WARNING: unrecognized categories SKIPPED: {sorted(unknown)}")
        print(f"           add them to ALIASES in this script, or rename them in the tool")

    total_unlabeled = 0
    total_images = 0
    for split in ("train", "valid"):
        split_dir = os.path.join(args.dataset_dir, split)
        existing = load_existing_by_class(split_dir, keep_classes)
        if existing:
            print(f"\n{split}: carrying through {sum(len(v) for v in existing.values())} existing "
                  f"{'/'.join(sorted(keep_classes))} boxes from disk")
        doc, unlabeled = build_split(split_dir, by_image, existing)
        total_unlabeled += len(unlabeled)
        total_images += len(doc["images"])
        out = os.path.join(split_dir, "_annotations.coco.json")

        print(f"\n{split}: {len(doc['images'])} images, {len(doc['annotations'])} boxes, "
              f"{len(unlabeled)} images with NO annotations")
        if args.dry_run:
            continue
        if os.path.exists(out):
            backup = out + ".draft.bak"
            if not os.path.exists(backup):
                shutil.copy2(out, backup)
                print(f"  previous (draft) annotations backed up -> {os.path.basename(backup)}")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
        print(f"  wrote {out}")

    labeled = total_images - total_unlabeled
    pct = 100.0 * labeled / total_images if total_images else 0.0
    print(f"\n--- coverage: {labeled}/{total_images} images have annotations ({pct:.0f}%) ---")
    if total_unlabeled:
        print(
            f"{total_unlabeled} images have NO boxes. Training treats those as images containing\n"
            "nothing - which teaches the model that bales and hooks are background. Either finish\n"
            "labeling them, or delete the unlabeled images from the dataset before training."
        )
    print("\nnext: cd ../training && py finetune_real.py --epochs 15 --batch-size 8")


if __name__ == "__main__":
    main()
