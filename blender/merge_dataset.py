"""Merges the real training set with the synthetic set (top-down + oblique,
bale + crane_hook + crane_spike) into one combined COCO-format dataset for
fine-tuning, so the model gains the new viewpoints/classes without
forgetting the original angled-view bale detection.
"""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
REAL_DIR = os.path.join(ROOT, "..", "reference_project", "dataset_sam3")
SYNTH_DIR = os.path.join(ROOT, "..", "synthetic_topdown")
OUT_DIR = os.path.join(ROOT, "..", "dataset_combined")
REAL_CRANE_ANNOTATIONS_PATH = os.path.join(ROOT, "real_crane_annotations.json")

CATEGORIES = [
    {"id": 1, "name": "bale", "supercategory": "bale"},
    {"id": 2, "name": "crane_hook", "supercategory": "crane"},
    {"id": 3, "name": "crane_spike", "supercategory": "crane"},
]
CATEGORY_NAME_TO_ID = {c["name"]: c["id"] for c in CATEGORIES}


def merge_split(split, real_crane_anns):
    real_split = os.path.join(REAL_DIR, split)
    synth_split = os.path.join(SYNTH_DIR, split)
    out_split = os.path.join(OUT_DIR, split)
    os.makedirs(out_split, exist_ok=True)

    real_coco = json.load(open(os.path.join(real_split, "_annotations.coco.json"), encoding="utf-8"))
    synth_coco = json.load(open(os.path.join(synth_split, "_annotations.coco.json"), encoding="utf-8"))

    images, annotations = [], []
    next_image_id, next_ann_id = 1, 1
    n_real, n_synth, n_crane_manual = 0, 0, 0

    # real: bale annotations come from the dataset as-is (all category_id=1
    # already); crane_hook/crane_spike are added on top from the manual
    # supplementary annotations, matched by original file_name
    id_map = {}
    for img in real_coco["images"]:
        new_id = next_image_id
        next_image_id += 1
        id_map[img["id"]] = new_id
        new_name = f"real_{img['file_name']}"
        shutil.copy(os.path.join(real_split, img["file_name"]), os.path.join(out_split, new_name))
        images.append({"id": new_id, "file_name": new_name, "width": img["width"], "height": img["height"]})
        n_real += 1

        for extra in real_crane_anns.get(split, {}).get(img["file_name"], []):
            annotations.append({
                "id": next_ann_id, "image_id": new_id,
                "category_id": CATEGORY_NAME_TO_ID[extra["category"]],
                "bbox": extra["bbox"],
                "area": extra["bbox"][2] * extra["bbox"][3],
                "iscrowd": 0,
            })
            next_ann_id += 1
            n_crane_manual += 1

    for ann in real_coco["annotations"]:
        annotations.append({
            "id": next_ann_id, "image_id": id_map[ann["image_id"]], "category_id": 1,
            "bbox": ann["bbox"], "area": ann["area"], "iscrowd": ann.get("iscrowd", 0),
        })
        next_ann_id += 1

    # synthetic: already has correct category_id (1=bale, 2=crane_hook,
    # 3=crane_spike) per annotation - use as-is, just remap ids
    id_map = {}
    for img in synth_coco["images"]:
        new_id = next_image_id
        next_image_id += 1
        id_map[img["id"]] = new_id
        new_name = f"synth_{img['file_name']}"
        shutil.copy(os.path.join(synth_split, img["file_name"]), os.path.join(out_split, new_name))
        images.append({"id": new_id, "file_name": new_name, "width": img["width"], "height": img["height"]})
        n_synth += 1
    for ann in synth_coco["annotations"]:
        annotations.append({
            "id": next_ann_id, "image_id": id_map[ann["image_id"]], "category_id": ann["category_id"],
            "bbox": ann["bbox"], "area": ann["area"], "iscrowd": ann.get("iscrowd", 0),
        })
        next_ann_id += 1

    combined = {"images": images, "annotations": annotations, "categories": CATEGORIES}
    with open(os.path.join(out_split, "_annotations.coco.json"), "w", encoding="utf-8") as f:
        json.dump(combined, f)

    print(f"{split}: {n_real} real (+{n_crane_manual} manual crane boxes) + {n_synth} synthetic "
          f"= {len(images)} images, {len(annotations)} annotations")


if __name__ == "__main__":
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    real_crane_anns = json.load(open(REAL_CRANE_ANNOTATIONS_PATH, encoding="utf-8"))
    for split in ("train", "valid"):
        merge_split(split, real_crane_anns)
    print(f"\nCombined dataset -> {OUT_DIR}")
