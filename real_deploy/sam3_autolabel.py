"""
Batch-labels one class across many frames by calling the SAM3 annotation
service's text-prompt API directly, and writes a COCO file that
`import_annotations.py` can consume.

## Why this exists

The SAM3 web UI does the same thing one image at a time, behind an "AI"
button. This does it for a whole folder unattended, which matters because
the ITC frame set averages ~18 bales per image - hand-drawing them across
even 80 images is ~1,400 boxes.

## What it can and cannot label

Text prompting works for BALES and fails for the CRANE HOOK. Measured on
`cam101_002510.jpg`:

    "tobacco bale"    -> 14 objects
    "conveyor belt"   ->  1 object  (0.96)
    "crane hook"      ->  0 objects
    ...plus 13 other hook phrasings down to threshold 0.15, all 0 or noise
    ("blue steel column" scores 0.94/0.89 on the GANTRY POSTS, not the hook)

The hook is a site-specific assembly with no clean word for it, and the
frame is full of other blue steelwork that any such phrase matches first.
So the intended division of labour is: this script does the bales, a human
draws the hook in the UI, and `import_annotations.py --export` takes BOTH
files and merges them.

Cross-image template propagation would have solved the hook too, but the
service rejects it outright: "Cross-image template detection is not
supported. SAM3 visual prompts are location-based."

## Usage

    py sam3_autolabel.py --dir ../dataset_real_itc/train --limit 80 \\
        --prompt "tobacco bale" --category bale --out bales_train.json

Then, after drawing hooks in the UI and exporting that as e.g. hooks.json:

    py import_annotations.py --export bales_train.json --export hooks.json

Note this uploads the frames to the SAM3 service, which is a remote host.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
import uuid

DEFAULT_BASE = "https://sam3.bravetree-718bdea9.francecentral.azurecontainerapps.io"


def upload_image(base: str, path: str, timeout: float) -> dict:
    data = open(path, "rb").read()
    boundary = "----b" + uuid.uuid4().hex
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{os.path.basename(path)}"\r\nContent-Type: image/jpeg\r\n\r\n'
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        base + "/api/upload", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def segment_text(base: str, image_id: str, prompt: str, threshold: float, timeout: float) -> list[dict]:
    payload = json.dumps({
        "image_id": image_id, "prompt": prompt, "confidence_threshold": threshold}).encode()
    req = urllib.request.Request(base + "/api/segment/text", payload, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))["results"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="Folder of images to label")
    p.add_argument("--out", required=True, help="COCO file to write")
    p.add_argument("--prompt", default="tobacco bale")
    p.add_argument("--category", default="bale", help="Category name written into the COCO file")
    p.add_argument("--threshold", type=float, default=0.35,
                   help="Lower than the UI default: for a reviewable draft it is cheaper to "
                        "delete a wrong box than to draw a missing one")
    p.add_argument("--limit", type=int, default=None, help="Only do the first N images")
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--delay", type=float, default=0.0,
                   help="seconds to sleep between requests, even on success. The service can "
                        "return HTTP 200 with an empty result list when rate-limited rather than "
                        "erroring, which retries won't catch - use this to slow down and avoid it.")
    args = p.parse_args()

    names = sorted(f for f in os.listdir(args.dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if args.limit:
        names = names[: args.limit]
    if not names:
        raise SystemExit(f"no images found in {args.dir}")

    images, annotations = [], []
    ann_id = 1
    failed: list[str] = []
    empty: list[str] = []
    t0 = time.time()

    for i, name in enumerate(names, start=1):
        path = os.path.join(args.dir, name)
        results = None
        for attempt in range(args.retries + 1):
            try:
                info = upload_image(args.base, path, args.timeout)
                results = segment_text(args.base, info["id"], args.prompt, args.threshold, args.timeout)
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
                if attempt == args.retries:
                    print(f"  [{i}/{len(names)}] {name}: FAILED after {args.retries + 1} tries ({e})")
                    failed.append(name)
                else:
                    time.sleep(2 * (attempt + 1))

        if results is None:
            continue

        images.append({"id": i, "file_name": name, "width": info["width"], "height": info["height"]})
        if not results:
            empty.append(name)
        for r in results:
            x1, y1, x2, y2 = (float(v) for v in r["box"])
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2 = min(float(info["width"]), x2)
            y2 = min(float(info["height"]), y2)
            w, h = x2 - x1, y2 - y1
            if w <= 1 or h <= 1:
                continue
            annotations.append({
                "id": ann_id, "image_id": i, "category_id": 1,
                "category_name": args.category,
                "bbox": [round(x1), round(y1), round(w), round(h)],
                "area": round(w * h), "iscrowd": 0, "score": round(float(r["score"]), 4),
            })
            ann_id += 1

        rate = (time.time() - t0) / i
        print(f"  [{i}/{len(names)}] {name}: {len(results)} boxes "
              f"({rate:.1f}s/img, ~{rate * (len(names) - i) / 60:.0f} min left)")

        if args.delay:
            time.sleep(args.delay)

    doc = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": args.category, "supercategory": args.category}],
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)

    print(f"\n{len(annotations)} '{args.category}' boxes across {len(images)} images -> {args.out}")
    if empty:
        print(f"{len(empty)} images got NO boxes - check these by hand: {empty[:5]}"
              f"{' ...' if len(empty) > 5 else ''}")
    if failed:
        print(f"{len(failed)} images FAILED entirely and are absent from the file: {failed[:5]}"
              f"{' ...' if len(failed) > 5 else ''}")
    print("\nDRAFT - review before training. The crane hook is NOT in this file; draw it in the "
          "SAM3 UI, export, and pass both files to import_annotations.py.")


if __name__ == "__main__":
    main()
