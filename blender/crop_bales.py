"""Crops each detected bale's bbox out of the source image into its own
PNG, for use as a Blender material texture (real photo per bale instead of
a flat color). Run with plain Python (PIL), not inside Blender.
"""
import json
import os

from PIL import Image

SIM_JSON = os.path.join(os.path.dirname(__file__), "_last_sim.json")
SOURCE_IMAGE = os.path.join(
    os.path.dirname(__file__), "..", "reference_project", "dataset_sam3", "train", "frame_0000.jpg"
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "crops")
MARGIN_FRAC = 0.04  # small margin so the crop doesn't hard-clip right at the box edge


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sim = json.load(open(SIM_JSON, encoding="utf-8"))
    img = Image.open(SOURCE_IMAGE).convert("RGB")
    w, h = img.size

    for det in sim["detections"]:
        bbox = det["bbox"]
        bw = bbox["x2"] - bbox["x1"]
        bh = bbox["y2"] - bbox["y1"]
        mx, my = bw * MARGIN_FRAC, bh * MARGIN_FRAC
        x1 = max(0, int(bbox["x1"] - mx))
        y1 = max(0, int(bbox["y1"] - my))
        x2 = min(w, int(bbox["x2"] + mx))
        y2 = min(h, int(bbox["y2"] + my))
        crop = img.crop((x1, y1, x2, y2))
        out_path = os.path.join(OUT_DIR, f"bale_{det['id']}.png")
        crop.save(out_path)
        print(f"bale {det['id']}: {crop.size} -> {out_path}")


if __name__ == "__main__":
    main()
