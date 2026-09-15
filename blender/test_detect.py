"""Runs the real RF-DETR bale detector against rendered Blender camera
frames, to check whether the synthetic bale textures are detectable by
the actual model (not just visually plausible to a human)."""
import os
import sys

import cv2
import supervision as sv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from app.model_service import detect_bales, warmup  # noqa: E402

RENDERS_DIR = os.path.join(os.path.dirname(__file__), "renders")
THRESHOLDS = [0.5, 0.3, 0.15]


def run(image_path, out_path):
    img = cv2.imread(image_path)
    print(f"\n=== {os.path.basename(image_path)} ({img.shape[1]}x{img.shape[0]}) ===")
    best_dets = None
    for th in THRESHOLDS:
        dets = detect_bales(img, threshold=th)
        print(f"  threshold {th}: {len(dets)} bales detected"
              + (f", confidences: {[round(d.confidence, 2) for d in dets]}" if dets else ""))
        if dets and best_dets is None:
            best_dets = (th, dets)

    if best_dets:
        th, dets = best_dets
        xyxy = [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in dets]
        confidence = [d.confidence for d in dets]
        import numpy as np
        detections_sv = sv.Detections(
            xyxy=np.array(xyxy), confidence=np.array(confidence),
            class_id=np.zeros(len(xyxy), dtype=int),
        )
        box_annotator = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=2, color_lookup=sv.ColorLookup.INDEX)
        label_annotator = sv.LabelAnnotator(
            color=sv.Color.GREEN, text_color=sv.Color.BLACK, text_scale=0.5, color_lookup=sv.ColorLookup.INDEX,
        )
        labels = [f"bale {c:.2f}" for c in confidence]
        annotated = box_annotator.annotate(scene=img.copy(), detections=detections_sv)
        annotated = label_annotator.annotate(scene=annotated, detections=detections_sv, labels=labels)
        cv2.imwrite(out_path, annotated)
        print(f"  annotated -> {out_path}")
    else:
        print("  NO detections at any threshold down to 0.15")


if __name__ == "__main__":
    warmup()
    run(os.path.join(RENDERS_DIR, "test_top.png"), os.path.join(RENDERS_DIR, "test_top_detected.png"))
    run(os.path.join(RENDERS_DIR, "test_back.png"), os.path.join(RENDERS_DIR, "test_back_detected.png"))
