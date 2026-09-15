"""
Milestone 1 deliverable: real two-camera capture -> real model inference ->
cross-camera matching -> real 3D triangulated positions. Prints a live
position for every matched bale/hook/spike detection. This is deliberately
as far as this script goes - it does NOT touch any relay, PLC, or actuator.
Nothing here can move the crane; it only proves the vision pipeline
produces trustworthy 3D positions, which is the prerequisite for anything
downstream ever being allowed to act on them.

Sources can be camera indices (0, 1, ...) for live hardware, or video file
paths for testing this against recorded footage before any camera is
connected - both go through cv2.VideoCapture identically.

## Known limitation: cross-camera detection matching

Matching WHICH detection in camera 2 corresponds to WHICH detection in
camera 1 (there can be several bales of the same class in frame at once) is
done here with a simple greedy nearest-epipolar-line + lowest-reprojection-
error heuristic (see _match_detections). That's a reasonable baseline, not
a robust solution - it has no memory across frames (no tracking) and can
mismatch when two same-class objects are close together in both views. If
real-world testing shows bad matches, the fix is temporal tracking (match
using the previous frame's known positions as a prior), not more one-shot
geometry.

## Known limitation: frame synchronization

For two independent live cameras (not hardware-triggered together), the
frames read in the same loop iteration are only approximately
simultaneous - fine for mostly-static bales, increasingly wrong for a
fast-moving hook. If that matters for your setup, look into
hardware-synchronized capture (genlock/trigger cables) or timestamping +
interpolation - neither is implemented here.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

from calibration import CameraCalibration
from hook_keypoint import contact_tip_px
from triangulate import epipolar_distance_px, reprojection_error_px, triangulate_point

CLASS_NAMES = {0: "bale", 1: "crane_hook", 2: "crane_spike"}

# How far a candidate cross-camera match may sit from the epipolar line
# before being rejected outright, and how much reprojection error a
# triangulation may have before being reported as "matched" rather than
# discarded as noise/mismatch. Both in pixels - loosen these if your
# calibration's reprojection error (reported by calibrate_extrinsics.py) is
# itself a few pixels, tighten them if it's under 1px and you're still
# seeing implausible triangulated positions.
EPIPOLAR_MATCH_THRESHOLD_PX = 15.0
REPROJECTION_ACCEPT_THRESHOLD_PX = 15.0


@dataclass
class Detection2D:
    class_id: int
    confidence: float
    center_px: tuple[float, float]


@dataclass
class MatchedDetection3D:
    class_id: int
    world_point: np.ndarray  # (x, y, z)
    confidence: float  # min of the two matched detections' confidences
    reprojection_error_px: float


def _open_source(source: str) -> cv2.VideoCapture:
    """`source` is a camera index (e.g. "0") or a file path - try int first,
    fall back to treating it as a path (matches reference_project/
    edge_deploy/detect.py's --webcam/--video split, just unified into one
    arg since real_deploy has two of everything)."""
    try:
        return cv2.VideoCapture(int(source))
    except ValueError:
        return cv2.VideoCapture(source)


def _load_model(checkpoint: str, resolution: int):
    import torch
    from rfdetr import RFDETRNano

    model = RFDETRNano(resolution=resolution, pretrain_weights=checkpoint)
    if torch.backends.mkldnn.is_available():
        torch.backends.mkldnn.enabled = True
    model.optimize_for_inference()
    return model


def _detect(model, frame_bgr: np.ndarray, threshold: float) -> list[Detection2D]:
    result = model.predict(frame_bgr, threshold=threshold)
    detections: list[Detection2D] = []
    if len(result) == 0:
        return detections
    for box, conf, cid in zip(result.xyxy, result.confidence, result.class_id):
        x1, y1, x2, y2 = [float(v) for v in box]
        cid = int(cid)
        # For the crane hook, the point that matters for triangulation is the
        # contact tip of the end-effector, not the box centre - it sits ~82%
        # of the way down the box (see hook_keypoint.py for the derivation).
        if CLASS_NAMES.get(cid) == "crane_hook":
            point = contact_tip_px(x1, y1, x2, y2)
        else:
            point = ((x1 + x2) / 2, (y1 + y2) / 2)
        detections.append(Detection2D(class_id=cid, confidence=float(conf), center_px=point))
    return detections


def _match_detections(
    cam1: CameraCalibration, dets1: list[Detection2D], cam2: CameraCalibration, dets2: list[Detection2D]
) -> list[MatchedDetection3D]:
    """Greedy same-class matching (see this module's docstring for the
    known limitations) - for each cam1 detection, picks the unclaimed
    same-class cam2 detection with the lowest reprojection error among
    those within EPIPOLAR_MATCH_THRESHOLD_PX, then triangulates."""
    matched: list[MatchedDetection3D] = []
    claimed2: set[int] = set()

    for d1 in dets1:
        best_j: int | None = None
        best_err = float("inf")
        for j, d2 in enumerate(dets2):
            if j in claimed2 or d2.class_id != d1.class_id:
                continue
            epi_dist = epipolar_distance_px(cam1, d1.center_px, cam2, d2.center_px)
            if epi_dist > EPIPOLAR_MATCH_THRESHOLD_PX:
                continue
            err = reprojection_error_px(cam1, d1.center_px, cam2, d2.center_px)
            if err < best_err:
                best_err, best_j = err, j

        if best_j is None or best_err > REPROJECTION_ACCEPT_THRESHOLD_PX:
            continue  # no acceptable match in the other view this frame - not fatal, just unreported

        claimed2.add(best_j)
        d2 = dets2[best_j]
        world_point = triangulate_point(cam1, d1.center_px, cam2, d2.center_px)
        matched.append(
            MatchedDetection3D(
                class_id=d1.class_id,
                world_point=world_point,
                confidence=min(d1.confidence, d2.confidence),
                reprojection_error_px=best_err,
            )
        )
    return matched


def run(
    cam1_cal: CameraCalibration,
    source1: str,
    cam2_cal: CameraCalibration,
    source2: str,
    checkpoint: str,
    resolution: int,
    threshold: float,
    display: bool,
    max_frames: int | None = None,
) -> None:
    print(f"loading model from {checkpoint}")
    model = _load_model(checkpoint, resolution)

    cap1, cap2 = _open_source(source1), _open_source(source2)
    if not cap1.isOpened():
        raise RuntimeError(f"could not open source1: {source1}")
    if not cap2.isOpened():
        raise RuntimeError(f"could not open source2: {source2}")

    print(f"{cam1_cal.name}: {source1}   {cam2_cal.name}: {source2}")
    print("(vision only - no relay/actuator output. Ctrl+C to stop)\n")

    frame_i = 0
    try:
        while True:
            ok1, frame1 = cap1.read()
            ok2, frame2 = cap2.read()
            if not ok1 or not ok2:
                print("end of stream (or a camera dropped) - stopping")
                break
            frame_i += 1
            if max_frames is not None and frame_i > max_frames:
                print(f"reached --max-frames ({max_frames}) - stopping")
                break

            t0 = time.perf_counter()
            dets1 = _detect(model, frame1, threshold)
            dets2 = _detect(model, frame2, threshold)
            matches = _match_detections(cam1_cal, dets1, cam2_cal, dets2)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            print(f"[frame {frame_i}] {cam1_cal.name}:{len(dets1)} {cam2_cal.name}:{len(dets2)} "
                  f"detections, {len(matches)} matched -> 3D ({elapsed_ms:.0f}ms)")
            for m in matches:
                x, y, z = m.world_point
                name = CLASS_NAMES.get(m.class_id, f"class_{m.class_id}")
                print(f"    {name}: ({x:+.3f}, {y:+.3f}, {z:+.3f}) m  "
                      f"conf={m.confidence:.2f} reproj_err={m.reprojection_error_px:.1f}px")

            if display:
                cv2.imshow(f"{cam1_cal.name}", frame1)
                cv2.imshow(f"{cam2_cal.name}", frame2)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        cap1.release()
        cap2.release()
        if display:
            cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cam1-calibration", required=True, help="Path to camera 1's full calibration JSON")
    p.add_argument("--source1", required=True, help="Camera index or video file path for camera 1")
    p.add_argument("--cam2-calibration", required=True, help="Path to camera 2's full calibration JSON")
    p.add_argument("--source2", required=True, help="Camera index or video file path for camera 2")
    p.add_argument("--checkpoint",
                   default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "training", "output_real_40ep", "checkpoint_best_ema.pth"),
                   help="RF-DETR checkpoint path (a real-data-trained one, not the simulation's). "
                        "Defaults to the 40-epoch dataset_real_itc model.")
    p.add_argument("--resolution", type=int, default=640)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--display", action="store_true", help="Show both camera feeds live (requires a display)")
    p.add_argument("--max-frames", type=int, default=None, help="Stop after this many frames (mainly for smoke-testing)")
    args = p.parse_args()

    cam1_cal = CameraCalibration.load(args.cam1_calibration)
    cam2_cal = CameraCalibration.load(args.cam2_calibration)

    run(
        cam1_cal, args.source1, cam2_cal, args.source2, args.checkpoint, args.resolution, args.threshold,
        args.display, args.max_frames,
    )


if __name__ == "__main__":
    main()
