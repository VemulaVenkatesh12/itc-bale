"""
Run the real-footage RF-DETR checkpoint over a recorded video and write an
annotated copy with bale / crane_hook / crane_spike boxes drawn on.

Review aid only - it does not calibrate, triangulate, or touch any relay.
Same model and class order as capture_infer.py.

Example:
    python3 annotate_video.py \
        --source ~/Documents/ITC_bale_unloading_simulator/192.168.1.101_01_20260825143221172.mp4 \
        --start-s 90 --end-s 330 --stride 2 --width 1280 \
        --out ~/Documents/ITC_bale_unloading_simulator/annotated_cam101_1432.mp4
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

CLASS_NAMES = {0: "bale", 1: "crane_hook", 2: "crane_spike"}
CLASS_BGR = {0: (80, 200, 80), 1: (0, 150, 255), 2: (40, 40, 235)}


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def dedup(result, iou_thr: float):
    """Class-aware greedy NMS. RF-DETR is set-prediction so this normally
    removes nothing; it's a safety net against the occasional near-identical
    pair. Keep iou_thr high (>=0.7) so genuinely adjacent bales - which touch
    and so share real IoU - are never merged. Returns kept indices."""
    idx = sorted(range(len(result.xyxy)), key=lambda i: -float(result.confidence[i]))
    kept: list[int] = []
    for i in idx:
        bi, ci = result.xyxy[i], int(result.class_id[i])
        if all(ci != int(result.class_id[k]) or _iou(bi, result.xyxy[k]) < iou_thr for k in kept):
            kept.append(i)
    return kept
DEFAULT_CKPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "training", "output_real_40ep", "checkpoint_best_ema.pth",
)


def load_model(checkpoint: str, resolution: int):
    import torch
    from rfdetr import RFDETRNano

    print(f"loading model: {checkpoint}")
    model = RFDETRNano(resolution=resolution, pretrain_weights=checkpoint)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    if torch.backends.mkldnn.is_available():
        torch.backends.mkldnn.enabled = True
    model.optimize_for_inference()
    return model


def draw(frame: np.ndarray, result, scale: float, keep=None) -> dict:
    counts = {0: 0, 1: 0, 2: 0}
    if len(result) == 0:
        return counts
    keep = range(len(result.xyxy)) if keep is None else keep
    for i in keep:
        box, conf, cid = result.xyxy[i], result.confidence[i], int(result.class_id[i])
        counts[cid] = counts.get(cid, 0) + 1
        x1, y1, x2, y2 = [int(round(float(v) * scale)) for v in box]
        color = CLASS_BGR.get(cid, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES.get(cid, cid)} {float(conf):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--resolution", type=int, default=640)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--start-s", type=float, default=0.0)
    p.add_argument("--end-s", type=float, default=None)
    p.add_argument("--stride", type=int, default=1, help="Run detection on every Nth frame")
    p.add_argument("--width", type=int, default=1280, help="Output width (keeps aspect)")
    p.add_argument("--nms-iou", type=float, default=0.0,
                   help="Drop same-class boxes overlapping an already-kept one by >= this IoU "
                        "(0 = off; RF-DETR rarely needs it). Use 0.7+ so touching bales aren't merged.")
    args = p.parse_args()

    cap = cv2.VideoCapture(os.path.expanduser(args.source))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scale = args.width / src_w
    out_w = args.width
    out_h = int(round(src_h * scale))
    out_fps = src_fps / max(1, args.stride)

    start_f = int(args.start_s * src_fps)
    end_f = int(args.end_s * src_fps) if args.end_s else n_frames
    end_f = min(end_f, n_frames)

    out_path = os.path.expanduser(args.out)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (out_w, out_h))
    if not writer.isOpened():
        raise SystemExit(f"cannot open writer for {out_path}")

    model = load_model(args.checkpoint, args.resolution)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    print(f"source {src_w}x{src_h}@{src_fps:.1f}  ->  {out_w}x{out_h}@{out_fps:.1f}")
    print(f"frames {start_f}..{end_f}  stride {args.stride}  threshold {args.threshold}")

    fidx = start_f
    done = 0
    tot = {0: 0, 1: 0, 2: 0}
    frames_with_hook = 0
    t0 = time.time()
    last_result = None
    while fidx < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        if (fidx - start_f) % args.stride == 0:
            small = cv2.resize(frame, (out_w, out_h))
            result = model.predict(frame, threshold=args.threshold)
            last_result = result
            keep = dedup(result, args.nms_iou) if args.nms_iou > 0 else None
            counts = draw(small, result, scale, keep)
            for k, v in counts.items():
                tot[k] = tot.get(k, 0) + v
            if counts.get(1, 0) > 0:
                frames_with_hook += 1
            t = fidx / src_fps
            hud = f"t={t:6.1f}s  bale:{counts.get(0,0)} hook:{counts.get(1,0)} spike:{counts.get(2,0)}"
            cv2.putText(small, hud, (10, out_h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(small, hud, (10, out_h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(small)
            done += 1
            if done % 100 == 0:
                dt = time.time() - t0
                print(f"  {done} frames  {dt/done*1000:.0f} ms/frame  "
                      f"(bale {tot[0]}, hook {tot[1]}, spike {tot[2]})", flush=True)
        fidx += 1

    cap.release()
    writer.release()
    dt = time.time() - t0
    print(f"\nDONE  {done} annotated frames in {dt:.0f}s")
    print(f"  total boxes: bale {tot[0]}  crane_hook {tot[1]}  crane_spike {tot[2]}")
    print(f"  frames with >=1 hook: {frames_with_hook}/{done}")
    print(f"  saved: {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
