"""
Sample live snapshots from both plant cameras into real_deploy/hook_pose_gap/
<cam_id>/ for later labeling - see README.md for why (the model's training
data under-covers the hook's retracted/occluded pose).

Saves a frame only when it differs meaningfully from the last one KEPT for
that camera, so pointing this at an idle crane doesn't fill the folder with
near-duplicates. Requires the backend to be running (BACKEND_URL).

Usage:
    python3 collect_frames.py --once                # one snapshot per camera, always saved
    python3 collect_frames.py --every-s 30           # keep sampling until Ctrl-C
    python3 collect_frames.py --every-s 30 --diff-threshold 8.0
"""
from __future__ import annotations

import argparse
import os
import time
import urllib.request

import cv2
import numpy as np

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
CAMERAS = ["cam101", "cam102"]
HERE = os.path.dirname(os.path.abspath(__file__))


def fetch_snapshot(cam_id: str) -> np.ndarray | None:
    try:
        data = urllib.request.urlopen(f"{BACKEND_URL}/api/cameras/{cam_id}/snapshot", timeout=6).read()
    except Exception as e:
        print(f"  {cam_id}: snapshot fetch failed ({e})")
        return None
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def frame_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean abs pixel diff on a downscaled grayscale copy - cheap and good
    enough to tell "the load/crane moved" from "identical idle frame"."""
    ga = cv2.cvtColor(cv2.resize(a, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.int16)
    gb = cv2.cvtColor(cv2.resize(b, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.int16)
    return float(np.abs(ga - gb).mean())


def sample_once(last_kept: dict, diff_threshold: float, force: bool) -> None:
    for cam_id in CAMERAS:
        out_dir = os.path.join(HERE, cam_id)
        os.makedirs(out_dir, exist_ok=True)
        img = fetch_snapshot(cam_id)
        if img is None:
            continue
        prev = last_kept.get(cam_id)
        d = frame_diff(prev, img) if prev is not None else diff_threshold + 1
        if not force and d < diff_threshold:
            print(f"  {cam_id}: skipped (diff {d:.1f} < {diff_threshold})")
            continue
        ts = time.strftime("%Y-%m-%dT%H-%M-%S")
        path = os.path.join(out_dir, f"{ts}.jpg")
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        last_kept[cam_id] = img
        print(f"  {cam_id}: saved {path} (diff {d:.1f})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--once", action="store_true", help="Grab one frame per camera and exit (always saved)")
    p.add_argument("--every-s", type=float, default=30.0, help="Sampling interval when not --once")
    p.add_argument("--diff-threshold", type=float, default=6.0,
                   help="Min mean grayscale diff (0-255 scale) vs the last KEPT frame to save a new one")
    args = p.parse_args()

    last_kept: dict[str, np.ndarray] = {}
    if args.once:
        print("grabbing one frame per camera...")
        sample_once(last_kept, args.diff_threshold, force=True)
        return

    print(f"sampling every {args.every_s}s, diff threshold {args.diff_threshold} - Ctrl-C to stop")
    try:
        while True:
            sample_once(last_kept, args.diff_threshold, force=False)
            time.sleep(args.every_s)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
