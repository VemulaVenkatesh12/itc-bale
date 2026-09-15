"""
Pulls live snapshots from a camera and keeps only the ones where a 9x6
checkerboard is actually detected - immediate feedback while someone is
physically holding the board in front of the camera, instead of recording a
video and discovering afterward that half the shots missed the corners.

Usage:
    python3 grab_checkerboard.py --cam cam101 --every-s 3 --target 20
    # Ctrl-C to stop early once you have enough.
"""
from __future__ import annotations

import argparse
import os
import time
import urllib.request

import cv2
import numpy as np

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cam", required=True)
    p.add_argument("--board-cols", type=int, default=9)
    p.add_argument("--board-rows", type=int, default=6)
    p.add_argument("--every-s", type=float, default=3.0)
    p.add_argument("--target", type=int, default=20, help="Stop once this many good frames are saved")
    args = p.parse_args()

    out_dir = os.path.join(HERE, "calib_images", args.cam)
    os.makedirs(out_dir, exist_ok=True)
    existing = len([f for f in os.listdir(out_dir) if f.endswith(".jpg")])
    saved = 0
    attempt = 0

    print(f"watching {args.cam} for a {args.board_cols}x{args.board_rows} checkerboard - "
          f"move the board around (near/far/tilted/into every corner). Ctrl-C to stop.")
    try:
        while saved < args.target:
            attempt += 1
            try:
                data = urllib.request.urlopen(f"{BACKEND_URL}/api/cameras/{args.cam}/snapshot", timeout=8).read()
            except Exception as e:
                print(f"  [{attempt}] snapshot fetch failed: {e}")
                time.sleep(args.every_s)
                continue
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                time.sleep(args.every_s)
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, (args.board_cols, args.board_rows),
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if found:
                idx = existing + saved
                path = os.path.join(out_dir, f"{idx:03d}.jpg")
                cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
                print(f"  [{attempt}] FOUND - saved {path}  ({saved}/{args.target})")
            else:
                print(f"  [{attempt}] no board detected - keep moving it into view")
            time.sleep(args.every_s)
    except KeyboardInterrupt:
        pass
    print(f"\ndone: {saved} new frames saved to {out_dir} ({existing + saved} total)")


if __name__ == "__main__":
    main()
