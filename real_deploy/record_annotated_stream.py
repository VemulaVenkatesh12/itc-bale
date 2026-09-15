"""
Records the LIVE annotated stream (backend/app/main.py's
/api/cameras/<id>/stream_annotated - real model boxes burned into real live
frames, see that endpoint's docstring) to an mp4 file for a fixed duration.
Same detections VLC would show if pointed at that URL; this just saves them.

Usage:
    python3 record_annotated_stream.py --cam cam101 --duration-s 600 \
        --out ~/Desktop/live_detect_cam101.mp4
"""
from __future__ import annotations

import argparse
import os
import signal
import time

import cv2
import numpy as np
import urllib.request

# Stopping this any other way (bare `kill`/pkill -9) skips VideoWriter.release(),
# which never writes the MP4 index (moov atom) - the whole file is then
# unplayable, not just truncated. SIGTERM/SIGINT are caught below and turned
# into a clean loop exit so `kill <pid>` or Ctrl-C always finalizes the file.
_stop = False


def _request_stop(signum, frame) -> None:
    global _stop
    _stop = True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cam", required=True, help="cam101 or cam102")
    p.add_argument("--duration-s", type=float, default=600.0)
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=0.15)
    p.add_argument("--backend-url", default=os.environ.get("BACKEND_URL", "http://127.0.0.1:8000"))
    p.add_argument("--fps", type=float, default=5.0, help="Output container fps (the stream is inference-bound, ~5-6fps single-viewer)")
    args = p.parse_args()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    url = f"{args.backend_url}/api/cameras/{args.cam}/stream_annotated?threshold={args.threshold}"
    print(f"connecting: {url}")
    resp = urllib.request.urlopen(url, timeout=15)

    writer = None
    buf = b""
    t_end = time.time() + args.duration_s
    n_frames = 0
    t0 = time.time()

    while time.time() < t_end and not _stop:
        chunk = resp.read(65536)
        if not chunk:
            break
        buf += chunk
        while True:
            start = buf.find(b"\xff\xd8")
            end = buf.find(b"\xff\xd9", start + 2) if start != -1 else -1
            if start == -1 or end == -1:
                break
            jpeg = buf[start:end + 2]
            buf = buf[end + 2:]
            img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(os.path.expanduser(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
                print(f"recording {w}x{h} @ {args.fps}fps -> {args.out}")
            writer.write(img)
            n_frames += 1
            if n_frames % 20 == 0:
                elapsed = time.time() - t0
                print(f"  {n_frames} frames, {elapsed:.0f}s elapsed, actual {n_frames/elapsed:.2f} fps")

    resp.close()
    if writer is not None:
        writer.release()
    dt = time.time() - t0
    print(f"\nDONE  {n_frames} frames over {dt:.0f}s (actual {n_frames/dt:.2f} fps)  -> {args.out}")


if __name__ == "__main__":
    main()
