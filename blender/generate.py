"""
End-to-end driver: uploads an image + areas to the running backend, gets a
plan back, and builds a 3D Blender blockout from it.

    py generate.py --image ..\reference_project\dataset_sam3\train\frame_0000.jpg \
        --detection-area 850,250,1280,650 --drop-area 50,550,350,680 --open

Requires the backend already running (see ../README.md).
"""
import argparse
import json
import os
import subprocess
import sys

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"


def parse_rect(s):
    x1, y1, x2, y2 = (float(v) for v in s.split(","))
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def parse_point(s):
    x, y = (float(v) for v in s.split(","))
    return {"x": x, "y": y}


def main():
    p = argparse.ArgumentParser(description="Generate a 3D Blender blockout from the bale-crane simulator.")
    p.add_argument("--image", required=True, help="Path to the source image")
    p.add_argument("--detection-area", required=True, type=parse_rect, help="x1,y1,x2,y2 in source-image pixels")
    p.add_argument("--drop-area", required=True, type=parse_rect, help="x1,y1,x2,y2 in source-image pixels")
    p.add_argument("--crane-home", type=parse_point, default=None, help="x,y in source-image pixels (optional)")
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--max-bales", type=int, default=None)
    p.add_argument("--api", default="http://127.0.0.1:8000")
    p.add_argument("--bale-width-m", type=float, default=1.0)
    p.add_argument("--out", default=os.path.join(SCRIPT_DIR, "scene.blend"))
    p.add_argument("--sim-json", default=os.path.join(SCRIPT_DIR, "_last_sim.json"), help="Where to cache the raw simulate() response")
    p.add_argument("--blender", default=DEFAULT_BLENDER, help="Path to blender.exe")
    p.add_argument("--open", action="store_true", help="Launch Blender's GUI on the result once built")
    args = p.parse_args()

    print(f"[generate] uploading {args.image} ...")
    with open(args.image, "rb") as f:
        up = requests.post(f"{args.api}/api/upload", files={"file": f})
    up.raise_for_status()
    image_id = up.json()["image_id"]
    print(f"[generate] image_id = {image_id}")

    req = {
        "image_id": image_id,
        "detection_area": args.detection_area,
        "drop_area": args.drop_area,
        "crane_home": args.crane_home,
        "threshold": args.threshold,
        "max_bales": args.max_bales,
    }
    print("[generate] running detection + planning ...")
    sim = requests.post(f"{args.api}/api/simulate", json=req)
    sim.raise_for_status()
    result = sim.json()
    print(f"[generate] {len(result['detections'])} bales detected, "
          f"{len(result['pick_cycles'])} scheduled, "
          f"{result['total_duration_ms'] / 1000:.1f}s total")
    if result["warnings"]:
        for w in result["warnings"]:
            print(f"[generate] WARNING: {w}")

    with open(args.sim_json, "w", encoding="utf-8") as f:
        json.dump(result, f)

    if not os.path.exists(args.blender):
        print(f"[generate] ERROR: blender.exe not found at {args.blender}. Pass --blender <path>.")
        sys.exit(1)

    build_script = os.path.join(SCRIPT_DIR, "build_scene.py")
    cmd = [
        args.blender, "--background", "--python", build_script, "--",
        "--sim", args.sim_json, "--out", args.out, "--bale-width-m", str(args.bale_width_m),
    ]
    print(f"[generate] running Blender headlessly: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    print(f"[generate] done -> {args.out}")
    if args.open:
        subprocess.Popen([args.blender, args.out])


if __name__ == "__main__":
    main()
