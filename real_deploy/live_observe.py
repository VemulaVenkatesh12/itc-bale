"""
Passive observer for a human-operated pick cycle (2026-09-12): the operator
drives the real crane via the physical SF-8DR remote directly - this script
never touches crane_remote, it only watches the cameras and logs what it
sees, timestamped, for later correlation with what the operator did and
when. Ctrl-C to stop.
"""
from __future__ import annotations

import json
import time
import urllib.request

BACKEND = "http://127.0.0.1:8000"
CAM = "cam101"
INTERVAL_S = 1.0
OUT_PATH = "/tmp/live_observe_log.jsonl"


def get(url, timeout=10):
    return json.loads(urllib.request.urlopen(url, timeout=timeout).read())


def main():
    print(f"watching {CAM} every {INTERVAL_S}s - Ctrl-C to stop. Logging to {OUT_PATH}")
    with open(OUT_PATH, "a") as f:
        while True:
            ts = time.strftime("%H:%M:%S")
            record = {"t": ts}
            try:
                det = get(f"{BACKEND}/api/cameras/{CAM}/detect?threshold=0.3", timeout=8)
                bales = [d for d in det["detections"] if d["class_id"] == 0]
                hooks = [d for d in det["detections"] if d["class_id"] == 1]
                record["bale_count"] = len(bales)
                record["hook"] = max(hooks, key=lambda d: d["confidence"]) if hooks else None
            except Exception as e:
                record["detect_error"] = str(e)

            try:
                bnd = get(f"{BACKEND}/api/boundary_guard/status", timeout=8)
                record["boundary"] = bnd["cameras"].get(CAM, {})
                record["violations"] = bnd["violation_count"]
            except Exception as e:
                record["boundary_error"] = str(e)

            try:
                cs = get(f"{BACKEND}/api/cameras/{CAM}/conveyor_status?threshold=0.3", timeout=8)
                record["conveyor"] = cs
            except Exception as e:
                record["conveyor_error"] = str(e)

            line = json.dumps(record)
            f.write(line + "\n")
            f.flush()

            hook_txt = f"hook@{record['hook']['bbox']}" if record.get("hook") else "no_hook"
            print(f"[{ts}] bales={record.get('bale_count','?')} {hook_txt} "
                  f"boundary={record.get('boundary',{}).get('status','?')} "
                  f"conveyor={record.get('conveyor',{}).get('count','?')}")
            time.sleep(INTERVAL_S)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped")
