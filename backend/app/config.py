import os

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BACKEND_DIR)


def _load_dotenv(path: str) -> None:
    """Minimal, dependency-free .env loader: `KEY=value` lines from
    backend/.env are copied into os.environ unless the var is already set
    (a real environment variable always wins). Blank lines, `#` comments and
    an optional `export ` prefix are ignored; surrounding quotes are
    stripped. Kept tiny on purpose - python-dotenv isn't a backend dep."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(os.path.join(BACKEND_DIR, ".env"))

# RF-DETR Nano fine-tuned on the REAL ITC footage dataset (dataset_real_itc):
# ~53k SAM3-labelled bale boxes at 97% frame coverage + 873 hand-verified
# crane_hook boxes (each with a contact-tip keypoint, though this detection
# checkpoint ignores the keypoints). 40-epoch run - see training/README.md
# and training/output_real_40ep/metrics.csv.
#   val mAP@50-95 (EMA): 0.508   bale AP: 0.82   crane_hook AP: 0.17
# This replaces the earlier Blender/Three.js-render-lineage checkpoint that
# lived in reference_project/edge_deploy/ (removed); the real-footage model
# is a strict upgrade for both classes on real camera frames.
#
# Override with BALE_CHECKPOINT_PATH to point at a different checkpoint
# (e.g. training/phaseA_15ep/checkpoint_best_ema.pth, the earlier 15-epoch
# Phase A model, mAP 0.475).
CHECKPOINT_PATH = os.environ.get(
    "BALE_CHECKPOINT_PATH",
    os.path.join(REPO_ROOT, "training", "output_real_40ep", "checkpoint_best_ema.pth"),
)

STORAGE_DIR = os.path.join(BACKEND_DIR, "storage")
UPLOAD_DIR = os.path.join(STORAGE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

INFERENCE_RESOLUTION = 640
# The real-footage model (output_real_40ep) is well-calibrated at 0.3: on 20
# held-out valid frames it predicted 912 bale / 11 hook vs 917 / 12 ground
# truth. At 0.6 it dropped every hook and ~35% of bales; at 0.15 it started
# double-boxing. The Live 3D tab already used 0.3 as its own default.
DEFAULT_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# Physical crane-remote bridge (the SF-8DR retrofit controller at 192.168.4.1)
#
# The backend's live-3D control loop decides *which* relay to energize each
# control tick (backend/app/live_controller.py -> /api/control_tick -> main.py).
# Those decisions currently only light up a virtual LED panel in the frontend.
# When CRANE_REMOTE_ENABLED is true, the bridge (backend/app/crane_remote.py)
# ALSO POSTs the matching real relay to the crane remote, so the physical
# remote's indicator LEDs light up in lock-step with the simulation - letting
# an operator verify end-to-end that a given signal (NORTH/SOUTH/EAST/WEST/
# UP/DOWN) actually drives the right physical output.
#
# The remote's channel layout (from 192.168.4.1's own page):
#   0=UP, 1=DOWN, 2=EAST, 3=WEST, 4=SOUTH, 5=NORTH, 6=START, 7=LIGHT
# See RELAY_CHANNEL_MAP in crane_remote.py for the sim-relay -> channel map.
#
# CRANE_REMOTE_BACKEND is the remote's control backend the bridge forces
# before every relay command. On THIS plant's hardware the "Direct GPIO"
# pins are not wired to the physical relays - only "Modbus RTU" (to the
# external RS-485 relay module) actually drives them - so the default is
# "modbus". Set to "gpio" only for a unit whose GPIO pins are wired.
#
# Set CRANE_REMOTE_ENABLED=0 (or leave the URL empty) to run the sim purely
# virtual with no physical signalling - the bridge then silently no-ops.
# CRANE_REMOTE_URL defaults to 192.168.4.1 (the crane's AP controller).
CRANE_REMOTE_ENABLED = os.environ.get("CRANE_REMOTE_ENABLED", "1").lower() in ("1", "true", "yes", "on")
CRANE_REMOTE_URL = os.environ.get("CRANE_REMOTE_URL", "http://192.168.4.1").rstrip("/")
CRANE_REMOTE_BACKEND = os.environ.get("CRANE_REMOTE_BACKEND", "modbus").lower()

# ---------------------------------------------------------------------------
# Physical plant cameras (the two Hikvision units on the plant LAN)
#
# cam101 @ 192.168.1.101 and cam102 @ 192.168.1.102 - the same two cameras
# real_deploy/ calibrates as cam101/cam102 (hooks_cam101.json etc.). The
# backend pulls each camera's RTSP main stream with OpenCV/FFmpeg in a
# background thread (backend/app/camera_feed.py) and re-serves it to the
# browser as MJPEG over /api/cameras/<id>/stream - browsers can't play RTSP
# directly, so this proxy is the bridge into the Live 3D page's feed panels.
#
# Hikvision RTSP path: rtsp://<user>:<pass>@<host>:554/Streaming/Channels/101
# (channel 1, stream 01 = main). The password is URL-encoded when the URL is
# built (camera_feed.py), so an "@" or ":" in CAM_PASS is fine here.
#
# Credentials come from backend/.env (loaded above) or the real environment -
# never hardcoded here. Copy backend/.env.example to backend/.env and fill in
# CAM_USER / CAM_PASS; override CAM101_HOST / CAM102_HOST / CAM_RTSP_PATH there
# too for any other site. Set CAMERAS_ENABLED=0 to disable the proxy entirely
# (the endpoints then report every camera offline and never dial out).
CAMERAS_ENABLED = os.environ.get("CAMERAS_ENABLED", "1").lower() in ("1", "true", "yes", "on")
CAM_USER = os.environ.get("CAM_USER", "")
CAM_PASS = os.environ.get("CAM_PASS", "")
CAM_RTSP_PATH = os.environ.get("CAM_RTSP_PATH", "/Streaming/Channels/101")
CAM_RTSP_PORT = int(os.environ.get("CAM_RTSP_PORT", "554"))

# id -> {name, host}. Order here is the order the frontend shows them in.
CAMERAS = {
    "cam101": {"name": "Camera 101", "host": os.environ.get("CAM101_HOST", "192.168.1.101")},
    "cam102": {"name": "Camera 102", "host": os.environ.get("CAM102_HOST", "192.168.1.102")},
}
