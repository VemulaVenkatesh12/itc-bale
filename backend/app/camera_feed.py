"""
Plant-camera RTSP -> MJPEG proxy.

The two physical Hikvision cameras (cam101 @ 192.168.1.101, cam102 @
192.168.1.102 - the same units real_deploy/ calibrates) only speak RTSP,
which a browser can't play in an <img>/<video>. This module keeps one
background thread per camera that pulls the RTSP main stream with
OpenCV/FFmpeg, holds only the most recent frame as JPEG bytes, and lets the
API re-serve it as either a single snapshot or an endless
multipart/x-mixed-replace MJPEG stream (see main.py's /api/cameras/*).

Design notes:
  - Latest-frame-only: each grabber overwrites a single JPEG buffer under a
    lock. Slow HTTP clients never build up a backlog or slow the grabber -
    they just skip frames. This is a monitoring view, not a recorder.
  - Self-healing: a camera that never comes up, or drops mid-stream, is
    retried forever with a capped backoff. `connected` in status() reflects
    the live state so the UI can show an offline badge.
  - Best-effort and isolated: nothing here can raise into the request path
    or the control loop; a dead camera is just an offline panel.
  - Credentials come from config (backend/.env) and are URL-encoded when the
    RTSP URL is built, so "@" / ":" in the password are fine.
"""
from __future__ import annotations

import os
import threading
import time
import urllib.parse
from typing import Iterator, Optional

# FFmpeg options MUST be set before the first cv2.VideoCapture is constructed.
# TCP transport (UDP drops badly on congested plant Wi-Fi) + a 5s open/read
# timeout in microseconds so a missing camera fails fast into the retry loop
# instead of blocking the grabber thread forever.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    # loglevel;quiet: a lossy H.265 main stream otherwise floods stdout with
    # "Error constructing the frame RPS" / "bad cseq" for every dropped RTP
    # packet - thousands of lines that bury the real log and, in a foreground
    # terminal, make the process look hung. Decoding still recovers on its own.
    "rtsp_transport;tcp|stimeout;5000000|max_delay;500000|loglevel;quiet",
)

import cv2  # noqa: E402  (import after the env var above)

from .config import (
    CAM_PASS,
    CAM_RTSP_PATH,
    CAM_RTSP_PORT,
    CAM_USER,
    CAMERAS,
    CAMERAS_ENABLED,
)

# Grabber -> buffer cadence cap and MJPEG send rate. The cameras run ~25fps;
# 15 is plenty for a monitoring panel and keeps JPEG-encode CPU modest.
_MAX_GRAB_FPS = 15.0
_MJPEG_FPS = 12.0
_JPEG_QUALITY = 80
# Downscale anything wider than this before JPEG encoding (1080p/4MP cameras
# would otherwise push multi-hundred-KB frames to every browser tab).
_MAX_WIDTH = 1280
# Reconnect backoff (seconds): start small, double up to the cap.
_RECONNECT_MIN_S = 1.0
_RECONNECT_MAX_S = 15.0
# A camera with no fresh frame for this long is reported disconnected even if
# the capture object hasn't errored yet (silent RTSP stalls happen).
_STALE_AFTER_S = 8.0


def _build_rtsp_url(host: str) -> str:
    path = CAM_RTSP_PATH if CAM_RTSP_PATH.startswith("/") else "/" + CAM_RTSP_PATH
    if CAM_USER or CAM_PASS:
        auth = f"{urllib.parse.quote(CAM_USER, safe='')}:{urllib.parse.quote(CAM_PASS, safe='')}@"
    else:
        auth = ""
    return f"rtsp://{auth}{host}:{CAM_RTSP_PORT}{path}"


def _redact(url: str) -> str:
    """RTSP URL with credentials masked, for logs."""
    return urllib.parse.urlunparse(
        urllib.parse.urlparse(url)._replace(netloc=url.split("@")[-1].split("/")[0])
    ) if "@" in url else url


class _Camera:
    def __init__(self, cam_id: str, name: str, host: str) -> None:
        self.id = cam_id
        self.name = name
        self.host = host
        self.url = _build_rtsp_url(host)
        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._frame_ts = 0.0
        self._connected = False
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"cam-{self.id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # -- readers ---------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._connected and (time.monotonic() - self._frame_ts) < _STALE_AFTER_S

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def status(self) -> dict:
        age = None
        if self._frame_ts:
            age = round(time.monotonic() - self._frame_ts, 1)
        return {
            "id": self.id,
            "name": self.name,
            "host": self.host,
            "connected": self.connected,
            "last_frame_age_s": age,
        }

    # -- grabber thread ------------------------------------------------------
    def _run(self) -> None:
        backoff = _RECONNECT_MIN_S
        min_dt = 1.0 / _MAX_GRAB_FPS
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except cv2.error:
                pass
            if not cap.isOpened():
                cap.release()
                self._connected = False
                print(f"[camera_feed] {self.id}: cannot open {_redact(self.url)} - retry in {backoff:.0f}s")
                if self._stop.wait(backoff):
                    break
                backoff = min(backoff * 2, _RECONNECT_MAX_S)
                continue

            print(f"[camera_feed] {self.id}: connected to {_redact(self.url)}")
            backoff = _RECONNECT_MIN_S
            last_grab = 0.0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    print(f"[camera_feed] {self.id}: stream ended / read failed - reconnecting")
                    break
                now = time.monotonic()
                if now - last_grab < min_dt:
                    continue
                last_grab = now

                h, w = frame.shape[:2]
                if w > _MAX_WIDTH:
                    scale = _MAX_WIDTH / float(w)
                    frame = cv2.resize(frame, (_MAX_WIDTH, int(round(h * scale))), interpolation=cv2.INTER_AREA)

                enc_ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
                if not enc_ok:
                    continue
                with self._lock:
                    self._jpeg = buf.tobytes()
                    self._frame_ts = now
                self._connected = True

            cap.release()
            self._connected = False
            if not self._stop.is_set():
                if self._stop.wait(backoff):
                    break
                backoff = min(backoff * 2, _RECONNECT_MAX_S)


# --- module-level registry ------------------------------------------------
_cameras: "dict[str, _Camera]" = {}
_init_lock = threading.Lock()


def _ensure_started() -> None:
    if not CAMERAS_ENABLED:
        return
    with _init_lock:
        if _cameras:
            return
        for cam_id, meta in CAMERAS.items():
            cam = _Camera(cam_id, meta["name"], meta["host"])
            cam.start()
            _cameras[cam_id] = cam
        if not (CAM_USER or CAM_PASS):
            print("[camera_feed] WARNING: no CAM_USER/CAM_PASS set - trying RTSP without auth")


def start_all() -> None:
    """Called from FastAPI startup so the cameras are already warming up
    before the first browser request. Safe to call more than once."""
    _ensure_started()


def stop_all() -> None:
    for cam in _cameras.values():
        cam.stop()


def list_status() -> list[dict]:
    if not CAMERAS_ENABLED:
        return [
            {"id": cid, "name": m["name"], "host": m["host"], "connected": False, "last_frame_age_s": None}
            for cid, m in CAMERAS.items()
        ]
    _ensure_started()
    return [_cameras[cid].status() for cid in CAMERAS if cid in _cameras]


def get_snapshot(cam_id: str) -> Optional[bytes]:
    if not CAMERAS_ENABLED:
        return None
    _ensure_started()
    cam = _cameras.get(cam_id)
    return cam.latest_jpeg() if cam else None


def get_frame_ts(cam_id: str) -> float:
    """Monotonic timestamp of the camera's current buffered frame (0.0 if
    none yet). Lets a caller (e.g. main.py's annotated stream) tell "new
    frame arrived" from "same frame as last time" without reaching into the
    _Camera internals directly."""
    cam = _cameras.get(cam_id)
    return cam._frame_ts if cam else 0.0


def has_camera(cam_id: str) -> bool:
    return cam_id in CAMERAS


_BOUNDARY = "frame"


def mjpeg_stream(cam_id: str) -> Iterator[bytes]:
    """multipart/x-mixed-replace generator: emits the latest JPEG at
    ~_MJPEG_FPS until the client disconnects. Skips ticks where no new frame
    is available yet rather than resending a stale one."""
    _ensure_started()
    cam = _cameras.get(cam_id)
    dt = 1.0 / _MJPEG_FPS
    last_sent_ts = -1.0
    idle_deadline = time.monotonic() + 30.0
    while cam is not None:
        jpeg = cam.latest_jpeg()
        fresh = cam._frame_ts if cam else 0.0
        if jpeg is not None and fresh != last_sent_ts:
            last_sent_ts = fresh
            idle_deadline = time.monotonic() + 30.0
            yield (
                b"--" + _BOUNDARY.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                + jpeg + b"\r\n"
            )
        elif time.monotonic() > idle_deadline:
            # No frame for 30s - close the response so the browser <img>
            # onerror fires and the UI can show/refresh the offline state.
            break
        time.sleep(dt)


def multipart_content_type() -> str:
    return f"multipart/x-mixed-replace; boundary={_BOUNDARY}"
