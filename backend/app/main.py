from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import (
    boundary_guard, camera_feed, conveyor_gate, crane_remote, fixed_cycle, horizontal_servo, scale_calibration,
    live_controller, model_service, pick_order,
)
from .config import DEFAULT_THRESHOLD, UPLOAD_DIR
from .planner import plan_simulation
from .schemas import ControlTickResponse, DetectResponse, SimulateRequest, SimulationResult, UploadResponse

app = FastAPI(title="Bale Crane Simulation Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # POC only — restrict before any real deployment
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@app.on_event("startup")
def _startup() -> None:
    # Warm the model in a background thread rather than inline here: on a
    # pre-Ampere GPU optimize_for_inference() can take tens of seconds, and
    # blocking the startup hook means uvicorn never binds / never answers
    # /api/health until it finishes (looks like a hung server). The first
    # inference still lazily loads it if warm-up hasn't finished yet.
    def _warm() -> None:
        try:
            model_service.warmup()
            print("[startup] model warm")
        except FileNotFoundError as e:
            print(f"[startup] WARNING: {e}")
        except Exception as e:  # noqa: BLE001 - never kill startup over warm-up
            print(f"[startup] model warm-up failed: {e}")

    threading.Thread(target=_warm, name="model-warmup", daemon=True).start()

    # Start the plant-camera RTSP grabbers (backend/app/camera_feed.py) so
    # they're already connected before the Live 3D page asks for a stream.
    # Best-effort: a missing camera just retries in the background.
    camera_feed.start_all()

    # Vision safety boundary (2026-09-12, see boundary_guard.py) - watches
    # for the hook crossing the keep-in region and calls crane_remote.stop_all()
    # if it does. Needs no calibration (pure pixel containment), runs
    # independently of everything else, active from the moment the backend
    # is up.
    boundary_guard.start()


@app.on_event("shutdown")
def _shutdown() -> None:
    camera_feed.stop_all()
    boundary_guard.stop()


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/cameras")
def cameras_list():
    """Status of the physical plant cameras (cam101 @ 192.168.1.101, cam102
    @ 192.168.1.102). Each entry: {id, name, host, connected,
    last_frame_age_s}. Never raises - a down camera just reports
    connected:false. See camera_feed.py."""
    return {"cameras": camera_feed.list_status()}


@app.get("/api/cameras/{cam_id}/stream")
def camera_stream(cam_id: str):
    """Endless MJPEG (multipart/x-mixed-replace) proxy of one camera's RTSP
    main stream, for a plain <img src> in the Live 3D page. Latest-frame-only:
    a slow client skips frames, it never backs up the grabber."""
    if not camera_feed.has_camera(cam_id):
        raise HTTPException(404, f"Unknown camera '{cam_id}'.")
    return StreamingResponse(
        camera_feed.mjpeg_stream(cam_id),
        media_type=camera_feed.multipart_content_type(),
        headers={"Cache-Control": "no-store", "Connection": "close"},
    )


@app.get("/api/cameras/{cam_id}/snapshot")
def camera_snapshot(cam_id: str):
    """Single most-recent JPEG from one camera - a lightweight poll
    alternative to the MJPEG stream (and the fallback the UI uses if the
    stream connection drops)."""
    if not camera_feed.has_camera(cam_id):
        raise HTTPException(404, f"Unknown camera '{cam_id}'.")
    jpeg = camera_feed.get_snapshot(cam_id)
    if jpeg is None:
        raise HTTPException(503, f"Camera '{cam_id}' has no frame yet (still connecting?).")
    return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/cameras/{cam_id}/detect", response_model=DetectResponse)
def camera_detect(cam_id: str, threshold: float = DEFAULT_THRESHOLD):
    """Run the RF-DETR detector (training/output_real_40ep, the 40-epoch
    real-footage model) on this camera's most recent frame and return the
    raw multi-class detections (bale / crane_hook / crane_spike) in that
    frame's pixel space. The Live 3D page's plant-camera panels poll this a
    couple of times a second and draw the boxes as an overlay - it's a
    display-only view of what the model sees on the REAL cameras, it does
    not touch the control loop."""
    if not camera_feed.has_camera(cam_id):
        raise HTTPException(404, f"Unknown camera '{cam_id}'.")
    jpeg = camera_feed.get_snapshot(cam_id)
    if jpeg is None:
        raise HTTPException(503, f"Camera '{cam_id}' has no frame yet (still connecting?).")

    img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(500, "Latest camera frame could not be decoded.")
    h, w = img.shape[:2]

    try:
        detections = model_service.detect_all(img, threshold=threshold)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))

    return DetectResponse(width=w, height=h, detections=detections)


_ANNOTATED_CLASS_BGR = {0: (80, 200, 80), 1: (0, 150, 255), 2: (40, 40, 235)}  # bale, crane_hook, crane_spike
_ANNOTATED_CLASS_NAME = {0: "bale", 1: "crane_hook", 2: "crane_spike"}


def _draw_detections(img: np.ndarray, detections) -> None:
    """Burns real model output into the frame pixels (in place) - same
    boxes/labels DetectResponse would return, just rasterized so a plain
    video player (VLC, an <img>, anything that only understands pixels and
    has no way to render a separate synced overlay) can show them."""
    for d in detections:
        color = _ANNOTATED_CLASS_BGR.get(d.class_id, (255, 255, 255))
        x1, y1, x2, y2 = int(d.bbox.x1), int(d.bbox.y1), int(d.bbox.x2), int(d.bbox.y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{_ANNOTATED_CLASS_NAME.get(d.class_id, d.class_id)} {d.confidence:.2f}"
        cv2.putText(img, label, (x1 + 2, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _annotated_mjpeg_stream(cam_id: str, threshold: float):
    """Same multipart/x-mixed-replace shape as camera_feed.mjpeg_stream(),
    but runs the REAL detector on each frame and burns the boxes in before
    encoding - so any plain video client (VLC's Open Network Stream, not
    just our React overlay) can watch detection happen. Inference-bound
    (~170-190ms/frame on this GPU - see model_service.py), so this stream
    runs at whatever fps that caps out to, not the raw feed's ~12fps; that's
    the real cost of "boxes burned into the video" vs. a separate JS overlay,
    not a bug. A camera with no model yet loaded just streams the raw
    (unboxed) frame rather than erroring the whole connection."""
    import time as _time

    last_frame_ts = -1.0
    idle_deadline = _time.monotonic() + 30.0
    while camera_feed.has_camera(cam_id):
        jpeg = camera_feed.get_snapshot(cam_id)
        fresh_ts = camera_feed.get_frame_ts(cam_id)
        if jpeg is None or fresh_ts == last_frame_ts:
            if _time.monotonic() > idle_deadline:
                break
            _time.sleep(0.05)
            continue
        last_frame_ts = fresh_ts
        idle_deadline = _time.monotonic() + 30.0

        img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        try:
            detections = model_service.detect_all(img, threshold=threshold)
            _draw_detections(img, detections)
        except FileNotFoundError:
            pass  # no checkpoint loaded yet - stream the raw frame rather than dying

        enc_ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not enc_ok:
            continue
        out = buf.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(out)).encode() + b"\r\n\r\n"
            + out + b"\r\n"
        )


@app.get("/api/cameras/{cam_id}/stream_annotated")
def camera_stream_annotated(cam_id: str, threshold: float = DEFAULT_THRESHOLD):
    """MJPEG stream with detection boxes burned into the pixels - point VLC's
    'Open Network Stream' (or any plain video client) straight at this URL
    to watch the real detector work outside the React app. Not for the
    React UI itself (PlantCameraFeed.tsx draws its own SVG overlay over the
    raw /stream so boxes stay crisp at any zoom); this exists specifically
    for clients that can only play pixels, not overlay separate JSON. See
    /api/cameras/{cam_id}/detect for the same detections as structured data."""
    if not camera_feed.has_camera(cam_id):
        raise HTTPException(404, f"Unknown camera '{cam_id}'.")
    return StreamingResponse(
        _annotated_mjpeg_stream(cam_id, threshold),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "Connection": "close"},
    )


# One tracker per camera so pick numbers stay stable across polls as bales
# are physically removed - see pick_order.PickOrderTracker. `start_xy`
# defaults to a top-left "home corner" stand-in for the crane's position;
# there is no real crane position from a single 2D camera (see pick_order.py's
# module docstring) - pass a better one via ?start_x=&start_y= if useful.
_pick_trackers: dict[str, pick_order.PickOrderTracker] = {}


def _get_tracker(cam_id: str, start_xy: tuple[float, float]) -> pick_order.PickOrderTracker:
    t = _pick_trackers.get(cam_id)
    if t is None:
        t = pick_order.PickOrderTracker(start_xy, cam_id=cam_id)
        _pick_trackers[cam_id] = t
    return t


@app.get("/api/cameras/{cam_id}/pick_order")
def camera_pick_order(
    cam_id: str, threshold: float = DEFAULT_THRESHOLD,
    max_layers: int = 2, start_x: float = 0.0, start_y: float = 0.0,
):
    """Numbered pick order for this camera's bales - topmost pixel-layer
    first, nearest-neighbour within a layer (2026-09-11 confirmed rule).
    PIXEL-SPACE APPROXIMATION, not real height - see pick_order.py's
    docstring for exactly what that means and why. `max_layers<=0` numbers
    everything visible instead of just the top two."""
    if not camera_feed.has_camera(cam_id):
        raise HTTPException(404, f"Unknown camera '{cam_id}'.")
    jpeg = camera_feed.get_snapshot(cam_id)
    if jpeg is None:
        raise HTTPException(503, f"Camera '{cam_id}' has no frame yet (still connecting?).")
    img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(500, "Latest camera frame could not be decoded.")
    h, w = img.shape[:2]

    try:
        detections = model_service.detect_all(img, threshold=threshold)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))

    tracker = _get_tracker(cam_id, (start_x, start_y))
    tracker.max_layers = max_layers if max_layers > 0 else None
    numbered = tracker.update(detections)
    return {
        "width": w, "height": h,
        "picks": [
            {"number": n.number, "layer": n.layer, "bbox": n.detection.bbox, "confidence": n.detection.confidence}
            for n in numbered
        ],
    }


# Both plant cameras are currently aimed at the same left-truck stack from
# different angles (cam101 - overhead/top; cam102 - eye-level/front) - see
# the 2026-09-12 conversation. There's no calibration yet to say "this bale
# in cam101 IS this bale in cam102" pixel-for-pixel; what this DOES check,
# honestly, is whether the two independent top-N-layer reads roughly AGREE
# in count. A big mismatch is a real, actionable signal (one camera is
# occluded, mis-aimed, or its ROI is stale) even without per-bale matching.
CROSS_VIEW_MIN_ABS_TOLERANCE = 2
CROSS_VIEW_REL_TOLERANCE = 0.3


def _agree(a: Optional[int], b: Optional[int]) -> Optional[bool]:
    if a is None or b is None:
        return None
    tolerance = max(CROSS_VIEW_MIN_ABS_TOLERANCE, CROSS_VIEW_REL_TOLERANCE * max(a, b))
    return abs(a - b) <= tolerance


@app.get("/api/pick_order/cross_view_status")
def pick_order_cross_view_status(
    cam_a: str = "cam101", cam_b: str = "cam102",
    threshold: float = DEFAULT_THRESHOLD, max_layers: int = 2,
):
    """Two-camera sanity check on the top-N-layer pick order (2026-09-12,
    see module note above the CROSS_VIEW_* constants): NOT per-bale identity
    matching (needs calibration, not done) - just whether both independent
    views roughly AGREE on (a) how many bales are in play and (b) how many
    ROWS the top layer is currently read as having (see
    pick_order.group_into_rows_px) - the row count is what the "checking the
    rows" cross-check from the front view (cam102) is actually looking at,
    same count-level honesty as the bale check: no per-row identity matching
    across cameras, just does the row structure roughly line up. Either
    camera missing/no frame yet reports that camera's numbers as null and
    the corresponding agree flag as None (can't judge with only one view)."""
    counts: dict[str, Optional[int]] = {}
    row_counts: dict[str, Optional[int]] = {}
    for cam_id in (cam_a, cam_b):
        if not camera_feed.has_camera(cam_id):
            counts[cam_id] = row_counts[cam_id] = None
            continue
        jpeg = camera_feed.get_snapshot(cam_id)
        if jpeg is None:
            counts[cam_id] = row_counts[cam_id] = None
            continue
        img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            counts[cam_id] = row_counts[cam_id] = None
            continue
        try:
            detections = model_service.detect_all(img, threshold=threshold)
        except FileNotFoundError:
            counts[cam_id] = row_counts[cam_id] = None
            continue

        tracker = _get_tracker(cam_id, (0.0, 0.0))
        tracker.max_layers = max_layers if max_layers > 0 else None
        counts[cam_id] = len(tracker.update(detections))

        bales = [d for d in detections if d.class_id == pick_order.CLASS_BALE
                 and pick_order.in_scope(d, cam_id)]
        layers = pick_order.group_into_layers_px(bales, tracker.layer_band_px)[:max_layers]
        row_counts[cam_id] = max(
            (len(pick_order.group_into_rows_px(layer, tracker.row_band_px)) for layer in layers),
            default=0,
        )

    count_a, count_b = counts[cam_a], counts[cam_b]
    row_a, row_b = row_counts[cam_a], row_counts[cam_b]

    return {
        "cam_a": cam_a, "cam_b": cam_b,
        "count_a": count_a, "count_b": count_b,
        "agree": _agree(count_a, count_b),
        "row_count_a": row_a, "row_count_b": row_b,
        "rows_agree": _agree(row_a, row_b),
        "note": (
            "count-level and row-count-level agreement only - "
            "no per-bale/per-row cross-camera identity matching without calibration"
        ),
    }


@app.get("/api/cameras/{cam_id}/conveyor_status")
def camera_conveyor_status(cam_id: str, threshold: float = DEFAULT_THRESHOLD, full_threshold: int = conveyor_gate.DEFAULT_FULL_THRESHOLD):
    """Requirement 3 (2026-09-11): is the conveyor/drop area already full of
    bales that haven't been carried away? {"count", "full", "ok_to_pick", ...}
    - ok_to_pick=False means hold every pick signal until this flips back.
    Pixel-ROI based (see conveyor_gate.py) - needs NO calibration, unlike
    real pick/place, so this is usable today. `roi_configured=False` for a
    camera with no traced conveyor region (currently only cam101)."""
    if not camera_feed.has_camera(cam_id):
        raise HTTPException(404, f"Unknown camera '{cam_id}'.")
    jpeg = camera_feed.get_snapshot(cam_id)
    if jpeg is None:
        raise HTTPException(503, f"Camera '{cam_id}' has no frame yet (still connecting?).")
    img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(500, "Latest camera frame could not be decoded.")
    try:
        detections = model_service.detect_all(img, threshold=threshold)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    return conveyor_gate.conveyor_status(detections, cam_id, full_threshold)


@app.get("/api/cameras/{cam_id}/servo_decision")
def camera_servo_decision(
    cam_id: str, threshold: float = DEFAULT_THRESHOLD,
    max_layers: int = 2, target_number: int = 1,
    start_x: float = 0.0, start_y: float = 0.0,
):
    """Closed-loop E/W/N/S nudge decision (2026-09-11) - compares the hook's
    CURRENT pixel position to the numbered target bale's pixel position, no
    metric calibration needed for this comparison. See horizontal_servo.py
    for exactly what this does and does not cover (vertical/pierce stroke is
    NOT this - stays under direct human direction).

    DISPLAY-ONLY: returns a suggested nudge, never calls crane_remote.
    `suggested_relay` is null until horizontal_servo.PIXEL_TO_RELAY_MAP has
    been filled in from a real short test (fire a known relay briefly, see
    which way the hook moves in frame) - until then this reports the
    image-space direction only, not a relay name, so nothing can mistake it
    for calibrated output."""
    if not camera_feed.has_camera(cam_id):
        raise HTTPException(404, f"Unknown camera '{cam_id}'.")
    jpeg = camera_feed.get_snapshot(cam_id)
    if jpeg is None:
        raise HTTPException(503, f"Camera '{cam_id}' has no frame yet (still connecting?).")
    img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(500, "Latest camera frame could not be decoded.")
    try:
        detections = model_service.detect_all(img, threshold=threshold)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))

    tracker = _get_tracker(cam_id, (start_x, start_y))
    tracker.max_layers = max_layers if max_layers > 0 else None
    numbered = tracker.update(detections)
    target = next((n for n in numbered if n.number == target_number), None)
    if target is None:
        return {"status": "no_target", "target_number": target_number}

    target_px = (target.detection.center.x, target.detection.center.y)
    decision = horizontal_servo.compute_servo_decision(detections, target_px, cam_id=cam_id)
    return {
        "target_number": target_number,
        "status": decision.status,
        "hook_px": decision.hook_px,
        "target_px": decision.target_px,
        "dx": decision.dx, "dy": decision.dy,
        "image_direction": decision.image_direction,
        "suggested_relay": decision.suggested_relay,
        "hook_confidence": decision.hook_confidence,
        "distance_m": decision.distance_m,
    }


@app.post("/api/cameras/{cam_id}/scale_reference")
def set_scale_reference(cam_id: str, distance_m: float, threshold: float = DEFAULT_THRESHOLD, target_number: int = 1):
    """One-time (per camera) scale capture (2026-09-12): the operator says
    'right now, hook to the targeted bale is <distance_m> metres', read off
    the LIVE frame at the moment this is called. Grabs the current hook and
    target-bale pixel positions from a fresh frame and derives
    pixels-per-metre from them - see scale_calibration.py for what this is
    and its limits (one scale factor for the whole traced zone, not true
    6-point calibration). MUST be called at the instant the stated distance
    is actually true in the live frame - it trusts the caller on that."""
    if not camera_feed.has_camera(cam_id):
        raise HTTPException(404, f"Unknown camera '{cam_id}'.")
    jpeg = camera_feed.get_snapshot(cam_id)
    if jpeg is None:
        raise HTTPException(503, f"Camera '{cam_id}' has no frame yet (still connecting?).")
    img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(500, "Latest camera frame could not be decoded.")
    try:
        detections = model_service.detect_all(img, threshold=threshold)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))

    hook = live_controller.best_hook_detection(detections)
    if hook is None:
        raise HTTPException(422, "No confidently-detected hook in the current frame - can't set the reference right now.")

    tracker = _get_tracker(cam_id, (0.0, 0.0))
    numbered = tracker.update(detections)
    target = next((n for n in numbered if n.number == target_number), None)
    if target is None:
        raise HTTPException(422, f"No numbered target #{target_number} in the current frame - can't set the reference right now.")

    hook_px = (hook.center.x, hook.center.y)
    target_px = (target.detection.center.x, target.detection.center.y)
    try:
        ref = scale_calibration.set_reference(cam_id, hook_px, target_px, distance_m)
    except ValueError as e:
        raise HTTPException(422, str(e))

    return {
        "cam_id": ref.cam_id,
        "pixels_per_metre": ref.pixels_per_metre,
        "ref_distance_m": ref.ref_distance_m,
        "ref_pixel_distance": ref.ref_pixel_distance,
        "hook_px": ref.hook_px,
        "target_px": ref.target_px,
        "set_at": ref.set_at,
    }


@app.get("/api/cameras/{cam_id}/scale_reference")
def get_scale_reference(cam_id: str):
    ref = scale_calibration.get_reference(cam_id)
    if ref is None:
        return {"configured": False}
    return {
        "configured": True,
        "cam_id": ref.cam_id,
        "pixels_per_metre": ref.pixels_per_metre,
        "ref_distance_m": ref.ref_distance_m,
        "ref_pixel_distance": ref.ref_pixel_distance,
        "hook_px": ref.hook_px,
        "target_px": ref.target_px,
        "set_at": ref.set_at,
    }


@app.get("/api/cameras/{cam_id}/boundary_status")
def camera_boundary_status(cam_id: str, threshold: float = DEFAULT_THRESHOLD):
    """One-shot read of the vision safety boundary (see boundary_guard.py) -
    is the hook inside the keep-in region, near its edge, or outside it right
    now? This is a snapshot; the actual enforcement is the always-running
    background watchdog (GET /api/boundary_guard/status for its live state -
    that's the one actually calling crane_remote.stop_all() on a violation)."""
    if not camera_feed.has_camera(cam_id):
        raise HTTPException(404, f"Unknown camera '{cam_id}'.")
    jpeg = camera_feed.get_snapshot(cam_id)
    if jpeg is None:
        raise HTTPException(503, f"Camera '{cam_id}' has no frame yet (still connecting?).")
    img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(500, "Latest camera frame could not be decoded.")
    try:
        detections = model_service.detect_all(img, threshold=threshold)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    status = boundary_guard.check_boundary(detections, cam_id)
    return {
        "status": status.status,
        "hook_px": status.hook_px,
        "margin_px": status.margin_px,
        "hook_confidence": status.hook_confidence,
        "boundary_configured": status.boundary_configured,
        "boundary_polygon": boundary_guard.BOUNDARY_POLYGON_PX.get(cam_id),
    }


@app.get("/api/boundary_guard/status")
def boundary_guard_status():
    """Live state of the ALWAYS-RUNNING background watchdog - this is the
    thing actually enforcing the boundary (calls crane_remote.stop_all() the
    instant it sees a confident violation), independent of any browser tab
    being open. Includes a short rolling log of recent events."""
    return boundary_guard.get_status()


@app.post("/api/boundary_guard/enable")
def boundary_guard_enable(enabled: bool = True):
    """Pause/resume the background watchdog. Pausing it means boundary
    violations will NOT trigger an automatic stop - only use this
    deliberately (e.g. while manually jogging near the edge on purpose)."""
    boundary_guard.set_enabled(enabled)
    return {"enabled": enabled}


def _boundary_mjpeg_stream(cam_id: str, threshold: float):
    """Same shape as the other stream_* generators - burns the boundary
    polygon, the hook marker, and the current status text into each frame so
    it's watchable in VLC or a browser, same as stream_annotated/
    stream_pick_order."""
    import time as _time

    poly = boundary_guard.BOUNDARY_POLYGON_PX.get(cam_id)
    last_frame_ts = -1.0
    idle_deadline = _time.monotonic() + 30.0
    while camera_feed.has_camera(cam_id):
        jpeg = camera_feed.get_snapshot(cam_id)
        fresh_ts = camera_feed.get_frame_ts(cam_id)
        if jpeg is None or fresh_ts == last_frame_ts:
            if _time.monotonic() > idle_deadline:
                break
            _time.sleep(0.05)
            continue
        last_frame_ts = fresh_ts
        idle_deadline = _time.monotonic() + 30.0

        img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        try:
            detections = model_service.detect_all(img, threshold=threshold)
            status = boundary_guard.check_boundary(detections, cam_id)
        except FileNotFoundError:
            status = None

        if poly:
            color = (0, 200, 0) if (status is None or status.status in ("inside", "no_hook")) else \
                    (0, 165, 255) if status.status == "near_edge" else (0, 0, 255)
            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, color, 3)
        if status and status.hook_px:
            hx, hy = int(status.hook_px[0]), int(status.hook_px[1])
            dot_color = (0, 0, 255) if status.status == "outside" else (0, 165, 255) if status.status == "near_edge" else (0, 200, 0)
            cv2.circle(img, (hx, hy), 10, dot_color, -1)
            label = f"{status.status.upper()} (conf {status.hook_confidence:.2f})"
            cv2.putText(img, label, (hx + 14, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dot_color, 2, cv2.LINE_AA)
        elif status:
            cv2.putText(img, "hook not confidently visible", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2, cv2.LINE_AA)

        enc_ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not enc_ok:
            continue
        out = buf.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(out)).encode() + b"\r\n\r\n"
            + out + b"\r\n"
        )


@app.get("/api/cameras/{cam_id}/stream_boundary")
def camera_stream_boundary(cam_id: str, threshold: float = DEFAULT_THRESHOLD):
    """MJPEG stream with the safety boundary + hook status burned in - green
    box/dot = inside, orange = near edge, red = outside (this is what
    triggers the watchdog's automatic stop). Point VLC or a browser at this
    to watch the safety net directly."""
    if not camera_feed.has_camera(cam_id):
        raise HTTPException(404, f"Unknown camera '{cam_id}'.")
    return StreamingResponse(
        _boundary_mjpeg_stream(cam_id, threshold),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "Connection": "close"},
    )


@app.post("/api/cameras/{cam_id}/pick_order/reset")
def camera_pick_order_reset(cam_id: str):
    """Forget this camera's current numbering (e.g. a new truck backed in) -
    the next /pick_order or /stream_pick_order call renumbers from scratch."""
    _pick_trackers.pop(cam_id, None)
    return {"ok": True}


def _pick_order_mjpeg_stream(cam_id: str, threshold: float, max_layers: int, start_xy: tuple[float, float]):
    """Same shape as _annotated_mjpeg_stream, but draws pick NUMBERS (not
    raw class/confidence labels) so the sequence is watchable live in VLC or
    the browser - big numbered circles in pick order, top-two-layers only by
    default. Bales outside max_layers are drawn faint/unlabeled so you can
    still see the whole load."""
    import time as _time

    last_frame_ts = -1.0
    idle_deadline = _time.monotonic() + 30.0
    tracker = _get_tracker(cam_id, start_xy)
    tracker.max_layers = max_layers if max_layers > 0 else None
    while camera_feed.has_camera(cam_id):
        jpeg = camera_feed.get_snapshot(cam_id)
        fresh_ts = camera_feed.get_frame_ts(cam_id)
        if jpeg is None or fresh_ts == last_frame_ts:
            if _time.monotonic() > idle_deadline:
                break
            _time.sleep(0.05)
            continue
        last_frame_ts = fresh_ts
        idle_deadline = _time.monotonic() + 30.0

        img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        try:
            detections = model_service.detect_all(img, threshold=threshold)
            numbered = tracker.update(detections)
            numbered_ids = {id(n.detection) for n in numbered}
            for d in detections:
                if d.class_id != 0 or id(d) in numbered_ids:
                    continue  # not a bale, or already drawn numbered below
                x1, y1, x2, y2 = int(d.bbox.x1), int(d.bbox.y1), int(d.bbox.x2), int(d.bbox.y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (90, 90, 90), 1)
            # Draw #1 (the target) LAST so it always paints on top of any
            # overlapping bale, however the stack is laid out - drawing in
            # plain ascending number order let a merely-higher-numbered
            # neighbor overdraw the target's highlight and hide its label
            # completely, which is what "target isn't showing" turned out
            # to be: #1 was still correct, just invisible under a later box.
            target_first = sorted(numbered, key=lambda n: n.number == 1)
            for n in target_first:
                x1, y1, x2, y2 = int(n.detection.bbox.x1), int(n.detection.bbox.y1), int(n.detection.bbox.x2), int(n.detection.bbox.y2)
                cx, cy = int(n.detection.center.x), int(n.detection.center.y)
                color = (0, 210, 255) if n.number == 1 else (80, 200, 80)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.circle(img, (cx, cy), 16, color, -1)
                cv2.putText(img, str(n.number), (cx - (9 if n.number < 10 else 15), cy + 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        except FileNotFoundError:
            pass

        enc_ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not enc_ok:
            continue
        out = buf.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(out)).encode() + b"\r\n\r\n"
            + out + b"\r\n"
        )


@app.get("/api/cameras/{cam_id}/stream_pick_order")
def camera_stream_pick_order(
    cam_id: str, threshold: float = DEFAULT_THRESHOLD,
    max_layers: int = 2, start_x: float = 0.0, start_y: float = 0.0,
):
    """MJPEG stream with the pick ORDER burned in as numbered circles (#1
    highlighted) instead of raw detection boxes - point VLC or a browser at
    this to watch the numbering live. See camera_pick_order for the same
    thing as structured JSON, and pick_order.py for what this can and can't
    promise without calibration."""
    if not camera_feed.has_camera(cam_id):
        raise HTTPException(404, f"Unknown camera '{cam_id}'.")
    return StreamingResponse(
        _pick_order_mjpeg_stream(cam_id, threshold, max_layers, (start_x, start_y)),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "Connection": "close"},
    )


@app.get("/api/crane_remote/status")
def crane_remote_status():
    """Read-only view of the PHYSICAL crane remote's current relay state
    (the SF-8DR at 192.168.4.1) plus whether the bridge is enabled. Returns
    {"enabled":..., "online":..., "remote":... or null} - never raises, so a
    down remote still returns a useful envelope. Observational aid for
    verifying the real LEDs match the sim. See crane_remote.py."""
    if not crane_remote.CRANE_REMOTE_ENABLED or not crane_remote.CRANE_REMOTE_URL:
        return {"enabled": False, "online": False, "remote": None}
    # One blocking probe, not two: status() does a urlopen() with a multi-second
    # timeout, and the frontend polls this at 1 Hz. Calling it twice per request
    # doubled the time each pooled worker thread was tied up waiting on an
    # unreachable remote - enough to exhaust the thread pool and stall the
    # whole backend.
    remote = crane_remote.status()
    return {"enabled": True, "online": remote is not None, "remote": remote}


@app.post("/api/crane_remote/stop")
def crane_remote_stop():
    """Release every relay on the physical crane remote immediately (STOP),
    and reset the bridge's change-tracking so the next tick on the next run
    re-fires cleanly. Useful to park the crane / clear latched outputs after
    a run. See crane_remote.stop_all."""
    crane_remote.stop_all()
    return {"ok": True}


@app.post("/api/crane_remote/relays")
async def crane_remote_relays(request: Request):
    """Bridge endpoint for the 2D simulator playback: the frontend computes
    the active relay set from the sim's relay_timeline client-side
    (controllerPanel -> activeRelays) - this endpoint lets it push that set
    to the REAL physical crane remote, exactly mirroring what the virtual
    LED panel shows. Body: {"relays": ["north_1s", ...]} or JSON array.
    Mirrors state (energize newly-on, release newly-off) - see
    crane_remote.sync_relays."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body.")
    if isinstance(body, list):
        relays = body
    elif isinstance(body, dict) and isinstance(body.get("relays"), list):
        relays = body["relays"]
    else:
        raise HTTPException(400, 'Expected {"relays": [...]} or a JSON array.')
    if not all(isinstance(r, str) for r in relays):
        raise HTTPException(400, "Relay names must be strings.")
    crane_remote.sync_relays(list(relays))
    return {"ok": True, "active": relays}


@app.post("/api/crane_remote/manual")
async def crane_remote_manual(request: Request):
    """Manual single-channel control for the in-app crane-remote button panel
    (CraneRemotePanel.tsx) - the same UP/DOWN/EAST/WEST/SOUTH/NORTH/START/
    LIGHT buttons as the SF-8DR's own page at 192.168.4.1, but inside the ITC
    UI. Independent of the sim control loop. Body: {"ch": 0-7, "on": bool}.
    Forces the remote onto GPIO and releases the interlocked opposite axis on
    turn-on. Returns the remote's fresh status so the panel can repaint."""
    try:
        body = await request.json()
        ch = int(body["ch"])
        on = bool(body["on"])
    except Exception:
        raise HTTPException(400, 'Expected {"ch": 0-7, "on": true|false}.')
    if not 0 <= ch <= 7:
        raise HTTPException(400, "ch must be 0-7.")
    ok = crane_remote.manual_channel(ch, on)
    return {"ok": ok, "ch": ch, "on": on, "remote": crane_remote.status()}


@app.post("/api/fixed_cycle/start")
def fixed_cycle_start():
    """Kick off the fixed, single-target POC pick-and-drop cycle (see
    fixed_cycle.py). Vision-gated at the pick step (top-2-layers only, see
    check_bale_present) but otherwise a hardcoded timed sequence - NOT the
    sim's vision-driven control loop, which needs real position feedback
    this hardware doesn't have. Refuses to start if the crane remote is
    unreachable or a cycle is already running."""
    return fixed_cycle.start_cycle()


@app.get("/api/fixed_cycle/status")
def fixed_cycle_status():
    """Poll this while a cycle runs - step index/label, error (e.g. vision
    gate failed), running flag."""
    return fixed_cycle.get_status()


@app.post("/api/fixed_cycle/stop")
def fixed_cycle_stop():
    """Immediately halt the running cycle and release all relays."""
    return fixed_cycle.stop_cycle()


@app.post("/api/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXT)}")

    image_id = uuid.uuid4().hex
    dest = Path(UPLOAD_DIR) / f"{image_id}{ext}"
    data = await file.read()
    dest.write_bytes(data)

    img = cv2.imread(str(dest))
    if img is None:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Could not decode image file.")
    h, w = img.shape[:2]

    return UploadResponse(image_id=f"{image_id}{ext}", url=f"/uploads/{image_id}{ext}", width=w, height=h)


def _load_image(image_id: str) -> np.ndarray:
    # image_id is a server-generated uuid+ext from /api/upload; still guard
    # against path traversal since it arrives back from the client.
    safe_name = Path(image_id).name
    path = Path(UPLOAD_DIR) / safe_name
    if not path.is_file() or path.parent.resolve() != Path(UPLOAD_DIR).resolve():
        raise HTTPException(404, "Unknown image_id. Upload the image first via /api/upload.")
    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(400, "Stored image could not be decoded.")
    return img


@app.post("/api/simulate", response_model=SimulationResult)
def simulate(req: SimulateRequest):
    img = _load_image(req.image_id)
    h, w = img.shape[:2]

    try:
        detections = model_service.detect_bales(img, threshold=req.threshold)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))

    result = plan_simulation(detections, req, image_w=w, image_h=h)
    return result


@app.post("/api/detect", response_model=DetectResponse)
async def detect_frame(file: UploadFile = File(...), threshold: float = Form(DEFAULT_THRESHOLD)):
    """Stateless single-frame multi-class detection (bale/crane_hook/
    crane_spike), for the live 3D control mode - called repeatedly against
    rendered camera frames, so this deliberately never touches disk (unlike
    /api/upload) to avoid storage bloat under a live polling loop."""
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image data.")
    h, w = img.shape[:2]

    try:
        detections = model_service.detect_all(img, threshold=threshold)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))

    return DetectResponse(width=w, height=h, detections=detections)


@app.post("/api/control_tick", response_model=ControlTickResponse)
async def control_tick(
    file: UploadFile = File(...),
    crane_x: float = Form(...),
    crane_y: float = Form(...),
    crane_z: float = Form(...),
    phase: str = Form(...),
    target_bale_id: Optional[int] = Form(None),
    elapsed_in_phase_s: float = Form(0.0),
    picked_bale_ids: str = Form("[]"),
    threshold: float = Form(DEFAULT_THRESHOLD),
):
    """The live 3D mode's actual control loop (see live_controller.py) -
    this is the ONLY place relay decisions get made; the frontend just
    reports its current pose and reacts to what comes back. Called
    repeatedly against the top camera's rendered frames, same
    never-touches-disk stance as /api/detect (this fires just as often)."""
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image data.")
    h, w = img.shape[:2]

    try:
        detections = model_service.detect_all(img, threshold=threshold)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))

    try:
        picked_ids = set(int(i) for i in json.loads(picked_bale_ids))
    except (json.JSONDecodeError, TypeError, ValueError):
        raise HTTPException(400, "picked_bale_ids must be a JSON array of integers.")

    live_bale_detections = []
    for d in detections:
        if d.class_id != 0:  # only bale-class detections drive control - see model_service.CLASS_BALE
            continue
        hit = live_controller.unproject_pixel_to_ground(d.center.x, d.center.y, w, h)
        if hit is not None:
            live_bale_detections.append((hit[0], hit[1], d.confidence))

    # Vision-measured crane position: unproject this tick's best crane_hook
    # detection (if confidently seen - see HOOK_POSITION_CONFIDENCE) through
    # the SELF-REPORTED crane_y as the ground plane, since a single top-down
    # camera can't recover height on its own. This is PURELY OBSERVATIONAL
    # (position_source, surfaced in the UI) - it must never feed compute_tick.
    # A per-tick vision fix has real pixel/detection-box noise; the
    # self-reported (x, z) below is the frontend's own dead-reckoned pose,
    # driven by controller.ts's driveToward() which moves at a constant rate
    # and clamps exactly at arrival - i.e. it's already exact/noise-free in
    # this sim, unlike a real encoder. Swapping it out for a noisy measurement
    # made the control law compare two independently-jittering numbers each
    # tick, so `remaining = target - cx` could flip sign every tick even
    # while the crane was genuinely converging - relays would then reverse
    # every scan (north/south.../hunting in place, never arriving - the
    # "crane isn't moving" bug this fixed). Keep the vision measurement for
    # display only until there's a real sensor to justify trusting it over
    # dead-reckoning.
    best_hook = live_controller.best_hook_detection(detections)
    vision_hit = (
        live_controller.unproject_pixel_to_ground(best_hook.center.x, best_hook.center.y, w, h, plane_y=crane_y)
        if best_hook is not None
        else None
    )
    position_source = "vision" if vision_hit is not None else "self_reported"

    state = live_controller.TickState(
        phase=phase,
        crane_x=crane_x,
        crane_y=crane_y,
        crane_z=crane_z,
        target_bale_id=target_bale_id,
        elapsed_in_phase_s=elapsed_in_phase_s,
        picked_ids=picked_ids,
    )
    decision = live_controller.compute_tick(state, live_bale_detections)

    # Reflect the control law's relay decision to the real crane remote (the
    # physical SF-8DR at 192.168.4.1) so its LEDs light in lock-step with the
    # sim. Best-effort only: if the reserve is disabled or unreachable this is
    # a silent no-op and never affects the loop above/below. See
    # crane_remote.py and config.py.
    crane_remote.sync_relays(decision.active_relays)

    return ControlTickResponse(
        width=w,
        height=h,
        detections=detections,
        phase=decision.phase,
        active_relays=decision.active_relays,
        target_bale_id=decision.target_bale_id,
        target_x=decision.target_x,
        target_y=decision.target_y,
        target_z=decision.target_z,
        mark_picked_bale_id=decision.mark_picked_bale_id,
        log=decision.log,
        position_source=position_source,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
