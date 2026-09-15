import type {
  BoundaryGuardStatus, CameraBoundaryStatus, ControlTickResponse, ConveyorStatus,
  CrossViewStatus, DetectResponse, PickOrderResponse, ScaleReferenceStatus, ServoDecision,
  SimulateRequest, SimulationResult, UploadResponse,
} from "./types";

export const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export async function uploadImage(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: form });
  if (!res.ok) {
    throw new Error(`Upload failed: ${res.status} ${await res.text()}`);
  }
  return res.json();
}

export async function runSimulation(req: SimulateRequest): Promise<SimulationResult> {
  const res = await fetch(`${API_BASE}/api/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    throw new Error(`Simulate failed: ${res.status} ${await res.text()}`);
  }
  return res.json();
}

export function imageUrl(path: string): string {
  return path.startsWith("http") ? path : `${API_BASE}${path}`;
}

// --- Physical plant cameras (Hikvision cam101 @ 192.168.1.101, cam102 @
// .102). The backend pulls each camera's RTSP stream and re-serves it as
// MJPEG / JPEG - see backend/app/camera_feed.py. These helpers just build the
// URLs; an <img src> does the actual streaming (see PlantCameraFeed.tsx). ---

export interface CameraStatus {
  id: string;
  name: string;
  host: string;
  connected: boolean;
  last_frame_age_s: number | null;
}

export function cameraStreamUrl(camId: string): string {
  return `${API_BASE}/api/cameras/${camId}/stream`;
}

// MJPEG stream with pick NUMBERS burned in server-side (backend/app/main.py's
// _pick_order_mjpeg_stream) - see RealPickDryRun.tsx.
export function cameraStreamPickOrderUrl(camId: string, threshold: number, maxLayers = 2): string {
  return `${API_BASE}/api/cameras/${camId}/stream_pick_order?threshold=${threshold}&max_layers=${maxLayers}`;
}

// `bust` forces the browser to re-fetch (MJPEG <img>s are cached by src);
// pass a changing value to reconnect after a drop.
export function cameraSnapshotUrl(camId: string, bust?: number | string): string {
  const q = bust != null ? `?t=${bust}` : "";
  return `${API_BASE}/api/cameras/${camId}/snapshot${q}`;
}

export async function getCameras(): Promise<CameraStatus[]> {
  try {
    const r = await fetch(`${API_BASE}/api/cameras`);
    if (!r.ok) return [];
    return ((await r.json()).cameras ?? []) as CameraStatus[];
  } catch (e) {
    console.error("[cameras]", e);
    return [];
  }
}

// Run the RF-DETR detector on a plant camera's latest frame and get back the
// raw detections (bale / crane_hook / crane_spike) in that frame's pixel
// space - see backend GET /api/cameras/<id>/detect. Display-only: the plant
// panels poll this and draw the boxes; it never feeds the control loop.
export async function cameraDetect(camId: string, threshold: number): Promise<DetectResponse> {
  const res = await fetch(`${API_BASE}/api/cameras/${camId}/detect?threshold=${threshold}`);
  if (!res.ok) {
    throw new Error(`Camera detect failed: ${res.status} ${await res.text()}`);
  }
  return res.json();
}

// Numbered pick order for a live plant camera - see backend/app/pick_order.py.
// PIXEL-SPACE approximation (no calibration yet), display/dry-run only -
// never wired to crane_remote. See RealPickDryRun.tsx.
export async function getPickOrder(camId: string, threshold: number, maxLayers = 2): Promise<PickOrderResponse> {
  const res = await fetch(`${API_BASE}/api/cameras/${camId}/pick_order?threshold=${threshold}&max_layers=${maxLayers}`);
  if (!res.ok) {
    throw new Error(`pick_order failed: ${res.status} ${await res.text()}`);
  }
  return res.json();
}

export async function resetPickOrder(camId: string): Promise<void> {
  await fetch(`${API_BASE}/api/cameras/${camId}/pick_order/reset`, { method: "POST" });
}

// Requirement 3 (2026-09-11): is the conveyor/drop area already full? See
// backend/app/conveyor_gate.py.
export async function getConveyorStatus(camId: string, threshold: number): Promise<ConveyorStatus> {
  const res = await fetch(`${API_BASE}/api/cameras/${camId}/conveyor_status?threshold=${threshold}`);
  if (!res.ok) {
    throw new Error(`conveyor_status failed: ${res.status} ${await res.text()}`);
  }
  return res.json();
}

// Vision safety boundary (senior's suggestion, 2026-09-12) - see
// backend/app/boundary_guard.py. The background watchdog enforces this on
// its own; these just let the UI show what it's seeing.
export async function getCameraBoundaryStatus(camId: string, threshold: number): Promise<CameraBoundaryStatus> {
  const res = await fetch(`${API_BASE}/api/cameras/${camId}/boundary_status?threshold=${threshold}`);
  if (!res.ok) {
    throw new Error(`boundary_status failed: ${res.status} ${await res.text()}`);
  }
  return res.json();
}

export async function getBoundaryGuardStatus(): Promise<BoundaryGuardStatus> {
  const res = await fetch(`${API_BASE}/api/boundary_guard/status`);
  if (!res.ok) {
    throw new Error(`boundary_guard status failed: ${res.status} ${await res.text()}`);
  }
  return res.json();
}

export async function setBoundaryGuardEnabled(enabled: boolean): Promise<void> {
  await fetch(`${API_BASE}/api/boundary_guard/enable?enabled=${enabled}`, { method: "POST" });
}

// MJPEG stream with the safety boundary + hook status burned in server-side
// (backend/app/main.py's _boundary_mjpeg_stream).
export function cameraStreamBoundaryUrl(camId: string, threshold: number): string {
  return `${API_BASE}/api/cameras/${camId}/stream_boundary?threshold=${threshold}`;
}

// Closed-loop E/W/N/S nudge decision - see backend/app/horizontal_servo.py.
// Display-only, never fires a relay itself.
export async function getServoDecision(
  camId: string, threshold: number, targetNumber = 1,
): Promise<ServoDecision> {
  const res = await fetch(
    `${API_BASE}/api/cameras/${camId}/servo_decision?threshold=${threshold}&target_number=${targetNumber}`,
  );
  if (!res.ok) {
    throw new Error(`servo_decision failed: ${res.status} ${await res.text()}`);
  }
  return res.json();
}

// See backend/app/scale_calibration.py - "right now, hook to the targeted
// bale is <distanceM> metres" read off the live frame. Captures the current
// hook + target pixel positions and derives pixels-per-metre from them.
export async function setScaleReference(
  camId: string, distanceM: number, threshold: number, targetNumber = 1,
): Promise<ScaleReferenceStatus> {
  const res = await fetch(
    `${API_BASE}/api/cameras/${camId}/scale_reference?distance_m=${distanceM}&threshold=${threshold}&target_number=${targetNumber}`,
    { method: "POST" },
  );
  if (!res.ok) {
    throw new Error(`scale_reference failed: ${res.status} ${await res.text()}`);
  }
  return res.json();
}

export async function getScaleReference(camId: string): Promise<ScaleReferenceStatus> {
  const res = await fetch(`${API_BASE}/api/cameras/${camId}/scale_reference`);
  if (!res.ok) {
    throw new Error(`scale_reference status failed: ${res.status} ${await res.text()}`);
  }
  return res.json();
}

// See backend/app/main.py's pick_order_cross_view_status - a count-level
// sanity check between the two cameras' independent top-N-layer reads.
export async function getCrossViewStatus(
  camA: string, camB: string, threshold: number, maxLayers = 2,
): Promise<CrossViewStatus> {
  const res = await fetch(
    `${API_BASE}/api/pick_order/cross_view_status?cam_a=${camA}&cam_b=${camB}&threshold=${threshold}&max_layers=${maxLayers}`,
  );
  if (!res.ok) {
    throw new Error(`cross_view_status failed: ${res.status} ${await res.text()}`);
  }
  return res.json();
}

// Push the currently-active relay set (from 2D playback's relay_timeline,
// computed client-side in ControllerPanel) to the REAL physical crane remote
// so its LEDs mirror the sim exactly. Body is a JSON object {"relays": [...]}
// - see backend/app/crane_remote.py's sync_relays (state-mirrored: turns on
// newly-active, releases newly-inactive). Best-effort: a failure here must
// never break playback.
export async function syncPhysicalRelays(relays: string[]): Promise<void> {
  try {
    await fetch(`${API_BASE}/api/crane_remote/relays`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ relays }),
    });
  } catch (e) {
    // Best-effort bridge - ignore network errors so playback is unaffected.
    console.error("[physical-relays]", e);
  }
}

export async function stopPhysicalRelays(): Promise<void> {
  try {
    await fetch(`${API_BASE}/api/crane_remote/stop`, { method: "POST" });
  } catch (e) {
    console.error("[physical-relays/stop]", e);
  }
}

export interface CraneRemoteStatus {
  enabled: boolean;
  online: boolean;
  remote: { ok: boolean; backend: string; relays: boolean[] } | null;
}

// Live state of the physical SF-8DR remote (backend proxies its /api/status).
export async function getCraneRemoteStatus(): Promise<CraneRemoteStatus | null> {
  try {
    const r = await fetch(`${API_BASE}/api/crane_remote/status`);
    if (!r.ok) return null;
    return (await r.json()) as CraneRemoteStatus;
  } catch {
    return null;
  }
}

// Manual single-channel control for the in-app crane-remote button panel
// (ch 0-7: UP DOWN EAST WEST SOUTH NORTH START LIGHT). Independent of the
// sim loop; backend forces GPIO + releases the interlocked opposite axis.
export async function manualCraneChannel(
  ch: number,
  on: boolean,
): Promise<CraneRemoteStatus["remote"]> {
  try {
    const r = await fetch(`${API_BASE}/api/crane_remote/manual`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ch, on }),
    });
    if (!r.ok) return null;
    return (await r.json()).remote ?? null;
  } catch (e) {
    console.error("[crane-manual]", e);
    return null;
  }
}

// Stateless single-frame multi-class detection (bale/crane_hook/crane_spike),
// used by the live 3D control mode's camera-feed panels. Never persisted to
// disk server-side (unlike uploadImage) since this is called repeatedly.
export async function detectFrame(blob: Blob, threshold: number): Promise<DetectResponse> {
  const form = new FormData();
  form.append("file", blob, "frame.jpg");
  form.append("threshold", String(threshold));
  const res = await fetch(`${API_BASE}/api/detect`, { method: "POST", body: form });
  if (!res.ok) {
    throw new Error(`Detect failed: ${res.status} ${await res.text()}`);
  }
  return res.json();
}

export interface ControlTickState {
  craneX: number;
  craneY: number;
  craneZ: number;
  phase: string;
  targetBaleId: number | null;
  elapsedInPhaseS: number;
  pickedBaleIds: number[];
  threshold: number;
}

// The live 3D mode's actual control loop - see backend/app/live_controller.py.
// Posts the top camera's captured frame PLUS the crane's self-reported
// current pose; the backend re-detects, re-confirms the target against live
// vision, and returns the relay decision + setpoint for this tick (not just
// raw detections, unlike detectFrame above).
export async function controlTick(blob: Blob, state: ControlTickState): Promise<ControlTickResponse> {
  const form = new FormData();
  form.append("file", blob, "frame.jpg");
  form.append("crane_x", String(state.craneX));
  form.append("crane_y", String(state.craneY));
  form.append("crane_z", String(state.craneZ));
  form.append("phase", state.phase);
  if (state.targetBaleId != null) form.append("target_bale_id", String(state.targetBaleId));
  form.append("elapsed_in_phase_s", String(state.elapsedInPhaseS));
  form.append("picked_bale_ids", JSON.stringify(state.pickedBaleIds));
  form.append("threshold", String(state.threshold));
  const res = await fetch(`${API_BASE}/api/control_tick`, { method: "POST", body: form });
  if (!res.ok) {
    throw new Error(`Control tick failed: ${res.status} ${await res.text()}`);
  }
  return res.json();
}
