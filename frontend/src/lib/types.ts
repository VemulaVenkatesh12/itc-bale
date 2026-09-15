// Mirrors backend/app/schemas.py exactly (field names kept snake_case to
// avoid a translation layer between wire format and UI state).

export interface Point {
  x: number;
  y: number;
}

export interface Rect {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface UploadResponse {
  image_id: string;
  url: string;
  width: number;
  height: number;
}

// 0=bale, 1=crane_hook, 2=crane_spike (see backend/app/model_service.py's
// CLASS_* constants). Defaults to 0 since older callers (the 2D /api/simulate
// flow, which only ever returns bales) don't send it.
export type DetectionClass = 0 | 1 | 2;

export interface Detection {
  id: number;
  bbox: Rect;
  confidence: number;
  center: Point;
  class_id: DetectionClass;
}

export interface DetectResponse {
  width: number;
  height: number;
  detections: Detection[];
}

// See backend/app/pick_order.py - PIXEL-SPACE pick order, not real 3D. One
// numbered entry per bale worth picking (top `max_layers` only), #1 first.
export interface PickOrderEntry {
  number: number;
  layer: number;
  bbox: Rect;
  confidence: number;
}

export interface PickOrderResponse {
  width: number;
  height: number;
  picks: PickOrderEntry[];
}

// See backend/app/conveyor_gate.py - requirement 3: hold pick signals while
// the conveyor/drop area is still full of un-cleared bales.
export interface ConveyorStatus {
  roi_configured: boolean;
  count: number;
  full_threshold: number;
  full: boolean;
  ok_to_pick: boolean | null; // null = no ROI traced for this camera yet
}

// See backend/app/boundary_guard.py - the always-running vision safety net
// (senior's suggestion, 2026-09-12): stops the crane the instant the hook
// is seen outside the keep-in region. "no_hook" is not an error - it means
// nothing confident enough to act on either way.
export type BoundaryState = "no_hook" | "inside" | "near_edge" | "outside";

export interface CameraBoundaryStatus {
  status: BoundaryState;
  hook_px: [number, number] | null;
  margin_px: number | null;
  hook_confidence: number | null;
  boundary_configured: boolean;
  boundary_polygon: [number, number][] | null;
}

export interface BoundaryGuardStatus {
  enabled: boolean;
  violation_count: number;
  cameras: Record<string, {
    status: BoundaryState;
    hook_px: [number, number] | null;
    margin_px: number | null;
    hook_confidence: number | null;
    boundary_configured: boolean;
  }>;
  log: string[];
}

// See backend/app/horizontal_servo.py - E/W/N/S nudge direction from
// comparing hook pixel position to the numbered target's pixel position.
// distance_m is only present once a scale_reference has been captured for
// this camera (see ScaleReference below) - otherwise it's null and only
// the image-space direction is meaningful.
export interface ServoDecision {
  target_number: number;
  status: "no_hook" | "no_target" | "aligned" | "nudge";
  hook_px: [number, number] | null;
  target_px: [number, number] | null;
  dx: number | null;
  dy: number | null;
  image_direction: string | null;
  suggested_relay: string | null;
  hook_confidence: number | null;
  distance_m: number | null;
}

// See backend/app/main.py's pick_order_cross_view_status (2026-09-12) - a
// count-level sanity check between the two cameras' independent top-N-layer
// reads. NOT per-bale identity matching across views (needs calibration,
// not done) - just whether both cameras currently agree on roughly how many
// bales are in play right now.
export interface CrossViewStatus {
  cam_a: string;
  cam_b: string;
  count_a: number | null;
  count_b: number | null;
  agree: boolean | null;
  row_count_a: number | null;
  row_count_b: number | null;
  rows_agree: boolean | null;
  note: string;
}

// See backend/app/scale_calibration.py - a single hook-to-target real
// distance (metres), captured from a live frame, used to turn pixel
// offsets into rough real-world metres within the traced POC zone. NOT
// full 6-point calibration - one scale factor for one rough depth plane.
export interface ScaleReferenceStatus {
  configured: boolean;
  cam_id?: string;
  pixels_per_metre?: number;
  ref_distance_m?: number;
  ref_pixel_distance?: number;
  hook_px?: [number, number];
  target_px?: [number, number];
  set_at?: string;
}

// Response for POST /api/control_tick - the live 3D mode's actual
// closed-loop control decision, computed server-side (see
// backend/app/live_controller.py). target_x/y/z is the CURRENT leg's
// setpoint; the frontend's local drive loop (live/controller.ts) moves
// toward it at the speed implied by active_relays and stops itself at
// arrival - see live_controller.py's module docstring for why.
export interface ControlTickResponse {
  width: number;
  height: number;
  detections: Detection[];

  phase: string;
  active_relays: string[];
  target_bale_id: number | null;
  target_x: number;
  target_y: number;
  target_z: number;
  mark_picked_bale_id: number | null;
  log: string | null;
  // "vision" when this tick's (x, z) came from a confidently-detected
  // crane_hook in THIS frame (backend/app/live_controller.py's
  // best_hook_detection); "self_reported" when the hook wasn't confidently
  // seen and the frontend's own dead-reckoned craneX/Z was used as fallback
  // instead - expected often (~1/3-2/5 of ticks per this session's
  // measured hook visibility rate), not just a rare edge case. Y is always
  // self-reported regardless.
  position_source: "vision" | "self_reported";
}

export interface SpeedConfig {
  ew_low_px_s: number;
  ew_high_px_s: number;
  ns_low_px_s: number;
  ns_high_px_s: number;
  high_speed_min_distance_px: number;
  creep_distance_px: number;
  hoist_clear_s: number;
  hoist_lower_s: number;
  hoist_creep_s: number;
  pierce_dwell_s: number;
  push_off_s: number;
  settle_s: number;
  playback_speed: number;
}

export interface OrientationConfig {
  flip_ns: boolean;
  flip_ew: boolean;
}

export interface SimulateRequest {
  image_id: string;
  detection_area: Rect;
  drop_area: Rect;
  crane_home?: Point | null;
  threshold: number;
  speed: SpeedConfig;
  orientation: OrientationConfig;
  max_bales?: number | null;
}

export type AxisName = "ew" | "ns" | "hoist" | "push" | "system";

export interface RelayEvent {
  relay: string;
  axis: AxisName;
  phase: string;
  on_at_ms: number;
  off_at_ms: number;
}

export interface CraneKeyframe {
  t_ms: number;
  x: number;
  y: number;
  hoist: number;
  spiked: boolean;
  active_bale_id: number | null;
  phase: string;
}

export interface PickCycle {
  order: number;
  bale_id: number;
  bbox: Rect;
  confidence: number;
  pick_point: Point;
  drop_point: Point;
  t_start_ms: number;
  t_end_ms: number;
}

export interface SimulationResult {
  image_width: number;
  image_height: number;
  detections: Detection[];
  pick_cycles: PickCycle[];
  relay_timeline: RelayEvent[];
  keyframes: CraneKeyframe[];
  total_duration_ms: number;
  crane_home: Point;
  warnings: string[];
}

export const DEFAULT_SPEED: SpeedConfig = {
  ew_low_px_s: 60,
  ew_high_px_s: 180,
  ns_low_px_s: 45,
  ns_high_px_s: 130,
  high_speed_min_distance_px: 150,
  creep_distance_px: 40,
  hoist_clear_s: 1.5,
  hoist_lower_s: 1.5,
  hoist_creep_s: 0.4,
  pierce_dwell_s: 0.8,
  push_off_s: 0.8,
  settle_s: 0.25,
  playback_speed: 1.0,
};

export const DEFAULT_ORIENTATION: OrientationConfig = {
  flip_ns: false,
  flip_ew: false,
};

// Fixed row order for the controller-panel / timeline display, matching the
// SF-8DR terminal legend (14-22, 1-13).
export const RELAY_ORDER: { relay: string; label: string; group: string }[] = [
  { relay: "north_1s", label: "NORTH 1S", group: "N/S" },
  { relay: "south_1s", label: "SOUTH 1S", group: "N/S" },
  { relay: "ns_2s", label: "N/S 2S", group: "N/S" },
  { relay: "east_1s", label: "EAST 1S", group: "E/W" },
  { relay: "west_1s", label: "WEST 1S", group: "E/W" },
  { relay: "ew_2s", label: "E/W 2S", group: "E/W" },
  { relay: "up_1s", label: "UP 1S", group: "HOIST" },
  { relay: "up_2s", label: "UP 2S", group: "HOIST" },
  { relay: "down_1s", label: "DOWN 1S", group: "HOIST" },
  { relay: "down_2s", label: "DOWN 2S", group: "HOIST" },
  { relay: "push", label: "PUSH-OFF (R1)", group: "EJECT" },
];
