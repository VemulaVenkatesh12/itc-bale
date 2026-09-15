// Shared state for the live 3D control mode.
//
// Split deliberately in two:
//  - `craneRef` (below): a plain mutable object, NOT zustand. The crane's
//    position/thrust update every animation frame (60fps); routing that
//    through zustand would re-render every subscribing React component 60
//    times/sec for no benefit. CraneRig reads this directly inside
//    useFrame - the standard r3f imperative-animation pattern (mirrors how
//    the 2D mode's usePlayback/interpolateCrane drive SVG attrs directly
//    today, just adapted to an Object3D instead).
//  - `useLiveStore` (zustand): everything that changes a few times/sec at
//    most and genuinely needs to trigger a UI re-render - phase, active
//    relays (LED panel), the bale list (picked/unpicked), live detections
//    (camera overlay boxes), run state, tunables.

import { create } from "zustand";
import * as THREE from "three";
import type { BaleSpec } from "./factoryLayout";
import { buildBaleGrid, TRUCK, CONVEYOR, CRANE, CAMERA_TOP, CAMERA_BACK } from "./factoryLayout";
import { baleBodies, resetBaleBodies } from "./physicsBridge";
import { useMatteMode } from "./matteMode";
import type { Detection } from "../lib/types";

// --- imperative (non-store) crane pose, read every frame by CraneRig -----
export const craneRef = {
  position: new THREE.Vector3(0, 5.0, 0),
  thrust: 0, // unused since the spike became a fixed prong - kept harmlessly, nothing reads it
};

/** The backend's last-commanded setpoint (see api.ts's controlTick /
 *  backend/app/live_controller.py) - a plain mutable object, not zustand,
 *  same rationale as craneRef: controller.ts's local drive loop reads this
 *  every rendered frame (60fps), while CameraFeed.tsx's top-camera tick
 *  writes it only every ~800ms. */
export const commandedRef = {
  target: new THREE.Vector3(0, 5.0, 0),
};

// --- live-mode types -------------------------------------------------------

export interface LiveBale extends BaleSpec {
  picked: boolean;
  /** Mirrors physicsBridge's per-bale flags, copied in here so the UI can
   *  show counts without subscribing to the (non-reactive) physics map. */
  knocked: boolean;
  fallen: boolean;
}

export interface LiveSpeedConfig {
  // Single speed tier only - see backend/app/live_controller.py's
  // _axis_relays module comment: exactly one relay energized at a time,
  // always the "1s" tier, no "2s" speed-boost relay combined with it.
  ewMs: number;
  nsMs: number;
  creepDistanceM: number;
  hoistClearS: number;
  hoistLowerS: number;
  pierceDwellS: number;
  pushOffS: number;
  settleS: number;
}

export const DEFAULT_LIVE_SPEED: LiveSpeedConfig = {
  ewMs: 0.6,
  nsMs: 0.45,
  creepDistanceM: 0.4,
  hoistClearS: 1.5,
  hoistLowerS: 1.5,
  pierceDwellS: 0.8,
  pushOffS: 0.8,
  settleS: 0.25,
};

export type LivePhase =
  | "idle"
  | "select_target"
  | "cross"
  | "descend"
  | "hoist_down_to_bale"
  | "pierce"
  | "pierce_settle"
  | "hoist_up_clear"
  | "drop_cross"
  | "drop_descend"
  | "hoist_down_to_drop"
  | "push_off"
  | "hoist_up_after_drop"
  | "return_home"
  | "done";

interface LiveStoreState {
  running: boolean;
  phase: LivePhase;
  activeRelays: Set<string>;
  bales: LiveBale[];
  backDetections: Detection[];
  threshold: number;
  detectIntervalMs: number;
  speed: LiveSpeedConfig;
  log: string[];
  targetBaleId: number | null;
  /** Where the LAST control_tick's (x, z) actually came from - "vision" when
   *  backend/app/live_controller.py's best_hook_detection confidently saw
   *  the crane_hook this tick and unprojected it, "self_reported" when it
   *  fell back to the frontend's own dead-reckoned craneRef.position. Purely
   *  observational (nothing reads this to drive behavior) - exists so this
   *  is visibly provable in the UI/debug hook, not just claimed. See
   *  training/README.md's hook-visibility-rate note for why "self_reported"
   *  is expected often, not just an edge case. */
  positionSource: "vision" | "self_reported";

  start: () => void;
  stop: () => void;
  reset: () => void;
  setPhase: (phase: LivePhase, note?: string) => void;
  setActiveRelays: (relays: Set<string>) => void;
  setBackDetections: (dets: Detection[]) => void;
  markBalePicked: (id: number) => void;
  markBaleKnocked: (id: number) => void;
  markBaleFallen: (id: number) => void;
  setTargetBaleId: (id: number | null) => void;
  setThreshold: (t: number) => void;
  setSpeed: (s: Partial<LiveSpeedConfig>) => void;
  setPositionSource: (source: "vision" | "self_reported") => void;
  pushLog: (line: string) => void;
}

function freshBales(): LiveBale[] {
  return buildBaleGrid(1).map((b) => ({ ...b, picked: false, knocked: false, fallen: false }));
}

export const useLiveStore = create<LiveStoreState>((set, get) => ({
  running: false,
  phase: "idle",
  activeRelays: new Set(),
  bales: freshBales(),
  backDetections: [],
  threshold: 0.3,
  detectIntervalMs: 800,
  speed: DEFAULT_LIVE_SPEED,
  log: [],
  targetBaleId: null,
  positionSource: "self_reported",

  start: () => set({ running: true, phase: "select_target" }),
  stop: () => set({ running: false, activeRelays: new Set() }),
  reset: () => {
    craneRef.position.set(0, 5.0, 0);
    craneRef.thrust = 0;
    commandedRef.target.set(0, 5.0, 0);
    resetBaleBodies();
    set({
      running: false,
      phase: "idle",
      activeRelays: new Set(),
      bales: freshBales(),
      backDetections: [],
      targetBaleId: null,
      log: [],
      positionSource: "self_reported",
    });
  },
  setPhase: (phase, note) => {
    set({ phase });
    if (note) get().pushLog(note);
  },
  setActiveRelays: (relays) => set({ activeRelays: relays }),
  setBackDetections: (dets) => set({ backDetections: dets }),
  markBalePicked: (id) =>
    set((s) => ({ bales: s.bales.map((b) => (b.id === id ? { ...b, picked: true } : b)) })),
  markBaleKnocked: (id) =>
    set((s) => ({ bales: s.bales.map((b) => (b.id === id ? { ...b, knocked: true } : b)) })),
  markBaleFallen: (id) =>
    set((s) => ({ bales: s.bales.map((b) => (b.id === id ? { ...b, fallen: true } : b)) })),
  setTargetBaleId: (id) => set({ targetBaleId: id }),
  setThreshold: (t) => set({ threshold: t }),
  setSpeed: (s) => set((state) => ({ speed: { ...state.speed, ...s } })),
  setPositionSource: (source) => set({ positionSource: source }),
  pushLog: (line) =>
    set((s) => ({ log: [...s.log.slice(-49), `${new Date().toLocaleTimeString()}  ${line}`] })),
}));

// Debug hook for manual inspection (browser devtools console) during
// development of the live 3D mode - not used by any app code path.
if (typeof window !== "undefined") {
  (window as unknown as { __live: unknown }).__live = {
    store: useLiveStore,
    craneRef,
    commandedRef,
    baleBodies,
    TRUCK,
    CONVEYOR,
    CRANE,
    CAMERA_TOP,
    CAMERA_BACK,
    // Dev-only matte-mode toggle (see matteMode.ts) - driven externally by
    // tools/generate_hook_dataset.mjs to extract crane_hook training bboxes.
    setMatteMode: useMatteMode.getState().setMatteMode,
  };
}
