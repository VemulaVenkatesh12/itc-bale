// Shared mutable per-bale physics state.
//
// Only ONE canvas actually runs the physics simulation (`PhysicsBales`,
// mounted inside MainView.tsx's Canvas - see that file). Real rigid-body
// collision/gravity only needs to be computed once; if all three canvases
// (main view + two fixed camera feeds) each ran their own `<Physics>` world
// they'd diverge frame-to-frame (independent float error, independent dt),
// so the camera feeds would show bales in slightly different places than
// the main view and than each other - useless for "does the top camera
// actually see the bale where it physically is" detection.
//
// Instead this map is the single source of truth: PhysicsBales writes into
// it once per frame from the simulated rigid bodies, and every
// FactoryContents instance's BaleStack (in all three canvases) reads from
// it to position the bale meshes - the same "one mutable object, many
// imperative followers" pattern `craneRef` (see liveStore.ts) already uses
// for the crane rig.

import * as THREE from "three";

export interface BalePhysicsState {
  position: THREE.Vector3;
  /** Touched by the crane's hook/spike while it was NOT the pick target - a mistake. */
  knocked: boolean;
  /** Dropped off the truck bed entirely (knocked hard enough to fall). */
  fallen: boolean;
}

export const baleBodies = new Map<number, BalePhysicsState>();

/** Get-or-create so readers (BaleStack, before physics has run a frame) and
 *  the writer (PhysicsBales) can both call this safely in any order. */
export function ensureBaleBody(id: number, initial: THREE.Vector3): BalePhysicsState {
  let state = baleBodies.get(id);
  if (!state) {
    state = { position: initial.clone(), knocked: false, fallen: false };
    baleBodies.set(id, state);
  }
  return state;
}

// Grace period after every (re)spawn before knock/fallen detection is
// trusted: a freshly-spawned stack has bales "just touching" each other and
// the bed floor with effectively zero gap (see factoryLayout.ts's
// buildBaleGrid), which needs a few physics steps to settle into a proper
// resting contact - without this, the settle itself can register as a
// false-positive "knock" before the crane has gone anywhere near the stack.
let armedAt = performance.now() + 1000;

export function armPhysicsDetection(graceMs = 1000): void {
  armedAt = performance.now() + graceMs;
}

export function isPhysicsDetectionArmed(): boolean {
  return performance.now() >= armedAt;
}

export function resetBaleBodies(): void {
  baleBodies.clear();
  armPhysicsDetection();
}

/** Phases during which the pick target is welded to the spike with a real
 *  Rapier fixed joint (see PhysicsBales.tsx's BaleJoint) instead of being
 *  simulated as a free dynamic body - the bale stays DYNAMIC throughout;
 *  it's the joint constraint, not a scripted position copy, that makes it
 *  follow the kinematic spike as the crane hoists/moves it to the drop
 *  point - so there's no visible "snap" the way a scripted follow would
 *  produce if the join point weren't already exact. */
export const CARRYING_PHASES = new Set([
  "hoist_up_clear",
  "drop_cross",
  "drop_descend",
  "hoist_down_to_drop",
  "push_off",
]);

/** Phases during which the pick target is held kinematically AT ITS OWN
 *  resting spot (not following the hook) while the crane approaches and the
 *  spike slides in - see PhysicsBales.tsx. A real solid spike sliding into a
 *  still-dynamic bale doesn't "enter" it, it bulldozes it (pushes it bodily
 *  out of the way, into whatever's behind it) since nothing here models an
 *  actual perforation. Pinning the target kinematically for the approach
 *  lets the crane's collider pass through it without any contact force,
 *  while it still visually stays put; it only starts following the hook
 *  once the insert is actually done (CARRYING_PHASES). */
export const PINNED_PHASES = new Set(["cross", "descend", "hoist_down_to_bale", "pierce", "pierce_settle"]);

/** A bale is considered "knocked" once it moves this fast while it is not
 *  the active pick target. The stack is deliberately packed with near-zero
 *  gaps (see factoryLayout.ts's buildBaleGrid), so pulling one bale free
 *  realistically jostles its immediate touching neighbours a little even on
 *  a clean pick - this threshold is tuned to sit above that routine settling
 *  (observed ~0.4-0.5 m/s) and only flag an actual hit. */
export const KNOCK_SPEED_THRESHOLD = 0.6; // m/s
export const KNOCK_LOG_COOLDOWN_MS = 2000;
