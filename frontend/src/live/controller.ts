// Local "drive" loop for the live 3D mode.
//
// All the actual control decisions - which bale to target, whether to
// re-confirm it against live vision, which relay(s) to energize, what the
// current setpoint is - now happen server-side, in
// backend/app/live_controller.py's compute_tick, driven by
// POST /api/control_tick (see CameraFeed.tsx's top-camera tick, which posts
// there and writes the result into `commandedRef`/the store).
//
// This hook's only job is to react to that, every rendered frame (60fps):
// move craneRef.position toward commandedRef.target on whichever axis has
// an active relay, at the speed that relay pair implies, and stop itself
// at arrival - the way a real servo/VFD drive respects a PLC-given
// setpoint between scans, rather than deciding anything on its own. A pure
// "just obey the relay for a fixed duration" model would overshoot at our
// ~1.3Hz vision tick rate (see live_controller.py's module docstring) -
// this local arrival-clamp is what avoids that without the frontend making
// any decisions of its own.

import { useEffect, useRef } from "react";
import { CRANE } from "./factoryLayout";
import { craneRef, commandedRef, useLiveStore, type LiveSpeedConfig } from "./liveStore";

// Every axis (E/W, N/S, hoist) drives at one constant speed now - the
// backend (live_controller.py's _axis_relays/_hoist_relays) only ever
// energizes a single "1s" relay per tick, never a "2s" speed-boost relay
// alongside it and never two axes at once (see that module's comment) -
// so there's no second speed tier for the frontend to pick between either.
const HOIST_SPEED_MS = (CRANE.hoistUpY - CRANE.hoistDownY) / 1.5; // hoistClearS

function xVelocity(relays: Set<string>, speed: LiveSpeedConfig): number {
  if (relays.has("east_1s")) return speed.ewMs;
  if (relays.has("west_1s")) return -speed.ewMs;
  return 0;
}

function zVelocity(relays: Set<string>, speed: LiveSpeedConfig): number {
  if (relays.has("south_1s")) return speed.nsMs;
  if (relays.has("north_1s")) return -speed.nsMs;
  return 0;
}

function yVelocity(relays: Set<string>): number {
  if (relays.has("up_1s")) return HOIST_SPEED_MS;
  if (relays.has("down_1s")) return -HOIST_SPEED_MS;
  return 0;
}

/** Moves `current` toward `target` at `velocityMs`, clamping exactly at
 *  arrival rather than overshooting past it - the local "drive" half of the
 *  setpoint + relay pair the backend sends each tick. */
function driveToward(current: number, target: number, velocityMs: number, dt: number): number {
  if (velocityMs === 0) return current;
  const step = velocityMs * dt;
  const remaining = target - current;
  if (Math.abs(step) >= Math.abs(remaining)) return target;
  return current + step;
}

// setInterval, not requestAnimationFrame: rAF is fully SUSPENDED (not just
// throttled) by the browser whenever the tab is backgrounded/hidden, which
// would silently stop the crane mid-motion while the backend's control tick
// (setInterval-based, in CameraFeed.tsx) kept right on advancing the phase -
// a real problem for a "live" view someone might reasonably tab away from,
// not just a quirk of automated testing. setInterval keeps running (Chrome
// only clamps its minimum period, doesn't suspend it) regardless of
// visibility, and dt is still measured from real elapsed time either way so
// this doesn't cost any smoothness while the tab IS visible.
const DRIVE_INTERVAL_MS = 16;

/** Mounts the local drive loop. Call once (e.g. from LiveControlPage) - it
 *  reads liveStore.activeRelays + commandedRef and writes craneRef.position
 *  directly, no props needed. */
export function useLiveController() {
  const lastTs = useRef<number | null>(null);

  useEffect(() => {
    function tick() {
      const ts = performance.now();
      const store = useLiveStore.getState();
      if (!store.running) {
        lastTs.current = null;
        return;
      }
      // No dt cap here (unlike the old leg-based controller, which needed one
      // to avoid overshooting a precomputed target on a long stall):
      // driveToward() below always clamps exactly at arrival regardless of
      // step size, so a big dt after a throttled/backgrounded gap just means
      // "snap straight to the current setpoint" - correct, not a bug - and a
      // small cap here would instead have silently put the crane into slow
      // motion under timer throttling (confirmed live: a hidden/backgrounded
      // tab clamps setInterval's actual firing rate well below 16ms).
      const dt = lastTs.current == null ? 0 : (ts - lastTs.current) / 1000;
      lastTs.current = ts;
      if (dt <= 0) return;

      const relays = store.activeRelays;
      const { target } = commandedRef;
      craneRef.position.x = driveToward(craneRef.position.x, target.x, xVelocity(relays, store.speed), dt);
      craneRef.position.y = driveToward(craneRef.position.y, target.y, yVelocity(relays), dt);
      craneRef.position.z = driveToward(craneRef.position.z, target.z, zVelocity(relays, store.speed), dt);
    }

    const id = setInterval(tick, DRIVE_INTERVAL_MS);
    return () => clearInterval(id);
  }, []);
}
