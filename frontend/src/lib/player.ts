import type { CraneKeyframe, RelayEvent } from "./types";

export interface CraneState {
  x: number;
  y: number;
  hoist: number;
  spiked: boolean;
  active_bale_id: number | null;
  phase: string;
}

/** Linearly interpolate crane position/hoist between the two keyframes
 * surrounding t_ms. Discrete fields (spiked/active_bale_id/phase) take the
 * value of the keyframe at/just-before t_ms. */
export function interpolateCrane(keyframes: CraneKeyframe[], t_ms: number): CraneState {
  if (keyframes.length === 0) {
    return { x: 0, y: 0, hoist: 1, spiked: false, active_bale_id: null, phase: "idle" };
  }
  if (t_ms <= keyframes[0].t_ms) {
    const k = keyframes[0];
    return { x: k.x, y: k.y, hoist: k.hoist, spiked: k.spiked, active_bale_id: k.active_bale_id, phase: k.phase };
  }
  const last = keyframes[keyframes.length - 1];
  if (t_ms >= last.t_ms) {
    return { x: last.x, y: last.y, hoist: last.hoist, spiked: last.spiked, active_bale_id: last.active_bale_id, phase: last.phase };
  }

  let lo = keyframes[0];
  let hi = keyframes[keyframes.length - 1];
  for (let i = 0; i < keyframes.length - 1; i++) {
    if (keyframes[i].t_ms <= t_ms && t_ms <= keyframes[i + 1].t_ms) {
      lo = keyframes[i];
      hi = keyframes[i + 1];
      break;
    }
  }
  const span = hi.t_ms - lo.t_ms;
  const frac = span > 0 ? (t_ms - lo.t_ms) / span : 0;
  return {
    x: lo.x + (hi.x - lo.x) * frac,
    y: lo.y + (hi.y - lo.y) * frac,
    hoist: lo.hoist + (hi.hoist - lo.hoist) * frac,
    spiked: lo.spiked,
    active_bale_id: lo.active_bale_id,
    phase: hi.phase,
  };
}

/** Which relays are energized at t_ms. */
export function activeRelays(timeline: RelayEvent[], t_ms: number): Set<string> {
  const active = new Set<string>();
  for (const ev of timeline) {
    if (ev.on_at_ms <= t_ms && t_ms < ev.off_at_ms) {
      active.add(ev.relay);
    }
  }
  return active;
}
