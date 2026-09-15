// Dev-only rendering mode used by tools/generate_hook_dataset.mjs to extract
// pixel-accurate crane_hook training bboxes: the same unique-color-matte
// technique blender/generate_synthetic_topdown.py already uses successfully
// (see its module docstring) - the hook (+ spike, same physical assembly)
// renders as one bright unlit color, everything else renders flat unlit
// black, so a bbox is just "scan for the tracked color", no per-object ID
// bookkeeping needed since only one object is tracked here. Real WebGL
// depth-buffer occlusion still applies normally, so a bale blocking the
// hook still blocks it in the matte render too - the resulting bbox matches
// exactly what's actually visible, not a naive projected-corners box.
//
// meshBasicMaterial (not meshStandardMaterial) for both colors when matte
// mode is on - fully unlit, so scene lighting can't introduce shading
// gradients that would throw off an exact-color-match scan.

import { createElement, type ReactElement } from "react";
import { create } from "zustand";

export const MATTE_HOOK_COLOR = "#00ff5f";
export const MATTE_BLACK = "#000000";

// Per-bale ID-color matte, mirroring blender/generate_synthetic_topdown.py's
// index_to_color/bboxes_from_matte approach (per that script's docstring) -
// instead of one matte pass per tracked object, every simultaneously-visible
// bale gets its OWN distinct flat color in the SAME pass the hook uses, so a
// single matte frame yields real-occlusion-aware bboxes for every bale AND
// the hook at once. This exists specifically to fix a real regression this
// session found: the first synth3d dataset (generate_hook_dataset.mjs
// before this was added) only ever emitted crane_hook boxes even though
// bales were clearly visible in nearly every frame - fine-tuning on ~300
// images where visible bales carried NO ground-truth box taught the model
// to suppress bale detections specifically on Three.js-rendered frames
// (confirmed: 0 bale detections down to threshold 0.05 on a live frame with
// an obviously visible bale stack). Encoded in the RED channel only (G=B=0)
// so it can never collide with MATTE_HOOK_COLOR's green-dominant range or
// with pure black - decoded with AA-tolerant rounding by the same scan that
// finds the hook.
export function baleMatteColor(id: number): string {
  const r = 20 + (id % 24) * 10; // 20..250 - safely within a byte, spaced 10 apart for AA tolerance
  return "#" + r.toString(16).padStart(2, "0") + "0000";
}

interface MatteModeState {
  matteMode: boolean;
  setMatteMode: (on: boolean) => void;
}

export const useMatteMode = create<MatteModeState>((set) => ({
  matteMode: false,
  setMatteMode: (on) => set({ matteMode: on }),
}));

/** In matte mode, every mesh is either the one bright tracked color (the
 *  crane hook + spike - same physical assembly, the only thing training
 *  data needs a bbox for) or flat black; normal mode renders as usual.
 *  Plain function, not its own component - callers read `matteMode` once
 *  per parent render and thread it through, instead of every single mesh
 *  subscribing to the store separately. */
export function matteAwareMaterial(
  matteMode: boolean,
  normalColor: string,
  /** true = the fixed hook/spike tracked color; false = flat black;
   *  a string = an explicit override color (e.g. baleMatteColor(id)) for
   *  per-instance ID-color matte extraction. */
  tracked: boolean | string = false,
  extra?: { metalness?: number; roughness?: number; map?: import("three").Texture },
): ReactElement {
  if (matteMode) {
    // toneMapped:false - confirmed live this session that THREE's default
    // tone-mapping pipeline shifts a flat hex color before it reaches the
    // framebuffer (#00ff5f rendered as rgb(147,228,137), not (0,255,95)) -
    // bypassing it is what makes the color-scan match what was actually
    // specified, not a tone-mapped approximation of it.
    const color = tracked === true ? MATTE_HOOK_COLOR : tracked === false ? MATTE_BLACK : tracked;
    return createElement("meshBasicMaterial", { color, toneMapped: false });
  }
  return createElement("meshStandardMaterial", { color: normalColor, ...extra });
}
