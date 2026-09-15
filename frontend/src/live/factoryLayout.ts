// Procedural factory-scene geometry for the live 3D control mode.
//
// Numbers here are transcribed from blender/factory.blend (dumped headlessly
// this session - see training/README.md's sibling investigation and
// blender/README.md's "Layout" section for the prose version). factory.blend
// is Z-up, meters, origin arbitrary (truck centered around blender X=9.0,
// Y=2.75). Three.js is Y-up, so every position below goes through
// `blenderToThree()`, re-centered on the truck so the live scene sits near
// the world origin.
//
// Simplified vs. the real .blend: we keep the load-bearing/visually-legible
// pieces (bed, rails, walls, cab, wheels, conveyor deck+legs+platform,
// rail+bridge+trolley+mast+hook+spike) and drop pure ornamentation (individual
// bed posts/rail tiers, stair treads) - recognizable proportions, not a 1:1
// mesh count match.

import * as THREE from "three";

// Blender truck-centerline origin, subtracted before axis conversion.
const ORIGIN_BX = 9.0;
const ORIGIN_BY = 2.75;

/** Blender (Z-up, meters) -> Three.js (Y-up) position, re-centered on the truck. */
export function blenderToThree(bx: number, by: number, bz: number): THREE.Vector3 {
  return new THREE.Vector3(bx - ORIGIN_BX, bz, -(by - ORIGIN_BY));
}

// ---------------------------------------------------------------------------
// Truck (blender/factory.blend: TruckBed_*, TruckCab, Wheel_*, Chassis_Rail_*)
// ---------------------------------------------------------------------------
export const TRUCK = {
  bedFloor: { center: blenderToThree(9.0, 2.75, 0.85), size: [2.2, 0.3, 5.5] as [number, number, number] },
  bedTopY: blenderToThree(9.0, 2.75, 1.0).y, // top surface of the bed floor
  wallBack: { center: blenderToThree(9.0, 5.5, 1.2), size: [2.2, 1.0, 0.12] as [number, number, number] },
  cab: { center: blenderToThree(9.0, 6.1, 1.45), size: [2.0, 1.5, 1.0] as [number, number, number] },
  wheels: [
    blenderToThree(7.85, 0.6, 0.45),
    blenderToThree(10.15, 0.6, 0.45),
    blenderToThree(7.85, 4.9, 0.45),
    blenderToThree(10.15, 4.9, 0.45),
  ],
  wheelRadius: 0.45,
  wheelWidth: 0.3,
  // bed footprint in local (x=E/W, z=N/S) - used to seed the bale grid and
  // as the live mode's fixed "detection area" (no manual marking here).
  bedMinX: -1.1,
  bedMaxX: 1.1,
  bedMinZ: -(5.5 - ORIGIN_BY), // blender y=5.5 -> most-negative three.z
  bedMaxZ: ORIGIN_BY, // blender y=0 -> most-positive three.z (open/back end)
};

// ---------------------------------------------------------------------------
// Conveyor (Conveyor_Platform/Deck/Leg_*, Stair_*)
// ---------------------------------------------------------------------------
export const CONVEYOR = {
  platform: { center: blenderToThree(11.4, 1.3, 0.55), size: [1.4, 1.1, 3.0] as [number, number, number] },
  deck: { center: blenderToThree(11.4, 1.3, 1.14), size: [1.3, 0.08, 2.9] as [number, number, number] },
  deckTopY: blenderToThree(11.4, 1.3, 1.18).y,
  legs: [
    blenderToThree(10.9, 0.0, 0.55),
    blenderToThree(10.9, 2.6, 0.55),
    blenderToThree(11.9, 0.0, 0.55),
    blenderToThree(11.9, 2.6, 0.55),
  ],
  legSize: [0.14, 1.1, 0.14] as [number, number, number],
  // simplified single ramp box standing in for the 6-step staircase
  stair: { center: blenderToThree(12.9, 0.1, 0.32), size: [0.7, 0.64, 1.4] as [number, number, number] },
  // fixed single drop point (conveyor carries dropped bales away - see
  // backend/app/planner.py's "one fixed drop point" convention).
  dropPoint: blenderToThree(11.4, 1.3, 1.4),
};

// ---------------------------------------------------------------------------
// Bale grid (Bale_0..19: 0.85 x 0.85 x 0.6m, 2 cols x 6 rows first layer,
// 2 cols x 4 rows partial second layer)
// ---------------------------------------------------------------------------
export const BALE_SIZE: [number, number, number] = [0.85, 0.6, 0.85]; // [x, y(height), z]

export interface BaleSpec {
  id: number;
  position: THREE.Vector3; // center, three.js space
  rotationY: number;
  layer: 0 | 1;
}

/** Deterministic small PRNG (mulberry32) so the layout is stable across reloads. */
function mulberry32(seed: number) {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function buildBaleGrid(seed = 1): BaleSpec[] {
  const rand = mulberry32(seed);
  const bales: BaleSpec[] = [];
  const [, bh] = BALE_SIZE;
  const colXs = [-0.45, 0.45]; // two columns across the bed width
  const rowCount = 6;
  const rowSpacing = 0.92;
  const rowStartZ = TRUCK.bedMaxZ - 0.6; // start near the open/back end
  let id = 0;

  // first layer: 2 cols x 6 rows
  const layer0Y = TRUCK.bedTopY + bh / 2;
  for (let row = 0; row < rowCount; row++) {
    for (const cx of colXs) {
      const jx = (rand() - 0.5) * 0.05;
      const jz = (rand() - 0.5) * 0.05;
      bales.push({
        id: id++,
        position: new THREE.Vector3(cx + jx, layer0Y, rowStartZ - row * rowSpacing + jz),
        rotationY: (rand() - 0.5) * 0.14, // +-4deg jitter
        layer: 0,
      });
    }
  }

  // second layer: partial coverage, 2 cols x 4 rows, offset toward the back end
  const layer1Y = layer0Y + bh;
  for (let row = 0; row < 4; row++) {
    for (const cx of colXs) {
      const jx = (rand() - 0.5) * 0.05;
      const jz = (rand() - 0.5) * 0.05;
      bales.push({
        id: id++,
        position: new THREE.Vector3(cx + jx, layer1Y, rowStartZ - row * rowSpacing + jz),
        rotationY: (rand() - 0.5) * 0.14,
        layer: 1,
      });
    }
  }

  return bales;
}

// ---------------------------------------------------------------------------
// Crane rig (Rail_*, RailLeg_*, Bridge, Trolley, Mast, Hook, Spike)
// ---------------------------------------------------------------------------
export const CRANE = {
  railHeightY: blenderToThree(0, 0, 5.5).y,
  // trolley (E/W = local x) travel range, between the two rails
  ewMin: blenderToThree(7.5, 0, 0).x + 0.3,
  ewMax: blenderToThree(12.9, 0, 0).x - 0.3,
  // bridge (N/S = local z) travel range, between the rail-leg spans
  nsMin: blenderToThree(0, 9.0, 0).z + 0.3,
  nsMax: blenderToThree(0, -3.5, 0).z - 0.3,
  // hoist (vertical) range - hook height at hoist=0 (fully down/loaded) vs 1 (clear)
  hoistDownY: 1.4,
  hoistUpY: 5.0,
  // The spike is a fixed rigid prong (like a real forklift/bale-grab tine),
  // not a telescoping actuator - it's mounted flush with the hook's own
  // front face and protrudes toward -z (the cab-facing side - see
  // CraneRig.tsx) by exactly one bale's depth (BALE_SIZE[2]), matching a
  // real bale spike sized to skewer a single bale end-to-end. "Picking" is
  // accomplished by MOVING the whole crane horizontally so the prong slides
  // into the bale (see controller.ts's standoff-then-insert approach), not
  // by animating the spike itself. controller.ts positions the crane so the
  // HOOK stops at the bale's near face (outside it) while only the spike
  // spans the bale's interior, reaching exactly to its far face - see
  // controller.ts's insertZFor.
  spikeReach: BALE_SIZE[2],
  spikeRadius: 0.05,
  hookSize: [0.24, 0.16, 0.24] as [number, number, number],
  railX: [
    blenderToThree(7.5, 0, 0).x,
    blenderToThree(12.9, 0, 0).x,
  ],
  railSpanZ: [blenderToThree(0, -3.5, 0).z, blenderToThree(0, 9.0, 0).z] as [number, number],
};

// ---------------------------------------------------------------------------
// Cameras (Camera_Top, Camera_Back)
// ---------------------------------------------------------------------------
export const SENSOR_WIDTH_MM = 36;

export const CAMERA_TOP = {
  position: blenderToThree(10.2, 2.75, 9.5),
  lookAt: blenderToThree(10.2, 2.75, 0), // straight down at the bed/stack center
  lensMm: 16,
};

// Was aimed low (bed height, Blender z=1.3) from close range (Blender
// y=-4.751, ~7.5m out) with a fairly tight 24mm lens - fine for the bale
// stack, but the crane hook spends most of the pick cycle well above that:
// it hoists up to CRANE.hoistUpY (three.js y=5.0) between every cross/drop
// move, and that's ~28deg above this camera's old boresight - outside its
// ~46deg vertical FOV, so the hook (the very thing this feed exists to show)
// dropped out of frame for a large fraction of every cycle. Pulled back
// further (Blender y=-6.75, ~9.5m out), raised, and re-aimed at Blender
// z=3.2 - the vertical midpoint of the hook's hoistDownY..hoistUpY travel,
// not the bed floor - so the hook stays inside frame at both hoist extremes
// with margin; widened the lens a bit for extra headroom.
export const CAMERA_BACK = {
  position: blenderToThree(10.2, -6.75, 3.8),
  lookAt: blenderToThree(10.2, 2.75, 3.2),
  lensMm: 20,
};

/** Blender sensor-width/lens -> horizontal FOV in degrees. */
export function horizontalFovDeg(lensMm: number, sensorMm = SENSOR_WIDTH_MM): number {
  return THREE.MathUtils.radToDeg(2 * Math.atan(sensorMm / (2 * lensMm)));
}

/** Horizontal FOV -> vertical FOV (what THREE.PerspectiveCamera.fov expects) for a given aspect ratio. */
export function verticalFovDeg(hFovDeg: number, aspect: number): number {
  const hHalfRad = THREE.MathUtils.degToRad(hFovDeg) / 2;
  return THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(hHalfRad) / aspect));
}
