// The crane rig: rails (static) -> bridge (travels N/S=z) -> trolley (travels
// E/W=x, rides the bridge) -> mast (spans trolley height down to the hook's
// live hoist height) -> hook -> spike (a FIXED rigid prong - see
// factoryLayout.ts's CRANE.spikeReach comment. "Picking" is the whole crane
// sliding the prong into a bale via a real horizontal move, driven by
// controller.ts - not the spike itself animating).
//
// Position comes from `craneRef` (a plain mutable object, not zustand - see
// liveStore.ts) and is applied directly to Object3D refs inside useFrame, so
// the crane animates at full frame rate without going through React state.

import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { CRANE } from "./factoryLayout";
import { craneRef } from "./liveStore";
import { useMatteMode, matteAwareMaterial } from "./matteMode";

// Mounted flush with the hook's own +z face, protruding further toward -z
// (the cab side - TRUCK.bedMaxZ, +z, is the open/back end, see
// factoryLayout.ts) by exactly CRANE.spikeReach (one bale's depth). The
// hook block itself must stay OUTSIDE the bale once inserted - only the
// spike enters it - so this mounts at the hook's real front face rather
// than its center; see controller.ts's insertZFor for the matching
// approach-position math.
const HOOK_FRONT_Z = CRANE.hookSize[2] / 2;
const SPIKE_BASE_Z = -HOOK_FRONT_Z;
const SPIKE_TIP_Z = -(HOOK_FRONT_Z + CRANE.spikeReach);
const SPIKE_CENTER_Z = (SPIKE_BASE_Z + SPIKE_TIP_Z) / 2;
const SPIKE_MESH_LENGTH = SPIKE_BASE_Z - SPIKE_TIP_Z; // == CRANE.spikeReach

export default function CraneRig() {
  const matteMode = useMatteMode((s) => s.matteMode);
  const bridgeGroup = useRef<THREE.Group>(null!);
  const trolleyGroup = useRef<THREE.Group>(null!);
  const mastMesh = useRef<THREE.Mesh>(null!);
  const hookGroup = useRef<THREE.Group>(null!);

  useFrame(() => {
    const { x, y, z } = craneRef.position;
    if (bridgeGroup.current) bridgeGroup.current.position.z = z;
    if (trolleyGroup.current) trolleyGroup.current.position.x = x;

    // hook Y is relative to the trolley (which sits at world Y=railHeightY)
    const hookLocalY = y - CRANE.railHeightY;
    if (hookGroup.current) hookGroup.current.position.y = hookLocalY;

    const mastLen = Math.max(0.05, -hookLocalY);
    if (mastMesh.current) {
      mastMesh.current.scale.y = mastLen;
      mastMesh.current.position.y = -mastLen / 2;
    }
  });

  const [railX0, railX1] = CRANE.railX;
  const [railZ0, railZ1] = CRANE.railSpanZ;
  const railLen = Math.abs(railZ1 - railZ0);
  const railCenterZ = (railZ0 + railZ1) / 2;
  const bridgeLen = Math.abs(railX1 - railX0) + 0.4;
  const bridgeCenterX = (railX0 + railX1) / 2; // rails aren't centered on world x=0 (factory.blend's
  // crane rail span is offset from the truck centerline - see factoryLayout.ts) - the bridge beam
  // itself must be centered here too, not at local x=0, or it won't actually reach both rails.

  return (
    <group>
      {/* static rails + support legs */}
      <mesh position={[railX0, CRANE.railHeightY, railCenterZ]}>
        <boxGeometry args={[0.12, 0.12, railLen]} />
        {matteAwareMaterial(matteMode, "#3b5bdb")}
      </mesh>
      <mesh position={[railX1, CRANE.railHeightY, railCenterZ]}>
        <boxGeometry args={[0.12, 0.12, railLen]} />
        {matteAwareMaterial(matteMode, "#3b5bdb")}
      </mesh>
      {[railZ0, railZ1].map((rz) =>
        [railX0, railX1].map((rx) => (
          <mesh key={`${rx}-${rz}`} position={[rx, CRANE.railHeightY / 2, rz]}>
            <boxGeometry args={[0.2, CRANE.railHeightY, 0.2]} />
            {matteAwareMaterial(matteMode, "#495057")}
          </mesh>
        )),
      )}

      {/* bridge travels along z (N/S) */}
      <group ref={bridgeGroup} position={[0, CRANE.railHeightY, 0]}>
        <mesh position={[bridgeCenterX, 0, 0]}>
          <boxGeometry args={[bridgeLen, 0.2, 0.2]} />
          {matteAwareMaterial(matteMode, "#5c7cfa")}
        </mesh>

        {/* trolley rides the bridge, travels along x (E/W) */}
        <group ref={trolleyGroup}>
          <mesh>
            <boxGeometry args={[0.5, 0.3, 0.5]} />
            {matteAwareMaterial(matteMode, "#495057")}
          </mesh>

          {/* mast: dynamic length between trolley and hook */}
          <mesh ref={mastMesh} position={[0, -1, 0]}>
            <cylinderGeometry args={[0.08, 0.08, 1, 12]} />
            {matteAwareMaterial(matteMode, "#868e96")}
          </mesh>

          {/* hook, positioned at the mast's live bottom - the tracked
              "crane head" for matte-mode training data (matteMode.ts) */}
          <group ref={hookGroup} position={[0, -(CRANE.railHeightY - CRANE.hoistDownY), 0]}>
            <mesh>
              <boxGeometry args={CRANE.hookSize} />
              {matteAwareMaterial(matteMode, "#fab005", true)}
            </mesh>
            <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, SPIKE_CENTER_Z]}>
              <cylinderGeometry args={[CRANE.spikeRadius * 1.4, CRANE.spikeRadius * 1.4, SPIKE_MESH_LENGTH, 10]} />
              {matteAwareMaterial(matteMode, "#e03131", true, { metalness: 0.5, roughness: 0.4 })}
            </mesh>
          </group>
        </group>
      </group>
    </group>
  );
}
