// The static-ish factory scene: truck, conveyor, live bale stack, crane rig.
// Rendered identically inside all three canvases (main orbit view + the two
// fixed camera feeds) so every view shows the same live state.

import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { useTexture } from "@react-three/drei";
import * as THREE from "three";
import { useLiveStore, type LiveBale } from "./liveStore";
import { TRUCK, CONVEYOR, BALE_SIZE } from "./factoryLayout";
import { ensureBaleBody } from "./physicsBridge";
import { useMatteMode, matteAwareMaterial, baleMatteColor } from "./matteMode";
import CraneRig from "./CraneRig";

const BALE_TEXTURES = [
  "/bale-textures/bale-1.png",
  "/bale-textures/bale-2.png",
  "/bale-textures/bale-3.png",
  "/bale-textures/bale-4.png",
];

// Position is read imperatively every frame from the shared physics bridge
// (see physicsBridge.ts) rather than from a React prop, the same pattern
// CraneRig uses for craneRef - the bale's real position comes from a single
// live physics simulation (PhysicsBales.tsx, mounted only in MainView's
// Canvas) so this mesh, rendered identically inside all three canvases,
// always matches what actually happened physically, not just where the
// static procedural grid put it.
function BaleMesh({ bale, texture }: { bale: LiveBale; texture: THREE.Texture }) {
  const matteMode = useMatteMode((s) => s.matteMode);
  const ref = useRef<THREE.Mesh>(null!);
  useFrame(() => {
    const state = ensureBaleBody(bale.id, bale.position);
    ref.current.position.copy(state.position);
  });
  return (
    <mesh ref={ref} rotation={[0, bale.rotationY, 0]} castShadow>
      <boxGeometry args={BALE_SIZE} />
      {/* same texture on all 6 faces - flat single-face UVs were the
          bug that made bales undetectable in the Blender pipeline
          (see blender/README.md); reusing one material for every
          face here avoids repeating that mistake. Each bale gets its OWN
          matte color (baleMatteColor(bale.id), not flat black) so a single
          matte pass yields real-occlusion-aware ground-truth boxes for
          every visible bale AND the hook at once - see matteMode.ts's
          doc comment for why (a real regression this session hit and
          fixed: unlabeled-but-visible bales in training data suppressed
          bale detection on live-mode frames). */}
      {matteAwareMaterial(matteMode, "#ffffff", baleMatteColor(bale.id), { map: texture, roughness: 0.9 })}
    </mesh>
  );
}

function BaleStack() {
  const bales = useLiveStore((s) => s.bales);
  const textures = useTexture(BALE_TEXTURES);

  return (
    <group>
      {bales
        .filter((b) => !b.picked)
        .map((b) => (
          <BaleMesh key={b.id} bale={b} texture={textures[b.id % textures.length]} />
        ))}
    </group>
  );
}

function Truck() {
  const matteMode = useMatteMode((s) => s.matteMode);
  return (
    <group>
      <mesh position={TRUCK.bedFloor.center}>
        <boxGeometry args={TRUCK.bedFloor.size} />
        {matteAwareMaterial(matteMode, "#6b4f3a")}
      </mesh>
      <mesh position={TRUCK.wallBack.center}>
        <boxGeometry args={TRUCK.wallBack.size} />
        {matteAwareMaterial(matteMode, "#4a3728")}
      </mesh>
      <mesh position={TRUCK.cab.center}>
        <boxGeometry args={TRUCK.cab.size} />
        {/* was #c92a2a (red) - close enough to the crane_spike's #e03131 that
            the top-camera detector kept flagging the cab as crane hardware
            (see live 3D mode bug reports). A plain white/gray cab keeps it
            visually far from every detection class' color (bale tan,
            hook amber, spike red). */}
        {matteAwareMaterial(matteMode, "#e9ecef")}
      </mesh>
      {TRUCK.wheels.map((w, i) => (
        <mesh key={i} position={w} rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[TRUCK.wheelRadius, TRUCK.wheelRadius, TRUCK.wheelWidth, 16]} />
          {matteAwareMaterial(matteMode, "#1a1a1a")}
        </mesh>
      ))}
    </group>
  );
}

function Conveyor() {
  const matteMode = useMatteMode((s) => s.matteMode);
  return (
    <group>
      <mesh position={CONVEYOR.platform.center}>
        <boxGeometry args={CONVEYOR.platform.size} />
        {matteAwareMaterial(matteMode, "#7d8590")}
      </mesh>
      <mesh position={CONVEYOR.deck.center}>
        <boxGeometry args={CONVEYOR.deck.size} />
        {matteAwareMaterial(matteMode, "#495057")}
      </mesh>
      {CONVEYOR.legs.map((l, i) => (
        <mesh key={i} position={l}>
          <boxGeometry args={CONVEYOR.legSize} />
          {matteAwareMaterial(matteMode, "#343a40")}
        </mesh>
      ))}
      <mesh position={CONVEYOR.stair.center}>
        <boxGeometry args={CONVEYOR.stair.size} />
        {matteAwareMaterial(matteMode, "#868e96")}
      </mesh>
    </group>
  );
}

export default function FactoryContents() {
  const matteMode = useMatteMode((s) => s.matteMode);
  return (
    <group>
      <ambientLight intensity={0.6} />
      <directionalLight position={[8, 12, 6]} intensity={1.1} castShadow />
      {/* CameraFeed.tsx's Canvas clears to a dark-navy CSS background
          (#12151a), not true black, wherever nothing is rendered - would
          corrupt an exact-color matte scan. Force a real black scene
          background while matte mode is on. */}
      {matteMode && <color attach="background" args={["#000000"]} />}
      {/* grid lines would show up as stray non-black pixels in a matte
          render (matteMode.ts) - hidden there, harmless either way. */}
      {!matteMode && <gridHelper args={[30, 30, "#495057", "#2b2f36"]} />}
      <Truck />
      <Conveyor />
      <BaleStack />
      <CraneRig />
    </group>
  );
}
