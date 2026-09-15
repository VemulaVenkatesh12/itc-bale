// Real rigid-body physics for the bale stack - gravity + collision, via
// @react-three/rapier. Mounted exactly once, inside MainView's Canvas (see
// MainView.tsx); it renders no visuals of its own (all colliders are
// invisible - the existing Truck()/BaleStack() meshes in FactoryContents.tsx
// already draw the truck bed and bales). Its job is purely to simulate and
// write results into `physicsBridge.baleBodies` for every canvas to read.
//
// This is what actually distinguishes a clean pick from the crane clipping
// a bale it isn't targeting: the crane's hook/spike is a real kinematic
// collider (`DangerZone`), every non-target bale is a real dynamic rigid
// body under gravity, and a genuine collision or an unexpected velocity spike
// on a non-target bale is logged as a knock - not inferred from timeline
// bookkeeping. Once a pick is complete, the bale is WELDED to the spike with
// a real Rapier fixed joint (BaleJoint) - the crane carries it because the
// joint constraint pulls it along, not because anything scripts its position.

import { useEffect, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import { Physics, RigidBody, CuboidCollider, useFixedJoint, type RapierRigidBody } from "@react-three/rapier";
import { TRUCK, CRANE, BALE_SIZE } from "./factoryLayout";
import { craneRef, useLiveStore, type LiveBale } from "./liveStore";
import {
  ensureBaleBody,
  armPhysicsDetection,
  isPhysicsDetectionArmed,
  CARRYING_PHASES,
  PINNED_PHASES,
  KNOCK_SPEED_THRESHOLD,
  KNOCK_LOG_COOLDOWN_MS,
} from "./physicsBridge";

const [BALE_HX, BALE_HY, BALE_HZ] = BALE_SIZE.map((s) => s / 2) as [number, number, number];

// The hook block has to stay OUTSIDE the bale once inserted - only the spike
// (mounted flush with the hook's own +z face, exactly one bale-depth long -
// see factoryLayout.ts's CRANE.spikeReach and CraneRig.tsx) enters it. Both
// offsets are relative to craneRef, which IS the hook's world position (see
// CraneRig.tsx's comment on why that nesting cancels out to identity).
const HOOK_FRONT_Z = CRANE.hookSize[2] / 2;
const SPIKE_TIP_Z = -(HOOK_FRONT_Z + CRANE.spikeReach);
const DANGER_ZONE_OFFSET_Z = (HOOK_FRONT_Z + SPIKE_TIP_Z) / 2;
const DANGER_ZONE_HALF: [number, number, number] = [
  CRANE.hookSize[0] / 2 + 0.03,
  CRANE.hookSize[1] / 2 + 0.03,
  (HOOK_FRONT_Z - SPIKE_TIP_Z) / 2,
];

// Shared handle to the spike/danger-zone rigid body - a plain mutable ref
// (same pattern as liveStore.ts's craneRef), not a React ref, since it has
// to be readable from every PhysicsBale instance to attach a joint to it.
// Structurally compatible with the RefObject useFixedJoint expects.
const spikeBodyRef: { current: RapierRigidBody | null } = { current: null };

function DangerZone() {
  useFrame(() => {
    if (!spikeBodyRef.current) return;
    const { x, y, z } = craneRef.position;
    spikeBodyRef.current.setNextKinematicTranslation({ x, y, z: z + DANGER_ZONE_OFFSET_Z });
  });
  return (
    <RigidBody
      ref={(r) => {
        spikeBodyRef.current = r;
      }}
      type="kinematicPosition"
      colliders={false}
      name="crane-danger-zone"
    >
      <CuboidCollider args={DANGER_ZONE_HALF} />
    </RigidBody>
  );
}

/** Welds a bale's rigid body to the spike with a real fixed joint, at a
 *  fixed local offset captured once at the moment the pick completes (see
 *  PhysicsBale's effect below) - from then on the bale follows the spike
 *  because the joint constrains it to, the same way a real skewered bale
 *  would ride along, not because anything overwrites its transform. */
function BaleJoint({
  baleRef,
  offset,
}: {
  baleRef: React.RefObject<RapierRigidBody | null>;
  offset: [number, number, number];
}) {
  // Only ever mounted once both bodies are confirmed non-null (see the
  // effect in PhysicsBale that captures `offset`) - useFixedJoint's types
  // want non-nullable RefObjects, hence the cast.
  useFixedJoint(spikeBodyRef as React.RefObject<RapierRigidBody>, baleRef as React.RefObject<RapierRigidBody>, [
    offset,
    [0, 0, 0, 1],
    [0, 0, 0],
    [0, 0, 0, 1],
  ]);
  return null;
}

function PhysicsBale({ bale }: { bale: LiveBale }) {
  const body = useRef<RapierRigidBody>(null);
  const phase = useLiveStore((s) => s.phase);
  const targetBaleId = useLiveStore((s) => s.targetBaleId);
  const lastLogRef = useRef(0);
  const [jointOffset, setJointOffset] = useState<[number, number, number] | null>(null);

  const isTarget = bale.id === targetBaleId;
  const isPinned = isTarget && PINNED_PHASES.has(phase);
  const isCarrying = isTarget && CARRYING_PHASES.has(phase);

  function logOnce(line: string) {
    const now = performance.now();
    if (now - lastLogRef.current < KNOCK_LOG_COOLDOWN_MS) return;
    lastLogRef.current = now;
    useLiveStore.getState().pushLog(line);
  }

  // The instant the pick completes (pierce -> hoist_up_clear), weld the
  // bale to the spike at whatever their CURRENT relative offset is - since
  // the bale was pinned exactly on-target the whole approach (see
  // PINNED_PHASES), this is already the correct "skewered" pose, so the
  // joint takes over with zero correction and no visible snap.
  useEffect(() => {
    if (isCarrying && !jointOffset && body.current && spikeBodyRef.current) {
      const balePos = body.current.translation();
      const spikePos = spikeBodyRef.current.translation();
      setJointOffset([balePos.x - spikePos.x, balePos.y - spikePos.y, balePos.z - spikePos.z]);
    }
    if (!isCarrying && jointOffset) {
      setJointOffset(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isCarrying]);

  useFrame(() => {
    if (!body.current) return;
    const state = ensureBaleBody(bale.id, bale.position);

    if (isPinned) {
      // Held kinematically at its own resting spot while the crane
      // approaches and the spike slides in - see PINNED_PHASES in
      // physicsBridge.ts for why this has to happen (a solid spike vs. a
      // still-dynamic bale would bulldoze it instead of "entering" it).
      body.current.setNextKinematicTranslation({ x: bale.position.x, y: bale.position.y, z: bale.position.z });
      state.position.copy(bale.position);
      return;
    }

    // Carrying (jointed) or free - either way the body is dynamic now, so
    // just read back wherever the physics step actually put it.
    const t = body.current.translation();
    state.position.set(t.x, t.y, t.z);

    if (!isPhysicsDetectionArmed()) return; // still settling from spawn - see physicsBridge.ts

    if (!isTarget) {
      const v = body.current.linvel();
      const speed = Math.hypot(v.x, v.y, v.z);
      if (speed > KNOCK_SPEED_THRESHOLD) {
        if (!state.knocked) useLiveStore.getState().markBaleKnocked(bale.id);
        state.knocked = true;
        logOnce(`⚠️ crane clipped bale #${bale.id} (not the target) - ${speed.toFixed(2)} m/s`);
      }
    }

    if (t.y < TRUCK.bedTopY - 0.5 && !state.fallen) {
      state.fallen = true;
      useLiveStore.getState().markBaleFallen(bale.id);
      logOnce(`⚠️ bale #${bale.id} fell off the truck bed`);
    }
  });

  return (
    <>
      <RigidBody
        ref={body}
        type={isPinned ? "kinematicPosition" : "dynamic"}
        colliders={false}
        position={bale.position}
        friction={1.2}
        restitution={0}
        linearDamping={0.6}
        angularDamping={0.6}
        onCollisionEnter={({ other }) => {
          if (isTarget) return; // deliberate contact with your own target isn't a mistake
          if (!isPhysicsDetectionArmed()) return; // still settling from spawn
          if (other.rigidBodyObject?.name === "crane-danger-zone") {
            useLiveStore.getState().markBaleKnocked(bale.id);
            logOnce(`⚠️ crane hit bale #${bale.id} while targeting #${targetBaleId ?? "none"}`);
          }
        }}
      >
        <CuboidCollider args={[BALE_HX, BALE_HY, BALE_HZ]} density={150} />
      </RigidBody>
      {jointOffset && <BaleJoint baleRef={body} offset={jointOffset} />}
    </>
  );
}

export default function PhysicsBales() {
  const bales = useLiveStore((s) => s.bales);
  const unpicked = bales.filter((b) => !b.picked);

  useEffect(() => {
    armPhysicsDetection();
  }, []);

  return (
    <Physics gravity={[0, -9.81, 0]} colliders={false}>
      {/* factory floor - lets a badly-knocked bale actually land somewhere instead of falling forever */}
      <RigidBody type="fixed" colliders={false} position={[0, -0.05, 0]}>
        <CuboidCollider args={[15, 0.05, 15]} />
      </RigidBody>
      {/* truck bed floor */}
      <RigidBody type="fixed" colliders={false} position={TRUCK.bedFloor.center}>
        <CuboidCollider
          args={TRUCK.bedFloor.size.map((s) => s / 2) as [number, number, number]}
        />
      </RigidBody>
      {/* back wall (cab side) - keeps resting bales from sliding into the cab.
          Deliberately no side walls / no wall at the open end (bedMaxZ) - a
          bale knocked sideways or off the open end SHOULD be able to fall,
          that's the failure signal this is here to catch. */}
      <RigidBody type="fixed" colliders={false} position={TRUCK.wallBack.center}>
        <CuboidCollider
          args={TRUCK.wallBack.size.map((s) => s / 2) as [number, number, number]}
        />
      </RigidBody>
      <DangerZone />
      {unpicked.map((b) => (
        <PhysicsBale key={b.id} bale={b} />
      ))}
    </Physics>
  );
}
