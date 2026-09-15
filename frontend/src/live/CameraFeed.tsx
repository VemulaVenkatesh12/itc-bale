// A fixed virtual camera (Camera_Top or Camera_Back, matching
// blender/factory.blend's placement) rendering the live scene, which
// periodically captures its own rendered frame and sends it to the backend
// for real inference - this is the actual CV loop, not a replay.
//
// Camera_Top's captured frames drive the actual closed-loop control: they go
// to POST /api/control_tick (backend/app/live_controller.py), which
// re-detects, re-confirms the target bale against LIVE vision every tick,
// and returns the relay decision + setpoint - the frontend just applies it
// (see controller.ts). Camera_Back's detections (bale/crane_hook/
// crane_spike) go to the plain /api/detect and are shown as a live overlay
// only - proving the model tracks the gripper, not driving control.

import { useCallback, useEffect, useRef, useState } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import * as THREE from "three";
import FactoryContents from "./FactoryContents";
import { CAMERA_TOP, CAMERA_BACK, horizontalFovDeg, verticalFovDeg } from "./factoryLayout";
import { craneRef, commandedRef, useLiveStore, type LivePhase } from "./liveStore";
import { detectFrame, controlTick } from "../lib/api";
import type { Detection, DetectResponse, ControlTickResponse } from "../lib/types";

const CLASS_COLOR: Record<number, string> = { 0: "#51cf66", 1: "#ff922b", 2: "#ff6b6b" };
const CLASS_LABEL: Record<number, string> = { 0: "bale", 1: "hook", 2: "spike" };
// Camera_Top looks straight down; its default (0,1,0) up would be parallel to
// the view direction (an undefined-roll case for lookAt()) - point "screen
// up" toward -z (world) instead, an arbitrary but fixed, well-defined choice.
// Also mirrored server-side in backend/app/live_controller.py's
// CAMERA_TOP_UP for its own unprojection - keep both in sync.
const TOP_CAMERA_UP = new THREE.Vector3(0, 0, -1);

interface OverlayBox {
  key: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  color: string;
  label: string;
}

function buildOverlay(detections: Detection[], frameW: number, frameH: number, panelW: number, panelH: number): OverlayBox[] {
  const sx = panelW / frameW;
  const sy = panelH / frameH;
  return detections.map((d, i) => ({
    key: `${d.id}-${i}`,
    x1: d.bbox.x1 * sx,
    y1: d.bbox.y1 * sy,
    x2: d.bbox.x2 * sx,
    y2: d.bbox.y2 * sy,
    color: CLASS_COLOR[d.class_id] ?? "#ffffff",
    label: `${CLASS_LABEL[d.class_id] ?? "?"} ${d.confidence.toFixed(2)}`,
  }));
}

function FixedCamera({
  position,
  lookAt,
  fov,
  up,
}: {
  position: THREE.Vector3;
  lookAt: THREE.Vector3;
  fov: number;
  /** Camera.lookAt() is degenerate (undefined roll) when the view direction is
   *  parallel to the up vector - Camera_Top looks straight down, so its
   *  default (0,1,0) up would hit exactly that case. Pass an explicit,
   *  non-parallel up for any near-vertical camera. */
  up?: THREE.Vector3;
}) {
  const { camera, size } = useThree();
  useEffect(() => {
    camera.position.copy(position);
    if (up) camera.up.copy(up);
    camera.lookAt(lookAt);
    if (camera instanceof THREE.PerspectiveCamera) {
      camera.fov = fov;
      camera.aspect = size.width / size.height;
      camera.updateProjectionMatrix();
    }
  }, [camera, position, lookAt, fov, size, up]);
  return null;
}

/** Plain display-only detection loop (Camera_Back): captures, posts to
 *  /api/detect, draws the overlay. Never touches control state. */
function Detector({
  threshold,
  intervalMs,
  onResult,
}: {
  threshold: number;
  intervalMs: number;
  onResult: (resp: DetectResponse) => void;
}) {
  const { gl } = useThree();
  const busy = useRef(false);

  useEffect(() => {
    const id = setInterval(() => {
      if (busy.current) return;
      busy.current = true;
      gl.domElement.toBlob(
        (blob) => {
          if (!blob) {
            busy.current = false;
            return;
          }
          detectFrame(blob, threshold)
            .then(onResult)
            .catch((e) => console.error("[detect]", e))
            .finally(() => {
              busy.current = false;
            });
        },
        "image/jpeg",
        0.85,
      );
    }, intervalMs);
    return () => clearInterval(id);
  }, [gl, threshold, intervalMs, onResult]);

  return null;
}

/** The actual control loop (Camera_Top): captures, posts to
 *  /api/control_tick along with the crane's self-reported current pose, and
 *  applies the returned relay decision + setpoint - see controller.ts for
 *  the local drive loop that acts on what this writes. */
function ControlDetector({
  threshold,
  intervalMs,
  onResult,
}: {
  threshold: number;
  intervalMs: number;
  onResult: (resp: ControlTickResponse) => void;
}) {
  const { gl } = useThree();
  const busy = useRef(false);
  const phaseStartMs = useRef(performance.now());
  const lastPhase = useRef<string | null>(null);

  useEffect(() => {
    const id = setInterval(() => {
      if (busy.current) return;
      busy.current = true;
      gl.domElement.toBlob(
        (blob) => {
          if (!blob) {
            busy.current = false;
            return;
          }
          const store = useLiveStore.getState();
          const bales = store.bales;
          const pickedBaleIds = bales.filter((b) => b.picked).map((b) => b.id);
          if (lastPhase.current === null) lastPhase.current = store.phase;
          const elapsedInPhaseS = (performance.now() - phaseStartMs.current) / 1000;

          controlTick(blob, {
            craneX: craneRef.position.x,
            craneY: craneRef.position.y,
            craneZ: craneRef.position.z,
            phase: store.phase,
            targetBaleId: store.targetBaleId,
            elapsedInPhaseS,
            pickedBaleIds,
            threshold,
          })
            .then((resp) => {
              if (resp.phase !== lastPhase.current) {
                lastPhase.current = resp.phase;
                phaseStartMs.current = performance.now();
              }
              onResult(resp);
            })
            .catch((e) => console.error("[control_tick]", e))
            .finally(() => {
              busy.current = false;
            });
        },
        "image/jpeg",
        0.85,
      );
    }, intervalMs);
    return () => clearInterval(id);
  }, [gl, threshold, intervalMs, onResult]);

  return null;
}

export default function CameraFeed({
  which,
  label,
  panelWidth = 320,
  panelHeight = 180,
  showOverlayCaption,
}: {
  which: "top" | "back";
  label: string;
  panelWidth?: number;
  panelHeight?: number;
  showOverlayCaption?: string;
}) {
  const camDef = which === "top" ? CAMERA_TOP : CAMERA_BACK;
  const threshold = useLiveStore((s) => s.threshold);
  const detectIntervalMs = useLiveStore((s) => s.detectIntervalMs);
  const running = useLiveStore((s) => s.running);
  const setBackDetections = useLiveStore((s) => s.setBackDetections);
  const [overlay, setOverlay] = useState<OverlayBox[]>([]);

  const hFov = horizontalFovDeg(camDef.lensMm);
  const vFov = verticalFovDeg(hFov, panelWidth / panelHeight);

  const handleBackResult = useCallback(
    (resp: DetectResponse) => {
      setOverlay(buildOverlay(resp.detections, resp.width, resp.height, panelWidth, panelHeight));
      setBackDetections(resp.detections);
    },
    [panelWidth, panelHeight, setBackDetections],
  );

  const handleControlResult = useCallback(
    (resp: ControlTickResponse) => {
      setOverlay(buildOverlay(resp.detections, resp.width, resp.height, panelWidth, panelHeight));

      const store = useLiveStore.getState();
      if (resp.phase !== store.phase) store.setPhase(resp.phase as LivePhase, resp.log ?? undefined);
      else if (resp.log) store.pushLog(resp.log);
      store.setActiveRelays(new Set(resp.active_relays));
      store.setPositionSource(resp.position_source);
      if (resp.target_bale_id !== store.targetBaleId) store.setTargetBaleId(resp.target_bale_id);
      commandedRef.target.set(resp.target_x, resp.target_y, resp.target_z);
      if (resp.mark_picked_bale_id != null) store.markBalePicked(resp.mark_picked_bale_id);
    },
    [panelWidth, panelHeight],
  );

  return (
    <div style={{ position: "relative", width: panelWidth, height: panelHeight }}>
      <Canvas
        // alpha:false - an alpha-blending canvas lets the wrapping div's CSS
        // background (#12151a, not true black) bleed through at anti-aliased
        // edges, which would otherwise slightly corrupt matte mode's
        // exact-color bbox scan (matteMode.ts) right where it matters most -
        // the tracked object's own silhouette edge.
        gl={{ preserveDrawingBuffer: true, alpha: false }}
        dpr={1}
        camera={{ fov: vFov, near: 0.1, far: 200 }}
        style={{ width: "100%", height: "100%", background: "#12151a" }}
      >
        <FixedCamera
          position={camDef.position}
          lookAt={camDef.lookAt}
          fov={vFov}
          up={which === "top" ? TOP_CAMERA_UP : undefined}
        />
        <FactoryContents />
        {running && which === "top" && (
          <ControlDetector threshold={threshold} intervalMs={detectIntervalMs} onResult={handleControlResult} />
        )}
        {running && which === "back" && (
          <Detector threshold={threshold} intervalMs={detectIntervalMs} onResult={handleBackResult} />
        )}
      </Canvas>
      <svg
        viewBox={`0 0 ${panelWidth} ${panelHeight}`}
        style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
      >
        {overlay.map((b) => (
          <g key={b.key}>
            <rect
              x={b.x1}
              y={b.y1}
              width={b.x2 - b.x1}
              height={b.y2 - b.y1}
              fill="none"
              stroke={b.color}
              strokeWidth={2}
            />
            <text x={b.x1 + 2} y={Math.max(10, b.y1 - 4)} fill={b.color} fontSize={10}>
              {b.label}
            </text>
          </g>
        ))}
      </svg>
      <div className="feed-panel-label">{label}</div>
      {showOverlayCaption && <div className="feed-panel-caption">{showOverlayCaption}</div>}
    </div>
  );
}
