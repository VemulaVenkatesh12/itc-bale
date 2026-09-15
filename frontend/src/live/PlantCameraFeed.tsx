// One physical plant camera (Hikvision cam101 @ 192.168.1.101 or cam102 @
// 192.168.1.102), shown as a live MJPEG <img> fed by the backend's RTSP->
// MJPEG proxy (backend/app/camera_feed.py -> /api/cameras/<id>/stream), with
// a live RF-DETR detection overlay on top.
//
// The overlay polls GET /api/cameras/<id>/detect (the 40-epoch real-footage
// model, backend/app/model_service.py) a couple of times a second and draws
// the boxes it gets back - bale / crane_hook / crane_spike. This is a
// DISPLAY-ONLY view of what the detector sees on the REAL cameras; it does
// not touch the control loop (that's still Camera_Top in CameraFeed.tsx).

import { useEffect, useRef, useState } from "react";
import { cameraDetect, cameraStreamUrl, getCameras, type CameraStatus } from "../lib/api";
import type { Detection } from "../lib/types";

const CLASS_COLOR: Record<number, string> = { 0: "#51cf66", 1: "#ff922b", 2: "#ff6b6b" };
const CLASS_LABEL: Record<number, string> = { 0: "bale", 1: "hook", 2: "spike" };

// Independent of useLiveStore's threshold (which also gates /api/control_tick
// and therefore real relay decisions - never lower that just to make this
// display prettier). This panel is display-only, so it can afford to show
// weaker hits: a parked/retracted hook squeezed between two loads is
// genuinely occluded and scores as low as ~0.05-0.2 (see the 2026-09-11
// investigation - the 40-epoch dataset has little coverage of that pose,
// only the hook-descending-over-an-open-bed one).
//
// FETCH_THRESHOLD is the floor sent to the backend - low enough to catch
// almost any real hook signal. DISPLAY_MIN is then applied PER CLASS
// client-side (see buildOverlay): bale/spike still need real confidence
// (0.3) to show, or the truck's tightly-packed stack floods the panel with
// low-confidence fragment boxes (measured: ~250+ at the fetch floor). Hook
// has NO display floor - every hook score the model returns is shown and
// labeled with its real number, however low, because "check the hook
// detection" means seeing what it actually found, not a filtered version of
// it. LOW_CONFIDENCE_BELOW controls the dashed/dimmed "shaky" styling only.
const FETCH_THRESHOLD = 0.05;
const DISPLAY_MIN: Record<number, number> = { 0: 0.3, 1: 0.0, 2: 0.3 };
const LOW_CONFIDENCE_BELOW = 0.3;

interface OverlayBox {
  key: string;
  x: number;
  y: number;
  w: number;
  h: number;
  color: string;
  label: string;
  tentative: boolean;
}

function buildOverlay(
  detections: Detection[],
  frameW: number,
  frameH: number,
  panelW: number,
  panelH: number,
): OverlayBox[] {
  const sx = panelW / frameW;
  const sy = panelH / frameH;
  return detections
    .filter((d) => d.confidence >= (DISPLAY_MIN[d.class_id] ?? 0.3))
    .map((d, i) => ({
    key: `${d.id}-${i}`,
    x: d.bbox.x1 * sx,
    y: d.bbox.y1 * sy,
    w: (d.bbox.x2 - d.bbox.x1) * sx,
    h: (d.bbox.y2 - d.bbox.y1) * sy,
    color: CLASS_COLOR[d.class_id] ?? "#ffffff",
    tentative: d.confidence < LOW_CONFIDENCE_BELOW,
    label: `${CLASS_LABEL[d.class_id] ?? "?"} ${d.confidence.toFixed(2)}`,
  }));
}

export default function PlantCameraFeed({
  camId,
  label,
  panelWidth = 360,
  panelHeight = 203, // 16:9, matches the backend's 1280x720 downscaled frame
}: {
  camId: string;
  label: string;
  panelWidth?: number;
  panelHeight?: number;
}) {
  const [errored, setErrored] = useState(false);
  const [status, setStatus] = useState<CameraStatus | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [detectOn, setDetectOn] = useState(true);
  const [overlay, setOverlay] = useState<OverlayBox[]>([]);
  const [counts, setCounts] = useState<{ bale: number; hook: number; spike: number }>({
    bale: 0,
    hook: 0,
    spike: 0,
  });
  const busy = useRef(false);

  const online = status?.connected ?? false;

  // Poll the backend's camera status so the badge reflects the real RTSP
  // connection, not just whether the <img> happened to load.
  useEffect(() => {
    let alive = true;
    const tick = () => {
      getCameras().then((cams) => {
        if (alive) setStatus(cams.find((c) => c.id === camId) ?? null);
      });
    };
    tick();
    const id = setInterval(tick, 5000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [camId]);

  // Retry the stream a few seconds after it errors.
  useEffect(() => {
    if (!errored) return;
    const id = setTimeout(() => {
      setErrored(false);
      setAttempt((a) => a + 1);
    }, 4000);
    return () => clearTimeout(id);
  }, [errored]);

  // Detection loop: poll /api/cameras/<id>/detect while the panel is on,
  // the camera is connected and the stream <img> is showing. Non-overlapping
  // (a slow inference just lowers the effective rate).
  useEffect(() => {
    if (!detectOn || !online || errored) {
      setOverlay([]);
      setCounts({ bale: 0, hook: 0, spike: 0 });
      return;
    }
    let alive = true;
    const run = () => {
      if (busy.current || !alive) return;
      busy.current = true;
      cameraDetect(camId, FETCH_THRESHOLD)
        .then((resp) => {
          if (!alive) return;
          const shown = resp.detections.filter((d) => d.confidence >= (DISPLAY_MIN[d.class_id] ?? 0.3));
          setOverlay(buildOverlay(resp.detections, resp.width, resp.height, panelWidth, panelHeight));
          setCounts({
            bale: shown.filter((d) => d.class_id === 0).length,
            hook: shown.filter((d) => d.class_id === 1).length,
            spike: shown.filter((d) => d.class_id === 2).length,
          });
        })
        .catch((e) => console.error("[cam-detect]", camId, e))
        .finally(() => {
          busy.current = false;
        });
    };
    run();
    const id = setInterval(run, 1200);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [camId, detectOn, online, errored, panelWidth, panelHeight]);

  const src = `${cameraStreamUrl(camId)}?a=${attempt}`;

  return (
    <div style={{ position: "relative", width: panelWidth, height: panelHeight, background: "#12151a" }}>
      {!errored && (
        <img
          src={src}
          alt={label}
          width={panelWidth}
          height={panelHeight}
          style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
          onError={() => setErrored(true)}
        />
      )}
      {errored && (
        <div className="feed-panel-offline">
          <div>no signal</div>
          <div style={{ fontSize: "0.6rem", opacity: 0.7 }}>retrying…</div>
        </div>
      )}

      {!errored && overlay.length > 0 && (
        <svg
          viewBox={`0 0 ${panelWidth} ${panelHeight}`}
          style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
        >
          {overlay.map((b) => (
            <g key={b.key}>
              <rect
                x={b.x} y={b.y} width={b.w} height={b.h} fill="none" stroke={b.color}
                strokeWidth={2} strokeDasharray={b.tentative ? "4 3" : undefined}
                opacity={b.tentative ? 0.65 : 1}
              />
              <text x={b.x + 2} y={Math.max(10, b.y - 4)} fill={b.color} fontSize={10} opacity={b.tentative ? 0.65 : 1}>
                {b.label}{b.tentative ? "?" : ""}
              </text>
            </g>
          ))}
        </svg>
      )}

      <div className="feed-panel-label">{label}</div>
      <div className={`feed-panel-badge feed-panel-badge--${online ? "online" : "offline"}`}>
        {online ? "● live" : "● offline"}
      </div>

      <label className="feed-panel-detect-toggle" title="Run the detector on this feed">
        <input type="checkbox" checked={detectOn} onChange={(e) => setDetectOn(e.target.checked)} />
        detect
      </label>

      {detectOn && online && !errored && (
        <div className="feed-panel-detect-counts">
          <span style={{ color: CLASS_COLOR[0] }}>{counts.bale} bale</span>
          <span style={{ color: CLASS_COLOR[1] }}>{counts.hook} hook</span>
          {counts.spike > 0 && <span style={{ color: CLASS_COLOR[2] }}>{counts.spike} spike</span>}
        </div>
      )}
    </div>
  );
}
