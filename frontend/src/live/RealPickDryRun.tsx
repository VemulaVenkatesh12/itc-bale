// "Start" for the REAL camera - same experience as the 3D sim's Start button
// (CameraFeed.tsx's ControlDetector: poll a frame, decide, show it, repeat),
// but pointed at live plant video instead of the simulated canvas.
//
// DRY-RUN BY DESIGN: this NEVER calls crane_remote / fires a real relay.
// /api/control_tick already forwards whatever it decides straight to the
// physical crane remote - reusing it here, before this camera is calibrated,
// would mean sending real relay signals computed from the SIMULATED
// camera's baked-in geometry onto real machinery. Until real pixel->world
// calibration exists (see PLANT_VISIT_CHECKLIST.md), this only SHOWS what
// it would decide each tick: next target bale, and whether the conveyor
// gate (backend/app/conveyor_gate.py) would hold the signal - both computed
// live, for real, against real video. Nothing here is simulated data.
//
// The one exception to "dry-run only": the vision safety boundary
// (backend/app/boundary_guard.py, the senior's suggestion 2026-09-12) DOES
// act for real - its background watchdog calls crane_remote.stop_all() the
// instant it sees the hook outside the keep-in region, independent of this
// panel being open at all. This panel just shows what that watchdog is
// seeing; it doesn't control it beyond the pause/resume toggle below.
import { useEffect, useRef, useState } from "react";
import {
  cameraStreamBoundaryUrl, cameraStreamPickOrderUrl, getBoundaryGuardStatus,
  getConveyorStatus, getCrossViewStatus, getPickOrder, getScaleReference, getServoDecision,
  resetPickOrder, setBoundaryGuardEnabled, setScaleReference,
} from "../lib/api";
import type {
  BoundaryGuardStatus, ConveyorStatus, CrossViewStatus, PickOrderResponse, ScaleReferenceStatus,
  ServoDecision,
} from "../lib/types";

const OTHER_CAM: Record<string, string> = { cam101: "cam102", cam102: "cam101" };

const POLL_MS = 1200; // matches PlantCameraFeed's live detect cadence
const THRESHOLD = 0.3;

const BOUNDARY_COLOR: Record<string, string> = {
  inside: "#51cf66", near_edge: "#ff922b", outside: "#ff6b6b", no_hook: "#888",
};

export default function RealPickDryRun({ camId = "cam101", label = "Real pick order (dry-run - no relay output)" }: {
  camId?: string;
  label?: string;
}) {
  const [running, setRunning] = useState(false);
  const [view, setView] = useState<"pick_order" | "boundary">("pick_order");
  const [picks, setPicks] = useState<PickOrderResponse | null>(null);
  const [conveyor, setConveyor] = useState<ConveyorStatus | null>(null);
  const [boundary, setBoundary] = useState<BoundaryGuardStatus | null>(null);
  const [servo, setServo] = useState<ServoDecision | null>(null);
  const [crossView, setCrossView] = useState<CrossViewStatus | null>(null);
  const [scaleRef, setScaleRef] = useState<ScaleReferenceStatus | null>(null);
  const [refDistanceInput, setRefDistanceInput] = useState("1.3");
  const [refCapturing, setRefCapturing] = useState(false);
  const [refMsg, setRefMsg] = useState<string | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const [tick, setTick] = useState(0);
  const busy = useRef(false);

  useEffect(() => {
    getScaleReference(camId).then(setScaleRef).catch(() => {});
  }, [camId]);

  const handleCaptureScaleRef = async () => {
    const distanceM = parseFloat(refDistanceInput);
    if (!Number.isFinite(distanceM) || distanceM <= 0) {
      setRefMsg("enter a valid distance in metres first");
      return;
    }
    setRefCapturing(true);
    setRefMsg(null);
    try {
      const ref = await setScaleReference(camId, distanceM, THRESHOLD);
      setScaleRef(ref);
      setRefMsg(`scale set: ${ref.pixels_per_metre?.toFixed(1)} px/m (from ${ref.ref_pixel_distance?.toFixed(0)}px = ${distanceM}m)`);
    } catch (e) {
      setRefMsg(`capture failed: ${e}`);
    } finally {
      setRefCapturing(false);
    }
  };

  useEffect(() => {
    if (!running) return;
    let alive = true;
    const run = async () => {
      if (busy.current || !alive) return;
      busy.current = true;
      try {
        const [po, cs, bg, sv, cv] = await Promise.all([
          getPickOrder(camId, THRESHOLD),
          getConveyorStatus(camId, THRESHOLD),
          getBoundaryGuardStatus(),
          getServoDecision(camId, THRESHOLD).catch(() => null),
          getCrossViewStatus(camId, OTHER_CAM[camId] ?? camId, THRESHOLD).catch(() => null),
        ]);
        if (!alive) return;
        setPicks(po);
        setConveyor(cs);
        setBoundary(bg);
        setServo(sv);
        setCrossView(cv);
        setTick((t) => t + 1);
        const camBoundary = bg.cameras[camId];
        const next = po.picks.find((p) => p.number === 1);
        const line = cs.roi_configured && !cs.ok_to_pick
          ? `HOLD - conveyor full (${cs.count}/${cs.full_threshold}) - no pick signal this tick`
          : camBoundary?.status === "outside"
            ? "BOUNDARY VIOLATION - watchdog is stopping the crane"
            : sv?.status === "nudge" && sv.image_direction
              ? `servo: ${sv.image_direction}`
              : sv?.status === "aligned"
                ? "servo: aligned with target"
                : next
                  ? `would pick #${next.number} (layer ${next.layer}, conf ${next.confidence.toFixed(2)}) - ${po.picks.length} bales numbered`
                  : "no bales numbered this tick";
        setLog((l) => [line, ...l].slice(0, 40));
      } catch (e) {
        setLog((l) => [`error: ${e}`, ...l].slice(0, 40));
      } finally {
        busy.current = false;
      }
    };
    run();
    const id = setInterval(run, POLL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [running, camId]);

  const handleStart = () => {
    setLog([]);
    setTick(0);
    setRunning(true);
  };
  const handleStop = () => setRunning(false);
  const handleReset = async () => {
    setRunning(false);
    setPicks(null);
    setConveyor(null);
    setLog([]);
    await resetPickOrder(camId);
  };

  const holding = conveyor?.roi_configured && conveyor.ok_to_pick === false;
  const camBoundary = boundary?.cameras[camId];
  const boundaryColor = camBoundary ? BOUNDARY_COLOR[camBoundary.status] : "#888";

  return (
    <div className="real-pick-dry-run">
      <div className="real-pick-dry-run-video" style={{ position: "relative", width: 480, height: 270, background: "#12151a" }}>
        {running ? (
          <img
            key={`${view}-${tick > 0 ? "on" : "off"}`}
            src={view === "pick_order" ? cameraStreamPickOrderUrl(camId, THRESHOLD) : cameraStreamBoundaryUrl(camId, THRESHOLD)}
            alt={label}
            style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
          />
        ) : (
          <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "#666", fontSize: "0.8rem" }}>
            press Start to watch it decide, live
          </div>
        )}
        {holding && (
          <div style={{ position: "absolute", top: 6, left: 6, background: "#ff6b6b", color: "#000", fontWeight: 700, fontSize: "0.7rem", padding: "3px 8px", borderRadius: 4 }}>
            HOLDING - conveyor full
          </div>
        )}
        <div className="feed-panel-detect-toggle" style={{ position: "absolute", bottom: 6, right: 6 }}>
          <button
            style={{ fontSize: "0.65rem", padding: "3px 8px" }}
            onClick={() => setView(view === "pick_order" ? "boundary" : "pick_order")}
          >
            view: {view === "pick_order" ? "pick order" : "safety boundary"} (click to switch)
          </button>
        </div>
      </div>

      <div className="real-pick-dry-run-panel">
        <div style={{ fontSize: "0.7rem", color: "var(--text-dim)", marginBottom: 6 }}>{label}</div>
        <div className="real-pick-dry-run-buttons">
          <button className="simulate-btn" onClick={handleStart} disabled={running}>Start</button>
          <button onClick={handleStop} disabled={!running}>Stop</button>
          <button onClick={handleReset}>Reset numbering</button>
        </div>

        {conveyor && (
          <div style={{ fontSize: "0.75rem", marginTop: 8 }}>
            conveyor: {conveyor.roi_configured
              ? `${conveyor.count}/${conveyor.full_threshold} - ${conveyor.full ? "FULL" : "clear"}`
              : "no ROI traced for this camera"}
          </div>
        )}
        {picks && (
          <div style={{ fontSize: "0.75rem", marginTop: 4 }}>
            {picks.picks.length} bale(s) numbered, next = {picks.picks.find((p) => p.number === 1)?.number ?? "-"}
          </div>
        )}
        {crossView && crossView.agree !== null && (
          <div style={{ fontSize: "0.75rem", marginTop: 4, color: crossView.agree ? "#51cf66" : "#ff922b" }}>
            {crossView.agree ? "views agree" : "views disagree"} - {crossView.cam_a}: {crossView.count_a}, {crossView.cam_b}: {crossView.count_b}
          </div>
        )}

        <div style={{ fontSize: "0.75rem", marginTop: 8, paddingTop: 8, borderTop: "1px solid var(--border)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ width: 10, height: 10, borderRadius: "50%", background: boundaryColor, display: "inline-block" }} />
            <strong>safety boundary:</strong> {camBoundary?.status ?? "-"}
            {boundary && (
              <button
                style={{ fontSize: "0.65rem", padding: "2px 6px", marginLeft: "auto" }}
                onClick={() => setBoundaryGuardEnabled(!boundary.enabled)}
              >
                watchdog: {boundary.enabled ? "ON (click to pause)" : "PAUSED (click to resume)"}
              </button>
            )}
          </div>
          {boundary && boundary.violation_count > 0 && (
            <div style={{ color: "#ff6b6b", marginTop: 4 }}>
              {boundary.violation_count} violation(s) since backend start - crane was auto-stopped each time
            </div>
          )}
        </div>

        <div style={{ fontSize: "0.75rem", marginTop: 8, paddingTop: 8, borderTop: "1px solid var(--border)" }}>
          <strong>scale reference:</strong>{" "}
          {scaleRef?.configured
            ? `${scaleRef.pixels_per_metre?.toFixed(1)} px/m (set ${scaleRef.set_at} from ${scaleRef.ref_distance_m}m)`
            : "not set - direction only, no real distance yet"}
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 4 }}>
            <span>hook -&gt; target is</span>
            <input
              type="text" inputMode="decimal" value={refDistanceInput}
              onChange={(e) => setRefDistanceInput(e.target.value)}
              style={{ width: 48, fontSize: "0.75rem" }}
            />
            <span>m right now</span>
            <button
              style={{ fontSize: "0.65rem", padding: "2px 8px" }}
              onClick={handleCaptureScaleRef} disabled={refCapturing}
            >
              {refCapturing ? "capturing..." : "capture now"}
            </button>
          </div>
          {refMsg && <div style={{ marginTop: 4, color: refMsg.startsWith("capture failed") ? "#ff6b6b" : "#51cf66" }}>{refMsg}</div>}
          {servo?.distance_m != null && (
            <div style={{ marginTop: 4 }}>current hook-target distance: ~{servo.distance_m.toFixed(2)}m</div>
          )}
        </div>

        <div className="real-pick-dry-run-log">
          {log.length === 0 ? (
            <div className="live-log-empty">Start to see live per-tick decisions.</div>
          ) : (
            log.map((line, i) => <div key={i}>{line}</div>)
          )}
        </div>
      </div>
    </div>
  );
}
