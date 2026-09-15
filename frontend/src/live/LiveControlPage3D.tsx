// PARKED, 2026-09-11 - not currently wired into the app. This is the full
// Three.js live-3D simulation view (virtual Camera_Top/Camera_Back, the
// closed-loop Start/Stop control against the SIMULATED scene, the relay LED
// panel) that used to be the "Live 3D Control" tab's content.
//
// Nothing about the BACKEND logic this drives was touched or removed -
// controller.ts, liveStore.ts, live_controller.py's compute_tick, the relay
// bridge - all of it is exactly as it was and still works; this file is the
// only thing that stopped being rendered. The tab now shows
// LiveControlPage.tsx instead (live camera + calibration-facing tools only,
// per 2026-09-11's requirement to make that tab about the real plant, not
// the simulation).
//
// To bring the 3D sim view back: import LiveControlPage3D from this file in
// App.tsx (or wherever the tab is wired) and render it in place of/alongside
// LiveControlPage. Nothing else needs to change - this component is
// otherwise unmodified.
import "./live.css";
import MainView from "./MainView";
import CameraFeed from "./CameraFeed";
import LiveRelayPanel from "./LiveRelayPanel";
import CraneRemotePanel from "../components/CraneRemotePanel";
import { useLiveController } from "./controller";
import { useLiveStore } from "./liveStore";
import { stopPhysicalRelays } from "../lib/api";

const PHASE_LABEL: Record<string, string> = {
  idle: "Idle",
  select_target: "Waiting for a confirmed bale detection…",
  cross: "Crossing to bale (E/W)",
  descend: "Descending onto bale (N/S)",
  hoist_down_to_bale: "Lowering onto bale",
  pierce: "Piercing (spike sliding in)",
  pierce_settle: "Piercing (settling)",
  hoist_up_clear: "Lifting clear",
  drop_cross: "Crossing to drop point (E/W)",
  drop_descend: "Descending to drop point (N/S)",
  hoist_down_to_drop: "Lowering to conveyor",
  push_off: "Push-off (ejecting bale onto conveyor)",
  hoist_up_after_drop: "Lifting clear",
  return_home: "Returning home",
  done: "Cycle complete",
};

export default function LiveControlPage3D() {
  useLiveController();

  const running = useLiveStore((s) => s.running);
  const phase = useLiveStore((s) => s.phase);
  const activeRelays = useLiveStore((s) => s.activeRelays);
  const bales = useLiveStore((s) => s.bales);
  const log = useLiveStore((s) => s.log);
  const positionSource = useLiveStore((s) => s.positionSource);
  const threshold = useLiveStore((s) => s.threshold);
  const detectIntervalMs = useLiveStore((s) => s.detectIntervalMs);
  const setThreshold = useLiveStore((s) => s.setThreshold);
  const start = useLiveStore((s) => s.start);
  const stop = useLiveStore((s) => s.stop);
  const reset = useLiveStore((s) => s.reset);

  async function handleStop() {
    stop();
    // Clear the physical remote's relays when the live loop stops/resets so
    // no output is left latched (see usePhysicalRelayBridge's stop path).
    stopPhysicalRelays();
  }

  async function handleReset() {
    reset();
    stopPhysicalRelays();
  }

  const pickedCount = bales.filter((b) => b.picked).length;
  const knockedCount = bales.filter((b) => b.knocked).length;

  return (
    <div className="live-page">
      <div className="live-main">
        <div className="live-main-view">
          <MainView />
          <div className="live-status-overlay">
            <div className="live-status-phase">{PHASE_LABEL[phase] ?? phase}</div>
            <div className="live-status-count">
              {pickedCount} / {bales.length} bales picked
            </div>
            {running && (
              <div className={`live-status-position-source live-status-position-source--${positionSource}`}>
                {/* Where the LAST control_tick's (x, z) actually came from - see
                    liveStore.ts's positionSource doc comment. Not decorative:
                    "self_reported" is expected often (the hook is
                    self-occluded by the mast pole from most crane
                    positions), not just a rare fallback. */}
                position: {positionSource === "vision" ? "📷 vision-measured" : "↺ self-reported (fallback)"}
              </div>
            )}
            {knockedCount > 0 && (
              <div className="live-status-knocked">⚠️ {knockedCount} bale(s) knocked by the crane</div>
            )}
          </div>
        </div>

        <div className="live-feeds">
          <CameraFeed which="top" label="Camera_Top (drives picking — live inference)" />
          <CameraFeed
            which="back"
            label="Camera_Back (live gripper detection)"
            showOverlayCaption="shown live, not fed into control"
          />
        </div>

        <div className="live-log">
          {log.length === 0 ? (
            <div className="live-log-empty">Press Start to begin the closed-loop pick cycle.</div>
          ) : (
            log
              .slice()
              .reverse()
              .map((line, i) => <div key={i}>{line}</div>)
          )}
        </div>
      </div>

      <div className="live-side">
        <LiveRelayPanel active={activeRelays} />
        <CraneRemotePanel />

        <div className="config-panel">
          <h3>Live control</h3>
          <div className="live-buttons">
            <button className="simulate-btn" onClick={start} disabled={running}>
              Start
            </button>
            <button onClick={handleStop} disabled={!running}>
              Stop
            </button>
            <button onClick={handleReset}>Reset scene</button>
          </div>

          <h3>Detection</h3>
          <div className="config-grid">
            <label>
              Confidence threshold ({threshold.toFixed(2)})
              <input
                type="range"
                min={0.05}
                max={0.9}
                step={0.05}
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
              />
            </label>
            <p className="hint">
              Inference tick: ~{(1000 / detectIntervalMs).toFixed(1)} Hz per camera, run live against the
              backend's RF-DETR model - see the Network tab for repeated POST /api/detect calls.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
