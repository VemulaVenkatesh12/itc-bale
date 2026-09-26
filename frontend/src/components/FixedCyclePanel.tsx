// POC "Start" button: runs the fixed, single-target pick-and-drop cycle
// (backend/app/fixed_cycle.py) against the REAL crane. Every run first homes
// on all 3 axes (runs into the physical limit switches) so it starts from a
// known position no matter where the crane currently is, then follows a
// fixed timed sequence with one vision check (top-2-layers presence) right
// before the pick hoist-down. NOT the sim's vision-driven control loop - see
// fixed_cycle.py's module docstring for why.
import { useCallback, useEffect, useState } from "react";
import {
  getFixedCycleStatus,
  startFixedCycle,
  stopFixedCycle,
  type FixedCycleStatus,
} from "../lib/api";

export default function FixedCyclePanel() {
  const [status, setStatus] = useState<FixedCycleStatus | null>(null);
  const [startError, setStartError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    setStatus(await getFixedCycleStatus());
  }, []);

  useEffect(() => {
    void refresh();
    const id = setInterval(refresh, 1000);
    return () => clearInterval(id);
  }, [refresh]);

  async function start() {
    if (busy || status?.running) return;
    setBusy(true);
    setStartError(null);
    const res = await startFixedCycle();
    if (!res.ok) setStartError(res.reason ?? "failed to start");
    await refresh();
    setBusy(false);
  }

  async function stop() {
    if (busy) return;
    setBusy(true);
    await stopFixedCycle();
    await refresh();
    setBusy(false);
  }

  const running = !!status?.running;

  return (
    <div className="controller-panel fixed-cycle-panel">
      <div className="controller-title">Fixed POC Cycle — top 2 layers only</div>

      <button
        className={`crane-remote-btn ${running ? "on" : "off"}`}
        style={{ width: "100%", marginBottom: 8 }}
        disabled={busy || running}
        onClick={start}
      >
        {running ? "RUNNING…" : "START"}
      </button>
      <button
        className="crane-remote-btn stop"
        style={{ width: "100%" }}
        disabled={busy || !running}
        onClick={stop}
      >
        STOP (release all)
      </button>

      {startError && <p className="crane-remote-note" style={{ color: "#c0392b" }}>{startError}</p>}

      {status && (
        <p className="crane-remote-note">
          Step {status.step_index + 1}/{status.total_steps}: {status.step_label || "—"}
          {status.error && <><br /><b style={{ color: "#c0392b" }}>{status.error}</b></>}
        </p>
      )}

      <p className="crane-remote-note">
        Every run first homes on WEST / NORTH / hoist-UP limits (works from any current
        position), then a fixed timed sequence with one vision check for a real bale
        before hoisting down. Not position-steered vision — see fixed_cycle.py.
      </p>
    </div>
  );
}
