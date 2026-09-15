// Same visual layout as components/ControllerPanel.tsx, but fed directly
// from a live Set<string> (liveStore.activeRelays) instead of scrubbing a
// precomputed timeline at a point in time.
import { RELAY_ORDER } from "../lib/types";

export default function LiveRelayPanel({ active }: { active: Set<string> }) {
  const groups = ["N/S", "E/W", "HOIST", "EJECT"];
  return (
    <div className="controller-panel">
      <div className="controller-title">INDUSTRIAL REMOTE CONTROL — SF-8DR (live)</div>
      {groups.map((g) => (
        <div className="relay-group" key={g}>
          <div className="relay-group-label">{g}</div>
          <div className="relay-leds">
            {RELAY_ORDER.filter((r) => r.group === g).map((r) => (
              <div key={r.relay} className={`led ${active.has(r.relay) ? "on" : ""}`}>
                <span className="led-dot" />
                <span className="led-label">{r.label}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
