import { RELAY_ORDER, type RelayEvent } from "../lib/types";
import { activeRelays } from "../lib/player";

interface Props {
  timeline: RelayEvent[];
  tMs: number;
}

/** Visual replica of the SF-8DR terminal legend — one LED per relay,
 * lit exactly when that relay is energized in the plan at time tMs. This is
 * the "industrial remote controller simulator" driving the animated crane:
 * swap this component's LED source for real GPIO/serial reads and the same
 * layout becomes a live panel for the actual controller. */
export default function ControllerPanel({ timeline, tMs }: Props) {
  const active = activeRelays(timeline, tMs);
  const groups = ["N/S", "E/W", "HOIST", "EJECT"];

  return (
    <div className="controller-panel">
      <div className="controller-title">INDUSTRIAL REMOTE CONTROL — SF-8DR (simulated)</div>
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
