// In-app copy of the SF-8DR crane remote's own button panel (the page at
// 192.168.4.1). Clickable UP/DOWN/EAST/WEST/SOUTH/NORTH/START/LIGHT + STOP
// that drive the REAL physical remote via /api/crane_remote/manual - manual
// override / signal-verification, independent of the sim control loop.
// Polls /api/crane_remote/status once a second to mirror the remote's live
// relay state (and its Modbus/GPIO backend + connection).
import { useCallback, useEffect, useState } from "react";
import { getCraneRemoteStatus, manualCraneChannel, stopPhysicalRelays, type CraneRemoteStatus } from "../lib/api";

const CHANNELS: { ch: number; label: string; group: string }[] = [
  { ch: 0, label: "UP", group: "Hoist" },
  { ch: 1, label: "DOWN", group: "Hoist" },
  { ch: 2, label: "EAST", group: "Long Travel (E/W)" },
  { ch: 3, label: "WEST", group: "Long Travel (E/W)" },
  { ch: 4, label: "SOUTH", group: "Cross Travel (S/N)" },
  { ch: 5, label: "NORTH", group: "Cross Travel (S/N)" },
  { ch: 6, label: "START", group: "Other" },
  { ch: 7, label: "LIGHT", group: "Other" },
];
const GROUPS = ["Hoist", "Long Travel (E/W)", "Cross Travel (S/N)", "Other"];

export default function CraneRemotePanel() {
  const [status, setStatus] = useState<CraneRemoteStatus | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    setStatus(await getCraneRemoteStatus());
  }, []);

  useEffect(() => {
    void refresh();
    const id = setInterval(refresh, 1000);
    return () => clearInterval(id);
  }, [refresh]);

  const relays = status?.remote?.relays ?? [];
  const online = !!status?.online && !!status?.remote;
  const backend = status?.remote?.backend ?? "?";

  async function toggle(ch: number) {
    if (busy) return;
    setBusy(true);
    const next = !relays[ch];
    const fresh = await manualCraneChannel(ch, next);
    if (fresh) setStatus((s) => (s ? { ...s, remote: fresh } : s));
    setBusy(false);
  }

  async function stop() {
    if (busy) return;
    setBusy(true);
    await stopPhysicalRelays();
    await refresh();
    setBusy(false);
  }

  return (
    <div className="controller-panel crane-remote-panel">
      <div className="controller-title">Crane Remote — SF-8DR (physical · 192.168.4.1)</div>
      <div className={`crane-remote-conn ${online ? "ok" : "err"}`}>
        {online ? `connected · backend: ${backend}` : "offline / bridge disabled"}
      </div>

      {GROUPS.map((g) => (
        <div className="relay-group" key={g}>
          <div className="relay-group-label">{g}</div>
          <div className="crane-remote-btns">
            {CHANNELS.filter((c) => c.group === g).map((c) => (
              <button
                key={c.ch}
                className={`crane-remote-btn ${relays[c.ch] ? "on" : "off"}`}
                disabled={!online || busy}
                onClick={() => toggle(c.ch)}
              >
                {c.label}
                <span>{relays[c.ch] ? "ON" : "OFF"}</span>
              </button>
            ))}
          </div>
        </div>
      ))}

      <button className="crane-remote-btn stop" disabled={!online || busy} onClick={stop}>
        STOP (release all)
      </button>
      <p className="crane-remote-note">
        1S (low speed) only. UP/DOWN, EAST/WEST, SOUTH/NORTH interlock — activating one releases its opposite.
        Buttons drive the real relay; the sim also drives it automatically while running.
      </p>
    </div>
  );
}
