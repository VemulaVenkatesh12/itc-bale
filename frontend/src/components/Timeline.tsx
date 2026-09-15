import { useRef, type MouseEvent as ReactMouseEvent } from "react";
import { RELAY_ORDER, type RelayEvent } from "../lib/types";

interface Props {
  timeline: RelayEvent[];
  totalDurationMs: number;
  tMs: number;
  onSeek: (ms: number) => void;
  playing: boolean;
  onTogglePlay: () => void;
  speedMult: number;
  onSpeedMult: (v: number) => void;
  onReset: () => void;
}

/** Logic-analyzer-style gantt strip: one row per relay, bars show on/off
 * windows, a scrubber line shows the current playhead. Click/drag to seek. */
export default function Timeline({
  timeline, totalDurationMs, tMs, onSeek, playing, onTogglePlay, speedMult, onSpeedMult, onReset,
}: Props) {
  const trackRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);

  function seekFromEvent(e: ReactMouseEvent) {
    const el = trackRef.current;
    if (!el || totalDurationMs <= 0) return;
    const rect = el.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    onSeek(frac * totalDurationMs);
  }

  const playheadPct = totalDurationMs > 0 ? (tMs / totalDurationMs) * 100 : 0;

  return (
    <div className="timeline-panel">
      <div className="timeline-controls">
        <button onClick={onTogglePlay} disabled={totalDurationMs <= 0}>{playing ? "Pause" : "Play"}</button>
        <button onClick={onReset} disabled={totalDurationMs <= 0}>Reset</button>
        <label>
          Playback speed
          <select value={speedMult} onChange={(e) => onSpeedMult(Number(e.target.value))}>
            {[0.25, 0.5, 1, 2, 4, 8].map((v) => (
              <option key={v} value={v}>{v}x</option>
            ))}
          </select>
        </label>
        <span className="timeline-clock">{(tMs / 1000).toFixed(1)}s / {(totalDurationMs / 1000).toFixed(1)}s</span>
      </div>

      <div
        className="timeline-track"
        ref={trackRef}
        onMouseDown={(e) => { draggingRef.current = true; seekFromEvent(e); }}
        onMouseMove={(e) => { if (draggingRef.current) seekFromEvent(e); }}
        onMouseUp={() => { draggingRef.current = false; }}
        onMouseLeave={() => { draggingRef.current = false; }}
      >
        {RELAY_ORDER.map((r) => (
          <div className="timeline-row" key={r.relay}>
            <div className="timeline-row-label">{r.label}</div>
            <div className="timeline-row-bars">
              {timeline.filter((ev) => ev.relay === r.relay).map((ev, i) => (
                <div
                  key={i}
                  className="timeline-bar"
                  title={`${r.label} — ${ev.phase} (${ev.on_at_ms}-${ev.off_at_ms}ms)`}
                  style={{
                    left: `${(ev.on_at_ms / totalDurationMs) * 100}%`,
                    width: `${((ev.off_at_ms - ev.on_at_ms) / totalDurationMs) * 100}%`,
                  }}
                />
              ))}
            </div>
          </div>
        ))}
        <div className="timeline-playhead" style={{ left: `${playheadPct}%` }} />
      </div>
    </div>
  );
}
