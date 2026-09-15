import type { OrientationConfig, SpeedConfig } from "../lib/types";

interface Props {
  speed: SpeedConfig;
  onSpeed: (s: SpeedConfig) => void;
  orientation: OrientationConfig;
  onOrientation: (o: OrientationConfig) => void;
  threshold: number;
  onThreshold: (v: number) => void;
  maxBales: number | "";
  onMaxBales: (v: number | "") => void;
}

const FIELDS: { key: keyof SpeedConfig; label: string; step?: number; hint?: string }[] = [
  { key: "ew_low_px_s", label: "East/West low speed (px/s)" },
  { key: "ew_high_px_s", label: "East/West high speed (px/s)" },
  { key: "ns_low_px_s", label: "North/South low speed (px/s)" },
  { key: "ns_high_px_s", label: "North/South high speed (px/s)" },
  { key: "high_speed_min_distance_px", label: "Min distance to engage high speed (px)" },
  { key: "creep_distance_px", label: "Final creep distance at low speed (px)" },
  { key: "hoist_clear_s", label: "Hoist lift duration (s)" },
  { key: "hoist_lower_s", label: "Hoist lower duration (s)" },
  { key: "hoist_creep_s", label: "Hoist creep tail (s)" },
  { key: "pierce_dwell_s", label: "Pierce settle time (s) — no relay, mechanical" },
  { key: "push_off_s", label: "Push-off relay pulse (s)" },
  { key: "settle_s", label: "Settle pause between phases (s)" },
];

/** Physical calibration: how many image pixels the crane covers per second
 * at each speed tier, so simulated travel time matches your real crane's
 * speed relative to this image's real-world scale. */
export default function ConfigPanel({
  speed, onSpeed, orientation, onOrientation, threshold, onThreshold, maxBales, onMaxBales,
}: Props) {
  function set(key: keyof SpeedConfig, value: number) {
    onSpeed({ ...speed, [key]: value });
  }

  return (
    <div className="config-panel">
      <h3>Calibration</h3>
      <div className="config-grid">
        {FIELDS.map((f) => (
          <label key={f.key}>
            {f.label}
            <input
              type="number"
              step={f.step ?? 0.1}
              value={speed[f.key]}
              onChange={(e) => set(f.key, Number(e.target.value))}
            />
          </label>
        ))}
      </div>

      <h3>Detection</h3>
      <div className="config-grid">
        <label>
          Confidence threshold
          <input
            type="number" min={0.05} max={0.99} step={0.05}
            value={threshold}
            onChange={(e) => onThreshold(Number(e.target.value))}
          />
        </label>
        <label>
          Max bales this cycle (blank = all detected)
          <input
            type="number" min={1}
            value={maxBales}
            onChange={(e) => onMaxBales(e.target.value === "" ? "" : Number(e.target.value))}
          />
        </label>
      </div>

      <h3>Camera orientation</h3>
      <p className="hint">
        Default assumes a map-like image: up = North, right = East. Flip if your
        camera is mounted rotated relative to the crane's real travel axes.
      </p>
      <div className="config-grid">
        <label className="checkbox">
          <input
            type="checkbox" checked={orientation.flip_ns}
            onChange={(e) => onOrientation({ ...orientation, flip_ns: e.target.checked })}
          />
          Flip N/S (image-down = North)
        </label>
        <label className="checkbox">
          <input
            type="checkbox" checked={orientation.flip_ew}
            onChange={(e) => onOrientation({ ...orientation, flip_ew: e.target.checked })}
          />
          Flip E/W (image-right = West)
        </label>
      </div>
    </div>
  );
}
