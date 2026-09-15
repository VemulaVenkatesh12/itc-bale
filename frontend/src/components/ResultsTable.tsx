import type { PickCycle } from "../lib/types";

interface Props {
  pickCycles: PickCycle[];
  tMs: number;
}

export default function ResultsTable({ pickCycles, tMs }: Props) {
  if (pickCycles.length === 0) return null;
  return (
    <div className="results-table">
      <h3>Pick sequence ({pickCycles.length} bales)</h3>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Bale</th>
            <th>Confidence</th>
            <th>Pick point</th>
            <th>Drop point</th>
            <th>Window</th>
          </tr>
        </thead>
        <tbody>
          {pickCycles.map((p) => (
            <tr key={p.bale_id} className={tMs >= p.t_start_ms && tMs < p.t_end_ms ? "active-row" : ""}>
              <td>{p.order}</td>
              <td>#{p.bale_id}</td>
              <td>{(p.confidence * 100).toFixed(0)}%</td>
              <td>({p.pick_point.x.toFixed(0)}, {p.pick_point.y.toFixed(0)})</td>
              <td>({p.drop_point.x.toFixed(0)}, {p.drop_point.y.toFixed(0)})</td>
              <td>{(p.t_start_ms / 1000).toFixed(1)}s – {(p.t_end_ms / 1000).toFixed(1)}s</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
