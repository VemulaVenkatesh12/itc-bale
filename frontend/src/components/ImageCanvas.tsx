import { useRef, useState, type MouseEvent as ReactMouseEvent } from "react";
import type { Point, Rect, SimulationResult } from "../lib/types";
import { imageUrl } from "../lib/api";
import { interpolateCrane } from "../lib/player";

export type DrawMode = "none" | "detection" | "drop" | "home";

interface Props {
  imageUrlPath: string;
  naturalWidth: number;
  naturalHeight: number;
  mode: DrawMode;
  detectionArea: Rect | null;
  dropArea: Rect | null;
  craneHome: Point | null;
  onDetectionArea: (r: Rect) => void;
  onDropArea: (r: Rect) => void;
  onCraneHome: (p: Point) => void;
  simResult: SimulationResult | null;
  tMs: number;
}

/** Renders the uploaded image plus an SVG overlay (in natural-image pixel
 * coordinates via viewBox, so no scale math is needed elsewhere) for
 * drawing the detection/drop rectangles, showing detections, and animating
 * the crane hook along the planned path. */
export default function ImageCanvas({
  imageUrlPath, naturalWidth, naturalHeight, mode,
  detectionArea, dropArea, craneHome,
  onDetectionArea, onDropArea, onCraneHome,
  simResult, tMs,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [draft, setDraft] = useState<Rect | null>(null);
  const draggingRef = useRef(false);

  function toImageCoords(e: ReactMouseEvent): Point {
    const svg = svgRef.current!;
    const rect = svg.getBoundingClientRect();
    const scaleX = naturalWidth / rect.width;
    const scaleY = naturalHeight / rect.height;
    return {
      x: Math.max(0, Math.min(naturalWidth, (e.clientX - rect.left) * scaleX)),
      y: Math.max(0, Math.min(naturalHeight, (e.clientY - rect.top) * scaleY)),
    };
  }

  function handleMouseDown(e: ReactMouseEvent) {
    if (mode === "none") return;
    const p = toImageCoords(e);
    if (mode === "home") {
      onCraneHome(p);
      return;
    }
    draggingRef.current = true;
    setDraft({ x1: p.x, y1: p.y, x2: p.x, y2: p.y });
  }

  function handleMouseMove(e: ReactMouseEvent) {
    if (!draggingRef.current || !draft) return;
    const p = toImageCoords(e);
    setDraft({ ...draft, x2: p.x, y2: p.y });
  }

  function handleMouseUp() {
    if (!draggingRef.current || !draft) return;
    draggingRef.current = false;
    const r: Rect = {
      x1: Math.min(draft.x1, draft.x2),
      y1: Math.min(draft.y1, draft.y2),
      x2: Math.max(draft.x1, draft.x2),
      y2: Math.max(draft.y1, draft.y2),
    };
    setDraft(null);
    if (r.x2 - r.x1 < 5 || r.y2 - r.y1 < 5) return; // ignore accidental clicks
    if (mode === "detection") onDetectionArea(r);
    if (mode === "drop") onDropArea(r);
  }

  const crane = simResult ? interpolateCrane(simResult.keyframes, tMs) : null;
  const home = simResult?.crane_home ?? craneHome;

  // Gantry silhouette geometry: the bridge beam hovers a fixed offset above
  // wherever the hook currently is, the trolley rides the beam at the
  // hook's X, and the hoist cable stretches/retracts between them — this
  // reads as an actual crane instead of a floating dot.
  const bridgeOffset = naturalHeight * 0.12;
  const hookRetract = Math.max(0, bridgeOffset - naturalHeight * 0.04);
  const bridgeY = crane ? crane.y - bridgeOffset : 0;
  const hookY = crane ? crane.y - crane.hoist * hookRetract : 0;

  const avgBaleW = simResult && simResult.detections.length
    ? simResult.detections.reduce((s, d) => s + (d.bbox.x2 - d.bbox.x1), 0) / simResult.detections.length
    : naturalWidth * 0.05;
  const avgBaleH = simResult && simResult.detections.length
    ? simResult.detections.reduce((s, d) => s + (d.bbox.y2 - d.bbox.y1), 0) / simResult.detections.length
    : naturalHeight * 0.05;

  function baleStatus(baleId: number): "pending" | "active" | "done" {
    const cycle = simResult?.pick_cycles.find((p) => p.bale_id === baleId);
    if (!cycle) return "pending";
    if (tMs >= cycle.t_end_ms) return "done";
    if (tMs >= cycle.t_start_ms) return "active";
    return "pending";
  }

  return (
    <div className="canvas-wrap" style={{ aspectRatio: `${naturalWidth} / ${naturalHeight}` }}>
      <img src={imageUrl(imageUrlPath)} alt="uploaded scene" draggable={false} />
      <svg
        ref={svgRef}
        viewBox={`0 0 ${naturalWidth} ${naturalHeight}`}
        className={`overlay mode-${mode}`}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        {detectionArea && <RectShape r={detectionArea} className="area-detection" label="DETECT" />}
        {dropArea && <RectShape r={dropArea} className="area-drop" label="DROP" />}
        {draft && (
          <RectShape
            r={{ x1: Math.min(draft.x1, draft.x2), y1: Math.min(draft.y1, draft.y2), x2: Math.max(draft.x1, draft.x2), y2: Math.max(draft.y1, draft.y2) }}
            className={mode === "detection" ? "area-detection draft" : "area-drop draft"}
          />
        )}

        {simResult?.detections.map((d) => {
          const order = simResult.pick_cycles.find((p) => p.bale_id === d.id)?.order;
          const status = baleStatus(d.id);
          // A bale leaves its original spot the instant the hook spikes it,
          // and never comes back (it's dropped at the target). While it's
          // being carried it's drawn on the hook instead; once the cycle is
          // done it's gone entirely.
          const lifted = crane?.spiked && crane.active_bale_id === d.id;
          if (status === "done" || lifted) return null;
          return (
            <g key={d.id} className={`bale-box ${status}`}>
              <rect x={d.bbox.x1} y={d.bbox.y1} width={d.bbox.x2 - d.bbox.x1} height={d.bbox.y2 - d.bbox.y1} />
              <text x={d.bbox.x1 + 4} y={d.bbox.y1 + 16}>
                {order ? `#${order}` : ""} {(d.confidence * 100).toFixed(0)}%
              </text>
            </g>
          );
        })}

        {home && (
          <g className="home-marker">
            <circle cx={home.x} cy={home.y} r={10} />
            <text x={home.x + 12} y={home.y + 4}>HOME</text>
          </g>
        )}

        {crane && (
          <g className="crane-rig">
            {/* Bridge (long-travel / N-S beam) — spans the full working width,
               hovers a fixed height above wherever the trolley currently is. */}
            <line x1={0} y1={bridgeY} x2={naturalWidth} y2={bridgeY} className="bridge-beam" />
            <line x1={4} y1={bridgeY} x2={4} y2={bridgeY + bridgeOffset * 0.5} className="bridge-leg" />
            <line x1={naturalWidth - 4} y1={bridgeY} x2={naturalWidth - 4} y2={bridgeY + bridgeOffset * 0.5} className="bridge-leg" />

            {/* Trolley (cross-travel / E-W carriage) riding the bridge */}
            <rect
              x={crane.x - naturalWidth * 0.018} y={bridgeY - naturalHeight * 0.012}
              width={naturalWidth * 0.036} height={naturalHeight * 0.024}
              className="trolley"
            />

            {/* Hoist cable + hook, extending down to the working point */}
            <line x1={crane.x} y1={bridgeY} x2={crane.x} y2={hookY} className="hoist-cable" />
            <g className={crane.spiked ? "hook spiked" : "hook"}>
              <circle cx={crane.x} cy={hookY} r={naturalWidth * 0.007} className="hook-body" />
              {crane.spiked && (
                <rect
                  x={crane.x - avgBaleW / 2} y={hookY}
                  width={avgBaleW} height={avgBaleH * 0.8}
                  className="carried-bale"
                />
              )}
            </g>
          </g>
        )}
      </svg>
    </div>
  );
}

function RectShape({ r, className, label }: { r: Rect; className: string; label?: string }) {
  return (
    <g className={className}>
      <rect x={r.x1} y={r.y1} width={Math.max(0, r.x2 - r.x1)} height={Math.max(0, r.y2 - r.y1)} />
      {label && <text x={r.x1 + 4} y={r.y1 + 16}>{label}</text>}
    </g>
  );
}
