import { useState } from "react";
import "./App.css";
import ImageCanvas, { type DrawMode } from "./components/ImageCanvas";
import ConfigPanel from "./components/ConfigPanel";
import ControllerPanel from "./components/ControllerPanel";
import Timeline from "./components/Timeline";
import ResultsTable from "./components/ResultsTable";
import LiveControlPage from "./live/LiveControlPage";
import LiveControlPage3D from "./live/LiveControlPage3D";
import { uploadImage, runSimulation } from "./lib/api";
import { usePlayback } from "./lib/usePlayback";
import { usePhysicalRelayBridge } from "./lib/usePhysicalRelayBridge";
import CraneRemotePanel from "./components/CraneRemotePanel";
import {
  DEFAULT_ORIENTATION, DEFAULT_SPEED,
  type OrientationConfig, type Point, type Rect, type SimulationResult, type SpeedConfig, type UploadResponse,
} from "./lib/types";

type UiMode = "2d" | "3d" | "3d-sim";

export default function App() {
  const [uiMode, setUiMode] = useState<UiMode>("2d");
  const [image, setImage] = useState<UploadResponse | null>(null);
  const [mode, setMode] = useState<DrawMode>("none");
  const [detectionArea, setDetectionArea] = useState<Rect | null>(null);
  const [dropArea, setDropArea] = useState<Rect | null>(null);
  const [craneHome, setCraneHome] = useState<Point | null>(null);

  const [speed, setSpeed] = useState<SpeedConfig>(DEFAULT_SPEED);
  const [orientation, setOrientation] = useState<OrientationConfig>(DEFAULT_ORIENTATION);
  const [threshold, setThreshold] = useState(0.6);
  const [maxBales, setMaxBales] = useState<number | "">("");

  const [simResult, setSimResult] = useState<SimulationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const playback = usePlayback(simResult?.total_duration_ms ?? 0);

  // Mirror the 2D playback's active relays onto the REAL crane remote
  // (physical LEDs) - only relevant in 2D sim mode. See
  // usePhysicalRelayBridge.ts. No-op when there's no timeline.
  usePhysicalRelayBridge(simResult?.relay_timeline, playback.tMs, playback.playing);

  async function handleFile(f: File) {
    setError(null);
    setSimResult(null);
    playback.reset();
    setDetectionArea(null);
    setDropArea(null);
    setCraneHome(null);
    try {
      const up = await uploadImage(f);
      setImage(up);
    } catch (e) {
      setError(String(e));
    }
  }

  async function handleSimulate() {
    if (!image || !detectionArea || !dropArea) {
      setError("Upload an image and mark both the detection area and the drop area first.");
      return;
    }
    setError(null);
    setLoading(true);
    playback.reset();
    try {
      const result = await runSimulation({
        image_id: image.image_id,
        detection_area: detectionArea,
        drop_area: dropArea,
        crane_home: craneHome,
        threshold,
        speed,
        orientation,
        max_bales: maxBales === "" ? null : maxBales,
      });
      setSimResult(result);
      playback.setPlaying(true);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Automated Bale Unloading — CV Simulator</h1>
        <p className="subtitle">
          {uiMode === "2d"
            ? "Upload a frame, mark the detection & drop areas, then Simulate. Output drives a simulated SF-8DR relay panel — plug real GPIO/serial in place of the panel later."
            : "The two real plant cameras, live — detection, numbered pick order, and the conveyor gate, run against real video. The Three.js simulation view is parked, not removed — see live/LiveControlPage3D.tsx."}
        </p>
        <div className="mode-tabs">
          <button className={uiMode === "2d" ? "active" : ""} onClick={() => setUiMode("2d")}>
            2D Simulator (uploaded photo)
          </button>
          <button className={uiMode === "3d" ? "active" : ""} onClick={() => setUiMode("3d")}>
            Live Plant Control (real cameras)
          </button>
          <button className={uiMode === "3d-sim" ? "active" : ""} onClick={() => setUiMode("3d-sim")}>
            3D Simulation (Three.js)
          </button>
        </div>
      </header>

      {uiMode === "3d" && <LiveControlPage />}
      {uiMode === "3d-sim" && <LiveControlPage3D />}

      {uiMode === "2d" && <div className="layout">
        <div className="main-col">
          <div className="toolbar">
            <label className="file-input">
              Upload image
              <input type="file" accept="image/*" onChange={(e) => e.target.files && handleFile(e.target.files[0])} />
            </label>
            {image && (
              <div className="mode-buttons">
                <button className={mode === "detection" ? "active" : ""} onClick={() => setMode(mode === "detection" ? "none" : "detection")}>
                  Mark detection area
                </button>
                <button className={mode === "drop" ? "active" : ""} onClick={() => setMode(mode === "drop" ? "none" : "drop")}>
                  Mark drop area
                </button>
                <button className={mode === "home" ? "active" : ""} onClick={() => setMode(mode === "home" ? "none" : "home")}>
                  Set crane home (optional)
                </button>
                <button className="simulate-btn" onClick={handleSimulate} disabled={loading}>
                  {loading ? "Simulating…" : "Simulate"}
                </button>
              </div>
            )}
          </div>

          {error && <div className="error-banner">{error}</div>}
          {simResult?.warnings.map((w, i) => <div className="warning-banner" key={i}>{w}</div>)}

          {image ? (
            <ImageCanvas
              imageUrlPath={image.url}
              naturalWidth={image.width}
              naturalHeight={image.height}
              mode={mode}
              detectionArea={detectionArea}
              dropArea={dropArea}
              craneHome={craneHome}
              onDetectionArea={(r) => { setDetectionArea(r); setMode("none"); }}
              onDropArea={(r) => { setDropArea(r); setMode("none"); }}
              onCraneHome={(p) => { setCraneHome(p); setMode("none"); }}
              simResult={simResult}
              tMs={playback.tMs}
            />
          ) : (
            <div className="empty-state">Upload an image of the truck bed / bale stack to begin.</div>
          )}

          {simResult && (
            <Timeline
              timeline={simResult.relay_timeline}
              totalDurationMs={simResult.total_duration_ms}
              tMs={playback.tMs}
              onSeek={playback.setTMs}
              playing={playback.playing}
              onTogglePlay={() => playback.setPlaying(!playback.playing)}
              speedMult={playback.speedMult}
              onSpeedMult={playback.setSpeedMult}
              onReset={playback.reset}
            />
          )}

          {simResult && <ResultsTable pickCycles={simResult.pick_cycles} tMs={playback.tMs} />}
        </div>

        <div className="side-col">
          <ControllerPanel timeline={simResult?.relay_timeline ?? []} tMs={playback.tMs} />
          <CraneRemotePanel />
          <ConfigPanel
            speed={speed} onSpeed={setSpeed}
            orientation={orientation} onOrientation={setOrientation}
            threshold={threshold} onThreshold={setThreshold}
            maxBales={maxBales} onMaxBales={setMaxBales}
          />
        </div>
      </div>}
    </div>
  );
}
