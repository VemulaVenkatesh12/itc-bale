// The "Live 3D Control" tab's content, 2026-09-11 onward: real plant camera
// feeds + calibration-facing tools only - no Three.js simulation here
// anymore. The full 3D sim view (virtual cameras, closed-loop Start/Stop
// against the simulated scene, relay LED panel) is parked, unmodified, in
// LiveControlPage3D.tsx - see that file's header for how to bring it back.
// Nothing about the backend control logic it used was touched.
import "./live.css";
import PlantCameraFeed from "./PlantCameraFeed";
import RealPickDryRun from "./RealPickDryRun";
import CraneRemotePanel from "../components/CraneRemotePanel";

export default function LiveControlPage() {
  return (
    <div className="live-page live-page--real-only">
      <div className="live-main">
        <div className="live-feeds live-feeds--plant">
          <PlantCameraFeed camId="cam101" label="Camera 101 · 192.168.1.101 (plant RTSP)" />
          <PlantCameraFeed camId="cam102" label="Camera 102 · 192.168.1.102 (plant RTSP)" />
        </div>

        <RealPickDryRun camId="cam101" />
      </div>

      <div className="live-side">
        <CraneRemotePanel />
      </div>
    </div>
  );
}
