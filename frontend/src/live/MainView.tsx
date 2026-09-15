import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import FactoryContents from "./FactoryContents";
import PhysicsBales from "./PhysicsBales";

export default function MainView() {
  return (
    <Canvas
      shadows
      camera={{ position: [8, 7, 8], fov: 45, near: 0.1, far: 200 }}
      style={{ width: "100%", height: "100%", background: "#12151a" }}
    >
      <FactoryContents />
      {/* The one and only physics simulation for the live bale stack - see
          PhysicsBales.tsx and physicsBridge.ts for why this must not be
          duplicated inside the camera-feed canvases. */}
      <PhysicsBales />
      <OrbitControls target={[0, 2, 0]} makeDefault />
    </Canvas>
  );
}
