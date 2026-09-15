import { useEffect, useRef, useState } from "react";

/** Drives a t_ms clock forward via requestAnimationFrame while playing,
 * scaled by a speed multiplier (the "simulation playback speed" control —
 * distinct from the crane's *physical* px/s calibration in SpeedConfig). */
export function usePlayback(totalDurationMs: number) {
  const [tMs, setTMs] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speedMult, setSpeedMult] = useState(1);
  const rafRef = useRef<number | undefined>(undefined);
  const lastTsRef = useRef<number | undefined>(undefined);
  const speedMultRef = useRef(speedMult);
  speedMultRef.current = speedMult;

  useEffect(() => {
    if (!playing || totalDurationMs <= 0) return;

    function tick(now: number) {
      if (lastTsRef.current === undefined) lastTsRef.current = now;
      const dt = now - lastTsRef.current;
      lastTsRef.current = now;
      setTMs((prev) => {
        const next = prev + dt * speedMultRef.current;
        if (next >= totalDurationMs) {
          setPlaying(false);
          return totalDurationMs;
        }
        return next;
      });
      rafRef.current = requestAnimationFrame(tick);
    }
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      lastTsRef.current = undefined;
    };
  }, [playing, totalDurationMs]);

  function reset() {
    setTMs(0);
    setPlaying(false);
  }

  function seek(ms: number) {
    setTMs(Math.max(0, Math.min(totalDurationMs, ms)));
  }

  return { tMs, setTMs: seek, playing, setPlaying, speedMult, setSpeedMult, reset };
}
