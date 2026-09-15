// Bridges the 2D simulator's playback relays to the REAL physical crane
// remote so its LEDs light while the sim plays. The 2D flow computes active
// relays client-side (ControllerPanel -> activeRelays) and previously never
// touched the backend, so the physical remote stayed dark. This hook watches
// the same relay set during playback and pushes state changes to
// POST /api/crane_remote/relays (backend/app/crane_remote.py's sync_relays,
// which auto-forces the remote's GPIO backend and state-mirrors the LEDs).
import { useEffect, useRef } from "react";
import type { RelayEvent } from "../lib/types";
import { activeRelays } from "../lib/player";
import { syncPhysicalRelays, stopPhysicalRelays } from "../lib/api";

export function usePhysicalRelayBridge(timeline: RelayEvent[] | undefined, tMs: number, playing: boolean) {
  const lastSentRef = useRef<string[] | null>(null);

  useEffect(() => {
    if (!timeline || timeline.length === 0) {
      if (lastSentRef.current !== null) {
        lastSentRef.current = null;
        void stopPhysicalRelays();
      }
      return;
    }
    const active = [...activeRelays(timeline, tMs)].sort();
    const key = active.join(",");
    if (lastSentRef.current === null || key !== lastSentRef.current.join(",")) {
      lastSentRef.current = active;
      if (active.length === 0) {
        void stopPhysicalRelays();
      } else {
        void syncPhysicalRelays(active);
      }
    }
  }, [timeline, tMs, playing]);
}