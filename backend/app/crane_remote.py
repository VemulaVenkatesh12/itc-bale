"""
Physical crane-remote bridge.

Routes the live-3D control loop's virtual relay decisions (back end
app/live_controller.py -> /api/control_tick) to the REAL SF-8DR crane remote
controller at 192.168.4.1, so the physical remote's indicator LEDs light up
in lock-step with the simulation. This is the verification aid the operator
uses: run the live/2D sim, watch the virtual relay panel, and confirm the
same NORTH/SOUTH/EAST/WEST/UP/DOWN LED turns on physically.

The remote's own page (192.168.4.1) exposes:
    GET  /api/status                      -> {"ok", "backend", "relays":[...]}
    POST /api/relay?ch=N&state=on|off     -> {ok, channel, state}   (latched)
    POST /api/stop                        -> releases every relay
    POST /api/backend?set=gpio|modbus

Its relay channel layout (gpio backend) is fixed in the remote's HTML:
    0=UP, 1=DOWN, 2=EAST, 3=WEST, 4=SOUTH, 5=NORTH, 6=START, 7=LIGHT

CRITICAL: the remote defaults to the Modbus RTU backend, and with no Modbus
device attached it reports ok:false -> the relays REFUSE to energize (this is
the "backend error - showing last known state" message on the remote's page).
On this plant only the Modbus RTU backend is wired to the physical relays
(the ESP's own GPIO pins go nowhere), so the bridge forces the remote onto
CRANE_REMOTE_BACKEND (config, default "modbus") before any relay work
(see _ensure_backend).

Relay signalling is STATE-MIRRORED, not momentary: a relay is energized on
the tick its sim relay first turns on, and released on the tick it turns off.
This makes the physical LED stay lit exactly as long as the virtual relay
panel shows it lit (the control law's phases last ~0.8-10s), which is what
makes it observable by eye. The old momentary 150ms blip was simply too fast
to see and was never what a real remote does - real remotes hold a button
while the motion is commanded.

Everything in here is best-effort and non-blocking: a failed POST (remote
unreachable, timeout, HTTP error) must never break the simulation loop, so
all errors are swallowed. The control law stays in charge - this module only
*reflects* its decisions outward.
"""
from __future__ import annotations

import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from .config import CRANE_REMOTE_BACKEND, CRANE_REMOTE_ENABLED, CRANE_REMOTE_URL

# sim relay name (as emitted by live_controller._axis_relays/_hoist_relays /
# the push phase) -> physical remote channel. See config.py's docstring for
# the remote's fixed 0-7 channel layout.
RELAY_CHANNEL_MAP = {
    "up_1s": 0,        # UP
    "down_1s": 1,      # DOWN
    "east_1s": 2,      # EAST
    "west_1s": 3,      # WEST
    "south_1s": 4,     # SOUTH
    "north_1s": 5,     # NORTH
    # "push" (the eject relay in the sim) has no dedicated physical button on
    # the 8-channel remote - intentionally unmapped so we never wrongly fire
    # the START/STOP keys for it.
}

# The remote's full fixed 0-7 channel layout, incl. the non-axis keys the
# manual button panel (CraneRemotePanel.tsx / /api/crane_remote/manual)
# exposes but the control loop never touches. Names match the remote's page.
CHANNEL_LABELS = ["UP", "DOWN", "EAST", "WEST", "SOUTH", "NORTH", "START", "LIGHT"]

# Axes that interlock in the remote's firmware (activating one auto-releases
# the other) - the manual panel enforces this client-side too for snappy UI.
INTERLOCK_PAIRS = [(0, 1), (2, 3), (4, 5)]

# Seconds to wait for the remote to answer. The Modbus RTU backend does a
# full RS-485 transaction (~3s when the slave is silent, fast once a real
# relay module answers), so this is generous. Every relay call is dispatched
# on a daemon thread (see _submit) so this timeout NEVER blocks the control
# loop regardless of how slow the remote is.
_TIMEOUT_S = 4.0

# backend name the remote must be on for its physical relays to energize.
# On this plant only "modbus" is wired to the real relays - see config.py.
_TARGET_BACKEND = CRANE_REMOTE_BACKEND

# Serialize access to the wire so concurrent control ticks don't interleave
# half-written requests.
_lock = threading.Lock()

# Single worker thread so the (slow, ~3s on a silent Modbus bus) HTTP calls
# NEVER run on the control-loop thread. Relay commands are queued in order.
_pool_lock = threading.Lock()


def _submit(fn) -> None:
    """Run fn() on a short-lived daemon thread, serialized behind _pool_lock
    so commands still apply in order. Fire-and-forget: the control loop
    returns immediately."""
    def _run():
        with _pool_lock:
            try:
                fn()
            except Exception:
                pass
    threading.Thread(target=_run, daemon=True).start()

# Last set of sim relays reported active (name-level), so we can detect
# turn-on/turn-off transitions and mirror them to the physical remote.
_last_active: frozenset[str] = frozenset()


def _post(url: str, data: Optional[dict] = None) -> bool:
    """Best-effort POST to the remote. Returns True only on a confirmed 2xx
    ok response; never raises."""
    if not CRANE_REMOTE_ENABLED or not CRANE_REMOTE_URL:
        return False
    try:
        if data:
            body = urllib.parse.urlencode(data).encode()
        else:
            body = b""
        req = urllib.request.Request(url, data=body, method="POST")
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
        return False


def _set_channel(ch: int, on: bool) -> bool:
    return _post(f"{CRANE_REMOTE_URL}/api/relay?ch={ch}&state={'on' if on else 'off'}")


def _ensure_backend() -> bool:
    """Force the remote onto CRANE_REMOTE_BACKEND (config; "modbus" on this
    plant - its GPIO pins aren't wired to the physical relays, only the
    external RS-485 Modbus relay module is). No-op if already on it."""
    try:
        with urllib.request.urlopen(f"{CRANE_REMOTE_URL}/api/status", timeout=_TIMEOUT_S) as resp:
            if 200 <= resp.status < 300:
                import json

                cur = json.loads(resp.read().decode())
                if cur.get("backend") == _TARGET_BACKEND:
                    return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError, ValueError):
        pass
    return _post(f"{CRANE_REMOTE_URL}/api/backend?set={_TARGET_BACKEND}")


def fire_relay(relay: str) -> None:
    """Energize a single sim relay on the physical remote (latched ON until
    explicitly released via release_relay/stop_all). No-op (and safe) when
    the bridge is disabled or the relay is unmapped."""
    ch = RELAY_CHANNEL_MAP.get(relay)
    if ch is None:
        return
    _ensure_backend()
    _set_channel(ch, on=True)


def release_relay(relay: str) -> None:
    """De-energize a single sim relay on the physical remote."""
    ch = RELAY_CHANNEL_MAP.get(relay)
    if ch is None:
        return
    _set_channel(ch, on=False)


def stop_all() -> None:
    """Release every relay on the remote (STOP). Used when the sim run is
    stopped/reset so no physical output is left energized."""
    global _last_active
    _last_active = frozenset()
    if not CRANE_REMOTE_ENABLED or not CRANE_REMOTE_URL:
        return

    def _apply():
        _ensure_backend()
        _post(f"{CRANE_REMOTE_URL}/api/stop")

    _submit(_apply)


def manual_channel(ch: int, on: bool) -> bool:
    """Directly drive one physical remote channel (0-7) for the manual
    button panel - independent of the sim control loop. Forces the configured backend first
    (CRANE_REMOTE_BACKEND, "modbus" on this plant) and,
    on turn-on, releases the interlocked opposite so the panel state stays
    consistent with the remote's own firmware interlock. Best-effort; returns
    True only on a confirmed ok. Safe no-op when the bridge is disabled."""
    if not CRANE_REMOTE_ENABLED or not CRANE_REMOTE_URL:
        return False
    if not (0 <= ch <= 7):
        return False

    def _apply():
        _ensure_backend()
        if on:
            for a, b in INTERLOCK_PAIRS:
                if ch == a:
                    _set_channel(b, on=False)
                elif ch == b:
                    _set_channel(a, on=False)
        _set_channel(ch, on=on)

    _submit(_apply)
    return True


def sync_relays(active: list[str]) -> None:
    """Mirror the control loop's CURRENT active relay set onto the physical
    remote: turn on relays newly activated this tick, release relays that
    turned off. State-mirroring (not momentary) so the physical LED stays lit
    exactly as long as the virtual panel - see module docstring. Never blocks
    or raises."""
    global _last_active
    if not CRANE_REMOTE_ENABLED or not CRANE_REMOTE_URL:
        _last_active = frozenset(active)
        return
    now = frozenset(active)
    with _lock:
        turned_on = now - _last_active
        turned_off = _last_active - now
        _last_active = now
    if not turned_on and not turned_off:
        return

    def _apply():
        if turned_on:
            _ensure_backend()
        for r in turned_on:
            if r in RELAY_CHANNEL_MAP:
                _set_channel(RELAY_CHANNEL_MAP[r], on=True)
        for r in turned_off:
            if r in RELAY_CHANNEL_MAP:
                _set_channel(RELAY_CHANNEL_MAP[r], on=False)

    _submit(_apply)


def status() -> Optional[dict]:
    """Best-effort current relay state of the physical remote, or None on
    any failure. Observational only."""
    if not CRANE_REMOTE_ENABLED or not CRANE_REMOTE_URL:
        return None
    try:
        with urllib.request.urlopen(f"{CRANE_REMOTE_URL}/api/status", timeout=_TIMEOUT_S) as resp:
            import json

            if 200 <= resp.status < 300:
                return json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError, ValueError):
        pass
    return None