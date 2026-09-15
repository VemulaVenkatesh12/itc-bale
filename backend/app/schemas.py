"""
Pydantic schemas shared across the API.

Coordinate convention: ALL pixel coordinates in this module are in the
*original uploaded image's* pixel space (origin top-left, x right, y down).
The frontend is responsible for converting from on-screen/canvas coordinates
to original-image coordinates before calling the API, and for converting
back when rendering results.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Point(BaseModel):
    x: float
    y: float


class Rect(BaseModel):
    """Axis-aligned rectangle, corners in image pixel space."""
    x1: float
    y1: float
    x2: float
    y2: float

    def normalized(self) -> "Rect":
        return Rect(
            x1=min(self.x1, self.x2),
            y1=min(self.y1, self.y2),
            x2=max(self.x1, self.x2),
            y2=max(self.y1, self.y2),
        )

    @property
    def width(self) -> float:
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return abs(self.y2 - self.y1)

    @property
    def center(self) -> Point:
        return Point(x=(self.x1 + self.x2) / 2, y=(self.y1 + self.y2) / 2)


class UploadResponse(BaseModel):
    image_id: str
    url: str
    width: int
    height: int


class Detection(BaseModel):
    id: int
    bbox: Rect
    confidence: float
    center: Point
    # 0=bale, 1=crane_hook, 2=crane_spike (see model_service.CLASS_*). Defaults
    # to 0 (bale) since detect_bales() only ever returns bale detections and
    # older callers don't pass this.
    class_id: int = 0


class DetectResponse(BaseModel):
    """Response for POST /api/detect - a single stateless frame's raw,
    multi-class detections (bale/crane_hook/crane_spike). Used by the live
    3D control mode, which calls this repeatedly against rendered camera
    frames instead of the one-shot upload+/api/simulate flow."""
    width: int
    height: int
    detections: list[Detection]


class ControlTickResponse(BaseModel):
    """Response for POST /api/control_tick - the live 3D mode's actual
    control-loop decision (see live_controller.py), computed server-side
    every tick from a fresh camera frame + the crane's self-reported current
    pose. Stateless: the frontend just applies this and reports its own
    resulting state back next tick, the way a real controller reads fresh
    encoder feedback each scan rather than trusting its own memory.

    target_x/y/z is the CURRENT leg's setpoint (not necessarily where the
    crane ends up this instant) - the frontend's local ~60fps loop drives
    toward it at the speed implied by active_relays and stops itself at
    arrival, mirroring how a real servo/VFD drive handles precise stopping
    between PLC scans (see live_controller.py's module docstring for why a
    pure "just say which relay" design would overshoot at our vision tick
    rate)."""
    width: int
    height: int
    detections: list[Detection]

    phase: str
    active_relays: list[str]
    target_bale_id: Optional[int] = None
    target_x: float
    target_y: float
    target_z: float
    mark_picked_bale_id: Optional[int] = None
    log: Optional[str] = None
    # "vision" when this tick's (x, z) came from unprojecting a confidently
    # (>= live_controller.HOOK_POSITION_CONFIDENCE) detected crane_hook in
    # THIS frame; "self_reported" when the hook wasn't confidently seen and
    # the frontend's own dead-reckoned craneX/Z (echoed back in the request)
    # was used instead - see live_controller.py's module docstring and
    # training/README.md's hook-visibility-rate note for why the fallback is
    # expected to fire often, not just as a rare edge case. Y is always
    # self-reported (a single top-down camera can't recover height without
    # depth/stereo), so this only ever describes (x, z).
    position_source: Literal["vision", "self_reported"] = "self_reported"


class SpeedConfig(BaseModel):
    """
    Physical calibration of the crane, in pixels/second of the *uploaded
    image*. Tune these so simulated travel time matches your real crane's
    speed relative to the image's real-world scale.

    High speed (2S) is only engaged for moves longer than
    `high_speed_min_distance_px`; the final `creep_distance_px` of any long
    move is always done at low speed (1S) for precision positioning - this
    mirrors how these two-speed contactors are actually driven (1S line
    alone = low speed, 1S+2S together = high speed).
    """
    ew_low_px_s: float = Field(default=60, gt=0)
    ew_high_px_s: float = Field(default=180, gt=0)
    ns_low_px_s: float = Field(default=45, gt=0)
    ns_high_px_s: float = Field(default=130, gt=0)
    high_speed_min_distance_px: float = Field(default=150, ge=0)
    creep_distance_px: float = Field(default=40, ge=0)

    hoist_clear_s: float = Field(default=1.5, ge=0)   # lift with/without load, to clear stack height
    hoist_lower_s: float = Field(default=1.5, ge=0)   # lower onto a bale or onto the drop surface
    hoist_creep_s: float = Field(default=0.4, ge=0)   # tail portion of each hoist move done at 1S

    pierce_dwell_s: float = Field(default=0.8, ge=0)  # settle time as the fixed spike seats into the bale (no relay — mechanical)
    push_off_s: float = Field(default=0.8, ge=0)      # hydraulic push-relay pulse ejecting the bale off the spike at the drop

    settle_s: float = Field(default=0.25, ge=0)       # brief pause between phases (relays fully de-energized)

    playback_speed: float = Field(default=1.0, gt=0)  # frontend-only convenience echoed back


class OrientationConfig(BaseModel):
    """
    Maps image screen directions to real-world compass directions used by
    the SF-8DR relay labels. Defaults assume a "map-like" image: up = North,
    right = East. Flip these if your camera is mounted rotated relative to
    the crane's actual N/S/E/W travel axes.
    """
    flip_ns: bool = False  # if True, image-down = North (instead of South)
    flip_ew: bool = False  # if True, image-right = West (instead of East)


class SimulateRequest(BaseModel):
    image_id: str
    detection_area: Rect
    drop_area: Rect
    crane_home: Optional[Point] = None
    threshold: float = Field(default=0.6, ge=0.05, le=0.99)
    speed: SpeedConfig = Field(default_factory=SpeedConfig)
    orientation: OrientationConfig = Field(default_factory=OrientationConfig)
    max_bales: Optional[int] = Field(default=None, ge=1, description="Limit how many detected bales to cycle through")


AxisName = Literal["ew", "ns", "hoist", "push", "system"]


class RelayEvent(BaseModel):
    relay: str          # e.g. "east_1s", "up_2s", "push"
    axis: AxisName
    phase: str           # human-readable phase label, e.g. "travel_to_bale#1"
    on_at_ms: int
    off_at_ms: int


class CraneKeyframe(BaseModel):
    t_ms: int
    x: float
    y: float
    hoist: float          # 0 = fully down/at load, 1 = fully up/clear (visual only)
    thrust: float = 0.0    # 0 = spike retracted, 1 = spike fully engaged in the bale (horizontal stab,
                            # parallel to ground — NOT the same axis as hoist). Unused by the 2D view;
                            # exists so the 3D (Blender) export has a real piercing axis to animate.
    spiked: bool           # bale currently impaled on the spike (animation state only — holding fires no relay)
    active_bale_id: Optional[int] = None
    phase: str


class PickCycle(BaseModel):
    order: int
    bale_id: int
    bbox: Rect
    confidence: float
    pick_point: Point
    drop_point: Point
    t_start_ms: int
    t_end_ms: int


class SimulationResult(BaseModel):
    image_width: int
    image_height: int
    detections: list[Detection]
    pick_cycles: list[PickCycle]
    relay_timeline: list[RelayEvent]
    keyframes: list[CraneKeyframe]
    total_duration_ms: int
    crane_home: Point
    warnings: list[str] = Field(default_factory=list)
