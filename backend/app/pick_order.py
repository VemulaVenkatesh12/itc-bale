"""
Live pick-order numbering for the REAL plant cameras - a 2D pixel-space
stand-in for real_deploy/planner3d.py's group_into_layers()/
order_bales_by_height(), usable TODAY because it needs only one live camera
and no calibration, at the cost of the limitation explained below.

Confirmed pick-order rule (revised 2026-09-12): ROW-major, alternating
layers within each row - not "clear the whole top layer, then the whole
second layer". Every truck this session backs in tailgate-toward-camera, so
"row" position along the truck bed maps to bbox bottom edge (y2) - nearest
the tailgate = nearest the camera = LARGEST y2. The actual sequence:

    row 0 (nearest tailgate), top layer bales (left-to-right)
    row 0, second layer bales (left-to-right)
    row 1 (next row toward the cab), top layer bales
    row 1, second layer bales
    ... and so on until every row in both layers is numbered.

This mirrors how the crane actually works a load: clear one row's top bale,
then the bale it exposes right below it, before moving on - not sweep the
entire top layer across the whole bed first and only then come back for the
bottom one (which the previous "topmost layer first, then work down" rule
did, and which meant far more back-and-forth travel than necessary). See
group_into_rows_px() for how a "row" is detected, and order_layer() for the
within-layer sweep direction (still last-loaded-first, unchanged).

If a truck is ever parked nose-in instead of backed-in, this direction
inverts (flip DESCENDING_Y2 below) - nothing else about the pipeline needs
to change, just confirm which end of the bed is nearest the camera before
trusting the numbering that session.

## The real limitation, stated plainly

"Layer" here is approximated by a bale's bounding-box TOP EDGE in the image
(smaller y = higher in frame). That is a reasonable proxy when the camera
looks across the stack at a shallow angle - which the current cam101
mounting mostly does - but it is NOT real height. A bale further from the
camera can appear higher in the image while sitting on the same physical
layer as one nearer the camera, and this has no way to tell the difference.
`real_deploy/planner3d.py`'s `group_into_layers()` does this correctly
because it groups by triangulated real-world Z instead - use that, not this,
once calibration (PLANT_VISIT_CHECKLIST.md) is done. This module exists to
make the pick ORDER visible and reviewable on live video before that, not to
replace it.

Likewise "crane's current position" here defaults to a configurable point
(e.g. a home corner) since a single 2D camera has no depth to place the real
crane in - real_deploy/track_reference.py does that properly once
calibrated. Pass a different `start_xy` if you have a better live estimate.
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .schemas import Detection

CLASS_BALE = 0

# Bounding-box top-edge spread (pixels) treated as "the same layer" - must be
# smaller than one bale's height in pixels but bigger than detection jitter.
# Tune per camera/zoom; this is a starting default for cam101's current framing.
DEFAULT_LAYER_BAND_PX = 60.0

# Bounding-box bottom-edge spread (pixels), WITHIN one layer, treated as "the
# same row" (same front-to-back position along the truck bed) - see
# group_into_rows_px(). Same tuning logic as DEFAULT_LAYER_BAND_PX, just
# along the other axis.
DEFAULT_ROW_BAND_PX = 60.0

# POC SCOPE (2026-09-12): only the LEFT truck + the conveyor are in play for
# this proof of concept - the right-side truck and background stacks are
# explicitly out of scope, not just unlikely to be picked. Bales outside this
# region are dropped before layering/numbering even starts, so they never
# get a number and never affect which layer is "topmost". Same polygon is
# used by boundary_guard.py's safety region - keep the two in sync if this
# is retraced.
SCOPE_ROI_PX: dict[str, list[tuple[float, float]]] = {
    "cam101": [(90, 0), (745, 65), (1155, 450), (700, 650), (90, 650)],
}


def in_scope(d: Detection, cam_id: str) -> bool:
    """True if this camera has no traced scope (nothing filtered) or the
    detection's centre falls inside the traced left-truck+conveyor region."""
    poly = SCOPE_ROI_PX.get(cam_id)
    if not poly:
        return True
    pts = np.array(poly, dtype=np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(pts, (float(d.center.x), float(d.center.y)), False) >= 0


@dataclass
class Numbered:
    number: int
    detection: Detection
    layer: int  # 0 = topmost


def _center(d: Detection) -> tuple[float, float]:
    return (d.center.x, d.center.y)


def group_into_layers_px(bales: list[Detection], band_px: float = DEFAULT_LAYER_BAND_PX) -> list[list[Detection]]:
    """Topmost-in-image layer first. Mirrors planner3d.group_into_layers()
    but keys off bbox.y1 (smaller = higher in frame) instead of real Z."""
    remaining = sorted(bales, key=lambda d: d.bbox.y1)
    layers: list[list[Detection]] = []
    while remaining:
        top_y = remaining[0].bbox.y1
        layer = [d for d in remaining if d.bbox.y1 <= top_y + band_px]
        layers.append(layer)
        layer_ids = {id(d) for d in layer}
        remaining = [d for d in remaining if id(d) not in layer_ids]
    return layers


# True while every truck backs in tailgate-toward-camera (every session so
# far) - see the module docstring for how to flip this if that ever changes.
DESCENDING_Y2 = True


def order_layer(layer: list[Detection], start_xy: tuple[float, float]) -> list[Detection]:
    """Last-loaded-first within one layer: nearest the open tailgate (in
    this camera's framing, the largest bbox bottom edge / y2) comes off
    first, working toward the cab end. A fixed sweep direction, not a
    distance-from-crane rule - see module docstring for why and how to flip
    it. `start_xy` is accepted for API compatibility with the caller
    (PickOrderTracker) but no longer used to choose the order."""
    return sorted(layer, key=lambda d: d.bbox.y2, reverse=DESCENDING_Y2)


def group_into_rows_px(bales: list[Detection], band_px: float = DEFAULT_ROW_BAND_PX) -> list[list[Detection]]:
    """Groups ONE layer's bales into rows along the truck bed's length,
    nearest-tailgate-first - same chain-grouping approach as
    group_into_layers_px, just keyed off bbox.y2 (bottom edge; larger =
    nearer the tailgate/camera, matching DESCENDING_Y2) instead of y1.
    Bales within a row are then ordered left-to-right (bbox.x1 ascending) -
    a fixed sweep direction across the row's width, not a distance rule."""
    remaining = sorted(bales, key=lambda d: d.bbox.y2, reverse=True)
    rows: list[list[Detection]] = []
    while remaining:
        near_y2 = remaining[0].bbox.y2
        row = [d for d in remaining if near_y2 - d.bbox.y2 <= band_px]
        rows.append(sorted(row, key=lambda d: d.bbox.x1))
        row_ids = {id(d) for d in row}
        remaining = [d for d in remaining if id(d) not in row_ids]
    return rows


def compute_pick_order(
    detections: list[Detection],
    start_xy: tuple[float, float],
    max_layers: Optional[int] = 2,
    layer_band_px: float = DEFAULT_LAYER_BAND_PX,
    cam_id: Optional[str] = None,
    row_band_px: float = DEFAULT_ROW_BAND_PX,
) -> list[Numbered]:
    """Bale-only detections -> numbered pick order, ROW-major and
    alternating layers within each row (see module docstring for why):
    nearest-tailgate row's top-layer bales, then that same row's
    second-layer bales, then the next row's top-layer bales, and so on.
    `max_layers=None` numbers everything visible; `max_layers=2` (default,
    per the confirmed top-two-layers scope) only numbers the top two pixel-
    layers and leaves the rest un-numbered. `cam_id` restricts to
    SCOPE_ROI_PX (POC scope: left truck + conveyor only) when traced for
    that camera - bales outside it are dropped before layering even starts,
    so they never affect which layer is "topmost"."""
    bales = [d for d in detections if d.class_id == CLASS_BALE]
    if cam_id is not None:
        bales = [d for d in bales if in_scope(d, cam_id)]
    layers = group_into_layers_px(bales, layer_band_px)
    if max_layers is not None:
        layers = layers[:max_layers]
    layer_rows = [group_into_rows_px(layer, row_band_px) for layer in layers]
    row_count = max((len(rows) for rows in layer_rows), default=0)

    numbered: list[Numbered] = []
    n = 1
    for row_idx in range(row_count):
        for layer_idx, rows in enumerate(layer_rows):
            if row_idx >= len(rows):
                continue  # this layer has fewer rows than another (e.g. partly picked already)
            for d in rows[row_idx]:
                numbered.append(Numbered(number=n, detection=d, layer=layer_idx))
                n += 1
    return numbered


class PickOrderTracker:
    """Keeps pick numbers STABLE across live polls as bales are physically
    removed. The truck doesn't move - only the bale count drops - so a
    picked-off bale simply stops matching anything next frame; everything
    else keeps its number. Re-numbers from scratch only when the scene
    changes too much to be "the same truck, fewer bales" (a big jump in
    count, e.g. a new truck backed in)."""

    MATCH_RADIUS_PX = 40.0

    # A bale can miss detection for a poll or two for reasons that have
    # nothing to do with it moving - the hook/spike passing in front of it,
    # a confidence dip, motion blur. Without tolerance for that, the target
    # (#1) identity flips to a different physical bale the instant that
    # happens, which is exactly the "moving here and there" symptom this is
    # fixing - a numbered bale now survives up to this many CONSECUTIVE
    # missed polls (holding its last-seen position) before it's dropped.
    MAX_CONSECUTIVE_MISSES = 3

    def __init__(self, start_xy: tuple[float, float], max_layers: Optional[int] = 2,
                 layer_band_px: float = DEFAULT_LAYER_BAND_PX, cam_id: Optional[str] = None,
                 row_band_px: float = DEFAULT_ROW_BAND_PX) -> None:
        self.start_xy = start_xy
        self.max_layers = max_layers
        self.layer_band_px = layer_band_px
        self.row_band_px = row_band_px
        self.cam_id = cam_id
        self._prev: list[Numbered] = []
        self._misses: dict[int, int] = {}  # pick number -> consecutive missed polls

    def reset(self) -> None:
        self._prev = []
        self._misses = {}

    def update(self, detections: list[Detection]) -> list[Numbered]:
        bales = [d for d in detections if d.class_id == CLASS_BALE]
        if self.cam_id is not None:
            bales = [d for d in bales if in_scope(d, self.cam_id)]
        if not self._prev:
            self._prev = compute_pick_order(bales, self.start_xy, self.max_layers, self.layer_band_px, row_band_px=self.row_band_px)
            return self._prev

        # Which CURRENT bales are actually within the top `max_layers` bands
        # right now - a bale that only just became detected (occlusion
        # cleared, confidence ticked up, etc.) must clear this same bar
        # before it can ever be numbered. Without it, any newly-seen bale
        # that failed to match an existing (already top-2-layer) numbered
        # entry fell through as "unmatched" and got appended as a fresh pick
        # target regardless of its real layer - silently breaking the
        # top-two-layers scope as bales were removed and lower layers came
        # into view.
        #
        # This must run BEFORE the big-jump check below: `bales` here is
        # EVERY in-scope bale (every layer, easily 30+ on a full truck), and
        # `self._prev` only ever holds the numbered top-`max_layers` set
        # (~10). Comparing those two directly (as this used to) means the
        # jump condition was almost always true - a normal frame's lower,
        # un-numbered layers flickering in and out of detection was enough
        # to trip a full re-sort on nearly EVERY poll, which is exactly what
        # was making target #1 hop between unrelated bales second to second.
        # The jump check has to compare like with like: current eligible
        # (top-`max_layers`) count against previous kept count.
        current_layers = group_into_layers_px(bales, self.layer_band_px)
        if self.max_layers is not None:
            current_layers = current_layers[: self.max_layers]
        eligible_layer_of: dict[int, int] = {
            id(d): layer_idx for layer_idx, layer in enumerate(current_layers) for d in layer
        }
        eligible = [d for d in bales if id(d) in eligible_layer_of]

        # A big jump upward in the ELIGIBLE count means a fresh load, not
        # bales reappearing - start numbering over rather than carry stale
        # IDs. (Recomputed from `bales`, not `eligible`, so max_layers is
        # re-applied cleanly against the new scene rather than the stale
        # eligibility map above.)
        if len(eligible) > len(self._prev) + 2:
            self._prev = compute_pick_order(bales, self.start_xy, self.max_layers, self.layer_band_px, row_band_px=self.row_band_px)
            return self._prev

        unmatched = list(eligible)
        kept: list[Numbered] = []
        next_misses: dict[int, int] = {}
        for prev in self._prev:
            pcx, pcy = _center(prev.detection)
            best, best_d2 = None, self.MATCH_RADIUS_PX ** 2
            for d in unmatched:
                cx, cy = _center(d)
                d2 = (cx - pcx) ** 2 + (cy - pcy) ** 2
                if d2 < best_d2:
                    best, best_d2 = d, d2
            if best is not None:
                kept.append(Numbered(number=prev.number, detection=best, layer=prev.layer))
                unmatched.remove(best)
                next_misses[prev.number] = 0
            else:
                misses = self._misses.get(prev.number, 0) + 1
                if misses <= self.MAX_CONSECUTIVE_MISSES:
                    # Hold its last-seen position/number rather than dropping
                    # it (and thus its identity as "the" target) over what's
                    # very likely a transient miss, not a real pick-off.
                    kept.append(prev)
                    next_misses[prev.number] = misses
        self._misses = next_misses
        # Anything left unmatched is new (rare mid-unload) - append after the
        # current max number, ordered nearest-first from the last kept spot.
        # Each keeps its REAL current layer index (not a placeholder), since
        # it's already been confirmed to be within the top `max_layers` bands
        # above.
        if unmatched:
            pos = _center(kept[-1].detection) if kept else self.start_xy
            next_n = (max((k.number for k in kept), default=0)) + 1
            for d in order_layer(unmatched, pos):
                kept.append(Numbered(number=next_n, detection=d, layer=eligible_layer_of[id(d)]))
                next_n += 1

        kept.sort(key=lambda k: k.number)
        self._prev = kept
        return kept
