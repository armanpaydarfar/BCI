"""
vlm/seg_ops.py — pure segmentation post-processing helpers.

Behaviour-preserving extraction of the module-level segmentation helpers that
used to live at the top of vlm_service.py: the SegConstraints dataclass (WS4 F1
geometry/depth filter), its _opt_float coercion helper, the median-mask-depth
sampler, the constraint applicator, the live-overlay detection reducer, and the
gaze-hit recognizer. Pure functions over harmony_vlm Detection objects + numpy
arrays; no VLMService state, no UDP, no _log. Lives in a leaf module so the
service re-imports these by name (call sites unchanged) and so they're directly
unit-testable without standing up the service (tests/test_seg_constraints.py,
test_recognize.py).

cv2 is imported lazily inside the functions that need it, matching the original
inline definitions — keeps import-time cost off this leaf for callers that only
need SegConstraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Live-overlay detection cap defaults. Paint cost on the Linux side is O(N)
# full-canvas alpha blends (Utils/scene_overlay_renderer.py); capping N keeps the
# overlay budget bounded. These are the *defaults* — the values are configurable
# per-deployment via --overlay-top-k / --overlay-contain-ratio /
# --overlay-area-ratio (WS4 E2), resolved into VLMService instance attributes in
# main(). They live here because _filter_overlay_dets binds them as defaults.
_OVERLAY_TOP_K = 20
_OVERLAY_CONTAIN_RATIO = 0.85
_OVERLAY_AREA_RATIO = 0.5


@dataclass(frozen=True)
class SegConstraints:
    """Optional geometry/depth constraints biasing FastSAM toward small
    tabletop objects (WS4 F1). Every field defaults to the "off" sentinel
    (None), so the filter is a no-op until a threshold is set via CLI/config —
    preserving today's unfiltered segment/decide/segment_stream output.

    - max_area_ratio: drop dets whose bbox area exceeds this fraction of the
      frame (the parked max-area filter — kills door-frame+table blobs,
      walls/floors).
    - min_area_ratio: drop dets whose bbox area is below this fraction (specks).
    - solidity_min:   drop dets whose mask-area / bbox-area is below this
      (low-solidity merged/elongated blobs spanning multiple objects). Only
      enforced when a mask polygon is present.
    - depth_band:     (near_m, far_m) — drop dets whose median mask depth falls
      outside this near-field band. Only enforced where a depth map is
      available (the decide path); see _segment_stream_loop for why the stream
      stays geometry-only.
    - gaze_roi:       half-extent (fraction of frame width/height) of a window
      centred on the current gaze; drop dets whose bbox centre falls outside
      it. Only enforced where a gaze coordinate is available.
    """
    max_area_ratio: Optional[float] = None
    min_area_ratio: Optional[float] = None
    solidity_min:   Optional[float] = None
    depth_band:     Optional[tuple] = None
    gaze_roi:       Optional[float] = None

    def is_active(self) -> bool:
        return any(v is not None for v in (
            self.max_area_ratio, self.min_area_ratio, self.solidity_min,
            self.depth_band, self.gaze_roi,
        ))

    @classmethod
    def from_args(cls, args) -> "SegConstraints":
        band = getattr(args, "seg_depth_band", None)
        if band is not None:
            band = (float(band[0]), float(band[1]))
        return cls(
            max_area_ratio=_opt_float(getattr(args, "seg_max_area_ratio", None)),
            min_area_ratio=_opt_float(getattr(args, "seg_min_area_ratio", None)),
            solidity_min=_opt_float(getattr(args, "seg_solidity_min", None)),
            depth_band=band,
            gaze_roi=_opt_float(getattr(args, "seg_gaze_roi", None)),
        )


def _opt_float(v):
    return None if v is None else float(v)


def _median_mask_depth(det, depth_map) -> Optional[float]:
    """Median depth (metres) within a detection's mask, or the bbox-centre
    pixel depth when no mask is present. Mirrors compute_3d_waypoints'
    mask-depth logic (perception/object_detector.py:98-123) — same rasterise +
    valid-depth filter — kept local so the vendored module stays byte-identical
    to upstream. Returns None when no valid depth samples remain.
    """
    import cv2
    h, w = depth_map.shape[:2]
    if det.mask_polygon is not None and len(det.mask_polygon) >= 3:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [det.mask_polygon], 1)
        depths = depth_map[mask == 1]
    else:
        cx, cy = det.box_center
        ix = max(0, min(int(round(cx)), w - 1))
        iy = max(0, min(int(round(cy)), h - 1))
        depths = np.array([depth_map[iy, ix]])
    depths = depths[(depths > 0) & np.isfinite(depths) & (depths < 10.0)]
    if len(depths) == 0:
        return None
    return float(np.median(depths))


def _apply_seg_constraints(dets, frame_shape, constraints, depth_map=None, gaze_xy=None):
    """Drop detections that violate the optional F1 geometry/depth constraints.

    No-op (returns dets unchanged) when constraints is None / inactive, so the
    default service output matches the pre-F1 unfiltered behaviour. depth_band
    is enforced only when depth_map is given; gaze_roi only when gaze_xy is.
    """
    if constraints is None or not constraints.is_active() or not dets:
        return dets

    h, w = frame_shape[:2]
    frame_area = float(max(h * w, 1))
    kept = []
    for d in dets:
        x1, y1, x2, y2 = d.box_xyxy
        bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        ratio = bbox_area / frame_area

        if constraints.max_area_ratio is not None and ratio > constraints.max_area_ratio:
            continue
        if constraints.min_area_ratio is not None and ratio < constraints.min_area_ratio:
            continue
        if (constraints.solidity_min is not None and d.mask_polygon is not None
                and bbox_area > 0):
            import cv2
            mask_area = abs(cv2.contourArea(d.mask_polygon.reshape(-1, 1, 2).astype(np.int32)))
            if mask_area / bbox_area < constraints.solidity_min:
                continue
        if constraints.gaze_roi is not None and gaze_xy is not None:
            cx, cy = d.box_center
            if (abs(cx - gaze_xy[0]) > constraints.gaze_roi * w
                    or abs(cy - gaze_xy[1]) > constraints.gaze_roi * h):
                continue
        if constraints.depth_band is not None and depth_map is not None:
            md = _median_mask_depth(d, depth_map)
            near, far = constraints.depth_band
            if md is None or md < near or md > far:
                continue
        kept.append(d)
    return kept


def _filter_overlay_dets(dets, top_k=_OVERLAY_TOP_K,
                         contain_ratio=_OVERLAY_CONTAIN_RATIO,
                         area_ratio=_OVERLAY_AREA_RATIO):
    """Reduce detection count for the live segment-stream cache.

    Two passes:
      1. Top-K by confidence — caps live-overlay paint cost.
      2. Containment drop — if det B is mostly inside det A
         (intersection / area(B) > contain_ratio) AND
         area(B) < area_ratio × area(A), drop B. Targets
         FastSAM-everything's parent+children pattern (e.g. monitor
         co-segmented with icons on the monitor); the gaze-pointing
         workflow prefers the parent.

    Applied only to the seg-stream cache; segment/decide one-shots are
    unfiltered. top_k/contain_ratio/area_ratio default to the module constants
    but are overridden per-deployment via the overlay CLI/config knobs (E2).
    """
    if not dets:
        return dets

    if len(dets) > top_k:
        dets = sorted(dets, key=lambda d: float(d.confidence), reverse=True)[:top_k]

    def _bbox_area(d):
        x1, y1, x2, y2 = d.box_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _intersection(a, b):
        ax1, ay1, ax2, ay2 = a.box_xyxy
        bx1, by1, bx2, by2 = b.box_xyxy
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return (ix2 - ix1) * (iy2 - iy1)

    areas = [_bbox_area(d) for d in dets]
    drop = [False] * len(dets)
    for i, di in enumerate(dets):
        if drop[i] or areas[i] <= 0:
            continue
        for j, dj in enumerate(dets):
            if i == j or drop[j] or areas[j] <= 0:
                continue
            if areas[j] >= area_ratio * areas[i]:
                continue
            if _intersection(di, dj) / areas[j] > contain_ratio:
                drop[j] = True

    return [d for d, dropped in zip(dets, drop) if not dropped]


def _gaze_hit_recognize(dets, gaze_xy):
    """Pick the detection under the gaze point for the F5 recognizer: the
    smallest box containing the gaze (most specific object), else the nearest
    box centre within a fallback radius, else None.
    """
    if not dets:
        return None
    gx, gy = gaze_xy
    containing = []
    for d in dets:
        x1, y1, x2, y2 = d.box_xyxy
        if x1 <= gx <= x2 and y1 <= gy <= y2:
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            containing.append((area, d))
    if containing:
        return min(containing, key=lambda t: t[0])[1]
    best = None
    best_dist = 80.0  # px fallback radius from the gaze point to a box centre
    for d in dets:
        cx, cy = d.box_center
        dist = ((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best = d
    return best
