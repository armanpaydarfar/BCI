"""
control_view.py — live visual interface for the 3-D gaze→robot control test.

Makes the depth-free target math *visible* so the operator can build intuition
and debug a miss at a glance. Two panes, redrawn on every resolve:

  - **scene** (left): the head-camera frame with the gaze point, the selected
    object's mask outline, and the target pixel (footprint/centroid) marked — i.e.
    *what was looked at* and *what was chosen*.
  - **table top-down** (right): the calibrated workspace in table ``(u,v)`` mm —
    the whole library's coverage as faint dots, the resolved target point, and the
    nearest library pose it snapped to. This is where "outside the calibrated
    region" becomes obvious: you see the target land inside or outside the dots.

Plus a text strip (world target, nearest dist, anchor quality, gaze divergence).

The ``(u,v)`` layout math is the pure, unit-tested ``fit_box``/``BoxLayout`` from
``coverage_view``; only the cv2 draw + window here is rig-only.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from Utils.gaze.coverage_view import BoxLayout, fit_box

_COLOR_BG = (24, 24, 24)
_COLOR_TEXT = (235, 235, 235)
_COLOR_GAZE = (60, 200, 235)       # amber-cyan — where you looked
_COLOR_MASK = (90, 185, 90)        # green — selected object
_COLOR_TARGET = (60, 60, 235)      # red — the target (pixel + table point)
_COLOR_LIB = (90, 90, 90)          # grey — library coverage dots
_COLOR_NEAR = (235, 200, 60)       # blue-ish — nearest library pose


def table_uv_bounds(library_uv: np.ndarray, extra_uv: Optional[np.ndarray] = None):
    """``(u_lo, u_hi, v_lo, v_hi)`` enclosing the library (and an optional extra
    point, e.g. the current target so it stays on-canvas even just outside the
    coverage), with a small margin. None if there are no finite points."""
    pts = np.asarray(library_uv, dtype=float).reshape(-1, 2)
    if extra_uv is not None:
        e = np.asarray(extra_uv, dtype=float).reshape(-1, 2)
        pts = np.vstack([pts, e]) if pts.size else e
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] == 0:
        return None
    u_lo, v_lo = pts.min(axis=0)
    u_hi, v_hi = pts.max(axis=0)
    pad = 0.05 * max(u_hi - u_lo, v_hi - v_lo, 1.0)
    return float(u_lo - pad), float(u_hi + pad), float(v_lo - pad), float(v_hi + pad)


class ControlView:
    """cv2 dual-pane control visualiser. ``update`` redraws one resolve; ``close``
    tears down. Non-blocking — the operator drives from the terminal; this is a
    glance surface, refreshed each resolve."""

    def __init__(self, library_uv: np.ndarray, *,
                 width: int = 1280, height: int = 560, panel_h: int = 96,
                 window: str = "AprilTag 3-D control") -> None:
        import cv2  # lazy: only the display path needs it
        self._cv2 = cv2
        self.library_uv = np.asarray(library_uv, dtype=float).reshape(-1, 2)
        self.width, self.height, self.panel_h = width, height, panel_h
        self.scene_w = width // 2
        self.window = window
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, width, height)

    def update(self, scene_bgr: Optional[np.ndarray], *,
               gaze_px, mask_polygon, target_px, target_uv, nearest_uv,
               lines: Sequence[str]) -> None:
        cv2 = self._cv2
        canvas = np.full((self.height, self.width, 3), _COLOR_BG, dtype=np.uint8)
        scene_h = self.height - self.panel_h
        self._draw_scene(canvas, scene_bgr, gaze_px, mask_polygon, target_px, scene_h)
        self._draw_table(canvas, target_uv, nearest_uv, scene_h)
        # text strip across the bottom
        for k, text in enumerate(list(lines)[:3]):
            cv2.putText(canvas, text, (12, scene_h + 26 + k * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.imshow(self.window, canvas)
        cv2.waitKey(1)

    def _draw_scene(self, canvas, scene_bgr, gaze_px, mask_polygon, target_px, scene_h):
        cv2 = self._cv2
        x0, w = 0, self.scene_w
        if scene_bgr is None or scene_bgr.size == 0:
            cv2.putText(canvas, "no scene frame", (20, scene_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, _COLOR_TEXT, 2, cv2.LINE_AA)
            sc = None
        else:
            fh, fw = scene_bgr.shape[:2]
            sc = min(w / fw, scene_h / fh)
            dw, dh = int(fw * sc), int(fh * sc)
            ox, oy = x0 + (w - dw) // 2, (scene_h - dh) // 2
            canvas[oy:oy + dh, ox:ox + dw] = cv2.resize(scene_bgr, (dw, dh))

            def to_px(p):
                return int(ox + float(p[0]) * sc), int(oy + float(p[1]) * sc)

            if mask_polygon is not None and len(mask_polygon) >= 3:
                poly = np.array([to_px(p) for p in mask_polygon], dtype=np.int32)
                cv2.polylines(canvas, [poly], True, _COLOR_MASK, 2, cv2.LINE_AA)
            if gaze_px is not None:
                cv2.circle(canvas, to_px(gaze_px), 7, _COLOR_GAZE, 2, cv2.LINE_AA)
            if target_px is not None:
                tx, ty = to_px(target_px)
                cv2.drawMarker(canvas, (tx, ty), _COLOR_TARGET, cv2.MARKER_TILTED_CROSS,
                               16, 2, cv2.LINE_AA)
        cv2.putText(canvas, "scene: gaze (cyan) | object (green) | target (red)",
                    (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, _COLOR_TEXT, 1, cv2.LINE_AA)

    def _draw_table(self, canvas, target_uv, nearest_uv, scene_h):
        cv2 = self._cv2
        x0, w = self.scene_w, self.width - self.scene_w
        cv2.rectangle(canvas, (x0, 0), (self.width, scene_h), (32, 32, 32), -1)
        bounds = table_uv_bounds(self.library_uv, target_uv)
        if bounds is None:
            cv2.putText(canvas, "no library coverage", (x0 + 20, scene_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COLOR_TEXT, 1, cv2.LINE_AA)
            return
        layout = fit_box(*bounds, w, scene_h, 30)

        def to_px(uv):
            px, py = layout.uv_to_px(uv)
            return int(px + x0), int(py)

        for uv in self.library_uv:
            if np.all(np.isfinite(uv)):
                cv2.circle(canvas, to_px(uv), 1, _COLOR_LIB, -1)
        if nearest_uv is not None and np.all(np.isfinite(nearest_uv)):
            cv2.circle(canvas, to_px(nearest_uv), 7, _COLOR_NEAR, 2, cv2.LINE_AA)
        if target_uv is not None and np.all(np.isfinite(target_uv)):
            cv2.drawMarker(canvas, to_px(target_uv), _COLOR_TARGET, cv2.MARKER_TILTED_CROSS,
                           18, 2, cv2.LINE_AA)
        cv2.putText(canvas, "table top-down: library (grey) | target (red) | nearest (blue)",
                    (x0 + 12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, _COLOR_TEXT, 1, cv2.LINE_AA)

    def close(self) -> None:
        try:
            self._cv2.destroyWindow(self.window)
        except Exception:
            # Best-effort teardown on a bench tool already exiting.
            pass
