"""
control_view.py — live visual interface for the 3-D gaze→robot control test.

Makes the depth-free target math visible each resolve so the operator can build
intuition and debug a miss at a glance. Three panes + a text strip:

  - **scene** (left): head-cam frame with the gaze point, the SELECTED object's
    mask outline, and the chosen 3-D target reprojected onto the image (rides up
    the object in gaze_height mode) — what was looked at and what was chosen.
  - **table top-down** (upper right): the workspace in table ``(u,v)`` — the
    library's coverage as dots **coloured by height**, the resolved target, the
    nearest library pose it snapped to, and the **current EE** (from the EE tag)
    with an arrow to the target. "Outside the calibrated region" = target off the
    cloud.
  - **side / elevation** (lower right): the vertical cross-section through the
    object and the camera — the table line, the object's vertical line, the gaze
    ray descending from the camera, the target on that line, and the library's
    height coverage at this spot. This is where gaze_height's geometry is visible.

The ``(u,v)``/``(h,v)`` layout math is the pure ``fit_box``/``BoxLayout`` from
``coverage_view`` plus ``object_target.elevation_coords`` (all unit-tested); only
the cv2 draw + window here is rig-only.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from Utils.gaze.coverage_view import fit_box
from Utils.gaze.object_target import elevation_coords

_COLOR_BG = (24, 24, 24)
_COLOR_PANEL = (32, 32, 32)
_COLOR_TEXT = (235, 235, 235)
_COLOR_DIM = (150, 150, 150)
_COLOR_GAZE = (60, 200, 235)       # amber-cyan — where you looked
_COLOR_MASK = (90, 185, 90)        # green — selected object
_COLOR_TARGET = (60, 60, 235)      # red — the target
_COLOR_NEAR = (235, 200, 60)       # blue-ish — nearest library pose
_COLOR_EE = (245, 245, 245)        # white — current end-effector
_COLOR_RAY = (70, 160, 235)        # orange — the gaze ray
_COLOR_AXIS = (90, 90, 90)         # grey — table / vertical reference lines


def height_color(z: float, z_lo: float, z_hi: float):
    """BGR colour for a library point's height: teal (low) → red (high)."""
    if not np.isfinite(z) or (z_hi - z_lo) < 1e-6:
        return _COLOR_DIM
    t = float(np.clip((z - z_lo) / (z_hi - z_lo), 0.0, 1.0))
    lo = np.array([235.0, 180.0, 40.0])   # BGR teal
    hi = np.array([40.0, 80.0, 235.0])    # BGR red
    c = (1.0 - t) * lo + t * hi
    return (int(c[0]), int(c[1]), int(c[2]))


def table_uv_bounds(library_uv: np.ndarray, extra_uv: Optional[np.ndarray] = None):
    """``(u_lo, u_hi, v_lo, v_hi)`` enclosing the library (and an optional extra
    point so a just-outside target stays on-canvas), with a small margin. None if
    no finite points."""
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
    """cv2 three-pane control visualiser. ``update`` redraws one resolve; ``poll_key``
    reads the operator's command FROM the window (so the terminal stays log-only);
    ``close`` tears down."""

    def __init__(self, library_uv: np.ndarray, library_z: np.ndarray, *,
                 width: int = 1500, height: int = 640, panel_h: int = 88,
                 window: str = "AprilTag 3-D control") -> None:
        import cv2  # lazy: only the display path needs it
        self._cv2 = cv2
        self.library_uv = np.asarray(library_uv, dtype=float).reshape(-1, 2)
        self.library_z = np.asarray(library_z, dtype=float).reshape(-1)
        zf = self.library_z[np.isfinite(self.library_z)]
        self.z_lo = float(zf.min()) if zf.size else 0.0
        self.z_hi = float(zf.max()) if zf.size else 1.0
        self.width, self.height, self.panel_h = width, height, panel_h
        self.scene_w = int(width * 0.46)
        self.window = window
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, width, height)
        # First frame so the window is focusable for key input before the first
        # resolve — the operator drives from THIS window (cv2 keys); terminal = logs.
        ph = np.full((height, width, 3), _COLOR_BG, dtype=np.uint8)
        cv2.putText(ph, "AprilTag 3-D control - click this window, then drive with the keys "
                    "below", (24, height // 2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    _COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(ph, "ENTER resolve    g GO    r re-resolve    h home    q quit",
                    (24, height // 2 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COLOR_TEXT, 2,
                    cv2.LINE_AA)
        self._last_canvas = ph
        cv2.imshow(window, ph)
        cv2.waitKey(1)

    def poll_key(self, timeout_ms: int = 50) -> int:
        """Re-show the last frame and return a key pressed IN the window (cv2 highgui),
        or -1 if none within the timeout. This is what lets the operator drive from the
        window instead of the terminal (which steals window focus)."""
        cv2 = self._cv2
        if self._last_canvas is not None:
            cv2.imshow(self.window, self._last_canvas)
        return cv2.waitKey(max(1, int(timeout_ms))) & 0xFF

    def update(self, scene_bgr: Optional[np.ndarray], *,
               gaze_px, mask_polygon, target_px,
               target_uv, nearest_uv, current_ee_uv,
               base_world, target_world, cam_center_world, table_normal,
               lines: Sequence[str]) -> None:
        cv2 = self._cv2
        canvas = np.full((self.height, self.width, 3), _COLOR_BG, dtype=np.uint8)
        scene_h = self.height - self.panel_h
        self._draw_scene(canvas, scene_bgr, gaze_px, mask_polygon, target_px, scene_h)
        self._draw_table(canvas, target_uv, nearest_uv, current_ee_uv, scene_h)
        self._draw_side(canvas, base_world, target_world, cam_center_world,
                        table_normal, target_uv, scene_h)
        for k, text in enumerate(list(lines)[:3]):
            cv2.putText(canvas, text, (12, scene_h + 24 + k * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, _COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas, "ENTER resolve   g GO   r re-resolve   h home   q quit",
                    (12, self.height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    _COLOR_TEXT, 1, cv2.LINE_AA)
        self._last_canvas = canvas
        cv2.imshow(self.window, canvas)
        cv2.waitKey(1)

    # ── scene ────────────────────────────────────────────────────────────────
    def _draw_scene(self, canvas, scene_bgr, gaze_px, mask_polygon, target_px, scene_h):
        cv2 = self._cv2
        w = self.scene_w
        if scene_bgr is None or scene_bgr.size == 0:
            cv2.putText(canvas, "no scene frame", (20, scene_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, _COLOR_TEXT, 2, cv2.LINE_AA)
        else:
            fh, fw = scene_bgr.shape[:2]
            sc = min(w / fw, scene_h / fh)
            dw, dh = int(fw * sc), int(fh * sc)
            ox, oy = (w - dw) // 2, (scene_h - dh) // 2
            canvas[oy:oy + dh, ox:ox + dw] = cv2.resize(scene_bgr, (dw, dh))

            def to_px(p):
                return int(ox + float(p[0]) * sc), int(oy + float(p[1]) * sc)

            if mask_polygon is not None and len(mask_polygon) >= 3:
                poly = np.array([to_px(p) for p in mask_polygon], dtype=np.int32)
                cv2.polylines(canvas, [poly], True, _COLOR_MASK, 2, cv2.LINE_AA)
            if gaze_px is not None:
                cv2.circle(canvas, to_px(gaze_px), 7, _COLOR_GAZE, 2, cv2.LINE_AA)
            if target_px is not None:
                cv2.drawMarker(canvas, to_px(target_px), _COLOR_TARGET,
                               cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
        cv2.putText(canvas, "scene: gaze(cyan) object(green) target(red)",
                    (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, _COLOR_TEXT, 1, cv2.LINE_AA)

    # ── table top-down ───────────────────────────────────────────────────────
    def _draw_table(self, canvas, target_uv, nearest_uv, current_ee_uv, scene_h):
        cv2 = self._cv2
        x0, w, h = self.scene_w, self.width - self.scene_w, scene_h // 2
        cv2.rectangle(canvas, (x0, 0), (self.width, h), _COLOR_PANEL, -1)
        bounds = table_uv_bounds(self.library_uv, _stack(target_uv, current_ee_uv))
        if bounds is None:
            cv2.putText(canvas, "no library coverage", (x0 + 20, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COLOR_TEXT, 1, cv2.LINE_AA)
            return
        layout = fit_box(*bounds, w, h, 26)

        def to_px(uv):
            px, py = layout.uv_to_px(uv)
            return int(px + x0), int(py)

        for k, uv in enumerate(self.library_uv):
            if np.all(np.isfinite(uv)):
                z = self.library_z[k] if k < self.library_z.size else np.nan
                cv2.circle(canvas, to_px(uv), 1, height_color(z, self.z_lo, self.z_hi), -1)
        if current_ee_uv is not None and np.all(np.isfinite(current_ee_uv)):
            cv2.circle(canvas, to_px(current_ee_uv), 6, _COLOR_EE, 1, cv2.LINE_AA)
            if target_uv is not None and np.all(np.isfinite(target_uv)):
                cv2.arrowedLine(canvas, to_px(current_ee_uv), to_px(target_uv),
                                _COLOR_EE, 1, cv2.LINE_AA, tipLength=0.06)
        if nearest_uv is not None and np.all(np.isfinite(nearest_uv)):
            cv2.circle(canvas, to_px(nearest_uv), 7, _COLOR_NEAR, 2, cv2.LINE_AA)
        if target_uv is not None and np.all(np.isfinite(target_uv)):
            cv2.drawMarker(canvas, to_px(target_uv), _COLOR_TARGET,
                           cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
        cv2.putText(canvas, "top-down: lib by height  target(red) nearest(blue) EE(white)",
                    (x0 + 10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _COLOR_TEXT, 1, cv2.LINE_AA)

    # ── side / elevation ─────────────────────────────────────────────────────
    def _draw_side(self, canvas, base_world, target_world, cam_center_world,
                   table_normal, target_uv, scene_h):
        cv2 = self._cv2
        x0, w = self.scene_w, self.width - self.scene_w
        y0, h = scene_h // 2, scene_h - scene_h // 2
        cv2.rectangle(canvas, (x0, y0), (self.width, scene_h), (28, 28, 28), -1)
        cv2.putText(canvas, "side (elevation): table | object vertical | gaze ray | target",
                    (x0 + 10, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, _COLOR_TEXT, 1, cv2.LINE_AA)
        if any(v is None for v in (base_world, target_world, cam_center_world, table_normal)):
            return
        base = np.asarray(base_world, float)
        tgt_hv = elevation_coords(np.asarray(target_world, float), base, cam_center_world, table_normal)
        cam_hv = elevation_coords(np.asarray(cam_center_world, float), base, cam_center_world, table_normal)
        # Library heights at this spot (uv near the target) → the covered z band here.
        near_z = self._heights_near(target_uv)
        v_hi = max(self.z_hi, float(tgt_hv[1]), (near_z.max() if near_z.size else 0.0)) + 30.0
        h_hi = max(80.0, float(abs(tgt_hv[0])) + 40.0)   # horizontal extent (near_z is heights → v_hi)
        layout = fit_box(-40.0, h_hi, -30.0, v_hi, w, h, 24)

        def to_px(hv):
            px, py = layout.uv_to_px(hv)
            return int(px + x0), int(py + y0)

        # table line (v=0) and the object's vertical line (h=0)
        cv2.line(canvas, to_px((-40.0, 0.0)), to_px((h_hi, 0.0)), _COLOR_AXIS, 1, cv2.LINE_AA)
        cv2.line(canvas, to_px((0.0, 0.0)), to_px((0.0, v_hi)), _COLOR_AXIS, 1, cv2.LINE_AA)
        # height coverage at this spot: ticks along the vertical line
        for z in near_z:
            yy = to_px((0.0, float(z)))
            cv2.line(canvas, (yy[0] - 5, yy[1]), (yy[0] + 5, yy[1]),
                     height_color(float(z), self.z_lo, self.z_hi), 1, cv2.LINE_AA)
        # gaze ray: from the target toward the camera (descends from upper area)
        d = cam_hv - tgt_hv
        nrm = float(np.linalg.norm(d))
        if nrm > 1e-6:
            far = tgt_hv + d / nrm * (h_hi + v_hi)   # extend to the camera side
            cv2.line(canvas, to_px(tuple(tgt_hv)), to_px(tuple(far)), _COLOR_RAY, 1, cv2.LINE_AA)
        cv2.drawMarker(canvas, to_px(tuple(tgt_hv)), _COLOR_TARGET,
                       cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"height {tgt_hv[1]:.0f} mm",
                    (x0 + w - 150, scene_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    _COLOR_TEXT, 1, cv2.LINE_AA)

    def _heights_near(self, target_uv, radius_mm: float = 60.0) -> np.ndarray:
        """Library heights whose (u,v) is within ``radius_mm`` of the target — the
        height coverage at the spot being targeted."""
        if target_uv is None or not np.all(np.isfinite(target_uv)) or self.library_uv.size == 0:
            return np.empty(0)
        d = np.linalg.norm(self.library_uv - np.asarray(target_uv, float), axis=1)
        sel = (d <= radius_mm) & np.isfinite(self.library_z[:self.library_uv.shape[0]])
        return self.library_z[:self.library_uv.shape[0]][sel]

    def close(self) -> None:
        try:
            self._cv2.destroyWindow(self.window)
        except Exception:
            # Best-effort teardown on a bench tool already exiting.
            pass


def _stack(a, b):
    """Stack up to two optional (2,) points into an (N,2) array (for bounds)."""
    pts = [np.asarray(p, float).reshape(2) for p in (a, b)
           if p is not None and np.all(np.isfinite(np.asarray(p, float).reshape(-1)[:2]))]
    return np.array(pts) if pts else None
