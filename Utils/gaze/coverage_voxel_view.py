"""
coverage_voxel_view.py — live 3-D (per-z-slice) coverage view for the WS-4
volumetric AprilTag sweep.

The 2-D ``CoverageBoxUI`` (``coverage_view.py``) draws one top-down table box; it
cannot show the z axis a volumetric sweep is about. This view renders the
``VoxelCoverage`` as a **stack of top-down (x,y) panels, one per occupied z-slice**
(highest z at the top), each tiled by that slice's voxels coloured
empty→partial→sufficient, with the live EE dot drawn in its slice and the
next-target voxel highlighted in its. All panels share one ``(x,y)`` layout so a
column of voxels lines up across heights and the operator can see *which heights*
are still thin — the visual analog of the headless ``status_text`` projection.

It mirrors ``CoverageBoxUI``'s interface (``wait_for_start``/``update``/
``should_quit``/``close``) so the sweep swaps it in behind ``--coverage-3d`` with no
loop changes. The cv2 draw is reused verbatim from the 2-D box helpers
(``fit_box``/``cell_rect_px``), so the only new code is the panel layout + slice
grouping — both **pure and unit-tested**. ``render`` builds the canvas without a
window (testable headless); only ``_show``/``wait_for_start`` touch the display.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from Utils.gaze.coverage_view import (
    _COLOR_BG,
    _COLOR_EE,
    _COLOR_GRID,
    _COLOR_PARTIAL,
    _COLOR_SUFFICIENT,
    _COLOR_TARGET,
    cell_rect_px,
    cells_uv_bounds,
    fit_box,
)

Voxel = Tuple[int, int, int]
Rect = Tuple[int, int, int, int]      # (x, y, w, h) in canvas pixels

_HEADER_H = 34
_FOOTER_H = 24
_GAP = 8
_MAX_COLS = 4


def panel_grid(n: int, canvas_w: int, canvas_h: int, *, top_h: int = _HEADER_H,
               bottom_h: int = _FOOTER_H, gap: int = _GAP,
               max_cols: int = _MAX_COLS) -> List[Rect]:
    """Tile ``n`` z-slice panels into the canvas interior (below the header, above
    the footer), left-to-right then top-to-bottom, ``max_cols`` per row. Returns one
    ``(x, y, w, h)`` per panel, all inside the canvas and non-overlapping. Pure."""
    if n <= 0:
        return []
    cols = min(n, max_cols)
    rows = -(-n // cols)                       # ceil
    area_w = canvas_w
    area_h = max(canvas_h - top_h - bottom_h, 1)
    pw = (area_w - (cols + 1) * gap) / cols
    ph = (area_h - (rows + 1) * gap) / rows
    rects: List[Rect] = []
    for idx in range(n):
        r, c = divmod(idx, cols)
        x = gap + c * (pw + gap)
        y = top_h + gap + r * (ph + gap)
        rects.append((int(round(x)), int(round(y)),
                      max(int(round(pw)), 1), max(int(round(ph)), 1)))
    return rects


def voxels_by_z(visited: List[Voxel], status: Dict[Voxel, str]
                ) -> Dict[int, List[Tuple[Tuple[int, int], str]]]:
    """Group visited voxels by z-index → list of ``((i, j), status)`` for that
    slice. ``status`` is ``VoxelCoverage.cell_status()``; a voxel missing from it
    defaults to ``'partial'``. Pure."""
    by_z: Dict[int, List[Tuple[Tuple[int, int], str]]] = {}
    for (i, j, k) in visited:
        by_z.setdefault(int(k), []).append(((int(i), int(j)),
                                            status.get((i, j, k), "partial")))
    return by_z


def _finite_xyz(p) -> Optional[np.ndarray]:
    if p is None:
        return None
    a = np.asarray(p, dtype=float).reshape(-1)
    if a.size < 3 or not np.all(np.isfinite(a[:3])):
        return None
    return a[:3]


class VoxelCoverageBoxUI:
    """Live per-z-slice coverage window for the volumetric sweep. Same interface as
    ``CoverageBoxUI`` (``wait_for_start``/``update``/``should_quit``/``close``) so the
    sweep loop is agnostic to which it holds."""

    def __init__(self, cell_size_mm: float, *, audio: bool = False,
                 width: int = 960, height: int = 640, margin_px: int = 14,
                 window: str = "WS-4 3-D coverage") -> None:
        import cv2  # lazy: only the display path needs it
        self._cv2 = cv2
        self.cell_size_mm = float(cell_size_mm)
        self.audio = bool(audio)   # accepted for parity; per-direction TTS is 2-D only
        self.width, self.height, self.margin_px = width, height, margin_px
        self.window = window
        self._quit = False
        self._window_open = False

    # ── display plumbing (window I/O; not unit-tested) ─────────────────────────
    def _ensure_window(self) -> None:
        if not self._window_open:
            cv2 = self._cv2
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window, self.width, self.height)
            self._window_open = True

    def _show(self, canvas: np.ndarray) -> int:
        self._ensure_window()
        self._cv2.imshow(self.window, canvas)
        return self._cv2.waitKey(1) & 0xFF

    # ── rendering (pure of window I/O → headless-testable) ─────────────────────
    def render(self, grid, cur_xyz, target_xyz) -> np.ndarray:
        """Build the full canvas for one frame WITHOUT touching a window. Draws the
        header, a top-down panel per occupied z-slice (highest z first), the
        per-slice cells coloured by sufficiency, the next-target voxel, and the live
        EE dot in its slice. Returns the BGR canvas."""
        cv2 = self._cv2
        W, H = self.width, self.height
        canvas = np.full((H, W, 3), _COLOR_BG, dtype=np.uint8)

        visited = [tuple(int(v) for v in c) for c in grid.visited_cells()]
        status = grid.cell_status()
        s = grid.summary()
        cur = _finite_xyz(cur_xyz)
        cur_cell = grid.cell_of(cur) if cur is not None else None
        cur_k = cur_cell[2] if cur_cell is not None else None
        header = (f"voxels {int(s['sufficient'])}/{int(s['visited'])} sufficient   "
                  f"samples {int(s['samples'])}   "
                  + ("COMPLETE" if grid.done() else "sweeping…")
                  + (f"   EE z-slice {cur_k}" if cur_k is not None else ""))
        cv2.putText(canvas, header, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    _COLOR_EE, 1, cv2.LINE_AA)
        footer = "hand-guide slowly  |  VARY HEIGHT (cover every z-slice)  |  [SPACE] stop & save"
        cv2.putText(canvas, footer, (12, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    _COLOR_EE, 1, cv2.LINE_AA)

        if not visited:
            return canvas

        by_z = voxels_by_z(visited, status)
        ks = sorted(by_z, reverse=True)                       # highest z at the top
        panels = panel_grid(len(ks), W, H)
        # One shared (x,y) layout (projected i,j over ALL slices) so a voxel column
        # lines up across heights and the EE dot moves consistently between panels.
        all_ij = [(i, j) for (i, j, _k) in visited]
        bounds = cells_uv_bounds(all_ij, self.cell_size_mm)
        target = _finite_xyz(target_xyz)
        target_cell = grid.cell_of(target) if target is not None else None

        for rect, k in zip(panels, ks):
            self._draw_slice(canvas, rect, bounds, by_z[k], k,
                             target_cell, cur_cell)
        return canvas

    def _draw_slice(self, canvas, rect: Rect, bounds, cells_status, k: int,
                    target_cell, cur_cell) -> None:
        cv2 = self._cv2
        px, py, pw, ph = rect
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), _COLOR_GRID, 1)
        label = f"z[{k}]  Z~{(k + 0.5) * self.cell_size_mm:.0f}mm"
        suff = sum(1 for _ij, st in cells_status if st == "sufficient")
        cv2.putText(canvas, f"{label}  {suff}/{len(cells_status)}",
                    (px + 4, py + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    _COLOR_EE, 1, cv2.LINE_AA)
        if bounds is None:
            return
        # Panel-local layout (leave a strip at the top for the label).
        lay = fit_box(*bounds, pw, max(ph - 18, 1), self.margin_px)

        def at(cell2):
            x0, y0, x1, y1 = cell_rect_px(lay, cell2, self.cell_size_mm)
            return (px + x0, py + 18 + y0, px + x1, py + 18 + y1)

        for (ij, st) in cells_status:
            x0, y0, x1, y1 = at(ij)
            color = _COLOR_SUFFICIENT if st == "sufficient" else _COLOR_PARTIAL
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, -1)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), _COLOR_GRID, 1)
        if target_cell is not None and target_cell[2] == k:
            x0, y0, x1, y1 = at((target_cell[0], target_cell[1]))
            cv2.rectangle(canvas, (x0, y0), (x1, y1), _COLOR_TARGET, 3)
        if cur_cell is not None and cur_cell[2] == k:
            x0, y0, x1, y1 = at((cur_cell[0], cur_cell[1]))
            cv2.circle(canvas, ((x0 + x1) // 2, (y0 + y1) // 2), 5, _COLOR_EE, -1)

    # ── interface parity with CoverageBoxUI ────────────────────────────────────
    def prompt(self, title: str, sublines) -> bool:
        """Centered on-window prompt; blocks until SPACE (True) or 'q'/closed
        (False). Mirrors CoverageBoxUI.prompt so the start gate is on the visual
        interface, not the terminal."""
        cv2 = self._cv2
        rows = [(title, 0.95, 2)] + [(s, 0.6, 1) for s in sublines]
        while True:
            canvas = np.full((self.height, self.width, 3), _COLOR_BG, dtype=np.uint8)
            for k, (text, scale, thick) in enumerate(rows):
                y = self.height // 2 - 28 + k * 36
                cv2.putText(canvas, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                            _COLOR_EE, thick, cv2.LINE_AA)
            self._ensure_window()
            cv2.imshow(self.window, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == ord(" "):
                return True
            if key == ord("q"):
                self._quit = True
                return False
            if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                self._quit = True
                return False

    def wait_for_start(self) -> bool:
        """Start gate: block until the operator has positioned the freed arm and
        presses SPACE (so the transit into the start pose never enters the library)."""
        return self.prompt("POSITION THE ARM",
                            ["press SPACE to START the volumetric sweep",
                             "cover the workspace at MANY heights",
                             "q aborts (arm re-locks + homes)"])

    def update(self, grid, cur_xyz, target_xyz) -> None:
        if self._show(self.render(grid, cur_xyz, target_xyz)) in (ord(" "), ord("q")):
            self._quit = True

    def should_quit(self) -> bool:
        return self._quit

    def close(self) -> None:
        try:
            if self._window_open:
                self._cv2.destroyWindow(self.window)
        except Exception:
            # Best-effort teardown on a bench tool already exiting; the window may
            # already be gone (operator closed it).
            pass
