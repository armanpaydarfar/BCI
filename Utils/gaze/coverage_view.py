"""
coverage_view.py — top-down BOX coverage view for the REV04 swept calibration
(methodology rev04 §3).

A glanceable, standalone OpenCV window (like ``neon_viewer.py``) that draws the
swept table region as a **rectangle in table-plane ``(u,v)``** — a clean schematic,
*not* an overlay on the moving scene frame (rev04 §3, Arman 2026-06-24). The box is
tiled by the `CoverageGrid` cells coloured empty→partial→sufficient, with the
current EE position as a dot and the next-target cell highlighted so the operator
knows where to move. Because the operator wears the Neon and may not watch the
screen, optional spoken cues announce the target direction and completion.

The **layout math is pure** (`BoxLayout` — table ``(u,v)`` mm → canvas pixels, with
``+v`` drawn up) so it is unit-tested without a display; the cv2 draw + window
handling in `CoverageBoxUI` is the thin hardware/display part exercised only on the
rig. The box auto-scales each frame to the cells visited so far (the reachable
region grows as the operator sweeps, rev04 §3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from Utils.gaze.coverage import Cell, CoverageGrid

# BGR colours for the cell states + markers (OpenCV is BGR, not RGB).
_COLOR_PARTIAL = (40, 140, 235)     # amber — visited, not yet sufficient
_COLOR_SUFFICIENT = (80, 185, 90)   # green — enough coverage
_COLOR_TARGET = (60, 60, 235)       # red-ish — the "go here" highlight
_COLOR_EE = (245, 245, 245)         # near-white — current EE dot
_COLOR_GRID = (70, 70, 70)          # faint cell outlines
_COLOR_BG = (24, 24, 24)


@dataclass
class BoxLayout:
    """Affine map from table-plane ``(u,v)`` mm to canvas pixels, aspect-preserving
    with ``+v`` drawn upward (image y grows downward, so v is flipped). Built from
    the ``(u,v)`` extent currently being covered."""
    scale: float       # px per mm
    ox: float          # pixel x of u_lo
    oy: float          # pixel y of v_hi (top edge, since v flips)
    u_lo: float
    v_hi: float

    def uv_to_px(self, uv) -> Tuple[int, int]:
        u, v = float(uv[0]), float(uv[1])
        px = self.ox + (u - self.u_lo) * self.scale
        py = self.oy + (self.v_hi - v) * self.scale
        return int(round(px)), int(round(py))


def fit_box(u_lo: float, u_hi: float, v_lo: float, v_hi: float,
            canvas_w: int, canvas_h: int, margin_px: int) -> BoxLayout:
    """Aspect-preserving fit of the ``(u,v)`` rectangle into the canvas interior
    (canvas minus ``margin_px`` on each side). Degenerate (zero-span) extents — a
    single visited cell — get a 1 mm floor so ``scale`` stays finite and the cell
    renders centred rather than div-by-zero."""
    span_u = max(u_hi - u_lo, 1.0)
    span_v = max(v_hi - v_lo, 1.0)
    avail_w = max(canvas_w - 2 * margin_px, 1)
    avail_h = max(canvas_h - 2 * margin_px, 1)
    scale = min(avail_w / span_u, avail_h / span_v)
    # Centre the (possibly letter-boxed) box in the available area.
    used_w = span_u * scale
    used_h = span_v * scale
    ox = margin_px + (avail_w - used_w) / 2.0
    oy = margin_px + (avail_h - used_h) / 2.0
    return BoxLayout(scale=scale, ox=ox, oy=oy, u_lo=u_lo, v_hi=v_hi)


def cells_uv_bounds(cells: List[Cell], cell_size_mm: float, pad_cells: int = 1
                    ) -> Optional[Tuple[float, float, float, float]]:
    """``(u_lo, u_hi, v_lo, v_hi)`` mm spanning the given cells padded by
    ``pad_cells`` on every side (so the box has a margin and the next-target cell
    just outside the visited cloud is visible). None if no cells yet."""
    if not cells:
        return None
    ii = [c[0] for c in cells]
    jj = [c[1] for c in cells]
    i_lo, i_hi = min(ii) - pad_cells, max(ii) + pad_cells + 1
    j_lo, j_hi = min(jj) - pad_cells, max(jj) + pad_cells + 1
    return (i_lo * cell_size_mm, i_hi * cell_size_mm,
            j_lo * cell_size_mm, j_hi * cell_size_mm)


def cell_rect_px(layout: BoxLayout, cell: Cell, cell_size_mm: float
                 ) -> Tuple[int, int, int, int]:
    """Pixel rectangle ``(x0, y0, x1, y1)`` (top-left, bottom-right) for a cell."""
    u0, v0 = cell[0] * cell_size_mm, cell[1] * cell_size_mm
    u1, v1 = u0 + cell_size_mm, v0 + cell_size_mm
    x0, y_bottom = layout.uv_to_px((u0, v0))
    x1, y_top = layout.uv_to_px((u1, v1))
    return min(x0, x1), min(y_top, y_bottom), max(x0, x1), max(y_top, y_bottom)


def target_direction(cur_uv, target_uv) -> str:
    """Coarse spoken cue: the dominant compass move from the current EE ``(u,v)`` to
    the target cell centre. ``+u`` → "right", ``+v`` → "up" (matching the box's
    drawn orientation). Returns "" when either point is missing."""
    if cur_uv is None or target_uv is None:
        return ""
    du = float(target_uv[0]) - float(cur_uv[0])
    dv = float(target_uv[1]) - float(cur_uv[1])
    if abs(du) >= abs(dv):
        return "right" if du > 0 else "left"
    return "up" if dv > 0 else "down"


def speak(text: str) -> None:
    """Best-effort, non-blocking spoken cue via ``spd-say``/``espeak`` if present.
    Audio is a non-critical solo-operator aid (rev04 §3); a missing TTS binary or a
    spawn failure is swallowed deliberately so it never disrupts the sweep loop."""
    import shutil
    import subprocess
    for exe, flag in (("spd-say", None), ("espeak", None)):
        if shutil.which(exe):
            try:
                cmd = [exe, text] if flag is None else [exe, flag, text]
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            except OSError:
                pass  # TTS is optional; never let it break the sweep
            return


class CoverageBoxUI:
    """OpenCV coverage box (rev04 §3). Thin display wrapper over the pure layout
    helpers; ``update`` redraws, ``should_quit`` polls the window for 'q', ``close``
    tears the window down. Optional audio announces the target direction (throttled
    to target-cell changes) and completion."""

    def __init__(self, cell_size_mm: float, *, audio: bool = False,
                 width: int = 720, height: int = 540, margin_px: int = 48,
                 window: str = "REV04 coverage") -> None:
        import cv2  # lazy: only the display path needs it
        self._cv2 = cv2
        self.cell_size_mm = float(cell_size_mm)
        self.audio = bool(audio)
        self.width, self.height, self.margin_px = width, height, margin_px
        self.window = window
        self._quit = False
        self._last_target_cell: Optional[Cell] = None
        self._announced_done = False
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, width, height)

    def wait_for_start(self) -> bool:
        """Start gate (rev04 §3, operator 2026-06-24): block — drawing an on-screen
        prompt so the operator can position the freed arm — until they press SPACE
        to begin (returns True) or 'q' / close the window to abort (returns False).
        No sample is recorded until this returns True, so the transit into the start
        pose never enters the library."""
        cv2 = self._cv2
        lines = [("POSITION THE ARM", 0.95, 2),
                 ("press SPACE to START the sweep", 0.6, 1),
                 ("q aborts (arm re-locks + homes)", 0.55, 1)]
        while True:
            canvas = np.full((self.height, self.width, 3), _COLOR_BG, dtype=np.uint8)
            for k, (text, scale, thick) in enumerate(lines):
                y = self.height // 2 - 28 + k * 36
                cv2.putText(canvas, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                            _COLOR_EE, thick, cv2.LINE_AA)
            cv2.imshow(self.window, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == ord(" "):
                return True
            if key == ord("q"):
                self._quit = True
                return False
            # Window closed via the title-bar X → abort rather than spin forever.
            if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                self._quit = True
                return False

    def update(self, grid: CoverageGrid, cur_uv, target_uv) -> None:
        cv2 = self._cv2
        canvas = np.full((self.height, self.width, 3), _COLOR_BG, dtype=np.uint8)
        cells = grid.visited_cells()
        bounds = cells_uv_bounds(cells, self.cell_size_mm)
        if bounds is not None:
            layout = fit_box(*bounds, self.width, self.height, self.margin_px)
            status = grid.cell_status()
            target_cell = grid.cell_of(target_uv) if target_uv is not None else None
            for cell in cells:
                x0, y0, x1, y1 = cell_rect_px(layout, cell, self.cell_size_mm)
                color = (_COLOR_SUFFICIENT if status.get(cell) == "sufficient"
                         else _COLOR_PARTIAL)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), color, -1)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), _COLOR_GRID, 1)
            if target_cell is not None:
                x0, y0, x1, y1 = cell_rect_px(layout, target_cell, self.cell_size_mm)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), _COLOR_TARGET, 3)
            if cur_uv is not None:
                cx, cy = layout.uv_to_px(cur_uv)
                cv2.circle(canvas, (cx, cy), 6, _COLOR_EE, -1)

        s = grid.summary()
        header = (f"cells {int(s['sufficient'])}/{int(s['visited'])} sufficient   "
                  f"samples {int(s['samples'])}   "
                  + ("COMPLETE" if grid.done() else "sweeping…"))
        cv2.putText(canvas, header, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    _COLOR_EE, 1, cv2.LINE_AA)

        # On-screen operator instructions (rev04 §3, operator 2026-06-24): the
        # operator wears the Neon and cannot watch the terminal during the sweep.
        footer = "hand-guide slowly  |  move toward the RED cell  |  [q] stop & save"
        cv2.putText(canvas, footer, (12, self.height - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR_EE, 1, cv2.LINE_AA)

        cv2.imshow(self.window, canvas)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            self._quit = True

        if self.audio:
            self._audio_cues(grid, cur_uv, target_uv)

    def _audio_cues(self, grid: CoverageGrid, cur_uv, target_uv) -> None:
        if grid.done():
            if not self._announced_done:
                speak("coverage complete")
                self._announced_done = True
            return
        target_cell = grid.cell_of(target_uv) if target_uv is not None else None
        if target_cell is not None and target_cell != self._last_target_cell:
            direction = target_direction(cur_uv, target_uv)
            if direction:
                speak(direction)
            self._last_target_cell = target_cell

    def should_quit(self) -> bool:
        return self._quit

    def close(self) -> None:
        try:
            self._cv2.destroyWindow(self.window)
        except Exception:
            # Best-effort teardown on a bench tool already exiting; the window may
            # already be gone (e.g. operator closed it).
            pass
