"""
coverage.py — adaptive workspace-coverage tracker for the REV04 planar gaze↔robot
calibration sweep (methodology rev04 §3).

During the continuous sweep the operator hand-guides the end-effector across the
table while a 20 Hz loop feeds each accepted sample's table-plane ``(u,v)`` here.
This module decides, **per region and in real time**, where coverage is thin and
when it is "enough" — driving the box UI's "go here" indicator and the sweep's
stop condition. Pure / hardware-free / deterministic (no RNG), so it is unit-tested
without a rig.

Model (rev04 §3):
  - The table is tiled into square cells of ``cell_size_mm`` in ``(u,v)``.
  - A cell becomes *visited* on its first sample; cells never visited are "n/a"
    (the operator defines the task extent by where they sweep — a cell the hand
    never reaches must not block ``done()``, rev04 §9).
  - A visited cell is *sufficient* when it has ``>= min_samples`` whose spatial
    spread (bounding-box diagonal) is ``>= min_spread_mm`` — the spread guard
    rejects ``min_samples`` frozen at one point (e.g. a paused hand at 20 Hz).
  - ``next_target()`` points at the weakest insufficient visited cell (the gap to
    fill); ``done()`` is true once every visited cell is sufficient.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

Cell = Tuple[int, int]


class CoverageGrid:
    """Bins planar ``(u,v)`` samples (mm) into a fixed grid and reports per-cell
    sufficiency, the next region to cover, and overall completion."""

    def __init__(self, cell_size_mm: float = 50.0, min_samples: int = 8,
                 min_spread_mm: float = 15.0) -> None:
        if cell_size_mm <= 0:
            raise ValueError("cell_size_mm must be positive")
        if min_spread_mm >= cell_size_mm:
            # A spread requirement at/above the cell size is unsatisfiable inside one
            # cell — almost certainly a misconfiguration, so fail fast.
            raise ValueError("min_spread_mm must be smaller than cell_size_mm")
        self.cell_size_mm = float(cell_size_mm)
        self.min_samples = int(min_samples)
        self.min_spread_mm = float(min_spread_mm)
        self._cells: Dict[Cell, List[np.ndarray]] = {}

    # ── ingest ────────────────────────────────────────────────────────────────
    def add(self, uv) -> Cell:
        """Record one accepted sample (a table-plane ``(u,v)`` in mm). Quality
        gating happens upstream in the sweep loop — only accepted samples reach
        here. Returns the cell it landed in."""
        p = np.asarray(uv, dtype=float).reshape(2)
        cell = self.cell_of(p)
        self._cells.setdefault(cell, []).append(p)
        return cell

    # ── cell geometry ─────────────────────────────────────────────────────────
    def cell_of(self, uv) -> Cell:
        p = np.asarray(uv, dtype=float).reshape(2)
        return (int(np.floor(p[0] / self.cell_size_mm)),
                int(np.floor(p[1] / self.cell_size_mm)))

    def cell_center(self, cell: Cell) -> np.ndarray:
        """Centre ``(u,v)`` of a cell — the approximate region the UI points the
        operator toward (rev04 §3, region-level not a precise point)."""
        return (np.asarray(cell, dtype=float) + 0.5) * self.cell_size_mm

    # ── per-cell stats ────────────────────────────────────────────────────────
    def count(self, cell: Cell) -> int:
        return len(self._cells.get(cell, ()))

    def spread(self, cell: Cell) -> float:
        """Bounding-box diagonal (mm) of the cell's samples; 0 for <2 samples."""
        pts = self._cells.get(cell)
        if not pts or len(pts) < 2:
            return 0.0
        arr = np.vstack(pts)
        return float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)))

    def is_sufficient(self, cell: Cell) -> bool:
        return (self.count(cell) >= self.min_samples
                and self.spread(cell) >= self.min_spread_mm)

    # ── views ─────────────────────────────────────────────────────────────────
    def visited_cells(self) -> List[Cell]:
        return list(self._cells.keys())

    def insufficient_cells(self) -> List[Cell]:
        return [c for c in self._cells if not self.is_sufficient(c)]

    def cell_status(self) -> Dict[Cell, str]:
        """Per-visited-cell status for the box UI: ``'partial'`` or ``'sufficient'``
        (unvisited cells are simply absent → drawn empty)."""
        return {c: ("sufficient" if self.is_sufficient(c) else "partial")
                for c in self._cells}

    # ── guidance + stopping ───────────────────────────────────────────────────
    def next_target(self) -> Optional[np.ndarray]:
        """Centre ``(u,v)`` of the visited cell most in need of more samples — the
        UI's "go here" indicator. The weakest cell is the one with the fewest
        samples (ties broken deterministically by cell index). Returns None when no
        cell is visited yet, or when every visited cell is already sufficient."""
        pending = self.insufficient_cells()
        if not pending:
            return None
        weakest = min(pending, key=lambda c: (self.count(c), c))
        return self.cell_center(weakest)

    def done(self) -> bool:
        """True once at least one cell is visited and every visited cell is
        sufficient. The operator defines the task extent by where they sweep, so an
        unvisited cell never blocks completion (rev04 §9)."""
        return bool(self._cells) and not self.insufficient_cells()

    def summary(self) -> Dict[str, float]:
        """Compact progress for the UI header."""
        visited = len(self._cells)
        sufficient = sum(1 for c in self._cells if self.is_sufficient(c))
        return {
            "visited": visited,
            "sufficient": sufficient,
            "fraction": (sufficient / visited) if visited else 0.0,
            "samples": sum(len(v) for v in self._cells.values()),
        }
