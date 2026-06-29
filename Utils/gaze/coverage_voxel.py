"""
coverage_voxel.py — 3-D VOLUMETRIC coverage tracker for the AprilTag sweep (WS-4).

The 2-D analog (`Utils/gaze/coverage.py`, `CoverageGrid`) tiles the table plane into
square cells and reports, per region and in real time, where ``(u,v)`` coverage is
thin and when it is "enough". That is correct for a planar table sweep but blind to
the third axis: a sweep that thoroughly covers ``(x,y)`` at a single height reads
"done" even though the workspace VOLUME above/below the table is untouched.

`VoxelCoverage` is the volumetric analog: it bins 3-D end-effector points ``(x,y,z)``
(mm) into cubic voxels and applies the SAME min-samples / min-spread / cell-size
sufficiency rule extended to the z axis, so the operator knows when the workspace
VOLUME — not just one slice — is covered. It mirrors `CoverageGrid`'s public method
names (``add``, ``summary``, ``next_target``, ``done``, ``visited_cells``,
``sufficient_mask`` …) so the sweep can swap one for the other behind a single flag.

Pure / hardware-free / deterministic (no RNG, no OpenCV) — unit-tested without a rig.
The only display surface here is text: `z_slice_occupancy` / `status_text` give a
per-z-slice occupancy projection for the headless summary. A live 3-D OpenCV (or
3-D-lib) view is out of scope for WS-4 and noted in the sweep tool.

Model (extends rev04 §3 to 3-D):
  - The workspace is tiled into cubic voxels of ``cell_size_mm`` in ``(x,y,z)``.
  - A voxel becomes *visited* on its first sample; voxels never visited are "n/a"
    (the operator defines the task volume by where they sweep — a voxel the hand
    never reaches must not block ``done()``).
  - A visited voxel is *sufficient* when it has ``>= min_samples`` whose spatial
    spread (3-D bounding-box diagonal, now including z) is ``>= min_spread_mm`` —
    the spread guard rejects ``min_samples`` frozen at one point (a paused hand).
  - ``next_target()`` points at the weakest insufficient visited voxel (the gap to
    fill); ``done()`` is true once every visited voxel is sufficient.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

Voxel = Tuple[int, int, int]


class VoxelCoverage:
    """Bins 3-D ``(x,y,z)`` samples (mm) into cubic voxels and reports per-voxel
    sufficiency, the next region to cover, and overall completion. The 3-D analog
    of `CoverageGrid`; same sufficiency semantics, one extra axis."""

    def __init__(self, cell_size_mm: float = 50.0, min_samples: int = 8,
                 min_spread_mm: float = 15.0) -> None:
        if cell_size_mm <= 0:
            raise ValueError("cell_size_mm must be positive")
        if min_spread_mm >= cell_size_mm:
            # A spread requirement at/above the voxel edge is unsatisfiable inside one
            # voxel (the bbox diagonal is bounded by the voxel diagonal, but a single
            # edge length cannot be exceeded along one axis) — almost certainly a
            # misconfiguration, so fail fast, matching CoverageGrid's guard.
            raise ValueError("min_spread_mm must be smaller than cell_size_mm")
        self.cell_size_mm = float(cell_size_mm)
        self.min_samples = int(min_samples)
        self.min_spread_mm = float(min_spread_mm)
        self._cells: Dict[Voxel, List[np.ndarray]] = {}

    # ── ingest ────────────────────────────────────────────────────────────────
    def add(self, xyz) -> Voxel:
        """Record one accepted sample (an end-effector ``(x,y,z)`` in mm). Quality
        gating happens upstream in the sweep loop — only accepted samples reach
        here. Returns the voxel it landed in."""
        p = np.asarray(xyz, dtype=float).reshape(3)
        cell = self.cell_of(p)
        self._cells.setdefault(cell, []).append(p)
        return cell

    # ── voxel geometry ──────────────────────────────────────────────────────────
    def cell_of(self, xyz) -> Voxel:
        p = np.asarray(xyz, dtype=float).reshape(3)
        return (int(np.floor(p[0] / self.cell_size_mm)),
                int(np.floor(p[1] / self.cell_size_mm)),
                int(np.floor(p[2] / self.cell_size_mm)))

    def cell_center(self, cell: Voxel) -> np.ndarray:
        """Centre ``(x,y,z)`` of a voxel — the approximate region the operator is
        pointed toward (region-level, not a precise point)."""
        return (np.asarray(cell, dtype=float) + 0.5) * self.cell_size_mm

    # ── per-voxel stats ─────────────────────────────────────────────────────────
    def count(self, cell: Voxel) -> int:
        return len(self._cells.get(cell, ()))

    def spread(self, cell: Voxel) -> float:
        """3-D bounding-box diagonal (mm) of the voxel's samples; 0 for <2 samples.
        Extends the 2-D grid's spread to the z axis, so a hand that moved only within
        a thin (x,y) sheet inside the voxel still reads low spread."""
        pts = self._cells.get(cell)
        if not pts or len(pts) < 2:
            return 0.0
        arr = np.vstack(pts)
        return float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)))

    def is_sufficient(self, cell: Voxel) -> bool:
        return (self.count(cell) >= self.min_samples
                and self.spread(cell) >= self.min_spread_mm)

    # ── views ─────────────────────────────────────────────────────────────────
    def visited_cells(self) -> List[Voxel]:
        return list(self._cells.keys())

    def insufficient_cells(self) -> List[Voxel]:
        return [c for c in self._cells if not self.is_sufficient(c)]

    def cell_status(self) -> Dict[Voxel, str]:
        """Per-visited-voxel status: ``'partial'`` or ``'sufficient'`` (unvisited
        voxels are simply absent). The data a future 3-D view would colour by."""
        return {c: ("sufficient" if self.is_sufficient(c) else "partial")
                for c in self._cells}

    def sufficient_mask(self, points) -> np.ndarray:
        """Boolean mask over a sequence of ``(x,y,z)`` samples: True where the
        sample's voxel is sufficient ("green"). Computed once at sweep end so the
        solve can keep green-voxel samples only — excluding transit samples in
        still-partial voxels (mirrors `CoverageGrid.sufficient_mask`). Empty input
        → empty mask."""
        return np.array([self.is_sufficient(self.cell_of(p)) for p in points],
                        dtype=bool)

    # ── per-z-slice text projection (lightweight display) ───────────────────────
    def z_slice_occupancy(self) -> Dict[int, Dict[str, int]]:
        """Project the visited voxels onto the z axis: per z-index (``cell[2]``), the
        count of visited voxels, sufficient voxels, and samples in that horizontal
        slice. This is the 2-D-projection status the headless sweep prints — the
        analog of the 2-D box UI for the volumetric case, until a real 3-D view
        lands. Sorted by z (bottom → top)."""
        slices: Dict[int, Dict[str, int]] = {}
        for cell, pts in self._cells.items():
            k = cell[2]
            s = slices.setdefault(k, {"visited": 0, "sufficient": 0, "samples": 0})
            s["visited"] += 1
            s["samples"] += len(pts)
            if self.is_sufficient(cell):
                s["sufficient"] += 1
        return {k: slices[k] for k in sorted(slices)}

    def status_text(self) -> str:
        """One-line per-z-slice occupancy projection for the headless summary, e.g.
        ``z[-1]:2/3  z[0]:4/4`` (sufficient/visited voxels per slice). Empty until a
        voxel is visited."""
        parts = [f"z[{k}]:{s['sufficient']}/{s['visited']}"
                 for k, s in self.z_slice_occupancy().items()]
        return "  ".join(parts)

    # ── guidance + stopping ───────────────────────────────────────────────────
    def next_target(self) -> Optional[np.ndarray]:
        """Centre ``(x,y,z)`` of the visited voxel most in need of more samples — the
        "go here" indicator. The weakest voxel is the one with the fewest samples
        (ties broken deterministically by voxel index). Returns None when no voxel is
        visited yet, or when every visited voxel is already sufficient."""
        pending = self.insufficient_cells()
        if not pending:
            return None
        weakest = min(pending, key=lambda c: (self.count(c), c))
        return self.cell_center(weakest)

    def done(self) -> bool:
        """True once at least one voxel is visited and every visited voxel is
        sufficient. The operator defines the task volume by where they sweep, so an
        unvisited voxel never blocks completion."""
        return bool(self._cells) and not self.insufficient_cells()

    def summary(self) -> Dict[str, float]:
        """Compact progress for the summary line — same keys as
        `CoverageGrid.summary` so the sweep's report code is shared, with ``visited``/
        ``sufficient`` now counting voxels rather than cells."""
        visited = len(self._cells)
        sufficient = sum(1 for c in self._cells if self.is_sufficient(c))
        return {
            "visited": visited,
            "sufficient": sufficient,
            "fraction": (sufficient / visited) if visited else 0.0,
            "samples": sum(len(v) for v in self._cells.values()),
        }
