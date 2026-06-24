"""
test_coverage_grid.py — REV04 adaptive coverage tracker (methodology rev04 §3).

Pure / synthetic: feed table-plane (u,v) samples and assert the per-cell
sufficiency rule, the "go here" target selection, and the stop condition. No
hardware, no RNG — deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from Utils.gaze.coverage import CoverageGrid  # noqa: E402


def _fill_cell(g: CoverageGrid, base_uv, n, spread_mm):
    """Add n samples spread across `spread_mm` (bbox diagonal) inside one cell."""
    base = np.asarray(base_uv, dtype=float)
    # walk a short diagonal line so the bounding-box diagonal == spread_mm
    step = spread_mm / np.sqrt(2.0) / max(n - 1, 1)
    for k in range(n):
        g.add(base + np.array([k * step, k * step]))


def test_cell_of_and_center():
    g = CoverageGrid(cell_size_mm=50.0)
    assert g.cell_of([10.0, 10.0]) == (0, 0)
    assert g.cell_of([60.0, 10.0]) == (1, 0)
    assert g.cell_of([-1.0, 10.0]) == (-1, 0)
    np.testing.assert_allclose(g.cell_center((0, 0)), [25.0, 25.0])
    np.testing.assert_allclose(g.cell_center((1, 0)), [75.0, 25.0])


def test_insufficient_when_too_few_samples():
    g = CoverageGrid(cell_size_mm=50.0, min_samples=8, min_spread_mm=15.0)
    _fill_cell(g, [20, 20], n=4, spread_mm=30.0)   # spread ok, count short
    assert g.count((0, 0)) == 4
    assert not g.is_sufficient((0, 0))
    assert not g.done()


def test_insufficient_when_frozen_point():
    # enough samples but all at one spot → spread 0 → rejected (the frozen-hand guard)
    g = CoverageGrid(cell_size_mm=50.0, min_samples=8, min_spread_mm=15.0)
    for _ in range(12):
        g.add([20.0, 20.0])
    assert g.count((0, 0)) == 12
    assert g.spread((0, 0)) == 0.0
    assert not g.is_sufficient((0, 0))


def test_sufficient_when_count_and_spread_met():
    g = CoverageGrid(cell_size_mm=50.0, min_samples=8, min_spread_mm=15.0)
    _fill_cell(g, [20, 20], n=10, spread_mm=30.0)
    assert g.is_sufficient((0, 0))
    assert g.done()  # the only visited cell is sufficient


def test_next_target_points_at_weakest_then_none():
    g = CoverageGrid(cell_size_mm=50.0, min_samples=8, min_spread_mm=15.0)
    _fill_cell(g, [20, 20], n=10, spread_mm=30.0)   # (0,0) sufficient
    _fill_cell(g, [120, 20], n=3, spread_mm=30.0)   # (2,0) weakest (3 samples)
    _fill_cell(g, [70, 20], n=6, spread_mm=30.0)    # (1,0) partial (6 samples)
    tgt = g.next_target()
    np.testing.assert_allclose(tgt, g.cell_center((2, 0)))  # fewest samples wins
    # fill them both → no pending → None, and done
    _fill_cell(g, [120, 20], n=8, spread_mm=30.0)
    _fill_cell(g, [70, 20], n=8, spread_mm=30.0)
    assert g.next_target() is None
    assert g.done()


def test_empty_grid_not_done_and_no_target():
    g = CoverageGrid()
    assert not g.done()
    assert g.next_target() is None
    assert g.summary()["visited"] == 0


def test_next_target_tiebreak_deterministic():
    # two equally-weak cells → deterministic choice by cell index, stable across runs
    g = CoverageGrid(cell_size_mm=50.0, min_samples=8, min_spread_mm=15.0)
    _fill_cell(g, [220, 20], n=2, spread_mm=20.0)   # (4,0)
    _fill_cell(g, [20, 20], n=2, spread_mm=20.0)    # (0,0) — lower index, should win
    np.testing.assert_allclose(g.next_target(), g.cell_center((0, 0)))


def test_summary_counts():
    g = CoverageGrid(cell_size_mm=50.0, min_samples=8, min_spread_mm=15.0)
    _fill_cell(g, [20, 20], n=10, spread_mm=30.0)   # sufficient
    _fill_cell(g, [120, 20], n=3, spread_mm=30.0)   # partial
    s = g.summary()
    assert s["visited"] == 2 and s["sufficient"] == 1
    assert abs(s["fraction"] - 0.5) < 1e-12
    assert s["samples"] == 13


def test_constructor_validation():
    with pytest.raises(ValueError):
        CoverageGrid(cell_size_mm=0.0)
    with pytest.raises(ValueError):
        CoverageGrid(cell_size_mm=50.0, min_spread_mm=50.0)  # spread >= cell unsatisfiable
