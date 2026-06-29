"""
test_coverage_voxel.py — WS-4 3-D volumetric coverage tracker.

Pure / synthetic: feed end-effector (x,y,z) samples and assert 3-D voxel binning,
the per-voxel sufficiency rule, the "go here" target, and the stop condition. The
distinguishing property vs. the 2-D grid (`test_coverage_grid.py`) is the z axis:
points that share (x,y) but differ in z must land in DIFFERENT voxels, so covering
one slice does not mark the volume done. No hardware, no RNG — deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from Utils.gaze.coverage_voxel import VoxelCoverage  # noqa: E402


def _fill_voxel(v: VoxelCoverage, base_xyz, n, spread_mm):
    """Add n samples spread across `spread_mm` (3-D bbox diagonal) inside one voxel
    by walking a short body diagonal so the bounding-box diagonal == spread_mm."""
    base = np.asarray(base_xyz, dtype=float)
    step = spread_mm / np.sqrt(3.0) / max(n - 1, 1)
    for k in range(n):
        v.add(base + np.array([k * step, k * step, k * step]))


def test_cell_of_and_center_are_3d():
    v = VoxelCoverage(cell_size_mm=50.0)
    assert v.cell_of([10.0, 10.0, 10.0]) == (0, 0, 0)
    assert v.cell_of([60.0, 10.0, 10.0]) == (1, 0, 0)
    assert v.cell_of([10.0, 10.0, 60.0]) == (0, 0, 1)
    assert v.cell_of([-1.0, 10.0, -1.0]) == (-1, 0, -1)
    np.testing.assert_allclose(v.cell_center((0, 0, 0)), [25.0, 25.0, 25.0])
    np.testing.assert_allclose(v.cell_center((1, 0, 2)), [75.0, 25.0, 125.0])


def test_z_separated_points_land_in_different_voxels():
    # The property that distinguishes this from the 2-D grid: same (x,y), different z
    # → different voxels. Covering one z-slice must NOT complete the volume.
    v = VoxelCoverage(cell_size_mm=50.0, min_samples=8, min_spread_mm=15.0)
    _fill_voxel(v, [20, 20, 20], n=10, spread_mm=30.0)    # voxel (0,0,0)
    _fill_voxel(v, [20, 20, 220], n=3, spread_mm=30.0)    # voxel (0,0,4) — same (x,y)
    assert v.cell_of([20, 20, 20]) != v.cell_of([20, 20, 220])
    assert len(v.visited_cells()) == 2
    assert v.is_sufficient((0, 0, 0))
    assert not v.is_sufficient((0, 0, 4))
    assert not v.done()  # the upper slice is still thin


def test_single_dwelt_voxel_does_not_complete_volume():
    # A fully-sufficient single voxel is "done" only in the sense that every VISITED
    # voxel is covered — but the moment a second (z-separated) voxel is touched and
    # left partial, the volume is not done. Guards against the 2-D failure mode where
    # one dwelt patch reads as full coverage of the workspace.
    v = VoxelCoverage(cell_size_mm=50.0, min_samples=8, min_spread_mm=15.0)
    _fill_voxel(v, [20, 20, 20], n=12, spread_mm=30.0)
    assert v.done()  # the only visited voxel is sufficient
    v.add([20.0, 20.0, 300.0])  # touch a far z voxel once
    assert not v.done()


def test_insufficient_when_too_few_samples():
    v = VoxelCoverage(cell_size_mm=50.0, min_samples=8, min_spread_mm=15.0)
    _fill_voxel(v, [20, 20, 20], n=4, spread_mm=30.0)
    assert v.count((0, 0, 0)) == 4
    assert not v.is_sufficient((0, 0, 0))
    assert not v.done()


def test_insufficient_when_frozen_point():
    # enough samples but all at one spot → spread 0 → rejected (frozen-hand guard)
    v = VoxelCoverage(cell_size_mm=50.0, min_samples=8, min_spread_mm=15.0)
    for _ in range(12):
        v.add([20.0, 20.0, 20.0])
    assert v.count((0, 0, 0)) == 12
    assert v.spread((0, 0, 0)) == 0.0
    assert not v.is_sufficient((0, 0, 0))


def test_spread_uses_z_axis():
    # Samples identical in (x,y) but spread along z must register that spread — the
    # 3-D extension of the bounding-box-diagonal rule.
    v = VoxelCoverage(cell_size_mm=50.0, min_samples=2, min_spread_mm=15.0)
    v.add([20.0, 20.0, 20.0])
    v.add([20.0, 20.0, 40.0])  # 20 mm apart along z only
    assert v.spread((0, 0, 0)) == pytest.approx(20.0)
    assert v.is_sufficient((0, 0, 0))


def test_sufficient_when_count_and_spread_met():
    v = VoxelCoverage(cell_size_mm=50.0, min_samples=8, min_spread_mm=15.0)
    _fill_voxel(v, [20, 20, 20], n=10, spread_mm=30.0)
    assert v.is_sufficient((0, 0, 0))
    assert v.done()


def test_sufficient_mask_marks_only_green_voxel_samples():
    v = VoxelCoverage(cell_size_mm=50.0, min_samples=8, min_spread_mm=15.0)
    green_pts = [list(np.array([20.0, 20.0, 20.0]) + [k * 3.0, k * 3.0, k * 3.0])
                 for k in range(10)]
    # a z-separated, still-partial voxel (only 3 samples)
    partial_pts = [[20.0, 20.0, 220.0], [22.0, 22.0, 223.0], [24.0, 24.0, 226.0]]
    for p in green_pts + partial_pts:
        v.add(p)
    mask = v.sufficient_mask(green_pts + partial_pts)
    assert mask.dtype == bool and mask.shape == (13,)
    assert mask[:10].all() and not mask[10:].any()
    assert v.sufficient_mask([]).shape == (0,)


def test_next_target_points_at_weakest_then_none():
    v = VoxelCoverage(cell_size_mm=50.0, min_samples=8, min_spread_mm=15.0)
    _fill_voxel(v, [20, 20, 20], n=10, spread_mm=30.0)     # (0,0,0) sufficient
    _fill_voxel(v, [20, 20, 220], n=3, spread_mm=30.0)     # (0,0,4) weakest (3)
    _fill_voxel(v, [20, 20, 120], n=6, spread_mm=30.0)     # (0,0,2) partial (6)
    np.testing.assert_allclose(v.next_target(), v.cell_center((0, 0, 4)))
    _fill_voxel(v, [20, 20, 220], n=8, spread_mm=30.0)
    _fill_voxel(v, [20, 20, 120], n=8, spread_mm=30.0)
    assert v.next_target() is None
    assert v.done()


def test_next_target_tiebreak_deterministic():
    # two equally-weak voxels → deterministic choice by voxel index
    v = VoxelCoverage(cell_size_mm=50.0, min_samples=8, min_spread_mm=15.0)
    _fill_voxel(v, [20, 20, 220], n=2, spread_mm=20.0)   # (0,0,4)
    _fill_voxel(v, [20, 20, 20], n=2, spread_mm=20.0)    # (0,0,0) — lower index wins
    np.testing.assert_allclose(v.next_target(), v.cell_center((0, 0, 0)))


def test_empty_volume_not_done_and_no_target():
    v = VoxelCoverage()
    assert not v.done()
    assert v.next_target() is None
    assert v.summary()["visited"] == 0


def test_summary_counts_voxels():
    v = VoxelCoverage(cell_size_mm=50.0, min_samples=8, min_spread_mm=15.0)
    _fill_voxel(v, [20, 20, 20], n=10, spread_mm=30.0)    # sufficient
    _fill_voxel(v, [20, 20, 220], n=3, spread_mm=30.0)    # partial
    s = v.summary()
    assert s["visited"] == 2 and s["sufficient"] == 1
    assert abs(s["fraction"] - 0.5) < 1e-12
    assert s["samples"] == 13


def test_z_slice_occupancy_projection():
    # The lightweight display surface: per-z-slice occupancy counts.
    v = VoxelCoverage(cell_size_mm=50.0, min_samples=8, min_spread_mm=15.0)
    _fill_voxel(v, [20, 20, 20], n=10, spread_mm=30.0)     # z-slice 0, sufficient
    _fill_voxel(v, [120, 20, 20], n=3, spread_mm=30.0)     # z-slice 0, partial
    _fill_voxel(v, [20, 20, 220], n=10, spread_mm=30.0)    # z-slice 4, sufficient
    occ = v.z_slice_occupancy()
    assert list(occ.keys()) == [0, 4]  # sorted by z
    assert occ[0] == {"visited": 2, "sufficient": 1, "samples": 13}
    assert occ[4] == {"visited": 1, "sufficient": 1, "samples": 10}
    assert "z[0]:1/2" in v.status_text() and "z[4]:1/1" in v.status_text()


def test_constructor_validation():
    with pytest.raises(ValueError):
        VoxelCoverage(cell_size_mm=0.0)
    with pytest.raises(ValueError):
        VoxelCoverage(cell_size_mm=50.0, min_spread_mm=50.0)
