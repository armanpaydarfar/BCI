"""
test_coverage_voxel_view.py — WS-4 per-z-slice coverage view.

Pins the pure layout/grouping helpers (no display) and a headless smoke test of
``render`` (OpenCV drawing on a numpy array needs no window). The window I/O
(``_show``/``wait_for_start``) is exercised only on the rig, like the 2-D box.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Utils.gaze.coverage_voxel import VoxelCoverage  # noqa: E402
from Utils.gaze.coverage_voxel_view import (  # noqa: E402
    panel_grid,
    voxels_by_z,
)


def _rects_overlap(a, b) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def test_panel_grid_empty():
    assert panel_grid(0, 960, 640) == []


def test_panel_grid_within_canvas_and_disjoint():
    W, H, top, bottom = 960, 640, 34, 24
    for n in (1, 2, 4, 5, 7):
        rects = panel_grid(n, W, H, top_h=top, bottom_h=bottom)
        assert len(rects) == n
        for (x, y, w, h) in rects:
            assert x >= 0 and y >= top and w > 0 and h > 0
            assert x + w <= W
            assert y + h <= H - bottom + 1            # within the interior band
        for i in range(n):
            for j in range(i + 1, n):
                assert not _rects_overlap(rects[i], rects[j]), (n, i, j)


def test_panel_grid_rows_from_max_cols():
    # 5 panels at 4 cols -> 2 rows; row 2's panel sits below row 1's.
    rects = panel_grid(5, 800, 600, max_cols=4)
    assert rects[0][1] < rects[4][1]                  # 5th panel is on a lower row


def test_voxels_by_z_groups_and_defaults_partial():
    visited = [(0, 0, 0), (1, 0, 0), (0, 0, 2)]
    status = {(0, 0, 0): "sufficient"}                # (1,0,0)/(0,0,2) absent
    by_z = voxels_by_z(visited, status)
    assert set(by_z) == {0, 2}
    assert sorted(by_z[0]) == [((0, 0), "sufficient"), ((1, 0), "partial")]
    assert by_z[2] == [((0, 0), "partial")]


def _grid_two_slices():
    g = VoxelCoverage(cell_size_mm=50.0, min_samples=8, min_spread_mm=15.0)
    rng = np.linspace(5, 45, 9)
    for t in rng:                                     # voxel (0,0,0): sufficient
        g.add([t, t, 10.0])
    g.add([10.0, 10.0, 10.0]); g.add([12.0, 12.0, 12.0])   # voxel (0,0,0) extra
    for t in rng:                                     # voxel (0,0,1): higher z slice
        g.add([t, t, 60.0])
    g.add([60.0, 10.0, 10.0])                         # voxel (1,0,0): partial (1 sample)
    return g


def test_render_smoke_headless():
    pytest.importorskip("cv2")
    from Utils.gaze.coverage_voxel_view import VoxelCoverageBoxUI
    ui = VoxelCoverageBoxUI(50.0, width=800, height=560)
    g = _grid_two_slices()
    canvas = ui.render(g, cur_xyz=[12.0, 12.0, 12.0], target_xyz=g.next_target())
    assert canvas.shape == (560, 800, 3)
    assert canvas.dtype == np.uint8
    assert not ui._window_open                        # render never opened a window


def test_render_handles_empty_and_nonfinite():
    pytest.importorskip("cv2")
    from Utils.gaze.coverage_voxel_view import VoxelCoverageBoxUI
    ui = VoxelCoverageBoxUI(50.0, width=640, height=480)
    empty = ui.render(VoxelCoverage(), cur_xyz=None, target_xyz=None)
    assert empty.shape == (480, 640, 3)
    g = _grid_two_slices()
    nf = ui.render(g, cur_xyz=[np.nan, 0.0, 0.0], target_xyz=None)   # non-finite EE
    assert nf.shape == (480, 640, 3)
