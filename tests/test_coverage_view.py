"""
test_coverage_view.py — pure layout math for the REV04 coverage box (rev04 §3).

The cv2 window + drawing in ``CoverageBoxUI`` need a display and are HIL-gated;
these pin the hardware-free geometry: the table ``(u,v)`` mm → canvas pixel map
(with ``+v`` drawn up), the auto-scaling bounds, cell rectangles, and the audio
direction cue.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Utils.gaze.coverage_view import (  # noqa: E402
    cell_rect_px,
    cells_uv_bounds,
    fit_box,
    target_direction,
)


def test_fit_box_v_axis_points_up():
    # A square uv region into a square canvas: larger v maps to a SMALLER pixel y
    # (image y grows downward, +v is drawn up).
    layout = fit_box(0.0, 100.0, 0.0, 100.0, 200, 200, 20)
    _, y_low_v = layout.uv_to_px((50.0, 10.0))
    _, y_high_v = layout.uv_to_px((50.0, 90.0))
    assert y_high_v < y_low_v


def test_fit_box_corners_inside_margin():
    layout = fit_box(0.0, 100.0, 0.0, 100.0, 200, 200, 20)
    x0, y0 = layout.uv_to_px((0.0, 0.0))
    x1, y1 = layout.uv_to_px((100.0, 100.0))
    for v in (x0, y0, x1, y1):
        assert 20 - 1 <= v <= 200 - 20 + 1


def test_fit_box_preserves_aspect():
    # A wide region in a square canvas is letter-boxed: the same scale on both axes.
    layout = fit_box(0.0, 200.0, 0.0, 100.0, 400, 400, 0)
    # 200mm over 400px width-limited → 2 px/mm; height uses the same scale.
    assert abs(layout.scale - 2.0) < 1e-9


def test_fit_box_degenerate_span_is_finite():
    layout = fit_box(50.0, 50.0, 50.0, 50.0, 200, 200, 20)
    assert layout.scale > 0
    px, py = layout.uv_to_px((50.0, 50.0))
    assert 0 <= px <= 200 and 0 <= py <= 200


def test_cells_uv_bounds_padded():
    # one cell at (0,0), 50mm cells, pad 1 → spans [-50, 100] on each axis.
    b = cells_uv_bounds([(0, 0)], 50.0, pad_cells=1)
    assert b == (-50.0, 100.0, -50.0, 100.0)


def test_cells_uv_bounds_empty_is_none():
    assert cells_uv_bounds([], 50.0) is None


def test_cell_rect_px_matches_cell_corners():
    layout = fit_box(-50.0, 100.0, -50.0, 100.0, 300, 300, 0)
    x0, y0, x1, y1 = cell_rect_px(layout, (0, 0), 50.0)
    # cell (0,0) spans uv [0,50]x[0,50]; rect is ordered top-left→bottom-right.
    assert x0 < x1 and y0 < y1
    # its corners agree with the raw uv→px map.
    cx0, cy_bottom = layout.uv_to_px((0.0, 0.0))
    cx1, cy_top = layout.uv_to_px((50.0, 50.0))
    assert (x0, x1) == (min(cx0, cx1), max(cx0, cx1))
    assert (y0, y1) == (min(cy_top, cy_bottom), max(cy_top, cy_bottom))


def test_target_direction_compass():
    assert target_direction((0.0, 0.0), (100.0, 5.0)) == "right"
    assert target_direction((0.0, 0.0), (-100.0, 5.0)) == "left"
    assert target_direction((0.0, 0.0), (5.0, 100.0)) == "up"
    assert target_direction((0.0, 0.0), (5.0, -100.0)) == "down"
    assert target_direction(None, (1.0, 1.0)) == ""
    assert target_direction((0.0, 0.0), None) == ""
