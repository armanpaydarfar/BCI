"""
test_control_view.py — pure layout helper for the 3-D control visualiser.

The cv2 dual-pane window in ``ControlView`` is rig-only; this pins the
hardware-free ``table_uv_bounds`` (the top-down extent that frames the library
coverage and keeps the current target on-canvas).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Utils.gaze.control_view import table_uv_bounds  # noqa: E402


def test_bounds_enclose_library_with_margin():
    lib = np.array([[0.0, 0.0], [100.0, 200.0]])
    u_lo, u_hi, v_lo, v_hi = table_uv_bounds(lib)
    assert u_lo < 0.0 and u_hi > 100.0 and v_lo < 0.0 and v_hi > 200.0


def test_bounds_include_extra_target_outside_library():
    lib = np.array([[0.0, 0.0], [100.0, 100.0]])
    _, u_hi, _, _ = table_uv_bounds(lib, extra_uv=np.array([500.0, 50.0]))
    assert u_hi > 500.0   # an out-of-coverage target still fits on-canvas


def test_bounds_none_when_no_finite_points():
    assert table_uv_bounds(np.array([[np.nan, np.nan]])) is None
    assert table_uv_bounds(np.empty((0, 2))) is None
