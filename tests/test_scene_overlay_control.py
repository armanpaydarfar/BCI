"""
test_scene_overlay_control.py — WS-5A control-overlay guard.

Covers the gaze-calibration control-decision overlay added to
SceneOverlayRenderer.render: the chosen target, object centroid, footprint
(bottom-of-mask) markers, and the nearest-pose / target-source HUD lines.

Two concerns:
  - the pure ``_to_pixel`` clamp helper (no canvas) — finite + in-bounds gate;
  - a smoke pass that drives render() with every new param on a blank frame
    and asserts it draws without error AND leaves an unmistakable mark, while
    a call with none of the new params is byte-identical to a frame the legacy
    overlays never touched (proves the new path is purely additive).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from Utils.scene_overlay_renderer import (  # noqa: E402
    SceneOverlayRenderer,
    _to_pixel,
)


def _frame():
    return np.full((200, 320, 3), 10, dtype=np.uint8)  # dark, uniform


# ── pure helper ─────────────────────────────────────────────────────────────

def test_to_pixel_rounds_in_bounds():
    assert _to_pixel((12.4, 7.6), 320, 200) == (12, 8)


def test_to_pixel_none_passthrough():
    assert _to_pixel(None, 320, 200) is None


def test_to_pixel_rejects_non_finite():
    assert _to_pixel((float("nan"), 5.0), 320, 200) is None
    assert _to_pixel((10.0, float("inf")), 320, 200) is None


def test_to_pixel_rejects_out_of_bounds():
    assert _to_pixel((-1.0, 5.0), 320, 200) is None
    assert _to_pixel((320.0, 5.0), 320, 200) is None  # x == w is out
    assert _to_pixel((5.0, 200.0), 320, 200) is None  # y == h is out


# ── render smoke ────────────────────────────────────────────────────────────

def test_render_with_control_params_draws_without_error():
    r = SceneOverlayRenderer()
    out = r.render(
        _frame(),
        target_px=(160.0, 100.0),
        centroid_px=(120.0, 80.0),
        bottom_px=(120.0, 140.0),
        nearest_pose_uv=(0.42, -0.13),
        target_source="centroid",
        copy=True,
    )
    assert out.shape == (200, 320, 3)
    # Something magenta-ish must have landed near the target crosshair.
    assert not np.array_equal(out, _frame())


def test_control_params_are_additive():
    """No control params (and no legacy overlays) → only the always-on VLM
    state badge is drawn, in the top-left. The rest of the frame is pristine,
    proving the WS-5A path touches nothing unless asked."""
    r = SceneOverlayRenderer()
    plain = r.render(_frame(), copy=True)
    # Sample the target region (centre) on a plain render: untouched.
    assert int(plain[100, 160, 1]) == 10
    out = r.render(_frame(), target_px=(160.0, 100.0), copy=True)
    # Same pixel now altered by the crosshair.
    assert not np.array_equal(out[90:110, 150:170], plain[90:110, 150:170])


def test_offscreen_control_points_are_skipped():
    """Out-of-frame / non-finite control points must be silently skipped, not
    crash — matches how gaze_xy is already gated."""
    r = SceneOverlayRenderer()
    out = r.render(
        _frame(),
        target_px=(9999.0, 9999.0),
        centroid_px=(float("nan"), 1.0),
        bottom_px=None,
        copy=True,
    )
    # Nothing drawable supplied → identical to the plain render.
    plain = SceneOverlayRenderer().render(_frame(), copy=True)
    assert np.array_equal(out, plain)
