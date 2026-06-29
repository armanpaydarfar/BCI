"""
test_object_target.py — depth-free 3-D object target (WS-4).

Pure geometry + selection; no display, no perception stack. The ray∩plane is
checked by round-trip (project a known world point on the plane to a pixel, then
recover it), the table plane by a synthetic coplanar tag set, and the mask
selection by the WS-1 rule (nearest-containing-centroid, NOT smallest segment).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Utils.gaze.apriltag_calib import invert_transform, make_transform, transform_point  # noqa: E402
from Utils.gaze.object_target import (  # noqa: E402
    gaze_divergence_ok,
    pixel_on_plane_world,
    select_object_pixel,
    table_plane_from_map,
)

_K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])


def _cam_looking_down(height_mm=1000.0):
    """T_cam_world for a camera ``height_mm`` above the z=0 plane looking straight
    down (+z_cam = −z_world)."""
    R_world_cam = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    T_world_cam = make_transform(R_world_cam, [0.0, 0.0, height_mm])
    return invert_transform(T_world_cam)


def _project(P_world, K, T_cam_world):
    p = transform_point(T_cam_world, np.asarray(P_world, float))
    return K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2]


def test_pixel_on_plane_world_roundtrip():
    Tcw = _cam_looking_down(1000.0)
    pp, pn = np.zeros(3), np.array([0.0, 0.0, 1.0])
    for P in ([0.0, 0.0, 0.0], [100.0, -50.0, 0.0], [-220.0, 180.0, 0.0]):
        u, v = _project(P, _K, Tcw)
        hit = pixel_on_plane_world(u, v, _K, Tcw, pp, pn)
        assert hit is not None
        np.testing.assert_allclose(hit, P, atol=1e-6)


def test_pixel_on_plane_world_misses_when_ray_parallel():
    Tcw = _cam_looking_down(1000.0)
    # A plane PARALLEL to the optical axis (normal ⟂ the down-axis) → the central
    # ray never meets it.
    hit = pixel_on_plane_world(320.0, 240.0, _K, Tcw,
                               np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert hit is None


def _flat_tag(x, y, z=0.0):
    """A tag lying flat on the z=0 plane (its +Z is world +Z)."""
    return make_transform(np.eye(3), [x, y, z])


def test_table_plane_from_table_tags_only():
    # Table tags 0-2 flat at z=0; a 'wall' tag 6 lifted to z=400 that must NOT
    # drag the plane (it is excluded by table_ids).
    rel = {0: _flat_tag(0, 0), 1: _flat_tag(400, 0), 2: _flat_tag(0, 300),
           6: make_transform(np.eye(3), [600.0, 0.0, 400.0])}
    wm = {"rel": rel, "ids": [0, 1, 2, 6]}
    point, normal, info = table_plane_from_map(wm, table_ids=[0, 1, 2])
    assert abs(abs(normal[2]) - 1.0) < 1e-9            # table normal is vertical
    assert info["max_resid_mm"] < 1e-6                  # table tags are coplanar
    assert info["table_ids"] == [0, 1, 2]
    np.testing.assert_allclose(point[2], 0.0, atol=1e-9)


def test_table_plane_needs_three_tags():
    import pytest
    rel = {0: _flat_tag(0, 0), 1: _flat_tag(400, 0)}
    with pytest.raises(ValueError):
        table_plane_from_map({"rel": rel, "ids": [0, 1]}, table_ids=[0, 1])


def _square(cx, cy, half):
    return [[cx - half, cy - half], [cx + half, cy - half],
            [cx + half, cy + half], [cx - half, cy + half]]


def test_select_picks_nearest_containing_centroid_not_smallest():
    # Gaze sits inside BOTH a big mask (centroid near gaze) and a small mask
    # (centroid farther). The WS-1 rule picks the big one (nearest centroid) — the
    # whole object — not the smallest segment.
    big = {"mask_polygon": _square(103, 103, 60)}    # contains gaze, centroid ~2.8 px away
    small = {"mask_polygon": _square(120, 120, 20)}  # also contains gaze, centroid ~21 px away
    px, py, info = select_object_pixel([small, big], (105.0, 105.0), "centroid")
    assert info["pick"] == "contained" and info["n_contained"] == 2
    np.testing.assert_allclose([px, py], [103.0, 103.0], atol=1.0)  # big mask centroid (nearer)


def test_select_bottom_is_footprint_row():
    det = {"mask_polygon": _square(100, 100, 50)}  # spans y∈[50,150]
    px, py, _ = select_object_pixel([det], (100.0, 100.0), "bottom")
    assert abs(py - 150.0) < 1e-6        # lowest mask row
    assert abs(px - 100.0) < 1.0         # centered footprint x


def test_select_miss_when_gaze_far_outside():
    det = {"mask_polygon": _square(100, 100, 20)}
    px, py, info = select_object_pixel([det], (1000.0, 1000.0), "centroid")
    assert px is None and py is None and info["pick"] == "miss"


def test_gaze_divergence_guard():
    assert gaze_divergence_ok([320.0, 240.0], [330.0, 245.0]) is True
    assert gaze_divergence_ok([320.0, 240.0], [500.0, 240.0]) is False   # >80 px
    assert gaze_divergence_ok([np.nan, 240.0], [320.0, 240.0]) is False
