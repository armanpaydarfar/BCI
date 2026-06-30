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
    elevation_coords,
    gaze_divergence_ok,
    height_on_vertical,
    pixel_on_plane_world,
    select_object_pixel,
    table_plane_from_map,
    world_to_pixel,
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


def _cam_lookat(eye, target):
    """T_cam_world for a camera at ``eye`` looking at ``target`` (OpenCV axes:
    +z forward, +x right, +y down). Oblique so object height is observable."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R_world_cam = np.column_stack([right, down, fwd])
    return invert_transform(make_transform(R_world_cam, eye))


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


def test_select_footprint_row():
    # Both the control tool's "footprint" and the alias "bottom" hit the lowest
    # mask row (regression: "footprint" used to fall through to centroid).
    det = {"mask_polygon": _square(100, 100, 50)}  # spans y∈[50,150]
    for source in ("footprint", "bottom"):
        px, py, _ = select_object_pixel([det], (100.0, 100.0), source)
        assert abs(py - 150.0) < 1e-6, source     # lowest mask row
        assert abs(px - 100.0) < 1.0, source      # centered footprint x
    # centroid is the middle, distinct from the footprint
    _, cy, _ = select_object_pixel([det], (100.0, 100.0), "centroid")
    assert abs(cy - 100.0) < 1.0


def test_select_rejects_table_sized_mask():
    # A huge "table" mask containing the gaze + a small object mask also containing
    # it. With the area guard, the table is rejected and the object is picked.
    frame_area = 640 * 480
    table = {"mask_polygon": _square(320, 240, 300)}   # ~0.78 of the frame
    obj = {"mask_polygon": _square(330, 250, 25)}
    px, py, info = select_object_pixel([table, obj], (332.0, 252.0), "centroid",
                                       frame_area_px=frame_area, max_area_frac=0.5)
    assert info["rejected_large"] == 1
    np.testing.assert_allclose([px, py], [330.0, 250.0], atol=1.0)   # the object, not the table
    # Without a frame area the guard is off (legacy behaviour) → table can win.
    _, _, info2 = select_object_pixel([table, obj], (332.0, 252.0), "centroid")
    assert info2["rejected_large"] == 0


def test_select_miss_when_gaze_far_outside():
    det = {"mask_polygon": _square(100, 100, 20)}
    px, py, info = select_object_pixel([det], (1000.0, 1000.0), "centroid")
    assert px is None and py is None and info["pick"] == "miss"


def test_height_on_vertical_recovers_looked_at_height():
    # Oblique view so height is observable. base on the table; look at a point 200 mm
    # up the object's vertical line → recover that height.
    Tcw = _cam_lookat([0.0, -700.0, 500.0], [0.0, 0.0, 100.0])
    base = np.array([0.0, 0.0, 0.0])
    n = np.array([0.0, 0.0, 1.0])
    for h in (0.0, 120.0, 250.0):
        looked = base + np.array([0.0, 0.0, h])
        u, v = _project(looked, _K, Tcw)
        got = height_on_vertical((u, v), base, n, _K, Tcw)
        assert got is not None
        np.testing.assert_allclose(got, looked, atol=1.0)


def test_height_on_vertical_clamps_and_xy_locked():
    Tcw = _cam_lookat([0.0, -700.0, 500.0], [0.0, 0.0, 100.0])
    base = np.array([50.0, -30.0, 0.0])
    n = np.array([0.0, 0.0, 1.0])
    # A gaze far above the object clamps to max_height, and XY stays the base XY.
    looked = base + np.array([0.0, 0.0, 2000.0])
    u, v = _project(looked, _K, Tcw)
    got = height_on_vertical((u, v), base, n, _K, Tcw, max_height_mm=400.0)
    assert abs(got[2] - 400.0) < 1e-6
    np.testing.assert_allclose(got[:2], base[:2], atol=1e-6)


def test_elevation_coords_section_axes():
    base = np.array([0.0, 0.0, 0.0])
    n = np.array([0.0, 0.0, 1.0])
    cam = np.array([0.0, -700.0, 500.0])   # horiz toward camera is -y
    # straight up the object → h=0, v=height
    np.testing.assert_allclose(elevation_coords([0.0, 0.0, 200.0], base, cam, n),
                               [0.0, 200.0], atol=1e-9)
    # on the table toward the camera → v=0, h>0
    np.testing.assert_allclose(elevation_coords([0.0, -100.0, 0.0], base, cam, n),
                               [100.0, 0.0], atol=1e-9)
    # batch form
    got = elevation_coords(np.array([[0, 0, 200.0], [0, -100, 0.0]]), base, cam, n)
    assert got.shape == (2, 2)


def test_world_to_pixel_roundtrips_with_project():
    Tcw = _cam_lookat([0.0, -700.0, 500.0], [0.0, 0.0, 100.0])
    P = np.array([80.0, -40.0, 150.0])
    u0, v0 = _project(P, _K, Tcw)
    px = world_to_pixel(P, _K, Tcw)
    assert px is not None
    np.testing.assert_allclose(px, (u0, v0), atol=1e-6)


def test_gaze_divergence_guard():
    assert gaze_divergence_ok([320.0, 240.0], [330.0, 245.0]) is True
    assert gaze_divergence_ok([320.0, 240.0], [500.0, 240.0]) is False   # >80 px
    assert gaze_divergence_ok([np.nan, 240.0], [320.0, 240.0]) is False


def test_height_on_vertical_needs_up_normal():
    """gaze_height accumulates above the table only when the table normal points UP
    (toward the camera). A down-pointing normal (the tags' +Z can face either way)
    sends the height the wrong way and clamps every resolve to 0 — the 2026-06-30 rig
    bug. The control loop orients the normal toward the calibrated library to fix it."""
    from Utils.gaze.object_target import height_on_vertical, world_to_pixel
    from Utils.gaze.apriltag_calib import make_transform
    K = np.array([[885., 0, 828.], [0, 885., 598.], [0, 0, 1.]])
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.]])      # world->cam, cam looks down +Zworld
    T_cam_world = make_transform(R, -R @ np.array([0., -300, 800]))
    base = np.zeros(3)
    n_up = np.array([0, 0, 1.])
    gpx = world_to_pixel(np.array([0, 0, 150.]), K, T_cam_world)   # gaze at 150 mm up
    h_up = height_on_vertical(gpx, base, n_up, K, T_cam_world)
    h_dn = height_on_vertical(gpx, base, -n_up, K, T_cam_world)
    assert abs(float((h_up - base) @ n_up) - 150.0) < 5.0        # up-normal recovers height
    assert abs(float((h_dn - base) @ n_up)) < 1e-6               # down-normal clamps to 0


def _rect(x0, y0, x1, y1):
    return {"mask_polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def test_column_merge_reaches_bottle_base_from_cap():
    """Fixating a bottle CAP (small, high) should take the footprint at the BOTTLE's
    base (the contiguous body mask below), not the cap's mid-air bottom."""
    from Utils.gaze.object_target import select_object_pixel
    cap = _rect(100, 100, 150, 150)          # gaze lands here
    body = _rect(90, 145, 160, 300)          # below, x-overlapping, contiguous
    px, py, info = select_object_pixel([cap, body], (125, 125), "footprint")
    assert info["pick"] == "contained"
    assert py > 250          # bottle base (~300), NOT the cap base (~150)
    # with column_merge off, it falls back to the cap's own bottom
    _, py0, _ = select_object_pixel([cap, body], (125, 125), "footprint", column_merge=False)
    assert py0 < 170


def test_noise_line_mask_rejected():
    """A thin line-like FastSAM artefact is dropped, not selected as the object."""
    from Utils.gaze.object_target import select_object_pixel
    line = _rect(100, 100, 500, 108)         # aspect ~50:1 → noise
    obj = _rect(200, 200, 260, 280)          # the real object (gaze inside)
    px, py, info = select_object_pixel([line, obj], (230, 240), "centroid")
    assert info["rejected_noise"] == 1
    assert info["pick"] == "contained" and 200 <= px <= 260


def test_column_merge_does_not_overreach_into_table():
    """The column must stop at the object — a wide table mask below (even contiguous)
    is NOT merged, so the footprint can't balloon to the image bottom (the 2026-06-30
    over-reach: footprints at y=1197)."""
    from Utils.gaze.object_target import select_object_pixel
    cap = _rect(100, 100, 150, 150)          # gaze here (w=50)
    body = _rect(90, 145, 160, 300)          # bottle body (w=70, ≤1.7×50) → merges
    table = _rect(0, 295, 1600, 1197)        # wide table below (w=1600) → must NOT merge
    px, py, info = select_object_pixel([cap, body, table], (125, 125), "footprint",
                                       frame_area_px=1656 * 1196, max_area_frac=0.5)
    assert py < 350          # reaches the body base (~300), NOT the table (~1197)
