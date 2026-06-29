"""
test_apriltag_control.py — pure helpers of the AprilTag gaze→robot control tool
(REV04 planar runtime).

The motion + relay paths need hardware; these pin the hardware-free geometry: the
REV04 gaze→table-plane ``(u,v)`` chain (`gaze_point_in_plane_uv`). The NN lookup
and the workspace clamp now live in ``GazeCalibrationMappingV3`` and are pinned in
``test_gaze_calibration_mapping.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from Utils.gaze.apriltag_calib import make_transform, transform_point  # noqa: E402
from Utils.gaze.apriltag_world import plane_coords  # noqa: E402
from tools.apriltag_control_test import (  # noqa: E402
    _object_target_pixel,
    gaze_point_in_plane_uv,
)


def _square(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _rot_x(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_gaze_point_in_plane_uv_principal_point():
    # World tag facing the camera at z=500mm (plane z=0 in world, normal +z). Gaze
    # at the principal point → ray +z → hits the world origin → (u,v) = (0,0).
    K = np.array([[1490.0, 0, 800.0], [0, 1490.0, 600.0], [0, 0, 1]])
    T_cam_world = make_transform(np.eye(3), [0.0, 0.0, 500.0])
    uv = gaze_point_in_plane_uv(800.0, 600.0, K, T_cam_world,
                                [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(uv, [0.0, 0.0], atol=1e-6)


def test_gaze_point_in_plane_uv_tilted_offaxis():
    # Forward-construct: a point ON the tilted world plane, project it to a gaze
    # pixel through a rotated/translated camera pose, then confirm the chain
    # recovers its (u,v) — consistent with plane_coords' deterministic basis.
    K = np.array([[1490.0, 0, 800.0], [0, 1490.0, 600.0], [0, 0, 1]])
    T_cam_world = make_transform(_rot_x(20.0) @ _rot_z(30.0), [40.0, -15.0, 500.0])
    plane_point = np.array([0.0, 0.0, 0.0])
    plane_normal = np.array([0.0, 0.0, 1.0])
    p_world_true = np.array([30.0, -20.0, 0.0])   # on the world plane (z=0)
    cam_pt = transform_point(T_cam_world, p_world_true)
    proj = K @ cam_pt
    px, py = proj[0] / proj[2], proj[1] / proj[2]   # the gaze pixel that sees it
    expected = plane_coords(p_world_true, plane_point, plane_normal)
    got = gaze_point_in_plane_uv(px, py, K, T_cam_world, plane_point, plane_normal)
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_gaze_point_in_plane_uv_nan_gaze_returns_none():
    K = np.array([[1490.0, 0, 800.0], [0, 1490.0, 600.0], [0, 0, 1]])
    T = make_transform(np.eye(3), [0.0, 0.0, 500.0])
    assert gaze_point_in_plane_uv(float("nan"), 600.0, K, T,
                                  [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]) is None


# ── WS-1 object-target pixel selection (pure) ────────────────────────────────


def test_object_target_pixel_centroid_of_hit_mask():
    dets = [{"mask_polygon": _square(200, 200, 300, 300)}]
    px = _object_target_pixel(dets, (250.0, 250.0), "centroid")
    np.testing.assert_allclose(px, [250.0, 250.0], atol=0.5)


def test_object_target_pixel_bottom_is_footprint():
    # Bottom = mean x along the lowest mask row → the object↔table contact centre,
    # invariant to where on the object gaze landed (the §1.3 overshoot fix).
    dets = [{"mask_polygon": _square(200, 100, 300, 320)}]
    px = _object_target_pixel(dets, (250.0, 150.0), "bottom")
    np.testing.assert_allclose(px, [250.0, 320.0], atol=0.5)


def test_object_target_pixel_picks_mask_gaze_is_inside():
    # Two masks; gaze sits inside the right one → its centroid, not the left's.
    dets = [{"mask_polygon": _square(0, 0, 100, 100)},
            {"mask_polygon": _square(400, 400, 500, 500)}]
    px = _object_target_pixel(dets, (450.0, 450.0), "centroid")
    np.testing.assert_allclose(px, [450.0, 450.0], atol=0.5)


def test_object_target_pixel_rejects_far_outside_gaze():
    dets = [{"mask_polygon": _square(200, 200, 300, 300)}]
    assert _object_target_pixel(dets, (1000.0, 1000.0), "centroid") is None


def test_object_target_pixel_none_when_no_masks():
    assert _object_target_pixel([], (250.0, 250.0), "centroid") is None
    assert _object_target_pixel([{"box_center": [1, 2]}], (1.0, 2.0), "bottom") is None


def test_object_target_pixel_gaze_source_is_noop():
    dets = [{"mask_polygon": _square(200, 200, 300, 300)}]
    assert _object_target_pixel(dets, (250.0, 250.0), "gaze") is None
