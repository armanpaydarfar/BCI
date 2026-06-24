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
from tools.apriltag_control_test import gaze_point_in_plane_uv  # noqa: E402


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
