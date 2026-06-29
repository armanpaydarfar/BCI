"""
test_apriltag_control_3d.py — pure helpers of the 3-D AprilTag gaze→robot control
tool (WS-4, ``world_xyz_nn``).

The motion + relay + vlm_service paths need hardware/services; these pin the
hardware-free composition that turns a perception waypoint into a joint target:
``object_point_world_mm`` (metres→mm + cam→world) and its hand-off to
``GazeCalibration3D.query_xyz`` (the world→library NN). The lookup/clamp itself is
pinned in ``test_calibration_mapping_3d.py``; here we pin the unit conversion and
the frame composition the live tool depends on.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from Utils.gaze.apriltag_calib import make_transform, transform_point  # noqa: E402
from Utils.gaze.calibration_mapping_3d import GazeCalibration3D  # noqa: E402
from tools.apriltag_control_test_3d import object_point_world_mm  # noqa: E402


def _rot_x(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ── object_point_world_mm: metres→mm + cam→world ──────────────────────────────


def test_object_point_world_mm_identity_pose_scales_metres_to_mm():
    # World frame == camera frame (T_cam_world = I): the world mm point is just the
    # cam point scaled metres→mm. 0.5 m → 500 mm.
    T_cam_world = make_transform(np.eye(3), [0.0, 0.0, 0.0])
    p = object_point_world_mm([0.1, -0.2, 0.5], T_cam_world)
    np.testing.assert_allclose(p, [100.0, -200.0, 500.0], atol=1e-9)


def test_object_point_world_mm_matches_forward_construction():
    # Forward-construct: pick a known world point (mm), push it through a
    # rotated/translated world→cam pose to get position_cam (metres), then confirm
    # object_point_world_mm recovers the original world point.
    T_cam_world = make_transform(_rot_x(15.0) @ _rot_z(40.0), [120.0, -60.0, 800.0])
    p_world_true_mm = np.array([250.0, 130.0, 90.0])
    p_cam_mm = transform_point(T_cam_world, p_world_true_mm)   # world→cam, mm
    position_cam_m = p_cam_mm / 1000.0                          # service reports metres
    got = object_point_world_mm(position_cam_m, T_cam_world)
    np.testing.assert_allclose(got, p_world_true_mm, atol=1e-6)


def test_object_point_world_mm_rejects_non_finite():
    T_cam_world = make_transform(np.eye(3), [0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        object_point_world_mm([float("nan"), 0.0, 0.5], T_cam_world)


# ── full composition: waypoint → world point → query_xyz picks the right row ───


def test_world_point_query_picks_matching_library_row():
    # A 3-D library whose EE world points are 3 well-separated columns; build a
    # position_cam that lands on row 1's world point through a non-trivial pose and
    # assert query_xyz returns row 1 with a ~0 mm NN distance.
    X = np.array([
        [0.0, 0.0, 0.0],
        [300.0, 100.0, 50.0],
        [-200.0, 250.0, 120.0],
    ])
    Q = np.array([
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
        [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6],
    ])
    mapping = GazeCalibration3D.from_arrays(X, Q)

    T_cam_world = make_transform(_rot_x(-25.0) @ _rot_z(10.0), [80.0, 40.0, 900.0])
    target_world_mm = X[1]
    position_cam_m = transform_point(T_cam_world, target_world_mm) / 1000.0

    p_world = object_point_world_mm(position_cam_m, T_cam_world)
    np.testing.assert_allclose(p_world, target_world_mm, atol=1e-6)

    result = mapping.query_xyz(p_world)
    assert result.idx == 1
    assert result.dist < 1e-6
    np.testing.assert_allclose(result.q_target, Q[1], atol=1e-9)
