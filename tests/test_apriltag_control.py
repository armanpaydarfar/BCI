"""
test_apriltag_control.py — pure helpers of the AprilTag gaze→robot control tool.

The motion + relay paths need hardware; these pin the hardware-free decision
logic: the workspace clamp (the only joint-safety guard, since the robot
enforces none), the nearest-pose lookup, and the gaze→base-frame chain.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from Utils.gaze.apriltag_calib import make_transform, transform_point  # noqa: E402
from tools.apriltag_control_test import (  # noqa: E402
    clamp_joints,
    gaze_point_in_base,
    nearest_pose,
    workspace_bounds,
)


def _rot_x(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_workspace_bounds_envelope():
    Q = np.array([[0.0] * 7, [1.0] * 7])  # span 1.0 per joint
    lo, hi = workspace_bounds(Q, margin=0.05)
    np.testing.assert_allclose(lo, [-0.05] * 7)
    np.testing.assert_allclose(hi, [1.05] * 7)


def test_clamp_joints_in_range_is_noop():
    q = np.array([0.2] * 7)
    lo, hi = np.full(7, -1.0), np.full(7, 1.0)
    out, clamped = clamp_joints(q, lo, hi)
    np.testing.assert_allclose(out, q)
    assert clamped is False


def test_clamp_joints_clips_and_flags():
    q = np.array([2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    lo, hi = np.full(7, -1.0), np.full(7, 1.0)
    out, clamped = clamp_joints(q, lo, hi)
    assert clamped is True
    assert out[0] == 1.0 and out[1] == -1.0


def test_nearest_pose():
    X = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0]], dtype=float)
    idx, dist = nearest_pose(X, np.array([90.0, 5.0, 0.0]))
    assert idx == 1
    assert dist == np.hypot(10.0, 5.0)


def test_gaze_point_in_base_principal_point():
    # World tag facing the camera at z=500mm (plane z=500, normal +z). Gaze at
    # the principal point → ray +z → hits the tag origin → P_world = origin →
    # P_base = T_base_world translation. An independent closed-form check.
    K = np.array([[1490.0, 0, 800.0], [0, 1490.0, 600.0], [0, 0, 1]])
    T_cam_world = make_transform(np.eye(3), [0.0, 0.0, 500.0])
    T_base_world = make_transform(np.eye(3), [300.0, -120.0, 50.0])
    # world-frame table plane = z=0 (point origin, normal +z).
    p_base = gaze_point_in_base(800.0, 600.0, K, T_cam_world, T_base_world,
                                [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(p_base, [300.0, -120.0, 50.0], atol=1e-6)


def test_gaze_point_in_base_tilted_offaxis():
    # Forward-construct: a point ON the tilted tag plane (z=0 in the tag frame),
    # project it to a gaze pixel through a rotated/translated camera pose, then
    # confirm the full chain recovers its base-frame coordinates. Exercises R≠I,
    # an off-principal pixel, and an oblique plane — the terms the frontal test
    # leaves at zero.
    K = np.array([[1490.0, 0, 800.0], [0, 1490.0, 600.0], [0, 0, 1]])
    T_cam_world = make_transform(_rot_x(20.0) @ _rot_z(30.0), [40.0, -15.0, 500.0])
    T_base_world = make_transform(_rot_z(10.0), [300.0, -120.0, 50.0])
    p_world_true = np.array([30.0, -20.0, 0.0])   # on the tag plane (tag-frame z=0)
    cam_pt = transform_point(T_cam_world, p_world_true)
    proj = K @ cam_pt
    px, py = proj[0] / proj[2], proj[1] / proj[2]   # the gaze pixel that sees it
    expected = transform_point(T_base_world, p_world_true)
    got = gaze_point_in_base(px, py, K, T_cam_world, T_base_world,
                             [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_gaze_point_in_base_nan_gaze_returns_none():
    K = np.array([[1490.0, 0, 800.0], [0, 1490.0, 600.0], [0, 0, 1]])
    T = make_transform(np.eye(3), [0.0, 0.0, 500.0])
    assert gaze_point_in_base(float("nan"), 600.0, K, T, T,
                              [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]) is None
