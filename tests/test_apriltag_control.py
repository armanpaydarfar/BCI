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

from Utils.gaze.apriltag_calib import make_transform  # noqa: E402
from tools.apriltag_control_test import (  # noqa: E402
    clamp_joints,
    gaze_point_in_base,
    nearest_pose,
    workspace_bounds,
)


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
    p_base = gaze_point_in_base(800.0, 600.0, K, T_cam_world, T_base_world)
    np.testing.assert_allclose(p_base, [300.0, -120.0, 50.0], atol=1e-6)


def test_gaze_point_in_base_nan_gaze_returns_none():
    K = np.array([[1490.0, 0, 800.0], [0, 1490.0, 600.0], [0, 0, 1]])
    T = make_transform(np.eye(3), [0.0, 0.0, 500.0])
    assert gaze_point_in_base(float("nan"), 600.0, K, T, T) is None
