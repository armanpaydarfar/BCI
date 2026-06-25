"""
test_apriltag_pnp.py — multi-tag board PnP world-pose recovery (REV05).

Synthetic: place the static world tags on a plane, view them from an OBLIQUE camera
pose, project their centres to pixels, and confirm `recover_world_pose_pnp` recovers
the true camera→world pose — the view-robust recovery that per-tag estimation can't
match at a seated 45° angle. cv2-only (no pupil-apriltags), skipped if cv2 is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

cv2 = pytest.importorskip("cv2")

from Utils.gaze.apriltag_calib import invert_transform, make_transform  # noqa: E402
from Utils.gaze.apriltag_detect import recover_world_pose_pnp  # noqa: E402


def _rot_x(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _project(T_cam_world, world_pts, K):
    """Pinhole-project world-frame points (mm) to pixels via T_cam_world + K."""
    pix = {}
    for i, p in world_pts.items():
        pc = T_cam_world[:3, :3] @ p + T_cam_world[:3, 3]
        assert pc[2] > 0, "point behind camera — test camera mis-posed"
        uv = K @ (pc / pc[2])
        pix[i] = uv[:2]
    return pix


def _look_at(eye, target, world_up=np.array([0.0, 0.0, 1.0])):
    """CV camera (+Z forward into scene, +X right, +Y down) looking from ``eye`` at
    ``target`` → T_cam_world, so projected points sit in front (positive cam z)."""
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    f = target - eye; f = f / np.linalg.norm(f)
    r = np.cross(f, world_up); r = r / np.linalg.norm(r)
    d = np.cross(f, r)
    return invert_transform(make_transform(np.column_stack([r, d, f]), eye))


def test_board_pnp_recovers_oblique_pose():
    # Static world tags on the table plane (z=0), mm — a ~900×500 spread like the rig.
    world_pts = {0: np.array([0.0, 0.0, 0.0]),
                 1: np.array([900.0, 0.0, 0.0]),
                 2: np.array([0.0, 500.0, 0.0]),
                 3: np.array([900.0, 500.0, 0.0]),
                 4: np.array([450.0, 250.0, 0.0])}
    world_map = {"ids": sorted(world_pts), "ref_id": 0,
                 "rel": {i: make_transform(np.eye(3), world_pts[i]) for i in world_pts},
                 "plane_point": np.zeros(3), "plane_normal": np.array([0.0, 0.0, 1.0])}
    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])

    # A 45°-oblique seated view: camera behind (-y) and above the table, looking at
    # the table centre — the geometry where per-tag pose is biased but board PnP holds.
    T_cam_world = _look_at(eye=[450.0, -550.0, 650.0], target=[450.0, 250.0, 0.0])
    detections = {i: {"center": c} for i, c in _project(T_cam_world, world_pts, K).items()}

    rec = recover_world_pose_pnp(detections, world_map, K)
    assert rec is not None
    np.testing.assert_allclose(rec, T_cam_world, atol=1e-3)


def test_board_pnp_none_below_min_tags():
    world_map = {"ids": [0, 1], "ref_id": 0,
                 "rel": {0: make_transform(np.eye(3), [0, 0, 0]),
                         1: make_transform(np.eye(3), [100, 0, 0])},
                 "plane_point": np.zeros(3), "plane_normal": np.array([0.0, 0.0, 1.0])}
    K = np.eye(3)
    dets = {0: {"center": np.array([1.0, 2.0])}, 1: {"center": np.array([3.0, 4.0])}}
    assert recover_world_pose_pnp(dets, world_map, K, min_tags=4) is None
