"""
test_apriltag_world.py — multi-tag world map (WS5 occlusion robustness).

Synthetic-only: place tags at known world poses, view them from a known camera
pose, and confirm the map round-trips and that ANY visible subset (occlusion)
recovers the same camera→world pose.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from Utils.gaze.apriltag_calib import make_transform  # noqa: E402
from Utils.gaze.apriltag_world import (  # noqa: E402
    average_pose,
    build_world_map,
    recover_world_pose,
    world_map_from_arrays,
    world_map_to_arrays,
)


def _rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _layout():
    # Reference tag (id 0) at the world origin; three more coplanar corner tags
    # (z=0). Camera at an arbitrary pose. Tags-in-cam = T_cam_world · world_T_tag.
    world_T = {
        0: make_transform(np.eye(3), [0.0, 0.0, 0.0]),
        1: make_transform(_rot_z(90.0), [400.0, 0.0, 0.0]),
        2: make_transform(np.eye(3), [0.0, 300.0, 0.0]),
        3: make_transform(_rot_z(45.0), [400.0, 300.0, 0.0]),
    }
    T_cam_world = make_transform(_rot_x(20.0) @ _rot_z(30.0), [10.0, -15.0, 500.0])
    cam_poses = {i: T_cam_world @ world_T[i] for i in world_T}
    return world_T, T_cam_world, cam_poses


def test_build_map_recovers_relative_poses():
    world_T, _, cam_poses = _layout()
    wm = build_world_map(cam_poses)
    assert wm["ref_id"] == 0 and wm["ids"] == [0, 1, 2, 3]
    for i in world_T:  # ref at origin → rel[i] == world_T[i]
        np.testing.assert_allclose(wm["rel"][i], world_T[i], atol=1e-9)


def test_recover_world_pose_full_set():
    _, T_cam_world, cam_poses = _layout()
    wm = build_world_map(cam_poses)
    np.testing.assert_allclose(recover_world_pose(cam_poses, wm), T_cam_world, atol=1e-9)


def test_recover_survives_occlusion_any_subset():
    _, T_cam_world, cam_poses = _layout()
    wm = build_world_map(cam_poses)
    # the arm occludes everything but one (or two) corner tags — still recovers.
    for subset_ids in ([1], [2], [3], [1, 3], [2, 3]):
        subset = {i: cam_poses[i] for i in subset_ids}
        np.testing.assert_allclose(recover_world_pose(subset, wm), T_cam_world, atol=1e-9)


def test_recover_none_when_no_mapped_tag_visible():
    _, _, cam_poses = _layout()
    wm = build_world_map(cam_poses)
    assert recover_world_pose({7: make_transform(np.eye(3), [1, 2, 3])}, wm) is None


def test_average_pose_symmetric_pair():
    R = average_pose([make_transform(_rot_z(12.0), [10, 0, 0]),
                      make_transform(_rot_z(-12.0), [-10, 0, 0])])
    np.testing.assert_allclose(R[:3, :3], np.eye(3), atol=1e-9)
    np.testing.assert_allclose(R[:3, 3], [0, 0, 0], atol=1e-9)


def test_world_map_arrays_roundtrip():
    _, _, cam_poses = _layout()
    wm = build_world_map(cam_poses)
    ref, ids, rels = world_map_to_arrays(wm)
    wm2 = world_map_from_arrays(ref, ids, rels)
    assert wm2["ref_id"] == wm["ref_id"] and wm2["ids"] == wm["ids"]
    for i in wm["ids"]:
        np.testing.assert_allclose(wm2["rel"][i], wm["rel"][i], atol=1e-12)
