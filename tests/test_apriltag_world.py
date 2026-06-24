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
from Utils.gaze.apriltag_calib import transform_point  # noqa: E402
from Utils.gaze.apriltag_world import (  # noqa: E402
    average_pose,
    build_world_map,
    fit_plane,
    plane_basis,
    plane_coords,
    recover_world_pose,
    world_from_plane_coords,
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
    ref, ids, rels, pp, pn = world_map_to_arrays(wm)
    wm2 = world_map_from_arrays(ref, ids, rels, pp, pn)
    assert wm2["ref_id"] == wm["ref_id"] and wm2["ids"] == wm["ids"]
    for i in wm["ids"]:
        np.testing.assert_allclose(wm2["rel"][i], wm["rel"][i], atol=1e-12)
    np.testing.assert_allclose(wm2["plane_point"], wm["plane_point"], atol=1e-12)
    np.testing.assert_allclose(wm2["plane_normal"], wm["plane_normal"], atol=1e-12)


def test_fit_plane_recovers_normal():
    # Points on z=5 (xy-spread) → unit normal ±z, point on the plane.
    pts = np.array([[0, 0, 5.0], [10, 0, 5.0], [0, 10, 5.0], [10, 10, 5.0]])
    c, n = fit_plane(pts)
    assert abs(abs(n[2]) - 1.0) < 1e-9          # normal is ±z
    assert abs(c[2] - 5.0) < 1e-9               # centroid on the plane


def test_plane_fit_robust_to_tilted_reference():
    # Tags are TRULY coplanar, but the reference tag (id 0) is DETECTED with a
    # 15° orientation error. The fitted world plane (from all tag origins) must
    # recover the TRUE table plane in cam — not the noisy reference's tilted +Z.
    T_cam_table = make_transform(_rot_x(10.0), [0.0, 0.0, 600.0])   # true table in cam
    table_T = {0: [0, 0, 0], 1: [400, 0, 0], 2: [0, 300, 0], 3: [400, 300, 0]}  # flat (z=0)
    cam_poses = {i: T_cam_table @ make_transform(np.eye(3), t) for i, t in table_T.items()}
    cam_poses[0] = cam_poses[0] @ make_transform(_rot_x(15.0), [0, 0, 0])  # ref mis-detected
    wm = build_world_map(cam_poses, ref_id=0)
    T_cam_world = recover_world_pose(cam_poses, wm)
    normal_cam = T_cam_world[:3, :3] @ wm["plane_normal"]
    true_normal = T_cam_table[:3, 2]
    # |cos| ≈ 1 → recovered plane normal is parallel to the true table normal.
    assert abs(abs(float(normal_cam @ true_normal)) - 1.0) < 1e-6


def test_world_frame_is_reference_frame_when_ref_not_at_origin():
    # Reviewer gap #2: ref tag NOT at the construction origin. recover must return
    # the REFERENCE tag's camera pose (the world frame == ref frame, by contract).
    constr_T = {0: make_transform(_rot_z(25.0), [120.0, -40.0, 0.0]),
                1: make_transform(np.eye(3), [400.0, 0.0, 0.0]),
                2: make_transform(np.eye(3), [0.0, 300.0, 0.0])}
    T_cam_constr = make_transform(_rot_x(15.0), [5.0, 5.0, 550.0])
    cam_poses = {i: T_cam_constr @ constr_T[i] for i in constr_T}
    wm = build_world_map(cam_poses, ref_id=0)
    np.testing.assert_allclose(recover_world_pose(cam_poses, wm), cam_poses[0], atol=1e-9)


def test_recover_fuses_disagreeing_estimates_stays_rigid():
    # Reviewer gap #4: perturb one tag's runtime pose so the per-tag estimates
    # disagree; the fused result must still be a proper rigid transform.
    _, _, cam_poses = _layout()
    wm = build_world_map(cam_poses)
    noisy = dict(cam_poses)
    noisy[1] = cam_poses[1] @ make_transform(_rot_z(3.0), [4.0, -2.0, 1.0])
    T = recover_world_pose(noisy, wm)
    np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9)  # orthonormal
    assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-9                          # proper rotation


# ── REV04 planar coordinates (plane_basis / plane_coords / round-trip) ──────────

def test_plane_basis_orthonormal_right_handed():
    for n in (np.array([0.0, 0.0, 1.0]), np.array([1.0, 2.0, 3.0]),
              np.array([0.0, 1.0, 0.0])):
        e1, e2 = plane_basis(n)
        nn = n / np.linalg.norm(n)
        assert abs(np.dot(e1, e2)) < 1e-12
        assert abs(np.linalg.norm(e1) - 1.0) < 1e-12
        assert abs(np.linalg.norm(e2) - 1.0) < 1e-12
        assert abs(np.dot(e1, nn)) < 1e-12 and abs(np.dot(e2, nn)) < 1e-12
        np.testing.assert_allclose(np.cross(e1, e2), nn, atol=1e-12)  # right-handed


def test_plane_coords_origin_and_offplane_drop():
    c = np.array([10.0, -5.0, 100.0])
    n = np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(plane_coords(c, c, n), [0.0, 0.0], atol=1e-12)
    # adding any out-of-plane component must not move (u,v)
    p = np.array([42.0, 7.0, 100.0])
    np.testing.assert_allclose(plane_coords(p, c, n),
                               plane_coords(p + 33.0 * n, c, n), atol=1e-12)


def test_plane_coords_isometry_and_roundtrip():
    # tilted plane; in-plane distances are preserved and the map round-trips
    c, n = np.array([1.0, 2.0, 3.0]), np.array([0.3, -0.4, 1.0])
    e1, e2 = plane_basis(n)
    pts = np.array([c + a * e1 + b * e2
                    for a, b in [(0, 0), (50, 0), (0, 70), (-20, 35), (120, -90)]])
    uv = plane_coords(pts, c, n)
    back = world_from_plane_coords(uv, c, n)
    np.testing.assert_allclose(back, pts, atol=1e-9)
    # distance preservation (rigid in-plane embedding)
    d3 = np.linalg.norm(pts[1] - pts[3])
    d2 = np.linalg.norm(uv[1] - uv[3])
    assert abs(d3 - d2) < 1e-9


def test_plane_coords_single_vs_batch_shapes():
    c, n = np.zeros(3), np.array([0.0, 0.0, 1.0])
    assert plane_coords(np.array([1.0, 2.0, 9.0]), c, n).shape == (2,)
    assert plane_coords(np.zeros((4, 3)), c, n).shape == (4, 2)
