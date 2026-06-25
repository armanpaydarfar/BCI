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
    register_world_map_multiview,
    world_map_geometry_report,
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


def test_world_map_geometry_report_flags_skew_and_passes_square():
    """The geometry gate: a square coplanar layout reports ~90°; a skewed one (the
    58° failure mode) is flagged; out-of-plane tags are measured."""
    # Square, coplanar (z=0), +Z up: 0 at origin, 1 along +x, 2 along +y.
    square = {0: make_transform(np.eye(3), [0.0, 0.0, 0.0]),
              1: make_transform(np.eye(3), [800.0, 0.0, 0.0]),
              2: make_transform(np.eye(3), [0.0, 500.0, 0.0])}
    wm = build_world_map(square)  # <3 tags would skip the normal; 3 is fine
    rep = world_map_geometry_report(wm, x_edge_ids=[0, 1], y_edge_ids=[2, 0])
    assert abs(rep["corner_angle_deg"] - 90.0) < 1.0
    assert rep["max_out_of_plane_mm"] < 1e-6
    assert abs(rep["x_edge_len_mm"] - 800.0) < 1.0 and abs(rep["y_edge_len_mm"] - 500.0) < 1.0

    # Skewed: tag 1 pulled off the +x axis so the corner is no longer square.
    skew = dict(square)
    skew[1] = make_transform(np.eye(3), [800.0, 500.0, 0.0])
    wmk = build_world_map(skew)
    repk = world_map_geometry_report(wmk, x_edge_ids=[0, 1], y_edge_ids=[2, 0])
    assert abs(repk["corner_angle_deg"] - 90.0) > 10.0  # flagged: deviates from square


def _rot_y(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def test_multiview_registration_beats_single_view_under_depth_noise():
    """REV05: static coplanar upward-facing tags, viewed across a head sweep with
    per-view single-tag DEPTH noise (perturbation along each view's optical axis,
    the real failure mode). Multi-view averaging must recover the true static
    geometry far better than the one-window `build_world_map`, and pin the normal."""
    rng = np.random.default_rng(0)
    # Coplanar (z=0), all +Z up; only yaw differs — the physical taped-table layout.
    world_T = {
        0: make_transform(np.eye(3), [0.0, 0.0, 0.0]),
        1: make_transform(_rot_z(90.0), [800.0, 0.0, 0.0]),
        2: make_transform(np.eye(3), [0.0, 500.0, 0.0]),
        3: make_transform(_rot_z(45.0), [800.0, 500.0, 0.0]),
        4: make_transform(_rot_z(20.0), [400.0, 250.0, 0.0]),
    }

    def noisy_view(T_cam_world, depth_sigma=40.0):
        frame = {}
        for i, wt in world_T.items():
            T = T_cam_world @ wt
            t = T[:3, 3].copy()
            t[2] += rng.normal(0.0, depth_sigma)          # depth along optical axis
            R = T[:3, :3] @ _rot_x(rng.normal(0.0, 1.5)) @ _rot_y(rng.normal(0.0, 1.5))
            frame[i] = make_transform(R, t)
        return frame

    # A head sweep: 24 viewpoints varying in angle + position around the table.
    frames = []
    for k in range(24):
        T_cam_world = make_transform(
            _rot_x(15.0 + 0.7 * k) @ _rot_z(-40.0 + 5.0 * k) @ _rot_y(rng.normal(0, 8)),
            [rng.normal(0, 120), rng.normal(0, 120), 700.0 + rng.normal(0, 80)])
        frames.append(noisy_view(T_cam_world))

    def rel_pos_error(wm):
        # ref is tag 0 at the origin, so rel[i] origin should equal world_T[i] origin.
        return np.mean([np.linalg.norm(wm["rel"][i][:3, 3] - world_T[i][:3, 3])
                        for i in world_T])

    single = build_world_map(frames[0])              # one noisy window (today's path)
    multi = register_world_map_multiview(frames)     # REV05 multi-view

    e_single = rel_pos_error(single)
    e_multi = rel_pos_error(multi)
    assert e_multi < 0.4 * e_single, f"multi-view {e_multi:.1f} mm not better than single {e_single:.1f} mm"
    assert e_multi < 20.0, f"multi-view residual {e_multi:.1f} mm too high"
    # Coplanar + up → the normal is pinned to the table normal (ref-tag +Z = [0,0,1]).
    np.testing.assert_allclose(np.abs(multi["plane_normal"]), [0, 0, 1], atol=0.05)
    # Origins are snapped coplanar: no residual height spread in the ref frame.
    heights = np.array([multi["rel"][i][:3, 3] @ multi["plane_normal"] for i in multi["ids"]])
    assert heights.std() < 1e-6


def test_multiview_registration_rejects_flipped_views():
    """A handful of flipped views (gross translation outliers) must not drag the
    fused map — the per-tag outlier gate drops them."""
    world_T = {0: make_transform(np.eye(3), [0.0, 0.0, 0.0]),
               1: make_transform(np.eye(3), [600.0, 0.0, 0.0]),
               2: make_transform(np.eye(3), [0.0, 400.0, 0.0])}
    T_cam_world = make_transform(_rot_x(15.0), [0.0, 0.0, 600.0])
    frames = [{i: T_cam_world @ world_T[i] for i in world_T} for _ in range(10)]
    # Two flipped views: tag 1 lands 300 mm off (its alternate PnP solution).
    for f in frames[:2]:
        bad = f[1].copy(); bad[:3, 3] = bad[:3, 3] + np.array([0.0, 0.0, 300.0]); f[1] = bad
    wm = register_world_map_multiview(frames)
    np.testing.assert_allclose(wm["rel"][1][:3, 3], [600.0, 0.0, 0.0], atol=5.0)


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


def test_plane_normal_from_orientation_survives_origin_height_noise():
    # The world tags are taped coplanar on the flat table, so each tag's +Z is the
    # true table normal — but their ESTIMATED origin heights are noisy (the single-
    # tag depth ambiguity). The plane normal must come from the (reliable)
    # orientations and recover the true table normal, where a fit through the noisy
    # origins would tilt (the HIL 65° failure, 2026-06-24).
    T_cam_table = make_transform(_rot_x(10.0), [0.0, 0.0, 600.0])   # true table in cam
    table_T = {0: [0, 0, 0], 1: [400, 0, 0], 2: [0, 300, 0], 3: [400, 300, 0]}  # flat (z=0)
    cam_poses = {i: T_cam_table @ make_transform(np.eye(3), t) for i, t in table_T.items()}
    # Corrupt each tag's estimated origin ALONG the table normal (height noise),
    # leaving the orientations — the reliable cue — intact.
    for i, dz in zip([0, 1, 2, 3], [60.0, -50.0, 40.0, -70.0]):
        cam_poses[i] = cam_poses[i] @ make_transform(np.eye(3), [0.0, 0.0, dz])
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


def test_recover_rejects_a_flipped_tag_by_consensus():
    # One world tag's pose has FLIPPED to a grossly wrong solution (decimetre-scale
    # error); the other three agree. The consensus must out-vote the flipped tag and
    # return the clean world pose, not a polluted average. This is the head-motion
    # flit fix (2026-06-24).
    _, T_cam_world, cam_poses = _layout()
    wm = build_world_map(cam_poses)
    flipped = dict(cam_poses)
    flipped[3] = cam_poses[3] @ make_transform(_rot_x(120.0), [200.0, 150.0, 300.0])
    # 3 clean tags agree exactly → their cluster wins; the flip is dropped.
    np.testing.assert_allclose(recover_world_pose(flipped, wm), T_cam_world, atol=1e-6)


def test_recover_small_noise_not_rejected_matches_plain_mean():
    # Sub-tolerance noise on one tag must NOT trip the consensus (all estimates stay
    # in one cluster) — the fused result equals the plain average of all four, so the
    # robust path is a no-op when nothing has flipped.
    _, _, cam_poses = _layout()
    wm = build_world_map(cam_poses)
    noisy = dict(cam_poses)
    noisy[1] = cam_poses[1] @ make_transform(_rot_z(1.0), [3.0, -1.0, 2.0])
    ests = [noisy[i] @ np.linalg.inv(wm["rel"][i]) for i in noisy]
    np.testing.assert_allclose(recover_world_pose(noisy, wm), average_pose(ests), atol=1e-9)


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
