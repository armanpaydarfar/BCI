"""
test_apriltag_world_3d.py — WS-4 first-pass: non-coplanar (3-D) world map.

Synthetic-only: place tags on TWO non-coplanar planes, view them across a head
sweep with per-view depth noise, register WITHOUT the coplanar snap, and assert
(a) the recovered 3-D layout matches the truth within tolerance, and (b) the
true z-spread is PRESERVED — i.e. the REV05 coplanar-snap path would have
flattened it and thus failed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from Utils.gaze.apriltag_calib import make_transform  # noqa: E402
from Utils.gaze.apriltag_world import (  # noqa: E402
    recover_world_pose,
    register_world_map_multiview,
    world_map_from_arrays,
    world_map_to_arrays,
)
from Utils.gaze.apriltag_world_3d import (  # noqa: E402
    register_world_map_3d,
    world_map_3d_geometry_report,
)


def _rot_x(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _two_plane_layout():
    """Tags 0-2 on the table (z=0), tags 3-4 on a raised back panel (a wall in
    the x=600 plane, facing -x). Genuinely non-coplanar: the z range spans
    0..400 mm and there is no single plane through all five origins."""
    return {
        0: make_transform(np.eye(3), [0.0, 0.0, 0.0]),
        1: make_transform(_rot_z(15.0), [400.0, 0.0, 0.0]),
        2: make_transform(np.eye(3), [0.0, 300.0, 0.0]),
        # Raised panel: rotate so +Z faces roughly -x, origins lifted off z=0.
        3: make_transform(_rot_y(90.0), [600.0, 0.0, 200.0]),
        4: make_transform(_rot_y(90.0) @ _rot_z(10.0), [600.0, 300.0, 400.0]),
    }


def _sweep(world_T, *, n=28, depth_sigma=30.0, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for k in range(n):
        T_cam_world = make_transform(
            _rot_x(15.0 + 0.6 * k) @ _rot_z(-40.0 + 4.0 * k)
            @ _rot_y(rng.normal(0, 8)),
            [rng.normal(0, 120), rng.normal(0, 120), 900.0 + rng.normal(0, 80)])
        frame = {}
        for i, wt in world_T.items():
            T = T_cam_world @ wt
            t = T[:3, 3].copy()
            t[2] += rng.normal(0.0, depth_sigma)   # depth noise along optical axis
            R = T[:3, :3] @ _rot_x(rng.normal(0, 1.5)) @ _rot_y(rng.normal(0, 1.5))
            frame[i] = make_transform(R, t)
        frames.append(frame)
    return frames


def _sweep_with_visibility(world_T, visible_per_frame, *, depth_sigma=30.0, seed=0):
    """Like ``_sweep`` but only the listed tags are visible in each frame, so a
    test can control co-visibility (which tag pairs ever share a frame)."""
    rng = np.random.default_rng(seed)
    frames = []
    for k, vis in enumerate(visible_per_frame):
        T_cam_world = make_transform(
            _rot_x(15.0 + 0.6 * k) @ _rot_z(-40.0 + 4.0 * k)
            @ _rot_y(rng.normal(0, 8)),
            [rng.normal(0, 120), rng.normal(0, 120), 900.0 + rng.normal(0, 80)])
        frame = {}
        for i in vis:
            T = T_cam_world @ world_T[i]
            t = T[:3, 3].copy()
            t[2] += rng.normal(0.0, depth_sigma)
            R = T[:3, :3] @ _rot_x(rng.normal(0, 1.5)) @ _rot_y(rng.normal(0, 1.5))
            frame[i] = make_transform(R, t)
        frames.append(frame)
    return frames


def test_pose_graph_places_tag_never_co_visible_with_reference():
    """The pose-graph win: a tag NEVER in the same frame as the reference is still
    placed (and accurately) by chaining through a shared neighbour. The old
    single-anchor fuse, which only used pairs through the reference, would drop it.
    """
    world_T = _two_plane_layout()
    # Even frames show {0,1,2}; odd frames show {1,2,3,4}. Tags 0 and 4 are never
    # co-visible, but 1 and 2 bridge them.
    vis = [[0, 1, 2] if k % 2 == 0 else [1, 2, 3, 4] for k in range(40)]
    frames = _sweep_with_visibility(world_T, vis, seed=7)
    wm = register_world_map_3d(frames, ref_id=0)
    assert 4 in wm["ids"], "tag 4 (never co-visible with ref) must still be placed"
    err4 = np.linalg.norm(wm["rel"][4][:3, 3] - world_T[4][:3, 3])
    assert err4 < 30.0, f"chained tag 4 residual {err4:.1f} mm too high"


def test_pose_graph_survives_reference_tag_flips():
    """Corrupting the reference tag's pose in ~40% of frames (the flip ambiguity
    that poisoned the single-anchor map) does not wreck the pose-graph fuse: the
    bad edges are out-voted by the good majority and the many other pairs."""
    world_T = _two_plane_layout()
    frames = _sweep(world_T, n=40, seed=8)
    flip = make_transform(_rot_x(35.0) @ _rot_y(25.0), [0.0, 0.0, 0.0])
    for k, fr in enumerate(frames):
        if k % 5 < 2 and 0 in fr:          # ~40% of frames
            fr[0] = fr[0] @ flip
    wm = register_world_map_3d(frames, ref_id=0)
    err = np.mean([np.linalg.norm(wm["rel"][i][:3, 3] - world_T[i][:3, 3])
                   for i in world_T])
    assert err < 25.0, f"pose-graph should reject ref flips; residual {err:.1f} mm"


def test_3d_registration_recovers_non_coplanar_layout():
    """Multi-view fuse with NO snap recovers the true 3-D tag origins (ref at
    tag 0 → rel[i] origin ≈ world_T[i] origin) within tolerance."""
    world_T = _two_plane_layout()
    frames = _sweep(world_T, seed=1)
    wm = register_world_map_3d(frames, ref_id=0)
    assert wm["ref_id"] == 0 and wm["ids"] == [0, 1, 2, 3, 4]
    err = np.mean([np.linalg.norm(wm["rel"][i][:3, 3] - world_T[i][:3, 3])
                   for i in world_T])
    assert err < 20.0, f"3-D residual {err:.1f} mm too high"


def test_z_spread_preserved_vs_coplanar_snap():
    """The defining property: the 3-D path PRESERVES the out-of-plane structure
    that the REV05 coplanar snap erases. The true origins span ~400 mm in z;
    after the snap that spread collapses to ~0, after the 3-D fuse it survives."""
    world_T = _two_plane_layout()
    frames = _sweep(world_T, seed=2)

    wm3d = register_world_map_3d(frames, ref_id=0)
    wm_snap = register_world_map_multiview(frames, ref_id=0)

    z3d = np.array([wm3d["rel"][i][:3, 3][2] for i in wm3d["ids"]])
    # Snapped map: residual height along ITS plane normal is ~0 by construction.
    n_snap = wm_snap["plane_normal"]
    h_snap = np.array([wm_snap["rel"][i][:3, 3] @ n_snap for i in wm_snap["ids"]])

    assert np.ptp(z3d) > 300.0, "3-D fuse should preserve the ~400 mm z spread"
    assert h_snap.std() < 1e-6, "REV05 snap should flatten out-of-plane spread"


def test_3d_geometry_report_flags_non_planarity_and_reproducibility():
    """The 3-D report: large ``max_out_of_plane_mm`` / ``z_spread_mm`` confirms
    genuine 3-D structure (not snapped flat); small ``max_fit_residual_mm``
    confirms the sweep had enough diversity to triangulate."""
    world_T = _two_plane_layout()
    frames = _sweep(world_T, seed=3)
    wm = register_world_map_3d(frames, ref_id=0)
    rep = world_map_3d_geometry_report(wm, frames)

    assert rep["num_tags"] == 5
    # Genuinely non-coplanar: a best-fit plane through the origins leaves a
    # large worst-case deviation and the z-extent is the real structure.
    assert rep["max_out_of_plane_mm"] > 50.0
    assert rep["z_spread_mm"] > 100.0
    # Reproducible fuse: per-tag estimates cluster tightly around the fused pose.
    assert rep["max_fit_residual_mm"] < 40.0
    assert set(rep["per_tag_residual_mm"].keys()) == set(world_T.keys())


def test_3d_map_is_drop_in_for_recover_and_arrays():
    """The 3-D map reuses the standard dict shape, so ``recover_world_pose`` and
    the array (de)serialisers work on it unchanged."""
    world_T = _two_plane_layout()
    # Noise-free single observation so recover_world_pose is exact.
    T_cam_world = make_transform(_rot_x(20.0) @ _rot_z(30.0), [10.0, -15.0, 800.0])
    frames = [{i: T_cam_world @ world_T[i] for i in world_T}]
    wm = register_world_map_3d(frames, ref_id=0)

    np.testing.assert_allclose(recover_world_pose(frames[0], wm), T_cam_world,
                               atol=1e-7)

    ref, ids, rels, pp, pn = world_map_to_arrays(wm)
    wm2 = world_map_from_arrays(ref, ids, rels, pp, pn)
    assert wm2["ids"] == wm["ids"]
    for i in wm["ids"]:
        np.testing.assert_allclose(wm2["rel"][i], wm["rel"][i], atol=1e-12)


def test_default_ref_is_most_seen_tag():
    """With ref_id=None the reference is the most-frequently-seen tag (ties →
    lowest id)."""
    world_T = _two_plane_layout()
    frames = _sweep(world_T, n=10, seed=4)
    # Drop tag 0 from most frames so tag 1 becomes the most-seen.
    for fr in frames[2:]:
        fr.pop(0, None)
    wm = register_world_map_3d(frames)
    assert wm["ref_id"] == 1


def test_empty_and_bad_ref_raise():
    import pytest
    with pytest.raises(ValueError):
        register_world_map_3d([])
    world_T = _two_plane_layout()
    frames = _sweep(world_T, n=5, seed=5)
    with pytest.raises(ValueError):
        register_world_map_3d(frames, ref_id=999)
