"""test_eehand_offset_ls.py — pins the WS-2b least-squares offset core.

Exercises solve_offset_ls (Analyze_eehand_offset_ls.py) on synthetic data with a
known offset x: generate N distinct rotations R(i) and EE-tag origins t(i), form
p(i) = t(i) + R(i) @ x (+ optional noise), and assert recovery. The segmentation
/ backprojection stage needs the perception stack + a rig and is not exercised
here; only the numerical LS core is pinned.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Analyze_eehand_offset_ls import (  # noqa: E402
    invert_se3,
    solve_offset_ls,
    world_eetag_from_sweep,
)


def _synthesize(x_true_m, n=80, noise_m=0.0, seed=0):
    """N random rotations + origins, p = t + R@x (+ Gaussian noise)."""
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(seed)
    R = Rotation.random(n, random_state=seed).as_matrix()
    t = rng.uniform(-0.5, 0.5, size=(n, 3))
    p = t + np.einsum("nij,j->ni", R, np.asarray(x_true_m))
    if noise_m > 0:
        p = p + rng.normal(0.0, noise_m, size=p.shape)
    return R, t, p


def test_recovers_known_offset_clean():
    x_true_mm = np.array([150.0, -200.0, 0.0])
    R, t, p = _synthesize(x_true_mm / 1000.0, n=80, noise_m=0.0)
    x_mm, stats = solve_offset_ls(R, t, p)
    assert np.allclose(x_mm, x_true_mm, atol=1e-6)
    assert stats["overall_rms_mm"] < 1e-6
    assert stats["n_samples"] == 80


def test_recovers_arbitrary_offset_clean():
    x_true_mm = np.array([37.5, 12.0, -88.25])
    R, t, p = _synthesize(x_true_mm / 1000.0, n=50, noise_m=0.0, seed=3)
    x_mm, stats = solve_offset_ls(R, t, p)
    assert np.allclose(x_mm, x_true_mm, atol=1e-6)
    assert stats["per_axis_rms_mm"].max() < 1e-6


def test_recovers_under_noise():
    # 1 mm per-axis centroid noise: averaging over 200 distinct poses should pull
    # the estimate well inside a few mm of truth.
    x_true_mm = np.array([150.0, -200.0, 0.0])
    R, t, p = _synthesize(x_true_mm / 1000.0, n=200, noise_m=1e-3, seed=1)
    x_mm, stats = solve_offset_ls(R, t, p)
    assert np.linalg.norm(x_mm - x_true_mm) < 5.0
    assert stats["median_err_mm"] < 3.0


def test_residual_stats_shapes_and_keys():
    x_true_mm = np.array([10.0, 20.0, 30.0])
    R, t, p = _synthesize(x_true_mm / 1000.0, n=40, noise_m=5e-4, seed=2)
    _, stats = solve_offset_ls(R, t, p)
    assert stats["per_axis_rms_mm"].shape == (3,)
    assert stats["per_sample_err_mm"].shape == (40,)
    for key in ("overall_rms_mm", "median_err_mm", "p95_err_mm", "n_samples"):
        assert key in stats


def test_rejects_too_few_samples():
    with pytest.raises(ValueError):
        solve_offset_ls(np.eye(3)[None], np.zeros((1, 3)), np.zeros((1, 3)))


def test_rejects_shape_mismatch():
    R = np.stack([np.eye(3), np.eye(3)])
    with pytest.raises(ValueError):
        solve_offset_ls(R, np.zeros((2, 3)), np.zeros((3, 3)))


def test_invert_se3_roundtrip():
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.random(1, random_state=7).as_matrix()[0]
    T[:3, 3] = [0.3, -0.2, 1.1]
    assert np.allclose(invert_se3(T) @ T, np.eye(4), atol=1e-12)


def test_world_eetag_from_sweep_matches_chain():
    from scipy.spatial.transform import Rotation
    T_cam_world = np.eye(4)
    T_cam_world[:3, :3] = Rotation.random(1, random_state=5).as_matrix()[0]
    T_cam_world[:3, 3] = [0.1, 0.2, 0.9]
    T_cam_eetag = np.eye(4)
    T_cam_eetag[:3, :3] = Rotation.random(1, random_state=6).as_matrix()[0]
    T_cam_eetag[:3, 3] = [0.4, -0.1, 0.8]
    T_we = world_eetag_from_sweep(T_cam_world, T_cam_eetag)
    assert np.allclose(T_we, invert_se3(T_cam_world) @ T_cam_eetag)


# ── WS-2b units regression (C2): sweep saves T_cam_* in MM, depth is metres ─────

from scipy.spatial.transform import Rotation  # noqa: E402

from Analyze_eehand_offset_ls import vision_transforms_to_metres  # noqa: E402
from Utils.gaze.apriltag_calib import make_transform  # noqa: E402


def test_vision_transforms_to_metres_scales_translation_only():
    R = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    T_mm = make_transform(R, [150.0, -200.0, 500.0])
    T_m = vision_transforms_to_metres(T_mm)
    np.testing.assert_allclose(T_m[:3, :3], R, atol=1e-12)
    np.testing.assert_allclose(T_m[:3, 3], [0.15, -0.20, 0.50], atol=1e-12)


def test_world_eetag_metres_from_mm_saved_transforms():
    """world_eetag_from_sweep must be metres so it differences cleanly against the
    metres Depth-Pro centroid; mm-in would be ~1000x off."""
    Rcw = Rotation.from_euler("xyz", [5, -15, 25], degrees=True).as_matrix()
    Rce = Rotation.from_euler("xyz", [-20, 10, 40], degrees=True).as_matrix()
    T_cw_m = make_transform(Rcw, [0.10, 0.20, 0.80])
    T_ce_m = make_transform(Rce, [0.05, -0.10, 0.70])
    B_truth = world_eetag_from_sweep(T_cw_m, T_ce_m)                  # metres truth
    T_cw_mm = T_cw_m.copy(); T_cw_mm[:3, 3] *= 1000.0
    T_ce_mm = T_ce_m.copy(); T_ce_mm[:3, 3] *= 1000.0
    B_hat = world_eetag_from_sweep(vision_transforms_to_metres(T_cw_mm),
                                   vision_transforms_to_metres(T_ce_mm))
    np.testing.assert_allclose(B_hat, B_truth, atol=1e-9)
    B_bug = world_eetag_from_sweep(T_cw_mm, T_ce_mm)
    assert np.linalg.norm(B_bug[:3, 3] - B_truth[:3, 3]) > 1.0
