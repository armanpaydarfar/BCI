"""
test_handeye_axzb.py — WS-2a robot-world / hand-eye solve (AX = ZB).

Synthetic-only. Plant a known Z=T_base_world and X=T_ee_eetag, generate base->ee
poses WITH rotational variety, derive the CONSISTENT world->eetag poses, and confirm
the solver recovers Z and X. Clean data -> tight tolerance; small noise -> loose.
Also confirm the conditioning warning fires on a no-variety (degenerate) set.

The math lives in Analyze_handeye_axzb so the test does not shell out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

cv2 = pytest.importorskip("cv2")

from Analyze_handeye_axzb import (  # noqa: E402
    handeye_residuals,
    rotation_geodesic_deg,
    rotation_variety_report,
    solve_handeye,
    t_world_eetag_from_vision,
)
from Utils.gaze.apriltag_calib import invert_transform, make_transform  # noqa: E402


def _T(rpy_deg, t_m):
    R = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    return make_transform(R, np.asarray(t_m, dtype=float))


def _synth_world(n, rng, rot_scale_deg=40.0):
    """Plant Z, X and generate n consistent (T_base_ee, T_world_eetag) pairs.

    Identity AX=ZB chain: T_world_eetag = inv(Z) @ T_base_ee @ X."""
    Z = _T([12.0, -7.0, 33.0], [0.40, -0.15, 0.22])     # T_base_world
    X = _T([5.0, -88.0, 17.0], [0.012, -0.004, 0.061])  # T_ee_eetag
    T_base_ee = []
    for _ in range(n):
        rpy = rng.uniform(-rot_scale_deg, rot_scale_deg, size=3)
        t = rng.uniform(-0.25, 0.25, size=3) + np.array([0.5, 0.0, 0.3])
        T_base_ee.append(_T(rpy, t))
    T_base_ee = np.stack(T_base_ee)
    T_world_eetag = np.stack([invert_transform(Z) @ A @ X for A in T_base_ee])
    return Z, X, T_base_ee, T_world_eetag


def _perturb(T, rng, rot_sigma_deg, trans_sigma_m):
    dR = Rotation.from_rotvec(rng.normal(0, np.radians(rot_sigma_deg), size=3)).as_matrix()
    dt = rng.normal(0, trans_sigma_m, size=3)
    out = T.copy()
    out[:3, :3] = dR @ T[:3, :3]
    out[:3, 3] = T[:3, 3] + dt
    return out


def test_recovery_clean_shah():
    rng = np.random.default_rng(0)
    Z, X, A, B = _synth_world(16, rng)
    res = solve_handeye(A, B, cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH)

    assert rotation_geodesic_deg(res["T_base_world"][:3, :3], Z[:3, :3]) < 1.0
    assert rotation_geodesic_deg(res["T_ee_eetag"][:3, :3], X[:3, :3]) < 1.0
    assert np.linalg.norm((res["T_base_world"][:3, 3] - Z[:3, 3]) * 1000) < 1.0
    assert np.linalg.norm((res["T_ee_eetag"][:3, 3] - X[:3, 3]) * 1000) < 1.0
    # Residual on clean data is essentially zero.
    assert res["rot_err_p95_deg"] < 1e-3
    assert res["trans_err_p95_mm"] < 1e-3


def test_recovery_clean_li():
    rng = np.random.default_rng(1)
    Z, X, A, B = _synth_world(16, rng)
    res = solve_handeye(A, B, cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI)
    assert rotation_geodesic_deg(res["T_base_world"][:3, :3], Z[:3, :3]) < 1.0
    assert rotation_geodesic_deg(res["T_ee_eetag"][:3, :3], X[:3, :3]) < 1.0
    assert np.linalg.norm((res["T_base_world"][:3, 3] - Z[:3, 3]) * 1000) < 1.0
    assert np.linalg.norm((res["T_ee_eetag"][:3, 3] - X[:3, 3]) * 1000) < 1.0


def test_recovery_with_noise():
    rng = np.random.default_rng(7)
    Z, X, A, B = _synth_world(40, rng)
    # Perturb the vision side (single-tag pose noise) — the realistic error source.
    B_noisy = np.stack([_perturb(b, rng, rot_sigma_deg=0.3, trans_sigma_m=0.0008) for b in B])
    res = solve_handeye(A, B_noisy, cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH)
    # Loose tolerance under noise.
    assert rotation_geodesic_deg(res["T_base_world"][:3, :3], Z[:3, :3]) < 3.0
    assert np.linalg.norm((res["T_base_world"][:3, 3] - Z[:3, 3]) * 1000) < 10.0


def test_world_eetag_helper_roundtrip():
    # T_world_eetag = inv(T_cam_world) @ T_cam_eetag.
    T_cam_world = _T([10, 20, 30], [0.1, 0.2, 1.5])
    T_cam_eetag = _T([-5, 8, 100], [0.05, -0.1, 1.2])
    got = t_world_eetag_from_vision(T_cam_world, T_cam_eetag)
    want = invert_transform(T_cam_world) @ T_cam_eetag
    assert np.allclose(got, want)


def test_residuals_zero_on_consistent_data():
    rng = np.random.default_rng(3)
    Z, X, A, B = _synth_world(12, rng)
    r = handeye_residuals(A, B, Z, X)
    assert r["rot_err_p95_deg"] < 1e-6
    assert r["trans_err_p95_mm"] < 1e-6


def test_variety_warns_on_degenerate():
    # All EE poses share one orientation: no rotational variety -> warn must fire.
    rng = np.random.default_rng(5)
    R_fixed = Rotation.from_euler("xyz", [3, -2, 9], degrees=True).as_matrix()
    R_list = np.stack([R_fixed for _ in range(15)])
    rep = rotation_variety_report(R_list)
    assert rep["warn"] is True
    assert rep["mean_pairwise_deg"] < rep["threshold_deg"]


def test_variety_ok_with_rotation():
    rng = np.random.default_rng(6)
    R_list = Rotation.random(20, random_state=rng).as_matrix()
    rep = rotation_variety_report(R_list)
    assert rep["warn"] is False
    assert rep["mean_pairwise_deg"] > rep["threshold_deg"]
