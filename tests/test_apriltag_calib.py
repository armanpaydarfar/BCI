"""
test_apriltag_calib.py — WS5 REV03 AprilTag calibration math core.

Exercises the pure-geometry pieces of Utils/gaze/apriltag_calib.py with
synthetic transforms and points — no Neon, robot, or pupil-apriltags. Pins the
load-bearing properties: the Umeyama solve recovers a known rigid transform and
never reflects, gaze unprojection + ray-plane intersection agree on a closed
form, and the EE-in-world offset composes correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from Utils.gaze.apriltag_calib import (  # noqa: E402
    angle_between_deg,
    ee_point_in_world,
    gaze_ray_cam,
    invert_transform,
    make_transform,
    per_point_errors,
    ray_plane_intersection,
    tag_plane_in_cam,
    transform_point,
    umeyama_rigid,
)


def _rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


# ── transform helpers ───────────────────────────────────────────────────────


def test_invert_transform_roundtrip():
    T = make_transform(_rot_z(37.0) @ _rot_x(20.0), [10.0, -5.0, 3.0])
    np.testing.assert_allclose(invert_transform(T) @ T, np.eye(4), atol=1e-12)


def test_transform_point_matches_manual():
    R = _rot_z(90.0)
    T = make_transform(R, [1.0, 2.0, 3.0])
    # +x axis rotates to +y, then translate.
    np.testing.assert_allclose(transform_point(T, [1.0, 0.0, 0.0]), [1.0, 3.0, 3.0], atol=1e-12)


# ── Umeyama rigid solve ──────────────────────────────────────────────────────


def test_umeyama_recovers_known_rigid_transform():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(25, 3)) * 100.0  # mm-scale spread
    R_true = _rot_z(25.0) @ _rot_x(40.0)
    t_true = np.array([120.0, -30.0, 75.0])
    dst = (R_true @ src.T).T + t_true

    T, rms = umeyama_rigid(src, dst)
    np.testing.assert_allclose(T[:3, :3], R_true, atol=1e-9)
    np.testing.assert_allclose(T[:3, 3], t_true, atol=1e-9)
    assert rms < 1e-9


def test_umeyama_returns_proper_rotation_not_reflection():
    # Near-planar points (a degenerate-ish config) must still yield det(R)=+1.
    rng = np.random.default_rng(1)
    src = rng.normal(size=(20, 3))
    src[:, 2] *= 1e-3  # squash onto a plane
    R_true = _rot_x(15.0)
    dst = (R_true @ src.T).T + np.array([5.0, 5.0, 5.0])
    T, _ = umeyama_rigid(src, dst)
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0, abs=1e-9)


def test_umeyama_residual_reports_noise():
    rng = np.random.default_rng(2)
    src = rng.normal(size=(50, 3)) * 50.0
    R_true = _rot_z(10.0)
    dst = (R_true @ src.T).T + np.array([1.0, 2.0, 3.0])
    dst_noisy = dst + rng.normal(scale=2.0, size=dst.shape)
    T, rms = umeyama_rigid(src, dst_noisy)
    assert rms > 0.5  # residual surfaces the injected noise
    errs = per_point_errors(T, src, dst_noisy)
    assert errs.shape == (50,)
    assert np.isclose(np.sqrt(np.mean(errs ** 2)), rms)


def test_umeyama_rejects_too_few_points():
    with pytest.raises(ValueError):
        umeyama_rigid(np.zeros((2, 3)), np.zeros((2, 3)))


# ── gaze ray unprojection ────────────────────────────────────────────────────


def _K(fx=1490.0, fy=1490.0, cx=800.0, cy=600.0):
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def test_gaze_ray_at_principal_point_is_optical_axis():
    ray = gaze_ray_cam(800.0, 600.0, _K())
    np.testing.assert_allclose(ray, [0.0, 0.0, 1.0], atol=1e-12)


def test_gaze_ray_offset_pixel_points_correct_quadrant():
    ray = gaze_ray_cam(900.0, 600.0, _K())  # +100 px in x
    assert ray[0] > 0 and ray[2] > 0
    assert ray[1] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(np.linalg.norm(ray), 1.0, atol=1e-12)


def test_gaze_ray_nan_returns_none():
    assert gaze_ray_cam(float("nan"), 600.0, _K()) is None


# ── ray-plane intersection ───────────────────────────────────────────────────


def test_ray_plane_hits_frontal_plane():
    # Plane z=500 with normal +z; ray straight down the optical axis.
    hit = ray_plane_intersection([0, 0, 0], [0, 0, 1], [0, 0, 500.0], [0, 0, 1])
    np.testing.assert_allclose(hit, [0, 0, 500.0], atol=1e-9)


def test_ray_plane_parallel_returns_none():
    assert ray_plane_intersection([0, 0, 0], [1, 0, 0], [0, 0, 500.0], [0, 0, 1]) is None


def test_ray_plane_behind_origin_returns_none():
    assert ray_plane_intersection([0, 0, 0], [0, 0, -1], [0, 0, 500.0], [0, 0, 1]) is None


def test_tag_plane_in_cam_extracts_origin_and_z_axis():
    R = _rot_x(30.0)
    T = make_transform(R, [10.0, 20.0, 300.0])
    point, normal = tag_plane_in_cam(T)
    np.testing.assert_allclose(point, [10.0, 20.0, 300.0], atol=1e-12)
    np.testing.assert_allclose(normal, R[:, 2], atol=1e-12)


# ── EE-in-world offset ───────────────────────────────────────────────────────


def test_ee_point_in_world_identity_rotation():
    T = make_transform(np.eye(3), [100.0, 0.0, 0.0])
    p = ee_point_in_world(T, [0.0, 0.0, 50.0])
    np.testing.assert_allclose(p, [100.0, 0.0, 50.0], atol=1e-12)


def test_ee_point_in_world_rotates_offset():
    T = make_transform(_rot_z(90.0), [0.0, 0.0, 0.0])
    # offset +x (10mm) rotates to +y under a 90° z-rotation.
    p = ee_point_in_world(T, [10.0, 0.0, 0.0])
    np.testing.assert_allclose(p, [0.0, 10.0, 0.0], atol=1e-12)


# ── angle helper ─────────────────────────────────────────────────────────────


def test_angle_between_deg():
    assert angle_between_deg([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
    assert angle_between_deg([1, 0, 0], [1, 0, 0]) == pytest.approx(0.0)
    assert np.isnan(angle_between_deg([0, 0, 0], [1, 0, 0]))
