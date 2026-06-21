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
    average_rotation,
    ee_point_in_world,
    eetag_to_world_point,
    gaze_ray_cam,
    geodesic_angle_deg,
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


def test_umeyama_returns_proper_rotation_on_true_reflection():
    # A genuine mirror of src (det would be -1 without the sign correction).
    # This actually exercises the d=-1 branch — verified to drive the correction,
    # unlike a near-planar config where d stays +1.
    rng = np.random.default_rng(1)
    src = rng.normal(size=(30, 3)) * 50.0
    dst = src.copy()
    dst[:, 2] *= -1.0  # reflection across the xy-plane
    T, _ = umeyama_rigid(src, dst)
    # Must return a PROPER rotation (det=+1), not the reflecting fit.
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0, abs=1e-9)


def test_umeyama_direction_inverse_roundtrip():
    # Swapping src/dst must give the inverse transform — pins the src→dst
    # orientation against a future xd/xs swap.
    rng = np.random.default_rng(5)
    src = rng.normal(size=(15, 3)) * 30.0
    R_true = _rot_z(20.0) @ _rot_x(35.0)
    dst = (R_true @ src.T).T + np.array([10.0, 20.0, 30.0])
    T_fwd, _ = umeyama_rigid(src, dst)
    T_inv, _ = umeyama_rigid(dst, src)
    np.testing.assert_allclose(T_inv @ T_fwd, np.eye(4), atol=1e-9)


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


# ── frame-chain compose (the §4.2 collect heart, previously untested) ─────────


def test_eetag_to_world_point_is_invariant_to_camera_pose():
    # Known EE-tag pose in WORLD + a known mount offset → a known EE-in-world
    # point. Recovering it through TWO different (arbitrary) camera poses must
    # give the SAME answer — this is the §1 head-pose-cancels guarantee.
    R_we = _rot_z(40.0) @ _rot_x(15.0)
    t_we = np.array([300.0, -120.0, 50.0])
    T_world_eetag = make_transform(R_we, t_we)
    offset = np.array([0.0, 0.0, 40.0])  # mm, in the eetag frame
    expected = t_we + R_we @ offset

    for cam_pose in (
        make_transform(_rot_x(25.0), [10.0, 0.0, 600.0]),
        make_transform(_rot_z(-70.0) @ _rot_x(80.0), [-200.0, 90.0, 1100.0]),
    ):
        # cam sees both tags: T_cam_world and T_cam_eetag share this cam pose.
        T_cam_world = cam_pose
        T_cam_eetag = cam_pose @ T_world_eetag
        p = eetag_to_world_point(T_cam_world, T_cam_eetag, offset)
        np.testing.assert_allclose(p, expected, atol=1e-9)


# ── rotation jitter helpers ──────────────────────────────────────────────────


def test_average_rotation_of_symmetric_pair_is_identity():
    R = average_rotation([_rot_z(12.0), _rot_z(-12.0)])
    np.testing.assert_allclose(R, np.eye(3), atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)


def test_geodesic_angle_deg():
    assert geodesic_angle_deg(np.eye(3), _rot_z(30.0)) == pytest.approx(30.0, abs=1e-9)
    assert geodesic_angle_deg(_rot_x(50.0), _rot_x(50.0)) == pytest.approx(0.0, abs=1e-9)


def test_ray_plane_hits_tilted_plane_on_plane_and_ray():
    # Non-axis-aligned plane (normal tilted), ray down +z. Hit must lie on the
    # plane and on the ray.
    normal = _rot_x(20.0) @ np.array([0.0, 0.0, 1.0])
    plane_pt = np.array([0.0, 0.0, 500.0])
    direction = np.array([0.1, 0.05, 1.0])
    hit = ray_plane_intersection([0, 0, 0], direction, plane_pt, normal)
    assert hit is not None
    assert abs((hit - plane_pt) @ normal) < 1e-9          # on the plane
    cross = np.cross(hit, direction)
    np.testing.assert_allclose(cross, np.zeros(3), atol=1e-9)  # on the ray (through origin)


# ── angle helper ─────────────────────────────────────────────────────────────


def test_angle_between_deg():
    assert angle_between_deg([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
    assert angle_between_deg([1, 0, 0], [1, 0, 0]) == pytest.approx(0.0)
    assert np.isnan(angle_between_deg([0, 0, 0], [1, 0, 0]))
