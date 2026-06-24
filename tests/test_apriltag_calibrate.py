"""
test_apriltag_calibrate.py — AprilTag calibration tool, hardware-free seams.

Covers what runs without a Neon/robot/pupil-apriltags: the offline `solve` stage
end-to-end (synthetic capture npz with a known T_base_world → recovered transform
+ consolidated _calib.npz), the intrinsics helper, and config-backed arg defaults.
The camera/robot stages (detect/gaze/collect) are validated on hardware.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import tools.apriltag_calibrate as calib  # noqa: E402
from Utils.gaze.apriltag_calib import make_transform  # noqa: E402
from Utils.gaze.apriltag_detect import camera_params, rescale_pose_t_mm  # noqa: E402


def _rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _write_capture(path, P_world, X, Q):
    np.savez_compressed(
        path,
        P_world=P_world, X=X, Q=Q,
        T_cam_world=np.zeros((P_world.shape[0], 4, 4)),
        T_cam_eetag=np.zeros((P_world.shape[0], 4, 4)),
        meta=np.array({"version": 3, "side": "R"}, dtype=object),
    )


def test_solve_recovers_transform_and_writes_calib(tmp_path, capsys):
    rng = np.random.default_rng(7)
    P_world = rng.normal(size=(20, 3)) * 100.0
    T_true = make_transform(_rot_z(33.0), [200.0, -50.0, 60.0])
    X = (T_true[:3, :3] @ P_world.T).T + T_true[:3, 3]   # EE-in-base = T·P_world
    Q = rng.normal(size=(20, 7))
    cap = tmp_path / "apriltag_capture_test.npz"
    _write_capture(cap, P_world, X, Q)

    rc = calib.main(["--stage", "solve", str(cap)])
    assert rc == 0
    assert "PASS" in capsys.readouterr().out

    out = tmp_path / "apriltag_test_calib.npz"
    assert out.is_file()
    z = np.load(out, allow_pickle=True)
    np.testing.assert_allclose(z["T_base_world"], T_true, atol=1e-6)
    assert z["X"].shape == (20, 3) and z["Q"].shape == (20, 7)


def test_solve_skips_nonfinite_rows(tmp_path):
    P_world = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=float)
    X = P_world + 5.0
    Q = np.zeros((4, 7))
    X[0] = np.nan  # a dry-run / failed capture row
    _write_capture(tmp_path / "cap_nan.npz", P_world, X, Q)
    assert calib.main(["--stage", "solve", str(tmp_path / "cap_nan.npz")]) == 0


def test_solve_too_few_points_returns_1(tmp_path):
    P_world = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    _write_capture(tmp_path / "tiny.npz", P_world, P_world + 1.0, np.zeros((2, 7)))
    assert calib.main(["--stage", "solve", str(tmp_path / "tiny.npz")]) == 1


def test_solve_missing_file_returns_2():
    assert calib.main(["--stage", "solve", "does_not_exist_42.npz"]) == 2


def test_camera_params_from_K():
    K = np.array([[1490.0, 0, 800.0], [0, 1480.0, 600.0], [0, 0, 1]])
    assert camera_params(K) == (1490.0, 1480.0, 800.0, 600.0)


def test_rescale_pose_t_mm_per_tag_size():
    # Detected at the world size → mm unchanged; a half-size (EE) tag → half the
    # translation, so world and EE tags can differ in size.
    pose_t_m = [0.0, 0.0, 0.5]
    np.testing.assert_allclose(rescale_pose_t_mm(pose_t_m, 0.06, 0.06), [0, 0, 500.0])
    np.testing.assert_allclose(rescale_pose_t_mm(pose_t_m, 0.03, 0.06), [0, 0, 250.0])


def test_arg_defaults_come_from_config():
    import config as cfg
    args = calib.parse_args(["--stage", "detect", "--world-tag-id", "0"])
    assert args.relay_host == getattr(cfg, "FRAME_RELAY_DIAL_HOST")
    assert args.bind_ip == cfg.UDP_CONTROL_BIND["IP"]
    assert args.side in ("R", "L")


# ── step 2: EE-tag bundle map generalisation ─────────────────────────────────


def test_resolve_ee_ids_plural_overrides_singular_else_none():
    plural = calib.parse_args(["--stage", "collect", "--ee-tag-ids", "5", "6", "7"])
    assert calib._resolve_ee_ids(plural) == [5, 6, 7]
    # --ee-tag-ids wins when both are given
    both = calib.parse_args(["--stage", "collect", "--ee-tag-id", "2",
                             "--ee-tag-ids", "5", "6"])
    assert calib._resolve_ee_ids(both) == [5, 6]
    singular = calib.parse_args(["--stage", "collect", "--ee-tag-id", "9"])
    assert calib._resolve_ee_ids(singular) == [9]
    neither = calib.parse_args(["--stage", "collect", "--world-tag-ids", "0"])
    assert calib._resolve_ee_ids(neither) is None


def test_ee_single_tag_map_equals_direct_tag_pose():
    """Back-compat: a 1-entry EE map reproduces the old single-tag P_world path
    exactly (recover_world_pose returns the tag pose itself, so eetag_to_world_point
    composes identically)."""
    from Utils.gaze.apriltag_world import build_world_map, recover_world_pose
    from Utils.gaze.apriltag_calib import eetag_to_world_point

    T_cam_world = make_transform(_rot_z(15.0), [40.0, -10.0, 600.0])
    T_cam_eetag = make_transform(_rot_z(-25.0), [120.0, 30.0, 550.0])
    offset = np.array([0.0, 0.0, 35.0])

    ee_map = build_world_map({9: T_cam_eetag})
    recovered = recover_world_pose({9: T_cam_eetag}, ee_map)
    np.testing.assert_allclose(recovered, T_cam_eetag, atol=1e-9)

    via_map = eetag_to_world_point(T_cam_world, recovered, offset)
    via_direct = eetag_to_world_point(T_cam_world, T_cam_eetag, offset)
    np.testing.assert_allclose(via_map, via_direct, atol=1e-9)


def test_ee_bundle_recovers_consistent_world_point_under_occlusion():
    """An EE bundle (≥2 rigid tags) recovers the SAME EE world point from any
    visible subset — the occlusion-robustness the HIL accuracy floor needed
    (verification report §5). Built on the analytically-known EE-reference frame."""
    from Utils.gaze.apriltag_world import build_world_map, recover_world_pose
    from Utils.gaze.apriltag_calib import eetag_to_world_point, ee_point_in_world, invert_transform

    T_cam_world = make_transform(_rot_z(10.0), [0.0, 0.0, 700.0])
    # EE reference-tag frame in the camera; second tag rigidly offset on the EE body.
    T_cam_eeref = make_transform(_rot_z(-30.0), [150.0, -40.0, 520.0])
    rel_2 = make_transform(_rot_z(90.0), [25.0, 0.0, 0.0])  # tag 2 in ee-ref frame
    T_cam_ee2 = T_cam_eeref @ rel_2
    offset = np.array([10.0, 0.0, 20.0])

    ee_map = build_world_map({3: T_cam_eeref, 4: T_cam_ee2}, ref_id=3)

    # Analytic expectation: EE point in world from the true EE-ref pose.
    expected = ee_point_in_world(invert_transform(T_cam_world) @ T_cam_eeref, offset)

    for view in ({3: T_cam_eeref, 4: T_cam_ee2}, {3: T_cam_eeref}, {4: T_cam_ee2}):
        T_cam_eetag = recover_world_pose(view, ee_map)
        p_world = eetag_to_world_point(T_cam_world, T_cam_eetag, offset)
        np.testing.assert_allclose(p_world, expected, atol=1e-6)
