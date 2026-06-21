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
from Utils.gaze.apriltag_detect import camera_params  # noqa: E402


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


def test_arg_defaults_come_from_config():
    import config as cfg
    args = calib.parse_args(["--stage", "detect", "--world-tag-id", "0"])
    assert args.relay_host == getattr(cfg, "FRAME_RELAY_DIAL_HOST")
    assert args.bind_ip == cfg.UDP_CONTROL_BIND["IP"]
    assert args.side in ("R", "L")
