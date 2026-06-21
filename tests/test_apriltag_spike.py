"""
test_apriltag_spike.py — WS5 spike CLI, hardware-free seams.

Covers what runs without a Neon/robot/pupil-apriltags: the offline `solve`
stage end-to-end against a synthetic npz with a known T_base_world, the
intrinsics helper, and the config-backed arg defaults. The camera stages
(detect/gaze/collect) and the lazy pupil-apriltags detector need hardware and
are validated on Monday's HIL run, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import tools.apriltag_calib_spike as spike  # noqa: E402
from Utils.gaze.apriltag_calib import make_transform  # noqa: E402


def _rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _write_npz(path, P_world, P_base):
    np.savez_compressed(
        path,
        P_world=P_world,
        P_base=P_base,
        T_cam_world=np.zeros((P_world.shape[0], 4, 4)),
        T_cam_eetag=np.zeros((P_world.shape[0], 4, 4)),
        meta=np.array({"version": 3}, dtype=object),
    )


def test_solve_recovers_known_transform(tmp_path, capsys):
    rng = np.random.default_rng(7)
    P_world = rng.normal(size=(20, 3)) * 100.0
    T_true = make_transform(_rot_z(33.0), [200.0, -50.0, 60.0])
    P_base = (T_true[:3, :3] @ P_world.T).T + T_true[:3, 3]
    npz = tmp_path / "apriltag_spike_test.npz"
    _write_npz(npz, P_world, P_base)

    rc = spike.main(["--stage", "solve", str(npz)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS" in out  # clean synthetic data → residual well under the bar


def test_solve_skips_nonfinite_rows(tmp_path):
    # Rows with NaN P_base (a dry-run capture with no --with-robot) are dropped.
    P_world = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=float)
    P_base = P_world + 5.0
    P_base[0] = np.nan
    npz = tmp_path / "apriltag_spike_nan.npz"
    _write_npz(npz, P_world, P_base)
    assert spike.main(["--stage", "solve", str(npz)]) == 0  # 3 finite rows remain


def test_solve_too_few_points_returns_1(tmp_path):
    P_world = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    _write_npz(tmp_path / "tiny.npz", P_world, P_world + 1.0)
    assert spike.main(["--stage", "solve", str(tmp_path / "tiny.npz")]) == 1


def test_solve_missing_file_returns_2():
    assert spike.main(["--stage", "solve", "does_not_exist_42.npz"]) == 2


def test_camera_params_from_K():
    K = np.array([[1490.0, 0, 800.0], [0, 1480.0, 600.0], [0, 0, 1]])
    assert spike._camera_params(K) == (1490.0, 1480.0, 800.0, 600.0)


def test_arg_defaults_come_from_config():
    import config as cfg
    args = spike.parse_args(["--stage", "detect", "--world-tag-id", "0"])
    assert args.relay_host == getattr(cfg, "FRAME_RELAY_DIAL_HOST")
    assert args.relay_port == int(getattr(cfg, "FRAME_RELAY_PORT"))
    assert args.robot_ip == cfg.UDP_ROBOT["IP"]
    assert args.side in ("R", "L")


def test_solve_stage_requires_npz_arg():
    assert spike.main(["--stage", "solve"]) == 2
