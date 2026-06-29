"""
test_apriltag_calibrate_3d.py — WS-4 3-D calibration path of the AprilTag tool.

Hardware-free seams of the two NEW additive stages:

  - ``solve-3d``: a synthetic sweep npz (in-memory transforms) → a world-frame
    ``(x,y,z)→Q`` calib npz whose points retain their z-variation (NOT projected
    to the table plane), loadable by ``GazeCalibration3D.from_calib_npz`` with an
    exact-point query returning the right ``Q``.
  - ``register-world-3d``: a non-coplanar map round-trips through the
    ``_save_world_map_3d`` / ``_load_world_map`` array (de)serialisers.

The 2-D planar/rigid solve is covered (and asserted unchanged) by
``tests/test_apriltag_calibrate.py``; this file only exercises the new 3-D seams.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import tools.apriltag_calibrate as calib  # noqa: E402
from Utils.gaze.apriltag_calib import eetag_to_world_point, make_transform  # noqa: E402
from Utils.gaze.apriltag_world_3d import (  # noqa: E402
    register_world_map_3d,
    world_map_3d_geometry_report,
)
from Utils.gaze.calibration_mapping_3d import GazeCalibration3D  # noqa: E402


def _rot_y(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _write_sweep_3d(path, Tce, Q, *, X=None, green=None,
                    method="pose", offset=(0.0, 0.0, 0.0)):
    """Minimal sweep npz the 3-D solve consumes: identity world pose so the EE
    world point equals each EE-tag origin (offset 0), plus the per-sample
    transforms / Q / world map / meta (carrying the EE-point recipe). X defaults to
    NaN (no-robot dry run on the BASE frame is irrelevant to the world-frame solve)."""
    N = Tce.shape[0]
    Tcw = np.broadcast_to(np.eye(4), (N, 4, 4)).copy()
    if X is None:
        X = np.full((N, 3), np.nan)
    extra = {} if green is None else {"green": np.asarray(green, dtype=bool)}
    np.savez_compressed(
        path,
        UV=np.zeros((N, 2)), Q=Q, X=X,
        T_cam_world=Tcw, T_cam_eetag=Tce,
        world_map_ref=np.array(0), world_map_ids=np.array([0]),
        world_map_rels=np.eye(4)[None, :, :],
        world_map_plane_point=np.zeros(3),
        world_map_plane_normal=np.array([0.0, 0.0, 1.0]),
        meta=np.array({"version": 3, "scheme": "planar_sweep", "side": "R",
                       "ee_point_method": method,
                       "t_eetag_ee_mm": list(offset)}, dtype=object),
        **extra,
    )


# ── solve-3d ─────────────────────────────────────────────────────────────────


def test_solve_3d_builds_library_loadable_with_exact_query(tmp_path):
    """solve-3d writes P_WORLD3D/Q; GazeCalibration3D.from_calib_npz loads it and
    an exact-point query returns that row's joints."""
    N = 12
    # EE-tag origins climb in all three axes — genuine 3-D structure.
    Tce = np.stack([make_transform(_rot_z(3.0 * i), [20.0 * i, 10.0 * i, 30.0 * i])
                    for i in range(N)])
    Q = np.linspace(0.1, 0.6, N)[:, None] * np.ones((1, 7))
    sweep = tmp_path / "apriltag_sweep_3dexact.npz"
    _write_sweep_3d(sweep, Tce, Q)

    assert calib.main(["--stage", "solve-3d", str(sweep), "--include-partial"]) == 0
    calib_path = tmp_path / "apriltag_3d_3dexact_calib.npz"
    assert calib_path.is_file()

    z = np.load(calib_path, allow_pickle=True)
    assert "P_WORLD3D" in z.files and z["P_WORLD3D"].shape == (N, 3)
    # The stored points are the full-3-D EE world points (identity world, offset 0).
    expected = np.stack([eetag_to_world_point(np.eye(4), Tce[i], np.zeros(3))
                         for i in range(N)])
    np.testing.assert_allclose(z["P_WORLD3D"], expected, atol=1e-9)
    meta = z["meta"].item()
    assert meta["scheme"] == "world_xyz_nn" and meta["n_points"] == N
    assert "world_map_plane_normal" in z.files  # world map carried through

    m = GazeCalibration3D.from_calib_npz(z)
    assert m.num_valid_samples == N
    res = m.query_xyz(expected[5])
    assert res.dist < 1e-9
    np.testing.assert_allclose(res.q_target, Q[5], atol=1e-9)


def test_solve_3d_retains_z_variation_not_plane_projected(tmp_path):
    """The defining 3-D property: the library keeps the table-height z instead of
    collapsing it onto the plane (what the planar solve's plane_coords would do)."""
    N = 8
    Tce = np.stack([make_transform(np.eye(3), [10.0 * i, 0.0, 50.0 * i])
                    for i in range(N)])  # z spans 0..350 mm
    Q = np.linspace(0.1, 0.4, N)[:, None] * np.ones((1, 7))
    sweep = tmp_path / "apriltag_sweep_zvar.npz"
    _write_sweep_3d(sweep, Tce, Q)

    assert calib.main(["--stage", "solve-3d", str(sweep), "--include-partial"]) == 0
    z = np.load(tmp_path / "apriltag_3d_zvar_calib.npz", allow_pickle=True)
    P = z["P_WORLD3D"]
    assert np.ptp(P[:, 2]) > 300.0     # NOT flattened to one plane
    assert z["meta"].item()["z_range_mm"] > 300.0


def test_solve_3d_green_only_filters_partial_cells(tmp_path, capsys):
    """Green-cell filtering mirrors the planar solve: partial-cell samples are
    dropped by default; --include-partial keeps all."""
    N = 6
    Tce = np.stack([make_transform(np.eye(3), [10.0 * i, 5.0 * i, 20.0 * i])
                    for i in range(N)])
    Q = np.linspace(0.1, 0.6, N)[:, None] * np.ones((1, 7))
    green = np.array([True, True, True, True, False, False])
    sweep = tmp_path / "apriltag_sweep_g3d.npz"
    _write_sweep_3d(sweep, Tce, Q, green=green)

    assert calib.main(["--stage", "solve-3d", str(sweep)]) == 0
    assert "green-only: 4/6" in capsys.readouterr().out
    z = np.load(tmp_path / "apriltag_3d_g3d_calib.npz", allow_pickle=True)
    assert z["P_WORLD3D"].shape == (4, 3)

    assert calib.main(["--stage", "solve-3d", "--include-partial", str(sweep)]) == 0
    z2 = np.load(tmp_path / "apriltag_3d_g3d_calib.npz", allow_pickle=True)
    assert z2["P_WORLD3D"].shape == (6, 3)


def test_solve_3d_drops_nonfinite_rows_and_too_few_returns_1(tmp_path):
    """A dry-run sweep leaves Q NaN; those rows drop, and <3 finite rows → rc 1."""
    N = 4
    Tce = np.stack([make_transform(np.eye(3), [10.0 * i, 0.0, 20.0 * i])
                    for i in range(N)])
    Q = np.full((N, 7), np.nan)        # no robot → all rows non-finite
    sweep = tmp_path / "apriltag_sweep_dry3d.npz"
    _write_sweep_3d(sweep, Tce, Q)
    assert calib.main(["--stage", "solve-3d", str(sweep), "--include-partial"]) == 1


def test_solve_3d_rejects_non_sweep_npz(tmp_path):
    """An npz without the per-sample transforms is not a sweep capture → rc 2."""
    bad = tmp_path / "apriltag_capture_x.npz"
    np.savez_compressed(bad, P_world=np.zeros((3, 3)), X=np.zeros((3, 3)),
                        Q=np.zeros((3, 7)), meta=np.array({}, dtype=object))
    assert calib.main(["--stage", "solve-3d", str(bad)]) == 2


def test_solve_3d_missing_file_returns_2():
    assert calib.main(["--stage", "solve-3d", "does_not_exist_3d.npz"]) == 2


def test_from_calib_npz_rejects_non_3d_calib():
    """The 3-D loader reads P_WORLD3D, not X — a planar calib (UV/X) is rejected
    rather than silently mis-read as base-frame points."""
    with pytest.raises(KeyError):
        GazeCalibration3D.from_calib_npz({"X": np.zeros((3, 3)), "Q": np.zeros((3, 7))})


# ── register-world-3d ────────────────────────────────────────────────────────


def _two_plane_frame():
    """One clean (noise-free) view of a two-plane layout: tags 0-2 on the table
    (z=0), tags 3-4 on a raised back panel (z=200/400). Non-coplanar by design."""
    world_T = {
        0: make_transform(np.eye(3), [0.0, 0.0, 0.0]),
        1: make_transform(_rot_z(15.0), [400.0, 0.0, 0.0]),
        2: make_transform(np.eye(3), [0.0, 300.0, 0.0]),
        3: make_transform(_rot_y(90.0), [600.0, 0.0, 200.0]),
        4: make_transform(_rot_y(90.0) @ _rot_z(10.0), [600.0, 300.0, 400.0]),
    }
    T_cam_world = make_transform(_rot_z(25.0), [10.0, -15.0, 800.0])
    return world_T, {i: T_cam_world @ world_T[i] for i in world_T}


def test_register_world_3d_map_round_trips_non_coplanar(tmp_path):
    """_save_world_map_3d → _load_world_map preserves the non-coplanar 3-D origins
    (the round-trip the register-world-3d stage relies on)."""
    world_T, frame = _two_plane_frame()
    wm = register_world_map_3d([frame], ref_id=0)
    report = world_map_3d_geometry_report(wm, [frame])
    # z_spread_mm is the extent along the best-fit-plane NORMAL (tilted for this
    # two-plane layout), so it reads ~157 mm — far above the ~0 a coplanar snap would
    # leave. The raw world-z preservation (~400 mm) is asserted below via the loaded
    # origins.
    assert report["z_spread_mm"] > 100.0   # genuine 3-D structure, not snapped flat

    path = calib._save_world_map_3d(wm, str(tmp_path), "20260101T000000Z",
                                    0.06, [0, 1, 2, 3, 4], report)
    assert path.name == "world_map_3d_20260101T000000Z.npz"

    loaded = calib._load_world_map(str(path))
    assert loaded["ids"] == wm["ids"]
    for i in wm["ids"]:
        np.testing.assert_allclose(loaded["rel"][i], wm["rel"][i], atol=1e-9)
    z = np.array([loaded["rel"][i][:3, 3][2] for i in loaded["ids"]])
    assert np.ptp(z) > 300.0               # height preserved through save/load
    # The geometry report survives in meta for later inspection.
    meta = np.load(path, allow_pickle=True)["meta"].item()
    assert meta["stage"] == "register-world-3d"
    assert meta["geometry_3d"]["z_spread_mm"] > 100.0
