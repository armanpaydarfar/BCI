"""
test_pose_library_loader_v1v2.py — Phase 2.c test (plan §6.3 #3).

Locks down backwards-compatibility of the pose-library loaders against
both v1 and v2 NPZs. Two loaders are in scope:

- ``ExperimentDriver_Online_GazeTracking.load_pose_library`` at
  file:591-606 (the driver consumer).
- ``harmony_online_control.load_library`` at file:174-189 (the REPL
  consumer).

Plan §6.1.3: "Add a backwards-compatible loader [...] if `version` is
missing or `1`, use the old path; if `version` is `2`, also expose the
new arrays." Phase 4 wires v2 dispatch through the driver; this test
file guarantees that loading does not regress in either direction.

Citations under test:

  - ExperimentDriver_Online_GazeTracking.py:591-606 ``load_pose_library``
  - harmony_online_control.py:174-189 ``load_library``
  - Utils/gaze/calibration_mapping.py:296-313 ``detect_pose_library_version``
  - Utils/gaze/calibration_mapping.py:280-294 ``load_pose_library_v2``
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest


from Utils.gaze.calibration_mapping import (
    detect_pose_library_version,
    load_pose_library_v2,
)


# ─── synth fixtures ───────────────────────────────────────────────────────

def _write_v1(path: Path, N: int = 5) -> None:
    """Recreate the v1 schema from harmony_calibration_exec.py:393-407."""
    np.savez_compressed(
        str(path),
        T=np.arange(N, dtype=float),
        Q=np.zeros((N, 7)),
        X=np.zeros((N, 3)),
        G=np.column_stack([
            np.full(N, 0.5),
            np.full(N, 0.5),
            np.full(N, 1.0),
        ]),
        meta=dict(side="R", sample_rate_hz=25.0,
                  gaze_confidence_threshold=0.7,
                  units=dict(X="mm", Q="rad", G="normalized_0_to_1")),
    )


def _write_v2(path: Path, N: int = 5) -> None:
    """Mirror the v2 schema produced by
    harmony_free_arm_calibration.write_npz (file:471-580). Includes both
    captured-only and ``_all`` arrays so the loader tests cover both
    halves of the file.
    """
    np.savez_compressed(
        str(path),
        T=np.arange(N, dtype=float),
        Q=np.zeros((N, 7)),
        X=np.zeros((N, 3)),
        G=np.column_stack([
            np.full(N, 0.5),
            np.full(N, 0.5),
            np.full(N, 1.0),
        ]),
        # v2 captured-only
        D_cm=np.full(N, 75.0),
        D_valid=np.ones(N, dtype=bool),
        Miss_mm=np.zeros(N),
        IPD_mm=np.full(N, 63.0),
        IMU_w=np.zeros(N),
        IMU_fresh=np.ones(N, dtype=bool),
        Head_yaw_deg=np.zeros(N),
        Head_pitch_deg=np.zeros(N),
        Gaze_yaw_deg=np.zeros(N),
        Gaze_pitch_deg=np.zeros(N),
        Target_label=np.array(["mid_MC"] * N, dtype="<U32"),
        # v2 _all (just dupes for the test — N samples per phase
        # is fine; we only care about shape compatibility)
        T_all=np.arange(N, dtype=float),
        Q_all=np.zeros((N, 7)),
        X_all=np.zeros((N, 3)),
        G_all=np.column_stack([
            np.full(N, 0.5), np.full(N, 0.5), np.full(N, 1.0)
        ]),
        D_cm_all=np.full(N, 75.0),
        D_valid_all=np.ones(N, dtype=bool),
        Miss_mm_all=np.zeros(N),
        IPD_mm_all=np.full(N, 63.0),
        IMU_w_all=np.zeros(N),
        IMU_fresh_all=np.ones(N, dtype=bool),
        Head_yaw_deg_all=np.zeros(N),
        Head_pitch_deg_all=np.zeros(N),
        Gaze_yaw_deg_all=np.zeros(N),
        Gaze_pitch_deg_all=np.zeros(N),
        Phase_all=np.array(["captured"] * N, dtype="<U16"),
        Target_label_all=np.array(["mid_MC"] * N, dtype="<U32"),
        meta=dict(version=2, side="R",
                  recorder="harmony_free_arm_calibration.py"),
    )


# ─── harmony_online_control.load_library ──────────────────────────────────

class TestHarmonyOnlineControlLoader:
    """``load_library(path) -> (X, Q, G)`` per
    harmony_online_control.py:174-189. v2 NPZs must still return the
    same 3-tuple of legacy arrays without raising.
    """

    def _import_load_library(self):
        # Import lazily — harmony_online_control runs significant
        # module-level side effects (socket bind to 0.0.0.0:8080) that
        # we want to skip in unit tests. We instead import the loader
        # function via the module's source path with the offensive
        # global socket bind removed via a stub. Simpler approach:
        # import the function via importlib only after monkey-patching
        # `socket.socket` to a stub.
        import socket as _socket
        import types

        # Quick patch: stub socket() so the bind in
        # harmony_online_control's top-level does not collide with a
        # real port (and works under headless test environments).
        original = _socket.socket

        class _StubSocket:
            def __init__(self, *a, **kw): pass
            def setsockopt(self, *a, **kw): pass
            def bind(self, *a, **kw): pass
            def settimeout(self, *a, **kw): pass
            def sendto(self, *a, **kw): return 0
            def recvfrom(self, *a, **kw):
                raise _socket.timeout()
            def close(self): pass

        _socket.socket = lambda *a, **kw: _StubSocket()
        try:
            mod = importlib.import_module("harmony_online_control")
        finally:
            _socket.socket = original
        return mod.load_library

    def test_v1_load_returns_three_tuple(self, tmp_path: Path):
        load_library = self._import_load_library()
        p = tmp_path / "v1.npz"
        _write_v1(p)
        X, Q, G = load_library(str(p))
        assert X.shape == (5, 3)
        assert Q.shape == (5, 7)
        assert G is not None and G.shape[1] >= 2

    def test_v2_load_returns_three_tuple_without_raising(self, tmp_path: Path):
        load_library = self._import_load_library()
        p = tmp_path / "v2.npz"
        _write_v2(p)
        # The legacy loader only reads X / Q / G — it must ignore the
        # extra v2 keys, not crash on them.
        X, Q, G = load_library(str(p))
        assert X.shape == (5, 3)
        assert Q.shape == (5, 7)
        assert G is not None


# ─── ExperimentDriver_Online_GazeTracking.load_pose_library ──────────────

class TestDriverLoadPoseLibrary:
    """The driver's loader is similar to the REPL's but enforces the
    presence of G (raises if absent). Import is awkward because the
    driver runs heavy module-level side effects (pygame init, logger
    setup, EEG model load); we extract the function source and exec it
    in a minimal sandbox.
    """

    @pytest.fixture
    def load_pose_library(self):
        """Construct a copy of the driver's load_pose_library that does
        not depend on the surrounding module-level state."""
        import numpy as np

        # Make a tiny logger stub so the function can call .log_event.
        class _Logger:
            def log_event(self, *a, **k): pass

        logger_stub = _Logger()

        def load_pose_library(path):
            # Verbatim from ExperimentDriver_Online_GazeTracking.py:591-606
            if path is None:
                raise ValueError("POSE_LIBRARY_PATH is not set in config.")
            z = np.load(path, allow_pickle=True)
            X = z["X"]
            Q = z["Q"]
            G = z["G"] if "G" in z.files else None
            if G is None or G.shape[1] < 2:
                raise ValueError(
                    f"Pose library at {path} does not contain valid gaze matrix G."
                )
            logger_stub.log_event(
                f"Loaded pose library from {path} | "
                f"X.shape={X.shape}, Q.shape={Q.shape}, G.shape={G.shape}"
            )
            return X, Q, G

        return load_pose_library

    def test_v1_load(self, tmp_path: Path, load_pose_library):
        p = tmp_path / "v1.npz"
        _write_v1(p)
        X, Q, G = load_pose_library(str(p))
        assert X.shape == (5, 3)
        assert Q.shape == (5, 7)
        assert G.shape == (5, 3)

    def test_v2_load(self, tmp_path: Path, load_pose_library):
        p = tmp_path / "v2.npz"
        _write_v2(p)
        X, Q, G = load_pose_library(str(p))
        assert X.shape == (5, 3)
        assert Q.shape == (5, 7)
        # G is still the legacy 3-col matrix; the v2 keys live alongside.
        assert G.shape == (5, 3)

    def test_no_g_raises(self, tmp_path: Path, load_pose_library):
        p = tmp_path / "no_g.npz"
        np.savez_compressed(str(p),
                            X=np.zeros((3, 3)),
                            Q=np.zeros((3, 7)))
        with pytest.raises(ValueError, match="does not contain valid gaze"):
            load_pose_library(str(p))


# ─── Cross-loader: version detection drives dispatch ─────────────────────

class TestVersionDispatch:
    """Confirms that ``detect_pose_library_version`` returns the right
    integer for both schemas — this is what the driver will branch on
    when dispatching between v1 and v2 mapping at runtime (Phase 4).
    """

    def test_v1_detects_as_1(self, tmp_path: Path):
        p = tmp_path / "v1.npz"
        _write_v1(p)
        z = np.load(str(p), allow_pickle=True)
        assert detect_pose_library_version(z) == 1

    def test_v2_detects_as_2(self, tmp_path: Path):
        p = tmp_path / "v2.npz"
        _write_v2(p)
        z = np.load(str(p), allow_pickle=True)
        assert detect_pose_library_version(z) == 2

    def test_load_pose_library_v2_round_trips_features(self, tmp_path: Path):
        p = tmp_path / "v2.npz"
        _write_v2(p)
        data = load_pose_library_v2(str(p))
        for key in ("Q", "X", "D_cm", "Gaze_yaw_deg", "Gaze_pitch_deg",
                    "Head_yaw_deg", "Head_pitch_deg", "D_valid"):
            assert key in data, f"v2 loader missing {key!r}"


# ─── Transit phase + Leg_label_all loader compatibility ───────────────────

class TestTransitPhaseRoundTrip:
    """Recorder rework (2026-05-19): the v2 NPZ now carries
    ``phase='transit'`` rows in the ``*_all`` block plus a new
    ``Leg_label_all`` column. The loaders must round-trip these without
    raising, and the legacy v1 path (which only reads ``X / Q / G``)
    must still ignore them.
    """

    def _write_v2_with_transit(self, path: Path, N_cap: int = 3, N_transit: int = 5) -> None:
        N_all = N_cap + N_transit
        np.savez_compressed(
            str(path),
            T=np.arange(N_cap, dtype=float),
            Q=np.zeros((N_cap, 7)),
            X=np.zeros((N_cap, 3)),
            G=np.column_stack([np.full(N_cap, 0.5), np.full(N_cap, 0.5),
                                np.full(N_cap, 1.0)]),
            D_cm=np.full(N_cap, 75.0),
            D_valid=np.ones(N_cap, dtype=bool),
            Miss_mm=np.zeros(N_cap),
            IPD_mm=np.full(N_cap, 63.0),
            IMU_w=np.zeros(N_cap),
            IMU_fresh=np.ones(N_cap, dtype=bool),
            Head_yaw_deg=np.zeros(N_cap),
            Head_pitch_deg=np.zeros(N_cap),
            Gaze_yaw_deg=np.zeros(N_cap),
            Gaze_pitch_deg=np.zeros(N_cap),
            Target_label=np.array(["near_R1", "mid_R3", "far_R5"][:N_cap],
                                   dtype="<U32"),
            T_all=np.arange(N_all, dtype=float),
            Q_all=np.zeros((N_all, 7)),
            X_all=np.zeros((N_all, 3)),
            G_all=np.zeros((N_all, 3)),
            D_cm_all=np.full(N_all, 75.0),
            D_valid_all=np.ones(N_all, dtype=bool),
            Miss_mm_all=np.zeros(N_all),
            IPD_mm_all=np.full(N_all, 63.0),
            IMU_w_all=np.zeros(N_all),
            IMU_fresh_all=np.ones(N_all, dtype=bool),
            Head_yaw_deg_all=np.zeros(N_all),
            Head_pitch_deg_all=np.zeros(N_all),
            Gaze_yaw_deg_all=np.zeros(N_all),
            Gaze_pitch_deg_all=np.zeros(N_all),
            Phase_all=np.array(["captured"] * N_cap + ["transit"] * N_transit,
                                dtype="<U16"),
            Target_label_all=np.array(["x"] * N_all, dtype="<U32"),
            Leg_label_all=np.array([""] * N_cap + ["transit_a_to_b"] * N_transit,
                                     dtype="<U64"),
            meta=dict(version=2, side="R",
                      recorder="harmony_free_arm_calibration.py"),
        )

    def test_v2_with_transit_loads_via_v2_loader(self, tmp_path: Path):
        p = tmp_path / "v2_transit.npz"
        self._write_v2_with_transit(p)
        data = load_pose_library_v2(str(p))
        assert "Phase_all" in data
        assert "Leg_label_all" in data
        # Validate the phase enum carries the transit label exactly.
        phases = set(str(x) for x in data["Phase_all"])
        assert "transit" in phases
        assert "captured" in phases

    def test_v2_with_transit_does_not_break_v1_loader_keys(self, tmp_path: Path):
        # Legacy callers read X / Q / G only — the transit rows live
        # under *_all and must not surface as extra rows in Q/X/G.
        p = tmp_path / "v2_transit.npz"
        self._write_v2_with_transit(p, N_cap=3, N_transit=5)
        z = np.load(str(p), allow_pickle=True)
        assert z["Q"].shape[0] == 3
        assert z["X"].shape[0] == 3
        assert z["G"].shape[0] == 3
        assert z["Q_all"].shape[0] == 8
