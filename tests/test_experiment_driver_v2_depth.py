"""
test_experiment_driver_v2_depth.py — Driver-side fail-fast probe (REV01
critic finding #2).

Mirror of ``tests/test_harmony_online_control_v2_depth.py::
TestRuntimeDepthPipelineDispatch::test_hybrid_without_affine_map_raises_at_startup``
but exercises the driver's ``_load_v2_mapping_if_enabled`` at
``ExperimentDriver_Online_GazeTracking.py:743-750`` instead of the
REPL's equivalent at ``harmony_online_control.py:622-629``. The two
fail-fast probes carry the same alignment invariant (hybrid NPZ
without an affine_map is a misconfiguration the runtime must refuse)
and both call sites need direct test coverage so a future refactor
that removes either guard fails CI.

This test loads the LIVE driver module via ``importlib.import_module``
(after stubbing the heavy side effects at module-load time —
pygame display init, model pickle load, XDF lookup, UDP socket
bind) rather than reproducing the probe inline. The reproduced-
helper approach used by ``tests/test_gaze_calibration_v1v2_dispatch.py``
would let a regression slip past CI because it tests the test code,
not the driver.

Citation under test (verified 2026-05-20):
  - ExperimentDriver_Online_GazeTracking.py:743-750 — the hybrid-and-
    None-affine_map RuntimeError raise inside
    ``_load_v2_mapping_if_enabled``.
"""

from __future__ import annotations

import importlib
import pickle
import socket as _socket
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest


# ─── socket / module-load stubs ───────────────────────────────────────────

class _StubSocket:
    def __init__(self, *a, **kw): pass
    def setsockopt(self, *a, **kw): pass
    def bind(self, *a, **kw): pass
    def settimeout(self, *a, **kw): pass
    def sendto(self, *a, **kw): return 0
    def recvfrom(self, *a, **kw):
        raise _socket.timeout()
    def close(self): pass


def _prepare_subject_dir(root: Path, subject: str) -> None:
    """Create the minimum ``sub-<subject>/`` layout the driver expects
    at import time: a pickled model file and at least one .xdf in
    training_data/. Provides pickles for every DECODER_BACKEND variant
    so the fixture is agnostic to the global config default."""
    subj = root / f"sub-{subject}"
    (subj / "models").mkdir(parents=True, exist_ok=True)
    (subj / "training_data").mkdir(parents=True, exist_ok=True)
    # The driver does ``pickle.load(f)`` on this; any picklable object
    # will satisfy the load — the test never invokes the model.
    stub = {"stub": True}
    for model_name in (
        f"sub-{subject}_model.pkl",
        f"sub-{subject}_xgb_cov_features.pkl",
        f"sub-{subject}_xgb_cov_erd_features.pkl",
    ):
        with open(subj / "models" / model_name, "wb") as f:
            pickle.dump(stub, f)
    # The driver's xdf listing only needs at least one non-OBS .xdf.
    (subj / "training_data" / "dummy.xdf").write_bytes(b"")


def _import_driver(monkeypatch, data_dir: Path, subject: str = "TESTSUBJ"):
    """Import ExperimentDriver_Online_GazeTracking with the heavy
    module-level side effects neutralised. Returns the (freshly-
    reloaded) module object so each test starts from a clean state.

    Mirrors the ``_import_module`` shim in
    ``tests/test_harmony_online_control_v2_depth.py`` but adds the
    extra stubs the driver needs at import: a fake subject directory
    so the model pickle + XDF discovery succeed, and socket
    neutralisation so the three UDP sockets the driver opens at module
    load don't bind real ports.
    """
    _prepare_subject_dir(data_dir, subject)

    # Patch config BEFORE import. The driver reads DATA_DIR /
    # TRAINING_SUBJECT at module load to locate the model.
    import config as _config
    monkeypatch.setattr(_config, "DATA_DIR", str(data_dir), raising=False)
    monkeypatch.setattr(_config, "TRAINING_SUBJECT", subject, raising=False)
    # GAZE_CALIBRATION_VERSION drives the v2 branch under test.
    monkeypatch.setattr(_config, "GAZE_CALIBRATION_VERSION", 2,
                         raising=False)
    monkeypatch.setattr(_config, "GAZE_CALIBRATION_USE_IMU", False,
                         raising=False)

    # Neutralise socket creation so the three module-level UDP sockets
    # (marker / robot / FES) don't bind real ports.
    original_socket = _socket.socket
    _socket.socket = lambda *a, **kw: _StubSocket()
    try:
        # Force a fresh import so config patches above are read.
        if "ExperimentDriver_Online_GazeTracking" in sys.modules:
            mod = importlib.reload(
                sys.modules["ExperimentDriver_Online_GazeTracking"])
        else:
            mod = importlib.import_module(
                "ExperimentDriver_Online_GazeTracking")
    finally:
        _socket.socket = original_socket
    return mod


# ─── NPZ fixture ──────────────────────────────────────────────────────────

def _write_rev01_hybrid_npz(path: Path, *, affine_map: Any,
                             N: int = 8) -> None:
    """Write a REV01 hybrid NPZ with meta['depth_source']=
    'hybrid_anchor_vlm_transit_vergence' and meta['affine_map']
    populated as given. Mirrors the REPL test fixture so the
    invariant under test (the fail-fast probe) sees identically-shaped
    NPZs on both sides."""
    meta = dict(version=2, side="R",
                depth_source="hybrid_anchor_vlm_transit_vergence",
                affine_map=affine_map)
    np.savez_compressed(
        str(path),
        T=np.arange(N, dtype=float),
        Q=np.zeros((N, 7)),
        X=np.zeros((N, 3)),
        G=np.column_stack([np.full(N, 0.5), np.full(N, 0.5),
                            np.full(N, 1.0)]),
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
        Target_label=np.array(["mid_R3"] * N, dtype="<U32"),
        meta=meta,
    )


# ─── test ─────────────────────────────────────────────────────────────────

class TestDriverHybridFailFast:
    """REV01 Plan §3.5 alignment invariant on the driver side: a hybrid
    NPZ pinned to ``meta['depth_source']=
    'hybrid_anchor_vlm_transit_vergence'`` but with
    ``meta['affine_map']=None`` must fail-fast at startup. Otherwise
    the driver would silently fall back to raw vergence and the per-
    feature Mahalanobis scale would be off by the missing affine
    factor."""

    def test_hybrid_without_affine_map_raises_at_startup(
            self, tmp_path: Path, monkeypatch):
        data_dir = tmp_path / "data"
        mod = _import_driver(monkeypatch, data_dir=data_dir)

        npz = tmp_path / "rev01_no_fit.npz"
        _write_rev01_hybrid_npz(npz, affine_map=None)

        with pytest.raises(RuntimeError) as excinfo:
            mod._load_v2_mapping_if_enabled(str(npz))

        msg = str(excinfo.value)
        # Message must name the NPZ path and the fit-script so the
        # operator knows which file is misconfigured and what to run.
        assert str(npz) in msg, (
            f"RuntimeError must name the NPZ path; got: {msg!r}")
        assert "fit_vergence_affine.py" in msg, (
            f"RuntimeError must point the operator at the fit script; "
            f"got: {msg!r}")
