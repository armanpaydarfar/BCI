"""
test_harmony_online_control_v2_depth.py — REPL VLM depth wiring (2026-05-19).

Locks down the depth_source-aware path added to ``harmony_online_control.main``
that mirrors the EEG-gated driver's commit d36711b. Four cases:

1. Vergence NPZ (or missing depth_source) → REPL uses snap["depth_cm"];
   VLMClient is never instantiated or called.
2. vlm_depth_pro NPZ + VLM healthy at startup → REPL passes the probe,
   vision-mode calls vlm_client.depth(at_gaze=True), and the resulting
   metres value is rescaled into D_cm (m * 100) for the v2 query.
3. vlm_depth_pro NPZ + VLM unreachable at startup → RuntimeError raised
   before the REPL loop opens; no q_target dispatched.
4. vlm_depth_pro NPZ + VLM healthy at startup but depth request fails
   mid-loop → REPL prints an error and continues without sending any
   robot command (no q_target dispatch).

These are all hardware-free: VLMClient, GazeStream, gaze_runner snapshot
RPC, and the UDP socket are all stubbed. We never go near a real port
or LSL stream.

Citations under test:
  - harmony_online_control.py main() — VLM probe at startup (post-v2_mapping)
  - harmony_online_control.py main() — vision-mode v2 dispatch depth branch
"""

from __future__ import annotations

import importlib
import socket as _socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest


# ─── shared socket-stub import helper (mirrors test_pose_library_loader_v1v2) ──

class _StubSocket:
    def __init__(self, *a, **kw): pass
    def setsockopt(self, *a, **kw): pass
    def bind(self, *a, **kw): pass
    def settimeout(self, *a, **kw): pass
    def sendto(self, *a, **kw): return 0
    def recvfrom(self, *a, **kw):
        raise _socket.timeout()
    def close(self): pass


def _import_module():
    """Import harmony_online_control with the module-level socket bind
    neutralized. Returns the (freshly-reloaded) module object so each
    test starts from a clean state — the module keeps singletons like
    a global sock that we don't want to share across tests."""
    original = _socket.socket
    _socket.socket = lambda *a, **kw: _StubSocket()
    try:
        if "harmony_online_control" in sys.modules:
            mod = importlib.reload(sys.modules["harmony_online_control"])
        else:
            mod = importlib.import_module("harmony_online_control")
    finally:
        _socket.socket = original
    return mod


# ─── NPZ fixtures ─────────────────────────────────────────────────────────

def _write_v2_npz(path: Path, depth_source: Optional[str], *, N: int = 8) -> None:
    """Write a v2 NPZ matching the harmony_free_arm_calibration schema,
    pinning meta['depth_source'] when given. Used by all four tests."""
    meta: Dict[str, Any] = dict(version=2, side="R")
    if depth_source is not None:
        meta["depth_source"] = depth_source
    np.savez_compressed(
        str(path),
        T=np.arange(N, dtype=float),
        Q=np.zeros((N, 7)),
        X=np.zeros((N, 3)),
        G=np.column_stack([np.full(N, 0.5), np.full(N, 0.5), np.full(N, 1.0)]),
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
        meta=meta,
    )


# ─── REPL harness ─────────────────────────────────────────────────────────

class _FakeGazeStream:
    """Stand-in for GazeStream that returns a fixed median gaze without
    touching LSL."""

    def __init__(self, *_args, **_kwargs):
        self.inlet = object()  # sentinel — flush_gaze_buffer must accept it

    def connect(self):
        return True

    def average_gaze_over_window(self, dur_s=1.0):
        return (0.5, 0.5)


class _FakeVLMClient:
    """Records every call so tests can assert on the call sequence."""

    def __init__(self, cfg) -> None:
        # Must expose the same .host/.port/.default_timeout_s attrs the
        # production code reads when building error messages.
        self.host = str(getattr(cfg, "VLM_SERVICE_HOST", "127.0.0.1"))
        self.port = int(getattr(cfg, "VLM_SERVICE_PORT", 5589))
        self.default_timeout_s = 1.0
        self.status_calls: int = 0
        self.depth_calls: List[Dict[str, Any]] = []
        self.status_response: Any = {"ok": True, "depth_enabled": True}
        self.depth_response: Any = {"ok": True, "depth_at_gaze_m": 0.75}
        self.status_raises: Optional[Exception] = None
        self.depth_raises: Optional[Exception] = None

    def status(self, *, timeout_s=None):
        self.status_calls += 1
        if self.status_raises is not None:
            raise self.status_raises
        return self.status_response

    def depth(self, *, at_gaze=True, save=False, timeout_s=15.0):
        self.depth_calls.append({"at_gaze": at_gaze, "save": save})
        if self.depth_raises is not None:
            raise self.depth_raises
        return self.depth_response


def _run_main_one_vision_then_quit(monkeypatch, mod, npz_path: Path,
                                    *, gaze_stream_cls=_FakeGazeStream,
                                    vlm_client_factory=None,
                                    snapshot: Optional[Dict[str, Any]] = None):
    """Run mod.main() through exactly one 'vision' iteration then 'quit'.

    Returns (sent_udp_messages, vlm_instances, raised).
      * sent_udp_messages: list of strings that udp_send would have
        transmitted. q_target dispatch shows up as a comma-separated
        ',;dur=' coordinate string.
      * vlm_instances: list of every VLMClient created during the run.
        Empty for vergence-NPZ paths.
      * raised: the exception object if main() raised, else None.
    """
    # CLI argv[1] = NPZ path. main() does sys.argv[1].
    monkeypatch.setattr(sys, "argv", ["harmony_online_control.py", str(npz_path)])

    # Inputs to ask_goal_and_duration: first call yields ("vision",
    # None, dur), second call yields ("quit", ...). The production
    # function builds these from input(); easier to bypass entirely.
    inputs = iter([("vision", None, 0.5), ("quit", None, 0.0)])
    monkeypatch.setattr(mod, "ask_goal_and_duration", lambda: next(inputs))

    # Replace GazeStream so we never go near LSL.
    monkeypatch.setattr(mod, "GazeStream", gaze_stream_cls)
    monkeypatch.setattr(mod, "flush_gaze_buffer", lambda inlet: None)
    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_kw: None)

    # Suppress telemetry capture — it polls a UDP socket we stubbed out.
    monkeypatch.setattr(mod, "collect_telemetry",
                         lambda dur: (np.array([]), np.zeros((0, 3)),
                                       np.zeros((0, 7)), np.zeros((0, 7))))

    # Capture every udp_send so we can assert no robot command was emitted
    # in the failure paths.
    sent: List[str] = []
    monkeypatch.setattr(mod, "udp_send", lambda msg: sent.append(msg))
    # send_and_wait_ack defaults to ACK-success so the REPL keeps moving.
    monkeypatch.setattr(mod, "send_and_wait_ack",
                         lambda msg, expect=None, timeout=None: (sent.append(msg) or True))

    # Stub _fetch_v2_snapshot — the production helper opens a UDP socket
    # to gaze_runner. We return a finite snapshot by default so the
    # vision path actually exercises the depth branch under test.
    if snapshot is None:
        snapshot = {
            "ok": True,
            "gaze_yaw_deg": 0.0,
            "gaze_pitch_deg": 0.0,
            "depth_cm": 75.0,
            "head_yaw_deg": 0.0,
            "head_pitch_deg": 0.0,
        }
    monkeypatch.setattr(mod, "_fetch_v2_snapshot", lambda: snapshot)

    # Inject VLMClient via the lazy-import sentinel. The production code
    # does ``from Utils.perception_clients import VLMClient`` at the two
    # use sites — patch the symbol on that module to redirect both.
    instances: List[Any] = []
    if vlm_client_factory is not None:
        import Utils.perception_clients as _pc

        def _factory(cfg):
            inst = vlm_client_factory(cfg)
            instances.append(inst)
            return inst

        monkeypatch.setattr(_pc, "VLMClient", _factory)

    raised: Optional[BaseException] = None
    try:
        mod.main()
    except BaseException as e:
        raised = e

    return sent, instances, raised


# ─── tests ────────────────────────────────────────────────────────────────

class TestVergenceNPZPath:
    """depth_source='vergence' (or missing) → snap['depth_cm'] is used and
    VLMClient is never touched."""

    def test_vergence_npz_uses_snapshot_depth(self, tmp_path: Path,
                                                monkeypatch, mod=None):
        mod = _import_module()
        # Force v=2 so the v2_mapping branch is taken.
        monkeypatch.setattr(mod.config, "GAZE_CALIBRATION_VERSION", 2,
                             raising=False)
        monkeypatch.setattr(mod.config, "GAZE_CALIBRATION_USE_IMU", False,
                             raising=False)

        npz = tmp_path / "v2_vergence.npz"
        _write_v2_npz(npz, depth_source="vergence")

        # Capture features passed to v2_mapping.query so we can assert
        # D_cm came from snap, not VLM.
        captured: Dict[str, Any] = {}
        from Utils.gaze.calibration_mapping import GazeCalibrationMappingV2
        real_query = GazeCalibrationMappingV2.query

        def spy_query(self, feats):
            captured["features"] = dict(feats)
            return real_query(self, feats)

        monkeypatch.setattr(GazeCalibrationMappingV2, "query", spy_query)

        # Run without any VLM injection: a passing test here proves the
        # vergence path never constructs the client.
        sent, instances, raised = _run_main_one_vision_then_quit(
            monkeypatch, mod, npz)

        assert raised is None, f"main() raised unexpectedly: {raised!r}"
        assert instances == [], "Vergence NPZ must not instantiate VLMClient"
        assert captured["features"]["D_cm"] == 75.0  # straight from snap

    def test_missing_depth_source_defaults_to_vergence(self, tmp_path: Path,
                                                        monkeypatch):
        mod = _import_module()
        monkeypatch.setattr(mod.config, "GAZE_CALIBRATION_VERSION", 2,
                             raising=False)
        monkeypatch.setattr(mod.config, "GAZE_CALIBRATION_USE_IMU", False,
                             raising=False)

        npz = tmp_path / "v2_legacy_no_meta.npz"
        _write_v2_npz(npz, depth_source=None)  # legacy NPZ with no depth_source

        sent, instances, raised = _run_main_one_vision_then_quit(
            monkeypatch, mod, npz)

        assert raised is None, f"main() raised unexpectedly: {raised!r}"
        # Default to 'vergence' → VLMClient is never built.
        assert instances == []


class TestVlmDepthProHealthyPath:
    """depth_source='vlm_depth_pro' + VLM healthy → status() succeeds at
    startup, depth(at_gaze=True) is called per-target, D_cm = m * 100."""

    def test_repl_passes_probe_and_uses_vlm_depth(self, tmp_path: Path,
                                                    monkeypatch):
        mod = _import_module()
        monkeypatch.setattr(mod.config, "GAZE_CALIBRATION_VERSION", 2,
                             raising=False)
        monkeypatch.setattr(mod.config, "GAZE_CALIBRATION_USE_IMU", False,
                             raising=False)
        monkeypatch.setattr(mod.config, "VLM_SERVICE_HOST", "192.168.99.99",
                             raising=False)
        monkeypatch.setattr(mod.config, "VLM_SERVICE_PORT", 5589,
                             raising=False)

        npz = tmp_path / "v2_vlm.npz"
        _write_v2_npz(npz, depth_source="vlm_depth_pro")

        # Spy on the v2 query so we can read D_cm fed into the metric.
        captured: Dict[str, Any] = {}
        from Utils.gaze.calibration_mapping import GazeCalibrationMappingV2
        real_query = GazeCalibrationMappingV2.query

        def spy_query(self, feats):
            captured["features"] = dict(feats)
            return real_query(self, feats)

        monkeypatch.setattr(GazeCalibrationMappingV2, "query", spy_query)

        # VLM reports depth_at_gaze_m=0.75 → D_cm should be 75.0.
        sent, instances, raised = _run_main_one_vision_then_quit(
            monkeypatch, mod, npz, vlm_client_factory=_FakeVLMClient)

        assert raised is None, f"main() raised unexpectedly: {raised!r}"
        assert len(instances) == 1, ("Exactly one VLMClient should be "
                                       "constructed at startup")
        client = instances[0]
        assert client.status_calls == 1, "Startup probe must call status()"
        assert len(client.depth_calls) == 1, (
            "Per-target vision dispatch must call depth(at_gaze=True)")
        assert client.depth_calls[0]["at_gaze"] is True

        assert captured["features"]["D_cm"] == pytest.approx(75.0)
        # And a robot command was actually emitted.
        joint_cmds = [m for m in sent if ";dur=" in m]
        assert len(joint_cmds) == 1, (
            f"Expected exactly one joint command; got {joint_cmds!r}")


class TestVlmDepthProStartupUnreachable:
    """Probe failure → RuntimeError before the REPL loop opens."""

    def test_probe_failure_raises_runtime_error(self, tmp_path: Path,
                                                  monkeypatch):
        mod = _import_module()
        monkeypatch.setattr(mod.config, "GAZE_CALIBRATION_VERSION", 2,
                             raising=False)
        monkeypatch.setattr(mod.config, "GAZE_CALIBRATION_USE_IMU", False,
                             raising=False)
        monkeypatch.setattr(mod.config, "VLM_SERVICE_HOST", "192.168.99.99",
                             raising=False)
        monkeypatch.setattr(mod.config, "VLM_SERVICE_PORT", 5589,
                             raising=False)

        npz = tmp_path / "v2_vlm_dead.npz"
        _write_v2_npz(npz, depth_source="vlm_depth_pro")

        def _factory(cfg):
            client = _FakeVLMClient(cfg)
            client.status_raises = OSError("connection refused")
            return client

        sent, instances, raised = _run_main_one_vision_then_quit(
            monkeypatch, mod, npz, vlm_client_factory=_factory)

        assert isinstance(raised, RuntimeError), (
            f"Expected RuntimeError on unreachable VLM, got {raised!r}")
        assert "vlm_service unreachable" in str(raised)
        # And nothing was emitted to the robot.
        joint_cmds = [m for m in sent if ";dur=" in m]
        assert joint_cmds == [], (
            f"No joint command should fire when probe fails; got {joint_cmds!r}")

    def test_status_not_ok_raises_runtime_error(self, tmp_path: Path,
                                                  monkeypatch):
        mod = _import_module()
        monkeypatch.setattr(mod.config, "GAZE_CALIBRATION_VERSION", 2,
                             raising=False)
        monkeypatch.setattr(mod.config, "GAZE_CALIBRATION_USE_IMU", False,
                             raising=False)
        monkeypatch.setattr(mod.config, "VLM_SERVICE_HOST", "192.168.99.99",
                             raising=False)
        monkeypatch.setattr(mod.config, "VLM_SERVICE_PORT", 5589,
                             raising=False)

        npz = tmp_path / "v2_vlm_depth_off.npz"
        _write_v2_npz(npz, depth_source="vlm_depth_pro")

        def _factory(cfg):
            client = _FakeVLMClient(cfg)
            # Probe survives the request but depth_enabled=False — driver
            # must refuse to load.
            client.status_response = {"ok": True, "depth_enabled": False}
            return client

        sent, instances, raised = _run_main_one_vision_then_quit(
            monkeypatch, mod, npz, vlm_client_factory=_factory)

        assert isinstance(raised, RuntimeError)
        assert "depth_enabled=False" in str(raised)


class TestVlmDepthProMidLoopFailure:
    """VLM healthy at startup but per-target depth() fails → REPL prints
    an error and does NOT dispatch a robot command."""

    def test_mid_loop_failure_skips_dispatch(self, tmp_path: Path,
                                               monkeypatch, capsys):
        mod = _import_module()
        monkeypatch.setattr(mod.config, "GAZE_CALIBRATION_VERSION", 2,
                             raising=False)
        monkeypatch.setattr(mod.config, "GAZE_CALIBRATION_USE_IMU", False,
                             raising=False)
        monkeypatch.setattr(mod.config, "VLM_SERVICE_HOST", "192.168.99.99",
                             raising=False)
        monkeypatch.setattr(mod.config, "VLM_SERVICE_PORT", 5589,
                             raising=False)

        npz = tmp_path / "v2_vlm_mid_fail.npz"
        _write_v2_npz(npz, depth_source="vlm_depth_pro")

        def _factory(cfg):
            client = _FakeVLMClient(cfg)
            # Healthy at startup, broken at depth() time.
            client.depth_raises = OSError("connection reset")
            return client

        sent, instances, raised = _run_main_one_vision_then_quit(
            monkeypatch, mod, npz, vlm_client_factory=_factory)

        assert raised is None, (
            f"Mid-loop VLM failure must NOT raise; got {raised!r}")
        assert len(instances) == 1
        # Probe succeeded once, depth was attempted once, then the loop
        # continued (no second depth call because the next ask_ returns
        # quit).
        assert instances[0].status_calls == 1
        assert len(instances[0].depth_calls) == 1

        # No joint dispatch — the only ";dur=" strings would be the
        # coords command + the 'g' ack. Assert nothing matches the
        # coordinate pattern.
        joint_cmds = [m for m in sent if ";dur=" in m]
        assert joint_cmds == [], (
            f"No joint command should fire after VLM failure; got "
            f"{joint_cmds!r}")

        # And the user did see an error explaining why.
        out = capsys.readouterr().out
        assert "VLM depth request failed" in out

    def test_mid_loop_non_finite_depth_skips_dispatch(self, tmp_path: Path,
                                                        monkeypatch, capsys):
        mod = _import_module()
        monkeypatch.setattr(mod.config, "GAZE_CALIBRATION_VERSION", 2,
                             raising=False)
        monkeypatch.setattr(mod.config, "GAZE_CALIBRATION_USE_IMU", False,
                             raising=False)
        monkeypatch.setattr(mod.config, "VLM_SERVICE_HOST", "192.168.99.99",
                             raising=False)
        monkeypatch.setattr(mod.config, "VLM_SERVICE_PORT", 5589,
                             raising=False)

        npz = tmp_path / "v2_vlm_nan.npz"
        _write_v2_npz(npz, depth_source="vlm_depth_pro")

        def _factory(cfg):
            client = _FakeVLMClient(cfg)
            client.depth_response = {"ok": True,
                                       "depth_at_gaze_m": float("nan")}
            return client

        sent, instances, raised = _run_main_one_vision_then_quit(
            monkeypatch, mod, npz, vlm_client_factory=_factory)

        assert raised is None
        joint_cmds = [m for m in sent if ";dur=" in m]
        assert joint_cmds == [], (
            "Non-finite VLM depth must not produce a robot command")
        out = capsys.readouterr().out
        assert "non-finite" in out


# ─── REV01 runtime_depth_pipeline dispatch (Plan §3.5, Step 6) ──────────────

def _write_rev01_hybrid_npz(path: Path, *, affine_map: Any, N: int = 8) -> None:
    """Write a REV01 hybrid NPZ with meta['depth_source']='hybrid_..._vergence'
    and meta['affine_map'] populated as given. Mirrors the recorder's
    Step 3 layout closely enough for the REPL's v2_mapping loader."""
    meta = dict(version=2, side="R",
                depth_source="hybrid_anchor_vlm_transit_vergence",
                affine_map=affine_map)
    np.savez_compressed(
        str(path),
        T=np.arange(N, dtype=float),
        Q=np.zeros((N, 7)),
        X=np.zeros((N, 3)),
        G=np.column_stack([np.full(N, 0.5), np.full(N, 0.5), np.full(N, 1.0)]),
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


class TestRuntimeDepthPipelineDispatch:
    """REV01 (Plan §3.5): the REPL dispatches on
    ``v2_mapping.runtime_depth_pipeline`` instead of branching on
    ``depth_source == "vlm_depth_pro"`` directly. Three cases:

    1. ``vergence_affine`` path applies ``a * d + b`` to the snapshot's
       depth_cm before the v2 query (no VLM call).
    2. ``vlm_depth_pro`` path unchanged from REV00 (regression test).
    3. ``hybrid_..._vergence`` NPZ with ``affine_map=None`` raises
       RuntimeError at startup with the NPZ path in the message
       (fail-fast on the missing-fit misconfiguration).
    """

    def test_vergence_affine_applies_linear_map(self, tmp_path: Path,
                                                 monkeypatch):
        mod = _import_module()
        monkeypatch.setattr(mod.config, "GAZE_CALIBRATION_VERSION", 2,
                             raising=False)
        monkeypatch.setattr(mod.config, "GAZE_CALIBRATION_USE_IMU", False,
                             raising=False)

        npz = tmp_path / "rev01_fitted.npz"
        # Picked coefficients so the apply step is unambiguous: with
        # snapshot depth_cm=80, runtime should produce
        # 1.5 * 80 + (-10) = 110.0.
        _write_rev01_hybrid_npz(npz, affine_map={
            "a": 1.5, "b": -10.0, "R2": 0.92, "max_abs_residual_cm": 4.0,
        })

        captured: Dict[str, Any] = {}
        from Utils.gaze.calibration_mapping import GazeCalibrationMappingV2
        real_query = GazeCalibrationMappingV2.query

        def spy_query(self, feats):
            captured["features"] = dict(feats)
            return real_query(self, feats)

        monkeypatch.setattr(GazeCalibrationMappingV2, "query", spy_query)

        # Snapshot returns raw vergence depth_cm=80; runtime must
        # apply a*d+b before the query lands.
        snapshot = {
            "ok": True, "gaze_yaw_deg": 0.0, "gaze_pitch_deg": 0.0,
            "depth_cm": 80.0, "head_yaw_deg": 0.0, "head_pitch_deg": 0.0,
        }
        sent, instances, raised = _run_main_one_vision_then_quit(
            monkeypatch, mod, npz, snapshot=snapshot)

        assert raised is None, f"main() raised unexpectedly: {raised!r}"
        # No VLM construction in the vergence_affine path.
        assert instances == [], (
            "vergence_affine must not instantiate VLMClient at runtime")
        # The query received the affine-mapped depth, not the raw 80.0.
        assert captured["features"]["D_cm"] == pytest.approx(110.0)
        # And a robot command was emitted.
        joint_cmds = [m for m in sent if ";dur=" in m]
        assert len(joint_cmds) == 1

    def test_vlm_depth_pro_path_unchanged(self, tmp_path: Path, monkeypatch):
        # Regression: pre-REV00 vlm_depth_pro NPZ (no transit promotion,
        # no affine_map). The REPL must still construct VLMClient,
        # probe at startup, and call depth(at_gaze=True) per target.
        mod = _import_module()
        monkeypatch.setattr(mod.config, "GAZE_CALIBRATION_VERSION", 2,
                             raising=False)
        monkeypatch.setattr(mod.config, "GAZE_CALIBRATION_USE_IMU", False,
                             raising=False)
        monkeypatch.setattr(mod.config, "VLM_SERVICE_HOST", "192.168.99.99",
                             raising=False)
        monkeypatch.setattr(mod.config, "VLM_SERVICE_PORT", 5589,
                             raising=False)

        npz = tmp_path / "v2_vlm_only.npz"
        _write_v2_npz(npz, depth_source="vlm_depth_pro")

        captured: Dict[str, Any] = {}
        from Utils.gaze.calibration_mapping import GazeCalibrationMappingV2
        real_query = GazeCalibrationMappingV2.query

        def spy_query(self, feats):
            captured["features"] = dict(feats)
            return real_query(self, feats)

        monkeypatch.setattr(GazeCalibrationMappingV2, "query", spy_query)

        sent, instances, raised = _run_main_one_vision_then_quit(
            monkeypatch, mod, npz, vlm_client_factory=_FakeVLMClient)

        assert raised is None, f"main() raised unexpectedly: {raised!r}"
        assert len(instances) == 1
        client = instances[0]
        assert client.status_calls == 1
        assert len(client.depth_calls) == 1
        # VLM reports 0.75 m -> D_cm == 75.0.
        assert captured["features"]["D_cm"] == pytest.approx(75.0)

    def test_hybrid_without_affine_map_raises_at_startup(self, tmp_path: Path,
                                                          monkeypatch):
        # Plan §3.5 alignment invariant: a REV01 NPZ pinned to the
        # hybrid string but with affine_map=None must fail-fast at
        # startup. Otherwise the runtime would silently fall back to
        # raw vergence and the per-feature Mahalanobis scale would be
        # off by the missing affine factor.
        mod = _import_module()
        monkeypatch.setattr(mod.config, "GAZE_CALIBRATION_VERSION", 2,
                             raising=False)
        monkeypatch.setattr(mod.config, "GAZE_CALIBRATION_USE_IMU", False,
                             raising=False)

        npz = tmp_path / "rev01_no_fit.npz"
        _write_rev01_hybrid_npz(npz, affine_map=None)

        sent, instances, raised = _run_main_one_vision_then_quit(
            monkeypatch, mod, npz)

        assert isinstance(raised, RuntimeError), (
            f"REV01 hybrid + None affine_map must raise; got {raised!r}")
        # Message must reference the NPZ path and tools/fit_vergence_affine.py
        # so the operator can act on it.
        msg = str(raised)
        assert str(npz) in msg, (
            f"RuntimeError must name the NPZ path; got: {msg!r}")
        assert "fit_vergence_affine.py" in msg, (
            f"RuntimeError must point the operator at the fit script; got: {msg!r}")
        # And nothing got past the probe to the robot.
        joint_cmds = [m for m in sent if ";dur=" in m]
        assert joint_cmds == []
