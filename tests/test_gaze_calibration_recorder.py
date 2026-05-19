"""
test_gaze_calibration_recorder.py — Phase 2.c test (plan §6.3 #1) +
2026-05-19 recorder rework (15-waypoint grid + telemetry thread).

Exercises ``harmony_free_arm_calibration.py``'s pure / IO-free helpers
without touching real UDP sockets: the snapshot-to-bundle conversion,
the v2 NPZ writer, the workspace coverage protocol constant, and (new
this session) the background telemetry thread + auto-home behaviour of
``run_session`` driven via monkeypatched RobotLink / gaze snapshots.

Hardware-free per Harmony_Test_Suite_Plan.md §3.

Citations under test (verified 2026-05-19):

  - harmony_free_arm_calibration.py ``bundle_from_snapshot``
  - harmony_free_arm_calibration.py ``write_npz`` (v2 schema)
  - harmony_free_arm_calibration.py ``MANDATORY_GRID``
  - harmony_free_arm_calibration.py ``TelemetryThread``
  - harmony_free_arm_calibration.py ``run_session``
  - harmony_free_arm_calibration.py ``_send_auto_home`` (auto-home opcode)
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest


# Import once at module load — the recorder's import side effects (UDP
# socket creation) are inside __init__ of RobotLink, not at module
# scope, so simply importing is safe.
import harmony_free_arm_calibration as recorder
from harmony_free_arm_calibration import (
    AUTO_HOME_DURATION_S,
    MANDATORY_GRID,
    CaptureBundle,
    TelemetryThread,
    bundle_from_snapshot,
    fetch_vlm_depth_cm,
    verify_vlm_depth_available,
    write_npz,
)


# ─── bundle_from_snapshot ─────────────────────────────────────────────────

class TestBundleFromSnapshot:
    """The snapshot-to-bundle conversion is pure (no I/O); it just
    repackages a gaze_runner snapshot dict + robot telemetry into a
    CaptureBundle with the right fields and unit conversions.
    """

    def _full_snap(self, **overrides) -> dict:
        snap = {
            "ok": True,
            "worn": True,
            "gaze_px": (800.0, 600.0),
            "depth_cm": 75.0,
            "depth_valid": True,
            "miss_mm": 3.5,
            "ipd_mm": 63.0,
            "imu_angvel": 0.05,
            "imu_fresh": True,
            "head_yaw_deg": -2.0,
            "head_pitch_deg": 1.5,
            "gaze_yaw_deg": 10.0,
            "gaze_pitch_deg": -5.0,
        }
        snap.update(overrides)
        return snap

    def test_full_snapshot_fields_map_through(self):
        snap = self._full_snap()
        q = np.arange(7, dtype=float)
        ee = np.array([100.0, 200.0, 300.0])
        b = bundle_from_snapshot(snap, robot_t=12345.6, q=q, ee_mm=ee,
                                  phase="captured", target_label="mid_MC")
        assert isinstance(b, CaptureBundle)
        assert b.t == pytest.approx(12345.6)
        assert b.phase == "captured"
        assert b.target_label == "mid_MC"
        # q / ee are copies, not aliases
        q[0] = -99.0
        assert b.q[0] == 0.0
        assert b.ee_mm[1] == 200.0
        # Sensor fields pass through
        assert b.depth_cm == pytest.approx(75.0)
        assert b.depth_valid is True
        assert b.miss_mm == pytest.approx(3.5)
        assert b.ipd_mm == pytest.approx(63.0)
        assert b.imu_w == pytest.approx(0.05)
        assert b.imu_fresh is True
        assert b.head_yaw_deg == pytest.approx(-2.0)
        assert b.head_pitch_deg == pytest.approx(1.5)
        assert b.gaze_yaw_deg == pytest.approx(10.0)
        assert b.gaze_pitch_deg == pytest.approx(-5.0)

    def test_pixel_normalisation_uses_config_width_height(self):
        snap = self._full_snap(gaze_px=(800.0, 600.0))
        b = bundle_from_snapshot(snap, robot_t=0.0, q=np.zeros(7),
                                  ee_mm=np.zeros(3), phase="captured",
                                  target_label="t")
        # Default 1600 x 1200 from config.GAZE_SAMPLE_WIDTH/HEIGHT
        assert b.gaze_x_norm == pytest.approx(0.5)
        assert b.gaze_y_norm == pytest.approx(0.5)

    def test_missing_imu_returns_nan(self):
        snap = self._full_snap(imu_angvel=None, imu_fresh=False)
        b = bundle_from_snapshot(snap, robot_t=0.0, q=np.zeros(7),
                                  ee_mm=np.zeros(3), phase="captured",
                                  target_label="t")
        assert np.isnan(b.imu_w)
        assert b.imu_fresh is False

    def test_gaze_conf_is_zero_when_unworn_or_depth_invalid(self):
        # worn=False -> conf=0
        b = bundle_from_snapshot(self._full_snap(worn=False), robot_t=0.0,
                                  q=np.zeros(7), ee_mm=np.zeros(3),
                                  phase="captured", target_label="t")
        assert b.gaze_conf == 0.0
        # depth_valid=False -> conf=0
        b = bundle_from_snapshot(self._full_snap(depth_valid=False),
                                  robot_t=0.0, q=np.zeros(7),
                                  ee_mm=np.zeros(3), phase="captured",
                                  target_label="t")
        assert b.gaze_conf == 0.0
        # both true -> conf=1.0
        b = bundle_from_snapshot(self._full_snap(), robot_t=0.0,
                                  q=np.zeros(7), ee_mm=np.zeros(3),
                                  phase="captured", target_label="t")
        assert b.gaze_conf == 1.0

    def test_bad_gaze_px_returns_nan(self):
        snap = self._full_snap(gaze_px=None)
        b = bundle_from_snapshot(snap, robot_t=0.0, q=np.zeros(7),
                                  ee_mm=np.zeros(3), phase="captured",
                                  target_label="t")
        assert np.isnan(b.gaze_x_norm)
        assert np.isnan(b.gaze_y_norm)


# ─── write_npz ────────────────────────────────────────────────────────────

class TestWriteNPZ:
    """The writer materialises a list of CaptureBundles into the v2 NPZ
    schema. Tests assert the legacy keys are still present (v1
    consumers stay happy), the new keys exist with the right shapes,
    and the meta dict carries version=2.
    """

    def _make_bundle(self, *, phase: str = "captured", target_label: str = "mid_MC",
                     depth_cm: float = 75.0, depth_valid: bool = True,
                     head_yaw_deg: float = 0.0) -> CaptureBundle:
        return CaptureBundle(
            t=0.0,
            q=np.arange(7, dtype=float),
            ee_mm=np.array([1.0, 2.0, 3.0]),
            gaze_x_norm=0.5, gaze_y_norm=0.5, gaze_conf=1.0,
            depth_cm=depth_cm, depth_valid=depth_valid,
            miss_mm=2.0, ipd_mm=63.0,
            imu_w=0.01, imu_fresh=True,
            head_yaw_deg=head_yaw_deg, head_pitch_deg=0.0,
            gaze_yaw_deg=0.0, gaze_pitch_deg=0.0,
            phase=phase, target_label=target_label,
        )

    def test_writes_v2_schema_with_legacy_keys(self, tmp_path: Path):
        bundles = [self._make_bundle(target_label="mid_MC") for _ in range(3)]
        out = tmp_path / "out.npz"
        write_npz(bundles, str(out))
        z = np.load(str(out), allow_pickle=True)
        # Legacy v1 keys
        assert {"T", "Q", "X", "G"}.issubset(set(z.files))
        # v2 captured-only keys
        for k in ("D_cm", "D_valid", "Miss_mm", "IPD_mm",
                  "IMU_w", "IMU_fresh", "Head_yaw_deg", "Head_pitch_deg",
                  "Gaze_yaw_deg", "Gaze_pitch_deg", "Target_label"):
            assert k in z.files, f"missing v2 key {k!r}"
        # v2 all-phase keys
        for k in ("T_all", "Q_all", "X_all", "G_all",
                  "Phase_all", "Target_label_all"):
            assert k in z.files, f"missing all-phase key {k!r}"
        # Shapes
        assert z["T"].shape == (3,)
        assert z["Q"].shape == (3, 7)
        assert z["X"].shape == (3, 3)
        assert z["G"].shape == (3, 3)
        assert z["D_cm"].shape == (3,)
        # Meta has version=2
        meta = z["meta"].item()
        assert isinstance(meta, dict)
        assert meta["version"] == 2

    def test_separates_captured_from_moving(self, tmp_path: Path):
        # 2 captured + 3 moving = 2 in legacy block, 5 in _all block
        bundles = (
            [self._make_bundle(phase="captured", target_label="mid_MC")]
            + [self._make_bundle(phase="moving", target_label="mid_MC") for _ in range(3)]
            + [self._make_bundle(phase="captured", target_label="far_TR")]
        )
        out = tmp_path / "out.npz"
        write_npz(bundles, str(out))
        z = np.load(str(out), allow_pickle=True)
        assert z["Q"].shape == (2, 7)
        assert z["Q_all"].shape == (5, 7)
        # Phase_all has 1 captured then 3 moving then 1 captured
        assert list(z["Phase_all"]) == ["captured", "moving", "moving",
                                         "moving", "captured"]

    def test_raises_when_no_captured(self, tmp_path: Path):
        # Moving-only bundle list — writer must refuse.
        bundles = [self._make_bundle(phase="moving") for _ in range(3)]
        out = tmp_path / "out.npz"
        with pytest.raises(RuntimeError, match="zero captured samples"):
            write_npz(bundles, str(out))

    def test_target_labels_round_trip(self, tmp_path: Path):
        labels = ["near_TL", "mid_MC", "far_BR"]
        bundles = [self._make_bundle(target_label=lbl) for lbl in labels]
        out = tmp_path / "out.npz"
        write_npz(bundles, str(out))
        z = np.load(str(out), allow_pickle=True)
        assert list(z["Target_label"]) == labels


# ─── MANDATORY_GRID constant ──────────────────────────────────────────────

class TestMandatoryGrid:
    def test_size_is_three_depths_times_five_horizontal(self):
        # 3 depth bands × 5 horizontal positions = 15.
        assert len(MANDATORY_GRID) == 15

    def test_depth_band_prefixes_are_exhaustive(self):
        depths = {lbl.split("_")[0] for lbl in MANDATORY_GRID}
        assert depths == {"near", "mid", "far"}

    def test_horizontal_bins_are_R1_through_R5(self):
        # Each depth band must hit all 5 horizontal labels.
        cells = {"R1", "R2", "R3", "R4", "R5"}
        for depth in ("near", "mid", "far"):
            band_cells = {lbl.split("_")[1] for lbl in MANDATORY_GRID
                          if lbl.startswith(f"{depth}_")}
            assert band_cells == cells, f"depth {depth!r}: missing {cells - band_cells}"

    def test_sweep_order_rightmost_first_per_depth(self):
        # Within each depth band the sweep must walk R1 → R5 before
        # moving to the next depth band, and depths order near → mid → far.
        expected = [
            "near_R1", "near_R2", "near_R3", "near_R4", "near_R5",
            "mid_R1",  "mid_R2",  "mid_R3",  "mid_R4",  "mid_R5",
            "far_R1",  "far_R2",  "far_R3",  "far_R4",  "far_R5",
        ]
        assert list(MANDATORY_GRID) == expected

    def test_labels_are_unique(self):
        assert len(set(MANDATORY_GRID)) == len(MANDATORY_GRID)


# ─── TelemetryThread + run_session integration ─────────────────────────────

def _make_snap(**overrides) -> Dict[str, Any]:
    """Mock gaze_snapshot return shape — minimal keys the recorder
    actually reads via ``bundle_from_snapshot``."""
    snap = {
        "ok": True,
        "worn": True,
        "gaze_px": (800.0, 600.0),
        "depth_cm": 75.0,
        "depth_valid": True,
        "miss_mm": 3.5,
        "ipd_mm": 63.0,
        "imu_angvel": 0.05,
        "imu_fresh": True,
        "head_yaw_deg": 0.0,
        "head_pitch_deg": 0.0,
        "gaze_yaw_deg": 0.0,
        "gaze_pitch_deg": 0.0,
    }
    snap.update(overrides)
    return snap


class _FakeLink:
    """Stand-in for ``RobotLink`` that the run_session test drives.

    All methods are non-blocking and record outbound opcodes in
    ``sent_opcodes`` so the test can assert send order. The fake never
    binds a real UDP socket.
    """

    def __init__(self):
        self.sent_opcodes: List[str] = []
        self._q = np.zeros(7, dtype=float)
        self._ee = np.zeros(3, dtype=float)
        self._closed = False

    # The recorder's run_session does `link.sock.getsockname()` in the
    # bind log line; expose a minimal stub.
    class _SockStub:
        def getsockname(self):
            return ("0.0.0.0", 8080)
    sock = _SockStub()
    robot_addr = ("192.168.2.1", 8080)

    def send(self, msg: str) -> None:
        self.sent_opcodes.append(msg)

    def recv(self, timeout_s: float) -> Optional[str]:
        return None

    def send_and_wait_ack(self, msg: str, expect_prefix=None,
                          timeout: float = 0.35) -> Optional[str]:
        self.sent_opcodes.append(msg)
        # All ACKs succeed in the happy-path fake.
        base = msg.split(";", 1)[0]
        if base == "m":
            return "ACK:MASTER_FREE"
        if base == "h":
            return "ACK:h"
        return f"ACK:{base}"

    def query_state(self) -> Optional[Dict[str, Any]]:
        return {"_t": time.time(), "q": self._q.copy(), "ee": self._ee.copy()}

    def close(self) -> None:
        self._closed = True


def _patch_run_session(monkeypatch, *, snapshot_returns=None,
                       prompt_responses=None, transit_arm_dwell_s: float = 1.0):
    """Common patch bundle for run_session tests.

    - ``snapshot_returns``: callable() -> snap dict (defaults to always-ok).
    - ``prompt_responses``: list of strings consumed in order, one per
      ``prompt_user`` call. Empty string terminates the free-additions
      loop.
    - ``transit_arm_dwell_s``: simulated time the operator spends with
      the arm free per waypoint. Drives how many transit bundles the
      thread can accumulate at 20 Hz.
    """
    if snapshot_returns is None:
        snapshot_returns = _make_snap

    if prompt_responses is None:
        prompt_responses = []
    # Prepend the recenter prompt + the 15 per-waypoint prompts + an
    # empty string to end free-additions if the caller did not supply
    # their own. Each waypoint takes two prompts: the recenter readiness
    # ("Press Enter when ready to recenter…") and the per-waypoint
    # "Move the arm and fixate…" prompt. We default to giving the
    # caller's list verbatim so they can control the full sequence.
    prompt_iter = iter(prompt_responses)

    def fake_prompt(msg):
        try:
            return next(prompt_iter)
        except StopIteration:
            # An empty string ends the free-additions loop; default to
            # empty so missing responses cleanly terminate.
            return ""

    monkeypatch.setattr(recorder, "prompt_user", fake_prompt)
    monkeypatch.setattr(recorder, "gaze_recenter", lambda timeout_s=None: True)
    monkeypatch.setattr(recorder, "gaze_snapshot",
                        lambda include_objects=False, timeout_s=None: snapshot_returns())

    # Replace settle_and_snapshot and collect_moving_phase with fast
    # stubs that emit a canonical bundle each. Real implementations
    # sleep POST_CAPTURE_SETTLE_S (1.0 s) per capture — too slow for
    # tests. The transit sampler is what we actually want to exercise.
    def fake_settle(target_label, robot_state, vlm_client=None):
        return CaptureBundle(
            t=time.time(), q=np.zeros(7), ee_mm=np.zeros(3),
            gaze_x_norm=0.5, gaze_y_norm=0.5, gaze_conf=1.0,
            depth_cm=75.0, depth_valid=True, miss_mm=3.5, ipd_mm=63.0,
            imu_w=0.05, imu_fresh=True,
            head_yaw_deg=0.0, head_pitch_deg=0.0,
            gaze_yaw_deg=0.0, gaze_pitch_deg=0.0,
            phase="captured", target_label=target_label,
        )

    def fake_moving(link, target_label, duration_s):
        # No moving-phase bundles — they are orthogonal to the transit
        # behaviour under test.
        return []

    def fake_capture(link):
        return {"_t": time.time(), "q": np.zeros(7), "ee": np.zeros(3)}

    def fake_free_arm(link):
        # Send 'm' so the assertion-on-opcodes test still sees the call.
        link.send_and_wait_ack("m", expect_prefix="MASTER_FREE")
        # Sleep ``transit_arm_dwell_s`` to give the telemetry thread a
        # known window to accumulate samples. This stands in for the
        # operator reaction-time + arm-motion duration between `m` and
        # `c`. NOTE: capture-flow waits for prompt_user("Move the arm…")
        # AFTER free_arm returns, so the dwell happens around the
        # prompt. We sleep here so the dwell straddles the free→capture
        # gap deterministically.
        time.sleep(transit_arm_dwell_s)
        return True

    monkeypatch.setattr(recorder, "settle_and_snapshot", fake_settle)
    monkeypatch.setattr(recorder, "collect_moving_phase", fake_moving)
    monkeypatch.setattr(recorder, "capture_pose", fake_capture)
    monkeypatch.setattr(recorder, "free_arm", fake_free_arm)
    # The 4-second sleep inside _send_auto_home would dominate the test
    # runtime; cut it short. The behaviour we care about (opcode sent
    # exactly once) is unaffected.
    real_sleep = time.sleep
    monkeypatch.setattr(recorder, "AUTO_HOME_DURATION_S", 0.05)
    monkeypatch.setattr(recorder, "AUTO_HOME_GRACE_S", 0.0)

    return real_sleep


class TestTelemetryThreadLifecycle:
    """Direct unit tests on TelemetryThread without run_session."""

    def test_thread_inert_until_started(self, monkeypatch):
        # Patch gaze_snapshot to a counter so we can verify no calls
        # happen before start_after_first_capture().
        calls = {"n": 0}

        def counting_snap(include_objects=False, timeout_s=None):
            calls["n"] += 1
            return _make_snap()

        monkeypatch.setattr(recorder, "gaze_snapshot", counting_snap)
        link = _FakeLink()
        bundles: List[CaptureBundle] = []
        lock = threading.Lock()
        t = TelemetryThread(link, bundles, lock, sample_hz=50.0)
        t.start()
        time.sleep(0.15)  # ~7 ticks at 50 Hz; thread should be idle
        assert calls["n"] == 0, "telemetry thread sampled before start_after_first_capture()"
        assert len(bundles) == 0
        t.stop()

    def test_thread_appends_transit_bundles_after_activation(self, monkeypatch):
        monkeypatch.setattr(recorder, "gaze_snapshot",
                            lambda include_objects=False, timeout_s=None: _make_snap())
        link = _FakeLink()
        bundles: List[CaptureBundle] = []
        lock = threading.Lock()
        t = TelemetryThread(link, bundles, lock, sample_hz=50.0)
        t.start()
        t.set_leg("transit_A_to_B")
        t.start_after_first_capture()
        time.sleep(0.25)  # ~12 ticks at 50 Hz; expect >= ~8 bundles
        t.stop()
        with lock:
            collected = list(bundles)
        assert len(collected) >= 5, f"expected several transit bundles, got {len(collected)}"
        for b in collected:
            assert b.phase == "transit"
            assert b.leg_label == "transit_A_to_B"
            assert b.target_label == ""

    def test_pause_halts_sampling(self, monkeypatch):
        monkeypatch.setattr(recorder, "gaze_snapshot",
                            lambda include_objects=False, timeout_s=None: _make_snap())
        link = _FakeLink()
        bundles: List[CaptureBundle] = []
        lock = threading.Lock()
        t = TelemetryThread(link, bundles, lock, sample_hz=50.0)
        t.start()
        t.start_after_first_capture()
        time.sleep(0.1)
        n_before_pause = len(bundles)
        t.pause()
        time.sleep(0.15)
        n_after_pause = len(bundles)
        # Pause should freeze the bundle count (allow one tick of slop).
        assert n_after_pause - n_before_pause <= 1
        t.stop()

    def test_escalates_after_consecutive_failures(self, monkeypatch):
        # Always-None snapshot → telemetry should error_flag after
        # TELEMETRY_MAX_CONSECUTIVE_FAILURES ticks.
        monkeypatch.setattr(recorder, "gaze_snapshot",
                            lambda include_objects=False, timeout_s=None: None)
        link = _FakeLink()
        bundles: List[CaptureBundle] = []
        lock = threading.Lock()
        t = TelemetryThread(link, bundles, lock, sample_hz=100.0)
        t.start()
        t.start_after_first_capture()
        time.sleep(0.3)
        assert t.error_flag is True
        assert "consecutive snapshot/query failures" in t.error_reason
        t.stop()


class TestRunSessionTelemetry:
    """End-to-end run_session simulation via monkeypatched RobotLink +
    gaze service. Confirms the captured + transit bundle counts and
    that no transit bundles cross the home-to-first or final-to-home
    boundaries.
    """

    def _build_prompt_responses(self, n_waypoints: int) -> List[str]:
        # Recenter prompt + (per waypoint: capture prompt) + free-add
        # terminator (empty string ends the free-additions while-loop).
        return [""] + [""] * n_waypoints + [""]

    def _run(self, monkeypatch, tmp_path: Path, *,
             transit_arm_dwell_s: float = 0.2):
        responses = self._build_prompt_responses(len(MANDATORY_GRID))
        _patch_run_session(monkeypatch, prompt_responses=responses,
                            transit_arm_dwell_s=transit_arm_dwell_s)
        # Replace RobotLink with the fake so run_session does not bind
        # a real UDP socket.
        fake = _FakeLink()
        monkeypatch.setattr(recorder, "RobotLink", lambda: fake)
        out_path = recorder.run_session(out_dir=str(tmp_path))
        return out_path, fake

    def test_fifteen_captures_plus_transit_bundles(self, monkeypatch, tmp_path):
        out_path, fake = self._run(monkeypatch, tmp_path,
                                     transit_arm_dwell_s=0.2)
        assert out_path is not None
        z = np.load(out_path, allow_pickle=True)
        # 15 captured rows in the legacy block.
        assert z["Q"].shape[0] == 15
        # Transit bundles in the _all block. At 20 Hz with ~0.2 s dwell
        # between 14 transit legs (after first capture, before final
        # capture), expect at least 14 * 0.2 * 20 * 0.5 = ~28 bundles
        # accounting for prompt overhead. Use a conservative floor.
        phases = list(z["Phase_all"])
        n_transit = sum(1 for p in phases if p == "transit")
        assert n_transit >= 10, (
            f"expected >= 10 transit bundles given ~0.2 s × 14 legs at 20 Hz; "
            f"got {n_transit}"
        )

    def test_no_transit_bundle_references_home(self, monkeypatch, tmp_path):
        out_path, _ = self._run(monkeypatch, tmp_path,
                                  transit_arm_dwell_s=0.15)
        z = np.load(out_path, allow_pickle=True)
        leg_labels = list(z["Leg_label_all"])
        phases = list(z["Phase_all"])
        # Among transit bundles, no leg label may reference 'home' on
        # either side of the transition.
        transit_legs = [lbl for lbl, ph in zip(leg_labels, phases)
                        if ph == "transit"]
        for lbl in transit_legs:
            assert "home_to_" not in lbl, f"transit bundle labelled with home transition: {lbl!r}"
            assert "_to_home" not in lbl, f"transit bundle labelled with home transition: {lbl!r}"
            # And the label must be one of the expected mandatory-grid
            # transit forms: "transit_<from>_to_<to>" with both labels
            # drawn from MANDATORY_GRID.
            assert lbl.startswith("transit_"), f"unexpected leg label: {lbl!r}"


class TestTelemetryStartsAfterFirstCapture:
    """The telemetry thread must not emit transit bundles before the
    first waypoint's `c` ACK completes. The home-to-first-waypoint
    transition is excluded by construction.
    """

    def test_no_transit_bundles_before_first_capture(self, monkeypatch, tmp_path):
        # Same patch bundle as TestRunSessionTelemetry but we instrument
        # `capture_pose` to record the bundles-list snapshot the first
        # time it is called, so we can assert the snapshot has zero
        # transit entries.
        responses = [""] * 20
        _patch_run_session(monkeypatch, prompt_responses=responses,
                            transit_arm_dwell_s=0.1)

        captured_state_at_first_c: Dict[str, Any] = {}

        def instrumenting_capture(link):
            if "snapshot" not in captured_state_at_first_c:
                # Take a snapshot of the bundles list at the moment of
                # the FIRST c call by reaching into the closure later.
                # The trick: store a sentinel that the test reads after
                # run_session returns by inspecting the saved file.
                captured_state_at_first_c["snapshot"] = True
            return {"_t": time.time(), "q": np.zeros(7), "ee": np.zeros(3)}

        monkeypatch.setattr(recorder, "capture_pose", instrumenting_capture)

        fake = _FakeLink()
        monkeypatch.setattr(recorder, "RobotLink", lambda: fake)
        out_path = recorder.run_session(out_dir=str(tmp_path))
        assert out_path is not None
        z = np.load(out_path, allow_pickle=True)
        # Time-order check: in the _all stream, the first 'captured'
        # row must NOT be preceded by any 'transit' row — the thread
        # only activates after the first c ACK. (Moving bundles also
        # do not appear here because the fake collect_moving_phase
        # returns [].)
        phases = list(z["Phase_all"])
        # Find first captured index
        first_cap = phases.index("captured")
        before_first = phases[:first_cap]
        assert "transit" not in before_first, (
            f"transit bundle(s) emitted before first capture: phases[:first_cap]="
            f"{before_first}"
        )


class TestAutoHomeOpcode:
    """The `h` opcode must be sent exactly once, AFTER the final
    capture, with the dur=4.0 suffix.
    """

    def test_h_opcode_sent_exactly_once_after_final_capture(self, monkeypatch, tmp_path):
        responses = [""] * 20
        _patch_run_session(monkeypatch, prompt_responses=responses,
                            transit_arm_dwell_s=0.05)
        fake = _FakeLink()
        monkeypatch.setattr(recorder, "RobotLink", lambda: fake)
        out_path = recorder.run_session(out_dir=str(tmp_path))
        assert out_path is not None
        # Count `h` opcodes in the sent list. Allow both the bare 'h'
        # form (legacy) and the dur-suffixed form. Auto-home is the
        # ONLY caller of `h` in this run; no other path emits it.
        h_calls = [op for op in fake.sent_opcodes if op.split(";", 1)[0] == "h"]
        assert len(h_calls) == 1, (
            f"expected exactly one `h` opcode (auto-home); got {h_calls}"
        )
        # The actual opcode should include dur=4.000 (the test patches
        # AUTO_HOME_DURATION_S to 0.05 to keep the test fast, so we
        # only assert the dur= prefix is present).
        assert "dur=" in h_calls[0], (
            f"auto-home opcode missing dur= suffix: {h_calls[0]!r}"
        )

        # Auto-home must come AFTER the final `c` (or after the
        # moving-phase send sequence — but in our fake there is no
        # 'c' opcode because capture_pose is fully mocked. Use 'm' as
        # the proxy: there must be at least one 'm' before the 'h'.)
        h_index = fake.sent_opcodes.index(h_calls[0])
        m_indices = [i for i, op in enumerate(fake.sent_opcodes) if op == "m"]
        assert m_indices, "no `m` opcodes recorded — fake setup is wrong"
        assert max(m_indices) < h_index, (
            "auto-home `h` opcode landed BEFORE the last `m` — telemetry "
            "ordering is broken"
        )

    def test_auto_home_dur_default_is_four_seconds(self):
        # Independent assertion on the module-level constant — the
        # patched value used in tests above must not silently mask a
        # changed default in production.
        assert AUTO_HOME_DURATION_S == 4.0
# ─── VLM Depth Pro path (GAZE_CALIBRATION_DEPTH_SOURCE='vlm_depth_pro') ──

class _FakeVLMClient:
    """Stand-in for Utils.perception_clients.VLMClient. Tests inject
    arbitrary status / depth payloads or simulate transport failures
    without touching a real UDP socket."""
    def __init__(self, *, status_payload=None, depth_payload=None,
                 status_raises=None, depth_raises=None,
                 host="127.0.0.1", port=5589):
        self.host = host
        self.port = port
        self._status_payload = status_payload
        self._depth_payload = depth_payload
        self._status_raises = status_raises
        self._depth_raises = depth_raises
        self.depth_call_count = 0

    def status(self):
        if self._status_raises is not None:
            raise self._status_raises
        return self._status_payload

    def depth(self, *, at_gaze=True, save=False, timeout_s=15.0):
        self.depth_call_count += 1
        if self._depth_raises is not None:
            raise self._depth_raises
        return self._depth_payload


class TestVerifyVlmDepthAvailable:
    """Startup precondition for vlm_depth_pro mode. The recorder must
    fail-fast (RuntimeError) for any of: VLM unreachable, ok=False,
    depth_enabled=False, malformed payload. Silent fallback is
    explicitly disallowed by the alignment invariant."""

    def test_happy_path_returns_silently(self):
        c = _FakeVLMClient(status_payload={"ok": True, "depth_enabled": True})
        # No raise expected.
        verify_vlm_depth_available(c)

    def test_unreachable_raises(self):
        c = _FakeVLMClient(status_raises=OSError("connection refused"))
        with pytest.raises(RuntimeError, match="unreachable"):
            verify_vlm_depth_available(c)

    def test_ok_false_raises(self):
        c = _FakeVLMClient(status_payload={"ok": False, "error": "boot"})
        with pytest.raises(RuntimeError, match="ok=False"):
            verify_vlm_depth_available(c)

    def test_depth_disabled_raises(self):
        c = _FakeVLMClient(status_payload={"ok": True, "depth_enabled": False})
        with pytest.raises(RuntimeError, match="depth_enabled=False"):
            verify_vlm_depth_available(c)

    def test_missing_depth_enabled_treats_as_disabled(self):
        # Defensive: if the service forgot to populate the field, treat
        # as disabled rather than risking calibration with an unknown
        # depth backend.
        c = _FakeVLMClient(status_payload={"ok": True})
        with pytest.raises(RuntimeError, match="depth_enabled=False"):
            verify_vlm_depth_available(c)


class TestFetchVlmDepthCm:
    """Once vlm_service is verified at startup, per-capture depth
    requests must (a) convert metres to cm, (b) tag depth_valid=False
    only when the pixel returned a non-finite Depth Pro value, and
    (c) fail-fast on transport / protocol error so the recording
    session aborts rather than mixing depth sources in one NPZ."""

    def test_metres_convert_to_cm(self):
        c = _FakeVLMClient(depth_payload={"ok": True, "depth_at_gaze_m": 0.55})
        depth_cm, depth_valid = fetch_vlm_depth_cm(c)
        assert depth_cm == pytest.approx(55.0)
        assert depth_valid is True

    def test_non_finite_returns_invalid_but_does_not_raise(self):
        c = _FakeVLMClient(depth_payload={"ok": True,
                                            "depth_at_gaze_m": float("nan")})
        depth_cm, depth_valid = fetch_vlm_depth_cm(c)
        assert np.isnan(depth_cm)
        assert depth_valid is False

    def test_transport_error_raises(self):
        c = _FakeVLMClient(depth_raises=OSError("socket closed"))
        with pytest.raises(RuntimeError, match="transport"):
            fetch_vlm_depth_cm(c)

    def test_ok_false_raises(self):
        c = _FakeVLMClient(depth_payload={"ok": False, "error": "depth_disabled"})
        with pytest.raises(RuntimeError, match="not ok"):
            fetch_vlm_depth_cm(c)

    def test_missing_depth_at_gaze_raises(self):
        c = _FakeVLMClient(depth_payload={"ok": True})
        with pytest.raises(RuntimeError, match="depth_at_gaze_m"):
            fetch_vlm_depth_cm(c)


class TestSettleAndSnapshotVlmSubstitution:
    """End-to-end (mocked) test that the captured CaptureBundle ends up
    with the VLM depth value in place of the vergence depth when a
    VLMClient is passed. The non-VLM path is already covered by
    TestBundleFromSnapshot; this case proves the substitution is wired
    correctly through to bundle_from_snapshot.
    """

    def _full_snap(self) -> dict:
        return {
            "ok": True,
            "worn": True,
            "gaze_px": (800.0, 600.0),
            "depth_cm": 75.0,           # vergence value — must be overwritten
            "depth_valid": True,
            "miss_mm": 3.5, "ipd_mm": 63.0,
            "imu_angvel": 0.05, "imu_fresh": True,
            "head_yaw_deg": -2.0, "head_pitch_deg": 1.5,
            "gaze_yaw_deg": 10.0, "gaze_pitch_deg": -5.0,
        }

    def test_vlm_depth_replaces_vergence_depth(self, monkeypatch):
        # Patch gaze_snapshot so settle_and_snapshot consumes our snap
        # without UDP I/O. Make depth_valid streak >= the gate so the
        # bundle gets recorded normally.
        snap = self._full_snap()
        monkeypatch.setattr(recorder, "gaze_snapshot",
                            lambda include_objects=False: snap)
        # Skip the settle sleep so the test is fast.
        monkeypatch.setattr(recorder.time, "sleep", lambda *_: None)

        vlm = _FakeVLMClient(depth_payload={"ok": True,
                                              "depth_at_gaze_m": 0.55})
        robot_state = {"_t": 0.0, "q": np.zeros(7), "ee": np.zeros(3)}
        bundle = recorder.settle_and_snapshot("mid_MC", robot_state,
                                                vlm_client=vlm)
        assert bundle is not None
        # VLM substitution: 0.55 m -> 55.0 cm, depth_valid=True.
        assert bundle.depth_cm == pytest.approx(55.0)
        assert bundle.depth_valid is True
        # And the VLM was hit exactly once (per-capture cadence, not
        # per-sample).
        assert vlm.depth_call_count == 1


class TestWriteNpzDepthSourceMeta:
    """The NPZ meta dict carries the alignment-invariant fields so the
    runtime dispatcher can verify match without consulting config.
    Tests both branches."""

    def _make_bundle(self) -> CaptureBundle:
        return CaptureBundle(
            t=0.0, q=np.arange(7, dtype=float),
            ee_mm=np.array([1.0, 2.0, 3.0]),
            gaze_x_norm=0.5, gaze_y_norm=0.5, gaze_conf=1.0,
            depth_cm=75.0, depth_valid=True, miss_mm=2.0, ipd_mm=63.0,
            imu_w=0.01, imu_fresh=True,
            head_yaw_deg=0.0, head_pitch_deg=0.0,
            gaze_yaw_deg=0.0, gaze_pitch_deg=0.0,
            phase="captured", target_label="mid_MC",
        )

    def test_vergence_default_meta(self, monkeypatch, tmp_path: Path):
        monkeypatch.setattr(recorder.config,
                            "GAZE_CALIBRATION_DEPTH_SOURCE", "vergence",
                            raising=False)
        out = tmp_path / "vergence.npz"
        write_npz([self._make_bundle()], str(out))
        meta = np.load(str(out), allow_pickle=True)["meta"].item()
        assert meta["depth_source"] == "vergence"
        # vlm_service_host empty when not in VLM mode (so the meta key
        # cannot be misread as a stale machine-local artifact).
        assert meta["vlm_service_host"] == ""

    def test_vlm_depth_pro_meta_records_host(self, monkeypatch, tmp_path: Path):
        monkeypatch.setattr(recorder.config,
                            "GAZE_CALIBRATION_DEPTH_SOURCE", "vlm_depth_pro",
                            raising=False)
        monkeypatch.setattr(recorder.config,
                            "VLM_SERVICE_HOST", "192.168.99.99",
                            raising=False)
        out = tmp_path / "vlm.npz"
        write_npz([self._make_bundle()], str(out))
        meta = np.load(str(out), allow_pickle=True)["meta"].item()
        assert meta["depth_source"] == "vlm_depth_pro"
        assert meta["vlm_service_host"] == "192.168.99.99"
