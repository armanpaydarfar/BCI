"""
test_gaze_calibration_recorder.py — Phase 2.c test (plan §6.3 #1).

Exercises ``harmony_free_arm_calibration.py``'s pure / IO-free helpers
without touching real UDP sockets: the snapshot-to-bundle conversion,
the v2 NPZ writer, and the workspace coverage protocol constant.

Hardware-free per Harmony_Test_Suite_Plan.md §3 — the recorder talks
to two UDP services in production, so this file stays well away from
``RobotLink``, ``free_arm``, ``capture_pose`` (those need integration
tests with the real research-interface binary).

Citations under test (verified 2026-05-19):

  - harmony_free_arm_calibration.py:213-258 ``bundle_from_snapshot``
  - harmony_free_arm_calibration.py:471-580 ``write_npz`` (v2 schema)
  - harmony_free_arm_calibration.py:104-115 ``MANDATORY_GRID``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


# Import once at module load — the recorder's import side effects (UDP
# socket creation) are inside __init__ of RobotLink, not at module
# scope, so simply importing is safe.
import harmony_free_arm_calibration as recorder
from harmony_free_arm_calibration import (
    MANDATORY_GRID,
    CaptureBundle,
    bundle_from_snapshot,
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
