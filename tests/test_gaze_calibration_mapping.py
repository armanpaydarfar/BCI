"""
test_gaze_calibration_mapping.py — Phase 2.c test (plan §6.3 #2).

Tests for ``Utils/gaze/calibration_mapping.py``: the v2 Mahalanobis NN
on (gaze_yaw_deg, gaze_pitch_deg, depth_cm) [Pass-1] or that triple
plus (head_yaw_deg, head_pitch_deg) [Pass-2], plus the workspace clamp.

Hardware-free; the mapping is pure linear algebra on a small NPZ.

Citations under test (verified 2026-05-19):

  - Utils/gaze/calibration_mapping.py:117-181 ``GazeCalibrationMappingV2.__init__``
  - Utils/gaze/calibration_mapping.py:186-217 ``query``
  - Utils/gaze/calibration_mapping.py:267-275 ``_robust_scale``
  - Utils/gaze/calibration_mapping.py:296-313 ``detect_pose_library_version``
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from Utils.gaze.calibration_mapping import (
    PASS1_FEATURE_KEYS,
    PASS2_FEATURE_KEYS,
    GazeCalibrationMappingV2,
    GazeMappingResult,
    MAD_TO_SIGMA,
    WORKSPACE_BOUNDS_MARGIN,
    _robust_scale,
    detect_pose_library_version,
    load_pose_library_v2,
)


# ─── shared fixtures ──────────────────────────────────────────────────────

class _ShimNpz(dict):
    """A dict-with-`.files` so we can pass synthetic data through the
    same code path that ``np.load(...)`` would deliver."""
    @property
    def files(self):
        return list(self.keys())


def _make_npz(N: int = 32, *, seed: int = 0, with_imu: bool = True,
              d_valid_all_true: bool = True) -> _ShimNpz:
    rng = np.random.default_rng(seed)
    data = _ShimNpz()
    data["Q"] = rng.standard_normal((N, 7)) * 0.5
    data["X"] = rng.standard_normal((N, 3)) * 100
    data["Gaze_yaw_deg"] = rng.standard_normal(N) * 10.0
    data["Gaze_pitch_deg"] = rng.standard_normal(N) * 8.0
    data["D_cm"] = rng.uniform(30.0, 200.0, N)
    data["D_valid"] = np.ones(N, dtype=bool) if d_valid_all_true \
        else (rng.random(N) > 0.3)
    if with_imu:
        data["Head_yaw_deg"] = rng.standard_normal(N) * 5.0
        data["Head_pitch_deg"] = rng.standard_normal(N) * 4.0
    return data


# ─── _robust_scale ────────────────────────────────────────────────────────

class TestRobustScale:
    def test_uniform_column_floors_to_eps(self):
        F = np.ones((10, 3))
        s = _robust_scale(F)
        # 1.4826 * MAD(constant) == 0 -> floored to _SCALE_EPS (very small)
        assert np.all(s < 1e-3)
        # Non-zero (the floor)
        assert np.all(s > 0)

    def test_robust_to_outliers(self):
        # Column of mostly-1.0 with one wild outlier; MAD should not be
        # dragged by the outlier the way stddev would.
        col = np.concatenate([np.ones(20), [100.0]])
        F = col[:, None]
        s = _robust_scale(F)
        # MAD of [1,1,...,1, 100] is 0 (median is 1, abs-dev is mostly 0).
        # The floor kicks in -> very small scale, NOT a ~22 std-derived scale.
        assert s[0] < 1e-3

    def test_mad_proportional_constant_correct(self):
        rng = np.random.default_rng(42)
        col = rng.normal(0.0, 1.0, 10000)[:, None]
        s = _robust_scale(col)
        # Under N(0,1), 1.4826 * MAD == ~1.0
        assert s[0] == pytest.approx(1.0, abs=0.05)


# ─── GazeCalibrationMappingV2.__init__ ─────────────────────────────────

class TestMappingConstruction:
    def test_pass1_default_features(self):
        m = GazeCalibrationMappingV2(_make_npz(), use_imu=False)
        assert m.feature_keys == PASS1_FEATURE_KEYS
        assert len(m.feature_scales) == 3
        assert m.num_valid_samples == 32

    def test_pass2_extends_features(self):
        m = GazeCalibrationMappingV2(_make_npz(), use_imu=True)
        assert m.feature_keys == PASS2_FEATURE_KEYS
        assert len(m.feature_scales) == 5

    def test_pass2_missing_head_keys_raises(self):
        # No head_yaw / head_pitch in the NPZ; Pass-2 must KeyError.
        data = _make_npz(with_imu=False)
        with pytest.raises(KeyError, match="Head_yaw_deg"):
            GazeCalibrationMappingV2(data, use_imu=True)

    def test_v1_npz_raises(self):
        # Strip the v2 feature keys -> the v1 NPZ shape -> mapping must
        # KeyError. The caller is expected to dispatch on
        # config.GAZE_CALIBRATION_VERSION before instantiating.
        data = _make_npz()
        del data["Gaze_yaw_deg"]
        with pytest.raises(KeyError, match="Gaze_yaw_deg"):
            GazeCalibrationMappingV2(data, use_imu=False)

    def test_filters_invalid_depth_rows_by_default(self):
        data = _make_npz(d_valid_all_true=False)
        n_invalid = int(np.sum(~data["D_valid"]))
        m = GazeCalibrationMappingV2(data, use_imu=False,
                                       require_depth_valid=True)
        assert m.num_valid_samples == 32 - n_invalid

    def test_can_disable_depth_filter(self):
        data = _make_npz(d_valid_all_true=False)
        m = GazeCalibrationMappingV2(data, use_imu=False,
                                       require_depth_valid=False)
        assert m.num_valid_samples == 32

    def test_all_invalid_raises(self):
        data = _make_npz()
        data["D_valid"] = np.zeros(len(data["D_cm"]), dtype=bool)
        with pytest.raises(ValueError, match="zero rows"):
            GazeCalibrationMappingV2(data, use_imu=False,
                                       require_depth_valid=True)

    def test_q_wrong_shape_raises(self):
        data = _make_npz()
        data["Q"] = data["Q"][:, :6]  # only 6 joints
        with pytest.raises(ValueError, match="Q must be"):
            GazeCalibrationMappingV2(data, use_imu=False)

    def test_feature_length_mismatch_raises(self):
        data = _make_npz()
        data["D_cm"] = data["D_cm"][:10]  # shorter than Q
        with pytest.raises(ValueError, match="does not match"):
            GazeCalibrationMappingV2(data, use_imu=False)


# ─── query() ──────────────────────────────────────────────────────────────

class TestQuery:
    def test_query_returns_known_sample_for_exact_match(self):
        data = _make_npz()
        m = GazeCalibrationMappingV2(data, use_imu=False)
        # Pick a row; query its features -> idx must match.
        i = 7
        features = {
            "Gaze_yaw_deg": float(data["Gaze_yaw_deg"][i]),
            "Gaze_pitch_deg": float(data["Gaze_pitch_deg"][i]),
            "D_cm": float(data["D_cm"][i]),
        }
        r = m.query(features)
        assert r.idx == i
        assert r.dist == pytest.approx(0.0, abs=1e-10)
        # q_target matches the sample's Q (no clamp needed in this synthetic data)
        np.testing.assert_allclose(r.q_target, data["Q"][i, :7])
        np.testing.assert_allclose(r.x_target, data["X"][i, :3])

    def test_query_missing_feature_key_raises(self):
        m = GazeCalibrationMappingV2(_make_npz(), use_imu=False)
        with pytest.raises(KeyError, match="D_cm"):
            m.query({"Gaze_yaw_deg": 1.0, "Gaze_pitch_deg": 2.0})

    def test_query_non_finite_feature_raises(self):
        m = GazeCalibrationMappingV2(_make_npz(), use_imu=False)
        with pytest.raises(ValueError, match="not finite"):
            m.query({"Gaze_yaw_deg": float("nan"), "Gaze_pitch_deg": 0.0,
                     "D_cm": 100.0})

    def test_pass2_uses_head_pose_to_disambiguate(self):
        # Two calibration samples with identical gaze+depth but different
        # head poses. Pass-1 sees them as a tie (depending on which
        # numpy.argmin returns); Pass-2 should pick the one whose head
        # pose matches the query.
        data = _ShimNpz()
        data["Q"] = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        ])
        data["X"] = np.array([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]])
        data["Gaze_yaw_deg"] = np.array([5.0, 5.0])
        data["Gaze_pitch_deg"] = np.array([0.0, 0.0])
        data["D_cm"] = np.array([100.0, 100.0])
        data["D_valid"] = np.array([True, True])
        data["Head_yaw_deg"] = np.array([20.0, -20.0])
        data["Head_pitch_deg"] = np.array([0.0, 0.0])

        m2 = GazeCalibrationMappingV2(data, use_imu=True)
        # Query with head_yaw=15 (closer to sample 0's 20 than to -20).
        r = m2.query({
            "Gaze_yaw_deg": 5.0, "Gaze_pitch_deg": 0.0, "D_cm": 100.0,
            "Head_yaw_deg": 15.0, "Head_pitch_deg": 0.0,
        })
        assert r.idx == 0
        # Query with head_yaw=-15 -> sample 1.
        r = m2.query({
            "Gaze_yaw_deg": 5.0, "Gaze_pitch_deg": 0.0, "D_cm": 100.0,
            "Head_yaw_deg": -15.0, "Head_pitch_deg": 0.0,
        })
        assert r.idx == 1


# ─── workspace clamp ──────────────────────────────────────────────────────

class TestWorkspaceClamp:
    def _data_with_known_q_range(self) -> _ShimNpz:
        # Two samples spanning a known range so we can predict the
        # clamp envelope.
        data = _ShimNpz()
        data["Q"] = np.zeros((2, 7))
        data["Q"][0, 0] = 0.0
        data["Q"][1, 0] = 2.0  # range is [0, 2] on joint 0 -> 5% margin = 0.1
        data["X"] = np.zeros((2, 3))
        data["Gaze_yaw_deg"] = np.array([0.0, 0.0])
        data["Gaze_pitch_deg"] = np.array([0.0, 0.0])
        data["D_cm"] = np.array([100.0, 100.0])
        data["D_valid"] = np.array([True, True])
        return data

    def test_bounds_envelope_uses_5pct_margin(self):
        m = GazeCalibrationMappingV2(self._data_with_known_q_range(),
                                       use_imu=False)
        q_lo, q_hi = m.workspace_bounds
        # Joint 0: min=0, max=2, span=2, margin=0.1 -> [-0.1, 2.1]
        assert q_lo[0] == pytest.approx(-WORKSPACE_BOUNDS_MARGIN * 2.0)
        assert q_hi[0] == pytest.approx(2.0 + WORKSPACE_BOUNDS_MARGIN * 2.0)
        # Joints 1..6 had zero range -> q_lo == q_hi (no clamp room).
        for j in range(1, 7):
            assert q_lo[j] == pytest.approx(0.0)
            assert q_hi[j] == pytest.approx(0.0)

    def test_in_bounds_query_not_clamped(self):
        m = GazeCalibrationMappingV2(self._data_with_known_q_range(),
                                       use_imu=False)
        r = m.query({"Gaze_yaw_deg": 0.0, "Gaze_pitch_deg": 0.0,
                     "D_cm": 100.0})
        # Both calibration rows are inside the envelope by construction.
        assert r.clamped is False
        assert np.all(r.clamp_violations == 0)


# ─── load_pose_library_v2 + detect_pose_library_version ──────────────────

class TestLoaderAndVersionDetect:
    def _write_v1(self, path: Path) -> None:
        np.savez_compressed(
            str(path),
            T=np.zeros(3),
            Q=np.zeros((3, 7)),
            X=np.zeros((3, 3)),
            G=np.zeros((3, 3)),
            meta=dict(side="R", sample_rate_hz=25.0,
                      gaze_confidence_threshold=0.7,
                      units=dict(X="mm", Q="rad", G="normalized_0_to_1")),
        )

    def _write_v2(self, path: Path) -> None:
        np.savez_compressed(
            str(path),
            T=np.zeros(3), Q=np.zeros((3, 7)), X=np.zeros((3, 3)),
            G=np.zeros((3, 3)),
            D_cm=np.full(3, 75.0), D_valid=np.ones(3, dtype=bool),
            Gaze_yaw_deg=np.zeros(3), Gaze_pitch_deg=np.zeros(3),
            Head_yaw_deg=np.zeros(3), Head_pitch_deg=np.zeros(3),
            Miss_mm=np.zeros(3), IPD_mm=np.zeros(3),
            IMU_w=np.zeros(3), IMU_fresh=np.ones(3, dtype=bool),
            meta=dict(version=2, side="R"),
        )

    def test_detect_v1(self, tmp_path: Path):
        p = tmp_path / "v1.npz"
        self._write_v1(p)
        z = np.load(str(p), allow_pickle=True)
        assert detect_pose_library_version(z) == 1

    def test_detect_v2(self, tmp_path: Path):
        p = tmp_path / "v2.npz"
        self._write_v2(p)
        z = np.load(str(p), allow_pickle=True)
        assert detect_pose_library_version(z) == 2

    def test_detect_v2_by_feature_presence_only(self, tmp_path: Path):
        # Older v2 file with no version in meta — sniff by feature keys.
        np.savez_compressed(
            str(tmp_path / "no_meta.npz"),
            Q=np.zeros((1, 7)), X=np.zeros((1, 3)),
            D_cm=np.array([50.0]), Gaze_yaw_deg=np.array([0.0]),
            Gaze_pitch_deg=np.array([0.0]),
        )
        z = np.load(str(tmp_path / "no_meta.npz"), allow_pickle=True)
        assert detect_pose_library_version(z) == 2

    def test_load_pose_library_v2_returns_dict_with_files_marker(self, tmp_path: Path):
        p = tmp_path / "v2.npz"
        self._write_v2(p)
        data = load_pose_library_v2(str(p))
        # Returns a plain dict; '__files__' is a recovery hatch so
        # downstream code that did ``hasattr(z, 'files')`` keeps working.
        assert isinstance(data, dict)
        assert "Q" in data
        assert "D_cm" in data
        assert "__files__" in data


# ─── Transit-phase filtering ──────────────────────────────────────────────

class TestMappingIgnoresTransitPhase:
    """GazeCalibrationMappingV2 reads ``Q`` / ``X`` / feature columns
    directly from the NPZ. The recorder's ``write_npz`` packs only
    ``phase='captured'`` rows into those legacy keys; all-phase data
    lives under ``*_all`` keys that the mapping never touches. This
    test locks that invariant in: a v2 NPZ containing a transit row
    in ``*_all`` MUST NOT influence the mapping's fit (its captured
    rows remain the only source of Q / X / features).
    """

    def test_transit_rows_in_all_keys_do_not_reach_fit(self, tmp_path: Path):
        # Hand-construct a v2 NPZ that mirrors the recorder's two-block
        # layout (legacy=captured-only, *_all=mixed phases).
        N_cap = 4  # 4 captured rows feed the mapping fit
        N_all = 10  # 4 captured + 6 transit in the all-block
        p = tmp_path / "v2_with_transit.npz"
        rng = np.random.default_rng(0)
        np.savez_compressed(
            str(p),
            T=np.zeros(N_cap),
            Q=rng.standard_normal((N_cap, 7)),
            X=rng.standard_normal((N_cap, 3)) * 100,
            G=np.column_stack([np.full(N_cap, 0.5), np.full(N_cap, 0.5),
                                np.full(N_cap, 1.0)]),
            D_cm=rng.uniform(30.0, 200.0, N_cap),
            D_valid=np.ones(N_cap, dtype=bool),
            Gaze_yaw_deg=rng.standard_normal(N_cap) * 10,
            Gaze_pitch_deg=rng.standard_normal(N_cap) * 8,
            Head_yaw_deg=np.zeros(N_cap),
            Head_pitch_deg=np.zeros(N_cap),
            Target_label=np.array(["near_R1", "near_R2", "mid_R1", "far_R3"],
                                   dtype="<U32"),
            # ALL-block — 4 captured + 6 transit. The transit rows have
            # WILDLY different gaze/depth values; if the mapping fit
            # accidentally read them, the Mahalanobis scale would shift.
            T_all=np.zeros(N_all),
            Q_all=np.zeros((N_all, 7)),
            X_all=np.zeros((N_all, 3)),
            G_all=np.zeros((N_all, 3)),
            D_cm_all=np.concatenate([np.full(N_cap, 75.0),
                                       np.full(6, 999.0)]),  # transit poisoned
            D_valid_all=np.ones(N_all, dtype=bool),
            Gaze_yaw_deg_all=np.concatenate([np.zeros(N_cap),
                                                np.full(6, 999.0)]),
            Gaze_pitch_deg_all=np.zeros(N_all),
            Head_yaw_deg_all=np.zeros(N_all),
            Head_pitch_deg_all=np.zeros(N_all),
            Phase_all=np.array(["captured"] * N_cap + ["transit"] * 6,
                                dtype="<U16"),
            Target_label_all=np.array(["x"] * N_all, dtype="<U32"),
            Leg_label_all=np.array([""] * N_cap + ["transit_a_to_b"] * 6,
                                     dtype="<U64"),
            meta=dict(version=2, side="R"),
        )
        z = np.load(str(p), allow_pickle=True)
        m = GazeCalibrationMappingV2(z, use_imu=False,
                                       require_depth_valid=True)
        # Fit must have seen ONLY the 4 captured rows.
        assert m.num_valid_samples == N_cap
        # And the per-feature scales must reflect the captured-row
        # distribution, not the 999-poisoned transit rows. The captured
        # D_cm rows come from rng.uniform(30, 200), so scale should be
        # comfortably below 200; if the transit rows leaked in, scale
        # would explode toward 1.4826 * MAD([... 999s]) >> 200.
        scales = m.feature_scales
        # feature_keys order: Gaze_yaw_deg, Gaze_pitch_deg, D_cm
        d_cm_scale = scales[-1]
        assert d_cm_scale < 200.0, (
            f"transit-phase rows appear to have leaked into the fit: "
            f"D_cm scale={d_cm_scale} (would be ~0 if mapping read only captured)"
        )
