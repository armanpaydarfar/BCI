"""
test_gaze_affine_fit.py — REV01 offline affine-fit script
(Plan §3.3, Step 7).

Synthetic-data coverage for ``tools/fit_vergence_affine.py``:

  1. Perfect data → recovers (a≈1.0, b≈0.0), R² ≈ 1.0, exits 0.
  2. Linear-with-offset data → recovers exact coefficients.
  3. Non-linear data → R² below default threshold → exits non-zero.
  4. Missing D_cm_vergence column → meaningful error.
  5. Rewrite preserves unrelated NPZ keys (G/Q/X/Phase_all, …).

Hardware-free; the fit is pure linear algebra on a synthetic NPZ.

Citations under test:
  - tools/fit_vergence_affine.py::_fit_affine
  - tools/fit_vergence_affine.py::_select_anchor_pairs
  - tools/fit_vergence_affine.py::_rewrite_npz
  - tools/fit_vergence_affine.py::main
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest


# The script lives under tools/; make sure pytest can import it without
# polluting sys.path globally.
_TOOLS_DIR = str(Path(__file__).resolve().parent.parent / "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import fit_vergence_affine as fva  # noqa: E402 — see above


# ─── synth NPZ fixture ────────────────────────────────────────────────────

def _make_hybrid_npz(path: Path, *,
                     d_vergence_anchors: np.ndarray,
                     d_vlm_anchors: np.ndarray,
                     n_transit: int = 6,
                     transit_depth_cm: float = 70.0,
                     include_d_cm_vergence: bool = True,
                     extra_meta: Optional[Dict[str, Any]] = None) -> None:
    """Build a minimal REV01 hybrid NPZ that the fit script can consume.

    The fit-block (legacy keys) layout:
      anchors first, then transit rows. ``D_cm_vergence`` carries the
      vergence reading on the anchor rows and NaN on transit rows (which
      is how the recorder's Step 2 path leaves it).
    """
    n_anchors = len(d_vergence_anchors)
    n_fit = n_anchors + n_transit

    d_cm = np.concatenate([d_vlm_anchors,
                            np.full(n_transit, transit_depth_cm)])
    d_cm_verg = np.concatenate([d_vergence_anchors,
                                  np.full(n_transit, float("nan"))])
    d_valid = np.ones(n_fit, dtype=bool)
    depth_source = np.array(
        ["vlm_depth_pro"] * n_anchors + ["vergence"] * n_transit,
        dtype="<U64",
    )

    fields: Dict[str, Any] = dict(
        T=np.arange(n_fit, dtype=float),
        Q=np.zeros((n_fit, 7)),
        X=np.zeros((n_fit, 3)),
        G=np.column_stack([np.full(n_fit, 0.5),
                            np.full(n_fit, 0.5),
                            np.full(n_fit, 1.0)]),
        D_cm=d_cm,
        D_valid=d_valid,
        Miss_mm=np.zeros(n_fit),
        IPD_mm=np.full(n_fit, 63.0),
        IMU_w=np.zeros(n_fit),
        IMU_fresh=np.ones(n_fit, dtype=bool),
        Head_yaw_deg=np.zeros(n_fit),
        Head_pitch_deg=np.zeros(n_fit),
        Gaze_yaw_deg=np.zeros(n_fit),
        Gaze_pitch_deg=np.zeros(n_fit),
        Target_label=np.array(
            [f"near_R{i+1}" for i in range(n_anchors)] + [""] * n_transit,
            dtype="<U32",
        ),
        Depth_source=depth_source,
        # *_all block (same content here for simplicity)
        T_all=np.arange(n_fit, dtype=float),
        Q_all=np.zeros((n_fit, 7)),
        X_all=np.zeros((n_fit, 3)),
        G_all=np.zeros((n_fit, 3)),
        D_cm_all=d_cm.copy(),
        D_valid_all=d_valid.copy(),
        Miss_mm_all=np.zeros(n_fit),
        IPD_mm_all=np.full(n_fit, 63.0),
        IMU_w_all=np.zeros(n_fit),
        IMU_fresh_all=np.ones(n_fit, dtype=bool),
        Head_yaw_deg_all=np.zeros(n_fit),
        Head_pitch_deg_all=np.zeros(n_fit),
        Gaze_yaw_deg_all=np.zeros(n_fit),
        Gaze_pitch_deg_all=np.zeros(n_fit),
        Phase_all=np.array(["captured"] * n_anchors + ["transit"] * n_transit,
                            dtype="<U16"),
        Target_label_all=np.array(
            [f"near_R{i+1}" for i in range(n_anchors)] + [""] * n_transit,
            dtype="<U32",
        ),
        Leg_label_all=np.array(
            [""] * n_anchors + ["transit_a_to_b"] * n_transit,
            dtype="<U64",
        ),
        Depth_source_all=depth_source.copy(),
    )

    if include_d_cm_vergence:
        fields["D_cm_vergence"] = d_cm_verg

    meta: Dict[str, Any] = dict(
        version=2, side="R",
        depth_source="hybrid_anchor_vlm_transit_vergence",
        affine_map=None,
        vlm_service_host="192.168.99.99",
    )
    if extra_meta is not None:
        meta.update(extra_meta)
    fields["meta"] = meta

    np.savez_compressed(str(path), **fields)


# ─── tests ────────────────────────────────────────────────────────────────

class TestFitVergenceAffine:
    """Direct unit tests on the fit + rewrite logic."""

    def test_perfect_data_recovers_identity(self, tmp_path: Path, capsys):
        # D_vlm == D_vergence exactly → a=1, b=0, R²=1.
        verg = np.array([30.0, 60.0, 90.0, 120.0, 150.0])
        in_path = tmp_path / "perfect.npz"
        _make_hybrid_npz(in_path, d_vergence_anchors=verg,
                          d_vlm_anchors=verg.copy())
        rc = fva.main([str(in_path)])
        assert rc == 0, "Perfect data must pass --min-r2 and --max-residual-cm"
        # Inspect the rewritten NPZ.
        out_path = tmp_path / "perfect_affine.npz"
        assert out_path.exists()
        z = np.load(str(out_path), allow_pickle=True)
        meta = z["meta"].item()
        am = meta["affine_map"]
        assert am["a"] == pytest.approx(1.0, abs=1e-9)
        assert am["b"] == pytest.approx(0.0, abs=1e-9)
        assert am["R2"] == pytest.approx(1.0, abs=1e-9)

    def test_linear_with_offset_recovers_coefficients(self, tmp_path: Path):
        # Synthetic ground truth: D_vlm = 1.3 · D_vergence - 12.0 + tiny noise.
        rng = np.random.default_rng(0)
        verg = np.linspace(40.0, 160.0, 15)
        true_a, true_b = 1.3, -12.0
        vlm = true_a * verg + true_b + rng.normal(0.0, 0.05, size=verg.shape)
        in_path = tmp_path / "linear.npz"
        _make_hybrid_npz(in_path, d_vergence_anchors=verg,
                          d_vlm_anchors=vlm)
        rc = fva.main([str(in_path)])
        assert rc == 0
        z = np.load(str(in_path.with_name("linear_affine.npz")),
                     allow_pickle=True)
        am = z["meta"].item()["affine_map"]
        assert am["a"] == pytest.approx(true_a, abs=0.01)
        assert am["b"] == pytest.approx(true_b, abs=0.5)
        # Transit rows had D_cm = 70.0 (vergence value); the rewrite
        # must apply a*70 + b to each transit row.
        d_cm = z["D_cm"]
        # Anchor rows are unchanged (recovered ~true_a*verg + true_b).
        np.testing.assert_allclose(d_cm[:len(verg)], vlm, atol=1e-9)
        # Transit rows are rewritten to a*70 + b.
        expected_transit = am["a"] * 70.0 + am["b"]
        np.testing.assert_allclose(d_cm[len(verg):], expected_transit,
                                    atol=1e-9)
        # And Depth_source on those transit rows now reads "vergence_affine".
        ds = z["Depth_source"]
        assert list(ds[len(verg):]) == ["vergence_affine"] * (len(d_cm) - len(verg))
        # Anchor rows still tagged "vlm_depth_pro".
        assert list(ds[:len(verg)]) == ["vlm_depth_pro"] * len(verg)

    def test_non_linear_data_exits_non_zero(self, tmp_path: Path):
        # D_vlm is wildly non-linear in D_vergence: R² will plummet
        # because vergence ramps monotonically and vlm zigzags.
        verg = np.linspace(40.0, 160.0, 15)
        rng = np.random.default_rng(0)
        vlm = 80.0 + 40.0 * rng.standard_normal(size=verg.shape)
        in_path = tmp_path / "nonlinear.npz"
        _make_hybrid_npz(in_path, d_vergence_anchors=verg,
                          d_vlm_anchors=vlm)
        rc = fva.main([str(in_path)])
        # Default --min-r2 is 0.85; random noise around an offset gives
        # R² well below that. Could exit 2 (R²) or 3 (residual); either
        # is fine as long as it's non-zero.
        assert rc in (2, 3), f"Expected non-zero exit; got {rc}"

    def test_missing_d_cm_vergence_column_meaningful_error(self,
                                                            tmp_path: Path):
        # Build a pre-REV01 v2 NPZ that has no D_cm_vergence column.
        verg = np.array([30.0, 60.0, 90.0])
        in_path = tmp_path / "no_vergence_col.npz"
        _make_hybrid_npz(in_path, d_vergence_anchors=verg,
                          d_vlm_anchors=verg.copy(),
                          include_d_cm_vergence=False)
        # _select_anchor_pairs surfaces a KeyError naming the column.
        with pytest.raises(KeyError, match="D_cm_vergence"):
            fva.main([str(in_path)])

    def test_rewrite_preserves_unrelated_keys(self, tmp_path: Path):
        verg = np.array([40.0, 80.0, 120.0])
        vlm = 1.2 * verg + 5.0
        in_path = tmp_path / "preserve.npz"
        _make_hybrid_npz(in_path, d_vergence_anchors=verg,
                          d_vlm_anchors=vlm,
                          n_transit=2,
                          transit_depth_cm=60.0,
                          extra_meta={"recorder": "harmony_free_arm_calibration.py"})
        rc = fva.main([str(in_path)])
        assert rc == 0
        out_path = in_path.with_name("preserve_affine.npz")
        z_in = np.load(str(in_path), allow_pickle=True)
        z_out = np.load(str(out_path), allow_pickle=True)
        try:
            # Unrelated keys round-trip byte-for-byte (G, Q, X,
            # Phase_all, Target_label_all, IMU_w, IPD_mm, ...).
            for key in ("G", "Q", "X", "Phase_all", "Target_label_all",
                         "IMU_w", "IPD_mm", "Head_yaw_deg", "Gaze_yaw_deg",
                         "G_all", "Phase_all", "Leg_label_all"):
                if key in z_in.files:
                    np.testing.assert_array_equal(z_in[key], z_out[key])
            # Meta keys outside affine_map are preserved.
            meta_in = z_in["meta"].item()
            meta_out = z_out["meta"].item()
            for k in ("version", "side", "depth_source",
                       "vlm_service_host", "recorder"):
                assert meta_in[k] == meta_out[k]
            # affine_map is populated.
            am = meta_out["affine_map"]
            assert isinstance(am, dict)
            assert {"a", "b", "R2", "max_abs_residual_cm"} <= set(am.keys())
        finally:
            z_in.close()
            z_out.close()
