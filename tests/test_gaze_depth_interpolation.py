"""
test_gaze_depth_interpolation.py — REV01 backup depth pipeline
(Plan §3.3 backup, Step 8).

Synthetic-data coverage for ``tools/fit_depth_interpolation.py``:

  1. KDTree correctness on synthetic 3-anchor data — each transit row's
     rewritten D_cm equals the nearest anchor's D_cm.
  2. Per-row Depth_source on transit rows is "vlm_interpolated_nearest_anchor"
     (and on the *_all block too).
  3. Metadata round-trip: meta["affine_map"] is None after the rewrite
     so the runtime falls back to raw vergence; depth_source string is
     unchanged.

Hardware-free; the script is pure numpy + scipy KDTree.

Citations under test:
  - tools/fit_depth_interpolation.py::_interpolate_transit_depths
  - tools/fit_depth_interpolation.py::_rewrite_npz
  - tools/fit_depth_interpolation.py::main
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest


_TOOLS_DIR = str(Path(__file__).resolve().parent.parent / "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import fit_depth_interpolation as fdi  # noqa: E402


# ─── synth NPZ fixture ────────────────────────────────────────────────────

def _make_hybrid_npz_with_known_geometry(
        path: Path,
        *,
        anchor_X: np.ndarray,
        anchor_d_cm: np.ndarray,
        transit_X: np.ndarray,
        transit_raw_d_cm: float = 70.0,
        extra_meta: Optional[Dict[str, Any]] = None) -> None:
    """Hand-build a REV01 hybrid NPZ where anchor and transit X are
    explicit, so a test can predict which anchor each transit row
    should latch onto."""
    n_anchors = len(anchor_X)
    n_transit = len(transit_X)
    n_fit = n_anchors + n_transit

    X = np.vstack([anchor_X, transit_X]).astype(float)
    d_cm = np.concatenate([anchor_d_cm, np.full(n_transit, transit_raw_d_cm)])
    d_cm_verg = np.concatenate([anchor_d_cm * 0.9,  # arbitrary; just must be finite
                                  np.full(n_transit, float("nan"))])
    d_valid = np.ones(n_fit, dtype=bool)
    depth_source = np.array(
        ["vlm_depth_pro"] * n_anchors + ["vergence"] * n_transit,
        dtype="<U64",
    )

    fields: Dict[str, Any] = dict(
        T=np.arange(n_fit, dtype=float),
        Q=np.zeros((n_fit, 7)),
        X=X,
        G=np.zeros((n_fit, 3)),
        D_cm=d_cm,
        D_cm_vergence=d_cm_verg,
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
            [f"a{i}" for i in range(n_anchors)] + [""] * n_transit,
            dtype="<U32",
        ),
        Depth_source=depth_source,
        T_all=np.arange(n_fit, dtype=float),
        Q_all=np.zeros((n_fit, 7)),
        X_all=X.copy(),
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
        Phase_all=np.array(
            ["captured"] * n_anchors + ["transit"] * n_transit,
            dtype="<U16",
        ),
        Target_label_all=np.array(
            [f"a{i}" for i in range(n_anchors)] + [""] * n_transit,
            dtype="<U32",
        ),
        Leg_label_all=np.array(
            [""] * n_anchors + ["transit_a_to_b"] * n_transit,
            dtype="<U64",
        ),
        Depth_source_all=depth_source.copy(),
    )

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

class TestFitDepthInterpolation:
    """Direct unit tests on the KDTree NN lookup + rewrite logic."""

    def test_kdtree_picks_nearest_anchor_d_cm(self, tmp_path: Path):
        # 3 anchors at well-separated EE positions with distinct depths,
        # 3 transit rows whose X clearly snaps to one specific anchor.
        anchor_X = np.array([
            [0.0, 0.0, 0.0],     # anchor 0
            [100.0, 0.0, 0.0],   # anchor 1
            [0.0, 100.0, 0.0],   # anchor 2
        ])
        anchor_d = np.array([55.0, 80.0, 110.0])  # distinctive
        transit_X = np.array([
            [5.0, 5.0, 0.0],     # nearest anchor 0
            [95.0, 5.0, 0.0],    # nearest anchor 1
            [5.0, 95.0, 0.0],    # nearest anchor 2
        ])
        in_path = tmp_path / "interp.npz"
        _make_hybrid_npz_with_known_geometry(
            in_path, anchor_X=anchor_X, anchor_d_cm=anchor_d,
            transit_X=transit_X)

        rc = fdi.main([str(in_path)])
        assert rc == 0

        z = np.load(str(in_path.with_name("interp_interp.npz")),
                     allow_pickle=True)
        d_cm = z["D_cm"]
        # Anchor rows unchanged.
        np.testing.assert_allclose(d_cm[:3], anchor_d)
        # Transit rows latch onto the predicted nearest anchor.
        np.testing.assert_allclose(d_cm[3:], anchor_d)

    def test_transit_rows_tag_interpolated(self, tmp_path: Path):
        anchor_X = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        anchor_d = np.array([55.0, 80.0])
        transit_X = np.array([[10.0, 0.0, 0.0], [90.0, 0.0, 0.0]])
        in_path = tmp_path / "tags.npz"
        _make_hybrid_npz_with_known_geometry(
            in_path, anchor_X=anchor_X, anchor_d_cm=anchor_d,
            transit_X=transit_X)
        rc = fdi.main([str(in_path)])
        assert rc == 0
        z = np.load(str(in_path.with_name("tags_interp.npz")),
                     allow_pickle=True)
        ds = list(z["Depth_source"])
        # Anchor rows keep "vlm_depth_pro", transit rows tag the
        # interpolated label.
        assert ds[:2] == ["vlm_depth_pro"] * 2
        assert ds[2:] == ["vlm_interpolated_nearest_anchor"] * 2
        # And the *_all block carries the same tags on transit rows.
        ds_all = list(z["Depth_source_all"])
        phase_all = list(z["Phase_all"])
        for tag, phase in zip(ds_all, phase_all):
            if phase == "transit":
                assert tag == "vlm_interpolated_nearest_anchor"
            else:
                assert tag == "vlm_depth_pro"

    def test_meta_round_trip_clears_affine_map(self, tmp_path: Path):
        # Seed the input with a populated affine_map dict; the backup
        # script must clear it to None so the runtime knows to fall
        # back to raw vergence (the backup pipeline does NOT extend
        # to runtime per Plan §3.3).
        anchor_X = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        anchor_d = np.array([55.0, 80.0])
        transit_X = np.array([[50.0, 0.0, 0.0]])
        in_path = tmp_path / "meta.npz"
        _make_hybrid_npz_with_known_geometry(
            in_path, anchor_X=anchor_X, anchor_d_cm=anchor_d,
            transit_X=transit_X,
            extra_meta={"recorder": "harmony_free_arm_calibration.py"})
        # Pre-populate affine_map so we can verify the script clears it.
        # (Reopen and rewrite meta only; the script will re-read.)
        z = np.load(str(in_path), allow_pickle=True)
        fields = {k: z[k] for k in z.files if k != "meta"}
        meta = z["meta"].item()
        meta["affine_map"] = {"a": 1.2, "b": -5.0, "R2": 0.5,
                                "max_abs_residual_cm": 12.0}
        fields["meta"] = meta
        z.close()
        np.savez_compressed(str(in_path), **fields)

        rc = fdi.main([str(in_path)])
        assert rc == 0
        z_out = np.load(str(in_path.with_name("meta_interp.npz")),
                         allow_pickle=True)
        meta_out = z_out["meta"].item()
        assert meta_out["affine_map"] is None
        # Unrelated meta keys preserved.
        assert meta_out["depth_source"] == "hybrid_anchor_vlm_transit_vergence"
        assert meta_out["recorder"] == "harmony_free_arm_calibration.py"
        assert meta_out["vlm_service_host"] == "192.168.99.99"
