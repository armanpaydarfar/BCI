"""
Per-file feature extraction and cheap separability / ERD metrics (no cross-file training).
"""

from __future__ import annotations

import io
import contextlib
from typing import Any

import numpy as np

import config
import Generate_Riemannian_adaptive as gr_adaptive
from pyriemann.tangentspace import tangent_space

from Utils.stream_utils import load_xdf
from Utils.xgb_feature_pipeline import segment_and_extract_cov_erd


def _tangent_from_cov(cov_matrices: np.ndarray) -> np.ndarray:
    n_ch = cov_matrices.shape[1]
    return tangent_space(cov_matrices, np.eye(n_ch), metric="riemann")


def _fisher_ratio_tangent(X: np.ndarray, y: np.ndarray, rest_label: int, mi_label: int) -> float:
    """Scalar Fisher criterion on full tangent features (higher = more separated means)."""
    m0 = X[y == rest_label]
    m1 = X[y == mi_label]
    if m0.shape[0] < 2 or m1.shape[0] < 2:
        return float("nan")
    mu0, mu1 = m0.mean(axis=0), m1.mean(axis=0)
    # pooled variance per dimension
    v0 = m0.var(axis=0, ddof=1)
    v1 = m1.var(axis=0, ddof=1)
    denom = np.mean(v0 + v1) + 1e-12
    return float(np.sum((mu1 - mu0) ** 2) / denom)


def _euclid_mean_distance_tangent(X: np.ndarray, y: np.ndarray, rest_label: int, mi_label: int) -> float:
    m0 = X[y == rest_label]
    m1 = X[y == mi_label]
    if m0.size == 0 or m1.size == 0:
        return float("nan")
    return float(np.linalg.norm(m0.mean(axis=0) - m1.mean(axis=0)))


def _cohens_d_1d(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt((va + vb) / 2 + 1e-12)
    return float((a.mean() - b.mean()) / sp)


def _erd_metrics(erd: np.ndarray | None, labels: np.ndarray, rest_label: int, mi_label: int) -> dict[str, float]:
    out: dict[str, float] = {
        "erd_mean_abs": float("nan"),
        "erd_median_abs": float("nan"),
        "erd_cohen_d_max": float("nan"),
        "erd_cohen_d_mean": float("nan"),
    }
    if erd is None or erd.size == 0:
        return out
    out["erd_mean_abs"] = float(np.mean(np.abs(erd)))
    out["erd_median_abs"] = float(np.median(np.abs(erd)))

    mask0 = labels == rest_label
    mask1 = labels == mi_label
    ds = []
    for j in range(erd.shape[1]):
        d = _cohens_d_1d(erd[mask1, j], erd[mask0, j])
        if not np.isnan(d):
            ds.append(abs(d))
    if ds:
        out["erd_cohen_d_max"] = float(max(ds))
        out["erd_cohen_d_mean"] = float(np.mean(ds))
    return out


def extract_features_for_file(
    xdf_path: str,
    *,
    include_erd: bool = True,
    include_beta_cov: bool = True,
    apply_csd: bool | None = None,
) -> dict[str, Any]:
    """
    Load XDF, segment, compute covariances and tangent features.

    Returns dict with arrays and labels, or error key.
    """
    if apply_csd is None:
        apply_csd = bool(getattr(config, "SURFACE_LAPLACIAN_TOGGLE", False))

    try:
        eeg_stream, marker_stream = load_xdf(xdf_path, report=False)
    except Exception as e:
        return {"error": f"load_xdf_failed: {e}"}

    try:
        out = segment_and_extract_cov_erd(
            eeg_stream,
            marker_stream,
            compute_erd=include_erd,
            apply_csd=apply_csd,
            return_beta_segments=include_beta_cov,
        )
        if include_beta_cov:
            segments, labels, erd, beta_segments, channel_names = out
        else:
            segments, labels, erd, channel_names = out
            beta_segments = None
    except Exception as e:
        return {"error": f"segment_failed: {e}"}

    labels = np.asarray(labels)
    if labels.size == 0:
        return {"error": "no_segments"}

    classes = np.sort(np.unique(labels))
    if len(classes) != 2:
        return {"error": f"expected_binary_got_{len(classes)}_classes"}

    rest_label, mi_label = int(classes[0]), int(classes[1])

    with contextlib.redirect_stdout(io.StringIO()):
        cov_mu = gr_adaptive.compute_processed_covariances(segments, labels, model_type="xgb")
    cov_mu_tangent = _tangent_from_cov(np.asarray(cov_mu))

    cov_beta_tangent = None
    if include_beta_cov and beta_segments is not None:
        with contextlib.redirect_stdout(io.StringIO()):
            cov_beta = gr_adaptive.compute_processed_covariances(beta_segments, labels, model_type="xgb")
        cov_beta_tangent = _tangent_from_cov(np.asarray(cov_beta))

    erd_arr = None if erd is None else np.asarray(erd)

    n_ch = int(np.asarray(segments).shape[1]) if len(segments) else 0

    return {
        "labels": labels,
        "cov_mu": np.asarray(cov_mu),
        "cov_mu_tangent": cov_mu_tangent,
        "cov_beta_tangent": cov_beta_tangent,
        "erd": erd_arr,
        "rest_label": rest_label,
        "mi_label": mi_label,
        "n_windows": int(len(labels)),
        "n_channels": n_ch,
        "channel_names": list(channel_names),
        "class_counts": {int(c): int(np.sum(labels == c)) for c in classes},
    }


def compute_static_metrics(feats: dict[str, Any]) -> dict[str, float]:
    """Geometry / ERD metrics that do not require a decoder fit."""
    if "error" in feats:
        return {}

    labels = feats["labels"]
    rest_label = feats["rest_label"]
    mi_label = feats["mi_label"]
    X = feats["cov_mu_tangent"]

    n0 = int(np.sum(labels == rest_label))
    n1 = int(np.sum(labels == mi_label))
    imbalance_ratio = max(n0, n1) / (min(n0, n1) + 1e-9)

    fisher = _fisher_ratio_tangent(X, labels, rest_label, mi_label)
    dist = _euclid_mean_distance_tangent(X, labels, rest_label, mi_label)

    erd_m = _erd_metrics(feats.get("erd"), labels, rest_label, mi_label)

    # Trial / marker proxy: unique REST/MI segment counts (windows overcount trials)
    return {
        "n_windows": float(feats["n_windows"]),
        "n_class_rest": float(n0),
        "n_class_mi": float(n1),
        "class_imbalance_ratio": float(imbalance_ratio),
        "fisher_tangent_mu": float(fisher),
        "tangent_mean_l2_distance": float(dist),
        **erd_m,
    }
