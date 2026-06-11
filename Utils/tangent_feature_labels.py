"""
Map tangent-space covariance feature indices to channel pairs (same convention as generate_xgboost_cov_*.py).
"""

from __future__ import annotations

import numpy as np

import config


def infer_n_ch_from_tangent_dim(d: int) -> int:
    """Recover n_ch from tangent length d = n_ch * (n_ch + 1) // 2."""
    return int((np.sqrt(1 + 8 * int(d)) - 1) / 2)


def cov_tangent_idx_to_pair(idx: int, tangent_dim: int) -> tuple[int, int]:
    """Upper-triangle order matching np.triu_indices(n_ch) (row-major within upper triangle)."""
    n_ch = infer_n_ch_from_tangent_dim(tangent_dim)
    tri_i, tri_j = np.triu_indices(n_ch)
    if idx < 0 or idx >= len(tri_i):
        return -1, -1
    return int(tri_i[idx]), int(tri_j[idx])


def erd_bands_from_config() -> list[tuple[float, float]]:
    bands = getattr(config, "XGB_ERD_BANDS", None)
    if bands is not None:
        return [tuple(map(float, b)) for b in bands]
    mu_lo, mu_hi = float(config.LOWCUT), float(config.HIGHCUT)
    beta_hi = float(getattr(config, "XGB_ERD_BETA_HIGH", 30.0))
    return [(mu_lo, mu_hi), (mu_hi, beta_hi)]


def _fmt_band(b: tuple[float, float]) -> str:
    lo, hi = b
    return f"{lo:g}-{hi:g}Hz"


def label_cov_tangent_feature(
    *,
    band: str,
    idx: int,
    tangent_dim: int,
    channel_names: list[str] | None,
) -> str:
    """
    band: 'mu' -> cov_mu_{idx}, 'beta' -> cov_beta_{idx} (training script naming).
    """
    i, j = cov_tangent_idx_to_pair(idx, tangent_dim)
    tri_kind = "diag" if i == j else "offdiag"
    prefix = "cov_mu" if band == "mu" else "cov_beta"
    if (
        channel_names is not None
        and i >= 0
        and j >= 0
        and i < len(channel_names)
        and j < len(channel_names)
    ):
        return f"{prefix}_{idx}[{tri_kind}; {channel_names[i]}-{channel_names[j]}]"
    return f"{prefix}_{idx}[{tri_kind}; ch{i}-ch{j}]"


def label_erd_flat_feature(
    idx: int,
    *,
    n_channels: int,
    erd_n_cols: int,
    channel_names: list[str] | None,
) -> str:
    """Flat ERD index: channel-major then band (same as erd.reshape(-1) in xgb_feature_pipeline)."""
    if n_channels <= 0 or erd_n_cols <= 0:
        return f"erd_{idx}"
    if erd_n_cols % n_channels != 0:
        return f"erd_{idx}"
    n_bands = erd_n_cols // n_channels
    ch_idx = idx // n_bands
    band_idx = idx % n_bands
    erd_bands = erd_bands_from_config()
    band_lbl = _fmt_band(erd_bands[band_idx]) if band_idx < len(erd_bands) else f"band{band_idx}"
    ch_lbl = (
        channel_names[ch_idx]
        if (channel_names is not None and 0 <= ch_idx < len(channel_names))
        else f"ch{ch_idx}"
    )
    return f"erd_{idx}[{band_lbl}; {ch_lbl}]"
