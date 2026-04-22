"""
Baseline pool file enumeration and reference distribution statistics.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from Utils.dataset_exploration.discovery import path_matches_exclude


def list_baseline_xdf_files(
    data_dir: str | Path,
    baseline_subject: str,
    *,
    exclude_substrings: tuple[str, ...] = ("OBS", "old"),
) -> list[str]:
    """
    List .xdf files directly in …/sub-<baseline>/training_data/ (not subfolders).

    Subdirectories (e.g. left-hand or ERRP test sets) are intentionally excluded from the
    baseline pool reference. Skip filenames containing any exclude token (case-insensitive).
    """
    eeg_dir = Path(data_dir) / f"sub-{baseline_subject}" / "training_data"
    if not eeg_dir.is_dir():
        return []
    out: list[str] = []
    for name in sorted(os.listdir(eeg_dir)):
        if not name.lower().endswith(".xdf"):
            continue
        full = eeg_dir / name
        if path_matches_exclude(full, exclude_substrings):
            continue
        out.append(str(full.resolve()))
    return out


def reference_stats(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    min_spread: float = 1e-9,
) -> dict[str, dict[str, float]]:
    """
    Robust location/spread per numeric column (baseline files only).
    """
    skip = {"size_bytes", "is_symlink", "in_baseline_pool", "proposal_rank"}
    if columns is None:
        columns = [
            c
            for c in df.columns
            if c not in skip and df[c].dtype.kind in "fiu" and str(c) != "eval_error"
        ]

    stats_out: dict[str, dict[str, float]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        q1, med, q3 = float(s.quantile(0.25)), float(s.quantile(0.5)), float(s.quantile(0.75))
        iqr = max(q3 - q1, min_spread)
        stats_out[col] = {
            "median": med,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "min": float(s.min()),
            "max": float(s.max()),
            "p95": float(s.quantile(0.95)),
        }

    # abs skew reference for penalty
    skew_cols = [c for c in df.columns if str(c).endswith("__skew_p_mi")]
    if skew_cols:
        chunks: list[pd.Series] = []
        for c in skew_cols:
            s = pd.to_numeric(df[c], errors="coerce").dropna().abs()
            chunks.append(s)
        if chunks:
            all_s = pd.concat(chunks, ignore_index=True)
            stats_out["_ref_abs_skew_p_mi_p95"] = {
                "p95": float(all_s.quantile(0.95)),
                "median": float(all_s.median()),
                "q1": 0.0,
                "q3": 0.0,
                "iqr": 1.0,
                "min": 0.0,
                "max": 0.0,
            }

    return stats_out


def config_snapshot() -> dict[str, Any]:
    """Minimal frozen config for run reproducibility."""
    keys = [
        "FS",
        "LOWCUT",
        "HIGHCUT",
        "CLASSIFY_WINDOW",
        "BASELINE_DURATION",
        "SURFACE_LAPLACIAN_TOGGLE",
        "SELECT_MOTOR_CHANNELS",
        "SHRINKAGE_PARAM_MDM",
        "SHRINKAGE_PARAM_XGB",
        "RECENTERING",
        "LEDOITWOLF",
        "DECODER_BACKEND",
        "TRAINING_SUBJECT",
        "DATA_DIR",
        "ARM_SIDE",
        "SELECT_ERRP_CHANNELS",
        "EXPERIMENT_TYPE",
    ]
    out: dict[str, Any] = {}
    import config as cfg

    for k in keys:
        if hasattr(cfg, k):
            out[k] = getattr(cfg, k)
    return out
