#!/usr/bin/env python3
"""
ErrP database traversal and file ranking.

Scans all XDF files under DATA_DIR for ErrP-relevant events
(ROBOT_EARLYSTOP=340 or ERRP_STIM_ERROR=430). For each file:
  - Extracts ErrP epochs (0–800 ms post-event, 1–10 Hz bandpass, frontocentral channels)
  - Computes ERP signal metrics: ERN peak amplitude difference, Cohen's d, SNR, d-prime
  - Runs xDAWN + MDM K-fold classification (AUC, balanced accuracy, Brier score)
  - Composites these into a ranking score

Outputs (default: ~/Documents/errp_exploration/):
  errp_candidates.csv   — per-file metrics (all evaluated files)
  errp_ranked.csv       — sorted by composite_score, ErrP-rich files first
  ERRP_REPORT.txt       — human-readable summary with top-ranked file list

Examples:
  python explore_errp_library.py
  python explore_errp_library.py --out-dir ~/Documents/errp_run_002
  python explore_errp_library.py --max-files 30 --n-splits 3
  python explore_errp_library.py --data-dir /mnt/nas/study
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable  # noqa: F401  (used in string annotations)

os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")
os.environ["MNE_USE_NUMBA"] = "false"

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

import config
from Utils.dataset_exploration.discovery import (
    discover_xdf_files,
    duplicate_groups_by_size,
    index_xdf_path,
)
from Utils.errp_feature_pipeline import load_and_preprocess_errp_xdf


# ---------------------------------------------------------------------------
# ERP signal metrics
# ---------------------------------------------------------------------------

_ERN_TMIN_MS = 80    # ms post-event
_ERN_TMAX_MS = 200   # ms — ERN peak window
_PE_TMIN_MS  = 200   # ms
_PE_TMAX_MS  = 400   # ms — Pe window


def _erp_signal_metrics(
    X: np.ndarray,
    y: np.ndarray,
    fs: float,
    epoch_tmin: float,
) -> dict[str, float]:
    """
    Compute ErrP signal quality metrics from epoched data.

    Args:
        X:          (n_epochs, n_channels, n_times)
        y:          (n_epochs,) — 1 = error, 0 = correct
        fs:         sampling frequency (Hz)
        epoch_tmin: epoch start offset from event (s, typically 0)

    Returns flat dict of signal metrics.
    """
    metrics: dict[str, float] = {}
    if X.shape[0] < 2 or len(np.unique(y)) < 2:
        return metrics

    # Sample indices for ERN and Pe windows
    def _ms_to_samp(ms: float) -> int:
        return int(round((ms / 1000.0 - epoch_tmin) * fs))

    ern_s = max(0, _ms_to_samp(_ERN_TMIN_MS))
    ern_e = min(X.shape[2], _ms_to_samp(_ERN_TMAX_MS))
    pe_s  = max(0, _ms_to_samp(_PE_TMIN_MS))
    pe_e  = min(X.shape[2], _ms_to_samp(_PE_TMAX_MS))

    error_mask   = y == 1
    correct_mask = y == 0

    X_err = X[error_mask]    # (n_err, n_ch, n_t)
    X_cor = X[correct_mask]  # (n_cor, n_ch, n_t)

    # Grand averages
    erp_err = X_err.mean(axis=0)   # (n_ch, n_t)
    erp_cor = X_cor.mean(axis=0)
    diff_wave = erp_err - erp_cor  # (n_ch, n_t)

    # ERN amplitude: min (most negative) of diff wave in ERN window, mean over channels
    if ern_e > ern_s:
        ern_window = diff_wave[:, ern_s:ern_e]      # (n_ch, n_ern_t)
        ern_peak_diff = float(np.min(ern_window))   # most negative = ERN
        ern_mean_diff = float(np.mean(ern_window))
        metrics["ern_peak_diff_uv"] = ern_peak_diff
        metrics["ern_mean_diff_uv"] = ern_mean_diff

        # Per-epoch ERN amplitude (mean over ERN window and channels) for Cohen's d
        ern_per_epoch = X[:, :, ern_s:ern_e].mean(axis=(1, 2))  # (n_epochs,)
        ern_err_vals  = ern_per_epoch[error_mask]
        ern_cor_vals  = ern_per_epoch[correct_mask]
        pooled_std    = float(np.sqrt(
            (np.var(ern_err_vals, ddof=1) + np.var(ern_cor_vals, ddof=1)) / 2.0
        )) if len(ern_err_vals) > 1 and len(ern_cor_vals) > 1 else 1e-9
        cohens_d = float((np.mean(ern_cor_vals) - np.mean(ern_err_vals)) / max(pooled_std, 1e-9))
        # Positive d → error epochs more negative than correct (correct sign for ERN)
        metrics["ern_cohens_d"] = cohens_d

        # SNR: mean signal amplitude / mean noise amplitude (baseline-noise via std of correct)
        signal_amp = abs(ern_peak_diff)
        noise_std  = float(np.std(ern_per_epoch[correct_mask])) if correct_mask.sum() > 1 else 1.0
        metrics["ern_snr"] = signal_amp / max(noise_std, 1e-9)

        # d-prime: separation of error vs correct ERN distributions
        mu_e = float(np.mean(ern_err_vals))
        mu_c = float(np.mean(ern_cor_vals))
        sd_e = float(np.std(ern_err_vals, ddof=1)) if len(ern_err_vals) > 1 else 1.0
        sd_c = float(np.std(ern_cor_vals, ddof=1)) if len(ern_cor_vals) > 1 else 1.0
        dprime_denom = math.sqrt((sd_e**2 + sd_c**2) / 2.0 + 1e-18)
        metrics["ern_dprime"] = abs(mu_e - mu_c) / dprime_denom
    else:
        for k in ("ern_peak_diff_uv", "ern_mean_diff_uv", "ern_cohens_d", "ern_snr", "ern_dprime"):
            metrics[k] = float("nan")

    # Pe amplitude: max of diff wave in Pe window (positive component)
    if pe_e > pe_s:
        pe_window = diff_wave[:, pe_s:pe_e]
        metrics["pe_peak_diff_uv"] = float(np.max(pe_window))
        metrics["pe_mean_diff_uv"] = float(np.mean(pe_window))
    else:
        metrics["pe_peak_diff_uv"] = float("nan")
        metrics["pe_mean_diff_uv"] = float("nan")

    # Frontocentral channel peak: find channel with largest ERN (most negative in ERN window)
    if ern_e > ern_s:
        ch_ern_min = diff_wave[:, ern_s:ern_e].min(axis=1)  # (n_ch,)
        metrics["best_ch_ern_idx"] = float(int(np.argmin(ch_ern_min)))
        metrics["best_ch_ern_amp"] = float(np.min(ch_ern_min))

    return metrics


# ---------------------------------------------------------------------------
# xDAWN + MDM K-fold evaluation
# ---------------------------------------------------------------------------

def _errp_kfold(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    n_filters: int = 4,
    random_state: int = 42,
    on_fold: "Callable[[int, int], None] | None" = None,
) -> dict[str, float]:
    """
    Stratified K-fold xDAWN + MDM evaluation on ErrP epochs.

    Args:
        X:  (n_epochs, n_channels, n_times)
        y:  (n_epochs,) — 1 = error, 0 = correct

    Returns flat dict of classifier metrics.
    """
    from pyriemann.estimation import XdawnCovariances
    from pyriemann.classification import MDM

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or np.min(counts) < 2:
        return {"kfold_error": "insufficient_class_samples"}

    n_splits = min(n_splits, int(np.min(counts)))
    if n_splits < 2:
        return {"kfold_error": "n_splits_lt_2"}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_prob_error = np.full(len(y), np.nan)
    oof_pred       = np.full(len(y), -1, dtype=int)

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        try:
            xdc = XdawnCovariances(nfilter=n_filters, estimator="lwf")
            xdc.fit(X[train_idx], y[train_idx])

            cov_train = xdc.transform(X[train_idx])
            cov_test  = xdc.transform(X[test_idx])

            clf = MDM()
            clf.fit(cov_train, y[train_idx])

            pr = clf.predict_proba(cov_test)  # (n_test, 2)
            # probability of class 1 (error)
            class_list = list(clf.classes_)
            if 1 in class_list:
                idx_err = class_list.index(1)
                oof_prob_error[test_idx] = pr[:, idx_err]
            pred = clf.predict(cov_test)
            oof_pred[test_idx] = pred
        except Exception as exc:
            return {"kfold_error": f"{type(exc).__name__}: {exc}"}

        if on_fold is not None:
            on_fold(fold_i, n_splits)

    valid = ~np.isnan(oof_prob_error)
    if not np.any(valid):
        return {"kfold_error": "no_oof"}

    yt  = y[valid]
    pm  = oof_prob_error[valid]
    pa  = oof_pred[valid]

    yb  = (yt == 1).astype(int)
    auc = float(roc_auc_score(yb, pm)) if np.unique(yb).size == 2 else float("nan")
    bal = float(balanced_accuracy_score(yt, pa))
    bri = float(brier_score_loss(yb, pm)) if np.unique(yb).size == 2 else float("nan")

    n_err = int(np.sum(y == 1))
    n_cor = int(np.sum(y == 0))

    return {
        "auc": auc,
        "balanced_accuracy": bal,
        "brier": bri,
        "n_error_epochs": n_err,
        "n_correct_epochs": n_cor,
        "n_total_epochs": int(len(y)),
        "class_balance": float(n_err) / float(max(n_err + n_cor, 1)),
    }


# ---------------------------------------------------------------------------
# Per-file evaluation
# ---------------------------------------------------------------------------

def evaluate_errp_path(
    xdf_path: str,
    *,
    n_splits: int = 5,
    n_filters: int = 4,
    min_epochs: int = 4,
    on_fold: "Callable[[int, int], None] | None" = None,
) -> dict:
    """
    Full ErrP evaluation for one XDF file.

    Returns a flat dict ready for pandas DataFrame construction.
    """
    row: dict = {
        "path": xdf_path,
        "filename": Path(xdf_path).name,
    }
    t0 = time.perf_counter()

    # --- Load and preprocess ---
    try:
        X, y, ch_names, stats = load_and_preprocess_errp_xdf(xdf_path)
    except Exception as exc:
        row["eval_error"] = f"{type(exc).__name__}: {exc}"
        row["eval_s_total"] = float(time.perf_counter() - t0)
        return row

    row["n_error_total"]   = int(stats.get("n_error_total", 0))
    row["n_correct_total"] = int(stats.get("n_correct_total", 0))
    row["n_error_epochs"]  = int(stats.get("n_error_kept", 0))
    row["n_correct_epochs"]= int(stats.get("n_correct_kept", 0))
    row["n_dropped_bounds"]    = int(stats.get("n_dropped_bounds", 0))
    row["n_dropped_artifact"]  = int(stats.get("n_dropped_artifact", 0))
    row["n_channels"]      = len(ch_names)
    row["ch_names"]        = ",".join(ch_names)

    # Determine which marker set was found
    n_err  = int(stats.get("n_error_kept", 0))
    n_cor  = int(stats.get("n_correct_kept", 0))
    n_any  = n_err + n_cor

    if n_any < min_epochs or n_err < 2 or n_cor < 2:
        row["eval_error"] = f"below_min_epochs({n_any}<{min_epochs} or err={n_err}<2 or cor={n_cor}<2)"
        row["eval_s_total"] = float(time.perf_counter() - t0)
        return row

    # --- ERP signal metrics ---
    try:
        tmin = float(getattr(config, "ERRP_EPOCH_TMIN", 0.0))
        fs   = float(getattr(config, "FS", 512))
        sig = _erp_signal_metrics(X, y, fs=fs, epoch_tmin=tmin)
        for k, v in sig.items():
            row[k] = v
    except Exception as exc:
        row["signal_metrics_error"] = f"{type(exc).__name__}: {exc}"

    # --- K-fold classification ---
    t_kfold = time.perf_counter()
    kf = _errp_kfold(X, y, n_splits=n_splits, n_filters=n_filters, on_fold=on_fold)
    row["eval_s_kfold_s"] = float(time.perf_counter() - t_kfold)

    if "kfold_error" in kf:
        row["eval_error"] = kf["kfold_error"]
    else:
        for k, v in kf.items():
            row[k] = v

    row["eval_s_total"] = float(time.perf_counter() - t0)
    return row


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def _build_reference(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute per-metric reference stats (median/IQR) from successfully evaluated rows."""
    ok = df[df.get("eval_error", pd.Series([""] * len(df))).isna()
            | (df.get("eval_error", pd.Series([""] * len(df))).astype(str).str.strip() == "")]
    ref: dict[str, dict[str, float]] = {}
    for col in ("auc", "balanced_accuracy", "ern_cohens_d", "ern_snr", "ern_dprime",
                "ern_peak_diff_uv", "pe_peak_diff_uv"):
        if col not in ok.columns:
            continue
        vals = pd.to_numeric(ok[col], errors="coerce").dropna().values
        if len(vals) < 2:
            continue
        q1, q3 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        ref[col] = {
            "median": float(np.median(vals)),
            "iqr":    float(q3 - q1) or 1.0,
            "q1": q1,
            "q3": q3,
        }
    return ref


def _z_higher(x: float, r: dict[str, float]) -> float:
    if math.isnan(x):
        return 0.0
    return (x - r["median"]) / (r.get("iqr", 1.0) or 1e-9)


def _z_lower(x: float, r: dict[str, float]) -> float:
    if math.isnan(x):
        return 0.0
    return (r["median"] - x) / (r.get("iqr", 1.0) or 1e-9)


def add_composite_score(df: pd.DataFrame, ref: dict[str, dict[str, float]]) -> pd.DataFrame:
    """
    Add z-score columns and composite_score to df in-place (copy returned).

    Composite = mean of three block z-scores:
      1. classifier quality   (AUC, balanced accuracy, -Brier)
      2. ERN amplitude        (cohens_d, snr, dprime)
      3. ERP peak             (ern_peak_diff_uv — more negative = better)

    ERN peak_diff is negative for a genuine ERN (error more negative than correct),
    so we use _z_lower (more negative = better score) for that column.
    """
    out = df.copy()

    # -- Classifier block z-scores --
    z_clf: list[str] = []
    for col, fn in [("auc", _z_higher), ("balanced_accuracy", _z_higher)]:
        if col in out.columns and col in ref:
            zc = f"z__{col}"
            out[zc] = out[col].apply(lambda v, r=ref[col], f=fn: f(float(v) if pd.notna(v) else float("nan"), r))
            z_clf.append(zc)
    if "brier" in out.columns and "brier" in ref:
        zc = "z__brier"
        out[zc] = out["brier"].apply(lambda v, r=ref["brier"]: _z_lower(float(v) if pd.notna(v) else float("nan"), r))
        z_clf.append(zc)

    # -- ERN amplitude block z-scores --
    z_ern: list[str] = []
    for col in ("ern_cohens_d", "ern_snr", "ern_dprime"):
        if col in out.columns and col in ref:
            zc = f"z__{col}"
            out[zc] = out[col].apply(lambda v, r=ref[col]: _z_higher(float(v) if pd.notna(v) else float("nan"), r))
            z_ern.append(zc)

    # -- ERP peak block z-scores --
    z_peak: list[str] = []
    # ern_peak_diff is negative → more negative = stronger ERN → lower is better for the raw value
    if "ern_peak_diff_uv" in out.columns and "ern_peak_diff_uv" in ref:
        zc = "z__ern_peak_diff_uv"
        out[zc] = out["ern_peak_diff_uv"].apply(
            lambda v, r=ref["ern_peak_diff_uv"]: _z_lower(float(v) if pd.notna(v) else float("nan"), r)
        )
        z_peak.append(zc)
    if "pe_peak_diff_uv" in out.columns and "pe_peak_diff_uv" in ref:
        zc = "z__pe_peak_diff_uv"
        out[zc] = out["pe_peak_diff_uv"].apply(
            lambda v, r=ref["pe_peak_diff_uv"]: _z_higher(float(v) if pd.notna(v) else float("nan"), r)
        )
        z_peak.append(zc)

    # -- Block means + composite --
    blocks: list[list[str]] = [b for b in [z_clf, z_ern, z_peak] if b]

    def _row_composite(row: pd.Series) -> float:
        block_means: list[float] = []
        for block in blocks:
            vals = [float(row[c]) for c in block if c in row.index and pd.notna(row[c])]
            if vals:
                block_means.append(float(np.mean(vals)))
        return float(np.mean(block_means)) if block_means else float("nan")

    ev_col = out.get("eval_error", pd.Series([""] * len(out)))
    has_error = ev_col.notna() & (ev_col.astype(str).str.strip() != "")
    out["composite_score"] = out.apply(_row_composite, axis=1)
    out.loc[has_error, "composite_score"] = float("nan")

    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _write_text_report(
    out_dir: Path,
    df: pd.DataFrame,
    ranked: pd.DataFrame,
    *,
    data_dir: Path,
    n_files_scanned: int,
    n_files_with_errp: int,
    elapsed_s: float,
    top_n: int = 30,
) -> Path:
    report_path = out_dir / "ERRP_REPORT.txt"
    with open(report_path, "w") as fh:
        fh.write("=" * 78 + "\n")
        fh.write("Harmony — ErrP database exploration report\n")
        fh.write("=" * 78 + "\n\n")
        fh.write(f"  data_dir          : {data_dir}\n")
        fh.write(f"  files scanned     : {n_files_scanned}\n")
        fh.write(f"  files with ErrP markers found : {n_files_with_errp}\n")
        fh.write(f"  files evaluated successfully  : {len(ranked)}\n")
        fh.write(f"  elapsed           : {elapsed_s:.1f}s\n\n")

        # Aggregate stats
        if len(ranked) > 0:
            fh.write("AGGREGATE METRICS (over successfully evaluated files)\n")
            fh.write("-" * 50 + "\n")
            for col, label in [
                ("auc",              "AUC (xDAWN+MDM)"),
                ("balanced_accuracy","Balanced Accuracy"),
                ("brier",            "Brier Score"),
                ("ern_cohens_d",     "ERN Cohen's d"),
                ("ern_snr",          "ERN SNR"),
                ("ern_dprime",       "ERN d-prime"),
                ("ern_peak_diff_uv", "ERN peak diff (µV)"),
                ("pe_peak_diff_uv",  "Pe  peak diff (µV)"),
            ]:
                if col not in ranked.columns:
                    continue
                vals = pd.to_numeric(ranked[col], errors="coerce").dropna()
                if len(vals) == 0:
                    continue
                fh.write(
                    f"  {label:<30}  median={np.median(vals):.3f}  "
                    f"IQR=[{np.percentile(vals,25):.3f},{np.percentile(vals,75):.3f}]  "
                    f"range=[{vals.min():.3f},{vals.max():.3f}]\n"
                )
            fh.write("\n")

        # Ranked list
        fh.write(f"TOP {min(top_n, len(ranked))} FILES BY COMPOSITE SCORE\n")
        fh.write("-" * 78 + "\n")
        fh.write(
            f"{'rank':>4}  {'composite':>9}  {'AUC':>6}  {'d':>5}  {'SNR':>5}  "
            f"{'d-prime':>7}  {'err':>4}  {'cor':>4}  subj              file\n"
        )
        fh.write("-" * 78 + "\n")
        for _, row in ranked.head(top_n).iterrows():
            def _f(k: str) -> str:
                v = row.get(k, float("nan"))
                try:
                    return f"{float(v):.3f}" if pd.notna(v) else "  —  "
                except (TypeError, ValueError):
                    return "  —  "

            fh.write(
                f"{int(row.get('proposal_rank', 0)):>4}  "
                f"{_f('composite_score'):>9}  "
                f"{_f('auc'):>6}  "
                f"{_f('ern_cohens_d'):>5}  "
                f"{_f('ern_snr'):>5}  "
                f"{_f('ern_dprime'):>7}  "
                f"{int(row.get('n_error_epochs', 0)):>4}  "
                f"{int(row.get('n_correct_epochs', 0)):>4}  "
                f"{str(row.get('subject', '—')):<18}"
                f"{row.get('filename', '')}\n"
            )

        # Error summary
        if "eval_error" in df.columns:
            ev = df["eval_error"].dropna().astype(str)
            ev = ev[ev.str.strip() != ""]
            if len(ev) > 0:
                fh.write("\nERROR SUMMARY\n" + "-" * 40 + "\n")
                ec: Counter = Counter()
                for e in ev:
                    ec[e.split("(")[0].split(":")[0].strip()[:50]] += 1
                for k, n in ec.most_common(10):
                    fh.write(f"  {k:<50}  n={n}\n")

        fh.write("\n" + "=" * 78 + "\n")
        fh.write(f"Full metrics: errp_candidates.csv  |  Ranked: errp_ranked.csv\n")

    return report_path


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traverse XDF database for ErrP signals and rank files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default=str(getattr(config, "DATA_DIR", "")),
        help="Root directory to scan for XDF files (default: config.DATA_DIR)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path.home() / "Documents" / "errp_exploration"),
        help="Output directory for reports (default: ~/Documents/errp_exploration/)",
    )
    parser.add_argument(
        "--n-splits", type=int, default=5,
        help="K-fold splits for xDAWN+MDM evaluation (default: 5)",
    )
    parser.add_argument(
        "--n-filters", type=int,
        default=int(getattr(config, "ERRP_XDAWN_N_FILTERS", 4)),
        help="xDAWN spatial filters (default: config.ERRP_XDAWN_N_FILTERS = 4)",
    )
    parser.add_argument(
        "--min-epochs", type=int, default=8,
        help="Minimum total ErrP epochs (error+correct) required to evaluate a file (default: 8)",
    )
    parser.add_argument(
        "--max-files", type=int, default=0,
        help="Cap number of files evaluated (0 = unlimited, for debugging)",
    )
    parser.add_argument(
        "--exclude", default="OBS,old",
        help="Comma-separated path substrings to exclude (default: OBS,old)",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=10,
        help="Print checkpoint stats every N files (default: 10)",
    )
    parser.add_argument(
        "--top-n", type=int, default=30,
        help="Number of top files to list in the text report (default: 30)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir  = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude_substrings = tuple(x.strip() for x in args.exclude.split(",") if x.strip())

    print("=" * 78)
    print("Harmony — ErrP dataset library exploration")
    print("=" * 78)
    print(f"  data_dir       : {data_dir}")
    print(f"  out_dir        : {out_dir}")
    print(f"  n_splits       : {args.n_splits}  n_filters : {args.n_filters}")
    print(f"  min_epochs     : {args.min_epochs}")
    print(f"  exclude        : {', '.join(exclude_substrings)}")
    if args.max_files > 0:
        print(f"  max_files      : {args.max_files}  (debug cap)")
    print("=" * 78)

    # --- Discovery ---
    all_paths = discover_xdf_files(data_dir, exclude_substrings=exclude_substrings)
    print(f"\n[Discovery] Found {len(all_paths)} XDF files under {data_dir}")
    dup = duplicate_groups_by_size(all_paths)
    if dup:
        n_dup = sum(len(g) - 1 for g in dup.values())
        print(f"  Hint: {len(dup)} size groups (~{n_dup} possible duplicates)")

    if args.max_files > 0:
        all_paths = all_paths[: args.max_files]
        print(f"  Capped at {args.max_files} files for this run.")

    # --- Error codes ---
    error_codes   = [int(config.TRIGGERS.get("ROBOT_EARLYSTOP", "340")),
                     int(config.TRIGGERS.get("ERRP_STIM_ERROR", "430"))]
    correct_codes = [int(config.TRIGGERS.get("ROBOT_END", "320")),
                     int(config.TRIGGERS.get("ERRP_STIM_CORRECT", "440"))]
    print(f"\n  Error   markers : {error_codes}")
    print(f"  Correct markers : {correct_codes}")

    # --- Scan ---
    n_total  = len(all_paths)
    rows: list[dict] = []
    n_with_errp = 0
    t_run0 = time.perf_counter()

    print(f"\n[Scan] Evaluating {n_total} files …\n")

    for idx, xdf_path in enumerate(all_paths, start=1):
        meta = index_xdf_path(xdf_path, data_dir=data_dir, baseline_subject="")
        t_file = time.perf_counter()

        fold_counter: list[int] = [0]
        def _on_fold(fi: int, nf: int, _fc: list = fold_counter) -> None:
            _fc[0] = fi
            print(f"    k-fold {fi}/{nf}", end="\r", flush=True)

        print(f"  → [{idx:>4}/{n_total}]  {meta.subject or '—':.<16}  {Path(xdf_path).name}", flush=True)

        row = evaluate_errp_path(
            xdf_path,
            n_splits=args.n_splits,
            n_filters=args.n_filters,
            min_epochs=args.min_epochs,
            on_fold=_on_fold if n_total > 1 else None,
        )
        row["subject"]        = meta.subject
        row["session_folder"] = meta.session_folder
        row["modality"]       = meta.modality
        row["size_bytes"]     = meta.size_bytes

        elapsed = time.perf_counter() - t_file
        ev = str(row.get("eval_error", "") or "")

        # Did this file have any ErrP markers at all?
        any_markers = (
            int(row.get("n_error_total", 0)) + int(row.get("n_correct_total", 0))
        ) > 0
        if any_markers:
            n_with_errp += 1

        if not ev.strip():
            auc_s = f"AUC={row.get('auc', float('nan')):.3f}" if pd.notna(row.get("auc", float("nan"))) else "AUC=—"
            cd_s  = f"d={row.get('ern_cohens_d', float('nan')):.3f}" if pd.notna(row.get("ern_cohens_d", float("nan"))) else "d=—"
            ne    = int(row.get("n_error_epochs", 0))
            nc    = int(row.get("n_correct_epochs", 0))
            print(
                f"    OK  {auc_s}  {cd_s}  "
                f"err={ne}  cor={nc}  ({elapsed:.1f}s)",
                flush=True,
            )
        elif ev.startswith("below_min_epochs"):
            ne  = int(row.get("n_error_total", 0))
            nc  = int(row.get("n_correct_total", 0))
            print(f"    LOW  skip — err_total={ne} cor_total={nc}  ({elapsed:.1f}s)", flush=True)
        else:
            short = ev[:70] + ("…" if len(ev) > 70 else "")
            print(f"    ERR  {short}  ({elapsed:.1f}s)", flush=True)

        rows.append(row)

        if idx % args.checkpoint_every == 0 or idx == n_total:
            ok_n  = sum(1 for r in rows if not str(r.get("eval_error") or "").strip())
            aucs  = [float(r["auc"]) for r in rows if "auc" in r and pd.notna(r.get("auc", float("nan")))]
            auc_m = f"{np.mean(aucs):.3f}" if aucs else "—"
            ela   = time.perf_counter() - t_run0
            eta   = (ela / idx) * (n_total - idx) if idx > 0 else 0.0
            print(
                f"\n  — checkpoint {idx}/{n_total}: OK={ok_n}  with_errp_markers={n_with_errp}"
                f"  mean_auc={auc_m}  elapsed={ela/60:.1f}m  ETA~{eta/60:.1f}m\n",
                flush=True,
            )

    elapsed_total = time.perf_counter() - t_run0

    # --- Build DataFrame ---
    df = pd.DataFrame(rows)

    # Add composite score
    ref = _build_reference(df)
    df  = add_composite_score(df, ref)

    # Rank
    def _has_error(row: pd.Series) -> bool:
        ev = row.get("eval_error", "")
        return pd.notna(ev) and str(ev).strip() != ""

    ranked = df[~df.apply(_has_error, axis=1)].copy()
    ranked = ranked.sort_values("composite_score", ascending=False, na_position="last").reset_index(drop=True)
    ranked["proposal_rank"] = np.arange(1, len(ranked) + 1)

    # --- Write outputs ---
    candidates_path = out_dir / "errp_candidates.csv"
    ranked_path     = out_dir / "errp_ranked.csv"
    df.to_csv(candidates_path, index=False)
    ranked.to_csv(ranked_path, index=False)

    report_path = _write_text_report(
        out_dir, df, ranked,
        data_dir=data_dir,
        n_files_scanned=n_total,
        n_files_with_errp=n_with_errp,
        elapsed_s=elapsed_total,
        top_n=args.top_n,
    )

    print("\n" + "=" * 78)
    print("Scan complete.")
    print(f"  Total files scanned      : {n_total}")
    print(f"  Files with ErrP markers  : {n_with_errp}")
    print(f"  Successfully evaluated   : {len(ranked)}")
    print(f"  Elapsed                  : {elapsed_total:.1f}s")
    print(f"\nOutputs:")
    print(f"  {candidates_path}")
    print(f"  {ranked_path}")
    print(f"  {report_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
