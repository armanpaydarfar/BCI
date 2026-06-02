#!/usr/bin/env python3
"""Per-session decoder performance longitudinal (Plan B of
`rev01-longitudinal-analysis-plan.md` §4).

Per (subject, session), pool all `ONLINE_*/decoder_output.csv` and
`trial_summary.csv` rows and compute:
  - sample_kappa, sample_acc  (per-window; from decoder_output.csv,
    Phase != ROBOT, predicted = argmax(P(MI)_avg, P(REST)_avg))
  - trial_kappa, acc_decided, acc_inclusive  (per-trial; from
    trial_summary.csv-style aggregation of decoder_output.csv last-row
    per GlobalTrialID, mirrors
    Analyze_experiment_logs_cross_subject.compute_accuracy_by_class_from_csv_trials)
  - nkv = trial_kappa * (1 - n_amb / n_total)

Fit `metric ~ 1 + session_idx + (1|subject)` per metric.

Pass 2 (2026-05-28):
  - M2 Option B: CLIN_SUBJ_002 only has `P(MI)` / `P(REST)`
    (instantaneous) columns; for sample-level kappa to be apples-to-
    apples with CLIN_SUBJ_003..008's leaky-integrated `_avg` streams,
    CLIN_SUBJ_002's instantaneous probs are integrated offline with
    `alpha=0.95` (per rev01-paper-angle.md §1.1 / pass-1 critic §M2).
  - Bonferroni-corrected verdict appended per metric (critic §M5).
  - LME slope + Bonferroni verdict annotated on each cohort plot (Mi5).

Outputs (`~/Pictures/clin_analysis/decoder_perf/`):
    <SUBJ>_decoder_kappa_nkv_acc_over_sessions.png   (per-subject panel)
    cohort_lme_kappa.png    cohort_lme_nkv.png    cohort_lme_acc.png
    decoder_perf_session_summary.csv
    decoder_perf_lme_results.txt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

_REPO_ROOT = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exploration.clinical_analysis._helpers import (  # noqa: E402
    DATA_DIR, clin_pictures_root, enumerate_clin_subjects,
    enumerate_online_sessions_for_subject, session_idx_from_label,
)

# Reuse parsing helpers from existing cross-subject analyser
from Analyze_experiment_logs_cross_subject import (  # noqa: E402
    build_unified_prob_cols, compute_accuracy_by_class_from_csv_trials,
    find_decoder_csv,
)

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

try:
    from scipy.stats import spearmanr
    HAS_SPEARMAN = True
except Exception:
    HAS_SPEARMAN = False


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

# Bonferroni α' across the longitudinal pack (longitudinal-plan §5;
# ~8 primary LMEs across contra/bilat ERD, sample-κ, trial-κ, NKV,
# acc_decided, Lean_MI, Lean_REST).
BONFERRONI_ALPHA_PRIMARY = 0.05 / 8

# CLIN_SUBJ_002 used the older runtime with INTEGRATOR_ALPHA = 0.95 and
# only logs instantaneous P(MI)/P(REST) (rev01-paper-angle.md §1.1).
# This matches `LeakyIntegrator` at `Utils/experiment_utils.py:51-69`:
#   a = a * alpha + p * (1 - alpha),  init 0.5.
CLIN002_INTEGRATOR_ALPHA = 0.95


def _bonferroni_verdict(p: float, alpha: float = BONFERRONI_ALPHA_PRIMARY) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < alpha:
        return f"PASS (p<{alpha:.4f})"
    return f"FAIL (p>={alpha:.4f})"


def _leaky_integrate(p: np.ndarray, alpha: float) -> np.ndarray:
    """Offline leaky integration matching the runtime LeakyIntegrator.
    Initial state 0.5 (matches `LeakyIntegrator.__init__`).
    """
    out = np.empty_like(p, dtype=float)
    state = 0.5
    for i, v in enumerate(p):
        if not np.isfinite(v):
            out[i] = state
            continue
        state = alpha * state + (1.0 - alpha) * float(v)
        out[i] = state
    return out


def _ensure_avg_cols(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """If `P(MI)_avg` / `P(REST)_avg` are missing, compute them offline
    by leaky-integrating the instantaneous columns with `alpha`. Reset
    state per (RunID, Trial) to match the runtime
    (`Utils/runtime_common.py:528` constructs a fresh LeakyIntegrator
    per trial). M2 Option B fix for CLIN_SUBJ_002.
    """
    if "P(MI)_avg" in df.columns and "P(REST)_avg" in df.columns:
        return df
    if "P(MI)" not in df.columns or "P(REST)" not in df.columns:
        return df
    df = df.copy()
    df["P(MI)_avg"] = np.nan
    df["P(REST)_avg"] = np.nan
    group_keys = [k for k in ("RunID", "Trial") if k in df.columns]
    if not group_keys:
        df["P(MI)_avg"] = _leaky_integrate(df["P(MI)"].values, alpha)
        df["P(REST)_avg"] = _leaky_integrate(df["P(REST)"].values, alpha)
        return df
    for _key, idx in df.groupby(group_keys).groups.items():
        df.loc[idx, "P(MI)_avg"] = _leaky_integrate(
            df.loc[idx, "P(MI)"].values, alpha,
        )
        df.loc[idx, "P(REST)_avg"] = _leaky_integrate(
            df.loc[idx, "P(REST)"].values, alpha,
        )
    return df


# ----------------------------------------------------------------------
# Per-session metric extraction
# ----------------------------------------------------------------------

def _load_session_run_dfs(
    subject: str, session: str,
) -> list[pd.DataFrame]:
    """Return per-run decoder_output DataFrames for one session.

    Adds RunID + GlobalTrialID columns to match the convention used by
    `Analyze_experiment_logs_cross_subject` helpers.
    """
    logs_dir = Path(DATA_DIR) / f"sub-{subject}" / f"ses-{session}" / "logs"
    if not logs_dir.is_dir():
        return []
    out: list[pd.DataFrame] = []
    for run_dir in sorted(p for p in logs_dir.iterdir() if p.is_dir()
                           and p.name.startswith("ONLINE_")):
        csv = find_decoder_csv(str(run_dir))
        if csv is None:
            continue
        df = pd.read_csv(csv)
        if "Trial" not in df.columns:
            continue
        df["SubjectID"] = subject
        df["SessionID"] = session
        df["RunFolder"] = run_dir.name
        df["RunID"] = f"{subject}__{session}__{run_dir.name}"
        df["GlobalTrialID"] = (
            df["RunID"].astype(str) + "_" + df["Trial"].astype(str)
        )
        df["__source_csv"] = csv
        # M2 Option B: integrate instantaneous → leaky-integrated for
        # CLIN_SUBJ_002 (old schema only). No-op for new-schema subjects.
        if subject == "CLIN_SUBJ_002":
            df = _ensure_avg_cols(df, CLIN002_INTEGRATOR_ALPHA)
        out.append(df)
    return out


def _session_metrics(run_dfs: list[pd.DataFrame]) -> dict:
    """Compute the six metrics for one session by pooling all its runs."""
    if not run_dfs:
        return {}
    pooled = pd.concat(run_dfs, ignore_index=True)
    pooled, MI_COL, REST_COL = build_unified_prob_cols(pooled)
    pooled["__avg_MI"] = (
        pooled["P(MI)_avg"]
        if "P(MI)_avg" in pooled.columns
        else pooled.get("P(MI)")
    )
    pooled["__avg_REST"] = (
        pooled["P(REST)_avg"]
        if "P(REST)_avg" in pooled.columns
        else pooled.get("P(REST)")
    )

    # ----- Trial-level metrics via the existing helper -----
    acc_dict = compute_accuracy_by_class_from_csv_trials(pooled)
    trials_mi = acc_dict.get("Trials_MI", 0)
    trials_re = acc_dict.get("Trials_REST", 0)
    amb_mi = acc_dict.get("Ambiguous_MI", 0)
    amb_re = acc_dict.get("Ambiguous_REST", 0)
    dec_mi = acc_dict.get("Decisions_MI", 0)
    dec_re = acc_dict.get("Decisions_REST", 0)

    n_total = trials_mi + trials_re
    n_decided = dec_mi + dec_re
    n_ambig = amb_mi + amb_re

    # Trial-level kappa = cohen_kappa on the decided trials
    sample_df = pooled[pooled["Phase"] != "ROBOT"].copy() if "Phase" in pooled.columns else pooled.copy()
    if "Timestamp" in sample_df.columns:
        sample_df = sample_df.sort_values("Timestamp")
    last = sample_df.groupby("GlobalTrialID", as_index=False).tail(1)
    true_num = pd.to_numeric(last["True Label"], errors="coerce")
    pred_num = pd.to_numeric(last["Predicted Label"], errors="coerce")
    decided_mask = pred_num.isin([100, 200])
    y_true = true_num[decided_mask].astype(int).values
    y_pred = pred_num[decided_mask].astype(int).values
    if len(y_true) >= 1 and len(np.unique(y_true)) >= 2:
        trial_kappa = float(
            cohen_kappa_score(y_true, y_pred, labels=[100, 200])
        )
    else:
        trial_kappa = np.nan
    acc_decided = (
        float((y_true == y_pred).mean()) if len(y_true) else np.nan
    )
    # acc_inclusive: count ambiguous as failures.
    if n_total > 0:
        correct_n = int((y_true == y_pred).sum())
        acc_inclusive = correct_n / n_total
    else:
        acc_inclusive = np.nan

    nkv = (
        trial_kappa * (1.0 - n_ambig / n_total)
        if (not np.isnan(trial_kappa) and n_total > 0)
        else np.nan
    )

    # ----- Sample-level metrics -----
    sample_df_kep = pooled[pooled["Phase"] != "ROBOT"].copy() if "Phase" in pooled.columns else pooled.copy()
    sample_df_kep = sample_df_kep.dropna(subset=["__avg_MI", "__avg_REST"])
    if len(sample_df_kep):
        p_mi = sample_df_kep["__avg_MI"].astype(float).values
        p_rest = sample_df_kep["__avg_REST"].astype(float).values
        sample_pred = np.where(p_mi >= p_rest, 200, 100)
        sample_true = pd.to_numeric(
            sample_df_kep["True Label"], errors="coerce",
        ).astype("Int64")
        keep = sample_true.notna()
        # Mi1 fix: explicit `.to_numpy()` so the boolean Series indexes
        # the numpy `sample_pred` array cleanly (pandas mixed indexing
        # is fragile across versions).
        keep_arr = keep.to_numpy()
        sample_true_arr = sample_true[keep].astype(int).values
        sample_pred_arr = sample_pred[keep_arr]
        if (len(sample_true_arr) > 0 and
                len(np.unique(sample_true_arr)) >= 2):
            sample_kappa = float(
                cohen_kappa_score(
                    sample_true_arr, sample_pred_arr, labels=[100, 200],
                )
            )
        else:
            sample_kappa = np.nan
        sample_acc = (
            float((sample_true_arr == sample_pred_arr).mean())
            if len(sample_true_arr) else np.nan
        )
    else:
        sample_kappa = np.nan
        sample_acc = np.nan

    return dict(
        n_total=int(n_total), n_decided=int(n_decided),
        n_ambig=int(n_ambig),
        trial_kappa=trial_kappa, acc_decided=acc_decided,
        acc_inclusive=acc_inclusive, nkv=nkv,
        sample_kappa=sample_kappa, sample_acc=sample_acc,
    )


# ----------------------------------------------------------------------
# LME with Spearman fallback
# ----------------------------------------------------------------------

def _fit_lme(
    df: pd.DataFrame, metric: str, label: str,
) -> tuple[str, float, float]:
    """Fit per-session LME. Returns (text block, slope, slope_p)."""
    lines = [f"=== {label} ==="]
    df_use = df.dropna(subset=[metric, "session_idx", "subject"]).copy()
    slope = np.nan
    slope_p = np.nan
    if len(df_use) < 5:
        lines.append(f"  Not enough rows ({len(df_use)}); skipping.")
        return "\n".join(lines), slope, slope_p
    if HAS_STATSMODELS:
        import warnings as _w
        try:
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                # Default BFGS optimizer; pass-1 `method="lbfgs"` was
                # causing singular-matrix failures on small per-session
                # samples (n=34 across 7 groups).
                model = smf.mixedlm(
                    f"{metric} ~ 1 + session_idx", df_use,
                    groups=df_use["subject"],
                ).fit(disp=False)
            lines.append(str(model.summary()))
            try:
                slope = float(model.params.get("session_idx", np.nan))
                slope_p = float(model.pvalues.get("session_idx", np.nan))
            except Exception:
                pass
        except Exception as exc:
            lines.append(
                f"  LME failed: {type(exc).__name__}: {exc}. "
                f"Spearman fallback below."
            )
    else:
        lines.append("  statsmodels missing; Spearman only.")
    lines.append(
        f"  Bonferroni-corrected slope test (α'=0.05/8={BONFERRONI_ALPHA_PRIMARY:.4f}): "
        f"slope p={slope_p:.4g} → {_bonferroni_verdict(slope_p)}"
    )
    if HAS_SPEARMAN:
        rho_rows = []
        for subj, sub in df_use.groupby("subject"):
            if sub["session_idx"].nunique() < 3:
                continue
            r, p = spearmanr(sub["session_idx"], sub[metric])
            rho_rows.append((subj, float(r), float(p), len(sub)))
        if rho_rows:
            lines.append(f"  Per-subject Spearman ρ({metric}):")
            for subj, r, p, n in rho_rows:
                lines.append(
                    f"    {subj}: ρ={r:+.3f}, p={p:.3g}, n={n}"
                )
            finite = [r for _, r, _, _ in rho_rows]
            lines.append(
                f"  Cohort median ρ = {np.median(finite):+.3f}; "
                f"n positive = {sum(1 for r in finite if r > 0)}/{len(finite)}"
            )
    return "\n".join(lines), slope, slope_p


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def _plot_subject_panel(df_subj: pd.DataFrame, subject: str, out_path: Path):
    """Per-subject 3-panel: trial_kappa, nkv, acc_decided over sessions."""
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    metrics = [
        (axes[0], "trial_kappa", "Trial-level Cohen's κ"),
        (axes[1], "nkv",         "NKV = κ × (1 − n_amb/n_total)"),
        (axes[2], "acc_decided", "Accuracy (decided trials)"),
    ]
    for ax, m, title in metrics:
        sub = df_subj.dropna(subset=[m])
        if sub.empty:
            ax.set_title(f"{title} (no data)")
            continue
        ax.plot(
            sub["session_idx"], sub[m], "o-", color="tab:blue", lw=2,
        )
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Session index")
    fig.suptitle(f"{subject} — decoder performance over sessions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_cohort_metric(
    df: pd.DataFrame, metric: str, label: str, out_path: Path,
    *, lme_annotation: str | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    sub = df.dropna(subset=[metric]).copy()
    for i, subj in enumerate(sorted(sub["subject"].unique())):
        ds = sub[sub.subject == subj]
        ax.plot(
            ds["session_idx"], ds[metric], "o-",
            color=cmap(i % 10), alpha=0.5, lw=1.0, markersize=4,
            label=subj,
        )
    cohort = sub.groupby("session_idx")[metric].agg(["mean", "sem"]).reset_index()
    ax.errorbar(
        cohort["session_idx"], cohort["mean"], yerr=cohort["sem"],
        color="black", lw=2.5, marker="s", markersize=8,
        label="Cohort mean ± SE",
    )
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Session index")
    ax.set_ylabel(label)
    ax.set_title(f"CLIN cohort — {label} over sessions")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncol=2)
    if lme_annotation:
        ax.text(
            0.02, 0.02, lme_annotation, transform=ax.transAxes,
            fontsize=8, va="bottom", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    out_dir = clin_pictures_root() / "decoder_perf"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for subject in enumerate_clin_subjects():
        sessions = enumerate_online_sessions_for_subject(subject)
        print(f"\n=== {subject} ({len(sessions)} sessions) ===")
        for sess in sessions:
            t0 = time.time()
            run_dfs = _load_session_run_dfs(subject, sess)
            if not run_dfs:
                print(f"  {sess}: no usable runs; skip")
                continue
            try:
                m = _session_metrics(run_dfs)
            except Exception as e:
                print(f"  {sess}: metric compute FAILED: {e}")
                continue
            if not m:
                continue
            rows.append({
                "subject": subject, "session": sess,
                "session_idx": session_idx_from_label(sess),
                "n_runs": len(run_dfs), **m,
            })
            print(
                f"  {sess}: trials={m['n_total']} "
                f"kappa={m['trial_kappa']:.3f} nkv={m['nkv']:.3f} "
                f"acc_dec={m['acc_decided']:.3f} "
                f"sample_kappa={m['sample_kappa']:.3f} "
                f"({time.time()-t0:.1f}s)"
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "decoder_perf_session_summary.csv", index=False)

    # Per-subject panels
    for subject in sorted(df["subject"].unique()):
        _plot_subject_panel(
            df[df.subject == subject], subject,
            out_dir / f"{subject}_decoder_kappa_nkv_acc_over_sessions.png",
        )

    # LME — fit per metric, capture slope + p for plot annotation (Mi5).
    blocks = []
    annotations: dict[str, str] = {}
    for metric, label in [
        ("trial_kappa", "Trial-level Cohen's κ"),
        ("nkv", "NKV"),
        ("acc_decided", "Accuracy (decided)"),
        ("sample_kappa", "Sample-level Cohen's κ"),
    ]:
        block, slope, slope_p = _fit_lme(df, metric, label)
        blocks.append(block)
        annotations[metric] = (
            f"LME slope = {slope:+.3f}/session, p = {slope_p:.3g}\n"
            f"Bonferroni α' = {BONFERRONI_ALPHA_PRIMARY:.4f}: "
            f"{_bonferroni_verdict(slope_p)}"
        )

    # Cohort panels (Mi5: annotated with LME slope/p + Bonferroni verdict)
    _plot_cohort_metric(
        df, "trial_kappa", "Trial-level Cohen's κ",
        out_dir / "cohort_lme_kappa.png",
        lme_annotation=annotations["trial_kappa"],
    )
    _plot_cohort_metric(
        df, "nkv", "NKV",
        out_dir / "cohort_lme_nkv.png",
        lme_annotation=annotations["nkv"],
    )
    _plot_cohort_metric(
        df, "acc_decided", "Accuracy (decided)",
        out_dir / "cohort_lme_acc.png",
        lme_annotation=annotations["acc_decided"],
    )

    (out_dir / "decoder_perf_lme_results.txt").write_text(
        "\n\n".join(blocks), encoding="utf-8",
    )
    print(f"\nDone. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
