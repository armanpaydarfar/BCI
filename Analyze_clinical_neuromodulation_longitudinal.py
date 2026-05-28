#!/usr/bin/env python3
"""Per-electrode mu-ERD% longitudinal trajectory (Plan A of
`rev01-longitudinal-analysis-plan.md` §3).

For each (subject, session), compute the per-trial mu-band ERD% over
the Contralateral cluster (primary) and Bilateral cluster (secondary)
at SCALAR_WIN = (1, 4) s, using Config A preprocessing.

Pass 2 (2026-05-28): LMEs now fit per-session response (median primary,
mean sensitivity), not per-trial, per `rev01-longitudinal-analysis-plan.md`
§3.3 / §3.4 and the pass-1 critic review §C1/§C2/§M6. Per-trial CSV is
retained for backward compatibility.

`--from-csv` reloads existing per-trial CSV produced by an earlier run
to skip the ~25 min Config-A TFR pass. Used in pass-2 to re-fit only the
LMEs and cohort plots without rerunning preprocessing.

Outputs (`~/Pictures/clin_analysis_pass1/neuromod/`):
    <SUBJ>_erd_over_sessions_contra.png    (per-subject, contra cluster)
    <SUBJ>_erd_over_sessions_bilat.png     (per-subject, bilateral)
    cohort_lme_erd_contra.png              (cohort)
    cohort_lme_erd_bilat.png               (cohort)
    erd_per_trial.csv                      (raw per-trial ERD% values)
    erd_session_summary.csv                (per-session mean/median/SE)
    erd_lme_results.txt                    (LME summaries / fallback)

Analysis-only. No Tier 1 / Tier 2 writes.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent
_SWEEP_DIR = _REPO_ROOT / "exploration" / "preprocessing_sweep"
for _p in (str(_REPO_ROOT), str(_SWEEP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exploration.clinical_analysis._helpers import (  # noqa: E402
    BILATERAL_MOTOR_CLUSTER, CONTRA_MOTOR_CLUSTER, clin_pictures_root,
    config_a_pipeline, enumerate_clin_subjects,
    enumerate_online_sessions_for_subject, session_idx_from_label,
)

# Mu band + scalar window constants
from sweep_phase2_round2 import MU_HI, MU_LO, SCALAR_WIN  # noqa: E402

import mne  # noqa: E402

mne.set_log_level("ERROR")

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
# Per-trial ERD%
# ----------------------------------------------------------------------

def _per_trial_cluster_erd_pct(
    tfr_trials, cluster_channels: list[str], marker: str = "200",
) -> tuple[np.ndarray | None, list[str]]:
    """One scalar ERD% per trial, averaged across (cluster channels,
    mu freqs, SCALAR_WIN). Trial-averaging happens in % space (one
    scalar per trial), so the per-trial summary is exactly:
        ERD%_trial = 100 * (10^mean_{c,f,t} logratio - 1)
    matching the longitudinal-plan §3.1 spec.

    Returns (array of shape (n_trials,), channels_actually_used).
    """
    if marker not in tfr_trials:
        return None, []
    tfr = tfr_trials[marker]
    present = [c for c in cluster_channels if c in tfr.ch_names]
    if not present:
        return None, []
    ch_idxs = [tfr.ch_names.index(c) for c in present]
    fmask = (tfr.freqs >= MU_LO) & (tfr.freqs <= MU_HI)
    tmask = (tfr.times >= SCALAR_WIN[0]) & (tfr.times <= SCALAR_WIN[1])
    # tfr.data: (trials, channels, freqs, times)
    per_trial_log = tfr.data[:, ch_idxs][:, :, fmask, :][:, :, :, tmask].mean(
        axis=(1, 2, 3),
    )
    per_trial_pct = 100.0 * (np.power(10.0, per_trial_log) - 1.0)
    return per_trial_pct, present


# ----------------------------------------------------------------------
# LME helper with Spearman fallback
# ----------------------------------------------------------------------

# Bonferroni alpha across the longitudinal pack:
# rev01-longitudinal-analysis-plan.md §5 lists ~6-8 primary LMEs
# (contra ERD, bilateral ERD, sample-κ, trial-κ, NKV, acc_decided,
# Lean_MI, Lean_REST). The conservative α' across the analysis pack is
# 0.05 / 8 ≈ 0.00625. Per the rev01-task-brief: "α' ≈ 0.006".
BONFERRONI_ALPHA_PRIMARY = 0.05 / 8


def _bonferroni_verdict(p: float, alpha: float = BONFERRONI_ALPHA_PRIMARY) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < alpha:
        return f"PASS (p<{alpha:.4f})"
    return f"FAIL (p>={alpha:.4f})"


def _fit_lme_or_fallback(
    df: pd.DataFrame, response: str, label: str,
) -> str:
    """Fit `response ~ 1 + session_idx + (1|subject)`. On failure,
    fall back to per-subject Spearman correlations of session_idx vs
    response and report a cohort summary (median ρ, n positive).

    Returns a multi-line string suitable for the .txt dump. Adds a
    'Bonferroni-corrected verdict' line keyed to the slope p-value at
    α' = 0.05 / 8 ≈ 0.006 (longitudinal-plan §5; pass-1 critic §M5).
    """
    lines = [f"=== {label} ==="]
    slope_p = np.nan
    if not HAS_STATSMODELS:
        lines.append("statsmodels not available; skipping LME.")
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Default optimizer (BFGS). The pass-1 `method="lbfgs"`
                # call was hitting singular-matrix errors on per-session
                # data (n=34, 7 groups); BFGS converges fine.
                model = smf.mixedlm(
                    f"{response} ~ 1 + session_idx", df,
                    groups=df["subject"],
                ).fit(disp=False)
            lines.append(str(model.summary()))
            try:
                slope_p = float(model.pvalues.get("session_idx", np.nan))
            except Exception:
                slope_p = np.nan
        except Exception as exc:
            lines.append(
                f"LME fit failed: {type(exc).__name__}: {exc}. "
                f"Falling back to per-subject Spearman."
            )
    lines.append(
        f"Bonferroni-corrected slope test (α'=0.05/8={BONFERRONI_ALPHA_PRIMARY:.4f}): "
        f"slope p={slope_p:.4g} → {_bonferroni_verdict(slope_p)}"
    )
    # Spearman per subject (always reported alongside)
    if HAS_SPEARMAN:
        rho_rows = []
        for subj, sub in df.groupby("subject"):
            if sub["session_idx"].nunique() < 3:
                rho_rows.append((subj, np.nan, np.nan, len(sub)))
                continue
            r, p = spearmanr(sub["session_idx"], sub[response])
            rho_rows.append((subj, float(r), float(p), len(sub)))
        lines.append("Per-subject Spearman ρ(session_idx, " + response + "):")
        for subj, r, p, n in rho_rows:
            r_str = "nan" if not np.isfinite(r) else f"{r:+.3f}"
            p_str = "nan" if not np.isfinite(p) else f"{p:.3g}"
            lines.append(f"  {subj}: ρ={r_str}, p={p_str}, n={n}")
        finite = [r for _, r, _, _ in rho_rows if np.isfinite(r)]
        if finite:
            lines.append(
                f"  Cohort median ρ = {np.median(finite):+.3f}; "
                f"n positive = {sum(1 for r in finite if r > 0)}/{len(finite)}"
            )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def _plot_subject_session_trajectory(
    df_subj: pd.DataFrame, subject: str, cluster_key: str, out_path: Path,
):
    """Per-subject ERD% over sessions: scatter of per-trial values +
    per-session mean ± SE."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = df_subj["session_idx"].values
    ys = df_subj["erd_pct"].values
    ax.scatter(xs, ys, s=16, alpha=0.35, color="0.5", label="per trial")
    sess_grp = df_subj.groupby("session_idx")["erd_pct"].agg(
        ["mean", "sem", "count"]
    ).reset_index()
    ax.errorbar(
        sess_grp["session_idx"], sess_grp["mean"], yerr=sess_grp["sem"],
        fmt="o-", color="tab:red", lw=2, markersize=6,
        label="session mean ± SE",
    )
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Session index")
    ax.set_ylabel("ERD %")
    ax.set_title(
        f"{subject} — Mu ERD% across sessions ({cluster_key} cluster)"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_cohort_trajectory(
    df_sess: pd.DataFrame, cluster_key: str, out_path: Path,
    *, lme_annotation: str | None = None,
):
    """Cohort figure: one line per subject (per-session median, semi-
    transparent) + cohort median across subjects (bold).

    Pass 2 (critic §M6): switched response from per-trial mean to
    per-session median for both the per-subject thin lines and the cohort
    bold line. Per `rev01-longitudinal-analysis-plan.md` §3.3, median is
    robust to single-trial outliers (CLIN_SUBJ_007 had per-trial ERD%
    extremes up to +235%).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    subjects = sorted(df_sess["subject"].unique())
    cmap = plt.get_cmap("tab10")
    for i, subj in enumerate(subjects):
        sub = df_sess[df_sess.subject == subj].sort_values("session_idx")
        ax.plot(
            sub["session_idx"], sub["erd_median"], "o-",
            color=cmap(i % 10), alpha=0.5, lw=1.0, markersize=4,
            label=subj,
        )
    # Cohort grand median (across subjects, per session_idx)
    cohort = df_sess.groupby("session_idx")["erd_median"].agg(
        ["median", "mean", "sem", "count"],
    ).reset_index()
    ax.errorbar(
        cohort["session_idx"], cohort["median"], yerr=cohort["sem"],
        color="black", lw=2.5, marker="s", markersize=8,
        label="Cohort median (± SE of session medians)",
    )
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Session index")
    ax.set_ylabel("Per-session median ERD %")
    ax.set_title(
        f"CLIN cohort — Mu ERD% over sessions ({cluster_key} cluster)\n"
        f"(per-session medians; LME on medians)"
    )
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


def _fit_session_lme(
    df_sess: pd.DataFrame, response: str, label: str,
) -> tuple[str, str]:
    """Fit per-session LME and return (full text block, short annotation)
    suitable for in-figure text. `response` is one of erd_median / erd_mean.
    """
    text = _fit_lme_or_fallback(df_sess, response, label)
    # Extract slope + p from a fresh fit for the annotation
    annot = ""
    if HAS_STATSMODELS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = smf.mixedlm(
                    f"{response} ~ 1 + session_idx", df_sess,
                    groups=df_sess["subject"],
                ).fit(disp=False)
            b = float(m.params.get("session_idx", np.nan))
            p = float(m.pvalues.get("session_idx", np.nan))
            verdict = _bonferroni_verdict(p)
            annot = (
                f"LME ({response}): slope = {b:+.3f}%/session, "
                f"p = {p:.3g}\nBonferroni α' = {BONFERRONI_ALPHA_PRIMARY:.4f}: "
                f"{verdict}"
            )
        except Exception:
            annot = "LME fit failed (see results.txt)"
    return text, annot


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _compute_from_tfr(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Config-A TFR for every (subject, session) and return
    (df_trials, df_sess). Side-effects: writes per-trial CSV and session-
    summary CSV under `out_dir` (these are reused by `--from-csv`)."""
    rows = []
    session_summary = []
    for subject in enumerate_clin_subjects():
        sessions = enumerate_online_sessions_for_subject(subject)
        print(f"\n=== {subject} ({len(sessions)} sessions) ===")
        for sess in sessions:
            t0 = time.time()
            try:
                out = config_a_pipeline(subject, sess)
            except Exception as e:
                print(f"  {sess}: FAILED ({type(e).__name__}: {e})")
                continue
            sess_idx = session_idx_from_label(sess)
            for cluster_key, cluster in [
                ("contra", CONTRA_MOTOR_CLUSTER),
                ("bilat",  BILATERAL_MOTOR_CLUSTER),
            ]:
                per_trial, present = _per_trial_cluster_erd_pct(
                    out["tfr_trials"], cluster,
                )
                if per_trial is None or len(per_trial) == 0:
                    continue
                for v in per_trial:
                    rows.append({
                        "subject": subject, "session": sess,
                        "session_idx": sess_idx, "cluster": cluster_key,
                        "channels_used": ",".join(present),
                        "erd_pct": float(v),
                    })
                session_summary.append({
                    "subject": subject, "session": sess,
                    "session_idx": sess_idx, "cluster": cluster_key,
                    "channels_used": ",".join(present),
                    "n_trials": int(len(per_trial)),
                    "erd_mean": float(per_trial.mean()),
                    "erd_median": float(np.median(per_trial)),
                    "erd_se": float(
                        per_trial.std(ddof=1) / np.sqrt(len(per_trial))
                    ) if len(per_trial) > 1 else 0.0,
                })
            print(f"  {sess}: ({time.time()-t0:.1f}s)")
            del out
            gc.collect()
    df_trials = pd.DataFrame(rows)
    df_sess = pd.DataFrame(session_summary)
    df_trials.to_csv(out_dir / "erd_per_trial.csv", index=False)
    df_sess.to_csv(out_dir / "erd_session_summary.csv", index=False)
    return df_trials, df_sess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-csv", action="store_true",
        help=("Skip the TFR pass and load erd_per_trial.csv + "
              "erd_session_summary.csv produced by an earlier run. "
              "Used in pass-2 to refit LMEs without rerunning Config-A."),
    )
    args = parser.parse_args()

    out_dir = clin_pictures_root() / "neuromod"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        trials_csv = out_dir / "erd_per_trial.csv"
        sess_csv = out_dir / "erd_session_summary.csv"
        if not (trials_csv.exists() and sess_csv.exists()):
            print(
                f"--from-csv requested but CSVs not found in {out_dir}. "
                "Falling back to TFR pass."
            )
            df_trials, df_sess = _compute_from_tfr(out_dir)
        else:
            df_trials = pd.read_csv(trials_csv)
            df_sess = pd.read_csv(sess_csv)
            print(
                f"Loaded {len(df_trials)} per-trial rows + "
                f"{len(df_sess)} session-summary rows from CSV."
            )
    else:
        df_trials, df_sess = _compute_from_tfr(out_dir)

    lme_text_blocks = []
    for cluster_key in ("contra", "bilat"):
        sub_trials = df_trials[df_trials.cluster == cluster_key].copy()
        sub_sess = df_sess[df_sess.cluster == cluster_key].copy()
        if sub_trials.empty or sub_sess.empty:
            continue

        # Per-subject plots (per-trial scatter + per-session mean ± SE
        # — unchanged from pass 1; useful for visualising single-trial
        # variability).
        for subject in sorted(sub_trials["subject"].unique()):
            df_subj = sub_trials[sub_trials.subject == subject]
            out_png = out_dir / f"{subject}_erd_over_sessions_{cluster_key}.png"
            _plot_subject_session_trajectory(
                df_subj, subject, cluster_key, out_png,
            )

        # Pass-2: per-session LMEs (primary: median; sensitivity: mean).
        # Per `rev01-longitudinal-analysis-plan.md` §3.3 / §3.4 and
        # pass-1 critic §C1.
        text_med, annot_med = _fit_session_lme(
            sub_sess, "erd_median",
            f"ERD% ({cluster_key} cluster) — per-session LME on MEDIAN",
        )
        text_mean, _annot_mean = _fit_session_lme(
            sub_sess, "erd_mean",
            f"ERD% ({cluster_key} cluster) — per-session LME on MEAN "
            f"(sensitivity check, per `rev01-longitudinal-analysis-plan.md` §3.3)",
        )
        lme_text_blocks.extend([text_med, text_mean])

        # Cohort plot (per-session median; annotated with LME slope/p +
        # Bonferroni verdict). Replaces the pass-1 mean ± SE per-trial plot.
        cohort_png = out_dir / f"cohort_lme_erd_{cluster_key}.png"
        _plot_cohort_trajectory(
            sub_sess, cluster_key, cohort_png, lme_annotation=annot_med,
        )

    (out_dir / "erd_lme_results.txt").write_text(
        "\n\n".join(lme_text_blocks), encoding="utf-8",
    )
    print(f"\nDone. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
