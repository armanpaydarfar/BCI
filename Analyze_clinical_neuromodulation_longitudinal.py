#!/usr/bin/env python3
"""Per-electrode mu-ERD% longitudinal trajectory (Plan A of
`rev01-longitudinal-analysis-plan.md` §3).

For each (subject, session, cluster), the per-trial mu-band ERD% scalar
is read from the canonical per-trial substrate produced by
`Analyze_clinical_erd_refined.py` (the `<SUBJ>_<SESS>_car.npz` files under
`erd_refined/per_trial/`). This is the SAME substrate the 6-panel ERD
timecourse figures are drawn from — cap=200 cluster-matched trial
rejection, CAR, display-matched baseline — so the longitudinal LME and the
timecourse figures cannot diverge. Run `Analyze_clinical_erd_refined.py`
first to (re)generate that substrate.

All averaging happens in LOG (logratio) space and is converted to % only at
the end — % is asymmetric (ERD floored at -100%, ERS unbounded) so an
arithmetic % mean is dominated by ERS and washes out real ERD (see
`Analyze_clinical_erd_refined._per_trial_cluster_logratio`).

Session-wise ERD definition:
  1. Per trial: ERD%_trial = back-convert the canonical per-trial ERD%(t)
     trace (`<cluster>_mi__ptp`, already log-averaged over ch+freq) to
     logratio, take the mean over SCALAR_WIN = (1, 4) s, convert to %.
  2. Per session: MEAN over trials taken in LOG space, then -> % (primary;
     matches the canonical timecourse). Median of the per-trial ERD% is kept
     as a robustness sensitivity check.
  3. Per cohort: mean across subjects of per-session means.

`--from-csv` reloads the per-trial / session-summary CSVs written by an
earlier run to re-fit only the LMEs and cohort plots without re-reading
the npz.

Outputs (`~/Pictures/clin_analysis/neuromod/`):
    <SUBJ>_erd_over_sessions_contra.png    (per-subject, contra cluster)
    <SUBJ>_erd_over_sessions_ipsi.png      (per-subject, ipsi cluster)
    <SUBJ>_erd_over_sessions_bilat.png     (per-subject, bilateral)
    cohort_lme_erd_contra.png              (cohort)
    cohort_lme_erd_ipsi.png                (cohort)
    cohort_lme_erd_bilat.png               (cohort)
    erd_per_trial.csv                      (raw per-trial ERD% values)
    erd_session_summary.csv                (per-session mean/median/SE)
    erd_lme_results.txt                    (LME summaries / fallback)

Analysis-only. No Tier 1 / Tier 2 writes.
"""

from __future__ import annotations

import argparse
import sys
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
    clin_pictures_root, session_idx_from_label,
)

# Scalar ERD integration window (post-cue, (1, 4) s), shared with the
# canonical ERD pipeline so the longitudinal scalar matches the timecourse.
from sweep_phase2_round2 import SCALAR_WIN  # noqa: E402

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
    """Cohort figure: one line per subject (per-session mean, semi-
    transparent) + cohort mean across subjects (bold).

    The per-session response is the arithmetic MEAN across trials, matching
    the canonical ERD aggregator (`canonical-update-2026-06-03.md` §3). The
    cap=200 cluster-matched rejection upstream removes the explosive ERS
    trials that previously motivated a median, so the mean of the cleaned
    pool is the appropriate central tendency.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    subjects = sorted(df_sess["subject"].unique())
    cmap = plt.get_cmap("tab10")
    for i, subj in enumerate(subjects):
        sub = df_sess[df_sess.subject == subj].sort_values("session_idx")
        ax.plot(
            sub["session_idx"], sub["erd_mean"], "o-",
            color=cmap(i % 10), alpha=0.5, lw=1.0, markersize=4,
            label=subj,
        )
    # Cohort grand mean (across subjects, per session_idx)
    cohort = df_sess.groupby("session_idx")["erd_mean"].agg(
        ["median", "mean", "sem", "count"],
    ).reset_index()
    ax.errorbar(
        cohort["session_idx"], cohort["mean"], yerr=cohort["sem"],
        color="black", lw=2.5, marker="s", markersize=8,
        label="Cohort mean (± SE of session means)",
    )
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Session index")
    ax.set_ylabel("Per-session mean ERD %")
    ax.set_title(
        f"CLIN cohort — Mu ERD% over sessions ({cluster_key} cluster)\n"
        f"(per-session means; LME on means)"
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

# Canonical per-trial substrate produced by `Analyze_clinical_erd_refined.py`.
# Each `<SUBJ>_<SESS>_car.npz` stores, per `<cluster>_<marker>` key, a
# `__ptp` matrix of shape (n_trials, n_time) holding the per-trial cluster-
# mean ERD%(t) trace AFTER cap=200 cluster-matched rejection, plus a
# `__times` axis (`Analyze_clinical_erd_refined.py:475-515`). Sourcing F4
# from this substrate guarantees the longitudinal LME and the 6-panel
# timecourse figures are computed on identical trials/aggregation.
_ERD_PER_TRIAL_DIR = clin_pictures_root() / "erd_refined" / "per_trial"


def _compute_from_canonical_npz(
    out_dir: Path, npz_dir: Path = _ERD_PER_TRIAL_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (df_trials, df_sess) from the canonical erd_refined per-trial
    npz files, not from an independent TFR pass.

    For each session npz and each cluster (contra/ipsi/bilat), the MI
    per-trial scalar ERD% is the mean of the cap=200-rejected
    `<cluster>_mi__ptp` trace over SCALAR_WIN = (1, 4) s. Per-session
    summaries carry both `erd_mean` (primary; matches the canonical mean-
    across-trials aggregator) and `erd_median` (sensitivity).

    Side-effects: writes per-trial + session-summary CSVs under `out_dir`
    (reused by `--from-csv`). Raises if the canonical npz dir is missing —
    run `Analyze_clinical_erd_refined.py` first.
    """
    if not npz_dir.is_dir():
        raise FileNotFoundError(
            f"Canonical per-trial npz dir not found: {npz_dir}. Run "
            f"Analyze_clinical_erd_refined.py before this script."
        )
    rows = []
    session_summary = []
    for npz_path in sorted(npz_dir.glob("*_car.npz")):
        d = np.load(npz_path, allow_pickle=True)
        subject = str(d["subject"])
        sess = str(d["session"])
        sess_idx = session_idx_from_label(sess)
        for cluster_key in ("contra", "ipsi", "bilat"):
            ptp_key = f"{cluster_key}_mi__ptp"
            if ptp_key not in d:
                continue
            ptp = d[ptp_key]                      # (n_trials, n_time) ERD%
            times = d[f"{cluster_key}_mi__times"]
            present = str(d[f"{cluster_key}_mi__channels"])
            if ptp.ndim != 2 or ptp.shape[0] == 0:
                continue
            tmask = (times >= SCALAR_WIN[0]) & (times <= SCALAR_WIN[1])
            # Average over the window AND across trials in LOG space (the npz
            # substrate is already log-averaged across ch+freq); convert to %
            # only at the end. Matches the canonical timecourse so the
            # longitudinal scalar and the 6-panel figures agree.
            ptp_log = np.log10(1.0 + ptp / 100.0)        # % -> logratio
            per_trial_log = ptp_log[:, tmask].mean(axis=1)   # per-trial window mean
            per_trial = 100.0 * (10.0 ** per_trial_log - 1.0)  # per-trial ERD%
            for v in per_trial:
                rows.append({
                    "subject": subject, "session": sess,
                    "session_idx": sess_idx, "cluster": cluster_key,
                    "channels_used": present, "erd_pct": float(v),
                })
            erd_mean = 100.0 * (10.0 ** per_trial_log.mean() - 1.0)
            session_summary.append({
                "subject": subject, "session": sess,
                "session_idx": sess_idx, "cluster": cluster_key,
                "channels_used": present,
                "n_trials": int(len(per_trial)),
                "erd_mean": float(erd_mean),             # log-space trial mean -> %
                "erd_median": float(np.median(per_trial)),
                "erd_se": float(
                    per_trial.std(ddof=1) / np.sqrt(len(per_trial))
                ) if len(per_trial) > 1 else 0.0,
            })
        print(f"  {subject} {sess}: {len(session_summary)} cluster rows so far")
    df_trials = pd.DataFrame(rows)
    df_sess = pd.DataFrame(session_summary)
    df_trials.to_csv(out_dir / "erd_per_trial.csv", index=False)
    df_sess.to_csv(out_dir / "erd_session_summary.csv", index=False)
    return df_trials, df_sess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-csv", action="store_true",
        help=("Skip the npz pass and load erd_per_trial.csv + "
              "erd_session_summary.csv produced by an earlier run. "
              "Used to refit LMEs / re-plot without re-reading the npz."),
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
                "Falling back to canonical-npz pass."
            )
            df_trials, df_sess = _compute_from_canonical_npz(out_dir)
        else:
            df_trials = pd.read_csv(trials_csv)
            df_sess = pd.read_csv(sess_csv)
            print(
                f"Loaded {len(df_trials)} per-trial rows + "
                f"{len(df_sess)} session-summary rows from CSV."
            )
    else:
        df_trials, df_sess = _compute_from_canonical_npz(out_dir)

    lme_text_blocks = []
    for cluster_key in ("contra", "ipsi", "bilat"):
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

        # Per-session LMEs. Primary = MEAN across trials, matching the
        # canonical ERD aggregator (`canonical-update-2026-06-03.md` §3:
        # cap=200 rejection drops the explosive ERS trials that historically
        # motivated the median, so mean is now the appropriate central
        # tendency on the cleaned pool). Median is retained as a sensitivity
        # check.
        text_mean, annot_mean = _fit_session_lme(
            sub_sess, "erd_mean",
            f"ERD% ({cluster_key} cluster) — per-session LME on MEAN",
        )
        text_med, _annot_med = _fit_session_lme(
            sub_sess, "erd_median",
            f"ERD% ({cluster_key} cluster) — per-session LME on MEDIAN "
            f"(sensitivity check)",
        )
        lme_text_blocks.extend([text_mean, text_med])

        # Cohort plot (per-session mean; annotated with LME slope/p +
        # Bonferroni verdict).
        cohort_png = out_dir / f"cohort_lme_erd_{cluster_key}.png"
        _plot_cohort_trajectory(
            sub_sess, cluster_key, cohort_png, lme_annotation=annot_mean,
        )

    (out_dir / "erd_lme_results.txt").write_text(
        "\n\n".join(lme_text_blocks), encoding="utf-8",
    )
    print(f"\nDone. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
