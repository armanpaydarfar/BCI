#!/usr/bin/env python3
"""Bar dynamics longitudinal (Plan C of
`rev01-longitudinal-analysis-plan.md` §4.5).

Per-trial summaries:
  - Lean% (`% samples with P(correct_class) > THRESH`) — reuses
    Analyze_experiment_logs_cross_subject.compute_lean16hz_per_trial.
    Per Arman 2026-05-28 (rejecting critic §C3): Lean% remains on the
    **instantaneous** P(MI) / P(REST) stream because the instantaneous
    probability determines bar direction-of-motion and so is more
    representative of bar dynamics than the smoothed integrator state.
  - Time-to-threshold: first time within trial that `P(correct)_avg`
    exceeds 0.6 (config.THRESHOLD); censored at trial end if never
    crossed.
  - Within-trial slope: linear regression of `P(correct)_avg` vs trial
    time (units: prob per second).

Pass 2 (2026-05-28):
  - LMEs fit per-session (response = MEDIAN, with MEAN sensitivity),
    not per-trial — critic §C1/§C2.
  - M2 Option B: CLIN_SUBJ_002 only has `P(MI)` / `P(REST)`
    (instantaneous) columns. To compare TTT and Slope apples-to-apples
    with CLIN_SUBJ_003..008 (whose values are computed on the leaky-
    integrated `_avg` stream), the script integrates CLIN_SUBJ_002's
    instantaneous probabilities offline with `alpha=0.95`
    (config_snapshot for CLIN_SUBJ_002, per rev01-paper-angle.md §1.1)
    before computing TTT/Slope. Lean% is on the instantaneous stream
    for all subjects (see above) — uniform by construction.

Outputs (`~/Pictures/clin_analysis_pass1/bar_dynamics/`):
    <SUBJ>_bar_dynamics_3metric_over_sessions.png    (per subject)
    cohort_lme_lean.png    cohort_lme_ttt.png    cohort_lme_slope.png
    bar_dynamics_per_trial.csv      bar_dynamics_session_summary.csv
    bar_dynamics_lme_results.txt
"""

from __future__ import annotations

import sys
import time
import warnings as _warnings_mod
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exploration.clinical_analysis._helpers import (  # noqa: E402
    DATA_DIR, clin_pictures_root, enumerate_clin_subjects,
    enumerate_online_sessions_for_subject, session_idx_from_label,
)

from Analyze_experiment_logs_cross_subject import (  # noqa: E402
    build_unified_prob_cols, compute_lean16hz_per_trial, find_decoder_csv,
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


# Constants (per rev01-longitudinal-analysis-plan.md §4.5)
LEAN_THRESHOLD = 0.50
TTT_THRESHOLD = 0.60
CLASSIFIER_HZ = 16.0

# Bonferroni α' across the longitudinal pack (longitudinal-plan §5).
BONFERRONI_ALPHA_PRIMARY = 0.05 / 8

# CLIN_SUBJ_002 used the older runtime with INTEGRATOR_ALPHA = 0.95 and
# only logs instantaneous P(MI)/P(REST) (rev01-paper-angle.md §1.1).
# Replicates LeakyIntegrator.update at
# `Utils/experiment_utils.py:51-69`: a = a * alpha + p * (1 - alpha),
# initial state a = 0.5.
CLIN002_INTEGRATOR_ALPHA = 0.95


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
    by leaky-integrating the instantaneous columns with `alpha`. Done
    per (RunID, Trial) so integrator state resets at trial boundaries —
    matches the runtime, which constructs a fresh `LeakyIntegrator` per
    trial (`Utils/runtime_common.py:528` resets at trial start).
    """
    if "P(MI)_avg" in df.columns and "P(REST)_avg" in df.columns:
        return df
    if "P(MI)" not in df.columns or "P(REST)" not in df.columns:
        # Nothing to integrate; the upstream `build_unified_prob_cols`
        # will raise if neither schema is present.
        return df
    df = df.copy()
    df["P(MI)_avg"] = np.nan
    df["P(REST)_avg"] = np.nan
    group_keys = [k for k in ("RunID", "Trial") if k in df.columns]
    if not group_keys:
        df["P(MI)_avg"] = _leaky_integrate(df["P(MI)"].values, alpha)
        df["P(REST)_avg"] = _leaky_integrate(df["P(REST)"].values, alpha)
        return df
    # Apply per (RunID, Trial) to match runtime per-trial integrator reset
    for _key, idx in df.groupby(group_keys).groups.items():
        df.loc[idx, "P(MI)_avg"] = _leaky_integrate(
            df.loc[idx, "P(MI)"].values, alpha,
        )
        df.loc[idx, "P(REST)_avg"] = _leaky_integrate(
            df.loc[idx, "P(REST)"].values, alpha,
        )
    return df


def _bonferroni_verdict(p: float, alpha: float = BONFERRONI_ALPHA_PRIMARY) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < alpha:
        return f"PASS (p<{alpha:.4f})"
    return f"FAIL (p>={alpha:.4f})"


# ----------------------------------------------------------------------
# Per-trial bar dynamics metrics from one decoder_output.csv
# ----------------------------------------------------------------------

def _per_trial_bar_dynamics(df_run: pd.DataFrame) -> pd.DataFrame:
    """Compute (Lean%, time-to-threshold, slope) for each trial in one
    run's decoder_output.csv.

    Returns DataFrame with columns:
      GlobalTrialID, Class, LeanPct, TimeToThresh_s (NaN if not crossed),
      Slope_per_s.
    """
    df_run = df_run[df_run["Phase"] != "ROBOT"].copy()
    df_run, MI_COL, REST_COL = build_unified_prob_cols(df_run)

    # Per-trial Lean% via the existing helper. Empty result (e.g. for
    # runs whose CSV is header-only or filtered down to zero rows by
    # Phase != ROBOT) collapses to an empty DataFrame missing
    # GlobalTrialID — handle that explicitly.
    lean = compute_lean16hz_per_trial(
        df_run, MI_COL, REST_COL, thresh=LEAN_THRESHOLD,
    )
    if lean.empty:
        lean = pd.DataFrame(columns=["GlobalTrialID", "Class", "LeanPct"])
    lean = lean.set_index("GlobalTrialID")

    # Per-trial TTT + slope on the leaky-integrated correct-class prob.
    # Schema: the runtime's bar is driven by P(MI)_avg / P(REST)_avg —
    # the leaky-integrator output (rev01-longitudinal-analysis-plan.md §4.5).
    # The caller (`_load_session_trials`) is expected to have applied
    # `_ensure_avg_cols` upstream for the CLIN_SUBJ_002 old-schema branch
    # (M2 Option B), so `_avg` columns are present for all subjects.
    avg_mi_col = "P(MI)_avg" if "P(MI)_avg" in df_run.columns else MI_COL
    avg_re_col = (
        "P(REST)_avg" if "P(REST)_avg" in df_run.columns else REST_COL
    )

    out_rows = []
    for gtid, tdf in df_run.groupby("GlobalTrialID"):
        if tdf.empty:
            continue
        true_label = int(tdf["True Label"].iloc[0])
        if true_label == 200:
            p = tdf[avg_mi_col].astype(float).values
            cls = "MI"
        elif true_label == 100:
            p = tdf[avg_re_col].astype(float).values
            cls = "REST"
        else:
            continue
        # Trial time relative to first sample
        if "Timestamp" in tdf.columns and np.issubdtype(
            tdf["Timestamp"].dtype, np.number,
        ):
            t = tdf["Timestamp"].values.astype(float)
            t = t - t[0]
        else:
            t = np.arange(len(p)) / CLASSIFIER_HZ
        mask = np.isfinite(p) & np.isfinite(t)
        p_clean = p[mask]
        t_clean = t[mask]
        if len(p_clean) == 0:
            continue
        crossing = np.where(p_clean > TTT_THRESHOLD)[0]
        if len(crossing) > 0:
            ttt = float(t_clean[crossing[0]])
        else:
            ttt = np.nan
        if len(p_clean) >= 2 and (t_clean[-1] - t_clean[0]) > 0:
            slope = float(np.polyfit(t_clean, p_clean, 1)[0])
        else:
            slope = np.nan
        lean_pct = (
            float(lean.loc[gtid, "LeanPct"])
            if gtid in lean.index else np.nan
        )
        out_rows.append({
            "GlobalTrialID": gtid, "Class": cls,
            "LeanPct": lean_pct, "TimeToThresh_s": ttt, "Slope_per_s": slope,
        })
    return pd.DataFrame(out_rows)


def _load_session_trials(
    subject: str, session: str,
) -> pd.DataFrame:
    """Return all per-trial bar-dynamics rows for a session.

    For CLIN_SUBJ_002 (old schema, instantaneous-only), this is where
    the offline leaky-integration is applied so the downstream TTT/Slope
    computation reads `P(MI)_avg` / `P(REST)_avg` like the new-schema
    subjects (M2 Option B).
    """
    logs_dir = Path(DATA_DIR) / f"sub-{subject}" / f"ses-{session}" / "logs"
    if not logs_dir.is_dir():
        return pd.DataFrame()
    rows = []
    for run_dir in sorted(p for p in logs_dir.iterdir() if p.is_dir()
                           and p.name.startswith("ONLINE_")):
        csv = find_decoder_csv(str(run_dir))
        if csv is None:
            continue
        df = pd.read_csv(csv)
        if "Trial" not in df.columns:
            continue
        df["RunID"] = f"{subject}__{session}__{run_dir.name}"
        df["GlobalTrialID"] = (
            df["RunID"].astype(str) + "_" + df["Trial"].astype(str)
        )
        # M2 Option B: offline-integrate for the old-schema subject
        # (CLIN_SUBJ_002). No-op for the new-schema subjects whose CSVs
        # already contain `_avg` columns.
        if subject == "CLIN_SUBJ_002":
            df = _ensure_avg_cols(df, CLIN002_INTEGRATOR_ALPHA)
        try:
            per_trial = _per_trial_bar_dynamics(df)
        except Exception as e:
            print(
                f"  {session}/{run_dir.name}: per-trial compute FAILED: {e}"
            )
            continue
        per_trial["subject"] = subject
        per_trial["session"] = session
        per_trial["session_idx"] = session_idx_from_label(session)
        per_trial["run_id"] = run_dir.name
        rows.append(per_trial)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------
# Plotting + LME
# ----------------------------------------------------------------------

def _plot_subject_panel(
    df_subj_session: pd.DataFrame, subject: str, out_path: Path,
):
    """Per-subject 3-panel: median Lean%, median TTT, median slope per
    session, with MI and REST overlaid for Lean%."""
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    panels = [
        (axes[0], "LeanPct",          "Median Lean% (per trial)"),
        (axes[1], "TimeToThresh_s",   "Median time-to-threshold (s)"),
        (axes[2], "Slope_per_s",      "Median within-trial slope (1/s)"),
    ]
    for ax, col, label in panels:
        for cls, color in [("MI", "tab:orange"), ("REST", "tab:blue")]:
            sub = df_subj_session[df_subj_session.Class == cls]
            if sub.empty:
                continue
            grp = sub.groupby("session_idx")[col].agg(
                ["median", "count"],
            ).reset_index()
            ax.plot(
                grp["session_idx"], grp["median"], "o-",
                color=color, lw=2, label=f"{cls} (n trials avg)",
            )
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Session index")
    fig.suptitle(f"{subject} — bar dynamics over sessions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_cohort_metric_session(
    df_sess: pd.DataFrame, metric_col: str, label: str, out_path: Path,
    *, cls_split: bool = True,
    annotations: dict | None = None,
):
    """Pass-2 cohort plot. Per-session medians (one per subject) drawn
    as semi-transparent thin lines, cohort median bold. Split MI vs REST
    into TWO STACKED PANELS (Mi4 fix — previously 16 lines per panel
    were unreadable).

    `df_sess` is the per-session-summary dataframe with columns
    (subject, session_idx, Class, lean_median / ttt_median / slope_median).
    `metric_col` is the column to plot (e.g. 'lean_median').

    `annotations` is an optional dict {class: text} to draw a small
    boxed LME/Bonferroni annotation in each panel (Mi5).
    """
    classes = (
        sorted(df_sess["Class"].unique()) if cls_split else ["__ALL__"]
    )
    n_panels = len(classes)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(8, 4 * n_panels), sharex=True, squeeze=False,
    )
    cmap = plt.get_cmap("tab10")
    for p_idx, cls in enumerate(classes):
        ax = axes[p_idx][0]
        if cls_split:
            d_cls = df_sess[df_sess.Class == cls]
            title_extra = f" — {cls}"
        else:
            d_cls = df_sess
            title_extra = ""
        for i, subj in enumerate(sorted(d_cls["subject"].unique())):
            sub = d_cls[d_cls.subject == subj].sort_values("session_idx")
            ax.plot(
                sub["session_idx"], sub[metric_col], "o-",
                color=cmap(i % 10), alpha=0.5, lw=1.0, markersize=4,
                label=subj,
            )
        # Cohort median across subjects (per session_idx)
        cohort = d_cls.groupby("session_idx")[metric_col].agg(
            ["median", "mean", "sem"],
        ).reset_index()
        ax.errorbar(
            cohort["session_idx"], cohort["median"], yerr=cohort["sem"],
            color="black", lw=2.5, marker="s", markersize=8,
            label="Cohort median (± SE of session medians)",
        )
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"CLIN cohort — {label} over sessions{title_extra} "
            f"(per-session median; LME on session medians)",
            fontsize=10,
        )
        ax.legend(loc="best", fontsize=7, ncol=2)
        if annotations and cls in annotations and annotations[cls]:
            ax.text(
                0.02, 0.02, annotations[cls], transform=ax.transAxes,
                fontsize=8, va="bottom", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            )
    axes[-1][0].set_xlabel("Session index")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _lme_annotation(slope_p: float, df: pd.DataFrame, metric: str) -> str:
    """Build a one-shot LME-slope + Bonferroni annotation for a panel."""
    try:
        with _suppress_warnings():
            m = smf.mixedlm(
                f"{metric} ~ 1 + session_idx", df,
                groups=df["subject"],
            ).fit(disp=False)
        b = float(m.params.get("session_idx", np.nan))
    except Exception:
        b = np.nan
    return (
        f"LME slope = {b:+.3f}/session, p = {slope_p:.3g}\n"
        f"Bonferroni α' = {BONFERRONI_ALPHA_PRIMARY:.4f}: "
        f"{_bonferroni_verdict(slope_p)}"
    )


@contextmanager
def _suppress_warnings():
    with _warnings_mod.catch_warnings():
        _warnings_mod.simplefilter("ignore")
        yield


def _fit_lme(
    df: pd.DataFrame, metric: str, label: str,
) -> tuple[str, float]:
    """Fit `metric ~ 1 + session_idx + (1|subject)` on the given df
    and return (full text block, slope_p_value). Bonferroni verdict is
    appended to the block.
    """
    lines = [f"=== {label} ==="]
    df_use = df.dropna(subset=[metric, "session_idx", "subject"]).copy()
    slope_p = np.nan
    if len(df_use) < 5:
        lines.append(f"  not enough rows ({len(df_use)}); skipping.")
        return "\n".join(lines), slope_p
    if HAS_STATSMODELS:
        import warnings as _w
        try:
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                # Default optimizer (BFGS); pass-1 `method="lbfgs"`
                # was hitting singular-matrix errors on per-session data.
                model = smf.mixedlm(
                    f"{metric} ~ 1 + session_idx", df_use,
                    groups=df_use["subject"],
                ).fit(disp=False)
            lines.append(str(model.summary()))
            try:
                slope_p = float(model.pvalues.get("session_idx", np.nan))
            except Exception:
                slope_p = np.nan
        except Exception as exc:
            lines.append(
                f"  LME failed: {type(exc).__name__}: {exc}. "
                f"Spearman fallback below."
            )
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
                lines.append(f"    {subj}: ρ={r:+.3f}, p={p:.3g}, n={n}")
            finite = [r for _, r, _, _ in rho_rows]
            lines.append(
                f"  Cohort median ρ = {np.median(finite):+.3f}; "
                f"n positive = {sum(1 for r in finite if r > 0)}/{len(finite)}"
            )
    return "\n".join(lines), slope_p


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    out_dir = clin_pictures_root() / "bar_dynamics"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_trials = []
    for subject in enumerate_clin_subjects():
        sessions = enumerate_online_sessions_for_subject(subject)
        print(f"\n=== {subject} ({len(sessions)} sessions) ===")
        for sess in sessions:
            t0 = time.time()
            df_sess = _load_session_trials(subject, sess)
            if df_sess.empty:
                print(f"  {sess}: no usable runs; skip")
                continue
            all_trials.append(df_sess)
            print(
                f"  {sess}: n_trials={len(df_sess)} "
                f"({time.time()-t0:.1f}s)"
            )

    df_trials = pd.concat(all_trials, ignore_index=True) if all_trials else pd.DataFrame()
    if df_trials.empty:
        print("No trials collected; nothing to plot.")
        return
    df_trials.to_csv(out_dir / "bar_dynamics_per_trial.csv", index=False)

    # Session-level summary (MEDIAN primary, MEAN sensitivity).
    sess_summary_rows = []
    for (subject, session, sess_idx, cls), sub in df_trials.groupby(
        ["subject", "session", "session_idx", "Class"], dropna=False,
    ):
        sess_summary_rows.append({
            "subject": subject, "session": session,
            "session_idx": sess_idx, "Class": cls,
            "n_trials": int(len(sub)),
            "lean_median": float(sub["LeanPct"].median()),
            "lean_mean":   float(sub["LeanPct"].mean()),
            "ttt_median":  float(sub["TimeToThresh_s"].median()),
            "ttt_mean":    float(sub["TimeToThresh_s"].mean()),
            "slope_median": float(sub["Slope_per_s"].median()),
            "slope_mean":   float(sub["Slope_per_s"].mean()),
        })
    df_sess = pd.DataFrame(sess_summary_rows)
    df_sess.to_csv(out_dir / "bar_dynamics_session_summary.csv", index=False)

    # Per-subject panels (use per-trial dataframe; medians computed
    # inside the plot helper).
    for subject in sorted(df_trials["subject"].unique()):
        sub = df_trials[df_trials.subject == subject]
        _plot_subject_panel(
            sub, subject,
            out_dir / f"{subject}_bar_dynamics_3metric_over_sessions.png",
        )

    # Pass-2: LME on per-session summary (n=34 obs across 7 groups),
    # MEDIAN response per critic §C1/§M6. MEAN sensitivity is run only
    # for the txt block, not the cohort plot.
    blocks = []
    cohort_annotations = {
        "lean": {}, "ttt": {}, "slope": {},
    }
    for cls in ("MI", "REST"):
        sub_sess = df_sess[df_sess.Class == cls].copy()
        if sub_sess.empty:
            continue
        # MEDIAN (primary)
        for metric_short, metric_col, units in [
            ("lean", "lean_median", "Lean%"),
            ("ttt",  "ttt_median",  "TTT s"),
            ("slope", "slope_median", "Slope /s"),
        ]:
            label = f"{units} ({cls}) — per-session LME on MEDIAN"
            block, slope_p = _fit_lme(sub_sess, metric_col, label)
            blocks.append(block)
            cohort_annotations[metric_short][cls] = _lme_annotation(
                slope_p, sub_sess, metric_col,
            )
        # MEAN (sensitivity)
        for metric_short, metric_col, units in [
            ("lean", "lean_mean", "Lean%"),
            ("ttt",  "ttt_mean",  "TTT s"),
            ("slope", "slope_mean", "Slope /s"),
        ]:
            label = (
                f"{units} ({cls}) — per-session LME on MEAN "
                "(sensitivity check; per longitudinal-plan §3.3)"
            )
            block, _ = _fit_lme(sub_sess, metric_col, label)
            blocks.append(block)

    # Cohort plots — split MI vs REST into stacked panels (Mi4 fix).
    _plot_cohort_metric_session(
        df_sess, "lean_median", "Lean% (correct-class > 0.5)",
        out_dir / "cohort_lme_lean.png", cls_split=True,
        annotations=cohort_annotations["lean"],
    )
    _plot_cohort_metric_session(
        df_sess, "ttt_median", "Time-to-threshold (s)",
        out_dir / "cohort_lme_ttt.png", cls_split=True,
        annotations=cohort_annotations["ttt"],
    )
    _plot_cohort_metric_session(
        df_sess, "slope_median", "Within-trial slope (prob/s)",
        out_dir / "cohort_lme_slope.png", cls_split=True,
        annotations=cohort_annotations["slope"],
    )

    (out_dir / "bar_dynamics_lme_results.txt").write_text(
        "\n\n".join(blocks), encoding="utf-8",
    )
    print(f"\nDone. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
