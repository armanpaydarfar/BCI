#!/usr/bin/env python3
"""Bar dynamics longitudinal (Plan C of
`rev01-longitudinal-analysis-plan.md` §4.5).

Per-trial summaries:
  - Lean% (`% samples with P(correct_class) > THRESH`) — reuses
    Analyze_experiment_logs_cross_subject.compute_lean16hz_per_trial.
    Per Arman 2026-05-28 (rejecting critic §C3): Lean% stays on the
    **instantaneous** P(MI) / P(REST) stream because the instantaneous
    probability determines bar direction-of-motion.
  - Time-to-threshold (TTT): first time within trial (relative to cue
    onset) that `P(correct)_avg` exceeds 0.6. Trial time is reconstructed
    from the per-run `config_snapshot.json` `CLASSIFY_WINDOW` — the
    runtime skips the first `CLASSIFY_WINDOW/1000` s before the first
    classification (`Utils/runtime_common.py:711`
    `next_tick = start_time + window_size`), so `t = t - t[0] + window_size`.
    Without this offset, TTT was anchored to the first classification
    timestamp instead of the cue, masking the shutout.
    Censored at trial end if never crossed.

Plot style (2026-05-29 revision):
  - Box-and-whisker, per-run points: x = session index; at each x there
    are two boxes (MI / REST) summarising per-run median values.
  - Per-subject panel: box = distribution of runs within that
    (subject, session, class).
  - Cohort panel: box = distribution of runs across all (subject, session,
    class) at the given session index.
  - Within-trial slope removed (low value; superseded by Lean% + TTT).

Pass 2 (2026-05-28) — still in force:
  - LMEs fit per-session (response = MEDIAN of per-run medians, with
    MEAN sensitivity), not per-trial — critic §C1/§C2.
  - M2 Option B: CLIN_SUBJ_002 only has `P(MI)` / `P(REST)`
    (instantaneous) columns. To compare TTT apples-to-apples with
    CLIN_SUBJ_003..008 (whose values are computed on the leaky-
    integrated `_avg` stream), the script integrates CLIN_SUBJ_002's
    instantaneous probabilities offline with `alpha=0.95`
    (rev01-paper-angle.md §1.1) before computing TTT.

Outputs (`~/Pictures/clin_analysis/bar_dynamics/`):
    <SUBJ>_bar_dynamics_over_sessions.png    (per subject; Lean + TTT)
    cohort_lme_lean.png    cohort_lme_ttt.png
    bar_dynamics_per_trial.csv     (Lean, TTT per trial)
    bar_dynamics_per_run.csv       (per-run medians; box plot source)
    bar_dynamics_session_summary.csv
    bar_dynamics_lme_results.txt
"""

from __future__ import annotations

import json
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


# Fallback shutout (seconds) when a run lacks config_snapshot.json.
# Standard CLIN_SUBJ_003..008 runs logged CLASSIFY_WINDOW = 1000 ms;
# CLIN_SUBJ_002 logged 500 ms — read per-run from the snapshot when
# available (see `_run_shutout_s`).
_DEFAULT_SHUTOUT_S = 1.0
_CLIN002_SHUTOUT_S = 0.5

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

def _per_trial_bar_dynamics(
    df_run: pd.DataFrame, shutout_s: float,
) -> pd.DataFrame:
    """Compute (Lean%, TTT) for each trial in one run's decoder_output.csv.

    `shutout_s` is the runtime's pre-classification window
    (`CLASSIFY_WINDOW / 1000`, per `Utils/runtime_common.py:711`).
    The first decoder_output row of a trial is logged at the trial cue
    time + `shutout_s`. To express TTT in trial-cue-relative time we
    therefore add `shutout_s` after rebasing to `t - t[0]`.

    Returns DataFrame with columns:
      GlobalTrialID, Class, LeanPct, TimeToThresh_s (NaN if not crossed).
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

    # Per-trial TTT on the leaky-integrated correct-class prob.
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
        # Trial time relative to the cue: first decoder_output sample
        # was logged at cue + shutout_s.
        if "Timestamp" in tdf.columns and np.issubdtype(
            tdf["Timestamp"].dtype, np.number,
        ):
            t = tdf["Timestamp"].values.astype(float)
            t = t - t[0] + shutout_s
        else:
            t = shutout_s + np.arange(len(p)) / CLASSIFIER_HZ
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
        lean_pct = (
            float(lean.loc[gtid, "LeanPct"])
            if gtid in lean.index else np.nan
        )
        out_rows.append({
            "GlobalTrialID": gtid, "Class": cls,
            "LeanPct": lean_pct, "TimeToThresh_s": ttt,
        })
    return pd.DataFrame(out_rows)


def _run_shutout_s(run_dir: Path, subject: str) -> float:
    """Read `CLASSIFY_WINDOW` (ms) from the run's `config_snapshot.json`
    and convert to seconds. Falls back to `_CLIN002_SHUTOUT_S` for
    CLIN_SUBJ_002 (500 ms window per rev01-paper-angle.md §1.1) or to
    `_DEFAULT_SHUTOUT_S` otherwise, if the snapshot is missing.
    """
    snap = run_dir / "config_snapshot.json"
    if snap.is_file():
        try:
            data = json.loads(snap.read_text())
            cw = data.get("CLASSIFY_WINDOW")
            if cw is not None:
                return float(cw) / 1000.0
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    return _CLIN002_SHUTOUT_S if subject == "CLIN_SUBJ_002" else _DEFAULT_SHUTOUT_S


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
        shutout_s = _run_shutout_s(run_dir, subject)
        try:
            per_trial = _per_trial_bar_dynamics(df, shutout_s)
        except Exception as e:
            print(
                f"  {session}/{run_dir.name}: per-trial compute FAILED: {e}"
            )
            continue
        per_trial["subject"] = subject
        per_trial["session"] = session
        per_trial["session_idx"] = session_idx_from_label(session)
        per_trial["run_id"] = run_dir.name
        per_trial["shutout_s"] = shutout_s
        rows.append(per_trial)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------
# Plotting + LME
# ----------------------------------------------------------------------

def _per_run_median(
    df_trials: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse per-trial rows to per-run medians, retaining
    (subject, session, session_idx, run_id, Class). Used as the box-plot
    point source: each run-class pair contributes one point per metric."""
    if df_trials.empty:
        return pd.DataFrame(columns=[
            "subject", "session", "session_idx", "run_id", "Class",
            "LeanPct", "TimeToThresh_s", "n_trials",
        ])
    grp = df_trials.groupby(
        ["subject", "session", "session_idx", "run_id", "Class"],
        dropna=False, as_index=False,
    )
    rows = grp.agg(
        LeanPct=("LeanPct", "median"),
        TimeToThresh_s=("TimeToThresh_s", "median"),
        n_trials=("LeanPct", "size"),
    )
    return rows


_CLASS_COLOR = {"MI": "tab:orange", "REST": "tab:blue"}


def _draw_box_panel(
    ax, df_runs: pd.DataFrame, metric_col: str, ylabel: str,
    *, shutout_s: float | None = None, point_alpha: float = 0.6,
):
    """Render a single box-and-whisker panel on `ax`. For each
    session_idx draw side-by-side boxes for whichever of (MI, REST) are
    present in `df_runs`. Overlay per-run points jittered around the
    box center."""
    if df_runs.empty:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="grey")
        return
    sess_idxs = sorted(df_runs["session_idx"].dropna().unique())
    present = set(df_runs["Class"].unique())
    classes = [c for c in ("MI", "REST") if c in present]
    if len(classes) == 1:
        box_width = 0.45
        offsets = {classes[0]: 0.0}
    else:
        box_width = 0.32
        offsets = {"MI": -box_width / 2, "REST": +box_width / 2}
    box_positions = []
    box_data = []
    box_colors = []
    for s in sess_idxs:
        for cls in classes:
            vals = df_runs[
                (df_runs.session_idx == s) & (df_runs.Class == cls)
            ][metric_col].dropna().values
            if len(vals) == 0:
                continue
            box_positions.append(s + offsets[cls])
            box_data.append(vals)
            box_colors.append(_CLASS_COLOR[cls])
    if box_data:
        bp = ax.boxplot(
            box_data, positions=box_positions, widths=box_width,
            patch_artist=True, showfliers=False, manage_ticks=False,
            medianprops=dict(color="black", linewidth=1.6),
            whiskerprops=dict(color="black", linewidth=1.0),
            capprops=dict(color="black", linewidth=1.0),
            boxprops=dict(linewidth=0.8),
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
    # Overlay per-run points (jittered)
    rng = np.random.default_rng(0)
    for cls in classes:
        sub = df_runs[df_runs.Class == cls]
        for s in sess_idxs:
            vals = sub[sub.session_idx == s][metric_col].dropna().values
            if len(vals) == 0:
                continue
            x = s + offsets[cls] + rng.uniform(
                -box_width * 0.25, box_width * 0.25, size=len(vals),
            )
            ax.scatter(
                x, vals, s=18, color=_CLASS_COLOR[cls],
                edgecolor="white", linewidth=0.4, alpha=point_alpha,
                zorder=3,
            )
    ax.set_xticks(sess_idxs)
    ax.set_xticklabels([str(int(s)) for s in sess_idxs])
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    if shutout_s is not None and metric_col == "TimeToThresh_s":
        ax.axhline(
            shutout_s, color="tab:red", lw=1, linestyle=":",
            alpha=0.6, zorder=1,
        )
        ax.text(
            ax.get_xlim()[0], shutout_s, " shutout",
            color="tab:red", fontsize=7, va="bottom", ha="left",
        )
    legend_handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=_CLASS_COLOR[cls], alpha=0.35,
            edgecolor="black", linewidth=0.8, label=cls,
        )
        for cls in classes
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=8)


def _plot_subject_panel(
    df_subj_trials: pd.DataFrame, subject: str, out_path: Path,
):
    """Per-subject box-and-whisker plot: 2 panels (Lean%, TTT). Points
    are per-run medians (one per run-class). X axis is session_idx;
    MI / REST shown as side-by-side boxes at each session."""
    runs_df = _per_run_median(df_subj_trials)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    shutout_s = (
        float(df_subj_trials["shutout_s"].median())
        if "shutout_s" in df_subj_trials.columns
        and not df_subj_trials["shutout_s"].dropna().empty
        else None
    )
    _draw_box_panel(
        axes[0], runs_df, "LeanPct",
        "Per-run median Lean% (correct-class > 0.5)",
    )
    _draw_box_panel(
        axes[1], runs_df, "TimeToThresh_s",
        "Per-run median TTT (s, relative to cue)",
        shutout_s=shutout_s,
    )
    axes[-1].set_xlabel("Session index")
    fig.suptitle(f"{subject} — bar dynamics over sessions (per-run points)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_cohort_box(
    df_runs: pd.DataFrame, metric_col: str, label: str, out_path: Path,
    *, shutout_s: float | None = None,
    annotations: dict | None = None,
):
    """Cohort box-and-whisker plot. X axis = session_idx; at each x two
    boxes (MI / REST) summarise the per-run medians across the cohort.

    `df_runs` is the per-run-medians dataframe (columns: subject,
    session_idx, Class, LeanPct, TimeToThresh_s, …) produced by
    `_per_run_median(df_trials)`.

    `annotations` is an optional dict {class: text} drawing a small
    LME/Bonferroni annotation in each panel (Mi5).
    """
    classes = ["MI", "REST"]
    fig, axes = plt.subplots(
        len(classes), 1, figsize=(8, 4 * len(classes)), sharex=True,
        squeeze=False,
    )
    for p_idx, cls in enumerate(classes):
        ax = axes[p_idx][0]
        d_cls = df_runs[df_runs.Class == cls]
        # Use a single-class draw on the panel
        _draw_box_panel(
            ax,
            d_cls.assign(Class=cls),
            metric_col, label, shutout_s=shutout_s,
        )
        ax.set_title(
            f"CLIN cohort — {label} — {cls} "
            f"(boxes = per-run medians; LME on per-session medians)",
            fontsize=10,
        )
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
    b = np.nan
    df_fit = df.dropna(subset=[metric, "session_idx", "subject"])
    if not df_fit.empty:
        try:
            with _suppress_warnings():
                m = smf.mixedlm(
                    f"{metric} ~ 1 + session_idx", df_fit,
                    groups=df_fit["subject"],
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

    # Per-run medians (box-plot source of truth).
    df_runs = _per_run_median(df_trials)
    df_runs.to_csv(out_dir / "bar_dynamics_per_run.csv", index=False)

    # Session-level summary (MEDIAN-of-per-run-medians, primary; MEAN
    # sensitivity). Keeps the LME at per-session granularity per the
    # pass-2 critic §C1/§M6 fix while letting the box plot show per-run
    # spread.
    sess_summary_rows = []
    for (subject, session, sess_idx, cls), sub in df_runs.groupby(
        ["subject", "session", "session_idx", "Class"], dropna=False,
    ):
        sess_summary_rows.append({
            "subject": subject, "session": session,
            "session_idx": sess_idx, "Class": cls,
            "n_runs": int(len(sub)),
            "lean_median": float(sub["LeanPct"].median()),
            "lean_mean":   float(sub["LeanPct"].mean()),
            "ttt_median":  float(sub["TimeToThresh_s"].median()),
            "ttt_mean":    float(sub["TimeToThresh_s"].mean()),
        })
    df_sess = pd.DataFrame(sess_summary_rows)
    df_sess.to_csv(out_dir / "bar_dynamics_session_summary.csv", index=False)

    # Per-subject box-and-whisker panels.
    for subject in sorted(df_trials["subject"].unique()):
        sub = df_trials[df_trials.subject == subject]
        _plot_subject_panel(
            sub, subject,
            out_dir / f"{subject}_bar_dynamics_over_sessions.png",
        )

    # Pass-2 LME on per-session summary (MEDIAN of per-run medians is
    # the primary response; MEAN sensitivity is included in the txt).
    blocks = []
    cohort_annotations = {"lean": {}, "ttt": {}}
    for cls in ("MI", "REST"):
        sub_sess = df_sess[df_sess.Class == cls].copy()
        if sub_sess.empty:
            continue
        # MEDIAN (primary)
        for metric_short, metric_col, units in [
            ("lean", "lean_median", "Lean%"),
            ("ttt",  "ttt_median",  "TTT s"),
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
        ]:
            label = (
                f"{units} ({cls}) — per-session LME on MEAN "
                "(sensitivity check; per longitudinal-plan §3.3)"
            )
            block, _ = _fit_lme(sub_sess, metric_col, label)
            blocks.append(block)

    cohort_shutout_s = (
        float(df_trials["shutout_s"].median())
        if "shutout_s" in df_trials.columns
        and not df_trials["shutout_s"].dropna().empty
        else None
    )
    _plot_cohort_box(
        df_runs, "LeanPct", "Lean% (correct-class > 0.5)",
        out_dir / "cohort_lme_lean.png",
        annotations=cohort_annotations["lean"],
    )
    _plot_cohort_box(
        df_runs, "TimeToThresh_s", "Time-to-threshold (s, cue-relative)",
        out_dir / "cohort_lme_ttt.png",
        shutout_s=cohort_shutout_s,
        annotations=cohort_annotations["ttt"],
    )

    (out_dir / "bar_dynamics_lme_results.txt").write_text(
        "\n\n".join(blocks), encoding="utf-8",
    )
    print(f"\nDone. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
