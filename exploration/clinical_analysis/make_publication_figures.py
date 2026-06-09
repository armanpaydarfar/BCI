#!/usr/bin/env python3
"""Render publication-ready versions of the selected CLIN-cohort figures.

This is a *presentation-only* script: it reads the already-regenerated
canonical outputs under ``~/Pictures/clin_analysis/`` (CSVs) and re-plots
them with diagnostic text stripped and simplified titles/legends/axes. It
does NOT recompute any analysis and does NOT touch the canonical
``Analyze_clinical_*.py`` scripts or their diagnostic figures --- the
diagnostic versions remain the source of truth for internal review.

Every transform that turns a canonical CSV into a plotted value mirrors the
canonical plotting code it reproduces (cited inline). Output goes to the
manuscript ``figures/`` directory as PNG (300 dpi).

The cohort ERD-power scalp topomap is generated separately by
``make_erd_topomap.py`` (it recomputes TFRs and is slow).

Run (Windows):
    PYTHONUTF8=1 python -u exploration/clinical_analysis/make_publication_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exploration.clinical_analysis._helpers import clin_pictures_root  # noqa: E402

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
PIC = clin_pictures_root()
OUT_DIR = Path(r"C:\Users\arman\Documents\harmony-als-preprint\figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Anonymized participant labels for the manuscript (CLIN_SUBJ_002 -> P1, ...).
SUBJ_LABEL = {f"CLIN_SUBJ_{2 + i:03d}": f"P{i + 1}" for i in range(7)}
SUBJ_ORDER = list(SUBJ_LABEL)

# The single sensorimotor cluster used for the ERD figures (per cohort decision).
ERD_CLUSTER = "bilat"
CLUSTER_NAME = {"contra": "Contralateral", "ipsi": "Ipsilateral",
                "bilat": "Bilateral"}

# Consistent class colors across figures.
COL_MI = "#c0392b"     # motor imagery
COL_REST = "#2c6fbb"   # rest

# A muted qualitative palette for per-participant lines (7 participants).
PART_COLORS = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[[0, 1, 2, 3, 4, 6, 8]]


def _apply_style():
    """Publication rcParams: clean sans-serif, generous fonts, no clutter."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 1.8,
        "savefig.bbox": "tight",
    })


def _save(fig, stem: str):
    fig.savefig(OUT_DIR / f"{stem}.png")
    plt.close(fig)
    print(f"  wrote {stem}.png")


def _label(subject: str) -> str:
    return SUBJ_LABEL.get(subject, subject)


def _stars(p: float) -> str:
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def _pct_to_db(pct):
    """ERD% -> dB (10 log10), mirrors Analyze_clinical_erd_refined._pct_to_db."""
    ratio = np.clip(1.0 + np.asarray(pct, float) / 100.0, 1e-3, None)
    return 10.0 * np.log10(ratio)


# ----------------------------------------------------------------------
# Fig: decoder accuracy across sessions (total primary + decided shown)
# ----------------------------------------------------------------------
def fig_decoder_accuracy():
    """Total accuracy (ambiguous trials = non-success) as the primary metric,
    with decision accuracy (correct | committed) shown for context, plus
    trial-level Cohen's kappa. `acc_inclusive` / `acc_decided` per
    Analyze_clinical_decoder_longitudinal.py:233-239."""
    df = pd.read_csv(PIC / "decoder_perf" / "decoder_perf_session_summary.csv")
    xs = sorted(df["session_idx"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))

    # Panel A: accuracy.
    ax = axes[0]
    for i, subj in enumerate(SUBJ_ORDER):
        s = df[df["subject"] == subj].sort_values("session_idx")
        if s.empty:
            continue
        ax.plot(s["session_idx"], s["acc_inclusive"], "-o", ms=3.5, lw=0.9,
                color=PART_COLORS[i], alpha=0.5)
    coh_tot = df.groupby("session_idx")["acc_inclusive"].mean()
    coh_dec = df.groupby("session_idx")["acc_decided"].mean()
    ax.plot(coh_tot.index, coh_tot.values, "-o", color="k", lw=2.8, ms=6,
            zorder=5, label="Total accuracy (cohort)")
    ax.plot(coh_dec.index, coh_dec.values, "--s", color="0.35", lw=2.2, ms=5,
            zorder=5, label="Decision accuracy (cohort)")
    ax.axhline(0.5, ls=":", color="grey", lw=1.0)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("Session")
    ax.set_ylabel("Accuracy")
    ax.set_title("Closed-loop accuracy")
    ax.set_xticks(xs)
    ax.legend(loc="lower center", fontsize=9)

    # Panel B: trial-level kappa.
    ax = axes[1]
    for i, subj in enumerate(SUBJ_ORDER):
        s = df[df["subject"] == subj].sort_values("session_idx")
        if s.empty:
            continue
        ax.plot(s["session_idx"], s["trial_kappa"], "-o", ms=3.5, lw=0.9,
                color=PART_COLORS[i], alpha=0.55, label=_label(subj))
    coh = df.groupby("session_idx")["trial_kappa"].mean()
    ax.plot(coh.index, coh.values, "-o", color="k", lw=2.8, ms=6, zorder=5,
            label="Cohort mean")
    ax.set_xlabel("Session")
    ax.set_ylabel("Trial-level $\\kappa$")
    ax.set_title("Agreement (Cohen's $\\kappa$)")
    ax.set_xticks(xs)
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    _save(fig, "fig_decoder_accuracy")


# ----------------------------------------------------------------------
# Fig: cohort confusion matrix (reuses the canonical aggregator)
# ----------------------------------------------------------------------
def fig_confusion_cohort():
    from Analyze_clinical_confusion_matrices import _aggregate_subject_cm
    from exploration.clinical_analysis._helpers import enumerate_clin_subjects
    from exploration.clinical_analysis._subj002 import is_subj002

    cm = np.zeros((2, 3), dtype=int)
    n_subj = 0
    for subject in enumerate_clin_subjects():
        if is_subj002(subject):  # cohort sum excludes 002 (channel/shrinkage)
            continue
        sub_cm, _, _ = _aggregate_subject_cm(subject)
        if sub_cm.sum() == 0:
            continue
        cm += sub_cm
        n_subj += 1

    row_tot = cm.sum(axis=1, keepdims=True)
    pct = np.where(row_tot > 0, 100.0 * cm / np.maximum(row_tot, 1), 0.0)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% of actual class")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["MI", "Rest", "Ambiguous"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["MI", "Rest"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    # Both the trial count and the row-normalized percent per cell: the
    # cohort ran many trials (3,083), so absolute counts carry weight.
    for i in range(2):
        for j in range(3):
            on_dark = pct[i, j] > 50
            txt_col = "white" if on_dark else "black"
            ax.text(j, i - 0.10, f"{cm[i, j]:,}", ha="center", va="center",
                    color=txt_col, fontsize=15, fontweight="bold")
            ax.text(j, i + 0.13, f"{pct[i, j]:.0f}%", ha="center", va="center",
                    color=txt_col, fontsize=11)
    _save(fig, "fig_confusion_cohort")
    print(f"    (cohort confusion over {n_subj} participants, "
          f"{int(cm.sum())} trials)")


# ----------------------------------------------------------------------
# Fig: feature distinctiveness across sessions (supporting result)
# ----------------------------------------------------------------------
def _by_session_bars(ax, df, value_col, bar_color, ylab):
    """Cohort-mean bars by session with +/- SE whiskers and the individual
    participant values jittered on top. SE (not SD) is used on the bar because
    the between-participant SD is dominated by 1-2 high participants and would
    cross the axis floor; the scatter conveys the true spread instead."""
    xs = sorted(df["session_idx"].unique())
    rng = np.random.default_rng(1)
    means, ses = [], []
    for s in xs:
        v = df[df["session_idx"] == s][value_col].dropna().values
        means.append(np.mean(v))
        ses.append(np.std(v, ddof=1) / np.sqrt(len(v)))
    ax.bar(xs, means, width=0.62, color=bar_color, alpha=0.55,
           edgecolor=bar_color, linewidth=1.0, zorder=1)
    ax.errorbar(xs, means, yerr=ses, fmt="none", ecolor="k", elinewidth=1.3,
                capsize=4, zorder=3)
    # Participant scatter: a single mid-grey so the bars stay the focal element.
    for s in xs:
        v = df[df["session_idx"] == s][value_col].dropna().values
        x = s + (rng.random(len(v)) - 0.5) * 0.30
        ax.plot(x, v, "o", ms=4, color="0.25", alpha=0.7, markeredgewidth=0,
                zorder=4)
    ax.set_xlabel("Session")
    ax.set_ylabel(ylab)
    ax.set_xticks(xs)


def fig_longitudinal():
    """Two-panel longitudinal composite: (A) bilateral mu-ERD magnitude is
    stable across sessions (F4 null), (B) Riemannian feature distinctiveness
    rises across sessions. Both as cohort-mean bars +/- s.e. across participants
    with the individual participants overlaid."""
    erd = pd.read_csv(PIC / "neuromod" / "erd_session_summary.csv")
    erd = erd[erd["cluster"] == ERD_CLUSTER]
    fd = pd.read_csv(PIC / "feat_dist" / "feat_dist_per_session.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    _by_session_bars(axes[0], erd, "erd_mean", COL_MI,
                     f"{CLUSTER_NAME[ERD_CLUSTER]} mu-ERD (%)")
    axes[0].axhline(0, color="grey", lw=0.8, zorder=2)
    axes[0].set_title("Sensorimotor rhythm is stable")
    _by_session_bars(axes[1], fd, "fd", COL_REST, "Feature distinctiveness")
    axes[1].set_title("Class separability increases")
    for ax, lab in zip(axes, "AB"):
        ax.text(-0.12, 1.02, lab, transform=ax.transAxes, fontsize=15,
                fontweight="bold", va="bottom")
    _save(fig, "fig_longitudinal")


# ----------------------------------------------------------------------
# Fig: mu-ERD magnitude across sessions (single bilateral cluster, F4 null)
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Fig: bilateral ERD time course, cohort MI vs REST. One line per session =
# cohort grand mean across participants, each with a +/-SE-across-participants
# shaded band (mirrors the erd_refined substrate column `mean_pct`, in dB).
# ----------------------------------------------------------------------
def fig_erd_timecourse():
    """Session-resolved bilateral mu-ERD time course, MI and REST in separate
    panels (the bilateral row of the diagnostic erd_refined 6-panel). One line
    per session index = cohort grand mean across participants at each time
    sample, displayed in dB (10 log10), as the canonical erd_refined does. Each
    session line carries a +/-SE shaded band across the participants contributing
    to that session, the cohort analogue of the diagnostic's per-session bands
    (Analyze_clinical_erd_refined.py:744, which band across a subject's trials)."""
    df = pd.read_csv(PIC / "erd_refined" / "erd_refined_data_car.csv")
    sub = df[df["cluster"] == ERD_CLUSTER]
    sessions = sorted(sub["session_idx"].unique())
    sess_colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.85, len(sessions)))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    for ax, marker, name in ((axes[0], "mi", "Motor imagery"),
                             (axes[1], "rest", "Rest")):
        m = sub[sub["marker"] == marker]
        for si, s in enumerate(sessions):
            ms = m[m["session_idx"] == s]
            # Cohort grand mean across participants at each time sample, with a
            # +/-SE-across-participants band. n per sample varies (a participant
            # missing a sample drops out), so SE = SD/sqrt(n_t) is per-sample;
            # samples with <2 participants get no band.
            piv = ms.pivot_table(index="t", columns="subject", values="mean_pct")
            t = piv.index.to_numpy()
            db = _pct_to_db(piv.to_numpy())
            mean_db = np.nanmean(db, axis=1)
            n_t = np.sum(~np.isnan(db), axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                se = np.nanstd(db, axis=1, ddof=1) / np.sqrt(n_t)
            se = np.where(n_t >= 2, se, 0.0)
            n_sub = int(np.sum(~np.all(np.isnan(db), axis=0)))
            ax.fill_between(t, mean_db - se, mean_db + se,
                            color=sess_colors[si], alpha=0.15, linewidth=0)
            ax.plot(t, mean_db, color=sess_colors[si], lw=1.9,
                    label=f"Session {s} (n={n_sub})")
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="k", ls="--", lw=0.7)
        ax.set_xlabel("Time (s)")
        ax.set_title(name)
        ax.legend(loc="lower left", fontsize=8.5)
    axes[0].set_ylabel(f"{CLUSTER_NAME[ERD_CLUSTER]} mu-ERD (dB)")
    fig.suptitle("Bilateral mu-ERD time course by session", y=1.00)
    _save(fig, "fig_erd_timecourse")


# ----------------------------------------------------------------------
# Fig: alignment ablation (three arms), with significance brackets
# ----------------------------------------------------------------------
def fig_gr_ablation():
    import json
    df = pd.read_csv(
        PIC / "gr_ablation" / "csv" / "gr_ablation_session_summary.csv"
    )
    stats = json.loads(
        (PIC / "gr_ablation" / "csv" / "gr_ablation_cohort_stats.json")
        .read_text()
    )
    # Arms: A=off (no alignment), C=ra (batch RA), B=on (Generic Recentering).
    arms = [("bal_acc_off", "No\nalignment"),
            ("bal_acc_ra", "Batch RA"),
            ("bal_acc_on", "Generic\nRecentering")]
    data = [df[c].dropna().values for c, _ in arms]
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    positions = [1, 2, 3]
    bp = ax.boxplot(data, positions=positions, widths=0.55,
                    showfliers=False, patch_artist=True)
    arm_colors = ["#bdbdbd", "#9ecae1", "#3182bd"]
    for patch, c in zip(bp["boxes"], arm_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    for med in bp["medians"]:
        med.set_color("k")
    rng = np.random.default_rng(0)
    for pos, d in zip(positions, data):
        x = pos + (rng.random(len(d)) - 0.5) * 0.22
        ax.plot(x, d, "o", ms=3.2, color="k", alpha=0.35)
    ax.set_xticks(positions)
    ax.set_xticklabels([lab for _, lab in arms])
    ax.set_ylabel("Balanced accuracy")
    ax.axhline(0.5, ls=":", color="grey", lw=1.0)

    pmap = {
        (1, 3): stats["metric_bal_acc_delta_BA"]["p"],  # B vs A
        (2, 3): stats["metric_bal_acc_delta_BC"]["p"],  # B vs C
        (1, 2): stats["metric_bal_acc_delta_CA"]["p"],  # C vs A
    }
    ymax = max(d.max() for d in data)
    step = 0.05 * (ymax - min(d.min() for d in data) + 0.1)
    levels = {(1, 2): ymax + step, (2, 3): ymax + step,
              (1, 3): ymax + 2.4 * step}
    for (lo, hi), p in pmap.items():
        y = levels[(lo, hi)]
        ax.plot([lo, lo, hi, hi], [y, y + step * 0.4, y + step * 0.4, y],
                color="k", lw=1.1)
        ax.text((lo + hi) / 2, y + step * 0.45, _stars(p),
                ha="center", va="bottom", fontsize=12)
    ax.set_ylim(top=ymax + 3.4 * step)
    _save(fig, "fig_gr_ablation")


# ----------------------------------------------------------------------
# Fig: decision-bar dynamics (lean and time-to-threshold) across sessions
# ----------------------------------------------------------------------
def fig_bar_dynamics():
    """Per-PARTICIPANT box-and-whisker of MI vs REST decision dynamics: each box
    pools that participant's per-run values across all sessions, with the runs
    jittered on top. Subject-wise (not session-wise) because the dynamics are
    stable across sessions with no longitudinal ramp -- the informative axis is
    between-participant consistency. Two panels: lean and time-to-threshold
    (per-run columns of bar_dynamics_per_run.csv)."""
    df = pd.read_csv(PIC / "bar_dynamics" / "bar_dynamics_per_run.csv")
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    offs = {"MI": -0.19, "REST": 0.19}

    def _panel(ax, col, ylab, title):
        for cls, color in (("MI", COL_MI), ("REST", COL_REST)):
            d = df[df["Class"] == cls]
            data, positions = [], []
            for i, subj in enumerate(SUBJ_ORDER):
                vals = d[d["subject"] == subj][col].dropna().values
                data.append(vals)
                positions.append(i + offs[cls])
            bp = ax.boxplot(data, positions=positions, widths=0.32,
                            showfliers=False, patch_artist=True,
                            medianprops=dict(color="k", lw=1.6))
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.45)
            for whisk in bp["whiskers"] + bp["caps"]:
                whisk.set_color(color)
            for pos, vals in zip(positions, data):
                x = pos + (rng.random(len(vals)) - 0.5) * 0.16
                ax.plot(x, vals, "o", ms=2.6, color=color, alpha=0.5,
                        markeredgewidth=0)
        ax.set_xticks(range(len(SUBJ_ORDER)))
        ax.set_xticklabels([_label(s) for s in SUBJ_ORDER])
        ax.set_xlabel("Participant")
        ax.set_ylabel(ylab)
        ax.set_title(title)

    _panel(axes[0], "LeanPct", "Lean toward correct class (%)", "Decision lean")
    axes[0].axhline(50, color="grey", ls=":", lw=1.0)
    _panel(axes[1], "TimeToThresh_s", "Time to decision (s)", "Decision speed")
    handles = [plt.Line2D([], [], color=COL_MI, lw=6, alpha=0.45, label="MI"),
               plt.Line2D([], [], color=COL_REST, lw=6, alpha=0.45, label="Rest")]
    axes[1].legend(handles=handles, loc="upper right", fontsize=10)
    _save(fig, "fig_bar_dynamics")


# ----------------------------------------------------------------------
# Fig: EDS topography (class = MI), cohort z-score on the scalp
# ----------------------------------------------------------------------
def fig_eds_topomap_mi():
    import mne
    df = pd.read_csv(PIC / "eds" / "eds_per_class_cohort_summary_mu.csv")
    d = df[(df["band"] == "mu") & (df["class"] == "mi")].copy()
    ch_names = list(d["channel"])
    values = d["cohort_z_score"].to_numpy(dtype=float)

    info = mne.create_info(ch_names, sfreq=512.0, ch_types="eeg")
    info.set_montage("standard_1020", match_case=False)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    vmax = float(np.nanmax(np.abs(values)))
    im, _ = mne.viz.plot_topomap(
        values, info, axes=ax, show=False, cmap="viridis",
        vlim=(-vmax, vmax), names=ch_names, contours=6, sensors=True,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label("EDS (z-score)")
    _save(fig, "fig_eds_topomap_mi")


def main():
    _apply_style()
    print(f"Writing publication figures to: {OUT_DIR}")
    fig_decoder_accuracy()
    fig_confusion_cohort()
    fig_bar_dynamics()
    fig_gr_ablation()
    fig_erd_timecourse()
    fig_longitudinal()
    fig_eds_topomap_mi()
    print("Done.")


if __name__ == "__main__":
    main()
