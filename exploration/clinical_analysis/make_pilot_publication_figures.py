#!/usr/bin/env python3
"""Render NER/EMBC conference figures for the tiagobot healthy-pilot cohort.

Presentation-only: reads the per-subject CSV / NPZ artifacts produced by
``scratch_oneshot_subj007_analysis.py`` over the tiagobot pilot tree
(ONESHOT_ROOT=...\\tiagobot, ONESHOT_OUT=...\\Pictures\\pilot_analysis,
ONESHOT_SUBJECT=PILOT001..008) and aggregates them across the eight pilots
into cohort figures. It recomputes no EEG analysis; every per-subject number
comes straight from those artifacts.

Aggregation rules mirror the canonical cohort plotters in
``make_publication_figures.py``:
  * decoder accuracy / confusion are pooled from per-run trial counts;
  * the mu-ERD time course is averaged in log-ratio (geometric) space, the
    standard ERSP convention, from the per-trial percent traces in the NPZ;
  * cohort EDS is each subject's per-channel EDS z-scored within subject
    (across channels) then averaged across subjects (matches the oneshot
    "z-scored per participant and averaged" pooled EDS).

Run (Windows):
    PYTHONUTF8=1 python -u exploration/clinical_analysis/make_pilot_publication_figures.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
PIC = Path(r"C:\Users\arman\Pictures\pilot_analysis")
OUT_DIR = Path(r"C:\Users\arman\Documents\tiagobot-bci-conference\figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PILOTS = [f"PILOT{i:03d}" for i in range(1, 9)]
# De-identified manuscript labels (healthy pilots H1..H8).
LABEL = {s: f"H{i + 1}" for i, s in enumerate(PILOTS)}

COL_MI = "#c0392b"
COL_REST = "#2c6fbb"
PART_COLORS = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[
    [0, 1, 2, 3, 4, 6, 8, 9]]


def _apply_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.9, "legend.frameon": False, "legend.fontsize": 9,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "lines.linewidth": 1.8, "savefig.bbox": "tight",
    })


def _save(fig, stem):
    fig.savefig(OUT_DIR / f"{stem}.png")
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    plt.close(fig)
    print(f"  wrote {stem}.png/.pdf")


def _cohen_kappa_2x2(a, b, c, d):
    """Cohen's kappa for a 2x2 decided confusion [[a,b],[c,d]] (rows=actual
    MI/REST, cols=predicted MI/REST)."""
    n = a + b + c + d
    if n == 0:
        return np.nan
    po = (a + d) / n
    p_mi = ((a + b) * (a + c)) / n**2
    p_rest = ((c + d) * (b + d)) / n**2
    pe = p_mi + p_rest
    return (po - pe) / (1 - pe) if (1 - pe) > 0 else np.nan


# ----------------------------------------------------------------------
# Load per-subject artifacts into cohort frames
# ----------------------------------------------------------------------
def _load_confusion():
    frames = []
    for s in PILOTS:
        f = PIC / "confusion_matrices" / f"{s}_S001ONLINE_per_run_confusion_summary.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
    return pd.concat(frames, ignore_index=True)


def _subject_pooled(conf):
    """Pool per-run confusion counts into one row per subject."""
    rows = []
    for s in PILOTS:
        d = conf[conf["subject"] == s]
        if d.empty:
            continue
        a, b = d["mi_mi"].sum(), d["mi_rest"].sum()
        c, e = d["rest_mi"].sum(), d["rest_rest"].sum()
        trials = d["trials"].sum()
        decisions = d["decisions"].sum()
        correct = d["correct"].sum()
        ambiguous = d["ambiguous"].sum()
        rows.append({
            "subject": s, "label": LABEL[s], "trials": trials,
            "decisions": decisions, "ambiguous": ambiguous,
            "dec_acc": correct / decisions if decisions else np.nan,
            "tot_acc": correct / trials if trials else np.nan,
            "kappa": _cohen_kappa_2x2(a, b, c, e),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Fig: decoder performance across the pilot cohort
# ----------------------------------------------------------------------
def fig_pilot_decoder(sub):
    """Per-participant (session-level) decoder metrics as line plots across the
    cohort: (A) decision and total accuracy, (B) trial-level Cohen's kappa."""
    # Single-column, vertically stacked so the figure flows inline in an
    # IEEEtran results column. Kept compact (short) so two figures can share a
    # column and the set packs without dedicated figure-only pages.
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.1))
    x = np.arange(len(sub))

    ax = axes[0]
    ax.plot(x, sub["dec_acc"], "-o", color="0.35", lw=1.8, ms=5,
            label="Decision acc.")
    ax.plot(x, sub["tot_acc"], "-s", color=COL_MI, lw=1.8, ms=5,
            label="Total acc.")
    ax.axhline(0.5, ls=":", color="grey", lw=1.0, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["label"])
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("Accuracy")
    ax.set_title("Closed-loop accuracy")
    ax.legend(loc="lower right", fontsize=8, ncol=3, columnspacing=1.0,
              handletextpad=0.4)

    ax = axes[1]
    ax.plot(x, sub["kappa"], "-o", color=COL_REST, lw=1.8, ms=5,
            label="Trial-level $\\kappa$")
    ax.axhline(sub["kappa"].mean(), ls="--", color="0.2", lw=1.2,
               label=f"Cohort mean = {sub['kappa'].mean():.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["label"])
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("Trial-level $\\kappa$")
    ax.set_xlabel("Healthy participant")
    ax.set_title("Agreement (Cohen's $\\kappa$)")
    ax.legend(loc="lower right", fontsize=8)
    for a, lab in zip(axes, "AB"):
        a.text(-0.18, 1.02, lab, transform=a.transAxes, fontsize=13,
               fontweight="bold", va="bottom")
    fig.tight_layout()
    _save(fig, "fig_pilot_decoder")


# ----------------------------------------------------------------------
# Fig: pooled cohort confusion matrix
# ----------------------------------------------------------------------
def fig_pilot_confusion(conf):
    cm = np.array([
        [conf["mi_mi"].sum(), conf["mi_rest"].sum(), conf["mi_amb"].sum()],
        [conf["rest_mi"].sum(), conf["rest_rest"].sum(), conf["rest_amb"].sum()],
    ], dtype=int)
    row_tot = cm.sum(axis=1, keepdims=True)
    pct = 100.0 * cm / np.maximum(row_tot, 1)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% of actual class")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["MI", "Rest", "Ambiguous"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["MI", "Rest"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(3):
            on_dark = pct[i, j] > 50
            col = "white" if on_dark else "black"
            ax.text(j, i - 0.10, f"{cm[i, j]:,}", ha="center", va="center",
                    color=col, fontsize=14, fontweight="bold")
            ax.text(j, i + 0.13, f"{pct[i, j]:.0f}%", ha="center", va="center",
                    color=col, fontsize=10)
    _save(fig, "fig_pilot_confusion")
    print(f"    (pooled over 8 pilots, {int(cm.sum())} trials)")


# ----------------------------------------------------------------------
# Fig: cohort bilateral mu-ERD time course (MI vs REST), log-ratio mean +/- SE
# ----------------------------------------------------------------------
def _pct_to_logratio(pct):
    return np.log10(np.clip(1.0 + pct / 100.0, 1e-3, None))


def fig_pilot_erd():
    # Common analysis grid (canonical TRIAL_WIN, ~512 Hz); subjects are
    # interpolated onto it so a +/-1-sample difference in trial cropping
    # cannot break the stack. One thin line per participant, a bold cohort mean,
    # and a +/-SE-across-participants band. Each subject's log-ratio trace is
    # re-zeroed over the -1..0 s baseline window in log space, matching the
    # preprint's per-trial log baseline re-zero (Analyze_clinical_erd_refined).
    times_ref = np.linspace(-1.0, 4.0, 1024)
    base = (times_ref >= -1.0) & (times_ref <= 0.0)
    subj_traces = {"mi": [], "rest": []}   # list of (label, trace)
    for i, s in enumerate(PILOTS):
        hits = glob.glob(str(PIC / "erd_refined" / "per_trial" / f"{s}_*npz"))
        if not hits:
            continue
        d = np.load(hits[0], allow_pickle=True)
        for marker in ("mi", "rest"):
            ptp = d[f"bilat_{marker}__ptp"]          # (n_trials, T) percent
            t = d[f"bilat_{marker}__times"]
            lr = _pct_to_logratio(ptp).mean(axis=0)  # geometric trial mean
            lr = np.interp(times_ref, t, lr)
            lr = lr - lr[base].mean()                # log-space baseline re-zero
            subj_traces[marker].append((LABEL[s], lr))

    # Single-column, MI stacked over REST (compact height for packing).
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.1), sharex=True, sharey=True)
    for ax, marker, name in ((axes[0], "mi", "Motor imagery"),
                             (axes[1], "rest", "Rest")):
        traces = subj_traces[marker]
        for i, (lab, tr) in enumerate(traces):
            ax.plot(times_ref, tr, color=PART_COLORS[i], lw=0.8, alpha=0.55,
                    label=lab)
        M = np.vstack([tr for _lab, tr in traces])
        mean = M.mean(axis=0)
        se = M.std(axis=0, ddof=1) / np.sqrt(M.shape[0])
        ax.fill_between(times_ref, mean - se, mean + se, color="k", alpha=0.18,
                        linewidth=0, zorder=4)
        ax.plot(times_ref, mean, color="k", lw=2.4, label="Cohort mean $\\pm$ SE",
                zorder=5)
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="k", ls="--", lw=0.7)
        ax.set_xlim(-1, 4)
        ax.set_ylabel("mu ERD/ERS (log-ratio)")
        ax.set_title(f"{name} (n={M.shape[0]})")
    axes[1].set_xlabel("Time from cue (s)")
    axes[0].legend(loc="lower left", fontsize=6.5, ncol=3, columnspacing=0.8,
                   handletextpad=0.3)
    fig.tight_layout()
    _save(fig, "fig_pilot_erd")


# Note: the headline cohort EDS figure (fig_pilot_eds.png) is NOT produced here.
# It is the MI-vs-REST EDS on the decoder's raw, online-matched covariance over
# the 15 motor channels, pooled across the 8 pilots, produced by the
# `consolidateeds` task of scratch_oneshot_subj007_analysis.py
# (ONESHOT_ROOT=...\tiagobot, ONESHOT_POOL_SUBJECTS=PILOT001..008) and copied in.
# The per-class CAR EDS CSVs this script could read are a different (per-class,
# CAR) contrast and are not the decoder-faithful map used in the paper.


# ----------------------------------------------------------------------
# Fig: decision dynamics (lean, time-to-threshold) per participant
# ----------------------------------------------------------------------
def bar_stacked(df, order, out_stem):
    """Single-column, vertically stacked online-feedback bar dynamics: lean
    (top) over time-to-threshold (bottom), per-trial box distributions,
    MI=tab:orange / REST=tab:blue. `order` is a list of (subject_value, label).
    Shared by the healthy cohort and the ALS demonstration so both match."""
    colmap = {"MI": "tab:orange", "REST": "tab:blue"}
    # Width scales with the number of groups so 8 participants stay legible;
    # 2 subjects (ALS) get a narrower box spacing.
    width = 3.4 if len(order) > 4 else 3.0
    fig, axes = plt.subplots(2, 1, figsize=(width, 4.3))
    for ax, metric, ylab in (
            (axes[0], "LeanPct", "Lean toward target (%)"),
            (axes[1], "TimeToThresh_s", "Time to threshold (s)")):
        handles = {}
        for si, (sval, _lab) in enumerate(order):
            d = df[df["subject"] == sval]
            for cls, off in (("MI", -0.18), ("REST", +0.18)):
                v = d[d["Class"] == cls][metric].dropna().values
                if len(v) == 0:
                    continue
                bp = ax.boxplot(v, positions=[si + off], widths=0.3,
                                patch_artist=True, showfliers=True,
                                medianprops=dict(color="black"),
                                flierprops=dict(marker=".", markersize=3,
                                                markerfacecolor=colmap[cls],
                                                markeredgecolor=colmap[cls],
                                                alpha=0.5))
                for box in bp["boxes"]:
                    box.set(facecolor=colmap[cls], alpha=0.55)
                handles[cls] = bp["boxes"][0]
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([lab for _s, lab in order])
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylabel(ylab)
        if handles and ax is axes[0]:
            ax.legend(handles.values(), handles.keys(), loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, out_stem)


def fig_pilot_bar():
    """Healthy-cohort online feedback bar dynamics (single-column, stacked)."""
    frames = []
    for s in PILOTS:
        f = PIC / "bar_dynamics" / f"{s}_S001ONLINE_per_trial_bar_dynamics.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    bar_stacked(df, [(s, LABEL[s]) for s in PILOTS], "fig_pilot_bar")


def main():
    _apply_style()
    print(f"Writing pilot figures to: {OUT_DIR}")
    conf = _load_confusion()
    sub = _subject_pooled(conf)
    print(sub.to_string(index=False))
    sub.to_csv(OUT_DIR / "pilot_decoder_summary.csv", index=False)
    fig_pilot_decoder(sub)
    fig_pilot_confusion(conf)
    fig_pilot_erd()
    fig_pilot_bar()
    # fig_pilot_eds.png is produced by the wrapper's `consolidateeds` task
    # (raw online-matched MI-vs-REST EDS); see note above.
    # Cohort-level numbers for the manuscript text.
    print("\nCohort summary (pooled):")
    print(f"  decision accuracy: mean={sub['dec_acc'].mean():.3f} "
          f"min={sub['dec_acc'].min():.3f} max={sub['dec_acc'].max():.3f}")
    print(f"  total accuracy:    mean={sub['tot_acc'].mean():.3f} "
          f"min={sub['tot_acc'].min():.3f} max={sub['tot_acc'].max():.3f}")
    print(f"  trial kappa:       mean={sub['kappa'].mean():.3f} "
          f"min={sub['kappa'].min():.3f} max={sub['kappa'].max():.3f}")
    print(f"  total trials: {int(sub['trials'].sum())}")
    print("Done.")


if __name__ == "__main__":
    main()
