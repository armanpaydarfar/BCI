#!/usr/bin/env python3
"""Investigation C — EDS cohort vs per-subject mismatch.

QUESTION (per the handoff): the cohort EDS topomap
(`Pictures/clin_analysis/eds/cohort_eds_mi_per_class_mu.png`) shows a
clean C3/CP5/P7/C4 pattern; the per-subject grid
(`per_subject_eds_mi_per_class_mu_grid.png`) shows weak, noisy,
idiosyncratic patterns. Do the per-subject panels visually "average to"
the cohort panel, or is something in the aggregation sharpening the
cohort signal?

CODE-READS THIS SESSION (file:line):
- Analyze_eds_topoplot_CLIN.py:866-918  _plot_per_subject_grid
    Per panel:
      sd = eds_vec.std(ddof=1)
      z  = (eds_vec - eds_vec.mean()) / sd if sd > 0 else zeros
    Each subject is z-scored INDEPENDENTLY across its own channels;
    shared vlim derived from global min/max of all per-subject z-scores
    (line 893-895).
- Analyze_eds_topoplot_CLIN.py:1297-1502  run_for_band_per_class
    Cohort path (lines 1430-1437, 1479-1486):
      stack = np.stack(aligned, axis=0)                 # subjects × ch
      cohort_mean = stack.mean(axis=0)                  # per-channel mean
      z_cohort    = _zscore(cohort_mean)                 # then z-score
      _plot_topomap_panel(z_cohort, ...)
    Each subject's `aligned` vector is itself the MEAN of that subject's
    last 2 sessions (lines 1387-1404) and intersected to cohort_common
    channels (lines 1424-1426).
- Analyze_eds_topoplot_CLIN.py:1411-1413  _zscore(v)
    (v - v.mean()) / v.std(ddof=1)  (zeros if sd == 0)
- Analyze_eds_topoplot_CLIN.py:1619-1621  cohort = CLIN_PRIMARY_SUBJECTS
    SUBJ_002 EXCLUDED by default (`--include-clin002` flag).
- _helpers.py:141  CLIN_PRIMARY_SUBJECTS = [003..008] (6 subjects).

The two operations are mathematically distinct:
  cohort:       z( mean_subj( raw_eds ) )
  per-subject:  z( raw_eds_subj )    independently per subject

If we manually average the per-subject z-scores, we get
  mean_subj( z(raw_eds_subj) )
which is generally NOT the cohort plot — within-subject z removes each
subject's mean and divides by their spread before averaging, while
the cohort z preserves the across-subject signal magnitude.

This script:
  (1) Loads the per-(subj,sess,channel) CSV.
  (2) Reproduces the canonical cohort aggregation step-by-step (last-2
      session average → subject mean → channel z) and verifies the
      result matches `eds_per_class_cohort_summary_mu.csv:cohort_z_score`.
  (3) Computes an alternative aggregation: mean of per-subject
      independently-z-scored EDS (mean_subj(z(raw_eds))). This is the
      "what the eye sees if you average the per-subject panels."
  (4) Plots side-by-side: canonical cohort, manual reconstruction,
      mean-of-per-subject-z, and (for the selection-bias hypothesis)
      a re-run including SUBJ_002. Both classes (MI, REST).

Outputs (scratch):
  C:\\Users\\arman\\Pictures\\clin_analysis_eds_cohort_vs_per_subject\\
    investigation_c_recon_check.txt    (per-channel diff vs canonical CSV)
    cohort_aggregation_compare_{mi,rest}.png  (side-by-side topomaps)
    investigation_c_report.txt
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SWEEP_DIR = _REPO_ROOT / "exploration" / "preprocessing_sweep"
for _p in (str(_REPO_ROOT), str(_SWEEP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

from config import MOTOR_CHANNEL_NAMES  # noqa: E402
from sweep_phase2_round2 import FS  # noqa: E402

CANONICAL_DIR = Path(r"C:\Users\arman\Pictures\clin_analysis\eds")
PER_SUBJ_SESS_CSV = CANONICAL_DIR / "eds_per_class_per_subject_session_mu.csv"
COHORT_SUMMARY_CSV = CANONICAL_DIR / "eds_per_class_cohort_summary_mu.csv"

SCRATCH_DIR = Path(
    r"C:\Users\arman\Pictures\clin_analysis_eds_cohort_vs_per_subject"
)
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

CLIN_PRIMARY_SUBJECTS = [f"CLIN_SUBJ_{i:03d}" for i in range(3, 9)]
MOTOR15 = list(MOTOR_CHANNEL_NAMES)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _zscore(v):
    """Mirror Analyze_eds_topoplot_CLIN.py:1411-1413 _zscore."""
    sd = v.std(ddof=1)
    return np.zeros_like(v) if sd == 0 else (v - v.mean()) / sd


def _subject_avg_last_two(df_subj, class_label):
    """Average a single subject's last-2-sessions EDS per channel,
    intersected to channels common to both sessions.

    Mirrors run_for_band_per_class:1387-1404 (last-2 averaging with
    channel intersection).

    Returns (eds_vec, channels) or (None, None) if not enough sessions
    / no common channels.
    """
    df = df_subj[df_subj["class"] == class_label]
    if df.empty:
        return None, None
    sessions = sorted(df["session"].unique())
    last_two = sessions[-2:]
    # Per session: channels list and EDS values
    per_sess = []
    for s in last_two:
        sub = df[df["session"] == s]
        # Preserve CSV row order (channels list per session is the order
        # rows were written).
        per_sess.append((s, list(sub["channel"]), sub["eds"].to_numpy()))
    if not per_sess:
        return None, None
    common = list(per_sess[0][1])
    for _, ch_list, _ in per_sess[1:]:
        common = [c for c in common if c in ch_list]
    if not common:
        return None, None
    vecs = []
    for _, ch_list, eds in per_sess:
        idx = [ch_list.index(c) for c in common]
        vecs.append(eds[idx])
    avg = np.mean(np.stack(vecs, axis=0), axis=0)
    return avg, common


def _cohort_aggregate(per_subj_eds, mode="canonical"):
    """Aggregate per-subject EDS vectors to a cohort topomap vector.

    `per_subj_eds`: dict subj -> (eds_vec, channels) (last-2 mean).
    `mode`:
        "canonical"       z( mean_subj(raw_eds) )            (cohort plot)
        "mean_of_z"       mean_subj( z(raw_eds_subj) )       (the eye)

    Returns (z_vec, cohort_common_channels).
    """
    if not per_subj_eds:
        return None, []
    cohort_common = list(next(iter(per_subj_eds.values()))[1])
    for _, ch_list in per_subj_eds.values():
        cohort_common = [c for c in cohort_common if c in ch_list]
    if not cohort_common:
        return None, []
    aligned = []
    for subj in sorted(per_subj_eds):
        vec, ch_list = per_subj_eds[subj]
        idx = [ch_list.index(c) for c in cohort_common]
        aligned.append(vec[idx])
    stack = np.stack(aligned, axis=0)  # (n_subj, n_ch)
    if mode == "canonical":
        cohort_mean = stack.mean(axis=0)
        return _zscore(cohort_mean), cohort_common
    if mode == "mean_of_z":
        z_per_subj = np.stack(
            [_zscore(s) for s in stack], axis=0,
        )
        return z_per_subj.mean(axis=0), cohort_common
    raise ValueError(f"Unknown mode {mode}")


def _info_for_channels(channels):
    info = mne.create_info(channels, sfreq=FS, ch_types="eeg")
    info.set_montage(
        mne.channels.make_standard_montage("standard_1020"),
        match_case=True, on_missing="warn",
    )
    return info


def _build_per_subj(df, class_label, cohort_subjects):
    per_subj = {}
    for subj in cohort_subjects:
        df_subj = df[df["subject"] == subj]
        if df_subj.empty:
            continue
        avg, common = _subject_avg_last_two(df_subj, class_label)
        if avg is None:
            continue
        per_subj[subj] = (avg, common)
    return per_subj


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------

def main():
    df = pd.read_csv(PER_SUBJ_SESS_CSV)
    df = df[df["band"] == "mu"]
    print(f"[load] per_subj_session rows: {len(df)} (mu only)")
    print(f"[load] subjects in CSV: "
          f"{sorted(df['subject'].unique())}")
    print(f"[load] sessions per subject:")
    for subj in sorted(df["subject"].unique()):
        print(f"  {subj}: "
              f"{sorted(df[df.subject==subj]['session'].unique())}")
    coh_df = pd.read_csv(COHORT_SUMMARY_CSV)
    print(f"[load] cohort_summary rows: {len(coh_df)}  "
          f"n_subjects col uniq: {sorted(coh_df['n_subjects'].unique())}")

    lines = []

    def w(s=""):
        print(s)
        lines.append(s)

    w("Investigation C — cohort vs per-subject EDS aggregation")
    w("=" * 60)

    # ---- For each class ------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(20, 9),
                             constrained_layout=True)
    for row, class_label in enumerate(("mi", "rest")):
        w(f"\n--- class={class_label.upper()} ---")
        # (A) Default cohort (SUBJ_002 EXCLUDED)
        per_subj_default = _build_per_subj(
            df, class_label, CLIN_PRIMARY_SUBJECTS,
        )
        w(f"  cohort subjects (default, SUBJ_002 EXCLUDED): "
          f"{sorted(per_subj_default)}  "
          f"(n={len(per_subj_default)})")
        z_can, common = _cohort_aggregate(per_subj_default, "canonical")
        z_mean_of_z, common2 = _cohort_aggregate(
            per_subj_default, "mean_of_z",
        )
        assert common == common2
        w(f"  cohort channel intersection (n={len(common)}): {common}")

        # Verify recon matches canonical CSV cohort_z_score
        coh_sub = coh_df[(coh_df["class"] == class_label)
                         & (coh_df["band"] == "mu")]
        canonical_z = {r["channel"]: r["cohort_z_score"]
                       for _, r in coh_sub.iterrows()}
        max_diff = 0.0
        for ch, z in zip(common, z_can):
            if ch in canonical_z:
                d = abs(z - canonical_z[ch])
                max_diff = max(max_diff, d)
        w(f"  RECONSTRUCTION CHECK: max |z_recon - cohort_z_score "
          f"(from CSV)| = {max_diff:.2e}")
        if max_diff < 1e-6:
            w("    PASS — canonical aggregation reproduced exactly.")
        else:
            w("    FAIL — reconstruction mismatches canonical CSV.")

        # (B) cohort including SUBJ_002 (selection-bias test)
        per_subj_with002 = _build_per_subj(
            df, class_label, ["CLIN_SUBJ_002"] + CLIN_PRIMARY_SUBJECTS,
        )
        z_with002, common002 = _cohort_aggregate(
            per_subj_with002, "canonical",
        )
        w(f"  with SUBJ_002:  subjects={sorted(per_subj_with002)}  "
          f"(n={len(per_subj_with002)})  channels(n={len(common002)})")
        if len(per_subj_with002) == len(per_subj_default):
            w("    (SUBJ_002 has no mu rows in CSV; with-002 path is "
              "identical to default — selection bias test trivial.)")

        # Plot 4 panels per class:
        #   (a) canonical cohort z (matches CSV; what the published plot shows)
        #   (b) mean of per-subject z (what the eye averages)
        #   (c) canonical with SUBJ_002 (if available)
        #   (d) per-subject z-vectors stacked — show the per-subject heterogeneity
        for col, (title, vec, ch_list) in enumerate([
            (f"(A) canonical cohort z = z(mean_subj(raw_eds))\n"
             f"n={len(per_subj_default)} subj, "
             f"n_ch={len(common)}  class={class_label.upper()}",
             z_can, common),
            (f"(B) mean_subj(z(raw_eds_subj))\n"
             f"(what averaging per-subject panels gives)",
             z_mean_of_z, common),
            (f"(C) canonical z including CLIN_SUBJ_002\n"
             f"n={len(per_subj_with002)} subj",
             z_with002, common002),
            (f"(D) per-subject raw EDS (one line / subject)\n"
             f"y = raw EDS at each channel",
             None, None),
        ]):
            ax = axes[row][col]
            if title.startswith("(D)"):
                # Plot per-subject raw EDS as line per subject
                for subj in sorted(per_subj_default):
                    vec_s, ch_s = per_subj_default[subj]
                    idx = [ch_s.index(c) for c in common]
                    ax.plot(range(len(common)),
                            vec_s[idx], marker="o",
                            label=subj.replace("CLIN_SUBJ_", "S"),
                            alpha=0.7)
                ax.set_xticks(range(len(common)))
                ax.set_xticklabels(common, rotation=90, fontsize=7)
                ax.set_ylabel("raw EDS")
                ax.legend(fontsize=7, loc="best")
                ax.set_title(title, fontsize=9)
                ax.grid(True, alpha=0.3)
                continue
            if vec is None or len(ch_list) == 0:
                ax.set_title(title + "\n(no data)", fontsize=9)
                continue
            info = _info_for_channels(ch_list)
            vmax = float(np.nanmax(np.abs(vec))) if vec.size else 1.0
            vlim = (-vmax, vmax) if vmax > 0 else (None, None)
            im, _ = mne.viz.plot_topomap(
                vec, info, axes=ax, cmap="viridis",
                names=ch_list, show=False, vlim=vlim,
            )
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.7,
                         label=f"vlim ±{vmax:.2f}")
        # Save per-class side-by-side
        # (we keep both rows in the same fig)

        # Per-channel deltas (canonical vs mean_of_z) printed
        w("  Per-channel (z_canonical, z_mean_of_z, raw_mean):")
        coh_means = (
            coh_df[(coh_df["class"] == class_label)
                   & (coh_df["band"] == "mu")]
            .set_index("channel")["cohort_eds_mean_raw"]
        )
        for ch, zc, zm in zip(common, z_can, z_mean_of_z):
            raw = float(coh_means.get(ch, np.nan))
            w(f"    {ch:4s}  z_can={zc:+.3f}  "
              f"z_mean_of_z={zm:+.3f}  raw_mean={raw:.4e}")

    fig.suptitle("Investigation C — EDS cohort vs per-subject "
                 "aggregation paths (rows: MI, REST)", fontsize=12)
    out_png = SCRATCH_DIR / "cohort_aggregation_compare_mi_rest.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    w(f"\nPlots written: {out_png}")

    # Write log
    log_path = SCRATCH_DIR / "investigation_c_recon_check.txt"
    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nLog written: {log_path}")


if __name__ == "__main__":
    main()
