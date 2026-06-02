#!/usr/bin/env python3
"""Refined ERD line plots for the CLIN_* cohort (Pass 1).

Three cluster-averaged ERD% metrics — Contralateral, Bilateral, and
Ipsilateral — with auto-dropped cluster channels removed and the
surviving subset reported in the legend.

Per-subject 6-panel figure: 3 rows (Contralateral, Bilateral,
Ipsilateral) x 2 cols (MI, Rest). Same y-axis per row so MI and Rest
are directly comparable on the same scale.

Cohort 6-panel figure: 3 rows (Contralateral, Bilateral, Ipsilateral)
x 2 cols (MI, Rest), one line per session_idx colour-coded by viridis.

Every figure carries a one-line preprocessing caption (spatial filter,
blink handling, mu band, baseline window) so the scheme behind the
plot is always visible.

Outputs to `~/Pictures/clin_analysis/erd_refined/`:
    <SUBJ>_6panel_mi_rest.png    (per subject)
    cohort_6panel_mi_rest.png    (cohort summary)
    erd_refined_data.csv         (cluster traces per (subj, sess, marker))

Analysis-only. No Tier 1 / Tier 2 writes.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
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
    BILATERAL_MOTOR_CLUSTER, CONTRA_MOTOR_CLUSTER, IPSI_MOTOR_CLUSTER,
    clin_pictures_root, enumerate_clin_subjects,
    enumerate_online_sessions_for_subject, resolve_motor_cluster,
    session_idx_from_label,
)

# Mu-band edges for the cluster ERD% average (same as the validated
# sweep). MU_LO, MU_HI come from sweep_phase2_round2.py:66.
from sweep_phase2_round2 import MU_HI, MU_LO  # noqa: E402

# Override Config A's spectral baseline to be contiguous to the cue
# (-1, 0) s. The helper's CONFIG_A uses (-1.5, -0.25) — chosen by the
# sweep to leave a quarter-second gap before the cue — but the
# downstream `tfr.crop(tmin=-1, tmax=4)` at
# `generate_plots_config_a.py:151` then hides the (-1.5, -1) portion
# from the displayed window. That asymmetry leaves the visible
# baseline offset by whatever average the hidden window held. Setting
# the baseline equal to the visible pre-cue window puts the baseline
# at 0 on the plot by construction.
CONFIG_A_DISPLAY_BASELINE = {
    "spatial_filter":    "car",
    "blink_removal":     "drop_fp",
    "baseline_mode":     "logratio",
    "spectral_baseline": (-1.0, 0.0),
    # Per-session channel qualification: railing electrodes (e.g.
    # CLIN_SUBJ_004 S002 P7/FC2, S005 FC5/T8) drive the wide median±SE
    # bands. Detect via broadband MAD-z > 3.5 and spherical-spline
    # interpolate before spatial filtering. reject_window restricts the
    # 50µV epoch reject to the displayed (-1, 4) s window. Scoped to this
    # ERD config only — neuromod (_helpers.CONFIG_A) and the topomap
    # (generate_plots_config_a.CONFIG_A) configs leave these keys unset
    # and so are unaffected.
    "channel_qualify":   True,
    "channel_qualify_z": 3.5,
    "reject_window":     (-1.0, 4.0),
}

import mne  # noqa: E402


def config_a_pipeline(subject, session):
    """Run Config A preprocess + TFR with a display-matched baseline.

    Local override of `_helpers.config_a_pipeline` — same Config A
    (CAR, drop_fp, logratio) but with `spectral_baseline=(-1, 0)` so
    the baseline window matches the visible pre-cue display window.
    """
    from generate_plots_config_a import preprocess_and_tfr
    return preprocess_and_tfr(subject, session, CONFIG_A_DISPLAY_BASELINE)

mne.set_log_level("ERROR")


MI_MARKER = "200"
REST_MARKER = "100"

# Description of the shaded band drawn around each median trace.
_BAND_LABEL = "shaded = median ± SE (std/√n across trials)"


def _preproc_caption():
    """One-line preprocessing caption rendered on every figure. Read
    from the live CONFIG at call time so it reflects the runtime
    `--spatial-filter` override.
    """
    cfg = CONFIG_A_DISPLAY_BASELINE
    caption = (
        f"Preproc: {cfg['spatial_filter'].upper()} spatial filter | "
        f"blink={cfg['blink_removal']} | "
        f"μ {MU_LO:g}–{MU_HI:g} Hz | "
        f"baseline {cfg['spectral_baseline']} s"
    )
    if cfg.get("channel_qualify"):
        rw = cfg.get("reject_window")
        rw_txt = f" | reject {rw[0]:g}..{rw[1]:g}s" if rw else ""
        caption += (
            f" | chan-qualify: MAD-z>{cfg['channel_qualify_z']:g} (interp)"
            f"{rw_txt}"
        )
    return caption


# ----------------------------------------------------------------------
# Cluster trace helpers
# ----------------------------------------------------------------------

def _logratio_to_pct(x):
    """Mirrors generate_plots_config_a.py:335-336."""
    return 100.0 * (10.0 ** x - 1.0)


def _cluster_timecourse(tfr_trials, cluster_channels, marker="200"):
    """Cluster-averaged ERD%(t), median ± SE across trials.

    Median (not mean) across trials: the per-trial arithmetic mean of
    P/P_bl is outlier-sensitive in clinical data — one high-ERS trial
    can flip the post-cue average positive even when most trials show
    desync. Median isolates the typical-trial response. The shaded band
    is the standard error of the per-trial distribution (sample
    std / sqrt(n)), centred on the median. Baseline sits at 0 by
    construction (each trial's pct over the baseline window averages to
    0 per the apply_baseline guarantee, and the median of zeros is 0).

    Returns (times, mean_pct, low_pct, up_pct, n_trials, surviving_channels)
    or None.
    """
    if marker not in tfr_trials:
        return None
    tfr = tfr_trials[marker]
    present = [c for c in cluster_channels if c in tfr.ch_names]
    if not present:
        return None
    ch_idxs = [tfr.ch_names.index(c) for c in present]
    fmask = (tfr.freqs >= MU_LO) & (tfr.freqs <= MU_HI)
    data_pct = _logratio_to_pct(
        tfr.data[:, ch_idxs][:, :, fmask],
    )
    n = data_pct.shape[0]
    if n < 1:
        return None
    per_trial_pct = data_pct.mean(axis=(1, 2))  # over cluster ch + mu freqs
    mean_pct = np.median(per_trial_pct, axis=0)
    if n > 1:
        sem = np.std(per_trial_pct, axis=0, ddof=1) / np.sqrt(n)
        low_pct = mean_pct - sem
        up_pct = mean_pct + sem
    else:
        low_pct = mean_pct.copy()
        up_pct = mean_pct.copy()
    return (
        tfr.times, mean_pct,
        low_pct, up_pct,
        n, present,
    )


# ----------------------------------------------------------------------
# Per-subject 6-panel figure (3 metric rows × 2 class cols)
# ----------------------------------------------------------------------

def _extract_session_traces(tfr_trials, dropped_channels):
    """Return a small dict of (session-level) traces needed for the
    6-panel figure (3 cluster metrics × {MI, REST}). Doing this
    once-per-session avoids holding all tfr_trials in RAM across
    sessions (each tfr_trials is ~1 GB for a 100-trial, 22-channel
    session at the default mu+beta TFR grid).
    """
    contra_mi = _cluster_timecourse(
        tfr_trials, CONTRA_MOTOR_CLUSTER, MI_MARKER,
    )
    contra_rest = _cluster_timecourse(
        tfr_trials, CONTRA_MOTOR_CLUSTER, REST_MARKER,
    )
    bilat_mi = _cluster_timecourse(
        tfr_trials, BILATERAL_MOTOR_CLUSTER, MI_MARKER,
    )
    bilat_rest = _cluster_timecourse(
        tfr_trials, BILATERAL_MOTOR_CLUSTER, REST_MARKER,
    )
    ipsi_mi = _cluster_timecourse(
        tfr_trials, IPSI_MOTOR_CLUSTER, MI_MARKER,
    )
    ipsi_rest = _cluster_timecourse(
        tfr_trials, IPSI_MOTOR_CLUSTER, REST_MARKER,
    )
    return {
        "contra_mi":   contra_mi,
        "contra_rest": contra_rest,
        "bilat_mi":    bilat_mi,
        "bilat_rest":  bilat_rest,
        "ipsi_mi":     ipsi_mi,
        "ipsi_rest":   ipsi_rest,
        "dropped_channels": list(dropped_channels),
    }


def _plot_subject_6panel(subject, session_traces, out_path):
    """Plot 3 rows (Contra, Bilat, Ipsi) × 2 cols (MI, REST).

    Y-axis is shared within each row so MI and REST are on the same
    scale for direct visual comparison. X-axis is shared across all
    panels.
    """
    if not session_traces:
        return
    fig, axes = plt.subplots(
        3, 2, figsize=(14, 11), sharex=True, sharey="row",
    )
    cmap = plt.get_cmap("viridis")
    n_sess = max(1, len(session_traces))
    # (row, class) -> (title, key_in_traces, cluster_for_legend)
    panel_specs = [
        (0, "mi",   "Contralateral ERD% — MI",
         "contra_mi",   CONTRA_MOTOR_CLUSTER),
        (0, "rest", "Contralateral ERD% — REST",
         "contra_rest", CONTRA_MOTOR_CLUSTER),
        (1, "mi",   "Bilateral ERD% — MI",
         "bilat_mi",    BILATERAL_MOTOR_CLUSTER),
        (1, "rest", "Bilateral ERD% — REST",
         "bilat_rest",  BILATERAL_MOTOR_CLUSTER),
        (2, "mi",   "Ipsilateral ERD% — MI",
         "ipsi_mi",     IPSI_MOTOR_CLUSTER),
        (2, "rest", "Ipsilateral ERD% — REST",
         "ipsi_rest",   IPSI_MOTOR_CLUSTER),
    ]
    col_of_class = {"mi": 0, "rest": 1}

    drew = False
    for i, (sess, traces) in enumerate(session_traces):
        color = cmap(i / max(1, n_sess - 1))
        for row, cls, title, key, cluster in panel_specs:
            ax = axes[row][col_of_class[cls]]
            res = traces.get(key)
            if res is None:
                ax.set_title(title)
                continue
            times, mean_pct, low_pct, up_pct, n_trials, present = res
            missing = [c for c in cluster if c not in present]
            tag = ", ".join(present)
            if missing:
                tag += f"  [missing: {','.join(missing)}]"
            label = f"{sess} (n={n_trials}; {tag})"
            ax.plot(times, mean_pct, color=color, label=label, linewidth=1.4)
            ax.fill_between(times, low_pct, up_pct, color=color, alpha=0.15)
            ax.set_title(title)
            ax.axhline(0, color="k", lw=0.6)
            ax.axvline(0, color="k", ls="--", lw=0.7)
            ax.axvline(1.0, color="k", ls=":", lw=0.7)
            ax.grid(True, alpha=0.25)
            drew = True
        # y-label on the left column only (sharey="row" mirrors the
        # tick labels)
        axes[row][0].set_ylabel("ERD %")

    if not drew:
        plt.close(fig)
        return

    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    # Legends only on the left column to keep the figure readable
    # (legends are identical between MI and REST columns: same sessions,
    # same colours; only the per-trial counts differ).
    for row in range(3):
        axes[row][0].legend(loc="best", fontsize=7)
        axes[row][1].legend(loc="best", fontsize=7)
    fig.suptitle(
        f"MU ERD across sessions — {subject} | MI vs REST\n"
        f"Contra: {CONTRA_MOTOR_CLUSTER} | "
        f"Bilateral: {BILATERAL_MOTOR_CLUSTER} | "
        f"Ipsi: {IPSI_MOTOR_CLUSTER}\n"
        f"{_preproc_caption()} | {_BAND_LABEL}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Cohort 4-panel figure (2 cluster rows × 2 class cols)
# ----------------------------------------------------------------------

def _plot_cohort_6panel(cohort_traces, out_path):
    """Plot 3 rows (Contralateral, Bilateral, Ipsilateral) × 2 cols
    (MI, REST).

    Within each panel, one line per session_idx (1..N) colour-coded
    via viridis (light early → dark late), showing the cohort grand
    mean across subjects per session.

    cohort_traces: dict keyed by ("contra_mi" | "contra_rest" |
    "bilat_mi" | "bilat_rest" | "ipsi_mi" | "ipsi_rest") containing
    list of (subject, session_label, times, mean_pct).
    """
    fig, axes = plt.subplots(
        3, 2, figsize=(14, 11), sharex=True, sharey="row",
    )
    panels = [
        (0, 0, "Contralateral ERD% — MI",   "contra_mi"),
        (0, 1, "Contralateral ERD% — REST", "contra_rest"),
        (1, 0, "Bilateral ERD% — MI",       "bilat_mi"),
        (1, 1, "Bilateral ERD% — REST",     "bilat_rest"),
        (2, 0, "Ipsilateral ERD% — MI",     "ipsi_mi"),
        (2, 1, "Ipsilateral ERD% — REST",   "ipsi_rest"),
    ]
    # Determine all session indices present (across all four panels)
    all_idxs = sorted({
        session_idx_from_label(sess)
        for traces in cohort_traces.values()
        for (_, sess, _, _) in traces
    })
    cmap = plt.get_cmap("viridis")
    if not all_idxs:
        plt.close(fig)
        return
    colors = {
        idx: cmap((i + 0.0) / max(1, len(all_idxs) - 1))
        for i, idx in enumerate(all_idxs)
    }

    for row, col, title, key in panels:
        ax = axes[row][col]
        traces = cohort_traces.get(key, [])
        per_idx: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        for subj, sess, times, mean_pct in traces:
            idx = session_idx_from_label(sess)
            per_idx.setdefault(idx, []).append((times, mean_pct))

        for idx in sorted(per_idx.keys()):
            entries = per_idx[idx]
            if not entries:
                continue
            t = entries[0][0]
            stack = np.stack(
                [e[1][:len(t)] for e in entries if len(e[1]) >= len(t)],
                axis=0,
            ) if entries else None
            if stack is None or stack.size == 0:
                continue
            mean_pct = stack.mean(axis=0)
            ax.plot(
                t, mean_pct, color=colors[idx],
                label=f"S{idx:03d} (n={stack.shape[0]} subj)",
                linewidth=1.6,
            )
        ax.set_title(title)
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="k", ls="--", lw=0.7)
        ax.axvline(1.0, color="k", ls=":", lw=0.7)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        if col == 0:
            ax.set_ylabel("ERD %")
        if row == 2:
            ax.set_xlabel("Time (s)")
    fig.suptitle(
        "CLIN cohort — MU ERD% by session index | MI vs REST "
        "(cohort grand mean per session)\n"
        f"{_preproc_caption()}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _redraw_from_csv(out_dir: Path):
    """Regenerate the per-subject 6-panel and cohort 6-panel from
    erd_refined_data.csv. Used to replot without re-running the
    ~25 min Config-A TFR pass.

    Backwards compat: if the CSV lacks a `marker` column it is from
    an older (MI-only) run; treat every row as MI and skip Rest.
    """
    csv_path = out_dir / "erd_refined_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found; re-run without --from-csv first."
        )
    df = pd.read_csv(csv_path)
    has_marker = "marker" in df.columns
    if not has_marker:
        print(
            "  [warn] CSV has no `marker` column (old schema); "
            "treating all rows as MI. Rerun without --from-csv to "
            "produce Rest panels."
        )
        df["marker"] = "mi"
    cohort_traces = {
        "contra_mi": [], "contra_rest": [],
        "bilat_mi":  [], "bilat_rest":  [],
        "ipsi_mi":   [], "ipsi_rest":   [],
    }
    cluster_to_key = {"contra": "contra", "bilat": "bilat", "ipsi": "ipsi"}
    for subject in sorted(df["subject"].unique()):
        sub = df[df.subject == subject]
        sessions_in_csv = list(sub["session"].drop_duplicates())
        session_traces: list[tuple[str, dict]] = []
        for sess in sessions_in_csv:
            s = sub[sub.session == sess].sort_values("t")
            traces = {
                "contra_mi":  None, "contra_rest": None,
                "bilat_mi":   None, "bilat_rest":  None,
                "ipsi_mi":    None, "ipsi_rest":   None,
                "dropped_channels": [],
            }
            for cluster in ("contra", "bilat", "ipsi"):
                for marker in ("mi", "rest"):
                    k = s[(s.cluster == cluster) & (s.marker == marker)]
                    if k.empty:
                        continue
                    times = k["t"].to_numpy(dtype=float)
                    mean_pct = k["mean_pct"].to_numpy(dtype=float)
                    low_pct = k["low_pct"].to_numpy(dtype=float)
                    up_pct = k["up_pct"].to_numpy(dtype=float)
                    n_trials = int(k["n_trials"].iloc[0])
                    present_str = k["channels_used"].iloc[0]
                    present = present_str.split(",") if isinstance(
                        present_str, str,
                    ) and present_str else []
                    key = f"{cluster_to_key[cluster]}_{marker}"
                    traces[key] = (
                        times, mean_pct, low_pct, up_pct, n_trials, present,
                    )
                    cohort_traces[key].append(
                        (subject, sess, times, mean_pct),
                    )
            session_traces.append((sess, traces))
        if session_traces:
            sub_path = out_dir / f"{subject}_6panel_mi_rest.png"
            _plot_subject_6panel(subject, session_traces, str(sub_path))
            print(f"  wrote: {sub_path.name}")
    _plot_cohort_6panel(
        cohort_traces, str(out_dir / "cohort_6panel_mi_rest.png"),
    )
    print(f"Done (re-plot from CSV). Outputs at: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-csv", action="store_true",
        help=("Skip the Config-A TFR pass; redraw per-subject 6-panel "
              "and cohort 6-panel (contra/bilat/ipsi) from "
              "erd_refined_data.csv."),
    )
    parser.add_argument(
        "--subjects", default="",
        help=("Comma-separated subject filter for smoke tests, e.g. "
              "`CLIN_SUBJ_005,CLIN_SUBJ_007`. Empty = full cohort. "
              "Skips the per-subject cohort accumulation for subjects "
              "outside the filter so this only meaningfully reduces "
              "runtime when one or two subjects are specified."),
    )
    parser.add_argument(
        "--spatial-filter", choices=("car", "csd", "hjorth"), default="car",
        help=("Spatial filter for the Config-A preprocessing pass. "
              "Output filenames are tagged with the filter name so "
              "car/csd/hjorth variants coexist in erd_refined/ for "
              "side-by-side review."),
    )
    args = parser.parse_args()
    # Apply the spatial-filter override to the shared local CONFIG; the
    # figure caption reads this at plot time so it always matches.
    CONFIG_A_DISPLAY_BASELINE["spatial_filter"] = args.spatial_filter
    subject_filter: set[str] = {
        s.strip() for s in args.subjects.split(",") if s.strip()
    }
    # Tag outputs by spatial filter (always) so the three variants
    # coexist, plus by subject when a --subjects filter is set so
    # smoke tests don't overwrite the full-cohort files.
    variant_tag = f"_{args.spatial_filter}"
    if subject_filter:
        variant_tag += "_subj-" + "-".join(
            s.replace("CLIN_SUBJ_", "") for s in sorted(subject_filter)
        )
        print(f"[variant] subjects={sorted(subject_filter)} → tag '{variant_tag}'")

    out_dir = clin_pictures_root() / "erd_refined"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        _redraw_from_csv(out_dir)
        return

    cohort_traces = {
        "contra_mi": [], "contra_rest": [],
        "bilat_mi":  [], "bilat_rest":  [],
        "ipsi_mi":   [], "ipsi_rest":   [],
    }
    csv_rows = []

    # Map (cluster_label, marker_label) -> (key_in_traces_dict, cluster_list)
    cluster_specs = [
        ("contra", "mi",   "contra_mi",   CONTRA_MOTOR_CLUSTER),
        ("contra", "rest", "contra_rest", CONTRA_MOTOR_CLUSTER),
        ("bilat",  "mi",   "bilat_mi",    BILATERAL_MOTOR_CLUSTER),
        ("bilat",  "rest", "bilat_rest",  BILATERAL_MOTOR_CLUSTER),
        ("ipsi",   "mi",   "ipsi_mi",     IPSI_MOTOR_CLUSTER),
        ("ipsi",   "rest", "ipsi_rest",   IPSI_MOTOR_CLUSTER),
    ]

    for subject in enumerate_clin_subjects():
        if subject_filter and subject not in subject_filter:
            continue
        sessions = enumerate_online_sessions_for_subject(subject)
        print(f"\n=== {subject} ({len(sessions)} sessions) ===")
        # Hold only the small extracted traces per session (not the
        # full TFR objects), to keep peak RSS modest on a 16 GB box.
        session_traces: list[tuple[str, dict]] = []
        for sess in sessions:
            t0 = time.time()
            try:
                out = config_a_pipeline(subject, sess)
            except Exception as e:
                print(
                    f"  {sess}: FAILED ({type(e).__name__}: {e})"
                )
                continue
            tfr_trials = out["tfr_trials"]
            traces = _extract_session_traces(
                tfr_trials, out.get("dropped_channels", []),
            )
            session_traces.append((sess, traces))

            # Cluster-mean traces for the cohort figure + CSV
            for cluster_label, marker_label, key, cluster_chs in cluster_specs:
                res = traces[key]
                if res is None:
                    continue
                times, mean_pct, low_pct, up_pct, n_trials, present = res
                cohort_traces[key].append(
                    (subject, sess, times, mean_pct)
                )
                for t_idx, t_val in enumerate(times):
                    csv_rows.append({
                        "subject": subject,
                        "session": sess,
                        "session_idx": session_idx_from_label(sess),
                        "cluster": cluster_label,
                        "marker": marker_label,
                        "channels_used": ",".join(present),
                        "t": float(t_val),
                        "mean_pct": float(mean_pct[t_idx]),
                        "low_pct": float(low_pct[t_idx]),
                        "up_pct": float(up_pct[t_idx]),
                        "n_trials": int(n_trials),
                    })
            print(
                f"  {sess}: n_kept={out['n_kept']}/{out['n_attempted']} "
                f"dropped={out['dropped_channels'] or '—'} "
                f"interp={out.get('interpolated_channels') or '—'} "
                f"({time.time()-t0:.1f}s)"
            )
            # Release heavy TFR objects immediately
            del out, tfr_trials
            gc.collect()

        if session_traces:
            sub_path = (
                out_dir / f"{subject}_6panel_mi_rest{variant_tag}.png"
            )
            _plot_subject_6panel(subject, session_traces, str(sub_path))
            print(f"  wrote: {sub_path.name}")
        del session_traces
        gc.collect()

    # Cohort figure
    _plot_cohort_6panel(
        cohort_traces,
        str(out_dir / f"cohort_6panel_mi_rest{variant_tag}.png"),
    )

    df = pd.DataFrame(csv_rows)
    df.to_csv(out_dir / f"erd_refined_data{variant_tag}.csv", index=False)
    print(f"\nDone. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
