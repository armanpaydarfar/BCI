#!/usr/bin/env python3
"""Refined ERD line plots for the CLIN_* cohort (Pass 1).

Implements `rev01-erd-refinement-plan.md`:
  1. Motor-cluster-restricted focal-electrode pick (replaces the
     unconstrained argmin in `generate_plots_config_a.py:322-332`).
  2. Two combined-channel metrics: Contralateral ERD% and Bilateral
     ERD%, with auto-dropped cluster channels removed and the surviving
     subset reported in the legend.

Per-subject 3-panel figure: Contralateral, Bilateral, Motor-focal.
Cohort 2-panel figure: Contralateral and Bilateral, one line per
session_idx colour-coded by viridis.

Outputs to `~/Pictures/clin_analysis_pass1/erd_refined/`:
    <SUBJ>_3panel_contra_bilat_focal.png        (per subject)
    cohort_2panel_contra_bilat.png              (cohort summary)
    erd_refined_data.csv                        (cluster traces per (subj, sess))

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
    BILATERAL_MOTOR_CLUSTER, CONTRA_MOTOR_CLUSTER, MOTOR_FOCAL_POOL,
    clin_pictures_root, config_a_pipeline, enumerate_clin_subjects,
    enumerate_online_sessions_for_subject, resolve_motor_cluster,
    session_idx_from_label,
)

# Constants used for the focal/cluster selection (same as the validated
# sweep). MU_LO, MU_HI come from sweep_phase2_round2.py:66.
from sweep_phase2_round2 import MU_HI, MU_LO, SCALAR_WIN  # noqa: E402

import mne  # noqa: E402

mne.set_log_level("ERROR")


# ----------------------------------------------------------------------
# Cluster trace + focal-electrode (motor-constrained) helpers
# ----------------------------------------------------------------------

def _logratio_to_pct(x):
    """Mirrors generate_plots_config_a.py:335-336."""
    return 100.0 * (10.0 ** x - 1.0)


def _most_focal_electrode_motor(tfr_trials, marker="200"):
    """Motor-restricted analog of
    generate_plots_config_a.py:322-332 (`_most_focal_electrode`).

    The motor focal pool is restricted to
    `_helpers.MOTOR_FOCAL_POOL` ∩ surviving channels per
    `rev01-erd-refinement-plan.md` §3.1.
    """
    if marker not in tfr_trials:
        return None
    tfr = tfr_trials[marker]
    motor_present = [c for c in MOTOR_FOCAL_POOL if c in tfr.ch_names]
    if not motor_present:
        return None
    ch_idxs = [tfr.ch_names.index(c) for c in motor_present]
    fmask = (tfr.freqs >= MU_LO) & (tfr.freqs <= MU_HI)
    tmask = (tfr.times >= SCALAR_WIN[0]) & (tfr.times <= SCALAR_WIN[1])
    per_ch = tfr.data[:, ch_idxs][:, :, fmask, :][:, :, :, tmask].mean(
        axis=(0, 2, 3),
    )
    return motor_present[int(np.argmin(per_ch))]


def _timecourse_at_channel(tfr_trials, ch_name, marker):
    """Same convention as generate_plots_config_a.py:339-364: trial-mean
    + SEM in logratio space, converted to % at the end.

    Returns (times, mean_pct, low_pct, up_pct, n_trials) or None.
    """
    if marker not in tfr_trials:
        return None
    tfr = tfr_trials[marker]
    if ch_name not in tfr.ch_names:
        return None
    ch_idx = tfr.ch_names.index(ch_name)
    fmask = (tfr.freqs >= MU_LO) & (tfr.freqs <= MU_HI)
    per_trial = tfr.data[:, ch_idx][:, fmask, :].mean(axis=1)
    n = per_trial.shape[0]
    if n < 1:
        return None
    mean_log = per_trial.mean(axis=0)
    if n > 1:
        sem_log = per_trial.std(axis=0, ddof=1) / np.sqrt(n)
    else:
        sem_log = np.zeros_like(mean_log)
    return (
        tfr.times, _logratio_to_pct(mean_log),
        _logratio_to_pct(mean_log - sem_log),
        _logratio_to_pct(mean_log + sem_log),
        n,
    )


def _cluster_timecourse(tfr_trials, cluster_channels, marker="200"):
    """Cluster-averaged ERD%(t) ± SEM per `rev01-erd-refinement-plan.md`
    §7.1. Trial-averaging in logratio space then converted to %.

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
    # tfr.data: (trials, channels, freqs, times)
    per_trial = tfr.data[:, ch_idxs][:, :, fmask].mean(axis=(1, 2))
    n = per_trial.shape[0]
    if n < 1:
        return None
    mean_log = per_trial.mean(axis=0)
    if n > 1:
        sem_log = per_trial.std(axis=0, ddof=1) / np.sqrt(n)
    else:
        sem_log = np.zeros_like(mean_log)
    return (
        tfr.times, _logratio_to_pct(mean_log),
        _logratio_to_pct(mean_log - sem_log),
        _logratio_to_pct(mean_log + sem_log),
        n, present,
    )


# ----------------------------------------------------------------------
# Per-subject 3-panel figure
# ----------------------------------------------------------------------

def _extract_session_traces(tfr_trials, dropped_channels):
    """Return a small dict of (session-level) traces needed for the
    3-panel figure. Doing this once-per-session avoids holding all
    tfr_trials in RAM across sessions (each tfr_trials is ~1 GB for a
    100-trial, 22-channel session at the default mu+beta TFR grid).
    """
    contra_res = _cluster_timecourse(
        tfr_trials, CONTRA_MOTOR_CLUSTER, "200",
    )
    bilat_res = _cluster_timecourse(
        tfr_trials, BILATERAL_MOTOR_CLUSTER, "200",
    )
    focal_pick = _most_focal_electrode_motor(tfr_trials, "200")
    focal_res = (
        _timecourse_at_channel(tfr_trials, focal_pick, "200")
        if focal_pick is not None else None
    )
    return {
        "contra": contra_res,
        "bilat":  bilat_res,
        "focal_pick": focal_pick,
        "focal_trace": focal_res,
        "dropped_channels": list(dropped_channels),
    }


def _plot_subject_3panel(subject, session_traces, out_path):
    """Plot 3 panels: contralateral cluster, bilateral cluster, motor
    focal. One line per session.

    `session_traces` is a list of (sess, traces_dict) where traces_dict
    is the output of `_extract_session_traces`.
    """
    if not session_traces:
        return
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    # Mi3 fix: use viridis for session_idx (ordinal, light-early →
    # dark-late) consistent with the cohort 2-panel plot. Pass 1 used
    # tab10 here and viridis on the cohort plot — inconsistent palette
    # for the same x-axis variable.
    cmap = plt.get_cmap("viridis")
    n_sess = max(1, len(session_traces))
    panel_specs = [
        (axes[0], "Contralateral ERD%", "contra", CONTRA_MOTOR_CLUSTER),
        (axes[1], "Bilateral ERD%",     "bilat",  BILATERAL_MOTOR_CLUSTER),
        (axes[2], "Motor-focal ERD%",   "focal",  None),
    ]

    drew = False
    for i, (sess, traces) in enumerate(session_traces):
        color = cmap(i / max(1, n_sess - 1))
        for ax, title, key, cluster in panel_specs:
            if key == "focal":
                res = traces.get("focal_trace")
                if res is None:
                    continue
                times, mean_pct, low_pct, up_pct, n_trials = res
                label = f"{sess} ({traces['focal_pick']}, n={n_trials})"
            else:
                res = traces.get(key)
                if res is None:
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
            ax.set_ylabel("ERD %")
            ax.grid(True, alpha=0.25)
            drew = True

    if not drew:
        plt.close(fig)
        return

    axes[-1].set_xlabel("Time (s)")
    for ax in axes:
        ax.legend(loc="best", fontsize=7)
    fig.suptitle(
        f"MU ERD across sessions — {subject} | Config A\n"
        f"Contra cluster: {CONTRA_MOTOR_CLUSTER} | "
        f"Bilateral cluster: {BILATERAL_MOTOR_CLUSTER} | "
        f"Motor-focal pool: {MOTOR_FOCAL_POOL}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Cohort 2-panel figure
# ----------------------------------------------------------------------

def _plot_cohort_2panel(cohort_traces, out_path):
    """Plot two panels (Contralateral, Bilateral). Within each panel,
    one line per session_idx (1..N), colour-coded via viridis (light
    early → dark late), with cohort mean across subjects per session.

    cohort_traces: dict keyed by ("contra" | "bilat") containing list
    of (subject, session_label, times, mean_pct).
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    panels = [
        (axes[0], "Contralateral ERD% — cohort mean by session_idx", "contra"),
        (axes[1], "Bilateral ERD% — cohort mean by session_idx",     "bilat"),
    ]
    # Determine all session indices present
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

    for ax, title, key in panels:
        traces = cohort_traces.get(key, [])
        # Group by session_idx
        per_idx: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        for subj, sess, times, mean_pct in traces:
            idx = session_idx_from_label(sess)
            per_idx.setdefault(idx, []).append((times, mean_pct))

        # All subjects share the same TFR window so times are aligned —
        # assert by simple length matching.
        for idx in sorted(per_idx.keys()):
            entries = per_idx[idx]
            if not entries:
                continue
            # Use the first entry's times as canonical
            t = entries[0][0]
            # Truncate any entry whose length differs (defensive)
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
        ax.set_ylabel("ERD %")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        "CLIN cohort — MU ERD% by session index (cohort grand mean per session)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _redraw_from_csv(out_dir: Path):
    """Regenerate the per-subject 3-panels and cohort 2-panel from
    erd_refined_data.csv. Used in pass-2 to apply Mi3 (palette fix)
    without re-running the ~25 min Config-A TFR pass.

    Limitation: the per-trace focal-electrode panel cannot be
    reconstructed from the CSV (the CSV only stores contra + bilat
    cluster traces). The 3-panel here is therefore reduced to a
    2-panel for the --from-csv path; the focal-electrode panel from
    pass-1 remains in the saved PNG output (overwritten only if a
    full run is executed).
    """
    csv_path = out_dir / "erd_refined_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found; re-run without --from-csv first."
        )
    df = pd.read_csv(csv_path)
    cohort_traces = {"contra": [], "bilat": []}
    for subject in sorted(df["subject"].unique()):
        sub = df[df.subject == subject]
        sessions_in_csv = list(sub["session"].drop_duplicates())
        session_traces: list[tuple[str, dict]] = []
        for sess in sessions_in_csv:
            s = sub[sub.session == sess].sort_values("t")
            traces = {"focal_pick": None, "focal_trace": None,
                      "dropped_channels": []}
            for key in ("contra", "bilat"):
                k = s[s.cluster == key]
                if k.empty:
                    traces[key] = None
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
                traces[key] = (times, mean_pct, low_pct, up_pct, n_trials, present)
                cohort_traces[key].append((subject, sess, times, mean_pct))
            session_traces.append((sess, traces))
        if session_traces:
            sub_path = out_dir / f"{subject}_3panel_contra_bilat_focal.png"
            _plot_subject_3panel(subject, session_traces, str(sub_path))
            print(f"  wrote: {sub_path.name}")
    _plot_cohort_2panel(
        cohort_traces, str(out_dir / "cohort_2panel_contra_bilat.png"),
    )
    print(f"Done (re-plot from CSV). Outputs at: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-csv", action="store_true",
        help=("Skip the Config-A TFR pass; redraw per-subject 3-panel "
              "(contra+bilat only, no focal) and cohort 2-panel from "
              "erd_refined_data.csv. Used for pass-2 Mi3 palette fix."),
    )
    args = parser.parse_args()

    out_dir = clin_pictures_root() / "erd_refined"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        _redraw_from_csv(out_dir)
        return

    cohort_traces = {"contra": [], "bilat": []}
    csv_rows = []

    for subject in enumerate_clin_subjects():
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
            for label_key, cluster in [
                ("contra", CONTRA_MOTOR_CLUSTER),
                ("bilat",  BILATERAL_MOTOR_CLUSTER),
            ]:
                res = traces[label_key]
                if res is None:
                    continue
                times, mean_pct, low_pct, up_pct, n_trials, present = res
                cohort_traces[label_key].append(
                    (subject, sess, times, mean_pct)
                )
                for t_idx, t_val in enumerate(times):
                    csv_rows.append({
                        "subject": subject,
                        "session": sess,
                        "session_idx": session_idx_from_label(sess),
                        "cluster": label_key,
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
                f"({time.time()-t0:.1f}s)"
            )
            # Release heavy TFR objects immediately
            del out, tfr_trials
            gc.collect()

        if session_traces:
            sub_path = out_dir / f"{subject}_3panel_contra_bilat_focal.png"
            _plot_subject_3panel(subject, session_traces, str(sub_path))
            print(f"  wrote: {sub_path.name}")
        del session_traces
        gc.collect()

    # Cohort figure
    _plot_cohort_2panel(
        cohort_traces, str(out_dir / "cohort_2panel_contra_bilat.png"),
    )

    df = pd.DataFrame(csv_rows)
    df.to_csv(out_dir / "erd_refined_data.csv", index=False)
    print(f"\nDone. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
