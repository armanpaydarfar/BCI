#!/usr/bin/env python3
"""Per-subject 6-panel plotter for the Phase 1 cohort cap sweep.

Reads per-trial npz side-cars under
  C:\\Users\\arman\\Pictures\\clin_analysis_cohort_cap_sweep\\per_trial\\
    {SUBJECT}_{session}_{variant_tag}.npz

and renders one canonical 6-panel figure per subject per variant_tag,
mirroring Analyze_clinical_erd_refined._plot_subject_6panel:445-530
line-for-line (median + % space; that is the rubric substrate).

Used as a library by scratch_cohort_cap_sweep.py (Stage F). May also be
run as a standalone CLI after Phase 1 to regenerate plots:

    python scratch_cohort_cap_sweep_plots.py cap200
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SWEEP_DIR = _REPO_ROOT / "exploration" / "preprocessing_sweep"
for _p in (str(_REPO_ROOT), str(_SWEEP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

from Analyze_clinical_erd_refined import (  # noqa: E402
    _BAND_LABEL, _panel_score_tag, _preproc_caption,
)
from exploration.clinical_analysis._helpers import (  # noqa: E402
    BILATERAL_MOTOR_CLUSTER, CONTRA_MOTOR_CLUSTER, IPSI_MOTOR_CLUSTER,
    enumerate_clin_subjects, enumerate_online_sessions_for_subject,
)


SCRATCH_ROOT = Path(r"C:\Users\arman\Pictures\clin_analysis_cohort_cap_sweep")
DEFAULT_PER_TRIAL_DIR = SCRATCH_ROOT / "per_trial"
DEFAULT_FIGS_DIR = SCRATCH_ROOT / "erd_refined"


def _reconstruct_traces(npz_path: Path) -> dict:
    """Build the `traces` dict consumed by the 6-panel plotter from a
    per-trial npz side-car. Median/SE math mirrors
    Analyze_clinical_erd_refined._cluster_timecourse:296-303.
    """
    z = np.load(npz_path)
    present_keys = [k for k in str(z["keys"]).split(",") if k]
    dropped = [c for c in str(z["dropped_channels"]).split(",") if c]
    traces: dict = {"dropped_channels": dropped}
    per_trial: dict = {}
    for key in present_keys:
        ptp = np.asarray(z[f"{key}__ptp"], dtype=np.float64)
        times = np.asarray(z[f"{key}__times"], dtype=np.float64)
        channels = [c for c in str(z[f"{key}__channels"]).split(",") if c]
        n = int(ptp.shape[0])
        if n == 0:
            traces[key] = None
            continue
        mean_pct = np.median(ptp, axis=0)
        if n > 1:
            sem = np.std(ptp, axis=0, ddof=1) / np.sqrt(n)
            low_pct = mean_pct - sem
            up_pct = mean_pct + sem
        else:
            low_pct = mean_pct.copy()
            up_pct = mean_pct.copy()
        traces[key] = (times, mean_pct, low_pct, up_pct, n, channels)
        per_trial[key] = {
            "per_trial_pct": ptp,
            "times": times,
            "channels_used": channels,
        }
    traces["per_trial"] = per_trial
    return traces


def _plot_subject_6panel(subject, session_traces, out_path,
                         variant_label, variant_note):
    """Line-for-line mirror of Analyze_clinical_erd_refined.
    _plot_subject_6panel:445-530, with a variant_label/note in the
    suptitle so different cap sweeps are visually distinguishable."""
    if not session_traces:
        return
    fig, axes = plt.subplots(
        3, 2, figsize=(14, 11), sharex=True, sharey="row",
    )
    cmap = plt.get_cmap("viridis")
    n_sess = max(1, len(session_traces))
    panel_specs = [
        (0, "mi",   "Contralateral ERD% — MI",   "contra_mi",
         CONTRA_MOTOR_CLUSTER),
        (0, "rest", "Contralateral ERD% — REST", "contra_rest",
         CONTRA_MOTOR_CLUSTER),
        (1, "mi",   "Bilateral ERD% — MI",       "bilat_mi",
         BILATERAL_MOTOR_CLUSTER),
        (1, "rest", "Bilateral ERD% — REST",     "bilat_rest",
         BILATERAL_MOTOR_CLUSTER),
        (2, "mi",   "Ipsilateral ERD% — MI",     "ipsi_mi",
         IPSI_MOTOR_CLUSTER),
        (2, "rest", "Ipsilateral ERD% — REST",   "ipsi_rest",
         IPSI_MOTOR_CLUSTER),
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
            ptp = traces.get("per_trial", {}).get(key)
            ptp_arr = ptp["per_trial_pct"] if ptp else None
            score_tag = _panel_score_tag(cls, times, mean_pct, ptp_arr)
            label = f"{sess} (n={n_trials}; {tag}){score_tag}"
            ax.plot(times, mean_pct, color=color, label=label,
                    linewidth=1.4)
            ax.fill_between(times, low_pct, up_pct, color=color,
                            alpha=0.15)
            ax.set_title(title)
            ax.axhline(0, color="k", lw=0.6)
            ax.axvline(0, color="k", ls="--", lw=0.7)
            ax.axvline(1.0, color="k", ls=":", lw=0.7)
            ax.grid(True, alpha=0.25)
            drew = True
        axes[row][0].set_ylabel("ERD %")

    if not drew:
        plt.close(fig)
        return

    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    for row in range(3):
        axes[row][0].legend(loc="best", fontsize=7)
        axes[row][1].legend(loc="best", fontsize=7)
    fig.suptitle(
        f"MU ERD across sessions — {subject} | MI vs REST  "
        f"[Phase 1 {variant_label}]\n"
        f"Contra: {CONTRA_MOTOR_CLUSTER} | "
        f"Bilateral ({len(BILATERAL_MOTOR_CLUSTER)} ch): "
        f"{BILATERAL_MOTOR_CLUSTER} | "
        f"Ipsi: {IPSI_MOTOR_CLUSTER}\n"
        f"{_preproc_caption()} | {_BAND_LABEL}\n"
        f"{variant_note}\n"
        "legend tags: D1=MI strength, sus=sustained frac, b/s=band/signal, "
        "G1!=retained outlier; D4=REST specificity, ES!=eyes-closed",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_cohort_for_variant(variant_tag: str, subjects: list[str],
                            per_trial_dir: Path, figs_dir: Path,
                            variant_note: str):
    """Generate per-subject 6-panel figures for one variant. Used by the
    Phase 1 driver as a library call."""
    figs_dir.mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        sessions = enumerate_online_sessions_for_subject(subject)
        session_traces = []
        for sess in sessions:
            npz = per_trial_dir / f"{subject}_{sess}_{variant_tag}.npz"
            if not npz.exists():
                continue
            session_traces.append((sess, _reconstruct_traces(npz)))
        if not session_traces:
            continue
        out_path = figs_dir / (
            f"{subject}_6panel_mi_rest_{variant_tag}.png"
        )
        _plot_subject_6panel(
            subject, session_traces, out_path,
            variant_label=variant_tag, variant_note=variant_note,
        )
        print(f"    [{variant_tag}] wrote: {out_path.name} "
              f"(n={len(session_traces)} sessions)")


def main():
    if len(sys.argv) < 2:
        print("Usage: scratch_cohort_cap_sweep_plots.py <variant_tag>")
        print("  e.g. scratch_cohort_cap_sweep_plots.py cap200")
        sys.exit(2)
    variant_tag = sys.argv[1]
    plot_cohort_for_variant(
        variant_tag, enumerate_clin_subjects(),
        DEFAULT_PER_TRIAL_DIR, DEFAULT_FIGS_DIR,
        variant_note=f"variant={variant_tag} (CLI standalone run)",
    )


if __name__ == "__main__":
    main()
