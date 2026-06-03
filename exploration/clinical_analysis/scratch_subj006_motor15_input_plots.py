#!/usr/bin/env python3
"""Plot the canonical 6-panel ERD figure for CLIN_SUBJ_006 under each
Investigation A variant, from the per-trial npz side-cars on disk.

Reads:
  C:\\Users\\arman\\Pictures\\clin_analysis_subj006_motor15_input\\per_trial\\
    CLIN_SUBJ_006_S00NONLINE_V1.npz   (canonical_full)
    CLIN_SUBJ_006_S00NONLINE_V2.npz   (motor15_input + bilat-8 cluster)
    CLIN_SUBJ_006_S00NONLINE_V3.npz   (motor15_input + motor15-15 cluster)

Writes:
  C:\\Users\\arman\\Pictures\\clin_analysis_subj006_motor15_input\\erd_refined\\
    CLIN_SUBJ_006_6panel_mi_rest_V1_canonical_full.png
    CLIN_SUBJ_006_6panel_mi_rest_V2_motor15-input_bilat-8.png
    CLIN_SUBJ_006_6panel_mi_rest_V3_motor15-input_motor15-15.png

The plotter mirrors Analyze_clinical_erd_refined._plot_subject_6panel
(file:line 445-530) one-to-one, except the suptitle takes the actual
per-variant bilat-cluster list as a parameter so V3's plot honestly
reports "Bilateral: motor15 (15 ch)" instead of the canonical bilat-8.
This is a scratch plotter; the canonical script is untouched.

The per-session legend tag (D1/sustained/band-signal/G1, D4/ES) is
computed from the same evaluate_erd_quality helpers the canonical
plotter uses, so figure annotations and the V1/V2/V3 scorecards cannot
drift.
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
from config import MOTOR_CHANNEL_NAMES  # noqa: E402
from exploration.clinical_analysis._helpers import (  # noqa: E402
    BILATERAL_MOTOR_CLUSTER, CONTRA_MOTOR_CLUSTER, IPSI_MOTOR_CLUSTER,
)


SUBJECT = "CLIN_SUBJ_006"
SESSIONS = [f"S{n:03d}ONLINE" for n in range(1, 6)]
SCRATCH_ROOT = Path(
    r"C:\Users\arman\Pictures\clin_analysis_subj006_motor15_input"
)
PER_TRIAL_DIR = SCRATCH_ROOT / "per_trial"
OUT_DIR = SCRATCH_ROOT / "erd_refined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MOTOR15 = list(MOTOR_CHANNEL_NAMES)

VARIANTS = [
    ("V1", "canonical_full",
     BILATERAL_MOTOR_CLUSTER, "bilat-8",
     "Bilateral cluster: canonical 8 channels."),
    ("V2", "motor15-input_bilat-8",
     BILATERAL_MOTOR_CLUSTER, "bilat-8",
     "motor15 channels picked BEFORE CAR; bilateral cluster = canonical 8 ch."),
    ("V3", "motor15-input_motor15-15",
     MOTOR15, "motor15-15",
     "motor15 channels picked BEFORE CAR; bilateral cluster = motor15 (15 ch)."),
]


# ----------------------------------------------------------------------
# Load a per-trial npz into the same shape the canonical plotter expects
# ----------------------------------------------------------------------

def _reconstruct_traces(npz_path: Path) -> dict:
    """Build the `traces` dict consumed by _plot_subject_6panel from one
    npz side-car. Mirrors `_extract_session_traces` output shape
    (Analyze_clinical_erd_refined.py:325-365):
      key -> (times, mean_pct, low_pct, up_pct, n_trials, present)
      "per_trial" -> {key: {per_trial_pct, times, channels_used}}

    Median/SE math mirrors _cluster_timecourse:296-303.
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


# ----------------------------------------------------------------------
# Plotter — line-for-line mirror of _plot_subject_6panel:445-530, except
# bilat_cluster and a variant_note are taken as parameters so the
# suptitle accurately reflects what was plotted for each variant.
# ----------------------------------------------------------------------

def _plot_subject_6panel_variant(subject, session_traces, out_path,
                                 bilat_cluster, variant_label,
                                 variant_note):
    if not session_traces:
        return
    fig, axes = plt.subplots(
        3, 2, figsize=(14, 11), sharex=True, sharey="row",
    )
    cmap = plt.get_cmap("viridis")
    n_sess = max(1, len(session_traces))
    panel_specs = [
        (0, "mi",   "Contralateral ERD% — MI",
         "contra_mi",   CONTRA_MOTOR_CLUSTER),
        (0, "rest", "Contralateral ERD% — REST",
         "contra_rest", CONTRA_MOTOR_CLUSTER),
        (1, "mi",   "Bilateral ERD% — MI",
         "bilat_mi",    bilat_cluster),
        (1, "rest", "Bilateral ERD% — REST",
         "bilat_rest",  bilat_cluster),
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
            ptp = traces.get("per_trial", {}).get(key)
            ptp_arr = ptp["per_trial_pct"] if ptp else None
            score_tag = _panel_score_tag(cls, times, mean_pct, ptp_arr)
            label = f"{sess} (n={n_trials}; {tag}){score_tag}"
            ax.plot(times, mean_pct, color=color, label=label, linewidth=1.4)
            ax.fill_between(times, low_pct, up_pct, color=color, alpha=0.15)
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
        f"[INVESTIGATION A {variant_label}]\n"
        f"Contra: {CONTRA_MOTOR_CLUSTER} | "
        f"Bilateral ({len(bilat_cluster)} ch): {bilat_cluster} | "
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


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    print(f"[setup] subject={SUBJECT}")
    print(f"[setup] per_trial_dir={PER_TRIAL_DIR}")
    print(f"[setup] out_dir={OUT_DIR}")

    for tag, suffix, bilat_cluster, bilat_label, note in VARIANTS:
        session_traces = []
        for sess in SESSIONS:
            npz_path = PER_TRIAL_DIR / f"{SUBJECT}_{sess}_{tag}.npz"
            if not npz_path.exists():
                print(f"  [{tag}] {sess}: missing {npz_path.name}; skip")
                continue
            traces = _reconstruct_traces(npz_path)
            session_traces.append((sess, traces))

        out_path = OUT_DIR / f"{SUBJECT}_6panel_mi_rest_{tag}_{suffix}.png"
        _plot_subject_6panel_variant(
            SUBJECT, session_traces, out_path,
            bilat_cluster, f"{tag} ({suffix})", note,
        )
        print(f"  [{tag}] wrote: {out_path.name} "
              f"(n={len(session_traces)} sessions)")

    print(f"\nDone. Plots at {OUT_DIR}")


if __name__ == "__main__":
    main()
