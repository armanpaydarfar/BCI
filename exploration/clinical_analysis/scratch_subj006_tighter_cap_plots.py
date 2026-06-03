#!/usr/bin/env python3
"""Plot the canonical 6-panel ERD figure for CLIN_SUBJ_006 under each
Investigation D variant, from the per-trial npz side-cars on disk.

Reads:
  C:\\Users\\arman\\Pictures\\clin_analysis_tighter_cap\\per_trial\\
    CLIN_SUBJ_006_S00NONLINE_V0.npz   (cap=600 — canonical sanity)
    CLIN_SUBJ_006_S00NONLINE_V1.npz   (cap=300)
    CLIN_SUBJ_006_S00NONLINE_V2.npz   (cap=200 — the user's headline ask)
    CLIN_SUBJ_006_S00NONLINE_V3.npz   (cap=100 — aggressive lower bound)

Writes:
  C:\\Users\\arman\\Pictures\\clin_analysis_tighter_cap\\erd_refined\\
    CLIN_SUBJ_006_6panel_mi_rest_V0_cap600.png   (canonical reproduction)
    CLIN_SUBJ_006_6panel_mi_rest_V1_cap300.png
    CLIN_SUBJ_006_6panel_mi_rest_V2_cap200.png   (publication-line comparator)
    CLIN_SUBJ_006_6panel_mi_rest_V3_cap100.png

The plotter mirrors Analyze_clinical_erd_refined._plot_subject_6panel
(line 445-530) one-to-one; the per-session legend tag (D1/sustained/band-
signal/G1, D4/ES) is computed from the same evaluate_erd_quality helpers
the canonical plotter uses, so figure annotations and the V0/V1/V2/V3
scorecards cannot drift.
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
)


SUBJECT = "CLIN_SUBJ_006"
SESSIONS = [f"S{n:03d}ONLINE" for n in range(1, 6)]
SCRATCH_ROOT = Path(r"C:\Users\arman\Pictures\clin_analysis_tighter_cap")
PER_TRIAL_DIR = SCRATCH_ROOT / "per_trial"
OUT_DIR = SCRATCH_ROOT / "erd_refined"
OUT_DIR.mkdir(parents=True, exist_ok=True)


VARIANTS = [
    ("V0", "cap600",
     "Trial-rejection abs cap = 600% (canonical default). "
     "Sanity reproduction of clin_analysis/erd_refined/per_trial/*_car.npz."),
    ("V1", "cap300",
     "Trial-rejection abs cap = 300%. Halfway between canonical and G1."),
    ("V2", "cap200",
     "Trial-rejection abs cap = 200% — same threshold as G1. Trials that "
     "would trip the diagnostic gate are now removed by the cleaner."),
    ("V3", "cap100",
     "Trial-rejection abs cap = 100% (aggressive). Approaches/triggers the "
     "50% G2 over-rejection guard on most sessions; interpret with care."),
]


def _reconstruct_traces(npz_path: Path) -> dict:
    """Build the `traces` dict consumed by the 6-panel plotter from one
    per-trial npz side-car. Mirrors
    Analyze_clinical_erd_refined._extract_session_traces:325-365 shape.

    Median/SE math mirrors _cluster_timecourse:296-303 line-for-line.
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


def _plot_subject_6panel_variant(subject, session_traces, out_path,
                                 variant_label, variant_note):
    """Line-for-line mirror of _plot_subject_6panel:445-530, except the
    suptitle takes a variant_label / variant_note so we can label each
    Investigation D variant honestly."""
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
        f"[INVESTIGATION D {variant_label}]\n"
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


def main():
    print(f"[setup] subject={SUBJECT}")
    print(f"[setup] per_trial_dir={PER_TRIAL_DIR}")
    print(f"[setup] out_dir={OUT_DIR}")

    for variant_tag, suffix, note in VARIANTS:
        session_traces = []
        for sess in SESSIONS:
            npz_path = PER_TRIAL_DIR / f"{SUBJECT}_{sess}_{variant_tag}.npz"
            if not npz_path.exists():
                print(f"  [{variant_tag}] {sess}: missing {npz_path.name}; skip")
                continue
            traces = _reconstruct_traces(npz_path)
            session_traces.append((sess, traces))

        out_path = OUT_DIR / (
            f"{SUBJECT}_6panel_mi_rest_{variant_tag}_{suffix}.png"
        )
        _plot_subject_6panel_variant(
            SUBJECT, session_traces, out_path,
            f"{variant_tag} ({suffix})", note,
        )
        print(f"  [{variant_tag}] wrote: {out_path.name} "
              f"(n={len(session_traces)} sessions)")

    print(f"\nDone. Plots at {OUT_DIR}")


if __name__ == "__main__":
    main()
