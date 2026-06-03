#!/usr/bin/env python3
"""Investigation B — per-subject overlay plots.

Reads the per-trial npz side-cars produced by
scratch_subj005_008_s001_trimming.py and produces, per subject:

  1. A 6-panel figure for each variant (V0, V25, V50, V75) — mirrors
     Analyze_clinical_erd_refined._plot_subject_6panel:445-530 but
     suptitle is annotated with the variant label and the post-rejection
     n_after_reject sum for S001 so the reader sees the trim drop count.
  2. A focused bilat MI ERD% overlay: S001 under each variant + S002-S005
     median trace (the "plateau" reference). This is the publication line
     for whether trimming pulls S001 toward where the later sessions sit.

Output:
  C:\\Users\\arman\\Pictures\\clin_analysis_subj005_008_s001_trimming\\
    erd_refined/
      CLIN_SUBJ_005_6panel_mi_rest_V0.png
      CLIN_SUBJ_005_6panel_mi_rest_V25.png
      CLIN_SUBJ_005_6panel_mi_rest_V50.png
      CLIN_SUBJ_005_6panel_mi_rest_V75.png
      CLIN_SUBJ_005_bilat_mi_overlay_s001_trimming.png
      (same for CLIN_SUBJ_008)
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

SUBJECTS = ["CLIN_SUBJ_005", "CLIN_SUBJ_008"]
SESSIONS = [f"S{n:03d}ONLINE" for n in range(1, 6)]
VARIANTS = ["V0", "V25", "V50", "V75"]
INTERVENTION_SESSION = "S001ONLINE"

SCRATCH_ROOT = Path(
    r"C:\Users\arman\Pictures\clin_analysis_subj005_008_s001_trimming"
)
PER_TRIAL_DIR = SCRATCH_ROOT / "per_trial"
OUT_DIR = SCRATCH_ROOT / "erd_refined"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _reconstruct_traces(npz_path: Path) -> dict:
    """Build the `traces` dict consumed by the canonical plot fn from one
    npz side-car. Mirrors _extract_session_traces:325-365 output shape.
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


def _plot_subject_6panel_variant(subject, session_traces, out_path,
                                 variant_label, variant_note):
    """Line-for-line mirror of _plot_subject_6panel:445-530 except the
    suptitle is variant-annotated."""
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
        f"[INVESTIGATION B {variant_label}]\n"
        f"Contra: {CONTRA_MOTOR_CLUSTER} | "
        f"Bilateral: {BILATERAL_MOTOR_CLUSTER} | "
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
# Focused overlay — bilat MI for S001 under all 4 variants + S002-S005
#   plateau reference.
# ----------------------------------------------------------------------

def _plot_bilat_mi_overlay(subject, traces_per_variant, plateau_traces,
                           out_path):
    """traces_per_variant: dict variant_tag -> (times, mean_pct, low_pct,
       up_pct, n_trials, present) for S001 bilat_mi.
       plateau_traces: list of (sess_label, times, mean_pct) for S002-S005
       bilat_mi (canonical).
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    plateau_colors = plt.get_cmap("Greys")(np.linspace(0.45, 0.85,
                                                       len(plateau_traces)))
    plateau_med = None
    plateau_t = None
    plateau_stack = []
    for (sess, t, m), color in zip(plateau_traces, plateau_colors):
        ax.plot(t, m, color=color, linewidth=1.0, alpha=0.6,
                label=f"{sess} (V0 canonical)")
        if plateau_t is None:
            plateau_t = t
        plateau_stack.append(m[:len(plateau_t)])
    if plateau_stack:
        plateau_med = np.median(np.stack(plateau_stack, axis=0), axis=0)
        ax.plot(plateau_t, plateau_med, color="black", linewidth=2.2,
                linestyle="--", label="plateau median (S002-S005)",
                alpha=0.85)
    variant_colors = {"V0": "tab:red", "V25": "tab:orange",
                      "V50": "tab:olive", "V75": "tab:green"}
    for tag in VARIANTS:
        if tag not in traces_per_variant:
            continue
        times, mean_pct, low_pct, up_pct, n_trials, _present = \
            traces_per_variant[tag]
        ax.plot(times, mean_pct, color=variant_colors[tag], linewidth=2.0,
                label=f"S001 {tag} (n={n_trials})")
        ax.fill_between(times, low_pct, up_pct,
                        color=variant_colors[tag], alpha=0.12)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", ls="--", lw=0.7)
    ax.axvline(1.0, color="k", ls=":", lw=0.7)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ERD %")
    ax.set_title(
        f"S001 MI-trimming overlay — {subject} | bilat MI ERD%\n"
        f"Bilateral cluster: {BILATERAL_MOTOR_CLUSTER}\n"
        f"{_preproc_caption()}\n"
        "Question: do any of V25/V50/V75 pull S001 toward (or below) the "
        "S002-S005 plateau median (dashed black)?",
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"[setup] subjects={SUBJECTS}")
    print(f"[setup] per_trial_dir={PER_TRIAL_DIR}")
    print(f"[setup] out_dir={OUT_DIR}")

    for subj in SUBJECTS:
        # --- 6-panel per variant -----------------------------------------
        for tag in VARIANTS:
            session_traces = []
            for sess in SESSIONS:
                npz_path = PER_TRIAL_DIR / f"{subj}_{sess}_{tag}.npz"
                if not npz_path.exists():
                    continue
                traces = _reconstruct_traces(npz_path)
                session_traces.append((sess, traces))
            if not session_traces:
                continue
            if tag == "V0":
                note = ("V0 baseline — all 5 sessions canonical (no MI "
                        "trimming). S001 here is the reference for the "
                        "trimming variants.")
            else:
                pct = tag[1:]
                note = (f"{tag} — S001 MI dropped first {pct}% (in "
                        f"chronological order, AFTER canonical artifact "
                        f"rejection). S002-S005 unchanged from V0.")
            out_path = OUT_DIR / f"{subj}_6panel_mi_rest_{tag}.png"
            _plot_subject_6panel_variant(
                subj, session_traces, out_path, tag, note,
            )
            print(f"  [{subj} {tag}] wrote: {out_path.name}")

        # --- Bilat MI overlay --------------------------------------------
        s001_per_variant = {}
        for tag in VARIANTS:
            p = PER_TRIAL_DIR / f"{subj}_{INTERVENTION_SESSION}_{tag}.npz"
            if not p.exists():
                continue
            t = _reconstruct_traces(p)
            if t.get("bilat_mi") is not None:
                s001_per_variant[tag] = t["bilat_mi"]
        plateau = []
        for sess in SESSIONS:
            if sess == INTERVENTION_SESSION:
                continue
            p = PER_TRIAL_DIR / f"{subj}_{sess}_V0.npz"
            if not p.exists():
                continue
            t = _reconstruct_traces(p)
            res = t.get("bilat_mi")
            if res is None:
                continue
            times, mean_pct, _low, _up, _n, _present = res
            plateau.append((sess, times, mean_pct))

        overlay_path = (
            OUT_DIR / f"{subj}_bilat_mi_overlay_s001_trimming.png"
        )
        _plot_bilat_mi_overlay(
            subj, s001_per_variant, plateau, overlay_path,
        )
        print(f"  [{subj} overlay] wrote: {overlay_path.name}")

    print(f"\nDone. Plots at {OUT_DIR}")


if __name__ == "__main__":
    main()
