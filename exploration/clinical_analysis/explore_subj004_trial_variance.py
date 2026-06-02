#!/usr/bin/env python3
"""Diagnostic: is CLIN_SUBJ_004's contralateral-MI SE balloon driven by a
few outlier TRIALS (so trial-level rejection is the right fix)?

Replicates the ERD timecourse's per-trial cluster-mean ERD% exactly
(`Analyze_clinical_erd_refined._cluster_timecourse`) for the contra cluster
+ MI, per session, then:
  1. ranks trials by their robust-z (median/MAD across trials, per timepoint)
     peak over the post-cue window (0, 4) s,
  2. reports how many trials exceed z thresholds and their peak ERD%,
  3. shows what the SE-band peak collapses to if those trials are dropped.

Analysis-only; no pipeline writes. CAR + drop_fp + logratio + baseline
(-1, 0) s — i.e. the ERD CONFIG_A_DISPLAY_BASELINE.
"""

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SWEEP_DIR = _REPO_ROOT / "exploration" / "preprocessing_sweep"
for _p in (str(_REPO_ROOT), str(_SWEEP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mne  # noqa: E402
mne.set_log_level("ERROR")

from generate_plots_config_a import preprocess_and_tfr  # noqa: E402
from sweep_phase2_round2 import MU_LO, MU_HI  # noqa: E402
from exploration.clinical_analysis._helpers import (  # noqa: E402
    enumerate_online_sessions_for_subject, CONTRA_MOTOR_CLUSTER,
)

SUBJECT = "CLIN_SUBJ_004"
MARKER = "200"  # MI
POST = (0.0, 4.0)
CONFIG = {
    "spatial_filter": "car", "blink_removal": "drop_fp",
    "baseline_mode": "logratio", "spectral_baseline": (-1.0, 0.0),
}


def _per_trial_pct(tfr, cluster):
    present = [c for c in cluster if c in tfr.ch_names]
    idx = [tfr.ch_names.index(c) for c in present]
    fmask = (tfr.freqs >= MU_LO) & (tfr.freqs <= MU_HI)
    pct = 100.0 * (10.0 ** tfr.data[:, idx][:, :, fmask] - 1.0)
    return pct.mean(axis=(1, 2)), tfr.times  # (n_trials, n_time)


def _se_peak(pt, tmask):
    n = pt.shape[0]
    if n < 2:
        return 0.0
    se = pt.std(axis=0, ddof=1) / np.sqrt(n)
    return float(np.max(se[tmask]))


def main():
    print(f"=== {SUBJECT}: contra cluster {CONTRA_MOTOR_CLUSTER}, MI, "
          f"per-trial ERD% over post-cue {POST}s ===\n")
    for sess in enumerate_online_sessions_for_subject(SUBJECT):
        out = preprocess_and_tfr(SUBJECT, sess, CONFIG)
        tfr = out["tfr_trials"].get(MARKER)
        if tfr is None:
            print(f"-- {sess}: no MI trials\n"); continue
        pt, times = _per_trial_pct(tfr, CONTRA_MOTOR_CLUSTER)
        n = pt.shape[0]
        tmask = (times >= POST[0]) & (times <= POST[1])

        med = np.median(pt, axis=0)
        mad = np.median(np.abs(pt - med), axis=0)
        mad_safe = np.where(mad > 0, mad, np.nan)
        z = (pt - med) / (1.4826 * mad_safe)            # (n_trials, n_time)
        trial_z = np.nanmax(np.abs(z[:, tmask]), axis=1)  # worst z per trial
        trial_peak = np.max(np.abs(pt[:, tmask]), axis=1)  # peak |ERD%| per trial

        order = np.argsort(trial_z)[::-1]
        se_all = _se_peak(pt, tmask)
        print(f"-- {sess}: {n} MI trials | SE-band peak (all trials) = "
              f"{se_all:6.1f}%  | median-trace peak = "
              f"{np.max(np.abs(med[tmask])):5.1f}%")
        for thr in (3.5, 5.0, 10.0):
            keep = trial_z <= thr
            n_drop = int((~keep).sum())
            se_kept = _se_peak(pt[keep], tmask) if keep.sum() >= 2 else float("nan")
            print(f"     drop trial-z>{thr:>4}: -{n_drop:2d} trials "
                  f"({n-n_drop} kept) -> SE-band peak = {se_kept:6.1f}%")
        print("     worst trials (idx: trial_z  peak_ERD%):")
        for i in order[:4]:
            print(f"       trial {int(i):3d}: z={trial_z[i]:6.1f}  "
                  f"peak={trial_peak[i]:8.1f}%")
        print()


if __name__ == "__main__":
    main()
