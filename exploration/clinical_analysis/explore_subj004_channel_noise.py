#!/usr/bin/env python3
"""Diagnostic: characterise the channel-level noise driving CLIN_SUBJ_004's
wide SE balloon (analysis-only; no pipeline writes).

For each ONLINE session, replicate the pipeline up to (but not including)
the spatial filter — notch + broadband (4-40 Hz) + drop_fp — then quantify
per-channel noise so we can ground the channel-qualification threshold:

  1. Broadband (4-40 Hz) per-channel RMS + robust MAD-z across the montage.
  2. Mu-band (8-13 Hz) per-channel RMS + MAD-z (the band we actually analyse).
  3. Per-epoch max|x| dominant channel over the analysis window (-1, 4) s and
     the padded window (-2, 5) s, plus the 50 uV epoch-rejection count.

Prints, per session, the top outlier channels with their z so we can see
(a) whether a clear 1-2 channel outlier exists, (b) what MAD-z threshold
catches it, (c) whether the noise is broadband (electrode) or mu-only, and
(d) whether the offender sits in a motor cluster.
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

from sweep_phase2_round2 import (  # noqa: E402
    load_raw_cached, apply_blink_removal,
    NOTCH, BB_LO, BB_HI, MU_LO, MU_HI, PAD_TFR, TRIAL_WIN, REJECT_MAX_ABS_UV,
)
from exploration.clinical_analysis._helpers import (  # noqa: E402
    enumerate_online_sessions_for_subject,
    CONTRA_MOTOR_CLUSTER, IPSI_MOTOR_CLUSTER, BILATERAL_MOTOR_CLUSTER,
)

SUBJECT = "CLIN_SUBJ_004"
ANALYSIS_WIN = (-1.0, 4.0)
CLUSTERS = {
    "contra": CONTRA_MOTOR_CLUSTER,
    "ipsi": IPSI_MOTOR_CLUSTER,
    "bilat": BILATERAL_MOTOR_CLUSTER,
}


def _mad_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.zeros_like(x)
    return (x - med) / (1.4826 * mad)


def _cluster_tag(ch):
    return ",".join(k for k, chs in CLUSTERS.items() if ch in chs) or "-"


def main():
    sessions = enumerate_online_sessions_for_subject(SUBJECT)
    print(f"=== {SUBJECT}: {len(sessions)} ONLINE sessions ===\n")
    for sess in sessions:
        raw, events, event_dict = load_raw_cached(SUBJECT, sess)
        raw_bb = raw.copy()
        raw_bb.notch_filter(NOTCH, method="iir", verbose=False)
        raw_bb.filter(l_freq=BB_LO, h_freq=BB_HI, method="iir", verbose=False)
        raw_bb, _ = apply_blink_removal(raw_bb, raw.copy(), "drop_fp")
        chs = raw_bb.ch_names

        # XDF data is already in native uV (the pipeline's REJECT=50 uV
        # compares against get_data() directly), so no volt->uV scaling.
        bb = raw_bb.get_data()
        rms_bb = bb.std(axis=1)
        z_bb = _mad_z(rms_bb)

        raw_mu = raw_bb.copy().filter(l_freq=MU_LO, h_freq=MU_HI,
                                      method="iir", verbose=False)
        rms_mu = raw_mu.get_data().std(axis=1)
        z_mu = _mad_z(rms_mu)

        t0, t1 = TRIAL_WIN
        epochs_mu = mne.Epochs(
            raw_mu, events, event_id=event_dict,
            tmin=t0 - PAD_TFR, tmax=t1 + PAD_TFR,
            baseline=None, detrend=1, preload=True, verbose=False,
        )
        mu_data = epochs_mu.get_data()  # native uV (n_ep, n_ch, n_time)
        times = epochs_mu.times
        amask = (times >= ANALYSIS_WIN[0]) & (times <= ANALYSIS_WIN[1])

        # per-epoch max|x| over full vs analysis window
        max_full = np.max(np.abs(mu_data), axis=(1, 2))
        max_anal = np.max(np.abs(mu_data[:, :, amask]), axis=(1, 2))
        rej_full = int((max_full > REJECT_MAX_ABS_UV).sum())
        rej_anal = int((max_anal > REJECT_MAX_ABS_UV).sum())
        n_ep = mu_data.shape[0]

        # dominant max|x| channel among the >50uV epochs (full window)
        bad_ix = np.where(max_full > REJECT_MAX_ABS_UV)[0]
        dom = "-"
        if bad_ix.size:
            worst = [int(np.argmax(np.max(np.abs(mu_data[ei]), axis=1)))
                     for ei in bad_ix]
            vals, cnts = np.unique(worst, return_counts=True)
            dch = int(vals[np.argmax(cnts)])
            dom = f"{chs[dch]} ({cnts.max()}/{bad_ix.size} bad ep; {_cluster_tag(chs[dch])})"

        order = np.argsort(z_bb)[::-1][:4]
        print(f"-- {sess}: {n_ep} epochs | reject>50uV: full={rej_full} "
              f"analysis(-1,4)={rej_anal}")
        print(f"   dominant offender (full win): {dom}")
        print("   top broadband-RMS outliers (ch: rms_uV  z_bb  z_mu  cluster):")
        for i in order:
            print(f"     {chs[i]:>5}: {rms_bb[i]:6.1f}  z_bb={z_bb[i]:5.2f}  "
                  f"z_mu={z_mu[i]:5.2f}  [{_cluster_tag(chs[i])}]")
        print()


if __name__ == "__main__":
    main()
