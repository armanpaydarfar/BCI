#!/usr/bin/env python3
"""Investigation A — Stage 0: pre-CAR channel inventory for CLIN_SUBJ_006.

For each of the 5 ONLINE sessions, prints raw.ch_names at three stages:
  (a) immediately after load_raw_cached
      (sweep_phase2_round2.load_raw_cached:92-161 — non-eeg/AUX filter,
       M1/M2 drop, FP*/CZ/POZ/OZ/FZ/PZ rename)
  (b) after apply_blink_removal with method="drop_fp"
      (sweep_phase2_round2.apply_blink_removal:220-231 → _drop_fp:187-192)
  (c) after the auto-drop loop, immediately before apply_spatial_filter
      (generate_plots_config_a.preprocess_and_tfr:101-134 — picks up to 4
       worst channels by AUTO_DROP_DOMINANCE_FRAC across rejected mu epochs)

Plus a single boolean per session for whether any T-row channel
(T7/T8/TP7/TP8/FT7/FT8 per sweep_phase2_round2.ZONES:83) is in the
raw, after drop_fp, and after auto-drop.

Read-only. Writes one log file:
  C:\\Users\\arman\\Pictures\\clin_analysis_subj006_motor15_input\\
      channel_inventory_subj006.txt

No canonical edits.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SWEEP_DIR = _REPO_ROOT / "exploration" / "preprocessing_sweep"
for _p in (str(_REPO_ROOT), str(_SWEEP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import warnings

import mne
import numpy as np

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

from sweep_phase2_round2 import (  # noqa: E402
    apply_blink_removal,
    BB_HI, BB_LO, FREQS, MU_HI, MU_LO, NOTCH, PAD_TFR, REJECT_MAX_ABS_UV,
    TRIAL_WIN, ZONES, load_raw_cached,
)
from sweep_phase3_validation import (  # noqa: E402
    _pick_dominant_bad_channel_max_abs,
    AUTO_DROP_DOMINANCE_FRAC, AUTO_DROP_MAX_CHANNELS, AUTO_DROP_MAX_ITERS,
    AUTO_DROP_REJECT_FRAC,
)


SUBJECT = "CLIN_SUBJ_006"
SESSIONS = [f"S{n:03d}ONLINE" for n in range(1, 6)]
OUT_DIR = Path(r"C:\Users\arman\Pictures\clin_analysis_subj006_motor15_input")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_LOG = OUT_DIR / "channel_inventory_subj006.txt"

T_ROW = set(ZONES["Temporal"])  # T7,T8,TP7,TP8,FT7,FT8


def _has_t(chs):
    return sorted(c for c in chs if c in T_ROW)


def _stage_raw(subject, session):
    """Stage (a) replica of load_raw_cached output. Returns (raw, events, event_dict)."""
    return load_raw_cached(subject, session)


def _stage_drop_fp(raw_bb):
    """Stage (b): notch + bandpass + drop_fp blink removal.

    Mirrors generate_plots_config_a.preprocess_and_tfr:92-99 — broadband
    filtered raw_bb then apply_blink_removal(method='drop_fp').
    """
    raw_bb.notch_filter(NOTCH, method="iir", verbose=False)
    raw_bb.filter(l_freq=BB_LO, h_freq=BB_HI, method="iir", verbose=False)
    raw_bb, _info = apply_blink_removal(raw_bb, raw_bb.copy(), "drop_fp")
    return raw_bb


def _stage_autodrop(raw_bb, events, event_dict):
    """Stage (c): run the auto-drop loop as preprocess_and_tfr:101-134 does.

    Returns (ch_names_after_autodrop, dropped_channels_list).
    """
    dropped = []
    iters = 0
    t0, t1 = TRIAL_WIN
    while True:
        iters += 1
        raw_mu = raw_bb.copy()
        raw_mu.filter(l_freq=MU_LO, h_freq=MU_HI, method="iir", verbose=False)
        epoch_kw = dict(
            event_id=event_dict,
            tmin=t0 - PAD_TFR, tmax=t1 + PAD_TFR,
            baseline=None, detrend=1, preload=True, verbose=False,
        )
        epochs_mu = mne.Epochs(raw_mu, events, reject=None, flat=None,
                               **epoch_kw)
        mu_data = epochs_mu.get_data()
        mask = np.max(np.abs(mu_data), axis=(1, 2)) <= REJECT_MAX_ABS_UV
        good_ix = np.where(mask)[0].tolist()
        bad_ix = np.where(~mask)[0]
        n_att = int(len(events))
        n_kept = int(len(good_ix))
        drop_frac = 1.0 - n_kept / n_att if n_att else 1.0
        if drop_frac < AUTO_DROP_REJECT_FRAC:
            break
        if len(dropped) >= AUTO_DROP_MAX_CHANNELS:
            break
        if iters > AUTO_DROP_MAX_ITERS:
            break
        bad_ch, _ = _pick_dominant_bad_channel_max_abs(
            mu_data, list(epochs_mu.ch_names), bad_ix,
            AUTO_DROP_DOMINANCE_FRAC,
        )
        if bad_ch is None or bad_ch not in raw_bb.ch_names:
            break
        raw_bb = raw_bb.copy().drop_channels([bad_ch])
        dropped.append(bad_ch)
    return list(raw_bb.ch_names), dropped, n_kept, n_att


def main():
    lines = []

    def w(s=""):
        print(s)
        lines.append(s)

    w(f"Channel-inventory sanity check — {SUBJECT}, 5 ONLINE sessions")
    w("=" * 72)
    w("Stage (a): raw.ch_names after sweep_phase2_round2.load_raw_cached")
    w("Stage (b): after apply_blink_removal(method='drop_fp') on broadband raw")
    w("Stage (c): after auto-drop loop, pre apply_spatial_filter")
    w(f"T-row set (ZONES['Temporal']): {sorted(T_ROW)}")
    w("")

    for sess in SESSIONS:
        w(f"--- {SUBJECT} / {sess} ---")
        raw, events, event_dict = _stage_raw(SUBJECT, sess)
        chs_a = list(raw.ch_names)
        w(f"  (a) n={len(chs_a)}  ch_names={chs_a}")
        w(f"      T-row present: {_has_t(chs_a)}")

        raw_b = _stage_drop_fp(raw.copy())
        chs_b = list(raw_b.ch_names)
        w(f"  (b) n={len(chs_b)}  ch_names={chs_b}")
        w(f"      T-row present: {_has_t(chs_b)}")
        delta_ab = [c for c in chs_a if c not in chs_b]
        w(f"      delta (a)→(b) dropped: {delta_ab}")

        chs_c, autodrop, n_kept, n_att = _stage_autodrop(
            raw_b.copy(), events, event_dict,
        )
        w(f"  (c) n={len(chs_c)}  ch_names={chs_c}")
        w(f"      T-row present: {_has_t(chs_c)}")
        w(f"      auto-drop fired on: {autodrop or '—'}  "
          f"(n_kept={n_kept}/{n_att})")
        w("")

    w("Summary:")
    w("  pre-CAR channel set = stage (c) above; CAR averages over that set.")
    w("")
    OUT_LOG.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote: {OUT_LOG}")


if __name__ == "__main__":
    main()
