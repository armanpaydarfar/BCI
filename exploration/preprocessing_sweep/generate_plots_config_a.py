#!/usr/bin/env python3
"""
Publication plot generator — Config A (CAR + drop_fp + logratio + (-1, 0)).

Plot style mirrors visualize_online_data.py:
  - cmap = "viridis" (visualize_online_data.py:1173)
  - 0.5s windows from 0–4s, eight panels per figure (1156)
  - vlim is dynamic per scope: for each "topomap pair" (Rest + MI), vmin/vmax
    are the [2, 98] percentile of the *concatenated* mu-band TFR data of all
    AverageTFRs in that scope (1149-1151). Within a scope the Rest and MI
    figures share a scale so they're directly comparable.
  - show_names=True (1177); colorbar at bottom labelled "ERD/ERS (logratio)"
  - figure suptitles "ERD/ERS Topomaps – Rest" / "Right Arm MI" (1192-1194)

NO MI-minus-Rest figures (per user request).

Outputs:
  ~/Pictures/clin_analysis/erd_topomaps/
    per_session/<SUBJ>_<SESSION>_{rest,mi}.png
    per_subject/<SUBJ>_{rest,mi}.png       (session-avg AverageTFR)
    grand_average/grand_avg_{rest,mi}.png   (subject+session grand avg)
"""

import os
import gc
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

# Make the repo root and this sweep dir importable when run as a script
# (so `Utils.*`, `exploration.*` and the sweep-local modules all resolve).
# Idempotent no-op when imported by a caller that already set up sys.path.
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sweep_phase2_round2 import (
    apply_blink_removal, apply_spatial_filter,
    ZONES,
    NOTCH, BB_LO, BB_HI, MU_LO, MU_HI, PAD_TFR, TRIAL_WIN,
    REJECT_MAX_ABS_UV, FREQS, N_CYCLES, ICA_HP_HZ,
)
from sweep_phase3_validation import (
    load_raw_cached, _pick_dominant_bad_channel_max_abs,
    enumerate_online_sessions,
    AUTO_DROP_REJECT_FRAC, AUTO_DROP_DOMINANCE_FRAC,
    AUTO_DROP_MAX_ITERS, AUTO_DROP_MAX_CHANNELS,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")


# ======================================================================
# Config
# ======================================================================

CONFIG_A = {
    "spatial_filter":    "car",
    "blink_removal":     "drop_fp",
    "baseline_mode":     "logratio",
    # Baseline matched to the ERD timecourse window (-1, 0) s so the
    # topomaps and `Analyze_clinical_erd_refined.py` reference the same
    # contiguous pre-cue period (2026-06-01).
    "spectral_baseline": (-1.0, 0.0),
}

SUBJECTS = [f"CLIN_SUBJ_{i:03d}" for i in (2, 3, 4, 5, 6, 7, 8)]
OUT_ROOT = os.path.expanduser("~/Pictures/clin_analysis/erd_topomaps")
APPEND_MODE = False  # flip to True to skip plots that already exist

DIR_PER_SESS   = os.path.join(OUT_ROOT, "per_session")
DIR_PER_SUBJ   = os.path.join(OUT_ROOT, "per_subject")
DIR_GRAND      = os.path.join(OUT_ROOT, "grand_average")

for d in (DIR_PER_SESS, DIR_PER_SUBJ, DIR_GRAND):
    os.makedirs(d, exist_ok=True)

# Topomap layout — match visualize_online_data.py:1153-1156
TOPO_WINDOW_SIZE = 0.5
TOPO_TIME_STARTS = np.arange(0.0, 4.0, TOPO_WINDOW_SIZE)  # 8 windows
CMAP = "viridis"


# ======================================================================
# Preprocess + TFR pipeline (Config A; returns AverageTFR + per-trial TFR)
# ======================================================================

def preprocess_and_tfr(subject, session, config):
    raw, events, event_dict = load_raw_cached(subject, session)
    raw_bb = raw.copy()
    raw_1hz = raw.copy()

    raw_bb.notch_filter(NOTCH, method="iir", verbose=False)
    raw_bb.filter(l_freq=BB_LO, h_freq=BB_HI, method="iir", verbose=False)

    if config["blink_removal"] == "ica_blink_1hz":
        raw_1hz.notch_filter(NOTCH, method="iir", verbose=False)
        raw_1hz.filter(l_freq=ICA_HP_HZ, h_freq=BB_HI, method="iir", verbose=False)

    raw_bb, _ = apply_blink_removal(raw_bb, raw_1hz, config["blink_removal"])

    dropped = []
    iters = 0
    # Display/analysis window. Defaults to the shared TRIAL_WIN (-1, 4) so the
    # topomap pass (CONFIG_A, which sets no `trial_win`) is byte-for-byte
    # unchanged; the ERD pass overrides it to (-1, 5) to cover the full 5 s MI
    # task. Epoching pads by PAD_TFR on each side and the TFR is cropped back to
    # (t0, t1), so the window only affects how far post-cue is retained.
    t0, t1 = config.get("trial_win", TRIAL_WIN)
    while True:
        iters += 1
        raw_mu = raw_bb.copy()
        raw_mu.filter(l_freq=MU_LO, h_freq=MU_HI, method="iir", verbose=False)
        epoch_kw = dict(
            event_id=event_dict,
            tmin=t0 - PAD_TFR, tmax=t1 + PAD_TFR,
            baseline=None, detrend=1, preload=True, verbose=False,
        )
        epochs_mu = mne.Epochs(raw_mu, events, reject=None, flat=None, **epoch_kw)
        epochs_bb = mne.Epochs(raw_bb, events, reject=None, flat=None, **epoch_kw)
        mu_data = epochs_mu.get_data()
        mask = np.max(np.abs(mu_data), axis=(1, 2)) <= REJECT_MAX_ABS_UV
        good_ix = np.where(mask)[0].tolist()
        bad_ix = np.where(~mask)[0]
        n_att = int(len(events)); n_kept = int(len(good_ix))
        drop_frac = 1.0 - n_kept / n_att if n_att else 1.0
        if drop_frac < AUTO_DROP_REJECT_FRAC: break
        if len(dropped) >= AUTO_DROP_MAX_CHANNELS: break
        if iters > AUTO_DROP_MAX_ITERS: break
        bad_ch, _ = _pick_dominant_bad_channel_max_abs(
            mu_data, list(epochs_mu.ch_names), bad_ix, AUTO_DROP_DOMINANCE_FRAC,
        )
        if bad_ch is None or bad_ch not in raw_bb.ch_names: break
        raw_bb = raw_bb.copy().drop_channels([bad_ch])
        dropped.append(bad_ch)

    epochs = epochs_bb[good_ix]
    if len(epochs) == 0:
        raise RuntimeError("All epochs rejected after auto-drop")

    epochs = apply_spatial_filter(epochs, config["spatial_filter"])

    tfr_trials = {}
    tfr_avg = {}
    spec_bl = config["spectral_baseline"]
    mode = config["baseline_mode"]
    for marker in ("100", "200"):
        if marker not in epochs.event_id or len(epochs[marker]) == 0:
            continue
        tfr = epochs[marker].compute_tfr(
            method="multitaper", freqs=FREQS, n_cycles=N_CYCLES,
            tmin=t0 - PAD_TFR, tmax=t1 + PAD_TFR,
            use_fft=True, return_itc=False, average=False, verbose=False,
        )
        tfr.apply_baseline(baseline=spec_bl, mode=mode, verbose=False)
        tfr.crop(tmin=t0, tmax=t1)
        tfr_trials[marker] = tfr
        tfr_avg[marker] = tfr.average()

    return {
        "tfr_avg":    tfr_avg,
        "tfr_trials": tfr_trials,
        "dropped_channels": dropped,
        "n_kept":     n_kept,
        "n_attempted": n_att,
    }


# ======================================================================
# Topomap plotting (viz style)
# ======================================================================

def _compute_dynamic_vlim(*avg_tfrs, fmin=MU_LO, fmax=MU_HI, percentile=(2, 98)):
    """vlim = [2, 98] percentile of the concatenated mu-band data of all
    given AverageTFRs. Mirrors visualize_online_data.py:1149-1151 but
    restricted to the band actually plotted."""
    vals = []
    for t in avg_tfrs:
        if t is None:
            continue
        fmask = (t.freqs >= fmin) & (t.freqs <= fmax)
        vals.append(t.data[:, fmask, :].flatten())
    if not vals:
        return -0.3, 0.3
    arr = np.concatenate(vals)
    vmin, vmax = np.percentile(arr, percentile)
    return float(vmin), float(vmax)


def _plot_topo_strip(avg_tfr, fmin, fmax, vlim, fig_title, out_path):
    """Plot 8 topomaps at 0.5s windows. Mirrors plot_topomaps in
    visualize_online_data.py:1136-1194."""
    n = len(TOPO_TIME_STARTS)
    fig, axes = plt.subplots(1, n, figsize=(15, 4), constrained_layout=True)

    mappable = None
    for ax, t_start in zip(axes, TOPO_TIME_STARTS):
        t_end = t_start + TOPO_WINDOW_SIZE
        avg_tfr.plot_topomap(
            tmin=t_start, tmax=t_end,
            fmin=fmin, fmax=fmax,
            axes=ax, cmap=CMAP, show=False,
            vlim=vlim, colorbar=False, show_names=True,
        )
        if hasattr(ax, "collections") and ax.collections:
            mappable = ax.collections[0]
        ax.set_title(f"{t_start:.1f}–{t_end:.1f}s")

    if mappable is not None:
        norm = plt.Normalize(*vlim)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
        sm.set_array([])
        cbar = fig.colorbar(
            sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1
        )
        cbar.set_label("ERD/ERS (logratio)", fontsize=12)

    fig.suptitle(fig_title, fontsize=14)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _build_session_avg_tfr(avg_tfrs):
    """Channel-intersect and grand-average a list of AverageTFR objects.
    Mirrors visualize_online_data.py:1452-1469."""
    if not avg_tfrs:
        return None
    common_set = set(avg_tfrs[0].ch_names)
    for a in avg_tfrs[1:]:
        common_set &= set(a.ch_names)
    common = [c for c in avg_tfrs[0].ch_names if c in common_set]
    if not common:
        return None
    aligned = [a.copy().pick(common, verbose=False) for a in avg_tfrs]
    data_stack = np.stack([a.data for a in aligned], axis=0)
    grand = aligned[0].copy()
    grand.data = data_stack.mean(axis=0)
    grand.nave = len(aligned)
    return grand


def _pool_avgs_weighted(avg_n_list):
    """n-weighted grand-average of AverageTFRs over a channel intersection.

    In logratio space the n-weighted mean of per-session averages equals the
    exact trial-pooled mean, so this pools CLIN_SUBJ_002's same-day S003+S004
    into one session-2 topomap without concatenating EpochsTFR. `avg_n_list`
    is a list of (AverageTFR, n_trials). Returns one AverageTFR or None.
    """
    avg_n_list = [(a, n) for a, n in avg_n_list if a is not None and n > 0]
    if not avg_n_list:
        return None
    avgs = [a for a, _ in avg_n_list]
    if len(avgs) == 1:
        return avgs[0]
    common_set = set(avgs[0].ch_names)
    for a in avgs[1:]:
        common_set &= set(a.ch_names)
    common = [c for c in avgs[0].ch_names if c in common_set]
    if not common:
        return None
    aligned = [a.copy().pick(common, verbose=False) for a in avgs]
    w = np.array([n for _, n in avg_n_list], dtype=float)
    w /= w.sum()
    stack = np.stack([a.data for a in aligned], axis=0)
    grand = aligned[0].copy()
    grand.data = np.tensordot(w, stack, axes=(0, 0))
    grand.nave = int(sum(n for _, n in avg_n_list))
    return grand


def plot_session_topos(subject, session, tfr_avg, out_dir):
    rest = tfr_avg.get("100"); mi = tfr_avg.get("200")
    vlim = _compute_dynamic_vlim(rest, mi)

    if rest is not None:
        out = os.path.join(out_dir, f"{subject}_{session}_rest.png")
        if not (APPEND_MODE and os.path.exists(out)):
            _plot_topo_strip(
                rest, MU_LO, MU_HI, vlim,
                f"ERD/ERS Topomaps – Rest | {subject} / {session}",
                out,
            )
    if mi is not None:
        out = os.path.join(out_dir, f"{subject}_{session}_mi.png")
        if not (APPEND_MODE and os.path.exists(out)):
            _plot_topo_strip(
                mi, MU_LO, MU_HI, vlim,
                f"ERD/ERS Topomaps – Right Arm MI | {subject} / {session}",
                out,
            )


def plot_subject_topo(subject, sess_tfrs, out_dir):
    rest_list = [t["100"] for t in sess_tfrs if "100" in t]
    mi_list   = [t["200"] for t in sess_tfrs if "200" in t]
    rest_avg = _build_session_avg_tfr(rest_list)
    mi_avg   = _build_session_avg_tfr(mi_list)
    vlim = _compute_dynamic_vlim(rest_avg, mi_avg)

    if rest_avg is not None:
        out = os.path.join(out_dir, f"{subject}_rest.png")
        if not (APPEND_MODE and os.path.exists(out)):
            _plot_topo_strip(
                rest_avg, MU_LO, MU_HI, vlim,
                f"ERD/ERS Topomaps – Rest | {subject} | "
                f"{len(rest_list)} sessions averaged",
                out,
            )
    if mi_avg is not None:
        out = os.path.join(out_dir, f"{subject}_mi.png")
        if not (APPEND_MODE and os.path.exists(out)):
            _plot_topo_strip(
                mi_avg, MU_LO, MU_HI, vlim,
                f"ERD/ERS Topomaps – Right Arm MI | {subject} | "
                f"{len(mi_list)} sessions averaged",
                out,
            )


def plot_grand_topo(all_subject_tfrs, out_dir):
    rest_all = [t["100"] for sess_tfrs in all_subject_tfrs.values()
                for t in sess_tfrs if "100" in t]
    mi_all   = [t["200"] for sess_tfrs in all_subject_tfrs.values()
                for t in sess_tfrs if "200" in t]
    rest_avg = _build_session_avg_tfr(rest_all)
    mi_avg   = _build_session_avg_tfr(mi_all)
    vlim = _compute_dynamic_vlim(rest_avg, mi_avg)

    n_subj = len(all_subject_tfrs)
    n_sess = len(mi_all) if mi_all else len(rest_all)

    if rest_avg is not None:
        out = os.path.join(out_dir, "grand_avg_rest.png")
        if not (APPEND_MODE and os.path.exists(out)):
            _plot_topo_strip(
                rest_avg, MU_LO, MU_HI, vlim,
                f"ERD/ERS Topomaps – Rest | GRAND AVG: "
                f"{n_subj} subjects, {n_sess} sessions",
                out,
            )
    if mi_avg is not None:
        out = os.path.join(out_dir, "grand_avg_mi.png")
        if not (APPEND_MODE and os.path.exists(out)):
            _plot_topo_strip(
                mi_avg, MU_LO, MU_HI, vlim,
                f"ERD/ERS Topomaps – Right Arm MI | GRAND AVG: "
                f"{n_subj} subjects, {n_sess} sessions",
                out,
            )


# ======================================================================
# Main
# ======================================================================

def _rejected_marker_avgs(tfr_trials):
    """Apply the canonical cap=200 bilateral-cluster (log-space) rejection and
    a per-channel baseline re-zero, then average each marker. Returns
    {marker: (AverageTFR, n_kept)}.

    The re-zero subtracts each trial/channel/freq's baseline-window mean of the
    logratio (the geometric-mean baseline), so the topomap sits at 0 in the
    pre-cue window like the erd_refined timecourse — without it, MNE's
    arithmetic-mean logratio baseline carries the Jensen offset (~-12% / -0.5 dB),
    leaving the topomaps ~12% deeper than the re-zeroed timecourses. A
    contra-cluster check confirmed this re-zero brings the two into agreement.

    Lazy imports keep this module importable by `Analyze_clinical_erd_refined`
    (which imports `preprocess_and_tfr` from here) without a circular import.
    """
    from Analyze_clinical_erd_refined import _reject_artifact_trials_for_cluster
    from exploration.clinical_analysis._helpers import BILATERAL_MOTOR_CLUSTER
    rejected, _rep = _reject_artifact_trials_for_cluster(
        tfr_trials, BILATERAL_MOTOR_CLUSTER,
    )
    bl0, bl1 = CONFIG_A["spectral_baseline"]
    out = {}
    for marker, tfr in rejected.items():
        n = int(tfr.data.shape[0])
        if n == 0:
            continue
        bmask = (tfr.times >= bl0) & (tfr.times <= bl1)
        tfr = tfr.copy()
        tfr.data = tfr.data - tfr.data[:, :, :, bmask].mean(axis=3,
                                                            keepdims=True)
        out[marker] = (tfr.average(), n)
    return out


def main():
    print(f"Output root: {OUT_ROOT}")
    print(f"Config A: {CONFIG_A}")
    # Feature-family policy for CLIN_SUBJ_002 (ERD is decoder-independent):
    # drop S001 (left arm); S003+S004 (same day) pool into one timepoint.
    from exploration.clinical_analysis._helpers import session_idx_from_label
    from exploration.clinical_analysis._subj002 import (
        is_subj002, subj002_feature_idx, subj002_feature_sessions,
    )
    all_subject_tfrs = {}

    for subj in SUBJECTS:
        sessions = enumerate_online_sessions(subj)
        if is_subj002(subj):
            valid = set(subj002_feature_sessions())
            sessions = [s for s in sessions if s in valid]
        # Group sessions by longitudinal index (pools SUBJ_002 S003+S004).
        groups: dict[int, list[str]] = {}
        for sess in sessions:
            gidx = (subj002_feature_idx(sess) if is_subj002(subj)
                    else session_idx_from_label(sess))
            groups.setdefault(gidx, []).append(sess)
        print(f"\n=== {subj} ({len(sessions)} sessions -> "
              f"{len(groups)} timepoints) ===")
        subj_tfr_avgs = []

        for gidx in sorted(groups):
            sess_list = groups[gidx]
            label = "+".join(sess_list)
            t0 = time.time()
            per_marker = {"100": [], "200": []}  # marker -> [(AverageTFR, n)]
            for sess in sess_list:
                try:
                    out = preprocess_and_tfr(subj, sess, CONFIG_A)
                except Exception as e:
                    print(f"  {sess}: FAILED ({type(e).__name__}: {e})")
                    continue
                for marker, (avg, n) in _rejected_marker_avgs(
                    out["tfr_trials"]
                ).items():
                    per_marker.setdefault(marker, []).append((avg, n))
                del out
                gc.collect()
            tfr_avg = {}
            for marker in ("100", "200"):
                pooled = _pool_avgs_weighted(per_marker.get(marker, []))
                if pooled is not None:
                    tfr_avg[marker] = pooled
            if not tfr_avg:
                continue
            plot_session_topos(subj, label, tfr_avg, DIR_PER_SESS)
            subj_tfr_avgs.append(tfr_avg)
            print(f"  t{gidx} {label}: ({time.time()-t0:.1f}s)")
            gc.collect()

        if subj_tfr_avgs:
            plot_subject_topo(subj, subj_tfr_avgs, DIR_PER_SUBJ)
            all_subject_tfrs[subj] = subj_tfr_avgs
        gc.collect()

    if all_subject_tfrs:
        plot_grand_topo(all_subject_tfrs, DIR_GRAND)

    print(f"\nDone. Plots at: {OUT_ROOT}")


if __name__ == "__main__":
    main()
