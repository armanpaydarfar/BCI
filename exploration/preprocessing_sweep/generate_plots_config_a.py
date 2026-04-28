#!/usr/bin/env python3
"""
Publication plot generator — Config A (CAR + drop_fp + logratio + (-1.5, -0.25)).

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
  ~/Pictures/clin_erd_plots/
    per_session_topomaps/<SUBJ>_<SESSION>_{rest,mi}.png
    per_subject_topomaps/<SUBJ>_{rest,mi}.png       (session-avg AverageTFR)
    grand_average_topomap/grand_avg_{rest,mi}.png   (subject+session grand avg)
    per_subject_timecourses/<SUBJ>_timecourse.png   (2 panels, rest + mi,
        one line per session at that session's most-focal MI electrode;
        legend = "<session> (<electrode>)")
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

from sweep_phase2_round2 import (
    apply_blink_removal, apply_spatial_filter,
    ZONES,
    NOTCH, BB_LO, BB_HI, MU_LO, MU_HI, PAD_TFR, TRIAL_WIN, SCALAR_WIN,
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
    "spectral_baseline": (-1.5, -0.25),
}

SUBJECTS = [f"CLIN_SUBJ_{i:03d}" for i in (2, 3, 4, 5, 6, 7, 8)]
OUT_ROOT = os.path.expanduser("~/Pictures/clin_erd_plots")
APPEND_MODE = False  # flip to True to skip plots that already exist

DIR_PER_SESS   = os.path.join(OUT_ROOT, "per_session_topomaps")
DIR_PER_SUBJ   = os.path.join(OUT_ROOT, "per_subject_topomaps")
DIR_GRAND      = os.path.join(OUT_ROOT, "grand_average_topomap")
DIR_TIMECOURSE = os.path.join(OUT_ROOT, "per_subject_timecourses")

for d in (DIR_PER_SESS, DIR_PER_SUBJ, DIR_GRAND, DIR_TIMECOURSE):
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
# Per-subject timecourse (viz multi-session-overlay style)
# ======================================================================

def _most_focal_electrode(tfr_trials, marker="200"):
    """Channel with strongest mu-band ERD over the (1, 4)s scalar window,
    averaged over trials. Negative logratio = stronger ERD."""
    if marker not in tfr_trials:
        return None
    tfr = tfr_trials[marker]
    freqs = tfr.freqs; times = tfr.times
    fmask = (freqs >= MU_LO) & (freqs <= MU_HI)
    tmask = (times >= SCALAR_WIN[0]) & (times <= SCALAR_WIN[1])
    per_ch = tfr.data[:, :, fmask, :][:, :, :, tmask].mean(axis=(0, 2, 3))
    return tfr.ch_names[int(np.argmin(per_ch))]


def _logratio_to_pct(x):
    return 100.0 * (10.0 ** x - 1.0)


def _timecourse_at_channel(tfr_trials, ch_name, marker):
    """Trial-mean and SEM at `ch_name` for the marker, in % space.

    Trial-averaging is done in logratio (stable), then converted.
    Mirrors compute_focal_timecourses in visualize_online_data.py:1058-1100,
    but for a single channel."""
    if marker not in tfr_trials:
        return None
    tfr = tfr_trials[marker]
    if ch_name not in tfr.ch_names:
        return None
    ch_idx = tfr.ch_names.index(ch_name)
    freqs = tfr.freqs; times = tfr.times
    fmask = (freqs >= MU_LO) & (freqs <= MU_HI)
    per_trial = tfr.data[:, ch_idx][:, fmask, :].mean(axis=1)  # (trials, time)
    if per_trial.shape[0] < 1:
        return None
    mean_log = per_trial.mean(axis=0)
    if per_trial.shape[0] > 1:
        sem_log = per_trial.std(axis=0, ddof=1) / np.sqrt(per_trial.shape[0])
    else:
        sem_log = np.zeros_like(mean_log)
    mean_pct = _logratio_to_pct(mean_log)
    low_pct = _logratio_to_pct(mean_log - sem_log)
    up_pct  = _logratio_to_pct(mean_log + sem_log)
    return times, mean_pct, low_pct, up_pct


def plot_subject_timecourse(subject, sessions_and_trials, out_dir):
    """Two-panel (Rest, MI) figure. Each session is one line; the focal
    electrode is selected from MI mu-band peak ERD on that session and is
    used for BOTH panels. Mirrors plot_multisession_overlay_timecourses_cached
    in visualize_online_data.py:524-578."""
    if not sessions_and_trials:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    ax_rest, ax_mi = axes
    cmap = plt.get_cmap("tab10")

    any_drawn = False
    for i, (sess, tfr_trials) in enumerate(sessions_and_trials):
        focal_ch = _most_focal_electrode(tfr_trials, marker="200")
        if focal_ch is None:
            continue
        line_label = f"{sess} ({focal_ch})"
        color = cmap(i % 10)

        for marker, ax in (("100", ax_rest), ("200", ax_mi)):
            res = _timecourse_at_channel(tfr_trials, focal_ch, marker=marker)
            if res is None:
                continue
            times, mean_pct, low_pct, up_pct = res
            ax.plot(times, mean_pct, color=color, label=line_label, linewidth=1.5)
            ax.fill_between(times, low_pct, up_pct, color=color, alpha=0.15)
            any_drawn = True

    if not any_drawn:
        plt.close(fig); return

    for ax, t in zip(axes, ["REST", "MI"]):
        ax.axhline(0, color="k", linewidth=0.6)
        ax.axvline(0, color="k", linestyle="--", linewidth=0.7)
        ax.axvline(1.0, color="k", linestyle=":", linewidth=0.7)
        ax.set_xlabel("Time (s)")
        ax.set_title(t)
        ax.grid(True, alpha=0.25)
    ax_rest.set_ylabel("ERD %")
    fig.suptitle(
        f"Focal MU ERD Across Sessions | {subject} | Config A", fontsize=12
    )
    ax_mi.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, f"{subject}_timecourse.png"),
        dpi=120, bbox_inches="tight",
    )
    plt.close(fig)


# ======================================================================
# Main
# ======================================================================

def main():
    print(f"Output root: {OUT_ROOT}")
    print(f"Config A: {CONFIG_A}")
    all_subject_tfrs = {}

    total_sessions = sum(len(enumerate_online_sessions(s)) for s in SUBJECTS)
    idx = 0

    for subj in SUBJECTS:
        sessions = enumerate_online_sessions(subj)
        print(f"\n=== {subj} ({len(sessions)} sessions) ===")
        subj_tfr_avgs = []
        subj_sess_trials = []

        for sess in sessions:
            idx += 1
            t0 = time.time()
            try:
                out = preprocess_and_tfr(subj, sess, CONFIG_A)
            except Exception as e:
                print(f"  [{idx}/{total_sessions}] {sess}: FAILED ({type(e).__name__}: {e})")
                continue

            plot_session_topos(subj, sess, out["tfr_avg"], DIR_PER_SESS)
            subj_tfr_avgs.append(out["tfr_avg"])
            subj_sess_trials.append((sess, out["tfr_trials"]))

            print(
                f"  [{idx}/{total_sessions}] {sess}: "
                f"n_kept={out['n_kept']}/{out['n_attempted']}  "
                f"dropped={out['dropped_channels'] or '—'}  "
                f"({time.time()-t0:.1f}s)"
            )
            gc.collect()

        if subj_tfr_avgs:
            plot_subject_topo(subj, subj_tfr_avgs, DIR_PER_SUBJ)
            plot_subject_timecourse(subj, subj_sess_trials, DIR_TIMECOURSE)
            all_subject_tfrs[subj] = subj_tfr_avgs

        del subj_sess_trials
        gc.collect()

    if all_subject_tfrs:
        plot_grand_topo(all_subject_tfrs, DIR_GRAND)

    print(f"\nDone. Plots at: {OUT_ROOT}")


if __name__ == "__main__":
    main()
