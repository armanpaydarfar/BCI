#!/usr/bin/env python3
"""Refined ERD line plots for the CLIN_* cohort (Pass 1).

Three cluster-averaged ERD% metrics — Contralateral, Bilateral, and
Ipsilateral — with auto-dropped cluster channels removed and the
surviving subset reported in the legend.

Per-subject 6-panel figure: 3 rows (Contralateral, Bilateral,
Ipsilateral) x 2 cols (MI, Rest). Same y-axis per row so MI and Rest
are directly comparable on the same scale.

Cohort 6-panel figure: 3 rows (Contralateral, Bilateral, Ipsilateral)
x 2 cols (MI, Rest), one line per session_idx colour-coded by viridis.

Every figure carries a one-line preprocessing caption (spatial filter,
blink handling, mu band, baseline window) so the scheme behind the
plot is always visible.

Outputs to `~/Pictures/clin_analysis/erd_refined/`:
    <SUBJ>_6panel_mi_rest.png    (per subject)
    cohort_6panel_mi_rest.png    (cohort summary)
    erd_refined_data.csv         (cluster traces per (subj, sess, marker))

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
    BILATERAL_MOTOR_CLUSTER, CONTRA_MOTOR_CLUSTER, IPSI_MOTOR_CLUSTER,
    clin_pictures_root, enumerate_clin_subjects,
    enumerate_online_sessions_for_subject, resolve_motor_cluster,
    session_idx_from_label,
)

# Mu-band edges for the cluster ERD% average (same as the validated
# sweep). MU_LO, MU_HI come from sweep_phase2_round2.py:66.
from sweep_phase2_round2 import MU_HI, MU_LO  # noqa: E402

# Quality-score primitives, imported so the per-panel legend tags are computed
# with the exact same functions/thresholds evaluate_erd_quality uses — the
# figure and the scorer can never disagree (single source of truth). The
# scorer's docstring designates these dimension helpers for annotation reuse.
from evaluate_erd_quality import (  # noqa: E402
    G1_OUTLIER_FRAC, G1_OUTLIER_PCT, _d1_mi_strength, _d2_sustained,
    _d4_rest_specificity, _d8_band_to_signal, _postcue_mask, _scalar_mask,
)

# Override Config A's spectral baseline to be contiguous to the cue
# (-1, 0) s. The helper's CONFIG_A uses (-1.5, -0.25) — chosen by the
# sweep to leave a quarter-second gap before the cue — but the
# downstream `tfr.crop(tmin=-1, tmax=4)` at
# `generate_plots_config_a.py:151` then hides the (-1.5, -1) portion
# from the displayed window. That asymmetry leaves the visible
# baseline offset by whatever average the hidden window held. Setting
# the baseline equal to the visible pre-cue window puts the baseline
# at 0 on the plot by construction.
CONFIG_A_DISPLAY_BASELINE = {
    "spatial_filter":    "car",
    "blink_removal":     "drop_fp",
    "baseline_mode":     "logratio",
    "spectral_baseline": (-1.0, 0.0),
    # ERD-only window override: cover the full 5 s MI task. preprocess_and_tfr
    # defaults to the shared TRIAL_WIN (-1, 4) when this key is absent, so the
    # topomap pass is unaffected. Set on the CLI via --window-end.
    "trial_win":         (-1.0, 5.0),
}

# Robust-z threshold for per-trial artifact rejection (Phase 4). A trial whose
# post-cue peak |logratio| over the bilateral montage is > this many MADs from
# the session median is dropped. Artifacts sit at z=15–789, normal trials z<5,
# so 5.0 cleanly separates them. Set on the CLI via --reject-z (<=0 disables).
TRIAL_REJECT_Z = 5.0

# A session marker is left un-rejected (raw, surfaced for review) rather than
# cleaned past this fraction — mirrors rubric gate G2. Set to 0.50 (per Arman):
# dropping up to half a run's trials is still a minority and acceptable when the
# rest are clean; only beyond half is the session "fundamentally bad". Must stay
# in sync with the scorer's G2 threshold (evaluate_erd_quality.G2_OVERREJECT_FRAC).
TRIAL_REJECT_MAX_FRAC = 0.50

# Absolute artifact cap: a trial whose post-cue peak |ERD%| exceeds this is
# dropped regardless of its robust-z. Set to 200% to match the G1 diagnostic
# threshold (evaluate_erd_quality.py:101) — cap closes the asymmetry where
# trials in the 200-600% band passed the old cleaner (600) but tripped the
# G1 gate (200). Cohort cap-sweep (clin-cohort-evidence-synthesis 2026-06-03)
# confirmed cap=200 + cluster-matched rejection raises cohort eligible rows
# from 31 to 92 of 102 without regressing any subject. Physiologically a
# sustained mu ERD sits in [0, -30]%; ERS peaks above ±200% are muscle bursts,
# not real signal, and per-trial diagnostics confirm those trials are
# sustained outliers (high frac_post_over_50pct) not transient spikes.
TRIAL_REJECT_ABS_PCT = 200.0

import mne  # noqa: E402


def config_a_pipeline(subject, session):
    """Run Config A preprocess + TFR with a display-matched baseline.

    Local override of `_helpers.config_a_pipeline` — same Config A
    (CAR, drop_fp, logratio) but with `spectral_baseline=(-1, 0)` so
    the baseline window matches the visible pre-cue display window.
    """
    from generate_plots_config_a import preprocess_and_tfr
    return preprocess_and_tfr(subject, session, CONFIG_A_DISPLAY_BASELINE)

mne.set_log_level("ERROR")


MI_MARKER = "200"
REST_MARKER = "100"

# Description of the shaded band drawn around each mean trace. The mean and
# the band are computed in LOG space (geometric mean of power ratios), then
# converted to %/dB — % is asymmetric and an arithmetic % mean is dominated by
# ERS bins (see `_per_trial_cluster_logratio`).
_BAND_LABEL = "shaded = mean ± SE (log-space, across trials)"


def _preproc_caption():
    """One-line preprocessing caption rendered on every figure. Read
    from the live CONFIG at call time so it reflects the runtime
    `--spatial-filter` override.
    """
    cfg = CONFIG_A_DISPLAY_BASELINE
    reject = (f"trial-z>{TRIAL_REJECT_Z:g}|cap{TRIAL_REJECT_ABS_PCT:g}%"
              if TRIAL_REJECT_Z > 0 else "off")
    return (
        f"Preproc: {cfg['spatial_filter'].upper()} spatial filter | "
        f"blink={cfg['blink_removal']} | "
        f"μ {MU_LO:g}–{MU_HI:g} Hz | "
        f"baseline {cfg['spectral_baseline']} s | "
        f"window {cfg['trial_win']} s | "
        f"trial-reject {reject} (cluster-matched) | "
        f"display: dB (10·log10), substrate: %"
    )


# ----------------------------------------------------------------------
# Cluster trace helpers
# ----------------------------------------------------------------------

def _logratio_to_pct(x):
    """Mirrors generate_plots_config_a.py:335-336."""
    return 100.0 * (10.0 ** x - 1.0)


def _per_trial_cluster_logratio(tfr, cluster_channels):
    """Per-trial cluster-mean mu logratio trace: (n_trials, n_time).

    Averages the mu-band logratio across the cluster's channels and mu
    freqs in LOG space (geometric mean of power ratios), NOT in % space.
    This is the single source of the per-trial cluster substrate, used by
    both the artifact rejection and the timecourse.

    Why log space: ERD% is asymmetric — an ERD bin is floored at -100% but
    an ERS bin is unbounded (+200%, +500%, ...). Arithmetic-averaging % across
    channels/freqs therefore lets a few ERS bins wash out genuine ERD; on the
    noisy subjects this collapsed real desync to ~0 (CLIN_SUBJ_008 contra MI:
    -24% in log space vs -0.5% under the old %-first averaging — a single
    clean channel like C3 read -31% in log space but -11% %-first, and CP1
    flipped from -16% to +8%). Log-space averaging is symmetric, matches the
    topomap's `tfr.average()` (`generate_plots_config_a.py:157`) and the
    standard dB ERSP convention, so the timecourse, topomap and neuromod
    longitudinal scalar agree by construction.

    Each trial's baseline window is re-zeroed in log space (subtract the
    per-trial baseline-window mean). This is required for log-space averaging:
    MNE's `apply_baseline(mode="logratio")` normalises by the ARITHMETIC mean
    of baseline power, so the baseline-window mean of the logratio is
    `mean(log P) - log(mean P)`, which is < 0 by Jensen's inequality (~-0.05
    logratio ≈ -0.5 dB). The old %-first averaging cancelled this exactly in
    linear space; log-space averaging does not, so without re-zeroing the
    baseline sits below 0. Subtracting the per-trial baseline-window mean
    redefines the reference as the GEOMETRIC mean of baseline power (the
    standard dB ERSP baseline), so the baseline is 0 by construction.

    Returns (per_trial_logratio, present_channels) or (None, []) if no
    cluster channel survived.
    """
    present = [c for c in cluster_channels if c in tfr.ch_names]
    if not present:
        return None, []
    ch_idxs = [tfr.ch_names.index(c) for c in present]
    fmask = (tfr.freqs >= MU_LO) & (tfr.freqs <= MU_HI)
    log = tfr.data[:, ch_idxs][:, :, fmask].mean(axis=(1, 2))
    bl0, bl1 = CONFIG_A_DISPLAY_BASELINE["spectral_baseline"]
    bmask = (tfr.times >= bl0) & (tfr.times <= bl1)
    if bmask.any():
        log = log - log[:, bmask].mean(axis=1, keepdims=True)
    return log, present


def _reject_artifact_trials_for_cluster(tfr_trials, cluster_chs,
                                        reject_z=TRIAL_REJECT_Z,
                                        max_frac=TRIAL_REJECT_MAX_FRAC,
                                        abs_cap=TRIAL_REJECT_ABS_PCT):
    """Non-mutating, cluster-matched, log-space artifact rejection.

    Returns (new_tfr_trials, report). Like the bilateral
    `_reject_artifact_trials` (kept %-first for the oneshot scratch stream that
    imports it) but: (a) the rejection substrate is the supplied
    `cluster_chs` instead of hardcoded `BILATERAL_MOTOR_CLUSTER`; (b) the input
    dict is NOT mutated — a fresh dict with sliced EpochsTFR entries is
    returned, so the same tfr_base can feed multiple cluster rejections in
    sequence; (c) the per-trial scalar is built by `_per_trial_cluster_logratio`
    (log-averaged across ch+freq, then % once), so the cap=200 test sees the
    same ERS-robust substrate as the timecourse rather than a %-first average.

    Used by `_extract_session_traces` to give each cluster (contra/bilat/
    ipsi) its own rejected pool. Rationale: G1 (evaluate_erd_quality.py:
    587-600) is evaluated per cluster, but the hardcoded-bilat substrate
    over-dilutes focal-channel spikes when the cluster being scored has
    only 4 channels (contra/ipsi). Cluster-matched rejection closes that
    asymmetry without changing the scorer.
    """
    new_trials = {}
    report = {}
    for marker, tfr in tfr_trials.items():
        n_before = int(tfr.data.shape[0])
        info = {"n_before": n_before, "n_dropped": 0,
                "kept": True, "over_gate": False}
        report[marker] = info
        new_trials[marker] = tfr  # default to no-op
        if reject_z <= 0 or n_before < 4:
            continue
        log, present = _per_trial_cluster_logratio(tfr, cluster_chs)
        if log is None:
            continue
        # Per-trial cluster ERD%(t): log-averaged across ch+freq, then % once
        # (the ERS-robust substrate; see `_per_trial_cluster_logratio`).
        pct = _logratio_to_pct(log)
        tmask = tfr.times >= 0.0
        scalar = np.max(np.abs(pct[:, tmask]), axis=1)
        abs_drop = scalar > abs_cap
        med = np.median(scalar)
        mad = np.median(np.abs(scalar - med))
        if mad > 0:
            z = (scalar - med) / (1.4826 * mad)
            z_drop = np.abs(z) > reject_z
        else:
            z_drop = np.zeros_like(scalar, dtype=bool)
        drop = z_drop | abs_drop
        n_drop = int(drop.sum())
        if n_drop == 0:
            continue
        max_drop = int(np.floor(max_frac * n_before))
        if n_drop > max_drop:
            info["over_gate"] = True
            worst = np.argsort(scalar)[::-1][:max_drop]
            keep_mask = np.ones(n_before, dtype=bool)
            keep_mask[worst] = False
            n_drop = max_drop
        else:
            keep_mask = ~drop
        if n_drop == 0:
            continue
        new_trials[marker] = tfr[np.where(keep_mask)[0]]
        info["n_dropped"] = n_drop
    return new_trials, report


def _reject_artifact_trials(tfr_trials, reject_z=TRIAL_REJECT_Z,
                            max_frac=TRIAL_REJECT_MAX_FRAC,
                            abs_cap=TRIAL_REJECT_ABS_PCT):
    """Drop baseline-normalization artifact trials per marker, in place.

    The 50µV epoch reject upstream catches amplitude artifacts but misses
    trials whose tiny pre-cue baseline makes the logratio explode (one
    CLIN_SUBJ_004 trial reached 20,649% ERD%). For each marker we form a
    per-trial scalar = post-cue peak |ERD%| over the bilateral motor montage,
    take a robust z (median/MAD across the session's trials), and drop trials
    with |z| > reject_z. The drop is applied to the marker's EpochsTFR, so it
    is consistent across the contra/bilat/ipsi clusters that index into the
    same trials.

    The scalar is built in PERCENT space (convert each channel to ERD% then
    average over the montage and mu band), matching the validated diagnostic
    `explore_subj004_trial_variance._per_trial_pct` and the scorer substrate.
    Averaging in logratio space first would geometrically dilute a focal
    single-channel blow-up (one channel at x≈2.3 with seven near 0 averages
    to ≈0.29) and hide the very trial we are trying to catch; in % space the
    arithmetic mean preserves it (the blow-up trial sits at z≈100s). MAD is a
    median statistic, so the single outlier does not corrupt the spread.

    A single per-trial scalar is used (not max-over-time z), which targets
    the few blow-up trials without the ~40% over-rejection a per-timepoint
    z-rule produces.

    A trial is also dropped if its scalar exceeds `abs_cap` (absolute peak
    |ERD%|), independent of z. The relative z-rule misses catastrophic trials
    in a *pervasively* noisy marker (CLIN_SUBJ_007/008 REST), where the MAD is
    inflated by surrounding noise so a 1000%+ trial reads as z<5; the absolute
    cap catches those. A few-hundred-percent mu ERD% is physiologically
    impossible, so the cap removes only artifact.

    Guard (rubric §4 G2): cleaning is capped at `max_frac` of a marker's
    trials. If more than that qualify, only the `max_frac` WORST offenders
    (highest post-cue peak |ERD%|) are dropped and the rest are kept — so a
    pervasively-noisy session is trimmed of its worst artifacts rather than
    left raw, but never cleaned past the cap into a corner. `over_gate` flags a
    session that hit the cap.

    Mutates `tfr_trials`. Returns a per-marker report:
    {marker: {"n_before", "n_dropped", "kept", "over_gate"}}.
    """
    report = {}
    for marker, tfr in list(tfr_trials.items()):
        n_before = int(tfr.data.shape[0])
        info = {"n_before": n_before, "n_dropped": 0,
                "kept": True, "over_gate": False}
        report[marker] = info
        if reject_z <= 0 or n_before < 4:
            continue  # disabled, or too few trials for a robust MAD
        present = [c for c in BILATERAL_MOTOR_CLUSTER if c in tfr.ch_names]
        if not present:
            continue
        ch_idxs = [tfr.ch_names.index(c) for c in present]
        fmask = (tfr.freqs >= MU_LO) & (tfr.freqs <= MU_HI)
        # Per-trial cluster-mean ERD% trace (% first, then mean over montage
        # channels + mu freqs) — same substrate as the scorer/diagnostic.
        pct = _logratio_to_pct(
            tfr.data[:, ch_idxs][:, :, fmask],
        ).mean(axis=(1, 2))
        # Full post-cue window [0, t_end] — onset-region (0–1 s) blow-ups are
        # real artifacts (one subj3 trial peaked 2213% there) and must be
        # caught here, matching the scorer's gate G1 window. The robust z only
        # flags statistical outliers, so a bounded physiological onset desync
        # is not rejected.
        tmask = tfr.times >= 0.0
        scalar = np.max(np.abs(pct[:, tmask]), axis=1)  # peak |ERD%| / trial
        # Absolute cap is independent of the spread, so it still fires when the
        # MAD is degenerate (one huge trial among many identical ones gives
        # MAD=0, which would otherwise skip rejection and keep the outlier).
        abs_drop = scalar > abs_cap
        med = np.median(scalar)
        mad = np.median(np.abs(scalar - med))
        if mad > 0:
            z = (scalar - med) / (1.4826 * mad)
            z_drop = np.abs(z) > reject_z
        else:
            z_drop = np.zeros_like(scalar, dtype=bool)
        drop = z_drop | abs_drop
        n_drop = int(drop.sum())
        if n_drop == 0:
            continue
        max_drop = int(np.floor(max_frac * n_before))
        if n_drop > max_drop:
            # More artifacts qualify than the cap allows. Rather than leave the
            # session raw, drop only the `max_drop` WORST offenders (highest
            # post-cue peak |ERD%|) and keep the rest. The kept remainder still
            # contains the sub-threshold qualifiers, so the session reads as
            # "less bad", not falsely pristine; over_gate flags that cleaning
            # was capped (the session needed maximal trimming — interpret with
            # care). Removing the high tail biases the kept median slightly
            # optimistic, an accepted trade for not discarding a whole session.
            info["over_gate"] = True
            worst = np.argsort(scalar)[::-1][:max_drop]
            keep_mask = np.ones(n_before, dtype=bool)
            keep_mask[worst] = False
            n_drop = max_drop
        else:
            keep_mask = ~drop
        if n_drop == 0:
            continue  # cap floored to 0 (tiny n); nothing to drop
        tfr_trials[marker] = tfr[np.where(keep_mask)[0]]
        info["n_dropped"] = n_drop
    return report


def _cluster_timecourse(tfr_trials, cluster_channels, marker="200",
                        return_per_trial=False):
    """Cluster-averaged ERD%(t), log-space mean ± SE across trials.

    All averaging over (channels, freqs, trials) happens in LOG space; the
    result is converted to % once at the end. This is required because % is
    asymmetric (ERD floored at -100%, ERS unbounded), so an arithmetic % mean
    is dominated by ERS bins and washes out genuine ERD — see
    `_per_trial_cluster_logratio` for the SUBJ_008 evidence. Log-space
    averaging is the geometric mean of power ratios: it matches the topomap's
    `tfr.average()` (`generate_plots_config_a.py:157`) and the standard dB
    ERSP convention, so the timecourse, topomap and neuromod scalar agree.

    `mean_pct(t) = _logratio_to_pct(mean_trials(per_trial_logratio(t)))`, i.e.
    the dB display is `10·mean_logratio`. The SE band is symmetric in log
    space (→ asymmetric in %). Baseline sits at 0 by construction.

    The per-trial substrate exposed via `return_per_trial` is
    `per_trial_pct[n_trials, n_time]` = the per-trial cluster ERD% with the
    same log-space ch+freq averaging (the quality scorer reads it in %, taking
    a median across trials internally).

    Returns (times, mean_pct, low_pct, up_pct, n_trials, surviving_channels),
    optionally + per_trial_pct, or None.
    """
    if marker not in tfr_trials:
        return None
    tfr = tfr_trials[marker]
    per_trial_log, present = _per_trial_cluster_logratio(tfr, cluster_channels)
    if per_trial_log is None:
        return None
    n = per_trial_log.shape[0]
    if n < 1:
        return None
    # Per-trial substrate in % (log-averaged over ch+freq, then % once).
    per_trial_pct = _logratio_to_pct(per_trial_log)
    # Central tendency across trials in LOG space, then -> %.
    mean_log = per_trial_log.mean(axis=0)
    mean_pct = _logratio_to_pct(mean_log)
    if n > 1:
        sem_log = np.std(per_trial_log, axis=0, ddof=1) / np.sqrt(n)
        low_pct = _logratio_to_pct(mean_log - sem_log)
        up_pct = _logratio_to_pct(mean_log + sem_log)
    else:
        low_pct = mean_pct.copy()
        up_pct = mean_pct.copy()
    if return_per_trial:
        return (
            tfr.times, mean_pct,
            low_pct, up_pct,
            n, present, per_trial_pct,
        )
    return (
        tfr.times, mean_pct,
        low_pct, up_pct,
        n, present,
    )


# ----------------------------------------------------------------------
# Per-subject 6-panel figure (3 metric rows × 2 class cols)
# ----------------------------------------------------------------------

def _extract_session_traces(tfr_base, dropped_channels):
    """Return per-cluster trace dict for the 6-panel figure, with each
    cluster's trace extracted from its OWN cluster-matched rejected pool.

    `tfr_base` is the pre-rejection `tfr_trials` dict (one entry per
    marker). For each cluster (contra/bilat/ipsi), this function calls
    `_reject_artifact_trials_for_cluster(tfr_base, cluster_chs)` to get a
    cluster-specific kept-trial pool, then extracts the cluster's trace
    from that pool. Cluster trial counts can therefore DIFFER within one
    session — bilat-8 dilutes focal spikes more than contra-4, so bilat's
    pool is the most stringent (most trials kept) while contra/ipsi drop
    more aggressively when their cluster carries a focal artifact.

    The session-level `n_after_reject` in the npz meta is set to the
    bilateral pool's count (most stringent substrate). The scorer's
    session-level G2 (evaluate_erd_quality.py:479) reads that value, so
    keeping bilat as the canonical session-viability proxy preserves the
    canonical G2 semantics without modifying the scorer.

    The returned dict carries the 6-tuple traces (used by plotting and the
    CSV) under their `<cluster>_<marker>` keys, plus a `per_trial` sub-dict
    holding the per-trial cluster-mean ERD% matrix for each key — the
    substrate the quality scorer consumes (written to the npz side-car).
    Per-trial arrays are kilobytes (n_trials × n_time), so stashing them
    does not change peak RSS.
    """
    cluster_specs = [
        ("contra", CONTRA_MOTOR_CLUSTER),
        ("bilat",  BILATERAL_MOTOR_CLUSTER),
        ("ipsi",   IPSI_MOTOR_CLUSTER),
    ]
    traces: dict = {"dropped_channels": list(dropped_channels)}
    per_trial: dict = {}
    per_cluster_report: dict = {}
    for cluster_label, cluster_chs in cluster_specs:
        # Cluster-matched rejection: each cluster's trace uses its own
        # cleaned pool. Non-mutating: tfr_base is unchanged for the next
        # cluster's rejection pass.
        tfr_pool, rej = _reject_artifact_trials_for_cluster(
            tfr_base, cluster_chs,
        )
        per_cluster_report[cluster_label] = rej
        for marker_label, marker in (("mi", MI_MARKER),
                                      ("rest", REST_MARKER)):
            key = f"{cluster_label}_{marker_label}"
            res = _cluster_timecourse(
                tfr_pool, cluster_chs, marker, return_per_trial=True,
            )
            if res is None:
                traces[key] = None
                continue
            times, mean_pct, low_pct, up_pct, n, present, ptp = res
            traces[key] = (times, mean_pct, low_pct, up_pct, n, present)
            per_trial[key] = {
                "per_trial_pct": ptp,
                "times": times,
                "channels_used": present,
            }
    traces["per_trial"] = per_trial
    traces["_per_cluster_report"] = per_cluster_report
    return traces


# ----------------------------------------------------------------------
# Per-trial side-car (consumed by evaluate_erd_quality.py)
# ----------------------------------------------------------------------

def _write_per_trial_npz(out_path, subject, session, traces, out_meta):
    """Write one compressed npz holding the per-trial ERD% distribution for
    every (cluster, marker) of one session, plus session-level rejection
    metadata. This is the substrate the quality scorer reads — it never
    re-runs the TFR pass.

    Schema (all flat keys so the scorer needs no pickle):
      keys                       : comma-joined list of present cluster_marker keys
      <key>__ptp                 : float32 (n_trials, n_time) per-trial ERD%
      <key>__times               : float64 (n_time,) time axis (s)
      <key>__channels            : comma-joined surviving channel names
      subject, session           : str (0-d)
      n_attempted, n_kept        : int  (session-level, after µV + channel drop)
      n_after_reject             : int  (after trial-z rejection; == n_kept until
                                          Phase 4 rejection lands)
      dropped_channels           : comma-joined auto-dropped channel names
    Per-marker rejection counts are added by the Phase 4 rejection step.
    """
    per_trial = traces.get("per_trial", {})
    present_keys = [k for k, v in per_trial.items() if v is not None]
    payload: dict = {
        "keys": np.array(",".join(present_keys)),
        "subject": np.array(str(subject)),
        "session": np.array(str(session)),
        "n_attempted": np.array(int(out_meta.get("n_attempted", 0))),
        "n_kept": np.array(int(out_meta.get("n_kept", 0))),
        "n_after_reject": np.array(int(out_meta.get("n_after_reject",
                                                    out_meta.get("n_kept", 0)))),
        "dropped_channels": np.array(",".join(out_meta.get("dropped_channels", []))),
    }
    for key in present_keys:
        entry = per_trial[key]
        payload[f"{key}__ptp"] = np.asarray(
            entry["per_trial_pct"], dtype=np.float32,
        )
        payload[f"{key}__times"] = np.asarray(entry["times"], dtype=np.float64)
        payload[f"{key}__channels"] = np.array(
            ",".join(entry["channels_used"]),
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)


def _panel_score_tag(cls, times, _displayed_trace, ptp):
    """Compact per-session quality tag appended to a panel's legend entry.

    Computed with the same functions/thresholds `evaluate_erd_quality` uses,
    so what the figure annotation reports and what the scorer records cannot
    drift. The rubric is calibrated in PERCENT space and uses the MEDIAN
    across trials internally (scorer:553), so this helper derives the median
    trace from `ptp` regardless of what is actually plotted — the legend tag
    stays rubric-correct even when the display layer renders mean+dB.

    `_displayed_trace` is accepted for caller-compat but no longer read.
    """
    smask = _scalar_mask(times)
    # No per-trial data → nothing to score.
    if ptp is None or ptp.shape[0] == 0:
        return ""
    median_pct = np.median(ptp, axis=0)
    if cls == "mi":
        d1 = _d1_mi_strength(median_pct, smask)
        d2 = _d2_sustained(median_pct, smask)
        _d8, ratio = _d8_band_to_signal(ptp, median_pct, smask)
        bs = f" b/s={ratio:.1f}" if not np.isnan(ratio) else ""
        flag = ""
        pmask = _postcue_mask(times)
        if pmask.any():
            trial_peaks = np.max(np.abs(ptp[:, pmask]), axis=1)
            n_out = int((trial_peaks > G1_OUTLIER_PCT).sum())
            if n_out / ptp.shape[0] > G1_OUTLIER_FRAC:
                flag = f"  G1!({n_out}/{ptp.shape[0]}>{G1_OUTLIER_PCT:.0f}%)"
        return f" | D1={d1:.2f} sus={d2:.2f}{bs}{flag}"
    d4, eyes = _d4_rest_specificity(median_pct, smask)
    return f" | D4={d4:.2f}{'  ES!' if eyes else ''}"


def _pct_to_db(pct):
    """Convert ERD% (Pfurtscheller, P/P_bl - 1) to decibels (10·log10).

    dB = 10·log10(P/P_bl) = 10·log10(1 + pct/100). Same data, scale-symmetric
    in multiplicative power space. Baseline window at pct=0 maps to dB=0.

    Inverse of `_logratio_to_pct(x/10)`. Used at the display layer only —
    the per-trial substrate stays in % so the scorer (calibrated in %)
    keeps working unchanged.
    """
    return 10.0 * np.log10(1.0 + pct / 100.0)


def _plot_subject_6panel(subject, session_traces, out_path):
    """Plot 3 rows (Contra, Bilat, Ipsi) × 2 cols (MI, REST).

    Y-axis: ERD in dB (10·log10(P/P_baseline)), scale-symmetric in
    multiplicative power space. Conversion from the underlying % substrate
    happens here at display time only — per-trial npzs stay in % so the
    scorer keeps reading rubric-correct values. Y-axis is shared within
    each row so MI and REST are on the same scale for direct visual
    comparison. X-axis is shared across all panels.
    """
    if not session_traces:
        return
    fig, axes = plt.subplots(
        3, 2, figsize=(14, 11), sharex=True, sharey="row",
    )
    cmap = plt.get_cmap("viridis")
    n_sess = max(1, len(session_traces))
    # (row, class) -> (title, key_in_traces, cluster_for_legend)
    panel_specs = [
        (0, "mi",   "Contralateral ERD — MI",
         "contra_mi",   CONTRA_MOTOR_CLUSTER),
        (0, "rest", "Contralateral ERD — REST",
         "contra_rest", CONTRA_MOTOR_CLUSTER),
        (1, "mi",   "Bilateral ERD — MI",
         "bilat_mi",    BILATERAL_MOTOR_CLUSTER),
        (1, "rest", "Bilateral ERD — REST",
         "bilat_rest",  BILATERAL_MOTOR_CLUSTER),
        (2, "mi",   "Ipsilateral ERD — MI",
         "ipsi_mi",     IPSI_MOTOR_CLUSTER),
        (2, "rest", "Ipsilateral ERD — REST",
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
            # Display in dB; per-trial % substrate stays unchanged on disk
            # so the scorer keeps reading rubric-correct values.
            mean_db = _pct_to_db(mean_pct)
            low_db = _pct_to_db(low_pct)
            up_db = _pct_to_db(up_pct)
            ax.plot(times, mean_db, color=color, label=label, linewidth=1.4)
            ax.fill_between(times, low_db, up_db, color=color, alpha=0.15)
            ax.set_title(title)
            ax.axhline(0, color="k", lw=0.6)
            ax.axvline(0, color="k", ls="--", lw=0.7)
            ax.axvline(1.0, color="k", ls=":", lw=0.7)
            ax.grid(True, alpha=0.25)
            drew = True

    if not drew:
        plt.close(fig)
        return

    # Y-label on the left column of every row (was previously inside the
    # outer session loop with row=2 leaking from the inner loop, so only
    # the bottom row got labelled).
    for r in range(3):
        axes[r][0].set_ylabel("ERD (dB)")
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    # Legends on both columns: the MI and REST legends now carry different
    # per-session quality tags (MI: D1/sustained/band-signal/G1; REST: D4/ES),
    # so they are no longer interchangeable.
    for row in range(3):
        axes[row][0].legend(loc="best", fontsize=7)
        axes[row][1].legend(loc="best", fontsize=7)
    fig.suptitle(
        f"MU ERD across sessions — {subject} | MI vs REST\n"
        f"Contra: {CONTRA_MOTOR_CLUSTER} | "
        f"Bilateral: {BILATERAL_MOTOR_CLUSTER} | "
        f"Ipsi: {IPSI_MOTOR_CLUSTER}\n"
        f"{_preproc_caption()} | {_BAND_LABEL}\n"
        "legend tags: D1=MI strength, sus=sustained frac, b/s=band/signal, "
        "G1!=retained outlier; D4=REST specificity, ES!=eyes-closed",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Cohort 4-panel figure (2 cluster rows × 2 class cols)
# ----------------------------------------------------------------------

def _plot_cohort_6panel(cohort_traces, out_path):
    """Plot 3 rows (Contralateral, Bilateral, Ipsilateral) × 2 cols
    (MI, REST).

    Within each panel, one line per session_idx (1..N) colour-coded
    via viridis (light early → dark late), showing the cohort grand
    mean across subjects per session.

    cohort_traces: dict keyed by ("contra_mi" | "contra_rest" |
    "bilat_mi" | "bilat_rest" | "ipsi_mi" | "ipsi_rest") containing
    list of (subject, session_label, times, mean_pct).
    """
    fig, axes = plt.subplots(
        3, 2, figsize=(14, 11), sharex=True, sharey="row",
    )
    panels = [
        (0, 0, "Contralateral ERD% — MI",   "contra_mi"),
        (0, 1, "Contralateral ERD% — REST", "contra_rest"),
        (1, 0, "Bilateral ERD% — MI",       "bilat_mi"),
        (1, 1, "Bilateral ERD% — REST",     "bilat_rest"),
        (2, 0, "Ipsilateral ERD% — MI",     "ipsi_mi"),
        (2, 1, "Ipsilateral ERD% — REST",   "ipsi_rest"),
    ]
    # Determine all session indices present (across all four panels)
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

    for row, col, title, key in panels:
        ax = axes[row][col]
        traces = cohort_traces.get(key, [])
        per_idx: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        for subj, sess, times, mean_pct in traces:
            idx = session_idx_from_label(sess)
            per_idx.setdefault(idx, []).append((times, mean_pct))

        for idx in sorted(per_idx.keys()):
            entries = per_idx[idx]
            if not entries:
                continue
            t = entries[0][0]
            stack = np.stack(
                [e[1][:len(t)] for e in entries if len(e[1]) >= len(t)],
                axis=0,
            ) if entries else None
            if stack is None or stack.size == 0:
                continue
            # Cohort grand mean per session in % space, then convert to dB
            # at display time (matching the per-subject 6-panel convention).
            cohort_mean_pct = stack.mean(axis=0)
            cohort_mean_db = _pct_to_db(cohort_mean_pct)
            ax.plot(
                t, cohort_mean_db, color=colors[idx],
                label=f"S{idx:03d} (n={stack.shape[0]} subj)",
                linewidth=1.6,
            )
        ax.set_title(title)
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="k", ls="--", lw=0.7)
        ax.axvline(1.0, color="k", ls=":", lw=0.7)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        if col == 0:
            ax.set_ylabel("ERD (dB)")
        if row == 2:
            ax.set_xlabel("Time (s)")
    fig.suptitle(
        "CLIN cohort — MU ERD% by session index | MI vs REST "
        "(cohort grand mean per session)\n"
        f"{_preproc_caption()}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _redraw_from_csv(out_dir: Path):
    """Regenerate the per-subject 6-panel and cohort 6-panel from
    erd_refined_data.csv. Used to replot without re-running the
    ~25 min Config-A TFR pass.

    Backwards compat: if the CSV lacks a `marker` column it is from
    an older (MI-only) run; treat every row as MI and skip Rest.
    """
    csv_path = out_dir / "erd_refined_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found; re-run without --from-csv first."
        )
    df = pd.read_csv(csv_path)
    has_marker = "marker" in df.columns
    if not has_marker:
        print(
            "  [warn] CSV has no `marker` column (old schema); "
            "treating all rows as MI. Rerun without --from-csv to "
            "produce Rest panels."
        )
        df["marker"] = "mi"
    cohort_traces = {
        "contra_mi": [], "contra_rest": [],
        "bilat_mi":  [], "bilat_rest":  [],
        "ipsi_mi":   [], "ipsi_rest":   [],
    }
    cluster_to_key = {"contra": "contra", "bilat": "bilat", "ipsi": "ipsi"}
    for subject in sorted(df["subject"].unique()):
        sub = df[df.subject == subject]
        sessions_in_csv = list(sub["session"].drop_duplicates())
        session_traces: list[tuple[str, dict]] = []
        for sess in sessions_in_csv:
            s = sub[sub.session == sess].sort_values("t")
            traces = {
                "contra_mi":  None, "contra_rest": None,
                "bilat_mi":   None, "bilat_rest":  None,
                "ipsi_mi":    None, "ipsi_rest":   None,
                "dropped_channels": [],
            }
            for cluster in ("contra", "bilat", "ipsi"):
                for marker in ("mi", "rest"):
                    k = s[(s.cluster == cluster) & (s.marker == marker)]
                    if k.empty:
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
                    key = f"{cluster_to_key[cluster]}_{marker}"
                    traces[key] = (
                        times, mean_pct, low_pct, up_pct, n_trials, present,
                    )
                    cohort_traces[key].append(
                        (subject, sess, times, mean_pct),
                    )
            session_traces.append((sess, traces))
        if session_traces:
            sub_path = out_dir / f"{subject}_6panel_mi_rest.png"
            _plot_subject_6panel(subject, session_traces, str(sub_path))
            print(f"  wrote: {sub_path.name}")
    _plot_cohort_6panel(
        cohort_traces, str(out_dir / "cohort_6panel_mi_rest.png"),
    )
    print(f"Done (re-plot from CSV). Outputs at: {out_dir}")


def main():
    global TRIAL_REJECT_Z  # CLI --reject-z overrides the module default
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-csv", action="store_true",
        help=("Skip the Config-A TFR pass; redraw per-subject 6-panel "
              "and cohort 6-panel (contra/bilat/ipsi) from "
              "erd_refined_data.csv."),
    )
    parser.add_argument(
        "--subjects", default="",
        help=("Comma-separated subject filter for smoke tests, e.g. "
              "`CLIN_SUBJ_005,CLIN_SUBJ_007`. Empty = full cohort. "
              "Skips the per-subject cohort accumulation for subjects "
              "outside the filter so this only meaningfully reduces "
              "runtime when one or two subjects are specified."),
    )
    parser.add_argument(
        "--spatial-filter", choices=("car", "csd", "hjorth"), default="car",
        help=("Spatial filter for the Config-A preprocessing pass. "
              "Output filenames are tagged with the filter name so "
              "car/csd/hjorth variants coexist in erd_refined/ for "
              "side-by-side review."),
    )
    parser.add_argument(
        "--reject-z", type=float, default=TRIAL_REJECT_Z,
        help=("Robust-z threshold for per-trial artifact rejection over the "
              "bilateral motor montage (post-cue peak |logratio|). <=0 "
              "disables rejection (for the sweep's rejection-off arm). "
              f"Default {TRIAL_REJECT_Z:g}."),
    )
    parser.add_argument(
        "--window-end", type=float, default=None,
        help=("Post-cue end of the ERD analysis/display window in seconds "
              "(start fixed at -1). Default uses the config value "
              f"{CONFIG_A_DISPLAY_BASELINE['trial_win'][1]:g}s; pass 4 to "
              "compare against the original window."),
    )
    args = parser.parse_args()
    # Apply the spatial-filter override to the shared local CONFIG; the
    # figure caption reads this at plot time so it always matches.
    CONFIG_A_DISPLAY_BASELINE["spatial_filter"] = args.spatial_filter
    if args.window_end is not None:
        CONFIG_A_DISPLAY_BASELINE["trial_win"] = (-1.0, float(args.window_end))
    TRIAL_REJECT_Z = float(args.reject_z)
    subject_filter: set[str] = {
        s.strip() for s in args.subjects.split(",") if s.strip()
    }
    # Tag outputs by spatial filter (always) so the three variants
    # coexist, plus by subject when a --subjects filter is set so
    # smoke tests don't overwrite the full-cohort files.
    variant_tag = f"_{args.spatial_filter}"
    if subject_filter:
        variant_tag += "_subj-" + "-".join(
            s.replace("CLIN_SUBJ_", "") for s in sorted(subject_filter)
        )
        print(f"[variant] subjects={sorted(subject_filter)} → tag '{variant_tag}'")
    # Tag non-default preprocessing arms so sweep variants coexist without
    # overwriting the default (rejection on, 5 s window) outputs.
    if TRIAL_REJECT_Z <= 0:
        variant_tag += "_noreject"
    _win_end = CONFIG_A_DISPLAY_BASELINE["trial_win"][1]
    if abs(_win_end - 5.0) > 1e-9:
        variant_tag += f"_w{_win_end:g}"

    out_dir = clin_pictures_root() / "erd_refined"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        _redraw_from_csv(out_dir)
        return

    cohort_traces = {
        "contra_mi": [], "contra_rest": [],
        "bilat_mi":  [], "bilat_rest":  [],
        "ipsi_mi":   [], "ipsi_rest":   [],
    }
    csv_rows = []

    # Map (cluster_label, marker_label) -> (key_in_traces_dict, cluster_list)
    cluster_specs = [
        ("contra", "mi",   "contra_mi",   CONTRA_MOTOR_CLUSTER),
        ("contra", "rest", "contra_rest", CONTRA_MOTOR_CLUSTER),
        ("bilat",  "mi",   "bilat_mi",    BILATERAL_MOTOR_CLUSTER),
        ("bilat",  "rest", "bilat_rest",  BILATERAL_MOTOR_CLUSTER),
        ("ipsi",   "mi",   "ipsi_mi",     IPSI_MOTOR_CLUSTER),
        ("ipsi",   "rest", "ipsi_rest",   IPSI_MOTOR_CLUSTER),
    ]

    for subject in enumerate_clin_subjects():
        if subject_filter and subject not in subject_filter:
            continue
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
            tfr_base = out["tfr_trials"]
            # Cluster-matched per-trial artifact rejection runs INSIDE
            # _extract_session_traces (each cluster gets its own pool;
            # tfr_base is not mutated). The session-level n_after_reject
            # reported here is taken from the bilateral pool (most
            # stringent substrate; canonical G2 reads this single value).
            traces = _extract_session_traces(
                tfr_base, out.get("dropped_channels", []),
            )
            per_cluster_report = traces.pop("_per_cluster_report", {})
            bilat_rep = per_cluster_report.get("bilat", {})
            n_after_reject = sum(
                info["n_before"] - info["n_dropped"]
                for info in bilat_rep.values()
            ) if bilat_rep else out.get("n_kept", 0)
            # Combined reject_report for the console line (per-cluster MI
            # drops, the publication-line metric for quick scan).
            reject_report = bilat_rep  # legacy compat: bilat treated as "the" report
            session_traces.append((sess, traces))

            # Per-trial side-car for the quality scorer. Written from the
            # already-extracted small arrays (not the TFR), so peak RSS is
            # unchanged; lands beside the figures, tagged by spatial filter.
            _write_per_trial_npz(
                out_dir / "per_trial" / f"{subject}_{sess}{variant_tag}.npz",
                subject, sess, traces,
                {
                    "n_attempted": out.get("n_attempted", 0),
                    "n_kept": out.get("n_kept", 0),
                    "n_after_reject": n_after_reject,
                    "dropped_channels": out.get("dropped_channels", []),
                },
            )

            # Cluster-mean traces for the cohort figure + CSV
            for cluster_label, marker_label, key, cluster_chs in cluster_specs:
                res = traces[key]
                if res is None:
                    continue
                times, mean_pct, low_pct, up_pct, n_trials, present = res
                cohort_traces[key].append(
                    (subject, sess, times, mean_pct)
                )
                for t_idx, t_val in enumerate(times):
                    csv_rows.append({
                        "subject": subject,
                        "session": sess,
                        "session_idx": session_idx_from_label(sess),
                        "cluster": cluster_label,
                        "marker": marker_label,
                        "channels_used": ",".join(present),
                        "t": float(t_val),
                        "mean_pct": float(mean_pct[t_idx]),
                        "low_pct": float(low_pct[t_idx]),
                        "up_pct": float(up_pct[t_idx]),
                        "n_trials": int(n_trials),
                    })
            rej = " ".join(
                f"{m}:-{r['n_dropped']}"
                + ("(capped)" if r["over_gate"] else "")
                for m, r in reject_report.items()
            ) or "—"
            print(
                f"  {sess}: n_kept={out['n_kept']}/{out['n_attempted']} "
                f"reject[{rej}] n_after={n_after_reject} "
                f"dropped={out['dropped_channels'] or '—'} "
                f"({time.time()-t0:.1f}s)"
            )
            # Release heavy TFR objects immediately
            del out, tfr_base
            gc.collect()

        if session_traces:
            sub_path = (
                out_dir / f"{subject}_6panel_mi_rest{variant_tag}.png"
            )
            _plot_subject_6panel(subject, session_traces, str(sub_path))
            print(f"  wrote: {sub_path.name}")
        del session_traces
        gc.collect()

    # Cohort figure
    _plot_cohort_6panel(
        cohort_traces,
        str(out_dir / f"cohort_6panel_mi_rest{variant_tag}.png"),
    )

    df = pd.DataFrame(csv_rows)
    df.to_csv(out_dir / f"erd_refined_data{variant_tag}.csv", index=False)
    print(f"\nDone. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
