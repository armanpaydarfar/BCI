#!/usr/bin/env python3
"""EDS topoplot replication for the CLIN_* (ALS) cohort (Pass 1).

Implements the Kumar 2024 Fig 5 EDS analysis on the ALS cohort, per
`Documents/SoftwareDocs/projects/harmony-bci/clinical-analysis/rev01-eds-analysis-plan.md`
(REV01.1). Decoder family is MDM-on-mu-cov; prototypes are recomputed
offline from the per-CLIN-subject `training_data/` pool using
non-recentered (Kumar "unmatched feature distributions") covariances.

Analysis-only. Tier 1 / Tier 2 files are READ-ONLY per CLAUDE.md;
covariance + shrinkage math is replicated inline from stable
`Utils/runtime_common.py:248-255` with citation comments.

Outputs (`~/Pictures/clin_analysis_pass1/eds/`):
    expert_eds_topoplot_mu.png
    cohort_eds_topoplot_mu.png
    cohort_minus_expert_eds_topoplot_mu.png
    per_subject_eds_topoplot_mu_grid.png
    eds_per_subject_session_mu.csv

With `--include-beta`, the four PNG/CSV companions for the beta band
are also produced.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from pyriemann.estimation import Shrinkage
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_riemann
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.covariance import LedoitWolf

# Make sweep helpers + clinical_analysis package importable
_REPO_ROOT = Path(__file__).resolve().parent
_SWEEP_DIR = _REPO_ROOT / "exploration" / "preprocessing_sweep"
for _p in (str(_REPO_ROOT), str(_SWEEP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exploration.clinical_analysis._helpers import (  # noqa: E402
    DATA_DIR,
    clin_pictures_root,
    enumerate_clin_subjects,
    enumerate_online_sessions_for_subject,
)

# Config A preprocessing constants (sweep_phase2_round2.py:63-73)
from sweep_phase2_round2 import (  # noqa: E402
    BB_HI, BB_LO, FS, MU_HI, MU_LO, NOTCH, REJECT_MAX_ABS_UV,
    apply_blink_removal, apply_spatial_filter, load_raw_cached,
)

# Auto-drop loop knobs (sweep_phase3_validation.py:123-126)
from sweep_phase3_validation import (  # noqa: E402
    AUTO_DROP_DOMINANCE_FRAC, AUTO_DROP_MAX_CHANNELS,
    AUTO_DROP_MAX_ITERS, AUTO_DROP_REJECT_FRAC,
    _pick_dominant_bad_channel_max_abs,
)

# Tier 1 read-only loader citations:
#   Utils.stream_utils — analysis-friendly per CLAUDE.md
#   Utils.preprocessing.concatenate_streams — same
from Utils.preprocessing import concatenate_streams  # noqa: E402
from Utils.stream_utils import (  # noqa: E402
    get_channel_names_from_xdf, load_xdf,
)

mne.set_log_level("ERROR")

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

SCALAR_WIN = (1.0, 4.0)
TRIAL_WIN = (-1.0, 4.0)
BETA_LO, BETA_HI = 13.0, 30.0

# 15-channel motor subset per config.py:26 (= harmony_stable:config.py:14)
MOTOR_CHANNEL_NAMES = [
    "FC1", "FC2", "C3", "Cz", "C4", "CP5", "CP1", "CP2", "CP6",
    "P7",  "P3",  "Pz", "P4", "P8", "POz",
]

# Full 22-channel montage per `rev01-eds-analysis-plan.md` §6.2 (the
# Kumar 2024 montage minus the Fp channels — drop_fp is still applied,
# so the actual EDS runs on the subset of this list that survives the
# blink-removal stage). Adds frontal F-row and FC5/FC6 on top of the
# motor15 set.
FULL22_CHANNELS = [
    "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "C3", "Cz", "C4",
    "CP5", "CP1", "CP2", "CP6",
    "P7", "P3", "Pz", "P4", "P8", "POz",
]

# Shared expert pool — the 6 OG_Right CLASS+LAB files identical across
# CLIN_SUBJ_003..008/training_data/ per rev01-paper-angle.md §1.1.
EXPERT_OG_RIGHT_BASENAMES = [
    "sub-CLASS_SUBJ_1032_ses-S001OFFLINE_FES_task-Default_run-001_eeg.xdf",
    "sub-CLASS_SUBJ_1032_ses-S002OFFLINE_NOFES_task-Default_run-001_eeg.xdf",
    "sub-CLASS_SUBJ_132_ses-S002OFFLINE_NOFES_task-Default_run-001_eeg.xdf",
    "sub-CLASS_SUBJ_831_ses-S002OFFLINE_NOFES_task-Default_run-001_eeg.xdf",
    "sub-LAB_SUBJ_001_ses-S001OFFLINE_task-Default_run-001_eeg.xdf",
    "sub-LAB_SUBJ_001_ses-S007OFFLINE_task-Default_run-001_eeg.xdf",
]
EXPERT_SOURCE_SUBJECT = "CLIN_SUBJ_003"   # any CLIN_SUBJ_003..008 works

CLIN_PRIMARY_SUBJECTS = [f"CLIN_SUBJ_{i:03d}" for i in (3, 4, 5, 6, 7, 8)]


# ----------------------------------------------------------------------
# Raw → time-domain mu/beta epoch builder (Config-A preprocessing,
# without TFR — EDS works directly on time-domain covariances)
# ----------------------------------------------------------------------

def _load_raw_from_xdf_path(xdf_path: str):
    """Single-XDF analog of sweep_phase2_round2.load_raw_cached.

    Mirrors the loader logic (xdf -> MNE Raw -> rename -> standard_1020
    montage -> chronological marker pairing). Required because the
    expert-pool XDFs live in `training_data/`, not in `ses-*/eeg/`.
    """
    eeg_stream, marker_stream = load_xdf(xdf_path)
    eeg_timestamps = np.array(eeg_stream["time_stamps"])
    eeg_data_raw = np.array(eeg_stream["time_series"]).T
    channel_names = get_channel_names_from_xdf(eeg_stream)
    marker_data = np.array([int(v[0]) for v in marker_stream["time_series"]])
    marker_timestamps = np.array(
        [float(v[1]) for v in marker_stream["time_series"]]
    )

    non_eeg = {"AUX1", "AUX2", "AUX3", "AUX7", "AUX8", "AUX9", "TRIGGER"}
    valid_idx = [i for i, ch in enumerate(channel_names) if ch not in non_eeg]
    eeg_data = eeg_data_raw[valid_idx, :]
    valid_ch = [channel_names[i] for i in valid_idx]

    info = mne.create_info(valid_ch, FS, "eeg")
    raw = mne.io.RawArray(eeg_data, info, verbose=False)
    if "M1" in raw.ch_names and "M2" in raw.ch_names:
        raw.drop_channels(["M1", "M2"])
    rename = {
        "FP1": "Fp1", "FPZ": "Fpz", "FP2": "Fp2", "FZ": "Fz",
        "CZ":  "Cz",  "PZ":  "Pz",  "POZ": "POz", "OZ": "Oz",
    }
    raw.rename_channels(rename)
    raw.set_montage(
        mne.channels.make_standard_montage("standard_1020"),
        match_case=True, on_missing="warn",
    )

    min_dur, max_dur, EPS = 1.0, 5.5, 0.02
    valid_start = []
    for idx, code in enumerate(marker_data):
        if code not in (100, 200):
            continue
        end_code = code + 20
        end_time = None
        for j in range(idx + 1, len(marker_data)):
            if marker_data[j] == end_code:
                end_time = marker_timestamps[j]
                break
        if end_time is None:
            continue
        duration = end_time - marker_timestamps[idx]
        if (duration + EPS) >= min_dur and (duration - EPS) <= max_dur:
            valid_start.append(idx)

    mdata = [marker_data[i] for i in valid_start]
    mtimes = [marker_timestamps[i] for i in valid_start]
    event_dict = {str(c): c for c in set(mdata)}
    samples = np.searchsorted(eeg_timestamps, mtimes)
    events = np.column_stack(
        (samples, np.zeros(len(mdata), dtype=int), mdata)
    )
    return raw, events, event_dict


def _config_a_epochs(
    raw, events, event_dict, band: tuple[float, float],
    *,
    blink_removal: str = "drop_fp",
    spatial_filter: str = "car",
):
    """Apply Config A preprocessing and return time-domain epochs at
    `band` cropped to SCALAR_WIN = (1, 4) s.

    Pipeline (matches `generate_plots_config_a.py:88-161`):
      1. Notch + 4-40 Hz broadband (IIR Butterworth).
      2. Blink removal: `drop_fp` (Config A default; drops Fp1/Fp2/Fpz)
         or `fp_regression` (regress Fp channels out of others while
         keeping them; `sweep_phase2_round2.py:225-226`).
      3. Auto-drop loop on mu-filtered epochs at 50 µV (sweep_phase3
         knobs).
      4. Spatial filter: `car` (Config A default), `csd`, or `hjorth`
         per `sweep_phase2_round2.py:253-261`.
      5. Bandpass to `band` (mu or beta).
      6. Epoch on TRIAL_WIN, keep `good_ix` only, crop to SCALAR_WIN.

    Returns (data_array, labels) where data has shape (n_kept, n_ch, n_t)
    and labels are the marker codes (100/200).
    """
    raw_bb = raw.copy()
    raw_1hz = raw.copy()
    raw_bb.notch_filter(NOTCH, method="iir", verbose=False)
    raw_bb.filter(l_freq=BB_LO, h_freq=BB_HI, method="iir", verbose=False)
    raw_bb, _ = apply_blink_removal(raw_bb, raw_1hz, blink_removal)

    # Auto-drop loop on mu-filtered copy. Same logic as
    # generate_plots_config_a.py:103-131.
    dropped = []
    iters = 0
    t0, t1 = TRIAL_WIN
    while True:
        iters += 1
        raw_mu = raw_bb.copy()
        raw_mu.filter(
            l_freq=MU_LO, h_freq=MU_HI, method="iir", verbose=False,
        )
        epoch_kw = dict(
            event_id=event_dict,
            tmin=t0, tmax=t1,
            baseline=None, detrend=1, preload=True, verbose=False,
        )
        epochs_mu = mne.Epochs(
            raw_mu, events, reject=None, flat=None, **epoch_kw,
        )
        epochs_bb = mne.Epochs(
            raw_bb, events, reject=None, flat=None, **epoch_kw,
        )
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

    if not good_ix:
        return None, None, dropped, []

    # Spatial filter (Config A default = "car"; can be overridden via
    # --spatial-filter to test CSD or Hjorth).
    epochs_bb = apply_spatial_filter(epochs_bb, spatial_filter)

    # Filter into the requested band on the spatial-filtered epochs by
    # reconstructing a Raw at this stage is heavy; instead, apply the
    # IIR filter to the epoch data using the same IIR filter MNE would
    # construct via Raw.filter. Operate on epochs.copy().filter which
    # uses the same IIR backend.
    epochs_band = epochs_bb.copy().filter(
        l_freq=band[0], h_freq=band[1], method="iir", verbose=False,
    )

    epochs_band = epochs_band[good_ix]
    epochs_band.crop(tmin=SCALAR_WIN[0], tmax=SCALAR_WIN[1])

    # Labels — pull the event codes from epochs.events[:, 2]
    labels = epochs_band.events[:, 2].astype(int)
    data = epochs_band.get_data()
    ch_names = list(epochs_band.ch_names)
    return data, labels, dropped, ch_names


# ----------------------------------------------------------------------
# Trial-level covariance estimation (replicates stable runtime math)
# ----------------------------------------------------------------------

def _trace_normalised_cov(x: np.ndarray) -> np.ndarray:
    """C = X X^T / trace(X X^T) per stable Utils/runtime_common.py:248
    and Generate_Riemannian_adaptive.py:269. Kumar 2024 Eq. 1."""
    c = x @ x.T
    tr = np.trace(c)
    if tr <= 0 or not np.isfinite(tr):
        return c
    return c / tr


def _shrink_pyriemann(covs: np.ndarray, shrinkage_param: float) -> np.ndarray:
    """pyriemann.estimation.Shrinkage with the requested λ. Matches
    stable Utils/runtime_common.py:252-254."""
    shr = Shrinkage(shrinkage=shrinkage_param)
    return shr.fit_transform(covs)


def _shrink_ledoitwolf(covs: np.ndarray) -> np.ndarray:
    """LedoitWolf().fit(cov).covariance_ — used for CLIN_SUBJ_002's
    older configuration. Matches stable Utils/runtime_common.py:251."""
    out = np.zeros_like(covs)
    for i, c in enumerate(covs):
        # LedoitWolf expects samples-as-rows; we feed an (n_ch, n_ch)
        # SPD matrix as if it were samples — that is the same call the
        # runtime makes at runtime_common.py:251. Reproduce verbatim.
        out[i] = LedoitWolf().fit(c).covariance_
    return out


def trial_covs(
    epoch_data: np.ndarray, *,
    use_ledoitwolf: bool, shrinkage_param: float,
) -> np.ndarray:
    """Per-trial trace-normalised + shrunk covariance, shape (N, C, C).

    `use_ledoitwolf` triggers the CLIN_SUBJ_002 branch (sklearn
    LedoitWolf, older config). Else uses pyriemann Shrinkage with the
    given λ.
    """
    raw_covs = np.stack(
        [_trace_normalised_cov(seg) for seg in epoch_data], axis=0,
    )
    if use_ledoitwolf:
        return _shrink_ledoitwolf(raw_covs)
    return _shrink_pyriemann(raw_covs, shrinkage_param)


# ----------------------------------------------------------------------
# EDS computation (Kumar 2024 p. 13 §EDS)
# ----------------------------------------------------------------------

def _safe_mean_riemann(covs: np.ndarray) -> np.ndarray | None:
    """Karcher mean with finiteness check. Returns None on failure."""
    if len(covs) == 0:
        return None
    try:
        m = mean_riemann(covs)
    except Exception:
        return None
    if not np.all(np.isfinite(m)):
        return None
    return m


def eds_from_class_covs(
    cov_mi: np.ndarray, cov_rest: np.ndarray, n_channels: int,
) -> np.ndarray | None:
    """EDS via backward elimination per Kumar 2024 p. 13.

    e_i = P22 - P21(i)
        P22 = 1 / (1 + exp(-δ_r(C̄_MI, C̄_Rest)²))
        P21(i) = same with row/col i deleted from both prototypes

    Returns shape (n_channels,) or None if Karcher means fail to
    converge.
    """
    c_mi = _safe_mean_riemann(cov_mi)
    c_rest = _safe_mean_riemann(cov_rest)
    if c_mi is None or c_rest is None:
        return None
    d22_sq = distance_riemann(c_mi, c_rest) ** 2
    p22 = 1.0 / (1.0 + np.exp(-d22_sq))
    out = np.zeros(n_channels)
    for i in range(n_channels):
        mi_drop = np.delete(np.delete(c_mi, i, axis=0), i, axis=1)
        rest_drop = np.delete(np.delete(c_rest, i, axis=0), i, axis=1)
        d21_sq = distance_riemann(mi_drop, rest_drop) ** 2
        p21 = 1.0 / (1.0 + np.exp(-d21_sq))
        out[i] = p22 - p21
    return out


# ----------------------------------------------------------------------
# Channel selection: restrict to motor subset, return aligned data/ch_names
# ----------------------------------------------------------------------

def _restrict_to_motor(
    data: np.ndarray, ch_names: list[str], motor_channels: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Return data restricted to the intersection of `ch_names` and
    `motor_channels`, preserving `motor_channels` order. Channels in
    the spec that are absent in `ch_names` (auto-dropped or off-cap)
    are silently skipped — the returned list is the subset that
    survived for this session."""
    keep = [c for c in motor_channels if c in ch_names]
    if not keep:
        return data[:, :0, :], []
    idx = [ch_names.index(c) for c in keep]
    return data[:, idx, :], keep


# ----------------------------------------------------------------------
# Expert pool EDS
# ----------------------------------------------------------------------

def expert_eds_shared(
    band: tuple[float, float], motor_channels: list[str], band_label: str,
    *,
    blink_removal: str = "drop_fp",
    spatial_filter: str = "car",
) -> tuple[np.ndarray | None, list[str], int]:
    """Compute EDS once on the shared 6-file OG_Right expert pool.

    Returns (eds_vec, channels_actually_used, n_trials_total).
    """
    expert_dir = os.path.join(
        DATA_DIR, f"sub-{EXPERT_SOURCE_SUBJECT}", "training_data",
    )
    cov_mi_list: list[np.ndarray] = []
    cov_rest_list: list[np.ndarray] = []
    used_channels: list[str] = []

    for basename in EXPERT_OG_RIGHT_BASENAMES:
        xdf = os.path.join(expert_dir, basename)
        if not os.path.exists(xdf):
            print(f"  [expert/{band_label}] MISSING {basename}; skip")
            continue
        try:
            raw, events, event_dict = _load_raw_from_xdf_path(xdf)
        except Exception as e:
            print(
                f"  [expert/{band_label}] FAILED load {basename}: "
                f"{type(e).__name__}: {e}"
            )
            continue
        try:
            data, labels, dropped, ch_names = _config_a_epochs(
                raw, events, event_dict, band,
                blink_removal=blink_removal,
                spatial_filter=spatial_filter,
            )
        except Exception as e:
            print(
                f"  [expert/{band_label}] FAILED preproc {basename}: "
                f"{type(e).__name__}: {e}"
            )
            continue
        if data is None or len(data) == 0:
            print(f"  [expert/{band_label}] no kept epochs in {basename}")
            continue

        data_motor, motor_kept = _restrict_to_motor(
            data, ch_names, motor_channels,
        )
        if not motor_kept:
            continue

        # Set a unified expert channel list lazily on the first file
        # then enforce alignment by intersecting with subsequent files.
        if not used_channels:
            used_channels = motor_kept
        else:
            # If a later file lost a channel via auto-drop, restrict to
            # the intersection so the covariance shapes stack.
            intersect = [c for c in used_channels if c in motor_kept]
            if intersect != used_channels:
                # Trim previously accumulated covariances to intersect
                old = used_channels
                used_channels = intersect
                if not used_channels:
                    print(
                        f"  [expert/{band_label}] channel intersection "
                        f"emptied at {basename}; aborting expert build"
                    )
                    return None, [], 0
                # Convert old cov_lists from their original channel
                # set to the intersected set
                old_idx = [old.index(c) for c in intersect]
                cov_mi_list = [c[np.ix_(old_idx, old_idx)] for c in cov_mi_list]
                cov_rest_list = [c[np.ix_(old_idx, old_idx)] for c in cov_rest_list]
            # restrict the current file's data to the intersection
            idx_now = [motor_kept.index(c) for c in used_channels]
            data_motor = data_motor[:, idx_now, :]

        covs = trial_covs(
            data_motor, use_ledoitwolf=False, shrinkage_param=0.02,
        )
        cov_mi_list.extend(covs[labels == 200])
        cov_rest_list.extend(covs[labels == 100])
        print(
            f"  [expert/{band_label}] {basename}: kept {len(data)} "
            f"epochs, dropped channels {dropped or '—'}"
        )

    if not cov_mi_list or not cov_rest_list:
        return None, [], 0
    cov_mi = np.stack(cov_mi_list, axis=0)
    cov_rest = np.stack(cov_rest_list, axis=0)
    eds = eds_from_class_covs(cov_mi, cov_rest, len(used_channels))
    n_trials = len(cov_mi) + len(cov_rest)
    return eds, used_channels, n_trials


# ----------------------------------------------------------------------
# Per (subject, session) EDS
# ----------------------------------------------------------------------

def per_session_eds(
    subject: str, session: str, band: tuple[float, float],
    motor_channels: list[str], band_label: str,
    *,
    blink_removal: str = "drop_fp",
    spatial_filter: str = "car",
) -> tuple[np.ndarray | None, list[str], int]:
    """EDS for one CLIN session.

    Uses Config A preprocessing via load_raw_cached + _config_a_epochs.
    For CLIN_SUBJ_002, uses LedoitWolf shrinkage (older config) per
    rev01-paper-angle.md §1.1.
    """
    try:
        raw, events, event_dict = load_raw_cached(subject, session)
    except Exception as e:
        print(
            f"  [{subject}/{session}/{band_label}] FAILED load: "
            f"{type(e).__name__}: {e}"
        )
        return None, [], 0

    try:
        data, labels, dropped, ch_names = _config_a_epochs(
            raw, events, event_dict, band,
            blink_removal=blink_removal,
            spatial_filter=spatial_filter,
        )
    except Exception as e:
        print(
            f"  [{subject}/{session}/{band_label}] FAILED preproc: "
            f"{type(e).__name__}: {e}"
        )
        return None, [], 0

    if data is None or len(data) == 0:
        return None, [], 0

    data_motor, motor_kept = _restrict_to_motor(
        data, ch_names, motor_channels,
    )
    if len(motor_kept) < 2:
        return None, motor_kept, 0

    # CLIN_SUBJ_002 used LedoitWolf + 0.1 shrinkage per its
    # config_snapshot.json (rev01-paper-angle.md §1.1).
    use_lw = (subject == "CLIN_SUBJ_002")
    shr = 0.1 if use_lw else 0.02
    covs = trial_covs(
        data_motor, use_ledoitwolf=use_lw, shrinkage_param=shr,
    )
    cov_mi = covs[labels == 200]
    cov_rest = covs[labels == 100]
    if len(cov_mi) < 5 or len(cov_rest) < 5:
        return None, motor_kept, len(cov_mi) + len(cov_rest)
    eds = eds_from_class_covs(cov_mi, cov_rest, len(motor_kept))
    return eds, motor_kept, len(cov_mi) + len(cov_rest)


# ----------------------------------------------------------------------
# Topomap plotting
# ----------------------------------------------------------------------

def _make_info_for_channels(channels: list[str]):
    """Build a minimal Info with standard_1020 positions for plotting."""
    info = mne.create_info(channels, sfreq=FS, ch_types="eeg")
    info.set_montage(
        mne.channels.make_standard_montage("standard_1020"),
        match_case=True, on_missing="warn",
    )
    return info


def _plot_topomap_panel(
    z_vec: np.ndarray, channels: list[str], title: str,
    save_path: str, cmap: str = "viridis",
    *, mask: np.ndarray | None = None,
):
    """Plot one EDS z-score topomap. Uses `names=channels` per MNE
    1.9.0 API (legacy `show_names=True` removed).

    `mask` (optional) is a boolean array marking electrodes that pass a
    Bonferroni-adjusted significance test (C4 fix). Marked electrodes
    are drawn with a white circle via MNE's `mask` argument per
    `rev01-eds-analysis-plan.md` §5.1 / §5.2.
    """
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    info = _make_info_for_channels(channels)
    kw: dict = dict(
        axes=ax, cmap=cmap, show=False, names=channels, vlim=(None, None),
    )
    if mask is not None:
        kw["mask"] = mask
        kw["mask_params"] = dict(
            marker="o", markerfacecolor="white", markeredgecolor="black",
            linewidth=0, markersize=8,
        )
    im, _ = mne.viz.plot_topomap(z_vec, info, **kw)
    fig.colorbar(im, ax=ax, shrink=0.8, label="EDS z-score")
    ax.set_title(title, fontsize=10)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Per-electrode significance (Wilcoxon + Bonferroni)
# ----------------------------------------------------------------------

def _wilcoxon_per_channel(
    arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided Wilcoxon signed-rank against zero, per column.

    `arr` shape (n_subjects, n_channels). Returns (W, p), each shape
    (n_channels,). NaN p if a column is all zeros (degenerate).
    """
    n_channels = arr.shape[1]
    W = np.full(n_channels, np.nan)
    p = np.full(n_channels, np.nan)
    for c in range(n_channels):
        col = arr[:, c]
        col_clean = col[np.isfinite(col)]
        if len(col_clean) < 1 or np.allclose(col_clean, 0.0):
            continue
        try:
            res = wilcoxon(
                col_clean, alternative="two-sided",
                zero_method="wilcox",
            )
            W[c] = float(res.statistic)
            p[c] = float(res.pvalue)
        except Exception:
            continue
    return W, p


def _wilcoxon_paired_per_channel(
    a: np.ndarray, b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Paired two-sided Wilcoxon signed-rank of `a - b` against zero,
    per channel. `a` shape (n_subjects, n_channels); `b` shape
    (n_channels,) (broadcast across subjects to form the diff array).
    Per `rev01-eds-analysis-plan.md` §5.2.
    """
    diff = a - b[np.newaxis, :]
    return _wilcoxon_per_channel(diff)


def _plot_per_subject_grid(
    per_subj_eds: dict[str, tuple[np.ndarray, list[str]]],
    title: str, save_path: str, cmap: str = "viridis",
):
    """Per-subject EDS topomap grid. Each panel z-scores its own EDS
    vector across electrodes; the colorbar is shared with a unified
    `vlim` derived from the global min/max of all panels' z-scores so
    cross-panel comparisons via the shared colorbar are meaningful
    (Mi6 fix — pass 1 used per-panel vlim with a single colorbar,
    misrepresenting the cross-subject mapping).
    """
    n = len(per_subj_eds)
    if n == 0:
        return
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
        constrained_layout=True, squeeze=False,
    )
    items = sorted(per_subj_eds.items())
    # Pre-compute z-vectors and find shared vlim
    panels: list[tuple[str, np.ndarray, list[str]]] = []
    for subj, (eds_vec, channels) in items:
        sd = eds_vec.std(ddof=1)
        z = (eds_vec - eds_vec.mean()) / sd if sd > 0 else np.zeros_like(eds_vec)
        panels.append((subj, z, channels))
    all_z = np.concatenate([z for _, z, _ in panels])
    vmax = float(np.nanmax(np.abs(all_z))) if len(all_z) else 1.0
    vlim = (-vmax, vmax) if vmax > 0 else (None, None)
    last_im = None
    for ax_idx, (subj, z, channels) in enumerate(panels):
        row, col = divmod(ax_idx, n_cols)
        ax = axes[row][col]
        info = _make_info_for_channels(channels)
        im, _ = mne.viz.plot_topomap(
            z, info, axes=ax, cmap=cmap, show=False, names=channels,
            vlim=vlim,
        )
        last_im = im
        ax.set_title(subj, fontsize=10)
    # blank any unused axes
    for k in range(len(items), n_rows * n_cols):
        row, col = divmod(k, n_cols)
        axes[row][col].axis("off")
    if last_im is not None:
        fig.colorbar(
            last_im, ax=axes, shrink=0.6,
            label=f"EDS z-score (shared vlim ±{vmax:.2f})",
        )
    fig.suptitle(title, fontsize=12)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def run_for_band(
    band_label: str, band: tuple[float, float],
    cohort_subjects: list[str], out_dir: Path,
    include_clin002: bool,
    from_csv: bool = False,
    *,
    channel_set: str = "motor15",
    blink_removal: str = "drop_fp",
    spatial_filter: str = "car",
    no_diff_plot: bool = False,
    variant_tag: str = "",
):
    """Build EDS, plot all four topomaps, and dump the CSV for one band.

    `from_csv=True` skips Karcher-mean computation and reloads
    `eds_per_subject_session_{band}{variant_tag}.csv` from a prior run.
    Note: requires the prior run to have written a non-degenerate CSV
    for the requested band + variant.

    `channel_set` selects the EDS channel pool: "motor15" (the deployed
    decoder subset, default) or "full22" (Kumar 2024 montage minus
    Fp's). `blink_removal` and `spatial_filter` are passed through to
    Config-A preprocessing. `no_diff_plot` skips the
    cohort_minus_expert topomap (the diff plot is harder to interpret;
    the separate expert + cohort plots cover the same content).
    `variant_tag` is appended to every output filename so non-default
    runs don't overwrite the default pass-1 outputs.
    """
    if channel_set == "full22":
        motor_channels = FULL22_CHANNELS
    else:
        motor_channels = MOTOR_CHANNEL_NAMES

    # 1. Expert EDS
    if from_csv:
        # Reuse a cached expert EDS if available; the prior pass-1 CSV
        # does not carry per-channel expert EDS, so we still need to
        # recompute the expert pool. There is no shortcut — but it's
        # ~3 min for n=6 expert files, not 22 min.
        print(f"\n=== Expert pool ({band_label}) — recomputing (no CSV cache) ===")
    else:
        print(f"\n=== Expert pool ({band_label}) ===")
    expert_eds, expert_channels, expert_n = expert_eds_shared(
        band, motor_channels, band_label,
        blink_removal=blink_removal,
        spatial_filter=spatial_filter,
    )
    if expert_eds is None:
        print(f"  ! expert EDS failed for {band_label}; skipping band")
        return

    # 2. Per-subject + per-session EDS (csv) and per-subject EDS (last 2)
    per_subj_eds: dict[str, tuple[np.ndarray, list[str]]] = {}
    rows = []

    if from_csv:
        csv_path = (
            out_dir / f"eds_per_subject_session_{band_label}{variant_tag}.csv"
        )
        if not csv_path.exists():
            print(
                f"  --from-csv requested but missing {csv_path}; "
                "rerun without --from-csv first."
            )
            return
        df_cached = pd.read_csv(csv_path)
        # Reconstruct per-session EDS vectors from the long-form CSV.
        # `rows` retains the original long-form rows so the new CSV
        # written at the end is a superset (long-form rows + extra
        # significance/raw columns at the end).
        rows = df_cached.to_dict("records")
        for subject in cohort_subjects:
            if subject == "CLIN_SUBJ_002" and not include_clin002:
                continue
            sub = df_cached[df_cached.subject == subject].dropna(
                subset=["channel", "eds"],
            )
            if sub.empty:
                continue
            sessions_in_csv = list(sub["session"].drop_duplicates())
            session_eds: list[tuple[str, np.ndarray, list[str]]] = []
            for sess in sessions_in_csv:
                s = sub[sub.session == sess]
                if s.empty:
                    continue
                ch_list = list(s["channel"])
                eds_vec = s["eds"].to_numpy(dtype=float)
                session_eds.append((sess, eds_vec, ch_list))
            last_two = session_eds[-2:]
            if not last_two:
                continue
            common = list(last_two[0][2])
            for _, _, ch_list in last_two[1:]:
                common = [c for c in common if c in ch_list]
            if not common:
                continue
            vecs = []
            for _, eds_vec, ch_list in last_two:
                idx = [ch_list.index(c) for c in common]
                vecs.append(eds_vec[idx])
            subj_eds = np.mean(np.stack(vecs, axis=0), axis=0)
            per_subj_eds[subject] = (subj_eds, common)
            print(f"  {subject}: loaded EDS from CSV ({len(common)} ch)")
    else:
        for subject in cohort_subjects:
            if subject == "CLIN_SUBJ_002" and not include_clin002:
                continue
            sessions = enumerate_online_sessions_for_subject(subject)
            print(f"\n=== {subject} ({band_label}, {len(sessions)} sessions) ===")
            session_eds = []
            for sess in sessions:
                t_s = time.time()
                eds_vec, channels, n_trials = per_session_eds(
                    subject, sess, band, motor_channels, band_label,
                    blink_removal=blink_removal,
                    spatial_filter=spatial_filter,
                )
                dt = time.time() - t_s
                if eds_vec is None or not channels:
                    print(
                        f"  {sess}: SKIP (no EDS), n_trials={n_trials} "
                        f"({dt:.1f}s)"
                    )
                    rows.append({
                        "subject": subject, "session": sess, "band": band_label,
                        "channels": ",".join(channels), "n_trials": n_trials,
                        "eds": None,
                    })
                    continue
                print(
                    f"  {sess}: EDS over {len(channels)} ch, "
                    f"n_trials={n_trials} ({dt:.1f}s)"
                )
                session_eds.append((sess, eds_vec, channels))
                for ch, v in zip(channels, eds_vec):
                    rows.append({
                        "subject": subject, "session": sess, "band": band_label,
                        "channel": ch, "eds": float(v), "n_trials": n_trials,
                    })
            # Per-subject reduction: average over LAST TWO sessions w/ EDS
            last_two = session_eds[-2:]
            if not last_two:
                print(f"  ! no usable last-two-session EDS for {subject}")
                continue
            # Intersect channels across the last-two sessions
            common = list(last_two[0][2])
            for _, _, ch_list in last_two[1:]:
                common = [c for c in common if c in ch_list]
            if not common:
                print(f"  ! {subject}: empty channel intersection across last-two")
                continue
            vecs = []
            for _, eds_vec, ch_list in last_two:
                idx = [ch_list.index(c) for c in common]
                vecs.append(eds_vec[idx])
            subj_eds = np.mean(np.stack(vecs, axis=0), axis=0)
            per_subj_eds[subject] = (subj_eds, common)

    # 3. Cohort grand average. Need a unified channel set across the
    # cohort (intersection of per-subject channel lists).
    if not per_subj_eds:
        print(f"  ! no per-subject EDS for {band_label} cohort; skip plots")
        return
    cohort_common = list(next(iter(per_subj_eds.values()))[1])
    for _, ch_list in per_subj_eds.values():
        cohort_common = [c for c in cohort_common if c in ch_list]
    if not cohort_common:
        print(f"  ! {band_label} cohort channel intersection empty; skip")
        return
    # Stack per-subject EDS vectors aligned to cohort_common
    per_subj_aligned: list[np.ndarray] = []
    per_subj_subject_order: list[str] = []
    for subj, (eds_vec, ch_list) in sorted(per_subj_eds.items()):
        idx = [ch_list.index(c) for c in cohort_common]
        per_subj_aligned.append(eds_vec[idx])
        per_subj_subject_order.append(subj)
    cohort_stack = np.stack(per_subj_aligned, axis=0)  # (n_subj, n_ch)
    cohort_eds = cohort_stack.mean(axis=0)

    # Align the expert EDS to the cohort channel set (intersection)
    expert_aligned_channels = [
        c for c in expert_channels if c in cohort_common
    ]
    common_for_diff = [
        c for c in cohort_common if c in expert_aligned_channels
    ]
    expert_idx = [expert_channels.index(c) for c in common_for_diff]
    cohort_idx = [cohort_common.index(c) for c in common_for_diff]
    expert_eds_aligned = expert_eds[expert_idx]
    cohort_eds_aligned = cohort_eds[cohort_idx]

    # ----- C4: Per-electrode Wilcoxon + Bonferroni (rev01-eds §5) -----
    # Important n-vs-Wilcoxon caveat: with n_subj = 6 (or 7), the
    # minimum two-sided Wilcoxon p-value is 2^-(n-1) = 1/32 = 0.03125
    # (n=6) or 1/64 (n=7). Bonferroni α' = 0.05/15 = 0.0033 is therefore
    # unattainable at n_subj ≤ 6 regardless of effect size. We report
    # both raw and Bonferroni-flagged channels in the CSV; topomap
    # masks use raw p<0.05 (matches Kumar 2024 §5.4 "Fig 5-style
    # z-thresholding without correction" recommendation; Bonferroni
    # column in CSV is for the conservative reader).
    n_subj_used = cohort_stack.shape[0]
    W_zero, p_zero = _wilcoxon_per_channel(cohort_stack)
    n_ch_cohort = len(cohort_common)
    alpha_bonf_cohort = 0.05 / n_ch_cohort
    # Bonferroni (CSV column, may always be False at n=6/7)
    sig_zero_mask_bonf = np.array(
        [(p is not None and np.isfinite(p) and p < alpha_bonf_cohort)
         for p in p_zero], dtype=bool,
    )
    # Uncorrected (topomap mask)
    sig_zero_mask = np.array(
        [(p is not None and np.isfinite(p) and p < 0.05)
         for p in p_zero], dtype=bool,
    )
    # Test 2 (§5.2): cohort − expert ≠ 0 per channel (paired across subj
    # vs expert scalar broadcast). Channel set = common_for_diff.
    cohort_stack_diff = cohort_stack[
        :, [cohort_common.index(c) for c in common_for_diff]
    ]
    W_diff, p_diff = _wilcoxon_paired_per_channel(
        cohort_stack_diff, expert_eds_aligned,
    )
    n_ch_diff = len(common_for_diff)
    alpha_bonf_diff = 0.05 / n_ch_diff
    sig_diff_mask_bonf = np.array(
        [(p is not None and np.isfinite(p) and p < alpha_bonf_diff)
         for p in p_diff], dtype=bool,
    )
    sig_diff_mask = np.array(
        [(p is not None and np.isfinite(p) and p < 0.05)
         for p in p_diff], dtype=bool,
    )

    # ----- M1: per-electrode raw-units mean + cohort consistency -----
    # Raw-units per-subject (unit: EDS = ΔP) and cohort mean.
    raw_cohort_mean = cohort_eds.copy()
    raw_cohort_std = cohort_stack.std(axis=0, ddof=1)
    # Per-subject ↔ cohort rank correlation (Spearman ρ of each
    # per-subject vector with the cohort mean, over electrodes).
    cohort_rank = rankdata(raw_cohort_mean)
    consistency_rows = []
    for subj, vec in zip(per_subj_subject_order, per_subj_aligned):
        rho, p_rho = spearmanr(rankdata(vec), cohort_rank)
        consistency_rows.append({
            "band": band_label, "subject": subj,
            "spearman_rho_vs_cohort_mean": float(rho),
            "spearman_p": float(p_rho),
            "n_channels": int(len(cohort_common)),
        })
    pd.DataFrame(consistency_rows).to_csv(
        out_dir / f"eds_cohort_consistency_{band_label}{variant_tag}.csv",
        index=False,
    )

    # 4. z-score across electrodes
    def _zscore(v):
        sd = v.std(ddof=1)
        if sd == 0:
            return np.zeros_like(v)
        return (v - v.mean()) / sd

    z_expert = _zscore(expert_eds)
    z_cohort = _zscore(cohort_eds)
    z_expert_a = _zscore(expert_eds_aligned)
    z_cohort_a = _zscore(cohort_eds_aligned)
    z_diff = z_cohort_a - z_expert_a

    # 5. Plot — significance marks (Bonferroni-adjusted) drawn via mask.
    variant_in_title = (
        f"  [channel-set={channel_set}, blink={blink_removal}, "
        f"spatial={spatial_filter}]" if variant_tag else ""
    )
    _plot_topomap_panel(
        z_expert, expert_channels,
        (f"Expert EDS (shared OG_Right pool, n_trials={expert_n}) — "
         f"{band_label}{variant_in_title}"),
        str(out_dir / f"expert_eds_topoplot_{band_label}{variant_tag}.png"),
    )
    _plot_topomap_panel(
        z_cohort, cohort_common,
        (f"CLIN cohort EDS grand average — {band_label}{variant_in_title}\n"
         f"(n={n_subj_used} subj, last 2 sessions; o = Wilcoxon vs 0 p<0.05 "
         f"uncorrected; {int(sig_zero_mask_bonf.sum())}/{n_ch_cohort} "
         f"pass Bonf α'={alpha_bonf_cohort:.4f}; min two-sided "
         f"Wilcoxon p at n={n_subj_used} is {2**-(n_subj_used-1):.4f})"),
        str(out_dir / f"cohort_eds_topoplot_{band_label}{variant_tag}.png"),
        mask=sig_zero_mask,
    )
    if not no_diff_plot:
        _plot_topomap_panel(
            z_diff, common_for_diff,
            (f"Cohort − Expert EDS — {band_label}{variant_in_title}\n"
             f"(o = paired Wilcoxon p<0.05 uncorrected; "
             f"{int(sig_diff_mask_bonf.sum())}/{n_ch_diff} pass Bonf "
             f"α'={alpha_bonf_diff:.4f})"),
            str(
                out_dir
                / f"cohort_minus_expert_eds_topoplot_{band_label}{variant_tag}.png"
            ),
            cmap="RdBu_r",
            mask=sig_diff_mask,
        )
    _plot_per_subject_grid(
        per_subj_eds,
        (f"Per-subject EDS (last 2 ONLINE sessions averaged) — "
         f"{band_label}{variant_in_title}"),
        str(
            out_dir
            / f"per_subject_eds_topoplot_{band_label}{variant_tag}_grid.png"
        ),
    )

    # 6. CSV (long form: original rows + significance + raw-units summary)
    df = pd.DataFrame(rows)
    df.to_csv(
        out_dir / f"eds_per_subject_session_{band_label}{variant_tag}.csv",
        index=False,
    )
    print(
        f"\n  wrote: eds_per_subject_session_{band_label}{variant_tag}.csv "
        f"({len(df)} rows)"
    )

    # Cohort summary CSV (M1 raw-units + C4 significance):
    cohort_summary_rows = []
    for i, ch in enumerate(cohort_common):
        # cohort vs zero
        p0 = p_zero[i]
        is_sig_zero = bool(sig_zero_mask[i])
        is_sig_zero_bonf = bool(sig_zero_mask_bonf[i])
        # cohort vs expert (only meaningful if ch in common_for_diff)
        if ch in common_for_diff:
            j = common_for_diff.index(ch)
            p_d = p_diff[j]
            is_sig_diff = bool(sig_diff_mask[j])
            is_sig_diff_bonf = bool(sig_diff_mask_bonf[j])
            expert_raw = float(expert_eds_aligned[j])
        else:
            p_d = np.nan
            is_sig_diff = False
            is_sig_diff_bonf = False
            expert_raw = np.nan
        cohort_summary_rows.append({
            "band": band_label, "channel": ch,
            "cohort_eds_mean_raw": float(raw_cohort_mean[i]),
            "cohort_eds_std_raw": float(raw_cohort_std[i]),
            "cohort_z_score": float(z_cohort[i]),
            "wilcoxon_vs0_p": float(p0) if np.isfinite(p0) else np.nan,
            "wilcoxon_vs0_sig_uncorr": is_sig_zero,
            "wilcoxon_vs0_sig_bonf": is_sig_zero_bonf,
            "wilcoxon_vs_expert_p": float(p_d) if np.isfinite(p_d) else np.nan,
            "wilcoxon_vs_expert_sig_uncorr": is_sig_diff,
            "wilcoxon_vs_expert_sig_bonf": is_sig_diff_bonf,
            "expert_eds_raw": expert_raw,
            "n_subjects": int(cohort_stack.shape[0]),
            "bonf_alpha_cohort": float(alpha_bonf_cohort),
            "bonf_alpha_diff": float(alpha_bonf_diff),
        })
    pd.DataFrame(cohort_summary_rows).to_csv(
        out_dir / f"eds_cohort_summary_{band_label}{variant_tag}.csv",
        index=False,
    )
    print(
        f"  wrote: eds_cohort_summary_{band_label}{variant_tag}.csv "
        f"({n_ch_cohort} channels; "
        f"uncorr: {sig_zero_mask.sum()} pass vs 0, "
        f"{sig_diff_mask.sum()} pass vs expert; "
        f"Bonf: {sig_zero_mask_bonf.sum()} vs 0, "
        f"{sig_diff_mask_bonf.sum()} vs expert)"
    )


_DEFAULT_CHANNEL_SET = "motor15"
_DEFAULT_BLINK = "drop_fp"
_DEFAULT_SPATIAL = "car"


def _build_variant_tag(channel_set: str, blink: str, spatial: str) -> str:
    """Return a filename suffix encoding any non-default variant choices.

    Default values (motor15, drop_fp, car) yield an empty string so
    re-running with defaults overwrites the existing pass-1 outputs.
    Any non-default selection yields a tag like `_full22_fpreg_csd`
    so different variants coexist in the same output dir.
    """
    parts: list[str] = []
    if channel_set != _DEFAULT_CHANNEL_SET:
        parts.append(channel_set)
    if blink != _DEFAULT_BLINK:
        parts.append("fpreg" if blink == "fp_regression" else blink)
    if spatial != _DEFAULT_SPATIAL:
        parts.append(spatial)
    return ("_" + "_".join(parts)) if parts else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-beta", action="store_true",
        help="Also compute beta-band EDS (supplementary).",
    )
    parser.add_argument(
        "--include-clin002", action="store_true",
        help="Include CLIN_SUBJ_002 (13-ch / LedoitWolf older config).",
    )
    parser.add_argument(
        "--from-csv", action="store_true",
        help=("Reload per-session EDS values from "
              "eds_per_subject_session_<band><variant>.csv (skips the "
              "~22 min preprocessing pass; expert EDS is still "
              "recomputed)."),
    )
    parser.add_argument(
        "--channel-set", choices=("motor15", "full22"),
        default=_DEFAULT_CHANNEL_SET,
        help=("EDS channel pool. `motor15` = deployed-decoder subset "
              "(default). `full22` = Kumar 2024 montage minus Fp "
              "channels (frontal F-row + FC5/FC6 added on top of "
              "motor15). Output filenames are tagged with the variant "
              "so motor15 + full22 results coexist."),
    )
    parser.add_argument(
        "--blink-removal", choices=("drop_fp", "fp_regression"),
        default=_DEFAULT_BLINK,
        help=("Blink-removal method passed to "
              "`sweep_phase2_round2.apply_blink_removal`. `drop_fp` "
              "(default) drops Fp1/Fp2/Fpz; `fp_regression` regresses "
              "them out of the other channels while keeping them in "
              "the data."),
    )
    parser.add_argument(
        "--spatial-filter", choices=("car", "csd", "hjorth"),
        default=_DEFAULT_SPATIAL,
        help=("Spatial filter passed to "
              "`sweep_phase2_round2.apply_spatial_filter`. `car` is "
              "Config A default; `csd` is MNE current-source-density; "
              "`hjorth` is k=4 nearest-neighbour Laplacian."),
    )
    parser.add_argument(
        "--no-diff-plot", action="store_true",
        help=("Skip the cohort_minus_expert topomap (harder to "
              "interpret; the separate expert + cohort plots cover "
              "the same content)."),
    )
    parser.add_argument(
        "--subjects", default="",
        help=("Comma-separated subject filter for smoke tests, e.g. "
              "`CLIN_SUBJ_005`. Empty = full cohort."),
    )
    args = parser.parse_args()
    subject_filter = {
        s.strip() for s in args.subjects.split(",") if s.strip()
    }

    out_dir = clin_pictures_root() / "eds"
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort = list(CLIN_PRIMARY_SUBJECTS)
    if args.include_clin002:
        cohort = ["CLIN_SUBJ_002"] + cohort
    if subject_filter:
        cohort = [s for s in cohort if s in subject_filter]
        if not cohort:
            print(
                f"  ! --subjects filter {sorted(subject_filter)} "
                "matches no cohort member; nothing to do."
            )
            return

    bands = [("mu", (MU_LO, MU_HI))]
    if args.include_beta:
        bands.append(("beta", (BETA_LO, BETA_HI)))

    variant_tag = _build_variant_tag(
        args.channel_set, args.blink_removal, args.spatial_filter,
    )
    # Safety: a subject filter on its own would otherwise overwrite the
    # full-cohort PNG/CSV with sub-cohort outputs (cohort PNG becomes
    # n=1, Wilcoxon NaN, etc.). Tag every --subjects run so smoke tests
    # land in dedicated filenames.
    if subject_filter:
        subjects_tag = "_subj-" + "-".join(
            s.replace("CLIN_SUBJ_", "") for s in sorted(subject_filter)
        )
        variant_tag = variant_tag + subjects_tag
    if variant_tag:
        print(
            f"[variant] channel_set={args.channel_set} "
            f"blink_removal={args.blink_removal} "
            f"spatial_filter={args.spatial_filter}"
            + (f" subjects={sorted(subject_filter)}" if subject_filter else "")
            + f" → filename tag = '{variant_tag}'"
        )

    for band_label, band in bands:
        run_for_band(
            band_label, band, cohort, out_dir,
            include_clin002=args.include_clin002,
            from_csv=args.from_csv,
            channel_set=args.channel_set,
            blink_removal=args.blink_removal,
            spatial_filter=args.spatial_filter,
            no_diff_plot=args.no_diff_plot,
            variant_tag=variant_tag,
        )

    print(f"\nDone. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
