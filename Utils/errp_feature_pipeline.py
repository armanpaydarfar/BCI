"""
ErrP (Error-Related Potential) feature extraction pipeline.

This module handles epoch segmentation and data loading for ErrP BCI decoding.

ErrP signal properties:
- ERN (Error-Related Negativity): frontocentral negativity, ~80-150 ms post-error
- Pe  (Error Positivity):         frontocentral positivity, ~200-400 ms post-error
- Epoch window: 0-800 ms post-event marker (captures both components at 512 Hz)
- Bandpass: 1-10 Hz  (DC removal + ERP isolation, removes high-freq muscle noise)
- Channels: frontocentral (F3, Fz, F4, FC1, FC2, Cz — configurable)

Marker conventions:
  Existing data (legacy):
    error   = ROBOT_EARLYSTOP (340) — unexpected robot stop
    correct = ROBOT_END       (320) — normal trajectory completion
  Dedicated ErrP experiment:
    error   = ERRP_STIM_ERROR   (430) — unexpected robot stop
    correct = ERRP_STIM_CORRECT (440) — normal robot completion

Both marker sets are auto-detected from the XDF data.
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Optional

os.environ["NUMBA_DISABLE_CACHING"] = "1"
os.environ["MNE_USE_NUMBA"] = "false"

import numpy as np

import config
from Utils.stream_utils import load_xdf, get_channel_names_from_xdf
from Utils.preprocessing import (
    get_valid_channel_mask_and_metadata,
    select_channels,
    initialize_filter_bank,
    apply_streaming_filters,
    car_rereference,
)

# ---------------------------------------------------------------------------
# Canonical marker codes — derived from config, with graceful fallback
# ---------------------------------------------------------------------------
_TRIGGERS = config.TRIGGERS

def _code(key: str) -> int:
    return int(_TRIGGERS[key])


def _default_error_codes() -> list[int]:
    """Error event codes present in existing + new experiment data."""
    codes = []
    for key in ("ROBOT_EARLYSTOP", "ERRP_STIM_ERROR"):
        if key in _TRIGGERS:
            codes.append(_code(key))
    return codes


def _default_correct_codes() -> list[int]:
    """Correct (no-ErrP) event codes."""
    codes = []
    for key in ("ROBOT_END", "ERRP_STIM_CORRECT"):
        if key in _TRIGGERS:
            codes.append(_code(key))
    return codes


# ---------------------------------------------------------------------------
# Epoch segmentation
# ---------------------------------------------------------------------------

def segment_errp_epochs(
    eeg_data: np.ndarray,
    eeg_timestamps: np.ndarray,
    marker_timestamps: np.ndarray,
    marker_values: np.ndarray,
    error_codes: list[int],
    correct_codes: list[int],
    tmin: float,
    tmax: float,
    fs: float,
    artifact_thresh_uv: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Extract fixed-length ERP epochs time-locked to error / correct event markers.

    Unlike MI segmentation (begin→end span with sliding windows), ErrP epochs
    are fixed-length windows anchored at the event timestamp.

    Args:
        eeg_data:          (n_channels, n_samples) filtered EEG
        eeg_timestamps:    (n_samples,) timestamps aligned to eeg_data
        marker_timestamps: (n_events,)  event timestamps
        marker_values:     (n_events,)  integer marker codes
        error_codes:       list of int  → label 1  (ErrP / error perceived)
        correct_codes:     list of int  → label 0  (no ErrP / correct)
        tmin:              epoch start offset from event (seconds, ≥ 0)
        tmax:              epoch end   offset from event (seconds, > tmin)
        fs:                sampling frequency (Hz)
        artifact_thresh_uv: if set, reject epochs where max|x| > threshold (µV)

    Returns:
        epochs:  np.ndarray (n_kept, n_channels, n_times)
        labels:  np.ndarray (n_kept,) — 1 = error, 0 = correct
        stats:   dict with counts of kept / dropped epochs per category
    """
    epoch_samples = int(round((tmax - tmin) * fs))
    tmin_samples  = int(round(tmin * fs))

    epochs: list[np.ndarray] = []
    labels: list[int] = []

    stats = {
        "n_error_kept":   0,
        "n_correct_kept": 0,
        "n_error_total":  0,
        "n_correct_total":0,
        "n_dropped_bounds":    0,
        "n_dropped_artifact":  0,
    }

    for ts, code in zip(marker_timestamps, marker_values):
        code = int(code)
        if code in error_codes:
            label = 1
            stats["n_error_total"] += 1
        elif code in correct_codes:
            label = 0
            stats["n_correct_total"] += 1
        else:
            continue

        # Find sample index closest to event timestamp
        idx_event = int(np.searchsorted(eeg_timestamps, ts))
        idx_start = idx_event + tmin_samples
        idx_end   = idx_start + epoch_samples

        if idx_start < 0 or idx_end > eeg_data.shape[1]:
            stats["n_dropped_bounds"] += 1
            continue

        epoch = eeg_data[:, idx_start:idx_end]

        if artifact_thresh_uv is not None:
            if np.max(np.abs(epoch)) > artifact_thresh_uv:
                stats["n_dropped_artifact"] += 1
                continue

        epochs.append(epoch)
        labels.append(label)

        if label == 1:
            stats["n_error_kept"] += 1
        else:
            stats["n_correct_kept"] += 1

    if not epochs:
        return np.empty((0, eeg_data.shape[0], epoch_samples)), np.empty(0, dtype=int), stats

    return np.stack(epochs, axis=0), np.array(labels, dtype=int), stats


# ---------------------------------------------------------------------------
# Per-file loading + filtering
# ---------------------------------------------------------------------------

def load_and_preprocess_errp_xdf(
    xdf_path: str,
    error_codes: Optional[list[int]] = None,
    correct_codes: Optional[list[int]] = None,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    artifact_thresh_uv: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """
    Load one XDF file, apply ErrP-band filtering, and extract epochs.

    Returns:
        X:        (n_epochs, n_channels, n_times)
        y:        (n_epochs,) labels — 1=error, 0=correct
        ch_names: list of channel names (post-selection)
        stats:    segmentation stats dict
    """
    if error_codes is None:
        error_codes = _default_error_codes()
    if correct_codes is None:
        correct_codes = _default_correct_codes()
    if tmin is None:
        tmin = float(getattr(config, "ERRP_EPOCH_TMIN", 0.0))
    if tmax is None:
        tmax = float(getattr(config, "ERRP_EPOCH_TMAX", 0.8))
    if artifact_thresh_uv is None:
        artifact_thresh_uv = float(getattr(config, "ERRP_ARTIFACT_MAX_ABS_UV", 80.0))

    eeg_stream, marker_stream = load_xdf(xdf_path)

    # --- EEG arrays ---
    eeg_data = np.array(eeg_stream["time_series"]).T        # (n_ch, n_samp)
    eeg_timestamps = np.array(eeg_stream["time_stamps"])

    # --- Marker arrays ---
    raw_markers = marker_stream["time_series"]
    marker_ts   = np.array(marker_stream["time_stamps"], dtype=float)
    marker_vals = np.array([int(float(m[0])) for m in raw_markers], dtype=int)

    # --- Channel selection ---
    all_ch = get_channel_names_from_xdf(eeg_stream)
    valid_ch_names, valid_raw, valid_indices = get_valid_channel_mask_and_metadata(
        eeg_data, all_ch, fs=config.FS, drop_mastoids=True
    )
    eeg_data = eeg_data[valid_indices, :]
    current_ch = list(valid_ch_names)

    # Select ErrP channels if enabled; fall back to all valid channels
    errp_ch = list(getattr(config, "ERRP_CHANNEL_NAMES", []))
    if errp_ch:
        available = [ch for ch in errp_ch if ch in current_ch]
        missing   = [ch for ch in errp_ch if ch not in current_ch]
        if missing:
            print(f"[ErrP] WARNING: channels not found in recording: {missing}")
        if available:
            indices = [current_ch.index(ch) for ch in available]
            eeg_data = eeg_data[indices, :]
            current_ch = available

    ch_names = current_ch

    # --- Common Average Reference across the ErrP subset ---
    # Applied before the causal bandpass so that the offline path matches the
    # online EEGStreamState exactly (both operate on the rereferenced subset).
    # Gated on config.ERRP_CAR_REREFERENCE for A/B testing; on by default.
    if int(getattr(config, "ERRP_CAR_REREFERENCE", 1)):
        eeg_data = car_rereference(eeg_data)

    # --- Apply ErrP bandpass + notch (causal streaming filter) ---
    lowcut  = float(getattr(config, "LOWCUT_ERRP",  1.0))
    highcut = float(getattr(config, "HIGHCUT_ERRP", 10.0))
    fb = initialize_filter_bank(
        fs=config.FS,
        lowcut=lowcut,
        highcut=highcut,
        notch_freqs=[60],
        notch_q=30,
    )
    chunk_size = 1024
    filter_state: dict = {}
    filtered = np.zeros_like(eeg_data)
    for s in range(0, eeg_data.shape[1], chunk_size):
        e = min(s + chunk_size, eeg_data.shape[1])
        chunk_f, filter_state = apply_streaming_filters(eeg_data[:, s:e], fb, filter_state)
        filtered[:, s:e] = chunk_f

    # --- Segment epochs (extended to include the pre-event baseline window) ---
    # We segment a wider window [baseline_tmin, tmax], use the pre-event slice
    # for per-epoch baseline subtraction, then crop down to the user-requested
    # [tmin, tmax].  Matches the classical ERP baseline used by the Liu offline
    # path and the online causal helper, so all three pipelines see the same
    # signal.  Artifact rejection is applied on the wide window so that
    # baseline-only artifacts also reject the trial.
    baseline_tmin = float(getattr(config, "ERRP_BASELINE_TMIN", -0.200))
    baseline_tmax = float(getattr(config, "ERRP_BASELINE_TMAX",  0.0))
    seg_tmin = min(tmin, baseline_tmin)

    X_wide, y, stats = segment_errp_epochs(
        filtered, eeg_timestamps,
        marker_ts, marker_vals,
        error_codes, correct_codes,
        seg_tmin, tmax, config.FS,
        artifact_thresh_uv=artifact_thresh_uv,
    )

    if X_wide.shape[0] == 0:
        return X_wide, y, ch_names, stats

    # Time axis for the wide segment, then per-epoch baseline subtract + crop.
    n_wide = X_wide.shape[2]
    time_wide = seg_tmin + np.arange(n_wide) / float(config.FS)
    base_mask = (time_wide >= baseline_tmin) & (time_wide <= baseline_tmax)
    if base_mask.any():
        X_wide = X_wide - X_wide[:, :, base_mask].mean(axis=2, keepdims=True)

    crop_mask = (time_wide >= tmin) & (time_wide < tmin + (tmax - tmin))
    X = X_wide[:, :, crop_mask]

    return X, y, ch_names, stats


# ---------------------------------------------------------------------------
# Multi-file loader
# ---------------------------------------------------------------------------

def load_errp_training_data(
    xdf_paths: list[str],
    error_codes: Optional[list[int]] = None,
    correct_codes: Optional[list[int]] = None,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    artifact_thresh_uv: Optional[float] = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load and concatenate ErrP epochs from multiple XDF files.

    Auto-detects which error/correct marker codes are present in each file.

    Returns:
        X:        (n_total_epochs, n_channels, n_times)
        y:        (n_total_epochs,) labels — 1=error, 0=correct
        ch_names: list of channel names (from first file)
    """
    if error_codes is None:
        error_codes = _default_error_codes()
    if correct_codes is None:
        correct_codes = _default_correct_codes()

    all_X, all_y = [], []
    ref_ch = None
    total_error = 0
    total_correct = 0

    for path in xdf_paths:
        try:
            X, y, ch_names, stats = load_and_preprocess_errp_xdf(
                path, error_codes, correct_codes, tmin, tmax, artifact_thresh_uv
            )
        except Exception as exc:
            print(f"[ErrP] WARNING: skipping {os.path.basename(path)}: {exc}", file=sys.stderr)
            continue

        if X.shape[0] == 0:
            if verbose:
                print(f"[ErrP] {os.path.basename(path)}: no ErrP epochs found (error codes={error_codes}, correct codes={correct_codes})")
            continue

        if ref_ch is None:
            ref_ch = ch_names
        elif ch_names != ref_ch:
            print(
                f"[ErrP] WARNING: channel mismatch in {os.path.basename(path)} "
                f"(expected {ref_ch}, got {ch_names}) — skipping file.",
                file=sys.stderr,
            )
            continue

        all_X.append(X)
        all_y.append(y)
        total_error   += int(stats["n_error_kept"])
        total_correct += int(stats["n_correct_kept"])

        if verbose:
            print(
                f"[ErrP] {os.path.basename(path)}: "
                f"error={stats['n_error_kept']}/{stats['n_error_total']} kept, "
                f"correct={stats['n_correct_kept']}/{stats['n_correct_total']} kept, "
                f"dropped_bounds={stats['n_dropped_bounds']}, "
                f"dropped_artifact={stats['n_dropped_artifact']}"
            )

    if not all_X:
        raise ValueError(
            f"No ErrP epochs found in any XDF file. "
            f"Looked for error codes {error_codes} and correct codes {correct_codes}. "
            f"Check that the XDF files contain ROBOT_EARLYSTOP (340) or ERRP_STIM_ERROR (430) markers."
        )

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    if verbose:
        print(f"\n[ErrP] Total epochs loaded: {X_all.shape[0]} "
              f"(error={total_error}, correct={total_correct}, "
              f"ratio={total_error/(total_error+total_correct)*100:.1f}% error)")

    return X_all, y_all, ref_ch
