"""
Feature extraction for XGBoost branches based on the canonical Riemannian pipeline.

This module provides:
- Optional surface Laplacian (CSD) preprocessing (offline; linear spatial transform).
- ERD-style, online-compatible features computed from baseline-relative bandpower
  using only the current analysis window + a causal baseline window.
- A shared segmentation routine that matches `Generate_Riemannian_adaptive.py`
  windowing/labeling logic.
"""

import os

# Force-disable Numba caching to avoid MNE import failures in constrained environments.
# (Some environments set this key but to a non-effective value, so we don't use setdefault.)
os.environ["NUMBA_DISABLE_CACHING"] = "1"
os.environ["MNE_USE_NUMBA"] = "false"

import sys
from typing import Optional

import numpy as np
import mne

import config
from Utils.stream_utils import get_channel_names_from_xdf
from Utils.preprocessing import (
    get_valid_channel_mask_and_metadata,
    select_channels,
    initialize_filter_bank,
    apply_streaming_filters,
)
from Utils.artifact_rejection import apply_training_artifact_rejection
from Utils.marker_pairing import build_trial_windows_chronological


# Mirror canonical trigger logic
TRIGGERS = config.TRIGGERS
EPOCHS_START_END = {
    TRIGGERS["REST_BEGIN"]: TRIGGERS["REST_END"],
    TRIGGERS["MI_BEGIN"]: TRIGGERS["MI_END"],
}


def apply_surface_laplacian_csd(
    data: np.ndarray,
    ch_names: list[str],
    fs: float,
    montage_name: str = "standard_1020",
    on_missing: str = "warn",
) -> np.ndarray:
    """
    Apply surface Laplacian via MNE current source density (CSD).

    Args:
        data: shape (n_channels, n_times)
        ch_names: channel names matching the montage
        fs: sampling frequency
        montage_name: MNE montage to use
        on_missing: how to handle missing channels in montage
    Returns:
        CSD-transformed data, same shape as input.
    """
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage, match_case=False, on_missing=on_missing)

    csd_raw = mne.preprocessing.compute_current_source_density(raw, verbose="ERROR")
    return csd_raw.get_data()


def _get_beta_high() -> float:
    """
    Upper frequency for the ERD beta band.

    Default is 30Hz to support a 13-30 "beta richness" feature set.
    """
    return float(getattr(config, "XGB_ERD_BETA_HIGH", 30.0))


def _band_edges_default() -> list[tuple[float, float]]:
    """
    Default ERD feature sets to compute.

    Default is mu-only unless users explicitly configure additional bands.
    """
    mu_lo, mu_hi = float(config.LOWCUT), float(config.HIGHCUT)
    return [(mu_lo, mu_hi)]


def _compute_bandpower_fft_features(
    signal: np.ndarray,
    fs: float,
    bands: list[tuple[float, float]],
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute per-channel bandpower via FFT magnitude squared.

    Args:
        signal: shape (n_channels, n_time)
        fs: sampling frequency
        bands: list of (low, high) Hz
    Returns:
        bandpower: shape (n_channels, n_bands)
    """
    n_ch, n_t = signal.shape
    freqs = np.fft.rfftfreq(n_t, d=1.0 / fs)
    fft = np.fft.rfft(signal, axis=1)
    power = (np.abs(fft) ** 2) / float(n_t)  # (n_ch, n_freq)

    out = np.empty((n_ch, len(bands)), dtype=float)
    for b_idx, (lo, hi) in enumerate(bands):
        mask = (freqs >= lo) & (freqs < hi)
        if np.any(mask):
            out[:, b_idx] = power[:, mask].mean(axis=1)
        else:
            out[:, b_idx] = eps

    return np.log10(out + eps)


def _get_bands_from_config() -> list[tuple[float, float]]:
    bands = getattr(config, "XGB_ERD_BANDS", None)
    if bands is None:
        return _band_edges_default()
    return bands


def segment_and_extract_cov_erd(
    eeg_stream,
    marker_stream,
    compute_erd: bool = True,
    apply_csd: bool = False,
    baseline_duration_sec: Optional[float] = None,
    return_beta_segments: bool = False,
):
    """
    Canonical segmentation + optional surface Laplacian + ERD features.

    This matches the windowing/labels used in `Generate_Riemannian_adaptive.py`.

    The **last** return value is always ``channel_names`` (after validity + optional motor subset).

    Return tuple:
        - ``return_beta_segments=False``: ``(segments, labels, erd_or_None, channel_names)``
        - ``return_beta_segments=True``: ``(segments, labels, erd_or_None, beta_segments, channel_names)``
    """
    marker_values = np.array([int(m[0]) for m in marker_stream["time_series"]])
    marker_timestamps = np.array([float(m[1]) for m in marker_stream["time_series"]])
    eeg_timestamps = np.array(eeg_stream["time_stamps"])
    eeg_data = np.array(eeg_stream["time_series"]).T  # (n_channels, n_samples)

    # Channel selection logic (must match canonical script)
    all_channel_names = get_channel_names_from_xdf(eeg_stream)
    valid_channel_names, valid_raw, valid_indices = get_valid_channel_mask_and_metadata(
        eeg_data, all_channel_names, fs=config.FS, drop_mastoids=True
    )
    eeg_data = eeg_data[valid_indices, :]
    current_channel_names = list(valid_channel_names)

    if config.SELECT_MOTOR_CHANNELS:
        sel_raw = select_channels(valid_raw, keep_channels=config.MOTOR_CHANNEL_NAMES)
        subset_indices_valid = [current_channel_names.index(ch) for ch in sel_raw.ch_names]
        eeg_data = eeg_data[subset_indices_valid, :]
        current_channel_names = list(sel_raw.ch_names)

    # === Filter setup ===
    # Canonical covariance features come from the mu band (LOWCUT..HIGHCUT).
    # ERD features can be expanded with a beta band (HIGHCUT..beta_high).
    filter_bank_mu = initialize_filter_bank(
        fs=config.FS,
        lowcut=float(config.LOWCUT),
        highcut=float(config.HIGHCUT),
        notch_freqs=[60],
        notch_q=30,
    )
    filter_bank_beta = initialize_filter_bank(
        fs=config.FS,
        lowcut=float(config.HIGHCUT),
        highcut=_get_beta_high(),
        notch_freqs=[60],
        notch_q=30,
    )

    window_size = config.CLASSIFY_WINDOW / 1000.0
    step_size = 1 / 16
    window_samples = int(window_size * config.FS)
    step_samples = int(step_size * config.FS)
    chunk_samples = int(window_size * config.FS)

    segments_all = []
    labels_all = []
    erd_feats_all = [] if compute_erd else None
    beta_segments_all = [] if return_beta_segments else None

    epochs_int = {int(a): int(b) for a, b in EPOCHS_START_END.items()}
    max_ep = float(config.MAX_EPOCH_MARKER_DURATION_SEC)
    trial_windows, mp_stats = build_trial_windows_chronological(
        marker_timestamps,
        marker_values,
        epochs_int,
        max_duration_sec=max_ep,
        min_duration_sec=1.0,
    )
    if mp_stats.get("skipped_long_duration", 0) > 0:
        print(
            f"[marker_pairing] dropped {mp_stats['skipped_long_duration']} epoch(s) "
            f"longer than {max_ep:g}s (mis-paired or duplicate markers?)",
            file=sys.stderr,
        )
    if not trial_windows:
        raise ValueError(
            "no_valid_trials: no REST/MI begin→end pairs produced epochs with duration > 1s "
            "(check markers 100/120 and 200/220 and XDF marker stream)."
        )

    filter_warmup = 1.0
    trial_bounds = [(start - 1.0, end) for (start, end, _) in trial_windows]
    valid_start = trial_bounds[0][0] - filter_warmup
    valid_end = trial_bounds[-1][1]

    global_start = np.searchsorted(eeg_timestamps, valid_start)
    global_end = np.searchsorted(eeg_timestamps, valid_end)
    raw_global = eeg_data[:, global_start:global_end]
    rel_timestamps = eeg_timestamps[global_start:global_end]

    # === Stream through global segment with filter continuity ===
    filter_state_mu = {}
    filter_state_beta = {}
    filtered_global_mu = np.zeros_like(raw_global)
    filtered_global_beta = np.zeros_like(raw_global)
    for chunk_start in range(0, raw_global.shape[1], chunk_samples):
        chunk_end = min(chunk_start + chunk_samples, raw_global.shape[1])
        chunk = raw_global[:, chunk_start:chunk_end]
        filtered_chunk_mu, filter_state_mu = apply_streaming_filters(
            chunk, filter_bank_mu, filter_state_mu
        )
        filtered_chunk_beta, filter_state_beta = apply_streaming_filters(
            chunk, filter_bank_beta, filter_state_beta
        )
        filtered_global_mu[:, chunk_start:chunk_end] = filtered_chunk_mu
        filtered_global_beta[:, chunk_start:chunk_end] = filtered_chunk_beta

    # === Optional surface Laplacian (ERD-only) ===
    # Per user requirement:
    # - covariance features should be computed from non-CSD signals
    # - ERD features should use CSD when enabled
    if apply_csd:
        filtered_global_mu_erd = apply_surface_laplacian_csd(
            filtered_global_mu, current_channel_names, fs=config.FS
        )
        filtered_global_beta_erd = apply_surface_laplacian_csd(
            filtered_global_beta, current_channel_names, fs=config.FS
        )
    else:
        filtered_global_mu_erd = filtered_global_mu
        filtered_global_beta_erd = filtered_global_beta

    erd_bands = _get_bands_from_config()
    mu_lo, mu_hi = float(config.LOWCUT), float(config.HIGHCUT)
    beta_lo, beta_hi = mu_hi, _get_beta_high()

    # Split bands by which filtered stream they belong to.
    mu_bands = []
    beta_bands = []
    for (lo, hi) in erd_bands:
        lo, hi = float(lo), float(hi)
        if lo >= mu_lo and hi <= mu_hi:
            mu_bands.append((lo, hi))
        elif lo >= beta_lo and hi <= beta_hi:
            beta_bands.append((lo, hi))
        else:
            raise ValueError(
                f"ERD band [{lo}, {hi}] not supported by this extractor. "
                f"Bands must be fully within mu=[{mu_lo}, {mu_hi}] or beta=[{beta_lo}, {beta_hi}]."
            )

    mu_bands = list(dict.fromkeys(mu_bands))  # preserve order, unique
    beta_bands = list(dict.fromkeys(beta_bands))
    mu_index = {b: i for i, b in enumerate(mu_bands)}
    beta_index = {b: i for i, b in enumerate(beta_bands)}
    if baseline_duration_sec is None:
        baseline_duration_sec = float(getattr(config, "BASELINE_DURATION", 1.0))

    # For each trial, extract and label segments
    for (ts_start, ts_end, label) in trial_windows:
        rel_end = np.searchsorted(rel_timestamps, ts_end)
        baseline_end = np.searchsorted(rel_timestamps, ts_start)
        decision_start = np.searchsorted(rel_timestamps, ts_start + 1.0)

        if rel_end <= decision_start or baseline_end <= 0:
            continue

        # Covariance branch (always non-CSD)
        baseline_mu_cov = filtered_global_mu[:, :baseline_end].mean(axis=1, keepdims=True)
        baseline_beta_cov = filtered_global_beta[:, :baseline_end].mean(axis=1, keepdims=True)
        trial_data_mu_cov = filtered_global_mu[:, decision_start:rel_end] - baseline_mu_cov
        trial_data_beta_cov = filtered_global_beta[:, decision_start:rel_end] - baseline_beta_cov

        # ERD branch (CSD-applied when toggle is ON)
        baseline_mu_erd = filtered_global_mu_erd[:, :baseline_end].mean(axis=1, keepdims=True)
        baseline_beta_erd = filtered_global_beta_erd[:, :baseline_end].mean(axis=1, keepdims=True)
        trial_data_mu_erd = filtered_global_mu_erd[:, decision_start:rel_end] - baseline_mu_erd
        trial_data_beta_erd = filtered_global_beta_erd[:, decision_start:rel_end] - baseline_beta_erd

        # `segments_all` must remain compatible with canonical covariance estimation.
        # So it is always derived from mu-filtered data.
        trial_data = trial_data_mu_cov
        n_samples = trial_data.shape[1]

        if n_samples < window_samples:
            continue

        # baseline power window for ERD (causal: immediately before decision_start)
        if compute_erd:
            bp_start = np.searchsorted(rel_timestamps, ts_start - baseline_duration_sec)
            bp_start = max(0, bp_start)
            baseline_segment_mu = (
                filtered_global_mu_erd[:, bp_start:baseline_end] - baseline_mu_erd
            )
            baseline_segment_beta = (
                filtered_global_beta_erd[:, bp_start:baseline_end] - baseline_beta_erd
            )

            base_bp_mu = (
                _compute_bandpower_fft_features(baseline_segment_mu, fs=config.FS, bands=mu_bands)
                if len(mu_bands)
                else np.zeros((filtered_global_mu.shape[0], 0), dtype=float)
            )
            base_bp_beta = (
                _compute_bandpower_fft_features(baseline_segment_beta, fs=config.FS, bands=beta_bands)
                if len(beta_bands)
                else np.zeros((filtered_global_mu.shape[0], 0), dtype=float)
            )

            # Stack base bandpowers in the exact order requested by `erd_bands`.
            base_bp_parts = []
            for band in erd_bands:
                band = (float(band[0]), float(band[1]))
                if band in mu_index:
                    base_bp_parts.append(base_bp_mu[:, mu_index[band]].reshape(-1, 1))
                else:
                    base_bp_parts.append(base_bp_beta[:, beta_index[band]].reshape(-1, 1))
            base_bp = np.hstack(base_bp_parts)  # (n_ch, n_erd_bands)

        for i in range(0, n_samples - window_samples + 1, step_samples):
            segment_mu_cov = trial_data_mu_cov[:, i:i + window_samples]
            segment_beta_cov = trial_data_beta_cov[:, i:i + window_samples]
            segment_mu_erd = trial_data_mu_erd[:, i:i + window_samples]
            segment_beta_erd = trial_data_beta_erd[:, i:i + window_samples]

            segments_all.append(segment_mu_cov)
            if return_beta_segments:
                beta_segments_all.append(segment_beta_cov)
            labels_all.append(label)

            if compute_erd:
                win_bp_mu = (
                    _compute_bandpower_fft_features(segment_mu_erd, fs=config.FS, bands=mu_bands)
                    if len(mu_bands)
                    else np.zeros((segment_mu_erd.shape[0], 0), dtype=float)
                )
                win_bp_beta = (
                    _compute_bandpower_fft_features(segment_beta_erd, fs=config.FS, bands=beta_bands)
                    if len(beta_bands)
                    else np.zeros((segment_mu_erd.shape[0], 0), dtype=float)
                )

                win_bp_parts = []
                for band in erd_bands:
                    band = (float(band[0]), float(band[1]))
                    if band in mu_index:
                        win_bp_parts.append(win_bp_mu[:, mu_index[band]].reshape(-1, 1))
                    else:
                        win_bp_parts.append(win_bp_beta[:, beta_index[band]].reshape(-1, 1))
                win_bp = np.hstack(win_bp_parts)  # (n_ch, n_erd_bands)

                # ERD in log space: log(Pwin/Pbase) = logPwin - logPbase
                erd = win_bp - base_bp
                erd_feats_all.append(erd.reshape(-1))

    segments_arr = np.array(segments_all)
    labels_arr = np.array(labels_all)
    erd_arr = np.array(erd_feats_all) if compute_erd else None
    beta_arr = np.array(beta_segments_all) if return_beta_segments else None

    n_seg = int(segments_arr.shape[0])
    if n_seg != int(labels_arr.shape[0]):
        raise RuntimeError(
            f"internal ordering bug: segments {n_seg} vs labels {int(labels_arr.shape[0])}"
        )
    if erd_arr is not None and int(erd_arr.shape[0]) != n_seg:
        raise RuntimeError(
            f"internal ordering bug: erd_feats {int(erd_arr.shape[0])} vs segments {n_seg}"
        )
    if beta_arr is not None and int(beta_arr.shape[0]) != n_seg:
        raise RuntimeError(
            f"internal ordering bug: beta_segments {int(beta_arr.shape[0])} vs segments {n_seg}"
        )

    segments_arr, labels_arr, extras = apply_training_artifact_rejection(
        segments_arr, labels_arr, erd_arr, beta_arr
    )
    erd_arr, beta_arr = extras[0], extras[1]

    n_kept = int(segments_arr.shape[0])
    if n_kept != int(labels_arr.shape[0]) or (
        erd_arr is not None and int(erd_arr.shape[0]) != n_kept
    ) or (beta_arr is not None and int(beta_arr.shape[0]) != n_kept):
        raise RuntimeError("artifact rejection row alignment broken")

    ch_out = list(current_channel_names)

    if return_beta_segments:
        return segments_arr, labels_arr, erd_arr, beta_arr, ch_out

    return segments_arr, labels_arr, erd_arr, ch_out

