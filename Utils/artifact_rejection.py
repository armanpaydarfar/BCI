"""
Shared artifact rejection for sliding-window training segments.

Training pipelines (`Generate_Riemannian_adaptive`, `segment_and_extract_cov_erd`)
produce windows in native amplitude units from the XDF + filter pipeline. By default
that is **microvolt-scale**, same as `eeg_stream["time_series"]` (see
`Utils.stream_utils.load_xdf`). Thresholds in config are in microvolts unless
`ARTIFACT_SEGMENT_AMPLITUDE_UNIT` is ``\"volts\"`` (then comparisons use the array’s
numeric scale with the documented 1e-6 conversion).

Modes:
  - max_abs: reject if max |x| over channels × time exceeds threshold.
  - peak_to_peak: reject if max over channels of (max(t) - min(t)) exceeds threshold.
  - zscore: reject if |z| of per-window max_abs exceeds SD (computed on this batch).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

import config


def _threshold_in_data_units(threshold_uv: float, *, amplitude_unit: str) -> float:
    u = str(amplitude_unit).lower()
    if u in ("microvolts", "uv", "µv"):
        return float(threshold_uv)
    if u in ("volts", "v"):
        return float(threshold_uv) * 1e-6
    raise ValueError(f"Unknown ARTIFACT_SEGMENT_AMPLITUDE_UNIT: {amplitude_unit!r}")


def keep_mask_max_abs(
    segments: np.ndarray,
    threshold_uv: float,
    *,
    amplitude_unit: Optional[str] = None,
) -> np.ndarray:
    """Keep windows where max |amplitude| <= threshold (threshold in µV by default)."""
    unit = amplitude_unit or getattr(config, "ARTIFACT_SEGMENT_AMPLITUDE_UNIT", "microvolts")
    thr = _threshold_in_data_units(threshold_uv, amplitude_unit=unit)
    max_vals = np.max(np.abs(segments), axis=(1, 2))
    return max_vals <= thr


def keep_mask_peak_to_peak(
    segments: np.ndarray,
    threshold_uv: float,
    *,
    amplitude_unit: Optional[str] = None,
) -> np.ndarray:
    """Keep windows where max over channels of peak-to-peak <= threshold."""
    unit = amplitude_unit or getattr(config, "ARTIFACT_SEGMENT_AMPLITUDE_UNIT", "microvolts")
    thr = _threshold_in_data_units(threshold_uv, amplitude_unit=unit)
    # per channel min/max over time, then max over channels of (max-min)
    ptp = np.max(np.ptp(segments, axis=2), axis=1)
    return ptp <= thr


def keep_mask_zscore_max_abs(segments: np.ndarray, z_sd: float) -> np.ndarray:
    """
    Keep windows where z-score of (max |x| per window) is within ±z_sd.
    If std is 0 or too few samples, keeps all.
    """
    max_vals = np.max(np.abs(segments), axis=(1, 2)).astype(float)
    mu = float(np.mean(max_vals))
    sd = float(np.std(max_vals, ddof=0))
    if sd <= 1e-30 or max_vals.size < 2:
        return np.ones(max_vals.shape[0], dtype=bool)
    z = (max_vals - mu) / sd
    return np.abs(z) <= float(z_sd)


def build_training_keep_mask(segments: np.ndarray) -> np.ndarray:
    """
    Build boolean keep mask from `config` ARTIFACT_* settings.
    If disabled or empty input, returns all-True.
    """
    if segments.size == 0:
        return np.zeros(0, dtype=bool)

    if not bool(getattr(config, "ARTIFACT_REJECT_ENABLE", True)):
        return np.ones(segments.shape[0], dtype=bool)

    mode = str(getattr(config, "ARTIFACT_REJECT_MODE", "max_abs")).lower().strip()
    unit = getattr(config, "ARTIFACT_SEGMENT_AMPLITUDE_UNIT", "microvolts")

    if mode == "max_abs":
        thr = float(getattr(config, "ARTIFACT_MAX_ABS_UV", 30.0))
        return keep_mask_max_abs(segments, thr, amplitude_unit=unit)
    if mode in ("peak_to_peak", "p2p", "ptp"):
        thr = float(getattr(config, "ARTIFACT_P2P_UV", 150.0))
        return keep_mask_peak_to_peak(segments, thr, amplitude_unit=unit)
    if mode == "zscore":
        z_sd = float(getattr(config, "ARTIFACT_ZSCORE_SD", 3.0))
        return keep_mask_zscore_max_abs(segments, z_sd)

    raise ValueError(f"Unknown ARTIFACT_REJECT_MODE: {mode!r}")


def apply_segment_mask(
    mask: np.ndarray,
    segments: np.ndarray,
    labels: np.ndarray,
    *optional_rows: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Tuple[Optional[np.ndarray], ...]]:
    """
    Apply the same row mask to segments, labels, and any optional 2D arrays
    with shape (n_windows, ...).
    """
    if optional_rows:
        out_extras: list[Optional[np.ndarray]] = []
        for arr in optional_rows:
            if arr is None:
                out_extras.append(None)
            else:
                if arr.shape[0] != mask.shape[0]:
                    raise ValueError(
                        f"Row count mismatch: mask {mask.shape[0]} vs array {arr.shape[0]}"
                    )
                out_extras.append(arr[mask])
        return segments[mask], labels[mask], tuple(out_extras)
    return segments[mask], labels[mask], tuple()


def apply_training_artifact_rejection(
    segments: np.ndarray,
    labels: np.ndarray,
    *optional_same_length: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Tuple[Optional[np.ndarray], ...]]:
    """
    Apply configured artifact rejection to parallel training arrays.

    `optional_same_length` may include ``erd_feats`` (n, n_erd) and/or
    ``beta_segments`` (n, ch, t) in any order; pass None for missing.
    """
    mask = build_training_keep_mask(segments)
    n_drop = int(np.sum(~mask))
    verbose = bool(getattr(config, "ARTIFACT_REJECT_VERBOSE", True))
    if n_drop and verbose:
        print(
            f"[artifact_rejection] dropped {n_drop} / {len(mask)} windows "
            f"(mode={getattr(config, 'ARTIFACT_REJECT_MODE', 'max_abs')})"
        )
    if (
        verbose
        and segments.size
        and len(mask) > 0
        and n_drop == len(mask)
    ):
        mx = np.max(np.abs(segments), axis=(1, 2)).astype(float)
        mode = str(getattr(config, "ARTIFACT_REJECT_MODE", "max_abs")).lower().strip()
        unit = getattr(config, "ARTIFACT_SEGMENT_AMPLITUDE_UNIT", "microvolts")
        thr_note = ""
        if mode == "max_abs":
            thr = float(getattr(config, "ARTIFACT_MAX_ABS_UV", 30.0))
            thr_note = f"threshold={thr:g} ({unit}); rule=max|x| over all channels×time per window"
        elif mode in ("peak_to_peak", "p2p", "ptp"):
            thr = float(getattr(config, "ARTIFACT_P2P_UV", 150.0))
            thr_note = f"threshold={thr:g} ({unit}); rule=max over ch of peak-to-peak per window"
        print(
            f"[artifact_rejection] all windows rejected — per-window stats (same scale as segments): "
            f"min={float(np.min(mx)):.4g} med={float(np.median(mx)):.4g} max={float(np.max(mx)):.4g}. "
            f"{thr_note}. "
            f"If amplitudes look fine in plots, check XDF units vs ARTIFACT_SEGMENT_AMPLITUDE_UNIT "
            f"or raise thresholds / use ARTIFACT_REJECT_MODE='zscore'."
        )
    return apply_segment_mask(mask, segments, labels, *optional_same_length)
