"""
visualize_errp_data.py
=======================
ERP visualization and quality-check tool for ErrP experiments.

Loads XDF files, extracts ErrP epochs (using same pipeline as generate_errp_decoder),
and produces diagnostic plots:

  1. ERP waveforms — mean ± 95% CI for error vs correct, per channel
  2. Difference wave — (error mean) − (correct mean) with CI
  3. Butterfly plot — all channels overlaid, both conditions
  4. Topographic maps — scalp distribution at key latencies
  5. Single-trial heatmap — per-trial sorted amplitude map (error + correct)
  6. Score histogram — (if a trained model .pkl is provided via --model flag)

Usage:
    python visualize_errp_data.py                         # auto-detects XDF files
    python visualize_errp_data.py --xdf path/to/file.xdf # specific file
    python visualize_errp_data.py --model path/to/errp_model.pkl

Markers used:
    Error:   ROBOT_EARLYSTOP (340) or ERRP_STIM_ERROR (430)
    Correct: ROBOT_END (320)       or ERRP_STIM_CORRECT (440)
"""

import argparse
import glob
import os
import pickle
import sys

os.environ["NUMBA_DISABLE_CACHING"] = "1"
os.environ["MNE_USE_NUMBA"] = "false"

import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # use non-interactive backend if display not available
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import mne

import config
from Utils.errp_feature_pipeline import (
    _default_error_codes,
    _default_correct_codes,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
TMIN     = float(getattr(config, "ERRP_EPOCH_TMIN", 0.0))
TMAX     = float(getattr(config, "ERRP_EPOCH_TMAX", 0.8))
FS       = float(config.FS)
N_TIMES  = int(round((TMAX - TMIN) * FS))
TIMES    = np.linspace(TMIN, TMAX, N_TIMES) * 1000   # ms

# Key latencies for topo maps (ms post-event)
TOPO_LATENCIES_MS = [100, 200, 300, 400, 600]

# Colors
COL_ERROR   = "#d62728"    # red
COL_CORRECT = "#1f77b4"    # blue
COL_DIFF    = "#2ca02c"    # green

# ──────────────────────────────────────────────────────────────────────────────
# Visualization-specific preprocessing constants
# (separate from the decoder pipeline — do NOT change the decoder path)
# ──────────────────────────────────────────────────────────────────────────────
# Pre-stimulus window: -200 to 0 ms is the standard baseline window for ErrP
# (Widmann et al. 2015; Iturrate et al. 2010; Chavarriaga & Millán 2014).
# The -200 ms onset captures any slow anticipatory activity and gives a stable
# amplitude reference uncontaminated by the ErrP itself.
VIZ_TMIN_S     = -0.2
VIZ_TMAX_S     = float(getattr(config, "ERRP_EPOCH_TMAX", 0.8))
VIZ_BASELINE_S = (-0.2, 0.0)   # correction window (seconds)

# Frontal EEG channels used as surrogate EOG for blink regression.
# No dedicated EOG channel was recorded; Fp1/Fpz/Fp2 are the closest scalp
# proxies for the vertical blink dipole (Gratton et al. 1983 regression method).
# These channels are loaded from the full 32-ch recording, used for regression
# against each analysis channel, then discarded before epoching.
BLINK_SURROGATE_CHS = ["Fp1", "Fpz", "Fp2"]


# ──────────────────────────────────────────────────────────────────────────────
# Visualization loading pipeline (MNE-based, zero-phase FIR + baseline)
# ──────────────────────────────────────────────────────────────────────────────

def _load_viz_epochs_mne(
    xdf_path: str,
    error_codes: list,
    correct_codes: list,
) -> tuple:
    """
    Load one XDF file and return baseline-corrected ErrP epochs.

    Pipeline:
      1. Load full recording (all valid channels)
      2. Zero-phase FIR bandpass 1-10 Hz + 60 Hz notch (no phase distortion)
      3. Gratton et al. (1983) surrogate blink regression — average of
         available Fp1/Fpz/Fp2 channels used as reference; per-channel
         least-squares coefficient subtracted before channel selection
      4. Select ERRP channels (config.ERRP_CHANNEL_NAMES)
      5. Epoch tmin=VIZ_TMIN_S to tmax=VIZ_TMAX_S with baseline=VIZ_BASELINE_S
      6. Reject epochs with P2P > 150 µV

    Returns:
        X: (n_epochs, n_ch, n_times) in µV, baseline-corrected
        y: (n_epochs,) — 1=error, 0=correct
        ch_names: list of channel names
    """
    from Utils.stream_utils import load_xdf, get_channel_names_from_xdf
    from Utils.preprocessing import get_valid_channel_mask_and_metadata

    eeg_stream, marker_stream = load_xdf(xdf_path)

    eeg_data = np.array(eeg_stream["time_series"]).T      # (n_ch, n_samp)
    eeg_ts   = np.array(eeg_stream["time_stamps"])
    all_ch   = get_channel_names_from_xdf(eeg_stream)

    # Drop mastoids and obviously bad channels
    valid_ch_names, _, valid_indices = get_valid_channel_mask_and_metadata(
        eeg_data, all_ch, fs=config.FS, drop_mastoids=True
    )
    eeg_data   = eeg_data[valid_indices, :]
    current_ch = list(valid_ch_names)

    # Build MNE Raw (µV → V for MNE conventions)
    info = mne.create_info(ch_names=current_ch, sfreq=config.FS, ch_types="eeg")
    raw  = mne.io.RawArray(eeg_data * 1e-6, info, verbose=False)
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="warn", verbose=False)
    except Exception:
        pass

    # Zero-phase FIR bandpass + 60 Hz notch
    raw.filter(
        l_freq=float(getattr(config, "LOWCUT_ERRP",  1.0)),
        h_freq=float(getattr(config, "HIGHCUT_ERRP", 10.0)),
        method="fir", phase="zero", fir_window="hamming",
        verbose=False,
    )
    raw.notch_filter(freqs=60.0, verbose=False)

    # Gratton surrogate blink regression on filtered continuous data
    raw_v = raw.get_data()                                # (n_ch, n_samp) in V
    available_fp = [ch for ch in BLINK_SURROGATE_CHS if ch in current_ch]
    if available_fp:
        fp_idx = [current_ch.index(ch) for ch in available_fp]
        fp_ref = raw_v[fp_idx, :].mean(axis=0)           # surrogate blink ref
        fp_var = np.var(fp_ref)
        if fp_var > 1e-30:
            for i in range(raw_v.shape[0]):
                b = np.cov(raw_v[i], fp_ref)[0, 1] / fp_var
                raw_v[i] -= b * fp_ref
        raw._data[:] = raw_v

    # Select ERRP channels; discard Fp surrogate channels
    errp_ch = list(getattr(config, "ERRP_CHANNEL_NAMES", []))
    if errp_ch:
        available = [ch for ch in errp_ch if ch in current_ch]
        if available:
            raw.pick_channels(available, ordered=True)
            current_ch = list(raw.ch_names)

    # Build MNE events from marker timestamps
    marker_ts   = np.array(marker_stream["time_stamps"], dtype=float)
    raw_markers = marker_stream["time_series"]
    marker_vals = np.array([int(float(m[0])) for m in raw_markers], dtype=int)

    events_list, labels_list = [], []
    for ts, code in zip(marker_ts, marker_vals):
        if code in error_codes:
            label = 1
        elif code in correct_codes:
            label = 0
        else:
            continue
        samp = int(np.searchsorted(eeg_ts, ts))
        samp = int(np.clip(samp, 0, raw.n_times - 1))
        events_list.append([samp, 0, label])
        labels_list.append(label)

    if not events_list:
        return np.empty((0, len(current_ch), 0)), np.empty(0, dtype=int), current_ch

    events = np.array(events_list, dtype=int)
    event_id = {}
    if any(l == 1 for l in labels_list):
        event_id["error"] = 1
    if any(l == 0 for l in labels_list):
        event_id["correct"] = 0

    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=VIZ_TMIN_S, tmax=VIZ_TMAX_S,
        baseline=VIZ_BASELINE_S,
        reject={"eeg": 150e-6},
        preload=True, verbose=False,
    )

    if len(epochs) == 0:
        return np.empty((0, len(current_ch), 0)), np.empty(0, dtype=int), current_ch

    X = epochs.get_data() * 1e6          # V → µV
    y = epochs.events[:, 2].astype(int)  # 1=error, 0=correct
    return X, y, list(epochs.ch_names)


def load_viz_data(
    xdf_files: list,
    error_codes: list,
    correct_codes: list,
    verbose: bool = True,
) -> tuple:
    """
    Multi-file loader using the visualization pipeline (zero-phase FIR + baseline).

    Returns:
        X:        (n_epochs, n_ch, n_times) in µV, baseline-corrected
        y:        (n_epochs,) labels
        ch_names: list of channel names
        times_ms: (n_times,) time axis in milliseconds
    """
    all_X, all_y = [], []
    ref_ch = None

    for path in xdf_files:
        try:
            X, y, ch_names = _load_viz_epochs_mne(path, error_codes, correct_codes)
        except Exception as exc:
            print(f"[ErrP viz] WARNING: skipping {os.path.basename(path)}: {exc}",
                  file=sys.stderr)
            continue

        if X.shape[0] == 0:
            if verbose:
                print(f"[ErrP viz] {os.path.basename(path)}: no epochs found")
            continue

        if ref_ch is None:
            ref_ch = ch_names
        elif ch_names != ref_ch:
            print(f"[ErrP viz] WARNING: channel mismatch in {os.path.basename(path)} — skipping",
                  file=sys.stderr)
            continue

        all_X.append(X)
        all_y.append(y)
        if verbose:
            n_e = int((y == 1).sum())
            n_c = int((y == 0).sum())
            print(f"[ErrP viz] {os.path.basename(path)}: error={n_e}, correct={n_c}")

    if not all_X:
        raise ValueError("No ErrP epochs found in any XDF file.")

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    n_times  = X_all.shape[2]
    times_ms = np.linspace(VIZ_TMIN_S, VIZ_TMAX_S, n_times) * 1000

    if verbose:
        print(f"\n[ErrP viz] Total: {X_all.shape[0]} epochs "
              f"(error={int((y_all==1).sum())}, correct={int((y_all==0).sum())})")

    return X_all, y_all, ref_ch, times_ms


# ──────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ──────────────────────────────────────────────────────────────────────────────

def _mean_ci(X: np.ndarray, ci: float = 0.95):
    """
    Return mean and confidence interval half-width for an (n_trials, n_times) array.
    Uses t-distribution CI.
    """
    from scipy import stats as sp_stats
    n = X.shape[0]
    mean = X.mean(axis=0)
    if n < 2:
        return mean, np.zeros_like(mean)
    se = X.std(axis=0, ddof=1) / np.sqrt(n)
    t = sp_stats.t.ppf((1 + ci) / 2, df=n - 1)
    return mean, t * se


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1 — Per-channel ERP waveforms
# ──────────────────────────────────────────────────────────────────────────────

def plot_erp_waveforms(
    X_err: np.ndarray,
    X_cor: np.ndarray,
    ch_names: list,
    title_suffix: str = "",
    times_ms: np.ndarray = None,
):
    """
    Plot mean ERP ± 95% CI for error vs correct, one subplot per channel.

    Args:
        X_err:    (n_error, n_ch, n_times)
        X_cor:    (n_correct, n_ch, n_times)
        times_ms: (n_times,) time axis in ms; falls back to global TIMES if None
    """
    _times = times_ms if times_ms is not None else TIMES
    n_ch = X_err.shape[1]
    ncols = min(n_ch, 3)
    nrows = int(np.ceil(n_ch / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    fig.suptitle(f"ErrP Waveforms — Error vs Correct{title_suffix}\n"
                 f"n_error={X_err.shape[0]}, n_correct={X_cor.shape[0]}", fontsize=13)

    for ch_idx, ch_name in enumerate(ch_names):
        row, col = divmod(ch_idx, ncols)
        ax = axes[row][col]

        err_ts = X_err[:, ch_idx, :]     # (n_error, n_times)
        cor_ts = X_cor[:, ch_idx, :]

        mu_e, ci_e = _mean_ci(err_ts)
        mu_c, ci_c = _mean_ci(cor_ts)

        ax.fill_between(_times, mu_e - ci_e, mu_e + ci_e, alpha=0.25, color=COL_ERROR)
        ax.fill_between(_times, mu_c - ci_c, mu_c + ci_c, alpha=0.25, color=COL_CORRECT)
        ax.plot(_times, mu_e, color=COL_ERROR,   lw=1.8, label=f"Error  (n={X_err.shape[0]})")
        ax.plot(_times, mu_c, color=COL_CORRECT, lw=1.8, label=f"Correct (n={X_cor.shape[0]})")
        ax.axhline(0, color="k", lw=0.7, ls="--")
        ax.axvline(0, color="gray", lw=0.8, ls=":")
        for lat in [100, 200, 400]:
            ax.axvline(lat, color="gray", lw=0.5, ls=":")
        ax.set_title(ch_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        ax.invert_yaxis()   # EEG convention: negative up
        if ch_idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    # Hide unused subplots
    for ch_idx in range(n_ch, nrows * ncols):
        row, col = divmod(ch_idx, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()


# ──────────────────────────────────────────────────────────────────────────────
# Plot 2 — Difference waveforms
# ──────────────────────────────────────────────────────────────────────────────

def plot_difference_wave(
    X_err: np.ndarray,
    X_cor: np.ndarray,
    ch_names: list,
    title_suffix: str = "",
    times_ms: np.ndarray = None,
):
    """
    Plot the difference wave (error − correct) per channel.
    """
    _times  = times_ms if times_ms is not None else TIMES
    _ntimes = len(_times)

    n_ch = X_err.shape[1]
    ncols = min(n_ch, 3)
    nrows = int(np.ceil(n_ch / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    fig.suptitle(f"ErrP Difference Wave (Error − Correct){title_suffix}", fontsize=13)

    for ch_idx, ch_name in enumerate(ch_names):
        row, col = divmod(ch_idx, ncols)
        ax = axes[row][col]

        diff_trials = X_err[:, ch_idx, :].mean(axis=0) - X_cor[:, ch_idx, :].mean(axis=0)

        # Bootstrap CI on difference
        n_boot = 1000
        n_e, n_c = X_err.shape[0], X_cor.shape[0]
        boot_diffs = np.empty((n_boot, _ntimes))
        rng = np.random.default_rng(0)
        for b in range(n_boot):
            re = X_err[rng.integers(0, n_e, n_e), ch_idx, :].mean(axis=0)
            rc = X_cor[rng.integers(0, n_c, n_c), ch_idx, :].mean(axis=0)
            boot_diffs[b] = re - rc
        ci_lo = np.percentile(boot_diffs, 2.5, axis=0)
        ci_hi = np.percentile(boot_diffs, 97.5, axis=0)

        ax.fill_between(_times, ci_lo, ci_hi, alpha=0.3, color=COL_DIFF)
        ax.plot(_times, diff_trials, color=COL_DIFF, lw=1.8)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.axvline(0, color="gray", lw=0.8, ls=":")
        for lat in [100, 200, 400]:
            ax.axvline(lat, color="gray", lw=0.5, ls=":")
        ax.set_title(f"{ch_name}  (Error − Correct)", fontsize=10)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Δ Amplitude (µV)")
        ax.invert_yaxis()

    for ch_idx in range(n_ch, nrows * ncols):
        row, col = divmod(ch_idx, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()


# ──────────────────────────────────────────────────────────────────────────────
# Plot 3 — Butterfly plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_butterfly(
    X_err: np.ndarray,
    X_cor: np.ndarray,
    ch_names: list,
    title_suffix: str = "",
    times_ms: np.ndarray = None,
):
    """Overlay all channels (butterfly) for both conditions."""
    _times = times_ms if times_ms is not None else TIMES
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    fig.suptitle(f"Butterfly Plot — All Channels{title_suffix}", fontsize=13)

    for ax, X, label, col, n in [
        (axes[0], X_err, "Error",   COL_ERROR,   X_err.shape[0]),
        (axes[1], X_cor, "Correct", COL_CORRECT, X_cor.shape[0]),
    ]:
        mean_X = X.mean(axis=0)   # (n_ch, n_times)
        for ch_idx in range(mean_X.shape[0]):
            ax.plot(_times, mean_X[ch_idx], lw=1.0, alpha=0.7,
                    label=ch_names[ch_idx] if ch_idx == 0 else "_nolegend_")
        ax.axhline(0, color="k", lw=0.7, ls="--")
        ax.axvline(0, color="gray", lw=0.8, ls=":")
        ax.set_title(f"{label}  (n={n})", fontsize=11)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        ax.invert_yaxis()

        # Annotate channel names on the right margin
        for ch_idx, ch in enumerate(ch_names):
            y_pos = mean_X[ch_idx, -1]
            ax.annotate(ch, xy=(_times[-1], y_pos), xytext=(3, 0),
                        textcoords="offset points", fontsize=7, va="center")

    plt.tight_layout()


# ──────────────────────────────────────────────────────────────────────────────
# Plot 4 — Topographic maps
# ──────────────────────────────────────────────────────────────────────────────

def plot_topomap(
    X_err: np.ndarray,
    X_cor: np.ndarray,
    ch_names: list,
    latencies_ms: list = None,
    title_suffix: str = "",
    times_ms: np.ndarray = None,
):
    """
    Scalp topography at key latencies using MNE.

    Shows error, correct, and difference (error−correct) maps.
    """
    _times = times_ms if times_ms is not None else TIMES
    if latencies_ms is None:
        latencies_ms = TOPO_LATENCIES_MS

    # Filter to latencies within epoch window
    latencies_ms = [l for l in latencies_ms if _times[0] <= l <= _times[-1]]
    if not latencies_ms:
        print("[ErrP topo] No valid latencies within epoch window — skipping topomaps.")
        return

    try:
        info = mne.create_info(ch_names=list(ch_names), sfreq=FS, ch_types="eeg")
        montage = mne.channels.make_standard_montage("standard_1020")
        info.set_montage(montage, match_case=False, on_missing="warn")
    except Exception as exc:
        print(f"[ErrP topo] Could not build MNE montage: {exc} — skipping topomaps.")
        return

    mean_err = X_err.mean(axis=0)   # (n_ch, n_times)
    mean_cor = X_cor.mean(axis=0)
    diff     = mean_err - mean_cor

    n_lat = len(latencies_ms)
    fig, axes = plt.subplots(3, n_lat, figsize=(3.5 * n_lat, 9))
    fig.suptitle(f"ErrP Topomaps{title_suffix}\n(rows: Error | Correct | Difference)", fontsize=12)

    for col_idx, lat_ms in enumerate(latencies_ms):
        # Find sample index closest to requested latency
        t_idx = int(np.argmin(np.abs(_times - lat_ms)))
        t_idx = int(np.clip(t_idx, 0, len(_times) - 1))

        for row_idx, (data_2d, row_label) in enumerate([
            (mean_err, "Error"),
            (mean_cor, "Correct"),
            (diff,     "Diff"),
        ]):
            ax = axes[row_idx][col_idx]
            amp = data_2d[:, t_idx]   # (n_ch,)

            try:
                im, _ = mne.viz.plot_topomap(
                    amp, info, axes=ax, show=False,
                    cmap="RdBu_r", vlim=(np.percentile(amp, 5), np.percentile(amp, 95)),
                    contours=4,
                )
            except Exception:
                ax.text(0.5, 0.5, f"{lat_ms}ms\n{row_label}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
                continue

            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=10)
            if row_idx == 0:
                ax.set_title(f"{lat_ms} ms", fontsize=10)

    plt.tight_layout()


# ──────────────────────────────────────────────────────────────────────────────
# Plot 5 — Single-trial heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_single_trial_heatmap(
    X_err: np.ndarray,
    X_cor: np.ndarray,
    ch_names: list,
    max_trials: int = 80,
    title_suffix: str = "",
    times_ms: np.ndarray = None,
):
    """
    Image plot: each row = one trial (amplitude over time), sorted by condition.

    Shows error trials on top, correct trials below, with a separator.
    The channel with highest error–correct amplitude difference is selected.
    """
    _times = times_ms if times_ms is not None else TIMES

    # Select "most informative" channel (max |mean_error - mean_correct|)
    diff_amp = np.abs(X_err.mean(axis=0) - X_cor.mean(axis=0))   # (n_ch, n_times)
    best_ch  = int(np.argmax(diff_amp.max(axis=1)))
    ch_label = ch_names[best_ch]

    err_1d  = X_err[:, best_ch, :]   # (n_err, n_times)
    cor_1d  = X_cor[:, best_ch, :]

    # Sub-sample if too many trials
    def _subsample(X, n):
        if X.shape[0] <= n:
            return X
        idx = np.random.default_rng(42).choice(X.shape[0], n, replace=False)
        idx.sort()
        return X[idx]

    err_1d = _subsample(err_1d, max_trials // 2)
    cor_1d = _subsample(cor_1d, max_trials // 2)

    combined = np.vstack([err_1d, cor_1d])
    n_err_show = err_1d.shape[0]

    vmax = float(np.percentile(np.abs(combined), 95))

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        combined,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        interpolation="nearest",
        extent=[_times[0], _times[-1], combined.shape[0], 0],
    )
    ax.axhline(n_err_show - 0.5, color="k", lw=2.0, ls="--", label="Error | Correct boundary")
    ax.axvline(0, color="gray", lw=1.0, ls=":")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Trial")
    ax.set_title(
        f"Single-Trial Heatmap — Channel: {ch_label}{title_suffix}\n"
        f"Top={n_err_show} error trials, Bottom={cor_1d.shape[0]} correct trials",
        fontsize=11,
    )
    plt.colorbar(im, ax=ax, label="Amplitude (µV)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()


# ──────────────────────────────────────────────────────────────────────────────
# Plot 6 — Classifier score distribution (if model provided)
# ──────────────────────────────────────────────────────────────────────────────

def plot_model_scores(
    X: np.ndarray,
    y: np.ndarray,
    bundle: dict,
    title_suffix: str = "",
):
    """Apply the trained ErrP model to held-out data and plot score distributions."""
    try:
        xdc = bundle["xdawn"]
        clf = bundle["classifier"]
        tl  = bundle["tl_star"]
        th  = bundle["th_star"]

        C = xdc.transform(X)
        scores = clf.predict_proba(C)[:, 1]   # P(error)

        base_module = __import__("Generate_Riemannian_adaptive")
        base_module._plot_scores_hist_with_thresholds(scores, y, tl, th)
        plt.suptitle(f"ErrP Score Distribution (P(error)){title_suffix}", fontsize=12)
    except Exception as exc:
        print(f"[ErrP] Score plot failed: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Print epoch summary
# ──────────────────────────────────────────────────────────────────────────────

def print_epoch_summary(X_err, X_cor, ch_names, times_ms=None):
    """Print amplitude statistics for error vs correct epochs."""
    _times   = times_ms if times_ms is not None else TIMES
    _ntimes  = len(_times)
    print(f"\n{'─'*55}")
    print(f"  ErrP Epoch Summary")
    print(f"{'─'*55}")
    print(f"  Error epochs:   {X_err.shape[0]}  |  Correct epochs: {X_cor.shape[0]}")
    print(f"  Channels:       {ch_names}")
    print(f"  Epoch window:   {_times[0]:.0f} – {_times[-1]:.0f} ms ({_ntimes} samples @ {FS:.0f} Hz)")

    print(f"\n  {'Channel':<8}  {'Error mean peak (µV)':>22}  {'Correct mean peak (µV)':>24}  {'Diff':>10}")
    print(f"  {'─'*7}  {'─'*22}  {'─'*24}  {'─'*10}")
    for ch_idx, ch in enumerate(ch_names):
        peak_e = float(np.abs(X_err[:, ch_idx, :].mean(axis=0)).max())
        peak_c = float(np.abs(X_cor[:, ch_idx, :].mean(axis=0)).max())
        print(f"  {ch:<8}  {peak_e:>22.3f}  {peak_c:>24.3f}  {peak_e-peak_c:>10.3f}")
    print(f"{'─'*55}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    parser = argparse.ArgumentParser(description="Visualize ErrP epochs from XDF files.")
    parser.add_argument("--xdf", nargs="*", help="XDF file(s) to load (default: auto-detect)")
    parser.add_argument("--model", default=None, help="Path to trained ErrP .pkl bundle (optional)")
    parser.add_argument("--no-topo", action="store_true", help="Skip topographic maps")
    parser.add_argument("--no-heatmap", action="store_true", help="Skip single-trial heatmap")
    ns = parser.parse_args(args)

    # --- Locate XDF files ---
    if ns.xdf:
        xdf_files = ns.xdf
    else:
        subject = config.TRAINING_SUBJECT
        training_dir = os.path.join(config.DATA_DIR, f"sub-{subject}", "training_data")
        xdf_files = sorted(glob.glob(os.path.join(training_dir, "**", "*.xdf"), recursive=True))
        xdf_files = [f for f in xdf_files if "OBS" not in os.path.basename(f)]
        if not xdf_files:
            raise FileNotFoundError(f"No XDF files found in: {training_dir}")

    print(f"[ErrP viz] Loading {len(xdf_files)} file(s) "
          f"(zero-phase FIR, Gratton blink regression, baseline {int(VIZ_TMIN_S*1000)}–0 ms)...")

    error_codes   = _default_error_codes()
    correct_codes = _default_correct_codes()

    X, y, ch_names, times_ms = load_viz_data(
        xdf_files,
        error_codes=error_codes,
        correct_codes=correct_codes,
        verbose=True,
    )

    if X.shape[0] == 0:
        print("[ErrP viz] No epochs found. Check marker codes in XDF files.")
        return

    X_err = X[y == 1]
    X_cor = X[y == 0]

    if ns.xdf:
        basenames = [os.path.splitext(os.path.basename(f))[0] for f in xdf_files]
        if len(basenames) == 1:
            suffix = f"  ({basenames[0]})"
        else:
            suffix = f"  ({len(xdf_files)} files pooled)"
    else:
        suffix = f"  (sub-{config.TRAINING_SUBJECT})"

    print_epoch_summary(X_err, X_cor, ch_names, times_ms=times_ms)

    # --- Generate all plots ---
    plot_erp_waveforms(X_err, X_cor, ch_names, title_suffix=suffix, times_ms=times_ms)
    plot_difference_wave(X_err, X_cor, ch_names, title_suffix=suffix, times_ms=times_ms)
    plot_butterfly(X_err, X_cor, ch_names, title_suffix=suffix, times_ms=times_ms)

    if not ns.no_topo:
        plot_topomap(X_err, X_cor, ch_names, title_suffix=suffix, times_ms=times_ms)

    if not ns.no_heatmap:
        plot_single_trial_heatmap(X_err, X_cor, ch_names, title_suffix=suffix, times_ms=times_ms)

    if ns.model:
        try:
            with open(ns.model, "rb") as f:
                bundle = pickle.load(f)
            plot_model_scores(X, y, bundle, title_suffix=suffix)
        except Exception as exc:
            print(f"[ErrP viz] Could not load model {ns.model}: {exc}")

    plt.show()
    print("[ErrP viz] Done.")


if __name__ == "__main__":
    main()
