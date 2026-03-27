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
    load_and_preprocess_errp_xdf,
    load_errp_training_data,
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
    ch_names: list[str],
    title_suffix: str = "",
):
    """
    Plot mean ERP ± 95% CI for error vs correct, one subplot per channel.

    Args:
        X_err: (n_error, n_ch, n_times)
        X_cor: (n_correct, n_ch, n_times)
    """
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

        ax.fill_between(TIMES, mu_e - ci_e, mu_e + ci_e, alpha=0.25, color=COL_ERROR)
        ax.fill_between(TIMES, mu_c - ci_c, mu_c + ci_c, alpha=0.25, color=COL_CORRECT)
        ax.plot(TIMES, mu_e, color=COL_ERROR,   lw=1.8, label=f"Error  (n={X_err.shape[0]})")
        ax.plot(TIMES, mu_c, color=COL_CORRECT, lw=1.8, label=f"Correct (n={X_cor.shape[0]})")
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
    ch_names: list[str],
    title_suffix: str = "",
):
    """
    Plot the difference wave (error − correct) per channel.
    """
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
        boot_diffs = np.empty((n_boot, N_TIMES))
        rng = np.random.default_rng(0)
        for b in range(n_boot):
            re = X_err[rng.integers(0, n_e, n_e), ch_idx, :].mean(axis=0)
            rc = X_cor[rng.integers(0, n_c, n_c), ch_idx, :].mean(axis=0)
            boot_diffs[b] = re - rc
        ci_lo = np.percentile(boot_diffs, 2.5, axis=0)
        ci_hi = np.percentile(boot_diffs, 97.5, axis=0)

        ax.fill_between(TIMES, ci_lo, ci_hi, alpha=0.3, color=COL_DIFF)
        ax.plot(TIMES, diff_trials, color=COL_DIFF, lw=1.8)
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
    ch_names: list[str],
    title_suffix: str = "",
):
    """Overlay all channels (butterfly) for both conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    fig.suptitle(f"Butterfly Plot — All Channels{title_suffix}", fontsize=13)

    for ax, X, label, col, n in [
        (axes[0], X_err, "Error",   COL_ERROR,   X_err.shape[0]),
        (axes[1], X_cor, "Correct", COL_CORRECT, X_cor.shape[0]),
    ]:
        mean_X = X.mean(axis=0)   # (n_ch, n_times)
        for ch_idx in range(mean_X.shape[0]):
            ax.plot(TIMES, mean_X[ch_idx], lw=1.0, alpha=0.7,
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
            ax.annotate(ch, xy=(TIMES[-1], y_pos), xytext=(3, 0),
                        textcoords="offset points", fontsize=7, va="center")

    plt.tight_layout()


# ──────────────────────────────────────────────────────────────────────────────
# Plot 4 — Topographic maps
# ──────────────────────────────────────────────────────────────────────────────

def plot_topomap(
    X_err: np.ndarray,
    X_cor: np.ndarray,
    ch_names: list[str],
    latencies_ms: list[int] = None,
    title_suffix: str = "",
):
    """
    Scalp topography at key latencies using MNE.

    Shows error, correct, and difference (error−correct) maps.
    """
    if latencies_ms is None:
        latencies_ms = TOPO_LATENCIES_MS

    # Filter to latencies within epoch window
    latencies_ms = [l for l in latencies_ms if TMIN * 1000 <= l <= TMAX * 1000]
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
        # Find sample index for this latency
        t_idx = int(round((lat_ms / 1000 - TMIN) * FS))
        t_idx = np.clip(t_idx, 0, N_TIMES - 1)

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
    ch_names: list[str],
    max_trials: int = 80,
    title_suffix: str = "",
):
    """
    Image plot: each row = one trial (amplitude over time), sorted by condition.

    Shows error trials on top, correct trials below, with a separator.
    The channel with highest error–correct amplitude difference is selected.
    """
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
        extent=[TIMES[0], TIMES[-1], combined.shape[0], 0],
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

def print_epoch_summary(X_err, X_cor, ch_names):
    """Print amplitude statistics for error vs correct epochs."""
    print(f"\n{'─'*55}")
    print(f"  ErrP Epoch Summary")
    print(f"{'─'*55}")
    print(f"  Error epochs:   {X_err.shape[0]}  |  Correct epochs: {X_cor.shape[0]}")
    print(f"  Channels:       {ch_names}")
    print(f"  Epoch window:   {TMIN*1000:.0f} – {TMAX*1000:.0f} ms ({N_TIMES} samples @ {FS:.0f} Hz)")

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

    print(f"[ErrP viz] Loading {len(xdf_files)} file(s)...")

    X, y, ch_names = load_errp_training_data(
        xdf_files,
        error_codes=_default_error_codes(),
        correct_codes=_default_correct_codes(),
        verbose=True,
    )

    if X.shape[0] == 0:
        print("[ErrP viz] No epochs found. Check marker codes in XDF files.")
        return

    X_err = X[y == 1]
    X_cor = X[y == 0]

    suffix = f"  (sub-{config.TRAINING_SUBJECT})"
    print_epoch_summary(X_err, X_cor, ch_names)

    # --- Generate all plots ---
    plot_erp_waveforms(X_err, X_cor, ch_names, title_suffix=suffix)
    plot_difference_wave(X_err, X_cor, ch_names, title_suffix=suffix)
    plot_butterfly(X_err, X_cor, ch_names, title_suffix=suffix)

    if not ns.no_topo:
        plot_topomap(X_err, X_cor, ch_names, title_suffix=suffix)

    if not ns.no_heatmap:
        plot_single_trial_heatmap(X_err, X_cor, ch_names, title_suffix=suffix)

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
