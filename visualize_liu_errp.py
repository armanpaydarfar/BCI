"""
visualize_liu_errp.py — ERP visualization for the Liu et al. (2025) ErrP dataset.

Loads combinedEpochs_v2.mat or eegEpochs_16subs_chanInterp_control.mat and
produces six diagnostic plots:

  1. Grand-average ERP waveforms — error vs correct, per channel
  2. Magnitude-stratified ERP waveforms at Cz
  3. Difference waveform (error − correct) per channel
  4. Per-subject ERP variability at Cz
  5. Pe amplitude by rotation magnitude (bar chart)
  6. Single-trial heatmap at Cz sorted by magnitude

Data notes:
  - MATLAB v7.3 (HDF5) format — loaded with h5py, not scipy.io.
  - Epochs are already theta-band filtered (1–10 Hz) by the authors.
  - Do not re-filter.
  - 32 EEG channels ordered as per Liu et al. Methods (ANT Neuro eego system).
  - 768 samples per epoch at 512 Hz → −0.498 s to +1.0 s relative to event onset.
  - label: 0 = correct (0° rotation), 1 = error (3/6/9/12° rotation).

Reference: Liu, Iwane et al. (2025). Brain-computer interface training fosters
perceptual skills to detect errors. bioRxiv. doi:10.1101/2025.04.26.650792

Usage:
  python visualize_liu_errp.py --file "/path/to/combinedEpochs_v2.mat"
  python visualize_liu_errp.py --file combined... --channels Fz,FCz,Cz --subjects 1,2,3
  python visualize_liu_errp.py --file combined... --harmony-channels-only
"""

import argparse
import os
import sys

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend — saves to file

# =============================================================================
# Channel layout (Liu et al. Methods, ANT Neuro eego system, 10-20 positions)
# Array index 0 = first channel in rotation_data[:, 0, :].
# =============================================================================
LIU_CHANNEL_NAMES = [
    "AF3", "AF4",
    "F3", "F1", "Fz", "F2", "F4",
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C3", "C1", "Cz", "C2", "C4",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P3", "P1", "Pz", "P2", "P4",
    "PO3", "POz", "PO4",
    "O1", "O2",
]
assert len(LIU_CHANNEL_NAMES) == 32

_CH_IDX = {name: i for i, name in enumerate(LIU_CHANNEL_NAMES)}

# Harmony ErrP channel names from config (order matters for model training).
HARMONY_ERRP_CHANNELS = ["F3", "Fz", "F4", "FC1", "FC2", "Cz"]

# Pe region from the paper: significant differences observed ~340–520 ms post-onset.
PE_TMIN_MS = 340
PE_TMAX_MS = 520

# ERN region.
ERN_TMIN_MS = 80
ERN_TMAX_MS = 150

# Baseline window for pre-stimulus mean subtraction.
BASELINE_TMIN_MS = -200
BASELINE_TMAX_MS = 0

FS = 512  # Hz — confirmed from params.fsamp in the mat file


# =============================================================================
# Data loading
# =============================================================================

def _resolve_ref(f, ref):
    """Dereference an HDF5 object reference and return the underlying dataset."""
    return f[ref]


def load_liu_epochs(mat_path: str,
                    subjects: list[int] | None = None,
                    channel_names: list[str] | None = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Load epochs from a Liu et al. .mat file.

    Parameters
    ----------
    mat_path : str
        Path to combinedEpochs_v2.mat or eegEpochs_16subs_chanInterp_control.mat.
    subjects : list[int] | None
        1-based subject indices to load (e.g. [1, 2, 3]).  None = all 16.
    channel_names : list[str] | None
        Channel names to retain (must be in LIU_CHANNEL_NAMES).  None = all 32.

    Returns
    -------
    X : ndarray, shape (n_epochs, n_channels, n_samples)
        EEG epochs (already filtered, µV).
    y : ndarray, shape (n_epochs,)
        Integer labels: 0 = correct, 1 = error.
    mag : ndarray, shape (n_epochs,)
        Rotation magnitude in degrees: 0, 3, 6, 9, or 12.
    sub_ids : ndarray, shape (n_epochs,)
        1-based subject index for each epoch.
    time_s : ndarray, shape (n_samples,)
        Time axis in seconds relative to event onset.
    loaded_subjects : list[int]
        Which 1-based subject indices were actually loaded.
    """
    with h5py.File(mat_path, "r") as f:
        # Determine which top-level struct to read.
        if "combinedEpochs" in f:
            struct = f["combinedEpochs"]
        elif "eegEpochs" in f:
            struct = f["eegEpochs"]
        else:
            raise ValueError(f"No recognised struct in {mat_path}. "
                             f"Keys: {list(f.keys())}")

        n_subjects_in_file = struct["label"].shape[0]
        if subjects is None:
            subjects = list(range(1, n_subjects_in_file + 1))

        # Validate subject indices.
        for s in subjects:
            if s < 1 or s > n_subjects_in_file:
                raise ValueError(f"Subject {s} out of range [1, {n_subjects_in_file}].")

        # Resolve channel selection.
        if channel_names is None:
            ch_idx = list(range(32))
        else:
            unknown = [c for c in channel_names if c not in _CH_IDX]
            if unknown:
                raise ValueError(f"Unknown channel(s): {unknown}. "
                                 f"Valid names: {LIU_CHANNEL_NAMES}")
            ch_idx = [_CH_IDX[c] for c in channel_names]

        Xs, ys, mags, sids = [], [], [], []

        for s in subjects:
            row = s - 1  # 0-based index into the struct arrays.
            eeg  = f[struct["rotation_data"][row, 0]][()]  # (epochs, 32, 768)
            lbl  = f[struct["label"][row, 0]][()].flatten().astype(int)
            mgn  = f[struct["magnitude"][row, 0]][()].flatten()

            eeg = eeg[:, ch_idx, :]  # (epochs, n_ch, 768)

            Xs.append(eeg)
            ys.append(lbl)
            mags.append(mgn)
            sids.append(np.full(len(lbl), s, dtype=int))

    X      = np.concatenate(Xs, axis=0).astype(np.float64)
    y      = np.concatenate(ys)
    mag    = np.concatenate(mags)
    sub_id = np.concatenate(sids)

    # Time axis: 768 samples, indices −255 … +512 at 512 Hz.
    time_s = np.arange(-255, 513) / FS

    return X, y, mag, sub_id, time_s, subjects


# =============================================================================
# Pre-processing helpers
# =============================================================================

def baseline_correct(X: np.ndarray, time_s: np.ndarray,
                     tmin: float = BASELINE_TMIN_MS / 1000,
                     tmax: float = BASELINE_TMAX_MS / 1000) -> np.ndarray:
    """
    Subtract the per-epoch mean over [tmin, tmax] seconds from each epoch.

    This is standard offline ERP baseline correction and is applied only for
    visualization — it is not part of the online inference path.
    """
    mask = (time_s >= tmin) & (time_s <= tmax)
    baseline = X[:, :, mask].mean(axis=2, keepdims=True)
    return X - baseline


def _ms(time_s: np.ndarray) -> np.ndarray:
    return time_s * 1000


# =============================================================================
# Per-subject averages (used by multiple plots)
# =============================================================================

def _subject_averages(X, y, mag, sub_id, loaded_subjects, condition):
    """
    Return per-subject condition averages.

    condition: "error", "correct", or an int magnitude.
    Returns ndarray of shape (n_subjects, n_ch, n_samples).
    """
    avgs = []
    for s in loaded_subjects:
        mask = sub_id == s
        if condition == "error":
            mask &= (y == 1)
        elif condition == "correct":
            mask &= (y == 0)
        else:
            mask &= (mag == condition)
        if mask.sum() == 0:
            avgs.append(np.full((X.shape[1], X.shape[2]), np.nan))
        else:
            avgs.append(X[mask].mean(axis=0))
    return np.array(avgs)  # (n_subjects, n_ch, n_samples)


# =============================================================================
# Plot 1 — Grand-average ERP: error vs correct
# =============================================================================

def plot_grand_average(X, y, sub_id, loaded_subjects, time_s, channel_names, out_dir):
    n_ch = len(channel_names)
    ncols = min(n_ch, 3)
    nrows = int(np.ceil(n_ch / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    sub_err  = _subject_averages(X, y, None, sub_id, loaded_subjects, "error")
    sub_corr = _subject_averages(X, y, None, sub_id, loaded_subjects, "correct")

    t_ms = _ms(time_s)

    for ci, ch_name in enumerate(channel_names):
        ax = axes[ci // ncols][ci % ncols]
        err_mean  = np.nanmean(sub_err[:, ci, :],  axis=0)
        err_sem   = np.nanstd(sub_err[:, ci, :],   axis=0) / np.sqrt(len(loaded_subjects))
        corr_mean = np.nanmean(sub_corr[:, ci, :], axis=0)
        corr_sem  = np.nanstd(sub_corr[:, ci, :],  axis=0) / np.sqrt(len(loaded_subjects))

        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
        ax.axvspan(ERN_TMIN_MS, ERN_TMAX_MS, alpha=0.10, color="blue",  label="_ERN")
        ax.axvspan(PE_TMIN_MS,  PE_TMAX_MS,  alpha=0.10, color="red",   label="_Pe")

        ax.fill_between(t_ms, err_mean - err_sem,   err_mean + err_sem,   alpha=0.25, color="crimson")
        ax.fill_between(t_ms, corr_mean - corr_sem, corr_mean + corr_sem, alpha=0.25, color="steelblue")
        ax.plot(t_ms, err_mean,  color="crimson",   linewidth=1.5, label="Error")
        ax.plot(t_ms, corr_mean, color="steelblue", linewidth=1.5, label="Correct")

        ax.set_title(ch_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        ax.set_xlim(-200, 900)
        if ci == 0:
            ax.legend(fontsize=8)

    # Hide unused axes.
    for ci in range(n_ch, nrows * ncols):
        axes[ci // ncols][ci % ncols].set_visible(False)

    fig.suptitle(
        f"Grand-average ERP: Error vs Correct  (N={len(loaded_subjects)} subjects)\n"
        f"Shaded bands: ERN [{ERN_TMIN_MS}–{ERN_TMAX_MS} ms], Pe [{PE_TMIN_MS}–{PE_TMAX_MS} ms]",
        fontsize=12
    )
    fig.tight_layout()
    out = os.path.join(out_dir, "plot1_grand_average.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# =============================================================================
# Plot 2 — Magnitude-stratified ERP at a single channel
# =============================================================================

def plot_magnitude_stratified(X, y, mag, sub_id, loaded_subjects, time_s,
                               channel_names, focal_channel, out_dir):
    if focal_channel not in channel_names:
        print(f"  [Plot 2] {focal_channel} not in selected channels — skipping.")
        return

    ch_pos  = channel_names.index(focal_channel)
    t_ms    = _ms(time_s)
    magnitudes = [0, 3, 6, 9, 12]
    colours    = ["steelblue", "#ffc107", "#ff7f0e", "#e74c3c", "#8e44ad"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
    ax.axvspan(PE_TMIN_MS, PE_TMAX_MS, alpha=0.10, color="red")

    for m, col in zip(magnitudes, colours):
        sub_avg = _subject_averages(X, y, mag, sub_id, loaded_subjects, m)
        mean    = np.nanmean(sub_avg[:, ch_pos, :], axis=0)
        sem     = np.nanstd(sub_avg[:, ch_pos, :], axis=0) / np.sqrt(len(loaded_subjects))
        label   = f"{int(m)}°" + (" (correct)" if m == 0 else "")
        ax.fill_between(t_ms, mean - sem, mean + sem, alpha=0.20, color=col)
        ax.plot(t_ms, mean, color=col, linewidth=1.5, label=label)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_xlim(-200, 900)
    ax.legend(title="Rotation magnitude", fontsize=9)
    ax.set_title(
        f"Magnitude-stratified ERP at {focal_channel}  (N={len(loaded_subjects)} subjects)\n"
        f"Pe region shaded [{PE_TMIN_MS}–{PE_TMAX_MS} ms]",
        fontsize=11
    )
    fig.tight_layout()
    out = os.path.join(out_dir, "plot2_magnitude_stratified.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# =============================================================================
# Plot 3 — Difference waveform (error − correct) per channel
# =============================================================================

def plot_difference_waveform(X, y, sub_id, loaded_subjects, time_s, channel_names, out_dir):
    n_ch  = len(channel_names)
    ncols = min(n_ch, 3)
    nrows = int(np.ceil(n_ch / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    sub_err  = _subject_averages(X, y, None, sub_id, loaded_subjects, "error")
    sub_corr = _subject_averages(X, y, None, sub_id, loaded_subjects, "correct")
    sub_diff = sub_err - sub_corr  # (n_subjects, n_ch, n_samples)
    t_ms     = _ms(time_s)

    for ci, ch_name in enumerate(channel_names):
        ax   = axes[ci // ncols][ci % ncols]
        mean = np.nanmean(sub_diff[:, ci, :], axis=0)
        sem  = np.nanstd(sub_diff[:, ci, :], axis=0) / np.sqrt(len(loaded_subjects))

        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
        ax.axvspan(ERN_TMIN_MS, ERN_TMAX_MS, alpha=0.10, color="blue")
        ax.axvspan(PE_TMIN_MS,  PE_TMAX_MS,  alpha=0.10, color="red")
        ax.fill_between(t_ms, mean - sem, mean + sem, alpha=0.25, color="purple")
        ax.plot(t_ms, mean, color="purple", linewidth=1.5)

        ax.set_title(ch_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Δ Amplitude (µV)")
        ax.set_xlim(-200, 900)

    for ci in range(n_ch, nrows * ncols):
        axes[ci // ncols][ci % ncols].set_visible(False)

    fig.suptitle(
        f"Difference waveform: Error − Correct  (N={len(loaded_subjects)} subjects)\n"
        f"ERN [{ERN_TMIN_MS}–{ERN_TMAX_MS} ms] blue, Pe [{PE_TMIN_MS}–{PE_TMAX_MS} ms] red",
        fontsize=12
    )
    fig.tight_layout()
    out = os.path.join(out_dir, "plot3_difference_waveform.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# =============================================================================
# Plot 4 — Per-subject ERP variability at a focal channel
# =============================================================================

def plot_subject_variability(X, y, sub_id, loaded_subjects, time_s,
                              channel_names, focal_channel, out_dir):
    if focal_channel not in channel_names:
        print(f"  [Plot 4] {focal_channel} not in selected channels — skipping.")
        return

    ch_pos = channel_names.index(focal_channel)
    t_ms   = _ms(time_s)
    cmap   = plt.cm.tab20

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
    ax.axvspan(PE_TMIN_MS, PE_TMAX_MS, alpha=0.08, color="red")

    sub_err = _subject_averages(X, y, None, sub_id, loaded_subjects, "error")

    for si, s in enumerate(loaded_subjects):
        waveform = sub_err[si, ch_pos, :]
        ax.plot(t_ms, waveform, color=cmap(si / max(len(loaded_subjects), 1)),
                linewidth=1.0, alpha=0.7, label=f"Sub {s}")

    grand_mean = np.nanmean(sub_err[:, ch_pos, :], axis=0)
    ax.plot(t_ms, grand_mean, color="black", linewidth=2.5, label="Grand mean")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_xlim(-200, 900)
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.set_title(
        f"Per-subject error ERP at {focal_channel}  (N={len(loaded_subjects)} subjects)\n"
        f"Pe region shaded [{PE_TMIN_MS}–{PE_TMAX_MS} ms]",
        fontsize=11
    )
    fig.tight_layout()
    out = os.path.join(out_dir, "plot4_subject_variability.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# =============================================================================
# Plot 5 — Pe amplitude by magnitude (bar chart)
# =============================================================================

def plot_pe_by_magnitude(X, y, mag, sub_id, loaded_subjects, time_s,
                          channel_names, focal_channel, out_dir):
    if focal_channel not in channel_names:
        print(f"  [Plot 5] {focal_channel} not in selected channels — skipping.")
        return

    ch_pos   = channel_names.index(focal_channel)
    pe_mask  = (time_s >= PE_TMIN_MS / 1000) & (time_s <= PE_TMAX_MS / 1000)
    magnitudes = [0, 3, 6, 9, 12]
    colours    = ["steelblue", "#ffc107", "#ff7f0e", "#e74c3c", "#8e44ad"]

    means, sems = [], []
    for m in magnitudes:
        sub_avg = _subject_averages(X, y, mag, sub_id, loaded_subjects, m)
        pe_amps = sub_avg[:, ch_pos, :][:, pe_mask].mean(axis=1)  # (n_subjects,)
        valid   = pe_amps[~np.isnan(pe_amps)]
        means.append(valid.mean() if len(valid) else np.nan)
        sems.append(valid.std() / np.sqrt(len(valid)) if len(valid) > 1 else 0)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(magnitudes))
    bars = ax.bar(x, means, yerr=sems, capsize=5, color=colours, edgecolor="black",
                  linewidth=0.8, error_kw={"linewidth": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}°" for m in magnitudes])
    ax.set_xlabel("Rotation magnitude")
    ax.set_ylabel("Mean Pe amplitude (µV)")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_title(
        f"Pe amplitude by rotation magnitude at {focal_channel}\n"
        f"Pe window: {PE_TMIN_MS}–{PE_TMAX_MS} ms  (N={len(loaded_subjects)} subjects, mean ± SEM)",
        fontsize=11
    )
    fig.tight_layout()
    out = os.path.join(out_dir, "plot5_pe_by_magnitude.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    # Print summary statistics.
    print(f"\n  Pe amplitude at {focal_channel} [{PE_TMIN_MS}–{PE_TMAX_MS} ms]:")
    for m, mn, se in zip(magnitudes, means, sems):
        print(f"    {m:2d}°: {mn:+.3f} ± {se:.3f} µV")


# =============================================================================
# Plot 6 — Single-trial heatmap at focal channel sorted by magnitude
# =============================================================================

def plot_single_trial_heatmap(X, y, mag, sub_id, loaded_subjects, time_s,
                               channel_names, focal_channel, out_dir,
                               max_trials_per_class=200):
    if focal_channel not in channel_names:
        print(f"  [Plot 6] {focal_channel} not in selected channels — skipping.")
        return

    ch_pos = channel_names.index(focal_channel)
    t_ms   = _ms(time_s)

    # Build sorted trial matrix: correct (0°) first, then error by ascending magnitude.
    rows, row_labels = [], []
    for m in [0, 3, 6, 9, 12]:
        mask = (mag == m)
        trials = X[mask, ch_pos, :]  # (n_trials_this_mag, n_samples)
        # Cap per-magnitude to keep the figure readable.
        if len(trials) > max_trials_per_class:
            rng = np.random.default_rng(seed=0)
            idx = rng.choice(len(trials), max_trials_per_class, replace=False)
            trials = trials[idx]
        rows.append(trials)
        row_labels.append((m, len(trials)))

    matrix = np.concatenate(rows, axis=0)  # (n_total, n_samples)

    # Colour scale clipped to 2 × std for readability.
    vmax = 2 * np.nanstd(matrix)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="upper",
        extent=[t_ms[0], t_ms[-1], matrix.shape[0], 0],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    plt.colorbar(im, ax=ax, label="Amplitude (µV)")

    # Draw magnitude boundary lines.
    cumulative = 0
    for m, n in row_labels:
        ax.axhline(cumulative, color="k", linewidth=0.8)
        ax.text(t_ms[0] + 5, cumulative + n / 2, f"{m}°",
                va="center", fontsize=8, color="black",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6))
        cumulative += n

    ax.axvline(0, color="white", linewidth=1.0, linestyle="--")
    ax.axvspan(PE_TMIN_MS, PE_TMAX_MS, alpha=0.15, color="yellow")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Trial")
    ax.set_title(
        f"Single-trial heatmap at {focal_channel}  (up to {max_trials_per_class} trials/magnitude)\n"
        f"Sorted by magnitude. Pe region [{PE_TMIN_MS}–{PE_TMAX_MS} ms] shaded.",
        fontsize=11
    )
    fig.tight_layout()
    out = os.path.join(out_dir, "plot6_heatmap.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# =============================================================================
# Summary statistics
# =============================================================================

def print_summary(X, y, mag, sub_id, loaded_subjects):
    print("\n── Dataset summary ──────────────────────────────────────────────")
    print(f"  Subjects loaded : {loaded_subjects}")
    print(f"  Total epochs    : {len(y)}")
    print(f"  Error epochs    : {(y == 1).sum()} ({100 * (y == 1).mean():.1f}%)")
    print(f"  Correct epochs  : {(y == 0).sum()} ({100 * (y == 0).mean():.1f}%)")
    print(f"  Channels        : {X.shape[1]}")
    print(f"  Samples/epoch   : {X.shape[2]}  ({X.shape[2] / FS * 1000:.0f} ms at {FS} Hz)")
    print()
    for m in [0, 3, 6, 9, 12]:
        n = (mag == m).sum()
        print(f"  Magnitude {m:2d}° : {n} epochs")
    print("─────────────────────────────────────────────────────────────────\n")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", required=True,
                   help="Path to combinedEpochs_v2.mat or eegEpochs_16subs_chanInterp_control.mat")
    p.add_argument("--subjects", default=None,
                   help="Comma-separated 1-based subject indices, e.g. '1,2,3'. Default: all.")
    p.add_argument("--channels", default=None,
                   help="Comma-separated channel names for multi-channel plots, "
                        "e.g. 'Fz,FCz,Cz,CP4,Pz,O2'. Default: key ErrP channels.")
    p.add_argument("--focal-channel", default="Cz",
                   help="Single channel for magnitude/variability/heatmap plots. Default: Cz.")
    p.add_argument("--harmony-channels-only", action="store_true",
                   help="Restrict to Harmony ErrP channels only: "
                        f"{HARMONY_ERRP_CHANNELS}")
    p.add_argument("--out", default="plots",
                   help="Output directory for PNG files. Default: ./plots/")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve subject list.
    subjects = None
    if args.subjects:
        subjects = [int(s.strip()) for s in args.subjects.split(",")]

    # Resolve channel list for multi-panel plots.
    if args.harmony_channels_only:
        display_channels = HARMONY_ERRP_CHANNELS
    elif args.channels:
        display_channels = [c.strip() for c in args.channels.split(",")]
    else:
        # Default: key ErrP channels covering frontocentral + centroparietal + occipital.
        display_channels = ["Fz", "FCz", "Cz", "CP4", "Pz", "O2"]

    # The union of display_channels and focal_channel must all be loaded.
    focal_channel = args.focal_channel
    load_channels = list(dict.fromkeys(display_channels +
                                       ([focal_channel] if focal_channel not in display_channels
                                        else [])))

    print(f"\nLoading: {args.file}")
    print(f"Channels to load: {load_channels}")

    X, y, mag, sub_id, time_s, loaded_subjects = load_liu_epochs(
        args.file, subjects=subjects, channel_names=load_channels
    )

    # Baseline-correct all epochs for visualization.
    X = baseline_correct(X, time_s)

    print_summary(X, y, mag, sub_id, loaded_subjects)

    os.makedirs(args.out, exist_ok=True)
    print(f"Saving plots to: {os.path.abspath(args.out)}\n")

    # Remap display_channels and focal_channel to indices in load_channels.
    # load_channels is the column order of X after loading.
    display_in_X = [c for c in display_channels if c in load_channels]

    print("Plot 1 — Grand-average ERP (error vs correct)")
    plot_grand_average(X, y, sub_id, loaded_subjects, time_s,
                       display_in_X, args.out)

    print("Plot 2 — Magnitude-stratified ERP")
    plot_magnitude_stratified(X, y, mag, sub_id, loaded_subjects, time_s,
                               load_channels, focal_channel, args.out)

    print("Plot 3 — Difference waveform (error − correct)")
    plot_difference_waveform(X, y, sub_id, loaded_subjects, time_s,
                              display_in_X, args.out)

    print("Plot 4 — Per-subject ERP variability")
    plot_subject_variability(X, y, sub_id, loaded_subjects, time_s,
                              load_channels, focal_channel, args.out)

    print("Plot 5 — Pe amplitude by magnitude")
    plot_pe_by_magnitude(X, y, mag, sub_id, loaded_subjects, time_s,
                          load_channels, focal_channel, args.out)

    print("Plot 6 — Single-trial heatmap")
    plot_single_trial_heatmap(X, y, mag, sub_id, loaded_subjects, time_s,
                               load_channels, focal_channel, args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()
