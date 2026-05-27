#!/usr/bin/env python3
# tools/quick_topo.py
"""
Quick side-by-side topographic comparison of two (or more) XDF files.

Loads each XDF's EEG stream, applies the standard 10-20 montage (same
renaming pattern as Visualize_offline_data_MNE.py:111-115), and plots
a topomap of mean log-power in the mu band (8-13 Hz) and beta band
(13-30 Hz) for each input file.

Use case: visually compare a fresh recording against an older one to
verify the sensor placement / noise profile didn't change.

Usage:
    python tools/quick_topo.py FILE.xdf [FILE2.xdf ...] \
        --out /tmp/topo.png

Lives under tools/ — pure visualization, not Tier 1/2.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")  # no DISPLAY required for headless runs
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import welch  # noqa: E402

from Utils.stream_utils import get_channel_names_from_xdf, load_xdf  # noqa: E402


FS_DEFAULT = 512.0

# Same rename + drop conventions as Visualize_offline_data_MNE.py
# so the montage attaches cleanly to the eegoSports channel set.
RENAME_DICT = {
    "FP1": "Fp1", "FPZ": "Fpz", "FP2": "Fp2",
    "FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "POZ": "POz", "OZ": "Oz",
}
NON_EEG_CHANNELS = {"AUX1", "AUX2", "AUX3", "AUX7", "AUX8", "AUX9", "TRIGGER"}
DROP_MASTOIDS = {"M1", "M2"}
DROP_TEMPORALS = {"T7", "T8"}  # Visualize_offline drops these too

BANDS = [
    ("mu (8-13 Hz)", 8.0, 13.0),
    ("beta (13-30 Hz)", 13.0, 30.0),
]


def _build_raw(xdf_path: Path, fs: float) -> "mne.io.RawArray":
    """Mirror the EEG-loading pipeline from Visualize_offline_data_MNE.py
    so the topomap projection is identical to what that script would produce."""
    eeg_stream, _markers = load_xdf(str(xdf_path), report=False)
    channel_names = get_channel_names_from_xdf(eeg_stream)
    eeg_data = np.asarray(eeg_stream["time_series"]).T  # (n_ch, n_samp)

    valid = [ch for ch in channel_names if ch not in NON_EEG_CHANNELS]
    idx = [channel_names.index(ch) for ch in valid]
    eeg_data = eeg_data[idx, :]

    info = mne.create_info(ch_names=valid, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(eeg_data, info, verbose="ERROR")
    for ch in raw.info["chs"]:
        ch["unit"] = 201  # microvolts in MNE FIFF code

    # Drop hardware mastoids + the temporals Visualize_offline drops.
    to_drop = [c for c in (DROP_MASTOIDS | DROP_TEMPORALS) if c in raw.ch_names]
    if to_drop:
        raw.drop_channels(to_drop)

    raw.rename_channels(RENAME_DICT)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=True, on_missing="warn", verbose="ERROR")
    return raw


def _band_power(raw: "mne.io.RawArray", lo: float, hi: float) -> np.ndarray:
    """Per-channel log10 mean PSD in [lo, hi] Hz via Welch.
    Returns 1-D array aligned with raw.ch_names."""
    data = raw.get_data()  # (n_ch, n_samp); MNE keeps the input units
    fs = float(raw.info["sfreq"])
    nperseg = int(min(data.shape[1], fs * 2))  # 2 s segments
    freqs, psd = welch(data, fs=fs, nperseg=nperseg, axis=1)
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return np.full(data.shape[0], np.nan)
    band_mean = psd[:, mask].mean(axis=1)
    # log10 power for a perceptually flatter colormap; +eps to guard zeros.
    return np.log10(band_mean + 1e-12)


def _list_streams(xdf_path: Path) -> str:
    """Compact one-liner per stream for the report header."""
    import pyxdf
    streams, _ = pyxdf.load_xdf(
        str(xdf_path), dejitter_timestamps=False, synchronize_clocks=False,
        verbose=False,
    )
    parts = []
    for s in streams:
        info = s["info"]
        name = info["name"][0] if info.get("name") else "?"
        stype = info["type"][0] if info.get("type") else "?"
        chans = int(info["channel_count"][0]) if info.get("channel_count") else 0
        n = len(s.get("time_stamps", []))
        parts.append(f"{name}[{stype}, {chans}ch, {n}smp]")
    return " | ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "xdf", nargs="+",
        help="One or more XDF files to compare (each becomes a row in the figure).",
    )
    parser.add_argument(
        "--out", type=str, default="/tmp/quick_topo.png",
        help="Output PNG path (default: /tmp/quick_topo.png).",
    )
    parser.add_argument(
        "--fs", type=float, default=FS_DEFAULT,
        help=f"EEG sampling rate (default: {FS_DEFAULT} Hz).",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Optional row labels (one per XDF). Defaults to filename stems.",
    )
    args = parser.parse_args()

    paths = [Path(p).expanduser().resolve() for p in args.xdf]
    for p in paths:
        if not p.is_file():
            print(f"ERROR: {p} not found", file=sys.stderr)
            return 1
    labels = args.labels or [p.stem for p in paths]
    if len(labels) != len(paths):
        print(
            f"ERROR: {len(labels)} labels for {len(paths)} XDFs",
            file=sys.stderr,
        )
        return 1

    # Stream-list dump up front so the user can verify what's in each file
    # (especially the NeonGaze stream presence/absence).
    print("=" * 78)
    for label, p in zip(labels, paths):
        print(f"[{label}] {p.name}")
        print(f"   streams: {_list_streams(p)}")
    print("=" * 78)

    n_rows = len(paths)
    n_cols = len(BANDS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 4.0 * n_rows), squeeze=False,
    )

    for r, (label, p) in enumerate(zip(labels, paths)):
        print(f"[{label}] loading + montaging...")
        raw = _build_raw(p, args.fs)
        print(f"[{label}] {len(raw.ch_names)} EEG channels: {raw.ch_names}")
        for c, (band_label, lo, hi) in enumerate(BANDS):
            power = _band_power(raw, lo, hi)
            ax = axes[r][c]
            im, _ = mne.viz.plot_topomap(
                power, raw.info, axes=ax, show=False, cmap="RdBu_r",
                contours=4, sensors=True,
            )
            ax.set_title(f"{label}\n{band_label}", fontsize=10)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("log10 PSD (µV²/Hz)", fontsize=8)

    fig.tight_layout()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
