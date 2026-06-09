#!/usr/bin/env python3
"""Generate per-participant ERD scalp topographies (mu band), preserving each
participant's individual spatial pattern (no cross-participant averaging).

Separate from ``make_publication_figures.py`` because it RECOMPUTES TFRs via
the canonical topomap pipeline (`generate_plots_config_a.preprocess_and_tfr`
+ `_rejected_marker_avgs`), which is slow. For each participant it pools the
per-session AverageTFRs (weighted by trial count, the same
`_pool_avgs_weighted` the canonical grand-average uses) separately for motor
imagery (marker 200) and rest (marker 100), and stores the per-channel mu-band
logratio in eight successive 0.5 s windows (0--4 s post-cue).

Rendered in the canonical diagnostic style (viridis on raw logratio, the same
colormap and value space as ``generate_plots_config_a._plot_topo_strip``), so
the publication topos match the subject-wise diagnostic figures. Two deliberate
divergences from the diagnostic: a single fixed color scale shared across all
panels (the diagnostic rescales per subject) so participants are directly
comparable, and 1 s display columns (the cached 0.5 s windows binned in pairs)
to declutter the report figure.

Renders three products from that one cache:
  * ``fig_erd_topostrip_mi.png`` -- per-participant motor-imagery TIME-STRIP grid
    (participants P1--P7 down rows, four 1 s columns over 0--4 s): shows where and
    when each participant desynchronizes. This is the main-text representation.
  * ``fig_erd_topostrip_rest.png`` -- the same grid for rest (supplementary).
  * ``fig_erd_topomap_grid.png`` -- per-participant single post-cue (1--4 s) map
    (P1--P7 x {MI, Rest}). A compact supplementary alternative.

CLIN_SUBJ_002 (P1) is on a reduced 13-channel montage (excluded from the cohort
confusion / EDS analyses); its maps render from those channels.

The per-window logratio is cached to CSV so the figures can be re-rendered (e.g.
to retune the color scale or layout) WITHOUT the ~15-min TFR recompute. Delete
the cache to force a fresh recompute.

Run (Windows, slow ~10-20 min first time, instant from cache):
    PYTHONUTF8=1 python -u exploration/clinical_analysis/make_erd_topomap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT),
           str(_REPO_ROOT / "exploration" / "preprocessing_sweep")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import generate_plots_config_a as g  # noqa: E402

OUT = Path(r"C:\Users\arman\Documents\harmony-als-preprint\figures")
# Candidate renders go to scratch first (per the clin_analysis_<tag> rule); the
# chosen layout is promoted to OUT once approved.
SCRATCH = Path.home() / "Pictures" / "clin_analysis_pub_topomap"
CACHE = SCRATCH / "topomap_windows_logr.csv"

SUBJECTS = [f"CLIN_SUBJ_{i:03d}" for i in range(2, 9)]   # 002..008 -> P1..P7
SUBJ_LABEL = {s: f"P{i + 1}" for i, s in enumerate(SUBJECTS)}
MARKERS = (("200", "Motor imagery"), ("100", "Rest"))

# Cache/compute granularity: eight 0.5 s windows, 0--4 s post-cue (matches the
# diagnostic strip). The cache is stored at this fine resolution so it stays
# reusable; report figures bin it down for display (PLOT_WINDOWS below).
WINDOWS = [(i * 0.5, i * 0.5 + 0.5) for i in range(8)]
# Report-figure display bins: four 1 s windows (user 2026-06-05, to halve the
# column count and declutter). Each is the mean of the two component 0.5 s cache
# windows -- a log-space mean, consistent with the pipeline's averaging
# convention. The diagnostic figures keep the native 0.5 s windows.
PLOT_WINDOWS = [(float(i), float(i + 1)) for i in range(4)]
POST_WIN = (1.0, 4.0)                       # single-map summary window
CMAP = "viridis"                            # canonical diagnostic colormap
                                            # (generate_plots_config_a.py:90)


def _subject_marker_windows(subject):
    """Pool a participant's sessions per marker; return, per marker,
    (ch_names, logr[n_win, n_ch]) -- per-channel mu logratio in each 0.5 s
    window. None if the participant has no usable trials for that marker."""
    per_marker = {m: [] for m, _ in MARKERS}
    for sess in g.enumerate_online_sessions(subject):
        try:
            out = g.preprocess_and_tfr(subject, sess, g.CONFIG_A)
        except Exception as e:
            print(f"  {subject} {sess}: FAILED ({type(e).__name__}: {e})")
            continue
        rej = g._rejected_marker_avgs(out["tfr_trials"])
        for m, _ in MARKERS:
            if m in rej:
                per_marker[m].append(rej[m])      # (AverageTFR, n_kept)
        print(f"  {subject} {sess}: ok")

    result = {}
    for m, _ in MARKERS:
        pooled = g._pool_avgs_weighted(per_marker[m])
        if pooled is None:
            result[m] = (None, None)
            continue
        fmask = (pooled.freqs >= g.MU_LO) & (pooled.freqs <= g.MU_HI)
        band = pooled.data[:, fmask]                      # (n_ch, n_f, n_t)
        logr = np.empty((len(WINDOWS), band.shape[0]))
        for wi, (lo, hi) in enumerate(WINDOWS):
            tmask = (pooled.times >= lo) & (pooled.times < hi)
            logr[wi] = band[:, :, tmask].mean(axis=(1, 2))
        result[m] = (list(pooled.ch_names), logr)
        print(f"    {subject} {m}: pooled {pooled.nave} trials, "
              f"{len(pooled.ch_names)} ch")
    return result


def _load_or_compute():
    """Return {subject: {marker: (ch_names, logr[n_win, n_ch])}} from the CSV
    cache when present, else recompute the TFRs and write the cache."""
    if CACHE.exists():
        print(f"Loading cached per-window logratio from {CACHE}")
        df = pd.read_csv(CACHE)
        maps = {}
        for s in SUBJECTS:
            maps[s] = {}
            for m, _ in MARKERS:
                d = df[(df["subject"] == s) & (df["marker"] == int(m))]
                if d.empty:
                    maps[s][m] = (None, None)
                    continue
                chans = list(dict.fromkeys(d["channel"]))   # preserve order
                piv = d.pivot(index="win_idx", columns="channel",
                              values="logr")[chans]
                maps[s][m] = (chans, piv.to_numpy())
        return maps

    maps = {s: _subject_marker_windows(s) for s in SUBJECTS}
    rows = []
    for s in SUBJECTS:
        for m, _ in MARKERS:
            chans, logr = maps[s][m]
            if logr is None:
                continue
            for wi, (lo, hi) in enumerate(WINDOWS):
                for ci, c in enumerate(chans):
                    rows.append({"subject": s, "marker": int(m), "win_idx": wi,
                                 "win_lo": lo, "win_hi": hi, "channel": c,
                                 "logr": logr[wi, ci]})
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(CACHE, index=False)
    print(f"\nWrote per-window logratio cache to {CACHE}")
    return maps


def _bin_to_plot(maps):
    """Bin the cached 0.5 s windows down to the 1 s PLOT_WINDOWS for the report
    figures by averaging adjacent pairs (log-space mean). Returns a new maps dict
    with logr arrays of shape (len(PLOT_WINDOWS), n_ch); the cache itself is
    untouched."""
    factor = len(WINDOWS) // len(PLOT_WINDOWS)
    out = {}
    for s in SUBJECTS:
        out[s] = {}
        for m, _ in MARKERS:
            ch_names, logr = maps[s][m]
            if logr is None:
                out[s][m] = (ch_names, None)
                continue
            binned = logr.reshape(len(PLOT_WINDOWS), factor, logr.shape[1]
                                  ).mean(axis=1)
            out[s][m] = (ch_names, binned)
    return out


def _grid_vlim(maps):
    """Single fixed color scale shared by every panel: the [2, 98] percentile of
    all participants' mu logratio (both classes, all windows). Mirrors the
    diagnostic's `_compute_dynamic_vlim` percentile (generate_plots_config_a.py:
    182) but pools across the whole grid so participants are directly comparable
    instead of each being rescaled to its own range."""
    vals = []
    for s in SUBJECTS:
        for m, _ in MARKERS:
            _, logr = maps[s][m]
            if logr is not None:
                vals.append(np.asarray(logr, float).ravel())
    if not vals:
        return (-0.3, 0.3)
    lo, hi = np.percentile(np.concatenate(vals), (2, 98))
    return (float(lo), float(hi))


def _topo(ax, logr, ch_names, vlim):
    """One mu-band logratio topomap in the canonical diagnostic style: viridis
    on raw logratio (same colormap/value space as
    `generate_plots_config_a._plot_topo_strip`)."""
    info = mne.create_info(ch_names, sfreq=512.0, ch_types="eeg")
    info.set_montage("standard_1020", match_case=False)
    im, _ = mne.viz.plot_topomap(logr, info, axes=ax, show=False, cmap=CMAP,
                                 vlim=vlim, contours=4, sensors=False)
    return im


def render_strip(maps, marker, mname, path, vlim):
    """Per-participant time-strip grid: rows = participants, cols = the 1 s
    PLOT_WINDOWS over 0--4 s, cells = that window's mu logratio topomap."""
    n = len(SUBJECTS)
    nw = len(PLOT_WINDOWS)
    fig, axes = plt.subplots(n, nw, figsize=(1.6 * nw, 1.15 * n))
    im = None
    for r, subject in enumerate(SUBJECTS):
        ch_names, logr = maps[subject][marker]
        for wi, (lo, hi) in enumerate(PLOT_WINDOWS):
            ax = axes[r, wi]
            if logr is None:
                ax.axis("off")
                continue
            im = _topo(ax, logr[wi], ch_names, vlim)
            if r == 0:
                ax.set_title(f"{lo:.0f}-{hi:.0f}s", fontsize=10, pad=4)
        axes[r, 0].set_ylabel(SUBJ_LABEL[subject], fontsize=12, rotation=0,
                              ha="right", va="center", labelpad=14)
    fig.suptitle(f"Per-participant mu-ERD topography over time -- {mname}",
                 fontsize=13, y=0.99)
    fig.subplots_adjust(left=0.06, right=0.90, top=0.93, bottom=0.02,
                        wspace=0.05, hspace=0.10)
    cax = fig.add_axes([0.92, 0.30, 0.012, 0.40])
    fig.colorbar(im, cax=cax).set_label("ERD/ERS (logratio)", fontsize=10)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def render_single_grid(maps, path, vlim):
    """Per-participant single post-cue (1--4 s) map, P1--P7 x {MI, Rest}."""
    win_sel = [i for i, (lo, hi) in enumerate(PLOT_WINDOWS) if lo >= POST_WIN[0]]
    n = len(SUBJECTS)
    fig, axes = plt.subplots(n, 2, figsize=(5.2, 2.35 * n))
    im = None
    for r, subject in enumerate(SUBJECTS):
        for c, (m, mname) in enumerate(MARKERS):
            ax = axes[r, c]
            ch_names, logr = maps[subject][m]
            if logr is None:
                ax.axis("off")
                continue
            im = _topo(ax, logr[win_sel].mean(axis=0), ch_names, vlim)
            if r == 0:
                ax.set_title(mname, fontsize=13, pad=8)
        axes[r, 0].set_ylabel(SUBJ_LABEL[subject], fontsize=13, rotation=0,
                              ha="right", va="center", labelpad=18)
    fig.subplots_adjust(left=0.10, right=0.86, top=0.97, bottom=0.03,
                        wspace=0.05, hspace=0.12)
    cax = fig.add_axes([0.89, 0.30, 0.025, 0.40])
    fig.colorbar(im, cax=cax).set_label("Post-cue ERD/ERS (logratio)",
                                        fontsize=11)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    maps = _bin_to_plot(_load_or_compute())     # 0.5 s cache -> 1 s display bins
    vlim = _grid_vlim(maps)
    print(f"Shared viridis logratio scale: [{vlim[0]:.3f}, {vlim[1]:.3f}]")
    OUT.mkdir(parents=True, exist_ok=True)
    # Main-text substrate topography = per-participant MI time-strip. The REST
    # strip and the compact single-window grid are supplementary.
    render_strip(maps, "200", "Motor imagery", OUT / "fig_erd_topostrip_mi.png",
                 vlim)
    render_strip(maps, "100", "Rest", OUT / "fig_erd_topostrip_rest.png", vlim)
    render_single_grid(maps, OUT / "fig_erd_topomap_grid.png", vlim)


if __name__ == "__main__":
    main()
