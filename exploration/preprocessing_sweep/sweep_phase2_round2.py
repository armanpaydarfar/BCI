#!/usr/bin/env python3
"""
Phase 2 — Round 2 preprocessing sweep.

Grid (Option B from Round 1):
  - filter_type:       iir (fixed; Round 1 winner)
  - blink_removal:     {none, fp_regression, drop_fp, ica_blink_1hz}
  - spatial_filter:    {car, csd, hjorth}
  - spectral_baseline: {(-1.0, 0.0), (-1.5, -0.25), (-0.75, -0.1)}   # REAL sweep via tfr.apply_baseline
  - baseline_mode:     {logratio, zscore}

= 4 × 3 × 3 × 2 = 72 configs × 8 sessions = 576 runs.

Sessions (last 2 per subject):
  CLIN_SUBJ_005  → S004ONLINE, S005ONLINE
  CLIN_SUBJ_006  → S004ONLINE, S005ONLINE
  CLIN_SUBJ_007  → S003ONLINE, S004ONLINE
  CLIN_SUBJ_008  → S001ONLINE, S002ONLINE

Key differences from Round 1:
  * Spectral baseline is now a real knob (passed through to tfr.apply_baseline).
  * New blink method 'drop_fp': drop Fp1/Fp2/Fpz channels entirely.
  * New blink method 'ica_blink_1hz': fit ICA on a 1 Hz-highpass copy of the raw
    (so blink components survive to be detected), then apply the unmixing matrix
    to the 4–40 Hz broadband data.
  * Multi-subject scoring: per-config median / min across the 4 subjects (not
    the 8 sessions), so one chatty subject can't dominate.
  * Ranking is produced within each baseline_mode separately (zscore and logratio
    have incommensurate magnitude units).
"""

import os
import csv
import time
import warnings
from itertools import product

import numpy as np
import mne

from Utils.stream_utils import load_xdf, get_channel_names_from_xdf
from Utils.preprocessing import concatenate_streams
from config import DATA_DIR

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")


# ======================================================================
# Run config
# ======================================================================

OUT_CSV   = "/home/arman-admin/Documents/SoftwareDocs/clin_erd_phase2_round2.csv"
OUT_MD    = "/home/arman-admin/Documents/SoftwareDocs/clin_erd_phase2_round2.md"

SESSION_MAP = {
    "CLIN_SUBJ_005": ["S004ONLINE", "S005ONLINE"],
    "CLIN_SUBJ_006": ["S004ONLINE", "S005ONLINE"],
    "CLIN_SUBJ_007": ["S003ONLINE", "S004ONLINE"],
    "CLIN_SUBJ_008": ["S001ONLINE", "S002ONLINE"],
}

FS           = 512
NOTCH        = 60
BB_LO, BB_HI = 4.0, 40.0
MU_LO, MU_HI = 8.0, 13.0
PAD_TFR      = 1.0
TRIAL_WIN    = (-1.0, 4.0)
SCALAR_WIN   = (1.0, 4.0)
REJECT_MAX_ABS_UV = 50.0
FREQS        = np.linspace(8, 30, 23)
N_CYCLES     = FREQS / 2.0
ICA_HP_HZ    = 1.0  # for ica_blink_1hz

ZONES = {
    "L-motor":   ["C3", "CP1", "CP5", "FC1", "FC5"],
    "R-motor":   ["C4", "CP2", "CP6", "FC2", "FC6"],
    "Midline":   ["Cz", "FCz", "CPz", "Fz", "Pz", "POz"],
    "Frontal":   ["Fp1", "Fp2", "Fpz", "F3", "F4", "F7", "F8",
                  "AF3", "AF4", "AF7", "AF8"],
    "Parietal":  ["P3", "P4", "P7", "P8", "P1", "P2"],
    "Occipital": ["O1", "O2", "Oz"],
    "Temporal":  ["T7", "T8", "TP7", "TP8", "FT7", "FT8"],
}
LAT_PAIRS_MAIN = [("C3", "C4"), ("CP1", "CP2"), ("FC1", "FC2")]


# ======================================================================
# Raw loading — cached per session
# ======================================================================

def load_raw_cached(subject, session):
    xdf_dir = os.path.join(DATA_DIR, f"sub-{subject}", f"ses-{session}", "eeg/")
    # Only accept canonical "..._eeg.xdf" — exclude retries / repairs like
    # "..._eeg_old1.xdf" or "..._eeg_repaired.xdf", which may have empty marker
    # streams and crash concatenate_streams.
    all_xdf = [f for f in os.listdir(xdf_dir) if f.endswith(".xdf")]
    xdf_files = sorted([os.path.join(xdf_dir, f) for f in all_xdf
                        if os.path.splitext(f)[0].endswith("_eeg")])
    excluded = sorted(set(all_xdf) - {os.path.basename(p) for p in xdf_files})
    if excluded:
        print(f"  (excluded non-canonical XDFs in {subject}/{session}: {excluded})")
    if not xdf_files:
        raise FileNotFoundError(f"No canonical XDF in {xdf_dir} (all={all_xdf})")

    eeg_streams, marker_streams = [], []
    for f in xdf_files:
        es, ms = load_xdf(f)
        eeg_streams.append(es)
        marker_streams.append(ms)

    if len(eeg_streams) == 1:
        eeg_stream, marker_stream = eeg_streams[0], marker_streams[0]
    else:
        eeg_stream, marker_stream = concatenate_streams(eeg_streams, marker_streams)

    eeg_timestamps    = np.array(eeg_stream["time_stamps"])
    eeg_data_raw      = np.array(eeg_stream["time_series"]).T
    channel_names     = get_channel_names_from_xdf(eeg_stream)
    marker_data       = np.array([int(v[0]) for v in marker_stream["time_series"]])
    marker_timestamps = np.array([float(v[1]) for v in marker_stream["time_series"]])

    non_eeg = {"AUX1","AUX2","AUX3","AUX7","AUX8","AUX9","TRIGGER"}
    valid_idx = [i for i, ch in enumerate(channel_names) if ch not in non_eeg]
    eeg_data = eeg_data_raw[valid_idx, :]
    valid_ch = [channel_names[i] for i in valid_idx]

    info = mne.create_info(valid_ch, FS, "eeg")
    raw = mne.io.RawArray(eeg_data, info, verbose=False)
    if "M1" in raw.ch_names and "M2" in raw.ch_names:
        raw.drop_channels(["M1", "M2"])
    rename = {"FP1":"Fp1","FPZ":"Fpz","FP2":"Fp2","FZ":"Fz","CZ":"Cz",
              "PZ":"Pz","POZ":"POz","OZ":"Oz"}
    raw.rename_channels(rename)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                    match_case=True, on_missing="warn")

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
    events = np.column_stack((samples, np.zeros(len(mdata), dtype=int), mdata))

    return raw, events, event_dict


# ======================================================================
# Blink removal
# ======================================================================

def _fp_regression(raw):
    fp_chans = [c for c in ("Fp1", "Fp2", "Fpz") if c in raw.ch_names]
    if not fp_chans:
        return raw, "fp_regression:no_fp"
    fp_idx = [raw.ch_names.index(c) for c in fp_chans]
    all_data = raw.get_data()
    X = all_data[fp_idx].T
    Xi = np.hstack([X, np.ones((X.shape[0], 1))])
    out = all_data.copy()
    for ch_idx, ch in enumerate(raw.ch_names):
        if ch in fp_chans:
            continue
        y = all_data[ch_idx]
        coefs, *_ = np.linalg.lstsq(Xi, y, rcond=None)
        out[ch_idx] = y - Xi @ coefs
    raw._data[:] = out
    return raw, f"fp_regression:{','.join(fp_chans)}"


def _drop_fp(raw):
    fp_chans = [c for c in ("Fp1", "Fp2", "Fpz") if c in raw.ch_names]
    if not fp_chans:
        return raw, "drop_fp:no_fp"
    raw.drop_channels(fp_chans)
    return raw, f"drop_fp:{','.join(fp_chans)}"


def _ica_blink_1hz(raw_bb, raw_1hz):
    """Fit ICA on a 1 Hz-highpassed copy (so blinks survive), find blink components
    using Fp1/Fp2 as surrogate EOG, then apply exclusion to the broadband raw."""
    ica = mne.preprocessing.ICA(
        n_components=0.99, method="picard",
        random_state=42, max_iter=200, verbose=False,
    )
    ica.fit(raw_1hz, verbose=False)
    bads = []
    for ch in ("Fp1", "Fp2"):
        if ch in raw_1hz.ch_names:
            try:
                b, _ = ica.find_bads_eog(raw_1hz, ch_name=ch, verbose=False)
                bads.extend(b)
            except Exception:
                pass
    bads = sorted(set(bads))
    info_str = f"ica_blink_1hz:removed={len(bads)}_of_{ica.n_components_}"
    if bads:
        ica.exclude = bads
        raw_new = ica.apply(raw_bb.copy(), verbose=False)
        raw_bb._data[:] = raw_new.get_data()
    return raw_bb, info_str


def apply_blink_removal(raw_bb, raw_1hz, method):
    """Apply the named blink-removal method to raw_bb. raw_1hz is the 1 Hz-HP
    copy (only used by ica_blink_1hz)."""
    if method == "none":
        return raw_bb, "none"
    if method == "fp_regression":
        return _fp_regression(raw_bb)
    if method == "drop_fp":
        return _drop_fp(raw_bb)
    if method == "ica_blink_1hz":
        return _ica_blink_1hz(raw_bb, raw_1hz)
    raise ValueError(f"Unknown blink_removal: {method}")


# ======================================================================
# Spatial filter
# ======================================================================

def _apply_hjorth(epochs, k=4):
    pos = np.array([ch["loc"][:3] for ch in epochs.info["chs"]])
    if not np.any(pos):
        raise RuntimeError("Hjorth requires channel positions.")
    diff = pos[:, None, :] - pos[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dists, np.inf)
    k = min(k, len(epochs.ch_names) - 1)
    nbr_idx = np.argsort(dists, axis=1)[:, :k]
    nbr_means = epochs._data[:, nbr_idx, :].mean(axis=2)
    out = epochs.copy()
    out._data -= nbr_means
    return out


def apply_spatial_filter(epochs, method):
    if method == "car":
        epochs.set_eeg_reference("average", projection=False, verbose=False)
        return epochs
    if method == "csd":
        return mne.preprocessing.compute_current_source_density(epochs, verbose=False)
    if method == "hjorth":
        return _apply_hjorth(epochs, k=4)
    raise ValueError(f"Unknown spatial_filter: {method}")


# ======================================================================
# Single config run
# ======================================================================

def run_config(raw_cached, config):
    raw_bb  = raw_cached["raw"].copy()     # will become broadband-filtered
    raw_1hz = raw_cached["raw"].copy()     # will become 1 Hz HP for ICA
    events  = raw_cached["events"]
    event_dict = raw_cached["event_dict"]

    # 1) Notch + broadband bandpass on raw_bb
    raw_bb.notch_filter(NOTCH, method="iir", verbose=False)
    raw_bb.filter(l_freq=BB_LO, h_freq=BB_HI, method="iir", verbose=False)

    # 1b) 1 Hz highpass on raw_1hz (only used if ICA is selected)
    if config["blink_removal"] == "ica_blink_1hz":
        raw_1hz.notch_filter(NOTCH, method="iir", verbose=False)
        raw_1hz.filter(l_freq=ICA_HP_HZ, h_freq=BB_HI, method="iir", verbose=False)

    # 2) Blink removal
    raw_bb, blink_info = apply_blink_removal(raw_bb, raw_1hz, config["blink_removal"])

    # 3) Mu-band copy for QC
    raw_mu = raw_bb.copy()
    raw_mu.filter(l_freq=MU_LO, h_freq=MU_HI, method="iir", verbose=False)

    # 4) Epoch
    t0, t1 = TRIAL_WIN
    epoch_kw = dict(
        event_id=event_dict,
        tmin=t0 - PAD_TFR, tmax=t1 + PAD_TFR,
        baseline=None,    # let tfr.apply_baseline handle it
        detrend=1, preload=True, verbose=False,
    )
    epochs_mu = mne.Epochs(raw_mu, events, reject=None, flat=None, **epoch_kw)
    epochs_bb = mne.Epochs(raw_bb, events, reject=None, flat=None, **epoch_kw)

    # 5) max_abs rejection on mu copy
    mu_data = epochs_mu.get_data()
    mask = np.max(np.abs(mu_data), axis=(1, 2)) <= REJECT_MAX_ABS_UV
    good_ix = np.where(mask)[0].tolist()
    epochs = epochs_bb[good_ix]
    n_attempted = int(len(events))
    n_kept = int(len(epochs))

    # 6) Spatial filter
    epochs = apply_spatial_filter(epochs, config["spatial_filter"])

    # 7) TFR per marker with configurable spectral baseline
    per_ch = {}
    spec_bl = config["spectral_baseline"]
    mode = config["baseline_mode"]
    for marker in ("100", "200"):
        if marker not in epochs.event_id or len(epochs[marker]) == 0:
            per_ch[marker] = {}
            continue
        tfr = epochs[marker].compute_tfr(
            method="multitaper", freqs=FREQS, n_cycles=N_CYCLES,
            tmin=t0 - PAD_TFR, tmax=t1 + PAD_TFR,
            use_fft=True, return_itc=False, average=False, verbose=False,
        )
        tfr.apply_baseline(baseline=spec_bl, mode=mode, verbose=False)
        tfr.crop(tmin=t0, tmax=t1)

        fmask = (tfr.freqs >= MU_LO) & (tfr.freqs <= MU_HI)
        tmask = (tfr.times >= SCALAR_WIN[0]) & (tfr.times <= SCALAR_WIN[1])
        vals = tfr.data[:, :, fmask, :][:, :, :, tmask].mean(axis=(0, 2, 3))
        per_ch[marker] = dict(zip(tfr.ch_names, vals))

    mi = per_ch.get("200", {})
    rest = per_ch.get("100", {})

    # 8) Scoring (native units; interpretation per mode)
    lis = []
    for c_ch, i_ch in LAT_PAIRS_MAIN:
        if c_ch in mi and i_ch in mi:
            c_v, i_v = mi[c_ch], mi[i_ch]
            lis.append((i_v - c_v) / (abs(c_v) + abs(i_v) + 1e-9))
    li_mean = float(np.mean(lis)) if lis else float("nan")

    l_chs = ZONES["L-motor"]; r_chs = ZONES["R-motor"]
    l_mag = float(-np.mean([mi[c] for c in l_chs if c in mi])) if any(c in mi for c in l_chs) else float("nan")
    r_mag = float(-np.mean([mi[c] for c in r_chs if c in mi])) if any(c in mi for c in r_chs) else float("nan")

    if mi:
        peak_ch = min(mi, key=lambda k: mi[k])
        peak_v  = mi[peak_ch]
        peak_zone = next((z for z, chs in ZONES.items() if peak_ch in chs), "Other")
    else:
        peak_ch, peak_v, peak_zone = "", float("nan"), ""

    rest_l_mag = float(np.mean([abs(rest[c]) for c in l_chs if c in rest])) \
                 if any(c in rest for c in l_chs) else float("nan")
    rest_frontal_mag = float(np.mean([abs(rest[c]) for c in ZONES["Frontal"] if c in rest])) \
                 if any(c in rest for c in ZONES["Frontal"]) else float("nan")

    c3_contrast = float(rest.get("C3", 0.0) - mi.get("C3", 0.0))

    return {
        "n_kept": n_kept,
        "n_attempted": n_attempted,
        "keep_frac": n_kept / n_attempted if n_attempted else 0.0,
        "blink_info": blink_info,
        "n_channels_after_preproc": len(epochs.ch_names),
        "li_mean": li_mean,
        "l_motor_mag": l_mag,
        "r_motor_mag": r_mag,
        "l_minus_r_mag": l_mag - r_mag if np.isfinite(l_mag) and np.isfinite(r_mag) else float("nan"),
        "peak_ch": peak_ch,
        "peak_zone": peak_zone,
        "peak_value": peak_v,
        "rest_l_mag": rest_l_mag,
        "rest_frontal_mag": rest_frontal_mag,
        "c3_contrast": c3_contrast,
        "c3_mi": mi.get("C3", float("nan")),
        "c3_rest": rest.get("C3", float("nan")),
        "fc1_mi": mi.get("FC1", float("nan")),
        "cp1_mi": mi.get("CP1", float("nan")),
    }


# ======================================================================
# Grid + main
# ======================================================================

CONFIG_GRID = {
    "blink_removal":     ["none", "fp_regression", "drop_fp", "ica_blink_1hz"],
    "spatial_filter":    ["car", "csd", "hjorth"],
    "spectral_baseline": [(-1.0, 0.0), (-1.5, -0.25), (-0.75, -0.1)],
    "baseline_mode":     ["logratio", "zscore"],
}


def rank_and_write_md(rows, configs):
    """Rank configs by per-subject median and minimum; write markdown report.
    Ranking is done separately within each baseline_mode (units incommensurate)."""
    # group ok rows by (config_idx, subject)
    by_cfg_subj = {}  # (ci, subj) -> list of per-session metrics
    for r in rows:
        if r.get("status") != "ok":
            continue
        key = (r["config_idx"], r["subject"])
        by_cfg_subj.setdefault(key, []).append(r)

    # subject-level: average the 2 sessions per subject
    by_cfg = {}  # ci -> {subj: subj_summary}
    for (ci, subj), rs in by_cfg_subj.items():
        subj_summary = {
            k: float(np.mean([r[k] for r in rs])) for k in (
                "li_mean", "l_minus_r_mag", "l_motor_mag", "r_motor_mag",
                "rest_l_mag", "rest_frontal_mag", "c3_contrast", "keep_frac",
            )
        }
        subj_summary["n_peak_lmotor_sessions"] = sum(1 for r in rs if r["peak_zone"] == "L-motor")
        subj_summary["n_sessions"] = len(rs)
        by_cfg.setdefault(ci, {})[subj] = subj_summary

    # composite per config: median and min across subjects
    cfg_summary = []
    for ci, subj_map in by_cfg.items():
        subjs = list(subj_map.keys())
        if len(subjs) != len(SESSION_MAP):
            continue
        li_vals = [subj_map[s]["li_mean"] for s in subjs]
        lmr = [subj_map[s]["l_minus_r_mag"] for s in subjs]
        restl = [subj_map[s]["rest_l_mag"] for s in subjs]
        lmag = [subj_map[s]["l_motor_mag"] for s in subjs]
        keepf = [subj_map[s]["keep_frac"] for s in subjs]
        peakL_total = sum(subj_map[s]["n_peak_lmotor_sessions"] for s in subjs)
        cfg_summary.append({
            "config_idx": ci,
            "cfg": configs[ci],
            "li_median": float(np.median(li_vals)),
            "li_min":    float(np.min(li_vals)),
            "li_mean":   float(np.mean(li_vals)),
            "lmr_median": float(np.median(lmr)),
            "lmr_min":    float(np.min(lmr)),
            "rest_l_median": float(np.median(restl)),
            "l_motor_mag_median": float(np.median(lmag)),
            "n_peak_lmotor_total_of_8": peakL_total,
            "keep_frac_mean": float(np.mean(keepf)),
            "per_subj_li": {s: subj_map[s]["li_mean"] for s in subjs},
        })

    md = []
    md.append("# Phase 2 Round 2 — preprocessing sweep results\n")
    md.append(f"**Subjects**: {list(SESSION_MAP.keys())}")
    md.append(f"**Sessions per subject**: {SESSION_MAP}")
    md.append(f"**Grid**: {len(configs)} configs × {sum(len(v) for v in SESSION_MAP.values())} sessions "
              f"= {len(configs)*sum(len(v) for v in SESSION_MAP.values())} runs")
    md.append("")
    md.append("**Primary metric**: `li_median` = median across 4 subjects of the per-subject mean "
              "Lateralization Index (mean of LI over the 3 main motor pairs C3/C4, CP1/CP2, FC1/FC2). "
              "Higher = more contralateral. Per-subject value is the mean of its two sessions.")
    md.append("")
    md.append("**Robustness metric**: `li_min` = minimum of the per-subject LI means across the 4 "
              "subjects. Higher `li_min` means the config doesn't catastrophically fail on any subject.")
    md.append("")

    for mode in ("logratio", "zscore"):
        md.append(f"---\n\n## baseline_mode = `{mode}`\n")
        cs = [c for c in cfg_summary if c["cfg"]["baseline_mode"] == mode]
        cs.sort(key=lambda x: -x["li_median"])

        md.append("### Top 10 by median LI across subjects\n")
        md.append("| rank | spatial | blink | spec_baseline | li_median | li_min | li_mean | lmr_med | rest_L_med | peakL/8 | keep |\n"
                  "|---|---|---|---|---|---|---|---|---|---|---|")
        for i, c in enumerate(cs[:10], 1):
            cfg = c["cfg"]
            md.append(
                f"| {i} | {cfg['spatial_filter']} | {cfg['blink_removal']} | "
                f"{cfg['spectral_baseline']} | "
                f"{c['li_median']:+.2f} | {c['li_min']:+.2f} | {c['li_mean']:+.2f} | "
                f"{c['lmr_median']:+.3f} | {c['rest_l_median']:.3f} | "
                f"{c['n_peak_lmotor_total_of_8']} | {c['keep_frac_mean']:.2f} |"
            )

        md.append("\n### Top 10 by minimum LI across subjects (robustness)\n")
        cs_rob = sorted(cs, key=lambda x: -x["li_min"])
        md.append("| rank | spatial | blink | spec_baseline | li_min | li_median | li_mean |\n"
                  "|---|---|---|---|---|---|---|")
        for i, c in enumerate(cs_rob[:10], 1):
            cfg = c["cfg"]
            md.append(
                f"| {i} | {cfg['spatial_filter']} | {cfg['blink_removal']} | "
                f"{cfg['spectral_baseline']} | "
                f"{c['li_min']:+.2f} | {c['li_median']:+.2f} | {c['li_mean']:+.2f} |"
            )

        md.append("\n### Per-subject LI for top 5 (by median) — see which subjects carry\n")
        md.append("| rank | cfg | " + " | ".join(SESSION_MAP.keys()) + " |")
        md.append("|---|---|" + "|".join("---" for _ in SESSION_MAP) + "|")
        for i, c in enumerate(cs[:5], 1):
            cfg = c["cfg"]
            cfg_short = f"{cfg['spatial_filter']}|{cfg['blink_removal']}|{cfg['spectral_baseline']}"
            subj_vals = " | ".join(
                f"{c['per_subj_li'].get(s, float('nan')):+.2f}" for s in SESSION_MAP.keys()
            )
            md.append(f"| {i} | {cfg_short} | {subj_vals} |")

        # Marginal effects within mode
        md.append("\n### Marginal effects (mean li_median across all other knobs, within mode)\n")
        for key in ("blink_removal", "spatial_filter", "spectral_baseline"):
            md.append(f"\n**By `{key}`**\n")
            md.append("| level | mean li_median | mean li_min | mean rest_L |")
            md.append("|---|---|---|---|")
            levels = {}
            for c in cs:
                lvl = str(c["cfg"][key])
                levels.setdefault(lvl, []).append(c)
            for lvl, cs_lvl in sorted(levels.items()):
                md.append(
                    f"| {lvl} | {np.mean([x['li_median'] for x in cs_lvl]):+.2f} | "
                    f"{np.mean([x['li_min'] for x in cs_lvl]):+.2f} | "
                    f"{np.mean([x['rest_l_median'] for x in cs_lvl]):.3f} |"
                )

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md))
    print(f"Markdown summary written to {OUT_MD}")


def main():
    keys = list(CONFIG_GRID.keys())
    configs = [dict(zip(keys, vals)) for vals in product(*CONFIG_GRID.values())]
    n_cfg = len(configs)
    all_sessions = [(subj, sess) for subj, sl in SESSION_MAP.items() for sess in sl]
    n_sess = len(all_sessions)
    print(f"Sweep: {n_cfg} configs × {n_sess} sessions = {n_cfg*n_sess} runs")

    # Load raw per (subject, session)
    raw_cache = {}
    for subj, sess in all_sessions:
        key = (subj, sess)
        print(f"Loading {subj}/{sess} ...")
        t0 = time.time()
        raw, events, event_dict = load_raw_cached(subj, sess)
        raw_cache[key] = {"raw": raw, "events": events, "event_dict": event_dict}
        print(f"  n_events={len(events)} n_channels={len(raw.ch_names)}  ({time.time()-t0:.1f}s)")

    rows = []
    total_runs = n_cfg * n_sess
    run_idx = 0
    for ci, cfg in enumerate(configs):
        for (subj, sess) in all_sessions:
            run_idx += 1
            t0 = time.time()
            try:
                m = run_config(raw_cache[(subj, sess)], cfg)
                m["status"] = "ok"
            except Exception as e:
                m = {"status": f"error:{type(e).__name__}:{e}"}
            dt = time.time() - t0
            row = {
                "config_idx": ci,
                "subject": subj,
                "session": sess,
                "elapsed_s": round(dt, 2),
                **{f"cfg_{k}": str(v) for k, v in cfg.items()},
                **m,
            }
            rows.append(row)
            cfg_tag = (
                f"{cfg['spatial_filter']}|{cfg['blink_removal']}|"
                f"bl={cfg['spectral_baseline']}|{cfg['baseline_mode']}"
            )
            li = m.get("li_mean", float("nan"))
            peak = m.get("peak_ch", "?")
            print(
                f"[{run_idx:04d}/{total_runs} cfg{ci+1:02d}/{n_cfg} {subj}/{sess}] "
                f"{dt:5.1f}s  {cfg_tag:<60s}  LI={li:+.2f}  peak={peak}"
            )

    fieldnames = sorted({k for r in rows for k in r.keys()})
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV written to {OUT_CSV}")

    rank_and_write_md(rows, configs)


if __name__ == "__main__":
    main()
