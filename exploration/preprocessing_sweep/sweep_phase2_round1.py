#!/usr/bin/env python3
"""
Phase 2 — Round 1 preprocessing sweep.

Scope: for each of the 3 calibration sessions (CLIN_SUBJ_003 S003/4/5), run the
Cartesian product of
  - filter_type:   {iir, fir}
  - blink_removal: {none, fp_regression, ica_blink}
  - spatial_filter:{car, csd, hjorth}
  - baseline:      {(-1.0, 0.0), (-1.5, -0.25)}

= 36 configs × 3 sessions = 108 runs.

For each (config, session) emit scalar metrics focused on MU-band ERD during
right-arm MI:
  - LI_mean: mean lateralization index over (C3/C4, CP1/CP2, FC1/FC2).
             >0 means contralateral (LEFT) dominant.
  - L-motor magnitude, R-motor magnitude (logratio, positive = strong ERD)
  - L−R magnitude (positive = contralateral dominates)
  - Peak electrode (name + zone)
  - Rest flatness at L-motor (smaller = better)
  - MI−Rest contrast at C3
  - Epochs kept / attempted, plus blink-removal diagnostics

Results written to
  /home/arman-admin/Documents/SoftwareDocs/clin_erd_phase2_round1.csv
and a markdown top-10 summary to
  /home/arman-admin/Documents/SoftwareDocs/clin_erd_phase2_round1.md
"""

import os
import csv
import time
import json
import warnings
from itertools import product

import numpy as np
import mne

from Utils.stream_utils import load_xdf, get_channel_names_from_xdf
from Utils.preprocessing import concatenate_streams
from config import DATA_DIR

# Silence MNE chatter for the sweep (we still print our own per-config line)
mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")


# ======================================================================
# Run config
# ======================================================================

SUBJECT   = "CLIN_SUBJ_003"
SESSIONS  = ["S003ONLINE", "S004ONLINE", "S005ONLINE"]
OUT_CSV   = "/home/arman-admin/Documents/SoftwareDocs/clin_erd_phase2_round1.csv"
OUT_MD    = "/home/arman-admin/Documents/SoftwareDocs/clin_erd_phase2_round1.md"

FS           = 512
NOTCH        = 60
BB_LO, BB_HI = 4.0, 40.0
MU_LO, MU_HI = 8.0, 13.0
PAD_TFR      = 1.0
TRIAL_WIN    = (-1.0, 4.0)          # epoch analysis window (padding applied)
SCALAR_WIN   = (1.0, 4.0)
REJECT_MAX_ABS_UV = 50.0             # fixed in Round 1
FREQS        = np.linspace(8, 30, 23)
N_CYCLES     = FREQS / 2.0

# 10-20 zones (same as numerical_erd_reader)
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
# Raw loading — done once per session, cached
# ======================================================================

def load_raw_cached(subject, session):
    """Load XDF → mne.Raw (unfiltered) + events. Mirrors viz loading block
    through montage setup (visualize_online_data.py:595–683) but stops before
    notch/bandpass.
    """
    xdf_dir = os.path.join(DATA_DIR, f"sub-{subject}", f"ses-{session}", "eeg/")
    xdf_files = sorted(
        [os.path.join(xdf_dir, f) for f in os.listdir(xdf_dir) if f.endswith(".xdf")]
    )
    if not xdf_files:
        raise FileNotFoundError(f"No XDF in {xdf_dir}")

    eeg_streams, marker_streams = [], []
    for f in xdf_files:
        es, ms = load_xdf(f)
        eeg_streams.append(es)
        marker_streams.append(ms)

    if len(eeg_streams) == 1:
        eeg_stream, marker_stream = eeg_streams[0], marker_streams[0]
    else:
        eeg_stream, marker_stream = concatenate_streams(eeg_streams, marker_streams)

    eeg_timestamps   = np.array(eeg_stream["time_stamps"])
    eeg_data_raw     = np.array(eeg_stream["time_series"]).T
    channel_names    = get_channel_names_from_xdf(eeg_stream)
    marker_data      = np.array([int(v[0]) for v in marker_stream["time_series"]])
    marker_timestamps = np.array([float(v[1]) for v in marker_stream["time_series"]])

    non_eeg = {"AUX1", "AUX2", "AUX3", "AUX7", "AUX8", "AUX9", "TRIGGER"}
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

    # --- Build events (match viz trial def; duration 1.0–5.5s) ---
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
        t_start = marker_timestamps[idx]
        duration = end_time - t_start
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

def apply_blink_removal(raw, method):
    """In-place blink removal on broadband-filtered raw. Returns (raw, info_str).

    - 'none': pass-through
    - 'fp_regression': OLS regress Fp1/Fp2/Fpz out of every other channel
    - 'ica_blink': fit Picard ICA; identify blink components by correlation
                   with Fp1 and Fp2 (surrogate EOG via find_bads_eog)
    """
    if method == "none":
        return raw, "none"

    if method == "fp_regression":
        fp_chans = [c for c in ("Fp1", "Fp2", "Fpz") if c in raw.ch_names]
        if not fp_chans:
            return raw, "fp_regression:no_fp"
        fp_idx = [raw.ch_names.index(c) for c in fp_chans]
        all_data = raw.get_data()
        X = all_data[fp_idx].T                         # (n_samp, n_fp)
        Xi = np.hstack([X, np.ones((X.shape[0], 1))])  # + intercept
        out = all_data.copy()
        for ch_idx, ch in enumerate(raw.ch_names):
            if ch in fp_chans:
                continue
            y = all_data[ch_idx]
            coefs, *_ = np.linalg.lstsq(Xi, y, rcond=None)
            out[ch_idx] = y - Xi @ coefs
        raw._data[:] = out
        return raw, f"fp_regression:{','.join(fp_chans)}"

    if method == "ica_blink":
        ica = mne.preprocessing.ICA(
            n_components=0.99, method="picard",
            random_state=42, max_iter=200, verbose=False,
        )
        ica.fit(raw, verbose=False)
        bads = []
        for ch in ("Fp1", "Fp2"):
            if ch in raw.ch_names:
                try:
                    b, _ = ica.find_bads_eog(raw, ch_name=ch, verbose=False)
                    bads.extend(b)
                except Exception:
                    pass
        bads = sorted(set(bads))
        if bads:
            ica.exclude = bads
            raw_new = ica.apply(raw.copy(), verbose=False)
            raw._data[:] = raw_new.get_data()
            return raw, f"ica_blink:removed={len(bads)}_of_{ica.n_components_}"
        return raw, f"ica_blink:removed=0_of_{ica.n_components_}"

    raise ValueError(f"Unknown blink_removal: {method}")


# ======================================================================
# Spatial filter
# ======================================================================

def _apply_hjorth(epochs, k=4):
    """Hjorth Laplacian: subtract mean of k nearest neighbors. Matches the
    logic of visualize_online_data.py:388–419 but kept local to avoid
    mutating viz's module state."""
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
# One config run
# ======================================================================

def run_config(raw_cached, config):
    """Apply one preprocessing config to cached raw+events, return metrics dict."""
    raw = raw_cached["raw"].copy()
    events = raw_cached["events"]
    event_dict = raw_cached["event_dict"]

    # 1) Notch + broadband bandpass
    raw.notch_filter(NOTCH, method=config["filter_type"], verbose=False)
    raw.filter(l_freq=BB_LO, h_freq=BB_HI, method=config["filter_type"], verbose=False)

    # 2) Blink removal
    raw, blink_info = apply_blink_removal(raw, config["blink_removal"])

    # 3) Mu-band copy for QC
    raw_mu = raw.copy()
    raw_mu.filter(l_freq=MU_LO, h_freq=MU_HI, method="iir", verbose=False)

    # 4) Epoch (padded) on both mu (QC) and broadband (TFR)
    t0, t1 = TRIAL_WIN
    epoch_kw = dict(
        event_id=event_dict,
        tmin=t0 - PAD_TFR, tmax=t1 + PAD_TFR,
        baseline=config["baseline"],
        detrend=1, preload=True, verbose=False,
    )
    epochs_mu = mne.Epochs(raw_mu, events, reject=None, flat=None, **epoch_kw)
    epochs_bb = mne.Epochs(raw,    events, reject=None, flat=None, **epoch_kw)

    # 5) max_abs rejection on mu copy
    mu_data = epochs_mu.get_data()
    mask = np.max(np.abs(mu_data), axis=(1, 2)) <= REJECT_MAX_ABS_UV
    good_ix = np.where(mask)[0].tolist()
    epochs = epochs_bb[good_ix]
    n_attempted = int(len(events))
    n_kept = int(len(epochs))

    # 6) Spatial filter
    epochs = apply_spatial_filter(epochs, config["spatial_filter"])

    # 7) TFR per marker
    per_ch = {}
    for marker in ("100", "200"):
        if marker not in epochs.event_id or len(epochs[marker]) == 0:
            per_ch[marker] = {}
            continue
        tfr = epochs[marker].compute_tfr(
            method="multitaper", freqs=FREQS, n_cycles=N_CYCLES,
            tmin=t0 - PAD_TFR, tmax=t1 + PAD_TFR,
            use_fft=True, return_itc=False, average=False, verbose=False,
        )
        tfr.apply_baseline(baseline=(t0, 0.0), mode="logratio", verbose=False)
        tfr.crop(tmin=t0, tmax=t1)

        fmask = (tfr.freqs >= MU_LO) & (tfr.freqs <= MU_HI)
        tmask = (tfr.times >= SCALAR_WIN[0]) & (tfr.times <= SCALAR_WIN[1])
        vals = tfr.data[:, :, fmask, :][:, :, :, tmask].mean(axis=(0, 2, 3))
        per_ch[marker] = dict(zip(tfr.ch_names, vals))

    mi = per_ch.get("200", {})
    rest = per_ch.get("100", {})

    # 8) Scoring
    lis = []
    for c_ch, i_ch in LAT_PAIRS_MAIN:
        if c_ch in mi and i_ch in mi:
            c_l, i_l = mi[c_ch], mi[i_ch]
            lis.append((i_l - c_l) / (abs(c_l) + abs(i_l) + 1e-9))
    li_mean = float(np.mean(lis)) if lis else float("nan")

    l_chs = ZONES["L-motor"]
    r_chs = ZONES["R-motor"]
    l_mag = float(-np.mean([mi[c] for c in l_chs if c in mi])) if any(c in mi for c in l_chs) else float("nan")
    r_mag = float(-np.mean([mi[c] for c in r_chs if c in mi])) if any(c in mi for c in r_chs) else float("nan")

    if mi:
        peak_ch = min(mi, key=lambda k: mi[k])
        peak_log = mi[peak_ch]
        peak_zone = next((z for z, chs in ZONES.items() if peak_ch in chs), "Other")
    else:
        peak_ch, peak_log, peak_zone = "", float("nan"), ""

    rest_l_mag = float(np.mean([abs(rest[c]) for c in l_chs if c in rest])) \
                 if any(c in rest for c in l_chs) else float("nan")
    rest_frontal_mag = float(np.mean([abs(rest[c]) for c in ZONES["Frontal"] if c in rest])) \
                 if any(c in rest for c in ZONES["Frontal"]) else float("nan")

    c3_contrast = float(rest.get("C3", 0.0) - mi.get("C3", 0.0))  # positive = MI stronger

    def pct(x):
        return 100.0 * (10.0 ** x - 1.0) if np.isfinite(x) else float("nan")

    return {
        "n_kept": n_kept,
        "n_attempted": n_attempted,
        "keep_frac": n_kept / n_attempted if n_attempted else 0.0,
        "blink_info": blink_info,
        "li_mean": li_mean,
        "l_motor_mag_log": l_mag,
        "r_motor_mag_log": r_mag,
        "l_minus_r_mag_log": l_mag - r_mag if np.isfinite(l_mag) and np.isfinite(r_mag) else float("nan"),
        "peak_ch": peak_ch,
        "peak_zone": peak_zone,
        "peak_pct": pct(peak_log),
        "rest_l_mag_log": rest_l_mag,
        "rest_frontal_mag_log": rest_frontal_mag,
        "c3_contrast_log": c3_contrast,
        "c3_mi_pct": pct(mi.get("C3", float("nan"))),
        "c3_rest_pct": pct(rest.get("C3", float("nan"))),
        "fc1_mi_pct": pct(mi.get("FC1", float("nan"))),
        "cp1_mi_pct": pct(mi.get("CP1", float("nan"))),
    }


# ======================================================================
# Main
# ======================================================================

CONFIG_GRID = {
    "filter_type":    ["iir", "fir"],
    "blink_removal":  ["none", "fp_regression", "ica_blink"],
    "spatial_filter": ["car", "csd", "hjorth"],
    "baseline":       [(-1.0, 0.0), (-1.5, -0.25)],
}


def main():
    keys = list(CONFIG_GRID.keys())
    configs = [dict(zip(keys, vals)) for vals in product(*CONFIG_GRID.values())]
    n_cfg = len(configs)
    print(f"Sweep: {n_cfg} configs × {len(SESSIONS)} sessions = {n_cfg*len(SESSIONS)} runs")

    # Load raw once per session
    raw_cache = {}
    for sess in SESSIONS:
        print(f"Loading {SUBJECT}/{sess} ...")
        t0 = time.time()
        raw, events, event_dict = load_raw_cached(SUBJECT, sess)
        raw_cache[sess] = {"raw": raw, "events": events, "event_dict": event_dict}
        print(f"  n_events={len(events)} n_channels={len(raw.ch_names)}  ({time.time()-t0:.1f}s)")

    rows = []
    for ci, cfg in enumerate(configs):
        for sess in SESSIONS:
            t0 = time.time()
            try:
                m = run_config(raw_cache[sess], cfg)
                m["status"] = "ok"
            except Exception as e:
                m = {"status": f"error:{type(e).__name__}:{e}"}
            dt = time.time() - t0
            row = {
                "config_idx": ci,
                "session": sess,
                "elapsed_s": round(dt, 2),
                **{f"cfg_{k}": str(v) for k, v in cfg.items()},
                **m,
            }
            rows.append(row)
            cfg_tag = (
                f"{cfg['filter_type']}|{cfg['spatial_filter']}|"
                f"{cfg['blink_removal']}|bl={cfg['baseline']}"
            )
            li = m.get("li_mean", float("nan"))
            lmag = m.get("l_motor_mag_log", float("nan"))
            peak = m.get("peak_ch", "?")
            print(
                f"[{ci+1:02d}/{n_cfg} {sess}] {dt:5.1f}s  {cfg_tag:<58s}  "
                f"LI={li:+.2f}  Lmag={lmag:+.3f}  peak={peak}"
            )

    # --- Write CSV ---
    fieldnames = sorted({k for r in rows for k in r.keys()})
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV written to {OUT_CSV}")

    # --- Build markdown summary: rank by mean score across the 3 sessions ---
    # Composite score = li_mean + l_minus_r_mag_log − 0.5*rest_l_mag_log
    # (positive = contra-dominant + strong L-motor + flat rest)
    cfg_scores = {}  # ci -> list of per-session dicts
    for r in rows:
        if r.get("status") != "ok":
            continue
        ci = r["config_idx"]
        cfg_scores.setdefault(ci, []).append(r)

    composite = []
    for ci, rs in cfg_scores.items():
        if len(rs) < len(SESSIONS):
            continue
        li = np.mean([r["li_mean"] for r in rs])
        lmr = np.mean([r["l_minus_r_mag_log"] for r in rs])
        restl = np.mean([r["rest_l_mag_log"] for r in rs])
        restfr = np.mean([r["rest_frontal_mag_log"] for r in rs])
        lmag = np.mean([r["l_motor_mag_log"] for r in rs])
        n_peak_lmotor = sum(1 for r in rs if r["peak_zone"] == "L-motor")
        keep_frac = np.mean([r["keep_frac"] for r in rs])
        score = li + lmr - 0.5 * restl
        composite.append({
            "config_idx": ci,
            "cfg": configs[ci],
            "score": score,
            "li_mean": li,
            "l_minus_r_mag_log": lmr,
            "l_motor_mag_log": lmag,
            "rest_l_mag_log": restl,
            "rest_frontal_mag_log": restfr,
            "n_peak_lmotor_of_3": n_peak_lmotor,
            "keep_frac": keep_frac,
        })

    composite.sort(key=lambda x: -x["score"])

    md = []
    md.append("# Phase 2 Round 1 — preprocessing sweep results\n")
    md.append(f"**Subject/sessions**: {SUBJECT} / {SESSIONS}\n")
    md.append(f"**Grid**: filter_type × blink_removal × spatial_filter × baseline = {n_cfg} configs\n")
    md.append("**Composite score** = mean(LI) + mean(L−R magnitude, log) − 0.5·mean(|REST L-motor|, log). "
              "Higher = more contralateral mu-band MI ERD with flatter rest.\n")
    md.append("## Top 10 configs\n")
    md.append("| rank | filter | spatial | blink | baseline | score | LI | L−R mag | L-mag | rest L | rest Fr | peak_Lmotor/3 | keep |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, c in enumerate(composite[:10], 1):
        cfg = c["cfg"]
        md.append(
            f"| {i} | {cfg['filter_type']} | {cfg['spatial_filter']} | "
            f"{cfg['blink_removal']} | {cfg['baseline']} | "
            f"{c['score']:+.3f} | {c['li_mean']:+.2f} | "
            f"{c['l_minus_r_mag_log']:+.3f} | {c['l_motor_mag_log']:+.3f} | "
            f"{c['rest_l_mag_log']:.3f} | {c['rest_frontal_mag_log']:.3f} | "
            f"{c['n_peak_lmotor_of_3']} | {c['keep_frac']:.2f} |"
        )

    md.append("\n## Bottom 5 configs (for contrast)\n")
    md.append("| rank | filter | spatial | blink | baseline | score | LI | L−R mag |\n"
              "|---|---|---|---|---|---|---|---|")
    for i, c in enumerate(composite[-5:], 1):
        cfg = c["cfg"]
        md.append(
            f"| {len(composite)-5+i} | {cfg['filter_type']} | {cfg['spatial_filter']} | "
            f"{cfg['blink_removal']} | {cfg['baseline']} | "
            f"{c['score']:+.3f} | {c['li_mean']:+.2f} | "
            f"{c['l_minus_r_mag_log']:+.3f} |"
        )

    # Marginal effects — average score by each knob level
    md.append("\n## Marginal effects (mean score across all other knobs)\n")
    for key in CONFIG_GRID.keys():
        md.append(f"\n### By `{key}`\n")
        md.append("| level | mean score | mean LI | mean L−R mag | mean rest_L |")
        md.append("|---|---|---|---|---|")
        levels = {}
        for c in composite:
            lvl = str(c["cfg"][key])
            levels.setdefault(lvl, []).append(c)
        for lvl, cs in sorted(levels.items()):
            md.append(
                f"| {lvl} | {np.mean([x['score'] for x in cs]):+.3f} | "
                f"{np.mean([x['li_mean'] for x in cs]):+.2f} | "
                f"{np.mean([x['l_minus_r_mag_log'] for x in cs]):+.3f} | "
                f"{np.mean([x['rest_l_mag_log'] for x in cs]):.3f} |"
            )

    with open(OUT_MD, "w") as f:
        f.write("\n".join(md))
    print(f"Markdown summary written to {OUT_MD}")


if __name__ == "__main__":
    main()
