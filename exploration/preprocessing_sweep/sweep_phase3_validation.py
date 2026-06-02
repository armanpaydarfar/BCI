#!/usr/bin/env python3
"""
Phase 3 — final validation sweep.

Runs 3 candidate configs across all 7 non-pilot CLIN subjects and every
online session, with auto-drop bad-channel logic ported from
visualize_online_data.py (line ~768–886). Produces a publication-ready
markdown report with per-subject per-session LI tables and sensitivity
analysis across the 3 configs.

Configs:
  A  primary: CAR + drop_fp + logratio + spec_baseline=(-1.5, -0.25)
  B  robustness alt: CSD + fp_regression + logratio + (-0.75, -0.1)
  C  current-default-like: CSD + fp_regression + logratio + (-1.0, 0.0)

Subjects / sessions:
  CLIN_SUBJ_002..008, every online session (excluding CLIN_PILOT_001).

Outputs:
  /home/arman-admin/Documents/SoftwareDocs/clin_erd_phase3_validation.csv
  /home/arman-admin/Documents/SoftwareDocs/clin_erd_phase3_validation.md
"""

import os
import csv
import tempfile
import time
import warnings
import atexit
import signal
import sys
import gc

# resource is POSIX-only; crash diagnostics degrade to a no-op on Windows.
if sys.platform != "win32":
    import resource
else:
    resource = None

import numpy as np
import mne


# ======================================================================
# Crash diagnostics
# ======================================================================

HEARTBEAT_PATH = os.path.join(tempfile.gettempdir(), "sweep_phase3_heartbeat.txt")


def _rss_gb():
    """Return peak RSS in GB (Linux reports ru_maxrss in kB). 0 on Windows."""
    if resource is None:
        return 0.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _heartbeat(note):
    try:
        with open(HEARTBEAT_PATH, "w") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  RSS={_rss_gb():.2f}GB  {note}\n")
    except Exception:
        pass


def _on_exit():
    _heartbeat(f"EXIT (normal), final_rss={_rss_gb():.2f}GB")


def _on_signal(signum, frame):
    try:
        name = signal.Signals(signum).name
    except Exception:
        name = str(signum)
    _heartbeat(f"SIGNAL {name} (signum={signum}), rss={_rss_gb():.2f}GB")
    # Re-raise default for fatal signals so Python prints a traceback if possible
    sys.exit(128 + signum)


atexit.register(_on_exit)
# SIGHUP / SIGUSR1 exist on POSIX only; reference by name to avoid
# AttributeError at import time on Windows.
_diag_signals = [signal.SIGTERM, signal.SIGINT]
for _name in ("SIGHUP", "SIGUSR1"):
    _sig = getattr(signal, _name, None)
    if _sig is not None:
        _diag_signals.append(_sig)
for _sig in _diag_signals:
    try:
        signal.signal(_sig, _on_signal)
    except Exception:
        pass  # some signals can't be caught; best-effort

from sweep_phase2_round2 import (
    load_raw_cached,
    apply_blink_removal, apply_spatial_filter,
    ZONES, LAT_PAIRS_MAIN,
    FS, NOTCH, BB_LO, BB_HI, MU_LO, MU_HI, PAD_TFR, TRIAL_WIN, SCALAR_WIN,
    REJECT_MAX_ABS_UV, FREQS, N_CYCLES, ICA_HP_HZ,
    DATA_DIR,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")


# ======================================================================
# Run config
# ======================================================================

OUT_CSV = "/home/arman-admin/Documents/SoftwareDocs/clin_erd_phase3_validation.csv"
OUT_MD  = "/home/arman-admin/Documents/SoftwareDocs/clin_erd_phase3_validation.md"

SUBJECTS = [f"CLIN_SUBJ_{i:03d}" for i in (2, 3, 4, 5, 6, 7, 8)]

CONFIGS = {
    "A_primary": {
        "spatial_filter":    "car",
        "blink_removal":     "drop_fp",
        "baseline_mode":     "logratio",
        "spectral_baseline": (-1.5, -0.25),
    },
    "B_csd_fpreg_tight": {
        "spatial_filter":    "csd",
        "blink_removal":     "fp_regression",
        "baseline_mode":     "logratio",
        "spectral_baseline": (-0.75, -0.1),
    },
    "C_default_like": {
        "spatial_filter":    "csd",
        "blink_removal":     "fp_regression",
        "baseline_mode":     "logratio",
        "spectral_baseline": (-1.0, 0.0),
    },
}

# Auto-drop tuning (mirrors visualize_online_data.py:181–190)
AUTO_DROP_REJECT_FRAC     = 0.75
AUTO_DROP_DOMINANCE_FRAC  = 0.60
AUTO_DROP_MAX_ITERS       = 4
AUTO_DROP_MAX_CHANNELS    = 4


# ======================================================================
# Session enumeration
# ======================================================================

def enumerate_online_sessions(subject):
    subj_dir = os.path.join(DATA_DIR, f"sub-{subject}")
    if not os.path.isdir(subj_dir):
        return []
    return sorted([
        d.removeprefix("ses-") for d in os.listdir(subj_dir)
        if d.startswith("ses-") and "ONLINE" in d
    ])


# ======================================================================
# run_config with auto-drop (ported from visualize_online_data.py)
# ======================================================================

def _pick_dominant_bad_channel_max_abs(mu_data, ch_names, bad_ix, dom_frac):
    """Among rejected epochs, count which channel most often attains max|x|.
    Mirrors visualize_online_data.py:279–302."""
    if bad_ix.size == 0:
        return None, 0.0
    counts = {ch: 0 for ch in ch_names}
    for ei in bad_ix:
        d = mu_data[int(ei)]
        per_ch_max = np.max(np.abs(d), axis=1)
        worst = int(np.argmax(per_ch_max))
        counts[ch_names[worst]] += 1
    bad_ch = max(counts, key=lambda k: counts[k])
    frac = counts[bad_ch] / bad_ix.size
    if counts[bad_ch] == 0 or frac < dom_frac:
        return None, frac
    return bad_ch, frac


def run_config_auto_drop(raw_cached, config):
    """Apply one preprocessing config with auto-drop bad-channel logic."""
    raw_bb  = raw_cached["raw"].copy()
    raw_1hz = raw_cached["raw"].copy()
    events  = raw_cached["events"]
    event_dict = raw_cached["event_dict"]

    # Notch + broadband
    raw_bb.notch_filter(NOTCH, method="iir", verbose=False)
    raw_bb.filter(l_freq=BB_LO, h_freq=BB_HI, method="iir", verbose=False)

    # 1 Hz copy for ICA (only if selected)
    if config["blink_removal"] == "ica_blink_1hz":
        raw_1hz.notch_filter(NOTCH, method="iir", verbose=False)
        raw_1hz.filter(l_freq=ICA_HP_HZ, h_freq=BB_HI, method="iir", verbose=False)

    # Blink removal (may drop Fp channels if drop_fp)
    raw_bb, blink_info = apply_blink_removal(raw_bb, raw_1hz, config["blink_removal"])

    # --- auto-drop loop ---
    dropped_channels = []
    iters = 0
    t0, t1 = TRIAL_WIN
    while True:
        iters += 1

        raw_mu = raw_bb.copy()
        raw_mu.filter(l_freq=MU_LO, h_freq=MU_HI, method="iir", verbose=False)

        epoch_kw = dict(
            event_id=event_dict,
            tmin=t0 - PAD_TFR, tmax=t1 + PAD_TFR,
            baseline=None, detrend=1, preload=True, verbose=False,
        )
        epochs_mu = mne.Epochs(raw_mu, events, reject=None, flat=None, **epoch_kw)
        epochs_bb = mne.Epochs(raw_bb, events, reject=None, flat=None, **epoch_kw)
        mu_data = epochs_mu.get_data()
        mask = np.max(np.abs(mu_data), axis=(1, 2)) <= REJECT_MAX_ABS_UV
        good_ix = np.where(mask)[0].tolist()
        bad_ix = np.where(~mask)[0]

        n_attempted = int(len(events))
        n_kept = int(len(good_ix))
        dropped_frac = 1.0 - (n_kept / n_attempted) if n_attempted else 1.0

        if dropped_frac < AUTO_DROP_REJECT_FRAC:
            break
        if len(dropped_channels) >= AUTO_DROP_MAX_CHANNELS:
            break
        if iters > AUTO_DROP_MAX_ITERS:
            break

        bad_ch, frac = _pick_dominant_bad_channel_max_abs(
            mu_data, list(epochs_mu.ch_names), bad_ix, AUTO_DROP_DOMINANCE_FRAC
        )
        if bad_ch is None or bad_ch not in raw_bb.ch_names:
            break

        raw_bb = raw_bb.copy().drop_channels([bad_ch])
        dropped_channels.append(bad_ch)

    epochs = epochs_bb[good_ix]
    if len(epochs) == 0:
        raise RuntimeError("All epochs rejected after auto-drop")

    # Spatial filter
    epochs = apply_spatial_filter(epochs, config["spatial_filter"])

    # TFR per marker
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

    lis = []
    for c_ch, i_ch in LAT_PAIRS_MAIN:
        if c_ch in mi and i_ch in mi:
            c_v, i_v = mi[c_ch], mi[i_ch]
            lis.append((i_v - c_v) / (abs(c_v) + abs(i_v) + 1e-9))
    li_mean = float(np.mean(lis)) if lis else float("nan")
    li_c3c4 = float("nan")
    if "C3" in mi and "C4" in mi:
        li_c3c4 = (mi["C4"] - mi["C3"]) / (abs(mi["C3"]) + abs(mi["C4"]) + 1e-9)

    l_chs = ZONES["L-motor"]; r_chs = ZONES["R-motor"]
    l_vals = [mi[c] for c in l_chs if c in mi]
    r_vals = [mi[c] for c in r_chs if c in mi]
    l_mag = float(-np.mean(l_vals)) if l_vals else float("nan")
    r_mag = float(-np.mean(r_vals)) if r_vals else float("nan")
    l_mean_log = float(np.mean(l_vals)) if l_vals else float("nan")
    r_mean_log = float(np.mean(r_vals)) if r_vals else float("nan")

    if mi:
        peak_ch = min(mi, key=lambda k: mi[k])
        peak_v = mi[peak_ch]
        peak_zone = next((z for z, chs in ZONES.items() if peak_ch in chs), "Other")
    else:
        peak_ch, peak_v, peak_zone = "", float("nan"), ""

    rest_l = [rest[c] for c in l_chs if c in rest]
    rest_l_mag = float(np.mean([abs(v) for v in rest_l])) if rest_l else float("nan")
    rest_fr = [rest[c] for c in ZONES["Frontal"] if c in rest]
    rest_fr_mag = float(np.mean([abs(v) for v in rest_fr])) if rest_fr else float("nan")

    return {
        "n_kept": n_kept,
        "n_attempted": n_attempted,
        "keep_frac": n_kept / n_attempted if n_attempted else 0.0,
        "dropped_channels": ",".join(dropped_channels),
        "n_dropped_channels": len(dropped_channels),
        "blink_info": blink_info,
        "n_channels_after_preproc": len(epochs.ch_names),
        "li_mean": li_mean,
        "li_c3c4": li_c3c4,
        "l_motor_mag_log": l_mag,
        "r_motor_mag_log": r_mag,
        "l_motor_mean_log": l_mean_log,
        "r_motor_mean_log": r_mean_log,
        "peak_ch": peak_ch,
        "peak_zone": peak_zone,
        "peak_value": peak_v,
        "rest_l_mag_log": rest_l_mag,
        "rest_frontal_mag_log": rest_fr_mag,
        "c3_mi_log": mi.get("C3", float("nan")),
        "c4_mi_log": mi.get("C4", float("nan")),
        "cp1_mi_log": mi.get("CP1", float("nan")),
        "fc1_mi_log": mi.get("FC1", float("nan")),
    }


# ======================================================================
# Report formatting
# ======================================================================

def log_to_pct(x):
    return 100.0 * (10.0 ** x - 1.0) if np.isfinite(x) else float("nan")


def classify_pattern(peak_zone, li_c3c4, l_mean_log, r_mean_log):
    both_neg = (l_mean_log < 0) and (r_mean_log < 0)
    if peak_zone == "L-motor" and li_c3c4 > 0.15:
        return "classical contralateral"
    if peak_zone == "R-motor" and li_c3c4 < -0.15:
        return "ipsilateral"
    if both_neg and abs(li_c3c4) <= 0.15:
        return "bilateral"
    if (l_mean_log > 0) and (r_mean_log > 0):
        return "no ERD (ERS/drift)"
    return "non-classical / mixed"


def write_report(rows):
    # group rows: by config, then by subject, then by session
    by_cfg = {}  # cfg_name -> {subject -> [row]}
    for r in rows:
        if r.get("status") != "ok":
            continue
        by_cfg.setdefault(r["config"], {}).setdefault(r["subject"], []).append(r)

    md = []
    md.append("# Phase 3 — final validation report\n")
    md.append(f"**Subjects**: {SUBJECTS}")
    md.append(f"**Configs evaluated**:")
    for name, cfg in CONFIGS.items():
        md.append(f"  - `{name}`: {cfg}")
    md.append("")
    md.append("**Primary metric**: Lateralization Index over 3 motor pairs "
              "(C3/C4, CP1/CP2, FC1/FC2). LI > 0 = contralateral (LEFT) dominant ERD, "
              "the classical pattern for right-arm MI. For interpretation: "
              "|LI| > 0.15 = clearly lateralized; |LI| ≤ 0.15 = bilateral.")
    md.append("")

    # ---- Per-config per-subject per-session tables ----
    for cfg_name, cfg in CONFIGS.items():
        md.append(f"---\n\n## Config `{cfg_name}` = {cfg}\n")

        if cfg_name not in by_cfg:
            md.append("_(no successful runs)_")
            continue

        # per-subject summary
        md.append("### Per-subject summary\n")
        md.append("| subject | n_sessions | median LI | min LI | max LI | "
                  "classical / bilateral / ipsi / mixed / failed |\n"
                  "|---|---|---|---|---|---|")
        subj_pattern_counts = {}
        for subj in SUBJECTS:
            rs = by_cfg[cfg_name].get(subj, [])
            if not rs:
                md.append(f"| {subj} | 0 | — | — | — | all failed |")
                continue
            lis = [r["li_mean"] for r in rs if np.isfinite(r["li_mean"])]
            if not lis:
                md.append(f"| {subj} | {len(rs)} | — | — | — | — |")
                continue
            counts = {"classical contralateral": 0, "bilateral": 0,
                      "ipsilateral": 0, "non-classical / mixed": 0,
                      "no ERD (ERS/drift)": 0}
            for r in rs:
                pat = classify_pattern(
                    r["peak_zone"], r["li_c3c4"],
                    r["l_motor_mean_log"], r["r_motor_mean_log"]
                )
                counts[pat] = counts.get(pat, 0) + 1
            subj_pattern_counts[subj] = counts
            pat_str = (
                f"{counts['classical contralateral']}/{counts['bilateral']}/"
                f"{counts['ipsilateral']}/{counts['non-classical / mixed']+counts['no ERD (ERS/drift)']}"
            )
            md.append(
                f"| {subj} | {len(rs)} | {np.median(lis):+.2f} | "
                f"{np.min(lis):+.2f} | {np.max(lis):+.2f} | {pat_str} |"
            )

        # per-session details for each subject
        md.append("\n### Per-session details\n")
        for subj in SUBJECTS:
            rs = by_cfg[cfg_name].get(subj, [])
            if not rs:
                continue
            md.append(f"\n**{subj}**\n")
            md.append("| session | LI | LI_C3/C4 | L-motor %ERD | R-motor %ERD | "
                      "peak | zone | pattern | kept | dropped_ch |")
            md.append("|---|---|---|---|---|---|---|---|---|---|")
            for r in sorted(rs, key=lambda x: x["session"]):
                pat = classify_pattern(
                    r["peak_zone"], r["li_c3c4"],
                    r["l_motor_mean_log"], r["r_motor_mean_log"]
                )
                l_pct = log_to_pct(r["l_motor_mean_log"])
                r_pct = log_to_pct(r["r_motor_mean_log"])
                md.append(
                    f"| {r['session']} | {r['li_mean']:+.2f} | "
                    f"{r['li_c3c4']:+.2f} | {l_pct:+.1f}% | {r_pct:+.1f}% | "
                    f"{r['peak_ch']} | {r['peak_zone']} | {pat} | "
                    f"{r['n_kept']}/{r['n_attempted']} | "
                    f"{r['dropped_channels'] or '—'} |"
                )

    # ---- Sensitivity analysis: config-vs-config per subject ----
    md.append("\n---\n\n## Sensitivity analysis — median LI per subject across configs\n")
    md.append("| subject | " + " | ".join(CONFIGS.keys()) + " |")
    md.append("|---|" + "|".join("---" for _ in CONFIGS) + "|")
    for subj in SUBJECTS:
        row = [subj]
        for cfg_name in CONFIGS:
            rs = by_cfg.get(cfg_name, {}).get(subj, [])
            lis = [r["li_mean"] for r in rs if np.isfinite(r["li_mean"])]
            row.append(f"{np.median(lis):+.2f}" if lis else "—")
        md.append("| " + " | ".join(row) + " |")

    md.append("\n## Sensitivity analysis — % of sessions classified 'classical contralateral'\n")
    md.append("| subject | " + " | ".join(CONFIGS.keys()) + " |")
    md.append("|---|" + "|".join("---" for _ in CONFIGS) + "|")
    for subj in SUBJECTS:
        row = [subj]
        for cfg_name in CONFIGS:
            rs = by_cfg.get(cfg_name, {}).get(subj, [])
            if not rs:
                row.append("—"); continue
            n_class = sum(
                1 for r in rs
                if classify_pattern(r["peak_zone"], r["li_c3c4"],
                                    r["l_motor_mean_log"], r["r_motor_mean_log"]) == "classical contralateral"
            )
            row.append(f"{n_class}/{len(rs)}")
        md.append("| " + " | ".join(row) + " |")

    # ---- Dataset-wide dashboard ----
    md.append("\n---\n\n## Dataset-wide rollup\n")
    md.append("| config | n_sessions_ok | median LI | subjects with median LI > 0 | "
              "sessions classified classical | mean keep_frac |")
    md.append("|---|---|---|---|---|---|")
    for cfg_name in CONFIGS:
        rs_all = [r for subj_rs in by_cfg.get(cfg_name, {}).values() for r in subj_rs]
        if not rs_all:
            md.append(f"| {cfg_name} | 0 | — | — | — | — |")
            continue
        lis_all = [r["li_mean"] for r in rs_all if np.isfinite(r["li_mean"])]
        subj_medians = {}
        for subj, rs in by_cfg[cfg_name].items():
            lis = [r["li_mean"] for r in rs if np.isfinite(r["li_mean"])]
            if lis:
                subj_medians[subj] = np.median(lis)
        n_subj_contra = sum(1 for v in subj_medians.values() if v > 0)
        n_class = sum(
            1 for r in rs_all
            if classify_pattern(r["peak_zone"], r["li_c3c4"],
                                r["l_motor_mean_log"], r["r_motor_mean_log"]) == "classical contralateral"
        )
        keep = np.mean([r["keep_frac"] for r in rs_all])
        md.append(
            f"| {cfg_name} | {len(rs_all)} | {np.median(lis_all):+.2f} | "
            f"{n_subj_contra}/{len(subj_medians)} | {n_class}/{len(rs_all)} | "
            f"{keep:.2f} |"
        )

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md))
    print(f"Report: {OUT_MD}")


# ======================================================================
# Main
# ======================================================================

def main():
    # enumerate (subject, session) pairs
    session_list = []
    for subj in SUBJECTS:
        for sess in enumerate_online_sessions(subj):
            session_list.append((subj, sess))
    print(f"Sessions: {len(session_list)} across {len(SUBJECTS)} subjects")
    print(f"Configs: {list(CONFIGS.keys())}")
    print(f"Total runs: {len(session_list) * len(CONFIGS)}")
    _heartbeat("main() start")

    # Load per-run (not upfront) — pre-caching all 30 sessions was the Round-3
    # OOM cause (10.6 GB RSS at session 13 of 30). On-demand is ~30s slower over
    # the whole sweep but keeps peak RSS bounded.
    rows = []
    total = len(session_list) * len(CONFIGS)
    idx = 0
    for cfg_name, cfg in CONFIGS.items():
        for (subj, sess) in session_list:
            idx += 1
            t0 = time.time()
            _heartbeat(f"starting run {idx}/{total} {cfg_name} {subj}/{sess}")
            raw = events = event_dict = cached = None
            try:
                raw, events, event_dict = load_raw_cached(subj, sess)
                cached = {"raw": raw, "events": events, "event_dict": event_dict}
                m = run_config_auto_drop(cached, cfg)
                m["status"] = "ok"
            except Exception as e:
                m = {"status": f"error:{type(e).__name__}:{e}"}
            finally:
                del raw, events, event_dict, cached
                gc.collect()
            dt = time.time() - t0
            row = {
                "config": cfg_name,
                "subject": subj,
                "session": sess,
                "elapsed_s": round(dt, 2),
                **{f"cfg_{k}": str(v) for k, v in cfg.items()},
                **m,
            }
            rows.append(row)
            li = m.get("li_mean", float("nan"))
            peak = m.get("peak_ch", "?")
            drop = m.get("dropped_channels", "")
            rss = _rss_gb()
            print(
                f"[{idx:03d}/{total}] {cfg_name:<22s} {subj}/{sess}  {dt:5.1f}s  "
                f"LI={li:+.2f}  peak={peak}  drop={drop or '—'}  rss={rss:.2f}GB",
                flush=True,
            )
            _heartbeat(f"finished run {idx}/{total} rss={rss:.2f}GB")

    fieldnames = sorted({k for r in rows for k in r.keys()})
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV: {OUT_CSV}")

    write_report(rows)


if __name__ == "__main__":
    main()
