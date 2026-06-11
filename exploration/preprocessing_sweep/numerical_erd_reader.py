#!/usr/bin/env python3
"""
Numerical ERD reader — Phase 1 calibration.

Reuses the load/epoch/TFR pipeline from visualize_online_data.py and emits a
text-only report of per-session ERD metrics. Intended to be compared against
the user's visual impressions from the topomap + focal-timecourse plots; if
the numerical reads agree with the visual reads on sessions the user knows
well, we proceed to Phase 2 (preprocessing sweep).

No plots are produced. No new preprocessing is introduced here.

Notes on pipeline state:
 - This script imports `visualize_online_data` as `viz` (that module guards
   with `if __name__ == "__main__"`, so import is side-effect free).
 - We broaden viz.FREQS to span 8-30 Hz so mu and beta are both reported from
   one TFR pass. All other viz.* globals (SPATIAL_FILTER, rejection mode,
   baseline, scalar window) are left at the defaults authored in that file.
"""

import os
import numpy as np

import visualize_online_data as viz


# ======================================================================
# Run config
# ======================================================================

SUBJECT = "CLIN_SUBJ_003"
SESSIONS = ["S003ONLINE", "S004ONLINE", "S005ONLINE"]
OUTPUT_PATH = "/home/arman-admin/Documents/SoftwareDocs/clin_erd_phase1_calibration.md"
SCALAR_WINDOW = (1.0, 4.0)

# For right-arm MI (marker 200), contralateral = LEFT hemisphere.
TASK_CONTRA_SIDE = "L"

# 10-20 zone mapping — channels not in any zone are reported as "Other".
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

# Matched left/right pairs for lateralization. For right-arm MI, the first
# entry is contralateral (left), the second is ipsilateral (right).
LAT_PAIRS = [
    ("C3",  "C4"),
    ("CP1", "CP2"),
    ("CP5", "CP6"),
    ("FC1", "FC2"),
    ("FC5", "FC6"),
    ("P3",  "P4"),
    ("F3",  "F4"),
]

BANDS_TO_REPORT = [("mu", 8, 13), ("beta", 13, 30)]


# ======================================================================
# Helpers
# ======================================================================

def zone_of(ch: str) -> str:
    for z, chans in ZONES.items():
        if ch in chans:
            return z
    return "Other"


def logratio_to_percent(x):
    return 100.0 * (10.0 ** x - 1.0)


def per_ch_scalar(tfr, band_lo, band_hi, t_lo, t_hi):
    """Return (mean_logratio_per_ch, per_trial_per_ch) averaged over freq band & time window.

    tfr.data shape: (trials, ch, freq, time). TFR is already baseline-normalized
    (logratio) by compute_tfr_epochs.
    """
    freqs = tfr.freqs
    times = tfr.times
    fmask = (freqs >= band_lo) & (freqs <= band_hi)
    tmask = (times >= t_lo) & (times <= t_hi)
    per_trial = tfr.data[:, :, fmask, :][:, :, :, tmask].mean(axis=(2, 3))  # (trials, ch)
    return per_trial.mean(axis=0), per_trial


def lateralization_index(contra_log, ipsi_log):
    """LI = (ipsi - contra) / (|contra| + |ipsi|).

    Positive => contralateral more negative (stronger ERD on contra side).
    Both inputs are logratio (ERD = negative).
    """
    denom = abs(contra_log) + abs(ipsi_log) + 1e-9
    return (ipsi_log - contra_log) / denom


def classify_pattern(peak_zone, li_c3c4, l_motor_mean_log, r_motor_mean_log):
    """One-line qualitative label for a session's MI-band pattern.

    Uses logratio inputs for the zone means (negative = ERD).
    """
    both_neg = (l_motor_mean_log < 0) and (r_motor_mean_log < 0)
    if peak_zone == "L-motor" and li_c3c4 > 0.15:
        return "classical contralateral"
    if peak_zone == "R-motor" and li_c3c4 < -0.15:
        return "ipsilateral"
    if both_neg and abs(li_c3c4) <= 0.15:
        return "bilateral"
    if l_motor_mean_log > 0 and r_motor_mean_log > 0:
        return "no ERD (ERS or drift)"
    return "non-classical / mixed"


# ======================================================================
# Session evaluation
# ======================================================================

def evaluate_session(subject, session):
    """Run the shared pipeline and extract scalar metrics."""
    epochs, raw, event_dict, meta = viz.load_and_preprocess_session(
        subject, session, prompt_selection=False
    )
    tfr_data = viz.compute_tfr_epochs(epochs)

    result = {
        "subject": subject,
        "session": session,
        "dropped_channels": meta.get("dropped_channels", []),
        "focal_electrodes_used": meta.get("focal_electrodes_used", []),
        "channels": list(raw.ch_names),
        "n_kept_rest": len(epochs["100"]) if "100" in epochs.event_id else 0,
        "n_kept_mi":   len(epochs["200"]) if "200" in epochs.event_id else 0,
        "bands": {},
    }

    for band_name, lo, hi in BANDS_TO_REPORT:
        band_out = {"markers": {}}
        for marker in ["100", "200"]:
            if marker not in tfr_data:
                continue
            tfr = tfr_data[marker]
            mean_log, _ = per_ch_scalar(tfr, lo, hi, *SCALAR_WINDOW)
            mean_pct = logratio_to_percent(mean_log)
            ch_names = tfr.ch_names
            per_ch_log = dict(zip(ch_names, mean_log))
            per_ch_pct = dict(zip(ch_names, mean_pct))
            sorted_by_pct = sorted(per_ch_pct.items(), key=lambda kv: kv[1])
            band_out["markers"][marker] = {
                "ch_names":    ch_names,
                "per_ch_log":  per_ch_log,
                "per_ch_pct":  per_ch_pct,
                "sorted":      sorted_by_pct,
            }
        result["bands"][band_name] = band_out

    return result


# ======================================================================
# Report formatting
# ======================================================================

def _zone_mean_log(per_ch_log, zone_chans):
    vals = [per_ch_log[c] for c in zone_chans if c in per_ch_log]
    return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)


def format_report(r):
    L = []
    L.append(f"## {r['subject']} / {r['session']}")
    L.append("")
    L.append(
        f"**Config**: spatial={viz.SPATIAL_FILTER!r} | "
        f"temporal=IIR {viz.BROADBAND_LOW}-{viz.BROADBAND_HIGH}Hz | "
        f"reject={viz.EPOCH_REJECT_MODE} {viz.EPOCH_MAX_ABS_UV}µV | "
        f"baseline=({viz.time_start}, {viz.time_start + viz.baseline_period}) | "
        f"scalar_window={SCALAR_WINDOW}"
    )
    L.append(
        f"**Epochs kept**: REST={r['n_kept_rest']}  MI={r['n_kept_mi']}  | "
        f"**auto-dropped**: {r['dropped_channels'] or 'none'}  | "
        f"**n_chans**: {len(r['channels'])}"
    )
    L.append("")

    for band_name, band_out in r["bands"].items():
        L.append(f"### Band: {band_name.upper()}")
        if "200" not in band_out["markers"]:
            L.append("  _(no MI trials)_")
            continue
        mi   = band_out["markers"]["200"]
        rest = band_out["markers"].get("100")

        # --- Top-10 strongest ERD electrodes for MI ---
        L.append("")
        L.append("**MI (200) — top-10 strongest ERD electrodes:**")
        L.append("")
        L.append("| rank | ch | zone | %ERD |")
        L.append("|---|---|---|---|")
        for i, (ch, pct) in enumerate(mi["sorted"][:10], 1):
            L.append(f"| {i} | {ch} | {zone_of(ch)} | {pct:+.1f}% |")

        # --- REST sanity (top-5 most-negative; should be weak) ---
        if rest is not None:
            L.append("")
            L.append("**REST (100) — top-5 most-negative (should be near 0):**")
            L.append("")
            L.append("| rank | ch | zone | %ERD |")
            L.append("|---|---|---|---|")
            for i, (ch, pct) in enumerate(rest["sorted"][:5], 1):
                L.append(f"| {i} | {ch} | {zone_of(ch)} | {pct:+.1f}% |")

        # --- Lateralization for matched pairs (right-arm MI => + = contra/LEFT) ---
        L.append("")
        L.append("**Lateralization** (right-arm MI; LI>0 = contralateral/LEFT dominant):")
        L.append("")
        L.append("| pair | contra(L) %ERD | ipsi(R) %ERD | LI | verdict |")
        L.append("|---|---|---|---|---|")
        mi_log = mi["per_ch_log"]
        mi_pct = mi["per_ch_pct"]
        li_c3c4 = None
        for contra_ch, ipsi_ch in LAT_PAIRS:
            if contra_ch in mi_log and ipsi_ch in mi_log:
                c_log = mi_log[contra_ch]; i_log = mi_log[ipsi_ch]
                li = lateralization_index(c_log, i_log)
                if contra_ch == "C3":
                    li_c3c4 = li
                verdict = "L-dom" if li > 0.15 else ("R-dom" if li < -0.15 else "~equal")
                L.append(
                    f"| {contra_ch}/{ipsi_ch} | {mi_pct[contra_ch]:+.1f}% | "
                    f"{mi_pct[ipsi_ch]:+.1f}% | {li:+.2f} | {verdict} |"
                )

        # --- Peak MI ERD electrode ---
        peak_ch, peak_pct = mi["sorted"][0]
        peak_zone = zone_of(peak_ch)
        L.append("")
        L.append(f"**Peak MI ERD electrode**: `{peak_ch}` ({peak_zone}) at {peak_pct:+.1f}%")

        # --- MI − Rest contrast at L-motor electrodes ---
        if rest is not None:
            rest_pct = rest["per_ch_pct"]
            L.append("")
            L.append("**MI − Rest contrast at L-motor electrodes:**")
            L.append("")
            L.append("| ch | MI %ERD | REST %ERD | Δ(MI−REST) |")
            L.append("|---|---|---|---|")
            for ch in ZONES["L-motor"]:
                if ch in mi_pct and ch in rest_pct:
                    d = mi_pct[ch] - rest_pct[ch]
                    L.append(f"| {ch} | {mi_pct[ch]:+.1f}% | {rest_pct[ch]:+.1f}% | {d:+.1f}% |")

        # --- Zone-aggregate MI ERD (mean across zone's present channels) ---
        L.append("")
        L.append("**Zone-aggregate MI ERD** (mean %ERD over each zone's present channels):")
        L.append("")
        L.append("| zone | mean %ERD | n_chans |")
        L.append("|---|---|---|")
        zone_log_means = {}
        for zone, chans in ZONES.items():
            present = [c for c in chans if c in mi_pct]
            if present:
                mean_pct = float(np.mean([mi_pct[c] for c in present]))
                zone_log_means[zone] = float(np.mean([mi_log[c] for c in present]))
                L.append(f"| {zone} | {mean_pct:+.1f}% | {len(present)} |")

        # --- One-line session classification ---
        l_log = zone_log_means.get("L-motor", float("nan"))
        r_log = zone_log_means.get("R-motor", float("nan"))
        verdict = classify_pattern(peak_zone, li_c3c4 if li_c3c4 is not None else 0.0, l_log, r_log)
        L.append("")
        L.append(
            f"**Session verdict ({band_name})**: {verdict}  "
            f"(peak_zone={peak_zone}, LI_C3/C4={li_c3c4 if li_c3c4 is None else f'{li_c3c4:+.2f}'}, "
            f"L-motor mean={logratio_to_percent(l_log):+.1f}%, "
            f"R-motor mean={logratio_to_percent(r_log):+.1f}%)"
        )
        L.append("")

    return "\n".join(L)


# ======================================================================
# Main
# ======================================================================

def main():
    # Broaden TFR coverage to span both mu and beta in a single pass. This is
    # a module-level mutation of `viz.FREQS` / `viz.N_CYCLES`; it only lasts
    # for this process.
    viz.FREQS = np.linspace(8, 30, int(30 - 8) + 1)
    viz.N_CYCLES = viz.FREQS / 2.0

    header = [
        "# Phase 1 calibration — numerical ERD reader",
        "",
        f"**Subject**: `{SUBJECT}`",
        f"**Sessions**: {SESSIONS}",
        f"**Task**: right-arm MI; contralateral side = LEFT hemisphere",
        f"**Pipeline** (from `visualize_online_data.py` defaults):",
        f"- spatial filter: `{viz.SPATIAL_FILTER}`",
        f"- temporal filter: IIR {viz.BROADBAND_LOW}-{viz.BROADBAND_HIGH} Hz "
        f"(notch {int(60)} Hz)",
        f"- epoch rejection: `{viz.EPOCH_REJECT_MODE}` at {viz.EPOCH_MAX_ABS_UV} µV "
        f"on mu-band ({viz.LOWCUT}-{viz.HIGHCUT} Hz)",
        f"- baseline: ({viz.time_start}, {viz.time_start + viz.baseline_period}) s, "
        f"logratio TFR",
        f"- TFR: multitaper over {viz.FREQS[0]:.0f}-{viz.FREQS[-1]:.0f} Hz, "
        f"scalar window {SCALAR_WINDOW}",
        f"- auto-drop bad channels: {viz.AUTO_DROP_BAD_CHANNELS}",
        "",
        "---",
        "",
    ]
    all_chunks = ["\n".join(header)]

    for sess in SESSIONS:
        print(f"\n{'#'*72}")
        print(f"# Processing {SUBJECT} / {sess}")
        print(f"{'#'*72}")
        r = evaluate_session(SUBJECT, sess)
        report = format_report(r)
        print(report)
        all_chunks.append(report)
        all_chunks.append("\n---\n")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(all_chunks))
    print(f"\nReport written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
