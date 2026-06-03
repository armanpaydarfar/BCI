#!/usr/bin/env python3
"""Investigation A — motor15-restricted input montage before CAR (CLIN_SUBJ_006 only).

Hypothesis: SUBJ_006's pre-CAR set (verified: 27 ch after drop_fp, no auto-
drop fires; non-motor15 = F7,F3,Fz,F4,F8,FC5,FC6,T7,T8,O1,Oz,O2) carries
EMG/breathing-alpha/eye residue that the CAR subtracts from every motor
channel. Test: drop non-motor15 channels BEFORE CAR, then run the rest of
the canonical ERD pipeline unchanged, and score with evaluate_erd_quality.

Three variants per session:
  V1 canonical_full        — sanity baseline; must match
                              ~/Pictures/clin_analysis/erd_refined/
                              erd_quality_scores_car.csv rows for SUBJ_006
                              to 4 dp.
  V2 motor15_input + bilat — drop non-motor15 before CAR; bilat row
                              averages ERD% over BILATERAL_MOTOR_CLUSTER
                              (8 ch). The "publication line".
  V3 motor15_input + m15   — drop non-motor15 before CAR; bilat row
                              averages ERD% over the full motor15 set
                              (15 ch — matches the deployed decoder
                              substrate).

For V2 and V3 the contra-4 and ipsi-4 cluster traces use canonical cluster
membership but motor15-input CAR'd data (so D5 lateralization remains
honest and comparable across variants).

Trial rejection (Analyze_clinical_erd_refined._reject_artifact_trials)
uses BILATERAL_MOTOR_CLUSTER as the substrate — present in all three
variants — so the rejection mask is consistent across variants for a
given session and the canonical contract is honored.

The scorer's _NOMINAL_CLUSTER_SIZE["bilat"] = 8 (evaluate_erd_quality.py:75)
means V3's D7 channel-retention ramp caps at 1.0 (15/8 clamped to 1.0 —
evaluate_erd_quality._d7_retention:403), so D7 is at worst inflated, which
biases V3 favorable on retention only. D1/D2/D3/D4/D6/D8/S are unaffected.

Outputs (scratch only — no canonical edits, no commits):
  C:\\Users\\arman\\Pictures\\clin_analysis_subj006_motor15_input\\
    per_trial/
      CLIN_SUBJ_006_S00NONLINE_V1.npz   (canonical_full)
      CLIN_SUBJ_006_S00NONLINE_V2.npz   (motor15_input + bilat-8)
      CLIN_SUBJ_006_S00NONLINE_V3.npz   (motor15_input + motor15-15)
    erd_quality_scores_V1.csv / .json
    erd_quality_scores_V2.csv / .json
    erd_quality_scores_V3.csv / .json
    sanity_v1_vs_canonical.txt          (per-(session,cluster) diff)
    investigation_a_report.txt          (deliverable: numbers table)
"""

from __future__ import annotations

import gc
import sys
import time
import warnings
from pathlib import Path

import mne
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SWEEP_DIR = _REPO_ROOT / "exploration" / "preprocessing_sweep"
for _p in (str(_REPO_ROOT), str(_SWEEP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Canonical imports (read-only use; no edits to these files)
# ----------------------------------------------------------------------

from sweep_phase2_round2 import (  # noqa: E402
    apply_blink_removal, apply_spatial_filter,
    BB_HI, BB_LO, FREQS, ICA_HP_HZ, MU_HI, MU_LO,
    N_CYCLES, NOTCH, PAD_TFR, REJECT_MAX_ABS_UV, TRIAL_WIN,
    load_raw_cached,
)
from sweep_phase3_validation import (  # noqa: E402
    _pick_dominant_bad_channel_max_abs,
    AUTO_DROP_DOMINANCE_FRAC, AUTO_DROP_MAX_CHANNELS,
    AUTO_DROP_MAX_ITERS, AUTO_DROP_REJECT_FRAC,
)
from Analyze_clinical_erd_refined import (  # noqa: E402
    CONFIG_A_DISPLAY_BASELINE,
    _cluster_timecourse, _logratio_to_pct,
    _reject_artifact_trials, _write_per_trial_npz,
)
# MOTOR_CHANNEL_NAMES — pulled from config.py:26 directly. The canonical
# motor15 list is also defined in Analyze_eds_topoplot_CLIN.py:111-114 with
# the same values, but importing that module pulls in a transitive helper
# that requires DATA_DIR; config.py is the original source.
from config import MOTOR_CHANNEL_NAMES  # noqa: E402
from exploration.clinical_analysis._helpers import (  # noqa: E402
    BILATERAL_MOTOR_CLUSTER, CONTRA_MOTOR_CLUSTER, IPSI_MOTOR_CLUSTER,
)
from evaluate_erd_quality import score_dir  # noqa: E402


# ----------------------------------------------------------------------
# Run config
# ----------------------------------------------------------------------

SUBJECT = "CLIN_SUBJ_006"
SESSIONS = [f"S{n:03d}ONLINE" for n in range(1, 6)]

OUT_DIR = Path(r"C:\Users\arman\Pictures\clin_analysis_subj006_motor15_input")
PER_TRIAL_DIR = OUT_DIR / "per_trial"
FIGS_DIR = OUT_DIR / "figs"  # reserved for follow-ups if needed
for d in (OUT_DIR, PER_TRIAL_DIR, FIGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

MI_MARKER = "200"
REST_MARKER = "100"
MOTOR15 = list(MOTOR_CHANNEL_NAMES)


# ----------------------------------------------------------------------
# Preprocess + TFR (mirrors generate_plots_config_a.preprocess_and_tfr
# at generate_plots_config_a.py:87-165, with an optional pre-CAR channel
# restriction inserted right before apply_spatial_filter)
# ----------------------------------------------------------------------

def preprocess_and_tfr_optional_pick(subject, session, config,
                                     pre_car_pick=None):
    """Run Config A preprocess + TFR. If `pre_car_pick` is a channel-name
    list, drop everything NOT in it from raw_bb right after the auto-drop
    loop and right before apply_spatial_filter.

    Mirrors generate_plots_config_a.preprocess_and_tfr:87-165 line-for-line
    except for the optional pick step.
    """
    raw, events, event_dict = load_raw_cached(subject, session)
    raw_bb = raw.copy()
    raw_1hz = raw.copy()

    # gpc:92-93
    raw_bb.notch_filter(NOTCH, method="iir", verbose=False)
    raw_bb.filter(l_freq=BB_LO, h_freq=BB_HI, method="iir", verbose=False)
    # gpc:95-97
    if config["blink_removal"] == "ica_blink_1hz":
        raw_1hz.notch_filter(NOTCH, method="iir", verbose=False)
        raw_1hz.filter(l_freq=ICA_HP_HZ, h_freq=BB_HI, method="iir",
                       verbose=False)
    # gpc:99
    raw_bb, _ = apply_blink_removal(raw_bb, raw_1hz, config["blink_removal"])

    dropped = []
    iters = 0
    # gpc:108  trial_win override (CONFIG_A_DISPLAY_BASELINE sets (-1, 5))
    t0, t1 = config.get("trial_win", TRIAL_WIN)
    # gpc:109-134  auto-drop loop (verbatim)
    while True:
        iters += 1
        raw_mu = raw_bb.copy()
        raw_mu.filter(l_freq=MU_LO, h_freq=MU_HI, method="iir", verbose=False)
        epoch_kw = dict(
            event_id=event_dict,
            tmin=t0 - PAD_TFR, tmax=t1 + PAD_TFR,
            baseline=None, detrend=1, preload=True, verbose=False,
        )
        epochs_mu = mne.Epochs(raw_mu, events, reject=None, flat=None,
                               **epoch_kw)
        epochs_bb = mne.Epochs(raw_bb, events, reject=None, flat=None,
                               **epoch_kw)
        mu_data = epochs_mu.get_data()
        mask = np.max(np.abs(mu_data), axis=(1, 2)) <= REJECT_MAX_ABS_UV
        good_ix = np.where(mask)[0].tolist()
        bad_ix = np.where(~mask)[0]
        n_att = int(len(events))
        n_kept = int(len(good_ix))
        drop_frac = 1.0 - n_kept / n_att if n_att else 1.0
        if drop_frac < AUTO_DROP_REJECT_FRAC:
            break
        if len(dropped) >= AUTO_DROP_MAX_CHANNELS:
            break
        if iters > AUTO_DROP_MAX_ITERS:
            break
        bad_ch, _ = _pick_dominant_bad_channel_max_abs(
            mu_data, list(epochs_mu.ch_names), bad_ix,
            AUTO_DROP_DOMINANCE_FRAC,
        )
        if bad_ch is None or bad_ch not in raw_bb.ch_names:
            break
        raw_bb = raw_bb.copy().drop_channels([bad_ch])
        dropped.append(bad_ch)

    # gpc:136  pick the kept (mu-amplitude-clean) trials
    epochs = epochs_bb[good_ix]
    if len(epochs) == 0:
        raise RuntimeError("All epochs rejected after auto-drop")

    # === Intervention: pre-CAR channel restriction ============================
    # Drop everything outside the motor15 set, IF requested. This shrinks the
    # set CAR averages over, removing the non-motor channels (F-row, FC5/FC6,
    # T7/T8, O1/Oz/O2 in SUBJ_006's case) that may carry EMG/breathing-alpha/
    # eye residue. The 4 contra channels and the bilat-8 set are subsets of
    # motor15, so the downstream cluster averages still have all canonical
    # channels available.
    if pre_car_pick is not None:
        keep = [c for c in pre_car_pick if c in epochs.ch_names]
        epochs = epochs.copy().pick(keep)
    # ==========================================================================

    # gpc:140
    epochs = apply_spatial_filter(epochs, config["spatial_filter"])

    # gpc:142-157  per-marker TFR
    tfr_trials = {}
    tfr_avg = {}
    spec_bl = config["spectral_baseline"]
    mode = config["baseline_mode"]
    for marker in (REST_MARKER, MI_MARKER):
        if marker not in epochs.event_id or len(epochs[marker]) == 0:
            continue
        tfr = epochs[marker].compute_tfr(
            method="multitaper", freqs=FREQS, n_cycles=N_CYCLES,
            tmin=t0 - PAD_TFR, tmax=t1 + PAD_TFR,
            use_fft=True, return_itc=False, average=False, verbose=False,
        )
        tfr.apply_baseline(baseline=spec_bl, mode=mode, verbose=False)
        tfr.crop(tmin=t0, tmax=t1)
        tfr_trials[marker] = tfr
        tfr_avg[marker] = tfr.average()

    return {
        "tfr_avg": tfr_avg,
        "tfr_trials": tfr_trials,
        "dropped_channels": dropped,
        "n_kept": n_kept,
        "n_attempted": n_att,
    }


# ----------------------------------------------------------------------
# Per-variant trace extraction → npz writer
# ----------------------------------------------------------------------

def _extract_traces_with_bilat_cluster(tfr_trials, dropped_channels,
                                       bilat_cluster):
    """Cluster traces for one variant.

    `bilat_cluster` is the cluster list used for the "bilat" key (V1/V2:
    BILATERAL_MOTOR_CLUSTER, V3: MOTOR15). contra/ipsi clusters are always
    the canonical CONTRA_MOTOR_CLUSTER / IPSI_MOTOR_CLUSTER lists. Mirrors
    Analyze_clinical_erd_refined._extract_session_traces:325-365.
    """
    cluster_specs = [
        ("contra_mi",   CONTRA_MOTOR_CLUSTER,    MI_MARKER),
        ("contra_rest", CONTRA_MOTOR_CLUSTER,    REST_MARKER),
        ("bilat_mi",    bilat_cluster,           MI_MARKER),
        ("bilat_rest",  bilat_cluster,           REST_MARKER),
        ("ipsi_mi",     IPSI_MOTOR_CLUSTER,      MI_MARKER),
        ("ipsi_rest",   IPSI_MOTOR_CLUSTER,      REST_MARKER),
    ]
    traces = {"dropped_channels": list(dropped_channels)}
    per_trial: dict = {}
    for key, cluster, marker in cluster_specs:
        res = _cluster_timecourse(tfr_trials, cluster, marker,
                                  return_per_trial=True)
        if res is None:
            traces[key] = None
            continue
        times, mean_pct, low_pct, up_pct, n, present, ptp = res
        traces[key] = (times, mean_pct, low_pct, up_pct, n, present)
        per_trial[key] = {
            "per_trial_pct": ptp,
            "times": times,
            "channels_used": present,
        }
    traces["per_trial"] = per_trial
    return traces


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    t_start = time.time()
    print(f"[setup] subject={SUBJECT} sessions={SESSIONS}")
    print(f"[setup] out_dir={OUT_DIR}")
    print(f"[setup] motor15 (n={len(MOTOR15)}): {MOTOR15}")

    for sess in SESSIONS:
        print(f"\n=== {SUBJECT} / {sess} ===")
        t_sess = time.time()

        # ----------------------------------------------------------------
        # V1 canonical_full
        # ----------------------------------------------------------------
        print(f"  [V1] canonical_full preprocessing...")
        out_v1 = preprocess_and_tfr_optional_pick(
            SUBJECT, sess, CONFIG_A_DISPLAY_BASELINE, pre_car_pick=None,
        )
        rej_v1 = _reject_artifact_trials(out_v1["tfr_trials"])
        n_after_v1 = sum(int(t.data.shape[0])
                         for t in out_v1["tfr_trials"].values())
        traces_v1 = _extract_traces_with_bilat_cluster(
            out_v1["tfr_trials"], out_v1["dropped_channels"],
            BILATERAL_MOTOR_CLUSTER,
        )
        _write_per_trial_npz(
            PER_TRIAL_DIR / f"{SUBJECT}_{sess}_V1.npz",
            SUBJECT, sess, traces_v1,
            {
                "n_attempted": out_v1["n_attempted"],
                "n_kept": out_v1["n_kept"],
                "n_after_reject": n_after_v1,
                "dropped_channels": out_v1["dropped_channels"],
            },
        )
        print(f"    n_kept={out_v1['n_kept']}/{out_v1['n_attempted']} "
              f"n_after_reject={n_after_v1} "
              f"reject_report={dict(rej_v1)}")
        del out_v1, traces_v1
        gc.collect()

        # ----------------------------------------------------------------
        # V2 + V3 share the motor15-input TFR pass
        # ----------------------------------------------------------------
        print(f"  [V2+V3] motor15_input preprocessing...")
        out_m15 = preprocess_and_tfr_optional_pick(
            SUBJECT, sess, CONFIG_A_DISPLAY_BASELINE,
            pre_car_pick=MOTOR15,
        )
        # One rejection pass (mutates in place); both extractions consume the
        # same cleaned trial set. Rejection substrate = BILATERAL_MOTOR_CLUSTER
        # (Analyze_clinical_erd_refined._reject_artifact_trials:210), which is
        # a subset of motor15 and present.
        rej_m15 = _reject_artifact_trials(out_m15["tfr_trials"])
        n_after_m15 = sum(int(t.data.shape[0])
                          for t in out_m15["tfr_trials"].values())

        # V2: bilat key uses BILATERAL_MOTOR_CLUSTER (8 ch).
        traces_v2 = _extract_traces_with_bilat_cluster(
            out_m15["tfr_trials"], out_m15["dropped_channels"],
            BILATERAL_MOTOR_CLUSTER,
        )
        _write_per_trial_npz(
            PER_TRIAL_DIR / f"{SUBJECT}_{sess}_V2.npz",
            SUBJECT, sess, traces_v2,
            {
                "n_attempted": out_m15["n_attempted"],
                "n_kept": out_m15["n_kept"],
                "n_after_reject": n_after_m15,
                "dropped_channels": out_m15["dropped_channels"],
            },
        )

        # V3: bilat key uses MOTOR15 (15 ch).
        traces_v3 = _extract_traces_with_bilat_cluster(
            out_m15["tfr_trials"], out_m15["dropped_channels"],
            MOTOR15,
        )
        _write_per_trial_npz(
            PER_TRIAL_DIR / f"{SUBJECT}_{sess}_V3.npz",
            SUBJECT, sess, traces_v3,
            {
                "n_attempted": out_m15["n_attempted"],
                "n_kept": out_m15["n_kept"],
                "n_after_reject": n_after_m15,
                "dropped_channels": out_m15["dropped_channels"],
            },
        )
        print(f"    motor15-input n_kept={out_m15['n_kept']}/"
              f"{out_m15['n_attempted']} n_after_reject={n_after_m15} "
              f"reject_report={dict(rej_m15)}")
        del out_m15, traces_v2, traces_v3
        gc.collect()

        print(f"  ({time.time() - t_sess:.1f}s)")

    # ----------------------------------------------------------------------
    # Score each variant via the unchanged evaluate_erd_quality.score_dir
    # ----------------------------------------------------------------------
    print("\n=== Scoring each variant ===")
    import csv, json
    for tag in ("V1", "V2", "V3"):
        rows = score_dir(PER_TRIAL_DIR, variant=f"_{tag}")
        json_path = OUT_DIR / f"erd_quality_scores_{tag}.json"
        csv_path = OUT_DIR / f"erd_quality_scores_{tag}.csv"
        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2)
        # Flatten for CSV (same _CSV_FIELDS as evaluate_erd_quality)
        from evaluate_erd_quality import _CSV_FIELDS
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            w.writeheader()
            for r in rows:
                flat = dict(r)
                flat["channels_used"] = ";".join(r["channels_used"])
                flat["gates_failed"] = ";".join(r["gates_failed"])
                w.writerow({k: ("" if flat.get(k) is None else flat[k])
                            for k in _CSV_FIELDS})
        elig = sum(1 for r in rows if r["eligible"])
        print(f"  {tag}: {len(rows)} rows, {elig} eligible "
              f"→ {csv_path.name}")

    print(f"\nDone ({time.time() - t_start:.1f}s). Outputs at {OUT_DIR}")


if __name__ == "__main__":
    main()
