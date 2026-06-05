#!/usr/bin/env python3
"""Oneshot-study analysis driver for sub-CLIN_SUBJ_007 (proposal pack).

SCRATCH / one-off. NOT a canonical analysis script. Drives the canonical
CLIN analysis functions over the *oneshot* study data, which lives outside
the canonical DATA_DIR (CurrentStudy) so the longitudinal pipeline keeps
running in parallel.

Design constraints (per task brief + CLAUDE.md):
  * No edits to any canonical script (Analyze_clinical_*, Analyze_eds_*,
    evaluate_erd_quality, exploration/preprocessing_sweep/*, Utils/*).
  * No config.py / config_local.py edits; DATA_DIR stays at CurrentStudy.
  * All outputs under ~/Pictures/clin_analysis_oneshot/ (never clin_analysis/).

How the data-root redirect works (runtime injection only):
  The canonical CLIN scripts read the data root via
  `exploration.clinical_analysis._helpers.DATA_DIR` and the output root via
  `_helpers.clin_pictures_root()`. As of commit 17f6508 `_helpers` no longer
  defines DATA_DIR (regression: the four Analyze_clinical_* / the EDS script
  still `from _helpers import DATA_DIR`, so they crash on import at HEAD).
  This wrapper *defines* `_helpers.DATA_DIR` (pointed at the oneshot root) and
  overrides `_helpers.clin_pictures_root` BEFORE importing any canonical
  module, which simultaneously (a) routes around the import regression and
  (b) redirects the data + output roots for this batch only. The XDF loader's
  data root (`sweep_phase2_round2.DATA_DIR`, read at call time) and the session
  enumerator (`sweep_phase3_validation.DATA_DIR`) are patched the same way.

Per-run policy (decided 2026-06-03 with Arman):
  The oneshot session is a SINGLE physical XDF containing multiple driver
  "runs" delineated by marker timestamps (not separate per-run XDFs).
    * XDF-based analyses (EDS, ERD-refined 6-panel, ERD topomap strip, and the
      ERD-quality scorecard derived from the ERD NPZ) run WHOLE-SESSION (n=1).
    * Log-based analyses (confusion matrices, decoder metrics, bar dynamics)
      run PER-RUN — each driver launch already wrote its own logs/ONLINE_* dir.

Usage:
  python scratch_oneshot_subj007_analysis.py <task>
  task in {inventory, eds, erd, topostrip, quality, decoder, confusion, bar}
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# Batch constants
# ----------------------------------------------------------------------
# Subject/session are env-selectable so the same driver runs both oneshot
# participants without editing the constant (ONESHOT_SUBJECT=CLIN_SUBJ_008 for
# the second). Default stays CLIN_SUBJ_007.
SUBJECT = os.environ.get("ONESHOT_SUBJECT", "CLIN_SUBJ_007")
SESSION = os.environ.get("ONESHOT_SESSION", "S001ONLINE")
ONESHOT_ROOT = r"C:\Users\arman\Documents\oneshot"
OUT_ROOT = Path(r"C:\Users\arman\Pictures\clin_analysis_oneshot")

# External-facing (shipped report_figures/) subject labels. The oneshot study
# re-numbers its participants 001, 002, ... for publication so the real CLIN
# cohort IDs (007, 008) are not exposed. Internal/diagnostic figures keep the
# real ID. Only task_report uses this map.
REPORT_SUBJECT_LABEL = {
    "CLIN_SUBJ_007": "Subject 001",
    "CLIN_SUBJ_008": "Subject 002",
}

# Per-trial absolute ERD% reject cap. Lowered from the canonical 600% to 200%
# (the G1 outlier threshold) for this artifact-heavy oneshot data, per Arman /
# the cohort-wide cap sweep on 2026-06-03. Applied by passing abs_cap to the
# local rejection helpers below (no canonical edit; the canonical default of
# 600% is never reached for any ERD-derived output here).
ERD_ABS_CAP = 200.0


# Canonical rejection constants (Analyze_clinical_erd_refined.py:91,98). Held
# as literals so the cluster-matched helper below can default to them without a
# canonical import at module-load time (canonical imports must follow
# _inject_roots). They mirror TRIAL_REJECT_Z / TRIAL_REJECT_MAX_FRAC.
REJECT_Z = 5.0
REJECT_MAX_FRAC = 0.50


def _reject_artifact_trials_for_cluster(tfr_trials, cluster_chs,
                                        reject_z=REJECT_Z,
                                        max_frac=REJECT_MAX_FRAC,
                                        abs_cap=ERD_ABS_CAP):
    """Cluster-substrate-parameterized re-implementation of
    Analyze_clinical_erd_refined._reject_artifact_trials (file:line 158-263).

    Returns a NEW dict of EpochsTFR (does NOT mutate the input) plus a
    per-marker report. Logic mirrors the canonical helper line-for-line,
    except `present` is filtered against the supplied `cluster_chs` instead of
    the hardcoded BILATERAL_MOTOR_CLUSTER (canonical file:line 210).

    This is Fix 1 from notes-to-oneshot-agent-2026-06-03.md: each cluster
    (contra/bilat/ipsi) is cleaned against its OWN channel pool, so a
    focal-spike trial that survives the 8-channel bilateral average but blows
    past the cap on the 4-channel contra/ipsi average is dropped from that
    cluster's pool — matching the scorer's per-cluster G1 gate, which reads
    each cluster's own per-trial peaks (evaluate_erd_quality.py:535,587-600).
    The canonical helper used the bilateral substrate for every cluster, so
    G1 kept firing on contra/ipsi even when the bilateral cleaner said the
    session was clean. Local helper, not a canonical edit.
    """
    from Analyze_clinical_erd_refined import _logratio_to_pct, MU_HI, MU_LO
    new_trials = {}
    report = {}
    for marker, tfr in tfr_trials.items():
        n_before = int(tfr.data.shape[0])
        info = {"n_before": n_before, "n_dropped": 0,
                "kept": True, "over_gate": False}
        report[marker] = info
        new_trials[marker] = tfr  # default: no-op (same object; read-only use)
        if reject_z <= 0 or n_before < 4:
            continue
        present = [c for c in cluster_chs if c in tfr.ch_names]
        if not present:
            continue
        ch_idxs = [tfr.ch_names.index(c) for c in present]
        fmask = (tfr.freqs >= MU_LO) & (tfr.freqs <= MU_HI)
        pct = _logratio_to_pct(
            tfr.data[:, ch_idxs][:, :, fmask],
        ).mean(axis=(1, 2))
        tmask = tfr.times >= 0.0
        scalar = np.max(np.abs(pct[:, tmask]), axis=1)  # peak |ERD%| / trial
        abs_drop = scalar > abs_cap
        med = np.median(scalar)
        mad = np.median(np.abs(scalar - med))
        if mad > 0:
            z = (scalar - med) / (1.4826 * mad)
            z_drop = np.abs(z) > reject_z
        else:
            z_drop = np.zeros_like(scalar, dtype=bool)
        drop = z_drop | abs_drop
        n_drop = int(drop.sum())
        if n_drop == 0:
            continue
        max_drop = int(np.floor(max_frac * n_before))
        if n_drop > max_drop:
            info["over_gate"] = True
            worst = np.argsort(scalar)[::-1][:max_drop]
            keep_mask = np.ones(n_before, dtype=bool)
            keep_mask[worst] = False
            n_drop = max_drop
        else:
            keep_mask = ~drop
        if n_drop == 0:
            continue
        new_trials[marker] = tfr[np.where(keep_mask)[0]]
        info["n_dropped"] = n_drop
    return new_trials, report


def _extract_cluster_matched_traces(tfr_base, dropped_channels,
                                    abs_cap=ERD_ABS_CAP):
    """Cluster-matched rejection + per-cluster trace extraction (Fix 1).

    For each cluster in {contra, bilat, ipsi}, re-runs artifact rejection with
    THAT cluster as the substrate and extracts its traces from THAT cluster's
    cleaned pool, so every <cluster>_<marker> key carries its own n_trials.
    Returns (traces, per_cluster_report): `traces` matches the canonical
    _extract_session_traces schema (consumed verbatim by _write_per_trial_npz
    and _plot_subject_6panel); per_cluster_report is {cluster: {marker: info}}.
    Mirrors scratch_cohort_cap_sweep._extract_cluster_matched_traces.
    """
    from Analyze_clinical_erd_refined import (
        BILATERAL_MOTOR_CLUSTER, CONTRA_MOTOR_CLUSTER, IPSI_MOTOR_CLUSTER,
        MI_MARKER, REST_MARKER, _cluster_timecourse,
    )
    cluster_specs = [
        ("contra", CONTRA_MOTOR_CLUSTER),
        ("bilat",  BILATERAL_MOTOR_CLUSTER),
        ("ipsi",   IPSI_MOTOR_CLUSTER),
    ]
    traces = {"dropped_channels": list(dropped_channels)}
    per_trial: dict = {}
    per_cluster_report: dict = {}
    for cluster_label, cluster_chs in cluster_specs:
        tfr_cluster, rej = _reject_artifact_trials_for_cluster(
            tfr_base, cluster_chs, abs_cap=abs_cap,
        )
        per_cluster_report[cluster_label] = rej
        for marker_label, marker in (("mi", MI_MARKER), ("rest", REST_MARKER)):
            key = f"{cluster_label}_{marker_label}"
            res = _cluster_timecourse(tfr_cluster, cluster_chs, marker,
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
    return traces, per_cluster_report


def _capped_tfr_trials():
    """Config-A per-trial TFR with the ERD_ABS_CAP artifact reject applied.
    Reuses the canonical pipeline + reject verbatim (only the cap value differs).
    Returns (tfr_trials, reject_report, pipe_dict)."""
    from Analyze_clinical_erd_refined import (
        config_a_pipeline, _reject_artifact_trials,
    )
    pipe = config_a_pipeline(SUBJECT, SESSION)
    tfr = pipe["tfr_trials"]
    rep = _reject_artifact_trials(tfr, abs_cap=ERD_ABS_CAP)
    return tfr, rep, pipe

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SWEEP_DIR = _REPO_ROOT / "exploration" / "preprocessing_sweep"
for _p in (str(_REPO_ROOT), str(_SWEEP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _inject_roots():
    """Point the canonical data root at the oneshot tree and the output root
    at clin_analysis_oneshot. MUST run before importing any canonical module.

    Returns the `_helpers` module so callers can introspect if needed.
    """
    import exploration.clinical_analysis._helpers as H

    # (a) Define the name the canonical scripts import (missing at HEAD) and
    #     point it at the oneshot root.
    H.DATA_DIR = ONESHOT_ROOT
    # (b) Redirect every canonical script's output root to clin_analysis_oneshot.
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    H.clin_pictures_root = lambda: OUT_ROOT

    # (c) The XDF loader (sweep_phase2_round2.load_raw_cached) and the session
    #     enumerator (sweep_phase3_validation.enumerate_online_sessions) read
    #     their own module-global DATA_DIR at call time. Patch both, plus config.
    import config
    config.DATA_DIR = ONESHOT_ROOT
    import sweep_phase2_round2 as s2
    s2.DATA_DIR = ONESHOT_ROOT
    import sweep_phase3_validation as s3
    s3.DATA_DIR = ONESHOT_ROOT
    return H


# ----------------------------------------------------------------------
# Per-run log-dir enumeration (substantive runs only)
# ----------------------------------------------------------------------
def _substantive_run_dirs(subject=None, session=None):
    """Return [(run_id, run_dir Path)] for runs whose decoder_output.csv has
    at least one data row, sorted by launch time and labelled R001.. in order.
    Aborted launches (header-only CSV) are dropped. Defaults to the module
    SUBJECT/SESSION; pass a subject to enumerate the other participant (used by
    the two-subject consolidation).
    """
    import pandas as pd
    from Analyze_experiment_logs_cross_subject import find_decoder_csv

    subject = subject or SUBJECT
    session = session or SESSION
    logs_dir = Path(ONESHOT_ROOT) / f"sub-{subject}" / f"ses-{session}" / "logs"
    out = []
    run_dirs = sorted(p for p in logs_dir.iterdir()
                      if p.is_dir() and p.name.startswith("ONLINE_"))
    n = 0
    for rd in run_dirs:
        csv = find_decoder_csv(str(rd))
        if csv is None:
            continue
        df = pd.read_csv(csv)
        if "Trial" not in df.columns or len(df) == 0:
            print(f"  (skip aborted launch {rd.name}: {len(df)} rows)")
            continue
        n += 1
        out.append((f"R{n:03d}", rd))
    return out


# ----------------------------------------------------------------------
# Task: channel inventory at 3 stages (mirrors
# scratch_subj006_motor15_input_sanity.py — whole session)
# ----------------------------------------------------------------------
def task_inventory():
    import warnings
    import mne
    mne.set_log_level("ERROR")
    warnings.filterwarnings("ignore")
    _inject_roots()

    from sweep_phase2_round2 import (
        apply_blink_removal, BB_HI, BB_LO, MU_HI, MU_LO, NOTCH, PAD_TFR,
        REJECT_MAX_ABS_UV, TRIAL_WIN, ZONES, load_raw_cached,
    )
    from sweep_phase3_validation import (
        _pick_dominant_bad_channel_max_abs,
        AUTO_DROP_DOMINANCE_FRAC, AUTO_DROP_MAX_CHANNELS, AUTO_DROP_MAX_ITERS,
        AUTO_DROP_REJECT_FRAC,
    )

    T_ROW = set(ZONES["Temporal"])
    lines = []

    def w(s=""):
        print(s)
        lines.append(s)

    w(f"Channel-inventory sanity check — {SUBJECT} / {SESSION} (oneshot)")
    w("=" * 72)
    w("Stage (a): raw.ch_names after sweep_phase2_round2.load_raw_cached")
    w("Stage (b): after apply_blink_removal(method='drop_fp') on broadband raw")
    w("Stage (c): after auto-drop loop, pre apply_spatial_filter")
    w(f"T-row set (ZONES['Temporal']): {sorted(T_ROW)}")
    w("")

    raw, events, event_dict = load_raw_cached(SUBJECT, SESSION)
    chs_a = list(raw.ch_names)
    w(f"  (a) n={len(chs_a)}  ch_names={chs_a}")
    w(f"      T-row present: {sorted(c for c in chs_a if c in T_ROW)}")

    raw_b = raw.copy()
    raw_b.notch_filter(NOTCH, method="iir", verbose=False)
    raw_b.filter(l_freq=BB_LO, h_freq=BB_HI, method="iir", verbose=False)
    raw_b, _info = apply_blink_removal(raw_b, raw_b.copy(), "drop_fp")
    chs_b = list(raw_b.ch_names)
    w(f"  (b) n={len(chs_b)}  ch_names={chs_b}")
    w(f"      delta (a)->(b) dropped: {[c for c in chs_a if c not in chs_b]}")

    # Stage (c): auto-drop loop (generate_plots_config_a.preprocess_and_tfr:101-134)
    dropped, iters = [], 0
    t0, t1 = TRIAL_WIN
    raw_c = raw_b.copy()
    n_kept = n_att = 0
    while True:
        iters += 1
        raw_mu = raw_c.copy()
        raw_mu.filter(l_freq=MU_LO, h_freq=MU_HI, method="iir", verbose=False)
        epochs_mu = mne.Epochs(
            raw_mu, events, event_id=event_dict,
            tmin=t0 - PAD_TFR, tmax=t1 + PAD_TFR,
            baseline=None, detrend=1, preload=True, verbose=False,
            reject=None, flat=None,
        )
        mu_data = epochs_mu.get_data()
        mask = np.max(np.abs(mu_data), axis=(1, 2)) <= REJECT_MAX_ABS_UV
        bad_ix = np.where(~mask)[0]
        n_att = int(len(events))
        n_kept = int(np.sum(mask))
        drop_frac = 1.0 - n_kept / n_att if n_att else 1.0
        if drop_frac < AUTO_DROP_REJECT_FRAC:
            break
        if len(dropped) >= AUTO_DROP_MAX_CHANNELS or iters > AUTO_DROP_MAX_ITERS:
            break
        bad_ch, _ = _pick_dominant_bad_channel_max_abs(
            mu_data, list(epochs_mu.ch_names), bad_ix, AUTO_DROP_DOMINANCE_FRAC,
        )
        if bad_ch is None or bad_ch not in raw_c.ch_names:
            break
        raw_c = raw_c.copy().drop_channels([bad_ch])
        dropped.append(bad_ch)
    chs_c = list(raw_c.ch_names)
    w(f"  (c) n={len(chs_c)}  ch_names={chs_c}")
    w(f"      auto-drop fired on: {dropped or '—'}  (n_kept={n_kept}/{n_att})")
    w("")
    w("Summary: pre-CAR/spatial channel set = stage (c); spatial filter "
      "averages/derives over that set.")

    out = OUT_ROOT / f"channel_inventory_{SUBJECT}_{SESSION}.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote: {out}")


# ----------------------------------------------------------------------
# Task: EDS per-class topomap (whole session)
# ----------------------------------------------------------------------
def task_eds():
    _inject_roots()
    import Analyze_eds_topoplot_CLIN as eds
    # Expert pool lives under <root>/sub-<EXPERT_SOURCE_SUBJECT>/training_data.
    # The oneshot tree only contains sub-CLIN_SUBJ_007, whose training_data
    # holds the identical 6-file OG_Right expert pool — point the expert
    # source at it so the expert panel resolves under the oneshot root.
    eds.EXPERT_SOURCE_SUBJECT = SUBJECT
    eds.clin_pictures_root = lambda: OUT_ROOT
    sys.argv = ["eds", "--subjects", SUBJECT, "--no-diff-plot"]
    eds.main()


# ----------------------------------------------------------------------
# Task: ERD-refined 6-panel + per-trial NPZ (whole session), with
# cluster-matched artifact rejection (Fix 1) at cap=ERD_ABS_CAP.
# ----------------------------------------------------------------------
def task_erd():
    """ERD-refined per-trial NPZ + per-subject 6-panel, cluster-matched.

    The canonical Analyze_clinical_erd_refined.main() rejects every cluster on
    the bilateral montage (file:line 210), which biases the contra/ipsi
    per-cluster traces and the NPZ the scorer reads (Fix 1). We cannot edit the
    canonical script, so this drives the canonical building blocks
    (config_a_pipeline, _cluster_timecourse, _write_per_trial_npz,
    _plot_subject_6panel) directly with cluster-matched rejection, mirroring
    exploration/clinical_analysis/scratch_cohort_cap_sweep.py. The 6-panel and
    NPZ replace what erd.main() produced; the n=1 cohort figure and per-trace
    CSV that main() also emitted are dropped (degenerate for a single session).
    """
    _inject_roots()
    import Analyze_clinical_erd_refined as erd
    erd.clin_pictures_root = lambda: OUT_ROOT
    out_dir = OUT_ROOT / "erd_refined"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Variant tag the scorer (task_quality) globs for: a --subjects run tags
    # the NPZ "_car_subj-<NNN>" (CAR spatial filter + subject filter).
    variant_tag = f"_car_subj-{SUBJECT.replace('CLIN_SUBJ_', '')}"

    pipe = erd.config_a_pipeline(SUBJECT, SESSION)
    traces, per_cluster_rej = _extract_cluster_matched_traces(
        pipe["tfr_trials"], pipe.get("dropped_channels", []),
    )
    # Session-level n_after_reject for the NPZ meta (scorer G2 substrate,
    # evaluate_erd_quality.py:479). Use the BILATERAL pool — the most stringent
    # substrate (8-ch averaging dilutes a focal artifact most, so it drops the
    # most survivors), preserving canonical G2 semantics without touching the
    # scorer (notes-to-oneshot-agent-2026-06-03.md Fix 1).
    bilat_rej = per_cluster_rej.get("bilat", {})
    n_after_bilat = sum(
        info["n_before"] - info["n_dropped"] for info in bilat_rej.values()
    ) if bilat_rej else pipe.get("n_kept", 0)

    erd._write_per_trial_npz(
        out_dir / "per_trial" / f"{SUBJECT}_{SESSION}{variant_tag}.npz",
        SUBJECT, SESSION, traces,
        {
            "n_attempted": pipe.get("n_attempted", 0),
            "n_kept": pipe.get("n_kept", 0),
            "n_after_reject": n_after_bilat,
            "dropped_channels": pipe.get("dropped_channels", []),
        },
    )
    erd._plot_subject_6panel(
        SUBJECT, [(SESSION, traces)],
        str(out_dir / f"{SUBJECT}_6panel_mi_rest{variant_tag}.png"),
    )
    for cl in ("contra", "bilat", "ipsi"):
        mi = per_cluster_rej.get(cl, {}).get("200", {})
        nb, nd = mi.get("n_before", 0), mi.get("n_dropped", 0)
        print(f"  {cl}_mi: kept {nb}-{nd}={nb - nd}"
              f"{'  [G2 over_gate]' if mi.get('over_gate') else ''}")
    print(f"  n_after_reject (bilat substrate) = {n_after_bilat}")
    print(f"Wrote cluster-matched NPZ + 6-panel (cap={ERD_ABS_CAP:.0f}%) "
          f"-> {out_dir}")


# ----------------------------------------------------------------------
# Task: ERD topomap strip at 8 windows (whole session, ERD_ABS_CAP, mean)
# ----------------------------------------------------------------------
def task_topostrip():
    _inject_roots()
    import generate_plots_config_a as gp
    from Analyze_clinical_erd_refined import MU_LO, MU_HI
    topo_dir = OUT_ROOT / "erd_topomaps" / "per_session"
    topo_dir.mkdir(parents=True, exist_ok=True)
    # Re-average the ERD_ABS_CAP-cleaned per-trial TFR (mean over trials) and
    # reuse the canonical strip plotter. The topomap is a grand average by
    # construction; the only change vs the canonical pass is the trial set
    # (cap=200 instead of no reject).
    tfr, rep, pipe = _capped_tfr_trials()
    tfr_avg = {mk: t.average() for mk, t in tfr.items()}
    rest = tfr_avg.get("100")
    mi = tfr_avg.get("200")
    vlim = gp._compute_dynamic_vlim(rest, mi)
    if rest is not None:
        gp._plot_topo_strip(
            rest, MU_LO, MU_HI, vlim,
            f"ERD/ERS Topomaps – Rest | {SUBJECT}/{SESSION} | "
            f"cap={ERD_ABS_CAP:.0f}% (n={rest.nave})",
            str(topo_dir / f"{SUBJECT}_{SESSION}_rest.png"))
    if mi is not None:
        gp._plot_topo_strip(
            mi, MU_LO, MU_HI, vlim,
            f"ERD/ERS Topomaps – Right Arm MI | {SUBJECT}/{SESSION} | "
            f"cap={ERD_ABS_CAP:.0f}% (n={mi.nave})",
            str(topo_dir / f"{SUBJECT}_{SESSION}_mi.png"))
    print(f"reject report: {rep}")
    print(f"Wrote capped topomap strips (mean, cap={ERD_ABS_CAP:.0f}%).")


# ----------------------------------------------------------------------
# Task: spatial-filter comparison for the topomap strips. The report default
# is CAR (a low-pass-ish reference that leaves broad modulations); CSD (the MNE
# spherical-spline surface Laplacian) and Hjorth (k=4 nearest-neighbour
# Laplacian) are spatial high-pass filters expected to sharpen / focalise the
# ERD. Regenerates MI + REST strips for BOTH subjects under all three filters
# into erd_topomaps/spatial_compare/ so the focalisation can be compared before
# deciding whether to switch the report away from CAR. Diagnostic only — does
# not touch the shipped report_figures/.
# ----------------------------------------------------------------------
def task_topocompare():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne
    _inject_roots()
    import generate_plots_config_a as gp
    from Analyze_clinical_erd_refined import (
        MU_LO, MU_HI, config_a_pipeline, _reject_artifact_trials,
        CONFIG_A_DISPLAY_BASELINE,
    )
    out_dir = OUT_ROOT / "erd_topomaps" / "spatial_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = [("CLIN_SUBJ_007", "Subject 001"),
                ("CLIN_SUBJ_008", "Subject 002")]
    filters = ("car", "csd", "hjorth")
    # Sustained-desync window for the single-panel focalisation comparison
    # (the MI mu trough plateaus over ~1-3 s in both subjects' timecourses).
    FOCAL_WIN = (1.0, 3.0)
    focal = {}  # (sid, sf) -> (scalar_map[n_ch], info)
    orig_sf = CONFIG_A_DISPLAY_BASELINE["spatial_filter"]
    try:
        for sid, slabel in subjects:
            for sf in filters:
                CONFIG_A_DISPLAY_BASELINE["spatial_filter"] = sf
                pipe = config_a_pipeline(sid, SESSION)
                tfr = pipe["tfr_trials"]
                # Same bilateral-substrate cap=200 reject as task_topostrip, so
                # the only difference vs the shipped strip is the spatial filter.
                _reject_artifact_trials(tfr, abs_cap=ERD_ABS_CAP)
                tfr_avg = {mk: t.average() for mk, t in tfr.items()}
                rest, mi = tfr_avg.get("100"), tfr_avg.get("200")
                vlim = gp._compute_dynamic_vlim(rest, mi)
                if mi is not None:
                    gp._plot_topo_strip(
                        mi, MU_LO, MU_HI, vlim,
                        f"ERD/ERS – MI | {slabel} | {sf.upper()} | "
                        f"cap={ERD_ABS_CAP:.0f}% (n={mi.nave})",
                        str(out_dir / f"{sid}_{SESSION}_mi_{sf}.png"))
                    fmask = (mi.freqs >= MU_LO) & (mi.freqs <= MU_HI)
                    tmask = (mi.times >= FOCAL_WIN[0]) & (mi.times <= FOCAL_WIN[1])
                    scalar = mi.data[:, fmask][:, :, tmask].mean(axis=(1, 2))
                    focal[(sid, sf)] = (scalar, mi.info)
                if rest is not None:
                    gp._plot_topo_strip(
                        rest, MU_LO, MU_HI, vlim,
                        f"ERD/ERS – Rest | {slabel} | {sf.upper()} | "
                        f"cap={ERD_ABS_CAP:.0f}% (n={rest.nave})",
                        str(out_dir / f"{sid}_{SESSION}_rest_{sf}.png"))
                print(f"  {sid} {sf}: MI n={getattr(mi, 'nave', 0)} "
                      f"rest n={getattr(rest, 'nave', 0)}")
    finally:
        CONFIG_A_DISPLAY_BASELINE["spatial_filter"] = orig_sf

    # Focalisation grid: rows = subject, cols = filter, one large MI topomap
    # (mean mu logratio over FOCAL_WIN). Per-panel symmetric vlim (each filter
    # scaled to its own |max|) so the comparison is of spatial EXTENT, not
    # amplitude — the Laplacians (CSD/Hjorth) reduce absolute magnitude, which
    # a shared scale would read as "weaker" rather than "more focal".
    fig, axes = plt.subplots(len(subjects), len(filters),
                             figsize=(4.2 * len(filters), 4.0 * len(subjects)),
                             squeeze=False)
    for ri, (sid, slabel) in enumerate(subjects):
        for ci, sf in enumerate(filters):
            ax = axes[ri][ci]
            if (sid, sf) not in focal:
                ax.axis("off")
                continue
            scalar, info = focal[(sid, sf)]
            vmax = float(np.nanmax(np.abs(scalar))) or 1.0
            im, _ = mne.viz.plot_topomap(
                scalar, info, axes=ax, cmap="RdBu_r", show=False,
                vlim=(-vmax, vmax), contours=6, sensors=True)
            fig.colorbar(im, ax=ax, shrink=0.7, label="logratio")
            ax.set_title(f"{slabel} · {sf.upper()}\n(|max|={vmax:.2f})",
                         fontsize=10)
    fig.suptitle(f"MI μ ERD focalisation by spatial filter "
                 f"(mean {FOCAL_WIN[0]:g}-{FOCAL_WIN[1]:g} s, "
                 f"per-panel vlim) — blue = desync", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    grid_png = out_dir / "focalisation_grid_mi.png"
    fig.savefig(grid_png, dpi=150)
    plt.close(fig)
    print(f"Wrote spatial-filter comparison strips + {grid_png.name} -> {out_dir}")


# ----------------------------------------------------------------------
# Task: ERD mean ± SE timecourses, 6-panel, in BOTH logratio and percent
# units (cap=ERD_ABS_CAP). Grand-average (mean across trials) to match the
# topomap; the canonical 6-panel uses median, kept separately for reference.
#
# Fix 1 (cluster-matched rejection): each cluster row is cleaned against its
# OWN channel pool, not the bilateral montage. Fix 2 (logratio baseline
# re-zero): the logratio panels subtract each trial's baseline-window mean so
# the baseline sits at 0 (MNE's logratio leaves a ~-0.05 Jensen offset).
# Both per notes-to-oneshot-agent-2026-06-03.md.
# ----------------------------------------------------------------------
def task_erdmean():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _inject_roots()
    from Analyze_clinical_erd_refined import (
        _logratio_to_pct, MU_LO, MU_HI, MI_MARKER, REST_MARKER,
        CONTRA_MOTOR_CLUSTER, BILATERAL_MOTOR_CLUSTER, IPSI_MOTOR_CLUSTER,
        config_a_pipeline,
    )
    out_dir = OUT_ROOT / "erd_refined"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [("contra", CONTRA_MOTOR_CLUSTER),
            ("bilat", BILATERAL_MOTOR_CLUSTER),
            ("ipsi", IPSI_MOTOR_CLUSTER)]
    cols = [("MI", MI_MARKER), ("REST", REST_MARKER)]

    # Fix 1: one cluster-matched kept-trial pool per cluster (each cleaned on
    # its own channels), computed once and reused for both unit passes.
    pipe = config_a_pipeline(SUBJECT, SESSION)
    tfr_base = pipe["tfr_trials"]
    cluster_pools = {}
    for cname, chans in rows:
        tfr_c, rej = _reject_artifact_trials_for_cluster(
            tfr_base, chans, abs_cap=ERD_ABS_CAP,
        )
        cluster_pools[cname] = (tfr_c, rej)

    def _per_trial_trace(marker, chans, unit, tfr_cluster):
        """(times, per_trial[n,t]) cluster+mu mean trace in 'logratio' or 'pct'.

        For 'logratio', each trial's baseline-window (-1..0 s) mean is
        subtracted (Fix 2): MNE's logratio baseline leaves a Jensen offset
        (mean(log P) < log(mean P)), so the baseline window averages ~-0.05,
        not 0. Re-zeroing is equivalent to a geometric-mean baseline and puts
        the window at 0 by construction per trial. The 'pct' path is
        Jensen-free (log->% is applied per (trial,ch,freq,time) before
        averaging, so the baseline cancels exactly) and needs no re-zero.
        """
        t = tfr_cluster[marker]
        idx = [t.ch_names.index(c) for c in chans if c in t.ch_names]
        fmask = (t.freqs >= MU_LO) & (t.freqs <= MU_HI)
        raw = t.data[:, idx][:, :, fmask]  # n, ch, f, t (logratio)
        if unit == "pct":
            raw = _logratio_to_pct(raw)
        pt = raw.mean(axis=(1, 2))  # n, t
        if unit == "logratio":
            bmask = (t.times >= -1.0) & (t.times <= 0.0)
            pt = pt - pt[:, bmask].mean(axis=1, keepdims=True)
        return t.times, pt

    for unit, ylab in (("logratio", "ERD/ERS (logratio, baseline re-zeroed)"),
                       ("pct", "ERD/ERS %")):
        fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
        for ri, (cname, chans) in enumerate(rows):
            tfr_c, _ = cluster_pools[cname]
            for ci, (clab, marker) in enumerate(cols):
                ax = axes[ri, ci]
                if marker not in tfr_c:
                    ax.set_visible(False)
                    continue
                times, pt = _per_trial_trace(marker, chans, unit, tfr_c)
                n = pt.shape[0]
                mean = pt.mean(axis=0)
                se = pt.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
                ax.axhline(0, color="k", lw=0.6)
                ax.axvline(0, color="grey", lw=0.6, ls=":")
                ax.plot(times, mean, color="tab:blue", lw=1.5)
                ax.fill_between(times, mean - se, mean + se, color="tab:blue", alpha=0.25)
                ax.set_title(f"{cname.capitalize()} ERD — {clab}  (n={n}, mean±SE)",
                             fontsize=10)
                if ci == 0:
                    ax.set_ylabel(ylab)
                if ri == 2:
                    ax.set_xlabel("time (s)")
                ax.grid(True, alpha=0.3)
        rezero = ", baseline re-zeroed" if unit == "logratio" else ""
        fig.suptitle(
            f"{SUBJECT}/{SESSION} — MEAN ERD timecourse ({unit}{rezero}), "
            f"cap={ERD_ABS_CAP:.0f}%  | CAR, mu {MU_LO:g}-{MU_HI:g} Hz, "
            f"baseline (-1,0)s, cluster-matched reject", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        png = out_dir / f"{SUBJECT}_{SESSION}_6panel_MEAN_{unit}_cap{int(ERD_ABS_CAP)}.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"  wrote {png.name}")
    for cname, (_tfr_c, rej) in cluster_pools.items():
        mi = rej.get(MI_MARKER, {})
        print(f"  {cname}: MI kept {mi.get('n_before', 0)}-{mi.get('n_dropped', 0)}"
              f"={mi.get('n_before', 0) - mi.get('n_dropped', 0)}"
              f"{'  [over_gate]' if mi.get('over_gate') else ''}")


# ----------------------------------------------------------------------
# Task: ERD-quality scorecard (whole session — scores the ERD NPZ)
# ----------------------------------------------------------------------
def task_quality():
    # evaluate_erd_quality.py is decoupled from DATA_DIR; drive it as a
    # subprocess pointed at the ERD per_trial NPZ dir produced by task_erd.
    import subprocess
    npz_dir = OUT_ROOT / "erd_refined" / "per_trial"
    out_dir = OUT_ROOT / "erd_refined"
    # task_erd tags the NPZ with the ERD script's variant tag, which for a
    # --subjects run is "_car_subj-<NNN>" (spatial filter + subject filter).
    # The scorer globs *<variant>.npz (endswith), so match that exact tag.
    variant = f"_car_subj-{SUBJECT.replace('CLIN_SUBJ_', '')}"
    cmd = [
        sys.executable, str(_REPO_ROOT / "evaluate_erd_quality.py"),
        "--npz-dir", str(npz_dir),
        "--out-dir", str(out_dir),
        "--variant", variant,
    ]
    env = dict(os.environ, PYTHONIOENCODING="utf-8",
               PYTHONPATH=str(_REPO_ROOT))
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


# ----------------------------------------------------------------------
# Task: per-run confusion matrices
# ----------------------------------------------------------------------
def task_confusion():
    import pandas as pd
    _inject_roots()
    import Analyze_clinical_confusion_matrices as cmmod
    from Analyze_experiment_logs_cross_subject import (
        compute_confusion_matrix_from_csv, find_decoder_csv,
    )
    out_dir = OUT_ROOT / "confusion_matrices"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for run_id, rd in _substantive_run_dirs():
        csv = find_decoder_csv(str(rd))
        df = pd.read_csv(csv)
        df["RunID"] = f"{SUBJECT}__{SESSION}__{rd.name}"
        df["GlobalTrialID"] = df["RunID"].astype(str) + "_" + df["Trial"].astype(str)
        cm = compute_confusion_matrix_from_csv(df)
        if cm is None:
            print(f"  {run_id}: confusion matrix None; skip")
            continue
        decisions_n = int(cm[:, :2].sum())
        correct_n = int(cm[0, 0] + cm[1, 1])
        ambiguous_n = int(cm[:, 2].sum())
        total = int(cm.sum())
        dec_acc = 100.0 * correct_n / decisions_n if decisions_n else float("nan")
        tot_acc = 100.0 * correct_n / total if total else float("nan")
        label = f"{SUBJECT} {SESSION} {run_id}"
        png = out_dir / f"{SUBJECT}_{SESSION}_{run_id}_confusion_matrix.png"
        cmmod._plot_subject_cm(cm, label, 1, 1, str(png))
        print(f"  {run_id}: trials={total} dec_acc={dec_acc:.1f}% "
              f"tot_acc={tot_acc:.1f}% amb={ambiguous_n}")
        summary.append(dict(
            subject=SUBJECT, session=SESSION, run=run_id, trials=total,
            mi_mi=int(cm[0, 0]), mi_rest=int(cm[0, 1]), mi_amb=int(cm[0, 2]),
            rest_mi=int(cm[1, 0]), rest_rest=int(cm[1, 1]), rest_amb=int(cm[1, 2]),
            decisions=decisions_n, correct=correct_n, ambiguous=ambiguous_n,
            dec_acc_pct=round(dec_acc, 2), tot_acc_pct=round(tot_acc, 2),
        ))
    pd.DataFrame(summary).to_csv(
        out_dir / f"{SUBJECT}_{SESSION}_per_run_confusion_summary.csv",
        index=False,
    )
    print(f"Wrote per-run confusion summary ({len(summary)} runs).")


# ----------------------------------------------------------------------
# Task: per-run decoder metrics
# ----------------------------------------------------------------------
def task_decoder():
    import pandas as pd
    _inject_roots()
    import Analyze_clinical_decoder_longitudinal as dec
    from Analyze_experiment_logs_cross_subject import find_decoder_csv

    out_dir = OUT_ROOT / "decoder_perf"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_id, rd in _substantive_run_dirs():
        csv = find_decoder_csv(str(rd))
        df = pd.read_csv(csv)
        df["SubjectID"] = SUBJECT
        df["SessionID"] = SESSION
        df["RunFolder"] = rd.name
        df["RunID"] = f"{SUBJECT}__{SESSION}__{rd.name}"
        df["GlobalTrialID"] = df["RunID"].astype(str) + "_" + df["Trial"].astype(str)
        m = dec._session_metrics([df])
        m = dict(subject=SUBJECT, session=SESSION, run=run_id, **m)
        rows.append(m)
        print(f"  {run_id}: n={m.get('n_total')} kappa={m.get('trial_kappa')}"
              f" NKV={m.get('nkv')} acc_dec={m.get('acc_decided')}")
    summ = pd.DataFrame(rows)
    summ.to_csv(
        out_dir / f"{SUBJECT}_{SESSION}_per_run_decoder_metrics.csv",
        index=False,
    )
    # Per-run figure (analog of the canonical per-subject
    # *_decoder_kappa_nkv_acc_over_sessions.png, x-axis = run not session).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(summ))
    for col, lab in (("trial_kappa", "trial κ"), ("nkv", "NKV"),
                     ("acc_decided", "acc (decided)"),
                     ("acc_inclusive", "acc (inclusive)")):
        ax.plot(x, summ[col], marker="o", label=lab)
    ax.set_xticks(list(x))
    ax.set_xticklabels(summ["run"])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("run")
    ax.set_ylabel("metric")
    ax.set_title(f"{SUBJECT} / {SESSION} — decoder metrics per run "
                 f"(n={int(summ['n_total'].sum())} trials)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{SUBJECT}_{SESSION}_decoder_kappa_nkv_acc_over_runs.png",
        dpi=200)
    plt.close(fig)
    print(f"Wrote per-run decoder metrics + figure ({len(rows)} runs).")


# ----------------------------------------------------------------------
# Task: per-run bar dynamics
# ----------------------------------------------------------------------
def task_bar():
    import pandas as pd
    _inject_roots()
    import Analyze_clinical_bar_dynamics_longitudinal as bar

    out_dir = OUT_ROOT / "bar_dynamics"
    out_dir.mkdir(parents=True, exist_ok=True)
    # _load_session_trials iterates runs internally and tags run_id; aborted
    # (header-only) runs collapse to empty and are skipped.
    df_trials = bar._load_session_trials(SUBJECT, SESSION)
    if df_trials.empty:
        print("  no per-trial bar-dynamics rows produced.")
        return
    # Map ONLINE_* folder names to R001.. in launch order for tagging.
    run_map = {rd.name: rid for rid, rd in _substantive_run_dirs()}
    df_trials["run"] = df_trials["run_id"].map(run_map)
    per_run = bar._per_run_median(df_trials)
    per_run["run"] = per_run["run_id"].map(run_map)
    df_trials.to_csv(
        out_dir / f"{SUBJECT}_{SESSION}_per_trial_bar_dynamics.csv", index=False)
    per_run.to_csv(
        out_dir / f"{SUBJECT}_{SESSION}_per_run_bar_dynamics.csv", index=False)
    for _, r in per_run.sort_values(["run", "Class"]).iterrows():
        print(f"  {r['run']} {r['Class']}: Lean%={r['LeanPct']:.1f} "
              f"TTT={r['TimeToThresh_s']} n={int(r['n_trials'])}")

    # Box-and-whisker over the per-TRIAL distribution: x = ["All", R001..R004],
    # two boxes per group (MI, REST). The whole-session "All" box answers the
    # session-level question; the per-run boxes show within-run spread.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = sorted(r for r in per_run["run"].dropna().unique())
    groups = ["All"] + runs
    colors = {"MI": "tab:orange", "REST": "tab:blue"}

    def _vals(metric, group, cls):
        d = df_trials[df_trials["Class"] == cls]
        if group != "All":
            d = d[d["run"] == group]
        return d[metric].dropna().values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, metric, ylab, ttl in (
            (ax1, "LeanPct", "Lean% (per trial)", "Lean% distribution"),
            (ax2, "TimeToThresh_s", "TTT (s, per trial)",
             "Time-to-threshold distribution")):
        handles = {}
        for gi, g in enumerate(groups):
            for cls, off in (("MI", -0.18), ("REST", +0.18)):
                v = _vals(metric, g, cls)
                if len(v) == 0:
                    continue
                bp = ax.boxplot(
                    v, positions=[gi + off], widths=0.3,
                    patch_artist=True, showfliers=True,
                    medianprops=dict(color="black"),
                    flierprops=dict(marker=".", markersize=4,
                                    markerfacecolor=colors[cls],
                                    markeredgecolor=colors[cls], alpha=0.5))
                for box in bp["boxes"]:
                    box.set(facecolor=colors[cls], alpha=0.55)
                handles[cls] = bp["boxes"][0]
        ax.axvline(0.5, color="grey", lw=0.8, ls="--")  # separate All from runs
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups)
        ax.set_xlabel("session / run")
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.grid(True, axis="y", alpha=0.3)
        if handles:
            ax.legend(handles.values(), handles.keys(), loc="best")
    fig.suptitle(f"{SUBJECT} / {SESSION} — bar dynamics "
                 f"(box = per-trial distribution; n={len(df_trials)} trials)")
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{SUBJECT}_{SESSION}_bar_dynamics_boxplot.png", dpi=200)
    plt.close(fig)
    print(f"Wrote per-run bar dynamics + box-and-whisker figure.")


# ----------------------------------------------------------------------
# Diagnostic: reconcile the ERD topomap (trial-MEAN logratio, no extra
# rejection) vs the ERD-refined 6-panel (trial-MEDIAN ERD%, post artifact
# rejection incl. 600% cap). Both derive from the same per-trial logratio
# TFR; this prints the cluster-level value under each estimator so the
# magnitude gap is quantified, not asserted.
# ----------------------------------------------------------------------
def task_erddiag():
    import numpy as np
    _inject_roots()
    import Analyze_clinical_erd_refined as erd
    from Analyze_clinical_erd_refined import (
        _logratio_to_pct, _reject_artifact_trials, config_a_pipeline,
        CONTRA_MOTOR_CLUSTER, BILATERAL_MOTOR_CLUSTER, IPSI_MOTOR_CLUSTER,
        MU_LO, MU_HI, MI_MARKER, REST_MARKER,
    )

    out = OUT_ROOT / f"erd_vs_topomap_diagnostic_{SUBJECT}_{SESSION}.txt"
    lines = []

    def w(s=""):
        print(s)
        lines.append(s)

    clusters = {"contra": CONTRA_MOTOR_CLUSTER,
                "bilat": BILATERAL_MOTOR_CLUSTER,
                "ipsi": IPSI_MOTOR_CLUSTER}
    win = (1.0, 4.0)  # post-cue SCALAR window

    pipe = config_a_pipeline(SUBJECT, SESSION)
    tfr_raw = {k: v.copy() for k, v in pipe["tfr_trials"].items()}  # pre-reject
    tfr_rej = {k: v.copy() for k, v in pipe["tfr_trials"].items()}
    rep = _reject_artifact_trials(tfr_rej)

    w(f"ERD topomap vs 6-panel reconciliation — {SUBJECT}/{SESSION}")
    w("=" * 70)
    w("Both start from the SAME per-trial logratio TFR (CAR, mu, baseline (-1,0)).")
    w("  topomap  = trial-MEAN of logratio -> %, NO extra rejection "
      "(generate_plots_config_a.py:157)")
    w("  6-panel  = trial-MEDIAN of per-trial %, AFTER _reject_artifact_trials "
      "(erd_refined.py:300,261)")
    w(f"  window = {win} s post-cue, mu {MU_LO:g}-{MU_HI:g} Hz")
    w("")
    for mk_lab, mk in (("MI", MI_MARKER), ("REST", REST_MARKER)):
        w(f"--- class {mk_lab} (marker {mk}) ---")
        if mk not in tfr_raw:
            w("  (absent)")
            continue
        traw = tfr_raw[mk]
        trej = tfr_rej.get(mk)
        fmask = (traw.freqs >= MU_LO) & (traw.freqs <= MU_HI)
        tmask = (traw.times >= win[0]) & (traw.times <= win[1])
        n_raw = traw.data.shape[0]
        n_rej = trej.data.shape[0] if trej is not None else 0
        w(f"  n_trials: raw(kept by 50uV+autodrop)={n_raw}  "
          f"after artifact reject={n_rej}  (dropped {n_raw - n_rej})")
        for cname, chans in clusters.items():
            idx = [traw.ch_names.index(c) for c in chans if c in traw.ch_names]
            # topomap estimator: mean over trials of logratio, then cluster+mu+win
            mean_lr = traw.data[:, idx][:, :, fmask][:, :, :, tmask].mean(axis=0)
            topo_pct = float(_logratio_to_pct(mean_lr).mean())
            # 6-panel estimator: per-trial cluster+mu+win mean %, then MEDIAN
            def _panel(tfr):
                pct = _logratio_to_pct(
                    tfr.data[:, idx][:, :, fmask][:, :, :, tmask])
                per_trial = pct.mean(axis=(1, 2, 3))  # one % per trial
                return per_trial
            pt_raw = _panel(traw)
            pt_rej = _panel(trej) if trej is not None else pt_raw
            w(f"  [{cname}] topomap(mean-logratio->%)={topo_pct:+7.1f}%   "
              f"6-panel(median%,reject)={np.median(pt_rej):+7.1f}%   "
              f"| mean%(no-rej)={pt_raw.mean():+8.1f}  "
              f"median%(no-rej)={np.median(pt_raw):+7.1f}")
            q = np.percentile(pt_raw, [0, 25, 50, 75, 100])
            w(f"        per-trial %% dist (no-rej) min/25/50/75/max = "
              f"[{q[0]:+.0f}, {q[1]:+.0f}, {q[2]:+.0f}, {q[3]:+.0f}, {q[4]:+.0f}]")
        w("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out}")


# ----------------------------------------------------------------------
# Diagnostic: does lowering the ERD reject abs-cap (600% -> 200%, the G1
# threshold) change the MI/REST signal? Reports n dropped + the bilat MI
# scalar (drives D1/G3) under each cap, plus how many MI trials have a
# post-cue PEAK |ERD%| over 200% (the substrate G1 actually counts).
# ----------------------------------------------------------------------
def task_capdiag():
    import numpy as np
    _inject_roots()
    from Analyze_clinical_erd_refined import (
        _logratio_to_pct, _reject_artifact_trials, config_a_pipeline,
        CONTRA_MOTOR_CLUSTER, BILATERAL_MOTOR_CLUSTER, IPSI_MOTOR_CLUSTER,
        MU_LO, MU_HI, MI_MARKER, REST_MARKER,
    )
    out = OUT_ROOT / f"erd_cap_sweep_{SUBJECT}_{SESSION}.txt"
    lines = []

    def w(s=""):
        print(s)
        lines.append(s)

    clusters = {"contra": CONTRA_MOTOR_CLUSTER,
                "bilat": BILATERAL_MOTOR_CLUSTER,
                "ipsi": IPSI_MOTOR_CLUSTER}
    win = (1.0, 4.0)
    pipe = config_a_pipeline(SUBJECT, SESSION)

    def _scalar_median(tfr, chans):
        """Per-trial cluster+mu mean %% trace, median across trials, then
        median over the post-cue window — the 'cluster MI scalar' that
        drives D1/G3."""
        idx = [tfr.ch_names.index(c) for c in chans if c in tfr.ch_names]
        fmask = (tfr.freqs >= MU_LO) & (tfr.freqs <= MU_HI)
        tmask = (tfr.times >= win[0]) & (tfr.times <= win[1])
        pct = _logratio_to_pct(tfr.data[:, idx][:, :, fmask])  # n,ch,f,t
        per_trial_t = pct.mean(axis=(1, 2))  # n, t
        med_trace = np.median(per_trial_t, axis=0)  # t
        return float(np.median(med_trace[tmask]))

    w(f"ERD reject abs-cap sweep — {SUBJECT}/{SESSION}")
    w("=" * 70)
    w("G1 counts MI trials whose post-cue PEAK |ERD%| > 200% "
      "(evaluate_erd_quality.py:587-589).")
    w("Cap=600 is the deployed TRIAL_REJECT_ABS_PCT; cap=200 = the G1 "
      "threshold the other agent is testing.")
    w(f"window {win} s, mu {MU_LO:g}-{MU_HI:g} Hz. "
      "MI scalar < 0 = desync (would pass G3).")
    w("")
    # How many MI trials exceed 200% / 600% PEAK over the bilateral montage
    # (the reject/G1 substrate)?
    tfr_mi = pipe["tfr_trials"][MI_MARKER]
    bidx = [tfr_mi.ch_names.index(c) for c in BILATERAL_MOTOR_CLUSTER
            if c in tfr_mi.ch_names]
    fmask = (tfr_mi.freqs >= MU_LO) & (tfr_mi.freqs <= MU_HI)
    tmask = tfr_mi.times >= 0.0
    pct_mi = _logratio_to_pct(tfr_mi.data[:, bidx][:, :, fmask]).mean(axis=(1, 2))
    peaks = np.max(np.abs(pct_mi[:, tmask]), axis=1)
    w(f"MI trials (n={len(peaks)}): peak>200%: {(peaks>200).sum()}  "
      f"peak>600%: {(peaks>600).sum()}  max peak={peaks.max():.0f}%")
    w("")

    for cap in (600.0, 200.0):
        tfr_c = {k: v.copy() for k, v in pipe["tfr_trials"].items()}
        rep = _reject_artifact_trials(tfr_c, abs_cap=cap)
        w(f"--- abs_cap = {cap:.0f}% ---")
        for mk_lab, mk in (("MI", MI_MARKER), ("REST", REST_MARKER)):
            if mk not in tfr_c:
                continue
            info = rep.get(mk, {})
            n_b = info.get("n_before", "?")
            n_d = info.get("n_dropped", 0)
            over = info.get("over_gate", False)
            scal = {c: _scalar_median(tfr_c[mk], ch)
                    for c, ch in clusters.items()}
            w(f"  {mk_lab}: kept {n_b}-{n_d}={tfr_c[mk].data.shape[0]}"
              f"{'  [G2 over_gate HIT]' if over else ''}  "
              f"scalar%% contra={scal['contra']:+.0f} "
              f"bilat={scal['bilat']:+.0f} ipsi={scal['ipsi']:+.0f}")
        w("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out}")


# ----------------------------------------------------------------------
# Task: shipped report figures (report_figures/).
#
# Canonical methods for the oneshot study = logratio units + 200% ERD cap +
# cluster-matched rejection (CAR, mu 8-13 Hz, baseline -1..0 s). Same methods
# as the diagnostic tasks above, but rendered with concise, high-level titles
# and legends for the write-up. The diagnostic-grade figures (percent ERD, the
# scored 6-panel with D1/G1 tags, per-run confusion grid, etc.) stay in their
# own directories (erd_refined/, eds/, ...). Built subject-parameterized so
# the SUBJ_008 pass and the two-subject consolidation drop in cleanly.
# ----------------------------------------------------------------------
def task_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    _inject_roots()

    # Masked, external-facing label (CLIN_SUBJ_007 -> "Subject 001").
    subj = REPORT_SUBJECT_LABEL.get(
        SUBJECT, SUBJECT.replace("CLIN_SUBJ_", "Subject "))
    # Per-subject subfolder so the two sessions' shipped figures stay separate
    # (generic filenames would otherwise overwrite). The top-level
    # report_figures/ is reserved for the consolidated 2-subject figures.
    rep = OUT_ROOT / "report_figures" / subj.replace(" ", "_")
    rep.mkdir(parents=True, exist_ok=True)

    # ---- (1) ERD/ERS timecourse — logratio, mean±SE, cluster-matched, re-zeroed.
    # Report version: one standalone 2-panel (MI | Rest) figure per cluster, so
    # the write-up can include whichever cluster it needs. The diagnostic 6-panel
    # (task_erdmean) keeps all three clusters in one figure. MI and Rest panels
    # autoscale independently (desync is small, the Rest ERS is large).
    from Analyze_clinical_erd_refined import (
        MU_LO, MU_HI, MI_MARKER, REST_MARKER, config_a_pipeline,
        CONTRA_MOTOR_CLUSTER, BILATERAL_MOTOR_CLUSTER, IPSI_MOTOR_CLUSTER,
    )
    pipe = config_a_pipeline(SUBJECT, SESSION)
    tfr_base = pipe["tfr_trials"]
    erd_clusters = [("Contralateral", "contra", CONTRA_MOTOR_CLUSTER),
                    ("Bilateral", "bilat", BILATERAL_MOTOR_CLUSTER),
                    ("Ipsilateral", "ipsi", IPSI_MOTOR_CLUSTER)]
    erd_cols = [("Motor imagery", MI_MARKER), ("Rest", REST_MARKER)]
    for cname, ckey, chans in erd_clusters:
        tfr_c, _ = _reject_artifact_trials_for_cluster(
            tfr_base, chans, abs_cap=ERD_ABS_CAP)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
        for ci, (clab, marker) in enumerate(erd_cols):
            ax = axes[ci]
            if marker not in tfr_c:
                ax.set_visible(False)
                continue
            t = tfr_c[marker]
            idx = [t.ch_names.index(c) for c in chans if c in t.ch_names]
            fmask = (t.freqs >= MU_LO) & (t.freqs <= MU_HI)
            pt = t.data[:, idx][:, :, fmask].mean(axis=(1, 2))  # logratio (n,t)
            bmask = (t.times >= -1.0) & (t.times <= 0.0)
            pt = pt - pt[:, bmask].mean(axis=1, keepdims=True)  # baseline re-zero
            n = pt.shape[0]
            mean = pt.mean(axis=0)
            se = (pt.std(axis=0, ddof=1) / np.sqrt(n)
                  if n > 1 else np.zeros_like(mean))
            ax.axhline(0, color="k", lw=0.6)
            ax.axvline(0, color="grey", lw=0.6, ls=":")
            ax.plot(t.times, mean, color="tab:blue", lw=1.6)
            ax.fill_between(t.times, mean - se, mean + se,
                            color="tab:blue", alpha=0.25)
            ax.set_title(clab, fontsize=12)
            ax.set_xlabel("Time from cue (s)")
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("ERD/ERS (logratio)")
        fig.suptitle(
            f"{subj} — μ (8–13 Hz) ERD/ERS · {cname} motor cluster (mean ± SE)",
            fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(rep / f"erd_timecourse_{ckey}.png", dpi=150)
        plt.close(fig)
        print(f"  wrote erd_timecourse_{ckey}.png")

    # ---- (2) ERD/ERS topography (mu) — MI + REST strips, logratio grand-avg
    import generate_plots_config_a as gp
    tfr_cap, _rep, _pipe = _capped_tfr_trials()
    tfr_avg = {mk: tt.average() for mk, tt in tfr_cap.items()}
    rest_avg, mi_avg = tfr_avg.get("100"), tfr_avg.get("200")
    vlim = gp._compute_dynamic_vlim(rest_avg, mi_avg)
    if mi_avg is not None:
        gp._plot_topo_strip(
            mi_avg, MU_LO, MU_HI, vlim,
            f"{subj} — μ ERD/ERS topography · Motor imagery",
            str(rep / "erd_topography_mi.png"))
    if rest_avg is not None:
        gp._plot_topo_strip(
            rest_avg, MU_LO, MU_HI, vlim,
            f"{subj} — μ ERD/ERS topography · Rest",
            str(rep / "erd_topography_rest.png"))
    print("  wrote erd_topography_{mi,rest}.png")

    # ---- (3) EDS topomaps (mu) — reuse the canonical compute, clean titles.
    # The canonical EDS plotters take (title, save_path) as args; wrap the
    # single-panel plotter to redirect the per-class cohort panel (== this one
    # subject) into report_figures with a high-level title, and skip the
    # per-subject grid (degenerate at n=1; it becomes the consolidation figure
    # once SUBJ_008 is added).
    import Analyze_eds_topoplot_CLIN as eds
    eds.EXPERT_SOURCE_SUBJECT = SUBJECT
    eds.clin_pictures_root = lambda: OUT_ROOT  # CSVs land in eds/ (diagnostic)
    _orig_panel = eds._plot_topomap_panel

    def _clean_eds_panel(z_vec, channels, title, save_path, *a, **k):
        name = Path(save_path).name
        if name.startswith("cohort_eds_mi"):
            _orig_panel(z_vec, channels,
                        f"{subj} — EDS (electrode discriminancy score) · "
                        f"Motor imagery (μ)",
                        str(rep / "eds_mi.png"), *a, **k)
        elif name.startswith("cohort_eds_rest"):
            _orig_panel(z_vec, channels,
                        f"{subj} — EDS (electrode discriminancy score) · "
                        f"Rest (μ)",
                        str(rep / "eds_rest.png"), *a, **k)
        # other panels (diffs) intentionally not shipped

    eds._plot_topomap_panel = _clean_eds_panel
    eds._plot_per_subject_grid = lambda *a, **k: None  # consolidation: later
    try:
        sys.argv = ["eds", "--subjects", SUBJECT, "--no-diff-plot"]
        eds.main()
    finally:
        eds._plot_topomap_panel = _orig_panel
    print("  wrote eds_{mi,rest}.png")

    # ---- (4) Decoder performance across runs (trial κ, NKV, accuracy)
    import Analyze_clinical_decoder_longitudinal as dec
    from Analyze_experiment_logs_cross_subject import find_decoder_csv
    runs = _substantive_run_dirs()
    metric_rows = []
    for run_id, rd in runs:
        df = pd.read_csv(find_decoder_csv(str(rd)))
        df["RunID"] = f"{SUBJECT}__{SESSION}__{rd.name}"
        df["GlobalTrialID"] = df["RunID"] + "_" + df["Trial"].astype(str)
        m = dec._session_metrics([df])
        metric_rows.append((run_id, m))
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(metric_rows))
    series = [("trial_kappa", "Trial κ (chance-corrected)"),
              ("nkv", "NKV (κ × decision rate)"),
              ("acc_decided", "Accuracy — decided trials"),
              ("acc_inclusive", "Accuracy — all trials")]
    for col, lab in series:
        ax.plot(x, [m.get(col) for _, m in metric_rows], marker="o", label=lab)
    ax.set_xticks(list(x))
    ax.set_xticklabels([r for r, _ in metric_rows])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Run")
    ax.set_ylabel("Metric")
    ax.set_title(f"{subj} — decoder performance across runs")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(rep / "decoder_performance.png", dpi=200)
    plt.close(fig)
    print("  wrote decoder_performance.png")

    # ---- (5) Pooled confusion matrix (whole session, all substantive runs)
    import Analyze_clinical_confusion_matrices as cmmod
    from Analyze_experiment_logs_cross_subject import (
        compute_confusion_matrix_from_csv,
    )
    pooled = []
    for run_id, rd in runs:
        d = pd.read_csv(find_decoder_csv(str(rd)))
        d["RunID"] = f"{SUBJECT}__{SESSION}__{rd.name}"
        d["GlobalTrialID"] = d["RunID"] + "_" + d["Trial"].astype(str)
        pooled.append(d)
    cm = compute_confusion_matrix_from_csv(pd.concat(pooled, ignore_index=True))
    if cm is not None:
        cmmod._plot_subject_cm(
            cm, subj, 1, len(runs), str(rep / "confusion_matrix.png"))
        print("  wrote confusion_matrix.png")

    # ---- (6) Feedback bar dynamics — Lean% and time-to-threshold per trial
    import Analyze_clinical_bar_dynamics_longitudinal as bar
    df_tr = bar._load_session_trials(SUBJECT, SESSION)
    if not df_tr.empty:
        run_map = {rd.name: rid for rid, rd in runs}
        df_tr["run"] = df_tr["run_id"].map(run_map)
        per_run = bar._per_run_median(df_tr)
        per_run["run"] = per_run["run_id"].map(run_map)
        groups = ["All"] + sorted(r for r in per_run["run"].dropna().unique())
        colors = {"MI": "tab:orange", "REST": "tab:blue"}

        def _vals(metric, group, cls):
            d = df_tr[df_tr["Class"] == cls]
            if group != "All":
                d = d[d["run"] == group]
            return d[metric].dropna().values

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, metric, ylab in (
                (ax1, "LeanPct", "Lean toward target (%)"),
                (ax2, "TimeToThresh_s", "Time to threshold (s)")):
            handles = {}
            for gi, g in enumerate(groups):
                for cls, off in (("MI", -0.18), ("REST", +0.18)):
                    v = _vals(metric, g, cls)
                    if len(v) == 0:
                        continue
                    bp = ax.boxplot(v, positions=[gi + off], widths=0.3,
                                    patch_artist=True, showfliers=True,
                                    medianprops=dict(color="black"),
                                    flierprops=dict(marker=".", markersize=4,
                                                    markerfacecolor=colors[cls],
                                                    markeredgecolor=colors[cls],
                                                    alpha=0.5))
                    for box in bp["boxes"]:
                        box.set(facecolor=colors[cls], alpha=0.55)
                    handles[cls] = bp["boxes"][0]
            ax.axvline(0.5, color="grey", lw=0.8, ls="--")
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(groups)
            ax.set_xlabel("Session / run")
            ax.set_ylabel(ylab)
            ax.grid(True, axis="y", alpha=0.3)
            if handles:
                ax.legend(handles.values(), handles.keys(), loc="best")
        fig.suptitle(f"{subj} — feedback bar dynamics "
                     f"(box = per-trial distribution)")
        fig.tight_layout()
        fig.savefig(rep / "bar_dynamics.png", dpi=200)
        plt.close(fig)
        print("  wrote bar_dynamics.png")

    print(f"\nShipped report figures -> {rep}")


# ----------------------------------------------------------------------
# Task: two-subject consolidated figures for the proposal (top-level
# report_figures/). Both oneshot participants on shared axes; the per-session
# figures stay in report_figures/Subject_00N/. Priority per Arman: the
# bilateral ERD/ERS overlay (neuromodulation) and the side-by-side EDS grid.
# Subject IDs are masked (Subject 001/002). Requires both subjects present in
# the oneshot tree.
# ----------------------------------------------------------------------
def task_consolidate():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _inject_roots()
    rep = OUT_ROOT / "report_figures"
    rep.mkdir(parents=True, exist_ok=True)
    subjects = [("CLIN_SUBJ_007", "Subject 001", "tab:blue"),
                ("CLIN_SUBJ_008", "Subject 002", "tab:orange")]

    # ---- (1) Bilateral ERD/ERS overlay (both subjects, MI | Rest) ----
    from Analyze_clinical_erd_refined import (
        MU_LO, MU_HI, MI_MARKER, REST_MARKER, config_a_pipeline,
        BILATERAL_MOTOR_CLUSTER,
    )
    cols = [("Motor imagery", MI_MARKER), ("Rest", REST_MARKER)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for sid, slabel, scolor in subjects:
        pipe = config_a_pipeline(sid, SESSION)
        tfr_c, _ = _reject_artifact_trials_for_cluster(
            pipe["tfr_trials"], BILATERAL_MOTOR_CLUSTER, abs_cap=ERD_ABS_CAP)
        for ci, (clab, marker) in enumerate(cols):
            if marker not in tfr_c:
                continue
            t = tfr_c[marker]
            idx = [t.ch_names.index(c) for c in BILATERAL_MOTOR_CLUSTER
                   if c in t.ch_names]
            fmask = (t.freqs >= MU_LO) & (t.freqs <= MU_HI)
            pt = t.data[:, idx][:, :, fmask].mean(axis=(1, 2))
            bmask = (t.times >= -1.0) & (t.times <= 0.0)
            pt = pt - pt[:, bmask].mean(axis=1, keepdims=True)
            n = pt.shape[0]
            mean = pt.mean(axis=0)
            se = (pt.std(axis=0, ddof=1) / np.sqrt(n)
                  if n > 1 else np.zeros_like(mean))
            axes[ci].plot(t.times, mean, color=scolor, lw=1.7, label=slabel)
            axes[ci].fill_between(t.times, mean - se, mean + se,
                                  color=scolor, alpha=0.18)
    for ci, (clab, _m) in enumerate(cols):
        ax = axes[ci]
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="grey", lw=0.6, ls=":")
        ax.set_title(clab, fontsize=12)
        ax.set_xlabel("Time from cue (s)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("ERD/ERS (logratio)")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("μ (8–13 Hz) ERD/ERS · bilateral sensorimotor cluster "
                 "(mean ± SE)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(rep / "consolidated_erd_timecourse_bilat.png", dpi=150)
    plt.close(fig)
    print("  wrote consolidated_erd_timecourse_bilat.png")

    # ---- (2) EDS side-by-side grid (both subjects), neutral method title ----
    # The canonical per-subject grid (_plot_per_subject_grid) is one panel per
    # subject — a side-by-side layout for n=2. Wrap it to relabel the panels
    # with the masked IDs and write a method-only title (no same-site claim).
    import mne
    import Analyze_eds_topoplot_CLIN as eds
    eds.EXPERT_SOURCE_SUBJECT = "CLIN_SUBJ_007"
    eds.clin_pictures_root = lambda: OUT_ROOT
    _orig_grid = eds._plot_per_subject_grid
    _orig_panel = eds._plot_topomap_panel

    def _clean_grid(per_subj_eds, title, save_path, cmap="viridis", *a, **k):
        # The canonical grid hardcodes a 4-column layout; with n=2 subjects its
        # two trailing blank columns left a large gap and pushed the colorbar
        # far to the right. Render a tight one-row, one-column-per-subject panel
        # instead (masked IDs, shared vlim — same z-score/colorbar semantics as
        # the canonical grid, Analyze_eds_topoplot_CLIN.py:870-922).
        name = Path(save_path).name
        if "_mi_" in name:
            cls, out = "Motor imagery", rep / "consolidated_eds_mi.png"
        elif "_rest_" in name:
            cls, out = "Rest", rep / "consolidated_eds_rest.png"
        else:
            return
        items = [(REPORT_SUBJECT_LABEL.get(s, s), v)
                 for s, v in sorted(per_subj_eds.items())]
        panels = []
        for slabel, (eds_vec, channels) in items:
            sd = eds_vec.std(ddof=1)
            z = ((eds_vec - eds_vec.mean()) / sd if sd > 0
                 else np.zeros_like(eds_vec))
            panels.append((slabel, z, channels))
        all_z = np.concatenate([z for _, z, _ in panels]) if panels else []
        vmax = float(np.nanmax(np.abs(all_z))) if len(all_z) else 1.0
        vlim = (-vmax, vmax) if vmax > 0 else (None, None)
        f, axs = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.4),
                              squeeze=False)
        last_im = None
        for ax, (slabel, z, channels) in zip(axs[0], panels):
            info = eds._make_info_for_channels(channels)
            last_im, _ = mne.viz.plot_topomap(
                z, info, axes=ax, cmap=cmap, show=False, names=channels,
                vlim=vlim)
            ax.set_title(slabel, fontsize=11)
        if last_im is not None:
            f.colorbar(last_im, ax=list(axs[0]), shrink=0.7,
                       label=f"EDS z-score (shared vlim ±{vmax:.2f})")
        f.suptitle(f"EDS (electrode discriminancy score) · {cls} (μ)",
                   fontsize=12)
        f.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(f)

    eds._plot_per_subject_grid = _clean_grid
    eds._plot_topomap_panel = lambda *a, **k: None  # skip cohort single panels
    try:
        sys.argv = ["eds", "--subjects", "CLIN_SUBJ_007,CLIN_SUBJ_008",
                    "--no-diff-plot"]
        eds.main()
    finally:
        eds._plot_per_subject_grid = _orig_grid
        eds._plot_topomap_panel = _orig_panel
    print("  wrote consolidated_eds_{mi,rest}.png")

    # ---- (3) Decoder performance overlay (both subjects): κ and NKV ----
    import pandas as pd
    import Analyze_clinical_decoder_longitudinal as dec
    from Analyze_experiment_logs_cross_subject import (
        find_decoder_csv, compute_confusion_matrix_from_csv,
    )

    def _run_metrics(sid):
        rows = []
        for rid, rd in _substantive_run_dirs(sid):
            df = pd.read_csv(find_decoder_csv(str(rd)))
            df["RunID"] = f"{sid}__{SESSION}__{rd.name}"
            df["GlobalTrialID"] = df["RunID"] + "_" + df["Trial"].astype(str)
            rows.append((rid, dec._session_metrics([df])))
        return rows

    fig, (axk, axn) = plt.subplots(1, 2, figsize=(11, 4.2))
    nmax = 0
    for sid, slabel, scolor in subjects:
        mr = _run_metrics(sid)
        nmax = max(nmax, len(mr))
        x = range(len(mr))
        axk.plot(x, [m.get("trial_kappa") for _, m in mr],
                 marker="o", color=scolor, label=slabel)
        axn.plot(x, [m.get("nkv") for _, m in mr],
                 marker="o", color=scolor, label=slabel)
    for ax, ttl in ((axk, "Trial κ (chance-corrected)"),
                    (axn, "NKV (κ × decision rate)")):
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(range(nmax))
        ax.set_xticklabels([f"R{i+1:03d}" for i in range(nmax)])
        ax.set_xlabel("Run")
        ax.set_title(ttl)
        ax.grid(True, alpha=0.3)
    axk.set_ylabel("Metric")
    axk.legend(loc="lower left", fontsize=9)
    fig.suptitle("Calibration-free decoder performance across runs", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(rep / "consolidated_decoder.png", dpi=200)
    plt.close(fig)
    print("  wrote consolidated_decoder.png")

    # ---- (4) Pooled confusion matrices side by side (both subjects) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    fig.subplots_adjust(wspace=0.08)
    im = None
    for ax_i, (ax, (sid, slabel, _c)) in enumerate(zip(axes, subjects)):
        pooled = []
        for rid, rd in _substantive_run_dirs(sid):
            d = pd.read_csv(find_decoder_csv(str(rd)))
            d["RunID"] = f"{sid}__{SESSION}__{rd.name}"
            d["GlobalTrialID"] = d["RunID"] + "_" + d["Trial"].astype(str)
            pooled.append(d)
        cm = compute_confusion_matrix_from_csv(
            pd.concat(pooled, ignore_index=True))
        row_tot = cm.sum(axis=1, keepdims=True)
        pct = np.where(row_tot > 0, 100.0 * cm / np.maximum(row_tot, 1), 0.0)
        im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Pred MI", "Pred REST", "Ambiguous"])
        ax.set_yticks([0, 1])
        # Row meaning is shared across the two panels; labelling only the left
        # panel keeps the right panel's "Actual REST" tick text from rendering
        # into the inter-panel gap and over Subject 001's matrix.
        ax.set_yticklabels(["Actual MI", "Actual REST"] if ax_i == 0 else [])
        for i in range(2):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.1f}%)", ha="center",
                        va="center", fontsize=11,
                        color="white" if pct[i, j] > 50 else "black")
        dec_n = int(cm[:, :2].sum())
        corr = int(cm[0, 0] + cm[1, 1])
        amb = int(cm[:, 2].sum())
        tot = int(cm.sum())
        dacc = 100.0 * corr / dec_n if dec_n else float("nan")
        ax.set_title(f"{slabel}\nDecision acc = {dacc:.1f}%  |  "
                     f"Ambiguous = {amb}/{tot}", fontsize=11)
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04,
                     label="% of actual class")
    fig.suptitle("Pooled whole-session confusion (calibration-free)",
                 fontsize=13)
    fig.savefig(rep / "consolidated_confusion.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  wrote consolidated_confusion.png")

    # ---- (5) Bar dynamics (both subjects): whole-session distributions ----
    # Per Arman (2026-06-05): report the session-level per-trial distribution
    # only. The earlier per-run breakdown ("All" + R001-R004) inflated apparent
    # instability and understated Subject 002's control; both subjects now share
    # each metric panel (x = subject, MI/REST boxes side by side).
    import Analyze_clinical_bar_dynamics_longitudinal as bar
    colmap = {"MI": "tab:orange", "REST": "tab:blue"}
    labels = [slabel for _sid, slabel, _c in subjects]
    sess_trials = {slabel: bar._load_session_trials(sid, SESSION)
                   for sid, slabel, _c in subjects}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ci, (metric, ylab) in enumerate(
            (("LeanPct", "Lean toward target (%)"),
             ("TimeToThresh_s", "Time to threshold (s)"))):
        ax = axes[ci]
        handles = {}
        for si, slabel in enumerate(labels):
            df_tr = sess_trials[slabel]
            if df_tr.empty or metric not in df_tr:
                continue
            for cls, off in (("MI", -0.18), ("REST", +0.18)):
                v = df_tr[df_tr["Class"] == cls][metric].dropna().values
                if len(v) == 0:
                    continue
                bp = ax.boxplot(v, positions=[si + off], widths=0.3,
                                patch_artist=True, showfliers=True,
                                medianprops=dict(color="black"),
                                flierprops=dict(marker=".", markersize=4,
                                                markerfacecolor=colmap[cls],
                                                markeredgecolor=colmap[cls],
                                                alpha=0.5))
                for box in bp["boxes"]:
                    box.set(facecolor=colmap[cls], alpha=0.55)
                handles[cls] = bp["boxes"][0]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylabel(ylab)
        if handles and ci == 0:
            ax.legend(handles.values(), handles.keys(), loc="best")
    fig.suptitle("Online feedback bar dynamics, whole session "
                 "(box = per-trial distribution)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(rep / "consolidated_bar.png", dpi=200)
    plt.close(fig)
    print("  wrote consolidated_bar.png")

    print(f"\nConsolidated figures -> {rep}")


# ----------------------------------------------------------------------
# Task: consolidated 2-subject ERD topography-over-time for the report
# (report_figures/consolidated_erd_topography_{mi,rest}.png). Rows = subject,
# cols = the 8 canonical post-cue 0.5 s windows. CAR spatial filter — matches
# the rest of the analysis (timecourse, decoder framing). CSD/Hjorth were
# evaluated (task_topocompare) but rejected 2026-06-05 with Arman: the surface
# Laplacian focalised onto midline-central rather than the expected
# contralateral hand area, and only for one subject, so its focalisation was
# not informative enough to justify a second reference scheme. Per-subject
# vlim: the two sessions differ ~2x in logratio amplitude, so a shared colour
# scale would wash Subject 002 out. Kept as its own task (not folded into
# task_consolidate) so the topography can be re-rendered without regenerating
# the other five consolidated figures.
# ----------------------------------------------------------------------
def task_consolidatetopo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _inject_roots()
    import generate_plots_config_a as gp
    from Analyze_clinical_erd_refined import (
        MU_LO, MU_HI, config_a_pipeline, _reject_artifact_trials,
    )
    rep = OUT_ROOT / "report_figures"
    rep.mkdir(parents=True, exist_ok=True)
    subjects = [("CLIN_SUBJ_007", "Subject 001"),
                ("CLIN_SUBJ_008", "Subject 002")]
    starts, win = gp.TOPO_TIME_STARTS, gp.TOPO_WINDOW_SIZE

    # Build the CAR MI/REST average TFRs + per-subject vlim once. The default
    # CONFIG_A_DISPLAY_BASELINE spatial_filter is already "car", so no override.
    avgs = {}
    for sid, slabel in subjects:
        pipe = config_a_pipeline(sid, SESSION)
        tfr = pipe["tfr_trials"]
        _reject_artifact_trials(tfr, abs_cap=ERD_ABS_CAP)
        tfr_avg = {mk: t.average() for mk, t in tfr.items()}
        rest, mi = tfr_avg.get("100"), tfr_avg.get("200")
        avgs[sid] = {"label": slabel, "mi": mi, "rest": rest,
                     "vlim": gp._compute_dynamic_vlim(rest, mi)}

    for cls_key, cls_name in (("mi", "Motor imagery"), ("rest", "Rest")):
        fig, axes = plt.subplots(
            len(subjects), len(starts),
            figsize=(1.7 * len(starts) + 1.0, 2.2 * len(subjects)),
            squeeze=False, constrained_layout=True)
        for ri, (sid, _slabel) in enumerate(subjects):
            a = avgs[sid]
            avg_tfr, vlim = a[cls_key], a["vlim"]
            for ci, t0 in enumerate(starts):
                ax = axes[ri][ci]
                if avg_tfr is None:
                    ax.axis("off")
                    continue
                avg_tfr.plot_topomap(
                    tmin=float(t0), tmax=float(t0 + win), fmin=MU_LO,
                    fmax=MU_HI, axes=ax, cmap=gp.CMAP, show=False,
                    vlim=vlim, colorbar=False, show_names=False)
                if ri == 0:
                    ax.set_title(f"{t0:.1f}–{t0 + win:.1f}s", fontsize=9)
            # Rotated subject label (plot_topomap clears y-axis labels, so place
            # it in axes-fraction coords on the leftmost panel).
            axes[ri][0].text(-0.22, 0.5, a["label"],
                             transform=axes[ri][0].transAxes, rotation=90,
                             va="center", ha="center", fontsize=12,
                             fontweight="bold")
            sm = plt.cm.ScalarMappable(norm=plt.Normalize(*vlim), cmap=gp.CMAP)
            sm.set_array([])
            fig.colorbar(sm, ax=list(axes[ri]), location="right",
                         shrink=0.85, label="ERD/ERS (logratio)")
        fig.suptitle(f"μ (8–13 Hz) ERD/ERS topography over time · {cls_name} "
                     f"· CAR (per-subject scale)", fontsize=13)
        out = rep / f"consolidated_erd_topography_{cls_key}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out.name}")
    print(f"\nConsolidated topography -> {rep}")


# ----------------------------------------------------------------------
# Task: EDS spatial-reference diagnostic. The deployed MDM decoder builds
# covariances on RAW (hardware-referenced) signals — no CAR
# (Generate_Riemannian_adaptive.py:360-404; Utils/runtime_common.py has no
# set_eeg_reference). Kumar 2024 likewise computes covariance directly from
# bandpassed EEG (Eq. 1), no CAR. But Analyze_eds_topoplot_CLIN applies CAR
# (line 389/697) over the full ~27-ch set and only then restricts to the
# motor-15 feature montage. This task recomputes per-class MI EDS and pooled
# MI-vs-REST EDS under three spatial references — `car` (current),
# `car15` (CAR on just the 15 feature channels), `none` (raw, = the decoder)
# — and reports where C3 / CP2 / POz land, to test whether the non-physiological
# attribution is a CAR artifact. Diagnostic only; no report figures touched.
# ----------------------------------------------------------------------
def task_edsdiag():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne
    _inject_roots()
    import Analyze_eds_topoplot_CLIN as eds
    eds.EXPERT_SOURCE_SUBJECT = "CLIN_SUBJ_007"
    eds.clin_pictures_root = lambda: OUT_ROOT
    out_dir = OUT_ROOT / "eds" / "spatial_diag"
    out_dir.mkdir(parents=True, exist_ok=True)

    motor = eds.MOTOR_CHANNEL_NAMES
    band = (eds.MU_LO, eds.MU_HI)
    _orig_asf = eds.apply_spatial_filter

    def _patched_asf(epochs, method):
        # Adds two references the canonical filter doesn't support: `none`
        # (identity = raw covariance, matching the deployed decoder) and
        # `car15` (average reference computed over ONLY the motor-15 feature
        # channels, so the reference is internal to the feature space).
        if method == "none":
            return epochs
        if method == "car15":
            picks = [c for c in motor if c in epochs.ch_names]
            idx = [epochs.ch_names.index(c) for c in picks]
            ep = epochs.copy()
            d = ep.get_data()
            d[:, idx, :] = d[:, idx, :] - d[:, idx, :].mean(axis=1, keepdims=True)
            ep._data = d
            return ep
        return _orig_asf(epochs, method)

    key = ["C3", "C4", "Cz", "CP1", "CP2", "CP5", "CP6", "Pz", "POz", "P3", "P4"]
    subjects = [("CLIN_SUBJ_007", "Subject 001"),
                ("CLIN_SUBJ_008", "Subject 002")]
    refs = ("car", "car15", "none")
    # results[(sid, ref)] = {"mi": (vec, kept), "pooled": (vec, kept)}
    results = {}
    eds.apply_spatial_filter = _patched_asf
    try:
        for sid, _slab in subjects:
            for ref in refs:
                mi_vec, _r, kept, nmi, nrest = eds.per_session_eds_per_class(
                    sid, SESSION, band, motor, "mu", spatial_filter=ref)
                pooled_vec, kept_p, _n = eds.per_session_eds(
                    sid, SESSION, band, motor, "mu", spatial_filter=ref)
                results[(sid, ref)] = {
                    "mi": (mi_vec, kept), "pooled": (pooled_vec, kept_p),
                    "nmi": nmi, "nrest": nrest}
    finally:
        eds.apply_spatial_filter = _orig_asf

    def _rank_table(which):
        # Print z-scored EDS + descending rank for the key channels, per ref.
        print(f"\n===== {which.upper()} EDS — z-score (rank/Nkept) =====")
        for sid, slab in subjects:
            print(f"\n  {slab} ({sid}):")
            hdr = "    {:<5}".format("ch") + "".join(f"{r:>16}" for r in refs)
            print(hdr)
            cols = {}
            for ref in refs:
                vec, kept = results[(sid, ref)][which]
                if vec is None:
                    cols[ref] = ({}, {}, 0)
                    continue
                z = (vec - vec.mean()) / (vec.std(ddof=1) or 1.0)
                order = np.argsort(-vec)  # rank 1 = highest raw EDS
                rank = {kept[order[k]]: k + 1 for k in range(len(kept))}
                cols[ref] = (dict(zip(kept, z)), rank, len(kept))
            for ch in key:
                cells = []
                for ref in refs:
                    zmap, rank, nk = cols[ref]
                    if ch in zmap:
                        cells.append(f"{zmap[ch]:+.2f} ({rank[ch]}/{nk})")
                    else:
                        cells.append("—")
                print("    {:<5}".format(ch) + "".join(f"{c:>16}" for c in cells))

    print(f"\nEDS spatial-reference diagnostic | mu {band[0]:g}-{band[1]:g} Hz | "
          f"motor-15 | per-class(MI vs own baseline) and pooled(MI vs REST)")
    for sid, slab in subjects:
        r = results[(sid, "car")]
        print(f"  {slab}: n_MI={r['nmi']} n_REST={r['nrest']}")
    _rank_table("pooled")   # the faithful Kumar variant (two classes)
    _rank_table("mi")       # the per-class variant currently shipped

    # Topomap comparison: rows = subject, cols = ref, for each of pooled/mi.
    for which, cls_name in (("pooled", "Pooled MI-vs-REST (Kumar)"),
                            ("mi", "Per-class MI-vs-baseline (shipped)")):
        fig, axes = plt.subplots(len(subjects), len(refs),
                                 figsize=(4.0 * len(refs), 3.8 * len(subjects)),
                                 squeeze=False)
        for ri, (sid, slab) in enumerate(subjects):
            for ci, ref in enumerate(refs):
                ax = axes[ri][ci]
                vec, kept = results[(sid, ref)][which]
                if vec is None or len(kept) < 2:
                    ax.axis("off")
                    continue
                z = (vec - vec.mean()) / (vec.std(ddof=1) or 1.0)
                vmax = float(np.nanmax(np.abs(z))) or 1.0
                info = eds._make_info_for_channels(kept)
                im, _ = mne.viz.plot_topomap(
                    z, info, axes=ax, cmap="viridis", show=False,
                    names=kept, vlim=(-vmax, vmax))
                fig.colorbar(im, ax=ax, shrink=0.7)
                ax.set_title(f"{slab} · {ref}", fontsize=10)
        fig.suptitle(f"EDS spatial-reference comparison · {cls_name} (z-scored)",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        png = out_dir / f"eds_spatial_diag_{which}.png"
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {png.name}")
    print(f"\nEDS spatial diagnostic -> {out_dir}")


# ----------------------------------------------------------------------
# Task: pooled (2-subject) EDS figure under RAW covariance (no CAR — matches
# the online MDM decoder, Generate_Riemannian_adaptive.py:360-404). Three
# contrasts side by side: MI-vs-REST (the faithful Kumar two-class EDS),
# MI-vs-baseline, and REST-vs-baseline (the per-class variants). Pooling
# follows Kumar 2024 (Fig 5 caption): EDS is computed per subject, z-scored
# across that subject's own electrodes, then averaged across subjects (equal
# weight — robust at n=2, where one subject's larger class separation would
# otherwise dominate a raw-EDS average). Channels are intersected across
# subjects. Exploratory; writes to eds/pooled/, no report figures touched.
# ----------------------------------------------------------------------
def task_edspooled():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne
    _inject_roots()
    import Analyze_eds_topoplot_CLIN as eds
    eds.EXPERT_SOURCE_SUBJECT = "CLIN_SUBJ_007"
    eds.clin_pictures_root = lambda: OUT_ROOT
    out_dir = OUT_ROOT / "eds" / "pooled"
    out_dir.mkdir(parents=True, exist_ok=True)

    motor = eds.MOTOR_CHANNEL_NAMES
    band = (eds.MU_LO, eds.MU_HI)
    subjects = ["CLIN_SUBJ_007", "CLIN_SUBJ_008"]
    _orig_asf = eds.apply_spatial_filter
    eds.apply_spatial_filter = (
        lambda epochs, method: epochs if method == "none"
        else _orig_asf(epochs, method))

    # contrast -> per-subject {channel: z-score}
    contrasts = ("MI vs REST", "MI vs baseline", "REST vs baseline")
    per_subj = {c: [] for c in contrasts}
    try:
        for sid in subjects:
            mi_rest, kept_p, n_p = eds.per_session_eds(
                sid, SESSION, band, motor, "mu", spatial_filter="none")
            mi_base, rest_base, kept_pc, nmi, nrest = (
                eds.per_session_eds_per_class(
                    sid, SESSION, band, motor, "mu", spatial_filter="none"))
            print(f"  {sid}: n_MI={nmi} n_REST={nrest} "
                  f"kept(pooled)={len(kept_p)} kept(perclass)={len(kept_pc)}")
            for vec, kept, cname in (
                    (mi_rest, kept_p, "MI vs REST"),
                    (mi_base, kept_pc, "MI vs baseline"),
                    (rest_base, kept_pc, "REST vs baseline")):
                if vec is None:
                    per_subj[cname].append({})
                    continue
                z = (vec - vec.mean()) / (vec.std(ddof=1) or 1.0)
                per_subj[cname].append(dict(zip(kept, z)))
    finally:
        eds.apply_spatial_filter = _orig_asf

    fig, axes = plt.subplots(1, len(contrasts), figsize=(4.4 * len(contrasts), 4.2),
                             squeeze=False)
    for ci, cname in enumerate(contrasts):
        ax = axes[0][ci]
        maps = [m for m in per_subj[cname] if m]
        if not maps:
            ax.axis("off")
            continue
        common = [ch for ch in motor if all(ch in m for m in maps)]
        if len(common) < 2:
            ax.axis("off")
            continue
        pooled = np.array([np.mean([m[ch] for m in maps]) for ch in common])
        vmax = float(np.nanmax(np.abs(pooled))) or 1.0
        info = eds._make_info_for_channels(common)
        im, _ = mne.viz.plot_topomap(
            pooled, info, axes=ax, cmap="viridis", show=False,
            names=common, vlim=(-vmax, vmax))
        fig.colorbar(im, ax=ax, shrink=0.7, label="mean EDS z (2 subj)")
        ax.set_title(cname, fontsize=11)
    fig.suptitle("Pooled 2-subject EDS · μ · raw covariance (no CAR, "
                 "online-matched)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    png = out_dir / "eds_pooled_none.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {png}")


# ----------------------------------------------------------------------
# Task: the shipped report EDS figure — a SINGLE pooled MI-vs-REST topomap
# under raw covariance (no CAR), written to
# report_figures/consolidated_eds_pooled.png. This is the headline EDS for the
# report (decided 2026-06-05 with Arman), superseding the per-subject per-class
# grid (consolidated_eds_{mi,rest}.png from task_consolidate, retained as a
# diagnostic). Rationale established in the EDS audit: (a) MI-vs-REST is the
# faithful Kumar 2024 two-class contrast (the per-class MI-vs-baseline variant
# is a divergence); (b) "none" matches the online MDM decoder's raw-covariance
# feature space (Generate_Riemannian_adaptive.py:360-404), which CAR did not;
# (c) pooling the two participants (per-subject z, then averaged — equal weight)
# is more stable than the n=1 per-subject maps and mirrors Kumar's Fig 5
# subject-averaging.
# ----------------------------------------------------------------------
def task_consolidateeds():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne
    _inject_roots()
    import Analyze_eds_topoplot_CLIN as eds
    eds.EXPERT_SOURCE_SUBJECT = "CLIN_SUBJ_007"
    eds.clin_pictures_root = lambda: OUT_ROOT
    rep = OUT_ROOT / "report_figures"
    rep.mkdir(parents=True, exist_ok=True)

    motor = eds.MOTOR_CHANNEL_NAMES
    band = (eds.MU_LO, eds.MU_HI)
    subjects = ["CLIN_SUBJ_007", "CLIN_SUBJ_008"]
    _orig_asf = eds.apply_spatial_filter
    eds.apply_spatial_filter = (
        lambda epochs, method: epochs if method == "none"
        else _orig_asf(epochs, method))

    zmaps = []  # per-subject {channel: z-score} for MI-vs-REST
    try:
        for sid in subjects:
            vec, kept, n = eds.per_session_eds(
                sid, SESSION, band, motor, "mu", spatial_filter="none")
            if vec is None:
                print(f"  {sid}: MI-vs-REST EDS unavailable (n={n})")
                continue
            z = (vec - vec.mean()) / (vec.std(ddof=1) or 1.0)
            zmaps.append(dict(zip(kept, z)))
            print(f"  {sid}: MI-vs-REST EDS over {len(kept)} channels (n={n})")
    finally:
        eds.apply_spatial_filter = _orig_asf

    if not zmaps:
        print("No EDS maps computed; aborting.")
        return
    common = [ch for ch in motor if all(ch in m for m in zmaps)]
    pooled = np.array([np.mean([m[ch] for m in zmaps]) for ch in common])
    vmax = float(np.nanmax(np.abs(pooled))) or 1.0
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    info = eds._make_info_for_channels(common)
    im, _ = mne.viz.plot_topomap(
        pooled, info, axes=ax, cmap="viridis", show=False,
        names=common, vlim=(-vmax, vmax))
    fig.colorbar(im, ax=ax, shrink=0.8, label="EDS z-score (pooled, 2 participants)")
    ax.set_title("Electrode Discriminancy Score (EDS)\n"
                 "MI vs REST · pooled across participants", fontsize=11)
    fig.tight_layout()
    out = rep / "consolidated_eds_pooled.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out}")


_TASKS = {
    "inventory": task_inventory,
    "eds": task_eds,
    "erd": task_erd,
    "erdmean": task_erdmean,
    "topostrip": task_topostrip,
    "topocompare": task_topocompare,
    "quality": task_quality,
    "confusion": task_confusion,
    "decoder": task_decoder,
    "bar": task_bar,
    "erddiag": task_erddiag,
    "capdiag": task_capdiag,
    "report": task_report,
    "consolidate": task_consolidate,
    "consolidatetopo": task_consolidatetopo,
    "edsdiag": task_edsdiag,
    "edspooled": task_edspooled,
    "consolidateeds": task_consolidateeds,
}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in _TASKS:
        print(f"usage: {Path(__file__).name} <{'|'.join(_TASKS)}>")
        sys.exit(2)
    _TASKS[sys.argv[1]]()
