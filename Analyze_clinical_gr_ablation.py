#!/usr/bin/env python3
"""Three-arm recentering ablation for the CLIN_* (ALS) cohort.

Pass-2-fix implementation per the critic response on the pass-2 GR
ablation. The pass-2 two-arm comparison (GR-off vs Kumar GR-on) was
critiqued for using a strawman GR-off baseline; this pass adds a third
arm (Zanini 2018 RA, §5.6 in `rev00-analysis-state.md`) so the
comparison isolates "any alignment" vs "online alignment".

Three arms per session per (trial chronological order):

  Arm A (GR-off / no alignment): score the raw trace-normalised +
    shrinkage-applied trial covariance directly against the deployed
    MDM. Deliberate mismatch (MDM prototypes are post-batch-recentering).
  Arm B (Kumar 2024 online GR): apply Eq. (8)/(9) recentering per
    `Utils/runtime_common.py:307-336` (replicated in
    `exploration/clinical_analysis/gr_replay.py:gr_apply`) with state
    reset PER RUN (matching the runtime, not per session as pass-2 did).
    This is the deployed runtime path.
  Arm C (Zanini 2018 batch RA): compute the Karcher mean R of the
    rest-only covariances within each run, whiten every trial in that
    run through R^{-1/2}, then score against the same MDM. Per Zanini
    2018 §IV the rest-period covariances are the "reference state" for
    MI tasks. Implemented at `gr_replay.zanini_ra_apply`.

Key methodological changes vs pass-2:
  - Per-RUN state reset (not per session). The runtime resets GR at
    every ONLINE_* driver process invocation (verified at
    `ExperimentDriver_Online.py:103-110` with `SAVE_ADAPTIVE_T = False`
    on stable for CLIN_SUBJ_003..008 + no adaptive_T.pkl on disk for
    any of the 6 subjects). Runs within a session are detected via
    inter-trial time gaps in the concatenated marker stream (>30 s gap
    = new run; sessions have 4-6 runs of ~20 trials each per
    `logs/ONLINE_*` directories).
  - CLIN_SUBJ_002 partial inclusion (right-arm sessions S002-S004
    only; S001 is left-arm and excluded). The 13-channel MDM channel
    set is reconstructed from the older runtime convention
    `select_motor_channels(keep_prefixes=("CP","P","C"))` (verified
    against the pre-cohort git history at commit c5d2886).
  - Cadence approximation: pass-2 ran one GR update per trial; the
    runtime updates at the 1/16-s window cadence (~48 updates per
    3-s trial, ~7000 updates per ~20-trial run). The pass-2-fix keeps
    per-trial cadence but documents this as a known approximation
    (Option B in the brief). Option A (sliding 1-s windows + leaky
    integrator emulation) was deferred — it would conflate GR with
    post-processing and is out of scope for the GR ablation.

Statistics:
  - Per-session McNemar on each pair (A vs B, A vs C, B vs C).
  - Cohort paired Wilcoxon on each of the three pairs.
  - Cohort LME: `metric ~ 1 + session_idx + (1|subject)` on bal_acc_A,
    bal_acc_B, bal_acc_C, B-A delta, C-A delta, B-C delta, plus per-
    class deltas (MI_B-A, REST_B-A). The degenerate `bal_acc_off` LME
    that pass-2 reported (response identically 0.5) is dropped.
  - Bonferroni correction over 6 primary metrics, α' = 0.00833
    (matches pass-2 M5 convention).

Outputs (`~/Pictures/clin_analysis/gr_ablation/`):
  per_subject/, cohort/, csv/

Analysis-only. Tier 1 / Tier 2 files are READ-ONLY per CLAUDE.md.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar

# Make sweep helpers + clinical_analysis package importable
_REPO_ROOT = Path(__file__).resolve().parent
_SWEEP_DIR = _REPO_ROOT / "exploration" / "preprocessing_sweep"
for _p in (str(_REPO_ROOT), str(_SWEEP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exploration.clinical_analysis._helpers import (  # noqa: E402
    enumerate_clin_subjects,
    enumerate_online_sessions_for_subject,
    session_idx_from_label,
)
from config import DATA_DIR  # noqa: E402  (moved out of _helpers in 17f658)
from exploration.clinical_analysis.gr_replay import (  # noqa: E402
    GRState,
    gr_apply,
    zanini_ra_apply,
)

# Config A preprocessing constants (sweep_phase2_round2.py:63-73)
from sweep_phase2_round2 import (  # noqa: E402
    BB_HI, BB_LO, FS, MU_HI, MU_LO, NOTCH, REJECT_MAX_ABS_UV,
    apply_blink_removal, apply_spatial_filter, load_raw_cached,
)
# Auto-drop loop knobs (sweep_phase3_validation.py:123-126)
from sweep_phase3_validation import (  # noqa: E402
    AUTO_DROP_DOMINANCE_FRAC, AUTO_DROP_MAX_CHANNELS,
    AUTO_DROP_MAX_ITERS, AUTO_DROP_REJECT_FRAC,
    _pick_dominant_bad_channel_max_abs,
)

from pyriemann.estimation import Shrinkage  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402

import mne  # noqa: E402
mne.set_log_level("ERROR")

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

# Offline-replay covariance window. The runtime classifies at 1-s
# sliding windows (`config.CLASSIFY_WINDOW = 1000` ms for
# CLIN_SUBJ_003..008, 500 ms for CLIN_SUBJ_002). The offline replay uses
# ONE covariance per trial over (1, 4) s — matches the canonical
# offline-replay choice and the pass-1 ERD window
# (`Analyze_eds_topoplot_CLIN.py:86`). The implication (3× larger sample
# window → better-conditioned cov → shrinkage less impactful) is
# documented as a known divergence from the runtime; see the report.
SCALAR_WIN = (1.0, 4.0)
TRIAL_WIN = (-1.0, 4.0)

# 15-channel motor subset (matches `config.MOTOR_CHANNEL_NAMES` on
# stable). The deployed MDM bundles for CLIN_SUBJ_003..008 have
# `covmeans_.shape = (2, 15, 15)`.
MOTOR_CHANNELS_15 = [
    "FC1", "FC2", "C3", "Cz", "C4", "CP5", "CP1", "CP2", "CP6",
    "P7",  "P3",  "Pz", "P4", "P8", "POz",
]

# Run-boundary detection: a >30-s gap between consecutive trial-start
# markers (codes 100/200) indicates a new ONLINE_* driver process.
# Calibrated on CLIN_SUBJ_005/S001ONLINE (5 detected boundaries → 6
# runs matching `logs/ONLINE_*` count). Within-run trial spacing is
# ~10-22 s; the gap between successive runs is >30 s (driver restart).
RUN_GAP_THRESHOLD_S = 30.0

# Bonferroni: 6 primary metrics per the brief (balanced acc, NKV, MI
# acc, REST acc, kappa, sample-level acc). Matches pass-2 M5 convention.
BONFERRONI_N_PRIMARY = 6
BONFERRONI_ALPHA = 0.05 / BONFERRONI_N_PRIMARY  # ≈ 0.00833

# CLIN_SUBJ_002 protocol divergences (motor channel set, right-arm
# valid sessions, LedoitWolf shrinkage) live in the shared module so
# the five clinical-analysis scripts that special-case her stay in
# sync (see exploration/clinical_analysis/_subj002.py).
from exploration.clinical_analysis._subj002 import (  # noqa: E402
    SUBJ002_MOTOR_CHANNELS_13 as _SUBJ002_MOTOR_CHANNELS_13,
    subj002_valid_sessions as _subj002_valid_sessions,
)
MOTOR_CHANNELS_13 = list(_SUBJ002_MOTOR_CHANNELS_13)
CLIN002_RIGHT_ARM_SESSIONS = set(_subj002_valid_sessions())


# ----------------------------------------------------------------------
# Per-subject runtime config (LedoitWolf vs pyriemann Shrinkage)
# ----------------------------------------------------------------------

def _runtime_shrinkage_for(subject: str, session: str) -> tuple[bool, float]:
    """Return (use_ledoitwolf, shrinkage_param) for the runtime as it
    was configured for this (subject, session). Falls back to the
    stable defaults if no snapshot is found.

    Note: when LEDOITWOLF=1 the runtime uses an *adaptive* λ fit on the
    raw EEG window (see `Utils/runtime_common.py:283`), not the
    SHRINKAGE_PARAM constant. We replicate that behaviour in
    `_shrink_ledoitwolf_adaptive` below; the returned `shrinkage_param`
    is then unused for CLIN_SUBJ_002.
    """
    if subject != "CLIN_SUBJ_002":
        return False, 0.02
    logs = Path(DATA_DIR) / f"sub-{subject}" / f"ses-{session}" / "logs"
    if logs.is_dir():
        for run_dir in sorted(logs.iterdir()):
            snap = run_dir / "config_snapshot.json"
            if snap.is_file():
                try:
                    d = json.loads(snap.read_text())
                    lw = bool(d.get("LEDOITWOLF", 1))
                    lam = float(d.get("SHRINKAGE_PARAM", 0.05))
                    return lw, lam
                except Exception:
                    continue
    return True, 0.1


def _motor_channels_for(subject: str) -> list[str]:
    """Return the canonical motor channel list for this subject's MDM."""
    if subject == "CLIN_SUBJ_002":
        return list(MOTOR_CHANNELS_13)
    return list(MOTOR_CHANNELS_15)


# ----------------------------------------------------------------------
# Config-A time-domain mu-band epoch builder
# ----------------------------------------------------------------------

def _config_a_mu_epochs(raw, events, event_dict):
    """Build Config-A mu-band time-domain epochs at SCALAR_WIN.

    Mirrors `Analyze_eds_topoplot_CLIN.py:178-268` for the mu band
    (8-13 Hz). Returns (data, labels, event_sample_idx, dropped, ch_names).
    `event_sample_idx` is the per-kept-epoch sample index into the raw
    timeline — used downstream to map epochs to runs by time gap.
    """
    band = (MU_LO, MU_HI)
    raw_bb = raw.copy()
    raw_1hz = raw.copy()
    raw_bb.notch_filter(NOTCH, method="iir", verbose=False)
    raw_bb.filter(l_freq=BB_LO, h_freq=BB_HI, method="iir", verbose=False)
    raw_bb, _ = apply_blink_removal(raw_bb, raw_1hz, "drop_fp")

    dropped: list[str] = []
    iters = 0
    t0, t1 = TRIAL_WIN
    good_ix: list[int] = []
    epochs_bb = None
    while True:
        iters += 1
        raw_mu = raw_bb.copy()
        raw_mu.filter(l_freq=MU_LO, h_freq=MU_HI, method="iir", verbose=False)
        epoch_kw = dict(
            event_id=event_dict, tmin=t0, tmax=t1,
            baseline=None, detrend=1, preload=True, verbose=False,
        )
        epochs_mu = mne.Epochs(
            raw_mu, events, reject=None, flat=None, **epoch_kw,
        )
        epochs_bb = mne.Epochs(
            raw_bb, events, reject=None, flat=None, **epoch_kw,
        )
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

    if not good_ix or epochs_bb is None:
        return None, None, None, dropped, []

    epochs_bb = apply_spatial_filter(epochs_bb, "car")
    epochs_band = epochs_bb.copy().filter(
        l_freq=band[0], h_freq=band[1], method="iir", verbose=False,
    )
    epochs_band = epochs_band[good_ix]
    epochs_band.crop(tmin=SCALAR_WIN[0], tmax=SCALAR_WIN[1])

    labels = epochs_band.events[:, 2].astype(int)
    data = epochs_band.get_data()
    ch_names = list(epochs_band.ch_names)
    # epochs.events[:, 0] is the sample index of the cue marker in the
    # (concatenated) raw timeline — used to detect run boundaries via
    # inter-marker time gap.
    event_samples = epochs_band.events[:, 0].astype(int)
    return data, labels, event_samples, dropped, ch_names


# ----------------------------------------------------------------------
# Trial covariance — matches stable Utils/runtime_common.py:248-285
# ----------------------------------------------------------------------

def _trace_normalised_cov(x: np.ndarray) -> np.ndarray:
    """C = X X^T / trace(X X^T). Stable Utils/runtime_common.py:261-265."""
    c = x @ x.T
    tr = np.trace(c)
    if tr <= 0 or not np.isfinite(tr):
        return c
    return c / tr


def _shrink_pyriemann(covs: np.ndarray, lam: float) -> np.ndarray:
    """pyriemann.estimation.Shrinkage(λ). Matches stable
    Utils/runtime_common.py:287-289 (DECODER_BACKEND=mdm path)."""
    return Shrinkage(shrinkage=lam).fit_transform(covs)


def _shrink_ledoitwolf_adaptive(
    epoch_data: np.ndarray, raw_covs: np.ndarray,
) -> np.ndarray:
    """LedoitWolf with adaptive λ per trial, then convex combination
    with identity. Mirrors stable Utils/runtime_common.py:275-285:

        lam = LedoitWolf().fit(raw_window.T).shrinkage_
        cov_shrunk = (1 - lam) * cov + lam * (trace/n) * I

    `epoch_data[i]` is the (n_ch, n_t) trial window; `raw_covs[i]` is
    the already-trace-normalised covariance.
    """
    n_trials, n_ch, _ = epoch_data.shape
    out = np.zeros_like(raw_covs)
    eye = np.eye(n_ch)
    for i in range(n_trials):
        lam = LedoitWolf().fit(epoch_data[i].T).shrinkage_
        cov = raw_covs[i]
        out[i] = (1 - lam) * cov + lam * (np.trace(cov) / n_ch) * eye
    return out


def _trial_covs(
    epoch_data: np.ndarray, *, use_lw: bool, lam: float,
) -> np.ndarray:
    raw_covs = np.stack(
        [_trace_normalised_cov(seg) for seg in epoch_data], axis=0,
    )
    if use_lw:
        return _shrink_ledoitwolf_adaptive(epoch_data, raw_covs)
    return _shrink_pyriemann(raw_covs, lam)


# ----------------------------------------------------------------------
# Channel restriction
# ----------------------------------------------------------------------

def _restrict_to_motor(
    data: np.ndarray, ch_names: list[str],
    motor_channels: list[str],
) -> tuple[np.ndarray, list[str]]:
    keep = [c for c in motor_channels if c in ch_names]
    if not keep:
        return data[:, :0, :], []
    idx = [ch_names.index(c) for c in keep]
    return data[:, idx, :], keep


# ----------------------------------------------------------------------
# Run boundary detection
# ----------------------------------------------------------------------

def _split_into_runs(
    event_samples: np.ndarray, fs: int,
) -> list[np.ndarray]:
    """Split kept-epoch indices into per-run index groups.

    Within a session, the concatenated marker stream contains successive
    ONLINE_* runs joined by inter-run gaps when the driver is restarted.
    Within-run trial spacing is ~10-22 s; between-run gap is >30 s
    (calibrated on CLIN_SUBJ_005/S001ONLINE — 5 detected boundaries
    match the 6 ONLINE_* subdirectories in that session's `logs/`).

    Returns a list of integer arrays; each array indexes into the
    chronological epoch order.
    """
    if len(event_samples) == 0:
        return []
    t_s = event_samples / float(fs)
    diffs = np.diff(t_s)
    boundaries = np.where(diffs > RUN_GAP_THRESHOLD_S)[0] + 1
    cuts = [0] + boundaries.tolist() + [len(event_samples)]
    return [np.arange(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


# ----------------------------------------------------------------------
# Per-session three-arm replay
# ----------------------------------------------------------------------

def _load_mdm(subject: str):
    """Load the per-subject deployed MDM model bundle."""
    path = (
        Path(DATA_DIR) / f"sub-{subject}" / "models"
        / f"sub-{subject}_model.pkl"
    )
    with open(path, "rb") as f:
        return pickle.load(f), path


def _per_class_acc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Return (acc_MI, acc_REST). MI=200, REST=100. NaN if a class is
    not present in `y_true`."""
    mi_mask = y_true == 200
    re_mask = y_true == 100
    acc_mi = float((y_pred[mi_mask] == 200).mean()) if mi_mask.any() else np.nan
    acc_re = float((y_pred[re_mask] == 100).mean()) if re_mask.any() else np.nan
    return acc_mi, acc_re


def _mcnemar(y_true: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> dict:
    """Per-trial paired McNemar of (Arm A correct vs Arm B correct).

    NB: when one arm is degenerate (constant-class prediction), McNemar
    reduces to a binomial test on the other arm's per-class correctness
    — this is acknowledged in the pass-2 critic M2 and remains true for
    Arm A here. The three-arm comparison (A vs B, A vs C, B vs C)
    mitigates: Arm B vs Arm C McNemar is non-degenerate when both arms
    produce non-trivial decisions.
    """
    ok_a = (pa == y_true).astype(int)
    ok_b = (pb == y_true).astype(int)
    n11 = int(((ok_a == 1) & (ok_b == 1)).sum())
    n10 = int(((ok_a == 1) & (ok_b == 0)).sum())
    n01 = int(((ok_a == 0) & (ok_b == 1)).sum())
    n00 = int(((ok_a == 0) & (ok_b == 0)).sum())
    table = [[n11, n10], [n01, n00]]
    discordant = n10 + n01
    if discordant == 0:
        return dict(
            n11=n11, n10=n10, n01=n01, n00=n00,
            statistic=np.nan, p=1.0,
        )
    try:
        res = mcnemar(
            table, exact=(discordant < 25), correction=True,
        )
        return dict(
            n11=n11, n10=n10, n01=n01, n00=n00,
            statistic=float(res.statistic),
            p=float(res.pvalue),
        )
    except Exception:
        return dict(
            n11=n11, n10=n10, n01=n01, n00=n00,
            statistic=np.nan, p=np.nan,
        )


def _three_arm_predict(
    covs: np.ndarray, labels: np.ndarray, run_ix: list[np.ndarray],
    mdm,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run three arms on `covs` in chronological order, with per-run
    state reset for Arm B and per-run Karcher-mean batch for Arm C.

    Returns (pred_off, pred_on, pred_ra) — each int array of length N.
    """
    n = covs.shape[0]
    pred_off = np.empty(n, dtype=int)
    pred_on = np.empty(n, dtype=int)
    pred_ra = np.empty(n, dtype=int)

    for ix in run_ix:
        if len(ix) == 0:
            continue
        run_covs = covs[ix]
        run_labels = labels[ix]

        # Arm A — no recentering. One MDM.predict per trial.
        for i, k in enumerate(ix):
            pred_off[k] = int(mdm.predict(run_covs[i][np.newaxis, ...])[0])

        # Arm B — Kumar GR, state reset at run start.
        state = GRState()
        for i, k in enumerate(ix):
            c_rec = gr_apply(state, run_covs[i])
            pred_on[k] = int(mdm.predict(c_rec[np.newaxis, ...])[0])

        # Arm C — Zanini RA. Karcher mean of rest covs in this run.
        # Fall back to "all" reference if a run has no rest trials
        # (vanishingly rare in this cohort; e.g. a single-trial
        # aborted run with class 200 only).
        try:
            covs_ra, _ = zanini_ra_apply(
                run_covs, run_labels, reference="rest", rest_label=100,
            )
        except ValueError:
            covs_ra, _ = zanini_ra_apply(
                run_covs, run_labels, reference="all", rest_label=100,
            )
        for i, k in enumerate(ix):
            pred_ra[k] = int(mdm.predict(covs_ra[i][np.newaxis, ...])[0])

    return pred_off, pred_on, pred_ra


def _gr_replay_session(
    subject: str, session: str, mdm,
    motor_channels: list[str],
) -> dict | None:
    """Run three-arm GR/RA replay for one session.

    Returns a dict with per-trial labels + per-session summary metrics,
    or None on preprocessing failure.
    """
    try:
        raw, events, event_dict = load_raw_cached(subject, session)
    except Exception as e:
        print(
            f"  [{subject}/{session}] FAILED load: "
            f"{type(e).__name__}: {e}"
        )
        return None
    try:
        data, labels, event_samples, dropped, ch_names = _config_a_mu_epochs(
            raw, events, event_dict,
        )
    except Exception as e:
        print(
            f"  [{subject}/{session}] FAILED preproc: "
            f"{type(e).__name__}: {e}"
        )
        return None
    if data is None or len(data) == 0:
        print(f"  [{subject}/{session}] no kept epochs; skip")
        return None

    data_motor, motor_kept = _restrict_to_motor(
        data, ch_names, motor_channels,
    )
    if len(motor_kept) != mdm.covmeans_.shape[1]:
        print(
            f"  [{subject}/{session}] motor channel count "
            f"({len(motor_kept)}) != MDM channel count "
            f"({mdm.covmeans_.shape[1]}); skip"
        )
        return None

    use_lw, lam = _runtime_shrinkage_for(subject, session)
    covs = _trial_covs(data_motor, use_lw=use_lw, lam=lam)

    run_ix = _split_into_runs(event_samples, FS)

    pred_off, pred_on, pred_ra = _three_arm_predict(
        covs, labels, run_ix, mdm,
    )

    y_true = labels

    def _metrics(p):
        acc = float((p == y_true).mean())
        bal = balanced_accuracy_score(y_true, p)
        mi, re = _per_class_acc(y_true, p)
        kappa = (
            float(cohen_kappa_score(y_true, p, labels=[100, 200]))
            if len(np.unique(y_true)) >= 2 else np.nan
        )
        return acc, bal, mi, re, kappa

    acc_a, bal_a, mi_a, re_a, kappa_a = _metrics(pred_off)
    acc_b, bal_b, mi_b, re_b, kappa_b = _metrics(pred_on)
    acc_c, bal_c, mi_c, re_c, kappa_c = _metrics(pred_ra)

    mcn_BA = _mcnemar(y_true, pred_off, pred_on)
    mcn_CA = _mcnemar(y_true, pred_off, pred_ra)
    mcn_BC = _mcnemar(y_true, pred_ra, pred_on)

    return dict(
        subject=subject, session=session,
        session_idx=session_idx_from_label(session),
        n_trials=int(len(y_true)),
        n_mi=int((y_true == 200).sum()),
        n_rest=int((y_true == 100).sum()),
        n_runs_detected=len(run_ix),
        dropped=",".join(dropped) if dropped else "",
        motor_n_channels=len(motor_kept),
        # Arm A (GR-off)
        acc_off=acc_a, bal_acc_off=bal_a,
        mi_acc_off=mi_a, rest_acc_off=re_a, kappa_off=kappa_a,
        # Arm B (Kumar GR-on)
        acc_on=acc_b, bal_acc_on=bal_b,
        mi_acc_on=mi_b, rest_acc_on=re_b, kappa_on=kappa_b,
        # Arm C (Zanini RA)
        acc_ra=acc_c, bal_acc_ra=bal_c,
        mi_acc_ra=mi_c, rest_acc_ra=re_c, kappa_ra=kappa_c,
        # Pairwise deltas
        bal_acc_delta_BA=bal_b - bal_a,
        bal_acc_delta_CA=bal_c - bal_a,
        bal_acc_delta_BC=bal_b - bal_c,
        mi_acc_delta_BA=mi_b - mi_a,
        rest_acc_delta_BA=re_b - re_a,
        mi_acc_delta_BC=mi_b - mi_c,
        rest_acc_delta_BC=re_b - re_c,
        kappa_delta_BA=kappa_b - kappa_a,
        kappa_delta_CA=kappa_c - kappa_a,
        kappa_delta_BC=kappa_b - kappa_c,
        # Pairwise McNemar
        mcnemar_BA_stat=mcn_BA["statistic"], mcnemar_BA_p=mcn_BA["p"],
        mcnemar_BA_n11=mcn_BA["n11"], mcnemar_BA_n10=mcn_BA["n10"],
        mcnemar_BA_n01=mcn_BA["n01"], mcnemar_BA_n00=mcn_BA["n00"],
        mcnemar_CA_stat=mcn_CA["statistic"], mcnemar_CA_p=mcn_CA["p"],
        mcnemar_BC_stat=mcn_BC["statistic"], mcnemar_BC_p=mcn_BC["p"],
        # Per-trial detail
        _y_true=y_true,
        _pred_off=pred_off,
        _pred_on=pred_on,
        _pred_ra=pred_ra,
    )


# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------

def _bonferroni_verdict(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return (
        f"PASS (p<{BONFERRONI_ALPHA:.4f})"
        if p < BONFERRONI_ALPHA
        else f"FAIL (p>={BONFERRONI_ALPHA:.4f})"
    )


def _plot_per_subject(
    df_subj: pd.DataFrame, subject: str, out_path: Path,
):
    """Per-subject 3-line plot: Arm A vs Arm B vs Arm C balanced acc."""
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    sub = df_subj.sort_values("session_idx")
    ax.plot(
        sub["session_idx"], sub["bal_acc_off"], "o-",
        color="tab:orange", lw=2, markersize=8, label="Arm A (GR-off)",
    )
    ax.plot(
        sub["session_idx"], sub["bal_acc_on"], "s-",
        color="tab:blue", lw=2, markersize=8, label="Arm B (Kumar GR)",
    )
    ax.plot(
        sub["session_idx"], sub["bal_acc_ra"], "^-",
        color="tab:green", lw=2, markersize=8, label="Arm C (Zanini RA)",
    )
    ax.axhline(0.5, color="k", lw=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Session index")
    ax.set_ylabel("Balanced accuracy")
    ax.set_xticks(sub["session_idx"].unique())
    all_vals = sub[["bal_acc_off", "bal_acc_on", "bal_acc_ra"]].values
    ax.set_ylim(min(0.0, float(np.nanmin(all_vals)) - 0.05),
                max(1.05, float(np.nanmax(all_vals)) + 0.05))
    ax.set_title(
        f"{subject} — three-arm balanced accuracy across sessions "
        f"(n={int(sub['n_trials'].sum())} trials)",
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_cohort_three_arm(
    df: pd.DataFrame, out_path: Path,
    *, lme_annotation: str | None = None,
):
    """Cohort three-arm comparison: per-session bal_acc by arm, with
    cohort means overlaid."""
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    arms = [
        ("bal_acc_off", "Arm A (GR-off)", "tab:orange", "o"),
        ("bal_acc_on", "Arm B (Kumar GR)", "tab:blue", "s"),
        ("bal_acc_ra", "Arm C (Zanini RA)", "tab:green", "^"),
    ]
    for col, label, color, marker in arms:
        cohort = (
            df.groupby("session_idx")[col]
            .agg(["mean", "sem"])
            .reset_index()
        )
        ax.errorbar(
            cohort["session_idx"], cohort["mean"],
            yerr=cohort["sem"],
            color=color, lw=2.0, marker=marker, markersize=9,
            label=label, capsize=3,
        )
    ax.axhline(0.5, color="k", lw=0.6, linestyle="--", alpha=0.5)
    ax.set_xlabel("Session index")
    ax.set_ylabel("Per-session balanced accuracy (cohort mean ± SE)")
    ax.set_title("CLIN cohort — three-arm recentering ablation")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    if lme_annotation:
        ax.text(
            0.02, 0.02, lme_annotation, transform=ax.transAxes,
            fontsize=8, va="bottom", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_cohort_paired_delta(
    df: pd.DataFrame, out_path: Path,
    *, lme_annotation: str | None = None,
):
    """Cohort paired-delta plot: per-session (B-A and B-C) over
    session_idx, with cohort mean ± SE line."""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for col, label, color in [
        ("bal_acc_delta_BA", "Δ(B-A) = Kumar GR − GR-off", "tab:blue"),
        ("bal_acc_delta_BC", "Δ(B-C) = Kumar GR − Zanini RA", "tab:purple"),
        ("bal_acc_delta_CA", "Δ(C-A) = Zanini RA − GR-off", "tab:green"),
    ]:
        cohort = (
            df.groupby("session_idx")[col]
            .agg(["mean", "sem"])
            .reset_index()
        )
        ax.errorbar(
            cohort["session_idx"], cohort["mean"],
            yerr=cohort["sem"],
            color=color, lw=2.0, marker="s", markersize=8,
            label=label, capsize=3,
        )
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("Session index")
    ax.set_ylabel(r"$\Delta$ balanced accuracy")
    ax.set_title("CLIN cohort — pairwise arm deltas over sessions")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    if lme_annotation:
        ax.text(
            0.02, 0.02, lme_annotation, transform=ax.transAxes,
            fontsize=8, va="bottom", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_cohort_box(df: pd.DataFrame, out_path: Path):
    """Boxplot of per-session balanced accuracy for the three arms."""
    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    data = [
        df["bal_acc_off"].dropna().values,
        df["bal_acc_on"].dropna().values,
        df["bal_acc_ra"].dropna().values,
    ]
    ax.boxplot(
        data,
        tick_labels=["Arm A\n(GR-off)", "Arm B\n(Kumar GR)",
                     "Arm C\n(Zanini RA)"],
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red",
                       markeredgecolor="black"),
    )
    for _, row in df.iterrows():
        ax.plot(
            [1, 2, 3],
            [row["bal_acc_off"], row["bal_acc_on"], row["bal_acc_ra"]],
            color="gray", alpha=0.35, lw=0.7, marker="o", markersize=3,
        )
    ax.axhline(0.5, color="k", lw=0.5, linestyle="--", alpha=0.4)
    ax.set_ylabel("Per-session balanced accuracy")
    ax.set_title(
        f"CLIN cohort — per-session balanced accuracy by arm "
        f"(n_sessions = {len(df)})",
        fontsize=10,
    )
    ax.grid(True, alpha=0.25)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_mcnemar_heatmap(
    df: pd.DataFrame, out_path: Path,
    *, col: str = "mcnemar_BA_p", title_suffix: str = "Arm B vs Arm A",
):
    """Subject × session heatmap of McNemar p-values (−log10)."""
    pivot = df.pivot_table(
        index="subject", columns="session_idx", values=col,
        aggfunc="first",
    )
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    mat = -np.log10(pivot.values.astype(float))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([f"S{int(s):03d}" for s in pivot.columns])
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(list(pivot.index))
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=8, color="white")
                continue
            txt = f"{v:.2g}"
            color = "white" if mat[i, j] > np.nanmedian(mat) else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7, color=color)
    bonf_thresh = 0.05 / max(1, np.isfinite(pivot.values.astype(float)).sum())
    fig.colorbar(im, ax=ax, label=r"$-\log_{10}$ McNemar p")
    ax.set_title(
        f"McNemar p-values — {title_suffix}\n"
        f"Uncorrected α=0.05; Bonferroni α'≈{bonf_thresh:.4f}",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------

def _per_subject_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-subject means + cross-session paired Wilcoxon for
    the three pairwise contrasts."""
    rows = []
    for subj, sub in df.groupby("subject"):

        def _wil(a, b):
            a = np.asarray(a)
            b = np.asarray(b)
            if len(a) >= 3 and not np.allclose(a, b):
                try:
                    res = wilcoxon(a, b, alternative="two-sided")
                    return float(res.statistic), float(res.pvalue)
                except Exception:
                    return np.nan, np.nan
            return np.nan, np.nan

        a = sub["bal_acc_off"].values
        b = sub["bal_acc_on"].values
        c = sub["bal_acc_ra"].values
        W_BA, p_BA = _wil(a, b)
        W_BC, p_BC = _wil(c, b)
        W_CA, p_CA = _wil(a, c)
        rows.append(dict(
            subject=subj, n_sessions=len(sub),
            mean_bal_acc_off=float(np.nanmean(a)),
            mean_bal_acc_on=float(np.nanmean(b)),
            mean_bal_acc_ra=float(np.nanmean(c)),
            mean_bal_acc_delta_BA=float(np.nanmean(b - a)),
            mean_bal_acc_delta_BC=float(np.nanmean(b - c)),
            mean_bal_acc_delta_CA=float(np.nanmean(c - a)),
            mean_kappa_off=float(np.nanmean(sub["kappa_off"])),
            mean_kappa_on=float(np.nanmean(sub["kappa_on"])),
            mean_kappa_ra=float(np.nanmean(sub["kappa_ra"])),
            paired_wilcoxon_BA_W=W_BA, paired_wilcoxon_BA_p=p_BA,
            paired_wilcoxon_BC_W=W_BC, paired_wilcoxon_BC_p=p_BC,
            paired_wilcoxon_CA_W=W_CA, paired_wilcoxon_CA_p=p_CA,
        ))
    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


def _cohort_paired_wilcoxon(
    df: pd.DataFrame, col_a: str, col_b: str,
) -> dict:
    """Paired Wilcoxon on per-session col_b − col_a."""
    a = df[col_a].dropna().values
    b = df[col_b].dropna().values
    if len(a) < 3 or np.allclose(a, b):
        return dict(W=np.nan, p=np.nan, n=len(a))
    try:
        res = wilcoxon(a, b, alternative="two-sided")
        return dict(W=float(res.statistic), p=float(res.pvalue), n=len(a))
    except Exception:
        return dict(W=np.nan, p=np.nan, n=len(a))


def _cohort_lme(df: pd.DataFrame, metric: str) -> dict:
    """Fit `metric ~ 1 + session_idx + (1|subject)`. Skip metrics with
    zero variance (e.g. degenerate Arm A on the 15-channel cohort)."""
    if not HAS_STATSMODELS:
        return dict(slope=np.nan, slope_p=np.nan, intercept=np.nan,
                    status="no statsmodels")
    sub = df.dropna(subset=[metric, "session_idx", "subject"]).copy()
    if len(sub) < 5:
        return dict(slope=np.nan, slope_p=np.nan, intercept=np.nan,
                    status=f"only {len(sub)} rows")
    if float(np.nanstd(sub[metric].values)) == 0.0:
        return dict(slope=np.nan, slope_p=np.nan, intercept=np.nan,
                    status="zero-variance metric; LME skipped")
    import warnings as _w
    try:
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            model = smf.mixedlm(
                f"{metric} ~ 1 + session_idx",
                sub, groups=sub["subject"],
            ).fit(disp=False)
        return dict(
            slope=float(model.params.get("session_idx", np.nan)),
            slope_p=float(model.pvalues.get("session_idx", np.nan)),
            intercept=float(model.params.get("Intercept", np.nan)),
            intercept_p=float(model.pvalues.get("Intercept", np.nan)),
            llf=float(model.llf),
            n=len(sub),
            status="ok",
        )
    except Exception as e:
        return dict(slope=np.nan, slope_p=np.nan, intercept=np.nan,
                    status=f"failed: {type(e).__name__}: {e}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exclude-clin002", action="store_true",
        help=(
            "Exclude CLIN_SUBJ_002 entirely. Default includes its "
            "right-arm sessions S002-S004 (S001 is left-arm and is "
            "always excluded to avoid conflating MI directions)."
        ),
    )
    args = parser.parse_args()

    out_root = Path.home() / "Pictures" / "clin_analysis" / "gr_ablation"
    (out_root / "per_subject").mkdir(parents=True, exist_ok=True)
    (out_root / "cohort").mkdir(parents=True, exist_ok=True)
    (out_root / "csv").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    trial_rows: list[dict] = []

    subjects = enumerate_clin_subjects()
    if args.exclude_clin002:
        subjects = [s for s in subjects if s != "CLIN_SUBJ_002"]
        print(
            "[note] CLIN_SUBJ_002 excluded via --exclude-clin002."
        )

    for subject in subjects:
        try:
            mdm, mdm_path = _load_mdm(subject)
        except Exception as e:
            print(
                f"[{subject}] FAILED to load MDM bundle: "
                f"{type(e).__name__}: {e}; skip"
            )
            continue
        n_ch = mdm.covmeans_.shape[1]
        motor_channels = _motor_channels_for(subject)
        print(
            f"\n=== {subject} (MDM covmeans {n_ch}x{n_ch}, "
            f"motor channels={len(motor_channels)}, "
            f"path={mdm_path.name}) ==="
        )
        sessions = enumerate_online_sessions_for_subject(subject)
        # CLIN_SUBJ_002 right-arm-only filter (S001 is left-arm).
        if subject == "CLIN_SUBJ_002":
            sessions = [
                s for s in sessions
                if s in CLIN002_RIGHT_ARM_SESSIONS
            ]
            print(
                f"[note] CLIN_SUBJ_002 restricted to right-arm sessions "
                f"({sorted(CLIN002_RIGHT_ARM_SESSIONS)}); S001 (left-arm) "
                f"excluded."
            )
        for sess in sessions:
            t0 = time.time()
            r = _gr_replay_session(subject, sess, mdm, motor_channels)
            if r is None:
                continue
            y_true = r.pop("_y_true")
            pred_off = r.pop("_pred_off")
            pred_on = r.pop("_pred_on")
            pred_ra = r.pop("_pred_ra")
            for ix in range(len(y_true)):
                trial_rows.append(dict(
                    subject=subject, session=sess,
                    session_idx=r["session_idx"],
                    trial_idx_in_session=int(ix),
                    true_label=int(y_true[ix]),
                    pred_off=int(pred_off[ix]),
                    pred_on=int(pred_on[ix]),
                    pred_ra=int(pred_ra[ix]),
                ))
            rows.append(r)
            print(
                f"  {sess}: n={r['n_trials']} "
                f"({r['n_runs_detected']} runs) | "
                f"A={r['bal_acc_off']:.3f} "
                f"B={r['bal_acc_on']:.3f} "
                f"C={r['bal_acc_ra']:.3f} | "
                f"Δ(B-A)={r['bal_acc_delta_BA']:+.3f} "
                f"Δ(B-C)={r['bal_acc_delta_BC']:+.3f} "
                f"({time.time()-t0:.1f}s)"
            )

    if not rows:
        print("No sessions completed; aborting.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df_trials = pd.DataFrame(trial_rows)
    df.to_csv(out_root / "csv" / "gr_ablation_session_summary.csv", index=False)
    df_trials.to_csv(out_root / "csv" / "gr_ablation_per_trial.csv", index=False)

    # ----- Per-subject plots -----
    for subj in sorted(df["subject"].unique()):
        _plot_per_subject(
            df[df.subject == subj], subj,
            out_root / "per_subject" / f"{subj}_three_arm_bal_acc.png",
        )

    # ----- Per-subject summary + paired Wilcoxon -----
    df_subj = _per_subject_summary(df)
    df_subj.to_csv(
        out_root / "csv" / "gr_ablation_per_subject_summary.csv",
        index=False,
    )

    # ----- Cohort stats -----
    cohort_pw_BA = _cohort_paired_wilcoxon(df, "bal_acc_off", "bal_acc_on")
    cohort_pw_BC = _cohort_paired_wilcoxon(df, "bal_acc_ra", "bal_acc_on")
    cohort_pw_CA = _cohort_paired_wilcoxon(df, "bal_acc_off", "bal_acc_ra")

    # LME fits — skip the degenerate `bal_acc_off` row if its variance
    # is zero (pass-2-fix M5: drop the degenerate LME from the report).
    lme_metrics = [
        "bal_acc_on", "bal_acc_ra",
        "bal_acc_delta_BA", "bal_acc_delta_BC", "bal_acc_delta_CA",
        "mi_acc_delta_BA", "rest_acc_delta_BA",
        "mi_acc_delta_BC", "rest_acc_delta_BC",
        "kappa_delta_BA", "kappa_delta_BC",
    ]
    lme_rows = {m: _cohort_lme(df, m) for m in lme_metrics}

    headline = {
        "metric_bal_acc_delta_BA": dict(
            test="Cohort paired Wilcoxon (bal_acc_on - bal_acc_off)",
            **cohort_pw_BA,
            bonferroni_alpha=BONFERRONI_ALPHA,
            bonferroni_verdict=_bonferroni_verdict(cohort_pw_BA["p"]),
        ),
        "metric_bal_acc_delta_BC": dict(
            test="Cohort paired Wilcoxon (bal_acc_on - bal_acc_ra)",
            **cohort_pw_BC,
            bonferroni_alpha=BONFERRONI_ALPHA,
            bonferroni_verdict=_bonferroni_verdict(cohort_pw_BC["p"]),
        ),
        "metric_bal_acc_delta_CA": dict(
            test="Cohort paired Wilcoxon (bal_acc_ra - bal_acc_off)",
            **cohort_pw_CA,
            bonferroni_alpha=BONFERRONI_ALPHA,
            bonferroni_verdict=_bonferroni_verdict(cohort_pw_CA["p"]),
        ),
    }
    for metric, lme in lme_rows.items():
        headline[f"lme_{metric}"] = dict(
            **lme,
            bonferroni_alpha=BONFERRONI_ALPHA,
            bonferroni_verdict_slope=_bonferroni_verdict(lme.get("slope_p", np.nan)),
        )
    with open(out_root / "csv" / "gr_ablation_cohort_stats.json", "w") as f:
        json.dump(headline, f, indent=2, default=str)

    # ----- LME results txt -----
    txt_lines: list[str] = []
    txt_lines.append("# CLIN three-arm recentering ablation — cohort statistics")
    txt_lines.append("")
    txt_lines.append("# Arms:")
    txt_lines.append("#   A = GR-off (raw cov scored against trained MDM)")
    txt_lines.append("#   B = Kumar 2024 online GR (deployed runtime path)")
    txt_lines.append("#   C = Zanini 2018 batch RA (Karcher mean of rest covs per run)")
    txt_lines.append("# State reset: per-RUN for arm B (matches runtime per pass-2-fix C2)")
    txt_lines.append("")
    for pair, label, pw in [
        ("BA", "Kumar GR vs GR-off",          cohort_pw_BA),
        ("BC", "Kumar GR vs Zanini RA",        cohort_pw_BC),
        ("CA", "Zanini RA vs GR-off",          cohort_pw_CA),
    ]:
        txt_lines.append(
            f"Cohort paired Wilcoxon ({label}) on per-session: "
            f"W = {pw['W']}, p = {pw['p']:.4g}, n = {pw['n']}. "
            f"Bonferroni (α' = {BONFERRONI_ALPHA:.4f}, n_primary = "
            f"{BONFERRONI_N_PRIMARY}): {_bonferroni_verdict(pw['p'])}"
        )
    txt_lines.append("")
    for metric, lme in lme_rows.items():
        txt_lines.append(f"=== LME: {metric} ~ 1 + session_idx + (1|subject) ===")
        for k, v in lme.items():
            txt_lines.append(f"    {k} = {v}")
        sp = lme.get("slope_p", float("nan"))
        txt_lines.append(
            "    Bonferroni-corrected slope test "
            f"(α' = {BONFERRONI_ALPHA:.4f}): "
            f"slope p = {sp:.4g} → {_bonferroni_verdict(sp)}"
        )
        txt_lines.append("")
    txt_lines.append("=== Per-subject paired Wilcoxon (sessions within subject) ===")
    for _, r in df_subj.iterrows():
        txt_lines.append(
            f"  {r['subject']}: n_sess={r['n_sessions']}\n"
            f"    Δ(B-A) mean = {r['mean_bal_acc_delta_BA']:+.3f}, "
            f"W = {r['paired_wilcoxon_BA_W']}, "
            f"p = {r['paired_wilcoxon_BA_p']}\n"
            f"    Δ(B-C) mean = {r['mean_bal_acc_delta_BC']:+.3f}, "
            f"W = {r['paired_wilcoxon_BC_W']}, "
            f"p = {r['paired_wilcoxon_BC_p']}\n"
            f"    Δ(C-A) mean = {r['mean_bal_acc_delta_CA']:+.3f}, "
            f"W = {r['paired_wilcoxon_CA_W']}, "
            f"p = {r['paired_wilcoxon_CA_p']}"
        )
    # Per-class breakdown (pass-2-fix M6): narrate the asymmetry
    # between MI gain and REST loss on the B-A and B-C contrasts.
    txt_lines.append("")
    txt_lines.append("=== Per-class B-A and B-C cohort means ===")
    for col, label in [
        ("mi_acc_off",  "MI acc, Arm A"),
        ("mi_acc_on",   "MI acc, Arm B"),
        ("mi_acc_ra",   "MI acc, Arm C"),
        ("rest_acc_off","REST acc, Arm A"),
        ("rest_acc_on", "REST acc, Arm B"),
        ("rest_acc_ra", "REST acc, Arm C"),
    ]:
        txt_lines.append(
            f"  {label}: {df[col].mean():.3f} ± {df[col].std():.3f}"
        )
    (out_root / "csv" / "gr_ablation_lme_results.txt").write_text(
        "\n".join(txt_lines)
    )

    # ----- Cohort plots -----
    delta_lme_BA = lme_rows["bal_acc_delta_BA"]
    delta_lme_BC = lme_rows["bal_acc_delta_BC"]
    annotation_three_arm = (
        f"Δ(B-A) cohort Wilcoxon W={cohort_pw_BA['W']}, "
        f"p={cohort_pw_BA['p']:.3g} "
        f"({_bonferroni_verdict(cohort_pw_BA['p'])})\n"
        f"Δ(B-C) cohort Wilcoxon W={cohort_pw_BC['W']}, "
        f"p={cohort_pw_BC['p']:.3g} "
        f"({_bonferroni_verdict(cohort_pw_BC['p'])})\n"
        f"Δ(C-A) cohort Wilcoxon W={cohort_pw_CA['W']}, "
        f"p={cohort_pw_CA['p']:.3g} "
        f"({_bonferroni_verdict(cohort_pw_CA['p'])})\n"
        f"Bonferroni α' = {BONFERRONI_ALPHA:.4f}"
    )
    _plot_cohort_three_arm(
        df, out_root / "cohort" / "cohort_three_arm_bal_acc.png",
        lme_annotation=annotation_three_arm,
    )
    annotation_delta = (
        f"LME Δ(B-A) slope = {delta_lme_BA.get('slope', float('nan')):+.4f}/s, "
        f"p = {delta_lme_BA.get('slope_p', float('nan')):.3g}\n"
        f"LME Δ(B-C) slope = {delta_lme_BC.get('slope', float('nan')):+.4f}/s, "
        f"p = {delta_lme_BC.get('slope_p', float('nan')):.3g}\n"
        + annotation_three_arm
    )
    _plot_cohort_paired_delta(
        df, out_root / "cohort" / "cohort_paired_delta_over_sessions.png",
        lme_annotation=annotation_delta,
    )
    _plot_cohort_box(
        df, out_root / "cohort" / "cohort_bal_acc_boxplot.png",
    )
    _plot_mcnemar_heatmap(
        df, out_root / "cohort" / "cohort_mcnemar_heatmap_BA.png",
        col="mcnemar_BA_p", title_suffix="Arm B (Kumar GR) vs Arm A (GR-off)",
    )
    _plot_mcnemar_heatmap(
        df, out_root / "cohort" / "cohort_mcnemar_heatmap_BC.png",
        col="mcnemar_BC_p", title_suffix="Arm B (Kumar GR) vs Arm C (Zanini RA)",
    )
    _plot_mcnemar_heatmap(
        df, out_root / "cohort" / "cohort_mcnemar_heatmap_CA.png",
        col="mcnemar_CA_p", title_suffix="Arm C (Zanini RA) vs Arm A (GR-off)",
    )

    # ----- Headline print -----
    print("\n=== Cohort summary (three-arm) ===")
    for label, col in [
        ("Arm A (GR-off)   ", "bal_acc_off"),
        ("Arm B (Kumar GR) ", "bal_acc_on"),
        ("Arm C (Zanini RA)", "bal_acc_ra"),
    ]:
        print(
            f"  {label}: {df[col].mean():.3f} ± {df[col].std():.3f}"
        )
    for label, pw in [
        ("Δ(B-A)", cohort_pw_BA),
        ("Δ(B-C)", cohort_pw_BC),
        ("Δ(C-A)", cohort_pw_CA),
    ]:
        print(
            f"  {label} paired Wilcoxon: W = {pw['W']}, "
            f"p = {pw['p']:.4g} "
            f"({_bonferroni_verdict(pw['p'])})"
        )


if __name__ == "__main__":
    main()
