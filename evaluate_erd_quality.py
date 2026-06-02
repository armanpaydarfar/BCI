"""Rule-based quality scorer for clinical ERD timecourse figures.

This is an analysis-only (Tier 3) companion to
`Analyze_clinical_erd_refined.py`. It reads the per-trial `.npz` side-cars
that the ERD pass already emitted (one per subject/session, flat keys, no
pickle) and produces a per-(subject, session, cluster) scorecard with eight
sub-scores (D1..D8), a weighted aggregate (S), and PASS/FAIL gates.

Why a separate scorer rather than annotating inside the TFR job: the ERD
pass is a heavy MNE/TFR pipeline (~1 GB per session). Figure-quality triage
needs to be re-runnable cheaply and tunable (weights, thresholds) without
re-decoding EEG. The npz side-car is the contract between the two; this file
never imports MNE and never re-runs a TFR.

The sub-score helpers (`lin01`, `lin01_inv`) and the per-cluster dimension
functions are kept dependency-light so the figure-annotation code can import
them later to print sub-scores on the panels.

CLI:
    python evaluate_erd_quality.py [--npz-dir PATH] [--out-dir PATH]
                                   [--variant TAG]

Reference: cluster definitions and nominal sizes are imported from
`exploration/clinical_analysis/_helpers.py:105-109`; output root from
`_helpers.clin_pictures_root` (_helpers.py:170); session index from
`_helpers.session_idx_from_label` (_helpers.py:185).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# Ensure the repo root is importable so `exploration.clinical_analysis`
# resolves regardless of the caller's working directory. The package itself
# also inserts the repo root, but we add it here defensively before import.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exploration.clinical_analysis._helpers import (  # noqa: E402
    BILATERAL_MOTOR_CLUSTER,
    CONTRA_MOTOR_CLUSTER,
    IPSI_MOTOR_CLUSTER,
    clin_pictures_root,
    session_idx_from_label,
)

# ----------------------------------------------------------------------
# Weights (module-level so calibration can override before scoring)
# ----------------------------------------------------------------------

# Per-dimension weights for the aggregate score S. NaN dimensions are
# dropped and the remaining weights renormalised, so these need not sum to
# 1 after a dimension goes missing (e.g. REST absent → D3, D4 dropped).
WEIGHTS: dict[str, float] = {
    "D1": 0.18,  # MI strength
    "D2": 0.16,  # sustained desync
    "D3": 0.14,  # MI-vs-REST contrast
    "D4": 0.14,  # REST specificity
    "D6": 0.12,  # artifact cleanliness
    "D8": 0.10,  # band-to-signal
    "D7": 0.08,  # retention
    "D5": 0.08,  # lateralization
}

# Nominal cluster sizes (full electrode count before any auto-drop) used by
# D7's channel-retention term. From _helpers.py:105-109.
_NOMINAL_CLUSTER_SIZE: dict[str, int] = {
    "bilat": len(BILATERAL_MOTOR_CLUSTER),
    "contra": len(CONTRA_MOTOR_CLUSTER),
    "ipsi": len(IPSI_MOTOR_CLUSTER),
}

# Per-subject expected lateralization class (rubric §2). Drives D5. Subjects
# absent from this table fall back to "bilateral" (the most forgiving class).
_LAT_CLASS_EXPECTED: dict[str, str] = {
    "CLIN_SUBJ_002": "bilateral",
    "CLIN_SUBJ_003": "classical-contra",
    "CLIN_SUBJ_004": "ipsi-lean",
    "CLIN_SUBJ_005": "classical-contra",
    "CLIN_SUBJ_006": "bilateral",
    "CLIN_SUBJ_007": "bilateral",
    "CLIN_SUBJ_008": "ipsi-lean",
}

_CLUSTERS = ("bilat", "contra", "ipsi")

# Gate G1: a kept trial whose post-cue |ERD%| peak exceeds G1_OUTLIER_PCT is a
# retained outlier (rubric §4 G1 / rev01 §8.3 "±200% invalidates"). G1 trips
# only when the FRACTION of such trials exceeds G1_OUTLIER_FRAC — a single
# residual outlier in an otherwise-clean session (e.g. subj3 S005: 1/124) must
# not invalidate a strong ERD, while a pervasively-noisy session (subj4: many
# trials >200%) still fails. Exposed so the figure-annotation layer flags the
# same sessions the gate does.
G1_OUTLIER_PCT = 200.0
G1_OUTLIER_FRAC = 0.05

# REST band-to-signal penalty (D8, REST half). A good REST line sits near zero,
# so the MI-style band/excursion ratio is undefined; instead penalize the
# ABSOLUTE median SE-band width (2*SE) over the window — wide REST bands are the
# "balloon" look. Clean REST bands run <~50%; >~250% is buried. Soft penalty
# (folded into D8), never a gate, so every session stays represented.
REST_BAND_LO = 50.0
REST_BAND_HI = 250.0

# Gate G2 over-rejection threshold: fail a session if > this fraction of its
# trials were dropped across the full chain (µV + channel + trial-z + abs-cap).
# 0.50 (per Arman): dropping up to half a run is an acceptable minority when the
# survivors are clean; only beyond half is the session cleaned-into-a-corner.
# Must match Analyze_clinical_erd_refined.TRIAL_REJECT_MAX_FRAC.
G2_OVERREJECT_FRAC = 0.50


# ----------------------------------------------------------------------
# Sub-score ramp helpers (imported by figure-annotation code later)
# ----------------------------------------------------------------------

def lin01(x: float, lo: float, hi: float) -> float:
    """Linear ramp mapping lo->0 and hi->1, clamped to [0, 1].

    Works for both lo<hi (rising ramp) and lo>hi (falling ramp): the value
    is computed as (x - lo) / (hi - lo) and clamped, so the direction is
    determined by the lo/hi ordering the caller passes. Returns 0.0 if
    lo == hi (degenerate range) to avoid a divide-by-zero.
    """
    if hi == lo:
        return 0.0
    t = (float(x) - lo) / (hi - lo)
    return float(min(1.0, max(0.0, t)))


def lin01_inv(x: float, lo: float, hi: float) -> float:
    """Inverse ramp mapping lo->1 and hi->0, clamped to [0, 1].

    Equivalent to 1 - lin01(x, lo, hi). Used where larger input means worse
    quality (e.g. wider error band, larger artifact peak).
    """
    return 1.0 - lin01(x, lo, hi)


# ----------------------------------------------------------------------
# Window resolution (derived per-npz from the times array)
# ----------------------------------------------------------------------

def _scalar_mask(times: np.ndarray) -> np.ndarray:
    """Boolean mask for the magnitude window [1, t_end], t_end = times.max().

    Auto-extends to whatever the session recorded (4 or 5 s post-cue) — the
    window is never hardcoded so 4 s and 5 s sessions score consistently.
    """
    return times >= 1.0  # t_end is times.max(), so [1, max] == (times >= 1)


def _onset_mask(times: np.ndarray) -> np.ndarray:
    """Boolean mask for the timing/onset window [0, 1] s."""
    return (times >= 0.0) & (times <= 1.0)


def _postcue_mask(times: np.ndarray) -> np.ndarray:
    """Boolean mask for post-cue samples (times >= 0), used by gates/D6."""
    return times >= 0.0


def _longest_run(boolean: np.ndarray) -> int:
    """Longest contiguous run of True in a 1-D boolean array."""
    best = cur = 0
    for v in boolean:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return best


def _mann_whitney_auc(a: np.ndarray, b: np.ndarray) -> float:
    """Probability P(b > a) via the Mann-Whitney U / rank-biserial AUC.

    Returns AUC = U_b / (n_a * n_b), the fraction of (a, b) pairs where
    b > a (ties counted as 0.5). 0.5 = no separation, 1.0 = every REST
    above every MI. Computed by rank counting on the pooled sample so no
    scipy dependency is required. Returns nan if either group is empty.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = a.size, b.size
    if na == 0 or nb == 0:
        return float("nan")
    pooled = np.concatenate([a, b])
    ranks = _rankdata_avg(pooled)
    rank_b = ranks[na:]
    u_b = rank_b.sum() - nb * (nb + 1) / 2.0
    return float(u_b / (na * nb))


def _rankdata_avg(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties share the mean of their rank span.

    Local reimplementation of scipy.stats.rankdata(method='average') so the
    Mann-Whitney AUC needs no scipy.
    """
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank for the tie block
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


# ----------------------------------------------------------------------
# npz loading
# ----------------------------------------------------------------------

class _Session:
    """Parsed view of one (subject, session) npz side-car.

    Holds only the arrays the scorer needs and exposes per-cluster getters
    that return None when a (cluster, marker) key was absent from `keys`
    (i.e. that condition had no surviving data). No allow_pickle is used.
    """

    def __init__(self, path: Path):
        z = np.load(path)  # no allow_pickle: schema is all flat arrays
        self.path = path
        self.subject = str(z["subject"])
        self.session = str(z["session"])
        self.n_attempted = int(z["n_attempted"])
        self.n_kept = int(z["n_kept"])
        self.n_after_reject = int(z["n_after_reject"])
        dc = str(z["dropped_channels"])
        self.dropped_channels = [c for c in dc.split(",") if c]
        present = str(z["keys"])
        self._present = set(k for k in present.split(",") if k)
        self._ptp: dict[str, np.ndarray] = {}
        self._times: dict[str, np.ndarray] = {}
        self._channels: dict[str, list[str]] = {}
        for key in self._present:
            self._ptp[key] = np.asarray(z[f"{key}__ptp"], dtype=np.float64)
            self._times[key] = np.asarray(z[f"{key}__times"], dtype=np.float64)
            chans = str(z[f"{key}__channels"])
            self._channels[key] = [c for c in chans.split(",") if c]

    def ptp(self, key: str) -> np.ndarray | None:
        """Per-trial ERD% matrix (n_trials, n_time) for a key, or None."""
        return self._ptp.get(key)

    def times(self, key: str) -> np.ndarray | None:
        return self._times.get(key)

    def channels(self, key: str) -> list[str] | None:
        return self._channels.get(key)


# ----------------------------------------------------------------------
# Per-cluster scalar substrate
# ----------------------------------------------------------------------

def _scalar_of_trace(trace: np.ndarray, smask: np.ndarray) -> float:
    """Median over the SCALAR window of a (median) trace."""
    return float(np.median(trace[smask]))


def _per_trial_scalars(ptp: np.ndarray, smask: np.ndarray) -> np.ndarray:
    """Per-trial scalar = mean over SCALAR-time of each trial's ptp row."""
    return ptp[:, smask].mean(axis=1)


# ----------------------------------------------------------------------
# Dimensions D1..D8 (each float in [0,1] or nan)
# ----------------------------------------------------------------------

def _d1_mi_strength(med_mi: np.ndarray, smask: np.ndarray) -> float:
    """D1: depth of MI desync. m = median over SCALAR of med_mi; forced 0 if
    m >= 0 (no desync at all). lin01(-m, 0, 40): -20% -> 0.50 (good), -30% ->
    0.75, -40%+ -> 1.0 (excellent). Calibrated to Arman's intuition that -20%
    is a good result and -30/-40% is progressively better (the cohort's real
    desync spans ~-12 to -43%, so the earlier (20,35) ramp wrongly zeroed most
    good sessions)."""
    m = _scalar_of_trace(med_mi, smask)
    if m >= 0:
        return 0.0
    return lin01(-m, 0.0, 40.0)


def _d2_sustained(med_mi: np.ndarray, smask: np.ndarray) -> float:
    """D2: sustained desync. frac = fraction of SCALAR samples <= -15%;
    run = longest contiguous such fraction. D2 = min of the two ramps."""
    seg = med_mi[smask] <= -15.0
    n = seg.size
    if n == 0:
        return float("nan")
    frac = float(seg.mean())
    run = _longest_run(seg) / n
    return min(lin01(frac, 0.4, 0.9), lin01(run, 0.4, 0.9))


def _best_of_strength(med_a: np.ndarray, med_b: np.ndarray | None,
                      smask: np.ndarray) -> tuple[float, float]:
    """D1/D2 from whichever cluster desyncs more (best-of bilateral/contra).

    Returns (D1, D2) computed on the median trace whose SCALAR median is more
    negative between med_a and med_b. This is the primary ERD-strength metric:
    bilateral is the default subject-agnostic substrate, but for a classical-
    contra subject (e.g. CLIN_SUBJ_003) the contralateral cluster carries the
    real desync that bilateral averaging dilutes, so the stronger of the two
    drives D1/D2. D1 and D2 come from the SAME (driver) cluster so the strength
    and its sustain are consistent. med_b is None when contra is unavailable.
    """
    if med_b is None:
        return _d1_mi_strength(med_a, smask), _d2_sustained(med_a, smask)
    if _scalar_of_trace(med_b, smask) < _scalar_of_trace(med_a, smask):
        return _d1_mi_strength(med_b, smask), _d2_sustained(med_b, smask)
    return _d1_mi_strength(med_a, smask), _d2_sustained(med_a, smask)


def _d3_contrast(mi_scalar: float, rest_scalar: float,
                 mi_trial: np.ndarray, rest_trial: np.ndarray) -> float:
    """D3: MI-vs-REST contrast. Directional — if MI is not below REST
    (mi_scalar >= rest_scalar) the contrast has the wrong sign -> 0.
    Otherwise combine a separation ramp and a rank-biserial AUC ramp."""
    if mi_scalar >= rest_scalar:
        return 0.0
    sep = rest_scalar - mi_scalar
    d3_sep = lin01(sep, 15.0, 50.0)
    auc = _mann_whitney_auc(mi_trial, rest_trial)  # P(REST > MI)
    if np.isnan(auc):
        return d3_sep  # no per-trial separation info; fall back to sep ramp
    d3_auc = lin01(2.0 * abs(auc - 0.5), 0.3, 1.0)
    return 0.5 * d3_sep + 0.5 * d3_auc


def _d4_rest_specificity(med_rest: np.ndarray, smask: np.ndarray
                         ) -> tuple[float, bool]:
    """D4: REST specificity, asymmetric. r = median over SCALAR of med_rest.

    r >= 0 (ERS/flat): `lin01_inv(r, 0, 300)` — near-zero REST = 1.0, and the
    score falls off gently with the ERS spike (100% -> 0.67, 200% -> 0.33,
    300%+ -> 0). This is a SOFT penalty, not a gate: large positive REST ERS is
    explainable (breathing/eyes-closed alpha) and not treated as a failure, but
    a lower ERS spike scores better so the preprocessing sweep has a gradient to
    optimize toward filters that reduce ballooning (per Arman, 2026-06-02). The
    earlier (80, 300) ramp left 0–80% flat at 1.0, giving the sweep no signal to
    prefer near-zero REST over moderate ERS.

    r < 0 (desync over motor at rest = the real failure): lin01(r, -25, -5) —
    -5% -> ~0.8, <= -25% -> 0. Also returns eyes-closed flag (r > 300)."""
    r = _scalar_of_trace(med_rest, smask)
    flag = bool(r > 300.0)
    if r >= 0:
        return lin01_inv(r, 0.0, 300.0), flag
    return lin01(r, -25.0, -5.0), flag


def _d5_lateralization(lat: float, lat_class: str) -> float:
    """D5: subject-level lateralization match. lat = contra_scalar -
    ipsi_scalar (negative = contra stronger desync).

    classical-contra: expect lat < 0; graded by |lat| (deeper contra lead =
    higher), 0 if lat >= 0.
    ipsi-lean: expect lat >= 0; graded by |lat|, 0 if lat < 0.
    bilateral: expect |lat| small; high when |lat| ~ 0, falling off with
    |lat|.
    Returns nan if lat is nan (a cluster scalar was missing)."""
    if np.isnan(lat):
        return float("nan")
    if lat_class == "classical-contra":
        return lin01(-lat, 5.0, 20.0) if lat < 0 else 0.0
    if lat_class == "ipsi-lean":
        return lin01(lat, 5.0, 20.0) if lat >= 0 else 0.0
    # bilateral (and any unknown class): symmetric — small |lat| is good.
    return lin01_inv(abs(lat), 5.0, 25.0)


def _d6_artifact(mi_trial: np.ndarray, ptp_mi: np.ndarray,
                 pmask: np.ndarray) -> float:
    """D6: artifact cleanliness. mad = MAD of per-trial MI scalars; peak =
    max over kept trials of max post-cue |ptp|. D6 = min of both inverse
    ramps (lower spread / lower peak = cleaner)."""
    med = np.median(mi_trial)
    mad = float(np.median(np.abs(mi_trial - med)))
    peak = float(np.max(np.abs(ptp_mi[:, pmask]))) if pmask.any() else 0.0
    return min(lin01_inv(mad, 10.0, 60.0), lin01_inv(peak, 150.0, 400.0))


def _d7_retention(n_kept: int, n_attempted: int,
                  n_channels: int, nominal: int) -> float:
    """D7: retention. ret = lin01(n_kept/n_attempted, 0.5, 0.9); chan =
    channels_used/nominal clamped to 1. D7 = min(ret, lin01(chan, 0.5, 1.0)).
    Returns nan if n_attempted == 0 (cannot form a fraction)."""
    if n_attempted <= 0:
        return float("nan")
    ret = lin01(n_kept / n_attempted, 0.5, 0.9)
    chan = min(1.0, n_channels / nominal) if nominal > 0 else 0.0
    return min(ret, lin01(chan, 0.5, 1.0))


def _d8_band_to_signal(ptp_mi: np.ndarray, med_mi: np.ndarray,
                       smask: np.ndarray) -> tuple[float, float]:
    """D8: band width vs signal excursion. band_width = 2*SE; bw = median
    over SCALAR of band_width; excursion = max over SCALAR of |med_mi|;
    ratio = bw/max(excursion, 1e-6). D8 = lin01_inv(ratio, 0.5, 2.0).
    Returns (D8, raw ratio). nan if fewer than 2 trials (SE undefined)."""
    n = ptp_mi.shape[0]
    if n < 2:
        return float("nan"), float("nan")
    se = np.std(ptp_mi, axis=0, ddof=1) / np.sqrt(n)
    band_width = 2.0 * se
    bw = float(np.median(band_width[smask]))
    excursion = float(np.max(np.abs(med_mi[smask])))
    ratio = bw / max(excursion, 1e-6)
    return lin01_inv(ratio, 0.5, 2.0), ratio


def _d8_rest_band(ptp_rest: np.ndarray, smask: np.ndarray
                  ) -> tuple[float, float]:
    """REST half of D8: absolute SE-band-width penalty for the REST panel.

    Unlike MI, a clean REST median sits near zero, so dividing the band by the
    REST excursion would explode (≈0 denominator). Instead penalize the
    absolute median band width (2*SE) over the window: wide REST bands are the
    'balloon' look Arman flags on CLIN_SUBJ_007/008. lin01_inv(bw, 50, 250):
    band <= 50% -> 1.0 (clean), >= 250% -> 0 (buried). Returns (score, raw
    band%); nan if fewer than 2 trials (SE undefined)."""
    n = ptp_rest.shape[0]
    if n < 2:
        return float("nan"), float("nan")
    se = np.std(ptp_rest, axis=0, ddof=1) / np.sqrt(n)
    bw = float(np.median((2.0 * se)[smask]))
    return lin01_inv(bw, REST_BAND_LO, REST_BAND_HI), bw


# ----------------------------------------------------------------------
# Weighted aggregate
# ----------------------------------------------------------------------

def _weighted_score(dims: dict[str, float]) -> float:
    """S = sum(w_i * D_i) over present (non-nan) dims, weights renormalised
    over the present set. Returns nan if every dimension is nan."""
    num = 0.0
    den = 0.0
    for name, w in WEIGHTS.items():
        d = dims.get(name, float("nan"))
        if d is None or np.isnan(d):
            continue
        num += w * d
        den += w
    return num / den if den > 0 else float("nan")


# ----------------------------------------------------------------------
# Per-session scoring (gates G1/G2/G4; G3 deferred to cohort pass)
# ----------------------------------------------------------------------

def score_npz(path: str | Path) -> list[dict]:
    """Score one session npz, returning one row dict per cluster (bilat,
    contra, ipsi). Applies gates G1 (retained outlier), G2 (over-rejection),
    and G4 (lost cluster). G3 (subject-level no-MI-signal) is left to
    `score_dir`'s cohort pass since it spans all of a subject's sessions.

    Rows are full scorecards even when a gate trips (eligible=False) or a
    dimension is nan — the figure-annotation layer still wants the numbers.
    """
    sess = _Session(Path(path))
    subject = sess.subject
    lat_class = _LAT_CLASS_EXPECTED.get(subject, "bilateral")

    # G2 is session-level: drop_frac over attempted events.
    if sess.n_attempted > 0:
        drop_frac = 1.0 - sess.n_after_reject / sess.n_attempted
    else:
        drop_frac = float("nan")
    g2_failed = (not np.isnan(drop_frac)) and drop_frac > G2_OVERREJECT_FRAC

    # Lateralization is a subject/session-level quantity shared by all rows:
    # contra MI scalar minus ipsi MI scalar over each cluster's own SCALAR.
    lat_observed = _cluster_mi_scalar(sess, "contra")
    ipsi_scalar = _cluster_mi_scalar(sess, "ipsi")
    if np.isnan(lat_observed) or np.isnan(ipsi_scalar):
        lat = float("nan")
    else:
        lat = lat_observed - ipsi_scalar
    d5 = _d5_lateralization(lat, lat_class)

    # Contra MI median trace feeds the bilat row's best-of D1/D2 (primary
    # strength). Computed once here so _score_cluster stays per-cluster.
    contra_ptp = sess.ptp("contra_mi")
    contra_med = (np.median(contra_ptp, axis=0)
                  if contra_ptp is not None and contra_ptp.shape[0] > 0
                  else None)

    rows: list[dict] = []
    for cluster in _CLUSTERS:
        alt = contra_med if cluster == "bilat" else None
        rows.append(_score_cluster(
            sess, cluster, lat_class, lat, d5, drop_frac, g2_failed,
            primary_alt_med=alt,
        ))
    return rows


def _cluster_mi_scalar(sess: _Session, cluster: str) -> float:
    """MI scalar (median over SCALAR of the MI median trace) for one cluster,
    or nan if that cluster's MI key is absent."""
    ptp = sess.ptp(f"{cluster}_mi")
    if ptp is None or ptp.shape[0] == 0:
        return float("nan")
    times = sess.times(f"{cluster}_mi")
    smask = _scalar_mask(times)
    med = np.median(ptp, axis=0)
    return _scalar_of_trace(med, smask)


def _score_cluster(sess: _Session, cluster: str, lat_class: str,
                   lat: float, d5: float, drop_frac: float,
                   g2_failed: bool, primary_alt_med: np.ndarray | None = None
                   ) -> dict:
    """Build one cluster's scorecard row (dims, S, gates G1/G2/G4).

    `primary_alt_med` (the contra MI median trace) is supplied only for the
    bilateral row so its D1/D2 use best-of(bilateral, contra) per the primary
    ERD-strength rule; other rows score their own cluster honestly.
    """
    mi_key = f"{cluster}_mi"
    rest_key = f"{cluster}_rest"
    ptp_mi = sess.ptp(mi_key)
    times_mi = sess.times(mi_key)
    channels = sess.channels(mi_key) or []
    nominal = _NOMINAL_CLUSTER_SIZE[cluster]

    gates: list[str] = []
    reasons: list[str] = []
    rest_eyesclosed_flag = False
    band_ratio = float("nan")
    rest_band_pct = float("nan")

    # Initialise all dims to nan; fill what the data supports.
    dims = {d: float("nan") for d in
            ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8")}
    dims["D5"] = d5  # subject-level value, shared across clusters

    if ptp_mi is not None and ptp_mi.shape[0] > 0:
        smask = _scalar_mask(times_mi)
        pmask = _postcue_mask(times_mi)
        med_mi = np.median(ptp_mi, axis=0)
        mi_trial = _per_trial_scalars(ptp_mi, smask)
        mi_scalar = _scalar_of_trace(med_mi, smask)

        # Primary D1/D2 = best-of(this cluster, contra) for the bilat row;
        # plain this-cluster otherwise (primary_alt_med is None).
        dims["D1"], dims["D2"] = _best_of_strength(
            med_mi, primary_alt_med, smask,
        )
        dims["D6"] = _d6_artifact(mi_trial, ptp_mi, pmask)
        dims["D8"], band_ratio = _d8_band_to_signal(ptp_mi, med_mi, smask)

        # D3 needs REST too.
        ptp_rest = sess.ptp(rest_key)
        if ptp_rest is not None and ptp_rest.shape[0] > 0:
            smask_r = _scalar_mask(sess.times(rest_key))
            med_rest = np.median(ptp_rest, axis=0)
            rest_trial = _per_trial_scalars(ptp_rest, smask_r)
            rest_scalar = _scalar_of_trace(med_rest, smask_r)
            dims["D3"] = _d3_contrast(
                mi_scalar, rest_scalar, mi_trial, rest_trial,
            )
            dims["D4"], rest_eyesclosed_flag = _d4_rest_specificity(
                med_rest, smask_r,
            )
            # Fold REST band quality into D8: a balloon in EITHER class lowers
            # it (min over present, non-nan values). Soft penalty, no gate.
            d8_rest, rest_band_pct = _d8_rest_band(ptp_rest, smask_r)
            d8_vals = [v for v in (dims["D8"], d8_rest)
                       if v is not None and not np.isnan(v)]
            if d8_vals:
                dims["D8"] = min(d8_vals)

        # G1: fraction of kept MI trials with post-cue |ptp| peak > threshold.
        if pmask.any():
            trial_peaks = np.max(np.abs(ptp_mi[:, pmask]), axis=1)
        else:
            trial_peaks = np.zeros(ptp_mi.shape[0])
        n_out = int((trial_peaks > G1_OUTLIER_PCT).sum())
        frac_out = n_out / ptp_mi.shape[0] if ptp_mi.shape[0] else 0.0
        if frac_out > G1_OUTLIER_FRAC:
            gates.append("G1")
            reasons.append(
                f"G1 outliers: {n_out}/{ptp_mi.shape[0]} trials "
                f">{G1_OUTLIER_PCT:.0f}% ({frac_out:.0%}); "
                f"max={trial_peaks.max():.0f}%"
            )

    # D7 uses session retention + this cluster's channel count.
    dims["D7"] = _d7_retention(
        sess.n_kept, sess.n_attempted, len(channels), nominal,
    )

    # G2: session-level over-rejection.
    if g2_failed:
        gates.append("G2")
        reasons.append(
            f"G2 over-rejection: drop_frac={drop_frac:.2f} > {G2_OVERREJECT_FRAC:g}"
        )

    # G4: lost cluster (fewer than 2 surviving channels).
    if len(channels) < 2:
        gates.append("G4")
        reasons.append(f"G4 lost cluster: channels_used={len(channels)} < 2")

    s = _weighted_score(dims)
    return {
        "subject": sess.subject,
        "session": sess.session,
        "session_idx": session_idx_from_label(sess.session),
        "cluster": cluster,
        "n_attempted": sess.n_attempted,
        "n_kept": sess.n_kept,
        "n_after_reject": sess.n_after_reject,
        "drop_frac": _round(drop_frac),
        "channels_used": channels,
        "D1": _round(dims["D1"]),
        "D2": _round(dims["D2"]),
        "D3": _round(dims["D3"]),
        "D4": _round(dims["D4"]),
        "D5": _round(dims["D5"]),
        "D6": _round(dims["D6"]),
        "D7": _round(dims["D7"]),
        "D8": _round(dims["D8"]),
        "S": _round(s),
        "eligible": len(gates) == 0,
        "gates_failed": gates,
        "gate_reasons": " | ".join(reasons),
        "rest_eyesclosed_flag": rest_eyesclosed_flag,
        "band_ratio": _round(band_ratio),
        "rest_band_pct": _round(rest_band_pct),
        "lat_observed": _round(lat),
        "lat_class_expected": lat_class,
    }


def _round(x) -> float:
    """Round a float to 4 dp for output; pass nan/None through as None for
    clean JSON (JSON has no NaN; None serialises and round-trips)."""
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, 4)


# ----------------------------------------------------------------------
# Cohort scoring (adds G3 across all of a subject's sessions)
# ----------------------------------------------------------------------

def score_dir(npz_dir: str | Path, variant: str = "") -> list[dict]:
    """Score every npz under `npz_dir`, then apply G3 (subject-level
    no-MI-signal) across each subject's sessions.

    G3 trips for a subject when the bilateral MI scalar is >= 0 (no desync)
    across ALL of that subject's sessions. Decision: when it trips we mark
    every row of that subject (all clusters, all sessions) as G3-failed —
    not just bilat rows — because a subject with no bilateral MI desync
    anywhere is a global recording/engagement failure that taints the whole
    subject's figures, so no row should be treated as eligible. This is
    documented here and in the report per the task's "your call, document
    it" latitude.
    """
    npz_dir = Path(npz_dir)
    pattern = f"*{variant}.npz" if variant else "*.npz"
    paths = sorted(npz_dir.glob(pattern))
    rows: list[dict] = []
    for p in paths:
        rows.extend(score_npz(p))

    # G3: per subject, did the bilateral MI scalar stay >= 0 everywhere?
    by_subject: dict[str, list[dict]] = {}
    for r in rows:
        by_subject.setdefault(r["subject"], []).append(r)

    for subject, subj_rows in by_subject.items():
        bilat_d1 = [r for r in subj_rows if r["cluster"] == "bilat"]
        # bilat MI scalar < 0 is captured by D1 > 0 (D1 forces 0 when m>=0).
        any_desync = any((r["D1"] or 0.0) > 0.0 for r in bilat_d1)
        # Only fire G3 if we actually observed bilat MI somewhere; if no
        # bilat MI data exists at all, G4 already covers the loss.
        observed_bilat = any(r["D1"] is not None for r in bilat_d1)
        if observed_bilat and not any_desync:
            for r in subj_rows:
                if "G3" not in r["gates_failed"]:
                    r["gates_failed"].append("G3")
                    extra = "G3 no-MI-signal: bilat MI scalar >= 0 in all sessions"
                    r["gate_reasons"] = (
                        f"{r['gate_reasons']} | {extra}"
                        if r["gate_reasons"] else extra
                    )
                r["eligible"] = False
    return rows


# ----------------------------------------------------------------------
# Output writers and cohort summary
# ----------------------------------------------------------------------

def _write_json(rows: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


_CSV_FIELDS = [
    "subject", "session", "session_idx", "cluster",
    "n_attempted", "n_kept", "n_after_reject", "drop_frac",
    "channels_used",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "S",
    "eligible", "gates_failed", "gate_reasons",
    "rest_eyesclosed_flag", "band_ratio", "rest_band_pct",
    "lat_observed", "lat_class_expected",
]


def _write_csv(rows: list[dict], path: Path) -> None:
    """Flatten list/dict fields for CSV: channels_used and gates_failed are
    ';'-joined; nan-as-None renders as an empty cell."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for r in rows:
            flat = dict(r)
            flat["channels_used"] = ";".join(r["channels_used"])
            flat["gates_failed"] = ";".join(r["gates_failed"])
            w.writerow({k: ("" if flat.get(k) is None else flat[k])
                        for k in _CSV_FIELDS})


def _print_summary(rows: list[dict]) -> None:
    """Print per-subject median S over eligible bilat rows and a tally of
    gate trips by type."""
    print("\n=== Cohort summary ===")
    by_subject: dict[str, list[float]] = {}
    for r in rows:
        if r["cluster"] == "bilat" and r["eligible"] and r["S"] is not None:
            by_subject.setdefault(r["subject"], []).append(r["S"])
    print("Per-subject median S over eligible bilat rows:")
    for subject in sorted(by_subject):
        vals = by_subject[subject]
        print(f"  {subject}: median S = {np.median(vals):.3f} "
              f"(n={len(vals)} sessions)")
    subjects_no_eligible = sorted(
        {r["subject"] for r in rows} - set(by_subject)
    )
    for subject in subjects_no_eligible:
        print(f"  {subject}: no eligible bilat rows")

    gate_counts: dict[str, int] = {}
    for r in rows:
        for g in r["gates_failed"]:
            gate_counts[g] = gate_counts.get(g, 0) + 1
    print("Gate trips by type (row-count):")
    if gate_counts:
        for g in sorted(gate_counts):
            print(f"  {g}: {gate_counts[g]}")
    else:
        print("  (none)")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz-dir", type=Path, default=None,
        help="Directory of per-trial npz side-cars "
             "(default: clin_pictures_root()/erd_refined/per_trial)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory for JSON/CSV (default: npz-dir parent)",
    )
    parser.add_argument(
        "--variant", type=str, default="",
        help="Variant tag; globs *<tag>.npz and suffixes the output names",
    )
    args = parser.parse_args(argv)

    npz_dir = args.npz_dir or (
        clin_pictures_root() / "erd_refined" / "per_trial"
    )
    if not npz_dir.exists():
        raise FileNotFoundError(f"npz dir does not exist: {npz_dir}")
    out_dir = args.out_dir or npz_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = score_dir(npz_dir, args.variant)
    if not rows:
        print(f"No npz matched in {npz_dir} "
              f"(pattern *{args.variant}.npz)" if args.variant
              else f"No npz matched in {npz_dir} (pattern *.npz)")
        return 1

    json_path = out_dir / f"erd_quality_scores{args.variant}.json"
    csv_path = out_dir / f"erd_quality_scores{args.variant}.csv"
    _write_json(rows, json_path)
    _write_csv(rows, csv_path)
    print(f"Wrote {len(rows)} rows:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    _print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
