"""
tune_xgb_hyperparams.py

Hyperparameter search for the XGBoost covariance feature decoder using the
configured CV mode (session-LOO or KFold).

CV mode is controlled by config.CV_MODE:
  "session_loo" — GroupKFold respecting session boundaries.
                  k = min(n_sessions, N_LOO_SPLITS).  Set N_LOO_SPLITS=100
                  to get true leave-one-session-out across all sessions.
  "kfold"       — Shuffled KFold(N_SPLITS).

The same CV split is applied uniformly to all candidates.  After all
candidates are evaluated, the winner is selected by the configured criterion
and its CV metrics are reported.

Criterion  : controlled by config.XGB_TUNE_CRITERION:
  "kl"  — minimize KL(empirical ‖ Beta(a, b)) averaged symmetrically
           across MI and REST classes, computed on pooled held-out
           P(correct class) scores.
  "auc" — maximize pooled ROC-AUC (MI vs REST) across held-out folds.

Search space (exhaustive grid):
  max_depth     : [3, 4, 5, 6]
  n_estimators  : [50, 100, 200, 300]
  learning_rate : [0.01, 0.03, 0.05, 0.10]
  Total         : 4 × 4 × 4 = 64 candidates

Fixed (not searched):
  shrinkage_param : SHRINKAGE_PARAM_XGB from config.py

Optional config overrides (not searched):
  XGB_SUBSAMPLE, XGB_COLSAMPLE_BYTREE,
  XGB_REG_ALPHA, XGB_REG_LAMBDA, XGB_MIN_CHILD_WEIGHT.
  If any of these are absent from config.py, XGBoost package defaults apply.
  XGB_LEARNING_RATE in config.py is ignored when learning_rate is searched.

Config keys consumed:
  CV_MODE, N_LOO_SPLITS, N_SPLITS,
  XGB_USE_COV_MU, XGB_USE_COV_BETA,
  XGB_TUNE_CRITERION,
  XGB_TUNE_BETA_ALPHA, XGB_TUNE_BETA_BETA, XGB_TUNE_KL_BINS,
  RECENTERING, LEDOITWOLF.


Requires LEDOITWOLF=0.

Output:
  - Per-candidate: KL score and a best-so-far marker.
  - Winner: KL breakdown, AUC, fixed-threshold sweep.
  - Recommended config values for the three searched parameters.

Usage:
  python tune_xgb_hyperparams.py

After reviewing the recommendations, copy the suggested values into config.py
and retrain with generate_xgboost_cov_features.py.
"""

import itertools
import os

import numpy as np

os.environ["NUMBA_DISABLE_CACHING"] = "1"
os.environ["MNE_USE_NUMBA"] = "false"

import mne
from pyriemann.tangentspace import TangentSpace, tangent_space
from pyriemann.estimation import Shrinkage
from pyriemann.preprocessing import Whitening
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import config
import Generate_Riemannian_adaptive as base
from Utils.stream_utils import load_xdf
from Utils.xgb_feature_pipeline import segment_and_extract_cov_erd


# ── Optional config overrides ──────────────────────────────────────────────────
# These params are not searched. If present in config.py they override XGBoost
# package defaults; if absent, XGBoost uses its own defaults automatically.

_OPTIONAL_CFG_KEYS = [
    ("learning_rate",    "XGB_LEARNING_RATE",    float),
    ("subsample",        "XGB_SUBSAMPLE",         float),
    ("colsample_bytree", "XGB_COLSAMPLE_BYTREE",  float),
    ("reg_alpha",        "XGB_REG_ALPHA",          float),
    ("reg_lambda",       "XGB_REG_LAMBDA",         float),
    ("min_child_weight", "XGB_MIN_CHILD_WEIGHT",   float),
]


def _config_overrides() -> dict:
    """Return XGB params explicitly set in config.py; omitted params use XGBoost defaults."""
    result = {}
    for param, cfg_key, cast in _OPTIONAL_CFG_KEYS:
        val = getattr(config, cfg_key, None)
        if val is not None:
            result[param] = cast(val)
    return result


_FIXED_XGB = dict(objective="binary:logistic", eval_metric="logloss", random_state=42)

# ── Exhaustive search grid ─────────────────────────────────────────────────────

_SEARCH_GRID = {
    "max_depth":     [3, 4, 5, 6],
    "n_estimators":  [50, 100, 200, 300],
    "learning_rate": [0.01, 0.03, 0.05, 0.10],
}

_XGB_CLASSIFIER_KEYS = frozenset({"max_depth", "n_estimators", "learning_rate"})


def _build_candidates() -> list[dict]:
    """Enumerate all combinations in _SEARCH_GRID as a flat list of dicts."""
    keys = list(_SEARCH_GRID.keys())
    combos = list(itertools.product(*[_SEARCH_GRID[k] for k in keys]))
    return [{k: v for k, v in zip(keys, combo)} for combo in combos]


# ── Covariance preprocessing ───────────────────────────────────────────────────

def _trace_norm_covs(segments: np.ndarray) -> np.ndarray:
    """Compute trace-normalized covariance matrices.  segments: (n_trials, C, T)."""
    return np.array([(s @ s.T) / np.trace(s @ s.T) for s in segments])


def _shrink_and_whiten_per_session(
    raw_covs_per_session: list, shrinkage_param: float
) -> np.ndarray:
    """
    Apply Shrinkage(lambda) then per-session Riemannian whitening to a list of
    per-session trace-normalized covariance arrays.

    Mirrors compute_processed_covariances with config.RECENTERING=1: whitening
    is fitted independently per session so each session's distribution is
    centred before cross-session concatenation.
    """
    result = []
    for raw_covs in raw_covs_per_session:
        shrunken = Shrinkage(shrinkage=shrinkage_param).fit_transform(raw_covs)
        if config.RECENTERING:
            whitened = Whitening(metric="riemann").fit_transform(shrunken)
            result.append(whitened)
        else:
            result.append(shrunken)
    return np.concatenate(result, axis=0)


# ── Tangent space helpers ──────────────────────────────────────────────────────

def _fit_tangent_ref(cov_matrices: np.ndarray) -> np.ndarray:
    ts = TangentSpace(metric="riemann")
    ts.fit(cov_matrices)
    return ts.reference_


def _cov_tangent_features(cov_matrices: np.ndarray, tangent_ref: np.ndarray) -> np.ndarray:
    return tangent_space(cov_matrices, tangent_ref, metric="riemann")


def _project_fold(
    processed_mu, processed_beta, use_mu, use_beta, tr_idx, te_idx
):
    """
    Project covariances to tangent space fitting the reference on tr_idx only.
    Returns (X_tr, X_te) with feature blocks horizontally stacked across bands.
    """
    blocks_tr, blocks_te = [], []
    if use_mu and processed_mu is not None:
        ref = _fit_tangent_ref(processed_mu[tr_idx])
        blocks_tr.append(_cov_tangent_features(processed_mu[tr_idx], ref))
        blocks_te.append(_cov_tangent_features(processed_mu[te_idx], ref))
    if use_beta and processed_beta is not None:
        ref = _fit_tangent_ref(processed_beta[tr_idx])
        blocks_tr.append(_cov_tangent_features(processed_beta[tr_idx], ref))
        blocks_te.append(_cov_tangent_features(processed_beta[te_idx], ref))
    return np.hstack(blocks_tr), np.hstack(blocks_te)


# ── Single candidate evaluation ────────────────────────────────────────────────

def _evaluate_candidate(
    cand: dict,
    fold_projections: dict,
    y_bin: np.ndarray,
    splits: list,
    XGBClassifier,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate one candidate across all pre-computed CV splits.

    fold_projections is keyed by (shrinkage_param, fold_idx) and was built
    before the search loop so that all candidates share the same projection
    (shrinkage is fixed, not searched).

    Config overrides are applied first; candidate values take precedence so
    that searched parameters cannot be stomped by config.

    Returns (pooled_scores, pooled_true_bin) — all held-out P(MI) values and
    corresponding binary labels, concatenated across folds in split order.
    """
    xgb_params = _config_overrides()
    xgb_params.update({k: cand[k] for k in _XGB_CLASSIFIER_KEYS})

    shrink = shrinkage
    all_scores   = []
    all_true_bin = []

    for fi, (tr, te) in enumerate(splits):
        X_tr, X_te = fold_projections[(shrink, fi)]
        scaler = StandardScaler()
        Xts = scaler.fit_transform(X_tr)
        Xvs = scaler.transform(X_te)
        clf = XGBClassifier(**xgb_params, **_FIXED_XGB)
        clf.fit(Xts, y_bin[tr], verbose=False)
        all_scores.extend(clf.predict_proba(Xvs)[:, 1])
        all_true_bin.extend(y_bin[te])

    return np.asarray(all_scores), np.asarray(all_true_bin)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    mne.set_log_level("WARNING")

    if config.LEDOITWOLF:
        raise RuntimeError(
            "Shrinkage grid search requires LEDOITWOLF=0 in config.py. "
            "Set LEDOITWOLF=0 and a baseline SHRINKAGE_PARAM_XGB, then re-run."
        )

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost required. Install with `pip install xgboost`.") from exc

    cv_mode      = str(getattr(config, "CV_MODE",        "session_loo")).lower()
    n_loo_splits = int(getattr(config, "N_LOO_SPLITS",   5))
    n_splits     = int(getattr(config, "N_SPLITS",       5))
    use_mu       = bool(getattr(config, "XGB_USE_COV_MU",  1))
    use_beta     = bool(getattr(config, "XGB_USE_COV_BETA", 0))

    criterion = str(getattr(config, "XGB_TUNE_CRITERION",   "kl")).lower()
    beta_a    = float(getattr(config, "XGB_TUNE_BETA_ALPHA", 18))
    beta_b    = float(getattr(config, "XGB_TUNE_BETA_BETA",   5))
    n_bins    = int(getattr(config,   "XGB_TUNE_KL_BINS",    15))

    candidates   = _build_candidates()
    n_candidates = len(candidates)

    print("=" * 64)
    print("XGBoost Hyperparameter Search")
    print("=" * 64)
    print(f"  Search: exhaustive grid  |  candidates: {n_candidates}")
    shrink_fixed = float(getattr(config, "SHRINKAGE_PARAM_XGB", 0.02))

    print(f"  Grid: max_depth={_SEARCH_GRID['max_depth']}")
    print(f"        n_estimators={_SEARCH_GRID['n_estimators']}")
    print(f"        learning_rate={_SEARCH_GRID['learning_rate']}")
    print(f"  Fixed: shrinkage_param={shrink_fixed}")
    overrides = {k: v for k, v in _config_overrides().items() if k != "learning_rate"}
    if overrides:
        print(f"  Config overrides: {overrides}")
    else:
        print(f"  Config overrides: none — XGBoost package defaults apply")
    if criterion == "auc":
        print(f"  Criterion: maximize ROC-AUC (pooled MI vs REST)")
    else:
        print(f"  Criterion: KL(empirical ‖ Beta({beta_a:.2g},{beta_b:.2g})) "
              f"[{n_bins} bins, avg MI+REST]")
    print(f"  CV mode: {cv_mode}  |  feature sets: mu={use_mu}  beta={use_beta}\n")

    # ── Data loading ───────────────────────────────────────────────────────────
    eeg_dir = os.path.join(
        config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "training_data"
    )
    xdf_files = sorted([
        os.path.join(eeg_dir, f) for f in os.listdir(eeg_dir)
        if f.endswith(".xdf") and "OBS" not in f
    ])
    if not xdf_files:
        raise FileNotFoundError(f"No XDF files in: {eeg_dir}")
    print(f"Found {len(xdf_files)} session(s):")

    all_labels        = []
    session_ids       = []
    raw_covs_mu_sess  = []
    raw_covs_beta_sess = [] if use_beta else None

    for sess_idx, xdf_path in enumerate(xdf_files):
        print(f"  [{sess_idx}] {os.path.basename(xdf_path)}")
        eeg_stream, marker_stream = load_xdf(xdf_path, report=False)

        out = segment_and_extract_cov_erd(
            eeg_stream, marker_stream,
            compute_erd=False,
            apply_csd=False,
            return_beta_segments=use_beta,
        )
        if use_beta:
            segments, labels, _, beta_segments, _ = out
        else:
            segments, labels, _, _ = out
            beta_segments = None

        raw_covs_mu_sess.append(_trace_norm_covs(segments))
        if use_beta and beta_segments is not None:
            raw_covs_beta_sess.append(_trace_norm_covs(beta_segments))

        all_labels.append(labels)
        session_ids.append(np.full(len(labels), sess_idx, dtype=int))

    y      = np.concatenate(all_labels)
    groups = np.concatenate(session_ids)

    classes = np.sort(np.unique(y))
    if len(classes) != 2:
        raise ValueError(f"Expected 2 classes, got {len(classes)}.")
    mi_label = classes[1]
    y_bin = (y == mi_label).astype(int)

    # ── Precompute processed covs for fixed shrinkage ─────────────────────────
    shrinkage_values = [shrink_fixed]
    print(f"\nPrecomputing processed covariances (shrinkage={shrink_fixed}) ...")

    processed_mu   = {}
    processed_beta = {}
    for shrink in shrinkage_values:
        processed_mu[shrink] = _shrink_and_whiten_per_session(raw_covs_mu_sess, shrink)
        if use_beta:
            processed_beta[shrink] = _shrink_and_whiten_per_session(
                raw_covs_beta_sess, shrink
            )

    # ── Build CV splits (same for all candidates) ─────────────────────────────
    ref_arr = processed_mu[shrinkage_values[0]]

    if cv_mode == "session_loo":
        n_sessions = len(xdf_files)
        k = min(n_sessions, n_loo_splits)
        splitter = GroupKFold(n_splits=k)
        splits = list(splitter.split(ref_arr, groups=groups))
        print(f"\nCV: session-LOO  |  folds: {k}  ({n_sessions} sessions)")
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(splitter.split(ref_arr))
        print(f"\nCV: KFold  |  folds: {n_splits}")

    # ── Precompute tangent projections for all (shrinkage, fold) combinations ──
    # Candidates sharing the same shrinkage_param reuse the same projection,
    # reducing tangent fits from n_candidates × n_folds to n_shrinkage × n_folds.
    n_unique_proj = len(shrinkage_values) * len(splits)
    print(f"Precomputing {n_unique_proj} tangent projections "
          f"({len(shrinkage_values)} shrinkage × {len(splits)} folds) ...\n")

    fold_projections = {}  # (shrink, fold_idx) -> (X_tr, X_te)
    for shrink in shrinkage_values:
        cov_mu_k   = processed_mu[shrink]
        cov_beta_k = processed_beta.get(shrink) if use_beta else None
        for fi, (tr, te) in enumerate(splits):
            fold_projections[(shrink, fi)] = _project_fold(
                cov_mu_k, cov_beta_k, use_mu, use_beta, tr, te
            )

    # ── Exhaustive search ──────────────────────────────────────────────────────
    print(f"Evaluating {n_candidates} candidates ...\n")

    # best_score tracks the criterion value: minimised for "kl", maximised for "auc".
    best_score  = -np.inf if criterion == "auc" else np.inf
    best_cand   = None
    best_scores = None
    best_true   = None

    for ci, cand in enumerate(candidates):
        scores, true_bin = _evaluate_candidate(
            cand, fold_projections, y_bin, splits, XGBClassifier, shrink_fixed
        )

        if criterion == "auc":
            crit_val = (
                float(roc_auc_score(true_bin, scores))
                if len(np.unique(true_bin)) == 2 else float("nan")
            )
            is_best = crit_val > best_score
        else:
            kl_result = base._compute_kl_breakdown(scores, true_bin, beta_a, beta_b, n_bins)
            crit_val  = kl_result["kl_combined"]
            is_best   = crit_val < best_score

        if is_best:
            best_score  = crit_val
            best_cand   = cand
            best_scores = scores
            best_true   = true_bin

        label  = "AUC" if criterion == "auc" else "KL"
        marker = " *" if is_best else ""
        print(
            f"[{ci+1:>3}/{n_candidates}]  "
            f"depth={cand['max_depth']}  n_est={cand['n_estimators']}  "
            f"lr={cand['learning_rate']:.2f}  "
            f"{label}={crit_val:.4f}{marker}"
        )

    # ── Report on winner ───────────────────────────────────────────────────────
    agg_auc = (
        best_score if criterion == "auc"
        else float(roc_auc_score(best_true, best_scores))
        if len(np.unique(best_true)) == 2 else float("nan")
    )

    print("\n" + "=" * 64)
    print("WINNER")
    print("=" * 64)
    print(f"  max_depth       = {best_cand['max_depth']}")
    print(f"  n_estimators    = {best_cand['n_estimators']}")
    print(f"  learning_rate   = {best_cand['learning_rate']}")
    print(f"  shrinkage_param = {shrink_fixed}  (fixed)")
    print(f"  AUC (pooled)    = {agg_auc:.4f}")

    base._print_kl_report(best_scores, best_true, beta_a, beta_b, n_bins)
    base._print_fixed_threshold_sweep(best_scores, best_true)

    print("\n── Paste into config.py ──────────────────────────────────────────")
    print(f"XGB_MAX_DEPTH        = {best_cand['max_depth']}")
    print(f"XGB_N_ESTIMATORS     = {best_cand['n_estimators']}")
    print(f"XGB_LEARNING_RATE    = {best_cand['learning_rate']}")
    print(f"SHRINKAGE_PARAM_XGB  = {shrink_fixed}  # fixed during this search")
    print("──────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
