"""
tune_xgb_hyperparams.py

Nested cross-validation hyperparameter search for the XGBoost covariance
feature decoder.

Outer loop : session-LOO GroupKFold (respects config.N_LOO_SPLITS).
Inner loop : KFold over the outer train split (config.XGB_TUNE_INNER_SPLITS).
Criterion  : ROC-AUC — threshold-free, window-level, comparable across models.

The tangent space reference is fitted on the outer train split and shared
across all inner candidates.  This makes absolute inner-AUC slightly optimistic
but does not bias candidate ranking — sufficient for hyperparameter selection.

Output:
  - Per outer fold: winning params and outer-fold AUC.
  - Aggregated: fixed-threshold sweep on pooled outer test scores.
  - Recommended config values: mode across outer folds for each param.

Usage:
  python tune_xgb_hyperparams.py

After reviewing the recommendations, copy the suggested XGB_* values into
config.py and retrain with generate_xgboost_cov_features.py.
"""

import os
from collections import Counter

import numpy as np

os.environ["NUMBA_DISABLE_CACHING"] = "1"
os.environ["MNE_USE_NUMBA"] = "false"

import mne
from pyriemann.tangentspace import TangentSpace, tangent_space
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import config
import Generate_Riemannian_adaptive as base
from Utils.stream_utils import load_xdf
from Utils.xgb_feature_pipeline import segment_and_extract_cov_erd


# ── Tangent space helpers (match generate_xgboost_cov_features.py exactly) ────

def _fit_tangent_ref(cov_matrices: np.ndarray) -> np.ndarray:
    ts = TangentSpace(metric="riemann")
    ts.fit(cov_matrices)
    return ts.reference_


def _cov_tangent_features(cov_matrices: np.ndarray, tangent_ref: np.ndarray) -> np.ndarray:
    return tangent_space(cov_matrices, tangent_ref, metric="riemann")


def _project(cov_mu, cov_beta, use_mu, use_beta, tr_idx, te_idx):
    """
    Project covariances to tangent space using a reference fitted on tr_idx only.
    Returns (X_tr, X_te) with features horizontally stacked across enabled bands.
    """
    blocks_tr, blocks_te = [], []
    if use_mu and cov_mu is not None:
        ref = _fit_tangent_ref(cov_mu[tr_idx])
        blocks_tr.append(_cov_tangent_features(cov_mu[tr_idx], ref))
        blocks_te.append(_cov_tangent_features(cov_mu[te_idx], ref))
    if use_beta and cov_beta is not None:
        ref = _fit_tangent_ref(cov_beta[tr_idx])
        blocks_tr.append(_cov_tangent_features(cov_beta[tr_idx], ref))
        blocks_te.append(_cov_tangent_features(cov_beta[te_idx], ref))
    return np.hstack(blocks_tr), np.hstack(blocks_te)


# ── Search space ──────────────────────────────────────────────────────────────

_PARAM_GRID = {
    "max_depth":        [2, 3, 4, 5],
    "min_child_weight": [1, 3, 5, 10],
    "reg_lambda":       [0.5, 1.0, 2.0, 5.0],
    "reg_alpha":        [0.0, 0.1, 0.5],
    "subsample":        [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "learning_rate":    [0.01, 0.03, 0.05, 0.1],
    "n_estimators":     [100, 200, 300, 500],
}

_FIXED_XGB = dict(objective="binary:logistic", eval_metric="logloss", random_state=42)


def _sample_params(rng: np.random.Generator) -> dict:
    return {k: rng.choice(v).item() for k, v in _PARAM_GRID.items()}


# ── Inner-fold AUC for one candidate ─────────────────────────────────────────

def _score_candidate(X_tr, y_tr_bin, params, n_inner_splits, XGBClassifier):
    """
    Mean ROC-AUC over n_inner_splits inner KFold on the outer-train features.
    Returns nan if any inner fold is single-class (degenerate split).
    """
    kf = KFold(n_splits=n_inner_splits, shuffle=True, random_state=0)
    aucs = []
    for itr, ival in kf.split(X_tr):
        if len(np.unique(y_tr_bin[ival])) < 2:
            continue
        scaler = StandardScaler()
        Xts = scaler.fit_transform(X_tr[itr])
        Xvs = scaler.transform(X_tr[ival])
        clf = XGBClassifier(**params, **_FIXED_XGB)
        clf.fit(Xts, y_tr_bin[itr], verbose=False)
        aucs.append(roc_auc_score(y_tr_bin[ival], clf.predict_proba(Xvs)[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    mne.set_log_level("WARNING")

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost required. Install with `pip install xgboost`.") from exc

    n_iter       = int(getattr(config, "XGB_TUNE_N_ITER",       30))
    n_inner      = int(getattr(config, "XGB_TUNE_INNER_SPLITS", 3))
    n_loo_splits = int(getattr(config, "N_LOO_SPLITS",          5))
    seed         = int(getattr(config, "XGB_TUNE_SEED",         42))
    use_mu       = bool(getattr(config, "XGB_USE_COV_MU",       1))
    use_beta     = bool(getattr(config, "XGB_USE_COV_BETA",     0))

    rng = np.random.default_rng(seed)

    print("=" * 64)
    print("XGBoost Hyperparameter Search")
    print("=" * 64)
    print(f"  n_iter={n_iter}  inner_splits={n_inner}  "
          f"n_loo_splits={n_loo_splits}  seed={seed}")
    print(f"  feature sets: mu={use_mu}  beta={use_beta}\n")

    # ── Data loading ──────────────────────────────────────────────────────────
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

    all_labels, cov_mu_all, cov_beta_all, session_ids = [], [], [], []

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

        if use_mu:
            cov_mu_all.append(
                base.compute_processed_covariances(segments, labels, model_type="xgb")
            )
        if use_beta:
            cov_beta_all.append(
                base.compute_processed_covariances(beta_segments, labels, model_type="xgb")
            )
        all_labels.append(labels)
        session_ids.append(np.full(len(labels), sess_idx, dtype=int))

    y      = np.concatenate(all_labels)
    groups = np.concatenate(session_ids)
    cov_mu  = np.concatenate(cov_mu_all,   axis=0) if use_mu   else None
    cov_beta = np.concatenate(cov_beta_all, axis=0) if use_beta else None

    classes = np.sort(np.unique(y))
    if len(classes) != 2:
        raise ValueError(f"Expected 2 classes, got {len(classes)}.")
    mi_label = classes[1]
    y_bin = (y == mi_label).astype(int)

    ref_arr = cov_mu if cov_mu is not None else cov_beta

    # ── Pre-sample candidates (same set across all outer folds) ───────────────
    candidates = [_sample_params(rng) for _ in range(n_iter)]

    # ── Outer LOO loop ────────────────────────────────────────────────────────
    n_sessions = len(xdf_files)
    k_outer    = min(n_sessions, n_loo_splits)
    outer_cv   = GroupKFold(n_splits=k_outer)

    all_outer_scores   = []
    all_outer_true_bin = []
    outer_aucs         = []
    best_params_log    = []

    print(f"\nOuter folds: {k_outer}  |  Inner folds: {n_inner}  |  Candidates: {n_iter}\n")

    for fold_idx, (tr_outer, te_outer) in enumerate(
        outer_cv.split(ref_arr, groups=groups), 1
    ):
        y_tr_bin = y_bin[tr_outer]
        y_te_bin = y_bin[te_outer]
        held_sess = np.unique(groups[te_outer]).tolist()

        print(f"── Outer fold {fold_idx}/{k_outer}  "
              f"(train={len(tr_outer)} windows, test={len(te_outer)} windows, "
              f"held-out session(s)={held_sess}) ──")

        # Project outer train and test to tangent space
        X_tr, X_te = _project(cov_mu, cov_beta, use_mu, use_beta, tr_outer, te_outer)

        # Inner random search — rank candidates by mean inner AUC
        best_auc  = -np.inf
        best_cand = candidates[0]

        for ci, cand in enumerate(candidates):
            auc = _score_candidate(X_tr, y_tr_bin, cand, n_inner, XGBClassifier)
            if auc > best_auc:
                best_auc  = auc
                best_cand = cand
            if (ci + 1) % 10 == 0:
                print(f"   [{ci+1:>3}/{n_iter}] best inner AUC so far: {best_auc:.4f}")

        best_params_log.append(best_cand)
        print(f"   Best inner AUC : {best_auc:.4f}")
        print(f"   Best params    : {best_cand}")

        # Refit on full outer train with winning params
        scaler = StandardScaler()
        Xts = scaler.fit_transform(X_tr)
        Xvs = scaler.transform(X_te)
        clf = XGBClassifier(**best_cand, **_FIXED_XGB)
        clf.fit(Xts, y_tr_bin, verbose=False)

        outer_scores = clf.predict_proba(Xvs)[:, 1]
        outer_auc = (
            float(roc_auc_score(y_te_bin, outer_scores))
            if len(np.unique(y_te_bin)) == 2 else float("nan")
        )
        outer_aucs.append(outer_auc)
        print(f"   Outer fold AUC : {outer_auc:.4f}\n")

        all_outer_scores.extend(outer_scores)
        all_outer_true_bin.extend(y_te_bin)

    all_outer_scores   = np.asarray(all_outer_scores)
    all_outer_true_bin = np.asarray(all_outer_true_bin)

    # ── Aggregated results ────────────────────────────────────────────────────
    agg_auc = (
        float(roc_auc_score(all_outer_true_bin, all_outer_scores))
        if len(np.unique(all_outer_true_bin)) == 2 else float("nan")
    )

    print("=" * 64)
    print("RESULTS")
    print("=" * 64)
    print(f"Per-fold outer AUC : {[f'{a:.4f}' for a in outer_aucs]}")
    print(f"Mean outer AUC     : {np.nanmean(outer_aucs):.4f}")
    print(f"Aggregated AUC     : {agg_auc:.4f}")

    # Fixed-threshold sweep on pooled outer test scores (no annotation — no
    # single th_star applies across folds with different best params)
    base._print_fixed_threshold_sweep(all_outer_scores, all_outer_true_bin)

    # ── Param recommendations ─────────────────────────────────────────────────
    print("\n====== Recommended Config Values ======")
    print("Mode across outer folds. Copy into config.py before retraining.\n")

    recommendations = {}
    for param in _PARAM_GRID:
        values = [bp[param] for bp in best_params_log]
        mode_val = Counter(values).most_common(1)[0][0]
        recommendations[param] = mode_val
        fold_str = "  ".join(str(v) for v in values)
        print(f"  {param:<22} = {str(mode_val):<8}  (folds: {fold_str})")

    print("\n── Paste into config.py ──────────────────────────────────────────")
    print(f"XGB_MAX_DEPTH        = {recommendations['max_depth']}")
    print(f"XGB_N_ESTIMATORS     = {recommendations['n_estimators']}")
    print(f"XGB_LEARNING_RATE    = {recommendations['learning_rate']}")
    print(f"XGB_SUBSAMPLE        = {recommendations['subsample']}")
    print(f"XGB_COLSAMPLE_BYTREE = {recommendations['colsample_bytree']}")
    print(f"XGB_REG_ALPHA        = {recommendations['reg_alpha']}")
    print(f"XGB_REG_LAMBDA       = {recommendations['reg_lambda']}")
    print(f"XGB_MIN_CHILD_WEIGHT = {recommendations['min_child_weight']}")
    print("──────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
