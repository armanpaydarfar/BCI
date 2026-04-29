"""
Shared evaluation/training for XGBoost branches with dual thresholds.

This matches the reporting conventions used in the current XGBoost feature
scripts and reuses `Generate_Riemannian_adaptive.py` threshold/plot helpers.
"""

import os
import numpy as np

os.environ["MNE_USE_NUMBA"] = "false"

import config
import Generate_Riemannian_adaptive as base

from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler


def train_xgb_dual_thresholds(
    cov_mu: np.ndarray | None,
    cov_beta: np.ndarray | None,
    tangent_fn: tuple,
    labels: np.ndarray,
    session_ids: np.ndarray,
    feature_tag: str,
    n_splits: int,
    target_ambig: float = float(getattr(config, "TARGET_AMBIG", 0.20)),
):
    """
    Train/evaluate an XGBClassifier with the same dual-threshold selection logic
    as the canonical Riemannian pipeline.

    The tangent space reference is fitted inside each CV fold on the train split
    only, eliminating the data leak that occurred when the reference was fitted
    on the full dataset before splitting.

    CV mode is controlled by config.CV_MODE:
      "kfold"       — shuffled KFold(n_splits).
      "session_loo" — GroupKFold respecting session boundaries; at most
                      min(n_sessions, config.N_LOO_SPLITS) folds.

    Args:
        cov_mu:      Whitened SPD covariances for the mu band, shape (N, C, C),
                     or None if mu features are disabled.
        cov_beta:    Whitened SPD covariances for the beta band, shape (N, C, C),
                     or None if beta features are disabled.
        tangent_fn:  Tuple of (fit_ref_fn, project_fn) used to project covs to
                     tangent space.  fit_ref_fn(covs) -> ref;
                     project_fn(covs, ref) -> features.
        labels:      Integer class labels, shape (N,).
        session_ids: Integer session index per trial, shape (N,).  Used by
                     session_loo to group trials.
        feature_tag: String label printed in the CV report.
        n_splits:    Number of KFold splits (ignored for session_loo when
                     n_sessions <= N_LOO_SPLITS).
        target_ambig: Target ambiguity fraction for dual-threshold selection.

    Returns:
        dict containing scaler, model, and label mapping.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is required for XGB branches. Install with `pip install xgboost`.") from exc

    if cov_mu is None and cov_beta is None:
        raise ValueError("At least one of cov_mu or cov_beta must be provided.")

    fit_ref, project = tangent_fn

    cv_mode = str(getattr(config, "CV_MODE", "kfold")).lower()
    n_loo_splits = int(getattr(config, "N_LOO_SPLITS", 5))

    classes = np.sort(np.unique(labels))
    if len(classes) != 2:
        raise ValueError("Function expects binary classes.")

    rest_label, mi_label = classes[0], classes[1]
    label_to_bin = {rest_label: 0, mi_label: 1}
    bin_to_label = {0: rest_label, 1: mi_label}
    y_bin = np.asarray([label_to_bin[v] for v in labels], dtype=int)

    xgb_params = dict(
        n_estimators=int(getattr(config, "XGB_N_ESTIMATORS", 300)),
        max_depth=int(getattr(config, "XGB_MAX_DEPTH", 6)),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    # Apply optional overrides from config; params absent from config use XGBoost defaults.
    for _param, _cfg_key, _cast in [
        ("learning_rate",    "XGB_LEARNING_RATE",    float),
        ("subsample",        "XGB_SUBSAMPLE",         float),
        ("colsample_bytree", "XGB_COLSAMPLE_BYTREE",  float),
        ("reg_alpha",        "XGB_REG_ALPHA",          float),
        ("reg_lambda",       "XGB_REG_LAMBDA",         float),
        ("min_child_weight", "XGB_MIN_CHILD_WEIGHT",   float),
    ]:
        _val = getattr(config, _cfg_key, None)
        if _val is not None:
            xgb_params[_param] = _cast(_val)

    # Build the CV splitter.
    # Use the first available covariance array as the object passed to .split()
    # (only its length matters for index generation).
    _ref_arr = cov_mu if cov_mu is not None else cov_beta

    if cv_mode == "session_loo":
        n_sessions = len(np.unique(session_ids))
        k = min(n_sessions, n_loo_splits)
        splitter = GroupKFold(n_splits=k)
        split_iter = splitter.split(_ref_arr, groups=session_ids)
        print(
            f"\n🚀 Starting Session-LOO CV ({feature_tag}) + XGBoost "
            f"[{k} folds over {n_sessions} sessions]...\n"
        )
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = splitter.split(_ref_arr)
        print(f"\n🚀 Starting K-Fold CV ({feature_tag}) + XGBoost [{n_splits} folds]...\n")

    acc_argmax = []
    t_lows, t_highs = [], []
    all_true, all_pred, all_scores, all_true_bin = [], [], [], []
    posterior_probs = {lbl: [] for lbl in classes}

    for fold_idx, (tr, te) in enumerate(split_iter, 1):
        # Project covs to tangent space using a reference fitted on train split only.
        feature_blocks_tr, feature_blocks_te = [], []
        if cov_mu is not None:
            ref_mu = fit_ref(cov_mu[tr])
            feature_blocks_tr.append(project(cov_mu[tr], ref_mu))
            feature_blocks_te.append(project(cov_mu[te], ref_mu))
        if cov_beta is not None:
            ref_beta = fit_ref(cov_beta[tr])
            feature_blocks_tr.append(project(cov_beta[tr], ref_beta))
            feature_blocks_te.append(project(cov_beta[te], ref_beta))

        X_tr = np.hstack(feature_blocks_tr)
        X_te = np.hstack(feature_blocks_te)
        y_tr, y_te = labels[tr], labels[te]
        y_tr_bin = y_bin[tr]

        scaler = StandardScaler()
        X_trs = scaler.fit_transform(X_tr)
        X_tes = scaler.transform(X_te)

        clf = XGBClassifier(**xgb_params)
        clf.fit(X_trs, y_tr_bin)

        prob_tr = clf.predict_proba(X_trs)
        prob_te = clf.predict_proba(X_tes)

        scr_tr = prob_tr[:, 1]
        scr_te = prob_te[:, 1]

        y_pred_bin = clf.predict(X_tes).astype(int)
        y_pred = np.asarray([bin_to_label[int(v)] for v in y_pred_bin], dtype=int)

        acc = accuracy_score(y_te, y_pred)
        acc_argmax.append(acc)
        print(f"✅ Fold {fold_idx} Argmax Accuracy ({feature_tag}): {acc:.4f}")

        tl, th, diag = base.pick_dual_thresholds_target_ambiguity(
            y_true_bin=y_tr_bin,
            pos_scores=scr_tr,
            target_ambig=target_ambig,
            c_fp=1.0,
            c_fn=1.0,
            n_grid=201,
            min_gap=0.0,
            tpr_min=None,
            fpr_max=None,
            ppv_min=None,
            npv_min=None,
            require_center_around_half=False,
        )
        print(
            f"   ↳ thresholds@ambig={target_ambig:.2f}: "
            f"t_low={tl:.3f}, t_high={th:.3f}, feasible={diag.get('feasible', False)}"
        )

        t_lows.append(tl)
        t_highs.append(th)

        pred = np.full_like(y_te, -1)
        pred[scr_te >= th] = mi_label
        pred[scr_te <= tl] = rest_label

        all_true.extend(y_te)
        all_pred.extend(pred)
        all_scores.extend(scr_te)
        all_true_bin.extend((y_te == mi_label).astype(int))

        for i, true_lbl in enumerate(y_te):
            posterior_probs[true_lbl].append(prob_te[i, label_to_bin[true_lbl]])

    all_true = np.asarray(all_true)
    all_pred = np.asarray(all_pred)
    all_scores = np.asarray(all_scores)
    all_true_bin = np.asarray(all_true_bin)

    roc_auc = float(roc_auc_score(all_true_bin, all_scores)) if np.unique(all_true_bin).size == 2 else np.nan

    decided = all_pred != -1
    cm_decided = confusion_matrix(all_true[decided], all_pred[decided], labels=[rest_label, mi_label])
    TN, FP, FN, TP = cm_decided.ravel()
    U = (all_pred == -1).sum()
    coverage = decided.mean()
    decided_acc = (TP + TN) / (TP + TN + FP + FN) if decided.any() else np.nan
    cost = 1.0 * FP + 1.0 * FN + 0.3 * U

    tl_star, th_star = float(np.median(t_lows)), float(np.median(t_highs))

    print("\n====== Aggregated Report ======")
    print(f"Argmax Accuracy (mean) [{feature_tag}]: {np.mean(acc_argmax):.4f}")
    print(f"ROC AUC (fold-test aggregated) [{feature_tag}]: {roc_auc:.4f}")
    print(f"Learned thresholds (medians) [{feature_tag}]: t_low*={tl_star:.3f}, t_high*={th_star:.3f}")
    print(f"Coverage (decided %): {coverage*100:.2f}% (Ambiguity {(1.0-coverage)*100:.2f}%)")
    print(f"Decided-only Accuracy: {decided_acc:.4f}")
    print(f"Overall Cost (c_fp=1.0, c_fn=1.0, c_reject=0.3): {cost:.1f}")
    print("\nConfusion (decided-only; rows=true [REST, MI], cols=pred [REST, MI]):")
    print(cm_decided)

    base._print_kl_report(
        all_scores, all_true_bin,
        beta_a=float(getattr(config, "XGB_TUNE_BETA_ALPHA", 18)),
        beta_b=float(getattr(config, "XGB_TUNE_BETA_BETA",   5)),
        n_bins=int(getattr(config,   "XGB_TUNE_KL_BINS",    15)),
    )
    base._print_fixed_threshold_sweep(all_scores, all_true_bin, th_star)

    base._plot_scores_hist_with_thresholds(all_scores, all_true_bin, tl_star, th_star)
    center = (tl_star + th_star) / 2.0
    widths = np.linspace(0.0, 0.9, 35)
    base._plot_risk_coverage(all_scores, all_true_bin, center, widths, 1.0, 1.0, 0.3)
    base._plot_roc_with_point(all_scores, all_true_bin, th_star)
    base._plot_confusion_fixed_threshold(all_scores, all_true_bin, rest_label, mi_label, t_high=0.65)
    for lbl in posterior_probs:
        posterior_probs[lbl] = np.asarray(posterior_probs[lbl])
    base.plot_posterior_probabilities(posterior_probs)

    # Final fit on full data using a reference fitted on all covariances.
    final_blocks = []
    if cov_mu is not None:
        final_blocks.append(project(cov_mu, fit_ref(cov_mu)))
    if cov_beta is not None:
        final_blocks.append(project(cov_beta, fit_ref(cov_beta)))
    X_all = np.hstack(final_blocks)
    final_scaler = StandardScaler()
    Xs = final_scaler.fit_transform(X_all)
    final_clf = XGBClassifier(**xgb_params)
    final_clf.fit(Xs, y_bin)

    return {
        "scaler": final_scaler,
        "model": final_clf,
        "label_to_bin": label_to_bin,
        "bin_to_label": bin_to_label,
        "tl_star": tl_star,
        "th_star": th_star,
        "roc_auc": roc_auc,
    }

