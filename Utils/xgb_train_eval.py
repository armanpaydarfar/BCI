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

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler


def train_xgb_dual_thresholds(
    X: np.ndarray,
    labels: np.ndarray,
    feature_tag: str,
    n_splits: int,
    target_ambig: float = float(getattr(config, "TARGET_AMBIG", 0.20)),
):
    """
    Train/evaluate an XGBClassifier with the same dual-threshold selection logic
    as the canonical Riemannian pipeline.

    Returns:
        dict containing scaler, model, and label mapping.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is required for XGB branches. Install with `pip install xgboost`.") from exc

    classes = np.sort(np.unique(labels))
    if len(classes) != 2:
        raise ValueError("Function expects binary classes.")

    rest_label, mi_label = classes[0], classes[1]
    label_to_bin = {rest_label: 0, mi_label: 1}
    bin_to_label = {0: rest_label, 1: mi_label}
    y_bin = np.asarray([label_to_bin[v] for v in labels], dtype=int)

    xgb_params = dict(
        n_estimators=int(getattr(config, "XGB_N_ESTIMATORS", 300)),
        max_depth=int(getattr(config, "XGB_MAX_DEPTH", 3)),
        learning_rate=float(getattr(config, "XGB_LEARNING_RATE", 0.03)),
        subsample=float(getattr(config, "XGB_SUBSAMPLE", 0.8)),
        colsample_bytree=float(getattr(config, "XGB_COLSAMPLE_BYTREE", 0.8)),
        reg_alpha=float(getattr(config, "XGB_REG_ALPHA", 0.0)),
        reg_lambda=float(getattr(config, "XGB_REG_LAMBDA", 2.0)),
        min_child_weight=float(getattr(config, "XGB_MIN_CHILD_WEIGHT", 3.0)),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )

    print(f"\n🚀 Starting K-Fold CV ({feature_tag}) + XGBoost...\n")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    acc_argmax = []
    t_lows, t_highs = [], []
    all_true, all_pred, all_scores, all_true_bin = [], [], [], []
    posterior_probs = {lbl: [] for lbl in classes}

    for fold_idx, (tr, te) in enumerate(kf.split(X), 1):
        X_tr, X_te = X[tr], X[te]
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

    base._plot_scores_hist_with_thresholds(all_scores, all_true_bin, tl_star, th_star)
    center = (tl_star + th_star) / 2.0
    widths = np.linspace(0.0, 0.9, 35)
    base._plot_risk_coverage(all_scores, all_true_bin, center, widths, 1.0, 1.0, 0.3)
    base._plot_roc_with_point(all_scores, all_true_bin, th_star)
    base._plot_confusion_with_reject(all_true, all_pred, rest_label, mi_label)
    for lbl in posterior_probs:
        posterior_probs[lbl] = np.asarray(posterior_probs[lbl])
    base.plot_posterior_probabilities(posterior_probs)

    # Final fit on full data
    final_scaler = StandardScaler()
    Xs = final_scaler.fit_transform(X)
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

