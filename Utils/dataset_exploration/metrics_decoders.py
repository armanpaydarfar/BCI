"""
Within-file K-fold evaluation for MDM and XGB backends (OOF scores for distribution metrics).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import config
import Generate_Riemannian_adaptive as base
from pyriemann.classification import MDM
from Utils.tangent_feature_labels import label_cov_tangent_feature, label_erd_flat_feature


def _xgb_params() -> dict:
    return dict(
        n_estimators=int(getattr(config, "XGB_N_ESTIMATORS", 300)),
        max_depth=int(getattr(config, "XGB_MAX_DEPTH", 5)),
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


def _xgb_feature_column_names(feats: dict[str, Any], backend: str, n_features: int) -> list[str]:
    """
    Names aligned with `_assemble_xgb_matrix` column order, using the same channel-pair / ERD
    labeling convention as ``generate_xgboost_cov_*.py`` (tangent index k ↔ upper-tri pair).
    """
    n_ch = int(feats.get("n_channels", 0) or 0)
    n_tang = n_ch * (n_ch + 1) // 2 if n_ch > 0 else 0
    raw_names = feats.get("channel_names")
    ch_list: list[str] | None
    if isinstance(raw_names, list) and len(raw_names) == n_ch and n_ch > 0:
        ch_list = [str(x) for x in raw_names]
    else:
        ch_list = None

    names: list[str] = []
    if n_tang > 0:
        for i in range(n_tang):
            names.append(
                label_cov_tangent_feature(
                    band="mu", idx=i, tangent_dim=n_tang, channel_names=ch_list
                )
            )
    use_beta = bool(getattr(config, "XGB_USE_COV_BETA", 1))
    if backend in ("xgb_cov", "xgb_cov_erd") and use_beta and n_tang > 0:
        for i in range(n_tang):
            names.append(
                label_cov_tangent_feature(
                    band="beta", idx=i, tangent_dim=n_tang, channel_names=ch_list
                )
            )
    if backend == "xgb_cov_erd":
        erd = feats.get("erd")
        if erd is not None and getattr(erd, "ndim", 0) == 2 and erd.shape[1] > 0:
            n_e = int(erd.shape[1])
            for j in range(n_e):
                names.append(
                    label_erd_flat_feature(
                        j,
                        n_channels=n_ch,
                        erd_n_cols=n_e,
                        channel_names=ch_list,
                    )
                )
    if len(names) != n_features:
        return [f"f{i}" for i in range(n_features)]
    return names


def _assemble_xgb_matrix(feats: dict[str, Any], backend: str) -> np.ndarray | None:
    blocks = []
    if backend in ("xgb_cov", "xgb_cov_erd"):
        blocks.append(feats["cov_mu_tangent"])
        use_beta = bool(getattr(config, "XGB_USE_COV_BETA", 1))
        if use_beta:
            bt = feats.get("cov_beta_tangent")
            if bt is None:
                return None
            blocks.append(bt)
    if backend == "xgb_cov_erd":
        erd = feats.get("erd")
        if erd is None:
            return None
        blocks.append(erd)
    return np.hstack(blocks) if blocks else None


def kfold_decoder_metrics(
    feats: dict[str, Any],
    *,
    backends: list[str],
    n_splits: int = 5,
    random_state: int = 42,
    target_ambig: float = float(getattr(config, "TARGET_AMBIG", 0.20)),
    score_threshold: float = 0.5,
    on_kfold_step: Callable[[int, int], None] | None = None,
) -> dict[str, float | str]:
    """
    Run stratified K-fold per backend. Returns flat metrics with prefixes mdm__, xgb_cov__, xgb_cov_erd__.

    on_kfold_step: if set, called as ``on_kfold_step(fold_1based, n_folds)`` after each fold finishes for
    all runnable backends (same CV split for every backend).
    """
    if "error" in feats:
        return {"eval_error": feats["error"]}

    labels = feats["labels"]
    rest_label = feats["rest_label"]
    mi_label = feats["mi_label"]
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) != 2 or np.min(counts) < 2:
        return {"eval_error": "insufficient_class_samples"}

    n_splits = int(min(n_splits, int(np.min(counts))))
    if n_splits < 2:
        return {"eval_error": "n_splits_lt_2"}

    y_bin = np.where(labels == mi_label, 1, 0).astype(int)
    out: dict[str, float | str] = {}

    to_run: list[tuple[str, np.ndarray]] = []
    for backend in backends:
        if backend == "mdm":
            X = feats["cov_mu"]
        elif backend in ("xgb_cov", "xgb_cov_erd"):
            Xb = _assemble_xgb_matrix(feats, backend)
            if Xb is None:
                out[f"{backend}__error"] = "missing_features"
                continue
            X = Xb
        else:
            out[f"{backend}__error"] = "unknown_backend"
            continue
        to_run.append((backend, X))

    if not to_run:
        return out

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_indices = list(skf.split(np.zeros((len(labels), 1)), labels))
    n_folds = len(fold_indices)

    state: dict[str, dict[str, Any]] = {}
    for backend, _X in to_run:
        state[backend] = {
            "oof_p_mi": np.full(len(labels), np.nan),
            "oof_pred_argmax": np.full(len(labels), -1, dtype=int),
            "decided_bal_accs": [],
            "ambig_fracs": [],
        }

    for fold_1b, (train_idx, test_idx) in enumerate(fold_indices, start=1):
        for backend, X in to_run:
            st = state[backend]
            oof_p_mi = st["oof_p_mi"]
            oof_pred_argmax = st["oof_pred_argmax"]
            decided_bal_accs: list[float] = st["decided_bal_accs"]
            ambig_fracs: list[float] = st["ambig_fracs"]

            if backend == "mdm":
                clf = MDM()
                clf.fit(X[train_idx], labels[train_idx])
                pr = clf.predict_proba(X[test_idx])
                idx_mi = int(np.where(clf.classes_ == mi_label)[0][0])
                p_mi_te = pr[:, idx_mi]
                pred = clf.predict(X[test_idx])
            else:
                from xgboost import XGBClassifier

                y_tr = y_bin[train_idx]
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X[train_idx])
                X_te = scaler.transform(X[test_idx])
                clf = XGBClassifier(**_xgb_params())
                clf.fit(X_tr, y_tr)
                pr = clf.predict_proba(X_te)
                p_mi_te = pr[:, 1]
                pred_bin = clf.predict(X_te).astype(int)
                pred = np.where(pred_bin == 1, mi_label, rest_label)

                prob_tr = clf.predict_proba(X_tr)
                scr_tr = prob_tr[:, 1]
                tl, th, _diag = base.pick_dual_thresholds_target_ambiguity(
                    y_true_bin=y_tr,
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
                pred_dt = np.full(len(test_idx), -1, dtype=int)
                pred_dt[p_mi_te >= th] = mi_label
                pred_dt[p_mi_te <= tl] = rest_label
                y_te = labels[test_idx]
                ambig_fracs.append(float(np.mean(pred_dt == -1)))
                decided = pred_dt != -1
                if decided.any():
                    decided_bal_accs.append(
                        float(balanced_accuracy_score(y_te[decided], pred_dt[decided]))
                    )

            oof_p_mi[test_idx] = p_mi_te
            oof_pred_argmax[test_idx] = pred

        if on_kfold_step is not None:
            on_kfold_step(fold_1b, n_folds)

    for backend, X in to_run:
        st = state[backend]
        oof_p_mi = st["oof_p_mi"]
        oof_pred_argmax = st["oof_pred_argmax"]
        decided_bal_accs = st["decided_bal_accs"]
        ambig_fracs = st["ambig_fracs"]

        valid = ~np.isnan(oof_p_mi)
        if not np.any(valid):
            out[f"{backend}__error"] = "no_oof"
            continue

        yt = labels[valid]
        pm = oof_p_mi[valid]
        pa = oof_pred_argmax[valid]

        yb = (yt == mi_label).astype(int)
        auc = float(roc_auc_score(yb, pm)) if np.unique(yb).size == 2 else float("nan")
        acc = float(accuracy_score(yt, pa))
        bal = float(balanced_accuracy_score(yt, pa))
        brier = float(brier_score_loss(yb, pm)) if np.unique(yb).size == 2 else float("nan")
        cm = confusion_matrix(yt, pa, labels=[rest_label, mi_label])

        out[f"{backend}__auc"] = auc
        out[f"{backend}__accuracy_argmax"] = acc
        out[f"{backend}__balanced_accuracy_argmax"] = bal
        out[f"{backend}__brier_pmi"] = brier

        p_rest = 1.0 - pm
        score_correct = np.where(yt == mi_label, pm, p_rest)
        thr_key = str(score_threshold).replace(".", "p")
        out[f"{backend}__frac_correct_prob_above_{thr_key}"] = float(
            np.mean(score_correct > score_threshold)
        )

        out[f"{backend}__skew_p_mi"] = float(scipy_stats.skew(pm)) if pm.size > 2 else float("nan")
        out[f"{backend}__kurtosis_p_mi"] = float(scipy_stats.kurtosis(pm)) if pm.size > 2 else float("nan")

        if cm.size == 4:
            out[f"{backend}__cm_tn"] = float(cm[0, 0])
            out[f"{backend}__cm_fp"] = float(cm[0, 1])
            out[f"{backend}__cm_fn"] = float(cm[1, 0])
            out[f"{backend}__cm_tp"] = float(cm[1, 1])

        if backend != "mdm" and decided_bal_accs:
            out[f"{backend}__decided_balanced_accuracy_mean"] = float(np.mean(decided_bal_accs))
        if backend != "mdm" and ambig_fracs:
            out[f"{backend}__ambiguous_rate_mean"] = float(np.mean(ambig_fracs))

        if backend in ("xgb_cov", "xgb_cov_erd"):
            top_n = max(1, int(getattr(config, "XGB_IMPORTANCE_TOP_K", 8)))
            col_names = _xgb_feature_column_names(feats, backend, int(X.shape[1]))
            try:
                from xgboost import XGBClassifier

                scaler_full = StandardScaler()
                X_full = scaler_full.fit_transform(X)
                clf_full = XGBClassifier(**_xgb_params())
                clf_full.fit(X_full, y_bin)
                imp = np.asarray(clf_full.feature_importances_, dtype=float)
                order = np.argsort(-imp)[:top_n]
                parts = [f"{col_names[int(i)]}={float(imp[int(i)]):.3f}" for i in order]
                out[f"{backend}__top_features_gain"] = "; ".join(parts)
            except Exception as e:
                out[f"{backend}__top_features_gain"] = f"failed: {type(e).__name__}"

    return out
