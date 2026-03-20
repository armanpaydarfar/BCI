"""
Improved hyperparameter grid search for XGBoost on cov+ERD features (KFold).

Testing convention:
  - Uses plain KFold over window-level samples (like `Utils/xgb_train_eval.py`
    and the previous cov+ERD training scripts).
  - For each `shrinkage_param`, covariance preprocessing (shrinkage + riemannian
    whitening) is computed on the full dataset first, then KFold is applied.
    This matches the previous scripts' evaluation convention (no LOO).

Feature vector:
  X = [cov_tangent_features | erd_logratio_bandpower_features]

Grid search:
  - max_depth (never < 3)
  - shrinkage_param (config.SHRINKAGE_PARAM in your pipeline)
  - optionally: min_child_weight, gamma, subsample, colsample_bytree, reg_lambda,
    learning_rate, n_estimators (controlled via env vars).

Outputs:
  - CSV report with ROC AUC as the main selector metric
  - ROC curves plot for the top-N parameter combinations
"""

import os
import csv
import pickle
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple

os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")
os.environ["MNE_USE_NUMBA"] = "false"

import numpy as np
import matplotlib.pyplot as plt

import config
import Generate_Riemannian_adaptive as base

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from xgboost import XGBClassifier

from pyriemann.estimation import Shrinkage
from pyriemann.preprocessing import Whitening
from pyriemann.utils.tangentspace import tangent_space

from sklearn.covariance import LedoitWolf

from Utils.stream_utils import load_xdf
from Utils.xgb_feature_pipeline import segment_and_extract_cov_erd


@dataclass(frozen=True)
class HyperGrid:
    max_depth: List[int]
    shrinkage: List[float]
    min_child_weight: List[float]
    gamma: List[float]
    subsample: List[float]
    colsample_bytree: List[float]
    reg_lambda: List[float]
    learning_rate: List[float]
    n_estimators: List[int]


def _env_list(name: str, default: List):
    v = os.getenv(name, None)
    if v is None:
        return default
    parts = [p.strip() for p in v.split(",") if p.strip()]
    if not parts:
        return default
    # Infer numeric types from default list element type.
    t = type(default[0]) if default else float
    if t is int:
        return [int(p) for p in parts]
    if t is float:
        return [float(p) for p in parts]
    return parts


def _compute_trace_normalized_covariances(segments: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    segments: (N, C, T)
    returns: (N, C, C) trace-normalized covariance (sample-wise)
    """
    cov = np.einsum("nct,ndt->ncd", segments, segments)
    tr = np.trace(cov, axis1=1, axis2=2)
    cov = cov / (tr[:, None, None] + eps)
    cov = 0.5 * (cov + np.transpose(cov, (0, 2, 1)))
    return cov


def _cov_preprocess_full_dataset(cov_raw: np.ndarray, shrinkage_param: float) -> np.ndarray:
    """
    Match canonical preprocessing boundaries (but computed on full dataset):
      trace-normalized cov -> shrinkage -> riemannian whitening (if enabled)
    """
    if config.LEDOITWOLF:
        cov_shrunk = np.array([LedoitWolf().fit(c).covariance_ for c in cov_raw])
    else:
        shrinker = Shrinkage(shrinkage=shrinkage_param)
        cov_shrunk = shrinker.fit_transform(cov_raw)

    if config.RECENTERING:
        whitener = Whitening(metric="riemann")
        cov_processed = whitener.fit_transform(cov_shrunk)
    else:
        cov_processed = cov_shrunk

    return cov_processed


def _tangent_features_at_identity(cov_processed: np.ndarray) -> np.ndarray:
    n_ch = cov_processed.shape[1]
    I = np.eye(n_ch)
    return tangent_space(cov_processed, I, metric="riemann")


def _train_eval_one_combo(
    X: np.ndarray,
    labels: np.ndarray,
    xgb_params: Dict,
    n_splits: int,
    target_ambig: float = 0.30,
):
    """
    KFold evaluation + dual thresholds on fold-train scores.
    Returns aggregated metrics and ROC arrays for plotting.
    """
    classes = np.sort(np.unique(labels))
    if len(classes) != 2:
        raise ValueError(f"Expected binary labels only; got classes={classes}")
    rest_label, mi_label = int(classes[0]), int(classes[1])
    label_to_bin = {rest_label: 0, mi_label: 1}
    bin_to_label = {0: rest_label, 1: mi_label}
    y_bin = np.asarray([label_to_bin[int(v)] for v in labels], dtype=int)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    acc_argmax_folds = []
    t_lows, t_highs = [], []
    all_true_bin, all_scores = [], []

    decided_mask_all = []
    all_true_all, all_pred_all = [], []

    for tr, te in kf.split(X):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = labels[tr], labels[te]
        y_tr_bin, y_te_bin = y_bin[tr], y_bin[te]

        scaler = StandardScaler()
        X_trs = scaler.fit_transform(X_tr)
        X_tes = scaler.transform(X_te)

        clf = XGBClassifier(**xgb_params)
        clf.fit(X_trs, y_tr_bin)

        prob_tr = clf.predict_proba(X_trs)
        prob_te = clf.predict_proba(X_tes)
        scr_tr = prob_tr[:, 1]
        scr_te = prob_te[:, 1]

        # Argmax
        y_pred_argmax_bin = clf.predict(X_tes).astype(int)
        y_pred_argmax = np.asarray([bin_to_label[int(v)] for v in y_pred_argmax_bin], dtype=int)
        acc_argmax = accuracy_score(y_te, y_pred_argmax)
        acc_argmax_folds.append(acc_argmax)

        # Dual thresholds on fold-train scores
        tl, th, _diag = base.pick_dual_thresholds_target_ambiguity(
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
        t_lows.append(tl)
        t_highs.append(th)

        pred = np.full_like(y_te, fill_value=-1, dtype=int)
        pred[scr_te >= th] = mi_label
        pred[scr_te <= tl] = rest_label

        decided = pred != -1
        decided_acc = accuracy_score(y_te[decided], pred[decided]) if np.any(decided) else np.nan

        all_true_all.extend(y_te.tolist())
        all_pred_all.extend(pred.tolist())

        all_true_bin.extend(y_te_bin.tolist())
        all_scores.extend(scr_te.tolist())

    all_true_all = np.asarray(all_true_all, dtype=int)
    all_pred_all = np.asarray(all_pred_all, dtype=int)
    all_true_bin = np.asarray(all_true_bin, dtype=int)
    all_scores = np.asarray(all_scores, dtype=float)

    roc_auc = roc_auc_score(all_true_bin, all_scores) if np.unique(all_true_bin).size == 2 else np.nan

    decided = all_pred_all != -1
    cm_decided = confusion_matrix(
        all_true_all[decided],
        all_pred_all[decided],
        labels=[rest_label, mi_label],
    )
    TN, FP, FN, TP = cm_decided.ravel()
    U = int((all_pred_all == -1).sum())
    coverage = float(decided.mean())
    decided_acc = (TP + TN) / (TP + TN + FP + FN) if decided.any() else np.nan
    cost = 1.0 * FP + 1.0 * FN + 0.3 * U

    tl_star = float(np.median(t_lows))
    th_star = float(np.median(t_highs))

    # ROC curve points for plotting
    fpr, tpr, _thr = roc_curve(all_true_bin, all_scores)

    metrics = {
        "roc_auc": float(roc_auc),
        "argmax_acc_mean": float(np.mean(acc_argmax_folds)),
        "decided_acc": float(decided_acc),
        "coverage": float(coverage),
        "cost": float(cost),
        "tl_star": tl_star,
        "th_star": th_star,
        "cm_decided": cm_decided.tolist(),
        "fpr": fpr,
        "tpr": tpr,
    }
    return metrics


def main():
    mne_verbose = "WARNING"
    try:
        import mne

        mne.set_log_level(mne_verbose)
    except Exception:
        pass

    grid = HyperGrid(
        max_depth=_env_list("XGB_GRID_MAX_DEPTH", [3, 4, 5, 6]),
        shrinkage=_env_list("XGB_GRID_SHRINKAGE", [0.01, 0.02, 0.05, 0.1, 0.2]),
        min_child_weight=_env_list("XGB_GRID_MIN_CHILD_WEIGHT", [3.0]),
        gamma=_env_list("XGB_GRID_GAMMA", [0.0]),
        subsample=_env_list("XGB_GRID_SUBSAMPLE", [0.8]),
        colsample_bytree=_env_list("XGB_GRID_COLSAMPLE_BYTREE", [0.8]),
        reg_lambda=_env_list("XGB_GRID_REG_LAMBDA", [2.0]),
        learning_rate=_env_list("XGB_GRID_LEARNING_RATE", [float(getattr(config, "XGB_LEARNING_RATE", 0.03))]),
        n_estimators=_env_list("XGB_GRID_N_ESTIMATORS", [int(getattr(config, "XGB_N_ESTIMATORS", 300))]),
    )

    # Hard constraint: do not go under 3
    grid.max_depth = [int(md) for md in grid.max_depth if int(md) >= 3]
    if not grid.max_depth:
        raise ValueError("max_depth grid ended up empty after enforcing md>=3")

    target_ambig = float(os.getenv("XGB_TARGET_AMBIG", "0.30"))
    top_k_roc = int(os.getenv("XGB_GRID_TOP_K_ROC", "5"))

    n_splits = int(getattr(config, "N_SPLITS", 8))
    if n_splits < 2:
        raise ValueError("Need N_SPLITS>=2")

    print("=== Improved KFold Grid Search (cov+ERD) ===")
    print(f"max_depth_grid={grid.max_depth}")
    print(f"shrinkage_grid={grid.shrinkage}")
    print(f"target_ambig={target_ambig}")
    print(f"n_splits={n_splits}")

    apply_csd = bool(getattr(config, "SURFACE_LAPLACIAN_TOGGLE", False))
    print(f"[cov+erd] APPLY_CSD(surface laplacian)={apply_csd}")

    eeg_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "training_data")
    xdf_files = [
        os.path.join(eeg_dir, f) for f in os.listdir(eeg_dir)
        if f.endswith(".xdf") and "OBS" not in f
    ]
    if not xdf_files:
        raise FileNotFoundError(f"No XDF files found in: {eeg_dir}")
    xdf_files = sorted(xdf_files)

    cov_raw_parts = []
    erd_parts = []
    labels_parts = []
    for xdf_path in xdf_files:
        print(f"\n📂 Loading: {xdf_path}")
        eeg_stream, marker_stream = load_xdf(xdf_path, report=False)
        segments, labels, erd_feats = segment_and_extract_cov_erd(
            eeg_stream,
            marker_stream,
            compute_erd=True,
            apply_csd=apply_csd,
        )
        if erd_feats is None:
            raise RuntimeError("Expected ERD features but got None.")

        cov_raw = _compute_trace_normalized_covariances(segments)
        cov_raw_parts.append(cov_raw)
        erd_parts.append(erd_feats)
        labels_parts.append(labels)

        # free memory
        del segments

    cov_raw_all = np.concatenate(cov_raw_parts, axis=0)
    erd_all = np.concatenate(erd_parts, axis=0)
    labels_all = np.concatenate(labels_parts, axis=0)

    print(f"\nLoaded dataset: N={labels_all.shape[0]} | cov_raw={cov_raw_all.shape} | erd={erd_all.shape}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "models")
    os.makedirs(models_dir, exist_ok=True)
    out_csv = os.path.join(models_dir, f"grid_kfold_improved_cov_erd_{timestamp}.csv")
    out_roc = os.path.join(models_dir, f"grid_kfold_improved_cov_erd_roc_{timestamp}.png")

    results_rows = []
    roc_store: Dict[Tuple, Tuple[np.ndarray, np.ndarray, float]] = {}

    # Iterate over shrinkage: covariance preprocessing is the expensive step
    for shrink in grid.shrinkage:
        print(f"\n=== Precomputing covariance preprocessing for shrinkage={shrink} ===")
        cov_processed = _cov_preprocess_full_dataset(cov_raw_all, shrinkage_param=float(shrink))
        cov_feats_all = _tangent_features_at_identity(cov_processed)  # tangent on whitened cov

        # Precompute once for speed
        X_cov_erd = None  # will be constructed per fold

        # Build X once for this shrinkage (cov_feats_all changes with shrinkage)
        X_all = np.hstack([cov_feats_all, erd_all])

        for md in grid.max_depth:
            for min_child_weight in grid.min_child_weight:
                for gamma in grid.gamma:
                    for reg_lambda in grid.reg_lambda:
                        for lr in grid.learning_rate:
                            for n_est in grid.n_estimators:
                                for subsample in grid.subsample:
                                    for colsample in grid.colsample_bytree:
                                        xgb_params = dict(
                                            n_estimators=int(n_est),
                                            max_depth=int(md),
                                            learning_rate=float(lr),
                                            subsample=float(subsample),
                                            colsample_bytree=float(colsample),
                                            reg_alpha=0.0,
                                            reg_lambda=float(reg_lambda),
                                            min_child_weight=float(min_child_weight),
                                            gamma=float(gamma),
                                            objective="binary:logistic",
                                            eval_metric="logloss",
                                            random_state=42,
                                        )

                                        metrics = _train_eval_one_combo(
                                            X=X_all,
                                            labels=labels_all,
                                            xgb_params=xgb_params,
                                            n_splits=n_splits,
                                            target_ambig=target_ambig,
                                        )

                                        row = {
                                            "max_depth": int(md),
                                            "shrinkage": float(shrink),
                                            "min_child_weight": float(min_child_weight),
                                            "gamma": float(gamma),
                                            "reg_lambda": float(reg_lambda),
                                            "learning_rate": float(lr),
                                            "n_estimators": int(n_est),
                                            "subsample": float(subsample),
                                            "colsample_bytree": float(colsample),
                                            "roc_auc": metrics["roc_auc"],
                                            "argmax_acc_mean": metrics["argmax_acc_mean"],
                                            "decided_acc": metrics["decided_acc"],
                                            "coverage": metrics["coverage"],
                                            "cost": metrics["cost"],
                                            "tl_star": metrics["tl_star"],
                                            "th_star": metrics["th_star"],
                                            "cm_decided": str(metrics["cm_decided"]),
                                        }
                                        results_rows.append(row)

                                        key = (
                                            int(md),
                                            round(float(shrink), 6),
                                            float(min_child_weight),
                                            float(gamma),
                                            float(reg_lambda),
                                            float(lr),
                                            int(n_est),
                                            float(subsample),
                                            float(colsample),
                                        )
                                        roc_store[key] = (metrics["fpr"], metrics["tpr"], metrics["roc_auc"])

                                        print(
                                            f"Combo md={md}, sh={shrink}, "
                                            f"min_child={min_child_weight}, gamma={gamma}, reg_lam={reg_lambda}, "
                                            f"lr={lr}, n_est={n_est}, subs={subsample}, col={colsample}: "
                                            f"AUC={metrics['roc_auc']:.4f} argmax={metrics['argmax_acc_mean']:.4f} "
                                            f"decided={metrics['decided_acc']:.4f} cov={metrics['coverage']*100:.1f}% cost={metrics['cost']:.1f}"
                                        )

    # Write CSV
    fieldnames = list(results_rows[0].keys()) if results_rows else []
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results_rows:
            writer.writerow(r)

    # Plot top ROC curves
    best_rows = sorted(results_rows, key=lambda r: r["roc_auc"], reverse=True)[:top_k_roc]
    plt.figure(figsize=(7, 6))
    for r in best_rows:
        key = (
            int(r["max_depth"]),
            round(float(r["shrinkage"]), 6),
            float(r["min_child_weight"]),
            float(r["gamma"]),
            float(r["reg_lambda"]),
            float(r["learning_rate"]),
            int(r["n_estimators"]),
            float(r["subsample"]),
            float(r["colsample_bytree"]),
        )
        fpr, tpr, auc = roc_store[key]
        plt.plot(
            fpr,
            tpr,
            lw=2,
            label=(
                f"md={int(r['max_depth'])}, sh={round(float(r['shrinkage']),6)} "
                f"AUC={auc:.3f}"
            ),
        )

    plt.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Top ROC Curves (cov+ERD) - KFold")
    plt.legend(fontsize=9)
    plt.grid(True, ls=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_roc, dpi=160)
    plt.close()

    # Report best
    best = max(results_rows, key=lambda r: r["roc_auc"]) if results_rows else None
    print("\n=== Grid Search Done ===")
    print(f"CSV: {out_csv}")
    print(f"ROC plot: {out_roc}")
    if best:
        print(
            f"Best by AUC: max_depth={best['max_depth']}, shrinkage={best['shrinkage']} -> "
            f"AUC={best['roc_auc']:.4f}, decided_acc={best['decided_acc']:.4f}, coverage={best['coverage']*100:.1f}%"
        )


if __name__ == "__main__":
    main()

