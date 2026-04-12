"""
Threshold-free session-held-out transfer benchmark core.

This module provides:
- Shared session-level data assembly
- Model adapters (MDM, XGB cov-only, XGB cov+ERD)
- Unified fold/aggregate metric reporting
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.preprocessing import StandardScaler

from pyriemann.classification import MDM
from pyriemann.tangentspace import tangent_space

import config
import Generate_Riemannian_adaptive as base
from Utils.stream_utils import load_xdf
from Utils.xgb_feature_pipeline import segment_and_extract_cov_erd


@dataclass
class SessionData:
    session_id: int
    session_name: str
    labels: np.ndarray
    cov_mu: np.ndarray
    cov_beta: np.ndarray | None
    cov_mu_tangent: np.ndarray
    cov_beta_tangent: np.ndarray | None
    erd: np.ndarray | None


def _cov_tangent_features(cov_matrices: np.ndarray) -> np.ndarray:
    n_ch = cov_matrices.shape[1]
    return tangent_space(cov_matrices, np.eye(n_ch), metric="riemann")


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


def discover_training_files() -> List[str]:
    eeg_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "training_data")
    xdf_files = sorted(
        [
            os.path.join(eeg_dir, f)
            for f in os.listdir(eeg_dir)
            if f.endswith(".xdf") and "OBS" not in f
        ]
    )
    if not xdf_files:
        raise FileNotFoundError(f"No XDF files found in: {eeg_dir}")
    return xdf_files


def build_session_dataset(
    xdf_files: List[str] | None = None,
    *,
    include_beta_cov: bool = True,
    include_erd: bool = True,
    apply_csd: bool | None = None,
) -> List[SessionData]:
    if xdf_files is None:
        xdf_files = discover_training_files()
    if apply_csd is None:
        apply_csd = bool(getattr(config, "SURFACE_LAPLACIAN_TOGGLE", False))

    sessions: List[SessionData] = []
    for sess_idx, xdf_path in enumerate(xdf_files):
        eeg_stream, marker_stream = load_xdf(xdf_path, report=False)
        out = segment_and_extract_cov_erd(
            eeg_stream,
            marker_stream,
            compute_erd=include_erd,
            apply_csd=apply_csd,
            return_beta_segments=include_beta_cov,
        )
        if include_beta_cov:
            segments, labels, erd, beta_segments, _channel_names = out
        else:
            segments, labels, erd, _channel_names = out
            beta_segments = None

        if len(labels) == 0:
            continue

        cov_mu = base.compute_processed_covariances(segments, labels, model_type="xgb")
        cov_mu_tangent = _cov_tangent_features(cov_mu)

        cov_beta = None
        cov_beta_tangent = None
        if include_beta_cov and beta_segments is not None:
            cov_beta = base.compute_processed_covariances(beta_segments, labels, model_type="xgb")
            cov_beta_tangent = _cov_tangent_features(cov_beta)

        sessions.append(
            SessionData(
                session_id=sess_idx,
                session_name=os.path.basename(xdf_path),
                labels=np.asarray(labels),
                cov_mu=np.asarray(cov_mu),
                cov_beta=None if cov_beta is None else np.asarray(cov_beta),
                cov_mu_tangent=np.asarray(cov_mu_tangent),
                cov_beta_tangent=None if cov_beta_tangent is None else np.asarray(cov_beta_tangent),
                erd=None if erd is None else np.asarray(erd),
            )
        )

    if len(sessions) < 2:
        raise RuntimeError("Need at least 2 valid sessions for held-out transfer benchmark.")
    return sessions


class ModelAdapter:
    name: str = "base"
    supports_importance: bool = False

    def fit(self, train_sessions: List[SessionData]) -> None:
        raise NotImplementedError

    def predict_proba(self, test_session: SessionData) -> np.ndarray:
        raise NotImplementedError

    def get_importance(self) -> np.ndarray | None:
        return None

    def get_feature_names(self) -> List[str] | None:
        return None


class MDMAdapter(ModelAdapter):
    name = "mdm_cov_mu"
    supports_importance = False

    def __init__(self, rest_label: int, mi_label: int):
        self.rest_label = rest_label
        self.mi_label = mi_label
        self.model = MDM()

    def fit(self, train_sessions: List[SessionData]) -> None:
        X = np.concatenate([s.cov_mu for s in train_sessions], axis=0)
        y = np.concatenate([s.labels for s in train_sessions], axis=0)
        self.model.fit(X, y)

    def predict_proba(self, test_session: SessionData) -> np.ndarray:
        p = self.model.predict_proba(test_session.cov_mu)
        rest_idx = int(np.where(self.model.classes_ == self.rest_label)[0][0])
        mi_idx = int(np.where(self.model.classes_ == self.mi_label)[0][0])
        return np.column_stack([p[:, rest_idx], p[:, mi_idx]])


class XGBCovAdapter(ModelAdapter):
    name = "xgb_cov"
    supports_importance = True

    def __init__(self, rest_label: int, mi_label: int, use_cov_mu: bool = True, use_cov_beta: bool = True):
        self.rest_label = rest_label
        self.mi_label = mi_label
        self.use_cov_mu = bool(use_cov_mu)
        self.use_cov_beta = bool(use_cov_beta)
        self.scaler = StandardScaler()
        self.model = None
        self._feature_names: List[str] = []

    def _assemble(self, s: SessionData) -> np.ndarray:
        blocks = []
        if self.use_cov_mu:
            blocks.append(s.cov_mu_tangent)
        if self.use_cov_beta:
            if s.cov_beta_tangent is None:
                raise RuntimeError("beta tangent features missing")
            blocks.append(s.cov_beta_tangent)
        return np.hstack(blocks)

    def fit(self, train_sessions: List[SessionData]) -> None:
        from xgboost import XGBClassifier

        X = np.concatenate([self._assemble(s) for s in train_sessions], axis=0)
        y = np.concatenate([s.labels for s in train_sessions], axis=0)
        y_bin = np.where(y == self.mi_label, 1, 0).astype(int)
        Xs = self.scaler.fit_transform(X)
        self.model = XGBClassifier(**_xgb_params())
        self.model.fit(Xs, y_bin)

        n_cov_mu = train_sessions[0].cov_mu_tangent.shape[1] if self.use_cov_mu else 0
        n_cov_beta = (
            train_sessions[0].cov_beta_tangent.shape[1]
            if self.use_cov_beta and train_sessions[0].cov_beta_tangent is not None
            else 0
        )
        self._feature_names = [f"cov_mu_{i}" for i in range(n_cov_mu)] + [f"cov_beta_{i}" for i in range(n_cov_beta)]

    def predict_proba(self, test_session: SessionData) -> np.ndarray:
        X = self._assemble(test_session)
        Xs = self.scaler.transform(X)
        p = self.model.predict_proba(Xs)
        return np.column_stack([p[:, 0], p[:, 1]])

    def get_importance(self) -> np.ndarray | None:
        return np.asarray(self.model.feature_importances_, dtype=float) if self.model is not None else None

    def get_feature_names(self) -> List[str] | None:
        return self._feature_names


class XGBCovErdAdapter(ModelAdapter):
    name = "xgb_cov_erd"
    supports_importance = True

    def __init__(self, rest_label: int, mi_label: int, use_cov_mu: bool = True, use_cov_beta: bool = True):
        self.rest_label = rest_label
        self.mi_label = mi_label
        self.use_cov_mu = bool(use_cov_mu)
        self.use_cov_beta = bool(use_cov_beta)
        self.scaler = StandardScaler()
        self.model = None
        self._feature_names: List[str] = []

    def _assemble(self, s: SessionData) -> np.ndarray:
        blocks = []
        if self.use_cov_mu:
            blocks.append(s.cov_mu_tangent)
        if self.use_cov_beta:
            if s.cov_beta_tangent is None:
                raise RuntimeError("beta tangent features missing")
            blocks.append(s.cov_beta_tangent)
        if s.erd is None:
            raise RuntimeError("ERD features missing for cov+ERD adapter")
        blocks.append(s.erd)
        return np.hstack(blocks)

    def fit(self, train_sessions: List[SessionData]) -> None:
        from xgboost import XGBClassifier

        X = np.concatenate([self._assemble(s) for s in train_sessions], axis=0)
        y = np.concatenate([s.labels for s in train_sessions], axis=0)
        y_bin = np.where(y == self.mi_label, 1, 0).astype(int)
        Xs = self.scaler.fit_transform(X)
        self.model = XGBClassifier(**_xgb_params())
        self.model.fit(Xs, y_bin)

        n_cov_mu = train_sessions[0].cov_mu_tangent.shape[1] if self.use_cov_mu else 0
        n_cov_beta = (
            train_sessions[0].cov_beta_tangent.shape[1]
            if self.use_cov_beta and train_sessions[0].cov_beta_tangent is not None
            else 0
        )
        n_erd = train_sessions[0].erd.shape[1] if train_sessions[0].erd is not None else 0
        self._feature_names = (
            [f"cov_mu_{i}" for i in range(n_cov_mu)]
            + [f"cov_beta_{i}" for i in range(n_cov_beta)]
            + [f"erd_{i}" for i in range(n_erd)]
        )

    def predict_proba(self, test_session: SessionData) -> np.ndarray:
        X = self._assemble(test_session)
        Xs = self.scaler.transform(X)
        p = self.model.predict_proba(Xs)
        return np.column_stack([p[:, 0], p[:, 1]])

    def get_importance(self) -> np.ndarray | None:
        return np.asarray(self.model.feature_importances_, dtype=float) if self.model is not None else None

    def get_feature_names(self) -> List[str] | None:
        return self._feature_names


def make_adapter(model_name: str, rest_label: int, mi_label: int) -> ModelAdapter:
    name = model_name.lower()
    use_cov_mu = bool(getattr(config, "XGB_USE_COV_MU", 1))
    use_cov_beta = bool(getattr(config, "XGB_USE_COV_BETA", 1))
    if name == "mdm":
        return MDMAdapter(rest_label=rest_label, mi_label=mi_label)
    if name == "xgb_cov":
        return XGBCovAdapter(rest_label=rest_label, mi_label=mi_label, use_cov_mu=use_cov_mu, use_cov_beta=use_cov_beta)
    if name == "xgb_cov_erd":
        return XGBCovErdAdapter(rest_label=rest_label, mi_label=mi_label, use_cov_mu=use_cov_mu, use_cov_beta=use_cov_beta)
    raise ValueError(f"Unknown model_name '{model_name}'")


def _summarize_importance(imp_rows: List[np.ndarray], feature_names: List[str], top_k: int) -> None:
    if not imp_rows:
        return
    M = np.vstack(imp_rows)
    mean_imp = M.mean(axis=0)
    std_imp = M.std(axis=0)
    top_hit = np.zeros(M.shape[1], dtype=int)
    for row in M:
        idxs = np.argsort(row)[::-1][: min(top_k, row.shape[0])]
        top_hit[idxs] += 1
    order = np.argsort(mean_imp)[::-1]
    k = min(top_k, len(order))
    print(f"\nTop {k} stable features (mean importance, top-hit count):")
    for rank, idx in enumerate(order[:k], 1):
        print(
            f"{rank:>2}. {feature_names[idx]:<25} "
            f"mean={mean_imp[idx]:.6f} std={std_imp[idx]:.6f} top_hits={int(top_hit[idx])}/{M.shape[0]}"
        )


def run_session_heldout_benchmark(
    model_names: List[str],
    sessions: List[SessionData],
    *,
    seed: int = 42,
    print_importance: bool = True,
) -> Dict[str, dict]:
    del seed  # deterministic by fixed session ordering
    labels_all = np.concatenate([s.labels for s in sessions], axis=0)
    classes = np.sort(np.unique(labels_all))
    if len(classes) != 2:
        raise ValueError("Binary labels required.")
    rest_label, mi_label = int(classes[0]), int(classes[1])
    top_k = int(getattr(config, "XGB_IMPORTANCE_TOP_K", 30))

    results: Dict[str, dict] = {}
    for model_name in model_names:
        adapter = make_adapter(model_name, rest_label=rest_label, mi_label=mi_label)
        print(f"\n===== Benchmark: {adapter.name} =====")
        fold_rows = []
        imp_rows = []
        feature_names = None

        all_true = []
        all_prob_mi = []
        all_pred = []

        for fold_idx, test_session in enumerate(sessions, 1):
            train_sessions = [s for s in sessions if s.session_id != test_session.session_id]
            adapter.fit(train_sessions)
            probs = adapter.predict_proba(test_session)  # [p_rest, p_mi]
            y_true = test_session.labels
            y_pred = np.where(probs[:, 1] >= probs[:, 0], mi_label, rest_label)

            y_true_bin = (y_true == mi_label).astype(int)
            auc = float(roc_auc_score(y_true_bin, probs[:, 1])) if np.unique(y_true_bin).size == 2 else np.nan
            acc = float(accuracy_score(y_true, y_pred))
            bal_acc = float(balanced_accuracy_score(y_true, y_pred))
            macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
            cm = confusion_matrix(y_true, y_pred, labels=[rest_label, mi_label])
            brier = float(brier_score_loss(y_true_bin, probs[:, 1]))

            fold_rows.append(
                dict(
                    fold=fold_idx,
                    session_id=test_session.session_id,
                    session_name=test_session.session_name,
                    n_test=int(len(y_true)),
                    acc=acc,
                    auc=auc,
                    bal_acc=bal_acc,
                    macro_f1=macro_f1,
                    brier=brier,
                    cm=cm,
                )
            )

            all_true.extend(y_true.tolist())
            all_prob_mi.extend(probs[:, 1].tolist())
            all_pred.extend(y_pred.tolist())

            if adapter.supports_importance and print_importance:
                imp = adapter.get_importance()
                names = adapter.get_feature_names()
                if imp is not None and names is not None and len(imp) == len(names):
                    imp_rows.append(imp)
                    feature_names = names

            print(
                f"Fold {fold_idx} session={test_session.session_name} n={len(y_true)} "
                f"acc={acc:.4f} auc={auc:.4f} bal_acc={bal_acc:.4f} macro_f1={macro_f1:.4f} brier={brier:.4f}"
            )

        all_true = np.asarray(all_true)
        all_pred = np.asarray(all_pred)
        all_prob_mi = np.asarray(all_prob_mi)
        all_true_bin = (all_true == mi_label).astype(int)
        agg_auc = float(roc_auc_score(all_true_bin, all_prob_mi)) if np.unique(all_true_bin).size == 2 else np.nan
        agg_acc = float(accuracy_score(all_true, all_pred))
        agg_bal = float(balanced_accuracy_score(all_true, all_pred))
        agg_f1 = float(f1_score(all_true, all_pred, average="macro"))
        agg_brier = float(brier_score_loss(all_true_bin, all_prob_mi))
        agg_cm = confusion_matrix(all_true, all_pred, labels=[rest_label, mi_label])

        print("\nAggregated:")
        print(
            f"acc={agg_acc:.4f} auc={agg_auc:.4f} bal_acc={agg_bal:.4f} "
            f"macro_f1={agg_f1:.4f} brier={agg_brier:.4f}"
        )
        print("confusion [rows=true REST,MI | cols=pred REST,MI]:")
        print(agg_cm)

        if adapter.supports_importance and print_importance and feature_names is not None:
            _summarize_importance(imp_rows, feature_names, top_k=top_k)

        results[model_name] = {
            "folds": fold_rows,
            "aggregate": {
                "acc": agg_acc,
                "auc": agg_auc,
                "bal_acc": agg_bal,
                "macro_f1": agg_f1,
                "brier": agg_brier,
                "cm": agg_cm,
            },
        }

    print("\n===== Cross-model Summary =====")
    for model_name in model_names:
        a = results[model_name]["aggregate"]
        print(
            f"{model_name:<12} acc={a['acc']:.4f} auc={a['auc']:.4f} "
            f"bal_acc={a['bal_acc']:.4f} macro_f1={a['macro_f1']:.4f} brier={a['brier']:.4f}"
        )
    return results

