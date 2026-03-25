"""
XGBoost branch using covariance-derived features only.

Backbone (preserved): EEG -> temporal filtering -> segmentation -> covariance ->
trace normalization -> shrinkage -> whitening (via Generate_Riemannian_adaptive).

Covariance feature formulation:
  cov_mats (whitened SPD) -> tangent_space at fitted reference -> tabular features
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

os.environ["NUMBA_DISABLE_CACHING"] = "1"
os.environ["MNE_USE_NUMBA"] = "false"

import mne

from pyriemann.tangentspace import TangentSpace, tangent_space

import config
import Generate_Riemannian_adaptive as base
from Utils.stream_utils import load_xdf
from Utils.xgb_feature_pipeline import segment_and_extract_cov_erd
from Utils.xgb_train_eval import train_xgb_dual_thresholds


def _fit_tangent_ref(cov_matrices: np.ndarray) -> np.ndarray:
    ts = TangentSpace(metric="riemann")
    ts.fit(cov_matrices)
    return ts.reference_


def _cov_tangent_features(cov_matrices: np.ndarray, tangent_ref: np.ndarray) -> np.ndarray:
    return tangent_space(cov_matrices, tangent_ref, metric="riemann")


def _report_xgb_importance_cov_only(
    model_bundle: dict,
    n_cov_mu: int,
    n_cov_beta: int,
    channel_names: list[str] | None = None,
    top_k: int = 20,
):
    """
    Lightweight explainability using built-in XGBoost feature importances
    for cov-only branch.
    """
    model = model_bundle["model"]
    imp = np.asarray(model.feature_importances_, dtype=float)
    names = [f"cov_mu_{i}" for i in range(n_cov_mu)] + [f"cov_beta_{i}" for i in range(n_cov_beta)]

    if imp.shape[0] != len(names):
        print(
            f"⚠️ Importance length mismatch: importances={imp.shape[0]} names={len(names)} "
            "(skipping named importance report)."
        )
        return

    total = float(np.sum(imp)) + 1e-12
    g_cov_mu = float(np.sum(imp[:n_cov_mu])) / total if n_cov_mu else 0.0
    g_cov_beta = float(np.sum(imp[n_cov_mu:])) / total if n_cov_beta else 0.0

    print("\n====== XGBoost Built-in Importance (gain-based, cov-only) ======")
    print(f"Group contribution: cov_mu={g_cov_mu*100:.2f}% | cov_beta={g_cov_beta*100:.2f}%")

    order = np.argsort(imp)[::-1]
    top_k = min(top_k, len(order))
    def _infer_n_ch_from_tangent_dim(d: int) -> int:
        n = int((np.sqrt(1 + 8 * d) - 1) / 2)
        return n

    def _cov_idx_to_pair(idx: int, cov_dim: int) -> tuple[int, int]:
        n_ch = _infer_n_ch_from_tangent_dim(cov_dim)
        tri_i, tri_j = np.triu_indices(n_ch)
        if idx < 0 or idx >= len(tri_i):
            return -1, -1
        return int(tri_i[idx]), int(tri_j[idx])

    # Build decoded names used for both terminal output and graph labels.
    decoded_names = list(names)
    for idx, feat_name in enumerate(names):
        if feat_name.startswith("cov_mu_"):
            local_idx = int(feat_name.split("_")[-1])
            i, j = _cov_idx_to_pair(local_idx, n_cov_mu)
            kind = "diag" if i == j else "offdiag"
            if channel_names and i >= 0 and j >= 0 and i < len(channel_names) and j < len(channel_names):
                decoded_names[idx] = f"cov_mu_{local_idx} [{kind}; {channel_names[i]}-{channel_names[j]}]"
            else:
                decoded_names[idx] = f"cov_mu_{local_idx} [{kind}; ch{i}-ch{j}]"
        elif feat_name.startswith("cov_beta_"):
            local_idx = int(feat_name.split("_")[-1])
            i, j = _cov_idx_to_pair(local_idx, n_cov_beta)
            kind = "diag" if i == j else "offdiag"
            if channel_names and i >= 0 and j >= 0 and i < len(channel_names) and j < len(channel_names):
                decoded_names[idx] = f"cov_beta_{local_idx} [{kind}; {channel_names[i]}-{channel_names[j]}]"
            else:
                decoded_names[idx] = f"cov_beta_{local_idx} [{kind}; ch{i}-ch{j}]"

    print(f"\nTop {top_k} features by model.feature_importances_:")
    for r, idx in enumerate(order[:top_k], 1):
        print(f"{r:>2}. {decoded_names[idx]:<45} {imp[idx]:.6f}")

    idxs = order[:top_k][::-1]
    vals = imp[idxs]
    lbls = [decoded_names[i] for i in idxs]
    plt.figure(figsize=(9, 6))
    plt.barh(range(len(idxs)), vals)
    plt.yticks(range(len(idxs)), lbls)
    plt.xlabel("XGBoost feature_importances_")
    plt.title(f"Top {top_k} Feature Importances (cov-only)")
    plt.tight_layout()
    plt.show()


def main():
    mne.set_log_level("WARNING")

    eeg_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "training_data")
    xdf_files = [
        os.path.join(eeg_dir, f) for f in os.listdir(eeg_dir)
        if f.endswith(".xdf") and "OBS" not in f
    ]
    if not xdf_files:
        raise FileNotFoundError(f"No XDF files found in: {eeg_dir}")

    all_labels = []
    cov_mu_all = []
    cov_beta_all = []
    channel_names = list(getattr(config, "MOTOR_CHANNEL_NAMES", [])) if getattr(config, "SELECT_MOTOR_CHANNELS", 0) else None

    use_cov_mu = bool(getattr(config, "XGB_USE_COV_MU", 1))
    use_cov_beta = bool(getattr(config, "XGB_USE_COV_BETA", 0))
    if not (use_cov_mu or use_cov_beta):
        raise ValueError("At least one of XGB_USE_COV_MU or XGB_USE_COV_BETA must be enabled.")
    print(f"[cov-only] covariance feature sets -> mu={use_cov_mu}, beta={use_cov_beta}")

    n_cov_mu = n_cov_beta = 0

    for xdf_path in xdf_files:
        print(f"\n📂 Processing file: {xdf_path}")
        eeg_stream, marker_stream = load_xdf(xdf_path, report=False)

        out = segment_and_extract_cov_erd(
            eeg_stream,
            marker_stream,
            compute_erd=False,
            apply_csd=False,  # CSD is ERD-only; cov windows are always non-CSD (see xgb_feature_pipeline)
            return_beta_segments=use_cov_beta,
        )
        if use_cov_beta:
            segments, labels, _, beta_segments, _ch_names = out
        else:
            segments, labels, _, _ch_names = out
            beta_segments = None

        if use_cov_mu:
            cov_matrices_mu = base.compute_processed_covariances(segments, labels, model_type="xgb")
            cov_mu_all.append(cov_matrices_mu)

        if use_cov_beta:
            if beta_segments is None:
                raise RuntimeError("Expected beta segments but got None.")
            cov_matrices_beta = base.compute_processed_covariances(beta_segments, labels, model_type="xgb")
            cov_beta_all.append(cov_matrices_beta)
        all_labels.append(labels)

    y = np.concatenate(all_labels)
    feature_blocks = []
    tangent_ref_mu = None
    tangent_ref_beta = None

    if use_cov_mu:
        cov_mu = np.concatenate(cov_mu_all, axis=0)
        tangent_ref_mu = _fit_tangent_ref(cov_mu)
        cov_feats_mu = _cov_tangent_features(cov_mu, tangent_ref_mu)
        n_cov_mu = int(cov_feats_mu.shape[1])
        feature_blocks.append(cov_feats_mu)

    if use_cov_beta:
        cov_beta = np.concatenate(cov_beta_all, axis=0)
        tangent_ref_beta = _fit_tangent_ref(cov_beta)
        cov_feats_beta = _cov_tangent_features(cov_beta, tangent_ref_beta)
        n_cov_beta = int(cov_feats_beta.shape[1])
        feature_blocks.append(cov_feats_beta)

    X = np.hstack(feature_blocks)
    print(
        f"Feature dimensions: cov_mu={int(n_cov_mu or 0)}, cov_beta={int(n_cov_beta or 0)}, "
        f"total={X.shape[1]}"
    )

    model_bundle = train_xgb_dual_thresholds(
        X=X,
        labels=y,
        feature_tag="cov_tangent_fittedref",
        n_splits=int(getattr(config, "N_SPLITS", 8)),
        target_ambig=float(getattr(config, "TARGET_AMBIG", 0.20)),
    )
    model_bundle["decoder_backend"] = "xgb_cov"
    model_bundle["tangent_ref_mu"] = tangent_ref_mu
    model_bundle["tangent_ref_beta"] = tangent_ref_beta
    model_bundle["feature_spec"] = {
        "type": "cov_only",
        "use_cov_mu": bool(use_cov_mu),
        "use_cov_beta": bool(use_cov_beta),
        "n_cov_mu": int(n_cov_mu),
        "n_cov_beta": int(n_cov_beta),
        "channel_names": list(channel_names) if channel_names is not None else None,
    }
    _report_xgb_importance_cov_only(
        model_bundle=model_bundle,
        n_cov_mu=int(n_cov_mu or 0),
        n_cov_beta=int(n_cov_beta or 0),
        channel_names=channel_names,
        top_k=int(getattr(config, "XGB_IMPORTANCE_TOP_K", 20)),
    )

    subject_model_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "models")
    os.makedirs(subject_model_dir, exist_ok=True)
    out_path = os.path.join(subject_model_dir, f"sub-{config.TRAINING_SUBJECT}_xgb_cov_features.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"✅ Saved covariance-only XGBoost model to: {out_path}")


if __name__ == "__main__":
    main()

