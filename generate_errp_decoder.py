"""
ErrP Decoder Training Pipeline
================================
Trains an xDAWN + Riemannian decoder for Error-Related Potentials (ErrP).

Decoder architecture (state-of-the-art for ERP-based BCIs):
  Raw EEG → 1-10 Hz bandpass → epoch [tmin, tmax] post-event
    → XdawnCovariances (fit xDAWN spatial filters + augmented covariance)
    → MDM  (Minimum Distance to Mean on Riemannian manifold)    [xdawn_mdm]
    → OR TangentSpace + LogisticRegression                       [xdawn_lr]

Why this approach:
  • xDAWN (Rivet et al. 2009) is designed specifically for ERP paradigms.
    It learns spatial filters that maximise the ERP SNR without looking at test data.
  • Riemannian geometry (Barachant & Congedo 2014) applied to xDAWN-augmented
    covariance matrices is the most transfer-robust EEG decoder known —
    naturally invariant to inter-session amplitude shifts (no normalisation needed).
  • Both components are in pyriemann and are already used by the Harmony MI pipeline.
  • xDAWN is a deterministic linear filter once fitted → trivially causal online.

Existing data:
  Uses ROBOT_EARLYSTOP (340) as error events and ROBOT_END (320) as correct events.
  These are already present in most Harmony XDF recordings.

New dedicated experiment data:
  Uses ERRP_STIM_ERROR (430) and ERRP_STIM_CORRECT (440) from ExperimentDriver_ErrP.

Usage:
    python generate_errp_decoder.py
    # Edit config.py to change subject, paths, and decoder params.
"""

import os
import sys
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["NUMBA_DISABLE_CACHING"] = "1"
os.environ["MNE_USE_NUMBA"] = "false"

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pyriemann.estimation import XdawnCovariances
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace

import config
import Generate_Riemannian_adaptive as base
from Utils.errp_feature_pipeline import (
    load_errp_training_data,
    _default_error_codes,
    _default_correct_codes,
)


# ---------------------------------------------------------------------------
# Parameters (pulled from config with safe defaults)
# ---------------------------------------------------------------------------
N_FILTERS      = int(getattr(config, "ERRP_XDAWN_N_FILTERS", 4))
N_SPLITS       = int(getattr(config, "N_SPLITS", 5))
TARGET_AMBIG   = float(getattr(config, "ERRP_TARGET_AMBIG", 0.20))
BACKEND        = str(getattr(config, "ERRP_DECODER_BACKEND", "xdawn_mdm")).lower()
TMIN           = float(getattr(config, "ERRP_EPOCH_TMIN", 0.0))
TMAX           = float(getattr(config, "ERRP_EPOCH_TMAX", 0.8))
ARTIFACT_UV    = float(getattr(config, "ERRP_ARTIFACT_MAX_ABS_UV", 80.0))

ERROR_CODES    = _default_error_codes()
CORRECT_CODES  = _default_correct_codes()

ERROR_LABEL    = 1   # binary: 1 = error (ErrP present)
CORRECT_LABEL  = 0   # binary: 0 = correct (no ErrP)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_classifier(backend: str):
    """Return an unfitted classifier that operates on augmented covariance matrices."""
    if backend == "xdawn_mdm":
        return MDM(metric="riemann")
    if backend == "xdawn_lr":
        return Pipeline([
            ("ts", TangentSpace(metric="riemann")),
            ("lr", LogisticRegression(
                C=float(getattr(config, "ERRP_LR_C", 1.0)),
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ])
    raise ValueError(f"Unknown ERRP_DECODER_BACKEND: {backend!r}. Choose 'xdawn_mdm' or 'xdawn_lr'.")


def _fit_xdawn_covs(X_epochs: np.ndarray, y: np.ndarray, n_filters: int) -> XdawnCovariances:
    """Fit XdawnCovariances on training epochs."""
    xdc = XdawnCovariances(nfilter=n_filters, estimator="lwf", xdawn_weights=None)
    xdc.fit(X_epochs, y)
    return xdc


# ---------------------------------------------------------------------------
# Cross-validation + dual-threshold evaluation
# ---------------------------------------------------------------------------

def train_errp_cv(
    X: np.ndarray,
    y: np.ndarray,
    backend: str = "xdawn_mdm",
    n_splits: int = 5,
    target_ambig: float = 0.20,
    n_filters: int = 4,
):
    """
    Stratified K-fold cross-validation for xDAWN + MDM/LR with dual-threshold selection.

    Returns:
        dict with model bundle (ready to save), CV metrics, and threshold parameters.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    acc_argmax, t_lows, t_highs = [], [], []
    all_true_bin, all_scores = [], []
    all_true_label, all_pred_label = [], []

    print(f"\n🧠 ErrP Decoder Training ({backend}, xDAWN n_filters={n_filters})")
    print(f"   Epochs: {X.shape[0]} total  |  error={int(y.sum())}  correct={int((y==0).sum())}")
    print(f"   Epoch shape: {X.shape[1]} ch × {X.shape[2]} samples ({X.shape[2]/config.FS*1000:.0f} ms)")
    print(f"   CV: {n_splits}-fold stratified  |  target_ambig={target_ambig:.2f}\n")

    for fold_idx, (tr, te) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        # 1. Fit xDAWN on training fold
        xdc = _fit_xdawn_covs(X_tr, y_tr, n_filters)

        # 2. Transform to augmented covariance matrices
        C_tr = xdc.transform(X_tr)   # (n_tr, aug_size, aug_size)
        C_te = xdc.transform(X_te)

        # 3. Fit classifier
        clf = _build_classifier(backend)
        clf.fit(C_tr, y_tr)

        # 4. Probabilities (column 1 = P(error))
        prob_tr = clf.predict_proba(C_tr)[:, 1]
        prob_te = clf.predict_proba(C_te)[:, 1]

        # 5. Argmax accuracy
        y_pred_te = (prob_te >= 0.5).astype(int)
        acc = accuracy_score(y_te, y_pred_te)
        acc_argmax.append(acc)
        print(f"  Fold {fold_idx}: argmax acc={acc:.4f}")

        # 6. Dual-threshold selection on training scores
        tl, th, diag = base.pick_dual_thresholds_target_ambiguity(
            y_true_bin=y_tr,
            pos_scores=prob_tr,
            target_ambig=target_ambig,
            c_fp=1.0, c_fn=1.0,
            n_grid=201, min_gap=0.0,
            tpr_min=None, fpr_max=None, ppv_min=None, npv_min=None,
            require_center_around_half=False,
        )
        t_lows.append(tl)
        t_highs.append(th)
        print(f"         thresholds: t_low={tl:.3f}  t_high={th:.3f}  feasible={diag.get('feasible', False)}")

        all_true_bin.extend(y_te.tolist())
        all_scores.extend(prob_te.tolist())

        # Dual-threshold predictions (-1 = ambiguous)
        pred = np.full_like(y_te, -1, dtype=int)
        pred[prob_te >= th] = 1   # error
        pred[prob_te <= tl] = 0   # correct
        all_true_label.extend(y_te.tolist())
        all_pred_label.extend(pred.tolist())

    # --- Aggregate metrics ---
    all_true_bin   = np.array(all_true_bin)
    all_scores     = np.array(all_scores)
    all_true_label = np.array(all_true_label)
    all_pred_label = np.array(all_pred_label)

    roc_auc = float(roc_auc_score(all_true_bin, all_scores)) if np.unique(all_true_bin).size == 2 else float("nan")

    decided  = all_pred_label != -1
    coverage = float(decided.mean())
    decided_acc = float(
        accuracy_score(all_true_label[decided], all_pred_label[decided])
    ) if decided.any() else float("nan")
    ambig_frac = 1.0 - coverage

    tl_star = float(np.median(t_lows))
    th_star = float(np.median(t_highs))

    cm = confusion_matrix(
        all_true_label[decided], all_pred_label[decided], labels=[0, 1]
    ) if decided.any() else np.zeros((2, 2), dtype=int)

    print(f"\n====== ErrP CV Results ({backend}) ======")
    print(f"  Mean argmax accuracy:  {np.mean(acc_argmax):.4f}")
    print(f"  ROC AUC:               {roc_auc:.4f}")
    print(f"  Decided-only accuracy: {decided_acc:.4f}  (coverage={coverage*100:.1f}%)")
    print(f"  Ambiguity fraction:    {ambig_frac*100:.1f}%  (target={target_ambig*100:.1f}%)")
    print(f"  Median thresholds:     t_low={tl_star:.3f}  t_high={th_star:.3f}")
    print(f"  Confusion (decided-only; rows=true [correct,error], cols=pred [correct,error]):")
    print(f"    {cm}")

    # --- Plots ---
    _plot_errp_cv_results(all_scores, all_true_bin, tl_star, th_star, backend)

    # --- Final model (full data) ---
    print(f"\n🔧 Fitting final {backend} model on full dataset ({X.shape[0]} epochs)...")
    xdc_final = _fit_xdawn_covs(X, y, n_filters)
    C_final   = xdc_final.transform(X)
    clf_final = _build_classifier(backend)
    clf_final.fit(C_final, y)

    epoch_samples = X.shape[2]
    bundle = {
        "xdawn":       xdc_final,
        "classifier":  clf_final,
        "backend":     backend,
        "tl_star":     tl_star,
        "th_star":     th_star,
        "roc_auc":     roc_auc,
        "feature_spec": {
            "tmin":          TMIN,
            "tmax":          TMAX,
            "n_filters":     n_filters,
            "epoch_samples": epoch_samples,
            "error_codes":   ERROR_CODES,
            "correct_codes": CORRECT_CODES,
        },
        # Keep binary convention: 1 = error, 0 = correct
        "label_to_bin":  {1: 1, 0: 0},
        "bin_to_label":  {1: 1, 0: 0},
        "cv_mean_acc":   float(np.mean(acc_argmax)),
    }

    return bundle


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def _plot_errp_cv_results(scores, y_true_bin, tl, th, backend: str):
    """Reuse base plotting utilities from Generate_Riemannian_adaptive."""
    try:
        base._plot_scores_hist_with_thresholds(scores, y_true_bin, tl, th)
        plt.suptitle(f"ErrP Score Distribution ({backend})", fontsize=13)

        center = (tl + th) / 2.0
        widths = np.linspace(0.0, 0.9, 35)
        base._plot_risk_coverage(scores, y_true_bin, center, widths, 1.0, 1.0, 0.3)
        plt.suptitle(f"ErrP Risk-Coverage Curve ({backend})", fontsize=13)

        base._plot_roc_with_point(scores, y_true_bin, th)
        plt.suptitle(f"ErrP ROC Curve ({backend})", fontsize=13)

        plt.show()
    except Exception as exc:
        print(f"[ErrP plots] Non-fatal plotting error: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # --- Locate XDF files ---
    subject = config.TRAINING_SUBJECT
    training_dir = os.path.join(config.DATA_DIR, f"sub-{subject}", "training_data")
    xdf_files = sorted(glob.glob(os.path.join(training_dir, "**", "*.xdf"), recursive=True))
    xdf_files = [f for f in xdf_files if "OBS" not in os.path.basename(f)]

    if not xdf_files:
        raise FileNotFoundError(
            f"No XDF files found in: {training_dir}\n"
            f"Ensure LabRecorder has saved files there for subject '{subject}'."
        )

    print(f"[ErrP] Subject: {subject}")
    print(f"[ErrP] Data dir: {training_dir}")
    print(f"[ErrP] XDF files found: {len(xdf_files)}")
    for f in xdf_files:
        print(f"         {os.path.basename(f)}")
    print(f"[ErrP] Error codes:   {ERROR_CODES}  (ROBOT_EARLYSTOP=340, ERRP_STIM_ERROR=430)")
    print(f"[ErrP] Correct codes: {CORRECT_CODES} (ROBOT_END=320, ERRP_STIM_CORRECT=440)")
    print(f"[ErrP] Epoch window:  {TMIN*1000:.0f} – {TMAX*1000:.0f} ms post-event")
    print(f"[ErrP] Artifact thresh: {ARTIFACT_UV} µV (max_abs)")

    # --- Load data ---
    X, y, ch_names = load_errp_training_data(
        xdf_files,
        error_codes=ERROR_CODES,
        correct_codes=CORRECT_CODES,
        tmin=TMIN,
        tmax=TMAX,
        artifact_thresh_uv=ARTIFACT_UV,
        verbose=True,
    )

    if X.shape[0] < 10:
        raise ValueError(
            f"Too few epochs ({X.shape[0]}) for reliable training. "
            f"Need at least 10 (ideally 40+). "
            f"Check that XDF files contain ROBOT_EARLYSTOP (340) or ERRP_STIM_ERROR (430) markers."
        )

    print(f"\n[ErrP] Channels ({len(ch_names)}): {ch_names}")

    # --- Cross-validate + train both backends ---
    backends_to_run = ["xdawn_mdm", "xdawn_lr"]
    bundles = {}
    for backend in backends_to_run:
        print(f"\n{'='*60}")
        print(f"  Backend: {backend}")
        print(f"{'='*60}")
        bundles[backend] = train_errp_cv(
            X, y,
            backend=backend,
            n_splits=N_SPLITS,
            target_ambig=TARGET_AMBIG,
            n_filters=N_FILTERS,
        )

    # --- Report comparison ---
    print(f"\n{'='*60}")
    print("  Backend Comparison")
    print(f"{'='*60}")
    for backend, bundle in bundles.items():
        print(f"  {backend:20s}  ROC-AUC={bundle['roc_auc']:.4f}  mean_acc={bundle['cv_mean_acc']:.4f}"
              f"  thresholds=[{bundle['tl_star']:.3f}, {bundle['th_star']:.3f}]")

    # --- Save selected backend ---
    selected_bundle = bundles[BACKEND]
    model_dir = os.path.join(config.DATA_DIR, f"sub-{subject}", "models")
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f"sub-{subject}_errp_{BACKEND}.pkl")

    with open(save_path, "wb") as f:
        pickle.dump(selected_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n✅ ErrP model saved: {save_path}")
    print(f"   Backend:      {selected_bundle['backend']}")
    print(f"   ROC AUC:      {selected_bundle['roc_auc']:.4f}")
    print(f"   Thresholds:   t_low={selected_bundle['tl_star']:.3f}  t_high={selected_bundle['th_star']:.3f}")
    print(f"   Channels:     {ch_names}")
    print(f"\n   To use online: set ERRP_DECODER_ENABLE = 1 in config.py")


if __name__ == "__main__":
    main()
