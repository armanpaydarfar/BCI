"""
generate_rbnnet_model.py
------------------------
Offline training script for the RBNNet Motor Imagery decoder.

RBNNet is a geometry-aware deep network on the SPD manifold (Liu et al.,
NER 2023).  This script follows the same conventions as
Generate_Riemannian_adaptive.py and generate_xgboost_cov_features.py:

  1. Load XDF recording(s) from config.DATA_DIR / config.TRAINING_SUBJECT
  2. Preprocess + segment (reusing existing pipeline)
  3. Compute shrinkage-regularized covariance matrices
  4. Determine ReEig epsilon from training data eigenvalues (99.5% variance)
  5. K-Fold CV — train and evaluate RBNNet, pick dual thresholds per fold
  6. Train final model on all data
  7. Save model bundle (pickle) to the subject data directory

Usage
-----
  python generate_rbnnet_model.py

All configuration is read from config.py (subject, paths, filter params,
K-fold, thresholds, RBNNET_* hyperparameters).
"""

import os
import sys
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

import config
from Utils.stream_utils import load_xdf
from Utils.rbnnet_model import (
    RBNNet,
    build_rbnnet,
    compute_epsilon_threshold,
    save_rbnnet_bundle,
)
from Generate_Riemannian_adaptive import (
    segment_and_label_one_run,
    compute_processed_covariances,
    pick_dual_thresholds_target_ambiguity,
)

# ---------------------------------------------------------------------------
# Label mapping (mirrors all other decoders)
# ---------------------------------------------------------------------------
LABEL_TO_BIN = {
    int(config.TRIGGERS["REST_BEGIN"]): 0,   # 100 → 0
    int(config.TRIGGERS["MI_BEGIN"]): 1,     # 200 → 1
}
BIN_TO_LABEL = {v: k for k, v in LABEL_TO_BIN.items()}


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _make_tensor_dataset(cov_matrices_np, labels_bin_np):
    """Convert numpy arrays to a PyTorch TensorDataset."""
    X = torch.tensor(cov_matrices_np, dtype=torch.float32)
    y = torch.tensor(labels_bin_np, dtype=torch.long)
    return TensorDataset(X, y)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one full training epoch; return mean loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_proba_np(model, cov_matrices_np, device, batch_size=128):
    """Run RBNNet inference on a numpy array of SPD matrices.

    Returns
    -------
    probs : np.ndarray, shape (n, 2)  — [P(REST), P(MI)] per sample
    """
    model.eval()
    X = torch.tensor(cov_matrices_np, dtype=torch.float32)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_probs = []
    for (batch,) in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def train_rbnnet(cov_train_np, labels_train_np, n_ch, epsilon, device,
                 epochs=None, lr=None, batch_size=None, weight_decay=None):
    """
    Train a fresh RBNNet on the given covariance matrices and binary labels.

    Returns
    -------
    model : trained RBNNet (in eval mode)
    history : list of per-epoch mean training losses
    """
    epochs = epochs or int(getattr(config, "RBNNET_EPOCHS", 200))
    lr = lr or float(getattr(config, "RBNNET_LR", 1e-3))
    batch_size = batch_size or int(getattr(config, "RBNNET_BATCH_SIZE", 32))
    weight_decay = weight_decay or float(getattr(config, "RBNNET_WEIGHT_DECAY", 1e-4))

    model = build_rbnnet(n_ch=n_ch, epsilon=epsilon, n_classes=2, n_blocks=2)
    model = model.to(device)

    dataset = _make_tensor_dataset(cov_train_np, labels_train_np)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.5)

    history = []
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loader, criterion, optimizer, device)
        scheduler.step()
        history.append(loss)
        if epoch % 20 == 0 or epoch == 1:
            print(f"  [RBNNet] Epoch {epoch:3d}/{epochs}  loss={loss:.4f}")

    model.eval()
    return model, history


# ---------------------------------------------------------------------------
# K-Fold cross-validation
# ---------------------------------------------------------------------------

def cross_validate_rbnnet(cov_matrices_np, labels_bin_np, epsilon, device):
    """
    Stratified K-Fold CV for RBNNet.  Returns per-fold metrics and
    the median dual thresholds to use for the final model.

    Returns
    -------
    results : list of dicts (one per fold)
    tl_median, th_median : median lower and upper thresholds across folds
    """
    n_splits = int(getattr(config, "N_SPLITS", 5))
    n_ch = cov_matrices_np.shape[1]
    target_ambig = float(getattr(config, "TARGET_AMBIG", 0.20))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(cov_matrices_np, labels_bin_np)
    ):
        print(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")
        cov_train = cov_matrices_np[train_idx]
        cov_val   = cov_matrices_np[val_idx]
        y_train   = labels_bin_np[train_idx]
        y_val     = labels_bin_np[val_idx]

        # Train
        model, _ = train_rbnnet(cov_train, y_train, n_ch, epsilon, device)

        # Predict on validation set
        probs_val = predict_proba_np(model, cov_val, device)
        scores_val = probs_val[:, 1]   # P(MI)

        # ROC-AUC
        try:
            auc = roc_auc_score(y_val, scores_val)
        except ValueError:
            auc = float("nan")

        # Dual thresholds from validation predictions
        tl, th, thresh_info = pick_dual_thresholds_target_ambiguity(
            y_val, scores_val, target_ambig=target_ambig
        )

        # Compute decided-only accuracy
        preds = np.full_like(y_val, -1)
        preds[scores_val >= th] = 1
        preds[scores_val <= tl] = 0
        decided = preds != -1
        acc = accuracy_score(y_val[decided], preds[decided]) if decided.any() else float("nan")
        ambig_frac = 1.0 - decided.mean()

        print(
            f"  ROC-AUC={auc:.3f}  acc(decided)={acc:.3f}  "
            f"ambiguous={ambig_frac:.1%}  tl={tl:.3f}  th={th:.3f}"
        )

        results.append({
            "fold": fold_idx + 1,
            "roc_auc": auc,
            "accuracy_decided": acc,
            "ambiguous_fraction": ambig_frac,
            "tl": tl,
            "th": th,
            "thresh_info": thresh_info,
        })

    tl_median = float(np.median([r["tl"] for r in results]))
    th_median = float(np.median([r["th"] for r in results]))
    mean_auc  = float(np.nanmean([r["roc_auc"] for r in results]))

    print(f"\n[CV Summary] mean_AUC={mean_auc:.3f}  "
          f"median_tl={tl_median:.3f}  median_th={th_median:.3f}")
    return results, tl_median, th_median, mean_auc


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_cv_results(results, subject_id, save_path=None):
    """Bar chart of per-fold ROC-AUC values."""
    aucs = [r["roc_auc"] for r in results]
    folds = [r["fold"] for r in results]
    plt.figure(figsize=(7, 4))
    plt.bar(folds, aucs, color="steelblue", alpha=0.8)
    plt.axhline(np.nanmean(aucs), color="red", linestyle="--", label=f"Mean={np.nanmean(aucs):.3f}")
    plt.xlabel("Fold")
    plt.ylabel("ROC-AUC")
    plt.title(f"RBNNet K-Fold CV — {subject_id}")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"[plot] Saved CV results → {save_path}")
    plt.show()


def plot_score_histogram(scores, labels_bin, tl, th, subject_id, save_path=None):
    """Histogram of P(MI) scores split by class, with threshold lines."""
    plt.figure(figsize=(8, 4))
    for cls, name, color in [(1, "MI", "darkorange"), (0, "REST", "steelblue")]:
        mask = labels_bin == cls
        sns.histplot(scores[mask], bins=30, alpha=0.5, label=name, color=color, kde=True)
    plt.axvline(tl, color="gray", linestyle="--", label=f"t_low={tl:.3f}")
    plt.axvline(th, color="black", linestyle="--", label=f"t_high={th:.3f}")
    plt.xlabel("P(MI)")
    plt.ylabel("Count")
    plt.title(f"RBNNet Score Distribution — {subject_id}")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"[plot] Saved score histogram → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    subject_id = config.TRAINING_SUBJECT
    data_dir = os.path.join(config.DATA_DIR, subject_id)
    print(f"\n{'='*60}")
    print(f"  RBNNet Offline Training")
    print(f"  Subject : {subject_id}")
    print(f"  Data dir: {data_dir}")
    print(f"{'='*60}\n")

    # --- Find XDF files ---
    xdf_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.xdf"), recursive=True))
    if not xdf_files:
        xdf_files = sorted(glob.glob(os.path.join(data_dir, "*.xdf")))
    if not xdf_files:
        print(f"[ERROR] No XDF files found in {data_dir}")
        sys.exit(1)
    print(f"Found {len(xdf_files)} XDF file(s):")
    for f in xdf_files:
        print(f"  {f}")

    # --- Aggregate segments across files ---
    all_segments = []
    all_labels   = []

    for xdf_path in xdf_files:
        print(f"\n[Loading] {os.path.basename(xdf_path)}")
        try:
            eeg_stream, marker_stream = load_xdf(xdf_path)
        except Exception as e:
            print(f"  [WARN] Failed to load {xdf_path}: {e}")
            continue

        try:
            segs, lbls = segment_and_label_one_run(eeg_stream, marker_stream)
        except Exception as e:
            print(f"  [WARN] Segmentation failed for {xdf_path}: {e}")
            continue

        print(f"  Segments: {segs.shape[0]}  Labels: {dict(zip(*np.unique(lbls, return_counts=True)))}")
        all_segments.append(segs)
        all_labels.append(lbls)

    if not all_segments:
        print("[ERROR] No valid segments extracted. Aborting.")
        sys.exit(1)

    segments_np = np.concatenate(all_segments, axis=0)
    labels_np   = np.concatenate(all_labels, axis=0)
    print(f"\nTotal segments: {len(segments_np)}")
    print(f"Label counts: {dict(zip(*np.unique(labels_np, return_counts=True)))}")

    # --- Artifact rejection is already done inside segment_and_label_one_run ---

    # --- Convert labels to binary (100→0, 200→1) ---
    labels_bin = np.array([LABEL_TO_BIN[int(l)] for l in labels_np])

    # --- Compute shrinkage-regularized covariance matrices ---
    print("\n[Covariance] Computing shrinkage-regularized SPD matrices...")
    lam = float(getattr(config, "SHRINKAGE_PARAM_RBNNET",
                        getattr(config, "SHRINKAGE_PARAM_MDM",
                                getattr(config, "SHRINKAGE_PARAM", 0.02))))
    cov_matrices_np = compute_processed_covariances(
        segments_np, labels_np, model_type="mdm", shrinkage_param=lam
    )
    n_ch = cov_matrices_np.shape[1]
    print(f"  SPD matrix dimension: {n_ch}x{n_ch}  (shrinkage lambda={lam})")

    # --- Compute epsilon threshold from calibration data ---
    variance_retained = float(getattr(config, "RBNNET_VARIANCE_RETAINED", 0.995))
    epsilon = compute_epsilon_threshold(cov_matrices_np, variance_retained=variance_retained)
    print(f"\n[Epsilon] ReEig threshold: eps={epsilon:.6f}  "
          f"(variance_retained={variance_retained:.1%})")

    # --- Device selection ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # --- K-Fold Cross-Validation ---
    print("\n[CV] Starting K-Fold cross-validation...")
    cv_results, tl_star, th_star, mean_auc = cross_validate_rbnnet(
        cov_matrices_np, labels_bin, epsilon, device
    )

    # --- Final model: train on ALL data ---
    print("\n[Final Model] Training on all segments...")
    final_model, loss_history = train_rbnnet(
        cov_matrices_np, labels_bin, n_ch, epsilon, device
    )

    # Evaluate final model on full training set (for reporting only)
    probs_all = predict_proba_np(final_model, cov_matrices_np, device)
    scores_all = probs_all[:, 1]

    try:
        final_auc = roc_auc_score(labels_bin, scores_all)
    except ValueError:
        final_auc = float("nan")

    # Use the CV-derived median thresholds for deployment
    print(f"\n[Thresholds] Using CV-median thresholds: tl={tl_star:.3f}  th={th_star:.3f}")
    print(f"[Final train AUC] {final_auc:.3f}")

    # --- Save model bundle ---
    out_dir = data_dir
    bundle_name = f"rbnnet_{subject_id}.pkl"
    bundle_path = os.path.join(out_dir, bundle_name)

    training_meta = {
        "subject": subject_id,
        "n_segments": int(len(segments_np)),
        "n_ch": int(n_ch),
        "epsilon": float(epsilon),
        "shrinkage_lambda": float(lam),
        "variance_retained": float(variance_retained),
        "cv_mean_auc": float(mean_auc),
        "final_train_auc": float(final_auc),
        "xdf_files": [os.path.basename(f) for f in xdf_files],
        "label_distribution": {
            str(k): int(v)
            for k, v in zip(*np.unique(labels_np, return_counts=True))
        },
    }

    # Retrieve channel names from the last successful segmentation run.
    # segment_and_label_one_run selects the motor subset defined in config.
    channel_names = list(config.MOTOR_CHANNEL_NAMES) if config.SELECT_MOTOR_CHANNELS else []

    save_rbnnet_bundle(
        model=final_model,
        label_to_bin=LABEL_TO_BIN,
        bin_to_label=BIN_TO_LABEL,
        tl_star=tl_star,
        th_star=th_star,
        roc_auc=mean_auc,
        channel_names=channel_names,
        training_meta=training_meta,
        path=bundle_path,
    )

    # --- Visualization ---
    plot_cv_results(
        cv_results, subject_id,
        save_path=os.path.join(out_dir, f"rbnnet_{subject_id}_cv_auc.png")
    )
    plot_score_histogram(
        scores_all, labels_bin, tl_star, th_star, subject_id,
        save_path=os.path.join(out_dir, f"rbnnet_{subject_id}_scores.png")
    )

    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Model saved : {bundle_path}")
    print(f"  CV mean AUC : {mean_auc:.3f}")
    print(f"  Thresholds  : tl={tl_star:.3f}  th={th_star:.3f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
