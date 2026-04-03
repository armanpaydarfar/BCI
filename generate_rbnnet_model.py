"""
generate_rbnnet_model.py
------------------------
Offline training script for the RBNNet Motor Imagery decoder.

Two architectures selectable via config.RBNNET_USE_BETA:

  RBNNET_USE_BETA = 0  (default)
    Single-band RBNNet, paper-faithful (Liu et al. NER 2023).
    Uses config.LOWCUT / config.HIGHCUT (default 8-13 Hz mu band).
    Set LOWCUT=8, HIGHCUT=30 to replicate the paper's wide-band variant.

  RBNNET_USE_BETA = 1
    DualBandRBNNet: independent mu (8-13 Hz) and beta (13-30 Hz) encoders
    with per-stream LayerNorm, concatenated before a shared Linear classifier.
    Beta band bounds controlled by RBNNET_LOWCUT_BETA / RBNNET_HIGHCUT_BETA.

Uses the same preprocessing pipeline as all other Harmony decoders:
  segment_and_label_one_run, compute_processed_covariances (from
  Generate_Riemannian_adaptive.py), and pick_dual_thresholds_target_ambiguity.

Usage:
  python generate_rbnnet_model.py
  (all parameters read from config.py)
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

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

import config
from Utils.stream_utils import load_xdf
from Utils.preprocessing import initialize_filter_bank, apply_streaming_filters
from Utils.artifact_rejection import apply_training_artifact_rejection
from Utils.rbnnet_model import (
    build_rbnnet,
    build_dual_band_rbnnet,
    compute_epsilon_threshold,
    save_rbnnet_bundle,
    DualBandRBNNet,
)
from Generate_Riemannian_adaptive import (
    segment_and_label_one_run,
    compute_processed_covariances,
    pick_dual_thresholds_target_ambiguity,
)

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
LABEL_TO_BIN = {
    int(config.TRIGGERS["REST_BEGIN"]): 0,
    int(config.TRIGGERS["MI_BEGIN"]):   1,
}
BIN_TO_LABEL = {v: k for k, v in LABEL_TO_BIN.items()}


# ---------------------------------------------------------------------------
# Beta-band covariance computation (dual-band only)
# ---------------------------------------------------------------------------

def compute_beta_covariances(segments_np, labels_np, lam):
    """
    Re-filter segments through the beta band and compute shrinkage covariances.
    Segments are (n_trials, n_ch, n_samples) — raw (pre-mu-filter) windows
    stored during segmentation. We apply a separate beta filter bank here.

    Returns
    -------
    cov_beta : np.ndarray (n_trials, n_ch, n_ch)
    """
    from pyriemann.estimation import Shrinkage

    lowcut  = float(getattr(config, "RBNNET_LOWCUT_BETA",  13.0))
    highcut = float(getattr(config, "RBNNET_HIGHCUT_BETA", 30.0))

    print(f"[Beta] Filtering {segments_np.shape[0]} segments through "
          f"{lowcut:.0f}-{highcut:.0f} Hz beta band...")

    fb_beta = initialize_filter_bank(
        fs=config.FS, lowcut=lowcut, highcut=highcut,
        notch_freqs=[60], notch_q=30,
    )

    cov_list = []
    for seg in segments_np:
        # seg: (n_ch, n_samples) — apply causal filter with fresh state per segment
        # (segments are already isolated windows; fresh state per segment is correct
        # for offline feature extraction, matching the per-window independence assumption)
        filtered, _ = apply_streaming_filters(seg, fb_beta, {})
        cov = (filtered @ filtered.T)
        tr  = np.trace(cov)
        cov = cov / tr if tr > 0 else cov
        cov_list.append(cov)

    cov_beta_raw = np.stack(cov_list, axis=0)

    print(f"[Beta] Applying shrinkage (lambda={lam})...")
    shrinker  = Shrinkage(shrinkage=lam)
    cov_beta  = shrinker.fit_transform(cov_beta_raw)
    return cov_beta


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _single_band_dataset(cov_mu_np, labels_bin_np):
    X = torch.tensor(cov_mu_np, dtype=torch.float32)
    y = torch.tensor(labels_bin_np, dtype=torch.long)
    return TensorDataset(X, y)


def _dual_band_dataset(cov_mu_np, cov_beta_np, labels_bin_np):
    X_mu   = torch.tensor(cov_mu_np,   dtype=torch.float32)
    X_beta = torch.tensor(cov_beta_np, dtype=torch.float32)
    y      = torch.tensor(labels_bin_np, dtype=torch.long)
    return TensorDataset(X_mu, X_beta, y)


def _train_one_epoch_single(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for (X_batch, y_batch) in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        criterion(model(X_batch), y_batch).backward()
        optimizer.step()
        total_loss += criterion(model(X_batch.detach()), y_batch).item() * len(y_batch)
    return total_loss / len(loader.dataset)


def _train_one_epoch_dual(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for (X_mu, X_beta, y_batch) in loader:
        X_mu, X_beta, y_batch = X_mu.to(device), X_beta.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_mu, X_beta), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def _predict_proba_single(model, cov_mu_np, device, batch_size=128):
    model.eval()
    ds     = TensorDataset(torch.tensor(cov_mu_np, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs  = []
    for (batch,) in loader:
        probs.append(torch.softmax(model(batch.to(device)), dim=-1).cpu().numpy())
    return np.concatenate(probs, axis=0)


@torch.no_grad()
def _predict_proba_dual(model, cov_mu_np, cov_beta_np, device, batch_size=128):
    model.eval()
    ds     = TensorDataset(
        torch.tensor(cov_mu_np,   dtype=torch.float32),
        torch.tensor(cov_beta_np, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs  = []
    for (bmu, bbeta) in loader:
        probs.append(
            torch.softmax(model(bmu.to(device), bbeta.to(device)), dim=-1).cpu().numpy()
        )
    return np.concatenate(probs, axis=0)


def _build_and_train(use_beta, n_ch, epsilon_mu, epsilon_beta,
                     cov_mu_tr, cov_beta_tr, y_tr, device,
                     epochs, lr, batch_size, weight_decay):
    """Construct, train, and return a fresh model."""
    if use_beta:
        model = build_dual_band_rbnnet(n_ch, epsilon_mu, epsilon_beta)
        ds    = _dual_band_dataset(cov_mu_tr, cov_beta_tr, y_tr)
    else:
        model = build_rbnnet(n_ch, epsilon_mu)
        ds    = _single_band_dataset(cov_mu_tr, y_tr)

    model = model.to(device)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)

    for epoch in range(1, epochs + 1):
        if use_beta:
            loss = _train_one_epoch_dual(model, loader, criterion, optimizer, device)
        else:
            loss = _train_one_epoch_single(model, loader, criterion, optimizer, device)
        scheduler.step()
        if epoch % 20 == 0 or epoch == 1:
            print(f"  [RBNNet] Epoch {epoch:3d}/{epochs}  loss={loss:.4f}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# K-Fold cross-validation
# ---------------------------------------------------------------------------

def cross_validate_rbnnet(use_beta, cov_mu_np, cov_beta_np, labels_bin_np,
                           epsilon_mu, epsilon_beta, device):
    n_splits     = int(getattr(config, "N_SPLITS", 5))
    target_ambig = float(getattr(config, "TARGET_AMBIG", 0.20))
    epochs       = int(getattr(config, "RBNNET_EPOCHS", 200))
    lr           = float(getattr(config, "RBNNET_LR", 1e-3))
    batch_size   = int(getattr(config, "RBNNET_BATCH_SIZE", 32))
    weight_decay = float(getattr(config, "RBNNET_WEIGHT_DECAY", 1e-4))
    n_ch         = cov_mu_np.shape[1]

    skf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(cov_mu_np, labels_bin_np)):
        print(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")

        cov_mu_tr  = cov_mu_np[tr_idx];   cov_mu_val  = cov_mu_np[val_idx]
        cov_b_tr   = cov_beta_np[tr_idx] if use_beta else None
        cov_b_val  = cov_beta_np[val_idx] if use_beta else None
        y_tr       = labels_bin_np[tr_idx]
        y_val      = labels_bin_np[val_idx]

        model = _build_and_train(
            use_beta, n_ch, epsilon_mu, epsilon_beta,
            cov_mu_tr, cov_b_tr, y_tr, device,
            epochs, lr, batch_size, weight_decay,
        )

        if use_beta:
            probs_val = _predict_proba_dual(model, cov_mu_val, cov_b_val, device)
        else:
            probs_val = _predict_proba_single(model, cov_mu_val, device)

        scores_val = probs_val[:, 1]

        try:
            auc = roc_auc_score(y_val, scores_val)
        except ValueError:
            auc = float("nan")

        tl, th, thresh_info = pick_dual_thresholds_target_ambiguity(
            y_val, scores_val, target_ambig=target_ambig
        )

        preds   = np.full_like(y_val, -1)
        preds[scores_val >= th] = 1
        preds[scores_val <= tl] = 0
        decided = preds != -1
        acc     = accuracy_score(y_val[decided], preds[decided]) if decided.any() else float("nan")
        ambig   = 1.0 - decided.mean()

        print(f"  ROC-AUC={auc:.3f}  acc(decided)={acc:.3f}  "
              f"ambiguous={ambig:.1%}  tl={tl:.3f}  th={th:.3f}")

        results.append({
            "fold": fold_idx + 1, "roc_auc": auc,
            "accuracy_decided": acc, "ambiguous_fraction": ambig,
            "tl": tl, "th": th,
        })

    tl_median  = float(np.median([r["tl"] for r in results]))
    th_median  = float(np.median([r["th"] for r in results]))
    mean_auc   = float(np.nanmean([r["roc_auc"] for r in results]))
    print(f"\n[CV Summary] mean_AUC={mean_auc:.3f}  "
          f"median_tl={tl_median:.3f}  median_th={th_median:.3f}")
    return results, tl_median, th_median, mean_auc


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_cv_auc(results, subject_id, save_path=None):
    aucs  = [r["roc_auc"] for r in results]
    folds = [r["fold"] for r in results]
    plt.figure(figsize=(7, 4))
    plt.bar(folds, aucs, color="steelblue", alpha=0.8)
    plt.axhline(np.nanmean(aucs), color="red", linestyle="--",
                label=f"Mean={np.nanmean(aucs):.3f}")
    plt.xlabel("Fold"); plt.ylabel("ROC-AUC")
    plt.title(f"RBNNet K-Fold CV — {subject_id}")
    plt.legend(); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"[plot] {save_path}")
    plt.show()


def _plot_scores(scores, labels_bin, tl, th, subject_id, save_path=None):
    plt.figure(figsize=(8, 4))
    for cls, name, color in [(1, "MI", "darkorange"), (0, "REST", "steelblue")]:
        sns.histplot(scores[labels_bin == cls], bins=30, alpha=0.5,
                     label=name, color=color, kde=True)
    plt.axvline(tl, color="gray",  linestyle="--", label=f"t_low={tl:.3f}")
    plt.axvline(th, color="black", linestyle="--", label=f"t_high={th:.3f}")
    plt.xlabel("P(MI)"); plt.ylabel("Count")
    plt.title(f"RBNNet Score Distribution — {subject_id}")
    plt.legend(); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"[plot] {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    use_beta   = bool(int(getattr(config, "RBNNET_USE_BETA", 0)))
    subject_id = config.TRAINING_SUBJECT
    data_dir   = os.path.join(config.DATA_DIR, f"sub-{subject_id}", "training_data")
    model_dir  = os.path.join(config.DATA_DIR, f"sub-{subject_id}", "models")
    arch_label = "dual_band" if use_beta else "single_band"

    print(f"\n{'='*60}")
    print(f"  RBNNet Offline Training  [{arch_label}]")
    print(f"  Subject : {subject_id}")
    print(f"  Data dir: {data_dir}")
    if use_beta:
        lb = float(getattr(config, "RBNNET_LOWCUT_BETA",  13.0))
        hb = float(getattr(config, "RBNNET_HIGHCUT_BETA", 30.0))
        print(f"  Mu  band: {config.LOWCUT}-{config.HIGHCUT} Hz")
        print(f"  Beta band: {lb:.0f}-{hb:.0f} Hz")
    else:
        print(f"  Band: {config.LOWCUT}-{config.HIGHCUT} Hz (single)")
    print(f"{'='*60}\n")

    # --- Locate XDF files (non-recursive, matching XGB convention) ---
    xdf_files = sorted([
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.endswith(".xdf") and "OBS" not in f
    ])
    if not xdf_files:
        print(f"[ERROR] No XDF files found in {data_dir}")
        sys.exit(1)
    print(f"Found {len(xdf_files)} XDF file(s):")
    for f in xdf_files:
        print(f"  {os.path.basename(f)}")

    # --- Segment all files ---
    all_segments, all_labels = [], []
    for xdf_path in xdf_files:
        print(f"\n[Loading] {os.path.basename(xdf_path)}")
        try:
            eeg_stream, marker_stream = load_xdf(xdf_path)
        except Exception as e:
            print(f"  [WARN] {e}")
            continue
        try:
            segs, lbls = segment_and_label_one_run(eeg_stream, marker_stream)
        except Exception as e:
            print(f"  [WARN] Segmentation failed: {e}")
            continue
        print(f"  Segments: {segs.shape[0]}  "
              f"Labels: {dict(zip(*np.unique(lbls, return_counts=True)))}")
        segs, lbls, _ = apply_training_artifact_rejection(segs, lbls)
        print(f"  Retained after artifact rejection: {segs.shape[0]}")
        all_segments.append(segs)
        all_labels.append(lbls)

    if not all_segments:
        print("[ERROR] No valid segments extracted.")
        sys.exit(1)

    segments_np = np.concatenate(all_segments, axis=0)
    labels_np   = np.concatenate(all_labels,   axis=0)
    labels_bin  = np.array([LABEL_TO_BIN[int(l)] for l in labels_np])

    print(f"\nTotal segments: {len(segments_np)}")
    print(f"Label counts:   {dict(zip(*np.unique(labels_np, return_counts=True)))}")

    # --- Mu-band covariance matrices (always needed) ---
    lam = float(getattr(config, "SHRINKAGE_PARAM_RBNNET",
                getattr(config, "SHRINKAGE_PARAM_MDM", 0.02)))
    print(f"\n[Mu cov] shrinkage lambda={lam}")
    cov_mu_np = compute_processed_covariances(
        segments_np, labels_np, model_type="mdm", shrinkage_param=lam
    )
    n_ch = cov_mu_np.shape[1]
    print(f"  SPD dimension: {n_ch}x{n_ch}")

    # --- Beta-band covariance matrices (dual-band only) ---
    cov_beta_np = None
    if use_beta:
        cov_beta_np = compute_beta_covariances(segments_np, labels_np, lam)

    # --- Epsilon per band ---
    var_ret   = float(getattr(config, "RBNNET_VARIANCE_RETAINED", 0.995))
    epsilon_mu = compute_epsilon_threshold(cov_mu_np, var_ret)
    print(f"\n[Epsilon] mu eps={epsilon_mu:.6f}  (variance_retained={var_ret:.1%})")

    epsilon_beta = epsilon_mu  # default: same unless computed separately
    if use_beta and cov_beta_np is not None:
        epsilon_beta = compute_epsilon_threshold(cov_beta_np, var_ret)
        print(f"[Epsilon] beta eps={epsilon_beta:.6f}")

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # --- Cross-validation ---
    print("\n[CV] Starting K-Fold cross-validation...")
    cv_results, tl_star, th_star, mean_auc = cross_validate_rbnnet(
        use_beta, cov_mu_np, cov_beta_np, labels_bin,
        epsilon_mu, epsilon_beta, device,
    )

    # --- Final model on all data ---
    epochs       = int(getattr(config, "RBNNET_EPOCHS", 200))
    lr           = float(getattr(config, "RBNNET_LR", 1e-3))
    batch_size   = int(getattr(config, "RBNNET_BATCH_SIZE", 32))
    weight_decay = float(getattr(config, "RBNNET_WEIGHT_DECAY", 1e-4))

    print("\n[Final Model] Training on all segments...")
    final_model = _build_and_train(
        use_beta, n_ch, epsilon_mu, epsilon_beta,
        cov_mu_np, cov_beta_np, labels_bin, device,
        epochs, lr, batch_size, weight_decay,
    )

    # Full-data scores for reporting
    if use_beta:
        probs_all = _predict_proba_dual(final_model, cov_mu_np, cov_beta_np, device)
    else:
        probs_all = _predict_proba_single(final_model, cov_mu_np, device)
    scores_all = probs_all[:, 1]

    try:
        final_auc = roc_auc_score(labels_bin, scores_all)
    except ValueError:
        final_auc = float("nan")

    print(f"\n[Thresholds] tl={tl_star:.3f}  th={th_star:.3f}")
    print(f"[Final train AUC] {final_auc:.3f}")

    # --- Save bundle ---
    os.makedirs(model_dir, exist_ok=True)
    bundle_name = f"rbnnet_{arch_label}_{subject_id}.pkl"
    bundle_path = os.path.join(model_dir, bundle_name)

    training_meta = {
        "subject":           subject_id,
        "architecture":      arch_label,
        "n_segments":        int(len(segments_np)),
        "n_ch":              int(n_ch),
        "epsilon_mu":        float(epsilon_mu),
        "epsilon_beta":      float(epsilon_beta) if use_beta else None,
        "shrinkage_lambda":  float(lam),
        "variance_retained": float(var_ret),
        "cv_mean_auc":       float(mean_auc),
        "final_train_auc":   float(final_auc),
        "mu_band":           [float(config.LOWCUT), float(config.HIGHCUT)],
        "beta_band":         [float(getattr(config, "RBNNET_LOWCUT_BETA",  13.0)),
                              float(getattr(config, "RBNNET_HIGHCUT_BETA", 30.0))]
                             if use_beta else None,
        "xdf_files": [os.path.basename(f) for f in xdf_files],
        "label_distribution": {
            str(k): int(v)
            for k, v in zip(*np.unique(labels_np, return_counts=True))
        },
    }

    channel_names = (list(config.MOTOR_CHANNEL_NAMES)
                     if config.SELECT_MOTOR_CHANNELS else [])

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

    # --- Plots ---
    _plot_cv_auc(cv_results, subject_id,
                 save_path=os.path.join(model_dir, f"rbnnet_{arch_label}_{subject_id}_cv_auc.png"))
    _plot_scores(scores_all, labels_bin, tl_star, th_star, subject_id,
                 save_path=os.path.join(model_dir, f"rbnnet_{arch_label}_{subject_id}_scores.png"))

    print(f"\n{'='*60}")
    print(f"  Training complete  [{arch_label}]")
    print(f"  Model saved : {bundle_path}")
    print(f"  CV mean AUC : {mean_auc:.3f}")
    print(f"  Thresholds  : tl={tl_star:.3f}  th={th_star:.3f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
