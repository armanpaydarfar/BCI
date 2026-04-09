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

# Prevent Intel Fortran/MKL runtime from intercepting Ctrl+C on Windows,
# ensuring Python's signal handler and try/finally blocks run on interrupt.
if sys.platform == "win32":
    os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

import glob
import pickle
import random
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — plots save to file, never block
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Environment diagnostics (printed once at startup)
# ---------------------------------------------------------------------------

def _print_env_diagnostics():
    """Print torch/CUDA/Triton environment summary at startup."""
    print("\n[Env] torch version  :", torch.__version__)
    print("[Env] CUDA version   :", torch.version.cuda)
    print("[Env] CUDA available :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[Env] GPU            :", torch.cuda.get_device_name(0))
    else:
        print("[Env] GPU            : N/A")

    # Triton availability check
    try:
        import triton  # noqa: F401
        print("[Env] Triton         : INSTALLED —", getattr(triton, "__version__", "unknown version"))
        print("[Env] inductor backend: available (Triton present)")
    except ImportError:
        print("[Env] Triton         : NOT installed")
        if sys.platform == "win32":
            print("[Env] inductor backend: UNAVAILABLE on Windows without Triton")
            print("[Env]   To enable inductor on Windows+CUDA+PyTorch 2.6, run:")
            print('[Env]     pip install "triton-windows<3.3"')
            print("[Env]   cudagraphs backend will be used instead (no Triton needed)")
        else:
            print("[Env] inductor backend: unavailable (install triton)")

    # torch.compile availability
    has_compile = hasattr(torch, "compile")
    print("[Env] torch.compile  :", "available" if has_compile else "NOT available (torch < 2.0)")
    print()

# ---------------------------------------------------------------------------
# Global seed for reproducibility
# ---------------------------------------------------------------------------
_SEED = 42
random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)
torch.cuda.manual_seed_all(_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# TF32 ('high') is NOT set — SPD matrix ops accumulate floating point error
# over training epochs with reduced mantissa precision, risking ill-conditioned
# matrices. Keep PyTorch default ('highest' / full float32) to match Linux.

# ---------------------------------------------------------------------------
# Sleep inhibitor (Windows only — silently skipped on Linux/Mac)
# ---------------------------------------------------------------------------
def _inhibit_sleep():
    """Disable standby sleep on Windows. No-op on other platforms."""
    if sys.platform != "win32":
        return
    try:
        import subprocess
        subprocess.run(["powercfg", "/change", "standby-timeout-ac", "0"],
                       check=True, capture_output=True)
        print("[Sleep] Sleep disabled for training run.")
    except Exception:
        pass  # powercfg is best-effort; training proceeds regardless

def _restore_sleep():
    """Restore standby sleep to 30 min on Windows. No-op on other platforms."""
    if sys.platform != "win32":
        return
    try:
        import subprocess
        subprocess.run(["powercfg", "/change", "standby-timeout-ac", "15"],
                       check=True, capture_output=True)
        print("[Sleep] Sleep restored.")
    except Exception:
        pass  # powercfg is best-effort; sleep policy is not critical to restore

from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

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
    shrinker = Shrinkage(shrinkage=lam)
    cov_beta = shrinker.fit_transform(cov_beta_raw)

    if config.RECENTERING:
        from pyriemann.preprocessing import Whitening
        print("[Beta] Applying Riemannian whitening (matching mu-band pipeline)...")
        whitener = Whitening(metric="riemann")
        cov_beta = whitener.fit_transform(cov_beta)

    return cov_beta


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def _predict_proba_single(model, cov_mu_np, device, batch_size=128):
    model.eval()
    X      = torch.tensor(cov_mu_np, dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
    probs  = []
    for (batch,) in loader:
        probs.append(torch.softmax(model(batch), dim=-1).cpu().numpy())
    return np.concatenate(probs, axis=0)


@torch.no_grad()
def _predict_proba_dual(model, cov_mu_np, cov_beta_np, device, batch_size=128):
    model.eval()
    X_mu   = torch.tensor(cov_mu_np,   dtype=torch.float32).to(device)
    X_beta = torch.tensor(cov_beta_np, dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(X_mu, X_beta), batch_size=batch_size, shuffle=False)
    probs  = []
    for (bmu, bbeta) in loader:
        probs.append(torch.softmax(model(bmu, bbeta), dim=-1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def _build_and_train(use_beta, n_ch, epsilon_mu, epsilon_beta,
                     cov_mu_tr, cov_beta_tr, y_tr, device,
                     epochs, lr, batch_size, weight_decay):
    """Construct, train, and return a fresh model."""
    if use_beta:
        model = build_dual_band_rbnnet(n_ch, epsilon_mu, epsilon_beta)
        ds = TensorDataset(
            torch.tensor(cov_mu_tr,   dtype=torch.float32).to(device),
            torch.tensor(cov_beta_tr, dtype=torch.float32).to(device),
            torch.tensor(y_tr,        dtype=torch.long).to(device),
        )
    else:
        model = build_rbnnet(n_ch, epsilon_mu)
        ds = TensorDataset(
            torch.tensor(cov_mu_tr, dtype=torch.float32).to(device),
            torch.tensor(y_tr,      dtype=torch.long).to(device),
        )

    model = model.to(device)

    # torch.compile is intentionally omitted. The RBNNet forward pass is dominated
    # by eigh calls (Karcher flow, ReEig, LogEig) which route through cuSOLVER and
    # cannot be fused or optimised by the inductor backend. compile adds ~40s of
    # per-fold recompile overhead with no runtime benefit, and the cuSOLVER dispatch
    # path it generates is less numerically robust than the eager path, causing
    # convergence failures (linalg.eigh error code 2) on the full training dataset.

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)

    n_samples  = len(loader.dataset)
    n_batches  = len(loader)
    print(f"[RBNNet] Starting: {n_samples} samples  batch={batch_size}"
          f"  {n_batches} batches/epoch  epochs={epochs}"
          f"  lr={lr:.2e}  wd={weight_decay:.2e}  device={device}")

    best_loss        = float('inf')
    loss_at_last_log = None
    last_improve_ep  = 0
    t_start          = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        model.train()
        epoch_loss    = 0.0
        grad_norm_sum = 0.0

        if use_beta:
            for (X_mu, X_beta, y_batch) in loader:
                optimizer.zero_grad(set_to_none=True)
                batch_loss = criterion(model(X_mu, X_beta), y_batch)
                if not torch.isfinite(batch_loss):
                    raise RuntimeError(
                        f"[RBNNet] Non-finite loss ({batch_loss.item()}) at epoch {epoch}.")
                batch_loss.backward()
                grad_norm_sum += torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float('inf')).item()
                optimizer.step()
                epoch_loss += batch_loss.item() * len(y_batch)
        else:
            for (X_batch, y_batch) in loader:
                optimizer.zero_grad(set_to_none=True)
                batch_loss = criterion(model(X_batch), y_batch)
                if not torch.isfinite(batch_loss):
                    raise RuntimeError(
                        f"[RBNNet] Non-finite loss ({batch_loss.item()}) at epoch {epoch}.")
                batch_loss.backward()
                grad_norm_sum += torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float('inf')).item()
                optimizer.step()
                epoch_loss += batch_loss.item() * len(y_batch)

        loss       = epoch_loss / n_samples
        epoch_time = time.time() - t_epoch
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        if loss < best_loss:
            best_loss       = loss
            last_improve_ep = epoch

        if epoch % 20 == 0 or epoch == 1:
            elapsed       = time.time() - t_start
            eta           = elapsed / epoch * (epochs - epoch)
            avg_grad_norm = grad_norm_sum / n_batches
            throughput    = n_samples / epoch_time
            delta         = (f"{loss - loss_at_last_log:+.4f}"
                             if loss_at_last_log is not None else "   n/a")

            flags = ""
            if epoch - last_improve_ep >= 40 and epoch > 40:
                flags += f"  [PLATEAU {epoch - last_improve_ep} epochs]"
            if avg_grad_norm > 10.0:
                flags += f"  [GRAD LARGE: {avg_grad_norm:.1f}]"

            print(f"  [RBNNet] Epoch {epoch:3d}/{epochs}"
                  f"  loss={loss:.4f} ({delta}, best={best_loss:.4f})"
                  f"  lr={current_lr:.2e}  |grad|={avg_grad_norm:.3f}"
                  f"  {throughput:.0f} smp/s"
                  f"  elapsed={elapsed:.0f}s  eta={eta:.0f}s"
                  f"{flags}")

            loss_at_last_log = loss

    model.eval()
    total_time = time.time() - t_start
    print(f"[RBNNet] Done: {epochs} epochs in {total_time:.0f}s"
          f"  best_loss={best_loss:.4f} (ep {last_improve_ep})"
          f"  final_loss={loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# K-Fold cross-validation
# ---------------------------------------------------------------------------

def cross_validate_rbnnet(use_beta, cov_mu_np, cov_beta_np, labels_bin_np,
                           epsilon_mu, epsilon_beta, device, trial_ids=None, file_ids=None):
    n_splits     = int(getattr(config, "N_SPLITS", 5))
    target_ambig = float(getattr(config, "TARGET_AMBIG", 0.20))
    epochs       = int(getattr(config, "RBNNET_EPOCHS", 200))
    lr           = float(getattr(config, "RBNNET_LR", 1e-3))
    batch_size   = int(getattr(config, "RBNNET_BATCH_SIZE", 32))
    weight_decay = float(getattr(config, "RBNNET_WEIGHT_DECAY", 1e-4))
    n_ch         = cov_mu_np.shape[1]

    # Use leave-one-session-out CV when file_ids are provided (preferred: prevents
    # session-level artifact leakage in addition to trial-level window leakage).
    # Fall back to StratifiedGroupKFold on trial_ids when file_ids are absent.
    if file_ids is not None:
        skf = LeaveOneGroupOut()
        split_groups = file_ids
        n_folds = len(np.unique(file_ids))
        print(f"\n[CV] Leave-One-Session-Out ({n_folds} sessions)")
    else:
        groups = trial_ids if trial_ids is not None else np.arange(len(labels_bin_np))
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_groups = groups
        n_folds = n_splits
        print(f"\n[CV] {n_splits}-Fold (trial-grouped)")

    results       = []
    oof_posterior = {0: [], 1: []}  # P(true class) per held-out sample, keyed by true label

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(cov_mu_np, labels_bin_np, groups=split_groups)):
        print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")

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

        for i, true_lbl in enumerate(y_val):
            oof_posterior[true_lbl].append(probs_val[i, true_lbl])

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
    oof_posterior = {k: np.array(v) for k, v in oof_posterior.items()}
    return results, tl_median, th_median, mean_auc, oof_posterior


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
    plt.legend(fontsize=9); plt.grid(True, ls=":", alpha=0.6); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
        print(f"[plot] {save_path}")
    plt.close()


def _plot_scores(scores, labels_bin, tl, th, subject_id, save_path=None):
    s0 = scores[labels_bin == 0]
    s1 = scores[labels_bin == 1]
    plt.figure(figsize=(9, 5))
    plt.hist(s0, bins=30, alpha=0.6, density=True, label="True REST")
    plt.hist(s1, bins=30, alpha=0.6, density=True, label="True MI")
    plt.axvline(tl, color="black", ls="--", lw=1.5, label=f"t_low={tl:.3f}")
    plt.axvline(th, color="black", ls="--", lw=1.5, label=f"t_high={th:.3f}")
    yl = plt.ylim()
    plt.fill_betweenx(yl, tl, th, color="gray", alpha=0.12, label="Ambiguous")
    plt.xlabel("Score (P[MI])"); plt.ylabel("Density")
    plt.title(f"RBNNet Score Distributions — {subject_id}")
    plt.legend(fontsize=9); plt.grid(True, ls=":", alpha=0.6); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
        print(f"[plot] {save_path}")
    plt.close()


def _plot_roc(scores, labels_bin, th, subject_id, save_path=None):
    fpr, tpr, thr = roc_curve(labels_bin, scores)
    auc = roc_auc_score(labels_bin, scores)
    idx = np.argmin(np.abs(thr - th))
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], ls="--", alpha=0.7, label="Chance")
    plt.scatter(fpr[idx], tpr[idx], s=60, label=f"operating @ t_high={th:.3f}")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"RBNNet ROC — {subject_id}")
    plt.legend(fontsize=9); plt.grid(True, ls=":", alpha=0.6); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
        print(f"[plot] {save_path}")
    plt.close()


def _plot_confusion_with_reject(scores, labels_bin, tl, th, subject_id, save_path=None):
    pred = np.full_like(labels_bin, -1)
    pred[scores >= th] = 1
    pred[scores <= tl] = 0
    mat = np.array([
        np.sum((labels_bin == 0) & (pred == 0)),
        np.sum((labels_bin == 0) & (pred == -1)),
        np.sum((labels_bin == 0) & (pred == 1)),
        np.sum((labels_bin == 1) & (pred == 0)),
        np.sum((labels_bin == 1) & (pred == -1)),
        np.sum((labels_bin == 1) & (pred == 1)),
    ], dtype=int).reshape(2, 3)
    plt.figure(figsize=(6.5, 4.8))
    im = plt.imshow(mat, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks([0, 1, 2], ["REST", "AMB", "MI"])
    plt.yticks([0, 1], ["REST (true)", "MI (true)"])
    for i in range(2):
        for j in range(3):
            plt.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=11)
    plt.title(f"RBNNet Confusion with Reject — {subject_id}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
        print(f"[plot] {save_path}")
    plt.close()


def _plot_risk_coverage(scores, labels_bin, tl, th, subject_id, save_path=None):
    center = (tl + th) / 2.0
    widths = np.linspace(0.0, 0.9, 35)
    cov, acc, cost = [], [], []
    for w in widths:
        _tl = np.clip(center - w / 2, 0, 1)
        _th = np.clip(center + w / 2, 0, 1)
        pred = np.full_like(labels_bin, -1)
        pred[scores >= _th] = 1
        pred[scores <= _tl] = 0
        TP = int(((pred == 1) & (labels_bin == 1)).sum())
        TN = int(((pred == 0) & (labels_bin == 0)).sum())
        FP = int(((pred == 1) & (labels_bin == 0)).sum())
        FN = int(((pred == 0) & (labels_bin == 1)).sum())
        U  = int((pred == -1).sum())
        decided = TP + TN + FP + FN
        cov.append(decided / len(labels_bin))
        acc.append((TP + TN) / decided if decided else np.nan)
        cost.append(1.0 * FP + 1.0 * FN + 0.3 * U)
    cov  = np.array(cov)
    acc  = np.array(acc)
    cost = np.array(cost)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(cov * 100, acc, marker="o")
    axes[0].set_xlabel("Coverage (%)"); axes[0].set_ylabel("Decided-only Accuracy")
    axes[0].set_title(f"RBNNet Risk–Coverage: Accuracy — {subject_id}")
    axes[0].grid(True, ls=":", alpha=0.6)
    axes[1].plot(cov * 100, cost, marker="o")
    axes[1].set_xlabel("Coverage (%)"); axes[1].set_ylabel("Cost (c_fp=1, c_fn=1, c_rej=0.3)")
    axes[1].set_title(f"RBNNet Risk–Coverage: Cost — {subject_id}")
    axes[1].grid(True, ls=":", alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
        print(f"[plot] {save_path}")
    plt.close()


def _plot_posterior_probabilities(oof_posterior, subject_id, save_path=None):
    # oof_posterior: {0: P(REST) for REST trials, 1: P(MI) for MI trials} — held-out only.
    # Matches the adaptive script: each value is P(true class) from the fold the sample was held out in.
    rest_probs = oof_posterior[0]
    mi_probs   = oof_posterior[1]
    bins = np.linspace(0, 1, 20)
    plt.figure(figsize=(10, 6))
    sns.histplot(rest_probs, bins=bins, alpha=0.6, label="Rest Probability",
                 kde=True, color="skyblue")
    sns.histplot(mi_probs,   bins=bins, alpha=0.6, label="MI Probability",
                 kde=True, color="darkorange")
    plt.axvline(np.mean(rest_probs), color="skyblue",    linestyle="--", linewidth=1.5,
                label=f"Rest Mean = {np.mean(rest_probs):.2f}")
    plt.axvline(np.mean(mi_probs),   color="darkorange", linestyle="--", linewidth=1.5,
                label=f"MI Mean = {np.mean(mi_probs):.2f}")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title(f"RBNNet Posterior Probability Distribution — {subject_id}")
    plt.legend(title="True Class")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
        print(f"[plot] {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _print_env_diagnostics()

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
    all_segments, all_labels, all_trial_ids, all_file_ids = [], [], [], []
    for file_idx, xdf_path in enumerate(xdf_files):
        print(f"\n[Loading] {os.path.basename(xdf_path)}")
        try:
            eeg_stream, marker_stream = load_xdf(xdf_path)
        except Exception as e:
            print(f"  [WARN] {e}")
            continue
        try:
            segs, lbls, trial_ids = segment_and_label_one_run(eeg_stream, marker_stream)
        except Exception as e:
            print(f"  [WARN] Segmentation failed: {e}")
            continue
        print(f"  Segments: {segs.shape[0]}  "
              f"Labels: {dict(zip(*np.unique(lbls, return_counts=True)))}")
        segs, lbls, (trial_ids,) = apply_training_artifact_rejection(segs, lbls, trial_ids)
        print(f"  Retained after artifact rejection: {segs.shape[0]}")
        offset = int(all_trial_ids[-1].max()) + 1 if all_trial_ids else 0
        all_trial_ids.append(trial_ids + offset)
        all_file_ids.append(np.full(len(lbls), file_idx, dtype=int))
        all_segments.append(segs)
        all_labels.append(lbls)

    if not all_segments:
        print("[ERROR] No valid segments extracted.")
        sys.exit(1)

    segments_np    = np.concatenate(all_segments, axis=0)
    labels_np      = np.concatenate(all_labels,   axis=0)
    trial_ids_np   = np.concatenate(all_trial_ids, axis=0)
    file_ids_np    = np.concatenate(all_file_ids,  axis=0)
    labels_bin     = np.array([LABEL_TO_BIN[int(l)] for l in labels_np])

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
    cv_results, tl_star, th_star, mean_auc, oof_posterior = cross_validate_rbnnet(
        use_beta, cov_mu_np, cov_beta_np, labels_bin,
        epsilon_mu, epsilon_beta, device,
        trial_ids=trial_ids_np,
        file_ids=file_ids_np,
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
    _plot_roc(scores_all, labels_bin, th_star, subject_id,
              save_path=os.path.join(model_dir, f"rbnnet_{arch_label}_{subject_id}_roc.png"))
    _plot_confusion_with_reject(scores_all, labels_bin, tl_star, th_star, subject_id,
                                save_path=os.path.join(model_dir, f"rbnnet_{arch_label}_{subject_id}_confusion.png"))
    _plot_risk_coverage(scores_all, labels_bin, tl_star, th_star, subject_id,
                        save_path=os.path.join(model_dir, f"rbnnet_{arch_label}_{subject_id}_risk_coverage.png"))
    _plot_posterior_probabilities(oof_posterior, subject_id,
                                  save_path=os.path.join(model_dir, f"rbnnet_{arch_label}_{subject_id}_posterior_probs.png"))

    print(f"\n{'='*60}")
    print(f"  Training complete  [{arch_label}]")
    print(f"  Model saved : {bundle_path}")
    print(f"  CV mean AUC : {mean_auc:.3f}")
    print(f"  Thresholds  : tl={tl_star:.3f}  th={th_star:.3f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    _inhibit_sleep()
    try:
        main()
    finally:
        _restore_sleep()
