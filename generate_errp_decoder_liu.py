"""
generate_errp_decoder_liu.py — Train an ErrP decoder from the Liu et al. (2025) dataset.

Produces a model bundle compatible with runtime_common.load_errp_model(), so the
resulting file is a drop-in replacement for a session-trained bundle.

Pipeline: xDAWN spatial filtering + Riemannian geometry (MDM or TangentSpace + LR).
Cross-validation: leave-one-subject-out (LOSO), which directly measures how well
the model generalises to an unseen subject — the relevant metric for a generic
pre-trained decoder intended for use on a new Harmony participant.

Data source: combinedEpochs_v2.mat (Liu, Iwane et al. 2025)
  - Already theta-band filtered (1–10 Hz); do not re-filter.
  - 16 subjects × ~1984 epochs each; 50% error / 50% correct.
  - 32 channels; we select only the Harmony ErrP channel subset.

Epoch window: [0, 800] ms post-event (our pipeline convention). The full epoch
in the mat file spans [−498, 1000] ms; we crop the [0, 800] ms portion for
training so the xDAWN spatial filters and Riemannian class prototypes are learned
on exactly the signal the online EEGStreamState will provide.

Baseline correction: per-epoch mean of [−200, 0] ms is subtracted before
cropping. This is applied offline only and does not affect the online path
(which uses session-rolling mean in EEGStreamState).

Usage:
  python generate_errp_decoder_liu.py --mat "/path/to/combinedEpochs_v2.mat"
  python generate_errp_decoder_liu.py --mat combined... --backend xdawn_lr
  python generate_errp_decoder_liu.py --mat combined... --out models/errp_liu.pkl

Reference: Liu, Iwane et al. (2025). Brain-computer interface training fosters
perceptual skills to detect errors. bioRxiv. doi:10.1101/2025.04.26.650792
"""

import argparse
import os
import pickle
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

import config
import Generate_Riemannian_adaptive as base
from generate_errp_decoder import _build_classifier, _fit_xdawn_covs
from Utils.liu_data_loader import (
    HARMONY_ERRP_CHANNELS,
    LIU_FS,
    load_liu_epochs,
)

# =============================================================================
# Epoch window (matching config and our existing pipeline convention)
# =============================================================================
TMIN = float(getattr(config, "ERRP_EPOCH_TMIN", 0.0))   # 0.0 s
TMAX = float(getattr(config, "ERRP_EPOCH_TMAX", 0.8))   # 0.8 s
N_FILTERS = int(getattr(config, "ERRP_XDAWN_N_FILTERS", 4))
TARGET_AMBIG = float(getattr(config, "ERRP_TARGET_AMBIG", 0.20))

# Pre-stimulus baseline window for offline correction.
_BASELINE_TMIN = -0.200  # s
_BASELINE_TMAX = 0.0     # s


def _baseline_correct(X: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """Subtract per-epoch mean of [−200, 0] ms from every epoch."""
    mask = (time_s >= _BASELINE_TMIN) & (time_s <= _BASELINE_TMAX)
    return X - X[:, :, mask].mean(axis=2, keepdims=True)


def _crop_epoch_window(X: np.ndarray, time_s: np.ndarray,
                       tmin: float, tmax: float) -> np.ndarray:
    """
    Crop X to [tmin, tmax] seconds.

    Uses the same rounding logic as errp_feature_pipeline.segment_errp_epochs
    so epoch_samples matches what the online path produces.
    """
    epoch_samples = int(round((tmax - tmin) * LIU_FS))
    tmin_sample   = int(round(tmin * LIU_FS))
    # Find index in the time_s array corresponding to t=0.
    zero_idx = int(np.argmin(np.abs(time_s)))
    start = zero_idx + tmin_sample
    end   = start + epoch_samples
    if start < 0 or end > X.shape[2]:
        raise ValueError(
            f"Requested crop [{tmin}, {tmax}] s is outside the available epoch "
            f"window [{time_s[0]:.3f}, {time_s[-1]:.3f}] s."
        )
    return X[:, :, start:end], epoch_samples


# =============================================================================
# LOSO cross-validation
# =============================================================================

def train_loso(
    X: np.ndarray,
    y: np.ndarray,
    sub_id: np.ndarray,
    subjects: list[int],
    backend: str,
    n_filters: int,
    target_ambig: float,
) -> dict:
    """
    Leave-one-subject-out cross-validation.

    For each fold: train on all other subjects, evaluate on held-out subject.
    Dual thresholds are selected on the training set of each fold (no leakage).
    Final model is trained on all subjects.

    Returns a complete model bundle ready for pickle serialisation.
    """
    n_epochs, n_ch, n_samples = X.shape
    print(f"\n  ErrP Decoder — LOSO CV ({backend}, xDAWN n_filters={n_filters})")
    print(f"  Epochs: {n_epochs}  error={int(y.sum())}  correct={int((y==0).sum())}")
    print(f"  Shape:  {n_ch} ch × {n_samples} samples  ({n_samples/LIU_FS*1000:.0f} ms)")
    print(f"  Subjects: {subjects}\n")

    accs, aucs, t_lows, t_highs = [], [], [], []
    subject_results = []

    for held_out in subjects:
        te_mask = sub_id == held_out
        tr_mask = ~te_mask

        X_tr, y_tr = X[tr_mask], y[tr_mask]
        X_te, y_te = X[te_mask], y[te_mask]

        # Fit xDAWN on training subjects only.
        xdc = _fit_xdawn_covs(X_tr, y_tr, n_filters)
        C_tr = xdc.transform(X_tr)
        C_te = xdc.transform(X_te)

        # Fit classifier.
        clf = _build_classifier(backend)
        clf.fit(C_tr, y_tr)

        # Probabilities (column 1 = P(error)).
        prob_tr = clf.predict_proba(C_tr)[:, 1]
        prob_te = clf.predict_proba(C_te)[:, 1]

        # Argmax accuracy and AUC on held-out subject.
        y_pred = (prob_te >= 0.5).astype(int)
        acc = accuracy_score(y_te, y_pred)
        auc = roc_auc_score(y_te, prob_te)
        accs.append(acc)
        aucs.append(auc)

        # Dual-threshold selection on training scores (no leakage to test set).
        tl, th, _ = base.pick_dual_thresholds_target_ambiguity(
            y_tr, prob_tr,
            target_ambig=target_ambig,
            c_fp=1.0, c_fn=1.0,
        )
        t_lows.append(tl)
        t_highs.append(th)

        subject_results.append((held_out, acc, auc, tl, th, te_mask.sum()))
        print(f"  LOSO sub {held_out:2d}: acc={acc:.4f}  AUC={auc:.4f}  "
              f"tl={tl:.3f}  th={th:.3f}  n={te_mask.sum()}")

    tl_median = float(np.median(t_lows))
    th_median = float(np.median(t_highs))
    mean_acc  = float(np.mean(accs))
    mean_auc  = float(np.mean(aucs))

    print(f"\n  ── LOSO summary ─────────────────────────────────────────")
    print(f"  Mean accuracy : {mean_acc:.4f}")
    print(f"  Mean AUC      : {mean_auc:.4f}")
    print(f"  Median tl     : {tl_median:.3f}")
    print(f"  Median th     : {th_median:.3f}")
    print(f"  ─────────────────────────────────────────────────────────\n")

    # Final model: refit on ALL subjects.
    print("  Fitting final model on all subjects … ", end="", flush=True)
    xdc_final = _fit_xdawn_covs(X, y, n_filters)
    C_all     = xdc_final.transform(X)
    clf_final = _build_classifier(backend)
    clf_final.fit(C_all, y)
    print("done.")

    return {
        "xdawn":       xdc_final,
        "classifier":  clf_final,
        "backend":     backend,
        "tl_star":     tl_median,
        "th_star":     th_median,
        "roc_auc":     mean_auc,
        "cv_mean_acc": mean_acc,
        "feature_spec": {
            "tmin":          TMIN,
            "tmax":          TMAX,
            "n_filters":     n_filters,
            "epoch_samples": n_samples,
            # No XDF marker codes — this is an external dataset.
            # The bundle is used as a generic decoder; error/correct codes
            # are not needed by classify_errp_real_time().
            "error_codes":   [],
            "correct_codes": [],
        },
        "label_to_bin": {1: 1, 0: 0},
        "bin_to_label": {1: 1, 0: 0},
    }, subject_results


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mat", required=True,
                   help="Path to combinedEpochs_v2.mat")
    p.add_argument("--backend", default="xdawn_mdm",
                   choices=["xdawn_mdm", "xdawn_lr"],
                   help="Classifier backend (default: xdawn_mdm)")
    p.add_argument("--channels", default=None,
                   help="Comma-separated channel names to use. "
                        f"Default: Harmony ErrP channels ({HARMONY_ERRP_CHANNELS})")
    p.add_argument("--subjects", default=None,
                   help="Comma-separated 1-based subject indices to include. "
                        "Default: all 16.")
    p.add_argument("--tmin", type=float, default=TMIN,
                   help=f"Epoch start (s, default: {TMIN})")
    p.add_argument("--tmax", type=float, default=TMAX,
                   help=f"Epoch end   (s, default: {TMAX})")
    p.add_argument("--n-filters", type=int, default=N_FILTERS,
                   help=f"xDAWN spatial filters (default: {N_FILTERS})")
    p.add_argument("--target-ambig", type=float, default=TARGET_AMBIG,
                   help=f"Dual-threshold ambiguity budget (default: {TARGET_AMBIG})")
    p.add_argument("--out", default="models/errp_liu_generic.pkl",
                   help="Output path for the model bundle pickle "
                        "(default: models/errp_liu_generic.pkl)")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve channel list.
    if args.channels:
        channel_names = [c.strip() for c in args.channels.split(",")]
    else:
        channel_names = HARMONY_ERRP_CHANNELS

    # Resolve subject list.
    subjects = None
    if args.subjects:
        subjects = [int(s.strip()) for s in args.subjects.split(",")]

    print(f"\nLoading: {args.mat}")
    print(f"Channels: {channel_names}")

    X, y, mag, sub_id, time_s, loaded_subjects = load_liu_epochs(
        args.mat,
        subjects=subjects,
        channel_names=channel_names,
    )

    # Baseline-correct with pre-stimulus window, then crop to training window.
    X = _baseline_correct(X, time_s)
    X, epoch_samples = _crop_epoch_window(X, time_s, args.tmin, args.tmax)

    print(f"After crop: shape={X.shape}  "
          f"window=[{args.tmin}, {args.tmax}] s  "
          f"({epoch_samples} samples)")

    bundle, subject_results = train_loso(
        X, y, sub_id, loaded_subjects,
        backend=args.backend,
        n_filters=args.n_filters,
        target_ambig=args.target_ambig,
    )

    # Per-backend comparison if both are requested is deferred — run twice with
    # different --backend flags and compare the printed AUC values manually.

    # Save bundle.
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(bundle, fh)

    print(f"\n  Bundle saved: {out_path}")
    print(f"  backend={bundle['backend']}  "
          f"AUC={bundle['roc_auc']:.4f}  "
          f"acc={bundle['cv_mean_acc']:.4f}  "
          f"tl={bundle['tl_star']:.3f}  "
          f"th={bundle['th_star']:.3f}")
    print(f"  epoch_samples={epoch_samples}  "
          f"n_filters={args.n_filters}  "
          f"channels={channel_names}")

    # Print per-subject table.
    print("\n  Per-subject LOSO results:")
    print(f"  {'Sub':>4}  {'Acc':>6}  {'AUC':>6}  {'tl':>6}  {'th':>6}  {'n_te':>5}")
    for sub, acc, auc, tl, th, n_te in subject_results:
        print(f"  {sub:>4}  {acc:>6.4f}  {auc:>6.4f}  {tl:>6.3f}  {th:>6.3f}  {n_te:>5}")


if __name__ == "__main__":
    main()
