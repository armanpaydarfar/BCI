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
import ctypes
import ctypes.util
import gc
import os
import pickle
import resource
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


# glibc malloc_trim(0) nudges malloc to return freed arenas to the OS.
# gc.collect() alone frees Python objects but keeps the pages in the
# glibc arena cache — on a 16 GiB box with 32 LOSO folds that cache can
# grow past the swap ceiling and OOM mid-loop even though Python's live
# working set is small. We call this between folds.
_LIBC = None
try:
    _libc_path = ctypes.util.find_library("c")
    if _libc_path:
        _LIBC = ctypes.CDLL(_libc_path)
except OSError:
    _LIBC = None


def _malloc_trim() -> None:
    if _LIBC is not None and hasattr(_LIBC, "malloc_trim"):
        try:
            _LIBC.malloc_trim(0)
        except Exception:
            pass


def _rss_gb() -> float:
    """Current RSS in GiB via /proc/self/status (Linux-only; falls back to getrusage)."""
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    kb = float(line.split()[1])
                    return kb / (1024.0 * 1024.0)
    except Exception:
        pass
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / (1024.0 * 1024.0)

import config
import Generate_Riemannian_adaptive as base
from generate_errp_decoder import _build_classifier, _fit_xdawn_covs
from Utils.errp_alignment import apply_ea, fit_ea_reference
from Utils.errp_liu_pipeline import LiuCCAFeaturizer
from Utils.liu_data_loader import (
    HARMONY_ERRP_CHANNELS,
    LIU_FS,
    load_liu_epochs,
)


_XDAWN_BACKENDS = {"xdawn_mdm", "xdawn_lr", "xdawn_xgb"}
_CCA_BACKENDS   = {"liu_cca_lda", "liu_cca_xgb"}


def _fit_feature_extractor(backend: str, X: np.ndarray, y: np.ndarray, n_filters: int):
    """Fit the spatial-filter/featurizer stage for the chosen backend."""
    if backend in _XDAWN_BACKENDS:
        return _fit_xdawn_covs(X, y, n_filters)
    if backend in _CCA_BACKENDS:
        return LiuCCAFeaturizer(n_components=3, fs=LIU_FS).fit(X, y)
    raise ValueError(f"Unknown backend: {backend!r}")


def _bundle_feature_key(backend: str) -> str:
    """Top-level bundle key under which the fitted feature extractor is saved."""
    if backend in _XDAWN_BACKENDS:
        return "xdawn"
    if backend in _CCA_BACKENDS:
        return "cca_featurizer"
    raise ValueError(f"Unknown backend: {backend!r}")

# =============================================================================
# Epoch window (matching config and our existing pipeline convention)
# =============================================================================
TMIN = float(getattr(config, "ERRP_EPOCH_TMIN", 0.0))   # 0.0 s
TMAX = float(getattr(config, "ERRP_EPOCH_TMAX", 0.8))   # 0.8 s
N_FILTERS = int(getattr(config, "ERRP_XDAWN_N_FILTERS", 4))
TARGET_AMBIG = float(getattr(config, "ERRP_TARGET_AMBIG", 0.20))

# Pre-stimulus baseline window for offline correction (sourced from config).
_BASELINE_TMIN = float(getattr(config, "ERRP_BASELINE_TMIN", -0.200))  # s
_BASELINE_TMAX = float(getattr(config, "ERRP_BASELINE_TMAX",  0.0))    # s


def _baseline_correct(X: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """Subtract per-epoch mean of [ERRP_BASELINE_TMIN, ERRP_BASELINE_TMAX] s."""
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

def _apply_ea_per_subject(X: np.ndarray, sub_id: np.ndarray, subjects: list[int]) -> np.ndarray:
    """
    Per-subject Euclidean Alignment, in place by subject group.

    Mutates X directly — callers pass either a fresh fancy-index copy
    (X_tr/X_te inside the LOSO loop) or the pooled X at the final-fit
    step (which is not reused afterwards). Avoiding an extra full-size
    copy here saves several GiB at peak on 32-subject pools.
    """
    for s in subjects:
        mask = (sub_id == s)
        if not mask.any():
            continue
        ref = fit_ea_reference(X[mask])
        X[mask] = apply_ea(X[mask], ref)
    return X


def train_loso(
    X: np.ndarray,
    y: np.ndarray,
    sub_id: np.ndarray,
    subjects: list[int],
    backend: str,
    n_filters: int,
    target_ambig: float,
    use_ea: bool = True,
) -> dict:
    """
    Leave-one-subject-out cross-validation.

    For each fold: train on all other subjects, evaluate on held-out subject.
    Dual thresholds are selected on the training set of each fold (no leakage).
    Final model is trained on all subjects.

    Returns a complete model bundle ready for pickle serialisation.
    """
    n_epochs, n_ch, n_samples = X.shape
    print(f"\n  ErrP Decoder — LOSO CV ({backend}, xDAWN n_filters={n_filters}, EA={'on' if use_ea else 'off'})")
    print(f"  Epochs: {n_epochs}  error={int(y.sum())}  correct={int((y==0).sum())}")
    print(f"  Shape:  {n_ch} ch × {n_samples} samples  ({n_samples/LIU_FS*1000:.0f} ms)")
    print(f"  Subjects: {subjects}")
    print(f"  RSS at LOSO start: {_rss_gb():.2f} GiB\n")

    accs, aucs, t_lows, t_highs = [], [], [], []
    subject_results = []
    # Per-fold held-out predictions, keyed by held-out subject ID.  The
    # ensemble in Phase 8 averages these across heads — saving them here
    # removes the need to retrain three heads inside the ensemble script.
    cv_predictions: dict[int, dict] = {}

    for held_out in subjects:
        te_mask = sub_id == held_out
        tr_mask = ~te_mask

        X_tr, y_tr = X[tr_mask], y[tr_mask]
        X_te, y_te = X[te_mask], y[te_mask]
        sub_id_tr  = sub_id[tr_mask]
        train_subjects = [s for s in subjects if s != held_out]

        # Per-subject Euclidean Alignment.  Each training subject's epochs
        # are whitened by their own mean covariance; the held-out subject is
        # whitened from its unlabelled epochs (no use of y_te).  This is the
        # exact LOSO simulation of the runtime EA bootstrap — at session
        # start the new subject's reference comes from a 30–60 s neutral
        # recording.
        if use_ea:
            X_tr = _apply_ea_per_subject(X_tr, sub_id_tr, train_subjects)
            X_te = _apply_ea_per_subject(X_te, np.full(len(X_te), held_out), [held_out])

        # Fit spatial-filter/featurizer stage on training subjects only.
        # xdawn_* backends return an XdawnCovariances; liu_cca_* backends
        # return a fitted LiuCCAFeaturizer.  Both expose .transform(X).
        feat = _fit_feature_extractor(backend, X_tr, y_tr, n_filters)
        F_tr = feat.transform(X_tr)
        F_te = feat.transform(X_te)

        # Fit classifier.  xgb-based backends need sub_id_tr for their
        # grouped calibration inner CV; the other backends ignore the kwarg.
        clf = _build_classifier(backend, sub_id_tr=sub_id_tr)
        clf.fit(F_tr, y_tr)

        # Probabilities (column 1 = P(error)).
        prob_tr = clf.predict_proba(F_tr)[:, 1]
        prob_te = clf.predict_proba(F_te)[:, 1]

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

        # Persist held-out predictions for ensemble aggregation (Phase 8).
        cv_predictions[int(held_out)] = {
            "y_true":     y_te.astype(np.int8).copy(),
            "prob_error": prob_te.astype(np.float32).copy(),
        }

        # Free per-fold arrays — without this the process OOMs around
        # fold 10 of 16 at 14 channels due to non-returning malloc arenas
        # in pyriemann's xDAWN+OAS path.
        del feat, F_tr, F_te, clf, X_tr, X_te, y_tr, y_te, prob_tr, prob_te
        del sub_id_tr, train_subjects
        gc.collect()
        _malloc_trim()

        rss = _rss_gb()
        print(f"  LOSO sub {held_out:2d}: acc={acc:.4f}  AUC={auc:.4f}  "
              f"tl={tl:.3f}  th={th:.3f}  n={te_mask.sum():4d}  rss={rss:.2f}G")

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

    # Final model: refit on ALL subjects.  EA is applied per training
    # subject before the feature-extractor fit so the operator the runtime
    # applies (per-session EA → spatial filter → classifier) is mirrored
    # end-to-end.
    print("  Fitting final model on all subjects … ", end="", flush=True)
    if use_ea:
        X_final = _apply_ea_per_subject(X, sub_id, subjects)
    else:
        X_final = X
    feat_final = _fit_feature_extractor(backend, X_final, y, n_filters)
    F_all = feat_final.transform(X_final)
    clf_final = _build_classifier(backend, sub_id_tr=sub_id)
    clf_final.fit(F_all, y)
    print("done.")

    feature_spec = {
        "tmin":          TMIN,
        "tmax":          TMAX,
        "n_filters":     n_filters,
        "ea_alignment":  bool(use_ea),
        "epoch_samples": n_samples,
        # No XDF marker codes — this is an external dataset.
        # The bundle is used as a generic decoder; error/correct codes
        # are not needed by classify_errp_real_time().
        "error_codes":   [],
        "correct_codes": [],
    }
    if backend in _CCA_BACKENDS:
        feature_spec["psd_bin_idx"] = feat_final.psd_bin_idx_.tolist()
        feature_spec["psd_freqs"]   = list(feat_final.psd_freqs)
        feature_spec["cca_n_components"] = feat_final.n_components
        feature_spec["cca_temporal_samples"] = feat_final.n_temporal_

    bundle = {
        _bundle_feature_key(backend): feat_final,
        "classifier":  clf_final,
        "backend":     backend,
        "tl_star":     tl_median,
        "th_star":     th_median,
        "roc_auc":     mean_auc,
        "cv_mean_acc": mean_acc,
        "feature_spec": feature_spec,
        "label_to_bin": {1: 1, 0: 0},
        "bin_to_label": {1: 1, 0: 0},
        "cv_predictions": cv_predictions,
    }
    return bundle, subject_results


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mat", required=True, nargs="+",
                   help="Path(s) to .mat file(s). Pass multiple paths to pool datasets. "
                        "When pooling, epochs from the same subject ID across files are "
                        "held out together in LOSO, so the generalisation estimate remains "
                        "subject-level (not condition-level).")
    p.add_argument("--backend", default="xdawn_mdm",
                   choices=["xdawn_mdm", "xdawn_lr", "xdawn_xgb",
                            "liu_cca_lda", "liu_cca_xgb"],
                   help="Classifier backend (default: xdawn_mdm). "
                        "xdawn_* runs xDAWN+classifier on augmented covariances; "
                        "liu_cca_* runs Liu et al. 2025 CCA+handcrafted features "
                        "with a diagonal LDA or calibrated XGB head.")
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
    p.add_argument("--car", dest="car", action="store_true", default=None,
                   help="Apply Common Average Reference across the selected "
                        "channels after loading and before baseline correction. "
                        "Default: on (follow config.ERRP_CAR_REREFERENCE).")
    p.add_argument("--no-car", dest="car", action="store_false",
                   help="Disable CAR (for A/B testing against the raw-reference baseline).")
    p.add_argument("--ea", dest="ea", action="store_true", default=True,
                   help="Apply per-subject Euclidean Alignment inside the LOSO loop "
                        "(default: on). The held-out subject is aligned from its own "
                        "unlabelled epochs, simulating the runtime EA bootstrap.")
    p.add_argument("--no-ea", dest="ea", action="store_false",
                   help="Disable EA (for A/B testing against the un-aligned baseline).")
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

    # Resolve CAR toggle: CLI wins, otherwise follow config.
    if args.car is None:
        car = bool(int(getattr(config, "ERRP_CAR_REREFERENCE", 1)))
    else:
        car = bool(args.car)

    print(f"\nChannels: {channel_names}")
    print(f"CAR:      {'on' if car else 'off'}")
    print(f"EA:       {'on' if args.ea else 'off'}")

    # Load one or more .mat files and concatenate.  The two Liu .mat files
    # contain non-overlapping cohorts (Experiment 1 / control: 16 subjects;
    # Experiment 2 / BCI: 16 different subjects — Liu et al. 2025 p. 4).  To
    # keep LOSO honest we must map per-file row indices to globally unique IDs
    # so a single fold holds out exactly one person.  We do this by offsetting
    # each file's sub_id by the cumulative max ID seen so far.  Result: 32
    # distinct subjects (when both files are passed), 32-fold LOSO with no
    # identity leak.
    Xs, ys, mags, sub_ids = [], [], [], []
    loaded_subjects: list[int] = []
    id_offset = 0
    for mat_path in args.mat:
        print(f"Loading: {mat_path}")
        Xi, yi, magi, sub_id_i, time_s, subs_i = load_liu_epochs(
            mat_path,
            subjects=subjects,
            channel_names=channel_names,
            car=car,
        )
        Xi = _baseline_correct(Xi, time_s)
        Xi, epoch_samples = _crop_epoch_window(Xi, time_s, args.tmin, args.tmax)
        # Detach the cropped slice from its (much larger) full-window parent
        # so the pre-crop array can be garbage-collected before the next file.
        # Downcast to float32 here — the signal is already 1–10 Hz bandpassed,
        # the downstream xDAWN/Riemannian operators use eigendecompositions
        # with shrinkage floors, and the 2× memory saving is the difference
        # between the 32-subject pooled LOSO fitting in a 16 GiB box and
        # tripping the OOM killer at fold 1 of 32.
        Xi = np.ascontiguousarray(Xi, dtype=np.float32)

        sub_id_i_global = sub_id_i + id_offset
        subs_i_global   = [s + id_offset for s in subs_i]
        loaded_subjects.extend(subs_i_global)
        id_offset += max(subs_i)

        Xs.append(Xi)
        ys.append(yi)
        mags.append(magi)
        sub_ids.append(sub_id_i_global)
        print(f"  → {Xi.shape[0]} epochs after crop  "
              f"(global subject IDs {subs_i_global[0]}-{subs_i_global[-1]})")

    X      = np.concatenate(Xs,      axis=0)
    y      = np.concatenate(ys)
    sub_id = np.concatenate(sub_ids)

    # Drop the per-file lists — after concatenation each one still holds a
    # full copy of that cohort's epochs, doubling the pooled footprint.
    Xs.clear(); ys.clear(); sub_ids.clear(); mags.clear()
    del Xs, ys, sub_ids, mags
    gc.collect()

    print(f"\nPooled: shape={X.shape}  dtype={X.dtype}  "
          f"subjects={len(loaded_subjects)}  "
          f"window=[{args.tmin}, {args.tmax}] s  "
          f"({epoch_samples} samples)")

    bundle, subject_results = train_loso(
        X, y, sub_id, loaded_subjects,
        backend=args.backend,
        n_filters=args.n_filters,
        target_ambig=args.target_ambig,
        use_ea=bool(args.ea),
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
