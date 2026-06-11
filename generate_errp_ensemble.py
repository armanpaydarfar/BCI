"""
generate_errp_ensemble.py — Assemble a three-head ErrP ensemble bundle.

Takes the three head bundles produced by ``generate_errp_decoder_liu.py``
(Phase 6 xDAWN+XGB, Phase 7A Liu-CCA+diag-LDA, Phase 7B Liu-CCA+XGB) and
combines them into a single wrapper that ``runtime_common`` can dispatch.

No retraining here.  The sub-bundles already carry per-fold held-out
predictions in their ``cv_predictions`` dict, keyed by held-out subject
ID.  For each subject we average the three heads' ``prob_error`` vectors
elementwise, giving an honest LOSO ensemble score with zero leakage.
From those pooled ensemble scores we compute LOSO AUC and re-run dual-
threshold selection (individual-head thresholds do not transfer).

Ensemble rule: equal-weight probability averaging,
  p_ensemble = (p_xgb_xdawn + p_lda_liu + p_xgb_liu) / 3.

All three heads must expose calibrated probabilities — the two XGB heads
use CalibratedClassifierCV(isotonic); the diagonal-LDA head outputs a
closed-form Gaussian posterior and is calibrated by construction.

Reference: ErrP_Cross_Subject_Decoder_Plan.md §Phase 8.
"""

import argparse
import os
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

import Generate_Riemannian_adaptive as base


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_HEAD_PATHS = {
    "xdawn_xgb":   "models/errp_liu_xdawn_xgb.pkl",
    "liu_cca_lda": "models/errp_liu_cca_lda.pkl",
    "liu_cca_xgb": "models/errp_liu_cca_xgb.pkl",
}

DEFAULT_OUT_PATH  = "models/errp_liu_ensemble.pkl"
DEFAULT_AMBIG     = 0.20


# -----------------------------------------------------------------------------
# Core logic
# -----------------------------------------------------------------------------

def _load_head(path: str, expected_backend: str) -> dict:
    """Load a head bundle and verify it matches the expected backend tag."""
    with open(path, "rb") as fh:
        b = pickle.load(fh)
    got = b.get("backend")
    if got != expected_backend:
        raise ValueError(
            f"Bundle {path!r} has backend={got!r}, expected {expected_backend!r}."
        )
    if "cv_predictions" not in b:
        raise ValueError(
            f"Bundle {path!r} has no 'cv_predictions' — it was produced before "
            f"Phase 8 wiring.  Retrain with the current generate_errp_decoder_liu.py."
        )
    return b


def _ensemble_predictions(heads: list[dict]) -> tuple[dict, list[int]]:
    """Elementwise-average probabilities from each head's cv_predictions.

    Returns ``(ensemble_cv, subjects)`` where ``ensemble_cv`` maps held-out
    subject ID to ``{"y_true", "prob_error"}``.  Raises if the three heads
    disagree on which subjects / trial counts / labels are held out.
    """
    cv_sets = [h["cv_predictions"] for h in heads]

    subj_lists = [sorted(d.keys()) for d in cv_sets]
    if not all(s == subj_lists[0] for s in subj_lists[1:]):
        raise ValueError(
            f"Heads hold out different subjects: {subj_lists}"
        )
    subjects = subj_lists[0]

    ensemble_cv: dict[int, dict] = {}
    for s in subjects:
        ys = [np.asarray(d[s]["y_true"],     dtype=np.int8)   for d in cv_sets]
        ps = [np.asarray(d[s]["prob_error"], dtype=np.float64) for d in cv_sets]

        n0 = ys[0].shape[0]
        if not all(y.shape[0] == n0 for y in ys) or not all(p.shape[0] == n0 for p in ps):
            raise ValueError(
                f"Subject {s}: head predictions disagree on trial count "
                f"({[y.shape[0] for y in ys]} vs {[p.shape[0] for p in ps]})."
            )
        if not all(np.array_equal(y, ys[0]) for y in ys[1:]):
            raise ValueError(
                f"Subject {s}: heads disagree on y_true — did they see different "
                f"epochs? (length matches but values differ)."
            )

        p_mean = np.mean(ps, axis=0).astype(np.float32)
        ensemble_cv[int(s)] = {
            "y_true":     ys[0],
            "prob_error": p_mean,
        }
    return ensemble_cv, subjects


def _summarise(ensemble_cv: dict, subjects: list[int]) -> dict:
    """Per-subject + pooled LOSO AUC/acc on the ensemble scores."""
    rows = []
    ys_all, ps_all = [], []
    for s in subjects:
        y = ensemble_cv[s]["y_true"]
        p = ensemble_cv[s]["prob_error"]
        yhat = (p >= 0.5).astype(int)
        acc  = accuracy_score(y, yhat)
        auc  = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
        rows.append((s, acc, auc, int(len(y))))
        ys_all.append(y); ps_all.append(p)

    y_cat = np.concatenate(ys_all)
    p_cat = np.concatenate(ps_all)
    pooled = {
        "mean_auc_per_subject": float(np.nanmean([r[2] for r in rows])),
        "mean_acc_per_subject": float(np.mean ([r[1] for r in rows])),
        "pooled_auc":           float(roc_auc_score(y_cat, p_cat)),
        "pooled_acc":           float(accuracy_score(y_cat, (p_cat >= 0.5).astype(int))),
        "rows":                 rows,
        "y_pooled":             y_cat,
        "p_pooled":             p_cat,
    }
    return pooled


def _pick_ensemble_thresholds(y: np.ndarray, p: np.ndarray, target_ambig: float) -> tuple[float, float, dict]:
    """Dual-threshold selection on pooled LOSO ensemble scores."""
    tl, th, diag = base.pick_dual_thresholds_target_ambiguity(
        y_true_bin=y,
        pos_scores=p,
        target_ambig=target_ambig,
        c_fp=1.0, c_fn=1.0,
    )
    return float(tl), float(th), diag


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xdawn-xgb",   default=DEFAULT_HEAD_PATHS["xdawn_xgb"])
    p.add_argument("--liu-cca-lda", default=DEFAULT_HEAD_PATHS["liu_cca_lda"])
    p.add_argument("--liu-cca-xgb", default=DEFAULT_HEAD_PATHS["liu_cca_xgb"])
    p.add_argument("--out",         default=DEFAULT_OUT_PATH)
    p.add_argument("--target-ambig", type=float, default=DEFAULT_AMBIG)
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading heads …")
    h1 = _load_head(args.xdawn_xgb,   "xdawn_xgb")
    h2 = _load_head(args.liu_cca_lda, "liu_cca_lda")
    h3 = _load_head(args.liu_cca_xgb, "liu_cca_xgb")

    print("Averaging per-fold predictions …")
    ensemble_cv, subjects = _ensemble_predictions([h1, h2, h3])

    summ = _summarise(ensemble_cv, subjects)
    tl, th, diag = _pick_ensemble_thresholds(summ["y_pooled"], summ["p_pooled"],
                                             args.target_ambig)

    # Feature spec — we carry only the fields the runtime dispatch needs
    # to validate the incoming epoch shape.  All three heads share the
    # same CAR / baseline / EA / [tmin, tmax] stages by construction
    # (they were trained from the same loader).  We assert agreement and
    # then keep one copy.
    spec_keys = ("tmin", "tmax", "epoch_samples", "ea_alignment")
    specs = [h.get("feature_spec", {}) for h in (h1, h2, h3)]
    for k in spec_keys:
        vals = [s.get(k) for s in specs]
        if not all(v == vals[0] for v in vals[1:]):
            raise ValueError(
                f"Heads disagree on feature_spec[{k!r}]: {vals}.  Retrain with a "
                f"consistent loader before ensembling."
            )
    shared_spec = {k: specs[0].get(k) for k in spec_keys}

    bundle = {
        "backend":        "errp_ensemble",
        "head_xdawn_xgb": h1,
        "head_liu_lda":   h2,
        "head_liu_xgb":   h3,
        "weights":        [1/3, 1/3, 1/3],
        "tl_star":        tl,
        "th_star":        th,
        "roc_auc":        summ["pooled_auc"],
        "cv_mean_acc":    summ["mean_acc_per_subject"],
        "cv_predictions": ensemble_cv,
        "feature_spec":   shared_spec,
        "label_to_bin":   {1: 1, 0: 0},
        "bin_to_label":   {1: 1, 0: 0},
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(bundle, fh)

    print(f"\n  ── Ensemble summary ────────────────────────────────────")
    print(f"  Pooled AUC       : {summ['pooled_auc']:.4f}")
    print(f"  Mean AUC / subj  : {summ['mean_auc_per_subject']:.4f}")
    print(f"  Mean acc / subj  : {summ['mean_acc_per_subject']:.4f}")
    print(f"  tl_star, th_star : {tl:.3f}, {th:.3f}   "
          f"(ambig target {args.target_ambig:.2f}, feasible={diag.get('feasible', '?')})")
    print(f"  ────────────────────────────────────────────────────────")
    print(f"\n  Per-subject LOSO (ensemble):")
    print(f"  {'Sub':>4}  {'Acc':>6}  {'AUC':>6}  {'n_te':>5}")
    for sub, acc, auc, n_te in summ["rows"]:
        auc_s = f"{auc:.4f}" if np.isfinite(auc) else "  nan "
        print(f"  {sub:>4}  {acc:>6.4f}  {auc_s:>6}  {n_te:>5}")

    print(f"\n  Bundle saved: {args.out}")


if __name__ == "__main__":
    main()
