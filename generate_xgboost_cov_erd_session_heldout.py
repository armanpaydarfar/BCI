"""
Compatibility wrapper for cov+ERD threshold-free session-held-out benchmark.

For full apples-to-apples model comparison, use:
    python run_transfer_benchmark.py --models mdm,xgb_cov,xgb_cov_erd
"""

from __future__ import annotations

import os
from typing import Any

import mne
import numpy as np

os.environ["NUMBA_DISABLE_CACHING"] = "1"
os.environ["MNE_USE_NUMBA"] = "false"

import config
from Utils.transfer_benchmark_core import (
    SessionData,
    build_session_dataset,
    discover_training_files,
    run_session_heldout_benchmark,
)


def _print_config_context() -> None:
    train_dir = os.path.join(
        config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "training_data"
    )
    print("=" * 72)
    print("Session-held-out benchmark: XGBoost (cov + ERD features)")
    print("=" * 72)
    print(f"  TRAINING_SUBJECT : {config.TRAINING_SUBJECT}")
    print(f"  DATA_DIR         : {config.DATA_DIR}")
    print(f"  training_data    : {train_dir}")
    print(f"  SURFACE_LAPLACIAN_TOGGLE : {getattr(config, 'SURFACE_LAPLACIAN_TOGGLE', False)}")
    print(f"  CLASSIFY_WINDOW (ms)   : {getattr(config, 'CLASSIFY_WINDOW', '—')}")
    print(f"  Bandpass (Hz)          : {config.LOWCUT}–{config.HIGHCUT}")
    print(
        "  XGBoost                : "
        f"n_estimators={getattr(config, 'XGB_N_ESTIMATORS', 300)}, "
        f"max_depth={getattr(config, 'XGB_MAX_DEPTH', 5)}, "
        f"lr={getattr(config, 'XGB_LEARNING_RATE', 0.03)}"
    )
    print(
        f"  XGB_USE_COV_MU={getattr(config, 'XGB_USE_COV_MU', 1)}  "
        f"XGB_USE_COV_BETA={getattr(config, 'XGB_USE_COV_BETA', 1)}"
    )
    print("=" * 72)


def _print_training_files(paths: list[str]) -> None:
    print("\n[1/3] Training .xdf files (session-held-out folds = one per file)")
    print("-" * 72)
    for i, p in enumerate(paths, 1):
        print(f"  {i:>3}. {os.path.basename(p)}")
        print(f"       {p}")
    print(f"\n  Total: {len(paths)} file(s). Need ≥2 for benchmark.")
    print("-" * 72)


def _print_built_sessions(sessions: list[SessionData], rest_label: int, mi_label: int) -> None:
    print("\n[2/3] Built session dataset (windows per file)")
    print("-" * 72)
    print(
        f"  {'Session':<42} {'n_win':>8} {'REST':>8} {'MI':>8} "
        f"{'feat_erd':>10}"
    )
    for s in sessions:
        y = s.labels
        n_rest = int(np.sum(y == rest_label))
        n_mi = int(np.sum(y == mi_label))
        erd_dim = 0 if s.erd is None else s.erd.shape[1]
        print(
            f"  {s.session_name:<42} {len(y):>8} {n_rest:>8} {n_mi:>8} "
            f"{erd_dim:>10}"
        )
    print("-" * 72)
    print(
        "  Note: Covariance / tangent extraction may print per-session diagnostics "
        "from Generate_Riemannian_adaptive."
    )


def _infer_class_labels(sessions: list[SessionData]) -> tuple[int, int]:
    labels_all = np.concatenate([s.labels for s in sessions], axis=0)
    classes = np.sort(np.unique(labels_all))
    if len(classes) != 2:
        raise ValueError("Binary labels required.")
    return int(classes[0]), int(classes[1])


def _print_aggregate_report(
    results: dict[str, dict[str, Any]],
    *,
    rest_label: int,
    mi_label: int,
) -> None:
    print("\n" + "=" * 72)
    print("AGGREGATE REPORT — session-held-out transfer (XGB cov+ERD)")
    print("=" * 72)

    for model_name, payload in results.items():
        agg = payload["aggregate"]
        folds: list[dict] = payload["folds"]
        cm = agg["cm"]

        print(f"\nModel: {model_name}")
        print("-" * 72)
        print("  Pooling: all held-out predictions across folds (one test session per fold).")
        print(
            f"  Accuracy          : {agg['acc']:.4f}\n"
            f"  Balanced accuracy : {agg['bal_acc']:.4f}\n"
            f"  Macro F1          : {agg['macro_f1']:.4f}\n"
            f"  ROC AUC (MI score): {agg['auc']:.4f}\n"
            f"  Brier (P_MI)      : {agg['brier']:.4f}"
        )
        print(
            f"\n  Confusion [rows=true REST({rest_label}), MI({mi_label}) | "
            f"cols=pred REST, MI]:\n"
            f"    {cm}"
        )

        aucs = [float(f["auc"]) for f in folds if not np.isnan(f["auc"])]
        valid_folds = [f for f in folds if not np.isnan(f["auc"])]
        if valid_folds:
            best = max(valid_folds, key=lambda f: float(f["auc"]))
            worst = min(valid_folds, key=lambda f: float(f["auc"]))
        else:
            best = worst = folds[0]

        print("\n  Per-session held-out (test = that file, train = all others)")
        print(
            f"    {'Fold':<5} {'Session':<36} {'n_test':>7} {'AUC':>8} "
            f"{'BalAcc':>8} {'Brier':>8}"
        )
        for f in folds:
            print(
                f"    {f['fold']:<5} {f['session_name'][:36]:<36} {f['n_test']:>7} "
                f"{f['auc']:>8.4f} {f['bal_acc']:>8.4f} {f['brier']:>8.4f}"
            )

        if aucs:
            print(
                f"\n  Fold AUC: mean={np.mean(aucs):.4f}  std={np.std(aucs):.4f}  "
                f"min={np.min(aucs):.4f}  max={np.max(aucs):.4f}"
            )
        print(f"  Best held-out session (by AUC) : {best['session_name']} (AUC={best['auc']:.4f})")
        print(f"  Worst held-out session (by AUC): {worst['session_name']} (AUC={worst['auc']:.4f})")

        # Short interpretation
        print("\n  Interpretation (heuristic)")
        if agg["auc"] >= 0.75:
            print("    • Pooled AUC suggests strong cross-session separability for this backend.")
        elif agg["auc"] >= 0.65:
            print("    • Pooled AUC is moderate; transfer varies by session (see fold spread).")
        else:
            print("    • Pooled AUC is weak; consider more/better training sessions or feature review.")

        spread = float(np.std(aucs)) if len(aucs) > 1 else 0.0
        if len(aucs) > 1 and spread > 0.12:
            print(
                "    • High fold-to-fold AUC spread → session-specific shifts dominate; "
                "recentering/adaptation may matter at runtime."
            )
        elif len(aucs) > 1 and spread <= 0.08:
            print("    • Relatively stable AUC across sessions → consistent transfer across files.")

    print("\n" + "=" * 72)
    print("Done.")
    print("=" * 72)


def main() -> None:
    mne.set_log_level("WARNING")

    _print_config_context()

    xdf_paths = discover_training_files()
    _print_training_files(xdf_paths)

    print("\n  Building features (load XDF → segment → cov + ERD → tangent space) …")
    sessions = build_session_dataset(
        xdf_paths,
        include_beta_cov=True,
        include_erd=True,
    )
    rest_label, mi_label = _infer_class_labels(sessions)
    _print_built_sessions(sessions, rest_label, mi_label)

    print("\n[3/3] Session-held-out evaluation (XGBoost cov+ERD)")
    print("-" * 72)
    results = run_session_heldout_benchmark(
        model_names=["xgb_cov_erd"],
        sessions=sessions,
        print_importance=True,
    )
    _print_aggregate_report(results, rest_label=rest_label, mi_label=mi_label)


if __name__ == "__main__":
    main()
