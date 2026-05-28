#!/usr/bin/env python3
"""Per-subject aggregate confusion matrices for the CLIN_* cohort.

Analog of `generate_per_subject_confusion_matrices.py` (which targets
the PILOT cohort, ses-S001ONLINE only). This version sums across all
ONLINE sessions and all ONLINE_* runs for each of CLIN_SUBJ_002..008.

Outputs (`~/Pictures/clin_analysis_pass1/confusion_matrices/`):
    <SUBJ>_aggregate_confusion_matrix.png
    cohort_aggregate_confusion_matrix.png    (CLIN_SUBJ_003..008 only;
        CLIN_SUBJ_002 excluded from cohort sum due to channel/shrinkage
        mismatch per rev01-paper-angle.md §1.1.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exploration.clinical_analysis._helpers import (  # noqa: E402
    DATA_DIR, clin_pictures_root, enumerate_clin_subjects,
    enumerate_online_sessions_for_subject,
)

from Analyze_experiment_logs_cross_subject import (  # noqa: E402
    compute_confusion_matrix_from_csv, find_decoder_csv,
)


def _plot_subject_cm(
    cm: np.ndarray, subject: str, n_sessions: int, n_runs: int,
    save_path: str,
):
    """Mirrors generate_per_subject_confusion_matrices.plot_subject_cm
    (lines 80-133) but with a per-subject (multi-session) heading."""
    row_totals = cm.sum(axis=1, keepdims=True)
    pct = np.where(row_totals > 0, 100.0 * cm / np.maximum(row_totals, 1), 0.0)

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("% of actual class", fontsize=11)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Pred MI", "Pred REST", "Ambiguous"], fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual MI", "Actual REST"], fontsize=12)
    for i in range(2):
        for j in range(3):
            ax.text(
                j, i, f"{cm[i, j]}\n({pct[i, j]:.1f}%)",
                ha="center", va="center",
                color="white" if pct[i, j] > 50 else "black", fontsize=12,
            )

    decisions_n = int(cm[:, :2].sum())
    correct_n = int(cm[0, 0] + cm[1, 1])
    ambiguous_n = int(cm[:, 2].sum())
    total_trials = int(cm.sum())
    dec_acc = (
        100.0 * correct_n / decisions_n if decisions_n else float("nan")
    )
    tot_acc = (
        100.0 * correct_n / total_trials if total_trials else float("nan")
    )

    ax.set_title(
        f"{subject} — all ONLINE sessions\n"
        f"Aggregate confusion matrix · {n_sessions} sessions · "
        f"{n_runs} runs · {total_trials} trials\n"
        f"Decision acc = {dec_acc:.1f}%   |   Total acc = {tot_acc:.1f}%   "
        f"|   Ambiguous = {ambiguous_n}/{total_trials}",
        fontsize=12, pad=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)


def _aggregate_subject_cm(subject: str) -> tuple[np.ndarray, int, int]:
    """Sum the 2x3 confusion matrix over all ONLINE sessions × runs."""
    cm_total = np.zeros((2, 3), dtype=int)
    n_sessions = 0
    n_runs = 0
    for sess in enumerate_online_sessions_for_subject(subject):
        logs_dir = Path(DATA_DIR) / f"sub-{subject}" / f"ses-{sess}" / "logs"
        if not logs_dir.is_dir():
            continue
        sess_used = False
        for run_dir in sorted(p for p in logs_dir.iterdir() if p.is_dir()
                               and p.name.startswith("ONLINE_")):
            csv = find_decoder_csv(str(run_dir))
            if csv is None:
                continue
            df = pd.read_csv(csv)
            if "Trial" not in df.columns:
                continue
            df["RunID"] = f"{subject}__{sess}__{run_dir.name}"
            df["GlobalTrialID"] = (
                df["RunID"].astype(str) + "_" + df["Trial"].astype(str)
            )
            cm_run = compute_confusion_matrix_from_csv(df)
            if cm_run is None:
                continue
            cm_total += cm_run
            n_runs += 1
            sess_used = True
        if sess_used:
            n_sessions += 1
    return cm_total, n_sessions, n_runs


def main():
    out_dir = clin_pictures_root() / "confusion_matrices"
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort_cm = np.zeros((2, 3), dtype=int)
    cohort_runs = 0
    cohort_sessions = 0
    cohort_subjects = 0

    for subject in enumerate_clin_subjects():
        cm, n_sess, n_runs = _aggregate_subject_cm(subject)
        if cm.sum() == 0:
            print(f"⚠️  {subject}: no usable runs; skip")
            continue
        print(
            f"\n--- {subject}: {n_sess} sessions, {n_runs} runs, "
            f"{cm.sum()} trials ---"
        )
        print("Rows: Actual [MI, REST] | Cols: Predicted [MI, REST, Ambiguous]")
        print(cm)
        save_path = out_dir / f"{subject}_aggregate_confusion_matrix.png"
        _plot_subject_cm(cm, subject, n_sess, n_runs, str(save_path))
        print(f"  wrote: {save_path.name}")

        # Cohort sum excludes CLIN_SUBJ_002 (different decoder channel
        # count / shrinkage; per rev01-paper-angle.md §1.1).
        if subject != "CLIN_SUBJ_002":
            cohort_cm += cm
            cohort_runs += n_runs
            cohort_sessions += n_sess
            cohort_subjects += 1

    if cohort_cm.sum() > 0:
        print(
            f"\n--- Cohort (CLIN_SUBJ_003..008): {cohort_subjects} subjects, "
            f"{cohort_sessions} sessions, {cohort_runs} runs, "
            f"{cohort_cm.sum()} trials ---"
        )
        print(cohort_cm)
        save_path = out_dir / "cohort_aggregate_confusion_matrix.png"
        _plot_subject_cm(
            cohort_cm, f"Cohort ({cohort_subjects} subjects)",
            cohort_sessions, cohort_runs, str(save_path),
        )
        print(f"  wrote: {save_path.name}")

    print(f"\nDone. Outputs at: {out_dir}")


if __name__ == "__main__":
    main()
