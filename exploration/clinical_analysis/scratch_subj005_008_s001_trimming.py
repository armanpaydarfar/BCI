#!/usr/bin/env python3
"""Investigation B — SUBJ_005 & SUBJ_008 S001 MI-trial trimming.

Hypothesis: S001 (BCI-naive) MI desync is dragged toward zero by early
trials, before the subject converges on a neuromodulation strategy.
Dropping the first 25/50/75% of S001 MI trials in chronological order
should pull the kept-trial median ERD% closer to S002-S005.

Trimming ORDER decision: trim AFTER canonical artifact rejection.

  Justification from code:
   * Analyze_clinical_erd_refined.py:158-263  _reject_artifact_trials
     mutates tfr_trials in place; the per-marker keep_mask is built
     from a MAD-based robust z on per-trial peak |ERD%| over the
     BILATERAL_MOTOR_CLUSTER (line 210), with an absolute cap.
   * If we trim BEFORE rejection, the MAD (line 232) is computed on a
     subset (the late half of the session), so the rejection mask is
     a different set of "outliers" than the canonical pipeline drops.
     We'd conflate "BCI-naive learning curve" with "MAD-set shift".
   * Trimming AFTER rejection preserves the canonical artifact mask —
     blow-up trials anywhere in the session are still removed — and
     the trimming becomes a pure temporal-trial filter applied to the
     already-clean survivor set. This is the order we use.

G2 over-rejection accounting:
   * evaluate_erd_quality.py:117  G2_OVERREJECT_FRAC = 0.50
   * evaluate_erd_quality.py:478-482  drop_frac = 1 - n_after_reject /
     n_attempted; gate trips when drop_frac > 0.50.
   * n_attempted stays at the original event count (typically 100);
     n_after_reject is the post-rejection + post-trimming total. So
     drop_first_75pct on a 100-event session with ~3 MI artifact drops
     gives drop_frac ≈ (37 + 3 + small REST drops) / 100 ≈ 0.40-0.45,
     within G2's 0.50 ceiling. drop_first_50pct sits at ≈ 0.28.

Subjects: CLIN_SUBJ_005, CLIN_SUBJ_008 only.
Sessions: all 5 ONLINE sessions per subject (canonical baseline for
S002-S005; S001 also gets the trimming variants).

Variants:
   V0  canonical_full          (reference; reproduces canonical CSV)
   V25 drop_first_25pct        (S001 only; MI trials only)
   V50 drop_first_50pct        (S001 only; MI trials only)
   V75 drop_first_75pct        (S001 only; MI trials only)

Outputs (scratch only — no canonical edits, no commits):
  C:\\Users\\arman\\Pictures\\clin_analysis_subj005_008_s001_trimming\\
    per_trial/
      CLIN_SUBJ_005_S00NONLINE_V0.npz       (all 5 sessions)
      CLIN_SUBJ_005_S001ONLINE_V25.npz      (S001 only)
      CLIN_SUBJ_005_S001ONLINE_V50.npz      (S001 only)
      CLIN_SUBJ_005_S001ONLINE_V75.npz      (S001 only)
      (same for CLIN_SUBJ_008)
    erd_quality_scores_V0.csv / .json
    erd_quality_scores_V25.csv / .json
    erd_quality_scores_V50.csv / .json
    erd_quality_scores_V75.csv / .json
"""

from __future__ import annotations

import csv
import gc
import json
import sys
import time
import warnings
from pathlib import Path

import mne
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SWEEP_DIR = _REPO_ROOT / "exploration" / "preprocessing_sweep"
for _p in (str(_REPO_ROOT), str(_SWEEP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

from Analyze_clinical_erd_refined import (  # noqa: E402
    CONFIG_A_DISPLAY_BASELINE, MI_MARKER,
    _extract_session_traces, _reject_artifact_trials, _write_per_trial_npz,
    config_a_pipeline,
)
from evaluate_erd_quality import _CSV_FIELDS, score_dir  # noqa: E402


SUBJECTS = ["CLIN_SUBJ_005", "CLIN_SUBJ_008"]
SESSIONS = [f"S{n:03d}ONLINE" for n in range(1, 6)]
INTERVENTION_SESSION = "S001ONLINE"

OUT_DIR = Path(
    r"C:\Users\arman\Pictures\clin_analysis_subj005_008_s001_trimming"
)
PER_TRIAL_DIR = OUT_DIR / "per_trial"
ERD_REFINED_DIR = OUT_DIR / "erd_refined"
for d in (OUT_DIR, PER_TRIAL_DIR, ERD_REFINED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Variant tag -> proportion of MI trials to drop from chronological start
TRIM_VARIANTS = {
    "V25": 0.25,
    "V50": 0.50,
    "V75": 0.75,
}


def _trim_mi_first_p(tfr_trials, p):
    """Drop the first ceil(n_mi * p) MI trials in chronological order
    from a post-rejection tfr_trials dict.

    REST (marker 100) is untouched per the handoff ("no 'learning to
    rest'"). Returns a new dict (MI trimmed, REST original); the input
    is not mutated. The trimming substrate is the post-rejection MI
    survivor set — see module docstring for the order justification.
    """
    out = dict(tfr_trials)
    if MI_MARKER not in out:
        return out
    tfr_mi = out[MI_MARKER]
    n_mi = int(tfr_mi.data.shape[0])
    if n_mi == 0:
        return out
    n_drop = int(np.ceil(n_mi * p))
    if n_drop >= n_mi:
        # Would empty MI entirely — leave the npz writer to skip the key,
        # which the scorer interprets as a lost cluster (G4).
        n_drop = n_mi
    keep_idx = np.arange(n_drop, n_mi)
    if keep_idx.size == 0:
        # Drop the MI key entirely so the npz writer omits it cleanly.
        del out[MI_MARKER]
        return out
    out[MI_MARKER] = tfr_mi[keep_idx]
    return out


def _write_variant_npz(out_path, subject, session, tfr_trials,
                       dropped_channels, n_attempted, n_kept):
    """Run _extract_session_traces on a (possibly trimmed) tfr_trials
    and write the npz side-car. n_after_reject is recomputed from the
    final tfr_trials so it includes the trimming drop count.
    """
    traces = _extract_session_traces(tfr_trials, dropped_channels)
    n_after_reject = sum(int(t.data.shape[0]) for t in tfr_trials.values())
    _write_per_trial_npz(
        out_path, subject, session, traces,
        {
            "n_attempted": n_attempted,
            "n_kept": n_kept,
            "n_after_reject": n_after_reject,
            "dropped_channels": dropped_channels,
        },
    )
    return n_after_reject


def _score_variant(tag):
    rows = score_dir(PER_TRIAL_DIR, variant=f"_{tag}")
    json_path = OUT_DIR / f"erd_quality_scores_{tag}.json"
    csv_path = OUT_DIR / f"erd_quality_scores_{tag}.csv"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for r in rows:
            flat = dict(r)
            flat["channels_used"] = ";".join(r["channels_used"])
            flat["gates_failed"] = ";".join(r["gates_failed"])
            w.writerow({k: ("" if flat.get(k) is None else flat[k])
                        for k in _CSV_FIELDS})
    elig = sum(1 for r in rows if r["eligible"])
    return rows, elig


def main():
    t_start = time.time()
    print(f"[setup] subjects={SUBJECTS} sessions={SESSIONS}")
    print(f"[setup] intervention_session={INTERVENTION_SESSION}")
    print(f"[setup] variants={list(TRIM_VARIANTS)}")
    print(f"[setup] out_dir={OUT_DIR}")

    for subj in SUBJECTS:
        print(f"\n=== {subj} ===")
        for sess in SESSIONS:
            t_sess = time.time()
            print(f"  {sess}:")
            out = config_a_pipeline(subj, sess)
            tfr_trials = out["tfr_trials"]
            n_attempted = int(out["n_attempted"])
            n_kept = int(out["n_kept"])
            dropped_channels = out["dropped_channels"]

            # Canonical artifact rejection (mutates in place).
            rej_report = _reject_artifact_trials(tfr_trials)

            # V0 — canonical baseline (no trimming).
            n_after_v0 = _write_variant_npz(
                PER_TRIAL_DIR / f"{subj}_{sess}_V0.npz",
                subj, sess, tfr_trials, dropped_channels,
                n_attempted, n_kept,
            )
            print(f"    V0  : n_attempted={n_attempted} n_kept={n_kept} "
                  f"n_after_reject={n_after_v0}  "
                  f"reject={dict(rej_report)}")

            if sess == INTERVENTION_SESSION:
                # Build trimming variants from the post-rejection tfr_trials.
                for tag, p in TRIM_VARIANTS.items():
                    trimmed = _trim_mi_first_p(tfr_trials, p)
                    n_after = _write_variant_npz(
                        PER_TRIAL_DIR / f"{subj}_{sess}_{tag}.npz",
                        subj, sess, trimmed, dropped_channels,
                        n_attempted, n_kept,
                    )
            else:
                # Duplicate canonical npz under each variant tag so the
                # scorer's G3 gate (evaluate_erd_quality.py:683-704), which
                # scans every npz in the variant glob, sees this subject's
                # full session set when the trimmed S001 alone wouldn't
                # show desync. The non-S001 sessions are untouched by the
                # intervention — same canonical data, four filenames.
                for tag in TRIM_VARIANTS:
                    _write_variant_npz(
                        PER_TRIAL_DIR / f"{subj}_{sess}_{tag}.npz",
                        subj, sess, tfr_trials, dropped_channels,
                        n_attempted, n_kept,
                    )

            del out, tfr_trials
            gc.collect()
            print(f"    ({time.time() - t_sess:.1f}s)")

    # Score each variant against the unchanged scorer.
    print("\n=== Scoring each variant ===")
    for tag in ("V0",) + tuple(TRIM_VARIANTS):
        rows, elig = _score_variant(tag)
        n_subj = len({r["subject"] for r in rows})
        n_sess = len({(r["subject"], r["session"]) for r in rows})
        print(f"  {tag:4s}: {len(rows)} rows  ({n_subj} subj × "
              f"{n_sess // max(n_subj, 1)} sess avg)  {elig} eligible")

    print(f"\nDone ({time.time() - t_start:.1f}s). Outputs at {OUT_DIR}")


if __name__ == "__main__":
    main()
