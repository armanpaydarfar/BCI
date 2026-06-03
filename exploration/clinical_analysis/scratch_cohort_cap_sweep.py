#!/usr/bin/env python3
"""Phase 1 of the cap-cohort + viz-style work — sweep trial-rejection
abs-cap values across the whole CLIN cohort and pick a winner.

Extends Investigation D (SUBJ_006-only) to the 7-subject longitudinal
cohort (CLIN_SUBJ_002..008). One TFR pass per (subject, session); the
same tfr_trials substrate feeds five rejection variants (cap = 600 / 200
/ 250 / 300 / 350) via shallow-copy of the dict (the rejector replaces
dict entries via slicing — Analyze_clinical_erd_refined.py:261 — never
mutates the original EpochsTFR, so dict-shallow-copy is enough).

Winner-selection rule:
  Start from the MOST AGGRESSIVE non-baseline cap (200). Accept it if the
  cohort-wide eligible-row count is NOT less than the canonical (600)
  baseline. If it regresses (releases fewer G1 than it newly trips with
  G2), step up the cap (200 -> 250 -> 300 -> 350) until net cohort
  eligibility is non-regressive or all candidates exhausted (error).

Outputs (scratch only — no canonical edits, no commits):
  C:\\Users\\arman\\Pictures\\clin_analysis_cohort_cap_sweep\\
    per_trial/
      CLIN_SUBJ_NNN_S00NONLINE_cap{600,200,250,300,350}.npz   (per-variant)
    erd_quality_scores_cap{600,200,250,300,350}.csv  / .json   (per-variant)
    sanity_cap600_vs_canonical.txt                             (per-cluster)
    cohort_cap_sweep_summary.csv  (per-subject, per-cap eligibility/G2)
    chosen_cap.json   {cap_pct, criterion, retry_count}        (Phase 2 reads this)
    erd_refined/
      CLIN_SUBJ_NNN_6panel_mi_rest_cap{600,WINNER}.png
    phase1_report.txt  (deliverable)
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

# Canonical imports (read-only; no edits to these files).
from generate_plots_config_a import preprocess_and_tfr  # noqa: E402
from Analyze_clinical_erd_refined import (  # noqa: E402
    CONFIG_A_DISPLAY_BASELINE, MU_HI, MU_LO,
    _cluster_timecourse, _logratio_to_pct,
    _reject_artifact_trials, _write_per_trial_npz,
)
from exploration.clinical_analysis._helpers import (  # noqa: E402
    BILATERAL_MOTOR_CLUSTER, CONTRA_MOTOR_CLUSTER, IPSI_MOTOR_CLUSTER,
    enumerate_clin_subjects, enumerate_online_sessions_for_subject,
)
from evaluate_erd_quality import _CSV_FIELDS, score_dir  # noqa: E402


# ----------------------------------------------------------------------
# Run config
# ----------------------------------------------------------------------

OUT_DIR = Path(r"C:\Users\arman\Pictures\clin_analysis_cohort_cap_sweep")
PER_TRIAL_DIR = OUT_DIR / "per_trial"
FIGS_DIR = OUT_DIR / "erd_refined"
for d in (OUT_DIR, PER_TRIAL_DIR, FIGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

MI_MARKER = "200"
REST_MARKER = "100"

# (variant_tag, cap_value). cap600 is the canonical sanity baseline; the
# remaining four are the candidate windows. Order matters: the winner
# search starts at the SECOND entry (most aggressive) and steps up.
CAP_VARIANTS = [
    ("cap600", 600.0),  # canonical baseline (sanity gate against /clin_analysis/)
    ("cap200", 200.0),  # most aggressive — close the asymmetry with G1
    ("cap250", 250.0),
    ("cap300", 300.0),
    ("cap350", 350.0),
]
CANDIDATE_ORDER = ["cap200", "cap250", "cap300", "cap350"]

CANONICAL_NPZ_DIR = Path(
    r"C:\Users\arman\Pictures\clin_analysis\erd_refined\per_trial"
)


def reject_for_cluster(tfr_trials, cluster_chs,
                       reject_z=5.0, max_frac=0.50, abs_cap=200.0):
    """Cluster-substrate-parameterized version of
    Analyze_clinical_erd_refined._reject_artifact_trials (file:line
    158-263). Returns a NEW dict of EpochsTFR (does not mutate input)
    plus a per-marker report. Logic mirrors the canonical helper
    line-for-line, except `present` is filtered against the supplied
    `cluster_chs` instead of the hardcoded BILATERAL_MOTOR_CLUSTER
    (canonical file:line 210). Local helper, not a canonical edit.

    The rejection decision per trial is:
      * abs_drop: per-trial cluster-mean post-cue peak |ERD%| > abs_cap
      * z_drop:   |robust-z over the cluster scalar| > reject_z
      * if more than max_frac of trials qualify, only the max_frac
        WORST offenders (largest cluster scalar) are dropped; over_gate
        flag is set so the caller can mark G2 if needed.
    """
    new_trials = {}
    report = {}
    for marker, tfr in tfr_trials.items():
        n_before = int(tfr.data.shape[0])
        info = {"n_before": n_before, "n_dropped": 0,
                "kept": True, "over_gate": False}
        report[marker] = info
        new_trials[marker] = tfr  # default to no-op
        if reject_z <= 0 or n_before < 4:
            continue
        present = [c for c in cluster_chs if c in tfr.ch_names]
        if not present:
            continue
        ch_idxs = [tfr.ch_names.index(c) for c in present]
        fmask = (tfr.freqs >= MU_LO) & (tfr.freqs <= MU_HI)
        pct = _logratio_to_pct(
            tfr.data[:, ch_idxs][:, :, fmask],
        ).mean(axis=(1, 2))
        tmask = tfr.times >= 0.0
        scalar = np.max(np.abs(pct[:, tmask]), axis=1)
        abs_drop = scalar > abs_cap
        med = np.median(scalar)
        mad = np.median(np.abs(scalar - med))
        if mad > 0:
            z = (scalar - med) / (1.4826 * mad)
            z_drop = np.abs(z) > reject_z
        else:
            z_drop = np.zeros_like(scalar, dtype=bool)
        drop = z_drop | abs_drop
        n_drop = int(drop.sum())
        if n_drop == 0:
            continue
        max_drop = int(np.floor(max_frac * n_before))
        if n_drop > max_drop:
            info["over_gate"] = True
            worst = np.argsort(scalar)[::-1][:max_drop]
            keep_mask = np.ones(n_before, dtype=bool)
            keep_mask[worst] = False
            n_drop = max_drop
        else:
            keep_mask = ~drop
        if n_drop == 0:
            continue
        new_trials[marker] = tfr[np.where(keep_mask)[0]]
        info["n_dropped"] = n_drop
    return new_trials, report


def _extract_cluster_matched_traces(tfr_base, dropped_channels, abs_cap):
    """Cluster-matched rejection + extraction. For each cluster ∈
    {bilat, contra, ipsi}, re-runs rejection with that cluster as the
    substrate and extracts THAT cluster's traces from THAT cluster's
    cleaned pool. Each cluster_marker key ends up with its own n_trials.

    Returns (traces, per_cluster_report) where per_cluster_report is
    {cluster: {marker: {n_before, n_dropped, over_gate}}}.
    """
    cluster_specs = [
        ("contra", CONTRA_MOTOR_CLUSTER),
        ("bilat",  BILATERAL_MOTOR_CLUSTER),
        ("ipsi",   IPSI_MOTOR_CLUSTER),
    ]
    traces = {"dropped_channels": list(dropped_channels)}
    per_trial: dict = {}
    per_cluster_report: dict = {}
    for cluster_label, cluster_chs in cluster_specs:
        tfr_cluster, rej = reject_for_cluster(
            tfr_base, cluster_chs, abs_cap=abs_cap,
        )
        per_cluster_report[cluster_label] = rej
        for marker_label, marker in (("mi", MI_MARKER),
                                      ("rest", REST_MARKER)):
            key = f"{cluster_label}_{marker_label}"
            res = _cluster_timecourse(tfr_cluster, cluster_chs, marker,
                                      return_per_trial=True)
            if res is None:
                traces[key] = None
                continue
            times, mean_pct, low_pct, up_pct, n, present, ptp = res
            traces[key] = (times, mean_pct, low_pct, up_pct, n, present)
            per_trial[key] = {
                "per_trial_pct": ptp,
                "times": times,
                "channels_used": present,
            }
    traces["per_trial"] = per_trial
    return traces, per_cluster_report


def _write_csv(rows, csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for r in rows:
            flat = dict(r)
            flat["channels_used"] = ";".join(r["channels_used"])
            flat["gates_failed"] = ";".join(r["gates_failed"])
            w.writerow({k: ("" if flat.get(k) is None else flat[k])
                        for k in _CSV_FIELDS})


def main():
    t_start = time.time()
    subjects = enumerate_clin_subjects()
    print(f"[setup] subjects (n={len(subjects)}): {subjects}")
    print(f"[setup] caps: {CAP_VARIANTS}")
    print(f"[setup] out_dir={OUT_DIR}")

    # ------------------------------------------------------------------
    # Stage A — TFR + per-variant rejection + per-trial npz.
    # One TFR pass per session; shallow-copy tfr_trials per cap variant.
    # ------------------------------------------------------------------
    all_reject_reports: dict[tuple[str, str, str], dict] = {}
    n_after_by: dict[tuple[str, str, str], int] = {}
    for subject in subjects:
        sessions = enumerate_online_sessions_for_subject(subject)
        print(f"\n=== {subject} ({len(sessions)} sessions) ===")
        for sess in sessions:
            # Resume-aware: skip TFR compute if every cap variant has
            # already produced its per-trial npz. Cheap, idempotent.
            expected_npzs = [
                PER_TRIAL_DIR / f"{subject}_{sess}_{tag}.npz"
                for tag, _ in CAP_VARIANTS
            ]
            if all(p.exists() for p in expected_npzs):
                print(f"  [skip] {sess}: all "
                      f"{len(CAP_VARIANTS)} cap variants already exist")
                continue
            t_sess = time.time()
            print(f"  [tfr] {sess} preprocess + TFR…")
            try:
                out = preprocess_and_tfr(subject, sess,
                                         CONFIG_A_DISPLAY_BASELINE)
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}; skip session")
                continue
            tfr_base = out["tfr_trials"]
            dropped = out.get("dropped_channels", [])
            n_att = out.get("n_attempted", 0)
            n_kept = out.get("n_kept", 0)
            print(f"    n_kept={n_kept}/{n_att} dropped={dropped or '—'} "
                  f"({time.time() - t_sess:.1f}s)")

            for variant_tag, cap_val in CAP_VARIANTS:
                # Cluster-matched rejection: contra/bilat/ipsi each get
                # their own kept-trial pool, derived from per-cluster
                # post-cue peak |ERD%|. Avoids dilution mismatch where a
                # focal-spike trial passes bilat-8 averaging but fails
                # contra-4 (notes-to-oneshot-agent-2026-06-03.md Fix 1).
                traces, per_cluster_rej = _extract_cluster_matched_traces(
                    tfr_base, dropped, cap_val,
                )
                # Session-level n_after_reject (for the npz meta the
                # scorer reads for G2): use the bilat pool, the most
                # stringent substrate. Keeps canonical G2 semantics
                # without modifying evaluate_erd_quality.
                bilat_rej = per_cluster_rej.get("bilat", {})
                n_after_bilat = sum(
                    info["n_before"] - info["n_dropped"]
                    for info in bilat_rej.values()
                ) if bilat_rej else n_kept
                all_reject_reports[(subject, sess, variant_tag)] = (
                    per_cluster_rej
                )
                n_after_by[(subject, sess, variant_tag)] = n_after_bilat
                _write_per_trial_npz(
                    PER_TRIAL_DIR
                    / f"{subject}_{sess}_{variant_tag}.npz",
                    subject, sess, traces,
                    {
                        "n_attempted": n_att,
                        "n_kept": n_kept,
                        "n_after_reject": n_after_bilat,
                        "dropped_channels": dropped,
                    },
                )
                # Console: per-cluster drop counts on the MI marker
                # (the publication-line metric) for quick scan.
                line = []
                for cl in ("contra", "bilat", "ipsi"):
                    rep = per_cluster_rej.get(cl, {}).get(MI_MARKER, {})
                    nd = rep.get("n_dropped", 0)
                    nb = rep.get("n_before", 0)
                    capped = "*" if rep.get("over_gate") else ""
                    line.append(f"{cl}_mi:-{nd}/{nb}{capped}")
                print(f"    [{variant_tag} cap={cap_val:.0f}%] "
                      f"n_after_bilat={n_after_bilat}  "
                      + " ".join(line))
                del traces
            del out, tfr_base
            gc.collect()

    # ------------------------------------------------------------------
    # Stage B — score every variant.
    # ------------------------------------------------------------------
    print("\n=== Scoring each cap variant ===")
    scorecards: dict[str, list[dict]] = {}
    for variant_tag, cap_val in CAP_VARIANTS:
        rows = score_dir(PER_TRIAL_DIR, variant=f"_{variant_tag}")
        scorecards[variant_tag] = rows
        with open(OUT_DIR / f"erd_quality_scores_{variant_tag}.json",
                  "w") as f:
            json.dump(rows, f, indent=2)
        _write_csv(rows, OUT_DIR / f"erd_quality_scores_{variant_tag}.csv")
        elig = sum(1 for r in rows if r["eligible"])
        print(f"  {variant_tag} cap={cap_val:.0f}%: "
              f"{len(rows)} rows, {elig} eligible")

    # ------------------------------------------------------------------
    # Stage C — cap=600 vs canonical sanity gate (BILAT ROWS ONLY).
    # Under cluster-matched rejection (notes-to-oneshot-agent Fix 1),
    # contra/ipsi rows use cluster-specific rejection pools — they will
    # NOT match the canonical run (which used bilat substrate for every
    # cluster). Only the bilat row's substrate is identical to canonical;
    # only it can be sanity-checked to 4 dp.
    # ------------------------------------------------------------------
    print("\n=== cap=600 vs canonical sanity check (bilat rows only) ===")
    canonical_rows = score_dir(CANONICAL_NPZ_DIR, variant="_car")
    canonical_by_key = {
        (r["subject"], r["session"], r["cluster"]): r
        for r in canonical_rows if r["cluster"] == "bilat"
    }
    cap600_by_key = {
        (r["subject"], r["session"], r["cluster"]): r
        for r in scorecards["cap600"] if r["cluster"] == "bilat"
    }
    sanity_lines = [
        "# cap600 vs canonical /clin_analysis/erd_refined/per_trial/*_car.npz",
        "# BILAT ROWS ONLY — under cluster-matched rejection, contra/ipsi",
        "# use a different rejection substrate than canonical and are not",
        "# expected to match. Bilat substrate is identical to canonical so",
        "# a 4 dp match here proves the cluster-matched helper reproduces",
        "# the canonical bilat path.",
        "",
    ]
    max_abs_diff = 0.0
    sanity_fields = ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8",
                     "S", "drop_frac", "n_after_reject")
    for key in sorted(canonical_by_key):
        ref = canonical_by_key[key]
        got = cap600_by_key.get(key)
        if got is None:
            sanity_lines.append(f"{key}: MISSING in cap600")
            continue
        diffs = []
        for f in sanity_fields:
            a = ref.get(f)
            b = got.get(f)
            if a is None or b is None:
                if a != b:
                    diffs.append(f"{f}=None/{a}/{b}")
                continue
            d = float(b) - float(a)
            if abs(d) > 1e-4:
                diffs.append(f"{f}={a}→{b} (Δ={d:+.4f})")
                max_abs_diff = max(max_abs_diff, abs(d))
        gates_ok = (set(ref.get("gates_failed", []))
                    == set(got.get("gates_failed", [])))
        if diffs or not gates_ok:
            line = f"{key}: " + ("; ".join(diffs) if diffs else "matches")
            if not gates_ok:
                line += (f"  gates mismatch: {ref.get('gates_failed')}"
                         f" vs {got.get('gates_failed')}")
            sanity_lines.append(line)
        else:
            sanity_lines.append(f"{key}: matches to 4 dp")
    sanity_lines.append("")
    sanity_lines.append(
        f"max abs diff across checked fields: {max_abs_diff:.6f}"
    )
    (OUT_DIR / "sanity_cap600_vs_canonical.txt").write_text(
        "\n".join(sanity_lines)
    )
    print(f"  max abs diff = {max_abs_diff:.6f}  → "
          f"sanity_cap600_vs_canonical.txt")

    # ------------------------------------------------------------------
    # Stage D — winner selection.
    # Net rule: pick the most aggressive cap whose total cohort eligible
    # count is >= cap600's. If it regresses, step up.
    # Tie-break: equal eligibility → prefer the more aggressive cap
    # (the more outliers it kicks the better; the score improvements
    # are real even when eligibility is unchanged).
    # ------------------------------------------------------------------
    cohort_elig = {
        tag: sum(1 for r in scorecards[tag] if r["eligible"])
        for tag, _ in CAP_VARIANTS
    }
    g2_trip_counts = {tag: 0 for tag, _ in CAP_VARIANTS}
    for tag, _ in CAP_VARIANTS:
        for r in scorecards[tag]:
            if "G2" in r.get("gates_failed", []):
                g2_trip_counts[tag] += 1
    print(f"\n=== Cohort eligibility by cap ===")
    for tag, cap_val in CAP_VARIANTS:
        print(f"  {tag} cap={cap_val:.0f}%: "
              f"{cohort_elig[tag]} eligible rows  "
              f"(G2 trips: {g2_trip_counts[tag]})")

    baseline = cohort_elig["cap600"]
    chosen: str | None = None
    chosen_retries = 0
    for tag in CANDIDATE_ORDER:
        if cohort_elig[tag] >= baseline:
            chosen = tag
            print(f"  → choose {tag} ({cohort_elig[tag]} >= "
                  f"baseline {baseline})")
            break
        else:
            chosen_retries += 1
            print(f"  ⨯ {tag} regresses cohort ({cohort_elig[tag]} < "
                  f"baseline {baseline}); retry with looser cap")
    if chosen is None:
        msg = (f"FATAL: every candidate cap ({CANDIDATE_ORDER}) regresses "
               f"cohort eligibility vs cap600 baseline ({baseline}). "
               "Cap cannot be safely tightened. Phase 2 not run.")
        print("\n" + msg)
        (OUT_DIR / "chosen_cap.json").write_text(json.dumps(
            {"cap_pct": None, "criterion": "fatal_all_regress",
             "retry_count": chosen_retries,
             "cohort_eligible_by_cap": cohort_elig,
             "g2_trips_by_cap": g2_trip_counts,
             "baseline_cap600_eligible": baseline}, indent=2))
        raise SystemExit(2)

    chosen_cap_val = dict(CAP_VARIANTS)[chosen]
    chosen_payload = {
        "cap_pct": chosen_cap_val,
        "variant_tag": chosen,
        "criterion": "most_aggressive_non_regressive",
        "retry_count": chosen_retries,
        "cohort_eligible_by_cap": cohort_elig,
        "g2_trips_by_cap": g2_trip_counts,
        "baseline_cap600_eligible": baseline,
    }
    (OUT_DIR / "chosen_cap.json").write_text(json.dumps(chosen_payload,
                                                        indent=2))

    # ------------------------------------------------------------------
    # Stage E — per-subject / per-cap summary CSV.
    # ------------------------------------------------------------------
    summary_path = OUT_DIR / "cohort_cap_sweep_summary.csv"
    summary_fields = ["subject", "variant", "cap_pct",
                      "n_sessions", "n_rows", "n_eligible_rows",
                      "n_bilat_eligible", "n_g1_trips", "n_g2_trips",
                      "n_g3_trips", "n_g4_trips",
                      "median_D1_bilat_mi", "median_S"]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for tag, cap_val in CAP_VARIANTS:
            for subj in subjects:
                rs = [r for r in scorecards[tag] if r["subject"] == subj]
                bilat_rs = [r for r in rs if r["cluster"] == "bilat"]
                d1_vals = [r["D1"] for r in bilat_rs
                           if r["D1"] is not None]
                s_vals = [r["S"] for r in rs if r["S"] is not None]
                w.writerow({
                    "subject": subj,
                    "variant": tag,
                    "cap_pct": cap_val,
                    "n_sessions": len({r["session"] for r in rs}),
                    "n_rows": len(rs),
                    "n_eligible_rows": sum(1 for r in rs if r["eligible"]),
                    "n_bilat_eligible": sum(1 for r in bilat_rs
                                            if r["eligible"]),
                    "n_g1_trips": sum(1 for r in rs
                                      if "G1" in r["gates_failed"]),
                    "n_g2_trips": sum(1 for r in rs
                                      if "G2" in r["gates_failed"]),
                    "n_g3_trips": sum(1 for r in rs
                                      if "G3" in r["gates_failed"]),
                    "n_g4_trips": sum(1 for r in rs
                                      if "G4" in r["gates_failed"]),
                    "median_D1_bilat_mi": (
                        round(float(np.median(d1_vals)), 4)
                        if d1_vals else None
                    ),
                    "median_S": (round(float(np.median(s_vals)), 4)
                                 if s_vals else None),
                })
    print(f"  wrote {summary_path.name}")

    # ------------------------------------------------------------------
    # Stage F — per-subject 6-panel figures for cap600 and the winner.
    # ------------------------------------------------------------------
    print("\n=== Plotting per-subject 6-panel figures (cap600 vs winner) ===")
    from scratch_cohort_cap_sweep_plots import (  # noqa: E402
        plot_cohort_for_variant,
    )
    plot_cohort_for_variant("cap600", subjects, PER_TRIAL_DIR, FIGS_DIR,
                            "cap=600% (canonical baseline)")
    plot_cohort_for_variant(
        chosen, subjects, PER_TRIAL_DIR, FIGS_DIR,
        f"cap={chosen_cap_val:.0f}% (Phase 1 winner)",
    )

    # ------------------------------------------------------------------
    # Stage G — Phase 1 report deliverable.
    # ------------------------------------------------------------------
    report_lines = [
        "Phase 1 — cohort cap sweep",
        "==========================",
        f"Total wall-time: {time.time() - t_start:.1f}s",
        f"Subjects: {subjects}",
        f"Caps swept: {[c[1] for c in CAP_VARIANTS]}",
        "",
        "Sanity gate (cap600 vs canonical):",
        f"  max abs diff = {max_abs_diff:.6f} (see "
        "sanity_cap600_vs_canonical.txt)",
        "",
        "Cohort eligibility (rows of "
        f"{len(scorecards['cap600'])}):",
    ]
    for tag, cap_val in CAP_VARIANTS:
        report_lines.append(
            f"  {tag} cap={cap_val:.0f}%: "
            f"{cohort_elig[tag]:3d} eligible  (G2 trips: "
            f"{g2_trip_counts[tag]:2d})"
        )
    report_lines += [
        "",
        f"Winner: {chosen} cap={chosen_cap_val:.0f}% "
        f"(retries: {chosen_retries}, criterion: "
        "most aggressive non-regressive)",
        "",
        "Per-subject summary at cohort_cap_sweep_summary.csv",
        "Per-variant scorecards at erd_quality_scores_<variant>.{csv,json}",
        "Per-trial substrates at per_trial/*.npz",
        "Per-subject 6-panel figures at erd_refined/",
        "",
        "Phase 2 will read chosen_cap.json and run three viz variants "
        "at the winner cap.",
    ]
    (OUT_DIR / "phase1_report.txt").write_text("\n".join(report_lines))
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
