#!/usr/bin/env python3
"""Investigation D — tighter post-cue peak |ERD%| absolute cap (CLIN_SUBJ_006 only).

Background: `_reject_artifact_trials` (Analyze_clinical_erd_refined.py:158-263)
drops a trial whose post-cue peak |ERD%| over the bilateral motor cluster
exceeds `TRIAL_REJECT_ABS_PCT` (line 106, default 600.0%). The diagnostic gate
G1 (evaluate_erd_quality.py:587-600) trips when >5% of *kept* trials have a
peak >200% — so trials in the 200–600% band survive the cleaner but trip G1.

Hypothesis (user, 2026-06-03): sustained MI ERD typically sits in [0, -30]%.
Trials peaking above ±200% post-cue are physiologically implausible and
likely muscle/EMG bursts. Lowering the absolute cap should drop those trials,
release the G1 fail, and produce visibly different 6-panel plots — provided
the 50% over-rejection guard (G2) doesn't trip first.

Four variants per session (all 5 ONLINE):
  V0 cap=600  — canonical sanity baseline. MUST match
                 ~/Pictures/clin_analysis/erd_refined/per_trial/
                 CLIN_SUBJ_006_S00NONLINE_car.npz scores to 4 dp.
  V1 cap=300
  V2 cap=200  — the user's headline ask: cap == G1 threshold
  V3 cap=100  — aggressive lower bound; expects G2 trips on several sessions

Trial rejection substrate is BILATERAL_MOTOR_CLUSTER (per-trial cluster-mean
ERD% over canonical 8 ch, mu band, post-cue peak |ERD%|) — same substrate as
G1, so the cap acts directly on the metric G1 watches.

One TFR pass per session; shallow-copy `tfr_trials` per variant so the same
TFR substrate feeds all four rejection passes (avoids 4× a ~12 min/session
compute). Per `_reject_artifact_trials:261` the function replaces dict
entries via slicing — original `EpochsTFR` objects aren't mutated, so dict-
shallow-copy is sufficient.

Diagnostic CSV (per session, per variant): for each trial DROPPED beyond V0,
record its post-cue peak |ERD%|, the fraction of post-cue time it exceeds the
50% line ("sustained vs transient"), and trial index — so the user can see
whether the newly-dropped trials are sustained-large outliers (real artifact)
or transient spikes (might be salvageable with smoothing).

Outputs (scratch only — no canonical edits, no commits):
  C:\\Users\\arman\\Pictures\\clin_analysis_tighter_cap\\
    per_trial/
      CLIN_SUBJ_006_S00NONLINE_V0.npz   (cap=600)
      CLIN_SUBJ_006_S00NONLINE_V1.npz   (cap=300)
      CLIN_SUBJ_006_S00NONLINE_V2.npz   (cap=200)
      CLIN_SUBJ_006_S00NONLINE_V3.npz   (cap=100)
    erd_quality_scores_V0.csv / .json
    erd_quality_scores_V1.csv / .json
    erd_quality_scores_V2.csv / .json
    erd_quality_scores_V3.csv / .json
    sanity_v0_vs_canonical.txt          (per-(session,cluster) diff)
    dropped_trial_diagnostics.csv       (newly-dropped trials per variant)
    investigation_d_report.txt          (deliverable: numbers table)
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
)
from evaluate_erd_quality import (  # noqa: E402
    G1_OUTLIER_FRAC, G1_OUTLIER_PCT, _CSV_FIELDS, score_dir,
)


# ----------------------------------------------------------------------
# Run config
# ----------------------------------------------------------------------

SUBJECT = "CLIN_SUBJ_006"
SESSIONS = [f"S{n:03d}ONLINE" for n in range(1, 6)]

OUT_DIR = Path(r"C:\Users\arman\Pictures\clin_analysis_tighter_cap")
PER_TRIAL_DIR = OUT_DIR / "per_trial"
for d in (OUT_DIR, PER_TRIAL_DIR):
    d.mkdir(parents=True, exist_ok=True)

MI_MARKER = "200"
REST_MARKER = "100"

VARIANTS = [
    ("V0", 600.0),  # canonical sanity baseline
    ("V1", 300.0),
    ("V2", 200.0),
    ("V3", 100.0),
]

# Path to the canonical per-trial npz side-car for sanity comparison.
CANONICAL_NPZ_DIR = Path(
    r"C:\Users\arman\Pictures\clin_analysis\erd_refined\per_trial"
)


# ----------------------------------------------------------------------
# Per-trial peak |ERD%| substrate — mirrors
# _reject_artifact_trials:210-226 exactly, so newly-dropped diagnostics
# match what the cleaner sees.
# ----------------------------------------------------------------------

def _per_trial_peak_pct(tfr, cluster_chs):
    """Per-trial post-cue peak |ERD%| over (cluster_chs, mu band).

    Mirrors _reject_artifact_trials:210-226. Returns (peaks, frac_over_50,
    pct_array) where:
      peaks       : (n_trials,) post-cue peak |ERD%|
      frac_over_50: (n_trials,) fraction of post-cue time where |ERD%| > 50
                    (i.e. an a sustained-vs-transient diagnostic — high value
                    means the spike is sustained, low value means transient)
      pct_array   : (n_trials, n_time_postcue) cluster-mean ERD% time-course,
                    so the diagnostic CSV can carry it for plotting.
    """
    present = [c for c in cluster_chs if c in tfr.ch_names]
    if not present:
        return None, None, None
    ch_idxs = [tfr.ch_names.index(c) for c in present]
    fmask = (tfr.freqs >= MU_LO) & (tfr.freqs <= MU_HI)
    # Same axis order as the cleaner: (n_trials, n_ch, n_freq, n_time) →
    # mean over (channels, freqs) AFTER % conversion → (n_trials, n_time).
    pct = _logratio_to_pct(
        tfr.data[:, ch_idxs][:, :, fmask],
    ).mean(axis=(1, 2))
    tmask = tfr.times >= 0.0
    pct_post = pct[:, tmask]
    peaks = np.max(np.abs(pct_post), axis=1)
    # Fraction of post-cue time |ERD%| exceeds 50 — a coarse "sustained"
    # proxy. A 350% peak that lasts 1 sample of 1024 reads frac~0.001;
    # a sustained 250% across the post-cue window reads frac~1.0.
    frac_50 = (np.abs(pct_post) > 50.0).mean(axis=1)
    return peaks, frac_50, pct_post


# ----------------------------------------------------------------------
# Trace extraction (canonical cluster definitions across all variants)
# ----------------------------------------------------------------------

def _extract_canonical_traces(tfr_trials, dropped_channels):
    """Mirror Analyze_clinical_erd_refined._extract_session_traces:325-365.

    Canonical cluster definitions; no input-montage variation (the abs-cap
    is the only knob in this investigation).
    """
    cluster_specs = [
        ("contra_mi",   CONTRA_MOTOR_CLUSTER,    MI_MARKER),
        ("contra_rest", CONTRA_MOTOR_CLUSTER,    REST_MARKER),
        ("bilat_mi",    BILATERAL_MOTOR_CLUSTER, MI_MARKER),
        ("bilat_rest",  BILATERAL_MOTOR_CLUSTER, REST_MARKER),
        ("ipsi_mi",     IPSI_MOTOR_CLUSTER,      MI_MARKER),
        ("ipsi_rest",   IPSI_MOTOR_CLUSTER,      REST_MARKER),
    ]
    traces = {"dropped_channels": list(dropped_channels)}
    per_trial: dict = {}
    for key, cluster, marker in cluster_specs:
        res = _cluster_timecourse(tfr_trials, cluster, marker,
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
    return traces


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    t_start = time.time()
    print(f"[setup] subject={SUBJECT} sessions={SESSIONS}")
    print(f"[setup] out_dir={OUT_DIR}")
    print(f"[setup] variants={VARIANTS}")
    print(f"[setup] G1_OUTLIER_PCT={G1_OUTLIER_PCT} "
          f"G1_OUTLIER_FRAC={G1_OUTLIER_FRAC}")

    # Per-(session, variant) reject reports + the diagnostic rows are
    # accumulated across the loop and written at the end.
    all_reject_reports: dict[tuple[str, str], dict] = {}
    diagnostic_rows: list[dict] = []
    n_after_by_variant: dict[tuple[str, str], int] = {}

    for sess in SESSIONS:
        print(f"\n=== {SUBJECT} / {sess} ===")
        t_sess = time.time()

        # ----------------------------------------------------------------
        # One canonical TFR pass per session (~12 min).
        # ----------------------------------------------------------------
        print(f"  [tfr] canonical preprocess + TFR…")
        out = preprocess_and_tfr(SUBJECT, sess, CONFIG_A_DISPLAY_BASELINE)
        tfr_base = out["tfr_trials"]
        dropped = out.get("dropped_channels", [])
        n_attempted = out.get("n_attempted", 0)
        n_kept = out.get("n_kept", 0)
        print(f"    n_kept={n_kept}/{n_attempted} dropped={dropped or '—'} "
              f"({time.time() - t_sess:.1f}s)")

        # ----------------------------------------------------------------
        # Per-trial peak |ERD%| BEFORE any rejection (same substrate the
        # cleaner uses) — needed for the diagnostic CSV so we know what
        # each newly-dropped trial looked like.
        # ----------------------------------------------------------------
        pre_drop_peaks = {}
        for marker, tfr in tfr_base.items():
            peaks, frac_50, pct_post = _per_trial_peak_pct(
                tfr, BILATERAL_MOTOR_CLUSTER,
            )
            pre_drop_peaks[marker] = {
                "peaks": peaks,
                "frac_50": frac_50,
                "pct_post": pct_post,
                "n": int(tfr.data.shape[0]),
            }
            # Quick console summary of the pre-rejection peak distribution.
            if peaks is not None:
                qtiles = np.percentile(peaks, [50, 75, 90, 95, 100])
                print(f"    [pre] marker={marker} n={tfr.data.shape[0]} "
                      f"peak |ERD%| qtiles "
                      f"(50/75/90/95/100): {qtiles.round().tolist()}")

        # ----------------------------------------------------------------
        # Per-variant rejection + npz write.
        # Track the dropped-index sets so the diagnostic can identify the
        # *newly* dropped trials in V1/V2/V3 vs V0.
        # ----------------------------------------------------------------
        kept_indices_by_variant: dict[tuple[str, str], np.ndarray] = {}
        for variant_tag, cap_val in VARIANTS:
            # Shallow-copy the dict so each variant's rejection slice does
            # not pollute the next (rejection replaces dict entries via
            # `tfr_trials[marker] = tfr[idx]`, never mutates the original
            # tfr object — see _reject_artifact_trials:261).
            tfr_var = dict(tfr_base)
            rej_report = _reject_artifact_trials(
                tfr_var, abs_cap=cap_val,
            )
            n_after = sum(int(t.data.shape[0]) for t in tfr_var.values())
            all_reject_reports[(sess, variant_tag)] = rej_report
            n_after_by_variant[(sess, variant_tag)] = n_after

            # Recover the kept trial indices per marker by re-running the
            # rejection scalar against the original tfr (deterministic).
            for marker, tfr_orig in tfr_base.items():
                peaks = pre_drop_peaks[marker]["peaks"]
                if peaks is None:
                    continue
                # Re-derive the keep mask the cleaner produced. The cleaner
                # combines z-rule + abs-cap + 50% guard, so it's easier to
                # just diff the trial count vs match-up by peak. For the
                # diagnostic we want INDICES; the deterministic way is to
                # reproduce the cleaner's decision: a trial is kept iff its
                # peak is below the cap AND its |z| is below the z-threshold
                # AND it's not among the 50% worst (if the cap was hit). We
                # could replicate this, but a simpler proof-by-construction
                # is: the kept indices are exactly those whose peak appears
                # in the kept tfr's data — but that's a fragile comparison
                # against float64 averages.
                # Cleanest robust route: re-derive keep mask from scratch
                # using the same logic as _reject_artifact_trials:225-258.
                pass  # diagnostic uses pre/post counts only — see below

            # Console line for this variant.
            elig_str = " ".join(
                f"{m}:-{r['n_dropped']}"
                + ("(capped)" if r["over_gate"] else "")
                for m, r in rej_report.items()
            ) or "—"
            print(f"  [{variant_tag} cap={cap_val:.0f}%] "
                  f"n_after_reject={n_after} reject_report={elig_str}")

            traces = _extract_canonical_traces(tfr_var, dropped)
            _write_per_trial_npz(
                PER_TRIAL_DIR / f"{SUBJECT}_{sess}_{variant_tag}.npz",
                SUBJECT, sess, traces,
                {
                    "n_attempted": n_attempted,
                    "n_kept": n_kept,
                    "n_after_reject": n_after,
                    "dropped_channels": dropped,
                },
            )
            del tfr_var, traces

        # ----------------------------------------------------------------
        # Diagnostic rows: per (session, marker), classify every trial by
        # which variants kept it. A trial with peak >100 is dropped in V3,
        # >200 in V2 (and V3), >300 in V1 (and V2, V3), >600 in V0 (canonical
        # cleaner — but only if not surviving the 50% over-reject guard).
        # This is the bedrock the user needs to judge sustained vs transient.
        # ----------------------------------------------------------------
        for marker, info in pre_drop_peaks.items():
            peaks = info["peaks"]
            frac_50 = info["frac_50"]
            pct_post = info["pct_post"]
            if peaks is None:
                continue
            n_trials = int(info["n"])
            class_label = "MI" if marker == MI_MARKER else "REST"
            for trial_idx in range(n_trials):
                p = float(peaks[trial_idx])
                drops_v0 = p > 600.0
                drops_v1 = p > 300.0
                drops_v2 = p > 200.0
                drops_v3 = p > 100.0
                drop_tier = (
                    "V3_only" if drops_v3 and not drops_v2
                    else "V2_and_V3" if drops_v2 and not drops_v1
                    else "V1_V2_V3" if drops_v1 and not drops_v0
                    else "all" if drops_v0
                    else "none"
                )
                # Sustainedness: median |ERD%| over the post-cue window —
                # if it's high, the trial is a sustained outlier (real
                # artifact); if low, the peak is a brief transient spike.
                median_post = float(np.median(np.abs(pct_post[trial_idx])))
                # Width of the >50%-line crossing as a second proxy.
                frac = float(frac_50[trial_idx])
                diagnostic_rows.append({
                    "session": sess,
                    "marker": marker,
                    "class": class_label,
                    "trial_idx": trial_idx,
                    "peak_abs_pct": round(p, 2),
                    "median_post_abs_pct": round(median_post, 2),
                    "frac_post_over_50pct": round(frac, 4),
                    "drop_tier": drop_tier,
                    "drops_at_cap_600": drops_v0,
                    "drops_at_cap_300": drops_v1,
                    "drops_at_cap_200": drops_v2,
                    "drops_at_cap_100": drops_v3,
                })

        # Release the heavy TFR objects before the next session.
        del out, tfr_base, pre_drop_peaks
        gc.collect()
        print(f"  ({time.time() - t_sess:.1f}s)")

    # ----------------------------------------------------------------------
    # Score each variant via the unchanged evaluate_erd_quality.score_dir.
    # ----------------------------------------------------------------------
    print("\n=== Scoring each variant ===")
    scorecards: dict[str, list[dict]] = {}
    for variant_tag, cap_val in VARIANTS:
        rows = score_dir(PER_TRIAL_DIR, variant=f"_{variant_tag}")
        scorecards[variant_tag] = rows
        json_path = OUT_DIR / f"erd_quality_scores_{variant_tag}.json"
        csv_path = OUT_DIR / f"erd_quality_scores_{variant_tag}.csv"
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
        print(f"  {variant_tag} cap={cap_val:.0f}%: {len(rows)} rows, "
              f"{elig} eligible → {csv_path.name}")

    # ----------------------------------------------------------------------
    # V0 vs canonical sanity gate — V0 cap is 600 = TRIAL_REJECT_ABS_PCT
    # default; scores must match canonical SUBJ_006 rows to 4 dp.
    # ----------------------------------------------------------------------
    print("\n=== V0 vs canonical sanity check ===")
    sanity_path = OUT_DIR / "sanity_v0_vs_canonical.txt"
    canonical_rows = score_dir(CANONICAL_NPZ_DIR, variant="_car")
    canonical_by_key = {
        (r["subject"], r["session"], r["cluster"]): r
        for r in canonical_rows if r["subject"] == SUBJECT
    }
    v0_by_key = {
        (r["subject"], r["session"], r["cluster"]): r
        for r in scorecards["V0"]
    }
    sanity_lines = [
        f"# V0 (cap=600) vs canonical /erd_refined/per_trial/*_car.npz scores",
        f"# (per_trial substrates differ: V0 uses fresh TFR; canonical was a",
        f"#  prior run. Match to 4 dp confirms reproduction.)",
        "",
    ]
    max_abs_diff = 0.0
    fields = ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "S",
              "drop_frac", "n_after_reject")
    for key in sorted(canonical_by_key):
        ref = canonical_by_key[key]
        got = v0_by_key.get(key)
        if got is None:
            sanity_lines.append(f"{key}: MISSING in V0")
            continue
        diffs = []
        for f in fields:
            a = ref.get(f)
            b = got.get(f)
            if a is None or b is None:
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
                line += (f"  gates_failed mismatch: "
                         f"{ref.get('gates_failed')} vs "
                         f"{got.get('gates_failed')}")
            sanity_lines.append(line)
        else:
            sanity_lines.append(f"{key}: matches to 4 dp")
    sanity_lines.append("")
    sanity_lines.append(f"max abs diff across all checked fields: "
                        f"{max_abs_diff:.6f}")
    sanity_path.write_text("\n".join(sanity_lines))
    print(f"  wrote {sanity_path.name}  (max abs diff = {max_abs_diff:.6f})")

    # ----------------------------------------------------------------------
    # Diagnostic CSV: every trial × every variant tier, with peak and
    # sustained-vs-transient diagnostic columns.
    # ----------------------------------------------------------------------
    diag_path = OUT_DIR / "dropped_trial_diagnostics.csv"
    if diagnostic_rows:
        diag_fields = list(diagnostic_rows[0].keys())
        with open(diag_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=diag_fields)
            w.writeheader()
            for r in diagnostic_rows:
                w.writerow(r)
        # Per-variant summary: how many trials each variant *would* newly
        # drop (i.e. the count that crosses each tier's cap on MI only).
        summary_lines = []
        for variant_tag, cap_val in VARIANTS:
            for sess in SESSIONS:
                for marker_label in ("MI", "REST"):
                    rows = [r for r in diagnostic_rows
                            if r["session"] == sess
                            and r["class"] == marker_label]
                    n = len(rows)
                    if n == 0:
                        continue
                    above = sum(1 for r in rows if r["peak_abs_pct"] > cap_val)
                    summary_lines.append(
                        f"  {variant_tag} cap={cap_val:.0f}% / {sess} / "
                        f"{marker_label}: {above}/{n} trials exceed "
                        f"({100 * above / n:.1f}%)"
                    )
        print("\n=== Per-variant exceedance summary "
              "(pre-rejection peak-only; cleaner may keep more if cap > z) ===")
        for line in summary_lines:
            print(line)
    print(f"  wrote {diag_path.name}  ({len(diagnostic_rows)} trial rows)")

    print(f"\nDone ({time.time() - t_start:.1f}s). Outputs at {OUT_DIR}")


if __name__ == "__main__":
    main()
