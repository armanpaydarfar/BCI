#!/usr/bin/env python3
"""Phase 2 of the cap-cohort + viz-style work — at the Phase 1 winner
cap, generate per-subject 6-panel figures under three viz styles so we
can visually compare the canonical (median + %) timecourse against the
two other shapes the literature uses and the topomap pipeline uses.

Reads chosen_cap.json from Phase 1 (Pictures/clin_analysis_cohort_cap_
sweep/) to get the cap. Recomputes TFRs (one pass per session) and
applies the chosen cap's rejection, then extracts THREE viz variants
from the same surviving trial set:

  W1 — median + %        (canonical timecourse pipeline,
                          Analyze_clinical_erd_refined.py:299-300)
  W2 — mean   + %        (Pfurtscheller's classical timecourse
                          definition)
  W3 — mean   + logratio (matches the topomap pipeline:
                          generate_plots_config_a.py:154,157 keeps data
                          in logratio; MNE `tfr.average()` is mean; the
                          colorbar at line 215 is labelled
                          "ERD/ERS (logratio)").

W3's substrate stays in logratio per (trial, channel, freq) and only
averages — no % conversion at any step. W1/W2 share the per-(trial,
channel, freq) -> % conversion at the finest granularity (Jensen-free
per the fix in bf980c9), differing only in the across-trial aggregator
(median vs mean).

The rubric in evaluate_erd_quality is calibrated to medians-in-% units
(W1). For W2 the same thresholds apply numerically but the calibration
assumed median; D1/D2 still parse, but flagging is informational. W3 is
in logratio, so the panel score tag would mismatch — we render a
unit-correct summary (mean log, post-cue median) instead.

Outputs (scratch only — no canonical edits, no commits):
  C:\\Users\\arman\\Pictures\\clin_analysis_cohort_viz_sweep\\
    per_trial/
      CLIN_SUBJ_NNN_S00NONLINE_W{1,2,3}.npz
    erd_refined/
      CLIN_SUBJ_NNN_6panel_mi_rest_W{1,2,3}_*.png        (per variant)
    phase2_report.txt  (deliverable)
"""

from __future__ import annotations

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

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from generate_plots_config_a import preprocess_and_tfr  # noqa: E402
from Analyze_clinical_erd_refined import (  # noqa: E402
    CONFIG_A_DISPLAY_BASELINE, MU_HI, MU_LO,
    _BAND_LABEL, _logratio_to_pct, _preproc_caption,
    _reject_artifact_trials,
)
from exploration.clinical_analysis._helpers import (  # noqa: E402
    BILATERAL_MOTOR_CLUSTER, CONTRA_MOTOR_CLUSTER, IPSI_MOTOR_CLUSTER,
    enumerate_clin_subjects, enumerate_online_sessions_for_subject,
)


# ----------------------------------------------------------------------
# Run config — paths and constants
# ----------------------------------------------------------------------

PHASE1_DIR = Path(r"C:\Users\arman\Pictures\clin_analysis_cohort_cap_sweep")
OUT_DIR = Path(r"C:\Users\arman\Pictures\clin_analysis_cohort_viz_sweep")
PER_TRIAL_DIR = OUT_DIR / "per_trial"
FIGS_DIR = OUT_DIR / "erd_refined"
for d in (OUT_DIR, PER_TRIAL_DIR, FIGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

MI_MARKER = "200"
REST_MARKER = "100"

# Viz variants. (tag, aggregator, unit, human-readable note)
VIZ_VARIANTS = [
    ("W1", "median", "pct",
     "median across trials + % units (canonical timecourse)."),
    ("W2", "mean", "pct",
     "mean across trials + % units (Pfurtscheller's classical "
     "timecourse definition; differs from canonical only in the "
     "across-trial aggregator)."),
    ("W3", "mean", "log",
     "mean across trials + logratio units (matches topomap pipeline; "
     "no % conversion at any step; rubric annotations omitted since "
     "evaluate_erd_quality is calibrated to %)."),
]


# ----------------------------------------------------------------------
# Viz-aware substrate extraction
# ----------------------------------------------------------------------

def _per_trial_cluster_trace(tfr, cluster_chs, unit):
    """Return (times, present_chs, per_trial_array) for one (tfr, cluster).

    Per-(trial, channel, freq) collapses BEFORE any averaging if unit=='pct'
    (Jensen-free per bf980c9); stays in logratio if unit=='log'. The result
    is averaged over (channels, freqs) per trial, yielding a
    (n_trials, n_time) substrate.

    Returns (None, [], None) when the cluster has no channels present.
    """
    present = [c for c in cluster_chs if c in tfr.ch_names]
    if not present:
        return None, [], None
    ch_idxs = [tfr.ch_names.index(c) for c in present]
    fmask = (tfr.freqs >= MU_LO) & (tfr.freqs <= MU_HI)
    raw = tfr.data[:, ch_idxs][:, :, fmask]
    if unit == "pct":
        data = _logratio_to_pct(raw)
    elif unit == "log":
        data = raw
    else:
        raise ValueError(f"unknown unit {unit!r}")
    per_trial = data.mean(axis=(1, 2))  # (n_trials, n_time)
    return np.asarray(tfr.times, dtype=np.float64), present, per_trial


def _aggregate_trials(per_trial, aggregator):
    """Aggregate (n_trials, n_time) across the trial axis.

    Returns (median_or_mean, low, high) where low/high is ±SE (std/√n)
    centred on the central tendency, matching what
    Analyze_clinical_erd_refined._cluster_timecourse:301-304 reports for
    the SE band.
    """
    if per_trial is None or per_trial.shape[0] == 0:
        return None, None, None
    n = per_trial.shape[0]
    if aggregator == "median":
        central = np.median(per_trial, axis=0)
    elif aggregator == "mean":
        central = np.mean(per_trial, axis=0)
    else:
        raise ValueError(f"unknown aggregator {aggregator!r}")
    if n > 1:
        sem = np.std(per_trial, axis=0, ddof=1) / np.sqrt(n)
        low = central - sem
        high = central + sem
    else:
        low = central.copy()
        high = central.copy()
    return central, low, high


def _extract_session_traces(tfr_trials, aggregator, unit):
    """6-key traces dict for one session under one viz variant."""
    cluster_specs = [
        ("contra_mi",   CONTRA_MOTOR_CLUSTER,    MI_MARKER),
        ("contra_rest", CONTRA_MOTOR_CLUSTER,    REST_MARKER),
        ("bilat_mi",    BILATERAL_MOTOR_CLUSTER, MI_MARKER),
        ("bilat_rest",  BILATERAL_MOTOR_CLUSTER, REST_MARKER),
        ("ipsi_mi",     IPSI_MOTOR_CLUSTER,      MI_MARKER),
        ("ipsi_rest",   IPSI_MOTOR_CLUSTER,      REST_MARKER),
    ]
    out = {}
    for key, cluster, marker in cluster_specs:
        if marker not in tfr_trials:
            out[key] = None
            continue
        tfr = tfr_trials[marker]
        times, present, per_trial = _per_trial_cluster_trace(
            tfr, cluster, unit,
        )
        if per_trial is None:
            out[key] = None
            continue
        central, low, high = _aggregate_trials(per_trial, aggregator)
        n = int(per_trial.shape[0])
        out[key] = {
            "times": times,
            "central": central,
            "low": low,
            "high": high,
            "n_trials": n,
            "present": present,
            "per_trial": per_trial,
        }
    return out


# ----------------------------------------------------------------------
# Per-variant npz writer (lightweight; viz-aware)
# ----------------------------------------------------------------------

def _write_viz_npz(out_path, subject, session, traces, meta):
    """Write a per-trial npz with viz-variant-specific layout.

    Schema:
      subject, session (str), n_attempted, n_kept, n_after_reject (int),
      dropped_channels, keys (csv strings),
      <key>__per_trial : (n_trials, n_time)  per-trial substrate
      <key>__times     : (n_time,)
      <key>__channels  : str (csv of present channels)
      aggregator (str), unit (str)
    """
    payload = {
        "subject": np.asarray(subject),
        "session": np.asarray(session),
        "n_attempted": np.asarray(int(meta["n_attempted"])),
        "n_kept": np.asarray(int(meta["n_kept"])),
        "n_after_reject": np.asarray(int(meta["n_after_reject"])),
        "dropped_channels": np.asarray(
            ",".join(meta.get("dropped_channels", []) or [])
        ),
        "aggregator": np.asarray(meta["aggregator"]),
        "unit": np.asarray(meta["unit"]),
    }
    present_keys = []
    for key, v in traces.items():
        if v is None:
            continue
        present_keys.append(key)
        payload[f"{key}__per_trial"] = np.asarray(v["per_trial"],
                                                   dtype=np.float32)
        payload[f"{key}__times"] = np.asarray(v["times"],
                                               dtype=np.float64)
        payload[f"{key}__channels"] = np.asarray(",".join(v["present"]))
    payload["keys"] = np.asarray(",".join(present_keys))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)


# ----------------------------------------------------------------------
# 6-panel plotter (mirrors _plot_subject_6panel:445-530, but takes a
# variant-tagged label and a unit-correct y-axis caption)
# ----------------------------------------------------------------------

def _short_score_tag(central, times, unit):
    """Compact unit-correct legend annotation.

    For % the rubric-style D1=… would mismatch on W2/W3, so we keep it
    simple: post-cue median value, in the actual unit, plus a sustain
    proxy (frac post-cue with central below the 0-line).
    """
    pmask = times >= 1.0  # rubric SCALAR window
    if pmask.sum() == 0:
        return ""
    seg = central[pmask]
    if unit == "pct":
        return (f" | post-med={float(np.median(seg)):+.1f}%  "
                f"neg-frac={float((seg < 0).mean()):.2f}")
    return (f" | post-med={float(np.median(seg)):+.3f}  "
            f"neg-frac={float((seg < 0).mean()):.2f}")


def _plot_subject_6panel_viz(subject, session_traces, out_path,
                             variant_label, variant_note, unit):
    """Per-subject 6-panel for one viz variant. session_traces is a list
    of (sess_label, traces_dict_from_extract). Mirrors
    Analyze_clinical_erd_refined._plot_subject_6panel:445-530 — same
    layout, same hairlines, different aggregator + unit."""
    if not session_traces:
        return
    fig, axes = plt.subplots(
        3, 2, figsize=(14, 11), sharex=True, sharey="row",
    )
    cmap = plt.get_cmap("viridis")
    n_sess = max(1, len(session_traces))
    panel_specs = [
        (0, "mi",   "Contralateral — MI",
         "contra_mi",   CONTRA_MOTOR_CLUSTER),
        (0, "rest", "Contralateral — REST",
         "contra_rest", CONTRA_MOTOR_CLUSTER),
        (1, "mi",   "Bilateral — MI",
         "bilat_mi",    BILATERAL_MOTOR_CLUSTER),
        (1, "rest", "Bilateral — REST",
         "bilat_rest",  BILATERAL_MOTOR_CLUSTER),
        (2, "mi",   "Ipsilateral — MI",
         "ipsi_mi",     IPSI_MOTOR_CLUSTER),
        (2, "rest", "Ipsilateral — REST",
         "ipsi_rest",   IPSI_MOTOR_CLUSTER),
    ]
    col_of_class = {"mi": 0, "rest": 1}

    drew = False
    for i, (sess, traces) in enumerate(session_traces):
        color = cmap(i / max(1, n_sess - 1))
        for row, cls, title, key, cluster in panel_specs:
            ax = axes[row][col_of_class[cls]]
            v = traces.get(key)
            if v is None:
                ax.set_title(title)
                continue
            times = v["times"]
            central = v["central"]
            low = v["low"]
            high = v["high"]
            n_trials = v["n_trials"]
            present = v["present"]
            missing = [c for c in cluster if c not in present]
            tag = ", ".join(present)
            if missing:
                tag += f"  [missing: {','.join(missing)}]"
            score_tag = _short_score_tag(central, times, unit)
            label = f"{sess} (n={n_trials}; {tag}){score_tag}"
            ax.plot(times, central, color=color, label=label,
                    linewidth=1.4)
            ax.fill_between(times, low, high, color=color, alpha=0.15)
            ax.set_title(title)
            ax.axhline(0, color="k", lw=0.6)
            ax.axvline(0, color="k", ls="--", lw=0.7)
            ax.axvline(1.0, color="k", ls=":", lw=0.7)
            ax.grid(True, alpha=0.25)
            drew = True
        ylabel = "ERD %" if unit == "pct" else "ERD (logratio)"
        axes[row][0].set_ylabel(ylabel)

    if not drew:
        plt.close(fig)
        return

    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    for row in range(3):
        axes[row][0].legend(loc="best", fontsize=7)
        axes[row][1].legend(loc="best", fontsize=7)
    fig.suptitle(
        f"MU ERD across sessions — {subject} | MI vs REST  "
        f"[Phase 2 {variant_label}]\n"
        f"Contra: {CONTRA_MOTOR_CLUSTER} | "
        f"Bilateral ({len(BILATERAL_MOTOR_CLUSTER)} ch): "
        f"{BILATERAL_MOTOR_CLUSTER} | "
        f"Ipsi: {IPSI_MOTOR_CLUSTER}\n"
        f"{_preproc_caption()} | {_BAND_LABEL}\n"
        f"{variant_note}\n"
        "legend: post-med=median over post-cue [1, t_end] window; "
        "neg-frac=fraction of post-cue samples below 0.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    t_start = time.time()
    chosen_path = PHASE1_DIR / "chosen_cap.json"
    if not chosen_path.exists():
        raise SystemExit(f"missing {chosen_path}; run Phase 1 first")
    chosen = json.loads(chosen_path.read_text())
    cap_val = chosen["cap_pct"]
    if cap_val is None:
        raise SystemExit("Phase 1 reported fatal; no winner cap. "
                         "Phase 2 cannot run.")
    chosen_tag = chosen["variant_tag"]
    print(f"[setup] winner cap from Phase 1: "
          f"{chosen_tag} = {cap_val:.0f}%")
    print(f"[setup] retry_count={chosen['retry_count']}  "
          f"baseline={chosen['baseline_cap600_eligible']}")
    print(f"[setup] cohort_elig_by_cap={chosen['cohort_eligible_by_cap']}")

    subjects = enumerate_clin_subjects()
    print(f"[setup] subjects (n={len(subjects)}): {subjects}")
    print(f"[setup] viz variants: {[v[0] for v in VIZ_VARIANTS]}")
    print(f"[setup] out_dir={OUT_DIR}")

    per_subject_traces: dict[str, dict[str, list]] = {
        s: {v[0]: [] for v in VIZ_VARIANTS} for s in subjects
    }

    for subject in subjects:
        sessions = enumerate_online_sessions_for_subject(subject)
        print(f"\n=== {subject} ({len(sessions)} sessions) ===")
        for sess in sessions:
            t_sess = time.time()
            print(f"  [tfr] {sess} preprocess + TFR…")
            try:
                out = preprocess_and_tfr(subject, sess,
                                         CONFIG_A_DISPLAY_BASELINE)
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}; skip session")
                continue
            tfr_trials = out["tfr_trials"]
            dropped = out.get("dropped_channels", [])
            n_att = out.get("n_attempted", 0)
            n_kept = out.get("n_kept", 0)
            # Apply the winner cap once; same rejected pool feeds all 3
            # viz variants.
            rej = _reject_artifact_trials(tfr_trials, abs_cap=cap_val)
            n_after = sum(int(t.data.shape[0]) for t in tfr_trials.values())
            rep = " ".join(
                f"{m}:-{r['n_dropped']}"
                + ("(capped)" if r["over_gate"] else "")
                for m, r in rej.items()
            ) or "—"
            print(f"    n_kept={n_kept}/{n_att} reject={rep} "
                  f"n_after={n_after}  ({time.time() - t_sess:.1f}s)")

            for tag, aggregator, unit, _note in VIZ_VARIANTS:
                traces = _extract_session_traces(tfr_trials, aggregator,
                                                  unit)
                meta = {
                    "n_attempted": n_att,
                    "n_kept": n_kept,
                    "n_after_reject": n_after,
                    "dropped_channels": dropped,
                    "aggregator": aggregator,
                    "unit": unit,
                }
                _write_viz_npz(
                    PER_TRIAL_DIR / f"{subject}_{sess}_{tag}.npz",
                    subject, sess, traces, meta,
                )
                per_subject_traces[subject][tag].append((sess, traces))
            del out, tfr_trials
            gc.collect()

    # Render per-subject × per-variant 6-panel figures.
    print("\n=== Plotting per-subject 6-panel figures (W1/W2/W3) ===")
    for subject in subjects:
        for tag, aggregator, unit, note in VIZ_VARIANTS:
            session_traces = per_subject_traces[subject][tag]
            if not session_traces:
                continue
            out_path = FIGS_DIR / (
                f"{subject}_6panel_mi_rest_{tag}_"
                f"{aggregator}_{unit}_cap{int(cap_val)}.png"
            )
            _plot_subject_6panel_viz(
                subject, session_traces, out_path,
                variant_label=(
                    f"{tag} ({aggregator}+{unit}, cap={cap_val:.0f}%)"
                ),
                variant_note=note,
                unit=unit,
            )
            print(f"    [{tag}] {subject}: "
                  f"{out_path.name}  ({len(session_traces)} sessions)")

    # Report deliverable.
    report_lines = [
        "Phase 2 — viz-style sweep at Phase 1 winner cap",
        "===============================================",
        f"Total wall-time: {time.time() - t_start:.1f}s",
        f"Winner cap (from Phase 1): {chosen_tag} = {cap_val:.0f}%",
        f"Subjects: {subjects}",
        "",
        "Viz variants:",
    ]
    for tag, aggregator, unit, note in VIZ_VARIANTS:
        report_lines.append(f"  {tag}: {aggregator} + {unit} — {note}")
    report_lines += [
        "",
        "Per-trial substrates at per_trial/*.npz "
        "(carries per_trial array in the variant's native unit).",
        "Per-subject 6-panel figures at erd_refined/.",
        "",
        "Visual comparison: open each subject's three figures "
        "(W1/W2/W3) side by side. Differences to watch for:",
        "  - W1 vs W2 (median vs mean, same % unit): how much do mean",
        "    traces wobble vs medians on subjects with heavy artifacts?",
        "  - W2 vs W3 (% vs logratio, same mean aggregator): does the",
        "    logratio rendering look more or less convincing than %?",
        "    Logratio compresses extreme values; % preserves them.",
        "  - Across subjects: does any one viz make the cohort story",
        "    clearer (e.g., bilateral MI dip more visible)?",
    ]
    (OUT_DIR / "phase2_report.txt").write_text("\n".join(report_lines))
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
