"""ECG / heart-rate analysis for the CLIN online cohort.

Question (user, 2026-06-07): does heart rate change between the protocol's
phases — pre-cue baseline vs Rest vs Motor Imagery? The primary readout is a
*within-trial* change score (ΔHR = task-window mean HR − that trial's own
pre-cue baseline mean HR), computed separately for MI and REST and aggregated
per participant, with the three absolute condition means reported descriptively.

Why this is new. The EEG pipeline discards every auxiliary channel
(`exploration/preprocessing_sweep/sweep_phase2_round2.py:123`,
`non_eeg = {"AUX1".."AUX9","TRIGGER"}`). The ECG was recorded on those aux
channels and has never been looked at. `config.py:27` reserves AUX1 as EOG, so
the cardiac signal lives on one of AUX2/3/7/8/9 — in practice AUX7 when present,
but the script auto-selects per recording rather than hardcoding it.

Not every session has a usable ECG (some aux channels are pure noise). Each run
is gated: a clean sinus ECG produces R-peaks with a very low inter-beat-interval
coefficient of variation (≤~0.07 in this data) and near-perfect beat-template
consistency, while noise sessions collapse to all five aux channels sharing an
identical HR≈107 / CV≈0.26 common-mode signature. The gate (IBI CV ≤ 0.12,
template corr ≥ 0.97, HR ∈ [40,150]) sits in the wide empty gap between the two
regimes. Every run's channel choice, metrics, and pass/fail verdict are written
to ecg_channel_selection.csv for audit.

Trial definition (cue onset, end marker, duration window) mirrors the EEG
analysis exactly (`sweep_phase2_round2.py:138-159`): marker 200 = MI cue,
100 = REST cue, +20 = trial end. SUBJ_002 S001 (left-arm, flipped MI/REST
semantics — `_subj002.py:20-22`) is excluded; ECG is decoder-independent so
002 otherwise uses its feature-family right-arm sessions (S002–S004).

Run (Windows): `C:\\Users\\arman\\miniconda3\\envs\\lsl\\python.exe`
exploration/clinical_analysis/explore_ecg_heart_rate.py
Outputs land in ~/Pictures/clin_analysis/ecg_hr/.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Package-relative imports work whether run as a script or a module.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _helpers import (  # noqa: E402
    enumerate_clin_subjects,
    enumerate_online_sessions_for_subject,
    clin_pictures_root,
)
from _subj002 import subj002_feature_sessions, is_subj002  # noqa: E402

# Read-only Tier 1 helpers (freely importable per CLAUDE.md analysis policy).
from Utils.stream_utils import load_xdf, get_channel_names_from_xdf  # noqa: E402
from config import DATA_DIR  # noqa: E402

warnings.filterwarnings("ignore")

# Report text uses the Δ glyph; Windows stdout defaults to cp1252, so force
# UTF-8 to avoid a UnicodeEncodeError on print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

FS = 512  # EEG/aux sample rate (sweep_phase2_round2.py:63). Aux shares the EEG clock.

# Candidate ECG channels: every aux input except AUX1 (EOG, config.py:27) and
# TRIGGER. AUX7 is the usual cardiac jack but selection is empirical per run.
ECG_CANDIDATES = ["AUX2", "AUX3", "AUX7", "AUX8", "AUX9"]

# ECG-validity gate (calibrated on first-session survey, 2026-06-07): clean ECG
# sits at CV≤0.07 / tmpl≥1.00; noise at CV≈0.26 / tmpl≤0.86. Thresholds sit in
# the empty gap so the choice is insensitive to small shifts.
GATE_MAX_IBI_CV = 0.12
GATE_MIN_TEMPLATE_CORR = 0.97
GATE_HR_RANGE = (40.0, 150.0)
MIN_BEATS = 30

# Marker codes (sweep_phase2_round2.py:140-152, :333-334).
CUE_CODES = {200: "MI", 100: "REST"}
TRIAL_MIN_DUR, TRIAL_MAX_DUR, TRIAL_EPS = 1.0, 5.5, 0.02

# Condition windows (within-trial ΔHR design, user-confirmed 2026-06-07).
BASELINE_WIN = (-4.0, -1.0)   # quiet pre-cue, relative to cue onset
TASK_MAX = 5.0                # cap the task window at cue+5 s
HR_GRID_HZ = 4.0              # uniform instantaneous-HR resample rate
TC_WIN = (-5.0, 8.0)         # cue-locked timecourse window for plotting

OUT_DIR = clin_pictures_root() / "ecg_hr"


# ----------------------------------------------------------------------
# Per-run loading (keeps aux channels, unlike the EEG loader)
# ----------------------------------------------------------------------

def _canonical_runs(subject: str, session: str) -> list[str]:
    """Canonical '..._eeg.xdf' run files for a session (excludes repairs)."""
    eegdir = os.path.join(DATA_DIR, f"sub-{subject}", f"ses-{session}", "eeg")
    if not os.path.isdir(eegdir):
        return []
    return sorted(
        os.path.join(eegdir, f)
        for f in os.listdir(eegdir)
        if os.path.splitext(f)[0].endswith("_eeg")
    )


def _load_run(path: str):
    """Return (aux_dict, eeg_ts, cues) for one run.

    aux_dict maps candidate channel name → float sample vector (EEG clock).
    cues is a list of (code, t_onset, t_end) in absolute LSL seconds, filtered
    to the same validity window the EEG analysis uses.
    """
    es, ms = load_xdf(path, report=False)
    ch = get_channel_names_from_xdf(es)
    data = np.asarray(es["time_series"]).T.astype(float)
    eeg_ts = np.asarray(es["time_stamps"], dtype=float)
    aux = {c: data[ch.index(c)] for c in ECG_CANDIDATES if c in ch}

    md = np.array([int(v[0]) for v in ms["time_series"]])
    mt = np.array([float(v[1]) for v in ms["time_series"]])
    cues = []
    for idx, code in enumerate(md):
        if code not in CUE_CODES:
            continue
        end_code = code + 20
        end_time = None
        for j in range(idx + 1, len(md)):
            if md[j] == end_code:
                end_time = mt[j]
                break
        if end_time is None:
            continue
        dur = end_time - mt[idx]
        if (dur + TRIAL_EPS) >= TRIAL_MIN_DUR and (dur - TRIAL_EPS) <= TRIAL_MAX_DUR:
            cues.append((CUE_CODES[code], float(mt[idx]), float(end_time)))
    return aux, eeg_ts, cues


# ----------------------------------------------------------------------
# R-peak detection + ECG quality metrics
# ----------------------------------------------------------------------

def _detect_rpeaks(x: np.ndarray):
    """Detect R-peaks on one aux vector.

    Bandpass 8–18 Hz, robust-z (MAD) normalise, then scan beat polarity and
    prominence and keep the configuration that minimises IBI CV (≥MIN_BEATS
    beats). Returns (peak_indices, ibi_cv, hr_median, template_corr) or None.
    The min-CV scan cleanly resolves a real ECG; noise channels still fail the
    downstream gate because no configuration drives their CV low.
    """
    b, a = signal.butter(3, [8 / (FS / 2), 18 / (FS / 2)], "bandpass")
    xf = signal.filtfilt(b, a, x)
    mad = np.median(np.abs(xf - np.median(xf))) + 1e-9
    xf = (xf - np.median(xf)) / mad

    best = None
    for pol in (1, -1):
        s = pol * xf
        for prom in (3, 4, 5, 6, 8):
            pk, _ = signal.find_peaks(s, distance=int(0.4 * FS), prominence=prom)
            if len(pk) < MIN_BEATS:
                continue
            ibi = np.diff(pk) / FS
            ibi = ibi[(ibi > 0.33) & (ibi < 2.0)]
            if len(ibi) < MIN_BEATS:
                continue
            cv = float(np.std(ibi) / np.mean(ibi))
            if best is None or cv < best[1] - 1e-3:
                hr = float(60.0 / np.median(ibi))
                half = int(0.12 * FS)
                segs = [xf[p - half:p + half] for p in pk
                        if p - half >= 0 and p + half < len(xf)]
                if len(segs) > 20:
                    segs = np.asarray(segs)
                    tmpl = np.median(segs, axis=0)
                    tcons = float(np.median(
                        [np.corrcoef(seg, tmpl)[0, 1] for seg in segs]))
                else:
                    tcons = float("nan")
                best = (pk, cv, hr, tcons)
    return best


def _select_ecg_channel(aux: dict):
    """Score every candidate channel, return (best_name, metrics, all_rows).

    best_name is the lowest-CV channel; metrics is its dict; all_rows holds
    per-channel metrics for the audit CSV. best_name may still fail the gate.
    """
    rows = {}
    for name, x in aux.items():
        res = _detect_rpeaks(x)
        if res is None:
            rows[name] = dict(cv=np.nan, hr=np.nan, tmpl=np.nan, nbeat=0)
            continue
        pk, cv, hr, tcons = res
        rows[name] = dict(cv=cv, hr=hr, tmpl=tcons, nbeat=len(pk), _peaks=pk)
    valid = {n: r for n, r in rows.items() if r["nbeat"] >= MIN_BEATS}
    if not valid:
        return None, None, rows
    best_name = min(valid, key=lambda n: valid[n]["cv"])
    return best_name, rows[best_name], rows


def _passes_gate(m: dict) -> bool:
    return (
        m is not None
        and m["nbeat"] >= MIN_BEATS
        and m["cv"] <= GATE_MAX_IBI_CV
        and m["tmpl"] >= GATE_MIN_TEMPLATE_CORR
        and GATE_HR_RANGE[0] <= m["hr"] <= GATE_HR_RANGE[1]
    )


# ----------------------------------------------------------------------
# Instantaneous HR series + trial epoching
# ----------------------------------------------------------------------

def _instantaneous_hr(peaks: np.ndarray, eeg_ts: np.ndarray):
    """Build a continuous HR(t) series on a uniform grid from R-peak indices.

    Artifact beats (IBI outside [0.33,2.0] s, or >30% from the local median)
    are dropped before interpolation. Returns (grid_t, hr) in absolute seconds.
    """
    t_r = eeg_ts[peaks]
    ibi = np.diff(t_r)
    hr_beat = 60.0 / ibi
    t_beat = t_r[1:]

    good = (ibi > 0.33) & (ibi < 2.0)
    med = np.median(hr_beat[good]) if good.any() else np.nan
    good &= np.abs(hr_beat - med) <= 0.30 * med
    t_beat, hr_beat = t_beat[good], hr_beat[good]
    if len(t_beat) < MIN_BEATS:
        return None, None, None

    grid = np.arange(eeg_ts[0], eeg_ts[-1], 1.0 / HR_GRID_HZ)
    hr = np.interp(grid, t_beat, hr_beat)
    return grid, hr, t_beat


def _window_mean(grid, hr, t_beat, lo, hi):
    """Mean HR in [lo,hi] s, or NaN if no real beat lies near the window."""
    if not np.any((t_beat >= lo - 1.5) & (t_beat <= hi + 1.5)):
        return np.nan
    m = (grid >= lo) & (grid <= hi)
    return float(np.mean(hr[m])) if m.any() else np.nan


# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------

def _sessions_for(subject: str) -> list[str]:
    if is_subj002(subject):
        return subj002_feature_sessions()  # excludes flipped left-arm S001
    return enumerate_online_sessions_for_subject(subject)


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chan_rows = []   # one row per run (audit)
    trial_rows = []  # one row per usable trial
    tc_rows = []     # cue-locked timecourse samples

    tc_offsets = np.arange(TC_WIN[0], TC_WIN[1], 1.0 / HR_GRID_HZ)

    for subject in enumerate_clin_subjects():
        for session in _sessions_for(subject):
            for run_path in _canonical_runs(subject, session):
                run_name = os.path.basename(run_path)
                try:
                    aux, eeg_ts, cues = _load_run(run_path)
                except Exception as exc:  # noqa: BLE001 — log & skip a bad file
                    chan_rows.append(dict(
                        subject=subject, session=session, run=run_name,
                        selected="LOAD_ERROR", cv=np.nan, hr=np.nan, tmpl=np.nan,
                        nbeat=0, n_cues=0, passed=False, note=str(exc)[:80]))
                    continue

                best_name, m, allm = _select_ecg_channel(aux)
                passed = _passes_gate(m)
                chan_rows.append(dict(
                    subject=subject, session=session, run=run_name,
                    selected=best_name or "NONE",
                    cv=None if m is None else round(m["cv"], 4),
                    hr=None if m is None else round(m["hr"], 1),
                    tmpl=None if m is None else round(m["tmpl"], 3),
                    nbeat=0 if m is None else m["nbeat"],
                    n_cues=len(cues), passed=passed,
                    note="" if passed else "no usable ECG"))
                if not passed:
                    continue

                grid, hr, t_beat = _instantaneous_hr(allm[best_name]["_peaks"], eeg_ts)
                if grid is None:
                    chan_rows[-1]["passed"] = False
                    chan_rows[-1]["note"] = "HR series too sparse"
                    continue

                for cond, t0, t_end in cues:
                    bl = _window_mean(grid, hr, t_beat, t0 + BASELINE_WIN[0], t0 + BASELINE_WIN[1])
                    tk = _window_mean(grid, hr, t_beat, t0, min(t_end, t0 + TASK_MAX))
                    if np.isnan(bl) or np.isnan(tk):
                        continue
                    trial_rows.append(dict(
                        subject=subject, session=session, run=run_name,
                        cond=cond, baseline_hr=bl, task_hr=tk,
                        delta=tk - bl, dur=t_end - t0))
                    # cue-locked, baseline-subtracted timecourse
                    tc = np.interp(t0 + tc_offsets, grid, hr,
                                   left=np.nan, right=np.nan) - bl
                    for off, val in zip(tc_offsets, tc):
                        if not np.isnan(val):
                            tc_rows.append(dict(subject=subject, cond=cond,
                                                offset=round(float(off), 3), hr=val))

    chan_df = pd.DataFrame(chan_rows)
    trial_df = pd.DataFrame(trial_rows)
    tc_df = pd.DataFrame(tc_rows)

    chan_df.to_csv(OUT_DIR / "ecg_channel_selection.csv", index=False)
    trial_df.to_csv(OUT_DIR / "ecg_hr_trials.csv", index=False)

    _report(chan_df, trial_df)
    if not trial_df.empty:
        _fig_delta_by_subject(trial_df)
        _fig_levels_by_subject(trial_df)
        _fig_timecourse(tc_df)
    print(f"Done. Outputs in {OUT_DIR}")


# ----------------------------------------------------------------------
# Stats report
# ----------------------------------------------------------------------

def _report(chan_df: pd.DataFrame, trial_df: pd.DataFrame):
    lines = ["ECG / HEART-RATE ANALYSIS — CLIN ONLINE COHORT", "=" * 52, ""]

    n_runs = len(chan_df)
    n_pass = int(chan_df["passed"].sum()) if n_runs else 0
    lines += [
        f"Runs scanned: {n_runs}   with usable ECG: {n_pass}   "
        f"rejected (noise/no ECG): {n_runs - n_pass}",
        "",
        "Per-run ECG channel selection & gate (full table in "
        "ecg_channel_selection.csv):",
    ]
    for subj in sorted(chan_df["subject"].unique()):
        sub = chan_df[chan_df["subject"] == subj]
        npass = int(sub["passed"].sum())
        chans = sorted(set(sub.loc[sub["passed"], "selected"]))
        lines.append(f"  {subj}: {npass}/{len(sub)} runs pass"
                     + (f"  (channel {','.join(chans)})" if chans else ""))
    lines.append("")

    if trial_df.empty:
        lines += ["No usable trials — no session passed the ECG gate.", ""]
        (OUT_DIR / "ecg_hr_report.txt").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        return

    # Per-subject means (pool trials across that subject's valid runs).
    per_subj = (trial_df.groupby(["subject", "cond"])
                .agg(delta=("delta", "mean"),
                     baseline=("baseline_hr", "mean"),
                     task=("task_hr", "mean"),
                     n=("delta", "size"))
                .reset_index())
    lines += ["Per-participant means (HR in bpm; ΔHR = task − own pre-cue baseline):",
              f"{'subj':14s}{'cond':6s}{'baseline':>9s}{'task':>8s}{'ΔHR':>8s}{'n_tr':>6s}"]
    for _, r in per_subj.iterrows():
        lines.append(f"{r['subject']:14s}{r['cond']:6s}{r['baseline']:9.1f}"
                     f"{r['task']:8.1f}{r['delta']:+8.2f}{int(r['n']):6d}")
    lines.append("")

    # Cohort tests on per-subject mean ΔHR (n = subjects with valid ECG).
    piv = per_subj.pivot(index="subject", columns="cond", values="delta")
    lines += ["Cohort tests (Wilcoxon signed-rank on per-participant mean ΔHR):"]
    for cond in ("MI", "REST"):
        if cond in piv:
            v = piv[cond].dropna().values
            lines.append(f"  {cond} ΔHR vs 0: " + _wilcox_str(v)
                         + f"   mean={np.mean(v):+.2f} bpm (n={len(v)})")
    if {"MI", "REST"}.issubset(piv.columns):
        both = piv[["MI", "REST"]].dropna()
        if len(both) >= 2:
            try:
                st, p = wilcoxon(both["MI"], both["REST"])
                lines.append(f"  MI vs REST ΔHR (paired): W={st:.1f}, p={p:.4f} "
                             f"(n={len(both)})")
            except ValueError as exc:
                lines.append(f"  MI vs REST ΔHR (paired): undefined ({exc})")
    lines.append("")

    lines.append(_lme_str(trial_df))
    lines += ["",
              "Floor note: with n≈7 participants the two-sided signed-rank "
              "p-value floors at 1/2^(n-1); treat near-floor results as "
              "descriptive, consistent with the EDS analysis."]

    (OUT_DIR / "ecg_hr_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def _wilcox_str(v: np.ndarray) -> str:
    v = v[~np.isnan(v)]
    if len(v) < 2 or np.allclose(v, 0):
        return "W=n/a"
    try:
        st, p = wilcoxon(v)
        return f"W={st:.1f}, p={p:.4f}"
    except ValueError as exc:
        return f"W=n/a ({exc})"


def _lme_str(trial_df: pd.DataFrame) -> str:
    """Trial-level LME: ΔHR ~ condition with a per-subject random intercept."""
    df = trial_df[trial_df["cond"].isin(["MI", "REST"])].copy()
    if df["subject"].nunique() < 3:
        return "LME (ΔHR ~ condition): skipped (<3 participants)."
    try:
        import statsmodels.formula.api as smf
        df["cond"] = pd.Categorical(df["cond"], categories=["REST", "MI"])
        md = smf.mixedlm("delta ~ C(cond)", df, groups=df["subject"])
        fit = md.fit(method="lbfgs", reml=False)
        coef = fit.params.get("C(cond)[T.MI]", float("nan"))
        pval = fit.pvalues.get("C(cond)[T.MI]", float("nan"))
        return ("Trial-level LME ΔHR ~ condition (random intercept / subject):\n"
                f"  MI vs REST effect = {coef:+.2f} bpm, p = {pval:.4f} "
                f"(n_trials={len(df)}, n_subj={df['subject'].nunique()})")
    except Exception as exc:  # noqa: BLE001 — LME is secondary to signed-rank
        return f"Trial-level LME: failed ({exc})."


# ----------------------------------------------------------------------
# Figures (emphasise per-participant heterogeneity)
# ----------------------------------------------------------------------

def _subject_labels(trial_df: pd.DataFrame):
    subs = sorted(trial_df["subject"].unique())
    return subs, {s: f"P{i+1}" for i, s in enumerate(subs)}


def _fig_delta_by_subject(trial_df: pd.DataFrame):
    subs, lab = _subject_labels(trial_df)
    per = (trial_df.groupby(["subject", "cond"])["delta"].mean().unstack())
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(subs))
    for i, s in enumerate(subs):
        for cond, dx, col in (("REST", -0.12, "#4477AA"), ("MI", 0.12, "#CC3311")):
            if cond in per.columns and not np.isnan(per.loc[s, cond]):
                ax.scatter(i + dx, per.loc[s, cond], color=col, s=70, zorder=3,
                           label=cond if i == 0 else None)
        if {"MI", "REST"}.issubset(per.columns):
            ax.plot([i - 0.12, i + 0.12], [per.loc[s, "REST"], per.loc[s, "MI"]],
                    color="gray", lw=0.8, zorder=1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([lab[s] for s in subs])
    ax.set_ylabel("ΔHR vs pre-cue baseline (bpm)")
    ax.set_title("Within-trial heart-rate change by participant — MI vs REST")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_ecg_hr_delta_by_subject.png", dpi=200)
    plt.close(fig)


def _fig_levels_by_subject(trial_df: pd.DataFrame):
    subs, lab = _subject_labels(trial_df)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(subs))
    w = 0.27
    # Pre-cue baseline is shared per trial; show MI-trial baseline as "baseline".
    bl = trial_df.groupby("subject")["baseline_hr"].mean()
    rest = trial_df[trial_df.cond == "REST"].groupby("subject")["task_hr"].mean()
    mi = trial_df[trial_df.cond == "MI"].groupby("subject")["task_hr"].mean()
    for off, ser, col, name in ((-w, bl, "#999999", "pre-cue baseline"),
                                (0.0, rest, "#4477AA", "REST"),
                                (w, mi, "#CC3311", "MI")):
        ax.bar(x + off, [ser.get(s, np.nan) for s in subs], width=w,
               color=col, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([lab[s] for s in subs])
    ax.set_ylabel("Mean heart rate (bpm)")
    ax.set_title("Absolute heart rate by protocol phase, per participant")
    ax.set_ylim(bottom=max(0, np.nanmin([bl.min(), rest.min(), mi.min()]) - 10))
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_ecg_hr_levels_by_subject.png", dpi=200)
    plt.close(fig)


def _fig_timecourse(tc_df: pd.DataFrame):
    if tc_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    subs = sorted(tc_df["subject"].unique())
    for ax, cond in zip(axes, ("MI", "REST")):
        d = tc_df[tc_df.cond == cond]
        for s in subs:  # faint per-participant lines
            ds = d[d.subject == s].groupby("offset")["hr"].mean()
            ax.plot(ds.index, ds.values, color="gray", lw=0.6, alpha=0.45)
        g = d.groupby("offset")["hr"]
        mean, n = g.mean(), g.count()
        se = g.std() / np.sqrt(n.where(n > 1, np.nan))
        col = "#CC3311" if cond == "MI" else "#4477AA"
        ax.plot(mean.index, mean.values, color=col, lw=2.2, label="cohort mean")
        ax.fill_between(mean.index, mean - se, mean + se, color=col, alpha=0.2)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(cond)
        ax.set_xlabel("time from cue (s)")
    axes[0].set_ylabel("ΔHR vs pre-cue baseline (bpm)")
    fig.suptitle("Cue-locked heart-rate response (per-participant + cohort mean ±SE)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_ecg_hr_timecourse.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run()
