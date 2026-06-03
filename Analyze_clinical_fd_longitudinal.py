#!/usr/bin/env python3
"""Feature Distinctiveness (FD) longitudinal trajectory — Kumar Fig 6 analog.

Per-(subject, session) scalar: Riemannian distance² between class
prototypes (MI vs REST Karcher means), normalised by within-class
variance (sum of mean-squared-distance-to-class-prototype across trials,
summed over MI + REST classes). Per Kumar 2024 p. 13:

    FD = δ_r(C̄_MI, C̄_REST)² / (var_MI + var_REST)

where δ_r is the Riemannian (affine-invariant) distance, C̄_k is the
class Karcher mean over trial covariances, and var_k =
mean_{i ∈ k}(δ_r(C_i, C̄_k)²).

Substrate matches the deployed-decoder convention via reuse of helpers
from `Analyze_clinical_gr_ablation.py` (trace-normalised covariance,
subject-specific Shrinkage vs LedoitWolf-adaptive). Channel set is the
subject's deployed-decoder motor montage: 15 channels for SUBJ_003..008,
13 channels for SUBJ_002 (older config).

SUBJ_002 handling: only her right-arm sessions count (S002/S003/S004 per
`Analyze_clinical_gr_ablation.py:172` `CLIN002_RIGHT_ARM_SESSIONS`). S001
is left-arm — markers 100/200 carry opposite semantic content and would
flip the class definition.

Outputs (canonical, not scratch):
  Pictures/clin_analysis/feat_dist/
    feat_dist_per_session.csv         (subject, session_idx, FD, dist², var_MI, var_REST, n_MI, n_REST)
    feat_dist_per_subject_lme.csv     (subject, slope, slope_p, slope_ci_low, slope_ci_high, n_sessions)
    feat_dist_cohort_lme.csv          (cohort-level LME slope across all subjects)
    feat_dist_per_subject_lines.png   (one panel per subject, FD vs session_idx + OLS fit)
    feat_dist_cohort_overlay.png      (all subjects + cohort LME line)
    feat_dist_report.txt              (deliverable)
"""

from __future__ import annotations

import csv
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent
_SWEEP_DIR = _REPO_ROOT / "exploration" / "preprocessing_sweep"
for _p in (str(_REPO_ROOT), str(_SWEEP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

from pyriemann.utils.mean import mean_riemann  # noqa: E402
from pyriemann.utils.distance import distance_riemann  # noqa: E402

# Reuse gr_ablation's per-trial cov pipeline (trace-norm + shrinkage)
# and subject-specific configuration (channel set, shrinkage backend,
# valid-session filter). Keeping these in a single source-of-truth file
# avoids the SUBJ_002 handling-drift the synthesis doc flags as P0.
from Analyze_clinical_gr_ablation import (  # noqa: E402
    CLIN002_RIGHT_ARM_SESSIONS, MOTOR_CHANNELS_13, MOTOR_CHANNELS_15,
    SCALAR_WIN, TRIAL_WIN, _motor_channels_for, _config_a_mu_epochs,
    _restrict_to_motor, _runtime_shrinkage_for, _trial_covs,
)
from exploration.clinical_analysis._helpers import (  # noqa: E402
    clin_pictures_root, enumerate_clin_subjects,
    enumerate_online_sessions_for_subject, session_idx_from_label,
)
from sweep_phase2_round2 import load_raw_cached  # noqa: E402

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


# ----------------------------------------------------------------------
# FD computation
# ----------------------------------------------------------------------

MI_LABEL = 200
REST_LABEL = 100


def _within_class_variance(covs: np.ndarray, mean_cov: np.ndarray) -> float:
    """var_k = (1/n_k) * Σ_i δ_r(C_i, C̄_k)² — Kumar p. 13 verbatim.

    The Riemannian "variance of the feature distribution" is the
    arithmetic mean of squared Riemannian distances from each trial cov
    to the class Karcher mean. Returns nan if covs is empty.
    """
    n = covs.shape[0]
    if n == 0:
        return float("nan")
    sq = np.array([distance_riemann(c, mean_cov) ** 2 for c in covs],
                  dtype=np.float64)
    return float(sq.mean())


def _compute_fd(covs_mi: np.ndarray,
                covs_rest: np.ndarray) -> dict:
    """Compute one (subject, session) FD scalar per the Kumar formula.

    Returns a dict with the FD value plus the components needed for
    downstream interpretation (dist², per-class variance, per-class n).
    """
    if covs_mi.shape[0] < 2 or covs_rest.shape[0] < 2:
        return {
            "fd": float("nan"), "dist_sq": float("nan"),
            "var_mi": float("nan"), "var_rest": float("nan"),
            "n_mi": int(covs_mi.shape[0]),
            "n_rest": int(covs_rest.shape[0]),
        }
    mean_mi = mean_riemann(covs_mi)
    mean_rest = mean_riemann(covs_rest)
    dist_sq = float(distance_riemann(mean_mi, mean_rest) ** 2)
    var_mi = _within_class_variance(covs_mi, mean_mi)
    var_rest = _within_class_variance(covs_rest, mean_rest)
    denom = var_mi + var_rest
    fd = float(dist_sq / denom) if denom > 0 else float("nan")
    return {
        "fd": fd, "dist_sq": dist_sq,
        "var_mi": var_mi, "var_rest": var_rest,
        "n_mi": int(covs_mi.shape[0]),
        "n_rest": int(covs_rest.shape[0]),
    }


# ----------------------------------------------------------------------
# Per-session FD via the gr_ablation preprocessing chain
# ----------------------------------------------------------------------

def _fd_for_session(subject: str, session: str) -> dict | None:
    """Run the canonical preprocessing on one (subject, session), then
    compute FD on the deployed-decoder motor channel set. Returns None if
    the session has too few clean trials to estimate two class means.
    """
    raw, events, event_dict = load_raw_cached(subject, session)
    out = _config_a_mu_epochs(raw, events, event_dict)
    data, labels, _event_samples, _dropped, ch_names = out
    if data is None or labels is None:
        return None
    motor_chs = _motor_channels_for(subject)
    data_motor, motor_kept = _restrict_to_motor(data, ch_names, motor_chs)
    if data_motor.size == 0:
        return None
    use_lw, lam = _runtime_shrinkage_for(subject, session)
    covs = _trial_covs(data_motor, use_lw=use_lw, lam=lam)
    mi_mask = labels == MI_LABEL
    rest_mask = labels == REST_LABEL
    return _compute_fd(covs[mi_mask], covs[rest_mask])


# ----------------------------------------------------------------------
# Valid session enumeration with SUBJ_002 right-arm filter
# ----------------------------------------------------------------------

def _valid_sessions(subject: str) -> list[str]:
    """Sessions we include in the FD trajectory.

    For SUBJ_002 only her right-arm sessions count
    (`Analyze_clinical_gr_ablation.py:172`). S001 was left-arm and the
    marker semantics flip there — including it would invert MI/REST and
    contaminate her FD trajectory.
    """
    sessions = enumerate_online_sessions_for_subject(subject)
    if subject == "CLIN_SUBJ_002":
        return [s for s in sessions if s in CLIN002_RIGHT_ARM_SESSIONS]
    return sessions


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def _plot_per_subject_lines(rows: list[dict], out_path: Path):
    """One panel per subject: FD vs session_idx with OLS fit overlay."""
    subjects = sorted({r["subject"] for r in rows})
    n = len(subjects)
    cols = 4
    n_rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        n_rows, cols, figsize=(cols * 3.5, n_rows * 3.0),
        sharey=True,
    )
    axes = np.atleast_2d(axes).reshape(n_rows, cols)
    for ax in axes.ravel():
        ax.axis("off")
    for i, subj in enumerate(subjects):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        ax.axis("on")
        srows = [x for x in rows if x["subject"] == subj
                 and not np.isnan(x["fd"])]
        if len(srows) < 2:
            ax.set_title(f"{subj} (n<2)", fontsize=9)
            continue
        x = np.array([x["session_idx"] for x in srows], dtype=float)
        y = np.array([x["fd"] for x in srows], dtype=float)
        ax.scatter(x, y, color="C0", s=42, zorder=3)
        # OLS slope (numpy.polyfit; statsmodels LME is computed separately
        # at the cohort level).
        if len(x) >= 2 and np.std(x) > 0:
            m, b = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 50)
            ax.plot(xx, m * xx + b, color="C3", lw=1.4,
                    label=f"slope={m:+.3f}")
            ax.legend(loc="best", fontsize=7)
        ax.set_title(subj, fontsize=10)
        ax.set_xlabel("session_idx", fontsize=8)
        if c == 0:
            ax.set_ylabel("FD = δ²/(var_MI+var_REST)", fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle(
        "Feature Distinctiveness vs session — Kumar Fig 6 analog "
        "(CLIN cohort)\n"
        "Per-trial trace-normalised + shrunk covariance on the deployed-"
        "decoder motor channel set (15 ch SUBJ_003..008, 13 ch SUBJ_002).\n"
        "MI vs REST Karcher-mean distance² normalised by within-class "
        "Riemannian variance. SUBJ_002: right-arm sessions only.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_cohort_overlay(rows: list[dict], cohort_lme: dict | None,
                         out_path: Path):
    """All subjects overlaid; cohort LME slope as a heavy black line."""
    fig, ax = plt.subplots(figsize=(8, 5))
    subjects = sorted({r["subject"] for r in rows})
    cmap = plt.get_cmap("tab10")
    for i, subj in enumerate(subjects):
        srows = [x for x in rows if x["subject"] == subj
                 and not np.isnan(x["fd"])]
        if not srows:
            continue
        x = np.array([x["session_idx"] for x in srows], dtype=float)
        y = np.array([x["fd"] for x in srows], dtype=float)
        ax.plot(x, y, color=cmap(i % 10), marker="o",
                label=subj.replace("CLIN_SUBJ_", "S"))
    if cohort_lme is not None and not np.isnan(cohort_lme.get(
            "slope", float("nan"))):
        slope = cohort_lme["slope"]
        intercept = cohort_lme["intercept"]
        p = cohort_lme.get("p", float("nan"))
        all_x = np.array([x["session_idx"] for x in rows
                          if not np.isnan(x["fd"])], dtype=float)
        if all_x.size > 0:
            xx = np.linspace(all_x.min(), all_x.max(), 50)
            ax.plot(xx, slope * xx + intercept, color="k", lw=2.5,
                    label=f"cohort LME slope={slope:+.3f} (p={p:.3g})")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("Session index")
    ax.set_ylabel("FD = δ²_r(C̄_MI, C̄_REST) / (var_MI + var_REST)")
    ax.set_title(
        "Feature Distinctiveness across sessions — Kumar Fig 6 analog\n"
        "CLIN longitudinal cohort (SUBJ_002 right-arm only)",
        fontsize=11,
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# LME
# ----------------------------------------------------------------------

def _fit_cohort_lme(rows: list[dict]) -> dict | None:
    """Fit `FD ~ session_idx + (1|subject)` via statsmodels MixedLM.

    Per Kumar p. 14 statistical analyses ("metric ~ 1 + (1|subjects) +
    group") with `group` swapped for `session_idx` to match Fig 6's
    longitudinal test. Returns dict with slope, p, CI, n.
    """
    if not HAS_STATSMODELS:
        return None
    import pandas as pd
    df = pd.DataFrame([r for r in rows if not np.isnan(r["fd"])])
    if df.empty or df["subject"].nunique() < 2:
        return None
    df["session_idx"] = df["session_idx"].astype(float)
    try:
        model = smf.mixedlm(
            "fd ~ session_idx", data=df, groups=df["subject"],
        ).fit(method="lbfgs", reml=False)
    except Exception:
        return None
    slope = float(model.params["session_idx"])
    intercept = float(model.params["Intercept"])
    p = float(model.pvalues["session_idx"])
    ci = model.conf_int().loc["session_idx"]
    return {
        "slope": slope, "intercept": intercept, "p": p,
        "ci_low": float(ci[0]), "ci_high": float(ci[1]),
        "n_obs": int(len(df)), "n_subjects": int(df["subject"].nunique()),
    }


def _fit_per_subject_ols(rows: list[dict]) -> list[dict]:
    """Per-subject OLS slope on FD vs session_idx — straight numpy
    polyfit so we don't need statsmodels for the per-subject report."""
    out = []
    for subj in sorted({r["subject"] for r in rows}):
        srows = [x for x in rows if x["subject"] == subj
                 and not np.isnan(x["fd"])]
        if len(srows) < 2:
            out.append({
                "subject": subj, "slope": float("nan"),
                "intercept": float("nan"), "r": float("nan"),
                "n_sessions": len(srows),
            })
            continue
        x = np.array([x["session_idx"] for x in srows], dtype=float)
        y = np.array([x["fd"] for x in srows], dtype=float)
        if np.std(x) == 0:
            out.append({
                "subject": subj, "slope": float("nan"),
                "intercept": float("nan"), "r": float("nan"),
                "n_sessions": len(srows),
            })
            continue
        m, b = np.polyfit(x, y, 1)
        # Pearson correlation
        rxy = float(np.corrcoef(x, y)[0, 1])
        out.append({
            "subject": subj, "slope": float(m), "intercept": float(b),
            "r": rxy, "n_sessions": len(srows),
        })
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    t_start = time.time()
    out_dir = clin_pictures_root() / "feat_dist"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] out_dir={out_dir}")
    subjects = enumerate_clin_subjects()
    print(f"[setup] subjects={subjects}")

    rows: list[dict] = []
    for subject in subjects:
        sessions = _valid_sessions(subject)
        motor_chs = _motor_channels_for(subject)
        print(f"\n=== {subject} ({len(sessions)} valid sessions, "
              f"{len(motor_chs)} motor ch) ===")
        for sess in sessions:
            t_sess = time.time()
            try:
                fd = _fd_for_session(subject, sess)
            except Exception as e:
                print(f"  {sess}: FAILED ({type(e).__name__}: {e})")
                continue
            if fd is None:
                print(f"  {sess}: no usable trials, skipping")
                continue
            sidx = session_idx_from_label(sess)
            row = {
                "subject": subject, "session": sess,
                "session_idx": sidx,
                "fd": fd["fd"], "dist_sq": fd["dist_sq"],
                "var_mi": fd["var_mi"], "var_rest": fd["var_rest"],
                "n_mi": fd["n_mi"], "n_rest": fd["n_rest"],
            }
            rows.append(row)
            print(f"  {sess} (sidx={sidx}): FD={fd['fd']:.4f}  "
                  f"dist²={fd['dist_sq']:.4f}  var_MI={fd['var_mi']:.4f}  "
                  f"var_REST={fd['var_rest']:.4f}  "
                  f"n_MI={fd['n_mi']} n_REST={fd['n_rest']}  "
                  f"({time.time() - t_sess:.1f}s)")

    # Write per-session CSV.
    csv_path = out_dir / "feat_dist_per_session.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {csv_path.name}  ({len(rows)} rows)")

    # Per-subject OLS slopes.
    per_subj = _fit_per_subject_ols(rows)
    ps_path = out_dir / "feat_dist_per_subject_lme.csv"
    with open(ps_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_subj[0].keys()))
        w.writeheader()
        for r in per_subj:
            w.writerow(r)
    print(f"wrote {ps_path.name}  ({len(per_subj)} subjects)")

    # Cohort LME.
    cohort_lme = _fit_cohort_lme(rows)
    coh_path = out_dir / "feat_dist_cohort_lme.csv"
    if cohort_lme is None:
        coh_path.write_text("# LME not computed (no statsmodels or n<2)\n")
    else:
        with open(coh_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cohort_lme.keys()))
            w.writeheader()
            w.writerow(cohort_lme)
    print(f"wrote {coh_path.name}")

    # Plots.
    _plot_per_subject_lines(
        rows, out_dir / "feat_dist_per_subject_lines.png",
    )
    print("wrote feat_dist_per_subject_lines.png")
    _plot_cohort_overlay(
        rows, cohort_lme, out_dir / "feat_dist_cohort_overlay.png",
    )
    print("wrote feat_dist_cohort_overlay.png")

    # Deliverable text report.
    report = [
        "Kumar Fig 6 analog — Feature Distinctiveness across sessions",
        "============================================================",
        f"Total wall-time: {time.time() - t_start:.1f}s",
        f"Per-session rows: {len(rows)}",
        "",
        "Per-subject OLS slope (FD vs session_idx):",
    ]
    for ps in per_subj:
        report.append(
            f"  {ps['subject']:18}  slope={ps['slope']:+.4f}  "
            f"r={ps['r']:+.3f}  n_sessions={ps['n_sessions']}"
        )
    report.append("")
    if cohort_lme is not None:
        report.append(
            f"Cohort LME (fd ~ session_idx + (1|subject)):  "
            f"slope={cohort_lme['slope']:+.4f}  "
            f"p={cohort_lme['p']:.4g}  "
            f"95% CI [{cohort_lme['ci_low']:+.4f}, "
            f"{cohort_lme['ci_high']:+.4f}]  "
            f"n_obs={cohort_lme['n_obs']}  "
            f"n_subj={cohort_lme['n_subjects']}"
        )
    else:
        report.append("Cohort LME: not computed (statsmodels unavailable)")
    report.append("")
    report.append("Outputs:")
    report.append("  feat_dist_per_session.csv")
    report.append("  feat_dist_per_subject_lme.csv")
    report.append("  feat_dist_cohort_lme.csv")
    report.append("  feat_dist_per_subject_lines.png")
    report.append("  feat_dist_cohort_overlay.png")
    (out_dir / "feat_dist_report.txt").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
