#!/usr/bin/env python3
"""
Compare performance across SUBJECTS / SESSIONS / RUNS.

Computes (per run):
  - Accuracy (%) per class (MI vs REST):
      • Decision accuracy: excludes ambiguous trials from the denominator
      • Total accuracy: includes ambiguous trials in the denominator
    NOTE: Per-class accuracy is computed from decoder_output*.csv (trial-final Predicted Label),
          because event_log ambiguity counts are not reliably class-split.

  - Bar dynamics per class (MI vs REST):
      • Lean16Hz per trial = % classifier samples where correct-class prob > THRESH
      • Per-run BarDyn_MI / BarDyn_REST = mean Lean16Hz across MI/REST trials in that run

Then generates plots across:
  - GROUP_LEVEL="run"     : bar plot (no variability needed)
  - GROUP_LEVEL="session" : box+whisker across runs within each session (+ optional dots)
  - GROUP_LEVEL="subject" : box+whisker across sessions/runs within each subject (+ optional dots)

By default, plots show separate bars/boxes for each Class (MI vs REST), but:
  - ROLL_CLASSES=True  -> collapses MI+REST into one distribution ("ALL")
  - SPLIT_BY_SUBJECT=True -> forces x-axis grouping to SubjectID (each subject gets its own box/bar)

Directory layout assumed:
  ~/Documents/CurrentStudy/
    sub-<SUBJECT>/
      ses-XXXX.../
        logs/
          ONLINE_YYYY.../
            decoder_output*.csv
            event_log.txt (optional)
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# USER CONFIG
# =========================================================

CURRENT_STUDY_ROOT = os.path.expanduser("~/Documents/CurrentStudy")

# ---- Selection (non-interactive) ----
# You can either list explicit subject IDs (without "sub-") ...
SUBJECTS = [
    # "F25CLASS_SUBJ_001",
    # "F25CLASS_SUBJ_002",
]

# ... OR use a glob-like prefix filter if SUBJECTS is empty.
SUBJECT_PREFIX_FILTER = "CLIN_SUBJ_"   # e.g. "F25CLASS_SUBJ_" or "CLIN_SUBJ_" or "" for ALL subjects

# Sessions: provide dict subject->list of session folder names (e.g., "ses-S001ONLINE") to include.
# If None, includes ALL sessions found (subject to exclusion rules).
#
# This dict currently corresponds to your "REGULAR online" selection per your description.
# For BIMANUAL, swap this dict for the bimanual session lists and re-run.

#SESSIONS_BY_SUBJECT = None
'''
SESSIONS_BY_SUBJECT = {
    "F25CLASS_SUBJ_001": ["ses-S002ONLINE"],
    "F25CLASS_SUBJ_002": ["ses-S002ONLINE"],
    "F25CLASS_SUBJ_003": ["ses-S002ONLINE"],
    
    "F25CLASS_SUBJ_004": ["ses-S002ONLINE","ses-S003ONLINE"],
    "F25CLASS_SUBJ_005": ["ses-S002ONLINE","ses-S003ONLINE"],
    "F25CLASS_SUBJ_006": ["ses-S002ONLINE","ses-S003ONLINE"],

    # Subjects 4–6: no regular sessions → intentionally omitted

    "F25CLASS_SUBJ_007": ["ses-S003ONLINE"],
    "F25CLASS_SUBJ_008": ["ses-S003ONLINE"],
    "F25CLASS_SUBJ_009": ["ses-S003ONLINE"],
}
'''

SESSIONS_BY_SUBJECT = {
    "CLIN_SUBJ_002": ["ses-S002ONLINE","ses-S004ONLINE"],
    "CLIN_SUBJ_003": ["ses-S001ONLINE","ses-S002ONLINE","ses-S003ONLINE","ses-S004ONLINE","ses-S005ONLINE"],
    "CLIN_SUBJ_004": ["ses-S001ONLINE","ses-S002ONLINE","ses-S003ONLINE","ses-S004ONLINE","ses-S005ONLINE"], 
}

# Runs: leave as None for "all ONLINE_ runs in logs/".
# Or provide a dict (subject, session)->list of run folder names.
RUNS_BY_SUBJECT_SESSION = None

# ---- Bar dynamics definition (Lean16Hz internal) ----
THRESH = 0.50  # correct-class probability threshold for "lean" (internal)

# ---- Accuracy definition ----
ACCURACY_MODE = "decision"      # "decision" (exclude ambiguous) OR "total" (include ambiguous)
ACCURACY_SOURCE = "csv"      # "csv" recommended for class-split accuracy; "event_log" provides only overall

# ---- Plot control ----
DO_PLOTS = True
METRICS_TO_PLOT = ["accuracy", "bar_dynamics"]  # subset of ["accuracy", "bar_dynamics"]

GROUP_LEVEL = "session"    # "run", "session", or "subject"
PLOT_STYLE = "box"         # "box" or "bar"
SHOW_DOTS = True           # overlay individual run points on boxplots
SHOW_RUN_LABELS_ON_BAR = False  # if GROUP_LEVEL=="run" and PLOT_STYLE=="bar", label x ticks heavily.
BOX_WIDTH = 0.25           # skinnier boxes (0.20–0.30 is typical)

# ---- New toggles ----
ROLL_CLASSES = False        # if True: MI + REST collapsed to one distribution ("ALL")
SPLIT_BY_SUBJECT = False    # if True: x-axis is SubjectID regardless of GROUP_LEVEL

# ---- Output ----
SAVE_SUMMARY_CSV = True
SUMMARY_CSV_NAME = "performance_summary_runs.csv"


# =========================================================
# EXCLUSION RULES
# =========================================================

EXCLUDE_SESSION_KEYWORDS = [
    "DEBUG",
    "TEST",
    "CALIB",
    "PRACTICE",
    "NOISE",
    "OLD",
    "OFFLINE",
]

CLASS_PALETTE = {
    "MI": "tab:orange",
    "REST": "tab:blue",
    "ALL": "tab:gray",
}


# =========================================================
# Helpers
# =========================================================

def list_subject_dirs(root: str) -> list[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"❌ CurrentStudy root not found: {root}")
    return sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and d.startswith("sub-")
    ])


def subject_dir_to_id(sub_dir: str) -> str:
    return sub_dir[len("sub-"):] if sub_dir.startswith("sub-") else sub_dir


def is_valid_session(ses_name: str) -> bool:
    return not any(k in ses_name.upper() for k in EXCLUDE_SESSION_KEYWORDS)


def safe_iqr(a: np.ndarray) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    q75, q25 = np.percentile(a, [75, 25])
    return float(q75 - q25)


def find_decoder_csv(run_path: str) -> str | None:
    if not os.path.isdir(run_path):
        return None
    csv_files = [
        f for f in os.listdir(run_path)
        if f.startswith("decoder_output") and f.endswith(".csv")
    ]
    if not csv_files:
        return None
    return os.path.join(run_path, sorted(csv_files)[0])


def session_sort_key(s: str) -> int:
    # ses-S002ONLINE should come before ses-S003ONLINE, etc.
    m = re.search(r"S(\d+)", s)
    return int(m.group(1)) if m else 999


def parse_confusion_from_event_log(event_log_path: str):
    """
    Returns:
      cm (2x2): [[a200_p200, a200_p100],[a100_p200, a100_p100]]
      ambiguous (int)
    or (None, None) if parse fails.
    """
    if not os.path.exists(event_log_path):
        return None, None

    with open(event_log_path, "r") as f:
        txt = f.read()

    m = re.search(
        r"Actual 200.*?MI\): (\d+).*?REST\): (\d+).*?"
        r"Actual 100.*?MI\): (\d+).*?REST\): (\d+)",
        txt,
        re.DOTALL
    )
    amb = re.search(r"Ambiguous trials.*?: (\d+)", txt)

    if not m:
        return None, None

    a200_p200 = int(m.group(1))
    a200_p100 = int(m.group(2))
    a100_p200 = int(m.group(3))
    a100_p100 = int(m.group(4))
    cm = np.array([[a200_p200, a200_p100],
                   [a100_p200, a100_p100]], dtype=int)

    ambiguous = int(amb.group(1)) if amb else 0
    return cm, ambiguous


def build_unified_prob_cols(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """
    Adds MI_prob and REST_prob handling old/new schema and mixed sessions.
    Returns (df, MI_COL, REST_COL).
    """
    has_old = ("P(MI)" in df.columns) and ("P(REST)" in df.columns)
    has_new = ("P(MI)_inst" in df.columns) and ("P(REST)_inst" in df.columns)

    if not (has_old or has_new):
        raise ValueError(
            "Missing probability columns: need either old (P(MI), P(REST)) "
            "or new (P(MI)_inst, P(REST)_inst) schema."
        )

    if has_old and has_new:
        df["MI_prob"] = df.get("P(MI)_inst").combine_first(df.get("P(MI)"))
        df["REST_prob"] = df.get("P(REST)_inst").combine_first(df.get("P(REST)"))
    elif has_new:
        df["MI_prob"] = df["P(MI)_inst"]
        df["REST_prob"] = df["P(REST)_inst"]
    else:
        df["MI_prob"] = df["P(MI)"]
        df["REST_prob"] = df["P(REST)"]

    return df, "MI_prob", "REST_prob"


def compute_lean16hz_per_trial(df_run: pd.DataFrame, MI_COL: str, REST_COL: str, thresh: float) -> pd.DataFrame:
    """
    Per run, per trial (GlobalTrialID): Lean% = % samples where correct-class prob > thresh.
    Returns per-trial dataframe: GlobalTrialID, Class, LeanPct
    """
    required = ["GlobalTrialID", "True Label", "Phase", MI_COL, REST_COL]
    for c in required:
        if c not in df_run.columns:
            raise ValueError(f"Missing required column for bar dynamics: {c}")

    df_run = df_run[df_run["Phase"] != "ROBOT"].copy()

    records = []
    for gtid, tdf in df_run.groupby("GlobalTrialID"):
        if tdf.empty:
            continue

        true_label = int(tdf["True Label"].iloc[0])
        if true_label == 200:
            p = tdf[MI_COL].astype(float).values
            cls = "MI"
        elif true_label == 100:
            p = tdf[REST_COL].astype(float).values
            cls = "REST"
        else:
            continue

        p = p[np.isfinite(p)]
        if p.size == 0:
            continue

        lean_pct = float((p > thresh).mean() * 100.0)
        records.append({"GlobalTrialID": gtid, "Class": cls, "LeanPct": lean_pct})

    return pd.DataFrame(records)


def compute_accuracy_by_class_from_csv_trials(df_run: pd.DataFrame) -> dict:
    """
    Per run, per class accuracy using final Predicted Label per trial.
    Returns dict with:
      AccDecision_MI, AccDecision_REST
      AccTotal_MI, AccTotal_REST
      Ambiguous_MI, Ambiguous_REST
      Trials_MI, Trials_REST
      Decisions_MI, Decisions_REST
    """
    required = ["GlobalTrialID", "True Label", "Predicted Label", "Phase"]
    for c in required:
        if c not in df_run.columns:
            return {}

    df_run = df_run[df_run["Phase"] != "ROBOT"].copy()

    if "Timestamp" in df_run.columns:
        df_sorted = df_run.sort_values("Timestamp")
    else:
        df_sorted = df_run.copy()

    last = df_sorted.groupby("GlobalTrialID", as_index=False).tail(1)

    out = {}
    for true_lab, name in [(200, "MI"), (100, "REST")]:
        true_num = pd.to_numeric(last["True Label"], errors="coerce").astype("Int64")
        sub = last[true_num == true_lab].copy()

        trials_n = int(sub.shape[0])
        if trials_n == 0:
            out[f"AccDecision_{name}"] = np.nan
            out[f"AccTotal_{name}"] = np.nan
            out[f"Ambiguous_{name}"] = 0
            out[f"Trials_{name}"] = 0
            out[f"Decisions_{name}"] = 0
            continue

        preds = pd.to_numeric(sub["Predicted Label"], errors="coerce")
        decided_mask = preds.isin([100, 200])

        decisions_n = int(decided_mask.sum())
        ambiguous_n = int((~decided_mask).sum())
        correct_n = int((preds[decided_mask].astype(int) == true_lab).sum())

        acc_dec = 100.0 * correct_n / decisions_n if decisions_n > 0 else np.nan
        acc_tot = 100.0 * correct_n / trials_n if trials_n > 0 else np.nan

        out[f"AccDecision_{name}"] = float(acc_dec)
        out[f"AccTotal_{name}"] = float(acc_tot)
        out[f"Ambiguous_{name}"] = int(ambiguous_n)
        out[f"Trials_{name}"] = int(trials_n)
        out[f"Decisions_{name}"] = int(decisions_n)

    return out


def compute_overall_accuracy_from_event_log(run_path: str) -> tuple[float, float, int]:
    """
    Overall (not per-class) accuracy from event_log, if available.
    Returns (AccDecision, AccTotal, AmbiguousN) or (nan, nan, 0) if parse fails.
    """
    cm, ambiguous = parse_confusion_from_event_log(os.path.join(run_path, "event_log.txt"))
    if cm is None:
        return np.nan, np.nan, 0

    total_decisions = int(cm.sum())
    correct = int(cm[0, 0] + cm[1, 1])
    ambiguous = int(ambiguous or 0)

    total_trials_including_amb = total_decisions + ambiguous

    AccDecision = 100.0 * correct / total_decisions if total_decisions > 0 else np.nan
    AccTotal = 100.0 * correct / total_trials_including_amb if total_trials_including_amb > 0 else np.nan
    return float(AccDecision), float(AccTotal), int(ambiguous)


def shorten_run_name(run_folder: str, maxlen: int = 28) -> str:
    if len(run_folder) <= maxlen:
        return run_folder
    return run_folder[:maxlen - 3] + "..."


def pick_xcol_for_group() -> str:
    # If SPLIT_BY_SUBJECT is on, force x to SubjectID (each subject gets its own box/bar)
    if SPLIT_BY_SUBJECT:
        return "SubjectID"

    if GROUP_LEVEL == "run":
        return "RunID"
    if GROUP_LEVEL == "session":
        return "SessionID"
    if GROUP_LEVEL == "subject":
        return "SubjectID"
    raise ValueError(f"Unknown GROUP_LEVEL: {GROUP_LEVEL}")


# =========================================================
# Plotting (long-form)
# =========================================================

def make_long_for_plot(runs_df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, str, str]:
    """
    Returns (df_long, ycol, y_label)
      df_long columns: SubjectID, SessionID, RunID, RunShort, Class, Value
    """
    base_cols = ["SubjectID", "SessionID", "RunID", "RunShort"]

    if metric == "accuracy":
        if ACCURACY_MODE == "decision":
            y_mi, y_re = "AccDecision_MI", "AccDecision_REST"
            ylab = "Accuracy (%) — Decision"
        else:
            y_mi, y_re = "AccTotal_MI", "AccTotal_REST"
            ylab = "Accuracy (%) — Total"

    elif metric == "bar_dynamics":
        y_mi, y_re = "BarDyn_MI", "BarDyn_REST"
        ylab = "Bar dynamics"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    dfp = runs_df[base_cols + [y_mi, y_re]].copy()

    mi = dfp[base_cols].copy()
    mi["Class"] = "MI"
    mi["Value"] = pd.to_numeric(dfp[y_mi], errors="coerce")

    re_ = dfp[base_cols].copy()
    re_["Class"] = "REST"
    re_["Value"] = pd.to_numeric(dfp[y_re], errors="coerce")

    out = pd.concat([mi, re_], ignore_index=True)
    out = out[np.isfinite(out["Value"].astype(float))]

    if ROLL_CLASSES:
        out["Class"] = "ALL"

    return out, "Value", ylab


def plot_bar_group_long(df_long: pd.DataFrame, xcol: str, ycol: str, ylab: str, title: str):
    fig, ax = plt.subplots(figsize=(10, 4))

    use_hue = df_long["Class"].nunique() > 1

    if use_hue:
        sns.barplot(
            data=df_long, x=xcol, y=ycol, hue="Class",
            ci=None, palette=CLASS_PALETTE, ax=ax
        )
    else:
        sns.barplot(
            data=df_long, x=xcol, y=ycol,
            ci=None, ax=ax
        )

    ax.set_ylabel(ylab)
    ax.set_xlabel(xcol)
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    if xcol == "SessionID":
        order = sorted(df_long["SessionID"].unique(), key=session_sort_key)
        ax.set_xticklabels(order, rotation=0)

    if xcol == "RunID":
        if SHOW_RUN_LABELS_ON_BAR:
            ax.tick_params(axis="x", rotation=90)
        else:
            run_order = df_long.drop_duplicates("RunID")["RunShort"].tolist()
            ax.set_xticklabels(run_order, rotation=90)

    if use_hue:
        ax.legend(title="Class")
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    plt.tight_layout()
    plt.show()


def plot_box_group_long(df_long: pd.DataFrame, xcol: str, ycol: str, ylab: str, title: str):
    fig, ax = plt.subplots(figsize=(10, 4))

    order = (
        sorted(df_long[xcol].unique(), key=session_sort_key)
        if xcol == "SessionID"
        else None
    )

    use_hue = df_long["Class"].nunique() > 1

    sns.boxplot(
        data=df_long,
        x=xcol,
        y=ycol,
        order=order,
        hue="Class" if use_hue else None,
        palette=CLASS_PALETTE if use_hue else None,
        showfliers=False,
        width=BOX_WIDTH,
        ax=ax
    )

    if SHOW_DOTS:
        sns.stripplot(
            data=df_long,
            x=xcol,
            y=ycol,
            order=order,
            hue="Class" if use_hue else None,
            palette=CLASS_PALETTE if use_hue else None,
            dodge=use_hue,
            jitter=0.18,
            alpha=0.6,
            size=4,
            ax=ax
        )

    ax.set_ylabel(ylab)
    ax.set_xlabel(xcol)
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Clean legend duplicates (boxplot + stripplot)
    if use_hue:
        handles, labels = ax.get_legend_handles_labels()
        # keep first N unique class labels (usually 2)
        seen = set()
        h2, l2 = [], []
        for h, l in zip(handles, labels):
            if l not in seen and l in ("MI", "REST", "ALL"):
                seen.add(l)
                h2.append(h)
                l2.append(l)
        ax.legend(h2, l2, title="Class")
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    if xcol in ("SessionID", "SubjectID"):
        ax.tick_params(axis="x", rotation=0)
    else:
        ax.tick_params(axis="x", rotation=90)

    plt.tight_layout()
    plt.show()


def dispatch_plot(runs_df: pd.DataFrame, metric: str):
    df_long, ycol, ylab = make_long_for_plot(runs_df, metric)

    if df_long.empty:
        print(f"⚠️ Nothing to plot for metric={metric} (all values missing).")
        return

    xcol = pick_xcol_for_group()

    # If we forced x=SubjectID but still want session ordering in some contexts, that's handled in plot functions.
    title_suffix = {
        "run": "Run-wise",
        "session": "Session-wise (runs as samples)",
        "subject": "Subject-wise (sessions/runs as samples)",
    }.get(GROUP_LEVEL, GROUP_LEVEL)

    # If SPLIT_BY_SUBJECT forced x=SubjectID, title should reflect that
    if SPLIT_BY_SUBJECT and GROUP_LEVEL != "subject":
        title_suffix += " — split by Subject"

    title = f"{ylab} — {title_suffix}"

    if PLOT_STYLE == "bar":
        plot_bar_group_long(df_long, xcol, ycol, ylab, title)
    else:
        plot_box_group_long(df_long, xcol, ycol, ylab, title)


# =========================================================
# Optional: quick console summary tables (split by Class)
# =========================================================

def summary_table(runs_df: pd.DataFrame, level: str, metric: str):
    df_long, ycol, ylab = make_long_for_plot(runs_df, metric)
    if df_long.empty:
        return None, ylab

    # If we're rolling classes, we can skip the "Class" dimension in the table if you want.
    include_class = df_long["Class"].nunique() > 1

    if level == "run":
        grp = ["SubjectID", "SessionID", "RunID"]
    elif level == "session":
        grp = ["SubjectID", "SessionID"]
    elif level == "subject":
        grp = ["SubjectID"]
    else:
        raise ValueError(level)

    if include_class:
        grp = grp + ["Class"]

    agg = df_long.groupby(grp)[ycol].agg(["count", "mean", "median"]).reset_index()
    iqr = df_long.groupby(grp)[ycol].apply(lambda x: safe_iqr(x.values)).reset_index(name="IQR")
    agg = agg.merge(iqr, on=grp, how="left")
    return agg, ylab


# =========================================================
# Main
# =========================================================

def main():
    # ---- Discover selection ----
    sub_dirs = list_subject_dirs(CURRENT_STUDY_ROOT)

    if SUBJECTS:
        wanted = set(SUBJECTS)
        selected_sub_dirs = [d for d in sub_dirs if subject_dir_to_id(d) in wanted]
    else:
        if SUBJECT_PREFIX_FILTER.strip() == "":
            selected_sub_dirs = sub_dirs
        else:
            selected_sub_dirs = [
                d for d in sub_dirs if subject_dir_to_id(d).startswith(SUBJECT_PREFIX_FILTER)
            ]

    if not selected_sub_dirs:
        raise RuntimeError("❌ No subject directories matched your SUBJECTS / SUBJECT_PREFIX_FILTER.")

    print("✅ Subjects included:")
    for d in selected_sub_dirs:
        print("  -", d)

    # ---- Main aggregation loop ----
    rows = []

    for sub_dir in selected_sub_dirs:
        subj_id = subject_dir_to_id(sub_dir)
        subj_path = os.path.join(CURRENT_STUDY_ROOT, sub_dir)

        all_sessions = sorted([
            s for s in os.listdir(subj_path)
            if os.path.isdir(os.path.join(subj_path, s)) and s.startswith("ses-")
        ], key=session_sort_key)

        all_sessions = [s for s in all_sessions if is_valid_session(s)]

        if SESSIONS_BY_SUBJECT is None:
            sessions = all_sessions
        else:
            sessions = [
                s for s in SESSIONS_BY_SUBJECT.get(subj_id, [])
                if (s in all_sessions) and is_valid_session(s)
            ]
            sessions = sorted(sessions, key=session_sort_key)

        if not sessions:
            print(f"⚠️ No sessions selected/found for {sub_dir}; skipping.")
            continue

        for sess in sessions:
            sess_path = os.path.join(subj_path, sess)
            logs_path = os.path.join(sess_path, "logs")
            if not os.path.isdir(logs_path):
                print(f"⚠️ logs/ missing: {logs_path}; skipping session.")
                continue

            all_runs = sorted([
                r for r in os.listdir(logs_path)
                if os.path.isdir(os.path.join(logs_path, r)) and r.startswith("ONLINE_")
            ])

            if RUNS_BY_SUBJECT_SESSION is None:
                runs = all_runs
            else:
                runs = [r for r in RUNS_BY_SUBJECT_SESSION.get((subj_id, sess), []) if r in all_runs]

            if not runs:
                print(f"⚠️ No runs selected/found for {subj_id} {sess}; skipping.")
                continue

            for run_folder in runs:
                run_path = os.path.join(logs_path, run_folder)
                run_id = f"{subj_id}__{sess}__{run_folder}"

                csv_path = find_decoder_csv(run_path)
                if csv_path is None:
                    print(f"⚠️ No decoder_output*.csv in {run_path}; skipping run.")
                    continue

                df_run = pd.read_csv(csv_path)

                if "Phase" in df_run.columns:
                    df_run = df_run[df_run["Phase"] != "ROBOT"].copy()
                else:
                    print(f"⚠️ Phase column missing in {csv_path}; continuing without ROBOT exclusion.")

                # add IDs
                df_run["SubjectID"] = subj_id
                df_run["SessionID"] = sess
                df_run["RunFolder"] = run_folder
                df_run["RunID"] = run_id

                if "Trial" not in df_run.columns:
                    raise ValueError(f"Missing Trial column in {csv_path}")

                df_run["GlobalTrialID"] = df_run["RunID"].astype(str) + "_" + df_run["Trial"].astype(str)

                # unify prob cols for bar dynamics
                df_run, MI_COL, REST_COL = build_unified_prob_cols(df_run)

                # ---------- Bar dynamics (Lean16Hz internal) ----------
                try:
                    trial_lean = compute_lean16hz_per_trial(df_run, MI_COL, REST_COL, thresh=THRESH)
                except Exception as e:
                    print(f"⚠️ Bar dynamics compute failed for {run_id}: {e}")
                    trial_lean = pd.DataFrame(columns=["GlobalTrialID", "Class", "LeanPct"])

                lean_mi = (
                    trial_lean.loc[trial_lean["Class"] == "MI", "LeanPct"].astype(float).values
                    if not trial_lean.empty else np.array([])
                )
                lean_re = (
                    trial_lean.loc[trial_lean["Class"] == "REST", "LeanPct"].astype(float).values
                    if not trial_lean.empty else np.array([])
                )

                BarDyn_MI = float(np.nanmean(lean_mi)) if lean_mi.size else np.nan
                BarDyn_REST = float(np.nanmean(lean_re)) if lean_re.size else np.nan

                # ---------- Accuracy (per class from CSV) ----------
                acc_by_class = compute_accuracy_by_class_from_csv_trials(df_run)

                AccDecision_MI = acc_by_class.get("AccDecision_MI", np.nan)
                AccDecision_REST = acc_by_class.get("AccDecision_REST", np.nan)
                AccTotal_MI = acc_by_class.get("AccTotal_MI", np.nan)
                AccTotal_REST = acc_by_class.get("AccTotal_REST", np.nan)

                Ambiguous_MI = acc_by_class.get("Ambiguous_MI", 0)
                Ambiguous_REST = acc_by_class.get("Ambiguous_REST", 0)
                Trials_MI = acc_by_class.get("Trials_MI", 0)
                Trials_REST = acc_by_class.get("Trials_REST", 0)
                Decisions_MI = acc_by_class.get("Decisions_MI", 0)
                Decisions_REST = acc_by_class.get("Decisions_REST", 0)

                # ---------- Optional: overall event_log accuracy (sanity check) ----------
                if ACCURACY_SOURCE == "event_log":
                    AccDecision_overall, AccTotal_overall, Ambiguous_overall = compute_overall_accuracy_from_event_log(run_path)
                else:
                    AccDecision_overall, AccTotal_overall, Ambiguous_overall = np.nan, np.nan, 0

                rows.append({
                    "SubjectID": subj_id,
                    "SessionID": sess,
                    "RunFolder": run_folder,
                    "RunShort": shorten_run_name(run_folder),
                    "RunID": run_id,
                    "DecoderCSV": csv_path,

                    # --- accuracy per class ---
                    "AccDecision_MI": AccDecision_MI,
                    "AccDecision_REST": AccDecision_REST,
                    "AccTotal_MI": AccTotal_MI,
                    "AccTotal_REST": AccTotal_REST,

                    "Ambiguous_MI": Ambiguous_MI,
                    "Ambiguous_REST": Ambiguous_REST,
                    "Trials_MI": Trials_MI,
                    "Trials_REST": Trials_REST,
                    "Decisions_MI": Decisions_MI,
                    "Decisions_REST": Decisions_REST,

                    # --- optional overall from event_log (not plotted by default) ---
                    "AccDecision_overall_eventlog": AccDecision_overall,
                    "AccTotal_overall_eventlog": AccTotal_overall,
                    "Ambiguous_overall_eventlog": Ambiguous_overall,

                    # --- bar dynamics per class ---
                    "BarDyn_MI": BarDyn_MI,
                    "BarDyn_REST": BarDyn_REST,

                    # counts for context
                    "TrialsN_total": int(trial_lean["GlobalTrialID"].nunique()) if not trial_lean.empty else 0,
                })

    runs_df = pd.DataFrame(rows)
    if runs_df.empty:
        raise RuntimeError("❌ No runs were successfully loaded. Check SUBJECTS/SESSIONS and directory structure.")

    print("\n✅ Aggregation complete.")
    print(
        runs_df[
            ["SubjectID", "SessionID", "RunShort",
             "AccDecision_MI", "AccDecision_REST",
             "AccTotal_MI", "AccTotal_REST",
             "BarDyn_MI", "BarDyn_REST",
             "TrialsN_total"]
        ].head(20)
    )

    if SAVE_SUMMARY_CSV:
        out_csv = os.path.join(CURRENT_STUDY_ROOT, SUMMARY_CSV_NAME)
        runs_df.to_csv(out_csv, index=False)
        print(f"\n✅ Saved summary CSV: {out_csv}")

    # ---- Plots ----
    if DO_PLOTS:
        for m in METRICS_TO_PLOT:
            dispatch_plot(runs_df, m)

    # ---- Summary tables ----
    print("\n--- Session-wise summary (mean/median/IQR across runs) ---")
    tab, _ = summary_table(runs_df, "session", "accuracy")
    if tab is not None:
        print(tab.to_string(index=False))

    tab, _ = summary_table(runs_df, "session", "bar_dynamics")
    if tab is not None:
        print("\n", tab.to_string(index=False))


if __name__ == "__main__":
    main()
