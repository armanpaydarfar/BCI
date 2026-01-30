import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import numpy as np
from matplotlib.lines import Line2D

# ---- Configurable Subject ----
subject = "CLIN_SUBJ_004"


# ==========================================
# Metric configuration (add these modes)
# ==========================================
#   "SOFT_LEAN"   : soft % above THRESH with power α (geometric-style weighting)
#   "MARGIN_AUC"  : time-weighted area of (p-THRESH)+ normalized to [0,1]
#   "LOGIT_SOFT"  : temperature-scaled logits, then soft-% above 0.5

# Example: try one of these
# METRIC_MODE = "SOFT_LEAN"
# METRIC_MODE = "MARGIN_AUC"
# METRIC_MODE = "LOGIT_SOFT"

SOFT_POWER   = 2.0    # α > 1 emphasizes strong modulation; α=1 reduces to linear
LOGIT_TEMP   = 0.6    # <1 sharpens, >1 flattens
EPS_CLIP     = 1e-6   # for stable logit transforms


# ==========================================
# Metric configuration (new)
# ==========================================
#   "LEAN_16HZ"        : original sample-weighted Lean% at classifier rate (raw metric)
#   "LEAN_TIMEWEIGHTED": time-weighted Lean% at 60 Hz with optional leaky integrator
#   "FINAL_WINDOW"     : Lean% computed only over the last LAST_N_SEC of each trial
#   "DERIV"            : % of (time) where d/dt(correct-class prob) > DERIV_EPS
METRIC_MODE      = "LEAN_16HZ"   # change to "FINAL_WINDOW" or "DERIV" to try new metrics
THRESH           = 0.50

# For LEAN_TIMEWEIGHTED / DERIV
CLASSIFIER_HZ    = 16.0
DISPLAY_HZ       = 60.0
USE_LEAKY        = False
TAU_SEC          = 0.96          # leaky time constant (seconds)

# For FINAL_WINDOW / DERIV
LAST_N_SEC       = 2.0           # use final N seconds (or whole trial if shorter)
APPLY_TAIL_ONLY  = True

# For DERIV
DERIV_EPS        = 0.002         # prob/sec threshold to count derivative as "positive"
DERIV_SMOOTH_WIN = 5             # moving-average window (samples) for smoothing before derivative


# ==========================================
# Plot configuration (new)
# ==========================================

# ---- Visualization selector ----
plot_style = "dot_ci"  # options: "box", "dot_std", "dot_ci", "violin", "ecdf", "forest_runwise"

DOT_CI_SHOW_DOTS      = False   # applies to both dot_ci AND dot_std
DOT_STD_ASYMMETRIC_SE = True    # if True, use different SE above vs below the mean
DOT_STD_USE_SE        = False    # if False, use SD instead of SE for the vertical line(s)

# How to aggregate for bar dynamics:
#   "total"   : all runs & sessions pooled
#   "session" : grouped by SessionID (each session is a group)
#   "run"     : grouped by RunID      (each run is a group; current behavior)
GROUP_MODE = "session"  # or "session" or "run"


# ==========================================
# Session + Run selection (now supports multi-session)
# ==========================================

# ---- Prompt User to Select Session Subdirectory ----
base_dir = os.path.expanduser(f"~/Documents/CurrentStudy/sub-{subject}")
session_root = os.path.join(base_dir)
if not os.path.exists(session_root):
    raise FileNotFoundError(f"❌ Subject directory not found: {session_root}")

session_dirs = [
    d for d in os.listdir(session_root)
    if os.path.isdir(os.path.join(session_root, d)) and d.startswith("ses-")
]
if not session_dirs:
    raise FileNotFoundError(f"❌ No session directories found in: {session_root}")

print("Available sessions:")
for idx, s in enumerate(session_dirs):
    print(f" [{idx}] {s}")

session_sel = input(
    "➡️  Select session index(es) (comma-separated) or press ENTER to use ALL sessions: "
).strip()

if session_sel:
    try:
        session_indices = [int(tok) for tok in session_sel.split(",")]
        selected_sessions = [session_dirs[i] for i in session_indices]
    except Exception as e:
        raise ValueError(f"❌ Invalid session selection: {e}")
else:
    selected_sessions = session_dirs

print(f"✅ Using sessions: {selected_sessions}")

# ==========================================
# Run selection logic
#   • If ONE session selected: per-run selection UI.
#   • If MULTIPLE sessions: take ALL ONLINE_ runs from each session.
# ==========================================

all_runs = []  # list of (session, run_folder)

if len(selected_sessions) == 1:
    # ---- Single-session behavior (preserves your existing workflow) ----
    session = selected_sessions[0]
    log_dir = os.path.join(session_root, session, "logs")

    run_dirs = [
        d for d in os.listdir(log_dir)
        if os.path.isdir(os.path.join(log_dir, d)) and d.startswith("ONLINE_")
    ]
    if not run_dirs:
        raise FileNotFoundError(f"❌ No ONLINE_ run folders found in: {log_dir}")

    print(f"Available run directories in {session}:")
    for idx, r in enumerate(run_dirs):
        print(f" [{idx}] {r}")

    selected_run = input(
        "➡️  Select run index(es) (comma-separated) or press ENTER to merge all: "
    ).strip()

    if selected_run:
        selected_indices = [int(i) for i in selected_run.split(",")]
        selected_run_dirs = [run_dirs[i] for i in selected_indices]
    else:
        selected_run_dirs = run_dirs

    for rd in selected_run_dirs:
        all_runs.append((session, rd))

else:
    # ---- Multi-session behavior: take ALL ONLINE_ runs in each chosen session ----
    for session in selected_sessions:
        log_dir = os.path.join(session_root, session, "logs")
        if not os.path.exists(log_dir):
            print(f"⚠️ logs/ not found for session {session} (skipping).")
            continue

        run_dirs = [
            d for d in os.listdir(log_dir)
            if os.path.isdir(os.path.join(log_dir, d)) and d.startswith("ONLINE_")
        ]
        if not run_dirs:
            print(f"⚠️ No ONLINE_ run folders found in: {log_dir} (session {session})")
            continue

        print(f"✅ Session {session}: found runs {run_dirs}")
        for rd in run_dirs:
            all_runs.append((session, rd))

if not all_runs:
    raise RuntimeError("❌ No runs found across selected sessions; nothing to analyze.")


# ==========================================
# Load and Combine Data (now across sessions+runs)
# ==========================================
df_list = []
conf_matrices = []
total_ambiguous = 0

for global_run_idx, (session, run_folder) in enumerate(all_runs):
    log_dir = os.path.join(session_root, session, "logs")
    run_path = os.path.join(log_dir, run_folder)

    csv_files = [
        f for f in os.listdir(run_path)
        if f.startswith("decoder_output") and f.endswith(".csv")
    ]
    if not csv_files:
        print(f"⚠️ No decoder output CSV in: {run_path}")
        continue

    csv_path = os.path.join(run_path, csv_files[0])
    print(f"✅ Loaded decoder output from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Exclude ROBOT phase
    df = df[df["Phase"] != "ROBOT"].copy()

    # Make SessionID + RunID explicit
    df["SessionID"] = session                   # e.g., "ses-01"
    df["RunID"]     = f"{session}_{run_folder}" # e.g., "ses-01_ONLINE_..."

    df_list.append(df)

    # Parse confusion matrix from event_log.txt
    log_path = os.path.join(run_path, "event_log.txt")
    if os.path.exists(log_path):
        with open(log_path, "r") as log_file:
            log_text = log_file.read()

        match = re.search(
            r"Actual 200.*?MI\): (\d+).*?REST\): (\d+).*?"
            r"Actual 100.*?MI\): (\d+).*?REST\): (\d+)",
            log_text,
            re.DOTALL
        )
        ambiguous_match = re.search(r"Ambiguous trials.*?: (\d+)", log_text)

        if match:
            a200_p200 = int(match.group(1))
            a200_p100 = int(match.group(2))
            a100_p200 = int(match.group(3))
            a100_p100 = int(match.group(4))
            conf_matrix = [[a200_p200, a200_p100],
                           [a100_p200, a100_p100]]
            conf_matrices.append(conf_matrix)
            print(f"✅ Parsed confusion matrix from: {log_path}")

        if ambiguous_match:
            total_ambiguous += int(ambiguous_match.group(1))
    else:
        print(f"⚠️ event_log.txt not found in: {run_path}")

# ---- Combine Decoder Data ----
df = pd.concat(df_list, ignore_index=True)

# ---- Flexible column detection (old vs new schema, per-row fallback) ----
has_old = ("P(MI)" in df.columns) and ("P(REST)" in df.columns)
has_new = ("P(MI)_inst" in df.columns) and ("P(REST)_inst" in df.columns)

if not (has_old or has_new):
    raise ValueError("Missing probability columns: need either old (P(MI), P(REST)) "
                     "or new (P(MI)_inst, P(REST)_inst) schema.")

# Build unified probability columns that work across mixed sessions
if has_old and has_new:
    # Prefer _inst when present; fall back to old columns otherwise
    df["MI_prob"]   = df.get("P(MI)_inst")  .combine_first(df.get("P(MI)"))
    df["REST_prob"] = df.get("P(REST)_inst").combine_first(df.get("P(REST)"))
elif has_new:
    df["MI_prob"]   = df["P(MI)_inst"]
    df["REST_prob"] = df["P(REST)_inst"]
else:  # has_old only
    df["MI_prob"]   = df["P(MI)"]
    df["REST_prob"] = df["P(REST)"]

MI_COL   = "MI_prob"
REST_COL = "REST_prob"

# ---- Validate necessary columns (non-prob cols) ----
required_always = ["Trial", "Timestamp", "True Label", "Predicted Label", "Phase", "RunID"]
for col in required_always:
    if col not in df.columns:
        raise ValueError(f"Missing required column in CSV: {col}")


# ---- Ensure unique identification per trial across runs ----
df["GlobalTrialID"] = df["RunID"] + "_" + df["Trial"].astype(str)

# Exclude ROBOT phase for these visualizations
df_vis = df[df["Phase"] != "ROBOT"].copy()


# ==========================================
# 1) Posterior Probability per Trial — ALL on ONE figure
#    (Correct-class instantaneous probability)
# ==========================================
plt.figure()
labeled = {"MI": False, "REST": False}  # to avoid legend spam

for gtid in df_vis["GlobalTrialID"].unique():
    trial_data = df_vis[df_vis["GlobalTrialID"] == gtid]
    if trial_data.empty:
        continue

    true_label = int(trial_data["True Label"].iloc[0])
    if true_label == 200:   # MI trial
        col_prob = MI_COL
        color    = "tab:orange"   # MI = orange
        lab      = "MI" if not labeled["MI"] else None
        labeled["MI"] = True
    elif true_label == 100: # REST trial
        col_prob = REST_COL
        color    = "tab:blue"     # REST = blue
        lab      = "REST" if not labeled["REST"] else None
        labeled["REST"] = True
    else:
        continue

    y = trial_data[col_prob].astype(float).values
    x = range(len(y))
    plt.plot(x, y, color=color, alpha=0.5, linewidth=1.2, label=lab)

plt.xlabel("Time (relative index)")
plt.ylabel("Posterior Probability (Correct Class)")
plt.title("Posterior Probability per Trial (Excl. ROBOT Phase) — All Trials")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()


# ==========================================
# Metric utilities
# ==========================================

def time_weighted_mean(y, t):
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    if y.size == 0:
        return np.nan
    if y.size == 1:
        return float(y[0])
    w = np.diff(t)
    return float(np.sum(y[:-1] * w) / np.sum(w))


# -------------------------
# 50-centered replacements
# -------------------------

def soft_lean(y, thresh=0.5, power=2.0):
    """
    Fair, 50-centered SOFT:
      score = 50 + 50 * mean( sign(m) * |m|^power )
    where m = (p - thresh)/(1 - thresh) ∈ [-1,1].
    power>1 emphasizes strong confidence both above and below chance.
    """
    p = np.asarray(y, float)
    m = (p - thresh) / (1.0 - thresh)
    s = np.sign(m) * np.power(np.abs(m), power)
    return float(np.clip(50.0 + 50.0 * np.nanmean(s), 0.0, 100.0))


def soft_lean_timeweighted(y, t, thresh=0.5, power=2.0):
    """
    Time-weighted fair SOFT (same as soft_lean but uses time_weighted_mean).
    Keep this if your timestamps can be non-uniform; otherwise unused.
    """
    p = np.asarray(y, float)
    m = (p - thresh) / (1.0 - thresh)
    s = np.sign(m) * np.power(np.abs(m), power)
    return float(np.clip(50.0 + 50.0 * time_weighted_mean(s, t), 0.0, 100.0))


def margin_auc(y, t, thresh=0.5):
    """
    Fair, 50-centered AUC-like (linear):
      score = 50 + 50 * mean( m )
    where m = (p - thresh)/(1 - thresh).
    Uses time_weighted_mean over t.
    """
    p = np.asarray(y, float)
    m = (p - thresh) / (1.0 - thresh)
    return float(np.clip(50.0 + 50.0 * time_weighted_mean(m, t), 0.0, 100.0))


def logit_soft_lean(y, temp=1.0, eps=1e-6):
    """
    Temperature-scale logits → fair SOFT (power=2):
      1) l = logit(p)/temp, s = sigmoid(l)
      2) m = (s - 0.5)/0.5
      3) score = 50 + 50 * mean( sign(m) * |m|^2 )
    Keeps signature; emphasizes decisive highs/lows after temp scaling.
    """
    p = np.clip(np.asarray(y, float), eps, 1.0 - eps)
    l = np.log(p / (1.0 - p)) / max(eps, float(temp))
    s = 1.0 / (1.0 + np.exp(-l))
    m = (s - 0.5) / 0.5
    z = np.sign(m) * np.power(np.abs(m), 2.0)
    return float(np.clip(50.0 + 50.0 * np.nanmean(z), 0.0, 100.0))


# ==========================================
# 2) Per-trial metrics (configurable) — builds lean_df
# ==========================================

def asymmetric_se(vals, use_se=True):
    """Return (m, down_err, up_err) around the mean using asymmetric SD/SE.
       down_err is for mean - err, up_err is for mean + err."""
    a = np.asarray(vals, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan, np.nan, np.nan
    m = float(np.nanmean(a))

    # Split around the mean (include mean in both sides so we don't get empty sets)
    lower = a[a <= m]
    upper = a[a >= m]

    # Deviations
    dev_dn = m - lower   # nonnegative
    dev_up = upper - m   # nonnegative

    # SDs of deviations; guard for size 1
    sd_dn = float(np.nanstd(dev_dn, ddof=1)) if dev_dn.size > 1 else 0.0
    sd_up = float(np.nanstd(dev_up, ddof=1)) if dev_up.size > 1 else 0.0
    if use_se:
        n_dn = max(1, lower.size)
        n_up = max(1, upper.size)
        err_dn = sd_dn / np.sqrt(n_dn)
        err_up = sd_up / np.sqrt(n_up)
    else:
        err_dn = sd_dn
        err_up = sd_up

    return m, err_dn, err_up


def leaky_step(y, x, dt, tau):
    if not np.isfinite(tau) or tau <= 0:
        return x
    k = np.exp(-dt / tau)
    return y * k + x * (1.0 - k)


def build_correct_class_series(trial_df):
    """Return (t, p) for the correct-class probability at classifier rate."""
    if int(trial_df["True Label"].iloc[0]) == 200:
        p = trial_df[MI_COL].astype(float).values
    else:
        p = trial_df[REST_COL].astype(float).values
    # Use provided timestamps if numeric; otherwise assume uniform 16 Hz
    t_col = trial_df["Timestamp"].values
    if np.issubdtype(trial_df["Timestamp"].dtype, np.number):
        t = t_col - t_col[0]
    else:
        t = np.arange(len(p)) / CLASSIFIER_HZ
    n = min(len(t), len(p))
    t, p = t[:n], p[:n]
    mask = np.isfinite(p)
    return t[mask], p[mask]


def series_to_ui_60hz(t_cls, p_cls, use_leaky=True, y0=None):
    """Zero-order hold from classifier ticks to 60 Hz; optional leaky integration."""
    if len(p_cls) == 0:
        return np.array([]), np.array([])
    t_end = float(t_cls[-1])
    t_ui = np.arange(0.0, t_end + 1e-9, 1.0 / DISPLAY_HZ)
    idx = np.searchsorted(t_cls, t_ui, side="right") - 1
    idx[idx < 0] = 0
    x_ui = p_cls[idx]
    if not use_leaky:
        return t_ui, x_ui
    y = np.empty_like(x_ui)
    y_prev = x_ui[0] if y0 is None else float(y0)
    y[0] = y_prev
    for i in range(1, len(x_ui)):
        y_prev = leaky_step(y_prev, x_ui[i], 1.0 / DISPLAY_HZ, TAU_SEC)
        y[i] = y_prev
    return t_ui, y


def smooth_movavg(a, win):
    a = np.asarray(a, float)
    if win is None or win <= 1 or a.size == 0:
        return a
    k = np.ones(win, float) / win
    return np.convolve(a, k, mode="same")


def tail_slice(t, y, last_sec):
    """Return the last 'last_sec' seconds (or the whole series if shorter) if APPLY_TAIL_ONLY."""
    if y.size == 0 or not APPLY_TAIL_ONLY or last_sec is None or last_sec <= 0:
        return t, y
    t_end = float(t[-1])
    t_start = max(0.0, t_end - last_sec)
    i0 = np.searchsorted(t, t_start, side="left")
    return t[i0:], y[i0:]


# ---- Build metric records
lean_records = []  # rows: {GlobalTrialID, RunID, SessionID, Class, LeanPct, Metric}

for gtid, trial_df in df_vis.groupby("GlobalTrialID"):
    if trial_df.empty:
        continue

    true_label = int(trial_df["True Label"].iloc[0])
    run_id     = str(trial_df["RunID"].iloc[0])
    session_id = trial_df["SessionID"].iloc[0] if "SessionID" in trial_df.columns else None

    cls = "MI" if true_label == 200 else ("REST" if true_label == 100 else None)
    if cls is None:
        continue

    # Base series at classifier rate
    t_cls, p_cls = build_correct_class_series(trial_df)
    if p_cls.size == 0:
        continue

    # Choose series for metric
    if METRIC_MODE == "LEAN_16HZ":
        t_use, y_use = t_cls, p_cls

    elif METRIC_MODE == "LEAN_TIMEWEIGHTED":
        t_use, y_use = series_to_ui_60hz(t_cls, p_cls, use_leaky=USE_LEAKY)

    elif METRIC_MODE == "FINAL_WINDOW":
        t_use, y_use = tail_slice(t_cls, p_cls, LAST_N_SEC)

    elif METRIC_MODE == "DERIV":
        # Use UI series for smoother derivative by default
        t_ui, y_ui = series_to_ui_60hz(t_cls, p_cls, use_leaky=USE_LEAKY)
        y_ui = smooth_movavg(y_ui, DERIV_SMOOTH_WIN)
        t_use, y_use = tail_slice(t_ui, y_ui, LAST_N_SEC)

    else:
        t_use, y_use = t_cls, p_cls

    if y_use.size == 0:
        continue

    # ---- Compute metric value (soft alternatives included) ----
    if METRIC_MODE in ("LEAN_16HZ", "LEAN_TIMEWEIGHTED", "FINAL_WINDOW"):
        # original hard threshold %
        lean_pct = float((y_use > THRESH).mean() * 100.0)

    elif METRIC_MODE == "SOFT_LEAN":
        # sample-wise soft weighting; if your t_use is nonuniform, consider soft_lean_timeweighted
        lean_pct = soft_lean(y_use, thresh=THRESH, power=SOFT_POWER)

    elif METRIC_MODE == "MARGIN_AUC":
        # time-weighted area above threshold (normalized); robust to irregular timestamps
        lean_pct = margin_auc(y_use, t_use, thresh=THRESH)

    elif METRIC_MODE == "LOGIT_SOFT":
        # sharpen/relax with LOGIT_TEMP, then soft % above 0.5
        lean_pct = logit_soft_lean(y_use, temp=LOGIT_TEMP, eps=EPS_CLIP)

    elif METRIC_MODE == "DERIV":
        if t_use.size > 1:
            dy_dt = np.gradient(y_use, t_use)
            good = dy_dt > DERIV_EPS
            lean_pct = float(good.mean() * 100.0)
        else:
            lean_pct = 0.0

    else:
        # fallback = original
        lean_pct = float((y_use > THRESH).mean() * 100.0)

    lean_records.append({
        "GlobalTrialID": gtid,
        "RunID": run_id,
        "SessionID": session_id,
        "Class": cls,
        "LeanPct": lean_pct,
        "Metric": METRIC_MODE
    })

lean_df = pd.DataFrame(lean_records)


COL_MI, COL_REST = "tab:orange", "tab:blue"
RUN_COLORS = plt.rcParams['axes.prop_cycle'].by_key().get(
    'color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
)

# Split TOTAL by class for convenience
mi_vals_total   = lean_df.loc[lean_df["Class"] == "MI",   "LeanPct"].astype(float).values
rest_vals_total = lean_df.loc[lean_df["Class"] == "REST", "LeanPct"].astype(float).values


def jittered_x(center, n, scale=0.06):
    if n <= 1:
        return np.array([center])
    j = np.linspace(-1, 1, n) * scale
    rng = np.random.default_rng(42)
    rng.shuffle(j)
    return center + j


# Helper: bootstrap CI for the mean (absolute bounds)
def mean_ci(a, n_boot=1000, alpha=0.05, rng_seed=42):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(rng_seed)
    boots = rng.choice(a, size=(n_boot, a.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return float(np.mean(a)), (float(lo), float(hi))


# ---- Run / Session ordering + helpers ----
runs_sorted = sorted(lean_df["RunID"].dropna().unique().tolist())
if "SessionID" in lean_df.columns:
    sessions_sorted = sorted(lean_df["SessionID"].dropna().unique().tolist())
else:
    sessions_sorted = []


def get_runwise_vals():
    """Return dicts: run->vals for MI and REST."""
    mi_by_run, rest_by_run = {}, {}
    for run in runs_sorted:
        sub = lean_df[lean_df["RunID"] == run]
        mi_by_run[run]   = sub.loc[sub["Class"]=="MI",   "LeanPct"].astype(float).values
        rest_by_run[run] = sub.loc[sub["Class"]=="REST", "LeanPct"].astype(float).values
    return mi_by_run, rest_by_run


def get_sessionwise_vals():
    """Return dicts: session->vals for MI and REST."""
    mi_by_sess, rest_by_sess = {}, {}
    for sess in sessions_sorted:
        sub = lean_df[lean_df["SessionID"] == sess]
        mi_by_sess[sess]   = sub.loc[sub["Class"]=="MI",   "LeanPct"].astype(float).values
        rest_by_sess[sess] = sub.loc[sub["Class"]=="REST", "LeanPct"].astype(float).values
    return mi_by_sess, rest_by_sess


def offsets(k, base=0.18):
    """Symmetric offsets for k groups within each class position."""
    if k == 1:
        return np.array([0.0])
    return np.linspace(-base, base, k)


# ---------- TOTAL plotting helpers ----------
def plot_box_total(mi_vals, rest_vals, title_suffix=""):
    fig, ax = plt.subplots()
    box = ax.boxplot(
        [mi_vals, rest_vals],
        labels=["MI", "REST"],
        widths=0.5,
        patch_artist=True,
        showfliers=False
    )
    for patch, c in zip(box["boxes"], [COL_MI, COL_REST]):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)
    for median in box["medians"]:
        median.set_linewidth(2)
    ax.scatter(
        jittered_x(1.0, len(mi_vals)), mi_vals, s=30, alpha=0.9,
        edgecolor="black", linewidth=0.5, color=COL_MI
    )
    ax.scatter(
        jittered_x(2.0, len(rest_vals)), rest_vals, s=30, alpha=0.9,
        edgecolor="black", linewidth=0.5, color=COL_REST
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Lean %")
    ax.set_title(f"Bar Dynamics — {METRIC_MODE} {title_suffix}")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    plt.show()


def plot_dot_std_total(mi_vals, rest_vals, title_suffix=""):
    fig, ax = plt.subplots()

    # choose symmetric vs asymmetric errors
    def err_pair(arr):
        if DOT_STD_ASYMMETRIC_SE:
            m, dn, up = asymmetric_se(arr, use_se=DOT_STD_USE_SE)
            return m, dn, up
        else:
            m = float(np.nanmean(arr))
            s = float(np.nanstd(arr, ddof=1))
            if DOT_STD_USE_SE:
                n = max(1, np.isfinite(arr).sum())
                s = s / np.sqrt(n)
            return m, s, s

    (m_mi, dn_mi, up_mi) = err_pair(mi_vals)
    (m_re, dn_re, up_re) = err_pair(rest_vals)

    # vertical lines + mean dots
    for x, m, dn, up, col in [
        (0, m_mi, dn_mi, up_mi, COL_MI),
        (1, m_re, dn_re, up_re, COL_REST)
    ]:
        if np.isfinite(m):
            ax.vlines(x, m - dn, m + up, color=col, linewidth=3, alpha=0.95)
            ax.plot(x, m, "o", ms=9, color=col)

    # (optional) raw dots, governed by DOT_CI_SHOW_DOTS too
    if DOT_CI_SHOW_DOTS:
        ax.scatter(
            jittered_x(0, len(mi_vals)), mi_vals, s=24, alpha=0.65,
            edgecolor="k", linewidth=0.3, color=COL_MI
        )
        ax.scatter(
            jittered_x(1, len(rest_vals)), rest_vals, s=24, alpha=0.65,
            edgecolor="k", linewidth=0.3, color=COL_REST
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["MI", "REST"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Lean %")
    err_label = "SE" if DOT_STD_USE_SE else "SD"
    asym = "asym" if DOT_STD_ASYMMETRIC_SE else "sym"
    ax.set_title(
        f"Bar Dynamics — Mean • ±{err_label} ({asym}) ({METRIC_MODE}) {title_suffix}"
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_dot_ci_total(mi_vals, rest_vals, title_suffix=""):
    fig, ax = plt.subplots()
    mi_mean,  (mi_lo,  mi_hi)  = mean_ci(mi_vals,  n_boot=1000, alpha=0.05)
    rest_mean, (re_lo, re_hi)  = mean_ci(rest_vals, n_boot=1000, alpha=0.05)

    if DOT_CI_SHOW_DOTS:
        ax.scatter(
            jittered_x(0, len(mi_vals)), mi_vals, s=24, alpha=0.5,
            edgecolor="k", linewidth=0.3, color=COL_MI
        )
        ax.scatter(
            jittered_x(1, len(rest_vals)), rest_vals, s=24, alpha=0.5,
            edgecolor="k", linewidth=0.3, color=COL_REST
        )
    ax.plot([0], [mi_mean], marker="o", ms=9, color=COL_MI)
    ax.plot([1], [rest_mean], marker="o", ms=9, color=COL_REST)
    ax.vlines(0, mi_lo, mi_hi, linewidth=3, color=COL_MI)
    ax.vlines(1, re_lo, re_hi, linewidth=3, color=COL_REST)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["MI", "REST"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Lean %")
    ax.set_title(f"Bar Dynamics — Mean with 95% CI")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_violin_total(lean_df_subset, title_suffix=""):
    fig, ax = plt.subplots()
    sns.violinplot(
        data=lean_df_subset,
        x="Class",
        y="LeanPct",
        inner=None,
        cut=0,
        ax=ax,
        palette={"MI": COL_MI, "REST": COL_REST}
    )
    # mean + CI segment
    for i, cls in enumerate(["MI", "REST"]):
        vals = lean_df_subset.loc[
            lean_df_subset["Class"] == cls, "LeanPct"
        ].astype(float).values
        m, (lo, hi) = mean_ci(vals)
        ax.vlines(i, lo, hi, color="k", linewidth=3, alpha=0.9)
        ax.plot(
            i, m, "o", color="white", markersize=10,
            markeredgecolor="k", markeredgewidth=1.5
        )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Lean %")
    ax.set_title(f"Bar Dynamics — Violin + Mean + 95% CI")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.show()


def plot_ecdf_total(mi_vals, rest_vals, title_suffix=""):
    def ecdf(a):
        a = np.sort(a[np.isfinite(a)])
        y = np.linspace(0, 1, len(a), endpoint=True)
        return a, y

    fig, ax = plt.subplots()
    x_mi, y_mi = ecdf(mi_vals)
    x_re, y_re = ecdf(rest_vals)
    ax.step(x_mi, y_mi, where="post", label="MI",   color=COL_MI)
    ax.step(x_re, y_re, where="post", label="REST", color=COL_REST)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Lean %")
    ax.set_ylabel("ECDF")
    ax.set_title(f"Lean% — ECDF ({METRIC_MODE}) {title_suffix}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ---------- RUNWISE (single figure) helpers ----------
def plot_box_runwise(lean_df_subset):
    fig, ax = plt.subplots()
    if lean_df_subset.empty:
        print("⚠️ Nothing to plot.")
        return
    sns.boxplot(
        data=lean_df_subset,
        x="Class",
        y="LeanPct",
        hue="RunID",
        dodge=True,
        showfliers=False,
        ax=ax
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Lean %")
    ax.set_title(f"Bar Dynamics")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Run")
    plt.tight_layout()
    plt.show()


def plot_dot_std_runwise(mi_by_run, rest_by_run):
    fig, ax = plt.subplots()
    k = len(runs_sorted)
    off = offsets(k, base=0.18)

    # legend proxies (one per run color)
    handles, labels = [], []
    seen = set()

    for j, run in enumerate(runs_sorted):
        col = RUN_COLORS[j % len(RUN_COLORS)]
        if run not in seen:
            handles.append(
                Line2D([0], [0], color=col, marker="o", linestyle="-", linewidth=3)
            )
            labels.append(run)
            seen.add(run)

        for x0, arr in [
            (0, mi_by_run.get(run, np.array([]))),
            (1, rest_by_run.get(run, np.array([])))
        ]:
            if arr.size == 0:
                continue
            if DOT_STD_ASYMMETRIC_SE:
                m, dn, up = asymmetric_se(arr, use_se=DOT_STD_USE_SE)
            else:
                m = float(np.nanmean(arr))
                s = float(np.nanstd(arr, ddof=1))
                if DOT_STD_USE_SE:
                    n = max(1, np.isfinite(arr).size)
                    s = s / np.sqrt(n)
                dn = up = s

            x = x0 + off[j]
            if np.isfinite(m):
                ax.vlines(x, m - dn, m + up, color=col, linewidth=3, alpha=0.95)
                ax.plot(x, m, "o", ms=8, color=col)

            # optional raw points (use same global toggle)
            if DOT_CI_SHOW_DOTS and arr.size:
                ax.scatter(
                    jittered_x(x, len(arr), 0.02), arr, s=18, alpha=0.55,
                    edgecolor="k", linewidth=0.3, color=col
                )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["MI", "REST"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Lean %")
    err_label = "SE" if DOT_STD_USE_SE else "SD"
    asym = "asym" if DOT_STD_ASYMMETRIC_SE else "sym"
    ax.set_title(f"Bar Dynamics — Mean • ±{err_label} by Run")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(handles, labels, title="Run")
    plt.tight_layout()
    plt.show()


def plot_dot_ci_runwise(mi_by_run, rest_by_run):
    fig, ax = plt.subplots()
    k = len(runs_sorted)
    off = offsets(k, base=0.18)

    # legend proxies (one per run color)
    handles, labels = [], []
    seen = set()

    for j, run in enumerate(runs_sorted):
        col = RUN_COLORS[j % len(RUN_COLORS)]
        if run not in seen:
            handles.append(
                Line2D([0], [0], color=col, marker="o", linestyle="-", linewidth=3)
            )
            labels.append(run)
            seen.add(run)

        mi_vals = mi_by_run.get(run, np.array([]))
        re_vals = rest_by_run.get(run, np.array([]))

        mi_mean, (mi_lo, mi_hi) = mean_ci(mi_vals) if mi_vals.size else (
            np.nan, (np.nan, np.nan)
        )
        re_mean, (re_lo, re_hi) = mean_ci(re_vals) if re_vals.size else (
            np.nan, (np.nan, np.nan)
        )

        if DOT_CI_SHOW_DOTS:
            if mi_vals.size:
                ax.scatter(
                    jittered_x(0 + off[j], len(mi_vals), 0.02), mi_vals,
                    s=20, alpha=0.5, edgecolor="k", linewidth=0.3, color=col
                )
            if re_vals.size:
                ax.scatter(
                    jittered_x(1 + off[j], len(re_vals), 0.02), re_vals,
                    s=20, alpha=0.5, edgecolor="k", linewidth=0.3, color=col
                )

        if np.isfinite(mi_mean):
            ax.plot([0 + off[j]], [mi_mean], marker="o", ms=8, color=col)
            ax.vlines(0 + off[j], mi_lo, mi_hi, color=col, linewidth=3)
        if np.isfinite(re_mean):
            ax.plot([1 + off[j]], [re_mean], marker="o", ms=8, color=col)
            ax.vlines(1 + off[j], re_lo, re_hi, color=col, linewidth=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["MI", "REST"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Lean %")
    ax.set_title(f"Bar Dynamics — Mean with 95% CI by Run")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(handles, labels, title="Run")
    plt.tight_layout()
    plt.show()


def plot_violin_runwise(lean_df_subset):
    fig, ax = plt.subplots()
    if lean_df_subset.empty:
        print("⚠️ Nothing to plot.")
        return
    sns.violinplot(
        data=lean_df_subset,
        x="Class",
        y="LeanPct",
        hue="RunID",
        inner=None,
        cut=0,
        dodge=True,
        ax=ax
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Lean %")
    ax.set_title(f"Bar Dynamics — Violin by Run")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(title="Run")
    plt.tight_layout()
    plt.show()


def plot_ecdf_runwise(mi_by_run, rest_by_run):
    def ecdf(a):
        a = np.sort(a[np.isfinite(a)])
        y = np.linspace(0, 1, len(a), endpoint=True)
        return a, y

    fig, ax = plt.subplots()
    for j, run in enumerate(runs_sorted):
        col = RUN_COLORS[j % len(RUN_COLORS)]
        if mi_by_run[run].size:
            x_mi, y_mi = ecdf(mi_by_run[run])
            ax.step(
                x_mi, y_mi, where="post", color=col, alpha=0.95,
                label=f"{run} (MI)" if j == 0 else None
            )
        if rest_by_run[run].size:
            x_re, y_re = ecdf(rest_by_run[run])
            ax.step(
                x_re, y_re, where="post", color=col, alpha=0.6,
                linestyle="--", label=f"{run} (REST)" if j == 0 else None
            )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Lean %")
    ax.set_ylabel("ECDF")
    ax.set_title(f"Lean% — ECDF by Run ({METRIC_MODE})")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ---------- SESSIONWISE (single figure) helpers ----------
def plot_box_sessionwise(lean_df_subset):
    fig, ax = plt.subplots()
    if lean_df_subset.empty or "SessionID" not in lean_df_subset.columns:
        print("⚠️ Nothing to plot (no SessionID).")
        return
    sns.boxplot(
        data=lean_df_subset,
        x="Class",
        y="LeanPct",
        hue="SessionID",
        dodge=True,
        showfliers=False,
        ax=ax
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Lean %")
    ax.set_title(f"Bar Dynamics — Session-wise, single figure")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Session")
    plt.tight_layout()
    plt.show()


def plot_dot_std_sessionwise(mi_by_sess, rest_by_sess):
    fig, ax = plt.subplots()
    k = len(sessions_sorted)
    off = offsets(k, base=0.18)

    handles, labels = [], []
    seen = set()

    for j, sess in enumerate(sessions_sorted):
        col = RUN_COLORS[j % len(RUN_COLORS)]
        if sess not in seen:
            handles.append(
                Line2D([0], [0], color=col, marker="o", linestyle="-", linewidth=3)
            )
            labels.append(sess)
            seen.add(sess)

        for x0, arr in [
            (0, mi_by_sess.get(sess, np.array([]))),
            (1, rest_by_sess.get(sess, np.array([])))
        ]:
            if arr.size == 0:
                continue
            if DOT_STD_ASYMMETRIC_SE:
                m, dn, up = asymmetric_se(arr, use_se=DOT_STD_USE_SE)
            else:
                m = float(np.nanmean(arr))
                s = float(np.nanstd(arr, ddof=1))
                if DOT_STD_USE_SE:
                    n = max(1, np.isfinite(arr).size)
                    s = s / np.sqrt(n)
                dn = up = s

            x = x0 + off[j]
            if np.isfinite(m):
                ax.vlines(x, m - dn, m + up, color=col, linewidth=3, alpha=0.95)
                ax.plot(x, m, "o", ms=8, color=col)

            if DOT_CI_SHOW_DOTS and arr.size:
                ax.scatter(
                    jittered_x(x, len(arr), 0.02), arr, s=18, alpha=0.55,
                    edgecolor="k", linewidth=0.3, color=col
                )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["MI", "REST"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Lean %")
    err_label = "SE" if DOT_STD_USE_SE else "SD"
    asym = "asym" if DOT_STD_ASYMMETRIC_SE else "sym"
    ax.set_title(
        f"Bar Dynamics — Mean • ±{err_label} by Session"
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(handles, labels, title="Session")
    plt.tight_layout()
    plt.show()


def plot_dot_ci_sessionwise(mi_by_sess, rest_by_sess):
    fig, ax = plt.subplots()
    k = len(sessions_sorted)
    off = offsets(k, base=0.18)

    handles, labels = [], []
    seen = set()

    for j, sess in enumerate(sessions_sorted):
        col = RUN_COLORS[j % len(RUN_COLORS)]
        if sess not in seen:
            handles.append(
                Line2D([0], [0], color=col, marker="o", linestyle="-", linewidth=3)
            )
            labels.append(sess)
            seen.add(sess)

        mi_vals = mi_by_sess.get(sess, np.array([]))
        re_vals = rest_by_sess.get(sess, np.array([]))

        mi_mean, (mi_lo, mi_hi) = mean_ci(mi_vals) if mi_vals.size else (
            np.nan, (np.nan, np.nan)
        )
        re_mean, (re_lo, re_hi) = mean_ci(re_vals) if re_vals.size else (
            np.nan, (np.nan, np.nan)
        )

        if DOT_CI_SHOW_DOTS:
            if mi_vals.size:
                ax.scatter(
                    jittered_x(0 + off[j], len(mi_vals), 0.02), mi_vals,
                    s=20, alpha=0.5, edgecolor="k", linewidth=0.3, color=col
                )
            if re_vals.size:
                ax.scatter(
                    jittered_x(1 + off[j], len(re_vals), 0.02), re_vals,
                    s=20, alpha=0.5, edgecolor="k", linewidth=0.3, color=col
                )

        if np.isfinite(mi_mean):
            ax.plot([0 + off[j]], [mi_mean], marker="o", ms=8, color=col)
            ax.vlines(0 + off[j], mi_lo, mi_hi, color=col, linewidth=3)
        if np.isfinite(re_mean):
            ax.plot([1 + off[j]], [re_mean], marker="o", ms=8, color=col)
            ax.vlines(1 + off[j], re_lo, re_hi, color=col, linewidth=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["MI", "REST"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Lean %")
    ax.set_title(f"Bar Dynamics — Mean with 95% CI by Session")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(handles, labels, title="Session")
    plt.tight_layout()
    plt.show()


def plot_violin_sessionwise(lean_df_subset):
    fig, ax = plt.subplots()
    if lean_df_subset.empty or "SessionID" not in lean_df_subset.columns:
        print("⚠️ Nothing to plot (no SessionID).")
        return
    sns.violinplot(
        data=lean_df_subset,
        x="Class",
        y="LeanPct",
        hue="SessionID",
        inner=None,
        cut=0,
        dodge=True,
        ax=ax
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Lean %")
    ax.set_title(f"Bar Dynamics — Violin by Session ")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(title="Session")
    plt.tight_layout()
    plt.show()


def plot_ecdf_sessionwise(mi_by_sess, rest_by_sess):
    def ecdf(a):
        a = np.sort(a[np.isfinite(a)])
        y = np.linspace(0, 1, len(a), endpoint=True)
        return a, y

    fig, ax = plt.subplots()
    for j, sess in enumerate(sessions_sorted):
        col = RUN_COLORS[j % len(RUN_COLORS)]
        if mi_by_sess[sess].size:
            x_mi, y_mi = ecdf(mi_by_sess[sess])
            ax.step(
                x_mi, y_mi, where="post", color=col, alpha=0.95,
                label=f"{sess} (MI)" if j == 0 else None
            )
        if rest_by_sess[sess].size:
            x_re, y_re = ecdf(rest_by_sess[sess])
            ax.step(
                x_re, y_re, where="post", color=col, alpha=0.6,
                linestyle="--", label=f"{sess} (REST)" if j == 0 else None
            )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Lean %")
    ax.set_ylabel("ECDF")
    ax.set_title(f"Bar Dynamics — ECDF by Session")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ---- Main plotting dispatch with GROUP_MODE ----
if plot_style == "forest_runwise":
    # Always run-wise; it already summarizes by run
    runs = runs_sorted
    rows = []
    for run in runs:
        for cls in ["MI", "REST"]:
            vals = lean_df.loc[
                (lean_df["RunID"] == run) & (lean_df["Class"] == cls),
                "LeanPct"
            ].astype(float).values
            m, (lo, hi) = mean_ci(vals, n_boot=1000, alpha=0.05)
            rows.append({"RunID": run, "Class": cls, "Mean": m, "Lo": lo, "Hi": hi})
    forest = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(runs))))
    y_ticks, y_labels, y = [], [], 0
    for run in runs:
        for cls in ["MI", "REST"]:
            row = forest[(forest["RunID"] == run) & (forest["Class"] == cls)].iloc[0]
            col = COL_MI if cls == "MI" else COL_REST
            ax.hlines(y, row["Lo"], row["Hi"], linewidth=3, color=col)
            ax.plot(row["Mean"], y, "o", color=col, ms=8)
            y_labels.append(f"{run} — {cls}")
            y_ticks.append(y)
            y += 1
        y += 0.3
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Lean % (mean with 95% CI)")
    ax.set_title(f"Bar Dynamics — Run-wise forest plot")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

else:
    if GROUP_MODE == "total":
        # ---- TOTAL (aggregate) mode: one figure for all runs & sessions combined ----
        if plot_style == "box":
            plot_box_total(mi_vals_total, rest_vals_total, title_suffix="(All Data)")
        elif plot_style == "dot_std":
            plot_dot_std_total(mi_vals_total, rest_vals_total, title_suffix="(All Data)")
        elif plot_style == "dot_ci":
            plot_dot_ci_total(mi_vals_total, rest_vals_total, title_suffix="(All Data)")
        elif plot_style == "violin":
            plot_violin_total(lean_df, title_suffix="(All Data)")
        elif plot_style == "ecdf":
            plot_ecdf_total(mi_vals_total, rest_vals_total, title_suffix="(All Data)")

    elif GROUP_MODE == "run":
        # ---- RUNWISE on a single figure (grouped by RunID) ----
        mi_by_run, rest_by_run = get_runwise_vals()
        if plot_style == "box":
            plot_box_runwise(lean_df)
        elif plot_style == "dot_std":
            plot_dot_std_runwise(mi_by_run, rest_by_run)
        elif plot_style == "dot_ci":
            plot_dot_ci_runwise(mi_by_run, rest_by_run)
        elif plot_style == "violin":
            plot_violin_runwise(lean_df)
        elif plot_style == "ecdf":
            plot_ecdf_runwise(mi_by_run, rest_by_run)

    elif GROUP_MODE == "session":
        # ---- SESSIONWISE on a single figure (grouped by SessionID) ----
        if not sessions_sorted:
            print("⚠️ No SessionID info; falling back to TOTAL.")
            if plot_style == "box":
                plot_box_total(mi_vals_total, rest_vals_total, title_suffix="(All Data)")
            elif plot_style == "dot_std":
                plot_dot_std_total(mi_vals_total, rest_vals_total, title_suffix="(All Data)")
            elif plot_style == "dot_ci":
                plot_dot_ci_total(mi_vals_total, rest_vals_total, title_suffix="(All Data)")
            elif plot_style == "violin":
                plot_violin_total(lean_df, title_suffix="(All Data)")
            elif plot_style == "ecdf":
                plot_ecdf_total(mi_vals_total, rest_vals_total, title_suffix="(All Data)")
        else:
            mi_by_sess, rest_by_sess = get_sessionwise_vals()
            if plot_style == "box":
                plot_box_sessionwise(lean_df)
            elif plot_style == "dot_std":
                plot_dot_std_sessionwise(mi_by_sess, rest_by_sess)
            elif plot_style == "dot_ci":
                plot_dot_ci_sessionwise(mi_by_sess, rest_by_sess)
            elif plot_style == "violin":
                plot_violin_sessionwise(lean_df)
            elif plot_style == "ecdf":
                plot_ecdf_sessionwise(mi_by_sess, rest_by_sess)

    else:
        raise ValueError(f"Unknown GROUP_MODE: {GROUP_MODE}")


# ---- 3) Aggregated Confusion Matrix ----
if conf_matrices:
    cm = np.sum(conf_matrices, axis=0)
    total = np.sum(cm)
    correct = cm[0][0] + cm[1][1]

    total_with_ambiguous = total + total_ambiguous

    accuracy_inclusive = (
        100.0 * correct / total_with_ambiguous
        if total_with_ambiguous > 0 else 0.0
    )
    accuracy_exclusive = 100.0 * correct / total if total > 0 else 0.0

    title = (
        f"Aggregated Confusion Matrix\n"
        f"Total Accuracy: {accuracy_inclusive:.2f}% | "
        f"Decision Accuracy: {accuracy_exclusive:.2f}% | "
        f"Ambiguous Trials: {total_ambiguous}"
    )

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred MI", "Pred REST"],
        yticklabels=["Actual MI", "Actual REST"]
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No confusion matrix data to plot.")
