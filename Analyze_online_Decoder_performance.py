import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import numpy as np

# ---- Configurable Subject ----
subject = "CLIN_SUBJ_003"

# ---- Prompt User to Select Session Subdirectory ----
base_dir = os.path.expanduser(f"~/Documents/CurrentStudy/sub-{subject}")
session_root = os.path.join(base_dir)
if not os.path.exists(session_root):
    raise FileNotFoundError(f"❌ Subject directory not found: {session_root}")

session_dirs = [d for d in os.listdir(session_root) if os.path.isdir(os.path.join(session_root, d)) and d.startswith("ses-")]
if not session_dirs:
    raise FileNotFoundError(f"❌ No session directories found in: {session_root}")

print("Available sessions:")
for idx, s in enumerate(session_dirs):
    print(f" [{idx}] {s}")

selected = input("➡️  Select a session by index: ").strip()
try:
    selected_idx = int(selected)
    session = session_dirs[selected_idx]
except:
    raise ValueError("❌ Invalid session selection.")

log_dir = os.path.join(session_root, session, "logs")

# Prompt for run folder
run_dirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d)) and d.startswith("ONLINE_")]
if not run_dirs:
    raise FileNotFoundError(f"❌ No ONLINE_ run folders found in: {log_dir}")

print("Available run directories:")
for idx, r in enumerate(run_dirs):
    print(f" [{idx}] {r}")

selected_run = input("➡️  Select run index(es) (comma-separated) or press ENTER to merge all: ").strip()

if selected_run:
    selected_indices = [int(i) for i in selected_run.split(",")]
    selected_run_dirs = [run_dirs[i] for i in selected_indices]
else:
    selected_run_dirs = run_dirs

# ---- Load and Combine Data ----
df_list = []
conf_matrices = []
total_ambiguous = 0
run_ids = []

for run_idx, run_folder in enumerate(selected_run_dirs):
    run_path = os.path.join(log_dir, run_folder)
    csv_files = [f for f in os.listdir(run_path) if f.startswith("decoder_output") and f.endswith(".csv")]
    if not csv_files:
        print(f"⚠️ No decoder output CSV in: {run_path}")
        continue

    csv_path = os.path.join(run_path, csv_files[0])
    print(f"✅ Loaded decoder output from: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df[df["Phase"] != "ROBOT"]  # Exclude ROBOT phase
    df["RunID"] = f"run_{run_idx}"  # Add run ID to identify source
    df_list.append(df)

    # Parse confusion matrix from event_log.txt
    log_path = os.path.join(run_path, "event_log.txt")
    if os.path.exists(log_path):
        with open(log_path, "r") as log_file:
            log_text = log_file.read()
        match = re.search(
            r"Actual 200.*?MI\): (\d+).*?REST\): (\d+).*?Actual 100.*?MI\): (\d+).*?REST\): (\d+)",
            log_text,
            re.DOTALL
        )
        ambiguous_match = re.search(r"Ambiguous trials.*?: (\d+)", log_text)

        if match:
            a200_p200 = int(match.group(1))
            a200_p100 = int(match.group(2))
            a100_p200 = int(match.group(3))
            a100_p100 = int(match.group(4))
            conf_matrix = [[a200_p200, a200_p100], [a100_p200, a100_p100]]
            conf_matrices.append(conf_matrix)
            print(f"✅ Parsed confusion matrix from: {log_path}")

        if ambiguous_match:
            total_ambiguous += int(ambiguous_match.group(1))

    else:
        print(f"⚠️ event_log.txt not found in: {run_path}")

# ---- Combine Decoder Data ----
df = pd.concat(df_list, ignore_index=True)

# ---- Flexible column detection (old vs new schema) ----
has_new = ("P(MI)_inst" in df.columns) and ("P(REST)_inst" in df.columns)
MI_COL   = "P(MI)_inst"   if has_new else "P(MI)"
REST_COL = "P(REST)_inst" if has_new else "P(REST)"

# ---- Validate necessary columns ----
required_always = ["Trial", "Timestamp", "True Label", "Predicted Label", "Phase", "RunID"]
for col in required_always:
    if col not in df.columns:
        raise ValueError(f"Missing required column in CSV: {col}")
for col in [MI_COL, REST_COL]:
    if col not in df.columns:
        raise ValueError(f"Missing required probability column in CSV: {col}")

# ---- Ensure unique identification per trial across runs ----
df["GlobalTrialID"] = df["RunID"] + "_" + df["Trial"].astype(str)

# Exclude ROBOT phase for these visualizations
df_vis = df[df["Phase"] != "ROBOT"].copy()

# ==========================================
# 1) Posterior Probability per Trial — ALL on ONE figure
#    (Correct-class instantaneous probability)
# ==========================================
import matplotlib.pyplot as plt

plt.figure()
labeled = {"MI": False, "REST": False}  # to avoid legend spam

for gtid in df_vis["GlobalTrialID"].unique():
    trial_data = df_vis[df_vis["GlobalTrialID"] == gtid]
    if trial_data.empty:
        continue

    true_label = int(trial_data["True Label"].iloc[0])
    if true_label == 200:   # MI trial
        col   = MI_COL
        color = "tab:orange"   # MI = orange
        lab   = "MI" if not labeled["MI"] else None
        labeled["MI"] = True
    elif true_label == 100: # REST trial
        col   = REST_COL
        color = "tab:blue"     # REST = blue
        lab   = "REST" if not labeled["REST"] else None
        labeled["REST"] = True
    else:
        continue

    y = trial_data[col].astype(float).values
    x = range(len(y))
    plt.plot(x, y, color=color, alpha=0.5, linewidth=1.2, label=lab)

plt.xlabel("Time (relative index)")
plt.ylabel("Posterior Probability (Correct Class)")
plt.title("Posterior Probability per Trial (Excl. ROBOT Phase) — All Trials")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

## ---- After you build df_vis, MI_COL, REST_COL as you already do ----
import numpy as np
import matplotlib.pyplot as plt

# Compute per-trial lean % (percentage of classifications where correct-class inst prob > 0.5)
lean_records = []  # rows: {GlobalTrialID, RunID, Class, LeanPct}

for gtid, trial_df in df_vis.groupby("GlobalTrialID"):
    if trial_df.empty:
        continue

    true_label = int(trial_df["True Label"].iloc[0])
    run_id     = str(trial_df["RunID"].iloc[0])

    if true_label == 200:   # MI trial -> use MI_COL
        cls = "MI"
        vals = trial_df[MI_COL].astype(float).values
    elif true_label == 100: # REST trial -> use REST_COL
        cls = "REST"
        vals = trial_df[REST_COL].astype(float).values
    else:
        continue

    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        continue

    lean_pct = float((vals > 0.5).mean() * 100.0)
    lean_records.append({
        "GlobalTrialID": gtid,
        "RunID": run_id,
        "Class": cls,
        "LeanPct": lean_pct
    })

# Convert to DataFrame for convenience (optional)
lean_df = pd.DataFrame(lean_records)

# Split by class for plotting
mi_vals   = lean_df.loc[lean_df["Class"] == "MI",   "LeanPct"].astype(float).values
rest_vals = lean_df.loc[lean_df["Class"] == "REST", "LeanPct"].astype(float).values

# ---- Single box-and-whisker figure (all runs on ONE plot) ----
fig, ax = plt.subplots()

# Boxplot: MI (orange) at x=1, REST (blue) at x=2
box = ax.boxplot(
    [mi_vals, rest_vals],
    labels=["MI", "REST"],
    widths=0.5,
    patch_artist=True,
    showfliers=False
)

# Color the boxes
colors = ["tab:orange", "tab:blue"]  # MI=orange, REST=blue
for patch, c in zip(box["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.45)
for median in box["medians"]:
    median.set_linewidth(2)

# Optional: overlay jittered points for each trial (to see distribution)
def jittered_x(center, n, scale=0.06):
    if n <= 1:
        return np.array([center])
    j = np.linspace(-1, 1, n) * scale
    # shuffle a bit so they aren't in order
    rng = np.random.default_rng(42)
    rng.shuffle(j)
    return center + j

# MI dots at x=1, REST dots at x=2 (aligned over their respective boxes)
x_mi   = jittered_x(1.0, len(mi_vals))
x_rest = jittered_x(2.0, len(rest_vals))
ax.scatter(x_mi,   mi_vals,   s=30, alpha=0.9, edgecolor="black", linewidth=0.5, color="tab:orange")
ax.scatter(x_rest, rest_vals, s=30, alpha=0.9, edgecolor="black", linewidth=0.5, color="tab:blue")

ax.set_ylim(0, 100)
ax.set_ylabel("Lean % (correct-class prob > 0.5)")
ax.set_title("Bar Dynamics — Lean % per Trial (All Runs on One Plot)")
ax.grid(True, axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
plt.show()

# ---- 3. Plot Aggregated Confusion Matrix ----
if conf_matrices:
    cm = np.sum(conf_matrices, axis=0)
    total = np.sum(cm)
    correct = cm[0][0] + cm[1][1]

    total_with_ambiguous = total + total_ambiguous

    accuracy_inclusive = 100.0 * correct / total_with_ambiguous if total_with_ambiguous > 0 else 0.0
    accuracy_exclusive = 100.0 * correct / total if total > 0 else 0.0

    title = (f"Aggregated Confusion Matrix\n"
             f"Total Accuracy: {accuracy_inclusive:.2f}% | "
             f"Decision Accuracy: {accuracy_exclusive:.2f}% | "
             f"Ambiguous Trials: {total_ambiguous}")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred MI", "Pred REST"],
                yticklabels=["Actual MI", "Actual REST"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No confusion matrix data to plot.")
