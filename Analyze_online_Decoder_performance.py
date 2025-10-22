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

# Validate necessary columns
required_cols = ["Trial", "Timestamp", "P(MI)", "P(REST)", "True Label", "Predicted Label", "Phase", "RunID"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column in CSV: {col}")

# ---- 1. Plot Posterior Probability Histograms ----
plt.figure(figsize=(10, 6))
bins = np.linspace(0, 1, 20)

sns.histplot(df[df["True Label"] == 200]["P(MI)"], bins=bins, kde=True, color="tab:blue", label="MI Trials")
sns.histplot(df[df["True Label"] == 100]["P(REST)"], bins=bins, kde=True, color="tab:orange", label="Rest Trials")

plt.xlabel("Posterior Probability")
plt.ylabel("Frequency")
plt.title("Histogram of Posterior Probabilities by True Class (Excl. ROBOT Phase)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# ---- 2. Plot All Trials Together (Correct Class Probabilities) ----
plt.figure(figsize=(12, 6))

# Ensure unique identification per trial across runs
df["GlobalTrialID"] = df["RunID"] + "_" + df["Trial"].astype(str)

for gtid in df["GlobalTrialID"].unique():
    trial_data = df[df["GlobalTrialID"] == gtid]
    true_label = trial_data["True Label"].iloc[0]

    if true_label == 200:
        color = "tab:blue"
        col = "P(MI)"
    elif true_label == 100:
        color = "tab:orange"
        col = "P(REST)"
    else:
        continue

    plt.plot(range(len(trial_data)), trial_data[col].values, color=color, alpha=0.5)

plt.xlabel("Time (relative index)")
plt.ylabel("Posterior Probability (Correct Class)")
plt.title("Posterior Probability per Trial (Excl. ROBOT Phase)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
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
