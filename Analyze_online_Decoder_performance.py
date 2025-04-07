import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config  # Import thresholds and paths from config
from sklearn.metrics import confusion_matrix
import os
import glob
from datetime import datetime

def plot_posterior_probabilities(posterior_probs):
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 20)

    for label, probs in posterior_probs.items():
        probs = np.array(probs).flatten()
        sns.histplot(probs, bins=bins, alpha=0.6, label=f"{label} Probability", kde=True)

    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Online - Posterior Probability Distribution Across Classes")
    plt.legend(title="True Class/Domain")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

def analyze_posterior_probabilities(probability_file):
    df = pd.read_csv(probability_file, delimiter=",")
    df.columns = ["P(REST)", "P(MI)", "Correct Class"]
    df["Class Label"] = df["Correct Class"].map({200: "MI", 100: "Rest"})

    threshold_mi = config.THRESHOLD_MI
    threshold_rest = config.THRESHOLD_REST

    print(f"Using Config Thresholds: MI={threshold_mi}, Rest={threshold_rest}")

    def classify(row):
        if row["P(MI)"] > threshold_mi:
            return 200
        elif row["P(REST)"] > threshold_rest:
            return 100
        else:
            return -1

    df["Predicted Class"] = df.apply(classify, axis=1)

    posterior_probs = {
        "Rest": df[df["Correct Class"] == 100]["P(REST)"],
        "MI": df[df["Correct Class"] == 200]["P(MI)"]
    }

    plot_posterior_probabilities(posterior_probs)

    

    

def aggregate_confusion_matrices(min_total=30):
    """
    Aggregates and plots confusion matrices grouped by date (model condition).
    Skips matrices with fewer than `min_total` total samples.

    Parameters:
    - subject (str): Participant identifier (e.g., "CLASS_SUBJ_833")
    - min_total (int): Minimum number of total classifications to include a confusion matrix.
    """
    subject = config.TRAINING_SUBJECT
    models_base_path = os.path.join("/home/arman-admin/Documents/CurrentStudy", f"sub-{subject}", "models")

    condition_groups = {}

    if not os.path.exists(models_base_path):
        print(f"[ERROR] Models directory not found: {models_base_path}")
        return

    for root, dirs, files in os.walk(models_base_path):
        for file in files:
            if "confusion_matrix" in file and file.endswith(".csv"):
                cm_path = os.path.join(root, file)
                try:
                    cm = pd.read_csv(cm_path, index_col=0).values
                except Exception as e:
                    print(f"Error loading {cm_path}: {e}")
                    continue

                if cm.sum() < min_total:
                    print(f"[SKIP] {cm_path} has only {cm.sum()} samples.")
                    continue

                try:
                    date_str = file.split("_")[-2]
                    timestamp = datetime.strptime(date_str, "%Y-%m-%d")
                    date_key = timestamp.date().isoformat()
                except Exception as e:
                    print(f"[ERROR] Couldn't parse date from {file}: {e}")
                    continue

                if date_key not in condition_groups:
                    condition_groups[date_key] = []
                condition_groups[date_key].append(cm)

    for date_key, matrices in condition_groups.items():
        aggregated = np.sum(matrices, axis=0)[:2, :2]
        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(
            aggregated,
            annot=True,
            fmt="d",
            cmap="Blues",
            linewidths=0.5,
            annot_kws={"size": 9},
            cbar=False,
            xticklabels=["Predicted 200 (Move)", "Predicted 100 (Rest)"],
            yticklabels=["Actual 200 (Correct Move)", "Actual 100 (Correct Rest)"]
        )
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        plt.title(f"Aggregated Confusion Matrix - {date_key}", fontsize=11)
        plt.xlabel("Predicted Class", fontsize=10)
        plt.ylabel("True Class", fontsize=10)

        # Calculate stats
        TP = aggregated[0, 0]
        FN = aggregated[0, 1]
        FP = aggregated[1, 0]
        TN = aggregated[1, 1]

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        stats_text = (f"Accuracy: {accuracy:.2f}  |  "
                      f"Precision: {precision:.2f}  |  "
                      f"Recall: {recall:.2f}  |  "
                      f"F1 Score: {f1:.2f}")

        ax.text(0.5, -0.25, stats_text, ha='center', va='center', transform=ax.transAxes, fontsize=8)        
        plt.tight_layout(rect=[0, 0.12, 1, 1])
        plt.show()
# Example usage
if __name__ == "__main__":
    analyze_posterior_probabilities(
        "/home/arman-admin/Documents/CurrentStudy/sub-PILOT007/models/03_14 testing (adaptive reimanian)/classification_probabilities_2025-03-14_15-26-50.csv"
    )

    # To aggregate and display confusion matrices for a subject:
    aggregate_confusion_matrices()
