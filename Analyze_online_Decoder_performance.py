import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

def find_optimal_thresholds(y_true, y_scores, pos_label):
    """
    Finds the optimal threshold for classification using the ROC curve.

    Parameters:
        y_true (array-like): True class labels (100 for Rest, 200 for MI).
        y_scores (array-like): Predicted probabilities for MI or Rest.
        pos_label (int): The label to treat as the positive class (100 or 200).

    Returns:
        float: Optimal threshold for the given class.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    # Find optimal threshold using Youden’s J statistic
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", linewidth=2)
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f"Optimal Threshold = {optimal_threshold:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(f"ROC Curve for Class {pos_label}")
    plt.legend()
    plt.grid(True)
    plt.show()

    return optimal_threshold

def analyze_posterior_probabilities(probability_file):
    """
    Imports the probability CSV, computes the confusion matrix,
    and performs ROC analysis to find separate optimal thresholds for MI and Rest.

    Parameters:
        probability_file (str): Path to the CSV file containing posterior probabilities.
    """
    # Load probability data
    df = pd.read_csv(probability_file, delimiter=",")
    df.columns = ["P(REST)", "P(MI)", "Correct Class"]

    # Map true labels to descriptive names
    df["Class Label"] = df["Correct Class"].map({200: "MI", 100: "Rest"})

    # Always compute optimal thresholds dynamically
    print("Performing ROC analysis to find optimal thresholds...")
    t_mi = find_optimal_thresholds(df["Correct Class"], df["P(MI)"], pos_label=200)
    t_rest = find_optimal_thresholds(df["Correct Class"], df["P(REST)"], pos_label=100)

    print(f"\nComputed Optimal Thresholds for this run:")
    print(f"MI Threshold (T_MI): {t_mi:.2f}")
    print(f"Rest Threshold (T_Rest): {t_rest:.2f}")

    # Dual-threshold classification:
    def classify(row):
        if row["P(MI)"] > t_mi:
            return 200  # MI
        elif row["P(REST)"] > t_rest:
            return 100  # Rest
        else:
            return -1  # Ambiguous zone (optional handling)

    df["Predicted Class"] = df.apply(classify, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(df["Correct Class"], df["Predicted Class"], labels=[200, 100, -1])

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5,
                xticklabels=["Predicted MI (200)", "Predicted Rest (100)", "Ambiguous (-1)"],
                yticklabels=["Actual MI (200)", "Actual Rest (100)", "Ambiguous (-1)"])
    plt.title("Confusion Matrix with Dual Thresholds")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.show()

    # Print confusion matrix
    print("\nConfusion Matrix with Dual Thresholds:\n", pd.DataFrame(cm,
        columns=["Predicted MI (200)", "Predicted Rest (100)", "Ambiguous (-1)"],
        index=["Actual MI (200)", "Actual Rest (100)", "Ambiguous (-1)"]))

# Example usage
analyze_posterior_probabilities(
    "/home/arman-admin/Documents/CurrentStudy/sub-PILOT007/models/03_14 testing (adaptive reimanian)/classification_probabilities_2025-03-14_15-26-50.csv"
)
