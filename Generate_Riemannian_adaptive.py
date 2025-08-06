import os
import numpy as np
import pickle
import mne
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from pyriemann.estimation import Shrinkage
from pyriemann.classification import MDM, FgMDM
from pyriemann.estimation import Covariances, XdawnCovariances
import config
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from Utils.stream_utils import load_xdf, get_channel_names_from_xdf
from Utils.preprocessing import select_motor_channels
import glob  # Required for multi-file loading
from scipy.stats import zscore
from pyriemann.utils.mean import mean_riemann
from scipy.linalg import sqrtm
import seaborn as sns
from sklearn.covariance import LedoitWolf
from pyriemann.preprocessing import Whitening
from Utils.preprocessing import (
    get_valid_channel_mask_and_metadata,
    initialize_filter_bank,
    apply_streaming_filters
)


# Load trigger mappings from config
TRIGGERS = config.TRIGGERS

# Define Relevant Markers for Classification (Exclude Robot Move - 300 & 320)
EPOCHS_START_END = {
    config.TRIGGERS["REST_BEGIN"]: config.TRIGGERS["REST_END"],  # 100 → 120
    config.TRIGGERS["MI_BEGIN"]: config.TRIGGERS["MI_END"],      # 200 → 220
}




def plot_posterior_probabilities(posterior_probs):
    """
    Plots the histogram of posterior probabilities for each class.

    Parameters:
        posterior_probs (dict): Dictionary containing posterior probabilities for each class.
    """
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 20)  # Set bins for histogram

    # Convert numerical labels to "Rest" and "MI"
    label_map = {100: "Rest", 200: "MI"}  # Ensures proper text labels
    renamed_probs = {label_map[int(label)]: probs for label, probs in posterior_probs.items()}

    for label, probs in renamed_probs.items():
        probs = np.array(probs).flatten()
        sns.histplot(probs, bins=bins, alpha=0.6, label=f"{label} Probability", kde=True)

    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Posterior Probability Distribution Across Classes")
    plt.legend(title="True Class")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def validate_trial_pairs(marker_values, marker_timestamps, eeg_timestamps, sfreq, EPOCHS_START_END, min_duration=1.0):
    """
    Validates trial start/end pairs and prints duration and skip-safety checks.

    Parameters:
        marker_values: np.array of int
        marker_timestamps: np.array of float (seconds)
        eeg_timestamps: np.array of float (seconds)
        sfreq: float sampling frequency
        EPOCHS_START_END: dict of {start_marker: end_marker}
        min_duration: float, minimum seconds required to allow 1s skip
    """
    print("\n🔍 Pre-validating trial start/end pairs...")
    
    for start_marker, end_marker in EPOCHS_START_END.items():
        start_indices = np.where(marker_values == int(start_marker))[0]
        end_indices = np.where(marker_values == int(end_marker))[0]

        print(f"\n🔹 Validating marker pair {start_marker} → {end_marker}")
        print(f"   Found {len(start_indices)} start markers, {len(end_indices)} end markers")

        if len(start_indices) != len(end_indices):
            print("   ⚠️ Mismatch in marker counts — trimming to shortest length")
            min_len = min(len(start_indices), len(end_indices))
            start_indices = start_indices[:min_len]
            end_indices = end_indices[:min_len]

        for i, (s_idx, e_idx) in enumerate(zip(start_indices, end_indices)):
            t_start = marker_timestamps[s_idx]
            t_end = marker_timestamps[e_idx]
            duration = t_end - t_start
            safe_to_skip = duration > min_duration

            if not safe_to_skip:
                print(f"   ❌ Trial {i}: {duration:.2f}s < {min_duration}s → will be invalid if 1s is skipped")
            else:
                print(f"   ✅ Trial {i}: {duration:.2f}s → OK to skip 1s")

        print(f"   Finished validating {len(start_indices)} trials for marker {start_marker}")

def segment_and_label_one_run(eeg_stream, marker_stream):
    """
    Preprocesses and segments EEG data for one run, closely replicating online preprocessing logic.
    Includes causal filtering with state tracking, baseline subtraction, and sliding window segmentation.

    Parameters:
        eeg_stream: dict containing EEG data and timestamps
        marker_stream: dict containing marker data and timestamps

    Returns:
        segments_all: np.ndarray of shape (n_segments, n_channels, n_samples)
        labels_all: np.ndarray of shape (n_segments,)
    """
    marker_values = np.array([int(m[0]) for m in marker_stream["time_series"]])
    marker_timestamps = np.array([float(m[1]) for m in marker_stream["time_series"]])
    eeg_timestamps = np.array(eeg_stream["time_stamps"])
    eeg_data = np.array(eeg_stream["time_series"]).T

    # Select valid EEG channels and metadata cleanup
    channel_names = get_channel_names_from_xdf(eeg_stream)
    eeg_data, valid_channels, dummy_raw = get_valid_channel_mask_and_metadata(eeg_data, channel_names)

    if config.SURFACE_LAPLACIAN_TOGGLE:
        dummy_raw = mne.preprocessing.compute_current_source_density(dummy_raw)

    if config.SELECT_MOTOR_CHANNELS:
        dummy_raw = select_motor_channels(dummy_raw)
        motor_indices = [valid_channels.index(ch) for ch in dummy_raw.ch_names]
        eeg_data = eeg_data[motor_indices]
        valid_channels = dummy_raw.ch_names

    # === Filter setup ===
    filter_bank = initialize_filter_bank(
        fs=config.FS,
        lowcut=config.LOWCUT,
        highcut=config.HIGHCUT,
        notch_freqs=[60],
        notch_q=30
    )
    filter_state = {}  # Stateful tracking


    # === Segmentation configuration ===
    window_size = config.CLASSIFY_WINDOW / 1000
    step_size = 1 / 16
    window_samples = int(window_size * config.FS)
    step_samples = int(step_size * config.FS)
    chunk_samples = int(window_size * config.FS)  # based on config.py parameters (256 for a 512Hz window)

    segments_all = []
    labels_all = []

    # === Precompute all trial windows ===
    trial_windows = []
    for start_marker, end_marker in EPOCHS_START_END.items():
        start_indices = np.where(marker_values == int(start_marker))[0]
        end_indices = np.where(marker_values == int(end_marker))[0]
        if len(start_indices) != len(end_indices):
            min_len = min(len(start_indices), len(end_indices))
            start_indices = start_indices[:min_len]
            end_indices = end_indices[:min_len]
        for s_idx, e_idx in zip(start_indices, end_indices):
            ts_start = marker_timestamps[s_idx]
            ts_end = marker_timestamps[e_idx]
            if ts_end - ts_start > 1.0:
                trial_windows.append((ts_start, ts_end, int(start_marker)))

    # === Sort windows and get min/max bounds ===
    trial_windows.sort()
    filter_warmup = 1.0 
    trial_bounds = [(start - 1.0, end) for (start, end, _) in trial_windows]
    valid_start = trial_bounds[0][0] - filter_warmup
    valid_end = trial_bounds[-1][1]

    # === Extract only relevant data segment ===
    global_start = np.searchsorted(eeg_timestamps, valid_start) 
    global_end = np.searchsorted(eeg_timestamps, valid_end)
    raw_global = eeg_data[:, global_start:global_end]
    rel_timestamps = eeg_timestamps[global_start:global_end]

    # === Stream through global segment with filter continuity ===
    filter_state = {}
    filtered_global = np.zeros_like(raw_global)

    for chunk_start in range(0, raw_global.shape[1], chunk_samples):
        chunk_end = min(chunk_start + chunk_samples, raw_global.shape[1])
        chunk = raw_global[:, chunk_start:chunk_end]
        filtered_chunk, filter_state = apply_streaming_filters(chunk, filter_bank, filter_state)
        filtered_global[:, chunk_start:chunk_end] = filtered_chunk

    # === For each trial, extract and label segments ===
    for trial_num, (ts_start, ts_end, label) in enumerate(trial_windows):
        rel_start = np.searchsorted(rel_timestamps, ts_start)
        rel_end = np.searchsorted(rel_timestamps, ts_end)
        baseline_end = np.searchsorted(rel_timestamps, ts_start)
        decision_start = np.searchsorted(rel_timestamps, ts_start + 1.0)

        if rel_end <= decision_start or baseline_end <= 0:
            continue

        baseline = filtered_global[:, :baseline_end].mean(axis=1, keepdims=True)
        trial_data = filtered_global[:, decision_start:rel_end] - baseline
        n_samples = trial_data.shape[1]

        if n_samples < window_samples:
            continue

        for i in range(0, n_samples - window_samples + 1, step_samples):
            segment = trial_data[:, i:i + window_samples]
            segments_all.append(segment)
            labels_all.append(label)


    return np.array(segments_all), np.array(labels_all)



def compute_processed_covariances(segments, labels):
    """
    Computes regularized and optionally whitened covariance matrices from EEG segments.

    Args:
        segments (np.ndarray): EEG data segments, shape (n_trials, n_channels, n_timepoints).
        labels (np.ndarray): Labels corresponding to each segment.
        fs (float): Sampling frequency, used for future flexibility (not required now).

    Returns:
        np.ndarray: Processed covariance matrices.
    """
    print("Computing raw covariance matrices...")
    cov_matrices = np.array([
        (seg @ seg.T) / np.trace(seg @ seg.T) for seg in segments
    ])

    print(f"Covariance shape: {cov_matrices.shape}")
    print("Label distribution:", dict(zip(*np.unique(labels, return_counts=True))))

    if config.LEDOITWOLF:
        print("Applying Ledoit-Wolf shrinkage...")
        cov_matrices = np.array([
            LedoitWolf().fit(cov).covariance_ for cov in cov_matrices
        ])
    else:
        print(f"Applying shrinkage (λ={config.SHRINKAGE_PARAM})...")
        shrinker = Shrinkage(shrinkage=config.SHRINKAGE_PARAM)
        cov_matrices = shrinker.fit_transform(cov_matrices)

    if config.RECENTERING:
        print("Applying Riemannian whitening...")
        whitener = Whitening(metric="riemann")
        cov_matrices = whitener.fit_transform(cov_matrices)

    print(f"Processed covariance matrices shape: {cov_matrices.shape}")
    return cov_matrices


def train_riemannian_model(cov_matrices, labels, n_splits=8, shrinkage_param=config.SHRINKAGE_PARAM):
    """
    Train an MDM classifier with k-fold cross-validation using Riemannian geometry 
    and plot posterior probability histograms.

    Parameters:
        cov_matrices (np.ndarray): Covariance matrices of shape (n_samples, n_channels, n_channels).
        labels (np.ndarray): Corresponding labels for the segments.
        n_splits (int): Number of splits for cross-validation.
        shrinkage_param (float): Regularization strength for Shrinkage.

    Returns:
        mdm (MDM): Trained MDM model.
    """

    print("\n🚀 Starting K-Fold Cross Validation with Riemannian MDM...\n")


    
    # Compute the reference matrix (Riemannian mean)
    '''
    reference_matrix = mean_riemann(cov_matrices, maxiter=1000)

    # Center covariance matrices
    
    cov_matrices = np.array([np.linalg.inv(reference_matrix) @ cov @ np.linalg.inv(reference_matrix)
                              for cov in cov_matrices])
    
    '''

    #cov_matrices = center_cov_matrices_riemannian(cov_matrices)
    #cov_matrices = center_cov_matrices(cov_matrices, reference_matrix)
    # Apply Shrinkage-based regularization

    # Initialize cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True)
    mdm = MDM()
    accuracies = []
    posterior_probs = {label: [] for label in np.unique(labels)}  # Store probabilities per class

    for fold_idx, (train_index, test_index) in enumerate(kf.split(cov_matrices), start=1):
        X_train, X_test = cov_matrices[train_index], cov_matrices[test_index]
        Y_train, Y_test = labels[train_index], labels[test_index]

        # Train and evaluate model
        mdm.fit(X_train, Y_train)
        Y_pred = mdm.predict(X_test)
        Y_predProb = mdm.predict_proba(X_test)  # Get class probabilities

        accuracy = accuracy_score(Y_test, Y_pred)
        accuracies.append(accuracy)
        print(f"\n ✅ Fold {fold_idx} Accuracy: {accuracy:.4f}")

        # Store probabilities per class
        for idx, true_label in enumerate(Y_test):
            class_idx = np.where(mdm.classes_ == true_label)[0][0]
            posterior_probs[true_label].append(Y_predProb[idx, class_idx])

    # Convert probability lists to numpy arrays
    for label in posterior_probs:
        posterior_probs[label] = np.array(posterior_probs[label])

    # Plot probability histograms
    plot_posterior_probabilities(posterior_probs)

    # Print overall accuracy
    avg_accuracy = np.mean(accuracies)
    print(f"\n🚀 **Final Average Accuracy:** {avg_accuracy:.4f}")

    # Retrain the model on all data
    mdm.fit(cov_matrices, labels)

    return mdm

def main():
    """
    Main function to generate a Riemannian-based EEG decoder.
    """
    mne.set_log_level("WARNING")

    print("Loading XDF data...")
    eeg_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "training_data")
    print(f"Script is looking for XDF files in: {eeg_dir}")

    xdf_files = [
        os.path.join(eeg_dir, f) for f in os.listdir(eeg_dir)
        if f.endswith(".xdf") and "OBS" not in f
    ]

    if not xdf_files:
        raise FileNotFoundError(f"No XDF files found in: {eeg_dir}")
    print(f"Found XDF files: {xdf_files}")

    all_cov_matrices = []
    all_labels = []

    for xdf_path in xdf_files:
        print(f"\n📂 Processing file: {xdf_path}")
        eeg_stream, marker_stream = load_xdf(xdf_path)

        segments, labels = segment_and_label_one_run(eeg_stream, marker_stream)

        # Print summary
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_summary = ", ".join([f"{int(lbl)}: {cnt}" for lbl, cnt in zip(unique_labels, counts)])
        print("\n📊 Segmentation Summary:")
        print(f"🔹 Total segments: {len(segments)}")
        print(f"🔹 Segment shape: {segments.shape} (n_segments, n_channels, n_timepoints)")
        print(f"🔹 Class distribution: {label_summary}")


    
        # === HARD REJECTION BASED ON PEAK AMPLITUDE ===
        REJECTION_THRESHOLD_UV = 30  # μV

        # Compute max abs amplitude per segment
        max_vals = np.max(np.abs(segments), axis=(1, 2))  # shape: (n_segments,)
        keep_mask = max_vals <= REJECTION_THRESHOLD_UV

        # Apply rejection
        segments = segments[keep_mask]
        labels = labels[keep_mask]

        print(f"Retained {len(segments)} segments after rejecting {np.sum(~keep_mask)} high-amplitude artifacts.")

        
        cov_matrices = compute_processed_covariances(segments, labels)
        
        #print(mean_riemann(cov_matrices))
        all_cov_matrices.append(cov_matrices)
        all_labels.append(labels)

    all_labels = np.concatenate(all_labels)
    all_cov_matrices = np.concatenate(all_cov_matrices)
    print("Training Riemannian Classifier...")
    Reimans_model = train_riemannian_model(all_cov_matrices, all_labels)

    #  Save the trained model
    # Define model save path (subject-level, not session-specific)
    subject_model_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "models")
    os.makedirs(subject_model_dir, exist_ok=True)

    subject_model_path = os.path.join(subject_model_dir, f"sub-{config.TRAINING_SUBJECT}_model.pkl")

    
    # Save the trained model
    with open(subject_model_path, 'wb') as f:
        pickle.dump(Reimans_model, f)

    print(f"✅ Trained model saved at: {subject_model_path}")
    #np.save(Training_mean_path, training_mean)
    #np.save(Training_std_path, training_std)
    #print(" Saved precomputed training mean and std for real-time use.")


if __name__ == "__main__":
    main()
