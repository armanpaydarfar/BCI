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
from scipy.signal import butter, lfilter, lfilter_zi
from Utils.preprocessing import butter_bandpass, concatenate_streams, select_motor_channels
import glob  # Required for multi-file loading
from scipy.stats import zscore
from pyriemann.utils.mean import mean_riemann
from scipy.linalg import sqrtm
import seaborn as sns
from sklearn.covariance import LedoitWolf
from pyriemann.preprocessing import Whitening

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




# Stateful Filtering Function
def apply_stateful_filter(raw, b, a):
    filter_states = {}  # Initialize state tracking dictionary
    for ch_idx in range(len(raw.ch_names)):
        if ch_idx not in filter_states:
            filter_states[ch_idx] = lfilter_zi(b, a) * raw._data[ch_idx][0]  # Initialize filter state
        raw._data[ch_idx], filter_states[ch_idx] = lfilter(b, a, raw._data[ch_idx], zi=filter_states[ch_idx])
    return raw

def center_cov_matrices_riemannian(cov_matrices):
    """
    Center a set of covariance matrices around the identity matrix using the Riemannian mean.

    Parameters:
        cov_matrices (np.ndarray): Array of shape (n_matrices, n_channels, n_channels)

    Returns:
        np.ndarray: Centered covariance matrices
    """
    # Compute the Riemannian mean of the covariance matrices
    mean_cov = mean_riemann(cov_matrices, maxiter = 5000)

    # Compute the inverse square root of the mean covariance
    eigvals, eigvecs = np.linalg.eigh(mean_cov)
    inv_sqrt_mean_cov = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    # Apply whitening transformation to center around the identity
    centered_matrices = np.array([inv_sqrt_mean_cov @ C @ inv_sqrt_mean_cov for C in cov_matrices])

    return centered_matrices

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


def segment_trials_from_markers(raw, marker_values, marker_timestamps, eeg_timestamps, sfreq):
    segments_all = []
    labels_all = []

    window_size = config.CLASSIFY_WINDOW / 1000
    step_size = 1 / 16
    window_samples = int(window_size * sfreq)
    step_samples = int(step_size * sfreq)

    for start_marker, end_marker in EPOCHS_START_END.items():
        start_indices = np.where(marker_values == int(start_marker))[0]
        end_indices = np.where(marker_values == int(end_marker))[0]

        if len(start_indices) != len(end_indices):
            print(f"⚠️ Mismatch in marker count for {start_marker}/{end_marker} — trimming")
            min_len = min(len(start_indices), len(end_indices))
            start_indices = start_indices[:min_len]
            end_indices = end_indices[:min_len]

        for trial_num, (s_idx, e_idx) in enumerate(zip(start_indices, end_indices)):
            ts_start = marker_timestamps[s_idx]
            ts_end = marker_timestamps[e_idx]
            if ts_end - ts_start <= 1.0:
                print(f"❌ Trial {trial_num} skipped — too short ({ts_end - ts_start:.2f}s)")
                continue

            start_sample = np.searchsorted(eeg_timestamps, ts_start + 1.0)
            end_sample = np.searchsorted(eeg_timestamps, ts_end)

            if end_sample <= start_sample:
                print(f"❌ Trial {trial_num} invalid — end sample before start")
                continue

            baseline_start = max(0, start_sample + int(sfreq * -1.0))
            baseline_end = start_sample
            baseline = raw._data[:, baseline_start:baseline_end].mean(axis=1, keepdims=True)
            trial_slice = slice(start_sample, end_sample)
            raw._data[:, trial_slice] -= baseline
            trial_data = raw._data[:, trial_slice]
            n_samples = trial_data.shape[1]
            if n_samples < window_samples:
                print(f"⚠️ Trial {trial_num} too short after skip — {n_samples} samples")
                continue

            n_windows = (n_samples - window_samples) // step_samples + 1
            print(f"✅ Trial {trial_num} (label {start_marker}): {n_windows} segments")

            for i in range(0, n_samples - window_samples + 1, step_samples):
                segment = trial_data[:, i:i + window_samples]
                segments_all.append(segment)
                labels_all.append(int(start_marker))

    return np.array(segments_all), np.array(labels_all)


def segment_and_label_one_run(eeg_stream, marker_stream):

    marker_values = np.array([int(m[0]) for m in marker_stream["time_series"]])
    marker_timestamps = np.array([float(m[1]) for m in marker_stream["time_series"]])
    eeg_timestamps = np.array(eeg_stream["time_stamps"])
    eeg_data = np.array(eeg_stream["time_series"]).T

    channel_names = get_channel_names_from_xdf(eeg_stream)
    valid_channels = [ch for ch in channel_names if ch not in {"AUX1", "AUX2", "AUX3", "AUX7", "AUX8", "AUX9", "TRIGGER"}]
    valid_indices = [channel_names.index(ch) for ch in valid_channels]
    eeg_data = eeg_data[valid_indices]

    info = mne.create_info(ch_names=valid_channels, sfreq=config.FS, ch_types="eeg")
    raw = mne.io.RawArray(eeg_data, info)

    # Unit + montage setup
    for ch in raw.info["chs"]:
        ch["unit"] = 201
    if "M1" in raw.ch_names and "M2" in raw.ch_names:
        raw.drop_channels(["M1", "M2"])
    raw.rename_channels({
        "FP1": "Fp1", "FPZ": "Fpz", "FP2": "Fp2",
        "FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "POZ": "POz", "OZ": "Oz"
    })
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"))

    raw.notch_filter(60, method="iir")
    raw.filter(l_freq=config.LOWCUT, h_freq=config.HIGHCUT, method="iir")
    if config.SURFACE_LAPLACIAN_TOGGLE:
        raw = mne.preprocessing.compute_current_source_density(raw)
    if config.SELECT_MOTOR_CHANNELS:
        raw = select_motor_channels(raw)

    # Segment this run
    return segment_trials_from_markers(
        raw, marker_values, marker_timestamps, eeg_timestamps, config.FS
    )

'''
def segment_epochs(epochs, window_size=config.CLASSIFY_WINDOW, step_size=0.1):
    """
    Slice each epoch into smaller overlapping windows for training.

    Parameters:
        epochs (mne.Epochs): The full 5s epochs.
        window_size (float): Length of each training segment (e.g., 0.5s).
        step_size (float): Overlap between consecutive windows (e.g., 0.1s).
    
    Returns:
        np.ndarray: Segmented data (n_segments, n_channels, n_timepoints).
        np.ndarray: Corresponding labels for each segment.
    """
    window_size = window_size/1000
    sfreq = epochs.info["sfreq"]  # Sampling frequency
    step_samples = int(step_size * sfreq)  # Convert step size to samples
    window_samples = int(window_size * sfreq)  # Convert window size to samples

    segmented_data = []
    segmented_labels = []

    for i, epoch in enumerate(epochs.get_data()):  # Iterate over each full epoch
        label = epochs.events[i, -1]  # Get the class label

        for start in range(0, epoch.shape[1] - window_samples + 1, step_samples):
            end = start + window_samples
            segmented_data.append(epoch[:, start:end])
            segmented_labels.append(label)  # Each slice gets the same label

    return np.array(segmented_data), np.array(segmented_labels)
'''
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

        
        # Compute Covariance Matrices (for Riemannian Classifier)
        print("Computing Covariance Matrices...")
        #cov_matrices = np.array([np.cov(segment) for segment in segments])
        cov_matrices = np.array([ (segment @ segment.T) / np.trace(segment @ segment.T) for segment in segments ])

        # Convert list to numpy array (shape: (n_epochs, n_channels, n_channels))
        cov_matrices = np.array(cov_matrices)
        #print(cov_matrices[0])
        print(f"Computed {len(cov_matrices)} covariance matrices with shape: {cov_matrices.shape}")
        #print(f" Sample cov matrix: {cov_matrices[0]}")

        # Train Riemannian MDM Model
        #print(cov_matrices)
        print("Unique training labels:", np.unique(labels))

        if config.LEDOITWOLF:
            # Compute covariance matrices with optimized shrinkage
            cov_matrices_shrinked = np.array([LedoitWolf().fit(cov).covariance_ for cov in cov_matrices])
            cov_matrices = cov_matrices_shrinked    
        else:
            shrinkage = Shrinkage(shrinkage=config.SHRINKAGE_PARAM)
            cov_matrices = shrinkage.fit_transform(cov_matrices)  

        
        # Apply Riemannian whitening
        if config.RECENTERING:
            whitener = Whitening(metric="riemann")  # Use Riemannian mean for whitening
            cov_matrices = whitener.fit_transform(cov_matrices)
        
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
