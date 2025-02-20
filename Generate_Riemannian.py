import os
import numpy as np
import pickle
import mne
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from pyriemann.estimation import Shrinkage
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
import config
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from Utils.stream_utils import load_xdf, get_channel_names_from_xdf
from scipy.signal import butter, lfilter, lfilter_zi
from Utils.preprocessing import butter_bandpass
import glob  # Required for multi-file loading
from scipy.stats import zscore

# Load trigger mappings from config
TRIGGERS = config.TRIGGERS

# Define Relevant Markers for Classification (Exclude Robot Move - 300 & 320)
EPOCHS_START_END = {
    config.TRIGGERS["REST_BEGIN"]: config.TRIGGERS["REST_END"],  # 100 → 120
    config.TRIGGERS["MI_BEGIN"]: config.TRIGGERS["MI_END"],      # 200 → 220
}


# Stateful Filtering Function
def apply_stateful_filter(raw, b, a):
    filter_states = {}  # Initialize state tracking dictionary
    for ch_idx in range(len(raw.ch_names)):
        if ch_idx not in filter_states:
            filter_states[ch_idx] = lfilter_zi(b, a) * raw._data[ch_idx][0]  # Initialize filter state
        raw._data[ch_idx], filter_states[ch_idx] = lfilter(b, a, raw._data[ch_idx], zi=filter_states[ch_idx])
    return raw



def concatenate_streams(eeg_streams, marker_streams):
    """
    Concatenates multiple EEG and marker streams into a single dataset.
    Ensures metadata (e.g., 'info') is preserved.
    """
    merged_eeg = {"time_series": [], "time_stamps": [], "info": eeg_streams[0].get("info", {})}
    merged_markers = {"time_series": [], "time_stamps": []}

    time_offset = 0
    for i, (eeg, markers) in enumerate(zip(eeg_streams, marker_streams)):
        if i == 0:
            merged_eeg["time_series"] = eeg["time_series"]
            merged_eeg["time_stamps"] = eeg["time_stamps"]
            merged_markers["time_series"] = markers["time_series"]
            merged_markers["time_stamps"] = markers["time_stamps"]
        else:
            time_offset = merged_eeg["time_stamps"][-1] - eeg["time_stamps"][0] + np.mean(np.diff(eeg["time_stamps"]))
            merged_eeg["time_series"] = np.vstack([merged_eeg["time_series"], eeg["time_series"]])
            merged_eeg["time_stamps"] = np.concatenate([merged_eeg["time_stamps"], eeg["time_stamps"] + time_offset])
            merged_markers["time_series"] = np.vstack([merged_markers["time_series"], markers["time_series"]])
            merged_markers["time_stamps"] = np.concatenate([merged_markers["time_stamps"], markers["time_stamps"] + time_offset])

    return merged_eeg, merged_markers




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

def train_riemannian_model(cov_matrices, labels, n_splits=10, shrinkage_param=0.1):
    """
    Train an MDM classifier with k-fold cross-validation using Riemannian geometry and plot probability histograms.

    Parameters:
        cov_matrices (np.ndarray): Covariance matrices.
        labels (np.ndarray): Corresponding labels for the segments.
        n_splits (int): Number of splits for cross-validation.
        shrinkage_param (float): Regularization strength for Shrinkage.

    Returns:
        mdm: Trained MDM model.
    """
    kf = KFold(n_splits=n_splits, shuffle=False)
    mdm = MDM()

    accuracies = []
    all_probabilities = {label: [] for label in np.unique(labels)}  # Store probabilities per class

    print("\n Starting K-Fold Cross Validation with Riemannian MDM...\n")

    # Apply Shrinkage-based regularization
    shrinkage = Shrinkage(shrinkage=shrinkage_param)
    cov_matrices = shrinkage.fit_transform(cov_matrices)  # Apply shrinkage to all covariance matrices

    for fold_idx, (train_index, test_index) in enumerate(kf.split(cov_matrices), start=1):
        X_train, X_test = cov_matrices[train_index], cov_matrices[test_index]
        Y_train, Y_test = labels[train_index], labels[test_index]

        # Train and evaluate model
        mdm.fit(X_train, Y_train)
        Y_pred = mdm.predict(X_test)
        Y_predProb = mdm.predict_proba(X_test)  # Get probabilities

        accuracy = accuracy_score(Y_test, Y_pred)
        accuracies.append(accuracy)

        print(f"\n **Fold {fold_idx} Accuracy: {accuracy:.4f}**")

        # Store probabilities per class
        for idx, true_label in enumerate(Y_test):
            all_probabilities[true_label].append(Y_predProb[idx, np.where(mdm.classes_ == true_label)[0][0]])

    # Convert probability lists to numpy arrays
    for label in all_probabilities:
        all_probabilities[label] = np.array(all_probabilities[label])

    # Plot probability distributions per class
    plt.figure(figsize=(10, 5))
    bins = np.linspace(0, 1, 20)  # Set bins for histogram

    for label, probs in all_probabilities.items():
        plt.hist(probs, bins=bins, alpha=0.6, label=f"Class {label}")

    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Probability Distribution Across Classes")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    # Print overall average accuracy
    print(f"\n🚀 **Final Average Accuracy:** {np.mean(accuracies):.4f}")

    # Retrain the model on all data
    mdm.fit(cov_matrices, labels)

    return mdm


def main():
    """
    Main function to generate a Riemannian-based EEG decoder.
    """
    mne.set_log_level("WARNING")  # Options: "ERROR", "WARNING", "INFO", "DEBUG"

    print("Loading XDF data...")

    # Construct EEG directory path dynamically
    eeg_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "training_data")    
    print(f" Script is looking for XDF files in: {eeg_dir}")

    # Find all .xdf files in the session directory, excluding those with "OBS" in the filename
    xdf_files = [
        os.path.join(eeg_dir, f) for f in os.listdir(eeg_dir) if f.endswith(".xdf") and "OBS" not in f
    ]
    # Construct full paths
    xdf_files = [os.path.join(eeg_dir, f) for f in xdf_files]

    # Debugging output
    print(f" Found XDF files: {xdf_files}")

    # Ensure at least one file exists
    if not xdf_files:
        raise FileNotFoundError(f"No XDF files found in: {eeg_dir}")

    print(f" Found XDF files: {xdf_files}")

    # Load multiple XDF files (concatenating EEG streams if needed)
    eeg_streams, marker_streams = [], []
    for xdf_file in xdf_files:
        eeg_s, marker_s = load_xdf(xdf_file)
        eeg_streams.append(eeg_s)
        marker_streams.append(marker_s)

    # Merge multiple runs (if necessary)
    eeg_stream, marker_stream = (
        (eeg_streams[0], marker_streams[0]) if len(eeg_streams) == 1 else concatenate_streams(eeg_streams, marker_streams)
    )

    # Extract EEG and marker data
    eeg_timestamps = np.array(eeg_stream['time_stamps'])
    eeg_data = np.array(eeg_stream['time_series']).T  # (n_channels, n_samples)
    channel_names = get_channel_names_from_xdf(eeg_stream)

    marker_timestamps = np.array(marker_stream['time_stamps'])
    marker_values = np.array([int(m[0]) for m in marker_stream['time_series']])


    # Load standard 10-20 montage
    montage = mne.channels.make_standard_montage("standard_1020")

    # Case-sensitive renaming dictionary
    rename_dict = {
        "FP1": "Fp1", "FPZ": "Fpz", "FP2": "Fp2",
        "FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "POZ": "POz", "OZ": "Oz"
    }

    # Drop non-EEG channels
    non_eeg_channels = {"AUX1", "AUX2", "AUX3", "AUX7", "AUX8", "AUX9", "TRIGGER"}
    valid_eeg_channels = [ch for ch in channel_names if ch not in non_eeg_channels]

    # Filter data to keep only valid EEG channels
    valid_indices = [channel_names.index(ch) for ch in valid_eeg_channels]  # Get indices
    eeg_data = eeg_data[valid_indices, :]  # Keep only valid EEG data

    # Create MNE Raw Object
    sfreq = config.FS
    info = mne.create_info(ch_names=valid_eeg_channels, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(eeg_data, info)

    first_channel_unit = raw.info["chs"][0]["unit"]
    #print(f" First Channel Unit (FIFF Code): {first_channel_unit}")

    # Convert data from Volts to microvolts (µV)
    # Convert raw data from Volts to microvolts (µV) IMMEDIATELY AFTER LOADING
    #raw._data /= 1e6  # Convert V → µV

    # Update channel metadata in MNE so the scaling is correctly reflected

    for ch in raw.info['chs']:
        ch['unit'] = 201  # 201 corresponds to µV in MNE’s standard units
    # Print the first EEG channel’s metadata

    # Print to confirm the change
    #print(f" Updated Units for EEG Channels: {[ch['unit'] for ch in raw.info['chs']]}")

    if "M1" in raw.ch_names and "M2" in raw.ch_names:
        raw.drop_channels(["M1", "M2"])
        print("Removed Mastoid Channels: M1, M2")
    else:
        print("No Mastoid Channels Found in Data")


    # Rename channels to match montage format
    raw.rename_channels(rename_dict)


    # Debug: Print missing channels
    missing_in_montage = set(raw.ch_names) - set(montage.ch_names)
    print(f" Channels in Raw but Missing in Montage: {missing_in_montage}")
    raw.set_montage(montage)

    # Apply Notch & Bandpass Filtering
    # Apply Notch Filtering (Remove Powerline Noise)
    raw.notch_filter(60, method="iir")  

    # **Apply Bandpass Filtering with State Preservation**
    b, a = butter_bandpass(config.LOWCUT, config.HIGHCUT, sfreq, order=4)
    raw = apply_stateful_filter(raw, b, a)
    # Apply Common Average Reference (CAR)
    #raw.set_eeg_reference("average")

    # Apply Surface Laplacian (CSD) if enabled

    if config.SURFACE_LAPLACIAN_TOGGLE:
        raw = mne.preprocessing.compute_current_source_density(raw)
    
    scaler = StandardScaler()
    raw._data = scaler.fit_transform(raw.get_data())    # Dynamic Epoching Based on Start & End Markers

    #training_mean = scaler.mean_
    #training_std = scaler.scale_
    #np.save(precomputed_mean_path, training_mean)
    #np.save(precomputed_std_path, training_std)

    print(" Saved precomputed training mean and std for real-time use.")

    events = []
    event_id_map = {}

    baseline_duration = 0.5  # 500ms baseline before the event start

    for start_marker, end_marker in EPOCHS_START_END.items():
        start_indices = np.where(marker_values == int(start_marker))[0]
        end_indices = np.where(marker_values == int(end_marker))[0]

        if len(start_indices) != len(end_indices):
            print(f"Unequal markers: {start_marker} (start) and {end_marker} (end). Adjusting...")
            min_length = min(len(start_indices), len(end_indices))
            start_indices = start_indices[:min_length]
            end_indices = end_indices[:min_length]

        for start_idx, end_idx in zip(start_indices, end_indices):
            # Get sample indices
            start_sample = np.searchsorted(eeg_timestamps, marker_timestamps[start_idx])
            end_sample = np.searchsorted(eeg_timestamps, marker_timestamps[end_idx])

            # Apply baseline shift (subtract 0.5s worth of samples)
            baseline_start_sample = max(0, start_sample - int(sfreq * baseline_duration))  # Prevent negative index

            # Compute epoch duration
            tmax = (end_sample - start_sample) / sfreq  # Convert to seconds

            # Store the event (aligned with new baseline)
            events.append([baseline_start_sample, 0, int(start_marker)])  # Event now starts 0.5s before marker
            event_id_map[str(start_marker)] = int(start_marker)


    # Convert to MNE format and sort
    events = np.array(events)
    events = events[np.argsort(events[:, 0])]  # Sort by time index

    # Create MNE Epochs
    epochs = mne.Epochs(
        raw, events, event_id=event_id_map, tmin=-baseline_duration, tmax=tmax,
        baseline=(None, 0), detrend=1, preload=True
    )


    # Define Rejection Criteria (Artifact Removal)
    # Compute max per epoch
    max_per_epoch = np.max(np.abs(epochs.get_data()), axis=(1, 2))  

    # Compute z-scores
    z_scores = zscore(max_per_epoch)

    # Define a rejection criterion (e.g., 3 standard deviations)
    reject_z_threshold = 3.0  
    bad_epochs = np.where(np.abs(z_scores) > reject_z_threshold)[0]  

    # Drop the bad epochs
    epochs.drop(bad_epochs)
    print(f"Dropped {len(bad_epochs)} bad epochs based on z-score method.")

   # Slice Epochs into Smaller Training Windows (e.g., 0.5s)
    print(f"Segmenting epochs into {config.CLASSIFY_WINDOW}ms training windows...")
    segments, labels = segment_epochs(epochs, window_size=config.CLASSIFY_WINDOW, step_size=0.1)

    print(f"🔹 Segmented Data Shape: {segments.shape}")  # Debugging output

    # Compute Covariance Matrices (for Riemannian Classifier)
    print("Computing Covariance Matrices...")

    cov_matrices = []
    info = epochs.info  # Use the same info as the original epochs
    
    # Compute Covariance Matrices (for Riemannian Classifier)
    print("Computing Covariance Matrices...")
    cov_matrices = np.array([np.cov(segment) for segment in segments])

    '''
    for segment in segments:
        # Convert segment into an MNE EpochsArray (shape needs to be (n_epochs, n_channels, n_samples))
        segment = mne.EpochsArray(segment[np.newaxis, :, :], info)  # Ensure correct shape

        # Compute covariance matrix using OAS regularization
        cov = mne.compute_covariance(segment, method="oas")

        # Extract covariance matrix and store
        cov_matrices.append(cov["data"])
    '''
    # Convert list to numpy array (shape: (n_epochs, n_channels, n_channels))
    cov_matrices = np.array(cov_matrices)
    print(f"Computed {len(cov_matrices)} covariance matrices with shape: {cov_matrices.shape}")

    # Train Riemannian MDM Model
    #print(cov_matrices)
    print("Training Riemannian Classifier...")
    Reimans_model = train_riemannian_model(cov_matrices, labels)

    #  Save the trained model
    # Define model save path (subject-level, not session-specific)
    subject_model_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "models")
    os.makedirs(subject_model_dir, exist_ok=True)

    subject_model_path = os.path.join(subject_model_dir, f"sub-{config.TRAINING_SUBJECT}_model.pkl")

    # Save the trained model
    with open(subject_model_path, 'wb') as f:
        pickle.dump(Reimans_model, f)

    print(f"✅ Trained model saved at: {subject_model_path}")


if __name__ == "__main__":
    main()
