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
from Utils.preprocessing import butter_bandpass, concatenate_streams, select_motor_channels
import glob  # Required for multi-file loading
from scipy.stats import zscore

from pyriemann.transfer import TLCenter
from pyriemann.classification import MDM
from pyriemann.utils.mean import mean_riemann
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

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



def train_riemannian_model(cov_matrices, labels, subject_ids="PILOT007", n_splits=8, shrinkage_param=config.SHRINKAGE_PARAM):
    """
    Train an MDM classifier using Riemannian Centering Transformation (RCT) for transfer learning,
    while collecting posterior probabilities and plotting their distributions.

    Parameters:
        cov_matrices (np.ndarray): Covariance matrices (n_trials, n_channels, n_channels).
        labels (np.ndarray): Corresponding labels for the segments.
        subject_ids (str or np.ndarray): Subject identifiers corresponding to each trial.
        n_splits (int): Number of splits for cross-validation.
        shrinkage_param (float): Regularization strength for Shrinkage.

    Returns:
        dict: Trained models and applied transformations.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mdm = MDM()

    accuracies = []
    posterior_probs = {label: [] for label in np.unique(labels)}  # Store probabilities for each class

    print("\n Starting K-Fold Cross Validation with RCT...\n")

    # **Step 1: Format subject_ids correctly as "Subject/Class"**
    if isinstance(subject_ids, str):  
        subject_ids = np.array([f"{subject_ids}/Class{label}" for label in labels])  # Ensure correct format

    print(f"🔹 Unique domain labels: {np.unique(subject_ids)}")

    # **Step 2: Apply Shrinkage Regularization**
    shrinkage = Shrinkage(shrinkage=shrinkage_param)
    cov_matrices = shrinkage.fit_transform(cov_matrices)

    # **Step 3: K-Fold Training and Evaluation**
    for fold_idx, (train_index, test_index) in enumerate(kf.split(cov_matrices), start=1):
        X_train, X_test = cov_matrices[train_index], cov_matrices[test_index]
        Y_train, Y_test = labels[train_index], labels[test_index]
        subjects_train, subjects_test = subject_ids[train_index], subject_ids[test_index]

        # **Step 4: Extract Subject Name Only for `target_domain`**
        target_domain = np.unique([subj.split("/")[0] for subj in subjects_test])[0]  # Extract "PILOT007" only

        print(f"🔹 Fold {fold_idx}: Target domain = {target_domain}")

        # **Step 5: Apply Re-Centering Transformation (RCT) on the TRAINING SET ONLY**
        rct = TLCenter(target_domain=target_domain)
        X_train_centered = rct.fit_transform(X_train, subjects_train)

        # **Step 6: Apply the same RCT transformation to the test set**
        X_test_centered = rct.transform(X_test)
        print(X_test_centered[0])
        # **Step 7: Train and Evaluate the Classifier**
        mdm.fit(X_train_centered, Y_train)
        Y_pred = mdm.predict(X_test_centered)
        Y_predProb = mdm.predict_proba(X_test_centered)  # Get posterior probabilities

        accuracy = accuracy_score(Y_test, Y_pred)
        accuracies.append(accuracy)

        print(f"\n **Fold {fold_idx} Accuracy: {accuracy:.4f}**")

        # **Step 8: Store Posterior Probabilities for Each True Class**
        for idx, true_label in enumerate(Y_test):
            posterior_probs[true_label].append(Y_predProb[idx, np.where(mdm.classes_ == true_label)[0][0]])

    # **Step 9: Print Final Accuracy**
    print(f"\n🚀 **Final Average Accuracy with RCT:** {np.mean(accuracies):.4f}")

    # **Step 10: Retrain MDM on the Entire Dataset Using RCT**
    target_domain = np.unique([subj.split("/")[0] for subj in subject_ids])[0]  # Pick a reference subject for final re-centering
    rct_final = TLCenter(target_domain=target_domain)
    cov_matrices_centered = rct_final.fit_transform(cov_matrices, subject_ids)
    
    mdm.fit(cov_matrices_centered, labels)

    # **Step 11: Plot Posterior Probability Distributions**
    plot_posterior_probabilities(posterior_probs)

    return {
        "classifier": mdm,
        "rct_transform": rct_final  # Save TLCenter transformation for online sessions
    }

def plot_posterior_probabilities(posterior_probs):
    """
    Plots the histogram of posterior probabilities for each class.

    Parameters:
        posterior_probs (dict): Dictionary containing posterior probabilities for each class.
    """
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 20)  # Set bins for histogram

    for label, probs in posterior_probs.items():
        probs = np.array(probs).flatten()
        sns.histplot(probs, bins=bins, alpha=0.6, label=f"Class {label}", kde=True)

    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Posterior Probability Distribution Across Classes")
    plt.legend(title="True Class/Domain")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

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
    #b, a = butter_bandpass(config.LOWCUT, config.HIGHCUT, sfreq, order=4)
    #raw = apply_stateful_filter(raw, b, a)

    raw.filter(l_freq=config.LOWCUT, h_freq=config.HIGHCUT, method="iir")  # Bandpass filter (8-16Hz)

    # Apply Common Average Reference (CAR)
    #raw.set_eeg_reference("average")

    # Apply Surface Laplacian (CSD) if enabled

    if config.SURFACE_LAPLACIAN_TOGGLE:
        raw = mne.preprocessing.compute_current_source_density(raw)

    '''
    scaler = StandardScaler()
    raw._data = scaler.fit_transform(raw.get_data().T).T  # Transpose before & after to preserve (n_channels, n_samples)
    training_mean = scaler.mean_
    training_std = scaler.scale_
    print(f" training_mean shape: {training_mean.shape}")
    print(f" training_mean: {training_mean}")
    print(f" training_std shape: {training_std.shape}")
    print(f" normed data shape: {raw._data.shape}")
    '''
    if config.SELECT_MOTOR_CHANNELS:
        raw = select_motor_channels(raw)
   
    # Print remaining channels to confirm
    print("Remaining channels:", raw.info["ch_names"])
    # Configurable baseline duration
    BASELINE_START = -1.0  # Start of baseline period (relative to event)
    BASELINE_END = 0  # End of baseline period (relative to event)
    TRIAL_START = 1.0  # Start of MI data (relative to event)
    TRIAL_END = 5.0  # End of MI data (relative to event)

    events = []
    event_id_map = {}

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

            # Compute baseline start sample (ensuring no negative index)
            baseline_start_sample = max(0, start_sample + int(sfreq * BASELINE_START))
            baseline_end_sample = start_sample + int(sfreq * BASELINE_END)

            # Compute baseline mean (per channel)
            baseline_mean = raw._data[:, baseline_start_sample:baseline_end_sample].mean(axis=1, keepdims=True)
            # Apply baseline correction manually
            raw._data -= baseline_mean  # Subtract baseline mean

            # Store the event (aligned with new baseline)
            events.append([start_sample, 0, int(start_marker)])  # Align events with actual trial start
            event_id_map[str(start_marker)] = int(start_marker)

    # Convert to MNE format and sort
    events = np.array(events)
    events = events[np.argsort(events[:, 0])]  # Sort by time index

    # Create MNE Epochs (without baseline correction)
    epochs = mne.Epochs(
        raw, 
        events, 
        event_id=event_id_map, 
        tmin=TRIAL_START,  # Only take seconds 1-4
        tmax=TRIAL_END,  
        baseline=None,  # Baseline correction was already applied manually
        detrend=1, 
        preload=True
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
    #cov_matrices = np.array([np.cov(segment) for segment in segments])
    cov_matrices = np.array([ (segment @ segment.T) / np.trace(segment @ segment.T) for segment in segments ])


    # Convert list to numpy array (shape: (n_epochs, n_channels, n_channels))
    cov_matrices = np.array(cov_matrices)
    #print(cov_matrices[0])
    print(f"Computed {len(cov_matrices)} covariance matrices with shape: {cov_matrices.shape}")
    #print(f" Sample cov matrix: {cov_matrices[0]}")

    # Train Riemannian MDM Model
    #print(cov_matrices)
    print("Training Riemannian Classifier...")
    Reimans_model = train_riemannian_model(cov_matrices, labels)

    #  Save the trained model
    # Define model save path (subject-level, not session-specific)
    subject_model_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "models")
    os.makedirs(subject_model_dir, exist_ok=True)

    subject_model_path = os.path.join(subject_model_dir, f"sub-{config.TRAINING_SUBJECT}_model.pkl")
    Training_mean_path = os.path.join(subject_model_dir, f"sub-{config.TRAINING_SUBJECT}_mean")
    Training_std_path = os.path.join(subject_model_dir, f"sub-{config.TRAINING_SUBJECT}_std")
    
    # Save the trained model
    with open(subject_model_path, 'wb') as f:
        pickle.dump(Reimans_model, f)

    print(f"✅ Trained model saved at: {subject_model_path}")
    #np.save(Training_mean_path, training_mean)
    #np.save(Training_std_path, training_std)
    #print(" Saved precomputed training mean and std for real-time use.")


if __name__ == "__main__":
    main()
