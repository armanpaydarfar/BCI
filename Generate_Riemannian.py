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
    renamed_probs = {label_map[label]: probs for label, probs in posterior_probs.items()}

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

    if config.LEDOITWOLF:
        # Compute covariance matrices with optimized shrinkage
        cov_matrices_shrinked = np.array([LedoitWolf().fit(cov).covariance_ for cov in cov_matrices])
        cov_matrices = cov_matrices_shrinked    
    else:
        shrinkage = Shrinkage(shrinkage=shrinkage_param)
        cov_matrices = shrinkage.fit_transform(cov_matrices)  

    
    # Apply Riemannian whitening
    if config.RECENTERING:
        whitener = Whitening(metric="riemann")  # Use Riemannian mean for whitening
        cov_matrices = whitener.fit_transform(cov_matrices)
    
    #print(mean_riemann(cov_matrices))
    
    
    # Initialize cross-validation
    kf = KFold(n_splits=n_splits, shuffle=False)
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
    #print(events)

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
    segments, labels = segment_epochs(epochs, window_size=config.CLASSIFY_WINDOW, step_size=1/16)

    print(f"🔹 Segmented Data Shape: {segments.shape}")  # Debugging output

    # Compute Covariance Matrices (for Riemannian Classifier)
    print("Computing Covariance Matrices...")

    cov_matrices = []
    info = epochs.info  # Use the same info as the original epochs
    
    # Compute Covariance Matrices (for Riemannian Classifier)
    print("Computing Covariance Matrices...")
    #cov_matrices = np.array([np.cov(segment) for segment in segments])
    cov_matrices = np.array([ (segment @ segment.T) / np.trace(segment @ segment.T) for segment in segments ])

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
