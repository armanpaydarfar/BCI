import numpy as np
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from Utils.stream_utils import load_xdf
import config
from Utils.stream_utils import get_channel_names_from_xdf
import os

import mne
from mne import concatenate_epochs
from scipy.signal import welch, resample, butter
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

def run_cca_spatial_filter(epochs, labels, n_components=3):
    """
    Computes a CCA-based spatial filter from the epochs.

    Parameters:
      epochs       : an MNE Epochs object (n_trials, n_channels, n_times)
      labels       : a 1D array of labels for each trial
      n_components : number of CCA components to compute

    Returns:
      spatial_filter: numpy array with shape (n_channels, n_components)
      cca_model     : the fitted CCA model
    """
    all_data = epochs.get_data()
    unique_labels = np.unique(labels)
    
    concat_data = []
    concat_avg  = []
    
    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        data_class = all_data[idx, ...]  # (n_trials_class, n_channels, n_times)
        n_trials, n_channels, n_times = data_class.shape
        data_class = np.transpose(data_class, (1, 2, 0))  # (n_channels, n_times, n_trials)
        avg_data = np.mean(data_class, axis=2)  # (n_channels, n_times)
        data_class_reshaped = data_class.reshape(n_channels, n_times * n_trials)
        avg_data_repeated = np.tile(avg_data, (1, n_trials))
        concat_data.append(data_class_reshaped)
        concat_avg.append(avg_data_repeated)
    
    concat_data = np.concatenate(concat_data, axis=1)  # (n_channels, total_samples)
    concat_avg  = np.concatenate(concat_avg, axis=1)      # (n_channels, total_samples)
    X = concat_data.T
    Y = concat_avg.T
    
    cca = CCA(n_components=n_components)
    cca.fit(X, Y)
    spatial_filter = cca.x_weights_
    
    return spatial_filter, cca

def apply_spatial_filter(epochs, spatial_filter):
    """
    Applies the spatial filter to the epochs.
    
    Parameters:
      epochs         : an MNE Epochs object (n_trials, n_channels, n_times)
      spatial_filter : numpy array with shape (n_channels, n_components)
    
    Returns:
      projected_data : numpy array of shape (n_trials, n_components, n_times)
    """
    data = epochs.get_data()  # (n_trials, n_channels, n_times)
    projected_data = np.array([trial.T @ spatial_filter for trial in data])
    projected_data = np.transpose(projected_data, (0, 2, 1))
    return projected_data

def compute_psd_pwelch(epochs, fs, window, nfft, noverlap, freq_range):
    """
    Compute power spectral density (PSD) for each trial and channel using Welch's method.
    
    Parameters:
      epochs     : numpy array of shape (n_trials, n_channels, n_times)
      fs         : sampling frequency (Hz)
      window     : 1D array containing the window to apply (e.g., Hanning window)
      nfft       : FFT length
      noverlap   : Number of overlapping samples
      freq_range : Array of frequencies (Hz) at which to extract the PSD estimates

    Returns:
      psd_array  : numpy array of shape (n_trials, n_channels, len(freq_range))
    """
    n_trials, n_channels, _ = epochs.shape
    psd_array = np.zeros((n_trials, n_channels, len(freq_range)))
    for t in range(n_trials):
        for ch in range(n_channels):
            # Compute Welch's PSD
            f, pxx = welch(epochs[t, ch, :], fs=fs, window=window, nfft=nfft, noverlap=noverlap)
            # For each target frequency, find the closest frequency index
            indices = [np.argmin(np.abs(f - freq)) for freq in freq_range]
            psd_array[t, ch, :] = pxx[indices]
    return psd_array

def main():
    """
    Main function to generate an LDA decoder from EEG data.
    """

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

    eeg_stream, marker_stream = load_xdf(xdf_files[0])
    # Extract EEG and marker data
    eeg_data = np.array(eeg_stream['time_series']).T
    eeg_timestamps = np.array(eeg_stream['time_stamps'])
    marker_timestamps = np.array(marker_stream['time_stamps'])
    marker_values = np.array([int(m[0]) for m in marker_stream['time_series']])
    channel_names = get_channel_names_from_xdf(eeg_stream)
    montage = mne.channels.make_standard_montage("standard_1020")
    rename_dict = {
        "FZ": "Fz", "CZ": "Cz"
    }
    non_eeg_channels = {"AUX1", "AUX2", "AUX3", "AUX7", "AUX8", "AUX9", "TRIGGER", }
    valid_eeg_channels = [ch for ch in channel_names if ch not in non_eeg_channels]
    valid_indices = [channel_names.index(ch) for ch in valid_eeg_channels]  
    eeg_data = eeg_data[valid_indices, :]  


    sfreq = config.FS 
    info = mne.create_info(ch_names=valid_eeg_channels, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(eeg_data, info)
    errp_channels = ["FZ", "CZ", "FC1", "FC2", "F3", "F4", "FC5", "FC6"]
    channels_to_keep = [ch for ch in raw.ch_names if ch in errp_channels]
    channels_to_drop = [ch for ch in raw.ch_names if ch not in errp_channels]
    if not channels_to_keep:
        print("Warning: No ErrP-relevant channels found. Check your channel names or update the errp_channels list.")
    else:
        raw.drop_channels(channels_to_drop)
        print("Retained ErrP-relevant channels:", channels_to_keep)
        print("Dropped channels not relevant to ErrP analysis:", channels_to_drop)
        
    first_channel_unit = raw.info["chs"][0]["unit"]
    print(f"First Channel Unit (FIFF Code): {first_channel_unit}")

    # Convert from volts to microvolts (data now in SI units)
    raw._data /= 1e3  
    for ch in raw.info['chs']:
        ch['unit'] = 201  # FIFF_UNIT_V

    print(f"Updated Units for EEG Channels: {[ch['unit'] for ch in raw.info['chs']]}")

    if "M1" in raw.ch_names and "M2" in raw.ch_names:
        raw.drop_channels(["M1", "M2"])
        print("Removed Mastoid Channels: M1, M2")
    else:
        print("No Mastoid Channels Found in Data")

    raw.rename_channels(rename_dict)
    missing_in_montage = set(raw.ch_names) - set(montage.ch_names)
    print(f"Channels in Raw but Missing in Montage: {missing_in_montage}")

    raw.set_montage(montage, match_case=True, on_missing="warn")
    highband = 10
    lowband = 1
    raw.notch_filter(60, method="iir")  
    raw.filter(l_freq=lowband, h_freq=highband, fir_design="firwin")  
    print("\n Final EEG Channels After Processing:", raw.ch_names)

    # Pre-process Data
    unique_markers = np.unique(marker_values)
    event_dict = {str(marker): marker for marker in unique_markers}

    events = np.column_stack((
    np.searchsorted(eeg_timestamps, marker_timestamps),  
    np.zeros(len(marker_values), dtype=int),
    marker_values
    ))

    marker_labels = {
    "100": "Rest",
    "200": "Right Arm MI",
    "300": "Robot Move",
    "340": "Robot Early Stop"
    }

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_dict,
        tmin=-1,
        tmax=5,
        baseline=None,
        detrend=1,
        preload=True
    )

    for marker in ["100", "200", "300", "340"]:
        if marker in epochs.event_id:
            print(f"Marker {marker}: {len(epochs[marker])} epochs")

    conditions = {
        "Rest": "100",
        "MI": "200",
        "Error": "340"
    }

    conditions_epochs = {}
    mi_events = events[events[:, 2] == 200]
    errp_events = events[events[:, 2] == 340]

    paired_mi_set = set()
    for err in errp_events:
        candidate_indices = np.where(mi_events[:, 0] < err[0])[0]
        if candidate_indices.size > 0:
            candidate = mi_events[candidate_indices[-1], 0]
            paired_mi_set.add(candidate)

    for cond_name, marker_str in conditions.items():
        if marker_str in epochs.event_id:
            ep_cond = epochs[marker_str].copy()
            if marker_str == "200":
                pure_indices = [i for i, ev in enumerate(ep_cond.events) if ev[0] not in paired_mi_set]
                ep_cond = ep_cond[pure_indices].copy()
                print(f"Selected {len(ep_cond)} pure MI trials (marker 200) after excluding error trials.")
            else:
                print(f"Found {len(ep_cond)} trials for condition {cond_name} (marker {marker_str}).")
            
            if cond_name in ['Rest', 'MI']:
                baseline_indices = ep_cond.time_as_index([2.5, 3])
                idx_start, idx_end = baseline_indices
                baseline_mean = np.mean(ep_cond._data[:, :, idx_start:idx_end], axis=2, keepdims=True)
                ep_baselined = ep_cond.copy()
                ep_baselined._data -= baseline_mean
                ep_crop = ep_baselined.crop(tmin=3.2, tmax=3.8)
            else:
                baseline_indices = ep_cond.time_as_index([-0.5, 0])
                idx_start, idx_end = baseline_indices
                baseline_mean = np.mean(ep_cond._data[:, :, idx_start:idx_end], axis=2, keepdims=True)
                ep_baselined = ep_cond.copy()
                ep_baselined._data -= baseline_mean
                ep_crop = ep_baselined.crop(tmin=0.2, tmax=0.8)
            
            conditions_epochs[cond_name] = ep_crop
        else:
            print(f"⚠️ Condition {cond_name} (marker {marker_str}) not found.")
            conditions_epochs[cond_name] = None

    mi_epochs = conditions_epochs.get("Rest")
    error_epochs = conditions_epochs.get("Error")

    if mi_epochs is None or error_epochs is None:
        raise ValueError("Both MI and Error epochs must be available for CCA.")

    mi_epochs_shifted = mi_epochs.copy().shift_time(-3.0, relative=True)
    error_epochs_shifted = error_epochs.copy()
    combined_epochs = concatenate_epochs([mi_epochs_shifted, error_epochs_shifted])
    print(f"Combined epochs: {len(combined_epochs)} trials.")

    n_mi = len(mi_epochs)
    n_error = len(error_epochs)
    combined_labels = np.concatenate([np.zeros(n_mi), np.ones(n_error)])
    print("Label vector shape:", combined_labels.shape)

    n_components = 3
    spatial_filter, cca_model = run_cca_spatial_filter(combined_epochs, combined_labels, n_components=n_components)
    print("Spatial filter shape:", spatial_filter.shape)

    projected_combined = apply_spatial_filter(combined_epochs, spatial_filter)
    print("Projected combined data shape:", projected_combined.shape)

    resample_factor = int(raw.info['sfreq'] / 32)
    n_trials, n_components, n_times = projected_combined.shape
    n_downsampled = int(n_times / resample_factor)
    resampled_data = resample(projected_combined, num=n_downsampled, axis=-1)
    time_features = resampled_data.reshape(n_trials, -1)
    fs = config.FS  # e.g., 512 Hz
    n_times_original = projected_combined.shape[-1]  # number of samples in the cropped window (0.2-0.8s)
    psd_window = np.hanning(n_times_original)
    nfft = 4 * fs
    noverlap = int(len(psd_window) / 2)
    freq_range = np.array([4, 6, 8, 10])
    psd_array = compute_psd_pwelch(projected_combined, fs, psd_window, nfft, noverlap, freq_range)
    psd_features = psd_array.reshape(n_trials, -1)

    features = np.concatenate([time_features, psd_features], axis=1)
    print("Feature matrix shape (trials x features):", features.shape)

    scaler = MinMaxScaler()
    features_norm = scaler.fit_transform(features)

    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=42)
    temp_lda = LDA(solver='lsqr', shrinkage='auto')
    cv_scores = cross_val_score(temp_lda, features_norm, combined_labels, cv=cv, scoring='accuracy')
    print("Cross-Validation Accuracy Scores:", cv_scores)
    print("Mean Cross-Validation Accuracy: {:.2f}%".format(np.mean(cv_scores) * 100))

    lda = LDA(solver='lsqr', shrinkage='auto')
    lda.fit(features_norm, combined_labels)
    print("Final LDA classifier trained.")
    
    #??? WHY BUTTER HERE
    order = 2
    lowcut = 1
    highcut = 10
    b, a = butter(order, [lowcut / (fs/2), highcut / (fs/2)], btype='bandpass')

    decoder = {
        'spectralFilter': {'b': b, 'a': a},
        'spatialFilter': spatial_filter,
        'resample': {'resample_factor': resample_factor},
        'psd': {
            'window': psd_window,
            'nfft': nfft,
            'noverlap': noverlap,
            'freq_range': freq_range
        },
        'scaler': scaler,  # Normalization component (MinMaxScaler)
        'classifier': lda  # Trained LDA classifier; use lda.predict_proba() or lda.predict() on new data
    }

    print("Decoder created with the following components:")
    for key in decoder:
        print(" -", key)

    
        #  Save the trained model
    # Define model save path (subject-level, not session-specific)
    subject_model_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "models")
    os.makedirs(subject_model_dir, exist_ok=True)

    subject_model_path = os.path.join(subject_model_dir, f"sub-{config.TRAINING_SUBJECT}_model_Errp.pkl")

    
    # Save the trained model
    with open(subject_model_path, 'wb') as f:
        pickle.dump(decoder, f)

    print(f"✅ Trained model saved at: {subject_model_path}")


if __name__ == "__main__":
    main()
