from scipy.signal import butter, filtfilt, iirnotch, lfilter
import numpy as np
import config
from sklearn.linear_model import LinearRegression



def apply_notch_filter(eeg_data, sampling_rate, line_freq=60, quality_factor=30, harmonics=2):
    """
    Apply a notch filter to the EEG dataset to remove line noise.

    Parameters:
        eeg_data (np.ndarray): EEG data of shape (n_samples, n_channels).
        sampling_rate (int): Sampling rate of the EEG data in Hz.
        line_freq (float): Line noise frequency to filter (default: 50 Hz).
        quality_factor (float): Quality factor of the notch filter (default: 30).
        harmonics (int): Number of harmonics to filter (default: 1, filters only the line frequency).

    Returns:
        np.ndarray: Filtered EEG data of the same shape as input.
    """
    filtered_data = eeg_data.copy()

    for harmonic in range(1, harmonics + 1):
        target_freq = line_freq * harmonic
        b, a = iirnotch(target_freq, quality_factor, sampling_rate)
        filtered_data = filtfilt(b, a, filtered_data, axis=0)

    return filtered_data


# Bandpass Filter Design Function
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to Apply Filter with State Tracking
def filter_with_state(data, b, a, zi):
    filtered_data, zf = lfilter(b, a, data, zi=zi)  # Apply filter with prior state
    return filtered_data, zf  # Return filtered signal & final state


def apply_car_filter(data):
    avg = np.mean(data, axis=0)
    return data - avg



def compute_grand_average(data, sampling_rate, mode="trials"):
    """
    Compute the grand average of EEG data within a specified time range.

    Parameters:
        data (np.ndarray): Input data of shape (n_samples, n_channels, n_trials).
        sampling_rate (int): Sampling rate of the EEG data in Hz.
        mode (str): Averaging mode. 
                    "trials" - Average across trials, retaining temporal resolution.
                    "trials_and_timepoints" - Average across trials and timepoints, one value per channel.

    Returns:
        np.ndarray: Grand average data.
                    Shape is (n_channels, n_window_samples) for "trials".
                    Shape is (n_channels,) for "trials_and_timepoints".
    """
    if mode == "trials":
        # Average across trials, retaining temporal resolution
        grand_average = np.mean(data, axis=2)  # Shape: (n_window_samples, n_channels)
        return grand_average  # Shape: (n_channels, n_window_samples)

    elif mode == "trials_and_timepoints":
        # Average across trials and timepoints
        grand_average = np.mean(data, axis=(0, 2))  # Shape: (n_channels,)
        return grand_average  # Shape: (n_channels,)

    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'trials' or 'trials_and_timepoints'.")



def extract_segments(eeg_data, eeg_timestamps, marker_timestamps, marker_values, window_size_ms, fs, 
                     offset_ms=0):
    """
    Extract EEG segments based on marker timestamps and a time window.

    Parameters:
        eeg_data (np.ndarray): 2D array of EEG data (samples x channels).
        eeg_timestamps (np.ndarray): 1D array of timestamps for EEG data.
        marker_timestamps (np.ndarray): 1D array of timestamps for markers.
        marker_values (np.ndarray): 1D array of marker values.
        window_size (int): Time window for each segment in milliseconds.
        fs (float): Sampling rate of the EEG data (Hz).
        offset_ms (int): Time offset after the marker in milliseconds (default: 0).
        pre_trigger_time_ms (int): Time in milliseconds to include before the marker timestamp (default: 0).

    Returns:
        np.ndarray: 3D array of segments with shape (time, trials, channels).
        np.ndarray: 1D array of labels corresponding to the trials.
    """
    # Convert time parameters from milliseconds to samples
    window_size_samples = int((window_size_ms / 1000) * fs)
    offset_samples = int((offset_ms / 1000) * fs)
    segments = []
    labels = []

    for marker_time, marker_value in zip(marker_timestamps, marker_values):
        if marker_value not in [100, 200]:  # Filter only markers of interest
            continue

        # Find the index of the marker in EEG timestamps
        closest_idx = np.searchsorted(eeg_timestamps, marker_time)
        start_idx = closest_idx + offset_samples
        end_idx = start_idx + window_size_samples

        # Ensure indices are within data bounds
        if start_idx < 0 or end_idx > len(eeg_data):
            print(f"Skipping marker at {marker_time:.2f}s: Out of bounds.")
            continue
        
        # Extract the segment
        segment = eeg_data[start_idx:end_idx, :config.CAP_TYPE]  # Shape: (time, channels)


        if segment.shape[0] == window_size_samples:  # Check correct size
            segments.append(segment)
            labels.append(marker_value)
        else:
            print(f"Skipping marker at {marker_time:.2f}s: Segment size mismatch.")

    # Convert to numpy arrays
    if segments:
        # Rearrange to (time, trials, channels)
        segments = np.stack(segments, axis=2)  # Stack trials on the third axis
    else:
        segments = np.empty((window_size_samples, eeg_data.shape[1], 0))  # Empty array if no segments

    labels = np.array(labels)

    return segments, labels



def flatten_segments(segments):
    """
    Flatten segmented EEG data from 3D to 2D.

    Parameters:
        segments (np.ndarray): Input data of shape (time, channels, trials).

    Returns:
        np.ndarray: Flattened data of shape (trials, time * channels).
    """
    # Ensure the input array has the correct dimensionality
    if segments.ndim != 3:
        raise ValueError("Input data must be a 3D array (time, channels, trials).")

    # Rearrange dimensions to (trials, time, channels)
    trials_first = np.transpose(segments, (2, 0, 1))  # Shape: (trials, time, channels)

    # Flatten the time and channels dimensions
    flattened = trials_first.reshape(trials_first.shape[0], -1)  # Shape: (trials, time * channels)

    return flattened


def flatten_single_segment(segment):
    """
    Flatten segmented EEG data from 2D to 1D. This is used for prediction during test

    Parameters:
        segments (np.ndarray): Input data of shape (time, channels).

    Returns:
        np.ndarray: Flattened data of shape (time * channels).
    """
    # Ensure the input array has the correct dimensionality
    if segment.ndim != 2:
        raise ValueError("Input data must be a 2D array (time, channels).")

    # Flatten the 2D array into 1D
    flattened = segment.flatten()  # Shape: (time * channels,)

    # Reshape into a row vector
    flattened_row = flattened.reshape(1, -1)  # Shape: (1, time * channels)

    return flattened_row



def extract_and_flatten_segment(eeg_data, start_time, fs, window_size_ms, offset_ms=0):
    """
    Extract and flatten a single EEG segment from raw data.

    Parameters:
        eeg_data (np.ndarray): 2D array of EEG data (samples x channels).
        start_time (float): Start time of the segment in seconds.
        fs (float): Sampling rate of the EEG data (Hz).
        window_size_ms (int): Time window for the segment in milliseconds.
        offset_ms (int): Time offset after the start_time in milliseconds (default: 0).

    Returns:
        np.ndarray: Flattened segment of shape (1, time * channels).
    """
    # Convert time parameters from milliseconds to samples
    window_size_samples = int((window_size_ms / 1000) * fs)
    offset_samples = int((offset_ms / 1000) * fs)

    # Calculate start and end indices
    start_idx = int(start_time * fs) + offset_samples
    end_idx = start_idx + window_size_samples

    # Ensure indices are within data bounds
    if start_idx < 0 or end_idx > eeg_data.shape[0]:
        raise ValueError(f"Segment indices out of bounds: start_idx={start_idx}, end_idx={end_idx}")

    # Extract the segment
    segment = eeg_data[start_idx:end_idx, :]

    # Ensure the segment has the correct size
    if segment.shape[0] != window_size_samples:
        raise ValueError(f"Segment size mismatch: expected {window_size_samples} samples, got {segment.shape[0]}")

    # Flatten the segment into a row vector
    flattened_segment = segment.flatten().reshape(1, -1)  # Shape: (1, time * channels)

    return flattened_segment



def separate_classes(segments, labels, class_1=100, class_2=200):
    """
    Separate EEG segments and labels into two classes.

    Parameters:
        segments (np.ndarray): Input data of shape (time, channels, trials).
        labels (np.ndarray): Corresponding labels for the trials.
        class_1 (int): Marker value for the first class (default: 100).
        class_2 (int): Marker value for the second class (default: 200).

    Returns:
        tuple: Two dictionaries, each containing 'data' (segments) and 'labels' for the two classes.
    """
    # Ensure the labels match the trials in the segments
    if segments.shape[2] != len(labels):
        raise ValueError("Mismatch between number of trials in segments and labels.")

    # Boolean masks for class separation
    mask_class_1 = labels == class_1
    mask_class_2 = labels == class_2

    # Separate the segments and labels
    class_1_data = segments[:, :, mask_class_1]
    class_1_labels = labels[mask_class_1]

    class_2_data = segments[:, :, mask_class_2]
    class_2_labels = labels[mask_class_2]

    # Return as dictionaries for easy access
    return (
        {"data": class_1_data, "labels": class_1_labels},
        {"data": class_2_data, "labels": class_2_labels},
    )




def remove_eog_artifacts(eeg_data, eog_data):
    """
    Remove EOG artifacts from EEG data using a regression-based approach.

    Parameters:
        eeg_data (np.ndarray): 2D array of EEG data (samples x channels).
        eog_data (np.ndarray): 2D array of EOG data (samples x EOG_channels).

    Returns:
        np.ndarray: Cleaned EEG data with EOG artifacts regressed out.
    """
    # Ensure EOG data is 2D
    if eog_data.ndim == 1:
        eog_data = eog_data[:, np.newaxis]  # Convert to (samples x 1)

    # Initialize a cleaned EEG array
    eeg_cleaned = np.zeros_like(eeg_data)

    # Perform regression for each EEG channel
    for channel_idx in range(eeg_data.shape[1]):
        # Extract the current EEG channel
        eeg_channel = eeg_data[:, channel_idx]

        # Fit a linear regression model
        regressor = LinearRegression()
        regressor.fit(eog_data, eeg_channel)

        # Predict the EOG contribution to the EEG channel
        eog_prediction = regressor.predict(eog_data)

        # Subtract the EOG contribution to clean the EEG channel
        eeg_cleaned[:, channel_idx] = eeg_channel - eog_prediction

    return eeg_cleaned


def parse_eeg_and_eog(eeg_stream, channel_names):
    """
    Parse EEG and EOG data from the given EEG stream based on configuration.

    Parameters:
        eeg_stream (dict): EEG stream containing 'time_series' and 'time_stamps'.
        channel_names (list): List of channel names corresponding to the data columns.

    Returns:
        tuple:
            - np.ndarray: EEG data from specified channels.
            - np.ndarray or None: EOG data from specified channels (if EOG_TOGGLE is enabled).
    """

    # Extract EEG data from the stream
    eeg_data = np.array(eeg_stream['time_series'])  # Shape: (N_samples, N_channels)

    # Handle "ALL" keyword for EEG channels
    if config.EEG_CHANNEL_NAMES == ['ALL']:
        eeg_selected = eeg_data[:, :config.CAP_TYPE]  # Assume the first 32 channels are EEG (value stored in config.CAP_TYPE)
    else:
        # Identify indices for EEG channels based on configuration
        eeg_indices = [channel_names.index(ch) for ch in config.EEG_CHANNEL_NAMES if ch in channel_names]
        if not eeg_indices:
            raise ValueError("No matching EEG channels found in the provided stream.")

        # Extract EEG data
        eeg_selected = eeg_data[:, eeg_indices]

    # Handle EOG data based on toggle
    if config.EOG_TOGGLE:
        # Identify indices for EOG channels based on configuration
        eog_indices = [channel_names.index(ch) for ch in config.EOG_CHANNEL_NAMES if ch in channel_names]
        if not eog_indices:
            print("Warning: No matching EOG channels found in the provided stream.")
            eog_selected = None
        else:
            # Extract EOG data
            eog_selected = eeg_data[:, eog_indices]
    else:
        eog_selected = None

    return eeg_selected, eog_selected


def extract_psd_features(eeg_data, fs=512, bands=[(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)]):
    """
    Extract power spectral density (PSD) features for predefined frequency bands.

    Parameters:
        eeg_data (np.ndarray): 3D array of EEG data (n_samples x window_samples x n_channels).
        fs (int): Sampling frequency of the EEG data in Hz.
        bands (list): List of tuples defining frequency bands (low_freq, high_freq).

    Returns:
        np.ndarray: Feature matrix of shape (n_samples, n_bands * n_channels).
    """
    n_samples, window_samples, n_channels = eeg_data.shape
    psd_features = []

    for sample_idx in range(n_samples):
        sample_features = []
        for band in bands:
            band_power = []
            for channel in range(n_channels):
                # Compute PSD using Welch's method
                freqs, psd = welch(eeg_data[sample_idx, :, channel], fs=fs, nperseg=128)
                
                # Integrate power within the frequency band
                band_power.append(np.sum(psd[(freqs >= band[0]) & (freqs <= band[1])]))
            
            sample_features.extend(band_power)
        psd_features.append(sample_features)
    
    return np.array(psd_features)