from scipy.signal import butter, filtfilt, iirnotch
import numpy as np
import config

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


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

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
        raise ValueError("Input data must be a 3D array (time, channels).")

    # Flatten the 2D array into 1D
    flattened = segment.flatten()  # Shape: (time * channels,)

    # Reshape into a row vector
    flattened_row = flattened.reshape(1, -1)  # Shape: (1, time * channels)

    return flattened_row


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
