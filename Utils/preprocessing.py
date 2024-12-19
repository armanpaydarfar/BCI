from scipy.signal import butter, filtfilt
import numpy as np

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def apply_car_filter(data):
    avg = np.mean(data, axis=0)
    return data - avg

def extract_segments(eeg_data, eeg_timestamps, marker_timestamps, marker_values, window_size, fs):
    segments = []
    labels = []
    samples_per_window = int(window_size * fs)

    for marker_time, marker_value in zip(marker_timestamps, marker_values):
        if marker_value not in [100, 200]:
            continue

        start_idx = np.searchsorted(eeg_timestamps, marker_time)
        end_idx = start_idx + samples_per_window

        if end_idx < len(eeg_data):
            segment = eeg_data[start_idx:end_idx]
            segments.append(segment)
            labels.append(marker_value)

    return np.array(segments), np.array(labels)