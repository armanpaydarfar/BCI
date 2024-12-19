import pyxdf
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pickle

# Butterworth bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

# Common Average Reference (CAR)
def apply_car_filter(data):
    avg = np.mean(data, axis=0)
    return data - avg

# Load XDF data
def load_xdf(file_path):
    streams, _ = pyxdf.load_xdf(file_path)
    eeg_stream = next((s for s in streams if s['info']['type'][0] == 'EEG'), None)
    marker_stream = next((s for s in streams if s['info']['type'][0] == 'Markers'), None)

    if eeg_stream is None or marker_stream is None:
        raise ValueError("Both EEG and Marker streams must be present in the XDF file.")

    print("EEG and Marker streams successfully loaded.")
    return eeg_stream, marker_stream

# Extract segments of EEG data based on markers
def extract_segments(eeg_data, eeg_timestamps, marker_timestamps, marker_values, window_size, fs):
    segments = []
    labels = []
    samples_per_window = int(window_size * fs)

    for marker_time, marker_value in zip(marker_timestamps, marker_values):
        if marker_value not in [100, 200]:
            continue

        # Find the start index for the marker
        start_idx = np.searchsorted(eeg_timestamps, marker_time)
        end_idx = start_idx + samples_per_window

        if end_idx < len(eeg_data):
            segment = eeg_data[start_idx:end_idx]
            segments.append(segment)
            labels.append(marker_value)

    return np.array(segments), np.array(labels)

def train_model(segments, labels, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=False)  # No randomization to preserve time structure
    lda = LDA()
    accuracies = []

    print("Starting K-Fold Cross Validation...")
    for train_index, test_index in kf.split(segments):
        X_train, X_test = segments[train_index], segments[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Flatten each segment for the LDA input
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        lda.fit(X_train_flat, y_train)
        y_pred = lda.predict(X_test_flat)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f"Fold Accuracy: {accuracy:.4f}")

    print(f"Average Accuracy: {np.mean(accuracies):.4f}")

    # Chance level accuracy
    chance_accuracy = max(np.bincount(labels)) / len(labels)
    print(f"Chance Level Accuracy: {chance_accuracy:.4f}")

    # Retrain the model on all data
    print("Retraining model on all data...")
    X_all = segments.reshape(segments.shape[0], -1)
    lda.fit(X_all, labels)

    return lda

def main():
    # Define parameters
    file_path = '/home/arman-admin/Documents/CurrentStudy/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf'
    lowcut = 0.1  # Hz
    highcut = 30  # Hz
    fs = 512  # Sampling frequency
    window_size = 1  # seconds

    # Load data
    eeg_stream, marker_stream = load_xdf(file_path)

    # Extract EEG and marker data
    eeg_data = np.array(eeg_stream['time_series'])
    eeg_timestamps = np.array(eeg_stream['time_stamps'])
    marker_timestamps = np.array(marker_stream['time_stamps'])
    marker_values = np.array([int(m[0]) for m in marker_stream['time_series']])

    # Apply pre-processing
    print("Applying Butterworth bandpass filter...")
    eeg_data = butter_bandpass_filter(eeg_data, lowcut, highcut, fs)

    print("Applying Common Average Reference (CAR) filter...")
    eeg_data = apply_car_filter(eeg_data)

    # Extract segments based on markers
    print("Extracting EEG segments based on markers...")
    segments, labels = extract_segments(eeg_data, eeg_timestamps, marker_timestamps, marker_values, window_size, fs)

    # Print segment distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Segment distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"Class {label}: {count} segments")

    # Train LDA model
    lda_model = train_model(segments, labels, n_splits=5)

    # Save the trained model
    model_path = 'lda_eeg_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(lda_model, f)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
