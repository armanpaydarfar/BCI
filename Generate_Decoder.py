import numpy as np
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from Utils.preprocessing import butter_bandpass_filter, apply_car_filter
from Utils.stream_utils import load_xdf, extract_segments
import config

def train_model(segments, labels, n_splits=5):
    """
    Train an LDA model with k-fold cross-validation and return the trained model.

    Parameters:
        segments (np.ndarray): EEG data segments.
        labels (np.ndarray): Corresponding labels for the segments.
        n_splits (int): Number of splits for cross-validation.

    Returns:
        lda: Trained LDA model.
    """
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
    """
    Main function to generate an LDA decoder from EEG data.
    """
    # Load data
    print("Loading XDF data...")
    eeg_stream, marker_stream = load_xdf(config.DATA_FILE_PATH)

    # Extract EEG and marker data
    eeg_data = np.array(eeg_stream['time_series'])
    eeg_timestamps = np.array(eeg_stream['time_stamps'])
    marker_timestamps = np.array(marker_stream['time_stamps'])
    marker_values = np.array([int(m[0]) for m in marker_stream['time_series']])

    # Apply pre-processing
    print("Applying Butterworth bandpass filter...")
    eeg_data = butter_bandpass_filter(eeg_data, config.LOWCUT, config.HIGHCUT, config.FS)

    print("Applying Common Average Reference (CAR) filter...")
    eeg_data = apply_car_filter(eeg_data)

    # Extract segments based on markers
    print("Extracting EEG segments based on markers...")
    segments, labels = extract_segments(
        eeg_data, eeg_timestamps, marker_timestamps, marker_values, config.WINDOW_SIZE, config.FS
    )

    # Print segment distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Segment distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"Class {label}: {count} segments")

    # Train LDA model
    lda_model = train_model(segments, labels, n_splits=config.N_SPLITS)

    # Save the trained model
    with open(config.MODEL_PATH, 'wb') as f:
        pickle.dump(lda_model, f)
    print(f"Model saved to {config.MODEL_PATH}")

if __name__ == "__main__":
    main()
