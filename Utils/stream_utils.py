from pylsl import resolve_stream, StreamInlet
import numpy as np
import pyxdf

def check_streams():
    print("Checking for EEG stream...")
    eeg_streams = resolve_stream('type', 'EEG')
    if not eeg_streams:
        print("Error: EEG stream not found.")
        exit(1)
    print("EEG stream is active.")
    return StreamInlet(eeg_streams[0])

def get_eeg_data(inlet, duration=1.0, sampling_rate=512):
    inlet.flush()
    samples = int(duration * sampling_rate)
    data, _ = inlet.pull_chunk(timeout=duration + 0.5, max_samples=samples)
    return np.array(data[-samples:])

def load_xdf(file_path):
    streams, _ = pyxdf.load_xdf(file_path)
    eeg_stream = next((s for s in streams if s['info']['type'][0] == 'EEG'), None)
    marker_stream = next((s for s in streams if s['info']['type'][0] == 'Markers'), None)

    if eeg_stream is None or marker_stream is None:
        raise ValueError("Both EEG and Marker streams must be present in the XDF file.")

    print("EEG and Marker streams successfully loaded.")
    return eeg_stream, marker_stream

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