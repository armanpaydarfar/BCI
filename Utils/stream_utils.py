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

