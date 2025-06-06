import mne
import numpy as np
import pyxdf
import matplotlib.pyplot as plt
import os
import config
from mne.time_frequency import tfr_multitaper

# Define Relevant Markers for Classification
EPOCHS_START_END = {
    config.TRIGGERS["REST_BEGIN"]: config.TRIGGERS["REST_END"],
    config.TRIGGERS["MI_BEGIN"]: config.TRIGGERS["MI_END"],
    config.TRIGGERS["ROBOT_BEGIN"]: config.TRIGGERS["ROBOT_END"]
}

# --- Load XDF ---
def load_xdf_and_parse(path):
    streams, header = pyxdf.load_xdf(path)
    eeg_stream = next(s for s in streams if s['info']['type'][0] == 'EEG')
    marker_stream = next(s for s in streams if s['info']['type'][0] == 'Markers')
    return eeg_stream, marker_stream

# --- Preprocess EEG ---
def preprocess_eeg(eeg_stream):
    data = np.array(eeg_stream["time_series"]).T
    timestamps = np.array(eeg_stream["time_stamps"])
    channel_names = [c['label'][0] for c in eeg_stream['info']['desc'][0]['channels'][0]['channel']]

    non_eeg_channels = {"AUX1", "AUX2", "AUX3", "AUX7", "AUX8", "AUX9", "TRIGGER"}
    rename_dict = {
        "FP1": "Fp1", "FPZ": "Fpz", "FP2": "Fp2",
        "FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "POZ": "POz", "OZ": "Oz"
    }
    valid_channel_names = [ch for ch in channel_names if ch not in non_eeg_channels]
    valid_indices = [channel_names.index(ch) for ch in valid_channel_names]
    data = data[valid_indices, :]

    info = mne.create_info(ch_names=valid_channel_names, sfreq=config.FS, ch_types="eeg")
    raw = mne.io.RawArray(data, info)

    for ch in raw.info['chs']:
        ch['unit'] = 201  # µV

    if "M1" in raw.ch_names and "M2" in raw.ch_names:
        raw.drop_channels(["M1", "M2"])

    raw.rename_channels(rename_dict)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"))

    raw.notch_filter(60, method="iir")
    raw.filter(config.LOWCUT, config.HIGHCUT, method="iir")

    if config.SURFACE_LAPLACIAN_TOGGLE:
        raw = mne.preprocessing.compute_current_source_density(raw)

    return raw, timestamps

# --- Extract Epochs with Variable Duration Padding ---
def extract_and_pad_tfr_epochs(raw, marker_data, marker_timestamps, eeg_timestamps, marker_start, marker_end,
                               baseline=(-1.0, 0.0), tmin=-1.0):
    marker_data = marker_data.astype(int)
    marker_start = int(marker_start)
    marker_end = int(marker_end)

    start_indices = np.where(marker_data == marker_start)[0]
    end_indices = np.where(marker_data == marker_end)[0]
    min_len = min(len(start_indices), len(end_indices))

    tfr_list = []
    max_samples = 0
    sfreq = raw.info['sfreq']

    for i in range(min_len):
        start_time = marker_timestamps[start_indices[i]]
        end_time = marker_timestamps[end_indices[i]]
        tmax = end_time - start_time

        event_sample = np.searchsorted(eeg_timestamps, start_time)
        events = np.array([[event_sample, 0, marker_start]])

        epoch = mne.Epochs(raw, events=events, event_id={str(marker_start): marker_start},
                           tmin=tmin, tmax=tmax, baseline=baseline, preload=True, verbose=False)

        freqs = np.linspace(config.LOWCUT, config.HIGHCUT, config.HIGHCUT - config.LOWCUT)
        tfr = epoch.compute_tfr(method="multitaper", freqs=freqs, n_cycles=2.5,
                                use_fft=True, return_itc=False, verbose=False)
        tfr.apply_baseline(baseline=baseline, mode="logratio")

        tfr_list.append(tfr)
        max_samples = max(max_samples, tfr.data.shape[-1])

    if not tfr_list:
        return None

    padded_data = []
    for tfr in tfr_list:
        pad_width = max_samples - tfr.data.shape[-1]
        pad = [(0, 0), (0, 0), (0, 0), (0, pad_width)]
        padded = np.pad(tfr.data, pad_width=pad, mode='constant', constant_values=np.nan)
        padded_data.append(padded)

    avg_data = np.nanmean(np.stack(padded_data), axis=0)
    avg_tfr = tfr_list[0].copy()
    avg_tfr.data = avg_data
    avg_tfr.times = np.linspace(tfr_list[0].times[0], tfr_list[0].times[0] + max_samples / sfreq, max_samples)
    return avg_tfr

# --- Main Visualization ---
def compute_and_plot_tfr_streaming(raw, marker_stream, eeg_timestamps):
    marker_values = np.array([int(m[0]) for m in marker_stream["time_series"]])
    marker_timestamps = np.array([float(m[1]) for m in marker_stream["time_series"]])

    print("Available markers:", np.unique(marker_values))

    all_tfrs = {}
    for start_marker, end_marker in EPOCHS_START_END.items():
        tfr = extract_and_pad_tfr_epochs(raw, marker_values, marker_timestamps, eeg_timestamps,
                                         start_marker, end_marker, baseline=(-1.0, 0.0), tmin=-1.0)
        if tfr is not None:
            all_tfrs[str(start_marker)] = tfr

    if not all_tfrs:
        print("⚠️ No valid TFR data was computed. Skipping plotting.")
        return

    all_vals = np.concatenate([t.data.flatten() for t in all_tfrs.values()])
    vmin, vmax = np.percentile(all_vals, [2, 98])

    for label, tfr in all_tfrs.items():
        t_start = 1.0
        t_end = min(tfr.times[-1], 5.0)
        fig = tfr.plot_topomap(tmin=t_start, tmax=t_end, ch_type='eeg',
                               fmin=config.LOWCUT, fmax=config.HIGHCUT,
                               vlim=(vmin, vmax), cmap='viridis', show=True)
        fig.suptitle(f"ERD/ERS Topomap (Avg {t_start}s to {t_end}s) - Marker {label}", fontsize=14)

# --- Main Entry ---
def main():
    subject = "CLASS_SUBJ_831"
    session = "S002OFFLINE_NOFES"

    xdf_dir = os.path.join("/home/arman-admin/Documents/CurrentStudy", f"sub-{subject}", f"ses-{session}", "eeg")
    xdf_files = [f for f in os.listdir(xdf_dir) if f.endswith(".xdf")]
    if not xdf_files:
        raise FileNotFoundError("No .xdf file found in specified directory.")
    xdf_path = os.path.join(xdf_dir, xdf_files[0])

    eeg_stream, marker_stream = load_xdf_and_parse(xdf_path)
    raw, eeg_timestamps = preprocess_eeg(eeg_stream)
    compute_and_plot_tfr_streaming(raw, marker_stream, eeg_timestamps)

if __name__ == "__main__":
    main()
