import os
import pyxdf
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import welch
import config

# Custom utility functions
from Utils.preprocessing import apply_notch_filter, butter_bandpass_filter, extract_segments, separate_classes, compute_grand_average
from Utils.stream_utils import get_channel_names_from_xdf


# ----- Load XDF Data -----
xdf_dir = "/home/arman-admin/Documents/CurrentStudy/sub-PILOT007/ses-S001/eeg"
xdf_file_path = os.path.join(xdf_dir, "sub-PILOT007_ses-S001_task-Default_run-001OFFLINE_eeg.xdf")

print(f"🔄 Loading XDF file: {xdf_file_path}")
streams, header = pyxdf.load_xdf(xdf_file_path)

# Find EEG and Marker streams
eeg_stream = next((s for s in streams if s["info"]["type"][0] == "EEG"), None)
marker_stream = next((s for s in streams if s["info"]["type"][0] == "Markers"), None)

if eeg_stream is None or marker_stream is None:
    raise ValueError("❌ EEG or Marker stream not found in the XDF file.")

print(f"✅ EEG Stream: {eeg_stream['info']['name'][0]}")
print(f"✅ Marker Stream: {marker_stream['info']['name'][0]}")

# Extract EEG timestamps and data
eeg_timestamps = np.array(eeg_stream["time_stamps"])  # (N_samples,)
eeg_data = np.array(eeg_stream["time_series"]).T  # Shape: (n_channels, n_samples)
channel_names = get_channel_names_from_xdf(eeg_stream)
marker_data = np.array([int(value[0]) for value in marker_stream['time_series']])
marker_timestamps = np.array(marker_stream['time_stamps'])
print("\n📌 EEG Channels from XDF:", channel_names)

# ✅ Load standard 10-20 montage
montage = mne.channels.make_standard_montage("standard_1020")

# ✅ Case-sensitive renaming dictionary
rename_dict = {
    "FP1": "Fp1", "FPZ": "Fpz", "FP2": "Fp2",
    "FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "POZ": "POz", "OZ": "Oz"
}

# ✅ Drop non-EEG channels
non_eeg_channels = {"AUX1", "AUX2", "AUX3", "AUX7", "AUX8", "AUX9", "TRIGGER"}
valid_eeg_channels = [ch for ch in channel_names if ch not in non_eeg_channels]

# ✅ Filter data to keep only valid EEG channels
valid_indices = [channel_names.index(ch) for ch in valid_eeg_channels]  # Get indices
eeg_data = eeg_data[valid_indices, :]  # Keep only valid EEG data

# ✅ Create MNE Raw Object
sfreq = config.FS
info = mne.create_info(ch_names=valid_eeg_channels, sfreq=sfreq, ch_types="eeg")
raw = mne.io.RawArray(eeg_data, info)

first_channel_unit = raw.info["chs"][0]["unit"]
print(f"🔎 First Channel Unit (FIFF Code): {first_channel_unit}")

# Convert data from Volts to microvolts (µV)
# ✅ Convert raw data from Volts to microvolts (µV) IMMEDIATELY AFTER LOADING
raw._data /= 1e6  # Convert V → µV

# ✅ Update channel metadata in MNE so the scaling is correctly reflected

for ch in raw.info['chs']:
    ch['unit'] = 201  # 201 corresponds to µV in MNE’s standard units
# ✅ Print the first EEG channel’s metadata

# ✅ Print to confirm the change
print(f"🔎 Updated Units for EEG Channels: {[ch['unit'] for ch in raw.info['chs']]}")

if "M1" in raw.ch_names and "M2" in raw.ch_names:
    raw.drop_channels(["M1", "M2"])
    print("✅ Removed Mastoid Channels: M1, M2")
else:
    print("ℹ️ No Mastoid Channels Found in Data")


# ✅ Rename channels to match montage format
raw.rename_channels(rename_dict)


# ✅ Debug: Print missing channels
missing_in_montage = set(raw.ch_names) - set(montage.ch_names)
print(f"⚠️ Channels in Raw but Missing in Montage: {missing_in_montage}")

# ✅ Validate channel positions (drop truly invalid ones)
'''
invalid_channels = [ch["ch_name"] for ch in raw.info["chs"] if np.isnan(ch["loc"]).any() or np.isinf(ch["loc"]).any()]

if invalid_channels:
    print(f"⚠️ Dropping channels with invalid positions: {invalid_channels}")
    if len(invalid_channels) < len(raw.ch_names):  # Avoid dropping all channels
        raw.drop_channels(invalid_channels)
    else:
        print("🚨 WARNING: Attempted to drop all channels. Keeping all channels instead.")


print("\n📍 Checking Electrode Positions:")
for ch in raw.info["chs"]:
    print(f"{ch['ch_name']}: {ch['loc'][:3]}")  # Only print X, Y, Z coordinates
'''
# Apply montage with "match_case=True" for exact name matching
raw.set_montage(montage, match_case=True, on_missing="warn")

'''
# Validate again
print("\n✅ Rechecking Channel Positions After Montage Application:")
for ch in raw.info["chs"]:
    print(f"{ch['ch_name']}: {ch['loc'][:3]}")
'''
highband = 12
lowband = 8

# ✅ Preprocessing
raw.notch_filter(60)  # Notch filter at 50Hz
raw.filter(l_freq=lowband, h_freq=highband, fir_design="firwin")  # Bandpass filter (8-16Hz)
#print(f"🔎 Data Range Before Scaling: min={raw._data.min()}, max={raw._data.max()}")

# ✅ Compute Surface Laplacian (CSD)
#raw.set_eeg_reference('average')
#raw_no_spatial_filter = raw.copy()
raw = mne.preprocessing.compute_current_source_density(raw)

# ✅ Print the first EEG channel’s metadata
#print(f"🔎 Updated Units for EEG Channels: {[ch['unit'] for ch in raw.info['chs']]}")


# ✅ Debug: Final check
print("\n✅ Final EEG Channels After Processing:", raw.ch_names)


# Create Events Array for MNE Epoching
unique_markers = np.unique(marker_data)
event_dict = {str(marker): marker for marker in unique_markers}
events = np.column_stack((
    np.searchsorted(eeg_timestamps, marker_timestamps),  # Sample index
    np.zeros(len(marker_data), dtype=int),              # MNE requires a placeholder column
    marker_data                                        # Marker values
))

# ✅ Define marker labels
marker_labels = {
    "100": "Rest",
    "200": "Right Arm MI",
    "300": "Robot Move"
}
# Create Epochs with rejection
epochs = mne.Epochs(
    raw, events, event_id=event_dict, tmin=-0.5, tmax=5.0, baseline = (None,0), detrend=1, preload=True
)
# Define a rejection threshold (adjust as needed)5
reject_threshold = 0.03 # 150 mV/m²

# Get absolute max per epoch
max_per_epoch = np.max(np.abs(epochs.get_data()), axis=(1, 2))  

# Identify bad epochs
bad_epochs = np.where(max_per_epoch > reject_threshold)[0]  

# Drop the bad epochs
epochs.drop(bad_epochs)

print(f"🚀 Dropped {len(bad_epochs)} bad epochs exceeding {reject_threshold} mV/m².")


# **Define time windows (instead of discrete time points)**
num_windows = 5  # Change this for finer resolution
window_size = 0.5  # Each window covers 500ms
time_windows = np.linspace(0.1, 5.0 - window_size, num_windows)  # Avoid end clipping

# **Step 1: Square all epochs to compute signal power**
print("🔄 Squaring all epochs for signal power computation...")

# Copy epochs
epochs_power = epochs.copy()

# ✅ Find baseline indices
baseline_indices = epochs.time_as_index([-0.5, 0])  # Baseline window (-0.5 to 0 sec)
idx_start, idx_end = baseline_indices

# ✅ Compute baseline mean **before squaring**
baseline_mean = np.mean(epochs_power._data[:, :, idx_start:idx_end], axis=2, keepdims=True)

# ✅ Subtract baseline mean from original signal
epochs_power._data -= baseline_mean

epochs_power._data = np.square(epochs_power._data)  # Square EEG signal

# **Step 2: Compute grand-average power per event type**
evoked_power = {}
for event_id in ["100", "200", "300"]:
    if event_id in event_dict:
        evoked_power[event_id] = epochs_power[event_id].average()
        print(f"✅ Computed grand average power for marker {event_id}.")

# **Step 3: Convert to proper display units (mV²/m⁴ → µV²/mm⁴)**
scaling_factor = 1e6  # Convert from mV²/m⁴ to µV²/mm⁴
for key in evoked_power.keys():
    evoked_power[key].data *= scaling_factor
    print(f"🔎 Post-Scaling Power Data (Marker {key}): "
          f"min={evoked_power[key].data.min()}, max={evoked_power[key].data.max()}")

# **Step 4: Compute mean power over each time window**
windowed_power = {}
for event_id in evoked_power.keys():
    windowed_power[event_id] = []

    for t in time_windows:
        # **Find time indices corresponding to the window**
        idx_start = np.argmin(np.abs(evoked_power[event_id].times - t))
        idx_end = np.argmin(np.abs(evoked_power[event_id].times - (t + window_size)))

        # **Compute mean power over the time window**
        mean_power = np.mean(evoked_power[event_id].data[:, idx_start:idx_end], axis=1)
        windowed_power[event_id].append(mean_power)

    windowed_power[event_id] = np.array(windowed_power[event_id])  # Convert to array

# **Step 5: Plot Topographic Maps for Each Time Window**
figures = {}
for event_id in evoked_power.keys():
    fig, axes = plt.subplots(1, len(time_windows), figsize=(15, 4))

    for i, (t, mean_power) in enumerate(zip(time_windows, windowed_power[event_id])):
        # **Plot topographic map using mean power over time window**
        im, _ = mne.viz.plot_topomap(
            mean_power,
            evoked_power[event_id].info,
            axes=axes[i],
            show=False
        )
        axes[i].set_title(f"{t:.1f} - {t + window_size:.1f}s")

    # **Add colorbar**
    cbar = plt.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1)
    cbar.set_label("Power (µV²/mm⁴)", fontsize=12)

    # ✅ Set figure title using the marker label


    marker_label = marker_labels.get(event_id, f"Marker {event_id}")  # Default to marker number if missing  
    fig.suptitle(f"Signal Power - {marker_label}", fontsize=14)

    #fig.suptitle(f"Signal Power Over Time Windows - Marker {event_id}", fontsize=14)
    figures[event_id] = fig

plt.show()

# **Step 6: Compute and Plot PSD**
#raw.compute_psd(fmax=50).plot()
epochs.compute_psd(fmax= 50).plot()
# **Step 7: Plot Event Markers Over Time**
fig, ax = plt.subplots(figsize=(10, 5))
sc = ax.scatter(marker_timestamps, marker_data, c=marker_data, cmap='coolwarm', alpha=0.7)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Marker Value")  # Set label correctly

ax.set_xlabel("Time (s)")
ax.set_ylabel("Marker Values")
ax.set_title("Markers vs Time")

plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

##################################################################################
#ERD or ERS analysis below
##################################################################################
##################################################################################

# ✅ Define Time Windows for ERD/ERS
baseline = (-0.5, 0)  # Pre-event window
window_size = 0.5  # Window duration in seconds
time_windows = np.arange(0, 5, window_size)  # Start times

# ✅ Compute ERD/ERS Using MNE's Updated API
tfr_data = {}

for marker in ["100", "200", "300"]:
    if marker in event_dict:
        tfr = epochs[marker].compute_tfr(
            method="multitaper", freqs=np.linspace(8, 12, 5), n_cycles=2.5, 
            use_fft=True, return_itc=False
        )

        # ✅ Apply baseline correction
        print(f"🔍 Before baseline correction: {np.mean(tfr.data):.6f}")  # Debugging
        tfr.apply_baseline(baseline=baseline, mode="percent")  
        print(f"✅ Computed ERD/ERS for marker {marker}. 🔍 After baseline correction: {np.mean(tfr.data):.6f}")

        # ✅ Convert to AverageTFR and store
        tfr_data[marker] = tfr.average()

# ✅ Adjust ERD/ERS values to be centered at 0%
for marker, tfr_avg in tfr_data.items():
    tfr_avg.data *= 100  # Convert to percentage
    tfr_avg.data -= 100  # Shift to center at 0%

# ✅ Compute dynamic vmin/vmax across all markers
all_erd_values = np.concatenate([tfr_avg.data.flatten() for tfr_avg in tfr_data.values()])
vmin, vmax = np.percentile(all_erd_values, [2, 98])  # Use percentiles to avoid extreme outliers

print(f"🔍 Dynamic ERD/ERS Color Scale: vmin={vmin:.2f}, vmax={vmax:.2f}")

# ✅ Plot ERD/ERS Topographic Maps Using `tmin` and `tmax`
figures = {}
skip_factor = 2  # ✅ Plot every 2nd time window

for marker, tfr_avg in tfr_data.items():
    selected_indices = range(0, len(time_windows), skip_factor)  # ✅ Select every other index
    fig, axes = plt.subplots(1, len(selected_indices), figsize=(15, 4), constrained_layout=True)

    im = None  # ✅ Store last valid image for the colorbar

    for ax, i in zip(axes, selected_indices):
        t_start = time_windows[i]
        t_end = t_start + window_size  # ✅ Dynamic time window

        # ✅ Extract correct mappable object for color bar
        img = tfr_avg.plot_topomap(
            tmin=t_start, tmax=t_end,
            axes=ax, cmap="viridis", show=False, vlim=(vmin, vmax), colorbar=False
        )

        # ✅ Extract the colorbar mappable from the plot
        if hasattr(ax, "collections") and len(ax.collections) > 0:
            im = ax.collections[0]  # ✅ First collection should be the colorbar mappable

        ax.set_title(f"{t_start:.1f} - {t_end:.1f}s")

    # ✅ Fix the global color bar using the last valid image
    if im is not None:
            norm = plt.Normalize(vmin, vmax)  # ✅ Ensure proper normalization
            sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")  # ✅ Create a proper color mapping
            sm.set_array([])  # ✅ Required for colorbar to display properly
            cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1)
            cbar.set_label("ERD/ERS (%)", fontsize=12)


    # ✅ Set figure title using the marker label
    marker_label = marker_labels.get(marker, f"Marker {marker}")  # Default to marker number if missing
    fig.suptitle(f"ERD/ERS Over Time Windows - {marker_label}", fontsize=14)
    figures[marker] = fig



plt.show()

#########################
#TESTING
#########################
# ✅ Compute ERD/ERS Using MNE's Updated API

'''


# ✅ Define Frequency Band (Alpha: 8-12 Hz)
freqs = np.linspace(8, 12, 5)  # Alpha range
n_cycles = freqs / 2  # Adaptive cycles

# ✅ Define Baseline and Analysis Window
baseline = (-0.5, 0)  # Pre-event window (-0.5 to 0 sec)
time_window = (0.1, 1.0)  # Analysis period

# ✅ Compute ERD/ERS Using MNE's Updated API
tfr_data = {}

for marker in ["100", "200", "300"]:
    if marker in event_dict:
        tfr = epochs[marker].compute_tfr(
            method="multitaper", freqs=freqs, n_cycles=n_cycles, 
            use_fft=True, return_itc=False
        )
        tfr.apply_baseline(baseline=baseline, mode="percent")  # Normalize by baseline
        print(f"✅ Computed ERD/ERS for marker {marker}.")
        
        tfr_data[marker] = tfr.average()  # Convert to AverageTFR

# ✅ Adjust ERD/ERS values: Center at 0% (baseline unchanged)
for marker in tfr_data.keys():
    tfr_data[marker].data *= 100
    tfr_data[marker].data -= 100  # Shift to center at 0%

# ✅ Compute dynamic vmin/vmax across all markers
all_erd_values = np.concatenate([tfr_avg.data.flatten() for tfr_avg in tfr_data.values()])
vmin, vmax = np.percentile(all_erd_values, [2, 98])  # Use percentiles to avoid extreme outliers

print(f"🔍 Dynamic ERD/ERS Color Scale: vmin={vmin:.2f}, vmax={vmax:.2f}")

# ✅ Plot ERD/ERS Topomap for the Alpha Band (8-12 Hz)
figures = {}

for marker, tfr_avg in tfr_data.items():
    fig = tfr_avg.plot_topomap(
        tmin=time_window[0], tmax=time_window[1], fmin=8, fmax=12, show=False,
        cmap="viridis", vlim=(vmin, vmax)  # ✅ Dynamic limits
    )
    fig.suptitle(f"ERD/ERS Topomap - Marker {marker} (Alpha Band)", fontsize=14)

    # ✅ Modify colorbar label
    for ax in fig.axes:
        if hasattr(ax, 'images') and ax.images:
            cbar = ax.images[0].colorbar
            if cbar:
                cbar.set_label("ERD/ERS (%)", fontsize=12)
                cbar.ax.set_title("")  # 🔴 Remove default label above colorbar

    figures[marker] = fig  # Store figures

plt.show()

'''