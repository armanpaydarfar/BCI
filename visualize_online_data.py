import os
import pyxdf
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import welch
import config
from scipy.stats import zscore
# Custom utility functions
from Utils.preprocessing import concatenate_streams
from Utils.stream_utils import get_channel_names_from_xdf, load_xdf

subject = "LAB_SUBJ_W"
session = "S001ONLINE"

# Construct the EEG directory path dynamically
xdf_dir = os.path.join("/home/arman-admin/Documents/CurrentStudy", f"sub-{subject}", f"ses-{session}", "eeg/")

# Ensure the directory exists
if not os.path.exists(xdf_dir):
    raise FileNotFoundError(f"❌ EEG directory not found: {xdf_dir}")

# Find all .xdf files in the EEG folder
xdf_files = [os.path.join(xdf_dir, f) for f in os.listdir(xdf_dir) if f.endswith(".xdf")]

if not xdf_files:
    raise FileNotFoundError(f"❌ No XDF files found in: {xdf_dir}")

print(f"📂 Found {len(xdf_files)} XDF files in: {xdf_dir}")

# Display available files with an index
for idx, file in enumerate(xdf_files, start=1):
    print(f" [{idx}] {os.path.basename(file)}")

# Prompt user for selection
print("\nPress ENTER to merge **all** files, or enter the number(s) of the file(s) to load (comma-separated, e.g., 1,3): ")
user_input = input("➡️  Selection: ").strip()

# Determine which files to load
selected_files = []

if user_input:  # If user enters a choice
    try:
        selected_indices = [int(i) - 1 for i in user_input.split(",")]
        selected_files = [xdf_files[i] for i in selected_indices if 0 <= i < len(xdf_files)]
    except ValueError:
        print("❌ Invalid input. Loading all files instead.")
        selected_files = xdf_files  # Default to all files
else:
    selected_files = xdf_files  # Default to all files

# Load and concatenate selected XDF files
all_streams = []
all_headers = []

# Load multiple XDF files (concatenating EEG streams if needed)
eeg_streams, marker_streams = [], []
for xdf_file in xdf_files:
    eeg_s, marker_s = load_xdf(xdf_file)
    eeg_streams.append(eeg_s)
    marker_streams.append(marker_s)

print(f"✅ Successfully loaded and merged {len(all_streams)} streams from {len(selected_files)} XDF file(s).")

# Find EEG and Marker streams
# Merge multiple runs (if necessary)
eeg_stream, marker_stream = (
    (eeg_streams[0], marker_streams[0]) if len(eeg_streams) == 1 else concatenate_streams(eeg_streams, marker_streams)
)
# Extract EEG timestamps and data
eeg_timestamps = np.array(eeg_stream["time_stamps"])  # (N_samples,)
eeg_data = np.array(eeg_stream["time_series"]).T  # Shape: (n_channels, n_samples)
channel_names = get_channel_names_from_xdf(eeg_stream)
marker_data = np.array([int(value[0]) for value in marker_stream['time_series']])
#marker_timestamps = np.array(marker_stream['time_stamps'])
marker_timestamps = np.array([float(value[1]) for value in marker_stream['time_series']])
#print(marker_stream['time_series'])
#print(marker_timestamps)
print("\n EEG Channels from XDF:", channel_names)
#print(marker_stream[0])
# Load standard 10-20 montage
montage = mne.channels.make_standard_montage("standard_1020")

# Case-sensitive renaming dictionary
rename_dict = {
    "FP1": "Fp1", "FPZ": "Fpz", "FP2": "Fp2",
    "FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "POZ": "POz", "OZ": "Oz"
}

# Drop non-EEG channels
non_eeg_channels = {"AUX1", "AUX2", "AUX3", "AUX7", "AUX8", "AUX9", "TRIGGER"}
valid_eeg_channels = [ch for ch in channel_names if ch not in non_eeg_channels]

# Filter data to keep only valid EEG channels
valid_indices = [channel_names.index(ch) for ch in valid_eeg_channels]  # Get indices
eeg_data = eeg_data[valid_indices, :]  # Keep only valid EEG data

# Create MNE Raw Object
sfreq = config.FS
info = mne.create_info(ch_names=valid_eeg_channels, sfreq=sfreq, ch_types="eeg")
raw = mne.io.RawArray(eeg_data, info)

first_channel_unit = raw.info["chs"][0]["unit"]
print(f"First Channel Unit (FIFF Code): {first_channel_unit}")

# Convert data from Volts to microvolts (µV)
# Convert raw data from Volts to microvolts (µV) IMMEDIATELY AFTER LOADING
#raw._data /= 1e6  # Convert V → µV

# Update channel metadata in MNE so the scaling is correctly reflected

for ch in raw.info['chs']:
    ch['unit'] = 201  # 201 corresponds to µV in MNE’s standard units
# print the first EEG channel’s metadata

#  Print to confirm the change
print(f"Updated Units for EEG Channels: {[ch['unit'] for ch in raw.info['chs']]}")

if "M1" in raw.ch_names and "M2" in raw.ch_names:
    raw.drop_channels(["M1", "M2"])
    print("Removed Mastoid Channels: M1, M2")
else:
    print("No Mastoid Channels Found in Data")


# Rename channels to match montage format
raw.rename_channels(rename_dict)


# Debug: Print missing channels
missing_in_montage = set(raw.ch_names) - set(montage.ch_names)
print(f"⚠️ Channels in Raw but Missing in Montage: {missing_in_montage}")

# Validate channel positions (drop truly invalid ones)
'''
invalid_channels = [ch["ch_name"] for ch in raw.info["chs"] if np.isnan(ch["loc"]).any() or np.isinf(ch["loc"]).any()]

if invalid_channels:
    print(f"⚠️ Dropping channels with invalid positions: {invalid_channels}")
    if len(invalid_channels) < len(raw.ch_names):  # Avoid dropping all channels
        raw.drop_channels(invalid_channels)
    else:
        print(" WARNING: Attempted to drop all channels. Keeping all channels instead.")


print("\n Checking Electrode Positions:")
for ch in raw.info["chs"]:
    print(f"{ch['ch_name']}: {ch['loc'][:3]}")  # Only print X, Y, Z coordinates
'''
# Apply montage with "match_case=True" for exact name matching
raw.set_montage(montage, match_case=True, on_missing="warn")

'''
# Validate again
print("\n Rechecking Channel Positions After Montage Application:")
for ch in raw.info["chs"]:
    print(f"{ch['ch_name']}: {ch['loc'][:3]}")
'''
highband = 13
lowband = 8

time_start = -1
baseline_period = 1
time_end = 2

raw._data /= 1e6


# Preprocessing
raw.notch_filter(60)  # Notch filter at 60Hz
raw.filter(l_freq=lowband, h_freq=highband, method = 'iir')  # Bandpass filter (8-16Hz)
#print(f" Data Range Before Scaling: min={raw._data.min()}, max={raw._data.max()}")

# Compute Surface Laplacian (CSD)
#raw.set_eeg_reference('average')
#raw_no_spatial_filter = raw.copy()
raw = mne.preprocessing.compute_current_source_density(raw)

# Print the first EEG channel’s metadata
#print(f"Updated Units for EEG Channels: {[ch['unit'] for ch in raw.info['chs']]}")


# Debug: Final check
print("\n Final EEG Channels After Processing:", raw.ch_names)
import config

# ---- Parameters ----
min_trial_duration = 1.5  # in seconds
voltage_threshold = 20.0  # in microvolts

# ---- Step 1: Match Start-End Markers and Prune by Duration ----
min_trial_duration = 1.0           # keep your current minimum
max_trial_duration = 5.4         # <5.00s => drop "timeouts"
EPS = 0.02                         # tolerance for float rounding

# Optional: per-class overrides (uncomment to use)
# per_class_max = {100: 4.99, 200: 4.99}  # e.g., same cap for REST(100) and MI(200)

valid_start_indices = []
durations_all = []                 # (optional) collect durations for QA / plotting

for idx, code in enumerate(marker_data):
    if code in [100, 200]:  # MI or REST starts
        t_start = marker_timestamps[idx]

        end_code = code + 20        # 120 for 100, 220 for 200
        end_time = None
        for j in range(idx + 1, len(marker_data)):
            if marker_data[j] == end_code:
                end_time = marker_timestamps[j]
                break

        if not end_time:
            print(f"⚠️ Skipped: No end marker found for start at {t_start:.2f}s")
            continue

        duration = end_time - t_start
        durations_all.append((idx, code, duration))
        print(f"Start: {t_start:.2f}s → End: {end_time:.2f}s | Duration: {duration:.2f}s")

        # Determine cap to use (per-class if provided, else global)
        cap = max_trial_duration  # default
        # if 'per_class_max' in locals() and code in per_class_max:
        #     cap = per_class_max[code]

        # Keep trials within [min, cap], dropping "timeouts" ~5s
        if (duration + EPS) >= min_trial_duration and (duration - EPS) <= cap:
            valid_start_indices.append(idx)
        else:
            reason = []
            if (duration + EPS) < min_trial_duration:
                reason.append(f"too short (<{min_trial_duration}s)")
            if (duration - EPS) > cap:
                reason.append(f"timeout/too long (>{cap}s)")
            print(f"⚠️ Skipped: Duration {duration:.2f}s ({', '.join(reason)})")

# ---- Step 2: Build Events Array ----
marker_data_valid = [marker_data[i] for i in valid_start_indices]
marker_timestamps_valid = [marker_timestamps[i] for i in valid_start_indices]
event_dict = {str(marker): marker for marker in set(marker_data_valid)}
event_samples = np.searchsorted(eeg_timestamps, marker_timestamps_valid)

events = np.column_stack((
    event_samples,
    np.zeros(len(marker_data_valid), dtype=int),
    marker_data_valid
))

# ---- Step 3: Create Epochs ----
epochs = mne.Epochs(
    raw, events, event_id=event_dict, tmin=time_start, tmax=time_end,
    baseline=None, detrend=1, preload=True
)

'''
# ---- Step 4: Voltage-Based Rejection ----
epoch_data = epochs.get_data()  # (n_epochs, n_channels, n_times)
max_voltages = np.max(np.abs(epoch_data), axis=(1, 2))  # Max abs voltage per epoch
bad_voltage_epochs = np.where(max_voltages > voltage_threshold)[0]

print(f"🔌 Rejecting {len(bad_voltage_epochs)} epochs above voltage threshold ({voltage_threshold} µV)")
epochs.drop(bad_voltage_epochs)
'''
# ---- Summary ----
for event_code in ["100", "200"]:
    if event_code in epochs.event_id:
        print(f"✅ Final Marker {event_code}: {len(epochs[event_code])} epochs")





# Define your desired range for each class
start_idx = 0
end_idx = 43

# Limit number of trials per class by index range
subset_epochs_list = []
for event_code in ["100", "200"]:  # REST and MI
    if event_code in epochs.event_id:
        class_epochs = epochs[event_code]
        total = len(class_epochs)

        # Safety check to avoid index error
        if end_idx > total:
            print(f"⚠️ Requested range {start_idx}:{end_idx} exceeds available {total} epochs for event {event_code}. Adjusting end index.")
            end_idx = total

        subset_epochs_list.append(class_epochs[start_idx:end_idx])

# Concatenate selected epochs
epochs = mne.concatenate_epochs(subset_epochs_list)
print(f"✅ Final subset has {len(epochs)} epochs from range {start_idx}:{end_idx}.")





##################################################################################
#ERD or ERS analysis below
##################################################################################
##################################################################################

# Define Time Windows for ERD/ERS
baseline = (time_start, time_start+baseline_period)  # Pre-event window
window_size = 0.5  # Window duration in seconds
time_windows = np.arange(0, 2, window_size)  # Start times

# Compute ERD/ERS Using MNE's Updated API
tfr_data = {}

for marker in ["100", "200"]:
    if marker in event_dict:
        tfr = epochs[marker].compute_tfr(
            method="multitaper", freqs=np.linspace(lowband, highband, highband - lowband),tmin = time_start, tmax= time_end, n_cycles=2.5, 
            use_fft=True, return_itc=False
        )

        # Apply baseline correction
        print(f"Before baseline correction: {np.mean(tfr.data):.6f}")  # Debugging
        tfr.apply_baseline(baseline=baseline, mode="logratio")  
        # Convert to AverageTFR and store
        tfr_data[marker] = tfr.average()
        


# Adjust ERD/ERS values to be centered at 0%
'''
for marker, tfr_avg in tfr_data.items():
    tfr_avg.data *= 100  # Convert to percentage
    tfr_avg.data -= 100  # Shift to center at 0%
'''
# Compute dynamic vmin/vmax across all markers
all_erd_values = np.concatenate([tfr_avg.data.flatten() for tfr_avg in tfr_data.values()])
vmin, vmax = np.percentile(all_erd_values, [2, 98])  # Use percentiles to avoid extreme outliers

print(f"Dynamic ERD/ERS Color Scale: vmin={vmin:.2f}, vmax={vmax:.2f}")

# Plot ERD/ERS Topographic Maps Using `tmin` and `tmax`
figures = {}
skip_factor = 1  # Plot every 2nd time window

for marker, tfr_avg in tfr_data.items():
    selected_indices = range(0, len(time_windows), skip_factor)  # Select every other index
    fig, axes = plt.subplots(1, len(selected_indices), figsize=(15, 4), constrained_layout=True)

    im = None  # Store last valid image for the colorbar

    for ax, i in zip(axes, selected_indices):
        t_start = time_windows[i]
        t_end = t_start + window_size  # Dynamic time window

        # Extract correct mappable object for color bar
        img = tfr_avg.plot_topomap(
            tmin=t_start, tmax=t_end,
            axes=ax, cmap="viridis", show=False, vlim=(vmin, vmax), colorbar=False
        )

        # Extract the colorbar mappable from the plot
        if hasattr(ax, "collections") and len(ax.collections) > 0:
            im = ax.collections[0]  # First collection should be the colorbar mappable

        ax.set_title(f"{t_start:.1f} - {t_end:.1f}s")

    # Fix the global color bar using the last valid image
    if im is not None:
            norm = plt.Normalize(vmin, vmax)  # Ensure proper normalization
            sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")  # Create a proper color mapping
            sm.set_array([])  # Required for colorbar to display properly
            cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1)
            cbar.set_label("ERD/ERS (logratio)", fontsize=12)

    marker_labels = {
        "100": "Rest",
        "200": "Right Arm MI",
        "300":"Robot Move"
    }
    # Set figure title using the marker label
    marker_label = marker_labels.get(marker, f"Marker {marker}")  # Default to marker number if missing
    fig.suptitle(f"ERD/ERS Over Time Windows - {marker_label}", fontsize=14)
    figures[marker] = fig

plt.show()



