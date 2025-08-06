import os
import pyxdf
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import welch
import config
from scipy.stats import zscore
# Custom utility functions
from Utils.preprocessing import apply_notch_filter, extract_segments, separate_classes, compute_grand_average,concatenate_streams
from Utils.stream_utils import get_channel_names_from_xdf, load_xdf

subject = "DELAND"
session = "S001OFFLINE"

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
raw._data /= 1e6  # Convert V → µV

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
highband = 12
lowband = 8

time_start = -1
baseline_period = 1
time_end = 5

# Preprocessing
raw.notch_filter(60)  # Notch filter at 50Hz
raw.filter(l_freq=lowband, h_freq=highband, fir_design="firwin")  # Bandpass filter (8-16Hz)
#print(f" Data Range Before Scaling: min={raw._data.min()}, max={raw._data.max()}")

# Compute Surface Laplacian (CSD)
#raw.set_eeg_reference('average')
#raw_no_spatial_filter = raw.copy()
raw = mne.preprocessing.compute_current_source_density(raw)

# Print the first EEG channel’s metadata
#print(f"Updated Units for EEG Channels: {[ch['unit'] for ch in raw.info['chs']]}")


# Debug: Final check
print("\n Final EEG Channels After Processing:", raw.ch_names)

#print(marker_data)
#print(marker_timestamps)
#print(eeg_timestamps)
# Create Events Array for MNE Epoching
unique_markers = np.unique(marker_data)
event_dict = {str(marker): marker for marker in unique_markers}
events = np.column_stack((
    np.searchsorted(eeg_timestamps, marker_timestamps),  # Sample index
    np.zeros(len(marker_data), dtype=int),              # MNE requires a placeholder column
    marker_data                                        # Marker values
))

# Ensure sample index uniqueness by nudging marker 300 forward by 1 sample if needed
unique_samples = set()
for i in range(len(events)):
    sample = events[i, 0]
    code = events[i, 2]

    if sample in unique_samples:
        # Conflict detected; check if it's marker 300
        if code == 300:
            print(f"Nudging marker 300 at index {i} forward by 1 sample.")
            events[i, 0] += 1  # Shift it forward by 1
        else:
            print(f"Warning: Non-300 event duplicated at sample {sample} – manual check suggested.")

    unique_samples.add(events[i, 0])


'''
# Get sample indices
sample_indices = events[:, 0]

# Find duplicates by checking for repeated sample indices
_, idx_counts = np.unique(sample_indices, return_counts=True)
duplicate_indices = sample_indices[np.where(idx_counts > 1)]

if duplicate_indices.size > 0:
    print("⚠️ Duplicate sample indices found:")
    print(duplicate_indices)
else:
    print("✅ No duplicate sample indices.")

'''
#print(events)
#print(marker_data)
#print(marker_timestamps)
#print(np.searchsorted(eeg_timestamps, marker_timestamps))
# Define marker labels
marker_labels = {
    "100": "Rest",
    "200": "Right Arm MI",
    "300":"Robot Move"
}
# Create Epochs with rejection
epochs = mne.Epochs(
    raw, events, event_id=event_dict, tmin=time_start, tmax=time_end, baseline = None, detrend=1, preload=True
)


#baseline =(None,time_start+0.5)

# Define a rejection threshold (adjust as needed)5


# Compute max per epoch
max_per_epoch = np.max(np.abs(epochs.get_data()), axis=(1, 2))  

# Compute z-scores
z_scores = zscore(max_per_epoch)

# Define a rejection criterion (e.g., 3 standard deviations)
reject_z_threshold = 3.0 
bad_epochs = np.where(np.abs(z_scores) > reject_z_threshold)[0]  

# Drop the bad epochs
epochs.drop(bad_epochs)
print(f"Dropped {len(bad_epochs)} bad epochs based on z-score method.")
for marker in ["100", "200"]:
    print(f"Marker {marker}: {len(epochs[marker])} epochs")


# **Define time windows (instead of discrete time points)**
num_windows = 5  # Change this for finer resolution
window_size = 0.5  # Each window covers 500ms
time_windows = np.linspace(0.1, 5.0 - window_size, num_windows)  # Avoid end clipping



# Config
channel_name = "P4"
target_marker = 200  # for example, MI trials
trial_index = 0      # 0 = first trial, 1 = second, etc.

# Get channel index
channel_index = epochs.ch_names.index(channel_name)

# Select the epochs corresponding to the target marker
epochs_for_marker = epochs[target_marker]  # or epochs['200'] if using string keys

# Get data: shape (n_trials, n_channels, n_times)
data = epochs_for_marker.get_data()
#print(data.shape)
# Extract signal for the desired trial and channel
signal = data[trial_index, channel_index, :]
times = epochs_for_marker.times

# Plot it
plt.figure(figsize=(10, 4))
plt.plot(times, signal)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (uV)")
plt.title(f"Trial #{trial_index+1} - Marker {target_marker}, Channel {channel_name}")
plt.axvline(0, color='k', linestyle='--', label='Stimulus Onset')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



'''

# **Step 1: Square all epochs to compute signal power**
print("Squaring all epochs for signal power computation...")

# Copy epochs
epochs_power = epochs.copy()

# Find baseline indices


baseline_indices = epochs_power.time_as_index([time_start, time_start+baseline_period])  # Baseline window (-0.5 to 0 sec)
idx_start, idx_end = baseline_indices

# Compute baseline mean **before squaring**

baseline_mean = np.mean(epochs_power._data[:, :, idx_start:idx_end], axis=2, keepdims=True)

# Subtract baseline mean from original signal
epochs_power._data -= baseline_mean

epochs_power._data = np.square(epochs_power._data)  # Square EEG signal

# **Step 2: Compute grand-average power per event type**
evoked_power = {}
for event_id in ["100", "200"]:
    if event_id in event_dict:
        evoked_power[event_id] = epochs_power[event_id].average()
        print(f"Computed grand average power for marker {event_id}.")

# **Step 3: Convert to proper display units (mV²/m⁴ → µV²/mm⁴)**
scaling_factor = 1e6  # Convert from mV²/m⁴ to µV²/mm⁴
#scaling_factor = 1
for key in evoked_power.keys():
    evoked_power[key].data *= scaling_factor
    print(f"Post-Scaling Power Data (Marker {key}): "
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

    # Set figure title using the marker label


    marker_label = marker_labels.get(event_id, f"Marker {event_id}")  # Default to marker number if missing  
    fig.suptitle(f"Signal Power - {marker_label}", fontsize=14)

    #fig.suptitle(f"Signal Power Over Time Windows - Marker {event_id}", fontsize=14)
    figures[event_id] = fig

plt.show()
'''
# **Step 6: Compute and Plot PSD**
#raw.compute_psd(fmax=50).plot()
# Select epochs based on marker values
epochs_marker1 = epochs['100']  # Replace '100' with the actual event ID for the first marker
epochs_marker2 = epochs['200']  # Replace '200' with the actual event ID for the second marker

# Compute and plot PSD for the first marker
psd_marker1 = epochs_marker1.compute_psd(method='welch', fmin=1, fmax=60, n_fft=512, n_overlap=256)
psd_marker1.plot()  # Modify title based on event ID
plt.title("PSD for rest")  # Set title separately
# Compute and plot PSD for the second marker
psd_marker2 = epochs_marker2.compute_psd(method='welch', fmin=1, fmax=60, n_fft=512, n_overlap=256)
psd_marker2.plot()  # Modify title based on event ID
plt.title("PSD for MI")  # Set title separately

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

# Define Time Windows for ERD/ERS
baseline = (time_start, time_start+baseline_period)  # Pre-event window
window_size = 0.5  # Window duration in seconds
time_windows = np.arange(0, 5, window_size)  # Start times

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
skip_factor = 2  # Plot every 2nd time window

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


    # Set figure title using the marker label
    marker_label = marker_labels.get(marker, f"Marker {marker}")  # Default to marker number if missing
    fig.suptitle(f"ERD/ERS Over Time Windows - {marker_label}", fontsize=14)
    figures[marker] = fig

plt.show()


'''

##################################################################################
#ERD or ERS analysis below
##################################################################################
##################################################################################



# Define motor cortex channels for ERD/ERS averaging
motor_cortex_channels = ["P3"]  # Adjust as needed
selected_ch_indices = [epochs.ch_names.index(ch) for ch in motor_cortex_channels if ch in epochs.ch_names]

# Time Windows for ERD/ERS
baseline = (time_start, time_start+baseline_period)  # Pre-event baseline
window_size = 0.5  # Window duration in seconds
time_windows = np.arange(0, 5, window_size)  # Start times for windows

# Compute ERD/ERS
tfr_data = {}

for marker in ["100", "200"]:
    if marker in event_dict:
        tfr = epochs[marker].compute_tfr(
            method="multitaper",
            freqs=np.linspace(lowband, highband, highband - lowband),
            tmin=time_start, tmax=time_end,
            n_cycles=2.5, use_fft=True, return_itc=False
        )
        # Apply baseline correction
        tfr.apply_baseline(baseline=baseline, mode="logratio")
        tfr_data[marker] = tfr

# Plot ERD/ERS Averaged Over Motor Cortex Channels
fig, axes = plt.subplots(1, len(tfr_data), figsize=(12, 6), sharex=True, sharey=True)

for col, (marker, tfr) in enumerate(tfr_data.items()):
    times = tfr.times  # Extract time points

    # Compute mean & std across selected motor cortex channels
    erd_ers_mean = np.mean(tfr.data[selected_ch_indices], axis=(0, 1, 2))  # Mean over channels, epochs, freqs
    erd_ers_std = np.std(tfr.data[selected_ch_indices], axis=(0, 1, 2))    # STD over channels, epochs, freqs

    ax = axes[col] if len(tfr_data) > 1 else axes
    ax.plot(times, erd_ers_mean, label=f"Avg ERD/ERS ({', '.join(motor_cortex_channels)})", color='b')

    # Plot shaded region for variability (mean ± std)
    ax.fill_between(times, erd_ers_mean - erd_ers_std, erd_ers_mean + erd_ers_std, color='b', alpha=0.3, label="±1 STD")

    # Highlight the baseline window
    ax.axvspan(baseline[0], baseline[1], color="gray", alpha=0.2, label="Baseline")

    # Formatting
    ax.axhline(0, linestyle="--", color="black", linewidth=1)  # Zero line
    ax.set_title(f"{marker_labels.get(marker, f'Marker {marker}')} -  ERD/ERS")
    ax.set_ylabel("ERD/ERS (logratio)")
    ax.set_xlabel("Time (s)")
    ax.legend()

# Dynamically get channel names being averaged
channel_labels = ", ".join(motor_cortex_channels)

# Update plot title to show exact channels
plt.suptitle(f"ERD/ERS Over Selected Channels: {channel_labels}", fontsize=14)
plt.tight_layout()
plt.show()




'''


#########################
#TESTING
#########################
# Compute ERD/ERS Using MNE's Updated API

'''


# Define Frequency Band (Alpha: 8-12 Hz)
freqs = np.linspace(8, 12, 5)  # Alpha range
n_cycles = freqs / 2  # Adaptive cycles

# Define Baseline and Analysis Window
baseline = (-0.5, 0)  # Pre-event window (-0.5 to 0 sec)
time_window = (0.1, 1.0)  # Analysis period

# Compute ERD/ERS Using MNE's Updated API
tfr_data = {}

for marker in ["100", "200", "300"]:
    if marker in event_dict:
        tfr = epochs[marker].compute_tfr(
            method="multitaper", freqs=freqs, n_cycles=n_cycles, 
            use_fft=True, return_itc=False
        )
        tfr.apply_baseline(baseline=baseline, mode="percent")  # Normalize by baseline
        print(f"Computed ERD/ERS for marker {marker}.")
        
        tfr_data[marker] = tfr.average()  # Convert to AverageTFR

# Adjust ERD/ERS values: Center at 0% (baseline unchanged)
for marker in tfr_data.keys():
    tfr_data[marker].data *= 100
    tfr_data[marker].data -= 100  # Shift to center at 0%

# Compute dynamic vmin/vmax across all markers
all_erd_values = np.concatenate([tfr_avg.data.flatten() for tfr_avg in tfr_data.values()])
vmin, vmax = np.percentile(all_erd_values, [2, 98])  # Use percentiles to avoid extreme outliers

print(f" Dynamic ERD/ERS Color Scale: vmin={vmin:.2f}, vmax={vmax:.2f}")

# Plot ERD/ERS Topomap for the Alpha Band (8-12 Hz)
figures = {}

for marker, tfr_avg in tfr_data.items():
    fig = tfr_avg.plot_topomap(
        tmin=time_window[0], tmax=time_window[1], fmin=8, fmax=12, show=False,
        cmap="viridis", vlim=(vmin, vmax)  # Dynamic limits
    )
    fig.suptitle(f"ERD/ERS Topomap - Marker {marker} (Alpha Band)", fontsize=14)

    # Modify colorbar label
    for ax in fig.axes:
        if hasattr(ax, 'images') and ax.images:
            cbar = ax.images[0].colorbar
            if cbar:
                cbar.set_label("ERD/ERS (%)", fontsize=12)
                cbar.ax.set_title("")  # Remove default label above colorbar

    figures[marker] = fig  # Store figures

plt.show()

'''
