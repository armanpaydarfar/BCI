import os
import pyxdf
import numpy as np
import matplotlib.pyplot as plt

# Directory containing XDF files
xdf_dir = '/home/arman-admin/Documents/CurrentStudy/sub-P001/ses-S001/eeg'

# Specify the XDF file
xdf_files = ['sub-P001_ses-S001_task-Default_run-001_eeg.xdf']

if not xdf_files:
    raise FileNotFoundError("No XDF files found in the specified directory.")

# Load the XDF file
xdf_file_path = os.path.join(xdf_dir, xdf_files[0])
print(f"Loading XDF file: {xdf_file_path}")

streams, header = pyxdf.load_xdf(xdf_file_path)

# Print stream names and information
print("Streams found in XDF file:")
for stream in streams:
    print(f"Stream Name: {stream['info']['name'][0]}")
    print(f"Stream Type: {stream['info']['type'][0]}")
    print(f"Number of samples: {len(stream['time_series'])}")
    print("-----")

# Find the first EEG stream and marker stream
eeg_stream = None
marker_stream = None

for stream in streams:
    if 'type' in stream['info'] and stream['info']['type'][0] == 'EEG':
        eeg_stream = stream
    elif 'type' in stream['info'] and stream['info']['type'][0] == 'Markers':
        marker_stream = stream

# Check if both streams were found
if eeg_stream is None or marker_stream is None:
    raise ValueError("EEG or Marker stream not found in the XDF file.")
else:
    print(f"Found EEG stream: {eeg_stream['info']['name'][0]}")
    print(f"Found Marker stream: {marker_stream['info']['name'][0]}")

# Extract EEG data and timestamps
eeg_data = np.array(eeg_stream['time_series'])  # Shape: (N_samples, N_channels)
eeg_timestamps = np.array(eeg_stream['time_stamps'])  # Shape: (N_samples,)

# Extract marker data and timestamps
marker_data = np.array([int(value[0]) for value in marker_stream['time_series']])  # Flatten marker values
marker_timestamps = np.array(marker_stream['time_stamps'])  # Timestamps for marker data

# Define the time window for EEG data
timewindow = 600  # Configurable time window in seconds
sampling_rate = 512  # Update this to the actual EEG sampling rate

# Define the start and end times for the selected time window
end_time = eeg_timestamps[-1]
start_time = max(eeg_timestamps[0], end_time - timewindow)  # Ensure start time is within bounds

# Mask for EEG data
eeg_mask = (eeg_timestamps >= start_time) & (eeg_timestamps <= end_time)

# Apply mask to EEG data and timestamps
eeg_data_window = eeg_data[eeg_mask, :]
eeg_timestamps_window = eeg_timestamps[eeg_mask]

# Mask for marker data
marker_mask = (marker_timestamps >= start_time) & (marker_timestamps <= end_time)
marker_data_window = marker_data[marker_mask]
marker_timestamps_window = marker_timestamps[marker_mask]

# Plot the EEG data from the first channel in the selected time window
plt.figure(figsize=(10, 5))

# Plot the EEG data for Channel 20 (example)
plt.plot(eeg_timestamps_window, eeg_data_window[:, 15], label='Channel 20')

# Plot vertical lines for markers
for i, marker_value in enumerate(marker_data_window):
    if marker_value == 100:
        plt.axvline(marker_timestamps_window[i], color='blue', linestyle='--', label="Marker 100" if i == 0 else "")
    elif marker_value == 200:
        plt.axvline(marker_timestamps_window[i], color='red', linestyle='--', label="Marker 200" if i == 0 else "")
    elif marker_value == 0:
        plt.axvline(marker_timestamps_window[i], color='green', linestyle='--', label="Marker 0" if i == 0 else "")

# Set labels and title
plt.xlabel('Timestamp (s)')
plt.ylabel('EEG Signal')
plt.title(f'EEG Data with Markers (Last {timewindow} seconds) - Channel 20')
plt.legend()

# Display the plot
plt.show()
