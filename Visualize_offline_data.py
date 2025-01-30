import os
import pyxdf
import numpy as np
import matplotlib.pyplot as plt
import mne
from matplotlib.cm import ScalarMappable # Add colorbar explicitly
from matplotlib.colors import Normalize  # Add colorbar explicitly
import matplotlib.pyplot as plt
from Utils.preprocessing import apply_car_filter, apply_notch_filter, butter_bandpass_filter, extract_segments, remove_eog_artifacts, separate_classes, compute_grand_average, parse_eeg_and_eog
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import config
from Utils.stream_utils import get_channel_names_from_xdf

sampling_rate = config.FS  # Update this to the actual EEG sampling rate


def import_montage(montage_name):

    # Define the path to the .fif montage file
    montage_path = os.path.join(os.path.dirname(__file__), 'Utils', montage_name)

    # Load the montage
    if os.path.exists(montage_path):
        montage = mne.channels.read_dig_fif(montage_path)
        print(f"Loaded montage from: {montage_path}")
    else:
        raise FileNotFoundError(f"Montage file not found at: {montage_path}")
    return montage



def compute_and_plot_psd(eeg_segmented, sampling_rate, channel_name, channel_names, nperseg=256):
    """
    Calculate and visualize PSD for a specific channel across trials.

    Parameters:
        eeg_segmented (np.ndarray): EEG data of shape (n_samples, n_channels, n_trials).
        sampling_rate (int): Sampling rate of the EEG data in Hz.
        channel_name (str): Name of the channel to analyze.
        channel_names (list): List of all channel names in the dataset.
        nperseg (int): Length of each segment for Welch's method.
    """
    # Find the index of the specified channel
    if channel_name not in channel_names:
        raise ValueError(f"Channel '{channel_name}' not found in the dataset.")
    
    channel_idx = channel_names.index(channel_name)

    # Extract data for the specified channel
    channel_data = eeg_segmented[:, channel_idx, :]  # Shape: (n_samples, n_trials)

    # Initialize lists to store PSD results
    psd_list = []
    freqs = None

    # Calculate PSD for each trial
    for trial in range(channel_data.shape[1]):
        f, Pxx = welch(channel_data[:, trial], fs=sampling_rate, nperseg=nperseg)
        psd_list.append(Pxx)
        if freqs is None:
            freqs = f  # Frequencies are the same for all trials

    # Convert PSD list to a 2D array (freqs x trials)
    psd_array = np.array(psd_list).T  # Shape: (n_freqs, n_trials)

    # Plot PSD as a heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(
        psd_array,
        aspect='auto',
        origin='lower',
        extent=[0, psd_array.shape[1], freqs[0], freqs[-1]],
        cmap='viridis'
    )
    plt.colorbar(label='Power Spectral Density (uV^2/Hz)')
    plt.title(f'PSD Heatmap for Channel: {channel_name}')
    plt.xlabel('Trial')
    plt.ylabel('Frequency (Hz)')
    plt.show()



def plot_markers_vs_time(marker_timestamps, marker_data):
    """
    Plot markers versus time with different colors for different marker value ranges.

    Parameters:
        marker_timestamps (np.ndarray): Array of marker timestamps.
        marker_data (np.ndarray): Array of marker values corresponding to the timestamps.
    """
    # Ensure the inputs are numpy arrays
    marker_timestamps = np.array(marker_timestamps)
    marker_data = np.array(marker_data)
    
    # Separate markers based on their value ranges
    marker_100_times = marker_timestamps[(marker_data >= 100) & (marker_data < 200)]
    marker_200_times = marker_timestamps[(marker_data >= 200) & (marker_data < 300)]
    marker_300_times = marker_timestamps[(marker_data >= 300) & (marker_data < 400)]

    marker_100_values = marker_data[(marker_data >= 100) & (marker_data < 200)]
    marker_200_values = marker_data[(marker_data >= 200) & (marker_data < 300)]
    marker_300_values = marker_data[(marker_data >= 300) & (marker_data < 400)]

    plt.figure(figsize=(10, 6))
    plt.scatter(marker_100_times, marker_100_values, color='blue', label='100s Markers', alpha=0.7)
    plt.scatter(marker_200_times, marker_200_values, color='red', label='200s Markers', alpha=0.7)
    plt.scatter(marker_300_times, marker_300_values, color='green', label='300s Markers', alpha=0.7)

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Marker Values', fontsize=12)
    plt.title('Markers vs Time', fontsize=14)
    plt.ylim(0, max(marker_data) + 50)  # Set the y-axis range dynamically
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_channel_timeseries(grand_avg_data, channel_name, channel_names, sampling_rate, time_offset=0):
    """
    Plot the grand-averaged EEG timeseries for a specific channel over time.

    Parameters:
        grand_avg_data (np.ndarray): Grand averaged data with shape (n_timepoints, n_channels).
        channel_name (str): Name of the channel to plot.
        channel_names (list of str): List of channel names corresponding to the data's second axis.
        sampling_rate (int): Sampling rate of the EEG data in Hz.
        time_offset (int): Time in milliseconds to include before the marker timestamp (default: 0).
    """
    # Ensure the specified channel is in the channel names
    if channel_name not in channel_names:
        raise ValueError(f"Channel '{channel_name}' not found in the provided channel names.")

    # Find the index of the specified channel
    channel_index = channel_names.index(channel_name)

    # Extract the timeseries for the specified channel
    channel_data = grand_avg_data[:, channel_index]

    # Generate the time vector in seconds
    n_timepoints = channel_data.shape[0]
    time_vector = np.linspace(0, n_timepoints / sampling_rate, n_timepoints)

    # Calculate the marker time in seconds if pre_marker_time_ms > 0
    marker_time_s = -time_offset / 1000 if time_offset < 0 else 0

    # Adjust the time vector to make the marker time the zero point
    adjusted_time_vector = time_vector - marker_time_s

    # Plot the timeseries
    plt.figure(figsize=(10, 6))
    plt.plot(adjusted_time_vector, channel_data, label=f'{channel_name} (Grand Avg)', color='b')
    plt.axhline(0, color='k', linestyle='--', linewidth=1)  # Add a baseline
    if time_offset < 0:
        plt.axvline(0, color='r', linestyle='--', linewidth=1, label='Marker Time')  # Marker time as zero point
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Amplitude (μV)', fontsize=14)
    plt.title(f'Grand Averaged Timeseries for {channel_name}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_topo(grand_average, montage):
    """
    Plot a topographic map using the grand average values.

    Parameters:
        grand_average (np.ndarray): Grand average data of shape (n_selected_channels,).
        montage (mne.channels.DigMontage): The loaded montage object.
    """

    # Get channel names from configuration
    grand_average_ch_names = config.EEG_CHANNEL_NAMES
    # Get channel positions
    ch_positions = montage.get_positions()["ch_pos"]
    montage_ch_names = list(ch_positions.keys())
    # Initialize an array for the updated grand average
    updated_grand_average = np.zeros(len(montage_ch_names))
    # Handle the "ALL" case
    if grand_average_ch_names == ["ALL"]:
        grand_average_ch_names = get_channel_names_from_xdf(eeg_stream)
    # Map grand_average values to montage channels
    for idx, ch_name in enumerate(montage_ch_names):
        ch_name = ch_name.upper() # uppercase so schemes match
        if ch_name in grand_average_ch_names:
            try:
                ga_idx = grand_average_ch_names.index(ch_name)
                updated_grand_average[idx] = grand_average[ga_idx]
            except ValueError:
                updated_grand_average[idx] = 0.0  # Channel not in grand_average, set to 0

    # Prepare data for the topomap
    pos = np.array([ch_positions[ch_name][:2] for ch_name in montage_ch_names])  # Extract (x, y) positions
    data = updated_grand_average  # Grand average values for each channel
    # Plot the topomap
    fig, ax = plt.subplots(figsize=(8, 8))
    im, _ = mne.viz.plot_topomap(
        data,
        pos,
        axes=ax,
        show=False,
        names=montage_ch_names,
        contours=0,  # Disable contour lines
        cmap='coolwarm'
    )

    # Create a ScalarMappable to manually add a colorbar
    norm = Normalize(vmin=data.min(), vmax=data.max())
    sm = ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])  # Required for matplotlib compatibility

    # Add the colorbar to the figure
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label("Grand Average (uV)", fontsize=12)
    fig.subplots_adjust(left=0.2, right=0.6, top=0.85, bottom=0.15)
    ax.set_title("Topographic Map (Grand Average)", fontsize=16)
    plt.show()


def display_eeg_channel_names(eeg_stream):
    """
    Display the channel names of the EEG stream.

    Parameters:
        eeg_stream (dict): The EEG stream from the XDF file.
    """
    if 'desc' in eeg_stream['info'] and 'channels' in eeg_stream['info']['desc'][0]:
        channel_desc = eeg_stream['info']['desc'][0]['channels'][0]['channel']
        channel_names = [channel['label'][0] for channel in channel_desc]
        print("EEG Stream Channel Names:")
        for idx, name in enumerate(channel_names, start=1):
            print(f"{idx}: {name}")
    else:
        print("Channel names not found in EEG stream metadata.")

# Directory containing XDF files
xdf_dir = '/home/arman-admin/Documents/CurrentStudy/sub-P001/ses-S001/eeg'

# Specify the XDF file
xdf_files = ['sub-P001_ses-S001_task-Default_run-001_eeg.xdf']

#xdf_files = config.DATA_FILE_PATH

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


# Extract channel names
channel_names = get_channel_names_from_xdf(eeg_stream)
print("Channel names:", channel_names)



# import montage
montage = import_montage('CA-209-dig.fif')
# Visualize the montage electrode locations
#fig = montage.plot(kind='topomap', show_names=True, show=True)  # 2D topographic view
#plt.show()


# Extract EEG data and timestamps
#eeg_data = np.array(eeg_stream['time_series'])  # Shape: (N_samples, N_channels)
eeg_timestamps = np.array(eeg_stream['time_stamps'])  # Shape: (N_samples,)

eeg_data, eog_data = parse_eeg_and_eog(eeg_stream, channel_names)

eeg_data = remove_eog_artifacts(eeg_data, eog_data) if config.EOG_TOGGLE == 1 else eeg_data

#apply filtering schemes before segmenting dataset
eeg_data = apply_notch_filter(eeg_data, sampling_rate)
eeg_data = butter_bandpass_filter(eeg_data, config.LOWCUT, config.HIGHCUT, sampling_rate, 4)
eeg_data = apply_car_filter(eeg_data)


# Extract marker data and timestamps
marker_data = np.array([int(value[0]) for value in marker_stream['time_series']])  # Flatten marker values
marker_timestamps = np.array(marker_stream['time_stamps'])  # Timestamps for marker data

# Define the time window for EEG data


'''
segmented = segment_data(
    eeg_data=eeg_data, 
    eeg_timestamps=eeg_timestamps, 
    marker_data=marker_data, 
    marker_timestamps=marker_timestamps, 
    time_window_ms=500,  # 500 ms
    sampling_rate=sampling_rate,    # 512 Hz
    pre_marker_time_ms=pre_trigger_window #200 ms before
)
'''
# Extract segments based on markers
print("Extracting EEG segments for timeseries view...")
timeseries_view_offset = -200
segments, labels = extract_segments(
    eeg_data, eeg_timestamps, marker_timestamps, marker_data,window_size_ms=1000, fs= config.FS, offset_ms=timeseries_view_offset
)
#segments = flatten_segments(segments)
class_1, class_2 = separate_classes(segments, labels)


# Assuming 'segmented' is the returned dictionary from the seperate_classes function
data_100 = class_1["data"]  # Access data for marker 100
data_200 = class_2["data"]  # Access data for marker 200

# Check if the data exists for each marker
if data_100 is not None:
    print(f"Data for marker 100: Shape {data_100.shape}")
else:
    print("No data found for marker 100.")

if data_200 is not None:
    print(f"Data for marker 200: Shape {data_200.shape}")
else:
    print("No data found for marker 200.")


GA_timeseries_100 = compute_grand_average(
    data=data_100,
    sampling_rate=sampling_rate,
    mode="trials" 
)
GA_timeseries_200 = compute_grand_average(
    data=data_200,
    sampling_rate=sampling_rate,
    mode="trials"  
)
print("GA trials 100 Shape:", GA_timeseries_100.shape)
print("GA trials 200 Shape:", GA_timeseries_200.shape)



plot_channel_timeseries(
    grand_avg_data=GA_timeseries_100,          # Grand averaged data for marker 100
    channel_name='FZ',                    # Plot for the channel 'Cz'
    channel_names=channel_names,          # List of channel names
    sampling_rate=sampling_rate,                    # EEG sampling rate in Hz
    time_offset=timeseries_view_offset                # 200 ms pre-marker time
)

plot_channel_timeseries(
    grand_avg_data=GA_timeseries_200,          # Grand averaged data for marker 100
    channel_name='FZ',                    # Plot for the channel 'Cz'
    channel_names=channel_names,          # List of channel names
    sampling_rate=sampling_rate,                    # EEG sampling rate in Hz
    time_offset=timeseries_view_offset                # 200 ms pre-marker time
)



# Extract segments based on markers
print("Extracting EEG segments for topo plots...")
segments, labels = extract_segments(
    eeg_data, eeg_timestamps, marker_timestamps, marker_data,window_size_ms=500, fs = config.FS, offset_ms=200
)
#segments = flatten_segments(segments)
class_1, class_2 = separate_classes(segments, labels)


# Assuming 'segmented' is the returned dictionary from the seperate_classes function
data_100 = class_1["data"]  # Access data for marker 100
data_200 = class_2["data"]  # Access data for marker 200


# Compute grand average in the range 100 ms to 500 ms
Topo_GA_100 = compute_grand_average(
    data=data_100,
    sampling_rate=sampling_rate,
    mode="trials_and_timepoints"  # Or "trials"
)

# Compute grand average in the range 100 ms to 500 ms
Topo_GA_200 = compute_grand_average(
    data=data_200,
    sampling_rate=sampling_rate,
    mode="trials_and_timepoints"  # Or "trials"
)


print("topo plot 100 Shape:", GA_timeseries_100.shape)
print("topo plot 200 Shape:", GA_timeseries_200.shape)



plot_topo(Topo_GA_100, montage)

plot_topo(Topo_GA_200, montage)

plot_markers_vs_time(marker_timestamps, marker_data)

# Example usage of the compute_and_plot_psd function
compute_and_plot_psd(
    eeg_segmented=data_100,  # Replace with your segmented data for marker 100
    sampling_rate=sampling_rate,                    # Replace with your actual sampling rate
    channel_name="FZ",                    # Replace with your desired channel (e.g., "Cz")
    channel_names=channel_names,          # Replace with the list of all channel names from your dataset
    nperseg=128                           # Replace with your desired segment length for Welch's method
)

# Example usage of the compute_and_plot_psd function
compute_and_plot_psd(
    eeg_segmented=data_200,  # Replace with your segmented data for marker 100
    sampling_rate=sampling_rate,                    # Replace with your actual sampling rate
    channel_name="FZ",                    # Replace with your desired channel (e.g., "Cz")
    channel_names=channel_names,          # Replace with the list of all channel names from your dataset
    nperseg=128                           # Replace with your desired segment length for Welch's method
)