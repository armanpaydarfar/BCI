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

def get_channel_names_from_xdf(eeg_stream):
    """
    Extract channel names from an EEG stream in a pyxdf file.

    Parameters:
        eeg_stream (dict): EEG stream from the loaded pyxdf file.

    Returns:
        list: A list of channel names.
    """
    if 'desc' in eeg_stream['info'] and 'channels' in eeg_stream['info']['desc'][0]:
        channel_desc = eeg_stream['info']['desc'][0]['channels'][0]['channel']
        channel_names = [channel['label'][0] for channel in channel_desc]
        return channel_names
    else:
        raise ValueError("Channel names not found in EEG stream metadata.")


def get_channel_names_from_lsl(stream_type='EEG'):
    """
    Retrieve channel names from an LSL stream.

    Parameters:
        stream_type (str): The type of stream to resolve (default is 'EEG').

    Returns:
        list: A list of channel names from the resolved LSL stream.
    """
    print(f"Looking for a {stream_type} stream...")

    # Resolve the stream
    streams = resolve_stream('type', stream_type)
    if not streams:
        raise RuntimeError(f"No {stream_type} stream found.")

    # Create an inlet to the first available stream
    inlet = StreamInlet(streams[0])

    # Get stream info and channel names
    stream_info = inlet.info()
    desc = stream_info.desc()
    channel_names = []

    # Parse the channel names from the stream description
    channels = desc.child('channels').child('channel')
    while channels.name() == 'channel':
        channel_names.append(channels.child_value('label'))
        channels = channels.next_sibling()

    return channel_names