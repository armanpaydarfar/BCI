import socket
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream, local_clock
import threading
import time

# Define marker values to be sent
possible_marker_values = [0, 100, 200]

print(f"Possible marker values: {possible_marker_values}")

# Create LSL stream for markers
info = StreamInfo('MarkerStream', 'Markers', 2, 0, 'float32', 'marker_stream_id')  # Sending marker and timestamp as a 2-element sample
outlet = StreamOutlet(info)

def get_eeg_inlet():
    """
    Resolve the EEG stream and return a StreamInlet for continuous access.
    """
    print("Resolving EEG stream...")
    streams = resolve_stream('type', 'EEG')  # Find an EEG stream
    inlet = StreamInlet(streams[0])  # Create the inlet
    print("EEG stream connected.")
    return inlet

def get_current_eeg_timestamp(inlet):
    """
    Pull the most recent sample and timestamp aligned with the current time.
    """
    inlet.flush()  # Flush the buffer to discard old samples
    sample, timestamp = inlet.pull_sample(timeout=1.0)  # Wait for a fresh sample
    if timestamp is not None:
        # Compare the LSL stream timestamp with the local system clock
        stream_offset = local_clock() - timestamp
        print(f"Current EEG timestamp: {timestamp}, Stream Offset: {stream_offset:.4f} seconds")
        return timestamp
    else:
        print("No new EEG timestamp available.")
        return None

def send_marker(value, timestamp):
    """
    Send the marker value and timestamp to an LSL stream and print for debugging purposes.
    
    Parameters:
        value (int): The marker value to send.
        timestamp (float): The corresponding EEG timestamp to sync with.
    """
    if value in possible_marker_values:
        outlet.push_sample([float(value), timestamp])  # Send marker and timestamp via LSL
        print(f"Sent marker: {value} at timestamp: {timestamp}")
    else:
        print(f"Invalid marker value: {value}. Allowed values are {possible_marker_values}.")

def handle_udp_requests(eeg_inlet, udp_port=12345):
    """
    Start a UDP server to listen for marker commands.
    
    Parameters:
        eeg_inlet (StreamInlet): The EEG stream inlet.
        udp_port (int): The UDP port to listen on.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", udp_port))
    print(f"UDP server is listening on port {udp_port}...")

    while True:
        data, addr = sock.recvfrom(1024)  # Receive up to 1024 bytes
        try:
            message = data.decode('utf-8').strip()
            print(f"Received UDP message: {message} from {addr}")
            marker_value = int(message)
            if marker_value in possible_marker_values:
                timestamp = get_current_eeg_timestamp(eeg_inlet)
                if timestamp is not None:
                    send_marker(marker_value, timestamp)
                else:
                    print("Failed to retrieve EEG timestamp. Marker not sent.")
            else:
                print(f"Invalid marker value received via UDP: {marker_value}")
        except ValueError:
            print(f"Invalid UDP message: {data}")

def main():
    """
    Main function to set up the EEG inlet and start the UDP server.
    """
    eeg_inlet = get_eeg_inlet()  # Set up the EEG stream inlet
    udp_thread = threading.Thread(target=handle_udp_requests, args=(eeg_inlet,), daemon=True)
    udp_thread.start()

    print("Marker utility is running. Send UDP messages to send markers.")
    print("Use Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("Exiting marker utility.")

if __name__ == "__main__":
    main()
