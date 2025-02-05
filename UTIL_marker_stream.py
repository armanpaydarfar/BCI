import socket
import threading
import time
import queue  # ✅ Import queue for ordered processing
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream, local_clock
import config  # Import configuration file

# Define marker values to be sent
possible_marker_values = [int(value) for value in config.TRIGGERS.values()]
print(f"Possible marker values: {possible_marker_values}")

# Create LSL stream for markers
info = StreamInfo('MarkerStream', 'Markers', 2, 0, 'float32', 'marker_stream_id')  # Sending marker and timestamp as a 2-element sample
outlet = StreamOutlet(info)

# ✅ Queue to store UDP messages in FIFO order
message_queue = queue.Queue()

def get_eeg_inlet():
    """
    Resolve the EEG stream and return a StreamInlet for continuous access.
    """
    print("Resolving EEG stream...")
    streams = resolve_stream('type', 'EEG')  # Find an EEG stream
    inlet = StreamInlet(streams[0])  # Create the inlet
    print("EEG stream connected.")
    return inlet

def get_current_eeg_timestamp(inlet, udp_received_time=0):
    """
    Pull the most recent sample and timestamp aligned with the current time.
    """
    inlet.flush()  # Flush the buffer to discard old samples
    sample, timestamp = inlet.pull_sample(timeout=1.0)  # Wait for a fresh sample
    temp_timestamp = local_clock()

    if timestamp is not None:
        # Compare the LSL stream timestamp with the local system clock
        stream_offset = temp_timestamp - udp_received_time
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

def udp_listener(udp_port):
    """
    Listens for UDP messages on a given port and adds them to the queue.

    Parameters:
        udp_port (int): The UDP port to listen on.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", udp_port))  # Bind to all interfaces
    print(f"Listening for UDP on port {udp_port}...")

    while True:
        data, addr = sock.recvfrom(1024)  # Receive up to 1024 bytes
        udp_received_time = local_clock()
        try:
            message = data.decode('utf-8').strip()

            # Handle ACK messages separately
            if message.startswith("ACK:"):
                print(f"Received ACK message: {message} from {addr}")
                continue  # Skip further processing

            # ✅ Add UDP message to the processing queue
            message_queue.put((message, udp_received_time, addr))

        except ValueError:
            print(f"Invalid UDP message: {data}")

def process_udp_messages(eeg_inlet):
    """
    Process UDP messages from the queue in order.

    Parameters:
        eeg_inlet (StreamInlet): The EEG stream inlet.
    """
    while True:
        message, udp_received_time, addr = message_queue.get()  # Retrieve next message

        print(f"Processing UDP message: {message} from {addr}")

        try:
            marker_value = int(message)
            if marker_value in possible_marker_values:
                timestamp = get_current_eeg_timestamp(eeg_inlet, udp_received_time)
                if timestamp is not None:
                    send_marker(marker_value, timestamp)
                else:
                    print("Failed to retrieve EEG timestamp. Marker not sent.")
            else:
                print(f"Invalid marker value received via UDP: {marker_value}")
        
        except ValueError:
            print(f"Invalid UDP message format: {message}")

        message_queue.task_done()  # ✅ Mark task as done

def handle_udp_requests(eeg_inlet):
    """
    Start UDP servers to listen on multiple ports.

    Parameters:
        eeg_inlet (StreamInlet): The EEG stream inlet.
    """
    local_port = config.UDP_MARKER["PORT"]  # Get local UDP port
    robot_port = config.UDP_ROBOT["PORT"]  # Get robot UDP port

    # ✅ Start UDP listener threads (Only Collects Messages)
    threading.Thread(target=udp_listener, args=(local_port,), daemon=True).start()
    threading.Thread(target=udp_listener, args=(robot_port,), daemon=True).start()
    
    # ✅ Start Single Processing Thread to Ensure Ordered Execution
    threading.Thread(target=process_udp_messages, args=(eeg_inlet,), daemon=True).start()

    print(f"UDP servers started for local (port {local_port}) and robot (port {robot_port}).")

def main():
    """
    Main function to set up the EEG inlet and start the UDP server.
    """
    eeg_inlet = get_eeg_inlet()  # Set up the EEG stream inlet
    handle_udp_requests(eeg_inlet)  # Start UDP listeners

    print("Marker utility is running. Send UDP messages to send markers.")
    print("Use Ctrl+C to exit.")
    
    try:
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("Exiting marker utility.")

if __name__ == "__main__":
    main()
