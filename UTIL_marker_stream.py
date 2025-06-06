import socket
import threading
import time
import queue
import logging
from datetime import datetime
from pathlib import Path
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream, local_clock
import config

# ─────────────────────────────────────────────────────────
# SET UP LOGGING INSIDE SUBJECT FOLDER
# ─────────────────────────────────────────────────────────

# Determine correct subject log directory
subject_log_dir = Path(config.DATA_DIR) / f"sub-{config.TRAINING_SUBJECT}" / "marker_logs"
subject_log_dir.mkdir(parents=True, exist_ok=True)

# Timestamped log filename
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = subject_log_dir / f"marker_utility_{timestamp}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w'),
        logging.StreamHandler()
    ]
)

# ─────────────────────────────────────────────────────────
# SETUP STREAM AND QUEUE
# ─────────────────────────────────────────────────────────

info = StreamInfo('MarkerStream', 'Markers', 4, 0, 'float32', 'marker_stream_id')
outlet = StreamOutlet(info)

message_queue = queue.Queue()

# ─────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────

def get_eeg_inlet():
    logging.info("Resolving EEG stream...")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    logging.info("EEG stream connected.")
    return inlet

def get_current_eeg_timestamp(inlet, udp_received_time=0):
    inlet.flush()
    local_timestamp = local_clock()
    sample, timestamp = inlet.pull_sample(timeout=1.0)
    timestamp_after_pull = local_clock()
    if timestamp is not None:
        stream_offset_sec = timestamp_after_pull - udp_received_time
        stream_offset_ms = stream_offset_sec * 1000  # Convert to milliseconds
        logging.info(f"Current EEG timestamp: {timestamp}, Stream Offset: {stream_offset_ms:.2f} ms")
        return timestamp
    else:
        logging.warning("EEG Timestamp not parsed - using local_clock() as fallback.")
        return local_timestamp


def send_marker(marker, timestamp, prob_mi=-1.0, prob_rest=-1.0):
    outlet.push_sample([float(marker), timestamp, prob_mi, prob_rest])
    logging.info(f"Sent marker: {marker} at timestamp: {timestamp} | P(MI): {prob_mi:.3f}, P(REST): {prob_rest:.3f}")

def udp_listener(udp_port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", udp_port))
    logging.info(f"Listening for UDP on port {udp_port}...")

    while True:
        data, addr = sock.recvfrom(1024)
        udp_received_time = local_clock()
        try:
            message = data.decode('utf-8').strip()

            if message.startswith("ACK:"):
                logging.info(f"Received ACK message: {message} from {addr}")
                continue

            message_queue.put((message, udp_received_time, addr))

        except ValueError:
            logging.warning(f"Invalid UDP message: {data}")


def process_udp_messages(eeg_inlet):
    while True:
        message, udp_received_time, addr = message_queue.get()
        logging.info(f"Processing UDP message: {message} from {addr}")

        try:
            parts = message.strip().split(',')

            if len(parts) == 1:
                # Case 1: Just a marker
                marker_value = int(parts[0])
                timestamp = get_current_eeg_timestamp(eeg_inlet, udp_received_time)
                if timestamp is not None:
                    send_marker(marker_value, timestamp)
                else:
                    logging.warning("Failed to retrieve EEG timestamp. Marker sent using local_clock().")

            elif len(parts) == 3:
                # Case 2: Marker + P(MI) + P(REST)
                marker_value = int(parts[0])
                prob_mi = float(parts[1])
                prob_rest = float(parts[2])
                timestamp = get_current_eeg_timestamp(eeg_inlet, udp_received_time)
                if timestamp is not None:
                    send_marker(marker_value, timestamp, prob_mi, prob_rest)
                else:
                    logging.warning("Failed to retrieve EEG timestamp. Marker not sent.")

            else:
                logging.warning(f"Unexpected UDP message format: {message}")

        except Exception as e:
            logging.warning(f"Failed to parse UDP message: {message} | Error: {e}")

        message_queue.task_done()



def handle_udp_requests(eeg_inlet):
    local_port = config.UDP_MARKER["PORT"]
    robot_port = config.UDP_ROBOT["PORT"]

    threading.Thread(target=udp_listener, args=(local_port,), daemon=True).start()
    threading.Thread(target=udp_listener, args=(robot_port,), daemon=True).start()
    threading.Thread(target=process_udp_messages, args=(eeg_inlet,), daemon=True).start()

    logging.info(f"UDP servers started for local (port {local_port}) and robot (port {robot_port}).")

def main():
    eeg_inlet = get_eeg_inlet()
    handle_udp_requests(eeg_inlet)

    logging.info("Marker utility is running. Send UDP messages to send markers.")
    logging.info("Use Ctrl+C to exit.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Exiting marker utility.")

if __name__ == "__main__":
    main()
