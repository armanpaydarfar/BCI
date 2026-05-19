"""
UTIL_marker_stream.py

UDP -> LSL marker streaming bridge.

This script listens on a UDP port for marker messages produced by experiment
drivers, converts them into a LSL `MarkerStream`, and timestamps them using
the associated EEG LSL stream.

UDP message formats accepted:
- `"<marker_int>"` (marker only; uses EEG LSL time as timestamp)
- `"<marker_int>,<prob_mi>,<prob_rest>"` (marker + classifier probabilities)

Output LSL stream:
- Stream type/name: `Markers` / `MarkerStream`
- Channels: 4 floats:
  1) marker (float representation of int)
  2) timestamp (LSL time, via EEG inlet)
  3) prob_mi (float, or -1.0 default if not provided)
  4) prob_rest (float, or -1.0 default if not provided)
"""

import socket
import threading
import time
import queue
import logging
from datetime import datetime
from pathlib import Path
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream, local_clock
import config

# Module-level state populated by main(); kept at import so the module can be
# imported under pytest without creating log directories or opening an LSL
# outlet on the system. See Harmony_Test_Suite_Plan.md §5.1.a.
subject_log_dir = None
log_filename = None
outlet = None
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

def get_current_eeg_timestamp(inlet, udp_received_time=0, timeout=0.02):
    inlet.flush()
    local_timestamp = local_clock()  # fallback
    sample, timestamp = inlet.pull_sample(timeout=timeout)
    timestamp_after_pull = local_clock()

    if timestamp is not None:
        return timestamp
    else:
        logging.warning("⚠️ EEG timestamp unavailable — using local_clock() fallback.")
        return local_timestamp

def send_marker(marker, timestamp, prob_mi=-1.0, prob_rest=-1.0):
    outlet.push_sample([float(marker), timestamp, prob_mi, prob_rest])
    logging.info(f"Sent marker: {marker} at timestamp: {timestamp} "
                 f"| P(MI): {prob_mi:.3f}, P(REST): {prob_rest:.3f}")

def udp_listener(udp_port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", udp_port))
    logging.info(f"Listening for UDP marker messages on port {udp_port}...")

    while True:
        data, addr = sock.recvfrom(1024)
        udp_received_time = local_clock()
        try:
            message = data.decode('utf-8').strip()
            message_queue.put((message, udp_received_time, addr))
        except ValueError:
            logging.warning(f"Invalid UDP message: {data}")

def parse_marker_message(message):
    """Parse a UDP marker payload into ``(marker_value, prob_mi, prob_rest)``.

    Supported wire formats (see module docstring):
        ``"<marker_int>"``                          → (int, None, None)
        ``"<marker_int>,<prob_mi>,<prob_rest>"``    → (int, float, float)

    Raises ``ValueError`` for any other arity or unparseable component.
    Pure function — no I/O, no logging — so the parsing contract can be
    tested in isolation (Plan §6 #11).
    """
    parts = [p.strip() for p in message.strip().split(',')]
    if len(parts) == 1:
        return int(parts[0]), None, None
    if len(parts) == 3:
        return int(parts[0]), float(parts[1]), float(parts[2])
    raise ValueError(
        f"Unexpected UDP marker arity ({len(parts)} parts): {message!r}; "
        f"expected '<marker>' or '<marker>,<prob_mi>,<prob_rest>'."
    )


def process_udp_messages(eeg_inlet):
    while True:
        message, udp_received_time, addr = message_queue.get()
        logging.info(f"Processing UDP message: {message} from {addr}")

        try:
            marker_value, prob_mi, prob_rest = parse_marker_message(message)
            timestamp = get_current_eeg_timestamp(eeg_inlet, udp_received_time)
            if prob_mi is None:
                send_marker(marker_value, timestamp)
            else:
                send_marker(marker_value, timestamp, prob_mi, prob_rest)

        except ValueError as e:
            logging.warning(f"Failed to parse UDP marker: {message} | Error: {e}")

        message_queue.task_done()

def handle_udp_requests(eeg_inlet):
    local_port = config.UDP_MARKER["PORT"]

    threading.Thread(target=udp_listener, args=(local_port,), daemon=True).start()
    threading.Thread(target=process_udp_messages, args=(eeg_inlet,), daemon=True).start()

    logging.info(f"Marker utility started on local port {local_port}.")

def _setup_logging_and_outlet():
    """Create the subject log directory, attach the file/console log handlers,
    and open the `MarkerStream` LSL outlet. Side effects deferred to main()
    so the module can be imported without hardware (Plan §5.1.a)."""
    global subject_log_dir, log_filename, outlet

    subject_log_dir = Path(config.DATA_DIR) / f"sub-{config.TRAINING_SUBJECT}" / "marker_logs"
    subject_log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = subject_log_dir / f"marker_utility_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler()
        ]
    )

    info = StreamInfo('MarkerStream', 'Markers', 4, 0, 'float32', 'marker_stream_id')
    outlet = StreamOutlet(info)


def main():
    _setup_logging_and_outlet()

    eeg_inlet = get_eeg_inlet()
    handle_udp_requests(eeg_inlet)

    logging.info("Marker utility is running (only streaming markers). Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Exiting marker utility.")

if __name__ == "__main__":
    main()
