"""
Utils for extracting EEG data from LSL (live) or XDF (recorded offline).

These helpers are used by different experiment workflows to:
- verify required streams exist,
- pull small fixed-size chunks from LSL,
- and load synchronized EEG/marker streams from XDF files.

Be mindful of sample layout:
- In most callers, downstream code expects arrays shaped like (n_channels, n_samples)
  for real-time windows, and `(n_samples, ...)` for certain offline operations.
"""

import sys
from pylsl import resolve_stream, resolve_byprop, StreamInlet
import numpy as np
import pyxdf

def require_marker_stream(logger=None, timeout=5.0):
    """
    Verify that the experiment MarkerStream LSL outlet is active.

    This is a hard startup guard for online experiment drivers.  It ensures
    UTIL_marker_stream.py is running and publishing before any trial data is
    recorded.  If the stream is not found within `timeout` seconds, an error
    is logged and the process exits.

    Parameters
    ----------
    logger : LoggerManager, optional
        If provided, errors are routed through logger.log_event(); otherwise
        they go to stdout/stderr.
    timeout : float
        Seconds to wait for the stream before giving up (default 5.0).
    """
    def _log(msg):
        if logger is not None:
            logger.log_event(msg)
        else:
            print(msg)

    _log("Checking for MarkerStream LSL outlet (UTIL_marker_stream.py)...")
    found = resolve_byprop('name', 'MarkerStream', minimum=1, timeout=timeout)
    if not found:
        msg = (
            "STARTUP ERROR: MarkerStream not found on the LSL network. "
            "UTIL_marker_stream.py must be running before the experiment starts. "
            "Experiment markers will not be recorded — cannot proceed."
        )
        _log(msg)
        sys.exit(1)
    _log("✅ MarkerStream detected — marker recording is active.")


def check_streams():
    """
    Resolve the expected EEG LSL stream and return an inlet for pulling samples.
    """
    print("Checking for EEG stream...")
    eeg_streams = resolve_stream('type', 'EEG')
    if not eeg_streams:
        print("Error: EEG stream not found.")
        exit(1)
    print("EEG stream is active.")
    return StreamInlet(eeg_streams[0])

def get_eeg_data(inlet, duration=1.0, sampling_rate=512):
    """
    Pull approximately `duration` seconds of EEG samples from an LSL inlet.

    Returns:
        numpy array of shape (n_samples, n_channels) as returned by LSL chunk pull.
        (Callers typically reshape/transposes as needed.)
    """
    inlet.flush()
    samples = int(duration * sampling_rate)
    data, _ = inlet.pull_chunk(timeout=duration + 0.5, max_samples=samples)
    return np.array(data[-samples:])

def load_xdf(file_path, dejitter=False, sync=False, report=True):
    """
    Load an XDF file and return `(eeg_stream, marker_stream)` dictionaries.

    This function searches stream metadata by type:
    - EEG: typ == "eeg"  (must have nominal_srate > 0)
    - markers: typ in {"markers", "marker"}, preferring name == "MarkerStream"
      when multiple candidate streams share the markers type.  This preference
      is needed when a file contains both a hardware amplifier marker channel
      (e.g. 'eegoSports-000104_markers', 1 channel) and the experiment marker
      stream produced by UTIL_marker_stream.py ('MarkerStream', 4 channels).

    The function also optionally prints EEG sampling interval statistics when
    `report=True`.

    Amplitude convention (Harmony): `eeg_stream["time_series"]` is treated throughout
    the repo as **microvolt-scale** numbers (typical LSL/amplifier exports), not SI volts.
    Do not multiply or divide by 1e-6 unless you have a file-specific reason. See
    `visualize_online_data.py` module docstring and `config.ARTIFACT_*` / training pipelines.
    """
    streams, _ = pyxdf.load_xdf(
        file_path,
        dejitter_timestamps=dejitter,
        synchronize_clocks=sync
    )

    eeg_stream = None
    marker_stream = None
    marker_stream_fallback = None  # first markers-type stream that is not "MarkerStream"
    for s in streams:
        typ  = s["info"].get("type", [""])[0].lower()
        name = s["info"].get("name", [""])[0]
        if eeg_stream is None and typ == "eeg":
            eeg_stream = s
        if typ in ("markers", "marker"):
            if name == "MarkerStream":
                # Prefer the named experiment marker stream unconditionally.
                marker_stream = s
            elif marker_stream_fallback is None:
                marker_stream_fallback = s

    # Use fallback (e.g. hardware marker channel) only when no "MarkerStream" found.
    if marker_stream is None:
        marker_stream = marker_stream_fallback

    if eeg_stream is None or marker_stream is None:
        raise ValueError("Both EEG and Marker streams must be present in the XDF file.")

    if report:
        info = eeg_stream["info"]
        name = info.get("name", [""])[0]
        src  = info.get("source_id", [""])[0]
        try:
            nominal = float(info.get("nominal_srate", ["nan"])[0])
        except Exception:
            nominal = float("nan")

        ts = np.asarray(eeg_stream["time_stamps"], dtype=float)
        dt = np.diff(ts)
        eff = 1.0 / np.median(dt) if dt.size else float("nan")

        if dt.size:
            # ---- config knobs ----
            spike_factor = 2.0          # dt > spike_factor * median => spike
            tol_ms = 0.001              # on-target tolerance in ms (±0.001 ms)
            # -----------------------

            info = eeg_stream["info"]
            try:
                nominal = float(info.get("nominal_srate", ["nan"])[0])
            except Exception:
                nominal = float("nan")

            dt_ms   = dt * 1e3
            n       = dt_ms.size
            med_ms  = float(np.median(dt_ms))
            mean_ms = float(np.mean(dt_ms))
            std_ms  = float(np.std(dt_ms))
            q01_ms  = float(np.quantile(dt_ms, 0.01))
            q25_ms  = float(np.quantile(dt_ms, 0.25))
            q75_ms  = float(np.quantile(dt_ms, 0.75))
            q99_ms  = float(np.quantile(dt_ms, 0.99))
            iqr_ms  = q75_ms - q25_ms
            min_ms  = float(dt_ms.min())
            max_ms  = float(dt_ms.max())

            eff_fs_med = 1000.0 / med_ms if med_ms > 0 else float("nan")

            neg      = int(np.sum(dt_ms < 0))
            neg_pct  = 100.0 * neg / n
            spikes   = int(np.sum(dt_ms > spike_factor * med_ms))
            spikes_pct = 100.0 * spikes / n

            if np.isfinite(nominal) and nominal > 0:
                exp_dt_ms    = 1000.0 / nominal
                delta_med_ms = med_ms - exp_dt_ms       # how far median dt is from ideal
                ppm          = 1e6 * (delta_med_ms / exp_dt_ms)
            else:
                exp_dt_ms = delta_med_ms = ppm = float("nan")

            correct = int(np.sum(np.abs(dt_ms - exp_dt_ms) <= tol_ms))
            correct_pct = 100.0 * correct / n

            # Trimmed mean rate (ignore big tails)
            mask_trim = (dt_ms > 0.5 * exp_dt_ms) & (dt_ms < 1.5 * exp_dt_ms)
            eff_fs_trim = 1000.0 / float(np.mean(dt_ms[mask_trim])) if np.any(mask_trim) else float("nan")

            name = info.get("name", [""])[0]; src = info.get("source_id", [""])[0]
            print(f"[EEG] name={name}  source_id={src}  nominal={nominal:.3f} Hz  effective (median)≈{eff_fs_med:.3f} Hz")
            print(f"[EEG ts stats] n_dt={n} | median={med_ms:.6f} ms | mean={mean_ms:.6f} ms | std={std_ms:.3f} ms | IQR={iqr_ms:.3f} ms")
            print(f"[EEG ts tails] min/1%/99%/max = [{min_ms:.6f}, {q01_ms:.6f}, {q99_ms:.6f}, {max_ms:.6f}] ms")
            print(f"[EEG dt quality] on-target(±{tol_ms:.3f} ms) = {correct}/{n} ({correct_pct:.2f}%) | "
                f"negatives = {neg} ({neg_pct:.2f}%) | spikes(>{spike_factor}×median) = {spikes} ({spikes_pct:.2f}%)")
            print(f"[EEG vs nominal] expected={exp_dt_ms:.6f} ms | Δmedian={delta_med_ms:.6f} ms ({ppm:.1f} ppm) | mean_trimmed≈{eff_fs_trim:.3f} Hz")



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