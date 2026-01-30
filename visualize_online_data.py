#!/usr/bin/env python3
# =========================================================
# ERD/ERS Multi-session analysis with:
#   ✅ Epoch rejection (peak-to-peak EEG amplitude)
#   ✅ Channel subset selection on import
#   ✅ TFR padding + crop to fix edge artifacts
#   ✅ CSD applied AFTER epoching (recommended w/ rejection)
#   ✅ Toggle plotting/metrics in % space or logratio space
#
# NEW FEATURES ADDED (toggleable):
#   ✅ (1) Per-session focal electrodes (C3 then CP1 then C4 ...)
#   ✅ (2) Auto-drop "bad" channels that dominate epoch rejections,
#          then re-epoch + re-run session (with safeguards)
#
# NOTE: Feature #3 (per-session start/stop indices) intentionally NOT added yet.
# =========================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import stats
import config

# Custom utility functions
from Utils.preprocessing import concatenate_streams
from Utils.stream_utils import get_channel_names_from_xdf, load_xdf

# =========================================================
# User Config
# =========================================================

subject = "CLIN_SUBJ_003"

# ---- Single-session mode ----
session = "S004ONLINE"
PROMPT_FOR_FILE_SELECTION = False

# ---- Multi-session mode ----
MULTI_SESSION_MODE = False
SESSION_LIST = ["S001ONLINE", "S002ONLINE", "S003ONLINE", "S004ONLINE", "S005ONLINE"]
#SESSION_LIST = ["S002ONLINE",  "S004ONLINE"]
# ---- Frequency band ----
BROADBAND_LOW = 4
BROADBAND_HIGH = 40
FREQ_BAND = "mu"  # "mu" or "beta"
BANDS = {
    "mu": (8, 13),
    "beta": (13, 30),
}
lowband, highband = BANDS[FREQ_BAND]
FREQS = np.linspace(lowband, highband, int(highband - lowband) + 1)
N_CYCLES = FREQS/2



# ---- Markers to analyze ----
MARKERS_FOR_ANALYSIS = ["100", "200"]

# ---- Time configuration (PLOT window / target window) ----
time_start = -1.0
baseline_period = 1.0
window_length = 5.0
time_end = time_start + window_length  # e.g. -1 -> +4
feedback_time = 1.0

# ---- Padding for TFR edge artifacts ----
PAD_TFR = 1.0  # seconds (epochs will be [-1-PAD, 4+PAD], then TFR cropped to [-1,4])

# =========================================================
# Focal vs. normalization electrodes
# =========================================================

# ------------------------------
# OLD behavior (single focal set)
# ------------------------------
FOCAL_ELECTRODES = ["CP5"]

# ------------------------------
# NEW: per-session focal mapping
# Toggle via USE_SESSION_SPECIFIC_FOCAL
# ------------------------------
USE_SESSION_SPECIFIC_FOCAL = False

FOCAL_ELECTRODES_BY_SESSION = {
    # Example defaults (edit these however you like):
    "S001ONLINE": ["CP1"],
    "S002ONLINE": ["CP1", "C3"],
    "S003ONLINE": ["C4"],
    "S004ONLINE": ["P3"],
    "S005ONLINE": ["C3"],
}

# If a session isn’t in the dict, fallback to this (or to FOCAL_ELECTRODES)
DEFAULT_FOCAL_ELECTRODES = FOCAL_ELECTRODES

MOTOR_NORM_ELECTRODES = [
    'FC1', 'FC2', 'C3', 'Cz', 'C4',
    'CP5', 'CP1', 'CP2', 'CP6',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'POz'
]

# ---- Variability method for timecourses ----
VAR_METHOD = "sem"  # "sem" or "std"

# ---- Scalar ERD metric window (for bar plots) ----
SCALAR_WINDOW = (1.0, 4.0)

# ---- Error bars on paired bars ----
BAR_ERROR_METHOD = "sem"  # "sem" or "std"

# ---- Toggles ----
DO_TOPO_MAPS = True
DO_FOCAL_TIMECOURSE = True

# ---- Multi-session overlay timecourse plot ----
DO_MULTISESSION_TIMECOURSE_OVERLAY = True
OVERLAY_SHOW_SHADING = True
LINE_YLIM = None  # or None

# ---- Bar plot normalization toggle ----
BAR_USE_NORMALIZATION = False
BAR_NORM_METHOD = "ratio"  # "ratio" or "difference"
BAR_YLIM = (-100, 100)  # or None

# ---- Epoch rejection ----
REJECT_EPOCHS = True
REJECT_P2P_UV = 150  # typical: 100–250 µV
FLAT_UV = None       # optional

# =========================================================
# NEW: Auto-drop bad channels that dominate rejections
# =========================================================

AUTO_DROP_BAD_CHANNELS = True
AUTO_DROP_REJECT_FRAC = 0.75        # if >90% epochs dropped -> consider auto-drop
AUTO_DROP_DOMINANCE_FRAC = 0.60     # culprit channel must explain >=60% of dropped epochs
AUTO_DROP_MAX_ITERS = 4             # maximum drop-and-retry loops
AUTO_DROP_MAX_CHANNELS_TOTAL = 4    # do not drop more than this many channels per session

# What to do if the session’s focal channel gets dropped:
#   "fallback" -> use DEFAULT_FOCAL_ELECTRODES (or old FOCAL_ELECTRODES) for that session
#   "skip"     -> skip focal plots/metrics for that session if focal is missing
FOCAL_IF_DROPPED_POLICY = "fallback"  # "fallback" or "skip"

# ---- Channel selection ----
USE_SUBSET_CHANNELS = False
CHANNEL_SUBSET = [
    # Frontal / premotor + SMA
    "F3", "Fz", "F4",
    "FC5","FC1", "FC2","FC6",
    # Primary motor strip
    "C3", "Cz", "C4",
    # Somatosensory / parietal around motor
    "CP5", "CP1", "CP2", "CP6",
    # Posterior parietal midline
    "P7", "P3", "Pz", "P4", "P8",
    "POz",
]

# ---- CSD toggle ----
APPLY_CSD = True

# ---- Plot/metric representation toggle ----
#   "percent"  -> plot metrics as % change from baseline
#   "logratio" -> plot metrics in logratio units
PLOT_SPACE = "percent"  # "percent" or "logratio"

# ---- Unit handling for XDF EEG stream ----
# If your eeg_stream["time_series"] is already in VOLTS, set this False.
# If it's in microvolts, set this True (we convert µV -> V).
XDF_DATA_IN_UV = False

# ---- Optional: limiting to specific index range (kept from your script) ----
LIMIT_EPOCH_RANGE = False
EPOCH_RANGE_START_IDX = 0
EPOCH_RANGE_END_IDX = 70


# =========================================================
# Helper functions
# =========================================================

def logratio_to_percent_change(x):
    """Convert log10 power ratio to % change from baseline."""
    return 100.0 * (10.0 ** x - 1.0)

def maybe_convert_for_plot(x_logratio):
    """Convert logratio -> percent for plotting/metrics if requested."""
    if PLOT_SPACE == "percent":
        return logratio_to_percent_change(x_logratio)
    return x_logratio

def ylabel_for_space():
    return "ERD %" if PLOT_SPACE == "percent" else "ERD (logratio)"

def get_error(arr, method):
    """Return SEM or STD along axis=0 for 2D array (trials x something)."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        if arr.shape[0] <= 1:
            return 0.0
        if method.lower() == "sem":
            return arr.std(ddof=1) / np.sqrt(arr.shape[0])
        elif method.lower() == "std":
            return arr.std(ddof=1)
        else:
            raise ValueError(f"Unknown method for variability: {method}")

    if arr.shape[0] <= 1:
        return np.zeros(arr.shape[1], dtype=float)

    if method.lower() == "sem":
        return arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    elif method.lower() == "std":
        return arr.std(axis=0, ddof=1)
    else:
        raise ValueError(f"Unknown method for variability: {method}")

def get_focal_for_session(sess: str, raw_or_tfr_ch_names=None):
    """
    Returns the focal electrode list to use for a given session,
    honoring USE_SESSION_SPECIFIC_FOCAL and fallbacks.

    If raw_or_tfr_ch_names is provided, we filter to those present.
    """
    if USE_SESSION_SPECIFIC_FOCAL:
        focals = FOCAL_ELECTRODES_BY_SESSION.get(sess, DEFAULT_FOCAL_ELECTRODES)
    else:
        focals = FOCAL_ELECTRODES

    # Fallback if someone accidentally sets empty list
    if not focals:
        focals = DEFAULT_FOCAL_ELECTRODES if DEFAULT_FOCAL_ELECTRODES else FOCAL_ELECTRODES

    if raw_or_tfr_ch_names is not None:
        focals_present = [ch for ch in focals if ch in raw_or_tfr_ch_names]
        return focals_present
    return list(focals)

def summarize_drop_log_channel_counts(drop_log):
    """
    Given epochs.drop_log, tally channel names in drop reasons.
    Returns dict channel->count and total_dropped.
    """
    chan_counts = {}
    dropped = 0

    for entry in drop_log:
        # entry is a tuple of reason strings (or empty tuple if kept)
        if not entry:
            continue
        dropped += 1
        for reason in entry:
            # Common patterns:
            #  - "EEG: C3" or "C3" or "BAD_artifact" etc.
            # We conservatively extract tokens that match actual channel names later.
            # Here we just count raw reason strings; caller will intersect with ch_names.
            chan_counts[reason] = chan_counts.get(reason, 0) + 1

    return chan_counts, dropped

def pick_dominant_bad_channel(epochs: mne.Epochs, dominance_frac: float):
    """
    Find a dominant channel causing rejections using epochs.drop_log.
    Returns (bad_channel_name, frac_of_dropped) or (None, 0.0).
    """
    drop_log = epochs.drop_log
    ch_names = set(epochs.ch_names)

    # Map: channel -> count of epochs where that channel appears in rejection reasons
    counts = {ch: 0 for ch in epochs.ch_names}
    total_dropped = 0

    for entry in drop_log:
        if not entry:
            continue
        total_dropped += 1
        # entry contains strings; channels may appear as exact channel names,
        # or as "EEG: C3" etc. We'll search for channel names as substrings.
        entry_str = " | ".join(entry)
        for ch in counts.keys():
            if ch in entry_str:
                counts[ch] += 1

    if total_dropped == 0:
        return None, 0.0

    bad_ch = max(counts, key=lambda k: counts[k])
    frac = counts[bad_ch] / total_dropped if total_dropped > 0 else 0.0

    if counts[bad_ch] == 0:
        return None, 0.0

    if frac >= dominance_frac:
        return bad_ch, frac
    return None, frac


# =========================================================
# Plotting: multi-session overlay timecourses
# =========================================================

def plot_multisession_overlay_timecourses_cached(session_cache):
    """
    Overlay focal timecourses for each session on one plot (cached).
    - MI and REST plotted separately (two subplots)
    - Each session is one line
    - Shading optional (OVERLAY_SHOW_SHADING)
    """
    if not DO_MULTISESSION_TIMECOURSE_OVERLAY:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    ax_rest, ax_mi = axes

    marker_axes = {"100": ax_rest, "200": ax_mi}
    marker_titles = {"100": "REST", "200": "MI"}

    for sess, pack in session_cache.items():
        tfr_data = pack["tfr"]
        focal_electrodes = pack.get("focal_electrodes_used", None)
        if focal_electrodes is None:
            # best effort fallback
            any_marker = next(iter(tfr_data.keys()))
            focal_electrodes = get_focal_for_session(sess, tfr_data[any_marker].ch_names)

        tcs = compute_focal_timecourses(tfr_data, focal_electrodes)

        # Label includes focal electrode(s)
        focal_label = ",".join(focal_electrodes) if focal_electrodes else "NO_FOCAL"
        line_label = f"{sess} ({focal_label})"

        for marker in ["100", "200"]:
            if marker not in tcs:
                continue
            times, mean_tc, lower, upper = tcs[marker]
            ax = marker_axes[marker]
            ax.plot(times, mean_tc, label=line_label)
            if OVERLAY_SHOW_SHADING:
                ax.fill_between(times, lower, upper, alpha=0.15)

    for marker, ax in marker_axes.items():
        ax.axhline(0, color="k", linewidth=0.8)
        ax.axvline(0.0, color="k", linestyle="--", linewidth=0.8)
        ax.axvline(feedback_time, color="k", linestyle=":", linewidth=0.8)
        ax.set_title(marker_titles[marker])
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)

    ax_rest.set_ylabel(ylabel_for_space())

    if LINE_YLIM is not None:
        ax_rest.set_ylim(*LINE_YLIM)

    fig.suptitle(f"Focal {FREQ_BAND.upper()} ERD Across Sessions ({PLOT_SPACE})", fontsize=12)
    ax_mi.legend(loc="best", fontsize=8)
    plt.tight_layout()


# =========================================================
# Loading / preprocessing
# =========================================================

def load_and_preprocess_session(subject, session, prompt_selection=True):
    """
    Load XDF files for a given subject/session, create Raw, preprocess,
    and return (epochs, raw, event_dict, meta_dict).

    Pipeline:
      Raw: notch + broad bandpass
      Epochs: padded window + rejection
      (Optional) Auto-drop dominant bad channels and re-epoch
      (Optional) CSD on epochs (after rejection)
    """
    xdf_dir = os.path.join(
        "/home/arman-admin/Documents/CurrentStudy",
        f"sub-{subject}", f"ses-{session}", "eeg/"
    )

    if not os.path.exists(xdf_dir):
        raise FileNotFoundError(f"❌ EEG directory not found: {xdf_dir}")

    xdf_files = [os.path.join(xdf_dir, f)
                 for f in os.listdir(xdf_dir) if f.endswith(".xdf")]
    if not xdf_files:
        raise FileNotFoundError(f"❌ No XDF files found in: {xdf_dir}")

    print(f"\n📂 [{session}] Found {len(xdf_files)} XDF files in: {xdf_dir}")
    for idx, file in enumerate(xdf_files, start=1):
        print(f" [{idx}] {os.path.basename(file)}")

    # Choose which XDFs to load
    if prompt_selection and not MULTI_SESSION_MODE:
        print("\nPress ENTER to merge **all** files, or enter the number(s) "
              "of the file(s) to load (comma-separated, e.g., 1,3): ")
        user_input = input("➡️  Selection: ").strip()

        if user_input:
            try:
                selected_indices = [int(i) - 1 for i in user_input.split(",")]
                selected_files = [xdf_files[i] for i in selected_indices
                                  if 0 <= i < len(xdf_files)]
            except ValueError:
                print("❌ Invalid input. Loading all files instead.")
                selected_files = xdf_files
        else:
            selected_files = xdf_files
    else:
        selected_files = xdf_files

    eeg_streams, marker_streams = [], []
    for xdf_file in selected_files:
        eeg_s, marker_s = load_xdf(xdf_file)
        eeg_streams.append(eeg_s)
        marker_streams.append(marker_s)

    print(f"✅ Successfully loaded and merged {len(selected_files)} XDF file(s).")

    if len(eeg_streams) == 1:
        eeg_stream, marker_stream = eeg_streams[0], marker_streams[0]
    else:
        eeg_stream, marker_stream = concatenate_streams(eeg_streams, marker_streams)

    # ---- Raw construction ----
    eeg_timestamps = np.array(eeg_stream["time_stamps"])
    eeg_data = np.array(eeg_stream["time_series"]).T   # (n_channels, n_samples)
    channel_names = get_channel_names_from_xdf(eeg_stream)

    marker_data = np.array([int(v[0]) for v in marker_stream["time_series"]])
    marker_timestamps = np.array([float(v[1]) for v in marker_stream["time_series"]])

    print("\n EEG Channels from XDF:", channel_names)

    montage = mne.channels.make_standard_montage("standard_1020")

    rename_dict = {
        "FP1": "Fp1", "FPZ": "Fpz", "FP2": "Fp2",
        "FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "POZ": "POz", "OZ": "Oz"
    }

    non_eeg_channels = {"AUX1", "AUX2", "AUX3", "AUX7", "AUX8", "AUX9", "TRIGGER"}
    valid_eeg_channels = [ch for ch in channel_names if ch not in non_eeg_channels]
    valid_indices = [channel_names.index(ch) for ch in valid_eeg_channels]
    eeg_data = eeg_data[valid_indices, :]

    sfreq = config.FS
    info = mne.create_info(
        ch_names=valid_eeg_channels,
        sfreq=sfreq,
        ch_types="eeg"
    )
    raw = mne.io.RawArray(eeg_data, info)

    # Drop mastoids if present
    if "M1" in raw.ch_names and "M2" in raw.ch_names:
        raw.drop_channels(["M1", "M2"])
        print("Removed Mastoid Channels: M1, M2")
    else:
        print("No Mastoid Channels Found in Data")

    raw.rename_channels(rename_dict)
    raw.set_montage(montage, match_case=True, on_missing="warn")

    # ---- Unit normalization ----
    # MNE expects VOLTS. If your XDF is in µV, convert µV -> V.
    if XDF_DATA_IN_UV:
        raw._data *= 1e-6  # µV -> V
        print("✅ Converted XDF EEG: µV -> V (raw in Volts).")
    else:
        print("✅ Assuming XDF EEG already in Volts (raw in Volts).")

    # ---- Channel subset selection (after rename/montage) ----
    if USE_SUBSET_CHANNELS:
        keep = [ch for ch in CHANNEL_SUBSET if ch in raw.ch_names]
        missing = sorted(set(CHANNEL_SUBSET) - set(keep))
        if missing:
            print("⚠️ Missing subset channels:", missing)
        raw.pick_channels(keep)
        print(f"✅ Picked {len(raw.ch_names)} subset channels.")

    # ---- Preprocessing on raw (broadband) ----
    raw.notch_filter(60)
    raw.filter(l_freq=BROADBAND_LOW, h_freq=BROADBAND_HIGH, method="iir")

    data = raw.get_data()  # shape: (channels, samples)
    uV = data  # NOTE: if raw is Volts, these are Volts (your original print kept as-is)

    print("EEG amplitude percentiles (µV):")
    print("  1% :", np.percentile(uV, 1))
    print(" 50% :", np.percentile(uV, 50))
    print(" 99% :", np.percentile(uV, 99))

    print("\n Final EEG Channels After Raw Processing:", raw.ch_names)

    # ---- Trial definition from markers ----
    min_trial_duration = 1.0
    max_trial_duration = 5.5
    EPS = 0.02

    valid_start_indices = []

    for idx, code in enumerate(marker_data):
        if code in [100, 200]:
            t_start = marker_timestamps[idx]
            end_code = code + 20  # 120 / 220
            end_time = None

            for j in range(idx + 1, len(marker_data)):
                if marker_data[j] == end_code:
                    end_time = marker_timestamps[j]
                    break

            if end_time is None:
                print(f"⚠️ Skipped: No end marker for start at {t_start:.2f}s")
                continue

            duration = end_time - t_start
            print(f"Start: {t_start:.2f}s → End: {end_time:.2f}s | Dur: {duration:.2f}s")

            if (duration + EPS) >= min_trial_duration and (duration - EPS) <= max_trial_duration:
                valid_start_indices.append(idx)
            else:
                reason = []
                if (duration + EPS) < min_trial_duration:
                    reason.append(f"<{min_trial_duration}s")
                if (duration - EPS) > max_trial_duration:
                    reason.append(f">{max_trial_duration}s")
                print(f"⚠️ Skipped trial (duration {duration:.2f}s: {', '.join(reason)})")

    marker_data_valid = [marker_data[i] for i in valid_start_indices]
    marker_timestamps_valid = [marker_timestamps[i] for i in valid_start_indices]
    event_dict = {str(code): code for code in set(marker_data_valid)}

    event_samples = np.searchsorted(eeg_timestamps, marker_timestamps_valid)
    events = np.column_stack(
        (event_samples, np.zeros(len(marker_data_valid), dtype=int), marker_data_valid)
    )

    # ---- Epoching with padding + rejection ----
    reject = None
    flat = None
    if REJECT_EPOCHS:
        reject = dict(eeg=REJECT_P2P_UV)
        if FLAT_UV is not None:
            flat = dict(eeg=FLAT_UV)

    dropped_channels = []
    iters = 0

    while True:
        iters += 1

        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_dict,
            tmin=time_start - PAD_TFR,
            tmax=time_end + PAD_TFR,
            baseline=(time_start, time_start + baseline_period),
            detrend=1,
            preload=True,
            reject=reject,
            flat=flat,
        )

        # Epoch summary
        print(f"✅ Epochs created (padded): tmin={time_start-PAD_TFR:.2f}, tmax={time_end+PAD_TFR:.2f}")
        for code in ["100", "200"]:
            if code in epochs.event_id:
                print(f"✅ Marker {code}: {len(epochs[code])} epochs (after rejection)")

        # Decide whether to auto-drop a dominant bad channel
        if not (AUTO_DROP_BAD_CHANNELS and REJECT_EPOCHS):
            break

        # Total epochs originally attempted == len(events)
        total_attempted = len(events)
        kept = len(epochs)
        dropped = total_attempted - kept
        dropped_frac = dropped / total_attempted if total_attempted > 0 else 0.0

        # If we are NOT in a pathological regime, stop.
        if dropped_frac < AUTO_DROP_REJECT_FRAC:
            break

        # Guard: don’t drop too many channels
        if len(dropped_channels) >= AUTO_DROP_MAX_CHANNELS_TOTAL:
            print(f"⚠️ Auto-drop reached channel limit ({AUTO_DROP_MAX_CHANNELS_TOTAL}); stopping auto-drop.")
            break

        # Guard: don’t loop too many times
        if iters > AUTO_DROP_MAX_ITERS:
            print(f"⚠️ Auto-drop reached max iterations ({AUTO_DROP_MAX_ITERS}); stopping auto-drop.")
            break

        bad_ch, frac = pick_dominant_bad_channel(epochs, dominance_frac=AUTO_DROP_DOMINANCE_FRAC)
        if bad_ch is None:
            print("⚠️ >90% epochs dropped, but no single dominant channel found in drop_log; stopping auto-drop.")
            break

        if bad_ch not in raw.ch_names:
            print(f"⚠️ Dominant bad channel '{bad_ch}' not in raw.ch_names; stopping auto-drop.")
            break

        print(
            f"🧹 AUTO-DROP: {dropped}/{total_attempted} epochs dropped ({dropped_frac*100:.1f}%). "
            f"Dominant channel: {bad_ch} (in ~{frac*100:.1f}% of dropped epochs). Dropping + retrying..."
        )
        raw = raw.copy().drop_channels([bad_ch])
        dropped_channels.append(bad_ch)

        # Loop continues to re-epoch with updated raw

    # ---- Apply CSD AFTER rejection (recommended) ----
    if APPLY_CSD:
        epochs = mne.preprocessing.compute_current_source_density(epochs)
        print("✅ Applied CSD to epochs (after rejection).")

    # ---- Optional: limiting to specific index range (kept from your script) ----
    if LIMIT_EPOCH_RANGE:
        start_idx, end_idx = EPOCH_RANGE_START_IDX, EPOCH_RANGE_END_IDX
        subset = []
        for code in ["100", "200"]:
            if code in epochs.event_id:
                ep = epochs[code]
                total = len(ep)
                if end_idx > total:
                    print(f"⚠️ Requested {end_idx} epochs but only {total} available "
                          f"for marker {code}; trimming.")
                    end_local = total
                else:
                    end_local = end_idx
                subset.append(ep[start_idx:end_local])

        if subset:
            epochs = mne.concatenate_epochs(subset)
            print(f"✅ Final subset across markers has {len(epochs)} epochs.")
        else:
            print("⚠️ No epochs selected after subsetting.")

    # Decide focal electrodes actually usable in this session, after dropping channels
    # (use raw.ch_names as the ground truth channel set for this session)
    focal_electrodes = get_focal_for_session(session, raw.ch_names)

    if USE_SESSION_SPECIFIC_FOCAL:
        desired = FOCAL_ELECTRODES_BY_SESSION.get(session, DEFAULT_FOCAL_ELECTRODES)
    else:
        desired = FOCAL_ELECTRODES

    desired = desired if desired else DEFAULT_FOCAL_ELECTRODES

    missing_desired = [ch for ch in desired if ch not in raw.ch_names]
    if missing_desired:
        print(f"⚠️ Session {session}: desired focal electrodes missing after preprocessing: {missing_desired}")

        if FOCAL_IF_DROPPED_POLICY == "fallback":
            fallback = DEFAULT_FOCAL_ELECTRODES if DEFAULT_FOCAL_ELECTRODES else FOCAL_ELECTRODES
            focal_electrodes = [ch for ch in fallback if ch in raw.ch_names]
            print(f"➡️ Using fallback focal electrodes: {focal_electrodes}")
        elif FOCAL_IF_DROPPED_POLICY == "skip":
            focal_electrodes = []
            print("➡️ Focal plotting/metrics will be skipped for this session (no usable focal electrode).")
        else:
            raise ValueError("FOCAL_IF_DROPPED_POLICY must be 'fallback' or 'skip'")

    meta = {
        "dropped_channels": dropped_channels,
        "focal_electrodes_used": focal_electrodes,
    }
    if dropped_channels:
        print(f"✅ Session {session}: auto-dropped channels: {dropped_channels}")
    print(f"✅ Session {session}: focal electrodes used: {focal_electrodes}")

    return epochs, raw, event_dict, meta


def load_all_sessions(subject, sessions):
    cache = {}
    for sess in sessions:
        print("\n" + "=" * 60)
        print(f"Loading session {sess}")
        print("=" * 60)

        epochs, raw, event_dict, meta = load_and_preprocess_session(
            subject, sess, prompt_selection=False
        )
        tfr_data = compute_tfr_epochs(epochs)
        cache[sess] = {"epochs": epochs, "raw": raw, "tfr": tfr_data, **meta}
    return cache


# =========================================================
# TFR computation (with padding -> crop)
# =========================================================

def compute_tfr_epochs(epochs):
    """
    Compute TFR (multitaper) for each marker in MARKERS_FOR_ANALYSIS.
    - We retain per-trial TFR: average=False
    - Baseline in logratio
    - Then crop TFR back to the plot window [-1,4] to remove edge artifacts
    """
    tfr_data = {}
    freqs = FREQS

    for marker in MARKERS_FOR_ANALYSIS:
        if marker not in epochs.event_id:
            print(f"⚠️ Marker {marker} not in epochs; skipping TFR.")
            continue

        print(f"Computing TFR for marker {marker} ...")
        tfr = epochs[marker].compute_tfr(
            method="multitaper",
            freqs=freqs,
            tmin=(time_start - PAD_TFR),
            tmax=(time_end + PAD_TFR),
            n_cycles=N_CYCLES,
            use_fft=True,
            return_itc=False,
            average=False,
        )

        tfr.apply_baseline(
            baseline=(time_start, time_start + baseline_period),
            mode="logratio"
        )

        # ✅ Crop away padded edges (this is the edge-artifact fix)
        tfr.crop(tmin=time_start, tmax=time_end)

        tfr_data[marker] = tfr

    return tfr_data


# =========================================================
# Focal timecourses (both markers) with SEM shading
# =========================================================

def compute_focal_timecourses(tfr_data, focal_electrodes):
    """
    Stable focal timecourses:
      - compute trialwise focal LOGRATIO timecourse
      - baseline-center in LOGRATIO domain
      - mean/SEM across trials in LOGRATIO domain
      - convert to requested plot space at the end
    """
    timecourses = {}

    if focal_electrodes is None:
        focal_electrodes = []

    for marker, tfr in tfr_data.items():
        times = tfr.times
        freqs = tfr.freqs
        data = tfr.data  # (trials, ch, freq, time) in logratio

        freq_mask = (freqs >= lowband) & (freqs <= highband)
        focal_idxs = [tfr.ch_names.index(ch) for ch in focal_electrodes if ch in tfr.ch_names]
        if not focal_idxs:
            print(f"⚠️ Marker {marker}: none of focal electrodes found in TFR channels. Skipping focal timecourse.")
            continue

        # trialwise focal logratio timecourse (trials, time)
        focal_log_trials = data[:, focal_idxs][:, :, freq_mask, :].mean(axis=(1, 2))

        # baseline-center each trial in logratio domain (stable)
        baseline_mask = (times >= time_start) & (times < 0.0)
        if baseline_mask.any():
            focal_log_trials = focal_log_trials - focal_log_trials[:, baseline_mask].mean(axis=1, keepdims=True)

        mean_log = focal_log_trials.mean(axis=0)
        err_log = get_error(focal_log_trials, VAR_METHOD)

        # Convert to requested plotting space
        mean_tc = maybe_convert_for_plot(mean_log)
        lower = maybe_convert_for_plot(mean_log - err_log)
        upper = maybe_convert_for_plot(mean_log + err_log)

        timecourses[marker] = (times, mean_tc, lower, upper)

    return timecourses


def plot_focal_timecourses(timecourses, session_label=None):
    """Plot focal timecourses for each marker with shaded variability band."""
    if not timecourses or not DO_FOCAL_TIMECOURSE:
        return

    plt.figure(figsize=(7, 4))
    color_map = {"100": "tab:gray", "200": "tab:orange"}
    label_map = {"100": "REST (100)", "200": "MI (200)"}

    for marker, (times, mean_tc, lower, upper) in timecourses.items():
        col = color_map.get(marker, None)
        lab = label_map.get(marker, f"Marker {marker}")
        plt.plot(times, mean_tc, label=lab, color=col)
        plt.fill_between(times, lower, upper, alpha=0.25, color=col)

    plt.axhline(0, color="k", linewidth=0.8)
    plt.axvline(0.0, color="k", linestyle="--", linewidth=0.8, label="Cue")
    plt.axvline(feedback_time, color="k", linestyle=":", linewidth=0.8, label="Feedback")

    plt.xlabel("Time (s)")
    plt.ylabel(f"{ylabel_for_space()}")
    title = f"Focal {FREQ_BAND.upper()} Timecourse ({PLOT_SPACE})"
    if session_label:
        title += f" | {session_label}"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")


# =========================================================
# Topographic maps
# =========================================================

def plot_topomaps(tfr_data):
    if not DO_TOPO_MAPS or not tfr_data:
        return

    # Compute vmin/vmax from averaged TFRs (in logratio space)
    all_vals = []
    for tfr in tfr_data.values():
        tfr_avg = tfr.average()
        all_vals.append(tfr_avg.data.flatten())
    if not all_vals:
        print("⚠️ No TFR data for topomaps.")
        return

    all_erd = np.concatenate(all_vals)
    vmin, vmax = np.percentile(all_erd, [2, 98])
    print(f"Topomap ERD/ERS color scale (logratio): vmin={vmin:.2f}, vmax={vmax:.2f}")

    window_size = 0.5
    time_windows = np.arange(0, 3, window_size)
    skip_factor = 1

    for marker, tfr in tfr_data.items():
        tfr_avg = tfr.average()
        selected_indices = range(0, len(time_windows), skip_factor)
        fig, axes = plt.subplots(
            1, len(selected_indices), figsize=(15, 4), constrained_layout=True
        )

        mappable = None
        for ax, i in zip(axes, selected_indices):
            t_start = time_windows[i]
            t_end = t_start + window_size

            tfr_avg.plot_topomap(
                tmin=t_start,
                tmax=t_end,
                axes=ax,
                cmap="viridis",
                show=False,
                vlim=(vmin, vmax),
                colorbar=False,
                show_names=True,
            )
            if hasattr(ax, "collections") and ax.collections:
                mappable = ax.collections[0]
            ax.set_title(f"{t_start:.1f}–{t_end:.1f}s")

        if mappable is not None:
            norm = plt.Normalize(vmin, vmax)
            sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
            sm.set_array([])
            cbar = fig.colorbar(
                sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1
            )
            cbar.set_label("ERD/ERS (logratio)", fontsize=12)

        label_map = {"100": "Rest", "200": "Right Arm MI", "300": "Robot Move"}
        marker_label = label_map.get(marker, f"Marker {marker}")
        fig.suptitle(f"ERD/ERS Topomaps – {marker_label}", fontsize=14)


# =========================================================
# Scalar ERD metric for bar plots
# =========================================================

def compute_scalar_erd_trials(tfr, time_window, focal_electrodes):
    """
    Returns one scalar per trial over time_window.

    Computation:
      - Always starts in logratio domain (tfr.data)
      - Outputs in the chosen space (PLOT_SPACE):
          * percent:  % change from baseline
          * logratio: logratio units
      - Optional normalization for bar plots:
          * difference: focal - motor (valid in both spaces)
          * ratio: only valid in percent space
    """
    data = tfr.data
    times = tfr.times
    freqs = tfr.freqs

    freq_mask = (freqs >= lowband) & (freqs <= highband)
    t0, t1 = time_window
    time_mask = (times >= t0) & (times <= t1)

    if not np.any(time_mask):
        raise RuntimeError(
            f"No time samples in scalar window {time_window} within TFR times "
            f"[{times[0]:.2f}, {times[-1]:.2f}]"
        )

    num_idxs = [tfr.ch_names.index(ch) for ch in focal_electrodes if ch in tfr.ch_names]
    den_idxs = [
        tfr.ch_names.index(ch) for ch in MOTOR_NORM_ELECTRODES
        if ch in tfr.ch_names and ch not in focal_electrodes
    ]

    if not num_idxs:
        raise RuntimeError("None of focal electrodes found in TFR channels.")

    focal_log = data[:, num_idxs][:, :, freq_mask][:, :, :, time_mask].mean(axis=(1, 2, 3))
    focal_out = maybe_convert_for_plot(focal_log)

    if not BAR_USE_NORMALIZATION:
        return focal_out

    if not den_idxs:
        print("⚠️ No MOTOR_NORM_ELECTRODES found (after excluding focal); returning focal-only.")
        return focal_out

    motor_log = data[:, den_idxs][:, :, freq_mask][:, :, :, time_mask].mean(axis=(1, 2, 3))
    motor_out = maybe_convert_for_plot(motor_log)

    if BAR_NORM_METHOD == "difference":
        return focal_out - motor_out

    if BAR_NORM_METHOD == "ratio":
        if PLOT_SPACE != "percent":
            raise ValueError("BAR_NORM_METHOD='ratio' only makes sense in percent space.")
        eps = 1e-3
        motor_safe = np.where(np.abs(motor_out) < eps, np.sign(motor_out) * eps, motor_out)
        return focal_out / motor_safe

    raise ValueError("BAR_NORM_METHOD must be 'ratio' or 'difference'")


def compare_sessions_cached(session_cache):
    """
    Multi-session paired-bar plot (cached).
    - For each session and each marker, compute scalar metric over SCALAR_WINDOW.
    - Plot paired bars per session with per-trial scatter + error bars.
    """
    metrics = {}

    for sess, pack in session_cache.items():
        print("\n" + "=" * 60)
        print(f"Analyzing session {sess} (cached)")
        print("=" * 60)

        tfr_data = pack["tfr"]
        focal_electrodes = pack.get("focal_electrodes_used", [])
        if not focal_electrodes:
            if FOCAL_IF_DROPPED_POLICY == "skip":
                print(f"⚠️ Session {sess}: no focal electrodes; skipping scalar metrics.")
                continue
            # fallback
            any_marker = next(iter(tfr_data.keys())) if tfr_data else None
            if any_marker is not None:
                focal_electrodes = get_focal_for_session(sess, tfr_data[any_marker].ch_names)

        sess_dict = {}

        for marker in MARKERS_FOR_ANALYSIS:
            if marker not in tfr_data:
                print(f"⚠️ Session {sess}: marker {marker} not present.")
                continue

            tfr = tfr_data[marker]
            try:
                erd_trials = compute_scalar_erd_trials(tfr, SCALAR_WINDOW, focal_electrodes)
            except Exception as e:
                print(f"⚠️ Session {sess}, marker {marker}: scalar ERD compute failed: {e}")
                continue

            sess_dict[marker] = erd_trials

        if sess_dict:
            metrics[sess] = sess_dict

    if not metrics:
        print("⚠️ No session metrics computed; cannot plot paired bars.")
        return

    sessions_order = list(metrics.keys())
    n_sessions = len(sessions_order)
    x = np.arange(n_sessions)

    bar_width = 0.35
    offsets = {"100": -bar_width / 2, "200": +bar_width / 2}
    color_map = {"100": "tab:gray", "200": "tab:orange"}
    label_map = {"100": "REST (100)", "200": "MI (200)"}

    plt.figure(figsize=(9, 5))
    ax = plt.gca()

    means_per_marker = {m: [] for m in MARKERS_FOR_ANALYSIS}

    for i, sess in enumerate(sessions_order):
        sess_metrics = metrics[sess]

        for marker in MARKERS_FOR_ANALYSIS:
            if marker not in sess_metrics:
                continue

            vals = np.asarray(sess_metrics[marker])
            mean_val = vals.mean()
            err_val = get_error(vals, BAR_ERROR_METHOD)

            means_per_marker[marker].append(mean_val)

            xpos = x[i] + offsets[marker]
            col = color_map.get(marker, None)

            jitter = (np.random.rand(len(vals)) - 0.5) * (bar_width * 0.5)
            ax.scatter(
                xpos + jitter,
                vals,
                color=col,
                alpha=0.4,
                s=15,
                linewidths=0,
            )

            ax.bar(
                xpos,
                mean_val,
                width=bar_width * 0.9,
                color=col,
                alpha=0.6,
                edgecolor="k",
                label=label_map.get(marker, f"Marker {marker}") if i == 0 else None
            )
            ax.errorbar(
                xpos,
                mean_val,
                yerr=err_val,
                fmt="none",
                ecolor="k",
                elinewidth=1,
                capsize=3,
            )

    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sessions_order, rotation=45)
    ax.set_ylabel(f"ERD metric ({ylabel_for_space()})")
    ax.set_title(f"{FREQ_BAND.upper()} ERD by Session ({PLOT_SPACE})")

    if BAR_YLIM is not None:
        ax.set_ylim(*BAR_YLIM)

    ax.legend(loc="upper right")
    plt.tight_layout()

    # ---- Optional: regression per marker across sessions ----
    xs = np.arange(1, n_sessions + 1)
    for marker in MARKERS_FOR_ANALYSIS:
        vals = means_per_marker.get(marker, [])
        if len(vals) != n_sessions:
            continue

        vals = np.asarray(vals)
        slope, intercept, r, p, _ = stats.linregress(xs, vals)
        y_fit = intercept + slope * xs
        col = color_map.get(marker, "k")
        ax.plot(xs - 1 + offsets[marker], y_fit, color=col, linestyle="-", linewidth=2)

        text = (
            f"{label_map.get(marker, marker)}:\n"
            f"slope={slope:.2f}, r={r:.2f}, p={p:.3f}"
        )
        ax.text(
            0.02,
            0.98 - 0.12 * MARKERS_FOR_ANALYSIS.index(marker),
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color=col,
        )


# =========================================================
# Main entry
# =========================================================

def main():
    """
    Multi-session mode:
      1) Load + preprocess + compute TFR once per session (cache)
      2) Reuse cached TFRs for:
         - paired bar plot comparison
         - multi-session overlay timecourses

    Single-session mode:
      - load, compute TFR, plot topos + focal timecourse
    """
    if MULTI_SESSION_MODE:
        session_cache = load_all_sessions(subject, SESSION_LIST)
        compare_sessions_cached(session_cache)
        plot_multisession_overlay_timecourses_cached(session_cache)
    else:
        epochs, raw, event_dict, meta = load_and_preprocess_session(
            subject, session, prompt_selection=PROMPT_FOR_FILE_SELECTION
        )
        tfr_data = compute_tfr_epochs(epochs)

        if DO_TOPO_MAPS:
            plot_topomaps(tfr_data)

        if DO_FOCAL_TIMECOURSE:
            focal_electrodes = meta.get("focal_electrodes_used", [])
            tcs = compute_focal_timecourses(tfr_data, focal_electrodes)
            plot_focal_timecourses(tcs, session_label=f"{session} ({','.join(focal_electrodes)})")

    plt.show()


if __name__ == "__main__":
    main()
