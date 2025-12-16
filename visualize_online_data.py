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
session = "S004ONLINE"          # used when MULTI_SESSION_MODE = False
PROMPT_FOR_FILE_SELECTION = True

# ---- Multi-session mode ----
MULTI_SESSION_MODE = True
SESSION_LIST = ["S001ONLINE","S002ONLINE","S003ONLINE","S004ONLINE","S005ONLINE"]
                

# ---- Frequency band ----
FREQ_BAND = "mu"                # "mu" or "beta"
BANDS = {
    "mu":   (8, 13),
    "beta": (13, 30),
}
lowband, highband = BANDS[FREQ_BAND]

# ---- Markers to analyze ----
# 100 = Rest start, 200 = MI start (your convention)
MARKERS_FOR_ANALYSIS = ["100", "200"]

# ---- Time configuration ----
time_start = -1.0               # seconds (tmin for epochs)
baseline_period = 1.0           # baseline length (e.g., -1 to 0)
window_length = 5.0             # total epoch length (e.g., -1 to 4)
time_end = time_start + window_length
feedback_time = 1.0             # when feedback begins (vertical dotted line)

# ---- Focal vs. normalization electrodes ----
# FOCAL_ELECTRODES: channels where ERD is most focal (e.g., contralateral motor)
FOCAL_ELECTRODES = ["C3", "FC1"]  # configurable

# MOTOR_NORM_ELECTRODES: broad bilateral motor set for normalization (bar plots)
MOTOR_NORM_ELECTRODES = ['FC1','FC2','C3', 'Cz', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'P7','P3', 'Pz', 'P4', 'P8', 'POz']

# ---- Variability method for timecourses ----
VAR_METHOD = "sem"              # "sem" or "std"

# ---- Scalar ERD metric window (for bar plots) ----
SCALAR_WINDOW = (1.0, 3.8)      # average ERD in this window (relative to cue)

# ---- Error bars on paired bars ----
BAR_ERROR_METHOD = "sem"        # "sem" or "std"

# ---- Toggles ----
DO_TOPO_MAPS = True
DO_FOCAL_TIMECOURSE = True

# ---- Multi-session overlay timecourse plot ----
DO_MULTISESSION_TIMECOURSE_OVERLAY = True
OVERLAY_SHOW_SHADING = True      # <-- new
LINE_YLIM = None                 # e.g. (-80, 40) or None

# Line plot y-lims (optional)
LINE_YLIM = (-80,60)          # e.g. (-80, 40) or None

# ---- Bar plot normalization toggle (ONLY affects bar plots) ----
BAR_USE_NORMALIZATION = False

# Normalization method for bar plots only:
#   "ratio"     -> focal / motor  (paper-style "normalized to motor area")
#   "difference"-> focal - motor  (more stable)
BAR_NORM_METHOD = "ratio"   # "ratio" or "difference"

# Bar plot y-lims (optional)
BAR_YLIM = (-100, 100)          # or (-100, 100) if using difference, or None




# =========================================================
# Helper functions
# =========================================================



def logratio_to_percent_change(x):
    """Convert log10 power ratio to % change from baseline."""
    return 100.0 * (10.0 ** x - 1.0)


def get_error(arr, method):
    """Return SEM or STD along axis=0 for 2D array (trials x something)."""
    if method.lower() == "sem":
        return arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    elif method.lower() == "std":
        return arr.std(axis=0, ddof=1)
    else:
        raise ValueError(f"Unknown method for variability: {method}")

def compute_timecourse_from_tfr(tfr, channels):
    """
    Unnormalized focal %ERD timecourse.
    """
    data = tfr.data
    times = tfr.times
    freqs = tfr.freqs

    freq_mask = (freqs >= lowband) & (freqs <= highband)

    idxs = [tfr.ch_names.index(ch) for ch in channels if ch in tfr.ch_names]
    if not idxs:
        raise RuntimeError(f"None of channels found for timecourse: {channels}")

    log_tc = data[:, idxs][:, :, freq_mask, :].mean(axis=(1, 2))  # (trials, time)
    pct_tc = logratio_to_percent_change(log_tc)                   # (trials, time)

    # baseline-center in % domain
    baseline_mask = (times >= time_start) & (times < 0.0)
    if baseline_mask.any():
        base = pct_tc[:, baseline_mask].mean(axis=1, keepdims=True)
        pct_tc = pct_tc - base

    mean_tc = pct_tc.mean(axis=0)
    err_tc = get_error(pct_tc, VAR_METHOD)

    return times, mean_tc, err_tc



def plot_multisession_overlay_timecourses_cached(session_cache):
    """
    Overlay focal %ERD timecourses for each session on one plot (cached).
    - Always UN-normalized (matches your single-session focal timecourse logic)
    - MI and REST plotted separately (two subplots)
    - Each session is one line
    - Shading optional (OVERLAY_SHOW_SHADING)

    IMPORTANT: This uses compute_focal_timecourses(tfr_data) so the methodology
    matches the single-session plot (central line from AverageTFR, shading from trials).
    """
    if not DO_MULTISESSION_TIMECOURSE_OVERLAY:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    ax_rest, ax_mi = axes

    marker_axes = {"100": ax_rest, "200": ax_mi}
    marker_titles = {"100": "REST", "200": "MI"}

    for sess, pack in session_cache.items():
        tfr_data = pack["tfr"]

        # Use the SAME methodology as single-session timecourses
        tcs = compute_focal_timecourses(tfr_data)

        for marker in ["100", "200"]:
            if marker not in tcs:
                continue

            times, mean_tc, err_tc = tcs[marker]
            ax = marker_axes[marker]
            ax.plot(times, mean_tc, label=sess)

            if OVERLAY_SHOW_SHADING:
                ax.fill_between(
                    times,
                    mean_tc - err_tc,
                    mean_tc + err_tc,
                    alpha=0.15
                )

    for marker, ax in marker_axes.items():
        ax.axhline(0, color="k", linewidth=0.8)
        ax.axvline(0.0, color="k", linestyle="--", linewidth=0.8)
        ax.axvline(feedback_time, color="k", linestyle=":", linewidth=0.8)
        ax.set_title(marker_titles[marker])
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)

    ax_rest.set_ylabel("%ERD")

    if LINE_YLIM is not None:
        ax_rest.set_ylim(*LINE_YLIM)

    fig.suptitle(f"Focal {FREQ_BAND.upper()} ERD Across Sessions", fontsize=12)

    ax_mi.legend(loc="best", fontsize=8)
    plt.tight_layout()





# ---------------------------------------------------------
# Loading / preprocessing
# ---------------------------------------------------------

def load_and_preprocess_session(subject, session, prompt_selection=True):
    """
    Load XDF files for a given subject/session, create Raw, preprocess,
    and return (epochs, raw, event_dict).
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
        # For multi-session mode, just merge all files
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

    for ch in raw.info["chs"]:
        ch["unit"] = 201  # µV

    if "M1" in raw.ch_names and "M2" in raw.ch_names:
        raw.drop_channels(["M1", "M2"])
        print("Removed Mastoid Channels: M1, M2")
    else:
        print("No Mastoid Channels Found in Data")

    raw.rename_channels(rename_dict)
    raw.set_montage(montage, match_case=True, on_missing="warn")

    # convert from V to µV
    raw._data /= 1e6

    # Preprocessing
    raw.notch_filter(60)
    raw.filter(l_freq=lowband, h_freq=highband, method="iir")
    raw = mne.preprocessing.compute_current_source_density(raw)

    print("\n Final EEG Channels After Processing:", raw.ch_names)

    # ---- Trial definition from markers ----
    min_trial_duration = 1.0
    max_trial_duration = 5.5
    EPS = 0.02

    valid_start_indices = []

    for idx, code in enumerate(marker_data):
        if code in [100, 200]:
            t_start = marker_timestamps[idx]
            end_code = code + 20   # 120 / 220
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

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_dict,
        tmin=time_start,
        tmax=time_end,
        baseline=(time_start, time_start + baseline_period),
        detrend=1,
        preload=True,
    )

    for code in ["100", "200"]:
        if code in epochs.event_id:
            print(f"✅ Marker {code}: {len(epochs[code])} epochs")

    # optional limiting to specific index range as in your original script
    start_idx, end_idx = 0, 70
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

    return epochs, raw, event_dict

def load_all_sessions(subject, sessions):
    cache = {}
    for sess in sessions:
        print("\n" + "=" * 60)
        print(f"Loading session {sess}")
        print("=" * 60)

        epochs, raw, event_dict = load_and_preprocess_session(
            subject, sess, prompt_selection=False
        )
        tfr_data = compute_tfr_epochs(epochs)
        cache[sess] = {
            "epochs": epochs,
            "raw": raw,
            "tfr": tfr_data,
        }
    return cache

# ---------------------------------------------------------
# TFR computation
# ---------------------------------------------------------

def compute_tfr_epochs(epochs):
    """
    Compute TFR (multitaper) for each marker in MARKERS_FOR_ANALYSIS.

    We keep average=False so we retain per-trial EpochsTFR, and later
    we call .average() on those objects both for:
      - topomaps
      - focal timecourses (central line)
    """
    tfr_data = {}
    freqs = np.linspace(lowband, highband, int(highband - lowband)+1)

    for marker in MARKERS_FOR_ANALYSIS:
        if marker in epochs.event_id:
            print(f"Computing TFR for marker {marker} ...")
            tfr = epochs[marker].compute_tfr(
                method="multitaper",
                freqs=freqs,
                tmin=time_start,
                tmax=time_end,
                n_cycles=2.5,
                use_fft=True,
                return_itc=False,
                average=False,   # keep trials
            )
            tfr.apply_baseline(
                baseline=(time_start, time_start + baseline_period),
                mode="logratio"
            )
            tfr_data[marker] = tfr
        else:
            print(f"⚠️ Marker {marker} not in epochs; skipping TFR.")

    return tfr_data


# ---------------------------------------------------------
# Focal timecourse (both markers) with SEM shading
#   IMPORTANT: central line uses AverageTFR (same as topomap),
#              shaded band uses trialwise %ERD.
# ---------------------------------------------------------

def compute_focal_timecourses(tfr_data):
    """
    Compute focal %ERD timecourses (NOT normalized) for each marker.

    Central line:
      - from AverageTFR (same as topomap)
      - avg over FOCAL_ELECTRODES and freq band
      - convert logratio -> %ERD
      - re-baseline in % domain so mean in [-1,0] is 0

    Shaded band:
      - SEM/STD across trials of the same %ERD metric
      - also re-baselined per trial in % domain
    """
    timecourses = {}

    for marker, tfr in tfr_data.items():
        # ---------- central line from AverageTFR (matches topomap) ----------
        tfr_avg = tfr.average()     # AverageTFR: (n_channels, n_freqs, n_times)
        times = tfr_avg.times

        freq_mask = (tfr_avg.freqs >= lowband) & (tfr_avg.freqs <= highband)
        focal_idxs = [
            tfr_avg.ch_names.index(ch)
            for ch in FOCAL_ELECTRODES
            if ch in tfr_avg.ch_names
        ]
        if not focal_idxs:
            print(f"⚠️ Marker {marker}: none of FOCAL_ELECTRODES found in TFR channels.")
            continue

        # tfr_avg.data: (n_channels, n_freqs, n_times)
        focal_log_avg = tfr_avg.data[focal_idxs][:, freq_mask, :].mean(axis=(0, 1))  # (time,)
        mean_tc = logratio_to_percent_change(focal_log_avg)  # (time,) in %ERD

        # ---------- variability band from trialwise EpochsTFR ----------
        data = tfr.data  # (n_epochs, n_channels, n_freqs, n_times)
        focal_log_trials = data[:, focal_idxs][:, :, freq_mask, :].mean(axis=(1, 2))  # (trials, time)
        focal_pct_trials = logratio_to_percent_change(focal_log_trials)               # (trials, time)

        # ---------- re-baseline in % domain (use [-1,0] window) ----------
        baseline_mask = (times >= time_start) & (times < 0.0)
        if baseline_mask.any():
            # grand-average baseline
            baseline_mean = mean_tc[baseline_mask].mean()
            mean_tc = mean_tc - baseline_mean

            # per-trial baseline (so SEM around 0 before cue)
            trial_baseline_means = focal_pct_trials[:, baseline_mask].mean(axis=1, keepdims=True)
            focal_pct_trials = focal_pct_trials - trial_baseline_means

        err_tc = get_error(focal_pct_trials, VAR_METHOD)  # SEM or STD over trials

        timecourses[marker] = (times, mean_tc, err_tc)

    return timecourses


def plot_focal_timecourses(timecourses, session_label=None):
    """
    Plot focal %ERD timecourses (not normalized) for each marker
    with shaded variability band.
    """
    if not timecourses or not DO_FOCAL_TIMECOURSE:
        return

    plt.figure(figsize=(7, 4))
    color_map = {
        "100": "tab:gray",
        "200": "tab:orange",
    }
    label_map = {
        "100": "REST (100)",
        "200": "MI (200)",
    }

    for marker, (times, mean_tc, err_tc) in timecourses.items():
        col = color_map.get(marker, None)
        lab = label_map.get(marker, f"Marker {marker}")
        plt.plot(times, mean_tc, label=lab, color=col)
        plt.fill_between(times, mean_tc - err_tc, mean_tc + err_tc,
                         alpha=0.25, color=col)

    plt.axhline(0, color="k", linewidth=0.8)
    plt.axvline(0.0, color="k", linestyle="--", linewidth=0.8, label="Cue")
    plt.axvline(feedback_time, color="k", linestyle=":", linewidth=0.8,
                label="Feedback")

    plt.xlabel("Time (s)")
    plt.ylabel("%ERD (Focal channels)")
    title = f"Focal {FREQ_BAND.upper()} %ERD"
    if session_label:
        title += f" | {session_label}"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")


# ---------------------------------------------------------
# Topographic maps (unchanged logic)
# ---------------------------------------------------------

def plot_topomaps(tfr_data):
    if not DO_TOPO_MAPS or not tfr_data:
        return

    # Compute vmin/vmax from averaged TFRs
    all_vals = []
    for tfr in tfr_data.values():
        tfr_avg = tfr.average()
        all_vals.append(tfr_avg.data.flatten())
    if not all_vals:
        print("⚠️ No TFR data for topomaps.")
        return

    all_erd = np.concatenate(all_vals)
    vmin, vmax = np.percentile(all_erd, [2, 98])
    print(f"Topomap ERD/ERS color scale: vmin={vmin:.2f}, vmax={vmax:.2f}")

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

def compute_scalar_erd_trials(tfr, time_window):
    """
    Returns one scalar per trial over time_window:
      - Always computes focal %ERD per trial (baselined by logratio).
      - If BAR_USE_NORMALIZATION:
          * "ratio":      focal / motor_mean   (paper-style)
          * "difference": focal - motor_mean
        else:
          * returns focal only (unnormalized)

    Numerator channels: FOCAL_ELECTRODES
    Denominator channels: MOTOR_NORM_ELECTRODES (excluding overlaps)
    """
    data = tfr.data  # (trials, ch, freq, time) in logratio
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

    # channel indices
    num_idxs = [tfr.ch_names.index(ch) for ch in FOCAL_ELECTRODES if ch in tfr.ch_names]
    den_idxs = [tfr.ch_names.index(ch) for ch in MOTOR_NORM_ELECTRODES
                if ch in tfr.ch_names and ch not in FOCAL_ELECTRODES]

    if not num_idxs:
        raise RuntimeError("None of FOCAL_ELECTRODES found in TFR channels.")

    # focal scalar (%ERD) per trial
    focal_log = data[:, num_idxs][:, :, freq_mask][:, :, :, time_mask].mean(axis=(1, 2, 3))
    focal_pct = logratio_to_percent_change(focal_log)  # (trials,)

    # if normalization is off, return focal-only
    if not BAR_USE_NORMALIZATION:
        return focal_pct

    if not den_idxs:
        print("⚠️ No MOTOR_NORM_ELECTRODES found (after excluding focal); "
              "returning focal-only scalars.")
        return focal_pct

    # motor scalar (%ERD) per trial
    motor_log = data[:, den_idxs][:, :, freq_mask][:, :, :, time_mask].mean(axis=(1, 2, 3))
    motor_pct = logratio_to_percent_change(motor_log)  # (trials,)

    if BAR_NORM_METHOD == "difference":
        return focal_pct - motor_pct

    if BAR_NORM_METHOD == "ratio":
        eps = 1e-3  # in % units
        motor_safe = np.where(np.abs(motor_pct) < eps, np.sign(motor_pct) * eps, motor_pct)
        return focal_pct / motor_safe

    raise ValueError("BAR_NORM_METHOD must be 'ratio' or 'difference'")


def compare_sessions_cached(session_cache):
    """
    Multi-session paired-bar plot (cached):
      - For each cached session and each marker (100, 200),
        compute scalar ERD (%), optionally normalized (BAR_USE_NORMALIZATION),
        averaged across SCALAR_WINDOW.
      - Plot paired bars per session (REST vs MI)
        with per-trial scatter and error bars.

    Uses cached TFRs to avoid re-loading/recomputing.
    """
    # metrics[session][marker] = vector of trial-level ERD values
    metrics = {}

    # session_cache: {sess: {"epochs":..., "raw":..., "tfr": tfr_data}}
    for sess, pack in session_cache.items():
        print("\n" + "=" * 60)
        print(f"Analyzing session {sess} (cached)")
        print("=" * 60)

        tfr_data = pack["tfr"]
        sess_dict = {}

        for marker in MARKERS_FOR_ANALYSIS:
            if marker not in tfr_data:
                print(f"⚠️ Session {sess}: marker {marker} not present.")
                continue

            tfr = tfr_data[marker]
            erd_trials = compute_scalar_erd_trials(tfr, SCALAR_WINDOW)
            sess_dict[marker] = erd_trials

        if sess_dict:
            metrics[sess] = sess_dict

    if not metrics:
        print("⚠️ No session metrics computed; cannot plot paired bars.")
        return

    # ---- Build paired-bar plot ----
    sessions_order = list(metrics.keys())
    n_sessions = len(sessions_order)
    x = np.arange(n_sessions)

    bar_width = 0.35
    offsets = {
        "100": -bar_width / 2,
        "200": +bar_width / 2,
    }
    color_map = {
        "100": "tab:gray",
        "200": "tab:orange",
    }
    label_map = {
        "100": "REST (100)",
        "200": "MI (200)",
    }

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

            # scatter of individual trials
            jitter = (np.random.rand(len(vals)) - 0.5) * (bar_width * 0.5)
            ax.scatter(
                xpos + jitter,
                vals,
                color=col,
                alpha=0.4,
                s=15,
                linewidths=0,
            )

            # bar & error bar
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
    ax.set_ylabel("ERD metric")
    ax.set_title(f"{FREQ_BAND.upper()} ERD by Session")

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



# ---------------------------------------------------------
# Main entry
# ---------------------------------------------------------

def main():
    """
    Entry point.

    Multi-session mode:
      1) Load + preprocess + compute TFR once per session (cache)
      2) Reuse cached TFRs for BOTH:
         - paired bar plot comparison
         - multi-session overlay timecourses

    Single-session mode:
      Unchanged behavior.
    """
    if MULTI_SESSION_MODE:
        session_cache = load_all_sessions(subject, SESSION_LIST)

        compare_sessions_cached(session_cache)
        plot_multisession_overlay_timecourses_cached(session_cache)
    else:
        epochs, raw, event_dict = load_and_preprocess_session(
            subject, session, prompt_selection=PROMPT_FOR_FILE_SELECTION
        )
        tfr_data = compute_tfr_epochs(epochs)

        if DO_TOPO_MAPS:
            plot_topomaps(tfr_data)

        if DO_FOCAL_TIMECOURSE:
            tcs = compute_focal_timecourses(tfr_data)
            plot_focal_timecourses(tcs, session_label=session)

    plt.show()





if __name__ == "__main__":
    main()
