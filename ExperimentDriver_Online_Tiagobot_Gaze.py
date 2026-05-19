"""
ExperimentDriver_Online_Tiagobot_Gaze.py

Gaze-driven variant of ExperimentDriver_Online_Tiagobot.py — same trial
loop, but per-trial letter selection comes from the user's gaze
(classified against a calibration NPZ) instead of random.choice.

Derived from ExperimentDriver_Online_Tiagobot.py @ b340e91 on
feature/tiagobot-gaze-integration (2026-05-19). The parent driver MUST
remain unchanged on this branch — it is the known-good fallback. If
the parent receives a bug fix upstream (trial-loop ordering change,
grip-hold tweak, etc.), port the same change here too. The two are
expected to stay in sync except for the gaze-selection block in main().

Flow per MI-success trial (changes vs parent in CAPS):
1. RUN A PER-TRIAL GAZE SELECTION WINDOW
   (run_tiago_gaze_selection_window — collects (t, x_norm, y_norm,
   conf) samples from the Pupil Labs Neon LSL stream).
2. AVERAGE THE WINDOW AND CLASSIFY TO ONE OF A-I VIA
   Utils.tiagobot_gaze.classify_gaze_to_letter.
3. If classification fails (no centroid within max distance, or no
   valid samples) → log + skip the GO for this trial (no random
   fallback — fail visibly per plan §6.3 step 4).
4. Otherwise write `'{analog},{angle},{delay}\\n'` for the chosen
   letter to the Tiagobot Arduino.
5. (Optional) Write config.ARDUINO_CMD_MI to the glove Arduino.
6. Run hold_messages_and_classify for TIME_ROB seconds.
7. Send `'h\\n'` to Tiagobot — actuator retracts and servo centers.
8. (Optional) Write config.ARDUINO_CMD_REST to the glove Arduino.

The gaze layer is read-only with respect to Utils/tiagobot.py (Tier 1).
All gaze logic lives in Utils/tiagobot_gaze.py (pure functions) and
the inline run_tiago_gaze_selection_window helper below.

Layout source of truth: Documents/SoftwareDocs/Tiagobot_Gaze_AI_Layout.md.
Calibration produced by: tiago_gaze_calibration_exec.py.
"""

import pygame
import socket
import pickle
import datetime
import time
import serial
import sys
import os
import numpy as np
from pylsl import StreamInlet, resolve_stream

# MNE for real-time EEG processing
import mne
mne.set_log_level("WARNING")

from Utils.visualization import (
    draw_arrow_fill,
    draw_ball_fill,
    draw_fixation_cross,
    draw_time_balls,
    draw_progress_bar
)

from Utils.experiment_utils import (
    generate_trial_sequence,
    save_transform,
    load_transform
)

from Utils.EEGStreamState import EEGStreamState

# `send_udp_message` carries the marker / FES triggers; the Tiagobot driver
# does not dial UDP_ROBOT (Tiagobot has no UDP interface). The
# `display_multiple_messages_with_udp` helper is invoked with
# udp_messages=None throughout, so it just runs the feedback UI loop.
from Utils.networking import send_udp_message, display_multiple_messages_with_udp

# Tiagobot serial helpers (Tier 1).
from Utils.tiagobot import (
    open_port as open_tiago_port,
    send_letter as tiago_send_letter,
    send_home as tiago_send_home,
    close_port as tiago_close_port,
    find_tiagobot_port,
    find_glove_port,
    wait_for_completion as tiago_wait_for_completion,
    TARGET_REACHED_MARKER, HOMED_MARKER,
)

# Tiagobot gaze helpers (pure functions; not Tier 1).
from Utils.tiagobot_gaze import (
    load_centroids as tiago_gaze_load_centroids,
    classify_gaze_to_letter as tiago_gaze_classify,
    average_gaze_over_window as tiago_gaze_average,
    LETTERS as TIAGO_LETTERS,
)

import config

from pathlib import Path
from Utils.logging_manager import LoggerManager

from Utils.runtime_common import (
    log_confusion_matrix_from_trial_summary,
    append_trial_probabilities_to_csv,
    display_fixation_period,
    hold_messages_and_classify,
    show_feedback,
)
import Utils.runtime_common as _RC

from Utils.stream_utils import require_marker_stream

# Initialize experiment logger (auto-detects active run or falls back to Debug)
logger = LoggerManager.auto_detect_from_subject(
    subject=config.TRAINING_SUBJECT,
    base_path=Path(config.DATA_DIR),
    mode="online"
)

# Log experiment configuration snapshot
loggable_fields = [
    "UDP_MARKER", "UDP_FES",
    "ARM_SIDE", "TOTAL_TRIALS", "MAX_REPEATS",
    "TIME_MI", "TIME_ROB", "TIME_STATIONARY",
    "SHAPE_MAX", "SHAPE_MIN",
    "FES_toggle", "FES_CHANNEL", "FES_TIMING_OFFSET",
    "WORKING_DIR", "DATA_DIR", "MODEL_PATH",
    "TRAINING_SUBJECT",
    "MOTOR_CHANNEL_NAMES", "CLASSIFY_WINDOW", "THRESHOLD_MI", "THRESHOLD_REST",
    "RELAXATION_RATIO", "MIN_PREDICTIONS", "SURFACE_LAPLACIAN_TOGGLE",
    "SELECT_MOTOR_CHANNELS", "INTEGRATOR_ALPHA",
    "SHRINKAGE_PARAM_MDM", "SHRINKAGE_PARAM_XGB",
    "LEDOITWOLF", "RECENTERING", "UPDATE_DURING_MOVE",
    # Tiagobot-specific
    "TIAGOBOT_PORT", "TIAGOBOT_BAUD", "TIAGOBOT_TRAJECTORY",
    "TIAGOBOT_USE_GLOVE", "TIAGOBOT_GRIP_HOLD_DURATION",
    "SIMULATION_MODE",
    # Tiagobot gaze-specific
    "TIAGOBOT_GAZE_CALIBRATION_PATH",
    "TIAGOBOT_GAZE_SELECTION_WINDOW",
    "TIAGOBOT_GAZE_CONFIDENCE_THRESHOLD",
    "TIAGOBOT_GAZE_MAX_DIST_NORM",
    "GAZE_SAMPLE_WIDTH", "GAZE_SAMPLE_HEIGHT",
]
config_log_subset = {
    key: getattr(config, key) for key in loggable_fields if hasattr(config, key)
}
logger.save_config_snapshot(config_log_subset)


eeg_dir = logger.log_base / "eeg"
adaptive_T_path = eeg_dir / "adaptive_T.pkl"

Prev_T, counter, Prev_T_beta, counter_beta = load_transform(adaptive_T_path)
if Prev_T is None:
    counter = 0
if Prev_T_beta is None:
    counter_beta = 0
if Prev_T is None and Prev_T_beta is None:
    logger.log_event("ℹ️ No adaptive transforms — cold start for μ and β.")
else:
    parts = []
    if Prev_T is not None:
        parts.append(f"μ counter={counter}")
    if Prev_T_beta is not None:
        parts.append(f"β counter={counter_beta}")
    logger.log_event("✅ Loaded adaptive transform: " + ", ".join(parts))

logger.log_event("Logger initialized for Tiagobot online experimental driver.")


pygame.init()

if config.BIG_BROTHER_MODE:
    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
    screen = pygame.display.set_mode((1920, 1080), pygame.NOFRAME)
    logger.log_event("🎥 Big Brother Mode ON — window placed at (0,0) on external monitor (HDMI-1).")
else:
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    logger.log_event("👤 Big Brother Mode OFF — fullscreen on active display.")

pygame.display.set_caption("EEG Online Interactive Loop — Tiagobot")
info = pygame.display.Info()
screen_width = info.current_w
screen_height = info.current_h
logger.log_event("Pygame initialized and display configured.")

# UDP sockets. udp_socket_robot is created for parity with runtime_common
# helpers that take it as a parameter; the Tiagobot driver never dials a
# robot UDP command, so writes through it land in a dead endpoint and are
# effectively no-ops.
udp_socket_marker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket_robot  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket_fes    = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
logger.log_event("UDP sockets initialized for marker, robot (unused), and FES channels.")

FES_toggle = config.FES_toggle
logger.log_event(f"FES toggle status: {'Enabled' if FES_toggle else 'Disabled'}.")

# Decoder model
subject_model_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "models")
decoder_backend = str(getattr(config, "DECODER_BACKEND", "mdm")).lower()
if decoder_backend == "xgb_cov":
    model_filename = f"sub-{config.TRAINING_SUBJECT}_xgb_cov_features.pkl"
elif decoder_backend == "xgb_cov_erd":
    model_filename = f"sub-{config.TRAINING_SUBJECT}_xgb_cov_erd_features.pkl"
else:
    model_filename = f"sub-{config.TRAINING_SUBJECT}_model.pkl"
subject_model_path = os.path.join(subject_model_dir, model_filename)

try:
    with open(subject_model_path, 'rb') as f:
        model = pickle.load(f)
    logger.log_event(f"✅ Model successfully loaded from: {subject_model_path}")
except FileNotFoundError:
    logger.log_event(f"❌ Error: Model file '{subject_model_path}' not found. Ensure the model has been trained.", level="error")
    sys.exit(1)


# ============================================================
# TIAGOBOT SETUP (mandatory unless SIMULATION_MODE)
# ============================================================
# Resolve the port: explicit config first, fall back to USB-ID scan
# (Arduino Mega 2560 R3, vid:pid 2341:0042). This lets the driver run
# with no manual config when only Tiagobot is plugged in (or with both
# Arduinos as long as the glove isn't also a Mega 2560).
tiago_port = config.TIAGOBOT_PORT
if not tiago_port:
    logger.log_event("TIAGOBOT_PORT is empty — scanning USB for a Mega 2560 R3...")
    tiago_port = find_tiagobot_port(logger=logger)
    if tiago_port:
        logger.log_event(f"Auto-detected Tiagobot port: {tiago_port}")

# open_tiago_port returns None in SIMULATION_MODE or when port is unset.
# Real open failures raise; we let them propagate per fail-fast policy on
# Tier 1 hardware paths (CLAUDE.md "Error Handling").
tiago = open_tiago_port(tiago_port, config.TIAGOBOT_BAUD, logger)
if tiago is None and not config.SIMULATION_MODE:
    logger.log_event(
        "❌ Tiagobot port unavailable and SIMULATION_MODE is False — aborting.",
        level="error",
    )
    sys.exit(1)


# ============================================================
# GLOVE SETUP (optional)
# ============================================================
arduino = None
if config.TIAGOBOT_USE_GLOVE:
    glove_port = config.ARDUINO_PORT
    if not glove_port:
        logger.log_event("ARDUINO_PORT is empty — scanning USB for an Arduino Uno R3 (glove)...")
        glove_port = find_glove_port(logger=logger)
        if glove_port:
            logger.log_event(f"Auto-detected glove port: {glove_port}")

    if glove_port:
        try:
            logger.log_event(f"Connecting to Glove (Arduino) on {glove_port}...")
            arduino = serial.Serial(glove_port, config.ARDUINO_BAUD, timeout=0.1)
            time.sleep(2)  # Arduino reset wait
            logger.log_event("✅ Glove connected successfully.")
        except Exception as e:
            logger.log_event(f"❌ Error connecting to Glove: {e}", level="error")
            arduino = None
    else:
        logger.log_event("ℹ️ TIAGOBOT_USE_GLOVE=True but no glove port available — glove disabled.")
else:
    logger.log_event("ℹ️ Glove integration disabled (TIAGOBOT_USE_GLOVE=False).")


# ============================================================
# TIAGOBOT GAZE CALIBRATION (mandatory)
# ============================================================
# Load the per-letter centroids produced by tiago_gaze_calibration_exec.py.
# Fail-fast at startup if missing or malformed — per CLAUDE.md, silently
# falling back to (a) random letter selection or (b) nominal grid centroids
# would mask a real configuration error during a hardware session.
_gaze_cal_path = str(getattr(config, "TIAGOBOT_GAZE_CALIBRATION_PATH", "") or "")
if not _gaze_cal_path:
    logger.log_event(
        "❌ TIAGOBOT_GAZE_CALIBRATION_PATH is empty. Set it in "
        "config_local.py (path to the NPZ from "
        "tiago_gaze_calibration_exec.py) before launching this driver.",
        level="error",
    )
    sys.exit(1)
try:
    _tiago_centroids = tiago_gaze_load_centroids(_gaze_cal_path)
except (FileNotFoundError, ValueError) as e:
    logger.log_event(
        f"❌ Failed to load Tiagobot gaze calibration from "
        f"{_gaze_cal_path}: {e}",
        level="error",
    )
    sys.exit(1)
logger.log_event(
    f"✅ Loaded Tiagobot gaze centroids from {_gaze_cal_path}: "
    f"{len(_tiago_centroids)}/{len(TIAGO_LETTERS)} letters with valid data."
)
_missing_letters = [ch for ch in TIAGO_LETTERS if ch not in _tiago_centroids]
if _missing_letters:
    logger.log_event(
        f"⚠️ Calibration missing letters: {_missing_letters}. Trials whose "
        f"gaze maps closest to these will return None (skip GO)."
    )


# ============================================================
# GAZE LSL STREAM (Pupil Labs Neon)
# ============================================================
# Resolved once at startup so the per-trial selection window only pulls
# from an inlet — keeps the trial loop allocation-free.
# Channel layout (matches harmony_calibration_exec.py:194-197):
#   sample[0]  gaze_x (pixels, 0..GAZE_SAMPLE_WIDTH)
#   sample[1]  gaze_y (pixels, 0..GAZE_SAMPLE_HEIGHT)
#   sample[15] confidence
TIAGO_NEON_X_INDEX = 0
TIAGO_NEON_Y_INDEX = 1
TIAGO_NEON_CONF_INDEX = 15
TIAGO_GAZE_SAMPLE_W = float(getattr(config, "GAZE_SAMPLE_WIDTH", 1600.0))
TIAGO_GAZE_SAMPLE_H = float(getattr(config, "GAZE_SAMPLE_HEIGHT", 1200.0))


logger.log_event("finding training dataset . . .")
eeg_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "training_data")
logger.log_event(f"Script is looking for XDF files in: {eeg_dir}")

xdf_files = [
    os.path.join(eeg_dir, f) for f in os.listdir(eeg_dir)
    if f.endswith(".xdf") and "OBS" not in f
]

if not xdf_files:
    raise FileNotFoundError(f"No XDF files found in: {eeg_dir}")
logger.log_event(f"training data: {xdf_files}")


predictions_list = []
ground_truth_list = []

fs = config.FS

SESSION_TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logger.log_event(f"Session timestamp set: {SESSION_TIMESTAMP}")

# Wire runtime objects into runtime_common globals
_RC.config = config
_RC.logger = logger
_RC.model = model

_RC.screen = screen
_RC.screen_width = screen_width
_RC.screen_height = screen_height

_RC.udp_socket_marker = udp_socket_marker
_RC.udp_socket_robot  = udp_socket_robot
_RC.udp_socket_fes    = udp_socket_fes

_RC.FES_toggle = FES_toggle

_RC.Prev_T = Prev_T
_RC.counter = counter
_RC.Prev_T_beta = Prev_T_beta
_RC.counter_beta = counter_beta


def _hardware_cleanup():
    """Best-effort release of the Tiagobot serial port and (if present)
    the glove port. Runs on every exit path of main() — including
    SystemExit from runtime_common when the operator closes the pygame
    window mid-session."""
    try:
        tiago_close_port(tiago, logger)
    except Exception as e:
        logger.log_event(f"Tiagobot close error: {e}", level="error")
    if arduino is not None:
        try:
            arduino.close()
        except Exception as e:
            logger.log_event(f"Glove close error: {e}", level="error")
    try:
        pygame.quit()
    except Exception:
        pass


def run_tiago_gaze_selection_window(gaze_inlet, eeg_state, duration_s):
    """Accumulate gaze samples from the Neon LSL inlet for `duration_s`
    seconds while keeping pygame + EEG state alive.

    Structurally analogous to
    ExperimentDriver_Online_GazeTracking.py:404-543
    (`run_gaze_selection_window`), but stripped to just sample
    accumulation — the Tiagobot variant has no object-tracker, no dwell
    contest, and no VLM bridge. Output is a flat list of
    `(t, x_norm, y_norm, conf)` tuples for
    `Utils.tiagobot_gaze.average_gaze_over_window`.

    LSL channel indices match the calibration script's contract:
    sample[0]=x_px, sample[1]=y_px, sample[15]=conf. Pixels are
    normalized against GAZE_SAMPLE_WIDTH / HEIGHT so the runtime values
    are directly comparable to the calibration centroids.

    Returns None if the operator quits the pygame window mid-window;
    otherwise the (possibly empty) sample list.
    """
    logger.log_event(f"Starting Tiagobot gaze selection window ({duration_s:.2f} s).")
    screen.fill(config.black)
    draw_fixation_cross(screen_width, screen_height)
    pygame.display.flip()

    samples = []
    start_t = time.monotonic()
    end_t = start_t + float(duration_s)

    while time.monotonic() < end_t:
        eeg_state.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return None

        # Drain the LSL inlet — keep all samples (not just latest) so
        # the average / median over the window has the full picture.
        while True:
            sample, t_unix = gaze_inlet.pull_sample(timeout=0.0)
            if not sample:
                break
            try:
                x_px = float(sample[TIAGO_NEON_X_INDEX])
                y_px = float(sample[TIAGO_NEON_Y_INDEX])
                conf = float(sample[TIAGO_NEON_CONF_INDEX])
            except (IndexError, TypeError, ValueError):
                continue
            if not (np.isfinite(x_px) and np.isfinite(y_px) and np.isfinite(conf)):
                continue
            samples.append((
                float(t_unix),
                x_px / TIAGO_GAZE_SAMPLE_W,
                y_px / TIAGO_GAZE_SAMPLE_H,
                conf,
            ))
        # Cooperative yield so we don't hog the GIL against eeg_state.update().
        time.sleep(0.005)

    logger.log_event(
        f"Tiagobot gaze selection window done: {len(samples)} samples in "
        f"{duration_s:.2f} s."
    )
    return samples


def main():
    """Run the online MI/REST trial loop for the gaze-driven Tiagobot.

    Per successful-MI trial: gaze samples are collected for
    TIAGOBOT_GAZE_SELECTION_WINDOW seconds, averaged, and classified to
    one of A-I; Tiagobot extends to that letter, the classification
    window runs, then Tiagobot is sent HOME. Glove (if enabled) closes
    on MI and opens on HOME. If gaze does not resolve, the GO is
    skipped (logged) and the trial advances.
    """
    require_marker_stream(logger=logger)

    logger.log_event("Resolving EEG data stream via LSL...")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    logger.log_event("✅ EEG stream detected and inlet established.")

    logger.log_event("Resolving Pupil Labs Neon gaze stream via LSL...")
    gaze_streams = resolve_stream('type', 'Gaze')
    if not gaze_streams:
        logger.log_event(
            "❌ No LSL stream of type='Gaze' found — required for the "
            "gaze-driven Tiagobot driver. Start Neon Companion + LSL "
            "relay before launching this driver.",
            level="error",
        )
        sys.exit(1)
    gaze_inlet = StreamInlet(gaze_streams[0])
    logger.log_event(f"✅ Gaze stream connected: {gaze_streams[0].name()}")

    eeg_state = EEGStreamState(inlet=inlet, config=config, logger=logger)
    logger.log_event("EEGStreamState object created — ready to pull and process data.")

    trial_sequence = generate_trial_sequence(total_trials=config.TOTAL_TRIALS, max_repeats=config.MAX_REPEATS)
    mode_labels = ["MI" if t == 0 else "REST" for t in trial_sequence]
    logger.log_event(f"Trial Sequence generated: {trial_sequence}")
    logger.log_event(f"Trial Sequence (labeled): {mode_labels}")
    current_trial = 0

    all_results = []
    running = True
    clock = pygame.time.Clock()

    display_fixation_period(duration=3, eeg_state=eeg_state)
    logger.log_event("Initial fixation period complete. Beginning experimental loop.")

    while running and current_trial < len(trial_sequence):
        logger.log_event(f"--- Trial {current_trial+1}/{len(trial_sequence)} START ---")

        # === Fixation cross + trial UI ===
        screen.fill(config.black)
        draw_fixation_cross(screen_width, screen_height)
        draw_arrow_fill(0, screen_width, screen_height)
        draw_ball_fill(0, screen_width, screen_height)
        draw_time_balls(0, screen_width, screen_height)
        pygame.display.flip()
        logger.log_event("Initial screen rendered: fixation cross, bar, ball, and time indicators.")

        # === Countdown + user input ===
        backdoor_mode = None
        waiting_for_press = True
        countdown_start = None
        countdown_duration = 3000  # ms

        while waiting_for_press:
            eeg_state.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    waiting_for_press = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        backdoor_mode = 0  # Force MI
                    elif event.key == pygame.K_DOWN:
                        backdoor_mode = 1  # Force REST
                    elif event.key == pygame.K_SPACE:
                        logger.log_event("Space bar pressed — proceeding without override.")
                    waiting_for_press = False

            if config.TIMING:
                if countdown_start is None:
                    countdown_start = pygame.time.get_ticks()
                    logger.log_event("Countdown timer initiated.")

                elapsed_time = pygame.time.get_ticks() - countdown_start
                draw_time_balls(1, screen_width, screen_height)
                pygame.display.flip()

                if elapsed_time >= countdown_duration:
                    logger.log_event("Countdown expired — proceeding to trial.")
                    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
                    waiting_for_press = False

        if not running:
            logger.log_event("Experiment terminated early via quit event.")
            break

        # === Trial mode selection ===
        if backdoor_mode is not None:
            mode = backdoor_mode
            logger.log_event(f"Backdoor override activated: {'MI' if mode == 0 else 'REST'}")
        else:
            mode = trial_sequence[current_trial]
            logger.log_event(f"Trial mode selected from sequence: {'MI' if mode == 0 else 'REST'}")

        # === Gaze selection window (MI trials only) ===
        # Decide which letter the subject is looking at BEFORE the MI
        # classification window starts. Gaze classification on REST
        # trials is wasted effort (no GO will be sent regardless), so
        # we skip the window entirely on those trials.
        _gaze_selected_letter = None
        if mode == 0:
            _gaze_samples = run_tiago_gaze_selection_window(
                gaze_inlet=gaze_inlet,
                eeg_state=eeg_state,
                duration_s=float(getattr(config, "TIAGOBOT_GAZE_SELECTION_WINDOW", 4.0)),
            )
            if _gaze_samples is None:
                # Operator quit / ESC during the window.
                running = False
                logger.log_event("Experiment terminated during gaze selection window.")
                break
            _gaze_avg = tiago_gaze_average(
                _gaze_samples,
                conf_threshold=float(getattr(config, "TIAGOBOT_GAZE_CONFIDENCE_THRESHOLD", 0.7)),
            )
            if _gaze_avg is None:
                logger.log_event(
                    "Gaze window produced no confidence-passing samples; "
                    "GO will be skipped on this trial."
                )
            else:
                gx, gy = _gaze_avg
                _gaze_selected_letter = tiago_gaze_classify(
                    gx, gy, _tiago_centroids,
                    available_letters=config.TIAGOBOT_TRAJECTORY,
                    max_dist_norm=float(getattr(config, "TIAGOBOT_GAZE_MAX_DIST_NORM", 0.2))
                    if getattr(config, "TIAGOBOT_GAZE_MAX_DIST_NORM", None) is not None
                    else None,
                )
                logger.log_event(
                    f"Gaze classification — avg_norm=({gx:.3f}, {gy:.3f}), "
                    f"letter={_gaze_selected_letter!r}, "
                    f"available={list(config.TIAGOBOT_TRAJECTORY)}"
                )

        # === Baseline ===
        try:
            eeg_state.compute_baseline(duration_sec=config.BASELINE_DURATION)
            logger.log_event(
                f"Computed baseline: shape={eeg_state.baseline_mean.shape}, "
                f"duration={config.BASELINE_DURATION}s"
            )
        except ValueError as e:
            logger.log_event(f"⚠️ Could not compute baseline: {e}")
            continue

        # === Classification ===
        logger.log_event(f"Starting feedback classification — Mode: {'MI' if mode == 0 else 'REST'}")
        prediction, confidence, leaky_integrator, trial_probs, earlystop_flag = show_feedback(
            duration=config.TIME_MI,
            mode=mode,
            eeg_state=eeg_state
        )

        pygame.display.flip()
        pygame.event.get()  # heartbeat to OS
        logger.log_event(f"Classification result — Predicted: {prediction}, Ground Truth: {200 if mode == 0 else 100}")

        append_trial_probabilities_to_csv(
            trial_probabilities=trial_probs,
            mode=mode,
            trial_number=current_trial + 1,
            predicted_label=prediction,
            early_cutout=earlystop_flag,
            mi_threshold=config.THRESHOLD_MI,
            rest_threshold=config.THRESHOLD_REST,
            logger=logger,
            phase="MI" if mode == 0 else "REST"
        )

        logger.log_event(f"Stored decoder output for trial {current_trial+1}: {len(trial_probs)} timepoints.")

        predictions_list.append(prediction)
        ground_truth_list.append(200 if mode == 0 else 100)

        # === Feedback branches ===
        if mode == 0:
            if prediction == 200 and _gaze_selected_letter is not None:
                # Correct MI AND gaze resolved -> drive Tiagobot.
                messages = ["Correct", f"Robot Move -> {_gaze_selected_letter}"]
                colors = [config.green, config.green]
                offsets = [-100, 100]
                duration = 0.01
                should_hold_and_classify = True
                logger.log_event(
                    f"Prediction correct for MI — driving Tiagobot to "
                    f"gaze-selected letter {_gaze_selected_letter!r} "
                    f"(and FES / glove if toggled)."
                )

                # Tiagobot GO: send the gaze-classified letter.
                selected_letter = _gaze_selected_letter
                tiago_send_letter(tiago, selected_letter, logger)

                if FES_toggle == 1:
                    send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_MOTOR_GO", logger=logger)
                else:
                    logger.log_event("FES disabled — skipping motor stimulation.")

                send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_BEGIN"], logger=logger)

            elif prediction == 200 and _gaze_selected_letter is None:
                # MI correct but gaze didn't resolve to a letter. Per
                # plan §6.3 step 4 (user-confirmed 2026-05-19): skip
                # the GO and log. No random fallback — failing visibly
                # is the whole point of this branch. The trial still
                # shows feedback to keep the cadence steady.
                messages = ["Correct", "No gaze target"]
                colors = [config.green, config.orange]
                offsets = [-100, 100]
                duration = config.TIME_STATIONARY
                should_hold_and_classify = False
                logger.log_event(
                    "Prediction correct for MI but gaze did not resolve "
                    "to a letter — skipping Tiagobot GO for this trial."
                )

            elif prediction is None:  # Ambiguous
                messages = ["Ambiguous", "Robot Stationary"]
                colors = [config.orange, config.white]
                offsets = [-100, 100]
                duration = config.TIME_STATIONARY
                should_hold_and_classify = False
                logger.log_event("Prediction ambiguous for MI — Tiagobot remains stationary.")

            else:  # Incorrect
                messages = ["Incorrect", "Robot Stationary"]
                colors = [config.red, config.white]
                offsets = [-100, 100]
                duration = config.TIME_STATIONARY
                should_hold_and_classify = False
                logger.log_event("Prediction incorrect for MI — Tiagobot remains stationary.")

        else:  # REST
            if prediction == 100:  # Correct
                messages = ["Correct", "Robot Stationary"]
                colors = [config.green, config.green]
                offsets = [-100, 100]
                duration = config.TIME_STATIONARY
                logger.log_event("Prediction correct for REST — Tiagobot remains stationary.")
            elif prediction is None:
                messages = ["Ambiguous", "Robot Stationary"]
                colors = [config.orange, config.white]
                offsets = [-100, 100]
                duration = config.TIME_STATIONARY
                logger.log_event("Prediction ambiguous for REST — Tiagobot remains stationary.")
            else:
                messages = ["Incorrect", "Robot Stationary"]
                colors = [config.red, config.white]
                offsets = [-100, 100]
                duration = config.TIME_STATIONARY
                logger.log_event("Prediction incorrect for REST — Tiagobot remains stationary.")

            should_hold_and_classify = False

        # Display feedback. udp_messages is always None for Tiagobot — the
        # serial GO was already sent above. The helper runs its UI loop and
        # skips its (Harmony-specific) UDP staging logic.
        logger.log_event(f"Displaying feedback: '{messages[0]}' | Action: '{messages[1]}' | Duration: {duration}s")
        display_multiple_messages_with_udp(
            messages=messages,
            colors=colors,
            offsets=offsets,
            duration=duration,
            udp_messages=None,
            udp_socket=udp_socket_robot,
            udp_ip=config.UDP_ROBOT["IP"],
            udp_port=config.UDP_ROBOT["PORT"],
            logger=logger,
            eeg_state=eeg_state
        )

        # If MI-correct, continue classifying for the early window, then
        # wait for the actual Tiagobot motion to complete (variable, 5-30+ s
        # depending on the chosen letter), grip, retract with the object,
        # release at home. The Tiagobot sketch has no mid-motion STOP, so
        # we always wait — even on early-stop the actuator runs to target.
        if should_hold_and_classify:
            logger.log_event("Entering real-time classification window during Tiagobot movement...")
            final_class_robot, robot_probs, robot_earlystop = hold_messages_and_classify(
                messages=messages,
                colors=colors,
                offsets=offsets,
                duration=config.TIME_ROB,
                mode=0,
                udp_socket=udp_socket_robot,
                udp_ip=config.UDP_ROBOT["IP"],
                udp_port=config.UDP_ROBOT["PORT"],
                eeg_state=eeg_state,
                leaky_integrator=leaky_integrator
            )
            append_trial_probabilities_to_csv(
                trial_probabilities=robot_probs,
                mode=0,
                trial_number=current_trial + 1,
                predicted_label=final_class_robot,
                early_cutout=robot_earlystop,
                mi_threshold=config.THRESHOLD_MI,
                rest_threshold=config.THRESHOLD_REST,
                logger=logger,
                phase="ROBOT"
            )

            # Drive-loop tick: pumped during the variable-time waits to
            # keep the pygame window responsive, propagate QUIT cleanly,
            # and stop the LSL EEG buffer from backing up.
            def _drive_tick():
                eeg_state.update()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        raise SystemExit
                pygame.display.flip()

            # Wait for the actual GO motion to finish. The sketch prints
            # "Target Location Reached." when the extend loop exits; the
            # serial port has been buffering this line while
            # hold_messages_and_classify ran, so on a fast move we read
            # it immediately, on a slow move we wait until it arrives.
            logger.log_event("Waiting for Tiagobot reach completion...")
            reached = tiago_wait_for_completion(
                tiago, TARGET_REACHED_MARKER, timeout=60.0,
                logger=logger, on_tick=_drive_tick,
            )
            if not reached:
                logger.log_event(
                    "Tiagobot reach timed out after 60 s — proceeding to grip anyway.",
                    level="error",
                )

            # Grip the target. On early-stop we skip the grip entirely
            # so the operator's "bad MI" interruption doesn't grasp.
            if not robot_earlystop and arduino is not None:
                arduino.write(config.ARDUINO_CMD_MI)
                logger.log_event("Glove closing — gripping target.")

            # Grip-hold period: lets the glove finish its close motion
            # before HOME starts retracting (otherwise the actuator pulls
            # back while the glove is still mid-close). Duration tunable
            # via config.TIAGOBOT_GRIP_HOLD_DURATION (default 4 s).
            if not robot_earlystop:
                grip_hold = float(getattr(config, "TIAGOBOT_GRIP_HOLD_DURATION", 4))
                display_fixation_period(duration=grip_hold, eeg_state=eeg_state)

            # Send HOME — Tiagobot retracts with the gripped object.
            send_udp_message(
                udp_socket_marker,
                config.UDP_MARKER["IP"],
                config.UDP_MARKER["PORT"],
                config.TRIGGERS["ROBOT_HOME"],
                logger=logger,
            )
            tiago_send_home(tiago, logger)
            logger.log_event("Sent HOME to Tiagobot — retracting.")

            # Wait for the retract to actually finish before opening
            # the glove (so the object goes back to the home position
            # rather than being dropped mid-retract).
            logger.log_event("Waiting for Tiagobot home completion...")
            homed = tiago_wait_for_completion(
                tiago, HOMED_MARKER, timeout=60.0,
                logger=logger, on_tick=_drive_tick,
            )
            if not homed:
                logger.log_event(
                    "Tiagobot home timed out after 60 s — releasing glove anyway.",
                    level="error",
                )

            # Release at home.
            if not robot_earlystop and arduino is not None:
                arduino.write(config.ARDUINO_CMD_REST)
                logger.log_event("Glove opening — releasing at home.")

            display_fixation_period(duration=3, eeg_state=eeg_state)
            logger.log_event("Robot reset fixation (3s) complete.")

        logger.log_trial_summary(
            trial_number=current_trial + 1,
            true_label=200 if mode == 0 else 100,
            predicted_label=prediction,
            early_cutout=earlystop_flag,
            accuracy_threshold=config.THRESHOLD_MI if mode == 0 else config.THRESHOLD_REST,
            confidence=confidence,
            num_predictions=len(trial_probs)
        )

        display_fixation_period(duration=3, eeg_state=eeg_state)
        logger.log_event(f"Trial {current_trial+1} complete. Proceeding to next.")

        current_trial += 1
        pygame.display.flip()
        clock.tick(60)

    if current_trial == len(trial_sequence) and config.SAVE_ADAPTIVE_T:
        try:
            save_transform(
                _RC.Prev_T,
                _RC.counter,
                adaptive_T_path,
                T_beta=_RC.Prev_T_beta,
                counter_beta=_RC.counter_beta,
            )
        except Exception as e:
            logger.log_event(f"⚠️ Could not save transform to {adaptive_T_path}: {e}")

    log_confusion_matrix_from_trial_summary(logger)
    logger.log_event("run complete")


if __name__ == "__main__":
    # try/finally so hardware cleanup runs on any exit path: clean
    # session end, SystemExit raised by runtime_common when the
    # operator closes the pygame window, KeyboardInterrupt, etc.
    try:
        main()
    finally:
        _hardware_cleanup()
