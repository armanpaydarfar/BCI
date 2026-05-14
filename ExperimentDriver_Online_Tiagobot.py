"""
ExperimentDriver_Online_Tiagobot.py

Online MI/REST BCI driver that targets the Tiagobot mobile-arm device.
Tiagobot is a servo + linear-actuator combo driven over USB serial by the
Arduino sketch at tools/tiago_arduino/Final_code.ino.

Flow per MI-success trial:
1. Pick a preset letter from config.TIAGOBOT_TRAJECTORY.
2. Write `'{analog},{angle},{delay}\\n'` to the Tiagobot Arduino — the
   actuator moves to the preset and holds.
3. (Optional) Write config.ARDUINO_CMD_MI to the glove Arduino to close it.
4. Run hold_messages_and_classify for TIME_ROB seconds.
5. Send `'h\\n'` to Tiagobot — actuator retracts and servo centers.
6. (Optional) Write config.ARDUINO_CMD_REST to the glove Arduino to open it.

This driver targets Tiagobot only. Harmony's UDP control vocabulary is not
used — `udp_socket_robot` is still created because runtime_common helpers
take it as a parameter, but no commands are dialed to it. Tiagobot's wire
protocol is serial-only.

The glove is optional: gate via config.TIAGOBOT_USE_GLOVE. When True (and
config.ARDUINO_PORT is set), the driver opens the glove port and emits the
same b'1' / b'0' bytes as ExperimentDriver_Online_Glove.py on MI / HOME.
"""

import pygame
import socket
import pickle
import datetime
import time
import serial
import sys
import os
import random
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
    "TIAGOBOT_USE_GLOVE", "SIMULATION_MODE",
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
# open_tiago_port returns None in SIMULATION_MODE or when port is unset.
# Real open failures raise; we let them propagate per fail-fast policy on
# Tier 1 hardware paths (CLAUDE.md "Error Handling").
tiago = open_tiago_port(config.TIAGOBOT_PORT, config.TIAGOBOT_BAUD, logger)
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
    if config.ARDUINO_PORT:
        try:
            logger.log_event(f"Connecting to Glove (Arduino) on {config.ARDUINO_PORT}...")
            arduino = serial.Serial(config.ARDUINO_PORT, config.ARDUINO_BAUD, timeout=0.1)
            time.sleep(2)  # Arduino reset wait
            logger.log_event("✅ Glove connected successfully.")
        except Exception as e:
            logger.log_event(f"❌ Error connecting to Glove: {e}", level="error")
            arduino = None
    else:
        logger.log_event("ℹ️ TIAGOBOT_USE_GLOVE=True but ARDUINO_PORT is empty — glove disabled.")
else:
    logger.log_event("ℹ️ Glove integration disabled (TIAGOBOT_USE_GLOVE=False).")


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


def main():
    """Run the online MI/REST trial loop for Tiagobot.

    Per successful-MI trial: Tiagobot extends to a random preset, the
    classification window runs, then Tiagobot is sent HOME. Glove (if
    enabled) closes on MI and opens on HOME.
    """
    require_marker_stream(logger=logger)

    logger.log_event("Resolving EEG data stream via LSL...")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    logger.log_event("✅ EEG stream detected and inlet established.")

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
            if prediction == 200:  # Correct MI -> drive Tiagobot
                messages = ["Correct", "Robot Move"]
                colors = [config.green, config.green]
                offsets = [-100, 100]
                duration = 0.01
                should_hold_and_classify = True
                logger.log_event("Prediction correct for MI — triggering Tiagobot (and FES / glove if toggled)")

                # Tiagobot GO: pick a preset letter and write the CSV.
                selected_letter = random.choice(config.TIAGOBOT_TRAJECTORY)
                tiago_send_letter(tiago, selected_letter, logger)

                if FES_toggle == 1:
                    send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_MOTOR_GO", logger=logger)
                else:
                    logger.log_event("FES disabled — skipping motor stimulation.")

                send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_BEGIN"], logger=logger)

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

        # If MI-correct, continue classifying during the actuator move and
        # then HOME regardless of early-stop (the Tiagobot sketch no longer
        # auto-retracts at end of GO).
        if should_hold_and_classify:
            if arduino is not None:
                arduino.write(config.ARDUINO_CMD_MI)
                logger.log_event("Glove closing after successful MI.")

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

            if arduino is not None:
                arduino.write(config.ARDUINO_CMD_REST)
                logger.log_event("Opening glove.")

            # Pre-home fixation only on a clean MI window; on early-stop
            # the time has already been spent on the abort.
            if not robot_earlystop:
                logger.log_event("Robot fixation period 2s before homing.")
                display_fixation_period(duration=2, eeg_state=eeg_state)

            # Always send HOME — Tiagobot does not auto-retract.
            send_udp_message(
                udp_socket_marker,
                config.UDP_MARKER["IP"],
                config.UDP_MARKER["PORT"],
                config.TRIGGERS["ROBOT_HOME"],
                logger=logger,
            )
            tiago_send_home(tiago, logger)
            logger.log_event("Sent HOME to Tiagobot at end of MI trial.")

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

    # Best-effort hardware cleanup.
    tiago_close_port(tiago, logger)
    if arduino is not None:
        try:
            arduino.close()
        except Exception as e:
            logger.log_event(f"Glove close error: {e}", level="error")

    pygame.quit()


if __name__ == "__main__":
    main()
