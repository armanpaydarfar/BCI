import pygame
import socket
import time
import sys
import pickle
import datetime
import os
from pylsl import StreamInlet, resolve_stream
# MNE for real-time EEG processing
import mne
mne.set_log_level("WARNING")  # Options: "ERROR", "WARNING", "INFO", "DEBUG"
# Preprocessing functions (updated for MNE integration)
# Visualization utilities
from Utils.visualization import (
    draw_arrow_fill,
    draw_ball_fill,
    draw_fixation_cross,
    draw_time_balls,
    draw_progress_bar
)

# Experiment utilities
from Utils.experiment_utils import (
    generate_trial_sequence,
    save_transform,
    load_transform
)

#import EEG stream object for tracking/filtering 
from Utils.EEGStreamState import EEGStreamState

# Networking utilities
from Utils.networking import send_udp_message,display_multiple_messages_with_udp

# Stream utilities (LSL channel names)
from Utils.stream_utils import get_channel_names_from_lsl

# Configuration parameters
import config

# Performance evaluation (classification metrics)
from sklearn.metrics import confusion_matrix


from pathlib import Path
from Utils.logging_manager import LoggerManager

# Import common experiment functions
from Utils.runtime_common import (
    log_confusion_matrix_from_trial_summary,
    append_trial_probabilities_to_csv,
    display_fixation_period,
    hold_messages_and_classify,
    show_feedback,
)

# Also import the module itself for wiring globals
import Utils.runtime_common as _RC

# Initialize experiment logger (auto-detects active run or falls back to Debug)
logger = LoggerManager.auto_detect_from_subject(
    subject=config.TRAINING_SUBJECT,
    base_path=Path(config.DATA_DIR),
    mode="online"  # <-- NEW: flag to determine log directory and filename suffix
)


# Log experiment configuration snapshot
loggable_fields = [
    # Standard fields
    "UDP_MARKER", "UDP_ROBOT", "UDP_FES",
    "ARM_SIDE", "TOTAL_TRIALS", "MAX_REPEATS",
    "TIME_MI", "TIME_ROB", "TIME_STATIONARY",
    "SHAPE_MAX", "SHAPE_MIN", "ROBOT_TRAJECTORY",
    "FES_toggle", "FES_CHANNEL", "FES_TIMING_OFFSET",
    "WORKING_DIR", "DATA_DIR", "MODEL_PATH",
    "TRAINING_SUBJECT",
    # Online-specific fields
    "CLASSIFY_WINDOW", "ACCURACY_THRESHOLD", "THRESHOLD_MI", "THRESHOLD_REST",
    "RELAXATION_RATIO", "MIN_PREDICTIONS", "SURFACE_LAPLACIAN_TOGGLE",
    "SELECT_MOTOR_CHANNELS", "INTEGRATOR_ALPHA", "SHRINKAGE_PARAM",
    "LEDOITWOLF", "RECENTERING", "UPDATE_DURING_MOVE"
]
config_log_subset = {
    key: getattr(config, key) for key in loggable_fields if hasattr(config, key)
}
logger.save_config_snapshot(config_log_subset)


eeg_dir = logger.log_base / "eeg"
adaptive_T_path = eeg_dir / "adaptive_T.pkl"

Prev_T, counter = load_transform(adaptive_T_path)
if Prev_T is None:
    counter = 0
    logger.log_event("ℹ️ No adaptive transform found — starting fresh with counter = 0.")
else:
    logger.log_event(f"✅ Loaded adaptive transform with counter = {counter}")

logger.log_event("Logger initialized for online experimental driver.")


pygame.init()

if config.BIG_BROTHER_MODE:
    # External display is at +0+0 (HDMI-1), so force window to (0,0)
    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
    screen = pygame.display.set_mode((1920, 1080), pygame.NOFRAME)
    logger.log_event("🎥 Big Brother Mode ON — window placed at (0,0) on external monitor (HDMI-1).")
else:
    # Default fullscreen on active display (where launched)
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    logger.log_event("👤 Big Brother Mode OFF — fullscreen on active display.")

# Set title and get screen dimensions for animations
pygame.display.set_caption("EEG Online Interactive Loop")
info = pygame.display.Info()
screen_width = info.current_w
screen_height = info.current_h
logger.log_event("Pygame initialized and display configured.")

# UDP Settings
udp_socket_marker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket_robot = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket_fes = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
logger.log_event("UDP sockets initialized for marker, robot, and FES channels.")

FES_toggle = config.FES_toggle
logger.log_event(f"FES toggle status: {'Enabled' if FES_toggle else 'Disabled'}.")

# Construct the correct model path based on the subject
subject_model_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "models")
subject_model_path = os.path.join(subject_model_dir, f"sub-{config.TRAINING_SUBJECT}_model.pkl")

# Load the trained model from the subject directory
try:
    with open(subject_model_path, 'rb') as f:
        model = pickle.load(f)
    logger.log_event(f"✅ Model successfully loaded from: {subject_model_path}")
except FileNotFoundError:
    logger.log_event(f"❌ Error: Model file '{subject_model_path}' not found. Ensure the model has been trained.", level="error")
    exit(1)



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


# (Optional) Log intended precomputed mean/std loading if re-enabled
# logger.log_event(\"Skipped loading precomputed mean/std: section commented out.\")

# Initialize runtime structures
predictions_list = []
ground_truth_list = []

fs = config.FS

# (Optional) Commented out rolling normalization
# logger.log_event("Rolling normalization block currently disabled.")
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

# Adaptive recentering state
_RC.Prev_T = Prev_T
_RC.counter = counter


def perform_master_positioning(
    eeg_state, udp_socket_robot, udp_socket_marker,
    screen, screen_width, screen_height, logger
):
    """
    Unlocks the master arm, allows user to position it with a timed visual,
    then locks it back in place.
    """

    # Pull duration from config (default fallback = 3s)
    duration = getattr(config, "TIME_MASTER_MOVE", 5)

    # Determine which arm is the master (opposite of ARM_SIDE)
    master_arm = "Left" if config.ARM_SIDE.lower() == "right" else "Right"

    # --- 1. UNLOCK master arm ---
    send_udp_message(
        udp_socket_robot,
        config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"],
        config.ROBOT_OPCODES["MASTER_UNLOCK"],
        logger=logger,
        expect_ack=True,           # <— wait & log ACK:c
        max_retries=1
    )

    send_udp_message(
        udp_socket_marker,
        config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"],
        config.TRIGGERS["MASTER_UNLOCK"],
        logger=logger
    )
    logger.log_event(f"{master_arm} arm UNLOCKED — user may position arm.")

    # --- 2. Timed positioning with progress visualization ---
    start_time = time.time()
    while time.time() - start_time < duration:
        elapsed_time = time.time() - start_time
        progress = elapsed_time / duration

        screen.fill(config.black)
        draw_arrow_fill(0, screen_width, screen_height, show_threshold=True)
        draw_ball_fill(0, screen_width, screen_height, show_threshold=True)
        draw_fixation_cross(screen_width, screen_height)
        draw_time_balls(0, screen_width, screen_height)
        # Green progress bar
        draw_progress_bar(progress, screen_width, screen_height)

        # Instruction text pushed further down (~85% of screen height)
        msg = pygame.font.SysFont(None, 72).render(
            f"Position {master_arm.upper()} Arm", True, config.white
        )
        msg_y = int(screen_height * 0.75)
        screen.blit(msg, (screen_width // 2 - msg.get_width() // 2, msg_y))


        pygame.display.flip()
        eeg_state.update()

    # --- 3. LOCK master arm ---
    send_udp_message(
        udp_socket_robot,
        config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"],
        config.ROBOT_OPCODES["MASTER_LOCK"],
        logger=logger,
        expect_ack=True,           # <— wait & log ACK:c
        max_retries=1
    )

    send_udp_message(
        udp_socket_marker,
        config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"],
        config.TRIGGERS["MASTER_LOCK"],
        logger=logger
    )

    logger.log_event(f"{master_arm} arm LOCKED in position.")

    # --- 4. Short fixation cross before MI starts ---
    display_fixation_period(duration=3, eeg_state=eeg_state)
    # --- Mandatory pre-trial 3s countdown ---
    countdown_start = pygame.time.get_ticks()
    logger.log_event("Pre-trial countdown initiated (3s).")

    while pygame.time.get_ticks() - countdown_start < 3000:  # 3 seconds
        eeg_state.update()

        # Draw "time balls" in countdown mode (code=1)
        screen.fill(config.black)
        draw_fixation_cross(screen_width, screen_height)
        draw_arrow_fill(0, screen_width, screen_height)
        draw_ball_fill(0, screen_width, screen_height)
        draw_time_balls(0, screen_width, screen_height)
        draw_time_balls(1, screen_width, screen_height)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logger.log_event("Experiment terminated early during pre-trial countdown.")
                pygame.quit()
                sys.exit()



def main():
    # === Main Game Loop Initialization ===

    # Connect to EEG stream
    logger.log_event("Resolving EEG data stream via LSL...")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    logger.log_event("✅ EEG stream detected and inlet established.")

    # Initialize EEG handler
    eeg_state = EEGStreamState(inlet=inlet, config=config, logger=logger)
    logger.log_event("EEGStreamState object created — ready to pull and process data.")

    # Generate and log trial sequence
    trial_sequence = generate_trial_sequence(total_trials=config.TOTAL_TRIALS, max_repeats=config.MAX_REPEATS)
    mode_labels = ["MI" if t == 0 else "REST" for t in trial_sequence]
    logger.log_event(f"Trial Sequence generated: {trial_sequence}")
    logger.log_event(f"Trial Sequence (labeled): {mode_labels}")
    current_trial = 0

    # Initialize experiment state
    all_results = []
    running = True
    clock = pygame.time.Clock()

    # Begin with fixation screen
    display_fixation_period(duration=3, eeg_state=eeg_state)
    logger.log_event("Initial fixation period complete. Beginning experimental loop.")

    while running and current_trial < len(trial_sequence):
        logger.log_event(f"--- Trial {current_trial+1}/{len(trial_sequence)} START ---")

        # === 1. Fixation Cross and Trial UI ===
        screen.fill(config.black)
        draw_fixation_cross(screen_width, screen_height)
        draw_arrow_fill(0, screen_width, screen_height)
        draw_ball_fill(0, screen_width, screen_height)
        draw_time_balls(0, screen_width, screen_height)
        pygame.display.flip()
        logger.log_event("Initial screen rendered: fixation cross, bar, ball, and time indicators.")

        # === 2. Countdown + User Input Handling ===
        backdoor_mode = None
        waiting_for_press = True
        countdown_start = None
        countdown_duration = 3000  # ms

        while waiting_for_press:
            eeg_state.update()
            # Handle input events
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

            # If TIMING mode, do automatic countdown
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

        # Handle early quit
        if not running:
            logger.log_event("Experiment terminated early via quit event.")
            break

        # === 3. Trial Mode Selection ===
        if backdoor_mode is not None:
            mode = backdoor_mode
            logger.log_event(f"Backdoor override activated: {'MI' if mode == 0 else 'REST'}")
        else:
            mode = trial_sequence[current_trial]
            logger.log_event(f"Trial mode selected from sequence: {'MI' if mode == 0 else 'REST'}")

        # === 4. Trial-specific prep ===

        # MI: do master positioning first
        logger.log_event("Performing master arm positioning phase before MI feedback...")
        perform_master_positioning(
            eeg_state=eeg_state,
            udp_socket_robot=udp_socket_robot,
            udp_socket_marker=udp_socket_marker,
            screen=screen,
            screen_width=screen_width,
            screen_height=screen_height,
            logger=logger
        )

        # === 4. Extract Baseline from EEG Buffer ===

        try:
            eeg_state.compute_baseline(duration_sec=config.BASELINE_DURATION)
            logger.log_event(
                f"Computed baseline: shape={eeg_state.baseline_mean.shape}, "
                f"duration={config.BASELINE_DURATION}s"
            )
        except ValueError as e:
            logger.log_event(f"⚠️ Could not compute baseline: {e}")
            continue  # Skip this trial if not enough data

                    
        # Show feedback and perform classification
        logger.log_event(f"Starting feedback classification — Mode: {'MI' if mode == 0 else 'REST'}")
        prediction, confidence, leaky_integrator, trial_probs, earlystop_flag = show_feedback(
            duration=config.TIME_MI,
            mode=mode,
            eeg_state=eeg_state
        )
        pygame.display.flip()
        pygame.event.get()     # heartbeat to OS
        # Log the classification result
        logger.log_event(f"Classification result — Predicted: {prediction}, Ground Truth: {200 if mode == 0 else 100}")

        # Log and store classification outcome
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
        # Red Arrow Mode (MI)
        if mode == 0:
            if prediction == 200:  # Correct
                messages = ["Correct", "Robot Move"]
                colors = [config.green, config.green]
                offsets = [-100, 100]
                udp_messages = [config.ROBOT_OPCODES["GO"]]
                duration = 0.01
                should_hold_and_classify = True

                logger.log_event("Prediction correct for MI — triggering robot movement (and FES if toggled)")

                if FES_toggle == 1:
                    send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_MOTOR_GO", logger=logger)
                else:
                    logger.log_event("FES disabled — skipping motor stimulation.")

                send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_BEGIN"], logger=logger)

            elif prediction is None:  # Ambiguous
                messages = ["Ambiguous", "Robot Stationary"]
                colors = [config.orange, config.white]  # Or config.yellow if orange isn't defined
                offsets = [-100, 100]
                udp_messages = [config.ROBOT_OPCODES["HOME"]]
                duration = config.TIME_STATIONARY
                should_hold_and_classify = False

                logger.log_event("Prediction ambiguous for MI — robot remains stationary.")

            else:  # Incorrect
                messages = ["Incorrect", "Robot Stationary"]
                colors = [config.red, config.white]
                offsets = [-100, 100]
                udp_messages = [config.ROBOT_OPCODES["HOME"]]
                duration = config.TIME_STATIONARY
                should_hold_and_classify = False

                logger.log_event("Prediction incorrect for MI — robot remains stationary.")

        # Blue Ball Mode (REST)
        else:
            if prediction == 100:  # Correct
                messages = ["Correct", "Robot Stationary"]
                colors = [config.green, config.green]
                offsets = [-100, 100]
                udp_messages = [config.ROBOT_OPCODES["HOME"]]
                duration = config.TIME_STATIONARY

                logger.log_event("Prediction correct for REST — robot remains stationary.")

            elif prediction is None:  # Ambiguous
                messages = ["Ambiguous", "Robot Stationary"]
                colors = [config.orange, config.white]
                offsets = [-100, 100]
                udp_messages = [config.ROBOT_OPCODES["HOME"]]
                duration = config.TIME_STATIONARY

                logger.log_event("Prediction ambiguous for REST — robot remains stationary.")

            else:  # Incorrect
                messages = ["Incorrect", "Robot Stationary"]
                colors = [config.red, config.white]
                offsets = [-100, 100]
                udp_messages = [config.ROBOT_OPCODES["HOME"]]
                duration = config.TIME_STATIONARY

                logger.log_event("Prediction incorrect for REST — robot remains stationary.")


            should_hold_and_classify = False  # No secondary classification logic in REST
        # Display the feedback messages and send UDP commands (if any)
        logger.log_event(f"Displaying feedback: '{messages[0]}' | Action: '{messages[1]}' | Duration: {duration}s")
        display_multiple_messages_with_udp(
            messages=messages,
            colors=colors,
            offsets=offsets,
            duration=duration,
            udp_messages=udp_messages,
            udp_socket=udp_socket_robot,
            udp_ip=config.UDP_ROBOT["IP"],
            udp_port=config.UDP_ROBOT["PORT"],
            logger=logger,  # Pass logger to internal UDP calls
            eeg_state=eeg_state  # Add EEG buffer updates during display loop
        )


        # If trial was a correct MI, continue classification during robot movement
        if should_hold_and_classify:
            logger.log_event("Entering real-time classification window during robot movement...")
            final_class_robot, robot_probs, robot_earlystop = hold_messages_and_classify(
                messages=messages, 
                colors=colors, 
                offsets=offsets, 
                duration=config.TIME_ROB,  # Monitor for 7s out of total 13s movement
                mode=0,  # Motor Imagery
                udp_socket=udp_socket_robot, 
                udp_ip=config.UDP_ROBOT["IP"], 
                udp_port=config.UDP_ROBOT["PORT"],
                eeg_state=eeg_state,
                leaky_integrator=leaky_integrator
            )
            append_trial_probabilities_to_csv(
                trial_probabilities=robot_probs,
                mode=0,  # still MI internally
                trial_number=current_trial + 1,
                predicted_label=final_class_robot,
                early_cutout=robot_earlystop,
                mi_threshold=config.THRESHOLD_MI,
                rest_threshold=config.THRESHOLD_REST,
                logger=logger,
                phase="ROBOT"
            )
            # --- Robot HOME + reset for MI trials ---
            display_fixation_period(duration=1, eeg_state=eeg_state)
            # Send HOME opcode to robot
            acked, _ = send_udp_message(
                udp_socket_robot,
                config.UDP_ROBOT["IP"],
                config.UDP_ROBOT["PORT"],
                config.ROBOT_OPCODES["HOME"],   # this is 'h'
                logger=logger,
                expect_ack=True,                # <--- wait for ACK
                ack_timeout=1.0,                # optional, default 0.5s
                max_retries=1                   # optional, resend once if timeout
            )

            logger.log_event("Sent HOME opcode to robot at end of MI trial.")
            # Run the same 3s reset fixation
            display_fixation_period(duration=4, eeg_state=eeg_state)
            logger.log_event("Robot reset fixation (4s) complete.")


        logger.log_trial_summary(
            trial_number=current_trial + 1,
            true_label=200 if mode == 0 else 100,
            predicted_label=prediction,
            early_cutout=earlystop_flag,
            accuracy_threshold=config.THRESHOLD_MI if mode == 0 else config.THRESHOLD_REST,
            confidence=confidence,
            num_predictions=len(trial_probs)
        )

        # Inter-trial fixation (common to all trials)
        display_fixation_period(duration=3, eeg_state=eeg_state)
        logger.log_event(f"Trial {current_trial+1} complete. Proceeding to next.")

        # Advance trial index and frame rate
        current_trial += 1
        pygame.display.flip()
        clock.tick(60)

    if current_trial == len(trial_sequence) and config.SAVE_ADAPTIVE_T:
        try:
            save_transform(Prev_T, counter, adaptive_T_path)
        except Exception as e:
            logger.log_event(f"⚠️ Could not save transform to {adaptive_T_path}: {e}")

    log_confusion_matrix_from_trial_summary(logger)
    logger.log_event(f"run complete")
    pygame.quit()

if __name__ == "__main__":
    main()
