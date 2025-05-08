import pygame
import socket
import time
import sys
import pickle
import datetime
import os
import csv
import pandas as pd
from pylsl import StreamInlet, resolve_stream
import numpy as np
from pyriemann.estimation import Shrinkage
from pyriemann.estimation import Shrinkage
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, lfilter, lfilter_zi
from pyriemann.utils.geodesic import geodesic_riemann
from pyriemann.utils.base import invsqrtm
from sklearn.covariance import LedoitWolf

# MNE for real-time EEG processing
import mne
mne.set_log_level("WARNING")  # Options: "ERROR", "WARNING", "INFO", "DEBUG"
# Preprocessing functions (updated for MNE integration)
from Utils.preprocessing import (
    butter_bandpass,
    filter_with_state,
    apply_car_filter,
    apply_notch_filter,
    flatten_single_segment,
    extract_and_flatten_segment,
    parse_eeg_and_eog,
    remove_eog_artifacts,
    select_motor_channels,
)

# Visualization utilities
from Utils.visualization import (
    draw_arrow_fill,
    draw_ball_fill,
    draw_fixation_cross,
    draw_time_balls,
)

# Experiment utilities
from Utils.experiment_utils import (
    generate_trial_sequence,
    display_multiple_messages_with_udp,
    LeakyIntegrator,
    RollingScaler,
)

# Networking utilities
from Utils.networking import send_udp_message

# Stream utilities (LSL channel names)
from Utils.stream_utils import get_channel_names_from_lsl

# Configuration parameters
import config

# Performance evaluation (classification metrics)
from sklearn.metrics import confusion_matrix


from pathlib import Path
from Utils.logging_manager import LoggerManager

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
logger.log_event("Logger initialized for online experimental driver.")


# Initialize Pygame with dimensions from config
pygame.init()
screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
pygame.display.set_caption("BCI Online Interactive Loop")

screen_width = config.SCREEN_WIDTH
screen_height = config.SCREEN_HEIGHT
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

# (Optional) Log intended precomputed mean/std loading if re-enabled
# logger.log_event(\"Skipped loading precomputed mean/std: section commented out.\")

# Initialize runtime structures
predictions_list = []
ground_truth_list = []

fs = config.FS
'''
# b, a = butter_bandpass(config.LOWCUT, config.HIGHCUT, fs, order=4)  # ← unused, from earlier iteration
logger.log_event(f"Sampling frequency set to {fs} Hz. Filter coefficients are unused in this version.")

global filter_states
filter_states = {}
'''
global Prev_T
global counter
counter = 0
# (Optional) Commented out rolling normalization
# logger.log_event("Rolling normalization block currently disabled.")
SESSION_TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logger.log_event(f"Session timestamp set: {SESSION_TIMESTAMP}")


def append_trial_probabilities_to_csv(trial_probabilities, mode, trial_number,
                                      predicted_label, early_cutout,
                                      mi_threshold, rest_threshold, logger):
    correct_class = 200 if mode == 0 else 100
    trial_probabilities = np.array(trial_probabilities)

    if trial_probabilities.shape[1] != 3:
        logger.log_event(f"❌ Error: Unexpected shape {trial_probabilities.shape}. Expected (N,3). Skipping save.")
        return

    for row in trial_probabilities:
        timestamp, prob_rest, prob_mi = row
        logger.log_decoder_output(
            trial=trial_number,
            timestamp=timestamp,
            prob_mi=prob_mi,
            prob_rest=prob_rest,
            true_label=correct_class,
            predicted_label=predicted_label,
            early_cutout=early_cutout,
            mi_threshold=mi_threshold,
            rest_threshold=rest_threshold
        )

    logger.log_event(
        f"✅ Logged {len(trial_probabilities)} rows for Trial {trial_number} | "
        f"True: {correct_class}, Predicted: {predicted_label}, Early Cut: {early_cutout}"
    )



def display_fixation_period(duration=3):
    """
    Displays a blank screen with fixation cross for a given duration.
    
    Parameters:
    - duration (int): Time in seconds for which the fixation period lasts.
    """
    start_time = time.time()
    clock = pygame.time.Clock()

    while time.time() - start_time < duration:
        # Fill screen with background color
        pygame.display.get_surface().fill(config.black)

        # Draw the fixation cross (assuming you have a function for it)
        draw_fixation_cross(screen_width, screen_height)  # Existing function in your code

        # Draw blank shapes (assuming placeholders)
        draw_ball_fill(0, screen_width, screen_height)  # Empty fill
        draw_arrow_fill(0, screen_width, screen_height)  # Empty fill
        draw_time_balls(0,screen_width,screen_height)
        pygame.display.flip()  # Update display

        # Check for quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        clock.tick(30)  # Maintain 30 FPS

# Interpolation function to compute fill amount between SHAPE_MIN and SHAPE_MAX
def interpolate_fill(value):
    return max(0, min(1, (value - config.SHAPE_MIN) / (config.SHAPE_MAX - config.SHAPE_MIN)))

def calculate_fill_levels(running_avg_confidence, mode):
    """
    Determines the fill levels for both MI (arrow) and Rest (ball) based on accumulated probability.

    Parameters:
        running_avg_confidence (float): The leaky-integrated probability estimate.
        mode (int): 0 for MI trial (fill square), 1 for Rest trial (fill ball).

    Returns:
        tuple: (fill_arrow, fill_ball) - Values between 0 and 1 indicating fill levels.
    """
    # Ensure probability stays within configured bounds
    prob = max(0, min(1, running_avg_confidence))
    prob_inverse = 1 - prob  # Inverse probability for the other shape


    # Determine fill levels
    fill_mi = interpolate_fill(prob) if prob >= config.SHAPE_MIN else 0  # MI shape fills when prob > SHAPE_MIN
    fill_rest = interpolate_fill(prob_inverse) if prob_inverse >= config.SHAPE_MIN else 0  # Rest shape fills when 1-prob > SHAPE_MIN

    # Swap roles if in Rest mode
    if mode == 1:
        return fill_rest, fill_mi  # Flip values for Rest condition
    return fill_mi, fill_rest  # Default for MI mode


def handle_fes_activation(mode, running_avg_confidence, fes_active):
    """
    Manages the activation of sensory FES based on the running average probability.

    Parameters:
        mode (int): 0 for MI (Motor Imagery), 1 for Rest.
        running_avg_confidence (float): Current probability estimate.
        fes_active (bool): Current state of FES (True if active, False if inactive).
        logger: LoggerManager instance used for structured logging.

    Returns:
        bool: Updated FES state after processing.
    """
    # Determine if FES should be active:
    # - If mode is MI (0) and confidence > 0.5 → Turn on FES
    # - If mode is Rest (1) and confidence < 0.5 → Turn on FES
    fes_should_be_active = (mode == 0 and running_avg_confidence > 0.5) or \
                           (mode == 1 and running_avg_confidence < 0.5)

    # Activate FES if needed
    if fes_should_be_active and not fes_active:
        if FES_toggle == 1:
            send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_SENS_GO", logger=logger)
            logger.log_event("Sensory FES activated.")
        else:
            logger.log_event("FES toggle is off — activation skipped.")
        return True

    # Deactivate FES if needed
    elif not fes_should_be_active and fes_active:
        if FES_toggle == 1:
            send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_STOP", logger=logger)
            logger.log_event("Sensory FES stopped.")
        else:
            logger.log_event("FES toggle is off — stop command skipped.")
        return False

    # No change in state
    return fes_active

def collect_baseline_during_countdown(inlet, countdown_start, countdown_duration, baseline_buffer):
    """
    Monitors countdown time and collects baseline EEG data during the last 0.5s.

    Parameters:
    - inlet: LSL EEG stream inlet.
    - countdown_start: Start time of countdown (pygame ticks).
    - countdown_duration: Total countdown duration in milliseconds.
    - baseline_buffer: List to store baseline EEG data.

    Returns:
    - Updated baseline_buffer containing collected baseline samples.
    """
    current_time = pygame.time.get_ticks()
    remaining_time = countdown_duration - (current_time - countdown_start)

    if remaining_time <= 1000 and not baseline_buffer:  # When 0.5s remain, flush and start collecting
        logger.log_event("Flushing buffer and collecting baseline data...")
        inlet.flush()  # Remove old EEG data

    if remaining_time <= 1000:  # Collect EEG data continuously
        new_data, _ = inlet.pull_chunk(timeout=0.1, max_samples=config.FS // 2)  # 0.5s worth of samples
        if new_data:
            baseline_buffer.extend(new_data)

    return baseline_buffer


def classify_real_time(inlet, window_size_samples, step_size_samples, all_probabilities, predictions, 
                        data_buffer, mode, leaky_integrator, baseline_mean=None, update_recentering=True):

    new_data, _ = inlet.pull_chunk(timeout=0.1, max_samples=int(step_size_samples))
    global counter
    global Prev_T
    
    if new_data:
        new_data_np = np.array(new_data)
        data_buffer.extend(new_data_np)

        if len(data_buffer) < config.FS:
            return leaky_integrator.accumulated_probability, predictions, all_probabilities, data_buffer

        sliding_window_np = np.array(data_buffer[-window_size_samples:]).T

        sfreq = config.FS
        info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(sliding_window_np, info)

        aux_channels = {"AUX1", "AUX2", "AUX3", "AUX7", "AUX8", "AUX9", "TRIGGER"}
        existing_aux = [ch for ch in aux_channels if ch in raw.ch_names]
        if existing_aux:
            raw.drop_channels(existing_aux)
            #logger.log_event(f"Dropped AUX channels: {existing_aux}")

        rename_dict = {
            "FP1": "Fp1", "FPZ": "Fpz", "FP2": "Fp2",
            "FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "POZ": "POz", "OZ": "Oz"
        }
        raw.rename_channels(rename_dict)

        mastoid_channels = ["M1", "M2"]
        existing_mastoids = [ch for ch in mastoid_channels if ch in raw.ch_names]
        if existing_mastoids:
            raw.drop_channels(existing_mastoids)
            #logger.log_event(f"Dropped mastoid channels: {existing_mastoids}")

        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=True, on_missing="warn")
        #logger.log_event("Montage set to standard_1020")

        for ch in raw.info['chs']:
            ch['unit'] = 201  # µV

        raw.notch_filter(60, method="iir")
        #logger.log_event("Applied 60Hz notch filter")

        raw.filter(l_freq=config.LOWCUT, h_freq=config.HIGHCUT, method="iir")
        #logger.log_event(f"Applied bandpass filter: {config.LOWCUT}–{config.HIGHCUT} Hz")

        if config.SURFACE_LAPLACIAN_TOGGLE:
            raw = mne.preprocessing.compute_current_source_density(raw)
            #logger.log_event("Applied surface Laplacian")

        if config.SELECT_MOTOR_CHANNELS:
            raw = select_motor_channels(raw)
            #logger.log_event("Selected motor channels")

        raw._data -= baseline_mean
        #logger.log_event("Baseline correction applied")

        eeg_data = raw.get_data()
        cov_matrix = (eeg_data @ eeg_data.T) / np.trace(eeg_data @ eeg_data.T)

        if config.LEDOITWOLF:
            cov_matrix_shrinked = np.array([LedoitWolf().fit(cov_matrix).covariance_])
            cov_matrix = cov_matrix_shrinked
            #logger.log_event("Applied Ledoit-Wolf shrinkage")
        else:
            cov_matrix = np.expand_dims(cov_matrix, axis=0)
            shrinkage = Shrinkage(shrinkage=config.SHRINKAGE_PARAM)
            cov_matrix = shrinkage.fit_transform(cov_matrix)
            #logger.log_event(f"Applied custom shrinkage: {config.SHRINKAGE_PARAM}")

        if config.RECENTERING:
            cov_matrix = np.squeeze(cov_matrix, axis=0)
            if counter == 0:
                Prev_T = cov_matrix
            T_test = geodesic_riemann(Prev_T, cov_matrix, 1/(counter+1))
            if update_recentering:
                Prev_T = T_test
                counter += 1
            T_test_invsqrtm = invsqrtm(T_test)
            cov_matrix = T_test_invsqrtm @ cov_matrix @ T_test_invsqrtm.T
            cov_matrix = np.expand_dims(cov_matrix, axis=0)
            #logger.log_event("Applied adaptive recentering")

        probabilities = model.predict_proba(cov_matrix)[0]
        predicted_label = model.classes_[np.argmax(probabilities)]

        predictions.append(predicted_label)
        all_probabilities.append([time.time(), probabilities[0], probabilities[1]])
        correct_label = 200 if mode == 0 else 100
        correct_class_idx = np.where(model.classes_ == correct_label)[0][0]
        current_confidence = probabilities[correct_class_idx]

        data_buffer = data_buffer[step_size_samples:]

        #logger.log_event(f"Predicted label: {predicted_label}, Confidence: {current_confidence:.3f}")
        return current_confidence, predictions, all_probabilities, data_buffer

    return leaky_integrator.accumulated_probability, predictions, all_probabilities, data_buffer





def hold_messages_and_classify(messages, colors, offsets, duration, inlet, mode, udp_socket, udp_ip, udp_port,
                               data_buffer, leaky_integrator, baseline_mean):
    """
    Holds visual messages on the screen while running real-time EEG classification in the background.
    If classification confidence drops below a threshold, sends an "s" message to stop the robot.
    
    Early stopping will only be enabled after a minimum number of classifications.

    Parameters:
    - messages (list): List of messages to display.
    - colors (list): Corresponding colors for each message.
    - offsets (list): Y-axis offsets for each message.
    - duration (int): Maximum duration to hold messages (seconds).
    - inlet: LSL EEG inlet for real-time data acquisition.
    - mode (int): 0 for Motor Imagery (200), 1 for Rest (100).
    - udp_socket: Socket for sending UDP messages.
    - udp_ip (str): IP address for UDP communication.
    - udp_port (int): Port for UDP communication.
    - data_buffer (list): Accumulated EEG data from `show_feedback`.
    - leaky_integrator (LeakyIntegrator): The existing leaky integrator instance.

    Returns:
    - int: Final classification result (200 or 100).
    """

    font = pygame.font.SysFont(None, 72)
    start_time = time.time()
    early_stop = False
    # EEG classification parameters
    step_size = 0.05  # Step size in seconds
    window_size = config.CLASSIFY_WINDOW / 1000  # Convert ms to seconds
    window_size_samples = int(window_size * config.FS)
    step_size_samples = int(step_size * config.FS)

    # Define correct and incorrect classes
    correct_class = 200 if mode == 0 else 100  # 200 = Motor Imagery, 100 = Rest
    incorrect_class = 100 if mode == 0 else 200  # Opposite class

    # Minimum number of predictions before early stopping is allowed
    min_predictions_before_stop = config.MIN_PREDICTIONS
    num_predictions = 0  # Counter for predictions made

    # accuracy threshold based on mode
    accuracy_threshold = config.THRESHOLD_MI if mode == 0 else config.THRESHOLD_REST 


    clock = pygame.time.Clock()

    while time.time() - start_time < duration:
        # Draw visual messages
        pygame.display.get_surface().fill((0, 0, 0))
        for i, text in enumerate(messages):
            message = font.render(text, True, colors[i])
            pygame.display.get_surface().blit(
                message,
                (pygame.display.get_surface().get_width() // 2 - message.get_width() // 2,
                 pygame.display.get_surface().get_height() // 2 + offsets[i])
            )
        pygame.display.flip()

        # Perform classification
        current_confidence, predictions, all_probabilities, data_buffer = classify_real_time(
            inlet, window_size_samples, step_size_samples, [], [], data_buffer, mode,leaky_integrator, baseline_mean, update_recentering = config.UPDATE_DURING_MOVE
        )

        # If a prediction was made, increase the counter
        if current_confidence > 0:
            num_predictions += 1

        # **Update leaky integrator confidence (uses same instance from `show_feedback`)**
        running_avg_confidence = leaky_integrator.update(current_confidence)
        # **Early stopping condition - stop the robot from moving, display message regarding early stop, and stop FES**
        if num_predictions >= min_predictions_before_stop and running_avg_confidence < config.RELAXATION_RATIO * accuracy_threshold:
            early_stop = True

            logger.log_event(f"Early stop triggered! Confidence: {running_avg_confidence:.2f} after {num_predictions} predictions")

            send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_EARLYSTOP"], logger=logger)
            send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_END"], logger=logger)

            if FES_toggle == 1:
                send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_STOP", logger=logger)
                logger.log_event("FES_STOP signal sent due to early stop.")
            else:
                logger.log_event("FES is disabled — no FES_STOP sent.")

            display_multiple_messages_with_udp(
                ["Stopping Robot"], [(255, 0, 0)], [0], duration=5,
                udp_messages=["s"], udp_socket=udp_socket, udp_ip=udp_ip, udp_port=udp_port, logger = logger
            )
            break


        # Check for quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        clock.tick(30)  # Maintain 30 FPS
    if early_stop == False:
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_END"], logger = logger)
    # Final Decision: Return correct or incorrect class based on confidence
    final_class = correct_class if running_avg_confidence >= config.RELAXATION_RATIO*config.ACCURACY_THRESHOLD else incorrect_class
    logger.log_event(f"Confidence at the end of motion: {running_avg_confidence:.2f} after {num_predictions} predictions")

    return final_class



def show_feedback(duration=5, mode=0, inlet=None, baseline_data=None):
    """
    Displays feedback animation, collects EEG data, and performs real-time classification
    using a sliding window approach with early stopping based on posterior probabilities.
    """
    start_time = time.time()
    step_size = 1 / 16  # Sliding window step size (seconds)
    window_size = config.CLASSIFY_WINDOW / 1000  # Convert ms to seconds
    window_size_samples = int(window_size * config.FS)
    step_size_samples = int(step_size * config.FS)
    global filter_states

    FES_active = False
    all_probabilities = []
    predictions = []
    data_buffer = []  # Buffer for EEG data
    leaky_integrator = LeakyIntegrator(alpha=config.INTEGRATOR_ALPHA)  # Confidence smoothing
    min_predictions = config.MIN_PREDICTIONS
    earlystop_flag = False

    classification_results = []
    # Define the correct class based on mode
    # Define the correct class based on mode
    correct_class = 200 if mode == 0 else 100  # 200 = Right Arm MI, 100 = Rest
    incorrect_class = 100 if mode == 0 else 200  # The opposite class

    # accuracy threshold based on mode
    accuracy_threshold = config.THRESHOLD_MI if mode == 0 else config.THRESHOLD_REST 
    opposed_threshold = config.THRESHOLD_REST if mode == 0 else config.THRESHOLD_MI
    # Preprocess the baseline dataset before feedback starts
    # Preprocess the baseline dataset before feedback starts
    if baseline_data is not None:
        logger.log_event("Processing baseline data...")

        # Convert to MNE RawArray
        sfreq = config.FS
        info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types="eeg")
        raw_baseline = mne.io.RawArray(baseline_data.T, info)

        # Drop AUX and Mastoid channels
        aux_channels = {"AUX1", "AUX2", "AUX3", "AUX7", "AUX8", "AUX9", "TRIGGER"}
        dropped_aux = [ch for ch in aux_channels if ch in raw_baseline.ch_names]
        raw_baseline.drop_channels(dropped_aux)
        logger.log_event(f"Dropped AUX channels: {dropped_aux}")

        mastoid_channels = ["M1", "M2"]
        dropped_mastoids = [ch for ch in mastoid_channels if ch in raw_baseline.ch_names]
        raw_baseline.drop_channels(dropped_mastoids)
        logger.log_event(f"Dropped mastoid channels: {dropped_mastoids}")

        # Ensure standard 10-20 montage
        montage = mne.channels.make_standard_montage("standard_1020")
        raw_baseline.rename_channels({
            "FP1": "Fp1", "FPZ": "Fpz", "FP2": "Fp2",
            "FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "POZ": "POz", "OZ": "Oz"
        })
        raw_baseline.set_montage(montage, match_case=True, on_missing="warn")
        logger.log_event("Standard 10–20 montage set on baseline data.")

        # Convert to µV
        for ch in raw_baseline.info['chs']:
            ch['unit'] = 201  # µV
        logger.log_event("Channel units set to µV.")

        # Apply Notch Filter
        raw_baseline.notch_filter(60, method="iir")
        logger.log_event("Applied 60Hz notch filter to baseline data.")

        # Apply Bandpass Filter
        raw_baseline.filter(l_freq=config.LOWCUT, h_freq=config.HIGHCUT, method="iir")
        logger.log_event(f"Applied bandpass filter: {config.LOWCUT}–{config.HIGHCUT} Hz.")

        if config.SURFACE_LAPLACIAN_TOGGLE:
            raw_baseline = mne.preprocessing.compute_current_source_density(raw_baseline)
            logger.log_event("Applied surface Laplacian to baseline data.")

        if config.SELECT_MOTOR_CHANNELS:
            raw_baseline = select_motor_channels(raw_baseline)
            logger.log_event("Selected motor channels from baseline data.")

        # Compute mean across time for baseline correction
        baseline_mean = np.mean(raw_baseline.get_data(), axis=1, keepdims=True)
        logger.log_event(f"Computed baseline mean: shape = {baseline_mean.shape}")

    else:
        baseline_mean = None
        logger.log_event("No baseline data provided — skipping baseline correction.")

        
    # Send UDP triggers
    if mode == 0:  # Red Arrow Mode (Motor Imagery)
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["MI_BEGIN"], logger=logger)
        if FES_toggle == 1:
            send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_SENS_GO", logger=logger)
            FES_active = True
        else:
            logger.log_event("FES is disabled.")
            FES_active = False
    else:  # Blue Ball Mode (Rest)
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["REST_BEGIN"], logger=logger)
        FES_active = False

    inlet.flush()

    clock = pygame.time.Clock()
    running_avg_confidence = 0.5  # Initial placeholder

    while time.time() - start_time < duration:
        current_confidence, predictions, all_probabilities, data_buffer = classify_real_time(
            inlet, window_size_samples, step_size_samples, all_probabilities, predictions,
            data_buffer, mode, leaky_integrator, baseline_mean
        )

        running_avg_confidence = leaky_integrator.update(current_confidence)
        if FES_toggle == 1:
            FES_active = handle_fes_activation(mode, running_avg_confidence, FES_active)

        screen.fill(config.black)
        MI_fill, Rest_fill = calculate_fill_levels(running_avg_confidence, mode)

        if mode == 0:
            draw_arrow_fill(MI_fill, screen_width, screen_height)
            draw_fixation_cross(screen_width, screen_height)
            draw_ball_fill(Rest_fill, screen_width, screen_height)
            draw_time_balls(2, screen_width, screen_height)
            message = pygame.font.SysFont(None, 96).render(f"Move {config.ARM_SIDE.upper()} Arm", True, config.white)
        else:
            draw_ball_fill(Rest_fill, screen_width, screen_height)
            draw_fixation_cross(screen_width, screen_height)
            draw_arrow_fill(MI_fill, screen_width, screen_height)
            draw_time_balls(3, screen_width, screen_height)
            message = pygame.font.SysFont(None, 96).render("Rest", True, config.white)

        screen.blit(message, (screen_width // 2 - message.get_width() // 2, screen_height // 2 + 300))
        pygame.display.flip()
        clock.tick(30)

        if len(predictions) >= min_predictions and running_avg_confidence >= accuracy_threshold:
            logger.log_event(f"Early stopping triggered! Confidence: {running_avg_confidence:.2f}")
            earlystop_flag = True
            if mode == 0:
                if FES_toggle == 1:
                    send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_STOP", logger=logger)
                else:
                    logger.log_event("FES is disabled.")
                send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["MI_EARLYSTOP"], logger=logger)
            else:
                send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["REST_EARLYSTOP"], logger=logger)
            break

    # Final Decision
    if running_avg_confidence >= accuracy_threshold:
        final_class = correct_class
    elif running_avg_confidence <= (1 - opposed_threshold):
        final_class = incorrect_class
    else:
        final_class = None  # Ambiguous zone
    
    if final_class is not None:
        logger.log_event(
            f"Final decision: {final_class}, Confidence for correct({correct_class}) class: "
            f"{running_avg_confidence:.2f}, at sample size {len(predictions)}"
        )
    else:
        logger.log_event(
            f"Ambiguous final decision — no threshold met. Confidence: {running_avg_confidence:.2f}, "
            f"MI threshold: {config.THRESHOLD_MI}, REST threshold: {config.THRESHOLD_REST}, "
            f"Samples: {len(predictions)}"
        )
    if FES_toggle == 1 and FES_active:
        send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_STOP", logger=logger)
    else:
        logger.log_event("FES disable not needed.")

    return final_class,running_avg_confidence, leaky_integrator, data_buffer, baseline_mean, all_probabilities, earlystop_flag



# Main Game Loop
logger.log_event("Resolving EEG data stream via LSL...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
logger.log_event("✅ EEG stream detected and inlet established.")

# Generate and log trial sequence
trial_sequence = generate_trial_sequence(total_trials=config.TOTAL_TRIALS, max_repeats=config.MAX_REPEATS)
logger.log_event(f"Trial Sequence generated: {trial_sequence}")
mode_labels = ["MI" if t == 0 else "REST" for t in trial_sequence]
logger.log_event(f"Trial Sequence (labeled): {mode_labels}")
current_trial = 0

# Fetch and log channel names from stream
channel_names = get_channel_names_from_lsl()
logger.log_event(f"Channel names detected in LSL stream: {channel_names}")

# Load 10–20 montage and rename channels for MNE compatibility
montage = mne.channels.make_standard_montage("standard_1020")
rename_dict = {
    "FP1": "Fp1", "FPZ": "Fpz", "FP2": "Fp2",
    "FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "POZ": "POz", "OZ": "Oz"
}

# Filter for EEG-only channels
non_eeg_channels = {"AUX1", "AUX2", "AUX3", "AUX7", "AUX8", "AUX9", "TRIGGER"}
valid_eeg_channels = [ch for ch in channel_names if ch not in non_eeg_channels]
valid_indices = [channel_names.index(ch) for ch in valid_eeg_channels]
logger.log_event(f"Filtered EEG channels (excluding AUX/Trigger): {valid_eeg_channels}")

# Initialize MNE Raw object for online data structure
sfreq = config.FS
info = mne.create_info(ch_names=valid_eeg_channels, sfreq=sfreq, ch_types="eeg")
raw = mne.io.RawArray(np.zeros((len(valid_eeg_channels), 1)), info)

# Apply montage and unit conversion
raw.rename_channels(rename_dict)
raw.set_montage(montage, match_case=True, on_missing="warn")
for ch in raw.info['chs']:
    ch['unit'] = 201  # Set unit to µV

logger.log_event(f"Applied 10–20 montage and prepared Raw object. Final channels: {raw.ch_names}")

# Initialize data buffer and experiment state
all_results = []
running = True
clock = pygame.time.Clock()

# Begin with fixation screen
display_fixation_period(duration=3)
logger.log_event("Initial fixation period complete. Beginning experimental loop.")

while running and current_trial < len(trial_sequence):
    logger.log_event(f"--- Trial {current_trial+1}/{len(trial_sequence)} START ---")

    # Initial fixation cross
    screen.fill(config.black)
    draw_fixation_cross(screen_width, screen_height)
    draw_arrow_fill(0, screen_width, screen_height)
    draw_ball_fill(0, screen_width, screen_height)
    draw_time_balls(0, screen_width, screen_height)
    pygame.display.flip()
    logger.log_event("Initial screen rendered: fixation cross, bar, ball, and time indicators.")

    # Wait for user input or automatic countdown
    backdoor_mode = None
    waiting_for_press = True
    countdown_start = None
    countdown_duration = 3000  # ms
    baseline_buffer = []

    while waiting_for_press:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                waiting_for_press = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    backdoor_mode = 0
                elif event.key == pygame.K_DOWN:
                    backdoor_mode = 1
                elif event.key == pygame.K_SPACE:
                    logger.log_event("Space bar pressed — proceeding without override.")
                waiting_for_press = False

        if config.TIMING:
            if countdown_start is None:
                countdown_start = pygame.time.get_ticks()
                logger.log_event("Countdown timer initiated.")

            elapsed_time = pygame.time.get_ticks() - countdown_start

            # Dynamically collect baseline EEG during countdown
            baseline_buffer = collect_baseline_during_countdown(
                inlet, countdown_start, countdown_duration, baseline_buffer
            )

            next_trial_mode = trial_sequence[current_trial]
            draw_time_balls(1, screen_width, screen_height)
            pygame.display.flip()

            if elapsed_time >= countdown_duration:
                logger.log_event("Countdown expired — preparing for feedback loop")
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
                waiting_for_press = False

    if not running:
        logger.log_event("Experiment terminated early via quit event.")
        break

    # Determine trial mode
    if backdoor_mode is not None:
        mode = backdoor_mode
        logger.log_event(f"Backdoor override activated: {'MI' if mode == 0 else 'REST'}")
    else:
        mode = trial_sequence[current_trial]
        logger.log_event(f"Trial mode selected from sequence: {'MI' if mode == 0 else 'REST'}")



    # Compute baseline mean after countdown ends
    baseline_data = np.array(baseline_buffer)
    logger.log_event(f"Collected baseline data: shape={baseline_data.shape}, total_samples={baseline_data.size}")
        
    # Show feedback and perform classification
    logger.log_event(f"Starting feedback classification — Mode: {'MI' if mode == 0 else 'REST'}")
    prediction,confidence, leaky_integrator, data_buffer, baseline_mean, trial_probs, earlystop_flag = show_feedback(
        duration=config.TIME_MI,
        mode=mode,
        inlet=inlet,
        baseline_data=baseline_data
    )

    # Log the classification result
    logger.log_event(f"Classification result — Predicted: {prediction}, Ground Truth: {200 if mode == 0 else 100}")

    # Send end-of-trial marker
    if mode == 0:
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["MI_END"], logger=logger)
    else:
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["REST_END"], logger=logger)

    # Log and store classification outcome
    append_trial_probabilities_to_csv(
        trial_probabilities=trial_probs,
        mode=mode,
        trial_number=current_trial + 1,
        predicted_label=prediction,
        early_cutout=earlystop_flag,
        mi_threshold=config.THRESHOLD_MI,
        rest_threshold=config.THRESHOLD_REST,
        logger=logger
    )
    logger.log_event(f"Stored decoder output for trial {current_trial+1}: {len(trial_probs)} timepoints.")

    predictions_list.append(prediction)
    ground_truth_list.append(200 if mode == 0 else 100)

    # Prepare messages and UDP logic based on the prediction
    if mode == 0:  # Red Arrow Mode (Right Arm Move)
        if prediction == 200:  # Correct prediction
            messages = ["Correct", "Robot Move"]
            colors = [config.green, config.green]
            offsets = [-100, 100]
            udp_messages = [config.ROBOT_TRAJECTORY, "g"]
            duration = 0.01  # Initial UDP command duration
            should_hold_and_classify = True

            logger.log_event("Prediction correct for MI — triggering robot movement (and FES if toggled)")

            if FES_toggle == 1:
                send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_MOTOR_GO", logger=logger)
            else:
                logger.log_event("FES disabled — skipping motor stimulation.")

            send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_BEGIN"], logger=logger)

        else:  # Incorrect prediction
            messages = ["Incorrect", "Robot Stationary"]
            colors = [config.red, config.white]
            offsets = [-100, 100]
            udp_messages = None
            duration = config.TIME_STATIONARY
            should_hold_and_classify = False

            logger.log_event("Prediction incorrect for MI — robot remains stationary.")

    else:  # Blue Ball Mode (Rest)
        if prediction == 100:  # Correct prediction
            messages = ["Correct", "Robot Stationary"]
            colors = [config.green, config.green]
            offsets = [-100, 100]
            udp_messages = None
            duration = config.TIME_STATIONARY

            logger.log_event("Prediction correct for REST — robot remains stationary.")

        else:  # Incorrect prediction
            messages = ["Incorrect", "Robot Stationary"]
            colors = [config.red, config.white]
            offsets = [-100, 100]
            udp_messages = None
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
        logger=logger  # Pass logger to internal UDP calls
    )

    # If trial was a correct MI, continue classification during robot movement
    if should_hold_and_classify:
        logger.log_event("Entering real-time classification window during robot movement...")
        hold_messages_and_classify(
            messages=messages, 
            colors=colors, 
            offsets=offsets, 
            duration=config.TIME_ROB - 6,  # Monitor for 7s out of total 13s movement
            inlet=inlet, 
            mode=0,  # Motor Imagery
            udp_socket=udp_socket_robot, 
            udp_ip=config.UDP_ROBOT["IP"], 
            udp_port=config.UDP_ROBOT["PORT"],
            data_buffer=data_buffer,
            leaky_integrator=leaky_integrator,
            baseline_mean=baseline_mean
        )
        display_fixation_period(duration=6)
        logger.log_event("Robot reset fixation (6s) complete after hold-and-classify phase.")

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
    display_fixation_period(duration=3)
    logger.log_event(f"Trial {current_trial+1} complete. Proceeding to next.")

    # Advance trial index and frame rate
    current_trial += 1
    pygame.display.flip()
    clock.tick(30)


logger.log_event(f"run complete")

pygame.quit()
