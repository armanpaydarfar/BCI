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

# ✅ MNE for real-time EEG processing
import mne
mne.set_log_level("WARNING")  # Options: "ERROR", "WARNING", "INFO", "DEBUG"
# ✅ Preprocessing functions (updated for MNE integration)
from Utils.preprocessing import (
    butter_bandpass_filter,
    apply_car_filter,
    apply_notch_filter,
    flatten_single_segment,
    extract_and_flatten_segment,
    parse_eeg_and_eog,
    remove_eog_artifacts,
)

# ✅ Visualization utilities
from Utils.visualization import (
    draw_arrow_fill,
    draw_ball_fill,
    draw_fixation_cross,
    draw_time_balls,
)

# ✅ Experiment utilities
from Utils.experiment_utils import (
    generate_trial_sequence,
    display_multiple_messages_with_udp,
    LeakyIntegrator,
)

# ✅ Networking utilities
from Utils.networking import send_udp_message

# ✅ Stream utilities (LSL channel names)
from Utils.stream_utils import get_channel_names_from_lsl

# ✅ Configuration parameters
import config

# ✅ Performance evaluation (classification metrics)
from sklearn.metrics import confusion_matrix

# Initialize Pygame with dimensions from config
pygame.init()
screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
pygame.display.set_caption("BCI Online Interactive Loop")

# Screen dimensions
screen_width = config.SCREEN_WIDTH
screen_height = config.SCREEN_HEIGHT

# UDP Settings
udp_socket_marker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket_robot = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket_fes = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

FES_toggle = config.FES_toggle


# Load the LDA model
try:
    with open(config.MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model successfully loaded.")
except FileNotFoundError:
    print(f"Error: Model file '{config.MODEL_PATH}' not found.")
    exit(1)


# Ensure the Data directory exists
data_folder = os.path.join('/home/arman-admin/Projects/Harmony', 'Data')
os.makedirs(data_folder, exist_ok=True)

predictions_list = []
ground_truth_list = []





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
        draw_time_balls(0,None,screen_width,screen_height)
        pygame.display.flip()  # Update display

        # Check for quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        clock.tick(60)  # Maintain 60 FPS

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

    Returns:
        bool: Updated FES state after processing.
    """
    # Determine if FES should be active:
    # - If mode is MI (0) and confidence > 50% → Turn on FES
    # - If mode is Rest (1) and confidence < 50% → Turn on FES
    fes_should_be_active = (mode == 0 and running_avg_confidence > 0.5) or \
                           (mode == 1 and running_avg_confidence < 0.5)

    # If FES should be ON but is currently OFF → Activate
    if fes_should_be_active and not fes_active:
        send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_SENS_GO") if FES_toggle == 1 else print("FES is disabled.")
        print("Sensory FES activated.")
        return True  # FES is now active

    # If FES should be OFF but is currently ON → Deactivate
    elif not fes_should_be_active and fes_active:
        send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_STOP") if FES_toggle == 1 else print("FES is disabled.")
        print("Sensory FES stopped.")
        return False  # FES is now inactive

    # No change in state, return the existing value
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

    if remaining_time <= 500 and not baseline_buffer:  # ✅ When 0.5s remain, flush and start collecting
        print("⏳ Less than 0.5s left in countdown. Flushing buffer and collecting baseline data...")
        inlet.flush()  # ✅ Remove old EEG data

    if remaining_time <= 500:  # ✅ Collect EEG data continuously
        new_data, _ = inlet.pull_chunk(timeout=0.1, max_samples=config.FS // 2)  # 0.5s worth of samples
        if new_data:
            baseline_buffer.extend(new_data)

    return baseline_buffer


def classify_real_time(inlet, window_size_samples, step_size_samples, all_probabilities, predictions, 
                        data_buffer, mode, leaky_integrator, baseline_mean=None):
    """
    Reads EEG data, applies preprocessing using MNE, extracts features, and classifies using a trained model.
    Maintains a sliding window approach with step_size shifting.

    Returns:
    - current probability for the correct class (single window output)
    - updated predictions list
    - updated all_probabilities list
    - updated data_buffer (preserves past EEG samples)
    """
    new_data, _ = inlet.pull_chunk(timeout=0.1, max_samples=int(step_size_samples))

    if new_data:
        new_data_np = np.array(new_data)
        data_buffer.extend(new_data_np)  # Append new data to buffer

        # ✅ Ensure sufficient data before classification
        if len(data_buffer) < window_size_samples:
            return leaky_integrator.accumulated_probability, predictions, all_probabilities, data_buffer  

        # ✅ Keep only the latest `window_size_samples` for classification
        sliding_window_np = np.array(data_buffer[-window_size_samples:]).T  # Transpose to (channels, samples)

        # ✅ Convert to MNE RawArray
        sfreq = config.FS
        info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(sliding_window_np, info)

        # ✅ Drop AUX Channels (These are NOT EEG)
        aux_channels = {"AUX1", "AUX2", "AUX3", "AUX7", "AUX8", "AUX9", "TRIGGER"}
        existing_aux = [ch for ch in aux_channels if ch in raw.ch_names]
        if existing_aux:
            raw.drop_channels(existing_aux)

        # ✅ Standardize Channel Naming to Match 10-20 Montage
        rename_dict = {
            "FP1": "Fp1", "FPZ": "Fpz", "FP2": "Fp2",
            "FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "POZ": "POz", "OZ": "Oz"
        }
        raw.rename_channels(rename_dict)

        # ✅ Remove Mastoid Channels if Present
        mastoid_channels = ["M1", "M2"]
        existing_mastoids = [ch for ch in mastoid_channels if ch in raw.ch_names]
        if existing_mastoids:
            raw.drop_channels(existing_mastoids)

        # ✅ Ensure Data Matches Standard 10-20 Montage
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=True, on_missing="warn")

        # ✅ Convert Data to Microvolts (µV)
        #raw._data /= 1e6  # Convert Volts → µV
        for ch in raw.info['chs']:
            ch['unit'] = 201  # MNE Code for µV

        # ✅ Apply Notch and Bandpass Filtering (IIR to avoid FIR length issues)
        raw.notch_filter(60, method="iir")  
        raw.filter(
            l_freq=config.LOWCUT,
            h_freq=config.HIGHCUT,
            method="iir"  # ✅ Use IIR filter to avoid large FIR filter lengths
        )

        # ✅ Apply Surface Laplacian (CSD) with Error Handling

        #raw.set_eeg_reference('average')
        if config.SURFACE_LAPLACIAN_TOGGLE:
            raw = mne.preprocessing.compute_current_source_density(raw)

        #print(raw)
        # ✅ Apply Baseline Correction (Subtract Precomputed Baseline Mean)
        raw._data -= baseline_mean  # Apply baseline correction
        #print(raw)
        #✅ Apply Z-score Normalization using StandardScaler (Same as in Training)
        
        scaler = StandardScaler()
        raw._data = scaler.fit_transform(raw.get_data()) 
        #print(raw)
        # ✅ Compute the Covariance Matrix (For Riemannian Classifier)
        info = raw.info  # Get the raw's info object

        # ✅ Compute Covariance Matrix using MNE
        #cov = mne.compute_covariance(mne.EpochsArray(raw.get_data()[np.newaxis, :, :], info), method="oas")
        cov_matrix = np.cov(raw.get_data())
        #print(cov_matrix)
        cov_matrix = np.expand_dims(cov_matrix, axis=0)  # Reshape to (1, channels, channels)
        # ✅ Apply Shrinkage Regularization
        shrinkage = Shrinkage(shrinkage=0.1)  
        cov_matrix = shrinkage.fit_transform(cov_matrix)  # Regularized covariance matrix

        # ✅ Now pass the properly formatted covariance matrix to the classifier
        probabilities = model.predict_proba(cov_matrix)[0]

        predicted_label = model.classes_[np.argmax(probabilities)]

        predictions.append(predicted_label)
        all_probabilities.append(probabilities)

        # ✅ Dynamically determine the correct class based on mode
        correct_label = 200 if mode == 0 else 100  # 200 = Right Arm MI, 100 = Rest
        correct_class_idx = np.where(model.classes_ == correct_label)[0][0]
        current_confidence = probabilities[correct_class_idx]  # Single window confidence

        # ✅ Slide the buffer forward by `step_size_samples`
        data_buffer = data_buffer[step_size_samples:]

        return current_confidence, predictions, all_probabilities, data_buffer

    return leaky_integrator.accumulated_probability, predictions, all_probabilities, data_buffer  # Default return when no data






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
            inlet, window_size_samples, step_size_samples, [], [], data_buffer, mode,leaky_integrator, baseline_mean
        )

        # If a prediction was made, increase the counter
        if current_confidence > 0:
            num_predictions += 1

        # **Update leaky integrator confidence (uses same instance from `show_feedback`)**
        running_avg_confidence = leaky_integrator.update(current_confidence)

        # **Early stopping condition - stop the robot from moving, display message regarding early stop, and stop FES**
        if num_predictions >= min_predictions_before_stop and running_avg_confidence < config.RELAXATION_RATIO * config.ACCURACY_THRESHOLD:
            early_stop = True
            print(f"Stopping robot! Confidence: {running_avg_confidence:.2f} after {num_predictions} predictions")
            send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_EARLYSTOP"])
            send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_END"])
            send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_STOP") if FES_toggle == 1 else print("FES is disabled.")
            display_multiple_messages_with_udp(
                ["Stopping Robot"], [(255, 0, 0)], [0], duration=5,
                udp_messages=["s"], udp_socket=udp_socket, udp_ip=udp_ip, udp_port=udp_port
            )
            break

        # Check for quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        clock.tick(60)  # Limit frame rate
    if early_stop == False:
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_END"])
    # Final Decision: Return correct or incorrect class based on confidence
    final_class = correct_class if running_avg_confidence >= config.RELAXATION_RATIO*config.ACCURACY_THRESHOLD else incorrect_class
    print(f"Confidence at the end of motion: {running_avg_confidence:.2f} after {num_predictions} predictions")

    return final_class



def show_feedback(duration=5, mode=0, inlet=None, baseline_data=None):
    """
    Displays feedback animation, collects EEG data, and performs real-time classification
    using a sliding window approach with early stopping based on posterior probabilities.
    """
    start_time = time.time()
    step_size = 0.05  # Sliding window step size (seconds)
    window_size = config.CLASSIFY_WINDOW / 1000  # Convert ms to seconds
    window_size_samples = int(window_size * config.FS)
    step_size_samples = int(step_size * config.FS)

    FES_active = False
    all_probabilities = []
    predictions = []
    data_buffer = []  # Buffer for EEG data
    leaky_integrator = LeakyIntegrator(alpha=0.92)  # Confidence smoothing
    min_predictions = config.MIN_PREDICTIONS

    # Define the correct class based on mode
    correct_class = 200 if mode == 0 else 100  # 200 = Right Arm MI, 100 = Rest
    incorrect_class = 100 if mode == 0 else 200  # The opposite class

    # ✅ Preprocess the baseline dataset before feedback starts
    if baseline_data is not None:
        print("🔍 Processing Baseline Data...")

        # Convert to MNE RawArray
        sfreq = config.FS
        info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types="eeg")
        raw_baseline = mne.io.RawArray(baseline_data.T, info)

        # Drop AUX and Mastoid channels
        aux_channels = {"AUX1", "AUX2", "AUX3", "AUX7", "AUX8", "AUX9", "TRIGGER"}
        mastoid_channels = ["M1", "M2"]
        raw_baseline.drop_channels([ch for ch in aux_channels if ch in raw_baseline.ch_names])
        #raw_baseline.drop_channels([ch for ch in mastoid_channels if ch in raw_baseline.ch_names])
        
        if "M1" in raw.ch_names and "M2" in raw.ch_names:
            raw.drop_channels(["M1", "M2"])
            print("✅ Removed Mastoid Channels: M1, M2")
        else:
            print("ℹ️ No Mastoid Channels Found in Data")


        
        # Ensure standard 10-20 montage
        montage = mne.channels.make_standard_montage("standard_1020")
        raw_baseline.rename_channels({
            "FP1": "Fp1", "FPZ": "Fpz", "FP2": "Fp2",
            "FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "POZ": "POz", "OZ": "Oz"
        })
        raw_baseline.set_montage(montage, match_case=True, on_missing="warn")

        # Convert to µV
        #raw_baseline._data /= 1e6  
        for ch in raw_baseline.info['chs']:
            ch['unit'] = 201  # µV

        # Apply preprocessing: Notch filter, Bandpass, CAR, CSD
        raw.notch_filter(60, method="iir")  
        raw_baseline.filter(
            l_freq=config.LOWCUT,
            h_freq=config.HIGHCUT,
            method="iir"  # ✅ Use IIR filter to avoid large FIR filter lengths
        )
        #raw_baseline.set_eeg_reference('average')


        if config.SURFACE_LAPLACIAN_TOGGLE:
            raw_baseline = mne.preprocessing.compute_current_source_density(raw)


        
        # Compute baseline mean across time
        baseline_mean = np.mean(raw_baseline.get_data(), axis=1, keepdims=True)  # Shape: (n_channels, 1)
        print(f"✅ Computed Baseline Mean: Shape {baseline_mean.shape}")

    else:
        baseline_mean = None  # No baseline correction applied
    # Send UDP triggers
    if mode == 0:  # Red Arrow Mode (Motor Imagery)
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["MI_BEGIN"])
        send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_SENS_GO") if FES_toggle == 1 else print("FES is disabled.")
        FES_active = True if FES_toggle == 1 else print("FES tracking N/A")
    else:  # Blue Ball Mode (Rest)
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["REST_BEGIN"])

    inlet.flush()

    clock = pygame.time.Clock()
    running_avg_confidence = 0.5  # Initial placeholder

    while time.time() - start_time < duration:
        # Perform classification
        current_confidence, predictions, all_probabilities, data_buffer = classify_real_time(
            inlet, window_size_samples, step_size_samples, all_probabilities, predictions, data_buffer, mode, leaky_integrator, baseline_mean
        )

        # Update leaky integrator confidence
        running_avg_confidence = leaky_integrator.update(current_confidence)
        FES_toggle == 1 and (FES_active := handle_fes_activation(mode, running_avg_confidence, FES_active))

        # Draw animation
        screen.fill(config.black)
        draw_time_balls(2000, next_trial_mode, screen_width, screen_height, ball_radius=30)
        if mode == 0:
            MI_fill, Rest_fill = calculate_fill_levels(running_avg_confidence, mode)
            draw_arrow_fill(MI_fill, screen_width, screen_height)
            draw_fixation_cross(screen_width, screen_height)
            draw_ball_fill(Rest_fill, screen_width, screen_height)
            message = pygame.font.SysFont(None, 48).render("Imagine Right Arm Movement", True, config.white)
        else:
            MI_fill, Rest_fill = calculate_fill_levels(running_avg_confidence, mode)
            draw_ball_fill(Rest_fill, screen_width, screen_height)
            draw_fixation_cross(screen_width, screen_height)
            draw_arrow_fill(MI_fill, screen_width, screen_height)
            message = pygame.font.SysFont(None, 72).render("Rest", True, config.white)

        screen.blit(message, (screen_width // 2 - message.get_width() // 2, screen_height // 2 + 150))
        pygame.display.flip()

        # Early stopping
        if len(predictions) >= min_predictions and running_avg_confidence >= config.ACCURACY_THRESHOLD:
            print(f"Early stopping triggered! Confidence: {running_avg_confidence:.2f}")
            if mode == 0:
                send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_STOP") if FES_toggle == 1 else print("FES is disabled.")
                send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["MI_EARLYSTOP"])
            else:
                send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["REST_EARLYSTOP"])
            break

        clock.tick(60)  # Limit frame rate

    # Final Decision: Return correct or incorrect class based on confidence
    final_class = correct_class if running_avg_confidence >= config.ACCURACY_THRESHOLD else incorrect_class
    print(f"Final decision: {final_class}, Confidence for correct({correct_class}) class: {running_avg_confidence:.2f}) at sample size {len(predictions)}")

    send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_STOP") if FES_toggle == 1 and FES_active else print("FES disable not needed")

    return final_class, leaky_integrator, data_buffer, baseline_mean



# Main Game Loop
# Attempt to resolve the stream
print("Looking for EEG data stream...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
print("EEG data stream detected. Starting experiment...")

# Generate trial sequence
trial_sequence = generate_trial_sequence(total_trials=config.TOTAL_TRIALS, max_repeats=config.MAX_REPEATS)
current_trial = 0  # Track the current trial index

# Fetch channel names from LSL
channel_names = get_channel_names_from_lsl()
print(f"Channel names in stream: {channel_names}")

# ✅ Load standard 10-20 montage
montage = mne.channels.make_standard_montage("standard_1020")

# ✅ Case-sensitive renaming dictionary (for consistency with montage)
rename_dict = {
    "FP1": "Fp1", "FPZ": "Fpz", "FP2": "Fp2",
    "FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "POZ": "POz", "OZ": "Oz"
}

# ✅ Drop non-EEG channels
non_eeg_channels = {"AUX1", "AUX2", "AUX3", "AUX7", "AUX8", "AUX9", "TRIGGER"}
valid_eeg_channels = [ch for ch in channel_names if ch not in non_eeg_channels]

# ✅ Ensure valid EEG channels are indexed correctly
valid_indices = [channel_names.index(ch) for ch in valid_eeg_channels]

# ✅ Set up the MNE Raw object for real-time processing
sfreq = config.FS
info = mne.create_info(ch_names=valid_eeg_channels, sfreq=sfreq, ch_types="eeg")

# ✅ Create an empty buffer (will be updated with real-time data)
raw = mne.io.RawArray(np.zeros((len(valid_eeg_channels), 1)), info)

# ✅ Apply montage to match offline pipeline
raw.rename_channels(rename_dict)
raw.set_montage(montage, match_case=True, on_missing="warn")

# ✅ Convert data from Volts to microvolts (µV) immediately
raw._data /= 1e6  # Convert from Volts → µV
for ch in raw.info['chs']:
    ch['unit'] = 201  # 201 corresponds to µV in MNE’s standard units

print(f"✅ Applied standard 10-20 montage. Final EEG channels: {raw.ch_names}")
# ✅ Store this fixed montage for use in classification

# Start experiment loop
running = True
clock = pygame.time.Clock()
display_fixation_period(duration=3)
while running and current_trial < len(trial_sequence):
    # Initial fixation cross
    screen.fill(config.black)
    draw_fixation_cross(screen_width, screen_height)
    draw_arrow_fill(0, screen_width, screen_height)  # Show the empty bar
    draw_ball_fill(0, screen_width, screen_height)  # Show the empty ball
    draw_time_balls(0, None, screen_width, screen_height, ball_radius=30)
    pygame.display.flip()

    # Wait for key press or countdown to determine backdoor
    backdoor_mode = None
    waiting_for_press = True
    countdown_start = None  # Reset countdown timer
    countdown_duration = 3000  # 3 seconds in milliseconds
    baseline_buffer = []  # ✅ Store EEG samples for baseline calculation

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
                    print("Space bar pressed, proceeding...")
                waiting_for_press = False

        if config.TIMING:
            if countdown_start is None:
                countdown_start = pygame.time.get_ticks()  # ✅ Start countdown

            elapsed_time = pygame.time.get_ticks() - countdown_start

            # ✅ Call function to collect baseline when time is running out
            baseline_buffer = collect_baseline_during_countdown(inlet, countdown_start, countdown_duration, baseline_buffer)

            next_trial_mode = trial_sequence[current_trial]  
            draw_time_balls(elapsed_time, next_trial_mode, screen_width, screen_height, ball_radius=30)
            pygame.display.flip()  

            if elapsed_time >= countdown_duration:
                print("✅ Countdown complete, computing baseline mean...")
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
                waiting_for_press = False


    if not running:
        break

    # Determine mode
    if backdoor_mode is not None:
        mode = backdoor_mode  # Override with backdoor mode
    else:
        mode = trial_sequence[current_trial]  # Use pseudo-randomized sequence



    # ✅ Compute baseline mean after countdown ends
    baseline_data = np.array(baseline_buffer)

    # Show feedback and classification
    prediction, leaky_integrator, data_buffer, baseline_mean = show_feedback(duration=config.TIME_MI, mode=mode, inlet=inlet, baseline_data = baseline_data)
    send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["MI_END"]) if mode == 0 else send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["REST_END"])


    predictions_list.append(prediction)
    ground_truth_list.append(200) if mode ==0 else ground_truth_list.append(100)
    # Prepare messages and UDP logic based on the prediction
    if mode == 0:  # Red Arrow Mode (Right Arm Move)
        if prediction == 200:  # Correct prediction
            messages = ["Correct", "Robot Move"]
            colors = [config.green, config.green]
            offsets = [-100, 100]
            udp_messages = ["x", "g"]
            duration = 0.01  # Short duration for initial command
            should_hold_and_classify = True  # Set flag for classification
            send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_MOTOR_GO") if FES_toggle == 1 else print("FES is disabled.")
            send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_BEGIN"])
        else:  # Incorrect prediction
            messages = ["Incorrect", "Robot Stationary"]
            colors = [config.red, config.white]
            offsets = [-100, 100]
            udp_messages = None
            duration = config.TIME_STATIONARY
            should_hold_and_classify = False  # No classification for incorrect prediction

    else:  # Blue Ball Mode (Rest)
        if prediction == 100:  # Correct prediction
            messages = ["Correct", "Robot Stationary"]
            colors = [config.green, config.green]
            offsets = [-100, 100]
            udp_messages = None
            duration = config.TIME_STATIONARY
        else:  # Incorrect prediction
            messages = ["Incorrect", "Robot Stationary"]
            colors = [config.red, config.white]
            offsets = [-100, 100]
            udp_messages = None
            duration = config.TIME_STATIONARY

        should_hold_and_classify = False  # No classification in Rest mode
    # Display the feedback messages and send UDP messages
    display_multiple_messages_with_udp(
        messages=messages,
        colors=colors,
        offsets=offsets,
        duration=duration,
        udp_messages=udp_messages,
        udp_socket=udp_socket_robot,
        udp_ip=config.UDP_ROBOT["IP"],
        udp_port=config.UDP_ROBOT["PORT"]
    )

    # Hold messages and classify in the background
    # **Invoke `hold_messages_and_classify` only if "Robot Move" and correct prediction**
    
    if should_hold_and_classify:
        hold_messages_and_classify(
            messages=messages, 
            colors=colors, 
            offsets=offsets, 
            duration=config.TIME_ROB,  # Monitor for 13 seconds
            inlet=inlet, 
            mode=0,  # Motor Imagery classification
            udp_socket=udp_socket_robot, 
            udp_ip=config.UDP_ROBOT["IP"], 
            udp_port=config.UDP_ROBOT["PORT"],
            data_buffer=data_buffer,  # Pass accumulated EEG data
            leaky_integrator=leaky_integrator,  # Pass the leaky integrator instance
            baseline_mean = baseline_mean
        )
                         
    display_fixation_period(duration=3)
    current_trial += 1  # Move to the next trial
    clock.tick(60)  # Keep the frame rate consistent



# Calculate confusion matrix
cm = confusion_matrix(ground_truth_list, predictions_list, labels=[200, 100])

# Add labels for clarity
labels = ['Correct Right Arm Move (200)', 'Correct Rest (100)']
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# After collecting predictions and ground truths:
if predictions_list and ground_truth_list:  # Ensure the lists are not empty
    cm = confusion_matrix(ground_truth_list, predictions_list, labels=[200, 100])

    # Convert confusion matrix to a DataFrame with labeled rows and columns
    cm_df = pd.DataFrame(
        cm,
        index=['Actual 200 (Correct Move)', 'Actual 100 (Correct Rest)'],
        columns=['Predicted 200 (Move)', 'Predicted 100 (Rest)']
    )

    # Save the confusion matrix to a CSV file
    cm_file_path = os.path.join(data_folder, f'confusion_matrix_{timestamp}.csv')
    cm_df.to_csv(cm_file_path)
    print(f"Confusion matrix saved to {cm_file_path}")
else:
    print("No predictions or ground truths available to calculate confusion matrix.")

pygame.quit()
