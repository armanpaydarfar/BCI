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
from Utils.preprocessing import butter_bandpass_filter, apply_car_filter, apply_notch_filter, flatten_single_segment, extract_and_flatten_segment, parse_eeg_and_eog, remove_eog_artifacts
from Utils.visualization import draw_arrow_fill, draw_ball_fill, draw_fixation_cross, draw_time_balls
from Utils.experiment_utils import generate_trial_sequence, display_multiple_messages_with_udp, LeakyIntegrator
from Utils.networking import send_udp_message
from Utils.stream_utils import get_channel_names_from_lsl
import config
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






def classify_real_time(inlet, window_size_samples, step_size_samples, all_probabilities, predictions, data_buffer, mode):
    """
    Reads EEG data, applies preprocessing, extracts features, and classifies using a trained model.
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

        # Debugging: Print buffer size
        #print(f"Data buffer size: {len(data_buffer)}, Required: {window_size_samples}")

        # Ensure there are enough samples before proceeding
        if len(data_buffer) < window_size_samples:
            #print(f"Warning: Not enough samples for classification ({len(data_buffer)} samples). Skipping.")
            return 0.0, predictions, all_probabilities, data_buffer  # Return 0.0 for no confidence update

        # Keep only the latest `window_size_samples` for classification
        sliding_window_np = np.array(data_buffer[-window_size_samples:])

        # Process EEG and EOG data
        sliding_data, sliding_eog = parse_eeg_and_eog({'time_series': sliding_window_np}, channel_names)

        # Preprocessing
        window_data = remove_eog_artifacts(sliding_data, sliding_eog) if config.EOG_TOGGLE == 1 else sliding_data
        window_data = apply_notch_filter(window_data, config.FS)
        window_data = butter_bandpass_filter(window_data, config.LOWCUT, config.HIGHCUT, fs=config.FS)
        window_data = apply_car_filter(window_data)

        # Extract features and classify
        features = extract_and_flatten_segment(window_data, start_time=0, fs=config.FS, window_size_ms=config.CLASSIFY_WINDOW, offset_ms=config.CLASSIFICATION_OFFSET)

        probabilities = model.predict_proba(features)[0]
        predicted_label = model.classes_[np.argmax(probabilities)]

        predictions.append(predicted_label)
        all_probabilities.append(probabilities)

        # **Dynamically determine the correct class based on mode**
        correct_label = 200 if mode == 0 else 100  # 200 = Right Arm MI, 100 = Rest
        correct_class_idx = np.where(model.classes_ == correct_label)[0][0]
        current_confidence = probabilities[correct_class_idx]  # Single window confidence

        # Slide the buffer forward by step_size_samples
        data_buffer = data_buffer[step_size_samples:]

        #print(f"Predicted label: {predicted_label}, Current Confidence for ({correct_label}): {current_confidence:.3f}")

        return current_confidence, predictions, all_probabilities, data_buffer

    return 0.0, predictions, all_probabilities, data_buffer  # Default return when no data

def hold_messages_and_classify(messages, colors, offsets, duration, inlet, mode, udp_socket, udp_ip, udp_port,
                               data_buffer, leaky_integrator):
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
            inlet, window_size_samples, step_size_samples, [], [], data_buffer, mode
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



def show_feedback(duration=5, mode=0, inlet=None):
    """
    Displays feedback animation, collects EEG data, and performs real-time classification
    using a sliding window approach with early stopping based on posterior probabilities.
    """
    start_time = time.time()
    step_size = 0.05  # Sliding window step size (seconds)
    window_size = config.CLASSIFY_WINDOW / 1000  # Convert ms to seconds
    window_size_samples = int(window_size * config.FS)
    step_size_samples = int(step_size * config.FS)

    all_probabilities = []
    predictions = []
    data_buffer = []  # Buffer for EEG data
    leaky_integrator = LeakyIntegrator(alpha=0.9)  # Confidence smoothing
    min_predictions = config.MIN_PREDICTIONS
    # Define the correct class based on mode
    correct_class = 200 if mode == 0 else 100  # 200 = Right Arm MI, 100 = Rest
    incorrect_class = 100 if mode == 0 else 200  # The opposite class
    # Send UDP triggers
    if mode == 0:  # Red Arrow Mode (Motor Imagery)
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["MI_BEGIN"])
        send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_SENS_GO") if FES_toggle == 1 else print("FES is disabled.")
    else:  # Blue Ball Mode (Rest)
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["REST_BEGIN"])

    inlet.flush()

    clock = pygame.time.Clock()
    running_avg_confidence = 0.0  # Initial placeholder

    while time.time() - start_time < duration:
        # Perform classification
        current_confidence, predictions, all_probabilities, data_buffer = classify_real_time(
            inlet, window_size_samples, step_size_samples, all_probabilities, predictions, data_buffer, mode
        )

        # Update leaky integrator confidence
        running_avg_confidence = leaky_integrator.update(current_confidence)

        # Early stopping
        if len(predictions) >= min_predictions and running_avg_confidence >= config.ACCURACY_THRESHOLD:
            print(f"Early stopping triggered! Confidence: {running_avg_confidence:.2f}")
            if mode ==0:
                send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_STOP") if FES_toggle == 1 else print("FES is disabled.")
                send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["MI_EARLYSTOP"])
            else:
                send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["REST_EARLYSTOP"])

            break

        # Draw animation
        screen.fill(config.black)
        draw_time_balls(2000, next_trial_mode, screen_width, screen_height, ball_radius=30)
        if mode == 0:
            draw_arrow_fill(running_avg_confidence, screen_width, screen_height)
            draw_fixation_cross(screen_width, screen_height)
            draw_ball_fill(0, screen_width, screen_height)
            message = pygame.font.SysFont(None, 48).render("Imagine Right Arm Movement", True, config.white)
        else:
            draw_ball_fill(running_avg_confidence, screen_width, screen_height)
            draw_fixation_cross(screen_width, screen_height)
            draw_arrow_fill(0, screen_width, screen_height)
            message = pygame.font.SysFont(None, 72).render("Rest", True, config.white)

        screen.blit(message, (screen_width // 2 - message.get_width() // 2, screen_height // 2 + 150))
        pygame.display.flip()

        # Check for quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        clock.tick(60)  # Limit frame rate

    # Final Decision: Return correct or incorrect class based on confidence
    final_class = correct_class if running_avg_confidence >= config.ACCURACY_THRESHOLD else incorrect_class

    print(f"Final decision: {final_class} (Confidence for correct class: {running_avg_confidence:.2f})")

    return final_class, leaky_integrator, data_buffer


# Main Game Loop
# Attempt to resolve the stream
print("Looking for EEG data stream...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
print("EEG data stream detected. Starting experiment...")
trial_sequence = generate_trial_sequence(total_trials=config.TOTAL_TRIALS, max_repeats=config.MAX_REPEATS)
current_trial = 0  # Track the current trial index

channel_names = get_channel_names_from_lsl()
print(f"channel names in stream {channel_names}")

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

    while waiting_for_press:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                waiting_for_press = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:  # Right arrow key for red arrow
                    backdoor_mode = 0
                elif event.key == pygame.K_DOWN:  # Down arrow key for rest
                    backdoor_mode = 1
                elif event.key == pygame.K_SPACE:  # Space bar can also act as a trigger
                    print("Space bar pressed, proceeding...")
                waiting_for_press = False

        # Timing-based execution logic
        if config.TIMING:
            if countdown_start is None:
                countdown_start = pygame.time.get_ticks()  # Start countdown

            elapsed_time = pygame.time.get_ticks() - countdown_start

            # Draw timing balls during countdown
            next_trial_mode = trial_sequence[current_trial]  # Get the mode for the next trial
            draw_time_balls(elapsed_time, next_trial_mode, screen_width, screen_height, ball_radius=30)
            
            pygame.display.flip()  # Update the display with time balls

            if elapsed_time >= countdown_duration:
                print("Countdown complete, proceeding automatically.")
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
                waiting_for_press = False

    if not running:
        break

    # Determine mode
    if backdoor_mode is not None:
        mode = backdoor_mode  # Override with backdoor mode
    else:
        mode = trial_sequence[current_trial]  # Use pseudo-randomized sequence

    # Show feedback and classification
    prediction, leaky_integrator, data_buffer = show_feedback(duration=config.TIME_MI, mode=mode, inlet=inlet)
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
            leaky_integrator=leaky_integrator  # Pass the leaky integrator instance
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
