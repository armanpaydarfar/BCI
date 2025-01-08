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
from Utils.preprocessing import butter_bandpass_filter, apply_car_filter, apply_notch_filter, flatten_single_segment
from Utils.visualization import draw_arrow_fill, draw_ball_fill, draw_fixation_cross
from Utils.experiment_utils import generate_trial_sequence, display_multiple_messages_with_udp
from Utils.networking import send_udp_message
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



# Show feedback
def show_feedback(duration=5, mode=0, inlet=None):
    """
    Displays feedback animation, collects EEG data, and runs classification 
    at the end using a sliding window approach.
    """
    start_time = time.time()
    collected_data = []  # Store incoming EEG data
    step_size = 0.1  # Sliding window step size (seconds)
    window_size = config.CLASSIFY_WINDOW / 1000  # Sliding window size (ms -> seconds)
    # Send UDP triggers at the start and flush buffer
    if mode == 0:  # Red Arrow Mode
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], "200")
        send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_GO") if FES_toggle == 1 else print("FES is disabled. Skipping interaction.")
    else:  # Blue Ball Mode
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], "100")

    inlet.flush()  # Ensure fresh data collection

    # Fonts for messages
    small_font = pygame.font.SysFont(None, 48)
    large_font = pygame.font.SysFont(None, 72)

    # Animation and Data Collection Loop
    clock = pygame.time.Clock()
    while time.time() - start_time < duration:
        elapsed_time = time.time() - start_time
        progress = elapsed_time / duration

        # Collect EEG data continuously
        new_data, _ = inlet.pull_chunk(timeout=0.01, max_samples=int(0.05 * config.FS))
        if new_data:
            collected_data.extend(new_data)

        # Draw the animation and display appropriate text
        screen.fill(config.black)
        if mode == 0:  # Red Arrow Mode
            draw_arrow_fill(progress, screen_width, screen_height)
            draw_ball_fill(0, screen_width, screen_height)
            message = small_font.render("Imagine Right Arm Movement", True, config.white)
        else:  # Blue Ball Mode
            draw_ball_fill(progress, screen_width, screen_height)
            draw_arrow_fill(0, screen_width, screen_height)
            message = large_font.render("Rest", True, config.white)

        # Center the text
        screen.blit(
            message,
            (screen_width // 2 - message.get_width() // 2,
             screen_height // 2)
        )

        pygame.display.flip()

        # Check for quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        clock.tick(60)  # Limit frame rate to 60 FPS

    predictions = classify_eeg(collected_data,step_size, model)
    # Calculate performance
    correct_label = 200 if mode == 0 else 100
    accuracy = predictions.count(correct_label) / len(predictions) if predictions else 0.0

    print(f"Feedback Mode: {'Red Arrow (Right Arm Move)' if mode == 0 else 'Blue Ball (Rest)'}")
    print(f"Number of Classifications: {len(predictions)}")
    print(f"Classification Accuracy: {accuracy:.2f}")

    # Determine the final decision
    if accuracy >= config.ACCURACY_THRESHOLD:
        final_class = correct_label
    else:
        final_class = 100 if mode == 0 else 200

    # After calculating `final_class` in show_feedback:
    predictions_list.append(final_class)
    ground_truth_list.append(correct_label)  # Add the true label

    
    return final_class

# Classify EEG


def classify_eeg(collected_data, step_size, model):
    """
    Classify EEG data using a trained model with a sliding window approach.

    Parameters:
        collected_data (list): Collected EEG data in a list format (time x channels).
        step_size (float): Step size in seconds for the sliding window.
        model (object): Trained classification model.

    Returns:
        list: Predicted classes for each sliding window.
    """
    # Convert the collected data into a NumPy array
    eeg_data = np.array(collected_data)
    total_samples = eeg_data.shape[0]  # Total number of samples in the collected data
    # Ensure EEG data has the correct shape (time x channels)
    eeg_data = eeg_data[:, :config.CAP_TYPE]  # Retain only the number of expected channels

    # Apply pre-processing to the entire EEG dataset
    eeg_data = apply_notch_filter(eeg_data, config.FS)  # Remove line noise with notch filter
    eeg_data = butter_bandpass_filter(eeg_data, config.LOWCUT, config.HIGHCUT, config.FS)  # Apply bandpass filter
    eeg_data = apply_car_filter(eeg_data)  # Apply common average referencing (CAR) filter

    # Initialize a list to store predictions
    predictions = []

    # Calculate the number of samples in the classification window
    classify_window_samples = int(config.CLASSIFY_WINDOW / 1000 * config.FS)
    step_size_samples = int(step_size * config.FS)

    if total_samples >= classify_window_samples:
        # Perform sliding window classification
        for start_idx in range(0, total_samples - classify_window_samples + 1, step_size_samples):
            # Extract the current window of EEG data
            eeg_window = eeg_data[start_idx:start_idx + classify_window_samples]  # Shape: (window_samples, channels)

            # Flatten the window into a feature vector for classification
            features = flatten_single_segment(eeg_window)  # Shape: (1, channels * window_samples)

            # Predict the class using the trained model
            prediction = model.predict(features)[0]

            # Append the prediction to the list
            predictions.append(prediction)
    else:
        print("Warning: Not enough data collected for classification!")

    return predictions

# Main Game Loop
# Attempt to resolve the stream
print("Looking for EEG data stream...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
print("EEG data stream detected. Starting experiment...")
trial_sequence = generate_trial_sequence(total_trials=config.TOTAL_TRIALS, max_repeats=config.MAX_REPEATS)
current_trial = 0  # Track the current trial index

running = True
clock = pygame.time.Clock()

while running and current_trial < len(trial_sequence):
    # Initial fixation cross
    screen.fill(config.black)
    draw_fixation_cross(screen_width, screen_height)
    draw_arrow_fill(0,screen_width, screen_height)  # Show the empty arrow
    draw_ball_fill(0,screen_width, screen_height)   # Show the empty ball
    pygame.display.flip()

    # Wait for key press to determine backdoor
    backdoor_mode = None
    waiting_for_press = True
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
                waiting_for_press = False

    if not running:
        break

    # Determine mode
    if backdoor_mode is not None:
        mode = backdoor_mode  # Override with backdoor mode
    else:
        mode = trial_sequence[current_trial]  # Use pseudo-randomized sequence

    # Show feedback and classification
    prediction = show_feedback(duration=config.TIME_MI, mode=mode, inlet=inlet)

    # Prepare messages and UDP logic based on the prediction
    if mode == 0:  # Red Arrow Mode (Right Arm Move)
        if prediction == 200:  # Correct prediction
            messages = ["Correct", "Robot Move"]
            colors = [config.green, config.green]
            offsets = [-100, 100]
            udp_messages = ["x", "g"]
            duration = config.TIME_ROB
        else:  # Incorrect prediction
            messages = ["Incorrect", "Robot Stationary"]
            colors = [config.red, config.white]
            offsets = [-100, 100]
            udp_messages = None
            duration = config.TIME_STATIONARY
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
