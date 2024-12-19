import pygame
from pylsl import StreamInlet, resolve_stream
import socket
import time
import pickle
import numpy as np
from scipy.signal import butter, filtfilt
import random






# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((1800, 1200))  # Enlarge the window
pygame.display.set_caption("BCI Online Interactive Loop")

# Colors
black = (0, 0, 0)
white = (255, 255, 255)
blue = (0, 0, 255)
red = (255, 0, 0)
green = (0, 255, 0)

# Screen dimensions
screen_width, screen_height = screen.get_size()

# Positions
fixation_x, fixation_y = screen_width // 2, screen_height // 2
arrow_x, arrow_y = (6 * screen_width) // 8, screen_height // 2
ball_x, ball_y = screen_width // 2, screen_height // 4

# UDP Settings
UDP_IP_MAIN = "127.0.0.1"
UDP_PORT_MAIN = 12345

UDP_IP_EXTRA = "192.168.2.1"
UDP_PORT_EXTRA = 8080

udp_socket_main = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket_extra = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Thresholds for classification
CLASSIFY_WINDOW = 1.0  # Duration of EEG data window for classification (in seconds)
ACCURACY_THRESHOLD = 0.51  # Accuracy threshold to determine "Correct" 


# Load the LDA model
model_path = "/home/arman-admin/Projects/Harmony/lda_eeg_model.pkl"
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model successfully loaded.")
except FileNotFoundError:
    print(f"Error: Model file '{model_path}' not found.")
    exit(1)

# Butterworth Bandpass Filter
def butter_bandpass_filter(data, lowcut=0.1, highcut=30, fs=512, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

# Common Average Reference (CAR)
def apply_car_filter(data):
    avg = np.mean(data, axis=1, keepdims=True)
    return data - avg

# Utility Functions


import random

def generate_trial_sequence(total_trials=30, max_repeats=3):
    """
    Generate a pseudo-randomized list of trials with constraints.
    
    Parameters:
        total_trials (int): Total number of trials (default: 30).
        max_repeats (int): Maximum consecutive repeats allowed.
    
    Returns:
        list: A pseudo-randomized list of modes (0 = red arrow, 1 = rest).
    """
    trials = [0] * (total_trials // 2) + [1] * (total_trials // 2)
    random.shuffle(trials)  # Start with a shuffled list
    
    # Enforce max_repeats constraint
    fixed_trials = []
    for trial in trials:
        if len(fixed_trials) >= max_repeats and all(t == trial for t in fixed_trials[-max_repeats:]):
            # Find an alternative mode that satisfies the constraints
            alternatives = [t for t in set([0, 1]) if t != trial]
            random.shuffle(alternatives)
            trial = alternatives[0]
        fixed_trials.append(trial)
    
    return fixed_trials




def send_udp_message(socket, ip, port, message):
    socket.sendto(message.encode('utf-8'), (ip, port))
    print(f"Sent UDP message to {ip}:{port}: {message}")

def display_multiple_messages_with_udp(messages, colors, offsets, duration=13, udp_messages=None, udp_socket=None, udp_ip=None, udp_port=None):
    """
    Display multiple messages on the screen for a given duration while allowing UDP messages to be sent.
    Keeps animations running smoothly.

    Parameters:
        messages (list): List of text strings to display.
        colors (list): List of RGB tuples for text colors corresponding to the messages.
        offsets (list): List of vertical offsets for each message. Negative = above center, Positive = below center.
        duration (int): Duration in seconds for which messages are displayed.
        udp_messages (list): List of messages (strings) to send over UDP.
        udp_socket (socket): The socket to send the UDP messages.
        udp_ip (str): IP address for the UDP messages.
        udp_port (int): Port for the UDP messages.
    """
    font = pygame.font.SysFont(None, 72)
    end_time = pygame.time.get_ticks() + duration * 1000  # Convert duration to milliseconds

    udp_sent = False  # Track if UDP messages have been sent

    while pygame.time.get_ticks() < end_time:
        screen.fill(black)  # Clear the screen
        draw_arrow_fill(0)  # Keep the arrow static
        draw_ball_fill(0)   # Keep the ball static

        # Display all messages with their respective offsets
        for i, text in enumerate(messages):
            message = font.render(text, True, colors[i])
            screen.blit(
                message,
                (screen_width // 2 - message.get_width() // 2, 
                 screen_height // 2 - message.get_height() // 2 + offsets[i])
            )
        
        pygame.display.flip()

        # Send all UDP messages at the start if not already sent
        if udp_messages and not udp_sent:
            for msg in udp_messages:
                udp_socket.sendto(msg.encode('utf-8'), (udp_ip, udp_port))
                print(f"Sent UDP message to {udp_ip}:{udp_port}: {msg}")
            udp_sent = True  # Prevent resending the messages

        # Allow quitting events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        clock.tick(60)  # Limit frame rate to 60 FPS



def show_feedback(duration=5, mode=0, inlet=None):
    """
    Displays feedback animation, collects EEG data, and runs classification 
    at the end using a sliding window approach.

    Parameters:
        duration (float): Total duration of the feedback phase.
        mode (int): 0 for Red Arrow (Right Arm Move), 1 for Blue Ball (Rest).
        inlet: LSL inlet for EEG data.

    Returns:
        int: 200 for "Correct Right Arm Move" or 100 for "Correct Rest" based on accuracy threshold.
    """
    start_time = time.time()
    collected_data = []  # Store incoming EEG data
    fs = 512  # Sampling frequency
    step_size = 0.1  # Sliding window step size (seconds)
    window_size = 1.0  # Sliding window size (seconds)

    # Send UDP triggers at the start and flush buffer
    if mode == 0:  # Red Arrow Mode
        send_udp_message(udp_socket_main, UDP_IP_MAIN, UDP_PORT_MAIN, "200")
        print("Sent triggers: '200'")
    else:  # Blue Ball Mode
        send_udp_message(udp_socket_main, UDP_IP_MAIN, UDP_PORT_MAIN, "100")
        print("Sent trigger: '100'")

    inlet.flush()  # Ensure fresh data collection

    # Fonts
    small_font = pygame.font.SysFont(None, 48)
    large_font = pygame.font.SysFont(None, 72)

    # Animation and Data Collection Loop
    clock = pygame.time.Clock()
    while time.time() - start_time < duration:
        elapsed_time = time.time() - start_time
        progress = elapsed_time / duration

        # Collect EEG data continuously
        new_data, _ = inlet.pull_chunk(timeout=0.01, max_samples=int(0.01 * fs))
        if new_data:
            collected_data.extend(new_data)

        # Draw the animation and display appropriate text
        screen.fill(black)
        if mode == 0:  # Red Arrow Mode
            draw_arrow_fill(progress)
            draw_ball_fill(0)
            message = small_font.render("Imagine Right Arm Movement", True, white)
        else:  # Blue Ball Mode
            draw_ball_fill(progress)
            draw_arrow_fill(0)
            message = large_font.render("Rest", True, white)

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

    # After the animation, run classification
    print("Animation complete. Running classification...")
    predictions = []  # Store predictions for sliding window
    total_samples = len(collected_data)

    if total_samples >= int(window_size * fs):
        # Use sliding window to classify
        for start_idx in range(0, total_samples - int(window_size * fs), int(step_size * fs)):
            eeg_window = np.array(collected_data[start_idx:start_idx + int(window_size * fs)])
            prediction = classify_eeg(eeg_window)  # Pass EEG window directly
            predictions.append(prediction)
    else:
        print("Warning: Not enough data collected for classification!")

    # Calculate performance
    correct_label = 200 if mode == 0 else 100
    accuracy = predictions.count(correct_label) / len(predictions) if predictions else 0.0

    print(f"Feedback Mode: {'Red Arrow (Right Arm Move)' if mode == 0 else 'Blue Ball (Rest)'}")
    print(f"Number of Classifications: {len(predictions)}")
    print(f"Classification Accuracy: {accuracy:.2f}")

    # Determine the final decision
    if accuracy >= ACCURACY_THRESHOLD:
        final_class = correct_label
    else:
        final_class = 100 if mode == 0 else 200

    return final_class


def classify_eeg(eeg_window):
    """
    Classify EEG data using the trained model.

    Parameters:
        eeg_window (np.ndarray): EEG data window for classification. Shape: (samples, channels).

    Returns:
        int: Predicted class (e.g., 200 for 'Right Arm Move', 100 for 'Rest').
    """
    # Apply preprocessing
    eeg_window = butter_bandpass_filter(eeg_window)  # Apply bandpass filter
    eeg_window = apply_car_filter(eeg_window)       # Apply CAR filter
    
    # Extract features
    features = eeg_window.flatten().reshape(1, -1)  # Flatten to match model dimensions
    
    # Predict class
    prediction = model.predict(features)[0]
    return prediction




def draw_fixation_cross():
    cross_length = 40
    line_thickness = 6
    pygame.draw.line(screen, white, (fixation_x, fixation_y - cross_length), (fixation_x, fixation_y + cross_length), line_thickness)
    pygame.draw.line(screen, white, (fixation_x - cross_length, fixation_y), (fixation_x + cross_length, fixation_y), line_thickness)

def draw_arrow_fill(progress):
    arrow_width, arrow_length, tip_length = 80, 200, 40
    arrow_outline = [
        (arrow_x - arrow_length // 2, arrow_y - arrow_width // 2),
        (arrow_x + arrow_length // 2 - tip_length, arrow_y - arrow_width // 2),
        (arrow_x + arrow_length // 2, arrow_y),
        (arrow_x + arrow_length // 2 - tip_length, arrow_y + arrow_width // 2),
        (arrow_x - arrow_length // 2, arrow_y + arrow_width // 2),
    ]
    pygame.draw.polygon(screen, white, arrow_outline, 2)
    filled_rect = pygame.Rect(arrow_x - arrow_length // 2, arrow_y - arrow_width // 2, int(progress * (arrow_length - tip_length)), arrow_width)
    pygame.draw.rect(screen, red, filled_rect)

def draw_ball_fill(progress):
    """
    Draws a ball that fills like a cup of water respecting circular boundaries.
    
    Parameters:
        progress (float): Progress value from 0.0 (empty) to 1.0 (full).
    """
    ball_radius = 100
    ball_center = (ball_x, ball_y)

    # Draw ball outline
    pygame.draw.circle(screen, white, ball_center, ball_radius, 2)

    # Create a surface for the water fill with transparency
    water_surface = pygame.Surface((ball_radius * 2, ball_radius * 2), pygame.SRCALPHA)

    # Calculate the fill height based on progress
    fill_height = int(progress * ball_radius * 2)
    
    # Draw the "water" rectangle on the water surface
    water_rect = pygame.Rect(0, ball_radius * 2 - fill_height, ball_radius * 2, fill_height)
    pygame.draw.rect(water_surface, (0, 0, 255, 180), water_rect)  # Blue color with transparency

    # Create a mask to clip the water fill to a circular shape
    mask_surface = pygame.Surface((ball_radius * 2, ball_radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(mask_surface, (255, 255, 255, 255), (ball_radius, ball_radius), ball_radius)

    # Use BLEND_RGBA_MULT to apply the mask and clip to circular boundaries
    water_surface.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    # Blit the final "water fill" onto the main screen
    screen.blit(water_surface, (ball_x - ball_radius, ball_y - ball_radius))

def check_streams():
    print("Checking for EEG stream...")
    eeg_streams = resolve_stream('type', 'EEG')
    if not eeg_streams:
        print("Error: EEG stream not found.")
        exit(1)
    print("EEG stream is active.")
    return StreamInlet(eeg_streams[0])

def get_eeg_data(inlet, duration=1.0, sampling_rate=512):
    inlet.flush()
    samples = int(duration * sampling_rate)
    data, _ = inlet.pull_chunk(timeout=duration + 0.5, max_samples=samples)
    return np.array(data[-samples:])


# Main Game Loop

# Generate pseudo-randomized trial sequence
inlet= check_streams()
trial_sequence = generate_trial_sequence(total_trials=30, max_repeats=3)
current_trial = 0  # Track the current trial index


running = True
clock = pygame.time.Clock()

while running and current_trial < len(trial_sequence):
    # Initial fixation cross
    screen.fill(black)
    draw_fixation_cross()
    draw_arrow_fill(0)  # Show the empty arrow
    draw_ball_fill(0)   # Show the empty ball
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
    prediction = show_feedback(duration=5, mode=mode, inlet=inlet)

    # Prepare messages and UDP logic based on the prediction
    if mode == 0:  # Red Arrow Mode (Right Arm Move)
        if prediction == 200:  # Correct prediction
            messages = ["Correct", "Robot Move"]
            colors = [green, green]
            offsets = [-100, 100]
            udp_messages = ["x", "g"]
            duration = 13  # Long duration for correct robot move
        else:  # Incorrect prediction
            messages = ["Incorrect", "Robot Stationary"]
            colors = [red, white]
            offsets = [-100, 100]
            udp_messages = None
            duration = 2  # Short duration for incorrect result
    else:  # Blue Ball Mode (Rest)
        if prediction == 100:  # Correct prediction
            messages = ["Correct", "Robot Stationary"]
            colors = [green, green]
            offsets = [-100, 100]
            udp_messages = None
            duration = 2  # Short duration for rest
        else:  # Incorrect prediction
            messages = ["Incorrect", "Robot Stationary"]
            colors = [red, white]
            offsets = [-100, 100]
            udp_messages = None
            duration = 2  # Short duration for incorrect result

    # Display the feedback messages and send UDP messages (if applicable)
    display_multiple_messages_with_udp(
        messages=messages,
        colors=colors,
        offsets=offsets,
        duration=duration,  # Dynamic duration based on prediction
        udp_messages=udp_messages,
        udp_socket=udp_socket_extra,
        udp_ip=UDP_IP_EXTRA,
        udp_port=UDP_PORT_EXTRA
    )

    current_trial += 1  # Move to the next trial
    clock.tick(60)  # Keep the frame rate consistent

pygame.quit()

