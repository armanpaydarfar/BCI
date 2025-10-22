import pygame
import numpy as np
import random
import pickle
from collections import deque
import os

class RollingScaler:
    def __init__(self, window_size=100, save_path=None):
        """
        A rolling Z-score normalizer that updates mean and std dynamically.
        
        - `window_size`: The max number of windows to store.
        - `save_path`: Optional path to save/load rolling stats between sessions.
        """
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.save_path = save_path  # File to save/load rolling mean & std

    def update(self, new_data):
        """Store new EEG data in the rolling window buffer."""
        self.buffer.append(new_data)

    def transform(self, new_data):
        """Apply z-score normalization using a single mean/std across all stored EEG values."""
        if len(self.buffer) == 0:
            mean = np.mean(new_data)  # Compute single mean over all values
            std = np.std(new_data) + 1e-8  # Compute single std over all values
        else:
            past_data = np.concatenate(self.buffer, axis=1).flatten()  # Flatten to (total_values,)
            mean = np.mean(past_data)  # Compute single mean across all data
            std = np.std(past_data)  # Compute single std across all data

        return (new_data - mean) / std


    def save(self):
        """Save the latest rolling buffer for continuity across runs."""
        if self.save_path:
            with open(self.save_path, 'wb') as f:
                pickle.dump(list(self.buffer), f)  # Convert deque to list before saving

    def load(self):
        """Load previously saved rolling buffer if available."""
        if self.save_path and os.path.exists(self.save_path):
            with open(self.save_path, 'rb') as f:
                self.buffer = deque(pickle.load(f), maxlen=self.window_size)
                print(f"✅ Loaded rolling stats from: {self.save_path}")


class LeakyIntegrator:
    def __init__(self, alpha=0.95):
        """
        Initializes the leaky integrator.
        
        :param alpha: Leak factor (0 < alpha < 1), where higher values retain past values longer.
        """
        self.alpha = alpha
        self.accumulated_probability = 0.5  # Initial probability

    def update(self, new_probability):
        """
        Updates the accumulated probability using a leaky integration method.

        :param new_probability: The new probability input from the classifier.
        :return: The updated accumulated probability.
        """
        self.accumulated_probability = self.alpha * self.accumulated_probability + (1 - self.alpha) * new_probability
        return self.accumulated_probability



def generate_trial_sequence(total_trials=30, max_repeats=3):
    """
    Generate a balanced binary sequence with no more than `max_repeats` consecutive values.
    Falls back to simple shuffle if constraints cannot be satisfied.

    Parameters:
        total_trials (int): Total number of trials (must be even).
        max_repeats (int): Maximum allowed repetitions of the same class.

    Returns:
        list: A valid binary sequence (0s and 1s).
    """
    assert total_trials % 2 == 0, "Total number of trials must be even."

    target_count = total_trials // 2
    counts = {0: 0, 1: 0}
    stack = [([], counts.copy())]

    while stack:
        seq, current_counts = stack.pop()

        if len(seq) == total_trials:
            return seq

        options = [0, 1]
        random.shuffle(options)

        for val in options:
            if len(seq) >= max_repeats and all(x == val for x in seq[-max_repeats:]):
                continue
            if current_counts[val] >= target_count:
                continue

            new_seq = seq + [val]
            new_counts = current_counts.copy()
            new_counts[val] += 1
            stack.append((new_seq, new_counts))

    # Fallback strategy: return a random balanced sequence
    fallback = [0] * target_count + [1] * target_count
    random.shuffle(fallback)
    print("⚠️ [generate_trial_sequence] Warning: Falling back to shuffled sequence due to constraint failure.")
    return fallback

'''
def display_multiple_messages_with_udp(
    messages, colors, offsets, duration=13,
    udp_messages=None, udp_socket=None, udp_ip=None, udp_port=None,
    logger=None, eeg_state=None):
    """
    Displays multiple messages on screen and optionally sends a UDP message once.

    Parameters:
    - messages (list of str): List of messages to display.
    - colors (list of tuples): Corresponding RGB color tuples.
    - offsets (list of int): Vertical offsets for each message.
    - duration (int): Duration in seconds to display messages.
    - udp_messages (list of str): UDP messages to send once.
    - udp_socket: UDP socket object for communication.
    - udp_ip (str): IP address to send UDP messages to.
    - udp_port (int): Port number to send UDP messages to.
    - logger: Logger object for event tracking.
    - eeg_state: Optional EEGState object to update during this display loop.
    """
    font = pygame.font.SysFont(None, 96)
    end_time = pygame.time.get_ticks() + duration * 1000
    udp_sent = False

    while pygame.time.get_ticks() < end_time:
        pygame.display.get_surface().fill((0, 0, 0))
        for i, text in enumerate(messages):
            message = font.render(text, True, colors[i])
            pygame.display.get_surface().blit(
                message,
                (pygame.display.get_surface().get_width() // 2 - message.get_width() // 2,
                 pygame.display.get_surface().get_height() // 2 - message.get_height() // 2 + offsets[i])
            )
        pygame.display.flip()

        # Update EEG buffer if provided
        if eeg_state is not None:
            eeg_state.update()

        # Send UDP messages once
        if udp_messages and not udp_sent:
            for msg in udp_messages:
                udp_socket.sendto(msg.encode('utf-8'), (udp_ip, udp_port))
                msg_str = f"Sent UDP message to {udp_ip}:{udp_port}: {msg}"
                if logger is not None:
                    logger.log_event(msg_str)
                else:
                    print(msg_str)
            udp_sent = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        pygame.time.Clock().tick(60)
'''
def save_transform(T, counter, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump({"T": T, "counter": counter}, f)
    print(f"✅ Saved adaptive transform and counter to: {save_path}")

def load_transform(load_path):
    if os.path.exists(load_path):
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "T" in data and "counter" in data:
            print(f"✅ Loaded adaptive transform and counter from: {load_path}")
            return data["T"], data["counter"]
        else:
            print(f"✅ Loaded legacy transform (no counter) from: {load_path}")
            return data, 1  # assume already had 1 update
    else:
        print(f"ℹ️ No saved adaptive transform found at: {load_path}")
        return None, 0
