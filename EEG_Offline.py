import pygame
from pylsl import StreamInlet, resolve_stream
import socket
import time
import random

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((1800, 1200))  # Enlarge the window
pygame.display.set_caption("EEG Offline Interactive Loop")

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

# Trial Parameters
total_trials = 30  # 15 of each type
max_repeats = 3    # Maximum consecutive repeats

# Function Definitions

def send_udp_message(socket, ip, port, message):
    socket.sendto(message.encode('utf-8'), (ip, port))
    print(f"Sent UDP message to {ip}:{port}: {message}")

def generate_trial_sequence(total_trials=30, max_repeats=3):
    trials = [0] * (total_trials // 2) + [1] * (total_trials // 2)
    random.shuffle(trials)
    fixed_trials = []
    for trial in trials:
        if len(fixed_trials) >= max_repeats and all(t == trial for t in fixed_trials[-max_repeats:]):
            alternatives = [t for t in set([0, 1]) if t != trial]
            random.shuffle(alternatives)
            trial = alternatives[0]
        fixed_trials.append(trial)
    return fixed_trials

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
    ball_radius = 100
    ball_center = (ball_x, ball_y)
    pygame.draw.circle(screen, white, ball_center, ball_radius, 2)
    water_surface = pygame.Surface((ball_radius * 2, ball_radius * 2), pygame.SRCALPHA)
    fill_height = int(progress * ball_radius * 2)
    water_rect = pygame.Rect(0, ball_radius * 2 - fill_height, ball_radius * 2, fill_height)
    pygame.draw.rect(water_surface, (0, 0, 255, 180), water_rect)
    mask_surface = pygame.Surface((ball_radius * 2, ball_radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(mask_surface, (255, 255, 255, 255), (ball_radius, ball_radius), ball_radius)
    water_surface.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    screen.blit(water_surface, (ball_x - ball_radius, ball_y - ball_radius))

def display_multiple_messages_with_udp(messages, colors, offsets, duration=13, udp_messages=None, udp_socket=None, udp_ip=None, udp_port=None):
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

def show_feedback(duration=5, mode=0):
    """
    Displays feedback animation for the specified duration.

    Parameters:
        duration (float): Duration for which the animation is displayed.
        mode (int): 0 for 'Imagine Right Arm Movement', 1 for 'Rest'.
    """
    start_time = time.time()
    
    # Define fonts
    small_font = pygame.font.SysFont(None, 48)  # Font size for 'Imagine Right Arm Movement'
    large_font = pygame.font.SysFont(None, 72)  # Larger font size for 'Rest'

    while time.time() - start_time < duration:
        elapsed_time = time.time() - start_time
        progress = elapsed_time / duration

        # Clear screen
        screen.fill(black)

        if mode == 0:
            # Draw the arrow filling
            draw_arrow_fill(progress)
            draw_ball_fill(0)

            # Render and center message with smaller font
            message = small_font.render("Imagine Right Arm Movement", True, white)
        else:
            # Draw the ball filling
            draw_ball_fill(progress)
            draw_arrow_fill(0)

            # Render and center message with larger font
            message = large_font.render("Rest", True, white)

        # Center the message properly on the screen
        screen.blit(
            message,
            (screen_width // 2 - message.get_width() // 2,
             screen_height // 2 - message.get_height() // 2)
        )

        pygame.display.flip()

        # Event handling to allow quitting
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

    return True


# Main Game Loop
trial_sequence = generate_trial_sequence(total_trials, max_repeats)
current_trial = 0
running = True
clock = pygame.time.Clock()

while running and current_trial < len(trial_sequence):
    screen.fill(black)
    draw_fixation_cross()
    draw_arrow_fill(0)
    draw_ball_fill(0)
    pygame.display.flip()

    # Backdoor logic
    backdoor_mode = None
    waiting_for_press = True
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
                waiting_for_press = False
    if not running:
        break

    # Determine trial mode
    if backdoor_mode is not None:
        mode = backdoor_mode
    else:
        mode = trial_sequence[current_trial]

    # Send UDP triggers
    if mode == 0:
        send_udp_message(udp_socket_main, UDP_IP_MAIN, UDP_PORT_MAIN, "200")
    else:
        send_udp_message(udp_socket_main, UDP_IP_MAIN, UDP_PORT_MAIN, "100")

    # Show feedback
    if not show_feedback(duration=5, mode=mode):
        break

    # Post-feedback message
    if mode == 0:
        messages = ["Robot Move"]
        udp_messages = ["x", "g"]
        colors = [white]
        duration = 13
    else:
        messages = ["Robot Stationary"]
        udp_messages = None
        colors = [white]
        duration = 2

    offsets = [0]
    display_multiple_messages_with_udp(
        messages=messages, colors=colors, offsets=offsets,
        duration=duration, udp_messages=udp_messages,
        udp_socket=udp_socket_extra, udp_ip=UDP_IP_EXTRA, udp_port=UDP_PORT_EXTRA
    )

    current_trial += 1
    clock.tick(60)

pygame.quit()

