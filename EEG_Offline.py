import pygame
import socket
import time
from Utils.visualization import draw_arrow_fill, draw_ball_fill, draw_fixation_cross
from Utils.experiment_utils import generate_trial_sequence, display_multiple_messages_with_udp
from Utils.networking import send_udp_message
import config

# Initialize Pygame with dimensions from config
pygame.init()
screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
pygame.display.set_caption("EEG Offline Interactive Loop")

# Screen dimensions
screen_width = config.SCREEN_WIDTH
screen_height = config.SCREEN_HEIGHT

# Initialize UDP sockets
udp_socket_main = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket_extra = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Utility function to show feedback
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
        screen.fill(config.black)

        if mode == 0:
            # Draw the arrow filling
            draw_arrow_fill(progress, screen_width, screen_height)
            draw_ball_fill(0, screen_width, screen_height)

            # Render and center message with smaller font
            message = small_font.render("Imagine Right Arm Movement", True, config.white)
        else:
            # Draw the ball filling
            draw_ball_fill(progress, screen_width, screen_height)
            draw_arrow_fill(0, screen_width, screen_height)

            # Render and center message with larger font
            message = large_font.render("Rest", True, config.white)

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
trial_sequence = generate_trial_sequence(config.TOTAL_TRIALS, config.MAX_REPEATS)
current_trial = 0
running = True
clock = pygame.time.Clock()

while running and current_trial < len(trial_sequence):
    screen.fill(config.black)
    draw_fixation_cross(screen_width, screen_height)
    draw_arrow_fill(0, screen_width, screen_height)
    draw_ball_fill(0, screen_width, screen_height)
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
        send_udp_message(udp_socket_main, config.UDP_MAIN["IP"], config.UDP_MAIN["PORT"], "200")
    else:
        send_udp_message(udp_socket_main, config.UDP_MAIN["IP"], config.UDP_MAIN["PORT"], "100")

    # Show feedback
    if not show_feedback(duration=5, mode=mode):
        break

    # Post-feedback message
    if mode == 0:
        messages = ["Robot Move"]
        udp_messages = ["x", "g"]
        colors = [config.green]
        duration = 13
    else:
        messages = ["Robot Stationary"]
        udp_messages = None
        colors = [config.white]
        duration = 2

    offsets = [0]
    display_multiple_messages_with_udp(
        messages=messages, colors=colors, offsets=offsets,
        duration=duration, udp_messages=udp_messages,
        udp_socket=udp_socket_extra, udp_ip=config.UDP_EXTRA["IP"], udp_port=config.UDP_EXTRA["PORT"]
    )

    current_trial += 1
    clock.tick(60)

pygame.quit()
