import pygame
import socket
import sys
import time
from Utils.visualization import draw_arrow_fill, draw_ball_fill, draw_fixation_cross, draw_time_balls
from Utils.experiment_utils import generate_trial_sequence, display_multiple_messages_with_udp
from Utils.networking import send_udp_message
import config
from pylsl import StreamInlet, resolve_stream



# Initialize Pygame with dimensions from config
pygame.init()
screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
#screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

pygame.display.set_caption("EEG Offline Interactive Loop")

# Screen dimensions
screen_width = config.SCREEN_WIDTH
screen_height = config.SCREEN_HEIGHT

# Initialize UDP sockets
udp_socket_marker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket_robot = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
fes_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


FES_toggle = config.FES_toggle



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
        draw_ball_fill(0, screen_width, screen_height, show_threshold=False)  # Empty fill
        draw_arrow_fill(0, screen_width, screen_height, show_threshold=False)  # Empty fill
        draw_time_balls(0,screen_width,screen_height)
        pygame.display.flip()  # Update display

        # Check for quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        clock.tick(60)  # Maintain 60 FPS


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
            draw_arrow_fill(progress, screen_width, screen_height, show_threshold=False)
            draw_ball_fill(0, screen_width, screen_height, show_threshold=False)
            draw_fixation_cross(screen_width, screen_height)
            draw_time_balls(2, screen_width, screen_height)

            # Render and center message with smaller font
            message = pygame.font.SysFont(None, 96).render(f"Move {config.ARM_SIDE.upper()} Arm", True, config.white)
        else:
            # Draw the ball filling
            draw_ball_fill(progress, screen_width, screen_height, show_threshold=False)
            draw_arrow_fill(0, screen_width, screen_height, show_threshold=False)
            draw_fixation_cross(screen_width, screen_height)
            draw_time_balls(3, screen_width, screen_height)


            # Render and center message with larger font
            message = pygame.font.SysFont(None, 96).render("Rest", True, config.white)


        # Center the message properly on the screen
        screen.blit(
            message,
            (screen_width // 2 - message.get_width() // 2,
             screen_height // 2 - message.get_height() // 2 + 300)
        )

        pygame.display.flip()

        # Event handling to allow quitting
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

    return True

# Main Game Loop
# Attempt to resolve the stream
print("Looking for EEG data stream...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
print("EEG data stream detected. Starting experiment...")
trial_sequence = generate_trial_sequence(config.TOTAL_TRIALS, config.MAX_REPEATS)
print(f"Trial Sequence: {trial_sequence}")
current_trial = 0
running = True
clock = pygame.time.Clock()
display_fixation_period(duration = 3)
while running and current_trial < len(trial_sequence):
    screen.fill(config.black)
    draw_fixation_cross(screen_width, screen_height)
    draw_arrow_fill(0, screen_width, screen_height, show_threshold=False)  # Replace arrow with bar
    draw_ball_fill(0, screen_width, screen_height, show_threshold=False)
    draw_time_balls(0,screen_width,screen_height)
    pygame.display.flip()

    # Backdoor logic
    backdoor_mode = None
    waiting_for_press = True
    countdown_start = None
    countdown_duration = 3000  # 3 seconds in milliseconds

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

        # Timing-based execution logic
        if config.TIMING:
            if countdown_start is None:
                countdown_start = pygame.time.get_ticks()  # Start countdown

            elapsed_time = pygame.time.get_ticks() - countdown_start

            # Draw timing balls during countdown
            next_trial_mode = trial_sequence[current_trial]  # Get the mode for the next trial
            draw_time_balls(1, screen_width, screen_height)
            
            pygame.display.flip()  # Update the display with time balls

            if elapsed_time >= countdown_duration:
                print("Countdown complete, proceeding automatically.")
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
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
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["MI_BEGIN"])
        send_udp_message(fes_socket, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_SENS_GO") if FES_toggle == 1 else print("FES is disabled. Skipping interaction.")
    else:
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["REST_BEGIN"])

    # Show feedback
    if not show_feedback(duration=config.TIME_MI, mode=mode):
        break

    # Post-feedback message
    if mode == 0:
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["MI_END"])
        messages = ["Robot Move"]
        udp_messages = ["x", "g"]
        colors = [config.green]
        duration = config.TIME_ROB
        send_udp_message(fes_socket, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_MOTOR_GO") if FES_toggle == 1 else print("FES is disabled. Skipping interaction.")
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_BEGIN"])
    else:
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["REST_END"])
        messages = ["Robot Stationary"]
        udp_messages = None
        colors = [config.white]
        duration = config.TIME_STATIONARY

    offsets = [0]
    display_multiple_messages_with_udp(
        messages=messages, colors=colors, offsets=offsets,
        duration=duration, udp_messages=udp_messages,
        udp_socket=udp_socket_robot, udp_ip=config.UDP_ROBOT["IP"], udp_port=config.UDP_ROBOT["PORT"]
    )
    if mode == 0:
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_END"])
    
    display_fixation_period(duration = 3)

    current_trial += 1
    clock.tick(60)

pygame.quit()
