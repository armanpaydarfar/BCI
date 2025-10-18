import pygame
import socket
import sys
import time
from Utils.visualization import draw_arrow_fill, draw_ball_fill, draw_fixation_cross, draw_time_balls
from Utils.experiment_utils import generate_trial_sequence
from Utils.networking import send_udp_message,display_multiple_messages_with_udp
import config
from pylsl import StreamInlet, resolve_stream
from pathlib import Path
from Utils.logging_manager import LoggerManager
import config
import random
import os


# Initialize UDP sockets
udp_socket_marker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket_robot = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
fes_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


FES_toggle = config.FES_toggle

# Auto-detect active recording (or fallback if none)
logger = LoggerManager.auto_detect_from_subject(
    subject=config.TRAINING_SUBJECT,
    base_path=Path(config.DATA_DIR)
)

# Log config snapshot
loggable_fields = [
    "UDP_MARKER", "UDP_ROBOT", "UDP_FES",
    "ARM_SIDE", "TOTAL_TRIALS", "MAX_REPEATS",
    "TIME_MI", "TIME_ROB", "TIME_STATIONARY",
    "SHAPE_MAX", "SHAPE_MIN", "ROBOT_TRAJECTORY",
    "FES_toggle", "FES_CHANNEL", "FES_TIMING_OFFSET",
    "WORKING_DIR", "DATA_DIR", "MODEL_PATH",
    "DATA_FILE_PATH", "TRAINING_SUBJECT"
]
config_log_subset = {
    key: getattr(config, key) for key in loggable_fields if hasattr(config, key)
}
logger.save_config_snapshot(config_log_subset)
# Log the start of the offline pipeline
logger.log_event("Initialized offline EEG processing pipeline.")


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
pygame.display.set_caption("EEG Offline Interactive Loop")
info = pygame.display.Info()
screen_width = info.current_w
screen_height = info.current_h


def display_fixation_period(duration=3):
    """
    Displays a blank screen with fixation cross for a given duration.
    
    Parameters:
    - duration (int): Time in seconds for which the fixation period lasts.
    """
    logger.log_event(f"Fixation period started for {duration} seconds.")
    start_time = time.time()
    clock = pygame.time.Clock()

    while time.time() - start_time < duration:
        pygame.display.get_surface().fill(config.black)

        draw_fixation_cross(screen_width, screen_height)
        draw_ball_fill(0, screen_width, screen_height, show_threshold=False)
        draw_arrow_fill(0, screen_width, screen_height, show_threshold=False)
        draw_time_balls(0, screen_width, screen_height)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                logger.log_event("Fixation interrupted — experiment manually terminated.")
                return

        clock.tick(60)

    logger.log_event("Fixation period complete.")



# Utility function to show feedback
def show_feedback(duration=5, mode=0):
    """
    Displays feedback animation for the specified duration.

    Parameters:
        duration (float): Duration for which the animation is displayed.
        mode (int): 0 for 'Imagine Right Arm Movement', 1 for 'Rest'.
    """
    logger.log_event(f"Feedback display started — Mode: {'MI' if mode == 0 else 'REST'}, Duration: {duration}s")

    start_time = time.time()

    small_font = pygame.font.SysFont(None, 48)
    large_font = pygame.font.SysFont(None, 72)

    while time.time() - start_time < duration:
        elapsed_time = time.time() - start_time
        progress = elapsed_time / duration

        screen.fill(config.black)
        if mode == 0:
            draw_arrow_fill(progress, screen_width, screen_height, show_threshold=False)
            draw_ball_fill(0, screen_width, screen_height, show_threshold=False)
            draw_fixation_cross(screen_width, screen_height)
            draw_time_balls(2, screen_width, screen_height)
            message = pygame.font.SysFont(None, 96).render(f"Move {config.ARM_SIDE.upper()} Arm", True, config.white)
        else:
            draw_ball_fill(progress, screen_width, screen_height, show_threshold=False)
            draw_arrow_fill(0, screen_width, screen_height, show_threshold=False)
            draw_fixation_cross(screen_width, screen_height)
            draw_time_balls(3, screen_width, screen_height)
            message = pygame.font.SysFont(None, 96).render("Rest", True, config.white)

        screen.blit(
            message,
            (screen_width // 2 - message.get_width() // 2,
             screen_height // 2 - message.get_height() // 2 + 300)
        )

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                logger.log_event("Feedback interrupted — experiment manually terminated.")
                return False

    logger.log_event("Feedback display complete.")
    return True

# Main Game Loop
# Attempt to resolve the stream
logger.log_event("Attempting to resolve EEG stream...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
logger.log_event("EEG data stream detected. Starting experiment...")
trial_sequence = generate_trial_sequence(config.TOTAL_TRIALS, config.MAX_REPEATS)
logger.log_event(f"Trial Sequence: {trial_sequence}")
mode_labels = ["MI" if t == 0 else "REST" for t in trial_sequence]
logger.log_event(f"Trial Sequence (labeled): {mode_labels}")
current_trial = 0
running = True
clock = pygame.time.Clock()
display_fixation_period(duration = 3)
while running and current_trial < len(trial_sequence):
    screen.fill(config.black)
    draw_fixation_cross(screen_width, screen_height)
    draw_arrow_fill(0, screen_width, screen_height, show_threshold=False)
    draw_ball_fill(0, screen_width, screen_height, show_threshold=False)
    draw_time_balls(0, screen_width, screen_height)
    pygame.display.flip()

    backdoor_mode = None
    waiting_for_press = True
    countdown_start = None
    countdown_duration = 3000  # ms

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

        if config.TIMING:
            if countdown_start is None:
                countdown_start = pygame.time.get_ticks()

            elapsed_time = pygame.time.get_ticks() - countdown_start
            draw_time_balls(1, screen_width, screen_height)
            pygame.display.flip()

            if elapsed_time >= countdown_duration:
                logger.log_event("Timing mode: Countdown expired, proceeding automatically.")
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
                waiting_for_press = False

    if not running:
        break

    if backdoor_mode is not None:
        mode = backdoor_mode
        logger.log_event(f"Backdoor override used: {'MI' if mode == 0 else 'REST'}")
    else:
        mode = trial_sequence[current_trial]

    logger.log_event(f"Starting trial {current_trial+1}/{len(trial_sequence)} — Mode: {'MI' if mode == 0 else 'REST'}")

    # Triggers
    if mode == 0:
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["MI_BEGIN"], logger=logger)
        logger.log_event("Sent MI_BEGIN trigger.")
        if FES_toggle == 1:
            send_udp_message(fes_socket, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_SENS_GO", logger=logger)
            logger.log_event("FES sensory stimulation sent.")
        else:
            logger.log_event("FES disabled — skipping sensory stimulation.")
    else:
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["REST_BEGIN"], logger=logger)
        logger.log_event("Sent REST_BEGIN trigger.")

    # Show feedback
    logger.log_event(f"Feedback period started ({'MI' if mode == 0 else 'REST'}) for {config.TIME_MI} sec.")
    if not show_feedback(duration=config.TIME_MI, mode=mode):
        break

    # Post-feedback
    if mode == 0:
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["MI_END"], logger=logger)
        logger.log_event("Sent MI_END trigger.")
        messages = ["Robot Move"]
        selected_trajectory = random.choice(config.ROBOT_TRAJECTORY)
        udp_messages = [selected_trajectory, "g"]
        colors = [config.green]
        duration = config.TIME_ROB
        if FES_toggle == 1:
            send_udp_message(fes_socket, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_MOTOR_GO", logger=logger)
            logger.log_event("FES motor stimulation sent.")
        else:
            logger.log_event("FES disabled — skipping motor stimulation.")
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_BEGIN"], logger=logger)
        logger.log_event(f"Sent ROBOT_BEGIN trigger with trajectory: {config.ROBOT_TRAJECTORY}")
    else:
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["REST_END"], logger=logger)
        logger.log_event("Sent REST_END trigger.")
        messages = ["Robot Stationary"]
        udp_messages = None
        colors = [config.white]
        duration = config.TIME_STATIONARY

    logger.log_event(f"Displayed message: '{messages[0]}' for {duration} sec.")
    display_multiple_messages_with_udp(
        messages=messages, colors=colors, offsets=[0],
        duration=duration, udp_messages=udp_messages,
        udp_socket=udp_socket_robot, udp_ip=config.UDP_ROBOT["IP"], udp_port=config.UDP_ROBOT["PORT"], logger = logger
    )

    if mode == 0:
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_END"], logger=logger)
        logger.log_event("Sent ROBOT_END trigger.")
        display_fixation_period(duration = 2)
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_HOME"], logger=logger)

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

    display_fixation_period(duration=3)
    logger.log_event("Displayed fixation period.")

    current_trial += 1
    clock.tick(60)


pygame.quit()
logger.log_event("Experiment terminated.")

