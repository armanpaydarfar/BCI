import pygame
import socket
import pickle
import datetime
import os
import random
import time
import serial
import sys  # ✅ faltaba (lo usas en sys.exit)
from pylsl import StreamInlet, resolve_stream

import mne
mne.set_log_level("WARNING")

from Utils.visualization import (
    draw_arrow_fill, draw_ball_fill, draw_fixation_cross,
    draw_progress_bar
)
from Utils.experiment_utils import (
    generate_trial_sequence, save_transform, load_transform
)
from Utils.EEGStreamState import EEGStreamState
from Utils.networking import send_udp_message, display_multiple_messages_with_udp
import config
from pathlib import Path
from Utils.logging_manager import LoggerManager

# Import runtime_common
from Utils.runtime_common import (
    log_confusion_matrix_from_trial_summary,
    append_trial_probabilities_to_csv,
    display_fixation_period,
    hold_messages_and_classify,
    classify_real_time, LeakyIntegrator, 
    handle_fes_activation,
    calculate_fill_levels
)
import Utils.runtime_common as _RC


# ============================================================
# LOGGING & CONFIG
# ============================================================
logger = LoggerManager.auto_detect_from_subject(
    subject=config.TRAINING_SUBJECT,
    base_path=Path(config.DATA_DIR),
    mode="online"
)
# Log config snapshot
loggable_fields = [
    "UDP_MARKER", "UDP_ROBOT", "UDP_FES", "ARM_SIDE", "TOTAL_TRIALS",
    "TIME_MI", "FES_toggle", "TRAINING_SUBJECT"
]
config_log_subset = {k: getattr(config, k) for k in loggable_fields if hasattr(config, k)}
logger.save_config_snapshot(config_log_subset)

eeg_dir = logger.log_base / "eeg"
adaptive_T_path = eeg_dir / "adaptive_T.pkl"

Prev_T, counter = load_transform(adaptive_T_path)
if Prev_T is None:
    counter = 0
    logger.log_event("ℹ️ No adaptive transform found — starting fresh.")
else:
    logger.log_event(f"✅ Loaded adaptive transform with counter = {counter}")

pygame.init()

# 1) Resolución actual del monitor ANTES de crear la ventana
info_monitor = pygame.display.Info()
monitor_w = info_monitor.current_w
monitor_h = info_monitor.current_h

if config.BIG_BROTHER_MODE:
    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
    screen = pygame.display.set_mode((1920, 1080),pygame.NOFRAME)
    # Si tú quieres forzar 1920x1080 aquí, lo puedes hacer,
    # pero para que el indicador se vea proporcional, lo dejamos dinámico:
    screen_width = monitor_w
    screen_height = monitor_h
else:
    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
    screen = pygame.display.set_mode((monitor_w, monitor_h), pygame.NOFRAME)
    screen_width = monitor_w
    screen_height = monitor_h

pygame.display.set_caption("EEG Online Interactive Loop")
info = pygame.display.Info()
screen_width = info.current_w
screen_height = info.current_h

# UDP Settings
udp_socket_marker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket_robot = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket_fes = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
FES_toggle = config.FES_toggle


# ============================================================
# ARDUINO SETUP
# ============================================================
ARDUINO_PORT = config.ARDUINO_PORT
ARDUINO_BAUD = int(os.environ.get("ARDUINO_BAUD", 9600))
arduino = None

if ARDUINO_PORT:
    try:
        logger.log_event(f"Connecting to Glove (Arduino) on {ARDUINO_PORT}...")
        arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=0.1)
        time.sleep(2)  # CRITICAL: Safety wait for Arduino reset
        logger.log_event("✅ Glove connected successfully.")
    except Exception as e:
        logger.log_event(f"❌ Error connecting to Glove: {e}", level="error")
        arduino = None
else:
    logger.log_event("ℹ️ No Arduino port configured.")

# Load Model
subject_model_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "models")
subject_model_path = os.path.join(subject_model_dir, f"sub-{config.TRAINING_SUBJECT}_model.pkl")

try:
    with open(subject_model_path, 'rb') as f:
        model = pickle.load(f)
    logger.log_event(f"✅ Model loaded: {subject_model_path}")
except FileNotFoundError:
    logger.log_event(f"❌ Model not found: {subject_model_path}", level="error")
    sys.exit(1)

predictions_list = []
ground_truth_list = []
# ============================================================
# WIRE RUNTIME OBJECTS
# ============================================================
_RC.config = config
_RC.logger = logger
_RC.model = model
_RC.screen = screen
_RC.screen_width = screen_width
_RC.screen_height = screen_height
_RC.udp_socket_marker = udp_socket_marker
_RC.udp_socket_robot  = udp_socket_robot
_RC.udp_socket_fes    = udp_socket_fes
_RC.FES_toggle = FES_toggle
_RC.Prev_T = Prev_T
_RC.counter = counter

# NOTE: We do not pass '_RC.arduino' because runtime_common
# will not handle the glove. The glove is handled by this main script.


# ============================================================
# ✅ PRE-TRIAL INDICATOR (MATCH OFFLINE LOOK)
# ============================================================
NEXT_INDICATOR_POS = (0.50, 0.28)
NEXT_INDICATOR_SCALE = 1.00



def show_feedback(duration=5, mode=0, eeg_state = None):
    """
    Displays feedback animation, collects EEG data, and performs real-time classification
    using a sliding window approach with early stopping based on posterior probabilities.
    """
    start_time = time.time()
    step_size = config.STEP_SIZE  # Sliding window step size (seconds)
    window_size = config.CLASSIFY_WINDOW / 1000  # Convert ms to seconds
    window_size_samples = int(window_size * config.FS)
    step_size_samples = int(step_size * config.FS)
    FES_active = False
    all_probabilities = []
    predictions = []
    running_avg_list = []
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
    pygame.display.flip()

    # Send UDP triggers
    if mode == 0:  # Red Arrow Mode (Motor Imagery)
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["MI_BEGIN"], logger=logger)
    else:  # Blue Ball Mode (Rest)
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["REST_BEGIN"], logger=logger)

    clock = pygame.time.Clock()
    running_avg_confidence = 0.5  # Initial placeholder
    current_confidence = 0.5 # Initial placeholder for initial window updates
    next_tick = start_time + window_size  # Skip first second

    while time.time() - start_time < duration:
        eeg_state.update()

        now = time.time()
        just_classified = False
        if now >= next_tick:
            current_confidence, predictions, all_probabilities = classify_real_time(
                eeg_state,
                window_size_samples,
                all_probabilities,
                predictions,
                mode,
                leaky_integrator
            )
            next_tick += step_size 
            if all_probabilities and getattr(config, "SEND_PROBS", False):
                prob_mi, prob_rest = all_probabilities[-1][2], all_probabilities[-1][1]
                send_udp_message(
                    udp_socket_marker,
                    config.UDP_MARKER["IP"],
                    config.UDP_MARKER["PORT"],
                    f"{config.TRIGGERS['MI_PROBS' if mode == 0 else 'REST_PROBS']},{prob_mi:.5f},{prob_rest:.5f}",
                    quiet=True
                )
            just_classified=True


        running_avg_confidence = leaky_integrator.update(current_confidence)
        if FES_toggle == 1:
            FES_active = handle_fes_activation(mode, running_avg_confidence, FES_active)

        screen.fill(config.black)
        MI_fill, Rest_fill = calculate_fill_levels(running_avg_confidence, mode)
        if just_classified and all_probabilities:
            ts, prest_inst, pmi_inst = all_probabilities[-1]
            if mode == 0:  # MI trial: confidence is P(MI)
                pmi_avg   = running_avg_confidence
                prest_avg = 1.0 - running_avg_confidence
            else:          # REST trial: confidence is P(REST)
                prest_avg = running_avg_confidence
                pmi_avg   = 1.0 - running_avg_confidence
            all_probabilities[-1] = [ts, prest_inst, pmi_inst, pmi_avg, prest_avg]

        if mode == 0:
            draw_arrow_fill(MI_fill, screen_width, screen_height)
            draw_fixation_cross(screen_width, screen_height)
            draw_ball_fill(Rest_fill, screen_width, screen_height)
            draw_time_balls(2, screen_width, screen_height)
            message = pygame.font.SysFont(None, 96).render(f"Imagine Close {config.ARM_SIDE.upper()} Hand", True, config.white)
        else:
            draw_ball_fill(Rest_fill, screen_width, screen_height)
            draw_fixation_cross(screen_width, screen_height)
            draw_arrow_fill(MI_fill, screen_width, screen_height)
            draw_time_balls(3, screen_width, screen_height)
            message = pygame.font.SysFont(None, 96).render("Rest", True, config.white)

        screen.blit(message, (screen_width // 2 - message.get_width() // 2, screen_height // 2 + 300))
        pygame.display.flip()
        clock.tick(60)
        # --- Early-stop logic (supports correct-only or either-threshold) ---
        hit_correct   = (len(predictions) >= min_predictions) and (running_avg_confidence >= accuracy_threshold)
        hit_incorrect = (len(predictions) >= min_predictions) and (running_avg_confidence <= (1 - opposed_threshold))

        should_earlystop = hit_correct or (config.EARLYSTOP_MODE == "either" and hit_incorrect)
        if should_earlystop:
            earlystop_flag = True

            # Figure out which class triggered the stop (for logging/triggers)
            if hit_correct:
                stop_reason = "correct"
                trigger_key = "MI_EARLYSTOP" if mode == 0 else "REST_EARLYSTOP"
            else:
                stop_reason = "incorrect"
                trigger_key = "REST_EARLYSTOP" if mode == 0 else "MI_EARLYSTOP"

            logger.log_event(
                f"Early stopping triggered ({stop_reason}). "
                f"Confidence={running_avg_confidence:.2f}, "
                f"min_preds={min_predictions}, "
                f"mode={'MI' if mode==0 else 'REST'}"
            )

            # Stop FES if active
            if FES_toggle == 1:
                send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_STOP", logger=logger)
            else:
                logger.log_event("FES is disabled.")

            # Emit the appropriate EARLYSTOP trigger
            send_udp_message(
                udp_socket_marker,
                config.UDP_MARKER["IP"],
                config.UDP_MARKER["PORT"],
                config.TRIGGERS[trigger_key],
                logger=logger
            )
            break

    
    pygame.display.flip()
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


    send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["MI_END" if mode==0 else "REST_END"], logger=logger)
    pygame.time.delay(300)  # ~300 ms delay to allow the visual feedback to complete rendering
    return final_class, running_avg_confidence, leaky_integrator, all_probabilities, earlystop_flag


def draw_time_balls(
    ball_state,
    screen_width,
    screen_height,
    ball_radius=None,
    mode="single",
    indicator_color=None,
    single_pos=(0.50, 0.28),   # (x_ratio, y_ratio) for the single indicator
    stack_pos=(0.12, 0.45),    # (x_ratio, y_ratio) for the top ball in the stack (left side)
    stack_spacing_ratio=0.08   # vertical spacing between stacked balls (as ratio of screen height)
):
    """
    Draw a time indicator ball with 4 possible states:
      - 0 = Empty (Outlined Ball)
      - 1 = White Ball (Baseline/Neutral)
      - 2 = Red Ball (Motor Imagery)
      - 3 = Blue Ball (Rest)

    Key design choice:
    - All geometry is defined using screen ratios so it stays consistent across resolutions/monitors.

    Args:
        ball_state (int): 0..3
        screen_width (int): Current screen width in pixels
        screen_height (int): Current screen height in pixels
        ball_radius (int|None): If None, radius auto-scales with screen size
        mode (str): "single" or "stack"
        indicator_color (tuple|None): If provided, overrides the fill color of the single ball
        single_pos (tuple): (x_ratio, y_ratio) anchor for single ball
        stack_pos (tuple): (x_ratio, y_ratio) anchor for stacked balls (top ball)
        stack_spacing_ratio (float): vertical spacing between stacked balls
    """

    # --- Auto-scale radius if not provided ---
    # This keeps the ball visually consistent across 1080p/4K, etc.
    if ball_radius is None:
        ball_radius = int(min(screen_width, screen_height) * 0.035)  # ~3.5% of min dimension

    # --- Define colors for each state ---
    color_map = {
        1: (255, 255, 255),  # White (Baseline)
        2: (255, 0, 0),      # Red (MI)
        3: (0, 120, 255)     # Blue (Rest) - slightly nicer blue than pure (0,0,255)
    }

    # Default color from state
    ball_color = color_map.get(ball_state, (255, 255, 255))

    # If user wants a custom indicator (e.g. MI/REST preview), override only for FILLED ball
    if indicator_color is not None and ball_state != 0:
        ball_color = indicator_color

    surf = pygame.display.get_surface()

    if mode == "single":
        # --- Single ball anchored by screen ratios ---
        ball_x = int(screen_width * single_pos[0])
        ball_y = int(screen_height * single_pos[1])

        if ball_state == 0:
            # Outlined ball (empty)
            pygame.draw.circle(surf, (255, 255, 255), (ball_x, ball_y), ball_radius, 2)
        else:
            # Filled ball
            pygame.draw.circle(surf, ball_color, (ball_x, ball_y), ball_radius)

    elif mode == "stack":
        # --- Three stacked balls for countdown (anchored by ratios) ---
        stack_x = int(screen_width * stack_pos[0])
        stack_y_start = int(screen_height * stack_pos[1])
        spacing = int(screen_height * stack_spacing_ratio)

        for i in range(3):
            ball_y = stack_y_start + i * spacing
            if ball_state == 0:
                pygame.draw.circle(surf, (255, 255, 255), (stack_x, ball_y), ball_radius, 2)
            else:
                pygame.draw.circle(surf, ball_color, (stack_x, ball_y), ball_radius)



def draw_arrow_directional(screen, pos_x, pos_y, size, color, direction="right"):
    """
    Draws a complete arrow (line + triangle tip) with offset correction.
    The line end is adjusted to stay behind the triangle's tip.
    """
    # 1. Geometry Setup
    line_len = size * 0.8
    tri_size = size // 2
    
    # OFFSET CORRECTION: Move the line end point slightly 'inwards' 
    # so it doesn't poke out of the triangle's tip.
    offset = 5  # pixels
    
    if direction == "right":
        line_start = (pos_x - line_len, pos_y)
        line_end = (pos_x + line_len - offset, pos_y) # Pulled back
        
        # Tip points (Right)
        points = [
            (pos_x + line_len, pos_y),                  # Tip
            (pos_x + line_len - tri_size, pos_y - tri_size), # Top back
            (pos_x + line_len - tri_size, pos_y + tri_size)  # Bottom back
        ]
    else:
        line_start = (pos_x + line_len, pos_y)
        line_end = (pos_x - line_len + offset, pos_y) # Pulled back
        
        # Tip points (Left)
        points = [
            (pos_x - line_len, pos_y),                  # Tip
            (pos_x - line_len + tri_size, pos_y - tri_size), # Top back
            (pos_x - line_len + tri_size, pos_y + tri_size)  # Bottom back
        ]

    # 2. Draw Body (Line)
    pygame.draw.line(screen, color, line_start, line_end, 12)

    # 3. Draw Tip (Triangle)
    pygame.draw.polygon(screen, color, points)


def draw_arrow_directional(screen, pos_x, pos_y, size, color, direction="right"):
    """
    Flecha completa: línea + triángulo (igual que offline).
    """
    line_len = size * 0.8
    tri_size = size // 2
    offset = 5  # px

    if direction == "right":
        line_start = (pos_x - line_len, pos_y)
        line_end = (pos_x + line_len - offset, pos_y)
        points = [
            (pos_x + line_len, pos_y),
            (pos_x + line_len - tri_size, pos_y - tri_size),
            (pos_x + line_len - tri_size, pos_y + tri_size),
        ]
    else:
        line_start = (pos_x + line_len, pos_y)
        line_end = (pos_x - line_len + offset, pos_y)
        points = [
            (pos_x - line_len, pos_y),
            (pos_x - line_len + tri_size, pos_y - tri_size),
            (pos_x - line_len + tri_size, pos_y + tri_size),
        ]

    pygame.draw.line(screen, color, line_start, line_end, 12)
    pygame.draw.polygon(screen, color, points)

def draw_pretrial_screen_online(mode, time_ball_state=1):
    """
    Replica el look de OFFLINE en preparación:
      - MI: cuadro rojo + flecha derecha
      - REST: círculo azul + flecha izquierda
      - time_balls en mode='single' en el indicador NEXT
    """
    screen.fill(config.black)
    draw_fixation_cross(screen_width, screen_height)

    pos_x = int(screen_width * NEXT_INDICATOR_POS[0])
    pos_y = int(screen_height * NEXT_INDICATOR_POS[1])
    base_size = int(min(screen_width, screen_height) * 0.08 * NEXT_INDICATOR_SCALE)

    is_mi = (mode == 0)
    next_color = (255, 50, 50) if is_mi else (0, 120, 255)

    # 1) Shape background
    if is_mi:
        bg_rect = pygame.Rect(pos_x - base_size // 2, pos_y - base_size // 2, base_size, base_size)
        pygame.draw.rect(screen, next_color, bg_rect)
    else:
        pygame.draw.circle(screen, next_color, (pos_x, pos_y), base_size // 2)

    # 2) Single time-ball indicator (igual al offline)
    draw_time_balls(
        time_ball_state,
        screen_width,
        screen_height,
        mode="single",
        indicator_color=next_color,
        single_pos=NEXT_INDICATOR_POS,
        ball_radius=int(base_size * 0.4),
    )

    # 3) Texto de preparación
    font_prep = pygame.font.SysFont(None, 96)
    if is_mi:
        prep_msg = f"Prepare to close {config.ARM_SIDE.upper()} hand"
    else:
        prep_msg = "Rest"

    txt_surface = font_prep.render(prep_msg, True, config.white)
    screen.blit(
        txt_surface,
        (screen_width // 2 - txt_surface.get_width() // 2, screen_height // 2 + 300),
    )

    # 4) Flecha direccional
    arrow_dir = "right" if is_mi else "left"
    draw_arrow_directional(screen, pos_x, pos_y, base_size // 2.5, (255, 255, 255), direction=arrow_dir)

    pygame.display.flip()


def main():
    logger.log_event("Resolving EEG data stream via LSL...")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    eeg_state = EEGStreamState(inlet=inlet, config=config, logger=logger)

    trial_sequence = generate_trial_sequence(total_trials=config.TOTAL_TRIALS, max_repeats=config.MAX_REPEATS)
    current_trial = 0
    running = True
    clock = pygame.time.Clock()

    display_fixation_period(duration=3, eeg_state=eeg_state)

    # Ensure glove is open at start
    if arduino:
        arduino.write(b'0')

    while running and current_trial < len(trial_sequence):
        logger.log_event(f"--- Trial {current_trial+1}/{len(trial_sequence)} START ---")

        # 1) Decide modo del trial
        mode = trial_sequence[current_trial]

        # 2) ✅ Pantalla de preparación con el mismo look que OFFLINE
        draw_pretrial_screen_online(mode=mode, time_ball_state=1)

        # 3) Waiting / Countdown
        waiting_for_press = True
        countdown_start = None
        countdown_duration = 1500  # ms

        while waiting_for_press:
            eeg_state.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    waiting_for_press = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting_for_press = False
            

            if config.TIMING:
                if countdown_start is None:
                    countdown_start = pygame.time.get_ticks()
                elapsed = pygame.time.get_ticks() - countdown_start

                # ✅ Re-dibujar la pantalla de preparación para mantener el indicador visible
                #    (puedes animar time_ball_state si quieres; aquí lo mantenemos en 1)
                draw_pretrial_screen_online(mode=mode, time_ball_state=1)

                if elapsed >= countdown_duration:
                    waiting_for_press = False
            else:
                # Si no hay timing, igual mantenemos la pantalla
                draw_pretrial_screen_online(mode=mode, time_ball_state=1)

            clock.tick(60)

        if not running:
            break

        mode = trial_sequence[current_trial]

        # 4) Baseline
        try:
            eeg_state.compute_baseline(duration_sec=config.BASELINE_DURATION)
        except ValueError:
            continue

        # -----------------------------------------------------------
        # PHASE 1: EFFORT (Sensory FES Only)
        # -----------------------------------------------------------
        prediction, confidence, leaky_integrator, trial_probs, earlystop_flag = show_feedback(
            duration=config.TIME_MI,
            mode=mode,
            eeg_state=eeg_state
        )

        append_trial_probabilities_to_csv(
            trial_probabilities=trial_probs, mode=mode, trial_number=current_trial + 1,
            predicted_label=prediction, early_cutout=earlystop_flag,
            mi_threshold=config.THRESHOLD_MI, rest_threshold=config.THRESHOLD_REST,
            logger=logger, phase="MI" if mode == 0 else "REST"
        )

        # -----------------------------------------------------------
        # PHASE 2: REWARD (Motor FES + Glove + Robot)
        # -----------------------------------------------------------

        predictions_list.append(prediction)
        ground_truth_list.append(200 if mode == 0 else 100)

        if mode == 0:  # MI Trial
            if prediction == 200:  # SUCCESS (Threshold reached)

                # 1) CLOSE GLOVE (Reward Trigger)
                if arduino:
                    arduino.write(b'1')
                    logger.log_event("✅ Prediction Success -> Closing Glove (Reward)")

                # 2) MOTOR FES
                if FES_toggle:
                    send_udp_message(
                        udp_socket_fes,
                        config.UDP_FES["IP"],
                        config.UDP_FES["PORT"],
                        "FES_MOTOR_GO",
                        logger=logger
                    )

                # 3) ROBOT
                messages = ["Correct", "Hand close"]
                colors = [config.green, config.green]
                send_udp_message(
                    udp_socket_marker,
                    config.UDP_MARKER["IP"],
                    config.UDP_MARKER["PORT"],
                    config.TRIGGERS["ROBOT_BEGIN"],
                    logger=logger
                )

                display_multiple_messages_with_udp(
                    messages=messages,
                    colors=colors,
                    offsets=[-100, 100],
                    duration=0.01,
                    udp_messages=[random.choice(config.ROBOT_TRAJECTORY), config.ROBOT_OPCODES["GO"]],
                    udp_socket=udp_socket_robot,
                    udp_ip=config.UDP_ROBOT["IP"],
                    udp_port=config.UDP_ROBOT["PORT"],
                    logger=logger,
                    eeg_state=eeg_state
                )

                final_class, robot_probs, early = hold_messages_and_classify(
                    messages, colors, [-100, 100],
                    5, 0,
                    udp_socket_robot, config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"],
                    eeg_state, leaky_integrator
                )

                # Robot home
                send_udp_message(
                    udp_socket_robot,
                    config.UDP_ROBOT["IP"],
                    config.UDP_ROBOT["PORT"],
                    config.ROBOT_OPCODES["HOME"],
                    logger=logger,
                    expect_ack=True
                )

            else:  # FAIL (Threshold not reached)
                if arduino:
                    arduino.write(b'0')
                display_multiple_messages_with_udp(
                    ["Incorrect", "Hand Stationary"],
                    [config.red, config.white],
                    [-100, 100],
                    config.TIME_STATIONARY,
                    None,
                    udp_socket_robot,
                    config.UDP_ROBOT["IP"],
                    config.UDP_ROBOT["PORT"],
                    logger,
                    eeg_state
                )

        else:  # REST Trial
            msg_txt = "Correct" if prediction == 100 else "Incorrect"
            col = config.green if prediction == 100 else config.red
            if arduino:
                arduino.write(b'0')
            display_multiple_messages_with_udp(
                [msg_txt, "Hand Stationary"],
                [col, config.white],
                [-100, 100],
                config.TIME_STATIONARY,
                None,
                udp_socket_robot,
                config.UDP_ROBOT["IP"],
                config.UDP_ROBOT["PORT"],
                logger,
                eeg_state
            )

        # -----------------------------------------------------------
        # PHASE 3: RELAXATION (End of Trial)
        # -----------------------------------------------------------
        if arduino:
            arduino.write(b'0')

        display_fixation_period(duration=3, eeg_state=eeg_state)
        current_trial += 1

    # Cleanup / Save adaptive
    if current_trial == len(trial_sequence) and config.SAVE_ADAPTIVE_T:
        save_transform(Prev_T, counter, adaptive_T_path)

    log_confusion_matrix_from_trial_summary(logger)

    if arduino:
        arduino.write(b'0')
        arduino.close()

    pygame.quit()


if __name__ == "__main__":
    main()
