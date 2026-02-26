#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pygame
import socket
import sys
import time
import random
import os
import serial  # <--- AGREGADO: Comunicación Serial
from pathlib import Path
from pylsl import StreamInlet, resolve_stream

# Personal modules
from Utils.visualization import draw_arrow_fill, draw_ball_fill, draw_fixation_cross
from Utils.experiment_utils import generate_trial_sequence
from Utils.networking import send_udp_message, display_multiple_messages_with_udp
import config
from Utils.logging_manager import LoggerManager

# ============================================================
# CONFIG
# ============================================================
NEXT_INDICATOR_POS = (0.50, 0.28)
NEXT_INDICATOR_SCALE = 1.00

# UDP Sockets
udp_socket_marker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket_robot = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
fes_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

FES_toggle = config.FES_toggle

# --- ARDUINO CONFIG (AGREGADO) ---
# OPCIÓN A: Usar con Control Panel (Déjala comentada para tu prueba de AHORITA)
ARDUINO_PORT = config.ARDUINO_PORT

# OPCIÓN B: Usar Manual desde Terminal (Descomentada AHORA)
# ARDUINO_PORT = "/dev/ttyACM0"

ARDUINO_BAUD = int(os.environ.get("ARDUINO_BAUD", 9600))
arduino_ser = None  # Variable para la conexión
# ---------------------------------

# Logging
logger = LoggerManager.auto_detect_from_subject(
    subject=config.TRAINING_SUBJECT,
    base_path=Path(config.DATA_DIR)
)

# Config snapshot
loggable_fields = ["UDP_MARKER", "UDP_ROBOT", "UDP_FES", "ARM_SIDE", "TOTAL_TRIALS", "TIME_MI", "FES_toggle"]
config_log_subset = {k: getattr(config, k) for k in loggable_fields if hasattr(config, k)}
logger.save_config_snapshot(config_log_subset)

pygame.init()

if config.BIG_BROTHER_MODE:
    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
    screen = pygame.display.set_mode((1920, 1080), pygame.NOFRAME)
else:
    screen = pygame.display.set_mode((0, 0), pygame.NOFRAME)

screen_width, screen_height = pygame.display.Info().current_w, pygame.display.Info().current_h

# ============================================================
# ARDUINO SETUP (AGREGADO)
# ============================================================
if ARDUINO_PORT:
    try:
        logger.log_event(f"Connecting to Glove (Arduino) on {ARDUINO_PORT}...")
        arduino_ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=0.1)
        # Espera de seguridad para el reinicio del Arduino
        time.sleep(2)
        logger.log_event("Glove connected successfully.")
    except Exception as e:
        logger.log_event(f"ERROR connecting to Glove: {e}")
        arduino_ser = None
else:
    logger.log_event("No Arduino port configured (Visual Only Mode).")

# ============================================================
# VISUAL FUNCTIONS (TUS FUNCIONES EXACTAS)
# ============================================================

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

def display_fixation_period(duration=3):
    start_time = time.time()
    clock = pygame.time.Clock()
    while time.time() - start_time < duration:
        pygame.display.get_surface().fill(config.black)
        draw_fixation_cross(screen_width, screen_height)
        draw_ball_fill(0, screen_width, screen_height, show_threshold=False)
        draw_arrow_fill(0, screen_width, screen_height, show_threshold=False)
        draw_time_balls(0, screen_width, screen_height, mode="single", single_pos=NEXT_INDICATOR_POS)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
        clock.tick(60)

def draw_pretrial_screen(next_color, time_ball_state):
    """Pre-trial with Line+Triangle arrow."""
    screen.fill(config.black)
    draw_fixation_cross(screen_width, screen_height)
    
    pos_x = int(screen_width * NEXT_INDICATOR_POS[0])
    pos_y = int(screen_height * NEXT_INDICATOR_POS[1])
    base_size = int(min(screen_width, screen_height) * 0.08 * NEXT_INDICATOR_SCALE)
    
    is_mi = (next_color == (255, 50, 50) or next_color == getattr(config, 'red', (255, 50, 50)))

    # 1. Outer Background (White)
    if is_mi:
        bg_rect = pygame.Rect(pos_x - base_size//2, pos_y - base_size//2, base_size, base_size)
        pygame.draw.rect(screen, (255, 50, 50), bg_rect)
    else:
        pygame.draw.circle(screen, (0, 120, 255), (pos_x, pos_y), base_size // 2)

    # 2. Middle Color
    draw_time_balls(time_ball_state, screen_width, screen_height, mode="single", 
                    indicator_color=next_color, single_pos=NEXT_INDICATOR_POS, ball_radius=int(base_size * 0.4))
    
    # Text Section
    font_prep = pygame.font.SysFont(None, 72) 
    if is_mi:
        prep_msg = f"Prepare: Flex {config.ARM_SIDE.upper()} Hand"
    else:
        prep_msg = "Rest"
    
    txt_surface = font_prep.render(prep_msg, True, config.white)
    screen.blit(txt_surface, (screen_width // 2 - txt_surface.get_width() // 2, screen_height // 2 + 300))

    # 3. Directional Arrow (Line + Triangle)
    arrow_dir = "right" if is_mi else "left"
    draw_arrow_directional(screen, pos_x, pos_y, base_size // 2.5, (255, 255, 255), direction=arrow_dir)
    
    pygame.display.flip()

def show_feedback(duration, mode):
    """Feedback phase keeping the arrow for continuity."""
    start_time = time.time()
    pos_x, pos_y = int(screen_width * NEXT_INDICATOR_POS[0]), int(screen_height * NEXT_INDICATOR_POS[1])
    base_size = int(min(screen_width, screen_height) * 0.08 * NEXT_INDICATOR_SCALE)

    # --- CONTROL ARDUINO (AGREGADO) ---
    if arduino_ser and arduino_ser.is_open:
        try:
            command = b'1' if mode == 0 else b'0'
            arduino_ser.write(command)
        except Exception as e:
            logger.log_event(f"Arduino Error: {e}")
    # ----------------------------------

    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        screen.fill(config.black)
        draw_fixation_cross(screen_width, screen_height)

        if mode == 0: # MI
            draw_arrow_fill(progress, screen_width, screen_height, False)
            # Maintain Visual Identity (Square)
            bg_rect = pygame.Rect(pos_x - base_size//2, pos_y - base_size//2, base_size, base_size)
            pygame.draw.rect(screen, (255, 50, 50), bg_rect)
            pygame.draw.rect(screen, (255, 50, 50), pygame.Rect(pos_x - int(base_size*0.35), 
                             pos_y - int(base_size*0.35), int(base_size*0.7), int(base_size*0.7)))
            # Keep Arrow
            draw_arrow_directional(screen, pos_x, pos_y, base_size // 2.5, (255, 255, 255), "right")
            msg = f"Imagine Closing {config.ARM_SIDE.upper()} Hand"
        else: # REST
            draw_ball_fill(progress, screen_width, screen_height, False)
            # Maintain Visual Identity (Circle)
            pygame.draw.circle(screen, (0, 120, 255), (pos_x, pos_y), base_size // 2)
            pygame.draw.circle(screen, (0, 120, 255), (pos_x, pos_y), int(base_size * 0.35))
            # Keep Arrow
            draw_arrow_directional(screen, pos_x, pos_y, base_size // 2.5, (255, 255, 255), "left")
            msg = "Rest"

        txt = pygame.font.SysFont(None, 96).render(msg, True, config.white)
        screen.blit(txt, (screen_width//2 - txt.get_width()//2, screen_height//2 + 300))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
    return True

# ============================================================
# MAIN LOOP
# ============================================================
logger.log_event("Resolving EEG stream...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])

trial_sequence = generate_trial_sequence(config.TOTAL_TRIALS, config.MAX_REPEATS)
current_trial = 0
clock = pygame.time.Clock()

display_fixation_period(duration=3)

try:
    while current_trial < len(trial_sequence):
        next_mode = trial_sequence[current_trial]
        next_color = (255, 50, 50) if next_mode == 0 else (0, 120, 255)

        # PRE-TRIAL
        draw_pretrial_screen(next_color, time_ball_state=1)
        
        backdoor_mode, waiting = None, True
        countdown_start = pygame.time.get_ticks()
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT: backdoor_mode = 0; waiting = False
                    if event.key == pygame.K_DOWN: backdoor_mode = 1; waiting = False
            if config.TIMING and (pygame.time.get_ticks() - countdown_start >= 2000):
                waiting = False
            draw_pretrial_screen(next_color, time_ball_state=1)
            clock.tick(60)

        mode = backdoor_mode if backdoor_mode is not None else next_mode

        # EXECUTION
        trig = config.TRIGGERS["MI_BEGIN"] if mode == 0 else config.TRIGGERS["REST_BEGIN"]
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], trig, logger)
        if mode == 0 and FES_toggle: send_udp_message(fes_socket, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_SENS_GO", logger)

        if not show_feedback(config.TIME_MI, mode): break

        # ==============================================================================
        # [NUEVO] FORZAR RELAJACIÓN INMEDIATA (MANDAR '0')
        # ==============================================================================
        # Esto asegura que apenas se quite el cuadro rojo, el guante se abra.
        if arduino_ser and arduino_ser.is_open:
            arduino_ser.write(b'0') 
        # ==============================================================================

        # END TRIAL / ROBOT
        end_trig = config.TRIGGERS["MI_END"] if mode == 0 else config.TRIGGERS["REST_END"]
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], end_trig, logger)

        if mode == 0:
            sel_traj = random.choice(config.ROBOT_TRAJECTORY)
            if FES_toggle: send_udp_message(fes_socket, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_MOTOR_GO", logger)
            send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_BEGIN"], logger)
            display_multiple_messages_with_udp([" "], [config.green], [0], 5, [sel_traj, "g"], udp_socket_robot, config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"], logger)
            send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_END"], logger)
            display_fixation_period(2)
            send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_HOME"], logger)
            send_udp_message(udp_socket_robot, config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"], config.ROBOT_OPCODES["HOME"], logger, expect_ack=True)
        else:
            display_multiple_messages_with_udp([" "], [config.white], [0], config.TIME_STATIONARY, None, udp_socket_robot, config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"], logger)

        # Relajar guante entre trials (AGREGADO)
        if arduino_ser and arduino_ser.is_open:
            arduino_ser.write(b'0')

        display_fixation_period(3)
        current_trial += 1

finally:
    pygame.quit()
    # Cierre seguro del puerto serial (AGREGADO)
    if arduino_ser and arduino_ser.is_open:
        arduino_ser.write(b'0')
        arduino_ser.close()
    [s.close() for s in [udp_socket_marker, udp_socket_robot, fes_socket]]

