"""
ExperimentDriver_Online_Tiagobot_Gaze.py

Gaze-driven variant of ExperimentDriver_Online_Tiagobot.py — same trial
loop, but per-trial letter selection comes from the user's gaze
classified against an on-screen 3x3 letter grid (head-fixed,
eyes-only) instead of random.choice.

Pivoted 2026-05-22 from LSL-resolved physical-board gaze to the
head-fixed on-screen path validated in tools/gaze_to_tiago_test.py.
The physical-board / pylsl 'Gaze' inlet variant is gone — recoverable
via `git revert` of commit feature/tiagobot-gaze-integration HEAD~1.

Flow per trial (changes vs parent in CAPS):
1. RENDER THE 3x3 LETTER GRID + CENTRAL FIXATION CROSS while the
   trial-start countdown ticks (user holds head pointed at cross).
2. RUN A PER-TRIAL CONTINUOUS-DWELL GAZE SELECTION (head-fixed
   on-screen): classify each fresh GazeSystem snapshot via
   Utils.tiagobot_gaze.classify_gaze_to_letter, accumulate dwell on
   the current letter, commit when dwell crosses
   TIAGOBOT_GAZE_DWELL_HIT_SEC, bail after
   TIAGOBOT_GAZE_SELECTION_TIMEOUT_SEC.
3. If classification fails (no centroid within max distance, or no
   valid samples) → log + skip the GO for this trial (no random
   fallback — fail visibly per plan §6.3 step 4).
4. Otherwise write `'{analog},{angle},{delay}\\n'` for the chosen
   letter to the Tiagobot Arduino.
5. (Optional) Write config.ARDUINO_CMD_MI to the glove Arduino.
6. Run hold_messages_and_classify for TIME_ROB seconds.
7. Send `'h\\n'` to Tiagobot — actuator retracts and servo centers.
8. (Optional) Write config.ARDUINO_CMD_REST to the glove Arduino.

The gaze layer is read-only with respect to Utils/tiagobot.py (Tier 1).
All gaze logic lives in Utils/tiagobot_gaze.py (pure functions) and
the inline run_tiago_gaze_selection_window helper below.

Layout source of truth: Utils/tiagobot_gaze.grid_centroids_norm()
(same layout the calibration UI in tiago_gaze_calibration_exec.py uses).
Calibration produced by: tiago_gaze_calibration_exec.py.
"""

import pygame
import socket
import pickle
import datetime
import time
import serial
import sys
import os
import numpy as np
from pylsl import StreamInlet, resolve_stream

# MNE for real-time EEG processing
import mne
mne.set_log_level("WARNING")

from Utils.visualization import (
    draw_arrow_fill,
    draw_ball_fill,
    draw_fixation_cross,
    draw_time_balls,
    draw_progress_bar
)

from Utils.experiment_utils import (
    generate_trial_sequence,
    save_transform,
    load_transform
)

from Utils.EEGStreamState import EEGStreamState

# `send_udp_message` carries the marker / FES triggers; the Tiagobot driver
# does not dial UDP_ROBOT (Tiagobot has no UDP interface). The
# `display_multiple_messages_with_udp` helper is invoked with
# udp_messages=None throughout, so it just runs the feedback UI loop.
from Utils.networking import send_udp_message, display_multiple_messages_with_udp

# Tiagobot serial helpers (Tier 1).
from Utils.tiagobot import (
    open_port as open_tiago_port,
    send_letter as tiago_send_letter,
    send_home as tiago_send_home,
    close_port as tiago_close_port,
    find_tiagobot_port,
    find_glove_port,
    wait_for_completion as tiago_wait_for_completion,
    TARGET_REACHED_MARKER, HOMED_MARKER,
)

# Tiagobot gaze helpers (pure functions; not Tier 1). The
# continuous-dwell selection classifies every snapshot in-loop, so
# `average_gaze_over_window` (the post-window median used by the
# prior fixed-window flow) is no longer needed here.
from Utils.tiagobot_gaze import (
    load_centroids as tiago_gaze_load_centroids,
    classify_gaze_to_letter as tiago_gaze_classify,
    grid_centroids_norm as tiago_grid_centroids_norm,
    LETTERS as TIAGO_LETTERS,
)

# Pupil Labs Neon realtime API wrapper. Headless config (no display, no
# CV, no tracker) — the driver only consumes gaze snapshots.
from Utils.gaze.gaze_system import GazeConfig, GazeSystem

import config

from pathlib import Path
from Utils.logging_manager import LoggerManager

from Utils.runtime_common import (
    log_confusion_matrix_from_trial_summary,
    append_trial_probabilities_to_csv,
    display_fixation_period,
    hold_messages_and_classify,
    show_feedback,
)
import Utils.runtime_common as _RC

from Utils.stream_utils import require_marker_stream

# Initialize experiment logger (auto-detects active run or falls back to Debug)
logger = LoggerManager.auto_detect_from_subject(
    subject=config.TRAINING_SUBJECT,
    base_path=Path(config.DATA_DIR),
    mode="online"
)

# Log experiment configuration snapshot
loggable_fields = [
    "UDP_MARKER", "UDP_FES",
    "ARM_SIDE", "TOTAL_TRIALS", "MAX_REPEATS",
    "TIME_MI", "TIME_ROB", "TIME_STATIONARY",
    "SHAPE_MAX", "SHAPE_MIN",
    "FES_toggle", "FES_CHANNEL", "FES_TIMING_OFFSET",
    "WORKING_DIR", "DATA_DIR", "MODEL_PATH",
    "TRAINING_SUBJECT",
    "MOTOR_CHANNEL_NAMES", "CLASSIFY_WINDOW", "THRESHOLD_MI", "THRESHOLD_REST",
    "RELAXATION_RATIO", "MIN_PREDICTIONS", "SURFACE_LAPLACIAN_TOGGLE",
    "SELECT_MOTOR_CHANNELS", "INTEGRATOR_ALPHA",
    "SHRINKAGE_PARAM_MDM", "SHRINKAGE_PARAM_XGB",
    "LEDOITWOLF", "RECENTERING", "UPDATE_DURING_MOVE",
    # Tiagobot-specific
    "TIAGOBOT_PORT", "TIAGOBOT_BAUD", "TIAGOBOT_TRAJECTORY",
    "TIAGOBOT_USE_GLOVE", "TIAGOBOT_GRIP_HOLD_DURATION",
    "TIAGOBOT_EMPTY_HOLD_DURATION",
    "TIAGOBOT_MODE_REVEAL_DURATION",
    "TIAGOBOT_ANTICIPATION_DURATION",
    "TIAGOBOT_TRIAL_PREP_DURATION",
    "SIMULATION_MODE",
    # Tiagobot gaze-specific
    "TIAGOBOT_GAZE_CALIBRATION_PATH",
    "TIAGOBOT_GAZE_CONFIDENCE_THRESHOLD",
    "TIAGOBOT_GAZE_MAX_DIST_NORM",
    "TIAGOBOT_GAZE_DWELL_HIT_SEC",
    "TIAGOBOT_GAZE_SELECTION_TIMEOUT_SEC",
    "TIAGOBOT_GAZE_CONFIRM_SELECTION_SEC",
    "GAZE_SAMPLE_WIDTH", "GAZE_SAMPLE_HEIGHT",
]
config_log_subset = {
    key: getattr(config, key) for key in loggable_fields if hasattr(config, key)
}
logger.save_config_snapshot(config_log_subset)


eeg_dir = logger.log_base / "eeg"
adaptive_T_path = eeg_dir / "adaptive_T.pkl"

Prev_T, counter, Prev_T_beta, counter_beta = load_transform(adaptive_T_path)
if Prev_T is None:
    counter = 0
if Prev_T_beta is None:
    counter_beta = 0
if Prev_T is None and Prev_T_beta is None:
    logger.log_event("ℹ️ No adaptive transforms — cold start for μ and β.")
else:
    parts = []
    if Prev_T is not None:
        parts.append(f"μ counter={counter}")
    if Prev_T_beta is not None:
        parts.append(f"β counter={counter_beta}")
    logger.log_event("✅ Loaded adaptive transform: " + ", ".join(parts))

logger.log_event("Logger initialized for Tiagobot online experimental driver.")


pygame.init()

if config.BIG_BROTHER_MODE:
    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
    screen = pygame.display.set_mode((1920, 1080), pygame.NOFRAME)
    logger.log_event("🎥 Big Brother Mode ON — window placed at (0,0) on external monitor (HDMI-1).")
else:
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    logger.log_event("👤 Big Brother Mode OFF — fullscreen on active display.")

pygame.display.set_caption("EEG Online Interactive Loop — Tiagobot")
info = pygame.display.Info()
screen_width = info.current_w
screen_height = info.current_h
logger.log_event("Pygame initialized and display configured.")

# UDP sockets. udp_socket_robot is created for parity with runtime_common
# helpers that take it as a parameter; the Tiagobot driver never dials a
# robot UDP command, so writes through it land in a dead endpoint and are
# effectively no-ops.
udp_socket_marker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket_robot  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket_fes    = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
logger.log_event("UDP sockets initialized for marker, robot (unused), and FES channels.")

FES_toggle = config.FES_toggle
logger.log_event(f"FES toggle status: {'Enabled' if FES_toggle else 'Disabled'}.")

# Decoder model
subject_model_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "models")
decoder_backend = str(getattr(config, "DECODER_BACKEND", "mdm")).lower()
if decoder_backend == "xgb_cov":
    model_filename = f"sub-{config.TRAINING_SUBJECT}_xgb_cov_features.pkl"
elif decoder_backend == "xgb_cov_erd":
    model_filename = f"sub-{config.TRAINING_SUBJECT}_xgb_cov_erd_features.pkl"
else:
    model_filename = f"sub-{config.TRAINING_SUBJECT}_model.pkl"
subject_model_path = os.path.join(subject_model_dir, model_filename)

try:
    with open(subject_model_path, 'rb') as f:
        model = pickle.load(f)
    logger.log_event(f"✅ Model successfully loaded from: {subject_model_path}")
except FileNotFoundError:
    logger.log_event(f"❌ Error: Model file '{subject_model_path}' not found. Ensure the model has been trained.", level="error")
    sys.exit(1)


# ============================================================
# TIAGOBOT SETUP (mandatory unless SIMULATION_MODE)
# ============================================================
# Resolve the port: explicit config first, fall back to USB-ID scan
# (Arduino Mega 2560 R3, vid:pid 2341:0042). This lets the driver run
# with no manual config when only Tiagobot is plugged in (or with both
# Arduinos as long as the glove isn't also a Mega 2560).
tiago_port = config.TIAGOBOT_PORT
if not tiago_port:
    logger.log_event("TIAGOBOT_PORT is empty — scanning USB for a Mega 2560 R3...")
    tiago_port = find_tiagobot_port(logger=logger)
    if tiago_port:
        logger.log_event(f"Auto-detected Tiagobot port: {tiago_port}")

# open_tiago_port returns None in SIMULATION_MODE or when port is unset.
# Real open failures raise; we let them propagate per fail-fast policy on
# Tier 1 hardware paths (CLAUDE.md "Error Handling").
tiago = open_tiago_port(tiago_port, config.TIAGOBOT_BAUD, logger)
if tiago is None and not config.SIMULATION_MODE:
    logger.log_event(
        "❌ Tiagobot port unavailable and SIMULATION_MODE is False — aborting.",
        level="error",
    )
    sys.exit(1)


# ============================================================
# GLOVE SETUP (optional)
# ============================================================
arduino = None
if config.TIAGOBOT_USE_GLOVE:
    glove_port = config.ARDUINO_PORT
    if not glove_port:
        logger.log_event("ARDUINO_PORT is empty — scanning USB for an Arduino Uno R3 (glove)...")
        glove_port = find_glove_port(logger=logger)
        if glove_port:
            logger.log_event(f"Auto-detected glove port: {glove_port}")

    if glove_port:
        try:
            logger.log_event(f"Connecting to Glove (Arduino) on {glove_port}...")
            arduino = serial.Serial(glove_port, config.ARDUINO_BAUD, timeout=0.1)
            time.sleep(2)  # Arduino reset wait
            logger.log_event("✅ Glove connected successfully.")
        except Exception as e:
            logger.log_event(f"❌ Error connecting to Glove: {e}", level="error")
            arduino = None
    else:
        logger.log_event("ℹ️ TIAGOBOT_USE_GLOVE=True but no glove port available — glove disabled.")
else:
    logger.log_event("ℹ️ Glove integration disabled (TIAGOBOT_USE_GLOVE=False).")


# ============================================================
# TIAGOBOT GAZE CALIBRATION (mandatory)
# ============================================================
# Load the per-letter centroids produced by tiago_gaze_calibration_exec.py.
# Fail-fast at startup if missing or malformed — per CLAUDE.md, silently
# falling back to (a) random letter selection or (b) nominal grid centroids
# would mask a real configuration error during a hardware session.
_gaze_cal_path = str(getattr(config, "TIAGOBOT_GAZE_CALIBRATION_PATH", "") or "")
if not _gaze_cal_path:
    logger.log_event(
        "❌ TIAGOBOT_GAZE_CALIBRATION_PATH is empty. Set it in "
        "config_local.py (path to the NPZ from "
        "tiago_gaze_calibration_exec.py) before launching this driver.",
        level="error",
    )
    sys.exit(1)
try:
    _tiago_centroids = tiago_gaze_load_centroids(_gaze_cal_path)
except (FileNotFoundError, ValueError) as e:
    logger.log_event(
        f"❌ Failed to load Tiagobot gaze calibration from "
        f"{_gaze_cal_path}: {e}",
        level="error",
    )
    sys.exit(1)
logger.log_event(
    f"✅ Loaded Tiagobot gaze centroids from {_gaze_cal_path}: "
    f"{len(_tiago_centroids)}/{len(TIAGO_LETTERS)} letters with valid data."
)
_missing_letters = [ch for ch in TIAGO_LETTERS if ch not in _tiago_centroids]
if _missing_letters:
    logger.log_event(
        f"⚠️ Calibration missing letters: {_missing_letters}. Trials whose "
        f"gaze maps closest to these will return None (skip GO)."
    )


# ============================================================
# ON-SCREEN GAZE CONSTANTS
# ============================================================
# Scene-camera pixel scale (Neon constants — same shape used by the
# calibration script and tools/gaze_to_tiago_test.py). Normalized
# (gaze_px / sample_dim) gives the (x, y) the calibration centroids
# live in.
TIAGO_GAZE_SAMPLE_W = float(getattr(config, "GAZE_SAMPLE_WIDTH", 1600.0))
TIAGO_GAZE_SAMPLE_H = float(getattr(config, "GAZE_SAMPLE_HEIGHT", 1200.0))

# Cap the gaze-selection snapshot loop so EEG / pygame stay responsive
# during the 4 s window. Matches tools/gaze_to_tiago_test.py.
TIAGO_GAZE_SNAPSHOT_POLL_HZ = 60.0

# Continuous-dwell selection parameters. Resolved once at module load
# and cached as module-level floats so the realtime loop doesn't pay
# a getattr() per frame.
TIAGO_GAZE_DWELL_HIT_SEC = float(getattr(config, "TIAGOBOT_GAZE_DWELL_HIT_SEC", 2.0))
TIAGO_GAZE_TIMEOUT_SEC = float(getattr(config, "TIAGOBOT_GAZE_SELECTION_TIMEOUT_SEC", 12.0))
TIAGO_GAZE_CONFIRM_SEC = float(getattr(config, "TIAGOBOT_GAZE_CONFIRM_SELECTION_SEC", 1.5))
TIAGO_GAZE_CONF_THRESHOLD = float(getattr(config, "TIAGOBOT_GAZE_CONFIDENCE_THRESHOLD", 0.7))
TIAGO_GAZE_MAX_DIST_NORM = (
    float(config.TIAGOBOT_GAZE_MAX_DIST_NORM)
    if getattr(config, "TIAGOBOT_GAZE_MAX_DIST_NORM", None) is not None
    else None
)
# If no fresh, worn, finite snapshot arrives for this long, treat the
# gaze as lost and reset the dwell counter — a brief tracker hiccup
# (sub-frame) shouldn't reset, but a sustained dropout (glasses off,
# blink longer than half a second) should.
TIAGO_GAZE_STALE_GAP_SEC = 0.5

# Render layout for the 3x3 grid — must match the centroids the
# calibration NPZ was captured against (same call as
# tiago_gaze_calibration_exec.py uses, so on-screen letter positions
# align with the calibrated scene-pixel centroids).
TIAGO_GRID_RENDER_CENTROIDS = tiago_grid_centroids_norm()


logger.log_event("finding training dataset . . .")
eeg_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "training_data")
logger.log_event(f"Script is looking for XDF files in: {eeg_dir}")

xdf_files = [
    os.path.join(eeg_dir, f) for f in os.listdir(eeg_dir)
    if f.endswith(".xdf") and "OBS" not in f
]

if not xdf_files:
    raise FileNotFoundError(f"No XDF files found in: {eeg_dir}")
logger.log_event(f"training data: {xdf_files}")


predictions_list = []
ground_truth_list = []

fs = config.FS

SESSION_TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logger.log_event(f"Session timestamp set: {SESSION_TIMESTAMP}")

# Wire runtime objects into runtime_common globals
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
_RC.Prev_T_beta = Prev_T_beta
_RC.counter_beta = counter_beta


# GazeSystem handle — populated in main() once Neon is connected, read
# back in _hardware_cleanup so the finally-block can stop it.
_tiago_gaze_system = None


def _hardware_cleanup():
    """Best-effort release of the Tiagobot serial port, the glove port
    (if present), and the Neon GazeSystem. Runs on every exit path of
    main() — including SystemExit from runtime_common when the operator
    closes the pygame window mid-session."""
    try:
        tiago_close_port(tiago, logger)
    except Exception as e:
        logger.log_event(f"Tiagobot close error: {e}", level="error")
    if arduino is not None:
        try:
            arduino.close()
        except Exception as e:
            logger.log_event(f"Glove close error: {e}", level="error")
    if _tiago_gaze_system is not None:
        try:
            _tiago_gaze_system.stop()
        except Exception as e:
            logger.log_event(f"GazeSystem stop error: {e}", level="error")
    try:
        pygame.quit()
    except Exception:
        pass


def _tiago_draw_grid_with_cross(countdown_text=None):
    """Render the 3x3 A-I grid + central fixation cross on `screen`.

    Mirrors tools/gaze_to_tiago_test.py:_draw_grid_with_cross — same
    fonts, same colors, same per-letter normalized centroids. The
    runtime layout must align byte-for-byte with the calibration UI
    (tiago_gaze_calibration_exec._draw_grid) so the scene-pixel
    centroids the NPZ stores still correspond to the rendered letter
    positions at runtime.

    `countdown_text` is an optional bottom-of-screen status line.
    """
    screen.fill(config.black)

    font = pygame.font.SysFont(None, 96)
    for ch in TIAGO_LETTERS:
        cx_n, cy_n = TIAGO_GRID_RENDER_CENTROIDS[ch]
        cx, cy = int(cx_n * screen_width), int(cy_n * screen_height)
        surf = font.render(ch, True, (180, 180, 180))
        rect = surf.get_rect(center=(cx, cy))
        screen.blit(surf, rect)

    # Central yellow-green cross — head-pose anchor. Matches the
    # calibration script's cross (color + 18 px arm + 3 px stroke).
    cx, cy = screen_width // 2, screen_height // 2
    arm = 18
    cross_color = (200, 200, 80)
    pygame.draw.line(screen, cross_color, (cx - arm, cy), (cx + arm, cy), 3)
    pygame.draw.line(screen, cross_color, (cx, cy - arm), (cx, cy + arm), 3)

    if countdown_text:
        font_cd = pygame.font.SysFont(None, 40)
        surf = font_cd.render(countdown_text, True, (200, 200, 200))
        rect = surf.get_rect(center=(screen_width // 2, screen_height - 40))
        screen.blit(surf, rect)

    pygame.display.flip()


def _tiago_anticipation_fixation_period(duration, eeg_state, message, target_letter=None):
    """Inter-trial anticipation fixation: render the central cross plus
    a *white-filled* small orb above it that fills linearly from 0 → 1
    over `duration` seconds, with `message` rendered below the cross.

    The orb's position matches Utils.visualization.draw_time_balls's
    "single" mode geometry (above the cross), since the timing orb in
    that position is already the patient's familiar cue for "something
    is about to switch". draw_time_balls itself only supports four
    discrete states (empty / white / red / blue), so we render a
    smooth fill manually at the same position rather than toggle a
    state mid-period — closer to the "fill up" cue the protocol
    needs. Fill colour stays white (matches state 1 of the existing
    timing orb) so the visual language is consistent.

    If `target_letter` is provided, it is rendered in a large font
    further below the instruction text — the prescribed gaze target
    for the next trial, so the patient can mentally pre-plan which
    letter to look at when the Phase 2 selection grid appears. The
    patient should still keep their eyes on the cross during the
    anticipation period; the target is a planning cue, not a gaze
    target during this period.

    Replaces `display_fixation_period` for the trial-wrap and initial
    fixation calls; runs the same EEG-state + QUIT-handling loop so
    timing semantics are preserved.
    """
    # Geometry matches Utils.visualization.draw_time_balls "single"
    # mode — small orb centred horizontally and offset upward above
    # the fixation cross.
    ball_radius = 60
    ball_x = screen_width // 2
    ball_y = screen_height // 2 - ball_radius * 4

    font = pygame.font.SysFont(None, 48)
    target_font = pygame.font.SysFont(None, 240) if target_letter is not None else None
    start_time = time.time()
    clock = pygame.time.Clock()

    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        progress = min(1.0, max(0.0, elapsed / float(duration)))

        screen.fill(config.black)
        draw_fixation_cross(screen_width, screen_height)

        # White orb outline
        pygame.draw.circle(
            screen, (255, 255, 255), (ball_x, ball_y), ball_radius, 2
        )
        # White fill rising from the bottom of the orb, masked to a
        # circle so it stays inside the outline.
        fill_height = int(progress * ball_radius * 2)
        fill_surface = pygame.Surface(
            (ball_radius * 2, ball_radius * 2), pygame.SRCALPHA
        )
        fill_rect = pygame.Rect(
            0, ball_radius * 2 - fill_height, ball_radius * 2, fill_height
        )
        pygame.draw.rect(fill_surface, (255, 255, 255, 200), fill_rect)
        mask_surface = pygame.Surface(
            (ball_radius * 2, ball_radius * 2), pygame.SRCALPHA
        )
        pygame.draw.circle(
            mask_surface,
            (255, 255, 255, 255),
            (ball_radius, ball_radius),
            ball_radius,
        )
        fill_surface.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        screen.blit(
            fill_surface, (ball_x - ball_radius, ball_y - ball_radius)
        )

        # Instruction text below the cross
        msg_surf = font.render(message, True, (200, 200, 200))
        msg_rect = msg_surf.get_rect(
            center=(screen_width // 2, screen_height // 2 + 120)
        )
        screen.blit(msg_surf, msg_rect)

        # Prescribed gaze target for the next trial. Large + green so
        # it's unmistakable; placed well below the instruction so it
        # doesn't compete with the "Look at the fixation cross" cue.
        if target_letter is not None:
            tgt_surf = target_font.render(target_letter, True, config.green)
            tgt_rect = tgt_surf.get_rect(
                center=(screen_width // 2, screen_height // 2 + 280)
            )
            screen.blit(tgt_surf, tgt_rect)

        pygame.display.flip()

        if eeg_state is not None:
            eeg_state.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
        clock.tick(60)


def _tiago_draw_message_screen(messages, colors, offsets):
    """Render `messages` centred on `screen` and flip the display once.

    No EEG pump, no event loop — pure one-shot render. Use this when
    the caller needs the message to persist while ANOTHER blocking
    call (e.g. `wait_for_completion(HOMED, on_tick=_drive_tick)`) pumps
    the loop, since `_drive_tick` flips the buffer without redrawing
    and pygame's software display keeps the last-blitted `screen`
    contents on subsequent flips.
    """
    screen.fill(config.black)
    font = pygame.font.SysFont(None, 72)
    for i, text in enumerate(messages):
        surf = font.render(text, True, colors[i])
        rect = surf.get_rect(
            center=(screen_width // 2, screen_height // 2 + offsets[i])
        )
        screen.blit(surf, rect)
    pygame.display.flip()


def _tiago_hold_message_screen(messages, colors, offsets, duration, eeg_state):
    """Render `messages` (via `_tiago_draw_message_screen`) and hold them
    for `duration` seconds while pumping EEG state and pygame events.
    Equivalent in structure to `display_fixation_period`, but renders a
    custom multi-line message instead of the idle fixation UI.
    """
    _tiago_draw_message_screen(messages, colors, offsets)
    end_time = time.time() + float(duration)
    clock = pygame.time.Clock()
    while time.time() < end_time:
        if eeg_state is not None:
            eeg_state.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
        clock.tick(60)


def _tiago_read_latest_valid_snapshot(gs, last_unix_t):
    """Pull one fresh, worn, finite gaze sample from `gs`.

    Returns `((t_unix, x_norm, y_norm, conf_or_worn, depth_cm), new_last_t)`
    on success, `(None, new_last_t)` if the snapshot is stale, the
    glasses are off, the gaze pixel is non-finite, or no snapshot is
    available yet. `conf_or_worn` is `1.0` (treat-as-confidence for
    Utils.tiagobot_gaze.average_gaze_over_window — the realtime API
    does not expose per-sample confidence; we filter on the worn flag
    instead, matching the calibration UI's contract). `depth_cm` is
    NaN when vergence is invalid.

    Identical dedupe/filter contract to
    tools/gaze_to_tiago_test.py:_read_latest_valid.
    """
    snap = gs.get_snapshot(include_objects=False, include_frame=False)
    if not snap or not snap.get("ok"):
        return None, last_unix_t
    t = snap.get("unix_t")
    if t is None:
        return None, last_unix_t
    if last_unix_t is not None and t <= last_unix_t:
        return None, last_unix_t
    if not bool(snap.get("worn")):
        return None, float(t)
    px = snap.get("gaze_px_raw")
    if px is None:
        return None, float(t)
    x_px, y_px = float(px[0]), float(px[1])
    if not (np.isfinite(x_px) and np.isfinite(y_px)):
        return None, float(t)
    depth_cm = float("nan")
    if bool(snap.get("depth_valid")):
        d = snap.get("depth_cm")
        if d is not None:
            df = float(d)
            if np.isfinite(df):
                depth_cm = df
    return (
        float(t),
        x_px / TIAGO_GAZE_SAMPLE_W,
        y_px / TIAGO_GAZE_SAMPLE_H,
        1.0,
        depth_cm,
    ), float(t)


def _tiago_draw_progress_bar(progress, y, fill_color):
    """Top-pinned countdown bar used by the selection screen. Same
    proportions as Harmony's selection bar (55% screen width, 28 px
    tall, white outline)."""
    bar_w = int(screen_width * 0.55)
    bar_x = (screen_width - bar_w) // 2
    bar_h = 28
    outer = pygame.Rect(bar_x, y, bar_w, bar_h)
    inner = pygame.Rect(bar_x, y, int(bar_w * max(0.0, min(1.0, progress))), bar_h)
    pygame.draw.rect(screen, (60, 60, 60), outer)
    pygame.draw.rect(screen, fill_color, inner)
    pygame.draw.rect(screen, config.white, outer, 2)


def _tiago_draw_centered_text(text, y, color, font_size):
    font = pygame.font.SysFont(None, font_size)
    surf = font.render(text, True, color)
    rect = surf.get_rect(center=(screen_width // 2, y))
    screen.blit(surf, rect)


def _tiago_draw_letter_grid(highlight_letter, available_set):
    """Render the 3x3 letter grid at the calibration-aligned nominal
    centroids. `highlight_letter` (if not None) is drawn larger and in
    config.green; letters not in `available_set` are dimmed so the
    user can see at a glance which ones are eligible
    (config.TIAGOBOT_TRAJECTORY).

    Cell centers are unchanged from `_tiago_draw_grid_with_cross` — the
    calibration NPZ centroids correspond to these positions, so visual
    size and color have no effect on classification."""
    for ch in TIAGO_LETTERS:
        cx_n, cy_n = TIAGO_GRID_RENDER_CENTROIDS[ch]
        cx, cy = int(cx_n * screen_width), int(cy_n * screen_height)
        if ch == highlight_letter:
            color = config.green
            font_size = 132
        elif ch in available_set:
            color = (180, 180, 180)
            font_size = 96
        else:
            color = (90, 90, 90)
            font_size = 96
        font = pygame.font.SysFont(None, font_size)
        surf = font.render(ch, True, color)
        rect = surf.get_rect(center=(cx, cy))
        screen.blit(surf, rect)


def _tiago_draw_fixation_cross():
    """Yellow-green head-pose anchor cross. Same color (200, 200, 80) /
    18 px arm / 3 px stroke as `_tiago_draw_grid_with_cross` so the
    head-fixed protocol's visual anchor is preserved during the dwell
    UI."""
    cx, cy = screen_width // 2, screen_height // 2
    arm = 18
    pygame.draw.line(screen, (200, 200, 80), (cx - arm, cy), (cx + arm, cy), 3)
    pygame.draw.line(screen, (200, 200, 80), (cx, cy - arm), (cx, cy + arm), 3)


def _tiago_draw_selection_screen(progress, dwell_sec, current_letter,
                                  available_set):
    """Live continuous-dwell screen.

      top:    progress bar + "Hold gaze: X.Xs / Y.Ys" label
      center: 3x3 letter grid (current letter highlighted, off-trajectory
              letters dimmed) + central yellow-green fixation cross
      bottom: "Looking at: <letter>" or "Looking at: —"
    """
    screen.fill(config.black)

    bar_y = 60
    fill_color = config.green if current_letter is not None else (180, 180, 180)
    _tiago_draw_progress_bar(progress, y=bar_y, fill_color=fill_color)
    _tiago_draw_centered_text(
        f"Hold gaze: {dwell_sec:.1f}s / {TIAGO_GAZE_DWELL_HIT_SEC:.1f}s",
        y=bar_y + 50, color=config.white, font_size=32,
    )

    _tiago_draw_letter_grid(current_letter, available_set)
    _tiago_draw_fixation_cross()

    if current_letter is not None:
        _tiago_draw_centered_text(
            f"Looking at: {current_letter}",
            y=screen_height - 100, color=config.green, font_size=64,
        )
    else:
        _tiago_draw_centered_text(
            "Looking at: —",
            y=screen_height - 100, color=config.orange, font_size=64,
        )

    pygame.display.flip()


def _tiago_draw_selection_confirmation(letter, available_set):
    """Held screen shown after a successful selection. Same grid +
    cross layout as the live selection screen so the head-fixed anchor
    is unchanged; a "Selected:" header at the top and a large letter
    glyph at the bottom make the choice unmistakable."""
    screen.fill(config.black)
    _tiago_draw_centered_text("Selected:", y=80, color=config.white, font_size=64)
    _tiago_draw_letter_grid(highlight_letter=letter, available_set=available_set)
    _tiago_draw_fixation_cross()
    _tiago_draw_centered_text(
        str(letter), y=screen_height - 100, color=config.green, font_size=140,
    )
    pygame.display.flip()


def hold_tiago_selection_confirmation(letter, eeg_state, available_set):
    """Hold the "Selected: <letter>" screen for TIAGO_GAZE_CONFIRM_SEC
    so the subject sees what was chosen before Phase 3 (mode reveal)
    begins. Returns False if the operator quits during the hold."""
    end_t = time.monotonic() + TIAGO_GAZE_CONFIRM_SEC
    while time.monotonic() < end_t:
        eeg_state.update()
        _tiago_draw_selection_confirmation(letter, available_set)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
    return True


def run_tiago_gaze_selection_window(gs, eeg_state, centroids, available_letters):
    """Continuous-dwell letter selection.

    Each loop iteration we pull one fresh Neon snapshot via
    `_tiago_read_latest_valid_snapshot`, classify its
    `(x_norm, y_norm)` against `centroids` restricted to
    `available_letters` with the TIAGO_GAZE_MAX_DIST_NORM gate, and
    update the continuous-dwell counter:

    - sample on the same letter as the previous "current letter" →
      accumulate dt onto the dwell counter
    - sample on a different letter → restart the counter at dt
    - sample off-grid (no centroid within the distance gate) → reset
      to 0 and clear the current letter
    - no fresh sample for more than TIAGO_GAZE_STALE_GAP_SEC →
      reset to 0 (treats as tracking loss)

    Exits the moment continuous dwell crosses TIAGO_GAZE_DWELL_HIT_SEC;
    bails after TIAGO_GAZE_TIMEOUT_SEC with no hit. The depth axis is
    disabled (`gaze_depth_cm=None`) to match the pre-pivot 2D mode
    used downstream in main() — the head-fixed (x, y) signal is
    well-separated and depth adds only noise here.

    Returns None on operator quit; otherwise a dict with:
      selected_letter (str or None on timeout),
      continuous_dwell_sec (float),
      samples (list[(t_unix, x_norm, y_norm, 1.0, depth_cm)]),
      samples_used (int),
      selection_attempt_success (bool).
    """
    logger.log_event(
        f"Starting Tiagobot continuous-dwell selection — "
        f"hit_sec={TIAGO_GAZE_DWELL_HIT_SEC:.2f}, "
        f"timeout_sec={TIAGO_GAZE_TIMEOUT_SEC:.2f}, "
        f"available={list(available_letters)}"
    )

    available_set = set(available_letters)
    current_letter = None
    continuous_dwell_sec = 0.0
    samples_on_current = []
    last_unix_t = None
    last_good_sample_mono = None

    poll_s = 1.0 / TIAGO_GAZE_SNAPSHOT_POLL_HZ
    selection_start = time.monotonic()
    last_loop_t = selection_start

    while True:
        eeg_state.update()

        now = time.monotonic()
        dt = max(0.0, now - last_loop_t)
        last_loop_t = now
        elapsed_total = now - selection_start

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return None

        # _tiago_read_latest_valid_snapshot returns the next fresh +
        # worn + finite gaze pixel (or None if stale / glasses off /
        # not yet available). It already filters bad samples for us.
        s, last_unix_t = _tiago_read_latest_valid_snapshot(gs, last_unix_t)

        if s is not None:
            t_unix, x_norm, y_norm, _conf, depth_cm = s
            letter = tiago_gaze_classify(
                x_norm, y_norm, centroids,
                gaze_depth_cm=None,  # 2D mode — matches Phase 2 downstream
                available_letters=available_letters,
                max_dist_norm=TIAGO_GAZE_MAX_DIST_NORM,
            )
            last_good_sample_mono = now

            if letter is None:
                # Off-grid (no centroid within the distance gate) →
                # treat as "no letter", reset the dwell counter.
                current_letter = None
                continuous_dwell_sec = 0.0
                samples_on_current = []
            elif letter == current_letter:
                continuous_dwell_sec += dt
                samples_on_current.append(s)
            else:
                # Switched to a new letter — restart the counter so the
                # subject has to hold the new target for the full
                # TIAGO_GAZE_DWELL_HIT_SEC.
                current_letter = letter
                continuous_dwell_sec = dt
                samples_on_current = [s]
        else:
            # No fresh sample this iteration. Reset only if the gap
            # exceeds TIAGO_GAZE_STALE_GAP_SEC (sustained dropout —
            # glasses off, long blink). For short gaps with a held
            # letter, advance the dwell counter by wall-clock dt so
            # the progress bar tracks real time even when Neon's
            # effective publish rate is slower than our 60 Hz poll;
            # otherwise stale-poll iterations drop their dt and the
            # bar lags visibly behind actual elapsed dwell.
            if (last_good_sample_mono is not None
                    and (now - last_good_sample_mono) > TIAGO_GAZE_STALE_GAP_SEC):
                current_letter = None
                continuous_dwell_sec = 0.0
                samples_on_current = []
                last_good_sample_mono = None
            elif current_letter is not None:
                continuous_dwell_sec += dt

        progress = (continuous_dwell_sec / TIAGO_GAZE_DWELL_HIT_SEC
                    if TIAGO_GAZE_DWELL_HIT_SEC > 0 else 1.0)
        _tiago_draw_selection_screen(
            progress=progress,
            dwell_sec=continuous_dwell_sec,
            current_letter=current_letter,
            available_set=available_set,
        )

        if (current_letter is not None
                and continuous_dwell_sec >= TIAGO_GAZE_DWELL_HIT_SEC
                and len(samples_on_current) > 0):
            logger.log_event(
                f"Tiagobot gaze hit — letter={current_letter}, "
                f"continuous_dwell={continuous_dwell_sec:.3f}s, "
                f"samples={len(samples_on_current)}"
            )
            return {
                "selected_letter": current_letter,
                "continuous_dwell_sec": continuous_dwell_sec,
                "samples": samples_on_current,
                "samples_used": len(samples_on_current),
                "selection_attempt_success": True,
            }

        if elapsed_total >= TIAGO_GAZE_TIMEOUT_SEC:
            logger.log_event(
                f"Tiagobot gaze selection timed out after {elapsed_total:.1f}s "
                f"with no {TIAGO_GAZE_DWELL_HIT_SEC:.1f}s continuous-dwell hit."
            )
            return {
                "selected_letter": None,
                "continuous_dwell_sec": 0.0,
                "samples": [],
                "samples_used": 0,
                "selection_attempt_success": False,
            }

        time.sleep(poll_s)


def main():
    """Run the online MI/REST trial loop for the gaze-driven Tiagobot.

    Per trial: continuous-dwell gaze selection commits a letter when
    the subject holds gaze on it for TIAGOBOT_GAZE_DWELL_HIT_SEC; on
    MI-correct trials Tiagobot extends to that letter, the
    classification window runs, then Tiagobot is sent HOME. Glove
    (if enabled) closes on MI and opens on HOME. If the dwell times
    out without a hit on an MI trial, Phase 2.5 aborts the trial
    (no GO) and advances the counter.
    """
    require_marker_stream(logger=logger)

    logger.log_event("Resolving EEG data stream via LSL...")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    logger.log_event("✅ EEG stream detected and inlet established.")

    # Pupil Labs Neon via the realtime API (no LSL outlet on the phone).
    # One GazeSystem instance per session — start here, stop in
    # _hardware_cleanup. Headless flags: the driver only reads gaze
    # snapshots, no scene CV / display / object tracker needed.
    global _tiago_gaze_system
    neon_host = str(getattr(config, "NEON_COMPANION_HOST", "") or "")
    logger.log_event(
        f"Starting GazeSystem (Neon Companion host={neon_host!r}, mDNS if empty)..."
    )
    _tiago_gaze_system = GazeSystem(GazeConfig(
        enable_prints=False, enable_display=False, enable_cv=False,
        use_tracker=False, neon_host=neon_host,
    ))
    _tiago_gaze_system.start()
    logger.log_event("✅ GazeSystem started.")

    eeg_state = EEGStreamState(inlet=inlet, config=config, logger=logger)
    logger.log_event("EEGStreamState object created — ready to pull and process data.")

    trial_sequence = generate_trial_sequence(total_trials=config.TOTAL_TRIALS, max_repeats=config.MAX_REPEATS)
    mode_labels = ["MI" if t == 0 else "REST" for t in trial_sequence]
    logger.log_event(f"Trial Sequence generated: {trial_sequence}")
    logger.log_event(f"Trial Sequence (labeled): {mode_labels}")

    # Per-trial prescribed gaze target — shown during the anticipation
    # fixation so the patient can pre-plan which letter to look at. The
    # pool is the configured trajectory intersected with the calibrated
    # letters, minus any letter listed in TIAGOBOT_GAZE_TARGET_EXCLUDE
    # (operator-controlled from the control panel — used to drop letters
    # that are reachable+calibrated but too hard for the patient to
    # acquire reliably). The gaze classifier itself is unchanged: the
    # patient is free to select any letter; both target and actual
    # selection are logged so offline analysis can measure target-match
    # rate.
    _target_exclude = {
        str(ch).upper()
        for ch in (getattr(config, "TIAGOBOT_GAZE_TARGET_EXCLUDE", []) or [])
    }
    _target_pool = [
        ch for ch in config.TIAGOBOT_TRAJECTORY
        if ch in _tiago_centroids and ch not in _target_exclude
    ]
    if not _target_pool:
        logger.log_event(
            "❌ Empty per-trial target pool: TIAGOBOT_TRAJECTORY ∩ "
            "calibration ∖ TIAGOBOT_GAZE_TARGET_EXCLUDE is empty. "
            "Check that the calibration covers the trajectory letters "
            "and that TIAGOBOT_GAZE_TARGET_EXCLUDE does not remove "
            "every remaining letter.",
            level="error",
        )
        sys.exit(1)
    if _target_exclude:
        logger.log_event(
            f"Target exclude list active — excluded={sorted(_target_exclude)}, "
            f"remaining pool={_target_pool}"
        )
    target_letter_sequence = [
        str(np.random.choice(_target_pool)) for _ in range(len(trial_sequence))
    ]
    logger.log_event(f"Target letter sequence: {target_letter_sequence}")

    current_trial = 0

    all_results = []
    running = True
    clock = pygame.time.Clock()

    _tiago_anticipation_fixation_period(
        duration=float(getattr(config, "TIAGOBOT_ANTICIPATION_DURATION", 3.0)),
        eeg_state=eeg_state,
        message="Look at the fixation cross",
        target_letter=target_letter_sequence[0],
    )
    logger.log_event("Initial anticipation fixation complete. Beginning experimental loop.")

    while running and current_trial < len(trial_sequence):
        logger.log_event(f"--- Trial {current_trial+1}/{len(trial_sequence)} START ---")

        # === Phase 1: grid + cross prep hold ===
        # Brief static hold on the gaze-grid screen — gives the patient
        # a moment to register the letters and pick a target with their
        # eyes before continuous-dwell begins. No on-screen countdown
        # here: the inter-trial anticipation fixation (cross + filling
        # white orb) already gave the timing cue. Backdoor RIGHT/DOWN
        # and SPACE-skip remain available silently for the operator.
        backdoor_mode = None
        waiting_for_press = True
        countdown_start = pygame.time.get_ticks() if config.TIMING else None
        prep_duration_s = float(getattr(config, "TIAGOBOT_TRIAL_PREP_DURATION", 2.0))
        countdown_duration = int(prep_duration_s * 1000)
        phase1_clock = pygame.time.Clock()

        # Single up-front render — no per-iter redraw needed because the
        # frame is static. Operator-mode (TIMING=False) keeps the SPACE
        # affordance visible; auto-paced mode shows the grid + cross
        # only.
        if config.TIMING:
            _tiago_draw_grid_with_cross(countdown_text=None)
            logger.log_event(f"Phase 1 trial-prep hold: {prep_duration_s:.1f} s")
        else:
            _tiago_draw_grid_with_cross(countdown_text="Press SPACE to start trial")

        while waiting_for_press:
            eeg_state.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    waiting_for_press = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        backdoor_mode = 0  # Force MI
                    elif event.key == pygame.K_DOWN:
                        backdoor_mode = 1  # Force REST
                    elif event.key == pygame.K_SPACE:
                        logger.log_event("Space bar pressed — proceeding without override.")
                    waiting_for_press = False

            if countdown_start is not None:
                elapsed_ms = pygame.time.get_ticks() - countdown_start
                if elapsed_ms >= countdown_duration:
                    waiting_for_press = False

            phase1_clock.tick(60)

        if not running:
            logger.log_event("Experiment terminated early via quit event.")
            break

        # === Trial mode selection ===
        # Selected BEFORE Phase 2 so the gaze window still runs for
        # both modes (uniform trial pacing), but the mode-reveal UI in
        # Phase 3 can be primed with the right cue color.
        if backdoor_mode is not None:
            mode = backdoor_mode
            logger.log_event(f"Backdoor override activated: {'MI' if mode == 0 else 'REST'}")
        else:
            mode = trial_sequence[current_trial]
            logger.log_event(f"Trial mode selected from sequence: {'MI' if mode == 0 else 'REST'}")

        # === Phase 2: Gaze selection window (every trial) ===
        # Runs on every trial so the user's behaviour is identical for
        # MI and Rest — they always pick a letter, the driver only
        # consumes it on MI-correct trials. Per brief 2026-05-22: a
        # uniform Phase 2 keeps trial timing predictable and removes
        # the "did this trial have a gaze window?" cue.
        #
        # Continuous-dwell semantics (2026-05-23): the subject must
        # hold their gaze on a single letter for
        # TIAGOBOT_GAZE_DWELL_HIT_SEC; switching letters, looking
        # off-grid, or losing tracking resets the counter. The window
        # exits as soon as the threshold is crossed (responsive UX)
        # or bails after TIAGOBOT_GAZE_SELECTION_TIMEOUT_SEC with no
        # hit (Phase 2.5 then aborts the MI trial).
        _gaze_result = run_tiago_gaze_selection_window(
            gs=_tiago_gaze_system,
            eeg_state=eeg_state,
            centroids=_tiago_centroids,
            available_letters=list(config.TIAGOBOT_TRAJECTORY),
        )
        if _gaze_result is None:
            # Operator quit / ESC during the window.
            running = False
            logger.log_event("Experiment terminated during gaze selection window.")
            break

        _gaze_selected_letter = _gaze_result["selected_letter"]
        _trial_target = target_letter_sequence[current_trial]
        _target_match = (_gaze_selected_letter == _trial_target)
        logger.log_event(
            f"Gaze selection — letter={_gaze_selected_letter!r}, "
            f"target={_trial_target!r}, match={_target_match}, "
            f"continuous_dwell={_gaze_result['continuous_dwell_sec']:.3f}s, "
            f"samples_used={_gaze_result['samples_used']}, "
            f"success={_gaze_result['selection_attempt_success']}, "
            f"available={list(config.TIAGOBOT_TRAJECTORY)}"
        )

        # Hold "Selected: <letter>" so the subject sees the chosen
        # letter before Phase 2.5 / Phase 3 / baseline. Skipped on
        # timeout (no letter to confirm — Phase 2.5 will abort the MI
        # trial; REST trials simply proceed without a selection).
        if (_gaze_selected_letter is not None
                and TIAGO_GAZE_CONFIRM_SEC > 0.0):
            ok = hold_tiago_selection_confirmation(
                letter=_gaze_selected_letter,
                eeg_state=eeg_state,
                available_set=set(config.TIAGOBOT_TRAJECTORY),
            )
            if not ok:
                running = False
                logger.log_event(
                    "Experiment terminated during selection confirmation."
                )
                break

        # === Phase 2.5: Abort MI trial on gaze failure ===
        # Per brief Phase 2 + operator confirmation 2026-05-22: when
        # the user IS looking at one of the 9 letters (the protocol
        # guarantee), `_gaze_selected_letter` is None only on a real
        # failure — zero confidence-passing samples (Neon disconnected,
        # glasses off, etc.) or gaze landed implausibly far from any
        # centroid (calibration drift). Both are abort-worthy on MI.
        # Rest trials are untouched: the letter is never consumed, so a
        # None letter has no effect on the trial outcome.
        if mode == 0 and _gaze_selected_letter is None:
            logger.log_event(
                "MI trial aborted: gaze did not resolve to a letter "
                "(no data or distance gate). Skipping baseline + "
                "feedback for this trial; advancing to next."
            )
            display_multiple_messages_with_udp(
                messages=["Trial aborted", "No gaze target"],
                colors=[config.orange, config.white],
                offsets=[-100, 100],
                duration=float(config.TIME_STATIONARY),
                udp_messages=None,
                udp_socket=udp_socket_robot,
                udp_ip=config.UDP_ROBOT["IP"],
                udp_port=config.UDP_ROBOT["PORT"],
                logger=logger,
                eeg_state=eeg_state,
            )
            _next_target = (
                target_letter_sequence[current_trial + 1]
                if current_trial + 1 < len(target_letter_sequence) else None
            )
            _tiago_anticipation_fixation_period(
                duration=float(getattr(config, "TIAGOBOT_ANTICIPATION_DURATION", 3.0)),
                eeg_state=eeg_state,
                message="Look at the fixation cross",
                target_letter=_next_target,
            )
            current_trial += 1
            continue

        # === Phase 3a: empty fixation period (matches base driver) ===
        # `display_fixation_period` renders cross + empty arrow +
        # empty ball + empty time-orb via `draw_class_fixation_idle`
        # — same call the base driver uses at trial wrap
        # (ExperimentDriver_Online.py:261/513). Holds for
        # TIAGOBOT_EMPTY_HOLD_DURATION s with EEG pump.
        display_fixation_period(
            duration=float(getattr(config, "TIAGOBOT_EMPTY_HOLD_DURATION", 3.0)),
            eeg_state=eeg_state,
        )

        # === Phase 3b: white time-orb countdown (base-driver cue) ===
        # Cross + empty arrow + empty ball + WHITE time-orb. Inlined
        # because we don't need the base driver's input handling here
        # (mode was already captured at Phase 1). Mirrors
        # ExperimentDriver_Online.py:268-311 — same draw calls,
        # `draw_time_balls(1, ...)`, same `countdown_duration` (3 s)
        # — minus the backdoor/SPACE event loop.
        screen.fill(config.black)
        draw_fixation_cross(screen_width, screen_height)
        draw_arrow_fill(0, screen_width, screen_height)
        draw_ball_fill(0, screen_width, screen_height)
        draw_time_balls(1, screen_width, screen_height)
        pygame.display.flip()

        _white_end = time.time() + float(
            getattr(config, "TIAGOBOT_MODE_REVEAL_DURATION", 3.0)
        )
        _white_clock = pygame.time.Clock()
        while time.time() < _white_end:
            eeg_state.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
            _white_clock.tick(60)

        # === Baseline ===
        try:
            eeg_state.compute_baseline(duration_sec=config.BASELINE_DURATION)
            logger.log_event(
                f"Computed baseline: shape={eeg_state.baseline_mean.shape}, "
                f"duration={config.BASELINE_DURATION}s"
            )
        except ValueError as e:
            logger.log_event(f"⚠️ Could not compute baseline: {e}")
            continue

        # === Classification ===
        logger.log_event(f"Starting feedback classification — Mode: {'MI' if mode == 0 else 'REST'}")
        prediction, confidence, leaky_integrator, trial_probs, earlystop_flag = show_feedback(
            duration=config.TIME_MI,
            mode=mode,
            eeg_state=eeg_state
        )

        pygame.display.flip()
        pygame.event.get()  # heartbeat to OS
        logger.log_event(f"Classification result — Predicted: {prediction}, Ground Truth: {200 if mode == 0 else 100}")

        append_trial_probabilities_to_csv(
            trial_probabilities=trial_probs,
            mode=mode,
            trial_number=current_trial + 1,
            predicted_label=prediction,
            early_cutout=earlystop_flag,
            mi_threshold=config.THRESHOLD_MI,
            rest_threshold=config.THRESHOLD_REST,
            logger=logger,
            phase="MI" if mode == 0 else "REST"
        )

        predictions_list.append(prediction)
        ground_truth_list.append(200 if mode == 0 else 100)

        # === Feedback branches ===
        if mode == 0:
            if prediction == 200 and _gaze_selected_letter is not None:
                # Correct MI AND gaze resolved -> drive Tiagobot.
                messages = ["Correct", f"Robot Move -> {_gaze_selected_letter}"]
                colors = [config.green, config.green]
                offsets = [-100, 100]
                duration = 0.01
                should_hold_and_classify = True
                logger.log_event(
                    f"Prediction correct for MI — driving Tiagobot to "
                    f"gaze-selected letter {_gaze_selected_letter!r} "
                    f"(and FES / glove if toggled)."
                )

                # Tiagobot GO: send the gaze-classified letter.
                selected_letter = _gaze_selected_letter
                tiago_send_letter(tiago, selected_letter, logger)

                if FES_toggle == 1:
                    send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_MOTOR_GO", logger=logger)
                else:
                    logger.log_event("FES disabled — skipping motor stimulation.")

                send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_BEGIN"], logger=logger)

            elif prediction == 200 and _gaze_selected_letter is None:
                # Defensive: MI-correct but no letter. Normally
                # unreachable — Phase 2.5 aborts MI trials with a None
                # letter before baseline/show_feedback runs (so we
                # never reach this elif). Kept as a safety net for
                # future refactors that might bypass the abort, and to
                # honor the brief's outcome matrix which preserves
                # this defensive branch explicitly.
                messages = ["Correct", "No gaze target"]
                colors = [config.green, config.orange]
                offsets = [-100, 100]
                duration = config.TIME_STATIONARY
                should_hold_and_classify = False
                logger.log_event(
                    "Defensive branch hit: MI-correct with no letter. "
                    "Phase 2.5 abort should have prevented this — "
                    "investigate if it fires."
                )

            elif prediction is None:  # Ambiguous
                messages = ["Ambiguous", "Robot Stationary"]
                colors = [config.orange, config.white]
                offsets = [-100, 100]
                duration = config.TIME_STATIONARY
                should_hold_and_classify = False
                logger.log_event("Prediction ambiguous for MI — Tiagobot remains stationary.")

            else:  # Incorrect
                messages = ["Incorrect", "Robot Stationary"]
                colors = [config.red, config.white]
                offsets = [-100, 100]
                duration = config.TIME_STATIONARY
                should_hold_and_classify = False
                logger.log_event("Prediction incorrect for MI — Tiagobot remains stationary.")

        else:  # REST
            if prediction == 100:  # Correct
                messages = ["Correct", "Robot Stationary"]
                colors = [config.green, config.green]
                offsets = [-100, 100]
                duration = config.TIME_STATIONARY
                logger.log_event("Prediction correct for REST — Tiagobot remains stationary.")
            elif prediction is None:
                messages = ["Ambiguous", "Robot Stationary"]
                colors = [config.orange, config.white]
                offsets = [-100, 100]
                duration = config.TIME_STATIONARY
                logger.log_event("Prediction ambiguous for REST — Tiagobot remains stationary.")
            else:
                messages = ["Incorrect", "Robot Stationary"]
                colors = [config.red, config.white]
                offsets = [-100, 100]
                duration = config.TIME_STATIONARY
                logger.log_event("Prediction incorrect for REST — Tiagobot remains stationary.")

            should_hold_and_classify = False

        # Display feedback. udp_messages is always None for Tiagobot — the
        # serial GO was already sent above. The helper runs its UI loop and
        # skips its (Harmony-specific) UDP staging logic.
        logger.log_event(f"Displaying feedback: '{messages[0]}' | Action: '{messages[1]}' | Duration: {duration}s")
        display_multiple_messages_with_udp(
            messages=messages,
            colors=colors,
            offsets=offsets,
            duration=duration,
            udp_messages=None,
            udp_socket=udp_socket_robot,
            udp_ip=config.UDP_ROBOT["IP"],
            udp_port=config.UDP_ROBOT["PORT"],
            logger=logger,
            eeg_state=eeg_state
        )

        # If MI-correct, continue classifying for the early window, then
        # wait for the actual Tiagobot motion to complete (variable, 5-30+ s
        # depending on the chosen letter), grip, retract with the object,
        # release at home. The Tiagobot sketch has no mid-motion STOP, so
        # we always wait — even on early-stop the actuator runs to target.
        if should_hold_and_classify:
            logger.log_event("Entering real-time classification window during Tiagobot movement...")
            final_class_robot, robot_probs, robot_earlystop = hold_messages_and_classify(
                messages=messages,
                colors=colors,
                offsets=offsets,
                duration=config.TIME_ROB,
                mode=0,
                udp_socket=udp_socket_robot,
                udp_ip=config.UDP_ROBOT["IP"],
                udp_port=config.UDP_ROBOT["PORT"],
                eeg_state=eeg_state,
                leaky_integrator=leaky_integrator
            )
            append_trial_probabilities_to_csv(
                trial_probabilities=robot_probs,
                mode=0,
                trial_number=current_trial + 1,
                predicted_label=final_class_robot,
                early_cutout=robot_earlystop,
                mi_threshold=config.THRESHOLD_MI,
                rest_threshold=config.THRESHOLD_REST,
                logger=logger,
                phase="ROBOT"
            )

            # Drive-loop tick: pumped during the variable-time waits to
            # keep the pygame window responsive, propagate QUIT cleanly,
            # and stop the LSL EEG buffer from backing up.
            def _drive_tick():
                eeg_state.update()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        raise SystemExit
                pygame.display.flip()

            # Wait for the actual GO motion to finish. The sketch prints
            # "Target Location Reached." when the extend loop exits; the
            # serial port has been buffering this line while
            # hold_messages_and_classify ran, so on a fast move we read
            # it immediately, on a slow move we wait until it arrives.
            logger.log_event("Waiting for Tiagobot reach completion...")
            reached = tiago_wait_for_completion(
                tiago, TARGET_REACHED_MARKER, timeout=60.0,
                logger=logger, on_tick=_drive_tick,
            )
            if not reached:
                logger.log_event(
                    "Tiagobot reach timed out after 60 s — proceeding to grip anyway.",
                    level="error",
                )

            # End-effector label: real glove writes if the Arduino port
            # opened, otherwise the user is gripping/releasing with
            # their own hand. The on-screen messages swap "Glove" ↔
            # "Hand" accordingly so the patient sees an instruction
            # that matches what's physically happening.
            effector_label = "Glove" if arduino is not None else "Hand"

            # Grip the target. On early-stop we skip the grip entirely
            # so the operator's "bad MI" interruption doesn't grasp.
            if not robot_earlystop and arduino is not None:
                arduino.write(config.ARDUINO_CMD_MI)
                logger.log_event("Glove closing — gripping target.")

            # Grip-hold period: lets the glove finish its close motion
            # before HOME starts retracting (otherwise the actuator pulls
            # back while the glove is still mid-close). Duration tunable
            # via config.TIAGOBOT_GRIP_HOLD_DURATION (default 4 s).
            # Show "{Glove|Hand} closing" so the patient knows to close
            # their hand (gloveless mode) or sees what the glove is
            # doing (glove enabled).
            if not robot_earlystop:
                grip_hold = float(getattr(config, "TIAGOBOT_GRIP_HOLD_DURATION", 4))
                _tiago_hold_message_screen(
                    messages=["Correct", f"{effector_label} closing"],
                    colors=[config.green, config.green],
                    offsets=[-100, 100],
                    duration=grip_hold,
                    eeg_state=eeg_state,
                )

            # Send HOME — Tiagobot retracts with the gripped object.
            send_udp_message(
                udp_socket_marker,
                config.UDP_MARKER["IP"],
                config.UDP_MARKER["PORT"],
                config.TRIGGERS["ROBOT_HOME"],
                logger=logger,
            )
            tiago_send_home(tiago, logger)
            logger.log_event("Sent HOME to Tiagobot — retracting.")

            # Render "Robot Move -> Home" once. The message persists
            # through the subsequent wait_for_completion(HOMED) because
            # _drive_tick only flips the buffer; the screen still holds
            # the contents we just drew. Style mirrors the outbound
            # "Robot Move -> {letter}" message.
            _tiago_draw_message_screen(
                messages=["Correct", "Robot Move -> Home"],
                colors=[config.green, config.green],
                offsets=[-100, 100],
            )

            # Wait for the retract to actually finish before opening
            # the glove (so the object goes back to the home position
            # rather than being dropped mid-retract).
            logger.log_event("Waiting for Tiagobot home completion...")
            homed = tiago_wait_for_completion(
                tiago, HOMED_MARKER, timeout=60.0,
                logger=logger, on_tick=_drive_tick,
            )
            if not homed:
                logger.log_event(
                    "Tiagobot home timed out after 60 s — releasing glove anyway.",
                    level="error",
                )

            # Release at home.
            if not robot_earlystop and arduino is not None:
                arduino.write(config.ARDUINO_CMD_REST)
                logger.log_event("Glove opening — releasing at home.")

            # Post-home: show "{Glove|Hand} opening" instead of the
            # idle fixation cross so the patient knows to release the
            # object (gloveless) or sees the glove opening. Replaces
            # the original `display_fixation_period(3)` call here.
            _tiago_hold_message_screen(
                messages=["Correct", f"{effector_label} opening"],
                colors=[config.green, config.green],
                offsets=[-100, 100],
                duration=3,
                eeg_state=eeg_state,
            )

        logger.log_trial_summary(
            trial_number=current_trial + 1,
            true_label=200 if mode == 0 else 100,
            predicted_label=prediction,
            early_cutout=earlystop_flag,
            accuracy_threshold=config.THRESHOLD_MI if mode == 0 else config.THRESHOLD_REST,
            confidence=confidence,
            num_predictions=len(trial_probs)
        )

        _next_target = (
            target_letter_sequence[current_trial + 1]
            if current_trial + 1 < len(target_letter_sequence) else None
        )
        _tiago_anticipation_fixation_period(
            duration=float(getattr(config, "TIAGOBOT_ANTICIPATION_DURATION", 3.0)),
            eeg_state=eeg_state,
            message="Look at the fixation cross",
            target_letter=_next_target,
        )
        logger.log_event(f"Trial {current_trial+1} complete. Proceeding to next.")

        current_trial += 1
        pygame.display.flip()
        clock.tick(60)

    if current_trial == len(trial_sequence) and config.SAVE_ADAPTIVE_T:
        try:
            save_transform(
                _RC.Prev_T,
                _RC.counter,
                adaptive_T_path,
                T_beta=_RC.Prev_T_beta,
                counter_beta=_RC.counter_beta,
            )
        except Exception as e:
            logger.log_event(f"⚠️ Could not save transform to {adaptive_T_path}: {e}")

    log_confusion_matrix_from_trial_summary(logger)
    logger.log_event("run complete")


if __name__ == "__main__":
    # try/finally so hardware cleanup runs on any exit path: clean
    # session end, SystemExit raised by runtime_common when the
    # operator closes the pygame window, KeyboardInterrupt, etc.
    try:
        main()
    finally:
        _hardware_cleanup()
