"""
ExperimentDriver_ErrP.py
========================
ErrP (Error-Related Potential) data collection paradigm.

Experimental design:
  The participant observes a robot arm performing reach movements.
  On ERRP_P_STOP fraction of trials the robot stops unexpectedly mid-trajectory
  (error condition). On the remaining trials it completes normally (correct condition).

  Because the participant cannot anticipate the early stop, the unexpected
  robot arrest reliably elicits an Error-Related Potential (ERN + Pe complex)
  over frontocentral scalp sites.

  The driver sends two simultaneous markers at the event moment:
    - ROBOT_EARLYSTOP  (340)  — backwards-compatible legacy marker
    - ERRP_STIM_ERROR  (430)  — dedicated ErrP marker for new pipeline

  And at normal completion:
    - ROBOT_END        (320)
    - ERRP_STIM_CORRECT (440)

LSL / LabRecorder:
  Start LabRecorder BEFORE running this script.
  The driver does NOT close or manage LabRecorder.

Robot:
  Set SIMULATION_MODE = True in config.py to run without the physical robot.

Usage:
    python ExperimentDriver_ErrP.py
"""

import os
import sys
import random
import socket
import time
import datetime
from pathlib import Path

import pygame

from pylsl import StreamInlet, resolve_stream

import config
from Utils.logging_manager import LoggerManager
from Utils.networking import send_udp_message
from Utils.visualization import draw_fixation_cross

# ──────────────────────────────────────────────────────────────────────────────
# Experiment parameters (from config with safe defaults)
# ──────────────────────────────────────────────────────────────────────────────
TOTAL_TRIALS    = int(getattr(config, "TOTAL_TRIALS_ERRP",     45))
P_STOP          = float(getattr(config, "ERRP_P_STOP",          0.5))
STOP_TMIN       = float(getattr(config, "ERRP_STOP_TMIN",       1.0))   # s into trajectory
STOP_TMAX_FRAC  = float(getattr(config, "ERRP_STOP_TMAX_FRACTION", 0.7))
ITI_FIXATION    = 3.0    # inter-trial fixation duration (s)
POST_EVENT_WAIT = 1.5    # blank period after error/correct event before home (s)

TIME_ROB   = float(config.TIME_ROB)
STOP_TMAX  = STOP_TMAX_FRAC * TIME_ROB

# Marker codes
TRIG = config.TRIGGERS
SIM  = bool(config.SIMULATION_MODE)


# ──────────────────────────────────────────────────────────────────────────────
# Logger + display
# ──────────────────────────────────────────────────────────────────────────────
logger = LoggerManager.auto_detect_from_subject(
    subject=config.TRAINING_SUBJECT,
    base_path=Path(config.DATA_DIR),
    mode="offline",
)

loggable_fields = [
    "TRAINING_SUBJECT", "DATA_DIR", "TOTAL_TRIALS_ERRP",
    "ERRP_P_STOP", "ERRP_STOP_TMIN", "ERRP_STOP_TMAX_FRACTION",
    "TIME_ROB", "LOWCUT_ERRP", "HIGHCUT_ERRP", "ERRP_CHANNEL_NAMES",
    "ERRP_EPOCH_TMIN", "ERRP_EPOCH_TMAX", "SIMULATION_MODE",
]
logger.save_config_snapshot({k: getattr(config, k) for k in loggable_fields if hasattr(config, k)})

pygame.init()
if config.BIG_BROTHER_MODE:
    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
    screen = pygame.display.set_mode((1920, 1080), pygame.NOFRAME)
else:
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

pygame.display.set_caption("ErrP Data Collection")
info      = pygame.display.Info()
SW, SH    = info.current_w, info.current_h
clock     = pygame.time.Clock()

FONT_LG   = pygame.font.SysFont(None, 96)
FONT_MD   = pygame.font.SysFont(None, 64)
FONT_SM   = pygame.font.SysFont(None, 42)

# UDP sockets
udp_marker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_robot  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


# ──────────────────────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────────────────────

def _send(marker_key: str, label: str = ""):
    """Send a UDP marker (suppressed in SIMULATION_MODE for robot socket)."""
    code = TRIG.get(marker_key, "")
    if code:
        send_udp_message(udp_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"],
                         code, logger=logger)
        logger.log_event(f"Marker {marker_key}={code} {label}")


def _send_robot(opcode: str, expect_ack: bool = False):
    """Send robot UDP command (no-op in SIMULATION_MODE)."""
    if SIM:
        logger.log_event(f"[SIM] Robot command suppressed: {opcode}")
        return True, None
    return send_udp_message(
        udp_robot,
        config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"],
        opcode,
        logger=logger,
        expect_ack=expect_ack,
        ack_timeout=1.5,
        max_retries=1,
    )


def _pump():
    """Pygame event pump — keeps OS responsive."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            _quit()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            _quit()


def _quit():
    pygame.quit()
    logger.log_event("Experiment terminated by operator.")
    sys.exit(0)


def _fill(color=config.black):
    screen.fill(color)


def _text(msg: str, font, color=config.white, y_offset: int = 0):
    surf = font.render(msg, True, color)
    screen.blit(surf, (SW // 2 - surf.get_width() // 2,
                        SH // 2 - surf.get_height() // 2 + y_offset))


def _flip():
    pygame.display.flip()
    _pump()


def show_fixation(duration: float):
    """Display fixation cross for `duration` seconds."""
    end = time.time() + duration
    while time.time() < end:
        _fill()
        draw_fixation_cross(SW, SH)
        _flip()
        clock.tick(60)


def show_text_screen(lines: list[tuple[str, object, tuple, int]], duration: float):
    """
    Display one or more text lines for `duration` seconds.

    lines: list of (text, font, color_rgb, y_offset)
    """
    end = time.time() + duration
    while time.time() < end:
        _fill()
        for (msg, font, col, yo) in lines:
            _text(msg, font, col, yo)
        _flip()
        clock.tick(60)


def show_robot_moving(stop_time: float | None, trajectory_duration: float) -> float:
    """
    Display animated progress bar while robot moves.

    Args:
        stop_time: if not None, the elapsed time at which we stop the robot
        trajectory_duration: total expected trajectory duration (TIME_ROB)

    Returns:
        elapsed time when the event (stop or completion) actually occurred
    """
    start = time.time()
    bar_w = int(SW * 0.6)
    bar_h = 40
    bar_x = (SW - bar_w) // 2
    bar_y = SH // 2 + 60

    event_elapsed = trajectory_duration   # default = completion

    while True:
        elapsed = time.time() - start
        progress = min(elapsed / trajectory_duration, 1.0)

        _fill()
        draw_fixation_cross(SW, SH)
        _text("Observe the robot", FONT_MD, config.white, -120)

        # Progress bar outline
        pygame.draw.rect(screen, config.white,
                         pygame.Rect(bar_x, bar_y, bar_w, bar_h), 2)
        # Progress bar fill
        fill_w = int(progress * bar_w)
        pygame.draw.rect(screen, (80, 140, 220),
                         pygame.Rect(bar_x, bar_y, fill_w, bar_h))

        # Expected duration marker
        _text(f"{elapsed:.1f} / {trajectory_duration:.0f} s", FONT_SM, config.white, 130)
        _flip()
        clock.tick(60)

        # Check for error stop
        if stop_time is not None and elapsed >= stop_time:
            event_elapsed = elapsed
            break

        # Check for normal completion
        if elapsed >= trajectory_duration:
            event_elapsed = elapsed
            break

    return event_elapsed


def show_event_feedback(is_error: bool, duration: float = 0.8):
    """
    Brief visual feedback to help participant perceive the event.
    Error = red flash "✕ STOP"  |  Correct = green flash "✓ DONE"
    """
    color = config.red   if is_error else config.green
    label = "STOP"       if is_error else "DONE"
    bg    = (60, 0, 0)   if is_error else (0, 60, 0)

    end = time.time() + duration
    while time.time() < end:
        screen.fill(bg)
        _text(label, FONT_LG, color, 0)
        _flip()
        clock.tick(60)


def show_instructions():
    """Show task instructions until participant presses SPACE."""
    lines = [
        ("ErrP Data Collection", FONT_LG, config.white,   -200),
        ("Watch the robot arm carefully.",    FONT_MD, config.white, -80),
        ("Sometimes it will stop unexpectedly.", FONT_SM, (200, 200, 200), 0),
        ("Stay still and focused throughout.", FONT_SM, (200, 200, 200), 60),
        ("Press SPACE to begin.", FONT_SM, config.green, 180),
    ]
    waiting = True
    while waiting:
        _fill()
        for (msg, font, col, yo) in lines:
            _text(msg, font, col, yo)
        _flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                _quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False
                if event.key == pygame.K_ESCAPE:
                    _quit()
        clock.tick(30)


# ──────────────────────────────────────────────────────────────────────────────
# Trial sequence generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_errp_trial_sequence(n_trials: int, p_stop: float) -> list[bool]:
    """
    Return a list of booleans: True = error trial, False = correct trial.

    Uses balanced random assignment. Ensures at least 1 error and 1 correct trial.
    """
    n_error = max(1, round(n_trials * p_stop))
    n_correct = n_trials - n_error
    seq = [True] * n_error + [False] * n_correct
    random.shuffle(seq)
    return seq


# ──────────────────────────────────────────────────────────────────────────────
# Single trial
# ──────────────────────────────────────────────────────────────────────────────

def run_trial(trial_num: int, is_error: bool, trajectory: str = "a") -> dict:
    """
    Execute one ErrP trial and return a result dict.

    Args:
        trial_num:  1-based trial index
        is_error:   True = unexpected robot stop; False = normal completion
        trajectory: robot trajectory opcode
    """
    label_str = "ERROR" if is_error else "CORRECT"
    logger.log_event(f"--- Trial {trial_num}/{TOTAL_TRIALS} START  ({label_str}) ---")

    # 1. Pre-trial fixation
    show_fixation(ITI_FIXATION)

    # 2. Ready cue (500ms)
    show_text_screen([("Get ready...", FONT_MD, (200, 200, 200), 0)], 0.5)

    # 3. Robot start
    _send("ROBOT_BEGIN", f"trial={trial_num}")
    if not SIM:
        _send_robot(trajectory)
        _send_robot(config.ROBOT_OPCODES.get("GO", "g"))

    # 4. Compute stop time for error trials
    if is_error:
        stop_time = random.uniform(
            max(STOP_TMIN, 0.5),
            min(STOP_TMAX, TIME_ROB - 0.5),
        )
        logger.log_event(f"   Planned stop at {stop_time:.2f}s into trajectory")
    else:
        stop_time = None

    # 5. Show robot moving + wait for event
    event_elapsed = show_robot_moving(stop_time, TIME_ROB)

    # 6. Send event markers
    if is_error:
        _send("ROBOT_EARLYSTOP",   f"t_stop={event_elapsed:.2f}s")
        _send("ERRP_STIM_ERROR",   f"t_stop={event_elapsed:.2f}s")
        if not SIM:
            _send_robot(config.ROBOT_OPCODES.get("STOP", "s"))
    else:
        _send("ROBOT_END",         "normal completion")
        _send("ERRP_STIM_CORRECT", "normal completion")

    # 7. Brief event feedback
    show_event_feedback(is_error=is_error, duration=0.6)

    # 8. Short blank + post-event fixation
    show_fixation(POST_EVENT_WAIT)

    # 9. Home robot
    _send("ROBOT_HOME", f"trial={trial_num}")
    if not SIM:
        _send_robot(config.ROBOT_OPCODES.get("HOME", "h;dur=3"), expect_ack=True)

    logger.log_event(f"--- Trial {trial_num} END  ({label_str}) ---")

    return {
        "trial":         trial_num,
        "is_error":      is_error,
        "event_elapsed": event_elapsed,
        "stop_time":     stop_time,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    logger.log_event("ExperimentDriver_ErrP started.")
    logger.log_event(
        f"Parameters: TOTAL_TRIALS={TOTAL_TRIALS}, P_STOP={P_STOP:.2f}, "
        f"STOP_TMIN={STOP_TMIN:.1f}s, STOP_TMAX={STOP_TMAX:.1f}s, "
        f"TIME_ROB={TIME_ROB:.1f}s, SIMULATION_MODE={SIM}"
    )

    # Optionally connect to LSL EEG stream (for real-time monitoring only)
    try:
        logger.log_event("Resolving EEG stream via LSL (for recording sync)...")
        streams = resolve_stream("type", "EEG")
        _inlet = StreamInlet(streams[0])
        logger.log_event("✅ EEG stream detected.")
    except Exception as exc:
        logger.log_event(f"⚠️ No EEG stream found ({exc}) — continuing without LSL sync.")

    # Show instructions
    show_instructions()

    # Generate trial sequence
    trial_seq = generate_errp_trial_sequence(TOTAL_TRIALS, P_STOP)
    n_error   = sum(trial_seq)
    n_correct = TOTAL_TRIALS - n_error
    logger.log_event(
        f"Trial sequence: {TOTAL_TRIALS} trials  |  error={n_error}  correct={n_correct}"
    )

    # Initial fixation
    show_fixation(3.0)

    results = []
    for trial_idx, is_error in enumerate(trial_seq):
        trial_num = trial_idx + 1

        trajectory = random.choice(config.ROBOT_TRAJECTORY)
        result = run_trial(trial_num, is_error, trajectory=trajectory)
        results.append(result)

        # Allow operator abort
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                logger.log_event("Operator aborted experiment.")
                break
        else:
            continue
        break

    # End screen
    n_done    = len(results)
    n_err_done = sum(r["is_error"] for r in results)
    show_text_screen([
        ("Session Complete!", FONT_LG, config.green, -80),
        (f"{n_done} trials completed  ({n_err_done} error, {n_done-n_err_done} correct)",
         FONT_SM, config.white, 20),
        ("Press SPACE to exit.", FONT_SM, (180, 180, 180), 100),
    ], duration=30.0)

    logger.log_event(
        f"Experiment complete: {n_done}/{TOTAL_TRIALS} trials, "
        f"{n_err_done} error, {n_done-n_err_done} correct."
    )
    pygame.quit()


if __name__ == "__main__":
    main()
