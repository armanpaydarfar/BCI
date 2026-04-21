"""
ExperimentDriver_ErrP_Online.py
================================
Combined MI + ErrP online experiment driver.

Paradigm:
  1. MI classification phase (identical to ExperimentDriver_Online.py).
  2. On successful MI: robot begins trajectory.
  3. At a random time within the movement window the robot PAUSES unexpectedly.
  4. The ErrP decoder monitors the 0-800 ms window following the pause.
  5. If an Error-Related Potential is detected (participant perceived the pause
     as an error — i.e., they still intend to move): robot RESUMES.
  6. If no ErrP / ambiguous: robot remains paused (intent assumed satisfied or
     uncertain); trial ends with robot homing after a short hold.

Why this works:
  An unexpected mid-trajectory pause is ecologically valid as an "error" from
  the participant's perspective when they are driving the robot with sustained MI.
  The ERN/Pe complex (~80-400 ms post-pause) at frontocentral channels encodes
  the participant's perceived error, directly signalling "continue motion".

Feature gating:
  ERRP_DECODER_ENABLE = 0 → ErrP pipeline disabled; robot moves without random
    pause and driver behaviour is identical to ExperimentDriver_Online.
  ERRP_DECODER_ENABLE = 1 → ErrP pause-detect-resume loop active.

Robot protocol:
  PAUSE  opcode = ROBOT_OPCODES["PAUSE"]  = "p"   (ROBOT_PAUSE marker 360)
  RESUME opcode = ROBOT_OPCODES["RESUME"] = "r"   (ROBOT_RESUME marker 370)
  Both commands have ACK variants (365, 375) and use expect_ack=True.

Two LSL inlets:
  LSL supports multiple independent consumers of the same stream.
  inlet_mi   → EEGStreamState(mode="motor") for continuous MI classification.
  inlet_errp → EEGStreamState(mode="errp")  for ErrP epoch classification.
  Both are updated every frame; the ErrP state accumulates 1-10 Hz filtered
  data and is ready to classify immediately after ERRP_EPOCH_TMAX seconds
  have elapsed since the pause.
"""

import os
import sys
import random
import socket
import pickle
import datetime
import time
from pathlib import Path

import pygame
from pylsl import StreamInlet, resolve_stream, local_clock

import mne
mne.set_log_level("WARNING")

import config
from Utils.EEGStreamState import EEGStreamState
from Utils.logging_manager import LoggerManager
from Utils.experiment_utils import generate_trial_sequence, LeakyIntegrator, save_transform, load_transform
from Utils.networking import send_udp_message, display_multiple_messages_with_udp
from Utils.visualization import (
    draw_arrow_fill, draw_ball_fill, draw_fixation_cross,
    draw_time_balls, draw_progress_bar,
)
from Utils.runtime_common import (
    log_confusion_matrix_from_trial_summary,
    append_trial_probabilities_to_csv,
    display_fixation_period,
    hold_messages_and_classify,
    show_feedback,
    load_errp_model,
    classify_errp_real_time,
    fit_errp_ea_bootstrap,
)
import Utils.runtime_common as _RC

# ──────────────────────────────────────────────────────────────────────────────
# ErrP timing parameters (from config with safe defaults)
# ──────────────────────────────────────────────────────────────────────────────
ERRP_ENABLE          = bool(getattr(config, "ERRP_DECODER_ENABLE",       0))
ERRP_PAUSE_TMIN      = float(getattr(config, "ERRP_STOP_TMIN",           1.0))
ERRP_PAUSE_TMAX_FRAC = float(getattr(config, "ERRP_STOP_TMAX_FRACTION",  0.7))
ERRP_PAUSE_TMAX      = ERRP_PAUSE_TMAX_FRAC * float(config.TIME_ROB)
ERRP_EPOCH_SEC       = float(getattr(config, "ERRP_EPOCH_TMAX",          0.8))
ERRP_NO_RESUME_HOLD  = float(getattr(config, "ERRP_NO_RESUME_TIMEOUT",   3.0))

# ──────────────────────────────────────────────────────────────────────────────
# Startup: logger, pygame, UDP, model loading
# ──────────────────────────────────────────────────────────────────────────────
logger = LoggerManager.auto_detect_from_subject(
    subject=config.TRAINING_SUBJECT,
    base_path=Path(config.DATA_DIR),
    mode="online",
)

loggable_fields = [
    "UDP_MARKER", "UDP_ROBOT", "UDP_FES",
    "ARM_SIDE", "TOTAL_TRIALS", "MAX_REPEATS",
    "TIME_MI", "TIME_ROB", "TIME_STATIONARY",
    "SHAPE_MAX", "SHAPE_MIN", "ROBOT_TRAJECTORY",
    "FES_toggle", "FES_CHANNEL", "FES_TIMING_OFFSET",
    "WORKING_DIR", "DATA_DIR", "TRAINING_SUBJECT",
    "MOTOR_CHANNEL_NAMES", "CLASSIFY_WINDOW",
    "THRESHOLD_MI", "THRESHOLD_REST", "MIN_PREDICTIONS",
    "SURFACE_LAPLACIAN_TOGGLE", "SELECT_MOTOR_CHANNELS",
    "INTEGRATOR_ALPHA", "SHRINKAGE_PARAM_MDM", "SHRINKAGE_PARAM_XGB",
    "LEDOITWOLF", "RECENTERING", "UPDATE_DURING_MOVE",
    # ErrP-specific
    "ERRP_DECODER_ENABLE", "ERRP_DECODER_BACKEND",
    "ERRP_EPOCH_TMIN", "ERRP_EPOCH_TMAX",
    "ERRP_STOP_TMIN", "ERRP_STOP_TMAX_FRACTION",
    "ERRP_XDAWN_N_FILTERS", "ERRP_CHANNEL_NAMES",
]
logger.save_config_snapshot({k: getattr(config, k) for k in loggable_fields if hasattr(config, k)})

pygame.init()
if config.BIG_BROTHER_MODE:
    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
    screen = pygame.display.set_mode((1920, 1080), pygame.NOFRAME)
else:
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("ErrP Online Driver")
disp_info   = pygame.display.Info()
screen_width  = disp_info.current_w
screen_height = disp_info.current_h

# UDP sockets
udp_socket_marker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket_robot  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket_fes    = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
FES_toggle = config.FES_toggle

# Adaptive recentering state
eeg_dir          = Path(config.DATA_DIR) / f"sub-{config.TRAINING_SUBJECT}" / "eeg"
adaptive_T_path  = eeg_dir / "adaptive_T.pkl"
Prev_T, counter, Prev_T_beta, counter_beta = load_transform(adaptive_T_path)

# ── MI model ─────────────────────────────────────────────────────────────────
subject_model_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "models")
decoder_backend   = str(getattr(config, "DECODER_BACKEND", "mdm")).lower()
if decoder_backend == "xgb_cov":
    mi_model_filename = f"sub-{config.TRAINING_SUBJECT}_xgb_cov_features.pkl"
elif decoder_backend == "xgb_cov_erd":
    mi_model_filename = f"sub-{config.TRAINING_SUBJECT}_xgb_cov_erd_features.pkl"
else:
    mi_model_filename = f"sub-{config.TRAINING_SUBJECT}_model.pkl"
mi_model_path = os.path.join(subject_model_dir, mi_model_filename)

try:
    with open(mi_model_path, "rb") as f:
        mi_model = pickle.load(f)
    logger.log_event(f"✅ MI model loaded: {mi_model_path}")
except FileNotFoundError:
    logger.log_event(f"❌ MI model not found: {mi_model_path}", level="error")
    pygame.quit(); sys.exit(1)

# ── ErrP model (optional; only needed when ERRP_DECODER_ENABLE = 1) ──────────
errp_model_path = None
if ERRP_ENABLE:
    errp_backend   = str(getattr(config, "ERRP_DECODER_BACKEND", "xdawn_mdm")).lower()
    errp_model_path = os.path.join(
        subject_model_dir,
        f"sub-{config.TRAINING_SUBJECT}_errp_{errp_backend}.pkl",
    )
    if not os.path.exists(errp_model_path):
        logger.log_event(
            f"❌ ErrP model not found: {errp_model_path}\n"
            f"   Train it first with: python generate_errp_decoder.py\n"
            f"   Or disable with: ERRP_DECODER_ENABLE = 0 in config.py",
            level="error",
        )
        pygame.quit(); sys.exit(1)
    logger.log_event(f"ErrP enabled: pause_tmin={ERRP_PAUSE_TMIN:.1f}s  "
                     f"pause_tmax={ERRP_PAUSE_TMAX:.1f}s  "
                     f"epoch_window={ERRP_EPOCH_SEC:.2f}s")
else:
    logger.log_event("ErrP decoder DISABLED (ERRP_DECODER_ENABLE = 0) — "
                     "robot will move without random pause.")

# ── Wire runtime globals ──────────────────────────────────────────────────────
_RC.config          = config
_RC.logger          = logger
_RC.model           = mi_model
_RC.screen          = screen
_RC.screen_width    = screen_width
_RC.screen_height   = screen_height
_RC.udp_socket_marker = udp_socket_marker
_RC.udp_socket_robot  = udp_socket_robot
_RC.udp_socket_fes    = udp_socket_fes
_RC.FES_toggle        = FES_toggle
_RC.Prev_T            = Prev_T
_RC.counter           = counter
_RC.Prev_T_beta       = Prev_T_beta
_RC.counter_beta      = counter_beta


# ──────────────────────────────────────────────────────────────────────────────
# ErrP pause-detect-resume movement phase
# ──────────────────────────────────────────────────────────────────────────────

def _pump_both(eeg_state: EEGStreamState, errp_eeg_state: EEGStreamState | None):
    """Update both EEGStreamState buffers and pump pygame events."""
    eeg_state.update()
    if errp_eeg_state is not None:
        errp_eeg_state.update()
    pygame.event.pump()


def _draw_movement_progress(elapsed: float, total: float, label: str, color):
    """Render a simple progress bar and status label during robot movement."""
    screen.fill(config.black)
    draw_fixation_cross(screen_width, screen_height)

    font_md = pygame.font.SysFont(None, 64)
    surf = font_md.render(label, True, color)
    screen.blit(surf, (screen_width // 2 - surf.get_width() // 2,
                        screen_height // 2 - 120))

    bar_w = int(screen_width * 0.55)
    bar_h = 36
    bar_x = (screen_width - bar_w) // 2
    bar_y = screen_height // 2 + 60
    progress = min(elapsed / max(total, 0.001), 1.0)
    pygame.draw.rect(screen, config.white, pygame.Rect(bar_x, bar_y, bar_w, bar_h), 2)
    pygame.draw.rect(screen, color,
                     pygame.Rect(bar_x, bar_y, int(progress * bar_w), bar_h))

    font_sm = pygame.font.SysFont(None, 38)
    t_surf  = font_sm.render(f"{elapsed:.1f} / {total:.0f} s", True, config.white)
    screen.blit(t_surf, (screen_width // 2 - t_surf.get_width() // 2, bar_y + bar_h + 12))
    pygame.display.flip()


def _draw_errp_status(label: str, color, prob: float | None = None):
    """Render a brief ErrP detection status overlay."""
    screen.fill((20, 20, 40))
    draw_fixation_cross(screen_width, screen_height)
    font_lg = pygame.font.SysFont(None, 96)
    font_sm = pygame.font.SysFont(None, 42)
    surf = font_lg.render(label, True, color)
    screen.blit(surf, (screen_width // 2 - surf.get_width() // 2,
                        screen_height // 2 - 80))
    if prob is not None:
        p_surf = font_sm.render(f"P(ErrP) = {prob:.2f}", True, (200, 200, 200))
        screen.blit(p_surf, (screen_width // 2 - p_surf.get_width() // 2,
                              screen_height // 2 + 60))
    pygame.display.flip()


def run_robot_movement_with_errp(
    eeg_state: EEGStreamState,
    errp_eeg_state: EEGStreamState | None,
    leaky_integrator: LeakyIntegrator,
    selected_trajectory: str,
) -> dict:
    """
    Execute the robot movement phase with optional ErrP-triggered pause/resume.

    State machine:
      MOVING     → robot moving; at t_pause transition to PAUSED_WAIT
      PAUSED_WAIT→ accumulate ErrP epoch window (ERRP_EPOCH_SEC)
      DECIDING   → single ErrP classification call
      RESUMING   → ErrP detected; robot resumes; finish remaining movement
      HOLDING    → no ErrP; robot stays paused for ERRP_NO_RESUME_HOLD then homes

    Returns:
      dict with keys: completed (bool), errp_detected (bool | None),
        prob_error (float | None), pause_time (float | None),
        robot_probs (list), robot_earlystop (bool)
    """
    result = {
        "completed":     False,
        "errp_detected": None,
        "prob_error":    None,
        "pause_time":    None,
        "robot_probs":   [],
        "robot_earlystop": False,
    }

    movement_total = float(config.TIME_ROB)
    clock_fps      = pygame.time.Clock()

    # ── Choose random pause time ──────────────────────────────────────────────
    if ERRP_ENABLE and errp_eeg_state is not None:
        t_pause = random.uniform(
            max(ERRP_PAUSE_TMIN, 0.5),
            min(ERRP_PAUSE_TMAX, movement_total - 0.5),
        )
        result["pause_time"] = t_pause
        logger.log_event(f"ErrP pause planned at t={t_pause:.2f}s into trajectory "
                         f"(total={movement_total:.1f}s)")
    else:
        t_pause = None   # no pause

    state        = "MOVING"
    move_start   = time.time()
    pause_ts     = None   # wall-clock time when pause occurred
    pause_ts_lsl = None   # LSL local_clock at pause-marker send; drives
                          # get_event_baseline_window in classify_errp_real_time

    while True:
        now     = time.time()
        elapsed = now - move_start

        # ── Update EEG buffers every frame ────────────────────────────────────
        _pump_both(eeg_state, errp_eeg_state)
        clock_fps.tick(60)

        # ─────────────────────────────────────────────────────────────────────
        # STATE: MOVING
        # ─────────────────────────────────────────────────────────────────────
        if state == "MOVING":
            _draw_movement_progress(elapsed, movement_total, "Robot Moving", (80, 140, 220))

            # Normal completion
            if elapsed >= movement_total:
                send_udp_message(
                    udp_socket_marker,
                    config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"],
                    config.TRIGGERS["ROBOT_END"], logger=logger,
                )
                logger.log_event("Robot movement complete — ROBOT_END sent.")
                result["completed"] = True
                break

            # Trigger pause
            if t_pause is not None and elapsed >= t_pause:
                # Capture the LSL clock right before the marker send so the
                # event timestamp matches (within sub-ms) the moment the
                # EEG buffer saw the robot pause. This timestamp is what
                # classify_errp_real_time passes to get_event_baseline_window
                # so the [-200, 0] ms baseline is anchored at the same
                # event the participant perceived.
                pause_ts_lsl = local_clock()
                # Send ROBOT_PAUSE marker + pause opcode (with ACK)
                send_udp_message(
                    udp_socket_marker,
                    config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"],
                    config.TRIGGERS["ROBOT_PAUSE"], logger=logger,
                )
                send_udp_message(
                    udp_socket_robot,
                    config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"],
                    config.ROBOT_OPCODES["PAUSE"],
                    logger=logger, expect_ack=True, ack_timeout=1.0, max_retries=1,
                )
                logger.log_event(f"Robot PAUSED at t={elapsed:.2f}s — "
                                 f"entering ErrP detection window ({ERRP_EPOCH_SEC:.2f}s)")
                pause_ts = time.time()
                state    = "PAUSED_WAIT"

        # ─────────────────────────────────────────────────────────────────────
        # STATE: PAUSED_WAIT — accumulate 0-ERRP_EPOCH_SEC of post-pause data
        # ─────────────────────────────────────────────────────────────────────
        elif state == "PAUSED_WAIT":
            wait_elapsed = now - pause_ts
            _draw_errp_status("Analyzing...", (180, 180, 60))

            if wait_elapsed >= ERRP_EPOCH_SEC:
                state = "DECIDING"

        # ─────────────────────────────────────────────────────────────────────
        # STATE: DECIDING — single ErrP classification call
        # ─────────────────────────────────────────────────────────────────────
        elif state == "DECIDING":
            prob_err, decision = classify_errp_real_time(
                errp_eeg_state, event_ts=pause_ts_lsl,
            )
            result["prob_error"]    = prob_err
            result["errp_detected"] = (decision == 1)
            logger.log_event(
                f"ErrP classification: P(error)={prob_err:.3f}  "
                f"decision={decision}  (tl={_RC.errp_model['tl_star']:.3f}  "
                f"th={_RC.errp_model['th_star']:.3f})"
            )

            if decision == 1:
                state = "RESUMING"
                # Send ROBOT_RESUME marker + resume opcode
                send_udp_message(
                    udp_socket_marker,
                    config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"],
                    config.TRIGGERS["ROBOT_RESUME"], logger=logger,
                )
                send_udp_message(
                    udp_socket_robot,
                    config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"],
                    config.ROBOT_OPCODES["RESUME"],
                    logger=logger, expect_ack=True, ack_timeout=1.0, max_retries=1,
                )
                logger.log_event("ErrP DETECTED — robot resuming.")
                _draw_errp_status("ErrP — Resuming", config.green, prob_err)
                time.sleep(0.4)   # brief visual confirmation
                resume_elapsed = time.time() - move_start  # re-anchor elapsed
            else:
                state = "HOLDING"
                _label = "No ErrP — Paused" if decision == 0 else "Ambiguous — Paused"
                logger.log_event(f"No ErrP / ambiguous — robot staying paused. label={decision}")
                _draw_errp_status(_label, config.orange, prob_err)
                hold_start = time.time()

        # ─────────────────────────────────────────────────────────────────────
        # STATE: RESUMING — robot has resumed; finish remaining movement
        # ─────────────────────────────────────────────────────────────────────
        elif state == "RESUMING":
            remaining = movement_total - resume_elapsed
            elapsed_since_resume = now - (pause_ts + ERRP_EPOCH_SEC + 0.4)
            _draw_movement_progress(
                min(resume_elapsed + elapsed_since_resume, movement_total),
                movement_total, "Resumed", (60, 200, 100),
            )

            if elapsed_since_resume >= max(remaining, 0):
                send_udp_message(
                    udp_socket_marker,
                    config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"],
                    config.TRIGGERS["ROBOT_END"], logger=logger,
                )
                logger.log_event("Robot movement complete after resume — ROBOT_END sent.")
                result["completed"] = True
                break

        # ─────────────────────────────────────────────────────────────────────
        # STATE: HOLDING — no ErrP; robot stays paused; wait then home
        # ─────────────────────────────────────────────────────────────────────
        elif state == "HOLDING":
            hold_elapsed = time.time() - hold_start
            _draw_errp_status(
                f"Paused ({ERRP_NO_RESUME_HOLD - hold_elapsed:.1f}s)",
                config.orange,
                result["prob_error"],
            )
            if hold_elapsed >= ERRP_NO_RESUME_HOLD:
                # Abandon: send ROBOT_EARLYSTOP + ERRP_STIM_ERROR for logging
                send_udp_message(
                    udp_socket_marker,
                    config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"],
                    config.TRIGGERS["ROBOT_EARLYSTOP"], logger=logger,
                )
                send_udp_message(
                    udp_socket_robot,
                    config.UDP_ROBOT["IP"], config.UDP_ROBOT["PORT"],
                    config.ROBOT_OPCODES["STOP"],
                    logger=logger, expect_ack=False,
                )
                logger.log_event("No-ErrP hold expired — robot abandoned (EARLYSTOP).")
                result["robot_earlystop"] = True
                break

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main trial loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    logger.log_event("ExperimentDriver_ErrP_Online: resolving EEG stream...")
    streams = resolve_stream("type", "EEG")
    if not streams:
        logger.log_event("No EEG stream found.", level="error")
        pygame.quit(); sys.exit(1)

    # Two independent LSL consumers — LSL is designed for this
    inlet_mi   = StreamInlet(streams[0])
    eeg_state  = EEGStreamState(inlet=inlet_mi,   config=config, mode="motor", logger=logger)
    logger.log_event("MI EEGStreamState created (mode=motor).")

    errp_eeg_state = None
    if ERRP_ENABLE:
        inlet_errp     = StreamInlet(streams[0])
        errp_eeg_state = EEGStreamState(inlet=inlet_errp, config=config, mode="errp", logger=logger)
        logger.log_event("ErrP EEGStreamState created (mode=errp, 1-10 Hz).")
        # Load ErrP model bundle into runtime_common global
        _RC.load_errp_model(errp_model_path)
        if _RC.errp_model is None:
            logger.log_event("ErrP model failed to load — aborting.", level="error")
            pygame.quit(); sys.exit(1)

    trial_sequence = generate_trial_sequence(config.TOTAL_TRIALS, config.MAX_REPEATS)
    logger.log_event(f"Trial sequence: {['MI' if t==0 else 'REST' for t in trial_sequence]}")

    current_trial  = 0
    running        = True
    clock          = pygame.time.Clock()
    predictions_list  = []
    ground_truth_list = []

    # Session-start EA bootstrap for the ErrP head.  Liu-trained bundles
    # all carry feature_spec["ea_alignment"] == True — without a session
    # reference the runtime falls back to an unaligned classify and logs
    # a warning on every epoch.  The fixation loop below keeps both
    # EEGStreamStates hot (MI baseline will be recomputed per trial, but
    # we let its buffer fill here too).
    if ERRP_ENABLE and errp_eeg_state is not None:
        ea_seconds = float(getattr(config, "ERRP_EA_BOOTSTRAP_SEC", 45.0))
        logger.log_event(
            f"ErrP EA bootstrap: {ea_seconds:.0f}s of quiet fixation before first trial."
        )
        boot_start = time.time()
        while time.time() - boot_start < ea_seconds:
            eeg_state.update()
            errp_eeg_state.update()
            screen.fill(config.black)
            draw_fixation_cross(screen_width, screen_height)
            remaining = ea_seconds - (time.time() - boot_start)
            font_sm = pygame.font.SysFont(None, 42)
            surf = font_sm.render(f"EA bootstrap — relax and fixate  ({remaining:4.1f}s)",
                                  True, (180, 180, 180))
            screen.blit(surf, (screen_width // 2 - surf.get_width() // 2,
                               screen_height // 2 + 140))
            pygame.display.flip()
            pygame.event.pump()
            clock.tick(60)
        if not fit_errp_ea_bootstrap(errp_eeg_state):
            logger.log_event(
                "⚠️ EA bootstrap failed — continuing unaligned. "
                "Expect degraded ErrP AUC."
            )

    display_fixation_period(duration=3, eeg_state=eeg_state)

    while running and current_trial < len(trial_sequence):
        logger.log_event(f"--- Trial {current_trial+1}/{len(trial_sequence)} START ---")

        # ── Inter-trial screen ────────────────────────────────────────────────
        screen.fill(config.black)
        draw_fixation_cross(screen_width, screen_height)
        draw_arrow_fill(0, screen_width, screen_height)
        draw_ball_fill(0, screen_width, screen_height)
        draw_time_balls(0, screen_width, screen_height)
        pygame.display.flip()

        # ── Countdown / wait for press ────────────────────────────────────────
        backdoor_mode     = None
        waiting_for_press = True
        countdown_start   = None
        countdown_ms      = 3000

        while waiting_for_press:
            _pump_both(eeg_state, errp_eeg_state)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False; waiting_for_press = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        backdoor_mode = 0
                    elif event.key == pygame.K_DOWN:
                        backdoor_mode = 1
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                    waiting_for_press = False

            if config.TIMING:
                if countdown_start is None:
                    countdown_start = pygame.time.get_ticks()
                if pygame.time.get_ticks() - countdown_start >= countdown_ms:
                    waiting_for_press = False
                draw_time_balls(1, screen_width, screen_height)
                pygame.display.flip()

        if not running:
            break

        mode = backdoor_mode if backdoor_mode is not None else trial_sequence[current_trial]
        logger.log_event(f"Trial mode: {'MI' if mode==0 else 'REST'}")

        # ── MI trigger + feedback ─────────────────────────────────────────────
        if mode == 0:
            send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"],
                             config.UDP_MARKER["PORT"], config.TRIGGERS["MI_BEGIN"], logger=logger)
            if FES_toggle == 1:
                send_udp_message(udp_socket_fes, config.UDP_FES["IP"],
                                 config.UDP_FES["PORT"], "FES_SENS_GO", logger=logger)
        else:
            send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"],
                             config.UDP_MARKER["PORT"], config.TRIGGERS["REST_BEGIN"], logger=logger)

        # ── Compute baseline for both EEGStreamStates ─────────────────────────
        # MI baseline: already available from fixation period
        try:
            eeg_state.compute_baseline(duration_sec=config.BASELINE_DURATION)
        except ValueError as exc:
            logger.log_event(f"⚠️ MI baseline failed: {exc}")
            current_trial += 1
            continue

        if ERRP_ENABLE and errp_eeg_state is not None:
            try:
                errp_eeg_state.compute_baseline(duration_sec=config.BASELINE_DURATION)
            except ValueError as exc:
                logger.log_event(f"⚠️ ErrP baseline failed: {exc} (non-fatal; using zero baseline)")

        # ── MI classification phase ───────────────────────────────────────────
        leaky_integrator = LeakyIntegrator(alpha=config.INTEGRATOR_ALPHA)
        prediction, confidence, leaky_integrator, trial_probs, earlystop_flag = show_feedback(
            duration=config.TIME_MI,
            mode=mode,
            eeg_state=eeg_state,
        )
        logger.log_event(f"MI result: prediction={prediction}, confidence={confidence:.3f}")

        append_trial_probabilities_to_csv(
            trial_probabilities=trial_probs,
            mode=mode,
            trial_number=current_trial + 1,
            predicted_label=prediction,
            early_cutout=earlystop_flag,
            mi_threshold=config.THRESHOLD_MI,
            rest_threshold=config.THRESHOLD_REST,
            logger=logger,
            phase="MI" if mode == 0 else "REST",
        )
        predictions_list.append(prediction)
        ground_truth_list.append(200 if mode == 0 else 100)

        # ── Post-classification robot phase ───────────────────────────────────
        should_move = (mode == 0 and prediction == 200)

        if mode == 0:
            if should_move:
                # ── Correct MI: start robot ───────────────────────────────────
                selected_trajectory = random.choice(config.ROBOT_TRAJECTORY)
                send_udp_message(
                    udp_socket_marker, config.UDP_MARKER["IP"],
                    config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_BEGIN"], logger=logger,
                )
                if FES_toggle == 1:
                    send_udp_message(udp_socket_fes, config.UDP_FES["IP"],
                                     config.UDP_FES["PORT"], "FES_MOTOR_GO", logger=logger)

                # Send trajectory + go
                display_multiple_messages_with_udp(
                    messages=["Correct", "Robot Move"],
                    colors=[config.green, config.green],
                    offsets=[-100, 100],
                    duration=0.01,
                    udp_messages=[selected_trajectory, config.ROBOT_OPCODES["GO"]],
                    udp_socket=udp_socket_robot,
                    udp_ip=config.UDP_ROBOT["IP"],
                    udp_port=config.UDP_ROBOT["PORT"],
                    logger=logger,
                    eeg_state=eeg_state,
                )

                # ── ErrP pause-detect-resume movement phase ───────────────────
                move_result = run_robot_movement_with_errp(
                    eeg_state, errp_eeg_state, leaky_integrator, selected_trajectory,
                )

                # Log ErrP outcome
                if ERRP_ENABLE and move_result["pause_time"] is not None:
                    logger.log_event(
                        f"ErrP outcome: pause_at={move_result['pause_time']:.2f}s  "
                        f"errp_detected={move_result['errp_detected']}  "
                        f"P(error)={move_result['prob_error']}  "
                        f"completed={move_result['completed']}  "
                        f"earlystop={move_result['robot_earlystop']}"
                    )

                # ── Home robot ────────────────────────────────────────────────
                if not move_result["robot_earlystop"]:
                    display_fixation_period(duration=2, eeg_state=eeg_state)
                    send_udp_message(
                        udp_socket_marker, config.UDP_MARKER["IP"],
                        config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_HOME"], logger=logger,
                    )
                    send_udp_message(
                        udp_socket_robot, config.UDP_ROBOT["IP"],
                        config.UDP_ROBOT["PORT"], config.ROBOT_OPCODES["HOME"],
                        logger=logger, expect_ack=True, ack_timeout=1.5, max_retries=1,
                    )
                display_fixation_period(duration=3, eeg_state=eeg_state)

            elif prediction is None:
                # Ambiguous
                display_multiple_messages_with_udp(
                    messages=["Ambiguous", "Robot Stationary"],
                    colors=[config.orange, config.white],
                    offsets=[-100, 100],
                    duration=config.TIME_STATIONARY,
                    udp_messages=None,
                    udp_socket=udp_socket_robot,
                    udp_ip=config.UDP_ROBOT["IP"],
                    udp_port=config.UDP_ROBOT["PORT"],
                    logger=logger, eeg_state=eeg_state,
                )
            else:
                # Incorrect
                display_multiple_messages_with_udp(
                    messages=["Incorrect", "Robot Stationary"],
                    colors=[config.red, config.white],
                    offsets=[-100, 100],
                    duration=config.TIME_STATIONARY,
                    udp_messages=None,
                    udp_socket=udp_socket_robot,
                    udp_ip=config.UDP_ROBOT["IP"],
                    udp_port=config.UDP_ROBOT["PORT"],
                    logger=logger, eeg_state=eeg_state,
                )

        else:   # REST mode — identical to ExperimentDriver_Online
            send_udp_message(
                udp_socket_marker, config.UDP_MARKER["IP"],
                config.UDP_MARKER["PORT"], config.TRIGGERS["REST_END"], logger=logger,
            )
            label = ("Correct" if prediction == 100 else
                     "Ambiguous" if prediction is None else "Incorrect")
            color = (config.green if prediction == 100 else
                     config.orange if prediction is None else config.red)
            display_multiple_messages_with_udp(
                messages=[label, "Robot Stationary"],
                colors=[color, config.white],
                offsets=[-100, 100],
                duration=config.TIME_STATIONARY,
                udp_messages=None,
                udp_socket=udp_socket_robot,
                udp_ip=config.UDP_ROBOT["IP"],
                udp_port=config.UDP_ROBOT["PORT"],
                logger=logger, eeg_state=eeg_state,
            )

        # ── Trial summary log ─────────────────────────────────────────────────
        logger.log_trial_summary(
            trial_number=current_trial + 1,
            true_label=200 if mode == 0 else 100,
            predicted_label=prediction,
            early_cutout=earlystop_flag,
            accuracy_threshold=config.THRESHOLD_MI if mode == 0 else config.THRESHOLD_REST,
            confidence=confidence,
            num_predictions=len(trial_probs),
        )

        display_fixation_period(duration=3, eeg_state=eeg_state)
        logger.log_event(f"Trial {current_trial+1} complete.")
        current_trial += 1
        clock.tick(60)

    # ── Save adaptive transforms ──────────────────────────────────────────────
    if current_trial == len(trial_sequence) and config.SAVE_ADAPTIVE_T:
        try:
            save_transform(
                _RC.Prev_T, _RC.counter, adaptive_T_path,
                T_beta=_RC.Prev_T_beta, counter_beta=_RC.counter_beta,
            )
        except Exception as exc:
            logger.log_event(f"⚠️ Could not save adaptive transform: {exc}")

    log_confusion_matrix_from_trial_summary(logger)
    logger.log_event("ExperimentDriver_ErrP_Online complete.")
    pygame.quit()


if __name__ == "__main__":
    main()
