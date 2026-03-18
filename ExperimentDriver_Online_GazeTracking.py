import pygame
import socket
import pickle
import datetime
import os
import random
import time
import json
from pylsl import StreamInlet, resolve_stream
import numpy as np

# MNE for real-time EEG processing
import mne
mne.set_log_level("WARNING")  # Options: "ERROR", "WARNING", "INFO", "DEBUG"

# Visualization utilities
from Utils.visualization import (
    draw_arrow_fill,
    draw_ball_fill,
    draw_fixation_cross,
    draw_time_balls,
    draw_progress_bar
)

# Experiment utilities
from Utils.experiment_utils import (
    generate_trial_sequence,
    save_transform,
    load_transform
)

# EEG stream handler
from Utils.EEGStreamState import EEGStreamState

# Networking utilities
from Utils.networking import send_udp_message, display_multiple_messages_with_udp

# Configuration parameters
import config

from pathlib import Path
from Utils.logging_manager import LoggerManager

# Common runtime functions
from Utils.runtime_common import (
    log_confusion_matrix_from_trial_summary,
    append_trial_probabilities_to_csv,
    display_fixation_period,
    hold_messages_and_classify,
    show_feedback,
)

# Also import the module itself for wiring globals
import Utils.runtime_common as _RC

# =========================================================
# Logger setup
# =========================================================
logger = LoggerManager.auto_detect_from_subject(
    subject=config.TRAINING_SUBJECT,
    base_path=Path(config.DATA_DIR),
    mode="online"
)

# Log experiment configuration snapshot
loggable_fields = [
    "UDP_MARKER", "UDP_ROBOT", "UDP_FES",
    "ARM_SIDE", "TOTAL_TRIALS", "MAX_REPEATS",
    "TIME_MI", "TIME_ROB", "TIME_STATIONARY",
    "SHAPE_MAX", "SHAPE_MIN", "ROBOT_TRAJECTORY",
    "FES_toggle", "FES_CHANNEL", "FES_TIMING_OFFSET",
    "WORKING_DIR", "DATA_DIR", "MODEL_PATH",
    "TRAINING_SUBJECT",
    "MOTOR_CHANNEL_NAMES", "CLASSIFY_WINDOW", "ACCURACY_THRESHOLD", "THRESHOLD_MI", "THRESHOLD_REST",
    "RELAXATION_RATIO", "MIN_PREDICTIONS", "SURFACE_LAPLACIAN_TOGGLE",
    "SELECT_MOTOR_CHANNELS", "INTEGRATOR_ALPHA", "SHRINKAGE_PARAM",
    "LEDOITWOLF", "RECENTERING", "UPDATE_DURING_MOVE",
    "GAZE_UDP_IP", "GAZE_UDP_PORT", "GAZE_SELECTION_WINDOW", "GAZE_AVG_WINDOW",
    "GAZE_MIN_DWELL_SEC", "GAZE_UDP_TIMEOUT", "GO_NOGO_PROMPT_SEC",
    "POSE_LIBRARY_PATH", "GAZE_SAMPLE_WIDTH", "GAZE_SAMPLE_HEIGHT",
    "ROBOT_MOVE_DUR", "SELECTION_TO_DECISION_FIXATION_SEC",
    "DECISION_SCREEN_SEC", "MAX_DECISION_ATTEMPTS",
    "PRETRIAL_NEUTRAL_SEC", "PRETRIAL_WHITE_ORB_SEC", "BASELINE_BEFORE_CUE_SEC",
    "RETRY_RESET_SEC"
]
config_log_subset = {
    key: getattr(config, key) for key in loggable_fields if hasattr(config, key)
}
logger.save_config_snapshot(config_log_subset)

eeg_dir = logger.log_base / "eeg"
adaptive_T_path = eeg_dir / "adaptive_T.pkl"

Prev_T, counter = load_transform(adaptive_T_path)
if Prev_T is None:
    counter = 0
    logger.log_event("ℹ️ No adaptive transform found — starting fresh with counter = 0.")
else:
    logger.log_event(f"✅ Loaded adaptive transform with counter = {counter}")

logger.log_event("Logger initialized for online experimental driver.")

# =========================================================
# Pygame setup
# =========================================================
pygame.init()

if config.BIG_BROTHER_MODE:
    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
    screen = pygame.display.set_mode((1920, 1080), pygame.NOFRAME)
    logger.log_event("🎥 Big Brother Mode ON — window placed at (0,0) on external monitor (HDMI-1).")
else:
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    logger.log_event("👤 Big Brother Mode OFF — fullscreen on active display.")

pygame.display.set_caption("EEG Online Interactive Loop")
info = pygame.display.Info()
screen_width = info.current_w
screen_height = info.current_h
logger.log_event("Pygame initialized and display configured.")

# =========================================================
# UDP sockets
# =========================================================
udp_socket_marker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket_robot = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket_fes = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
logger.log_event("UDP sockets initialized for marker, robot, and FES channels.")

FES_toggle = config.FES_toggle
logger.log_event(f"FES toggle status: {'Enabled' if FES_toggle else 'Disabled'}.")

# =========================================================
# Model loading
# =========================================================
subject_model_dir = os.path.join(config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "models")
subject_model_path = os.path.join(subject_model_dir, f"sub-{config.TRAINING_SUBJECT}_model.pkl")

try:
    with open(subject_model_path, 'rb') as f:
        model = pickle.load(f)
    logger.log_event(f"✅ Model successfully loaded from: {subject_model_path}")
except FileNotFoundError:
    logger.log_event(
        f"❌ Error: Model file '{subject_model_path}' not found. Ensure the model has been trained.",
        level="error"
    )
    exit(1)

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

# =========================================================
# Runtime globals
# =========================================================
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
_RC.udp_socket_robot = udp_socket_robot
_RC.udp_socket_fes = udp_socket_fes

_RC.FES_toggle = FES_toggle

_RC.Prev_T = Prev_T
_RC.counter = counter

# =========================================================
# Gaze experiment helpers
# =========================================================
GAZE_UDP_IP = config.GAZE_UDP_IP
GAZE_UDP_PORT = config.GAZE_UDP_PORT
GAZE_SELECTION_WINDOW = config.GAZE_SELECTION_WINDOW
GAZE_AVG_WINDOW = config.GAZE_AVG_WINDOW
GAZE_MIN_DWELL_SEC = config.GAZE_MIN_DWELL_SEC
GAZE_UDP_TIMEOUT = config.GAZE_UDP_TIMEOUT
GO_NOGO_PROMPT_SEC = config.GO_NOGO_PROMPT_SEC
POSE_LIBRARY_PATH = config.POSE_LIBRARY_PATH
GAZE_SAMPLE_WIDTH = float(config.GAZE_SAMPLE_WIDTH)
GAZE_SAMPLE_HEIGHT = float(config.GAZE_SAMPLE_HEIGHT)
ROBOT_MOVE_DUR = float(config.ROBOT_MOVE_DUR)

SELECTION_TO_DECISION_FIXATION_SEC = float(getattr(config, "SELECTION_TO_DECISION_FIXATION_SEC", 1.5))
DECISION_SCREEN_SEC = float(getattr(config, "DECISION_SCREEN_SEC", GO_NOGO_PROMPT_SEC))
MAX_DECISION_ATTEMPTS = int(getattr(config, "MAX_DECISION_ATTEMPTS", 2))

PRETRIAL_NEUTRAL_SEC = float(getattr(config, "PRETRIAL_NEUTRAL_SEC", 4.0))
PRETRIAL_WHITE_ORB_SEC = float(getattr(config, "PRETRIAL_WHITE_ORB_SEC", 3.0))
BASELINE_BEFORE_CUE_SEC = float(getattr(config, "BASELINE_BEFORE_CUE_SEC", 1.0))
RETRY_RESET_SEC = float(getattr(config, "RETRY_RESET_SEC", 2.0))

# =========================================================
# Drawing helpers
# =========================================================
def draw_centered_text(text, y, color=config.white, font_size=48):
    font = pygame.font.SysFont(None, font_size)
    surf = font.render(text, True, color)
    rect = surf.get_rect(center=(screen_width // 2, y))
    screen.blit(surf, rect)


def draw_loading_bar(progress, y=None, width_ratio=0.55, height=28,
                     fill_color=None, bg_color=(60, 60, 60), border_color=config.white):
    if fill_color is None:
        fill_color = config.green

    progress = max(0.0, min(1.0, float(progress)))

    bar_width = int(screen_width * width_ratio)
    bar_x = (screen_width - bar_width) // 2
    bar_y = y if y is not None else (screen_height // 2 + 130)

    outer_rect = pygame.Rect(bar_x, bar_y, bar_width, height)
    inner_rect = pygame.Rect(bar_x, bar_y, int(bar_width * progress), height)

    pygame.draw.rect(screen, bg_color, outer_rect)
    pygame.draw.rect(screen, fill_color, inner_rect)
    pygame.draw.rect(screen, border_color, outer_rect, 2)


def draw_selection_screen(progress, leading_obj_name=None):
    screen.fill(config.black)
    #draw_fixation_cross(screen_width, screen_height)

    draw_centered_text("Select an object with your gaze", screen_height // 2 - 160, config.white, 56)

    if leading_obj_name is not None:
        draw_centered_text(f"Current selection: {leading_obj_name}", screen_height // 2 - 20, config.green, 44)
    else:
        draw_centered_text("Current selection: None", screen_height // 2 - 20, config.orange, 44)

    draw_centered_text("Keep looking steadily at the object you want.", screen_height // 2 + 60, config.white, 36)
    draw_loading_bar(progress=progress, y=screen_height // 2 + 130)

    pygame.display.flip()


def draw_trial_onset_screen(selected_name, mode):
    """
    Trial onset screen:
    - shown exactly at cue onset / trial start
    - no separate waiting period before classification
    - words are color-matched to the cue
    """
    screen.fill(config.black)
    draw_fixation_cross(screen_width, screen_height)

    # Keep the selected object visible
    draw_centered_text(f"Selected object: {selected_name}", screen_height // 2 - 130, config.white, 54)

    if mode == 0:
        cue_color = config.red
        headline = "CONFIRM MOVEMENT"
        subtext = "Perform motor imagery now"
    else:
        cue_color = getattr(config, "blue", config.white)
        headline = "CANCEL MOVEMENT"
        subtext = "Return to rest now"

    draw_centered_text(headline, screen_height // 2 + 40, cue_color, 60)
    draw_centered_text(subtext, screen_height // 2 + 95, cue_color, 42)

    pygame.display.flip()


def draw_neutral_trial_prep_screen(white_orb_on=False, message="Fixate and prepare"):
    screen.fill(config.black)
    draw_fixation_cross(screen_width, screen_height)

    # Empty shapes / neutral layout
    draw_arrow_fill(0, screen_width, screen_height)
    draw_ball_fill(0, screen_width, screen_height)

    # White orb indicator: turn on during final 3 sec
    orb_color = config.white if white_orb_on else (90, 90, 90)
    orb_radius = 18
    orb_x = screen_width // 2
    orb_y = screen_height // 2 - 180
    pygame.draw.circle(screen, orb_color, (orb_x, orb_y), orb_radius)

    draw_centered_text(message, screen_height // 2 + 90, config.white, 38)
    pygame.display.flip()


def draw_retry_reset_screen():
    screen.fill(config.black)
    draw_fixation_cross(screen_width, screen_height)

    draw_arrow_fill(0, screen_width, screen_height)
    draw_ball_fill(0, screen_width, screen_height)

    orb_x = screen_width // 2
    orb_y = screen_height // 2 + 180
    pygame.draw.circle(screen, config.white, (orb_x, orb_y), 18)

    draw_centered_text("Relax and fixate", screen_height // 2 + 90, config.white, 42)
    draw_centered_text("Prepare for retry", screen_height // 2 + 140, config.white, 34)

    pygame.display.flip()

# =========================================================
# Gaze helpers
# =========================================================
def gaze_udp_request(sock, payload, timeout=GAZE_UDP_TIMEOUT):
    """
    Request a gaze snapshot from `gaze_runner.py` via UDP.

    Args:
        sock: UDP socket connected to the gaze runner's request port.
        payload: JSON-serializable dict. Expected keys:
          - `cmd`: currently `"snapshot"`
          - `include_objects`: bool (whether tracked detections are included)
          - `query_id`: opaque value used only for debugging/logging
        timeout: recv timeout in seconds.

    Returns:
        Decoded JSON response dict, or `None` on failure.

    Expected snapshot fields (when `cmd="snapshot"`):
        - `ok`: bool (true if snapshot is valid)
        - `gaze_px`: (x_px, y_px) in Neon scene pixel coordinates
          (the experiment later normalizes using `GAZE_SAMPLE_WIDTH/HEIGHT`)
        - `gaze_hit`: None or dict describing the selected tracked object
        - `objects`: list of tracked detections (only present when `include_objects=True`)
    """
    try:
        sock.settimeout(timeout)
        msg = json.dumps(payload).encode("utf-8")
        sock.sendto(msg, (GAZE_UDP_IP, GAZE_UDP_PORT))
        data, _ = sock.recvfrom(65535)
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        logger.log_event(f"⚠️ Gaze UDP request failed: {e}")
        return None


def make_object_key(hit):
    if not hit:
        return None

    name = str(hit.get("name", "unknown"))
    track_id = hit.get("track_id", None)

    if track_id is None or int(track_id) < 0:
        return f"{name}"
    return f"{name}#{int(track_id)}"


def object_display_name(obj_key):
    if obj_key is None:
        return None
    return obj_key.split("#")[0]


def run_gaze_selection_window(gaze_sock, eeg_state, duration_s=GAZE_SELECTION_WINDOW):
    logger.log_event(f"Starting gaze selection window ({duration_s:.2f}s).")

    dwell_sec_by_obj = {}
    valid_samples_by_obj = {}
    last_loop_t = time.time()

    selection_start = time.time()
    running_window = True

    while running_window:
        eeg_state.update()

        now = time.time()
        dt = max(0.0, now - last_loop_t)
        last_loop_t = now
        elapsed = now - selection_start
        progress = elapsed / duration_s if duration_s > 0 else 1.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return None

        snap = gaze_udp_request(
            gaze_sock,
            {
                "cmd": "snapshot",
                "include_objects": True,
                "query_id": f"sel_{int(now * 1000)}"
            }
        )

        leading_obj_name = None

        if snap and snap.get("ok", False):
            hit = snap.get("gaze_hit", None)
            gaze_px = snap.get("gaze_px", None)

            if hit is not None and gaze_px is not None and len(gaze_px) >= 2:
                obj_key = make_object_key(hit)
                # `gaze_px` are raw scene pixels; normalization happens later when mapping to the pose library.
                x, y = float(gaze_px[0]), float(gaze_px[1])

                dwell_sec_by_obj[obj_key] = dwell_sec_by_obj.get(obj_key, 0.0) + dt
                valid_samples_by_obj.setdefault(obj_key, []).append((now, x, y))

            if dwell_sec_by_obj:
                leading_key = max(dwell_sec_by_obj, key=dwell_sec_by_obj.get)
                leading_obj_name = object_display_name(leading_key)

        draw_selection_screen(progress=progress, leading_obj_name=leading_obj_name)

        if elapsed >= duration_s:
            running_window = False

    if not dwell_sec_by_obj:
        logger.log_event("No gaze selection detected in window.")
        return {
            "selected_key": None,
            "selected_name": None,
            "avg_px": None,
            "dwell_sec_by_obj": {},
            "samples_used": 0,
            "selection_attempt_success": False,
        }

    selected_key = max(dwell_sec_by_obj, key=dwell_sec_by_obj.get)
    selected_dwell = dwell_sec_by_obj[selected_key]

    if selected_dwell < GAZE_MIN_DWELL_SEC:
        logger.log_event(
            f"No valid object passed dwell threshold. Best={selected_key}, dwell={selected_dwell:.3f}s, "
            f"threshold={GAZE_MIN_DWELL_SEC:.3f}s"
        )
        return {
            "selected_key": None,
            "selected_name": None,
            "avg_px": None,
            "dwell_sec_by_obj": dwell_sec_by_obj,
            "samples_used": 0,
            "selection_attempt_success": False,
        }

    selected_name = object_display_name(selected_key)
    end_t = time.time()
    recent_cutoff = end_t - GAZE_AVG_WINDOW

    all_samples = valid_samples_by_obj.get(selected_key, [])
    recent_samples = [(t, x, y) for (t, x, y) in all_samples if t >= recent_cutoff]
    samples_for_avg = recent_samples if len(recent_samples) > 0 else all_samples

    if len(samples_for_avg) == 0:
        logger.log_event(f"Selected object {selected_name}, but no valid gaze samples were available for averaging.")
        return {
            "selected_key": None,
            "selected_name": None,
            "avg_px": None,
            "dwell_sec_by_obj": dwell_sec_by_obj,
            "samples_used": 0,
            "selection_attempt_success": False,
        }

    avg_x = sum(x for _, x, _ in samples_for_avg) / len(samples_for_avg)
    avg_y = sum(y for _, _, y in samples_for_avg) / len(samples_for_avg)

    logger.log_event(
        f"Gaze selection success — object={selected_name}, dwell={selected_dwell:.3f}s, "
        f"avg_px=({avg_x:.1f}, {avg_y:.1f}), samples_used={len(samples_for_avg)}"
    )

    return {
        "selected_key": selected_key,
        "selected_name": selected_name,
        "avg_px": (avg_x, avg_y),
        "dwell_sec_by_obj": dwell_sec_by_obj,
        "samples_used": len(samples_for_avg),
        "selection_attempt_success": True,
    }


def repeat_until_valid_selection(gaze_sock, eeg_state):
    attempt = 0
    while True:
        attempt += 1
        logger.log_event(f"Selection attempt #{attempt} starting.")
        result = run_gaze_selection_window(gaze_sock, eeg_state, duration_s=GAZE_SELECTION_WINDOW)

        if result is None:
            return None

        if result["selection_attempt_success"]:
            result["selection_attempt"] = attempt
            return result

        logger.log_event(f"Selection attempt #{attempt} failed — repeating selection window.")

# =========================================================
# Pose library helpers
# =========================================================
def load_pose_library(path):
    if path is None:
        raise ValueError("POSE_LIBRARY_PATH is not set in config.")

    z = np.load(path, allow_pickle=True)
    X = z["X"]
    Q = z["Q"]
    G = z["G"] if "G" in z.files else None

    if G is None or G.shape[1] < 2:
        raise ValueError(f"Pose library at {path} does not contain valid gaze matrix G.")

    logger.log_event(
        f"Loaded pose library from {path} | X.shape={X.shape}, Q.shape={Q.shape}, G.shape={G.shape}"
    )
    return X, Q, G


def nearest_idx_gaze(G, gaze_xy_norm):
    gx = G[:, 0]
    gy = G[:, 1]
    valid_mask = np.isfinite(gx) & np.isfinite(gy)

    if not np.any(valid_mask):
        raise ValueError("Pose library G contains no valid gaze rows.")

    g_valid = np.column_stack([gx[valid_mask], gy[valid_mask]])
    target = np.asarray(gaze_xy_norm, dtype=float).ravel()

    d = np.linalg.norm(g_valid - target[None, :], axis=1)
    best_local = int(np.argmin(d))
    valid_indices = np.flatnonzero(valid_mask)
    idx_global = int(valid_indices[best_local])
    return idx_global, float(d[best_local])


def build_joint_command(q_target, dur_s):
    return ",".join(f"{v:.6f}" for v in np.asarray(q_target).tolist()) + f";dur={dur_s:.3f}"


def resolve_robot_target_from_gaze(avg_px, selected_name, X_lib, Q_lib, G_lib):
    if avg_px is None or len(avg_px) < 2:
        raise ValueError("avg_px is invalid for pose lookup.")

    x_px, y_px = float(avg_px[0]), float(avg_px[1])
    x_norm = x_px / GAZE_SAMPLE_WIDTH
    y_norm = y_px / GAZE_SAMPLE_HEIGHT

    idx, dist = nearest_idx_gaze(G_lib, (x_norm, y_norm))
    q_target = Q_lib[idx]
    x_target = X_lib[idx]
    joint_cmd = build_joint_command(q_target, ROBOT_MOVE_DUR)

    logger.log_event(
        f"Gaze NN lookup — object={selected_name}, "
        f"avg_px=({x_px:.1f},{y_px:.1f}), "
        f"gaze_norm=({x_norm:.3f},{y_norm:.3f}), "
        f"idx={idx}, dist={dist:.4f}, "
        f"x_target={np.round(x_target, 1).tolist()}"
    )

    return {
        "idx": idx,
        "dist": dist,
        "gaze_norm": (x_norm, y_norm),
        "q_target": q_target,
        "x_target": x_target,
        "joint_cmd": joint_cmd,
    }

# =========================================================
# Decision-phase helpers
# =========================================================
def run_pretrial_neutral_phase(eeg_state, duration_s, white_orb_last_s, baseline_sec):
    """
    Neutral pre-trial phase with:
      - fixation cross
      - empty shapes
      - white orb turns on during final `white_orb_last_s`
      - baseline computed from final `baseline_sec`
    """
    logger.log_event(
        f"Pre-trial neutral phase: duration={duration_s:.2f}s, "
        f"white_orb_last_s={white_orb_last_s:.2f}s, baseline_sec={baseline_sec:.2f}s"
    )

    start_t = time.time()
    end_t = start_t + duration_s

    while time.time() < end_t:
        eeg_state.update()
        remaining = end_t - time.time()
        white_orb_on = remaining <= white_orb_last_s

        draw_neutral_trial_prep_screen(
            white_orb_on=white_orb_on,
            message="Fixate and prepare"
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

    try:
        eeg_state.compute_baseline(duration_sec=baseline_sec)
        logger.log_event(
            f"Computed baseline from final neutral pre-trial window: "
            f"shape={eeg_state.baseline_mean.shape}, duration={baseline_sec}s"
        )
    except ValueError as e:
        logger.log_event(f"⚠️ Could not compute baseline from neutral pre-trial window: {e}")
        return False

    return True


def run_retry_reset_phase(eeg_state, duration_s, baseline_sec):
    """
    Retry reset phase:
      - fixation cross
      - empty shapes
      - explicit relax / fixate message
      - recompute baseline at end
    """
    logger.log_event(
        f"Retry reset phase: duration={duration_s:.2f}s, baseline_sec={baseline_sec:.2f}s"
    )

    start_t = time.time()
    end_t = start_t + duration_s

    while time.time() < end_t:
        eeg_state.update()
        draw_retry_reset_screen()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

    try:
        eeg_state.compute_baseline(duration_sec=baseline_sec)
        logger.log_event(
            f"Computed baseline from retry reset phase: "
            f"shape={eeg_state.baseline_mean.shape}, duration={baseline_sec}s"
        )
    except ValueError as e:
        logger.log_event(f"⚠️ Could not compute baseline from retry reset phase: {e}")
        return False

    return True


def run_decision_attempt(selected_name, mode, eeg_state, trial_number, decision_attempt):
    """
    One decision attempt:
      - first attempt:
          neutral pre-trial phase -> baseline -> cue onset/trial onset -> classification
      - retry attempt:
          retry reset phase -> baseline -> cue onset/trial onset -> classification

    Important:
      - there is NO separate decision-screen dwell anymore
      - cue onset is the start of the actual trial
    """
    if decision_attempt == 1:
        ok = run_pretrial_neutral_phase(
            eeg_state=eeg_state,
            duration_s=PRETRIAL_NEUTRAL_SEC,
            white_orb_last_s=PRETRIAL_WHITE_ORB_SEC,
            baseline_sec=BASELINE_BEFORE_CUE_SEC
        )
        if not ok:
            return None
    else:
        ok = run_retry_reset_phase(
            eeg_state=eeg_state,
            duration_s=RETRY_RESET_SEC,
            baseline_sec=BASELINE_BEFORE_CUE_SEC
        )
        if not ok:
            return None

    # Trial onset = cue onset
    #draw_trial_onset_screen(selected_name, mode)

    logger.log_event(
        f"Starting decision attempt {decision_attempt} at cue onset — "
        f"Mode: {'GO/MI' if mode == 0 else 'NO-GO/REST'} | selected_object={selected_name}"
    )

    headline = "CONFIRM MOVEMENT" if mode == 0 else "CANCEL MOVEMENT"
    instruction = "Perform motor imagery now" if mode == 0 else "Return to rest now"

    prediction, confidence, leaky_integrator, trial_probs, earlystop_flag = show_feedback(
        duration=config.TIME_MI,
        mode=mode,
        eeg_state=eeg_state,
        headline_text=headline,
        subtext=instruction,
        object_text=f"Selected object: {selected_name}"
    )

    pygame.display.flip()
    pygame.event.get()

    logger.log_event(
        f"Decision attempt {decision_attempt} result — Predicted: {prediction}, "
        f"Ground Truth: {200 if mode == 0 else 100}"
    )

    append_trial_probabilities_to_csv(
        trial_probabilities=trial_probs,
        mode=mode,
        trial_number=trial_number,
        predicted_label=prediction,
        early_cutout=earlystop_flag,
        mi_threshold=config.THRESHOLD_MI,
        rest_threshold=config.THRESHOLD_REST,
        logger=logger,
        phase=f"{'GO' if mode == 0 else 'NO_GO'}_ATTEMPT_{decision_attempt}"
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "leaky_integrator": leaky_integrator,
        "trial_probs": trial_probs,
        "earlystop_flag": earlystop_flag,
        "aborted": False
    }

# =========================================================
# Main
# =========================================================
def main():
    logger.log_event("Resolving EEG data stream via LSL...")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    logger.log_event("✅ EEG stream detected and inlet established.")

    eeg_state = EEGStreamState(inlet=inlet, config=config, logger=logger)
    logger.log_event("EEGStreamState object created — ready to pull and process data.")

    gaze_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    logger.log_event(f"Gaze UDP client ready for {GAZE_UDP_IP}:{GAZE_UDP_PORT}")

    X_lib, Q_lib, G_lib = load_pose_library(POSE_LIBRARY_PATH)

    trial_sequence = generate_trial_sequence(
        total_trials=config.TOTAL_TRIALS,
        max_repeats=config.MAX_REPEATS
    )
    mode_labels = ["GO_MI" if t == 0 else "NO_GO_REST" for t in trial_sequence]
    logger.log_event(f"Trial Sequence generated: {trial_sequence}")
    logger.log_event(f"Trial Sequence (labeled): {mode_labels}")
    current_trial = 0

    running = True
    clock = pygame.time.Clock()

    display_fixation_period(duration=3, eeg_state=eeg_state)
    logger.log_event("Initial fixation period complete. Beginning experimental loop.")

    while running and current_trial < len(trial_sequence):
        logger.log_event(f"--- Trial {current_trial + 1}/{len(trial_sequence)} START ---")

        mode = trial_sequence[current_trial]
        logger.log_event(f"Trial type: {'GO (MI)' if mode == 0 else 'NO-GO (REST)'}")

        # ---------------------------------
        # 1. Gaze selection phase
        # ---------------------------------
        selection_result = repeat_until_valid_selection(gaze_sock, eeg_state)
        if selection_result is None:
            logger.log_event("Experiment terminated during selection phase.")
            running = False
            break

        selected_name = selection_result["selected_name"]
        avg_px = selection_result["avg_px"]
        selection_attempt = selection_result["selection_attempt"]

        logger.log_event(
            f"Final selected object for trial {current_trial+1}: {selected_name} | "
            f"avg_px={avg_px} | selection_attempt={selection_attempt}"
        )

        # ---------------------------------
        # 2. Decision phase with ambiguous retry
        # ---------------------------------
        decision_attempt = 1
        ambiguous_retry_used = False
        decision_result = None

        while decision_attempt <= MAX_DECISION_ATTEMPTS:
            decision_result = run_decision_attempt(
                selected_name=selected_name,
                mode=mode,
                eeg_state=eeg_state,
                trial_number=current_trial + 1,
                decision_attempt=decision_attempt
            )

            if decision_result is None:
                logger.log_event("Experiment terminated during decision phase.")
                running = False
                break

            if decision_result.get("aborted", False):
                logger.log_event("Decision attempt aborted due to baseline failure.")
                break

            prediction = decision_result["prediction"]

            if prediction is not None:
                break

            if decision_attempt < MAX_DECISION_ATTEMPTS:
                ambiguous_retry_used = True
                logger.log_event(
                    f"Ambiguous result on decision attempt {decision_attempt}. "
                    f"Re-running decision phase with reset screen and fresh baseline."
                )
                decision_attempt += 1
                continue

            logger.log_event("Decision remained ambiguous after all allowed attempts.")
            break

        if not running:
            break

        if decision_result is None:
            break

        prediction = decision_result["prediction"]
        confidence = decision_result["confidence"]
        leaky_integrator = decision_result["leaky_integrator"]
        trial_probs = decision_result["trial_probs"]
        earlystop_flag = decision_result["earlystop_flag"]

        predictions_list.append(prediction)
        ground_truth_list.append(200 if mode == 0 else 100)

        # ---------------------------------
        # 3. Outcome / robot action
        # ---------------------------------
        should_hold_and_classify = False

        if mode == 0:
            if prediction == 200:
                messages = ["Confirmed", f"Move to {selected_name}"]
                colors = [config.green, config.green]
                offsets = [-100, 100]

                target_info = resolve_robot_target_from_gaze(
                    avg_px=avg_px,
                    selected_name=selected_name,
                    X_lib=X_lib,
                    Q_lib=Q_lib,
                    G_lib=G_lib,
                )

                udp_messages = [target_info["joint_cmd"], config.ROBOT_OPCODES["GO"]]
                duration = 0.01
                should_hold_and_classify = True

                logger.log_event(
                    f"GO trial success — selected_object={selected_name}, avg_px={avg_px}, "
                    f"gaze_norm={target_info['gaze_norm']}, "
                    f"pose_idx={target_info['idx']}, nn_dist={target_info['dist']:.4f}, "
                    f"x_target={np.round(target_info['x_target'], 1).tolist()}"
                )

                if FES_toggle == 1:
                    send_udp_message(
                        udp_socket_fes,
                        config.UDP_FES["IP"],
                        config.UDP_FES["PORT"],
                        "FES_MOTOR_GO",
                        logger=logger
                    )
                else:
                    logger.log_event("FES disabled — skipping motor stimulation.")

                send_udp_message(
                    udp_socket_marker,
                    config.UDP_MARKER["IP"],
                    config.UDP_MARKER["PORT"],
                    config.TRIGGERS["ROBOT_BEGIN"],
                    logger=logger
                )

            elif prediction is None:
                messages = ["Ambiguous", "Robot Stationary"]
                colors = [config.orange, config.white]
                offsets = [-100, 100]
                udp_messages = None
                duration = config.TIME_STATIONARY
                logger.log_event("GO trial ambiguous after all attempts — robot remains stationary.")

            else:
                messages = ["Incorrect", "Robot Stationary"]
                colors = [config.red, config.white]
                offsets = [-100, 100]
                udp_messages = None
                duration = config.TIME_STATIONARY
                logger.log_event("GO trial incorrect — robot remains stationary.")

        else:
            if prediction == 100:
                messages = ["Cancelled", "Robot Stationary"]
                colors = [config.green, config.green]
                offsets = [-100, 100]
                udp_messages = None
                duration = config.TIME_STATIONARY
                logger.log_event("NO-GO trial correct — movement cancelled.")

            elif prediction is None:
                messages = ["Ambiguous", "Movement Cancelled"]
                colors = [config.orange, config.white]
                offsets = [-100, 100]
                udp_messages = None
                duration = config.TIME_STATIONARY
                logger.log_event("NO-GO trial ambiguous after all attempts — robot remains stationary.")

            else:
                messages = ["Incorrect", "False Move Attempt"]
                colors = [config.red, config.white]
                offsets = [-100, 100]
                udp_messages = None
                duration = config.TIME_STATIONARY
                logger.log_event("NO-GO trial incorrect — classifier produced a false positive.")

            should_hold_and_classify = False

        logger.log_event(
            f"Displaying feedback: '{messages[0]}' | Action: '{messages[1]}' | Duration: {duration}s"
        )

        display_multiple_messages_with_udp(
            messages=messages,
            colors=colors,
            offsets=offsets,
            duration=duration,
            udp_messages=udp_messages,
            udp_socket=udp_socket_robot,
            udp_ip=config.UDP_ROBOT["IP"],
            udp_port=config.UDP_ROBOT["PORT"],
            logger=logger,
            eeg_state=eeg_state
        )

        # ---------------------------------
        # 4. Continue classification during robot movement for successful GO trials
        # ---------------------------------
        if should_hold_and_classify:
            logger.log_event("Entering real-time classification window during robot movement...")
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

            if not robot_earlystop:
                logger.log_event("Robot fixation period 2s before homing.")
                display_fixation_period(duration=2, eeg_state=eeg_state)

                send_udp_message(
                    udp_socket_marker,
                    config.UDP_MARKER["IP"],
                    config.UDP_MARKER["PORT"],
                    config.TRIGGERS["ROBOT_HOME"],
                    logger=logger
                )

                acked, _ = send_udp_message(
                    udp_socket_robot,
                    config.UDP_ROBOT["IP"],
                    config.UDP_ROBOT["PORT"],
                    config.ROBOT_OPCODES["HOME"],
                    logger=logger,
                    expect_ack=True,
                    ack_timeout=1.0,
                    max_retries=1
                )
                logger.log_event(f"Sent HOME opcode to robot at end of GO trial. acked={acked}")

            display_fixation_period(duration=3, eeg_state=eeg_state)
            logger.log_event("Robot reset fixation (3s) complete.")

        # ---------------------------------
        # 5. Trial summary + inter-trial rest
        # ---------------------------------
        logger.log_trial_summary(
            trial_number=current_trial + 1,
            true_label=200 if mode == 0 else 100,
            predicted_label=prediction,
            early_cutout=earlystop_flag,
            accuracy_threshold=config.THRESHOLD_MI if mode == 0 else config.THRESHOLD_REST,
            confidence=confidence,
            num_predictions=len(trial_probs)
        )

        logger.log_event(
            f"Trial {current_trial+1} summary extras — selected_object={selected_name}, "
            f"avg_px={avg_px}, selection_attempt={selection_attempt}, "
            f"samples_used={selection_result['samples_used']}, "
            f"decision_attempts_used={decision_attempt}, "
            f"ambiguous_retry_used={ambiguous_retry_used}"
        )

        display_fixation_period(duration=3, eeg_state=eeg_state)
        logger.log_event(f"Trial {current_trial+1} complete. Proceeding to next.")

        current_trial += 1
        pygame.display.flip()
        clock.tick(60)

    if current_trial == len(trial_sequence) and config.SAVE_ADAPTIVE_T:
        try:
            save_transform(Prev_T, counter, adaptive_T_path)
        except Exception as e:
            logger.log_event(f"⚠️ Could not save transform to {adaptive_T_path}: {e}")

    log_confusion_matrix_from_trial_summary(logger)
    logger.log_event("run complete")

    try:
        gaze_sock.close()
    except Exception:
        pass

    pygame.quit()


if __name__ == "__main__":
    main()