# Configuration for EEG experiments — runtime drivers, utilities, and training schemes.
# Visualization and analysis scripts should not read from this file; they own their own params.
import os

# =============================================================================
# Paths and subject
# =============================================================================
# WORKING_DIR / DATA_DIR are machine-local — set them in config_local.py.
# The defaults here are sentinels; a fresh checkout still imports cleanly,
# but realtime code that depends on these paths will fail loudly until
# config_local.py is created (see config_local.example.py).
WORKING_DIR = ""
DATA_DIR = ""
TRAINING_SUBJECT = "PILOT007"

# =============================================================================
# EEG acquisition and channels
# =============================================================================
FS = 512  # Sampling frequency (Hz)
CAP_TYPE = 32
LOWCUT = 8   # Hz — motor imagery mu band
HIGHCUT = 13
LOWCUT_ERRP = 1   # Hz — ErrP band
HIGHCUT_ERRP = 10
FILTER_BUFFER_SIZE = 2048  # ~4 s at 512 Hz
MOTOR_CHANNEL_NAMES = ['FC1','FC2','C3', 'Cz', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'P7','P3', 'Pz', 'P4', 'P8', 'POz']
EOG_CHANNEL_NAMES   = ['AUX1']

# =============================================================================
# Experiment design — trials, timing, feedback geometry
# =============================================================================
ARM_SIDE = "Right"
EXPERIMENT_TYPE = "BASE"  # BIMANUAL or BASE
TOTAL_TRIALS = 20
MAX_REPEATS = 3

TIME_MI = 5          # Motor imagery / rest cue duration (s)
TIME_ROB = 7         # Robot movement window (s)
TIME_STATIONARY = 2  # Stationary feedback after failed/no movement (s)
TIME_MASTER_MOVE = 5 # Bimanual: time to position master arm (s)
TIMING = True        # If True, drivers use automatic countdown paths where implemented

# Segmentation: begin→end spans longer than TIME_MI + 0.5 s are dropped (mis-pairs / missing end).
# Slack absorbs clock/marker jitter; set to 0 for a hard cap at TIME_MI.
MAX_EPOCH_MARKER_DURATION_SEC = float(TIME_MI) + 0.5

# Feedback fill mapping
SHAPE_MAX = 0.7
SHAPE_MIN = 0.5
CLASS_VISUAL_STYLE = "classic"  # "classic" or "modern"
BIG_BROTHER_MODE = True         # If True, force pygame window to external display (0,0) at 1920x1080
ROBOT_TRAJECTORY = ["a"]        # Opcode pool for random trajectory choice where used
SEND_PROBS = False              # If True, stream classifier probs over UDP marker channel
EARLYSTOP_MODE = "correct_only"       # "correct_only" or "either"
# =============================================================================
# Runtime decoder — online classification and thresholds
# =============================================================================
DECODER_BACKEND = "xgb_cov"   # "mdm" | "xgb_cov" | "xgb_cov_erd"
CLASSIFY_WINDOW = 1000        # EEG window length for classification (ms)
BASELINE_DURATION = 1         # seconds
THRESHOLD_MI = 0.6
THRESHOLD_REST = 0.6
RELAXATION_RATIO = 0.0
MIN_PREDICTIONS = 16
STEP_SIZE = 1/16
INTEGRATOR_ALPHA = 0.97
SELECT_MOTOR_CHANNELS = 1
SELECT_ERRP_CHANNELS = 0
SURFACE_LAPLACIAN_TOGGLE = 1

# =============================================================================
# Covariance, shrinkage, and adaptive recentering
# =============================================================================
# Model-specific covariance shrinkage defaults.
SHRINKAGE_PARAM_MDM = 0.02  # MDM path (runtime + MDM-centric analyses)
SHRINKAGE_PARAM_XGB = 0.02  # XGB feature pipelines (covariance preprocessing before tangent features)
LEDOITWOLF = 0

RECENTERING = 1
UPDATE_DURING_MOVE = 0
SAVE_ADAPTIVE_T = False

# Dual-threshold ambiguity target for learned reject/decide thresholds (U/N fraction).
TARGET_AMBIG = 0.20

# =============================================================================
# Training — artifact rejection (sliding-window training segments)
# =============================================================================
# Amplitude unit of segment arrays from XDF + streaming filters. Default "microvolts"
# matches project XDF convention (see Utils.stream_utils.load_xdf docstring).
ARTIFACT_REJECT_ENABLE = 1                 # 0 = keep all windows
ARTIFACT_REJECT_MODE = "max_abs"           # "max_abs" | "peak_to_peak" | "zscore"
ARTIFACT_MAX_ABS_UV = 30.0                 # used when MODE == max_abs
ARTIFACT_P2P_UV = 150.0                    # used when MODE == peak_to_peak
ARTIFACT_ZSCORE_SD = 3.0                   # used when MODE == zscore
ARTIFACT_SEGMENT_AMPLITUDE_UNIT = "microvolts"  # "microvolts" | "volts"
ARTIFACT_REJECT_VERBOSE = 1

# =============================================================================
# Training — cross-validation
# =============================================================================
CV_MODE = "session_loo"   # "kfold" | "session_loo"
N_SPLITS = 5              # KFold splits — used when CV_MODE == "kfold"
# session_loo: GroupKFold respecting session boundaries.  N_LOO_SPLITS caps the
# number of folds so that large datasets (e.g. 21 sessions) don't explode.
# When n_sessions <= N_LOO_SPLITS the split degenerates to true leave-one-session-out.
N_LOO_SPLITS = 100

# =============================================================================
# XGBoost — training and hyperparameter tuning
# =============================================================================
XGB_MAX_DEPTH    = 6
XGB_N_ESTIMATORS = 300
XGB_LEARNING_RATE = 0.05
XGB_USE_COV_MU   = 1
XGB_USE_COV_BETA = 1  # mu-only default; enable beta explicitly when needed
# Online beta band is HIGHCUT..XGB_ERD_BETA_HIGH (consumed by Utils/EEGStreamState.py
# when DECODER_BACKEND is xgb_cov / xgb_cov_erd).
XGB_ERD_BETA_HIGH = 30.0
XGB_ERD_BANDS    = [(float(LOWCUT), float(HIGHCUT))]  # mu-only unless overridden
XGB_IMPORTANCE_TOP_K = 20

# Hyperparameter search (tune_xgb_hyperparams.py)
XGB_TUNE_CRITERION = "auc"   # "kl" | "auc"
# KL criterion target: Beta(BETA_ALPHA, BETA_BETA). Beta(6.1, 2.3) → mode≈0.80, mean≈0.73.
XGB_TUNE_BETA_ALPHA = 6.1
XGB_TUNE_BETA_BETA  = 2.3
XGB_TUNE_KL_BINS    = 15

# =============================================================================
# Gaze / object-selection experiment
# =============================================================================
# Bind vs dial split: GAZE_UDP_IP is the address clients dial (panel +
# experiment driver). GAZE_BIND_HOST is the address gaze_runner.py binds
# its UDP socket on. In production, set GAZE_BIND_HOST = "0.0.0.0" on the
# Windows GPU host and GAZE_UDP_IP = <windows_lan_ip> on Linux. Both
# default to 127.0.0.1 for the single-machine dev configuration.
# Machine-local — override in config_local.py.
GAZE_UDP_IP = "127.0.0.1"
GAZE_BIND_HOST = "127.0.0.1"
GAZE_UDP_PORT = 5588
GAZE_UDP_TIMEOUT = 0.15
GAZE_SELECTION_WINDOW = 5.0
GAZE_AVG_WINDOW = 2.0
GAZE_MIN_DWELL_SEC = 0.75
GO_NOGO_PROMPT_SEC = 1.25
GAZE_SAMPLE_WIDTH = 1600.0
GAZE_SAMPLE_HEIGHT = 1200.0
# POSE_LIBRARY_PATH derives from WORKING_DIR — defined at the bottom of
# this file so it picks up any config_local.py override.

# =============================================================================
# Pupil Labs Neon — device connection
# =============================================================================
# IP address of the phone running the Pupil Labs Companion app.
# Leave empty ("") to use mDNS auto-discovery, which works on home/hotspot
# networks but is blocked on most enterprise/IoT VLANs.
# Find the IP in the Companion app: tap the streaming icon → note the
# address shown (e.g. "10.42.0.100"). Machine-local — override in
# config_local.py per network.
NEON_COMPANION_HOST = ""

# =============================================================================
# Perception frame source — local Neon vs. remote frame_relay
# =============================================================================
# Selects how vlm_service.py and gaze_runner.py acquire scene frames.
#   "local"  — service opens the Neon device directly (today's behaviour;
#              works on a single machine that owns the Companion phone).
#   "remote" — service consumes envelopes from a Utils/frame_relay.py TCP
#              server. Production topology: Linux runs the relay against
#              the Neon device, Windows runs the perception services with
#              --frame-source=remote.
# Reference: SoftwareDocs/GPU_Service_Host_Architecture_Plan.md §3.4.
# Machine-local — override in config_local.py.
PERCEPTION_FRAME_SOURCE = "local"

# Where do the perception services (vlm_service.py, gaze_runner.py) run
# from this panel's perspective? On the Linux device host this is True
# (services live on the Windows GPU host); on Windows or single-machine
# dev this is False. Drives panel UX: when True, Start/Stop buttons that
# would spawn local conda subprocesses are disabled and replaced with
# remote-status badges fed by `cmd: status` UDP pings.
# Machine-local — override in config_local.py.
SERVICES_HOSTED_REMOTELY = False

# Bind host for the relay server (the machine that owns Neon). 0.0.0.0
# accepts connections from any LAN peer; 127.0.0.1 keeps it loopback-only
# for single-machine validation. Machine-local — override in config_local.py.
FRAME_RELAY_HOST = "127.0.0.1"
# Dial host for the relay client (the machine that runs the models). Set
# this to the relay host's LAN IP in production. For loopback validation
# leave it as 127.0.0.1. Machine-local — override in config_local.py.
FRAME_RELAY_DIAL_HOST = "127.0.0.1"
FRAME_RELAY_PORT = 5591
# Default 15 Hz — chosen to fit a UT IoT / cellular uplink budget. With
# both vlm_service and gaze_runner pulling concurrently the steady-state
# wire load is ~36 Mbit/s at q=75 JPEG; comfortable on a typical IoT VLAN
# or 5G hotspot. Raise to 30.0 on LAN to match Neon's native scene-camera
# FPS for sharper fixation timing; at 30 Hz both consumers together push
# ~72 Mbit/s which UT IoT will not carry. The relay can never exceed the
# Neon producer (~30 Hz), so values above 30 are silently capped.
FRAME_RELAY_HZ = 15.0
# When True (default) the control panel hosts the frame relay in-process,
# so launching the panel is sufficient on the Neon machine. Set to False
# when an out-of-process relay is being run separately (e.g. for testing
# or when a third machine sits between the Neon owner and the model host).
# Plain-default key — committable from any machine.
FRAME_RELAY_EMBEDDED = True

# =============================================================================
# VLM integration (harmony_vlm subprocess)
# =============================================================================
# Gaze/object-recognition backend selector:
#   "legacy" — our gaze_runner service with YOLO + SORT tracker
#   "vlm"    — our vlm_service subprocess, which imports harmony_vlm's utils/
#              (FastSAM + Depth Pro + Gemini) and exposes them over UDP
GAZE_OR_BACKEND = "vlm"

# Sibling directory holding the harmony_vlm clone. Machine-local —
# override in config_local.py.
VLM_REPO_DIR = ""

# Conda env used to launch vlm_service.py. Separate from "lsl" because depth-pro
# pins numpy<2, which is incompatible with pyriemann and opencv in the BCI stack.
VLM_CONDA_ENV = "harmony_vlm"

# Gemini model for the VLM reasoner. "gemini-2.5-flash" is free-tier available;
# "gemini-2.5-pro" requires a paid Google AI account.
VLM_MODEL = "gemini-2.5-flash"

# Whether to load Depth Pro at service startup. Depth Pro on CPU is slow
# (~1-3 s per call). Disable to skip scene depth while testing VLM reasoning
# alone; segment/reason/decide endpoints return without depth fields.
VLM_ENABLE_DEPTH = True

# UDP endpoint for the vlm_service request-reply protocol. Must differ from
# GAZE_UDP_PORT (5588) since both services can run concurrently on localhost.
# Bind vs dial split: VLM_SERVICE_HOST is the dial address (panel /
# experiment driver). VLM_BIND_HOST is the address vlm_service.py binds on
# (UDP request socket + TCP overlay socket). In production set
# VLM_BIND_HOST = "0.0.0.0" on Windows and VLM_SERVICE_HOST = <windows_lan_ip>
# on Linux. Both default to 127.0.0.1 for single-machine dev.
# Machine-local — override in config_local.py.
VLM_SERVICE_HOST = "127.0.0.1"
VLM_BIND_HOST = "127.0.0.1"
VLM_SERVICE_PORT = 5589
VLM_SERVICE_TIMEOUT = 0.5

# VLM_SESSION_ROOT derives from DATA_DIR — defined at the bottom of this
# file so it picks up any config_local.py override.

# =============================================================================
# FES
# =============================================================================
FES_toggle = 0
FES_CHANNEL = "red"
FES_TIMING_OFFSET = 7  # Seconds before end of movement for motor FES cutoff (successful case)

# =============================================================================
# Arduino actuator
# =============================================================================
USE_ARDUINO = True
# ARDUINO_PORT is the OS-level serial device path; differs between Linux
# (/dev/ttyACM0) and Windows (COM3 etc.). Machine-local — override in
# config_local.py.
ARDUINO_PORT = ""
ARDUINO_BAUD = 9600
ARDUINO_CMD_MI   = b"1"
ARDUINO_CMD_REST = b"0"

# =============================================================================
# Tiagobot (mobile-arm Arduino device; separate from the glove Arduino)
# =============================================================================
# TIAGOBOT_PORT is machine-local — set in config_local.py. When both the
# glove and Tiagobot are connected, /dev/ttyACM0 vs /dev/ttyACM1 enumeration
# is non-deterministic; prefer the stable /dev/serial/by-id/ path.
TIAGOBOT_PORT = ""
TIAGOBOT_BAUD = 9600
# Pool of preset locations to choose from per MI-success trial (letters
# defined in Utils/tiagobot.py:LOCATIONS).
TIAGOBOT_TRAJECTORY = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
# Optional glove integration in the Tiagobot driver. When True, the driver
# also opens config.ARDUINO_PORT and writes ARDUINO_CMD_MI / ARDUINO_CMD_REST
# on each MI / HOME phase, exactly like ExperimentDriver_Online_Glove.py.
TIAGOBOT_USE_GLOVE = False
# Seconds the driver waits after writing ARDUINO_CMD_MI (close glove) and
# before sending HOME. The glove's mechanical close takes time and HOME
# starting too early means the actuator retracts mid-grip. Tune to match
# the glove's actual close duration on this hardware.
TIAGOBOT_GRIP_HOLD_DURATION = 5
# Seconds the gaze driver holds the cross+shapes pre-task frame on
# screen between the Phase 2 gaze window and the Phase 4
# show_feedback call. Matches the base driver's `countdown_duration =
# 3000` ms convention: a solid white timing orb above the cross
# signals the MI/Rest task is about to begin in this many seconds.
# (ExperimentDriver_Online.py line 305: `draw_time_balls(1, ...)`.)
TIAGOBOT_MODE_REVEAL_DURATION = 3.0
# Seconds of inter-trial anticipation fixation. The driver renders a
# fixation cross plus a white orb that fills linearly over this
# duration, with a "Look at the fixation cross" instruction below.
# Acts as a visual countdown so the patient knows the gaze-grid
# screen is about to appear.
TIAGOBOT_ANTICIPATION_DURATION = 3.0
# Seconds of Phase 1 trial-prep hold on the gaze-grid screen before
# the continuous-dwell selection window opens. Gives the patient a
# moment to register the grid and pick a target with their eyes
# without imposing a visible countdown.
TIAGOBOT_TRIAL_PREP_DURATION = 2.0

# Tiagobot gaze calibration NPZ produced by tiago_gaze_calibration_exec.py.
# Consumed by ExperimentDriver_Online_Tiagobot_Gaze.py to map averaged
# gaze samples to one of the 9 A-I letters per
# Documents/SoftwareDocs/Tiagobot_Gaze_AI_Layout.md. Empty string means
# "no calibration available" — the gaze driver fails at startup in that
# case (fail-fast on Tier 2 paths per CLAUDE.md). Default empty so the
# global config can ship platform-neutral; set the real path per machine
# in config_local.py.
TIAGOBOT_GAZE_CALIBRATION_PATH = ""
# Per-trial gaze accumulation window (seconds) used by the gaze driver
# to collect samples before classifying to a letter. Matches the
# structure of GAZE_SELECTION_WINDOW used in the Harmony gaze driver,
# but tuned independently for the rudimentary 9-letter classifier.
TIAGOBOT_GAZE_SELECTION_WINDOW = 4.0
# Minimum gaze confidence (Pupil Labs Neon) for a sample to count in
# the per-trial average. Matches harmony_calibration_exec.py:43
# (GAZE_CONFIDENCE_THRESHOLD = 0.7) for symmetry between calibration
# and online operation.
TIAGOBOT_GAZE_CONFIDENCE_THRESHOLD = 0.7
# Optional Euclidean-distance ceiling in normalized [0,1] units. If the
# best-match centroid is farther than this from the averaged gaze, the
# classifier returns None and the driver skips the GO (logs and waits).
# None disables the check (always pick the nearest letter). 0.2 is a
# reasonable default for a 0.25-spaced grid — the gaze needs to be
# within ~half a cell of the centroid.
TIAGOBOT_GAZE_MAX_DIST_NORM = 0.2
# Scale factor that maps vergence depth (cm) into the same numerical
# range as the normalized [0, 1] gaze axes so the 3D classifier's
# Euclidean distance is meaningful. 0.01 means 10 cm of depth contributes
# 0.1 units of distance — roughly comparable to one grid cell. Set
# to 0 to disable the depth axis (pure 2D classification, same as the
# pre-2026-05-20 behaviour). Bench-tuned against an oblique seating
# angle where row centroids collapse on the y axis but separate on
# depth; raise if the depth measurement is reliable and you want it
# to dominate the decision.
TIAGOBOT_GAZE_DEPTH_WEIGHT_CM_INV = 0.01
# Ceiling on the Mahalanobis distance returned by
# `classify_gaze_mahalanobis`. None always picks the closest letter
# (no skip), which is fine when the user is reliably looking at *some*
# letter on the board. Set to a chi-squared critical value if you'd
# rather skip trials where no letter is plausible: ~7.81 for p<0.05
# with 3 DOF, ~11.34 for p<0.01. Has no effect on the centroid
# classifier (`classify_gaze_to_letter`), which still uses
# `TIAGOBOT_GAZE_MAX_DIST_NORM`.
TIAGOBOT_GAZE_MAX_MAHAL_DIST = None
# Continuous-dwell letter selection. The Tiagobot gaze driver now
# classifies every Neon snapshot in-loop and accumulates "continuous
# dwell" on the current letter; switching letters, looking off-grid
# (no centroid within TIAGOBOT_GAZE_MAX_DIST_NORM), or losing
# tracking for longer than the stale gap resets the counter. Selection
# fires when continuous dwell crosses TIAGOBOT_GAZE_DWELL_HIT_SEC.
# TIAGOBOT_GAZE_SELECTION_TIMEOUT_SEC bounds how long we wait for a
# hit before returning no-selection (Phase 2.5 aborts MI trials in
# that case). TIAGOBOT_GAZE_CONFIRM_SELECTION_SEC is the duration of
# the explicit "Selected: <letter>" screen that follows a successful
# selection so the subject sees the chosen letter before Phase 3 /
# baseline begins — most important on REST trials where the action
# feedback never names the letter.
TIAGOBOT_GAZE_DWELL_HIT_SEC = 2.0
TIAGOBOT_GAZE_SELECTION_TIMEOUT_SEC = 12.0
TIAGOBOT_GAZE_CONFIRM_SELECTION_SEC = 1.5

# =============================================================================
# Display colors (RGB)
# =============================================================================
black = (0, 0, 0)
white = (255, 255, 255)
blue = (0, 0, 255)
red = (255, 0, 0)
green = (0, 255, 0)
orange = (255, 165, 0)

# =============================================================================
# Networking — UDP endpoints and protocol strings
# =============================================================================
UDP_MARKER = {
    "IP": "127.0.0.1",
    "PORT": 12345
}
UDP_ROBOT = {
    "IP": "192.168.2.1",
    "PORT": 8080
}
UDP_FES = {
    "IP": "127.0.0.1",
    "PORT": 5005
}
UDP_CONTROL_BIND = {
    "IP": "192.168.2.2",
    "PORT": 8080
}

# When True (default), Utils/networking.py binds a UDP socket to
# UDP_CONTROL_BIND at module import for sending Harmony robot opcodes.
# On Tiagobot-only rigs the Harmony bind address is not assigned to a
# local interface — the bind fails with EADDRNOTAVAIL on every send and
# floods the log. Override to False in config_local.py on those rigs;
# Tiagobot drivers never send to UDP_ROBOT (the actuator speaks serial)
# so the missing socket has no functional impact.
BIND_ROBOT_CONTROL_SOCKET = True

TRIGGERS = {
    "MI_BEGIN": "200",
    "MI_END": "220",
    "MI_EARLYSTOP": "240",
    "MI_PROBS": "2000",
    "ROBOT_BEGIN": "300",
    "ACK_ROBOT_BEGIN": "305",
    "ROBOT_END": "320",
    "ACK_ROBOT_END": "325",
    "ROBOT_EARLYSTOP": "340",
    "ACK_ROBOT_STOP": "345",
    "ROBOT_PROBS": "3000",
    "ROBOT_PAUSE": "360",
    "ACK_ROBOT_PAUSE": "365",
    "ROBOT_RESUME": "370",
    "ACK_ROBOT_RESUME": "375",
    "ROBOT_HOME": "380",
    "ACK_ROBOT_HOME": "385",
    "ERRP_BEGIN": "400",
    "ERRP_END": "420",
    "REST_BEGIN": "100",
    "REST_END": "120",
    "REST_EARLYSTOP": "140",
    "REST_PROBS": "1000",
    "MASTER_UNLOCK": "500",
    "ACK_MASTER_UNLOCK": "505",
    "MASTER_LOCK": "520",
    "ACK_MASTER_LOCK": "525",
}

ROBOT_OPCODES = {
    "TRAJECTORY_A": "a",
    "TRAJECTORY_X": "x",
    "TRAJECTORY_Y": "y",
    "TRAJECTORY_Z": "z",
    "GO": "g",
    "HOME": "h;dur=3",
    "STOP": "s",
    "PAUSE": "p",
    "RESUME": "r",
    "MASTER_UNLOCK": "m",
    "MASTER_LOCK": "c",
    "QUERY": "q",
    "EXIT": "e"
}

# =============================================================================
# ErrP decoder
# =============================================================================
# Master toggle: 0 = ErrP pipeline disabled (MI pipeline unaffected), 1 = enabled
ERRP_DECODER_ENABLE = 0
# Classifier backend — must match the suffix of the model file on disk:
#   DATA_DIR/sub-{SUBJECT}/models/sub-{SUBJECT}_errp_{BACKEND}.pkl
# Options validated live: "liu_cca_xgb", "xdawn_xgb"
ERRP_DECODER_BACKEND = "liu_cca_xgb"
# Epoch window anchored at the event marker (seconds post-event).
# 0-800 ms captures ERN (~80-150 ms) and Pe (~200-400 ms).
ERRP_EPOCH_TMIN = 0.0
ERRP_EPOCH_TMAX = 0.8
# xDAWN spatial filters per class (4 is standard for P300/ErrP paradigms)
ERRP_XDAWN_N_FILTERS = 4
# Artifact rejection threshold for ErrP epochs (µV, max_abs on 1-10 Hz filtered signal)
ERRP_ARTIFACT_MAX_ABS_UV = 80.0
# Dual-threshold ambiguity target for ErrP (fraction of trials allowed to be ambiguous)
ERRP_TARGET_AMBIG = 0.20
# LogisticRegression regularization (xdawn_lr backend only)
ERRP_LR_C = 1.0
# CAR rereferencing for ErrP stream (1 = on, matches offline training path)
ERRP_CAR_REREFERENCE = 1
# Bootstrap: seconds of quiet fixation before first trial used to fit EA reference
ERRP_EA_BOOTSTRAP_SEC = 45.0
# Minimum pseudo-epochs for EA bootstrap (each epoch = ERRP_EPOCH_TMAX * FS samples)
ERRP_EA_MIN_EPOCHS = 20
# ErrP experiment paradigm
# Probability that the robot stops mid-trajectory (error condition) — online driver
ERRP_ONLINE_P_STOP = 0.3
# Minimum/maximum stop time bounds (seconds into trajectory)
ERRP_STOP_TMIN = 1.0
ERRP_STOP_TMAX_FRACTION = 0.7
# Seconds the robot stays paused if no ErrP is detected before homing
ERRP_NO_RESUME_TIMEOUT = 3.0
# Total trials for dedicated ErrP experiment driver
TOTAL_TRIALS_ERRP = 45
# ErrP channel set — expanded to 14 channels based on cross-subject decoder validation
ERRP_CHANNEL_NAMES = ['F3', 'Fz', 'F4', 'FC1', 'FC2', 'C3', 'Cz', 'C4', 'CP1', 'CP2', 'Pz', 'POz', 'O1', 'O2']

# =============================================================================
# Global runtime flags
# =============================================================================
SIMULATION_MODE = False
try:
    from config_local import *  # noqa: F401, F403
except ImportError:
    pass

# =============================================================================
# Derived paths — placed AFTER the local-override import so they pick up
# the per-machine WORKING_DIR / DATA_DIR. Defining these earlier would
# bake the empty defaults into every consumer.
# =============================================================================
POSE_LIBRARY_PATH = os.path.join(WORKING_DIR, "poses_with_gaze_20251202_153040.npz")
VLM_SESSION_ROOT  = os.path.join(DATA_DIR, "vlm_sessions")
