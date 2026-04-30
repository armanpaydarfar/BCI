# Configuration file for EEG experiments — runtime and experiment defaults.
# Section order is for human operators; all public names are kept stable for import compatibility.
import os

# =============================================================================
# Paths and subject
# =============================================================================
WORKING_DIR = "C:/Users/arman/Projects/BCI/"
DATA_DIR = "C:/Users/arman/Documents/CurrentStudy"
TRAINING_SUBJECT = "PILOT007"
# =============================================================================
# EEG acquisition and channels
# =============================================================================
CAP_TYPE = 32
LOWCUT = 8  # Hz
HIGHCUT = 13  # Hz
LOWCUT_ERRP = 1  # Hz
HIGHCUT_ERRP = 10  # Hz
FS = 512  # Sampling frequency (Hz)
MOTOR_CHANNEL_NAMES = ['FC1','FC2','C3', 'Cz', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'P7','P3', 'Pz', 'P4', 'P8', 'POz']
ERRP_CHANNEL_NAMES = ['F3', 'Fz', 'F4', 'FC1', 'FC2', 'Cz']
EOG_CHANNEL_NAMES = ['AUX1']  # List of EOG channel names to use
# =============================================================================
# Experiment design (trials, timing, trajectories)
# =============================================================================
ARM_SIDE = "Right"
EXPERIMENT_TYPE = "BASE"  # BIMANUAL or BASE
TOTAL_TRIALS = 20
TOTAL_TRIALS_ERRP = 45
MAX_REPEATS = 3
N_SPLITS = 5  # KFold splits (training scripts)
# CV strategy for offline training: "kfold" = standard StratifiedKFold (no session grouping),
# "session_grouped" = StratifiedGroupKFold grouped by session file (folds capped at N_SPLITS).
CV_MODE = "kfold"
TIME_MI = 5  # Motor imagery / rest cue duration (s)
TIME_ROB = 7  # Robot movement window (s)
TIME_STATIONARY = 2  # Stationary feedback after failed/no movement (s)
TIME_MASTER_MOVE = 5  # Bimanual: time to position master arm (s)
# Segmentation: begin→end spans longer than TIME_MI + slack are dropped (mis-pairs / missing end).
# Keep slack small but nonzero for clock/marker jitter; tighten slack to 0.0 if you want a hard cap at TIME_MI.
MAX_EPOCH_MARKER_SLACK_SEC = 0.5
MAX_EPOCH_MARKER_DURATION_SEC = float(TIME_MI) + float(MAX_EPOCH_MARKER_SLACK_SEC)
TIMING = True  # If True, drivers use automatic countdown paths where implemented
SHAPE_MAX = 0.8  # Upper bound for feedback fill mapping
SHAPE_MIN = 0.5  # Lower bound for feedback fill mapping
ROBOT_TRAJECTORY = ["a"]  # Opcode pool for random trajectory choice where used
BIG_BROTHER_MODE = True  # If True, force pygame window to external display (0,0) at 1920x1080
SEND_PROBS = False  # If True, stream classifier probs over UDP marker channel
# Early-stop policy: "correct_only" or "either"
EARLYSTOP_MODE = "correct_only"

# =============================================================================
# Gaze / object-selection experiment
# =============================================================================
GAZE_UDP_IP = "127.0.0.1"
GAZE_UDP_PORT = 5588
GAZE_UDP_TIMEOUT = 0.15
GAZE_SELECTION_WINDOW = 5.0
GAZE_AVG_WINDOW = 2.0
GAZE_MIN_DWELL_SEC = 0.75
GO_NOGO_PROMPT_SEC = 1.25
GAZE_SAMPLE_WIDTH = 1600.0
GAZE_SAMPLE_HEIGHT = 1200.0
POSE_LIBRARY_FILENAME = "poses_with_gaze_20251202_153040.npz"
POSE_LIBRARY_PATH = os.path.join(WORKING_DIR, POSE_LIBRARY_FILENAME)
ROBOT_MOVE_DUR = TIME_ROB  # Alias used by gaze experiment code

# =============================================================================
# Classification, decoder, and feedback parameters
# =============================================================================
CLASSIFY_WINDOW = 1000  # EEG window length for classification (ms)
FILTER_BUFFER_SIZE = 2048  # ~4 s at 512 Hz
BASELINE_DURATION = 1  # seconds
ACCURACY_THRESHOLD = 0.6  # Legacy / logging only; see CHANGELOG.md — thresholds below drive decisions
THRESHOLD_MI = 0.6
THRESHOLD_REST = 0.6
RELAXATION_RATIO = 0.5
MIN_PREDICTIONS = 32
STEP_SIZE = 1/16
CLASSIFICATION_OFFSET = 0
CLASSIFICATION_SCHEME_OPT = "FREQUENCY"  # or "TIMESERIES"
SURFACE_LAPLACIAN_TOGGLE = 0

# =============================================================================
# Dual-threshold ambiguity target (used for learned reject/decide thresholds)
# =============================================================================
# target_ambig is the desired ambiguity fraction U/N (ambiguous/rejected samples)
# during threshold selection.
TARGET_AMBIG = 0.20
SELECT_MOTOR_CHANNELS = 1
SELECT_ERRP_CHANNELS = 0
INTEGRATOR_ALPHA = 0.96
# Model-specific covariance shrinkage defaults.
# - MDM path (runtime + MDM-centric analyses)
SHRINKAGE_PARAM_MDM = 0.02
# - XGB feature pipelines (covariance preprocessing before tangent features)
SHRINKAGE_PARAM_XGB = 0.02
# Backward-compatible alias (legacy code may still read SHRINKAGE_PARAM).
SHRINKAGE_PARAM = SHRINKAGE_PARAM_MDM
LEDOITWOLF = 0

# =============================================================================
# Offline artifact rejection (sliding-window training segments)
# =============================================================================
# Amplitude unit of segment arrays from XDF + streaming filters. Default "microvolts"
# matches project XDF convention (see Utils.stream_utils.load_xdf docstring).
ARTIFACT_REJECT_ENABLE = 1  # 0 = keep all windows
ARTIFACT_REJECT_MODE = "max_abs"  # "max_abs" | "peak_to_peak" | "zscore"
ARTIFACT_MAX_ABS_UV = 30.0  # used when MODE == max_abs (same default as legacy adaptive script)
ARTIFACT_P2P_UV = 150.0  # used when MODE == peak_to_peak (order-of-magnitude match to visualize QC)
ARTIFACT_ZSCORE_SD = 3.0  # used when MODE == zscore (|z| on per-window max |x|)
ARTIFACT_SEGMENT_AMPLITUDE_UNIT = "microvolts"  # "microvolts" | "volts"
ARTIFACT_REJECT_VERBOSE = 1  # print drop counts per file

# =============================================================================
# visualize_online_data.py — epoch QC (µV, same numeric scale as XDF / raw._data)
# =============================================================================
# max_abs: matches training artifact logic — mu-band (LOWCUT..HIGHCUT) after notch, then
#   max|x| over channels×time per epoch; broadband-filtered epochs are subset to match.
# peak_to_peak: MNE’s built-in epoch reject (P2P) on broadband raw used for plotting.
VISUALIZE_EPOCH_REJECT_MODE = "max_abs"  # "max_abs" | "peak_to_peak"
VISUALIZE_EPOCH_MAX_ABS_UV = 45.0  # align with ARTIFACT_MAX_ABS_UV when using max_abs
VISUALIZE_EPOCH_REJECT_P2P_UV = 150.0  # used when MODE == peak_to_peak
VISUALIZE_EPOCH_FLAT_UV = None  # e.g. 1.0 for 1 µV flat criterion; None disables

# =============================================================================
# Adaptive recentering (Riemannian)
# =============================================================================
RECENTERING = 0
UPDATE_DURING_MOVE = 0
SAVE_ADAPTIVE_T = False

# =============================================================================
# XGBoost defaults (offline feature pipelines)
# =============================================================================
XGB_MAX_DEPTH = 5
XGB_USE_COV_MU = 1
# Default XGB covariance branch is mu-only. Enable beta explicitly when needed.
XGB_USE_COV_BETA = 1
# Default ERD bands are also mu-only unless overridden (e.g., add beta bands explicitly).
XGB_ERD_BANDS = [(float(LOWCUT), float(HIGHCUT))]
XGB_IMPORTANCE_TOP_K = 20

# Online decoder backend: "mdm" | "xgb_cov" | "xgb_cov_erd" | "rbnnet"
DECODER_BACKEND = "rbnnet"

# =============================================================================
# RBNNet hyperparameters (generate_rbnnet_model.py)
# =============================================================================
RBNNET_EPOCHS = 200             # Training epochs (maximum; early stopping may terminate sooner)
RBNNET_EARLY_STOP_PATIENCE = 50 # Stop if training loss does not improve for this many epochs
RBNNET_LR = 1e-3                # Adam learning rate
RBNNET_BATCH_SIZE = 64          # Mini-batch size
RBNNET_WEIGHT_DECAY = 1e-4      # Adam L2 weight decay
RBNNET_VARIANCE_RETAINED = 0.90   # Fraction of variance for epsilon (ReEig threshold)
SHRINKAGE_PARAM_RBNNET = 0.02   # Covariance shrinkage lambda (same as MDM)
RBNNET_USE_BETA = 0             # 0 = single-band mu only, 1 = dual-band mu+beta
RBNNET_LOWCUT_BETA = 13         # Hz — beta band low edge
RBNNET_HIGHCUT_BETA = 30        # Hz — beta band high edge
RBNNET_ONLINE_ADAPT = 0         # 1 = update RBN running means online (train-mode forward pass, no weight update)
MAX_LOSO_FOLDS = 8              # Cap on LOSO CV folds; consecutive sessions are grouped when n_files exceeds this

# =============================================================================
# FES
# =============================================================================
FES_toggle = 0
FES_CHANNEL = "red"
FES_TIMING_OFFSET = 7  # Seconds before end of movement for motor FES cutoff (successful case)

# =============================================================================
# Display / pygame feedback
# =============================================================================
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
# Feedback geometry: "classic" (default) or "modern" (refined shapes + accumulation bar in driver)
CLASS_VISUAL_STYLE = "classic"

# Colors (RGB)
black = (0, 0, 0)
white = (255, 255, 255)
blue = (0, 0, 255)
red = (255, 0, 0)
green = (0, 255, 0)
orange = (255, 165, 0)

# =============================================================================
# UDP endpoints
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

# =============================================================================
# Marker and robot protocol strings
# =============================================================================
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
# Arduino actuator
# =============================================================================
USE_ARDUINO = True
ARDUINO_PORT = "COM3"
ARDUINO_BAUD = 9600
ARDUINO_CMD_MI   = b"1"
ARDUINO_CMD_REST = b"0"

# =============================================================================
# Global runtime flags
# =============================================================================
SIMULATION_MODE = True