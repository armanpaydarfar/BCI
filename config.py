# Configuration for EEG experiments — runtime drivers, utilities, and training schemes.
# Visualization and analysis scripts should not read from this file; they own their own params.
import os

# =============================================================================
# Paths and subject
# =============================================================================
WORKING_DIR = "/home/arman-admin/Projects/Harmony/"
DATA_DIR = "/home/arman-admin/Documents/CurrentStudy"
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
ERRP_CHANNEL_NAMES  = ['F3', 'Fz', 'F4', 'FC1', 'FC2', 'Cz']
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
EARLYSTOP_MODE = "either"       # "correct_only" or "either"

# =============================================================================
# Runtime decoder — online classification and thresholds
# =============================================================================
DECODER_BACKEND = "xgb_cov"   # "mdm" | "xgb_cov" | "xgb_cov_erd"
CLASSIFY_WINDOW = 1000        # EEG window length for classification (ms)
BASELINE_DURATION = 1         # seconds
THRESHOLD_MI = 0.65
THRESHOLD_REST = 0.65
RELAXATION_RATIO = 0.5
MIN_PREDICTIONS = 16
STEP_SIZE = 1/16
INTEGRATOR_ALPHA = 0.94
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
GAZE_UDP_IP = "127.0.0.1"
GAZE_UDP_PORT = 5588
GAZE_UDP_TIMEOUT = 0.15
GAZE_SELECTION_WINDOW = 5.0
GAZE_AVG_WINDOW = 2.0
GAZE_MIN_DWELL_SEC = 0.75
GO_NOGO_PROMPT_SEC = 1.25
GAZE_SAMPLE_WIDTH = 1600.0
GAZE_SAMPLE_HEIGHT = 1200.0
POSE_LIBRARY_PATH = os.path.join(WORKING_DIR, "poses_with_gaze_20251202_153040.npz")

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
ARDUINO_PORT = "/dev/ttyACM0"
ARDUINO_BAUD = 9600
ARDUINO_CMD_MI   = b"1"
ARDUINO_CMD_REST = b"0"

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
# Global runtime flags
# =============================================================================
SIMULATION_MODE = False
