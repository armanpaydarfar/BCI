# Configuration file for EEG experiments

# UDP Settings
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


# EEG Settings
CAP_TYPE = 32
LOWCUT = 8  # Hz
HIGHCUT = 30  # Hz
FS = 512  # Sampling frequency (Hz)

# Experiment Parameters
TOTAL_TRIALS = 30  # Total number of trials
MAX_REPEATS = 3  # Maximum consecutive repeats of the same condition
N_SPLITS = 5  # Number of splits for KFold cross-validation
TIME_MI = 5 # time for motor imagery and rest
TIME_ROB = 13 # time allocated for robot to move
TIME_STATIONARY = 2 # time for stationary feedback after no movement/failed movement trial


# Classification Parameters
CLASSIFY_WINDOW = 1000  # Duration of EEG data window for classification (milliseconds)
ACCURACY_THRESHOLD = 0.51  # Accuracy threshold to determine "Correct"



# FES Parameters
FES_toggle = 1
FES_CHANNEL = "red"

# Screen Dimensions
SCREEN_WIDTH = 1800
SCREEN_HEIGHT = 1200

# LDA Model Path
MODEL_PATH = "/home/arman-admin/Projects/Harmony/lda_eeg_model.pkl"
DATA_FILE_PATH = "/home/arman-admin/Documents/CurrentStudy/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
# Colors
black = (0, 0, 0)
white = (255, 255, 255)
blue = (0, 0, 255)
red = (255, 0, 0)
green = (0, 255, 0)

