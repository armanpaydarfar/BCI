# Configuration file for EEG experiments

# UDP Settings
UDP_MAIN = {
    "IP": "127.0.0.1",
    "PORT": 12345
}

UDP_EXTRA = {
    "IP": "192.168.2.1",
    "PORT": 8080
}

# EEG Settings
CLASSIFY_WINDOW = 1.0  # Duration of EEG data window for classification (seconds)
ACCURACY_THRESHOLD = 0.51  # Accuracy threshold to determine "Correct"
LOWCUT = 0.1  # Hz
HIGHCUT = 30  # Hz
FS = 512  # Sampling frequency (Hz)

# Experiment Parameters
TOTAL_TRIALS = 30  # Total number of trials
MAX_REPEATS = 3  # Maximum consecutive repeats of the same condition
WINDOW_SIZE = 1.0  # EEG segment size in seconds for LDA
N_SPLITS = 5  # Number of splits for KFold cross-validation

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

