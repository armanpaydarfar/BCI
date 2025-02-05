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
HIGHCUT = 12  # Hz
FS = 512  # Sampling frequency (Hz)

EEG_CHANNEL_NAMES = ['F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'C3', 'CZ', 'C4', 
                     'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4', 'P8', 'POZ'] # List of EEG channel names to use
'''
EEG_CHANNEL_NAMES = ['ALL']
'''
EOG_CHANNEL_NAMES = ['AUX1'] # List of EOG channel names to use
EOG_TOGGLE = 1  # Toggle to enable or disable EOG processing (1 = enabled, 0 = disabled)



# Experiment Parameters
TOTAL_TRIALS = 30  # Total number of trials
MAX_REPEATS = 3  # Maximum consecutive repeats of the same condition
N_SPLITS = 5  # Number of splits for KFold cross-validation
TIME_MI = 5 # time for motor imagery and rest
TIME_ROB = 13 # time allocated for robot to move
TIME_STATIONARY = 2 # time for stationary feedback after no movement/failed movement trial
TIMING = True
SHAPE_MAX = 0.9 #maximum fill 
SHAPE_MIN = 0.5 #minimum fill 


# Classification Parameters
CLASSIFY_WINDOW = 1000  # Duration of EEG data window for classification (milliseconds)
ACCURACY_THRESHOLD = 0.7  # Accuracy threshold to determine "Correct"
RELAXATION_RATIO = 0.8
MIN_PREDICTIONS = 45 # Min number of predictions during Online experiment before the decoder can end early
CLASSIFICATION_OFFSET = 0 # Offset for "classification window" starting point
#CLASSIFICATION_SCHEME_OPT = "TIMESERIES"
CLASSIFICATION_SCHEME_OPT = "FREQUENCY"

# FES Parameters
FES_toggle = 1
FES_CHANNEL = "red"
FES_TIMING_OFFSET = 4 
# above for motor FES, cut out X seconds before the full duration of movement. This should represent when the robot will naturally reach the end of motion (in successful case)

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

# software triggers
TRIGGERS = {
    "MI_BEGIN": "200",
    "MI_END": "220",
    "MI_EARLYSTOP": "240",
    "ROBOT_BEGIN": "300",
    "ROBOT_END": "320",
    "ROBOT_EARLYSTOP": "340",
    "ROBOT_CONFIRM_STOP": "345",
    "REST_BEGIN": "100",
    "REST_END": "120",
    "REST_EARLYSTOP": "140"
    
}
