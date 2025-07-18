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
EOG_TOGGLE = 0  # Toggle to enable or disable EOG processing (1 = enabled, 0 = disabled)

# Experiment Parameters
ARM_SIDE = "Right"
TOTAL_TRIALS = 20  # Total number of trials
TOTAL_TRIALS_ERRP = 45 # Total number of trials for ErrP experiment
MAX_REPEATS = 3  # Maximum consecutive repeats of the same condition
N_SPLITS = 5  # Number of splits for KFold cross-validation
TIME_MI = 5 # time for motor imagery and rest
TIME_ROB = 13 # time allocated for robot to move
TIME_STATIONARY = 2 # time for stationary feedback after no movement/failed movement trial
TIMING = True
SHAPE_MAX = 0.7 #maximum fill 
SHAPE_MIN = 0.5 #minimum fill 
ROBOT_TRAJECTORY = ["a"]
BIG_BROTHER_MODE = True #this toggle exports the game to the second monitor automatically, while retaining the running log in the first windows linux terminal

# Classification Parameters
CLASSIFY_WINDOW = 500  # Duration of EEG data window for classification (milliseconds)
ACCURACY_THRESHOLD = 0.55  # Accuracy threshold to determine "Correct" (plan to obsolete)
THRESHOLD_MI = 0.57 #Threshold for MI "correct"
THRESHOLD_REST = 0.57 #Threshold for REST "Correct"
RELAXATION_RATIO = 1
MIN_PREDICTIONS = 10 # Min number of predictions during Online experiment before the decoder can end early
STEP_SIZE = 1/16
CLASSIFICATION_OFFSET = 0 # Offset for "classification window" starting point
#CLASSIFICATION_SCHEME_OPT = "TIMESERIES"
CLASSIFICATION_SCHEME_OPT = "FREQUENCY"
SURFACE_LAPLACIAN_TOGGLE = 0 #apply the surface laplacian spatial filter during online
SELECT_MOTOR_CHANNELS = 1 # toggle to select motor channels or not (can be used to select other channels too)
INTEGRATOR_ALPHA = 0.95 # defines how fast the accumulated probability may change as new data comes in
SHRINKAGE_PARAM = 0.1 # hyperparameter for shrinkage regularization
LEDOITWOLF = 1 #Set to true to use ledoit wolf shrinkage regularization - otherwise pyreimannian will be used w/ shrinkage param shown above

# adaptive Recentering parameters for config
RECENTERING = 1 # adaptive recentering toggle
USE_CONFIDENCE_GATE = 0 #update Previous transform ONLY in the event the current prob is above the MI/REST threshold (or lean condition)
UPDATE_DURING_MOVE = 0 #this toggle defines whether or not the reimannian adaptive recentering scheme updates when the robot is moving. 0 = no, 1 = yes. The algo will update always during MI
SAVE_ADAPTIVE_T = False #this toggle saves "Adaptive_T" to the EEG directory during an active session between runs - this way, we can continue w/ the current estimated whitening transform. Disabling this will start a fresh transform each time


# FES Parameters
FES_toggle = 0
FES_CHANNEL = "red"
FES_TIMING_OFFSET = 4 
# above for motor FES, cut out X seconds before the full duration of movement. This should represent when the robot will naturally reach the end of motion (in successful case)

# Screen Dimensions
#SCREEN_WIDTH = 3840
#SCREEN_HEIGHT = 2160

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800


# Relevant Directories
WORKING_DIR = "/home/millanslab/Projects/Harmony/"
DATA_DIR = "/home/millanslab/Documents/CurrentStudy"

#MODEL_PATH = "/home/arman-admin/Projects/Harmony/Reiman_eeg_model.pkl"
#MODEL_PATH_ERRP = "C:/Users/ccsmy/Downloads/thesis/new/BCI/coledata/Errp_model.pkl"
#DATA_FILE_PATH = "/home/millanslab/Documents/CurrentStudy/sub-P444/ses-S001/eeg/sub-P444_ses-S001_task-Default_run-001OFFLINE_eeg.xdf"
#DATA_FILE_PATH_ERRP = "C:/Users/ccsmy/Downloads/thesis/new/BCI/coledata/sub-P503_ses-S001_task-Default_run-001_eeg.xdf"

TRAINING_SUBJECT = "P503"


#TRAINING_SESSION = "001OFFLINE"


USE_PREVIOUS_ONLINE_STATS = False # for z-score normalization of data coming in - this defines the starting point, False = use the stats from the training session, true = use previous online stats


# Colors
black = (0, 0, 0)
white = (255, 255, 255)
blue = (0, 0, 255)
red = (255, 0, 0)
green = (0, 255, 0)
orange = (255, 165, 0)

# software triggers
TRIGGERS = {
    "MI_BEGIN": "200",
    "MI_END": "220",
    "MI_EARLYSTOP": "240",
    "MI_PROBS": "2000",

    "ROBOT_BEGIN": "300",
    "ROBOT_END": "320",
    "ROBOT_EARLYSTOP": "340",
    "ROBOT_CONFIRM_STOP": "345",
    "ROBOT_PAUSE": "360",
    "ROBOT_CONFIRM_PAUSE": "365",
    "ROBOT_RESUME": "370",
    "ROBOT_CONFIRM_RESUME": "375",
    "ROBOT_PROBS": "3000",

    "ERRP_BEGIN": "400",
    "ERRP_END": "420",    
    
    "REST_BEGIN": "100",
    "REST_END": "120",
    "REST_EARLYSTOP": "140",
    "REST_PROBS": "1000"
}
