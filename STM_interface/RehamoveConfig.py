import random
import numpy as np
import sys
from itertools import repeat 
import numpy as np
# TIMING CONSTANTS:
# ***************20
ExperimentConfigureTime=0.5
fixationCrossTime=2
cueTime= 2
timeout =7
taskTime =6
resultTime=2
restTime=4
# fixationTime = 2

# trials = [-1,1,1,-1,-1,1]; flipped


thresholdRight = 0.66 # right Hand
thresholdLeft = 0.55 # left Hand

sensorMean = [0.9,0.9];
sensorStd = [0.1,0.1];
detectionThresholdLH = (thresholdRight-0.5)/0.5
detectionThresholdRH = (thresholdLeft-0.5)/0.5

n_LH = 10
n_RH = 10

LH_array = list(repeat(-1, n_LH))
RH_array = list(repeat(+1, n_RH))
total = len(LH_array) + len(RH_array)
trials = np.zeros(total)
all_arrays = LH_array + RH_array

for i in range(0, total, 1):
    trials[i] = random.sample(all_arrays, 1)[0]
    all_arrays.remove(trials[i])

###################################################
###################################################
#
#             TESS Configurations  
#
###################################################
###################################################

FES1_port = '/dev/ttyUSB0'
FES2_port = '/dev/ttyUSB1'
FES_pulseWidth = 250

distalChannel_RH = 'red'
proximalChannel_RH = 'blue'
distalChannel_LH = 'black'
proximalChannel_LH = 'white'


FES_freq = 30 	 # should be 30 Hz (Hertz)
chnName1 = 'blue' # which channel to use
rampTime = 0.5 # in minutes (duration to ramp up and down with TESS)

TESS_dur = 20 # in minutes
# get equivalent number of trials for 20 minutes given the trial timings
# 20 minutes ~= 60 trials
n_LH_PAS = 35
n_RH_PAS = 35

LH_array_PAS = list(repeat(-1, n_LH_PAS))
RH_array_PAS = list(repeat(+1, n_RH_PAS))
total_PAS = len(LH_array_PAS) + len(RH_array_PAS)
trials_PAS = np.zeros(total_PAS)
all_arrays_PAS = LH_array_PAS + RH_array_PAS

for i in range(0, total_PAS, 1):
    trials_PAS[i] = random.sample(all_arrays_PAS, 1)[0]
    all_arrays_PAS.remove(trials_PAS[i])
# 	For stim method 1 (FES):
current = 14	   	# in miiliamperes (mA)
# Minsu = 25, 45, 42, 42
# Diego = 16, 14, 14
pulseWidth = 100 	# in microseconds (usec)
pulsearray = []
for i in range(10):
	if (i % 2) == 0: # even
		pulsearray.append([current, pulseWidth])
	else:            # odd
		pulsearray.append([-current, pulseWidth])

pulsearray = np.array(pulsearray)
print(pulsearray)
#  NOTES:
# ********
# Class (1): Right hand is Extension
# Class (-1): Left hand is Flexion

####################
## IMPORTANT NOTE ##
####################
# the following 17 lines must be the last 17 lines in this file!
# Not eeven empty lines shall be after them becasue another script reads the last 16 lines and modifies them based on the STM setup thresholds
FES_freq = 30.000000
I_STS_distRH = 3.000000
I_STS_proxRH = 0.000000
I_MTS_distRH = 0.000000
I_MTS_proxRH = 0.000000
I_STS_proxLH = 0.000000
I_STS_distLH = 0.000000
I_MTS_proxLH = 0.000000
I_MTS_distLH = 0.000000
p_STS_distRH = 250.000000
p_STS_proxRH = 250.000000
p_MTS_distRH = 250.000000
p_MTS_proxRH = 250.000000
p_STS_proxLH = 250.000000
p_STS_distLH = 250.000000
p_MTS_proxLH = 250.000000
p_MTS_distLH = 250.000000