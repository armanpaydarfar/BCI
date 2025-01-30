from tkinter import *
import tkinter as tk
import sys
import json
import config

# ## LOOP + Serial Comm Packages
import serial
import os
from pathlib import Path
dirP = os.path.abspath(os.getcwd())
print(dirP)
# #print(dirP + '/4_ref_other')
sys.path.append(dirP + '/z1_ref_other/0_lib')

## Other Packages:
import pygame

import time
import random
from pygame.locals import *
import pyautogui


import os
os.system("setserial /dev/ttyUSB0 low_latency")
## LOAD CONFIGURATIONS FOR THE TASK 

## FES1 Rehamove Library
sys.path.append(dirP + '/STM_interface/1_packages/rehamoveLibrary')
print(dirP + '/STM_interface/1_packages/rehamoveLibrary')
from rehamove import *          # Import our library
import time
import math 


from STM_interface.RehamoveConfig import *
filename = 'STM_interface/RehamoveConfig_simple.json'
print('STM selected')

FES1 = Rehamove(FES1_port)    # Open USB port (on Windows)

root = Tk()
root.title("Hands Stimulation")

RH_stsRed_i = DoubleVar()
LH_stsBlack_i = DoubleVar()
RH_stsBlue_i = DoubleVar()
LH_stsWhite_i = DoubleVar()
RH_stsRed_p = DoubleVar()
LH_stsBlack_p = DoubleVar()
RH_stsBlue_p = DoubleVar()
LH_stsWhite_p = DoubleVar()
RH_mtsRed_i = DoubleVar()
LH_mtsBlack_i = DoubleVar()
RH_mtsBlue_i = DoubleVar()
LH_mtsWhite_i = DoubleVar()
RH_mtsRed_p = DoubleVar()
LH_mtsBlack_p = DoubleVar()
RH_mtsBlue_p = DoubleVar()
LH_mtsWhite_p = DoubleVar()
freq = DoubleVar()
stsTime = 2;
mtsTime = 2;
global FES1_freq 
FES1_freq = 30

def RH_LH_freqChange():
	global FES1_freq
	FES1_freq = int(freq.get())


## Sensory

def RH_LH_stsRed():
	RH_LH_freqChange()
	print(FES1_freq)
	prevTime=time.time() 
	while (time.time() - prevTime)<=stsTime:
		cycle_time = time.time()
		FES1.pulse('red',RH_stsRed_i.get(), int(RH_stsRed_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		time.sleep(max(0, (1 / FES1_freq) - (time.time() - cycle_time)))
		#time.sleep(1/FES1_freq)

def RH_LH_stsBlack():
	RH_LH_freqChange()
	prevTime=time.time() 
	while (time.time() - prevTime)<=stsTime:
		cycle_time = time.time()
		FES1.pulse('black',LH_stsBlack_i.get(), int(LH_stsBlack_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		time.sleep(max(0, (1 / FES1_freq) - (time.time() - cycle_time)))
		#time.sleep(1/FES1_freq)
def RH_LH_stsBlue():
	RH_LH_freqChange()
	prevTime=time.time() 
	while (time.time() - prevTime)<=stsTime:
		cycle_time = time.time()
		FES1.pulse('blue',RH_stsBlue_i.get(), int(RH_stsBlue_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		time.sleep(max(0, (1 / FES1_freq) - (time.time() - cycle_time)))
		#time.sleep(1/FES1_freq)
def RH_LH_stsWhite():
	RH_LH_freqChange()
	prevTime=time.time() 
	while (time.time() - prevTime)<=stsTime:
		cycle_time = time.time()
		FES1.pulse('white',LH_stsWhite_i.get(), int(LH_stsWhite_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		time.sleep(max(0, (1 / FES1_freq) - (time.time() - cycle_time)))
		#time.sleep(1/FES1_freq)

def RH_LH_stsRedBlue():
	RH_LH_freqChange()
	prevTime=time.time() 
	while (time.time() - prevTime)<=stsTime:
		cycle_time = time.time()
		FES1.pulse('red',RH_stsRed_i.get(), int(RH_stsRed_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		FES1.pulse('blue',RH_stsBlue_i.get(), int(RH_stsBlue_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		time.sleep(max(0, (1 / FES1_freq) - (time.time() - cycle_time)))
		#time.sleep(1/FES1_freq)

def RH_LH_stsBlackWhite():
	RH_LH_freqChange()
	prevTime=time.time() 
	while (time.time() - prevTime)<=stsTime:
		cycle_time = time.time()
		FES1.pulse('black',LH_stsBlack_i.get(), int(LH_stsBlack_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		FES1.pulse('white',LH_stsWhite_i.get(), int(LH_stsWhite_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		time.sleep(max(0, (1 / FES1_freq) - (time.time() - cycle_time)))
		#time.sleep(1/FES1_freq)
## Motor

def RH_LH_mtsRed():
	RH_LH_freqChange()
	prevTime=time.time() 
	while (time.time() - prevTime)<=mtsTime:
		cycle_time = time.time()
		FES1.pulse('red',RH_mtsRed_i.get(), int(RH_mtsRed_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		time.sleep(max(0, (1 / FES1_freq) - (time.time() - cycle_time)))
		#time.sleep(1/FES1_freq)
def RH_LH_mtsBlack():
	RH_LH_freqChange()
	prevTime=time.time() 
	while (time.time() - prevTime)<=mtsTime:
		cycle_time = time.time()
		FES1.pulse('black',LH_mtsBlack_i.get(), int(LH_mtsBlack_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		time.sleep(max(0, (1 / FES1_freq) - (time.time() - cycle_time)))
		#time.sleep(1/FES1_freq)

def RH_LH_mtsBlue():
	RH_LH_freqChange()
	prevTime=time.time() 
	while (time.time() - prevTime)<=mtsTime:
		cycle_time = time.time()
		FES1.pulse('blue',RH_mtsBlue_i.get(), int(RH_mtsBlue_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		time.sleep(max(0, (1 / FES1_freq) - (time.time() - cycle_time)))
		#time.sleep(1/FES1_freq)

def RH_LH_mtsWhite():
	RH_LH_freqChange()
	prevTime=time.time() 
	while (time.time() - prevTime)<=mtsTime:
		cycle_time = time.time()
		FES1.pulse('white',LH_mtsWhite_i.get(), int(LH_mtsWhite_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		time.sleep(max(0, (1 / FES1_freq) - (time.time() - cycle_time)))
		#time.sleep(1/FES1_freq)

def RH_LH_mtsRedBlue():
	RH_LH_freqChange()
	prevTime=time.time() 
	while (time.time() - prevTime)<=mtsTime:
		cycle_time = time.time()
		FES1.pulse('red',RH_mtsRed_i.get(), int(RH_mtsRed_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		FES1.pulse('blue',RH_mtsBlue_i.get(), int(RH_mtsBlue_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		time.sleep(max(0, (1 / FES1_freq) - (time.time() - cycle_time)))
		#time.sleep(1/FES1_freq)

def RH_LH_mtsBlackWhite():
	RH_LH_freqChange()
	prevTime=time.time() 
	while (time.time() - prevTime)<=mtsTime:
		cycle_time = time.time()
		FES1.pulse('black',LH_mtsBlack_i.get(), int(LH_mtsBlack_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		FES1.pulse('white',LH_mtsWhite_i.get(), int(LH_mtsWhite_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		time.sleep(max(0, (1 / FES1_freq) - (time.time() - cycle_time)))
		#time.sleep(1/FES1_freq)

# Full Sequence STS/MTS

def RH_LH_redBlueseq():
	RH_LH_freqChange()
	prevTime=time.time() 
	while (time.time() - prevTime)<=stsTime:
		cycle_time = time.time()
		FES1.pulse('red',RH_stsRed_i.get(), int(RH_stsRed_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		time.sleep(max(0, (1 / FES1_freq) - (time.time() - cycle_time)))
		#time.sleep(1/FES1_freq)
	prevTime=time.time() 
	while (time.time() - prevTime)<=stsTime:
		cycle_time = time.time()
		FES1.pulse('red',RH_stsRed_i.get(), int(RH_stsRed_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		FES1.pulse('blue',RH_stsBlue_i.get(), int(RH_stsBlue_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		time.sleep(max(0, (1 / FES1_freq) - (time.time() - cycle_time)))
		#time.sleep(1/FES1_freq)
	prevTime=time.time() 
	while (time.time() - prevTime)<=mtsTime:
		cycle_time = time.time()
		FES1.pulse('red',RH_mtsRed_i.get(), int(RH_mtsRed_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		FES1.pulse('blue',RH_mtsBlue_i.get(), int(RH_mtsBlue_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		time.sleep(max(0, (1 / FES1_freq) - (time.time() - cycle_time)))
		#time.sleep(1/FES1_freq)
def RH_LH_blackWhiteseq():
	RH_LH_freqChange()
	prevTime=time.time() 
	while (time.time() - prevTime)<=stsTime:
		cycle_time = time.time()
		FES1.pulse('black',LH_stsBlack_i.get(), int(LH_stsBlack_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		time.sleep(max(0, (1 / FES1_freq) - (time.time() - cycle_time)))
		#time.sleep(1/FES1_freq)
	prevTime=time.time() 
	while (time.time() - prevTime)<=stsTime:
		cycle_time = time.time()
		FES1.pulse('black',LH_stsBlack_i.get(), int(LH_stsBlack_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		FES1.pulse('white',LH_stsWhite_i.get(), int(LH_stsWhite_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		time.sleep(max(0, (1 / FES1_freq) - (time.time() - cycle_time)))
		#time.sleep(1/FES1_freq)
	prevTime=time.time() 
	while (time.time() - prevTime)<=mtsTime:
		cycle_time = time.time()
		FES1.pulse('black',LH_mtsBlack_i.get(), int(LH_mtsBlack_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		FES1.pulse('white',LH_mtsWhite_i.get(), int(LH_mtsWhite_p.get())) # 3 250+int(((t_passed - t_init)/STS_duration[0])*250)
		time.sleep(max(0, (1 / FES1_freq) - (time.time() - cycle_time)))
		#time.sleep(1/FES1_freq)

def endSetup():
    # Prepare the configuration dictionary
	# Load the configuration
	with open(filename, "r") as file:
	    conf = json.load(file)


	conf["FES_frequency"] = FES1_freq
	# Update pulse width for a channel
	conf["channels"]["red"]["Sensory_current_mA"] = RH_stsRed_i.get()
	conf["channels"]["red"]["Motor_current_mA"] = RH_mtsRed_i.get()
	conf["channels"]["red"]["duration_sense"] = config.TIME_MI
	conf["channels"]["red"]["duration_Motor"] = config.TIME_ROB - config.FES_TIMING_OFFSET
	conf["channels"]["red"]["pulse_width"] = RH_stsRed_p.get()
	
	conf["channels"]["blue"]["Sensory_current_mA"] = RH_stsBlue_i.get()
	conf["channels"]["blue"]["Motor_current_mA"] = RH_mtsBlue_i.get()
	conf["channels"]["blue"]["duration_sense"] = config.TIME_MI
	conf["channels"]["blue"]["duration_Motor"] = config.TIME_ROB - config.FES_TIMING_OFFSET
	conf["channels"]["blue"]["pulse_width"] = RH_stsBlue_p.get()
	
	conf["channels"]["black"]["Sensory_current_mA"] = LH_stsBlack_i.get()
	conf["channels"]["black"]["Motor_current_mA"] = LH_mtsBlack_i.get()
	conf["channels"]["black"]["duration_sense"] = config.TIME_MI
	conf["channels"]["black"]["duration_Motor"] = config.TIME_ROB - config.FES_TIMING_OFFSET
	conf["channels"]["black"]["pulse_width"] = LH_stsBlack_p.get()
	
	
	
	# Save the updated configuration
	with open(filename, "w") as file:
	    json.dump(conf, file, indent=4)

	print("Configuration updated!")

###########################################################################################################
#### SENSORY-THRESHOLD ####
###########################
tempL = Label(root, text="Stimulation Frequency").grid(row = 0, column = 0)
freqSlider = Scale( root, variable = freq , from_ = 10, to = 500, length = 300, resolution = 1, orient = HORIZONTAL)
freqSlider.set(30)
freqSlider.grid(row = 0,column=1)

shiftCol = 2;

fspace = tk.Frame(root)
tempL = Label(fspace)
fspace.grid(row = shiftCol +0,column=1, sticky="nsew")
tempL.pack(pady = 20)

tempL = Label(root, text="Sensory-Threshold Stimulation").grid(row = shiftCol + 0, column = 0)
current = Label(root, text = 'Current').grid(row = shiftCol + 0, column = 1)
pulseWidth = Label(root, text = 'Pulse Width').grid(row = shiftCol + 0, column = 2)


buttonReds = Button(root, text = " STS @ red (RH)", command = RH_LH_stsRed).grid(row = shiftCol +1,column=0)
redChannel_sts_i = Scale( root, variable = RH_stsRed_i , from_ = 0, to = 30, length = 300, resolution = 0.5, orient = HORIZONTAL).grid(row = shiftCol + 1,column=1)
redChannel_sts_p = Scale( root, variable = RH_stsRed_p , from_ = 10, to = 500, length = 600, resolution = 1, orient = HORIZONTAL)
redChannel_sts_p.set(250)
redChannel_sts_p.grid(row = shiftCol +1,column=2)
buttonBlues = Button(root, text = "STS @ blue (RH)", command = RH_LH_stsBlue).grid(row = shiftCol +2,column=0)
blueChannel_sts_i = Scale( root, variable = RH_stsBlue_i , from_ = 0, to = 30, length = 300, resolution = 0.5, orient = HORIZONTAL).grid(row = shiftCol +2,column=1)
blueChannel_sts_p = Scale( root, variable = RH_stsBlue_p , from_ = 10, to = 500, length = 600, resolution = 1, orient = HORIZONTAL)
blueChannel_sts_p.set(250)
blueChannel_sts_p.grid(row = shiftCol +2,column=2)

f1 = tk.Frame(root)
buttonRedBlues = Button(f1, text = "STS @ red & blue", command = RH_LH_stsRedBlue)
f1.grid(row = shiftCol +3,column=1, sticky="nsew")
buttonRedBlues.pack(anchor = CENTER)

buttonBlacks = Button(root, text = "STS @ black (LH) ", command = RH_LH_stsBlack).grid(row = shiftCol +4,column=0)
blackChannel_sts_i = Scale( root, variable = LH_stsBlack_i , from_ = 0, to = 30, length = 300, resolution = 0.5, orient = HORIZONTAL).grid(row = shiftCol +4,column=1)
blackChannel_sts_p = Scale( root, variable = LH_stsBlack_p , from_ = 10, to = 500, length = 600, resolution = 1, orient = HORIZONTAL)
blackChannel_sts_p.set(250)
blackChannel_sts_p.grid(row = shiftCol +4,column=2)
buttonWhites = Button(root, text = "STS @ white (LH)", command = RH_LH_stsWhite).grid(row = shiftCol +5,column=0)
whiteChannel_sts_i = Scale( root, variable = LH_stsWhite_i , from_ = 0, to = 30, length = 300, resolution = 0.5, orient = HORIZONTAL).grid(row = shiftCol +5,column=1)
whiteChannel_sts_p = Scale( root, variable = LH_stsWhite_p , from_ = 10, to = 500, length = 600, resolution = 1, orient = HORIZONTAL)
whiteChannel_sts_p.set(250)
whiteChannel_sts_p.grid(row = shiftCol +5,column=2)

f2 = tk.Frame(root)
buttonBlackWhites = Button(f2, text = "STS @ black & white", command = RH_LH_stsBlackWhite)
f2.grid(row = shiftCol +6,column=1, sticky="nsew")
buttonBlackWhites.pack(anchor = CENTER)

fspace = tk.Frame(root)
tempL = Label(fspace)
fspace.grid(row = shiftCol +8,column=1, sticky="nsew")
tempL.pack(pady = 20)

###########################################################################################################
#### MOTOR-THRESHOLD ####
#########################

fspace = tk.Frame(root)
tempL = Label(fspace)
fspace.grid(row = shiftCol +10,column=1, sticky="nsew")
tempL.pack(pady = 20)

tempL = Label(root, text="Motor-Threshold Stimulation").grid(row = shiftCol + 10, column = 0)
current = Label(root, text = 'Current').grid(row = shiftCol + 10, column = 1)
pulseWidth = Label(root, text = 'Pulse Width').grid(row = shiftCol + 10, column = 2)


buttonRedm = Button(root, text = " MTS @ red (RH)", command = RH_LH_mtsRed).grid(row = shiftCol +11,column=0)
redChannel_mts_i = Scale( root, variable = RH_mtsRed_i , from_ = 0, to = 30, length = 300, resolution = 0.5, orient = HORIZONTAL).grid(row = shiftCol +11,column=1)
redChannel_mts_p = Scale( root, variable = RH_mtsRed_p , from_ = 10, to = 500, length = 600, resolution = 1, orient = HORIZONTAL)
redChannel_mts_p.set(250)
redChannel_mts_p.grid(row = shiftCol +11,column=2)

buttonBlackm = Button(root, text = "MTS @ blue (RH)", command = RH_LH_mtsBlue).grid(row = shiftCol +12,column=0)
blueChannel_mts_i = Scale( root, variable = RH_mtsBlue_i , from_ = 0, to = 30, length = 300, resolution = 0.5, orient = HORIZONTAL).grid(row = shiftCol +12,column=1)
blueChannel_mts_p = Scale( root, variable = RH_mtsBlue_p , from_ = 10, to = 500, length = 600, resolution = 1, orient = HORIZONTAL)
blueChannel_mts_p.set(250)
blueChannel_mts_p.grid(row = shiftCol +12,column=2)


f1 = tk.Frame(root)
buttonRedBluem = Button(f1, text = "MTS @ red & blue", command = RH_LH_mtsRedBlue)
f1.grid(row = shiftCol +13,column=1, sticky="nsew")
buttonRedBluem.pack(anchor = CENTER)

buttonBlackm = Button(root, text = "MTS @ black (LH)", command = RH_LH_mtsBlack).grid(row = shiftCol +14,column=0)
blackChannel_mts_i = Scale( root, variable = LH_mtsBlack_i , from_ = 0, to = 30, length = 300, resolution = 0.5, orient = HORIZONTAL).grid(row = shiftCol +14,column=1)
blackChannel_mts_p = Scale( root, variable = LH_mtsBlack_p , from_ = 10, to = 500, length = 600, resolution = 1, orient = HORIZONTAL)
blackChannel_mts_p.set(250)
blackChannel_mts_p.grid(row = shiftCol +14,column=2)

buttonWhitem = Button(root, text = "MTS @ white (LH)", command = RH_LH_stsWhite).grid(row = shiftCol +15,column=0)
whiteChannel_mts_i = Scale( root, variable = LH_mtsWhite_i , from_ = 0, to = 30, length = 300, resolution = 0.5, orient = HORIZONTAL).grid(row = shiftCol +15,column=1)
whiteChannel_mts_p = Scale( root, variable = LH_mtsWhite_p , from_ = 10, to = 500, length = 600, resolution = 1, orient = HORIZONTAL)
whiteChannel_mts_p.set(250)
whiteChannel_mts_p.grid(row = shiftCol +15,column=2)


f2 = tk.Frame(root)
buttonBlackWhitem = Button(f2, text = "MTS @ black & white", command = RH_LH_mtsBlackWhite)
f2.grid(row = shiftCol +16,column=1, sticky="nsew")
buttonBlackWhitem.pack(anchor = CENTER)

fspace = tk.Frame(root)
tempL = Label(fspace)
fspace.grid(row = shiftCol +17,column=1, sticky="nsew")
tempL.pack(pady = 20)

buttonRedBlueseq = Button(root, text = "Red/Blue Sequence", command = RH_LH_redBlueseq).grid(row = shiftCol +18,column=0)
buttonBlackWhiteseq = Button(root, text = "Black/White Sequence", command = RH_LH_blackWhiteseq).grid(row = shiftCol +18,column=1)
buttonEnd = Button(root, text = "End", command = endSetup).grid(row = shiftCol +18,column=2)

root.mainloop() 
