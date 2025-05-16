import os
import sys
import time

# === SETUP CORRECT PATH ===
dirP = os.path.abspath(os.getcwd())
sys.path.append(dirP + '/STM_interface/1_packages/rehamoveLibrary')
print("Loaded rehamove from:", dirP + '/STM_interface/1_packages/rehamoveLibrary')

# === IMPORT LIBRARY ===
from rehamove import *  # Import rehamove library

# === CONFIGURATION PARAMETERS ===
FES_freq = 30       # Hz
chnName1 = 'red'   # channel name to stimulate

pulseWidth = 100    # microseconds
rampTime = 5        # seconds
TESS_dur = 0.25        # minutes
burst = 10          # number of high-frequency pulses per cycle

# === INITIALIZE DEVICE ===
r = Rehamove("/dev/ttyUSB0")  # adjust if needed
r.battery()  # check battery status

# === MAIN INTERACTION LOOP ===
while True:
    resp = int(input(
        "\nWhich stim type do you want to test:\n"
        "[1] Single-pulse FES every 30Hz\n"
        "[2] 5kHz TESS pulses every 30Hz\n"
        "[3] Low-burst TESS pulses every 30Hz\n"
        "[4] Quit\nResponse: "
    ))

    if resp == 4:
        print("Quitting program.")
        break

    current = int(input("Current (mA): "))

    if resp == 1:
        # Single-pulse FES
        prevTime = time.time()
        while (time.time() - prevTime) <= 2:
            r.pulse(chnName1, current, pulseWidth)
            time.sleep(1 / FES_freq)

    elif resp in [2, 3]:
        # Prepare burst configuration
        num_bursts = burst if resp == 2 else 2

        # === RAMP UP ===
        prevTime = time.time()
        while (time.time() - prevTime) <= rampTime:
            curr = current * (time.time() - prevTime) / rampTime
            pulseNew = [
                [curr if i % 2 == 0 else -curr, pulseWidth]
                for i in range(num_bursts)
            ]
            print(f"Ramping Up Current: {pulseNew[0][0]:.2f} mA")
            r.custom_pulse(chnName1, pulseNew)
            time.sleep(max(0, 1 / FES_freq - pulseWidth * num_bursts / 1e6))

        # === CONSTANT TESS BURST ===
        pulseNew = [
            [current if i % 2 == 0 else -current, pulseWidth]
            for i in range(num_bursts)
        ]
        print("Running TESS at target current...")
        prevTime = time.time()
        while (time.time() - prevTime) <= TESS_dur * 60:
            r.custom_pulse(chnName1, pulseNew)
            time.sleep(max(0, 1 / FES_freq - pulseWidth * num_bursts / 1e6))

        # === RAMP DOWN ===
        prevTime = time.time()
        while (time.time() - prevTime) <= rampTime:
            curr = current * (1 - (time.time() - prevTime) / rampTime)
            pulseNew = [
                [curr if i % 2 == 0 else -curr, pulseWidth]
                for i in range(num_bursts)
            ]
            print(f"Ramping Down Current: {pulseNew[0][0]:.2f} mA")
            r.custom_pulse(chnName1, pulseNew)
            time.sleep(max(0, 1 / FES_freq - pulseWidth * num_bursts / 1e6))

    else:
        print("Invalid selection. Please try again.")
