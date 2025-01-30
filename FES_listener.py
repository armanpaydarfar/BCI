import time
import json
import socket
import sys
import os
from pathlib import Path
import config
import select

dirP = os.path.abspath(os.getcwd())
sys.path.append(dirP + '/STM_interface/1_packages/rehamoveLibrary')
from rehamove import *  # Import our library

# Configuration
filename = 'STM_interface/RehamoveConfig_simple.json'
with open(filename, "r") as file:
    fes_config = json.load(file)

FES_device = Rehamove("/dev/ttyUSB0")  # Update port as necessary
FES_frequency = fes_config["FES_frequency"]

# Set up UDP socket
UDP_IP = config.UDP_FES["IP"]  # Use loopback IP for local communication
UDP_PORT = config.UDP_FES["PORT"]       # Use the same port as in your EEG script
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

def trigger_fes(channel_name, mode = 'SENSORY'):
    """Trigger FES for a specific channel and duration."""
    channel_config = fes_config["channels"].get(channel_name, {})
    if not channel_config:
        print(f"No configuration found for channel: {channel_name}")
        return

    current = channel_config["Sensory_current_mA"] if mode == 'SENSORY' else channel_config["Motor_current_mA"] #utilize sensory threshold or motor threshold depending on mode
    duration = channel_config["duration_sense"] if mode == 'SENSORY' else channel_config["duration_Motor"] # utilize MI/rest duration or robot movement duration
    pulse_width = int(channel_config["pulse_width"])  # Ensure pulse_width is an integer

    print(f"Triggering FES on {channel_name} for {duration} seconds.")
    start_time = time.time()

    while time.time() - start_time < duration:
        # Check for UDP messages without blocking
        cycle_time = time.time()
        readable, _, _ = select.select([sock], [], [], 0)  # polling appreach with 0s timeout - minimal latency
        if readable:
            data, addr = sock.recvfrom(1024)
            trigger = str(data.decode()).strip()
            if trigger == "FES_STOP":
                print("Received FES_STOP. Stopping stimulation.")
                break

        FES_device.pulse(channel_name, current, pulse_width)
        time.sleep(max(0, (1 / FES_frequency) - (time.time() - cycle_time)))
        #time.sleep(1/FES_frequency)
    print(f"Stimulation on {channel_name} completed. Returning to listening mode.")

while True:
    # Receive a trigger
    data, addr = sock.recvfrom(1024)  # Buffer size is 1024 bytes
    trigger = str(data.decode()).strip()  # Decode the received message and strip extra spaces

    if trigger == "ping":  # Respond to ping
        print("Received ping. Listener is active.")
        sock.sendto(b"pong", addr)  # Reply with a pong
    elif trigger == "FES_SENS_GO":  # Red arrow trigger
        channel = config.FES_CHANNEL
        trigger_fes(channel, mode = 'SENSORY')
    elif trigger == "FES_MOTOR_GO":
        channel = config.FES_CHANNEL
        trigger_fes(channel, mode = "MOTOR")
    else:
        print(f"Received unrecognized trigger: {trigger}")

    # After processing the trigger, the script continues listening for new triggers

