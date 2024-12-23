#!/bin/bash

# Path to Conda activation script
CONDA_SCRIPT="/home/arman-admin/opt/miniconda/etc/profile.d/conda.sh"

# Set this to the path of the script directory
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Function to open a new terminal and run a command
open_terminal() {
  local cmd=$1
  gnome-terminal -- bash -c "source $CONDA_SCRIPT && conda activate mne && $cmd; exec bash"
}

read -p "Is Harmony already initialized? (default yes) (y/n): " harmony_initialized
read -p "Do you want to start mne-lsl viewer? (default no) (y/n): " start_mne

echo "Initializing experiment devices..."

# Step 1: Start eegoSports
echo "Starting eegoSports..."
open_terminal "eegoSports"

# Step 2: Run Util_marker_stream.py
echo "Running Util_marker_stream.py..."
open_terminal "python3 $SCRIPT_DIR/UTIL_marker_stream.py"

# Step 3: Start LabRecorder
echo "Starting LabRecorder..."
open_terminal "LabRecorder"

# Step 4: Optionally start mne-lsl viewer
if [[ "$start_mne" == "y" ]]; then
  echo "Starting mne-lsl viewer..."
  open_terminal "mne-lsl viewer"
else
  echo "Skipping mne-lsl viewer."
fi

# Step 5: SSH into the robot and execute commands if Harmony is not already initialized
if [[ "$harmony_initialized" == "n" ]]; then
  echo "Connecting to the robot via SSH and running commands..."
  open_terminal "sshpass -p 'Harmonic-03' ssh -tt root@192.168.2.1 'cd /opt/hbi/dev/bin && ./killall.sh && sleep 10 && ./run.sh && exec bash'"

  sleep 60
  echo "Starting harmony process..."
  open_terminal "sshpass -p 'Harmonic-03' ssh -tt root@192.168.2.1 'cd /opt/hbi/dev/bin/tools && ./bmi_exercise && exec bash'"
else
  echo "Skipping Harmony initialization as it is already running."
fi

echo "Initialization complete. Check the opened terminals for individual processes."

