#!/bin/bash

# Parameters
FES_LISTENER_COMMAND="python FES_listener.py"
EEG_OFFLINE_COMMAND="python ExperimentDriver_Online.py"
CONDA_ENV="mne"  # Name of the conda environment

# Extract FES_TOGGLE, IP, and UDP_PORT from config.py
FES_TOGGLE=$(python -c "from config import FES_toggle; print(FES_toggle)")
UDP_IP=$(python -c "from config import UDP_FES; print(UDP_FES['IP'])")
UDP_PORT=$(python -c "from config import UDP_FES; print(UDP_FES['PORT'])")

echo "Configuration: FES_TOGGLE=$FES_TOGGLE, IP=$UDP_IP, PORT=$UDP_PORT"

if [ "$FES_TOGGLE" -eq 1 ]; then
    echo "FES is enabled. Checking for existing FES_listener.py process using ping..."

    # Send a ping to check if FES_listener.py is active and look for a "pong" response
    RESPONSE=$(echo "ping" | nc -u -w1 $UDP_IP $UDP_PORT)

    if [[ "$RESPONSE" == "pong" ]]; then
        echo "FES_listener.py responded to ping with pong. Continuing..."
    else
        echo "FES_listener.py is not responding. Starting a new instance in a separate terminal..."
        gnome-terminal -- bash -c "eval \"\$(conda shell.bash hook)\" && conda activate $CONDA_ENV && $FES_LISTENER_COMMAND; exec bash" &
    fi
else
    echo "FES is disabled. Skipping FES_listener.py setup."
fi

# Start EEG_offline.py in a separate terminal
echo "Starting EEG_offline.py in a new terminal..."
gnome-terminal -- bash -c "eval \"\$(conda shell.bash hook)\" && conda activate $CONDA_ENV && $EEG_OFFLINE_COMMAND; exec bash"

echo "EEG Offline Experiment initialized."

