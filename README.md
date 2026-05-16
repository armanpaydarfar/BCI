## 1. Project Overview

This repository contains a research-oriented **EEG-based brain–computer
interface (BCI)** stack used to drive the **Harmony robot** — a
rehabilitation upper-body exoskeleton (initially developed by the
**ReNu Lab at UT Austin**) — in BCI studies.

The Harmony robot itself is not an EEG/BCI system. The “Harmony project”
in this repo is the software stack that combines:

- **Motor-imagery EEG decoding** for high-level intent (move vs rest),
  built on a Riemannian-geometry pipeline (Ledoit–Wolf shrinkage
  covariances → adaptive recentering → tangent space / MDM-class
  classifiers).
- **Real-time experiment drivers** that orchestrate the closed-loop
  trial structure (cue display, decoding, feedback, actuation).
- **Networking and actuation** layers that translate decoded intent
  into safe robot motion and optional FES (Functional Electrical
  Stimulation) commands.

The primary use case is **assistive robotics and rehabilitation
research**, including studies with motor-impaired participants under
appropriate clinical and ethical oversight.

## 2. System Architecture (High-Level)

- **Control Panel (GUI launcher)**
  - `control_panel.py`, PySide6.
  - Single interface to start/stop experiment drivers, the marker
    stream, the FES listener, and STM setup tools, and to edit a small
    set of fields in `config.py` (subject ID, `SIMULATION_MODE`,
    `FES_toggle`).

- **Experiment Drivers**
  - `ExperimentDriver_Online.py` — main online motor-imagery loop.
  - `ExperimentDriver_Offline.py` — offline replay / analysis.
  - `ExperimentDriver_Bimanual.py` — bimanual variant.
  - Each driver: connects to EEG via LSL, runs a Pygame-based feedback
    UI, classifies motor imagery vs rest in real time, and triggers
    robot/FES actions over UDP.

- **Runtime and utilities (`Utils/`)**
  - `Utils/runtime_common.py` — shared realtime decoder dispatch and
    trial-state helpers used by all experiment drivers.
  - `Utils/EEGStreamState.py` — wraps the LSL EEG inlet and maintains
    a rolling buffer for windowed classification.
  - `Utils/preprocessing.py` — causal streaming filters
    (`apply_streaming_filters` with persistent `lfilter` state),
    rereferencing, segment extraction.
  - `Utils/experiment_utils.py` — trial sequence generation, rolling
    normalisation, adaptive Riemannian transforms.
  - `Utils/networking.py` — UDP networking for robot, FES, and marker
    streams, ACK handling, and simulation gating.
  - `Utils/stream_utils.py`, `Utils/visualization.py`,
    `Utils/logging_manager.py`, `Utils/Montage_creator.py` — supporting
    helpers.

- **Robot control (UDP)**
  - Robot opcodes and trigger mappings are defined in `config.py`
    (`ROBOT_OPCODES`, `TRIGGERS`).
  - `Utils/networking.py`, `UDPRobot.py`, and `udp_send.py` send robot
    commands over UDP with ACK / safety gating (staged trajectories +
    `GO` confirmation pattern).

- **FES / STM interface**
  - **FES** = Functional Electrical Stimulation, used for somatosensory
    feedback during the real-time loop. Gated by `FES_toggle` in
    `config.py`.
  - `FES_listener.py` listens for UDP commands and dispatches stimulator
    actions.
  - `STM_interface/` contains the Rehamove/STM device library, JSON
    config, and serial communication code.
  - `STMsetup.py` configures the stimulator hardware.

### Data flow at a glance

1. **Acquisition** — EEG via LSL into `EEGStreamState`.
2. **Real-time decoding** — windowed Riemannian / adaptive features in
   the experiment drivers via `runtime_common`.
3. **Feedback** — Pygame UI + loggers.
4. **Actuation** — UDP commands to the robot, FES, and marker stream
   for synchronisation with downstream recording (e.g. LabRecorder).

## 3. Repository Structure

Key folders and scripts (non-exhaustive):

- **`control_panel.py`** — Qt launcher for experiments, robot/FES
  tools, and supporting utilities.

- **Experiment drivers (root-level)**
  - `ExperimentDriver_Online.py` — main online motor-imagery loop.
  - `ExperimentDriver_Offline.py` — offline replay / analysis driver.
  - `ExperimentDriver_Bimanual.py` — bimanual variant of the online
    driver.
  - `.sh` wrappers (`ExperimentDriver_Online.sh` etc.) — convenience
    launchers used by the control panel.

- **Training / model generation**
  - `Generate_Riemannian_adaptive.py` — **current canonical training
    script.** Builds the adaptive Riemannian decoder used at runtime,
    including the streaming/adaptive recentering transforms. Produces
    `sub-*_model.pkl` artefacts consumed by the realtime drivers.
  - `Generate_Riemannian.py` — non-adaptive MDM baseline (kept for
    comparison; the adaptive version is the load-bearing one).
  - `Generate_Decoder.py` — legacy LDA baseline (kept for reference).

- **Analysis and visualisation**
  - `Analyze_online_Decoder_performance.py` — log/performance analysis
    across sessions.
  - `Visualize_offline_data_MNE.py`, `visualize_online_data.py` — ERD /
    TFR plots and topomaps. `visualize_online_data.py` uses multitaper
    TFR with per-trial logratio baseline normalisation.
  - `Visualize_offline_data.py` — legacy visualisation script.

- **`Utils/`** — core runtime, EEG, preprocessing, networking, and
  supporting modules (see Section 2).

- **`STM_interface/`** — Rehamove/STM configuration
  (`RehamoveConfig.py`, JSON) and serial communication
  (`serialCommunication.py`, `rehamoveLibrary/`).

- **Robot / FES / marker utilities (root-level)**
  - `UDPRobot.py` — standalone manual console for robot UDP testing.
  - `udp_send.py` — generic UDP sender utility.
  - `UTIL_marker_stream.py` — LSL marker stream output.
  - `FES_listener.py`, `STMsetup.py` — FES/STM listener and setup tools.

- **Configuration and data**
  - `config.py` — central configuration module (EEG, robot, FES,
    networking, display, decoder).
  - `Data/` — empty placeholder at the root; experiment data and logs
    live under the directory configured as `DATA_DIR` in `config.py`
    (default: `~/Documents/CurrentStudy`).

### Standard data layout (under `config.DATA_DIR`)

Typical per-subject layout (`sub-<SUBJECT_ID>`):

- `marker_logs/` — logs from the marker utility.
- `ses-Debug/` — sessions recorded without LabRecorder streams.
- `training_data/` — `.xdf` recordings used for model and transform
  generation.
- `models/` — saved model artefacts and session-specific folders.
- `ses-<SESSION_NAME>/` — per-session data directories.
  - `eeg/` — recorded EEG `.xdf` files.
  - `logs/<RUN>/` — per-run protocol logs (`config_snapshot.json`,
    `decoder_output.csv`, `event_log.txt`, `trial_summary.csv`).

## 4. Setup / Environment

This branch does not ship an `environment.yml`. A working environment is
typically configured manually with conda, targeting Python 3.12, with
the following major dependencies:

- **Scientific Python** — numpy, scipy, pandas, scikit-learn.
- **EEG / signal processing** — `mne`, `pyriemann`, `mne-lsl`, `pylsl`.
- **Realtime UI** — `pygame`, `PySide6` (or `PyQt5`).
- **Visualisation** — `matplotlib`, `seaborn`.
- **Serial / hardware** — `pyserial` (for STM/Arduino-class devices).

### External dependencies (non-exhaustive)

- **EEG acquisition**
  - **eegoSports** (Linux build) streaming EEG over **LSL**. The repo
    is configured for a 32-channel cap (`CAP_TYPE = 32`, `FS = 512` Hz).
  - **LabRecorder** to capture all relevant LSL streams (EEG, markers,
    auxiliary streams) into `.xdf`.
  - `mne-lsl viewer` for inspecting LSL streams during setup.
- **Robot control**
  - Network connectivity to the robot controller at the IP/port
    specified by `config.UDP_ROBOT`.
  - SSH access for some helper commands launched by `control_panel.py`.
- **FES / STM**
  - Rehamove/STM hardware compatible with `STM_interface/`.

### Hardware setup

Exact wiring, electrode placement, and safety procedures are
hardware- and lab-specific and are **not defined by this repository**.
Follow your institution's protocols.

## 5. Running the System

> **Safety note:** Many scripts in this repo can command real hardware
> (robot, FES). Always verify `config.py` (especially `SIMULATION_MODE`
> and `FES_toggle`) and hardware status before running an experiment.

### Primary entry point: Control Panel

```
python control_panel.py
```

The control panel provides buttons to:

- Start the marker stream.
- Start the FES listener (if FES is enabled).
- Launch the experiment driver (online / offline / bimanual variants).
- Edit subject ID, `SIMULATION_MODE`, and `FES_toggle` in `config.py`.

### Online experiment (typical flow)

1. Configure subject and toggles in the control panel.
2. Ensure the marker stream and (if FES enabled) FES listener are
   running.
3. Select `ExperimentDriver_Online` and the desired mode.
4. Start the driver from the control panel.

### Offline experiment

`ExperimentDriver_Offline.py` replays and analyses previously recorded
data. It does not control hardware but still relies on `config.py` for
paths and decoder parameters.

### Transform / model generation (offline)

`Generate_Riemannian_adaptive.py`:

- Trains the adaptive Riemannian decoder used by the online drivers.
- Expects `.xdf` recordings created by LabRecorder, with both an EEG
  stream and a marker stream so it can align segments by experimental
  events.
- Processes each EEG file in `training_data/` under the configured
  subject folder, including the whitening / adaptive recentering
  transforms.
- Produces a model artefact (`sub-*_model.pkl`) loaded by the online
  driver at runtime.

## 6. Configuration

`config.py` is the single source of truth for run-time behaviour.
Notable groups:

- **EEG** — `FS` (sampling rate), `CAP_TYPE` (32), `LOWCUT`/`HIGHCUT`
  (mu band, 8-13 Hz by default), `MOTOR_CHANNEL_NAMES`,
  `CLASSIFY_WINDOW`, `FILTER_BUFFER_SIZE`, baseline window.
- **Experiment** — `TOTAL_TRIALS`, trial timings (`TIME_MI`,
  `TIME_ROB`, etc.), `THRESHOLD_MI`, `THRESHOLD_REST`,
  `RELAXATION_RATIO`, early-stop policy.
- **Robot** — UDP endpoints (`UDP_ROBOT`, `UDP_CONTROL_BIND`), opcodes
  (`ROBOT_OPCODES`), software trigger codes (`TRIGGERS`).
- **FES / STM / markers** — UDP endpoints, `FES_toggle`.
- **Toggles**
  - `SIMULATION_MODE` — when `True`, networking utilities suppress
    robot UDP commands and ACK waits while still emitting marker and
    FES messages. Use for dry runs.
  - `FES_toggle` — gates FES command emission.
  - `TRAINING_SUBJECT` — controls which subject's models and training
    data are loaded.

### Machine-local paths

`WORKING_DIR` and `DATA_DIR` in `config.py` point at lab-specific
locations and must be edited per machine before running anything.

## 7. Real-Time / Hardware Considerations (IMPORTANT)

Parts of this system directly control real hardware (robot via UDP/SSH;
FES/STM via serial) and run inside real-time loops with timing
guarantees. When developing or modifying:

- Prefer `SIMULATION_MODE=True` and FES disabled until behaviour is
  well understood.
- Avoid blocking operations, heavy allocations, or extra I/O inside
  realtime loops.
- Treat changes to networking, ACK logic, trigger mappings, FES, or
  the adaptive transforms as safety-critical and review call paths
  carefully.
- For the conventions this codebase follows for high-risk files and
  realtime safety, see `CLAUDE.md`.

## 8. Status / Notes

This branch (`harmony_stable`) is a snapshot of the core online motor-
imagery + robot pipeline. Active development continues on
`harmony_dev`, which adds gaze-based selection, Harmony-specific
calibration / online-control scripts, additional experiment variants
(glove, gaze-tracking, ErrP), and broader analysis tooling. `main` is
intended to track stable snapshots suitable for reference and external
review.

- Some scripts (e.g. `Test_TESS.py`, the legacy `Generate_Decoder.py`
  and `Generate_Riemannian.py`) are kept for reference or comparison
  and are not part of the day-to-day online pipeline.
- Hardware availability and local lab practices vary; specific scripts
  may be used only in certain study setups.
- Before modifying behaviour:
  - Trace how a script is launched (e.g. from `control_panel.py`).
  - Identify which utilities and configuration fields it depends on
    (`config.py`, `Utils/networking.py`, `Utils/runtime_common.py`).
  - Consider the real-time and hardware implications, and always test
    in simulation or low-risk conditions before running with
    participants or full hardware.
