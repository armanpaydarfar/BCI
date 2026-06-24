## 1. Project Overview

This repository adds a research-oriented **EEG-based brain–computer interface (BCI)** (and optional gaze/automation layers) to control the **Harmony robot**.

The **Harmony robot** itself is a **rehabilitation upper‑body exoskeleton** (initially developed by the **ReNu Lab at UT Austin**) and is not inherently an EEG/BCI system. The “Harmony project” in this repo is the software stack that **integrates EEG decoding, real-time experiment drivers, and networking/actuation** so the robot can be driven in BCI studies.

At a high level, the stack combines:

- **Motor-imagery EEG decoding** for high-level intent (move vs rest).
- **Gaze-based selection and depth estimation** to choose objects or targets.
- **Automation and safety logic** to translate intent into safe robot and FES actions.

The primary intended use cases are **assistive robotics and rehabilitation research**, including studies with populations such as ALS or motor-impaired participants, under appropriate clinical and ethical oversight.

## 2. System Architecture (High-Level)

- **Control Panel (GUI launcher)**  
  - Implemented in `control_panel.py` using PySide6.  
  - Provides a single interface to start/stop:
    - Online and offline experiment drivers (`ExperimentDriver_*.py`).
    - Marker stream, FES listener, STM setup tools.
    - Gaze runner/service and Harmony calibration/online control.
  - Edits `config.py` for subject ID, `SIMULATION_MODE`, and `FES_toggle`.

- **Experiment Drivers (online / offline / bimanual / variants)**  
  - Scripts such as `ExperimentDriver_Online.py`, `ExperimentDriver_Offline.py`, `ExperimentDriver_Bimanual.py`, `ExperimentDriver_Online_GazeTracking.py`, `ExperimentDriver_Online_Glove.py`.  
  - Implement the real-time experimental loops: connecting to EEG via LSL, drawing Pygame-based feedback, classifying motor imagery vs rest, and triggering robot/FES actions over UDP.

- **Runtime and utilities (`Utils/`)**  
  - `Utils/runtime_common.py`: shared runtime state and helper functions for classification, feedback, and trial management.  
  - `Utils/EEGStreamState.py`: wraps the EEG LSL inlet and maintains rolling buffers.  
  - `Utils/experiment_utils.py`: trial sequence generation, rolling normalization, adaptive Riemannian transforms.  
  - `Utils/networking.py`: UDP networking for robot and marker control, ACK handling, and simulation mode.  
  - Additional helpers for visualization, preprocessing, logging, and gaze math.

- **Gaze system**  
  - `gaze_runner.py` and `gaze_visualizer.py` orchestrate real-time gaze input and visualization.  
  - `Utils/gaze/` contains gaze tracking, math, UI, and rendering utilities.  
  - Pupil Labs **Neon** is the gaze/scene-camera device; Linux owns the device I/O
    and ships frames to the perception host (see below).

- **Perception subsystem (segmentation / depth / VLM reasoning)**  
  - `perception/` holds the vision stack — class-agnostic segmentation (FastSAM),
    metric depth (Depth Pro), gaze-fixation detection, and Gemini-backed VLM intent
    reasoning. It is **vendored from Vivian Chen's `harmony_vlm`** (see
    [`NOTICE.md`](NOTICE.md)) and runs as a **separate process**, never imported into
    the realtime EEG/robot loops.
  - `vlm_service.py` is the perception host service; `vlm_bridge.py` /
    `Utils/perception_clients.py` are the UDP request/reply clients the drivers use;
    `Utils/frame_relay.py` + `Utils/remote_frame_reader.py` ship Neon frames Linux→host
    over TCP. The split is config-gated (`PERCEPTION_FRAME_SOURCE`,
    `SERVICES_HOSTED_REMOTELY`); see
    `Documents/SoftwareDocs/projects/harmony-bci/gpu-service/architecture-plan.md`.

- **Robot control (UDP)**  
  - Robot opcodes and trigger mappings are defined in `config.py` (`ROBOT_OPCODES`, `TRIGGERS`).  
  - `Utils/networking.py`, `UDPRobot.py`, and related scripts send robot commands over UDP and manage acknowledgements and safety gating (e.g., staged trajectories + `GO`).

- **FES / STM interface**  
  - **FES** stands for **Functional Electrical Stimulation**: peripheral nerve stimulation used to provide **somatosensory feedback** during the real-time loop (e.g., for assistance with neuromodulation), controlled via the `FES_toggle` configuration.  
  - `FES_listener.py` listens for UDP commands to drive Functional Electrical Stimulation (FES).  
  - `STM_interface/` provides the Rehamove/STM configuration and serial drivers.  
  - `STMsetup.py` configures the stimulator hardware.

At a high level, data flows as follows:

1. **EEG and gaze acquisition** via LSL and vendor APIs into `EEGStreamState` and gaze utilities.  
2. **Real-time decoding** using trained models and adaptive transforms (e.g., Riemannian methods) in the experiment drivers and utilities.  
3. **Decision and feedback** via Pygame UI and loggers.  
4. **Actuation** by sending UDP/serial commands to the robot, FES/STM, and optional Arduino-based actuators, with triggers mirrored to marker streams for synchronization.

## 3. Repository Structure

Key folders and scripts (non-exhaustive):

- **`control_panel.py`**  
  - Qt-based launcher for experiments, robot/FES tools, gaze services, and Harmony calibration/online control.

- **Experiment drivers (root-level scripts)**  
  - `ExperimentDriver_Online.py`: main online motor-imagery BCI loop.  
  - `ExperimentDriver_Offline.py`: offline replay / analysis drivers.  
  - `ExperimentDriver_Bimanual.py`: bimanual variant.  
  - `ExperimentDriver_Online_GazeTracking.py`, `ExperimentDriver_Online_Glove.py`: gaze/glove-oriented variants.  
  - `harmony_calibration_exec.py`, `harmony_online_control.py`: Harmony-specific calibration and online control.  
  - `gaze_runner.py`, `gaze_visualizer.py`: gaze service and visualization tools.

- **`Utils/`**  
  - Core runtime and helper modules:
    - `runtime_common.py`, `EEGStreamState.py`, `experiment_utils.py`, `logging_manager.py`, `visualization.py`, `preprocessing.py`, `stream_utils.py`.  
    - `networking.py`: robot/marker/FES UDP networking.  
    - `gaze/`: gaze tracking, math, UI, and rendering.

- **`STM_interface/`**  
  - Rehamove/STM configuration (`RehamoveConfig.py`, JSON) and serial communication packages (`serialCommunication.py`, `rehamoveLibrary/`).

- **`perception/`** (vision stack; vendored from `harmony_vlm` — see [`NOTICE.md`](NOTICE.md))  
  - **Live** modules (wired into `vlm_service.py` / the frame relay, importable in the
    unified env): `object_detector.py` (FastSAM segmentation), `depth_estimator.py`
    (Depth Pro), `fixation_detector.py`, `intent_reasoner.py` (Gemini VLM),
    `pupil_reader.py`, `visualize_neon.py`, `neon/`.  
  - **Staged** modules are vendored for upcoming WS4/WS5 work and are deliberately
    **not import-safe** in the current env (their deps are excluded): `apriltag_detector.py`,
    `realsense_camera.py`, `gaze_grounder.py`, `overlay_renderer.py`, `exo_controller.py`,
    `core/`. The authoritative live-vs-staged list is the docstring in
    [`perception/__init__.py`](perception/__init__.py).
  - The perception process talks to the drivers only over UDP/TCP (`vlm_service.py`,
    `vlm_bridge.py`, `Utils/perception_clients.py`, `Utils/frame_relay.py`) — there is no
    in-process coupling to the realtime EEG/robot loops.

- **Robot / FES / markers (root-level scripts)**  
  - `UDPRobot.py`, `udp_send.py`, `UTIL_marker_stream.py`: robot and marker utilities.  
  - `FES_listener.py`, `STMsetup.py`: FES/STM listener and setup tools.

- **Analysis and training**  
  - `Generate_Riemannian_adaptive.py`: adaptive Riemannian / MDM training (current path; produces `sub-*_model.pkl`).  
  - Legacy trainers `Generate_Decoder.py` (LDA) and `Generate_Riemannian.py` (non-adaptive MDM) were **removed** from the repo—use the adaptive script above.  
  - `Analyze_experiment_logs_cross_subject.py`, `Analyze_online_Decoder_performance.py`: log and performance analysis.  
  - Visualization utilities:
    - `Visualize_offline_data_MNE.py`, `visualize_online_data.py`

#### Artifact rejection (offline training vs QC plots)

- **Sliding-window training** (`Generate_Riemannian_adaptive.py`, XGB pipelines via `Utils/xgb_feature_pipeline.py`) uses `Utils/artifact_rejection.py` with `config.py` knobs:
  - `ARTIFACT_REJECT_ENABLE`, `ARTIFACT_REJECT_MODE` (`max_abs`, `peak_to_peak`, `zscore`)
  - `ARTIFACT_MAX_ABS_UV`, `ARTIFACT_P2P_UV`, `ARTIFACT_ZSCORE_SD`
  - `ARTIFACT_SEGMENT_AMPLITUDE_UNIT` (`microvolts` vs `volts`, must match your XDF/filter pipeline scale)
- **`visualize_online_data.py`** applies **MNE epoch** peak-to-peak rejection on `Raw` in **volts**; thresholds are set in µV as `VISUALIZE_EPOCH_REJECT_P2P_UV` (and optional `VISUALIZE_EPOCH_FLAT_UV`) in `config.py`, converted to volts for `mne.Epochs`. That criterion is **not identical** to training-window `max_abs` unless you choose compatible `ARTIFACT_REJECT_MODE`/thresholds on purpose.

- **Configuration and data**
  - `config.py`: central configuration module (EEG, robot, FES, networking, display, etc.).  
  - `environment.yml`: conda environment with scientific, MNE/LSL, Qt, and visualization dependencies.  
  - `Data/`, `telemetry_logs/` (if present): logged experiment data and robot telemetry.

### Standard data layout (under `config.DATA_DIR`)

This repo expects experiment data and logs under the directory configured in `config.py` as `DATA_DIR` (default: `~/Documents/CurrentStudy`).

Typical layout per subject (`sub-<SUBJECT_ID>`):

- `marker_logs/`: logs produced by the marker utility (used for synchronization/debugging).
- `ses-Debug/`: sessions recorded without LabRecorder streams (useful for troubleshooting).
- `training_data/`: one or more `.xdf` recordings used for model/transform generation and training (includes the exact EEG data the models/whitening were derived from).
- `models/`: saved model artifacts and session-specific folders (e.g., adaptive/transform artifacts used at runtime).
- `ses-<SESSION_NAME>/`: per-session data directories (e.g., `ses-S001ONLINE`, `ses-S003ONLINE`, etc.).
  - `eeg/`: the recorded EEG `.xdf` files for the session (when present).
  - `logs/`: per-run protocol logs. Each `logs/<RUN>/` typically contains:
    - `config_snapshot.json`: a frozen subset of config values at runtime.
    - `decoder_output.csv`: decoder probabilities emitted during trials.
    - `event_log.txt`: high-level timeline (including protocol events, robot ACKs/commands, and confusion-matrix summary at the end).
    - `trial_summary.csv`: per-trial outcome summary (hit/miss/ambiguous style metrics).

## 4. Setup / Environment

- **Conda environment (one file, two roles)**  
  - `environment.yml` is a single curated, cross-platform **superset** conda env
    named `lsl`: Python `3.12`, MNE, `pylsl`/`pyserial` (pip), PySide6, the
    scientific Python stack (numpy/scipy/scikit-learn/pyriemann/pandas), xgboost,
    OpenCV, *and* the perception stack. **Every host installs the same deps**, so
    every module imports everywhere. It is a curated spec (not a frozen export) —
    conda-forge-only with no OS-pinned packages — so the same file creates on both
    Linux and native Windows. Create with `conda env create -f environment.yml`
    (or `tools/bootstrap_machine.sh`, default `--role control`).
  - **Roles select setup steps, not deps.** `--role control` (Linux operator
    host) runs the full setup. `--role server` (GPU perception host, Linux or
    Windows) skips the control-only steps. There is no torch knob: the default
    PyPI torch wheel is a CUDA build that also runs on CPU, so a plain create
    works on both GPU and CPU hosts (a CPU-only box can optionally install the
    `.../whl/cpu` wheel to skip the CUDA download). Only the numerical/decoder
    core is version-pinned, plus two justified guards — `pandas<3` and
    `pylsl==1.16.2` (+ conda `liblsl`) for realtime API stability; everything
    else floats to the per-OS solve.

- **Python version**  
  - `environment.yml` pins `python=3.12.x`. New development should target this version (or a compatible minor release) unless the environment is updated.

- **External dependencies (non-exhaustive)**  
  - **EEG / LSL tools**:
    - **eegoSports** (Linux build; used instead of antNeuro software) streaming EEG over **LSL**.
    - **LabRecorder** to record all related LSL streams (EEG, markers, and any other LSL streams used in your setup) into `.xdf`.
    - **mne-lsl** / `mne-lsl viewer` for inspecting/debugging available LSL streams — optional, not bundled in `environment.yml` (the runtime uses `pylsl` directly); install separately with `pip install mne-lsl` if you want the viewer.
  - **Gaze / video**:
    - Pupil Labs / Neon APIs (e.g., `pupil_labs.realtime_api`) if using Neon-based gaze tracking.  
  - **Perception host (GPU)**:
    - The perception stack (`perception/` via `vlm_service.py`) runs on a CUDA host;
      a `GOOGLE_API_KEY` (in `config_local.py`) is needed for the Gemini VLM path.
  - **Robot control**:
    - Network connectivity to the robot controller at the IP/port specified in `config.UDP_ROBOT`.  
    - SSH access for helper commands in `control_panel.py` (e.g., `sshpass`, `gnome-terminal`).  
  - **FES / STM / Arduino**:
    - Rehamove/STM hardware and drivers compatible with the `STM_interface/` code.  
    - An Arduino-compatible actuator (if `USE_ARDUINO` is enabled) and a working serial device at `ARDUINO_PORT`.

- **Hardware setup**  
  - Correct setup of EEG hardware, robot, FES/STM, gaze camera, and any Arduino device is required for full functionality.  
  - Exact wiring and safety procedures are **hardware- and lab-specific** and are not defined by this repository; follow your institution’s protocols.

## 5. Running the System

> **Safety note:** Many scripts can command real hardware. Always verify `config.py` (especially `SIMULATION_MODE`) and hardware status before running experiments.

- **Primary entry point: Control Panel**
  - Activate the conda environment defined by `environment.yml`.  
  - From the repository root:
    - `python control_panel.py`
  - Use the GUI to:
    - Start marker stream, FES listener, and experiment driver.  
    - Launch Harmony calibration (`harmony_calibration_exec.py`) and online control (`harmony_online_control.py`).  
    - Start gaze runner/service and external tools like LabRecorder and EEG acquisition software.

- **Online experiment**
  - Typical flow (if applicable to your setup):
    - Configure subject and toggles in the control panel.  
    - Ensure marker stream and FES listener (if FES enabled) are running.  
    - Select `ExperimentDriver_Online` and appropriate mode (e.g., MI_Bimanual, Gaze_Tracking, Simulation).  
    - Start the experiment driver from the control panel.

- **Offline experiment**
  - Use `ExperimentDriver_Offline.py` or related offline scripts to replay and analyze previously recorded data.  
  - These scripts usually do not control hardware but may still rely on `config.py` for data paths.

- **Calibration workflows**
  - **Harmony calibration (`harmony_calibration_exec.py`)**:
    - Calibrates the gaze tracking device into the robot coordinate space: it estimates the mapping from **gaze coordinates** to **robot end-effector pose** and **joint angles**.
    - Generates or updates pose/gaze calibration libraries (`*.npz`) used by `harmony_online_control.py` for gaze -> robot motion.
    - Launchable from the control panel’s Harmony tab.  
  - **Gaze calibration / services**:
    - `gaze_runner.py` provides UI and service modes for gaze input and telemetry.  
    - Relevant parameters (e.g., UDP host/port) are configured in the gaze scripts and/or `config.py`.

### Transform / model generation (offline)

- **`Generate_Riemannian_adaptive.py`**
  - Generates the adaptive transforms used by the online decoder pipeline.
  - Expects `.xdf` recordings created by **LabRecorder** and streaming EEG from **eegoSports**.
  - Requires **both**:
    - the EEG stream, and
    - the **marker stream**,
    so it can separate/align EEG segments by experimental events.
  - Processes each EEG file in `training_data/` separately under the subject folder in `DATA_DIR`, and includes whitening/transform steps derived from those recorded datasets.

### Experiment modes & hardware dependencies (quick reference)

The drivers in this repo are “modular” in what they require:

- **Online/offline MI** (`ExperimentDriver_Online.py`, `ExperimentDriver_Offline.py`)
  - Can use `SIMULATION_MODE` (so you can run without commanding the robot).
- **Bimanual** (`ExperimentDriver_Bimanual.py`)
  - Requires the **Harmony robot** (robot access is part of the intended workflow).
- **Glove** (`ExperimentDriver_Online_Glove.py`)
  - Requires the **Arduino** (for the current glove driver interface; the rest of the system may not use Arduino).
- **Gaze tracking** (`ExperimentDriver_Online_GazeTracking.py`)
  - Requires both the **pupil/Neon device** and the **robot** (gaze -> motion requires the Harmony calibration mapping).
- **FES**
  - Design intention is an **end-to-end optional toggle** controlled by `FES_toggle` (see `config.py`). It is treated as optional like the robot in simulation workflows; when disabled, the system should not command stimulation.

## 6. Configuration

- **Central role of `config.py`**
  - Defines:
    - EEG settings (sampling rate, channels, filters).  
    - Experiment parameters (trial counts, durations, thresholds, adaptive recentering).  
    - Gaze and pose library settings.  
    - Robot, FES, and marker UDP endpoints (`UDP_ROBOT`, `UDP_FES`, `UDP_MARKER`, `UDP_CONTROL_BIND`).  
    - Robot opcodes (`ROBOT_OPCODES`) and software trigger codes (`TRIGGERS`).  
    - Arduino configuration and **`SIMULATION_MODE`**.

- **Control panel dynamic edits**
  - `control_panel.py` programmatically updates key fields in `config.py`:
    - `TRAINING_SUBJECT` (subject ID).  
    - `FES_toggle` (whether FES is expected to run).  
    - `SIMULATION_MODE` (whether robot commands should be suppressed).

- **Important toggles**
  - **`SIMULATION_MODE`**:
    - When `True`, networking utilities suppress robot UDP commands and ACK waits while allowing marker/FES messages.  
    - Use this for dry runs and development without moving hardware.
  - **`FES_toggle`**:
    - When enabled, some experiment flows will send FES-related commands to `UDP_FES`.  
    - Ensure the FES hardware and safety protocols are correctly configured before toggling on.
  - **Subject IDs (`TRAINING_SUBJECT`)**:
    - Controls which models and training data are loaded (e.g., from `Data/` or subject-specific folders).

- **Networking and hardware impact**
  - Changes in `config.py` for IP/ports, opcodes, or trigger codes directly affect:
    - Robot control behavior and safety gating in `Utils/networking.py`.  
    - Marker streams used for synchronization.  
    - FES/STM and Arduino communication endpoints.

- **Config is the single source of truth**
  - Every knob declared in `config.py` must be load-bearing. Runtime code is expected to read the value, not treat it as a decorative default that a downstream artifact silently overrides.
  - When a trained-model bundle (or similar artifact) ships the same parameter the runtime also reads from config (e.g., epoch window, channel list), the runtime asserts **equality** on load and raises on mismatch — no silent fallback from config to bundle. Deploying a bundle trained at different values requires updating config to match.
  - This is how the ErrP decoder is wired (`Utils/runtime_common.py` checks `bundle["feature_spec"]` against `config.ERRP_EPOCH_TMIN/TMAX`); apply the same pattern when adding new config-controlled pipelines.

## 7. Real-Time / Hardware Considerations (IMPORTANT)

- Parts of this system:
  - **Directly control real hardware**:
    - Robot (via UDP and SSH), FES/STM, and Arduino actuators.  
  - **Rely on real-time loops and timing**:
    - Online EEG decoding, gaze processing, and robot control loops (e.g., `ExperimentDriver_Online.py`, `harmony_online_control.py`, `gaze_runner.py`).

- When developing or modifying:
  - Prefer using `SIMULATION_MODE=True` and/or running without FES enabled until behavior is well understood.  
  - Avoid introducing blocking operations, heavy computation, or extra I/O inside real-time loops.  
  - Be cautious when changing networking, ACK logic, or trigger mappings, as these can affect safety and synchronization.  

## 8. Development Notes

- **Changelog and project log**  
  - **`CHANGELOG.md`**: notable integration changes, plus operational tables (config shadowing, hardware checklist, obsolete-script tombstones). Update **[Unreleased]** when behavior visible to operators or analysts changes.  
  - **Cursor agents:** the rule **`.cursor/rules/finalize-documentation.md`** instructs the agent to refresh `CHANGELOG.md` / `README.md` when you ask to **commit**, **push**, or **finalize** work (unless you say to skip docs). This is not a built-in Cursor push hook; it is project convention for agent-driven sessions.  
  - **Optional Git hooks:** **reminders only** (not auto-edits) when core paths change without staging the changelog:
    - `git config core.hooksPath .githooks`  
    - See `.githooks/README.md` for details.

- **Reuse utilities in `Utils/`**
  - Before adding new helpers, check for existing functions in `Utils/`, `Utils/gaze/`, `STM_interface/`, and `perception/`.  
  - Extend or wrap existing utilities when possible instead of duplicating logic.

- **Entry-point heavy design**
  - Many root-level scripts are **entry points** rather than importable libraries.  
  - When adding new functionality, consider:
    - Whether it belongs in a shared utility first.  
    - Keeping new scripts small and focused, delegating logic to `Utils/` where possible.

- **Minimal, surgical changes**
  - Keep diffs as small and targeted as possible.  
  - Avoid broad refactors, renames, or code style changes unless explicitly needed.  
  - For high-risk areas (online loops, networking, FES/robot control, adaptive transforms), treat any change as safety-critical and review call paths carefully.

- **Documentation architecture for plan-driven features**
  - Features with substantial design work (e.g., the ErrP cross-subject decoder under `Documents/SoftwareDocs/`) maintain three separate documents with distinct temporal stances:
    - **Plan** (`*_Plan.md`) — *living*, forward-looking. Current intent and rationale (the "why"). When the approach changes, edit the plan in place so it reads as if the current design was always intended. Do not accumulate dated amendment blocks — absorb them.
    - **Report** (`*_Report.md`) — *historical*, chronological. What was actually done, with numeric evidence (AUC tables, commit hashes, pivots). This is the proof-of-plan document.
    - **Reference** (`*_Reference.md`) — *evergreen*, code-anatomy. How the code is organised right now (the "how"). No phase references, no dates.
  - Numbers live in the Report, not the Plan or Reference. Bug fixes done after the plan completes are just bug fixes — do not write them up as new phases unless they contradict something previously documented.

## 9. Status / Notes

- Some components, especially certain analysis/visualization scripts, may be **experimental or legacy** and not required for a minimal online BCI + robot pipeline.  
- Hardware availability and local lab practices vary; some scripts may be used only in specific setups or studies.  
- Before modifying behavior:
  - Trace how a script is launched (e.g., from `control_panel.py` or other drivers).  
  - Identify which utilities and configuration fields it depends on (`config.py`, `Utils/networking.py`, `Utils/runtime_common.py`, etc.).  
  - Consider the real-time and hardware implications of your changes, and always test in simulation or low-risk conditions before using with participants or full hardware.

