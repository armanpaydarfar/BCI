---
description: Core Harmony project rules for safe, minimal, and hardware-aware changes
alwaysApply: true
---

# Harmony Project Rules

## Global behavior

- **Prefer minimal, targeted diffs**
  - Change only what is necessary to achieve the requested behavior.
  - Avoid opportunistic refactors, renames, or reformatting that are unrelated to the task.
  - Preserve existing patterns and style unless the user explicitly asks to change them.

- **Reuse existing utilities and patterns**
  - Before adding new helpers, check `Utils/`, `OBS/`, and `STM_interface/` for existing functionality to extend or wrap.
  - Prefer updating or composing existing utilities over duplicating logic.

- **No broad structural refactors by default**
  - Do not reorganize modules, move files, or change public APIs unless explicitly requested.
  - Keep behavior backward compatible unless the task explicitly calls for a breaking change.

## Risk levels by area

### High‑risk (real‑time, networking, hardware)

Treat these files and modules as **high‑risk**; make the smallest possible changes, and avoid altering control flow or timing unless absolutely required:

- **Real‑time EEG / BCI runtime**
  - `ExperimentDriver_Online.py` and other `ExperimentDriver_*.py` drivers
  - `Utils/runtime_common.py`
  - `Utils/EEGStreamState.py`
  - `Utils/experiment_utils.py` functions used in online loops
  - `Generate_Riemannian_adaptive.py`

- **Networking and robot communication**
  - `Utils/networking.py`
  - `Robot_Control_WBC.py`
  - `UDPRobot.py`
  - `udp_send.py`
  - `UTIL_marker_stream.py`
  - Any code using `config.UDP_*`, `ROBOT_OPCODES`, or `TRIGGERS`

- **FES / STM / stimulator interfaces**
  - `FES_listener.py`
  - `STMsetup.py`
  - `STM_interface/**` (including `serialCommunication.py` and Rehamove libraries)
  - Any modules that send commands to FES, STM, or other medical/actuator hardware

- **Gaze and real‑time streaming**
  - `gaze_runner.py`, `gaze_visualizer.py`
  - `OBS/**` real‑time streaming/overlay scripts
  - Any modules consuming LSL streams for EEG or gaze and driving robot or FES

For these areas:

- Avoid adding blocking calls, heavy computations, or extra I/O inside tight loops.
- Preserve existing timing, loop structure, and safety checks.
- When changing protocol details (messages, opcodes, triggers), clearly call out the impact in the summary.

### Low‑risk (documentation, visualization, offline analysis)

These areas are generally **low‑risk** and can tolerate more refactoring, documentation, and cosmetic improvements:

- Markdown / text docs and comments.
- Offline analysis and training scripts:
  - `Analyze_*.py`
  - `Generate_Decoder.py`, `Generate_Riemannian*.py`
  - `Visualize_offline_data*.py`
- Standalone plotting, report generation, and Jupyter notebooks (if present).
- Pure visualization helpers that do not affect online control logic.

Even in low‑risk areas, keep diffs focused on the task and avoid unnecessary churn.

## Required workflow

- **Inspect before editing**
  - Always read the relevant files and surrounding context before making any changes.
  - Identify how the code is used (call sites, shared utilities, configuration) to avoid breaking dependencies.

- **Propose a plan first**
  - Before applying edits for non‑trivial tasks, outline a brief plan to the user:
    - What you intend to change.
    - Which files/functions will be touched.
    - Any expected behavioral or interface changes.
  - Adjust the plan only if new information arises; keep the implementation aligned with the agreed approach.

## Change summaries and testing

- **Always summarize changes**
  - After edits, provide a concise summary focusing on behavior, not just files changed.
  - Explicitly note any impact on real‑time behavior, networking, or hardware‑facing code.

- **Always suggest tests or checks**
  - Propose concrete tests the user can run (scripts, modes in `control_panel.py`, or manual procedures).
  - For high‑risk areas, include:
    - Dry‑run or simulation‑mode steps where available.
    - Safety checks for robot/FES/STM (e.g., ensure emergency stops and low‑intensity settings during first runs).
  - When touching analysis or visualization, suggest sample commands or datasets to validate the output.

