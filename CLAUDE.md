# CLAUDE.md — Collaboration Rules for BCI Repository

This file defines strict rules for modifying a real-time BCI + robotics
codebase. Violations may introduce silent numerical errors or unsafe
runtime behavior.

---

## Platform Policy

- **Linux is the primary platform.** All code must run correctly on Linux
  (CPU only) as the baseline. Windows (CUDA, RTX 4070 Ti) is a secondary
  development machine.

- **Realtime/online applications are Linux only.** Any file that drives
  closed-loop BCI feedback, robot control, or LSL streaming is
  Linux-exclusive. Do not add Windows-specific code to these files.

- **Training and analysis scripts run on both.** Platform-specific
  optimisations (e.g. GPU compilation, sleep inhibition) must be gated:
  `if sys.platform == "win32"` or `if torch.cuda.is_available()`. They
  must degrade gracefully to CPU/Linux behaviour when the condition is
  not met.

- **Perception services are Windows-hosted in production.** All ML
  inference (FastSAM, Depth Pro, Gemini-backed VLM, YOLO + SORT) runs on
  the Windows GPU host. Linux owns device I/O (Pupil Labs Neon, EEG, FES,
  robot) and ships frames to Windows via a TCP relay; Windows ships
  results back over UDP. The toggle is the `PERCEPTION_FRAME_SOURCE`
  config key (`local` opens Neon directly, `remote` consumes from
  `Utils/frame_relay.py`); default is `local` so single-machine workflows
  keep working without config edits. See
  `Documents/SoftwareDocs/GPU_Service_Host_Architecture_Plan.md` for the
  full architecture, wire format, and deployment notes.

- **Perception source lives in-tree under `perception/`.** The FastSAM /
  Depth Pro / Gemini-reasoner / Neon-reader code was folded from the
  `harmony_vlm` repo into the `perception/` package (WS3, 2026-06-15),
  vendored with attribution headers; edit it here, not upstream. It runs
  in the single unified conda env (`environment.yml` — the separate
  `harmony_vlm` env is retired), still as a separate process behind the
  UDP/GPU-host split above. Model weights resolve from the machine-local
  `PERCEPTION_MODELS_DIR` and the Gemini key from `GOOGLE_API_KEY` in
  `config_local.py`. `perception/`'s live modules (object_detector,
  depth_estimator, fixation_detector, intent_reasoner, pupil_reader,
  visualize_neon, neon/) are import-safe; the rest are **staged** for
  WS4/WS5 and not importable until their deps land — see
  `perception/__init__.py`.

---

## Source of Truth

- The canonical implementation of the EEG processing pipeline is:
  `Generate_Riemannian_adaptive.py`.

- All derived methods (e.g., tangent space, XGBoost, RBNNet variants)
  must:
  - Import and reuse shared logic from the base implementation
  - Not duplicate preprocessing steps (filtering, covariance, whitening)

- Any change to core preprocessing must be made in the base file first.

- When two code paths process the same data type (e.g. mu-band and
  beta-band covariances), verify they follow the same pipeline steps
  before assuming consistency. Read both paths — do not infer.

---

## High-Risk Files

Changes to these files require extra caution. Read the file in full
before proposing any change. State what was read. Prefer no change over
an uncertain change.

### Tier 1 — Direct Hardware I/O
Physical consequences if misused (incorrect stimulation, robot motion,
lost EEG signal). Treat as safety-critical.

| File | Hardware |
|---|---|
| `Utils/networking.py` | All UDP comms — robot, FES, marker stream. Self-described as performance- and safety-critical. Wire protocol between all experiment drivers and hardware. |
| `FES_listener.py` | FES trigger listener |
| `STMsetup.py` | FES/STM device setup and serial initialisation |
| `STM_interface/` (entire directory) | Rehamove FES library, serial communication, device config |
| `Utils/EEGStreamState.py` | EEG LSL stream state management |
| `Utils/stream_utils.py` | EEG stream loading and parsing |
| `UTIL_marker_stream.py` | LSL marker stream output |
| `Utils/gaze/gaze_tracking.py` | Gaze device data acquisition |
| `Utils/gaze/gaze_system.py` | Gaze system coordination |

### Tier 2 — Realtime Orchestration
Timing-critical. These files call into Tier 1 and run in closed-loop
experiment contexts. Changes must preserve timing guarantees.

| File | Role |
|---|---|
| `ExperimentDriver_Online.py` | Main online BCI experiment loop |
| `ExperimentDriver_Bimanual.py` | Bimanual robot experiment loop |
| `ExperimentDriver_Online_GazeTracking.py` | Gaze-controlled experiment loop |
| `ExperimentDriver_Online_Glove.py` | Glove + Arduino experiment loop |
| `harmony_online_control.py` | Harmony robot online control |
| `harmony_calibration_exec.py` | Harmony robot calibration |
| `Utils/runtime_common.py` | Shared realtime decoder dispatch logic |

### Tier 3 — Standalone / Supporting
Lower direct risk. Used for testing, visualisation, or supporting
realtime but not in the critical path.

| File | Role |
|---|---|
| `UDPRobot.py` | Standalone manual console for robot testing only. Not imported elsewhere. |
| `udp_send.py` | Utility UDP sender |
| `Utils/gaze/gaze_math.py` | Gaze geometry calculations |
| `Utils/gaze/gaze_render.py` | Gaze visualisation rendering |
| `Utils/gaze/gaze_ui.py` | Gaze UI components |
| `gaze_runner.py` | Gaze experiment runner |

---

## Realtime Safety Constraints

- Realtime loops must:
  - Avoid blocking operations (no file I/O, no heavy allocations inside
    the loop)
  - Avoid dynamic memory growth
  - Maintain deterministic timing where possible

- Any change to EEG streaming, robot control, or FES triggering must
  preserve timing guarantees and be tested in a realtime context.

- If unsure, prefer no change over a potentially unsafe change.

---

## config.py and config_local.py

- `config.py` is **global and committed**. It holds algorithm /
  decoder / EEG / protocol settings, plus *safe defaults* for every
  machine-local key (loopback IPs, empty paths, `False` flags). Changes
  here are committed from either machine.

- `config_local.py` is **per-machine and gitignored**. It supplies real
  values for paths and network endpoints on this host. `config.py`
  imports it via `from config_local import *` at the bottom of the file,
  so anything defined in `config_local.py` shadows the default.

- A new machine bootstraps with:
  ```bash
  cp config_local.example.py config_local.py
  # edit config_local.py
  ```

- The following keys are machine-local and must only be assigned in
  `config_local.py`. A pre-commit hook
  (`~/.claude/hooks/config-py-guard.sh`) blocks any non-default value
  for these keys appearing in a staged `config.py` diff:
  - `WORKING_DIR`, `DATA_DIR`
  - `GAZE_UDP_IP`, `GAZE_BIND_HOST`
  - `NEON_COMPANION_HOST`
  - `PERCEPTION_FRAME_SOURCE`, `SERVICES_HOSTED_REMOTELY`
  - `FRAME_RELAY_HOST`, `FRAME_RELAY_DIAL_HOST`
  - `PERCEPTION_MODELS_DIR`, `GOOGLE_API_KEY`
  - `VLM_SERVICE_HOST`, `VLM_BIND_HOST`
  - `ARDUINO_PORT`

- All other config changes are safe to commit provided they are
  platform-neutral.

---

## Reference Literature and Documentation

Two directories serve as the canonical locations for papers and
implementation references related to this project. On Linux the paths
are absolute as shown; on Windows replace `/home/arman-admin` with the
appropriate Windows home prefix — the subdirectory structure is the
same on both machines.

### `/home/arman-admin/Documents/studies/`
Contains academic papers providing context for methods implemented or
planned in this repo. Papers here are reference material — read them
for background and motivation, but do not assume the implementation
follows them exactly. Divergences from the paper are intentional and
documented in the relevant implementation reference.

### `/home/arman-admin/Documents/SoftwareDocs/`
Contains implementation references, planning documents, and post-
implementation technical records for features developed in this repo.
These are the primary written records for features that may not have
a separate planning document in the codebase. When working on a
feature, check here first for an existing reference document before
reading the code.

When a feature is derived from a paper in `Documents/studies/`, note
the reference in the commit message and relevant docstrings.

---

## Code Analysis Protocol

- **Read before you claim.** Before asserting what a function or code
  path does, read it with the Read tool. Do not infer behaviour from
  naming conventions, paper formulas, or context.

- **Read both sides before comparing.** When analysing a discrepancy
  between two code paths, read both in full before drawing any
  conclusion. Never compare a read path against an assumed path.

- **Cite sources.** Any code referenced in an analysis must include the
  exact file path and line number (e.g.
  `Generate_Riemannian_adaptive.py:304`). If a snippet is not directly
  quoted from the repo, label it explicitly as pseudocode or assumption.

- **No fabricated code.** Never present inferred or reconstructed code
  as if it were observed in the repo. If the actual implementation is
  unknown, say so and read the file before continuing.

---

## Proposing Changes

- Do not propose a fix for a discrepancy until it has been confirmed by
  reading the relevant source files.

- State what was read and where before proposing any change.

- Do not add features, refactor, or clean up code beyond what was asked.
  A bug fix does not justify touching surrounding code.

- Do not add error handling or validation for scenarios that cannot
  happen given existing system guarantees.

---

## Error Handling

- **Surface real errors.** Do not silently suppress exceptions. If an
  exception is caught, it must be re-raised, logged with enough context
  to diagnose the failure, or suppressed with an explicit inline comment
  explaining why suppression is safe in that specific case.

- **`try/except` is acceptable when architecturally justified** — resource
  cleanup, protocol-level recovery, or documented degradation paths (e.g.
  GPU → CPU fallback). It is not a substitute for understanding a failure
  mode, and it is not a way to make code "more robust" by default.

- **Prefer fail-fast in realtime loops and Tier 1 hardware files.**
  Silent recovery from an unexpected error can mask unsafe state. Unless
  a specific recovery action is defined and safe, let the exception
  propagate and crash the process.

---

## Software Development Practices

- **Single responsibility.** Functions and modules should do one thing.
  If a helper is only used once, inline it rather than abstracting it.

- **No speculative abstractions.** Implement what the task requires.
  Do not design for hypothetical future requirements.

- **Prefer editing existing files.** Do not create new files unless
  genuinely necessary.

- **No dead code.** Do not leave commented-out code, unused imports, or
  removed-feature stubs in committed files.

- **Comments explain why, not what.** Inline comments should document
  intent and non-obvious reasoning, not restate what the code does. Be
  substantive enough to serve as documentation, but not padded — one
  clear sentence beats three vague ones.

- **Keep documentation current.** When modifying a function, update its
  docstring to reflect the change. When adding a function, add a
  docstring. When a change affects behaviour described in `README.md` or
  another documentation file, update that file too. This applies only to
  code and documentation directly touched by the task.

- **Do not add comments to unrelated code.** Do not add or update
  docstrings, inline comments, or documentation for code that is not
  part of the current task — unless the task is explicitly a
  documentation update.

- **Numerical code warrants extra care.** This repo operates on SPD
  matrices where floating point drift compounds over training. Prefer
  full float32 precision. Justify any precision tradeoff explicitly
  before introducing it.

---

## Commit Hygiene

- Commit proactively at clean logical boundaries: one coherent diff
  (feature, bug fix, or refactor step) with no half-finished work.
- Do not bundle unrelated changes into one commit — split them.
- After every commit, push to origin immediately so remote stays
  current. Exception: never auto-push `main` / `master` — confirm with
  the user first. (A PostToolUse hook handles the push automatically on
  feature branches.)
- Stage files individually — never `git add .` or `git add -A`. (A
  PreToolUse hook blocks the wildcard forms.)
- Do not commit machine-local keys (paths, network IPs, USB device
  paths) into `config.py`; they belong in `config_local.py`. (A
  PreToolUse hook blocks `git commit` if a staged `config.py` diff
  assigns one of these keys to a non-default value. The full list of
  guarded keys is in the `config.py and config_local.py` section
  above.)
- Commit messages should explain *why*, not just *what*.
- All commits must be compatible with Linux (CPU) regardless of which
  machine they were developed on.
- Do not add `Co-Authored-By` or any AI attribution lines to commit
  messages.

---

## External System Dependencies

These are standalone applications that are not managed by conda. They must
be installed and running before any realtime session can function. They will
not appear in `environment.yml`.

| Application | Role | Notes |
|---|---|---|
| **eegoSports** (ANT Neuro, Linux) | Publishes EEG data as an LSL stream | Without it, `Utils/EEGStreamState.py` has no stream to connect to and realtime drivers will fail at startup |
| **LabRecorder** | Records all active LSL streams to `.xdf` | All training `.xdf` files were produced by this tool; required before and during data collection sessions |

---

## Dependency and Environment

Two install **roles**, selected at setup time (see
`tools/bootstrap_machine.sh --role`):

- **`control`** — the Linux operator host. The **superset** env: device I/O,
  EEG/LSL decoder pipeline, Qt control panel, gaze, *and* a CPU build of the
  perception stack, so a single-box dev machine
  (`PERCEPTION_FRAME_SOURCE=local`, the default) runs everything from one env.
  - Env: `environment.yml`, conda env named `lsl`, Python 3.12.
  - This is the reference Linux env. Do not introduce dependencies that
    cannot be satisfied on Linux. Windows-only packages (e.g.
    `triton-windows`) must never appear in `environment.yml` or
    `requirements.txt`.
  - **Control is Linux-only** (realtime/online is Linux-only).

- **`server`** — the GPU perception host (Linux or Windows). Runs the
  perception stack only (`vlm_service.py` + live `perception/` modules); no
  device I/O, decoder, or Qt. The perception-only subset of the control env
  plus a **CUDA** torch build. Env name `harmony-server`, single
  cross-platform file `environment.server.yml`: it is a curated spec with no
  OS-specific conda packages, so conda solves the core per-platform and pip
  selects the right torch/opencv wheel automatically — one file works on both
  Linux and Windows. The CUDA wheel tag (`cu124`) is the one knob; split into
  per-OS files only if two hosts ever need *different* tags. Keep the conda
  core versions in sync with `environment.yml` so a single-box dev machine
  stays consistent.

The perception source was folded in-tree in WS3 — there is **no** sibling
`harmony_vlm` repo or env to clone/create (`VLM_REPO_DIR` / `VLM_CONDA_ENV`
are retired).
