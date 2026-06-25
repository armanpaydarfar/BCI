# Harmony test suite

Hardware-free pytest suite that runs on the Linux primary BCI machine
(and on Windows when the lsl conda env is available). No Neon, no
eegoSports, no Rehamove, no robot, no network peers — every test uses
synthetic fixtures or loopback sockets, so the whole thing runs on a
laptop with the `lsl` conda env and nothing else.

As of this writing: **~400 test functions across 48 files** (the fast
subset reports ~436 passed, 456 incl. the slow-marked set; the difference
between functions and passes is parametrised cases). The
detailed design rationale and per-test plan live in the team's internal
docs (`SoftwareDocs/projects/harmony-bci/test-suite/{plan,report}.md`)
for maintainers who have access; this README is the self-contained
in-repo reference.

## Running

The default fast subset (excludes anything marked `@pytest.mark.slow`):

```bash
pytest tests/ -m "not slow" -q
```

The full suite (includes loopback / network-adjacent tests in
`test_relay_loopback.py` etc.):

```bash
pytest tests/ -q
```

A single file:

```bash
pytest tests/test_networking_protocol.py -v
```

## Pre-commit gate

The `.githooks/pre-commit` hook (enabled per clone with
`git config core.hooksPath .githooks`, see `.githooks/README.md`)
runs the fast subset before every commit and aborts the commit on
failure. The pytest invocation lives in `tools/pre-commit-pytest.sh`
so it can be run standalone too.

To bypass the gate in an emergency (NEVER on `main` / `master`):

```bash
git commit --no-verify
```

Acceptable bypass reasons (rare): an in-progress feature branch where
the failing test is exactly what the next commit is about to fix.
Unacceptable: pushing onto a shared branch with red tests.

## Adding a new test

Verification protocol — *before* writing any test against code in this
repo (this mirrors the repo's `CLAUDE.md` "read before you claim" rule):

1. Read the target file:line range with the `Read` tool (or open it).
   Confirm the function name, signature, and behavior are as described —
   don't write a test against an assumed API.
2. Cite the file:line you are pinning in the docstring of every test
   function, so the next reviewer can audit the test against the source.
3. Prefer pure, synthetic fixtures over real I/O. If the behavior needs
   a socket, use loopback (`127.0.0.1:0`) with a short timeout.
4. New file goes under `tests/` with a name beginning `test_`. Pytest
   discovers it automatically (see `pytest.ini`).

If your test is slow (real-socket loopback with sleeps, large fixture
load, > ~1 s) mark it `@pytest.mark.slow` so the pre-commit fast subset
stays under ~10 seconds. Run slow tests explicitly with
`pytest tests/ -m slow` or the whole suite with `pytest tests/`.

## Fixtures

`tests/conftest.py` provides:

- Headless pygame: sets `SDL_VIDEODRIVER=dummy` before any pygame
  import. Required because several modules call `pygame.display.set_mode`
  at import time.
- `sys.path` includes the repo root and the vendored Rehamove
  library, so `import Utils.foo` and `from rehamove import *` work
  without per-file path dancing.
- `config.SIMULATION_MODE = True` is set at session start so any
  module that snapshots it at import reads the simulation value.
- The `sim_mode_networking` pytest fixture monkeypatches the
  `Utils.networking.SIMULATION_MODE` snapshot (which is taken at
  import time per `Utils/networking.py:64-67`) for the duration of
  a single test.

## What's covered, and why the suite is shaped this way

The suite is deliberately **dense on cheap, deterministic pure-function
and wire-protocol code** (where a silent sign/transform/format bug would
mis-target the robot or corrupt the EEG decode) and **thin where testing
is expensive** (Qt GUI, god-class service main loops). That is why the
counts cluster the way they do — most of the mass is the newest,
safety-critical gaze→robot geometry.

| Subsystem | Files | What the tests guard |
|---|---|---|
| **AprilTag calibration** | `test_apriltag_calib`, `_calibrate`, `_world`, `_sweep`, `_control` | Umeyama solve, ray–plane intersection, transform composition, multi-tag world map / occlusion, sweep gating, gaze→plane control chain. A clean math → offline-solve → online → control stack (low overlap). |
| **Gaze calibration / mapping** | `test_gaze_calibration_mapping`, `_v1v2_dispatch`, `test_pose_library_loader_v1v2`, `test_gaze_affine_fit` | v2/v3 nearest-neighbour mapping, Mahalanobis scaling, workspace clamp, v1/v2 back-compat loaders, vergence affine fit. The v1/v2 dispatch duplication is **intentional** (fails CI if the driver unwires v2). |
| **Gaze geometry / tracking** | `test_gaze_tracking_pure`, `test_coverage_grid`, `_view`, `test_harmony_link` | IoU / SORT tracking, coverage-sufficiency rule, view-plane projection, robot joint-command + telemetry strings. |
| **VLM / perception service** | `test_seg_constraints`, `test_ws4_wiring`, `test_recognize`, `test_decide_waypoint_pairing`, `test_seg_tracking`, `test_vlm_subscriber_guard`, `test_vlm_results_push`, `test_vlm_thinking_budget` | Segmentation constraint filter, F1/F4/F5 command wiring, label-based waypoint pairing, anti-flicker temporal tracking, results-push schema + subscriber ordering/TTL. |
| **Networking (UDP)** | `test_networking_protocol`, `_sockets` | Wire format, ACK base-token matching, socket routing, `SIMULATION_MODE` suppression. The transport between every driver and the hardware — pinned thoroughly on purpose. |
| **EEG decoder core** | `test_eeg_stream_state`, `test_runtime_common_features`, `test_stream_utils_xdf` | Covariance / shrinkage / tangent-space algebra, stream-state accumulation, XDF marker selection. SPD-matrix drift is the failure mode `CLAUDE.md` warns about. |
| **Frame relay** | `test_relay_loopback` (slow), `test_relay_local_subscriber`, `test_frame_relay_envelope`, `test_decode_contention_guard` | TCP round-trip + reconnect/drop-oldest (loopback), envelope sad-paths, the Neon decode-tearing regression guard. |
| **Markers / ErrP** | `test_marker_stream_parsing`, `test_errp_driver_state_machine` | LSL marker wire format, ErrP driver marker ordering (AST-level). |
| **FES safety** | `test_fes_listener_params` | Stimulation-current parameter selection (safety-critical path). |
| **Contracts / smoke** | `test_imports_smoke`, `test_config_contract`, `test_xgb_config_lock`, `test_overlay_no_accumulation`, `test_neon_lsl_bridge`, `test_experiment_driver_v2_depth` | Every file compiles + imports without hardware; config keys / hyperparameters don't silently drift; misc behaviour guards. |

**Known-thin areas** (where coverage is expensive and intentionally
sparse for now): the Qt control panel (`control_panel.py`, GUI), the
service main loops (`vlm_service._serve_forever`, the online driver
`main()`), and the perception UDP clients. These are the natural targets
for characterization tests when those modules are refactored.

## Cross-platform parity

Cross-platform (Linux ↔ Windows) decoder equivalence — Plan §6.1 —
is **deferred** until a Windows CI runner exists. Both machines run
the same suite manually for now.
