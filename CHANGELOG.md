# Changelog

Notable changes and project history are recorded here. This file replaces the older `CONFIG_AUDIT.md` name: it still contains **runtime and configuration notes** (below the dated log), not only `config.py`.

The format is **lightweight** (this repo is not strictly semver-tagged). Add bullet points under **[Unreleased]** when you merge user-visible behavior changes.

## [Unreleased]

### Behaviour

- **Recorder + runtime: vlm-only operation via frame_relay.** `harmony_free_arm_calibration.py` no longer consumes from `gaze_runner.py`'s UDP snapshot. It dials `config.FRAME_RELAY_DIAL_HOST:FRAME_RELAY_PORT` as a TCP consumer (sharing the panel's embedded relay or a standalone `python -m Utils.frame_relay`), computes `(gaze_yaw_deg, gaze_pitch_deg)` locally from `gaze_px + camera_matrix` (camera-frame angles), and pulls anchor depth from VLM Depth Pro. Transit rows leave `D_cm = NaN` and are tagged `Depth_source = "pending_interpolation"`; the offline `tools/fit_depth_interpolation.py` post-fills them with a leg-aware bracketed linear interp (each transit row reads its `Leg_label`, finds the from/to anchors, blends by EE-distance ratio; falls back to global KDTree NN when bracketing anchors are unavailable). NPZ `meta["depth_source"]` is now `"vlm_depth_pro"` end-to-end. `harmony_online_control.py`'s `_fetch_v2_snapshot` mirrors the same frame_relay path so query-time features match calibration. `GAZE_CALIBRATION_DEPTH_SOURCE` is `"vlm_depth_pro"`-only in this build; the per-session affine fit and head-pose recenter are deferred and can be re-added when the gaze_system pipeline is reinstated. `tests/test_gaze_calibration_recorder.py`, `tests/test_gaze_depth_interpolation.py::TestFitDepthInterpolation`, and `tests/test_pose_library_loader_v1v2.py::TestHybridMetaRoundTrip` skip pending rewrite.

### Tooling

- **`tools/deploy_robot_gaze.sh` single-command wrapper.** Invokes the C++ repo's `build_and_deploy.sh --push 192.168.2.1 --tool Gaze_Tracking` after checking the cross-repo `deploy.config` and the built `dist/01d91ea/Gaze_Tracking` exist. The underlying script (in `DockerProjects/ubuntu1804_container/HARMONY-UNIT-4/tools/`) now backs up the prior remote binary to `<name>.bak.<UTC-ISO-timestamp>` before rsync and verifies md5sum on both sides post-push; supports `DEPLOY_REMOTE_PASS` for sshpass-based auth against the lab Harmony robot. Rollback is one `mv` over SSH plus the standard `killall.sh && run.sh` re-init.
- **Hardware-free pytest suite + pre-commit gate (Phase 1a).** New `tests/` directory with `test_imports_smoke.py` (compile-all + curated-import + post-refactor importability), `test_networking_protocol.py` (`_to_wire` / `_is_coords_string` / `_base_token` / `_build_ack_map`), `test_config_contract.py` (machine-local key safe-defaults, panel-vs-config key drift, driver `TRIGGERS`/`ROBOT_OPCODES` references), and `test_gaze_tracking_pure.py` (`iou_xyxy`, `size_similarity_ratio`, `gaze_object_hit`, `SimpleSORTTracker`). The four pre-existing loopback scripts under `tools/test_*.py` move into `tests/`. New `pytest.ini` registers `testpaths = tests` and the `slow` marker. New `tools/pre-commit-pytest.sh` runs `pytest tests/ -m "not slow" -q` on every commit, wired in via `.githooks/pre-commit`; bypass with `git commit --no-verify` (never on `main`/`master`). Two prerequisite refactors (`UTIL_marker_stream.py` and `FES_listener.py`) defer import-time I/O into `main()` so the modules import without hardware. Cross-platform parity (Plan §6.1) deferred until a Windows CI runner exists. See `tests/README.md` and `Documents/SoftwareDocs/Harmony_Test_Suite_Plan.md`.
- **`FRAME_RELAY_EMBEDDED` default added to `config.py`** (`True`). Was previously read only via `getattr(_HCFG, ..., True)` in `control_panel.py` and missing from `config.py`; the test suite caught this drift on first run.

### Behaviour

- **Free-arm calibration recorder: 15-waypoint grid + transit telemetry.** `harmony_free_arm_calibration.py`'s mandatory grid is now 3 depths × 5 horizontal positions (rightmost-first sweep R1→R5) for 15 captures, down from the prior 27; a background 20 Hz `TelemetryThread` records `phase="transit"` bundles (tagged with a per-leg `leg_label`) while the arm is free between waypoints and stops before the final `c` so the home transitions are excluded. After the final capture the recorder auto-homes via `h;dur=4.0` without an extra operator prompt. NPZ gains a `Leg_label_all` column; `GazeCalibrationMappingV2` already ignores non-captured phases. See `Documents/SoftwareDocs/Reports/Harmony_Gaze_Calibration_Python_Report.md` (recorder-rework section, 2026-05-19).
- **Gaze calibration v1/v2 dispatch + Pass-1/Pass-2 IMU toggle (Project 2 Python side).** Two new `config.py` keys default-safe to legacy behaviour: `GAZE_CALIBRATION_VERSION = 1` (selects the legacy 2D-NN-on-pixels lookup at `ExperimentDriver_Online_GazeTracking.resolve_robot_target_from_gaze`) and `GAZE_CALIBRATION_USE_IMU = False` (gates Pass-2 head-pose features in the v2 mapping). Flipping `GAZE_CALIBRATION_VERSION` to `2` switches to the depth/IMU-aware Mahalanobis NN over `(gaze_yaw_deg, gaze_pitch_deg, depth_cm)`; the IMU toggle is gated on Pass-2 measurably beating Pass-1 by >= 10% reduction in held-out joint MSE per `Documents/SoftwareDocs/Gaze_Calibration_Sensor_Characterization.md` §5. Both keys are algorithm settings, committable from any machine.
- **Gaze calibration depth-source toggle (`GAZE_CALIBRATION_DEPTH_SOURCE`, default `"vergence"`).** Opt-in `"vlm_depth_pro"` value routes the free-arm recorder and the v2 runtime through `vlm_service`'s Depth Pro endpoint instead of binocular vergence; source is pinned in the NPZ meta and the recorder/driver fail-fast on mismatch. See `Documents/SoftwareDocs/Harmony_Gaze_Calibration_Upgrade_Plan.md` §6.5 for the alignment invariant and runtime cadence.
- **Control panel: seg-stream status poll pauses while a synchronous VLM command is in flight.** The 5 s `_poll_seg_stream_stats` tick in `control_panel.py` now skips its 500 ms `status` probe whenever `_vlm_command_threaded` has a worker outstanding (decide / depth / capture_first / decide_pair). Eliminates the spurious `seg-stream status: unreachable (timed out)` lines that fired whenever the server's single-threaded request loop (`vlm_service.py:403-437`) was blocked on Gemini — even though the seg-stream worker thread (`vlm_service.py:608-611`) and results-push thread (`vlm_service.py:1074-1075`) were healthy throughout. A genuine outage (no command in flight, status still times out) still logs as before.
- **REPL (`harmony_online_control.py`) honours `GAZE_CALIBRATION_DEPTH_SOURCE`.** When a v2 NPZ pins `depth_source="vlm_depth_pro"` the EEG-free vision-mode REPL now (a) verifies `vlm_service` is reachable with `depth_enabled=True` at startup (RuntimeError otherwise) and (b) calls `vlm_client.depth(at_gaze=True)` once per `v`-press, feeding `depth_at_gaze_m * 100` into the Mahalanobis `D_cm` feature. Vergence NPZs keep the existing snapshot-depth path byte-for-byte; mid-loop VLM failures print a diagnostic and skip the robot dispatch without falling back to vergence. See the "REPL VLM depth wiring (2026-05-19)" section in `Documents/SoftwareDocs/Reports/Harmony_Gaze_Calibration_Python_Report.md`.
- **Free-arm recorder: REV01 hybrid depth (Depth Pro at anchors + vergence-affine in transit)** — see `Documents/SoftwareDocs/Harmony_Gaze_Calibration_REV01_Plan.md`. The 20 Hz transit telemetry now feeds the v2 fit, the writer emits a per-row `Depth_source` column plus a hybrid `meta["depth_source"]`, and the runtime dispatches on a new `runtime_depth_pipeline` (vergence_affine | vlm_depth_pro | vergence). Operator runs `tools/fit_vergence_affine.py` between recording and online use; `tools/fit_depth_interpolation.py` is the KDTree NN backup when the affine fit's R² / max-residual thresholds reject. Operator instruction wording flips to "LOOK AT THE END EFFECTOR" (matches the legacy `harmony_calibration_exec.py:248` literal). Default `GAZE_CALIBRATION_VERSION = 1` is unchanged; flipping `config_local.py` to 2 with a REV01 `_affine.npz` lights up the new path.
- **`vlm_service.py` exposes `--max-output-tokens` (default `8192`)** and passes it through to `IntentReasoner(max_tokens=...)`. The upstream `harmony_vlm` default is `1024`, which truncates the JSON response mid-token on scenes with many segments (~18+ detections in practice — each candidate is ~50-100 output tokens) and trips `[VLM] Failed to parse response, using fallback`. `8192` gives ~4-5× headroom for typical scenes and is well under Gemini 2.5 Flash's 65,536 output-token ceiling. Billing is per emitted token, not per cap. Override via `--max-output-tokens` or `VLM_MAX_OUTPUT_TOKENS` in config.
- **`vlm_service.py` CLI defaults now read from `config.py`** (which layers `config_local.py`). On the Windows GPU host, `python vlm_service.py --repo-dir <harmony_vlm>` is now sufficient to bring the service up in the production remote topology — bind host, frame source, frame-relay dial host, port, model, Neon host, and depth toggle all resolve from config instead of hardcoded argparse defaults. Single-machine dev (Linux with `PERCEPTION_FRAME_SOURCE=local`) is unaffected — config defaults still give `127.0.0.1` / `local`. `--enable-depth` switched to `argparse.BooleanOptionalAction` so the config-driven default can be negated with `--no-enable-depth`; `vlm_launcher.py` updated to pass an explicit positive/negative form to match the new semantics.
- **Online ErrP driver pause rate:** new `config.ERRP_ONLINE_P_STOP` (default `0.3`) gates the per-move Bernoulli draw in `ExperimentDriver_ErrP_Online.py`; previously every successful-MI move was paused. Training driver's `ERRP_P_STOP` (default `0.5`) is unchanged and intentionally separate.
- **Perception Pipeline — per-Connect verification state machine.** The Main-tab Send / Compute / Receive LEDs now report distinct sequential checks per Connect press: Send = relay TCP handshake delivered (no longer waits on Pupil Labs SDK first-frame, which can be >10 s); Compute = GPU `cmd=status` reply with `ok=True` (no longer requires `frames_received>0`); Receive = token-matched `chain_verify` round-trip, where the panel generates a fresh `_connect_token` per Connect and the GPU echoes it in the synthetic push payload. Stale GPU-cache pushes from prior sessions can no longer trip Receive on a reconnect. LEDs paint yellow during verification and reset to gray on Disconnect. New audit-trail lines per Connect cycle are tee'd into `<DATA_DIR>/sub-<SUBJECT>/vlm_logs/vlm_panel_*.log`. See the per-Connect verification state machine row in `Documents/SoftwareDocs/GPU_Service_Cross_Host_Hardening_Notes.md`. Commits `e967d3b`, `3c5fb6c`, `ba18dc9`, `2834546`. Requires the matching `vlm_service.py` on the GPU host (echoes the request token in `_cmd_verify_chain`); panel-only updates do not need a GPU restart.
- Dataset exploration CLI (`explore_dataset_library.py`) writes under **`~/Documents/exploration_run_001`** by default (outputs stay outside the repo); override with `--out-dir`.
- Root `.gitignore` no longer lists `exploration_run_001/` (obsolete for in-repo paths).
- Renamed **`CONFIG_AUDIT.md` → `CHANGELOG.md`** and expanded scope to a project log.
- Optional **Git hooks** (`.githooks/pre-commit`): reminds you to update this file / `README.md` when key paths change (does not auto-edit files).
- **Cursor:** `.cursor/rules/finalize-documentation.md` tells the agent to update changelog/README when you request commit/push/finalize (no automatic Cursor hook on push—agent follows the rule in chat).

## 2026-03 — harmony_dev integration (decoder + tooling)

- **XGBoost (cov / cov+ERD):** Fitted PyRiemann `TangentSpace` reference per band; model bundles store `tangent_ref_mu` / `tangent_ref_beta`; online uses the same `tangent_space(..., reference=...)` mapping (no identity fallback).
- **Adaptive recentering:** Separate μ / β state (`Prev_T` vs `Prev_T_beta`); `save_transform` / `load_transform` v2 and drivers use a four-value tuple; `update_recentering` is threaded so recentering updates during MI/REST feedback, not during robot-motion classification, when callers disable updates.
- **MDM path:** `USE_CONFIDENCE_GATE` removed; behavior matches historical always-on recentering updates when `RECENTERING` is active.
- **Offline training:** Marker pairing (`Utils/marker_pairing.py`), optional artifact rejection (`Utils/artifact_rejection.py`), `Utils/dataset_exploration/*`, transfer-benchmark helpers, grid-search / held-out session scripts as applicable.
- **Control panel:** Runtime config tab, safer `config.py` edits, Arduino port/baud UX; gaze bind uses `config.GAZE_UDP_*`.
- **Hygiene:** Removed broken third-party `STM_interface/.../examples2.py`; `config.py` reorganized; visualization style hooks (`CLASS_VISUAL_STYLE`).

---

## Project log — configuration shadowing, tombstones, and checklists

Categories used in tables below: **(1)** active keep, **(2)** centralize / align sources, **(3)** obsolete removal candidate, **(4)** ambiguous, **(5)** analysis-only.

### Competing or shadowed sources (2 / 4)

| Topic | Sources | Notes |
|-------|---------|--------|
| Gaze bind host/port | `config.GAZE_UDP_*` vs control panel | Panel spawns gaze service using imported `config`. |
| Marker UDP | `config.UDP_MARKER["PORT"]` | Panel should align any readiness probe with `config`. |
| Arduino serial | `config.ARDUINO_*` vs panel | Online Glove reads **config** for port; panel can sync into `config.py`. |
| Simulation | `config.SIMULATION_MODE` | `Utils/networking` snapshots at import — restart processes after changes. |

### Category 1 — Active (retain)

Symbols used by drivers, `Utils/networking.py`, `runtime_common`, `visualization`, listeners, UDP dicts, triggers, timing, gaze block, decoder fields, FES, Arduino, etc.

### Category 2 — Centralize / operator surface

Runtime tab + single source for gaze UDP; prefer `config` as authority for ports and toggles the drivers read.

### Category 3 — Removal candidates (justify before delete)

| Symbol | Justification |
|--------|---------------|
| `ACCURACY_THRESHOLD` | Logging lists only; runtime decisions use `THRESHOLD_MI` / `THRESHOLD_REST`. |

### Control panel: `_marker_udp_port()`

Small helper in `control_panel.py` that returns `int(config.UDP_MARKER["PORT"])` (fallback `12345` if config import fails). Intended so marker readiness / UDP checks can follow `config` instead of a hardcoded port. **There may be no call sites** yet — safe to ignore, or wire a future “marker port reachable” check to it.

### Obsolete / not pursued — classifier-suite thread (no repo files)

Alternate offline trainers were explored in a Cursor thread; **not adopted** (low value vs maintaining `Generate_Riemannian_adaptive.py` and current XGB scripts). **They are not in this tree** — listed so nobody hunts for them.

| Intended script (never landed or removed) | Role (brief) |
|-------------------------------------------|--------------|
| `Generate_KernelRiemannian_adaptive.py` | Channel-kernel matrices + MDM-style training |
| `generate_fgmdm.py` | fgMDM after canonical cov pipeline |
| `generate_tangent_logreg.py` | Tangent space + logistic regression |
| `generate_tangent_mlp.py` | Tangent space + small MLP |
| `generate_spdnet.py` | SPDNet-style PyTorch classifier |
| `generate_tangent_laplacian_svm.py` | Tangent features + Laplacian-kernel SVM |
| `generate_riemannian_kernel_svm.py` | Precomputed Riemannian-distance kernel SVM |
| `Generate_TangentSpace_adaptive.py` | Early tangent-space variant (not kept) |

**Canonical offline Riemannian trainer:** `Generate_Riemannian_adaptive.py`.

### Third-party FES demos — `examples.py` / `examples2.py`

Paths: `STM_interface/1_packages/rehamoveLibrary/examples.py` (and formerly `examples2.py`).

Vendor demo scripts for the Rehamove Python API (`from rehamove import *`), not part of the Harmony BCI application. They hit serial devices (e.g. `/dev/ttyUSB0`) for hardware bring-up.

- **`examples2.py`** removed (broken / unused).
- **`examples.py`** remains a short demo; not imported by Harmony drivers.

### Category 4 — Ambiguous

`USE_ARDUINO` is mostly documentary relative to which drivers require hardware.

### Category 5 — Analysis-only

`Utils/dataset_exploration/*`, `explore_dataset_library.py`, grid-search scripts, transfer-benchmark utilities.

### Hardware validation checklist

- Restart marker driver / experiment processes after **`SIMULATION_MODE`** or UDP edits (`networking` import cache).
- **Save port/baud → config.py** before runs that read `config.ARDUINO_PORT`.
- **Runtime config** tab: restart driver after decoder / threshold / **`CLASS_VISUAL_STYLE`** changes.

### XGB adaptive recentering (reference)

`_adaptive_recenter_cov` in `Utils/runtime_common.py` applies Riemannian recentering separately to **mu** and **beta** covariance branches (`Prev_T` / `counter` vs `Prev_T_beta` / `counter_beta`) when `RECENTERING` is on, reducing cross-band leakage vs sharing one reference for both bands.
