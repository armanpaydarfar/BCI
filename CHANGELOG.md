# Changelog

Notable changes and project history are recorded here. This file replaces the older `CONFIG_AUDIT.md` name: it still contains **runtime and configuration notes** (below the dated log), not only `config.py`.

The format is **lightweight** (this repo is not strictly semver-tagged). Add bullet points under **[Unreleased]** when you merge user-visible behavior changes.

## [Unreleased]

### Tooling

- **Role-based env system + repo hygiene.** Installation now resolves by **role**, selected via `tools/bootstrap_machine.sh --role`: `control` (the Linux operator host — `environment.yml`, conda env `lsl`, the superset incl. CPU perception so single-box dev works) vs `server` (the GPU perception host — perception-only subset plus a CUDA torch build). The server env has two per-OS files, `environment.server.linux.yml` and `environment.server.windows.yml` (conda env `harmony-server`), identical except the CUDA torch pin. This supersedes the "single OS-agnostic `environment.yml`" framing in the WS3 bullet below: a fully-pinned export carries linux-64-only conda packages and can't solve on Windows, so the GPU host gets its own file. `bootstrap_machine.sh` also drops the retired sibling-`harmony_vlm` clone and `VLM_REPO_DIR` prompt. Separately, removed orphaned/stale tracked files from version control: `Data/` (111 `confusion_matrix_*.csv` from a retired writer), an unused `calib_pose_library_*.npz`, and unused `yolo11n.pt`/`yolov8n.pt`; the in-use `yolo26n.pt` and `poses_with_gaze_*.npz` are untracked-but-kept and now gitignored (`*.pt` / `*.npz`).
- **Hardware-free pytest suite + pre-commit gate (Phase 1a).** New `tests/` directory with `test_imports_smoke.py` (compile-all + curated-import + post-refactor importability), `test_networking_protocol.py` (`_to_wire` / `_is_coords_string` / `_base_token` / `_build_ack_map`), `test_config_contract.py` (machine-local key safe-defaults, panel-vs-config key drift, driver `TRIGGERS`/`ROBOT_OPCODES` references), and `test_gaze_tracking_pure.py` (`iou_xyxy`, `size_similarity_ratio`, `gaze_object_hit`, `SimpleSORTTracker`). The four pre-existing loopback scripts under `tools/test_*.py` move into `tests/`. New `pytest.ini` registers `testpaths = tests` and the `slow` marker. New `tools/pre-commit-pytest.sh` runs `pytest tests/ -m "not slow" -q` on every commit, wired in via `.githooks/pre-commit`; bypass with `git commit --no-verify` (never on `main`/`master`). Two prerequisite refactors (`UTIL_marker_stream.py` and `FES_listener.py`) defer import-time I/O into `main()` so the modules import without hardware. Cross-platform parity (Plan §6.1) deferred until a Windows CI runner exists. See `tests/README.md` and `Documents/SoftwareDocs/Harmony_Test_Suite_Plan.md`.
- **`FRAME_RELAY_EMBEDDED` default added to `config.py`** (`True`). Was previously read only via `getattr(_HCFG, ..., True)` in `control_panel.py` and missing from `config.py`; the test suite caught this drift on first run.

### Behaviour

- **harmony_vlm folded into `perception/` + single unified env (WS3).** The FastSAM / Depth Pro / Gemini-reasoner / Neon-reader source is now vendored in-tree under `perception/` (from `harmony_vlm` @ `cfa01b6`, with attribution headers) instead of imported from a sibling clone via `sys.path`. The separate `harmony_vlm` conda env is retired: the perception deps are merged into `environment.yml` (conda-forge numerical core + pip perception layer; the file is relaxed to name=version pins, conda-forge-only + `nodefaults`, OS-agnostic), and `vlm_service.py` now launches with the panel's own interpreter — still a separate process behind the UDP/GPU-host split, which is unchanged. This is a functional-equivalence reorg (where the code lives, not how it runs). **Config:** `VLM_REPO_DIR` and `VLM_CONDA_ENV` are retired; new machine-local guarded keys `PERCEPTION_MODELS_DIR` (model-weights dir) and `GOOGLE_API_KEY` (Gemini key, replacing the `harmony_vlm/.env`) live in `config_local.py`. `vlm_service.py` no longer takes `--repo-dir` (this supersedes the `--repo-dir` bullet below). `perception/`'s AprilTag / RealSense / overlay / Pupil-Core modules are vendored but **staged** — not importable until WS4/WS5 add their deps (see `perception/__init__.py`). `conda env create` cross-platform validation is pending a real Linux-GPU host / Windows rebuild (Phase 0 proved the dep set resolves on Linux-CPU + Windows-CUDA). See `Documents/SoftwareDocs/projects/harmony-bci/vlm-integration/`.
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
