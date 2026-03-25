# Harmony runtime configuration audit

Categories: **(1)** active keep, **(2)** centralize in config / align sources, **(3)** obsolete removal candidate, **(4)** ambiguous, **(5)** analysis-only.

## Competing or shadowed sources (category 2 / 4)

| Topic | Sources | Notes |
|-------|---------|--------|
| Gaze bind host/port | `config.GAZE_UDP_*` vs control panel | Panel spawns gaze service using imported `config`. |
| Marker UDP | `config.UDP_MARKER["PORT"]` | Panel should align any readiness probe with config. |
| Arduino serial | `config.ARDUINO_*` vs panel env | Online Glove reads **config** for port; panel can sync into `config.py`. |
| Simulation | `config.SIMULATION_MODE` | `Utils/networking` snapshots at import — restart processes after changes. |

## Category 1 — Active (retain)

Symbols used by drivers, `Utils/networking.py`, `runtime_common`, `visualization`, listeners, UDP dicts, triggers, timing, gaze block, decoder fields, FES, Arduino, etc.

## Category 2 — Centralize / operator surface

Runtime tab + single source for gaze UDP.

## Category 3 — Removal candidates (justify before delete)

| Symbol | Justification |
|--------|----------------|
| `ACCURACY_THRESHOLD` | Logging lists only; decisions use `THRESHOLD_MI` / `THRESHOLD_REST`. |

## Control panel: `_marker_udp_port()`

Small helper in `control_panel.py` that returns `int(config.UDP_MARKER["PORT"])` (fallback `12345` if config import fails). Added so marker readiness / UDP checks can stay aligned with `config` instead of a hardcoded port. **Currently no call sites** in `control_panel.py` — safe to ignore, or wire a future “marker port reachable” check to it.

## Obsolete / not pursued — classifier-suite thread (no repo files)

The following were explored in a Cursor thread as alternate offline trainers; **they were not adopted** (minimal value add vs maintaining `Generate_Riemannian_adaptive.py` and existing XGB scripts). **They do not exist in this tree** and are listed here so nobody hunts for them.

| Intended script (never landed or removed) | Role (brief) |
|-------------------------------------------|--------------|
| `Generate_KernelRiemannian_adaptive.py` | Channel-kernel matrices + MDM-style training |
| `generate_fgmdm.py` | fgMDM after canonical cov pipeline |
| `generate_tangent_logreg.py` | Tangent space + logistic regression |
| `generate_tangent_mlp.py` | Tangent space + small MLP |
| `generate_spdnet.py` | SPDNet-style PyTorch classifier |
| `generate_tangent_laplacian_svm.py` | Tangent features + Laplacian-kernel SVM |
| `generate_riemannian_kernel_svm.py` | Precomputed Riemannian-distance kernel SVM |
| `Generate_TangentSpace_adaptive.py` | Early tangent-space variant (superseded by planned suite; not kept) |

**Canonical offline Riemannian trainer:** `Generate_Riemannian_adaptive.py`.

## Third-party FES demos — `examples.py` / `examples2.py`

Paths: `STM_interface/1_packages/rehamoveLibrary/examples.py` and `examples2.py`.

These are **vendor / library demo scripts** for the **Rehamove** Python API (`from rehamove import *`), not part of the Harmony BCI application. They open `/dev/ttyUSB0` and run stimulation patterns for hardware bring-up.

- **`examples2.py`** has been removed from this repo snapshot because it was a broken scratch/demo and not referenced by the Harmony runtime.

**`examples.py`** is a separate, shorter demo in the same folder; it is ordinary script-style code (still not imported by Harmony drivers).

## Category 4 — Ambiguous

`USE_ARDUINO` mostly documentary.

## Category 5 — Analysis-only

`Utils/dataset_exploration/*`, `explore_dataset_library.py`, grid-search scripts.

## Hardware validation checklist

- Restart Marker/Driver after **SIMULATION_MODE** or UDP edits (`networking` import cache).
- **Save port/baud → config.py** before runs that read `config.ARDUINO_PORT`.
- **Runtime config** tab: restart driver after decoder/threshold/**CLASS_VISUAL_STYLE** changes.

## XGB adaptive recentering

`_adaptive_recenter_cov` in `Utils/runtime_common.py` applies Riemannian recentering separately to **mu** and **beta** covariance branches (`Prev_T`/`counter` vs `Prev_T_beta`/`counter_beta`) when `RECENTERING` is on, reducing cross-band leakage vs sharing one reference for both.
