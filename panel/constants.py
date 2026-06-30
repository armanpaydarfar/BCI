"""
panel/constants.py — module-level paths and config-derived globals.

Behaviour-preserving extraction of the path constants, the config.py import,
and the config-derived service/relay globals that used to live at the top of
control_panel.py. No Qt, no panel state — just the repo-root path math, the
`*_PY` / `*_SH` script paths, and the `getattr(config, ...)`-with-default
globals the panel and its collaborators read at call time. Lives in a leaf
module so panel controllers import these by name instead of the former bare
module-level references (no import cycle: this module depends only on os/sys
and the optional `config` module, never on control_panel or panel.netutils).
"""

from __future__ import annotations

import os
import sys

# ----------------- Paths & constants -----------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
try:
    import config as _HCFG  # noqa: E402
except Exception:
    _HCFG = None

_IS_WINDOWS = sys.platform == "win32"

MARKER_PY = os.path.join(ROOT, "UTIL_marker_stream.py")
DRIVER_ONLINE_PY = os.path.join(ROOT, "ExperimentDriver_Online.py")
DRIVER_ONLINE_GAZE_PY = os.path.join(ROOT, "ExperimentDriver_Online_GazeTracking.py")
DRIVER_ONLINE_GLOVE_PY = os.path.join(ROOT, "ExperimentDriver_Online_Glove.py")
DRIVER_BIMANUAL_PY = os.path.join(ROOT, "ExperimentDriver_Bimanual.py")
DRIVER_OFFLINE_PY = os.path.join(ROOT, "ExperimentDriver_Offline.py")
DRIVER_ERRP_ONLINE_PY = os.path.join(ROOT, "ExperimentDriver_ErrP_Online.py")
FES_PY = os.path.join(ROOT, "FES_listener.py")
STMSETUP_PY = os.path.join(ROOT, "STMsetup.py")
INIT_SH = os.path.join(ROOT, "initialize_devices.sh")

# ---- Harmony scripts you want on tab 2 ----
HARMONY_CALIBRATION_EXEC_PY = os.path.join(ROOT, "harmony_calibration_exec.py")
HARMONY_ONLINE_CONTROL_PY   = os.path.join(ROOT, "harmony_online_control.py")
# WS5 AprilTag gaze↔robot calibration + control test (rev03).
APRILTAG_CALIBRATE_PY       = os.path.join(ROOT, "tools", "apriltag_calibrate.py")
APRILTAG_CONTROL_TEST_PY    = os.path.join(ROOT, "tools", "apriltag_control_test.py")
APRILTAG_CONTROL_TEST_3D_PY = os.path.join(ROOT, "tools", "apriltag_control_test_3d.py")  # REV06

# ---- Gaze scripts (same folder as control_panel.py per your note) ----
GAZE_RUNNER_PY = os.path.join(ROOT, "gaze_runner.py")
GAZE_SERVICE_PY = os.path.join(ROOT, "gaze_runner.py")

GAZE_SERVICE_HOST = getattr(_HCFG, "GAZE_UDP_IP", "127.0.0.1") if _HCFG else "127.0.0.1"
# Bind vs dial: GAZE_SERVICE_HOST is what the panel dials; GAZE_BIND_HOST
# is what gaze_runner.py binds on. Production sets BIND=0.0.0.0 on Windows
# and SERVICE_HOST=<windows_lan_ip> on Linux.
GAZE_BIND_HOST = getattr(_HCFG, "GAZE_BIND_HOST", GAZE_SERVICE_HOST) if _HCFG else GAZE_SERVICE_HOST
GAZE_SERVICE_PORT = int(getattr(_HCFG, "GAZE_UDP_PORT", 5588)) if _HCFG else 5588
GAZE_QUERY_TIMEOUT_S = 0.8

# ---- VLM service (harmony_vlm) ----
VLM_SERVICE_PY      = os.path.join(ROOT, "vlm_service.py")
VLM_SERVICE_HOST    = getattr(_HCFG, "VLM_SERVICE_HOST", "127.0.0.1") if _HCFG else "127.0.0.1"
# Bind vs dial: VLM_SERVICE_HOST is what the panel dials; VLM_BIND_HOST is
# what vlm_service.py binds on (both UDP request and TCP overlay).
VLM_BIND_HOST       = getattr(_HCFG, "VLM_BIND_HOST", VLM_SERVICE_HOST) if _HCFG else VLM_SERVICE_HOST
VLM_SERVICE_PORT    = int(getattr(_HCFG, "VLM_SERVICE_PORT", 5589)) if _HCFG else 5589
PERCEPTION_MODELS_DIR = getattr(_HCFG, "PERCEPTION_MODELS_DIR", None) if _HCFG else None
VLM_MODEL           = getattr(_HCFG, "VLM_MODEL", "gemini-2.5-flash") if _HCFG else "gemini-2.5-flash"
VLM_ENABLE_DEPTH    = bool(getattr(_HCFG, "VLM_ENABLE_DEPTH", True)) if _HCFG else True
VLM_SESSION_ROOT    = getattr(_HCFG, "VLM_SESSION_ROOT", None) if _HCFG else None
# Reasoning commands can take tens of seconds; cheap status queries finish fast.
VLM_QUERY_TIMEOUT_S = 2.0
VLM_DECIDE_TIMEOUT_S = 40.0
GAZE_OR_BACKEND     = str(getattr(_HCFG, "GAZE_OR_BACKEND", "legacy")).lower() if _HCFG else "legacy"
NEON_COMPANION_HOST = str(getattr(_HCFG, "NEON_COMPANION_HOST", "")) if _HCFG else ""

# Remote-services mode: when True, the panel does NOT spawn local VLM /
# gaze service subprocesses; instead it shows remote-status badges fed by
# periodic `cmd: status` UDP pings (gaze_runner / vlm_service). Linux device
# host runs with this True; Windows GPU host runs with this False (services
# live locally and the panel manages their lifecycle).
SERVICES_HOSTED_REMOTELY = bool(getattr(_HCFG, "SERVICES_HOSTED_REMOTELY", False)) if _HCFG else False
PERCEPTION_FRAME_SOURCE  = str(getattr(_HCFG, "PERCEPTION_FRAME_SOURCE", "local")) if _HCFG else "local"

# Frame relay (used by the new Linux-side scene renderer in the VLM Video
# tab). Dial host comes from FRAME_RELAY_DIAL_HOST in production; loopback
# in single-machine dev. Bind/dial split mirrors VLM_BIND_HOST / VLM_SERVICE_HOST.
FRAME_RELAY_DIAL_HOST = str(getattr(_HCFG, "FRAME_RELAY_DIAL_HOST", "127.0.0.1")) if _HCFG else "127.0.0.1"
FRAME_RELAY_PORT      = int(getattr(_HCFG, "FRAME_RELAY_PORT", 5591)) if _HCFG else 5591
FRAME_RELAY_BIND_HOST = str(getattr(_HCFG, "FRAME_RELAY_HOST", "0.0.0.0")) if _HCFG else "0.0.0.0"
FRAME_RELAY_HZ        = float(getattr(_HCFG, "FRAME_RELAY_HZ", 15.0)) if _HCFG else 15.0
# When True, the panel hosts FrameRelayServer in-process so its
# scene-tab widget can consume bundles via add_local_subscriber (raw
# BGR, no JPEG encode/decode). This runs the Neon H.264 decode in the
# panel's own process, where the Qt paint + JPEG-encode load contends
# for the GIL with the SDK's RTP-receive thread and corrupts the lower
# image slices under motion ("tearing" — rootcause record 2026-06-22).
# False → the panel spawns `python -m Utils.frame_relay` as a managed
# child process (see _start_relay_subprocess) and the widget consumes it
# via RemoteFrameReader; the decode then runs in its own GIL, isolated
# from the panel load. False is the validated fix for the tearing.
# Windows/remote clients connect to the same TCP port either way.
FRAME_RELAY_EMBEDDED  = bool(getattr(_HCFG, "FRAME_RELAY_EMBEDDED", True)) if _HCFG else True

# Modes choose which robot tool to launch remotely
MODES = ["Gaze_Tracking", "MI_Bimanual", "Simulation"]

# Driver choices
DRIVERS = [
    "ExperimentDriver_Online",
    "ExperimentDriver_ErrP_Online",
    "ExperimentDriver_Bimanual",
    "ExperimentDriver_Offline",
    "ExperimentDriver_Online_GazeTracking",
    "ExperimentDriver_Online_Glove",
]

TRAINING_SCRIPT_ENTRIES = [
    ("Riemannian adaptive → sub-*_model.pkl", "Generate_Riemannian_adaptive.py"),
    ("XGBoost covariance features", "generate_xgboost_cov_features.py"),
]

__all__ = [
    "ROOT", "_HCFG", "_IS_WINDOWS",
    "MARKER_PY", "DRIVER_ONLINE_PY", "DRIVER_ONLINE_GAZE_PY",
    "DRIVER_ONLINE_GLOVE_PY", "DRIVER_BIMANUAL_PY", "DRIVER_OFFLINE_PY",
    "DRIVER_ERRP_ONLINE_PY", "FES_PY", "STMSETUP_PY", "INIT_SH",
    "HARMONY_CALIBRATION_EXEC_PY", "HARMONY_ONLINE_CONTROL_PY",
    "APRILTAG_CALIBRATE_PY", "APRILTAG_CONTROL_TEST_PY", "APRILTAG_CONTROL_TEST_3D_PY",
    "GAZE_RUNNER_PY", "GAZE_SERVICE_PY", "VLM_SERVICE_PY",
    "GAZE_SERVICE_HOST", "GAZE_BIND_HOST", "GAZE_SERVICE_PORT",
    "GAZE_QUERY_TIMEOUT_S", "VLM_SERVICE_HOST", "VLM_BIND_HOST",
    "VLM_SERVICE_PORT", "PERCEPTION_MODELS_DIR", "VLM_MODEL",
    "VLM_ENABLE_DEPTH", "VLM_SESSION_ROOT", "VLM_QUERY_TIMEOUT_S",
    "VLM_DECIDE_TIMEOUT_S", "GAZE_OR_BACKEND", "NEON_COMPANION_HOST",
    "SERVICES_HOSTED_REMOTELY", "PERCEPTION_FRAME_SOURCE",
    "FRAME_RELAY_DIAL_HOST", "FRAME_RELAY_PORT", "FRAME_RELAY_BIND_HOST",
    "FRAME_RELAY_HZ", "FRAME_RELAY_EMBEDDED",
    "MODES", "DRIVERS", "TRAINING_SCRIPT_ENTRIES",
]
