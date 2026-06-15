#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Harmony_Bimanual — Control Panel (Simplified, no polling)
Requires: pip install PySide6 psutil pyserial

Repo layout assumed:
~/Projects/Harmony_Bimanual/
    control_panel.py
    gaze_runner.py
    UTIL_marker_stream.py
    ExperimentDriver_Online.py
    ExperimentDriver_Bimanual.py
    ExperimentDriver_Offline.py
    FES_listener.py
    UDPRobot.py
    STMsetup.py
    initialize_devices.sh
    config.py

Gaze additions (NEW):
- Adds "Gaze" to View: dropdown and log buffers.
- Adds a Gaze tab with:
    * Run Gaze (UI mode) button (runner)
- Adds on Main tab (above Robot row):
    * Gaze Service LED + Start Headless + Start With UI + Stop + Query Telemetry (UDP)
- All gaze stdout/stderr is captured into the "Gaze" log view (no terminal logs).
- Telemetry query uses UDP JSON request/response (single datagram, newline-free).

IMPORTANT:
- gaze_runner.py must support:
    --mode {runner,service}
    --display {0,1}
    --prints {0,1}
    --host HOST
    --port PORT
- NO --telemetry flag is passed (that caused argparse errors).
"""

import os, sys, shlex, time, re, tempfile, socket, subprocess, json, threading, glob
import serial
import serial.tools.list_ports

# Import ARDUINO_PORT from config; fallback to default if unavailable
try:
    from config import ARDUINO_PORT
except ImportError:
    ARDUINO_PORT = ""

from dataclasses import dataclass, field
from typing import Optional, Dict

from PySide6.QtCore import Qt, QTimer, QProcess, QByteArray, QSize, QThread, Signal
from PySide6.QtGui import QAction, QClipboard, QTextCursor, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QGridLayout, QLineEdit,
    QTextEdit, QGroupBox, QMessageBox, QSplitter, QToolBar, QStyle,
    QScrollArea, QFormLayout, QDoubleSpinBox, QSpinBox,
    QListWidget, QSizePolicy,
)


def _fixed_v(widget: QWidget) -> QWidget:
    """Pin a widget's vertical size policy so it stops absorbing leftover
    grid space. QWidget defaults to Preferred-vertical, which makes any
    HBox-holder row in a QGridLayout stretch to 4-5x its natural height
    when the panel has spare vertical room. Fixed clamps it at the
    sizeHint."""
    sp = widget.sizePolicy()
    sp.setVerticalPolicy(QSizePolicy.Fixed)
    widget.setSizePolicy(sp)
    return widget

# ----------------- Paths & constants -----------------
ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PY = os.path.join(ROOT, "config.py")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
try:
    import config as _HCFG  # noqa: E402
except Exception:
    _HCFG = None

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
# BGR, no JPEG encode/decode). Windows clients still connect to the
# same TCP port. False → widget falls back to RemoteFrameReader and
# the user is expected to run `python -m Utils.frame_relay` separately.
FRAME_RELAY_EMBEDDED  = bool(getattr(_HCFG, "FRAME_RELAY_EMBEDDED", True)) if _HCFG else True


_IS_WINDOWS = sys.platform == "win32"


def _sleep_inhibit(enable: bool) -> None:
    """Prevent / allow Windows from sleeping while perception services run.

    Windows-only — uses kernel32.SetThreadExecutionState with
    ES_CONTINUOUS | ES_SYSTEM_REQUIRED. Per
    SoftwareDocs/GPU_Service_Host_Architecture_Plan.md §4.10 the GPU host
    must not sleep mid-session. POSIX is a no-op (Linux deployment uses
    systemd-inhibit / external power management).
    """
    if not _IS_WINDOWS:
        return
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enable else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))
    except Exception:
        # Best-effort — sleep inhibit is a hardening, not a correctness gate.
        pass


def _detect_lan_ip() -> Optional[str]:
    """Best-effort LAN IP discovery. Opens a UDP socket and asks the
    kernel which interface it would use to reach a routable address —
    no packet is actually sent. Returns None if no route is available
    (offline, captive portal, etc.)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        try:
            s.close()
        except OSError:
            pass


def _detect_tailscale_ip() -> Optional[str]:
    """Return this host's Tailscale IPv4 if the daemon is up + signed in,
    else None. The tailnet IP is the address remote peers (Windows GPU
    host, lab box) should dial — it's stable across physical networks,
    unlike the LAN IP from _detect_lan_ip()."""
    try:
        out = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True, text=True, timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    ip = (out.stdout or "").strip().splitlines()
    return ip[0].strip() if ip and ip[0].strip() else None


_LAN_IP = _detect_lan_ip()
_TS_IP = _detect_tailscale_ip()
if _LAN_IP:
    print(f"[control_panel] Linux LAN IP = {_LAN_IP} (physical network)", flush=True)
else:
    print("[control_panel] Linux LAN IP = <unavailable> (no route)", flush=True)
if _TS_IP:
    print(
        f"[control_panel] Linux Tailscale IP = {_TS_IP}  "
        f"— remote peers should use this in FRAME_RELAY_DIAL_HOST "
        f"(stable across networks)",
        flush=True,
    )
else:
    print(
        "[control_panel] Tailscale not active — peers reach this box via the LAN IP above. "
        "Run `sudo tailscale up` to get a stable tailnet IP for cross-network use.",
        flush=True,
    )


def _kill_orphan_vlm_service() -> None:
    """Kill any leftover vlm_service.py processes from a prior crash.

    A killed launch can leave the service's python alive (mid-startup, or
    children it spawned) holding the UDP port. POSIX uses `pkill -f`; Windows
    walks tasklist for python.exe with the script in its command line and
    kills with taskkill.
    """
    if _IS_WINDOWS:
        try:
            out = subprocess.check_output(
                ["wmic", "process", "where",
                 "name='python.exe' and CommandLine like '%vlm_service.py%'",
                 "get", "ProcessId"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return
        for line in out.splitlines():
            line = line.strip()
            if line.isdigit():
                subprocess.run(
                    ["taskkill", "/F", "/PID", line],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
    else:
        subprocess.run(["pkill", "-f", "vlm_service.py"], capture_output=True)


def _marker_udp_port() -> int:
    if _HCFG is not None:
        return int(_HCFG.UDP_MARKER["PORT"])
    return 12345

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

# ----------------- Config read/write helpers -----------------
SUBJECT_RE = re.compile(r'^(TRAINING_SUBJECT\s*=\s*)([\'"])([^\'"]+)\2\s*$', re.M)
FES_RE     = re.compile(r'^(FES_toggle\s*=\s*)([01])\s*$', re.M)
SIM_RE     = re.compile(r'^(SIMULATION_MODE\s*=\s*)(True|False)(\s*(#.*)?)\s*$', re.M)

def read_text(path: str) -> str:
    """
    Read a UTF-8 text file.

    Returns an empty string if the file does not exist.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def write_atomic(path: str, text: str):
    """
    Write text to `path` using an atomic replace.

    This reduces the chance of producing a partially-written `config.py`
    if the application crashes mid-write.
    """
    tmp = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
    try:
        tmp.write(text)
        tmp.flush(); os.fsync(tmp.fileno()); tmp.close()
        os.replace(tmp.name, path)
    except Exception:
        try: os.unlink(tmp.name)
        except Exception: pass
        raise

def read_simulation_mode(default=False) -> bool:
    """
    Read `SIMULATION_MODE` from `config.py` using a regex match.
    """
    txt = read_text(CONFIG_PY)
    m = SIM_RE.search(txt)
    if not m:
        return bool(default)
    return (m.group(2) == "True")

def write_simulation_mode(val: bool):
    """
    Update `SIMULATION_MODE = True/False` inside `config.py` in-place.

    Implementation detail: uses regex substitution so we don't need to import
    or rewrite unrelated config lines.
    """
    val_txt = "True" if val else "False"
    txt = read_text(CONFIG_PY)
    if SIM_RE.search(txt):
        new = SIM_RE.sub(rf'\g<1>{val_txt}', txt)
    else:
        sep = "" if (txt.endswith("\n") or txt == "") else "\n"
        new = txt + f"{sep}SIMULATION_MODE = {val_txt}\n"
    write_atomic(CONFIG_PY, new)

def read_training_subject(default="PILOT007"):
    """
    Read `TRAINING_SUBJECT` from `config.py`.
    """
    txt = read_text(CONFIG_PY)
    m = SUBJECT_RE.search(txt)
    return m.group(3) if m else default

def write_training_subject(val: str):
    """
    Update `TRAINING_SUBJECT = "<val>"` inside `config.py`.
    """
    txt = read_text(CONFIG_PY)
    if SUBJECT_RE.search(txt):
        new = SUBJECT_RE.sub(rf'\g<1>"{val}"', txt)
    else:
        sep = "" if (txt.endswith("\n") or txt == "") else "\n"
        new = txt + f"{sep}TRAINING_SUBJECT = \"{val}\"\n"
    write_atomic(CONFIG_PY, new)

def read_fes_toggle(default=0):
    """
    Read `FES_toggle` from `config.py` as an integer 0/1.
    """
    txt = read_text(CONFIG_PY)
    m = FES_RE.search(txt)
    try:
        return int(m.group(2)) if m else default
    except Exception:
        return default

def write_fes_toggle(val: int):
    """
    Update `FES_toggle = 0/1` inside `config.py`.

    Any truthy value is normalized to 1, otherwise 0.
    """
    val = 1 if val else 0
    txt = read_text(CONFIG_PY)
    if FES_RE.search(txt):
        new = FES_RE.sub(rf'\g<1>{val}', txt)
    else:
        sep = "" if (txt.endswith("\n") or txt == "") else "\n"
        new = txt + f"{sep}FES_toggle = {val}\n"
    write_atomic(CONFIG_PY, new)


CONFIG_LOCAL_PY = os.path.join(ROOT, "config_local.py")

# Keys that live in config_local.py (machine-local). config.py holds safe
# defaults; the bottom of config.py does `from config_local import *` so a
# value set here shadows the default at import time. Mirrored in
# ~/.claude/hooks/config-py-guard.sh — keep both lists in sync.
LOCAL_CONFIG_KEYS = frozenset({
    "WORKING_DIR", "DATA_DIR",
    "GAZE_UDP_IP", "GAZE_BIND_HOST",
    "NEON_COMPANION_HOST",
    "PERCEPTION_FRAME_SOURCE", "SERVICES_HOSTED_REMOTELY",
    "FRAME_RELAY_HOST", "FRAME_RELAY_DIAL_HOST",
    "PERCEPTION_MODELS_DIR", "GOOGLE_API_KEY",
    "VLM_SERVICE_HOST", "VLM_BIND_HOST",
    "ARDUINO_PORT",
})


def _assign_line_re(name: str) -> re.Pattern:
    return re.compile(rf"^(\s*{re.escape(name)}\s*=\s*)([^#\n]+?)(\s*(#.*)?)\s*$", re.M)


def _find_assignment(name: str):
    """Return (file_path, match_object) for the first config file that
    contains an assignment to ``name``. config_local.py is checked first
    so local overrides take precedence — same precedence the live import
    uses at runtime. Returns (None, None) if neither file has the key."""
    pat = _assign_line_re(name)
    if os.path.isfile(CONFIG_LOCAL_PY):
        m = pat.search(read_text(CONFIG_LOCAL_PY))
        if m:
            return CONFIG_LOCAL_PY, m
    m = pat.search(read_text(CONFIG_PY))
    if m:
        return CONFIG_PY, m
    return None, None


def _read_float_key(name: str, default: float) -> float:
    _, m = _find_assignment(name)
    if not m:
        return default
    try:
        return float(m.group(2).strip())
    except ValueError:
        return default


def _read_int_key(name: str, default: int) -> int:
    _, m = _find_assignment(name)
    if not m:
        return default
    try:
        return int(float(m.group(2).strip()))
    except ValueError:
        return default


def _read_bool_key(name: str, default: bool) -> bool:
    _, m = _find_assignment(name)
    if not m:
        return default
    v = m.group(2).strip()
    if v == "True":
        return True
    if v == "False":
        return False
    return default


def _read_quoted_str_key(name: str, default: str) -> str:
    _, m = _find_assignment(name)
    if not m:
        return default
    raw = m.group(2).strip()
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    return raw


def _write_assign_rhs_local(name: str, rhs: str) -> None:
    """Write/update an assignment in config_local.py.

    config_local.py is gitignored and may not exist on a fresh checkout —
    if missing, we create it with a single-line header. If the key is
    already assigned, the line is rewritten in place; otherwise a new
    line is appended.
    """
    header = (
        "# Machine-local config overrides. Gitignored. Loaded by config.py\n"
        "# via `from config_local import *`. Schema in config_local.example.py.\n"
    )
    if not os.path.isfile(CONFIG_LOCAL_PY):
        write_atomic(CONFIG_LOCAL_PY, header)
    txt = read_text(CONFIG_LOCAL_PY)
    pat = _assign_line_re(name)
    if pat.search(txt):
        new = pat.sub(rf"\g<1>{rhs}\3", txt, count=1)
    else:
        sep = "" if (txt.endswith("\n") or txt == "") else "\n"
        new = txt + f"{sep}{name} = {rhs}\n"
    write_atomic(CONFIG_LOCAL_PY, new)


def _write_assign_rhs(name: str, rhs: str):
    """Route writes to the right file. Machine-local keys land in
    config_local.py; everything else stays in config.py (where the
    runtime-config tab's algorithm settings live)."""
    if name in LOCAL_CONFIG_KEYS:
        _write_assign_rhs_local(name, rhs)
        return
    txt = read_text(CONFIG_PY)
    pat = _assign_line_re(name)
    if not pat.search(txt):
        raise ValueError(f"config.py: assignment for {name} not found")
    new = pat.sub(rf"\g<1>{rhs}\3", txt, count=1)
    write_atomic(CONFIG_PY, new)


def _read_01_key(name: str, default: int) -> int:
    v = _read_int_key(name, default)
    return 1 if v else 0


ARDUINO_PORT_RE = re.compile(
    r"^(\s*ARDUINO_PORT\s*=\s*)([\"'])([^\"']*)\2(\s*(#.*)?)\s*$", re.M
)
ARDUINO_BAUD_RE = re.compile(
    r"^(\s*ARDUINO_BAUD\s*=\s*)(\d+)(\s*(#.*)?)\s*$", re.M
)


def read_arduino_baud_from_config(default: int = 9600) -> int:
    txt = read_text(CONFIG_PY)
    m = ARDUINO_BAUD_RE.search(txt)
    try:
        return int(m.group(2)) if m else default
    except Exception:
        return default


def write_arduino_port_to_config(port: str):
    """ARDUINO_PORT is machine-local — route through the local writer so
    the value lands in config_local.py rather than the committed
    config.py default."""
    _write_assign_rhs_local("ARDUINO_PORT", f'"{port}"')


def write_arduino_baud_to_config(baud: int):
    txt = read_text(CONFIG_PY)
    b = int(baud)
    if ARDUINO_BAUD_RE.search(txt):
        new = ARDUINO_BAUD_RE.sub(rf"\g<1>{b}\3", txt, count=1)
    else:
        sep = "" if (txt.endswith("\n") or txt == "") else "\n"
        new = txt + f"{sep}ARDUINO_BAUD = {b}\n"
    write_atomic(CONFIG_PY, new)


TRAINING_SCRIPT_ENTRIES = [
    ("Riemannian adaptive → sub-*_model.pkl", "Generate_Riemannian_adaptive.py"),
    ("XGBoost covariance features", "generate_xgboost_cov_features.py"),
]

# ----------------- UDP readiness probe -----------------
def _is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """
    UDP bind probe:
      - If we can bind, nobody is bound -> NOT in use
      - If bind fails, in use
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind((host, port))
        s.close()
        return False
    except OSError:
        s.close()
        return True

# ----------------- Process model -----------------
@dataclass
class Proc:
    name: str
    cmd: Optional[str]
    cwd: str
    env: Dict[str, str] = field(default_factory=dict)
    q: Optional[QProcess] = None
    status: str = "stopped"  # stopped|starting|running|error
    pid: Optional[int] = None
    out: bytearray = field(default_factory=bytearray)
    err: bytearray = field(default_factory=bytearray)

# ----------------- Main Window -----------------
class ControlPanel(QMainWindow):
    # Emitted by the off-thread remote-status worker. Marshals the UDP
    # status reply (or a "down" sentinel) back to the GUI thread so the
    # 0.4 s socket timeout doesn't block paint.
    _remote_status_received = Signal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Harmony Control Panel — Simplified")
        # Default size tuned for a typical 1366x768 laptop / second
        # monitor. setMinimumSize is intentionally small so the operator
        # can drag the window narrower without Qt clamping at the
        # natural-size of the widest child (the original 1250 default
        # was rooted in inner-widget sizeHints rather than a deliberate
        # choice and made the window unusable on smaller displays).
        self.resize(1100, 700)
        self.setMinimumSize(700, 500)
        self._remote_status_in_flight = False
        # Per-Connect verification state machine. Each Connect generates
        # a fresh `_connect_token` (time.time_ns()); flags advance as
        # each phase completes:
        #   PHASE_SEND     — handshake from relay → _send_observed=True
        #   PHASE_COMPUTE  — cmd=status reply ok=True → _compute_observed=True
        #   PHASE_RECEIVE  — relay first publish + GPU has fresh bundle
        #                    → fire cmd=verify_chain with token
        #   DONE           — chain_verify push with matching token
        # Stale GPU-cache pushes from a prior session carry no token
        # (or an old one), so they cannot trip Receive on a reconnect.
        # Also tracks `_verify_chain_attempts` for the one-retry
        # policy on `no_frame` races.
        self._connect_token: Optional[int] = None
        self._send_observed: bool = False
        self._compute_observed: bool = False
        self._receive_observed: bool = False
        self._verify_chain_in_flight: bool = False
        self._verify_chain_attempts: int = 0
        self._remote_status_received.connect(self._apply_remote_status)

        # State
        self.mode = MODES[0]
        try:
            sim_cfg = read_simulation_mode(default=False)
        except Exception:
            sim_cfg = False
        self.mode = "Simulation" if sim_cfg else MODES[0]

        self.driver_choice = DRIVERS[0]
        self.training_subject = read_training_subject()
        self.fes_enabled_pref = read_fes_toggle()

        # Arduino / BCI online config
        self.serial_port_name = ""
        try:
            self.serial_baudrate = str(read_arduino_baud_from_config(9600))
        except Exception:
            self.serial_baudrate = "9600"

        # Procs (QProcess-managed)
        self.marker = Proc("Marker Stream", f'python -u "{MARKER_PY}"', ROOT)
        self.driver = Proc("Experiment Driver", None, ROOT)
        self.fes    = Proc("FES Listener", f'python -u "{FES_PY}"', ROOT)

        # ---- Gaze procs (NEW) ----
        self.gaze_runner = Proc("Gaze Runner", None, ROOT)
        self.gaze_service = Proc("Gaze Service", None, ROOT)

        # ---- VLM service proc ----
        self.vlm_service = Proc("VLM Service", None, ROOT)
        self._vlm_last_snapshot_id: Optional[str] = None

        # Robot terminal
        self.robot_term: Optional[QProcess] = None
        self.labrec_term: Optional[QProcess] = None
        self.eego_term: Optional[QProcess] = None

        # Logs
        self._log_buffers: Dict[str, str] = {"Marker": "", "FES": "", "Driver": "", "Gaze": "", "VLM": "", "Relay": "", "Robot": "", "Panel": ""}
        self._current_log_target = "Panel"
        # Subject-tied VLM log file. Captures only the "VLM" buffer
        # (vlm_service stdout when local + every panel-side UDP TX/RX
        # trace + the periodic seg-stream readouts). Other buffers are
        # intentionally NOT teed here — Marker/FES/Driver have their
        # own files via their respective scripts; Robot/Gaze/Panel
        # stay in-memory only for now (revisit if a forensic need
        # appears). Path mirrors marker_logs / impedance_logs naming.
        self._vlm_log_subject: Optional[str] = None
        self._vlm_log_path: Optional[str] = None
        self._vlm_log_fh = None
        self._open_vlm_log_file(self.training_subject)
        # Subject-tied frame_relay log file. Co-located with the VLM
        # log under <DATA_DIR>/sub-<SUBJECT>/vlm_logs/ — the relay is
        # the upstream half of the same perception pipeline, so
        # keeping the two files together makes post-session forensics
        # easier. Default file naming: frame_relay_<timestamp>.log.
        self._relay_log_subject: Optional[str] = None
        self._relay_log_path: Optional[str] = None
        self._relay_log_fh = None
        self._open_relay_log_file(self.training_subject)
        # Replace the default stdout sinks of frame_relay AND
        # scene_only_neon_reader with one that tees lines into the
        # panel's "Relay" buffer + the file opened above. The two
        # modules are halves of the same upstream pipeline (reader
        # opens Neon → relay pumps frames out), so co-locating their
        # output is what an operator wants when troubleshooting why
        # frames aren't flowing. Standalone usages keep the default
        # print sinks (set_log_callback isn't called there).
        try:
            from Utils.frame_relay import set_log_callback as _set_relay_log_cb
            _set_relay_log_cb(self._relay_log_callback)
        except Exception as e:
            self._append_log(
                "Panel",
                f"[{self._ts()}] WARN: could not install relay log callback: {e}\n",
            )
        try:
            from Utils.scene_only_neon_reader import set_log_callback as _set_reader_log_cb
            _set_reader_log_cb(self._relay_log_callback)
        except Exception as e:
            self._append_log(
                "Panel",
                f"[{self._ts()}] WARN: could not install neon-reader log callback: {e}\n",
            )

        # Build UI
        self._build_ui()

        # Configure initial commands
        self._set_cmds_for_mode_and_driver()

        # Initialize LEDs
        self._set_led(self.lbl_robot_init, "stopped")
        self._set_led(self.lbl_robot, "stopped")
        self._set_led(self.lbl_marker, "stopped")
        self._set_led(self.lbl_fes, "stopped")
        self._set_led(self.lbl_driver, "stopped")
        self._set_led(self.lbl_eego, "stopped")
        self._set_led(self.lbl_labrec, "stopped")
        self._set_led(self.lbl_gaze_service, "stopped")
        self._set_led(self.lbl_compute_led, "stopped")
        self._set_led(self.lbl_arduino, "stopped")

        # When services are hosted remotely (Linux operator panel pointed at
        # a Windows GPU host) the start/stop buttons can't drive local
        # processes. Disable them and stand up a remote-status timer.
        if SERVICES_HOSTED_REMOTELY:
            self._configure_remote_services_ui()

        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(400)
        self.ui_timer.timeout.connect(self._tick)
        self.ui_timer.start()

        # Periodic remote-status poller (cheap UDP ping). 1 s cadence.
        self._remote_status_timer = QTimer(self)
        self._remote_status_timer.setInterval(1000)
        self._remote_status_timer.timeout.connect(self._poll_remote_status)
        if SERVICES_HOSTED_REMOTELY:
            self._remote_status_timer.start()

        # Frame relay status — surfaced in either mode. The relay is a
        # separate process from the perception services; running on Linux
        # in production but optionally on Windows for single-machine tests.
        self._relay_status_timer = QTimer(self)
        self._relay_status_timer.setInterval(2000)
        self._relay_status_timer.timeout.connect(self._poll_relay_status)
        if PERCEPTION_FRAME_SOURCE == "remote" or SERVICES_HOSTED_REMOTELY:
            self._relay_status_timer.start()

        # Seg-stream readout — only running while the operator has the
        # stream toggled on. Cadence matches the service's stats window
        # (_SEG_STREAM_STATS_S = 5s) so the same numbers refresh once per
        # log line. Emits to the VLM log buffer; a separate timer keeps
        # this off the 1 Hz remote-status path so latency on that hot
        # path stays unchanged.
        self._seg_stream_log_timer = QTimer(self)
        self._seg_stream_log_timer.setInterval(5000)
        self._seg_stream_log_timer.timeout.connect(self._poll_seg_stream_stats)
        # Suppress duplicate emissions when last_emit_t hasn't advanced.
        self._last_seg_stream_emit_t: float = 0.0

    # ---------- UI build ----------
    def _build_ui(self):
        self._building_ui = True

        tb = QToolBar("Main")
        tb.setIconSize(QSize(18, 18))
        self.addToolBar(tb)
        act_init = QAction(self.style().standardIcon(QStyle.SP_ComputerIcon), "Initialize (open script)", self)
        act_init.triggered.connect(self.on_initialize)
        tb.addAction(act_init)

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # Main tab
        main = QWidget(); tabs.addTab(main, "Main")
        mv = QVBoxLayout(main)

        # Top row: Mode + Driver + Subject + FES + Tools
        # Tight margins — the default QHBoxLayout spacing leaves the
        # group boxes feeling separated; with five of them in a row that
        # extra padding pushes the window width up by ~40 px.
        top = QHBoxLayout(); top.setContentsMargins(0, 0, 0, 0)
        mv.addLayout(top)
        mv.setSpacing(4)

        # Mode
        gb_mode = QGroupBox("Mode"); fm = QHBoxLayout(gb_mode)
        self.cmb_mode = QComboBox(); self.cmb_mode.addItems(MODES)
        self.cmb_mode.setCurrentText(self.mode)
        self.cmb_mode.currentTextChanged.connect(self.on_mode_changed)
        fm.addWidget(QLabel("Mode:"))
        fm.addWidget(self.cmb_mode)
        top.addWidget(gb_mode)

        # Driver
        gb_drv = QGroupBox("Driver"); fd = QHBoxLayout(gb_drv)
        self.cmb_driver = QComboBox(); self.cmb_driver.addItems(DRIVERS)
        self.cmb_driver.setCurrentText(self.driver_choice)
        self.cmb_driver.currentTextChanged.connect(self.on_driver_choice_changed)
        # Driver names like "ExperimentDriver_Online_GazeTracking" are
        # ~37 chars long; QComboBox's default AdjustToContentsOnFirstShow
        # sizes the widget to the longest item and that single combo
        # used to push the panel >1100 px wide. Cap the visible width
        # — the full text is still visible in the dropdown.
        self.cmb_driver.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_driver.setMinimumContentsLength(15)
        fd.addWidget(QLabel("Driver:"))
        fd.addWidget(self.cmb_driver)
        top.addWidget(gb_drv, 2)

        # Subject
        gb_subj = QGroupBox("Training Subject"); fs = QHBoxLayout(gb_subj)
        self.cmb_subject = QComboBox(); self.cmb_subject.setEditable(True)
        self.cmb_subject.addItem(self.training_subject)
        self.cmb_subject.setCurrentText(self.training_subject)
        btn_save_subj = QPushButton("Save"); btn_copy_subj = QPushButton("Copy")
        btn_save_subj.clicked.connect(self.on_save_subject)
        btn_copy_subj.clicked.connect(self.on_copy_subject)
        fs.addWidget(self.cmb_subject, 1); fs.addWidget(btn_save_subj); fs.addWidget(btn_copy_subj)
        top.addWidget(gb_subj, 2)

        # FES toggle
        gb_fes = QGroupBox("FES"); ff = QHBoxLayout(gb_fes)
        self.chk_fes = QCheckBox("Enable")
        self.chk_fes.setChecked(bool(self.fes_enabled_pref))
        self.chk_fes.toggled.connect(self.on_fes_pref_toggled)
        btn_fes_cfg = QPushButton("Configure")
        btn_fes_cfg.setToolTip("Open STMsetup.py")
        btn_fes_cfg.clicked.connect(self.on_open_fes_cfg)
        ff.addWidget(self.chk_fes); ff.addWidget(btn_fes_cfg)
        top.addWidget(gb_fes)

        # Utilities
        gb_utils = QGroupBox("Utilities"); fu = QHBoxLayout(gb_utils)
        self.btn_mne = QPushButton("MNE Viewer")
        self.btn_mne.setToolTip("Open MNE-LSL viewer")
        self.btn_mne.clicked.connect(self.on_open_mne_viewer)
        fu.addWidget(self.btn_mne)
        self.btn_impedance = QPushButton("Impedance")
        self.btn_impedance.setToolTip("Open impedance monitor")
        self.btn_impedance.clicked.connect(self.on_open_impedance_monitor)
        fu.addWidget(self.btn_impedance)
        top.addWidget(gb_utils)

        # Middle: Controls + Logs
        split = QSplitter(); mv.addWidget(split, 1)
        controls = QWidget(); split.addWidget(controls)
        grid = QGridLayout(controls)
        # Tighten row pitch — Qt's default vertical spacing (~6 px) plus
        # default margins make the module rows feel sparse. Pulling them
        # together makes the whole control column scan as one block of
        # related actions rather than a list with gaps.
        grid.setVerticalSpacing(2)
        grid.setHorizontalSpacing(6)
        grid.setContentsMargins(6, 4, 6, 4)

        row = 0
        # ===== Robot =====
        # Init + Start + Remove Overrides on one row — these three are
        # invariably done in sequence at the start of a session, so keeping
        # them adjacent matches the operator's actual workflow.
        self.lbl_robot_init = QLabel("●"); self._set_led(self.lbl_robot_init, "stopped")
        self.lbl_robot      = QLabel("●"); self._set_led(self.lbl_robot, "stopped")
        led_box = QHBoxLayout()
        led_box.setContentsMargins(0, 0, 0, 0)
        led_box.addWidget(self.lbl_robot_init)
        led_box.addWidget(self.lbl_robot)
        led_holder = _fixed_v(QWidget()); led_holder.setLayout(led_box)
        btn_init_robot = QPushButton("Init Robot (SSH)")
        btn_init_robot.clicked.connect(self.on_init_robot)
        self.btn_robot_start     = QPushButton("Start (SSH terminal)")
        self.btn_robot_removeovr = QPushButton("Remove Overrides")
        self.btn_robot_start.clicked.connect(self.on_robot_start)
        self.btn_robot_removeovr.clicked.connect(self.on_robot_remove_overrides)
        grid.addWidget(QLabel("<b>Robot</b>"), row, 0)
        grid.addWidget(led_holder, row, 1)
        grid.addWidget(btn_init_robot, row, 2)
        grid.addWidget(self.btn_robot_start, row, 3)
        grid.addWidget(self.btn_robot_removeovr, row, 4)
        row += 1

        # eegoSports
        self.lbl_eego = QLabel("●"); self._set_led(self.lbl_eego, "stopped")
        grid.addWidget(QLabel("<b>eegoSports</b>"), row, 0)
        grid.addWidget(self.lbl_eego, row, 1)
        btn_eego = QPushButton("Open eegoSports")
        btn_eego.clicked.connect(self.on_open_eego)
        grid.addWidget(btn_eego, row, 2)
        row += 1

        # ===== Marker =====
        self.lbl_marker = QLabel("●"); self._set_led(self.lbl_marker, "stopped")
        grid.addWidget(QLabel("<b>Marker Stream</b>"), row, 0)
        grid.addWidget(self.lbl_marker, row, 1)
        self.btn_marker_start = QPushButton("Start")
        self.btn_marker_stop  = QPushButton("Stop")
        self.btn_marker_refresh = QPushButton("Refresh")
        self.btn_marker_start.clicked.connect(self.on_marker_start)
        self.btn_marker_stop.clicked.connect(self.on_marker_stop)
        self.btn_marker_refresh.clicked.connect(self.on_marker_refresh)
        grid.addWidget(self.btn_marker_start, row, 2)
        grid.addWidget(self.btn_marker_stop, row, 3)
        grid.addWidget(self.btn_marker_refresh, row, 4)
        row += 1

        # ===== FES =====
        self.lbl_fes = QLabel("●"); self._set_led(self.lbl_fes, "stopped")
        grid.addWidget(QLabel("<b>FES Listener</b>"), row, 0)
        grid.addWidget(self.lbl_fes, row, 1)
        self.btn_fes_start = QPushButton("Start")
        self.btn_fes_stop  = QPushButton("Stop")
        self.btn_fes_refresh = QPushButton("Refresh")
        self.btn_fes_start.clicked.connect(self.on_fes_start)
        self.btn_fes_stop.clicked.connect(self.on_fes_stop)
        self.btn_fes_refresh.clicked.connect(self.on_fes_refresh)
        grid.addWidget(self.btn_fes_start, row, 2)
        grid.addWidget(self.btn_fes_stop, row, 3)
        grid.addWidget(self.btn_fes_refresh, row, 4)
        row += 1

        # ===== LabRecorder =====
        self.lbl_labrec = QLabel("●"); self._set_led(self.lbl_labrec, "stopped")
        grid.addWidget(QLabel("<b>LabRecorder</b>"), row, 0)
        grid.addWidget(self.lbl_labrec, row, 1)
        btn_labrec = QPushButton("Open LabRecorder")
        btn_labrec.clicked.connect(self.on_open_labrec)
        grid.addWidget(btn_labrec, row, 2)
        row += 1

        # ===== Gaze Service =====
        # Collected in self._gaze_row_widgets so _apply_backend_visibility()
        # can hide the whole block when GAZE_OR_BACKEND == "vlm". An empty
        # grid row collapses to zero height in Qt once all its items are
        # hidden, so toggling visibility is sufficient.
        self.lbl_gaze_service = QLabel("●"); self._set_led(self.lbl_gaze_service, "stopped")
        gaze_lbl_title = QLabel("<b>Gaze Service</b>")
        grid.addWidget(gaze_lbl_title, row, 0)
        grid.addWidget(self.lbl_gaze_service, row, 1)

        self.btn_gaze_service_headless = QPushButton("Start (Headless)")
        self.btn_gaze_service_ui = QPushButton("Start (With UI)")
        self.btn_gaze_service_stop = QPushButton("Stop")
        self.btn_gaze_service_query = QPushButton("Query Telemetry (UDP)")

        self.btn_gaze_service_headless.clicked.connect(self.on_gaze_service_start_headless)
        self.btn_gaze_service_ui.clicked.connect(self.on_gaze_service_start_ui)
        self.btn_gaze_service_stop.clicked.connect(self.on_gaze_service_stop)
        self.btn_gaze_service_query.clicked.connect(self.on_gaze_service_query)

        grid.addWidget(self.btn_gaze_service_headless, row, 2)
        grid.addWidget(self.btn_gaze_service_ui, row, 3)
        grid.addWidget(self.btn_gaze_service_stop, row, 4)
        row += 1

        gaze_telemetry_lbl = QLabel("<i>Telemetry:</i> view output in View: Gaze")
        grid.addWidget(gaze_telemetry_lbl, row, 0, 1, 2)
        grid.addWidget(self.btn_gaze_service_query, row, 2, 1, 3)
        row += 1

        self._gaze_row_widgets = [
            gaze_lbl_title, self.lbl_gaze_service,
            self.btn_gaze_service_headless, self.btn_gaze_service_ui,
            self.btn_gaze_service_stop,
            gaze_telemetry_lbl, self.btn_gaze_service_query,
        ]

        # ===== Perception Pipeline =====
        # Three pipeline stages, each with its own LED in the LED column:
        #   Send    — local frame_relay (Utils/frame_relay.py), opens Neon
        #             and ships TCP envelopes. Driven by _poll_relay_status.
        #   Compute — remote vlm_service.py (or local QProcess in single-
        #             machine mode). Driven by _apply_remote_status when
        #             SERVICES_HOSTED_REMOTELY=True, by _start_proc/QProcess
        #             state in local mode.
        #   Receive — local Utils/vlm_subscriber.py (instantiated inside
        #             VLMSceneWidget). Driven by the subscriber's
        #             state_changed signal, bubbled up via the widget's
        #             subscriber_state_changed signal.
        # Layout matches the Robot row's inline style — row 1: title +
        # 3 LEDs + lifecycle/runtime buttons; row 2: VLM-specific
        # commands (no separate "Continuous / Pair" row, those merge in).
        # Per-stage detail text lives on each LED's tooltip rather than
        # a third row, keeping the block compact while preserving the
        # diagnostic information for hover.
        self.lbl_send_led    = QLabel("●"); self._set_led(self.lbl_send_led,    "stopped")
        self.lbl_compute_led = QLabel("●"); self._set_led(self.lbl_compute_led, "stopped")
        self.lbl_receive_led = QLabel("●"); self._set_led(self.lbl_receive_led, "stopped")
        # Initial tooltip text — updated in place by the same drivers
        # (_apply_remote_status, _poll_relay_status, _on_subscriber_state)
        # that set the LED colour.
        self.lbl_send_led.setToolTip("send: idle")
        self.lbl_compute_led.setToolTip("compute: --")
        self.lbl_receive_led.setToolTip("receive: --")

        leds_box = QHBoxLayout()
        leds_box.setContentsMargins(0, 0, 0, 0)
        leds_box.setSpacing(2)
        for led in (self.lbl_send_led, self.lbl_compute_led, self.lbl_receive_led):
            leds_box.addWidget(led)
        leds_holder = _fixed_v(QWidget()); leds_holder.setLayout(leds_box)

        # Lifecycle + runtime actions. btn_vlm_service_start/stop are
        # constructed here but rebranded to "Connect"/"Disconnect" in
        # remote mode by _configure_remote_services_ui (which also
        # rewires their handlers to _on_vlm_video_connect/disconnect).
        self.btn_vlm_service_start  = QPushButton("Start")
        self.btn_vlm_service_stop   = QPushButton("Stop")
        self.btn_vlm_service_status = QPushButton("Status")
        self.btn_vlm_service_decide = QPushButton("Decide Now")
        self.btn_vlm_service_depth  = QPushButton("Depth Now")
        self.btn_vlm_service_start.clicked.connect(self.on_vlm_service_start)
        self.btn_vlm_service_stop.clicked.connect(self.on_vlm_service_stop)
        self.btn_vlm_service_status.clicked.connect(self.on_vlm_service_status)
        self.btn_vlm_service_decide.clicked.connect(self.on_vlm_service_decide)
        self.btn_vlm_service_depth.clicked.connect(self.on_vlm_service_depth)

        # Continuous segmentation toggle + cadence — merged onto the
        # main perception row instead of a separate "Continuous / Pair"
        # row, since stream control belongs alongside lifecycle in the
        # operator's mental model.
        self.btn_vlm_seg_stream = QPushButton("Stream Seg: OFF")
        self.btn_vlm_seg_stream.setCheckable(True)
        self.spin_vlm_seg_hz = QDoubleSpinBox()
        self.spin_vlm_seg_hz.setRange(1.0, 30.0)
        self.spin_vlm_seg_hz.setSingleStep(1.0)
        self.spin_vlm_seg_hz.setDecimals(1)
        self.spin_vlm_seg_hz.setValue(10.0)
        self.spin_vlm_seg_hz.setSuffix(" Hz")
        self.btn_vlm_seg_stream.toggled.connect(self.on_vlm_seg_stream_toggled)
        self.spin_vlm_seg_hz.valueChanged.connect(self.on_vlm_seg_stream_hz_changed)

        # Sequential (two-object) decide controls.
        self.btn_vlm_capture_first = QPushButton("Capture First")
        self.btn_vlm_decide_pair = QPushButton("Decide Pair")
        self.lbl_vlm_pair_token = QLabel("<i>snapshot:</i> (none)")
        self.btn_vlm_capture_first.clicked.connect(self.on_vlm_capture_first)
        self.btn_vlm_decide_pair.clicked.connect(self.on_vlm_decide_pair)

        # Row 1: lifecycle (Connect/Disconnect) + stream-seg toggle + cadence.
        # Each widget gets stretch=1 with no trailing addStretch so the
        # row fills the col 2-4 span (matches the right edge set by
        # Robot's "Remove Overrides" / Marker's "Refresh") instead of
        # clustering on the left.
        actions_row1 = QHBoxLayout()
        actions_row1.setContentsMargins(0, 0, 0, 0)
        for w in (self.btn_vlm_service_start, self.btn_vlm_service_stop,
                  self.btn_vlm_seg_stream, self.spin_vlm_seg_hz):
            actions_row1.addWidget(w, 1)
        actions_row1_holder = _fixed_v(QWidget()); actions_row1_holder.setLayout(actions_row1)

        # Row 2: ad-hoc commands (status, decide-once, depth-once,
        # sequential-pair). Same col span and stretch policy as row 1
        # so Status sits directly under Connect.
        actions_row2 = QHBoxLayout()
        actions_row2.setContentsMargins(0, 0, 0, 0)
        for w in (self.btn_vlm_service_status, self.btn_vlm_service_decide,
                  self.btn_vlm_service_depth, self.btn_vlm_capture_first,
                  self.btn_vlm_decide_pair, self.lbl_vlm_pair_token):
            actions_row2.addWidget(w, 1)
        actions_row2_holder = _fixed_v(QWidget()); actions_row2_holder.setLayout(actions_row2)

        # Title carries the legend so dots-only LED column matches the
        # Robot row's style — see GPU_Service_Cross_Host_Hardening_Notes
        # for the design rationale.
        pipeline_title = QLabel(
            "<b>Perception Pipeline</b><br>"
            "<i>(send / compute / receive)</i>"
        )
        grid.addWidget(pipeline_title,      row, 0)
        grid.addWidget(leds_holder,         row, 1)
        grid.addWidget(actions_row1_holder, row, 2, 1, 3)
        row += 1
        grid.addWidget(actions_row2_holder, row, 2, 1, 3)
        row += 1

        # Aggregate VLM-specific widgets so backend gating (legacy mode)
        # can hide them in one shot. The pipeline-block title and the
        # Send LED stay visible because the frame_relay is shared infra
        # (gaze_runner consumes it too); only Compute/Receive LEDs and
        # the VLM-only commands are hidden.
        self._vlm_row_widgets = [
            self.lbl_compute_led, self.lbl_receive_led,
            self.btn_vlm_seg_stream, self.spin_vlm_seg_hz,
            self.btn_vlm_service_status,
            self.btn_vlm_service_decide,
            self.btn_vlm_service_depth,
            self.btn_vlm_capture_first,
            self.btn_vlm_decide_pair,
            self.lbl_vlm_pair_token,
        ]

        # ===== Arduino =====
        # Single-line layout matching the other module rows. Baud lives in
        # the Runtime config tab (rarely changed); per-test status updates
        # land in the Panel log buffer rather than a dedicated label, and
        # the LED reflects the last connection-test / send result.
        self.lbl_arduino = QLabel("●"); self._set_led(self.lbl_arduino, "stopped")
        self.cmb_serial_port = QComboBox()
        self.cmb_serial_port.currentIndexChanged.connect(self.on_serial_port_changed)
        self.btn_serial_refresh = QPushButton("Refresh")
        self.btn_serial_refresh.clicked.connect(self.on_serial_refresh)
        self.btn_serial_test = QPushButton("Test")
        self.btn_serial_test.clicked.connect(self.on_serial_test)
        self.btn_save_serial_to_config = QPushButton("Save → config")
        self.btn_save_serial_to_config.setToolTip(
            "Writes ARDUINO_PORT to config_local.py (machine-local) and "
            "ARDUINO_BAUD to config.py."
        )
        self.btn_save_serial_to_config.clicked.connect(self.on_save_serial_to_config)
        self.btn_send_1 = QPushButton("Send 1 (close)")
        self.btn_send_1.clicked.connect(self.on_send_arduino_one)
        self.btn_send_0 = QPushButton("Send 0 (open)")
        self.btn_send_0.clicked.connect(self.on_send_arduino_zero)

        arduino_row = QHBoxLayout()
        arduino_row.setContentsMargins(0, 0, 0, 0)
        arduino_row.addWidget(self.cmb_serial_port, 1)
        for w in (self.btn_serial_refresh, self.btn_serial_test,
                  self.btn_save_serial_to_config,
                  self.btn_send_1, self.btn_send_0):
            arduino_row.addWidget(w)
        arduino_row_holder = _fixed_v(QWidget()); arduino_row_holder.setLayout(arduino_row)
        grid.addWidget(QLabel("<b>Arduino</b>"), row, 0)
        grid.addWidget(self.lbl_arduino, row, 1)
        grid.addWidget(arduino_row_holder, row, 2, 1, 3)
        row += 1

        # ===== Driver =====
        # Anchored at the bottom — starting the experiment driver is the
        # last step before a session begins, so keeping it visually
        # separate from device setup matches the operator's flow.
        self.lbl_driver = QLabel("●"); self._set_led(self.lbl_driver, "stopped")
        grid.addWidget(QLabel("<b>Experiment Driver</b>"), row, 0)
        grid.addWidget(self.lbl_driver, row, 1)
        self.btn_driver_start = QPushButton("Start")
        self.btn_driver_stop  = QPushButton("Stop")
        self.btn_driver_start.clicked.connect(self.on_driver_start)
        self.btn_driver_stop.clicked.connect(self.on_driver_stop)
        grid.addWidget(self.btn_driver_start, row, 2)
        grid.addWidget(self.btn_driver_stop, row, 3)
        row += 1

        # Bottom stretch: absorbs any leftover vertical space in the
        # controls panel so the data rows above stay packed at the
        # natural row pitch instead of distributing slack between them.
        grid.setRowStretch(row, 1)

        # ===== Logs Pane =====
        logw = QWidget(); split.addWidget(logw)
        vl = QVBoxLayout(logw)

        pick_row = QHBoxLayout()
        self.log_title = QLabel("Logs:")
        self.log_selector = QComboBox()
        self.log_selector.addItems(["Marker", "FES", "Driver", "Gaze", "VLM", "Relay", "Robot", "Panel"])
        self.log_selector.setCurrentText(self._current_log_target)
        self.log_selector.currentTextChanged.connect(self._on_log_target_changed)
        pick_row.addWidget(self.log_title); pick_row.addStretch(1)
        pick_row.addWidget(QLabel("View:")); pick_row.addWidget(self.log_selector)

        self.txt_logs = QTextEdit()
        self.txt_logs.setReadOnly(True)
        self.txt_logs.setLineWrapMode(QTextEdit.NoWrap)

        vl.addLayout(pick_row)
        vl.addWidget(self.txt_logs, 1)

        robot_tab = QWidget()
        tabs.addTab(robot_tab, "Robot Test")
        rt = QVBoxLayout(robot_tab)

        udp_row = QHBoxLayout()
        btn_open_udp_robot = QPushButton("Open UDPRobot.py (terminal)")
        btn_open_udp_robot.setMaximumWidth(280)
        btn_open_udp_robot.clicked.connect(
            lambda: self._spawn_external(f'python -u "{os.path.join(ROOT, "UDPRobot.py")}"')
        )
        udp_row.addWidget(btn_open_udp_robot)
        udp_row.addStretch(1)
        rt.addLayout(udp_row)

        harmony_box = QGroupBox("Harmony calibration / online control")
        hb = QVBoxLayout(harmony_box)
        lib_row = QHBoxLayout()
        lib_row.addWidget(QLabel("Calibration library:"))
        self.cmb_calibration_lib = QComboBox()
        lib_row.addWidget(self.cmb_calibration_lib, 1)
        self.btn_refresh_calibration_libs = QPushButton("Refresh")
        self.btn_refresh_calibration_libs.setMaximumWidth(90)
        self.btn_refresh_calibration_libs.clicked.connect(self.on_refresh_calibration_libs)
        lib_row.addWidget(self.btn_refresh_calibration_libs)
        hb.addLayout(lib_row)

        hbtn_row = QHBoxLayout()
        self.btn_run_harmony_calibration = QPushButton("Run harmony_calibration_exec.py")
        self.btn_run_harmony_calibration.setMaximumWidth(260)
        self.btn_run_harmony_calibration.clicked.connect(self.on_run_harmony_calibration)
        self.btn_run_harmony_online = QPushButton("Run harmony_online_control.py")
        self.btn_run_harmony_online.setMaximumWidth(240)
        self.btn_run_harmony_online.clicked.connect(self.on_run_harmony_online_control)
        hbtn_row.addWidget(self.btn_run_harmony_calibration)
        hbtn_row.addWidget(self.btn_run_harmony_online)
        hbtn_row.addStretch(1)
        hb.addLayout(hbtn_row)
        rt.addWidget(harmony_box)

        train_box = QGroupBox("Model training (uses config.py DATA_DIR + subject below)")
        tv = QVBoxLayout(train_box)
        self.lbl_training_subject_ctx = QLabel("")
        self.lbl_training_subject_ctx.setWordWrap(True)
        tv.addWidget(self.lbl_training_subject_ctx)
        trow = QHBoxLayout()
        trow.addWidget(QLabel("Script:"))
        self.cmb_train_script = QComboBox()
        trow.addWidget(self.cmb_train_script, 1)
        self.btn_refresh_training_data = QPushButton("Refresh data list")
        self.btn_refresh_training_data.setMaximumWidth(130)
        self.btn_refresh_training_data.clicked.connect(self.on_refresh_training_data_list)
        trow.addWidget(self.btn_refresh_training_data)
        tv.addLayout(trow)
        self.lst_training_files = QListWidget()
        self.lst_training_files.setMaximumHeight(140)
        tv.addWidget(self.lst_training_files)
        self.lbl_train_cmd_preview = QLabel("")
        self.lbl_train_cmd_preview.setWordWrap(True)
        self.lbl_train_cmd_preview.setStyleSheet("color: #666; font-family: monospace;")
        tv.addWidget(self.lbl_train_cmd_preview)
        train_btn_row = QHBoxLayout()
        self.btn_launch_training = QPushButton("Launch training (terminal)")
        self.btn_launch_training.clicked.connect(self.on_launch_model_training)
        self.btn_launch_training.setMaximumWidth(220)
        train_btn_row.addWidget(self.btn_launch_training)
        train_btn_row.addStretch(1)
        tv.addLayout(train_btn_row)
        rt.addWidget(train_box)

        self.txt_udp_log = QTextEdit()
        self.txt_udp_log.setReadOnly(True)
        self.txt_udp_log.setMaximumHeight(140)
        rt.addWidget(QLabel("Notes:"))
        rt.addWidget(self.txt_udp_log)

        self._populate_training_script_combo()
        self._build_vlm_video_tab(tabs)
        self._build_runtime_config_tab(tabs)
        self._build_errp_config_tab(tabs)

        # Initial serial refresh
        self.on_serial_refresh()
        self.on_refresh_calibration_libs()
        self.on_refresh_training_data_list()

        self._building_ui = False
        self._refresh_log_view()

        self._update_robot_buttons_for_mode()
        self._apply_backend_visibility()

    def _apply_backend_visibility(self) -> None:
        """Hide rows that are inert for the current GAZE_OR_BACKEND.

        legacy → vlm_service is not running, so the VLM rows + remote-intake
        badge would only display dead controls. vlm → gaze_runner is not
        running, so the Gaze service rows have nothing to drive. Frame Relay
        is shared by both backends in remote mode and is gated separately
        by the perception-source flag, not by this method."""
        is_vlm = (GAZE_OR_BACKEND == "vlm")
        for w in getattr(self, "_gaze_row_widgets", ()):
            w.setVisible(not is_vlm)
        for w in getattr(self, "_vlm_row_widgets", ()):
            w.setVisible(is_vlm)

    def _build_vlm_video_tab(self, tabs: QTabWidget) -> None:
        """Linux-side scene + JSON-overlay renderer
        (Render_Layer_Refactor.md §4). Bundles from a panel-hosted
        FrameRelayServer (or an externally-supplied one when
        FRAME_RELAY_EMBEDDED=False), detection JSON from vlm_service.py
        UDP 5589 subscribe, gaze tracks from gaze_runner.py UDP 5588,
        composited at native frame rate via Utils.scene_overlay_renderer.

        Windows TCP clients still dial the embedded relay's listening
        socket — the wire is unchanged.

        When FRAME_RELAY_EMBEDDED is True (the default), the user must
        NOT also run `python -m Utils.frame_relay` separately on this
        machine: two NeonLiveReaders conflict at the SDK level.
        """
        vvt = QWidget()
        tabs.addTab(vvt, "VLM Video")
        vl = QVBoxLayout(vvt)

        from Utils.vlm_scene_widget import VLMSceneWidget
        embedded_relay = None
        if FRAME_RELAY_EMBEDDED:
            embedded_relay = {
                "bind_host": FRAME_RELAY_BIND_HOST,
                "bind_port": FRAME_RELAY_PORT,
                "hz": FRAME_RELAY_HZ,
                "neon_host": NEON_COMPANION_HOST,
            }
        # GAZE_OR_BACKEND selects which perception service the panel
        # subscribes to (same semantic as ExperimentDriver_Online_GazeTracking
        # uses to pick the active backend):
        #   "vlm"    → only subscribe to vlm_service.py (UDP 5589). The
        #              gaze_runner channel is left dark.
        #   "legacy" → only subscribe to gaze_runner.py (UDP 5588). The
        #              VLM channel is left dark.
        # Either subscriber's _JsonPushSubscriber heartbeats every ~10 s,
        # so a service that goes down then comes back is reconnected
        # automatically with no panel restart.
        if GAZE_OR_BACKEND == "vlm":
            vlm_host_arg, vlm_port_arg = VLM_SERVICE_HOST, VLM_SERVICE_PORT
            gaze_host_arg, gaze_port_arg = None, None
        else:
            vlm_host_arg, vlm_port_arg = None, None
            gaze_host_arg, gaze_port_arg = GAZE_SERVICE_HOST, GAZE_SERVICE_PORT
        self.vlm_scene_widget = VLMSceneWidget(
            vlm_host=vlm_host_arg,
            vlm_port=vlm_port_arg,
            gaze_host=gaze_host_arg,
            gaze_port=gaze_port_arg,
            relay_dial_host=FRAME_RELAY_DIAL_HOST,
            relay_dial_port=FRAME_RELAY_PORT,
            embedded_relay=embedded_relay,
        )
        # Drive the Main-tab Receive LED from whichever JsonPushSubscriber
        # the widget instantiates (vlm or gaze, by GAZE_OR_BACKEND). The
        # widget re-emits its inner subscribers' state on a single
        # bubbled signal so the panel doesn't have to know which one is
        # active.
        self.vlm_scene_widget.subscriber_state_changed.connect(
            self._on_subscriber_state
        )
        # Send LED gates on the relay's TCP handshake to its first
        # consumer — the relay has demonstrated "send is good" the
        # moment _install_client successfully delivers the handshake
        # envelope (Utils/frame_relay.py:_install_client). This is
        # independent of Pupil Labs SDK first-frame latency.
        self.vlm_scene_widget.handshake_observed.connect(
            self._on_handshake_observed
        )
        # First-publish event is repurposed: it now triggers the
        # verify_chain firing path once the GPU has confirmed a
        # fresh bundle (proving end-to-end traversal of the pipeline
        # for THIS Connect). It no longer flips the Send LED.
        self.vlm_scene_widget.first_publish_observed.connect(
            self._on_first_publish_observed
        )
        # Subscriber payloads are token-checked here so a stale
        # `chain_verify` from a prior Connect's GPU cache cannot
        # trip the Receive LED on a fresh Connect.
        self.vlm_scene_widget.vlm_payload_received.connect(
            self._on_vlm_payload_received
        )
        vl.addWidget(self.vlm_scene_widget, 1)

    def _on_vlm_video_connect(self) -> None:
        """Connect button handler. Drives the per-Connect verification
        state machine described in __init__:
          1. Generate a fresh ``_connect_token`` (time.time_ns()).
          2. Paint Send / Compute / Receive yellow ("starting") so the
             operator sees the verification is in progress.
          3. Reset the observed-flags + verify_chain attempt counter so
             a stale push from a prior session can't satisfy this
             cycle's verification.
          4. Start the widget (relay + subscriber threads), kick an
             immediate cmd=status preflight so Compute responds within
             one UDP RTT rather than waiting up to 1 s for the next
             timer tick.

        After this call returns, the state machine advances on three
        Qt-signal-driven events:
          - handshake_observed → Send green (control_panel.py:_on_handshake_observed)
          - status reply ok=True (with _send_observed) → Compute green
          - first_publish_observed + GPU has fresh bundle → fire
            cmd=verify_chain {token}; matching push → Receive green
        """
        self._connect_token = time.time_ns()
        self._send_observed = False
        self._compute_observed = False
        self._receive_observed = False
        self._verify_chain_in_flight = False
        self._verify_chain_attempts = 0
        self._set_led(self.lbl_send_led, "starting")
        self.lbl_send_led.setToolTip("send: verifying — awaiting handshake")
        self._set_led(self.lbl_compute_led, "starting")
        self.lbl_compute_led.setToolTip("compute: verifying — awaiting GPU status reply")
        self._set_led(self.lbl_receive_led, "starting")
        self.lbl_receive_led.setToolTip(
            "receive: verifying — awaiting end-to-end chain_verify response"
        )
        self._append_log(
            "VLM",
            f"[{self._ts()}] chain: connect armed token={self._connect_token}\n",
        )
        if hasattr(self, "vlm_scene_widget"):
            self.vlm_scene_widget.start()
        if getattr(self, "_relay_status_timer", None) is not None:
            self._poll_relay_status()
        if SERVICES_HOSTED_REMOTELY and getattr(self, "_remote_status_timer", None) is not None:
            self._poll_remote_status()

    def _on_vlm_video_disconnect(self) -> None:
        token = self._connect_token
        self._append_log("VLM", f"[{self._ts()}] chain: disconnect token={token}\n")
        if hasattr(self, "vlm_scene_widget"):
            self.vlm_scene_widget.stop()
        # Tear down state-machine state so a subsequent Connect starts
        # fresh. _connect_token=None disqualifies any late-arriving
        # chain_verify push from the prior session even if it slips
        # through the subscriber teardown race.
        self._connect_token = None
        self._send_observed = False
        self._compute_observed = False
        self._receive_observed = False
        self._verify_chain_in_flight = False
        self._verify_chain_attempts = 0
        # Reset all three LEDs to gray explicitly. _poll_relay_status
        # gates on _send_observed and won't repaint Send after we
        # cleared the flag above (control_panel.py:_poll_relay_status),
        # so without this Send would remain stuck on its last green
        # state. Doing the same for Compute and Receive keeps the
        # idle-state appearance consistent across all three rather
        # than relying on side-channels (next status poll for Compute,
        # the "unsubscribed" handler for Receive).
        self._set_led(self.lbl_send_led, "stopped")
        self.lbl_send_led.setToolTip("send: idle")
        self._set_led(self.lbl_compute_led, "stopped")
        self.lbl_compute_led.setToolTip("compute: idle")
        self._set_led(self.lbl_receive_led, "stopped")
        self.lbl_receive_led.setToolTip("receive: idle")

    def _on_handshake_observed(self, addr) -> None:
        """Slot for VLMSceneWidget.handshake_observed. The relay has
        successfully delivered its handshake envelope to a TCP
        consumer, so the Send utility is provably operational —
        independent of how long the SDK takes to deliver the first
        scene frame. Flip Send green and kick the status RPC so
        Compute can follow within one UDP roundtrip.
        """
        try:
            host, port = (addr[0], int(addr[1]))
        except (TypeError, ValueError, IndexError):
            host, port = "?", 0
        self._send_observed = True
        self._set_led(self.lbl_send_led, "running")
        self.lbl_send_led.setToolTip(
            f"send: handshake delivered to {host}:{port}"
        )
        self._append_log(
            "VLM",
            f"[{self._ts()}] chain: send handshake to {host}:{port}\n",
        )
        if SERVICES_HOSTED_REMOTELY and getattr(self, "_remote_status_timer", None) is not None:
            self._poll_remote_status()

    def _on_first_publish_observed(self, addr) -> None:
        """Slot for VLMSceneWidget.first_publish_observed. The relay
        has delivered the FIRST real frame envelope to its consumer,
        so we know the TCP send path is exercised end-to-end with
        actual data this Connect. Kick a status poll so Compute can
        confirm a fresh bundle on the GPU side; the verify_chain
        firing path runs from _apply_remote_status once that fresh
        bundle is observed.

        The Send LED itself was already flipped on the earlier
        handshake event (see _on_handshake_observed); this slot does
        not touch it.
        """
        try:
            host, port = (addr[0], int(addr[1]))
        except (TypeError, ValueError, IndexError):
            host, port = "?", 0
        self._append_log(
            "VLM",
            f"[{self._ts()}] chain: first frame published to {host}:{port}\n",
        )
        if SERVICES_HOSTED_REMOTELY and getattr(self, "_remote_status_timer", None) is not None:
            self._poll_remote_status()

    def _on_vlm_payload_received(self, payload: dict) -> None:
        """Slot for VLMSceneWidget.vlm_payload_received. Token-checks
        chain_verify responses against the current Connect's token to
        flip Receive green ONLY for this cycle's verification round-
        trip. Stale pushes from the GPU's prior-session cache (which
        survive across reconnects per vlm_service.py:344-353) carry
        either no token or a stale one and are ignored here for LED
        purposes — they still hit the video-tab render pipeline via
        the widget's own _on_vlm_payload handler.
        """
        if not isinstance(payload, dict):
            return
        if payload.get("type") != "chain_verify":
            return
        token = payload.get("token")
        current = self._connect_token
        if current is None or token != current:
            return
        if not self._compute_observed:
            # Out-of-order: chain_verify came back before the panel
            # observed Compute green. Defer the Receive flip — the
            # next status reply will catch up and we'll re-evaluate.
            # In practice this is a vanishingly small race window,
            # but we'd rather hold than violate the order invariant.
            return
        self._receive_observed = True
        self._set_led(self.lbl_receive_led, "running")
        self.lbl_receive_led.setToolTip(
            f"receive: end-to-end verified (token={token})"
        )
        self._append_log(
            "VLM",
            f"[{self._ts()}] chain: receive verified token={token}\n",
        )

    def _on_subscriber_state(self, state: str) -> None:
        """Slot for VLMSceneWidget.subscriber_state_changed. Under
        the new verification model, the Receive LED's green flip is
        owned by _on_vlm_payload_received (token-matched chain_verify),
        so this handler only manages the non-running cases:
          - "subscribed"     → keep yellow (verification in progress)
          - "receiving:<t>"  → no LED change here; token check decides
          - "error: …"       → red
          - "unsubscribed"   → gray
        """
        if state.startswith("receiving"):
            return
        if state == "subscribed":
            # Don't downgrade from a previously-green Receive (could
            # happen if the subscriber thread races a token-matched
            # push). Otherwise hold yellow during verification.
            return
        if state.startswith("error"):
            self._set_led(self.lbl_receive_led, "error")
            self.lbl_receive_led.setToolTip(f"receive: {state}")
            return
        # "unsubscribed" or anything we don't recognise — gray.
        self._set_led(self.lbl_receive_led, "stopped")
        self.lbl_receive_led.setToolTip(f"receive: {state}")

    def _fire_verify_chain(self) -> None:
        """Send `cmd=verify_chain {token}` to the GPU on a worker
        thread. The token-matched chain_verify push back to the
        subscriber is what flips Receive green via
        _on_vlm_payload_received; this method only initiates the
        round-trip and handles the failure paths.

        Retries once after 200 ms on `no_frame` (a short race window
        where the relay's first frame is still in flight to the GPU
        when verify_chain arrives). Beyond that, paints Receive red.

        Schedules a 5 s deadline check for the case where the GPU
        replies ok=True with subscribers_notified=0 (subscribe-RPC
        race) or where the chain_verify push is lost in transit. If
        no token-matched push has landed by the deadline, Receive
        goes red.
        """
        if self._connect_token is None:
            return
        self._verify_chain_in_flight = True
        self._verify_chain_attempts += 1
        token = self._connect_token
        attempt = self._verify_chain_attempts
        self._append_log(
            "VLM",
            f"[{self._ts()}] chain: verify_chain TX token={token} attempt={attempt}\n",
        )
        # Receive deadline: per-token guard so an old timer can't
        # nuke a current-Connect verification.
        QTimer.singleShot(
            5000, self,
            lambda t=token: self._verify_chain_deadline_check(t),
        )

        def worker():
            try:
                resp = self._vlm_udp_request(
                    {"cmd": "verify_chain", "token": token},
                    timeout_s=2.0,
                )
            except Exception as e:
                resp = {"ok": False, "error": f"udp_exception: {e}"}
            try:
                QTimer.singleShot(
                    0, self,
                    lambda r=resp, t=token: self._on_verify_chain_reply(r, t),
                )
            except RuntimeError:
                # Window closed mid-RPC.
                pass

        threading.Thread(
            target=worker, daemon=True, name="panel-verify-chain"
        ).start()

    def _on_verify_chain_reply(self, resp: dict, token: int) -> None:
        """GUI-thread slot for verify_chain RPC reply. ok=True here
        means "GPU dispatched the synthetic push" — the actual Receive
        LED flip waits for the push to land at the subscriber and
        pass the token check (_on_vlm_payload_received). This handler
        only covers failure cases.
        """
        self._verify_chain_in_flight = False
        if token != self._connect_token:
            # Connect changed under us; the response belongs to a
            # prior cycle. Drop it.
            return
        if resp.get("ok"):
            return  # Success path — push will land on subscriber.
        err = resp.get("error", "unknown")
        if err == "no_frame" and self._verify_chain_attempts < 2:
            QTimer.singleShot(200, self, self._fire_verify_chain)
            return
        self._set_led(self.lbl_receive_led, "error")
        self.lbl_receive_led.setToolTip(
            f"receive: verify_chain RPC failed: {err}"
        )
        self._append_log(
            "VLM",
            f"[{self._ts()}] chain: verify_chain FAILED token={token} error={err}\n",
        )

    def _verify_chain_deadline_check(self, token: int) -> None:
        """Fired 5 s after _fire_verify_chain. If no token-matched
        chain_verify push has landed by now, paint Receive red.
        Token-discriminated so a stale older-Connect timer doesn't
        revert a successful current verification."""
        if token != self._connect_token:
            return
        if self._receive_observed:
            return
        self._set_led(self.lbl_receive_led, "error")
        self.lbl_receive_led.setToolTip(
            "receive: verify_chain push did not arrive (timeout)"
        )
        self._append_log(
            "VLM",
            f"[{self._ts()}] chain: verify_chain TIMEOUT token={token}\n",
        )

    def _build_runtime_config_tab(self, tabs: QTabWidget):
        rtc = QWidget()
        tabs.addTab(rtc, "Runtime config")
        outer = QVBoxLayout(rtc)
        # setWordWrap on this label is load-bearing: without it, the
        # full unwrapped text width becomes the tab's minimumSizeHint
        # (~1900 px) and propagates to the whole window's minimum size.
        intro = QLabel(
            "<b>Edits config.py / config_local.py on disk.</b> "
            "Machine-local keys (PERCEPTION_FRAME_SOURCE, "
            "SERVICES_HOSTED_REMOTELY) write to config_local.py; "
            "everything else writes to config.py. Restart "
            "Marker/Driver/FES after changing simulation or network "
            "flags (<code>Utils/networking</code> caches SIMULATION_MODE "
            "at import)."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        form = QFormLayout(inner)
        self.rc_decoder = QComboBox()
        self.rc_decoder.addItems(["mdm", "xgb_cov", "xgb_cov_erd"])
        self.rc_earlystop = QComboBox()
        self.rc_earlystop.addItems(["either", "correct_only"])
        self.rc_visual = QComboBox()
        self.rc_visual.addItems(["classic", "modern"])
        self.rc_th_mi = QDoubleSpinBox()
        self.rc_th_mi.setRange(0.0, 1.0)
        self.rc_th_mi.setSingleStep(0.01)
        self.rc_th_mi.setDecimals(3)
        self.rc_th_rest = QDoubleSpinBox()
        self.rc_th_rest.setRange(0.0, 1.0)
        self.rc_th_rest.setSingleStep(0.01)
        self.rc_th_rest.setDecimals(3)
        self.rc_int_alpha = QDoubleSpinBox()
        self.rc_int_alpha.setRange(0.0, 1.0)
        self.rc_int_alpha.setSingleStep(0.01)
        self.rc_int_alpha.setDecimals(3)
        self.rc_classify_ms = QSpinBox()
        self.rc_classify_ms.setRange(100, 8000)
        self.rc_min_pred = QSpinBox()
        self.rc_min_pred.setRange(1, 500)
        self.rc_time_mi = QSpinBox()
        self.rc_time_mi.setRange(1, 300)
        self.rc_time_rob = QSpinBox()
        self.rc_time_rob.setRange(1, 300)
        self.rc_big_brother = QCheckBox("BIG_BROTHER_MODE (second display layout)")
        self.rc_send_probs = QCheckBox("SEND_PROBS (extra UDP probability traffic)")
        self.rc_save_adaptive_t = QCheckBox("SAVE_ADAPTIVE_T (persist adaptive μ/β transform across sessions)")
        self.rc_recentering = QCheckBox("RECENTERING")
        self.rc_update_move = QCheckBox("UPDATE_DURING_MOVE")
        self.rc_laplacian = QCheckBox("SURFACE_LAPLACIAN_TOGGLE")
        self.rc_sel_motor = QCheckBox("SELECT_MOTOR_CHANNELS")
        self.rc_xgb_beta = QCheckBox("XGB_USE_COV_BETA (enable beta-band covariance features)")
        self.rc_total_trials = QSpinBox()
        self.rc_total_trials.setRange(1, 500)
        self.rc_shape_max = QDoubleSpinBox()
        self.rc_shape_max.setRange(0.0, 1.0)
        self.rc_shape_max.setSingleStep(0.01)
        self.rc_shape_max.setDecimals(2)
        self.rc_shape_min = QDoubleSpinBox()
        self.rc_shape_min.setRange(0.0, 1.0)
        self.rc_shape_min.setSingleStep(0.01)
        self.rc_shape_min.setDecimals(2)

        # Perception / VLM section (mix of global + machine-local keys —
        # the writer routes by key name).
        self.rc_gaze_backend = QComboBox()
        self.rc_gaze_backend.addItems(["legacy", "vlm"])
        self.rc_perception_source = QComboBox()
        self.rc_perception_source.addItems(["local", "remote"])
        self.rc_services_remote = QCheckBox("SERVICES_HOSTED_REMOTELY (panel → remote GPU host; machine-local)")
        self.rc_relay_hz = QDoubleSpinBox()
        self.rc_relay_hz.setRange(1.0, 30.0)
        self.rc_relay_hz.setSingleStep(1.0)
        self.rc_relay_hz.setDecimals(1)
        self.rc_relay_hz.setSuffix(" Hz")
        self.rc_vlm_depth = QCheckBox("VLM_ENABLE_DEPTH (load Depth Pro at vlm_service start)")
        self.rc_arduino_baud = QSpinBox()
        self.rc_arduino_baud.setRange(300, 1_000_000)
        self.rc_arduino_baud.setSingleStep(100)
        form.addRow("DECODER_BACKEND", self.rc_decoder)
        form.addRow("EARLYSTOP_MODE", self.rc_earlystop)
        form.addRow("CLASS_VISUAL_STYLE", self.rc_visual)
        form.addRow("THRESHOLD_MI", self.rc_th_mi)
        form.addRow("THRESHOLD_REST", self.rc_th_rest)
        form.addRow("INTEGRATOR_ALPHA", self.rc_int_alpha)
        form.addRow("CLASSIFY_WINDOW (ms)", self.rc_classify_ms)
        form.addRow("MIN_PREDICTIONS", self.rc_min_pred)
        form.addRow("TIME_MI (s)", self.rc_time_mi)
        form.addRow("TIME_ROB (s)", self.rc_time_rob)
        form.addRow(self.rc_big_brother)
        form.addRow(self.rc_send_probs)
        form.addRow(self.rc_save_adaptive_t)
        form.addRow(self.rc_recentering)
        form.addRow(self.rc_update_move)
        form.addRow(self.rc_laplacian)
        form.addRow(self.rc_sel_motor)
        form.addRow(self.rc_xgb_beta)
        form.addRow("TOTAL_TRIALS", self.rc_total_trials)
        form.addRow("SHAPE_MAX", self.rc_shape_max)
        form.addRow("SHAPE_MIN", self.rc_shape_min)

        # Perception / VLM rows — mark local keys explicitly in the label
        # so the operator knows which file the Apply will touch.
        form.addRow("GAZE_OR_BACKEND", self.rc_gaze_backend)
        form.addRow("PERCEPTION_FRAME_SOURCE  [local]", self.rc_perception_source)
        form.addRow(self.rc_services_remote)
        form.addRow("FRAME_RELAY_HZ", self.rc_relay_hz)
        form.addRow(self.rc_vlm_depth)
        form.addRow("ARDUINO_BAUD", self.rc_arduino_baud)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)
        btn_row = QHBoxLayout()
        btn_reload = QPushButton("Reload from config.py")
        btn_reload.clicked.connect(self.on_runtime_reload_config)
        btn_apply = QPushButton("Apply to config.py")
        btn_apply.clicked.connect(self.on_runtime_apply_config)
        btn_row.addWidget(btn_reload)
        btn_row.addWidget(btn_apply)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)
        self.on_runtime_reload_config()

    def _rc_set_combo(self, cb: QComboBox, text: str, fallback_index: int = 0):
        idx = cb.findText(text)
        cb.setCurrentIndex(idx if idx >= 0 else fallback_index)

    def on_runtime_reload_config(self):
        if not hasattr(self, "rc_decoder"):
            return
        self._rc_set_combo(self.rc_decoder, _read_quoted_str_key("DECODER_BACKEND", "mdm"))
        self._rc_set_combo(self.rc_earlystop, _read_quoted_str_key("EARLYSTOP_MODE", "either"))
        vis = _read_quoted_str_key("CLASS_VISUAL_STYLE", "classic")
        self._rc_set_combo(self.rc_visual, vis if vis in ("classic", "modern") else "classic")
        self.rc_th_mi.setValue(_read_float_key("THRESHOLD_MI", 0.65))
        self.rc_th_rest.setValue(_read_float_key("THRESHOLD_REST", 0.65))
        self.rc_int_alpha.setValue(_read_float_key("INTEGRATOR_ALPHA", 0.96))
        self.rc_classify_ms.setValue(_read_int_key("CLASSIFY_WINDOW", 1000))
        self.rc_min_pred.setValue(_read_int_key("MIN_PREDICTIONS", 8))
        self.rc_time_mi.setValue(_read_int_key("TIME_MI", 5))
        self.rc_time_rob.setValue(_read_int_key("TIME_ROB", 7))
        self.rc_big_brother.setChecked(_read_bool_key("BIG_BROTHER_MODE", True))
        self.rc_send_probs.setChecked(_read_bool_key("SEND_PROBS", False))
        self.rc_save_adaptive_t.setChecked(_read_bool_key("SAVE_ADAPTIVE_T", False))
        self.rc_recentering.setChecked(bool(_read_01_key("RECENTERING", 1)))
        self.rc_update_move.setChecked(bool(_read_01_key("UPDATE_DURING_MOVE", 0)))
        self.rc_laplacian.setChecked(bool(_read_01_key("SURFACE_LAPLACIAN_TOGGLE", 1)))
        self.rc_sel_motor.setChecked(bool(_read_01_key("SELECT_MOTOR_CHANNELS", 1)))
        self.rc_xgb_beta.setChecked(bool(_read_01_key("XGB_USE_COV_BETA", 0)))
        self.rc_total_trials.setValue(_read_int_key("TOTAL_TRIALS", 10))
        self.rc_shape_max.setValue(_read_float_key("SHAPE_MAX", 0.7))
        self.rc_shape_min.setValue(_read_float_key("SHAPE_MIN", 0.5))
        # Perception / VLM (readers consult config_local.py first, then
        # config.py — same precedence as the live import).
        gob = _read_quoted_str_key("GAZE_OR_BACKEND", "legacy").lower()
        self._rc_set_combo(self.rc_gaze_backend, gob if gob in ("legacy", "vlm") else "legacy")
        pfs = _read_quoted_str_key("PERCEPTION_FRAME_SOURCE", "local").lower()
        self._rc_set_combo(self.rc_perception_source, pfs if pfs in ("local", "remote") else "local")
        self.rc_services_remote.setChecked(_read_bool_key("SERVICES_HOSTED_REMOTELY", False))
        self.rc_relay_hz.setValue(_read_float_key("FRAME_RELAY_HZ", 15.0))
        self.rc_vlm_depth.setChecked(_read_bool_key("VLM_ENABLE_DEPTH", True))
        self.rc_arduino_baud.setValue(_read_int_key("ARDUINO_BAUD", 9600))
        self._append_log("Panel", f"[{self._ts()}] Runtime config widgets reloaded from config.py / config_local.py\n")

    def on_runtime_apply_config(self):
        try:
            def _fmtf(x: float) -> str:
                t = f"{x:.6f}".rstrip("0").rstrip(".")
                return t if t else "0"

            _write_assign_rhs("DECODER_BACKEND", f'"{self.rc_decoder.currentText()}"')
            _write_assign_rhs("EARLYSTOP_MODE", f'"{self.rc_earlystop.currentText()}"')
            _write_assign_rhs("CLASS_VISUAL_STYLE", f'"{self.rc_visual.currentText()}"')
            _write_assign_rhs("THRESHOLD_MI", _fmtf(self.rc_th_mi.value()))
            _write_assign_rhs("THRESHOLD_REST", _fmtf(self.rc_th_rest.value()))
            _write_assign_rhs("INTEGRATOR_ALPHA", _fmtf(self.rc_int_alpha.value()))
            _write_assign_rhs("CLASSIFY_WINDOW", str(self.rc_classify_ms.value()))
            _write_assign_rhs("MIN_PREDICTIONS", str(self.rc_min_pred.value()))
            _write_assign_rhs("TIME_MI", str(self.rc_time_mi.value()))
            _write_assign_rhs("TIME_ROB", str(self.rc_time_rob.value()))
            _write_assign_rhs("BIG_BROTHER_MODE", "True" if self.rc_big_brother.isChecked() else "False")
            _write_assign_rhs("SEND_PROBS", "True" if self.rc_send_probs.isChecked() else "False")
            _write_assign_rhs("SAVE_ADAPTIVE_T", "True" if self.rc_save_adaptive_t.isChecked() else "False")
            _write_assign_rhs("RECENTERING", "1" if self.rc_recentering.isChecked() else "0")
            _write_assign_rhs("UPDATE_DURING_MOVE", "1" if self.rc_update_move.isChecked() else "0")
            _write_assign_rhs("SURFACE_LAPLACIAN_TOGGLE", "1" if self.rc_laplacian.isChecked() else "0")
            _write_assign_rhs("SELECT_MOTOR_CHANNELS", "1" if self.rc_sel_motor.isChecked() else "0")
            _write_assign_rhs("XGB_USE_COV_BETA", "1" if self.rc_xgb_beta.isChecked() else "0")
            _write_assign_rhs("TOTAL_TRIALS", str(self.rc_total_trials.value()))
            _write_assign_rhs("SHAPE_MAX", _fmtf(self.rc_shape_max.value()))
            _write_assign_rhs("SHAPE_MIN", _fmtf(self.rc_shape_min.value()))
            # Perception / VLM (the writer routes machine-local keys to
            # config_local.py automatically).
            _write_assign_rhs("GAZE_OR_BACKEND", f'"{self.rc_gaze_backend.currentText()}"')
            _write_assign_rhs("PERCEPTION_FRAME_SOURCE", f'"{self.rc_perception_source.currentText()}"')
            _write_assign_rhs("SERVICES_HOSTED_REMOTELY", "True" if self.rc_services_remote.isChecked() else "False")
            _write_assign_rhs("FRAME_RELAY_HZ", _fmtf(self.rc_relay_hz.value()))
            _write_assign_rhs("VLM_ENABLE_DEPTH", "True" if self.rc_vlm_depth.isChecked() else "False")
            _write_assign_rhs("ARDUINO_BAUD", str(self.rc_arduino_baud.value()))
        except Exception as e:
            QMessageBox.warning(self, "Runtime config", f"Failed to update config files:\n{e}")
            self._append_log("Panel", f"[{self._ts()}] Runtime config apply FAILED: {e}\n")
            return
        self._append_log("Panel", f"[{self._ts()}] Runtime config written to config.py / config_local.py\n")
        QMessageBox.information(
            self, "Runtime config",
            "config.py / config_local.py updated. Restart experiment driver / marker stream if a "
            "process was already running so it reloads settings.",
        )

    def _build_errp_config_tab(self, tabs: QTabWidget):
        rtc = QWidget()
        tabs.addTab(rtc, "ErrP config")
        outer = QVBoxLayout(rtc)
        outer.addWidget(QLabel(
            "<b>Edits ErrP-specific keys in config.py.</b> The bundle on disk at "
            "<code>DATA_DIR/sub-&lt;SUBJECT&gt;/models/sub-&lt;SUBJECT&gt;_errp_&lt;BACKEND&gt;.pkl</code> "
            "must exist for the selected backend. Runtime asserts the bundle's "
            "feature_spec matches config; mismatch raises."
        ))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        form = QFormLayout(inner)

        self.errp_enable = QCheckBox("ERRP_DECODER_ENABLE (ErrP gating active)")
        self.errp_backend = QComboBox()
        self.errp_backend.addItems(["liu_cca_xgb", "xdawn_xgb"])
        self.errp_sel_channels = QCheckBox("SELECT_ERRP_CHANNELS")
        self.errp_ea_bootstrap_sec = QDoubleSpinBox()
        self.errp_ea_bootstrap_sec.setRange(1.0, 300.0)
        self.errp_ea_bootstrap_sec.setSingleStep(1.0)
        self.errp_ea_bootstrap_sec.setDecimals(1)
        self.errp_ea_min_epochs = QSpinBox()
        self.errp_ea_min_epochs.setRange(1, 500)
        self.errp_p_stop = QDoubleSpinBox()
        self.errp_p_stop.setRange(0.0, 1.0)
        self.errp_p_stop.setSingleStep(0.05)
        self.errp_p_stop.setDecimals(2)

        form.addRow(self.errp_enable)
        form.addRow("ERRP_DECODER_BACKEND", self.errp_backend)
        form.addRow(self.errp_sel_channels)
        form.addRow("ERRP_EA_BOOTSTRAP_SEC", self.errp_ea_bootstrap_sec)
        form.addRow("ERRP_EA_MIN_EPOCHS", self.errp_ea_min_epochs)
        form.addRow("ERRP_ONLINE_P_STOP", self.errp_p_stop)

        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)
        btn_row = QHBoxLayout()
        btn_reload = QPushButton("Reload from config.py")
        btn_reload.clicked.connect(self.on_errp_config_reload)
        btn_apply = QPushButton("Apply to config.py")
        btn_apply.clicked.connect(self.on_errp_config_apply)
        btn_row.addWidget(btn_reload)
        btn_row.addWidget(btn_apply)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)
        self.on_errp_config_reload()

    def on_errp_config_reload(self):
        if not hasattr(self, "errp_backend"):
            return
        self.errp_enable.setChecked(bool(_read_01_key("ERRP_DECODER_ENABLE", 0)))
        backend = _read_quoted_str_key("ERRP_DECODER_BACKEND", "liu_cca_xgb")
        self._rc_set_combo(self.errp_backend, backend)
        self.errp_sel_channels.setChecked(bool(_read_01_key("SELECT_ERRP_CHANNELS", 0)))
        self.errp_ea_bootstrap_sec.setValue(_read_float_key("ERRP_EA_BOOTSTRAP_SEC", 45.0))
        self.errp_ea_min_epochs.setValue(_read_int_key("ERRP_EA_MIN_EPOCHS", 20))
        self.errp_p_stop.setValue(_read_float_key("ERRP_ONLINE_P_STOP", 0.3))
        self._append_log("Panel", f"[{self._ts()}] ErrP config widgets reloaded from config.py\n")

    def on_errp_config_apply(self):
        try:
            def _fmtf(x: float) -> str:
                t = f"{x:.6f}".rstrip("0").rstrip(".")
                return t if t else "0"

            _write_assign_rhs("ERRP_DECODER_ENABLE", "1" if self.errp_enable.isChecked() else "0")
            _write_assign_rhs("ERRP_DECODER_BACKEND", f'"{self.errp_backend.currentText()}"')
            _write_assign_rhs("SELECT_ERRP_CHANNELS", "1" if self.errp_sel_channels.isChecked() else "0")
            _write_assign_rhs("ERRP_EA_BOOTSTRAP_SEC", _fmtf(self.errp_ea_bootstrap_sec.value()))
            _write_assign_rhs("ERRP_EA_MIN_EPOCHS", str(self.errp_ea_min_epochs.value()))
            _write_assign_rhs("ERRP_ONLINE_P_STOP", _fmtf(self.errp_p_stop.value()))
        except Exception as e:
            QMessageBox.warning(self, "config.py", f"Failed to update config.py:\n{e}")
            self._append_log("Panel", f"[{self._ts()}] ErrP config apply FAILED: {e}\n")
            return
        self._append_log("Panel", f"[{self._ts()}] ErrP config written to config.py\n")
        QMessageBox.information(
            self, "ErrP config",
            "config.py updated. Restart ExperimentDriver_ErrP_Online if a session was "
            "already running so it reloads the selected bundle.",
        )

    def _populate_training_script_combo(self):
        if not hasattr(self, "cmb_train_script"):
            return
        self.cmb_train_script.blockSignals(True)
        self.cmb_train_script.clear()
        for label, fname in TRAINING_SCRIPT_ENTRIES:
            path = os.path.join(ROOT, fname)
            if os.path.isfile(path):
                self.cmb_train_script.addItem(label, path)
        if self.cmb_train_script.count() == 0:
            self.cmb_train_script.addItem("No training scripts found", "")
        self.cmb_train_script.blockSignals(False)
        self.cmb_train_script.currentIndexChanged.connect(self._update_train_cmd_preview)

    def _update_train_cmd_preview(self, *_args):
        if not hasattr(self, "lbl_train_cmd_preview"):
            return
        script = self.cmb_train_script.currentData()
        if script and os.path.isfile(script):
            self.lbl_train_cmd_preview.setText(f'cd "{ROOT}" && python -u "{script}"')
        else:
            self.lbl_train_cmd_preview.setText("(no script selected)")

    def on_refresh_training_data_list(self):
        if not hasattr(self, "lst_training_files"):
            return
        sub = (self.cmb_subject.currentText().strip() if hasattr(self, "cmb_subject") else "") or self.training_subject
        if _HCFG is None:
            self.lbl_training_subject_ctx.setText("DATA_DIR not available (config import failed).")
            self.btn_launch_training.setEnabled(False)
            self._update_train_cmd_preview()
            return
        data_dir = os.path.expanduser(getattr(_HCFG, "DATA_DIR", "") or "")
        tdir = os.path.join(data_dir, f"sub-{sub}", "training_data")
        self.lbl_training_subject_ctx.setText(f"<b>Subject:</b> {sub}<br><b>training_data:</b> {tdir}")
        self.lst_training_files.clear()
        xdffc = []
        if os.path.isdir(tdir):
            for fn in sorted(os.listdir(tdir)):
                if fn.lower().endswith(".xdf"):
                    full = os.path.join(tdir, fn)
                    try:
                        mtime = os.path.getmtime(full)
                        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
                    except OSError:
                        ts = "?"
                    self.lst_training_files.addItem(f"{fn}  ({ts})")
                    xdffc.append(full)
        script_ok = bool(self.cmb_train_script.currentData()) and os.path.isfile(self.cmb_train_script.currentData() or "")
        self.btn_launch_training.setEnabled(len(xdffc) > 0 and script_ok)
        self._update_train_cmd_preview()

    def on_launch_model_training(self):
        script = self.cmb_train_script.currentData()
        if not script or not os.path.isfile(script):
            QMessageBox.warning(self, "Training", "Select a valid training script.")
            return
        sub = (self.cmb_subject.currentText().strip() or self.training_subject)
        if _HCFG is None:
            QMessageBox.warning(self, "Training", "config module not loaded.")
            return
        data_dir = os.path.expanduser(getattr(_HCFG, "DATA_DIR", "") or "")
        tdir = os.path.join(data_dir, f"sub-{sub}", "training_data")
        if not os.path.isdir(tdir):
            QMessageBox.warning(self, "Training", f"training_data folder not found:\n{tdir}")
            return
        xdffc = [f for f in os.listdir(tdir) if f.lower().endswith(".xdf")]
        if not xdffc:
            QMessageBox.warning(self, "Training", "No .xdf files in training_data.")
            return
        cmd = f'python -u "{script}"'
        self._spawn_external(cmd)
        self._append_log("Panel", f"[{self._ts()}] Launched training: {cmd}\n")

    def on_save_serial_to_config(self):
        port = (self.serial_port_name or self.cmb_serial_port.currentData() or "").strip()
        if not port:
            QMessageBox.warning(self, "Serial", "Select a serial port first.")
            return
        try:
            baud = int(str(self.serial_baudrate).strip())
        except ValueError:
            QMessageBox.warning(self, "Serial", "Baud must be an integer (set it in Runtime config).")
            return
        try:
            write_arduino_port_to_config(port)
            write_arduino_baud_to_config(baud)
        except Exception as e:
            QMessageBox.warning(self, "config.py", f"Failed to write Arduino settings:\n{e}")
            return
        self._append_log("Panel", f"[{self._ts()}] Saved ARDUINO_PORT={port} ARDUINO_BAUD={baud} to config.py\n")
        QMessageBox.information(self, "Serial", "ARDUINO_PORT and ARDUINO_BAUD saved to config.py.")

    # ---------- LED helper ----------
    def _set_led(self, label: QLabel, state: str):
        color = {
            "stopped": "#888",
            "starting": "#e6a700",   # yellow
            "running": "#18a558",    # green
            "error": "#c62828",      # red
        }.get(state, "#888")
        label.setText("●")
        label.setStyleSheet(f"color: {color}; font-size: 18px;")

    # ---------- Command wiring ----------
    def _set_cmds_for_mode_and_driver(self):
        mode_flag = {
            "MI_Bimanual": "--mode mi_bimanual",
            "Gaze_Tracking": "--mode gaze",
            "Simulation": "--mode sim --no-robot",
        }[self.mode]

        if self.driver_choice == "ExperimentDriver_Online":
            driver_path = DRIVER_ONLINE_PY
        elif self.driver_choice == "ExperimentDriver_ErrP_Online":
            driver_path = DRIVER_ERRP_ONLINE_PY
        elif self.driver_choice == "ExperimentDriver_Bimanual":
            driver_path = DRIVER_BIMANUAL_PY
        elif self.driver_choice == "ExperimentDriver_Offline":
            driver_path = DRIVER_OFFLINE_PY
        elif self.driver_choice == "ExperimentDriver_Online_GazeTracking":
            driver_path = DRIVER_ONLINE_GAZE_PY
        elif self.driver_choice == "ExperimentDriver_Online_Glove":
            driver_path = DRIVER_ONLINE_GLOVE_PY
        else:
            QMessageBox.warning(self, "Driver", f"Unknown driver selected: {self.driver_choice}")
            return
        self.driver.cmd = f'python -u "{driver_path}" {mode_flag}'

        for p in (self.marker, self.driver, self.fes, self.gaze_runner, self.gaze_service, self.vlm_service):
            p.env["PYTHONUNBUFFERED"] = "1"
            p.env["TRAINING_SUBJECT"] = self.training_subject
            p.env["ARDUINO_PORT"]      = getattr(self, "serial_port_name", "") or ""
            p.env["ARDUINO_BAUD"]      = str(getattr(self, "serial_baudrate", "9600"))

        self._update_robot_buttons_for_mode()

    def _update_robot_buttons_for_mode(self):
        sim = (self.mode == "Simulation")
        self.btn_robot_start.setEnabled(not sim)
        if sim:
            self.btn_robot_start.setToolTip("Disabled in Simulation mode.")
        else:
            self.btn_robot_start.setToolTip("Open SSH terminal running the selected robot tool.")

    # ---------- Actions ----------
    def on_initialize(self):
        if not os.path.exists(INIT_SH):
            QMessageBox.warning(self, "Missing", f"Not found:\n{INIT_SH}")
            return
        cmd = f'gnome-terminal -- bash -lc "chmod +x \\"{INIT_SH}\\"; \\"{INIT_SH}\\"; exec bash"'
        subprocess.Popen(cmd, shell=True)
        QMessageBox.information(self, "Initialize", "Opened initialize_devices.sh in a new terminal.")

    def on_mode_changed(self, text: str):
        self.mode = text
        sim_on = (self.mode == "Simulation")
        try:
            write_simulation_mode(sim_on)
            self._append_log("Panel", f"[{self._ts()}] SIMULATION_MODE set to {sim_on}\n")
        except Exception as e:
            self._append_log("Panel", f"[{self._ts()}] Failed to write SIMULATION_MODE: {e}\n")

        self._set_cmds_for_mode_and_driver()
        self._append_log("Panel", f"[{self._ts()}] Mode set to {self.mode}\n")

    def on_driver_choice_changed(self, text: str):
        self.driver_choice = text
        self._set_cmds_for_mode_and_driver()
        self._append_log("Panel", f"[{self._ts()}] Driver selected: {self.driver_choice}\n")

    def on_save_subject(self):
        val = self.cmb_subject.currentText().strip()
        if not val:
            QMessageBox.warning(self, "Subject", "Subject cannot be empty.")
            return
        prev_subject = self.training_subject
        self.training_subject = val
        write_training_subject(val)
        for p in (self.marker, self.driver, self.fes, self.gaze_runner, self.gaze_service, self.vlm_service):
            p.env["TRAINING_SUBJECT"] = self.training_subject
        self._append_log("Panel", f"[{self._ts()}] TRAINING_SUBJECT saved: {val}\n")
        # Rotate the VLM log file into the new subject's directory so
        # session events stay sorted by who they belong to. No-op if
        # the subject hasn't actually changed.
        if val != prev_subject:
            self._open_vlm_log_file(val)
            self._open_relay_log_file(val)
        if hasattr(self, "on_refresh_training_data_list"):
            self.on_refresh_training_data_list()

    def on_copy_subject(self):
        val = self.cmb_subject.currentText().strip()
        QApplication.clipboard().setText(val, QClipboard.Clipboard)
        self._append_log("Panel", f"[{self._ts()}] Copied subject: {val}\n")

    def on_fes_pref_toggled(self, checked: bool):
        self.fes_enabled_pref = 1 if checked else 0
        write_fes_toggle(self.fes_enabled_pref)
        self._append_log("Panel", f"[{self._ts()}] FES_toggle set to {self.fes_enabled_pref}\n")

    def on_open_fes_cfg(self):
        if not os.path.exists(STMSETUP_PY):
            QMessageBox.warning(self, "Missing", f"Not found:\n{STMSETUP_PY}")
            return
        self._spawn_external(f'python -u "{STMSETUP_PY}"')
        self._append_log("Panel", f"[{self._ts()}] Opened STMsetup.py\n")

    def on_open_mne_viewer(self):
        self._spawn_external('mne-lsl viewer')
        self._append_log("Panel", f"[{self._ts()}] Opened mne-lsl viewer\n")

    def on_open_impedance_monitor(self):
        sub = (self.cmb_subject.currentText().strip() if hasattr(self, "cmb_subject") else "") or self.training_subject
        data_dir = os.path.expanduser(getattr(_HCFG, "DATA_DIR", "") or "") if _HCFG else ""
        if data_dir and sub:
            cmd = f'impedance-monitor --mode live --cap ca209 --subject "{sub}" --data-dir "{data_dir}"'
        else:
            # Fall back — the tool will default to ~/impedance_logs/
            cmd = 'impedance-monitor --mode live --cap ca209'
        self._spawn_external(cmd)
        log_dir = os.path.join(data_dir, f"sub-{sub}", "impedance_logs") if data_dir and sub else "~/impedance_logs"
        self._append_log("Panel", f"[{self._ts()}] Opened impedance monitor (logs → {log_dir})\n")

    # ----- Marker -----
    def on_marker_start(self):
        self._start_proc(self.marker, self.lbl_marker, "Marker")
    def on_marker_stop(self):
        self._stop_proc(self.marker, self.lbl_marker, "Marker")
    def on_marker_refresh(self):
        self.on_marker_stop()
        time.sleep(0.1)
        self.on_marker_start()
        self._append_log("Marker", f"[{self._ts()}] Refreshed marker stream\n")

    # ----- FES -----
    def on_fes_start(self):
        if not os.path.exists(FES_PY):
            QMessageBox.warning(self, "Missing", f"Not found:\n{FES_PY}")
            return
        self._start_proc(self.fes, self.lbl_fes, "FES")
    def on_fes_stop(self):
        self._stop_proc(self.fes, self.lbl_fes, "FES")
    def on_fes_refresh(self):
        self.on_fes_stop()
        time.sleep(0.1)
        self.on_fes_start()
        self._append_log("FES", f"[{self._ts()}] Refreshed FES listener\n")

    # ----- Gaze (NEW) -----
    def _ensure_gaze_paths(self, which: str) -> bool:
        path = GAZE_RUNNER_PY if which == "runner" else GAZE_SERVICE_PY
        if not os.path.exists(path):
            QMessageBox.warning(self, "Missing", f"Not found:\n{path}")
            return False
        return True

    def on_gaze_runner_start(self):
        if not self._ensure_gaze_paths("runner"):
            return
        # Runner: UI + prints for testing, but logs are captured into View: Gaze.
        neon_arg = f'--neon-device-host "{NEON_COMPANION_HOST}"' if NEON_COMPANION_HOST else ""
        self.gaze_runner.cmd = f'python -u "{GAZE_RUNNER_PY}" --mode runner --display 1 --prints 1 {neon_arg}'
        self._start_proc(self.gaze_runner, None, "Gaze")
        self._append_log("Gaze", f"[{self._ts()}] Runner start requested\n")

    def on_gaze_runner_stop(self):
        self._stop_proc(self.gaze_runner, None, "Gaze")

    def on_gaze_service_start_headless(self):
        self._start_gaze_service(display=0)

    def on_gaze_service_start_ui(self):
        self._start_gaze_service(display=1)

    def _start_gaze_service(self, *, display: int):
        if not self._ensure_gaze_paths("service"):
            return

        # Guard: avoid confusing "address already in use" if already running
        if _is_port_in_use(int(GAZE_SERVICE_PORT), GAZE_SERVICE_HOST):
            QMessageBox.warning(
                self,
                "Gaze service port in use",
                f"UDP port {GAZE_SERVICE_HOST}:{GAZE_SERVICE_PORT} appears in use.\n"
                f"If gaze service is already running, use Stop first.\n"
                f"Otherwise change GAZE_SERVICE_PORT."
            )

        # Service: prints can be 0 (supressed) or 1 (verbose) — either way logs go to View: Gaze
        neon_arg = f'--neon-device-host "{NEON_COMPANION_HOST}"' if NEON_COMPANION_HOST else ""
        # GPU-host topology: when PERCEPTION_FRAME_SOURCE=remote, gaze_runner
        # consumes envelopes from the relay instead of opening Neon directly.
        # Dial host comes from FRAME_RELAY_DIAL_HOST in config.
        remote_arg = ""
        if PERCEPTION_FRAME_SOURCE == "remote":
            relay_dial = str(getattr(_HCFG, "FRAME_RELAY_DIAL_HOST", "127.0.0.1") or "127.0.0.1") if _HCFG else "127.0.0.1"
            relay_port = int(getattr(_HCFG, "FRAME_RELAY_PORT", 5591)) if _HCFG else 5591
            remote_arg = (
                f'--frame-source remote '
                f'--remote-frame-host {relay_dial} '
                f'--remote-frame-port {relay_port}'
            )
        self.gaze_service.cmd = (
            f'python -u "{GAZE_SERVICE_PY}" --mode service '
            f'--display {int(display)} --prints 1 '
            f'--host {GAZE_BIND_HOST} --port {int(GAZE_SERVICE_PORT)} '
            f'--udp_log 1 --udp_log_hz 50 {neon_arg} {remote_arg}'
        )
        self._start_proc(self.gaze_service, self.lbl_gaze_service, "Gaze")
        self._append_log("Gaze", f"[{self._ts()}] Service start requested (display={display})\n")

    def on_gaze_service_stop(self):
        self._stop_proc(self.gaze_service, self.lbl_gaze_service, "Gaze")

    def on_gaze_service_query(self):
        import threading
        query_id = int(time.time() * 1000)

        # TX log (already correct)
        self._append_log("Panel",
            f"[{self._ts()}] Gaze UDP TX query_id={query_id} -> {GAZE_SERVICE_HOST}:{GAZE_SERVICE_PORT}\n"
        )

        def worker():
            t0 = time.time()
            try:
                req = {"cmd": "snapshot", "include_objects": True, "query_id": query_id}
                resp = self._gaze_udp_request(req, timeout_s=float(GAZE_QUERY_TIMEOUT_S))

                dt_ms = (time.time() - t0) * 1000.0
                pretty = json.dumps(resp, indent=2, sort_keys=True)

                msg = (
                    f"[{self._ts()}] Gaze UDP RX OK query_id={query_id} "
                    f"({dt_ms:.0f} ms)\n{pretty}\n"
                )

                # existing
                self._append_log_ui("Gaze", msg)

                # ✅ ADD THIS LINE — this is all you need
                self._append_log_ui("Panel",
                    f"[{self._ts()}] Gaze UDP RX OK query_id={query_id} ({dt_ms:.0f} ms)\n"
                )

            except Exception as e:
                dt_ms = (time.time() - t0) * 1000.0
                err = (
                    f"[{self._ts()}] Gaze UDP RX ERROR query_id={query_id} "
                    f"({dt_ms:.0f} ms): {e}\n"
                )

                self._append_log_ui("Panel", err)
                self._append_log_ui("Gaze", err)

        threading.Thread(target=worker, daemon=True).start()

    # ----- VLM service handlers -----

    def _configure_remote_services_ui(self) -> None:
        """When SERVICES_HOSTED_REMOTELY=True, the GPU service runs on
        another host so spawning a local copy is meaningless. We
        rebrand the VLM Start/Stop pair to "Connect"/"Disconnect" and
        rewire them to the local-pipeline lifecycle (Neon reader,
        embedded frame_relay, push subscribers via VLMSceneWidget) —
        the user's actual pre-session toggle on this box.

        Local-spawn handlers (on_vlm_service_start/stop) are
        disconnected and replaced with _on_vlm_video_connect/disconnect.
        The lbl_compute_led keeps its existing semantic — green
        when the GPU service is reachable AND has a frame source,
        which is exactly what "connected" should mean here. Status,
        Decide, Depth buttons stay as-is — they're plain UDP queries
        that travel across the LAN unchanged.

        Gaze Start variants stay visible-but-disabled so the
        surrounding grid row keeps its column alignment with the
        other module rows.
        """
        for btn, new_text, new_handler, old_handler, tip in (
            (
                getattr(self, "btn_vlm_service_start", None),
                "Connect",
                self._on_vlm_video_connect,
                self.on_vlm_service_start,
                "Open Neon over the tailnet, start the embedded "
                "frame_relay, and subscribe to vlm_service push on "
                "the GPU host. The GPU service itself runs on a "
                "different machine and is not started by this button.",
            ),
            (
                getattr(self, "btn_vlm_service_stop", None),
                "Disconnect",
                self._on_vlm_video_disconnect,
                self.on_vlm_service_stop,
                "Stop the local pipeline (paint loop, push "
                "subscribers, embedded relay, Neon reader). The GPU "
                "service is unaffected — it stays up waiting for the "
                "next client.",
            ),
        ):
            if btn is None:
                continue
            try:
                btn.clicked.disconnect(old_handler)
            except (RuntimeError, TypeError):
                pass
            btn.clicked.connect(new_handler)
            btn.setText(new_text)
            btn.setToolTip(tip)
        for btn_name in (
            "btn_gaze_service_headless", "btn_gaze_service_ui",
            "btn_gaze_service_stop",
        ):
            btn = getattr(self, btn_name, None)
            if btn is not None:
                btn.setEnabled(False)
                btn.setToolTip("Disabled: SERVICES_HOSTED_REMOTELY=True. "
                               "Manage these on the GPU host.")

    def _poll_remote_status(self) -> None:
        """1 s cadence. Spawns a daemon thread to do the UDP RTT off the
        GUI thread; the result comes back via the _remote_status_received
        signal. The status request itself is cheap, but its 0.4 s timeout
        blocked the Qt event loop when the Windows host is unreachable —
        visible as a sub-second stutter in the VLM Video tab paint pass.

        Skip if a previous poll is still in flight so a flaky link can't
        accumulate worker threads.
        """
        if self._remote_status_in_flight:
            return
        self._remote_status_in_flight = True

        def _worker() -> None:
            try:
                resp = self._vlm_udp_request({"cmd": "status"}, timeout_s=0.4)
            except Exception:
                resp = {"ok": False, "_unreachable": True}
            try:
                self._remote_status_received.emit(resp or {})
            except RuntimeError:
                # Window closed while we were mid-UDP — the underlying
                # C++ ControlPanel is already gone. Drop the late status
                # reply silently; the alternative is a noisy traceback
                # at every shutdown when the GPU host is unreachable.
                # Other workers route through _append_log_ui's
                # QTimer.singleShot(self, ...) which Qt auto-cancels;
                # this one is the lone direct-emit path.
                pass

        threading.Thread(target=_worker, daemon=True,
                         name="panel-remote-status").start()

    def _apply_remote_status(self, resp: dict) -> None:
        """GUI-thread slot for _remote_status_received.

        Compute LED semantic (per the per-Connect verification model):
          - gray  (stopped)  → unreachable
          - red   (error)    → reachable but ok=False
          - green (running)  → reachable AND ok AND _send_observed
        "Compute green" means "the GPU script is alive and ready to
        accept data" — explicitly NOT gated on frames_received, since
        actual data flow is what the Receive verification step proves.

        Side-effect: when this reply confirms a fresh bundle on the
        GPU side (frame_age < 2 s) AND the verification phase is
        ready to fire (_send_observed and _compute_observed and a
        token is armed), trigger cmd=verify_chain with the current
        token. The token-matched chain_verify push back to the panel
        is what flips the Receive LED via _on_vlm_payload_received.
        """
        self._remote_status_in_flight = False
        if resp.get("_unreachable"):
            self._set_led(self.lbl_compute_led, "stopped")
            self.lbl_compute_led.setToolTip("compute: unreachable")
            return
        ok = bool(resp.get("ok"))
        connected = bool(resp.get("frame_source_connected"))
        frames = int(resp.get("frames_received") or 0)
        src = resp.get("frame_source", "?")
        age = resp.get("frame_age_s")
        age_txt = f"{float(age):.2f}s" if isinstance(age, (int, float)) else "--"
        if not ok:
            led_state = "error"
        elif self._send_observed:
            # User spec: Compute = "GPU script is alive and ready". Send
            # must already be observed for visible ordering, but we do
            # not require frames_received>0 here — that's the Receive
            # phase's job.
            led_state = "running"
        elif self._connect_token is not None:
            # Verification in progress (Connect armed) but Send hasn't
            # been observed yet — hold yellow.
            led_state = "starting"
        else:
            # Idle (no Connect armed). The 1 s status poll runs
            # continuously even before the operator clicks Connect, so
            # we must NOT paint yellow here — yellow is the "verifying"
            # state and is reserved for the active Connect cycle. Keep
            # gray to match the other LEDs in the idle state.
            led_state = "stopped"
        self._set_led(self.lbl_compute_led, led_state)
        self.lbl_compute_led.setToolTip(
            f"compute: src={src} connected={connected} frames={frames} age={age_txt}"
        )
        if led_state == "running" and not self._compute_observed:
            self._compute_observed = True
            self._append_log(
                "VLM",
                f"[{self._ts()}] chain: compute green ok=True frames={frames}\n",
            )
        # End-to-end Receive verification trigger. We need:
        #   - both Send and Compute observed (predecessor phases done)
        #   - GPU has a fresh bundle (connected=True, frame_age<2s) so
        #     verify_chain doesn't return no_frame
        #   - this Connect's token is armed
        #   - no in-flight verify_chain already
        # The verify_chain RPC echoes our token in its push payload;
        # _on_vlm_payload_received does the matching to flip Receive
        # green. Stale GPU-cache pushes (no token / old token) cannot
        # trip Receive on a reconnect.
        if (self._compute_observed
                and self._send_observed
                and connected
                and self._connect_token is not None
                and not self._verify_chain_in_flight
                and self._verify_chain_attempts == 0):
            self._fire_verify_chain()

    def _poll_relay_status(self) -> None:
        """2 s cadence. Reflects whether the frame relay is alive.

        Under the per-Connect verification model, the Send LED's
        green flip is owned by the handshake event
        (_on_handshake_observed). This poll therefore only owns the
        running→stopped edge — i.e. detecting that the relay thread
        has died after Send was already observed. While verification
        is in progress (_send_observed=False) this poll is a no-op
        for the LED; the state machine paints it.

        When the panel hosts the relay in-process (FRAME_RELAY_EMBEDDED),
        we ask the widget directly — TCP-pinging localhost would create
        phantom client churn (each ping does connect-then-close, the
        relay's accept loop installs the dead socket, the pump pays a
        full JPEG encode + sendall before discovering the peer is gone,
        and the SDK iterator stalls behind that work → visible stutter
        in the local subscriber path).
        """
        if not self._send_observed:
            return  # State machine owns Send before handshake.
        widget = getattr(self, "vlm_scene_widget", None)
        if widget is not None and getattr(widget, "_embedded_relay", None) is not None:
            thread = getattr(widget, "_embedded_relay_thread", None)
            alive = thread is not None and thread.is_alive()
            relay = widget._embedded_relay
            published = int(getattr(relay, "published_count", 0) or 0)
            if alive:
                # Send was observed via handshake; keep green and
                # surface the published-count in the tooltip for
                # diagnostics.
                self._set_led(self.lbl_send_led, "running")
                self.lbl_send_led.setToolTip(
                    f"send: in-process @ {FRAME_RELAY_BIND_HOST}:{FRAME_RELAY_PORT} "
                    f"(published={published})"
                )
            else:
                # Relay thread died after Send was observed.
                self._set_led(self.lbl_send_led, "stopped")
                self.lbl_send_led.setToolTip("send: in-process — thread exited")
            return

        # External relay (FRAME_RELAY_EMBEDDED=False or remote host) —
        # fall back to the TCP ping for the running→stopped edge.
        try:
            from Utils.perception_clients import FrameRelayController
        except Exception:
            return
        ctl = FrameRelayController(_HCFG) if _HCFG else None
        if ctl is None:
            return
        ping = ctl.ping(timeout_s=0.5)
        if ping.get("ok"):
            self._set_led(self.lbl_send_led, "running")
            self.lbl_send_led.setToolTip(
                f"send: reachable @ {ping['host']}:{ping['port']}"
            )
        else:
            self._set_led(self.lbl_send_led, "stopped")
            self.lbl_send_led.setToolTip(
                f"send: unreachable @ {ping['host']}:{ping['port']}"
            )

    def _vlm_udp_request(self, payload: dict, timeout_s: float = VLM_QUERY_TIMEOUT_S) -> dict:
        """One-shot JSON request against vlm_service.py.

        Delegates to ``Utils.perception_clients.udp_request`` so the wire
        format (JSON, UDP, single datagram round-trip) lives in one place;
        the panel layer here is purely UI plumbing.
        """
        from Utils.perception_clients import udp_request
        return udp_request(VLM_SERVICE_HOST, VLM_SERVICE_PORT, payload, float(timeout_s))

    def on_vlm_service_start(self):
        if not os.path.exists(VLM_SERVICE_PY):
            QMessageBox.warning(self, "Missing", f"Not found:\n{VLM_SERVICE_PY}")
            return
        if not PERCEPTION_MODELS_DIR or not os.path.isdir(PERCEPTION_MODELS_DIR):
            QMessageBox.warning(self, "Perception models missing",
                                f"PERCEPTION_MODELS_DIR not a dir:\n{PERCEPTION_MODELS_DIR}")
            return

        # Reap any orphaned vlm_service.py left over from a previous crash or
        # incomplete stop (a killed service can leave its python holding the port).
        _kill_orphan_vlm_service()

        if _is_port_in_use(int(VLM_SERVICE_PORT), VLM_SERVICE_HOST):
            QMessageBox.warning(
                self,
                "VLM service port in use",
                f"UDP port {VLM_SERVICE_HOST}:{VLM_SERVICE_PORT} appears in use.\n"
                f"Use Stop first or change VLM_SERVICE_PORT."
            )

        session_dir = ""
        if VLM_SESSION_ROOT:
            try:
                os.makedirs(VLM_SESSION_ROOT, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                session_dir = os.path.join(VLM_SESSION_ROOT, f"session_{ts}")
                os.makedirs(session_dir, exist_ok=True)
            except OSError as e:
                self._append_log("VLM", f"[{self._ts()}] Failed to create session dir: {e}\n")

        depth_flag = "--enable-depth" if VLM_ENABLE_DEPTH else ""
        session_arg = f'--session-dir "{session_dir}"' if session_dir else ""
        # On the Windows dev box (RTX 4070 Ti) we want FastSAM + Depth Pro on
        # CUDA; the Linux deployment is CPU-only (no NVIDIA driver).
        device_flag = "--device cuda" if _IS_WINDOWS else "--device cpu"
        # Launch vlm_service.py with the panel's own interpreter. Since WS3
        # unified the env (perception deps now live in this env), there is no
        # separate harmony_vlm env to resolve — same env, still a separate
        # process so a blocking model call can't stall the panel.
        py = sys.executable
        self._append_log("VLM", f"[{self._ts()}] using python: {py}\n")
        # --neon-host "" forces discover_one_device in perception.neon's
        # NeonLiveReader (neon/reader.py), matching our gaze_system.py:250 pattern.
        # GPU-host topology: when PERCEPTION_FRAME_SOURCE=remote, vlm_service
        # consumes envelopes from the Linux frame_relay rather than opening
        # Neon itself. Dial host comes from FRAME_RELAY_DIAL_HOST.
        remote_arg = ""
        if PERCEPTION_FRAME_SOURCE == "remote":
            relay_dial = str(getattr(_HCFG, "FRAME_RELAY_DIAL_HOST", "127.0.0.1") or "127.0.0.1") if _HCFG else "127.0.0.1"
            relay_port = int(getattr(_HCFG, "FRAME_RELAY_PORT", 5591)) if _HCFG else 5591
            remote_arg = (
                f'--frame-source remote '
                f'--remote-frame-host {relay_dial} '
                f'--remote-frame-port {relay_port}'
            )
        self.vlm_service.cmd = (
            f'"{py}" -u "{VLM_SERVICE_PY}" '
            f'--host {VLM_BIND_HOST} --port {int(VLM_SERVICE_PORT)} '
            f'--neon-host "{NEON_COMPANION_HOST}" '
            f'--model {VLM_MODEL} {device_flag} '
            f'{depth_flag} {session_arg} {remote_arg}'
        )
        self._start_proc(self.vlm_service, self.lbl_compute_led, "VLM")
        self._append_log(
            "VLM",
            f"[{self._ts()}] Service start requested "
            f"(depth={VLM_ENABLE_DEPTH}, model={VLM_MODEL})\n",
        )
        # Block Windows from sleeping while the GPU stack is alive.
        _sleep_inhibit(True)
        self._on_vlm_video_connect()

    def on_vlm_service_stop(self):
        # Reset the streaming toggle so the panel doesn't claim "ON" while
        # the service it was driving is dead. blockSignals prevents the
        # toggle handler from firing a doomed UDP send to the dying service.
        if self.btn_vlm_seg_stream.isChecked():
            self.btn_vlm_seg_stream.blockSignals(True)
            self.btn_vlm_seg_stream.setChecked(False)
            self.btn_vlm_seg_stream.setText("Stream Seg: OFF")
            self.btn_vlm_seg_stream.blockSignals(False)
        # Stop the seg-stream readout timer too — its 5 s tick would
        # otherwise spam "unreachable" lines into the VLM log while the
        # service tears down.
        self._seg_stream_log_timer.stop()
        # Ask vlm_service to exit gracefully before killing the process. A hard
        # kill alone can leave children it spawned orphaned holding the UDP port.
        try:
            self._vlm_udp_request({"cmd": "stop"}, timeout_s=0.5)
        except Exception:
            pass
        self._stop_proc(self.vlm_service, self.lbl_compute_led, "VLM")
        # Belt-and-suspenders: reap any surviving orphan regardless.
        _kill_orphan_vlm_service()
        self._on_vlm_video_disconnect()
        _sleep_inhibit(False)

    def _vlm_command_threaded(self, payload: dict, timeout_s: float, label: str) -> None:
        import threading as _threading
        self._append_log("VLM", f"[{self._ts()}] {label} TX -> {VLM_SERVICE_HOST}:{VLM_SERVICE_PORT}\n")

        def worker():
            t0 = time.time()
            try:
                resp = self._vlm_udp_request(payload, timeout_s=timeout_s)
                dt_ms = (time.time() - t0) * 1000.0
                pretty = json.dumps(resp, indent=2, sort_keys=True)
                self._append_log_ui("VLM", f"[{self._ts()}] {label} RX OK ({dt_ms:.0f} ms)\n{pretty}\n")
            except Exception as e:
                dt_ms = (time.time() - t0) * 1000.0
                self._append_log_ui("VLM", f"[{self._ts()}] {label} RX ERROR ({dt_ms:.0f} ms): {e}\n")

        _threading.Thread(target=worker, daemon=True).start()

    def on_vlm_service_status(self):
        self._vlm_command_threaded({"cmd": "status"}, VLM_QUERY_TIMEOUT_S, "status")

    def on_vlm_service_decide(self):
        self._vlm_command_threaded({"cmd": "decide"}, VLM_DECIDE_TIMEOUT_S, "decide")

    def on_vlm_seg_stream_toggled(self, checked: bool) -> None:
        hz = float(self.spin_vlm_seg_hz.value())
        self.btn_vlm_seg_stream.setText(f"Stream Seg: {'ON' if checked else 'OFF'}")
        self._vlm_command_threaded(
            {"cmd": "segment_stream", "enabled": bool(checked), "hz": hz},
            VLM_QUERY_TIMEOUT_S,
            f"segment_stream({'on' if checked else 'off'}, {hz:.1f} Hz)",
        )
        # Drive the seg-stream readout off the toggle: the timer is only
        # meaningful while the stream is running. Reset the dedup marker
        # on each on-edge so the very first stats window emits.
        if checked:
            self._last_seg_stream_emit_t = 0.0
            self._seg_stream_log_timer.start()
        else:
            self._seg_stream_log_timer.stop()

    def on_vlm_seg_stream_hz_changed(self, hz: float) -> None:
        # Only push a rate change if the stream is currently on; otherwise
        # the spinner just sets the rate the next toggle-on will use.
        if not self.btn_vlm_seg_stream.isChecked():
            return
        self._vlm_command_threaded(
            {"cmd": "segment_stream", "enabled": True, "hz": float(hz)},
            VLM_QUERY_TIMEOUT_S,
            f"segment_stream(rate={hz:.1f} Hz)",
        )

    def on_vlm_service_depth(self):
        self._vlm_command_threaded({"cmd": "depth", "at_gaze": True}, 15.0, "depth")

    def _poll_seg_stream_stats(self) -> None:
        """Pull the seg-stream stats block from the VLM status reply and
        append a one-line summary to the VLM log buffer. Runs on a 5 s
        timer that's only active while the operator has Stream Seg on.

        The status request is the same UDP roundtrip the existing handlers
        use; we run it on a worker thread so a slow GPU host (or a service
        that just died) cannot stall the GUI thread."""
        def worker():
            try:
                resp = self._vlm_udp_request({"cmd": "status"}, timeout_s=0.5)
            except Exception as e:
                # Timer is short-lived; if status is unreachable while the
                # stream is supposedly on, surface that — most likely the
                # service died and the toggle is stale.
                self._append_log_ui(
                    "VLM",
                    f"[{self._ts()}] seg-stream status: unreachable ({e})\n",
                )
                return
            stats = resp.get("seg_stream") if isinstance(resp, dict) else None
            if not isinstance(stats, dict):
                return
            # Skip ticks where the service hasn't refreshed its window yet
            # — avoids three identical lines while the first 5 s window
            # accumulates.
            emit_t = float(stats.get("last_emit_t") or 0.0)
            if emit_t and emit_t == self._last_seg_stream_emit_t:
                return
            self._last_seg_stream_emit_t = emit_t
            active = bool(stats.get("active"))
            target = float(stats.get("hz_target") or 0.0)
            achieved = float(stats.get("hz_achieved") or 0.0)
            mean_dets = float(stats.get("mean_dets") or 0.0)
            mean_infer = float(stats.get("mean_infer_ms") or 0.0)
            errors = int(stats.get("errors") or 0)
            self._append_log_ui(
                "VLM",
                f"[{self._ts()}] seg-stream: active={active} "
                f"target={target:.1f}Hz achieved={achieved:.1f}Hz "
                f"mean_dets={mean_dets:.1f} mean_infer={mean_infer:.0f}ms "
                f"errors={errors}\n",
            )

        threading.Thread(target=worker, daemon=True,
                         name="panel-seg-stream-stats").start()

    def on_vlm_capture_first(self):
        import threading as _threading
        self._append_log("VLM", f"[{self._ts()}] capture_first TX -> {VLM_SERVICE_HOST}:{VLM_SERVICE_PORT}\n")

        def worker():
            t0 = time.time()
            try:
                resp = self._vlm_udp_request({"cmd": "capture_first"}, timeout_s=12.0)
                dt_ms = (time.time() - t0) * 1000.0
                if isinstance(resp, dict) and resp.get("ok") and resp.get("snapshot_id"):
                    self._vlm_last_snapshot_id = str(resp["snapshot_id"])
                    hit = resp.get("hit_waypoint")
                    hit_lbl = hit.get("label") if isinstance(hit, dict) else "—"
                    self._append_log_ui(
                        "VLM",
                        f"[{self._ts()}] capture_first RX OK ({dt_ms:.0f} ms)\n"
                        f"{json.dumps(resp, indent=2, sort_keys=True)}\n",
                    )
                    # Update token label on the main thread
                    from PySide6.QtCore import QMetaObject, Qt, Q_ARG
                    QMetaObject.invokeMethod(
                        self.lbl_vlm_pair_token, "setText", Qt.QueuedConnection,
                        Q_ARG(str, f"<i>snapshot:</i> {self._vlm_last_snapshot_id} ({hit_lbl})"),
                    )
                else:
                    self._append_log_ui(
                        "VLM",
                        f"[{self._ts()}] capture_first RX (no snapshot_id) ({dt_ms:.0f} ms)\n"
                        f"{json.dumps(resp, indent=2, sort_keys=True)}\n",
                    )
            except Exception as e:
                dt_ms = (time.time() - t0) * 1000.0
                self._append_log_ui("VLM", f"[{self._ts()}] capture_first RX ERROR ({dt_ms:.0f} ms): {e}\n")

        _threading.Thread(target=worker, daemon=True).start()

    def on_vlm_decide_pair(self):
        if not self._vlm_last_snapshot_id:
            QMessageBox.information(
                self, "No snapshot",
                "Click 'Capture First' on the source object before running 'Decide Pair'.",
            )
            return
        snap_id = self._vlm_last_snapshot_id
        self._vlm_command_threaded(
            {"cmd": "decide_pair", "snapshot_id": snap_id, "timeout": 45.0},
            60.0,
            f"decide_pair(snapshot_id={snap_id})",
        )
        # Cleared after use so accidental re-presses don't replay a stale snapshot.
        self._vlm_last_snapshot_id = None
        self.lbl_vlm_pair_token.setText("<i>snapshot:</i> (consumed)")

    def _format_gaze_telemetry_line(self, snap: dict) -> str:
        """
        Build a line similar to your terminal prints, using fields from telemetry JSON.
        This requires the service to include these keys in its response.
        """
        def _f(key, default="--"):
            v = snap.get(key, None)
            if v is None:
                return default
            return v

        t = _f("t", None)
        t_txt = f"t={t:.3f}" if isinstance(t, (int, float)) else f"t={t}"

        worn = bool(snap.get("worn", False))
        gaze_px = snap.get("gaze_px", None)
        gaze_txt = f"gaze=({gaze_px[0]:.1f},{gaze_px[1]:.1f})" if isinstance(gaze_px, (list, tuple)) and len(gaze_px) >= 2 else "gaze=(--,--)"

        loop_hz = snap.get("loop_hz", float("nan"))
        video_hz = snap.get("video_hz", float("nan"))
        det_hz = snap.get("det_hz", float("nan"))
        det_age_s = snap.get("det_age_s", float("nan"))
        infer_ms = snap.get("infer_ms", float("nan"))

        imu_w = snap.get("imu_angvel", None)
        imu_txt = f"imu|w|={imu_w:.2f}rad/s" if isinstance(imu_w, (int, float)) else "imu|w|=--"

        yolo_enabled = bool(snap.get("gov_enabled", True))
        reason = str(snap.get("gov_reason", "healthy"))
        cd = float(snap.get("gov_cd_left", 0.0))
        yolo_txt = f"YOLO={'ON' if yolo_enabled else 'OFF'}({reason}) cd={cd:.2f}s"

        tracks = snap.get("tracks", None)
        if tracks is None:
            # fallback: infer from objects list if present
            objs = snap.get("objects", None)
            tracks = len(objs) if isinstance(objs, list) else 0

        objs = snap.get("objects", None)
        if isinstance(objs, list) and len(objs) > 0:
            # expect entries like {"name": "...", "track_id": 1, "conf": 0.95}
            parts = []
            for o in objs[:6]:
                nm = str(o.get("name", "?"))
                tid = o.get("track_id", None)
                cf = o.get("conf", None)
                if tid is not None and cf is not None:
                    parts.append(f"{nm}#{int(tid)}({float(cf):.2f})")
                elif cf is not None:
                    parts.append(f"{nm}({float(cf):.2f})")
                else:
                    parts.append(nm)
            objs_txt = "objs: " + ", ".join(parts)
        else:
            objs_txt = "objs: none"

        hit = snap.get("gaze_hit", None)
        if isinstance(hit, dict):
            nm = str(hit.get("name", "none"))
            tid = hit.get("track_id", None)
            cf = hit.get("conf", None)
            if tid is not None and cf is not None:
                gaze_on = f"gaze_on: {nm}#{int(tid)}({float(cf):.2f})"
            elif cf is not None:
                gaze_on = f"gaze_on: {nm}({float(cf):.2f})"
            else:
                gaze_on = f"gaze_on: {nm}"
            mode = str(hit.get("mode", "--"))
            d = hit.get("dist_px", None)
            d_txt = f"{float(d):.1f}px" if isinstance(d, (int, float)) else "--"
            gaze_on = f"{gaze_on} mode={mode} d={d_txt}"
        else:
            gaze_on = "gaze_on: none"

        depth_cm = snap.get("depth_cm", float("nan"))
        miss_mm = snap.get("miss_mm", float("nan"))
        ipd_mm = snap.get("ipd_mm", float("nan"))
        depth_txt = f"depth={depth_cm:.1f}cm miss={miss_mm:.1f}mm IPD={ipd_mm:.1f}mm" if all(isinstance(x, (int, float)) for x in [depth_cm, miss_mm, ipd_mm]) else "depth=--"

        hy = snap.get("head_yaw_deg", float("nan"))
        hp = snap.get("head_pitch_deg", float("nan"))
        gy = snap.get("gaze_yaw_deg", float("nan"))
        gp = snap.get("gaze_pitch_deg", float("nan"))
        head_txt = f"HEAD(yaw,pitch)=({hy:+.0f},{hp:+.0f})"
        gaze_ang_txt = f"GAZE(yaw,pitch)=({gy:+.0f},{gp:+.0f})"

        return (
            f"{t_txt} worn={worn} {gaze_txt} | "
            f"rates: loop={loop_hz:.1f}Hz video={video_hz:.1f}Hz det~{det_hz:.1f}Hz det_age={det_age_s:.2f}s infer={infer_ms:.0f}ms | "
            f"{imu_txt} | {yolo_txt} | tracks={tracks} | {objs_txt} | {gaze_on} | {depth_txt} | {head_txt} {gaze_ang_txt}"
        )
    def _gaze_udp_request(self, payload: dict, timeout_s: float = 0.8) -> dict:
        """One-shot JSON request against gaze_runner.py (service mode).

        Delegates to ``Utils.perception_clients.udp_request`` to keep the
        wire format colocated with the VLM client and reusable from the
        experiment driver.
        """
        from Utils.perception_clients import udp_request
        return udp_request(GAZE_SERVICE_HOST, int(GAZE_SERVICE_PORT), payload, float(timeout_s))

    # ----- Arduino / Online BCI panel -----
    def on_serial_refresh(self):
        self.cmb_serial_port.blockSignals(True)
        self.cmb_serial_port.clear()

        try:
            ports = list(serial.tools.list_ports.comports())
        except Exception as e:
            self.cmb_serial_port.blockSignals(False)
            self._append_log("Panel", f"[{self._ts()}] Error listing serial ports: {e}\n")
            self._set_led(self.lbl_arduino, "error")
            return

        if not ports:
            self.cmb_serial_port.addItem("No ports found", "")
            self.serial_port_name = ""
            self.cmb_serial_port.blockSignals(False)
            self._append_log("Panel", f"[{self._ts()}] No serial ports found\n")
            self._set_led(self.lbl_arduino, "stopped")
            return

        for p in ports:
            desc = p.description or "n/a"
            text = f"{p.device} ({desc})"
            self.cmb_serial_port.addItem(text, p.device)

        idx = -1
        if self.serial_port_name:
            idx = self.cmb_serial_port.findData(self.serial_port_name)
        if idx < 0:
            idx = self.cmb_serial_port.findData(ARDUINO_PORT)
        if idx < 0:
            idx = 0

        self.cmb_serial_port.setCurrentIndex(idx)
        self.serial_port_name = self.cmb_serial_port.currentData() or ""
        self.cmb_serial_port.blockSignals(False)

        self._append_log("Panel", f"[{self._ts()}] Serial ports refreshed. Selected: {self.serial_port_name or 'None'}\n")

        self._set_cmds_for_mode_and_driver()

    def on_serial_port_changed(self, index: int):
        device = self.cmb_serial_port.itemData(index)
        self.serial_port_name = device or ""
        self._append_log("Panel", f"[{self._ts()}] Serial port set to: {self.serial_port_name}\n")
        # New port not yet validated — clear any prior pass/fail signal.
        self._set_led(self.lbl_arduino, "stopped")
        self._set_cmds_for_mode_and_driver()

    def _serial_baud_int(self) -> Optional[int]:
        """Return the configured baud rate as int, or None on parse failure.
        Source of truth is ``self.serial_baudrate`` (loaded from config and
        editable from the Runtime config tab)."""
        try:
            return int(str(self.serial_baudrate).strip())
        except (TypeError, ValueError):
            return None

    def on_serial_test(self):
        port = self.serial_port_name or self.cmb_serial_port.currentData()
        if not port:
            self._append_log("Panel", f"[{self._ts()}] Serial test: no port selected\n")
            self._set_led(self.lbl_arduino, "error")
            QMessageBox.information(self, "Serial test", "No serial port selected.")
            return

        baud = self._serial_baud_int()
        if baud is None:
            self._append_log("Panel", f"[{self._ts()}] Serial test: invalid baudrate {self.serial_baudrate!r}\n")
            self._set_led(self.lbl_arduino, "error")
            QMessageBox.warning(self, "Serial test", "Invalid baudrate (set ARDUINO_BAUD in Runtime config).")
            return

        self._set_led(self.lbl_arduino, "starting")
        try:
            ser = serial.Serial(port, baudrate=baud, timeout=1)
            time.sleep(2)
            if ser.is_open:
                self.serial_port_name = port
                self._append_log("Panel", f"[{self._ts()}] Serial test OK on {port} @ {baud}\n")
                ser.close()
                self._set_led(self.lbl_arduino, "running")
                self._set_cmds_for_mode_and_driver()
            else:
                self._append_log("Panel", f"[{self._ts()}] Serial test FAILED (not open)\n")
                self._set_led(self.lbl_arduino, "error")
        except Exception as e:
            self._append_log("Panel", f"[{self._ts()}] Serial test ERROR: {e}\n")
            self._set_led(self.lbl_arduino, "error")
            QMessageBox.warning(self, "Serial test", f"Error opening {port}:\n{e}")

    def _send_arduino_manual_value(self, value: str):
        port = self.serial_port_name or self.cmb_serial_port.currentData()
        if not port:
            self._append_log("Panel", f"[{self._ts()}] Arduino send: no port selected\n")
            self._set_led(self.lbl_arduino, "error")
            QMessageBox.information(self, "Arduino manual test", "No serial port selected.")
            return

        baud = self._serial_baud_int()
        if baud is None:
            self._append_log("Panel", f"[{self._ts()}] Arduino send: invalid baudrate {self.serial_baudrate!r}\n")
            self._set_led(self.lbl_arduino, "error")
            QMessageBox.warning(self, "Arduino manual test", "Invalid baudrate (set ARDUINO_BAUD in Runtime config).")
            return

        self._set_led(self.lbl_arduino, "starting")
        try:
            ser = serial.Serial(port, baudrate=baud, timeout=1)
            self._append_log("Panel", f"[{self._ts()}] Waiting for Arduino reset (2s)...\n")
            QApplication.processEvents()
            time.sleep(2)

            if not ser.is_open:
                self._append_log("Panel", f"[{self._ts()}] Arduino manual: failed to open {port}\n")
                self._set_led(self.lbl_arduino, "error")
                return

            ser.write(value.encode("ascii"))
            ser.flush()
            self._append_log("Panel", f"[{self._ts()}] Arduino manual: sent '{value}' on {port}\n")
            self._set_led(self.lbl_arduino, "running")
            ser.close()

        except Exception as e:
            self._append_log("Panel", f"[{self._ts()}] Arduino manual ERROR: {e}\n")
            self._set_led(self.lbl_arduino, "error")
            QMessageBox.warning(self, "Arduino manual test", f"Error sending '{value}' on {port}:\n{e}")

    def on_send_arduino_one(self):
        self._send_arduino_manual_value("1")

    def on_send_arduino_zero(self):
        self._send_arduino_manual_value("0")

    # ----- Harmony calibration / online control -----
    def on_refresh_calibration_libs(self):
        if not hasattr(self, "cmb_calibration_lib"):
            return

        current = self.cmb_calibration_lib.currentData()
        self.cmb_calibration_lib.clear()

        # Search for .npz libraries in ROOT
        libs = sorted(glob.glob(os.path.join(ROOT, "*.npz")))

        if not libs:
            self.cmb_calibration_lib.addItem("No calibration libraries found", "")
            self._append_log("Panel", f"[{self._ts()}] No calibration libraries (*.npz) found in {ROOT}\n")
            return

        for lib in libs:
            self.cmb_calibration_lib.addItem(os.path.basename(lib), lib)

        # Try to restore previous selection if still present
        if current:
            idx = self.cmb_calibration_lib.findData(current)
            if idx >= 0:
                self.cmb_calibration_lib.setCurrentIndex(idx)

        self._append_log("Panel", f"[{self._ts()}] Refreshed calibration libraries ({len(libs)} found)\n")

    def _get_selected_calibration_library(self) -> str:
        if not hasattr(self, "cmb_calibration_lib"):
            return ""
        return self.cmb_calibration_lib.currentData() or ""

    def on_run_harmony_calibration(self):
        if not os.path.exists(HARMONY_CALIBRATION_EXEC_PY):
            QMessageBox.warning(self, "Missing", f"Not found:\n{HARMONY_CALIBRATION_EXEC_PY}")
            return

        self._spawn_external(f'python -u "{HARMONY_CALIBRATION_EXEC_PY}"')
        self._append_log("Panel", f"[{self._ts()}] Opened harmony_calibration_exec.py\n")

    def on_run_harmony_online_control(self):
        if not os.path.exists(HARMONY_ONLINE_CONTROL_PY):
            QMessageBox.warning(self, "Missing", f"Not found:\n{HARMONY_ONLINE_CONTROL_PY}")
            return

        calib_lib = self._get_selected_calibration_library()
        if not calib_lib or not os.path.exists(calib_lib):
            QMessageBox.warning(self, "Calibration Library", "Please select a valid calibration library (.npz).")
            return

        # Assumes harmony_online_control.py takes the calibration library as a positional argument.
        # If your script expects a flag instead (for example --calib_lib), change the line below accordingly.
        self._spawn_external(f'python -u "{HARMONY_ONLINE_CONTROL_PY}" "{calib_lib}"')
        self._append_log("Panel", f"[{self._ts()}] Opened harmony_online_control.py with calibration library:\n  {calib_lib}\n")

    # ----- Driver -----
    def on_driver_start(self):
        if not (self.marker.q and self.marker.q.state() != QProcess.NotRunning):
            QMessageBox.warning(self, "Gating", "Marker not running. Start/refresh Marker first.")
            return
        if self.fes_enabled_pref:
            if not (self.fes.q and self.fes.q.state() != QProcess.NotRunning):
                QMessageBox.warning(self, "Gating", "FES is enabled but not running. Start FES first.")
                return
        self._start_proc(self.driver, self.lbl_driver, "Driver")

    def on_driver_stop(self):
        self._stop_proc(self.driver, self.lbl_driver, "Driver")

    # ----- Robot (no polling) -----
    def on_init_robot(self):
        ssh = (
            "sshpass -p 'Harmonic-03' ssh -tt root@192.168.2.1 "
            "'cd /opt/hbi/dev/bin && ./killall.sh && sleep 10 && ./run.sh'"
        )
        cmd = f'gnome-terminal -- bash -lc "{ssh}; exec bash"'
        try:
            subprocess.Popen(cmd, shell=True)
            self._set_led(self.lbl_robot_init, "starting")
            QTimer.singleShot(11_000, lambda: self._set_led(self.lbl_robot_init, "running"))
            self._append_log("Robot", f"[{self._ts()}] Robot init sequence launched\n")
        except Exception as e:
            self._set_led(self.lbl_robot_init, "error")
            self._append_log("Robot", f"[{self._ts()}] Init launch error: {e}\n")
            QMessageBox.critical(self, "Initialize Robot", f"Failed to start init sequence:\n{e}")

    def _on_robot_term_finished(self, code: int, status):
        self._set_led(self.lbl_robot, "stopped")
        self.btn_robot_start.setEnabled(True)
        self._append_log("Robot", f"[{self._ts()}] SSH terminal closed (code={code})\n")
        self.robot_term = None

    def on_robot_start(self):
        if self.mode == "Simulation":
            QMessageBox.information(self, "Simulation", "Robot disabled in Simulation mode.")
            return

        if self.mode == "MI_Bimanual":
            tool = "MI_Bimanual"
        elif self.mode == "Gaze_Tracking":
            tool = "Gaze_Tracking"
        else:
            QMessageBox.warning(self, "Robot", "No robot tool for this mode.")
            return

        if self.robot_term and self.robot_term.state() != QProcess.NotRunning:
            return

        self.robot_term = QProcess(self)
        command = (
            "sshpass -p 'Harmonic-03' ssh -tt root@192.168.2.1 "
            f"'cd /opt/hbi/dev/bin/tools && ./{tool} && exec bash'"
        )

        self.robot_term.started.connect(lambda: (
            self._set_led(self.lbl_robot, "running"),
            self.btn_robot_start.setEnabled(False),
            self._append_log("Robot", f"[{self._ts()}] SSH terminal opened for {tool}\n")
        ))
        self.robot_term.finished.connect(self._on_robot_term_finished)

        self.robot_term.setProgram("gnome-terminal")
        self.robot_term.setArguments(["--wait", "--", "bash", "-lc", command])
        self.robot_term.start()

    def on_robot_remove_overrides(self):
        try:
            res = subprocess.run(
                ["sshpass","-p","Harmonic-03","ssh","-o","StrictHostKeyChecking=no","-tt",
                 "root@192.168.2.1","cd /opt/hbi/dev/bin/tools && ./RemoveOverrides"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True
            )
            if res.stdout: self._append_log("Robot", res.stdout)
            if res.stderr: self._append_log("Robot", res.stderr)
            self._append_log("Robot", f"[{self._ts()}] RemoveOverrides rc={res.returncode}\n")
        except Exception as e:
            self._append_log("Robot", f"[{self._ts()}] RemoveOverrides error: {e}\n")
            QMessageBox.warning(self, "Robot", f"RemoveOverrides failed:\n{e}")

    # ----- External apps -----
    def on_open_labrec(self):
        if self.labrec_term and self.labrec_term.state() != QProcess.NotRunning:
            return

        self.labrec_term = QProcess(self)
        self.labrec_term.started.connect(lambda: (
            self._set_led(self.lbl_labrec, "running"),
            self._append_log("Panel", f"[{self._ts()}] LabRecorder terminal opened\n")
        ))
        def _labrec_closed(code, status):
            self._set_led(self.lbl_labrec, "stopped")
            self._append_log("Panel", f"[{self._ts()}] LabRecorder terminal closed (code={code})\n")
            self.labrec_term = None
        self.labrec_term.finished.connect(_labrec_closed)

        self.labrec_term.setProgram("gnome-terminal")
        self.labrec_term.setArguments(["--wait", "--", "bash", "-lc", "LabRecorder"])
        self.labrec_term.start()

    def on_open_eego(self):
        if self.eego_term and self.eego_term.state() != QProcess.NotRunning:
            return

        self.eego_term = QProcess(self)
        self.eego_term.started.connect(lambda: (
            self._set_led(self.lbl_eego, "running"),
            self._append_log("Panel", f"[{self._ts()}] eegoSports terminal opened\n")
        ))
        def _eego_closed(code, status):
            self._set_led(self.lbl_eego, "stopped")
            self._append_log("Panel", f"[{self._ts()}] eegoSports terminal closed (code={code})\n")
            self.eego_term = None
        self.eego_term.finished.connect(_eego_closed)

        self.eego_term.setProgram("gnome-terminal")
        self.eego_term.setArguments(["--wait", "--", "bash", "-lc", "eegoSports"])
        self.eego_term.start()

    def _on_gaze_ready_read(self, p: Proc):
        # MergedChannels → readAll() gets stdout + stderr in order
        data: QByteArray = p.q.readAll()
        if not data:
            return
        try:
            txt = bytes(data).decode("utf-8", errors="replace")
        except Exception:
            txt = "<binary>\n"
        self._append_log("Gaze", txt)
    # ---------- Process helpers ----------
    def _start_proc(self, p: Proc, led: Optional[QLabel], title: str):
        if p.cmd is None:
            QMessageBox.information(self, "Disabled", f"{p.name} is disabled for this mode.")
            return
        if p.q and p.q.state() != QProcess.NotRunning:
            return

        q = QProcess(self)

        # ✅ Gaze: merge stdout+stderr and stream like a terminal
        is_gaze = (title == "Gaze")
        if is_gaze:
            q.setProcessChannelMode(QProcess.MergedChannels)

        parts = shlex.split(p.cmd)
        q.setProgram(parts[0])
        q.setArguments(parts[1:])
        q.setWorkingDirectory(p.cwd)

        env = os.environ.copy()
        env.update(p.env)
        from PySide6.QtCore import QProcessEnvironment
        qenv = QProcessEnvironment()
        for k, v in env.items():
            qenv.insert(k, v)
        q.setProcessEnvironment(qenv)

        q.started.connect(lambda: self._on_started(p, led, title))
        q.finished.connect(lambda code, status: self._on_finished(p, led, title, code, status))
        # Without errorOccurred, a FailedToStart (e.g. program not on PATH) is
        # silent — the panel shows "start requested" with no STARTED/FINISHED.
        q.errorOccurred.connect(
            lambda err: self._append_log(
                title,
                f"[{self._ts()}] QProcess error: {err} (program={parts[0]!r})\n",
            )
        )

        if is_gaze:
            # single unified stream
            q.readyRead.connect(lambda: self._on_gaze_ready_read(p))
        else:
            q.readyReadStandardOutput.connect(lambda: self._on_stdout(p, title))
            q.readyReadStandardError.connect(lambda: self._on_stderr(p, title))

        p.out.clear()
        p.err.clear()
        p.q = q
        p.status = "starting"
        if led is not None:
            self._set_led(led, "starting")

        q.start()

    def _stop_proc(self, p: Proc, led: Optional[QLabel], title: str):
        if not p.q:
            p.status = "stopped"
            if led is not None:
                self._set_led(led, "stopped")
            return
        # Drop signal connections first so a late-fired finished/errorOccurred
        # during/after window teardown can't call back into deleted widgets.
        try:
            p.q.disconnect()
        except Exception:
            pass
        if p.q.state() != QProcess.NotRunning:
            p.q.terminate()
            if not p.q.waitForFinished(1500):
                p.q.kill(); p.q.waitForFinished(1500)
        p.status = "stopped"; p.pid = None
        if led is not None:
            self._set_led(led, "stopped")
        self._append_log(title, f"[{self._ts()}] STOPPED\n")

    def _on_started(self, p: Proc, led: Optional[QLabel], title: str):
        p.status = "running"; p.pid = p.q.processId()
        if led is not None:
            self._set_led(led, "running")
        self._append_log(title, f"[{self._ts()}] STARTED pid={p.pid} cmd={p.cmd}\n")

    def _on_finished(self, p: Proc, led: Optional[QLabel], title: str, code: int, status):
        p.pid = None
        p.status = "stopped" if code == 0 else "error"
        if led is not None:
            self._set_led(led, p.status)
        self._append_log(title, f"[{self._ts()}] FINISHED code={code}\n")

    def _on_stdout(self, p: Proc, title: str):
        data: QByteArray = p.q.readAllStandardOutput()
        chunk = bytes(data)
        if not chunk:
            return

        # For Gaze: stream append so it behaves like a terminal
        if title == "Gaze":
            try:
                txt = chunk.decode("utf-8", errors="replace")
            except Exception:
                txt = "<binary>\n"
            self._append_log("Gaze", txt)
            return

        # default behavior (unchanged for other procs)
        p.out.extend(chunk)
        self._render_combined_log(title, p)


    def _on_stderr(self, p: Proc, title: str):
        data: QByteArray = p.q.readAllStandardError()
        chunk = bytes(data)
        if not chunk:
            return

        if title == "Gaze":
            try:
                txt = chunk.decode("utf-8", errors="replace")
            except Exception:
                txt = "<binary>\n"
            # keep stderr visible but clearly marked
            self._append_log("Gaze", txt)
            return

        p.err.extend(chunk)
        self._render_combined_log(title, p)
    # ---------- Log helpers ----------
    def _on_log_target_changed(self, target: str):
        self._current_log_target = target
        if getattr(self, "_building_ui", False):
            return
        self._refresh_log_view()

    def _refresh_log_view(self):
        if not hasattr(self, "txt_logs"):
            return
        buf = self._log_buffers.get(self._current_log_target, "")
        self.txt_logs.setPlainText(buf)
        self.txt_logs.moveCursor(QTextCursor.End)
        self.txt_logs.ensureCursorVisible()

    def _spawn_external(self, cmd: str):
        quoted = cmd.replace('"', r'\"')
        full = f'gnome-terminal -- bash -lc "{quoted}; exec bash"'
        subprocess.Popen(full, shell=True)

    def _open_vlm_log_file(self, subject: str) -> None:
        """Open (or rotate to) the subject-tied VLM log file under
        ``<DATA_DIR>/sub-<SUBJECT>/vlm_logs/``. Closes any prior handle
        first so a subject change cleanly switches files. Safe to call
        before _HCFG / DATA_DIR is set — it just no-ops in that case
        and logs land only in the in-memory panel buffer.
        """
        self._close_vlm_log_file()
        if not subject or _HCFG is None:
            return
        data_dir = os.path.expanduser(getattr(_HCFG, "DATA_DIR", "") or "")
        if not data_dir:
            return
        try:
            log_dir = os.path.join(data_dir, f"sub-{subject}", "vlm_logs")
            os.makedirs(log_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(log_dir, f"vlm_panel_{ts}.log")
            fh = open(path, "a", encoding="utf-8")
            # Header — one line per panel launch / subject rotation so a
            # later reader can match the file to a config snapshot.
            fh.write(
                f"# vlm_panel log opened {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"# subject={subject}\n"
                f"# GAZE_OR_BACKEND={GAZE_OR_BACKEND}\n"
                f"# PERCEPTION_FRAME_SOURCE={PERCEPTION_FRAME_SOURCE}\n"
                f"# SERVICES_HOSTED_REMOTELY={SERVICES_HOSTED_REMOTELY}\n"
                f"# VLM_SERVICE_HOST={VLM_SERVICE_HOST}\n"
                f"# VLM_MODEL={VLM_MODEL}\n"
            )
            fh.flush()
        except OSError as e:
            # Disk full / permission denied / etc. — surface in the panel
            # buffer and keep running with file logging disabled.
            self._append_log(
                "Panel",
                f"[{self._ts()}] WARN: could not open VLM log file: {e}\n",
            )
            return
        self._vlm_log_subject = subject
        self._vlm_log_path = path
        self._vlm_log_fh = fh

    def _close_vlm_log_file(self) -> None:
        fh = self._vlm_log_fh
        if fh is not None:
            try:
                fh.write(f"# vlm_panel log closed {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                fh.close()
            except OSError:
                pass
        self._vlm_log_fh = None
        self._vlm_log_path = None
        self._vlm_log_subject = None

    def _open_relay_log_file(self, subject: str) -> None:
        """Open (or rotate to) the subject-tied frame_relay log file
        under ``<DATA_DIR>/sub-<SUBJECT>/vlm_logs/``. Mirrors
        :meth:`_open_vlm_log_file` exactly, just with a distinct
        filename prefix so the two channels don't collide. Co-located
        intentionally — relay + vlm_service are halves of one pipeline.
        """
        self._close_relay_log_file()
        if not subject or _HCFG is None:
            return
        data_dir = os.path.expanduser(getattr(_HCFG, "DATA_DIR", "") or "")
        if not data_dir:
            return
        try:
            log_dir = os.path.join(data_dir, f"sub-{subject}", "vlm_logs")
            os.makedirs(log_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(log_dir, f"frame_relay_{ts}.log")
            fh = open(path, "a", encoding="utf-8")
            fh.write(
                f"# frame_relay log opened {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"# subject={subject}\n"
                f"# FRAME_RELAY_BIND_HOST={FRAME_RELAY_BIND_HOST}\n"
                f"# FRAME_RELAY_PORT={FRAME_RELAY_PORT}\n"
                f"# FRAME_RELAY_HZ={FRAME_RELAY_HZ}\n"
                f"# FRAME_RELAY_EMBEDDED={FRAME_RELAY_EMBEDDED}\n"
                f"# NEON_COMPANION_HOST={NEON_COMPANION_HOST}\n"
                f"# PERCEPTION_FRAME_SOURCE={PERCEPTION_FRAME_SOURCE}\n"
            )
            fh.flush()
        except OSError as e:
            self._append_log(
                "Panel",
                f"[{self._ts()}] WARN: could not open relay log file: {e}\n",
            )
            return
        self._relay_log_subject = subject
        self._relay_log_path = path
        self._relay_log_fh = fh

    def _close_relay_log_file(self) -> None:
        fh = self._relay_log_fh
        if fh is not None:
            try:
                fh.write(f"# frame_relay log closed {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                fh.close()
            except OSError:
                pass
        self._relay_log_fh = None
        self._relay_log_path = None
        self._relay_log_subject = None

    def _relay_log_callback(self, line: str) -> None:
        """Shared sink for both ``Utils.frame_relay._log`` and
        ``Utils.scene_only_neon_reader._log``. Routes both to the
        "Relay" channel because reader + relay are halves of the same
        upstream pipeline; the line's source is already self-evident
        from its embedded prefix (``[frame_relay] …`` vs.
        ``[scene_only_neon_reader] …``).

        Called from worker threads (relay's pump thread and per-client
        send threads) as well as the UI thread (the reader is
        constructed synchronously from VLMSceneWidget._start_embedded_relay
        on the main thread). ``QTimer.singleShot`` is safe in both
        cases — it's the same marshalling pattern as
        :meth:`_append_log_ui` — and it prepends a ``[HH:MM:SS]``
        stamp so the in-panel buffer and the on-disk file both have
        time context. Standalone CLI usage of either module is
        unaffected because the default sink stays in place there.
        """
        stamped = f"[{self._ts()}] {line}\n"
        QTimer.singleShot(0, self, lambda: self._append_log("Relay", stamped))

    def _append_log(self, title: str, text: str):
        key = title if title in self._log_buffers else "Panel"
        self._log_buffers[key] = (self._log_buffers.get(key, "") + text)[-2_000_000:]
        if self._current_log_target == key:
            self.txt_logs.moveCursor(QTextCursor.End)
            self.txt_logs.insertPlainText(text)
            self.txt_logs.moveCursor(QTextCursor.End)
            self.txt_logs.ensureCursorVisible()
        # Tee VLM events to the subject-tied log file. Other buffers
        # stay in-memory only — see _open_vlm_log_file docstring.
        if key == "VLM" and self._vlm_log_fh is not None:
            try:
                self._vlm_log_fh.write(text)
                self._vlm_log_fh.flush()
            except OSError:
                # If the disk drops out mid-session there's nothing useful
                # to do but stop tee'ing — the panel buffer still works.
                self._close_vlm_log_file()
        # Same tee, separate file, for the frame_relay channel. Lines
        # arrive pre-stamped from _relay_log_callback so the file has
        # usable time context on its own.
        if key == "Relay" and self._relay_log_fh is not None:
            try:
                self._relay_log_fh.write(text)
                self._relay_log_fh.flush()
            except OSError:
                self._close_relay_log_file()

    def _append_log_ui(self, title: str, text: str):
        # Force execution on the Qt main thread by providing a receiver (self).
        QTimer.singleShot(0, self, lambda: self._append_log(title, text))
    def _render_combined_log(self, title: str, p: Proc):
        combined = p.out + (b"\n[stderr]\n" + p.err if p.err else b"")
        if len(combined) > 2 * 1024 * 1024:
            combined = combined[-2 * 1024 * 1024:]
        try:
            txt = combined.decode("utf-8", errors="replace")
        except Exception:
            txt = "<binary>\n"
        key = title if title in self._log_buffers else "Panel"
        self._log_buffers[key] = txt
        if self._current_log_target == key:
            self.txt_logs.setPlainText(txt)
            self.txt_logs.moveCursor(QTextCursor.End)
            self.txt_logs.ensureCursorVisible()

    @staticmethod
    def _ts() -> str:
        return time.strftime("%H:%M:%S")

    # ---------- Cheap LED maintainer for QProcess-procs ----------
    def _tick(self):
        for p, led in (
            (self.marker, self.lbl_marker),
            (self.fes, self.lbl_fes),
            (self.driver, self.lbl_driver),
            (self.gaze_service, self.lbl_gaze_service),
        ):
            if p.q and p.q.state() != QProcess.NotRunning and p.status != "error":
                p.status = "running"
            if p.q and led is not None:
                self._set_led(led, p.status)

    # ---------- Close cleanup ----------
    def closeEvent(self, event):
        # Stop the periodic poll timers up front so no new worker
        # threads can be spawned during teardown. The
        # `panel-remote-status` worker calls Signal.emit() directly
        # (the only place in the panel that bypasses
        # _append_log_ui's QTimer.singleShot guard), and once the C++
        # object is destroyed that emit() raises RuntimeError. Qt
        # auto-cancels QTimer-based slots tied to `self`, so stopping
        # the timer here closes the spawn window cleanly.
        for t_attr in (
            "_remote_status_timer", "_relay_status_timer",
            "_seg_stream_log_timer", "ui_timer",
        ):
            t = getattr(self, t_attr, None)
            if t is not None:
                try:
                    t.stop()
                except Exception:
                    pass
        # Stop the overlay reader thread before tearing down its target service,
        # otherwise QProcess teardown logs errors that try to write to widgets
        # Qt has already destroyed.
        try:
            self._on_vlm_video_disconnect()
        except Exception:
            pass
        for p, led, title in (
            (self.driver, self.lbl_driver, "Driver"),
            (self.fes,    self.lbl_fes,    "FES"),
            (self.marker, self.lbl_marker, "Marker"),
            (self.gaze_service, self.lbl_gaze_service, "Gaze"),
            (self.gaze_runner, None, "Gaze"),
            (self.vlm_service, self.lbl_compute_led, "VLM"),
        ):
            try:
                self._stop_proc(p, led, title)
            except Exception:
                pass
        # Belt-and-suspenders: same reap as on_vlm_service_stop, in case the
        # conda-launched python orphaned itself between terminate and exit.
        try:
            _kill_orphan_vlm_service()
        except Exception:
            pass
        try:
            self._close_vlm_log_file()
        except Exception:
            pass
        try:
            self._close_relay_log_file()
        except Exception:
            pass
        event.accept()

# ----------------- Entrypoint -----------------
def main():
    os.chdir(ROOT)
    app = QApplication(sys.argv)
    win = ControlPanel()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()