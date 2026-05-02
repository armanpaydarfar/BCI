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
    QListWidget,
)

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
VLM_REPO_DIR        = getattr(_HCFG, "VLM_REPO_DIR", None) if _HCFG else None
VLM_CONDA_ENV       = getattr(_HCFG, "VLM_CONDA_ENV", "harmony_vlm") if _HCFG else "harmony_vlm"
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
FRAME_RELAY_HZ        = float(getattr(_HCFG, "FRAME_RELAY_HZ", 30.0)) if _HCFG else 30.0
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


def _resolve_conda_env_python(env_name: str) -> Optional[str]:
    """Return the absolute path to a conda env's python interpreter.

    Invoking the env's python directly avoids `conda run`, which on Windows
    QProcess is unreliable (.bat resolution) and on POSIX swallows signals
    (the inner python becomes an orphan when the wrapper is killed).

    Resolution order:
      1. Sibling of the panel's own env: <sys.prefix>/../<env_name>/python(.exe)
         — fastest path; works whenever the panel and the target env share
         a parent envs/ directory (true for stock conda installs).
      2. $CONDA_PREFIX/../<env_name>/python(.exe) — same idea via env var.
      3. `conda run` shell-out — last resort, slow.
    """
    py_name = "python.exe" if _IS_WINDOWS else "python"
    py_subdir = "" if _IS_WINDOWS else "bin"

    candidates = []
    parent = os.path.dirname(sys.prefix)
    if parent:
        candidates.append(os.path.join(parent, env_name, py_subdir, py_name))
    cp = os.environ.get("CONDA_PREFIX")
    if cp:
        cp_parent = os.path.dirname(cp)
        if cp_parent:
            candidates.append(os.path.join(cp_parent, env_name, py_subdir, py_name))
    for c in candidates:
        if os.path.isfile(c):
            return c

    try:
        out = subprocess.check_output(
            ["conda", "run", "-n", env_name, "python", "-c",
             "import sys; print(sys.executable)"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return out if out and os.path.isfile(out) else None


VLM_ENV_PYTHON: Optional[str] = _resolve_conda_env_python(VLM_CONDA_ENV) if VLM_REPO_DIR else None
print(f"[control_panel] VLM_ENV_PYTHON = {VLM_ENV_PYTHON!r}", flush=True)


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

    `conda run` does not forward SIGTERM to its child, so a hard stop on the
    panel can leave the inner python alive holding the UDP port. POSIX uses
    `pkill -f`; Windows walks tasklist for python.exe with the script in its
    command line and kills with taskkill.
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


def _assign_line_re(name: str) -> re.Pattern:
    return re.compile(rf"^(\s*{re.escape(name)}\s*=\s*)([^#\n]+?)(\s*(#.*)?)\s*$", re.M)


def _read_float_key(name: str, default: float) -> float:
    txt = read_text(CONFIG_PY)
    m = _assign_line_re(name).search(txt)
    if not m:
        return default
    try:
        return float(m.group(2).strip())
    except ValueError:
        return default


def _read_int_key(name: str, default: int) -> int:
    txt = read_text(CONFIG_PY)
    m = _assign_line_re(name).search(txt)
    if not m:
        return default
    try:
        return int(float(m.group(2).strip()))
    except ValueError:
        return default


def _read_bool_key(name: str, default: bool) -> bool:
    txt = read_text(CONFIG_PY)
    m = _assign_line_re(name).search(txt)
    if not m:
        return default
    v = m.group(2).strip()
    if v == "True":
        return True
    if v == "False":
        return False
    return default


def _read_quoted_str_key(name: str, default: str) -> str:
    txt = read_text(CONFIG_PY)
    m = _assign_line_re(name).search(txt)
    if not m:
        return default
    raw = m.group(2).strip()
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    return raw


def _write_assign_rhs(name: str, rhs: str):
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
    txt = read_text(CONFIG_PY)
    if ARDUINO_PORT_RE.search(txt):
        new = ARDUINO_PORT_RE.sub(rf'\g<1>"{port}"\3', txt, count=1)
    else:
        sep = "" if (txt.endswith("\n") or txt == "") else "\n"
        new = txt + f'{sep}ARDUINO_PORT = "{port}"\n'
    write_atomic(CONFIG_PY, new)


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
        self.resize(1250, 800)
        self._remote_status_in_flight = False
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
        self._log_buffers: Dict[str, str] = {"Marker": "", "FES": "", "Driver": "", "Gaze": "", "VLM": "", "Robot": "", "Panel": ""}
        self._current_log_target = "Panel"

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
        self._set_led(self.lbl_vlm_service, "stopped")

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
        top = QHBoxLayout(); mv.addLayout(top)

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
        fd.addWidget(QLabel("Experiment Driver:"))
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
        btn_fes_cfg = QPushButton("Configure FES (STMsetup)")
        btn_fes_cfg.clicked.connect(self.on_open_fes_cfg)
        ff.addWidget(self.chk_fes); ff.addWidget(btn_fes_cfg)
        top.addWidget(gb_fes)

        # Utilities
        gb_utils = QGroupBox("Utilities"); fu = QHBoxLayout(gb_utils)
        self.btn_mne = QPushButton("Open MNE-LSL Viewer")
        self.btn_mne.clicked.connect(self.on_open_mne_viewer)
        fu.addWidget(self.btn_mne)
        self.btn_impedance = QPushButton("Impedance Monitor")
        self.btn_impedance.clicked.connect(self.on_open_impedance_monitor)
        fu.addWidget(self.btn_impedance)
        top.addWidget(gb_utils)

        # Middle: Controls + Logs
        split = QSplitter(); mv.addWidget(split, 1)
        controls = QWidget(); split.addWidget(controls)
        grid = QGridLayout(controls)

        row = 0
        # ===== Initialize Robot =====
        self.lbl_robot_init = QLabel("●"); self._set_led(self.lbl_robot_init, "stopped")
        grid.addWidget(QLabel("<b>Initialize Robot</b>"), row, 0)
        grid.addWidget(self.lbl_robot_init, row, 1)
        btn_init_robot = QPushButton("Init Robot (SSH)")
        btn_init_robot.clicked.connect(self.on_init_robot)
        grid.addWidget(btn_init_robot, row, 2, 1, 2)
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

        # ===== Gaze Service (NEW) =====
        self.lbl_gaze_service = QLabel("●"); self._set_led(self.lbl_gaze_service, "stopped")
        grid.addWidget(QLabel("<b>Gaze Service</b>"), row, 0)
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

        grid.addWidget(QLabel("<i>Telemetry:</i> view output in View: Gaze"), row, 0, 1, 2)
        grid.addWidget(self.btn_gaze_service_query, row, 2, 1, 3)
        row += 1

        # ===== VLM Service (harmony_vlm subprocess — FastSAM + Depth Pro + Gemini) =====
        self.lbl_vlm_service = QLabel("●"); self._set_led(self.lbl_vlm_service, "stopped")
        grid.addWidget(QLabel("<b>VLM Service</b>"), row, 0)
        grid.addWidget(self.lbl_vlm_service, row, 1)

        self.btn_vlm_service_start = QPushButton("Start")
        self.btn_vlm_service_stop = QPushButton("Stop")
        self.btn_vlm_service_status = QPushButton("Status")
        self.btn_vlm_service_decide = QPushButton("Decide Now")

        self.btn_vlm_service_start.clicked.connect(self.on_vlm_service_start)
        self.btn_vlm_service_stop.clicked.connect(self.on_vlm_service_stop)
        self.btn_vlm_service_status.clicked.connect(self.on_vlm_service_status)
        self.btn_vlm_service_decide.clicked.connect(self.on_vlm_service_decide)

        grid.addWidget(self.btn_vlm_service_start, row, 2)
        grid.addWidget(self.btn_vlm_service_stop, row, 3)
        grid.addWidget(self.btn_vlm_service_status, row, 4)
        row += 1

        self.btn_vlm_service_segment = QPushButton("Segment Now")
        self.btn_vlm_service_depth = QPushButton("Depth Now")
        self.btn_vlm_service_segment.clicked.connect(self.on_vlm_service_segment)
        self.btn_vlm_service_depth.clicked.connect(self.on_vlm_service_depth)

        grid.addWidget(QLabel(f"<i>Backend:</i> {GAZE_OR_BACKEND}"), row, 0, 1, 2)
        grid.addWidget(self.btn_vlm_service_decide, row, 2)
        grid.addWidget(self.btn_vlm_service_segment, row, 3)
        grid.addWidget(self.btn_vlm_service_depth, row, 4)
        row += 1

        # Sequential (two-object) decide — look at A, capture; look at B, decide pair.
        # Continuous segmentation: toggle drives FastSAM @ N Hz on the
        # service side; results stream into the overlay (VLM Video tab).
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

        grid.addWidget(QLabel("<i>Continuous:</i>"), row, 0)
        grid.addWidget(self.spin_vlm_seg_hz, row, 1)
        grid.addWidget(self.btn_vlm_seg_stream, row, 2)
        row += 1

        # ===== Frame relay status (GPU-host architecture; see SoftwareDocs/
        # GPU_Service_Host_Architecture_Plan.md §4.7/§4.8). Visible whenever
        # PERCEPTION_FRAME_SOURCE=remote (Windows consumer side) or
        # SERVICES_HOSTED_REMOTELY=True (Linux operator side); stays inert
        # in pure single-machine local mode.
        self.lbl_relay_status_led = QLabel("●"); self._set_led(self.lbl_relay_status_led, "stopped")
        self.lbl_relay_status_text = QLabel("relay: idle")
        grid.addWidget(QLabel("<b>Frame Relay</b>"), row, 0)
        grid.addWidget(self.lbl_relay_status_led, row, 1)
        grid.addWidget(self.lbl_relay_status_text, row, 2, 1, 3)
        row += 1

        # Remote VLM intake badge: only meaningful when services run remotely.
        self.lbl_remote_intake_text = QLabel("intake: --")
        grid.addWidget(QLabel("<i>Remote VLM intake:</i>"), row, 0, 1, 2)
        grid.addWidget(self.lbl_remote_intake_text, row, 2, 1, 3)
        row += 1

        self.btn_vlm_capture_first = QPushButton("Capture First")
        self.btn_vlm_decide_pair = QPushButton("Decide Pair")
        self.lbl_vlm_pair_token = QLabel("<i>snapshot:</i> (none)")
        self.btn_vlm_capture_first.clicked.connect(self.on_vlm_capture_first)
        self.btn_vlm_decide_pair.clicked.connect(self.on_vlm_decide_pair)

        grid.addWidget(QLabel("<i>Sequential:</i>"), row, 0)
        grid.addWidget(self.lbl_vlm_pair_token, row, 1, 1, 2)
        grid.addWidget(self.btn_vlm_capture_first, row, 3)
        grid.addWidget(self.btn_vlm_decide_pair, row, 4)
        row += 1

        # Robot
        self.lbl_robot = QLabel("●"); self._set_led(self.lbl_robot, "stopped")
        grid.addWidget(QLabel("<b>Robot</b>"), row, 0)
        grid.addWidget(self.lbl_robot, row, 1)
        self.btn_robot_start = QPushButton("Start (SSH terminal)")
        self.btn_robot_removeovr = QPushButton("Remove Overrides")
        self.btn_robot_start.clicked.connect(self.on_robot_start)
        self.btn_robot_removeovr.clicked.connect(self.on_robot_remove_overrides)
        grid.addWidget(self.btn_robot_start, row, 2)
        grid.addWidget(self.btn_robot_removeovr, row, 3)
        row += 1

        # ===== Driver =====
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

        grid.addWidget(QLabel("<i>External Apps:</i> eegoSports, LabRecorder (use Initialize / buttons)"), row, 0, 1, 5)
        row += 1

        # ===== Arduino / Online BCI (grouped) =====
        arduino_group = QGroupBox("Arduino / Online BCI")
        ag_outer = QVBoxLayout(arduino_group)

        conn_box = QGroupBox("Serial connection")
        conn_grid = QGridLayout(conn_box)
        conn_grid.addWidget(QLabel("Device:"), 0, 0)
        self.cmb_serial_port = QComboBox()
        self.cmb_serial_port.currentIndexChanged.connect(self.on_serial_port_changed)
        conn_grid.addWidget(self.cmb_serial_port, 0, 1)
        self.btn_serial_refresh = QPushButton("Refresh")
        self.btn_serial_refresh.setMaximumWidth(100)
        self.btn_serial_refresh.clicked.connect(self.on_serial_refresh)
        conn_grid.addWidget(self.btn_serial_refresh, 0, 2)

        conn_grid.addWidget(QLabel("Baud:"), 1, 0)
        self.le_serial_baud = QLineEdit(self.serial_baudrate)
        self.le_serial_baud.setMaximumWidth(120)
        conn_grid.addWidget(self.le_serial_baud, 1, 1)
        self.le_serial_baud.editingFinished.connect(self.on_serial_baud_changed)

        row_conn = QHBoxLayout()
        self.btn_serial_test = QPushButton("Test connection")
        self.btn_serial_test.setMaximumWidth(160)
        self.btn_serial_test.clicked.connect(self.on_serial_test)
        self.btn_save_serial_to_config = QPushButton("Save port/baud → config.py")
        self.btn_save_serial_to_config.setToolTip(
            "Writes ARDUINO_PORT and ARDUINO_BAUD in config.py (e.g. for Online Glove driver)."
        )
        self.btn_save_serial_to_config.clicked.connect(self.on_save_serial_to_config)
        row_conn.addWidget(self.btn_serial_test)
        row_conn.addWidget(self.btn_save_serial_to_config)
        row_conn.addStretch(1)
        conn_grid.addLayout(row_conn, 2, 0, 1, 3)

        self.lbl_serial_status = QLabel("Status: Not tested")
        self.lbl_serial_status.setWordWrap(True)
        conn_grid.addWidget(self.lbl_serial_status, 3, 0, 1, 3)
        ag_outer.addWidget(conn_box)

        manual_box = QGroupBox("Manual exo / actuator test")
        man_row = QHBoxLayout(manual_box)
        self.btn_send_1 = QPushButton("Send '1' (close)")
        self.btn_send_1.setMaximumWidth(140)
        self.btn_send_1.clicked.connect(self.on_send_arduino_one)
        self.btn_send_0 = QPushButton("Send '0' (open)")
        self.btn_send_0.setMaximumWidth(140)
        self.btn_send_0.clicked.connect(self.on_send_arduino_zero)
        man_row.addWidget(self.btn_send_1)
        man_row.addWidget(self.btn_send_0)
        man_row.addStretch(1)
        ag_outer.addWidget(manual_box)

        grid.addWidget(arduino_group, row, 0, 1, 5)
        row += 1

        # ===== Logs Pane =====
        logw = QWidget(); split.addWidget(logw)
        vl = QVBoxLayout(logw)

        pick_row = QHBoxLayout()
        self.log_title = QLabel("Logs:")
        self.log_selector = QComboBox()
        self.log_selector.addItems(["Marker", "FES", "Driver", "Gaze", "VLM", "Robot", "Panel"])
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

        # Initial serial refresh
        self.on_serial_refresh()
        self.on_refresh_calibration_libs()
        self.on_refresh_training_data_list()

        self._building_ui = False
        self._refresh_log_view()

        self._update_robot_buttons_for_mode()

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
                "repo_dir": VLM_REPO_DIR,
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
        vl.addWidget(self.vlm_scene_widget, 1)

    def _on_vlm_video_connect(self) -> None:
        """Auto-connect hook invoked from on_vlm_service_start. The
        VLMSceneWidget is also user-startable from its own button —
        calling start() twice is idempotent."""
        if hasattr(self, "vlm_scene_widget"):
            self.vlm_scene_widget.start()
            self._sync_backend_intake()

    def _on_vlm_video_disconnect(self) -> None:
        if hasattr(self, "vlm_scene_widget"):
            self.vlm_scene_widget.stop()

    def _sync_backend_intake(self) -> None:
        """Tell the GAZE_OR_BACKEND-active service to resume frame intake
        and the inactive one to pause. Enforces single-active-backend
        end-to-end so only one perception service consumes relay
        bandwidth at a time. Fire-and-forget on a daemon thread —
        UDP timeout would otherwise block the Qt event loop for up
        to ~1 s when a service is unreachable.
        """
        active_target = (VLM_SERVICE_HOST, VLM_SERVICE_PORT) \
            if GAZE_OR_BACKEND == "vlm" else (GAZE_SERVICE_HOST, GAZE_SERVICE_PORT)
        inactive_target = (GAZE_SERVICE_HOST, GAZE_SERVICE_PORT) \
            if GAZE_OR_BACKEND == "vlm" else (VLM_SERVICE_HOST, VLM_SERVICE_PORT)

        def _worker() -> None:
            from Utils.perception_clients import udp_request
            for target, payload in (
                (active_target,   {"cmd": "resume_intake"}),
                (inactive_target, {"cmd": "pause_intake"}),
            ):
                try:
                    udp_request(target[0], int(target[1]), payload, 0.8)
                except Exception:
                    # Silent — service unreachable / older version. The
                    # widget keeps working; only the bandwidth-saving
                    # optimisation is lost.
                    pass

        threading.Thread(target=_worker, daemon=True,
                         name="panel-backend-sync").start()

    def _build_runtime_config_tab(self, tabs: QTabWidget):
        rtc = QWidget()
        tabs.addTab(rtc, "Runtime config")
        outer = QVBoxLayout(rtc)
        outer.addWidget(QLabel(
            "<b>Edits config.py on disk.</b> Restart Marker/Driver/FES after changing simulation "
            "or network flags (<code>Utils/networking</code> caches SIMULATION_MODE at import)."
        ))
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
        self.rc_sel_errp = QCheckBox("SELECT_ERRP_CHANNELS")
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
        form.addRow(self.rc_sel_errp)
        form.addRow(self.rc_xgb_beta)
        form.addRow("TOTAL_TRIALS", self.rc_total_trials)
        form.addRow("SHAPE_MAX", self.rc_shape_max)
        form.addRow("SHAPE_MIN", self.rc_shape_min)
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
        self.rc_sel_errp.setChecked(bool(_read_01_key("SELECT_ERRP_CHANNELS", 0)))
        self.rc_xgb_beta.setChecked(bool(_read_01_key("XGB_USE_COV_BETA", 0)))
        self.rc_total_trials.setValue(_read_int_key("TOTAL_TRIALS", 10))
        self.rc_shape_max.setValue(_read_float_key("SHAPE_MAX", 0.7))
        self.rc_shape_min.setValue(_read_float_key("SHAPE_MIN", 0.5))
        self._append_log("Panel", f"[{self._ts()}] Runtime config widgets reloaded from config.py\n")

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
            _write_assign_rhs("SELECT_ERRP_CHANNELS", "1" if self.rc_sel_errp.isChecked() else "0")
            _write_assign_rhs("XGB_USE_COV_BETA", "1" if self.rc_xgb_beta.isChecked() else "0")
            _write_assign_rhs("TOTAL_TRIALS", str(self.rc_total_trials.value()))
            _write_assign_rhs("SHAPE_MAX", _fmtf(self.rc_shape_max.value()))
            _write_assign_rhs("SHAPE_MIN", _fmtf(self.rc_shape_min.value()))
        except Exception as e:
            QMessageBox.warning(self, "config.py", f"Failed to update config.py:\n{e}")
            self._append_log("Panel", f"[{self._ts()}] Runtime config apply FAILED: {e}\n")
            return
        self._append_log("Panel", f"[{self._ts()}] Runtime config written to config.py\n")
        QMessageBox.information(
            self, "Runtime config",
            "config.py updated. Restart experiment driver / marker stream if a process "
            "was already running so it reloads settings.",
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
            baud = int(self.le_serial_baud.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Serial", "Baud must be an integer.")
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
        self.training_subject = val
        write_training_subject(val)
        for p in (self.marker, self.driver, self.fes, self.gaze_runner, self.gaze_service, self.vlm_service):
            p.env["TRAINING_SUBJECT"] = self.training_subject
        self._append_log("Panel", f"[{self._ts()}] TRAINING_SUBJECT saved: {val}\n")
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
        """Disable buttons that would spawn local conda subprocesses when
        SERVICES_HOSTED_REMOTELY=True. Status queries remain functional —
        they're plain UDP and travel transparently across the LAN."""
        for btn_name in (
            "btn_vlm_service_start", "btn_vlm_service_stop",
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
            self._remote_status_received.emit(resp or {})

        threading.Thread(target=_worker, daemon=True,
                         name="panel-remote-status").start()

    def _apply_remote_status(self, resp: dict) -> None:
        """GUI-thread slot for _remote_status_received."""
        self._remote_status_in_flight = False
        if resp.get("_unreachable"):
            self._set_led(self.lbl_vlm_service, "error")
            self.lbl_remote_intake_text.setText("intake: unreachable")
            return
        ok = bool(resp.get("ok"))
        connected = bool(resp.get("frame_source_connected"))
        frames = int(resp.get("frames_received") or 0)
        src = resp.get("frame_source", "?")
        age = resp.get("frame_age_s")
        age_txt = f"{float(age):.2f}s" if isinstance(age, (int, float)) else "--"
        self._set_led(self.lbl_vlm_service, "running" if (ok and connected) else "stopped")
        self.lbl_remote_intake_text.setText(
            f"intake: src={src} connected={connected} frames={frames} age={age_txt}"
        )

    def _poll_relay_status(self) -> None:
        """2 s cadence. Reflects whether the frame relay is alive.

        When the panel hosts the relay in-process (FRAME_RELAY_EMBEDDED),
        we ask the widget directly — TCP-pinging localhost would create
        phantom client churn (each ping does connect-then-close, the
        relay's accept loop installs the dead socket, the pump pays a
        full JPEG encode + sendall before discovering the peer is gone,
        and the SDK iterator stalls behind that work → visible stutter
        in the local subscriber path).
        """
        widget = getattr(self, "vlm_scene_widget", None)
        if widget is not None and getattr(widget, "_embedded_relay", None) is not None:
            thread = getattr(widget, "_embedded_relay_thread", None)
            alive = thread is not None and thread.is_alive()
            if alive:
                self._set_led(self.lbl_relay_status_led, "running")
                self.lbl_relay_status_text.setText(
                    f"relay: in-process @ {FRAME_RELAY_BIND_HOST}:{FRAME_RELAY_PORT}"
                )
            else:
                self._set_led(self.lbl_relay_status_led, "stopped")
                self.lbl_relay_status_text.setText("relay: in-process — thread exited")
            return

        # External relay (FRAME_RELAY_EMBEDDED=False or remote host) —
        # fall back to the TCP ping.
        try:
            from Utils.perception_clients import FrameRelayController
        except Exception:
            return
        ctl = FrameRelayController(_HCFG) if _HCFG else None
        if ctl is None:
            return
        ping = ctl.ping(timeout_s=0.5)
        if ping.get("ok"):
            self._set_led(self.lbl_relay_status_led, "running")
            self.lbl_relay_status_text.setText(
                f"relay: reachable @ {ping['host']}:{ping['port']}"
            )
        else:
            self._set_led(self.lbl_relay_status_led, "stopped")
            self.lbl_relay_status_text.setText(
                f"relay: unreachable @ {ping['host']}:{ping['port']}"
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
        if not VLM_REPO_DIR or not os.path.isdir(VLM_REPO_DIR):
            QMessageBox.warning(self, "VLM repo missing", f"VLM_REPO_DIR not a dir:\n{VLM_REPO_DIR}")
            return

        # Reap any orphaned vlm_service.py left over from a previous crash or
        # incomplete stop (conda run does not forward SIGTERM to child processes).
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
        # Invoke the env's python directly. `conda run` is fragile here — on
        # Windows QProcess can't always resolve the conda.bat shim, and on POSIX
        # it doesn't forward SIGTERM, leaving the inner python as an orphan.
        if not VLM_ENV_PYTHON:
            QMessageBox.warning(
                self,
                "harmony_vlm env not found",
                f"Could not resolve python for conda env {VLM_CONDA_ENV!r}.\n"
                "Verify the env exists with `conda env list`.",
            )
            return
        py = VLM_ENV_PYTHON
        self._append_log("VLM", f"[{self._ts()}] using python: {py}\n")
        # --neon-host "" forces discover_one_device in harmony_vlm's NeonLiveReader
        # (utils/neon/reader.py:224), matching our gaze_system.py:250 pattern.
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
            f'--repo-dir "{VLM_REPO_DIR}" '
            f'--host {VLM_BIND_HOST} --port {int(VLM_SERVICE_PORT)} '
            f'--neon-host "{NEON_COMPANION_HOST}" '
            f'--model {VLM_MODEL} {device_flag} '
            f'{depth_flag} {session_arg} {remote_arg}'
        )
        self._start_proc(self.vlm_service, self.lbl_vlm_service, "VLM")
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
        # Ask vlm_service to exit gracefully before killing the conda wrapper.
        # conda run does not forward SIGTERM to child processes, so without this
        # the inner Python process survives as an orphan holding the UDP port.
        try:
            self._vlm_udp_request({"cmd": "stop"}, timeout_s=0.5)
        except Exception:
            pass
        self._stop_proc(self.vlm_service, self.lbl_vlm_service, "VLM")
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

    def on_vlm_service_segment(self):
        self._vlm_command_threaded({"cmd": "segment"}, 5.0, "segment")

    def on_vlm_seg_stream_toggled(self, checked: bool) -> None:
        hz = float(self.spin_vlm_seg_hz.value())
        self.btn_vlm_seg_stream.setText(f"Stream Seg: {'ON' if checked else 'OFF'}")
        self._vlm_command_threaded(
            {"cmd": "segment_stream", "enabled": bool(checked), "hz": hz},
            VLM_QUERY_TIMEOUT_S,
            f"segment_stream({'on' if checked else 'off'}, {hz:.1f} Hz)",
        )

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
            self.lbl_serial_status.setText("Status: Error listing ports")
            return

        if not ports:
            self.cmb_serial_port.addItem("No ports found", "")
            self.serial_port_name = ""
            self.cmb_serial_port.blockSignals(False)
            self.lbl_serial_status.setText("Status: No ports")
            self._append_log("Panel", f"[{self._ts()}] No serial ports found\n")
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

        self.lbl_serial_status.setText(f"Status: Selected {self.serial_port_name}" if self.serial_port_name else "Status: No port selected")
        self._append_log("Panel", f"[{self._ts()}] Serial ports refreshed. Selected: {self.serial_port_name or 'None'}\n")

        self._set_cmds_for_mode_and_driver()

    def on_serial_port_changed(self, index: int):
        device = self.cmb_serial_port.itemData(index)
        self.serial_port_name = device or ""
        self._append_log("Panel", f"[{self._ts()}] Serial port set to: {self.serial_port_name}\n")
        self._set_cmds_for_mode_and_driver()

    def on_serial_baud_changed(self):
        text = self.le_serial_baud.text().strip()
        if not text:
            return
        try:
            int(text)
        except ValueError:
            QMessageBox.warning(self, "Baudrate", "Baudrate must be an integer, e.g., 9600.")
            self.le_serial_baud.setText(self.serial_baudrate)
            return
        self.serial_baudrate = text
        self._append_log("Panel", f"[{self._ts()}] Serial baudrate set to: {self.serial_baudrate}\n")
        self._set_cmds_for_mode_and_driver()

    def on_serial_test(self):
        port = self.serial_port_name or self.cmb_serial_port.currentData()
        if not port:
            self.lbl_serial_status.setText("Status: No port selected")
            QMessageBox.information(self, "Serial test", "No serial port selected.")
            return

        try:
            baud = int(self.le_serial_baud.text().strip())
        except ValueError:
            self.lbl_serial_status.setText("Status: Invalid baudrate")
            QMessageBox.warning(self, "Serial test", "Invalid baudrate.")
            return

        try:
            ser = serial.Serial(port, baudrate=baud, timeout=1)
            time.sleep(2)
            if ser.is_open:
                self.lbl_serial_status.setText(f"Status: OK on {port}")
                self.serial_port_name = port
                self.serial_baudrate = str(baud)
                self._append_log("Panel", f"[{self._ts()}] Serial test OK on {port} @ {baud}\n")
                ser.close()
                self._set_cmds_for_mode_and_driver()
            else:
                self.lbl_serial_status.setText("Status: Failed to open")
                self._append_log("Panel", f"[{self._ts()}] Serial test FAILED (not open)\n")
        except Exception as e:
            self.lbl_serial_status.setText("Status: Error")
            self._append_log("Panel", f"[{self._ts()}] Serial test ERROR: {e}\n")
            QMessageBox.warning(self, "Serial test", f"Error opening {port}:\n{e}")

    def _send_arduino_manual_value(self, value: str):
        port = self.serial_port_name or self.cmb_serial_port.currentData()
        if not port:
            self.lbl_serial_status.setText("Status: No port selected")
            QMessageBox.information(self, "Arduino manual test", "No serial port selected.")
            return

        try:
            baud = int(self.le_serial_baud.text().strip())
        except ValueError:
            self.lbl_serial_status.setText("Status: Invalid baudrate")
            QMessageBox.warning(self, "Arduino manual test", "Invalid baudrate.")
            return

        try:
            ser = serial.Serial(port, baudrate=baud, timeout=1)
            self._append_log("Panel", f"[{self._ts()}] Waiting for Arduino reset (2s)...\n")
            QApplication.processEvents()
            time.sleep(2)

            if not ser.is_open:
                self.lbl_serial_status.setText("Status: Failed to open")
                self._append_log("Panel", f"[{self._ts()}] Arduino manual: failed to open {port}\n")
                return

            ser.write(value.encode("ascii"))
            ser.flush()
            self._append_log("Panel", f"[{self._ts()}] Arduino manual: sent '{value}' on {port}\n")
            self.lbl_serial_status.setText(f"Status: Sent '{value}' on {port}")
            ser.close()

        except Exception as e:
            self.lbl_serial_status.setText("Status: Error")
            self._append_log("Panel", f"[{self._ts()}] Arduino manual ERROR: {e}\n")
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

    def _append_log(self, title: str, text: str):
        key = title if title in self._log_buffers else "Panel"
        self._log_buffers[key] = (self._log_buffers.get(key, "") + text)[-2_000_000:]
        if self._current_log_target == key:
            self.txt_logs.moveCursor(QTextCursor.End)
            self.txt_logs.insertPlainText(text)
            self.txt_logs.moveCursor(QTextCursor.End)
            self.txt_logs.ensureCursorVisible()

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
            (self.vlm_service, self.lbl_vlm_service, "VLM"),
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