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

from PySide6.QtCore import Qt, QTimer, QProcess, QByteArray, QSize
from PySide6.QtGui import QAction, QClipboard, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QGridLayout, QLineEdit,
    QTextEdit, QGroupBox, QMessageBox, QSplitter, QToolBar, QStyle,
    QFileDialog
)

# ----------------- Paths & constants -----------------
ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PY = os.path.join(ROOT, "config.py")

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

# Telemetry (UDP) config for service
GAZE_SERVICE_HOST = "127.0.0.1"
GAZE_SERVICE_PORT = 5588
GAZE_QUERY_TIMEOUT_S = 0.8

UDP_MARKER = ("127.0.0.1", 15000)  # readiness check (port-in-use)

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
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def write_atomic(path: str, text: str):
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
    txt = read_text(CONFIG_PY)
    m = SIM_RE.search(txt)
    if not m:
        return bool(default)
    return (m.group(2) == "True")

def write_simulation_mode(val: bool):
    val_txt = "True" if val else "False"
    txt = read_text(CONFIG_PY)
    if SIM_RE.search(txt):
        new = SIM_RE.sub(rf'\g<1>{val_txt}', txt)
    else:
        sep = "" if (txt.endswith("\n") or txt == "") else "\n"
        new = txt + f"{sep}SIMULATION_MODE = {val_txt}\n"
    write_atomic(CONFIG_PY, new)

def read_training_subject(default="PILOT007"):
    txt = read_text(CONFIG_PY)
    m = SUBJECT_RE.search(txt)
    return m.group(3) if m else default

def write_training_subject(val: str):
    txt = read_text(CONFIG_PY)
    if SUBJECT_RE.search(txt):
        new = SUBJECT_RE.sub(rf'\g<1>"{val}"', txt)
    else:
        sep = "" if (txt.endswith("\n") or txt == "") else "\n"
        new = txt + f"{sep}TRAINING_SUBJECT = \"{val}\"\n"
    write_atomic(CONFIG_PY, new)

def read_fes_toggle(default=0):
    txt = read_text(CONFIG_PY)
    m = FES_RE.search(txt)
    try:
        return int(m.group(2)) if m else default
    except Exception:
        return default

def write_fes_toggle(val: int):
    val = 1 if val else 0
    txt = read_text(CONFIG_PY)
    if FES_RE.search(txt):
        new = FES_RE.sub(rf'\g<1>{val}', txt)
    else:
        sep = "" if (txt.endswith("\n") or txt == "") else "\n"
        new = txt + f"{sep}FES_toggle = {val}\n"
    write_atomic(CONFIG_PY, new)

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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Harmony Control Panel — Simplified")
        self.resize(1250, 800)

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
        self.arduino_enabled = False
        self.serial_port_name = ""
        self.serial_baudrate = "9600"
        self.classifier_model_path = ""

        # Procs (QProcess-managed)
        self.marker = Proc("Marker Stream", f'python -u "{MARKER_PY}"', ROOT)
        self.driver = Proc("Experiment Driver", None, ROOT)
        self.fes    = Proc("FES Listener", f'python -u "{FES_PY}"', ROOT)

        # ---- Gaze procs (NEW) ----
        self.gaze_runner = Proc("Gaze Runner", None, ROOT)
        self.gaze_service = Proc("Gaze Service", None, ROOT)

        # Robot terminal
        self.robot_term: Optional[QProcess] = None
        self.labrec_term: Optional[QProcess] = None
        self.eego_term: Optional[QProcess] = None

        # Logs
        self._log_buffers: Dict[str, str] = {"Marker": "", "FES": "", "Driver": "", "Gaze": "", "Robot": "", "Panel": ""}
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

        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(400)
        self.ui_timer.timeout.connect(self._tick)
        self.ui_timer.start()

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

        # ===== Arduino / Online BCI =====
        arduino_group = QGroupBox("Arduino / Online BCI")
        ag_layout = QGridLayout(arduino_group)

        ag_layout.addWidget(QLabel("Serial port:"), 0, 0)
        self.cmb_serial_port = QComboBox()
        ag_layout.addWidget(self.cmb_serial_port, 0, 1)
        self.btn_serial_refresh = QPushButton("Refresh")
        self.btn_serial_refresh.clicked.connect(self.on_serial_refresh)
        ag_layout.addWidget(self.btn_serial_refresh, 0, 2)

        ag_layout.addWidget(QLabel("Baudrate:"), 1, 0)
        self.le_serial_baud = QLineEdit(self.serial_baudrate)
        ag_layout.addWidget(self.le_serial_baud, 1, 1)
        self.le_serial_baud.editingFinished.connect(self.on_serial_baud_changed)

        self.btn_serial_test = QPushButton("Test connection")
        self.btn_serial_test.clicked.connect(self.on_serial_test)
        ag_layout.addWidget(self.btn_serial_test, 2, 0)
        self.lbl_serial_status = QLabel("Status: Not tested")
        ag_layout.addWidget(self.lbl_serial_status, 2, 1, 1, 2)

        self.chk_enable_arduino = QCheckBox("Enable Arduino control")
        self.chk_enable_arduino.setChecked(self.arduino_enabled)
        self.chk_enable_arduino.toggled.connect(self.on_arduino_toggled)
        ag_layout.addWidget(self.chk_enable_arduino, 3, 0, 1, 3)

        ag_layout.addWidget(QLabel("Classifier model (.pkl):"), 4, 0)
        self.le_model_path = QLineEdit(self.classifier_model_path)
        self.le_model_path.setReadOnly(True)
        ag_layout.addWidget(self.le_model_path, 4, 1)
        self.btn_browse_model = QPushButton("Browse...")
        self.btn_browse_model.clicked.connect(self.on_browse_model)
        ag_layout.addWidget(self.btn_browse_model, 4, 2)

        # MANUAL TEST BUTTONS
        ag_layout.addWidget(QLabel("Manual test:"), 5, 0)
        self.btn_send_1 = QPushButton("Send '1' (close exo)")
        self.btn_send_1.clicked.connect(self.on_send_arduino_one)
        ag_layout.addWidget(self.btn_send_1, 5, 1)

        self.btn_send_0 = QPushButton("Send '0' (open exo)")
        self.btn_send_0.clicked.connect(self.on_send_arduino_zero)
        ag_layout.addWidget(self.btn_send_0, 5, 2)

        grid.addWidget(arduino_group, row, 0, 1, 5)
        row += 1

        # ===== Logs Pane =====
        logw = QWidget(); split.addWidget(logw)
        vl = QVBoxLayout(logw)

        pick_row = QHBoxLayout()
        self.log_title = QLabel("Logs:")
        self.log_selector = QComboBox()
        self.log_selector.addItems(["Marker", "FES", "Driver", "Gaze", "Robot", "Panel"])
        self.log_selector.setCurrentText(self._current_log_target)
        self.log_selector.currentTextChanged.connect(self._on_log_target_changed)
        pick_row.addWidget(self.log_title); pick_row.addStretch(1)
        pick_row.addWidget(QLabel("View:")); pick_row.addWidget(self.log_selector)

        self.txt_logs = QTextEdit()
        self.txt_logs.setReadOnly(True)
        self.txt_logs.setLineWrapMode(QTextEdit.NoWrap)

        vl.addLayout(pick_row)
        vl.addWidget(self.txt_logs, 1)

        # Robot Test tab
        robot_tab = QWidget(); tabs.addTab(robot_tab, "Robot Test")
        rt = QVBoxLayout(robot_tab)

        btn_open_udp_robot = QPushButton("Open UDPRobot.py (terminal)")
        btn_open_udp_robot.clicked.connect(
            lambda: self._spawn_external(f'python -u "{os.path.join(ROOT, "UDPRobot.py")}"')
        )
        rt.addWidget(btn_open_udp_robot)

        # --- New Harmony controls on tab 2 ---
        harmony_box = QGroupBox("Harmony Calibration / Online Control")
        hb = QGridLayout(harmony_box)

        hb.addWidget(QLabel("Calibration library:"), 0, 0)

        self.cmb_calibration_lib = QComboBox()
        hb.addWidget(self.cmb_calibration_lib, 0, 1)

        self.btn_refresh_calibration_libs = QPushButton("Refresh")
        self.btn_refresh_calibration_libs.clicked.connect(self.on_refresh_calibration_libs)
        hb.addWidget(self.btn_refresh_calibration_libs, 0, 2)

        self.btn_run_harmony_calibration = QPushButton("Run harmony_calibration_exec.py")
        self.btn_run_harmony_calibration.clicked.connect(self.on_run_harmony_calibration)
        hb.addWidget(self.btn_run_harmony_calibration, 1, 0, 1, 3)

        self.btn_run_harmony_online = QPushButton("Run harmony_online_control.py")
        self.btn_run_harmony_online.clicked.connect(self.on_run_harmony_online_control)
        hb.addWidget(self.btn_run_harmony_online, 2, 0, 1, 3)

        rt.addWidget(harmony_box)

        self.txt_udp_log = QTextEdit()
        self.txt_udp_log.setReadOnly(True)
        self.txt_udp_log.setMaximumHeight(180)
        rt.addWidget(QLabel("Notes:"))
        rt.addWidget(self.txt_udp_log)

        # Initial serial refresh
        self.on_serial_refresh()
        self.on_refresh_calibration_libs()

        self._building_ui = False
        self._refresh_log_view()

        self._update_robot_buttons_for_mode()

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

        for p in (self.marker, self.driver, self.fes, self.gaze_runner, self.gaze_service):
            p.env["PYTHONUNBUFFERED"] = "1"
            p.env["TRAINING_SUBJECT"] = self.training_subject
            p.env["ARDUINO_ENABLED"]   = "1" if getattr(self, "arduino_enabled", False) else "0"
            p.env["ARDUINO_PORT"]      = getattr(self, "serial_port_name", "") or ""
            p.env["ARDUINO_BAUD"]      = str(getattr(self, "serial_baudrate", "9600"))
            p.env["BCI_MODEL_PATH"]    = getattr(self, "classifier_model_path", "") or ""

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
        for p in (self.marker, self.driver, self.fes, self.gaze_runner, self.gaze_service):
            p.env["TRAINING_SUBJECT"] = self.training_subject
        self._append_log("Panel", f"[{self._ts()}] TRAINING_SUBJECT saved: {val}\n")

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
        self.gaze_runner.cmd = f'python -u "{GAZE_RUNNER_PY}" --mode runner --display 1 --prints 1'
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
        self.gaze_service.cmd = (
            f'python -u "{GAZE_SERVICE_PY}" --mode service '
            f'--display {int(display)} --prints 1 '
            f'--host {GAZE_SERVICE_HOST} --port {int(GAZE_SERVICE_PORT)} '
            f'--udp_log 1 --udp_log_hz 50'
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
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(float(timeout_s))

        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        s.sendto(data, (GAZE_SERVICE_HOST, int(GAZE_SERVICE_PORT)))

        resp, _addr = s.recvfrom(65535)
        s.close()

        txt = resp.decode("utf-8", errors="replace").strip()
        if not txt:
            raise RuntimeError("empty UDP response from gaze service")
        return json.loads(txt)

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

    def on_arduino_toggled(self, checked: bool):
        self.arduino_enabled = bool(checked)
        state_txt = "ENABLED" if self.arduino_enabled else "DISABLED"
        self._append_log("Panel", f"[{self._ts()}] Arduino control {state_txt}\n")
        self._set_cmds_for_mode_and_driver()

    def on_browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select classifier model (.pkl)",
            ROOT,
            "Pickle files (*.pkl);;All files (*.*)"
        )
        if not path:
            return
        self.classifier_model_path = path
        self.le_model_path.setText(path)
        self._append_log("Panel", f"[{self._ts()}] Classifier model selected:\n  {path}\n")
        self._set_cmds_for_mode_and_driver()


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
        for p, led, title in (
            (self.driver, self.lbl_driver, "Driver"),
            (self.fes,    self.lbl_fes,    "FES"),
            (self.marker, self.lbl_marker, "Marker"),
            (self.gaze_service, self.lbl_gaze_service, "Gaze"),
            (self.gaze_runner, None, "Gaze"),
        ):
            try:
                self._stop_proc(p, led, title)
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