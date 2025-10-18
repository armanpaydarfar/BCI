#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Harmony_Bimanual — Control Panel (Simplified, no polling)
Requires: pip install PySide6 psutil

Repo layout assumed:
~/Projects/Harmony_Bimanual/
    control_panel.py
    UTIL_marker_stream.py
    ExperimentDriver_Online.py
    ExperimentDriver_Bimanual.py
    ExperimentDriver_Offline.py
    FES_listener.py
    UDPRobot.py
    STMsetup.py
    initialize_devices.sh
    config.py
"""

import os, sys, shlex, time, re, tempfile, shutil, socket, subprocess, pathlib
from dataclasses import dataclass, field
from typing import Optional, Dict

from PySide6.QtCore import Qt, QTimer, QProcess, QByteArray, QSize
from PySide6.QtGui import QAction, QClipboard, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QGridLayout, QLineEdit,
    QTextEdit, QGroupBox, QFormLayout, QMessageBox, QSplitter, QToolBar, QStyle
)

# ----------------- Paths & constants -----------------
ROOT = os.path.expanduser("~/Projects/Harmony_Bimanual")
CONFIG_PY = os.path.join(ROOT, "config.py")

MARKER_PY = os.path.join(ROOT, "UTIL_marker_stream.py")
DRIVER_ONLINE_PY = os.path.join(ROOT, "ExperimentDriver_Online.py")
DRIVER_BIMANUAL_PY = os.path.join(ROOT, "ExperimentDriver_Bimanual.py")
DRIVER_OFFLINE_PY = os.path.join(ROOT, "ExperimentDriver_Offline.py")
FES_PY = os.path.join(ROOT, "FES_listener.py")
STMSETUP_PY = os.path.join(ROOT, "STMsetup.py")
INIT_SH = os.path.join(ROOT, "initialize_devices.sh")

UDP_MARKER = ("127.0.0.1", 12345)  # readiness check (port-in-use)

# Modes choose which robot tool to launch remotely
MODES = ["Gaze_Tracking", "MI_Bimanual", "Simulation"]

# Driver choices
DRIVERS = [
    "ExperimentDriver_Online",
    "ExperimentDriver_Bimanual",
    "ExperimentDriver_Offline",
]


# ----------------- Config read/write helpers -----------------
SUBJECT_RE = re.compile(r'^(TRAINING_SUBJECT\s*=\s*)([\'"])([^\'"]+)\2\s*$', re.M)
FES_RE     = re.compile(r'^(FES_toggle\s*=\s*)([01])\s*$', re.M)

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
        # No backup copy
        os.replace(tmp.name, path)
    except Exception:
        try: os.unlink(tmp.name)
        except Exception: pass
        raise

def read_training_subject(default="PILOT007"):
    txt = read_text(CONFIG_PY)
    m = SUBJECT_RE.search(txt)
    return m.group(3) if m else default

def write_training_subject(val: str):
    txt = read_text(CONFIG_PY)
    if SUBJECT_RE.search(txt):
        # Use \g<1> to avoid \11 ambiguity
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
        # Use \g<1> to avoid \11 ambiguity when val==1
        new = FES_RE.sub(rf'\g<1>{val}', txt)
    else:
        sep = "" if (txt.endswith("\n") or txt == "") else "\n"
        new = txt + f"{sep}FES_toggle = {val}\n"
    write_atomic(CONFIG_PY, new)


# ----------------- UDP readiness probe -----------------
def _is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind((host, port))
        s.close()
        return False  # bind worked -> nobody is listening
    except OSError:
        s.close()
        return True   # address in use -> listener present

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
        self.driver_choice = DRIVERS[0]
        self.training_subject = read_training_subject()
        self.fes_enabled_pref = read_fes_toggle()

        # Procs (QProcess-managed)
        self.marker = Proc("Marker Stream", f'python -u "{MARKER_PY}"', ROOT)
        self.driver = Proc("Experimental Driver", None, ROOT)  # set by driver_choice+mode
        self.fes    = Proc("FES Listener", f'python -u "{FES_PY}"', ROOT)

        # Robot terminal (we only track the terminal, not remote PIDs)
        self.robot_term: Optional[QProcess] = None

        self.labrec_term: Optional[QProcess] = None
        self.eego_term: Optional[QProcess] = None

        # Logs
        self._log_buffers: Dict[str, str] = {"Marker": "", "FES": "", "Driver": "", "Robot": "", "Panel": ""}
        self._current_log_target = "Panel"

        # Build UI (buttons/labels defined here)
        self._build_ui()

        # Configure initial commands
        self._set_cmds_for_mode_and_driver()

        # Initialize LEDs based on preferences (no polling)
        self._set_led(self.lbl_robot_init, "stopped")
        self._set_led(self.lbl_robot, "stopped")
        self._set_led(self.lbl_marker, "stopped")
        self._set_led(self.lbl_fes, "stopped")
        self._set_led(self.lbl_driver, "stopped")
        self._set_led(self.lbl_eego, "stopped")
        # Cheap timer: keep LEDs for QProcess-managed procs in sync
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
        fd.addWidget(QLabel("Experimental Driver:"))
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

        # FES toggle + config button
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
        grid.addWidget(QLabel("<b>Experimental Driver</b>"), row, 0)
        grid.addWidget(self.lbl_driver, row, 1)
        self.btn_driver_start = QPushButton("Start")
        self.btn_driver_stop  = QPushButton("Stop")
        self.btn_driver_start.clicked.connect(self.on_driver_start)
        self.btn_driver_stop.clicked.connect(self.on_driver_stop)
        grid.addWidget(self.btn_driver_start, row, 2)
        grid.addWidget(self.btn_driver_stop, row, 3)
        row += 1

        # External apps info
        grid.addWidget(QLabel("<i>External Apps:</i> eegoSports, LabRecorder (use Initialize / buttons)"), row, 0, 1, 5)
        row += 1

        # ===== Logs Pane =====
        logw = QWidget(); split.addWidget(logw)
        vl = QVBoxLayout(logw)

        pick_row = QHBoxLayout()
        self.log_title = QLabel("Logs:")
        self.log_selector = QComboBox()
        self.log_selector.addItems(["Marker", "FES", "Driver", "Robot", "Panel"])
        self.log_selector.setCurrentText(self._current_log_target)
        self.log_selector.currentTextChanged.connect(self._on_log_target_changed)
        pick_row.addWidget(self.log_title); pick_row.addStretch(1)
        pick_row.addWidget(QLabel("View:")); pick_row.addWidget(self.log_selector)

        self.txt_logs = QTextEdit(); self.txt_logs.setReadOnly(True); self.txt_logs.setLineWrapMode(QTextEdit.NoWrap)

        vl.addLayout(pick_row)
        vl.addWidget(self.txt_logs, 1)

        # Robot Test tab
        robot_tab = QWidget(); tabs.addTab(robot_tab, "Robot Test")
        rt = QVBoxLayout(robot_tab)
        btn_open_udp_robot = QPushButton("Open UDPRobot.py (terminal)")
        btn_open_udp_robot.clicked.connect(lambda: self._spawn_external(f'python -u "{os.path.join(ROOT, "UDPRobot.py")}"'))
        rt.addWidget(btn_open_udp_robot)
        self.txt_udp_log = QTextEdit(); self.txt_udp_log.setReadOnly(True); self.txt_udp_log.setMaximumHeight(180)
        rt.addWidget(QLabel("Notes:")); rt.addWidget(self.txt_udp_log)

        self._building_ui = False
        self._refresh_log_view()

        # Disable Robot start in Simulation
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
        # Driver (local python) based on driver_choice and mode
        mode_flag = {
            "MI_Bimanual": "--mode mi_bimanual",
            "Gaze_Tracking": "--mode gaze",
            "Simulation": "--mode sim --no-robot",
        }[self.mode]

        if self.driver_choice == "ExperimentDriver_Online":
            driver_path = DRIVER_ONLINE_PY
        elif self.driver_choice == "ExperimentDriver_Bimanual":
            driver_path = DRIVER_BIMANUAL_PY
        else:  # ExperimentalDriver_Offline
            driver_path = DRIVER_OFFLINE_PY

        self.driver.cmd = f'python -u "{driver_path}" {mode_flag}'

        # Update env on local python procs
        for p in (self.marker, self.driver, self.fes):
            p.env["PYTHONUNBUFFERED"] = "1"
            p.env["TRAINING_SUBJECT"] = self.training_subject

        # Robot button state by mode
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
        for p in (self.marker, self.driver, self.fes):
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

    # ----- Driver -----
    def on_driver_start(self):
        # Gate: Marker must be ready
        if not _is_port_in_use(UDP_MARKER[1], "127.0.0.1"):
            QMessageBox.warning(self, "Gating", "Marker not ready. Start/refresh Marker first.")
            return
        # Gate: FES must be running if enabled
        if self.fes_enabled_pref:
            if not (self.fes.q and self.fes.q.state() != QProcess.NotRunning):
                QMessageBox.warning(self, "Gating", "FES is enabled but not running. Start FES first.")
                return
        self._start_proc(self.driver, self.lbl_driver, "Driver")

    def on_driver_stop(self):
        self._stop_proc(self.driver, self.lbl_driver, "Driver")

    # ----- Robot (no polling, no remote kill) -----
    def on_init_robot(self):
        """
        Initialize robot base stack in a terminal:
          killall.sh -> (LED yellow) -> sleep 10 -> run.sh -> (LED green)
        """
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
        # Terminal is gone => tool not running (for our purposes)
        self._set_led(self.lbl_robot, "stopped")
        self.btn_robot_start.setEnabled(True)
        self._append_log("Robot", f"[{self._ts()}] SSH terminal closed (code={code})\n")
        self.robot_term = None


    def on_robot_start(self):
        if self.mode == "Simulation":
            QMessageBox.information(self, "Simulation", "Robot disabled in Simulation mode.")
            return

        # Pick tool from mode
        if self.mode == "MI_Bimanual":
            tool = "MI_Bimanual"
        elif self.mode == "Gaze_Tracking":
            tool = "Gaze_Tracking"
        else:
            QMessageBox.warning(self, "Robot", "No robot tool for this mode.")
            return

        # If a terminal is already open, do nothing
        if self.robot_term and self.robot_term.state() != QProcess.NotRunning:
            return

        # Launch gnome-terminal as a QProcess so we can track its lifetime
        self.robot_term = QProcess(self)
        # gnome-terminal -- bash -lc "<SSH ... 'cd ... && ./TOOL && exec bash'>"
        command = (
            "sshpass -p 'Harmonic-03' ssh -tt root@192.168.2.1 "
            f"'cd /opt/hbi/dev/bin/tools && ./{tool} && exec bash'"
        )

        # When the terminal process starts, turn LED green + disable Start
        self.robot_term.started.connect(lambda: (
            self._set_led(self.lbl_robot, "running"),
            self.btn_robot_start.setEnabled(False),
            self._append_log("Robot", f"[{self._ts()}] SSH terminal opened for {tool}\n")
        ))

        # When the terminal closes (Ctrl-C or manual close), turn LED gray + re-enable Start
        self.robot_term.finished.connect(self._on_robot_term_finished)

        # Program + args for gnome-terminal
        self.robot_term.setProgram("gnome-terminal")
        self.robot_term.setArguments(["--wait", "--", "bash", "-lc", command])
        self.robot_term.start()

    def on_robot_remove_overrides(self):
        """Run ./RemoveOverrides on the robot (non-interactive)."""
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
        # If already open, do nothing
        if self.labrec_term and self.labrec_term.state() != QProcess.NotRunning:
            return

        self.labrec_term = QProcess(self)
        # When the terminal starts, LED -> green
        self.labrec_term.started.connect(lambda: (
            self._set_led(self.lbl_labrec, "running"),
            self._append_log("Panel", f"[{self._ts()}] LabRecorder terminal opened\n")
        ))
        # When it closes, LED -> gray
        def _labrec_closed(code, status):
            self._set_led(self.lbl_labrec, "stopped")
            self._append_log("Panel", f"[{self._ts()}] LabRecorder terminal closed (code={code})\n")
            self.labrec_term = None
        self.labrec_term.finished.connect(_labrec_closed)

        self.labrec_term.setProgram("gnome-terminal")
        # --wait keeps this QProcess alive until the terminal tab/window exits
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

    # ---------- Process helpers ----------
    def _start_proc(self, p: Proc, led: QLabel, title: str):
        if p.cmd is None:
            QMessageBox.information(self, "Disabled", f"{p.name} is disabled for this mode.")
            return
        if p.q and p.q.state() != QProcess.NotRunning:
            return
        q = QProcess(self)
        parts = shlex.split(p.cmd)
        q.setProgram(parts[0]); q.setArguments(parts[1:])
        q.setWorkingDirectory(p.cwd)
        # env
        env = os.environ.copy(); env.update(p.env)
        from PySide6.QtCore import QProcessEnvironment
        qenv = QProcessEnvironment()
        for k, v in env.items(): qenv.insert(k, v)
        q.setProcessEnvironment(qenv)
        # connect
        q.started.connect(lambda: self._on_started(p, led, title))
        q.finished.connect(lambda code, status: self._on_finished(p, led, title, code, status))
        q.readyReadStandardOutput.connect(lambda: self._on_stdout(p, title))
        q.readyReadStandardError.connect(lambda: self._on_stderr(p, title))
        # go
        p.out.clear(); p.err.clear()
        p.q = q; p.status = "starting"; self._set_led(led, "starting")
        q.start()

    def _stop_proc(self, p: Proc, led: QLabel, title: str):
        if not p.q:
            p.status = "stopped"; self._set_led(led, "stopped"); return
        if p.q.state() != QProcess.NotRunning:
            p.q.terminate()
            if not p.q.waitForFinished(1500):
                p.q.kill(); p.q.waitForFinished(1500)
        p.status = "stopped"; p.pid = None
        self._set_led(led, "stopped")
        self._append_log(title, f"[{self._ts()}] STOPPED\n")

    def _on_started(self, p: Proc, led: QLabel, title: str):
        p.status = "running"; p.pid = p.q.processId()
        self._set_led(led, "running")
        self._append_log(title, f"[{self._ts()}] STARTED pid={p.pid} cmd={p.cmd}\n")

    def _on_finished(self, p: Proc, led: QLabel, title: str, code: int, status):
        p.pid = None
        p.status = "stopped" if code == 0 else "error"
        self._set_led(led, p.status)
        self._append_log(title, f"[{self._ts()}] FINISHED code={code}\n")

    def _on_stdout(self, p: Proc, title: str):
        data: QByteArray = p.q.readAllStandardOutput()
        p.out.extend(bytes(data)); self._render_combined_log(title, p)

    def _on_stderr(self, p: Proc, title: str):
        data: QByteArray = p.q.readAllStandardError()
        p.err.extend(bytes(data)); self._render_combined_log(title, p)

    # ---------- Log helpers ----------
    def _on_log_target_changed(self, target: str):
        self._current_log_target = target
        if getattr(self, "_building_ui", False): return
        self._refresh_log_view()

    def _refresh_log_view(self):
        if not hasattr(self, "txt_logs"): return
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

    def _append_udp_log(self, line: str):
        self.txt_udp_log.moveCursor(QTextCursor.End)
        self.txt_udp_log.insertPlainText(line + "\n")
        self.txt_udp_log.moveCursor(QTextCursor.End)
        self.txt_udp_log.ensureCursorVisible()

    @staticmethod
    def _ts() -> str:
        return time.strftime("%H:%M:%S")

    # ---------- Cheap LED maintainer for QProcess-procs ----------
    def _tick(self):
        # Keep QProcess-managed LEDs in sync (Marker, FES, Driver). Robot LEDs are manual.
        for p, led in ((self.marker, self.lbl_marker),
                       (self.fes, self.lbl_fes),
                       (self.driver, self.lbl_driver)):
            if p.q and p.q.state() != QProcess.NotRunning and p.status != "error":
                p.status = "running"
            if p.q:
                self._set_led(led, p.status)

    # ---------- Close cleanup ----------
    def closeEvent(self, event):
        # Try to stop local processes
        for p, led, title in (
            (self.driver, self.lbl_driver, "Driver"),
            (self.fes,    self.lbl_fes,    "FES"),
            (self.marker, self.lbl_marker, "Marker"),
        ):
            try: self._stop_proc(p, led, title)
            except Exception: pass
        # Robot terminal just closes with the app; no remote kill
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
