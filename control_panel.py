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

import os, sys, time, subprocess

from typing import Dict

from PySide6.QtCore import QTimer, QProcess, QSize
from PySide6.QtGui import QAction, QClipboard, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QGridLayout,
    QTextEdit, QGroupBox, QMessageBox, QSplitter, QToolBar, QStyle,
)

from panel.process_manager import Proc, ProcessManager
from panel.config_io import (
    read_simulation_mode, write_simulation_mode,
    read_training_subject, write_training_subject,
    read_fes_toggle, write_fes_toggle,
)

from panel.serial_controller import SerialController
from panel.device_launchers import DeviceLaunchersController
from panel.external_tools import ExternalToolsController
from panel.robot_controller import RobotController
from panel.calibration_controller import CalibrationController
from panel.gaze_controller import GazeController
from panel.vlm_controller import VlmController
from panel.runtime_config_controller import RuntimeConfigController
from panel.log_file_controller import LogFileController
from panel.constants import *  # noqa: F401,F403 — module-level paths + config-derived globals (see panel/constants.py __all__)
# netutils' import-time block runs the LAN / Tailscale IP discovery + report
# prints, exactly as they did at the top of this module before the extraction.
from panel.netutils import (  # noqa: F401 — _LAN_IP/_TS_IP re-exported for any external reader
    _sleep_inhibit, _kill_orphan_vlm_service, _is_port_in_use,
    _marker_udp_port, _LAN_IP, _TS_IP,
)

# ----------------- Process model -----------------
# ----------------- Main Window -----------------
class ControlPanel(QMainWindow):
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

        # Arduino / serial controls — SerialController owns its UI row, widgets,
        # state (port + baud) and handlers; built into the grid in _build_ui.
        self.serial = SerialController(
            self,
            log=self._append_log,
            set_led=self._set_led,
            timestamp=self._ts,
            refresh_cmds=self._set_cmds_for_mode_and_driver,
        )

        # Procs (QProcess-managed). ProcessManager owns their QProcess lifecycle
        # and reports status back through the panel's log/LED callbacks.
        self.procs = ProcessManager(
            self,
            log=self._append_log,
            set_led=self._set_led,
            render_combined=self._render_combined_log,
            timestamp=self._ts,
        )
        self.marker = Proc("Marker Stream", f'python -u "{MARKER_PY}"', ROOT)
        self.driver = Proc("Experiment Driver", None, ROOT)
        self.fes    = Proc("FES Listener", f'python -u "{FES_PY}"', ROOT)

        # Marker / FES / Driver launch rows — DeviceLaunchersController owns those
        # rows' LEDs/buttons + start-stop-refresh handlers, built into the grid in
        # _build_ui. The Proc handles + ProcessManager stay on the panel (shared by
        # command wiring, subject rotation, _tick, closeEvent); driver gating reads
        # fes_enabled_pref through the getter.
        self.devices = DeviceLaunchersController(
            self,
            procs=self.procs,
            marker=self.marker,
            fes=self.fes,
            driver=self.driver,
            fes_py=FES_PY,
            get_fes_enabled=lambda: self.fes_enabled_pref,
            log=self._append_log,
            set_led=self._set_led,
            timestamp=self._ts,
        )

        # ---- Gaze procs (NEW) ----
        self.gaze_runner = Proc("Gaze Runner", None, ROOT)
        self.gaze_service = Proc("Gaze Service", None, ROOT)

        # Gaze Service row — GazeController owns the row's LED/buttons + the gaze
        # handlers (on_gaze_*, _start_gaze_service, _gaze_udp_request,
        # _format_gaze_telemetry_line), built into the grid in _build_ui. The Proc
        # handles + ProcessManager stay on the panel (shared by command wiring,
        # subject rotation, _tick, closeEvent); the off-thread Query-Telemetry
        # worker uses the marshaled log_ui sink.
        self.gaze = GazeController(
            self,
            procs=self.procs,
            gaze_runner=self.gaze_runner,
            gaze_service=self.gaze_service,
            log=self._append_log,
            log_ui=self._append_log_ui,
            set_led=self._set_led,
            timestamp=self._ts,
        )

        # ---- VLM service proc ----
        # The Proc handle stays on the panel — it's iterated by the cmd-wiring
        # (_set_cmds_for_mode_and_driver), subject-rotation (on_save_subject) and
        # closeEvent loops. VlmController is constructed below (after log_files)
        # and the handle is injected into it.
        self.vlm_service = Proc("VLM Service", None, ROOT)

        # Robot — RobotController owns the Robot row (Init / Start / Remove
        # Overrides), its LEDs/buttons + handlers and the robot_term QProcess
        # handle, built into the grid in _build_ui. Robot tool selection depends
        # on the panel's mode, read through the getter at call time.
        self.robot = RobotController(
            self,
            get_mode=lambda: self.mode,
            log=self._append_log,
            set_led=self._set_led,
            timestamp=self._ts,
        )

        # Harmony / AprilTag calibration — CalibrationController owns the
        # "Harmony calibration / online control" QGroupBox (calib-library +
        # AprilTag-calib dropdowns + the Run buttons) and its handlers, built into
        # the Robot Test tab in _build_ui. Script paths + the AprilTag rig config
        # are injected here.
        self.calibration = CalibrationController(
            self,
            root=ROOT,
            harmony_calibration_exec_py=HARMONY_CALIBRATION_EXEC_PY,
            harmony_online_control_py=HARMONY_ONLINE_CONTROL_PY,
            apriltag_calibrate_py=APRILTAG_CALIBRATE_PY,
            apriltag_control_test_py=APRILTAG_CONTROL_TEST_PY,
            hcfg=_HCFG,
            spawn_external=self._spawn_external,
            log=self._append_log,
            timestamp=self._ts,
        )

        # External-app launchers — ExternalToolsController owns the LabRecorder /
        # eegoSports / MNE / impedance / STMsetup / initialize handlers and the
        # labrec_term / eego_term QProcess handles. Its buttons live in several
        # panel sections, so it builds no UI row; the panel wires each button's
        # clicked to a handler here. The two status LEDs stay panel-built and are
        # reached through getters.
        _data_dir = os.path.expanduser(getattr(_HCFG, "DATA_DIR", "") or "") if _HCFG else ""
        self.external_tools = ExternalToolsController(
            self,
            init_sh=INIT_SH,
            stmsetup_py=STMSETUP_PY,
            data_dir=_data_dir,
            get_subject_text=lambda: (self.cmb_subject.currentText().strip() if hasattr(self, "cmb_subject") else ""),
            get_training_subject=lambda: self.training_subject,
            eego_led=lambda: self.lbl_eego,
            labrec_led=lambda: self.lbl_labrec,
            spawn_external=self._spawn_external,
            log=self._append_log,
            set_led=self._set_led,
            timestamp=self._ts,
        )

        # Runtime-config / ErrP-config editor tabs + the Model-training box —
        # RuntimeConfigController owns the rc_*/errp_*/training-* widgets and the
        # config read/write + training-launch handlers. The two config tabs are
        # built via build_tabs(tabs); the training box (Robot Test tab) is returned
        # by build_training_box(). The live subject is read lazily so the launcher
        # picks up whichever subject is selected at click time.
        self.runtime_config = RuntimeConfigController(
            self,
            spawn_external=self._spawn_external,
            log=self._append_log,
            get_subject_text=lambda: (self.cmb_subject.currentText() if hasattr(self, "cmb_subject") else ""),
            get_training_subject=lambda: self.training_subject,
            timestamp=self._ts,
        )

        # Logs
        self._log_buffers: Dict[str, str] = {"Marker": "", "FES": "", "Driver": "", "Gaze": "", "VLM": "", "Relay": "", "Robot": "", "Panel": ""}
        self._current_log_target = "Panel"
        # Subject-tied on-disk log files (VLM-panel + frame-relay channels) —
        # LogFileController owns the two file handles + their open/close/rotate
        # logic and the relay log callback. Constructed here (before the open
        # calls) so its handles exist for the _append_log tee. The handles are
        # exposed as .vlm_log_fh / .relay_log_fh so the panel's _append_log can
        # tee the "VLM" / "Relay" buffers to disk unchanged.
        self.log_files = LogFileController(
            self,
            log=self._append_log,
            timestamp=self._ts,
        )
        self.log_files.open_vlm(self.training_subject)
        self.log_files.open_relay(self.training_subject)
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
            _set_relay_log_cb(self.log_files.relay_callback)
        except Exception as e:
            self._append_log(
                "Panel",
                f"[{self._ts()}] WARN: could not install relay log callback: {e}\n",
            )
        try:
            from Utils.scene_only_neon_reader import set_log_callback as _set_reader_log_cb
            _set_reader_log_cb(self.log_files.relay_callback)
        except Exception as e:
            self._append_log(
                "Panel",
                f"[{self._ts()}] WARN: could not install neon-reader log callback: {e}\n",
            )

        # VLM / perception subsystem — VlmController owns the Perception-Pipeline
        # row (3 LEDs + lifecycle/command buttons), the VLM Video tab, the
        # per-Connect verify-chain state machine, the remote/relay status pollers
        # and the VLM service commands. Constructed AFTER self.procs /
        # self.vlm_service / self.log_files exist (it injects all three) and
        # BEFORE _build_ui() (which delegates the pipeline row + video tab to it).
        # The _remote_status_received Signal + the three status QTimers live on
        # the controller now; the vlm_service Proc + ProcessManager stay on the
        # panel (shared by the cmd/subject/closeEvent loops) and are injected.
        self.vlm = VlmController(
            self,
            procs=self.procs,
            vlm_service=self.vlm_service,
            log_files=self.log_files,
            log=self._append_log,
            log_ui=self._append_log_ui,
            set_led=self._set_led,
            timestamp=self._ts,
        )

        # Build UI
        self._build_ui()

        # Configure initial commands
        self._set_cmds_for_mode_and_driver()

        # Initialize LEDs
        self._set_led(self.robot.lbl_robot_init, "stopped")
        self._set_led(self.robot.lbl_robot, "stopped")
        self._set_led(self.devices.lbl_marker, "stopped")
        self._set_led(self.devices.lbl_fes, "stopped")
        self._set_led(self.devices.lbl_driver, "stopped")
        self._set_led(self.lbl_eego, "stopped")
        self._set_led(self.lbl_labrec, "stopped")
        self._set_led(self.gaze.lbl_gaze_service, "stopped")
        self._set_led(self.vlm.lbl_compute_led, "stopped")
        self._set_led(self.serial.lbl_arduino, "stopped")

        # When services are hosted remotely (Linux operator panel pointed at
        # a Windows GPU host) the start/stop buttons can't drive local
        # processes. Disable them and stand up a remote-status timer (the
        # status timers themselves are created + started in VlmController).
        if SERVICES_HOSTED_REMOTELY:
            self.vlm.configure_remote_services_ui()

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
        act_init.triggered.connect(self.external_tools.on_initialize)
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
        btn_fes_cfg.clicked.connect(self.external_tools.on_open_fes_cfg)
        ff.addWidget(self.chk_fes); ff.addWidget(btn_fes_cfg)
        top.addWidget(gb_fes)

        # Utilities
        gb_utils = QGroupBox("Utilities"); fu = QHBoxLayout(gb_utils)
        self.btn_mne = QPushButton("MNE Viewer")
        self.btn_mne.setToolTip("Open MNE-LSL viewer")
        self.btn_mne.clicked.connect(self.external_tools.on_open_mne_viewer)
        fu.addWidget(self.btn_mne)
        self.btn_impedance = QPushButton("Impedance")
        self.btn_impedance.setToolTip("Open impedance monitor")
        self.btn_impedance.clicked.connect(self.external_tools.on_open_impedance_monitor)
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
        # ===== Robot ===== (RobotController owns the row, widgets + handlers)
        row = self.robot.build_into(grid, row)

        # eegoSports
        self.lbl_eego = QLabel("●"); self._set_led(self.lbl_eego, "stopped")
        grid.addWidget(QLabel("<b>eegoSports</b>"), row, 0)
        grid.addWidget(self.lbl_eego, row, 1)
        btn_eego = QPushButton("Open eegoSports")
        btn_eego.clicked.connect(self.external_tools.on_open_eego)
        grid.addWidget(btn_eego, row, 2)
        row += 1

        # ===== Marker + FES ===== (DeviceLaunchersController owns the rows + handlers)
        row = self.devices.build_marker_fes_into(grid, row)

        # ===== LabRecorder =====
        self.lbl_labrec = QLabel("●"); self._set_led(self.lbl_labrec, "stopped")
        grid.addWidget(QLabel("<b>LabRecorder</b>"), row, 0)
        grid.addWidget(self.lbl_labrec, row, 1)
        btn_labrec = QPushButton("Open LabRecorder")
        btn_labrec.clicked.connect(self.external_tools.on_open_labrec)
        grid.addWidget(btn_labrec, row, 2)
        row += 1

        # ===== Gaze Service ===== (GazeController owns the row, widgets + handlers)
        row = self.gaze.build_into(grid, row)

        # ===== Perception Pipeline ===== (VlmController owns the row + handlers)
        row = self.vlm.build_pipeline_row_into(grid, row)

        # ===== Arduino ===== (SerialController owns the row, widgets + handlers)
        row = self.serial.build_into(grid, row)

        # ===== Driver ===== (DeviceLaunchersController owns the row + handlers)
        row = self.devices.build_driver_into(grid, row)

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

        # Harmony / AprilTag calibration QGroupBox (CalibrationController owns it).
        self.calibration.build_into(rt)

        # Model-training box (RuntimeConfigController owns its widgets + handlers).
        rt.addWidget(self.runtime_config.build_training_box())

        self.txt_udp_log = QTextEdit()
        self.txt_udp_log.setReadOnly(True)
        self.txt_udp_log.setMaximumHeight(140)
        rt.addWidget(QLabel("Notes:"))
        rt.addWidget(self.txt_udp_log)

        self.vlm.build_vlm_video_tab(tabs)
        self.runtime_config.build_tabs(tabs)

        # Initial serial refresh
        self.serial.on_serial_refresh()
        self.calibration.on_refresh_calibration_libs()
        self.calibration.on_refresh_apriltag_calibs()
        self.runtime_config.on_refresh_training_data_list()

        self._building_ui = False
        self._refresh_log_view()

        self.robot._update_robot_buttons_for_mode()
        self._apply_backend_visibility()

    def _apply_backend_visibility(self) -> None:
        """Hide rows that are inert for the current GAZE_OR_BACKEND.

        legacy → vlm_service is not running, so the VLM rows + remote-intake
        badge would only display dead controls. vlm → gaze_runner is not
        running, so the Gaze service rows have nothing to drive. Frame Relay
        is shared by both backends in remote mode and is gated separately
        by the perception-source flag, not by this method."""
        is_vlm = (GAZE_OR_BACKEND == "vlm")
        for w in getattr(self.gaze, "gaze_row_widgets", ()):
            w.setVisible(not is_vlm)
        for w in getattr(self.vlm, "vlm_row_widgets", ()):
            w.setVisible(is_vlm)

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
            p.env["ARDUINO_PORT"]      = self.serial.serial_port_name or ""
            p.env["ARDUINO_BAUD"]      = str(self.serial.serial_baudrate)

        self.robot._update_robot_buttons_for_mode()

    # ---------- Actions ----------
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
            self.log_files.open_vlm(val)
            self.log_files.open_relay(val)
        self.runtime_config.on_refresh_training_data_list()

    def on_copy_subject(self):
        val = self.cmb_subject.currentText().strip()
        QApplication.clipboard().setText(val, QClipboard.Clipboard)
        self._append_log("Panel", f"[{self._ts()}] Copied subject: {val}\n")

    def on_fes_pref_toggled(self, checked: bool):
        self.fes_enabled_pref = 1 if checked else 0
        write_fes_toggle(self.fes_enabled_pref)
        self._append_log("Panel", f"[{self._ts()}] FES_toggle set to {self.fes_enabled_pref}\n")

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
        # Tee VLM events to the subject-tied log file (LogFileController owns
        # the handle). Other buffers stay in-memory only — see
        # LogFileController.open_vlm docstring.
        if key == "VLM" and self.log_files.vlm_log_fh is not None:
            try:
                self.log_files.vlm_log_fh.write(text)
                self.log_files.vlm_log_fh.flush()
            except OSError:
                # If the disk drops out mid-session there's nothing useful
                # to do but stop tee'ing — the panel buffer still works.
                self.log_files.close_vlm()
        # Same tee, separate file, for the frame_relay channel. Lines
        # arrive pre-stamped from LogFileController.relay_callback so the file
        # has usable time context on its own.
        if key == "Relay" and self.log_files.relay_log_fh is not None:
            try:
                self.log_files.relay_log_fh.write(text)
                self.log_files.relay_log_fh.flush()
            except OSError:
                self.log_files.close_relay()

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
            (self.marker, self.devices.lbl_marker),
            (self.fes, self.devices.lbl_fes),
            (self.driver, self.devices.lbl_driver),
            (self.gaze_service, self.gaze.lbl_gaze_service),
        ):
            if p.q and p.q.state() != QProcess.NotRunning and p.status != "error":
                p.status = "running"
            if p.q and led is not None:
                self._set_led(led, p.status)

    # ---------- Close cleanup ----------
    def closeEvent(self, event):
        # Stop the panel's own 400 ms _tick timer up front. The VLM-specific
        # teardown (the three status poll timers + the overlay-reader
        # disconnect) is owned by VlmController.shutdown() — it stops those
        # timers before any new worker thread can be spawned during teardown
        # (the `panel-remote-status` worker calls Signal.emit() directly, which
        # raises RuntimeError once the C++ object is destroyed) and tears down
        # the local pipeline before its target service is killed below.
        if self.ui_timer is not None:
            try:
                self.ui_timer.stop()
            except Exception:
                pass
        try:
            self.vlm.shutdown()
        except Exception:
            pass
        for p, led, title in (
            (self.driver, self.devices.lbl_driver, "Driver"),
            (self.fes,    self.devices.lbl_fes,    "FES"),
            (self.marker, self.devices.lbl_marker, "Marker"),
            (self.gaze_service, self.gaze.lbl_gaze_service, "Gaze"),
            (self.gaze_runner, None, "Gaze"),
            (self.vlm_service, self.vlm.lbl_compute_led, "VLM"),
        ):
            try:
                self.procs.stop(p, led, title)
            except Exception:
                pass
        # Belt-and-suspenders: same reap as on_vlm_service_stop, in case the
        # conda-launched python orphaned itself between terminate and exit.
        try:
            _kill_orphan_vlm_service()
        except Exception:
            pass
        try:
            self.log_files.close_vlm()
        except Exception:
            pass
        try:
            self.log_files.close_relay()
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