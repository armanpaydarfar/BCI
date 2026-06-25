"""
panel/device_launchers.py — Marker / FES / Experiment-Driver start-stop rows.

Widget-owning collaborator following the SerialController shape: it builds its own
rows into the panel's main grid (Marker + FES contiguously via
build_marker_fes_into(); the Driver row, which is anchored at the bottom of the
control column, via build_driver_into()), owns those rows' LEDs/buttons, and holds
the start/stop/refresh handlers — all transcribed verbatim from ControlPanel.

The QProcess lifecycle stays with the panel's ProcessManager + Proc handles
(self.marker / self.fes / self.driver), which are shared across the panel (command
wiring, subject rotation, _tick, closeEvent); those are injected here. The
FES-gating check in on_driver_start reads the panel's fes_enabled_pref via the
get_fes_enabled callback (the FES toggle itself lives in the top groupbox, not in
these rows). Cross-cutting concerns are injected as callbacks (log / set_led /
timestamp) so the controller has no back-reference into the panel beyond a
QMessageBox parent.
"""

from __future__ import annotations

import os
import time
from typing import Callable

from PySide6.QtCore import QObject, QProcess
from PySide6.QtWidgets import QGridLayout, QLabel, QMessageBox, QPushButton

from panel.process_manager import Proc, ProcessManager


class DeviceLaunchersController(QObject):
    """Owns the Marker / FES / Driver rows and their handlers.

    Injected dependencies (behaviour-identical to the former in-class calls):
      procs            — the panel's ProcessManager (QProcess lifecycle)
      marker/fes/driver— the Proc handles (kept on the panel, shared elsewhere)
      fes_py           — path to FES_listener.py (existence check on FES start)
      get_fes_enabled  — read the panel's fes_enabled_pref for driver gating
      log(title, text) — append to the panel's log buffer
      set_led(led, st) — colour an LED for a state string
      timestamp()      — "HH:MM:SS" for log lines
    """

    def __init__(
        self,
        parent,
        *,
        procs: ProcessManager,
        marker: Proc,
        fes: Proc,
        driver: Proc,
        fes_py: str,
        get_fes_enabled: Callable[[], object],
        log: Callable[[str, str], None],
        set_led: Callable[[object, str], None],
        timestamp: Callable[[], str],
    ) -> None:
        super().__init__(parent)
        self._parent = parent
        self.procs = procs
        self.marker = marker
        self.fes = fes
        self.driver = driver
        self._fes_py = fes_py
        self._get_fes_enabled = get_fes_enabled
        self._log = log
        self._set_led = set_led
        self._ts = timestamp

    def build_marker_fes_into(self, grid: QGridLayout, row: int) -> int:
        """Build the Marker + FES rows into the panel's main grid starting at
        ``row``; return the next free row. Widget tree + grid placement are
        identical to the former inline _build_ui block."""
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
        return row

    def build_driver_into(self, grid: QGridLayout, row: int) -> int:
        """Build the Experiment Driver row into the panel's main grid at
        ``row``; return the next free row. Widget tree + grid placement are
        identical to the former inline _build_ui block."""
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
        return row

    # ----- Marker -----
    def on_marker_start(self):
        self.procs.start(self.marker, self.lbl_marker, "Marker")
    def on_marker_stop(self):
        self.procs.stop(self.marker, self.lbl_marker, "Marker")
    def on_marker_refresh(self):
        self.on_marker_stop()
        time.sleep(0.1)
        self.on_marker_start()
        self._log("Marker", f"[{self._ts()}] Refreshed marker stream\n")

    # ----- FES -----
    def on_fes_start(self):
        if not os.path.exists(self._fes_py):
            QMessageBox.warning(self._parent, "Missing", f"Not found:\n{self._fes_py}")
            return
        self.procs.start(self.fes, self.lbl_fes, "FES")
    def on_fes_stop(self):
        self.procs.stop(self.fes, self.lbl_fes, "FES")
    def on_fes_refresh(self):
        self.on_fes_stop()
        time.sleep(0.1)
        self.on_fes_start()
        self._log("FES", f"[{self._ts()}] Refreshed FES listener\n")

    # ----- Driver -----
    def on_driver_start(self):
        if not (self.marker.q and self.marker.q.state() != QProcess.NotRunning):
            QMessageBox.warning(self._parent, "Gating", "Marker not running. Start/refresh Marker first.")
            return
        if self._get_fes_enabled():
            if not (self.fes.q and self.fes.q.state() != QProcess.NotRunning):
                QMessageBox.warning(self._parent, "Gating", "FES is enabled but not running. Start FES first.")
                return
        self.procs.start(self.driver, self.lbl_driver, "Driver")

    def on_driver_stop(self):
        self.procs.stop(self.driver, self.lbl_driver, "Driver")
