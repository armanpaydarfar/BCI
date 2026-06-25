"""
panel/robot_controller.py — Robot row (Init / Start / Remove Overrides) for the
control panel.

Widget-owning collaborator following the SerialController shape: it builds the
Robot row (two status LEDs + the Init / Start / Remove-Overrides buttons) into the
panel's main grid via build_into(), owns the robot_term QProcess handle, and holds
the handlers (on_init_robot, on_robot_start, on_robot_remove_overrides,
_on_robot_term_finished, _update_robot_buttons_for_mode) — all transcribed verbatim
from ControlPanel.

The robot tool selection depends on the panel's current mode (Simulation /
MI_Bimanual / Gaze_Tracking), which stays panel state; it's read through the
get_mode getter at call time. Cross-cutting concerns are injected as callbacks
(log / set_led / timestamp) so the controller has no back-reference into the panel
beyond a QMessageBox parent.
"""

from __future__ import annotations

import subprocess
from typing import Callable, Optional

from PySide6.QtCore import QObject, QProcess, QTimer
from PySide6.QtWidgets import (
    QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton, QWidget,
)

from panel.ui_utils import _fixed_v


class RobotController(QObject):
    """Owns the Robot row and its handlers.

    Injected dependencies (behaviour-identical to the former in-class calls):
      get_mode()       — current panel mode (gates Start + tool selection)
      log(title, text) — append to the panel's log buffer
      set_led(led, st) — colour an LED for a state string
      timestamp()      — "HH:MM:SS" for log lines
    """

    def __init__(
        self,
        parent,
        *,
        get_mode: Callable[[], str],
        log: Callable[[str, str], None],
        set_led: Callable[[object, str], None],
        timestamp: Callable[[], str],
    ) -> None:
        super().__init__(parent)
        self._parent = parent
        self._get_mode = get_mode
        self._log = log
        self._set_led = set_led
        self._ts = timestamp

        self.robot_term: Optional[QProcess] = None

    def build_into(self, grid: QGridLayout, row: int) -> int:
        """Build the Robot row into the panel's main grid at ``row``; return the
        next free row. Widget tree + grid placement are identical to the former
        inline _build_ui block."""
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
        return row

    def _update_robot_buttons_for_mode(self):
        sim = (self._get_mode() == "Simulation")
        self.btn_robot_start.setEnabled(not sim)
        if sim:
            self.btn_robot_start.setToolTip("Disabled in Simulation mode.")
        else:
            self.btn_robot_start.setToolTip("Open SSH terminal running the selected robot tool.")

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
            self._log("Robot", f"[{self._ts()}] Robot init sequence launched\n")
        except Exception as e:
            self._set_led(self.lbl_robot_init, "error")
            self._log("Robot", f"[{self._ts()}] Init launch error: {e}\n")
            QMessageBox.critical(self._parent, "Initialize Robot", f"Failed to start init sequence:\n{e}")

    def _on_robot_term_finished(self, code: int, status):
        self._set_led(self.lbl_robot, "stopped")
        self.btn_robot_start.setEnabled(True)
        self._log("Robot", f"[{self._ts()}] SSH terminal closed (code={code})\n")
        self.robot_term = None

    def on_robot_start(self):
        if self._get_mode() == "Simulation":
            QMessageBox.information(self._parent, "Simulation", "Robot disabled in Simulation mode.")
            return

        if self._get_mode() == "MI_Bimanual":
            tool = "MI_Bimanual"
        elif self._get_mode() == "Gaze_Tracking":
            tool = "Gaze_Tracking"
        else:
            QMessageBox.warning(self._parent, "Robot", "No robot tool for this mode.")
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
            self._log("Robot", f"[{self._ts()}] SSH terminal opened for {tool}\n")
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
            if res.stdout: self._log("Robot", res.stdout)
            if res.stderr: self._log("Robot", res.stderr)
            self._log("Robot", f"[{self._ts()}] RemoveOverrides rc={res.returncode}\n")
        except Exception as e:
            self._log("Robot", f"[{self._ts()}] RemoveOverrides error: {e}\n")
            QMessageBox.warning(self._parent, "Robot", f"RemoveOverrides failed:\n{e}")
