"""
panel/external_tools.py — launchers for external applications (LabRecorder,
eegoSports, MNE viewer, impedance monitor, STMsetup, initialize_devices.sh).

Handler-only collaborator: unlike SerialController / DeviceLaunchersController,
these buttons live in several different panel sections (toolbar, Mode/FES and
Utilities groupboxes, the eegoSports and LabRecorder grid rows), so the controller
does NOT build a UI row — the panel keeps building those widgets and just wires
each button's `clicked` to a handler here. The handlers are transcribed verbatim
from ControlPanel.

State the controller owns: the LabRecorder / eegoSports QProcess terminal handles
(labrec_term / eego_term), previously bare panel attributes. The two status LEDs
stay panel-built (their grid rows aren't being extracted) and are reached through
injected getters. Cross-cutting concerns are injected as callbacks (spawn_external
/ log / set_led / timestamp); subject context (for the impedance launch) comes
through getters. No panel back-reference beyond a QMessageBox parent.
"""

from __future__ import annotations

import os
import subprocess
from typing import Callable, Optional

from PySide6.QtCore import QObject, QProcess
from PySide6.QtWidgets import QMessageBox


class ExternalToolsController(QObject):
    """Owns the external-app launch handlers + the LabRecorder/eegoSports
    terminal handles.

    Injected dependencies (behaviour-identical to the former in-class calls):
      init_sh / stmsetup_py — script paths for on_initialize / on_open_fes_cfg
      data_dir              — config DATA_DIR (or "") for the impedance log path
      get_subject_text()    — current cmb_subject text (impedance launch)
      get_training_subject()— current training_subject (impedance fallback)
      eego_led() / labrec_led() — return the panel-built status LEDs
      spawn_external(cmd)   — open a command in a gnome-terminal
      log(title, text)      — append to the panel's log buffer
      set_led(led, state)   — colour an LED for a state string
      timestamp()           — "HH:MM:SS" for log lines
    """

    def __init__(
        self,
        parent,
        *,
        init_sh: str,
        stmsetup_py: str,
        data_dir: str,
        get_subject_text: Callable[[], str],
        get_training_subject: Callable[[], str],
        eego_led: Callable[[], object],
        labrec_led: Callable[[], object],
        spawn_external: Callable[[str], None],
        log: Callable[[str, str], None],
        set_led: Callable[[object, str], None],
        timestamp: Callable[[], str],
    ) -> None:
        super().__init__(parent)
        self._parent = parent
        self._init_sh = init_sh
        self._stmsetup_py = stmsetup_py
        self._data_dir = data_dir
        self._get_subject_text = get_subject_text
        self._get_training_subject = get_training_subject
        self._eego_led = eego_led
        self._labrec_led = labrec_led
        self._spawn_external = spawn_external
        self._log = log
        self._set_led = set_led
        self._ts = timestamp

        # Terminal handles (formerly bare panel attributes).
        self.labrec_term: Optional[QProcess] = None
        self.eego_term: Optional[QProcess] = None

    def on_initialize(self):
        if not os.path.exists(self._init_sh):
            QMessageBox.warning(self._parent, "Missing", f"Not found:\n{self._init_sh}")
            return
        cmd = f'gnome-terminal -- bash -lc "chmod +x \\"{self._init_sh}\\"; \\"{self._init_sh}\\"; exec bash"'
        subprocess.Popen(cmd, shell=True)
        QMessageBox.information(self._parent, "Initialize", "Opened initialize_devices.sh in a new terminal.")

    def on_open_fes_cfg(self):
        if not os.path.exists(self._stmsetup_py):
            QMessageBox.warning(self._parent, "Missing", f"Not found:\n{self._stmsetup_py}")
            return
        self._spawn_external(f'python -u "{self._stmsetup_py}"')
        self._log("Panel", f"[{self._ts()}] Opened STMsetup.py\n")

    def on_open_mne_viewer(self):
        self._spawn_external('mne-lsl viewer')
        self._log("Panel", f"[{self._ts()}] Opened mne-lsl viewer\n")

    def on_open_impedance_monitor(self):
        sub = self._get_subject_text() or self._get_training_subject()
        data_dir = self._data_dir
        if data_dir and sub:
            cmd = f'impedance-monitor --mode live --cap ca209 --subject "{sub}" --data-dir "{data_dir}"'
        else:
            # Fall back — the tool will default to ~/impedance_logs/
            cmd = 'impedance-monitor --mode live --cap ca209'
        self._spawn_external(cmd)
        log_dir = os.path.join(data_dir, f"sub-{sub}", "impedance_logs") if data_dir and sub else "~/impedance_logs"
        self._log("Panel", f"[{self._ts()}] Opened impedance monitor (logs → {log_dir})\n")

    def on_open_labrec(self):
        if self.labrec_term and self.labrec_term.state() != QProcess.NotRunning:
            return

        self.labrec_term = QProcess(self)
        self.labrec_term.started.connect(lambda: (
            self._set_led(self._labrec_led(), "running"),
            self._log("Panel", f"[{self._ts()}] LabRecorder terminal opened\n")
        ))
        def _labrec_closed(code, status):
            self._set_led(self._labrec_led(), "stopped")
            self._log("Panel", f"[{self._ts()}] LabRecorder terminal closed (code={code})\n")
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
            self._set_led(self._eego_led(), "running"),
            self._log("Panel", f"[{self._ts()}] eegoSports terminal opened\n")
        ))
        def _eego_closed(code, status):
            self._set_led(self._eego_led(), "stopped")
            self._log("Panel", f"[{self._ts()}] eegoSports terminal closed (code={code})\n")
            self.eego_term = None
        self.eego_term.finished.connect(_eego_closed)

        self.eego_term.setProgram("gnome-terminal")
        self.eego_term.setArguments(["--wait", "--", "bash", "-lc", "eegoSports"])
        self.eego_term.start()
