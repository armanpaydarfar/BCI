"""
panel/process_manager.py — QProcess subprocess lifecycle for the control panel.

Behaviour-preserving extraction of ControlPanel's former _start_proc / _stop_proc
/ _on_started / _on_finished / _on_stdout / _on_stderr / _on_gaze_ready_read
methods and the Proc dataclass.

ProcessManager owns the QProcess lifecycle for each Proc the panel holds (marker,
driver, fes, gaze, vlm_service, frame_relay). It reports status to the UI through
injected callbacks (log / set_led / render_combined / timestamp) so it has no
dependency on the widget tree beyond a parent widget used only as the QMessageBox
parent. This keeps the subprocess machinery independently unit-testable: a test
constructs it with stub callbacks and never needs the full ControlPanel.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from PySide6.QtCore import QByteArray, QObject, QProcess, QProcessEnvironment
from PySide6.QtWidgets import QLabel, QMessageBox, QWidget


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


class ProcessManager(QObject):
    """Owns QProcess lifecycle for the panel's subprocesses.

    Callbacks (injected by the panel, all behaviour-identical to the former
    in-class calls):
      log(title, text)            — append to the panel's per-title log buffer
      set_led(led, state)         — set an LED QLabel's colour for a state string
      render_combined(title, proc)— re-render the combined stdout/stderr log view
      timestamp()                 — "HH:MM:SS" string for log lines
    """

    def __init__(
        self,
        parent: QWidget,
        *,
        log: Callable[[str, str], None],
        set_led: Callable[[Optional[QLabel], str], None],
        render_combined: Callable[[str, "Proc"], None],
        timestamp: Callable[[], str],
    ) -> None:
        super().__init__(parent)
        self._ui_parent = parent
        self._log = log
        self._set_led = set_led
        self._render_combined = render_combined
        self._ts = timestamp

    # ---------- lifecycle ----------
    def start(self, p: Proc, led: Optional[QLabel], title: str):
        if p.cmd is None:
            QMessageBox.information(self._ui_parent, "Disabled", f"{p.name} is disabled for this mode.")
            return
        if p.q and p.q.state() != QProcess.NotRunning:
            return

        q = QProcess(self)

        # ✅ Gaze: merge stdout+stderr and stream like a terminal
        is_gaze = (title == "Gaze")
        if is_gaze:
            q.setProcessChannelMode(QProcess.MergedChannels)

        # shlex defaults to POSIX mode, which treats backslashes as escape
        # characters and so mangles Windows paths — the python.exe in
        # sys.executable becomes a non-existent path, QProcess FailedToStart,
        # and no STARTED is ever logged. Gate on platform: Linux (the realtime
        # host) keeps exact POSIX splitting; Windows (the dev box) parses
        # backslash paths correctly.
        parts = shlex.split(p.cmd, posix=(os.name != "nt"))
        q.setProgram(parts[0])
        q.setArguments(parts[1:])
        q.setWorkingDirectory(p.cwd)

        env = os.environ.copy()
        env.update(p.env)
        qenv = QProcessEnvironment()
        for k, v in env.items():
            qenv.insert(k, v)
        q.setProcessEnvironment(qenv)

        q.started.connect(lambda: self._on_started(p, led, title))
        q.finished.connect(lambda code, status: self._on_finished(p, led, title, code, status))
        # Without errorOccurred, a FailedToStart (e.g. program not on PATH) is
        # silent — the panel shows "start requested" with no STARTED/FINISHED.
        q.errorOccurred.connect(
            lambda err: self._log(
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

    def stop(self, p: Proc, led: Optional[QLabel], title: str):
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
        self._log(title, f"[{self._ts()}] STOPPED\n")

    # ---------- QProcess slots ----------
    def _on_started(self, p: Proc, led: Optional[QLabel], title: str):
        p.status = "running"; p.pid = p.q.processId()
        if led is not None:
            self._set_led(led, "running")
        self._log(title, f"[{self._ts()}] STARTED pid={p.pid} cmd={p.cmd}\n")

    def _on_finished(self, p: Proc, led: Optional[QLabel], title: str, code: int, status):
        p.pid = None
        p.status = "stopped" if code == 0 else "error"
        if led is not None:
            self._set_led(led, p.status)
        self._log(title, f"[{self._ts()}] FINISHED code={code}\n")

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
            self._log("Gaze", txt)
            return

        # default behavior (unchanged for other procs)
        p.out.extend(chunk)
        self._render_combined(title, p)

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
            self._log("Gaze", txt)
            return

        p.err.extend(chunk)
        self._render_combined(title, p)

    def _on_gaze_ready_read(self, p: Proc):
        # MergedChannels → readAll() gets stdout + stderr in order
        data: QByteArray = p.q.readAll()
        if not data:
            return
        try:
            txt = bytes(data).decode("utf-8", errors="replace")
        except Exception:
            txt = "<binary>\n"
        self._log("Gaze", txt)
