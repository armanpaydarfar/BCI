"""
test_process_manager.py — unit tests for panel.process_manager.ProcessManager.

This is the payoff of the composition extraction: the subprocess-lifecycle logic
that used to be buried in the 3800-line ControlPanel is now testable in isolation
with stub UI callbacks — no full panel, no widgets beyond a bare parent.

Marked slow (needs a QApplication and, for one test, spawns a real short-lived
subprocess); excluded from the ~10s fast pre-commit gate.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

pytestmark = pytest.mark.slow

from panel.process_manager import Proc, ProcessManager  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


class _Sink:
    """Records the UI callbacks ProcessManager fires, in place of the panel."""
    def __init__(self):
        self.logs = []     # (title, text)
        self.leds = []     # (led, state)
        self.renders = []  # (title, proc)

    def log(self, title, text):
        self.logs.append((title, text))

    def set_led(self, led, state):
        self.leds.append((led, state))

    def render(self, title, p):
        self.renders.append((title, p))

    def ts(self):
        return "00:00:00"


@pytest.fixture()
def pm(qapp):
    from PySide6.QtWidgets import QWidget
    parent = QWidget()
    sink = _Sink()
    mgr = ProcessManager(
        parent, log=sink.log, set_led=sink.set_led,
        render_combined=sink.render, timestamp=sink.ts,
    )
    mgr.sink = sink  # test handle
    yield mgr
    parent.deleteLater()


def test_stop_with_no_process_marks_stopped(pm):
    led = object()
    p = Proc("X", "irrelevant", ".")  # never started → p.q is None
    pm.stop(p, led, "X")
    assert p.status == "stopped"
    assert (led, "stopped") in pm.sink.leds


def test_start_with_cmd_none_is_disabled(pm, monkeypatch):
    from panel import process_manager as pmmod
    shown = []
    monkeypatch.setattr(pmmod.QMessageBox, "information", lambda *a, **k: shown.append(a))
    p = Proc("Driver", None, ".")  # cmd None → disabled branch
    pm.start(p, None, "Driver")
    assert p.q is None          # nothing was spawned
    assert shown               # the "Disabled" dialog was raised


def test_real_subprocess_lifecycle(pm, qapp):
    led = object()
    p = Proc("Test", f'{sys.executable} -c "print(1)"', str(ROOT))
    pm.start(p, led, "Test")
    # Synchronous side effects of start(): start() sets status "starting" then
    # calls q.start(). On Windows the QProcess `started` signal can fire before
    # start() returns (the process launches faster than the signal is deferred),
    # flipping status to "running" — so accept either. The "starting" LED is
    # still recorded synchronously (before q.start()), so the transition intent
    # is verified there regardless of the race.
    assert p.q is not None
    assert p.status in ("starting", "running")
    assert (led, "starting") in pm.sink.leds
    # Drive the process to completion and flush queued Qt slot calls.
    p.q.waitForStarted(3000)
    p.q.waitForFinished(3000)
    qapp.processEvents()
    log_text = "".join(t for (_, t) in pm.sink.logs)
    assert "STARTED" in log_text
    assert "FINISHED" in log_text
    assert p.status == "stopped"          # exit code 0
    assert (led, "stopped") in pm.sink.leds
