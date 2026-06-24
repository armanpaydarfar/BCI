"""
test_control_panel_construction.py — headless construction + method-surface guard
for control_panel.ControlPanel.

control_panel.py is a ~3800-line, ~150-method Qt god class that is being
decomposed into mixins. Before the decomposition it had only AST-level coverage
(test_config_contract.py). This pins two things the mixin extraction must keep:

1. **It still constructs headless.** `_build_ui()` connects Qt signals to
   `self.on_*` / `self._on_*` slots during __init__, so if a mixin extraction
   drops, renames, or misplaces a slot, construction raises AttributeError and
   this test fails — a strong, cheap regression guard for the whole refactor.
2. **The method surface is intact.** A representative method from every proposed
   mixin group must still resolve on the instance via the MRO.

Runs under the Qt 'offscreen' platform — no display, no hardware. Marked slow
because it imports control_panel (which does one-time LAN/Tailscale IP detection
at module import) and builds the full widget tree; keep it out of the
~10s fast pre-commit gate but runnable via `pytest -m slow`.
"""

from __future__ import annotations

import os

# Must be set before PySide6 is imported so Qt picks the headless platform.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

pytestmark = pytest.mark.slow

# One representative method per proposed mixin group (see the decomposition map).
# If an extraction drops or misnames any of these, this list catches it even if
# construction somehow doesn't.
_EXPECTED_METHODS = [
    # logging / status
    "_append_log", "_set_led", "_refresh_log_view",
    # process management
    "_start_proc", "_stop_proc", "_on_finished",
    # log files
    "_open_vlm_log_file", "_open_relay_log_file", "_relay_log_callback",
    # arduino / serial
    "on_serial_refresh", "on_send_arduino_one", "on_save_serial_to_config",
    # gaze controls
    "on_gaze_service_query", "_start_gaze_service", "_gaze_udp_request",
    # marker / fes / driver
    "on_marker_start", "on_fes_start", "on_driver_start",
    # robot controls
    "on_robot_start", "_update_robot_buttons_for_mode",
    # training / calibration
    "on_run_harmony_calibration", "on_run_apriltag_control_test",
    "_get_selected_apriltag_calib",
    # external tools
    "on_open_labrec", "on_open_mne_viewer",
    # config read/write
    "on_runtime_apply_config", "on_runtime_reload_config", "on_errp_config_apply",
    # mode / driver / subject
    "on_mode_changed", "on_save_subject", "_set_cmds_for_mode_and_driver",
    # vlm frame-relay state machine
    "_on_vlm_video_connect", "_on_vlm_video_disconnect", "_fire_verify_chain",
    # remote services
    "_poll_remote_status", "_apply_remote_status",
    # vlm commands
    "on_vlm_service_decide", "on_vlm_capture_first", "on_vlm_decide_pair",
    # ui build / lifecycle
    "_build_ui", "closeEvent",
]


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def panel(qapp, monkeypatch):
    import config
    monkeypatch.setattr(config, "SIMULATION_MODE", True, raising=False)
    import control_panel
    cp = control_panel.ControlPanel()
    try:
        yield cp
    finally:
        cp.close()


def test_constructs_headless(panel):
    from PySide6.QtWidgets import QMainWindow
    assert isinstance(panel, QMainWindow)
    # MRO sanity: ControlPanel is the leaf type.
    assert type(panel).__name__ == "ControlPanel"


def test_all_grouped_methods_resolve(panel):
    missing = [m for m in _EXPECTED_METHODS if not callable(getattr(panel, m, None))]
    assert not missing, f"methods missing after extraction: {missing}"
