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
    # process management now lives in panel.process_manager.ProcessManager
    # (asserted by test_process_manager_collaborator_wired below)
    # log files now live in panel.log_file_controller.LogFileController
    # (asserted by test_log_file_controller_wired below)
    # arduino / serial now lives in panel.serial_controller.SerialController
    # (asserted by test_serial_controller_wired below)
    # gaze controls now live in panel.gaze_controller.GazeController
    # (asserted by test_gaze_controller_wired below)
    # marker / fes / driver now live in panel.device_launchers.DeviceLaunchersController
    # (asserted by test_device_launchers_controller_wired below)
    # robot controls now live in panel.robot_controller.RobotController
    # (asserted by test_robot_controller_wired below)
    # harmony / AprilTag calibration now live in
    # panel.calibration_controller.CalibrationController
    # (asserted by test_calibration_controller_wired below)
    # external tools now live in panel.external_tools.ExternalToolsController
    # (asserted by test_external_tools_controller_wired below)
    # runtime-config / ErrP-config / model-training now live in
    # panel.runtime_config_controller.RuntimeConfigController
    # (asserted by test_runtime_config_controller_wired below)
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


def test_process_manager_collaborator_wired(panel):
    """Subprocess lifecycle was extracted into a ProcessManager collaborator;
    the panel must hold it and delegate through it."""
    from panel.process_manager import ProcessManager
    assert isinstance(panel.procs, ProcessManager)
    assert callable(panel.procs.start) and callable(panel.procs.stop)


def test_serial_controller_wired(panel):
    """The Arduino/serial UI row + handlers were extracted into a
    SerialController that owns its widgets; the panel holds it and the
    controller built its widgets into the grid during construction."""
    from panel.serial_controller import SerialController
    assert isinstance(panel.serial, SerialController)
    # build_into() ran during _build_ui → the controller owns its widgets.
    assert panel.serial.lbl_arduino is not None
    assert panel.serial.cmb_serial_port is not None
    for m in ("on_serial_refresh", "on_serial_test", "on_send_arduino_one",
              "on_save_serial_to_config"):
        assert callable(getattr(panel.serial, m, None)), m


def test_device_launchers_controller_wired(panel):
    """The Marker / FES / Driver launch rows + handlers were extracted into a
    DeviceLaunchersController that owns those rows' widgets; the panel holds it
    and the controller built its widgets into the grid during construction."""
    from panel.device_launchers import DeviceLaunchersController
    assert isinstance(panel.devices, DeviceLaunchersController)
    # build_marker_fes_into() + build_driver_into() ran during _build_ui → the
    # controller owns its widgets.
    for w in ("lbl_marker", "lbl_fes", "lbl_driver",
              "btn_marker_start", "btn_fes_start", "btn_driver_start"):
        assert getattr(panel.devices, w, None) is not None, w
    for m in ("on_marker_start", "on_marker_stop", "on_marker_refresh",
              "on_fes_start", "on_fes_stop", "on_fes_refresh",
              "on_driver_start", "on_driver_stop"):
        assert callable(getattr(panel.devices, m, None)), m


def test_external_tools_controller_wired(panel):
    """The external-app launchers (LabRecorder / eegoSports / MNE / impedance /
    STMsetup / initialize) were extracted into an ExternalToolsController that
    owns the handlers + the labrec_term / eego_term handles; the panel holds it."""
    from panel.external_tools import ExternalToolsController
    assert isinstance(panel.external_tools, ExternalToolsController)
    for m in ("on_initialize", "on_open_fes_cfg", "on_open_mne_viewer",
              "on_open_impedance_monitor", "on_open_labrec", "on_open_eego"):
        assert callable(getattr(panel.external_tools, m, None)), m


def test_robot_controller_wired(panel):
    """The Robot row (Init / Start / Remove Overrides) + handlers were extracted
    into a RobotController that owns its widgets; the panel holds it and the
    controller built its widgets into the grid during construction."""
    from panel.robot_controller import RobotController
    assert isinstance(panel.robot, RobotController)
    for w in ("lbl_robot_init", "lbl_robot", "btn_robot_start", "btn_robot_removeovr"):
        assert getattr(panel.robot, w, None) is not None, w
    for m in ("on_init_robot", "on_robot_start", "on_robot_remove_overrides",
              "_on_robot_term_finished", "_update_robot_buttons_for_mode"):
        assert callable(getattr(panel.robot, m, None)), m


def test_gaze_controller_wired(panel):
    """The Gaze Service row (LED + Start-Headless / Start-With-UI / Stop /
    Query-Telemetry buttons) + the gaze handlers were extracted into a
    GazeController that owns its widgets; the panel holds it and the controller
    built its widgets into the grid during construction."""
    from panel.gaze_controller import GazeController
    assert isinstance(panel.gaze, GazeController)
    # build_into() ran during _build_ui → the controller owns its widgets.
    assert panel.gaze.lbl_gaze_service is not None
    for w in ("btn_gaze_service_headless", "btn_gaze_service_ui",
              "btn_gaze_service_stop", "btn_gaze_service_query"):
        assert getattr(panel.gaze, w, None) is not None, w
    for m in ("on_gaze_service_start_headless", "on_gaze_service_start_ui",
              "on_gaze_service_stop", "on_gaze_service_query",
              "on_gaze_runner_start", "on_gaze_runner_stop",
              "_start_gaze_service", "_gaze_udp_request",
              "_format_gaze_telemetry_line", "_ensure_gaze_paths"):
        assert callable(getattr(panel.gaze, m, None)), m


def test_log_file_controller_wired(panel):
    """The subject-tied VLM-panel + frame-relay log files (their open/close/rotate
    logic + the relay log callback) were extracted into a LogFileController that
    owns the file handles; the panel holds it and tees its log buffers through the
    controller's handles. The files opened during __init__."""
    from panel.log_file_controller import LogFileController
    assert isinstance(panel.log_files, LogFileController)
    # The handle attributes the panel's _append_log tees to must exist (open or
    # None depending on whether DATA_DIR resolved under the test config).
    assert hasattr(panel.log_files, "vlm_log_fh")
    assert hasattr(panel.log_files, "relay_log_fh")
    for m in ("open_vlm", "close_vlm", "open_relay", "close_relay",
              "relay_callback"):
        assert callable(getattr(panel.log_files, m, None)), m


def test_runtime_config_controller_wired(panel):
    """The Runtime-config / ErrP-config editor tabs + the Model-training box (and
    their config read/write + training-launch handlers) were extracted into a
    RuntimeConfigController that owns those widgets; the panel holds it and the
    controller built its widgets during construction."""
    from panel.runtime_config_controller import RuntimeConfigController
    assert isinstance(panel.runtime_config, RuntimeConfigController)
    # build_tabs() + build_training_box() ran during _build_ui → the controller
    # owns its widgets.
    for w in ("rc_decoder", "rc_arduino_baud", "errp_backend",
              "cmb_train_script", "lst_training_files", "btn_launch_training"):
        assert getattr(panel.runtime_config, w, None) is not None, w
    for m in ("on_runtime_reload_config", "on_runtime_apply_config",
              "on_errp_config_reload", "on_errp_config_apply",
              "on_refresh_training_data_list", "on_launch_model_training",
              "_populate_training_script_combo", "_update_train_cmd_preview"):
        assert callable(getattr(panel.runtime_config, m, None)), m


def test_calibration_controller_wired(panel):
    """The Harmony / AprilTag calibration QGroupBox + handlers were extracted into
    a CalibrationController that owns its dropdowns; the panel holds it and the
    controller built its widgets into the Robot Test tab during construction."""
    from panel.calibration_controller import CalibrationController
    assert isinstance(panel.calibration, CalibrationController)
    for w in ("cmb_calibration_lib", "cmb_apriltag_calib",
              "btn_run_harmony_calibration", "btn_run_apriltag_calibrate"):
        assert getattr(panel.calibration, w, None) is not None, w
    for m in ("on_refresh_calibration_libs", "_get_selected_calibration_library",
              "on_run_harmony_calibration", "on_run_harmony_online_control",
              "on_refresh_apriltag_calibs", "_get_selected_apriltag_calib",
              "on_run_apriltag_calibrate", "on_run_apriltag_control_test"):
        assert callable(getattr(panel.calibration, m, None)), m
