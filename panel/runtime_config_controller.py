"""
panel/runtime_config_controller.py — Runtime-config + ErrP-config editor tabs and
the model-training launcher for the control panel.

Non-row collaborator following the SerialController/GazeController shape: it owns
two whole QTabWidget tabs (Runtime config, ErrP config) plus the Model-training
QGroupBox in the Robot Test tab, the widgets inside them (the rc_* / errp_* /
training-* widgets), and the handlers that read/write config.py / config_local.py
via panel.config_io and launch the offline training scripts. All widget trees and
handler bodies are transcribed verbatim from ControlPanel.

The two config tabs are added via build_tabs(tabs); the training box is returned
by build_training_box() for the panel to drop into the Robot Test tab (its widgets
sit between the calibration box and the Notes box, so the panel keeps owning that
tab's layout and just asks for the box).

Cross-cutting concerns are injected as callbacks (spawn_external / log) and the
live subject is read lazily (get_subject_text / get_training_subject) — the
training launcher reads whichever subject is selected at click time, exactly as
the former in-class calls did. The controller has no back-reference into the panel
beyond a QMessageBox parent.
"""

from __future__ import annotations

import os
import time
from typing import Callable

from PySide6.QtCore import QObject
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QListWidget, QMessageBox, QPushButton, QScrollArea, QSpinBox,
    QTabWidget, QVBoxLayout, QWidget,
)

from panel.config_io import (
    _read_float_key, _read_int_key, _read_bool_key, _read_quoted_str_key,
    _write_assign_rhs, _read_01_key,
)
from panel.constants import ROOT, TRAINING_SCRIPT_ENTRIES, _HCFG


class RuntimeConfigController(QObject):
    """Owns the Runtime-config / ErrP-config tabs + the Model-training box.

    Injected dependencies (behaviour-identical to the former in-class calls):
      spawn_external(cmd)        — open a command in a gnome-terminal
      log(title, text)           — append to the panel's log buffer
      get_subject_text()         — current text of the panel's subject combo
      get_training_subject()     — the panel's persisted TRAINING_SUBJECT
      timestamp()                — "HH:MM:SS" for log lines
    """

    def __init__(
        self,
        parent,
        *,
        spawn_external: Callable[[str], None],
        log: Callable[[str, str], None],
        get_subject_text: Callable[[], str],
        get_training_subject: Callable[[], str],
        timestamp: Callable[[], str],
    ) -> None:
        super().__init__(parent)
        self._parent = parent
        self._spawn_external = spawn_external
        self._log = log
        self._get_subject_text = get_subject_text
        self._get_training_subject = get_training_subject
        self._ts = timestamp

    def build_tabs(self, tabs: QTabWidget) -> None:
        """Build the Runtime-config + ErrP-config tabs into ``tabs``. Widget
        trees + tab placement are identical to the former inline _build_*_config_tab
        blocks."""
        self._build_runtime_config_tab(tabs)
        self._build_errp_config_tab(tabs)

    def build_training_box(self) -> QGroupBox:
        """Build and return the Model-training QGroupBox (widget tree transcribed
        verbatim from the former inline _build_ui block); the panel adds it to the
        Robot Test tab. The script combo is populated on construction."""
        train_box = QGroupBox("Model training (uses config.py DATA_DIR + subject below)")
        tv = QVBoxLayout(train_box)
        self.lbl_training_subject_ctx = QLabel("")
        self.lbl_training_subject_ctx.setWordWrap(True)
        tv.addWidget(self.lbl_training_subject_ctx)
        trow = QHBoxLayout()
        trow.addWidget(QLabel("Script:"))
        self.cmb_train_script = QComboBox()
        trow.addWidget(self.cmb_train_script, 1)
        self.btn_refresh_training_data = QPushButton("Refresh data list")
        self.btn_refresh_training_data.setMaximumWidth(130)
        self.btn_refresh_training_data.clicked.connect(self.on_refresh_training_data_list)
        trow.addWidget(self.btn_refresh_training_data)
        tv.addLayout(trow)
        self.lst_training_files = QListWidget()
        self.lst_training_files.setMaximumHeight(140)
        tv.addWidget(self.lst_training_files)
        self.lbl_train_cmd_preview = QLabel("")
        self.lbl_train_cmd_preview.setWordWrap(True)
        self.lbl_train_cmd_preview.setStyleSheet("color: #666; font-family: monospace;")
        tv.addWidget(self.lbl_train_cmd_preview)
        train_btn_row = QHBoxLayout()
        self.btn_launch_training = QPushButton("Launch training (terminal)")
        self.btn_launch_training.clicked.connect(self.on_launch_model_training)
        self.btn_launch_training.setMaximumWidth(220)
        train_btn_row.addWidget(self.btn_launch_training)
        train_btn_row.addStretch(1)
        tv.addLayout(train_btn_row)
        self._populate_training_script_combo()
        return train_box

    def _build_runtime_config_tab(self, tabs: QTabWidget):
        rtc = QWidget()
        tabs.addTab(rtc, "Runtime config")
        outer = QVBoxLayout(rtc)
        # setWordWrap on this label is load-bearing: without it, the
        # full unwrapped text width becomes the tab's minimumSizeHint
        # (~1900 px) and propagates to the whole window's minimum size.
        intro = QLabel(
            "<b>Edits config.py / config_local.py on disk.</b> "
            "Machine-local keys (PERCEPTION_FRAME_SOURCE, "
            "SERVICES_HOSTED_REMOTELY) write to config_local.py; "
            "everything else writes to config.py. Restart "
            "Marker/Driver/FES after changing simulation or network "
            "flags (<code>Utils/networking</code> caches SIMULATION_MODE "
            "at import)."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        form = QFormLayout(inner)
        self.rc_decoder = QComboBox()
        self.rc_decoder.addItems(["mdm", "xgb_cov", "xgb_cov_erd"])
        self.rc_earlystop = QComboBox()
        self.rc_earlystop.addItems(["either", "correct_only"])
        self.rc_visual = QComboBox()
        self.rc_visual.addItems(["classic", "modern"])
        self.rc_th_mi = QDoubleSpinBox()
        self.rc_th_mi.setRange(0.0, 1.0)
        self.rc_th_mi.setSingleStep(0.01)
        self.rc_th_mi.setDecimals(3)
        self.rc_th_rest = QDoubleSpinBox()
        self.rc_th_rest.setRange(0.0, 1.0)
        self.rc_th_rest.setSingleStep(0.01)
        self.rc_th_rest.setDecimals(3)
        self.rc_int_alpha = QDoubleSpinBox()
        self.rc_int_alpha.setRange(0.0, 1.0)
        self.rc_int_alpha.setSingleStep(0.01)
        self.rc_int_alpha.setDecimals(3)
        self.rc_classify_ms = QSpinBox()
        self.rc_classify_ms.setRange(100, 8000)
        self.rc_min_pred = QSpinBox()
        self.rc_min_pred.setRange(1, 500)
        self.rc_time_mi = QSpinBox()
        self.rc_time_mi.setRange(1, 300)
        self.rc_time_rob = QSpinBox()
        self.rc_time_rob.setRange(1, 300)
        self.rc_big_brother = QCheckBox("BIG_BROTHER_MODE (second display layout)")
        self.rc_send_probs = QCheckBox("SEND_PROBS (extra UDP probability traffic)")
        self.rc_save_adaptive_t = QCheckBox("SAVE_ADAPTIVE_T (persist adaptive μ/β transform across sessions)")
        self.rc_recentering = QCheckBox("RECENTERING")
        self.rc_update_move = QCheckBox("UPDATE_DURING_MOVE")
        self.rc_laplacian = QCheckBox("SURFACE_LAPLACIAN_TOGGLE")
        self.rc_sel_motor = QCheckBox("SELECT_MOTOR_CHANNELS")
        self.rc_xgb_beta = QCheckBox("XGB_USE_COV_BETA (enable beta-band covariance features)")
        self.rc_total_trials = QSpinBox()
        self.rc_total_trials.setRange(1, 500)
        self.rc_shape_max = QDoubleSpinBox()
        self.rc_shape_max.setRange(0.0, 1.0)
        self.rc_shape_max.setSingleStep(0.01)
        self.rc_shape_max.setDecimals(2)
        self.rc_shape_min = QDoubleSpinBox()
        self.rc_shape_min.setRange(0.0, 1.0)
        self.rc_shape_min.setSingleStep(0.01)
        self.rc_shape_min.setDecimals(2)

        # Perception / VLM section (mix of global + machine-local keys —
        # the writer routes by key name).
        self.rc_gaze_backend = QComboBox()
        self.rc_gaze_backend.addItems(["legacy", "vlm"])
        self.rc_perception_source = QComboBox()
        self.rc_perception_source.addItems(["local", "remote"])
        self.rc_services_remote = QCheckBox("SERVICES_HOSTED_REMOTELY (panel → remote GPU host; machine-local)")
        self.rc_relay_hz = QDoubleSpinBox()
        self.rc_relay_hz.setRange(1.0, 30.0)
        self.rc_relay_hz.setSingleStep(1.0)
        self.rc_relay_hz.setDecimals(1)
        self.rc_relay_hz.setSuffix(" Hz")
        self.rc_vlm_depth = QCheckBox("VLM_ENABLE_DEPTH (load Depth Pro at vlm_service start)")
        self.rc_arduino_baud = QSpinBox()
        self.rc_arduino_baud.setRange(300, 1_000_000)
        self.rc_arduino_baud.setSingleStep(100)
        form.addRow("DECODER_BACKEND", self.rc_decoder)
        form.addRow("EARLYSTOP_MODE", self.rc_earlystop)
        form.addRow("CLASS_VISUAL_STYLE", self.rc_visual)
        form.addRow("THRESHOLD_MI", self.rc_th_mi)
        form.addRow("THRESHOLD_REST", self.rc_th_rest)
        form.addRow("INTEGRATOR_ALPHA", self.rc_int_alpha)
        form.addRow("CLASSIFY_WINDOW (ms)", self.rc_classify_ms)
        form.addRow("MIN_PREDICTIONS", self.rc_min_pred)
        form.addRow("TIME_MI (s)", self.rc_time_mi)
        form.addRow("TIME_ROB (s)", self.rc_time_rob)
        form.addRow(self.rc_big_brother)
        form.addRow(self.rc_send_probs)
        form.addRow(self.rc_save_adaptive_t)
        form.addRow(self.rc_recentering)
        form.addRow(self.rc_update_move)
        form.addRow(self.rc_laplacian)
        form.addRow(self.rc_sel_motor)
        form.addRow(self.rc_xgb_beta)
        form.addRow("TOTAL_TRIALS", self.rc_total_trials)
        form.addRow("SHAPE_MAX", self.rc_shape_max)
        form.addRow("SHAPE_MIN", self.rc_shape_min)

        # Perception / VLM rows — mark local keys explicitly in the label
        # so the operator knows which file the Apply will touch.
        form.addRow("GAZE_OR_BACKEND", self.rc_gaze_backend)
        form.addRow("PERCEPTION_FRAME_SOURCE  [local]", self.rc_perception_source)
        form.addRow(self.rc_services_remote)
        form.addRow("FRAME_RELAY_HZ", self.rc_relay_hz)
        form.addRow(self.rc_vlm_depth)
        form.addRow("ARDUINO_BAUD", self.rc_arduino_baud)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)
        btn_row = QHBoxLayout()
        btn_reload = QPushButton("Reload from config.py")
        btn_reload.clicked.connect(self.on_runtime_reload_config)
        btn_apply = QPushButton("Apply to config.py")
        btn_apply.clicked.connect(self.on_runtime_apply_config)
        btn_row.addWidget(btn_reload)
        btn_row.addWidget(btn_apply)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)
        self.on_runtime_reload_config()

    def _rc_set_combo(self, cb: QComboBox, text: str, fallback_index: int = 0):
        idx = cb.findText(text)
        cb.setCurrentIndex(idx if idx >= 0 else fallback_index)

    def on_runtime_reload_config(self):
        if not hasattr(self, "rc_decoder"):
            return
        self._rc_set_combo(self.rc_decoder, _read_quoted_str_key("DECODER_BACKEND", "mdm"))
        self._rc_set_combo(self.rc_earlystop, _read_quoted_str_key("EARLYSTOP_MODE", "either"))
        vis = _read_quoted_str_key("CLASS_VISUAL_STYLE", "classic")
        self._rc_set_combo(self.rc_visual, vis if vis in ("classic", "modern") else "classic")
        self.rc_th_mi.setValue(_read_float_key("THRESHOLD_MI", 0.65))
        self.rc_th_rest.setValue(_read_float_key("THRESHOLD_REST", 0.65))
        self.rc_int_alpha.setValue(_read_float_key("INTEGRATOR_ALPHA", 0.96))
        self.rc_classify_ms.setValue(_read_int_key("CLASSIFY_WINDOW", 1000))
        self.rc_min_pred.setValue(_read_int_key("MIN_PREDICTIONS", 8))
        self.rc_time_mi.setValue(_read_int_key("TIME_MI", 5))
        self.rc_time_rob.setValue(_read_int_key("TIME_ROB", 7))
        self.rc_big_brother.setChecked(_read_bool_key("BIG_BROTHER_MODE", True))
        self.rc_send_probs.setChecked(_read_bool_key("SEND_PROBS", False))
        self.rc_save_adaptive_t.setChecked(_read_bool_key("SAVE_ADAPTIVE_T", False))
        self.rc_recentering.setChecked(bool(_read_01_key("RECENTERING", 1)))
        self.rc_update_move.setChecked(bool(_read_01_key("UPDATE_DURING_MOVE", 0)))
        self.rc_laplacian.setChecked(bool(_read_01_key("SURFACE_LAPLACIAN_TOGGLE", 1)))
        self.rc_sel_motor.setChecked(bool(_read_01_key("SELECT_MOTOR_CHANNELS", 1)))
        self.rc_xgb_beta.setChecked(bool(_read_01_key("XGB_USE_COV_BETA", 0)))
        self.rc_total_trials.setValue(_read_int_key("TOTAL_TRIALS", 10))
        self.rc_shape_max.setValue(_read_float_key("SHAPE_MAX", 0.7))
        self.rc_shape_min.setValue(_read_float_key("SHAPE_MIN", 0.5))
        # Perception / VLM (readers consult config_local.py first, then
        # config.py — same precedence as the live import).
        gob = _read_quoted_str_key("GAZE_OR_BACKEND", "legacy").lower()
        self._rc_set_combo(self.rc_gaze_backend, gob if gob in ("legacy", "vlm") else "legacy")
        pfs = _read_quoted_str_key("PERCEPTION_FRAME_SOURCE", "local").lower()
        self._rc_set_combo(self.rc_perception_source, pfs if pfs in ("local", "remote") else "local")
        self.rc_services_remote.setChecked(_read_bool_key("SERVICES_HOSTED_REMOTELY", False))
        self.rc_relay_hz.setValue(_read_float_key("FRAME_RELAY_HZ", 15.0))
        self.rc_vlm_depth.setChecked(_read_bool_key("VLM_ENABLE_DEPTH", True))
        self.rc_arduino_baud.setValue(_read_int_key("ARDUINO_BAUD", 9600))
        self._log("Panel", f"[{self._ts()}] Runtime config widgets reloaded from config.py / config_local.py\n")

    def on_runtime_apply_config(self):
        try:
            def _fmtf(x: float) -> str:
                t = f"{x:.6f}".rstrip("0").rstrip(".")
                return t if t else "0"

            _write_assign_rhs("DECODER_BACKEND", f'"{self.rc_decoder.currentText()}"')
            _write_assign_rhs("EARLYSTOP_MODE", f'"{self.rc_earlystop.currentText()}"')
            _write_assign_rhs("CLASS_VISUAL_STYLE", f'"{self.rc_visual.currentText()}"')
            _write_assign_rhs("THRESHOLD_MI", _fmtf(self.rc_th_mi.value()))
            _write_assign_rhs("THRESHOLD_REST", _fmtf(self.rc_th_rest.value()))
            _write_assign_rhs("INTEGRATOR_ALPHA", _fmtf(self.rc_int_alpha.value()))
            _write_assign_rhs("CLASSIFY_WINDOW", str(self.rc_classify_ms.value()))
            _write_assign_rhs("MIN_PREDICTIONS", str(self.rc_min_pred.value()))
            _write_assign_rhs("TIME_MI", str(self.rc_time_mi.value()))
            _write_assign_rhs("TIME_ROB", str(self.rc_time_rob.value()))
            _write_assign_rhs("BIG_BROTHER_MODE", "True" if self.rc_big_brother.isChecked() else "False")
            _write_assign_rhs("SEND_PROBS", "True" if self.rc_send_probs.isChecked() else "False")
            _write_assign_rhs("SAVE_ADAPTIVE_T", "True" if self.rc_save_adaptive_t.isChecked() else "False")
            _write_assign_rhs("RECENTERING", "1" if self.rc_recentering.isChecked() else "0")
            _write_assign_rhs("UPDATE_DURING_MOVE", "1" if self.rc_update_move.isChecked() else "0")
            _write_assign_rhs("SURFACE_LAPLACIAN_TOGGLE", "1" if self.rc_laplacian.isChecked() else "0")
            _write_assign_rhs("SELECT_MOTOR_CHANNELS", "1" if self.rc_sel_motor.isChecked() else "0")
            _write_assign_rhs("XGB_USE_COV_BETA", "1" if self.rc_xgb_beta.isChecked() else "0")
            _write_assign_rhs("TOTAL_TRIALS", str(self.rc_total_trials.value()))
            _write_assign_rhs("SHAPE_MAX", _fmtf(self.rc_shape_max.value()))
            _write_assign_rhs("SHAPE_MIN", _fmtf(self.rc_shape_min.value()))
            # Perception / VLM (the writer routes machine-local keys to
            # config_local.py automatically).
            _write_assign_rhs("GAZE_OR_BACKEND", f'"{self.rc_gaze_backend.currentText()}"')
            _write_assign_rhs("PERCEPTION_FRAME_SOURCE", f'"{self.rc_perception_source.currentText()}"')
            _write_assign_rhs("SERVICES_HOSTED_REMOTELY", "True" if self.rc_services_remote.isChecked() else "False")
            _write_assign_rhs("FRAME_RELAY_HZ", _fmtf(self.rc_relay_hz.value()))
            _write_assign_rhs("VLM_ENABLE_DEPTH", "True" if self.rc_vlm_depth.isChecked() else "False")
            _write_assign_rhs("ARDUINO_BAUD", str(self.rc_arduino_baud.value()))
        except Exception as e:
            QMessageBox.warning(self._parent, "Runtime config", f"Failed to update config files:\n{e}")
            self._log("Panel", f"[{self._ts()}] Runtime config apply FAILED: {e}\n")
            return
        self._log("Panel", f"[{self._ts()}] Runtime config written to config.py / config_local.py\n")
        QMessageBox.information(
            self._parent, "Runtime config",
            "config.py / config_local.py updated. Restart experiment driver / marker stream if a "
            "process was already running so it reloads settings.",
        )

    def _build_errp_config_tab(self, tabs: QTabWidget):
        rtc = QWidget()
        tabs.addTab(rtc, "ErrP config")
        outer = QVBoxLayout(rtc)
        outer.addWidget(QLabel(
            "<b>Edits ErrP-specific keys in config.py.</b> The bundle on disk at "
            "<code>DATA_DIR/sub-&lt;SUBJECT&gt;/models/sub-&lt;SUBJECT&gt;_errp_&lt;BACKEND&gt;.pkl</code> "
            "must exist for the selected backend. Runtime asserts the bundle's "
            "feature_spec matches config; mismatch raises."
        ))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        form = QFormLayout(inner)

        self.errp_enable = QCheckBox("ERRP_DECODER_ENABLE (ErrP gating active)")
        self.errp_backend = QComboBox()
        self.errp_backend.addItems(["liu_cca_xgb", "xdawn_xgb"])
        self.errp_sel_channels = QCheckBox("SELECT_ERRP_CHANNELS")
        self.errp_ea_bootstrap_sec = QDoubleSpinBox()
        self.errp_ea_bootstrap_sec.setRange(1.0, 300.0)
        self.errp_ea_bootstrap_sec.setSingleStep(1.0)
        self.errp_ea_bootstrap_sec.setDecimals(1)
        self.errp_ea_min_epochs = QSpinBox()
        self.errp_ea_min_epochs.setRange(1, 500)
        self.errp_p_stop = QDoubleSpinBox()
        self.errp_p_stop.setRange(0.0, 1.0)
        self.errp_p_stop.setSingleStep(0.05)
        self.errp_p_stop.setDecimals(2)

        form.addRow(self.errp_enable)
        form.addRow("ERRP_DECODER_BACKEND", self.errp_backend)
        form.addRow(self.errp_sel_channels)
        form.addRow("ERRP_EA_BOOTSTRAP_SEC", self.errp_ea_bootstrap_sec)
        form.addRow("ERRP_EA_MIN_EPOCHS", self.errp_ea_min_epochs)
        form.addRow("ERRP_ONLINE_P_STOP", self.errp_p_stop)

        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)
        btn_row = QHBoxLayout()
        btn_reload = QPushButton("Reload from config.py")
        btn_reload.clicked.connect(self.on_errp_config_reload)
        btn_apply = QPushButton("Apply to config.py")
        btn_apply.clicked.connect(self.on_errp_config_apply)
        btn_row.addWidget(btn_reload)
        btn_row.addWidget(btn_apply)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)
        self.on_errp_config_reload()

    def on_errp_config_reload(self):
        if not hasattr(self, "errp_backend"):
            return
        self.errp_enable.setChecked(bool(_read_01_key("ERRP_DECODER_ENABLE", 0)))
        backend = _read_quoted_str_key("ERRP_DECODER_BACKEND", "liu_cca_xgb")
        self._rc_set_combo(self.errp_backend, backend)
        self.errp_sel_channels.setChecked(bool(_read_01_key("SELECT_ERRP_CHANNELS", 0)))
        self.errp_ea_bootstrap_sec.setValue(_read_float_key("ERRP_EA_BOOTSTRAP_SEC", 45.0))
        self.errp_ea_min_epochs.setValue(_read_int_key("ERRP_EA_MIN_EPOCHS", 20))
        self.errp_p_stop.setValue(_read_float_key("ERRP_ONLINE_P_STOP", 0.3))
        self._log("Panel", f"[{self._ts()}] ErrP config widgets reloaded from config.py\n")

    def on_errp_config_apply(self):
        try:
            def _fmtf(x: float) -> str:
                t = f"{x:.6f}".rstrip("0").rstrip(".")
                return t if t else "0"

            _write_assign_rhs("ERRP_DECODER_ENABLE", "1" if self.errp_enable.isChecked() else "0")
            _write_assign_rhs("ERRP_DECODER_BACKEND", f'"{self.errp_backend.currentText()}"')
            _write_assign_rhs("SELECT_ERRP_CHANNELS", "1" if self.errp_sel_channels.isChecked() else "0")
            _write_assign_rhs("ERRP_EA_BOOTSTRAP_SEC", _fmtf(self.errp_ea_bootstrap_sec.value()))
            _write_assign_rhs("ERRP_EA_MIN_EPOCHS", str(self.errp_ea_min_epochs.value()))
            _write_assign_rhs("ERRP_ONLINE_P_STOP", _fmtf(self.errp_p_stop.value()))
        except Exception as e:
            QMessageBox.warning(self._parent, "config.py", f"Failed to update config.py:\n{e}")
            self._log("Panel", f"[{self._ts()}] ErrP config apply FAILED: {e}\n")
            return
        self._log("Panel", f"[{self._ts()}] ErrP config written to config.py\n")
        QMessageBox.information(
            self._parent, "ErrP config",
            "config.py updated. Restart ExperimentDriver_ErrP_Online if a session was "
            "already running so it reloads the selected bundle.",
        )

    def _populate_training_script_combo(self):
        if not hasattr(self, "cmb_train_script"):
            return
        self.cmb_train_script.blockSignals(True)
        self.cmb_train_script.clear()
        for label, fname in TRAINING_SCRIPT_ENTRIES:
            path = os.path.join(ROOT, fname)
            if os.path.isfile(path):
                self.cmb_train_script.addItem(label, path)
        if self.cmb_train_script.count() == 0:
            self.cmb_train_script.addItem("No training scripts found", "")
        self.cmb_train_script.blockSignals(False)
        self.cmb_train_script.currentIndexChanged.connect(self._update_train_cmd_preview)

    def _update_train_cmd_preview(self, *_args):
        if not hasattr(self, "lbl_train_cmd_preview"):
            return
        script = self.cmb_train_script.currentData()
        if script and os.path.isfile(script):
            self.lbl_train_cmd_preview.setText(f'cd "{ROOT}" && python -u "{script}"')
        else:
            self.lbl_train_cmd_preview.setText("(no script selected)")

    def on_refresh_training_data_list(self):
        if not hasattr(self, "lst_training_files"):
            return
        sub = (self._get_subject_text().strip()) or self._get_training_subject()
        if _HCFG is None:
            self.lbl_training_subject_ctx.setText("DATA_DIR not available (config import failed).")
            self.btn_launch_training.setEnabled(False)
            self._update_train_cmd_preview()
            return
        data_dir = os.path.expanduser(getattr(_HCFG, "DATA_DIR", "") or "")
        tdir = os.path.join(data_dir, f"sub-{sub}", "training_data")
        self.lbl_training_subject_ctx.setText(f"<b>Subject:</b> {sub}<br><b>training_data:</b> {tdir}")
        self.lst_training_files.clear()
        xdffc = []
        if os.path.isdir(tdir):
            for fn in sorted(os.listdir(tdir)):
                if fn.lower().endswith(".xdf"):
                    full = os.path.join(tdir, fn)
                    try:
                        mtime = os.path.getmtime(full)
                        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
                    except OSError:
                        ts = "?"
                    self.lst_training_files.addItem(f"{fn}  ({ts})")
                    xdffc.append(full)
        script_ok = bool(self.cmb_train_script.currentData()) and os.path.isfile(self.cmb_train_script.currentData() or "")
        self.btn_launch_training.setEnabled(len(xdffc) > 0 and script_ok)
        self._update_train_cmd_preview()

    def on_launch_model_training(self):
        script = self.cmb_train_script.currentData()
        if not script or not os.path.isfile(script):
            QMessageBox.warning(self._parent, "Training", "Select a valid training script.")
            return
        sub = (self._get_subject_text().strip() or self._get_training_subject())
        if _HCFG is None:
            QMessageBox.warning(self._parent, "Training", "config module not loaded.")
            return
        data_dir = os.path.expanduser(getattr(_HCFG, "DATA_DIR", "") or "")
        tdir = os.path.join(data_dir, f"sub-{sub}", "training_data")
        if not os.path.isdir(tdir):
            QMessageBox.warning(self._parent, "Training", f"training_data folder not found:\n{tdir}")
            return
        xdffc = [f for f in os.listdir(tdir) if f.lower().endswith(".xdf")]
        if not xdffc:
            QMessageBox.warning(self._parent, "Training", "No .xdf files in training_data.")
            return
        cmd = f'python -u "{script}"'
        self._spawn_external(cmd)
        self._log("Panel", f"[{self._ts()}] Launched training: {cmd}\n")
