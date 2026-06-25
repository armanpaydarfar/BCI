"""
panel/calibration_controller.py — Harmony calibration / online-control + AprilTag
gaze→robot calibration controls (the "Harmony calibration / online control"
QGroupBox in the Robot Test tab).

Widget-owning collaborator following the SerialController shape: it builds its own
QGroupBox (calibration-library + AprilTag-calib dropdowns, their Refresh buttons,
and the four Run buttons) and adds it to the Robot Test tab layout via
build_into(parent_layout), owns the two combos, and holds the handlers
(on_refresh_calibration_libs / on_run_harmony_* / on_refresh_apriltag_calibs /
on_run_apriltag_*) — all transcribed verbatim from ControlPanel.

The script paths and the AprilTag rig config (config.APRILTAG_*) are injected at
construction. Cross-cutting concerns are injected as callbacks (spawn_external /
log / timestamp) so the controller has no back-reference into the panel beyond a
QMessageBox parent.
"""

from __future__ import annotations

import glob
import os
from typing import Callable

from PySide6.QtCore import QObject
from PySide6.QtWidgets import (
    QComboBox, QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QVBoxLayout,
)


class CalibrationController(QObject):
    """Owns the Harmony / AprilTag calibration QGroupBox and its handlers.

    Injected dependencies (behaviour-identical to the former in-class calls):
      root                       — repo root for the *.npz / runs/ globs
      harmony_calibration_exec_py / harmony_online_control_py /
      apriltag_calibrate_py / apriltag_control_test_py — tool script paths
      hcfg                       — the config module (APRILTAG_* rig params), or None
      spawn_external(cmd)        — open a command in a gnome-terminal
      log(title, text)           — append to the panel's log buffer
      timestamp()                — "HH:MM:SS" for log lines
    """

    def __init__(
        self,
        parent,
        *,
        root: str,
        harmony_calibration_exec_py: str,
        harmony_online_control_py: str,
        apriltag_calibrate_py: str,
        apriltag_control_test_py: str,
        hcfg,
        spawn_external: Callable[[str], None],
        log: Callable[[str, str], None],
        timestamp: Callable[[], str],
    ) -> None:
        super().__init__(parent)
        self._parent = parent
        self._root = root
        self._harmony_calibration_exec_py = harmony_calibration_exec_py
        self._harmony_online_control_py = harmony_online_control_py
        self._apriltag_calibrate_py = apriltag_calibrate_py
        self._apriltag_control_test_py = apriltag_control_test_py
        self._hcfg = hcfg
        self._spawn_external = spawn_external
        self._log = log
        self._ts = timestamp

    def build_into(self, parent_layout) -> None:
        """Build the calibration QGroupBox and add it to ``parent_layout`` (the
        Robot Test tab's vertical layout). Widget tree + placement are identical
        to the former inline _build_ui block."""
        harmony_box = QGroupBox("Harmony calibration / online control")
        hb = QVBoxLayout(harmony_box)
        lib_row = QHBoxLayout()
        lib_row.addWidget(QLabel("Calibration library:"))
        self.cmb_calibration_lib = QComboBox()
        lib_row.addWidget(self.cmb_calibration_lib, 1)
        self.btn_refresh_calibration_libs = QPushButton("Refresh")
        self.btn_refresh_calibration_libs.setMaximumWidth(90)
        self.btn_refresh_calibration_libs.clicked.connect(self.on_refresh_calibration_libs)
        lib_row.addWidget(self.btn_refresh_calibration_libs)
        hb.addLayout(lib_row)

        hbtn_row = QHBoxLayout()
        self.btn_run_harmony_calibration = QPushButton("Run harmony_calibration_exec.py")
        self.btn_run_harmony_calibration.setMaximumWidth(260)
        self.btn_run_harmony_calibration.clicked.connect(self.on_run_harmony_calibration)
        self.btn_run_harmony_online = QPushButton("Run harmony_online_control.py")
        self.btn_run_harmony_online.setMaximumWidth(240)
        self.btn_run_harmony_online.clicked.connect(self.on_run_harmony_online_control)
        hbtn_row.addWidget(self.btn_run_harmony_calibration)
        hbtn_row.addWidget(self.btn_run_harmony_online)
        hbtn_row.addStretch(1)
        hb.addLayout(hbtn_row)

        # WS5 REV04 AprilTag gaze→robot calibration (HIL PASS 2026-06-24). The
        # calibration button runs the swept capture + planar solve in one terminal
        # using the verified rig config (config.APRILTAG_*); the control-test button
        # drives the robot from the AprilTag calibration selected below (the dropdown
        # lists runs/apriltag_*_calib.npz newest-first, so it defaults to the latest).
        atag_lib_row = QHBoxLayout()
        atag_lib_row.addWidget(QLabel("AprilTag calib:"))
        self.cmb_apriltag_calib = QComboBox()
        atag_lib_row.addWidget(self.cmb_apriltag_calib, 1)
        self.btn_refresh_apriltag_calibs = QPushButton("Refresh")
        self.btn_refresh_apriltag_calibs.setMaximumWidth(90)
        self.btn_refresh_apriltag_calibs.clicked.connect(self.on_refresh_apriltag_calibs)
        atag_lib_row.addWidget(self.btn_refresh_apriltag_calibs)
        hb.addLayout(atag_lib_row)

        atag_row = QHBoxLayout()
        self.btn_run_apriltag_calibrate = QPushButton("Run AprilTag calibration")
        self.btn_run_apriltag_calibrate.setMaximumWidth(220)
        self.btn_run_apriltag_calibrate.clicked.connect(self.on_run_apriltag_calibrate)
        self.btn_run_apriltag_control = QPushButton("Run AprilTag control test")
        self.btn_run_apriltag_control.setMaximumWidth(220)
        self.btn_run_apriltag_control.clicked.connect(self.on_run_apriltag_control_test)
        atag_row.addWidget(self.btn_run_apriltag_calibrate)
        atag_row.addWidget(self.btn_run_apriltag_control)
        atag_row.addStretch(1)
        hb.addLayout(atag_row)
        parent_layout.addWidget(harmony_box)

    # ----- Harmony calibration / online control -----
    def on_refresh_calibration_libs(self):
        if not hasattr(self, "cmb_calibration_lib"):
            return

        current = self.cmb_calibration_lib.currentData()
        self.cmb_calibration_lib.clear()

        # Search for .npz libraries in ROOT
        libs = sorted(glob.glob(os.path.join(self._root, "*.npz")))

        if not libs:
            self.cmb_calibration_lib.addItem("No calibration libraries found", "")
            self._log("Panel", f"[{self._ts()}] No calibration libraries (*.npz) found in {self._root}\n")
            return

        for lib in libs:
            self.cmb_calibration_lib.addItem(os.path.basename(lib), lib)

        # Try to restore previous selection if still present
        if current:
            idx = self.cmb_calibration_lib.findData(current)
            if idx >= 0:
                self.cmb_calibration_lib.setCurrentIndex(idx)

        self._log("Panel", f"[{self._ts()}] Refreshed calibration libraries ({len(libs)} found)\n")

    def _get_selected_calibration_library(self) -> str:
        if not hasattr(self, "cmb_calibration_lib"):
            return ""
        return self.cmb_calibration_lib.currentData() or ""

    def on_run_harmony_calibration(self):
        if not os.path.exists(self._harmony_calibration_exec_py):
            QMessageBox.warning(self._parent, "Missing", f"Not found:\n{self._harmony_calibration_exec_py}")
            return

        self._spawn_external(f'python -u "{self._harmony_calibration_exec_py}"')
        self._log("Panel", f"[{self._ts()}] Opened harmony_calibration_exec.py\n")

    def on_run_harmony_online_control(self):
        if not os.path.exists(self._harmony_online_control_py):
            QMessageBox.warning(self._parent, "Missing", f"Not found:\n{self._harmony_online_control_py}")
            return

        calib_lib = self._get_selected_calibration_library()
        if not calib_lib or not os.path.exists(calib_lib):
            QMessageBox.warning(self._parent, "Calibration Library", "Please select a valid calibration library (.npz).")
            return

        # Assumes harmony_online_control.py takes the calibration library as a positional argument.
        # If your script expects a flag instead (for example --calib_lib), change the line below accordingly.
        self._spawn_external(f'python -u "{self._harmony_online_control_py}" "{calib_lib}"')
        self._log("Panel", f"[{self._ts()}] Opened harmony_online_control.py with calibration library:\n  {calib_lib}\n")

    def on_refresh_apriltag_calibs(self):
        """List runs/apriltag_*_calib.npz newest-first so the dropdown defaults to the
        latest calibration (the operator does not need to know the filename)."""
        if not hasattr(self, "cmb_apriltag_calib"):
            return
        current = self.cmb_apriltag_calib.currentData()
        self.cmb_apriltag_calib.clear()
        calibs = sorted(glob.glob(os.path.join(self._root, "runs", "apriltag_*_calib.npz")),
                        key=os.path.getmtime, reverse=True)
        if not calibs:
            self.cmb_apriltag_calib.addItem("No AprilTag calibrations in runs/", "")
            self._log("Panel", f"[{self._ts()}] No AprilTag calibrations "
                      f"(runs/apriltag_*_calib.npz) yet — run a calibration first\n")
            return
        for c in calibs:
            self.cmb_apriltag_calib.addItem(os.path.basename(c), c)
        # Keep the prior selection if still present; otherwise index 0 (= newest).
        if current:
            idx = self.cmb_apriltag_calib.findData(current)
            if idx >= 0:
                self.cmb_apriltag_calib.setCurrentIndex(idx)
        self._log("Panel", f"[{self._ts()}] AprilTag calibrations: {len(calibs)} "
                  f"(newest first → {os.path.basename(calibs[0])})\n")

    def _get_selected_apriltag_calib(self) -> str:
        if not hasattr(self, "cmb_apriltag_calib"):
            return ""
        return self.cmb_apriltag_calib.currentData() or ""

    def on_run_apriltag_calibrate(self):
        if not os.path.exists(self._apriltag_calibrate_py):
            QMessageBox.warning(self._parent, "Missing", f"Not found:\n{self._apriltag_calibrate_py}")
            return
        # REV04 swept calibration + planar solve in one terminal, from the verified
        # rig config (config.APRILTAG_*). The EE-point method + offset come from config
        # so the operator switches recipes without retyping: the verified-good recipe is
        # APRILTAG_EE_POINT_METHOD='rayplane' with a zero offset (re-confirmed on the rig
        # 2026-06-25; 'pose'+offset regressed via single-EE-tag flips). --then-solve
        # writes the *_calib.npz; the operator hits Refresh and the control-test button
        # picks it up.
        world = " ".join(str(int(i)) for i in getattr(self._hcfg, "APRILTAG_WORLD_TAG_IDS", [0, 1, 2, 3, 4]))
        ee = " ".join(str(int(i)) for i in getattr(self._hcfg, "APRILTAG_EE_TAG_IDS", [5]))
        tag = float(getattr(self._hcfg, "APRILTAG_TAG_SIZE_M", 0.08))
        ee_tag = float(getattr(self._hcfg, "APRILTAG_EE_TAG_SIZE_M", 0.04))
        method = getattr(self._hcfg, "APRILTAG_EE_POINT_METHOD", "rayplane")
        off = list(getattr(self._hcfg, "APRILTAG_T_EETAG_EE_MM", [0.0, 0.0, 0.0]))
        # --side defaults to env HARMONY_ACTIVE_SIDE or 'R' in the tool itself.
        cmd = (f'python -u "{self._apriltag_calibrate_py}" --stage sweep --with-robot '
               f'--world-tag-ids {world} --ee-tag-ids {ee} '
               f'--tag-size {tag} --ee-tag-size {ee_tag} '
               f'--ee-point-method {method} --t-eetag-ee {off[0]} {off[1]} {off[2]} '
               f'--out-dir runs --then-solve')
        self._spawn_external(cmd)
        self._log("Panel", f"[{self._ts()}] Launched AprilTag calibration "
                  f"(sweep → solve): world {world}, EE {ee}, method {method}, offset {off}\n")

    def on_run_apriltag_control_test(self):
        if not os.path.exists(self._apriltag_control_test_py):
            QMessageBox.warning(self._parent, "Missing", f"Not found:\n{self._apriltag_control_test_py}")
            return
        calib = self._get_selected_apriltag_calib()
        if not calib or not os.path.exists(calib):
            QMessageBox.warning(self._parent, "AprilTag calibration",
                                "No AprilTag calibration selected. Run an AprilTag "
                                "calibration first, then click Refresh.")
            return
        # tag ids / sizes / plane default from the calibration's own meta + world map.
        self._spawn_external(f'python -u "{self._apriltag_control_test_py}" --calib "{calib}"')
        self._log("Panel", f"[{self._ts()}] Launched AprilTag control test with:\n  {calib}\n")
