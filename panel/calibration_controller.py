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
        apriltag_control_test_3d_py: str = "",
    ) -> None:
        super().__init__(parent)
        self._parent = parent
        self._root = root
        self._harmony_calibration_exec_py = harmony_calibration_exec_py
        self._harmony_online_control_py = harmony_online_control_py
        self._apriltag_calibrate_py = apriltag_calibrate_py
        self._apriltag_control_test_py = apriltag_control_test_py
        self._apriltag_control_test_3d_py = apriltag_control_test_3d_py
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

        # WS4 REV06 depth-free 3-D AprilTag gaze→robot calibration (canonical
        # 2026-06-30, rig-verified). End-to-end from this tab: (1) register the static
        # world tags as a true-3-D map (table + wall groups → snapped coplanar +
        # table⟂wall squared), camera-only; (2) the seated swept calibration over the
        # held-object tag → 3-D (x,y,z)→Q library, which AUTO-RUNS the telemetry scale
        # gate at the end (scale≈1.0 = metrically correct); (3) the 3-D control test
        # drives the robot from the selected calibration (depth-free gaze_height).
        # Tag ids/sizes come from config (APRILTAG_TABLE/WALL/OBJECT/STABILIZER_*).
        # Both dropdowns list runs/ newest-first, defaulting to the latest after Refresh.
        wm_row = QHBoxLayout()
        wm_row.addWidget(QLabel("World map:"))
        self.cmb_world_map = QComboBox()
        wm_row.addWidget(self.cmb_world_map, 1)
        self.btn_refresh_world_maps = QPushButton("Refresh")
        self.btn_refresh_world_maps.setMaximumWidth(90)
        self.btn_refresh_world_maps.clicked.connect(self.on_refresh_world_maps)
        wm_row.addWidget(self.btn_refresh_world_maps)
        hb.addLayout(wm_row)

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
        self.btn_register_world = QPushButton("1. Register world map (3-D)")
        self.btn_register_world.setMaximumWidth(240)
        self.btn_register_world.clicked.connect(self.on_register_world_map)
        self.btn_run_apriltag_calibrate = QPushButton("2. Run 3-D calibration (sweep→solve)")
        self.btn_run_apriltag_calibrate.setMaximumWidth(260)
        self.btn_run_apriltag_calibrate.clicked.connect(self.on_run_apriltag_calibrate)
        self.btn_run_apriltag_control = QPushButton("3. Run 3-D control test")
        self.btn_run_apriltag_control.setMaximumWidth(220)
        self.btn_run_apriltag_control.clicked.connect(self.on_run_apriltag_control_test)
        atag_row.addWidget(self.btn_register_world)
        atag_row.addWidget(self.btn_run_apriltag_calibrate)
        atag_row.addWidget(self.btn_run_apriltag_control)
        atag_row.addStretch(1)
        hb.addLayout(atag_row)
        parent_layout.addWidget(harmony_box)
        self.on_refresh_world_maps()      # populate from existing runs/ at startup

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

    def on_refresh_world_maps(self):
        """List runs/world_map_*.npz newest-first so the dropdown defaults to the most
        recent top-down registration (rev05 §2A). The world tags are static, so a map
        is reused across seated sessions until a tag is physically bumped."""
        if not hasattr(self, "cmb_world_map"):
            return
        current = self.cmb_world_map.currentData()
        self.cmb_world_map.clear()
        maps = sorted(glob.glob(os.path.join(self._root, "runs", "world_map_*.npz")),
                      key=os.path.getmtime, reverse=True)
        if not maps:
            self.cmb_world_map.addItem("No world maps — register one first", "")
            self._log("Panel", f"[{self._ts()}] No world maps (runs/world_map_*.npz) "
                      "yet — click 'Register world map (top-down)' first\n")
            return
        for m in maps:
            self.cmb_world_map.addItem(os.path.basename(m), m)
        if current:
            idx = self.cmb_world_map.findData(current)
            if idx >= 0:
                self.cmb_world_map.setCurrentIndex(idx)
        self._log("Panel", f"[{self._ts()}] World maps: {len(maps)} "
                  f"(newest first → {os.path.basename(maps[0])})\n")

    def _get_selected_world_map(self) -> str:
        if not hasattr(self, "cmb_world_map"):
            return ""
        return self.cmb_world_map.currentData() or ""

    def on_register_world_map(self):
        """Step 1 (REV06 §3): register the static world tags as a true-3-D map — camera
        only, no robot. The table + wall groups (config) are snapped coplanar and the
        table↔wall corner squared to 90°; saves runs/world_map_3d_<UTC>.npz. Move your
        head over MANY heights + angles until every tag goes green, SPACE to accept,
        then click Refresh next to 'World map'."""
        if not os.path.exists(self._apriltag_calibrate_py):
            QMessageBox.warning(self._parent, "Missing", f"Not found:\n{self._apriltag_calibrate_py}")
            return
        table = [int(i) for i in getattr(self._hcfg, "APRILTAG_TABLE_TAG_IDS", [0, 1, 2, 3, 4, 12])]
        wall = [int(i) for i in getattr(self._hcfg, "APRILTAG_WALL_TAG_IDS", [6, 7, 8, 9, 10, 11])]
        world = sorted(set(table) | set(wall))  # the full world map = both groups
        tag = float(getattr(self._hcfg, "APRILTAG_TAG_SIZE_M", 0.08))
        cmd = (f'python -u "{self._apriltag_calibrate_py}" --stage register-world-3d '
               f'--world-tag-ids {" ".join(map(str, world))} '
               f'--table-plane-tag-ids {" ".join(map(str, table))} '
               f'--wall-plane-tag-ids {" ".join(map(str, wall))} '
               f'--tag-size {tag} --out-dir runs')
        self._spawn_external(cmd)
        self._log("Panel", f"[{self._ts()}] Launched 3-D world-map registration "
                  f"(table {table}, wall {wall}; no robot). Move your head over MANY "
                  "heights/angles until the tags go green, SPACE to accept, then click "
                  "Refresh next to 'World map'.\n")

    def on_run_apriltag_calibrate(self):
        """Step 2 (REV06): seated swept calibration over the held-object tag against the
        selected 3-D world map, chained in ONE terminal as sweep → solve-3d → telemetry
        scale gate. The sweep keeps the EE point in full 3-D (--coverage-3d), solve-3d
        builds the (x,y,z)→Q library, and Analyze_handeye_axzb prints the telemetry
        similarity check (scale≈1.0 + residual mm = metrically correct; a low scale is
        a silent fail the vision-only gates can't see, rev05 §5.5). Coverage/display are
        telemetry-anchored (rev05 §2C). The object's controlled point IS the tag, so the
        EE offset is 0 (the object tag sits where we grasp)."""
        if not os.path.exists(self._apriltag_calibrate_py):
            QMessageBox.warning(self._parent, "Missing", f"Not found:\n{self._apriltag_calibrate_py}")
            return
        world_map = self._get_selected_world_map()
        if not world_map or not os.path.exists(world_map):
            QMessageBox.warning(self._parent, "World map",
                                "No world map selected. Click '1. Register world map "
                                "(3-D)' first, then Refresh the 'World map' dropdown and "
                                "select the newest world_map_3d_*.npz.")
            return
        table = [int(i) for i in getattr(self._hcfg, "APRILTAG_TABLE_TAG_IDS", [0, 1, 2, 3, 4, 12])]
        wall = [int(i) for i in getattr(self._hcfg, "APRILTAG_WALL_TAG_IDS", [6, 7, 8, 9, 10, 11])]
        world = " ".join(map(str, sorted(set(table) | set(wall))))
        obj = " ".join(str(int(i)) for i in getattr(self._hcfg, "APRILTAG_OBJECT_TAG_IDS", [18]))
        stab = int(getattr(self._hcfg, "APRILTAG_STABILIZER_TAG_ID", 5))
        tag = float(getattr(self._hcfg, "APRILTAG_TAG_SIZE_M", 0.08))
        ee_tag = float(getattr(self._hcfg, "APRILTAG_EE_TAG_SIZE_M", 0.04))
        cal = self._apriltag_calibrate_py
        handeye = os.path.join(self._root, "Analyze_handeye_axzb.py")
        sweep = (f'python -u "{cal}" --stage sweep --with-robot --coverage-3d '
                 f'--world-map "{world_map}" --world-tag-ids {world} '
                 f'--ee-tag-ids {obj} --ee-tag-size {ee_tag} '
                 f'--ee-point-method pose --t-eetag-ee 0 0 0 '
                 f'--stabilizer-tag-id {stab} --stabilizer-tag-size {ee_tag} '
                 f'--tag-size {tag} --out-dir runs')
        # Chain the three steps in one terminal. $NPZ binds to the just-written sweep so
        # solve-3d + the scale gate run on it; the '\\$' survives _spawn_external (which
        # only escapes "), so the inner bash -lc — not the launching /bin/sh — expands it.
        chain = (f'{sweep} && '
                 f'NPZ=\\$(ls -t runs/apriltag_sweep_*.npz | head -1) && '
                 f'echo "── solve-3d on \\$NPZ ──" && '
                 f'python -u "{cal}" --stage solve-3d \\$NPZ && '
                 f'echo "── telemetry scale gate (want scale ~1.0) ──" && '
                 f'python -u "{handeye}" --npz \\$NPZ')
        self._spawn_external(chain)
        self._log("Panel", f"[{self._ts()}] Launched 3-D calibration on "
                  f"{os.path.basename(world_map)}: sweep (object {obj}, stabilizer "
                  f"{stab}) → solve-3d → telemetry scale gate. Sweep LOW + fill the "
                  "corners; watch the final 'scale ~1.0'. Refresh 'AprilTag calib' when "
                  "done.\n")

    def on_run_apriltag_control_test(self):
        """Step 3 (REV06): drive the robot from the selected 3-D calibration. The
        depth-free target is the gazed object's footprint∩table plus the gaze-ray height
        (--target-source object_plane --target-point gaze_height), so objects above the
        table are reachable without a depth sensor. Tag ids/sizes default from config;
        drive from the ControlView window (ENTER resolve, g GO, r re-resolve, h home,
        q quit)."""
        tool = self._apriltag_control_test_3d_py or self._apriltag_control_test_py
        if not os.path.exists(tool):
            QMessageBox.warning(self._parent, "Missing", f"Not found:\n{tool}")
            return
        calib = self._get_selected_apriltag_calib()
        if not calib or not os.path.exists(calib):
            QMessageBox.warning(self._parent, "AprilTag calibration",
                                "No 3-D calibration selected. Run '2. Run 3-D "
                                "calibration' first, then click Refresh.")
            return
        table = " ".join(str(int(i)) for i in getattr(self._hcfg, "APRILTAG_TABLE_TAG_IDS", [0, 1, 2, 3, 4, 12]))
        stab = int(getattr(self._hcfg, "APRILTAG_STABILIZER_TAG_ID", 5))
        ee_tag = float(getattr(self._hcfg, "APRILTAG_EE_TAG_SIZE_M", 0.04))
        cmd = (f'python -u "{tool}" --calib "{calib}" '
               f'--target-source object_plane --target-point gaze_height '
               f'--table-tag-ids {table} --ee-tag-id {stab} --ee-tag-size {ee_tag}')
        self._spawn_external(cmd)
        self._log("Panel", f"[{self._ts()}] Launched 3-D control test (gaze_height) "
                  f"with:\n  {calib}\nDrive from the ControlView window: ENTER resolve, "
                  "g GO, r re-resolve, h home, q quit.\n")
