"""
panel/gaze_controller.py — Gaze-service start-stop / telemetry row for the control panel.

Widget-owning collaborator following the SerialController / DeviceLaunchersController
shape: it builds the Gaze Service row (status LED + Start-Headless / Start-With-UI /
Stop / Query-Telemetry buttons) into the panel's main grid via build_into(), owns
those widgets, and holds the gaze handlers (on_gaze_*, _start_gaze_service,
_gaze_udp_request, _format_gaze_telemetry_line) — all transcribed verbatim from
ControlPanel.

The QProcess lifecycle stays with the panel's ProcessManager + Proc handles
(self.gaze_runner / self.gaze_service), which are shared across the panel (command
wiring at _set_cmds_for_mode_and_driver, subject rotation, _tick, closeEvent); those
are injected here. The controller only reads/mutates their .cmd and drives them via
the injected ProcessManager.

Cross-cutting concerns are injected as callbacks (log / log_ui / set_led / timestamp)
so the controller has no back-reference into the panel beyond a QMessageBox parent.
log_ui is the QTimer-marshaled variant the off-thread Query-Telemetry worker needs to
touch the log buffer safely.

self._gaze_row_widgets is exposed as self.gaze_row_widgets so the panel's
_apply_backend_visibility (which spans gaze + vlm rows and stays on the panel) can
hide the whole block when GAZE_OR_BACKEND == "vlm".
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QGridLayout, QLabel, QMessageBox, QPushButton

from panel.process_manager import Proc, ProcessManager
from panel.netutils import _is_port_in_use
from panel.constants import (
    GAZE_RUNNER_PY, GAZE_SERVICE_PY,
    GAZE_SERVICE_HOST, GAZE_BIND_HOST, GAZE_SERVICE_PORT,
    GAZE_QUERY_TIMEOUT_S, PERCEPTION_FRAME_SOURCE, NEON_COMPANION_HOST,
)

# _HCFG carries the live config object for the remote-frame-source branch of
# _start_gaze_service (FRAME_RELAY_DIAL_HOST / FRAME_RELAY_PORT), mirroring the
# panel's use; netutils re-exports nothing here, so read it from config directly.
try:
    import config as _HCFG
except Exception:
    _HCFG = None


class GazeController(QObject):
    """Owns the Gaze Service row and the gaze handlers.

    Injected dependencies (behaviour-identical to the former in-class calls):
      procs              — the panel's ProcessManager (QProcess lifecycle)
      gaze_runner/service— the Proc handles (kept on the panel, shared elsewhere)
      log(title, text)   — append to the panel's log buffer
      log_ui(title, text)— same, marshaled onto the Qt main thread (worker safe)
      set_led(led, st)   — colour an LED for a state string
      timestamp()        — "HH:MM:SS" for log lines
    """

    def __init__(
        self,
        parent,
        *,
        procs: ProcessManager,
        gaze_runner: Proc,
        gaze_service: Proc,
        log: Callable[[str, str], None],
        log_ui: Callable[[str, str], None],
        set_led: Callable[[object, str], None],
        timestamp: Callable[[], str],
    ) -> None:
        super().__init__(parent)
        self._parent = parent
        self.procs = procs
        self.gaze_runner = gaze_runner
        self.gaze_service = gaze_service
        self._log = log
        self._log_ui = log_ui
        self._set_led = set_led
        self._ts = timestamp

    def build_into(self, grid: QGridLayout, row: int) -> int:
        """Build the Gaze Service row into the panel's main grid starting at
        ``row``; return the next free row. Widget tree + grid placement are
        identical to the former inline _build_ui block."""
        # ===== Gaze Service =====
        # Collected in self.gaze_row_widgets so _apply_backend_visibility()
        # can hide the whole block when GAZE_OR_BACKEND == "vlm". An empty
        # grid row collapses to zero height in Qt once all its items are
        # hidden, so toggling visibility is sufficient.
        self.lbl_gaze_service = QLabel("●"); self._set_led(self.lbl_gaze_service, "stopped")
        gaze_lbl_title = QLabel("<b>Gaze Service</b>")
        grid.addWidget(gaze_lbl_title, row, 0)
        grid.addWidget(self.lbl_gaze_service, row, 1)

        self.btn_gaze_service_headless = QPushButton("Start (Headless)")
        self.btn_gaze_service_ui = QPushButton("Start (With UI)")
        self.btn_gaze_service_stop = QPushButton("Stop")
        self.btn_gaze_service_query = QPushButton("Query Telemetry (UDP)")

        self.btn_gaze_service_headless.clicked.connect(self.on_gaze_service_start_headless)
        self.btn_gaze_service_ui.clicked.connect(self.on_gaze_service_start_ui)
        self.btn_gaze_service_stop.clicked.connect(self.on_gaze_service_stop)
        self.btn_gaze_service_query.clicked.connect(self.on_gaze_service_query)

        grid.addWidget(self.btn_gaze_service_headless, row, 2)
        grid.addWidget(self.btn_gaze_service_ui, row, 3)
        grid.addWidget(self.btn_gaze_service_stop, row, 4)
        row += 1

        gaze_telemetry_lbl = QLabel("<i>Telemetry:</i> view output in View: Gaze")
        grid.addWidget(gaze_telemetry_lbl, row, 0, 1, 2)
        grid.addWidget(self.btn_gaze_service_query, row, 2, 1, 3)
        row += 1

        self.gaze_row_widgets = [
            gaze_lbl_title, self.lbl_gaze_service,
            self.btn_gaze_service_headless, self.btn_gaze_service_ui,
            self.btn_gaze_service_stop,
            gaze_telemetry_lbl, self.btn_gaze_service_query,
        ]
        return row

    # ----- handlers -----
    # ----- Gaze (NEW) -----
    def _ensure_gaze_paths(self, which: str) -> bool:
        path = GAZE_RUNNER_PY if which == "runner" else GAZE_SERVICE_PY
        if not os.path.exists(path):
            QMessageBox.warning(self._parent, "Missing", f"Not found:\n{path}")
            return False
        return True

    def on_gaze_runner_start(self):
        if not self._ensure_gaze_paths("runner"):
            return
        # Runner: UI + prints for testing, but logs are captured into View: Gaze.
        neon_arg = f'--neon-device-host "{NEON_COMPANION_HOST}"' if NEON_COMPANION_HOST else ""
        self.gaze_runner.cmd = f'python -u "{GAZE_RUNNER_PY}" --mode runner --display 1 --prints 1 {neon_arg}'
        self.procs.start(self.gaze_runner, None, "Gaze")
        self._log("Gaze", f"[{self._ts()}] Runner start requested\n")

    def on_gaze_runner_stop(self):
        self.procs.stop(self.gaze_runner, None, "Gaze")

    def on_gaze_service_start_headless(self):
        self._start_gaze_service(display=0)

    def on_gaze_service_start_ui(self):
        self._start_gaze_service(display=1)

    def _start_gaze_service(self, *, display: int):
        if not self._ensure_gaze_paths("service"):
            return

        # Guard: avoid confusing "address already in use" if already running
        if _is_port_in_use(int(GAZE_SERVICE_PORT), GAZE_SERVICE_HOST):
            QMessageBox.warning(
                self._parent,
                "Gaze service port in use",
                f"UDP port {GAZE_SERVICE_HOST}:{GAZE_SERVICE_PORT} appears in use.\n"
                f"If gaze service is already running, use Stop first.\n"
                f"Otherwise change GAZE_SERVICE_PORT."
            )

        # Service: prints can be 0 (supressed) or 1 (verbose) — either way logs go to View: Gaze
        neon_arg = f'--neon-device-host "{NEON_COMPANION_HOST}"' if NEON_COMPANION_HOST else ""
        # GPU-host topology: when PERCEPTION_FRAME_SOURCE=remote, gaze_runner
        # consumes envelopes from the relay instead of opening Neon directly.
        # Dial host comes from FRAME_RELAY_DIAL_HOST in config.
        remote_arg = ""
        if PERCEPTION_FRAME_SOURCE == "remote":
            relay_dial = str(getattr(_HCFG, "FRAME_RELAY_DIAL_HOST", "127.0.0.1") or "127.0.0.1") if _HCFG else "127.0.0.1"
            relay_port = int(getattr(_HCFG, "FRAME_RELAY_PORT", 5591)) if _HCFG else 5591
            remote_arg = (
                f'--frame-source remote '
                f'--remote-frame-host {relay_dial} '
                f'--remote-frame-port {relay_port}'
            )
        self.gaze_service.cmd = (
            f'python -u "{GAZE_SERVICE_PY}" --mode service '
            f'--display {int(display)} --prints 1 '
            f'--host {GAZE_BIND_HOST} --port {int(GAZE_SERVICE_PORT)} '
            f'--udp_log 1 --udp_log_hz 50 {neon_arg} {remote_arg}'
        )
        self.procs.start(self.gaze_service, self.lbl_gaze_service, "Gaze")
        self._log("Gaze", f"[{self._ts()}] Service start requested (display={display})\n")

    def on_gaze_service_stop(self):
        self.procs.stop(self.gaze_service, self.lbl_gaze_service, "Gaze")

    def on_gaze_service_query(self):
        import threading
        query_id = int(time.time() * 1000)

        # TX log (already correct)
        self._log("Panel",
            f"[{self._ts()}] Gaze UDP TX query_id={query_id} -> {GAZE_SERVICE_HOST}:{GAZE_SERVICE_PORT}\n"
        )

        def worker():
            t0 = time.time()
            try:
                req = {"cmd": "snapshot", "include_objects": True, "query_id": query_id}
                resp = self._gaze_udp_request(req, timeout_s=float(GAZE_QUERY_TIMEOUT_S))

                dt_ms = (time.time() - t0) * 1000.0
                pretty = json.dumps(resp, indent=2, sort_keys=True)

                msg = (
                    f"[{self._ts()}] Gaze UDP RX OK query_id={query_id} "
                    f"({dt_ms:.0f} ms)\n{pretty}\n"
                )

                # existing
                self._log_ui("Gaze", msg)

                # ✅ ADD THIS LINE — this is all you need
                self._log_ui("Panel",
                    f"[{self._ts()}] Gaze UDP RX OK query_id={query_id} ({dt_ms:.0f} ms)\n"
                )

            except Exception as e:
                dt_ms = (time.time() - t0) * 1000.0
                err = (
                    f"[{self._ts()}] Gaze UDP RX ERROR query_id={query_id} "
                    f"({dt_ms:.0f} ms): {e}\n"
                )

                self._log_ui("Panel", err)
                self._log_ui("Gaze", err)

        threading.Thread(target=worker, daemon=True).start()

    def _format_gaze_telemetry_line(self, snap: dict) -> str:
        """
        Build a line similar to your terminal prints, using fields from telemetry JSON.
        This requires the service to include these keys in its response.
        """
        def _f(key, default="--"):
            v = snap.get(key, None)
            if v is None:
                return default
            return v

        t = _f("t", None)
        t_txt = f"t={t:.3f}" if isinstance(t, (int, float)) else f"t={t}"

        worn = bool(snap.get("worn", False))
        gaze_px = snap.get("gaze_px", None)
        gaze_txt = f"gaze=({gaze_px[0]:.1f},{gaze_px[1]:.1f})" if isinstance(gaze_px, (list, tuple)) and len(gaze_px) >= 2 else "gaze=(--,--)"

        loop_hz = snap.get("loop_hz", float("nan"))
        video_hz = snap.get("video_hz", float("nan"))
        det_hz = snap.get("det_hz", float("nan"))
        det_age_s = snap.get("det_age_s", float("nan"))
        infer_ms = snap.get("infer_ms", float("nan"))

        imu_w = snap.get("imu_angvel", None)
        imu_txt = f"imu|w|={imu_w:.2f}rad/s" if isinstance(imu_w, (int, float)) else "imu|w|=--"

        yolo_enabled = bool(snap.get("gov_enabled", True))
        reason = str(snap.get("gov_reason", "healthy"))
        cd = float(snap.get("gov_cd_left", 0.0))
        yolo_txt = f"YOLO={'ON' if yolo_enabled else 'OFF'}({reason}) cd={cd:.2f}s"

        tracks = snap.get("tracks", None)
        if tracks is None:
            # fallback: infer from objects list if present
            objs = snap.get("objects", None)
            tracks = len(objs) if isinstance(objs, list) else 0

        objs = snap.get("objects", None)
        if isinstance(objs, list) and len(objs) > 0:
            # expect entries like {"name": "...", "track_id": 1, "conf": 0.95}
            parts = []
            for o in objs[:6]:
                nm = str(o.get("name", "?"))
                tid = o.get("track_id", None)
                cf = o.get("conf", None)
                if tid is not None and cf is not None:
                    parts.append(f"{nm}#{int(tid)}({float(cf):.2f})")
                elif cf is not None:
                    parts.append(f"{nm}({float(cf):.2f})")
                else:
                    parts.append(nm)
            objs_txt = "objs: " + ", ".join(parts)
        else:
            objs_txt = "objs: none"

        hit = snap.get("gaze_hit", None)
        if isinstance(hit, dict):
            nm = str(hit.get("name", "none"))
            tid = hit.get("track_id", None)
            cf = hit.get("conf", None)
            if tid is not None and cf is not None:
                gaze_on = f"gaze_on: {nm}#{int(tid)}({float(cf):.2f})"
            elif cf is not None:
                gaze_on = f"gaze_on: {nm}({float(cf):.2f})"
            else:
                gaze_on = f"gaze_on: {nm}"
            mode = str(hit.get("mode", "--"))
            d = hit.get("dist_px", None)
            d_txt = f"{float(d):.1f}px" if isinstance(d, (int, float)) else "--"
            gaze_on = f"{gaze_on} mode={mode} d={d_txt}"
        else:
            gaze_on = "gaze_on: none"

        depth_cm = snap.get("depth_cm", float("nan"))
        miss_mm = snap.get("miss_mm", float("nan"))
        ipd_mm = snap.get("ipd_mm", float("nan"))
        depth_txt = f"depth={depth_cm:.1f}cm miss={miss_mm:.1f}mm IPD={ipd_mm:.1f}mm" if all(isinstance(x, (int, float)) for x in [depth_cm, miss_mm, ipd_mm]) else "depth=--"

        hy = snap.get("head_yaw_deg", float("nan"))
        hp = snap.get("head_pitch_deg", float("nan"))
        gy = snap.get("gaze_yaw_deg", float("nan"))
        gp = snap.get("gaze_pitch_deg", float("nan"))
        head_txt = f"HEAD(yaw,pitch)=({hy:+.0f},{hp:+.0f})"
        gaze_ang_txt = f"GAZE(yaw,pitch)=({gy:+.0f},{gp:+.0f})"

        return (
            f"{t_txt} worn={worn} {gaze_txt} | "
            f"rates: loop={loop_hz:.1f}Hz video={video_hz:.1f}Hz det~{det_hz:.1f}Hz det_age={det_age_s:.2f}s infer={infer_ms:.0f}ms | "
            f"{imu_txt} | {yolo_txt} | tracks={tracks} | {objs_txt} | {gaze_on} | {depth_txt} | {head_txt} {gaze_ang_txt}"
        )

    def _gaze_udp_request(self, payload: dict, timeout_s: float = 0.8) -> dict:
        """One-shot JSON request against gaze_runner.py (service mode).

        Delegates to ``Utils.perception_clients.udp_request`` to keep the
        wire format colocated with the VLM client and reusable from the
        experiment driver.
        """
        from Utils.perception_clients import udp_request
        return udp_request(GAZE_SERVICE_HOST, int(GAZE_SERVICE_PORT), payload, float(timeout_s))
