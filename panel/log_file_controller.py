"""
panel/log_file_controller.py — subject-tied on-disk log files for the control
panel (the VLM-panel + frame-relay channels).

Non-widget collaborator following the established controller shape: it owns the
two subject-tied file handles (vlm_log_fh / relay_log_fh) and their open/close/
rotate logic plus the relay log callback (the shared stdout sink installed on
Utils.frame_relay + Utils.scene_only_neon_reader). All handler bodies are
transcribed verbatim from ControlPanel.

The file handles are exposed as public attributes so the panel's log sink
(_append_log) can tee the "VLM" / "Relay" buffers to disk exactly as before — the
tee logic stays on the panel, only the handle reference is repointed here. The
panel keeps owning when files open (startup), rotate (subject save) and close
(closeEvent); it just calls open_vlm/open_relay/close_vlm/close_relay.

Cross-cutting concerns are injected as callbacks (log / timestamp). DATA_DIR and
the config-snapshot header values come from panel.constants. The controller is a
QObject so QTimer.singleShot can marshal the off-thread relay callback onto the
GUI thread, the same pattern the former in-class method used.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

from PySide6.QtCore import QObject, QTimer

from panel.constants import (
    _HCFG,
    GAZE_OR_BACKEND, PERCEPTION_FRAME_SOURCE, SERVICES_HOSTED_REMOTELY,
    VLM_SERVICE_HOST, VLM_MODEL,
    FRAME_RELAY_BIND_HOST, FRAME_RELAY_PORT, FRAME_RELAY_HZ,
    FRAME_RELAY_EMBEDDED, NEON_COMPANION_HOST,
)


class LogFileController(QObject):
    """Owns the subject-tied VLM-panel + frame-relay log files.

    Injected dependencies (behaviour-identical to the former in-class calls):
      log(title, text)   — append to the panel's log buffer
      timestamp()        — "HH:MM:SS" for log lines

    Public attributes the panel's _append_log tees to:
      vlm_log_fh / relay_log_fh — open file handles (or None when not logging)
    """

    def __init__(
        self,
        parent,
        *,
        log: Callable[[str, str], None],
        timestamp: Callable[[], str],
    ) -> None:
        super().__init__(parent)
        self._parent = parent
        self._log = log
        self._ts = timestamp

        # Subject-tied VLM log file. Captures only the "VLM" buffer
        # (vlm_service stdout when local + every panel-side UDP TX/RX
        # trace + the periodic seg-stream readouts). Other buffers are
        # intentionally NOT teed here — Marker/FES/Driver have their
        # own files via their respective scripts; Robot/Gaze/Panel
        # stay in-memory only for now (revisit if a forensic need
        # appears). Path mirrors marker_logs / impedance_logs naming.
        self._vlm_log_subject: Optional[str] = None
        self._vlm_log_path: Optional[str] = None
        self.vlm_log_fh = None
        # Subject-tied frame_relay log file. Co-located with the VLM
        # log under <DATA_DIR>/sub-<SUBJECT>/vlm_logs/ — the relay is
        # the upstream half of the same perception pipeline, so
        # keeping the two files together makes post-session forensics
        # easier. Default file naming: frame_relay_<timestamp>.log.
        self._relay_log_subject: Optional[str] = None
        self._relay_log_path: Optional[str] = None
        self.relay_log_fh = None

    def open_vlm(self, subject: str) -> None:
        """Open (or rotate to) the subject-tied VLM log file under
        ``<DATA_DIR>/sub-<SUBJECT>/vlm_logs/``. Closes any prior handle
        first so a subject change cleanly switches files. Safe to call
        before _HCFG / DATA_DIR is set — it just no-ops in that case
        and logs land only in the in-memory panel buffer.
        """
        self.close_vlm()
        if not subject or _HCFG is None:
            return
        data_dir = os.path.expanduser(getattr(_HCFG, "DATA_DIR", "") or "")
        if not data_dir:
            return
        try:
            log_dir = os.path.join(data_dir, f"sub-{subject}", "vlm_logs")
            os.makedirs(log_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(log_dir, f"vlm_panel_{ts}.log")
            fh = open(path, "a", encoding="utf-8")
            # Header — one line per panel launch / subject rotation so a
            # later reader can match the file to a config snapshot.
            fh.write(
                f"# vlm_panel log opened {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"# subject={subject}\n"
                f"# GAZE_OR_BACKEND={GAZE_OR_BACKEND}\n"
                f"# PERCEPTION_FRAME_SOURCE={PERCEPTION_FRAME_SOURCE}\n"
                f"# SERVICES_HOSTED_REMOTELY={SERVICES_HOSTED_REMOTELY}\n"
                f"# VLM_SERVICE_HOST={VLM_SERVICE_HOST}\n"
                f"# VLM_MODEL={VLM_MODEL}\n"
            )
            fh.flush()
        except OSError as e:
            # Disk full / permission denied / etc. — surface in the panel
            # buffer and keep running with file logging disabled.
            self._log(
                "Panel",
                f"[{self._ts()}] WARN: could not open VLM log file: {e}\n",
            )
            return
        self._vlm_log_subject = subject
        self._vlm_log_path = path
        self.vlm_log_fh = fh

    def close_vlm(self) -> None:
        fh = self.vlm_log_fh
        if fh is not None:
            try:
                fh.write(f"# vlm_panel log closed {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                fh.close()
            except OSError:
                pass
        self.vlm_log_fh = None
        self._vlm_log_path = None
        self._vlm_log_subject = None

    def open_relay(self, subject: str) -> None:
        """Open (or rotate to) the subject-tied frame_relay log file
        under ``<DATA_DIR>/sub-<SUBJECT>/vlm_logs/``. Mirrors
        :meth:`open_vlm` exactly, just with a distinct
        filename prefix so the two channels don't collide. Co-located
        intentionally — relay + vlm_service are halves of one pipeline.
        """
        self.close_relay()
        if not subject or _HCFG is None:
            return
        data_dir = os.path.expanduser(getattr(_HCFG, "DATA_DIR", "") or "")
        if not data_dir:
            return
        try:
            log_dir = os.path.join(data_dir, f"sub-{subject}", "vlm_logs")
            os.makedirs(log_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(log_dir, f"frame_relay_{ts}.log")
            fh = open(path, "a", encoding="utf-8")
            fh.write(
                f"# frame_relay log opened {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"# subject={subject}\n"
                f"# FRAME_RELAY_BIND_HOST={FRAME_RELAY_BIND_HOST}\n"
                f"# FRAME_RELAY_PORT={FRAME_RELAY_PORT}\n"
                f"# FRAME_RELAY_HZ={FRAME_RELAY_HZ}\n"
                f"# FRAME_RELAY_EMBEDDED={FRAME_RELAY_EMBEDDED}\n"
                f"# NEON_COMPANION_HOST={NEON_COMPANION_HOST}\n"
                f"# PERCEPTION_FRAME_SOURCE={PERCEPTION_FRAME_SOURCE}\n"
            )
            fh.flush()
        except OSError as e:
            self._log(
                "Panel",
                f"[{self._ts()}] WARN: could not open relay log file: {e}\n",
            )
            return
        self._relay_log_subject = subject
        self._relay_log_path = path
        self.relay_log_fh = fh

    def close_relay(self) -> None:
        fh = self.relay_log_fh
        if fh is not None:
            try:
                fh.write(f"# frame_relay log closed {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                fh.close()
            except OSError:
                pass
        self.relay_log_fh = None
        self._relay_log_path = None
        self._relay_log_subject = None

    def relay_callback(self, line: str) -> None:
        """Shared sink for both ``Utils.frame_relay._log`` and
        ``Utils.scene_only_neon_reader._log``. Routes both to the
        "Relay" channel because reader + relay are halves of the same
        upstream pipeline; the line's source is already self-evident
        from its embedded prefix (``[frame_relay] …`` vs.
        ``[scene_only_neon_reader] …``).

        Called from worker threads (relay's pump thread and per-client
        send threads) as well as the UI thread (the reader is
        constructed synchronously from VLMSceneWidget._start_embedded_relay
        on the main thread). ``QTimer.singleShot`` is safe in both
        cases — it's the same marshalling pattern as
        :meth:`_append_log_ui` — and it prepends a ``[HH:MM:SS]``
        stamp so the in-panel buffer and the on-disk file both have
        time context. Standalone CLI usage of either module is
        unaffected because the default sink stays in place there.
        """
        stamped = f"[{self._ts()}] {line}\n"
        QTimer.singleShot(0, self, lambda: self._log("Relay", stamped))
