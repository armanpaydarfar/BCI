"""
vlm_launcher.py — spawn vlm_service.py as a managed subprocess.

The service runs in the `harmony_vlm` conda env (separate from `lsl` because
depth-pro pins numpy<2, incompatible with pyriemann/opencv in the BCI stack).
stdout/stderr are redirected to files under the session directory so the
child can't deadlock on a full pipe buffer and so the control panel can tail
the logs independently.

wait_until_ready() polls the service's UDP `status` command until it responds
or the budget elapses — lets callers block on "models loaded + Neon live"
rather than guessing with a fixed sleep.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

_IS_WINDOWS = sys.platform == "win32"


class VLMLauncher:
    def __init__(
        self,
        *,
        conda_env: str,
        model: str,
        enable_depth: bool,
        session_root: str | Path,
        service_script: str | Path,
        service_host: str = "127.0.0.1",
        service_port: int = 5589,
        neon_host: str = "",
        device: str = "cpu",
        enable_overlay: bool = False,
        overlay_port: int = 5590,
        logger=None,
    ) -> None:
        self.conda_env = str(conda_env)
        self.model = str(model)
        self.enable_depth = bool(enable_depth)
        self.session_root = Path(session_root)
        self.service_script = Path(service_script)
        self.service_host = str(service_host)
        self.service_port = int(service_port)
        self.neon_host = str(neon_host)
        self.device = str(device)
        self.enable_overlay = bool(enable_overlay)
        self.overlay_port = int(overlay_port)
        self._logger = logger

        self._proc: Optional[subprocess.Popen] = None
        self._session_dir: Optional[Path] = None
        self._stdout_fh = None
        self._stderr_fh = None

        if not self.service_script.exists():
            raise FileNotFoundError(f"vlm_service script not found at {self.service_script}")

    def _log(self, msg: str) -> None:
        if self._logger is not None:
            try:
                self._logger.log_event(msg)
                return
            except Exception:
                pass
        print(msg, flush=True)

    def _build_cmd(self, session_dir: Path) -> list[str]:
        cmd = [
            "conda", "run",
            "--no-capture-output",
            "-n", self.conda_env,
            "python", "-u", str(self.service_script),
            "--host", self.service_host,
            "--port", str(self.service_port),
            "--neon-host", self.neon_host,
            "--model", self.model,
            "--device", self.device,
            "--session-dir", str(session_dir),
        ]
        cmd.append("--enable-depth" if self.enable_depth else "--no-enable-depth")
        if self.enable_overlay:
            cmd.extend(["--enable-overlay", "--overlay-port", str(self.overlay_port)])
        return cmd

    def start(self) -> subprocess.Popen:
        if self.is_running():
            raise RuntimeError("VLMLauncher.start() called while a process is already running")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.session_root / f"session_{ts}"
        session_dir.mkdir(parents=True, exist_ok=True)
        self._session_dir = session_dir

        cmd = self._build_cmd(session_dir)
        self._log(f"[vlm_launcher] starting: {' '.join(cmd)}")

        self._stdout_fh = open(session_dir / "vlm.stdout.log", "wb")
        self._stderr_fh = open(session_dir / "vlm.stderr.log", "wb")

        # Put the child in its own process group so a single signal reaches
        # the python worker, not just `conda run`. On POSIX use start_new_session;
        # on Windows use CREATE_NEW_PROCESS_GROUP and tear down with taskkill /T.
        popen_kwargs = dict(
            cwd=str(self.service_script.parent),
            stdout=self._stdout_fh,
            stderr=self._stderr_fh,
        )
        if _IS_WINDOWS:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
        self._proc = subprocess.Popen(cmd, **popen_kwargs)
        self._log(f"[vlm_launcher] pid={self._proc.pid} session_dir={session_dir}")
        return self._proc

    def wait_until_ready(self, timeout_s: float = 60.0, poll_interval_s: float = 0.5) -> bool:
        deadline = time.monotonic() + float(timeout_s)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.3)
        try:
            while time.monotonic() < deadline:
                if not self.is_running():
                    return False
                try:
                    sock.sendto(
                        json.dumps({"cmd": "status"}).encode("utf-8"),
                        (self.service_host, self.service_port),
                    )
                    data, _ = sock.recvfrom(65535)
                    resp = json.loads(data.decode("utf-8", errors="replace"))
                    if resp.get("ok") and resp.get("neon_connected"):
                        return True
                except (socket.timeout, OSError):
                    pass
                except json.JSONDecodeError:
                    pass
                time.sleep(poll_interval_s)
            return False
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def stop(self, timeout_s: float = 5.0) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            self._close_log_files()
            self._proc = None
            return

        pid = self._proc.pid
        if _IS_WINDOWS:
            self._stop_windows(pid, timeout_s)
        else:
            self._stop_posix(pid, timeout_s)
        self._close_log_files()
        self._proc = None

    def _stop_posix(self, pid: int, timeout_s: float) -> None:
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return

        try:
            self._proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self._log(f"[vlm_launcher] pid={pid} did not exit after SIGTERM; sending SIGKILL")
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                return
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._log(f"[vlm_launcher] pid={pid} still alive after SIGKILL")

    def _stop_windows(self, pid: int, timeout_s: float) -> None:
        # taskkill /T walks the job tree from `conda run` down to the python
        # worker; /F is force. Plain SIGTERM via Popen.terminate() only kills
        # the conda wrapper and orphans the worker.
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            self._log(f"[vlm_launcher] pid={pid} taskkill timed out")
        try:
            self._proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self._log(f"[vlm_launcher] pid={pid} still alive after taskkill")

    def _close_log_files(self) -> None:
        for fh in (self._stdout_fh, self._stderr_fh):
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass
        self._stdout_fh = None
        self._stderr_fh = None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def proc(self) -> Optional[subprocess.Popen]:
        return self._proc

    @property
    def session_dir(self) -> Optional[Path]:
        return self._session_dir
