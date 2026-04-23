"""
vlm_launcher.py — spawn the harmony_vlm demo.py as a subprocess.

The VLM runs in a separate conda env (depth-pro pins numpy<2, incompatible
with the BCI stack's pyriemann/opencv in the `lsl` env). All lifecycle
management is process-boundary only; stdout/stderr are kept as pipes so the
control panel can capture them into a log buffer.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


class VLMLauncher:
    def __init__(
        self,
        *,
        repo_dir: str | Path,
        conda_env: str,
        model: str,
        enable_depth: bool,
        session_root: str | Path,
        logger=None,
    ) -> None:
        self.repo_dir = Path(repo_dir)
        self.conda_env = str(conda_env)
        self.model = str(model)
        self.enable_depth = bool(enable_depth)
        self.session_root = Path(session_root)
        self._logger = logger

        self._proc: Optional[subprocess.Popen] = None
        self._session_dir: Optional[Path] = None

        demo = self.repo_dir / "demo.py"
        if not demo.exists():
            raise FileNotFoundError(f"harmony_vlm demo.py not found at {demo}")

    def _log(self, msg: str) -> None:
        if self._logger is not None:
            try:
                self._logger.log_event(msg)
                return
            except Exception:
                pass
        print(msg, flush=True)

    def _build_cmd(self, session_dir: Path) -> list[str]:
        # --host "" forces their discover_one_device fallback at
        # utils/neon/reader.py:224, matching harmony_dev's connection pattern
        # and bypassing their hard-coded dev-machine default at demo.py:109.
        cmd = [
            "conda", "run",
            "--no-capture-output",
            "-n", self.conda_env,
            "python", "-u", "demo.py",
            "--live",
            "--host", "",
            "--vlm-model", self.model,
            "--save-decisions", str(session_dir / "session.jsonl"),
            "--overlay",
        ]
        if self.enable_depth:
            cmd.append("--depth")
        return cmd

    def start(self) -> subprocess.Popen:
        if self.is_running():
            raise RuntimeError("VLMLauncher.start() called while a process is already running")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.session_root / f"session_{ts}"
        session_dir.mkdir(parents=True, exist_ok=True)
        self._session_dir = session_dir

        cmd = self._build_cmd(session_dir)
        self._log(f"[vlm_launcher] starting: cwd={self.repo_dir} cmd={' '.join(cmd)}")

        # start_new_session=True puts the child in its own process group so
        # SIGTERM on the pgid reaches the python worker, not just `conda run`.
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self.repo_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self._log(f"[vlm_launcher] pid={self._proc.pid} session_dir={session_dir}")
        return self._proc

    def stop(self, timeout_s: float = 5.0) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            self._proc = None
            return

        pid = self._proc.pid
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            self._proc = None
            return

        try:
            self._proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self._log(f"[vlm_launcher] pid={pid} did not exit after SIGTERM; sending SIGKILL")
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._log(f"[vlm_launcher] pid={pid} still alive after SIGKILL")
        finally:
            self._proc = None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def proc(self) -> Optional[subprocess.Popen]:
        return self._proc

    @property
    def session_dir(self) -> Optional[Path]:
        return self._session_dir

    @property
    def session_jsonl_path(self) -> Optional[Path]:
        return (self._session_dir / "session.jsonl") if self._session_dir else None
