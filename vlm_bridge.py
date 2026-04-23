"""
vlm_bridge.py — tail the harmony_vlm session.jsonl and expose the latest decision.

The decision dict shape is what harmony_vlm/utils/exo_controller.py:1081-1092
serializes via json.dumps(decision): keys include `object`, `second_object`,
`candidates`, `clarification_question`, `waypoints`, and matched `hit_waypoint`.
Parsing is permissive — missing keys return None rather than raise.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional


class VLMBridge:
    def __init__(self, jsonl_path: str | Path, poll_interval_s: float = 0.1) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.poll_interval_s = float(poll_interval_s)

        self._lock = threading.Lock()
        self._latest: Optional[dict] = None
        self._latest_monotonic: Optional[float] = None
        self._new_decision_event = threading.Event()

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._tail_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_latest_decision(self, max_age_s: Optional[float] = None) -> Optional[dict]:
        with self._lock:
            if self._latest is None:
                return None
            if max_age_s is not None and self._latest_monotonic is not None:
                if (time.monotonic() - self._latest_monotonic) > float(max_age_s):
                    return None
            return dict(self._latest)

    def wait_for_next_decision(self, timeout_s: float) -> Optional[dict]:
        self._new_decision_event.clear()
        if self._new_decision_event.wait(timeout=float(timeout_s)):
            return self.get_latest_decision()
        return None

    def _tail_loop(self) -> None:
        # Wait for the file to appear — demo.py creates it only on its first
        # write, so a fresh session will start with the path absent.
        while not self._stop.is_set() and not self.jsonl_path.exists():
            time.sleep(self.poll_interval_s)

        if self._stop.is_set():
            return

        try:
            f = open(self.jsonl_path, "r", encoding="utf-8")
        except OSError as e:
            print(f"[vlm_bridge] open failed: {e}", flush=True)
            return

        try:
            f.seek(0, 2)  # jump to end; only consume new lines from here on
            pending = ""
            while not self._stop.is_set():
                chunk = f.read()
                if chunk:
                    pending += chunk
                    while "\n" in pending:
                        line, pending = pending.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        self._consume_line(line)
                else:
                    time.sleep(self.poll_interval_s)
        finally:
            try:
                f.close()
            except Exception:
                pass

    def _consume_line(self, line: str) -> None:
        try:
            decision = json.loads(line)
        except json.JSONDecodeError:
            print(f"[vlm_bridge] skipping unparseable line: {line[:120]}", flush=True)
            return
        if not isinstance(decision, dict):
            return
        with self._lock:
            self._latest = decision
            self._latest_monotonic = time.monotonic()
        self._new_decision_event.set()


def top_intent(decision: Optional[dict]) -> Optional[str]:
    if not isinstance(decision, dict):
        return None
    cands = decision.get("candidates")
    if not isinstance(cands, list) or not cands:
        return None
    top = cands[0]
    if isinstance(top, dict):
        return top.get("intent")
    return None


def hit_waypoint_xyz(decision: Optional[dict]) -> Optional[tuple[float, float, float]]:
    if not isinstance(decision, dict):
        return None
    wp = decision.get("hit_waypoint")
    if not isinstance(wp, dict):
        return None
    pos = wp.get("position_cam")
    if not isinstance(pos, (list, tuple)) or len(pos) != 3:
        return None
    try:
        return (float(pos[0]), float(pos[1]), float(pos[2]))
    except (TypeError, ValueError):
        return None
