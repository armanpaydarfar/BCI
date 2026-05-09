"""
vlm_bridge.py — UDP client for vlm_service.py.

One method per service command. Each call is a single JSON request/response
round-trip with a command-specific recv timeout, because latencies differ by
orders of magnitude (status <50ms, depth 1-3s, reason 2-10s).

Mirrors the `gaze_udp_request` pattern in
ExperimentDriver_Online_GazeTracking.py:347-377 so consumers see the same
shape of interaction (send JSON, receive JSON, bool-check `ok`).
"""

from __future__ import annotations

import json
import socket
from typing import Any, Dict, Optional


class VLMBridge:
    def __init__(self, host: str = "127.0.0.1", port: int = 5589, default_timeout_s: float = 0.5) -> None:
        self.host = str(host)
        self.port = int(port)
        self.default_timeout_s = float(default_timeout_s)
        self._sock: Optional[socket.socket] = None

    def _get_sock(self) -> socket.socket:
        if self._sock is None:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return self._sock

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _request(self, cmd: str, params: Optional[dict] = None, *, recv_timeout_s: Optional[float] = None) -> Optional[dict]:
        payload = {"cmd": cmd}
        if params:
            payload.update(params)
        s = self._get_sock()
        s.settimeout(float(recv_timeout_s if recv_timeout_s is not None else self.default_timeout_s))
        try:
            s.sendto(json.dumps(payload).encode("utf-8"), (self.host, self.port))
            data, _ = s.recvfrom(65535)
            return json.loads(data.decode("utf-8", errors="replace"))
        except (socket.timeout, OSError, json.JSONDecodeError):
            return None

    # ── commands ──────────────────────────────────────────────────────────

    def status(self, recv_timeout_s: float = 0.3) -> Optional[dict]:
        return self._request("status", recv_timeout_s=recv_timeout_s)

    def snapshot(self, recv_timeout_s: float = 0.3) -> Optional[dict]:
        return self._request("snapshot", recv_timeout_s=recv_timeout_s)

    def segment(self, *, include_masks: bool = False, recv_timeout_s: float = 3.0) -> Optional[dict]:
        return self._request("segment", {"include_masks": include_masks}, recv_timeout_s=recv_timeout_s)

    def depth(self, *, at_gaze: bool = True, save: bool = False, recv_timeout_s: float = 10.0) -> Optional[dict]:
        return self._request("depth", {"at_gaze": at_gaze, "save": save}, recv_timeout_s=recv_timeout_s)

    def reason(self, *, include_segments: bool = True, timeout: float = 30.0, recv_timeout_s: float = 35.0) -> Optional[dict]:
        return self._request(
            "reason",
            {"include_segments": include_segments, "timeout": timeout},
            recv_timeout_s=recv_timeout_s,
        )

    def decide(self, *, timeout: float = 30.0, recv_timeout_s: float = 40.0) -> Optional[dict]:
        return self._request("decide", {"timeout": timeout}, recv_timeout_s=recv_timeout_s)

    def capture_first(self, *, recv_timeout_s: float = 10.0) -> Optional[dict]:
        """Snapshot the current frame+gaze+detections+waypoints under a server-side
        token. Pair with decide_pair(snapshot_id=...) after the user's second gaze."""
        return self._request("capture_first", recv_timeout_s=recv_timeout_s)

    def decide_pair(self, *, snapshot_id: str, timeout: float = 45.0, recv_timeout_s: float = 60.0) -> Optional[dict]:
        """Combine the cached first snapshot with the current frame+gaze and run
        reason_async_pair. Returns the paired decision including first_waypoint
        and second_waypoint (both in Neon camera frame, metres)."""
        return self._request(
            "decide_pair",
            {"snapshot_id": str(snapshot_id), "timeout": timeout},
            recv_timeout_s=recv_timeout_s,
        )

    def camera_matrix(self, recv_timeout_s: float = 0.3) -> Optional[dict]:
        return self._request("camera_matrix", recv_timeout_s=recv_timeout_s)

    def stop_service(self, recv_timeout_s: float = 1.0) -> Optional[dict]:
        return self._request("stop", recv_timeout_s=recv_timeout_s)


# ── convenience helpers for decision payloads ────────────────────────────

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
    """Return the gaze-matched waypoint's 3D position in Neon camera frame (metres)."""
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
