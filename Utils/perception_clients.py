"""
perception_clients.py — Thin client layer between the panel/UI and the
gaze + VLM perception services.

Localises wire format and endpoint resolution so panel UI changes (button
moves, tab reshuffles) don't perturb the wire and so the experiment driver
and the panel can share one source of truth for the UDP payloads.

Endpoints come from ``config.py`` at construction time. Callers do not
hard-code hosts or ports.

Currently implements:

- ``GazeClient``     — UDP 5588 (gaze_runner.py service mode).
- ``VLMClient``      — UDP 5589 (vlm_service.py request-reply) plus a
                       helper for the TCP 5590 overlay handshake. The
                       overlay reader thread itself stays in the panel
                       since it needs Qt threading hooks; the client just
                       owns the host/port resolution.
- ``FrameRelayController`` — UDP-style status helper for the new TCP
                       relay (Utils/frame_relay.py). Phase 1 placeholder;
                       the relay does not yet expose a status query — the
                       client returns whether a TCP connect succeeds.

Each method returns a plain ``dict`` (or ``None`` on transport failure).
No business logic, no caching, no retries beyond what the legacy panel
code does — this is intentionally thin.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any, Dict, Optional


def udp_request(host: str, port: int, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    """One-shot JSON-over-UDP request. Public so non-panel callers (experiment
    driver, latency probe) can reuse the wire idiom without rebuilding sockets.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.settimeout(float(timeout_s))
        sock.sendto(json.dumps(payload, separators=(",", ":")).encode("utf-8"), (host, int(port)))
        data, _ = sock.recvfrom(65535)
    finally:
        sock.close()
    return json.loads(data.decode("utf-8", errors="replace"))


def udp_request_using(sock: socket.socket, host: str, port: int,
                      payload: Dict[str, Any], timeout_s: float) -> Optional[Dict[str, Any]]:
    """Same wire format as :func:`udp_request`, but reuses a caller-owned
    socket so realtime loops don't allocate per call. Returns the decoded
    response dict, or ``None`` on transport failure (the realtime loop
    should keep running rather than crash on a transient drop).

    Used by ``ExperimentDriver_Online_GazeTracking.gaze_udp_request`` so the
    panel and the experiment loop share one source of truth for the
    request wire format while the driver keeps full control of socket
    lifecycle.
    """
    try:
        sock.settimeout(float(timeout_s))
        sock.sendto(json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                    (host, int(port)))
        data, _ = sock.recvfrom(65535)
        return json.loads(data.decode("utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None


# Backwards-compatible alias — keep the leading-underscore name for any
# accidental imports while internal callers transition to ``udp_request``.
_udp_request = udp_request


# ── gaze service client ────────────────────────────────────────────────────


class GazeClient:
    """Wraps the UDP request-reply interface served by gaze_runner.py in
    ``--mode service``. Endpoint comes from ``config.GAZE_UDP_IP`` /
    ``config.GAZE_UDP_PORT``.
    """

    def __init__(self, cfg) -> None:
        self.host = str(getattr(cfg, "GAZE_UDP_IP", "127.0.0.1"))
        self.port = int(getattr(cfg, "GAZE_UDP_PORT", 5588))
        self.default_timeout_s = float(getattr(cfg, "GAZE_UDP_TIMEOUT", 0.8) or 0.8)

    def status(self, *, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        return _udp_request(self.host, self.port, {"cmd": "status"}, timeout_s or self.default_timeout_s)

    def snapshot(self, *, include_objects: bool = True, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        return _udp_request(
            self.host, self.port,
            {"cmd": "snapshot", "include_objects": bool(include_objects)},
            timeout_s or self.default_timeout_s,
        )

    def recenter(self, *, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        return _udp_request(self.host, self.port, {"cmd": "recenter"}, timeout_s or self.default_timeout_s)

    def set_cv_enabled(self, enabled: bool, *, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        return _udp_request(
            self.host, self.port,
            {"cmd": "set_cv", "enabled": bool(enabled)},
            timeout_s or self.default_timeout_s,
        )

    def stop(self, *, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        return _udp_request(self.host, self.port, {"cmd": "stop"}, timeout_s or self.default_timeout_s)


# ── VLM service client ─────────────────────────────────────────────────────


class VLMClient:
    """Wraps the UDP request-reply interface served by vlm_service.py.
    Endpoint comes from ``config.VLM_SERVICE_HOST`` /
    ``config.VLM_SERVICE_PORT``. Overlay TCP host/port are exposed for the
    panel's overlay reader thread but the actual TCP read loop stays in
    the panel (it needs Qt threading hooks)."""

    def __init__(self, cfg) -> None:
        self.host = str(getattr(cfg, "VLM_SERVICE_HOST", "127.0.0.1"))
        self.port = int(getattr(cfg, "VLM_SERVICE_PORT", 5589))
        self.default_timeout_s = float(getattr(cfg, "VLM_SERVICE_TIMEOUT", 2.0) or 2.0)
        self.overlay_host = self.host
        self.overlay_port = int(getattr(cfg, "VLM_OVERLAY_PORT", 5590))

    def status(self, *, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        return _udp_request(self.host, self.port, {"cmd": "status"}, timeout_s or self.default_timeout_s)

    def snapshot(self, *, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        return _udp_request(self.host, self.port, {"cmd": "snapshot"}, timeout_s or self.default_timeout_s)

    def segment(self, *, include_masks: bool = False, timeout_s: float = 5.0) -> Dict[str, Any]:
        return _udp_request(
            self.host, self.port,
            {"cmd": "segment", "include_masks": bool(include_masks)},
            timeout_s,
        )

    def segment_stream(self, *, enabled: bool, hz: float = 10.0,
                       timeout_s: Optional[float] = None) -> Dict[str, Any]:
        return _udp_request(
            self.host, self.port,
            {"cmd": "segment_stream", "enabled": bool(enabled), "hz": float(hz)},
            timeout_s or self.default_timeout_s,
        )

    def depth(self, *, at_gaze: bool = True, save: bool = False,
              timeout_s: float = 15.0) -> Dict[str, Any]:
        return _udp_request(
            self.host, self.port,
            {"cmd": "depth", "at_gaze": bool(at_gaze), "save": bool(save)},
            timeout_s,
        )

    def decide(self, *, vlm_timeout_s: float = 30.0,
               sock_timeout_s: float = 40.0) -> Dict[str, Any]:
        return _udp_request(
            self.host, self.port,
            {"cmd": "decide", "timeout": float(vlm_timeout_s)},
            sock_timeout_s,
        )

    def capture_first(self, *, sock_timeout_s: float = 12.0) -> Dict[str, Any]:
        return _udp_request(
            self.host, self.port,
            {"cmd": "capture_first"},
            sock_timeout_s,
        )

    def decide_pair(self, *, snapshot_id: str, vlm_timeout_s: float = 45.0,
                    sock_timeout_s: float = 60.0) -> Dict[str, Any]:
        return _udp_request(
            self.host, self.port,
            {"cmd": "decide_pair", "snapshot_id": str(snapshot_id),
             "timeout": float(vlm_timeout_s)},
            sock_timeout_s,
        )

    def camera_matrix(self, *, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        return _udp_request(self.host, self.port, {"cmd": "camera_matrix"},
                            timeout_s or self.default_timeout_s)

    def stop(self, *, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        return _udp_request(self.host, self.port, {"cmd": "stop"},
                            timeout_s or self.default_timeout_s)


# ── frame relay controller ─────────────────────────────────────────────────


class FrameRelayController:
    """Status helper for the TCP frame relay (Utils/frame_relay.py).

    The relay does not yet expose a status query channel. This client
    provides ``ping()`` which performs a TCP connect-and-disconnect against
    the relay's bind address and reports whether it succeeded — enough for
    the panel to badge "relay reachable / not reachable" without spawning
    a real envelope-consuming reader.

    Endpoint resolution: dial host comes from ``FRAME_RELAY_DIAL_HOST`` if
    set, else falls back to ``FRAME_RELAY_HOST``.
    """

    def __init__(self, cfg) -> None:
        self.host = str(
            getattr(cfg, "FRAME_RELAY_DIAL_HOST", None)
            or getattr(cfg, "FRAME_RELAY_HOST", "127.0.0.1")
        )
        self.port = int(getattr(cfg, "FRAME_RELAY_PORT", 5591))

    def ping(self, *, timeout_s: float = 1.0) -> Dict[str, Any]:
        """TCP connect-test. Returns ``{"ok": bool, "host":..., "port":..., "error": str|None}``."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(float(timeout_s))
        try:
            s.connect((self.host, int(self.port)))
            return {"ok": True, "host": self.host, "port": self.port, "error": None}
        except OSError as e:
            return {"ok": False, "host": self.host, "port": self.port, "error": str(e)}
        finally:
            try:
                s.close()
            except OSError:
                pass

    def read_handshake(self, *, timeout_s: float = 5.0) -> Optional[Dict[str, Any]]:
        """Connect, read one envelope, and return the JSON header if it is
        a handshake. Used by the panel's diagnostic 'Probe Relay' button to
        confirm camera intrinsics arrive."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(float(timeout_s))
        try:
            s.connect((self.host, int(self.port)))
            prefix = _recv_exact(s, 8)
            if prefix is None:
                return None
            json_len, _jpeg_len = struct.unpack(">II", prefix)
            json_buf = _recv_exact(s, json_len)
            if json_buf is None:
                return None
            hdr = json.loads(json_buf.decode("utf-8", errors="replace"))
            if hdr.get("type") != "handshake":
                return None
            return hdr
        except (OSError, json.JSONDecodeError):
            return None
        finally:
            try:
                s.close()
            except OSError:
                pass


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    chunks = []
    remaining = n
    while remaining > 0:
        try:
            chunk = sock.recv(remaining)
        except OSError:
            return None
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
