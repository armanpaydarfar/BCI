"""
test_vlm_client.py — wire-contract characterization for
``Utils.perception_clients.VLMClient``.

VLMClient is the canonical UDP request-reply client for ``vlm_service.py``.
It had ZERO functional tests, yet it is about to become the *sole* VLM client
(``vlm_bridge.VLMBridge`` is being folded into it). This file pins, per method:

1. the exact ``{"cmd": ..., <params>}`` JSON it puts on the wire, and
2. that it returns the service's decoded reply dict verbatim,

so the consolidation — and any later refactor of the client — cannot silently
drift the protocol the Windows-hosted service depends on. Loopback UDP only;
no vlm_service, no Neon, no models.

Wire payloads pinned here are read from ``Utils/perception_clients.py:150-232``.
"""

from __future__ import annotations

import json
import socket
import threading
import types
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from Utils.perception_clients import VLMClient  # noqa: E402


class _MockService:
    """Loopback UDP server: records each request's parsed JSON and echoes a
    canned reply so the client's send *and* receive paths are both exercised.
    One daemon thread; one round-trip per client call."""

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        self.requests: list[dict] = []
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        self.sock.settimeout(0.2)
        while not self._stop:
            try:
                data, addr = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            req = json.loads(data.decode("utf-8"))
            self.requests.append(req)
            # Echo the request back inside the reply so the test can assert
            # the client returns exactly what the service sent.
            self.sock.sendto(
                json.dumps({"ok": True, "cmd_seen": req.get("cmd"), "echo": req}).encode("utf-8"),
                addr,
            )

    def close(self) -> None:
        self._stop = True
        self.sock.close()


@pytest.fixture()
def mock_service():
    svc = _MockService()
    try:
        yield svc
    finally:
        svc.close()


def _client(svc: _MockService) -> VLMClient:
    cfg = types.SimpleNamespace(
        VLM_SERVICE_HOST="127.0.0.1",
        VLM_SERVICE_PORT=svc.port,
        VLM_SERVICE_TIMEOUT=1.0,
    )
    return VLMClient(cfg)


# ── no-argument commands: pin the bare {"cmd": ...} wire payload ──────────────

@pytest.mark.parametrize("method, expected_cmd", [
    ("status", "status"),
    ("snapshot", "snapshot"),
    ("camera_matrix", "camera_matrix"),
    ("stop", "stop"),
    ("capture_first", "capture_first"),
])
def test_bare_command_wire_payload(mock_service, method, expected_cmd):
    client = _client(mock_service)
    reply = getattr(client, method)()
    assert mock_service.requests[-1] == {"cmd": expected_cmd}
    # Reply is returned decoded, verbatim.
    assert reply["ok"] is True
    assert reply["cmd_seen"] == expected_cmd


# ── parameterised commands: pin cmd + every param it serialises ──────────────

def test_segment_wire_payload(mock_service):
    client = _client(mock_service)
    client.segment(include_masks=True)
    assert mock_service.requests[-1] == {"cmd": "segment", "include_masks": True}
    client.segment()  # default
    assert mock_service.requests[-1] == {"cmd": "segment", "include_masks": False}


def test_segment_stream_wire_payload(mock_service):
    client = _client(mock_service)
    client.segment_stream(enabled=True, hz=12.5)
    assert mock_service.requests[-1] == {"cmd": "segment_stream", "enabled": True, "hz": 12.5}


def test_depth_wire_payload(mock_service):
    client = _client(mock_service)
    client.depth(at_gaze=False, save=True)
    assert mock_service.requests[-1] == {"cmd": "depth", "at_gaze": False, "save": True}
    client.depth()  # defaults: at_gaze=True, save=False
    assert mock_service.requests[-1] == {"cmd": "depth", "at_gaze": True, "save": False}


def test_decide_wire_payload(mock_service):
    client = _client(mock_service)
    client.decide(vlm_timeout_s=25.0)
    # vlm_timeout_s is serialised on the wire as "timeout".
    assert mock_service.requests[-1] == {"cmd": "decide", "timeout": 25.0}


def test_decide_pair_wire_payload(mock_service):
    client = _client(mock_service)
    client.decide_pair(snapshot_id="snap-7", vlm_timeout_s=40.0)
    assert mock_service.requests[-1] == {
        "cmd": "decide_pair", "snapshot_id": "snap-7", "timeout": 40.0,
    }


def test_subscribe_wire_payload(mock_service):
    client = _client(mock_service)
    client.subscribe(hz=15.0, ttl_s=20.0)
    assert mock_service.requests[-1] == {
        "cmd": "subscribe", "stream": "results", "hz": 15.0, "ttl_s": 20.0,
    }


def test_unsubscribe_wire_payload(mock_service):
    client = _client(mock_service)
    client.unsubscribe(subscriber_id="abc123")
    assert mock_service.requests[-1] == {"cmd": "unsubscribe", "subscriber_id": "abc123"}


# ── failure semantics: VLMClient RAISES on transport failure (no catch) ──────

def test_transport_failure_raises():
    """VLMClient's udp_request does not swallow errors — a dead endpoint must
    surface (OSError), NOT return None. The gaze-driver consolidation relies on
    this being the contract it has to wrap to preserve None-on-failure."""
    # Bind+close a socket to grab a port nothing is listening on.
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    dead_port = s.getsockname()[1]
    s.close()
    cfg = types.SimpleNamespace(
        VLM_SERVICE_HOST="127.0.0.1", VLM_SERVICE_PORT=dead_port, VLM_SERVICE_TIMEOUT=0.2,
    )
    client = VLMClient(cfg)
    with pytest.raises(OSError):
        client.status()
