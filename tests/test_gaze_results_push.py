#!/usr/bin/env python3
"""
test_gaze_results_push.py — smoke-test gaze_runner.py's subscribe-mode JSON
push without spinning up YOLO + Neon.

Constructs a GazeUDPServer with a stub GazeSystem, drives the dispatch
table directly, and verifies:

    1. cmd=subscribe registers a subscriber and returns subscriber_id + hz.
    2. cmd=unsubscribe removes it; second unsubscribe returns removed=False.
    3. _build_gaze_results_payload emits the schema documented in
       Render_Layer_Refactor.md §3 and is JSON-serialisable.
    4. _tick_send sends a UDP datagram to the subscribed addr.
    5. Subscribers past their TTL are pruned.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gaze_runner import GazeUDPServer  # noqa: E402


class _StubGazeSystem:
    def get_snapshot(self, *, include_objects: bool = True, include_frame: bool = True):
        return {
            "ok": True,
            "wall_t": 1700.0,
            "worn": True,
            "gaze_px": (640.0, 480.0),
            "objects": [
                {"track_id": 7, "name": "cup", "conf": 0.88,
                 "xyxy": [10, 20, 100, 200], "age": 14, "lost": 0},
            ],
            "gaze_hit": {"track_id": 7, "name": "cup", "conf": 0.88,
                         "xyxy": [10, 20, 100, 200], "mode": "inside"},
            "gov_enabled": True,
            "gov_reason": "healthy",
            "gov_cd_left": 0.0,
            "loop_hz": 20.0,
            "video_hz": 30.0,
            "det_hz": 3.0,
        }


def _make_server() -> GazeUDPServer:
    # We never call udp.start() — that would bind the recv socket. Instead
    # exercise handle() directly.
    return GazeUDPServer(
        host="127.0.0.1", port=0,
        system=_StubGazeSystem(),
        stop_event=threading.Event(),
    )


def test_subscribe_unsubscribe_lifecycle() -> None:
    udp = _make_server()
    addr = ("127.0.0.1", 65000)

    # 1. subscribe.
    resp = udp.handle({"cmd": "subscribe", "hz": 50.0}, addr)
    assert resp.get("ok") is True, "subscribe ok=True expected"
    sid = resp["subscriber_id"]
    assert isinstance(sid, str) and len(sid) >= 8, "bad subscriber_id"
    assert resp["hz"] == GazeUDPServer._TICK_HZ, \
        f"hz should clamp to tick rate, got {resp['hz']}"

    # Idempotent re-subscribe → same id.
    resp2 = udp.handle({"cmd": "subscribe", "hz": 5.0}, addr)
    assert resp2["subscriber_id"] == sid, "subscribe should be idempotent on (addr,port)"

    # 2. unsubscribe.
    resp3 = udp.handle({"cmd": "unsubscribe", "subscriber_id": sid}, addr)
    assert resp3.get("removed") is True, "expected removed=True"
    resp4 = udp.handle({"cmd": "unsubscribe", "subscriber_id": sid}, addr)
    assert resp4.get("removed") is False, "expected removed=False"


def test_build_gaze_results_payload_schema() -> None:
    udp = _make_server()

    # 3. payload schema.
    payload = udp._build_gaze_results_payload()
    json.dumps(payload)  # must be JSON-serialisable end-to-end
    assert payload["type"] == "gaze_results", "type should be gaze_results"
    assert payload["worn"] is True, "worn missing"
    assert payload["gaze_px"] == [640.0, 480.0], "gaze_px mismatched"
    assert len(payload["tracks"]) == 1 and payload["tracks"][0]["id"] == 7, \
        "tracks shape wrong"
    assert payload["current_hit"]["track_id"] == 7, "current_hit wrong"
    assert payload["governor"]["cv_enabled"] is True, "governor.cv_enabled wrong"


def test_tick_send_emits_datagram_on_wire() -> None:
    udp = _make_server()

    # 4. tick → datagram on the wire.
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx_port = rx.getsockname()[1]
    udp.handle({"cmd": "subscribe", "hz": 50.0}, ("127.0.0.1", rx_port))
    udp._push_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        udp._tick_send(time.monotonic())
        rx.settimeout(1.0)
        data, _from = rx.recvfrom(65535)
        decoded = json.loads(data.decode("utf-8"))
        assert decoded["type"] == "gaze_results", "wire payload type wrong"
        assert decoded["tracks"][0]["label"] == "cup", "wire payload track label wrong"
    finally:
        udp._push_sock.close()
        rx.close()


def test_expired_subscriber_pruned() -> None:
    udp = _make_server()

    # 5. TTL prune.
    rx2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx2.bind(("127.0.0.1", 0))
    addr2 = ("127.0.0.1", rx2.getsockname()[1])
    udp.handle({"cmd": "subscribe", "hz": 50.0, "ttl_s": 0.0}, addr2)
    udp._push_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # ttl_s=0.0 => expires_at == subscribe time. Drive the prune with a clock
        # value unambiguously past it (now + 1 s) rather than a real sleep:
        # time.monotonic() resolution on Windows is ~15.6 ms, so a 10 ms sleep
        # often leaves now == expires_at and the strict '<' prune in _tick_send
        # does not fire (~34% flake). Injecting the clock makes it deterministic.
        udp._tick_send(time.monotonic() + 1.0)
    finally:
        udp._push_sock.close()
    rx2.settimeout(0.1)
    try:
        rx2.recvfrom(65535)
        assert False, "expired subscriber should not receive"
    except socket.timeout:
        pass
    finally:
        rx2.close()
