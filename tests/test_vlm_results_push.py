#!/usr/bin/env python3
"""
test_vlm_results_push.py — smoke-test vlm_service.py's subscribe-mode JSON
push without spinning up Neon, FastSAM, Depth Pro, or Gemini.

Constructs a VLMService with stub reader/detector/etc, drives the dispatch
table directly (mirroring serve_forever's flow), and verifies:

    1. cmd=subscribe registers a subscriber, returns subscriber_id + hz.
    2. cmd=unsubscribe with that id removes it.
    3. _build_vlm_results_payload emits the schema documented in
       Render_Layer_Refactor.md §3 and is JSON-serialisable.
    4. _tick_send_results sends a UDP datagram to the subscribed addr at the
       requested hz and not faster (tick rate cap is honoured).
    5. Subscribers past their TTL are pruned without an explicit unsubscribe.

Designed to run in any env with numpy installed — no harmony_vlm needed —
since VLMService.__init__ doesn't touch the model handles, only stores them.
"""

from __future__ import annotations

import json
import socket
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vlm_service import VLMService  # noqa: E402


def _stub_args() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        host="127.0.0.1", port=0, model="stub", verbose=False,
        frame_source="local",
    )


class _StubBundle:
    def __init__(self) -> None:
        self.video = types.SimpleNamespace(
            bgr=np.zeros((10, 10, 3), dtype=np.uint8),
            timestamp_ns=1_700_000_000_000_000_000,
            frame_idx=42,
        )
        self.gaze = types.SimpleNamespace(x=512.0, y=384.0, worn=True)
        self.worn = True
        self.imu = None


class _StubFix:
    active = True
    is_stable = True
    duration_ns = 250_000_000


class _StubFixDet:
    def update(self, _gaze):
        return _StubFix()


class _StubDetection:
    label = "cup"
    confidence = 0.91
    box_xyxy = (10.0, 20.0, 100.0, 200.0)
    box_center = (55.0, 110.0)
    mask_polygon = np.array([[10, 20], [100, 20], [100, 200], [10, 200]], dtype=np.float32)


class _StubReader:
    camera_matrix = np.eye(3, dtype=np.float64)
    distortion_coeffs = None

    def __iter__(self):
        return iter(())

    def close(self):
        pass


def _make_service() -> VLMService:
    svc = VLMService(
        _stub_args(),
        reader=_StubReader(),
        detector=types.SimpleNamespace(detect=lambda _img: []),
        depth_estimator=None,
        reasoner=types.SimpleNamespace(shutdown=lambda: None),
        fix_det=_StubFixDet(),
        fixation_state_cls=lambda **kw: types.SimpleNamespace(**kw),
    )
    # Hand-populate the cached state the way the running service would.
    svc._latest_bundle = _StubBundle()
    svc._latest_bundle_t = time.time()
    svc._latest_fix = _StubFix()
    svc._cache_dets([_StubDetection()],
                    hit_det=_StubDetection(),
                    hit_wp={"pixel_center": [55, 110]})
    svc._set_vlm_state("DECIDED",
                       decision={"text": "pick the cup", "elapsed_s": 1.2,
                                 "depth_at_gaze_m": 0.42,
                                 "waypoints": ["should-be-stripped"]})
    return svc


def test_subscribe_unsubscribe_lifecycle() -> None:
    svc = _make_service()
    addr = ("127.0.0.1", 65000)

    # 1. subscribe → get id + hz
    resp = svc._dispatch("subscribe", {"hz": 50.0}, addr)
    assert resp.get("ok") is True, "subscribe ok=True expected"
    sid = resp["subscriber_id"]
    assert isinstance(sid, str) and len(sid) >= 8, "subscriber_id missing/short"
    assert resp["hz"] == VLMService._RESULTS_TICK_HZ, \
        f"hz should clamp to tick rate, got {resp['hz']}"

    # Idempotent re-subscribe from same addr returns same id.
    resp2 = svc._dispatch("subscribe", {"hz": 5.0}, addr)
    assert resp2["subscriber_id"] == sid, "subscribe should be idempotent on (addr,port)"

    # 2. unsubscribe removes it.
    resp3 = svc._dispatch("unsubscribe", {"subscriber_id": sid}, addr)
    assert resp3.get("ok") is True and resp3.get("removed") is True, \
        "unsubscribe should report removed=True"
    resp4 = svc._dispatch("unsubscribe", {"subscriber_id": sid}, addr)
    assert resp4.get("removed") is False, "second unsubscribe should report removed=False"


def test_build_vlm_results_payload_schema() -> None:
    svc = _make_service()

    # 3. payload schema sanity.
    payload = svc._build_vlm_results_payload()
    json.dumps(payload)  # must be JSON-serialisable end-to-end
    assert payload["type"] == "vlm_results", "type must be 'vlm_results'"
    assert payload["frame_idx"] == 42, "frame_idx mismatched"
    assert payload["frame_ts_ns"] == 1_700_000_000_000_000_000, "frame_ts_ns mismatched"
    assert payload["vlm_state"] == "DECIDED", "vlm_state mismatched"
    assert isinstance(payload["detections"], list) and len(payload["detections"]) == 1, \
        "expected one detection in payload"
    det = payload["detections"][0]
    assert det["label"] == "cup" and "mask_polygon" in det, "detection schema wrong"
    assert payload["fixation"] is not None and payload["fixation"]["active"] is True, \
        "fixation should be active"
    assert payload["depth_at_gaze_m"] == 0.42, "depth_at_gaze_m not threaded through"
    assert payload["decision"] == {"text": "pick the cup", "elapsed_s": 1.2}, \
        "decision should be trimmed to the renderer-relevant fields"
    assert "waypoints" not in (payload.get("decision") or {}), \
        "decision should not carry waypoints (large + unused by renderer)"
    assert payload["gaze_px"] == [512.0, 384.0], "gaze_px mismatched"


def test_tick_send_results_emits_datagram_on_wire() -> None:
    svc = _make_service()

    # 4. tick path: subscribe a real socket, run one tick, expect the datagram.
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx_port = rx.getsockname()[1]
    rx.settimeout(1.0)
    svc._dispatch("subscribe", {"hz": 50.0}, ("127.0.0.1", rx_port))

    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        svc._tick_send_results(tx, time.monotonic())
        data, _from = rx.recvfrom(65535)
        decoded = json.loads(data.decode("utf-8"))
        assert decoded["type"] == "vlm_results", "wire payload missing type"
        assert decoded["frame_idx"] == 42, "wire payload frame_idx mismatched"
    finally:
        tx.close()
        rx.close()


def test_expired_subscriber_pruned() -> None:
    svc = _make_service()

    # A live subscriber (default TTL) must survive the prune that drops the
    # expired one — the original test relied on a leftover live subscriber
    # from the tick section; register one explicitly here so the surviving
    # count assertion (n_left == 1) checks the same thing in isolation.
    live_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    live_rx.bind(("127.0.0.1", 0))
    live_addr = ("127.0.0.1", live_rx.getsockname()[1])
    svc._dispatch("subscribe", {"hz": 50.0}, live_addr)

    # 5. TTL prune: subscribe with a 0 TTL, immediately tick → subscriber gone.
    rx2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx2.bind(("127.0.0.1", 0))
    addr2 = ("127.0.0.1", rx2.getsockname()[1])
    svc._dispatch("subscribe", {"hz": 50.0, "ttl_s": 0.0}, addr2)
    time.sleep(0.01)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        svc._tick_send_results(tx, time.monotonic())
    finally:
        tx.close()
    rx2.settimeout(0.1)
    try:
        rx2.recvfrom(65535)
        assert False, "expired subscriber should not have received a datagram"
    except socket.timeout:
        pass
    finally:
        rx2.close()
        live_rx.close()
    with svc._subscribers_lock:
        n_left = len(svc._subscribers)
    assert n_left == 1, f"expired subscriber should be pruned; got {n_left}"
