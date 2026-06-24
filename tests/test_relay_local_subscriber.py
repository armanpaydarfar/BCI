#!/usr/bin/env python3
"""
test_relay_local_subscriber.py — verify FrameRelayServer.add_local_subscriber
fans bundles out in-process without needing a Neon device.

Constructs a FrameRelayServer (stays unbound — never calls serve_forever),
registers two callbacks via add_local_subscriber, drives _dispatch_local
with stub bundles, and verifies:

    1. Every registered callback receives every dispatched bundle.
    2. remove_local_subscriber stops the callback.
    3. A subscriber that raises does not stop the others from receiving.
    4. add_local_subscriber is idempotent (re-registering the same callable
       doesn't double-fire).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Utils.frame_relay import FrameRelayServer  # noqa: E402


def test_dispatch_fans_out_to_all_subscribers_idempotently() -> None:
    srv = FrameRelayServer(bind_host="127.0.0.1", bind_port=0, hz=30.0)

    a_calls: list = []
    b_calls: list = []

    def cb_a(bundle): a_calls.append(bundle)
    def cb_b(bundle): b_calls.append(bundle)

    srv.add_local_subscriber(cb_a)
    srv.add_local_subscriber(cb_b)
    # Idempotent re-add should not double-fire.
    srv.add_local_subscriber(cb_a)

    srv._dispatch_local("frame-1")
    srv._dispatch_local("frame-2")

    assert a_calls == ["frame-1", "frame-2"], f"cb_a wrong: {a_calls}"
    assert b_calls == ["frame-1", "frame-2"], f"cb_b wrong: {b_calls}"


def test_remove_local_subscriber_stops_callback() -> None:
    srv = FrameRelayServer(bind_host="127.0.0.1", bind_port=0, hz=30.0)

    a_calls: list = []
    b_calls: list = []

    def cb_a(bundle): a_calls.append(bundle)
    def cb_b(bundle): b_calls.append(bundle)

    srv.add_local_subscriber(cb_a)
    srv.add_local_subscriber(cb_b)
    srv._dispatch_local("frame-1")
    srv._dispatch_local("frame-2")

    # Removing cb_a stops it; cb_b keeps going.
    srv.remove_local_subscriber(cb_a)
    srv._dispatch_local("frame-3")
    assert a_calls == ["frame-1", "frame-2"], "cb_a should not fire after removal"
    assert b_calls == ["frame-1", "frame-2", "frame-3"], "cb_b missed a frame after removal"


def test_raising_subscriber_does_not_block_others() -> None:
    srv = FrameRelayServer(bind_host="127.0.0.1", bind_port=0, hz=30.0)

    b_calls: list = []

    def cb_b(bundle): b_calls.append(bundle)

    srv.add_local_subscriber(cb_b)

    # A raising subscriber must not block others.
    def cb_raise(_b): raise RuntimeError("boom")
    srv.add_local_subscriber(cb_raise)
    srv._dispatch_local("frame-4")
    assert b_calls[-1] == "frame-4", "raising callback should not stop other subscribers"
