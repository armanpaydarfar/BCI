"""
test_vlm_subscriber_guard.py — WS4 F2 transport guard in vlm_subscriber.py.

Exercises the ordering-drop decision (_accept_seq) and the staleness query
without opening a socket or a Qt event loop. Guards the keep-working
invariant: unstamped streams degrade to pass-through, and one-shot control
payloads are never subject to the result-stream ordering filter.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from Utils.vlm_subscriber import JsonPushSubscriber  # noqa: E402


def _sub():
    return JsonPushSubscriber("127.0.0.1", 5589)


def test_in_order_seqs_all_accepted():
    s = _sub()
    assert s._accept_seq(100)
    assert s._accept_seq(101)
    assert s._accept_seq(200)
    assert s.dropped_out_of_order() == 0


def test_out_of_order_dropped():
    s = _sub()
    assert s._accept_seq(500)
    assert not s._accept_seq(400)   # reordered, older → drop
    assert not s._accept_seq(499)   # still older than watermark → drop
    assert s._accept_seq(501)       # advances again
    assert s.dropped_out_of_order() == 2


def test_zero_seq_is_passthrough_and_does_not_advance_watermark():
    s = _sub()
    assert s._accept_seq(0)         # unstamped → accept
    assert s._accept_seq(0)         # still accept
    assert s._accept_seq(10)        # a stamped one sets the watermark
    assert not s._accept_seq(5)     # now older is dropped
    # An unstamped datagram after a watermark is still accepted (no regression
    # for streams that don't stamp ts_send_ns).
    assert s._accept_seq(0)


def test_staleness_query():
    s = _sub()
    # Before any stream payload: infinitely stale-age but is_stale() is False
    # (nothing has flowed yet — don't false-alarm on a fresh subscriber).
    assert s.seconds_since_stream() == float("inf")
    assert s.is_stale() is False
    # Simulate a stream payload that arrived 5 s ago.
    import time
    s._stream_seen = True
    s._last_stream_t = time.monotonic() - 5.0
    assert s.seconds_since_stream() > s.STALE_AFTER_S
    assert s.is_stale() is True
    # A fresh payload clears staleness.
    s._last_stream_t = time.monotonic()
    assert s.is_stale() is False
