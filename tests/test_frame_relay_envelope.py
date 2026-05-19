"""
test_frame_relay_envelope.py

Guards the frame-relay wire-format contract (Plan §6 #12). Complements
the existing `tests/test_relay_loopback.py` (promoted from `tools/` in
Phase 1a §5.1.c) — that file covers the happy-path handshake +
FrameBundle round-trip via a real mock server; this file targets the
sad-path branches of the envelope reader directly.

Envelope wire format (Utils/remote_frame_reader.py:29):

    [4-byte JSON length][4-byte JPEG length][JSON header][JPEG bytes]

Both length fields are big-endian uint32. The reader (`_recv_envelope`,
file:502-526) aborts (returns None) when:

  * Either length field exceeds 4 MB (file:510-512).
  * The JSON header is not valid UTF-8 / JSON (file:521-525).

Citations under test (verified 2026-05-18):

  - Utils/remote_frame_reader.py:502-526  `_recv_envelope`
  - Utils/remote_frame_reader.py:529-541  `_recv_exact`
  - Utils/frame_relay.py:192-196          `_send_envelope`
"""

from __future__ import annotations

import json
import socket
import struct
import threading

import pytest

from Utils.frame_relay import _send_envelope
from Utils.remote_frame_reader import _RelayConnection


# ─── helpers ──────────────────────────────────────────────────────────────

def _socketpair_tcp():
    """Bind a tiny TCP listener and return (server_side, client_side)
    sockets connected to each other. Used because `socket.socketpair`
    is SOCK_STREAM (good) but we want explicit endpoints rather than
    a unix-domain pair so we mirror the production transport."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    box = {}

    def accept():
        srv, _ = listener.accept()
        box["srv"] = srv

    t = threading.Thread(target=accept)
    t.start()
    client.connect(("127.0.0.1", port))
    t.join()
    listener.close()
    return box["srv"], client


def _new_relay_connection():
    """`_RelayConnection.__init__` does not start any thread (the manager
    only starts in `.connect()`), so we can instantiate one to call
    `_recv_envelope` directly."""
    return _RelayConnection(host="127.0.0.1", port=0)


# ─── happy-path round-trip ────────────────────────────────────────────────

class TestHappyPath:
    def test_send_envelope_then_recv_envelope_roundtrip(self):
        srv, cli = _socketpair_tcp()
        try:
            header = {"type": "frame", "ts_ns": 12345, "ok": True}
            payload = b"\xff\xd8\xff\xe0fake-jpeg-body\x00"
            _send_envelope(srv, header, payload)

            conn = _new_relay_connection()
            out = conn._recv_envelope(cli)
            assert out is not None
            hdr, jpeg = out
            assert hdr == header
            assert jpeg == payload
        finally:
            srv.close()
            cli.close()


# ─── malformed JSON ──────────────────────────────────────────────────────

class TestMalformedJson:
    def test_invalid_json_returns_none(self):
        """A correct-arity envelope whose JSON header is not parseable
        JSON should be rejected with a None return — not raise into the
        reader loop (file:521-525)."""
        srv, cli = _socketpair_tcp()
        try:
            bad_json = b"\xff\xfe\xfd<<not json>>"
            jpeg = b""
            srv.sendall(struct.pack(">II", len(bad_json), len(jpeg))
                        + bad_json + jpeg)

            conn = _new_relay_connection()
            out = conn._recv_envelope(cli)
            assert out is None
        finally:
            srv.close()
            cli.close()

    def test_truncated_json_returns_none(self):
        """If the sender hangs up mid-header, _recv_exact returns None
        and _recv_envelope propagates that as None."""
        srv, cli = _socketpair_tcp()
        try:
            # Promise 100 bytes of JSON, then close after delivering only
            # the prefix.
            srv.sendall(struct.pack(">II", 100, 0))
            srv.close()
            conn = _new_relay_connection()
            out = conn._recv_envelope(cli)
            assert out is None
        finally:
            cli.close()


# ─── oversize envelope ───────────────────────────────────────────────────

class TestOversizeEnvelope:
    OVERSIZE = 5 * 1024 * 1024  # 5 MB > the 4 MB cap (file:510)

    def test_oversize_json_aborts(self):
        srv, cli = _socketpair_tcp()
        try:
            # Promise 5 MB of JSON in the prefix — reader should bail
            # before reading any of it, returning None.
            srv.sendall(struct.pack(">II", self.OVERSIZE, 0))
            conn = _new_relay_connection()
            out = conn._recv_envelope(cli)
            assert out is None
        finally:
            srv.close()
            cli.close()

    def test_oversize_jpeg_aborts(self):
        srv, cli = _socketpair_tcp()
        try:
            srv.sendall(struct.pack(">II", 10, self.OVERSIZE))
            conn = _new_relay_connection()
            out = conn._recv_envelope(cli)
            assert out is None
        finally:
            srv.close()
            cli.close()

    def test_at_or_below_cap_is_accepted(self):
        """Exactly 4 MB JPEG is still allowed (file:510 uses `>`, not
        `>=`). Locked here so a future tightening of the cap is an
        explicit decision.

        The 4 MB payload exceeds the kernel's TCP send buffer (~200 KB
        on Linux by default), so `sendall` blocks until the receiver
        drains. Push the send onto a thread so the synchronous
        `_recv_envelope` call can interleave with it.
        """
        srv, cli = _socketpair_tcp()
        header = {"type": "frame"}
        hdr_bytes = json.dumps(header).encode("utf-8")
        jpeg = b"\x00" * (4 * 1024 * 1024)
        wire = struct.pack(">II", len(hdr_bytes), len(jpeg)) + hdr_bytes + jpeg

        def send_all():
            try:
                srv.sendall(wire)
            except OSError:
                pass

        t = threading.Thread(target=send_all, daemon=True)
        t.start()
        try:
            conn = _new_relay_connection()
            out = conn._recv_envelope(cli)
            assert out is not None
            assert out[0] == header
            assert len(out[1]) == 4 * 1024 * 1024
        finally:
            t.join(timeout=5.0)
            srv.close()
            cli.close()


# ─── EOF / disconnect ────────────────────────────────────────────────────

class TestPeerDisconnect:
    def test_immediate_eof_returns_none(self):
        srv, cli = _socketpair_tcp()
        srv.close()
        try:
            conn = _new_relay_connection()
            out = conn._recv_envelope(cli)
            assert out is None
        finally:
            cli.close()
