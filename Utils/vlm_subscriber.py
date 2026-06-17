"""
vlm_subscriber.py — UDP subscribe-mode listener for the GPU host's
vlm_service.py (and gaze_runner.py) push streams.

This is the third process in the perception pipeline:
  1. ``Utils/frame_relay.py``       — Send  (Linux outbound, raw frames)
  2. ``vlm_service.py``             — Compute (Windows GPU-side ML)
  3. ``Utils/vlm_subscriber.py``    — Receive (Linux inbound, results)

Wire protocol (matches ``vlm_service.py``'s subscribe handler):

  Subscribe handshake — client sends a single UDP datagram::

      {"cmd": "subscribe", "stream": "results", "hz": <float>, "ttl_s": <float>}

  Service replies on the same socket::

      {"ok": true, "subscriber_id": "<uuid>"}    # success
      {"ok": false, "error": "..."}              # rejection

  After success, the service pushes one JSON datagram per result frame to
  the client's source address. Each push payload carries a ``type`` field
  (e.g. ``"vlm_results"``, ``"gaze_results"``) so subscribe replies and
  push payloads can be distinguished on the shared socket.

  Heartbeat — client re-sends ``cmd=subscribe`` every ``HEARTBEAT_S``
  seconds. The service treats a re-subscribe from the same (addr, port)
  as a refresh, returning the same ``subscriber_id`` so server-side state
  is idempotent. ``ttl_s`` lets the service prune subscribers that go
  silent, so the client side doesn't need its own dead-peer detection.

  Unsubscribe (best-effort, on stop)::

      {"cmd": "unsubscribe", "subscriber_id": "<uuid>"}

The class lives here (rather than baked into the panel widget) for two
reasons: (a) it mirrors the script-per-pipeline-stage structure used by
``frame_relay.py`` and ``vlm_service.py``, so a future engineer reading
``ls Utils/`` sees the architecture immediately; (b) it can be invoked
standalone for diagnostics — see the ``__main__`` block below — to verify
the GPU host is actually pushing without involving Qt or the panel.

Standalone usage::

    python -m Utils.vlm_subscriber <host> <port>

Prints each received payload as a one-liner JSON dump to stdout. Useful
when the panel's "Receive" LED is gray and you want to know whether the
problem is the wire or the panel-side rendering.
"""

from __future__ import annotations

import json
import socket
import time
from typing import Any, Dict, Optional

from PySide6.QtCore import QThread, Signal


class JsonPushSubscriber(QThread):
    """Subscribe-mode UDP listener. Sends ``cmd=subscribe`` to the configured
    service, then rx-loops on the same socket for pushed JSON datagrams.

    Emits one Signal per received payload. Re-subscribes on a heartbeat
    timer so the service-side TTL can prune dead clients without the
    widget having to track its own watchdog.
    """

    payload_received = Signal(dict)
    # State machine: "subscribed" (handshake replied ok) → "receiving:<type>"
    # (first push payload actually arrived; <type> is the payload's
    # ``type`` field — e.g. ``vlm_results`` or ``chain_verify``) →
    # "unsubscribed" (stop path) | "error: …". Receivers that want a
    # "real data is flowing" signal should treat any state matching
    # ``startswith("receiving")`` as the green-light state. The type
    # suffix is informational; it lets the panel log which kind of
    # push triggered the chain-of-causation transition.
    state_changed = Signal(str)

    HEARTBEAT_S = 10.0  # well below vlm_service's 30 s TTL.

    # WS4 F2 transport guard. Continuous result streams whose datagrams carry a
    # monotonic ts_send_ns; an out-of-order (network-reordered) datagram on
    # these is dropped so it can't overwrite fresher panel state. One-shot
    # control payloads (e.g. chain_verify) are exempt — they must always be
    # delivered for the verification handshake.
    _STREAM_TYPES = ("vlm_results", "gaze_results")
    # Age (s) past which the stream is reported stale via is_stale(). The panel
    # surfaces this read-only; it does NOT drive the Receive LED state machine
    # (a transient stall must not permanently degrade a working Connect).
    STALE_AFTER_S = 2.0

    def __init__(self, host: str, port: int, *, hz: float = 20.0,
                 ttl_s: float = 30.0, parent=None) -> None:
        super().__init__(parent)
        self._host = str(host)
        self._port = int(port)
        self._hz = float(hz)
        self._ttl_s = float(ttl_s)
        self._sock: Optional[socket.socket] = None
        self._running = False
        self._subscriber_id: Optional[str] = None
        self._last_subscribe_t: float = 0.0
        self._first_payload_seen: bool = False
        # Transport-guard state (written on the rx thread, read by the panel).
        self._last_seq_ns: int = 0           # newest ts_send_ns forwarded
        self._last_stream_t: float = 0.0     # monotonic of last stream payload
        self._stream_seen: bool = False
        self._dropped_out_of_order: int = 0

    def stop(self) -> None:
        self._running = False
        # Unblock the recvfrom by closing the socket; the run loop
        # tolerates the resulting OSError and exits.
        try:
            if self._sock is not None:
                self._sock.close()
        except OSError:
            pass

    # ── thread body ───────────────────────────────────────────────────────

    def run(self) -> None:
        self._running = True
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.bind(("", 0))  # ephemeral port; service replies here.
            self._sock.settimeout(0.5)
        except OSError as e:
            self.state_changed.emit(f"error: bind: {e}")
            return

        if not self._subscribe():
            self.state_changed.emit("error: initial subscribe failed")
        else:
            self.state_changed.emit("subscribed")

        while self._running:
            self._maybe_heartbeat()
            try:
                data, _addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                continue
            try:
                payload = json.loads(data.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            # Subscribe replies and push payloads share this socket. Only
            # forward push payloads (they carry a `type` field; the
            # subscribe reply has `ok`/`subscriber_id`).
            if "type" not in payload:
                continue
            ptype = str(payload.get("type", "?"))
            # WS4 F2 ordering guard: drop a result-stream datagram whose
            # ts_send_ns predates the newest one already forwarded (UDP can
            # deliver reordered, and a stale datagram overwriting fresher
            # detections is exactly what makes the overlay look wrong). Exempt
            # one-shot control payloads (chain_verify) so verification never
            # loses a datagram to this filter.
            if ptype in self._STREAM_TYPES:
                seq = int(payload.get("ts_send_ns", 0) or 0)
                if not self._accept_seq(seq):
                    continue
                self._last_stream_t = time.monotonic()
                self._stream_seen = True
            self.payload_received.emit(payload)
            # First-payload state transition. Receivers gate the
            # "real data flowing" indicator on this rather than on
            # the subscribe handshake (which only proves the
            # service answered our subscribe — not that it's
            # producing output).
            if not self._first_payload_seen:
                self._first_payload_seen = True
                self.state_changed.emit(f"receiving:{ptype}")

        # Best-effort unsubscribe so the service prunes immediately.
        if self._subscriber_id:
            try:
                self._send({"cmd": "unsubscribe",
                            "subscriber_id": self._subscriber_id})
            except OSError:
                pass
        try:
            if self._sock is not None:
                self._sock.close()
        except OSError:
            pass
        self.state_changed.emit("unsubscribed")

    # ── transport guard ────────────────────────────────────────────────────

    def _accept_seq(self, seq: int) -> bool:
        """Decide whether a result-stream datagram with ts_send_ns ``seq`` is
        fresh enough to forward. Drops (returns False) when it predates the
        newest one already forwarded — UDP reordering / a late stale datagram.
        ``seq == 0`` (no stamp) is always accepted and never advances the
        watermark, so an unstamped stream degrades to pass-through. Advances the
        watermark and returns True on accept.
        """
        if seq and seq < self._last_seq_ns:
            self._dropped_out_of_order += 1
            return False
        if seq:
            self._last_seq_ns = seq
        return True

    # ── transport-guard introspection (read-only, panel-facing) ────────────

    def seconds_since_stream(self) -> float:
        """Monotonic age (s) of the last in-order result-stream datagram, or
        ``inf`` before the first one. Used by the panel to show a 'stale' hint
        without touching the Receive LED state machine."""
        if not self._stream_seen:
            return float("inf")
        return time.monotonic() - self._last_stream_t

    def is_stale(self) -> bool:
        """True when the result stream has gone quiet for > STALE_AFTER_S after
        having flowed at least once."""
        return self._stream_seen and self.seconds_since_stream() > self.STALE_AFTER_S

    def dropped_out_of_order(self) -> int:
        """Count of result-stream datagrams dropped as reordered/stale."""
        return self._dropped_out_of_order

    # ── helpers ───────────────────────────────────────────────────────────

    def _send(self, payload: Dict[str, Any]) -> None:
        if self._sock is None:
            return
        self._sock.sendto(
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            (self._host, self._port),
        )

    def _subscribe(self) -> bool:
        try:
            self._send({"cmd": "subscribe", "stream": "results",
                        "hz": self._hz, "ttl_s": self._ttl_s})
            assert self._sock is not None
            self._sock.settimeout(1.5)
            data, _addr = self._sock.recvfrom(65535)
            self._sock.settimeout(0.5)
        except (OSError, AssertionError):
            return False
        try:
            resp = json.loads(data.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return False
        if not resp.get("ok"):
            return False
        self._subscriber_id = resp.get("subscriber_id")
        self._last_subscribe_t = time.monotonic()
        return True

    def _maybe_heartbeat(self) -> None:
        if time.monotonic() - self._last_subscribe_t < self.HEARTBEAT_S:
            return
        self._subscribe()  # idempotent on (addr, port) — re-uses same id


def _main() -> int:
    """Standalone diagnostic runner. Subscribes to <host> <port>, prints
    each received payload as a one-line JSON dump, exits cleanly on
    Ctrl-C. Headless — uses QCoreApplication, no GUI dependency.
    """
    import argparse
    import signal as _signal
    import sys

    from PySide6.QtCore import QCoreApplication

    p = argparse.ArgumentParser(description="Subscribe to a vlm_service / "
                                            "gaze_runner push stream and "
                                            "print payloads to stdout.")
    p.add_argument("host", help="Service host (e.g. 100.118.53.46)")
    p.add_argument("port", type=int, help="Service UDP port (e.g. 5589)")
    p.add_argument("--hz", type=float, default=20.0,
                   help="Requested push rate cap (default: 20)")
    p.add_argument("--ttl", type=float, default=30.0,
                   help="Subscription TTL in seconds (default: 30)")
    args = p.parse_args()

    app = QCoreApplication(sys.argv)
    sub = JsonPushSubscriber(args.host, args.port, hz=args.hz, ttl_s=args.ttl)

    def _on_payload(payload: dict) -> None:
        # One-line JSON dump. Stable key order so diffs across runs are
        # readable; no indentation so each push is a single line.
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")),
              flush=True)

    def _on_state(state: str) -> None:
        print(f"[vlm_subscriber] state: {state}", file=sys.stderr, flush=True)

    sub.payload_received.connect(_on_payload)
    sub.state_changed.connect(_on_state)

    # Clean shutdown on Ctrl-C — Python's default SIGINT handler doesn't
    # interrupt the Qt event loop, so we install one that quits the app.
    def _on_sigint(_signum, _frame) -> None:
        sub.stop()
        sub.wait(2000)
        app.quit()

    _signal.signal(_signal.SIGINT, _on_sigint)
    sub.start()
    rc = app.exec()
    sub.stop()
    sub.wait(2000)
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(_main())
