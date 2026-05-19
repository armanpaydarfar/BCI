"""
test_networking_sockets.py

Guards the cross-host UDP / TCP resilience class (Plan §6 #8). Example
commits:

  - 61ad53f, 1830174, 4310c69, 60e7f1f — sequence of fixes to the
    socket-binding contract in Utils/networking.py. After these
    fixes the rules are:

      * SIMULATION_MODE suppresses robot traffic completely.
      * Marker and FES sockets are dedicated; never reuse the robot
        control socket.
      * If a "FES_" message reaches a non-FES destination, the
        function re-routes to the configured UDP_FES endpoint
        (file:742-751).
      * A non-robot / non-marker / non-FES destination uses the
        generic ephemeral socket (file:782-791).

Citations under test (verified 2026-05-18):

  - Utils/networking.py:329-336  `_sendto_robot`
  - Utils/networking.py:338-391  `_await_ack_blocking`
  - Utils/networking.py:682-852  `send_udp_message`
  - Utils/networking.py:64-67    SIMULATION_MODE snapshot

The tests use real UDP loopback sockets (ephemeral ports) rather than
`socket.socketpair()` because `socketpair` returns connected SOCK_STREAM
pairs on Linux — the networking helpers all use `recvfrom` / `sendto`
which only work on UDP.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

import Utils.networking as net


# ─── helpers ──────────────────────────────────────────────────────────────

def _bind_recv_udp():
    """Bind a UDP receiver on 127.0.0.1:<ephemeral>; return (sock, port)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    s.setblocking(False)
    return s, s.getsockname()[1]


def _drain(s, timeout=0.2):
    """Pull a single datagram from `s` with a short timeout. Returns
    (payload_str, src_addr) or (None, None) on timeout."""
    end = time.time() + timeout
    while time.time() < end:
        try:
            data, addr = s.recvfrom(65535)
            return data.decode("utf-8", "ignore"), addr
        except BlockingIOError:
            time.sleep(0.01)
    return None, None


@pytest.fixture(autouse=True)
def _reset_module_sockets(monkeypatch):
    """Force networking helpers to allocate fresh sockets per test rather
    than reuse a process-global. Without this, the control / marker /
    FES / generic sockets persist across tests and confuse the bookkeeping."""
    monkeypatch.setattr(net, "_marker_sock", None)
    monkeypatch.setattr(net, "_ROBOT_SOCK", None)
    monkeypatch.setattr(net, "_generic_sock", None)
    monkeypatch.setattr(net, "_fes_sock", None)


@pytest.fixture
def patched_endpoints(monkeypatch):
    """Spin up three UDP receivers (robot/marker/FES) on 127.0.0.1 and
    monkeypatch the networking module's endpoint snapshots so the
    `send_udp_message` resolver picks them. Returns the receiver sockets
    keyed by role."""
    robot_s, robot_port = _bind_recv_udp()
    marker_s, marker_port = _bind_recv_udp()
    fes_s, fes_port = _bind_recv_udp()

    # Override module globals AND the cached _config so both the
    # endpoint constants (used by _is_robot etc.) and the live config
    # lookup return the same values.
    monkeypatch.setattr(net, "_ROBOT_IP", "127.0.0.1")
    monkeypatch.setattr(net, "_ROBOT_PORT", robot_port)
    monkeypatch.setattr(net, "_MARKER_IP", "127.0.0.1")
    monkeypatch.setattr(net, "_MARKER_PORT", marker_port)

    # `send_udp_message` calls getattr(_config, "UDP_ROBOT", {}) etc. —
    # patch the cached _config namespace so those reads agree.
    cfg = net._config
    monkeypatch.setitem(cfg.UDP_ROBOT, "IP", "127.0.0.1")
    monkeypatch.setitem(cfg.UDP_ROBOT, "PORT", robot_port)
    monkeypatch.setitem(cfg.UDP_MARKER, "IP", "127.0.0.1")
    monkeypatch.setitem(cfg.UDP_MARKER, "PORT", marker_port)
    monkeypatch.setitem(cfg.UDP_FES, "IP", "127.0.0.1")
    monkeypatch.setitem(cfg.UDP_FES, "PORT", fes_port)

    yield {"robot": robot_s, "marker": marker_s, "fes": fes_s,
           "robot_port": robot_port, "marker_port": marker_port,
           "fes_port": fes_port}

    for s in (robot_s, marker_s, fes_s):
        s.close()


# ─── SIMULATION_MODE robot suppression ────────────────────────────────────

class TestSimulationModeRobotSuppression:
    def test_robot_send_is_suppressed(self, patched_endpoints, monkeypatch):
        """With SIMULATION_MODE=True, send_udp_message to the robot
        endpoint must NOT actually put a datagram on the wire
        (file:755-759)."""
        monkeypatch.setattr(net, "SIMULATION_MODE", True)
        result = net.send_udp_message(
            None, "127.0.0.1", patched_endpoints["robot_port"], "h;dur=3",
            quiet=True,
        )
        payload, _ = _drain(patched_endpoints["robot"], timeout=0.15)
        assert payload is None, "Robot datagram leaked under SIMULATION_MODE"
        assert result is None  # non-ack path returns None

    def test_robot_send_with_expect_ack_returns_simulated_ok(
            self, patched_endpoints, monkeypatch):
        """In simulation, expect_ack robot calls still need to return
        (True, None) so the driver's call-site doesn't hang on a fake
        ACK (file:757-758)."""
        monkeypatch.setattr(net, "SIMULATION_MODE", True)
        result = net.send_udp_message(
            None, "127.0.0.1", patched_endpoints["robot_port"], "h;dur=3",
            quiet=True, expect_ack=True,
        )
        assert result == (True, None)

    def test_simulation_mode_does_not_suppress_marker(
            self, patched_endpoints, monkeypatch):
        """Marker sends must always go through, even in SIMULATION_MODE
        (file:696, file:755 — only the robot is gated)."""
        monkeypatch.setattr(net, "SIMULATION_MODE", True)
        net.send_udp_message(
            None, "127.0.0.1", patched_endpoints["marker_port"], "200",
            quiet=True,
        )
        payload, _ = _drain(patched_endpoints["marker"], timeout=0.3)
        assert payload == "200"

    def test_simulation_mode_does_not_suppress_fes(
            self, patched_endpoints, monkeypatch):
        """FES sends must always go through (file:695 — Active in
        simulation)."""
        monkeypatch.setattr(net, "SIMULATION_MODE", True)
        net.send_udp_message(
            None, "127.0.0.1", patched_endpoints["fes_port"], "FES_SENS_GO",
            quiet=True,
        )
        payload, _ = _drain(patched_endpoints["fes"], timeout=0.3)
        assert payload == "FES_SENS_GO"


# ─── FES re-routing ──────────────────────────────────────────────────────

class TestFESRerouting:
    def test_fes_message_to_wrong_endpoint_is_rerouted_to_fes(
            self, patched_endpoints, monkeypatch):
        """A "FES_" message sent to any non-FES IP/port should be
        re-routed to the configured UDP_FES endpoint
        (file:742-751). This prevents the historical bind-error class
        where FES traffic landed on the robot control socket."""
        monkeypatch.setattr(net, "SIMULATION_MODE", False)
        # Send to the robot endpoint with a FES_ prefix — must end up
        # on the FES receiver, NOT the robot receiver.
        net.send_udp_message(
            None, "127.0.0.1", patched_endpoints["robot_port"],
            "FES_MOTOR_GO", quiet=True,
        )
        # FES receiver gets it
        payload, _ = _drain(patched_endpoints["fes"], timeout=0.3)
        assert payload == "FES_MOTOR_GO"
        # Robot receiver does NOT
        robot_payload, _ = _drain(patched_endpoints["robot"], timeout=0.05)
        assert robot_payload is None


# ─── socket separation ───────────────────────────────────────────────────

class TestSocketSeparation:
    def test_marker_send_uses_marker_socket_not_robot_socket(
            self, patched_endpoints, monkeypatch):
        """The marker socket is allocated separately by
        `_ensure_marker_socket` (file:228-243) and must not be the same
        object as the control / generic / FES sockets."""
        monkeypatch.setattr(net, "SIMULATION_MODE", True)
        # Trigger a marker send — should populate _marker_sock.
        net.send_udp_message(
            None, "127.0.0.1", patched_endpoints["marker_port"], "100",
            quiet=True,
        )
        assert net._marker_sock is not None
        # The control socket should still be None — marker traffic never
        # binds the control socket.
        assert net._ROBOT_SOCK is None

    def test_unknown_destination_uses_generic_socket_not_control(
            self, patched_endpoints, monkeypatch):
        """Sending to an unknown endpoint (not robot/marker/FES) must use
        the generic ephemeral socket, not the control socket
        (file:782-791)."""
        monkeypatch.setattr(net, "SIMULATION_MODE", True)
        # 127.0.0.1:1 is not configured as any role.
        recv, port = _bind_recv_udp()
        try:
            net.send_udp_message(
                None, "127.0.0.1", port, "hello",
                quiet=True,
            )
            payload, _ = _drain(recv, timeout=0.3)
            assert payload == "hello"
        finally:
            recv.close()
        # Generic socket was bound; control socket was NOT.
        assert net._generic_sock is not None
        assert net._ROBOT_SOCK is None


# ─── _await_ack_blocking ─────────────────────────────────────────────────

class TestAwaitAckBlocking:
    def test_simulation_mode_short_circuits_to_true(self, monkeypatch):
        monkeypatch.setattr(net, "SIMULATION_MODE", True)
        assert net._await_ack_blocking("h", logger=None) is True

    def test_matching_base_token_returns_true(self, monkeypatch):
        """An ACK whose base token matches the expected base token should
        return True. Exercises the post-7b20b1c base-token matching path
        (file:355, 383-387)."""
        monkeypatch.setattr(net, "SIMULATION_MODE", False)
        # Bind a control socket that the helper will recv from.
        ctrl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ctrl.bind(("127.0.0.1", 0))
        ctrl.setblocking(False)
        monkeypatch.setattr(net, "_ROBOT_SOCK", ctrl)
        ctrl_port = ctrl.getsockname()[1]

        # Send the ACK *from* a separate socket (any source addr is fine —
        # `_await_ack_blocking` does not filter by source). Background
        # thread so we can call the blocking helper concurrently.
        def reply_after_a_beat():
            time.sleep(0.05)
            tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            tx.sendto(b"ACK:h;dur=3.000000", ("127.0.0.1", ctrl_port))
            tx.close()
        t = threading.Thread(target=reply_after_a_beat, daemon=True)
        t.start()

        try:
            ok = net._await_ack_blocking("h", logger=None)
        finally:
            t.join(timeout=1.0)
            ctrl.close()

        assert ok is True

    def test_mismatched_token_times_out_false(self, monkeypatch):
        """An ACK for a different command should not satisfy the wait —
        the function should keep listening until timeout and return
        False."""
        monkeypatch.setattr(net, "SIMULATION_MODE", False)
        # Tight timeout so the test finishes fast.
        monkeypatch.setattr(net, "ACK_TIMEOUT", 0.25)

        ctrl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ctrl.bind(("127.0.0.1", 0))
        ctrl.setblocking(False)
        monkeypatch.setattr(net, "_ROBOT_SOCK", ctrl)
        ctrl_port = ctrl.getsockname()[1]

        def reply_wrong_ack():
            time.sleep(0.05)
            tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            tx.sendto(b"ACK:s", ("127.0.0.1", ctrl_port))
            tx.close()
        t = threading.Thread(target=reply_wrong_ack, daemon=True)
        t.start()

        try:
            ok = net._await_ack_blocking("h", logger=None)
        finally:
            t.join(timeout=1.0)
            ctrl.close()

        assert ok is False


# ─── _sendto_robot ───────────────────────────────────────────────────────

class TestSendtoRobot:
    def test_simulation_mode_raises(self, monkeypatch):
        """`_sendto_robot` is the low-level robot send. Under simulation
        it must hard-fail rather than silently no-op — the caller is
        supposed to gate at the higher level (file:331-332)."""
        monkeypatch.setattr(net, "SIMULATION_MODE", True)
        with pytest.raises(RuntimeError, match="SIMULATION_MODE"):
            net._sendto_robot(b"h;dur=3", logger=None)
