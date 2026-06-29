"""
harmony_link.py — minimal UDP client for the Harmony robot, shared by the
AprilTag calibration + control-test tools (WS5).

Mirrors the proven wire contract of
``harmony_free_arm_calibration.RobotLink`` (:144-228) and the online command
path in ``Utils/networking.py`` (the coords→``g`` staged-trajectory sequence),
without importing those Tier-1/2 modules (which drag pygame / experiment-loop
deps). Methodology + verified robot contract:
SoftwareDocs/_archive/harmony-bci/gaze-calibration/rev03-apriltag-methodology.md
(archived; current model in the active rev04-planar-coverage-methodology.md) and
reports/cpp.md (the C++ ``Gaze_Tracking.cpp`` / ``wire_protocol.md`` side).

Verified robot facts this honours (reports/cpp.md):
  - Position is commanded ONLY as 7 comma-separated joint angles (radians) with
    an optional ``;dur=`` suffix, then ``g`` to commit. There is no Cartesian/EE
    command — hence the calibration's (X→Q) library + this joint command.
  - ``q;seq`` telemetry returns BOTH joint angles (qR/qL) and EE position
    (eeR/eeL.pos_mm) in one JSON reply.
  - Command ACKs (m/c/coords/g/h) go to the FIXED control endpoint
    (``config.UDP_CONTROL_BIND``); only ``q`` replies to the sender. So this
    binds the control endpoint (with SO_REUSEADDR) to receive ACKs — an
    ephemeral source port would miss them. Because only one process may hold
    that endpoint, running alongside the recorder/online driver fails fast with
    EADDRINUSE (the one-binder-per-host guard).
  - The robot enforces NO workspace bounds (C++ §7.2 deferred) — callers MUST
    clamp joint commands themselves.

The pure helpers (``build_joint_command_str``, ``parse_telemetry``) are
hardware-free and unit-tested; the socket methods need the robot.
"""

from __future__ import annotations

import json
import socket
import time
from typing import Dict, Optional

import numpy as np

# Wire timings mirror the recorder / networking defaults.
ACK_TIMEOUT_S = 2.0
TEL_TIMEOUT_S = 0.5
CMD_PROCESS_GRACE_S = 0.40   # settle after an ACK before the next opcode
STAGE_TO_GO_DELAY_S = 0.10   # between staging coords and sending `g`
_ARM_JOINTS = 7


def build_joint_command_str(q, dur_s: float) -> str:
    """The robot's joint-command wire string: 7 comma-separated radians + a
    ``;dur=`` suffix (matches ``ExperimentDriver_Online_GazeTracking.build_joint_command``
    and the C++ 7-CSV opcode). Raises if ``q`` is not exactly 7 values — the C++
    side rejects anything else with ``ERR:coords_require_exactly_7_radians``, and
    a malformed command must never reach the robot."""
    q = np.asarray(q, dtype=float).ravel()
    if q.size != _ARM_JOINTS:
        raise ValueError(f"joint command needs exactly {_ARM_JOINTS} values; got {q.size}")
    if not np.all(np.isfinite(q)):
        raise ValueError("joint command contains non-finite values")
    return ",".join(f"{v:.6f}" for v in q.tolist()) + f";dur={float(dur_s):.3f}"


def parse_telemetry(text: str, side: str) -> Optional[Dict[str, np.ndarray]]:
    """Parse a ``q;seq`` / capture telemetry JSON line into
    ``{q (7,), ee (3,), ee_quat (4,)|None}`` for the active side, or None if it
    isn't a usable telemetry packet. Mirrors ``RobotLink.query_state`` (:210-221) —
    q is joint angles (rad), ee is EE position (mm, base frame). Used for both the
    calibration (X,Q) capture and the settle check.

    ``ee_quat`` is the EE orientation quaternion ``[x, y, z, w]`` (scalar-last),
    which the firmware ships alongside ``pos_mm`` (``Gaze_Tracking.cpp:836/839``,
    ``pose.h:15-17``) but every Python consumer historically dropped. It is parsed
    **defensively** — absent or malformed → None — so the long-standing q/ee
    contract is byte-identical for callers that don't need orientation. Consumed
    by the WS-2a FK-orientation hand-eye solve (offline); never by the motion
    path. The frame of the quat is fixed by the (closed-source) Harmony SHR
    manager, not this codebase; callers must establish it empirically."""
    text = text.strip()
    if not text.startswith("{"):
        return None
    try:
        pkt = json.loads(text)
    except json.JSONDecodeError:
        return None
    key_q = "qR" if side.upper() == "R" else "qL"
    key_ee = "eeR" if side.upper() == "R" else "eeL"
    try:
        q = np.asarray(pkt[key_q], dtype=float).ravel()
        ee = np.asarray(pkt[key_ee]["pos_mm"], dtype=float).ravel()
    except (KeyError, TypeError, ValueError):
        return None
    if q.size < _ARM_JOINTS or ee.size < 3:
        return None
    ee_quat = None
    try:
        quat = np.asarray(pkt[key_ee]["quat"], dtype=float).ravel()
        if quat.size >= 4 and np.all(np.isfinite(quat[:4])):
            ee_quat = quat[:4].copy()
    except (KeyError, TypeError, ValueError):
        ee_quat = None
    return {"q": q[:_ARM_JOINTS].copy(), "ee": ee[:3].copy(), "ee_quat": ee_quat}


class HarmonyLink:
    """UDP link to the robot. Binds the fixed control endpoint (so it receives
    command ACKs) and dials the robot. Sends only documented opcodes; the
    control tool's caller is responsible for the workspace clamp before any
    ``send_joint_command``."""

    def __init__(self, robot_ip: str, robot_port: int, bind_ip: str,
                 bind_port: int, side: str = "R", timeout_s: float = TEL_TIMEOUT_S):
        self._addr = (robot_ip, int(robot_port))
        self._side = side.upper()
        self._seq = 0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((bind_ip, int(bind_port)))  # EADDRINUSE if a recorder holds it
        self._sock.settimeout(timeout_s)

    # ── low-level ────────────────────────────────────────────────────────────
    def send(self, msg: str) -> None:
        self._sock.sendto(msg.encode("utf-8"), self._addr)

    def recv(self, timeout_s: float) -> Optional[str]:
        self._sock.settimeout(timeout_s)
        try:
            data, _ = self._sock.recvfrom(65535)
        except socket.timeout:
            return None
        return data.decode("utf-8", errors="ignore")

    def send_and_wait_ack(self, msg: str, expect_prefix: Optional[str] = None,
                          timeout: float = ACK_TIMEOUT_S) -> Optional[str]:
        """Send an opcode, return the matched ``ACK:...`` string or None on
        timeout / mismatch / ``ERR:``. Mirrors ``RobotLink.send_and_wait_ack``
        (:167-194) — the m/c ACKs break the legacy ``ACK:<opcode>`` shape, so
        callers pass the explicit ``expect_prefix`` (e.g. ``MASTER_FREE``)."""
        self.send(msg)
        t0 = time.time()
        while time.time() - t0 < timeout:
            r = self.recv(timeout)
            if not r:
                continue
            r = r.strip()
            if r.startswith("ACK:"):
                if expect_prefix is None or r.startswith(f"ACK:{expect_prefix}"):
                    time.sleep(CMD_PROCESS_GRACE_S)
                    return r
            elif r.startswith("ERR:"):
                print(f"[harmony_link] {r}", flush=True)
                return None
        return None

    # ── telemetry (read-only) ─────────────────────────────────────────────────
    def query_state(self) -> Optional[Dict[str, np.ndarray]]:
        """One ``q;seq`` round-trip → ``{q (7,), ee (3,), ee_quat (4,)|None, _t}``
        or None. Read-only.

        ``_t`` is the control-host ``time.time()`` at which the telemetry reply was
        received — the SAME clock the relay consumer stamps a frame's arrival with
        (`RelayConsumer.latest_with_time`), so the sweep's frame↔telemetry Δt gate
        (rev04 §2) is a pure host-clock staleness measure, not a cross-clock skew."""
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        self.send(f"q;seq={self._seq}")
        t0 = time.time()
        while time.time() - t0 < TEL_TIMEOUT_S:
            r = self.recv(TEL_TIMEOUT_S)
            if r is None:
                continue
            parsed = parse_telemetry(r, self._side)
            if parsed is not None:
                parsed["_t"] = time.time()
                return parsed
        return None

    # ── free-arm capture (m/c) ────────────────────────────────────────────────
    def free_arm(self) -> bool:
        """``m`` — zero-stiffness free the arm. ACK ``MASTER_FREE``."""
        return self.send_and_wait_ack("m", expect_prefix="MASTER_FREE") is not None

    def capture_pose(self) -> Optional[Dict[str, np.ndarray]]:
        """``c`` — capture + lock at the current joints; consume the follow-up
        telemetry line. Returns ``{q (7,), ee (3,), ee_quat (4,)|None}`` or None.
        Mirrors ``harmony_free_arm_calibration.capture_pose`` (:607-645)."""
        self.send("c")
        t0 = time.time()
        ack_seen = False
        while time.time() - t0 < ACK_TIMEOUT_S + TEL_TIMEOUT_S:
            r = self.recv(TEL_TIMEOUT_S)
            if not r:
                continue
            r = r.strip()
            if r.startswith("ACK:CAPTURED_LOCKED"):
                ack_seen = True
                continue
            if r.startswith("ERR:"):
                print(f"[harmony_link] {r}", flush=True)
                return None
            if ack_seen:
                parsed = parse_telemetry(r, self._side)
                if parsed is not None:
                    return parsed
        return None

    # ── motion ────────────────────────────────────────────────────────────────
    def home(self, dur_s: float = 4.0) -> bool:
        """``h;dur=`` — drive to the home pose over ``dur_s`` seconds."""
        return self.send_and_wait_ack(f"h;dur={float(dur_s):.3f}",
                                      expect_prefix="h") is not None

    def send_joint_command(self, q, dur_s: float) -> str:
        """Command a joint pose: stage the 7-CSV trajectory (await
        ``ACK:COORDS_STAGED_RAD``), then ``g`` to commit (await ``ACK:g``). The
        caller MUST have clamped ``q`` to the calibrated workspace first — the
        robot enforces no bounds.

        Returns a status so a lost go-ACK is never confused with a no-op:
          - ``"ok"``            — both ACKs seen; move committed.
          - ``"stage_failed"``  — coords not staged; the arm did NOT move.
          - ``"go_unconfirmed"``— staged but no ``ACK:g`` (lost on the return
            path or the robot is mid-move). The arm MAY be moving; the caller
            must read back and verify rather than treat this as a clean failure."""
        coords = build_joint_command_str(q, dur_s)
        if self.send_and_wait_ack(coords, expect_prefix="COORDS_STAGED_RAD") is None:
            return "stage_failed"
        time.sleep(STAGE_TO_GO_DELAY_S)
        if self.send_and_wait_ack("g", expect_prefix="g") is None:
            return "go_unconfirmed"
        return "ok"

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass
