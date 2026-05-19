#!/usr/bin/env python3
"""
harmony_free_arm_calibration.py — Free-arm gaze-calibration recorder
(Phase 2.a Track B, per Harmony_Gaze_Calibration_Upgrade_Plan.md §6.1).

Replaces the preset-visit flow used by ``harmony_calibration_exec.py``
with the user-driven free-arm paradigm described in plan §5.3. The
operator (or the participant) physically moves the active arm of the
Harmony robot to gaze-targeted positions; the recorder sends ``m`` to
release the arm, ``c`` to capture the bundle of (joint angles, EE
position, gaze, depth, IMU, head pose), and ``m`` again to free it for
the next target. Coverage is hybrid: a mandatory 3-depth x 3x3 angular
grid (9 points) plus optional free additions.

The recorder talks to two services:

- The robot research interface running ``Gaze_Tracking`` with the
  ``m``/``c`` opcodes (HARMONY-UNIT-4 branch
  ``feature/research-interface-onboard/free-arm`` head ``01d91ea`` or
  later, per
  ``Documents/SoftwareDocs/Reports/Harmony_Gaze_Calibration_CPP_Report.md``
  §2). Standalone UDP socket bound to ``0.0.0.0:8080``, dial
  ``192.168.2.1:8080`` (same wire as
  ``harmony_calibration_exec.py:21-27, 85-88``).
- ``gaze_runner.py`` in ``--mode service`` on the gaze UDP port
  (``config.GAZE_UDP_IP:GAZE_UDP_PORT``). Snapshots include depth /
  IMU / head pose per ``Utils/gaze/gaze_system.py:636-647`` and the
  wire route at ``gaze_runner.py:265-270``.

Output: ``poses_with_gaze_<UTC>_v2_freearm.npz`` with v2 schema —
legacy keys ``T, Q, X, G`` preserved (so v1 readers keep working when
they ignore unknown keys); new keys ``D_cm, D_valid, Miss_mm, IPD_mm,
IMU_w, IMU_fresh, Head_yaw_deg, Head_pitch_deg, Gaze_yaw_deg,
Gaze_pitch_deg, Phase, Target_label`` carry the new sensor channels;
``meta`` carries ``version=2``.

CLAUDE.md alignment: fail-fast on Tier-1-adjacent UDP I/O; no silent
suppression; no scope creep beyond plan §6.1 Track B and §5.7 Phase 1
decision rules (hybrid coverage, capture IMU even when Pass-1 is
active).
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from Utils.perception_clients import udp_request


# =============================================================================
# Network configuration — mirror harmony_calibration_exec.py:21-27, 85-88
# =============================================================================
ROBOT_IP = "192.168.2.1"
ROBOT_PORT = 8080
CONTROL_IP = "0.0.0.0"
CONTROL_PORT = 8080
ACK_TIMEOUT_S = 0.35
TEL_TIMEOUT_S = 0.60
CMD_PROCESS_GRACE_S = 0.40

# Gaze service endpoint comes from config (so a single config_local.py
# edit moves the recorder to a remote gaze_runner host without code
# changes).
GAZE_HOST = str(getattr(config, "GAZE_UDP_IP", "127.0.0.1"))
GAZE_PORT = int(getattr(config, "GAZE_UDP_PORT", 5588))
GAZE_RPC_TIMEOUT_S = float(getattr(config, "GAZE_UDP_TIMEOUT", 0.8) or 0.8)


# =============================================================================
# Recorder configuration
# =============================================================================
ACTIVE_SIDE = os.getenv("HARMONY_ACTIVE_SIDE", "R").upper()
if ACTIVE_SIDE not in ("L", "R"):
    raise ValueError(f"HARMONY_ACTIVE_SIDE must be 'L' or 'R'; got {ACTIVE_SIDE!r}")

# Settle window — discard the first N seconds after a `c` capture is
# committed so the depth/angle smoothers (`_depth_smoother`,
# `_head_yaw_smoother`, etc. in Utils/gaze/gaze_system.py:419-560) have
# converged before we sample the snapshot. Phase 1 doc §8 follow-up #3.
POST_CAPTURE_SETTLE_S = 1.0

# How many consecutive depth_valid=True snapshots we need before we
# accept a capture. Guards against the recorder logging stale NaN
# depth from a momentary unworn transition.
DEPTH_VALID_MIN_CONSECUTIVE = 5

# Sample rate for the "moving" phase log (gaze samples while the user
# moves the arm). 20 Hz mirrors the gaze_runner internal loop cap
# (gaze_runner.py:496 target_loop_hz=20.0); no point asking faster.
MOVING_PHASE_SAMPLE_HZ = 20.0


# =============================================================================
# Hybrid coverage protocol — locked per
# Gaze_Calibration_Sensor_Characterization.md §5
# =============================================================================
# 3 depth bands x 3x3 angular grid = 9 mandatory points; user labels are
# semantic so the analysis script can group by depth band when fitting
# the Mahalanobis metric.
MANDATORY_GRID: List[str] = [
    "near_TL", "near_TC", "near_TR",
    "near_ML", "near_MC", "near_MR",
    "near_BL", "near_BC", "near_BR",
    "mid_TL",  "mid_TC",  "mid_TR",
    "mid_ML",  "mid_MC",  "mid_MR",
    "mid_BL",  "mid_BC",  "mid_BR",
    "far_TL",  "far_TC",  "far_TR",
    "far_ML",  "far_MC",  "far_MR",
    "far_BL",  "far_BC",  "far_BR",
]


# =============================================================================
# UDP plumbing
# =============================================================================
def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


@dataclass
class RobotLink:
    """Owns the UDP socket to Gaze_Tracking. Single instance per session."""
    sock: socket.socket = field(default_factory=lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
    robot_addr: Tuple[str, int] = (ROBOT_IP, ROBOT_PORT)

    def __post_init__(self) -> None:
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((CONTROL_IP, CONTROL_PORT))
        self.sock.settimeout(0.5)

    def send(self, msg: str) -> None:
        print(f"[{_ts()}] TX: {msg}")
        self.sock.sendto(msg.encode("utf-8"), self.robot_addr)

    def recv(self, timeout_s: float) -> Optional[str]:
        self.sock.settimeout(timeout_s)
        try:
            data, _ = self.sock.recvfrom(65535)
            return data.decode("utf-8", errors="ignore")
        except socket.timeout:
            return None

    def send_and_wait_ack(self, msg: str, expect_prefix: Optional[str] = None,
                          timeout: float = ACK_TIMEOUT_S) -> Optional[str]:
        """Send opcode, wait for ACK / ERR. Returns the matched ACK string
        on success, or None on timeout / mismatch. ERR strings are logged
        and returned as None — caller decides whether to retry or abort.

        Plan §7.1 (Session B1 report §3): the `m`/`c` ACKs break the
        legacy `ACK:<opcode>` shape (they emit `ACK:MASTER_FREE` /
        `ACK:CAPTURED_LOCKED`); callers must pass the explicit
        ``expect_prefix`` when needed.
        """
        self.send(msg)
        t0 = time.time()
        while time.time() - t0 < timeout:
            r = self.recv(timeout)
            if not r:
                continue
            r = r.strip()
            if r.startswith("ACK:"):
                print(f"[{_ts()}] {r}")
                if expect_prefix is None or r.startswith(f"ACK:{expect_prefix}"):
                    time.sleep(CMD_PROCESS_GRACE_S)
                    return r
            elif r.startswith("ERR:"):
                print(f"[{_ts()}] {r}")
                return None
        print(f"[{_ts()}] ACK timeout for {msg!r}")
        return None

    def query_state(self) -> Optional[Dict[str, np.ndarray]]:
        """One `q;seq=...` round trip. Returns dict with 'q' (7,) rad and
        'ee' (3,) mm, plus a UTC wall-time stamp, or None on timeout."""
        seq = int(time.time() * 1000) & 0xFFFFFFFF
        self.send(f"q;seq={seq}")
        t0 = time.time()
        while time.time() - t0 < TEL_TIMEOUT_S:
            r = self.recv(TEL_TIMEOUT_S)
            if not r or not r.startswith("{"):
                continue
            try:
                pkt = json.loads(r)
            except json.JSONDecodeError:
                continue
            try:
                if ACTIVE_SIDE == "R":
                    q = np.asarray(pkt["qR"], dtype=float).ravel()
                    ee = np.asarray(pkt["eeR"]["pos_mm"], dtype=float).ravel()
                else:
                    q = np.asarray(pkt["qL"], dtype=float).ravel()
                    ee = np.asarray(pkt["eeL"]["pos_mm"], dtype=float).ravel()
            except (KeyError, TypeError):
                continue
            if q.size < 7 or ee.size < 3:
                continue
            return {"_t": time.time(), "q": q[:7].copy(), "ee": ee[:3].copy()}
        return None

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def gaze_snapshot(include_objects: bool = False,
                  timeout_s: float = GAZE_RPC_TIMEOUT_S) -> Optional[Dict[str, Any]]:
    """Pull a fresh snapshot from gaze_runner. Returns the dict on
    success or None on transport failure / not-ok response. ``ok=False``
    snapshots are surfaced as None so the recorder treats them the
    same as transport failure (caller will retry / abort)."""
    try:
        snap = udp_request(GAZE_HOST, GAZE_PORT,
                           {"cmd": "snapshot", "include_objects": include_objects},
                           timeout_s)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(snap, dict) or not snap.get("ok", False):
        return None
    return snap


def gaze_recenter(timeout_s: float = GAZE_RPC_TIMEOUT_S) -> bool:
    """Call `cmd: recenter` on gaze_runner. Used at the start of a
    session to zero out the head_yaw/pitch offsets relative to the
    user's neutral pose (Utils/gaze/gaze_system.py:334-356).
    """
    try:
        resp = udp_request(GAZE_HOST, GAZE_PORT, {"cmd": "recenter"}, timeout_s)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(isinstance(resp, dict) and resp.get("ok", False))


# =============================================================================
# Capture bundle
# =============================================================================
@dataclass
class CaptureBundle:
    """One captured sample (`phase` is 'captured' or 'moving')."""
    t: float
    q: np.ndarray
    ee_mm: np.ndarray
    gaze_x_norm: float
    gaze_y_norm: float
    gaze_conf: float
    depth_cm: float
    depth_valid: bool
    miss_mm: float
    ipd_mm: float
    imu_w: float
    imu_fresh: bool
    head_yaw_deg: float
    head_pitch_deg: float
    gaze_yaw_deg: float
    gaze_pitch_deg: float
    phase: str
    target_label: str


def bundle_from_snapshot(snap: Dict[str, Any], robot_t: float,
                         q: np.ndarray, ee_mm: np.ndarray,
                         phase: str, target_label: str) -> CaptureBundle:
    """Pull every recorder-relevant field out of a gaze_runner snapshot
    and pair it with the most recent robot telemetry. Fields source:
    Utils/gaze/gaze_system.py:623-674 (snapshot contract).
    """
    gaze_px = snap.get("gaze_px") or (float("nan"), float("nan"))
    try:
        gx_px, gy_px = float(gaze_px[0]), float(gaze_px[1])
    except (TypeError, ValueError, IndexError):
        gx_px, gy_px = float("nan"), float("nan")

    # Normalise to [0,1] to match the legacy G column convention
    # (harmony_calibration_exec.py:395-401).
    width = float(getattr(config, "GAZE_SAMPLE_WIDTH", 1600.0))
    height = float(getattr(config, "GAZE_SAMPLE_HEIGHT", 1200.0))
    gx_norm = gx_px / width if np.isfinite(gx_px) else float("nan")
    gy_norm = gy_px / height if np.isfinite(gy_px) else float("nan")

    imu_w = snap.get("imu_angvel")
    imu_w_f = float(imu_w) if imu_w is not None and np.isfinite(imu_w) else float("nan")

    return CaptureBundle(
        t=robot_t,
        q=q.copy(),
        ee_mm=ee_mm.copy(),
        gaze_x_norm=gx_norm,
        gaze_y_norm=gy_norm,
        # Snapshot has `worn` but no per-gaze conf; v1 G[:,2] carried
        # `confidence` from the LSL eye-tracking sample. We approximate
        # with `worn AND depth_valid` mapped to {0.0, 1.0}. Downstream
        # NN treats this as a validity gate, not a continuous score.
        gaze_conf=float(1.0 if (snap.get("worn") and snap.get("depth_valid")) else 0.0),
        depth_cm=float(snap.get("depth_cm", float("nan"))),
        depth_valid=bool(snap.get("depth_valid", False)),
        miss_mm=float(snap.get("miss_mm", float("nan"))),
        ipd_mm=float(snap.get("ipd_mm", float("nan"))),
        imu_w=imu_w_f,
        imu_fresh=bool(snap.get("imu_fresh", False)),
        head_yaw_deg=float(snap.get("head_yaw_deg", float("nan"))),
        head_pitch_deg=float(snap.get("head_pitch_deg", float("nan"))),
        gaze_yaw_deg=float(snap.get("gaze_yaw_deg", float("nan"))),
        gaze_pitch_deg=float(snap.get("gaze_pitch_deg", float("nan"))),
        phase=phase,
        target_label=target_label,
    )


# =============================================================================
# Capture-flow primitives
# =============================================================================
def free_arm(link: RobotLink) -> bool:
    """Send `m` and wait for `ACK:MASTER_FREE` (Session B1 report §3:
    wire protocol additions for m/c break the legacy `ACK:<opcode>`
    shape; we expect the literal `MASTER_FREE` body)."""
    return link.send_and_wait_ack("m", expect_prefix="MASTER_FREE") is not None


def capture_pose(link: RobotLink) -> Optional[Dict[str, np.ndarray]]:
    """Send `c`, wait for `ACK:CAPTURED_LOCKED`, then consume the
    follow-up telemetry JSON the C++ side emits in the same round trip
    (Session B1 report §3 / wire_protocol.md "m/c" rows). Returns the
    captured `{q, ee, _t}` dict or None on failure.
    """
    link.send("c")
    t0 = time.time()
    ack_seen = False
    while time.time() - t0 < ACK_TIMEOUT_S + TEL_TIMEOUT_S:
        r = link.recv(TEL_TIMEOUT_S)
        if not r:
            continue
        r = r.strip()
        if r.startswith("ACK:CAPTURED_LOCKED"):
            print(f"[{_ts()}] {r}")
            ack_seen = True
            continue
        if r.startswith("ERR:"):
            print(f"[{_ts()}] {r}")
            return None
        if ack_seen and r.startswith("{"):
            try:
                pkt = json.loads(r)
            except json.JSONDecodeError:
                continue
            try:
                if ACTIVE_SIDE == "R":
                    q = np.asarray(pkt["qR"], dtype=float).ravel()
                    ee = np.asarray(pkt["eeR"]["pos_mm"], dtype=float).ravel()
                else:
                    q = np.asarray(pkt["qL"], dtype=float).ravel()
                    ee = np.asarray(pkt["eeL"]["pos_mm"], dtype=float).ravel()
            except (KeyError, TypeError):
                continue
            if q.size < 7 or ee.size < 3:
                continue
            return {"_t": time.time(), "q": q[:7].copy(), "ee": ee[:3].copy()}
    return None


def settle_and_snapshot(target_label: str,
                        robot_state: Dict[str, np.ndarray]) -> Optional[CaptureBundle]:
    """Wait POST_CAPTURE_SETTLE_S after a `c` lock, then sample the
    gaze snapshot. Returns the bundle, or None if we cannot find a
    snapshot with DEPTH_VALID_MIN_CONSECUTIVE consecutive valid depth
    samples inside a bounded retry window.

    Phase 1 doc §8 follow-up #3: the depth smoother is an EMA, so the
    first sample after a worn-transition under-reports. The settle
    plus the consecutive-valid gate together protect against logging
    a stale value.
    """
    print(f"[{_ts()}] Settling {POST_CAPTURE_SETTLE_S:.2f}s before snapshot…")
    time.sleep(POST_CAPTURE_SETTLE_S)

    consecutive_valid = 0
    last_snap: Optional[Dict[str, Any]] = None
    poll_dt = 1.0 / MOVING_PHASE_SAMPLE_HZ
    deadline = time.time() + 2.0  # bounded retry; never spin forever
    while time.time() < deadline:
        snap = gaze_snapshot(include_objects=False)
        if snap is None:
            consecutive_valid = 0
            time.sleep(poll_dt)
            continue
        last_snap = snap
        if bool(snap.get("depth_valid", False)):
            consecutive_valid += 1
            if consecutive_valid >= DEPTH_VALID_MIN_CONSECUTIVE:
                break
        else:
            consecutive_valid = 0
        time.sleep(poll_dt)

    if last_snap is None:
        print(f"[{_ts()}] No gaze snapshot available for capture {target_label}")
        return None

    if consecutive_valid < DEPTH_VALID_MIN_CONSECUTIVE:
        print(f"[{_ts()}] WARN: depth_valid streak {consecutive_valid} "
              f"< required {DEPTH_VALID_MIN_CONSECUTIVE} — recording bundle anyway "
              f"(Pass-1 IMU path tolerates NaN depth; Pass-2 will drop)")
    return bundle_from_snapshot(last_snap, robot_state["_t"], robot_state["q"],
                                robot_state["ee"], phase="captured",
                                target_label=target_label)


def collect_moving_phase(link: RobotLink, target_label: str,
                         duration_s: float) -> List[CaptureBundle]:
    """Stream gaze + telemetry while the user is actively moving the
    arm (i.e. between `m` and `c`). Used to log the user's reach path,
    which downstream trajectory-based methods may consume. Plan §6.1
    Track B step 3.
    """
    bundles: List[CaptureBundle] = []
    poll_dt = 1.0 / MOVING_PHASE_SAMPLE_HZ
    t_end = time.time() + duration_s
    next_t = time.time()
    while time.time() < t_end:
        next_t += poll_dt
        snap = gaze_snapshot(include_objects=False)
        rstate = link.query_state()
        if snap is None or rstate is None:
            sleep_for = max(0.0, next_t - time.time())
            time.sleep(sleep_for)
            continue
        bundles.append(bundle_from_snapshot(snap, rstate["_t"], rstate["q"],
                                            rstate["ee"], phase="moving",
                                            target_label=target_label))
        sleep_for = max(0.0, next_t - time.time())
        time.sleep(sleep_for)
    return bundles


# =============================================================================
# Operator interaction
# =============================================================================
def prompt_user(message: str) -> str:
    """Stdin read with a timestamp prefix. Newlines flush so the
    operator console stays readable when stdout is line-buffered."""
    print(f"[{_ts()}] {message}", flush=True)
    try:
        return input().strip()
    except EOFError:
        return ""


def announce_target(target_label: str, idx: int, total: int) -> None:
    print()
    print("=" * 70)
    print(f"[{_ts()}] TARGET {idx}/{total}: {target_label}")
    print(f"[{_ts()}] >>> LOOK AT THE TARGET. MOVE THE ARM THERE BY HAND. <<<")
    print(f"[{_ts()}] Press Enter when arm is positioned and you are fixating.")
    print("=" * 70)


# =============================================================================
# Output writer — v2 NPZ schema (plan §6.1 Track A step 2; Track B reuses)
# =============================================================================
def write_npz(bundles: List[CaptureBundle], out_path: str) -> None:
    """Materialise CaptureBundle list as v2 NPZ. Legacy keys T/Q/X/G
    are populated from captured (phase='captured') samples only, so
    legacy v1 consumers see exactly the same content shape they
    expected — the new keys are additive. Moving-phase bundles are
    written under the *_all keys for trajectory-based consumers.

    Output contract (v2):
      Legacy (v1-compatible) — captured samples only:
        T: (N,) float64           timestamps (s)
        Q: (N, 7) float64         joint angles (rad)
        X: (N, 3) float64         EE positions (mm)
        G: (N, 3) float64         (gaze_x_norm, gaze_y_norm, gaze_conf)
      New (v2 only) — captured samples:
        D_cm           (N,) float64
        D_valid        (N,) bool
        Miss_mm        (N,) float64
        IPD_mm         (N,) float64
        IMU_w          (N,) float64
        IMU_fresh      (N,) bool
        Head_yaw_deg   (N,) float64
        Head_pitch_deg (N,) float64
        Gaze_yaw_deg   (N,) float64
        Gaze_pitch_deg (N,) float64
        Target_label   (N,) <U32
      New (v2 only) — all samples (captured + moving), indexed by phase:
        T_all, Q_all, X_all, G_all, D_cm_all, D_valid_all,
        Miss_mm_all, IPD_mm_all, IMU_w_all, IMU_fresh_all,
        Head_yaw_deg_all, Head_pitch_deg_all, Gaze_yaw_deg_all,
        Gaze_pitch_deg_all, Phase_all, Target_label_all
      meta: dict with `version`, `side`, recorder identifiers.
    """
    captured = [b for b in bundles if b.phase == "captured"]
    if not captured:
        raise RuntimeError("Refusing to write NPZ with zero captured samples.")

    def stack_field(items, attr, dtype=float):
        return np.asarray([getattr(b, attr) for b in items], dtype=dtype)

    def vstack_field(items, attr):
        return np.vstack([getattr(b, attr) for b in items])

    T = stack_field(captured, "t")
    Q = vstack_field(captured, "q")
    X = vstack_field(captured, "ee_mm")
    G = np.column_stack([
        stack_field(captured, "gaze_x_norm"),
        stack_field(captured, "gaze_y_norm"),
        stack_field(captured, "gaze_conf"),
    ])

    D_cm = stack_field(captured, "depth_cm")
    D_valid = stack_field(captured, "depth_valid", dtype=bool)
    Miss_mm = stack_field(captured, "miss_mm")
    IPD_mm = stack_field(captured, "ipd_mm")
    IMU_w = stack_field(captured, "imu_w")
    IMU_fresh = stack_field(captured, "imu_fresh", dtype=bool)
    Head_yaw_deg = stack_field(captured, "head_yaw_deg")
    Head_pitch_deg = stack_field(captured, "head_pitch_deg")
    Gaze_yaw_deg = stack_field(captured, "gaze_yaw_deg")
    Gaze_pitch_deg = stack_field(captured, "gaze_pitch_deg")
    Target_label = np.asarray([b.target_label for b in captured], dtype="<U32")

    # "All" arrays — captured + moving, in time order.
    T_all = stack_field(bundles, "t")
    Q_all = vstack_field(bundles, "q")
    X_all = vstack_field(bundles, "ee_mm")
    G_all = np.column_stack([
        stack_field(bundles, "gaze_x_norm"),
        stack_field(bundles, "gaze_y_norm"),
        stack_field(bundles, "gaze_conf"),
    ])
    D_cm_all = stack_field(bundles, "depth_cm")
    D_valid_all = stack_field(bundles, "depth_valid", dtype=bool)
    Miss_mm_all = stack_field(bundles, "miss_mm")
    IPD_mm_all = stack_field(bundles, "ipd_mm")
    IMU_w_all = stack_field(bundles, "imu_w")
    IMU_fresh_all = stack_field(bundles, "imu_fresh", dtype=bool)
    Head_yaw_deg_all = stack_field(bundles, "head_yaw_deg")
    Head_pitch_deg_all = stack_field(bundles, "head_pitch_deg")
    Gaze_yaw_deg_all = stack_field(bundles, "gaze_yaw_deg")
    Gaze_pitch_deg_all = stack_field(bundles, "gaze_pitch_deg")
    Phase_all = np.asarray([b.phase for b in bundles], dtype="<U16")
    Target_label_all = np.asarray([b.target_label for b in bundles], dtype="<U32")

    meta = dict(
        version=2,
        side=ACTIVE_SIDE,
        recorder="harmony_free_arm_calibration.py",
        plan_reference="Harmony_Gaze_Calibration_Upgrade_Plan.md §6.1 Track B",
        cpp_branch="feature/research-interface-onboard/free-arm @ 01d91ea",
        wire_protocol="HARMONY-UNIT-4 tools/wire_protocol.md (m/c rows)",
        gaze_sample_width=float(getattr(config, "GAZE_SAMPLE_WIDTH", 1600.0)),
        gaze_sample_height=float(getattr(config, "GAZE_SAMPLE_HEIGHT", 1200.0)),
        post_capture_settle_s=POST_CAPTURE_SETTLE_S,
        depth_valid_min_consecutive=DEPTH_VALID_MIN_CONSECUTIVE,
        units=dict(X="mm", Q="rad", G="normalized_0_to_1",
                   D_cm="cm", Miss_mm="mm", IPD_mm="mm",
                   IMU_w="rad/s", Head_yaw_deg="deg",
                   Head_pitch_deg="deg", Gaze_yaw_deg="deg",
                   Gaze_pitch_deg="deg"),
    )

    np.savez_compressed(
        out_path,
        # legacy v1 keys
        T=T, Q=Q, X=X, G=G,
        # v2 captured-only keys
        D_cm=D_cm, D_valid=D_valid, Miss_mm=Miss_mm, IPD_mm=IPD_mm,
        IMU_w=IMU_w, IMU_fresh=IMU_fresh,
        Head_yaw_deg=Head_yaw_deg, Head_pitch_deg=Head_pitch_deg,
        Gaze_yaw_deg=Gaze_yaw_deg, Gaze_pitch_deg=Gaze_pitch_deg,
        Target_label=Target_label,
        # v2 all-phase keys
        T_all=T_all, Q_all=Q_all, X_all=X_all, G_all=G_all,
        D_cm_all=D_cm_all, D_valid_all=D_valid_all,
        Miss_mm_all=Miss_mm_all, IPD_mm_all=IPD_mm_all,
        IMU_w_all=IMU_w_all, IMU_fresh_all=IMU_fresh_all,
        Head_yaw_deg_all=Head_yaw_deg_all, Head_pitch_deg_all=Head_pitch_deg_all,
        Gaze_yaw_deg_all=Gaze_yaw_deg_all, Gaze_pitch_deg_all=Gaze_pitch_deg_all,
        Phase_all=Phase_all, Target_label_all=Target_label_all,
        meta=meta,
    )


# =============================================================================
# Main session loop
# =============================================================================
def run_session(out_dir: str = ".") -> Optional[str]:
    """Run the full free-arm calibration session. Returns the output
    NPZ path on success, or None if the session was aborted."""
    print()
    print("=" * 70)
    print("  HARMONY FREE-ARM GAZE CALIBRATION (Track B)")
    print("=" * 70)
    print(f"[{_ts()}] Active side: {ACTIVE_SIDE}")
    print(f"[{_ts()}] Robot endpoint: {ROBOT_IP}:{ROBOT_PORT}")
    print(f"[{_ts()}] Gaze service:  {GAZE_HOST}:{GAZE_PORT}")
    print()

    link = RobotLink()
    print(f"[{_ts()}] Bound UDP socket {link.sock.getsockname()} -> {link.robot_addr}")

    bundles: List[CaptureBundle] = []
    try:
        # Recenter head/gaze offsets relative to the user's neutral pose
        # before any captures. Without this, head_yaw_deg etc. read in
        # the world frame and the Pass-2 mapping inherits whatever offset
        # the user was sitting at. Phase 1 doc §8 follow-up #1.
        print(f"[{_ts()}] Ask the user to face their neutral position, then press Enter.")
        prompt_user("Press Enter when ready to recenter the gaze tracker…")
        if not gaze_recenter():
            print(f"[{_ts()}] WARN: gaze_runner did not ACK recenter. "
                  f"Head/gaze angles will be recorded in the absolute world frame.")
        else:
            print(f"[{_ts()}] Gaze recentered.")

        # Workspace coverage protocol (Phase 1 doc §5 lock-in): mandatory
        # 3-depth x 3x3 grid, then user-driven free additions.
        operator_targets = list(MANDATORY_GRID)
        print(f"[{_ts()}] Mandatory grid has {len(operator_targets)} points.")
        print(f"[{_ts()}] After the grid, you'll be prompted for optional free additions.")

        idx = 0
        total = len(operator_targets)
        while idx < total:
            target_label = operator_targets[idx]
            announce_target(target_label, idx + 1, total)

            # 1) Free the arm.
            if not free_arm(link):
                print(f"[{_ts()}] ERR: `m` rejected — likely robot is still in MOVING. Retrying after 1s.")
                time.sleep(1.0)
                continue

            # 2) Wait for the operator to position the arm. The prompt
            #    returns the line so the operator can type 's' to skip
            #    a target they cannot reach with the active arm.
            ans = prompt_user("Move the arm and fixate, then press Enter to capture (or 's' to skip)…")
            if ans.lower() == "s":
                print(f"[{_ts()}] Operator skipped {target_label}.")
                idx += 1
                continue

            # 3) Stream the arm motion as "moving" samples for ~1s prior
            #    to capture — gives the trajectory-based consumers a
            #    short pre-capture history without depending on the
            #    operator's reaction time.
            moving = collect_moving_phase(link, target_label, duration_s=1.0)

            # 4) Capture.
            captured_state = capture_pose(link)
            if captured_state is None:
                print(f"[{_ts()}] ERR: `c` failed at {target_label}. Retrying this target.")
                continue

            # 5) Settle + sample.
            bundle = settle_and_snapshot(target_label, captured_state)
            if bundle is None:
                print(f"[{_ts()}] WARN: no gaze snapshot at {target_label}. Skipping.")
                # Re-free for the next attempt at the same target.
                idx += 1  # advance anyway — operator will fill via free-additions phase if needed
                continue

            bundles.extend(moving)
            bundles.append(bundle)
            print(f"[{_ts()}] Captured {target_label}: "
                  f"q={np.round(bundle.q, 3).tolist()}, "
                  f"depth={bundle.depth_cm:.1f}cm (valid={bundle.depth_valid}), "
                  f"head=({bundle.head_yaw_deg:+.1f},{bundle.head_pitch_deg:+.1f})")
            idx += 1

        # Free-additions phase.
        print()
        print("=" * 70)
        print(f"[{_ts()}] Mandatory grid complete ({len(bundles)} bundles, "
              f"{sum(1 for b in bundles if b.phase == 'captured')} captures).")
        print(f"[{_ts()}] Optional free-additions phase: add reach hotspots.")
        print("=" * 70)

        free_idx = 0
        while True:
            free_idx += 1
            ans = prompt_user(f"Add free target #{free_idx}? Enter label (e.g. 'mug_handle'), or empty to finish: ")
            if not ans:
                break
            target_label = f"free_{ans}"
            announce_target(target_label, free_idx, free_idx)

            if not free_arm(link):
                print(f"[{_ts()}] ERR: `m` rejected. Skipping.")
                continue
            prompt_user("Move the arm and fixate, then press Enter to capture…")
            moving = collect_moving_phase(link, target_label, duration_s=1.0)
            captured_state = capture_pose(link)
            if captured_state is None:
                print(f"[{_ts()}] ERR: `c` failed at {target_label}. Skipping.")
                continue
            bundle = settle_and_snapshot(target_label, captured_state)
            if bundle is None:
                print(f"[{_ts()}] WARN: no gaze snapshot at {target_label}. Skipping.")
                continue
            bundles.extend(moving)
            bundles.append(bundle)
            print(f"[{_ts()}] Captured {target_label}.")

        # Final home — release the arm into a held position so the user
        # is not holding it up at session end.
        print()
        print(f"[{_ts()}] Returning the arm to home position…")
        link.send_and_wait_ack("h", expect_prefix="h")

        # Write the NPZ.
        if not bundles:
            print(f"[{_ts()}] No bundles captured. Aborting (no file written).")
            return None

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_name = f"poses_with_gaze_{stamp}_v2_freearm.npz"
        out_path = os.path.join(out_dir, out_name)
        write_npz(bundles, out_path)
        captured_n = sum(1 for b in bundles if b.phase == "captured")
        print(f"[{_ts()}] Wrote {out_path}")
        print(f"[{_ts()}]   captured samples: {captured_n}")
        print(f"[{_ts()}]   moving samples:   {len(bundles) - captured_n}")
        return out_path

    finally:
        link.close()


def main() -> int:
    try:
        out = run_session(out_dir=".")
    except KeyboardInterrupt:
        print(f"\n[{_ts()}] KeyboardInterrupt — aborting session.")
        return 1
    return 0 if out else 2


if __name__ == "__main__":
    sys.exit(main())
