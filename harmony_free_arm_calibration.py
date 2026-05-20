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
the next target. Coverage is hybrid: a mandatory 3-depth × 5-horizontal
grid = 15 capture points (rightmost-first sweep R1→R5 per depth band)
plus optional free additions. Between captures a background telemetry
thread streams workspace-coverage "transit" samples at ~20 Hz.

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
from Utils.perception_clients import VLMClient, udp_request


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
# 3 depth bands × 5 horizontal positions = 15 mandatory capture points.
# Horizontal labels R1..R5 sweep rightmost-first (R1 = participant's
# right, R5 = participant's left); within each depth band the operator
# walks R1 → R5 before advancing to the next depth band. This pattern
# was reduced from the prior 3×3 grid (27 pts) to cut operator workload
# in half while keeping the same three depth bands; the new background
# telemetry thread covers vertical/intermediate workspace samples as
# transit data.
MANDATORY_GRID: List[str] = [
    "near_R1", "near_R2", "near_R3", "near_R4", "near_R5",
    "mid_R1",  "mid_R2",  "mid_R3",  "mid_R4",  "mid_R5",
    "far_R1",  "far_R2",  "far_R3",  "far_R4",  "far_R5",
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
# VLM depth (Depth Pro) — opt-in via config.GAZE_CALIBRATION_DEPTH_SOURCE
# =============================================================================
def verify_vlm_depth_available(vlm_client: VLMClient) -> None:
    """Fail-fast precondition for `GAZE_CALIBRATION_DEPTH_SOURCE='vlm_depth_pro'`.

    The Mahalanobis NN treats depth as a learned-distribution feature;
    silently falling back to vergence mid-session would poison the metric
    by mixing two scales (CLAUDE.md §"Realtime Safety Constraints" plus
    user instruction in the §"Depth source selection" plan addition).
    This function raises RuntimeError if vlm_service is unreachable or
    has Depth Pro disabled — the caller aborts the recording session.
    """
    try:
        status = vlm_client.status()
    except OSError as e:
        raise RuntimeError(
            f"VLM service unreachable at {vlm_client.host}:{vlm_client.port} ({e}); "
            f"refusing to record with GAZE_CALIBRATION_DEPTH_SOURCE='vlm_depth_pro'. "
            f"Start vlm_service.py with --enable-depth from the control panel first."
        ) from e
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"VLM service at {vlm_client.host}:{vlm_client.port} returned a malformed "
            f"status reply ({e}); refusing to record with "
            f"GAZE_CALIBRATION_DEPTH_SOURCE='vlm_depth_pro'."
        ) from e
    if not isinstance(status, dict) or not status.get("ok", False):
        raise RuntimeError(
            f"VLM service at {vlm_client.host}:{vlm_client.port} returned ok=False "
            f"on status ({status!r}); refusing to record."
        )
    if not bool(status.get("depth_enabled", False)):
        raise RuntimeError(
            f"VLM service at {vlm_client.host}:{vlm_client.port} reports "
            f"depth_enabled=False; restart it with --enable-depth (and ensure "
            f"the Depth Pro model loaded successfully) before recording with "
            f"GAZE_CALIBRATION_DEPTH_SOURCE='vlm_depth_pro'."
        )


def fetch_vlm_depth_cm(vlm_client: VLMClient) -> Tuple[float, bool]:
    """Fetch one (depth_cm, depth_valid) pair from vlm_service at the
    current gaze fixation. Converts the service's `depth_at_gaze_m`
    (Depth Pro is metric) to cm so the recorded NPZ field matches the
    vergence schema in scale; the source distinction is preserved via
    `meta["depth_source"]` and consumers must NOT mix sources.

    Fail-fast policy: any transport failure / not-ok response /
    malformed payload raises RuntimeError. The caller aborts the
    recording session rather than logging a synthetic vergence value.
    """
    try:
        resp = vlm_client.depth(at_gaze=True)
    except OSError as e:
        raise RuntimeError(f"VLM depth request failed (transport: {e})") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"VLM depth response malformed ({e})") from e
    if not isinstance(resp, dict) or not resp.get("ok", False):
        raise RuntimeError(f"VLM depth response not ok: {resp!r}")
    if "depth_at_gaze_m" not in resp:
        raise RuntimeError(
            f"VLM depth response missing depth_at_gaze_m field: {resp!r} "
            f"(did the request omit at_gaze=True somewhere?)"
        )
    depth_m = float(resp["depth_at_gaze_m"])
    if not np.isfinite(depth_m):
        # The pixel under gaze produced a non-finite depth (rare; e.g.
        # extreme background pixel). Mark invalid so the calibration
        # filter (D_valid in calibration_mapping) drops the row.
        return float("nan"), False
    return depth_m * 100.0, True


# =============================================================================
# Capture bundle
# =============================================================================
@dataclass
class CaptureBundle:
    """One sample. ``phase`` is one of ``'captured'``, ``'moving'``, or
    ``'transit'``. ``leg_label`` is set on ``'transit'`` bundles only
    (e.g. ``'transit_near_R2_to_near_R3'``); other phases leave it
    empty.

    REV01 (per Harmony_Gaze_Calibration_REV01_Plan.md §3.2):

    - ``depth_source`` names the depth pipeline that produced
      ``depth_cm``: ``"vergence"`` for transit / moving / vergence-mode
      anchors, ``"vlm_depth_pro"`` for VLM-substituted anchors. The
      offline affine-fit script may later rewrite transit rows to
      ``"vergence_affine"``; the recorder itself emits only the first
      two values.
    - ``depth_cm_vergence`` preserves the live vergence reading at VLM
      anchors so the offline affine fit can solve
      ``D_vlm = a · D_vergence + b`` without re-reading the snapshot
      JSON. ``nan`` on transit and moving rows (no parallel VLM
      reading) and on vergence-mode anchors (vergence is already in
      ``depth_cm``).
    """
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
    leg_label: str = ""
    depth_source: str = "vergence"
    depth_cm_vergence: float = float("nan")


def bundle_from_snapshot(snap: Dict[str, Any], robot_t: float,
                         q: np.ndarray, ee_mm: np.ndarray,
                         phase: str, target_label: str,
                         leg_label: str = "",
                         depth_source: str = "vergence",
                         depth_cm_vergence: float = float("nan")
                         ) -> CaptureBundle:
    """Pull every recorder-relevant field out of a gaze_runner snapshot
    and pair it with the most recent robot telemetry. Fields source:
    Utils/gaze/gaze_system.py:623-674 (snapshot contract).

    ``leg_label`` is set only on ``phase='transit'`` bundles and names
    the current transit leg (e.g. ``'transit_near_R2_to_near_R3'``).

    REV01 (Plan §3.2): ``depth_source`` and ``depth_cm_vergence`` are
    passed straight through to the returned ``CaptureBundle``. Callers
    set ``depth_source="vlm_depth_pro"`` for VLM-substituted anchors;
    transit and moving rows keep the ``"vergence"`` default.
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
        leg_label=leg_label,
        depth_source=depth_source,
        depth_cm_vergence=depth_cm_vergence,
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
                        robot_state: Dict[str, np.ndarray],
                        vlm_client: Optional[VLMClient] = None
                        ) -> Optional[CaptureBundle]:
    """Wait POST_CAPTURE_SETTLE_S after a `c` lock, then sample the
    gaze snapshot. Returns the bundle, or None if we cannot find a
    snapshot with DEPTH_VALID_MIN_CONSECUTIVE consecutive valid depth
    samples inside a bounded retry window.

    Phase 1 doc §8 follow-up #3: the depth smoother is an EMA, so the
    first sample after a worn-transition under-reports. The settle
    plus the consecutive-valid gate together protect against logging
    a stale value.

    When ``vlm_client`` is not None, the recorded bundle's `depth_cm`
    and `depth_valid` are overwritten with the VLM Depth Pro reading
    at the same fixation — depth_source must equal the runtime source
    or the Mahalanobis NN scale assumptions break. Any VLM failure
    bubbles up to abort the recording session (fail-fast per CLAUDE.md
    §"Error Handling"); we never silently fall back to vergence.
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

    # VLM-depth path overwrites the vergence depth fields BEFORE
    # bundle construction so the rest of the pipeline (writer,
    # diagnostics) sees a single coherent source per row. We do this
    # via a dict copy because the snapshot is the gaze_runner's owned
    # data structure.
    #
    # REV01 (Plan §3.3): when VLM substitutes, preserve the live
    # vergence reading on the bundle's ``depth_cm_vergence`` field so
    # the offline affine fit script can solve
    # ``D_vlm = a · D_vergence + b`` from the NPZ alone.
    if vlm_client is not None:
        snap_for_bundle = dict(last_snap)
        vergence_depth_cm = float(last_snap.get("depth_cm", float("nan")))
        depth_cm, depth_valid = fetch_vlm_depth_cm(vlm_client)
        snap_for_bundle["depth_cm"] = depth_cm
        snap_for_bundle["depth_valid"] = depth_valid
        print(f"[{_ts()}] VLM depth at gaze: {depth_cm:.1f}cm (valid={depth_valid})")
        return bundle_from_snapshot(
            snap_for_bundle, robot_state["_t"], robot_state["q"],
            robot_state["ee"], phase="captured",
            target_label=target_label,
            depth_source="vlm_depth_pro",
            depth_cm_vergence=vergence_depth_cm,
        )

    return bundle_from_snapshot(last_snap, robot_state["_t"], robot_state["q"],
                                robot_state["ee"], phase="captured",
                                target_label=target_label,
                                depth_source="vergence")


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
# Background workspace-telemetry thread
# =============================================================================
# Max consecutive snapshot-or-query failures before the telemetry thread
# escalates by setting its ``error_flag`` for the main loop to inspect.
# 5 ticks at 20 Hz = 0.25 s of dead telemetry, which is well above a
# single dropped UDP packet but well below an operator-perceptible stall.
TELEMETRY_MAX_CONSECUTIVE_FAILURES = 5


class TelemetryThread(threading.Thread):
    """Background ~20 Hz sampler that logs gaze + joint telemetry as
    ``phase='transit'`` bundles while the arm is free between captures.

    Lifecycle is driven from the main loop:

    - ``start_after_first_capture()`` flips an internal Event so the
      sampling loop begins only after the first waypoint's ``c`` ACK
      has landed; ticks before that are no-ops.
    - ``set_leg(label)`` is called each time the main loop advances to
      a new transit leg, BEFORE sending the next ``m``. The label is
      written into the bundle's ``leg_label`` field so downstream
      analysis can group samples by which transit they belong to.
    - ``pause()`` / ``resume()`` gate sampling around the ``c`` opcode
      lock and the POST_CAPTURE_SETTLE_S window so the depth /
      head-pose smoothers (Utils/gaze/gaze_system.py:419-560) converge
      before we sample again.
    - ``stop()`` joins cleanly; the main loop calls this before sending
      the final waypoint's ``c`` (so the home transition is not
      recorded) and on any error path.

    The thread fails-fast: TELEMETRY_MAX_CONSECUTIVE_FAILURES dead
    snapshot-or-query cycles in a row sets ``self.error_flag`` and
    halts the loop. The main loop polls ``error_flag`` between
    waypoints and decides whether to abort.
    """

    def __init__(self, link: "RobotLink", bundles: List[CaptureBundle],
                 bundles_lock: threading.Lock,
                 sample_hz: float = MOVING_PHASE_SAMPLE_HZ) -> None:
        super().__init__(name="harmony-telemetry", daemon=True)
        self._link = link
        self._bundles = bundles
        self._bundles_lock = bundles_lock
        self._period = 1.0 / float(sample_hz)
        self._stop_event = threading.Event()
        # _active gates the actual sampling — when clear, the loop
        # spins-with-sleep but does not query or append. The thread is
        # created with _active CLEARED so it sits idle until
        # ``start_after_first_capture()`` is called.
        self._active = threading.Event()
        self._leg_label: str = ""
        self._leg_lock = threading.Lock()
        self.error_flag: bool = False
        self.error_reason: str = ""
        self.consecutive_failures: int = 0

    # ------------------------------------------------------------------
    # Lifecycle control (called from main loop)
    # ------------------------------------------------------------------
    def start_after_first_capture(self) -> None:
        """Enable sampling. Idempotent."""
        self._active.set()

    def pause(self) -> None:
        """Halt sampling without joining the thread. Used around the
        ``c`` opcode lock and the settle window so the smoothers
        converge before the next transit sample lands."""
        self._active.clear()

    def resume(self) -> None:
        """Re-enable sampling after a pause()."""
        self._active.set()

    def stop(self) -> None:
        """Signal the loop to exit. Caller should ``join()`` after."""
        self._stop_event.set()
        self._active.set()  # unblock any wait
        self.join(timeout=2.0)

    def set_leg(self, leg_label: str) -> None:
        """Update the leg label written into subsequent transit bundles.
        Called by the main loop BEFORE sending the next ``m``."""
        with self._leg_lock:
            self._leg_label = leg_label

    def current_leg(self) -> str:
        with self._leg_lock:
            return self._leg_label

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------
    def run(self) -> None:  # noqa: D401 — Thread.run
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            next_tick += self._period
            if not self._active.is_set():
                # Idle tick. Reset the failure counter so a pause does
                # not accumulate dead ticks toward the escalation
                # threshold.
                self.consecutive_failures = 0
                self._sleep_until(next_tick)
                continue

            leg = self.current_leg()
            snap = gaze_snapshot(include_objects=False)
            rstate = self._link.query_state()
            if snap is None or rstate is None:
                self.consecutive_failures += 1
                print(f"[{_ts()}] TELEMETRY WARN: missing "
                      f"{'snapshot ' if snap is None else ''}"
                      f"{'robot-query ' if rstate is None else ''}"
                      f"(streak={self.consecutive_failures}, leg={leg!r})",
                      flush=True)
                if self.consecutive_failures >= TELEMETRY_MAX_CONSECUTIVE_FAILURES:
                    self.error_flag = True
                    self.error_reason = (
                        f"telemetry thread saw {self.consecutive_failures} "
                        f"consecutive snapshot/query failures on leg {leg!r}"
                    )
                    print(f"[{_ts()}] TELEMETRY ERROR: {self.error_reason}",
                          flush=True)
                    return
                self._sleep_until(next_tick)
                continue

            self.consecutive_failures = 0
            bundle = bundle_from_snapshot(
                snap, rstate["_t"], rstate["q"], rstate["ee"],
                phase="transit", target_label="", leg_label=leg,
                depth_source="vergence",
            )
            with self._bundles_lock:
                self._bundles.append(bundle)
            self._sleep_until(next_tick)

    def _sleep_until(self, deadline: float) -> None:
        # Use monotonic clock to keep cadence jitter-tolerant even
        # under load. If we are already behind, fall straight through.
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)


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
    print(f"[{_ts()}] >>> LOOK AT THE END EFFECTOR <<<")
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
      New (v2 only) — all samples (captured + moving + transit), indexed by phase:
        T_all, Q_all, X_all, G_all, D_cm_all, D_valid_all,
        Miss_mm_all, IPD_mm_all, IMU_w_all, IMU_fresh_all,
        Head_yaw_deg_all, Head_pitch_deg_all, Gaze_yaw_deg_all,
        Gaze_pitch_deg_all, Phase_all, Target_label_all, Leg_label_all
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
    Leg_label_all = np.asarray([b.leg_label for b in bundles], dtype="<U64")

    # depth_source pins the calibration-time depth pipeline; the v2
    # runtime dispatch in ExperimentDriver_Online_GazeTracking refuses
    # to load an NPZ whose runtime source disagrees, since mixing
    # vergence and Depth Pro within the Mahalanobis NN breaks the
    # learned per-feature scales.
    depth_source = str(getattr(config, "GAZE_CALIBRATION_DEPTH_SOURCE",
                                "vergence"))
    vlm_service_host = (str(getattr(config, "VLM_SERVICE_HOST", ""))
                        if depth_source == "vlm_depth_pro" else "")

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
        depth_source=depth_source,
        vlm_service_host=vlm_service_host,
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
        Leg_label_all=Leg_label_all,
        meta=meta,
    )


# =============================================================================
# Main session loop
# =============================================================================
# Duration baked into the auto-home ``h;dur=...`` opcode. 4.0 s
# matches the locked-in calibration-recorder home-wait budget; see
# Documents/SoftwareDocs/Harmony_Gaze_Calibration_Upgrade_Plan.md §6.1
# (recorder spec rework, 2026-05-19) for the rationale.
AUTO_HOME_DURATION_S = 4.0
# Small grace period beyond ``dur`` so the robot has time to settle and
# emit its terminal ACK before we close the UDP socket.
AUTO_HOME_GRACE_S = 0.5


def _send_auto_home(link: "RobotLink") -> None:
    """Send the home-with-duration opcode after the final capture and
    wait long enough for the motion to complete.

    Per CLAUDE.md realtime safety, this is fire-once; no operator
    confirmation prompt is interposed. The dispatcher prints a clear
    "stand clear" line before TX so the operator knows the arm is about
    to move on its own.
    """
    print()
    print(f"[{_ts()}] [done] Robot returning to home — please stand clear of "
          f"the workspace.")
    link.send_and_wait_ack(f"h;dur={AUTO_HOME_DURATION_S:.3f}",
                           expect_prefix="h")
    time.sleep(AUTO_HOME_DURATION_S + AUTO_HOME_GRACE_S)


def run_session(out_dir: str = ".") -> Optional[str]:
    """Run the full free-arm calibration session. Returns the output
    NPZ path on success, or None if the session was aborted.

    Bundle phases written into the NPZ:

    - ``'captured'`` — one per accepted waypoint (the locked-in
      reference sample used by the v2 mapping fit).
    - ``'moving'`` — pre-capture 1-second window per waypoint (legacy
      trajectory-history channel, retained for back-compat).
    - ``'transit'`` — continuous ~20 Hz background telemetry emitted
      by ``TelemetryThread`` while the arm is free between waypoints,
      labelled with ``leg_label='transit_<from>_to_<to>'``. Used for
      workspace-coverage analysis only; the v2 mapping fit ignores
      this phase.

    The home-to-first-waypoint and final-waypoint-to-home transitions
    are deliberately NOT recorded as transit: the telemetry thread
    starts after the first waypoint's ``c`` ACK and stops before the
    final waypoint's ``c`` is sent.
    """
    print()
    print("=" * 70)
    print("  HARMONY FREE-ARM GAZE CALIBRATION (Track B)")
    print("=" * 70)
    print(f"[{_ts()}] Active side: {ACTIVE_SIDE}")
    print(f"[{_ts()}] Robot endpoint: {ROBOT_IP}:{ROBOT_PORT}")
    print(f"[{_ts()}] Gaze service:  {GAZE_HOST}:{GAZE_PORT}")

    # Resolve depth-source preflight BEFORE binding the robot socket so
    # a misconfigured VLM host does not leave the robot link orphaned.
    # Fail-fast policy: the recorder refuses to start if vlm_depth_pro
    # is requested but the service is unreachable or has depth disabled.
    depth_source = str(getattr(config, "GAZE_CALIBRATION_DEPTH_SOURCE",
                                "vergence"))
    vlm_client: Optional[VLMClient] = None
    if depth_source == "vlm_depth_pro":
        vlm_client = VLMClient(config)
        verify_vlm_depth_available(vlm_client)
        print(f"[{_ts()}] [recorder] depth source: vlm_depth_pro "
              f"(host={vlm_client.host}, depth_enabled=True)")
    elif depth_source == "vergence":
        print(f"[{_ts()}] [recorder] depth source: vergence "
              f"(gaze_runner vergence path)")
    else:
        raise RuntimeError(
            f"Unknown GAZE_CALIBRATION_DEPTH_SOURCE={depth_source!r}; "
            f"must be 'vergence' or 'vlm_depth_pro'."
        )
    print()

    link = RobotLink()
    print(f"[{_ts()}] Bound UDP socket {link.sock.getsockname()} -> {link.robot_addr}")

    bundles: List[CaptureBundle] = []
    # Shared with the telemetry thread; ALL writes go through the lock.
    # The lock is cheap (one append per ~50 ms tick) and keeps the
    # NPZ writer's len(bundles) snapshot consistent.
    bundles_lock = threading.Lock()
    telemetry: Optional[TelemetryThread] = None

    def _check_telemetry_health() -> None:
        """Re-raise telemetry's failure as a session-aborting error.
        Called between waypoints (never inside a Tier-1 UDP round trip).
        """
        if telemetry is not None and telemetry.error_flag:
            raise RuntimeError(
                f"Telemetry thread aborted: {telemetry.error_reason}"
            )

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
        # 3-depth × 5-horizontal grid, then user-driven free additions.
        operator_targets = list(MANDATORY_GRID)
        print(f"[{_ts()}] Mandatory grid has {len(operator_targets)} points.")
        print(f"[{_ts()}] After the grid, you'll be prompted for optional free additions.")

        telemetry = TelemetryThread(link, bundles, bundles_lock)
        telemetry.start()
        # The thread is alive but inert — it sits in the "inactive"
        # branch of its loop until ``start_after_first_capture()`` is
        # called after the first waypoint's ``c`` ACK below.

        idx = 0
        total = len(operator_targets)
        last_captured_label: Optional[str] = None
        while idx < total:
            target_label = operator_targets[idx]
            announce_target(target_label, idx + 1, total)

            # The leg label is set BEFORE sending `m`: any transit
            # samples that land during the operator's reading time +
            # arm motion to ``target_label`` carry the right name.
            # Skipped only for idx==0, where the telemetry thread is
            # not yet active.
            if idx > 0 and last_captured_label is not None:
                telemetry.set_leg(f"transit_{last_captured_label}_to_{target_label}")
                telemetry.resume()

            # 1) Free the arm.
            if not free_arm(link):
                print(f"[{_ts()}] ERR: `m` rejected — likely robot is still in MOVING. Retrying after 1s.")
                time.sleep(1.0)
                _check_telemetry_health()
                continue

            # 2) Wait for the operator to position the arm. The prompt
            #    returns the line so the operator can type 's' to skip
            #    a target they cannot reach with the active arm.
            ans = prompt_user("Move the arm and fixate, then press Enter to capture (or 's' to skip)…")
            if ans.lower() == "s":
                print(f"[{_ts()}] Operator skipped {target_label}.")
                idx += 1
                _check_telemetry_health()
                continue

            # 3) Stream the arm motion as "moving" samples for ~1s prior
            #    to capture — gives the trajectory-based consumers a
            #    short pre-capture history without depending on the
            #    operator's reaction time. Pause the transit sampler
            #    while collect_moving_phase runs so we do not double-
            #    log the same telemetry under two different phase tags.
            telemetry.pause()
            moving = collect_moving_phase(link, target_label, duration_s=1.0)

            # 4) If this is the final waypoint, the telemetry thread
            #    must be stopped BEFORE we send `c` — the post-capture
            #    auto-home transition is explicitly excluded from the
            #    transit log per the recorder spec.
            is_final_waypoint = (idx == total - 1)
            if is_final_waypoint:
                telemetry.stop()
                telemetry = None

            # 5) Capture.
            captured_state = capture_pose(link)
            if captured_state is None:
                print(f"[{_ts()}] ERR: `c` failed at {target_label}. Retrying this target.")
                # If we just stopped the telemetry thread, re-create it so
                # a retry of the final waypoint still has transit coverage
                # on the way back in.
                if is_final_waypoint:
                    telemetry = TelemetryThread(link, bundles, bundles_lock)
                    telemetry.start()
                    telemetry.start_after_first_capture()
                continue

            # 6) Settle + sample. Telemetry stays paused.
            bundle = settle_and_snapshot(target_label, captured_state,
                                          vlm_client=vlm_client)
            if bundle is None:
                print(f"[{_ts()}] WARN: no gaze snapshot at {target_label}. Skipping.")
                idx += 1  # advance anyway — operator will fill via free-additions phase if needed
                _check_telemetry_health()
                continue

            with bundles_lock:
                bundles.extend(moving)
                bundles.append(bundle)
            print(f"[{_ts()}] Captured {target_label}: "
                  f"q={np.round(bundle.q, 3).tolist()}, "
                  f"depth={bundle.depth_cm:.1f}cm (valid={bundle.depth_valid}), "
                  f"head=({bundle.head_yaw_deg:+.1f},{bundle.head_pitch_deg:+.1f})")

            # First successful capture: activate the telemetry thread so
            # it begins sampling on the next iteration's "free" window.
            if idx == 0 and telemetry is not None:
                telemetry.start_after_first_capture()

            last_captured_label = target_label
            idx += 1
            _check_telemetry_health()

        # Free-additions phase.
        print()
        print("=" * 70)
        captured_count = sum(1 for b in bundles if b.phase == "captured")
        transit_count = sum(1 for b in bundles if b.phase == "transit")
        print(f"[{_ts()}] Mandatory grid complete ({len(bundles)} bundles, "
              f"{captured_count} captures, {transit_count} transit).")
        print(f"[{_ts()}] Optional free-additions phase: add reach hotspots.")
        print("=" * 70)

        # Telemetry coverage continues across the free-additions phase.
        # Restart the thread (the mandatory-grid loop stopped it before
        # the final `c`) so free-add transit legs are also recorded.
        telemetry = TelemetryThread(link, bundles, bundles_lock)
        telemetry.start()
        telemetry.start_after_first_capture()

        free_idx = 0
        prev_free_label = last_captured_label
        while True:
            free_idx += 1
            telemetry.pause()
            ans = prompt_user(f"Add free target #{free_idx}? Enter label (e.g. 'mug_handle'), or empty to finish: ")
            if not ans:
                break
            target_label = f"free_{ans}"
            announce_target(target_label, free_idx, free_idx)

            if prev_free_label is not None:
                telemetry.set_leg(f"transit_free_{prev_free_label}_to_{target_label}")
            telemetry.resume()

            if not free_arm(link):
                print(f"[{_ts()}] ERR: `m` rejected. Skipping.")
                continue
            prompt_user("Move the arm and fixate, then press Enter to capture…")
            telemetry.pause()
            moving = collect_moving_phase(link, target_label, duration_s=1.0)
            captured_state = capture_pose(link)
            if captured_state is None:
                print(f"[{_ts()}] ERR: `c` failed at {target_label}. Skipping.")
                continue
            bundle = settle_and_snapshot(target_label, captured_state,
                                          vlm_client=vlm_client)
            if bundle is None:
                print(f"[{_ts()}] WARN: no gaze snapshot at {target_label}. Skipping.")
                continue
            with bundles_lock:
                bundles.extend(moving)
                bundles.append(bundle)
            print(f"[{_ts()}] Captured {target_label}.")
            prev_free_label = target_label
            _check_telemetry_health()

        # Stop the free-additions telemetry thread before the auto-home
        # opcode goes out so the home transition is not recorded.
        if telemetry is not None:
            telemetry.stop()
            telemetry = None

        # Auto-home after the final accepted capture (no operator
        # confirmation prompt — per locked recorder spec).
        _send_auto_home(link)

        # Write the NPZ.
        if not bundles:
            print(f"[{_ts()}] No bundles captured. Aborting (no file written).")
            return None

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_name = f"poses_with_gaze_{stamp}_v2_freearm.npz"
        out_path = os.path.join(out_dir, out_name)
        write_npz(bundles, out_path)
        captured_n = sum(1 for b in bundles if b.phase == "captured")
        moving_n = sum(1 for b in bundles if b.phase == "moving")
        transit_n = sum(1 for b in bundles if b.phase == "transit")
        print(f"[{_ts()}] Wrote {out_path}")
        print(f"[{_ts()}]   captured samples: {captured_n}")
        print(f"[{_ts()}]   moving samples:   {moving_n}")
        print(f"[{_ts()}]   transit samples:  {transit_n}")
        return out_path

    finally:
        # Always reap the telemetry thread before closing the socket so
        # a stray query_state() does not fire after RobotLink.close().
        if telemetry is not None:
            telemetry.stop()
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
