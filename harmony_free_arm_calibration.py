#!/usr/bin/env python3
"""
harmony_free_arm_calibration.py — Interactive free-arm gaze-calibration
recorder. Operator-paced; each phase transition is gated on an explicit
Enter so the operator can read the terminal between captures without
poisoning the data.

Data flow (vlm-only operation):

- Gaze (x, y), worn, and IMU come from ``Utils.frame_relay`` on Linux
  loopback (the panel's embedded relay or a standalone
  ``python -m Utils.frame_relay`` — only one Neon SDK subscription can
  exist at a time). ``Utils.remote_frame_reader.RemoteFrameReader``
  yields ``FrameBundleStub`` objects; the recorder maintains the
  latest bundle in a background ``RelayBundleConsumer`` thread.
- Depth (cm) at calibration anchors comes from the VLM Depth Pro
  service on Windows over UDP (``Utils.perception_clients.VLMClient``).
  Transit rows leave ``depth_cm = NaN``; the offline
  ``tools/fit_depth_interpolation.py`` fills them post-hoc with a
  leg-aware bracketed linear interpolation from the anchor depths.
- Gaze yaw/pitch (degrees) are computed locally from
  ``(gaze_px, camera_matrix)`` via a standard pixel-unprojection,
  giving camera-frame angles. Head yaw/pitch are NaN (Pass-2 features
  remain off by default; the IMU pipeline can be added back later).
- ``GAZE_CALIBRATION_DEPTH_SOURCE`` MUST be ``"vlm_depth_pro"``; the
  recorder no longer supports the gaze_runner vergence path.

Per-waypoint cycle (steady state, WP2+):

1. [telemetry paused] Prompt: "Ready to move? Enter to unlock + begin
   transit recording."
2. ↓ Enter → send ``m``, ``telemetry.resume()`` (transit recording on).
3. Operator hand-guides arm to next position while fixating EE.
4. ↓ Enter → ``telemetry.pause()``, send ``c`` (arm locks), stash
   (q, ee) from the C++ telemetry payload.
5. [telemetry paused] Prompt: "Locked. Enter when fixating EE to
   record anchor."
6. ↓ Enter → depth_valid streak poll + (if VLM) Depth Pro at gaze →
   append anchor bundle.

WP1 is special: no transit recording on the home→WP1 leg, so the cycle
has 2 Enter presses instead of 3 (skip step 1, telemetry stays inert).

Waypoints are anonymous (``wp01``, ``wp02``, ...). The operator decides
how many to record; Ctrl+C at any prompt exits cleanly — the finally
block locks the arm if currently free, sends auto-home, and writes the
NPZ with whatever bundles were collected.

Wire interfaces (unchanged from prior design):

- Robot research interface running ``Gaze_Tracking`` with the
  ``m``/``c``/``h;dur=`` opcodes. UDP socket bound to ``0.0.0.0:8080``,
  dial ``192.168.2.1:8080``.
- ``gaze_runner.py`` in ``--mode service`` on
  ``config.GAZE_UDP_IP:GAZE_UDP_PORT``. Snapshots carry gaze, depth,
  IMU, and head pose.

Output: ``poses_with_gaze_<UTC>_v2_freearm.npz`` with the v2/REV01
schema — captured anchors and transit telemetry both contribute to the
fit-relevant block; ``meta["depth_source"]`` records the depth pipeline
(``"vergence"`` or ``"hybrid_anchor_vlm_transit_vergence"`` per
``config.GAZE_CALIBRATION_DEPTH_SOURCE``).
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
from Utils.perception_clients import VLMClient
from Utils.remote_frame_reader import RemoteFrameReader


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

# Frame relay endpoint — the recorder dials this as a second TCP
# consumer alongside vlm_service (on Windows) and any panel-embedded
# relay (loopback). config.FRAME_RELAY_DIAL_HOST is the host clients
# dial; on Linux this defaults to loopback because the panel's
# embedded relay binds 0.0.0.0:FRAME_RELAY_PORT.
RELAY_HOST = str(getattr(config, "FRAME_RELAY_DIAL_HOST", "127.0.0.1"))
RELAY_PORT = int(getattr(config, "FRAME_RELAY_PORT", 5591))
# Wait this many seconds for the first relay handshake before declaring
# the relay unreachable. The panel's embedded relay handshakes within
# ~100 ms of the first frame; 5 s is generous.
RELAY_HANDSHAKE_TIMEOUT_S = 5.0


# =============================================================================
# Recorder configuration
# =============================================================================
ACTIVE_SIDE = os.getenv("HARMONY_ACTIVE_SIDE", "R").upper()
if ACTIVE_SIDE not in ("L", "R"):
    raise ValueError(f"HARMONY_ACTIVE_SIDE must be 'L' or 'R'; got {ACTIVE_SIDE!r}")

# Reject a relay bundle older than this when sampling for snapshots
# (transit telemetry + anchor capture poll). The panel relay publishes
# at ~10 Hz so a bundle older than 250 ms means the relay has stalled.
MAX_BUNDLE_AGE_S = 0.25

# How many consecutive valid (worn=True, finite gaze) bundles we need
# before we accept an anchor capture. Guards against logging a sample
# during a blink or an unworn moment. The new interactive flow has no
# implicit settle — the operator paces the "ready for anchor?" gate
# manually — but the worn streak still protects the anchor sample.
WORN_STREAK_MIN_CONSECUTIVE = 5
# Legacy alias kept so the NPZ meta and any external consumers keep
# seeing the same key.
DEPTH_VALID_MIN_CONSECUTIVE = WORN_STREAK_MIN_CONSECUTIVE

# Sample rate for the background transit-telemetry thread and the
# poll cadence inside the anchor depth-valid wait. 20 Hz mirrors the
# gaze_runner internal loop cap (gaze_runner.py:496 target_loop_hz=20.0);
# no point asking faster.
SAMPLE_HZ = 20.0


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


# Module-level relay consumer, initialised at the top of run_session.
# gaze_snapshot() reads from this; module-level state keeps the
# call sites in TelemetryThread and capture_anchor unchanged in shape.
_RELAY_CONSUMER: Optional["RelayBundleConsumer"] = None


def _gaze_px_to_yaw_pitch_deg(gx_px: float, gy_px: float,
                                K: np.ndarray) -> Tuple[float, float]:
    """Unproject a gaze pixel through the camera matrix to a ray in the
    camera frame, then convert to (yaw, pitch) in degrees. Yaw is
    rotation about Y (positive right of the optical axis); pitch is
    rotation about X (positive up). Camera optical axis is +Z.

    Yields camera-frame angles, NOT world-frame — gaze_system.py's
    ``gaze_yaw_deg`` is world-frame (IMU-rotated). The recorder writes
    these camera-frame angles into the NPZ; the runtime consumer must
    compute features with the same convention for the v2 Mahalanobis
    NN to align.
    """
    if not (np.isfinite(gx_px) and np.isfinite(gy_px)):
        return float("nan"), float("nan")
    try:
        ray = np.linalg.inv(K) @ np.array([gx_px, gy_px, 1.0], dtype=np.float64)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    norm = float(np.linalg.norm(ray))
    if norm <= 0.0 or not np.isfinite(norm):
        return float("nan"), float("nan")
    ray = ray / norm
    yaw = float(np.degrees(np.arctan2(ray[0], ray[2])))
    pitch = float(np.degrees(np.arctan2(-ray[1], np.sqrt(ray[0] ** 2 + ray[2] ** 2))))
    return yaw, pitch


def _make_snapshot_from_bundle(bundle: Any, camera_matrix: np.ndarray,
                                 t_recv: float) -> Optional[Dict[str, Any]]:
    """Build a snapshot dict in the same shape ``bundle_from_snapshot``
    expects, sourced from a frame_relay ``FrameBundleStub``. Returns
    None if the gaze sample in the bundle is non-finite (the relay
    forwards NaN gaze during blinks).
    """
    try:
        gx_px = float(bundle.gaze.x)
        gy_px = float(bundle.gaze.y)
    except (AttributeError, TypeError, ValueError):
        return None
    if not (np.isfinite(gx_px) and np.isfinite(gy_px)):
        return None
    yaw_deg, pitch_deg = _gaze_px_to_yaw_pitch_deg(gx_px, gy_px, camera_matrix)
    return {
        "ok": True,
        "gaze_px": (gx_px, gy_px),
        "worn": bool(getattr(bundle, "worn", True)),
        # Depth is filled per-anchor by fetch_vlm_depth_cm; transit
        # rows leave this NaN and are filled offline by
        # tools/fit_depth_interpolation.py.
        "depth_cm": float("nan"),
        "depth_valid": False,
        "miss_mm": float("nan"),
        "ipd_mm": float("nan"),
        "imu_angvel": float("nan"),
        "imu_fresh": bool(getattr(bundle, "imu", None) is not None),
        "head_yaw_deg": float("nan"),
        "head_pitch_deg": float("nan"),
        "gaze_yaw_deg": yaw_deg,
        "gaze_pitch_deg": pitch_deg,
        "t_recv": t_recv,
    }


class RelayBundleConsumer(threading.Thread):
    """Background TCP consumer of the frame_relay envelope stream. Wraps
    ``Utils.remote_frame_reader.RemoteFrameReader`` and maintains the
    most recent ``FrameBundleStub`` plus the camera_matrix from the
    handshake. Thread-safe ``latest()`` accessor returns the bundle and
    its receive timestamp; callers gate on
    ``MAX_BUNDLE_AGE_S`` to avoid logging stale state.

    The recorder requires a working relay before any waypoint is
    accepted — ``run_session`` calls ``wait_for_first_bundle()`` after
    starting the thread and aborts if the timeout elapses.
    """

    def __init__(self, dial_host: str, dial_port: int,
                 handshake_timeout_s: float = RELAY_HANDSHAKE_TIMEOUT_S) -> None:
        super().__init__(name="recorder-relay-consumer", daemon=True)
        self._dial_host = dial_host
        self._dial_port = dial_port
        self._handshake_timeout_s = handshake_timeout_s
        self._reader: Optional[RemoteFrameReader] = None
        self._latest: Optional[Any] = None
        self._latest_t_recv: float = 0.0
        self._latest_lock = threading.Lock()
        self._first_bundle = threading.Event()
        self._stop_event = threading.Event()
        self.error_flag: bool = False
        self.error_reason: str = ""

    def run(self) -> None:  # noqa: D401 — Thread.run
        try:
            self._reader = RemoteFrameReader(
                host=self._dial_host,
                port=self._dial_port,
                wait_for_handshake_s=self._handshake_timeout_s,
                auto_reconnect=True,
            )
        except Exception as e:  # noqa: BLE001 — surface any setup failure
            self.error_flag = True
            self.error_reason = f"relay dial failed: {e}"
            self._first_bundle.set()  # unblock waiters; they should re-check error_flag
            return
        try:
            for bundle in self._reader:
                if self._stop_event.is_set():
                    break
                with self._latest_lock:
                    self._latest = bundle
                    self._latest_t_recv = time.time()
                if not self._first_bundle.is_set():
                    self._first_bundle.set()
        except Exception as e:  # noqa: BLE001
            self.error_flag = True
            self.error_reason = f"relay reader raised: {e}"
            self._first_bundle.set()

    def latest(self) -> Tuple[Optional[Any], float]:
        with self._latest_lock:
            return self._latest, self._latest_t_recv

    @property
    def camera_matrix(self) -> np.ndarray:
        if self._reader is None:
            # Pre-handshake fallback — RemoteFrameReader returns the
            # Neon defaults when its handshake hasn't landed yet.
            return np.array(
                [[1490.0, 0.0, 800.0],
                 [0.0, 1490.0, 600.0],
                 [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        return self._reader.camera_matrix

    def wait_for_first_bundle(self, timeout_s: float) -> bool:
        """Block until the first envelope arrives or the timeout
        elapses. Returns True on first-bundle, False on timeout.
        """
        return self._first_bundle.wait(timeout=timeout_s)

    def stop(self) -> None:
        self._stop_event.set()
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:  # noqa: BLE001
                pass
        self.join(timeout=2.0)


def gaze_snapshot(include_objects: bool = False) -> Optional[Dict[str, Any]]:
    """Return a snapshot dict built from the most recent frame_relay
    bundle, or None if the bundle is missing / too stale / has a NaN
    gaze sample. ``include_objects`` is accepted for legacy API
    compatibility but ignored — the recorder never consumed object
    detections.

    Module-level ``_RELAY_CONSUMER`` must be initialised by
    ``run_session`` before any call (the TelemetryThread and
    ``capture_anchor`` both invoke this).
    """
    if _RELAY_CONSUMER is None:
        return None
    bundle, t_recv = _RELAY_CONSUMER.latest()
    if bundle is None:
        return None
    age = time.time() - t_recv
    if age > MAX_BUNDLE_AGE_S:
        return None
    return _make_snapshot_from_bundle(bundle, _RELAY_CONSUMER.camera_matrix, t_recv)


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


def capture_anchor(target_label: str,
                   robot_state: Dict[str, np.ndarray],
                   vlm_client: VLMClient
                   ) -> Optional[CaptureBundle]:
    """Sample a snapshot at the current locked pose and build the
    anchor bundle. Operator paces fixation via the "Ready for anchor?"
    Enter gate; the worn-streak gate (below) protects against logging
    a sample mid-blink or while the headset is briefly unworn.

    Returns the bundle, or None if no valid snapshot was obtained at
    all inside the bounded retry window.

    Anchor depth is fetched from VLM Depth Pro at the current gaze
    pixel; ``depth_source="vlm_depth_pro"`` is pinned on the row. The
    VLM call is mandatory in this build — the vergence pipeline is no
    longer running.
    """
    consecutive_worn = 0
    last_snap: Optional[Dict[str, Any]] = None
    poll_dt = 1.0 / SAMPLE_HZ
    deadline = time.time() + 2.0  # bounded retry; never spin forever
    while time.time() < deadline:
        snap = gaze_snapshot()
        if snap is None:
            consecutive_worn = 0
            time.sleep(poll_dt)
            continue
        last_snap = snap
        if bool(snap.get("worn", False)):
            consecutive_worn += 1
            if consecutive_worn >= WORN_STREAK_MIN_CONSECUTIVE:
                break
        else:
            consecutive_worn = 0
        time.sleep(poll_dt)

    if last_snap is None:
        print(f"[{_ts()}] No relay bundle available for capture {target_label} "
              f"(relay stalled, gaze NaN, or first handshake not yet landed).")
        return None

    if consecutive_worn < WORN_STREAK_MIN_CONSECUTIVE:
        print(f"[{_ts()}] WARN: worn streak {consecutive_worn} "
              f"< required {WORN_STREAK_MIN_CONSECUTIVE} — recording bundle "
              f"anyway, but verify the headset is seated correctly.")

    # Anchor depth: fetch from VLM Depth Pro at the current gaze and
    # pin the row's depth_source to "vlm_depth_pro". The recorder no
    # longer has access to vergence; depth_cm_vergence stays NaN.
    snap_for_bundle = dict(last_snap)
    depth_cm, depth_valid = fetch_vlm_depth_cm(vlm_client)
    snap_for_bundle["depth_cm"] = depth_cm
    snap_for_bundle["depth_valid"] = depth_valid
    print(f"[{_ts()}] VLM depth at gaze: {depth_cm:.1f}cm (valid={depth_valid})")
    return bundle_from_snapshot(
        snap_for_bundle, robot_state["_t"], robot_state["q"],
        robot_state["ee"], phase="captured",
        target_label=target_label,
        depth_source="vlm_depth_pro",
    )


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
    - ``pause()`` / ``resume()`` gate sampling around each ``c`` opcode
      lock and the "Ready for anchor?" Enter gate so the depth /
      head-pose smoothers (Utils/gaze/gaze_system.py:419-560) converge
      between transit legs and so the operator can read the terminal
      without polluting transit data.
    - ``stop()`` joins cleanly; called from the run_session finally
      block on session exit (clean or Ctrl+C).

    The thread fails-fast: TELEMETRY_MAX_CONSECUTIVE_FAILURES dead
    snapshot-or-query cycles in a row sets ``self.error_flag`` and
    halts the loop. The main loop polls ``error_flag`` between
    waypoints and decides whether to abort.
    """

    def __init__(self, link: "RobotLink", bundles: List[CaptureBundle],
                 bundles_lock: threading.Lock,
                 sample_hz: float = SAMPLE_HZ) -> None:
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
            # Transit rows carry no depth at recording time — the
            # offline tools/fit_depth_interpolation.py post-fills
            # depth_cm from the bracketing anchor depths and rewrites
            # depth_source to its interpolation tag.
            bundle = bundle_from_snapshot(
                snap, rstate["_t"], rstate["q"], rstate["ee"],
                phase="transit", target_label="", leg_label=leg,
                depth_source="pending_interpolation",
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


def announce_waypoint(label: str, hint: str = "") -> None:
    print()
    print("=" * 70)
    print(f"[{_ts()}] {label}{('  ' + hint) if hint else ''}")
    print(f"[{_ts()}] >>> LOOK AT THE END EFFECTOR <<<")
    print("=" * 70)


# =============================================================================
# Output writer — v2 NPZ schema (plan §6.1 Track A step 2; Track B reuses)
# =============================================================================
def write_npz(bundles: List[CaptureBundle], out_path: str) -> None:
    """Materialise CaptureBundle list as v2 NPZ.

    REV01 (Plan §3.2): the legacy block (``T/Q/X/G/D_cm/...``) now
    contains captured + transit rows so the v2 Mahalanobis NN sees the
    full transit cloud as training data. Moving-phase bundles stay out
    of the legacy block (they would be a redundant subset of transit)
    and live under the ``*_all`` keys only. Two new per-row columns:
    ``Depth_source`` (per fit-row provenance,
    ``"vlm_depth_pro" | "vergence" | "vergence_affine"``) and
    ``D_cm_vergence`` (parallel vergence reading at VLM anchors; NaN
    on transit). The offline ``tools/fit_vergence_affine.py`` script
    rewrites transit rows to ``"vergence_affine"`` and fills
    ``meta["affine_map"]`` post-hoc.

    Output contract (REV01):
      Legacy (v1-compatible) — captured + transit fit rows:
        T: (N,) float64           timestamps (s)
        Q: (N, 7) float64         joint angles (rad)
        X: (N, 3) float64         EE positions (mm)
        G: (N, 3) float64         (gaze_x_norm, gaze_y_norm, gaze_conf)
      New (v2 only) — captured + transit fit rows:
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
        Depth_source   (N,) <U64
        D_cm_vergence  (N,) float64   NaN on transit rows
      New (v2 only) — all samples (captured + moving + transit), indexed by phase:
        T_all, Q_all, X_all, G_all, D_cm_all, D_valid_all,
        Miss_mm_all, IPD_mm_all, IMU_w_all, IMU_fresh_all,
        Head_yaw_deg_all, Head_pitch_deg_all, Gaze_yaw_deg_all,
        Gaze_pitch_deg_all, Phase_all, Target_label_all, Leg_label_all,
        Depth_source_all
      meta: dict with ``version``, ``side``, ``depth_source``,
        ``affine_map`` (placeholder ``None``; filled by the offline
        affine-fit script), and recorder identifiers.
    """
    captured = [b for b in bundles if b.phase == "captured"]
    if not captured:
        raise RuntimeError("Refusing to write NPZ with zero captured samples.")

    # REV01 (Plan §3.2 item 1): the fit-relevant block now includes
    # transit ticks alongside the captured anchors. Moving rows stay
    # out (they are a ~1 s pre-capture trajectory tail, redundant with
    # the surrounding transit cloud).
    fit_rows = [b for b in bundles if b.phase in ("captured", "transit")]

    def stack_field(items, attr, dtype=float):
        return np.asarray([getattr(b, attr) for b in items], dtype=dtype)

    def vstack_field(items, attr):
        return np.vstack([getattr(b, attr) for b in items])

    T = stack_field(fit_rows, "t")
    Q = vstack_field(fit_rows, "q")
    X = vstack_field(fit_rows, "ee_mm")
    G = np.column_stack([
        stack_field(fit_rows, "gaze_x_norm"),
        stack_field(fit_rows, "gaze_y_norm"),
        stack_field(fit_rows, "gaze_conf"),
    ])

    D_cm = stack_field(fit_rows, "depth_cm")
    D_valid = stack_field(fit_rows, "depth_valid", dtype=bool)
    Miss_mm = stack_field(fit_rows, "miss_mm")
    IPD_mm = stack_field(fit_rows, "ipd_mm")
    IMU_w = stack_field(fit_rows, "imu_w")
    IMU_fresh = stack_field(fit_rows, "imu_fresh", dtype=bool)
    Head_yaw_deg = stack_field(fit_rows, "head_yaw_deg")
    Head_pitch_deg = stack_field(fit_rows, "head_pitch_deg")
    Gaze_yaw_deg = stack_field(fit_rows, "gaze_yaw_deg")
    Gaze_pitch_deg = stack_field(fit_rows, "gaze_pitch_deg")
    Target_label = np.asarray([b.target_label for b in fit_rows], dtype="<U32")
    # Per-fit-row leg label so tools/fit_depth_interpolation.py can
    # identify each transit row's bracketing anchors. Captured rows
    # leave this empty (TelemetryThread is the only producer of
    # non-empty leg_label values).
    Leg_label = np.asarray([b.leg_label for b in fit_rows], dtype="<U64")

    # REV01 (Plan §3.2 items 2, 3): per-row provenance + the parallel
    # vergence reading preserved at VLM anchors (NaN on transit). <U64
    # over-provisions for the longest expected value
    # "vlm_interpolated_nearest_anchor" (32 chars) without truncation.
    Depth_source = np.asarray([b.depth_source for b in fit_rows], dtype="<U64")
    D_cm_vergence = stack_field(fit_rows, "depth_cm_vergence")

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
    Depth_source_all = np.asarray([b.depth_source for b in bundles],
                                    dtype="<U64")

    # depth_source is always "vlm_depth_pro" in this build — anchors
    # come from VLM Depth Pro and transit rows get bracketed-interp
    # depths from tools/fit_depth_interpolation.py post-hoc. The
    # runtime then dispatches to its per-query VLM Depth Pro path
    # (Utils/gaze/calibration_mapping.py runtime_depth_pipeline).
    vlm_service_host = str(getattr(config, "VLM_SERVICE_HOST", ""))

    meta = dict(
        version=2,
        side=ACTIVE_SIDE,
        recorder="harmony_free_arm_calibration.py",
        wire_protocol="HARMONY-UNIT-4 tools/wire_protocol.md (m/c rows)",
        gaze_sample_width=float(getattr(config, "GAZE_SAMPLE_WIDTH", 1600.0)),
        gaze_sample_height=float(getattr(config, "GAZE_SAMPLE_HEIGHT", 1200.0)),
        depth_valid_min_consecutive=DEPTH_VALID_MIN_CONSECUTIVE,
        depth_source="vlm_depth_pro",
        vlm_service_host=vlm_service_host,
        # Legacy field — preserved for back-compat with REV01 readers
        # that probe ``affine_map``. The new flow does not fit a
        # vergence-to-Depth-Pro affine; the depth source is
        # vlm_depth_pro all the way through.
        affine_map=None,
        units=dict(X="mm", Q="rad", G="normalized_0_to_1",
                   D_cm="cm", Miss_mm="mm", IPD_mm="mm",
                   IMU_w="rad/s", Head_yaw_deg="deg",
                   Head_pitch_deg="deg", Gaze_yaw_deg="deg",
                   Gaze_pitch_deg="deg"),
    )

    np.savez_compressed(
        out_path,
        # legacy v1 keys (REV01: now captured + transit, not captured-only)
        T=T, Q=Q, X=X, G=G,
        # v2 fit-block (captured + transit) keys
        D_cm=D_cm, D_valid=D_valid, Miss_mm=Miss_mm, IPD_mm=IPD_mm,
        IMU_w=IMU_w, IMU_fresh=IMU_fresh,
        Head_yaw_deg=Head_yaw_deg, Head_pitch_deg=Head_pitch_deg,
        Gaze_yaw_deg=Gaze_yaw_deg, Gaze_pitch_deg=Gaze_pitch_deg,
        Target_label=Target_label,
        # Per-fit-row leg label for transit rows so the offline
        # interpolation tool can find each row's bracketing anchors.
        Leg_label=Leg_label,
        # REV01 (Plan §3.2 items 2, 3): per-row provenance and the
        # parallel vergence reading on anchor rows.
        Depth_source=Depth_source, D_cm_vergence=D_cm_vergence,
        # v2 all-phase keys
        T_all=T_all, Q_all=Q_all, X_all=X_all, G_all=G_all,
        D_cm_all=D_cm_all, D_valid_all=D_valid_all,
        Miss_mm_all=Miss_mm_all, IPD_mm_all=IPD_mm_all,
        IMU_w_all=IMU_w_all, IMU_fresh_all=IMU_fresh_all,
        Head_yaw_deg_all=Head_yaw_deg_all, Head_pitch_deg_all=Head_pitch_deg_all,
        Gaze_yaw_deg_all=Gaze_yaw_deg_all, Gaze_pitch_deg_all=Gaze_pitch_deg_all,
        Phase_all=Phase_all, Target_label_all=Target_label_all,
        Leg_label_all=Leg_label_all,
        # REV01 (Plan §3.2 item 2): every-row provenance for the
        # *_all block, padded to "vergence" for phase="moving" rows
        # because moving samples never receive VLM substitution.
        Depth_source_all=Depth_source_all,
        meta=meta,
    )


# =============================================================================
# Main session loop
# =============================================================================
# Duration baked into the auto-home ``h;dur=...`` opcode. 4.0 s
# matches the locked-in calibration-recorder home-wait budget; see
# Documents/SoftwareDocs/Harmony_Gaze_Calibration_REV00_Plan.md §6.1
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
    """Run the interactive free-arm calibration session. Returns the
    output NPZ path on success (≥1 anchor captured), or None if the
    session ended with no anchors collected.

    The session runs forever until Ctrl+C; each waypoint is operator-
    paced via explicit Enter gates. On any exit path (clean Ctrl+C or
    unexpected error), the finally block stops telemetry, locks the
    arm if it is currently free, sends ``h;dur=4.0`` for auto-home,
    closes the UDP socket, and writes the NPZ with whatever anchors
    were collected.

    Bundle phases written into the NPZ:

    - ``'captured'`` — one per accepted waypoint anchor.
    - ``'transit'`` — continuous ~20 Hz background samples emitted by
      ``TelemetryThread`` while the arm is in master_free between
      waypoints, labelled with
      ``leg_label='transit_wp<N>_to_wp<N+1>'``. v2 fit consumes these
      alongside the anchors (REV01 §3.2 #1).

    The home→WP1 leg is NOT recorded (telemetry stays inert until the
    operator triggers the WP2 transit). The terminal auto-home is sent
    from the finally block after telemetry has stopped, so it never
    appears as transit.
    """
    global _RELAY_CONSUMER

    print()
    print("=" * 70)
    print("  HARMONY INTERACTIVE FREE-ARM GAZE CALIBRATION")
    print("=" * 70)
    print(f"[{_ts()}] Active side: {ACTIVE_SIDE}")
    print(f"[{_ts()}] Robot endpoint: {ROBOT_IP}:{ROBOT_PORT}")
    print(f"[{_ts()}] Frame relay:    {RELAY_HOST}:{RELAY_PORT}")

    # Preflight (runs before any robot socket is bound, so a failure
    # here cannot leave hardware in an unsafe state).
    depth_source = str(getattr(config, "GAZE_CALIBRATION_DEPTH_SOURCE",
                                "vlm_depth_pro"))
    if depth_source != "vlm_depth_pro":
        raise RuntimeError(
            f"GAZE_CALIBRATION_DEPTH_SOURCE={depth_source!r} is not supported "
            f"in this build. Set it to 'vlm_depth_pro' in config_local.py — "
            f"the vergence pipeline (gaze_runner) is no longer consumed by "
            f"the recorder; depth at anchors comes from VLM Depth Pro and "
            f"transit-row depths are post-filled by "
            f"tools/fit_depth_interpolation.py."
        )
    vlm_client = VLMClient(config)
    verify_vlm_depth_available(vlm_client)
    print(f"[{_ts()}] [recorder] depth source: vlm_depth_pro "
          f"(host={vlm_client.host}, depth_enabled=True)")
    print()

    link = RobotLink()
    print(f"[{_ts()}] Bound UDP socket {link.sock.getsockname()} -> {link.robot_addr}")

    # Start the relay consumer BEFORE anything else — the recorder
    # needs at least one bundle to compute gaze features. Failure here
    # raises so the operator sees a precise reason without bothering
    # with prompts.
    _RELAY_CONSUMER = RelayBundleConsumer(RELAY_HOST, RELAY_PORT)
    _RELAY_CONSUMER.start()
    if not _RELAY_CONSUMER.wait_for_first_bundle(timeout_s=RELAY_HANDSHAKE_TIMEOUT_S + 1.0):
        _RELAY_CONSUMER.stop()
        _RELAY_CONSUMER = None
        link.close()
        raise RuntimeError(
            f"No frame_relay bundle arrived at {RELAY_HOST}:{RELAY_PORT} "
            f"within {RELAY_HANDSHAKE_TIMEOUT_S:.1f}s. Start the control "
            f"panel (which embeds the relay) or `python -m Utils.frame_relay "
            f"--bind 0.0.0.0 --port {RELAY_PORT}` first."
        )
    if _RELAY_CONSUMER.error_flag:
        msg = _RELAY_CONSUMER.error_reason
        _RELAY_CONSUMER.stop()
        _RELAY_CONSUMER = None
        link.close()
        raise RuntimeError(f"Relay consumer error: {msg}")
    print(f"[{_ts()}] Relay handshake landed; first bundle received.")

    bundles: List[CaptureBundle] = []
    bundles_lock = threading.Lock()
    telemetry: Optional[TelemetryThread] = None
    # Tracks whether the arm is currently in master_free. The finally
    # block uses this to decide whether to send a safety `c` before
    # auto-home so the home motion starts from a locked state.
    arm_is_free = False
    out_path: Optional[str] = None
    last_label: Optional[str] = None

    def _check_telemetry_health() -> None:
        """Re-raise telemetry failure as a session-ending error so the
        finally block saves whatever has been collected."""
        if telemetry is not None and telemetry.error_flag:
            raise RuntimeError(
                f"Telemetry thread aborted: {telemetry.error_reason}"
            )

    try:
        # Head-pose recenter is not available in vlm-only mode (the
        # gaze_runner pipeline that owned the recenter RPC is not
        # running). Pass-2 features remain off by default; the
        # participant's seated posture at session start is the de
        # facto neutral for Pass-1.
        print(f"[{_ts()}] Head-pose recenter unavailable in vlm-only mode; "
              f"Pass-1 features are camera-frame and do not require it.")

        telemetry = TelemetryThread(link, bundles, bundles_lock)
        telemetry.start()  # alive but inert until the first resume()

        # ──────────────────────────────────────────────────────────────
        # WP1 — no transit recording on the home→WP1 leg.
        # ──────────────────────────────────────────────────────────────
        announce_waypoint("WP01", hint="(first waypoint — no transit on this leg)")
        prompt_user("Press Enter to unlock the arm. "
                    "Then move it to the first position while fixating the EE, "
                    "and press Enter again to lock.")
        if not free_arm(link):
            raise RuntimeError("Initial `m` rejected — robot may be in MOVING state. Aborting.")
        arm_is_free = True
        prompt_user("Arm is free. Move to WP01 while fixating EE, "
                    "then press Enter to lock.")
        captured_state = capture_pose(link)
        if captured_state is None:
            raise RuntimeError("Initial `c` failed. Aborting.")
        arm_is_free = False
        prompt_user("Locked. When your eyes are fixated on the EE, "
                    "press Enter to record the anchor.")
        bundle = capture_anchor("wp01", captured_state, vlm_client=vlm_client)
        if bundle is None:
            print(f"[{_ts()}] WARN: no gaze snapshot at wp01 — anchor dropped.")
        else:
            with bundles_lock:
                bundles.append(bundle)
            print(f"[{_ts()}] Captured wp01: "
                  f"depth={bundle.depth_cm:.1f}cm (valid={bundle.depth_valid}), "
                  f"head=({bundle.head_yaw_deg:+.1f},{bundle.head_pitch_deg:+.1f})")
            last_label = "wp01"

        # ──────────────────────────────────────────────────────────────
        # WP2+ steady-state loop. Ctrl+C exits via the finally block,
        # which saves whatever anchors and transit data have landed.
        # ──────────────────────────────────────────────────────────────
        wp_idx = 2
        while True:
            label = f"wp{wp_idx:02d}"
            announce_waypoint(label, hint="(Ctrl+C any time to finish and save)")

            from_label = last_label if last_label is not None else "start"
            prompt_user(f"Press Enter to unlock the arm and begin "
                        f"{from_label}→{label} transit recording.")
            telemetry.set_leg(f"transit_{from_label}_to_{label}")
            telemetry.resume()
            if not free_arm(link):
                telemetry.pause()
                print(f"[{_ts()}] ERR: `m` rejected. Skipping {label}.")
                wp_idx += 1
                _check_telemetry_health()
                continue
            arm_is_free = True

            prompt_user(f"Arm free + transit recording. Move to {label} while "
                        f"fixating EE, then press Enter to lock.")
            telemetry.pause()
            captured_state = capture_pose(link)
            if captured_state is None:
                print(f"[{_ts()}] ERR: `c` failed at {label}. Skipping.")
                wp_idx += 1
                _check_telemetry_health()
                continue
            arm_is_free = False

            prompt_user(f"{label} locked. When your eyes are fixated on the EE, "
                        f"press Enter to record the anchor.")
            bundle = capture_anchor(label, captured_state, vlm_client=vlm_client)
            if bundle is None:
                print(f"[{_ts()}] WARN: no gaze snapshot at {label}; anchor "
                      f"dropped (transit data preserved).")
            else:
                with bundles_lock:
                    bundles.append(bundle)
                print(f"[{_ts()}] Captured {label}: "
                      f"depth={bundle.depth_cm:.1f}cm (valid={bundle.depth_valid}), "
                      f"head=({bundle.head_yaw_deg:+.1f},{bundle.head_pitch_deg:+.1f})")
                last_label = label

            wp_idx += 1
            _check_telemetry_health()

    except KeyboardInterrupt:
        print()
        print(f"[{_ts()}] Ctrl+C received — finishing session.")
    except Exception as e:
        print()
        print(f"[{_ts()}] ERROR: {type(e).__name__}: {e}")
        print(f"[{_ts()}] Saving what has been collected so far.")

    finally:
        # 1. Stop telemetry first so it cannot fire another query_state()
        #    during the shutdown sequence.
        if telemetry is not None:
            try:
                telemetry.stop()
            except Exception as e:
                print(f"[{_ts()}] WARN: telemetry.stop() raised: {e}")

        # 2. Stop the relay consumer so it stops adding to the bundle
        #    queue and releases the TCP connection.
        if _RELAY_CONSUMER is not None:
            try:
                _RELAY_CONSUMER.stop()
            except Exception as e:
                print(f"[{_ts()}] WARN: relay consumer stop raised: {e}")
            _RELAY_CONSUMER = None

        # 3. If the arm is currently in master_free, lock it before
        #    sending h;dur= so the home motion transitions from a known
        #    locked state. The resulting bundle is discarded — this is
        #    a safety lock, not a calibration anchor.
        if arm_is_free:
            print(f"[{_ts()}] Locking arm before auto-home…")
            try:
                capture_pose(link)
            except Exception as e:
                print(f"[{_ts()}] WARN: safety `c` raised: {e}")
            arm_is_free = False

        # 4. Auto-home so the robot does not sit in an arbitrary pose
        #    after the operator walks away.
        try:
            _send_auto_home(link)
        except Exception as e:
            print(f"[{_ts()}] WARN: auto-home failed: {e}")

        # 5. Close the UDP socket.
        try:
            link.close()
        except Exception:
            pass

        # 5. Write the NPZ if we have at least one captured anchor.
        #    write_npz refuses to emit an empty captured block.
        captured_n = sum(1 for b in bundles if b.phase == "captured")
        transit_n = sum(1 for b in bundles if b.phase == "transit")
        if captured_n == 0:
            print(f"[{_ts()}] No anchors captured — no NPZ written.")
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_name = f"poses_with_gaze_{stamp}_v2_freearm.npz"
            out_path = os.path.join(out_dir, out_name)
            write_npz(bundles, out_path)
            print(f"[{_ts()}] Wrote {out_path}")
            print(f"[{_ts()}]   captured anchors: {captured_n}")
            print(f"[{_ts()}]   transit samples:  {transit_n}")

    return out_path


def main() -> int:
    out = run_session(out_dir=".")
    return 0 if out else 2


if __name__ == "__main__":
    sys.exit(main())
