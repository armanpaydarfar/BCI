#!/usr/bin/env python3
"""
test_relay_loopback.py — exercise the wire format of Utils/frame_relay.py
and Utils/remote_frame_reader.py without needing a live Neon device.

Spins up a *minimal* TCP server that hand-builds handshake + frame
envelopes (using the same private send helper) and then connects both
``RemoteFrameReader`` and ``RemoteNeonDevice`` to it. Verifies:

    1. Handshake fields surface as expected (camera_matrix, intrinsics).
    2. RemoteFrameReader yields a FrameBundle-shaped object with .video.bgr,
       .gaze.x/y, .gaze.eyeball_center_*, .gaze.optical_axis_*, .imu.*
    3. RemoteNeonDevice.receive_scene_video_frame / receive_gaze_datum /
       receive_imu_datum return values that match the input fixtures.
    4. Drop-oldest queueing keeps the wire reader alive when consumers
       are slow.

Run from the BCI root with the harmony_vlm conda env (cv2 + numpy):

    C:/Users/arman/miniconda3/envs/harmony_vlm/python.exe tests/test_relay_loopback.py

Exit code 0 = all checks passed. Non-zero = one or more checks failed
(stderr names the failing assertion).
"""

from __future__ import annotations

import json
import socket
import struct
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from Utils.frame_relay import _send_envelope  # type: ignore  # noqa: E402
from Utils.remote_frame_reader import (  # noqa: E402
    RemoteFrameReader,
    RemoteNeonDevice,
)


# ── tiny mock relay ────────────────────────────────────────────────────────


CAMERA_MATRIX = [[800.0, 0.0, 800.0],
                 [0.0, 800.0, 600.0],
                 [0.0, 0.0, 1.0]]
DISTORTION = [-0.01, 0.02, 0.0, 0.0, 0.0]


def _build_test_frame(idx: int) -> np.ndarray:
    """Make a 320x240 BGR test pattern with index baked in for inspection."""
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:, :, idx % 3] = 255
    return img


def _gaze_dict(idx: int) -> dict:
    return {
        "x": 100.0 + idx, "y": 50.0 + idx,
        "worn": True,
        "timestamp_unix_seconds": time.time(),
        "eyeball_center_left_x": 1.0, "eyeball_center_left_y": 2.0, "eyeball_center_left_z": 3.0,
        "eyeball_center_right_x": -1.0, "eyeball_center_right_y": 2.0, "eyeball_center_right_z": 3.0,
        "optical_axis_left_x": 0.0, "optical_axis_left_y": 0.0, "optical_axis_left_z": 1.0,
        "optical_axis_right_x": 0.0, "optical_axis_right_y": 0.0, "optical_axis_right_z": 1.0,
    }


def _imu_dict() -> dict:
    return {
        "quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        "gyro": {"x": 0.1, "y": 0.2, "z": 0.3},
        "accel": {"x": 0.0, "y": 9.81, "z": 0.0},
        "timestamp_unix_seconds": time.time(),
    }


class MockRelay:
    """Minimal one-client TCP server emitting handshake + N frames."""

    def __init__(self, port: int = 0, n_frames: int = 5, hz: float = 50.0,
                 stay_open_seconds: float = 30.0) -> None:
        self.port = port
        self.n_frames = int(n_frames)
        self.period = 1.0 / max(float(hz), 1e-6)
        self.stay_open_seconds = float(stay_open_seconds)
        self._stop = threading.Event()
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", port))
        self._srv.listen(1)
        self.bound_port = self._srv.getsockname()[1]
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._srv.close()
        except OSError:
            pass

    def _run(self) -> None:
        self._srv.settimeout(5.0)
        try:
            conn, _addr = self._srv.accept()
        except socket.timeout:
            return
        try:
            handshake = {
                "type": "handshake",
                "camera_matrix": CAMERA_MATRIX,
                "distortion_coeffs": DISTORTION,
                "scene_width": 320, "scene_height": 240,
                "fps_video": 30.0, "relay_hz": 50.0,
            }
            _send_envelope(conn, handshake, b"")
            for i in range(self.n_frames):
                if self._stop.is_set():
                    break
                bgr = _build_test_frame(i)
                ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                jpg = bytes(buf) if ok else b""
                hdr = {
                    "type": "frame",
                    "frame_idx": i,
                    "ts_send_ns": int(time.monotonic_ns()),
                    "ts_video_ns": int(1_000_000_000 + i),
                    "ts_gaze_ns":  int(2_000_000_000 + i),
                    "ts_imu_ns":   int(3_000_000_000 + i),
                    "gaze": _gaze_dict(i),
                    "imu": _imu_dict(),
                }
                _send_envelope(conn, hdr, jpg)
                time.sleep(self.period)
            # Hold the socket open briefly so the consumer side can read all
            # buffered frames before EOF.
            time.sleep(0.2)
        finally:
            try:
                conn.close()
            except OSError:
                pass


# ── checks ─────────────────────────────────────────────────────────────────


def check_remote_frame_reader() -> None:
    relay = MockRelay(port=0, n_frames=5, hz=200.0)
    relay.start()
    reader = RemoteFrameReader("127.0.0.1", relay.bound_port,
                               wait_for_handshake_s=3.0,
                               auto_reconnect=False)
    try:
        # Handshake fields
        assert reader.width == 320, f"width={reader.width}"
        assert reader.height == 240, f"height={reader.height}"
        K = reader.camera_matrix
        assert abs(K[0, 0] - 800.0) < 1e-9
        assert reader.distortion_coeffs is not None
        assert abs(reader.distortion_coeffs[0] - (-0.01)) < 1e-9

        bundles = []
        deadline = time.time() + 5.0
        for bundle in reader:
            bundles.append(bundle)
            if len(bundles) >= 5 or time.time() > deadline:
                break
        assert len(bundles) >= 5, f"only got {len(bundles)} bundles"

        b0 = bundles[0]
        assert b0.video.bgr.shape == (240, 320, 3)
        assert b0.video.frame_idx == 0
        assert b0.gaze.worn is True
        assert abs(b0.gaze.x - 100.0) < 1e-9
        # eyeball_center fields surfaced as attributes
        assert hasattr(b0.gaze, "eyeball_center_left_x")
        assert b0.imu is not None
        # harmony_vlm IMUSample stores quaternion as (x, y, z, w) — verify.
        assert b0.imu.quaternion.shape == (4,)
        assert abs(b0.imu.quaternion[3] - 1.0) < 1e-9, "expected w=1"
    finally:
        reader.close()
        relay.stop()


def check_remote_neon_device() -> None:
    relay = MockRelay(port=0, n_frames=5, hz=200.0)
    relay.start()
    dev = RemoteNeonDevice("127.0.0.1", relay.bound_port,
                           wait_for_handshake_s=3.0,
                           auto_reconnect=False)
    try:
        bgr, _dt = dev.receive_scene_video_frame()
        assert bgr is not None and bgr.shape == (240, 320, 3)

        gaze = dev.receive_gaze_datum()
        assert gaze is not None
        assert gaze.worn is True
        assert hasattr(gaze, "optical_axis_left_z")
        assert abs(gaze.optical_axis_left_z - 1.0) < 1e-9

        imu = dev.receive_imu_datum()
        assert imu is not None
        assert abs(imu.quaternion.w - 1.0) < 1e-9
        # gyro_data + angular_velocity_* both present (gaze_system uses both)
        assert abs(imu.gyro_data.x - 0.1) < 1e-9
        assert abs(imu.angular_velocity_x - 0.1) < 1e-9
    finally:
        dev.close()
        relay.stop()


def check_drop_oldest_resilience() -> None:
    """Send 50 frames at 1000 Hz to a slow consumer (sleeps between reads).
    With a queue cap of 4 and drop-oldest semantics the producer must not
    block; the consumer must still see the *most recent* frames."""
    relay = MockRelay(port=0, n_frames=50, hz=1000.0)
    relay.start()
    reader = RemoteFrameReader("127.0.0.1", relay.bound_port,
                               wait_for_handshake_s=3.0,
                               auto_reconnect=False)
    try:
        seen = []
        deadline = time.time() + 5.0
        for bundle in reader:
            seen.append(bundle.video.frame_idx)
            time.sleep(0.05)  # 20 Hz consumer vs 1000 Hz producer
            if time.time() > deadline:
                break
        assert seen, "no frames received under load"
        # We don't insist on a specific count — just that the *last* frame
        # observed is recent (>= last quartile of the produced range).
        assert max(seen) >= 35, f"consumer fell behind, max seen idx={max(seen)}"
    finally:
        reader.close()
        relay.stop()


def check_consumer_starts_before_relay() -> None:
    """The user use case: Windows services start at home, then later the
    Linux relay comes up at the office. The consumer must wait patiently
    and connect once the relay arrives — no manual restart needed."""
    # Pick a fixed port and ensure it's free, then start the consumer
    # against it before the relay binds.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    reader = RemoteFrameReader("127.0.0.1", port,
                               wait_for_handshake_s=0.0,
                               auto_reconnect=True)
    try:
        # Give the consumer ~1.5 s to fail one connect attempt (refused).
        time.sleep(1.5)
        assert not reader._conn.connected, "consumer should still be retrying"

        # Now stand up the relay on the same port.
        relay = MockRelay(port=port, n_frames=10, hz=60.0)
        relay.start()
        try:
            # Pull a few bundles — the consumer's manager should have
            # reconnected by now.
            bundles = []
            deadline = time.time() + 8.0
            for bundle in reader:
                bundles.append(bundle)
                if len(bundles) >= 3 or time.time() > deadline:
                    break
            assert len(bundles) >= 3, f"only got {len(bundles)} after relay start"
            assert reader.width == 320, f"handshake didn't update width: {reader.width}"
        finally:
            relay.stop()
    finally:
        reader.close()


def check_reconnect_after_relay_drop() -> None:
    """Relay drops mid-session (Wi-Fi blip / box restart). Consumer should
    reconnect on the next cycle and resume yielding."""
    relay1 = MockRelay(port=0, n_frames=5, hz=200.0)
    relay1.start()
    port = relay1.bound_port

    reader = RemoteFrameReader("127.0.0.1", port,
                               wait_for_handshake_s=3.0,
                               auto_reconnect=True)
    try:
        # Pull a couple frames from relay1, then kill it.
        seen_a = []
        for bundle in reader:
            seen_a.append(bundle.video.frame_idx)
            if len(seen_a) >= 2:
                break
        assert seen_a, "didn't see any frames from relay1"

        relay1.stop()
        time.sleep(0.5)
        # Consumer's manager is now in backoff. Bring up a NEW relay on
        # the same port and verify the consumer reconnects + yields.
        relay2 = MockRelay(port=port, n_frames=10, hz=200.0)
        relay2.start()
        try:
            seen_b = []
            deadline = time.time() + 8.0
            for bundle in reader:
                seen_b.append(bundle.video.frame_idx)
                if len(seen_b) >= 2 or time.time() > deadline:
                    break
            assert seen_b, f"consumer didn't reconnect — got 0 frames from relay2"
        finally:
            relay2.stop()
    finally:
        reader.close()


# ── runner ─────────────────────────────────────────────────────────────────


def main() -> int:
    checks = [
        ("RemoteFrameReader", check_remote_frame_reader),
        ("RemoteNeonDevice", check_remote_neon_device),
        ("drop-oldest under load", check_drop_oldest_resilience),
        ("consumer starts before relay", check_consumer_starts_before_relay),
        ("reconnect after relay drop", check_reconnect_after_relay_drop),
    ]
    failures: list = []
    for label, fn in checks:
        t0 = time.time()
        try:
            fn()
            dt = (time.time() - t0) * 1000
            print(f"[OK  ] {label}  ({dt:.0f} ms)")
        except AssertionError as e:
            failures.append((label, str(e)))
            print(f"[FAIL] {label}: {e}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 — surface in CI/log
            failures.append((label, repr(e)))
            print(f"[ERR ] {label}: {e!r}", file=sys.stderr)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
