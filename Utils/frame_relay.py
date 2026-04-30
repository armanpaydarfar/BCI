"""
frame_relay.py — Linux-side TCP server that pumps Pupil Labs Neon frames
(plus gaze + IMU metadata) to a single remote consumer.

Wire protocol (binary, length-prefixed, big-endian):

    Each envelope on the wire is:
        struct.pack(">II", len(json_hdr), len(jpeg_bytes))
        + json_hdr_bytes
        + jpeg_bytes

    Two envelope types are sent, distinguished by the JSON `type` field:

    1. Handshake (sent once, immediately after a client connects):
       JSON =
           {
             "type": "handshake",
             "camera_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]],
             "distortion_coeffs": [...]   # optional, may be null
             "scene_width":  1600,
             "scene_height": 1200,
             "fps_video":    30.0,
             "relay_hz":     10.0,
           }
       jpeg_bytes is empty (length 0).

    2. Frame:
       JSON =
           {
             "type": "frame",
             "frame_idx":   N,
             "ts_send_ns":  <relay send time, monotonic_ns>,
             "ts_video_ns": <neon scene timestamp, unix ns>,
             "ts_gaze_ns":  <neon gaze timestamp, unix ns>,
             "ts_imu_ns":   <neon imu timestamp, unix ns> | null,
             "gaze": {
                "x": float, "y": float,
                "worn": bool,
                "timestamp_unix_seconds": float,
                "eyeball_center_left_x":  float, ... right_x/y/z, etc.
                "optical_axis_left_x":   float, ... right_x/y/z, etc.
             },
             "imu": {                   # may be null if not available
                "quaternion": {"w":..,"x":..,"y":..,"z":..},
                "gyro":       {"x":..,"y":..,"z":..},   # rad/s
                "accel":      {"x":..,"y":..,"z":..},   # m/s^2
                "timestamp_unix_seconds": float,
             }
           }
       jpeg_bytes carries the JPEG-encoded BGR scene frame.

The gaze block is intentionally rich — gaze_system.py needs the eyeball-center
and optical-axis fields for vergence depth, and the imu quaternion for head
angles. vlm_service.py only consumes (x, y, worn) from the gaze block, so the
extra fields are transparent on that side.

Backpressure: the pump never blocks the realtime loop. If the socket can't
keep up the frame is dropped. The server accepts at most one client at a
time; subsequent connection attempts replace the old client.

Run as a module:

    python -m Utils.frame_relay --bind 0.0.0.0 --port 5591 --hz 30

For Phase 1 single-machine validation, --bind 127.0.0.1 keeps the relay
loopback-only on Windows.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np


def _log(msg: str) -> None:
    print(f"[frame_relay] {msg}", flush=True)


# Names of richer gaze-datum fields gaze_system.py needs. Best-effort —
# missing fields surface as None on the consumer side.
_GAZE_EXTRA_FIELDS = (
    "eyeball_center_left_x", "eyeball_center_left_y", "eyeball_center_left_z",
    "eyeball_center_right_x", "eyeball_center_right_y", "eyeball_center_right_z",
    "optical_axis_left_x", "optical_axis_left_y", "optical_axis_left_z",
    "optical_axis_right_x", "optical_axis_right_y", "optical_axis_right_z",
)


def _gaze_to_dict(gaze) -> Dict[str, Any]:
    """Best-effort serialisation of a Pupil Labs gaze datum."""
    out: Dict[str, Any] = {
        "x": float(getattr(gaze, "x", float("nan"))),
        "y": float(getattr(gaze, "y", float("nan"))),
        "worn": bool(getattr(gaze, "worn", True)),
        "timestamp_unix_seconds": float(getattr(gaze, "timestamp_unix_seconds", 0.0)),
    }
    for fname in _GAZE_EXTRA_FIELDS:
        val = getattr(gaze, fname, None)
        if val is not None:
            try:
                out[fname] = float(val)
            except (TypeError, ValueError):
                pass
    return out


def _imu_to_dict(imu) -> Optional[Dict[str, Any]]:
    if imu is None:
        return None
    out: Dict[str, Any] = {
        "timestamp_unix_seconds": float(getattr(imu, "timestamp_unix_seconds", 0.0)),
    }
    q = getattr(imu, "quaternion", None)
    if q is not None:
        out["quaternion"] = {
            "w": float(getattr(q, "w", 0.0)),
            "x": float(getattr(q, "x", 0.0)),
            "y": float(getattr(q, "y", 0.0)),
            "z": float(getattr(q, "z", 0.0)),
        }
    # Gyro / accel field names vary across SDK versions; mirror the candidate
    # list used by gaze_system.GazeSystem._gyro_mag_from_imu_datum.
    for sub_name, out_key in (("gyro_data", "gyro"), ("accel_data", "accel")):
        sub = getattr(imu, sub_name, None)
        if sub is not None:
            try:
                out[out_key] = {
                    "x": float(getattr(sub, "x", 0.0)),
                    "y": float(getattr(sub, "y", 0.0)),
                    "z": float(getattr(sub, "z", 0.0)),
                }
            except (TypeError, ValueError):
                pass
    return out


def _send_envelope(sock: socket.socket, header: Dict[str, Any], jpeg: bytes) -> None:
    """Serialise + write one envelope. Caller is responsible for catching socket errors."""
    hdr = json.dumps(header, separators=(",", ":")).encode("utf-8")
    prefix = struct.pack(">II", len(hdr), len(jpeg))
    sock.sendall(prefix + hdr + jpeg)


class FrameRelayServer:
    """TCP server that pumps Neon FrameBundles to a single connected consumer.

    The server hosts a NeonLiveReader internally (one Pupil Labs subscription).
    On every accepted client connection we send a handshake envelope, then
    forward frames encoded as JPEG until the client disconnects.

    Frames are produced by NeonLiveReader at video rate (~30 Hz). The relay
    downsamples to `hz` (default 10 Hz) by skipping bundles. Gaze + IMU
    fields travel inside each forwarded envelope, which means in remote mode
    consumers see gaze + IMU at relay_hz, not at native Neon rates. For
    Phase 1 validation that's an accepted limitation; production may need
    split streams (video at 10 Hz, gaze/IMU at native).
    """

    def __init__(
        self,
        *,
        bind_host: str,
        bind_port: int,
        hz: float,
        neon_host: str = "",
        jpeg_quality: int = 75,
        repo_dir: Optional[str] = None,
    ) -> None:
        self.bind_host = bind_host
        self.bind_port = int(bind_port)
        self.hz = float(hz)
        self.neon_host = neon_host or ""
        self.jpeg_quality = int(jpeg_quality)
        self.repo_dir = repo_dir

        self._stop_event = threading.Event()
        self._client_lock = threading.Lock()
        self._client_sock: Optional[socket.socket] = None
        self._client_addr: Optional[tuple] = None

        self._frame_count_published = 0
        self._frame_count_dropped = 0
        self._reader = None  # NeonLiveReader, lazily imported

    # ── public API ─────────────────────────────────────────────────────────

    def serve_forever(self) -> None:
        """Open the Neon device, then run the listen + pump loops together."""
        self._reader = self._open_reader()

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.bind_host, self.bind_port))
        srv.listen(1)
        srv.settimeout(0.5)
        _log(f"listening on tcp://{self.bind_host}:{self.bind_port} (hz={self.hz:.1f})")

        # Accept loop runs in the main thread; the pump runs in a worker.
        pump_thread = threading.Thread(target=self._pump_loop, daemon=True, name="relay-pump")
        pump_thread.start()

        try:
            while not self._stop_event.is_set():
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                self._install_client(conn, addr)
        finally:
            self._stop_event.set()
            try:
                srv.close()
            except Exception:
                pass
            self._close_client()
            try:
                if self._reader is not None:
                    self._reader.close()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop_event.set()

    # ── internals ──────────────────────────────────────────────────────────

    def _open_reader(self):
        """Import NeonLiveReader from the harmony_vlm clone and connect."""
        if self.repo_dir:
            repo_dir = os.path.abspath(self.repo_dir)
            if not os.path.isdir(repo_dir):
                raise SystemExit(f"frame_relay: --repo-dir not a directory: {repo_dir}")
            if repo_dir not in sys.path:
                sys.path.insert(0, repo_dir)
        try:
            from utils.neon import NeonLiveReader  # type: ignore
        except ImportError as e:
            raise SystemExit(
                f"frame_relay: cannot import harmony_vlm utils.neon ({e}). "
                "Pass --repo-dir <harmony_vlm clone> or run with the harmony_vlm conda env."
            ) from e
        _log(f"connecting to Neon (host={self.neon_host or 'auto-discover'})…")
        return NeonLiveReader(host=self.neon_host or None)

    def _install_client(self, conn: socket.socket, addr: tuple) -> None:
        """Replace any existing client with the new connection."""
        with self._client_lock:
            self._close_client_locked()
            try:
                conn.settimeout(2.0)
                # Send handshake immediately. Reader provides camera matrix +
                # distortion (factory-calibrated). Width/height come from the
                # NeonLiveReader class attributes.
                K = np.asarray(self._reader.camera_matrix, dtype=float)
                dist = self._reader.distortion_coeffs
                handshake = {
                    "type": "handshake",
                    "camera_matrix": K.tolist(),
                    "distortion_coeffs": (
                        np.asarray(dist).tolist() if dist is not None else None
                    ),
                    "scene_width": int(getattr(self._reader, "width", 1600)),
                    "scene_height": int(getattr(self._reader, "height", 1200)),
                    "fps_video": float(getattr(self._reader, "fps", 30.0)),
                    "relay_hz": float(self.hz),
                }
                _send_envelope(conn, handshake, b"")
            except OSError as e:
                _log(f"client {addr} dropped during handshake: {e}")
                try:
                    conn.close()
                except Exception:
                    pass
                return
            self._client_sock = conn
            self._client_addr = addr
            _log(f"client connected: {addr}")

    def _close_client(self) -> None:
        with self._client_lock:
            self._close_client_locked()

    def _close_client_locked(self) -> None:
        sock = self._client_sock
        addr = self._client_addr
        self._client_sock = None
        self._client_addr = None
        if sock is None:
            return
        try:
            sock.close()
        except Exception:
            pass
        if addr is not None:
            _log(f"client disconnected: {addr}")

    def _pump_loop(self) -> None:
        """Iterate the Neon reader at video rate and forward at relay_hz."""
        period = 1.0 / max(self.hz, 1e-6)
        next_send = time.perf_counter()
        last_log_t = time.time()
        try:
            for bundle in self._reader:
                if self._stop_event.is_set():
                    break

                now_pc = time.perf_counter()
                if now_pc < next_send:
                    # Skip this bundle — not yet time to send.
                    continue

                with self._client_lock:
                    sock = self._client_sock
                    addr = self._client_addr

                if sock is None:
                    # No consumer; just keep iterating to keep the SDK alive.
                    next_send = now_pc + period
                    continue

                ok = self._send_frame(sock, bundle)
                if not ok:
                    _log(f"client send failed; dropping {addr}")
                    self._close_client()

                next_send += period
                # If we fell behind by >2 periods (slow encode, slow socket),
                # resync rather than burst-catch-up.
                now_pc2 = time.perf_counter()
                if next_send < now_pc2 - 2.0 * period:
                    next_send = now_pc2

                # Periodic stats line; cheap, throttled.
                if (time.time() - last_log_t) > 5.0:
                    last_log_t = time.time()
                    _log(
                        f"published={self._frame_count_published} "
                        f"dropped={self._frame_count_dropped}"
                    )
        except Exception as e:
            _log(f"pump loop exited: {e}")
            self._stop_event.set()

    def _send_frame(self, sock: socket.socket, bundle) -> bool:
        try:
            ok, buf = cv2.imencode(
                ".jpg",
                bundle.video.bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok:
                self._frame_count_dropped += 1
                return True  # encoding failed but socket is fine
            jpeg = bytes(buf)

            header = {
                "type": "frame",
                "frame_idx": int(getattr(bundle.video, "frame_idx", 0)),
                "ts_send_ns": int(time.monotonic_ns()),
                "ts_video_ns": int(getattr(bundle.video, "timestamp_ns", 0)),
                "ts_gaze_ns": int(getattr(bundle.gaze, "timestamp_ns", 0)),
                "ts_imu_ns": (
                    int(bundle.imu.timestamp_ns) if bundle.imu is not None else None
                ),
                "gaze": _gaze_to_dict(bundle.gaze),
                "imu": _imu_to_dict(bundle.imu),
            }
            _send_envelope(sock, header, jpeg)
            self._frame_count_published += 1
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False
        except Exception as e:
            _log(f"send_frame error: {e}")
            self._frame_count_dropped += 1
            return True


# ── module CLI ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linux-side Neon frame relay (TCP)")
    p.add_argument("--bind", default="0.0.0.0", help="Bind host (default 0.0.0.0)")
    p.add_argument("--port", type=int, default=5591, help="Bind port (default 5591)")
    p.add_argument("--hz", type=float, default=30.0,
                   help="Relay rate in Hz (default 30, matches Neon scene-camera native FPS)")
    p.add_argument("--neon-host", default="",
                   help="Companion phone IP; empty = mDNS auto-discovery")
    p.add_argument("--jpeg-quality", type=int, default=75, help="JPEG quality 1-100")
    p.add_argument("--repo-dir", default=None,
                   help="Path to harmony_vlm clone (needed for utils.neon import "
                        "if not running inside the harmony_vlm conda env)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    server = FrameRelayServer(
        bind_host=args.bind,
        bind_port=args.port,
        hz=args.hz,
        neon_host=args.neon_host,
        jpeg_quality=args.jpeg_quality,
        repo_dir=args.repo_dir,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("KeyboardInterrupt — stopping")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
