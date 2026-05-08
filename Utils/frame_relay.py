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
keep up the frame is dropped (per-client). Multiple clients are supported —
each accepted connection gets its own handshake and receives every frame
the pump produces. A client whose send fails is removed from the broadcast
set without disturbing the others. This is the configuration the GPU-host
topology actually needs: vlm_service.py and gaze_runner.py both consume
the same relay concurrently.

Run as a module:

    python -m Utils.frame_relay --bind 0.0.0.0 --port 5591 --hz 15

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
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _default_log_callback(line: str) -> None:
    print(line, flush=True)


# Module-level sink for log lines. The control_panel swaps this for a
# callback that tees into the in-panel "Relay" buffer + a subject-tied
# file; standalone `python -m Utils.frame_relay` leaves the default in
# place so terminal behaviour is unchanged.
_LOG_CALLBACK: Callable[[str], None] = _default_log_callback


def set_log_callback(fn: Optional[Callable[[str], None]]) -> None:
    """Install a custom sink for ``_log`` lines. ``None`` restores the
    stdout default. Lines arrive already prefixed with ``[frame_relay] ``.
    """
    global _LOG_CALLBACK
    _LOG_CALLBACK = fn if fn is not None else _default_log_callback


def _log(msg: str) -> None:
    _LOG_CALLBACK(f"[frame_relay] {msg}")


def _pct(samples: Deque[float], qs: list) -> list:
    """Quick percentile from a deque (small N, doesn't justify numpy). Returns
    NaN for empty input so the stats line can show '--' early on."""
    if not samples:
        return [float("nan")] * len(qs)
    s = sorted(samples)
    n = len(s)
    out = []
    for q in qs:
        idx = max(0, min(n - 1, int(round(q * (n - 1)))))
        out.append(s[idx])
    return out


def _fmt_addrs(addrs: list) -> str:
    """Compact addr summary for the stats line: '[10.0.0.5,10.0.0.6]'."""
    if not addrs:
        return ""
    return "[" + ",".join(f"{ip}:{port}" for ip, port in addrs) + "]"


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
    downsamples to `hz` (default 15 Hz — chosen to fit a UT IoT / cellular
    uplink budget; LAN deployments can raise to 30 with --hz 30) by
    skipping bundles. Per-client steady-state at 15 Hz × ~150 KB JPEG is
    ~18 Mbit/s; both perception services pulling concurrently is ~36
    Mbit/s. If you see `dropped > 0` and `bundle_age p99 > 1s` in the
    stats line, the network is the bottleneck and `--jpeg-quality 60` is
    the next lever (~30% smaller frames, no measurable accuracy hit).

    Gaze + IMU fields travel inside each forwarded envelope, which means
    in remote mode consumers see gaze + IMU at relay_hz, not at native
    Neon rates. Production may need split streams (video at relay_hz,
    gaze/IMU at native) if smoothing-quality is observed to suffer.
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
        reader: Optional[Any] = None,
        on_first_publish: Optional[Callable[[Tuple[str, int]], None]] = None,
    ) -> None:
        self.bind_host = bind_host
        self.bind_port = int(bind_port)
        self.hz = float(hz)
        self.neon_host = neon_host or ""
        self.jpeg_quality = int(jpeg_quality)
        self.repo_dir = repo_dir
        # Optional pre-opened reader (anything that quacks like
        # NeonLiveReader: __iter__ yielding bundles, camera_matrix,
        # distortion_coeffs, width/height/fps/close). Used by the
        # operator panel to inject Utils.scene_only_neon_reader.SceneOnlyNeonReader,
        # which mirrors neon_viewer.py's clean SDK call pattern.
        # When None, _open_reader falls back to harmony_vlm's
        # NeonLiveReader (matched-API path).
        self._reader = reader

        self._stop_event = threading.Event()
        # Multi-client: every accepted connection joins this dict. The pump
        # iterates a snapshot under the lock and removes any socket that
        # fails to send (peer crashed, peer rebooted, etc.) without
        # disrupting the others.
        self._client_lock = threading.Lock()
        self._clients: Dict[tuple, socket.socket] = {}

        self._frame_count_published = 0
        self._frame_count_dropped = 0
        # First-publish event hook. Fired once, on the pump thread, the
        # first time `_send_envelope_safe` returns True — i.e. the very
        # first frame the relay actually delivers to a TCP consumer.
        # Subsequent publishes do not re-fire. Used by the operator
        # panel to flip the Send LED green at the moment of the event
        # instead of relying on a 2 s polling timer; see
        # control_panel.py:_on_first_publish_observed.
        self._first_publish_fired = False
        self._on_first_publish = on_first_publish

        # Rolling latency samples for the stats line. Each deque is (ms).
        # bundle_age   — Neon scene timestamp → moment we're ready to send
        # send_ms      — wall time spent in sendall() for this broadcast tick
        # send_per_cli — per-client send time (helps spot one slow consumer)
        self._lat_bundle_age: Deque[float] = deque(maxlen=120)
        self._lat_send_ms: Deque[float] = deque(maxlen=120)

        # In-process fan-out: callbacks registered here are invoked with the
        # raw NeonLiveReader bundle on the pump thread, before the JPEG
        # encode/broadcast step. Used by the Linux-side scene/overlay
        # renderer (Render_Layer_Refactor.md §4.2) so the panel can paint
        # frames without a second NeonLiveReader subscription or a localhost
        # TCP loopback.
        self._local_subs_lock = threading.Lock()
        self._local_subs: List[Callable[[Any], None]] = []

    @property
    def published_count(self) -> int:
        """Monotonic count of frames successfully broadcast since this
        relay started. Read by ``control_panel.py:_poll_relay_status``
        to gate the Send LED on actual data flow rather than just
        thread liveness — the user's chain-of-causation semantic.
        """
        return self._frame_count_published

    # ── public API ─────────────────────────────────────────────────────────

    def serve_forever(self) -> None:
        """Open the Neon device, then run the listen + pump loops together."""
        if self._reader is None:
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
            self._close_all_clients()
            try:
                if self._reader is not None:
                    self._reader.close()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop_event.set()

    # ── in-process fan-out ────────────────────────────────────────────────

    def add_local_subscriber(self, callback: Callable[[Any], None]) -> None:
        """Register an in-process callback invoked with each new bundle on
        the pump thread, before the broadcast step. The callback MUST NOT
        block — it is on the realtime path. Used by the Linux-side scene
        renderer to paint frames directly from the SDK reader without
        opening a second NeonLiveReader."""
        with self._local_subs_lock:
            if callback not in self._local_subs:
                self._local_subs.append(callback)

    def remove_local_subscriber(self, callback: Callable[[Any], None]) -> None:
        with self._local_subs_lock:
            try:
                self._local_subs.remove(callback)
            except ValueError:
                pass

    def _dispatch_local(self, bundle) -> None:
        """Call every registered callback. Exceptions are swallowed (with
        verbose log) so one buggy subscriber can't take the pump down —
        every other consumer (panel renderer, downstream TCP client) keeps
        getting frames."""
        with self._local_subs_lock:
            subs = list(self._local_subs)
        for cb in subs:
            try:
                cb(bundle)
            except Exception as e:
                _log(f"local subscriber raised, dropping its frame: {e}")

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
        """Add a new client and send its handshake. Existing clients are
        unaffected — the pump broadcasts every frame to all of them."""
        try:
            # Per-client send timeout. 2 s was fine on LAN where sendall
            # finishes in <1 ms, but tunneled / cellular clients can take
            # longer than that on the first 150 KB JPEG envelope and would
            # be marked dead before they finished receiving frame 1. 10 s
            # is generous enough for most slow uplinks; truly hung clients
            # still get dropped eventually.
            conn.settimeout(10.0)
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
        with self._client_lock:
            self._clients[addr] = conn
            n = len(self._clients)
        _log(f"client connected: {addr} (total clients: {n})")

    def _close_all_clients(self) -> None:
        with self._client_lock:
            clients = list(self._clients.items())
            self._clients.clear()
        for addr, sock in clients:
            try:
                sock.close()
            except Exception:
                pass
            _log(f"client disconnected: {addr}")

    def _drop_client(self, addr: tuple) -> None:
        with self._client_lock:
            sock = self._clients.pop(addr, None)
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
            _log(f"client disconnected: {addr}")

    def _pump_loop(self) -> None:
        """Iterate the Neon reader at video rate and forward at relay_hz to
        every connected client. JPEG encoding happens once per frame and
        the bytes are sent verbatim to each client; encoding cost does not
        scale with client count."""
        period = 1.0 / max(self.hz, 1e-6)
        next_send = time.perf_counter()
        last_log_t = time.time()
        try:
            for bundle in self._reader:
                if self._stop_event.is_set():
                    break

                # Local in-process subscribers (e.g. the panel renderer)
                # always see every bundle at native Neon frame rate. The
                # relay-hz throttle below only applies to the TCP broadcast.
                self._dispatch_local(bundle)

                now_pc = time.perf_counter()
                if now_pc < next_send:
                    # Skip this bundle — not yet time to send.
                    continue

                with self._client_lock:
                    snapshot = list(self._clients.items())

                if not snapshot:
                    # No consumers; just keep iterating to keep the SDK alive.
                    next_send = now_pc + period
                    continue

                # Encode once, broadcast many.
                envelope = self._encode_frame(bundle)
                if envelope is not None:
                    # bundle_age: Neon scene timestamp → just before sendall.
                    # video.timestamp_ns is unix-ns; clock-aligned via SDK.
                    ts_video_ns = int(getattr(bundle.video, "timestamp_ns", 0))
                    if ts_video_ns:
                        age_ms = max(0.0, (time.time_ns() - ts_video_ns) / 1e6)
                        self._lat_bundle_age.append(age_ms)

                    send_t0 = time.perf_counter()
                    for addr, sock in snapshot:
                        if not self._send_envelope_safe(sock, addr, envelope):
                            self._drop_client(addr)
                    send_ms = (time.perf_counter() - send_t0) * 1e3
                    self._lat_send_ms.append(send_ms)

                next_send += period
                # If we fell behind by >2 periods (slow encode, slow socket),
                # resync rather than burst-catch-up.
                now_pc2 = time.perf_counter()
                if next_send < now_pc2 - 2.0 * period:
                    next_send = now_pc2

                # Periodic stats line; cheap, throttled.
                if (time.time() - last_log_t) > 5.0:
                    last_log_t = time.time()
                    with self._client_lock:
                        n = len(self._clients)
                        client_addrs = list(self._clients.keys())
                    bp50, bp99 = _pct(self._lat_bundle_age, [0.5, 0.99])
                    sp50, sp99 = _pct(self._lat_send_ms, [0.5, 0.99])
                    _log(
                        f"published={self._frame_count_published} "
                        f"dropped={self._frame_count_dropped} "
                        f"clients={n}{_fmt_addrs(client_addrs)} "
                        f"bundle_age p50/p99={bp50:.0f}/{bp99:.0f} ms "
                        f"send p50/p99={sp50:.1f}/{sp99:.1f} ms"
                    )
        except Exception as e:
            _log(f"pump loop exited: {e}")
            self._stop_event.set()

    def _encode_frame(self, bundle) -> Optional[bytes]:
        """Build the wire-ready bytes for one frame envelope.

        Returns the full ``[json_len][jpeg_len][json][jpeg]`` byte string so
        the broadcast loop can ``sendall`` it directly to each client.
        Returns ``None`` if JPEG encoding fails (rare; logged + counted as
        dropped)."""
        try:
            ok, buf = cv2.imencode(
                ".jpg",
                bundle.video.bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok:
                self._frame_count_dropped += 1
                return None
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
            hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
            return struct.pack(">II", len(hdr_bytes), len(jpeg)) + hdr_bytes + jpeg
        except Exception as e:
            _log(f"encode_frame error: {e}")
            self._frame_count_dropped += 1
            return None

    def _send_envelope_safe(
        self, sock: socket.socket, addr: Tuple[str, int], envelope: bytes
    ) -> bool:
        """Write a pre-encoded envelope to one client socket. Returns False
        on socket failure so the pump can drop that client.

        Fires ``self._on_first_publish(addr)`` exactly once across the
        relay's lifetime — on the first `sendall` that returns True. The
        callback runs on this (pump) thread, so it must not block; the
        panel uses a Qt-signal emit which is thread-safe."""
        try:
            sock.sendall(envelope)
            self._frame_count_published += 1
            if not self._first_publish_fired:
                self._first_publish_fired = True
                _log(f"first frame published to {addr[0]}:{addr[1]}")
                cb = self._on_first_publish
                if cb is not None:
                    try:
                        cb((addr[0], int(addr[1])))
                    except Exception as e:
                        _log(f"on_first_publish callback raised: {e}")
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False


# ── module CLI ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linux-side Neon frame relay (TCP)")
    p.add_argument("--bind", default="0.0.0.0", help="Bind host (default 0.0.0.0)")
    p.add_argument("--port", type=int, default=5591, help="Bind port (default 5591)")
    p.add_argument("--hz", type=float, default=15.0,
                   help="Relay rate in Hz (default 15 — fits UT IoT / cellular "
                        "uplink budget at q=75 JPEG; raise to 30 on LAN to match "
                        "Neon scene-camera native FPS)")
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
