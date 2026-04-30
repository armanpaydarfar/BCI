"""
remote_frame_reader.py — Consumer side of the BCI/Utils/frame_relay.py wire.

Provides two adapters over a single TCP relay connection:

1. ``RemoteFrameReader``
       Drop-in replacement for ``harmony_vlm.utils.neon.NeonLiveReader``
       used by ``vlm_service.py``. Yields ``FrameBundle``-shaped objects
       through ``__iter__`` and exposes ``camera_matrix`` /
       ``distortion_coeffs`` properties so Depth Pro and the waypoint
       compute code work unchanged.

2. ``RemoteNeonDevice``
       Drop-in replacement for ``pupil_labs.realtime_api.simple.Device``
       used by ``Utils.gaze.gaze_system.GazeSystem``. Implements
       ``receive_scene_video_frame()``, ``receive_gaze_datum()``,
       ``receive_imu_datum()``, and ``close()`` — the exact methods the
       three Neon threads consume.

Both adapters share one ``_RelayConnection`` that owns the TCP socket and
parses envelopes off the wire. The connection re-distributes each envelope
into per-stream queues so the two adapters can be active concurrently if
the same process needs both (not required in production — vlm_service.py
uses only the FrameBundle adapter, gaze_system.py uses only the device
adapter).

Wire protocol matches ``Utils/frame_relay.py``:

    [4-byte JSON length][4-byte JPEG length][JSON header][JPEG bytes]

Each envelope is either a ``handshake`` (sent once at connect) or a
``frame``. See ``Utils/frame_relay.py`` for the field schema.

Phase 1 limitation: gaze + IMU are subsampled to the relay rate (default
30 Hz, matching Neon scene-camera FPS). Native Neon gaze is ~200 Hz;
gaze_system smoothers were tuned for that. This is acceptable for the
single-machine validation pass; a production deployment likely needs
split streams.
"""

from __future__ import annotations

import json
import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import cv2
import numpy as np


def _log(msg: str) -> None:
    print(f"[remote_frame_reader] {msg}", flush=True)


# Default Neon scene-camera intrinsics; only used if the relay handshake
# omits camera_matrix (it shouldn't, but matches harmony_vlm's fallback).
_DEFAULT_NEON_FX = 800.0
_DEFAULT_NEON_FY = 800.0
_DEFAULT_NEON_CX = 800.0
_DEFAULT_NEON_CY = 600.0


# ── lightweight stub types mimicking pupil_labs / harmony_vlm shapes ───────


@dataclass
class _VideoFrameStub:
    timestamp_ns: int
    frame_idx: int
    bgr: np.ndarray


class _GazeStub:
    """Mirrors the attribute surface of pupil_labs' gaze datum + harmony_vlm's
    GazeSample. Provides ``x``, ``y``, ``worn``, ``timestamp_unix_seconds``
    and best-effort ``eyeball_center_*`` / ``optical_axis_*`` fields."""

    __slots__ = ("__dict__",)  # allow dynamic attribute set

    def __init__(self, gaze_dict: Dict[str, Any], ts_ns: int) -> None:
        self.x = float(gaze_dict.get("x", float("nan")))
        self.y = float(gaze_dict.get("y", float("nan")))
        self.worn = bool(gaze_dict.get("worn", True))
        ts = gaze_dict.get("timestamp_unix_seconds")
        if ts is None and ts_ns:
            ts = float(ts_ns) / 1e9
        self.timestamp_unix_seconds = float(ts or 0.0)
        # gaze_system reads these via getattr; populating dynamically keeps
        # the stub permissive when the relay omits them on older firmware.
        for k, v in gaze_dict.items():
            if k in ("x", "y", "worn", "timestamp_unix_seconds"):
                continue
            try:
                setattr(self, k, float(v))
            except (TypeError, ValueError):
                setattr(self, k, v)

    @property
    def timestamp_unix_ns(self) -> int:
        # NeonLiveReader uses this name on the matched-frame gaze datum.
        return int(self.timestamp_unix_seconds * 1e9)


class _Vec3Stub:
    __slots__ = ("x", "y", "z")

    def __init__(self, d: Dict[str, Any]) -> None:
        self.x = float(d.get("x", 0.0))
        self.y = float(d.get("y", 0.0))
        self.z = float(d.get("z", 0.0))


class _QuatStub:
    __slots__ = ("w", "x", "y", "z")

    def __init__(self, d: Dict[str, Any]) -> None:
        self.w = float(d.get("w", 1.0))
        self.x = float(d.get("x", 0.0))
        self.y = float(d.get("y", 0.0))
        self.z = float(d.get("z", 0.0))


class _IMUStub:
    """Mirrors pupil_labs IMU datum: quaternion, gyro_data, accel_data."""

    def __init__(self, imu_dict: Dict[str, Any], ts_ns: Optional[int]) -> None:
        q = imu_dict.get("quaternion") or {}
        self.quaternion = _QuatStub(q)
        gyro = imu_dict.get("gyro") or {}
        accel = imu_dict.get("accel") or {}
        self.gyro_data = _Vec3Stub(gyro)
        self.accel_data = _Vec3Stub(accel)
        # pupil_labs' field name is angular_velocity_*; expose both for the
        # gaze_system gyro_mag candidate list.
        self.angular_velocity_x = self.gyro_data.x
        self.angular_velocity_y = self.gyro_data.y
        self.angular_velocity_z = self.gyro_data.z
        ts = imu_dict.get("timestamp_unix_seconds")
        if ts is None and ts_ns:
            ts = float(ts_ns) / 1e9
        self.timestamp_unix_seconds = float(ts or 0.0)


@dataclass
class _IMUSampleStub:
    """harmony_vlm pupil_reader.IMUSample-shaped object for FrameBundle.imu."""
    timestamp_ns: int
    quaternion: np.ndarray  # (x, y, z, w) per harmony_vlm convention
    accel: np.ndarray
    gyro: np.ndarray


@dataclass
class _FrameBundleStub:
    video: _VideoFrameStub
    gaze: _GazeStub
    worn: bool
    imu: Optional[_IMUSampleStub]


# ── shared TCP connection ──────────────────────────────────────────────────


class _RelayConnection:
    """Owns the TCP socket, parses envelopes, and fans out to subscriber queues.

    Two queue slots:
      - ``frame_q``: full envelopes for ``RemoteFrameReader``
      - ``video_q`` / ``gaze_q`` / ``imu_q``: split slices for
        ``RemoteNeonDevice``

    All queues are bounded; producer drops the oldest item on overflow so a
    slow consumer cannot starve the wire reader.
    """

    def __init__(self, host: str, port: int, *, connect_timeout_s: float = 10.0) -> None:
        self.host = host
        self.port = int(port)
        self.connect_timeout_s = float(connect_timeout_s)

        self._sock: Optional[socket.socket] = None
        self._stop_event = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None
        self._handshake_event = threading.Event()
        self._handshake: Dict[str, Any] = {}

        # Bounded queues — drop-oldest semantics in _put_drop_oldest.
        self._frame_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=4)
        self._video_q: "queue.Queue[Tuple[np.ndarray, float, int, int]]" = queue.Queue(maxsize=4)
        self._gaze_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=8)
        self._imu_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=8)

        # Subscribers register what they actually need so we don't waste CPU
        # decoding JPEG / building stubs nobody reads.
        self._want_frame_bundle = False
        self._want_device_split = False

    # ── lifecycle ─────────────────────────────────────────────────────────

    def connect(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.connect_timeout_s)
        _log(f"connecting to relay tcp://{self.host}:{self.port}…")
        s.connect((self.host, self.port))
        s.settimeout(None)
        self._sock = s
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="relay-reader"
        )
        self._reader_thread.start()
        # Block until handshake arrives so callers can read camera_matrix.
        if not self._handshake_event.wait(timeout=5.0):
            raise RuntimeError("relay handshake did not arrive within 5 s")
        _log("handshake received")

    def close(self) -> None:
        self._stop_event.set()
        try:
            if self._sock is not None:
                self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            if self._sock is not None:
                self._sock.close()
        except OSError:
            pass
        self._sock = None

    # ── subscription markers ──────────────────────────────────────────────

    def enable_frame_bundle(self) -> None:
        self._want_frame_bundle = True

    def enable_device_split(self) -> None:
        self._want_device_split = True

    @property
    def handshake(self) -> Dict[str, Any]:
        return dict(self._handshake)

    # ── queues exposed to adapters ────────────────────────────────────────

    @property
    def frame_q(self) -> "queue.Queue[Dict[str, Any]]":
        return self._frame_q

    @property
    def video_q(self) -> "queue.Queue[Tuple[np.ndarray, float, int, int]]":
        return self._video_q

    @property
    def gaze_q(self) -> "queue.Queue[Dict[str, Any]]":
        return self._gaze_q

    @property
    def imu_q(self) -> "queue.Queue[Dict[str, Any]]":
        return self._imu_q

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    # ── internals ─────────────────────────────────────────────────────────

    def _reader_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                env = self._recv_envelope()
                if env is None:
                    break
                hdr, jpeg = env
                etype = hdr.get("type")
                if etype == "handshake":
                    self._handshake = hdr
                    self._handshake_event.set()
                    continue
                if etype != "frame":
                    continue
                # Decode JPEG once; both adapters share the same ndarray.
                bgr = None
                if jpeg:
                    arr = np.frombuffer(jpeg, dtype=np.uint8)
                    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if self._want_frame_bundle:
                    self._put_drop_oldest(self._frame_q, {"hdr": hdr, "bgr": bgr})
                if self._want_device_split:
                    if bgr is not None:
                        ts_video_ns = int(hdr.get("ts_video_ns") or 0)
                        frame_idx = int(hdr.get("frame_idx") or 0)
                        self._put_drop_oldest(
                            self._video_q,
                            (bgr, time.time(), ts_video_ns, frame_idx),
                        )
                    gaze = hdr.get("gaze")
                    if gaze is not None:
                        self._put_drop_oldest(
                            self._gaze_q,
                            {"gaze": gaze, "ts_ns": int(hdr.get("ts_gaze_ns") or 0)},
                        )
                    imu = hdr.get("imu")
                    if imu is not None:
                        self._put_drop_oldest(
                            self._imu_q,
                            {"imu": imu, "ts_ns": int(hdr.get("ts_imu_ns") or 0)},
                        )
        except Exception as e:
            _log(f"reader loop exited: {e}")
        finally:
            self._stop_event.set()
            # Wake any blocked consumers so they can observe shutdown.
            for q_ in (self._frame_q, self._video_q, self._gaze_q, self._imu_q):
                try:
                    q_.put_nowait(None)  # type: ignore[arg-type]
                except queue.Full:
                    pass

    def _recv_envelope(self) -> Optional[Tuple[Dict[str, Any], bytes]]:
        """Read one envelope. Returns None on EOF / shutdown."""
        if self._sock is None:
            return None
        prefix = self._recv_exact(8)
        if prefix is None:
            return None
        json_len, jpeg_len = struct.unpack(">II", prefix)
        # Sanity guard — handshake should be small, frames at q=75 are
        # typically <300 KB. 4 MB is a generous ceiling for both.
        if json_len > 4 * 1024 * 1024 or jpeg_len > 4 * 1024 * 1024:
            _log(f"oversized envelope (json={json_len}, jpeg={jpeg_len}); aborting")
            return None
        json_buf = self._recv_exact(json_len)
        if json_buf is None:
            return None
        jpeg_buf = b""
        if jpeg_len > 0:
            jpeg_buf = self._recv_exact(jpeg_len) or b""
            if not jpeg_buf:
                return None
        try:
            hdr = json.loads(json_buf.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as e:
            _log(f"bad json envelope: {e}")
            return None
        return hdr, jpeg_buf

    def _recv_exact(self, n: int) -> Optional[bytes]:
        sock = self._sock
        if sock is None:
            return None
        chunks = []
        remaining = n
        while remaining > 0:
            try:
                chunk = sock.recv(remaining)
            except OSError:
                return None
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    @staticmethod
    def _put_drop_oldest(q_: "queue.Queue[Any]", item: Any) -> None:
        try:
            q_.put_nowait(item)
        except queue.Full:
            try:
                q_.get_nowait()
            except queue.Empty:
                pass
            try:
                q_.put_nowait(item)
            except queue.Full:
                pass


# ── adapter 1: FrameBundle iterator (vlm_service.py) ───────────────────────


class RemoteFrameReader:
    """Drop-in for ``harmony_vlm.utils.neon.NeonLiveReader``.

    Yields FrameBundle-shaped stub objects via ``__iter__`` and exposes the
    ``camera_matrix`` / ``distortion_coeffs`` / ``width`` / ``height`` /
    ``fps`` properties consumers like ``vlm_service.py`` rely on.
    """

    fps: float = 30.0
    n_frames: int = 0
    width: int = 1600
    height: int = 1200

    def __init__(self, host: str, port: int) -> None:
        self._conn = _RelayConnection(host, port)
        self._conn.enable_frame_bundle()
        self._conn.connect()
        h = self._conn.handshake
        self.fps = float(h.get("fps_video", 30.0))
        self.width = int(h.get("scene_width", 1600))
        self.height = int(h.get("scene_height", 1200))
        K = h.get("camera_matrix")
        self._camera_matrix = (
            np.asarray(K, dtype=np.float64) if K is not None else None
        )
        dist = h.get("distortion_coeffs")
        self._distortion_coeffs = (
            np.asarray(dist, dtype=np.float64).ravel() if dist is not None else None
        )

    @property
    def camera_matrix(self) -> np.ndarray:
        if self._camera_matrix is not None:
            return self._camera_matrix
        return np.array(
            [
                [_DEFAULT_NEON_FX, 0.0, _DEFAULT_NEON_CX],
                [0.0, _DEFAULT_NEON_FY, _DEFAULT_NEON_CY],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @property
    def distortion_coeffs(self) -> Optional[np.ndarray]:
        return self._distortion_coeffs

    def __iter__(self) -> Iterator[_FrameBundleStub]:
        q_ = self._conn.frame_q
        while not self._conn.stopped:
            try:
                item = q_.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            hdr = item["hdr"]
            bgr = item["bgr"]
            if bgr is None:
                continue
            yield _build_bundle_stub(hdr, bgr)

    def close(self) -> None:
        self._conn.close()


def _build_bundle_stub(hdr: Dict[str, Any], bgr: np.ndarray) -> _FrameBundleStub:
    video = _VideoFrameStub(
        timestamp_ns=int(hdr.get("ts_video_ns") or 0),
        frame_idx=int(hdr.get("frame_idx") or 0),
        bgr=bgr,
    )
    gaze_dict = hdr.get("gaze") or {}
    gaze = _GazeStub(gaze_dict, int(hdr.get("ts_gaze_ns") or 0))
    worn = bool(gaze_dict.get("worn", True))
    imu_dict = hdr.get("imu")
    imu_stub: Optional[_IMUSampleStub] = None
    if imu_dict is not None:
        q = imu_dict.get("quaternion") or {}
        accel = imu_dict.get("accel") or {}
        gyro = imu_dict.get("gyro") or {}
        imu_stub = _IMUSampleStub(
            timestamp_ns=int(hdr.get("ts_imu_ns") or 0),
            quaternion=np.array(
                [float(q.get("x", 0.0)), float(q.get("y", 0.0)),
                 float(q.get("z", 0.0)), float(q.get("w", 1.0))],
                dtype=np.float64,
            ),
            accel=np.array(
                [float(accel.get("x", 0.0)), float(accel.get("y", 0.0)),
                 float(accel.get("z", 0.0))], dtype=np.float64,
            ),
            gyro=np.array(
                [float(gyro.get("x", 0.0)), float(gyro.get("y", 0.0)),
                 float(gyro.get("z", 0.0))], dtype=np.float64,
            ),
        )
    return _FrameBundleStub(video=video, gaze=gaze, worn=worn, imu=imu_stub)


# ── adapter 2: pupil_labs Device shim (gaze_system.py) ─────────────────────


class RemoteNeonDevice:
    """Drop-in for ``pupil_labs.realtime_api.simple.Device``.

    Implements the three ``receive_*`` methods that
    ``Utils.gaze.gaze_system.GazeSystem`` consumes from its threads.
    Returns wire-deserialised stubs whose attribute surface matches the
    real Pupil Labs SDK objects closely enough for gaze_system's use.

    Phase 1 limitation: gaze + IMU update at ``relay_hz`` (default 30 Hz,
    matching Neon scene-camera FPS), not native gaze/IMU rates (~200 Hz).
    Smoothing alphas in gaze_system were tuned for higher rates; expect
    coarser behaviour in remote mode.
    """

    def __init__(self, host: str, port: int) -> None:
        self._conn = _RelayConnection(host, port)
        self._conn.enable_device_split()
        self._conn.connect()
        self._handshake = self._conn.handshake

    @property
    def handshake(self) -> Dict[str, Any]:
        return dict(self._handshake)

    @property
    def camera_matrix(self) -> Optional[np.ndarray]:
        K = self._handshake.get("camera_matrix")
        return np.asarray(K, dtype=np.float64) if K is not None else None

    # ── methods consumed by GazeSystem threads ────────────────────────────

    def receive_scene_video_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Block until the next frame arrives. Returns ``(bgr, dt)`` to
        match ``pupil_labs.realtime_api.simple.Device.receive_scene_video_frame``.
        ``dt`` is wall-clock time-since-receive (relative); gaze_system
        only consumes the ``bgr`` array, so the second slot is best-effort.
        """
        while not self._conn.stopped:
            try:
                item = self._conn.video_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                return None, 0.0
            bgr, recv_t, _ts_ns, _idx = item
            return bgr, time.time() - recv_t
        return None, 0.0

    def receive_gaze_datum(self) -> Optional[_GazeStub]:
        while not self._conn.stopped:
            try:
                item = self._conn.gaze_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                return None
            return _GazeStub(item["gaze"], item["ts_ns"])
        return None

    def receive_imu_datum(self) -> Optional[_IMUStub]:
        while not self._conn.stopped:
            try:
                item = self._conn.imu_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                return None
            return _IMUStub(item["imu"], item["ts_ns"])
        return None

    def close(self) -> None:
        self._conn.close()

    def __repr__(self) -> str:  # pragma: no cover — diagnostic
        return f"<RemoteNeonDevice {self._conn.host}:{self._conn.port}>"
