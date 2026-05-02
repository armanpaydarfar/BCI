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


# Mirror of harmony_vlm.utils.visualize_neon.EYE_STATE_DTYPE so we can build
# a stub eye_state record on the consumer side without importing harmony_vlm.
# harmony_vlm's FixationDetector reads sample.eye_state["eyelid_aperture_*_mm"]
# for blink suppression; NeonLiveReader supplies a stub with both fields set
# to 99 (== "eyes wide open") since the live realtime SDK doesn't surface
# eye_state per frame. We replicate that here so RemoteFrameReader-fed
# consumers behave identically.
_EYE_STATE_DTYPE = np.dtype([
    ("pupil_diameter_left_mm",   "<f4"),
    ("eyeball_center_left_x",    "<f4"),
    ("eyeball_center_left_y",    "<f4"),
    ("eyeball_center_left_z",    "<f4"),
    ("optical_axis_left_x",      "<f4"),
    ("optical_axis_left_y",      "<f4"),
    ("optical_axis_left_z",      "<f4"),
    ("pupil_diameter_right_mm",  "<f4"),
    ("eyeball_center_right_x",   "<f4"),
    ("eyeball_center_right_y",   "<f4"),
    ("eyeball_center_right_z",   "<f4"),
    ("optical_axis_right_x",     "<f4"),
    ("optical_axis_right_y",     "<f4"),
    ("optical_axis_right_z",     "<f4"),
    ("eyelid_angle_top_left",    "<f4"),
    ("eyelid_angle_bottom_left", "<f4"),
    ("eyelid_aperture_left_mm",  "<f4"),
    ("eyelid_angle_top_right",   "<f4"),
    ("eyelid_angle_bottom_right","<f4"),
    ("eyelid_aperture_right_mm", "<f4"),
])


def _build_eye_state(gaze_dict: Dict[str, Any]) -> np.void:
    """Construct a stub eye_state record. Eyelid apertures default to 99 mm
    (no blink) — matches harmony_vlm.utils.neon.NeonLiveReader.__iter__.
    Eyeball-center / optical-axis fields are filled from the relay envelope
    when present, zero otherwise (consumers that need them generally read
    them from the gaze datum directly, not via eye_state)."""
    rec = np.zeros(1, dtype=_EYE_STATE_DTYPE)[0]
    rec["eyelid_aperture_left_mm"]  = 99.0
    rec["eyelid_aperture_right_mm"] = 99.0
    for fname in (
        "eyeball_center_left_x", "eyeball_center_left_y", "eyeball_center_left_z",
        "eyeball_center_right_x", "eyeball_center_right_y", "eyeball_center_right_z",
        "optical_axis_left_x", "optical_axis_left_y", "optical_axis_left_z",
        "optical_axis_right_x", "optical_axis_right_y", "optical_axis_right_z",
    ):
        v = gaze_dict.get(fname)
        if v is not None:
            try:
                rec[fname] = float(v)
            except (TypeError, ValueError):
                pass
    return rec


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
        # harmony_vlm's GazeSample (pupil_reader.py) carries timestamp_ns;
        # FixationDetector reads sample.timestamp_ns for fixation duration.
        # Source the value from the relay's ts_gaze_ns envelope field; if
        # missing, derive from timestamp_unix_seconds.
        self.timestamp_ns = int(ts_ns) if ts_ns else int(self.timestamp_unix_seconds * 1e9)
        # gaze_system reads these via getattr; populating dynamically keeps
        # the stub permissive when the relay omits them on older firmware.
        for k, v in gaze_dict.items():
            if k in ("x", "y", "worn", "timestamp_unix_seconds"):
                continue
            try:
                setattr(self, k, float(v))
            except (TypeError, ValueError):
                setattr(self, k, v)
        # harmony_vlm's FixationDetector reads sample.eye_state[...] for
        # blink suppression; supply a record with eyelid_aperture=99 so it
        # behaves the same as a live NeonLiveReader-fed sample.
        self.eye_state = _build_eye_state(gaze_dict)

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

    Connection lifecycle is managed by a background thread:
      - ``connect()`` returns immediately and starts the manager.
      - The manager dials the relay with retry+backoff (default infinite)
        so Windows services can be started before the Linux relay comes
        up, then continue waiting until it does.
      - On EOF/error after a successful handshake, the manager reconnects
        automatically (same backoff) — Wi-Fi blips and relay restarts
        survive without taking the perception services down.
      - Consumers (RemoteFrameReader.__iter__,
        RemoteNeonDevice.receive_*) loop on the queues and tolerate
        empty stretches; they only exit when the user calls ``close()``.
    """

    # Backoff bounds for connect retries.
    _BACKOFF_INITIAL_S = 0.5
    _BACKOFF_MAX_S = 10.0

    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_timeout_s: float = 5.0,
        auto_reconnect: bool = True,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.connect_timeout_s = float(connect_timeout_s)
        self.auto_reconnect = bool(auto_reconnect)

        self._sock: Optional[socket.socket] = None
        self._sock_lock = threading.Lock()

        # `_user_closed` is set only by close(); the manager and consumers
        # use it as the sole exit condition. Don't conflate it with
        # transient disconnects.
        self._user_closed = threading.Event()
        # `_paused` is set by pause() and cleared by resume(). When set,
        # the manager loop tears down any active connection and parks
        # until cleared. Lets the operator-side panel toggle the relay
        # client off without exiting the service — used to enforce
        # single-active-backend end-to-end via GAZE_OR_BACKEND.
        self._paused = threading.Event()
        # `_connected` reflects whether the wire is currently up AND the
        # most recent handshake has arrived.
        self._connected = threading.Event()
        self._handshake: Dict[str, Any] = {}
        self._handshake_lock = threading.Lock()
        self._mgr_thread: Optional[threading.Thread] = None

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

    def connect(self, *, wait_for_handshake_s: float = 0.0) -> None:
        """Start the connection manager. Returns immediately so service
        startup can proceed in parallel with the dial.

        Set ``wait_for_handshake_s > 0`` to block up to that many seconds
        for the first handshake (handy for one-shot probes / tests).
        Production startup paths should leave this 0 and rely on
        consumer-side queue blocking, which tolerates the wait gracefully.
        """
        if self._mgr_thread is not None:
            return
        self._mgr_thread = threading.Thread(
            target=self._mgr_loop, daemon=True, name="relay-mgr"
        )
        self._mgr_thread.start()
        if wait_for_handshake_s > 0:
            self.wait_connected(timeout=wait_for_handshake_s)

    def wait_connected(self, *, timeout: Optional[float] = None) -> bool:
        """Block until the first (or next) handshake lands. Returns True
        if connected, False on timeout or close()."""
        return self._connected.wait(timeout=timeout)

    def close(self) -> None:
        self._user_closed.set()
        # Drop any cached socket so the read loop unblocks promptly.
        with self._sock_lock:
            sock = self._sock
            self._sock = None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        # Wake any blocked consumers so they exit their loops.
        for q_ in (self._frame_q, self._video_q, self._gaze_q, self._imu_q):
            try:
                q_.put_nowait(None)  # type: ignore[arg-type]
            except queue.Full:
                pass

    def pause(self) -> None:
        """Tear down any active TCP connection and stop the manager loop
        from reconnecting until resume() is called. Idempotent.

        Used by the panel to enforce single-active-backend end-to-end:
        when GAZE_OR_BACKEND switches from one service to the other,
        the inactive service's intake is paused so it stops consuming
        relay bandwidth on the GPU host. The service stays alive — its
        UDP request-reply surface keeps responding — only the frame
        wire is torn down.
        """
        self._paused.set()
        with self._sock_lock:
            sock = self._sock
            self._sock = None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    def resume(self) -> None:
        """Allow the manager loop to reconnect to the relay. Idempotent."""
        self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    # ── subscription markers ──────────────────────────────────────────────

    def enable_frame_bundle(self) -> None:
        self._want_frame_bundle = True

    def enable_device_split(self) -> None:
        self._want_device_split = True

    @property
    def handshake(self) -> Dict[str, Any]:
        with self._handshake_lock:
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
    def user_closed(self) -> bool:
        return self._user_closed.is_set()

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    # `stopped` is preserved for any external caller that still reads it,
    # but its meaning is now "user closed", not "wire disconnected".
    @property
    def stopped(self) -> bool:
        return self._user_closed.is_set()

    # ── internals ─────────────────────────────────────────────────────────

    def _mgr_loop(self) -> None:
        """Connect → read until EOF → backoff → retry, until close().
        When `_paused` is set, parks here until cleared."""
        backoff = self._BACKOFF_INITIAL_S
        attempt = 0
        while not self._user_closed.is_set():
            # Park while paused. Resume restarts the connect cycle from
            # a clean attempt count and reset backoff.
            if self._paused.is_set():
                while self._paused.is_set() and not self._user_closed.is_set():
                    self._user_closed.wait(timeout=0.5)
                if self._user_closed.is_set():
                    break
                attempt = 0
                backoff = self._BACKOFF_INITIAL_S
                _log("intake resumed; reconnecting to relay")
                continue
            attempt += 1
            sock: Optional[socket.socket] = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.connect_timeout_s)
                _log(
                    f"connecting to relay tcp://{self.host}:{self.port} "
                    f"(attempt {attempt})…"
                )
                sock.connect((self.host, self.port))
                sock.settimeout(None)
                with self._sock_lock:
                    if self._user_closed.is_set():
                        sock.close()
                        return
                    self._sock = sock
                # Successful TCP — reset backoff for the next reconnect cycle.
                backoff = self._BACKOFF_INITIAL_S
                self._read_until_eof(sock)
                # _read_until_eof returns when EOF / IO error / close.
            except (ConnectionRefusedError, ConnectionAbortedError,
                    ConnectionResetError, socket.timeout, OSError) as e:
                _log(f"connect/read failed ({e}); retry in {backoff:.1f} s")
            finally:
                self._connected.clear()
                with self._sock_lock:
                    if self._sock is sock:
                        self._sock = None
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass

            if self._user_closed.is_set():
                break
            if not self.auto_reconnect:
                # Manager is the sole owner of the wire; if reconnect is
                # off and we just dropped, treat the connection as
                # permanently dead so consumer iters terminate (matches
                # one-shot test semantics).
                self._user_closed.set()
                break
            self._user_closed.wait(timeout=backoff)
            backoff = min(backoff * 1.5, self._BACKOFF_MAX_S)

        # Final shutdown: wake any consumers still parked on a queue.
        for q_ in (self._frame_q, self._video_q, self._gaze_q, self._imu_q):
            try:
                q_.put_nowait(None)  # type: ignore[arg-type]
            except queue.Full:
                pass

    def _read_until_eof(self, sock: socket.socket) -> None:
        """Parse envelopes off `sock` until EOF. First envelope must be a
        handshake; subsequent ones are frames."""
        try:
            while not self._user_closed.is_set():
                env = self._recv_envelope(sock)
                if env is None:
                    return  # EOF or socket error → caller decides reconnect
                hdr, jpeg = env
                etype = hdr.get("type")
                if etype == "handshake":
                    with self._handshake_lock:
                        self._handshake = hdr
                    self._connected.set()
                    _log("handshake received")
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
        except OSError:
            return
        except Exception as e:
            _log(f"read loop unexpected error: {e}")
            return

    def _recv_envelope(self, sock: socket.socket) -> Optional[Tuple[Dict[str, Any], bytes]]:
        """Read one envelope. Returns None on EOF / shutdown."""
        prefix = self._recv_exact(sock, 8)
        if prefix is None:
            return None
        json_len, jpeg_len = struct.unpack(">II", prefix)
        # Sanity guard — handshake should be small, frames at q=75 are
        # typically <300 KB. 4 MB is a generous ceiling for both.
        if json_len > 4 * 1024 * 1024 or jpeg_len > 4 * 1024 * 1024:
            _log(f"oversized envelope (json={json_len}, jpeg={jpeg_len}); aborting")
            return None
        json_buf = self._recv_exact(sock, json_len)
        if json_buf is None:
            return None
        jpeg_buf = b""
        if jpeg_len > 0:
            jpeg_buf = self._recv_exact(sock, jpeg_len) or b""
            if not jpeg_buf:
                return None
        try:
            hdr = json.loads(json_buf.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as e:
            _log(f"bad json envelope: {e}")
            return None
        return hdr, jpeg_buf

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
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

    def __init__(
        self,
        host: str,
        port: int,
        *,
        wait_for_handshake_s: float = 0.0,
        auto_reconnect: bool = True,
    ) -> None:
        self._conn = _RelayConnection(host, port, auto_reconnect=auto_reconnect)
        self._conn.enable_frame_bundle()
        self._conn.connect(wait_for_handshake_s=wait_for_handshake_s)
        # Initialise from whatever's in the handshake (may be empty if the
        # caller didn't wait — properties below re-read on every access).
        self._refresh_metadata()

    def _refresh_metadata(self) -> None:
        h = self._conn.handshake
        if h:
            self.fps = float(h.get("fps_video", 30.0))
            self.width = int(h.get("scene_width", 1600))
            self.height = int(h.get("scene_height", 1200))

    @property
    def camera_matrix(self) -> np.ndarray:
        K = self._conn.handshake.get("camera_matrix")
        if K is not None:
            return np.asarray(K, dtype=np.float64)
        # Pre-handshake (relay not yet up): return Neon defaults so callers
        # like Depth Pro's _focal_px don't crash; the real intrinsics
        # replace these as soon as the first handshake lands.
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
        dist = self._conn.handshake.get("distortion_coeffs")
        if dist is None:
            return None
        return np.asarray(dist, dtype=np.float64).ravel()

    def __iter__(self) -> Iterator[_FrameBundleStub]:
        """Yield FrameBundles forever, surviving any number of relay
        disconnect/reconnect cycles. Exits only when ``close()`` is called."""
        q_ = self._conn.frame_q
        while not self._conn.user_closed:
            try:
                item = q_.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                # Sentinel only published on close(); any other gap (relay
                # outage) is just an empty period — keep looping.
                if self._conn.user_closed:
                    break
                continue
            hdr = item["hdr"]
            bgr = item["bgr"]
            if bgr is None:
                continue
            # Refresh cached metadata in case a reconnect handshake
            # changed dimensions / intrinsics.
            self._refresh_metadata()
            yield _build_bundle_stub(hdr, bgr)

    def close(self) -> None:
        self._conn.close()

    def pause(self) -> None:
        self._conn.pause()

    def resume(self) -> None:
        self._conn.resume()

    @property
    def paused(self) -> bool:
        return self._conn.paused


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

    def __init__(
        self,
        host: str,
        port: int,
        *,
        wait_for_handshake_s: float = 0.0,
        auto_reconnect: bool = True,
    ) -> None:
        self._conn = _RelayConnection(host, port, auto_reconnect=auto_reconnect)
        self._conn.enable_device_split()
        self._conn.connect(wait_for_handshake_s=wait_for_handshake_s)

    @property
    def handshake(self) -> Dict[str, Any]:
        return self._conn.handshake

    @property
    def camera_matrix(self) -> Optional[np.ndarray]:
        K = self._conn.handshake.get("camera_matrix")
        return np.asarray(K, dtype=np.float64) if K is not None else None

    # ── methods consumed by GazeSystem threads ────────────────────────────

    def receive_scene_video_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Block until the next frame arrives. Returns ``(bgr, dt)`` to
        match ``pupil_labs.realtime_api.simple.Device.receive_scene_video_frame``.
        Survives relay outages — keeps waiting on the queue until either
        a frame arrives or ``close()`` is called. ``dt`` is wall-clock
        time-since-receive; gaze_system only consumes the ``bgr`` array."""
        while not self._conn.user_closed:
            try:
                item = self._conn.video_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                if self._conn.user_closed:
                    return None, 0.0
                continue
            bgr, recv_t, _ts_ns, _idx = item
            return bgr, time.time() - recv_t
        return None, 0.0

    def receive_gaze_datum(self) -> Optional[_GazeStub]:
        while not self._conn.user_closed:
            try:
                item = self._conn.gaze_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                if self._conn.user_closed:
                    return None
                continue
            return _GazeStub(item["gaze"], item["ts_ns"])
        return None

    def receive_imu_datum(self) -> Optional[_IMUStub]:
        while not self._conn.user_closed:
            try:
                item = self._conn.imu_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                if self._conn.user_closed:
                    return None
                continue
            return _IMUStub(item["imu"], item["ts_ns"])
        return None

    def close(self) -> None:
        self._conn.close()

    def pause(self) -> None:
        self._conn.pause()

    def resume(self) -> None:
        self._conn.resume()

    @property
    def paused(self) -> bool:
        return self._conn.paused

    def __repr__(self) -> str:  # pragma: no cover — diagnostic
        return f"<RemoteNeonDevice {self._conn.host}:{self._conn.port}>"
