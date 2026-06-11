"""
scene_only_neon_reader.py — direct ``device.receive_scene_video_frame()``
bundle source, drop-in for ``harmony_vlm.utils.neon.NeonLiveReader``.

Why this exists
---------------
``NeonLiveReader.__iter__`` (harmony_vlm/utils/neon/reader.py:309) calls
``device.receive_matched_scene_and_eyes_video_frames_and_gaze()`` —
the SDK's "give me time-aligned scene+eyes+gaze" path. Compared to
``device.receive_scene_video_frame()`` (used by ``neon_viewer.py:128``
and other simple Pupil Labs sample code) the matched call appears to
deliver a visibly grainier scene image on the panel's display, even
though both reach the same RTSP scene track underneath. The matched
path also has internal queueing/skip semantics that compound the
problem under load.

This module provides a slim alternative that mirrors neon_viewer.py:
one ``device.receive_scene_video_frame()`` per iteration, plus a
best-effort ``device.receive_gaze_datum()`` for the gaze cursor and
a non-blocking ``device.receive_imu_datum()`` for downstream code
that needs head-pose fields. The output bundle has the same attribute
shape ``Utils/frame_relay.py:_pump_loop`` and
``Utils/frame_relay.py:_encode_frame`` consume from NeonLiveReader,
so the relay's TCP wire and Windows-side ``RemoteFrameReader`` are
unaffected.

Tradeoff: gaze and scene aren't time-matched here — the iteration
takes the latest of each at the moment ``receive_*`` returns. Gaze
is ~200 Hz from the Companion phone; scene is ~30 Hz. The temporal
mismatch is at most one scene-frame period (~33 ms), which is below
human gaze-saccade latency. Acceptable for both display and any
downstream model code that already tolerates per-frame jitter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

import numpy as np


_DEFAULT_NEON_FX = 800.0
_DEFAULT_NEON_FY = 800.0
_DEFAULT_NEON_CX = 800.0
_DEFAULT_NEON_CY = 600.0


def _default_log_callback(line: str) -> None:
    print(line, flush=True)


# Module-level sink for log lines. control_panel swaps this so reader
# events ("connecting to <host>… connected to Device(...) … factory
# intrinsics …") land in the panel's "Relay" channel alongside the
# frame_relay output — the reader is the upstream half of the same
# perception pipeline, so co-locating their logs makes triage easier.
# Standalone usage keeps the default print sink.
_LOG_CALLBACK: Callable[[str], None] = _default_log_callback


def set_log_callback(fn: Optional[Callable[[str], None]]) -> None:
    """Install a custom sink for ``_log`` lines. ``None`` restores the
    stdout default. Lines arrive already prefixed with
    ``[scene_only_neon_reader] ``.
    """
    global _LOG_CALLBACK
    _LOG_CALLBACK = fn if fn is not None else _default_log_callback


def _log(msg: str) -> None:
    _LOG_CALLBACK(f"[scene_only_neon_reader] {msg}")


# ── bundle stub types (mirror harmony_vlm/utils/pupil_reader shapes) ──────


@dataclass
class _VideoFrame:
    timestamp_ns: int
    frame_idx: int
    bgr: np.ndarray


class _Gaze:
    """Mirrors the relevant attribute surface of harmony_vlm.GazeSample +
    pupil_labs gaze datum so Utils/frame_relay._gaze_to_dict and the
    panel's renderer don't need to care which reader produced it."""

    __slots__ = ("__dict__",)

    def __init__(self, datum, ts_ns_fallback: int) -> None:
        if datum is None:
            self.x = float("nan")
            self.y = float("nan")
            self.worn = False
            self.timestamp_unix_seconds = 0.0
            self.timestamp_ns = int(ts_ns_fallback)
            return
        self.x = float(getattr(datum, "x", float("nan")))
        self.y = float(getattr(datum, "y", float("nan")))
        self.worn = bool(getattr(datum, "worn", True))
        ts = getattr(datum, "timestamp_unix_seconds", None)
        if ts is None:
            ts = float(ts_ns_fallback) / 1e9 if ts_ns_fallback else 0.0
        self.timestamp_unix_seconds = float(ts)
        self.timestamp_ns = int(self.timestamp_unix_seconds * 1e9)
        # gaze_system.py and the relay's _gaze_to_dict probe these via
        # getattr; populate them when present so downstream code that
        # needs vergence depth or head-pose deltas gets the same data
        # NeonLiveReader would have surfaced.
        for fname in (
            "eyeball_center_left_x", "eyeball_center_left_y", "eyeball_center_left_z",
            "eyeball_center_right_x", "eyeball_center_right_y", "eyeball_center_right_z",
            "optical_axis_left_x", "optical_axis_left_y", "optical_axis_left_z",
            "optical_axis_right_x", "optical_axis_right_y", "optical_axis_right_z",
            "pupil_diameter_left", "pupil_diameter_right",
        ):
            v = getattr(datum, fname, None)
            if v is not None:
                try:
                    setattr(self, fname, float(v))
                except (TypeError, ValueError):
                    pass


@dataclass
class _IMU:
    timestamp_ns: int
    quaternion: np.ndarray  # (x, y, z, w)
    accel: np.ndarray
    gyro: np.ndarray


@dataclass
class _Bundle:
    video: _VideoFrame
    gaze: _Gaze
    worn: bool
    imu: Optional[_IMU]


# ── reader ────────────────────────────────────────────────────────────────


class SceneOnlyNeonReader:
    """Drop-in for ``harmony_vlm.utils.neon.NeonLiveReader``. Same
    public surface (``__iter__``, ``camera_matrix``, ``distortion_coeffs``,
    ``width``, ``height``, ``fps``, ``close``); different SDK call.
    """

    fps: float = 30.0
    n_frames: int = 0
    width: int = 1600
    height: int = 1200

    def __init__(self, host: Optional[str] = None, *,
                 max_search_s: float = 10.0) -> None:
        if host:
            from pupil_labs.realtime_api.simple import Device
            _log(f"connecting to {host}…")
            self._device = Device(address=host, port=8080)
        else:
            from pupil_labs.realtime_api.simple import discover_one_device
            _log("discovering Neon on LAN…")
            self._device = discover_one_device(max_search_duration_seconds=max_search_s)
        if self._device is None:
            raise RuntimeError("No Neon device found. Is the Companion app running?")
        _log(f"connected to {self._device}")

        self._distortion_coeffs: Optional[np.ndarray] = None
        self._camera_intrinsics = self._load_factory_calibration()

    # ── calibration / metadata ────────────────────────────────────────────

    def _load_factory_calibration(self) -> Optional[np.ndarray]:
        import time as _t
        for attempt in range(3):
            try:
                cal = self._device.get_calibration()
                K = np.array(cal["scene_camera_matrix"], dtype=np.float64)
                if K.ndim == 1:
                    K = K.reshape(3, 3)
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                _log(f"factory intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
                # pupil_labs returns a numpy record (structured-array element),
                # not a dict — bracket access works on both, .get() doesn't.
                # See harmony_vlm/utils/neon/reader.py:252 for the same pattern.
                try:
                    dist = cal["scene_distortion_coefficients"]
                except (KeyError, ValueError, IndexError):
                    dist = None
                if dist is not None:
                    self._distortion_coeffs = np.array(dist, dtype=np.float64).ravel()
                    _log(f"distortion coeffs: {self._distortion_coeffs}")
                return K
            except Exception as e:
                if attempt < 2:
                    _log(f"calibration attempt {attempt+1} failed: {e}; retrying")
                    _t.sleep(1)
                else:
                    _log(f"calibration unavailable after 3 attempts: {e}; using defaults")
                    return None

    @property
    def camera_matrix(self) -> np.ndarray:
        if self._camera_intrinsics is not None:
            return self._camera_intrinsics
        return np.array(
            [[_DEFAULT_NEON_FX, 0.0, _DEFAULT_NEON_CX],
             [0.0, _DEFAULT_NEON_FY, _DEFAULT_NEON_CY],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def distortion_coeffs(self) -> Optional[np.ndarray]:
        return self._distortion_coeffs

    # ── iteration ─────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[_Bundle]:
        """Mirrors neon_viewer.py:128 — one receive_scene_video_frame per
        loop, paired with the latest gaze + IMU. The matched-API queueing
        that turned NeonLiveReader's output grainy is bypassed entirely."""
        import time as _t
        fi = 0
        while True:
            bgr_and_dt = self._device.receive_scene_video_frame()
            if bgr_and_dt is None:
                continue
            bgr, _frame_dt = bgr_and_dt
            if bgr is None:
                continue
            ts_ns_video = int(_t.time() * 1e9)

            try:
                gaze = self._device.receive_gaze_datum()
            except Exception:
                gaze = None
            gaze_stub = _Gaze(gaze, ts_ns_video)

            imu_stub: Optional[_IMU] = None
            try:
                raw = self._device.receive_imu_datum()
                if raw is not None:
                    q = raw.quaternion
                    g = raw.gyro_data
                    a = raw.accel_data
                    imu_stub = _IMU(
                        timestamp_ns=int(getattr(raw, "timestamp_unix_seconds", 0.0) * 1e9),
                        quaternion=np.array([q.x, q.y, q.z, q.w], dtype=np.float64),
                        accel=np.array([a.x, a.y, a.z], dtype=np.float64),
                        gyro=np.array([g.x, g.y, g.z], dtype=np.float64),
                    )
            except Exception:
                pass

            yield _Bundle(
                video=_VideoFrame(timestamp_ns=ts_ns_video, frame_idx=fi, bgr=bgr),
                gaze=gaze_stub,
                worn=bool(gaze_stub.worn),
                imu=imu_stub,
            )
            fi += 1

    def close(self) -> None:
        try:
            self._device.close()
        except Exception:
            pass
