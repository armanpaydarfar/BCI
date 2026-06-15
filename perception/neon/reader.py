# Vendored from harmony_vlm (https://github.com/vivianchen98/harmony_vlm) @ cfa01b6
# by Vivian Chen. Folded into the BCI repo for WS3 (2026-06-15). Edit here, not
# upstream; see Documents/SoftwareDocs/projects/harmony-bci/vlm-integration/.
"""Pupil Labs Neon eye tracker — live and recording readers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ..visualize_neon import (
    GAZE_DTYPE,
    EYE_STATE_DTYPE,
    load_stream,
    load_video_timestamps,
    nearest_idx,
)
from ..pupil_reader import FrameBundle, GazeSample, IMUSample, VideoFrame

# Default Neon scene camera intrinsics (1600x1200, approximate)
_DEFAULT_NEON_FX = 800.0
_DEFAULT_NEON_FY = 800.0
_DEFAULT_NEON_CX = 800.0
_DEFAULT_NEON_CY = 600.0

WORN_DTYPE = np.dtype([("worn", "u1")])


def _load_gaze_calibration(path: Path | None) -> tuple[float, float]:
    """Load pixel offset (dx, dy) from a gaze_calibration.json file.

    Returns (0.0, 0.0) if path is None or file missing.
    """
    if path is None:
        return (0.0, 0.0)
    path = Path(path)
    if not path.exists():
        print(f"[Reader] Gaze calibration file not found: {path}", flush=True)
        return (0.0, 0.0)
    try:
        data = json.loads(path.read_text())
        dx, dy = data["gaze_offset_px"]
        print(
            f"[Reader] Gaze calibration loaded: dx={dx:.1f} dy={dy:.1f} px "
            f"(from {path})",
            flush=True,
        )
        return (float(dx), float(dy))
    except Exception as e:
        print(f"[Reader] WARNING: Failed to load gaze calibration: {e}", flush=True)
        return (0.0, 0.0)


# ── recording reader ──────────────────────────────────────────────────────────


class RecordingReader:
    """
    Loads all streams from a Neon recording into RAM, then yields FrameBundle
    objects (video + nearest gaze/eye-state + worn flag).

    Parameters
    ----------
    rec_dir   : path to a recording folder (e.g. neon/QuickShare_2603111547)
    real_time : if True, sleep between frames to honour wall-clock pace
    """

    def __init__(self, rec_dir: str | Path, *, real_time: bool = False,
                 gaze_calibration: str | Path | None = None):
        self.rec_dir   = Path(rec_dir)
        self.real_time = real_time
        self._gaze_offset = _load_gaze_calibration(gaze_calibration)

        # ── load streams ──────────────────────────────────────────────────
        self.gaze_data, self.gaze_ts = load_stream(
            self.rec_dir, "gaze", GAZE_DTYPE
        )
        self.eye_data, self.eye_ts = load_stream(
            self.rec_dir, "eye_state", EYE_STATE_DTYPE
        )

        worn_raw = (self.rec_dir / "worn ps1.raw").read_bytes()
        self.worn_data = np.frombuffer(worn_raw, dtype=np.uint8)
        min_len = min(len(self.worn_data), len(self.gaze_ts))
        self.worn_data = self.worn_data[:min_len]

        self.frame_ts = load_video_timestamps(self.rec_dir)
        self.video_path = self.rec_dir / "Neon Scene Camera v1 ps1.mp4"

        cap = cv2.VideoCapture(str(self.video_path))
        self.fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        self._camera_intrinsics = self._load_intrinsics()

    def _load_intrinsics(self) -> np.ndarray | None:
        """Try to load intrinsics from scene_camera.json in the recording dir."""
        json_path = self.rec_dir / "scene_camera.json"
        if not json_path.exists():
            print(
                f"[RecordingReader] No scene_camera.json found in {self.rec_dir}, "
                "using defaults.",
                flush=True,
            )
            return None
        try:
            data = json.loads(json_path.read_text())
            K = np.array(data["scene_camera_matrix"][0], dtype=np.float64)
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            print(
                f"[RecordingReader] Loaded intrinsics from {json_path}: "
                f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}",
                flush=True,
            )
            return K
        except Exception as e:
            print(
                f"[RecordingReader] WARNING: Failed to load {json_path}: {e}. "
                "Using defaults.",
                flush=True,
            )
            return None

    @property
    def camera_matrix(self) -> np.ndarray:
        """3x3 camera intrinsic matrix for the Neon scene camera."""
        if self._camera_intrinsics is not None:
            return self._camera_intrinsics
        return np.array(
            [
                [_DEFAULT_NEON_FX, 0, _DEFAULT_NEON_CX],
                [0, _DEFAULT_NEON_FY, _DEFAULT_NEON_CY],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    @property
    def gaze_offset(self) -> tuple[float, float]:
        """Current gaze pixel offset (dx, dy)."""
        return self._gaze_offset

    @gaze_offset.setter
    def gaze_offset(self, value: tuple[float, float]) -> None:
        self._gaze_offset = value

    def __iter__(self) -> Iterator[FrameBundle]:
        cap = cv2.VideoCapture(str(self.video_path))
        try:
            frame_period_s = 1.0 / self.fps
            t_start_wall   = time.monotonic()
            ts_start       = self.frame_ts[0] if len(self.frame_ts) else 0

            for fi in range(self.n_frames):
                ok, bgr = cap.read()
                if not ok:
                    break

                fts = self.frame_ts[fi] if fi < len(self.frame_ts) else self.frame_ts[-1]

                g_idx = nearest_idx(self.gaze_ts, fts)
                g_row = self.gaze_data[g_idx]
                e_row = self.eye_data[g_idx]
                w_val = bool(self.worn_data[g_idx]) if g_idx < len(self.worn_data) else True

                gaze_sample = GazeSample(
                    timestamp_ns=int(self.gaze_ts[g_idx]),
                    x=float(g_row["x"]) - self._gaze_offset[0],
                    y=float(g_row["y"]) - self._gaze_offset[1],
                    eye_state=e_row,
                )
                video_frame = VideoFrame(
                    timestamp_ns=int(fts),
                    frame_idx=fi,
                    bgr=bgr,
                )

                yield FrameBundle(video=video_frame, gaze=gaze_sample, worn=w_val)

                if self.real_time:
                    elapsed_rec_s = (fts - ts_start) / 1e9
                    elapsed_wall  = time.monotonic() - t_start_wall
                    sleep_s = elapsed_rec_s - elapsed_wall
                    if sleep_s > 0:
                        time.sleep(sleep_s)
        finally:
            cap.release()


# ── live reader ───────────────────────────────────────────────────────────────


class NeonLiveReader:
    """
    Streams from a Pupil Labs Neon device via the realtime API.

    Factory-calibrated intrinsics are fetched from the device automatically
    via ``device.get_calibration()``.

    Parameters
    ----------
    host         : Companion app IP; if None, auto-discovers on LAN
    max_search_s : seconds to wait during discovery
    """

    fps      = 30.0
    n_frames = 0
    width    = 1600
    height   = 1200

    def __init__(self, host: str | None = None, max_search_s: float = 10.0,
                 gaze_calibration: str | Path | None = None):
        self._gaze_offset = _load_gaze_calibration(gaze_calibration)
        if host:
            from pupil_labs.realtime_api.simple import Device
            self._device = Device(address=host, port=8080)
            print(f"[NeonLiveReader] Connecting to {host}…")
        else:
            from pupil_labs.realtime_api.simple import discover_one_device
            print("[NeonLiveReader] Searching for device on network…")
            self._device = discover_one_device(max_search_duration_seconds=max_search_s)
        if self._device is None:
            raise RuntimeError("No Neon device found. Is the Companion app running?")
        print(f"[NeonLiveReader] Connected to {self._device}")

        self._distortion_coeffs: np.ndarray | None = None
        self._camera_intrinsics = self._load_factory_calibration()

    def _load_factory_calibration(self) -> np.ndarray | None:
        """Fetch factory-calibrated intrinsics and distortion from the device."""
        import time as _time

        for attempt in range(3):
            try:
                calibration = self._device.get_calibration()
                K = np.array(calibration["scene_camera_matrix"], dtype=np.float64)
                if K.ndim == 1:
                    K = K.reshape(3, 3)
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                print(
                    f"[NeonLiveReader] Factory intrinsics: "
                    f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}",
                    flush=True,
                )
                dist = calibration["scene_distortion_coefficients"]
                if dist is not None:
                    self._distortion_coeffs = np.array(dist, dtype=np.float64).ravel()
                    print(
                        f"[NeonLiveReader] Distortion coeffs: {self._distortion_coeffs}",
                        flush=True,
                    )
                return K
            except Exception as e:
                if attempt < 2:
                    print(
                        f"[NeonLiveReader] Calibration attempt {attempt+1} failed: {e}. Retrying...",
                        flush=True,
                    )
                    _time.sleep(1)
                else:
                    print(
                        f"[NeonLiveReader] WARNING: Failed to get calibration after 3 attempts: {e}. "
                        "Using defaults.",
                        flush=True,
                    )
                    return None

    @property
    def distortion_coeffs(self) -> np.ndarray | None:
        """Distortion coefficients from factory calibration, or None."""
        return self._distortion_coeffs

    @property
    def camera_matrix(self) -> np.ndarray:
        """3x3 camera intrinsic matrix for the Neon scene camera."""
        if self._camera_intrinsics is not None:
            return self._camera_intrinsics
        return np.array(
            [
                [_DEFAULT_NEON_FX, 0, _DEFAULT_NEON_CX],
                [0, _DEFAULT_NEON_FY, _DEFAULT_NEON_CY],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    @property
    def gaze_offset(self) -> tuple[float, float]:
        """Current gaze pixel offset (dx, dy)."""
        return self._gaze_offset

    @gaze_offset.setter
    def gaze_offset(self, value: tuple[float, float]) -> None:
        self._gaze_offset = value

    def __iter__(self) -> Iterator[FrameBundle]:
        stub_eye = np.zeros(1, dtype=EYE_STATE_DTYPE)[0]
        stub_eye["eyelid_aperture_left_mm"]  = 99.0
        stub_eye["eyelid_aperture_right_mm"] = 99.0
        fi = 0
        while True:
            matched = self._device.receive_matched_scene_and_eyes_video_frames_and_gaze()
            if matched is None:
                continue

            # Fetch latest IMU datum (non-blocking best-effort)
            imu_sample = None
            try:
                imu_raw = self._device.receive_imu_datum()
                if imu_raw is not None:
                    q = imu_raw.quaternion
                    a = imu_raw.accel_data
                    g = imu_raw.gyro_data
                    imu_sample = IMUSample(
                        timestamp_ns=int(imu_raw.timestamp_unix_seconds * 1e9),
                        quaternion=np.array([q.x, q.y, q.z, q.w], dtype=np.float64),
                        accel=np.array([a.x, a.y, a.z], dtype=np.float64),
                        gyro=np.array([g.x, g.y, g.z], dtype=np.float64),
                    )
            except Exception:
                pass  # IMU not available on older firmware

            gaze_sample = GazeSample(
                timestamp_ns=int(matched.gaze.timestamp_unix_ns),
                x=float(matched.gaze.x) - self._gaze_offset[0],
                y=float(matched.gaze.y) - self._gaze_offset[1],
                eye_state=stub_eye,
            )
            video_frame = VideoFrame(
                timestamp_ns=int(matched.scene.timestamp_unix_ns),
                frame_idx=fi,
                bgr=matched.scene.bgr_pixels.copy(),
            )
            worn = bool(getattr(matched.gaze, "worn", True))
            yield FrameBundle(video=video_frame, gaze=gaze_sample, worn=worn, imu=imu_sample)
            fi += 1

    def close(self) -> None:
        if self._device is not None:
            self._device.close()
