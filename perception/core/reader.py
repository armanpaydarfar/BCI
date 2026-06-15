# Vendored from harmony_vlm (https://github.com/vivianchen98/harmony_vlm) @ cfa01b6
# by Vivian Chen. Folded into the BCI repo for WS3 (2026-06-15). Edit here, not
# upstream; see Documents/SoftwareDocs/projects/harmony-bci/vlm-integration/.
"""Pupil Labs Core eye tracker — live ZMQ reader + intrinsics loader."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ..visualize_neon import EYE_STATE_DTYPE
from ..pupil_reader import FrameBundle, GazeSample, VideoFrame

# ── intrinsics loading ────────────────────────────────────────────────────────

_DEFAULT_INTRINSICS_PATH = Path.home() / "pupil_capture_settings" / "world.intrinsics"


def load_pupil_intrinsics(
    path: str | Path,
    resolution: tuple[int, int] = (1280, 720),
) -> np.ndarray:
    """Load a Pupil Capture ``.intrinsics`` msgpack file and return the 3x3 K matrix.

    Parameters
    ----------
    path       : Path to the ``.intrinsics`` file.
    resolution : Desired ``(width, height)``; used to look up the correct calibration entry.
    """
    import msgpack

    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Pupil intrinsics file not found: {path}")

    with open(path, "rb") as fp:
        data = msgpack.unpack(fp, raw=False)

    res_key = str(resolution)
    entry = data.get(res_key)

    if entry is None:
        for key, val in data.items():
            if isinstance(val, dict) and "camera_matrix" in val:
                entry = val
                print(
                    f"[load_pupil_intrinsics] Exact resolution {resolution} not found, "
                    f"using entry for key={key!r}",
                    flush=True,
                )
                break

    if entry is None:
        raise ValueError(
            f"No calibration entry found in {path}. "
            f"Keys present: {list(data.keys())}"
        )

    return np.array(entry["camera_matrix"], dtype=np.float64)


# ── reader ────────────────────────────────────────────────────────────────────


class PupilCoreReader:
    """
    Streams from Pupil Core via Pupil Capture ZMQ PUB-SUB + msgpack.

    Parameters
    ----------
    host              : Pupil Capture host (default: localhost)
    port              : Pupil Remote REP port (default: 50020)
    camera_intrinsics : optional 3x3 numpy camera matrix; if None, auto-loads
                        from ``~/pupil_capture_settings/world.intrinsics``.
                        Falls back to hardcoded defaults if the file is missing.
    """

    fps      = 30.0
    n_frames = 0
    width    = 1280
    height   = 720

    _DEFAULT_FX = 794.0
    _DEFAULT_FY = 794.0
    _DEFAULT_CX = 640.0
    _DEFAULT_CY = 360.0

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50020,
        camera_intrinsics: np.ndarray | None = None,
    ):
        import zmq
        import msgpack  # noqa: F401

        ctx = zmq.Context()
        req = ctx.socket(zmq.REQ)
        req.connect(f"tcp://{host}:{port}")
        req.send_string("SUB_PORT")
        sub_port = int(req.recv_string())
        req.close()
        print(f"[PupilCoreReader] SUB_PORT={sub_port} on {host}")

        self._sub = ctx.socket(zmq.SUB)
        self._sub.connect(f"tcp://{host}:{sub_port}")
        self._sub.subscribe(b"frame.world")
        self._sub.subscribe(b"gaze")
        self._ctx = ctx

        if camera_intrinsics is None:
            camera_intrinsics = self._try_load_default_intrinsics()
        self._camera_intrinsics = camera_intrinsics

    @staticmethod
    def _try_load_default_intrinsics() -> np.ndarray | None:
        path = _DEFAULT_INTRINSICS_PATH
        if not path.exists():
            print(
                f"[PupilCoreReader] WARNING: {path} not found, using hardcoded defaults. "
                "Run Pupil Capture's Camera Intrinsics Estimation plugin to calibrate.",
                flush=True,
            )
            return None
        try:
            K = load_pupil_intrinsics(path, resolution=(1280, 720))
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            print(
                f"[PupilCoreReader] Loaded intrinsics from {path}: "
                f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}",
                flush=True,
            )
            return K
        except Exception as e:
            print(
                f"[PupilCoreReader] WARNING: Failed to load {path}: {e}. "
                "Using hardcoded defaults.",
                flush=True,
            )
            return None

    @property
    def camera_matrix(self) -> np.ndarray:
        """3x3 camera intrinsic matrix for the Pupil Core world camera."""
        if self._camera_intrinsics is not None:
            return self._camera_intrinsics
        return np.array(
            [
                [self._DEFAULT_FX, 0, self._DEFAULT_CX],
                [0, self._DEFAULT_FY, self._DEFAULT_CY],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    def __iter__(self) -> Iterator[FrameBundle]:
        import zmq
        import msgpack

        stub_eye = np.zeros(1, dtype=EYE_STATE_DTYPE)[0]
        stub_eye["eyelid_aperture_left_mm"]  = 99.0
        stub_eye["eyelid_aperture_right_mm"] = 99.0

        latest_gaze: dict | None = None
        fi = 0

        while True:
            parts = self._sub.recv_multipart()
            topic = parts[0].decode()

            if topic.startswith("gaze"):
                latest_gaze = msgpack.unpackb(parts[1], raw=False)
                continue

            if topic != "frame.world" or latest_gaze is None:
                continue

            latest_frame_parts = parts
            while True:
                try:
                    newer = self._sub.recv_multipart(zmq.NOBLOCK)
                    t = newer[0].decode()
                    if t.startswith("gaze"):
                        latest_gaze = msgpack.unpackb(newer[1], raw=False)
                    elif t == "frame.world":
                        latest_frame_parts = newer
                except zmq.Again:
                    break
            parts = latest_frame_parts

            header = msgpack.unpackb(parts[1], raw=False)
            raw    = parts[2] if len(parts) > 2 else b""
            w      = int(header.get("width",  self.width))
            h      = int(header.get("height", self.height))
            fmt    = header.get("format", "bgr")
            self.width, self.height = w, h

            if fmt == "bgr":
                bgr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
            elif fmt == "jpeg":
                bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                continue

            norm_x, norm_y = latest_gaze["norm_pos"]
            gaze_sample = GazeSample(
                timestamp_ns=int(float(latest_gaze["timestamp"]) * 1e9),
                x=float(norm_x) * w,
                y=(1.0 - float(norm_y)) * h,
                eye_state=stub_eye,
            )
            video_frame = VideoFrame(
                timestamp_ns=int(float(header["timestamp"]) * 1e9),
                frame_idx=fi,
                bgr=bgr,
            )
            yield FrameBundle(video=video_frame, gaze=gaze_sample, worn=True)
            fi += 1

    def close(self) -> None:
        self._sub.close()
        self._ctx.term()
