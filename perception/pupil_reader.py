# Vendored from harmony_vlm (https://github.com/vivianchen98/harmony_vlm) @ cfa01b6
# by Vivian Chen. Folded into the BCI repo for WS3 (2026-06-15). Edit here, not
# upstream; see Documents/SoftwareDocs/projects/harmony-bci/vlm-integration/.
"""
pupil_reader.py — Shared data structures for Pupil Labs eye tracker readers.

Defines the common FrameBundle / GazeSample / VideoFrame dataclasses used by
both ``utils.core.PupilCoreReader`` and ``utils.neon.NeonLiveReader``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .visualize_neon import EYE_STATE_DTYPE  # noqa: F401 — re-export


# ── shared dataclasses ────────────────────────────────────────────────────────


@dataclass
class IMUSample:
    """IMU reading from Neon (quaternion + accelerometer + gyroscope)."""
    timestamp_ns: int
    quaternion: np.ndarray  # (x, y, z, w)
    accel: np.ndarray       # (x, y, z) in m/s²
    gyro: np.ndarray        # (x, y, z) in deg/s


@dataclass
class GazeSample:
    timestamp_ns: int
    x: float
    y: float
    eye_state: np.void  # one row of EYE_STATE_DTYPE


@dataclass
class VideoFrame:
    timestamp_ns: int
    frame_idx: int
    bgr: np.ndarray


@dataclass
class FrameBundle:
    video: VideoFrame
    gaze: GazeSample
    worn: bool
    imu: IMUSample | None = None
