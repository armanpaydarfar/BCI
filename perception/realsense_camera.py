# Vendored from harmony_vlm (https://github.com/vivianchen98/harmony_vlm) @ cfa01b6
# by Vivian Chen. Folded into the BCI repo for WS3 (2026-06-15). Edit here, not
# upstream; see Documents/SoftwareDocs/projects/harmony-bci/vlm-integration/.
"""
realsense_camera.py — Intel RealSense D435 wrapper for aligned RGB-D capture.

Adapted from realsense-utils-dev-main/realsense/camera.py with:
  - BGR output (OpenCV convention)
  - camera_matrix property (3x3 numpy)
  - deproject_pixel / project_point helpers
  - No matplotlib / MultiCamera
"""

from __future__ import annotations

import numpy as np
import pyrealsense2 as rs


class RealSenseCamera:
    """Single Intel RealSense camera with aligned depth."""

    def __init__(
        self,
        serial: str | None = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ):
        self._pipeline = rs.pipeline()
        config = rs.config()
        if serial is not None:
            config.enable_device(str(serial))
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

        profile = self._pipeline.start(config)
        self._align = rs.align(rs.stream.color)
        self._intrinsics = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self.width = width
        self.height = height

    # ── intrinsics ────────────────────────────────────────────────────────

    @property
    def intrinsics(self) -> rs.intrinsics:
        return self._intrinsics

    @property
    def camera_matrix(self) -> np.ndarray:
        """3x3 camera intrinsic matrix (numpy)."""
        i = self._intrinsics
        return np.array(
            [[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]], dtype=np.float64
        )

    @property
    def camera_params(self) -> tuple[float, float, float, float]:
        """(fx, fy, cx, cy) tuple for AprilTag detector."""
        i = self._intrinsics
        return (i.fx, i.fy, i.ppx, i.ppy)

    # ── capture ───────────────────────────────────────────────────────────

    def get_rgbd(self) -> dict[str, np.ndarray] | None:
        """Return {'color': BGR uint8, 'depth': float64 meters}."""
        frames = self._pipeline.wait_for_frames()
        if frames.size() < 2:
            return None
        aligned = self._align.process(frames)
        color_rgb = np.asanyarray(aligned.get_color_frame().get_data())
        color_bgr = color_rgb[:, :, ::-1].copy()  # RGB -> BGR
        depth = np.asanyarray(aligned.get_depth_frame().get_data()) / 1000.0
        return {"color": color_bgr, "depth": depth}

    def get_color(self) -> np.ndarray | None:
        rgbd = self.get_rgbd()
        return rgbd["color"] if rgbd else None

    def get_depth(self) -> np.ndarray | None:
        rgbd = self.get_rgbd()
        return rgbd["depth"] if rgbd else None

    # ── 3D helpers ────────────────────────────────────────────────────────

    def deproject_pixel(self, u: float, v: float, depth: float) -> np.ndarray:
        """Deproject (u, v, depth_meters) to [X, Y, Z] in camera frame."""
        point = rs.rs2_deproject_pixel_to_point(self._intrinsics, [u, v], depth)
        return np.array(point, dtype=np.float64)

    def project_point(self, point: np.ndarray) -> tuple[float, float]:
        """Project [X, Y, Z] in camera frame to (u, v) pixel."""
        px = rs.rs2_project_point_to_pixel(self._intrinsics, point.tolist())
        return (px[0], px[1])

    # ── cleanup ───────────────────────────────────────────────────────────

    def close(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()
