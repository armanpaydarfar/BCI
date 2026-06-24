# Vendored from harmony_vlm (https://github.com/vivianchen98/harmony_vlm) @ cfa01b6
# by Vivian Chen. Folded into the BCI repo for WS3 (2026-06-15). Edit here, not
# upstream; see Documents/SoftwareDocs/projects/harmony-bci/vlm-integration/.
# STAGED — not import-safe in this env (deps deliberately excluded); see the
# live-vs-staged list in perception/__init__.py before importing.
"""
gaze_grounder.py — Ground 2D gaze from Pupil Core into 3D via RealSense + AprilTag.

Math pipeline:
  gaze (gx, gy) in Pupil Core pixels
    -> backproject to ray:  ray_pupil = K_pupil_inv @ [gx, gy, 1]
    -> transform to world:  ray_world = R_world_from_pupil @ ray_pupil
    -> transform to RS:     ray_rs = R_rs_from_world @ ray_world
    -> ray-march + project onto RS image -> (u, v)
    -> depth lookup at (u, v) in aligned depth image
    -> deproject to 3D:     point_rs = deproject(intrinsics, [u,v], depth)
    -> transform to world:  point_world = T_world_from_rs @ point_rs
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import numpy as np

from perception.apriltag_detector import AprilTagDetector
from perception.pupil_reader import IMUSample
from perception.realsense_camera import RealSenseCamera


def _quat_to_rotation(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


@dataclass
class GazePoint3D:
    point_rs: np.ndarray = field(default_factory=lambda: np.zeros(3))
    point_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pixel_rs: tuple[float, float] = (0.0, 0.0)
    depth: float = 0.0
    valid: bool = False
    error: str | None = None


class GazeGrounder:
    """Grounds 2D Pupil Core gaze into 3D world coordinates via RealSense depth."""

    # Stale pose threshold (seconds)
    STALE_POSE_TIMEOUT = 0.5

    # Ray-march parameters
    RAY_STEP_M = 0.02       # 2 cm steps
    RAY_MIN_M = 0.3          # start at 30 cm (skip near-field)
    RAY_MAX_M = 5.0          # max 5 m
    DEPTH_TOLERANCE_M = 0.05  # 5 cm intersection tolerance

    def __init__(
        self,
        rs_camera: RealSenseCamera,
        pupil_intrinsics: np.ndarray,
        tag_size: float = 0.08,
        depth_search_radius: int = 5,
        T_world_from_tag: np.ndarray | None = None,
    ):
        """
        Parameters
        ----------
        rs_camera         : initialised RealSenseCamera
        pupil_intrinsics  : 3x3 Pupil Core camera matrix
        tag_size          : AprilTag physical size in meters
        depth_search_radius : pixel radius for depth fallback search
        T_world_from_tag  : 4x4 tag pose in world frame (default: identity = tag is origin)
        """
        self.rs = rs_camera
        self.K_pupil = pupil_intrinsics.astype(np.float64)
        self.K_pupil_inv = np.linalg.inv(self.K_pupil)
        self.depth_search_radius = depth_search_radius

        self.T_world_from_tag = (
            T_world_from_tag if T_world_from_tag is not None else np.eye(4)
        )

        # AprilTag detectors for each camera
        fx, fy, cx, cy = (
            pupil_intrinsics[0, 0],
            pupil_intrinsics[1, 1],
            pupil_intrinsics[0, 2],
            pupil_intrinsics[1, 2],
        )
        self._pupil_tag = AprilTagDetector(
            camera_params=(fx, fy, cx, cy), tag_size=tag_size
        )
        self._rs_tag = AprilTagDetector(
            camera_params=rs_camera.camera_params, tag_size=tag_size
        )

        # Calibrated transforms (set by calibrate_realsense)
        self._T_world_from_rs: np.ndarray | None = None
        self._T_rs_from_world: np.ndarray | None = None

        # Cached Pupil Core pose (updated every frame)
        self._T_world_from_pupil: np.ndarray | None = None
        self._pupil_pose_time: float = 0.0

        # IMU fusion state
        self._imu_R: np.ndarray | None = None          # latest IMU rotation
        self._imu_R_offset: np.ndarray | None = None    # R_apriltag @ R_imu^-1
        self._last_tag_translation: np.ndarray | None = None  # last known t from AprilTag
        self._imu_calibrated: bool = False

    # ── calibration ───────────────────────────────────────────────────────

    def calibrate_realsense(self, max_attempts: int = 100) -> bool:
        """One-time calibration: detect AprilTag in RS view and store T_world_from_rs.

        Blocks until the tag is found or max_attempts is reached.
        Returns True on success.
        """
        print("[GazeGrounder] Calibrating RealSense — show AprilTag to RS camera…",
              file=sys.stderr)
        for i in range(max_attempts):
            rgbd = self.rs.get_rgbd()
            if rgbd is None:
                continue
            T = self._rs_tag.get_T_world_from_camera(
                rgbd["color"], self.T_world_from_tag
            )
            if T is not None:
                self._T_world_from_rs = T
                self._T_rs_from_world = np.linalg.inv(T)
                t = T[:3, 3]
                print(
                    f"[GazeGrounder] RS calibrated. Translation from world origin: "
                    f"[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m",
                    file=sys.stderr,
                )
                return True
        print("[GazeGrounder] Calibration failed — tag not found.", file=sys.stderr)
        return False

    @property
    def is_calibrated(self) -> bool:
        return self._T_world_from_rs is not None

    # ── IMU fusion ──────────────────────────────────────────────────────

    def update_imu(self, imu: IMUSample | None) -> None:
        """Feed latest IMU reading for pose fusion."""
        if imu is None:
            return
        self._imu_R = _quat_to_rotation(imu.quaternion)

    # ── per-frame grounding ───────────────────────────────────────────────

    def ground(
        self,
        gaze_x: float,
        gaze_y: float,
        pupil_frame_bgr: np.ndarray,
        rs_rgbd: dict[str, np.ndarray] | None = None,
    ) -> GazePoint3D:
        """Ground a 2D gaze point to 3D.

        Parameters
        ----------
        gaze_x, gaze_y   : gaze pixel in Pupil Core image
        pupil_frame_bgr   : current Pupil Core scene camera frame
        rs_rgbd           : pre-fetched {'color': BGR, 'depth': float64} or None

        Returns GazePoint3D (check .valid before using .point_world).
        """
        if not self.is_calibrated:
            return GazePoint3D(valid=False, error="RealSense not calibrated")

        # 1. Update Pupil Core pose via AprilTag
        T_wf_pupil = self._pupil_tag.get_T_world_from_camera(
            pupil_frame_bgr, self.T_world_from_tag
        )
        if T_wf_pupil is not None:
            self._T_world_from_pupil = T_wf_pupil
            self._pupil_pose_time = time.monotonic()
            self._last_tag_translation = T_wf_pupil[:3, 3].copy()
            # Calibrate IMU offset whenever tag is visible
            if self._imu_R is not None:
                R_tag = T_wf_pupil[:3, :3]
                self._imu_R_offset = R_tag @ self._imu_R.T
                if not self._imu_calibrated:
                    self._imu_calibrated = True
                    print("[GazeGrounder] IMU-AprilTag offset calibrated.", file=sys.stderr)
        elif (
            self._imu_calibrated
            and self._imu_R is not None
            and self._last_tag_translation is not None
        ):
            # AprilTag lost — reconstruct pose from IMU
            R_fused = self._imu_R_offset @ self._imu_R
            T_fused = np.eye(4)
            T_fused[:3, :3] = R_fused
            T_fused[:3, 3] = self._last_tag_translation
            self._T_world_from_pupil = T_fused
            self._pupil_pose_time = time.monotonic()
        elif (
            self._T_world_from_pupil is None
            or (time.monotonic() - self._pupil_pose_time) > self.STALE_POSE_TIMEOUT
        ):
            return GazePoint3D(valid=False, error="AprilTag not visible in Pupil Core")

        # 2. Get RealSense RGBD
        if rs_rgbd is None:
            rs_rgbd = self.rs.get_rgbd()
        if rs_rgbd is None:
            return GazePoint3D(valid=False, error="No RealSense frame")

        depth_image = rs_rgbd["depth"]

        # 3. Backproject gaze to ray in Pupil Core frame
        ray_pupil = self.K_pupil_inv @ np.array([gaze_x, gaze_y, 1.0])
        ray_pupil = ray_pupil / np.linalg.norm(ray_pupil)

        # 4. Transform ray direction and origin to RS frame
        T_wf_p = self._T_world_from_pupil
        R_world_from_pupil = T_wf_p[:3, :3]
        origin_pupil_world = T_wf_p[:3, 3]

        ray_world = R_world_from_pupil @ ray_pupil

        T_rs_fw = self._T_rs_from_world
        R_rs_from_world = T_rs_fw[:3, :3]
        t_rs_from_world = T_rs_fw[:3, 3]

        ray_rs = R_rs_from_world @ ray_world
        origin_rs = R_rs_from_world @ origin_pupil_world + t_rs_from_world

        # 5. Ray-march to find depth intersection
        result = self._ray_march(origin_rs, ray_rs, depth_image)
        if result is not None:
            u, v, depth_m, point_rs = result
        else:
            # Fallback: direct projection of gaze direction
            result = self._direct_project_fallback(
                origin_rs, ray_rs, depth_image
            )
            if result is None:
                return GazePoint3D(valid=False, error="Ray-depth intersection failed")
            u, v, depth_m, point_rs = result

        # 6. Transform to world frame
        point_world = (self._T_world_from_rs[:3, :3] @ point_rs
                       + self._T_world_from_rs[:3, 3])

        return GazePoint3D(
            point_rs=point_rs,
            point_world=point_world,
            pixel_rs=(u, v),
            depth=depth_m,
            valid=True,
        )

    # ── ray-marching ──────────────────────────────────────────────────────

    def _ray_march(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        depth_image: np.ndarray,
    ) -> tuple[float, float, float, np.ndarray] | None:
        """March along ray in RS frame, project each point, compare to depth map.

        Returns (u, v, depth, point_rs) or None.
        """
        h, w = depth_image.shape[:2]
        direction = direction / np.linalg.norm(direction)

        t = self.RAY_MIN_M
        while t <= self.RAY_MAX_M:
            pt = origin + direction * t
            if pt[2] <= 0:
                t += self.RAY_STEP_M
                continue

            u, v = self.rs.project_point(pt)
            ui, vi = int(round(u)), int(round(v))

            if 0 <= ui < w and 0 <= vi < h:
                measured_depth = depth_image[vi, ui]
                if measured_depth > 0:
                    ray_depth = pt[2]  # Z in RS frame
                    if abs(ray_depth - measured_depth) < self.DEPTH_TOLERANCE_M:
                        point_rs = self.rs.deproject_pixel(u, v, measured_depth)
                        return (u, v, measured_depth, point_rs)

            t += self.RAY_STEP_M

        return None

    def _direct_project_fallback(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        depth_image: np.ndarray,
    ) -> tuple[float, float, float, np.ndarray] | None:
        """Fallback: project at a typical distance and search for valid depth."""
        h, w = depth_image.shape[:2]
        direction = direction / np.linalg.norm(direction)

        # Try projecting at 1m, 1.5m, 2m to find a valid pixel
        for dist in [1.0, 1.5, 2.0, 0.5, 3.0]:
            pt = origin + direction * dist
            if pt[2] <= 0:
                continue
            u, v = self.rs.project_point(pt)
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < w and 0 <= vi < h:
                depth_m = self._search_depth(depth_image, ui, vi)
                if depth_m > 0:
                    point_rs = self.rs.deproject_pixel(float(ui), float(vi), depth_m)
                    return (float(ui), float(vi), depth_m, point_rs)

        return None

    def _search_depth(
        self, depth_image: np.ndarray, u: int, v: int
    ) -> float:
        """Search neighborhood for valid depth value."""
        h, w = depth_image.shape[:2]
        r = self.depth_search_radius

        # Check center first
        if depth_image[v, u] > 0:
            return float(depth_image[v, u])

        # Spiral search outward
        best_depth = 0.0
        best_dist = float("inf")
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ny, nx = v + dy, u + dx
                if 0 <= nx < w and 0 <= ny < h:
                    d = depth_image[ny, nx]
                    if d > 0:
                        dist = dx * dx + dy * dy
                        if dist < best_dist:
                            best_dist = dist
                            best_depth = float(d)

        return best_depth
