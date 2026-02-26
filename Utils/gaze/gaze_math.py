# Utils/gaze/gaze_math.py
"""
gaze_math.py

Pure math + transforms utilities for the Neon gaze pipeline.

Rules:
- NO device I/O
- NO threading
- NO Ultralytics/YOLO
- NO OpenCV
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# -----------------------
# Small helpers
# -----------------------
def safe_unit(v: np.ndarray) -> np.ndarray:
    """Return v normalized to unit length, or zeros if too small."""
    v = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def wrap_deg_180(a: float) -> float:
    """Wrap degrees into [-180, 180)."""
    if not np.isfinite(a):
        return float(a)
    return float((a + 180.0) % 360.0 - 180.0)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass
class EWMASmoother:
    """Exponential moving average for scalars."""
    alpha: float
    y: float = float("nan")

    def update(self, x: float) -> float:
        x = float(x)
        if not np.isfinite(x):
            return float(self.y)
        if not np.isfinite(self.y):
            self.y = x
        else:
            self.y = float(self.alpha) * x + (1.0 - float(self.alpha)) * float(self.y)
        return float(self.y)


# -----------------------
# Vergence depth
# -----------------------
def vergence_depth_from_eyestate(
    L_mm: np.ndarray,
    u: np.ndarray,
    R_mm: np.ndarray,
    v: np.ndarray,
    *,
    miss_max_mm: float = 30.0,
    denom_min: float = 1e-4,
) -> tuple[bool, float, float]:
    """
    Compute vergence depth from two 3D rays defined by eye centers and optical axes.

    Returns:
      (valid, depth_m, miss_mm)

    Notes:
      - 'miss' is the distance between closest points on the two rays.
      - depth is distance from midpoint of eye centers to midpoint of closest points.
    """
    L = np.asarray(L_mm, dtype=float).reshape(3)
    R = np.asarray(R_mm, dtype=float).reshape(3)
    u = np.asarray(u, dtype=float).reshape(3)
    v = np.asarray(v, dtype=float).reshape(3)

    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-9 or nv < 1e-9:
        return False, float("nan"), float("nan")

    u = u / nu
    v = v / nv

    w0 = L - R
    b = float(np.dot(u, v))
    d = float(np.dot(u, w0))
    e = float(np.dot(v, w0))

    denom = 1.0 - b * b
    if denom < float(denom_min):
        return False, float("nan"), float("nan")

    t = (b * e - d) / denom
    s = (e - b * d) / denom

    P_L = L + t * u
    P_R = R + s * v
    miss_mm_val = float(np.linalg.norm(P_L - P_R))

    C = 0.5 * (L + R)
    P = 0.5 * (P_L + P_R)
    d_mm = float(np.linalg.norm(P - C))
    d_m = d_mm / 1000.0

    valid = (miss_mm_val <= float(miss_max_mm)) and np.isfinite(d_m) and (d_m > 0.05)
    return bool(valid), (float(d_m) if valid else float("nan")), (float(miss_mm_val) if valid else float("nan"))


# -----------------------
# Quaternions / rotations
# -----------------------
def quat_to_rotmat_wxyz(w: float, x: float, y: float, z: float) -> np.ndarray:
    """
    Convert quaternion (w,x,y,z) to rotation matrix R (3x3).
    """
    q = np.array([w, x, y, z], dtype=float)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3, dtype=float)
    q /= n
    w, x, y, z = q
    R = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )
    return R


def scene_to_imu_matrix(imu_scene_rot_deg: float) -> np.ndarray:
    """
    Scene->IMU rotation about X axis (degrees).
    Matches your earlier pipeline (IMU_SCENE_ROT_DEG).
    """
    a = np.deg2rad(float(imu_scene_rot_deg))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(a), -np.sin(a)],
            [0.0, np.sin(a),  np.cos(a)],
        ],
        dtype=float,
    )


# -----------------------
# Angle conversions
# -----------------------
def cartesian_to_spherical_world(world_vec: np.ndarray) -> tuple[float, float]:
    """
    Convert a world vector to (elevation_deg, azimuth_deg) using your existing conventions.
    """
    v = safe_unit(world_vec)
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    r = max(np.sqrt(x * x + y * y + z * z), 1e-9)

    elevation = -(np.arccos(z / r) - np.pi / 2.0)
    azimuth = np.arctan2(y, x) - np.pi / 2.0

    if azimuth < -np.pi:
        azimuth += 2 * np.pi
    if azimuth > np.pi:
        azimuth -= 2 * np.pi

    return float(np.rad2deg(elevation)), float(np.rad2deg(azimuth))


def gaze_ray_from_optical_axes(gaze) -> np.ndarray:
    """
    Return a cyclopean gaze ray from the Neon gaze datum optical axes.
    Expects attributes:
      optical_axis_left_{x,y,z}, optical_axis_right_{x,y,z}
    """
    uL = np.array([gaze.optical_axis_left_x, gaze.optical_axis_left_y, gaze.optical_axis_left_z], dtype=float)
    uR = np.array([gaze.optical_axis_right_x, gaze.optical_axis_right_y, gaze.optical_axis_right_z], dtype=float)
    return safe_unit(safe_unit(uL) + safe_unit(uR))
