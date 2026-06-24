"""
apriltag_calib.py — pure-geometry core for the REV03 AprilTag gaze↔robot
calibration (WS5).

Methodology: SoftwareDocs/_archive/harmony-bci/gaze-calibration/
rev03-apriltag-methodology.md (archived; the § citations below are its sections —
the current model is the active rev04-planar-coverage-methodology.md). This module
holds ONLY the hardware-free math so
it is unit-testable without a Neon, a robot, or `pupil-apriltags`:

  - rigid 4×4 transform helpers (build / invert / apply),
  - the Umeyama/Kabsch paired-point rigid solve for ``T_base_world`` (§4.3),
  - gaze-pixel → camera-frame ray unprojection (§5.1, same convention as
    ``harmony_free_arm_calibration._gaze_px_to_yaw_pitch_deg``),
  - ray-plane intersection for the table-plane gaze hit (§5.3),
  - EE-origin-in-world from the EE-tag pose + hand-measured mount offset (§4.2).

All lengths are in **millimetres** to match robot telemetry (``eeR.pos_mm``)
and the ``X`` calibration column; tag-detector translations (metres) must be
scaled to mm before entering these functions. Rotations are dimensionless 3×3.

Detection (``pupil-apriltags``) + the frame relay live in ``apriltag_detect.py``,
the robot link in ``harmony_link.py``, and the CLI stages in
``tools/apriltag_calibrate.py`` + ``tools/apriltag_control_test.py``. The
experiment-driver integration (the ``V3`` NN over ``X``) lands after the
calibration is validated on hardware (§9). This module is import-safe and
depends only on numpy.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Parallel-ray / singular-denominator floor. A gaze ray within this dot-product
# of perpendicular to the plane normal is treated as non-intersecting rather
# than producing a wildly extrapolated hit.
_PARALLEL_EPS = 1e-9


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4×4 homogeneous transform from a 3×3 rotation and a 3-vector
    translation. No orthonormality check — callers pass solver output."""
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).ravel()
    if R.shape != (3, 3) or t.shape != (3,):
        raise ValueError(f"R must be (3,3) and t (3,); got {R.shape}, {t.shape}")
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert a rigid 4×4 transform analytically (Rᵀ, −Rᵀt) — cheaper and more
    numerically stable than a general inverse, and exact for a rigid T."""
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"T must be (4,4); got {T.shape}")
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=float)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply a 4×4 transform to a single 3-point: ``R·p + t``."""
    T = np.asarray(T, dtype=float)
    p = np.asarray(p, dtype=float).ravel()
    if p.shape != (3,):
        raise ValueError(f"p must be (3,); got {p.shape}")
    return T[:3, :3] @ p + T[:3, 3]


def umeyama_rigid(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float]:
    """Best-fit **rigid** transform ``T`` (rotation + translation, no scale)
    mapping ``src`` onto ``dst`` in the least-squares sense — Umeyama (1991) /
    Kabsch, with the determinant-sign correction so noise cannot yield a
    reflection instead of a proper rotation.

    For REV03: ``src = P_world`` (EE points in the world-tag frame),
    ``dst = P_base`` (the same EE points in the robot base frame, from
    telemetry). The returned ``T`` is ``T_base_world`` (§4.3).

    Args:
        src: (N, 3) source points.
        dst: (N, 3) destination points, row-aligned with ``src``.

    Returns:
        ``(T, rms)`` — the 4×4 rigid transform and the RMS residual
        ``sqrt(mean ‖dst − T·src‖²)`` in input units (mm).

    Raises:
        ValueError: fewer than 3 points, mismatched shapes, or non-finite
            input (the rigid fit needs ≥3 non-degenerate correspondences).
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"src/dst must be matching (N,3); got {src.shape}, {dst.shape}")
    n = src.shape[0]
    if n < 3:
        raise ValueError(f"need ≥3 correspondences for a rigid fit; got {n}")
    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(dst))):
        raise ValueError("src/dst contain non-finite values")

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    xs = src - mu_s
    xd = dst - mu_d

    # Cross-covariance (dst × src); SVD gives the optimal rotation.
    cov = (xd.T @ xs) / n
    U, _S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    t = mu_d - R @ mu_s

    T = make_transform(R, t)
    pred = (R @ src.T).T + t
    rms = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))
    return T, rms


def umeyama_similarity_2d(src: np.ndarray, dst: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Best-fit 2-D **similarity** (rotation + uniform scale + translation) mapping
    ``src`` onto ``dst`` — Umeyama (1991) *with* scale, in the plane. The REV04 A2
    quality readout (rev04 §1): fit ``base(x,y) ← plane(u,v)`` from the swept
    correspondences and report the in-plane RMS, the well-conditioned 2-D analogue
    of the old 3-D Umeyama residual (no height ⇒ no non-coplanarity degeneracy).

    This is a **diagnostic only** — the command path is the A1 nearest-neighbour
    library, not this fit.

    Args:
        src: (N, 2) source points (table-plane ``(u,v)``, mm).
        dst: (N, 2) destination points (base-frame ``(x,y)``, mm), row-aligned.

    Returns:
        ``(A, t, s, rms)`` — the 2×2 linear part ``A = s·R``, the 2-vector
        translation ``t`` (so ``dst ≈ A·src + t``), the scalar scale ``s``, and the
        RMS residual (mm).

    Raises:
        ValueError: fewer than 2 points, mismatched shapes, or non-finite input.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(f"src/dst must be matching (N,2); got {src.shape}, {dst.shape}")
    n = src.shape[0]
    if n < 2:
        raise ValueError(f"need ≥2 correspondences for a 2-D similarity fit; got {n}")
    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(dst))):
        raise ValueError("src/dst contain non-finite values")

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    xs = src - mu_s
    xd = dst - mu_d
    cov = (xd.T @ xs) / n
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    D = np.diag([1.0, d])
    R = U @ D @ Vt
    var_s = float(np.mean(np.sum(xs ** 2, axis=1)))
    # Degenerate src (all points coincident) → no scale is recoverable; fall back to
    # unit scale so the fit is a pure rotation+translation rather than dividing by 0.
    s = float(np.sum(S * np.array([1.0, d])) / var_s) if var_s > 0.0 else 1.0
    A = s * R
    t = mu_d - A @ mu_s
    pred = (A @ src.T).T + t
    rms = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))
    return A, t, s, rms


def per_point_errors(T: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Per-correspondence Euclidean residual ‖dst − T·src‖ (mm), length N —
    for the solve stage's outlier inspection."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    pred = (T[:3, :3] @ src.T).T + T[:3, 3]
    return np.linalg.norm(pred - dst, axis=1)


def gaze_ray_cam(px: float, py: float, K: np.ndarray) -> Optional[np.ndarray]:
    """Unproject a gaze pixel to a **unit** ray in the camera frame
    (+Z out of the lens), ``dir ∝ K⁻¹·[u,v,1]`` — the §5.1 convention, matching
    ``harmony_free_arm_calibration._gaze_px_to_yaw_pitch_deg``.

    Returns ``None`` on non-finite input or a singular ``K`` (so callers skip
    blink/NaN gaze frames rather than propagate garbage)."""
    K = np.asarray(K, dtype=float)
    if not (np.isfinite(px) and np.isfinite(py)) or not np.all(np.isfinite(K)):
        return None
    try:
        ray = np.linalg.inv(K) @ np.array([float(px), float(py), 1.0])
    except np.linalg.LinAlgError:
        return None
    norm = np.linalg.norm(ray)
    if not np.isfinite(norm) or norm == 0.0:
        return None
    return ray / norm


def ray_plane_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> Optional[np.ndarray]:
    """Intersect a ray with a plane. Returns the 3-point hit, or ``None`` if the
    ray is parallel to the plane (|n·d| < eps) or the hit is behind the origin
    (t ≤ 0) — a fixation cannot land behind the camera.

    Args:
        origin: ray origin (camera origin = zeros in REV03 runtime).
        direction: ray direction (need not be unit).
        plane_point: any point on the plane (the world-tag origin in cam frame).
        plane_normal: plane normal (the world-tag +Z axis in cam frame).

    Used by the control tool's gaze→base chain (§5); the experiment-driver
    integration with the V3 mapping lands later (§9).
    """
    origin = np.asarray(origin, dtype=float).ravel()
    direction = np.asarray(direction, dtype=float).ravel()
    plane_point = np.asarray(plane_point, dtype=float).ravel()
    plane_normal = np.asarray(plane_normal, dtype=float).ravel()
    denom = float(direction @ plane_normal)
    if abs(denom) < _PARALLEL_EPS:
        return None
    t = float((plane_point - origin) @ plane_normal / denom)
    if t <= 0.0:
        return None
    return origin + t * direction


def tag_plane_in_cam(T_cam_tag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The tag's own plane expressed in the camera frame: ``(point, normal)``
    where point = the tag origin (T translation) and normal = the tag +Z axis
    (third column of the rotation).

    **Modeling assumption:** this equals the *table* plane only when the world
    tag lies flat in the table plane with its +Z along the table normal (the
    REV03 world-bundle mounting assumption, §5.2). It is not a general
    table-plane fit.

    Used by the control tool's gaze→base chain (§5.2) to build the table plane;
    the experiment-driver integration with the V3 mapping lands later (§9)."""
    T_cam_tag = np.asarray(T_cam_tag, dtype=float)
    return T_cam_tag[:3, 3].copy(), T_cam_tag[:3, 2].copy()


def ee_point_in_world(T_world_eetag: np.ndarray, t_eetag_ee: np.ndarray) -> np.ndarray:
    """EE origin expressed in the world frame, applying the hand-measured
    tag→EE offset rotated by the recovered EE-tag orientation (§4.2):
    ``p_world = t_world_eetag + R_world_eetag · t_eetag_ee``."""
    T_world_eetag = np.asarray(T_world_eetag, dtype=float)
    t_eetag_ee = np.asarray(t_eetag_ee, dtype=float).ravel()
    if t_eetag_ee.shape != (3,):
        raise ValueError(f"t_eetag_ee must be (3,); got {t_eetag_ee.shape}")
    return T_world_eetag[:3, 3] + T_world_eetag[:3, :3] @ t_eetag_ee


def eetag_to_world_point(T_cam_world: np.ndarray, T_cam_eetag: np.ndarray,
                         t_eetag_ee: np.ndarray) -> np.ndarray:
    """The §4.2 capture compose as one tested unit: the EE origin in the world
    frame from the two camera-frame tag poses + the hand-measured mount offset.

    ``p_world = ee_point_in_world( (T_cam_world)⁻¹ · T_cam_eetag , t_eetag_ee )``

    The head pose cancels in the composition (§1) — both tag poses are in the
    same camera frame, so ``world_T_eetag = world_T_cam · cam_T_eetag``. Kept
    as a single function so the inversion/compose convention is unit-testable
    without a Neon or robot (the calibration tool's collect stage just calls this)."""
    T_world_eetag = invert_transform(T_cam_world) @ np.asarray(T_cam_eetag, dtype=float)
    return ee_point_in_world(T_world_eetag, t_eetag_ee)


def eetag_rayplane_point_world(T_cam_world: np.ndarray, T_cam_eetag: np.ndarray,
                               plane_point_world, plane_normal_world):
    """Depth-ambiguity-free EE position for capture (rev04 §5 follow-up,
    2026-06-24). Instead of the EE tag's full 3-D pose translation — whose range
    along the line of sight is poorly constrained for a single small planar tag
    (the ``more than one minima`` ambiguity that set the HIL repeatability floor) —
    back-project the tag CENTRE's line of sight (its *direction* is the sub-pixel
    detected centre, reliable even when the range is not) and intersect it with the
    known world table plane. This is the **same ray∩plane** the runtime uses for
    gaze (``apriltag_control_test.gaze_point_in_plane_uv``), so the calibration
    point and the runtime point are produced by an identical operation.

    The tag centre in the camera frame is ``T_cam_eetag[:,3]``; the camera origin is
    the camera-frame origin, so the head pose cancels exactly as in §1. The plane is
    given in WORLD coords and transformed into the camera frame via ``T_cam_world``.
    Returns the world-frame hit point, or ``None`` if the ray is parallel to / behind
    the plane. No EE-tag→EE offset is applied — this is a centroid (line-of-sight)
    estimate by construction (operator 2026-06-24)."""
    T_cam_world = np.asarray(T_cam_world, dtype=float)
    T_cam_eetag = np.asarray(T_cam_eetag, dtype=float)
    center_cam = T_cam_eetag[:3, 3]  # tag centre in cam = the line-of-sight direction
    pp_world = np.append(np.asarray(plane_point_world, dtype=float), 1.0)
    point_cam = (T_cam_world @ pp_world)[:3]
    normal_cam = T_cam_world[:3, :3] @ np.asarray(plane_normal_world, dtype=float)
    hit_cam = ray_plane_intersection(np.zeros(3), center_cam, point_cam, normal_cam)
    if hit_cam is None:
        return None
    return (invert_transform(T_cam_world) @ np.append(hit_cam, 1.0))[:3]


def average_rotation(rotations: np.ndarray) -> np.ndarray:
    """Chordal-L2 mean of a stack of 3×3 rotations via SVD projection of the
    elementwise mean back onto SO(3) (with the determinant-sign correction).
    Used as the reference for geodesic rotation-jitter (the proper metric vs
    per-axis Rodrigues-component std, which suffers axis ambiguity at small
    angles)."""
    M = np.mean(np.asarray(rotations, dtype=float), axis=0)
    U, _S, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def geodesic_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic (rotation) angle between two 3×3 rotations, degrees:
    ``arccos((trace(Raᵀ·Rb) − 1) / 2)``. Wraparound- and axis-ambiguity-free,
    unlike the std of Rodrigues-vector components."""
    R = np.asarray(Ra, dtype=float).T @ np.asarray(Rb, dtype=float)
    cos = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def angle_between_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Angle (degrees) between two vectors — the gaze-stage metric (gaze ray vs
    ray to the recovered tag centre, §7 gaze stage). Returns NaN on a zero
    vector."""
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0 or not (np.isfinite(nu) and np.isfinite(nv)):
        return float("nan")
    cos = float(np.clip((u @ v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))
