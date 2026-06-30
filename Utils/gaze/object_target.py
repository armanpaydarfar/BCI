"""
object_target.py — depth-free 3-D object target for gaze→robot control (WS-4).

The robust alternative to monocular Depth Pro for objects resting on a known
surface: pick the WHOLE fixated object's mask, take its footprint (or centroid)
pixel, and cast that pixel through the camera onto the calibrated TABLE plane to
get a world ``(x, y, z)``. This is

  - **depth-free** — the table plane supplies depth, pinned by the table tags to
    ~mm, instead of monocular Depth Pro (whose lateral error grows with offset and
    distance);
  - **overshoot-free** — the footprint is where the object meets the table, so
    looking at a tall object's top no longer casts a ray that sails over it
    (the §1.3 overshoot);
  - **cross-frame-robust** — the caller recasts the *pixel* through its own
    anchored ``T_cam_world`` rather than transforming a stale service-frame 3-D
    point, and the service's ``gaze_px`` is checked against the caller's own gaze
    to reject a divergent frame.

This lifts the validated WS-1 selection + ray∩plane logic from the 2-D control
tool (``tools/apriltag_control_test.py``) into a shared, world-3-D form used by
the 3-D control tool and the evaluation harness. The 2-D tool keeps its own copy
(REV05 is frozen); these are the same rules in world-point form. Pure numpy +
the transform helpers → unit-tested without a display or the perception stack.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from Utils.gaze.apriltag_calib import (
    gaze_ray_cam,
    invert_transform,
    ray_plane_intersection,
    transform_point,
)
from Utils.gaze.apriltag_world import table_normal_from_rel

# Selection tolerances, lifted verbatim from the 2-D WS-1 path
# (tools/apriltag_control_test.py:84-86) so the two regimes agree.
HIT_OUTSIDE_TOL_PX = 40.0       # gaze-outside-every-mask slack before a hit is rejected
BOTTOM_BAND_PX = 3.0            # rows above the lowest mask point averaged for the footprint x
GAZE_DIVERGENCE_TOL_PX = 80.0   # service-frame vs our-frame gaze gap above which masks are distrusted


def table_plane_from_map(world_map: Dict,
                         table_ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Table plane ``(point, normal)`` in the WORLD frame from the table-tag subset.

    The 3-D world map stores a best-fit plane through ALL fused tags (table +
    walls), which is meaningless as a table. Reproduce the validated REV05 table
    plane from the TABLE tags only: the normal is the orientation-average of their
    +Z faces (``table_normal_from_rel`` — robust where per-tag origin heights are
    noisy), the point is the centroid of their origins.

    Returns ``(point, normal, info)``. ``info`` carries ``table_ids`` actually
    used and the table tags' out-of-plane residual (``max_resid_mm`` /
    ``rms_resid_mm``) — a flatness check the caller should log: a large residual
    means the "table" tags aren't coplanar (wrong id set, or a bumped tag).

    Raises ValueError with fewer than 3 table tags present in the map.
    """
    rel = world_map["rel"]
    present = [int(i) for i in table_ids if int(i) in rel]
    if len(present) < 3:
        raise ValueError(
            f"need ≥3 table tags present in the map for a plane; have {present} "
            f"(map tags {sorted(int(i) for i in rel)})")
    origins = np.array([np.asarray(rel[i][:3, 3], dtype=float) for i in present])
    point = origins.mean(axis=0)
    normal = table_normal_from_rel(rel, present)
    resid = np.abs((origins - point) @ normal)
    info = {
        "table_ids": present,
        "max_resid_mm": float(resid.max()),
        "rms_resid_mm": float(np.sqrt(np.mean(resid ** 2))),
    }
    return point, normal, info


def select_object_pixel(detections: List[Dict], gaze_xy: Tuple[float, float],
                        source: str, *,
                        outside_tol_px: float = HIT_OUTSIDE_TOL_PX,
                        bottom_band_px: float = BOTTOM_BAND_PX
                        ) -> Tuple[Optional[float], Optional[float], Dict]:
    """Pick the fixated WHOLE object's mask and return its target pixel.

    ``detections``: ``vlm.segment(include_masks=True)`` output — dicts with a
    ``mask_polygon`` ``[[x,y],...]``. ``source``: ``'centroid'`` (mask centroid) or
    ``'footprint'`` (alias ``'bottom'``: mean x of the lowest ``bottom_band_px``
    rows, at the max y — the object↔table contact line).

    Selection (the validated WS-1 rule, ``apriltag_control_test.py:149-178``):
    among masks whose polygon CONTAINS the gaze, choose the one whose centroid is
    nearest the gaze, area as tie-break — NOT the first / smallest segment, which
    is what made the 3-D ``waypoints`` path's target trace the gaze across an
    over-segmented object. With no containing mask, fall back to the nearest mask
    within ``outside_tol_px``; a clean miss returns ``(None, None, info)``.

    Returns ``(px, py, info)``. ``info`` has ``n_dets``/``n_contained``/``pick``
    and, on a hit, ``mask_area_px`` — log it to see WHY a target was chosen.
    """
    import cv2  # lazy: only the perception path needs it

    gx, gy = float(gaze_xy[0]), float(gaze_xy[1])
    contained: List[Tuple[float, float, np.ndarray]] = []   # (centroid_dist, area, contour)
    nearest_outside: Optional[Tuple[float, np.ndarray]] = None  # (outside_dist, contour)
    for det in detections:
        poly = det.get("mask_polygon")
        if poly is None or len(poly) < 3:
            continue
        cnt = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        signed = cv2.pointPolygonTest(cnt, (gx, gy), True)  # ≥0 inside, signed mm-of-pixels
        M = cv2.moments(cnt)
        area = float(M["m00"])
        if area > 0:
            cxy = (M["m10"] / area, M["m01"] / area)
        else:
            cxy = tuple(cnt.reshape(-1, 2).mean(axis=0))
        cdist = float(np.hypot(cxy[0] - gx, cxy[1] - gy))
        if signed >= 0:
            contained.append((cdist, area, cnt))
        else:
            od = float(-signed)
            if nearest_outside is None or od < nearest_outside[0]:
                nearest_outside = (od, cnt)

    info: Dict = {"n_dets": len(detections), "n_contained": len(contained)}
    if contained:
        cnt = min(contained, key=lambda c: (c[0], c[1]))[2]
        info["pick"] = "contained"
    elif nearest_outside is not None and nearest_outside[0] <= outside_tol_px:
        cnt = nearest_outside[1]
        info["pick"] = "nearest_outside"
    else:
        info["pick"] = "miss"
        return None, None, info

    pts = cnt.reshape(-1, 2).astype(float)
    if source in ("footprint", "bottom"):
        y_max = float(pts[:, 1].max())
        on_bottom = pts[:, 1] >= y_max - bottom_band_px
        px, py = float(pts[on_bottom, 0].mean()), y_max
    else:  # centroid
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            px, py = float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])
        else:
            px, py = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    info["mask_area_px"] = float(cv2.contourArea(cnt))
    info["mask_polygon"] = pts.astype(int).tolist()  # chosen object outline, for the viz
    return px, py, info


def pixel_on_plane_world(px: float, py: float, K: np.ndarray,
                         T_cam_world: np.ndarray,
                         plane_point: np.ndarray,
                         plane_normal: np.ndarray) -> Optional[np.ndarray]:
    """World 3-D point where the camera ray through pixel ``(px, py)`` meets the
    plane — the depth-free target.

    Same chain as the 2-D ``gaze_point_in_plane_uv`` but returns the WORLD point
    (mm) instead of projecting to table ``(u, v)``: unproject the pixel to a unit
    ray in the camera frame, express the (world-frame) plane in the camera frame,
    intersect there, then map the hit back to world via ``T_world_cam``. The plane
    supplies depth, so no monocular depth is needed. ``T_cam_world`` maps WORLD→CAM
    (the world frame's pose in the camera). Returns None if the pixel/K is
    degenerate or the ray misses the plane (parallel / behind the camera).
    """
    ray = gaze_ray_cam(px, py, K)
    if ray is None:
        return None
    pp_cam = transform_point(T_cam_world, np.asarray(plane_point, dtype=float))
    nn_cam = T_cam_world[:3, :3] @ np.asarray(plane_normal, dtype=float)
    hit_cam = ray_plane_intersection(np.zeros(3), ray, pp_cam, nn_cam)
    if hit_cam is None:
        return None
    return transform_point(invert_transform(T_cam_world), hit_cam)


def height_on_vertical(gaze_px, base_world: np.ndarray, table_normal: np.ndarray,
                       K: np.ndarray, T_cam_world: np.ndarray, *,
                       max_height_mm: float = 600.0) -> Optional[np.ndarray]:
    """Target on the object's vertical centerline at the height the gaze looks at.

    The depth-free height extension: with the XY anchored at ``base_world`` (the
    object's footprint on the table), the operator's height along the object is the
    point on the VERTICAL line ``base_world + t·n`` (n = table normal) closest to
    the gaze ray. Look at the base → ~table height; look higher → up the object.
    This is what lets "look at a part → move there" work without depth (and lets a
    stacked object be targeted up its supporting column). ``t`` is clamped to
    ``[0, max_height_mm]`` so a near-side-on ray can't yield an absurd height.

    Assumes the object stands roughly vertical at ``base_world``'s XY. Returns the
    world point, or None on a bad pixel/K. Ill-conditioned from straight overhead
    (the vertical is along the view) — t then falls back toward the base.
    """
    ray_cam = gaze_ray_cam(float(gaze_px[0]), float(gaze_px[1]), K)
    if ray_cam is None:
        return None
    T_wc = invert_transform(T_cam_world)            # world ← cam
    cam_org = T_wc[:3, 3]
    ray_w = T_wc[:3, :3] @ ray_cam
    nrm = float(np.linalg.norm(ray_w))
    if nrm < 1e-9:
        return None
    ray_w = ray_w / nrm
    n = np.asarray(table_normal, dtype=float)
    n = n / max(float(np.linalg.norm(n)), 1e-9)
    P1 = np.asarray(base_world, dtype=float)
    # Closest point on the vertical line (P1, n) to the gaze ray (cam_org, ray_w).
    w0 = P1 - cam_org
    b = float(n @ ray_w)
    denom = 1.0 - b * b                              # a = c = 1 (unit dirs)
    if abs(denom) < 1e-9:                            # ray ∥ vertical → height unobservable
        t = 0.0
    else:
        t = (b * float(ray_w @ w0) - float(n @ w0)) / denom
    t = float(np.clip(t, 0.0, max_height_mm))
    return P1 + t * n


def world_to_pixel(p_world: np.ndarray, K: np.ndarray,
                   T_cam_world: np.ndarray) -> Optional[tuple]:
    """Project a world point to a scene-camera pixel (for the visual interface, e.g.
    drawing where the chosen 3-D target lands on the image). None if behind the
    camera."""
    p_cam = transform_point(T_cam_world, np.asarray(p_world, dtype=float))
    if p_cam[2] <= 1e-6:
        return None
    return (float(K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]),
            float(K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]))


def elevation_coords(points: np.ndarray, base_world: np.ndarray,
                     cam_center_world: np.ndarray,
                     table_normal: np.ndarray) -> np.ndarray:
    """Project world points into the vertical cross-section through the object's
    base and the camera, for the side/elevation view that makes the gaze-height
    geometry visible.

    The section's 2-D axes (origin at ``base_world``): ``h`` = horizontal distance
    along the in-plane direction from the base toward the camera; ``v`` = height
    along the table normal. So the table reads as the line v=0, the object's
    vertical line as h=0, and the gaze ray / camera / target plot in true
    edge-on geometry. ``points`` is ``(3,)`` or ``(N,3)``; returns ``(2,)`` or
    ``(N,2)``."""
    n = np.asarray(table_normal, dtype=float)
    n = n / max(float(np.linalg.norm(n)), 1e-9)
    base = np.asarray(base_world, dtype=float)
    toward = np.asarray(cam_center_world, dtype=float) - base
    horiz = toward - (toward @ n) * n
    if float(np.linalg.norm(horiz)) < 1e-6:        # camera ~straight overhead
        # degenerate: any in-plane axis ⟂ n (camera directly above the base)
        ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        horiz = ref - (ref @ n) * n
    h_hat = horiz / float(np.linalg.norm(horiz))
    P = np.asarray(points, dtype=float)
    single = (P.ndim == 1)
    P = np.atleast_2d(P)
    d = P - base
    out = np.column_stack([d @ h_hat, d @ n])
    return out[0] if single else out


def gaze_divergence_ok(service_gaze_px, our_gaze_px,
                       tol_px: float = GAZE_DIVERGENCE_TOL_PX) -> bool:
    """True if the service-frame gaze and our-frame gaze agree within ``tol_px``.

    The mask + ``gaze_px`` come from the service's frame instant; we cast the
    pixel through OUR anchored ``T_cam_world``. If the head moved between them the
    two gazes diverge — reject (the caller falls back) rather than aim at the wrong
    place. Non-finite inputs → not OK."""
    s = np.asarray(service_gaze_px, dtype=float)
    o = np.asarray(our_gaze_px, dtype=float)
    if s.shape != (2,) or o.shape != (2,) or not (np.all(np.isfinite(s)) and np.all(np.isfinite(o))):
        return False
    return float(np.hypot(*(s - o))) <= tol_px
