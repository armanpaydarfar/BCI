"""
apriltag_world.py — multi-tag world map for the AprilTag gaze↔robot calibration
(WS5). Makes the world frame robust to **occlusion** (the exoskeleton arm sweeps
over individual tags) and more accurate across the table (fusion = triangulation).

The problem a single world tag has: each tag on its own defines a *different*
world frame, so `T_base_world` (calibrated against one) doesn't transfer when a
different tag is visible, and a single tag's plane-orientation error is
lever-armed across the table. The fix:

  1. **Register once** (all world tags visible, arm clear): learn each tag's pose
     RELATIVE to a reference tag → a rigid map (`build_world_map`).
  2. **Recover per frame** from whichever tags are currently visible: each mapped
     tag gives an estimate of the *same* world frame; fuse them
     (`recover_world_pose`). Any ≥1 mapped tag in view → the full world pose, so
     occluding some tags is fine; multiple in view → averaged (more accurate).

Pose convention matches the rest of the pipeline: ``T_cam_X`` is tag X's pose in
the camera frame (maps a point in X's frame to camera coords). The world frame is
the reference tag's frame at registration. Pure numpy + the transform helpers in
`apriltag_calib`; hardware-free and unit-tested.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from Utils.gaze.apriltag_calib import average_rotation, invert_transform, make_transform


def average_pose(transforms: List[np.ndarray]) -> np.ndarray:
    """Mean of rigid 4×4 transforms: arithmetic mean translation + chordal-L2
    mean rotation (SVD-projected back onto SO(3)). Used to denoise a tag's pose
    over a registration window and to fuse per-tag world-pose estimates."""
    arr = [np.asarray(T, dtype=float) for T in transforms]
    t = np.mean([T[:3, 3] for T in arr], axis=0)
    R = average_rotation([T[:3, :3] for T in arr])
    return make_transform(R, t)


def fit_plane(points: np.ndarray):
    """Best-fit plane through ≥3 points → ``(centroid, unit_normal)``. The normal
    is the least-significant right-singular vector of the centred points (its sign
    is arbitrary, which is fine — ray-plane intersection is sign-invariant)."""
    P = np.asarray(points, dtype=float)
    c = P.mean(axis=0)
    _U, _S, Vt = np.linalg.svd(P - c)
    n = Vt[-1]
    return c, n / np.linalg.norm(n)


def plane_basis(plane_normal: np.ndarray):
    """Right-handed orthonormal in-plane basis ``(e1, e2)`` spanning the plane with
    the given normal. Deterministic (REV04 §6): seed e1 from the world axis *least*
    aligned with the normal so the projection never degenerates, then
    ``e2 = n × e1``. A stable normal → a stable basis, so the same physical table
    point always maps to the same ``(u,v)`` across captures."""
    n = np.asarray(plane_normal, dtype=float)
    n = n / np.linalg.norm(n)
    seed = np.eye(3)[int(np.argmin(np.abs(n)))]
    e1 = seed - np.dot(seed, n) * n
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2


def plane_coords(points: np.ndarray, plane_point: np.ndarray,
                 plane_normal: np.ndarray) -> np.ndarray:
    """Project world-frame point(s) onto the plane's 2-D ``(u,v)`` basis with origin
    at ``plane_point`` (REV04 §1, the planar calibration coordinate). Accepts a
    single point ``(3,)`` → ``(2,)`` or a stack ``(N,3)`` → ``(N,2)``. Any
    out-of-plane component is dropped (orthogonal projection)."""
    P = np.asarray(points, dtype=float)
    single = P.ndim == 1
    P = np.atleast_2d(P)
    c = np.asarray(plane_point, dtype=float)
    e1, e2 = plane_basis(plane_normal)
    d = P - c
    uv = np.stack([d @ e1, d @ e2], axis=1)
    return uv[0] if single else uv


def world_from_plane_coords(uv: np.ndarray, plane_point: np.ndarray,
                            plane_normal: np.ndarray) -> np.ndarray:
    """Inverse of `plane_coords` for in-plane points: ``(u,v)`` → world-frame point
    on the plane. Used by the coverage UI (draw cell centres back in the scene) and
    for round-trip tests. ``(2,)`` → ``(3,)`` or ``(N,2)`` → ``(N,3)``."""
    uv = np.asarray(uv, dtype=float)
    single = uv.ndim == 1
    uv = np.atleast_2d(uv)
    c = np.asarray(plane_point, dtype=float)
    e1, e2 = plane_basis(plane_normal)
    P = c + uv[:, [0]] * e1 + uv[:, [1]] * e2
    return P[0] if single else P


def build_world_map(cam_poses_by_id: Dict[int, np.ndarray],
                    ref_id: Optional[int] = None) -> Dict:
    """Build the rigid tag map from one registration observation.

    Args:
        cam_poses_by_id: ``{tag_id: T_cam_tag}`` — each world tag's pose in the
            camera frame (already averaged over the registration window).
        ref_id: which tag defines the world frame; default = the lowest id present.

    Returns:
        ``{"ref_id", "ids", "rel": {id: T_ref_id}, "plane_point", "plane_normal"}``.
        ``T_ref_id`` is tag ``id``'s pose in the world (reference-tag) frame
        (``rel[ref_id]`` = identity). ``plane_point``/``plane_normal`` are the
        table plane in the world frame, **fit from all tag origins** when ≥3 tags
        are present — so the depth plane's orientation is the consensus of all
        coplanar tags, not one (possibly noisy/tilted) tag's normal. With <3 tags
        it falls back to the reference tag's own plane (world z=0).
    """
    ids = sorted(int(i) for i in cam_poses_by_id)
    if not ids:
        raise ValueError("build_world_map needs at least one tag")
    ref = int(ref_id) if ref_id is not None else ids[0]
    if ref not in ids:
        raise ValueError(f"ref_id {ref} not among observed tags {ids}")
    T_cam_ref = np.asarray(cam_poses_by_id[ref], dtype=float)
    T_ref_cam = invert_transform(T_cam_ref)
    rel = {int(i): T_ref_cam @ np.asarray(cam_poses_by_id[i], dtype=float) for i in ids}
    if len(ids) >= 3:
        plane_point, plane_normal = fit_plane(np.array([rel[i][:3, 3] for i in ids]))
    else:
        plane_point, plane_normal = np.zeros(3), np.array([0.0, 0.0, 1.0])
    return {"ref_id": ref, "ids": ids, "rel": rel,
            "plane_point": plane_point, "plane_normal": plane_normal}


def recover_world_pose(tags_in_view: Dict[int, np.ndarray],
                       world_map: Dict) -> Optional[np.ndarray]:
    """Fuse a world-frame pose ``T_cam_world`` from whichever mapped tags are
    currently visible. Returns None if no mapped tag is in view.

    For each visible mapped tag ``i``: ``T_cam_world = T_cam_i · (T_ref_i)⁻¹``
    (the world = reference frame, seen via tag ``i``). Estimates from all visible
    mapped tags are averaged (`average_pose`)."""
    rel = world_map.get("rel", {})
    ests: List[np.ndarray] = []
    for tag_id, T_cam_i in tags_in_view.items():
        key = int(tag_id)
        if key not in rel:
            continue
        T_cam_i = np.asarray(T_cam_i, dtype=float)
        ests.append(T_cam_i @ invert_transform(np.asarray(rel[key], dtype=float)))
    if not ests:
        return None
    if len(ests) == 1:
        return ests[0]
    return average_pose(ests)


def world_map_to_arrays(world_map: Dict):
    """Flatten a map to npz-friendly arrays:
    ``(ref_id, ids[int], rels[N,4,4], plane_point[3], plane_normal[3])``."""
    ids = list(world_map["ids"])
    rels = np.stack([world_map["rel"][int(i)] for i in ids])
    return (int(world_map["ref_id"]), np.asarray(ids, dtype=int), rels,
            np.asarray(world_map["plane_point"], dtype=float),
            np.asarray(world_map["plane_normal"], dtype=float))


def world_map_from_arrays(ref_id, ids, rels, plane_point, plane_normal) -> Dict:
    """Inverse of ``world_map_to_arrays`` (rebuild after np.load)."""
    ids = [int(i) for i in np.asarray(ids).ravel()]
    rels = np.asarray(rels, dtype=float)
    return {"ref_id": int(ref_id), "ids": ids,
            "rel": {ids[k]: rels[k] for k in range(len(ids))},
            "plane_point": np.asarray(plane_point, dtype=float).ravel(),
            "plane_normal": np.asarray(plane_normal, dtype=float).ravel()}
