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


def table_normal_from_rel(rel: Dict[int, np.ndarray], ids: List[int]) -> np.ndarray:
    """Table-plane normal from the world tags' shared ORIENTATION — the mean of each
    tag's +Z face normal in the world frame — not a plane fit through their origin
    positions.

    The world tags are taped coplanar on the flat table (operator 2026-06-24), so
    every tag's +Z is the table normal. Their orientations are reliable (rev03
    detect: sub-degree rotation jitter) while their estimated origin HEIGHTS are not
    (the single-tag depth ambiguity). Fitting a plane through the noisy estimated
    origins tilted the HIL table normal 65° from vertical even though the tags are
    physically coplanar, which mis-projected the elevated EE tag relative to the gaze
    ray into a large systematic control-test offset (2026-06-24). Averaging the
    (reliable) orientations recovers the true near-vertical normal. Signs are aligned
    to the first tag so a 180°-flipped detection does not cancel the mean."""
    normals: List[np.ndarray] = []
    ref_n: Optional[np.ndarray] = None
    for i in ids:
        n = np.asarray(rel[int(i)], dtype=float)[:3, 2]
        if ref_n is None:
            ref_n = n
        normals.append(n if float(n @ ref_n) >= 0.0 else -n)
    m = np.mean(normals, axis=0)
    return m / np.linalg.norm(m)


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
        (``rel[ref_id]`` = identity). ``plane_point`` is the centroid of the tag
        origins; ``plane_normal`` is the mean of the tags' +Z face normals
        (`table_normal_from_rel`) when ≥3 tags are present — the tags are coplanar on
        the flat table, and their orientations are reliable where their estimated
        origin heights are not (origin-fit tilted the HIL plane 65°). With <3 tags it
        falls back to the reference tag's own plane (world z=0).
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
        plane_point = np.mean([rel[i][:3, 3] for i in ids], axis=0)
        plane_normal = table_normal_from_rel(rel, ids)
    else:
        plane_point, plane_normal = np.zeros(3), np.array([0.0, 0.0, 1.0])
    return {"ref_id": ref, "ids": ids, "rel": rel,
            "plane_point": plane_point, "plane_normal": plane_normal}


_CONSENSUS_TOL_MM = 60.0


def recover_world_pose(tags_in_view: Dict[int, np.ndarray],
                       world_map: Dict, *,
                       consensus_tol_mm: float = _CONSENSUS_TOL_MM) -> Optional[np.ndarray]:
    """Fuse a world-frame pose ``T_cam_world`` from whichever mapped tags are
    currently visible. Returns None if no mapped tag is in view.

    For each visible mapped tag ``i``: ``T_cam_world = T_cam_i · (T_ref_i)⁻¹``
    (the world = reference frame, seen via tag ``i``).

    **Robust consensus (2026-06-24).** A single planar tag's pose can flip to its
    alternate solution (the ``more than one minima`` ambiguity, worst under
    head-worn motion); its world-frame estimate then lands far from the others, and
    a plain mean would jump the whole frame — the operator-observed flit between
    distant cells. With ≥3 estimates, keep the LARGEST cluster whose world origins
    (each estimate's translation, mm) agree within ``consensus_tol_mm`` and fuse
    only those, so one flipped tag among several is out-voted (a 5-tag world map is
    hard to flip). Small per-tag noise stays inside the tolerance, so the fused
    result is unchanged when no tag has flipped. With 1–2 estimates there is no
    majority to appeal to, so they are fused directly (`average_pose`)."""
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
    if len(ests) <= 2:
        return ests[0] if len(ests) == 1 else average_pose(ests)
    origins = np.array([T[:3, 3] for T in ests])
    best: List[int] = []
    for i in range(len(ests)):
        inliers = [j for j in range(len(ests))
                   if np.linalg.norm(origins[j] - origins[i]) <= consensus_tol_mm]
        if len(inliers) > len(best):
            best = inliers
    return average_pose([ests[k] for k in best]) if len(best) > 1 else ests[best[0]]


def register_world_map_multiview(frames: List[Dict[int, np.ndarray]],
                                 ref_id: Optional[int] = None, *,
                                 outlier_tol_mm: float = _CONSENSUS_TOL_MM) -> Dict:
    """Robust world-tag map from MANY viewpoints (REV05, 2026-06-25). Replaces the
    single-viewpoint window of `_register_tag_map`, which could not defeat the
    single-tag depth ambiguity: averaging one biased view only sharpened the bias,
    so each registration produced a different tilted/displaced map (≈70° normal
    swing, ≈300 mm origin wander run-to-run) and the calibration was a coin-flip.

    The world tags are STATIC, COPLANAR at one height, and all face UP (operator
    2026-06-25). This fuses a head sweep over them into ONE reproducible map by
    exploiting all three:

      - **static + many views → triangulate.** For every frame where the reference
        tag is co-visible, each other tag yields an estimate of its pose *in the
        reference frame* (`T_ref_i = T_ref_cam · T_cam_i`). The per-view depth bias
        points along that view's camera axis, so averaging `T_ref_i` over
        geometrically diverse views decorrelates the bias toward the true static
        geometry — what one static window cannot do.
      - **coplanar at one height → project out depth.** All averaged tag origins are
        projected onto their common best plane, zeroing the residual per-tag height
        the depth ambiguity injects.
      - **all up → one shared normal.** `table_normal_from_rel` already takes the
        flip-aligned mean of the tags' +Z; over the multi-view-averaged `rel` it is
        the single, stable table normal.

    Args:
        frames: per-video-frame detections ``[{tag_id: T_cam_tag}, ...]`` collected
            across the head sweep (tag poses already in the camera frame, mm).
        ref_id: the tag defining the world frame; default = the tag seen in the most
            frames (ties → lowest id), the most-constrained anchor.
        outlier_tol_mm: a per-tag relative-pose estimate whose translation is farther
            than this from that tag's median estimate is dropped before averaging —
            rejects the occasional flipped view so it cannot drag the fused pose.

    Returns the same dict shape as `build_world_map`
    (``ref_id, ids, rel, plane_point, plane_normal``), so it is a drop-in for the
    downstream solve / `recover_world_pose` runtime.
    """
    seen_count: Dict[int, int] = {}
    for fr in frames:
        for i in fr:
            seen_count[int(i)] = seen_count.get(int(i), 0) + 1
    if not seen_count:
        raise ValueError("register_world_map_multiview needs ≥1 detection")
    if ref_id is None:
        ref = min(seen_count, key=lambda i: (-seen_count[i], i))
    else:
        ref = int(ref_id)
        if ref not in seen_count:
            raise ValueError(f"ref_id {ref} never observed across the {len(frames)} frames")

    # Per-frame relative poses, anchored on the reference tag (skip frames missing it).
    obs: Dict[int, List[np.ndarray]] = {}
    for fr in frames:
        if ref not in fr:
            continue
        T_ref_cam = invert_transform(np.asarray(fr[ref], dtype=float))
        for i, T in fr.items():
            obs.setdefault(int(i), []).append(T_ref_cam @ np.asarray(T, dtype=float))
    if ref not in obs:
        raise ValueError(f"ref tag {ref} never co-visible with itself — no usable frames")

    # Robust-average each tag's relative pose; drop translation outliers (flips).
    rel: Dict[int, np.ndarray] = {}
    for i, mats in obs.items():
        ts = np.array([M[:3, 3] for M in mats])
        med = np.median(ts, axis=0)
        keep = [k for k in range(len(mats)) if np.linalg.norm(ts[k] - med) <= outlier_tol_mm]
        if not keep:                       # all dispersed (rare): fall back to all
            keep = list(range(len(mats)))
        rel[int(i)] = average_pose([mats[k] for k in keep])

    ids = sorted(rel)
    if len(ids) >= 3:
        # Shared up-normal, then snap every origin onto the common plane (coplanar).
        plane_normal = table_normal_from_rel(rel, ids)
        origins = np.array([rel[i][:3, 3] for i in ids])
        centroid = origins.mean(axis=0)
        projected = origins - np.outer((origins - centroid) @ plane_normal, plane_normal)
        for k, i in enumerate(ids):
            rel[i] = make_transform(rel[i][:3, :3], projected[k])
        plane_point = projected.mean(axis=0)
    else:
        plane_point, plane_normal = np.zeros(3), np.array([0.0, 0.0, 1.0])
    return {"ref_id": ref, "ids": ids, "rel": rel,
            "plane_point": plane_point, "plane_normal": plane_normal}


def world_map_geometry_report(world_map: Dict, x_edge_ids: List[int],
                              y_edge_ids: List[int]) -> Dict:
    """Sanity-check a registered world map against the KNOWN physical layout
    (operator 2026-06-25: the world tags are coplanar, all facing up, taped on a
    flat table whose worked corner is ~90°). Returns the recovered in-plane angle
    between the +X and +Y table edges, the two edge lengths, and the worst tag
    out-of-plane deviation — so registration can REJECT a skewed/tilted map before it
    poisons a whole sweep.

    Why this gate exists: a head sweep with too little viewpoint diversity (operator
    sitting at a fixed oblique angle) leaves the single-tag depth ambiguity
    unresolved, so the multiview map comes out skewed — observed 2026-06-25 as a 58.7°
    corner where the table is ~90°. The geometry is wrong but every per-frame health
    counter reads fine, so we check the one thing we know a-priori: the corner is
    square and the tags are coplanar.

    ``x_edge_ids`` / ``y_edge_ids`` are tag-id sequences whose endpoints define the +X
    and +Y table edges (e.g. ``[0, 1]`` and ``[2, 0]``); the in-plane edge vector is
    ``pos(ids[-1]) - pos(ids[0])`` projected onto the map plane. A ``corner_angle_deg``
    of NaN means an edge tag was missing or degenerate."""
    rel = world_map["rel"]
    n = np.asarray(world_map["plane_normal"], dtype=float)
    n = n / np.linalg.norm(n)

    def pos(i: int) -> Optional[np.ndarray]:
        return np.asarray(rel[int(i)][:3, 3], dtype=float) if int(i) in rel else None

    def in_plane_edge(ids: List[int]) -> Optional[np.ndarray]:
        a, b = pos(ids[0]), pos(ids[-1])
        if a is None or b is None:
            return None
        v = b - a
        return v - (v @ n) * n

    ex, ey = in_plane_edge(x_edge_ids), in_plane_edge(y_edge_ids)
    angle = float("nan")
    lx = float(np.linalg.norm(ex)) if ex is not None else float("nan")
    ly = float(np.linalg.norm(ey)) if ey is not None else float("nan")
    if ex is not None and ey is not None and lx > 1e-6 and ly > 1e-6:
        angle = float(np.degrees(np.arccos(np.clip(ex @ ey / (lx * ly), -1.0, 1.0))))

    c = np.asarray(world_map["plane_point"], dtype=float)
    devs = [abs(float((np.asarray(rel[i][:3, 3], dtype=float) - c) @ n))
            for i in world_map["ids"]]
    return {"corner_angle_deg": angle, "x_edge_len_mm": lx, "y_edge_len_mm": ly,
            "max_out_of_plane_mm": float(max(devs)) if devs else 0.0}


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
