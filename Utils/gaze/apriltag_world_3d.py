"""
apriltag_world_3d.py — WS-4 first-pass: NON-coplanar (true 3-D) world-tag map.

The REV05 ``register_world_map_multiview`` in ``apriltag_world.py`` fuses a
head sweep into one reproducible world map, then SNAPS every tag origin onto
their common best-fit plane (apriltag_world.py:279-287). That snap is correct
for REV05's physical setup — the world tags are taped coplanar on a flat table
— and it actively projects out the single-tag depth ambiguity. But it also
*destroys any genuine 3-D structure*: tags on two different planes collapse to
one. WS-4 needs a world frame that spans more than the table (objects at
height, a back wall, a shelf), so this module builds a true-3-D world frame
WITHOUT the coplanar snap.

**Pose-graph fuse (2026-06-29).** The first 3-D pass reused REV05's
single-anchor build: every tag's pose was expressed RELATIVE to one reference
tag, so a tag rarely co-visible with the anchor was triangulated from few,
noisy estimates and — worse — *every* tag inherited the anchor's per-frame pose
flips (the ``more than one minima`` ambiguity). On a 12-tag multi-plane layout
this left most tags stuck at ~30 mm reproducibility no matter how the operator
moved. This module now fuses a **pose graph** over ALL co-visible tag pairs
instead:
  - reference-tag selection unchanged (most-seen, ties → lowest id) — but the
    reference is only a GAUGE (the coordinate origin), not a data bottleneck;
  - every co-visible pair contributes a relative-pose edge, robust-averaged
    across frames with the same translation-outlier gate (``average_pose`` +
    median/tolerance reject) so one flipped view per pair is dropped;
  - global tag rotations come from a max-co-visibility spanning tree from the
    reference; global tag positions from a least-squares solve over EVERY
    pairwise translation constraint ``p_b − p_a = R_a · t_ab``. One tag's flips
    are out-voted by the dozens of other edges, and a tag reachable only THROUGH
    other tags is still well-placed.

What is DROPPED: the coplanar projection. Tag origins keep their fused 3-D
positions, so a two-plane layout produces a true 3-D world frame.

What changes in the report: REV05's ``world_map_geometry_report`` gates on a
~90° table corner and near-zero out-of-plane deviation — assumptions that are
false by construction here. :func:`world_map_3d_geometry_report` instead
reports (a) how reproducible the fused positions are (per-tag RMS of the kept
per-frame estimates around the fused origin) and (b) the non-planarity of the
layout (max deviation of tag origins from their own best-fit plane) — the
positive signal that the 3-D structure was preserved rather than snapped flat.

The returned map dict shape is identical to ``build_world_map`` /
``register_world_map_multiview`` (``ref_id, ids, rel, plane_point,
plane_normal``), so ``recover_world_pose`` and ``world_map_to_arrays`` /
``world_map_from_arrays`` work on a 3-D map unchanged — only the meaning of
``plane_*`` differs (here it is the best-fit plane through the 3-D origins,
informational only; nothing is projected onto it).

Frozen-core note (WS-4 constraint): NEW SURFACE. Does not edit
``apriltag_world.py``; it imports the shared primitives (``average_pose``,
``fit_plane``) and the transform helpers. Pure numpy, hardware-free,
unit-tested.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from Utils.gaze.apriltag_calib import invert_transform, make_transform, umeyama_rigid
from Utils.gaze.apriltag_world import (
    _CONSENSUS_TOL_MM,
    average_pose,
    fit_plane,
    table_normal_from_rel,
)


def _pairwise_edges(frames: List[Dict[int, np.ndarray]], outlier_tol_mm: float):
    """Accumulate one robust relative-pose edge per co-visible tag pair.

    For every frame and every visible pair ``(a, b)`` with ``a < b`` the camera
    cancels in ``T_a_b = T_a_cam · T_cam_b`` (the pose of ``b`` in ``a``'s frame),
    so the edge is viewpoint-independent. Across frames each pair's edge is
    robust-averaged with the median/tolerance translation gate that drops a
    flipped per-frame view. Returns ``(edges, node_seen)`` with
    ``edges[(a, b)] = (T_a_b, kept_view_count)`` and ``node_seen[id] = frames``."""
    raw: Dict[tuple, List[np.ndarray]] = {}
    node_seen: Dict[int, int] = {}
    for fr in frames:
        ids = sorted(int(i) for i in fr)
        for i in ids:
            node_seen[i] = node_seen.get(i, 0) + 1
        for ia in range(len(ids)):
            for ib in range(ia + 1, len(ids)):
                a, b = ids[ia], ids[ib]
                T_a_b = (invert_transform(np.asarray(fr[a], dtype=float))
                         @ np.asarray(fr[b], dtype=float))
                raw.setdefault((a, b), []).append(T_a_b)
    edges: Dict[tuple, tuple] = {}
    for (a, b), mats in raw.items():
        ts = np.array([M[:3, 3] for M in mats])
        med = np.median(ts, axis=0)
        keep = [k for k in range(len(mats))
                if np.linalg.norm(ts[k] - med) <= outlier_tol_mm]
        if not keep:                       # all dispersed (rare): fall back to all
            keep = list(range(len(mats)))
        edges[(a, b)] = (average_pose([mats[k] for k in keep]), len(keep))
    return edges, node_seen


def _spanning_tree_rotations(ref: int, edges: Dict[tuple, tuple]):
    """Global tag rotations (gauge ``R_ref = I``) via a max-co-visibility spanning
    tree from the reference (Prim's, edge weight = kept-view count). Returns
    ``(R, reachable)`` — a rotation per tag connected to the reference and the set
    of those tags. Tags in a disconnected component can't be placed in the
    reference's frame and are dropped (re-shown upstream)."""
    import heapq
    adj: Dict[int, List[tuple]] = {}
    for (a, b), (T, w) in edges.items():
        R_ab = T[:3, :3]
        adj.setdefault(a, []).append((b, R_ab, w))
        adj.setdefault(b, []).append((a, R_ab.T, w))   # pose of a in b: inverse rotation
    R: Dict[int, np.ndarray] = {ref: np.eye(3)}
    reachable = {ref}
    heap: list = []
    counter = 0  # unique tiebreaker so heapq never compares the ndarray payload

    def _push(n: int) -> None:
        nonlocal counter
        for (m, R_nm, w) in adj.get(n, []):
            if m not in reachable:
                heapq.heappush(heap, (-int(w), counter, n, m, R_nm))
                counter += 1

    _push(ref)
    while heap:
        _, _, n, m, R_nm = heapq.heappop(heap)
        if m in reachable:
            continue
        R[m] = R[n] @ R_nm                 # R_world_m = R_world_n · R_n_m
        reachable.add(m)
        _push(m)
    return R, reachable


def _ls_positions(ref: int, reachable: set, edges: Dict[tuple, tuple],
                  R: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """Global tag positions (gauge ``p_ref = 0``) by least-squares over EVERY
    co-visible pair's constraint ``p_b − p_a = R_a · t_ab`` (weighted by
    kept-view count). Using all pairs — not chains through one anchor — is what
    makes the positions robust to any single tag's pose flips."""
    nodes = sorted(reachable)
    idx = {n: k for k, n in enumerate(nodes)}
    N = len(nodes)
    rows_A: List[np.ndarray] = []
    rows_b: List[np.ndarray] = []
    for (a, b), (T, w) in edges.items():
        if a not in reachable or b not in reachable:
            continue
        sw = np.sqrt(float(w))
        A = np.zeros((3, 3 * N))
        A[:, 3 * idx[b]:3 * idx[b] + 3] = np.eye(3)
        A[:, 3 * idx[a]:3 * idx[a] + 3] = -np.eye(3)
        rows_A.append(A * sw)
        rows_b.append((R[a] @ T[:3, 3]) * sw)
    if not rows_A:                          # lone reference tag, no pairs
        return {n: np.zeros(3) for n in nodes}
    A = np.vstack(rows_A)
    bvec = np.concatenate(rows_b)
    free = [c for c in range(3 * N) if c // 3 != idx[ref]]   # drop ref cols → p_ref=0
    sol, *_ = np.linalg.lstsq(A[:, free], bvec, rcond=None)
    p: Dict[int, np.ndarray] = {ref: np.zeros(3)}
    for k, n in enumerate(n for n in nodes if n != ref):
        p[n] = sol[3 * k:3 * k + 3]
    return p


def register_world_map_3d(frames: List[Dict[int, np.ndarray]],
                          ref_id: Optional[int] = None, *,
                          outlier_tol_mm: float = _CONSENSUS_TOL_MM) -> Dict:
    """True-3-D world-tag map from many viewpoints, fused as a POSE GRAPH over all
    co-visible tag pairs (no coplanar snap).

    Unlike the REV05 single-anchor build (every tag relative to one reference, so
    the reference's per-frame flips poison the whole map), this aggregates every
    co-visible pair: robust relative-pose edges → global rotations from a
    max-co-visibility spanning tree → global positions from a least-squares solve
    over all pairwise translation constraints. The reference tag is only the gauge
    (origin); no tag's noise dominates, and tags connected only through other tags
    are still well-placed. Use this for a non-coplanar workspace (objects at
    height, a wall, a shelf); use ``register_world_map_multiview`` when the tags
    are physically coplanar and the snap helps.

    Args:
        frames: per-video-frame detections ``[{tag_id: T_cam_tag}, ...]``
            across the head sweep (tag poses in the camera frame, mm).
        ref_id: the tag defining the world-frame gauge; default = the tag seen in
            the most frames (ties → lowest id).
        outlier_tol_mm: a per-pair relative-pose estimate whose translation is
            farther than this from that pair's median is dropped before averaging
            — rejects the occasional flipped view.

    Returns:
        ``{"ref_id", "ids", "rel": {id: T_ref_id}, "plane_point",
        "plane_normal"}``. Same shape as ``build_world_map`` so it is a drop-in
        for ``recover_world_pose`` and the array (de)serialisers. ``ids`` are the
        tags connected to the reference; any disconnected tag is omitted (re-shown
        upstream). ``plane_point``/``plane_normal`` describe the best-fit plane
        through the fused 3-D origins (informational; nothing is projected onto
        it). With <3 tags the plane falls back to ``(origin, +Z)``.

    Raises:
        ValueError: no detections, or the requested reference tag is never
            observed.
    """
    edges, node_seen = _pairwise_edges(frames, outlier_tol_mm)
    if not node_seen:
        raise ValueError("register_world_map_3d needs ≥1 detection")
    if ref_id is None:
        ref = min(node_seen, key=lambda i: (-node_seen[i], i))
    else:
        ref = int(ref_id)
        if ref not in node_seen:
            raise ValueError(
                f"ref_id {ref} never observed across the {len(frames)} frames")

    R, reachable = _spanning_tree_rotations(ref, edges)
    p = _ls_positions(ref, reachable, edges, R)
    rel = {i: make_transform(R[i], p[i]) for i in reachable}

    ids = sorted(rel)
    # NO coplanar snap — report the best-fit plane through the fused 3-D origins
    # for reference, but leave the origins in 3-D.
    if len(ids) >= 3:
        origins = np.array([rel[i][:3, 3] for i in ids])
        plane_point, plane_normal = fit_plane(origins)
    else:
        plane_point, plane_normal = np.zeros(3), np.array([0.0, 0.0, 1.0])
    return {"ref_id": ref, "ids": ids, "rel": rel,
            "plane_point": plane_point, "plane_normal": plane_normal}


def world_map_3d_geometry_report(world_map: Dict,
                                 frames: List[Dict[int, np.ndarray]]) -> Dict:
    """3-D sanity report for a map from :func:`register_world_map_3d`.

    REV05's ``world_map_geometry_report`` gates on a square table corner and
    near-zero out-of-plane deviation — both meaningless for a non-coplanar
    layout. This reports two things appropriate to 3-D:

      - **reproducibility** — for each tag, the RMS of its pairwise constraint
        residuals: over every frame and every co-visible pair, compare the
        measured relative position (``b`` in ``a``'s frame) to what the fused map
        predicts (``inv(rel[a]) · rel[b]``), and attribute the error to both
        tags. This is **anchor-free** — unlike a single-tag spread it does not
        inherit the reference tag's per-frame flip noise, so a globally
        consistent pose-graph map reads low even when one tag flips occasionally.
        A small ``max_fit_residual_mm`` means the map explains the raw
        observations; a large one means it cannot and should be rejected (the
        3-D analogue of REV05's skew gate).
      - **non-planarity** — ``max_out_of_plane_mm`` is the largest deviation of
        a fused tag origin from the best-fit plane through all origins. For a
        genuine multi-plane layout this is LARGE; if it is near zero the layout
        was effectively coplanar and the cheaper REV05 path would have done.
        This is the positive check that 3-D structure was preserved, not
        snapped flat.

    Args:
        world_map: a map dict from :func:`register_world_map_3d`.
        frames: the same per-frame detections passed to registration (needed to
            recompute per-frame estimates for the residual).

    Returns:
        ``{"max_fit_residual_mm", "mean_fit_residual_mm", "per_tag_residual_mm",
        "max_out_of_plane_mm", "z_spread_mm", "num_tags"}``. ``z_spread_mm`` is
        the peak-to-peak extent of the origins along the best-fit plane normal —
        the raw amount of out-of-plane structure the snap would have erased.
        ``per_tag_residual_mm`` maps ``tag_id`` → its RMS residual (mm).
    """
    rel = world_map["rel"]

    # Anchor-free reproducibility: over every co-visible pair in every frame,
    # compare the measured relative position (b in a's frame) to the fused map's
    # prediction, drop flipped views with the same gate as the fuse, and attribute
    # the squared error to both tags. A tag's RMS over all its pair-residuals is
    # then how well the global map explains the raw observations touching it.
    per_tag_sq: Dict[int, List[float]] = {}
    for fr in frames:
        ids_fr = sorted(int(i) for i in fr if int(i) in rel)
        for ia in range(len(ids_fr)):
            for ib in range(ia + 1, len(ids_fr)):
                a, b = ids_fr[ia], ids_fr[ib]
                meas = (invert_transform(np.asarray(fr[a], dtype=float))
                        @ np.asarray(fr[b], dtype=float))[:3, 3]
                pred = (invert_transform(rel[a]) @ rel[b])[:3, 3]
                d = float(np.linalg.norm(meas - pred))
                if d > _CONSENSUS_TOL_MM:   # flipped per-frame view — same gate as the fuse
                    continue
                per_tag_sq.setdefault(a, []).append(d * d)
                per_tag_sq.setdefault(b, []).append(d * d)

    per_tag: Dict[int, float] = {
        i: float(np.sqrt(np.mean(v))) for i, v in per_tag_sq.items() if v}
    residuals = list(per_tag.values())
    ids = list(world_map["ids"])
    origins = np.array([np.asarray(rel[int(i)][:3, 3], dtype=float)
                        for i in ids])
    if len(ids) >= 3:
        c, n = fit_plane(origins)
        signed = (origins - c) @ n
        max_oop = float(np.max(np.abs(signed)))
        z_spread = float(np.ptp(signed))
    else:
        # Too few origins to fit a plane; non-planarity is undefined.
        max_oop = float("nan")
        z_spread = float("nan")

    return {
        "max_fit_residual_mm": float(max(residuals)) if residuals else 0.0,
        "mean_fit_residual_mm": float(np.mean(residuals)) if residuals else 0.0,
        "per_tag_residual_mm": per_tag,
        "max_out_of_plane_mm": max_oop,
        "z_spread_mm": z_spread,
        "num_tags": len(ids),
    }


def world_map_3d_reproducibility(frames: List[Dict[int, np.ndarray]], *,
                                 seed: int = 0) -> Dict[int, float]:
    """Split-half reproducibility of the fused map (per-tag, mm) — the
    ground-truth-free estimate of map ACCURACY.

    The per-frame ``world_map_3d_geometry_report`` residual measures raw sensor
    *scatter* (the AprilTag per-frame pose noise), which floors out and barely
    improves with more views — so it is the wrong thing to gate registration on.
    What matters is whether the FUSED map is accurate, i.e. reproducible from
    independent data. This randomly splits the frames into two halves, fuses each
    into its own pose-graph map, rigidly aligns the two on their common tags, and
    returns per tag the distance between the two independent estimates of that
    tag's position. Small ⇒ the map is reproducible (the original "≈300 mm
    run-to-run wander" was exactly large disagreement here); large ⇒ the geometry
    is not pinned down regardless of how low the per-frame scatter looks.

    Returns ``{tag_id: disagreement_mm}``; empty if there are too few frames, a
    half fails to fuse, or fewer than 3 common tags to align on. The half-map
    disagreement slightly OVER-estimates the full-map error (each half has half
    the data), so it is a conservative gate.
    """
    if len(frames) < 4:
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(frames))
    h1 = [frames[i] for i in idx[:len(frames) // 2]]
    h2 = [frames[i] for i in idx[len(frames) // 2:]]
    try:
        m1 = register_world_map_3d(h1)
        m2 = register_world_map_3d(h2)
    except ValueError:
        return {}
    common = [i for i in m1["ids"] if i in m2["ids"]]
    if len(common) < 3:
        return {}
    A = np.array([m1["rel"][i][:3, 3] for i in common], dtype=float)
    B = np.array([m2["rel"][i][:3, 3] for i in common], dtype=float)
    T, _ = umeyama_rigid(A, B)
    A_aligned = (T[:3, :3] @ A.T).T + T[:3, 3]
    return {int(common[k]): float(np.linalg.norm(A_aligned[k] - B[k]))
            for k in range(len(common))}


# ── structural constraints (Tier-1 hardening: known coplanar/perpendicular rig) ──


def _orthogonalize_normals(n1: np.ndarray, n2: np.ndarray):
    """Closest perpendicular unit pair to ``(n1, n2)``, staying in their span — used
    to enforce that two physically-perpendicular planes (table ⊥ wall) read 90°
    despite registration noise. Equal split (both move the same amount). Returns
    ``(n1', n2')``; if the inputs are ~parallel (degenerate) returns them unchanged."""
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    w = n2 - (n2 @ n1) * n1
    wn = float(np.linalg.norm(w))
    if wn < 1e-9:
        return n1, n2
    e1, e2 = n1, w / wn
    th = np.arctan2(float(n2 @ e2), float(n2 @ e1))   # angle of n2 from n1 (n1 at 0)
    a1, a2 = th / 2 - np.pi / 4, th / 2 + np.pi / 4
    return (np.cos(a1) * e1 + np.sin(a1) * e2), (np.cos(a2) * e1 + np.sin(a2) * e2)


def _align_z(R: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Minimal rotation of ``R`` so its +Z column becomes ``n`` (keeps the in-plane
    spin). Used to make a group's tags share the group plane normal."""
    z = R[:, 2] / np.linalg.norm(R[:, 2])
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    c = float(np.clip(z @ n, -1.0, 1.0))
    if c > 1.0 - 1e-12:
        return R.copy()
    if c < -1.0 + 1e-9:                          # opposite: 180° about any axis ⟂ z
        a = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(z, a); axis /= np.linalg.norm(axis); angle = np.pi
    else:
        axis = np.cross(z, n); s = float(np.linalg.norm(axis)); axis /= s
        angle = np.arctan2(s, c)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    Rrot = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    return Rrot @ R


def apply_plane_structure(world_map: Dict, groups: Dict[str, List[int]],
                          perpendicular_pairs=()) -> tuple:
    """Impose the rig's known structure on a fused 3-D world map (Tier-1 hardening):
    each named ``groups`` set is made coplanar, and each ``perpendicular_pairs``
    group pair is made 90°.

    The single-tag depth ambiguity leaves a fused map non-flat (table tags scatter
    in depth) and non-square (table↔wall ≠ 90°) even though the rig is physically
    flat + perpendicular. Using those facts: per group, take the (reliable)
    orientation-average normal (``table_normal_from_rel``) and origin centroid;
    orthogonalize the perpendicular pair's normals; then **snap** each tag's origin
    onto its group plane and **re-align** its rotation so +Z = the group normal. The
    result is structurally correct — board-PnP recovery (which uses the tag origins)
    and the table-plane target both ride on a flat, square map.

    ``groups``: e.g. ``{"table": [0,1,2,3,4,12], "wall": [6,7,8,9,10,11]}``.
    ``perpendicular_pairs``: e.g. ``[("table", "wall")]``. Groups with <3 mapped
    tags are skipped. Returns ``(structured_map, report)`` — ``report`` has per-group
    pre-snap flatness (how far the registration was from coplanar) and per-pair
    pre-orthogonalization angle + how far each normal moved (the squareness gate)."""
    rel = {int(i): np.asarray(world_map["rel"][i], dtype=float).copy()
           for i in world_map["rel"]}
    planes: Dict[str, Dict] = {}
    for name, ids in groups.items():
        present = [int(i) for i in ids if int(i) in rel]
        if len(present) < 3:
            continue
        planes[name] = {"ids": present,
                        "normal": table_normal_from_rel(rel, present),
                        "point": np.mean([rel[i][:3, 3] for i in present], axis=0)}

    report: Dict = {"groups": {}, "perpendicular": {}}
    for a, b in perpendicular_pairs:
        if a in planes and b in planes:
            na, nb = planes[a]["normal"], planes[b]["normal"]
            ang = float(np.degrees(np.arccos(abs(np.clip(na @ nb, -1.0, 1.0)))))
            na2, nb2 = _orthogonalize_normals(na, nb)
            planes[a]["normal"], planes[b]["normal"] = na2, nb2
            report["perpendicular"][f"{a}-{b}"] = {
                "angle_before_deg": ang,
                "normal_moved_deg": float(np.degrees(np.arccos(abs(np.clip(na @ na2, -1.0, 1.0))))),
            }

    for name, pl in planes.items():
        n, c, resid = pl["normal"], pl["point"], []
        for i in pl["ids"]:
            d = float((rel[i][:3, 3] - c) @ n)
            resid.append(abs(d))
            rel[i][:3, 3] = rel[i][:3, 3] - d * n          # snap onto the plane
            rel[i][:3, :3] = _align_z(rel[i][:3, :3], n)   # +Z = group normal
        resid = np.asarray(resid)
        report["groups"][name] = {
            "ids": pl["ids"],
            "flatness_max_mm": float(resid.max()),
            "flatness_rms_mm": float(np.sqrt(np.mean(resid ** 2))),
        }

    out = dict(world_map)
    out["rel"] = rel
    ids = sorted(rel)
    if len(ids) >= 3:                                       # refresh informational plane
        origins = np.array([rel[i][:3, 3] for i in ids])
        out["plane_point"], out["plane_normal"] = fit_plane(origins)
    return out, report
