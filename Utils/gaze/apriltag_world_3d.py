"""
apriltag_world_3d.py — WS-4 first-pass: NON-coplanar (true 3-D) world-tag map.

The REV05 ``register_world_map_multiview`` in ``apriltag_world.py`` fuses a
head sweep into one reproducible world map, then SNAPS every tag origin onto
their common best-fit plane (apriltag_world.py:279-287). That snap is correct
for REV05's physical setup — the world tags are taped coplanar on a flat table
— and it actively projects out the single-tag depth ambiguity. But it also
*destroys any genuine 3-D structure*: tags on two different planes collapse to
one. WS-4 needs a world frame that spans more than the table (objects at
height, a back wall, a shelf), so this module runs the SAME multi-view fuse
WITHOUT the coplanar snap.

What is reused (the multiview fuse machinery, unchanged in spirit):
  - reference-tag selection (most-seen tag, ties → lowest id),
  - per-frame relative pose ``T_ref_i = T_ref_cam · T_cam_i`` anchored on the
    reference tag,
  - robust per-tag averaging with a translation-outlier gate that drops
    flipped PnP views (``average_pose`` + the median/tolerance reject).

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

from Utils.gaze.apriltag_calib import invert_transform
from Utils.gaze.apriltag_world import _CONSENSUS_TOL_MM, average_pose, fit_plane


def register_world_map_3d(frames: List[Dict[int, np.ndarray]],
                          ref_id: Optional[int] = None, *,
                          outlier_tol_mm: float = _CONSENSUS_TOL_MM) -> Dict:
    """True-3-D world-tag map from many viewpoints — the multi-view fuse of
    ``register_world_map_multiview`` WITHOUT the coplanar snap.

    Identical fuse to REV05 (anchor on the most-seen reference tag, average
    each tag's relative pose over frames, reject translation outliers from
    flipped PnP views) but the tag origins keep their fused 3-D positions, so
    tags on multiple planes give a genuine 3-D world frame rather than a flat
    table. Use this when the calibrated workspace is not coplanar (objects at
    height, a wall, a shelf); use ``register_world_map_multiview`` when the
    tags are physically coplanar and the snap helps.

    Args:
        frames: per-video-frame detections ``[{tag_id: T_cam_tag}, ...]``
            across the head sweep (tag poses in the camera frame, mm).
        ref_id: the tag defining the world frame; default = the tag seen in the
            most frames (ties → lowest id), the most-constrained anchor.
        outlier_tol_mm: a per-tag relative-pose estimate whose translation is
            farther than this from that tag's median estimate is dropped before
            averaging — rejects the occasional flipped view.

    Returns:
        ``{"ref_id", "ids", "rel": {id: T_ref_id}, "plane_point",
        "plane_normal"}``. Same shape as ``build_world_map`` so it is a drop-in
        for ``recover_world_pose`` and the array (de)serialisers. Here
        ``plane_point``/``plane_normal`` describe the best-fit plane through the
        fused 3-D origins (informational; nothing is projected onto it). With
        <3 tags the plane falls back to ``(origin, +Z)`` — undetermined from so
        few points, and unused downstream.

    Raises:
        ValueError: no detections, or the requested/derived reference tag is
            never co-visible with itself (no usable frames).
    """
    seen_count: Dict[int, int] = {}
    for fr in frames:
        for i in fr:
            seen_count[int(i)] = seen_count.get(int(i), 0) + 1
    if not seen_count:
        raise ValueError("register_world_map_3d needs ≥1 detection")
    if ref_id is None:
        ref = min(seen_count, key=lambda i: (-seen_count[i], i))
    else:
        ref = int(ref_id)
        if ref not in seen_count:
            raise ValueError(
                f"ref_id {ref} never observed across the {len(frames)} frames")

    # Per-frame relative poses, anchored on the reference tag (skip frames
    # missing it). Same construction as register_world_map_multiview.
    obs: Dict[int, List[np.ndarray]] = {}
    for fr in frames:
        if ref not in fr:
            continue
        T_ref_cam = invert_transform(np.asarray(fr[ref], dtype=float))
        for i, T in fr.items():
            obs.setdefault(int(i), []).append(
                T_ref_cam @ np.asarray(T, dtype=float))
    if ref not in obs:
        raise ValueError(
            f"ref tag {ref} never co-visible with itself — no usable frames")

    # Robust-average each tag's relative pose; drop translation outliers
    # (flipped PnP views) via the same median/tolerance gate as REV05.
    rel: Dict[int, np.ndarray] = {}
    for i, mats in obs.items():
        ts = np.array([M[:3, 3] for M in mats])
        med = np.median(ts, axis=0)
        keep = [k for k in range(len(mats))
                if np.linalg.norm(ts[k] - med) <= outlier_tol_mm]
        if not keep:                       # all dispersed (rare): fall back to all
            keep = list(range(len(mats)))
        rel[int(i)] = average_pose([mats[k] for k in keep])

    ids = sorted(rel)
    # NO coplanar snap — this is the whole point. Report the best-fit plane
    # through the fused 3-D origins for reference, but leave the origins in 3-D.
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

      - **reproducibility** — for each tag, the RMS distance of its kept
        per-frame relative-position estimates (recomputed from ``frames``,
        anchored on the map's reference tag, same outlier gate) to the fused
        origin in the map. A small ``max_fit_residual_mm`` means the sweep had
        enough viewpoint diversity to triangulate the static geometry; a large
        one means the fuse averaged inconsistent estimates and the map should
        be rejected (the 3-D analogue of REV05's skew gate).
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
    ref = int(world_map["ref_id"])

    # Recompute per-frame relative origins per tag, anchored on the reference
    # tag, applying the SAME outlier gate registration used so the residual is
    # measured over the estimates that actually contributed to the fuse.
    obs: Dict[int, List[np.ndarray]] = {}
    for fr in frames:
        if ref not in fr:
            continue
        T_ref_cam = invert_transform(np.asarray(fr[ref], dtype=float))
        for i, T in fr.items():
            key = int(i)
            if key not in rel:
                continue
            obs.setdefault(key, []).append(
                (T_ref_cam @ np.asarray(T, dtype=float))[:3, 3])

    per_tag: Dict[int, float] = {}
    for i, pts in obs.items():
        ts = np.asarray(pts, dtype=float)
        med = np.median(ts, axis=0)
        keep = ts[np.linalg.norm(ts - med, axis=1) <= _CONSENSUS_TOL_MM]
        if keep.shape[0] == 0:
            keep = ts
        fused = np.asarray(rel[i][:3, 3], dtype=float)
        per_tag[int(i)] = float(np.sqrt(np.mean(
            np.sum((keep - fused) ** 2, axis=1))))

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
