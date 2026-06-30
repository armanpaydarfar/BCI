"""Analyze_world_map_ba.py — WS-4 Tier-2: constrained bundle adjustment of the
3-D AprilTag world map from corner reprojection + the rig's planar/perpendicular
structure.

OFFLINE ANALYSIS ONLY. Reads a register-world-3d npz that carries per-frame tag
CORNER observations (``corner_obs``, saved since 2026-06-30) plus the fused world
map, and refines the tag geometry by minimising the 2-D corner reprojection error
across all frames SUBJECT TO the known structure: the table tags are coplanar, the
wall tags are coplanar, and the two planes are perpendicular. This resolves the
single-tag depth ambiguity the per-frame pose can't (the pose-graph fuse left the
table ~80 mm non-flat and table↔wall at 78.5°). Writes a refined
``world_map_3d_*_ba.npz`` the sweep/control chain can use via ``--world-map``.

Model (standard SfM BA with a known-size square per tag + structural priors):
  unknowns = each tag's world pose (rvec,tvec) [the reference tag is the fixed
  gauge] + each frame's camera pose (rvec,tvec) + the two planes (normal as
  spherical angles + offset). Residuals = corner reprojection (px) and, weighted,
  the coplanarity / perpendicularity penalties. Solved with scipy least_squares
  (TRF + a sparsity pattern). The corner order is auto-selected at init by the
  order that reprojects best on the fused map (detector-convention-agnostic).

Usage:
    python Analyze_world_map_ba.py runs/world_map_3d_<UTC>.npz \
        --table-tag-ids 0 1 2 3 4 12 --wall-tag-ids 6 7 8 9 10 11
    python Analyze_world_map_ba.py --self-test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── geometry helpers (pure) ─────────────────────────────────────────────────

def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Rotation matrix from a rotation vector (no cv2 dependency in the core)."""
    rvec = np.asarray(rvec, dtype=float)
    th = float(np.linalg.norm(rvec))
    if th < 1e-12:
        return np.eye(3)
    k = rvec / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _inv_rodrigues(R: np.ndarray) -> np.ndarray:
    """Rotation vector from a rotation matrix."""
    c = (np.trace(R) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    th = np.arccos(c)
    if th < 1e-12:
        return np.zeros(3)
    if abs(th - np.pi) < 1e-6:                       # 180°: use the symmetric form
        # axis from the largest diagonal of (R+I)/2
        A = (R + np.eye(3)) / 2.0
        k = np.sqrt(np.clip(np.diag(A), 0, None))
        i = int(np.argmax(k))
        k = k / (np.linalg.norm(k) + 1e-12)
        return th * k
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return th * ax / (2.0 * np.sin(th))


def _normal_from_angles(theta: float, phi: float) -> np.ndarray:
    """Unit normal from spherical angles (so the BA normal is unit by construction)."""
    return np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)])


def _angles_from_normal(n: np.ndarray) -> Tuple[float, float]:
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    return float(np.arccos(np.clip(n[2], -1, 1))), float(np.arctan2(n[1], n[0]))


def tag_object_corners(size_m: float, order: int = 0) -> np.ndarray:
    """The tag's 4 corners in its own frame (mm, z=0), as an (4,3) array. ``order``
    cyclically rotates the corner sequence (0..3) so it can be matched to the
    detector's corner convention; the base order wraps CCW."""
    s = 1000.0 * size_m / 2.0
    base = np.array([[-s, -s, 0.0], [s, -s, 0.0], [s, s, 0.0], [-s, s, 0.0]])
    return np.roll(base, order, axis=0)


# ── BA core ─────────────────────────────────────────────────────────────────

def _residuals(params, layout, obs_arr, K, obj, group_pos, struct_w):
    """Reprojection (px) + structural (weighted) residuals — VECTORISED (no per-obs
    Python loop, or the BA is unusably slow). ``obs_arr`` = (tag_pos, frame_pos,
    corner_px (n,4,2)); ``group_pos`` = {name: tag-position array}; rows in the same
    order as ``_jac_sparsity`` (per obs: u0..3 then v0..3; then coplanarity; then ⟂)."""
    tag_R, tag_t, cam_R, cam_t, planes = _unpack(params, layout)
    tpos, fpos, px = obs_arr
    Rt, tt = tag_R[tpos], tag_t[tpos]
    Rc, tc = cam_R[fpos], cam_t[fpos]
    pw = np.einsum("oij,cj->oci", Rt, obj) + tt[:, None, :]   # (n_obs,4,3) world
    pc = np.einsum("oij,ocj->oci", Rc, pw) + tc[:, None, :]   # camera
    z = pc[:, :, 2]
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = K[0, 0] * pc[:, :, 0] / z + K[0, 2]
    v = K[1, 1] * pc[:, :, 1] / z + K[1, 2]
    parts = [np.concatenate([u - px[:, :, 0], v - px[:, :, 1]], axis=1).ravel()]
    for name, positions in group_pos.items():
        n, d = planes[name]
        parts.append(struct_w * (tag_t[positions] @ n - d))      # coplanarity (mm)
    names = list(layout["plane_names"])
    if len(names) >= 2:
        nt = planes[names[0]][0]; nw = planes[names[1]][0]
        parts.append(np.array([struct_w * 50.0 * float(nt @ nw)]))   # perpendicular
    return np.concatenate(parts)


def _pack(tag_pose, cam_pose, planes, tag_ids, n_frames, plane_names):
    """Pack ALL tag poses + cameras 1..n-1 (camera 0 is the fixed gauge — fixing a
    camera, not a tag, so no tag is pinned to a noisy estimate) + planes."""
    p = []
    for ti in tag_ids:
        R, t = tag_pose[ti]
        p.extend(_inv_rodrigues(R)); p.extend(t)
    for fi in range(1, n_frames):
        R, t = cam_pose[fi]
        p.extend(_inv_rodrigues(R)); p.extend(t)
    for name in plane_names:
        n, d = planes[name]
        th, ph = _angles_from_normal(n)
        p.extend([th, ph, d])
    layout = {"tag_ids": list(tag_ids), "n_frames": n_frames,
              "plane_names": list(plane_names), "cam0_pose": cam_pose[0]}
    return np.array(p, dtype=float), layout


def _unpack(params, layout):
    """Array-form unpack (mirrors ``_pack``): tag_R/tag_t indexed by ``tag_ids``
    position (the gauge ref filled fixed from layout), cam_R/cam_t by frame, planes
    by name. Array form so ``_residuals`` can vectorise."""
    tag_ids = layout["tag_ids"]
    n_frames = layout["n_frames"]; plane_names = layout["plane_names"]
    nt = len(tag_ids)
    tag_R = np.empty((nt, 3, 3)); tag_t = np.empty((nt, 3))
    i = 0
    for pos in range(nt):
        tag_R[pos] = _rodrigues(params[i:i + 3]); tag_t[pos] = params[i + 3:i + 6]; i += 6
    cam_R = np.empty((n_frames, 3, 3)); cam_t = np.empty((n_frames, 3))
    cam_R[0], cam_t[0] = layout["cam0_pose"]                  # fixed gauge
    for fi in range(1, n_frames):
        cam_R[fi] = _rodrigues(params[i:i + 3]); cam_t[fi] = params[i + 3:i + 6]; i += 6
    planes = {}
    for name in plane_names:
        th, ph, d = params[i], params[i + 1], params[i + 2]; i += 3
        planes[name] = (_normal_from_angles(th, ph), d)
    return tag_R, tag_t, cam_R, cam_t, planes


def _pnp(obj_world, img, K):
    """Camera pose (R, t) world→cam from ≥4 world points + their pixels, via cv2
    (frames are already rectified, so zero distortion)."""
    import cv2
    ok, rvec, tvec = cv2.solvePnP(np.asarray(obj_world, float), np.asarray(img, float),
                                  np.asarray(K, float), np.zeros((4, 1)),
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    return _rodrigues(rvec.ravel()), tvec.ravel()


def _param_cols(layout):
    """Column offset of each tag / camera / plane block in the packed param vector
    (mirrors ``_pack``). Returns (tag_cols, cam_cols, plane_cols, n_params)."""
    i = 0
    tag_cols = {}
    for ti in layout["tag_ids"]:
        tag_cols[ti] = i; i += 6
    cam_cols = {}                                            # camera 0 is fixed (no cols)
    for fi in range(1, layout["n_frames"]):
        cam_cols[fi] = i; i += 6
    plane_cols = {}
    for name in layout["plane_names"]:
        plane_cols[name] = i; i += 3
    return tag_cols, cam_cols, plane_cols, i


def _jac_sparsity(layout, obs, plane_groups):
    """Sparsity pattern of the residual Jacobian (rows in the same order as
    ``_residuals``), so least_squares does sparse finite-differencing."""
    from scipy.sparse import lil_matrix
    tag_cols, cam_cols, plane_cols, n_params = _param_cols(layout)
    n_rows = (8 * len(obs)
              + sum(len(ids) for ids in plane_groups.values())
              + (1 if len(layout["plane_names"]) >= 2 else 0))
    S = lil_matrix((n_rows, n_params), dtype=float)
    r = 0
    for (fi, ti, _) in obs:
        cols = list(range(tag_cols[ti], tag_cols[ti] + 6))    # all tags optimized
        if fi in cam_cols:                                    # camera 0 fixed → no cols
            cols += list(range(cam_cols[fi], cam_cols[fi] + 6))
        for rr in range(8):
            for c in cols:
                S[r + rr, c] = 1.0
        r += 8
    for name, ids in plane_groups.items():
        pc = plane_cols[name]
        for ti in ids:
            cols = [pc, pc + 1, pc + 2,                        # th, ph, d
                    tag_cols[ti] + 3, tag_cols[ti] + 4, tag_cols[ti] + 5]  # tag translation
            for c in cols:
                S[r, c] = 1.0
            r += 1
    if len(layout["plane_names"]) >= 2:
        for name in layout["plane_names"]:
            pc = plane_cols[name]
            S[r, pc] = 1.0; S[r, pc + 1] = 1.0               # both normals' angles
        r += 1
    return S


def bundle_adjust(corner_frames, K, tag_size_m, rel_init, ref_tag, plane_groups,
                  *, struct_w: float = 5.0, max_frames: int = 80, verbose=lambda s: None):
    """Constrained BA of the world map. ``corner_frames``: list of {tag_id:(4,2)}.
    ``rel_init``: {tag_id: T_world_tag (4×4)} initial guess (the fused map). Returns
    ``(rel_refined, report)``."""
    from scipy.optimize import least_squares
    from Utils.gaze.apriltag_calib import invert_transform
    import cv2  # noqa: F401  (PnP for init + corner-order pick)

    K = np.asarray(K, float)
    tag_ids = sorted(rel_init)
    # subsample frames evenly for tractability
    fr = corner_frames
    if len(fr) > max_frames:
        sel = np.linspace(0, len(fr) - 1, max_frames).astype(int)
        fr = [fr[i] for i in sel]

    # auto-pick corner order: the order with lowest reprojection on the fused map
    def reproj_rms(obj):
        errs = []
        for f in fr:
            ids = [t for t in f if t in rel_init]
            if len(ids) < 1:
                continue
            ow = np.vstack([(rel_init[t][:3, :3] @ obj.T).T + rel_init[t][:3, 3] for t in ids])
            ip = np.vstack([np.asarray(f[t], float) for t in ids])
            cp = _pnp(ow, ip, K)
            if cp is None:
                continue
            Rc, tc = cp
            pc = (Rc @ ow.T).T + tc
            z = np.where(np.abs(pc[:, 2]) < 1e-6, 1e-6, pc[:, 2])
            uv = np.column_stack([K[0, 0] * pc[:, 0] / z + K[0, 2],
                                  K[1, 1] * pc[:, 1] / z + K[1, 2]])
            errs.append(np.linalg.norm(uv - ip, axis=1))
        return float(np.sqrt(np.mean(np.concatenate(errs) ** 2))) if errs else 1e9

    best = min(range(4), key=lambda o: reproj_rms(tag_object_corners(tag_size_m, o)))
    obj = tag_object_corners(tag_size_m, best)
    verbose(f"corner order {best} (init reproj rms {reproj_rms(obj):.1f}px)")

    # init: tag poses from the map, camera poses from per-frame PnP
    tag_pose = {t: (rel_init[t][:3, :3].copy(), rel_init[t][:3, 3].copy()) for t in tag_ids}
    cam_pose = {}
    obs = []
    kept_frames = []
    for f in fr:
        ids = [t for t in f if t in rel_init]
        if len(ids) < 2:
            continue
        ow = np.vstack([(rel_init[t][:3, :3] @ obj.T).T + rel_init[t][:3, 3] for t in ids])
        ip = np.vstack([np.asarray(f[t], float) for t in ids])
        cp = _pnp(ow, ip, K)
        if cp is None:
            continue
        fi = len(kept_frames)
        cam_pose[fi] = cp
        kept_frames.append(f)
        for t in ids:
            obs.append((fi, t, np.asarray(f[t], float)))
    n_frames = len(kept_frames)
    if n_frames < 3:
        raise ValueError(f"too few usable frames for BA ({n_frames})")

    # init planes from the group orientation-normals + centroids
    from Utils.gaze.apriltag_world import table_normal_from_rel
    planes = {}
    for name, ids in plane_groups.items():
        present = [t for t in ids if t in rel_init]
        if len(present) < 3:
            continue
        n = table_normal_from_rel({t: rel_init[t] for t in present}, present)
        c = np.mean([rel_init[t][:3, 3] for t in present], axis=0)
        planes[name] = (n, float(c @ n))
    plane_names = list(planes)
    groups_present = {k: [t for t in v if t in rel_init] for k, v in plane_groups.items()
                      if k in planes}

    p0, layout = _pack(tag_pose, cam_pose, planes, tag_ids, n_frames, plane_names)

    # Vectorised-residual inputs: per-obs tag/frame POSITIONS + corner pixels, and
    # each group's tag positions (order matches _jac_sparsity).
    tagpos = {ti: i for i, ti in enumerate(tag_ids)}
    obs_arr = (np.array([tagpos[ti] for (_, ti, _) in obs]),
               np.array([fi for (fi, _, _) in obs]),
               np.array([cp for (_, _, cp) in obs]))
    group_pos = {name: np.array([tagpos[t] for t in ids])
                 for name, ids in groups_present.items()}

    def fun(p):
        return _residuals(p, layout, obs_arr, K, obj, group_pos, struct_w)

    # Sparsity: each reprojection residual touches only its tag + its frame; each
    # coplanarity row only its tag's translation + that plane; perpendicular only
    # the two plane normals. Without this the dense numerical Jacobian (~220 params)
    # allows ~1 iteration per max_nfev and the BA never converges.
    S = _jac_sparsity(layout, obs, groups_present)
    rms0 = float(np.sqrt(np.mean(fun(p0)[:2 * 4 * len(obs)] ** 2)))
    sol = least_squares(fun, p0, method="trf", jac_sparsity=S, tr_solver="lsmr",
                        ftol=1e-4, xtol=1e-4, max_nfev=400)
    tag_R_r, tag_t_r, _, _, planes_r = _unpack(sol.x, layout)
    rms1 = float(np.sqrt(np.mean(fun(sol.x)[:2 * 4 * len(obs)] ** 2)))

    rel_ref = {}
    for pos, t in enumerate(tag_ids):
        T = np.eye(4); T[:3, :3] = tag_R_r[pos]; T[:3, 3] = tag_t_r[pos]
        rel_ref[t] = T
    # Re-express in the ORIGINAL ref-tag gauge (BA ran in camera-0's frame): apply the
    # rigid transform mapping the BA's ref-tag pose to its input pose, so the refined
    # map shares the input world frame (the downstream chain expects that gauge).
    if ref_tag in rel_ref and ref_tag in rel_init:
        G = rel_init[ref_tag] @ invert_transform(rel_ref[ref_tag])
        rel_ref = {t: G @ T for t, T in rel_ref.items()}

    # report: reprojection + post flatness/perpendicularity, measured GAUGE-FREE on
    # the final origins (best-fit plane per group), not the BA's internal plane
    # params (which live in the un-re-expressed gauge).
    from Utils.gaze.apriltag_world import fit_plane
    rep = {"reproj_rms_px_before": rms0, "reproj_rms_px_after": rms1,
           "n_frames": n_frames, "n_obs": len(obs), "corner_order": best, "planes": {}}
    grp_normal = {}
    for name, ids in groups_present.items():
        O = np.array([rel_ref[t][:3, 3] for t in ids])
        c, n = fit_plane(O)
        grp_normal[name] = n
        r = np.abs((O - c) @ n)
        rep["planes"][name] = {"flatness_max_mm": float(r.max()),
                               "flatness_rms_mm": float(np.sqrt(np.mean(r ** 2)))}
    if "table" in grp_normal and "wall" in grp_normal:
        dot = abs(float(grp_normal["table"] @ grp_normal["wall"]))
        rep["table_wall_angle_deg"] = float(np.degrees(np.arccos(np.clip(dot, -1, 1))))  # 90 = ⟂
    return rel_ref, rep


def _self_test() -> int:
    """Plant a two-plane layout, render noisy corner observations from many views,
    perturb the init map, and confirm BA recovers the geometry + the structure."""
    rng = np.random.default_rng(0)

    def rotx(d):
        a = np.radians(d); c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def roty(d):
        a = np.radians(d); c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def T(R, t):
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t; return M

    s = 0.04
    true = {}                                              # table z=0 (+Z up), wall x=600 (+Z=+X)
    for i, (x, y) in enumerate([(0, 0), (400, 0), (0, 300), (400, 300), (200, 150), (200, 0)]):
        true[i] = T(np.eye(3), [x, y, 0.0])
    for j, (y, z) in enumerate([(0, 100), (300, 100), (0, 300), (300, 300), (150, 200), (150, 400)]):
        true[6 + j] = T(roty(90), [600.0, y, z])
    K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
    obj = tag_object_corners(s, 0)

    frames = []
    for k in range(24):
        Rc = rotx(20 + 0.5 * k) @ roty(-30 + 3 * k)
        tc = np.array([rng.normal(0, 120), rng.normal(0, 120), 900 + rng.normal(0, 60)])
        f = {}
        for t, Tt in true.items():
            ow = (Tt[:3, :3] @ obj.T).T + Tt[:3, 3]
            pc = (Rc @ ow.T).T + tc
            if np.any(pc[:, 2] <= 1):
                continue
            uv = np.column_stack([K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2],
                                  K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]])
            f[t] = uv + rng.normal(0, 0.4, uv.shape)        # 0.4 px corner noise
        if len(f) >= 3:
            frames.append(f)
    rel_init = {}                                          # depth-noisy + orientation-noisy init
    for t, Tt in true.items():
        R = Tt[:3, :3] @ rotx(rng.normal(0, 5)) @ roty(rng.normal(0, 5))
        off = Tt[:3, :3] @ np.array([0, 0, rng.normal(0, 60)])
        rel_init[t] = T(R, Tt[:3, 3] + off)

    groups = {"table": [0, 1, 2, 3, 4, 5], "wall": [6, 7, 8, 9, 10, 11]}
    rel_ref, rep = bundle_adjust(frames, K, s, rel_init, ref_tag=0, plane_groups=groups,
                                 verbose=print)

    def shape_err(rel):                                     # Procrustes vs truth
        ids = sorted(rel); A = np.array([rel[i][:3, 3] for i in ids])
        B = np.array([true[i][:3, 3] for i in ids])
        ca, cb = A.mean(0), B.mean(0)
        U, _, Vt = np.linalg.svd((B - cb).T @ (A - ca)); R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1; R = U @ Vt
        return float(np.mean(np.linalg.norm(((R @ (A - ca).T).T + cb) - B, axis=1)))

    e0, e1 = shape_err(rel_init), shape_err(rel_ref)
    print(f"reproj rms {rep['reproj_rms_px_before']:.1f}->{rep['reproj_rms_px_after']:.2f} px | "
          f"shape err {e0:.1f}->{e1:.1f} mm | table-wall {rep['table_wall_angle_deg']:.1f} deg | "
          f"table flat rms {rep['planes']['table']['flatness_rms_mm']:.1f} mm")
    ok = (e1 < 15.0 and e1 < 0.5 * e0            # shape substantially improved vs init
          and rep["reproj_rms_px_after"] < 1.5    # corners satisfied
          and abs(rep["table_wall_angle_deg"] - 90.0) < 2.5   # squared
          and rep["planes"]["table"]["flatness_rms_mm"] < 5.0)  # flattened
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="WS-4 Tier-2 constrained BA of the 3-D world map")
    p.add_argument("npz", nargs="?", help="register-world-3d npz with corner_obs")
    p.add_argument("--table-tag-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4, 12])
    p.add_argument("--wall-tag-ids", type=int, nargs="+", default=[6, 7, 8, 9, 10, 11])
    p.add_argument("--ref-tag", type=int, default=None, help="gauge tag (default: map ref)")
    p.add_argument("--struct-weight", type=float, default=5.0)
    p.add_argument("--max-frames", type=int, default=80)
    p.add_argument("--k-npz", help="borrow camera intrinsics K from another npz (e.g. a "
                   "sweep) when the map npz predates K-in-map; the Neon scene cam is "
                   "device-fixed so any same-device capture's K is valid")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)
    if args.self_test:
        return _self_test()
    if not args.npz:
        print("need an npz (or --self-test)"); return 2

    from Utils.gaze.apriltag_world import world_map_from_arrays, world_map_to_arrays
    z = np.load(args.npz, allow_pickle=True)
    if "corner_obs" not in z.files:
        print(f"{args.npz} has no corner_obs — re-register (corner logging added 2026-06-30)")
        return 2
    wm = world_map_from_arrays(z["world_map_ref"], z["world_map_ids"], z["world_map_rels"],
                               z["world_map_plane_point"], z["world_map_plane_normal"])
    rel_init = {int(i): np.asarray(wm["rel"][i], float) for i in wm["rel"]}
    corner_frames = [{int(t): np.asarray(c, float) for t, c in f.items()}
                     for f in z["corner_obs"]]
    meta = z["meta"].item() if "meta" in z.files else {}
    tag_size = float(meta.get("tag_size_m", 0.08))
    ref = args.ref_tag if args.ref_tag is not None else int(wm["ref_id"])
    groups = {"table": args.table_tag_ids, "wall": args.wall_tag_ids}
    if "K" in z.files:
        K = np.asarray(z["K"], float)
    elif args.k_npz:
        kz = np.load(args.k_npz, allow_pickle=True)
        if "K" not in kz.files:
            print(f"{args.k_npz} has no K either"); return 2
        K = np.asarray(kz["K"], float)
        print(f"borrowed K from {args.k_npz} (map npz predates K-in-map)")
    else:
        print("map npz has no K (registered before K-in-map). Re-run with "
              "--k-npz <a sweep npz from the SAME Neon> to borrow intrinsics."); return 2
    rel_ref, rep = bundle_adjust(corner_frames, K, tag_size, rel_init,
                                 ref_tag=ref, plane_groups=groups, struct_w=args.struct_weight,
                                 max_frames=args.max_frames, verbose=print)
    print(f"BA: reproj rms {rep['reproj_rms_px_before']:.1f}->{rep['reproj_rms_px_after']:.2f} px, "
          f"table-wall {rep.get('table_wall_angle_deg', float('nan')):.1f} deg")
    for name, pl in rep["planes"].items():
        print(f"  {name} flatness max={pl['flatness_max_mm']:.1f} rms={pl['flatness_rms_mm']:.1f} mm")
    out = Path(args.npz).with_name(Path(args.npz).stem + "_ba.npz")
    wm_ref = dict(wm); wm_ref["rel"] = rel_ref
    ref_id, ids, rels, pp, pn = world_map_to_arrays(wm_ref)
    np.savez_compressed(out, world_map_ref=np.array(ref_id), world_map_ids=ids,
                        world_map_rels=rels, world_map_plane_point=pp, world_map_plane_normal=pn,
                        meta=np.array({"stage": "world_map_ba", "tag_size_m": tag_size,
                                       "ba_report": rep}, dtype=object))
    print(f"refined world map → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
