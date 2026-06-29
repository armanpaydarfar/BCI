"""
Analyze_handeye_axzb.py — WS-2a robot-world / hand-eye solve (OFFLINE, analysis-only).

Goal: bring the robot's FK end-effector ORIENTATION into the AprilTag world frame.

Input is an AprilTag calibration sweep npz produced by
``tools/apriltag_calibrate.py`` stage_sweep (save block apriltag_calibrate.py:873-889),
with index-aligned arrays:
  X        (N,3)   robot EE position, mm, BASE frame (telemetry pos_mm)
  EEQUAT   (N,4)   robot EE orientation quaternion [x,y,z,w] scalar-last, BASE frame
  T_cam_world (N,4,4)  camera->world tag pose, saved in MM; load_sweep converts to m
  T_cam_eetag (N,4,4)  camera->eetag tag pose, saved in MM; load_sweep converts to m
  K        (3,3)   scene-cam intrinsics
  green    (N,)    bool, sample lands in a sufficient coverage cell
  meta     (obj)   dict with t_eetag_ee_mm, tag sizes, side

The math (load-bearing convention — see module docstring of solve_handeye):
  per sample i,
    T_base_ee(i)     = base->ee from FK telemetry (X mm->m, EEQUAT [x,y,z,w])
    T_world_eetag(i) = inv(T_cam_world[i]) @ T_cam_eetag[i]   (world->eetag, vision)
  unknowns Z=T_base_world, X_he=T_ee_eetag satisfy
    T_base_ee(i) @ T_ee_eetag = T_base_world @ T_world_eetag(i)      (A X = Z B)
  solved with cv2.calibrateRobotWorldHandEye (A=base2gripper, B=world2cam,
  X=gripper2cam, Z=base2world). cv2 names its args in the "maps X->Y" direction,
  the inverse of our "pose of" matrices, so solve_handeye feeds inv(T_base_ee) and
  inv(T_world_eetag) and inverts the returned base2world/gripper2cam back to our
  T_base_world / T_ee_eetag (verified empirically; see solve_handeye docstring).

The deliverable orientation transform is R_world_base = inv(R_base2world); with it
the FK EE orientation in WORLD is R_world_ee(i) = R_world_base @ R(T_base_ee(i)).

Analysis-only: reads an npz, prints/saves a report. Does not touch runtime code,
open device streams, or publish markers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from Utils.gaze.apriltag_calib import invert_transform, make_transform  # noqa: E402

# Heuristic conditioning floor (degrees). Hand-eye / robot-world is ill-conditioned
# without rotational variety in the gripper motions: a flat planar sweep that keeps
# the wrist at a near-constant orientation gives the solver no leverage on the
# rotation unknowns. 15 deg is a hand-picked sanity floor, NOT a derived bound.
ROT_VARIETY_WARN_DEG = 15.0

# cv2 method ids by name; populated lazily so the module imports without cv2.
_METHOD_NAMES = ("CALIB_ROBOT_WORLD_HAND_EYE_SHAH", "CALIB_ROBOT_WORLD_HAND_EYE_LI")


def t_base_ee_from_telemetry(x_mm: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """base->ee FK transform from telemetry position (mm) and quaternion [x,y,z,w].

    Position is converted mm->meters to match the vision transforms, which
    load_sweep has already converted to metres (the sweep saves T_cam_* in mm)."""
    R = Rotation.from_quat(np.asarray(quat_xyzw, dtype=float)).as_matrix()
    t_m = np.asarray(x_mm, dtype=float).ravel() / 1000.0
    return make_transform(R, t_m)


def vision_transforms_to_metres(T):
    """Convert tag-pose transform translations from MM to metres (rotation
    untouched). The sweep saves T_cam_* with mm translations (apriltag_detect.py
    rescales m->mm to match robot telemetry / the X column), but this script works
    in metres (FK X is converted to m in t_base_ee_from_telemetry). Mixing the two
    makes the AX=ZB translation block 1000x wrong (the rotation deliverable
    R_world_base is scale-independent, which masked the bug). Pure; accepts a single
    (4,4) or a batch (N,4,4)."""
    T = np.asarray(T, dtype=float).copy()
    T[..., :3, 3] /= 1000.0
    return T


def t_world_eetag_from_vision(T_cam_world: np.ndarray, T_cam_eetag: np.ndarray) -> np.ndarray:
    """world->eetag = inv(cam->world) @ cam->eetag. Inputs are metres (caller passes
    transforms already through vision_transforms_to_metres)."""
    return invert_transform(T_cam_world) @ T_cam_eetag


def rotation_geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle (deg) between two rotation matrices."""
    return float(Rotation.from_matrix(Ra.T @ Rb).magnitude() * 180.0 / np.pi)


def rotation_variety_report(R_list: np.ndarray) -> dict:
    """Quantify rotational variety of a set of rotations.

    Returns the mean and max pairwise geodesic angle (deg) plus the angular spread
    of the rotation axes. Hand-eye conditioning hinges on this: with little variety
    the AX=ZB rotation block is near-degenerate. ``warn`` is True when variety sits
    below the heuristic ROT_VARIETY_WARN_DEG floor."""
    rots = Rotation.from_matrix(np.asarray(R_list, dtype=float))
    n = len(rots)
    mats = rots.as_matrix()
    pair_angles = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_angles.append(rotation_geodesic_deg(mats[i], mats[j]))
    pair_angles = np.asarray(pair_angles) if pair_angles else np.array([0.0])

    # Axis spread: mean angle of each sample's rotation vector from the mean axis.
    rotvecs = rots.as_rotvec()
    angles = np.linalg.norm(rotvecs, axis=1)
    moving = angles > 1e-9
    if moving.sum() >= 2:
        axes = rotvecs[moving] / angles[moving, None]
        mean_axis = axes.mean(axis=0)
        nrm = np.linalg.norm(mean_axis)
        if nrm > 1e-12:
            mean_axis = mean_axis / nrm
            cos = np.clip(axes @ mean_axis, -1.0, 1.0)
            axis_spread = float(np.degrees(np.arccos(cos)).mean())
        else:
            axis_spread = float("nan")
    else:
        axis_spread = float("nan")

    mean_pair = float(pair_angles.mean())
    return {
        "n": int(n),
        "mean_pairwise_deg": mean_pair,
        "max_pairwise_deg": float(pair_angles.max()),
        "axis_spread_deg": axis_spread,
        "warn": mean_pair < ROT_VARIETY_WARN_DEG,
        "threshold_deg": ROT_VARIETY_WARN_DEG,
    }


def solve_handeye(T_base_ee: np.ndarray, T_world_eetag: np.ndarray, method) -> dict:
    """Solve A X = Z B for Z=T_base_world and X=T_ee_eetag via cv2.

    Our transforms use the "pose of" convention: T_base_ee is the pose of the EE in
    the base frame (maps EE-coords -> base-coords) and T_world_eetag is the pose of
    the eetag in the world frame. cv2.calibrateRobotWorldHandEye names its arguments
    in the OPPOSITE "maps X -> Y" direction: base2gripper maps base->gripper and
    world2cam maps world->cam. Feeding our pose-of matrices directly recovers a
    transform ~107 deg wrong (verified empirically), so we invert on the way in:
        base2gripper := inv(T_base_ee)        (A, maps base->ee)
        world2cam    := inv(T_world_eetag)    (B, maps world->eetag; our "cam" is eetag)
    cv2 returns base2world (maps base->world) and gripper2cam (maps ee->eetag), the
    inverses of our deliverables, so we invert back:
        T_base_world = inv(base2world),  T_ee_eetag = inv(gripper2cam).

    Returns a dict with T_base_world, R_world_base, T_ee_eetag and the residuals.
    """
    import cv2

    A = np.asarray(T_base_ee, dtype=float)
    B = np.asarray(T_world_eetag, dtype=float)
    if A.shape[0] != B.shape[0] or A.shape[0] < 3:
        raise ValueError(f"need >=3 paired poses; got A={A.shape}, B={B.shape}")

    A_in = [invert_transform(a) for a in A]   # base2gripper (maps base->ee)
    B_in = [invert_transform(b) for b in B]   # world2cam (maps world->eetag)
    R_base2gripper = [m[:3, :3] for m in A_in]
    t_base2gripper = [m[:3, 3] for m in A_in]
    R_world2cam = [m[:3, :3] for m in B_in]
    t_world2cam = [m[:3, 3] for m in B_in]

    R_b2w, t_b2w, R_g2c, t_g2c = cv2.calibrateRobotWorldHandEye(
        R_world2cam, t_world2cam, R_base2gripper, t_base2gripper, method=method
    )
    T_base_world = invert_transform(make_transform(R_b2w, t_b2w))   # Z
    T_ee_eetag = invert_transform(make_transform(R_g2c, t_g2c))     # X
    R_world_base = T_base_world[:3, :3].T

    resid = handeye_residuals(A, B, T_base_world, T_ee_eetag)
    return {
        "T_base_world": T_base_world,
        "R_world_base": R_world_base,
        "T_ee_eetag": T_ee_eetag,
        **resid,
    }


def handeye_residuals(T_base_ee: np.ndarray, T_world_eetag: np.ndarray,
                      T_base_world: np.ndarray, T_ee_eetag: np.ndarray) -> dict:
    """Per-sample residual of AX vs ZB.

    AX = T_base_ee @ T_ee_eetag (eetag pose in base, via FK + hand-eye)
    ZB = T_base_world @ T_world_eetag (eetag pose in base, via base-world + vision)
    Reports rotation error (deg) and translation error (mm), median + 95th pct."""
    rot_err, trans_err = [], []
    for A, B in zip(T_base_ee, T_world_eetag):
        AX = A @ T_ee_eetag
        ZB = T_base_world @ B
        rot_err.append(rotation_geodesic_deg(AX[:3, :3], ZB[:3, :3]))
        trans_err.append(float(np.linalg.norm(AX[:3, 3] - ZB[:3, 3]) * 1000.0))
    rot_err = np.asarray(rot_err)
    trans_err = np.asarray(trans_err)
    return {
        "rot_err_deg": rot_err,
        "trans_err_mm": trans_err,
        "rot_err_median_deg": float(np.median(rot_err)),
        "rot_err_p95_deg": float(np.percentile(rot_err, 95)),
        "trans_err_median_mm": float(np.median(trans_err)),
        "trans_err_p95_mm": float(np.percentile(trans_err, 95)),
    }


def fk_vs_vision_orientation(T_base_ee: np.ndarray, T_world_eetag: np.ndarray,
                             R_world_base: np.ndarray) -> np.ndarray:
    """Per-sample angular difference (deg) between the FK-derived EE orientation in
    world and the raw single-tag vision orientation.

    R_world_ee(i)    = R_world_base @ R(T_base_ee(i))           (FK, hand-eye lifted)
    R_world_eetag(i) = R(T_world_eetag(i))                      (vision, single tag)

    The eetag and the EE differ by the fixed X=T_ee_eetag rotation, which is constant,
    so the time-VARYING part of this difference is exactly the single-tag orientation
    noise that WS-2a exists to replace with FK. We report the geodesic angle after
    removing the per-sample-constant component is NOT done here; instead we report the
    raw geodesic between R_world_ee and R_world_eetag, whose spread (std) is the noise
    of interest while its median reflects the fixed eetag->ee offset."""
    diffs = []
    for A, B in zip(T_base_ee, T_world_eetag):
        R_world_ee = R_world_base @ A[:3, :3]
        R_world_eetag = B[:3, :3]
        diffs.append(rotation_geodesic_deg(R_world_ee, R_world_eetag))
    return np.asarray(diffs)


def load_sweep(npz_path: str):
    """Load the sweep npz and drop rows with NaN telemetry (robot absent)."""
    d = np.load(npz_path, allow_pickle=True)
    if "EEQUAT" not in d.files:
        raise KeyError(
            f"{npz_path} has no EEQUAT array — this sweep was captured without a "
            "robot (with_robot=False), so it carries no EE orientation telemetry. "
            "WS-2a hand-eye needs a robot-present sweep. Available arrays: "
            f"{sorted(d.files)}")
    X = np.asarray(d["X"], dtype=float)
    EEQUAT = np.asarray(d["EEQUAT"], dtype=float)
    T_cam_world = vision_transforms_to_metres(d["T_cam_world"])
    T_cam_eetag = vision_transforms_to_metres(d["T_cam_eetag"])
    green = np.asarray(d["green"]).astype(bool) if "green" in d else None
    meta = d["meta"].item() if "meta" in d else {}

    finite = (np.isfinite(X).all(axis=1)
              & np.isfinite(EEQUAT).all(axis=1)
              & np.isfinite(T_cam_world).reshape(len(X), -1).all(axis=1)
              & np.isfinite(T_cam_eetag).reshape(len(X), -1).all(axis=1))
    n_drop = int((~finite).sum())
    return {
        "X": X[finite], "EEQUAT": EEQUAT[finite],
        "T_cam_world": T_cam_world[finite], "T_cam_eetag": T_cam_eetag[finite],
        "green": green[finite] if green is not None else None,
        "meta": meta, "n_total": len(X), "n_dropped_nan": n_drop,
    }


def build_pose_pairs(data: dict):
    """Assemble (T_base_ee, T_world_eetag) stacks from a loaded sweep dict."""
    n = len(data["X"])
    T_base_ee = np.stack([
        t_base_ee_from_telemetry(data["X"][i], data["EEQUAT"][i]) for i in range(n)
    ])
    T_world_eetag = np.stack([
        t_world_eetag_from_vision(data["T_cam_world"][i], data["T_cam_eetag"][i])
        for i in range(n)
    ])
    return T_base_ee, T_world_eetag


def _fmt_T(T: np.ndarray) -> str:
    rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
    return (f"  t (mm) = {np.round(T[:3, 3] * 1000.0, 2)}\n"
            f"  rpy (deg) = {np.round(rpy, 3)}\n"
            f"  R =\n{np.array2string(T[:3, :3], precision=5, suppress_small=True)}")


def run(npz_path: str) -> int:
    data = load_sweep(npz_path)
    print(f"[load] {npz_path}")
    print(f"[load] {data['n_total']} rows, dropped {data['n_dropped_nan']} NaN-telemetry rows")

    # Prefer green/sufficient samples for the solve when the mask is present.
    T_base_ee, T_world_eetag = build_pose_pairs(data)
    if data["green"] is not None and data["green"].sum() >= 4:
        mask = data["green"]
        print(f"[load] using {int(mask.sum())} green/sufficient samples for solve")
        T_base_ee, T_world_eetag = T_base_ee[mask], T_world_eetag[mask]
    else:
        print(f"[load] using all {len(T_base_ee)} samples (green mask absent/sparse)")

    if len(T_base_ee) < 3:
        print("[error] fewer than 3 usable paired poses; cannot solve.")
        return 2

    # --- conditioning / orientation variety (the documented open risk) ---
    R_fk = T_base_ee[:, :3, :3]
    variety = rotation_variety_report(R_fk)
    print("\n=== FK rotation variety (conditioning) ===")
    print(f"  n                 = {variety['n']}")
    print(f"  mean pairwise     = {variety['mean_pairwise_deg']:.2f} deg")
    print(f"  max  pairwise     = {variety['max_pairwise_deg']:.2f} deg")
    print(f"  axis spread       = {variety['axis_spread_deg']:.2f} deg")
    print(f"  heuristic floor   = {variety['threshold_deg']:.1f} deg (hand-picked, not derived)")
    if variety["warn"]:
        print("  *** WARNING: rotational variety BELOW the heuristic floor. ***")
        print("  *** Robot-world / hand-eye is ILL-CONDITIONED without wrist rotation. ***")
        print("  *** A flat planar sweep likely did not rotate the EE enough; the     ***")
        print("  *** recovered T_base_world orientation may be unreliable.             ***")
    else:
        print("  OK: rotational variety above the heuristic floor.")

    # --- solve with each available method ---
    import cv2
    print("\n=== Robot-world / hand-eye solve (AX = ZB) ===")
    results = {}
    for name in _METHOD_NAMES:
        if not hasattr(cv2, name):
            print(f"\n[{name}] not available in this cv2 build — skipped.")
            continue
        method = getattr(cv2, name)
        try:
            res = solve_handeye(T_base_ee, T_world_eetag, method)
        except cv2.error as e:  # solver-level failure: report, keep going
            print(f"\n[{name}] cv2 solver failed: {e}")
            continue
        results[name] = res
        print(f"\n[{name}]")
        print(" T_base_world (Z):")
        print(_fmt_T(res["T_base_world"]))
        print(" R_world_base = inv(R_base2world):")
        print(np.array2string(res["R_world_base"], precision=5, suppress_small=True))
        print(" T_ee_eetag (X):")
        print(_fmt_T(res["T_ee_eetag"]))
        print(" residual AX vs ZB:")
        print(f"   rotation    median={res['rot_err_median_deg']:.3f} deg  "
              f"p95={res['rot_err_p95_deg']:.3f} deg")
        print(f"   translation median={res['trans_err_median_mm']:.3f} mm  "
              f"p95={res['trans_err_p95_mm']:.3f} mm")

    if not results:
        print("\n[error] no hand-eye method succeeded.")
        return 3

    # --- FK vs vision orientation cross-check (the WS-2a motivation) ---
    best = next(iter(results.values()))
    diffs = fk_vs_vision_orientation(T_base_ee, T_world_eetag, best["R_world_base"])
    print("\n=== FK-derived vs raw single-tag vision EE orientation ===")
    print("  geodesic(R_world_ee_FK, R_world_eetag_vision) per sample:")
    print(f"   median = {np.median(diffs):.3f} deg   (~ fixed eetag->ee offset)")
    print(f"   std    = {np.std(diffs):.3f} deg   (~ single-tag orientation noise)")
    print(f"   p95    = {np.percentile(diffs, 95):.3f} deg")
    print(f"   range  = [{diffs.min():.3f}, {diffs.max():.3f}] deg")
    print("  -> a large std/range is the single-tag orientation jitter that WS-2a")
    print("     replaces by lifting FK orientation into the world frame.")

    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, help="path to apriltag_sweep_*.npz")
    args = ap.parse_args(argv)
    return run(args.npz)


if __name__ == "__main__":
    raise SystemExit(main())
