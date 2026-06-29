"""Analyze_eehand_offset_ls.py — WS-2b: least-squares EE-tag -> held-object offset.

OFFLINE ANALYSIS ONLY. Does not open device streams, send UDP, or publish LSL.
Reads a stored AprilTag sweep npz (produced by tools/apriltag_calibrate.py
stage_sweep, save block apriltag_calibrate.py:830-889) plus its scene-frame
sidecar, segments the held object per frame, backprojects its centroid into the
world frame, and solves the fixed tag-frame offset

    p_centroid_world(i) = t_world_eetag(i) + R(i) @ x

for x = t_eetag_ee (the vector from the EE-tag origin to the held-object
centroid, expressed in the EE-tag frame). This is the data-driven alternative to
the hand-measured config offset APRILTAG_T_EETAG_EE_MM = [150, -200, 0] mm
(config.py:211).

Two rotation sources select how R(i) is built (--r-source):
  vision (default) : R(i) = rotation part of T_world_eetag(i) = inv(T_cam_world)
                     @ T_cam_eetag, fully self-contained from the sweep npz.
  fk               : R(i) = R_world_base @ R(quat(i)), where R_world_base comes
                     from the WS-2a hand-eye result (--handeye-npz) and the EE
                     orientation is the FK quaternion EEQUAT (the robot position
                     X is not used — only orientation is taken from FK).

The segmentation/backprojection stage needs the perception stack (FastSAM +
Depth Pro) and the machine-local PERCEPTION_MODELS_DIR; on a host without it the
script prints a clear "perception unavailable" message, skips segmentation, and
still exercises the numerical LS core via the synthetic self-test (--self-test).

Usage:
    python Analyze_eehand_offset_ls.py path/to/apriltag_sweep_*.npz
    python Analyze_eehand_offset_ls.py SWEEP.npz --r-source fk --handeye-npz HE.npz
    python Analyze_eehand_offset_ls.py --self-test           # no npz needed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Current hand-measured config offset (config.py:211), in mm. Imported lazily in
# main() so the pure LS core and the self-test do not depend on a config_local.
CONFIG_OFFSET_MM_FALLBACK = np.array([150.0, -200.0, 0.0], dtype=np.float64)


# ── pure numerical core (unit-tested) ──────────────────────────────────────────

def solve_offset_ls(R_list, t_world_eetag_list, p_centroid_world_list):
    """Least-squares solve for the fixed EE-tag -> object-centroid offset.

    Solves, in one stacked linear system over all samples i,

        R(i) @ x = p_centroid_world(i) - t_world_eetag(i)

    for x (3,), the offset in the EE-tag frame. All spatial inputs are in metres;
    the recovered offset is returned in millimetres to compare against the config
    offset (config.py:211).

    Parameters
    ----------
    R_list                 : (N, 3, 3) rotations world<-eetag per sample.
    t_world_eetag_list     : (N, 3) EE-tag origin in world frame, metres.
    p_centroid_world_list  : (N, 3) object-centroid in world frame, metres.

    Returns
    -------
    x_mm : (3,) recovered offset in mm.
    stats : dict with residual diagnostics:
        per_axis_rms_mm  (3,)  RMS of the LS residual per world axis,
        overall_rms_mm   float RMS over all 3N residual components,
        per_sample_err_mm (N,) ||R(i)x + t - p|| per sample,
        median_err_mm, p95_err_mm  float summaries of per_sample_err_mm,
        n_samples        int.
    """
    R = np.asarray(R_list, dtype=np.float64)
    t = np.asarray(t_world_eetag_list, dtype=np.float64)
    p = np.asarray(p_centroid_world_list, dtype=np.float64)
    if R.ndim != 3 or R.shape[1:] != (3, 3):
        raise ValueError(f"R_list must be (N,3,3); got {R.shape}")
    n = R.shape[0]
    if t.shape != (n, 3) or p.shape != (n, 3):
        raise ValueError(
            f"t/p must be (N,3) matching R; got t={t.shape} p={p.shape} N={n}"
        )
    if n < 2:
        # 3 equations/sample, but a single rigid rotation is rank-deficient for x
        # only when R is the same across samples; require >=2 distinct poses.
        raise ValueError(f"need >=2 samples for a determined solve; got {n}")

    A = R.reshape(3 * n, 3)                          # (3N, 3)
    b = (p - t).reshape(3 * n)                        # (3N,)
    x_m, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    resid = (A @ x_m - b).reshape(n, 3)              # metres
    per_sample_err_m = np.linalg.norm(resid, axis=1)
    per_axis_rms_mm = np.sqrt(np.mean(resid ** 2, axis=0)) * 1000.0
    overall_rms_mm = float(np.sqrt(np.mean(resid ** 2)) * 1000.0)
    per_sample_err_mm = per_sample_err_m * 1000.0

    stats = {
        "per_axis_rms_mm": per_axis_rms_mm,
        "overall_rms_mm": overall_rms_mm,
        "per_sample_err_mm": per_sample_err_mm,
        "median_err_mm": float(np.median(per_sample_err_mm)),
        "p95_err_mm": float(np.percentile(per_sample_err_mm, 95)),
        "n_samples": n,
    }
    return x_m * 1000.0, stats


# ── geometry helpers ────────────────────────────────────────────────────────────

def invert_se3(T):
    """Inverse of a 4x4 rigid transform without a full matrix inverse."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    p = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


def quat_xyzw_to_R(q):
    """Rotation matrix from a scalar-last [x,y,z,w] quaternion (EEQUAT order)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(np.asarray(q, dtype=np.float64)).as_matrix()


def vision_transforms_to_metres(T):
    """Convert tag-pose transform translations from MM to metres (rotation
    untouched). The sweep saves T_cam_* with mm translations (apriltag_detect.py
    rescales m->mm to match robot telemetry), but this script is metres-internal
    (Depth Pro depth is in m; the LS core declares metres). Mixing the two scales
    a centroid/offset 1000x — so every loaded vision transform passes through here.
    Pure; accepts a single (4,4) or a batch (N,4,4)."""
    T = np.asarray(T, dtype=np.float64).copy()
    T[..., :3, 3] /= 1000.0
    return T


def world_eetag_from_sweep(T_cam_world_i, T_cam_eetag_i):
    """T_world_eetag(i) = inv(T_cam_world) @ T_cam_eetag. Inputs are metres (the
    caller passes transforms already through vision_transforms_to_metres)."""
    return invert_se3(T_cam_world_i) @ np.asarray(T_cam_eetag_i, dtype=np.float64)


# ── segmentation / backprojection (rig-only, perception-gated) ──────────────────

def _try_load_perception():
    """Build (ObjectDetector, DepthEstimator) or return (None, None, reason).

    Mirrors the model-path resolution the perception services use: weights live
    under the machine-local PERCEPTION_MODELS_DIR (config.py:288-291). Any import
    or model-load failure is caught and surfaced as a reason string; the caller
    falls back to the synthetic self-test rather than crashing.
    """
    try:
        import config  # noqa: F401  (pulls config_local for PERCEPTION_MODELS_DIR)
        from perception.object_detector import ObjectDetector
        from perception.depth_estimator import DepthEstimator
    except Exception as exc:  # import-time failure: deps/weights unavailable
        return None, None, f"import failed: {exc!r}"

    models_dir = getattr(config, "PERCEPTION_MODELS_DIR", "") or ""
    if not models_dir or not Path(models_dir).is_dir():
        return None, None, (
            f"PERCEPTION_MODELS_DIR not usable ({models_dir!r}); set it in "
            "config_local.py on the rig"
        )
    try:
        seg_weights = str(Path(models_dir) / "FastSAM-s.pt")
        depth_weights = str(Path(models_dir) / "depth_pro.pt")
        conf = float(getattr(config, "SEG_CONF_THRESHOLD", 0.4))
        detector = ObjectDetector(model_size=seg_weights, conf_threshold=conf)
        depther = DepthEstimator(checkpoint=depth_weights)
    except Exception as exc:  # weights missing or no GPU on this host
        return None, None, f"model load failed: {exc!r}"
    return detector, depther, None


def _project_point(p_cam, K):
    """Pinhole projection of a camera-frame point (metres) to a pixel (u,v)."""
    fx, fy = K[0, 0], K[1, 1]
    cx0, cy0 = K[0, 2], K[1, 2]
    X, Y, Z = p_cam
    if Z <= 0:
        return None
    return (fx * X / Z + cx0, fy * Y / Z + cy0)


def select_inhand_detection(detections, eetag_pixel, frame_shape):
    """Pick the detection most likely to be the held object.

    Preference order: a detection whose mask contains the projected EE-tag pixel,
    else the detection whose box centre is nearest that pixel, else the
    highest-confidence detection. Falls back to largest mask area when the EE-tag
    pixel is unavailable (off-frame).
    """
    import cv2
    if not detections:
        return None

    if eetag_pixel is not None:
        u, v = eetag_pixel
        h, w = frame_shape[:2]
        if 0 <= u < w and 0 <= v < h:
            containing = []
            for det in detections:
                if det.mask_polygon is not None and len(det.mask_polygon) >= 3:
                    inside = cv2.pointPolygonTest(
                        det.mask_polygon.reshape(-1, 1, 2).astype(np.int32),
                        (float(u), float(v)), False,
                    )
                    if inside >= 0:
                        containing.append(det)
            if containing:
                return max(containing, key=lambda d: d.confidence)
        # nearest box centre to the EE-tag pixel
        return min(
            detections,
            key=lambda d: (d.box_center[0] - u) ** 2 + (d.box_center[1] - v) ** 2,
        )

    # No EE-tag pixel: largest mask by polygon area, else highest confidence.
    def _area(det):
        if det.mask_polygon is not None and len(det.mask_polygon) >= 3:
            return float(cv2.contourArea(det.mask_polygon.reshape(-1, 1, 2)))
        x1, y1, x2, y2 = det.box_xyxy
        return float(abs(x2 - x1) * abs(y2 - y1))

    return max(detections, key=_area)


def centroid_world_from_frame(frame_bgr, K, T_cam_world_i, T_cam_eetag_i,
                              detector, depther):
    """Segment the held object and backproject its centroid into the world frame.

    Returns p_centroid_world (3,) in metres, or None if no usable detection.
    Backprojection mirrors compute_3d_waypoints (object_detector.py:91-125):
    X=(cx-cx0)*Z/fx, Y=(cy-cy0)*Z/fy, Z=median in-mask depth, camera frame metres.
    """
    import cv2
    detections = detector.detect(frame_bgr)
    if not detections:
        return None

    # Project the EE-tag origin so we can prefer the object held at the tag.
    t_cam_eetag = np.asarray(T_cam_eetag_i, dtype=np.float64)[:3, 3]
    eetag_pixel = _project_point(t_cam_eetag, np.asarray(K, dtype=np.float64))
    det = select_inhand_detection(detections, eetag_pixel, frame_bgr.shape)
    if det is None:
        return None

    depth_map, _ = depther.estimate(frame_bgr)
    h, w = depth_map.shape[:2]
    if det.mask_polygon is not None and len(det.mask_polygon) >= 3:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [det.mask_polygon.reshape(-1, 1, 2).astype(np.int32)], 1)
        depths = depth_map[mask == 1]
        M = cv2.moments(mask, binaryImage=True)
        if M["m00"] > 0:
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        else:
            cx, cy = det.box_center
    else:
        cx, cy = det.box_center
        ix = max(0, min(int(round(cx)), w - 1))
        iy = max(0, min(int(round(cy)), h - 1))
        depths = np.array([depth_map[iy, ix]])

    depths = depths[(depths > 0) & np.isfinite(depths) & (depths < 10.0)]
    if len(depths) == 0:
        return None
    Z = float(np.median(depths))
    fx, fy = K[0, 0], K[1, 1]
    cx0, cy0 = K[0, 2], K[1, 2]
    p_cam = np.array([(cx - cx0) * Z / fx, (cy - cy0) * Z / fy, Z, 1.0],
                     dtype=np.float64)
    p_world = invert_se3(T_cam_world_i) @ p_cam
    return p_world[:3]


# ── R(i) sources ────────────────────────────────────────────────────────────────

def build_rotations_vision(T_cam_world, T_cam_eetag, idx):
    """R(i), t_world_eetag(i) from the sweep transforms (self-contained)."""
    R_list, t_list = [], []
    for i in idx:
        T_we = world_eetag_from_sweep(T_cam_world[i], T_cam_eetag[i])
        R_list.append(T_we[:3, :3])
        t_list.append(T_we[:3, 3])
    return np.stack(R_list), np.stack(t_list)


def load_R_world_base(handeye_npz):
    """Read R_world_base (3x3) from the WS-2a hand-eye result npz.

    Accepts a stored 3x3 ('R_world_base'), a 4x4 ('T_world_base'), or a quaternion
    ('q_world_base', scalar-last). Raises with a clear message if none is present.
    """
    d = np.load(handeye_npz, allow_pickle=True)
    keys = set(d.files)
    if "R_world_base" in keys:
        return np.asarray(d["R_world_base"], dtype=np.float64).reshape(3, 3)
    if "T_world_base" in keys:
        return np.asarray(d["T_world_base"], dtype=np.float64)[:3, :3]
    if "q_world_base" in keys:
        return quat_xyzw_to_R(d["q_world_base"])
    raise KeyError(
        f"--handeye-npz {handeye_npz} has none of R_world_base / T_world_base / "
        f"q_world_base; found {sorted(keys)}"
    )


def build_rotations_fk(EEQUAT, T_cam_world, T_cam_eetag, idx, R_world_base):
    """R(i) = R_world_base @ R(quat(i)); t_world_eetag(i) from the sweep.

    The rotation comes purely from the FK orientation quaternion (EEQUAT) lifted
    into world frame by R_world_base — the robot EE position X is not needed here.
    The translation arm of the LS equation still comes from the vision-observed
    EE-tag origin in the world frame (inv(T_cam_world)@T_cam_eetag); only the
    rotation is swapped to the FK chain so the two R-sources share one geometry.
    """
    R_list, t_list = [], []
    for i in idx:
        R_base_ee = quat_xyzw_to_R(EEQUAT[i])
        R_list.append(R_world_base @ R_base_ee)
        T_we = world_eetag_from_sweep(T_cam_world[i], T_cam_eetag[i])
        t_list.append(T_we[:3, 3])
    return np.stack(R_list), np.stack(t_list)


# ── reporting ───────────────────────────────────────────────────────────────────

def report(x_mm, stats, config_offset_mm, meta_offset_mm, r_source):
    print("\n" + "=" * 70)
    print(f"WS-2b least-squares EE-tag -> object-centroid offset  [R-source: {r_source}]")
    print("=" * 70)
    print(f"samples used                : {stats['n_samples']}")
    print(f"recovered x_hat (mm)        : "
          f"[{x_mm[0]:8.2f}, {x_mm[1]:8.2f}, {x_mm[2]:8.2f}]")
    print(f"config offset (config.py:211): "
          f"[{config_offset_mm[0]:8.2f}, {config_offset_mm[1]:8.2f}, {config_offset_mm[2]:8.2f}]")
    delta = x_mm - config_offset_mm
    print(f"delta x_hat - config (mm)   : "
          f"[{delta[0]:8.2f}, {delta[1]:8.2f}, {delta[2]:8.2f}]   "
          f"|delta|={np.linalg.norm(delta):.2f}")
    if meta_offset_mm is not None:
        dm = x_mm - meta_offset_mm
        print(f"meta t_eetag_ee_mm (sweep)  : "
              f"[{meta_offset_mm[0]:8.2f}, {meta_offset_mm[1]:8.2f}, {meta_offset_mm[2]:8.2f}]   "
              f"|x_hat-meta|={np.linalg.norm(dm):.2f}")
    pa = stats["per_axis_rms_mm"]
    print("-" * 70)
    print(f"LS residual per-axis RMS(mm): [{pa[0]:6.2f}, {pa[1]:6.2f}, {pa[2]:6.2f}]")
    print(f"LS residual overall RMS (mm): {stats['overall_rms_mm']:.2f}")
    print(f"reproj err median / p95 (mm): "
          f"{stats['median_err_mm']:.2f} / {stats['p95_err_mm']:.2f}")
    print("-" * 70)
    norm_delta = np.linalg.norm(delta)
    if norm_delta < 15.0:
        verdict = ("CONFIRMS the hand-measured offset (within 15 mm); the "
                   "config value is well-supported by the segmented data.")
    elif norm_delta < 50.0:
        verdict = ("DIFFERS moderately (15-50 mm) from the hand-measured offset; "
                   "worth a rig re-measure or accepting x_hat if residuals are low.")
    else:
        verdict = ("DIFFERS substantially (>50 mm) from the hand-measured offset; "
                   "treat with care — check residuals and segmentation quality "
                   "before trusting x_hat.")
    print(f"INTERPRETATION: {verdict}")
    print("=" * 70 + "\n")


# ── synthetic self-test (exercises the LS core without perception) ──────────────

def run_self_test(n=60, noise_m=0.0, seed=0):
    """Generate a known x and N rotations, then recover x via solve_offset_ls."""
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(seed)
    x_true_mm = np.array([150.0, -200.0, 0.0])
    x_true_m = x_true_mm / 1000.0
    R = Rotation.random(n, random_state=seed).as_matrix()
    t = rng.uniform(-0.5, 0.5, size=(n, 3))           # arbitrary EE-tag origins
    p = t + np.einsum("nij,j->ni", R, x_true_m)
    if noise_m > 0:
        p = p + rng.normal(0.0, noise_m, size=p.shape)
    x_mm, stats = solve_offset_ls(R, t, p)
    err = np.linalg.norm(x_mm - x_true_mm)
    print("\n[self-test] synthetic recovery of a known offset")
    print(f"  x_true (mm) : [{x_true_mm[0]:.2f}, {x_true_mm[1]:.2f}, {x_true_mm[2]:.2f}]")
    print(f"  x_hat  (mm) : [{x_mm[0]:.2f}, {x_mm[1]:.2f}, {x_mm[2]:.2f}]")
    print(f"  |x_hat-x_true| = {err:.4f} mm   (noise={noise_m*1000:.1f} mm, N={n})")
    print(f"  LS overall RMS = {stats['overall_rms_mm']:.4f} mm")
    return x_mm, stats


# ── main ────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", nargs="?", default=None,
                    help="AprilTag sweep npz (tools/apriltag_calibrate.py stage_sweep)")
    ap.add_argument("--r-source", choices=["vision", "fk"], default="vision",
                    help="rotation source for R(i): vision (default) or fk")
    ap.add_argument("--handeye-npz", default=None,
                    help="WS-2a hand-eye result npz (required for --r-source fk)")
    ap.add_argument("--green-only", action="store_true",
                    help="restrict to green-flagged samples (sufficient coverage)")
    ap.add_argument("--self-test", action="store_true",
                    help="run only the synthetic self-test (no npz/perception)")
    ap.add_argument("--self-test-noise-mm", type=float, default=0.0,
                    help="add Gaussian noise (mm) to the self-test centroids")
    args = ap.parse_args(argv)

    if args.self_test or args.npz is None:
        if args.npz is None and not args.self_test:
            print("no npz given — running synthetic self-test only "
                  "(pass a sweep npz to solve from real frames).")
        run_self_test(noise_m=args.self_test_noise_mm / 1000.0)
        return 0

    npz_path = Path(args.npz)
    if not npz_path.is_file():
        print(f"ERROR: npz not found: {npz_path}", file=sys.stderr)
        return 2

    d = np.load(npz_path, allow_pickle=True)
    T_cam_world = vision_transforms_to_metres(d["T_cam_world"])
    T_cam_eetag = vision_transforms_to_metres(d["T_cam_eetag"])
    green = np.asarray(d["green"], dtype=bool) if "green" in d.files else None
    meta = d["meta"].item() if "meta" in d.files else {}
    n_total = T_cam_world.shape[0]
    # EEQUAT (fk only) and K (perception backprojection only) arrived with the
    # version-3 sweep save block (apriltag_calibrate.py:879-883); older runs on
    # disk omit them, so load lazily and require them only where actually used.
    EEQUAT = (np.asarray(d["EEQUAT"], dtype=np.float64)
              if "EEQUAT" in d.files else None)
    K = np.asarray(d["K"], dtype=np.float64) if "K" in d.files else None

    meta_offset = meta.get("t_eetag_ee_mm")
    meta_offset_mm = (np.asarray(meta_offset, dtype=np.float64)
                      if meta_offset is not None else None)

    # config offset for the side-by-side; fall back to the literal if config
    # cannot be imported on this host.
    try:
        import config
        config_offset_mm = np.asarray(config.APRILTAG_T_EETAG_EE_MM, dtype=np.float64)
    except Exception:
        config_offset_mm = CONFIG_OFFSET_MM_FALLBACK.copy()

    # Sample selection: drop NaN poses; optional green-only.
    idx = np.arange(n_total)
    if args.green_only and green is not None:
        idx = idx[green[idx]]
    finite_pose = np.array([np.all(np.isfinite(T_cam_world[i])) and
                            np.all(np.isfinite(T_cam_eetag[i])) for i in idx])
    idx = idx[finite_pose]
    if args.r_source == "fk":
        if EEQUAT is None:
            print("ERROR: --r-source fk needs EEQUAT, absent from this (pre-v3) "
                  "sweep. Re-run with --r-source vision.", file=sys.stderr)
            return 2
        finite_quat = np.array([np.all(np.isfinite(EEQUAT[i])) for i in idx])
        dropped = int((~finite_quat).sum())
        if dropped:
            print(f"fk: dropping {dropped} samples with NaN EEQUAT")
        idx = idx[finite_quat]

    print(f"loaded {npz_path.name}: {n_total} rows -> {len(idx)} usable samples "
          f"(green_only={args.green_only}, r-source={args.r_source})")

    # Build R(i), t_world_eetag(i).
    if args.r_source == "vision":
        R_list, t_list = build_rotations_vision(T_cam_world, T_cam_eetag, idx)
    else:
        if not args.handeye_npz:
            print("ERROR: --r-source fk requires --handeye-npz PATH "
                  "(WS-2a hand-eye result). Re-run with --r-source vision to use "
                  "the self-contained vision rotations.", file=sys.stderr)
            return 2
        R_world_base = load_R_world_base(args.handeye_npz)
        R_list, t_list = build_rotations_fk(
            EEQUAT, T_cam_world, T_cam_eetag, idx, R_world_base)

    # Segmentation / backprojection (perception-gated).
    frames_dir_name = meta.get("frames_dir")
    if not frames_dir_name:
        print("\nNo frames_dir in meta — this sweep was saved without "
              "--save-frames, so there are no scene frames to segment. "
              "Running the synthetic self-test only.\n")
        run_self_test()
        return 0
    frames_dir = npz_path.parent / frames_dir_name
    if not frames_dir.is_dir():
        print(f"\nframes_dir {frames_dir} missing on disk. "
              "Running the synthetic self-test only.\n")
        run_self_test()
        return 0
    if K is None:
        print("\nframes present but no K (camera intrinsics) in this sweep — "
              "cannot backproject. Running the synthetic self-test only.\n")
        run_self_test()
        return 0

    detector, depther, reason = _try_load_perception()
    if detector is None:
        print(f"\nperception unavailable on this host ({reason}) — segmentation "
              "step deferred to the rig; running synthetic self-test only.\n")
        run_self_test()
        return 0

    import cv2
    R_keep, t_keep, p_keep = [], [], []
    n_seg_fail = 0
    for k, i in enumerate(idx):
        frame_path = frames_dir / f"{i:05d}.jpg"
        if not frame_path.is_file():
            n_seg_fail += 1
            continue
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            n_seg_fail += 1
            continue
        p_world = centroid_world_from_frame(
            frame_bgr, K, T_cam_world[i], T_cam_eetag[i], detector, depther)
        if p_world is None:
            n_seg_fail += 1
            continue
        R_keep.append(R_list[k])
        t_keep.append(t_list[k])
        p_keep.append(p_world)
        if (k + 1) % 20 == 0:
            print(f"  segmented {k + 1}/{len(idx)} "
                  f"(kept {len(p_keep)}, failed {n_seg_fail})")

    if len(p_keep) < 2:
        print(f"ERROR: only {len(p_keep)} segmented samples (<2); cannot solve. "
              f"({n_seg_fail} frames failed segmentation)", file=sys.stderr)
        return 1

    print(f"segmentation: kept {len(p_keep)}/{len(idx)} "
          f"({n_seg_fail} failed)")
    x_mm, stats = solve_offset_ls(R_keep, t_keep, p_keep)
    report(x_mm, stats, config_offset_mm, meta_offset_mm, args.r_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
