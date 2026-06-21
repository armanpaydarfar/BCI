#!/usr/bin/env python3
"""
apriltag_calibrate.py — AprilTag gaze↔robot calibration tool (WS5 REV03).

Methodology: SoftwareDocs/projects/harmony-bci/gaze-calibration/
rev03-apriltag-methodology.md. Produces the calibration the control tool
(`tools/apriltag_control_test.py`) uses to drive the robot from gaze.

Tier-3 operator tool. Camera frames + gaze + intrinsics come from the existing
frame relay (`Utils.remote_frame_reader`), so it runs in the `lsl` env. The
panel's embedded relay (or `python -m Utils.frame_relay`) must be up first — the
sole Neon subscriber. The `collect` stage commands the free-arm opcodes
(`m`/`c`, and `h` home) exactly like the existing free-arm recorder; the
`detect`/`gaze` stages are camera-only; `solve` is offline.

Stages (`--stage`):

    detect   camera-only. Per-tag detection rate, decision-margin/hamming, pose
             jitter (translation mm / geodesic rotation deg), and a pose-flip
             count over a static window.
    gaze     camera-only, operator fixates the tag. Angular error between the
             gaze ray and the ray to the recovered tag centre.
    collect  + robot. Free-arm capture: per pose `m` (free) → hand-guide +
             fixate → `c` (capture+lock) records X (EE pos) + Q (joint angles)
             from one telemetry reply, plus the EE-tag/world-tag poses → P_world.
             Saves apriltag_capture_<UTC>.npz.
    solve    offline. Umeyama rigid fit of T_base_world from {P_world ↔ X},
             leave-one-out cross-validation, and writes a consolidated
             <stem>_calib.npz (X, Q, T_base_world) the control tool consumes.

Single-tag-first: each anchor is ONE tag id (`--world-tag-id`/`--ee-tag-id`).
The production method uses tag BUNDLES to defeat the planar pose-flip ambiguity;
the `detect` stage measures the flip rate so a non-zero count says escalate to a
bundle.

Examples:
    python tools/apriltag_calibrate.py --stage detect --world-tag-id 0 --tag-size 0.06
    python tools/apriltag_calibrate.py --stage gaze   --world-tag-id 0 --tag-size 0.06
    python tools/apriltag_calibrate.py --stage collect --with-robot \\
        --world-tag-id 0 --ee-tag-id 1 --tag-size 0.06 --t-eetag-ee 0 0 50
    python tools/apriltag_calibrate.py --stage solve runs/apriltag_capture_<UTC>.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from Utils.gaze.apriltag_calib import (  # noqa: E402
    angle_between_deg,
    average_rotation,
    eetag_to_world_point,
    gaze_ray_cam,
    geodesic_angle_deg,
    per_point_errors,
    umeyama_rigid,
)
from Utils.gaze.apriltag_detect import (  # noqa: E402
    RelayConsumer,
    detect_tags,
    load_detector,
)
from Utils.gaze.harmony_link import HarmonyLink  # noqa: E402


def _log(msg: str) -> None:
    print(f"[apriltag_calibrate] {msg}", flush=True)


# ── stage: detect ─────────────────────────────────────────────────────────────


def stage_detect(args, consumer: RelayConsumer) -> int:
    detector = load_detector(args.families)
    K = consumer.camera_matrix
    tag_id = args.world_tag_id if args.world_tag_id is not None else args.ee_tag_id
    if tag_id is None:
        _log("detect needs --world-tag-id (or --ee-tag-id) to track")
        return 2
    _log(f"detect: SINGLE tag {tag_id} for {args.duration:.0f}s "
         f"(tag-size {args.tag_size} m). Hold the tag static in view.")
    _log("NOTE: single-tag-first. The production method uses a tag BUNDLE; "
         "pose-flip (the dominant failure mode) is UNMITIGATED here and is "
         "measured below — a non-zero flip count means escalate to a bundle.")

    translations: List[np.ndarray] = []
    rmats: List[np.ndarray] = []
    zaxes: List[np.ndarray] = []
    margins: List[float] = []
    hammings: List[int] = []
    total = 0
    seen = 0
    last_idx = None
    deadline = time.time() + args.duration
    while time.time() < deadline:
        b = consumer.latest()
        if b is None or b.video is None or b.video.bgr is None:
            time.sleep(0.005)
            continue
        if b.video.frame_idx == last_idx:
            time.sleep(0.005)
            continue
        last_idx = b.video.frame_idx
        total += 1
        tags = detect_tags(detector, b.video.bgr, K, args.tag_size)
        if tag_id in tags:
            seen += 1
            T = tags[tag_id]["T"]
            translations.append(T[:3, 3].copy())
            rmats.append(T[:3, :3].copy())
            zaxes.append(T[:3, 2].copy())
            margins.append(tags[tag_id]["margin"])
            hammings.append(tags[tag_id]["hamming"])

    if total == 0:
        _log("no frames received — is the relay up?")
        return 1
    rate = seen / total
    _log(f"frames={total} detected={seen} rate={rate:.1%}")
    if seen >= 2:
        tr = np.vstack(translations)
        jit_mm = np.std(tr, axis=0)
        R_mean = average_rotation(rmats)
        geo = np.array([geodesic_angle_deg(R_mean, R) for R in rmats])
        rot_jit = float(np.std(geo))
        zmean = np.mean(np.vstack(zaxes), axis=0)
        nrm = np.linalg.norm(zmean)
        flips = int(np.sum([float(z @ zmean) < 0.0 for z in zaxes])) if nrm > 0 else 0
        _log(f"translation jitter (std) mm: x={jit_mm[0]:.2f} y={jit_mm[1]:.2f} "
             f"z={jit_mm[2]:.2f}  (norm {np.linalg.norm(jit_mm):.2f})")
        _log(f"rotation jitter (geodesic std) deg: {rot_jit:.2f}")
        _log(f"pose flips: {flips}/{seen} (tag +Z sign reversals — want 0)")
        _log(f"decision margin: median={np.median(margins):.1f} "
             f"min={np.min(margins):.1f}   hamming max={max(hammings)}")
        ok = (rate >= 0.90 and np.linalg.norm(jit_mm) < 5.0
              and rot_jit < 2.0 and flips == 0)
        _log(f"VERDICT: {'PASS' if ok else 'REVIEW'} "
             f"(targets: rate≥90%, |trans jitter|<5mm, rot<2°, flips=0)")
    return 0


# ── stage: gaze ───────────────────────────────────────────────────────────────


def stage_gaze(args, consumer: RelayConsumer) -> int:
    detector = load_detector(args.families)
    K = consumer.camera_matrix
    tag_id = args.world_tag_id if args.world_tag_id is not None else args.ee_tag_id
    if tag_id is None:
        _log("gaze needs --world-tag-id (or --ee-tag-id) to fixate")
        return 2
    _log(f"gaze: fixate tag {tag_id} steadily for {args.duration:.0f}s.")

    errors: List[float] = []
    last_idx = None
    deadline = time.time() + args.duration
    while time.time() < deadline:
        b = consumer.latest()
        if b is None or b.video is None or b.video.bgr is None:
            time.sleep(0.005)
            continue
        if b.video.frame_idx == last_idx:
            time.sleep(0.005)
            continue
        last_idx = b.video.frame_idx
        if b.gaze is None or not (np.isfinite(b.gaze.x) and np.isfinite(b.gaze.y)):
            continue
        if not getattr(b, "worn", True):
            continue
        tags = detect_tags(detector, b.video.bgr, K, args.tag_size)
        if tag_id not in tags:
            continue
        ray = gaze_ray_cam(b.gaze.x, b.gaze.y, K)
        if ray is None:
            continue
        tag_centre_ray = tags[tag_id]["T"][:3, 3]
        errors.append(angle_between_deg(ray, tag_centre_ray))

    errors = [e for e in errors if np.isfinite(e)]
    if not errors:
        _log("no valid (gaze + tag) frames — check worn/fixation/relay")
        return 1
    arr = np.array(errors)
    _log(f"samples={arr.size} gaze-to-tag angular error deg: "
         f"median={np.median(arr):.2f} p90={np.percentile(arr, 90):.2f} "
         f"max={arr.max():.2f}")
    ok = np.median(arr) <= 1.8
    _log(f"VERDICT: {'PASS' if ok else 'REVIEW'} (target median ≤1.8°, Neon budget)")
    return 0


# ── stage: collect (free-arm m/c capture; records X + Q) ─────────────────────


def stage_collect(args, consumer: RelayConsumer) -> int:
    if args.world_tag_id is None or args.ee_tag_id is None:
        _log("collect needs both --world-tag-id and --ee-tag-id")
        return 2
    detector = load_detector(args.families)
    K = consumer.camera_matrix
    offset_mm = np.asarray(args.t_eetag_ee, dtype=float)
    link: Optional[HarmonyLink] = None
    if args.with_robot:
        link = HarmonyLink(args.robot_ip, args.robot_port, args.bind_ip,
                           args.bind_port, side=args.side)
        _log(f"robot: dial {args.robot_ip}:{args.robot_port} bind "
             f"{args.bind_ip}:{args.bind_port} side={args.side} (m/c free-arm)")
    else:
        _log("no --with-robot: X/Q will be NaN (camera-side dry run, no capture)")

    _log("collect: per pose — Enter to FREE the arm (it goes limp), hand-guide "
         "it + fixate the workspace, then Enter to CAPTURE (lock + record). "
         "'q'+Enter to finish.")
    p_world_rows: List[np.ndarray] = []
    x_rows: List[np.ndarray] = []
    q_rows: List[np.ndarray] = []
    tcw_rows: List[np.ndarray] = []
    tce_rows: List[np.ndarray] = []

    try:
        while True:
            cmd = input(f"[{len(p_world_rows)} captured] Enter=free arm / q=finish > ").strip().lower()
            if cmd == "q":
                break
            if link is not None and not link.free_arm():
                _log("  could not free the arm (ACK:MASTER_FREE not seen); retry")
                continue
            input("    hand-guide + fixate, then Enter to capture > ")

            if link is not None:
                # Settle guard: the arm is limp (zero stiffness) and can drift,
                # so a capture mid-drift would record the wrong joints (the
                # captured pose IS the calibration ground truth). Reject if the
                # joints are still moving between two quick telemetry reads.
                s1 = link.query_state()
                time.sleep(0.05)
                s2 = link.query_state()
                if s1 is not None and s2 is not None:
                    dmax = float(np.max(np.abs(s1["q"] - s2["q"])))
                    if dmax > args.settle_eps_rad:
                        _log(f"    arm still moving (max|dq|={dmax:.4f} rad > "
                             f"{args.settle_eps_rad}); hold steady and retry")
                        continue
                cap = link.capture_pose()
                if cap is None:
                    _log("    capture failed (no ACK:CAPTURED_LOCKED); retry")
                    continue
                x_ee, q_joints = cap["ee"], cap["q"]
            else:
                x_ee, q_joints = np.full(3, np.nan), np.full(7, np.nan)

            b = consumer.latest()
            if b is None or b.video is None or b.video.bgr is None:
                _log("    no frame; relay up? (capture discarded)")
                continue
            tags = detect_tags(detector, b.video.bgr, K, args.tag_size)
            if args.world_tag_id not in tags or args.ee_tag_id not in tags:
                _log(f"    need both tags; saw {sorted(tags.keys())}. (discarded)")
                continue
            p_world = eetag_to_world_point(
                tags[args.world_tag_id]["T"], tags[args.ee_tag_id]["T"], offset_mm)

            p_world_rows.append(p_world)
            x_rows.append(x_ee)
            q_rows.append(q_joints)
            tcw_rows.append(tags[args.world_tag_id]["T"])
            tce_rows.append(tags[args.ee_tag_id]["T"])
            _log(f"    captured #{len(p_world_rows)}: P_world={np.round(p_world,1)} "
                 f"X={np.round(x_ee,1)} mm")
    finally:
        if link is not None:
            link.close()

    if len(p_world_rows) < 3:
        _log(f"only {len(p_world_rows)} captures (<3) — not saving a solvable npz")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone
    stamp = args.utc_stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"apriltag_capture_{stamp}.npz"
    meta = {
        "version": 3,
        "scheme": "ee_mounted_tag_free_roam",
        "side": args.side,
        "world_tag_id": args.world_tag_id,
        "ee_tag_id": args.ee_tag_id,
        "tag_size_m": args.tag_size,
        "t_eetag_ee_mm": offset_mm.tolist(),
        "with_robot": bool(args.with_robot),
        "units": {"P_world": "mm", "X": "mm", "Q": "rad", "t_eetag_ee": "mm"},
    }
    np.savez_compressed(
        out_path,
        P_world=np.vstack(p_world_rows),
        X=np.vstack(x_rows),
        Q=np.vstack(q_rows),
        T_cam_world=np.stack(tcw_rows),
        T_cam_eetag=np.stack(tce_rows),
        meta=np.array(meta, dtype=object),
    )
    _log(f"saved {len(p_world_rows)} captures → {out_path}")
    if args.with_robot:
        _log("solve it: "
             f"python tools/apriltag_calibrate.py --stage solve {out_path}")
    return 0


# ── stage: solve (offline) — writes the consolidated calibration ─────────────


def stage_solve(args) -> int:
    npz_path = Path(args.npz)
    if not npz_path.is_file():
        _log(f"npz not found: {npz_path}")
        return 2
    z = np.load(npz_path, allow_pickle=True)
    P_world = np.asarray(z["P_world"], dtype=float)
    X = np.asarray(z["X"], dtype=float)
    Q = np.asarray(z["Q"], dtype=float)

    finite = (np.all(np.isfinite(P_world), axis=1) & np.all(np.isfinite(X), axis=1)
              & np.all(np.isfinite(Q), axis=1))
    P_world, X, Q = P_world[finite], X[finite], Q[finite]
    n = P_world.shape[0]
    if n < 3:
        _log(f"only {n} finite (P_world,X,Q) rows (<3) — cannot solve. "
             "Was --with-robot used during collect?")
        return 1

    T_base_world, rms = umeyama_rigid(P_world, X)  # map world points → base frame
    errs = per_point_errors(T_base_world, P_world, X)
    _log(f"Umeyama T_base_world from {n} points:")
    for row in T_base_world:
        _log("  [" + "  ".join(f"{v:9.3f}" for v in row) + "]")
    _log(f"RMS residual = {rms:.2f} mm   per-point: "
         f"median={np.median(errs):.2f} max={errs.max():.2f} mm")

    if n >= 4:
        loo = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            T_i, _ = umeyama_rigid(P_world[mask], X[mask])
            pred = T_i[:3, :3] @ P_world[i] + T_i[:3, 3]
            loo.append(np.linalg.norm(pred - X[i]))
        loo = np.array(loo)
        _log(f"leave-one-out error: median={np.median(loo):.2f} "
             f"max={loo.max():.2f} mm (generalisation estimate)")

    ok = rms < 20.0
    _log(f"VERDICT: {'PASS' if ok else 'REVIEW'} (target RMS ≲10–20 mm; refine vs REV01)")

    # Consolidated calibration the control tool consumes: the (X→Q) library +
    # the world→base transform + provenance, written beside the capture.
    src_meta = z["meta"].item() if "meta" in z.files else {}
    calib_meta = dict(src_meta)
    calib_meta.update({"umeyama_rms_mm": float(rms), "n_points": int(n),
                       "source_capture": npz_path.name})
    out_path = npz_path.with_name(npz_path.stem.replace("apriltag_capture", "apriltag")
                                  + "_calib.npz")
    np.savez_compressed(
        out_path,
        X=X, Q=Q, T_base_world=T_base_world,
        per_point_errors_mm=errs,
        meta=np.array(calib_meta, dtype=object),
    )
    _log(f"calibration → {out_path}")
    _log("drive the robot: "
         f"python tools/apriltag_control_test.py --calib {out_path} "
         "--world-tag-id <id> --tag-size <m>")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    import config as cfg
    p = argparse.ArgumentParser(description="AprilTag gaze↔robot calibration tool")
    p.add_argument("--stage", required=True,
                   choices=["detect", "gaze", "collect", "solve"])
    p.add_argument("npz", nargs="?", default=None,
                   help="solve stage: path to an apriltag_capture_*.npz")
    p.add_argument("--families", default="tag36h11")
    p.add_argument("--tag-size", type=float, default=0.06,
                   help="tag edge length in METRES (default 0.06)")
    p.add_argument("--world-tag-id", type=int, default=None)
    p.add_argument("--ee-tag-id", type=int, default=None)
    p.add_argument("--t-eetag-ee", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   metavar=("X", "Y", "Z"),
                   help="hand-measured EE-tag→EE offset vector in MM")
    p.add_argument("--duration", type=float, default=10.0,
                   help="detect/gaze sampling window (s)")
    p.add_argument("--with-robot", action="store_true",
                   help="collect: free-arm m/c capture of X (EE pos) + Q (joints)")
    p.add_argument("--settle-eps-rad", type=float, default=0.01,
                   help="collect: reject a capture if max|dq| between two reads "
                        "exceeds this (arm still moving)")
    p.add_argument("--side", default=None, help="robot active side R/L "
                   "(default: env HARMONY_ACTIVE_SIDE or R)")
    p.add_argument("--relay-host", default=None, help="default: config.FRAME_RELAY_DIAL_HOST")
    p.add_argument("--relay-port", type=int, default=None, help="default: config.FRAME_RELAY_PORT")
    p.add_argument("--robot-ip", default=None, help="default: config.UDP_ROBOT[IP]")
    p.add_argument("--robot-port", type=int, default=None, help="default: config.UDP_ROBOT[PORT]")
    p.add_argument("--bind-ip", default=None,
                   help="default: config.UDP_CONTROL_BIND[IP] (mirrors the recorder)")
    p.add_argument("--bind-port", type=int, default=None,
                   help="default: config.UDP_CONTROL_BIND[PORT]")
    p.add_argument("--out-dir", default="runs", help="collect: directory for the saved npz")
    p.add_argument("--utc-stamp", default=None,
                   help="collect: override the auto UTC stamp in the npz filename")
    args = p.parse_args(argv)

    import os
    args.relay_host = args.relay_host or getattr(cfg, "FRAME_RELAY_DIAL_HOST", "127.0.0.1")
    args.relay_port = args.relay_port or int(getattr(cfg, "FRAME_RELAY_PORT", 5591))
    robot = getattr(cfg, "UDP_ROBOT", {"IP": "192.168.2.1", "PORT": 8080})
    bind = getattr(cfg, "UDP_CONTROL_BIND", {"IP": "0.0.0.0", "PORT": 8080})
    args.robot_ip = args.robot_ip or robot["IP"]
    args.robot_port = args.robot_port or int(robot["PORT"])
    args.bind_ip = args.bind_ip or bind["IP"]
    args.bind_port = args.bind_port or int(bind["PORT"])
    args.side = (args.side or os.environ.get("HARMONY_ACTIVE_SIDE", "R")).upper()
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.stage == "solve":
        if not args.npz:
            _log("solve needs an npz path argument")
            return 2
        return stage_solve(args)

    _log(f"connecting to relay {args.relay_host}:{args.relay_port} …")
    consumer = RelayConsumer(args.relay_host, args.relay_port)
    try:
        if consumer.latest() is None:
            time.sleep(0.5)
        if args.stage == "detect":
            return stage_detect(args, consumer)
        if args.stage == "gaze":
            return stage_gaze(args, consumer)
        if args.stage == "collect":
            return stage_collect(args, consumer)
    finally:
        consumer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
