#!/usr/bin/env python3
"""
apriltag_control_test.py — drive the Harmony robot from gaze using an AprilTag
calibration (WS5 REV03). The experimental validation of the gaze↔robot mapping,
standalone — NOT the EEG-gated experiment driver.

Methodology: SoftwareDocs/projects/harmony-bci/gaze-calibration/
rev03-apriltag-methodology.md §5. Per fixation:

    gaze pixel → ray (cam frame) → intersect the world-tag plane → P_world
      → P_base = T_base_world · P_world → nearest calibrated EE pose X[idx]
      → joint vector Q[idx] → workspace clamp → command the robot.

The robot accepts only joint angles (verified in reports/cpp.md), so the command
is the calibrated, known-safe Q[idx] — no inverse kinematics. **Tier-1: this
commands motion.** Safety:

  - Operator-gated SINGLE moves: every move shows the target, the chosen joint
    pose, the nearest-neighbour distance, and the clamp state, then waits for an
    explicit confirm. There is no autonomous loop.
  - Workspace clamp: Q is clipped to the calibration envelope (±5% margin,
    matching Utils/gaze/calibration_mapping.WORKSPACE_BOUNDS_MARGIN). The robot
    enforces NO bounds (reports/cpp.md §7.2), so this is the only guard.
  - A far-fixation gate: if the nearest calibrated pose is farther than
    --max-nn-dist-mm from the fixated point, the move requires an extra
    confirmation (the fixation is outside the calibrated region).
  - 'h' homes the arm; Ctrl-C / 'q' quits.

The relay (sole Neon subscriber) must be up. Binds the robot control endpoint,
so it must NOT run alongside the recorder/online driver (fails fast EADDRINUSE).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from Utils.gaze.apriltag_calib import (  # noqa: E402
    gaze_ray_cam,
    invert_transform,
    ray_plane_intersection,
    tag_plane_in_cam,
    transform_point,
)
from Utils.gaze.apriltag_detect import RelayConsumer, detect_tags, load_detector  # noqa: E402
from Utils.gaze.apriltag_world import recover_world_pose, world_map_from_arrays  # noqa: E402
from Utils.gaze.calibration_mapping import WORKSPACE_BOUNDS_MARGIN  # noqa: E402
from Utils.gaze.harmony_link import HarmonyLink  # noqa: E402


def _log(msg: str) -> None:
    print(f"[apriltag_control] {msg}", flush=True)


# ── pure helpers (hardware-free, unit-tested) ────────────────────────────────


def workspace_bounds(Q: np.ndarray, margin: float = WORKSPACE_BOUNDS_MARGIN
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-joint clamp envelope from the calibration library: [min-margin·span,
    max+margin·span]. Mirrors GazeCalibrationMappingV2's clamp."""
    q_min = Q.min(axis=0)
    q_max = Q.max(axis=0)
    span = q_max - q_min
    return q_min - margin * span, q_max + margin * span


def clamp_joints(q: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray
                 ) -> Tuple[np.ndarray, bool]:
    """Clip a joint vector into the workspace envelope. Returns (clipped,
    was_clamped)."""
    clipped = np.clip(q, q_lo, q_hi)
    return clipped, bool(np.any(clipped != q))


def nearest_pose(X: np.ndarray, p_base: np.ndarray) -> Tuple[int, float]:
    """Index + Euclidean distance (mm) of the nearest calibrated EE position to
    a fixated base-frame point."""
    d = np.linalg.norm(X - p_base[None, :], axis=1)
    idx = int(np.argmin(d))
    return idx, float(d[idx])


def gaze_point_in_base(gaze_x: float, gaze_y: float, K: np.ndarray,
                       T_cam_world: np.ndarray, T_base_world: np.ndarray
                       ) -> Optional[np.ndarray]:
    """Full §5 chain for one gaze sample: pixel → ray → world-tag-plane hit →
    world frame → base frame. Returns P_base (mm) or None if the ray misses."""
    ray = gaze_ray_cam(gaze_x, gaze_y, K)
    if ray is None:
        return None
    point, normal = tag_plane_in_cam(T_cam_world)
    hit_cam = ray_plane_intersection(np.zeros(3), ray, point, normal)
    if hit_cam is None:
        return None
    p_world = transform_point(invert_transform(T_cam_world), hit_cam)
    return transform_point(T_base_world, p_world)


# ── gaze sampling (median over a short window) ───────────────────────────────


def _sample_p_base(consumer: RelayConsumer, detector, K, T_base_world,
                   world_map: dict, tag_size: float, dur_s: float
                   ) -> Optional[np.ndarray]:
    """Average (median) the fixated base-frame point over a short window, using
    only frames where gaze is valid and ≥1 mapped world tag is seen. The world
    pose is fused from whichever mapped tags are visible (occlusion-robust)."""
    pts: List[np.ndarray] = []
    last_idx = None
    deadline = time.time() + dur_s
    while time.time() < deadline:
        b = consumer.latest()
        if b is None or b.video is None or b.video.bgr is None or b.video.frame_idx == last_idx:
            time.sleep(0.005)
            continue
        last_idx = b.video.frame_idx
        if b.gaze is None or not (np.isfinite(b.gaze.x) and np.isfinite(b.gaze.y)):
            continue
        if not getattr(b, "worn", True):
            continue
        tags = detect_tags(detector, b.video.bgr, K, tag_size)
        world_view = {i: tags[i]["T"] for i in world_map["ids"] if i in tags}
        T_cam_world = recover_world_pose(world_view, world_map)
        if T_cam_world is None:
            continue
        p_base = gaze_point_in_base(b.gaze.x, b.gaze.y, K, T_cam_world, T_base_world)
        if p_base is not None:
            pts.append(p_base)
    if not pts:
        return None
    return np.median(np.vstack(pts), axis=0)


# ── main control loop ─────────────────────────────────────────────────────────


def _commit_move(link: HarmonyLink, q_cmd: np.ndarray, idx: int, dur_s: float) -> None:
    """Command one move and ALWAYS read back the actual pose. ``send_joint_command``
    distinguishes 'stage_failed' (arm did not move) from 'go_unconfirmed' (coords
    staged but no ACK:g — the arm MAY be moving), so a lost go-ACK is never
    reported as a clean failure (the fail-open hazard)."""
    _log(f"GO → pose #{idx}")
    status = link.send_joint_command(q_cmd, dur_s)
    if status == "ok":
        _log("move: committed (ACK:g)")
    elif status == "stage_failed":
        _log("move: NOT sent — coords not staged (no ACK:COORDS_STAGED_RAD); arm did NOT move")
    else:  # go_unconfirmed
        _log("move: WARNING — coords staged but no ACK:g; the arm MAY be moving. "
             "Verify visually before the next command.")
    st = link.query_state()
    if st is not None:
        _log(f"  readback: actual EE = {np.round(st['ee'], 1)} mm")
    else:
        _log("  readback: telemetry timed out — verify the arm visually")


def run(args, consumer: RelayConsumer, link: HarmonyLink) -> int:
    z = np.load(args.calib, allow_pickle=True)
    missing = [k for k in ("X", "Q", "T_base_world") if k not in z.files]
    if missing:
        _log(f"{args.calib} is not an AprilTag calibration (missing {missing}); "
             "select an apriltag_*_calib.npz from the calibration tool's solve stage")
        return 2
    X = np.asarray(z["X"], dtype=float)
    Q = np.asarray(z["Q"], dtype=float)
    T_base_world = np.asarray(z["T_base_world"], dtype=float)
    # The world map (registered during collect) lets any visible subset of world
    # tags recover the SAME world frame — occlusion-robust. Tag size defaults
    # from meta so the panel button needs only --calib.
    if not all(k in z.files for k in ("world_map_ref", "world_map_ids", "world_map_rels")):
        _log(f"{args.calib} has no world map — re-run the calibration (collect "
             "now registers a multi-tag world map)")
        return 2
    world_map = world_map_from_arrays(z["world_map_ref"], z["world_map_ids"], z["world_map_rels"])
    meta = z["meta"].item() if "meta" in z.files else {}
    tag_size = (args.tag_size if args.tag_size is not None
                else float(meta.get("tag_size_m", 0.06)))
    q_lo, q_hi = workspace_bounds(Q)
    detector = load_detector(args.families)
    K = consumer.camera_matrix
    _log(f"calibration: {X.shape[0]} poses, RMS in meta. Workspace clamp from Q±"
         f"{WORKSPACE_BOUNDS_MARGIN:.0%}. Robot dur={args.dur:.1f}s.")
    _log("Per move: fixate a calibrated target, Enter to RESOLVE; review; then "
         "'g' to GO ('g' again to confirm a far fixation), 'r' to re-resolve, "
         "'h' to home, 'q' to quit. NO autonomous motion.")

    pending: Optional[Tuple[np.ndarray, int, float, bool]] = None  # (q_cmd, idx, dist, clamped)
    far_armed = False  # a far fixation needs a SECOND 'g' to commit (review safety)
    while True:
        cmd = input("> ").strip().lower()
        if cmd == "q":
            break
        if cmd == "h":
            ok = link.home(args.dur)
            _log("home: " + ("ACK" if ok else "no ACK"))
            pending, far_armed = None, False
            continue
        if cmd == "g":
            if pending is None:
                _log("nothing resolved yet — press Enter to resolve a fixation first")
                continue
            q_cmd, idx, dist, clamped = pending
            if dist > args.max_nn_dist_mm and not far_armed:
                _log(f"nearest calibrated pose is {dist:.0f} mm away (> "
                     f"{args.max_nn_dist_mm}); fixation is outside the calibrated "
                     "region. Press 'g' AGAIN to override, or 'r' to re-resolve.")
                far_armed = True
                continue
            _commit_move(link, q_cmd, idx, args.dur)
            pending, far_armed = None, False
            continue

        # default (Enter / 'r'): resolve a fixation
        _log(f"resolving — fixate the target for {args.sample_s:.1f}s …")
        p_base = _sample_p_base(consumer, detector, K, T_base_world,
                                world_map, tag_size, args.sample_s)
        if p_base is None:
            _log("no valid gaze+world-tag samples — keep the world tag in view and fixate")
            pending, far_armed = None, False
            continue
        idx, dist = nearest_pose(X, p_base)
        q_cmd, clamped = clamp_joints(Q[idx], q_lo, q_hi)
        _log(f"fixated P_base = {np.round(p_base,1)} mm")
        _log(f"nearest calibrated pose #{idx}: X={np.round(X[idx],1)} mm, "
             f"dist={dist:.1f} mm, clamped={clamped}")
        _log(f"joint target (rad) = {np.round(q_cmd,4).tolist()}")
        if dist > args.max_nn_dist_mm:
            _log(f"WARNING: {dist:.0f} mm from the nearest calibrated pose "
                 "(outside the calibrated region).")
        _log("press 'g' to GO, 'r' to re-resolve")
        pending, far_armed = (q_cmd, idx, dist, clamped), False
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    import config as cfg
    p = argparse.ArgumentParser(description="Drive the robot from gaze via an AprilTag calibration")
    p.add_argument("--calib", required=True, help="apriltag_*_calib.npz from the calibration tool")
    p.add_argument("--tag-size", type=float, default=None,
                   help="world tag edge length in METRES (default: from the calibration meta)")
    p.add_argument("--families", default="tag36h11")
    p.add_argument("--dur", type=float, default=5.0, help="robot move duration (s)")
    p.add_argument("--sample-s", type=float, default=0.6, help="gaze-fixation sampling window (s)")
    p.add_argument("--max-nn-dist-mm", type=float, default=80.0,
                   help="warn/guard if the nearest calibrated pose is farther than this")
    p.add_argument("--side", default=None, help="robot active side R/L")
    p.add_argument("--relay-host", default=None)
    p.add_argument("--relay-port", type=int, default=None)
    p.add_argument("--robot-ip", default=None)
    p.add_argument("--robot-port", type=int, default=None)
    p.add_argument("--bind-ip", default=None)
    p.add_argument("--bind-port", type=int, default=None)
    args = p.parse_args(argv)

    import os
    args.relay_host = args.relay_host or getattr(cfg, "FRAME_RELAY_DIAL_HOST", "127.0.0.1")
    args.relay_port = args.relay_port or int(getattr(cfg, "FRAME_RELAY_PORT", 5591))
    robot = getattr(cfg, "UDP_ROBOT", {"IP": "192.168.2.1", "PORT": 8080})
    # The robot sends command ACKs to a FIXED control address (192.168.2.2 per
    # the C++ wire protocol), so the bind must be that address — not a wildcard.
    bind = getattr(cfg, "UDP_CONTROL_BIND", {"IP": "192.168.2.2", "PORT": 8080})
    args.robot_ip = args.robot_ip or robot["IP"]
    args.robot_port = args.robot_port or int(robot["PORT"])
    args.bind_ip = args.bind_ip or bind["IP"]
    args.bind_port = args.bind_port or int(bind["PORT"])
    args.side = (args.side or os.environ.get("HARMONY_ACTIVE_SIDE", "R")).upper()
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not Path(args.calib).is_file():
        _log(f"calibration not found: {args.calib}")
        return 2
    _log(f"connecting to relay {args.relay_host}:{args.relay_port} …")
    consumer = RelayConsumer(args.relay_host, args.relay_port)
    link = HarmonyLink(args.robot_ip, args.robot_port, args.bind_ip,
                       args.bind_port, side=args.side)
    try:
        if consumer.latest() is None:
            time.sleep(0.5)
        return run(args, consumer, link)
    finally:
        link.close()
        consumer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
