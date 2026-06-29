#!/usr/bin/env python3
"""
apriltag_control_test.py — drive the Harmony robot from gaze using an AprilTag
calibration (WS5 REV04, planar). The experimental validation of the gaze↔robot
mapping, standalone — NOT the EEG-gated experiment driver.

Methodology: SoftwareDocs/projects/harmony-bci/gaze-calibration/
rev04-planar-coverage-methodology.md §6. Per fixation:

    gaze pixel → ray (cam frame) → intersect the table plane → P_world → table
      (u,v) → nearest calibrated (u,v) (GazeCalibrationMappingV3) → joint vector
      Q[idx] → workspace clamp → command the robot.

REV04 drops the REV03 3-D T_base_world step: the runtime gaze∩plane point is on
the table, so mapping table (u,v)→Q directly removes the height mismatch that
put every first-HIL fixation >500 mm from the library (verification report §5).
A REV03 rigid calib (T_base_world) is rejected — re-solve a sweep npz.

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
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from Utils.gaze.apriltag_calib import (  # noqa: E402
    gaze_ray_cam,
    invert_transform,
    ray_plane_intersection,
    transform_point,
)
from Utils.gaze.apriltag_detect import (  # noqa: E402
    RelayConsumer,
    detect_tags,
    load_detector,
    recover_world_pose_pnp,
)
from Utils.gaze.apriltag_world import (  # noqa: E402
    plane_coords,
    recover_world_pose,
    world_map_from_arrays,
)
from Utils.gaze.calibration_mapping import (  # noqa: E402
    WORKSPACE_BOUNDS_MARGIN,
    GazeCalibrationMappingV3,
)
from Utils.gaze.harmony_link import HarmonyLink  # noqa: E402
from Utils.perception_clients import VLMClient  # noqa: E402


def _log(msg: str) -> None:
    print(f"[apriltag_control] {msg}", flush=True)


# WS-1 object-target tolerances (pixels, scene-cam full-res).
_HIT_OUTSIDE_TOL_PX = 40.0   # max gaze-outside-mask slack before the hit is rejected
_BOTTOM_BAND_PX = 3.0        # rows above the lowest mask point averaged for the footprint x
_GAZE_DIVERGENCE_TOL_PX = 80.0  # service-frame vs our gaze gap above which we distrust the masks


# ── pure helpers (hardware-free, unit-tested) ────────────────────────────────


def gaze_point_in_plane_uv(gaze_x: float, gaze_y: float, K: np.ndarray,
                           T_cam_world: np.ndarray,
                           plane_point_world, plane_normal_world) -> Optional[np.ndarray]:
    """REV04 §6 runtime chain for one gaze sample: pixel → ray → table-plane hit →
    world frame → table-plane ``(u,v)``. Returns ``(u,v)`` (mm) or None if the ray
    misses. This is the REV03 ``gaze_point_in_base`` chain with the ``T_base_world``
    step **removed** (rev04 §1) and a final projection into the deterministic
    in-plane basis (``plane_coords``) — so the runtime point and the calibration
    library point are the same kind of point (both on the table plane). The world
    plane is fitted across all world tags (robust to any one tag's orientation
    noise), transformed into the camera frame via the fused ``T_cam_world``."""
    ray = gaze_ray_cam(gaze_x, gaze_y, K)
    if ray is None:
        return None
    point = transform_point(T_cam_world, plane_point_world)        # plane → cam
    normal = T_cam_world[:3, :3] @ np.asarray(plane_normal_world, dtype=float)
    hit_cam = ray_plane_intersection(np.zeros(3), ray, point, normal)
    if hit_cam is None:
        return None
    p_world = transform_point(invert_transform(T_cam_world), hit_cam)
    return plane_coords(p_world, plane_point_world, plane_normal_world)


def _object_target_pixel(detections: List[dict], gaze_xy: Tuple[float, float],
                         source: str) -> Optional[Tuple[float, float]]:
    """WS-1: pick the scene-frame pixel to aim at from the fixated object's mask,
    so the target is invariant to *where on the object* gaze landed (killing the
    tall-object top/bottom overshoot). ``detections`` are vlm_service ``segment``
    dicts; only those with a ``mask_polygon`` are candidates.

    The fixated object is the **smallest mask that contains the gaze**. This is
    deliberate, not "most inside": FastSAM returns nested masks (a cup sits inside
    the table/tray blob), and ``segment`` is unfiltered, so several masks contain
    the gaze. Picking by *largest signed point-in-polygon distance* would always
    return the enclosing blob (a point is "more inside" the bigger region) → the
    table centroid, not the cup. Smallest-containing-area selects the specific
    fixated object. If no mask contains the gaze, fall back to the nearest mask
    within ``_HIT_OUTSIDE_TOL_PX`` (jitter slack); beyond that the hit is rejected
    (None → caller falls back to raw gaze). From the chosen mask:
      - ``centroid``: the mask area centroid (cv2 moments) — the §1a controlled
        point, consistent with WS-2b's calibration target.
      - ``bottom``: the footprint pixel — mean x along the lowest mask row — i.e.
        the object↔table contact line, the depth-free overshoot fix (§1.3).
    Returns ``(px, py)`` or None. Pure (no I/O); unit-tested."""
    if source not in ("centroid", "bottom"):
        return None
    gx, gy = float(gaze_xy[0]), float(gaze_xy[1])
    contained = []          # (area, cnt) for masks that contain the gaze
    nearest_cnt = None      # fallback: closest mask when gaze is outside all
    nearest_dist = -1e18
    for det in detections:
        poly = det.get("mask_polygon")
        if poly is None or len(poly) < 3:
            continue
        cnt = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        dist = cv2.pointPolygonTest(cnt, (gx, gy), True)  # +inside, -outside (px)
        if dist >= 0.0:
            contained.append((float(cv2.contourArea(cnt)), cnt))
        elif dist > nearest_dist:
            nearest_dist, nearest_cnt = dist, cnt
    if contained:
        best_poly = min(contained, key=lambda c: c[0])[1]   # smallest containing
    elif nearest_cnt is not None and nearest_dist >= -_HIT_OUTSIDE_TOL_PX:
        best_poly = nearest_cnt                              # gaze just outside
    else:
        return None
    pts = best_poly.reshape(-1, 2).astype(float)
    if source == "centroid":
        m = cv2.moments(best_poly)
        if abs(m["m00"]) > 1e-6:
            return (m["m10"] / m["m00"], m["m01"] / m["m00"])
        return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
    # bottom: mean x along the lowest few rows of the mask (footprint centre)
    y_max = float(pts[:, 1].max())
    on_bottom = pts[:, 1] >= y_max - _BOTTOM_BAND_PX
    return (float(pts[on_bottom, 0].mean()), y_max)


def _object_target_uv(vlm: "VLMClient", source: str, gaze_uv: np.ndarray, diag: dict,
                      K: np.ndarray, world_map: dict) -> Tuple[np.ndarray, str]:
    """WS-1: try to replace the raw-gaze table ``(u,v)`` with the fixated object's
    centroid/footprint ``(u,v)``. Asks vlm_service for masks, picks the target pixel
    (``_object_target_pixel``), and casts it through this fixation's pose
    (``diag['T_cam_world']``) onto the table plane. Returns ``(uv, source_label)``;
    on ANY miss — service down, no mask hit, ray misses the plane — returns the
    untouched ``gaze_uv`` with label ``'gaze'`` (and logs why), so the degraded path
    is exactly the frozen REV05 behaviour. The try/except is a documented degradation
    path (CLAUDE.md), not blanket suppression — the reason is always surfaced.

    Cross-frame caveat: the masks are FastSAM run on the **service's** latest frame
    (a different process/instant), while the pose and gaze come from **this tool's**
    sampling window. Under a static fixation the 2-D pixel drift is small and the
    plane absorbs depth error, but a moving head would reproject a stale pixel. We
    guard it by comparing the service's gaze (returned in the reply) to our median
    gaze and falling back to raw gaze when they diverge beyond
    ``_GAZE_DIVERGENCE_TOL_PX`` — i.e. the two frames disagree on where the eye is."""
    tcw = diag.get("T_cam_world")
    gpx = diag.get("gaze_px")
    if tcw is None or gpx is None:
        _log("WS-1: no pose/gaze for this fixation; using raw gaze")
        return gaze_uv, "gaze"
    try:
        seg = vlm.segment(include_masks=True)
    except (OSError, json.JSONDecodeError) as exc:
        _log(f"WS-1: segment request failed ({exc!r}); using raw gaze")
        return gaze_uv, "gaze"
    if not (isinstance(seg, dict) and seg.get("ok")):
        _log(f"WS-1: segmentation unavailable ({(seg or {}).get('error')}); using raw gaze")
        return gaze_uv, "gaze"
    svc_gaze = seg.get("gaze_px")
    if svc_gaze is not None and np.hypot(svc_gaze[0] - gpx[0],
                                         svc_gaze[1] - gpx[1]) > _GAZE_DIVERGENCE_TOL_PX:
        _log(f"WS-1: service gaze {np.round(svc_gaze,0)} diverges from ours "
             f"{np.round(gpx,0)} (stale/moving frame); using raw gaze")
        return gaze_uv, "gaze"
    tpx = _object_target_pixel(seg.get("detections", []), (gpx[0], gpx[1]), source)
    if tpx is None:
        _log("WS-1: no mask hit at the fixation; using raw gaze")
        return gaze_uv, "gaze"
    uv_obj = gaze_point_in_plane_uv(tpx[0], tpx[1], K, tcw,
                                    world_map["plane_point"], world_map["plane_normal"])
    if uv_obj is None:
        _log("WS-1: object pixel ray missed the plane; using raw gaze")
        return gaze_uv, "gaze"
    return uv_obj, source


# ── gaze sampling (median over a short window) ───────────────────────────────


def _sample_uv(consumer: RelayConsumer, detector, K, world_map: dict,
               tag_size: float, dur_s: float) -> Tuple[Optional[np.ndarray], dict]:
    """Median fixated table-plane ``(u,v)`` over a short window, using only frames
    where gaze is valid and ≥1 mapped world tag is seen. The world pose is fused
    from whichever mapped tags are visible (occlusion-robust). Returns
    ``(uv_or_None, diag)`` where ``diag`` reports world-tag visibility — how many of
    the mapped world tags were detected per frame and which — so the operator can see
    the head-invariant anchor is solid (more tags → a more stable, flip-proof pose).

    ``diag`` also carries ``T_cam_world`` (the last frame's recovered pose) and
    ``gaze_px`` (the median gaze pixel) so WS-1 can recast an object-derived target
    pixel through THIS fixation's pose without a second capture window."""
    pts: List[np.ndarray] = []
    gaze_pts: List[np.ndarray] = []
    last_tcw: Optional[np.ndarray] = None
    last_idx = None
    gaze_frames = 0
    per_frame_counts: List[int] = []
    ids_seen: dict = {}
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
        gaze_frames += 1
        tags = detect_tags(detector, b.video.bgr, K, tag_size)
        world_view = {i: tags[i] for i in world_map["ids"] if i in tags}
        per_frame_counts.append(len(world_view))
        for i in world_view:
            ids_seen[int(i)] = ids_seen.get(int(i), 0) + 1
        # Same view-robust board PnP the sweep used (consistency: capture and runtime
        # must recover the world pose identically). Per-tag consensus is the <4-tag
        # occlusion fallback.
        T_cam_world = recover_world_pose_pnp(world_view, world_map, K)
        if T_cam_world is None:
            T_cam_world = recover_world_pose(
                {i: world_view[i]["T"] for i in world_view}, world_map)
        if T_cam_world is None:
            continue
        uv = gaze_point_in_plane_uv(b.gaze.x, b.gaze.y, K, T_cam_world,
                                    world_map["plane_point"], world_map["plane_normal"])
        if uv is not None:
            pts.append(uv)
            gaze_pts.append(np.array([b.gaze.x, b.gaze.y], dtype=float))
            last_tcw = T_cam_world
    diag = {
        "frames": len(pts),
        "gaze_frames": gaze_frames,
        "median_world_tags": float(np.median(per_frame_counts)) if per_frame_counts else 0.0,
        "tags_seen": sorted(ids_seen),
        "mapped_tags": sorted(int(i) for i in world_map["ids"]),
        "T_cam_world": last_tcw,
        "gaze_px": (np.median(np.vstack(gaze_pts), axis=0) if gaze_pts else None),
    }
    if not pts:
        return None, diag
    return np.median(np.vstack(pts), axis=0), diag


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
    # REV04: the command path is the planar (u,v)→Q library. A REV03 rigid calib
    # (T_base_world) is no longer accepted — its EE-hover poses and a table fixation
    # live at different heights (verification report §5), the failure REV04 removes.
    if "UV" not in z.files:
        if "T_base_world" in z.files:
            _log(f"{args.calib} is a REV03 RIGID calibration (T_base_world). The "
                 "REV04 control test uses the planar (u,v) chain — re-solve a sweep "
                 "npz so the solve writes scheme='planar_uv_nn' (UV/Q).")
        else:
            _log(f"{args.calib} is not a planar AprilTag calibration (missing UV); "
                 "select an apriltag_*_calib.npz from the planar solve stage")
        return 2
    # The world map (registered during the sweep) lets any visible subset of world
    # tags recover the SAME world frame — occlusion-robust. Tag size defaults from
    # meta so the panel button needs only --calib.
    wm_keys = ("world_map_ref", "world_map_ids", "world_map_rels",
               "world_map_plane_point", "world_map_plane_normal")
    if not all(k in z.files for k in wm_keys):
        _log(f"{args.calib} has no world map — re-run the calibration (the sweep "
             "registers a multi-tag world map)")
        return 2
    world_map = world_map_from_arrays(z["world_map_ref"], z["world_map_ids"],
                                      z["world_map_rels"], z["world_map_plane_point"],
                                      z["world_map_plane_normal"])
    try:
        mapping = GazeCalibrationMappingV3(z)
    except (KeyError, ValueError) as exc:
        _log(f"{args.calib} planar library is unusable: {exc}")
        return 2
    meta = z["meta"].item() if "meta" in z.files else {}
    tag_size = (args.tag_size if args.tag_size is not None
                else float(meta.get("tag_size_m", 0.06)))
    detector = load_detector(args.families)
    K = consumer.camera_matrix
    _log(f"calibration: {mapping.num_valid_samples} planar (u,v)→Q poses. Workspace "
         f"clamp from Q±{WORKSPACE_BOUNDS_MARGIN:.0%}. Robot dur={args.dur:.1f}s.")
    _log("Per move: fixate a calibrated target, Enter to RESOLVE; review; then "
         "'g' to GO ('g' again to confirm a far fixation), 'r' to re-resolve, "
         "'h' to home, 'q' to quit. NO autonomous motion.")

    # WS-1: optionally aim at the fixated object's centroid / footprint instead of
    # the raw gaze∩plane point. 'gaze' (default) is the frozen REV05 path and never
    # touches the service. Other modes ask vlm_service.segment for masks and fall
    # back to raw gaze on any miss (logged) — so the worst case is REV05 behaviour.
    vlm: Optional[VLMClient] = None
    if args.target_source != "gaze":
        import config as cfg
        vlm = VLMClient(cfg)
        _log(f"WS-1 target source = {args.target_source!r} (FastSAM via vlm_service); "
             "falls back to raw gaze on any miss")

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
        uv, diag = _sample_uv(consumer, detector, K, world_map, tag_size, args.sample_s)
        if uv is None:
            _log(f"no valid gaze+world-tag samples (gaze frames={diag['gaze_frames']}, "
                 f"world tags seen={diag['tags_seen']} of {diag['mapped_tags']}) — "
                 "keep ≥1 world tag in view and fixate")
            pending, far_armed = None, False
            continue
        uv_src = "gaze"
        if vlm is not None:
            uv, uv_src = _object_target_uv(vlm, args.target_source, uv, diag, K, world_map)
        _log(f"WS-1 target source: {uv_src}")
        result = mapping.query_uv(uv)
        idx, dist = result.idx, result.dist
        q_cmd, clamped = result.q_target, result.clamped
        _log(f"world anchor: {diag['frames']} frames, median "
             f"{diag['median_world_tags']:.0f}/{len(diag['mapped_tags'])} world tags/frame, "
             f"saw {diag['tags_seen']}")
        _log(f"fixated table (u,v) = {np.round(uv,1)} mm")
        _log(f"nearest calibrated pose #{idx}: library (u,v)={np.round(result.x_target,1)} "
             f"mm, dist={dist:.1f} mm, clamped={clamped}")
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
    p.add_argument("--target-source", choices=["gaze", "centroid", "bottom"], default=None,
                   help="WS-1: what to aim at. 'gaze' (REV05 default) = raw gaze∩plane; "
                        "'centroid'/'bottom' = the fixated object's mask centroid / "
                        "footprint (FastSAM via vlm_service), falling back to gaze on a "
                        "miss. Default: config.APRILTAG_WS1_TARGET_SOURCE.")
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
    args.target_source = (args.target_source
                          or getattr(cfg, "APRILTAG_WS1_TARGET_SOURCE", "gaze"))
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
