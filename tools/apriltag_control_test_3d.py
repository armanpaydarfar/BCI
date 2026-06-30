#!/usr/bin/env python3
"""
apriltag_control_test_3d.py — drive the Harmony robot from gaze in FULL 3-D using
a WS-4 ``world_xyz_nn`` AprilTag calibration. The 3-D sibling of
``tools/apriltag_control_test.py`` (REV04 planar); standalone validation of the
gaze↔robot 3-D mapping, NOT the EEG-gated experiment driver.

Where the 2-D tool intersects the gaze ray with the table plane and maps the
resulting table ``(u,v)`` to a joint pose, this tool takes the 3-D target from
perception: vlm_service segments the fixated object, lifts its centre to 3-D with
Depth Pro, and returns the point in the scene-camera frame. We transform that
point into the calibration WORLD frame and look it up against the 3-D library
(``GazeCalibration3D``, EE positions in the world frame → recorded joint vectors).
No table plane is involved — the query keeps its full 3-D height.

Per fixation:

    recover T_cam_world (board PnP over the world tags, same as the 2-D tool) →
      vlm.waypoints() → hit_waypoint.position_cam (cam frame, METRES) →
      ×1000 to mm → world frame via invert(T_cam_world) → P_world (mm) →
      nearest calibrated world point (GazeCalibration3D) → joint vector Q[idx] →
      workspace clamp → command the robot.

The robot accepts only joint angles (verified in reports/cpp.md), so the command
is the calibrated, known-safe Q[idx] — no inverse kinematics. **Tier-1: this
commands motion.** Safety model is COPIED from the 2-D tool, not loosened:

  - Operator-gated SINGLE moves: every move shows the target, the chosen joint
    pose, the nearest-neighbour distance, and the clamp state, then waits for an
    explicit confirm. There is no autonomous loop.
  - Workspace clamp: Q is clipped to the calibration envelope (±5% margin,
    matching Utils/gaze/calibration_mapping.WORKSPACE_BOUNDS_MARGIN). The robot
    enforces NO bounds (reports/cpp.md §7.2), so this is the only guard.
  - A far-fixation gate: if the nearest calibrated pose is farther than
    --max-nn-dist-mm from the resolved 3-D point, the move requires a SECOND 'g'
    (the target is outside the calibrated region).
  - This tool is ALWAYS object-3-D. There is no gaze-plane fallback target here —
    a 3-D point needs depth, and a plane projection would be a different (2-D)
    answer. If waypoints is unavailable, has no depth, or finds no object at the
    fixation, the tool REFUSES to move (Tier-1 fail-safe) rather than fall open.
  - 'h' homes the arm; Ctrl-C / 'q' quits.

WS-4 KNOWN LIMITS (documented, not fixed here — these bound when a resolve is
trustworthy):
  - Cross-frame head motion: ``position_cam`` is in the SERVICE's camera frame at
    the SERVICE's frame instant, while ``T_cam_world`` is recovered from THIS
    tool's own frame window. The two only compose correctly if the head is ~static
    between them — i.e. during a genuine fixation. If the head moves, the object
    point is lifted through a stale pose and the world point is wrong. Resolve
    while holding still; re-resolve if you moved.
  - Monocular depth scale: Depth Pro is monocular, so its metric scale carries an
    error that propagates directly into the 3-D world point. The 2-D path hides
    this by snapping to the table plane (the plane absorbs the range error); the
    3-D path has no plane, so a depth-scale bias shifts the target along the line
    of sight. Trust the far-NN gate and the readback.

The relay (sole Neon subscriber) must be up. Binds the robot control endpoint,
so it must NOT run alongside the recorder/online driver (fails fast EADDRINUSE).
vlm_service must be running WITH depth enabled (--enable-depth); without it the
tool cannot resolve a 3-D target and refuses every move.
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
from Utils.gaze.control_view import ControlView  # noqa: E402
from Utils.gaze.calibration_mapping import WORKSPACE_BOUNDS_MARGIN  # noqa: E402
from Utils.gaze.calibration_mapping_3d import GazeCalibration3D  # noqa: E402
from Utils.gaze.harmony_link import HarmonyLink  # noqa: E402
from Utils.gaze.object_target import (  # noqa: E402
    GAZE_DIVERGENCE_TOL_PX,
    gaze_divergence_ok,
    height_on_vertical,
    pixel_on_plane_world,
    select_object_pixel,
    select_vertical_line,
    table_plane_from_map,
    triangulate_vertical_line,
)
from Utils.perception_clients import VLMClient  # noqa: E402


def _log(msg: str) -> None:
    print(f"[apriltag_control_3d] {msg}", flush=True)


# ── pure helper (hardware-free, unit-tested) ──────────────────────────────────


def object_point_world_mm(position_cam_m, T_cam_world: np.ndarray) -> np.ndarray:
    """Lift a perception waypoint into the calibration WORLD frame (mm).

    ``position_cam_m`` is the fixated object's 3-D centre in the SCENE-CAMERA frame
    in **METRES** (Depth Pro output, vlm_service ``hit_waypoint['position_cam']``).
    The AprilTag calibration chain is in **MILLIMETRES** (``apriltag_calib.py:18``),
    so we scale ×1000 before composing. ``T_cam_world`` maps WORLD→CAM (the world
    tag's pose in the camera frame, the same object the 2-D tool recovers); its
    inverse maps CAM→WORLD, so the world point is
    ``transform_point(invert(T_cam_world), p_cam_mm)`` — the 2-D tool's cam→world
    step (``apriltag_control_test.gaze_point_in_plane_uv``) WITHOUT the table-plane
    projection, keeping the full 3-D height.

    float64 throughout (the result feeds a joint command). Raises ValueError on a
    non-finite input so an unusable depth value fails fast at the caller (Tier-1
    fail-safe) rather than producing a garbage target.
    """
    p_cam_m = np.asarray(position_cam_m, dtype=float).reshape(3)
    if not np.all(np.isfinite(p_cam_m)):
        raise ValueError(f"position_cam must be finite; got {p_cam_m!r}")
    p_cam_mm = p_cam_m * 1000.0  # metres (Depth Pro) → mm (apriltag chain)
    return transform_point(invert_transform(T_cam_world), p_cam_mm)


# ── world-pose recovery (over a short window) ─────────────────────────────────


def _make_frame_undistorter(K, dist_coeffs):
    """``bgr -> rectified bgr`` (cv2 remap, SAME K), or None if no usable distortion.
    The Neon ships raw wide-FOV frames; tag pose/PnP assume a pinhole, so frames must
    be undistorted before detection (2026-06-30 root cause). Lazy maps."""
    if dist_coeffs is None:
        return None
    dist = np.asarray(dist_coeffs, float).ravel()
    if not np.any(np.abs(dist) > 1e-9):
        return None
    import cv2
    Kf = np.asarray(K, float)
    maps = [None, None]

    def _u(bgr):
        if maps[0] is None:
            h, w = bgr.shape[:2]
            maps[0], maps[1] = cv2.initUndistortRectifyMap(Kf, dist, None, Kf, (w, h), cv2.CV_32FC1)
        return cv2.remap(bgr, maps[0], maps[1], cv2.INTER_LINEAR)

    return _u


def _undistort_px(xy, K, dist):
    """Map a RAW pixel (a VLM mask/gaze pixel, localized on a raw frame) into the
    rectified pinhole frame, so it can be cast through the tag-recovered (rectified)
    pose with the same K. No-op if dist is None. ``xy`` = (x, y) or None."""
    if xy is None or dist is None:
        return xy
    import cv2
    a = np.asarray(xy, float).reshape(1, 1, 2)
    u = cv2.undistortPoints(a, np.asarray(K, float), np.asarray(dist, float),
                            P=np.asarray(K, float))
    return (float(u[0, 0, 0]), float(u[0, 0, 1]))


def _project_world_px(p_world, K, T_cam_world, dist):
    """Project a world point to the RAW (distorted) scene pixel — so the viz marker
    lands correctly on the raw display frame the operator sees. Pinhole if dist None."""
    import cv2
    Tcw = np.asarray(T_cam_world, float)
    rvec, _ = cv2.Rodrigues(Tcw[:3, :3])
    d = np.asarray(dist, float) if dist is not None else np.zeros(5)
    px, _ = cv2.projectPoints(np.asarray(p_world, float).reshape(1, 3), rvec,
                              Tcw[:3, 3], np.asarray(K, float), d)
    return (float(px[0, 0, 0]), float(px[0, 0, 1]))


def _recover_world_pose(consumer: RelayConsumer, detector, K, world_map: dict,
                        tag_size: float, dur_s: float, *,
                        ee_tag_id: Optional[int] = None, ee_tag_size: Optional[float] = None,
                        dist_coeffs=None) -> Tuple[Optional[np.ndarray], dict]:
    """Recover ``T_cam_world`` over a short window, keeping the LAST frame whose
    pose solved, using the SAME view-robust board PnP the sweep/2-D tool use
    (per-tag consensus is the <4-tag occlusion fallback). Unlike the 2-D tool's
    ``_sample_uv`` this needs no gaze and no table-plane projection — the 3-D
    target comes from perception; here we only anchor the world frame.

    Returns ``(T_cam_world_or_None, diag)`` where ``diag`` reports world-tag
    visibility (how many of the mapped world tags were seen, and which) so the
    operator can confirm the head-invariant anchor is solid before committing a
    move (more tags → a more stable, flip-proof pose)."""
    last_tcw: Optional[np.ndarray] = None
    last_gaze: Optional[Tuple[float, float]] = None  # our-frame gaze at the solved frame
    last_ee_world: Optional[np.ndarray] = None       # EE tag position in world (robot-state viz)
    last_idx = None
    frames_with_pose = 0
    per_frame_counts: List[int] = []
    ids_seen: dict = {}
    # Size the EE tag correctly if we're also locating it (it's usually smaller than
    # the world tags), else its pose comes out mis-scaled.
    tag_sizes = ({int(ee_tag_id): float(ee_tag_size)}
                 if ee_tag_id is not None and ee_tag_size else None)
    undist = _make_frame_undistorter(K, dist_coeffs)
    deadline = time.time() + dur_s
    while time.time() < deadline:
        b = consumer.latest()
        if b is None or b.video is None or b.video.bgr is None or b.video.frame_idx == last_idx:
            time.sleep(0.005)
            continue
        last_idx = b.video.frame_idx
        if not getattr(b, "worn", True):
            continue
        bgr = undist(b.video.bgr) if undist is not None else b.video.bgr
        tags = detect_tags(detector, bgr, K, tag_size, tag_sizes=tag_sizes)
        world_view = {i: tags[i] for i in world_map["ids"] if i in tags}
        per_frame_counts.append(len(world_view))
        for i in world_view:
            ids_seen[int(i)] = ids_seen.get(int(i), 0) + 1
        T_cam_world = recover_world_pose_pnp(world_view, world_map, K)
        if T_cam_world is None:
            T_cam_world = recover_world_pose(
                {i: world_view[i]["T"] for i in world_view}, world_map)
        if T_cam_world is None:
            continue
        frames_with_pose += 1
        last_tcw = T_cam_world
        # Capture OUR-frame gaze on the solved frame, for the object-plane path's
        # cross-frame guard (service gaze vs ours). Absent/non-finite → leave None.
        g = getattr(b, "gaze", None)
        if g is not None and np.isfinite(getattr(g, "x", np.nan)) and np.isfinite(getattr(g, "y", np.nan)):
            last_gaze = (float(g.x), float(g.y))
        # Current EE position in world (from the EE tag, if visible) for the viz.
        if ee_tag_id is not None and int(ee_tag_id) in tags:
            ee_cam = np.asarray(tags[int(ee_tag_id)]["T"], dtype=float)
            last_ee_world = transform_point(invert_transform(T_cam_world), ee_cam[:3, 3])
    diag = {
        "frames": frames_with_pose,
        "median_world_tags": float(np.median(per_frame_counts)) if per_frame_counts else 0.0,
        "tags_seen": sorted(ids_seen),
        "mapped_tags": sorted(int(i) for i in world_map["ids"]),
        "gaze_px": last_gaze,
        "ee_tag_world": last_ee_world,
    }
    return last_tcw, diag


# ── main control loop ─────────────────────────────────────────────────────────


def _commit_move(link: HarmonyLink, q_cmd: np.ndarray, idx: int, dur_s: float) -> None:
    """Command one move and ALWAYS read back the actual pose. ``send_joint_command``
    distinguishes 'stage_failed' (arm did not move) from 'go_unconfirmed' (coords
    staged but no ACK:g — the arm MAY be moving), so a lost go-ACK is never
    reported as a clean failure (the fail-open hazard).

    Copied verbatim from the 2-D tool (apriltag_control_test._commit_move): the
    staged-trajectory + readback semantics are the Tier-1 contract and must not
    diverge between the two tools."""
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


def _fmt_px(p) -> str:
    if p is None:
        return "n/a"
    try:
        return f"({float(p[0]):.0f},{float(p[1]):.0f})"
    except (TypeError, ValueError):
        return "n/a"


def _px_dist(a, b) -> float:
    try:
        return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))
    except (TypeError, ValueError):
        return float("nan")


def _resolve_object_once(vlm, args, K, T_cam_world, diag, table_point, table_normal,
                         dist_coeffs=None, *, quiet=False):
    """ONE segmentation → depth-free target. Returns ``(p_world, viz, reason)``:
    ``reason`` is None on success, else a short tag for the windowed caller's summary.
    ``quiet`` suppresses per-frame logs (the window logs a summary instead)."""
    def log(m):
        if not quiet:
            _log(m)
    try:
        seg = vlm.segment(include_masks=True)
    except OSError as exc:
        log(f"segment request failed ({exc!r}); is vlm_service up? NOT moving.")
        return None, None, "segment_failed"
    if not (isinstance(seg, dict) and seg.get("ok")):
        log(f"segment unavailable ({(seg or {}).get('error')}); NOT moving.")
        return None, None, "segment_unavailable"
    dets = seg.get("detections", [])
    svc_gaze, our_gaze = seg.get("gaze_px"), diag.get("gaze_px")
    if not gaze_divergence_ok(svc_gaze, our_gaze):
        if svc_gaze is None or our_gaze is None:
            log(f"segment: {len(dets)} masks, but NO live gaze captured "
                f"(service={_fmt_px(svc_gaze)}, ours={_fmt_px(our_gaze)}) — is the Neon "
                "gaze stream flowing through the relay? NOT moving.")
            return None, None, "no_gaze"
        log(f"segment: {len(dets)} masks, service gaze={_fmt_px(svc_gaze)} vs ours "
            f"{_fmt_px(our_gaze)} diverged >{GAZE_DIVERGENCE_TOL_PX:.0f}px (head moved "
            "during resolve). NOT moving — hold still and re-resolve.")
        return None, None, "gaze_diverged"
    frame_area = 4.0 * float(K[0, 2]) * float(K[1, 2])
    base = None
    if args.target_point == "gaze_height":
        fpx, fpy, sel = select_object_pixel(dets, svc_gaze, "footprint",
                                            frame_area_px=frame_area,
                                            max_area_frac=args.max_object_area_frac)
        if fpx is None:
            log(f"segment: {sel['n_dets']} masks, none under gaze (n_contained="
                f"{sel['n_contained']}, {sel.get('rejected_large',0)} table-sized, "
                f"{sel.get('rejected_noise',0)} noise dropped); no object. NOT moving.")
            return None, None, "no_object"
        # Undistort the VLM pixels (raw-frame coords) before casting through the
        # rectified tag pose — else lens distortion (tens of px at the edges) corrupts
        # the ray, the same root cause that broke the map.
        ufp = _undistort_px((fpx, fpy), K, dist_coeffs)
        base = pixel_on_plane_world(ufp[0], ufp[1], K, T_cam_world, table_point, table_normal)
        if base is None:
            log("footprint ray missed the table plane (parallel/behind). NOT moving.")
            return None, None, "ray_missed"
        p_world = height_on_vertical(_undistort_px(svc_gaze, K, dist_coeffs), base,
                                     table_normal, K, T_cam_world)
        if p_world is None:
            log("gaze-height ray unusable; NOT moving.")
            return None, None, "height_unusable"
        n = np.asarray(table_normal, float) / max(np.linalg.norm(table_normal), 1e-9)
        h_mm = float((np.asarray(p_world) - base) @ n)
        log(f"object: {sel['pick']} mask, area={sel.get('mask_area_px', 0):.0f}px → "
            f"footprint=({fpx:.0f},{fpy:.0f}), gaze height={h_mm:.0f} mm above table  "
            f"[{len(dets)} masks, gaze div={_px_dist(svc_gaze, our_gaze):.0f}px]")
    else:
        px, py, sel = select_object_pixel(dets, svc_gaze, args.target_point,
                                          frame_area_px=frame_area,
                                          max_area_frac=args.max_object_area_frac)
        if px is None:
            log(f"segment: {sel['n_dets']} masks, none under gaze (n_contained="
                f"{sel['n_contained']}, {sel.get('rejected_large',0)} table-sized, "
                f"{sel.get('rejected_noise',0)} noise dropped); no object. NOT moving.")
            return None, None, "no_object"
        up = _undistort_px((px, py), K, dist_coeffs)
        p_world = pixel_on_plane_world(up[0], up[1], K, T_cam_world, table_point, table_normal)
        if p_world is None:
            log("object pixel ray missed the table plane (parallel/behind). NOT moving.")
            return None, None, "ray_missed"
        log(f"object: {sel['pick']} mask, area={sel.get('mask_area_px', 0):.0f}px → "
            f"{args.target_point} pixel=({px:.0f},{py:.0f})  "
            f"[{len(dets)} masks, gaze div={_px_dist(svc_gaze, our_gaze):.0f}px]")
    tgt_px = _project_world_px(p_world, K, T_cam_world, dist_coeffs)
    base_world = base if args.target_point == "gaze_height" else p_world
    viz = {"gaze_px": svc_gaze, "mask_polygon": sel.get("mask_polygon"),
           "target_px": tgt_px, "base_world": base_world}
    return np.asarray(p_world, float), viz, None


def _resolve_object_plane(vlm, args, K, T_cam_world, diag,
                          table_point, table_normal, dist_coeffs=None):
    """Depth-free target, robust to a single bad segmentation frame. Calls segment
    ``--resolve-samples`` times over a short window and returns the per-component
    MEDIAN target — so a transient line/partial/cap-only mask is an outlier the median
    discards, not the whole answer (operator idea, 2026-06-30). N=1 keeps the original
    single-shot behaviour + per-frame logs. Returns ``(p_world, viz)`` or
    ``(None, None)`` on a logged miss."""
    n = max(1, int(args.resolve_samples))
    pts: List[np.ndarray] = []
    vizs: List[dict] = []
    reasons: List[str] = []
    for i in range(n):
        p, viz, reason = _resolve_object_once(
            vlm, args, K, T_cam_world, diag, table_point, table_normal,
            dist_coeffs, quiet=(n > 1))
        if p is not None:
            pts.append(p); vizs.append(viz)
        elif reason is not None:
            reasons.append(reason)
        if n > 1 and i < n - 1:
            time.sleep(0.03)
    if n == 1:
        return (pts[0], vizs[0]) if pts else (None, None)
    if len(pts) < (n + 1) // 2:                          # need a majority to agree
        from collections import Counter
        why = Counter(reasons).most_common(1)
        _log(f"object: only {len(pts)}/{n} segmentation samples found the object"
             + (f" (mostly: {why[0][0]})" if why else "")
             + " — unstable, NOT moving. Re-resolve.")
        return None, None
    arr = np.asarray(pts, float)
    med = np.median(arr, axis=0)
    spread = float(np.median(np.linalg.norm(arr - med, axis=1)))   # robust spread (MAD)
    if spread > args.resolve_spread_mm:
        _log(f"object: {len(pts)}/{n} samples disagree by {spread:.0f} mm "
             f"(> {args.resolve_spread_mm:.0f}) — segmentation unstable. NOT moving.")
        return None, None
    best = int(np.argmin(np.linalg.norm(arr - med, axis=1)))       # viz nearest the median
    _log(f"object: median of {len(pts)}/{n} samples, spread {spread:.0f} mm "
         "(transient bad frames discarded)")
    return med, vizs[best]


def _resolve_depth(vlm, T_cam_world):
    """Legacy Depth Pro target: vlm_service.waypoints → hit_waypoint.position_cam →
    world. Kept as an explicit fallback (--target-source depth). Returns
    ``(p_world, None)`` (no viz for the legacy path) or ``(None, None)``."""
    try:
        wp = vlm.waypoints()
    except OSError as exc:
        _log(f"waypoints request failed ({exc!r}); is vlm_service up? NOT moving.")
        return None, None
    if not (isinstance(wp, dict) and wp.get("ok")):
        _log(f"waypoints unavailable ({(wp or {}).get('error')}); NOT moving.")
        return None, None
    if wp.get("depth_enabled") is False:
        _log("vlm_service has NO depth (started without --enable-depth). NOT moving.")
        return None, None
    hit = wp.get("hit_waypoint")
    if hit is None:
        _log("no object hit at the fixation. NOT moving.")
        return None, None
    _log(f"object hit: {hit.get('label')!r} @ depth {hit.get('depth_median_m')} m, "
         f"position_cam={np.round(np.asarray(hit['position_cam'], float), 3).tolist()} m")
    return object_point_world_mm(hit["position_cam"], T_cam_world), None


def _read_command(ui) -> str:
    """Read the operator's command. With a ControlView, read it FROM the window
    (cv2 keys) so the terminal stays log-only and the window keeps focus; otherwise
    fall back to terminal input(). Returns '' (resolve), 'g', 'r', 'h', or 'q'."""
    if ui is None:
        return input("> ").strip().lower()
    while True:
        k = ui.poll_key(60)
        if k in (ord("q"), 27):              # q or ESC → quit
            return "q"
        if k == ord("h"):
            return "h"
        if k == ord("g"):
            return "g"
        if k == ord("r"):
            return "r"
        if k in (13, 10):                    # ENTER only → resolve (space removed: it
            return ""                        # was firing accidental resolves after GO)


# ── registration-based target (gaze_lines): no segmentation ───────────────────

_FOOTPRINT_RESID_MAX_MM = 60.0     # triangulation residual above which a capture is too unstable
_MIN_REG_RAYS = 12                 # gaze rays (head motion) needed for a footprint


def _read_register_key(ui, msg: str) -> str:
    """Prompt + read ENTER / d / q (window keys with a ui, else terminal)."""
    _log(msg)
    if ui is None:
        c = input("> ").strip().lower()
        return c if c in ("d", "q") else ""
    while True:
        k = ui.poll_key(60)
        if k in (ord("q"), 27):
            return "q"
        if k == ord("d"):
            return "d"
        if k in (13, 10):
            return ""


def _register_footprints(consumer, detector, K, world_map, tag_size, table_point,
                         table_normal, dist_coeffs, ui):
    """Register each graspable object's vertical-line FOOTPRINT by triangulation: the
    operator looks at the object and MOVES THEIR HEAD; gaze rays from different views
    converge on its table XY (never aiming at the occluded base). Returns the list of
    footprint world points, or None if aborted."""
    undist = _make_frame_undistorter(K, dist_coeffs)
    foots: List[np.ndarray] = []
    while True:
        cmd = _read_register_key(
            ui, f"REGISTER object #{len(foots) + 1}: look at it and MOVE YOUR HEAD "
                "around it; ENTER = capture (~4s), d = done, q = abort.")
        if cmd == "q":
            return None
        if cmd == "d":
            return foots
        _log("  capturing ~4s — keep looking at the object and moving your head …")
        origins, dirs = [], []
        last_idx = None
        deadline = time.time() + 4.0
        while time.time() < deadline:
            b = consumer.latest()
            if (b is None or b.video is None or b.video.bgr is None
                    or b.video.frame_idx == last_idx):
                time.sleep(0.005)
                continue
            last_idx = b.video.frame_idx
            bgr = undist(b.video.bgr) if undist is not None else b.video.bgr
            tags = detect_tags(detector, bgr, K, tag_size)
            wv = {i: tags[i] for i in world_map["ids"] if i in tags}
            Tcw = recover_world_pose_pnp(wv, world_map, K)
            if Tcw is None:
                Tcw = recover_world_pose({i: wv[i]["T"] for i in wv}, world_map)
            if Tcw is None:
                continue
            g = getattr(b, "gaze", None)
            if g is None or not (np.isfinite(getattr(g, "x", np.nan))
                                 and np.isfinite(getattr(g, "y", np.nan))):
                continue
            ug = _undistort_px((float(g.x), float(g.y)), K, dist_coeffs)
            rc = gaze_ray_cam(ug[0], ug[1], K)
            if rc is None:
                continue
            Twc = invert_transform(Tcw)
            origins.append(Twc[:3, 3]); dirs.append(Twc[:3, :3] @ rc)
        f, resid = triangulate_vertical_line(origins, dirs, table_point, table_normal)
        if f is None or len(origins) < _MIN_REG_RAYS or resid > _FOOTPRINT_RESID_MAX_MM:
            _log(f"  poor capture ({len(origins)} rays, resid {resid:.0f} mm > "
                 f"{_FOOTPRINT_RESID_MAX_MM:.0f}) — MOVE YOUR HEAD MORE while looking "
                 "steadily. Not stored; retry.")
            continue
        foots.append(np.asarray(f, dtype=float))
        _log(f"  OK object #{len(foots)} footprint = {np.round(f, 0).tolist()} mm "
             f"(resid {resid:.0f} mm, {len(origins)} rays)")


def _load_or_register_footprints(args, consumer, detector, K, world_map, tag_size,
                                 table_point, table_normal, dist_coeffs, ui):
    path = args.footprints_file
    if path and Path(path).is_file():
        foots = [np.asarray(f, float)
                 for f in np.atleast_2d(np.load(path)["footprints"])]
        _log(f"loaded {len(foots)} object footprints from {path}")
        return foots
    foots = _register_footprints(consumer, detector, K, world_map, tag_size,
                                 table_point, table_normal, dist_coeffs, ui)
    if foots and path:
        np.savez(path, footprints=np.asarray(foots, float))
        _log(f"saved {len(foots)} footprints -> {path} (reuse with --footprints-file)")
    return foots


def _resolve_gaze_lines(diag, footprints, table_point, table_normal, K, T_cam_world,
                        args, dist_coeffs=None):
    """Registration-based target: the gaze ray's closest registered vertical line ->
    the target at the gaze height on it. No segmentation. Returns (p_world, viz)."""
    gaze = diag.get("gaze_px")
    if gaze is None:
        _log("no live gaze captured — is the Neon gaze stream flowing? NOT moving.")
        return None, None
    ug = _undistort_px(gaze, K, dist_coeffs)
    target, info = select_vertical_line(ug, footprints, table_normal, K, T_cam_world,
                                        max_dist_mm=args.line_max_dist_mm)
    if target is None:
        if info.get("reason") == "too_far":
            _log(f"gaze not on a registered object (nearest line {info['nearest_dist']:.0f} "
                 f"mm > {args.line_max_dist_mm:.0f} mm). NOT moving.")
        else:
            _log("gaze ray unusable for line selection. NOT moving.")
        return None, None
    foot = np.asarray(footprints[info["idx"]], float)
    n = np.asarray(table_normal, float) / max(np.linalg.norm(table_normal), 1e-9)
    h_mm = float((np.asarray(target) - foot) @ n)
    _log(f"gaze_lines: object #{info['idx']} (gaze {info['dist']:.0f} mm from its line), "
         f"height {h_mm:.0f} mm above table")
    tgt_px = _project_world_px(target, K, T_cam_world, dist_coeffs)
    viz = {"gaze_px": gaze, "mask_polygon": None, "target_px": tgt_px, "base_world": foot}
    return np.asarray(target, float), viz


def run(args, consumer: RelayConsumer, link: HarmonyLink) -> int:
    z = np.load(args.calib, allow_pickle=True)
    meta = z["meta"].item() if "meta" in z.files else {}
    # WS-4: the command path is the 3-D world-frame (x,y,z)→Q library. Only a
    # solve-3d calib (scheme="world_xyz_nn", P_WORLD3D/Q) carries the full-height
    # library this tool queries — a planar (REV04) or rigid (REV03) npz would map
    # the wrong point space, so refuse it explicitly rather than mis-index.
    scheme = meta.get("scheme")
    if scheme != "world_xyz_nn":
        _log(f"{args.calib} has scheme={scheme!r}, not 'world_xyz_nn'. The 3-D "
             "control test needs a solve-3d calibration (P_WORLD3D/Q) — run "
             "tools/apriltag_calibrate.py --stage solve-3d on a sweep npz.")
        return 2
    # The world map (registered during the sweep, carried through solve-3d) lets
    # any visible subset of world tags recover the SAME world frame the library
    # points live in — occlusion-robust, and the frame the 3-D target is mapped to.
    wm_keys = ("world_map_ref", "world_map_ids", "world_map_rels",
               "world_map_plane_point", "world_map_plane_normal")
    if not all(k in z.files for k in wm_keys):
        _log(f"{args.calib} has no world map — re-run the calibration (the sweep "
             "registers a multi-tag world map; solve-3d carries it through)")
        return 2
    world_map = world_map_from_arrays(z["world_map_ref"], z["world_map_ids"],
                                      z["world_map_rels"], z["world_map_plane_point"],
                                      z["world_map_plane_normal"])
    try:
        mapping = GazeCalibration3D.from_calib_npz(z)
    except (KeyError, ValueError) as exc:
        _log(f"{args.calib} 3-D library is unusable: {exc}")
        return 2
    tag_size = (args.tag_size if args.tag_size is not None
                else float(meta.get("tag_size_m", 0.06)))
    detector = load_detector(args.families)
    K = consumer.camera_matrix
    dist_coeffs = consumer.distortion_coeffs
    if dist_coeffs is not None:
        _log("  undistorting frames + VLM pixels before casting (Neon lens) — matches the map")

    import config as cfg
    vlm = VLMClient(cfg)

    # object_plane (default, robust): fit the TABLE plane from the table tags ONCE,
    # up front, so a bad table-tag set / non-coplanar tags fail loud here, not
    # mid-session. The plane supplies depth at runtime (no Depth Pro).
    table_point = table_normal = None
    if args.target_source == "object_plane":
        try:
            table_point, table_normal, tinfo = table_plane_from_map(
                world_map, args.table_tag_ids)
        except ValueError as exc:
            _log(f"cannot fit the table plane: {exc}. Pass --table-tag-ids (table tags "
                 "only), or fall back to --target-source depth.")
            return 2
        _log(f"table plane from tags {tinfo['table_ids']}: out-of-plane residual "
             f"max={tinfo['max_resid_mm']:.1f} mm, rms={tinfo['rms_resid_mm']:.1f} mm "
             "(large ⇒ wrong table-tag set or a bumped tag)")
        # Orient the table normal to point UP. The normal comes from the tags' +Z
        # faces and can point either way; a down-pointing normal silently sends
        # gaze_height the wrong way and clamps EVERY height to 0 (observed at the rig
        # 2026-06-30 — the library z came out negative). "Up" = toward the calibrated
        # library, which was swept ABOVE the table; flip the normal to that side.
        lib_xyz = np.asarray(z["P_WORLD3D"], dtype=float)
        if float((lib_xyz.mean(axis=0) - np.asarray(table_point, float)) @ table_normal) < 0:
            table_normal = -np.asarray(table_normal, float)
            _log("  table normal flipped to point UP (toward the calibrated library) "
                 "— gaze_height now accumulates above the table")

    # Visual interface (object_plane only): project the library to table (u,v) once
    # for the top-down coverage backdrop, so each resolve shows the target landing
    # inside or outside the calibrated region.
    ui: Optional[ControlView] = None
    if args.target_source == "object_plane" and not args.no_ui:
        lib_uv = plane_coords(lib_xyz, table_point, table_normal)
        n_hat = np.asarray(table_normal, float) / max(np.linalg.norm(table_normal), 1e-9)
        lib_z = (lib_xyz - np.asarray(table_point, float)) @ n_hat   # height above table
        ui = ControlView(lib_uv, lib_z)

    # gaze_lines: register each graspable object's vertical line up front (or load a
    # saved set) — no segmentation at runtime.
    footprints = None
    if args.target_point == "gaze_lines":
        footprints = _load_or_register_footprints(
            args, consumer, detector, K, world_map, tag_size, table_point,
            table_normal, dist_coeffs, ui)
        if not footprints:
            _log("gaze_lines: no registered objects — nothing to target. Exiting.")
            if ui is not None:
                ui.close()
            return 1

    _log(f"calibration: {mapping.num_valid_samples} world (x,y,z)→Q poses. Workspace "
         f"clamp from Q±{WORKSPACE_BOUNDS_MARGIN:.0%}. Robot dur={args.dur:.1f}s.")
    if args.target_point == "gaze_lines":
        _log(f"3-D target: {len(footprints)} REGISTERED object lines — the gaze ray "
             "picks the nearest; no segmentation, no VLM.")
    elif args.target_source == "object_plane":
        _log(f"3-D target: DEPTH-FREE — fixated object's {args.target_point} pixel ∩ "
             f"table plane (segment masks from {vlm.host}:{vlm.port}; no Depth Pro).")
    else:
        _log(f"3-D target: Depth Pro position_cam (vlm_service.waypoints "
             f"{vlm.host}:{vlm.port}); depth REQUIRED.")
    _log("Per move (keys IN the control window when shown, else this terminal): "
         "fixate the object, ENTER to RESOLVE; review; then 'g' to GO ('g' again to "
         "confirm a far fixation), 'r' to re-resolve, 'h' to home, 'q' to quit. NO "
         "autonomous motion; refuses to move on any uncertainty.")

    pending: Optional[Tuple[np.ndarray, int, float, bool]] = None  # (q_cmd, idx, dist, clamped)
    far_armed = False  # a far fixation needs a SECOND 'g' to commit (review safety)
    while True:
        cmd = _read_command(ui)
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
        _log(f"── resolve ── fixate the object; anchoring world frame for "
             f"{args.sample_s:.1f}s …")
        T_cam_world, diag = _recover_world_pose(
            consumer, detector, K, world_map, tag_size, args.sample_s,
            ee_tag_id=(args.ee_tag_id if ui is not None else None),
            ee_tag_size=args.ee_tag_size, dist_coeffs=dist_coeffs)
        if T_cam_world is None:
            _log(f"no world pose (world tags seen={diag['tags_seen']} of "
                 f"{diag['mapped_tags']}) — keep ≥1 world tag in view")
            pending, far_armed = None, False
            continue
        _log(f"world anchor: {diag['frames']} frames, median "
             f"{diag['median_world_tags']:.0f}/{len(diag['mapped_tags'])} world tags/frame, "
             f"saw {diag['tags_seen']}, our gaze={_fmt_px(diag.get('gaze_px'))}")

        # Target (Tier-1 fail-safe: any uncertainty → return None → NOT moving).
        if args.target_point == "gaze_lines":
            p_world, viz = _resolve_gaze_lines(diag, footprints, table_point,
                                               table_normal, K, T_cam_world, args, dist_coeffs)
        elif args.target_source == "object_plane":
            p_world, viz = _resolve_object_plane(vlm, args, K, T_cam_world, diag,
                                                 table_point, table_normal, dist_coeffs)
        else:
            p_world, viz = _resolve_depth(vlm, T_cam_world)
        if p_world is None:
            pending, far_armed = None, False
            continue

        try:
            result = mapping.query_xyz(p_world)
        except (KeyError, ValueError) as exc:
            _log(f"resolved target is unusable ({exc}); NOT moving.")
            pending, far_armed = None, False
            continue
        idx, dist = result.idx, result.dist
        q_cmd, clamped = result.q_target, result.clamped
        _log(f"target world point = {np.round(p_world, 1).tolist()} mm")
        _log(f"nearest calibrated pose #{idx}: library xyz={np.round(result.x_target,1).tolist()} "
             f"mm, dist={dist:.1f} mm, clamped={clamped}")
        # Hard refuse a target far beyond the calibrated region: a stray gaze (e.g. the
        # operator glancing at the computer screen to read the result) casts to a point
        # ~1 m away. Refuse outright — no double-g — so it can't be committed and the
        # log isn't cluttered with garbage resolves.
        if dist > args.max_resolve_dist_mm:
            _log(f"REFUSED: target is {dist:.0f} mm from the nearest calibrated pose "
                 f"(> {args.max_resolve_dist_mm:.0f} mm hard limit) — almost certainly a "
                 "mis-fixation (gaze past the object, e.g. the screen). Re-resolve.")
            pending, far_armed = None, False
            continue
        _log(f"joint target (rad) = {np.round(q_cmd,4).tolist()}")
        if dist > args.max_nn_dist_mm:
            _log(f"WARNING: {dist:.0f} mm from the nearest calibrated pose — outside the "
                 "calibrated region (sweep didn't cover here, or the target is wrong).")
        _log("press 'g' to GO, 'r' to re-resolve")
        if ui is not None and viz is not None:
            b = consumer.latest()
            scene = b.video.bgr if (b is not None and b.video is not None) else None
            tgt_uv = plane_coords(np.asarray(p_world, float), table_point, table_normal)
            near_uv = plane_coords(np.asarray(result.x_target, float), table_point, table_normal)
            ee_world = diag.get("ee_tag_world")
            ee_uv = (plane_coords(np.asarray(ee_world, float), table_point, table_normal)
                     if ee_world is not None else None)
            cam_center = invert_transform(T_cam_world)[:3, 3]
            ui.update(
                scene, gaze_px=viz["gaze_px"], mask_polygon=viz["mask_polygon"],
                target_px=viz["target_px"], target_uv=tgt_uv, nearest_uv=near_uv,
                current_ee_uv=ee_uv, base_world=viz.get("base_world"),
                target_world=np.asarray(p_world, float), cam_center_world=cam_center,
                table_normal=table_normal,
                lines=[
                    f"target world {np.round(p_world,0).tolist()} mm   "
                    f"nearest #{idx} dist={dist:.0f} mm   clamped={clamped}",
                    f"anchor {diag['frames']}f median {diag['median_world_tags']:.0f}"
                    f"/{len(diag['mapped_tags'])} tags   EE tag "
                    + ("seen" if ee_world is not None else "not seen") + "   "
                    + ("OUTSIDE calibrated region" if dist > args.max_nn_dist_mm
                       else "in calibrated region"),
                ])
        pending, far_armed = (q_cmd, idx, dist, clamped), False
    if ui is not None:
        ui.close()
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    import config as cfg
    p = argparse.ArgumentParser(
        description="Drive the robot from gaze in 3-D via an AprilTag calibration")
    p.add_argument("--calib", required=True,
                   help="apriltag_3d_*_calib.npz (scheme='world_xyz_nn') from solve-3d")
    p.add_argument("--tag-size", type=float, default=None,
                   help="world tag edge length in METRES (default: from the calibration meta)")
    p.add_argument("--families", default="tag36h11")
    p.add_argument("--dur", type=float, default=5.0, help="robot move duration (s)")
    p.add_argument("--sample-s", type=float, default=0.6,
                   help="world-pose anchoring window (s) — hold the head still")
    p.add_argument("--max-nn-dist-mm", type=float, default=80.0,
                   help="warn/guard if the nearest calibrated pose is farther than this")
    p.add_argument("--max-resolve-dist-mm", type=float, default=300.0,
                   help="HARD refuse a target this far from any calibrated pose — a "
                        "mis-fixation (e.g. gaze drifting to the computer screen lands "
                        "~1m away); refused outright, no double-g override")
    p.add_argument("--resolve-samples", type=int, default=5,
                   help="segment this many times per resolve and take the MEDIAN target, "
                        "so a transient bad frame (a line, a partial/cap-only mask) is an "
                        "outlier the median discards. 1 = single-shot (original behaviour)")
    p.add_argument("--resolve-spread-mm", type=float, default=40.0,
                   help="refuse the resolve if the windowed samples disagree (median "
                        "deviation) by more than this — segmentation too unstable to trust")
    p.add_argument("--target-source", choices=["object_plane", "depth"],
                   default="object_plane",
                   help="how to get the 3-D target. 'object_plane' (default, robust): "
                        "fixated object's footprint/centroid pixel cast onto the TABLE "
                        "plane (depth-free). 'depth': legacy Depth Pro position_cam "
                        "(vlm_service.waypoints).")
    p.add_argument("--target-point",
                   choices=["footprint", "centroid", "gaze_height", "gaze_lines"],
                   default="footprint",
                   help="object_plane target. 'footprint' (default): lowest mask row ∩ "
                        "table (overshoot-free, table height). 'centroid': mask centroid "
                        "∩ table. 'gaze_height': footprint XY but height from the gaze ray "
                        "∩ the object's vertical line — look at a part, move to its height "
                        "(for tall/stacked objects). 'gaze_lines': NO segmentation — "
                        "register each object's vertical line once (look + move head), "
                        "then the gaze ray's nearest registered line gives XY + height "
                        "(robust object-on-object: cup-on-block, cap-on-bottle).")
    p.add_argument("--footprints-file", default=None,
                   help="gaze_lines: load registered object footprints from this .npz "
                        "(skip registration); written here after a fresh registration")
    p.add_argument("--line-max-dist-mm", type=float, default=60.0,
                   help="gaze_lines: refuse if the gaze ray is farther than this from "
                        "every registered object's vertical line (gaze not on an object)")
    p.add_argument("--max-object-area-frac", type=float, default=0.5,
                   help="object_plane: drop any segment larger than this fraction of "
                        "the frame as the TABLE / background (so the table is never "
                        "picked as the target). Default 0.5.")
    p.add_argument("--table-tag-ids", type=int, nargs="+", default=None,
                   help="world tag ids physically ON the table, used to fit the table "
                        "plane (default: config.APRILTAG_WORLD_TAG_IDS). Exclude "
                        "wall/elevated tags here.")
    p.add_argument("--no-ui", action="store_true",
                   help="disable the visual interface (scene + top-down + side views)")
    p.add_argument("--ee-tag-id", type=int, default=5,
                   help="EE tag id to locate for the robot-state overlay (current EE on "
                        "the views). Default 5; detection is best-effort (skipped if unseen).")
    p.add_argument("--ee-tag-size", type=float, default=0.04,
                   help="EE tag edge length in METRES (default 0.04) so its pose is scaled "
                        "right when located for the overlay")
    p.add_argument("--side", default=None, help="robot active side R/L")
    p.add_argument("--relay-host", default=None)
    p.add_argument("--relay-port", type=int, default=None)
    p.add_argument("--robot-ip", default=None)
    p.add_argument("--robot-port", type=int, default=None)
    p.add_argument("--bind-ip", default=None)
    p.add_argument("--bind-port", type=int, default=None)
    args = p.parse_args(argv)

    if args.table_tag_ids is None:
        args.table_tag_ids = list(getattr(cfg, "APRILTAG_WORLD_TAG_IDS", [0, 1, 2, 3, 4]))

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
