#!/usr/bin/env python3
"""
apriltag_calibrate.py — AprilTag gaze↔robot calibration tool (WS5 REV03).

Methodology: the REV03 origin is archived at SoftwareDocs/_archive/harmony-bci/
gaze-calibration/rev03-apriltag-methodology.md (its § citations below); the REV04
sweep + planar solve added here follow the active SoftwareDocs/projects/harmony-bci/
gaze-calibration/rev04-planar-coverage-methodology.md. Produces the calibration the
control tool (`tools/apriltag_control_test.py`) uses to drive the robot from gaze.

Tier-3 operator tool. Camera frames + gaze + intrinsics come from the existing
frame relay (`Utils.remote_frame_reader`), so it runs in the `lsl` env. The
panel's embedded relay (or `python -m Utils.frame_relay`) must be up first — the
sole Neon subscriber. The `collect` stage commands the free-arm opcodes
(`m` free / `c` capture-lock, plus `q` telemetry) exactly like the existing
free-arm recorder — it sends no trajectory and cannot drive the arm to a pose;
the `detect`/`gaze` stages are camera-only; `solve` is offline.

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
    sweep    + robot (REV04). CONTINUOUS swept capture: free the arm and
             hand-guide the EE across the table while a ~20 Hz loop pairs each
             fresh frame with telemetry, time-aligns + quality-gates it, derives
             the table-plane (u,v), and drives an adaptive coverage box UI. Stops
             on full coverage. Saves apriltag_sweep_<UTC>.npz (UV, Q, X, maps).
    solve    offline, auto-detected from the npz: a REV04 sweep (UV) → the planar
             (u,v)→Q library + A2 in-plane similarity residual/LOO
             (scheme="planar_uv_nn"); a REV03 collect (P_world) → the rigid
             Umeyama T_base_world fit. Writes a <stem>_calib.npz the control
             tool consumes.

World tags: `--world-tag-ids` takes one or more (corner) tags. `collect` first
**registers a rigid world map** over them (all visible once, arm clear), so at
capture/runtime any visible subset recovers the SAME world frame — robust to the
exoskeleton arm occluding individual tags, and more accurate (fusion) when
several are visible. `detect`/`gaze` track the first id for the per-tag jitter /
pose-flip diagnostic.

Examples:
    python tools/apriltag_calibrate.py --stage detect --world-tag-ids 0 --tag-size 0.10
    python tools/apriltag_calibrate.py --stage gaze   --world-tag-ids 0 --tag-size 0.10
    python tools/apriltag_calibrate.py --stage collect --with-robot \\
        --world-tag-ids 0 1 2 --ee-tag-ids 8 9 --tag-size 0.10 --ee-tag-size 0.04 --t-eetag-ee 0 0 50
    python tools/apriltag_calibrate.py --stage solve runs/apriltag_capture_<UTC>.npz

The EE side mirrors the world side: ``--ee-tag-ids`` registers a rigid EE
**bundle** (occlusion-robust, ambiguity-reducing — the HIL accuracy floor of a
single small EE tag, verification report §5); the back-compat singular
``--ee-tag-id`` is just a 1-entry bundle.
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
    eetag_rayplane_point_world,
    eetag_to_world_point,
    gaze_ray_cam,
    geodesic_angle_deg,
    per_point_errors,
    umeyama_rigid,
    umeyama_similarity_2d,
)
from Utils.gaze.apriltag_detect import (  # noqa: E402
    RelayConsumer,
    detect_tags,
    load_detector,
)
from Utils.gaze.apriltag_world import (  # noqa: E402
    average_pose,
    build_world_map,
    plane_coords,
    recover_world_pose,
    world_map_to_arrays,
)
from Utils.gaze.apriltag_sweep import (  # noqa: E402
    accept_sweep_sample,
    frame_telemetry_dt,
)
from Utils.gaze.coverage import CoverageGrid  # noqa: E402
from Utils.gaze.harmony_link import HarmonyLink  # noqa: E402


def _log(msg: str) -> None:
    print(f"[apriltag_calibrate] {msg}", flush=True)


# ── stage: detect ─────────────────────────────────────────────────────────────


def stage_detect(args, consumer: RelayConsumer) -> int:
    detector = load_detector(args.families)
    K = consumer.camera_matrix
    ee_ids = _resolve_ee_ids(args)
    tag_id = (args.world_tag_ids[0] if args.world_tag_ids
              else (ee_ids[0] if ee_ids else None))
    if tag_id is None:
        _log("detect needs --world-tag-ids (or --ee-tag-id/--ee-tag-ids) to track")
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
    ee_ids = _resolve_ee_ids(args)
    tag_id = (args.world_tag_ids[0] if args.world_tag_ids
              else (ee_ids[0] if ee_ids else None))
    if tag_id is None:
        _log("gaze needs --world-tag-ids (or --ee-tag-id/--ee-tag-ids) to fixate")
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


def _register_tag_map(consumer, detector, K, ids, tag_size, tag_sizes,
                      label="tag", dur=2.0):
    """Average each tag's pose over a short window (all tags visible, the body
    they sit on held still) and build a rigid map over them via
    ``build_world_map``. Returns ``(map, sorted_seen_ids)`` or ``(None, [])`` if
    no listed tag was detected.

    Used for BOTH the static world-tag map and the EE-tag bundle map (rev04 §7):
    ``build_world_map``/``recover_world_pose`` are frame-agnostic, so the EE
    bundle is registered exactly like the world bundle — its "world" frame is the
    EE reference tag's frame, and the hand-measured offset is taken w.r.t. that
    reference tag."""
    _log(f"{label}-map registration: keep ALL {label} tags {ids} visible and the "
         f"body they sit on STILL for ~{dur:.0f}s …")
    obs = {int(i): [] for i in ids}
    last_idx = None
    deadline = time.time() + dur
    while time.time() < deadline:
        b = consumer.latest()
        if b is None or b.video is None or b.video.bgr is None or b.video.frame_idx == last_idx:
            time.sleep(0.005)
            continue
        last_idx = b.video.frame_idx
        tags = detect_tags(detector, b.video.bgr, K, tag_size, tag_sizes=tag_sizes)
        for i in ids:
            if int(i) in tags:
                obs[int(i)].append(tags[int(i)]["T"])
    seen = {i: average_pose(v) for i, v in obs.items() if v}
    if not seen:
        return None, []
    return build_world_map(seen), sorted(seen)


def _resolve_ee_ids(args) -> Optional[List[int]]:
    """The EE tag id list from either ``--ee-tag-ids`` (plural, a bundle) or the
    back-compat singular ``--ee-tag-id`` (a 1-entry map). Returns None if neither
    was given. A single id still flows through the map machinery as a 1-tag map,
    so the capture/runtime code has exactly one EE-pose path."""
    if args.ee_tag_ids:
        return [int(i) for i in args.ee_tag_ids]
    if args.ee_tag_id is not None:
        return [int(args.ee_tag_id)]
    return None


def _register_map_interactive(consumer, detector, K, ids, world_tag_size,
                              tag_sizes, label, body_hint, ui=None):
    """Prompt the operator to show all ``ids`` and register a rigid tag map over
    them, retrying until every listed tag is captured. Shared by collect + sweep
    for BOTH the world bundle and the EE bundle (a single EE id is a trivial
    1-entry map — ``recover_world_pose`` returns that tag's pose directly).

    With a coverage ``ui`` the prompt is on the visual interface (SPACE to register,
    q to abort) so the operator never switches to the terminal (operator
    2026-06-24); headless it is a terminal Enter. Returns the map, or ``None`` if the
    operator aborted from the window."""
    while True:
        if ui is not None:
            if not ui.prompt(f"Show {label} tags {ids}",
                             [body_hint, "press SPACE to register, q to abort"]):
                _log(f"  {label}-map registration aborted from the coverage window")
                return None
        else:
            input(f"place ALL {label} tags {ids} visible + {body_hint}, Enter to "
                  f"register the {label}-tag map > ")
        tag_map, seen = _register_tag_map(consumer, detector, K, ids,
                                          world_tag_size, tag_sizes, label=label)
        if tag_map is None:
            _log(f"  no {label} tags detected — reposition and retry")
            continue
        missing = [i for i in ids if i not in seen]
        if missing:
            _log(f"  registered {seen}, MISSING {missing} — re-show all {label} tags "
                 "(the map needs every tag so any subset works later), then retry")
            continue
        _log(f"  {label} map OK: ref={tag_map['ref_id']}, tags={seen}")
        return tag_map


def stage_collect(args, consumer: RelayConsumer) -> int:
    ee_ids = _resolve_ee_ids(args)
    if not args.world_tag_ids or ee_ids is None:
        _log("collect needs --world-tag-ids (one or more) and --ee-tag-id / --ee-tag-ids")
        return 2
    world_ids = [int(i) for i in args.world_tag_ids]
    detector = load_detector(args.families)
    K = consumer.camera_matrix
    offset_mm = np.asarray(args.t_eetag_ee, dtype=float)
    # The EE tags may be smaller than the world tags (they have to fit the EE);
    # pass their true size so their poses are scaled correctly. --tag-size is the
    # world size.
    ee_size = args.ee_tag_size or args.tag_size
    tag_sizes = {int(i): ee_size for i in ee_ids}
    _log(f"tag sizes: world {args.tag_size} m, EE {ee_size} m; world tags "
         f"{world_ids}, EE tags {ee_ids}")

    # Register the multi-tag world map first (occlusion robustness): all world
    # tags must be seen once, arm clear, so any visible subset later recovers the
    # same world frame. Then the EE-tag bundle map (rev04 §7) — a bundle resolves
    # the single-small-tag pose ambiguity that set the HIL accuracy floor
    # (verification report §5); a single id is a 1-entry map. The offset is w.r.t.
    # the EE-map reference tag.
    world_map = _register_map_interactive(consumer, detector, K, world_ids,
                                          args.tag_size, tag_sizes, "world", "arm CLEAR")
    ee_map = _register_map_interactive(consumer, detector, K, ee_ids,
                                       args.tag_size, tag_sizes, "EE", "the EE STILL")
    ee_ref = ee_map["ref_id"]

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
                # Torn-read detector (cpp.md §9.1.2): the `c` telemetry is 28
                # non-atomic reads and can interleave joints sampled ms apart.
                # The captured Q becomes a commanded pose later, so reject a
                # capture that disagrees with the next steady-state read.
                after = link.query_state()
                if after is not None:
                    qd = float(np.max(np.abs(q_joints - after["q"])))
                    if qd > args.settle_eps_rad:
                        _log(f"    captured pose disagrees with readback "
                             f"(max|dq|={qd:.4f} rad); possible torn capture — retry")
                        continue
            else:
                x_ee, q_joints = np.full(3, np.nan), np.full(7, np.nan)

            b = consumer.latest()
            if b is None or b.video is None or b.video.bgr is None:
                _log("    no frame; relay up? (capture discarded)")
                continue
            tags = detect_tags(detector, b.video.bgr, K, args.tag_size, tag_sizes=tag_sizes)
            world_view = {i: tags[i]["T"] for i in world_ids if i in tags}
            ee_view = {i: tags[i]["T"] for i in ee_ids if i in tags}
            T_cam_world = recover_world_pose(world_view, world_map)
            T_cam_eetag = recover_world_pose(ee_view, ee_map)
            if T_cam_world is None or T_cam_eetag is None:
                _log(f"    need ≥1 world tag + ≥1 EE tag; saw {sorted(tags.keys())}. (discarded)")
                continue
            p_world = eetag_to_world_point(T_cam_world, T_cam_eetag, offset_mm)

            p_world_rows.append(p_world)
            x_rows.append(x_ee)
            q_rows.append(q_joints)
            tcw_rows.append(T_cam_world)
            tce_rows.append(T_cam_eetag)
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
    wm_ref, wm_ids, wm_rels, wm_pp, wm_pn = world_map_to_arrays(world_map)
    em_ref, em_ids, em_rels, em_pp, em_pn = world_map_to_arrays(ee_map)
    meta = {
        "version": 3,
        "scheme": "ee_mounted_tag_free_roam",
        "side": args.side,
        "world_tag_ids": world_ids,
        "ee_tag_ids": ee_ids,
        "ee_tag_id": ee_ref,  # back-compat scalar = the EE-map reference tag
        "tag_size_m": args.tag_size,
        "ee_tag_size_m": ee_size,
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
        world_map_ref=np.array(wm_ref),
        world_map_ids=wm_ids,
        world_map_rels=wm_rels,
        world_map_plane_point=wm_pp,
        world_map_plane_normal=wm_pn,
        ee_map_ref=np.array(em_ref),
        ee_map_ids=em_ids,
        ee_map_rels=em_rels,
        ee_map_plane_point=em_pp,
        ee_map_plane_normal=em_pn,
        meta=np.array(meta, dtype=object),
    )
    _log(f"saved {len(p_world_rows)} captures → {out_path}")
    if args.with_robot:
        _log("solve it: "
             f"python tools/apriltag_calibrate.py --stage solve {out_path}")
    return 0


# ── stage: sweep (REV04 continuous swept capture) ────────────────────────────

# Auto-home move duration after a sweep ends — matches the REV01 free-arm
# teardown budget (harmony_free_arm_calibration AUTO_HOME_DURATION_S).
_SWEEP_AUTO_HOME_DUR_S = 4.0


def stage_sweep(args, consumer: RelayConsumer, ui=None) -> int:
    """REV04 continuous swept capture (rev04 §2). The operator frees the arm and
    hand-guides the EE **across the table surface** while a ~20 Hz loop pairs each
    fresh relay frame with a robot-telemetry sample, time-aligns + quality-gates it
    (`apriltag_sweep`), derives the table-plane ``(u,v)``, and feeds a
    `CoverageGrid`. One slow sweep yields hundreds of correspondences instead of the
    ~23 discrete collect poses. Stops when coverage is sufficient, the time budget
    is hit, the UI requests quit, or Ctrl-C. Writes ``apriltag_sweep_<UTC>.npz``.

    ``ui`` is an optional coverage view (step 4) exposing ``update(grid, cur_uv,
    target_uv)`` / ``should_quit()`` / ``close()``; ``None`` runs headless with a
    periodic text summary (the camera-side verification path)."""
    ee_ids = _resolve_ee_ids(args)
    if not args.world_tag_ids or ee_ids is None:
        _log("sweep needs --world-tag-ids and --ee-tag-id/--ee-tag-ids")
        return 2
    world_ids = [int(i) for i in args.world_tag_ids]
    detector = load_detector(args.families)
    K = consumer.camera_matrix
    offset_mm = np.asarray(args.t_eetag_ee, dtype=float)
    ee_size = args.ee_tag_size or args.tag_size
    tag_sizes = {int(i): ee_size for i in ee_ids}
    _log(f"tag sizes: world {args.tag_size} m, EE {ee_size} m; world tags "
         f"{world_ids}, EE tags {ee_ids}")

    world_map = _register_map_interactive(consumer, detector, K, world_ids,
                                          args.tag_size, tag_sizes, "world", "arm CLEAR",
                                          ui=ui)
    if world_map is None:
        if ui is not None:
            ui.close()
        return 1
    ee_map = _register_map_interactive(consumer, detector, K, ee_ids,
                                       args.tag_size, tag_sizes, "EE", "the EE STILL",
                                       ui=ui)
    if ee_map is None:
        if ui is not None:
            ui.close()
        return 1
    ee_ref = ee_map["ref_id"]
    plane_point = world_map["plane_point"]
    plane_normal = world_map["plane_normal"]

    grid = CoverageGrid(cell_size_mm=args.cell_size_mm, min_samples=args.min_samples,
                        min_spread_mm=args.min_spread_mm)

    link: Optional[HarmonyLink] = None
    arm_is_free = False
    if args.with_robot:
        link = HarmonyLink(args.robot_ip, args.robot_port, args.bind_ip,
                           args.bind_port, side=args.side)
        if not link.free_arm():
            _log("could not free the arm (ACK:MASTER_FREE not seen) — aborting sweep")
            link.close()
            return 1
        arm_is_free = True
        _log("arm FREED — hand-guide the EE slowly across the table surface")
    else:
        _log("no --with-robot: Q/X will be NaN (camera-side dry run — verifies "
             "(u,v) + coverage on a recorded stream, not a solvable library)")

    uv_rows: List[np.ndarray] = []
    q_rows: List[np.ndarray] = []
    x_rows: List[np.ndarray] = []
    tcw_rows: List[np.ndarray] = []
    tce_rows: List[np.ndarray] = []
    t_rows: List[float] = []

    period = 1.0 / args.sample_hz
    last_idx = None
    accepted = 0
    try:
        # Start gate (rev04 §3, operator 2026-06-24): with the arm freed, let the
        # operator position it before any sample is recorded, so the transit into
        # the start pose never enters the library. The deadline starts AFTER the
        # gate so positioning time is not charged against the sweep budget.
        started = _await_sweep_start(ui)
        if not started:
            _log("sweep: start aborted before any capture")
        else:
            _log(f"sweep: ≤{args.max_sweep_s:.0f}s, stop early on full coverage "
                 f"(cell {args.cell_size_mm:.0f}mm, ≥{args.min_samples} samples, spread "
                 f"≥{args.min_spread_mm:.0f}mm). Ctrl-C to stop + save.")
        next_tick = time.monotonic()
        last_report = time.monotonic()
        deadline = time.monotonic() + args.max_sweep_s
        while started and time.monotonic() < deadline:
            next_tick += period
            b, t_frame = consumer.latest_with_time()
            if b is None or b.video is None or b.video.bgr is None or b.video.frame_idx == last_idx:
                _sleep_until(next_tick)
                continue
            last_idx = b.video.frame_idx

            if link is not None:
                rstate = link.query_state()
                if rstate is None:
                    _sleep_until(next_tick)
                    continue
                q_joints, x_ee, t_robot = rstate["q"], rstate["ee"], rstate["_t"]
            else:
                q_joints, x_ee, t_robot = np.full(7, np.nan), np.full(3, np.nan), time.time()

            tags = detect_tags(detector, b.video.bgr, K, args.tag_size, tag_sizes=tag_sizes)
            world_view = {i: tags[i]["T"] for i in world_ids if i in tags}
            ee_view = {i: tags[i]["T"] for i in ee_ids if i in tags}
            T_cam_world = recover_world_pose(world_view, world_map)
            T_cam_eetag = recover_world_pose(ee_view, ee_map)
            margins = ([tags[i]["margin"] for i in world_view]
                       + [tags[i]["margin"] for i in ee_view])
            dt = frame_telemetry_dt(t_frame, t_robot)
            ok, reason = accept_sweep_sample(
                world_seen=T_cam_world is not None, ee_seen=T_cam_eetag is not None,
                margins=margins, dt_s=dt,
                min_margin=args.min_margin, max_align_dt_s=args.max_align_dt_s)

            cur_uv = None
            if ok:
                # Depth-ambiguity-free EE position: the tag-centre line of sight ∩
                # the known table plane — the SAME ray∩plane runtime uses for gaze
                # (rev04 §5 follow-up, 2026-06-24). The single small EE tag's 3-D
                # pose range is too noisy (HIL repeatability 52 mm); its centre
                # direction is not. None = ray parallel/behind → drop the sample.
                p_world = eetag_rayplane_point_world(T_cam_world, T_cam_eetag,
                                                     plane_point, plane_normal)
                if p_world is not None:
                    cur_uv = plane_coords(p_world, plane_point, plane_normal)
                    grid.add(cur_uv)
                    uv_rows.append(cur_uv)
                    q_rows.append(q_joints)
                    x_rows.append(x_ee)
                    tcw_rows.append(T_cam_world)
                    tce_rows.append(T_cam_eetag)
                    t_rows.append(t_robot)
                    accepted += 1

            target = grid.next_target()
            if ui is not None:
                ui.update(grid, cur_uv, target)
                if ui.should_quit():
                    _log("sweep: quit requested from the coverage view")
                    break

            now = time.monotonic()
            if now - last_report >= 1.0:
                s = grid.summary()
                tgt = "—" if target is None else np.round(target, 0).tolist()
                _log(f"  accepted={accepted} cells={int(s['visited'])} "
                     f"sufficient={int(s['sufficient'])} → go to {tgt}"
                     + ("" if ok else f"  (last drop: {reason})"))
                last_report = now

            # Auto-stop needs ≥ min_cells covered, not just "every visited cell
            # sufficient" — otherwise an operator who dwells in ONE cell at the
            # start (filling min_samples with a small wiggle that clears the
            # frozen-hand spread guard) would end the sweep with a single cell
            # (critic I1). The operator can still stop early manually (UI 'q' /
            # Ctrl-C); this only gates the AUTOMATIC stop.
            if grid.done() and len(grid.visited_cells()) >= args.min_cells:
                _log(f"sweep: coverage COMPLETE — {len(grid.visited_cells())} cells, "
                     "every visited cell sufficient")
                break
            _sleep_until(next_tick)
    except KeyboardInterrupt:
        _log("sweep: Ctrl-C — stopping and saving what we have")
    finally:
        if link is not None:
            # The sweep leaves the arm in zero-stiffness master_free; on ANY exit
            # path (coverage done, Ctrl-C, UI quit, error) re-lock it with `c` and
            # auto-home before releasing the socket — otherwise the operator lets go
            # of a LIMP arm and it drops under gravity. Mirrors the REV01 free-arm
            # teardown (harmony_free_arm_calibration.py:1311-1328), the pattern this
            # sweep ports (rev04 §2). The safety-lock pose is discarded.
            if arm_is_free:
                _log("re-locking the arm (`c`) before auto-home — keep clear")
                try:
                    link.capture_pose()
                except Exception as exc:  # best-effort safety lock; still try home
                    _log(f"  WARNING: safety lock raised {exc!r}; homing anyway")
                _log("auto-homing the arm — STAND CLEAR of the workspace")
                if not link.home(_SWEEP_AUTO_HOME_DUR_S):
                    _log("  WARNING: no home ACK — verify the arm is safe visually")
            link.close()
        if ui is not None:
            ui.close()

    if len(uv_rows) < 3:
        _log(f"only {len(uv_rows)} accepted samples (<3) — not saving a solvable npz")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone
    stamp = args.utc_stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"apriltag_sweep_{stamp}.npz"
    # Tag each saved sample green/amber by its cell's FINAL sufficiency so the solve
    # can build the library from green-cell samples only (rev04 §3, operator
    # 2026-06-24): extraneous transit samples land in still-partial cells and would
    # otherwise pollute the (u,v)→Q nearest-neighbour lookup.
    green = grid.sufficient_mask(uv_rows)
    wm_ref, wm_ids, wm_rels, wm_pp, wm_pn = world_map_to_arrays(world_map)
    em_ref, em_ids, em_rels, em_pp, em_pn = world_map_to_arrays(ee_map)
    meta = {
        "version": 3,
        "scheme": "planar_sweep",
        "side": args.side,
        "world_tag_ids": world_ids,
        "ee_tag_ids": ee_ids,
        "ee_tag_id": ee_ref,
        "tag_size_m": args.tag_size,
        "ee_tag_size_m": ee_size,
        "t_eetag_ee_mm": offset_mm.tolist(),
        # EE (u,v) from the tag-centre line of sight ∩ table plane (ambiguity-free),
        # not the tag's 3-D pose translation (rev04 §5 follow-up, 2026-06-24). The
        # offset above is recorded for provenance but not applied in this mode.
        "ee_point_method": "rayplane_center",
        "with_robot": bool(args.with_robot),
        "sample_hz": args.sample_hz,
        "coverage": {"cell_size_mm": args.cell_size_mm,
                     "min_samples": args.min_samples,
                     "min_spread_mm": args.min_spread_mm,
                     "complete": bool(grid.done())},
        "units": {"UV": "mm (table plane)", "X": "mm", "Q": "rad", "t_eetag_ee": "mm"},
    }
    np.savez_compressed(
        out_path,
        UV=np.vstack(uv_rows),
        Q=np.vstack(q_rows),
        X=np.vstack(x_rows),
        green=green,
        T_cam_world=np.stack(tcw_rows),
        T_cam_eetag=np.stack(tce_rows),
        t=np.asarray(t_rows, dtype=float),
        world_map_ref=np.array(wm_ref), world_map_ids=wm_ids, world_map_rels=wm_rels,
        world_map_plane_point=wm_pp, world_map_plane_normal=wm_pn,
        ee_map_ref=np.array(em_ref), ee_map_ids=em_ids, ee_map_rels=em_rels,
        ee_map_plane_point=em_pp, ee_map_plane_normal=em_pn,
        meta=np.array(meta, dtype=object),
    )
    _log(f"saved {len(uv_rows)} swept samples ({int(green.sum())} in green/sufficient "
         f"cells) → {out_path}")
    if args.with_robot:
        _log("solve it: "
             f"python tools/apriltag_calibrate.py --stage solve {out_path}")
    return 0


def _sleep_until(deadline: float) -> None:
    """Monotonic-clock pacing for the sweep loop; falls straight through if behind
    (detection can exceed one tick under load)."""
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def _make_coverage_ui(args):
    """The sweep's coverage view: the OpenCV box (rev04 §3) unless ``--no-ui``
    (headless camera-side verification, where only the text summary runs)."""
    if args.no_ui:
        return None
    from Utils.gaze.coverage_view import CoverageBoxUI
    return CoverageBoxUI(args.cell_size_mm, audio=args.audio)


def _await_sweep_start(ui) -> bool:
    """Gate between freeing the arm and recording the first sample (rev04 §3,
    operator 2026-06-24). With the coverage window up the prompt is on-screen
    (SPACE to start / q to abort); headless (``--no-ui``) it is a terminal Enter.
    Returns True to begin sampling, False to abort with nothing captured."""
    if ui is not None:
        return ui.wait_for_start()
    try:
        input("position the freed arm, then press Enter to START sampling "
              "(Ctrl-C aborts) > ")
        return True
    except EOFError:
        return False


# ── stage: solve (offline) — writes the consolidated calibration ─────────────


def stage_solve(args) -> int:
    """Dispatch the offline solve by capture type: a REV04 sweep npz (``UV``
    present) → the planar ``(u,v)→Q`` solve; a REV03 collect npz (``P_world``) →
    the rigid ``T_base_world`` Umeyama fit."""
    npz_path = Path(args.npz)
    if not npz_path.is_file():
        _log(f"npz not found: {npz_path}")
        return 2
    z = np.load(npz_path, allow_pickle=True)
    if "UV" in z.files:
        return stage_solve_planar(args, z, npz_path)
    if "P_world" in z.files:
        return stage_solve_rigid(args, z, npz_path)
    _log(f"{npz_path.name} has neither UV (planar sweep) nor P_world (rigid "
         "collect) — not a solvable capture")
    return 2


def stage_solve_rigid(args, z, npz_path) -> int:
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
    # Carry the world map through so the control tool recovers the SAME world
    # frame from whichever tags are visible (occlusion-robust).
    extra = {}
    for k in ("world_map_ref", "world_map_ids", "world_map_rels",
              "world_map_plane_point", "world_map_plane_normal"):
        if k in z.files:
            extra[k] = z[k]
    np.savez_compressed(
        out_path,
        X=X, Q=Q, T_base_world=T_base_world,
        per_point_errors_mm=errs,
        meta=np.array(calib_meta, dtype=object),
        **extra,
    )
    _log(f"calibration → {out_path}")
    _log("drive the robot: "
         f"python tools/apriltag_control_test.py --calib {out_path}")
    return 0


def stage_solve_planar(args, z, npz_path) -> int:
    """REV04 planar solve (rev04 §1). Builds the ``(u,v)→Q`` nearest-neighbour
    library (A1, the command path) from a sweep npz and reports the A2 2-D
    similarity residual + leave-one-out as the quality readout — the well-conditioned
    in-plane analogue of the old Umeyama RMS. Writes ``<stem>_calib.npz`` with
    ``scheme="planar_uv_nn"`` (UV, Q, world map + plane) for the control tool."""
    UV = np.asarray(z["UV"], dtype=float)
    Q = np.asarray(z["Q"], dtype=float)
    X = np.asarray(z["X"], dtype=float) if "X" in z.files else np.full((UV.shape[0], 3), np.nan)

    # Green-only (rev04 §3, operator 2026-06-24): build the library from samples in
    # sufficient ("green") coverage cells only — the default, since transit /
    # extraneous samples in still-partial cells pollute the NN lookup. ``green`` is
    # the per-sample sufficiency mask the sweep stored; a pre-fix npz lacks it
    # (then we cannot filter and keep all). ``--include-partial`` keeps every sample.
    if args.green_only and "green" in z.files:
        gmask = np.asarray(z["green"], dtype=bool)
        kept = int(gmask.sum())
        _log(f"green-only: {kept}/{len(gmask)} samples in sufficient cells "
             f"(dropped {len(gmask) - kept} partial-cell; --include-partial keeps all)")
        UV, Q, X = UV[gmask], Q[gmask], X[gmask]
    elif args.green_only:
        _log("green-only requested but this npz has no 'green' mask (pre-fix sweep) "
             "— using all accepted samples")
    else:
        _log("--include-partial: using all accepted samples (green + amber cells)")

    # A1 library needs finite (UV, Q); a no-robot dry-run sweep has NaN Q.
    lib_ok = np.all(np.isfinite(UV), axis=1) & np.all(np.isfinite(Q[:, :7]), axis=1)
    UV, Q, X = UV[lib_ok], Q[lib_ok], X[lib_ok]
    n = UV.shape[0]
    if n < 3:
        _log(f"only {n} finite (UV,Q) rows (<3) — cannot build a library. "
             "Was --with-robot used during the sweep?")
        return 1
    _log(f"planar library: {n} (u,v)→Q rows; u∈[{UV[:,0].min():.0f},{UV[:,0].max():.0f}] "
         f"v∈[{UV[:,1].min():.0f},{UV[:,1].max():.0f}] mm")

    # A2 (diagnostic): in-plane similarity fit base(x,y) ← plane(u,v) where the
    # base-frame EE position is finite. Reports residual + LOO, NOT the command path.
    a2_ok = X is not None and np.all(np.isfinite(X), axis=1)
    n2 = int(np.sum(a2_ok)) if X is not None else 0
    rms2 = float("nan")
    if n2 >= 2:
        uv_f, xy_f = UV[a2_ok], X[a2_ok, :2]
        _A, _t, _s, rms2 = umeyama_similarity_2d(uv_f, xy_f)
        _log(f"A2 in-plane similarity residual (uv→base xy, {n2} pts): RMS={rms2:.2f} mm "
             f"(scale {_s:.4f})")
        if n2 >= 4:
            loo = []
            for i in range(n2):
                mask = np.ones(n2, dtype=bool)
                mask[i] = False
                A_i, t_i, _si, _r = umeyama_similarity_2d(uv_f[mask], xy_f[mask])
                pred = A_i @ uv_f[i] + t_i
                loo.append(float(np.linalg.norm(pred - xy_f[i])))
            loo = np.array(loo)
            _log(f"A2 leave-one-out: median={np.median(loo):.2f} max={loo.max():.2f} mm")
        ok = rms2 < 20.0
        _log(f"VERDICT: {'PASS' if ok else 'REVIEW'} (A2 in-plane RMS target ≲10–20 mm)")
    else:
        _log("A2 residual skipped: <2 rows with a finite base-frame X (no-robot "
             "dry run?). The A1 (u,v)→Q library is still written.")

    src_meta = z["meta"].item() if "meta" in z.files else {}
    calib_meta = dict(src_meta)
    calib_meta.update({"scheme": "planar_uv_nn", "n_points": int(n),
                       "a2_inplane_rms_mm": rms2, "a2_n_points": n2,
                       "source_capture": npz_path.name})
    out_path = npz_path.with_name(npz_path.stem.replace("apriltag_sweep", "apriltag")
                                  + "_calib.npz")
    # Carry the world map + plane through so the control tool recovers the SAME
    # world frame and the SAME (u,v) basis used at capture.
    extra = {}
    for k in ("world_map_ref", "world_map_ids", "world_map_rels",
              "world_map_plane_point", "world_map_plane_normal"):
        if k in z.files:
            extra[k] = z[k]
    np.savez_compressed(
        out_path,
        UV=UV, Q=Q, X=X,
        meta=np.array(calib_meta, dtype=object),
        **extra,
    )
    _log(f"planar calibration → {out_path}")
    _log("drive the robot: "
         f"python tools/apriltag_control_test.py --calib {out_path}")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    import config as cfg
    p = argparse.ArgumentParser(description="AprilTag gaze↔robot calibration tool")
    p.add_argument("--stage", required=True,
                   choices=["detect", "gaze", "collect", "sweep", "solve"])
    p.add_argument("npz", nargs="?", default=None,
                   help="solve stage: path to an apriltag_capture_*.npz")
    p.add_argument("--families", default="tag36h11")
    p.add_argument("--tag-size", type=float, default=0.06,
                   help="WORLD tag edge length in METRES (default 0.06)")
    p.add_argument("--ee-tag-size", type=float, default=None,
                   help="EE tag edge length in METRES if it differs from the "
                        "world tag (the EE tag is usually smaller; default: --tag-size)")
    p.add_argument("--world-tag-ids", type=int, nargs="+", default=[0],
                   help="one or more world tag ids; collect registers a map over "
                        "them so any visible subset recovers the world frame "
                        "(occlusion-robust). detect/gaze use the first.")
    p.add_argument("--ee-tag-id", type=int, default=None,
                   help="single EE tag id (back-compat: a 1-entry EE map)")
    p.add_argument("--ee-tag-ids", type=int, nargs="+", default=None,
                   help="one or more EE tag ids forming a rigid EE BUNDLE; "
                        "collect/sweep register a map over them so any visible "
                        "subset recovers the EE pose (resolves the single-small-tag "
                        "ambiguity that set the HIL accuracy floor). The offset "
                        "--t-eetag-ee is measured w.r.t. the bundle's reference "
                        "(lowest) tag. Overrides --ee-tag-id when given.")
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
    p.add_argument("--out-dir", default="runs", help="collect/sweep: directory for the saved npz")
    p.add_argument("--utc-stamp", default=None,
                   help="collect/sweep: override the auto UTC stamp in the npz filename")
    # sweep stage (REV04 continuous capture + coverage)
    p.add_argument("--sample-hz", type=float, default=20.0,
                   help="sweep: telemetry/frame sampling rate (REV01 transit rate)")
    p.add_argument("--max-sweep-s", type=float, default=180.0,
                   help="sweep: hard time budget; the sweep also stops on full coverage")
    p.add_argument("--cell-size-mm", type=float, default=50.0,
                   help="sweep: coverage cell size on the table plane (mm)")
    p.add_argument("--min-cells", type=int, default=4,
                   help="sweep: minimum covered cells before the AUTOMATIC stop may "
                        "fire (guards against ending after a single dwelt cell)")
    p.add_argument("--min-samples", type=int, default=8,
                   help="sweep: samples needed per cell for sufficiency")
    p.add_argument("--min-spread-mm", type=float, default=15.0,
                   help="sweep: required spatial spread within a cell (frozen-hand guard)")
    p.add_argument("--min-margin", type=float, default=20.0,
                   help="sweep: reject a sample whose weakest contributing tag's "
                        "decision margin is below this (motion blur / glancing view)")
    p.add_argument("--max-align-dt-ms", type=float, default=50.0,
                   help="sweep: reject a sample whose frame↔telemetry offset exceeds "
                        "this (stale frame); default = one 20 Hz tick")
    p.add_argument("--no-ui", action="store_true",
                   help="sweep: run headless with a text summary (no OpenCV coverage box)")
    p.add_argument("--audio", action="store_true",
                   help="sweep: speak coverage cues (solo-operator aid, rev04 §3)")
    p.add_argument("--include-partial", dest="green_only", action="store_false",
                   help="solve: build the library from ALL accepted samples, not just "
                        "those in sufficient ('green') coverage cells. Default is "
                        "green-only — partial-cell transit samples pollute the NN "
                        "lookup (rev04 §3).")
    p.set_defaults(green_only=True)
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
    args.max_align_dt_s = args.max_align_dt_ms / 1000.0
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
        if args.stage == "sweep":
            return stage_sweep(args, consumer, ui=_make_coverage_ui(args))
    finally:
        consumer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
