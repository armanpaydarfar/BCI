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
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

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
    recover_world_pose_pnp,
)
from Utils.gaze.apriltag_world import (  # noqa: E402
    plane_coords,
    recover_world_pose,
    register_world_map_multiview,
    table_normal_from_rel,
    world_map_from_arrays,
    world_map_geometry_report,
    world_map_to_arrays,
)
from Utils.gaze.apriltag_world_3d import (  # noqa: E402
    apply_plane_structure,
    register_world_map_3d,
    world_map_3d_geometry_report,
    world_map_3d_reproducibility,
)
from Utils.gaze.apriltag_sweep import (  # noqa: E402
    accept_sweep_sample,
    frame_telemetry_dt,
)
from Utils.gaze.coverage import CoverageGrid  # noqa: E402
from Utils.gaze.coverage_voxel import VoxelCoverage  # noqa: E402
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
                      label="tag", dur=6.0):
    """Multi-view registration (REV05, 2026-06-25): collect the listed tags across a
    slow HEAD SWEEP — the body they sit on held STILL — and fuse them into one
    reproducible map via ``register_world_map_multiview``. Seeing the static tags
    from many viewpoints triangulates away the single-tag depth ambiguity that made
    the old single-window average wander run-to-run (≈300 mm origins, ≈70° normal) —
    the real reason a calibration was a coin-flip. Returns ``(map, sorted_seen_ids)``
    or ``(None, [])`` if no listed tag was detected.

    Used for BOTH the static world-tag map (move the head around the table) and the
    EE-tag bundle map (hold the EE still, move the head around IT): the EE side is
    momentarily static during its own registration, so the same multi-view fuse
    applies; a single EE tag is a 1-entry map (no triangulation, just denoised)."""
    _log(f"{label}-map registration: keep ALL {label} tags {ids} visible and the body "
         f"they sit on STILL, then SLOWLY move your head around them for ~{dur:.0f}s "
         "(many viewpoints → a stable, reproducible map) …")
    frames: List[Dict[int, np.ndarray]] = []
    last_idx = None
    deadline = time.time() + dur
    while time.time() < deadline:
        b = consumer.latest()
        if b is None or b.video is None or b.video.bgr is None or b.video.frame_idx == last_idx:
            time.sleep(0.005)
            continue
        last_idx = b.video.frame_idx
        tags = detect_tags(detector, b.video.bgr, K, tag_size, tag_sizes=tag_sizes)
        frame = {int(i): tags[int(i)]["T"] for i in ids if int(i) in tags}
        if frame:
            frames.append(frame)
    if not frames:
        return None, []
    # Report the MAP's ids (tags fused into it = co-visible with the reference), not
    # every tag merely glimpsed — so a tag that never shared a frame with the ref is
    # flagged MISSING upstream and the operator re-shows it, rather than silently
    # dropping out of the map.
    wm = register_world_map_multiview(frames)
    return wm, list(wm["ids"])


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


def _check_world_geometry(world_map) -> str:
    """Log the world-map geometry verdict against the known table layout (corner ~90°,
    coplanar) and return ``'GOOD'``/``'SKEWED'``/``'UNKNOWN'``. A TOP-DOWN registration
    should read GOOD; a seated view reads SKEWED because per-tag pose is oblique-biased
    (operator 2026-06-25) — which is exactly why the map is registered top-down and
    reused, with seated frames recovered by the multi-tag board PnP."""
    import config as _cfg
    xe = list(getattr(_cfg, "APRILTAG_TABLE_X_EDGE_IDS", [0, 1]))
    ye = list(getattr(_cfg, "APRILTAG_TABLE_Y_EDGE_IDS", [2, 0]))
    if not all(int(i) in world_map["ids"] for i in (*xe, *ye)):
        _log("world-map geometry: edge tags not all present — squareness check skipped")
        return "UNKNOWN"
    geo = world_map_geometry_report(world_map, xe, ye)
    corner = geo["corner_angle_deg"]
    verdict = "GOOD" if np.isfinite(corner) and abs(corner - 90.0) <= 10.0 else "SKEWED"
    _log(f"world-map geometry: corner {corner:.1f}° (want ~90°), "
         f"edges {geo['x_edge_len_mm']:.0f}×{geo['y_edge_len_mm']:.0f} mm, "
         f"out-of-plane {geo['max_out_of_plane_mm']:.1f} mm → {verdict}")
    return verdict


def _save_world_map(world_map, out_dir: str, stamp: Optional[str], tag_size: float,
                    world_ids: List[int]) -> Path:
    """Persist a registered world map to ``runs/world_map_<UTC>.npz`` for reuse across
    seated sweeps (the static tags never move)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"world_map_{stamp}.npz"
    ref, ids, rels, pp, pn = world_map_to_arrays(world_map)
    np.savez_compressed(path, world_map_ref=np.array(ref), world_map_ids=ids,
                        world_map_rels=rels, world_map_plane_point=pp,
                        world_map_plane_normal=pn,
                        meta=np.array({"stage": "register-world", "tag_size_m": tag_size,
                                       "world_tag_ids": list(world_ids)}, dtype=object))
    return path


def _load_world_map(path: str):
    """Load a world map saved by `_save_world_map`."""
    z = np.load(path, allow_pickle=True)
    return world_map_from_arrays(z["world_map_ref"], z["world_map_ids"],
                                 z["world_map_rels"], z["world_map_plane_point"],
                                 z["world_map_plane_normal"])


def stage_register_world(args, consumer: RelayConsumer, ui=None) -> int:
    """Register the STATIC world-tag map ONCE from a TOP-DOWN view and save it (REV05,
    2026-06-25). Per-tag AprilTag pose is oblique-biased, so the map geometry is only
    accurate viewed near top-down (operator: ~88° corner standing vs 60–75° seated).
    Register here (stand over the table — NO robot needed) and reuse the saved map for
    seated sweeps, where each frame's camera pose is recovered by the multi-tag board
    PnP — accurate even at the seated 45° angle. Writes ``runs/world_map_<UTC>.npz``."""
    if not args.world_tag_ids:
        _log("register-world needs --world-tag-ids")
        return 2
    world_ids = [int(i) for i in args.world_tag_ids]
    detector = load_detector(args.families)
    K = consumer.camera_matrix
    _log(f"register-world: view tags {world_ids} @ {args.tag_size} m from TOP-DOWN "
         "(stand over the table; no robot needed). Geometry is accurate near top-down.")
    world_map = _register_map_interactive(consumer, detector, K, world_ids,
                                          args.tag_size, {}, "world",
                                          "viewed TOP-DOWN, tags STILL", ui=ui)
    if ui is not None:
        ui.close()
    if world_map is None:
        return 1
    if _check_world_geometry(world_map) == "SKEWED":
        _log("  ⚠ geometry SKEWED — re-run from a more TOP-DOWN view (corner should be "
             "~90°). Saving anyway so you can inspect/retry.")
    path = _save_world_map(world_map, args.out_dir, args.utc_stamp, args.tag_size, world_ids)
    _log(f"saved world map → {path}")
    _log("now sweep seated with it: python tools/apriltag_calibrate.py --stage sweep "
         f"--with-robot --world-map {path} --world-tag-ids {' '.join(map(str, world_ids))} "
         "--ee-tag-ids 5 --tag-size 0.08 --ee-tag-size 0.04 --ee-point-method pose "
         "--t-eetag-ee 150 -200 0 --then-solve")
    return 0


# ── stage: register-world-3d (WS-4 NON-coplanar world map) ───────────────────

# Cap the frames the LIVE quality recompute fuses each tick so its cost is
# constant (not O(accumulated frames)); split-half on ~180 frames is plenty for a
# stable estimate. The final post-capture fuse still uses every frame.
_LIVE_SAMPLE_FRAMES = 180


def _register_world_map_3d_interactive(consumer, detector, K, ids, *, dur=6.0,
                                       max_dur=90.0, tag_size, ui=None,
                                       dist_coeffs=None):
    """Collect a head sweep over the static world tags and fuse a TRUE-3-D map
    (``register_world_map_3d`` — NO coplanar snap), retrying until every listed tag
    is captured. Mirrors ``_register_map_interactive``'s prompt/retry, but uses the
    3-D fuse and RETURNS the raw per-frame detections alongside the map so
    ``world_map_3d_geometry_report`` can recompute the per-tag reproducibility
    residual. Returns ``(map, frames)`` or ``(None, [])`` if aborted from the window.

    The operator hint differs from REV05's: a 3-D layout WANTS to be seen from many
    heights/angles (so off-table tags triangulate), and being non-coplanar is
    expected rather than a fault.

    ``dist_coeffs``: the Neon lens distortion. When given, every frame is UNDISTORTED
    (cv2.remap, same K) before detection, so the fused poses + saved corners are in a
    rectified pinhole frame. The Neon ships raw wide-FOV frames; without this the
    pinhole pose is biased by tens of px at the edges (the 2026-06-30 root cause of
    the ~15px BA reproj floor / 42mm scatter / 0.65 scale).

    With a ``RegistrationView`` the capture is operator-driven: the per-tag
    reproducibility residual + viewpoint diversity are recomputed live on the frames
    so far and shown on the window, so the operator sweeps until the tags go green
    and presses SPACE to accept (capped at ``max_dur``). Headless it falls back to
    the original blind fixed ``dur`` window."""
    import cv2  # lazy: only the camera path needs it
    dist = (np.asarray(dist_coeffs, dtype=float).ravel()
            if dist_coeffs is not None and np.any(np.abs(np.asarray(dist_coeffs)) > 1e-9)
            else None)
    umap = [None, None]      # lazily-built undistort remap (K + frame size are constant)
    if dist is not None:
        _log(f"  undistorting frames before detection (Neon lens, |k1|={abs(dist[0]):.3f})")
    while True:
        if ui is not None:
            from Utils.gaze.registration_view import (
                classify_tags, cone_half_angle_deg, registration_summary)
            if not ui.prompt(f"Show world tags {ids}",
                             ["viewed from MANY heights/angles (non-coplanar is OK)",
                              "SPACE begins LIVE registration; sweep until green",
                              "then SPACE again to accept, q to abort"]):
                _log("  world-3d registration aborted from the coverage window")
                return None, [], []
        else:
            input(f"place ALL world tags {ids} visible, then Enter to register the "
                  "3-D world map > ")
        frames: List[Dict[int, np.ndarray]] = []
        # Per-frame tag CORNER pixels, saved for the Tier-2 constrained bundle
        # adjustment (corner reprojection s.t. coplanar/perpendicular). Index-aligned
        # with ``frames``.
        corner_frames: List[Dict[int, np.ndarray]] = []
        # Live per-tag counters (UI path only): detection count and the viewing
        # directions seen, which drive the diversity/quality readout.
        views: Dict[int, int] = {}
        bearings: Dict[int, List[np.ndarray]] = {}
        last_idx = None
        # The split-half reproducibility re-fuse is far too heavy to run on the draw
        # thread — doing so froze the feed every 0.4 s (operator-reported). Run it in a
        # background worker that reads a SNAPSHOT of the frames so far and publishes a
        # cached readout; the main loop only appends frames + redraws (both cheap), so
        # the video stays smooth. Detection still runs per frame on the main thread.
        qualities = classify_tags(ids, {}, None, {}) if ui is not None else None
        summary = registration_summary(qualities) if ui is not None else None
        if summary is not None:
            summary["scatter_mm"] = None
        data_lock = threading.Lock()
        readout_lock = threading.Lock()
        stop_worker = threading.Event()

        def _recompute_loop():
            nonlocal qualities, summary
            while not stop_worker.wait(0.4):
                with data_lock:
                    # Bound the snapshot to a recent window so the worker's per-cycle
                    # cost is CONSTANT — copying ALL frames each tick made the capture
                    # laggier as it grew. The final fuse (post-capture) uses every frame.
                    snap = frames[-_LIVE_SAMPLE_FRAMES:]
                    v = dict(views)
                    bd = {k: val[-_LIVE_SAMPLE_FRAMES:] for k, val in bearings.items()}
                try:
                    repro_map = None
                    scatter_mm = None
                    if len(snap) >= 4:
                        # Bound to an evenly-spaced subsample so each recompute is
                        # CONSTANT cost; the final fuse (post-capture) uses every frame.
                        if len(snap) > _LIVE_SAMPLE_FRAMES:
                            sel = np.linspace(0, len(snap) - 1, _LIVE_SAMPLE_FRAMES).astype(int)
                            live = [snap[i] for i in sel]
                        else:
                            live = snap
                        try:
                            repro_map = world_map_3d_reproducibility(live)
                            rep = world_map_3d_geometry_report(
                                register_world_map_3d(live), live)
                            scatter_mm = rep["mean_fit_residual_mm"]
                        except ValueError:
                            repro_map = None     # pre-convergence; views/diversity only
                    diversity = {i: cone_half_angle_deg(bd[i]) for i in bd}
                    q = classify_tags(ids, v, repro_map, diversity)
                    s = registration_summary(q)
                    s["scatter_mm"] = scatter_mm
                    with readout_lock:
                        qualities, summary = q, s
                except Exception as exc:         # keep the readout thread alive on a
                    _log(f"  (live quality recompute hiccup: {exc!r})")  # transient error

        worker = None
        if ui is not None:
            worker = threading.Thread(target=_recompute_loop, name="reg-recompute",
                                      daemon=True)
            worker.start()
        deadline = time.time() + (max_dur if ui is not None else dur)
        try:
            while time.time() < deadline:
                b = consumer.latest()
                if (b is None or b.video is None or b.video.bgr is None
                        or b.video.frame_idx == last_idx):
                    time.sleep(0.005)
                    continue
                last_idx = b.video.frame_idx
                bgr = b.video.bgr
                if dist is not None:                       # rectify before detection
                    if umap[0] is None:
                        h, w = bgr.shape[:2]
                        umap[0], umap[1] = cv2.initUndistortRectifyMap(
                            K, dist, None, K, (w, h), cv2.CV_32FC1)
                    bgr = cv2.remap(bgr, umap[0], umap[1], cv2.INTER_LINEAR)
                tags = detect_tags(detector, bgr, K, tag_size)
                frame = {int(i): tags[int(i)]["T"] for i in ids if int(i) in tags}
                if frame:
                    with data_lock:
                        frames.append(frame)
                        corner_frames.append({int(i): tags[int(i)]["corners"]
                                              for i in ids if int(i) in tags})
                        for i, T in frame.items():
                            views[i] = views.get(i, 0) + 1
                            bearings.setdefault(i, []).append(
                                np.asarray(T, dtype=float)[:3, 3])
                if ui is not None:
                    with readout_lock:
                        q, s = qualities, summary
                    # Redraw EVERY frame (cheap cv2 blit) so the video is smooth and
                    # SPACE/q stay responsive; the heavy re-fuse is off-thread above.
                    ui.update(bgr, tags, q, s)
                    if ui.aborted():
                        _log("  world-3d registration aborted from the coverage window")
                        return None, [], []
                    if ui.finished():
                        break
        finally:
            # Join generously: a daemon worker still mid-fuse at interpreter teardown
            # aborts the process ("terminate called" core dump). With the bounded
            # snapshot above a fuse is well under this, so the worker exits cleanly.
            stop_worker.set()
            if worker is not None:
                worker.join(timeout=3.0)
        if not frames:
            _log("  no world tags detected — reposition and retry")
            continue
        world_map = register_world_map_3d(frames)
        seen = list(world_map["ids"])
        missing = [i for i in ids if i not in seen]
        if missing:
            _log(f"  registered {seen}, MISSING {missing} — re-show all world tags "
                 "(the map needs every tag so any subset works later), then retry")
            continue
        _log(f"  world 3-D map OK: ref={world_map['ref_id']}, tags={seen}")
        return world_map, frames, corner_frames


def _save_world_map_3d(world_map, out_dir: str, stamp: Optional[str], tag_size: float,
                       world_ids: List[int], report: Dict, corner_frames=None,
                       K=None) -> Path:
    """Persist a 3-D (non-coplanar) world map to ``runs/world_map_3d_<UTC>.npz``.

    Reuses ``world_map_to_arrays`` — the 3-D map dict shape is identical to the REV05
    map (only the meaning of ``plane_*`` differs: here the best-fit plane through the
    fused 3-D origins, informational only). The geometry report is stored in meta so
    the non-planarity / reproducibility readout travels with the map."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"world_map_3d_{stamp}.npz"
    ref, ids, rels, pp, pn = world_map_to_arrays(world_map)
    # Corner observations (per-frame {tag_id: (4,2) px}) for the Tier-2 constrained
    # bundle adjustment; an object array since tags-per-frame vary. Saved only when
    # captured so a structure-free map's keys are unchanged.
    extra = ({"corner_obs": np.array(corner_frames, dtype=object)}
             if corner_frames else {})
    if K is not None:                              # intrinsics travel with the map so the
        extra["K"] = np.asarray(K, dtype=float)    # Tier-2 BA needs no --k-npz borrow
    np.savez_compressed(path, world_map_ref=np.array(ref), world_map_ids=ids,
                        world_map_rels=rels, world_map_plane_point=pp,
                        world_map_plane_normal=pn,
                        meta=np.array({"stage": "register-world-3d",
                                       "tag_size_m": tag_size,
                                       "world_tag_ids": list(world_ids),
                                       "geometry_3d": report}, dtype=object),
                        **extra)
    return path


def stage_register_world_3d(args, consumer: RelayConsumer, ui=None) -> int:
    """Register a NON-coplanar (true-3-D) STATIC world-tag map and save it (WS-4).

    The 3-D sibling of ``stage_register_world``: same camera-only multi-view fuse,
    but it calls ``register_world_map_3d`` (no coplanar snap) so tags on more than
    one plane — a back panel, a shelf, objects at height — keep their genuine 3-D
    positions instead of being flattened onto the table. Reports non-planarity
    (the positive signal the structure was preserved), the split-half
    ``world_map_3d_reproducibility`` (the real quality gate — map accuracy from
    independent data), and the per-frame scatter (raw sensor noise, context only),
    then writes ``runs/world_map_3d_<UTC>.npz``. No robot needed."""
    if not args.world_tag_ids:
        _log("register-world-3d needs --world-tag-ids")
        return 2
    world_ids = [int(i) for i in args.world_tag_ids]
    detector = load_detector(args.families)
    K = consumer.camera_matrix
    _log(f"register-world-3d: view tags {world_ids} @ {args.tag_size} m from MANY "
         "heights/angles (no robot needed). A non-coplanar layout is expected — the "
         "3-D fuse keeps tags off the table plane (NO coplanar snap).")
    world_map, frames, corner_frames = _register_world_map_3d_interactive(
        consumer, detector, K, world_ids, tag_size=args.tag_size, ui=ui,
        dist_coeffs=consumer.distortion_coeffs)
    if ui is not None:
        ui.close()
    if world_map is None:
        return 1
    # Tier-1 structural hardening (opt-in): if the rig's coplanar/perpendicular tag
    # groups are given, snap each group onto its plane and square the planes — this
    # corrects the single-tag depth ambiguity the fuse can't (table reads flat, the
    # table⊥wall reads 90°), so board-PnP recovery + the table target ride on a
    # structurally-correct map.
    struct = None
    if args.table_plane_tag_ids or args.wall_plane_tag_ids:
        groups = {}
        if args.table_plane_tag_ids:
            groups["table"] = [int(i) for i in args.table_plane_tag_ids]
        if args.wall_plane_tag_ids:
            groups["wall"] = [int(i) for i in args.wall_plane_tag_ids]
        # Loudly flag plane tags that never made it into the map — apply_plane_structure
        # silently skips a group with <3 mapped tags, so a typo or an uncaptured tag
        # would DISABLE structural hardening with no other sign (audit 2026-06-30).
        mapped = {int(i) for i in world_map.get("rel", {})}
        for name, ids in groups.items():
            missing = [i for i in ids if i not in mapped]
            if missing:
                _log(f"structure: WARNING {name}-plane tags {missing} are NOT in the map "
                     f"(not in --world-tag-ids or never captured) — that group keeps only "
                     f"{[i for i in ids if i in mapped]}; <3 ⇒ its hardening is DISABLED.")
        perp = [("table", "wall")] if {"table", "wall"} <= set(groups) else []
        world_map, struct = apply_plane_structure(world_map, groups, perpendicular_pairs=perp)
        for name, g in struct["groups"].items():
            _log(f"structure: {name} plane — corrected origin flatness max={g['flatness_max_mm']:.1f} "
                 f"rms={g['flatness_rms_mm']:.1f} mm (tags {g['ids']})")
        for pair, p in struct["perpendicular"].items():
            off = abs(90.0 - p["angle_before_deg"])
            verdict = "OK" if off <= 8.0 else "POOR — registration was far from square; re-capture"
            _log(f"structure: {pair} was {p['angle_before_deg']:.1f}° (off 90° by {off:.1f}°) "
                 f"→ squared to 90°, each normal moved {p['normal_moved_deg']:.1f}° [{verdict}]")
    report = world_map_3d_geometry_report(world_map, frames)
    if struct is not None:
        report["structure"] = struct
    _log(f"3-D geometry: non-planarity max_out_of_plane={report['max_out_of_plane_mm']:.1f} mm, "
         f"z_spread={report['z_spread_mm']:.1f} mm (large ⇒ genuine 3-D structure; "
         "near-0 ⇒ effectively coplanar, REV05 would do)")
    # Reproducibility (split-half map disagreement) is the real quality verdict;
    # the per-frame scatter is raw AprilTag sensor noise and floors out regardless
    # of how the operator sweeps, so it is reported only as context.
    repro = world_map_3d_reproducibility(frames)
    report["reproducibility_mm"] = repro
    if repro:
        rv = np.array(list(repro.values()))
        _log(f"3-D reproducibility (split-half map disagreement): mean={rv.mean():.1f} mm, "
             f"max={rv.max():.1f} mm — THIS is the quality gate (small ⇒ the fused map "
             "is accurate). Worst tags: "
             f"{sorted(repro, key=lambda i: -repro[i])[:4]}")
    else:
        _log("3-D reproducibility: too few frames / common tags to split-half — capture longer")
    _log(f"3-D per-frame scatter (sensor noise, NOT the gate): mean={report['mean_fit_residual_mm']:.1f} mm, "
         f"max={report['max_fit_residual_mm']:.1f} mm")
    path = _save_world_map_3d(world_map, args.out_dir, args.utc_stamp, args.tag_size,
                              world_ids, report, corner_frames=corner_frames, K=K)
    _log(f"saved 3-D world map → {path} ({len(corner_frames)} frames of corner "
         "observations for Tier-2 bundle adjustment)")
    return 0


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
_SWEEP_AUTO_HOME_DUR_S = 5.0


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
    if args.coverage_3d and not args.with_robot:
        # 3-D voxel coverage bins the EE (x,y,z) from robot telemetry; a camera-side
        # dry run has only the 2-D table (u,v) and no z, so there is no volume to
        # cover. Fail fast rather than silently degrade to a meaningless single slice.
        _log("--coverage-3d needs --with-robot (the z axis comes from EE telemetry)")
        return 2
    world_ids = [int(i) for i in args.world_tag_ids]
    detector = load_detector(args.families)
    K = consumer.camera_matrix
    offset_mm = np.asarray(args.t_eetag_ee, dtype=float)
    ee_size = args.ee_tag_size or args.tag_size
    tag_sizes = {int(i): ee_size for i in ee_ids}
    ee_point_method = args.ee_point_method or "pose"
    # Optional stabilizer tag (e.g. the rigid EE tag 5): record its pose per sample
    # for the wobble diagnostic. Size it (own --stabilizer-tag-size or the EE size)
    # so its pose is scaled right when detected alongside the controlled tag.
    stab_id = int(args.stabilizer_tag_id) if args.stabilizer_tag_id is not None else None
    if stab_id is not None:
        tag_sizes[stab_id] = args.stabilizer_tag_size or ee_size
    _log(f"tag sizes: world {args.tag_size} m, EE {ee_size} m; world tags "
         f"{world_ids}, EE tags {ee_ids}; EE point method: {ee_point_method}"
         + (f"; stabilizer tag {stab_id}" if stab_id is not None else ""))

    # Opt-in 3-D volumetric coverage (WS-4): bin the EE (x,y,z) into voxels so the
    # operator covers the workspace VOLUME, not just one table slice. Default off →
    # the 2-D top-down grid path below is unchanged. Both classes share the public
    # API (add/summary/next_target/done/visited_cells/sufficient_mask), so the rest
    # of the loop is agnostic to which one it holds.
    if args.coverage_3d:
        grid = VoxelCoverage(cell_size_mm=args.cell_size_mm, min_samples=args.min_samples,
                             min_spread_mm=args.min_spread_mm)
    else:
        grid = CoverageGrid(cell_size_mm=args.cell_size_mm, min_samples=args.min_samples,
                            min_spread_mm=args.min_spread_mm)

    # Connect the robot but keep the arm LOCKED through world registration (operator
    # 2026-06-25): a freed arm is limp and can droop into the tags' view, corrupting
    # the world map. It is freed only AFTER the world map is registered — for EE
    # placement and the hand-guided sweep (see below). The `finally` re-locks + homes
    # on every exit, so a freed arm is never left limp.
    link: Optional[HarmonyLink] = None
    arm_is_free = False
    if args.with_robot:
        link = HarmonyLink(args.robot_ip, args.robot_port, args.bind_ip,
                           args.bind_port, side=args.side)
        _log("robot linked — arm stays LOCKED for world registration (keep it clear)")
    else:
        _log("no --with-robot: Q/X will be NaN (camera-side dry run — verifies "
             "(u,v) + coverage on a recorded stream, not a solvable library)")

    world_map = ee_map = None
    uv_rows: List[np.ndarray] = []
    q_rows: List[np.ndarray] = []
    x_rows: List[np.ndarray] = []
    eequat_rows: List[np.ndarray] = []
    tcw_rows: List[np.ndarray] = []
    tce_rows: List[np.ndarray] = []
    tstab_rows: List[np.ndarray] = []   # stabilizer tag pose per accepted sample (NaN if unseen)
    t_rows: List[float] = []
    # WS-1/WS-2b offline prototyping needs the scene frame at each accepted sample
    # (to segment the held/target object); --save-frames writes them as a JPEG
    # sidecar (raw BGR would balloon the npz). frame_rows holds (sample_idx, bgr)
    # only when enabled.
    frame_rows: List[np.ndarray] = []

    period = 1.0 / args.sample_hz
    last_idx = None
    accepted = 0
    # Diagnostics so a pasted sweep log explains itself: why samples were dropped
    # (tag not seen / low margin / time-misaligned) and how reliably the controlled
    # (EE/object) tag was recovered — the operator's "tag had a bad view at some
    # angles" concern shows up here as a low ee-seen fraction + 'ee_not_seen' drops.
    drop_reasons: Dict[str, int] = {}
    n_processed = 0       # frames that reached the accept gate
    ee_seen_frames = 0    # of those, how many recovered the controlled tag pose
    try:
        # World map: LOAD the pre-registered top-down map (preferred — accurate geometry
        # the seated view can't produce) or, as a fallback, register it now seated. The
        # arm stays LOCKED and clear through this (a limp arm can droop into view).
        if args.world_map:
            world_map = _load_world_map(args.world_map)
            world_ids = list(world_map["ids"])  # the loaded map defines the world tags
            _log(f"loaded world map {args.world_map}: ref={world_map['ref_id']}, "
                 f"tags={world_ids}")
            _check_world_geometry(world_map)
        else:
            world_map = _register_map_interactive(consumer, detector, K, world_ids,
                                                  args.tag_size, tag_sizes, "world",
                                                  "arm clear & LOCKED", ui=ui)
            if world_map is None:
                return 1
            # Seated registration is oblique-biased (operator 2026-06-25): the verdict
            # will usually read SKEWED here, which is WHY you should --stage register-world
            # from a top-down view and pass --world-map instead.
            if _check_world_geometry(world_map) == "SKEWED":
                _log("  ⚠ SKEWED from this (seated) view — register top-down once with "
                     "`--stage register-world` and reuse it via --world-map for an "
                     "accurate map. Board PnP recovers the seated pose from that map.")

        # World map is ready — NOW free the arm for EE placement + the hand-guided
        # sweep (operator 2026-06-25: not before, so a limp arm can't droop into the
        # world tags). The `finally` re-locks + auto-homes on every exit from here.
        if args.with_robot:
            if not link.free_arm():
                _log("could not free the arm (ACK:MASTER_FREE not seen) — aborting sweep")
                return 1
            arm_is_free = True
            _log("arm FREED — place the EE for its registration, then hand-guide the sweep")

        ee_map = _register_map_interactive(consumer, detector, K, ee_ids,
                                           args.tag_size, tag_sizes, "EE",
                                           "the EE held STILL", ui=ui)
        if ee_map is None:
            return 1
        ee_ref = ee_map["ref_id"]
        plane_point = world_map["plane_point"]
        plane_normal = world_map["plane_normal"]

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
                ee_quat = rstate.get("ee_quat")
            else:
                q_joints, x_ee, t_robot = np.full(7, np.nan), np.full(3, np.nan), time.time()
                ee_quat = None

            tags = detect_tags(detector, b.video.bgr, K, args.tag_size, tag_sizes=tag_sizes)
            world_view = {i: tags[i] for i in world_ids if i in tags}
            ee_view = {i: tags[i]["T"] for i in ee_ids if i in tags}
            # View-robust world pose: one multi-tag board PnP over the visible tag
            # centres (accurate at the seated oblique angle). Falls back to the per-tag
            # consensus when <4 mapped tags are visible (heavy occlusion).
            T_cam_world = recover_world_pose_pnp(world_view, world_map, K)
            if T_cam_world is None:
                T_cam_world = recover_world_pose(
                    {i: world_view[i]["T"] for i in world_view}, world_map)
            T_cam_eetag = recover_world_pose(ee_view, ee_map)
            margins = ([tags[i]["margin"] for i in world_view]
                       + [tags[i]["margin"] for i in ee_view])
            dt = frame_telemetry_dt(t_frame, t_robot)
            ok, reason = accept_sweep_sample(
                world_seen=T_cam_world is not None, ee_seen=T_cam_eetag is not None,
                margins=margins, dt_s=dt,
                min_margin=args.min_margin, max_align_dt_s=args.max_align_dt_s)
            n_processed += 1
            if T_cam_eetag is not None:
                ee_seen_frames += 1
            if not ok:
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

            cur_uv = None      # vision (u,v) — the gaze-side command coordinate (library)
            disp_xy = None      # what the coverage grid + display use (telemetry base)
            if ok:
                p_world = _ee_point_world(ee_point_method, T_cam_world, T_cam_eetag,
                                          plane_point, plane_normal, offset_mm)
                if p_world is not None:
                    cur_uv = plane_coords(p_world, plane_point, plane_normal)
                    # Telemetry-anchored coverage + display (REV05 §2C, operator
                    # 2026-06-25): bin and draw the robot's OWN end-effector — the
                    # single most consistent metric (no AprilTag depth ambiguity/flips/
                    # head-dependence) — so the screen shows exactly where the arm is.
                    # The command library still stores the vision controlled point
                    # (gaze is the only runtime signal); telemetry anchors coverage/
                    # display/quality, NOT the command. --coverage-3d voxel-bins the
                    # full telemetry EE (x,y,z). No-robot dry run → vision (u,v).
                    if args.coverage_3d:
                        disp_xy = x_ee[:3]
                    else:
                        disp_xy = x_ee[:2] if np.all(np.isfinite(x_ee[:2])) else cur_uv
                    grid.add(disp_xy)
                    uv_rows.append(cur_uv)
                    q_rows.append(q_joints)
                    x_rows.append(x_ee)
                    eequat_rows.append(ee_quat if ee_quat is not None
                                       else np.full(4, np.nan))
                    tcw_rows.append(T_cam_world)
                    tce_rows.append(T_cam_eetag)
                    t_rows.append(t_robot)
                    if stab_id is not None:
                        tstab_rows.append(tags[stab_id]["T"] if stab_id in tags
                                          else np.full((4, 4), np.nan))
                    if args.save_frames:
                        frame_rows.append(b.video.bgr.copy())
                    accepted += 1

            target = grid.next_target()
            if ui is not None:
                ui.update(grid, disp_xy, target)
                if ui.should_quit():
                    _log("sweep: quit requested from the coverage view")
                    break

            now = time.monotonic()
            if now - last_report >= 1.0:
                s = grid.summary()
                tgt = "—" if target is None else np.round(target, 0).tolist()
                label = "voxels" if args.coverage_3d else "cells"
                ee_pct = (100.0 * ee_seen_frames / n_processed) if n_processed else 0.0
                _log(f"  accepted={accepted} {label}={int(s['visited'])} "
                     f"sufficient={int(s['sufficient'])} ee-tag seen {ee_pct:.0f}% → go to {tgt}"
                     + ("" if ok else f"  (last drop: {reason})"))
                if args.coverage_3d:
                    # The volumetric analog of the 2-D box: a per-z-slice occupancy
                    # projection so the operator can see which heights are still thin.
                    _log(f"    z-slices (sufficient/visited): {grid.status_text()}")
                last_report = now

            # Stopping is operator-driven by default (SPACE in the coverage window /
            # Ctrl-C): the coverage extent the operator wants is theirs to judge, and
            # the automatic "every visited cell sufficient" stop fired far too early
            # (5 cells, a small patch — operator 2026-06-24). Opt back into the
            # automatic stop with --auto-stop; it still needs ≥ min_cells covered so
            # one dwelt cell cannot end it (critic I1).
            if (args.auto_stop and grid.done()
                    and len(grid.visited_cells()) >= args.min_cells):
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

    # Sweep diagnostics (always — a short/failed sweep still explains itself).
    ee_pct = (100.0 * ee_seen_frames / n_processed) if n_processed else 0.0
    drops = (", ".join(f"{r}={c}" for r, c
                       in sorted(drop_reasons.items(), key=lambda kv: -kv[1])) or "none")
    _log(f"sweep summary: {n_processed} frames processed, {accepted} accepted; "
         f"controlled tag {ee_ids} recovered in {ee_pct:.0f}% of frames")
    _log(f"  drops by reason: {drops}  "
         "(ee_not_seen ⇒ the tracked tag was occluded/oblique → re-orient it; "
         "low_margin ⇒ tag too small/blurred; misaligned ⇒ slow down)")
    if x_rows:
        X = np.array(x_rows, dtype=float)
        fin = X[np.all(np.isfinite(X[:, :3]), axis=1)] if X.ndim == 2 and X.shape[1] >= 3 else X[:0]
        if fin.shape[0]:
            _log(f"  arm reach (telemetry EE) mm: "
                 f"x[{fin[:,0].min():.0f},{fin[:,0].max():.0f}] "
                 f"y[{fin[:,1].min():.0f},{fin[:,1].max():.0f}] "
                 f"z[{fin[:,2].min():.0f},{fin[:,2].max():.0f}]")

    # Stabilizer wobble: over samples where BOTH the controlled tag and the
    # stabilizer (rigid EE) tag were seen, the spread of the controlled↔stabilizer
    # offset IS the grip/hand-wobble error. Small ⇒ the controlled tag alone is
    # rigid enough; large ⇒ rebuild the library from stabilizer + the solved offset.
    if stab_id is not None and tstab_rows:
        offs = []
        for tce, tstab in zip(tce_rows, tstab_rows):
            if np.all(np.isfinite(tce)) and np.all(np.isfinite(tstab)):
                offs.append((invert_transform(np.asarray(tstab, float))
                             @ np.asarray(tce, float))[:3, 3])
        if len(offs) >= 2:
            O = np.array(offs)
            sd = O.std(axis=0)
            _log(f"  stabilizer tag {stab_id}: both tags seen in {len(offs)}/{len(tce_rows)} "
                 f"samples; controlled↔stabilizer offset mean={np.round(O.mean(axis=0),1).tolist()} "
                 f"mm, wobble std=[{sd[0]:.1f},{sd[1]:.1f},{sd[2]:.1f}] mm "
                 f"(|sd|={float(np.linalg.norm(sd)):.1f}); small ⇒ rigid, large ⇒ use "
                 "stabilizer+offset for the library)")
        else:
            _log(f"  stabilizer tag {stab_id}: <2 samples with both tags visible "
                 "— couldn't measure wobble (keep it in view alongside the controlled tag)")

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
    # otherwise pollute the (u,v)→Q nearest-neighbour lookup. Coverage is in the SAME
    # frame the grid was fed during the sweep — telemetry base (x,y) with a robot, else
    # vision (u,v) — so the mask matches what the operator saw fill in.
    x_arr = np.vstack(x_rows)
    if args.coverage_3d:
        # Voxel coverage was fed the EE (x,y,z); recompute green over the same 3-D
        # points so the mask matches the voxels the operator saw fill in.
        cov_pts = x_arr[:, :3]
    else:
        cov_pts = (x_arr[:, :2] if np.all(np.isfinite(x_arr[:, :2]))
                   else np.vstack(uv_rows))
    green = grid.sufficient_mask(cov_pts)
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
        # How each sample's EE (u,v) was derived (rev04, 2026-06-24): 'pose' = the EE
        # tag's 3-D pose origin + offset, orthogonally projected (default, most head-
        # invariant once the world frame is consensus-stable); 'rayplane' = tag-centre
        # line of sight ∩ plane (depth-ambiguity-free but parallax-prone).
        "ee_point_method": ee_point_method,
        "with_robot": bool(args.with_robot),
        "sample_hz": args.sample_hz,
        "coverage": {"cell_size_mm": args.cell_size_mm,
                     "min_samples": args.min_samples,
                     "min_spread_mm": args.min_spread_mm,
                     "complete": bool(grid.done())},
        "units": {"UV": "mm (table plane)", "X": "mm", "Q": "rad",
                  "t_eetag_ee": "mm", "EEQUAT": "[x,y,z,w] scalar-last"},
    }
    if args.coverage_3d:
        # Record the volumetric mode only when used, so a default (2-D) sweep's saved
        # meta is byte-identical to before this feature; green was computed over (x,y,z).
        meta["coverage"]["mode"] = "voxel_3d"
    # WS-1/WS-2b scene-frame sidecar: JPEG per accepted sample, index-aligned with
    # UV/Q/X/EEQUAT rows. Kept out of the npz (raw BGR is ~6 MB/frame); the offline
    # scripts join by row index via the recorded directory name.
    frames_dir = None
    if args.save_frames and frame_rows:
        import cv2
        frames_dir = out_dir / f"apriltag_sweep_{stamp}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, bgr in enumerate(frame_rows):
            cv2.imwrite(str(frames_dir / f"{i:05d}.jpg"), bgr,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        _log(f"saved {len(frame_rows)} scene frames → {frames_dir}")
    meta["frames_dir"] = frames_dir.name if frames_dir is not None else None
    meta["stabilizer_tag_id"] = stab_id
    # Stabilizer poses (index-aligned with T_cam_eetag) only when recorded, so a
    # default sweep's npz keys are unchanged. Lets an offline solve rebuild the
    # library from the rigid tag + its solved offset without re-capturing.
    stab_save = ({"T_cam_stab": np.stack(tstab_rows)}
                 if stab_id is not None and tstab_rows else {})
    np.savez_compressed(
        out_path,
        UV=np.vstack(uv_rows),
        Q=np.vstack(q_rows),
        X=np.vstack(x_rows),
        EEQUAT=np.vstack(eequat_rows),
        K=np.asarray(K, dtype=float),
        green=green,
        T_cam_world=np.stack(tcw_rows),
        T_cam_eetag=np.stack(tce_rows),
        t=np.asarray(t_rows, dtype=float),
        world_map_ref=np.array(wm_ref), world_map_ids=wm_ids, world_map_rels=wm_rels,
        world_map_plane_point=wm_pp, world_map_plane_normal=wm_pn,
        ee_map_ref=np.array(em_ref), ee_map_ids=em_ids, ee_map_rels=em_rels,
        ee_map_plane_point=em_pp, ee_map_plane_normal=em_pn,
        meta=np.array(meta, dtype=object),
        **stab_save,
    )
    _log(f"saved {len(uv_rows)} swept samples ({int(green.sum())} in green/sufficient "
         f"cells) → {out_path}")
    if args.with_robot and args.then_solve:
        # Wrap solve into calibration (the control-panel "calibration" button): the
        # sweep already baked the EE method, plane (orientation normal), and offset
        # into the stored (u,v), so solve from those — no recompute.
        _log("--then-solve: building the planar calibration from this sweep …")
        args.npz, args.ee_point_method = str(out_path), None
        return stage_solve(args)
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
    """The sweep's coverage view, unless ``--no-ui`` (headless, text summary only).

    ``register-world-3d`` gets the live ``RegistrationView`` (per-tag residual +
    viewpoint-diversity quality, operator-driven finish). ``--coverage-3d`` on the
    sweep gets the per-z-slice volumetric view (``VoxelCoverageBoxUI``); otherwise
    the 2-D top-down box (rev04 §3). The 2-D ``register-world`` stage still uses the
    box (for its SPACE prompts)."""
    if args.no_ui:
        return None
    if args.stage == "register-world-3d":
        from Utils.gaze.registration_view import RegistrationView
        return RegistrationView(args.world_tag_ids or [])
    if getattr(args, "coverage_3d", False) and args.stage == "sweep":
        from Utils.gaze.coverage_voxel_view import VoxelCoverageBoxUI
        return VoxelCoverageBoxUI(args.cell_size_mm, audio=args.audio)
    from Utils.gaze.coverage_view import CoverageBoxUI
    return CoverageBoxUI(args.cell_size_mm, audio=args.audio)


def _ee_point_world(method, T_cam_world, T_cam_eetag, plane_point, plane_normal, offset_mm):
    """The EE point in the world frame by the selected method (rev04, 2026-06-24).

    'pose' (default): the EE tag's 3-D pose origin + the measured offset, which
    `plane_coords` then projects orthogonally onto the table — the EE's true table
    position. This is correct (runtime gaze gives the true table target) and the
    most head-invariant choice once the world frame is stabilised by the consensus
    fusion (HIL repeatability 12 mm).

    'rayplane': the tag-centre line of sight ∩ table plane (depth-ambiguity-free).
    Robust to the single-tag flip but adds a head-angle-dependent parallax; it only
    wins when the EE/world pose is still flip-prone. Returns None when the ray
    misses the plane (drop the sample); 'pose' never returns None."""
    if method == "rayplane":
        return eetag_rayplane_point_world(T_cam_world, T_cam_eetag, plane_point, plane_normal)
    return eetag_to_world_point(T_cam_world, T_cam_eetag, offset_mm)


def _recompute_planar_uv(z, method, offset):
    """Re-derive every sweep sample's table ``(u,v)`` from the stored per-sample
    transforms (``T_cam_world`` / ``T_cam_eetag``) using ``method`` and the EE-tag→
    hand ``offset`` (tag frame, mm) — AND recompute the table plane from the stored
    world map's tag ORIENTATIONS (`table_normal_from_rel`), correcting an old
    origin-fit plane (idempotent for an orientation-built map). The green/sufficient
    mask is recomputed too (the cells move with the points). Returns
    ``(UV, green, plane_point, plane_normal)`` — the corrected plane is written into
    the calib so the runtime gaze uses the same one. Lets a captured sweep be
    re-solved a different way (EE method, plane fix, measured offset), no rig."""
    Tcw = np.asarray(z["T_cam_world"], dtype=float)
    Tce = np.asarray(z["T_cam_eetag"], dtype=float)
    rels = np.asarray(z["world_map_rels"], dtype=float)
    ids = [int(i) for i in np.asarray(z["world_map_ids"]).ravel()]
    rel = {ids[k]: rels[k] for k in range(len(ids))}
    if len(ids) >= 3:
        pp = np.mean([rel[i][:3, 3] for i in ids], axis=0)
        pn = table_normal_from_rel(rel, ids)
    else:
        pp = np.asarray(z["world_map_plane_point"], dtype=float)
        pn = np.asarray(z["world_map_plane_normal"], dtype=float)
    offset = np.asarray(offset, dtype=float)
    pts = [(_ee_point_world(method, Tcw[i], Tce[i], pp, pn, offset)) for i in range(len(Tcw))]
    pts = np.array([p if p is not None else np.full(3, np.nan) for p in pts])
    UV = plane_coords(pts, pp, pn)
    meta = z["meta"].item() if "meta" in z.files else {}
    cov = meta.get("coverage", {})
    grid = CoverageGrid(cell_size_mm=cov.get("cell_size_mm", 50.0),
                        min_samples=cov.get("min_samples", 8),
                        min_spread_mm=cov.get("min_spread_mm", 15.0))
    # Coverage is computed in telemetry base (x,y) when present — the same physical,
    # EE-method-independent frame the sweep used (so re-solving with a different EE
    # method doesn't move the cells). Falls back to vision (u,v) for a no-robot capture.
    X = np.asarray(z["X"], dtype=float) if "X" in z.files else None
    use_base = (X is not None and X.ndim == 2 and X.shape[1] >= 2
                and bool(np.isfinite(X[:, :2]).all()))
    cov_pts = X[:, :2] if use_base else UV
    finite = np.all(np.isfinite(UV), axis=1) & np.all(np.isfinite(cov_pts), axis=1)
    for p in cov_pts[finite]:
        grid.add(p)
    green = np.zeros(len(UV), dtype=bool)
    green[finite] = grid.sufficient_mask(cov_pts[finite])
    return UV, green, pp, pn


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

    # Optionally re-derive (u,v) with a different EE-point method from the stored
    # per-sample transforms (rev04, 2026-06-24): the npz carries T_cam_world /
    # T_cam_eetag + the world plane, so a sweep captured one way can be re-solved
    # another way (e.g. tag 'pose' once consensus stabilised the world frame) with no
    # re-sweep. The coverage/green mask is recomputed too, since the cells move.
    green_src = np.asarray(z["green"], dtype=bool) if "green" in z.files else None
    plane_fix = None  # (plane_point, plane_normal) to override in the calib, if recomputed
    # The EE-point method/offset the OUTPUT library is actually built with — recorded
    # into the calib meta below. Default to the sweep's (no recompute = library is the
    # sweep's stored (u,v)); overwritten when we re-derive. Without recording these the
    # calib meta silently inherited the sweep's values and MISREPORTED a re-solved
    # library (e.g. a rayplane/0 sweep re-solved to pose+offset still read rayplane/0),
    # which cost a debugging session 2026-06-25. Read the meta, trust the meta.
    src_meta_for_recipe = z["meta"].item() if "meta" in z.files else {}
    applied_method = src_meta_for_recipe.get("ee_point_method")
    applied_offset = src_meta_for_recipe.get("t_eetag_ee_mm", [0.0, 0.0, 0.0])
    if args.ee_point_method is not None and "T_cam_world" in z.files:
        # EE-tag→hand offset (tag frame, mm): an explicit --t-eetag-ee overrides the
        # zero the sweep stored, so a measured mount offset can be applied on re-solve.
        meta0 = src_meta_for_recipe
        offset = np.asarray(args.t_eetag_ee, dtype=float)
        if not offset.any():
            offset = np.asarray(meta0.get("t_eetag_ee_mm", [0.0, 0.0, 0.0]), dtype=float)
        UV, green_src, _pp_fix, _pn_fix = _recompute_planar_uv(z, args.ee_point_method, offset)
        plane_fix = (_pp_fix, _pn_fix)
        applied_method, applied_offset = args.ee_point_method, offset.tolist()
        _log(f"recomputed (u,v): ee-point-method={args.ee_point_method}, "
             f"t_eetag_ee={offset.tolist()} mm (tag frame); table normal from tag "
             f"orientations = {np.round(_pn_fix, 3).tolist()}")

    # Green-only (rev04 §3, operator 2026-06-24): build the library from samples in
    # sufficient ("green") coverage cells only — the default, since transit /
    # extraneous samples in still-partial cells pollute the NN lookup. A pre-fix npz
    # lacks the mask (then we cannot filter and keep all). ``--include-partial``
    # keeps every sample.
    if args.green_only and green_src is not None:
        gmask = green_src
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
                       "source_capture": npz_path.name,
                       # Record the recipe the LIBRARY was actually built with (not the
                       # sweep's) so the meta never lies about a re-solved calib again.
                       "ee_point_method": applied_method,
                       "t_eetag_ee_mm": applied_offset})
    out_path = npz_path.with_name(npz_path.stem.replace("apriltag_sweep", "apriltag")
                                  + "_calib.npz")
    # Carry the world map + plane through so the control tool recovers the SAME
    # world frame and the SAME (u,v) basis used at capture.
    extra = {}
    for k in ("world_map_ref", "world_map_ids", "world_map_rels",
              "world_map_plane_point", "world_map_plane_normal"):
        if k in z.files:
            extra[k] = z[k]
    if plane_fix is not None:
        # The (u,v) were rebuilt on the orientation-derived plane; the runtime gaze
        # must use the SAME plane, so override the stored (possibly origin-fit) one.
        extra["world_map_plane_point"], extra["world_map_plane_normal"] = plane_fix
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


# ── stage: solve-3d (WS-4 world-frame (x,y,z)→Q library) ─────────────────────


def stage_solve_3d(args) -> int:
    """WS-4 3-D solve: build a world-frame ``(x,y,z)→Q`` library from a sweep npz.

    Mirrors ``stage_solve_planar`` but keeps the EE point in FULL 3-D — it omits the
    table-plane projection (``plane_coords``) the planar solve applies. Each accepted
    sample's ``p_world = eetag_to_world_point(T_cam_world, T_cam_eetag, offset)`` is
    the SAME world EE point the planar path computes; the planar path then flattens it
    to ``(u,v)`` on the table, here it is stored as-is so the table-height ``z`` is
    retained. No plane projection → 3-D: objects above the table become reachable.

    Self-contained in the WORLD frame: the runtime 3-D target (gaze→object centroid
    in world frame) is queried against this library by ``GazeCalibration3D``, so no
    robot base-frame / hand-eye transform is needed. (The planar solve's A2 base-xy
    similarity is a 2-D-only diagnostic with a non-coplanarity degeneracy, so it has
    no 3-D analogue and is dropped.)

    EE-point method + mount offset come from the sweep's meta (the recipe the sweep
    baked in), exactly as ``--then-solve`` relies on for the planar path. Green-cell
    filtering matches the planar solve. Writes ``apriltag_3d_<UTC>_calib.npz`` with
    ``P_WORLD3D`` (N,3) + ``Q`` (N,7) + the world-map arrays, loadable by
    ``GazeCalibration3D.from_calib_npz``."""
    npz_path = Path(args.npz)
    if not npz_path.is_file():
        _log(f"npz not found: {npz_path}")
        return 2
    z = np.load(npz_path, allow_pickle=True)
    if "T_cam_world" not in z.files or "T_cam_eetag" not in z.files:
        _log(f"{npz_path.name} lacks T_cam_world/T_cam_eetag — not a sweep capture. "
             "solve-3d recomputes the 3-D EE world point from the per-sample transforms.")
        return 2

    Tcw = np.asarray(z["T_cam_world"], dtype=float)
    Tce = np.asarray(z["T_cam_eetag"], dtype=float)
    Q = np.asarray(z["Q"], dtype=float)
    if Tcw.shape[0] != Tce.shape[0] or Tcw.shape[0] != Q.shape[0]:
        _log(f"row mismatch: T_cam_world {Tcw.shape[0]}, T_cam_eetag {Tce.shape[0]}, "
             f"Q {Q.shape[0]} — corrupt sweep npz")
        return 2

    # Recipe from the sweep's meta — the EE-point method and the hand-measured mount
    # offset the sweep recorded. Read the meta, trust the meta (the planar solve learnt
    # the same lesson, 2026-06-25): the calib must report the recipe it was built with.
    src_meta = z["meta"].item() if "meta" in z.files else {}
    method = src_meta.get("ee_point_method", "pose")
    offset = np.asarray(src_meta.get("t_eetag_ee_mm", [0.0, 0.0, 0.0]), dtype=float)
    # Re-solve override: an explicit --ee-point-method takes BOTH the method and the
    # --t-eetag-ee offset from the CLI verbatim (a literal [0,0,0] IS honoured — e.g.
    # to put the controlled point at the EE tag), so a sweep can be re-solved with a
    # corrected mount/grasp offset without re-capturing. The calib meta below records
    # whatever was actually applied. Mirrors the planar solve's override.
    if args.ee_point_method is not None:
        method = args.ee_point_method
        offset = np.asarray(args.t_eetag_ee, dtype=float)
        _log(f"re-solve override: ee_point_method={method}, "
             f"t_eetag_ee_mm={offset.tolist()} (controlled point = EE tag + this offset)")
    # A 'rayplane' sweep PROJECTS every EE point onto the table plane
    # (eetag_rayplane_point_world), so the z height — the whole reason for a 3-D
    # library — is already destroyed in the source geometry. Building a 3-D calib
    # from it would yield a coplanar library mislabeled scheme="world_xyz_nn" that
    # the 3-D control loop would trust. Refuse: a 3-D library needs a 'pose' sweep.
    if method == "rayplane":
        _log("solve-3d: this sweep used ee_point_method='rayplane', which projects "
             "EE points onto the table plane — there is no z to build a 3-D library "
             "from. Re-capture/solve with the 'pose' method (config "
             "APRILTAG_EE_POINT_METHOD='pose').")
        return 2
    # The world plane is only consulted by the 'rayplane' EE method; the default
    # 'pose' ignores it. Read whatever the sweep stored (fall back to z=0 / +Z).
    pp = (np.asarray(z["world_map_plane_point"], dtype=float)
          if "world_map_plane_point" in z.files else np.zeros(3))
    pn = (np.asarray(z["world_map_plane_normal"], dtype=float)
          if "world_map_plane_normal" in z.files else np.array([0.0, 0.0, 1.0]))

    # The EE world point in FULL 3-D — the SAME p_world the planar sweep computed
    # via _ee_point_world, but WITHOUT the subsequent plane_coords projection. For the
    # default 'pose' method this is exactly eetag_to_world_point(...); the table-height
    # z survives, which is the entire point of the 3-D path.
    pts = [_ee_point_world(method, Tcw[i], Tce[i], pp, pn, offset) for i in range(len(Tcw))]
    P_world3d = np.array([p if p is not None else np.full(3, np.nan) for p in pts],
                         dtype=float)

    # Green-only filtering — identical policy to the planar solve (rev04 §3): transit
    # samples in still-partial coverage cells pollute the NN lookup. A pre-mask npz
    # keeps all; --include-partial keeps all.
    green_src = np.asarray(z["green"], dtype=bool) if "green" in z.files else None
    if args.green_only and green_src is not None:
        kept = int(green_src.sum())
        _log(f"green-only: {kept}/{len(green_src)} samples in sufficient cells "
             f"(dropped {len(green_src) - kept} partial-cell; --include-partial keeps all)")
        P_world3d, Q = P_world3d[green_src], Q[green_src]
    elif args.green_only:
        _log("green-only requested but this npz has no 'green' mask (pre-fix sweep) "
             "— using all accepted samples")
    else:
        _log("--include-partial: using all accepted samples (green + amber cells)")

    # The library needs finite (p_world, Q); a no-robot dry-run sweep has NaN Q, and a
    # rayplane miss leaves a NaN point.
    lib_ok = np.all(np.isfinite(P_world3d), axis=1) & np.all(np.isfinite(Q[:, :7]), axis=1)
    P_world3d, Q = P_world3d[lib_ok], Q[lib_ok]
    n = P_world3d.shape[0]
    if n < 3:
        _log(f"only {n} finite (P_WORLD3D,Q) rows (<3) — cannot build a 3-D library. "
             "Was --with-robot used during the sweep?")
        return 1
    z_ptp = float(np.ptp(P_world3d[:, 2]))
    _log(f"3-D library: {n} (x,y,z)→Q rows; "
         f"x∈[{P_world3d[:,0].min():.0f},{P_world3d[:,0].max():.0f}] "
         f"y∈[{P_world3d[:,1].min():.0f},{P_world3d[:,1].max():.0f}] "
         f"z∈[{P_world3d[:,2].min():.0f},{P_world3d[:,2].max():.0f}] mm")
    _log(f"z range = {z_ptp:.0f} mm (≈0 would mean a flat/planar sweep — the planar "
         "solve would then suffice; a 3-D library wants real height variation)")

    calib_meta = dict(src_meta)
    calib_meta.update({
        "scheme": "world_xyz_nn",
        "n_points": int(n),
        "source_capture": npz_path.name,
        # The recipe the LIBRARY was actually built with (from the sweep meta).
        "ee_point_method": method,
        "t_eetag_ee_mm": offset.tolist(),
        "z_range_mm": z_ptp,
        "units": {"P_WORLD3D": "mm (world frame, full 3-D)", "Q": "rad",
                  "t_eetag_ee": "mm"},
    })
    out_path = npz_path.with_name(
        npz_path.stem.replace("apriltag_sweep", "apriltag_3d") + "_calib.npz")
    # Carry the world map through so the runtime recovers the SAME world frame the
    # library points live in (the gaze→object 3-D target is expressed in it).
    extra = {}
    for k in ("world_map_ref", "world_map_ids", "world_map_rels",
              "world_map_plane_point", "world_map_plane_normal"):
        if k in z.files:
            extra[k] = z[k]
    np.savez_compressed(
        out_path,
        P_WORLD3D=P_world3d, Q=Q,
        meta=np.array(calib_meta, dtype=object),
        **extra,
    )
    _log(f"3-D calibration → {out_path}")
    _log("consume it (live loop): GazeCalibration3D.from_calib_npz(np.load(...))")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    import config as cfg
    p = argparse.ArgumentParser(description="AprilTag gaze↔robot calibration tool")
    p.add_argument("--stage", required=True,
                   choices=["detect", "gaze", "collect", "sweep", "solve",
                            "register-world", "register-world-3d", "solve-3d"])
    p.add_argument("npz", nargs="?", default=None,
                   help="solve stage: path to an apriltag_capture_*.npz")
    p.add_argument("--world-map", default=None,
                   help="sweep stage: load a pre-registered world map (from "
                        "--stage register-world) instead of registering seated. "
                        "Register the static tags ONCE from a top-down view (where the "
                        "geometry is accurate) and reuse it across seated sessions.")
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
    p.add_argument("--table-plane-tag-ids", type=int, nargs="+", default=None,
                   help="register-world-3d: tag ids physically coplanar ON THE TABLE. "
                        "Given (with --wall-plane-tag-ids), the map is hardened — these "
                        "tags are snapped coplanar and the table⊥wall planes squared to "
                        "90° (fixes the single-tag depth ambiguity). Off = pure fuse.")
    p.add_argument("--wall-plane-tag-ids", type=int, nargs="+", default=None,
                   help="register-world-3d: tag ids coplanar ON THE WALL (perpendicular "
                        "to the table). See --table-plane-tag-ids.")
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
    p.add_argument("--stabilizer-tag-id", type=int, default=None,
                   help="sweep: also record this tag's pose per sample (e.g. the rigid "
                        "EE tag 5 while the controlled tag 18 is hand-held). Logs the "
                        "controlled↔stabilizer offset wobble and saves the poses so an "
                        "offline solve can rebuild the library from the rigid tag. "
                        "Diagnostic only — does not change the captured library.")
    p.add_argument("--stabilizer-tag-size", type=float, default=None,
                   help="stabilizer tag edge length in METRES (default: --ee-tag-size)")
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
    p.add_argument("--save-frames", action="store_true",
                   help="sweep: also save the scene frame at each accepted sample as a "
                        "JPEG sidecar dir (WS-1/WS-2b offline object segmentation). Off "
                        "by default — adds ~100s of MB per sweep.")
    p.add_argument("--utc-stamp", default=None,
                   help="collect/sweep: override the auto UTC stamp in the npz filename")
    # sweep stage (REV04 continuous capture + coverage)
    p.add_argument("--sample-hz", type=float, default=20.0,
                   help="sweep: telemetry/frame sampling rate (REV01 transit rate)")
    p.add_argument("--max-sweep-s", type=float, default=600.0,
                   help="sweep: hard time budget (safety backstop); the operator "
                        "normally ends the sweep with SPACE in the coverage window")
    p.add_argument("--auto-stop", action="store_true",
                   help="sweep: also stop AUTOMATICALLY once every visited cell is "
                        "sufficient (≥ --min-cells). Off by default — the operator "
                        "judges coverage and ends with SPACE (operator 2026-06-24)")
    p.add_argument("--cell-size-mm", type=float, default=50.0,
                   help="sweep: coverage cell size on the table plane (mm)")
    p.add_argument("--min-cells", type=int, default=4,
                   help="sweep: with --auto-stop, the minimum covered cells before the "
                        "automatic stop may fire (guards against ending after one cell)")
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
    p.add_argument("--coverage-3d", action="store_true",
                   help="sweep: track 3-D VOLUMETRIC coverage (x,y,z voxels) instead of "
                        "the 2-D top-down grid — bins the EE (x,y,z) telemetry so the "
                        "operator knows when the workspace VOLUME (not one slice) is "
                        "covered. Runs headless with a per-z-slice text summary (no 3-D "
                        "OpenCV view yet); requires --with-robot for the z axis. Default "
                        "off → the 2-D grid path is unchanged.")
    p.add_argument("--no-ui", action="store_true",
                   help="sweep: run headless with a text summary (no OpenCV coverage box)")
    p.add_argument("--audio", action="store_true",
                   help="sweep: speak coverage cues (solo-operator aid, rev04 §3)")
    p.add_argument("--then-solve", action="store_true",
                   help="sweep: immediately run the planar solve on the npz just saved "
                        "(wraps calibration → solve in one terminal; the control panel "
                        "uses this). No-op without --with-robot (a dry run has NaN Q).")
    p.add_argument("--ee-point-method", choices=["pose", "rayplane"], default=None,
                   help="how each sample's EE table-(u,v) is derived. sweep: default "
                        "'pose' (the EE tag's 3-D origin projected onto the table — "
                        "most head-invariant once the world frame is consensus-stable); "
                        "'rayplane' = tag-centre line of sight ∩ plane (ambiguity-free "
                        "but parallax-prone). solve: when given, RE-derives (u,v) from "
                        "the stored transforms with this method (re-solve a sweep a "
                        "different way, no re-sweep); omit to use the stored (u,v).")
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
    if args.stage == "solve-3d":
        if not args.npz:
            _log("solve-3d needs an npz path argument")
            return 2
        return stage_solve_3d(args)

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
        if args.stage == "register-world":
            return stage_register_world(args, consumer, ui=_make_coverage_ui(args))
        if args.stage == "register-world-3d":
            return stage_register_world_3d(args, consumer, ui=_make_coverage_ui(args))
        if args.stage == "sweep":
            return stage_sweep(args, consumer, ui=_make_coverage_ui(args))
    finally:
        consumer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
