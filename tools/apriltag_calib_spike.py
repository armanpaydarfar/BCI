#!/usr/bin/env python3
"""
apriltag_calib_spike.py — WS5 REV03 AprilTag gaze↔robot de-risking spike.

Methodology: SoftwareDocs/projects/harmony-bci/gaze-calibration/
rev03-apriltag-methodology.md §7. This is a **Tier-3 standalone bench tool**.
It is **read-only with respect to the robot** — it sends only `q;seq` telemetry
queries and **never any motion opcode** (no `m`/`c`/`g`/`h`/coords). It cannot
move the arm.

It consumes camera frames + gaze + intrinsics from the existing frame relay
(`Utils.remote_frame_reader.RemoteFrameReader`), so it runs in the `lsl` env and
needs no `harmony_vlm`/`perception` import. The panel's embedded relay (or
`python -m Utils.frame_relay`) must be up first — it is the sole Neon subscriber.

Four stages (`--stage`):

    detect   camera-only. Per-tag detection rate, decision-margin/hamming, and
             pose jitter (translation mm / rotation deg) over a static window.
    gaze     camera-only, operator fixates the tag. Angular error between the
             gaze ray and the ray to the recovered tag centre.
    collect  + robot. Operator-gated capture of paired
             (P_world from the EE-tag, P_base from `q;seq` telemetry). Saves an
             apriltag_spike_<UTC>.npz. EE-mounted-tag free-roam scheme (§4.2).
    solve    offline. Umeyama rigid fit of T_base_world from a saved npz, with
             leave-one-out cross-validation. No hardware.

`pupil-apriltags` is imported lazily; detect/gaze/collect fail fast with a
`pip install pupil-apriltags` remediation if it is absent (it is intentionally
not yet in environment.yml — added only once the spike is adopted, §6.4). The
offline `solve` stage needs only numpy.

Examples:
    python tools/apriltag_calib_spike.py --stage detect --world-tag-id 0 --tag-size 0.06
    python tools/apriltag_calib_spike.py --stage gaze   --world-tag-id 0 --tag-size 0.06
    python tools/apriltag_calib_spike.py --stage collect --with-robot \\
        --world-tag-id 0 --ee-tag-id 1 --tag-size 0.06 --t-eetag-ee 0 0 50
    python tools/apriltag_calib_spike.py --stage solve runs/apriltag_spike_20260620T1200Z.npz
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from Utils.gaze.apriltag_calib import (  # noqa: E402
    angle_between_deg,
    ee_point_in_world,
    gaze_ray_cam,
    invert_transform,
    make_transform,
    per_point_errors,
    umeyama_rigid,
)

_M_TO_MM = 1000.0


def _log(msg: str) -> None:
    print(f"[apriltag_spike] {msg}", flush=True)


# ── AprilTag detection (lazy pupil-apriltags) ────────────────────────────────


def _load_detector(families: str):
    """Build a pupil-apriltags Detector, failing fast with a remediation
    message if the (intentionally not-yet-vendored) dep is missing."""
    try:
        from pupil_apriltags import Detector
    except ImportError as exc:
        raise SystemExit(
            "pupil-apriltags is required for this stage but is not installed.\n"
            "  pip install pupil-apriltags\n"
            "(It is deliberately not in environment.yml until WS5 is adopted — "
            "see rev03-apriltag-methodology.md §6.4.)"
        ) from exc
    return Detector(families=families)


def _camera_params(K: np.ndarray) -> Tuple[float, float, float, float]:
    """(fx, fy, cx, cy) for pupil-apriltags from the 3×3 intrinsics."""
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def _detect_tags(detector, bgr: np.ndarray, K: np.ndarray,
                 tag_size_m: float) -> Dict[int, Dict]:
    """Detect tags and return {tag_id: {T (4×4, mm), margin, hamming, center}}.
    Tag-pose translation is converted metres→mm so it shares units with robot
    telemetry and the X calibration column."""
    import cv2  # lazy: only the camera stages need it
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    results = detector.detect(
        gray, estimate_tag_pose=True,
        camera_params=_camera_params(K), tag_size=tag_size_m,
    )
    out: Dict[int, Dict] = {}
    for r in results:
        if r.pose_R is None or r.pose_t is None:
            continue
        T = make_transform(np.asarray(r.pose_R, dtype=float),
                           np.asarray(r.pose_t, dtype=float).ravel() * _M_TO_MM)
        out[int(r.tag_id)] = {
            "T": T,
            "margin": float(r.decision_margin),
            "hamming": int(r.hamming),
            "center": np.asarray(r.center, dtype=float),
        }
    return out


# ── relay consumer (background latest-bundle thread) ─────────────────────────


class _RelayConsumer:
    """Owns a RemoteFrameReader and keeps the latest bundle available on demand.
    The reader's __iter__ blocks, so a daemon thread drains it; stages poll
    ``latest()`` and dedup on ``frame_idx``."""

    def __init__(self, host: str, port: int, handshake_s: float = 5.0):
        from Utils.remote_frame_reader import RemoteFrameReader
        self._reader = RemoteFrameReader(
            host, port, wait_for_handshake_s=handshake_s, auto_reconnect=False)
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="relay-consumer",
                                        daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            for bundle in self._reader:
                if self._stop.is_set():
                    break
                with self._lock:
                    self._latest = bundle
        except Exception as exc:  # surface, don't swallow: the relay died
            _log(f"relay consumer stopped: {exc!r}")

    def latest(self):
        with self._lock:
            return self._latest

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.asarray(self._reader.camera_matrix, dtype=float)

    def close(self) -> None:
        self._stop.set()
        try:
            self._reader.close()
        except Exception:
            pass


# ── read-only robot telemetry (q;seq ONLY — no motion opcodes) ────────────────


class RobotTelemetry:
    """Minimal read-only telemetry client. Sends `q;seq=<n>` and parses the EE
    position (`eeR/eeL.pos_mm`, mm, base frame). It sends **no** motion opcode —
    by construction it cannot command the arm.

    Binds the same control endpoint the recorder uses, so it must not run
    alongside `harmony_free_arm_calibration.py` (one binder per port)."""

    def __init__(self, dial_ip: str, dial_port: int, bind_ip: str,
                 bind_port: int, side: str = "R", timeout_s: float = 0.5):
        self._addr = (dial_ip, int(dial_port))
        self._side = side.upper()
        self._seq = 0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((bind_ip, int(bind_port)))
        self._sock.settimeout(timeout_s)

    def query_ee_mm(self) -> Optional[np.ndarray]:
        """Return the active-side EE position (3,) mm, or None on timeout/parse
        failure. ``q;seq`` is a pure telemetry read — no side effects."""
        self._seq += 1
        self._sock.sendto(f"q;seq={self._seq}".encode("ascii"), self._addr)
        key_ee = "eeR" if self._side == "R" else "eeL"
        deadline = time.time() + 1.0
        while time.time() < deadline:
            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                return None
            try:
                pkt = json.loads(data.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            if isinstance(pkt, dict) and key_ee in pkt \
                    and isinstance(pkt[key_ee], dict) and "pos_mm" in pkt[key_ee]:
                return np.asarray(pkt[key_ee]["pos_mm"], dtype=float).ravel()[:3]
        return None

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


# ── stage: detect ─────────────────────────────────────────────────────────────


def stage_detect(args, consumer: _RelayConsumer) -> int:
    import cv2  # for Rodrigues
    detector = _load_detector(args.families)
    K = consumer.camera_matrix
    tag_id = args.world_tag_id if args.world_tag_id is not None else args.ee_tag_id
    if tag_id is None:
        _log("detect needs --world-tag-id (or --ee-tag-id) to track")
        return 2
    _log(f"detect: tracking tag {tag_id} for {args.duration:.0f}s "
         f"(tag-size {args.tag_size} m). Hold the tag static in view.")

    translations: List[np.ndarray] = []
    rvecs: List[np.ndarray] = []
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
        tags = _detect_tags(detector, b.video.bgr, K, args.tag_size)
        if tag_id in tags:
            seen += 1
            T = tags[tag_id]["T"]
            translations.append(T[:3, 3].copy())
            rvec, _ = cv2.Rodrigues(T[:3, :3])
            rvecs.append(rvec.ravel())
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
        rv = np.vstack(rvecs)
        jit_deg = np.degrees(np.std(rv, axis=0))
        _log(f"translation jitter (std) mm: x={jit_mm[0]:.2f} y={jit_mm[1]:.2f} "
             f"z={jit_mm[2]:.2f}  (norm {np.linalg.norm(jit_mm):.2f})")
        _log(f"rotation jitter (std) deg:   {np.max(jit_deg):.2f} (max axis)")
        _log(f"decision margin: median={np.median(margins):.1f} "
             f"min={np.min(margins):.1f}   hamming max={max(hammings)}")
        # Pass/fail vs §7.1 starting thresholds.
        ok = (rate >= 0.90 and np.linalg.norm(jit_mm) < 5.0 and np.max(jit_deg) < 2.0)
        _log(f"VERDICT: {'PASS' if ok else 'REVIEW'} "
             f"(targets: rate≥90%, |jitter|<5mm, rot<2°)")
    return 0


# ── stage: gaze ───────────────────────────────────────────────────────────────


def stage_gaze(args, consumer: _RelayConsumer) -> int:
    detector = _load_detector(args.families)
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
        tags = _detect_tags(detector, b.video.bgr, K, args.tag_size)
        if tag_id not in tags:
            continue
        ray = gaze_ray_cam(b.gaze.x, b.gaze.y, K)
        if ray is None:
            continue
        tag_centre_ray = tags[tag_id]["T"][:3, 3]  # tag origin in cam frame
        errors.append(angle_between_deg(ray, tag_centre_ray))

    errors = [e for e in errors if np.isfinite(e)]
    if not errors:
        _log("no valid (gaze + tag) frames — check worn/fixation/relay")
        return 1
    arr = np.array(errors)
    _log(f"samples={arr.size} gaze-to-tag angular error deg: "
         f"median={np.median(arr):.2f} p90={np.percentile(arr, 90):.2f} "
         f"max={arr.max():.2f}")
    ok = np.median(arr) <= 2.0  # Neon budget ≈1.3–1.8°, allow a little margin
    _log(f"VERDICT: {'PASS' if ok else 'REVIEW'} (target median ≲1.5°)")
    return 0


# ── stage: collect ────────────────────────────────────────────────────────────


def stage_collect(args, consumer: _RelayConsumer) -> int:
    if args.world_tag_id is None or args.ee_tag_id is None:
        _log("collect needs both --world-tag-id and --ee-tag-id")
        return 2
    detector = _load_detector(args.families)
    K = consumer.camera_matrix
    offset_mm = np.asarray(args.t_eetag_ee, dtype=float)
    robot: Optional[RobotTelemetry] = None
    if args.with_robot:
        robot = RobotTelemetry(args.robot_ip, args.robot_port, args.bind_ip,
                               args.bind_port, side=args.side)
        _log(f"robot telemetry: dial {args.robot_ip}:{args.robot_port} "
             f"side={args.side} (READ-ONLY q;seq)")
    else:
        _log("no --with-robot: P_base will be NaN (camera-side dry run)")

    _log("collect: hand-guide the EE so BOTH tags are in view, fixate, then "
         "press Enter to capture. Type 'q' + Enter to finish. NO ROBOT MOTION "
         "is commanded.")
    p_world_rows: List[np.ndarray] = []
    p_base_rows: List[np.ndarray] = []
    tcw_rows: List[np.ndarray] = []
    tce_rows: List[np.ndarray] = []

    while True:
        cmd = input(f"[{len(p_world_rows)} captured] Enter=capture / q=finish > ").strip().lower()
        if cmd == "q":
            break
        b = consumer.latest()
        if b is None or b.video is None or b.video.bgr is None:
            _log("  no frame available; relay up?")
            continue
        tags = _detect_tags(detector, b.video.bgr, K, args.tag_size)
        if args.world_tag_id not in tags or args.ee_tag_id not in tags:
            have = sorted(tags.keys())
            _log(f"  need both tags; saw {have}. Reposition so both are visible.")
            continue
        T_cam_world = tags[args.world_tag_id]["T"]
        T_cam_eetag = tags[args.ee_tag_id]["T"]
        T_world_eetag = invert_transform(T_cam_world) @ T_cam_eetag
        p_world = ee_point_in_world(T_world_eetag, offset_mm)
        if robot is not None:
            p_base = robot.query_ee_mm()
            if p_base is None:
                _log("  robot telemetry timed out; skipping this capture")
                continue
        else:
            p_base = np.full(3, np.nan)
        p_world_rows.append(p_world)
        p_base_rows.append(p_base)
        tcw_rows.append(T_cam_world)
        tce_rows.append(T_cam_eetag)
        _log(f"  captured #{len(p_world_rows)}: "
             f"P_world={np.round(p_world, 1)} mm  P_base={np.round(p_base, 1)} mm")

    if robot is not None:
        robot.close()
    if len(p_world_rows) < 3:
        _log(f"only {len(p_world_rows)} captures (<3) — not saving a solvable npz")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = args.utc_stamp or "unstamped"
    out_path = out_dir / f"apriltag_spike_{stamp}.npz"
    meta = {
        "version": 3,
        "scheme": "ee_mounted_tag_free_roam",
        "side": args.side,
        "world_tag_id": args.world_tag_id,
        "ee_tag_id": args.ee_tag_id,
        "tag_size_m": args.tag_size,
        "t_eetag_ee_mm": offset_mm.tolist(),
        "with_robot": bool(args.with_robot),
        "units": {"P_world": "mm", "P_base": "mm", "t_eetag_ee": "mm"},
    }
    np.savez_compressed(
        out_path,
        P_world=np.vstack(p_world_rows),
        P_base=np.vstack(p_base_rows),
        T_cam_world=np.stack(tcw_rows),
        T_cam_eetag=np.stack(tce_rows),
        meta=np.array(meta, dtype=object),
    )
    _log(f"saved {len(p_world_rows)} captures → {out_path}")
    if args.with_robot:
        _log("run the solve: "
             f"python tools/apriltag_calib_spike.py --stage solve {out_path}")
    return 0


# ── stage: solve (offline, no hardware) ──────────────────────────────────────


def stage_solve(args) -> int:
    npz_path = Path(args.npz)
    if not npz_path.is_file():
        _log(f"npz not found: {npz_path}")
        return 2
    z = np.load(npz_path, allow_pickle=True)
    P_world = np.asarray(z["P_world"], dtype=float)
    P_base = np.asarray(z["P_base"], dtype=float)

    finite = np.all(np.isfinite(P_world), axis=1) & np.all(np.isfinite(P_base), axis=1)
    P_world, P_base = P_world[finite], P_base[finite]
    n = P_world.shape[0]
    if n < 3:
        _log(f"only {n} finite paired points (<3) — cannot solve. "
             "Was --with-robot used during collect?")
        return 1

    T_base_world, rms = umeyama_rigid(P_world, P_base)
    errs = per_point_errors(T_base_world, P_world, P_base)
    _log(f"Umeyama T_base_world from {n} points:")
    for row in T_base_world:
        _log("  [" + "  ".join(f"{v:9.3f}" for v in row) + "]")
    _log(f"RMS residual = {rms:.2f} mm   per-point: "
         f"median={np.median(errs):.2f} max={errs.max():.2f} mm")

    # Leave-one-out cross-validation: fit on N-1, predict the held-out point.
    if n >= 4:
        loo = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            T_i, _ = umeyama_rigid(P_world[mask], P_base[mask])
            pred = T_i[:3, :3] @ P_world[i] + T_i[:3, 3]
            loo.append(np.linalg.norm(pred - P_base[i]))
        loo = np.array(loo)
        _log(f"leave-one-out error: median={np.median(loo):.2f} "
             f"max={loo.max():.2f} mm (generalisation estimate)")

    ok = rms < 20.0  # §7.1 starting bar ~1–2 cm; refine vs REV01 accuracy
    _log(f"VERDICT: {'PASS' if ok else 'REVIEW'} (target RMS ≲10–20 mm)")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    import config as cfg
    p = argparse.ArgumentParser(description="WS5 REV03 AprilTag calibration spike")
    p.add_argument("--stage", required=True,
                   choices=["detect", "gaze", "collect", "solve"])
    p.add_argument("npz", nargs="?", default=None,
                   help="solve stage: path to an apriltag_spike_*.npz")
    p.add_argument("--families", default="tag36h11")
    p.add_argument("--tag-size", type=float, default=0.06,
                   help="tag edge length in METRES (default 0.06)")
    p.add_argument("--world-tag-id", type=int, default=None)
    p.add_argument("--ee-tag-id", type=int, default=None)
    p.add_argument("--t-eetag-ee", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   metavar=("X", "Y", "Z"),
                   help="hand-measured EE-tag→EE offset vector in MM (§4.2)")
    p.add_argument("--duration", type=float, default=10.0,
                   help="detect/gaze sampling window (s)")
    p.add_argument("--with-robot", action="store_true",
                   help="collect: also read q;seq EE telemetry (read-only)")
    p.add_argument("--side", default=None, help="robot active side R/L "
                   "(default: env HARMONY_ACTIVE_SIDE or R)")
    p.add_argument("--relay-host", default=None,
                   help="default: config.FRAME_RELAY_DIAL_HOST")
    p.add_argument("--relay-port", type=int, default=None,
                   help="default: config.FRAME_RELAY_PORT")
    p.add_argument("--robot-ip", default=None, help="default: config.UDP_ROBOT[IP]")
    p.add_argument("--robot-port", type=int, default=None,
                   help="default: config.UDP_ROBOT[PORT]")
    p.add_argument("--bind-ip", default="0.0.0.0")
    p.add_argument("--bind-port", type=int, default=None,
                   help="default: config.UDP_ROBOT[PORT]")
    p.add_argument("--out-dir", default="runs",
                   help="collect: directory for the saved npz")
    p.add_argument("--utc-stamp", default=None,
                   help="collect: UTC stamp for the filename (Date.now is "
                        "unavailable here; pass one for a stable name)")
    args = p.parse_args(argv)

    import os
    args.relay_host = args.relay_host or getattr(cfg, "FRAME_RELAY_DIAL_HOST", "127.0.0.1")
    args.relay_port = args.relay_port or int(getattr(cfg, "FRAME_RELAY_PORT", 5591))
    robot = getattr(cfg, "UDP_ROBOT", {"IP": "192.168.2.1", "PORT": 8080})
    args.robot_ip = args.robot_ip or robot["IP"]
    args.robot_port = args.robot_port or int(robot["PORT"])
    args.bind_port = args.bind_port or int(robot["PORT"])
    args.side = (args.side or os.environ.get("HARMONY_ACTIVE_SIDE", "R")).upper()
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.stage == "solve":
        if not args.npz:
            _log("solve needs an npz path argument")
            return 2
        return stage_solve(args)

    # Camera stages need the relay.
    _log(f"connecting to relay {args.relay_host}:{args.relay_port} …")
    consumer = _RelayConsumer(args.relay_host, args.relay_port)
    try:
        if consumer.latest() is None:
            time.sleep(0.5)  # brief grace for the first bundle
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
