#!/usr/bin/env python3
"""
vlm_service.py — UDP request-reply service wrapping harmony_vlm's capabilities.

Runs in the `harmony_vlm` conda env and imports harmony_vlm's utils/ via
sys.path. Loads the Neon live reader, FastSAM, Depth Pro (optional), Gemini,
and the I-VT fixation detector once at startup; then exposes each capability
as a distinct UDP command so the control panel and experiment drivers can
consume them independently.

Mirrors the request-reply idiom of gaze_runner.py:GazeUDPServer (L43-204) —
single-threaded dispatch, JSON in / JSON out, single datagram round-trip.

Supported commands (JSON `cmd` field, all lower-case):
    status         — service + Neon + models health (cheap, always ok)
    snapshot       — latest gaze px + frame metadata (no model calls)
    segment        — run FastSAM on latest frame; returns detections
                     (include_masks=True to include mask polygons)
    depth          — run Depth Pro on latest frame; returns depth stats
                     (at_gaze=True includes depth_at_gaze_m)
    reason         — Gemini/OpenAI reasoning on latest frame+gaze
    decide         — full pipeline: segment + depth + reason + waypoints
                     (returns the same dict shape demo.py emits to JSONL)
    capture_first  — snapshot current frame/gaze/detections/waypoints under a
                     server-side token; used as the first fixation in a
                     two-object sequential decide flow
    decide_pair    — combine a previously-captured snapshot with the current
                     frame/gaze and run reason_async_pair; returns object,
                     second_object, paired candidates, and waypoints for both
    camera_matrix  — Neon intrinsics + distortion, for robot-calibration work
    stop           — shutdown the service

Per-command latency budgets (clients should set recv timeout accordingly):
    status/snapshot/camera_matrix : <50 ms
    segment                       : 200-800 ms (CPU FastSAM)
    depth                         : 1-3 s (CPU Depth Pro)
    reason                        : 2-10 s (Gemini round-trip)
    decide                        : sum of the above, typically 3-15 s
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import uuid
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class SnapshotCache:
    """
    Bounded TTL cache of captured frame+gaze+detections+waypoints snapshots.

    Used by the two-object (sequential-decide) flow: capture_first stores a
    snapshot under a random id; decide_pair retrieves it to pair with the
    current frame and run the VLM's reason_async_pair. Frames are large
    (~5 MB each) so we keep the cache small and short-lived.
    """

    def __init__(self, ttl_s: float = 60.0, max_size: int = 4) -> None:
        self._items: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._ttl = float(ttl_s)
        self._max = int(max_size)

    def put(self, data: dict) -> str:
        snap_id = uuid.uuid4().hex[:8]
        now = time.monotonic()
        with self._lock:
            self._prune_locked(now)
            if len(self._items) >= self._max:
                oldest = min(self._items.items(), key=lambda kv: kv[1]["_t"])[0]
                self._items.pop(oldest, None)
            self._items[snap_id] = {**data, "_t": now}
        return snap_id

    def get(self, snap_id: str) -> Optional[dict]:
        with self._lock:
            self._prune_locked(time.monotonic())
            return self._items.get(snap_id)

    def pop(self, snap_id: str) -> Optional[dict]:
        with self._lock:
            return self._items.pop(snap_id, None)

    def _prune_locked(self, now: float) -> None:
        dead = [k for k, v in self._items.items() if now - v["_t"] > self._ttl]
        for k in dead:
            self._items.pop(k, None)


def parse_args():
    p = argparse.ArgumentParser(description="harmony_vlm UDP service")
    p.add_argument("--repo-dir", required=True, help="Path to harmony_vlm clone")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5589)
    p.add_argument("--neon-host", default="", help="Empty string triggers LAN discovery")
    p.add_argument("--model", default="gemini-2.5-flash", help="VLM model name")
    p.add_argument("--seg-model", default="models/FastSAM-s.pt", help="Relative to repo-dir")
    p.add_argument("--device", default="cpu")
    p.add_argument("--enable-depth", action="store_true")
    p.add_argument("--depth-checkpoint", default="models/depth_pro.pt", help="Relative to repo-dir")
    p.add_argument("--session-dir", default=None, help="Where to save depth PNGs etc.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _log(msg: str) -> None:
    print(f"[vlm_service] {msg}", flush=True)


class VLMService:
    def __init__(self, args, *, reader, detector, depth_estimator, reasoner, fix_det, fixation_state_cls):
        self.args = args
        self.reader = reader
        self.detector = detector
        self.depth_estimator = depth_estimator
        self.reasoner = reasoner
        self.fix_det = fix_det
        self._FixationState = fixation_state_cls

        self._frame_lock = threading.Lock()
        self._latest_bundle = None
        self._latest_bundle_t: float = 0.0
        self._latest_fix = None

        self._stop_event = threading.Event()
        self._frame_thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None

        # Two-object (sequential-decide) snapshot cache. TTL long enough for a
        # user to look at A, think, then look at B; short enough that stale
        # frames can't linger indefinitely.
        self._snapshots = SnapshotCache(ttl_s=60.0, max_size=4)

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start_frame_thread(self) -> None:
        self._frame_thread = threading.Thread(target=self._frame_loop, daemon=True)
        self._frame_thread.start()

    def _frame_loop(self) -> None:
        try:
            for bundle in self.reader:
                if self._stop_event.is_set():
                    break
                fix_state = self.fix_det.update(bundle.gaze)
                with self._frame_lock:
                    self._latest_bundle = bundle
                    self._latest_bundle_t = time.time()
                    self._latest_fix = fix_state
        except Exception as e:
            _log(f"frame loop exited: {e}")
            self._stop_event.set()

    def serve_forever(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.args.host, self.args.port))
        self._sock.settimeout(0.5)
        _log(f"listening on udp://{self.args.host}:{self.args.port}")

        while not self._stop_event.is_set():
            try:
                data, addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                req = json.loads(data.decode("utf-8", errors="replace"))
            except Exception as e:
                resp = {"ok": False, "error": f"bad json: {e}"}
                self._send(resp, addr)
                continue

            cmd = str(req.get("cmd", "")).lower()
            try:
                resp = self._dispatch(cmd, req)
            except Exception as e:
                resp = {"ok": False, "error": f"handler exception: {e}", "cmd": cmd}

            resp.setdefault("cmd", cmd)
            self._send(resp, addr)

    def _send(self, resp: dict, addr) -> None:
        try:
            out = json.dumps(resp, default=_json_default).encode("utf-8")
            if len(out) > 60 * 1024:
                out = json.dumps({"ok": False, "error": "response_too_large", "cmd": resp.get("cmd")}).encode("utf-8")
            self._sock.sendto(out, addr)
        except Exception as e:
            if self.args.verbose:
                _log(f"send failed to {addr}: {e}")

    def stop(self) -> None:
        self._stop_event.set()
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass
        try:
            self.reader.close()
        except Exception:
            pass
        try:
            self.reasoner.shutdown()
        except Exception:
            pass

    # ── helpers ───────────────────────────────────────────────────────────

    def _latest(self):
        with self._frame_lock:
            return self._latest_bundle, self._latest_fix, self._latest_bundle_t

    def _focal_px(self) -> Optional[float]:
        K = getattr(self.reader, "camera_matrix", None)
        if K is None:
            return None
        return float((K[0, 0] + K[1, 1]) / 2.0)

    # ── dispatch ──────────────────────────────────────────────────────────

    def _dispatch(self, cmd: str, req: dict) -> dict:
        handlers = {
            "status": lambda: self._cmd_status(),
            "snapshot": lambda: self._cmd_snapshot(),
            "segment": lambda: self._cmd_segment(req),
            "depth": lambda: self._cmd_depth(req),
            "reason": lambda: self._cmd_reason(req),
            "decide": lambda: self._cmd_decide(req),
            "capture_first": lambda: self._cmd_capture_first(req),
            "decide_pair": lambda: self._cmd_decide_pair(req),
            "camera_matrix": lambda: self._cmd_camera_matrix(),
            "stop": lambda: self._cmd_stop(),
        }
        handler = handlers.get(cmd)
        if handler is None:
            return {"ok": False, "error": f"unknown cmd '{cmd}'"}
        return handler()

    # ── commands ──────────────────────────────────────────────────────────

    def _cmd_status(self) -> dict:
        bundle, fix, t = self._latest()
        return {
            "ok": True,
            "neon_connected": bundle is not None,
            "frame_age_s": (time.time() - t) if t else None,
            "depth_enabled": self.depth_estimator is not None,
            "model": self.args.model,
            "fixation_active": bool(fix.active) if fix is not None else False,
        }

    def _cmd_snapshot(self) -> dict:
        bundle, fix, t = self._latest()
        if bundle is None:
            return {"ok": False, "error": "no_frame"}
        return {
            "ok": True,
            "gaze_px": [float(bundle.gaze.x), float(bundle.gaze.y)],
            "worn": bool(bundle.worn),
            "frame_age_s": time.time() - t,
            "frame_shape": list(bundle.video.bgr.shape),
            "fixation_active": bool(fix.active) if fix is not None else False,
            "fixation_stable": bool(fix.is_stable) if fix is not None else False,
            "fixation_duration_ms": (fix.duration_ns / 1_000_000) if fix is not None else 0.0,
        }

    def _cmd_segment(self, req: dict) -> dict:
        bundle, _, _ = self._latest()
        if bundle is None:
            return {"ok": False, "error": "no_frame"}

        include_masks = bool(req.get("include_masks", False))
        t0 = time.time()
        dets = self.detector.detect(bundle.video.bgr)
        elapsed = time.time() - t0

        out = []
        for d in dets:
            obj = {
                "label": d.label,
                "confidence": float(d.confidence),
                "box_xyxy": [float(v) for v in d.box_xyxy],
                "box_center": [float(v) for v in d.box_center],
            }
            if include_masks and d.mask_polygon is not None:
                obj["mask_polygon"] = d.mask_polygon.reshape(-1, 2).astype(int).tolist()
            out.append(obj)

        return {"ok": True, "detections": out, "elapsed_s": elapsed, "n": len(out)}

    def _cmd_depth(self, req: dict) -> dict:
        if self.depth_estimator is None:
            return {"ok": False, "error": "depth_disabled"}
        bundle, _, _ = self._latest()
        if bundle is None:
            return {"ok": False, "error": "no_frame"}

        at_gaze = bool(req.get("at_gaze", True))
        save = bool(req.get("save", False))
        gaze_xy = (float(bundle.gaze.x), float(bundle.gaze.y))

        t0 = time.time()
        prev_save = self.depth_estimator.save_path
        self.depth_estimator.save_path = prev_save if save else None
        try:
            depth_map, saved_path = self.depth_estimator.estimate(
                bundle.video.bgr, f_px=self._focal_px(), gaze_xy=gaze_xy,
            )
        finally:
            self.depth_estimator.save_path = prev_save
        elapsed = time.time() - t0

        resp: Dict[str, Any] = {
            "ok": True,
            "elapsed_s": elapsed,
            "saved_path": str(saved_path) if saved_path else None,
            "depth_shape": list(depth_map.shape),
            "depth_min_m": float(depth_map.min()),
            "depth_max_m": float(depth_map.max()),
            "depth_median_m": float(np.median(depth_map)),
        }
        if at_gaze:
            h, w = depth_map.shape[:2]
            gx = int(np.clip(round(gaze_xy[0]), 0, w - 1))
            gy = int(np.clip(round(gaze_xy[1]), 0, h - 1))
            resp["depth_at_gaze_m"] = float(depth_map[gy, gx])
        return resp

    def _cmd_reason(self, req: dict) -> dict:
        bundle, fix, _ = self._latest()
        if bundle is None:
            return {"ok": False, "error": "no_frame"}

        include_segments = bool(req.get("include_segments", True))
        timeout_s = float(req.get("timeout", 30.0))
        dets = self.detector.detect(bundle.video.bgr) if include_segments else []

        fix_state = fix if fix is not None else self._FixationState(active=True)
        t0 = time.time()
        future = self.reasoner.reason_async(bundle.gaze, fix_state, bundle.video.bgr, dets, "")
        try:
            result = future.result(timeout=timeout_s)
        except FutureTimeoutError:
            return {"ok": False, "error": "vlm_timeout"}
        elapsed = time.time() - t0

        if not isinstance(result, dict):
            return {"ok": False, "error": "vlm_non_dict_response"}
        return {"ok": True, "elapsed_s": elapsed, **result}

    def _cmd_decide(self, req: dict) -> dict:
        # Import here so this file parses even if harmony_vlm utils aren't loaded
        from utils.object_detector import compute_3d_waypoints

        bundle, fix, _ = self._latest()
        if bundle is None:
            return {"ok": False, "error": "no_frame"}

        gaze_xy = (float(bundle.gaze.x), float(bundle.gaze.y))
        dets = self.detector.detect(bundle.video.bgr)

        waypoints_out: list[dict] = []
        depth_at_gaze_m: Optional[float] = None
        if self.depth_estimator is not None:
            depth_map, _ = self.depth_estimator.estimate(
                bundle.video.bgr, f_px=self._focal_px(), gaze_xy=gaze_xy,
            )
            K = self.reader.camera_matrix
            wps = compute_3d_waypoints(dets, depth_map, K)
            waypoints_out = [
                {
                    "label": wp.label,
                    "position_cam": list(wp.position_cam),
                    "pixel_center": list(wp.pixel_center),
                    "depth_median_m": wp.depth_median_m,
                }
                for wp in wps
            ]
            h, w = depth_map.shape[:2]
            gx = int(np.clip(round(gaze_xy[0]), 0, w - 1))
            gy = int(np.clip(round(gaze_xy[1]), 0, h - 1))
            depth_at_gaze_m = float(depth_map[gy, gx])

        # Pick the waypoint whose detection bounding box contains the gaze px
        hit_waypoint = None
        for d, wp in zip(dets, waypoints_out):
            x1, y1, x2, y2 = d.box_xyxy
            if x1 <= gaze_xy[0] <= x2 and y1 <= gaze_xy[1] <= y2:
                hit_waypoint = wp
                break

        fix_state = fix if fix is not None else self._FixationState(active=True)
        timeout_s = float(req.get("timeout", 30.0))
        future = self.reasoner.reason_async(bundle.gaze, fix_state, bundle.video.bgr, dets, "")
        try:
            result = future.result(timeout=timeout_s)
        except FutureTimeoutError:
            return {
                "ok": False,
                "error": "vlm_timeout",
                "waypoints": waypoints_out,
                "hit_waypoint": hit_waypoint,
                "depth_at_gaze_m": depth_at_gaze_m,
            }

        if not isinstance(result, dict):
            result = {}

        return {
            "ok": True,
            "waypoints": waypoints_out,
            "hit_waypoint": hit_waypoint,
            "depth_at_gaze_m": depth_at_gaze_m,
            "gaze_px": list(gaze_xy),
            **result,
        }

    def _process_frame_and_gaze(self, frame_bgr, gaze_xy: tuple[float, float]) -> dict:
        """Segment + depth + waypoints + hit-test for an arbitrary frame.

        Returns native Detection objects (for passing into reason_async_pair)
        alongside JSON-safe waypoint dicts, so one call covers both the cache
        payload and the UDP response payload.
        """
        from utils.object_detector import compute_3d_waypoints

        dets = self.detector.detect(frame_bgr)

        waypoints_out: list[dict] = []
        depth_at_gaze: Optional[float] = None
        if self.depth_estimator is not None:
            depth_map, _ = self.depth_estimator.estimate(
                frame_bgr, f_px=self._focal_px(), gaze_xy=gaze_xy,
            )
            wps = compute_3d_waypoints(dets, depth_map, self.reader.camera_matrix)
            waypoints_out = [
                {
                    "label": wp.label,
                    "position_cam": list(wp.position_cam),
                    "pixel_center": list(wp.pixel_center),
                    "depth_median_m": wp.depth_median_m,
                }
                for wp in wps
            ]
            h, w = depth_map.shape[:2]
            gx = int(np.clip(round(gaze_xy[0]), 0, w - 1))
            gy = int(np.clip(round(gaze_xy[1]), 0, h - 1))
            depth_at_gaze = float(depth_map[gy, gx])

        hit_waypoint: Optional[dict] = None
        for d, wp in zip(dets, waypoints_out):
            x1, y1, x2, y2 = d.box_xyxy
            if x1 <= gaze_xy[0] <= x2 and y1 <= gaze_xy[1] <= y2:
                hit_waypoint = wp
                break

        return {
            "detections": dets,
            "waypoints": waypoints_out,
            "hit_waypoint": hit_waypoint,
            "depth_at_gaze_m": depth_at_gaze,
        }

    def _cmd_capture_first(self, req: dict) -> dict:
        bundle, fix, _ = self._latest()
        if bundle is None:
            return {"ok": False, "error": "no_frame"}

        gaze_xy = (float(bundle.gaze.x), float(bundle.gaze.y))
        t0 = time.time()
        processed = self._process_frame_and_gaze(bundle.video.bgr, gaze_xy)
        elapsed = time.time() - t0

        # Copy the frame because NeonLiveReader.__iter__ reuses its buffer.
        snap_id = self._snapshots.put({
            "frame_bgr": bundle.video.bgr.copy(),
            "gaze_xy": gaze_xy,
            "fix_state": fix if fix is not None else self._FixationState(active=True),
            "detections": processed["detections"],
            "waypoints": processed["waypoints"],
            "hit_waypoint": processed["hit_waypoint"],
            "depth_at_gaze_m": processed["depth_at_gaze_m"],
        })

        return {
            "ok": True,
            "snapshot_id": snap_id,
            "n_detections": len(processed["detections"]),
            "waypoints": processed["waypoints"],
            "hit_waypoint": processed["hit_waypoint"],
            "depth_at_gaze_m": processed["depth_at_gaze_m"],
            "gaze_px": list(gaze_xy),
            "elapsed_s": elapsed,
        }

    def _cmd_decide_pair(self, req: dict) -> dict:
        snap_id = req.get("snapshot_id")
        if not snap_id:
            return {"ok": False, "error": "missing_snapshot_id"}
        first = self._snapshots.get(str(snap_id))
        if first is None:
            return {"ok": False, "error": "snapshot_not_found_or_expired"}

        bundle, fix, _ = self._latest()
        if bundle is None:
            return {"ok": False, "error": "no_frame"}

        second_gaze_xy = (float(bundle.gaze.x), float(bundle.gaze.y))
        second_frame = bundle.video.bgr

        t0 = time.time()
        second = self._process_frame_and_gaze(second_frame, second_gaze_xy)

        first_fix_state = first["fix_state"]
        second_fix_state = fix if fix is not None else self._FixationState(active=True)
        timeout_s = float(req.get("timeout", 45.0))

        future = self.reasoner.reason_async_pair(
            first["gaze_xy"],
            second_gaze_xy,
            first_fix_state,
            second_fix_state,
            first["frame_bgr"],
            second_frame,
            first["detections"],
            second["detections"],
            "",
        )
        try:
            result = future.result(timeout=timeout_s)
        except FutureTimeoutError:
            return {
                "ok": False,
                "error": "vlm_timeout",
                "snapshot_id": snap_id,
                "first_waypoint": first["hit_waypoint"],
                "second_waypoint": second["hit_waypoint"],
            }
        elapsed = time.time() - t0

        if not isinstance(result, dict):
            result = {}

        return {
            "ok": True,
            "snapshot_id": snap_id,
            "elapsed_s": elapsed,
            "first_gaze_px": list(first["gaze_xy"]),
            "second_gaze_px": list(second_gaze_xy),
            "first_waypoints": first["waypoints"],
            "second_waypoints": second["waypoints"],
            "first_waypoint": first["hit_waypoint"],
            "second_waypoint": second["hit_waypoint"],
            "first_depth_at_gaze_m": first["depth_at_gaze_m"],
            "second_depth_at_gaze_m": second["depth_at_gaze_m"],
            **result,
        }

    def _cmd_camera_matrix(self) -> dict:
        K = getattr(self.reader, "camera_matrix", None)
        if K is None:
            return {"ok": False, "error": "no_camera_matrix"}
        dist = getattr(self.reader, "distortion_coeffs", None)
        return {
            "ok": True,
            "camera_matrix": K.tolist(),
            "distortion_coeffs": dist.tolist() if dist is not None else None,
        }

    def _cmd_stop(self) -> dict:
        self._stop_event.set()
        return {"ok": True}


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"unsupported type for JSON: {type(o)}")


def main() -> None:
    args = parse_args()

    repo_dir = os.path.abspath(args.repo_dir)
    if not os.path.isdir(repo_dir):
        _log(f"FATAL: repo-dir not a directory: {repo_dir}")
        sys.exit(2)
    sys.path.insert(0, repo_dir)
    # .env at repo root holds GOOGLE_API_KEY / OPENAI_API_KEY for IntentReasoner.
    # Change cwd so load_dotenv() and relative model paths resolve correctly.
    os.chdir(repo_dir)

    from dotenv import load_dotenv
    load_dotenv()

    from utils.neon import NeonLiveReader
    from utils.object_detector import ObjectDetector
    from utils.fixation_detector import FixationDetector, FixationState
    from utils.intent_reasoner import IntentReasoner

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        _log("FATAL: no GOOGLE_API_KEY / OPENAI_API_KEY in env or .env")
        sys.exit(2)

    _log(f"connecting to Neon (host={args.neon_host or 'auto-discover'})…")
    reader = NeonLiveReader(host=args.neon_host or None)

    _log(f"loading segmentation model: {args.seg_model} on {args.device}…")
    detector = ObjectDetector(model_size=args.seg_model, device=args.device)

    depth_estimator = None
    if args.enable_depth:
        from utils.depth_estimator import DepthEstimator
        save_path = None
        if args.session_dir:
            save_path = Path(args.session_dir) / "depth"
        _log(f"loading Depth Pro: {args.depth_checkpoint} on {args.device}…")
        depth_estimator = DepthEstimator(
            checkpoint=args.depth_checkpoint,
            device=args.device,
            save_path=save_path,
        )

    _log(f"loading reasoner: {args.model}…")
    reasoner = IntentReasoner(api_key=api_key, model=args.model)
    fix_det = FixationDetector()

    service = VLMService(
        args,
        reader=reader,
        detector=detector,
        depth_estimator=depth_estimator,
        reasoner=reasoner,
        fix_det=fix_det,
        fixation_state_cls=FixationState,
    )
    service.start_frame_thread()

    _log("ready")
    try:
        service.serve_forever()
    except KeyboardInterrupt:
        _log("KeyboardInterrupt — stopping")
    finally:
        service.stop()
        _log("stopped")


if __name__ == "__main__":
    main()
