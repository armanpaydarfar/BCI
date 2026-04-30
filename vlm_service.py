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

import struct

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
    # `--host` is the bind address for both the UDP request socket and the
    # TCP overlay socket. Production deployments set this to 0.0.0.0 so the
    # Linux panel/driver can dial in across the LAN; single-machine dev
    # keeps it on 127.0.0.1.
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5589)
    p.add_argument("--neon-host", default="", help="Empty string triggers LAN discovery")
    p.add_argument("--model", default="gemini-2.5-flash", help="VLM model name")
    p.add_argument("--seg-model", default="models/FastSAM-s.pt", help="Relative to repo-dir")
    p.add_argument("--device", default="cpu")
    p.add_argument("--enable-depth", action="store_true")
    p.add_argument("--depth-checkpoint", default="models/depth_pro.pt", help="Relative to repo-dir")
    p.add_argument("--session-dir", default=None, help="Where to save depth PNGs etc.")
    p.add_argument("--enable-overlay", action="store_true",
                   help="Start the annotated JPEG TCP push server for control-panel display")
    p.add_argument("--overlay-port", type=int, default=5590,
                   help="TCP port for the overlay frame push server")
    p.add_argument("--verbose", action="store_true")
    # Frame source toggle for the GPU-host migration plan (see SoftwareDocs/
    # GPU_Service_Host_Architecture_Plan.md §3.4). Default `local` preserves
    # today's behaviour (open Neon directly via NeonLiveReader). `remote`
    # consumes envelopes from a Utils/frame_relay.py TCP server instead.
    p.add_argument("--frame-source", choices=["local", "remote"], default="local",
                   help="local=open Neon directly; remote=consume Utils/frame_relay envelopes")
    p.add_argument("--remote-frame-host", default=None,
                   help="Host of the frame_relay server (required when --frame-source=remote)")
    p.add_argument("--remote-frame-port", type=int, default=5591,
                   help="Port of the frame_relay server (default 5591)")
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
        # Telemetry for the status payload — surfaced to the Windows panel's
        # "Frame intake" badge and to the Linux panel's remote-status query.
        self._frames_received: int = 0

        self._stop_event = threading.Event()
        self._frame_thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None

        # Two-object (sequential-decide) snapshot cache. TTL long enough for a
        # user to look at A, think, then look at B; short enough that stale
        # frames can't linger indefinitely.
        self._snapshots = SnapshotCache(ttl_s=60.0, max_size=4)

        # Render-state cache: shared between the serving thread (writers) and
        # the overlay render thread (reader). Protected by _render_lock.
        self._render_lock = threading.Lock()
        self._cached_dets: list = []
        self._cached_hit_det = None          # raw Detection object under gaze
        self._cached_hit_wp: Optional[dict] = None   # JSON-safe hit waypoint dict
        self._vlm_state: str = "IDLE"        # IDLE | THINKING | AWAITING_SECOND | DECIDED
        self._last_decision: Optional[dict] = None
        self._first_snap_det = None          # first-fixation Detection for AWAITING_SECOND badge

        # Overlay frame store: render thread writes, TCP server thread reads.
        self._overlay_lock = threading.Lock()
        self._latest_overlay_jpg: Optional[bytes] = None
        self._frame_count: int = 0
        self._start_time: float = time.time()

        # Continuous segmentation stream (toggle from the panel). When on,
        # _segment_stream_loop calls detector.detect at seg_stream_hz and
        # writes into _cached_dets so the overlay stays fresh without manual
        # "Segment Now" clicks.
        self._seg_stream_thread: Optional[threading.Thread] = None
        self._seg_stream_stop = threading.Event()
        self._seg_stream_hz: float = 10.0

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
                    self._frames_received += 1
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

    def _cache_dets(self, dets: list, hit_det=None, hit_wp: Optional[dict] = None) -> None:
        with self._render_lock:
            self._cached_dets = list(dets)
            self._cached_hit_det = hit_det
            self._cached_hit_wp = hit_wp

    def _set_vlm_state(self, state: str, decision: Optional[dict] = None, first_det=None) -> None:
        with self._render_lock:
            self._vlm_state = state
            if decision is not None:
                self._last_decision = decision
            if first_det is not None:
                self._first_snap_det = first_det
            if state == "IDLE":
                self._first_snap_det = None

    # ── dispatch ──────────────────────────────────────────────────────────

    def _dispatch(self, cmd: str, req: dict) -> dict:
        handlers = {
            "status": lambda: self._cmd_status(),
            "snapshot": lambda: self._cmd_snapshot(),
            "segment": lambda: self._cmd_segment(req),
            "segment_stream": lambda: self._cmd_segment_stream(req),
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
        # Signal the overlay renderer that a long model call is in progress.
        if cmd in ("decide", "reason", "capture_first", "decide_pair"):
            self._set_vlm_state("THINKING")
        return handler()

    # ── commands ──────────────────────────────────────────────────────────

    def _cmd_status(self) -> dict:
        bundle, fix, t = self._latest()
        with self._frame_lock:
            frames_received = int(self._frames_received)
        # `frame_source_connected` reports whether the underlying reader is
        # currently producing frames — for `local` mode this means the Neon
        # device is talking to the SDK; for `remote` mode it means the
        # frame_relay TCP connection is alive and shipping envelopes.
        # 2 s heuristic is generous compared to relay default of 10 Hz.
        frame_age = (time.time() - t) if t else None
        connected = bool(bundle is not None and frame_age is not None and frame_age < 2.0)
        return {
            "ok": True,
            "neon_connected": bundle is not None,
            "frame_age_s": frame_age,
            "depth_enabled": self.depth_estimator is not None,
            "model": self.args.model,
            "fixation_active": bool(fix.active) if fix is not None else False,
            "frame_source": getattr(self.args, "frame_source", "local"),
            "frame_source_connected": connected,
            "frames_received": frames_received,
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

        self._cache_dets(dets)
        return {"ok": True, "detections": out, "elapsed_s": elapsed, "n": len(out)}

    def _cmd_segment_stream(self, req: dict) -> dict:
        enabled = bool(req.get("enabled", False))
        hz = float(req.get("hz", self._seg_stream_hz))
        if hz <= 0.0:
            return {"ok": False, "error": "hz must be > 0"}
        if enabled:
            self._start_segment_stream(hz)
        else:
            self._stop_segment_stream()
        return {"ok": True, "enabled": enabled, "hz": hz}

    def _start_segment_stream(self, hz: float) -> None:
        # Idempotent: if already running at a different rate, swap the rate
        # without restarting the thread.
        self._seg_stream_hz = hz
        if self._seg_stream_thread is not None and self._seg_stream_thread.is_alive():
            return
        self._seg_stream_stop.clear()
        self._seg_stream_thread = threading.Thread(
            target=self._segment_stream_loop, daemon=True, name="vlm-seg-stream",
        )
        self._seg_stream_thread.start()
        _log(f"segment stream started at {hz:.1f} Hz")

    def _stop_segment_stream(self) -> None:
        self._seg_stream_stop.set()
        t = self._seg_stream_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._seg_stream_thread = None
        # Clear cached detections so the overlay doesn't keep drawing the last
        # mask set after the stream is turned off.
        self._cache_dets([])
        _log("segment stream stopped")

    def _segment_stream_loop(self) -> None:
        next_run = time.perf_counter()
        last_hz = self._seg_stream_hz
        period = 1.0 / max(last_hz, 1e-6)
        while not self._seg_stream_stop.is_set() and not self._stop_event.is_set():
            # Re-read the rate each iteration so live changes take effect.
            if self._seg_stream_hz != last_hz:
                last_hz = self._seg_stream_hz
                period = 1.0 / max(last_hz, 1e-6)

            now_pc = time.perf_counter()
            if now_pc < next_run:
                time.sleep(min(0.005, next_run - now_pc))
                continue

            bundle, _, _ = self._latest()
            if bundle is None:
                next_run = now_pc + period
                time.sleep(0.020)
                continue

            try:
                dets = self.detector.detect(bundle.video.bgr)
                self._cache_dets(dets)
            except Exception as e:
                if self.args.verbose:
                    _log(f"segment stream error: {e}")

            # Schedule next tick relative to the original cadence, but if we're
            # falling behind by more than 2 periods just resync — avoids a
            # runaway catch-up burst after a slow inference.
            next_run += period
            now_pc2 = time.perf_counter()
            if next_run < now_pc2 - 2.0 * period:
                next_run = now_pc2

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
        hit_det = None
        hit_waypoint = None
        for d, wp in zip(dets, waypoints_out):
            x1, y1, x2, y2 = d.box_xyxy
            if x1 <= gaze_xy[0] <= x2 and y1 <= gaze_xy[1] <= y2:
                hit_det = d
                hit_waypoint = wp
                break

        self._cache_dets(dets, hit_det, hit_waypoint)

        fix_state = fix if fix is not None else self._FixationState(active=True)
        timeout_s = float(req.get("timeout", 30.0))
        future = self.reasoner.reason_async(bundle.gaze, fix_state, bundle.video.bgr, dets, "")
        try:
            result = future.result(timeout=timeout_s)
        except FutureTimeoutError:
            self._set_vlm_state("IDLE")
            return {
                "ok": False,
                "error": "vlm_timeout",
                "waypoints": waypoints_out,
                "hit_waypoint": hit_waypoint,
                "depth_at_gaze_m": depth_at_gaze_m,
            }

        if not isinstance(result, dict):
            result = {}

        decision = {
            "waypoints": waypoints_out,
            "hit_waypoint": hit_waypoint,
            "depth_at_gaze_m": depth_at_gaze_m,
            "gaze_px": list(gaze_xy),
            **result,
        }
        self._set_vlm_state("DECIDED", decision=decision)
        return {"ok": True, **decision}

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

        hit_det = None
        hit_waypoint: Optional[dict] = None
        for d, wp in zip(dets, waypoints_out):
            x1, y1, x2, y2 = d.box_xyxy
            if x1 <= gaze_xy[0] <= x2 and y1 <= gaze_xy[1] <= y2:
                hit_det = d
                hit_waypoint = wp
                break

        return {
            "detections": dets,
            "hit_det": hit_det,
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
            "hit_det": processed["hit_det"],
            "waypoints": processed["waypoints"],
            "hit_waypoint": processed["hit_waypoint"],
            "depth_at_gaze_m": processed["depth_at_gaze_m"],
        })

        self._cache_dets(processed["detections"], processed["hit_det"], processed["hit_waypoint"])
        self._set_vlm_state("AWAITING_SECOND", first_det=processed["hit_det"])

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

        self._cache_dets(second["detections"], second["hit_det"], second["hit_waypoint"])
        decision = {
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
        self._set_vlm_state("DECIDED", decision=decision)
        return {"ok": True, **decision}

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
        self._stop_segment_stream()
        self._stop_event.set()
        return {"ok": True}

    # ── overlay rendering + TCP push server ───────────────────────────────

    def start_overlay_server(self) -> None:
        """Spawn the render thread and TCP push server thread."""
        t_render = threading.Thread(target=self._render_loop, daemon=True, name="vlm-render")
        t_server = threading.Thread(target=self._overlay_server_loop, daemon=True, name="vlm-overlay-tcp")
        t_render.start()
        t_server.start()
        _log(f"overlay server started (tcp port {self.args.overlay_port})")

    def _render_loop(self) -> None:
        """Render annotated frames at ~5 Hz and store the latest JPEG."""
        import cv2 as _cv2
        from utils.overlay_renderer import OverlayRenderer

        renderer = OverlayRenderer()
        while not self._stop_event.is_set():
            try:
                bundle, fix, _ = self._latest()
                if bundle is not None:
                    with self._render_lock:
                        dets = list(self._cached_dets)
                        hit_det = self._cached_hit_det
                        hit_wp = self._cached_hit_wp
                        state = self._vlm_state
                        decision = self._last_decision
                        first_det = self._first_snap_det
                        fc = self._frame_count

                    fix_state = fix if fix is not None else self._FixationState(active=False)
                    canvas = renderer.render(
                        frame_bgr=bundle.video.bgr.copy(),
                        detections=dets,
                        hit=hit_det,
                        fixation=fix_state,
                        gaze=bundle.gaze,
                        vlm_state=state,
                        last_decision=decision,
                        api_mode="VISION",
                        pending_detection=None,
                        frame_idx=fc,
                        total_frames=0,
                        duration_s=time.time() - self._start_time,
                        first_object=first_det,
                        hit_waypoint=hit_wp,
                    )
                    _, jpg_buf = _cv2.imencode(
                        ".jpg", canvas, [_cv2.IMWRITE_JPEG_QUALITY, 75]
                    )
                    with self._overlay_lock:
                        self._latest_overlay_jpg = bytes(jpg_buf)
                        self._frame_count = fc + 1
            except Exception as e:
                if self.args.verbose:
                    _log(f"render loop error: {e}")
            time.sleep(0.2)  # 5 Hz

    def _overlay_server_loop(self) -> None:
        """TCP server: accept one client at a time and push length-prefixed JPEGs."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind((self.args.host, self.args.overlay_port))
        except OSError as e:
            _log(f"overlay server bind failed on port {self.args.overlay_port}: {e}")
            return
        srv.listen(1)
        srv.settimeout(1.0)
        try:
            while not self._stop_event.is_set():
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                _log(f"overlay client connected: {addr}")
                conn.settimeout(2.0)
                try:
                    while not self._stop_event.is_set():
                        with self._overlay_lock:
                            jpg = self._latest_overlay_jpg
                        if jpg:
                            try:
                                conn.sendall(struct.pack(">I", len(jpg)) + jpg)
                            except (BrokenPipeError, ConnectionResetError, OSError):
                                break
                        time.sleep(0.2)
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    _log(f"overlay client disconnected: {addr}")
        finally:
            try:
                srv.close()
            except Exception:
                pass


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
    # Change cwd so relative model paths (FastSAM-s.pt, depth_pro.pt) resolve correctly.
    os.chdir(repo_dir)

    # Read .env values and force-set them in os.environ so that a pre-existing
    # empty-string value inherited from the parent process doesn't shadow the file.
    # Using dotenv_values (returns a plain dict) then updating os.environ directly
    # avoids load_dotenv's override=False default and any version-specific behaviour.
    env_file = os.path.join(repo_dir, ".env")
    if os.path.isfile(env_file):
        try:
            from dotenv import dotenv_values
            for _k, _v in dotenv_values(env_file).items():
                if _v is not None:
                    os.environ[_k] = _v
        except Exception as _e:
            _log(f"warning: could not read .env via dotenv ({_e}); falling back to manual parse")
            with open(env_file) as _fh:
                for _line in _fh:
                    _line = _line.strip()
                    if _line and not _line.startswith("#") and "=" in _line:
                        _ek, _ev = _line.split("=", 1)
                        os.environ[_ek.strip()] = _ev.strip().strip('"').strip("'")

    from utils.neon import NeonLiveReader
    from utils.object_detector import ObjectDetector
    from utils.fixation_detector import FixationDetector, FixationState
    from utils.intent_reasoner import IntentReasoner

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        _log(f"FATAL: no GOOGLE_API_KEY / OPENAI_API_KEY in env or .env (looked in {env_file})")
        sys.exit(2)

    if args.frame_source == "remote":
        if not args.remote_frame_host:
            _log("FATAL: --frame-source=remote requires --remote-frame-host")
            sys.exit(2)
        # BCI/Utils/remote_frame_reader.py exposes a NeonLiveReader-shaped
        # iterator over the TCP envelope stream produced by Utils/frame_relay.py.
        # bundle.video.bgr / bundle.gaze / bundle.imu / camera_matrix all
        # match NeonLiveReader, so the rest of this file is unchanged.
        bci_root = os.path.dirname(os.path.abspath(__file__))
        if bci_root not in sys.path:
            sys.path.insert(0, bci_root)
        from Utils.remote_frame_reader import RemoteFrameReader
        _log(f"connecting to frame relay tcp://{args.remote_frame_host}:{args.remote_frame_port}…")
        reader = RemoteFrameReader(args.remote_frame_host, args.remote_frame_port)
    else:
        _log(f"connecting to Neon (host={args.neon_host or 'auto-discover'})…")
        reader = NeonLiveReader(host=args.neon_host or None)

    _log(f"loading segmentation model: {args.seg_model} on {args.device}…")
    detector = ObjectDetector(model_size=args.seg_model, device=args.device)

    depth_estimator = None
    if args.enable_depth:
        from utils.depth_estimator import DepthEstimator
        import torch as _torch
        save_path = None
        if args.session_dir:
            save_path = Path(args.session_dir) / "depth"
        # fp16 on CUDA is ~2× faster per Depth Pro's own guidance and gives
        # virtually identical depth (sub-cm differences for our use). CPU keeps
        # fp32 because half-precision CPU kernels are slower, not faster.
        precision = _torch.float16 if args.device == "cuda" else _torch.float32
        _log(f"loading Depth Pro: {args.depth_checkpoint} on {args.device} ({precision})…")
        depth_estimator = DepthEstimator(
            checkpoint=args.depth_checkpoint,
            device=args.device,
            precision=precision,
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

    if args.enable_overlay:
        service.start_overlay_server()

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
