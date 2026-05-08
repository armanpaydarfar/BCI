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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


# Live-overlay detection cap. Paint cost on the Linux side is O(N) full-canvas
# alpha blends (Utils/scene_overlay_renderer.py). Capping N here keeps the
# overlay budget bounded; one-shot segment/decide command paths intentionally
# bypass this filter so the reasoner sees the full mask set.
_OVERLAY_TOP_K = 20
_OVERLAY_CONTAIN_RATIO = 0.85
_OVERLAY_AREA_RATIO = 0.5


def _filter_overlay_dets(dets):
    """Reduce detection count for the live segment-stream cache.

    Two passes:
      1. Top-K by confidence — caps live-overlay paint cost.
      2. Containment drop — if det B is mostly inside det A
         (intersection / area(B) > _OVERLAY_CONTAIN_RATIO) AND
         area(B) < _OVERLAY_AREA_RATIO × area(A), drop B. Targets
         FastSAM-everything's parent+children pattern (e.g. monitor
         co-segmented with icons on the monitor); the gaze-pointing
         workflow prefers the parent.

    Applied only to the seg-stream cache; segment/decide one-shots are
    unfiltered.
    """
    if not dets:
        return dets

    if len(dets) > _OVERLAY_TOP_K:
        dets = sorted(dets, key=lambda d: float(d.confidence), reverse=True)[:_OVERLAY_TOP_K]

    def _bbox_area(d):
        x1, y1, x2, y2 = d.box_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _intersection(a, b):
        ax1, ay1, ax2, ay2 = a.box_xyxy
        bx1, by1, bx2, by2 = b.box_xyxy
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return (ix2 - ix1) * (iy2 - iy1)

    areas = [_bbox_area(d) for d in dets]
    drop = [False] * len(dets)
    for i, di in enumerate(dets):
        if drop[i] or areas[i] <= 0:
            continue
        for j, dj in enumerate(dets):
            if i == j or drop[j] or areas[j] <= 0:
                continue
            if areas[j] >= _OVERLAY_AREA_RATIO * areas[i]:
                continue
            if _intersection(di, dj) / areas[j] > _OVERLAY_CONTAIN_RATIO:
                drop[j] = True

    return [d for d, dropped in zip(dets, drop) if not dropped]


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
    # Remote-stop guard: by default, cmd=stop is honoured only when it
    # arrives from 127.0.0.1. This protects the unattended GPU host from
    # an accidental click on a remote panel taking it down — operator
    # would then have to physically reach the box to restart. Set this
    # flag if you intentionally want the panel to be able to stop the
    # service from across the network (e.g. for scripted teardowns).
    p.add_argument("--allow-remote-stop", action="store_true",
                   help="Honour cmd=stop from non-loopback addresses (off by default)")
    # Anonymous service-side log file. Defaults to ~/.harmony_vlm_logs/
    # (host-local, NOT under any subject directory) — the GPU host runs
    # continuously across multiple subjects and this log is only read
    # for service-level troubleshooting. Set "" / "off" / "none" to
    # disable file logging entirely; stdout is unaffected.
    p.add_argument("--service-log-dir", default=None,
                   help="Directory for the service-side log file. "
                        "Default: ~/.harmony_vlm_logs/. Pass an empty "
                        "string to disable.")
    return p.parse_args()


# Module-level handle for the anonymous service log, opened in main()
# once --service-log-dir is resolved. _log() tees to it when present.
_LOG_FILE = None


def _log(msg: str) -> None:
    line = f"[vlm_service] {msg}"
    print(line, flush=True)
    fh = _LOG_FILE
    if fh is not None:
        try:
            fh.write(f"{datetime.now().isoformat(timespec='milliseconds')} {line}\n")
            fh.flush()
        except OSError:
            # Disk drop-out shouldn't crash the service. Stop tee'ing
            # by clearing the global; stdout still works.
            globals()["_LOG_FILE"] = None


def _disable_udp_connreset(sock) -> None:
    """Call WSAIoctl(SIO_UDP_CONNRESET, FALSE) on a Windows UDP socket so
    ICMP "port unreachable" replies no longer poison recvfrom() with
    WSAECONNRESET (10054). Python's stdlib ``socket.ioctl`` only accepts
    SIO_RCVALL / SIO_KEEPALIVE_VALS / SIO_LOOPBACK_FAST_PATH, so we go
    around it via ctypes. Raises OSError on failure; caller swallows.
    """
    import ctypes
    from ctypes import wintypes

    SIO_UDP_CONNRESET = 0x9800000C  # IOC_IN | IOC_VENDOR | 12
    ws2 = ctypes.WinDLL("Ws2_32", use_last_error=True)
    WSAIoctl = ws2.WSAIoctl
    WSAIoctl.argtypes = [
        wintypes.HANDLE,                  # SOCKET
        wintypes.DWORD,                   # dwIoControlCode
        ctypes.c_void_p,                  # lpvInBuffer
        wintypes.DWORD,                   # cbInBuffer
        ctypes.c_void_p,                  # lpvOutBuffer
        wintypes.DWORD,                   # cbOutBuffer
        ctypes.POINTER(wintypes.DWORD),   # lpcbBytesReturned
        ctypes.c_void_p,                  # lpOverlapped
        ctypes.c_void_p,                  # lpCompletionRoutine
    ]
    WSAIoctl.restype = ctypes.c_int

    inbuf = ctypes.c_uint32(0)  # FALSE
    bytes_returned = wintypes.DWORD(0)
    rc = WSAIoctl(
        sock.fileno(),
        SIO_UDP_CONNRESET,
        ctypes.byref(inbuf), ctypes.sizeof(inbuf),
        None, 0,
        ctypes.byref(bytes_returned),
        None, None,
    )
    if rc != 0:
        err = ctypes.get_last_error()
        raise OSError(f"WSAIoctl(SIO_UDP_CONNRESET) failed (WSAGetLastError={err})")


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

        # Continuous segmentation stream (toggle from the panel). When on,
        # _segment_stream_loop calls detector.detect at seg_stream_hz and
        # writes into _cached_dets so the overlay stays fresh without manual
        # "Segment Now" clicks.
        self._seg_stream_thread: Optional[threading.Thread] = None
        self._seg_stream_stop = threading.Event()
        self._seg_stream_hz: float = 10.0
        # Telemetry for the seg-stream loop. Updated under no lock from the
        # loop thread and read read-only from _cmd_status; consumers tolerate
        # a slightly stale snapshot. Each window covers _SEG_STREAM_STATS_S
        # seconds; periodic stats lines log to stdout, the latest snapshot is
        # included in status replies so the panel can surface it.
        self._seg_stream_stats: Dict[str, Any] = {
            "active": False,
            "hz_target": 0.0,
            "hz_achieved": 0.0,
            "mean_dets": 0.0,
            "mean_infer_ms": 0.0,
            "errors": 0,
            "window_s": 0.0,
            "last_emit_t": 0.0,
        }

        # Subscribe-mode push: panel/clients send `cmd=subscribe` and we
        # broadcast `vlm_results` JSON datagrams from a tick thread. The
        # render-on-Linux refactor (Render_Layer_Refactor.md §3) replaces
        # the JPEG overlay round-trip with this JSON channel.
        self._subscribers_lock = threading.Lock()
        # subscriber_id (uuid hex) → {addr, hz, last_sent_t, expires_at}
        self._subscribers: Dict[str, Dict[str, Any]] = {}
        self._results_tick_thread: Optional[threading.Thread] = None
        self._results_tick_stop = threading.Event()

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start_frame_thread(self) -> None:
        self._frame_thread = threading.Thread(target=self._frame_loop, daemon=True)
        self._frame_thread.start()

    def _frame_loop(self) -> None:
        """Pull bundles from the reader and cache the latest one for UDP
        handlers to read. Resilient against reader exceptions: a transient
        error (relay disconnect, bad bundle, etc.) does NOT kill the
        service. Logs and retries; the UDP service keeps responding to
        status/decide/segment so the operator panel doesn't lose its
        session every time the wire hiccups.

        RemoteFrameReader's internal auto-reconnect handles relay
        disconnects below this layer; this outer retry just covers any
        per-bundle parsing/processing errors that bubble up here.
        """
        retry_delay_s = 1.0
        while not self._stop_event.is_set():
            try:
                for bundle in self.reader:
                    if self._stop_event.is_set():
                        break
                    try:
                        fix_state = self.fix_det.update(bundle.gaze)
                    except Exception as e:
                        if self.args.verbose:
                            _log(f"frame loop: per-bundle error, skipping: {e}")
                        continue
                    with self._frame_lock:
                        self._latest_bundle = bundle
                        self._latest_bundle_t = time.time()
                        self._latest_fix = fix_state
                        self._frames_received += 1
                # Iterator returned cleanly (only happens on reader.close()).
                # Don't kill the service — the operator may want to keep using
                # cached state via UDP. Wait briefly then re-enter in case the
                # reader can be re-iterated.
                if not self._stop_event.is_set():
                    _log("frame loop: reader iterator returned; retrying")
                    time.sleep(retry_delay_s)
            except Exception as e:
                _log(f"frame loop: reader error, will retry in {retry_delay_s:.1f}s: {e}")
                time.sleep(retry_delay_s)

    def serve_forever(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.args.host, self.args.port))
        self._sock.settimeout(0.5)
        # Windows: disable SIO_UDP_CONNRESET so an ICMP "port unreachable"
        # bounced back from a dead-or-firewalled peer of a previous reply
        # doesn't poison the next recvfrom() with WSAECONNRESET (10054).
        # Without this, the GPU host is silently killed every time a remote
        # operator panel terminates without a clean unsubscribe.
        # Python's socket.ioctl rejects arbitrary IOCTLs; call WSAIoctl
        # directly via ctypes instead.
        if sys.platform == "win32":
            try:
                _disable_udp_connreset(self._sock)
            except (OSError, ValueError, AttributeError) as e:
                _log(f"WARN: could not disable SIO_UDP_CONNRESET ({e}); "
                     f"relying on ConnectionResetError catch in recv loop")
        _log(f"listening on udp://{self.args.host}:{self.args.port}")

        while not self._stop_event.is_set():
            try:
                data, addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except ConnectionResetError as e:
                # WinError 10054 — ICMP unreachable from a stale peer.
                # Belt-and-suspenders in case the ioctl above is unsupported
                # on this Windows build. UDP is connectionless; keep serving.
                if self.args.verbose:
                    _log(f"recvfrom WSAECONNRESET ({e}); continuing")
                continue
            except OSError as e:
                # Real socket failure (e.g. socket was closed by stop()).
                # Only break if we're shutting down anyway.
                if self._stop_event.is_set():
                    break
                _log(f"recvfrom OSError ({e}); continuing")
                continue

            try:
                req = json.loads(data.decode("utf-8", errors="replace"))
            except Exception as e:
                resp = {"ok": False, "error": f"bad json: {e}"}
                self._send(resp, addr)
                continue

            cmd = str(req.get("cmd", "")).lower()
            try:
                resp = self._dispatch(cmd, req, addr)
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

    def _dispatch(self, cmd: str, req: dict, addr: tuple) -> dict:
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
            "subscribe": lambda: self._cmd_subscribe(req, addr),
            "unsubscribe": lambda: self._cmd_unsubscribe(req),
            "stop": lambda: self._cmd_stop(addr),
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
            # Seg-stream telemetry — refreshed every _SEG_STREAM_STATS_S
            # while the loop is running. Stays present (active=False) so
            # clients can render a single status panel regardless of
            # whether streaming is on.
            "seg_stream": dict(self._seg_stream_stats),
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

    # Window size for seg-stream stats accumulation. 5 s is short enough
    # to feel live in the panel readout but long enough that a single slow
    # inference doesn't dominate the average.
    _SEG_STREAM_STATS_S: float = 5.0

    def _segment_stream_loop(self) -> None:
        next_run = time.perf_counter()
        last_hz = self._seg_stream_hz
        period = 1.0 / max(last_hz, 1e-6)
        # Stats accumulators for the current window. Reset after each
        # periodic emit so each line summarises fresh activity.
        win_start = time.perf_counter()
        win_ticks = 0
        win_dets = 0
        win_infer_s = 0.0
        win_errors = 0
        self._seg_stream_stats.update({
            "active": True, "hz_target": last_hz, "last_emit_t": time.time(),
        })
        try:
            while not self._seg_stream_stop.is_set() and not self._stop_event.is_set():
                # Re-read the rate each iteration so live changes take effect.
                if self._seg_stream_hz != last_hz:
                    last_hz = self._seg_stream_hz
                    period = 1.0 / max(last_hz, 1e-6)
                    self._seg_stream_stats["hz_target"] = last_hz

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
                    t0 = time.perf_counter()
                    dets = self.detector.detect(bundle.video.bgr)
                    dets = _filter_overlay_dets(dets)
                    self._cache_dets(dets)
                    win_ticks += 1
                    win_dets += len(dets)
                    win_infer_s += (time.perf_counter() - t0)
                except Exception as e:
                    # Errors always log — silent failure here used to hide
                    # detector misconfiguration (wrong device, missing
                    # weights) until the operator wondered why the overlay
                    # was empty.
                    win_errors += 1
                    _log(f"segment stream error: {e}")

                # Periodic stats emit. Logs one line and refreshes the
                # status-reply snapshot so the panel readout stays current.
                window_s = time.perf_counter() - win_start
                if window_s >= self._SEG_STREAM_STATS_S:
                    achieved = win_ticks / max(window_s, 1e-6)
                    mean_dets = (win_dets / win_ticks) if win_ticks else 0.0
                    mean_infer_ms = (win_infer_s / win_ticks * 1000.0) if win_ticks else 0.0
                    _log(
                        f"seg-stream: target={last_hz:.1f}Hz achieved={achieved:.1f}Hz "
                        f"ticks={win_ticks} mean_dets={mean_dets:.1f} "
                        f"mean_infer={mean_infer_ms:.0f}ms errors={win_errors}"
                    )
                    self._seg_stream_stats.update({
                        "active": True,
                        "hz_target": last_hz,
                        "hz_achieved": achieved,
                        "mean_dets": mean_dets,
                        "mean_infer_ms": mean_infer_ms,
                        "errors": win_errors,
                        "window_s": window_s,
                        "last_emit_t": time.time(),
                    })
                    win_start = time.perf_counter()
                    win_ticks = 0
                    win_dets = 0
                    win_infer_s = 0.0
                    win_errors = 0

                # Schedule next tick relative to the original cadence, but if we're
                # falling behind by more than 2 periods just resync — avoids a
                # runaway catch-up burst after a slow inference.
                next_run += period
                now_pc2 = time.perf_counter()
                if next_run < now_pc2 - 2.0 * period:
                    next_run = now_pc2
        finally:
            # Mark the loop dead so a stale snapshot doesn't make the panel
            # claim the stream is still healthy after a stop.
            self._seg_stream_stats["active"] = False

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

        cmd_t0 = time.time()
        bundle, fix, _ = self._latest()
        if bundle is None:
            _log("decide: no_frame")
            return {"ok": False, "error": "no_frame"}

        gaze_xy = (float(bundle.gaze.x), float(bundle.gaze.y))
        _log(f"decide IN: gaze=({gaze_xy[0]:.0f},{gaze_xy[1]:.0f})")
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
            elapsed_ms = (time.time() - cmd_t0) * 1000.0
            _log(f"decide OUT: vlm_timeout dets={len(dets)} elapsed={elapsed_ms:.0f}ms")
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
        hit_lbl = (hit_waypoint or {}).get("label") if isinstance(hit_waypoint, dict) else None
        elapsed_ms = (time.time() - cmd_t0) * 1000.0
        _log(f"decide OUT: hit={hit_lbl!r} dets={len(dets)} elapsed={elapsed_ms:.0f}ms")
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
            _log("capture_first: no_frame")
            return {"ok": False, "error": "no_frame"}

        gaze_xy = (float(bundle.gaze.x), float(bundle.gaze.y))
        _log(f"capture_first IN: gaze=({gaze_xy[0]:.0f},{gaze_xy[1]:.0f})")
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

        hit_lbl = (processed["hit_waypoint"] or {}).get("label") if isinstance(processed["hit_waypoint"], dict) else None
        _log(
            f"capture_first OUT: snap={snap_id} hit={hit_lbl!r} "
            f"dets={len(processed['detections'])} elapsed={elapsed*1000:.0f}ms"
        )
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
            _log("decide_pair: missing_snapshot_id")
            return {"ok": False, "error": "missing_snapshot_id"}
        first = self._snapshots.get(str(snap_id))
        if first is None:
            _log(f"decide_pair: snapshot {snap_id} not found or expired")
            return {"ok": False, "error": "snapshot_not_found_or_expired"}

        bundle, fix, _ = self._latest()
        if bundle is None:
            _log(f"decide_pair: no_frame (snap={snap_id})")
            return {"ok": False, "error": "no_frame"}

        second_gaze_xy = (float(bundle.gaze.x), float(bundle.gaze.y))
        second_frame = bundle.video.bgr
        _log(f"decide_pair IN: snap={snap_id} second_gaze=({second_gaze_xy[0]:.0f},{second_gaze_xy[1]:.0f})")

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
            elapsed_ms = (time.time() - t0) * 1000.0
            _log(f"decide_pair OUT: vlm_timeout snap={snap_id} elapsed={elapsed_ms:.0f}ms")
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
        first_lbl = (first["hit_waypoint"] or {}).get("label") if isinstance(first["hit_waypoint"], dict) else None
        second_lbl = (second["hit_waypoint"] or {}).get("label") if isinstance(second["hit_waypoint"], dict) else None
        _log(
            f"decide_pair OUT: snap={snap_id} first={first_lbl!r} second={second_lbl!r} "
            f"elapsed={elapsed*1000:.0f}ms"
        )
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

    def _cmd_stop(self, addr: tuple) -> dict:
        """Shut the service down. Honoured only from loopback unless the
        operator explicitly passed --allow-remote-stop. This protects an
        unattended GPU host from being taken offline by a remote panel
        click, which would otherwise force a physical restart."""
        host = addr[0] if addr else ""
        is_local = isinstance(host, str) and (host == "127.0.0.1" or host.startswith("127."))
        if not is_local and not getattr(self.args, "allow_remote_stop", False):
            _log(f"refusing remote stop from {host} (use --allow-remote-stop to override)")
            return {
                "ok": False,
                "error": "remote_stop_disabled",
                "hint": "stop the service locally with Ctrl-C, "
                        "or restart with --allow-remote-stop to allow this",
            }
        self._stop_segment_stream()
        self._stop_event.set()
        return {"ok": True}

    # ── subscribe-mode JSON push (Render_Layer_Refactor.md §3) ────────────

    # Internal tick rate caps the per-subscriber rate. Subscribers may
    # request lower hz; higher requests are clamped at this ceiling.
    _RESULTS_TICK_HZ: float = 20.0
    # Subscribers expire after this long without a refresh. The panel is
    # expected to re-subscribe every ~10 s as a heartbeat — drops dead
    # subscribers automatically when a client crashes.
    _SUBSCRIBER_TTL_S: float = 30.0

    def start_results_push(self) -> None:
        """Spawn the tick thread that broadcasts vlm_results JSON to subscribers."""
        if self._results_tick_thread is not None and self._results_tick_thread.is_alive():
            return
        self._results_tick_stop.clear()
        self._results_tick_thread = threading.Thread(
            target=self._results_tick_loop, daemon=True, name="vlm-results-push",
        )
        self._results_tick_thread.start()
        _log("results push thread started")

    def _cmd_subscribe(self, req: dict, addr: tuple) -> dict:
        """Add (or refresh) a subscriber. Idempotent on (addr, port) so a
        client re-subscribing as a heartbeat doesn't accumulate ghosts."""
        try:
            hz = float(req.get("hz", self._RESULTS_TICK_HZ))
        except (TypeError, ValueError):
            return {"ok": False, "error": "bad_hz"}
        hz = max(0.5, min(hz, self._RESULTS_TICK_HZ))
        ttl_s = float(req.get("ttl_s", self._SUBSCRIBER_TTL_S))
        now = time.monotonic()
        with self._subscribers_lock:
            # Reuse existing id when the same (addr, port) re-subscribes;
            # this keeps the subscriber set stable across heartbeats and
            # lets the client treat subscribe as idempotent.
            existing_id: Optional[str] = None
            for sid, info in self._subscribers.items():
                if info.get("addr") == addr:
                    existing_id = sid
                    break
            sid = existing_id or uuid.uuid4().hex[:12]
            self._subscribers[sid] = {
                "addr": tuple(addr),
                "hz": hz,
                "last_sent_t": 0.0,
                "expires_at": now + ttl_s,
            }
        return {"ok": True, "stream": "results", "subscriber_id": sid, "hz": hz}

    def _cmd_unsubscribe(self, req: dict) -> dict:
        sid = req.get("subscriber_id")
        if not sid:
            return {"ok": False, "error": "missing_subscriber_id"}
        with self._subscribers_lock:
            removed = self._subscribers.pop(str(sid), None)
        return {"ok": True, "removed": bool(removed)}

    def _results_tick_loop(self) -> None:
        """Build one payload per tick, send to each due subscriber. Reads
        cached state under the existing _render_lock — no model calls in
        this thread."""
        push_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        period = 1.0 / max(self._RESULTS_TICK_HZ, 1e-6)
        # Periodic stats — emitted every _RESULTS_LOG_PERIOD_S so the
        # operator can see whether subscribers exist + whether sends are
        # actually happening. Counts reset after each emit.
        stats_period_s = 30.0
        last_emit = time.monotonic()
        ticks_run = 0
        sends_done = 0
        try:
            while not self._stop_event.is_set() and not self._results_tick_stop.is_set():
                t0 = time.monotonic()
                sent_this_tick = self._tick_send_results(push_sock, t0)
                ticks_run += 1
                sends_done += sent_this_tick
                if (t0 - last_emit) >= stats_period_s:
                    with self._subscribers_lock:
                        n_subs = len(self._subscribers)
                    _log(
                        f"results push: subs={n_subs} ticks={ticks_run} "
                        f"sends={sends_done} window={t0 - last_emit:.0f}s"
                    )
                    last_emit = t0
                    ticks_run = 0
                    sends_done = 0
                slept = time.monotonic() - t0
                remaining = period - slept
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            try:
                push_sock.close()
            except OSError:
                pass

    def _tick_send_results(self, push_sock: socket.socket, now: float) -> int:
        """One pass: prune expired, build payload if any subscriber is due,
        send. Returns the number of datagrams successfully transmitted on
        this tick so _results_tick_loop can roll them up into the
        periodic stats line."""
        # Drop expired subscribers up front so we don't serialise for nobody.
        with self._subscribers_lock:
            for sid, info in list(self._subscribers.items()):
                if info["expires_at"] < now:
                    self._subscribers.pop(sid, None)
            due = []
            for sid, info in self._subscribers.items():
                period = 1.0 / max(info["hz"], 1e-6)
                if (now - info["last_sent_t"]) >= period:
                    due.append((sid, info["addr"]))
                    info["last_sent_t"] = now
        if not due:
            return 0
        try:
            payload_dict = self._build_vlm_results_payload()
        except Exception as e:
            # Always log payload build failures — silent failure here used
            # to mask broken cached_dets contents and the panel just kept
            # reporting "intake: connected" with no detections drawn.
            _log(f"results push: payload build failed: {e}")
            return 0
        if payload_dict is None:
            # No frame received from the relay yet. Pushing an empty
            # placeholder (frame_idx=0, detections=[], …) would light
            # the panel's Receive LED green before any real data has
            # flowed end-to-end, breaking the chain-of-causation
            # semantic the operator panel relies on. Skip until we
            # have a real bundle.
            return 0
        payload = json.dumps(payload_dict, default=_json_default).encode("utf-8")
        if len(payload) > 60 * 1024:
            # UDP datagrams >~64 KB get IP-fragmented. Keep payloads small;
            # the panel will fall back to the next tick.
            _log(f"results push: payload {len(payload)} B exceeds 60 KB; skipping tick")
            return 0
        sent = 0
        for _sid, addr in due:
            try:
                push_sock.sendto(payload, addr)
                sent += 1
            except OSError:
                pass
        return sent

    def _build_vlm_results_payload(self) -> Optional[dict]:
        """Snapshot the current detection / hit / fixation / decision state
        as the JSON push payload defined in Render_Layer_Refactor.md §3.

        Returns ``None`` when no frame has been received yet from the
        upstream relay (``bundle is None``). The tick-send loop skips
        the broadcast in that case so subscribers never see a placeholder
        payload before real data is flowing — the panel's "Receive" LED
        is gated on actual content under the chain-of-causation semantic.
        """
        bundle, fix, _ = self._latest()
        if bundle is None:
            return None
        with self._render_lock:
            dets = list(self._cached_dets)
            hit_det = self._cached_hit_det
            hit_wp = self._cached_hit_wp
            state = self._vlm_state
            decision = self._last_decision
            first_det = self._first_snap_det

        detections_out = [_serialize_detection_for_push(d) for d in dets]
        hit_payload = None
        if hit_det is not None:
            hit_id = next((i for i, d in enumerate(dets) if d is hit_det), -1)
            wp_pixel = (hit_wp or {}).get("pixel_center") if isinstance(hit_wp, dict) else None
            hit_payload = {
                "det_id": hit_id,
                "waypoint": list(wp_pixel) if wp_pixel is not None else None,
                "label": getattr(hit_det, "label", None),
            }

        fixation_payload = None
        if fix is not None and getattr(fix, "active", False):
            duration_ns = int(getattr(fix, "duration_ns", 0))
            fx = float(bundle.gaze.x) if bundle is not None else None
            fy = float(bundle.gaze.y) if bundle is not None else None
            fixation_payload = {
                "active": True,
                "duration_ms": duration_ns / 1_000_000.0,
                "x": fx,
                "y": fy,
                "stable": bool(getattr(fix, "is_stable", False)),
            }

        depth_at_gaze_m = None
        if isinstance(decision, dict):
            depth_at_gaze_m = decision.get("depth_at_gaze_m")

        first_object_payload = None
        if first_det is not None:
            first_object_payload = _serialize_detection_for_push(first_det)

        return {
            "type": "vlm_results",
            "frame_idx": int(getattr(getattr(bundle, "video", None), "frame_idx", 0)) if bundle is not None else 0,
            "frame_ts_ns": int(getattr(getattr(bundle, "video", None), "timestamp_ns", 0)) if bundle is not None else 0,
            "ts_send_ns": int(time.time_ns()),
            "detections": detections_out,
            "hit": hit_payload,
            "fixation": fixation_payload,
            "decision": _serialize_decision_for_push(decision) if isinstance(decision, dict) else None,
            "depth_at_gaze_m": depth_at_gaze_m,
            "vlm_state": state,
            "first_object": first_object_payload,
            "gaze_px": [float(bundle.gaze.x), float(bundle.gaze.y)] if bundle is not None else None,
        }

def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"unsupported type for JSON: {type(o)}")


def _serialize_detection_for_push(d) -> dict:
    """Compact JSON-safe view of a harmony_vlm Detection for the UDP push.

    Mask polygons are int-quantised vertex lists (already int via .astype(int)
    in _cmd_segment) — small enough that 5-10 detections at 20 Hz stay
    comfortably under the 60 KB datagram budget."""
    box_xyxy = [float(v) for v in getattr(d, "box_xyxy", (0.0, 0.0, 0.0, 0.0))]
    out = {
        "label": getattr(d, "label", None),
        "confidence": float(getattr(d, "confidence", 0.0)),
        "box_xyxy": box_xyxy,
        "box_center": [float(v) for v in getattr(d, "box_center", (0.0, 0.0))],
    }
    poly = getattr(d, "mask_polygon", None)
    if poly is not None:
        try:
            out["mask_polygon"] = poly.reshape(-1, 2).astype(int).tolist()
        except Exception:
            pass
    return out


def _serialize_decision_for_push(decision: dict) -> dict:
    """Trim a decision dict to fields the renderer actually paints.

    The full _last_decision can hold large nested fields (waypoints lists,
    paired-object metadata). The push payload only carries what the
    Linux-side renderer needs to draw the decision badge."""
    keep = ("text", "object_label", "object", "second_object",
            "ts_ns", "model", "elapsed_s", "summary")
    return {k: decision[k] for k in keep if k in decision}


def _open_service_log_file(arg_value):
    """Open the anonymous service log unless explicitly disabled.

    arg_value semantics:
      - None        → use default (~/.harmony_vlm_logs/)
      - ""/off/none → disable file logging entirely
      - any other   → use that directory verbatim
    The directory is host-local and intentionally NOT under any
    subject path: the GPU host runs across multiple subjects and the
    Linux operator panel owns the subject-tied logs.
    """
    if arg_value is not None and arg_value.strip().lower() in ("", "off", "none"):
        return None, None
    log_dir = arg_value if arg_value else os.path.join(
        os.path.expanduser("~"), ".harmony_vlm_logs"
    )
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        _log(f"WARN: could not create service log dir {log_dir}: {e}; file logging disabled")
        return None, None
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"vlm_service_{ts}.log")
    try:
        fh = open(path, "a", encoding="utf-8")
        fh.write(
            f"# vlm_service log opened {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# host=anonymous service-side log; not tied to any subject\n"
        )
        fh.flush()
    except OSError as e:
        _log(f"WARN: could not open service log file {path}: {e}; file logging disabled")
        return None, None
    return fh, path


def main() -> None:
    args = parse_args()

    # Open the anonymous service log before anything else can _log() —
    # otherwise the early FATAL paths below escape the file. The
    # globals() set is intentional; _log() reads the module attribute.
    global _LOG_FILE
    _log_fh, _log_path = _open_service_log_file(args.service_log_dir)
    _LOG_FILE = _log_fh
    if _log_path is not None:
        _log(f"service log file: {_log_path}")

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
    # conf_threshold raised from harmony_vlm's 0.4 default — at 0.4 the
    # FastSAM-everything output is dense enough (40+ masks under bright lab
    # lighting) to swamp the Linux overlay's per-detection alpha blend.
    detector = ObjectDetector(
        model_size=args.seg_model, device=args.device, conf_threshold=0.6,
    )

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
    # Always run the JSON push loop. Subscribers self-register; if none ever
    # subscribe the loop is essentially idle (one prune pass per tick).
    service.start_results_push()

    # Keep Windows awake while the service runs (no-op on POSIX). Without
    # this an unattended GPU host can sleep mid-session and the operator
    # has to physically wake it.
    bci_root = os.path.dirname(os.path.abspath(__file__))
    if bci_root not in sys.path:
        sys.path.insert(0, bci_root)
    from Utils.sleep_inhibit import inhibit as _sleep_inhibit, release as _sleep_release
    _sleep_inhibit()

    _log("ready")
    try:
        service.serve_forever()
    except KeyboardInterrupt:
        _log("KeyboardInterrupt — stopping")
    finally:
        service.stop()
        _sleep_release()
        _log("stopped")
        # Close the anonymous service log last so the "stopped" line
        # makes it onto disk before the handle goes away. The `global`
        # declaration at the top of main() (line ~1336) already covers
        # this scope; a second `global` here is a SyntaxError because
        # the name has been assigned earlier in the function.
        fh = _LOG_FILE
        _LOG_FILE = None
        if fh is not None:
            try:
                fh.write(f"# vlm_service log closed {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                fh.close()
            except OSError:
                pass


if __name__ == "__main__":
    main()
