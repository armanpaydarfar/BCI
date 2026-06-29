#!/usr/bin/env python3
"""
vlm_service.py — UDP request-reply service wrapping harmony_vlm's capabilities.

Runs in the unified `lsl` conda env (the separate `harmony_vlm` env is retired)
and imports the in-tree `perception` package. Loads the Neon live reader, FastSAM,
Depth Pro (optional), Gemini, and the I-VT fixation detector once at startup; then
exposes each capability as a distinct UDP command so the control panel and
experiment drivers can consume them independently.

Mirrors the request-reply idiom of gaze_runner.py:GazeUDPServer (L43-204) —
single-threaded dispatch, JSON in / JSON out, single datagram round-trip.

Internal structure (this is a large module — the map a newcomer needs):
    VLMService is the single hub class. It owns (1) the blocking UDP server loop +
    a dict-based command dispatch (`cmd` → `_cmd_*` handler), (2) a background
    thread that ingests the latest Neon frame+gaze from the frame relay, (3) the
    model handles loaded once at startup (segmenter / depth / reasoner / fixation),
    (4) the per-command handlers (`_cmd_status/segment/depth/reason/decide/...`),
    and (5) caches for the latest snapshot, segmentation, and rendered overlay plus
    the subscription push channel. The `decide` and `decide_pair` handlers compose
    segment→depth→reason into the waypoint dict the drivers consume.

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


# Pull argparse defaults from BCI/config.py (which layers config_local.py over
# committed defaults). The service may be launched from a different cwd, so
# config.py is not guaranteed on sys.path — insert the BCI dir explicitly. Falls
# back to hardcoded safe defaults if the import fails, so a stripped-down
# deployment without config_local still works.
_BCI_DIR = Path(__file__).resolve().parent
if str(_BCI_DIR) not in sys.path:
    sys.path.insert(0, str(_BCI_DIR))
try:
    import config as _bci_config
except Exception as _cfg_exc:  # noqa: BLE001 — documented degradation (see above)
    # Safe, intentional fallback to hardcoded defaults; surface a breadcrumb so a
    # config that was *expected* but failed to import isn't silently ignored.
    print(f"[vlm_service] WARN: BCI config import failed ({_cfg_exc!r}); using "
          "hardcoded argparse defaults", file=sys.stderr)
    _bci_config = None


def _cfg_default(name: str, fallback):
    if _bci_config is None:
        return fallback
    return getattr(_bci_config, name, fallback)


# Pure module-level helpers extracted into leaf modules under vlm/ (behaviour-
# preserving). Some are referenced by VLMService methods / parse_args as bare
# names; _opt_float / _median_mask_depth / _filter_overlay_dets are re-exported
# here for tests/test_seg_constraints.py, which imports them from vlm_service to
# pin the seg-op contract at the service's public surface. The overlay-cap
# constants (_OVERLAY_*) live in vlm/seg_ops.py as the defaults
# _filter_overlay_dets binds; parse_args() / VLMService.__init__ use them.
from vlm.seg_ops import (
    SegConstraints,
    _opt_float,
    _median_mask_depth,
    _apply_seg_constraints,
    _filter_overlay_dets,
    _gaze_hit_recognize,
    _OVERLAY_TOP_K,
    _OVERLAY_CONTAIN_RATIO,
    _OVERLAY_AREA_RATIO,
)
from vlm.snapshot_cache import SnapshotCache

# Subsystems extracted from the VLMService god class (behaviour-preserving): each
# is a plain collaborator holding a back-reference to the service, constructed in
# __init__ after the state it reads exists. ResultsPusher owns the subscribe-mode
# JSON push (subscribers + tick thread); SegmentStream owns the continuous
# segmentation stream (worker thread + tracker + stats).
from vlm.results_pusher import ResultsPusher
from vlm.segment_stream import SegmentStream


def parse_args():
    p = argparse.ArgumentParser(description="harmony_vlm UDP service")
    # `--host` is the bind address for both the UDP request socket and the
    # TCP overlay socket. Production deployments set this to 0.0.0.0 so the
    # Linux panel/driver can dial in across the LAN; single-machine dev
    # keeps it on 127.0.0.1.
    p.add_argument("--host", default=_cfg_default("VLM_BIND_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=_cfg_default("VLM_SERVICE_PORT", 5589))
    p.add_argument("--neon-host", default=_cfg_default("NEON_COMPANION_HOST", ""),
                   help="Empty string triggers LAN discovery")
    p.add_argument("--model", default=_cfg_default("VLM_MODEL", "gemini-2.5-flash"),
                   help="VLM model name")
    # IntentReasoner's upstream default (harmony_vlm) is max_tokens=1024,
    # which truncates the JSON response on scenes with many candidates
    # (each candidate is ~50-100 output tokens; 18 dets already overflows).
    # 8192 gives ~4-5x headroom for typical scenes and is well under
    # Gemini 2.5 Flash's 65,536 output-token ceiling. Billing is per emitted
    # token, not per cap, so raising it costs nothing when responses are short.
    p.add_argument("--max-output-tokens", type=int,
                   default=int(_cfg_default("VLM_MAX_OUTPUT_TOKENS", 8192)),
                   help="Cap for the VLM JSON response in tokens. Override "
                        "with VLM_MAX_OUTPUT_TOKENS in config. Default 8192.")
    # Gemini "thinking" budget. The committed default is 0 (thinking disabled)
    # — benchmark-chosen for ~2.9× lower decide latency (see config.py
    # VLM_THINKING_BUDGET). A None value passes no thinking_config (Gemini's own
    # high default budget, the pre-2026-06-17 behaviour); a positive int caps
    # thinking without disabling it. type=int only coerces a supplied value, so
    # a config default of None survives. Honoured only by the Gemini backend.
    p.add_argument("--vlm-thinking-budget", type=int, dest="vlm_thinking_budget",
                   default=_cfg_default("VLM_THINKING_BUDGET", 0),
                   help="Gemini thinking budget in tokens. 0 (default) = disable "
                        "thinking for lowest latency; a positive int caps it. "
                        "Override with VLM_THINKING_BUDGET in config.")
    p.add_argument("--seg-model", default="FastSAM-s.pt",
                   help="Segmentation weights: filename within PERCEPTION_MODELS_DIR, or an absolute path")

    # ── Segmentation tuning (WS4 F1 + E1/E2) ─────────────────────────────────
    # All default to today's behaviour: conf 0.6, overlay knobs at the module
    # constants, and every F1 constraint OFF (None) so output is unchanged until
    # a threshold is set. conf/overlay knobs (E2) make the previously-hardcoded
    # values configurable; the seg-* constraints (F1/E1) bias toward small
    # tabletop objects when enabled.
    seg = p.add_argument_group("segmentation tuning (WS4 F1/E2)")
    seg.add_argument("--seg-conf-threshold", type=float,
                     default=float(_cfg_default("SEG_CONF_THRESHOLD", 0.6)),
                     help="FastSAM min confidence (was hardcoded 0.6). "
                          "Override with SEG_CONF_THRESHOLD in config.")
    seg.add_argument("--overlay-top-k", type=int,
                     default=int(_cfg_default("SEG_OVERLAY_TOP_K", _OVERLAY_TOP_K)),
                     help="Cap on detections kept for the live overlay "
                          "(seg-stream cache), by confidence. Default 20.")
    seg.add_argument("--overlay-contain-ratio", type=float,
                     default=float(_cfg_default("SEG_OVERLAY_CONTAIN_RATIO", _OVERLAY_CONTAIN_RATIO)),
                     help="Overlay containment-drop ratio (child mostly inside "
                          "parent). Default 0.85.")
    seg.add_argument("--overlay-area-ratio", type=float,
                     default=float(_cfg_default("SEG_OVERLAY_AREA_RATIO", _OVERLAY_AREA_RATIO)),
                     help="Overlay containment-drop max child/parent area ratio. "
                          "Default 0.5.")
    seg.add_argument("--seg-max-area-ratio", type=float,
                     default=_cfg_default("SEG_MAX_AREA_RATIO", None),
                     help="F1: drop dets whose bbox covers more than this "
                          "fraction of the frame (kills whole-scene blobs). "
                          "Unset = off.")
    seg.add_argument("--seg-min-area-ratio", type=float,
                     default=_cfg_default("SEG_MIN_AREA_RATIO", None),
                     help="F1: drop dets whose bbox is below this fraction of "
                          "the frame (specks). Unset = off.")
    seg.add_argument("--seg-solidity-min", type=float,
                     default=_cfg_default("SEG_SOLIDITY_MIN", None),
                     help="F1: drop dets whose mask-area/bbox-area is below this "
                          "(merged/elongated blobs). Unset = off.")
    seg.add_argument("--seg-depth-band", type=float, nargs=2, metavar=("NEAR_M", "FAR_M"),
                     default=_cfg_default("SEG_DEPTH_BAND", None),
                     help="F1: keep only dets whose median mask depth is within "
                          "[NEAR_M, FAR_M] metres. decide-only (needs depth). "
                          "Unset = off.")
    seg.add_argument("--seg-gaze-roi", type=float, metavar="FRAC",
                     default=_cfg_default("SEG_GAZE_ROI", None),
                     help="F1: keep only dets whose bbox centre is within "
                          "FRAC*frame (half-extent) of the current gaze. "
                          "Unset = off.")
    seg.add_argument("--seg-track", action=argparse.BooleanOptionalAction,
                     default=bool(_cfg_default("SEG_TRACK", False)),
                     help="F3: temporal tracking/smoothing of the seg-stream "
                          "(SORT, IoU + min_hits/max_age hysteresis; stable "
                          "track_id). Off by default (stateless overlay).")
    # F5: optional fast COCO recognizer. Empty (default) = the `recognize`
    # command returns recognizer_disabled and no extra model is loaded. A
    # filename resolves under PERCEPTION_MODELS_DIR (like --seg-model); the
    # name must NOT contain "fastsam" so ObjectDetector loads it as a
    # COCO-labelled YOLO model rather than class-agnostic FastSAM.
    seg.add_argument("--recognizer-model", default=_cfg_default("VLM_RECOGNIZER_MODEL", ""),
                     help="F5: YOLO weights for fast COCO recognition of the "
                          "gaze object (filename within PERCEPTION_MODELS_DIR "
                          "or absolute path). Empty = recognizer disabled.")

    # Default "auto" resolves to cuda if torch.cuda.is_available(), else cpu
    # (see main()). Hosts without a usable GPU degrade gracefully to CPU,
    # matching CLAUDE.md's "platform-specific optimisations must be gated on
    # torch.cuda.is_available()" rule.
    p.add_argument("--device", default="auto",
                   help="Compute device: 'auto' (default; cuda if available, "
                        "else cpu), 'cuda', or 'cpu'.")
    # Depth is on by default; pass --no-enable-depth to disable. Loading
    # depth_pro.pt adds ~1-2 s and ~1 GB of VRAM on cuda (or ~3 GB RAM on cpu),
    # but the downstream panel/driver consumes depth whenever perception runs,
    # so the right default is "on" — opt-out rather than opt-in.
    p.add_argument("--enable-depth", action=argparse.BooleanOptionalAction,
                   default=bool(_cfg_default("VLM_ENABLE_DEPTH", True)))
    p.add_argument("--depth-checkpoint", default="depth_pro.pt",
                   help="Depth Pro weights: filename within PERCEPTION_MODELS_DIR, or an absolute path")
    p.add_argument("--session-dir", default=None, help="Where to save depth PNGs etc.")
    p.add_argument("--verbose", action="store_true")
    # Frame source toggle for the GPU-host migration plan (see SoftwareDocs/
    # projects/harmony-bci/gpu-service/architecture-plan.md §3.4). Default `local` preserves
    # today's behaviour (open Neon directly via NeonLiveReader). `remote`
    # consumes envelopes from a Utils/frame_relay.py TCP server instead.
    p.add_argument("--frame-source", choices=["local", "remote"],
                   default=_cfg_default("PERCEPTION_FRAME_SOURCE", "local"),
                   help="local=open Neon directly; remote=consume Utils/frame_relay envelopes")
    p.add_argument("--remote-frame-host",
                   default=_cfg_default("FRAME_RELAY_DIAL_HOST", None),
                   help="Host of the frame_relay server (required when --frame-source=remote)")
    p.add_argument("--remote-frame-port", type=int,
                   default=_cfg_default("FRAME_RELAY_PORT", 5591),
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
    def __init__(self, args, *, reader, detector, depth_estimator, reasoner, fix_det, fixation_state_cls,
                 recognizer=None):
        self.args = args
        self.reader = reader
        self.detector = detector
        self.depth_estimator = depth_estimator
        self.reasoner = reasoner
        self.fix_det = fix_det
        self._FixationState = fixation_state_cls
        # WS4 F5: optional fast COCO recognizer (a YOLO ObjectDetector). None
        # unless --recognizer-model / VLM_RECOGNIZER_MODEL is set; the recognize
        # command returns recognizer_disabled in that case. Names the gaze
        # object without a Gemini round-trip.
        self.recognizer = recognizer

        # WS4 F1/E2: optional segmentation constraints + configurable overlay
        # reduction knobs, resolved from CLI/config. getattr fallbacks keep the
        # stub-args test path (and any minimal Namespace) on today's behaviour:
        # constraints inactive, overlay knobs at the module defaults.
        self._seg_constraints = SegConstraints.from_args(args)
        self._overlay_top_k = int(getattr(args, "overlay_top_k", _OVERLAY_TOP_K))
        self._overlay_contain_ratio = float(
            getattr(args, "overlay_contain_ratio", _OVERLAY_CONTAIN_RATIO))
        self._overlay_area_ratio = float(
            getattr(args, "overlay_area_ratio", _OVERLAY_AREA_RATIO))

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

        # Subsystems extracted from this hub (each holds a back-ref to self and
        # owns its own thread + state). Constructed last so the shared state they
        # read through the back-ref (_frame_lock/_latest, _render_lock/_cached_*,
        # detector, _seg_constraints, the overlay knobs, _stop_event) already
        # exists. SegmentStream owns the continuous-segmentation worker thread +
        # tracker + stats; ResultsPusher owns the subscribe-mode JSON push
        # (subscribers + tick thread, Render_Layer_Refactor.md §3).
        self.segment_stream = SegmentStream(self)
        self.results_pusher = ResultsPusher(self)

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
        # Stop the subsystem threads. Both loops also gate on _stop_event (set
        # above), so this is belt-and-suspenders for the results push and a
        # join for the seg-stream worker; ordering matches the pre-refactor
        # teardown (seg-stream first via _cmd_stop, push loop on _stop_event).
        self.segment_stream.stop()
        self.results_pusher.stop()
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
            "segment_stream": lambda: self.segment_stream.cmd_segment_stream(req),
            "recognize": lambda: self._cmd_recognize(req),
            "depth": lambda: self._cmd_depth(req),
            "waypoints": lambda: self._cmd_waypoints(req),
            "reason": lambda: self._cmd_reason(req),
            "decide": lambda: self._cmd_decide(req),
            "capture_first": lambda: self._cmd_capture_first(req),
            "decide_pair": lambda: self._cmd_decide_pair(req),
            "camera_matrix": lambda: self._cmd_camera_matrix(),
            "subscribe": lambda: self.results_pusher.cmd_subscribe(req, addr),
            "unsubscribe": lambda: self.results_pusher.cmd_unsubscribe(req),
            "verify_chain": lambda: self.results_pusher.cmd_verify_chain(req),
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
            "seg_stream": dict(self.segment_stream._seg_stream_stats),
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
        # F1 constraints (no-op unless enabled). No depth map here, so the
        # depth-band constraint is skipped; geometry + gaze-ROI still apply.
        dets = _apply_seg_constraints(
            dets, bundle.video.bgr.shape, self._seg_constraints,
            gaze_xy=(float(bundle.gaze.x), float(bundle.gaze.y)),
        )
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

    def _cmd_recognize(self, req: dict) -> dict:
        """F5: fast COCO-class naming of the gaze object — no Gemini round-trip.

        Runs the optional YOLO recognizer on the latest frame, hit-tests the
        gaze point against the detected boxes, and returns the class name of the
        object being looked at (plus the full detection list). A lightweight
        confirmation step the operator can fire before paying the high-latency
        VLM `decide`. Returns ``recognizer_disabled`` when the service was
        started without --recognizer-model.
        """
        if self.recognizer is None:
            return {"ok": False, "error": "recognizer_disabled"}
        bundle, _, _ = self._latest()
        if bundle is None:
            return {"ok": False, "error": "no_frame"}

        gaze_xy = (float(bundle.gaze.x), float(bundle.gaze.y))
        t0 = time.time()
        dets = self.recognizer.detect(bundle.video.bgr)
        elapsed = time.time() - t0

        out = [
            {
                "label": d.label,
                "confidence": float(d.confidence),
                "box_xyxy": [float(v) for v in d.box_xyxy],
                "box_center": [float(v) for v in d.box_center],
            }
            for d in dets
        ]
        hit = _gaze_hit_recognize(dets, gaze_xy)
        hit_payload = None
        if hit is not None:
            hit_payload = {
                "label": hit.label,
                "confidence": float(hit.confidence),
                "box_xyxy": [float(v) for v in hit.box_xyxy],
            }
        _log(f"recognize OUT: hit={(hit_payload or {}).get('label')!r} "
             f"dets={len(dets)} elapsed={elapsed * 1000:.0f}ms")
        return {
            "ok": True,
            "label": (hit_payload or {}).get("label"),
            "hit": hit_payload,
            "detections": out,
            "n": len(out),
            "elapsed_s": elapsed,
        }

    def _cmd_depth(self, req: dict) -> dict:
        at_gaze = bool(req.get("at_gaze", True))
        save = bool(req.get("save", False))
        if self.depth_estimator is None:
            _log("depth: depth_disabled (start vlm_service with --enable-depth)")
            return {"ok": False, "error": "depth_disabled"}
        bundle, _, _ = self._latest()
        if bundle is None:
            _log("depth: no_frame")
            return {"ok": False, "error": "no_frame"}

        gaze_xy = (float(bundle.gaze.x), float(bundle.gaze.y))
        _log(f"depth IN: gaze=({gaze_xy[0]:.0f},{gaze_xy[1]:.0f}) "
             f"at_gaze={at_gaze} save={save}")

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
        gaze_str = ""
        if at_gaze:
            h, w = depth_map.shape[:2]
            gx = int(np.clip(round(gaze_xy[0]), 0, w - 1))
            gy = int(np.clip(round(gaze_xy[1]), 0, h - 1))
            resp["depth_at_gaze_m"] = float(depth_map[gy, gx])
            gaze_str = f" at_gaze={resp['depth_at_gaze_m']:.2f}m"
        _log(f"depth OUT: shape={depth_map.shape[0]}x{depth_map.shape[1]} "
             f"median={resp['depth_median_m']:.2f}m "
             f"range=[{resp['depth_min_m']:.2f},{resp['depth_max_m']:.2f}]m"
             f"{gaze_str}"
             f"{' saved' if saved_path else ''} "
             f"elapsed={elapsed * 1000.0:.0f}ms")
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

    @staticmethod
    def _hit_det_and_waypoint(dets, waypoints_out, gaze_xy):
        """Find the detection whose bbox contains the gaze point, paired with
        its 3D waypoint BY LABEL — never by positional zip.

        compute_3d_waypoints (perception/object_detector.py) skips detections
        whose mask region has no valid depth, so ``waypoints_out`` can be
        shorter than ``dets``; a positional ``zip(dets, waypoints_out)`` would
        then pair the gaze hit with a *different* object's 3D point. FastSAM
        emits unique per-frame ``segment_N`` labels that compute_3d_waypoints
        copies onto each waypoint, so the label lookup is exact. Returns
        ``hit_waypoint=None`` when the hit detection had no valid depth (correct
        — there is no 3D point for it), instead of a misattributed one.
        """
        hit_det = None
        for d in dets:
            x1, y1, x2, y2 = d.box_xyxy
            if x1 <= gaze_xy[0] <= x2 and y1 <= gaze_xy[1] <= y2:
                hit_det = d
                break
        hit_waypoint = None
        if hit_det is not None:
            hit_waypoint = next(
                (wp for wp in waypoints_out if wp.get("label") == hit_det.label),
                None,
            )
        return hit_det, hit_waypoint

    def _segment_depth_waypoints(self, frame_bgr, gaze_xy, *, apply_constraints: bool, depth_log_tag: str):
        """Shared segment → depth → 3D-waypoints → gaze hit-test pipeline behind
        both the decide path and the two-object (capture/pair) path.

        Returns ``(dets, waypoints_out, depth_at_gaze, hit_det, hit_waypoint)``.

        ``apply_constraints`` gates the F1 ``_apply_seg_constraints`` filter — the
        ONE behavioural divergence between the two former copies, now an explicit
        flag rather than two drifting code paths:
          * decide path (True): filter detections BEFORE computing waypoints /
            hit-testing / reasoning, so everything downstream sees one set.
          * capture/pair path (False): no filter — waypoint + hit-test every det.
        """
        # Import here so this file parses even if harmony_vlm utils aren't loaded
        from perception.object_detector import compute_3d_waypoints

        dets = self.detector.detect(frame_bgr)

        waypoints_out: list[dict] = []
        depth_at_gaze: Optional[float] = None
        if self.depth_estimator is not None:
            _depth_t0 = time.time()
            depth_map, _ = self.depth_estimator.estimate(
                frame_bgr, f_px=self._focal_px(), gaze_xy=gaze_xy,
            )
            _depth_elapsed_ms = (time.time() - _depth_t0) * 1000.0
            if apply_constraints:
                # F1 constraints (no-op unless enabled). Depth map is available
                # here, so the depth-band constraint can run; filter before
                # waypoints/hit-test/reasoning so all downstream sees the same set.
                dets = _apply_seg_constraints(
                    dets, frame_bgr.shape, self._seg_constraints,
                    depth_map=depth_map, gaze_xy=gaze_xy,
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
            depth_at_gaze = float(depth_map[gy, gx])
            _log(f"  depth ({depth_log_tag}): shape={depth_map.shape[0]}x{depth_map.shape[1]} "
                 f"at_gaze={depth_at_gaze:.2f}m elapsed={_depth_elapsed_ms:.0f}ms")
        elif apply_constraints:
            # No depth estimator: apply the geometry + gaze-ROI constraints
            # (depth-band needs a depth map and is silently skipped here).
            dets = _apply_seg_constraints(
                dets, frame_bgr.shape, self._seg_constraints, gaze_xy=gaze_xy,
            )

        hit_det, hit_waypoint = self._hit_det_and_waypoint(dets, waypoints_out, gaze_xy)
        return dets, waypoints_out, depth_at_gaze, hit_det, hit_waypoint

    def _cmd_decide(self, req: dict) -> dict:
        cmd_t0 = time.time()
        bundle, fix, _ = self._latest()
        if bundle is None:
            _log("decide: no_frame")
            return {"ok": False, "error": "no_frame"}

        gaze_xy = (float(bundle.gaze.x), float(bundle.gaze.y))
        _log(f"decide IN: gaze=({gaze_xy[0]:.0f},{gaze_xy[1]:.0f})")

        # Shared pipeline; apply_constraints=True filters detections before
        # waypoints/hit-test/reasoning (the decide-path divergence).
        dets, waypoints_out, depth_at_gaze_m, hit_det, hit_waypoint = self._segment_depth_waypoints(
            bundle.video.bgr, gaze_xy, apply_constraints=True, depth_log_tag="in-decide",
        )

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

    def _cmd_waypoints(self, req: dict) -> dict:
        """Fast 3-D-waypoints path for WS4's live control loop: ``decide`` minus
        the Gemini reasoner.

        Runs the exact shared segment → depth → 3D-waypoints → gaze hit-test
        pipeline that ``_cmd_decide`` runs (``apply_constraints=True``, same dets
        cached for the overlay), but DOES NOT call ``reasoner.reason_async`` and
        DOES NOT enter the THINKING overlay state. The reasoner is unusable in a
        per-fixation control loop — Gemini's 30-40 s round-trip dwarfs the loop
        period — whereas the geometry pipeline (FastSAM + Depth Pro) returns in
        ~1 s, so this exposes just the 3-D output the loop needs.

        Returns ``waypoints`` / ``hit_waypoint`` / ``depth_at_gaze_m`` /
        ``gaze_px``. When the service was started without --enable-depth there is
        no depth map, so ``waypoints`` is empty and ``hit_waypoint`` is None;
        ``depth_enabled: False`` is set so the caller can distinguish "depth off"
        from "nothing at gaze".
        """
        cmd_t0 = time.time()
        bundle, _, _ = self._latest()
        if bundle is None:
            _log("waypoints: no_frame")
            return {"ok": False, "error": "no_frame"}

        gaze_xy = (float(bundle.gaze.x), float(bundle.gaze.y))
        _log(f"waypoints IN: gaze=({gaze_xy[0]:.0f},{gaze_xy[1]:.0f})")

        dets, waypoints_out, depth_at_gaze_m, hit_det, hit_waypoint = self._segment_depth_waypoints(
            bundle.video.bgr, gaze_xy, apply_constraints=True, depth_log_tag="in-waypoints",
        )
        self._cache_dets(dets, hit_det, hit_waypoint)

        resp: Dict[str, Any] = {
            "ok": True,
            "waypoints": waypoints_out,
            "hit_waypoint": hit_waypoint,
            "depth_at_gaze_m": depth_at_gaze_m,
            "gaze_px": list(gaze_xy),
        }
        if self.depth_estimator is None:
            resp["depth_enabled"] = False
        hit_lbl = (hit_waypoint or {}).get("label") if isinstance(hit_waypoint, dict) else None
        elapsed_ms = (time.time() - cmd_t0) * 1000.0
        _log(f"waypoints OUT: hit={hit_lbl!r} dets={len(dets)} "
             f"wps={len(waypoints_out)} elapsed={elapsed_ms:.0f}ms")
        return resp

    def _process_frame_and_gaze(self, frame_bgr, gaze_xy: tuple[float, float]) -> dict:
        """Segment + depth + waypoints + hit-test for an arbitrary frame.

        Returns native Detection objects (for passing into reason_async_pair)
        alongside JSON-safe waypoint dicts, so one call covers both the cache
        payload and the UDP response payload. apply_constraints=False: unlike
        the decide path, the two-object flow does NOT filter detections.
        """
        dets, waypoints_out, depth_at_gaze, hit_det, hit_waypoint = self._segment_depth_waypoints(
            frame_bgr, gaze_xy, apply_constraints=False, depth_log_tag="in-frame",
        )

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
        self.segment_stream.stop()
        self._stop_event.set()
        return {"ok": True}

# Wire-protocol JSON serialization helper extracted into vlm/wire.py
# (behaviour-preserving — re-imported so _send references it as a bare name
# exactly as before). The detection/decision serializers also live there but are
# now imported directly by vlm/results_pusher.py, the only remaining caller.
from vlm.wire import _json_default


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

    # Perception source is in-tree under BCI/perception/ (folded from
    # harmony_vlm, WS3). The BCI dir is already on sys.path (_BCI_DIR at the
    # module top), so the perception package and config import directly — no
    # --repo-dir, no os.chdir, no .env file.

    # Resolve model weights against PERCEPTION_MODELS_DIR (machine-local). A
    # bare filename joins onto the models dir; an absolute --seg-model /
    # --depth-checkpoint override is taken as-is.
    models_dir = _cfg_default("PERCEPTION_MODELS_DIR", "")
    if not os.path.isabs(args.seg_model):
        args.seg_model = os.path.join(models_dir, args.seg_model)
    if args.enable_depth and not os.path.isabs(args.depth_checkpoint):
        args.depth_checkpoint = os.path.join(models_dir, args.depth_checkpoint)
    if args.recognizer_model and not os.path.isabs(args.recognizer_model):
        args.recognizer_model = os.path.join(models_dir, args.recognizer_model)

    # Resolve --device=auto before any model loading. Gated on
    # torch.cuda.is_available() so the same default works on GPU hosts (cuda)
    # and CPU-only Linux hosts (cpu).
    if args.device == "auto":
        import torch as _torch_probe
        args.device = "cuda" if _torch_probe.cuda.is_available() else "cpu"
        _log(f"--device auto-resolved to {args.device}")

    from perception.neon import NeonLiveReader
    from perception.object_detector import ObjectDetector
    from perception.fixation_detector import FixationDetector, FixationState
    from perception.intent_reasoner import IntentReasoner

    # Gemini/OpenAI key from BCI config (GOOGLE_API_KEY in config_local.py),
    # with an env-var fallback for ad-hoc runs.
    api_key = (
        _cfg_default("GOOGLE_API_KEY", "")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        _log("FATAL: no GOOGLE_API_KEY in config_local.py or environment")
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

    _log(f"loading segmentation model: {args.seg_model} on {args.device} "
         f"(conf={args.seg_conf_threshold})…")
    # conf_threshold default 0.6 raised from harmony_vlm's 0.4 — at 0.4 the
    # FastSAM-everything output is dense enough (40+ masks under bright lab
    # lighting) to swamp the Linux overlay's per-detection alpha blend. Now
    # configurable via --seg-conf-threshold / SEG_CONF_THRESHOLD (WS4 E2).
    detector = ObjectDetector(
        model_size=args.seg_model, device=args.device,
        conf_threshold=args.seg_conf_threshold,
    )

    # WS4 F5: optional fast COCO recognizer (a second, YOLO-backed
    # ObjectDetector). Only loaded when --recognizer-model is set; otherwise the
    # recognize command reports recognizer_disabled and nothing extra is loaded.
    recognizer = None
    if args.recognizer_model:
        _log(f"loading fast recognizer: {args.recognizer_model} on {args.device}…")
        recognizer = ObjectDetector(
            model_size=args.recognizer_model, device=args.device,
            conf_threshold=args.seg_conf_threshold,
        )

    depth_estimator = None
    if args.enable_depth:
        from perception.depth_estimator import DepthEstimator
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
    reasoner = IntentReasoner(
        api_key=api_key,
        model=args.model,
        max_tokens=args.max_output_tokens,
        thinking_budget=args.vlm_thinking_budget,
    )
    fix_det = FixationDetector()

    service = VLMService(
        args,
        reader=reader,
        detector=detector,
        depth_estimator=depth_estimator,
        reasoner=reasoner,
        fix_det=fix_det,
        fixation_state_cls=FixationState,
        recognizer=recognizer,
    )
    service.start_frame_thread()
    # Always run the JSON push loop. Subscribers self-register; if none ever
    # subscribe the loop is essentially idle (one prune pass per tick).
    service.results_pusher.start()

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
