#!/usr/bin/env python3
"""
Neon unified runner/service.

Modes:
  --mode runner   : UI + keyboard (ESC quit, c recenter)
  --mode service  : JSON-over-TCP commands (status/snapshot/recenter/stop),
                    with optional UI preview via --display 1

Headless semantics:
  --display 0 does NOT disable CV.
  Frames are still consumed + YOLO runs; we simply don't render a window.

Performance:
  - Runner uses include_frame=True (UI needs pixels)
  - Service uses include_frame=False by default (telemetry only, avoids buf.copy)
"""

import os

# Hard clamp thread pools (must be before torch/ultralytics import anywhere)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")

import argparse
import json
import socket
import sys
import threading
import time
import uuid
from typing import Any, Dict, Optional, Tuple

from Utils.gaze.gaze_system import GazeSystem, GazeConfig
from Utils.gaze.gaze_ui import GazeUI, UIConfig, RenderInputs




class GazeUDPServer(threading.Thread):
    """
    JSON request/response over UDP.
    """
    def __init__(
        self,
        host: str,
        port: int,
        system: GazeSystem,
        stop_event: threading.Event,
        *,
        quiet: bool = True,
        allow_shutdown: bool = True,
        allow_remote_stop: bool = False,
        max_req_bytes: int = 64 * 1024,
        max_resp_bytes: int = 64 * 1024,
        udp_log: bool = False,
        udp_log_hz: float = 10.0,
    ):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.system = system
        self.stop_event = stop_event
        self.quiet = quiet
        self.allow_shutdown = allow_shutdown
        self.allow_remote_stop = bool(allow_remote_stop)
        self.max_req_bytes = int(max_req_bytes)
        self.max_resp_bytes = int(max_resp_bytes)
        self.udp_log = bool(udp_log)
        self.udp_log_hz = float(udp_log_hz)
        self._sock: Optional[socket.socket] = None

        self._last_log_t = 0.0

        # Subscribe-mode push (Render_Layer_Refactor.md §3). One tick thread
        # builds `gaze_results` JSON datagrams and broadcasts to subscribed
        # panels. Existing request-reply commands keep working unchanged.
        self._subscribers_lock = threading.Lock()
        # subscriber_id (uuid hex) → {addr, hz, last_sent_t, expires_at}
        self._subscribers: Dict[str, Dict[str, Any]] = {}
        self._tick_thread: Optional[threading.Thread] = None
        self._push_sock: Optional[socket.socket] = None

    def log(self, msg: str):
        if not self.quiet:
            print(f"[gaze_udp] {msg}")

    def _maybe_log(self, msg: str):
        """Rate-limited, and only when udp_log is enabled."""
        if not self.udp_log:
            return
        now = time.time()
        min_dt = 1.0 / max(self.udp_log_hz, 1e-6)
        if (now - self._last_log_t) < min_dt:
            return
        self._last_log_t = now
        print(msg)

    def run(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(0.5)
        # Windows: disable SIO_UDP_CONNRESET so an ICMP "port unreachable"
        # bounced back from a dead-or-firewalled previous reply target
        # doesn't poison the next recvfrom() with WSAECONNRESET (10054)
        # and silently kill the gaze service. Same fix as vlm_service.py.
        if sys.platform == "win32":
            try:
                _SIO_UDP_CONNRESET = 0x9800000C
                self._sock.ioctl(_SIO_UDP_CONNRESET, False)
            except OSError as e:
                self.log(f"WARN: could not disable SIO_UDP_CONNRESET: {e}")
        self.log(f"listening on udp://{self.host}:{self.port}")

        # Tick thread broadcasts gaze_results datagrams to subscribed
        # clients. Idle when no subscribers — one prune pass per tick.
        self._push_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._tick_thread = threading.Thread(
            target=self._results_tick_loop, daemon=True, name="gaze-results-push",
        )
        self._tick_thread.start()

        try:
            while not self.stop_event.is_set():
                try:
                    data, addr = self._sock.recvfrom(self.max_req_bytes)
                except socket.timeout:
                    continue
                except ConnectionResetError as e:
                    # WinError 10054 — belt-and-suspenders if SIO_UDP_CONNRESET
                    # ioctl above isn't honoured. UDP is connectionless; keep
                    # serving the next request.
                    self._maybe_log(
                        f"[gaze_udp] recvfrom WSAECONNRESET ({e}); continuing"
                    )
                    continue
                except OSError as e:
                    # Real socket failure (e.g. socket closed via stop_event).
                    # Only break if we're shutting down anyway.
                    if self.stop_event.is_set():
                        break
                    self.log(f"recvfrom OSError ({e}); continuing")
                    continue

                if not data:
                    continue

                # Decode + handle
                req = None
                resp: Dict[str, Any]
                cmd = "?"
                qid = None

                try:
                    req = json.loads(data.decode("utf-8", errors="replace"))
                    cmd = str(req.get("cmd", "?")).lower()
                    qid = req.get("query_id", None)

                    self._maybe_log(
                        f"[gaze_udp] RX cmd={cmd} query_id={qid} bytes={len(data)} from={addr[0]}:{addr[1]}"
                    )

                    resp = self.handle(req, addr)

                except Exception as e:
                    resp = {"ok": False, "error": str(e), "cmd": cmd}
                    if qid is not None:
                        resp["query_id"] = qid

                # Reply
                try:
                    out = json.dumps(resp, separators=(",", ":")).encode("utf-8")
                    if len(out) > self.max_resp_bytes:
                        out = json.dumps({"ok": False, "error": "response_too_large"}).encode("utf-8")

                    self._sock.sendto(out, addr)

                    # Log TX after send
                    qid2 = resp.get("query_id", qid)
                    cmd2 = resp.get("cmd", cmd)
                    ok2 = resp.get("ok", None)
                    self._maybe_log(
                        f"[gaze_udp] TX cmd={cmd2} query_id={qid2} ok={ok2} bytes={len(out)} to={addr[0]}:{addr[1]}"
                    )

                except Exception:
                    pass

        finally:
            try:
                if self._sock:
                    self._sock.close()
            except Exception:
                pass
            try:
                if self._push_sock is not None:
                    self._push_sock.close()
            except Exception:
                pass
            self.log("stopped")

    def handle(self, req: Dict[str, Any], addr: Tuple[str, int]) -> Dict[str, Any]:
        """
        Handle an incoming JSON UDP request and return a JSON response.

        Supported `cmd` values:
          - `"ping"` / `"status"`: snapshot without objects (cheap telemetry)
          - `"snapshot"`: snapshot; optional `include_objects` flag
          - `"recenter"`: recenter head/gaze offsets using the most recent measurements
          - `"set_cv"`: enable/disable YOLO/object detection (saves compute)
          - `"stop"`: stop the gaze system (sets `stop_event`)

        Snapshot responses are directly compatible with what
        `ExperimentDriver_Online_GazeTracking.py` expects to read:
        `gaze_px`, `gaze_hit`, and (when enabled) `objects`/tracks.
        """
        cmd = str(req.get("cmd", "")).lower()

        if cmd in ("ping", "status"):
            snap = self.system.get_snapshot(include_objects=False, include_frame=False)
            snap["cmd"] = "status"
            snap["frame_bgr"] = None
            return snap

        if cmd == "snapshot":
            include_objects = bool(req.get("include_objects", True))
            snap = self.system.get_snapshot(include_objects=include_objects, include_frame=False)
            snap["cmd"] = "snapshot"
            snap["frame_bgr"] = None
            return snap

        if cmd == "recenter":
            ok = bool(self.system.recenter())
            return {"ok": ok, "cmd": "recenter"}

        if cmd == "set_cv":
            enabled = bool(req.get("enabled", True))
            self.system.set_cv_enabled(enabled)
            return {"ok": True, "cmd": "set_cv", "enabled": enabled}

        if cmd == "stop":
            if not self.allow_shutdown:
                return {"ok": False, "error": "shutdown not allowed", "cmd": "stop"}
            # Refuse remote stops by default — protects an unattended GPU
            # host from being taken offline by an accidental panel click,
            # which would otherwise force a physical restart.
            host = addr[0] if addr else ""
            is_local = isinstance(host, str) and (host == "127.0.0.1" or host.startswith("127."))
            if not is_local and not self.allow_remote_stop:
                self.log(f"refusing remote stop from {host}")
                return {
                    "ok": False, "cmd": "stop",
                    "error": "remote_stop_disabled",
                    "hint": "stop the service locally with Ctrl-C, "
                            "or restart with --allow-remote-stop to allow this",
                }
            self.stop_event.set()
            return {"ok": True, "cmd": "stop"}

        if cmd == "subscribe":
            return self._handle_subscribe(req, addr)

        if cmd == "unsubscribe":
            return self._handle_unsubscribe(req)

        return {"ok": False, "error": f"unknown cmd '{cmd}'"}

    # ── subscribe-mode JSON push (Render_Layer_Refactor.md §3) ────────────

    # Internal tick rate caps the per-subscriber push rate. The gaze
    # detector itself runs at det_hz (default 3 Hz), so 20 Hz is plenty
    # to ride on top of cached state without re-running the model.
    _TICK_HZ: float = 20.0
    _SUBSCRIBER_TTL_S: float = 30.0

    def _handle_subscribe(self, req: Dict[str, Any], addr: Tuple[str, int]) -> Dict[str, Any]:
        try:
            hz = float(req.get("hz", self._TICK_HZ))
        except (TypeError, ValueError):
            return {"ok": False, "error": "bad_hz", "cmd": "subscribe"}
        hz = max(0.5, min(hz, self._TICK_HZ))
        ttl_s = float(req.get("ttl_s", self._SUBSCRIBER_TTL_S))
        now = time.monotonic()
        with self._subscribers_lock:
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
        return {"ok": True, "cmd": "subscribe", "stream": "results",
                "subscriber_id": sid, "hz": hz}

    def _handle_unsubscribe(self, req: Dict[str, Any]) -> Dict[str, Any]:
        sid = req.get("subscriber_id")
        if not sid:
            return {"ok": False, "error": "missing_subscriber_id", "cmd": "unsubscribe"}
        with self._subscribers_lock:
            removed = self._subscribers.pop(str(sid), None)
        return {"ok": True, "cmd": "unsubscribe", "removed": bool(removed)}

    def _results_tick_loop(self) -> None:
        period = 1.0 / max(self._TICK_HZ, 1e-6)
        while not self.stop_event.is_set():
            t0 = time.monotonic()
            try:
                self._tick_send(t0)
            except Exception as e:
                if not self.quiet:
                    print(f"[gaze_udp] tick error: {e}")
            slept = time.monotonic() - t0
            remaining = period - slept
            if remaining > 0:
                # Honour stop_event quickly so service shutdown isn't blocked
                # by the tick period.
                self.stop_event.wait(timeout=remaining)

    def _tick_send(self, now: float) -> None:
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
        if not due or self._push_sock is None:
            return
        try:
            payload = json.dumps(self._build_gaze_results_payload(),
                                 separators=(",", ":")).encode("utf-8")
        except Exception:
            return
        if len(payload) > 60 * 1024:
            return
        for _sid, addr in due:
            try:
                self._push_sock.sendto(payload, addr)
            except OSError:
                pass

    def _build_gaze_results_payload(self) -> Dict[str, Any]:
        """Assemble a `gaze_results` payload from a fresh non-frame snapshot."""
        snap = self.system.get_snapshot(include_objects=True, include_frame=False)
        gaze_px = snap.get("gaze_px") or (None, None)
        objects = snap.get("objects") or []
        tracks = []
        for d in objects:
            if not isinstance(d, dict):
                continue
            xyxy = d.get("xyxy") or d.get("box_xyxy") or []
            tracks.append({
                "id": int(d.get("track_id", -1)),
                "bbox": [float(v) for v in xyxy] if xyxy else None,
                "label": str(d.get("name", "?")),
                "score": float(d.get("conf", 0.0)),
                "age": int(d.get("age", 0)) if "age" in d else None,
                "lost": int(d.get("lost", 0)) if "lost" in d else None,
            })
        gaze_hit = snap.get("gaze_hit")
        current_hit = None
        if isinstance(gaze_hit, dict):
            current_hit = {
                "track_id": int(gaze_hit.get("track_id", -1)),
                "name": str(gaze_hit.get("name", "?")),
                "conf": float(gaze_hit.get("conf", 0.0)),
            }
        return {
            "type": "gaze_results",
            "ts_send_ns": int(time.time_ns()),
            "wall_t": float(snap.get("wall_t", time.time())),
            "worn": bool(snap.get("worn", False)),
            "gaze_px": [float(gaze_px[0]), float(gaze_px[1])]
                if (gaze_px[0] is not None and gaze_px[1] is not None) else None,
            "tracks": tracks,
            "current_hit": current_hit,
            "governor": {
                "cv_enabled": bool(snap.get("gov_enabled", True)),
                "reason": str(snap.get("gov_reason", "")),
                "cd_left_s": float(snap.get("gov_cd_left", 0.0)),
            },
            "rates": {
                "loop_hz": float(snap.get("loop_hz", float("nan"))),
                "video_hz": float(snap.get("video_hz", float("nan"))),
                "det_hz": float(snap.get("det_hz", float("nan"))),
            },
        }

# -------------------------
# Helpers
# -------------------------
def _hit_text_from_snap(snap: Dict[str, Any]) -> str:
    hit = snap.get("gaze_hit", None)
    if hit is None:
        return "none"
    tid = int(hit.get("track_id", -1))
    nm = str(hit.get("name", "?"))
    cf = float(hit.get("conf", 0.0))
    return f"{nm}#{tid} ({cf:.2f})" if tid >= 0 else f"{nm} ({cf:.2f})"


def _publish_ui(ui: GazeUI, sys_cfg: GazeConfig, snap: Dict[str, Any]) -> None:
    hit = snap.get("gaze_hit", None)
    inputs = RenderInputs(
        frame_bgr=snap.get("frame_bgr", None),

        loop_hz=snap.get("loop_hz", float("nan")),
        video_hz=snap.get("video_hz", float("nan")),
        vid_stale_s=snap.get("vid_stale_s", float("nan")),
        det_hz=snap.get("det_hz", float("nan")),
        det_age_s=snap.get("det_age_s", float("nan")),
        infer_ms=snap.get("infer_ms", float("nan")),

        imu_angvel=snap.get("imu_angvel", None),
        imu_gate_enabled=bool(sys_cfg.enable_imu_gate),

        yolo_enabled=bool(snap.get("gov_enabled", True)),
        yolo_reason=str(snap.get("gov_reason", "healthy")),
        yolo_cd_left=float(snap.get("gov_cd_left", 0.0)),

        head_yaw_deg=snap.get("head_yaw_deg", float("nan")),
        head_pitch_deg=snap.get("head_pitch_deg", float("nan")),
        gaze_yaw_deg=snap.get("gaze_yaw_deg", float("nan")),
        gaze_pitch_deg=snap.get("gaze_pitch_deg", float("nan")),

        worn=bool(snap.get("worn", False)),
        depth_valid=bool(snap.get("depth_valid", False)),
        depth_cm=snap.get("depth_cm", float("nan")),
        miss_mm=snap.get("miss_mm", float("nan")),
        ipd_mm=snap.get("ipd_mm", float("nan")),

        gaze_px=snap.get("gaze_px", None),
        dets=snap.get("objects", None),

        gaze_hit=hit,
        gaze_hit_text=_hit_text_from_snap(snap),
    )
    ui.publish(inputs)


def build_sys_cfg() -> GazeConfig:
    # Your current defaults; keep here so CLI changes are minimal.
    return GazeConfig(
        enable_prints=True,
        enable_display=False,   # UI handled by GazeUI
        show_video=True,

        target_loop_hz=20.0,

        model_name="yolo26n.pt",
        det_hz=3.0,
        det_conf=0.50,
        det_iou=0.20,
        det_max_det=15,
        #det_classes=[0, 63, 64, 67, 39, 41],
        det_classes = None,
        detect_resize_width=None,

        # Governor
        gov_vid_stale_disable_s=0.20,
        gov_loop_min_hz=14.0,
        gov_slow_infer_s=0.30,
        gov_cooldown_s=0.35,
        gov_reenable_stable_s=0.50,

        # IMU gate
        enable_imu_gate=True,
        gov_imu_angvel_disable=20.0,
        imu_fresh_s=0.25,
    )


def build_ui_cfg(display_hz: float) -> UIConfig:
    return UIConfig(
        win_name="Neon (modular) - YOLO + Depth + Head/Gaze Angles",
        window_resizable=True,
        start_fullscreen=False,
        window_w=1280,
        window_h=720,
        display_hz=float(display_hz),
        enable_display=True,
        show_video=True,
    )


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["runner", "service"], default="runner")
    p.add_argument("--display", type=int, choices=[0, 1], default=1, help="show UI window")
    p.add_argument("--prints", type=int, choices=[0, 1], default=1, help="enable console prints in GazeSystem")
    p.add_argument("--loop_hz", type=float, default=None, help="override target_loop_hz")
    p.add_argument("--det_hz", type=float, default=None, help="override det_hz")
    p.add_argument("--model", type=str, default=None, help="override model_name")
    p.add_argument("--udp_log", type=int, choices=[0, 1], default=0,
                   help="log per-UDP request/response (RX/TX)")
    p.add_argument("--udp_log_hz", type=float, default=50.0,
                   help="max log rate (events/sec) for udp_log")
    # Service IPC
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=5588)
    p.add_argument("--ipc_verbose", action="store_true")
    p.add_argument("--allow-remote-stop", action="store_true",
                   dest="allow_remote_stop",
                   help="Honour cmd=stop from non-loopback addresses (off by default)")
    p.add_argument("--neon-device-host", type=str, default="",
                   dest="neon_device_host",
                   help="Companion app IP for direct connection; empty = mDNS discovery")
    # Frame-source toggle for the GPU-host migration plan; mirrors the
    # vlm_service.py flag. Default `local` keeps today's behaviour
    # (gaze_system opens Neon directly).
    p.add_argument("--frame-source", choices=["local", "remote"], default="local",
                   dest="frame_source",
                   help="local=open Neon directly; remote=consume Utils/frame_relay envelopes")
    p.add_argument("--remote-frame-host", type=str, default="",
                   dest="remote_frame_host",
                   help="Host of the frame_relay server (required when --frame-source=remote)")
    p.add_argument("--remote-frame-port", type=int, default=5591,
                   dest="remote_frame_port",
                   help="Port of the frame_relay server (default 5591)")
    return p.parse_args()


def main():
    args = parse_args()
    stop_event = threading.Event()

    sys_cfg = build_sys_cfg()
    sys_cfg.enable_prints = bool(args.prints)
    sys_cfg.neon_host = str(args.neon_device_host)
    sys_cfg.frame_source = str(args.frame_source)
    sys_cfg.remote_frame_host = str(args.remote_frame_host)
    sys_cfg.remote_frame_port = int(args.remote_frame_port)
    if sys_cfg.frame_source == "remote" and not sys_cfg.remote_frame_host:
        raise SystemExit("--frame-source=remote requires --remote-frame-host")

    if args.loop_hz is not None:
        sys_cfg.target_loop_hz = float(args.loop_hz)
    if args.det_hz is not None:
        sys_cfg.det_hz = float(args.det_hz)
    if args.model is not None:
        sys_cfg.model_name = str(args.model)

    system = GazeSystem(sys_cfg)
    system.start()

    ui = None
    if bool(args.display):
        ui_cfg = build_ui_cfg(display_hz=sys_cfg.target_loop_hz)
        ui = GazeUI(ui_cfg)
        ui.start()
        print("Connected. Keys: ESC quit | c recenter")

    udp = None
    if args.mode == "service":
        udp = GazeUDPServer(
            host=args.host,
            port=args.port,
            system=system,
            stop_event=stop_event,
            quiet=not bool(args.ipc_verbose),
            allow_shutdown=True,
            allow_remote_stop=bool(getattr(args, "allow_remote_stop", False)),
            udp_log=bool(args.udp_log),
            udp_log_hz=float(args.udp_log_hz),
        )
        udp.start()
        if not args.ipc_verbose:
            print(f"[service] UDP on {args.host}:{args.port} | display={args.display}")
    loop_period = 1.0 / max(float(sys_cfg.target_loop_hz), 1e-6)

    # Keep Windows awake while the service runs (no-op on POSIX). Mirrors
    # the inhibit in vlm_service.py's main(); see Utils/sleep_inhibit.py.
    try:
        from Utils.sleep_inhibit import inhibit as _sleep_inhibit, release as _sleep_release
        _sleep_inhibit()
    except Exception:
        _sleep_release = lambda: None  # noqa: E731

    try:
        while not stop_event.is_set():
            try:
                t0 = time.time()

                # In runner mode, UI controls stop condition; in service mode, we run until stop_event
                if args.mode == "runner" and ui is not None and ui.should_stop():
                    break

                # Key handling (only if UI is present)
                if ui is not None:
                    k = ui.get_last_key()
                    if k == 27:  # ESC
                        break
                    if k in (ord("c"), ord("C")):
                        system.recenter()

                # Runner needs frames; service typically doesn't.
                include_frame = bool(ui is not None)

                snap = system.get_snapshot(include_objects=True, include_frame=include_frame)
                if not snap.get("ok", False):
                    time.sleep(0.002)
                    continue

                if ui is not None:
                    _publish_ui(ui, sys_cfg, snap)
            except Exception as e:
                # Per-iteration error must not kill the service. Log and
                # continue so the UDP request-reply / push channels stay
                # alive even if a snapshot read fails transiently.
                print(f"[gaze_runner] main loop iteration error: {e}", flush=True)
                time.sleep(0.05)
                continue

            # Pace
            dt = time.time() - t0
            sleep_s = loop_period - dt
            if sleep_s > 0:
                time.sleep(sleep_s)

    finally:
        print("Stopping...")
        stop_event.set()
        try:
            _sleep_release()
        except Exception:
            pass
        try:
            if ui is not None:
                ui.stop()
        except Exception:
            pass
        try:
            system.stop()
        except Exception:
            pass
        print("Done.")


if __name__ == "__main__":
    main()
