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
import threading
import time
from typing import Any, Dict, Optional

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
        self.max_req_bytes = int(max_req_bytes)
        self.max_resp_bytes = int(max_resp_bytes)
        self.udp_log = bool(udp_log)
        self.udp_log_hz = float(udp_log_hz)
        self._sock: Optional[socket.socket] = None

        self._last_log_t = 0.0

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
        self.log(f"listening on udp://{self.host}:{self.port}")

        try:
            while not self.stop_event.is_set():
                try:
                    data, addr = self._sock.recvfrom(self.max_req_bytes)
                except socket.timeout:
                    continue
                except OSError:
                    break

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

                    resp = self.handle(req)

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
            self.log("stopped")

    def handle(self, req: Dict[str, Any]) -> Dict[str, Any]:
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
                return {"ok": False, "error": "shutdown not allowed"}
            self.stop_event.set()
            return {"ok": True, "cmd": "stop"}

        return {"ok": False, "error": f"unknown cmd '{cmd}'"}

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
    return p.parse_args()


def main():
    args = parse_args()
    stop_event = threading.Event()

    sys_cfg = build_sys_cfg()
    sys_cfg.enable_prints = bool(args.prints)

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
            udp_log=bool(args.udp_log),
            udp_log_hz=float(args.udp_log_hz),
        )
        udp.start()
        if not args.ipc_verbose:
            print(f"[service] UDP on {args.host}:{args.port} | display={args.display}")
    loop_period = 1.0 / max(float(sys_cfg.target_loop_hz), 1e-6)

    try:
        while not stop_event.is_set():
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

            # Pace
            dt = time.time() - t0
            sleep_s = loop_period - dt
            if sleep_s > 0:
                time.sleep(sleep_s)

    finally:
        print("Stopping...")
        stop_event.set()
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
