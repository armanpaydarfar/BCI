# Utils/gaze/gaze_ui.py
"""
gaze_ui.py

Pure UI / rendering utilities for the Neon gaze + YOLO + IMU pipeline.

Design goals:
- NO device I/O here (no Pupil device, no YOLO, no threads besides the UI loop).
- Given an image (or None) + overlay info, it renders a HUD and publishes frames.
- Optional OpenCV window loop (owning imshow/waitKey), so main loop never blocks.
- Can be run on the same computer in "big brother" mode (2nd window/monitor),
  but still no networking.

Expected usage:
    from Utils.gaze.gaze_ui import GazeUI, UIConfig, RenderInputs

    ui = GazeUI(UIConfig(...))
    ui.start()

    while running:
        frame = ...  # BGR uint8 or None
        inputs = RenderInputs(
            frame_bgr=frame,
            # ... fill gaze, dets, angles, etc ...
        )
        ui.publish(inputs)

    ui.stop()

Notes:
- This module intentionally mirrors your prior script's on-screen look/feel.
- It does not know about your tracker/governor classes—just consumes their outputs.

"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import cv2


# -------------------------
# Data contracts
# -------------------------

Det = Dict[str, Any]  # expects keys: xyxy, name, conf, track_id (opt), age (opt)


@dataclass
class UIConfig:
    # Window
    win_name: str = "Neon Gaze UI (ESC to quit)"
    window_resizable: bool = True
    start_fullscreen: bool = False
    window_w: int = 1280
    window_h: int = 720

    # Display pacing (set <=0 to uncap)
    display_hz: float = 20.0

    # Display behavior
    enable_display: bool = True
    show_video: bool = True  # if False, draws on black canvas
    canvas_w: int = 1280
    canvas_h: int = 720

    # Optional one-time screen fit (tkinter, done by caller if desired)
    fit_to_screen: bool = False
    max_disp_w: Optional[int] = None
    max_disp_h: Optional[int] = None

    # Display-only resize
    enable_display_resize: bool = False
    display_resize_width: Optional[int] = 960

    # Overlays toggles
    draw_gaze: bool = True
    draw_depth_bar: bool = True
    draw_angle_hud: bool = True
    draw_detections: bool = True
    draw_gaze_target_panel: bool = True

    # HUD layout (mirrors your script)
    hud_radius: int = 70
    hud_margin: int = 20
    hud_gap_y: int = 45

    depth_bar_w: int = 360
    depth_bar_h: int = 22
    depth_bar_x: int = 20
    depth_bar_y: int = 255  # you used DEPTH_BAR_Y + 100 in main

    depth_bar_min_cm: float = 10.0
    depth_bar_max_cm: float = 70.0

    gaze_panel_x: int = 20
    gaze_panel_y: int = 190


@dataclass
class RenderInputs:
    # Base image
    frame_bgr: Optional[np.ndarray] = None  # BGR uint8
    # If frame_bgr is None or show_video=False, UI draws on black canvas

    # Rates / status strings
    loop_hz: float = np.nan
    video_hz: float = np.nan
    vid_stale_s: float = np.nan
    det_hz: float = np.nan
    det_age_s: float = np.nan
    infer_ms: float = np.nan

    # IMU / governor status
    imu_angvel: Optional[float] = None
    imu_gate_enabled: bool = True

    yolo_enabled: bool = True
    yolo_reason: str = "healthy"
    yolo_cd_left: float = 0.0

    # Angles
    head_yaw_deg: float = np.nan
    head_pitch_deg: float = np.nan
    gaze_yaw_deg: float = np.nan
    gaze_pitch_deg: float = np.nan

    # Depth
    worn: bool = False
    depth_valid: bool = False
    depth_cm: float = np.nan
    miss_mm: float = np.nan
    ipd_mm: float = np.nan

    # Gaze pixel (already smoothed by caller)
    gaze_px: Optional[Tuple[float, float]] = None  # (x,y) in pixel coords

    # Detections (already tracked if you want)
    dets: Optional[List[Det]] = None

    # Current gaze hit info (already computed by caller)
    gaze_hit: Optional[Det] = None  # det dict
    gaze_hit_text: str = "none"     # e.g. "laptop#1 (0.94)"
    gaze_hit_box_highlight: bool = True


# -------------------------
# Small helpers
# -------------------------

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _safe_int(x: float) -> int:
    return int(round(float(x)))


# -------------------------
# Drawing primitives (match your style)
# -------------------------

def draw_detections(bgr: np.ndarray, dets: List[Det]) -> None:
    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        name = d.get("name", "obj")
        conf = float(d.get("conf", 0.0))
        tid = int(d.get("track_id", -1))
        age = float(d.get("age", 0.0))

        p1 = (_safe_int(x1), _safe_int(y1))
        p2 = (_safe_int(x2), _safe_int(y2))
        cv2.rectangle(bgr, p1, p2, (0, 255, 0), 2)

        label = f"{name}#{tid} {conf:.2f} age={age:.2f}s" if tid >= 0 else f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text = max(p1[1] - 8, th + 8)

        cv2.rectangle(bgr, (p1[0], y_text - th - 6), (p1[0] + tw + 6, y_text + 4), (0, 255, 0), -1)
        cv2.putText(bgr, label, (p1[0] + 3, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


def draw_gaze_target_panel(bgr: np.ndarray, text: str, *, x: int, y: int) -> None:
    panel_text = f"GAZE ON: {text}"
    (tw, th), _ = cv2.getTextSize(panel_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    pad = 10
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + tw + 2 * pad), int(y + th + 2 * pad)

    H, W = bgr.shape[:2]
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return

    roi = bgr[y1:y2, x1:x2]
    cv2.addWeighted(roi, 0.45, np.zeros_like(roi), 0.55, 0.0, roi)
    cv2.putText(bgr, panel_text, (x1 + pad, y1 + th + pad - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def draw_depth_bar(
    img: np.ndarray,
    *,
    d_cm: float,
    valid: bool,
    miss_mm: float,
    ipd_mm: float,
    x: int,
    y: int,
    w: int,
    h: int,
    dmin_cm: float,
    dmax_cm: float,
) -> None:
    cv2.putText(img, "Depth (vergence)", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    cv2.rectangle(img, (x + 2, y + 2), (x + w - 2, y + h - 2), (25, 25, 25), -1)

    if valid and np.isfinite(d_cm):
        frac = (float(d_cm) - float(dmin_cm)) / (float(dmax_cm) - float(dmin_cm) + 1e-9)
        frac = _clamp01(frac)
        fill_w = int((w - 4) * frac)
        cv2.rectangle(img, (x + 2, y + 2), (x + 2 + fill_w, y + h - 2), (0, 180, 0), -1)
        depth_txt = f"{d_cm:.1f} cm"
    else:
        depth_txt = "N/A"

    miss_txt = f"{miss_mm:.1f}mm" if np.isfinite(miss_mm) else "N/A"
    ipd_txt = f"{ipd_mm:.1f}mm" if np.isfinite(ipd_mm) else "N/A"
    cv2.putText(img, f"valid={valid} | miss={miss_txt} | IPD={ipd_txt} | range={dmin_cm:.0f}-{dmax_cm:.0f}cm",
                (x, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2, lineType=cv2.LINE_AA)

    cv2.putText(img, depth_txt, (x + w + 16, y + h - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, lineType=cv2.LINE_AA)


def draw_dual_angle_circle(
    img: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    label: str,
    head_angle_deg: float,
    gaze_angle_deg: float,
) -> None:
    cx, cy = int(center[0]), int(center[1])

    cv2.circle(img, (cx, cy), radius, (255, 255, 255), 2)
    cv2.line(img, (cx - radius, cy), (cx + radius, cy), (120, 120, 120), 1)
    cv2.line(img, (cx, cy - radius), (cx, cy + radius), (120, 120, 120), 1)

    cv2.putText(img, label, (cx - radius, cy - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    def draw_ray(angle_deg: float, color: Tuple[int, int, int], tag: str):
        if not np.isfinite(angle_deg):
            return
        theta = np.radians(float(angle_deg))
        dx = radius * np.sin(theta)
        dy = -radius * np.cos(theta)
        x2 = int(cx + dx)
        y2 = int(cy + dy)
        cv2.line(img, (cx, cy), (x2, y2), color, 3, lineType=cv2.LINE_AA)
        cv2.circle(img, (x2, y2), 6, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(img, tag, (x2 + 8, y2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, lineType=cv2.LINE_AA)

    # Match your script: Head=blue, Gaze=red (BGR)
    draw_ray(head_angle_deg, (255, 0, 0), "H")
    draw_ray(gaze_angle_deg, (0, 0, 255), "G")


# -------------------------
# GazeUI class (UI thread owns imshow/waitKey)
# -------------------------

class GazeUI:
    """
    Owns the OpenCV window and event loop. Main thread calls publish(inputs).

    - UI thread is the only place that calls cv2.imshow / cv2.waitKey.
    - Latest-frame semantics: if rendering is slow, it drops frames and shows newest.
    """

    def __init__(self, cfg: UIConfig):
        self.cfg = cfg
        self._stop = threading.Event()

        self._lock = threading.Lock()
        self._front: Optional[np.ndarray] = None
        self._back: Optional[np.ndarray] = None
        self._has_frame: bool = False
        self._last_key: int = -1

        self._thread: Optional[threading.Thread] = None
        self._started = False

        # Reduce OpenCV oversubscription jitter (safe even if already set elsewhere)
        try:
            cv2.setNumThreads(1)
            cv2.setUseOptimized(True)
            try:
                cv2.ocl.setUseOpenCL(True)
            except Exception:
                pass
        except Exception:
            pass

    # ---- Public API ----

    def start(self) -> None:
        if not self.cfg.enable_display:
            self._started = True
            return
        if self._started:
            return
        self._thread = threading.Thread(target=self._ui_loop, daemon=True)
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.cfg.enable_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def should_stop(self) -> bool:
        return self._stop.is_set()

    def get_last_key(self) -> int:
        with self._lock:
            k = self._last_key
            self._last_key = -1
        return k

    def publish(self, inputs: RenderInputs) -> None:
        """
        Compose overlays and publish a frame for UI thread to display.
        This is safe to call even if enable_display=False (no-op).
        """
        if not self.cfg.enable_display:
            return

        out = self._compose(inputs)
        if out is None:
            return

        with self._lock:
            if (self._back is None) or (self._back.shape != out.shape):
                self._back = np.empty_like(out)
            np.copyto(self._back, out)
            self._front = self._back
            self._has_frame = True

    # ---- Internals ----

    def _compose(self, inputs: RenderInputs) -> Optional[np.ndarray]:
        cfg = self.cfg

        # Base image
        if (not cfg.show_video) or (inputs.frame_bgr is None):
            h, w = int(cfg.canvas_h), int(cfg.canvas_w)
            bgr = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            bgr = inputs.frame_bgr
            if bgr.dtype != np.uint8:
                bgr = np.clip(bgr, 0, 255).astype(np.uint8, copy=False)

        # Copy to avoid mutating caller buffer
        out = bgr.copy()

        H, W = out.shape[:2]

        dets = inputs.dets or []

        # Detections
        if cfg.draw_detections and dets:
            draw_detections(out, dets)

        # Gaze pixel
        if cfg.draw_gaze and inputs.worn and inputs.gaze_px is not None:
            gx, gy = inputs.gaze_px
            if 0 <= gx < W and 0 <= gy < H:
                cv2.circle(out, (_safe_int(gx), _safe_int(gy)), 10, (0, 0, 255), 2)

        # HUD text (mirrors your previous script)
        loop_hz = inputs.loop_hz
        video_hz = inputs.video_hz if np.isfinite(inputs.video_hz) else 0.0
        vid_stale = inputs.vid_stale_s if np.isfinite(inputs.vid_stale_s) else 0.0

        det_hz = inputs.det_hz if np.isfinite(inputs.det_hz) else 0.0
        det_age = inputs.det_age_s if np.isfinite(inputs.det_age_s) else 0.0
        infer_ms = inputs.infer_ms if np.isfinite(inputs.infer_ms) else 0.0

        imu_w = inputs.imu_angvel
        imu_w_txt = f"{imu_w:.2f}" if (imu_w is not None and np.isfinite(imu_w)) else "0.00"

        yolo_txt = f"YOLO={'ON' if inputs.yolo_enabled else 'OFF'} | {inputs.yolo_reason} | cd={inputs.yolo_cd_left:.2f}s"
        imu_txt = f"imu|w|={imu_w_txt} rad/s (gate={'on' if inputs.imu_gate_enabled else 'off'})"
        ang_txt = (
            f"HEAD(yaw,pitch)=({inputs.head_yaw_deg:+.0f},{inputs.head_pitch_deg:+.0f})  "
            f"GAZE(yaw,pitch)=({inputs.gaze_yaw_deg:+.0f},{inputs.gaze_pitch_deg:+.0f})"
        )

        cv2.putText(out, f"loop={loop_hz:.1f}Hz video={video_hz:.1f}Hz | vid_stale={vid_stale:.3f}s",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, f"det={det_hz:.1f}Hz age={det_age:.2f}s infer={infer_ms:.0f}ms | tracks={len(dets)}",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, imu_txt, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, yolo_txt, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, ang_txt, (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # Depth bar
        if cfg.draw_depth_bar:
            draw_depth_bar(
                out,
                d_cm=inputs.depth_cm,
                valid=bool(inputs.worn and inputs.depth_valid),
                miss_mm=inputs.miss_mm,
                ipd_mm=inputs.ipd_mm,
                x=int(cfg.depth_bar_x),
                y=int(cfg.depth_bar_y),
                w=int(cfg.depth_bar_w),
                h=int(cfg.depth_bar_h),
                dmin_cm=float(cfg.depth_bar_min_cm),
                dmax_cm=float(cfg.depth_bar_max_cm),
            )

        # Angle HUD circles (right side)
        if cfg.draw_angle_hud:
            cx = W - cfg.hud_margin - cfg.hud_radius
            cy1 = cfg.hud_margin + cfg.hud_radius
            cy2 = cy1 + 2 * cfg.hud_radius + cfg.hud_gap_y

            draw_dual_angle_circle(out, (cx, cy1), cfg.hud_radius, "Yaw / Azimuth", inputs.head_yaw_deg, inputs.gaze_yaw_deg)
            draw_dual_angle_circle(out, (cx, cy2), cfg.hud_radius, "Pitch / Elev", inputs.head_pitch_deg, inputs.gaze_pitch_deg)

            cv2.putText(out, "c=recenter | ESC=quit",
                        (cx - cfg.hud_radius, cy2 + cfg.hud_radius + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2, cv2.LINE_AA)

        # Gaze target panel + highlight
        if cfg.draw_gaze_target_panel:
            draw_gaze_target_panel(out, inputs.gaze_hit_text or "none", x=cfg.gaze_panel_x, y=cfg.gaze_panel_y)

            if inputs.gaze_hit is not None and inputs.gaze_hit_box_highlight:
                try:
                    x1, y1, x2, y2 = inputs.gaze_hit["xyxy"]
                    cv2.rectangle(out, (_safe_int(x1), _safe_int(y1)),
                                  (_safe_int(x2), _safe_int(y2)), (0, 0, 255), 3)
                except Exception:
                    pass

        # Display-only resize
        if cfg.enable_display_resize and cfg.display_resize_width is not None:
            src_h, src_w = out.shape[:2]
            new_w = int(cfg.display_resize_width)
            if src_w != new_w and new_w > 0:
                new_h = int(round(src_h * (new_w / float(src_w))))
                out = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Fit-to-screen (caller can set max_disp_w/h once; avoids tkinter here)
        if cfg.fit_to_screen and (cfg.max_disp_w is not None) and (cfg.max_disp_h is not None):
            hh, ww = out.shape[:2]
            scale = min(cfg.max_disp_w / float(ww), cfg.max_disp_h / float(hh))
            if scale < 1.0:
                out = cv2.resize(
                    out,
                    (max(1, int(ww * scale)), max(1, int(hh * scale))),
                    interpolation=cv2.INTER_AREA,
                )

        return out

    def _ui_loop(self) -> None:
        cfg = self.cfg

        # Guard: if no display, just idle
        if not cfg.enable_display:
            return

        # Workaround for some OpenCV backend quirks
        try:
            cv2.imshow("cv/av bug", np.zeros(1, dtype=np.uint8))
            cv2.destroyAllWindows()
        except Exception:
            pass

        flags = cv2.WINDOW_NORMAL if cfg.window_resizable else cv2.WINDOW_AUTOSIZE
        cv2.namedWindow(cfg.win_name, flags)

        if cfg.start_fullscreen:
            cv2.setWindowProperty(cfg.win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif cfg.window_resizable:
            cv2.resizeWindow(cfg.win_name, int(cfg.window_w), int(cfg.window_h))

        period = None if (cfg.display_hz is None or cfg.display_hz <= 0) else (1.0 / float(cfg.display_hz))
        next_t = time.perf_counter()

        while not self._stop.is_set():
            with self._lock:
                frame = self._front if self._has_frame else None

            if frame is not None:
                cv2.imshow(cfg.win_name, frame)

            k = cv2.waitKey(1) & 0xFF
            if k != 255:
                with self._lock:
                    self._last_key = k
                if k == 27:  # ESC
                    self._stop.set()
                    break

            if period is not None:
                next_t += period
                now = time.perf_counter()
                sleep_s = next_t - now
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_t = now
