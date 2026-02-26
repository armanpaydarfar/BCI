#!/usr/bin/env python3
"""
Neon_Stream_GazeUtil_MERGED_DepthBar.py

Realtime Pupil Labs Neon utility (threaded acquisition) with:

- Optional scene video display + gaze overlay
- Vergence depth estimate + visual depth bar + numeric display
- Head direction HUD (IMU quaternion -> head forward vector -> yaw/pitch)
- Gaze direction HUD (optical_axis_{L,R} -> cyclopean ray -> scene->imu->world -> yaw/pitch)
- Recenter key: 'c' sets current head yaw/pitch = 0 (applies to head + gaze)
- Per-stream rate estimates (Hz) (EWMA dt per thread) + UI Hz

NEW controls / integration features:
- ENABLE_PRINTS: suppress ALL print statements
- ENABLE_DISPLAY: suppress ALL windows (headless utility mode)
- FIT_TO_SCREEN: scale the displayed frame to your monitor resolution

Keys (only when ENABLE_DISPLAY=True):
  ESC   quit
  c     recenter head yaw/pitch (sets current to 0)
  g     toggle drawing gaze dot on video
  h     toggle HUD circles (head+gaze)
"""

import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

from pupil_labs.realtime_api.simple import discover_one_device


# -----------------------
# CONFIG
# -----------------------

# Suppression toggles
ENABLE_PRINTS = True       # If False: no console prints at all
ENABLE_DISPLAY = True      # If False: no cv2 windows / headless
FIT_TO_SCREEN = True       # If True: scale displayed frame to your monitor size

# Drawing toggles (only relevant if ENABLE_DISPLAY=True)
SHOW_VIDEO = True          # If False: draws overlays on black background window (still a window)
DRAW_GAZE_DOT = True
DRAW_HUD = True

UI_FPS = 30.0              # UI refresh rate (Hz) - independent of stream rates
PRINT_HZ = 10.0            # Console print rate (Hz) if ENABLE_PRINTS=True

# Depth estimator settings (vergence)
MISS_MAX_MM = 30.0
DENOM_MIN = 1e-4
SMOOTH_ALPHA = 0.15

# Depth bar widget
DEPTH_BAR_MIN_CM = 10.0
DEPTH_BAR_MAX_CM = 70.0
DEPTH_BAR_W = 360
DEPTH_BAR_H = 22
DEPTH_BAR_X = 20
DEPTH_BAR_Y = 155
DEPTH_BAR_SHOW_NUM_BIG = True

# HUD settings
HUD_RADIUS = 70
HUD_MARGIN = 20
HUD_GAP_Y = 45
ANGLE_ALPHA = 0.25         # EWMA smoothing for displayed angles

# "Nose forward" axis in the IMU's LOCAL frame
HEAD_FORWARD_LOCAL = np.array([0.0, 1.0, 0.0], dtype=float)

# Scene->IMU rotation about X (deg)
IMU_SCENE_ROT_DEG = -90.0 - 12.0

# Assume optical axes are in SCENE coords. If gaze seems "over-rotated", set False.
APPLY_SCENE_TO_IMU_FOR_GAZE = True

# Stream dt EWMA smoothing for rate estimation
RATE_ALPHA = 0.15

# Fit-to-screen constraints (fraction of screen)
SCREEN_FIT_MAX_W_FRAC = 0.95
SCREEN_FIT_MAX_H_FRAC = 0.90


# -----------------------
# Small utilities
# -----------------------
def log(*args, **kwargs):
    if ENABLE_PRINTS:
        print(*args, **kwargs)


def ewma_update(prev: float, x: float, alpha: float) -> float:
    if not np.isfinite(prev):
        return float(x)
    return float(alpha * x + (1.0 - alpha) * prev)


def hz_from_dt(dt: float) -> float:
    return (1.0 / dt) if (np.isfinite(dt) and dt > 1e-6) else np.nan


def safe_unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def wrap_deg_180(a: float) -> float:
    if not np.isfinite(a):
        return a
    return (a + 180.0) % 360.0 - 180.0


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# -----------------------
# Depth from vergence
# -----------------------
def vergence_depth_from_eyestate(
    L_mm,
    u,
    R_mm,
    v,
    *,
    miss_max_mm: float = 30.0,
    denom_min: float = 1e-4,
):
    """
    Estimate vergence depth from two eye rays.

    Returns:
        valid (bool),
        d_m (float),
        miss_m (float),
        P_mm (np.ndarray shape (3,))
    """
    L = np.asarray(L_mm, dtype=float)
    R = np.asarray(R_mm, dtype=float)

    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return False, np.nan, np.nan, np.full(3, np.nan)

    u = u / nu
    v = v / nv

    # Closest approach between two rays:
    # pL(t)=L+t*u, pR(s)=R+s*v
    w0 = L - R
    b = float(np.dot(u, v))
    d = float(np.dot(u, w0))
    e = float(np.dot(v, w0))

    denom = 1.0 - b * b
    if denom < denom_min:
        return False, np.nan, np.nan, np.full(3, np.nan)

    t = (b * e - d) / denom
    s = (e - b * d) / denom

    P_L = L + t * u
    P_R = R + s * v
    P = 0.5 * (P_L + P_R)

    miss_mm = float(np.linalg.norm(P_L - P_R))
    miss_m = miss_mm / 1000.0

    # cyclopean eye origin (midpoint between eye centers)
    C = 0.5 * (L + R)
    d_mm = float(np.linalg.norm(P - C))
    d_m = d_mm / 1000.0

    valid = (miss_mm <= miss_max_mm) and np.isfinite(d_m) and (d_m > 0.05)
    return bool(valid), (d_m if valid else np.nan), miss_m, (P if valid else np.full(3, np.nan))


class EWMASmoother:
    def __init__(self, alpha: float, init_value: float = np.nan):
        self.alpha = float(alpha)
        self.y = float(init_value)

    def update(self, x: float) -> float:
        x = float(x)
        if not np.isfinite(x):
            return self.y
        if not np.isfinite(self.y):
            self.y = x
        else:
            self.y = self.alpha * x + (1.0 - self.alpha) * self.y
        return self.y


# -----------------------
# Quaternion utilities (scalar_first=True => (w,x,y,z))
# -----------------------
def quat_to_rotmat_wxyz(w: float, x: float, y: float, z: float) -> np.ndarray:
    q = np.array([w, x, y, z], dtype=float)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3, dtype=float)
    q /= n
    w, x, y, z = q

    R = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )
    return R


# -----------------------
# Notebook transforms
# -----------------------
def scene_to_imu_matrix() -> np.ndarray:
    a = np.deg2rad(IMU_SCENE_ROT_DEG)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(a), -np.sin(a)],
            [0.0, np.sin(a),  np.cos(a)],
        ],
        dtype=float,
    )


def cartesian_to_spherical_world(world_vec: np.ndarray) -> Tuple[float, float]:
    """
    Returns: (elevation_deg, azimuth_deg)
    (matches your existing convention)
    """
    v = safe_unit(world_vec)
    x, y, z = float(v[0]), float(v[1]), float(v[2])

    r = max(np.sqrt(x * x + y * y + z * z), 1e-9)

    elevation = -(np.arccos(z / r) - np.pi / 2.0)
    azimuth = np.arctan2(y, x) - np.pi / 2.0

    if azimuth < -np.pi:
        azimuth += 2 * np.pi
    if azimuth > np.pi:
        azimuth -= 2 * np.pi

    return float(np.rad2deg(elevation)), float(np.rad2deg(azimuth))


# -----------------------
# HUD drawing: head + gaze rays on circle
# -----------------------
def draw_dual_angle_circle(
    img,
    center,
    radius,
    label,
    head_angle_deg,
    gaze_angle_deg,
    *,
    head_color=(255, 0, 0),  # blue (BGR)
    gaze_color=(0, 0, 255),  # red  (BGR)
):
    cx, cy = int(center[0]), int(center[1])

    cv2.circle(img, (cx, cy), radius, (255, 255, 255), 2)
    cv2.line(img, (cx - radius, cy), (cx + radius, cy), (120, 120, 120), 1)
    cv2.line(img, (cx, cy - radius), (cx, cy + radius), (120, 120, 120), 1)

    cv2.putText(
        img,
        label,
        (cx - radius, cy - radius - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )

    def draw_ray(angle_deg, color, tag):
        if not np.isfinite(angle_deg):
            return
        theta = np.radians(float(angle_deg))
        dx = radius * np.sin(theta)
        dy = -radius * np.cos(theta)
        x2 = int(cx + dx)
        y2 = int(cy + dy)
        cv2.line(img, (cx, cy), (x2, y2), color, 3, lineType=cv2.LINE_AA)
        cv2.circle(img, (x2, y2), 6, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            img,
            tag,
            (x2 + 8, y2 + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            lineType=cv2.LINE_AA,
        )

    draw_ray(head_angle_deg, head_color, "H")
    draw_ray(gaze_angle_deg, gaze_color, "G")


# -----------------------
# Depth bar widget
# -----------------------
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
    show_big_number: bool = True,
):
    # Title
    cv2.putText(
        img,
        "Depth (vergence)",
        (x, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )

    # Outer frame + dark background
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    cv2.rectangle(img, (x + 2, y + 2), (x + w - 2, y + h - 2), (25, 25, 25), -1)

    if valid and np.isfinite(d_cm):
        frac = (float(d_cm) - float(dmin_cm)) / (float(dmax_cm) - float(dmin_cm) + 1e-9)
        frac = clamp01(frac)
        fill_w = int((w - 4) * frac)
        cv2.rectangle(img, (x + 2, y + 2), (x + 2 + fill_w, y + h - 2), (0, 180, 0), -1)
        depth_txt = f"{d_cm:.1f} cm"
    else:
        depth_txt = "N/A"

    miss_txt = f"{miss_mm:.1f}mm" if np.isfinite(miss_mm) else "N/A"
    ipd_txt = f"{ipd_mm:.1f}mm" if np.isfinite(ipd_mm) else "N/A"

    cv2.putText(
        img,
        f"valid={valid} | miss={miss_txt} | IPD={ipd_txt} | range={dmin_cm:.0f}-{dmax_cm:.0f}cm",
        (x, y + h + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        2,
        lineType=cv2.LINE_AA,
    )

    if show_big_number:
        cv2.putText(
            img,
            depth_txt,
            (x + w + 16, y + h - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
    else:
        cv2.putText(
            img,
            depth_txt,
            (x + 6, y + h - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )


# -----------------------
# Gaze ray from optical axes
# -----------------------
def gaze_ray_from_optical_axes(gaze) -> np.ndarray:
    uL = np.array([gaze.optical_axis_left_x, gaze.optical_axis_left_y, gaze.optical_axis_left_z], dtype=float)
    uR = np.array([gaze.optical_axis_right_x, gaze.optical_axis_right_y, gaze.optical_axis_right_z], dtype=float)
    return safe_unit(safe_unit(uL) + safe_unit(uR))


# -----------------------
# Screen fitting
# -----------------------
def get_screen_size() -> Tuple[int, int]:
    """
    Best-effort screen resolution query for Linux/Windows/macOS without extra deps.
    """
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = int(root.winfo_screenwidth())
        h = int(root.winfo_screenheight())
        root.destroy()
        return w, h
    except Exception:
        # Fallback: don't crash; just return something sane
        return 1920, 1080


def resize_to_fit(bgr: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w <= 0 or h <= 0:
        return bgr
    scale = min(max_w / float(w), max_h / float(h))
    if scale >= 1.0:
        return bgr
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


# -----------------------
# Threaded acquisition
# -----------------------
@dataclass
class Latest:
    gaze: Optional[object] = None
    imu: Optional[object] = None
    frame_bgr: Optional[np.ndarray] = None
    frame_hw: Optional[Tuple[int, int]] = None

    t_gaze: float = 0.0
    t_imu: float = 0.0
    t_frame: float = 0.0

    gaze_dt_ewma: float = np.nan
    imu_dt_ewma: float = np.nan
    frame_dt_ewma: float = np.nan


def start_reader_threads(device):
    latest = Latest()
    lock = threading.Lock()
    stop_evt = threading.Event()

    def gaze_thread():
        t_prev = None
        while not stop_evt.is_set():
            try:
                g = device.receive_gaze_datum()  # blocking
                t_now = time.time()
                dt = (t_now - t_prev) if (t_prev is not None) else np.nan
                t_prev = t_now
                with lock:
                    latest.gaze = g
                    latest.t_gaze = t_now
                    if np.isfinite(dt) and dt > 1e-6:
                        latest.gaze_dt_ewma = ewma_update(latest.gaze_dt_ewma, dt, RATE_ALPHA)
            except Exception:
                time.sleep(0.005)

    def imu_thread():
        t_prev = None
        while not stop_evt.is_set():
            try:
                m = device.receive_imu_datum()  # blocking
                t_now = time.time()
                dt = (t_now - t_prev) if (t_prev is not None) else np.nan
                t_prev = t_now
                with lock:
                    latest.imu = m
                    latest.t_imu = t_now
                    if np.isfinite(dt) and dt > 1e-6:
                        latest.imu_dt_ewma = ewma_update(latest.imu_dt_ewma, dt, RATE_ALPHA)
            except Exception:
                time.sleep(0.005)

    def video_thread():
        if not SHOW_VIDEO:
            return
        t_prev = None
        while not stop_evt.is_set():
            try:
                bgr, _dt = device.receive_scene_video_frame()  # blocking
                t_now = time.time()
                dt = (t_now - t_prev) if (t_prev is not None) else np.nan
                t_prev = t_now
                with lock:
                    latest.frame_bgr = bgr
                    latest.frame_hw = (bgr.shape[0], bgr.shape[1])
                    latest.t_frame = t_now
                    if np.isfinite(dt) and dt > 1e-6:
                        latest.frame_dt_ewma = ewma_update(latest.frame_dt_ewma, dt, RATE_ALPHA)
            except Exception:
                time.sleep(0.005)

    threads = [
        threading.Thread(target=gaze_thread, daemon=True),
        threading.Thread(target=imu_thread, daemon=True),
    ]
    if SHOW_VIDEO:
        threads.append(threading.Thread(target=video_thread, daemon=True))

    for th in threads:
        th.start()

    def snapshot():
        with lock:
            return (
                latest.gaze,
                latest.imu,
                latest.frame_bgr,
                latest.frame_hw,
                latest.t_gaze,
                latest.t_imu,
                latest.t_frame,
                latest.gaze_dt_ewma,
                latest.imu_dt_ewma,
                latest.frame_dt_ewma,
            )

    return stop_evt, snapshot


# -----------------------
# Main
# -----------------------
def main():
    global DRAW_GAZE_DOT, DRAW_HUD, SHOW_VIDEO

    if ENABLE_DISPLAY and SHOW_VIDEO:
        # Workaround for https://github.com/opencv/opencv/issues/21952
        cv2.imshow("cv/av bug", np.zeros(1))
        cv2.destroyAllWindows()

    log("Discovering Pupil Labs device...")
    device = discover_one_device(max_search_duration_seconds=10)
    if device is None:
        log("No device found.")
        raise SystemExit(-1)

    log("Connected.")
    if ENABLE_DISPLAY:
        log("Controls: ESC quit | c recenter | g toggle gaze dot | h toggle HUD")

    stop_evt, snapshot = start_reader_threads(device)

    depth_smoother = EWMASmoother(alpha=SMOOTH_ALPHA)

    head_yaw_smoother = EWMASmoother(alpha=ANGLE_ALPHA)
    head_pitch_smoother = EWMASmoother(alpha=ANGLE_ALPHA)
    gaze_yaw_smoother = EWMASmoother(alpha=ANGLE_ALPHA)
    gaze_pitch_smoother = EWMASmoother(alpha=ANGLE_ALPHA)

    yaw_offset = 0.0
    pitch_offset = 0.0
    have_recenter = False

    next_print_t = time.time()

    S2I = scene_to_imu_matrix()

    ui_dt = 1.0 / max(UI_FPS, 1e-6)
    t_next = time.time()

    ui_dt_ewma = np.nan
    ui_prev = None

    # Screen fit config
    screen_w, screen_h = get_screen_size()
    max_disp_w = int(screen_w * SCREEN_FIT_MAX_W_FRAC)
    max_disp_h = int(screen_h * SCREEN_FIT_MAX_H_FRAC)

    win_name = "Neon: head+gaze+depth (ESC quit)"
    if ENABLE_DISPLAY:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            now = time.time()
            if now < t_next:
                time.sleep(min(0.002, t_next - now))
                continue
            t_next += ui_dt

            # UI FPS estimate
            if ui_prev is not None:
                dt_ui = now - ui_prev
                if dt_ui > 1e-6:
                    ui_dt_ewma = ewma_update(ui_dt_ewma, dt_ui, 0.10)
            ui_prev = now
            ui_hz = hz_from_dt(ui_dt_ewma)

            (
                gaze,
                imu,
                frame_bgr,
                frame_hw,
                t_gaze,
                t_imu,
                t_frame,
                gaze_dt,
                imu_dt,
                vid_dt,
            ) = snapshot()

            gaze_hz = hz_from_dt(gaze_dt)
            imu_hz = hz_from_dt(imu_dt)
            vid_hz = hz_from_dt(vid_dt)

            # Base canvas
            if SHOW_VIDEO and frame_bgr is not None:
                bgr = frame_bgr.copy()
                h, w = frame_hw
            else:
                h, w = 720, 1280
                bgr = np.zeros((h, w, 3), dtype=np.uint8)

            if imu is None:
                if ENABLE_DISPLAY:
                    cv2.putText(bgr, "Waiting for IMU...", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                    cv2.putText(
                        bgr,
                        f"rates(Hz): gaze={gaze_hz:5.1f} imu={imu_hz:5.1f} vid={vid_hz:5.1f} ui={ui_hz:5.1f}",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, lineType=cv2.LINE_AA
                    )

                    disp = resize_to_fit(bgr, max_disp_w, max_disp_h) if FIT_TO_SCREEN else bgr
                    cv2.imshow(win_name, disp)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        break
                continue

            # IMU -> world
            q = imu.quaternion  # w,x,y,z
            R_imu_to_world = quat_to_rotmat_wxyz(float(q.w), float(q.x), float(q.y), float(q.z))

            # Head direction (yaw/pitch)
            head_world = safe_unit(R_imu_to_world @ HEAD_FORWARD_LOCAL)
            head_elev, head_azim = cartesian_to_spherical_world(head_world)
            head_yaw_meas = wrap_deg_180(head_azim)
            head_pitch_meas = float(np.clip(head_elev, -90.0, 90.0)) if np.isfinite(head_elev) else head_elev

            # Defaults
            gx = gy = np.nan
            worn = False
            pupil_left = pupil_right = np.nan

            d_cm = np.nan
            miss_mm = np.nan
            ipd_mm = np.nan
            valid_depth = False

            gaze_yaw_meas = np.nan
            gaze_pitch_meas = np.nan

            if gaze is not None:
                gx = float(getattr(gaze, "x", np.nan))
                gy = float(getattr(gaze, "y", np.nan))
                worn = bool(getattr(gaze, "worn", False))
                pupil_left = float(getattr(gaze, "pupil_diameter_left", np.nan))
                pupil_right = float(getattr(gaze, "pupil_diameter_right", np.nan))

                # Vergence depth + IPD
                try:
                    L_mm = np.array(
                        [gaze.eyeball_center_left_x, gaze.eyeball_center_left_y, gaze.eyeball_center_left_z],
                        dtype=float,
                    )
                    R_mm = np.array(
                        [gaze.eyeball_center_right_x, gaze.eyeball_center_right_y, gaze.eyeball_center_right_z],
                        dtype=float,
                    )
                    uL = np.array(
                        [gaze.optical_axis_left_x, gaze.optical_axis_left_y, gaze.optical_axis_left_z],
                        dtype=float,
                    )
                    uR = np.array(
                        [gaze.optical_axis_right_x, gaze.optical_axis_right_y, gaze.optical_axis_right_z],
                        dtype=float,
                    )

                    ipd_mm = float(np.linalg.norm(L_mm - R_mm))

                    valid_depth, d_m, miss_m, _ = vergence_depth_from_eyestate(
                        L_mm, uL, R_mm, uR,
                        miss_max_mm=MISS_MAX_MM,
                        denom_min=DENOM_MIN,
                    )

                    # Smooth only when worn+valid; otherwise hold last
                    if worn and valid_depth:
                        d_s = depth_smoother.update(d_m)
                    else:
                        d_s = depth_smoother.y

                    d_cm = (d_s * 100.0) if np.isfinite(d_s) else np.nan
                    miss_mm = (miss_m * 1000.0) if np.isfinite(miss_m) else np.nan
                except Exception:
                    pass

                # Gaze ray direction -> yaw/pitch
                try:
                    gaze_ray = gaze_ray_from_optical_axes(gaze)  # SCENE coords (assumed)
                    if APPLY_SCENE_TO_IMU_FOR_GAZE:
                        gaze_imu = safe_unit(S2I @ gaze_ray)
                    else:
                        gaze_imu = gaze_ray

                    gaze_world = safe_unit(R_imu_to_world @ gaze_imu)

                    g_elev, g_azim = cartesian_to_spherical_world(gaze_world)
                    gaze_yaw_meas = wrap_deg_180(g_azim)
                    gaze_pitch_meas = float(np.clip(g_elev, -90.0, 90.0)) if np.isfinite(g_elev) else g_elev
                except Exception:
                    pass

            # Apply recenter offsets (to display only)
            if have_recenter:
                head_yaw_disp = wrap_deg_180(head_yaw_meas - yaw_offset)
                head_pitch_disp = float(np.clip(head_pitch_meas - pitch_offset, -90.0, 90.0)) if np.isfinite(head_pitch_meas) else head_pitch_meas

                gaze_yaw_disp = wrap_deg_180(gaze_yaw_meas - yaw_offset) if np.isfinite(gaze_yaw_meas) else gaze_yaw_meas
                gaze_pitch_disp = float(np.clip(gaze_pitch_meas - pitch_offset, -90.0, 90.0)) if np.isfinite(gaze_pitch_meas) else gaze_pitch_meas
            else:
                head_yaw_disp, head_pitch_disp = head_yaw_meas, head_pitch_meas
                gaze_yaw_disp, gaze_pitch_disp = gaze_yaw_meas, gaze_pitch_meas

            # Smooth displayed angles
            head_yaw_s = head_yaw_smoother.update(head_yaw_disp)
            head_pitch_s = head_pitch_smoother.update(head_pitch_disp)
            gaze_yaw_s = gaze_yaw_smoother.update(gaze_yaw_disp)
            gaze_pitch_s = gaze_pitch_smoother.update(gaze_pitch_disp)

            # ---------- Draw overlays (only if ENABLE_DISPLAY) ----------
            if ENABLE_DISPLAY:
                # Gaze dot
                if SHOW_VIDEO and DRAW_GAZE_DOT and np.isfinite(gx) and np.isfinite(gy):
                    if 0 <= gx < w and 0 <= gy < h:
                        cv2.circle(bgr, (int(gx), int(gy)), 15, (0, 0, 255), 2)

                # Text HUD
                cv2.putText(
                    bgr,
                    f"worn={worn} | pupil(L,R)=({pupil_left:.2f},{pupil_right:.2f})",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )

                hy = f"{head_yaw_s:+.0f}" if np.isfinite(head_yaw_s) else "N/A"
                hp = f"{head_pitch_s:+.0f}" if np.isfinite(head_pitch_s) else "N/A"
                gyaw = f"{gaze_yaw_s:+.0f}" if np.isfinite(gaze_yaw_s) else "N/A"
                gp = f"{gaze_pitch_s:+.0f}" if np.isfinite(gaze_pitch_s) else "N/A"

                cv2.putText(
                    bgr,
                    f"HEAD(yaw,pitch)=({hy},{hp}) deg | GAZE(yaw,pitch)=({gyaw},{gp}) deg",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )

                cv2.putText(
                    bgr,
                    f"rates(Hz): gaze={gaze_hz:5.1f} imu={imu_hz:5.1f} vid={vid_hz:5.1f} ui={ui_hz:5.1f}",
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (200, 200, 200),
                    2,
                    lineType=cv2.LINE_AA,
                )

                # Depth bar
                draw_depth_bar(
                    bgr,
                    d_cm=d_cm,
                    valid=bool(worn and valid_depth),
                    miss_mm=miss_mm,
                    ipd_mm=ipd_mm,
                    x=DEPTH_BAR_X,
                    y=DEPTH_BAR_Y,
                    w=DEPTH_BAR_W,
                    h=DEPTH_BAR_H,
                    dmin_cm=DEPTH_BAR_MIN_CM,
                    dmax_cm=DEPTH_BAR_MAX_CM,
                    show_big_number=DEPTH_BAR_SHOW_NUM_BIG,
                )

                # HUD circles
                if DRAW_HUD:
                    cx = w - HUD_MARGIN - HUD_RADIUS
                    cy1 = HUD_MARGIN + HUD_RADIUS
                    cy2 = cy1 + 2 * HUD_RADIUS + HUD_GAP_Y

                    draw_dual_angle_circle(bgr, (cx, cy1), HUD_RADIUS, "Yaw / Azimuth", head_yaw_s, gaze_yaw_s)
                    draw_dual_angle_circle(bgr, (cx, cy2), HUD_RADIUS, "Pitch / Elev", head_pitch_s, gaze_pitch_s)

                    cv2.putText(
                        bgr,
                        "c=recenter | g=gaze dot | h=HUD | ESC=quit",
                        (cx - HUD_RADIUS, cy2 + HUD_RADIUS + 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (200, 200, 200),
                        2,
                        lineType=cv2.LINE_AA,
                    )

            # ---------- Console print (suppressed if ENABLE_PRINTS=False) ----------
            if ENABLE_PRINTS:
                now2 = time.time()
                if now2 >= next_print_t:
                    next_print_t = now2 + (1.0 / max(PRINT_HZ, 1e-6))
                    age_g = (now2 - t_gaze) if t_gaze > 0 else np.nan
                    age_i = (now2 - t_imu) if t_imu > 0 else np.nan
                    age_v = (now2 - t_frame) if t_frame > 0 else np.nan
                    depth_str = f"{d_cm:.1f}cm" if np.isfinite(d_cm) else "N/A"
                    miss_str = f"{miss_mm:.1f}mm" if np.isfinite(miss_mm) else "N/A"
                    ipd_str = f"{ipd_mm:.1f}mm" if np.isfinite(ipd_mm) else "N/A"

                    print(
                        f"ages(s): gaze={age_g:.3f} imu={age_i:.3f} vid={age_v:.3f} | "
                        f"Hz: gaze={gaze_hz:5.1f} imu={imu_hz:5.1f} vid={vid_hz:5.1f} ui={ui_hz:5.1f} | "
                        f"depth={depth_str} miss={miss_str} IPD={ipd_str} valid={bool(worn and valid_depth)} | "
                        f"HEAD(yaw,pitch)=({head_yaw_s:+.0f},{head_pitch_s:+.0f}) "
                        f"GAZE(yaw,pitch)=({gaze_yaw_s:+.0f},{gaze_pitch_s:+.0f})"
                    )

            # ---------- Display & key handling ----------
            if ENABLE_DISPLAY:
                disp = resize_to_fit(bgr, max_disp_w, max_disp_h) if FIT_TO_SCREEN else bgr
                cv2.imshow(win_name, disp)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                elif key in (ord("c"), ord("C")):
                    yaw_offset = head_yaw_meas
                    pitch_offset = head_pitch_meas
                    have_recenter = True
                    head_yaw_smoother.y = 0.0
                    head_pitch_smoother.y = 0.0
                    gaze_yaw_smoother.y = 0.0
                    gaze_pitch_smoother.y = 0.0
                    log(f"[recenter] yaw_offset={yaw_offset:+.2f} deg, pitch_offset={pitch_offset:+.2f} deg")
                elif key in (ord("g"), ord("G")):
                    DRAW_GAZE_DOT = not DRAW_GAZE_DOT
                    log(f"[toggle] DRAW_GAZE_DOT={DRAW_GAZE_DOT}")
                elif key in (ord("h"), ord("H")):
                    DRAW_HUD = not DRAW_HUD
                    log(f"[toggle] DRAW_HUD={DRAW_HUD}")
            else:
                # Headless mode: no window, no keys. Quit via Ctrl-C.
                pass

    except KeyboardInterrupt:
        log("\nKeyboardInterrupt received. Stopping...")

    finally:
        log("Stopping threads/streams and closing windows...")
        stop_evt.set()
        try:
            device.close()
        except Exception:
            pass
        if ENABLE_DISPLAY:
            cv2.destroyAllWindows()
        log("Done.")


if __name__ == "__main__":
    main()
