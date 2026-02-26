#!/usr/bin/env python3
"""
Neon + YOLO Governor (dynamic enable/disable) + IMU gating + SORT tracking
PLUS:
- Vergence depth estimate (depth cm + miss mm + IPD mm) + optional depth bar
- Head yaw/pitch from IMU quaternion
- Gaze yaw/pitch from optical axes (scene->imu correction) then IMU->world
- Recenter key 'c' confirmed: applies to head + gaze (yaw/pitch offsets)
- Console logs show what objects are tracked (top-K summary)

Notes:
- UI thread owns imshow/waitKey. Main loop never calls imshow/waitKey.
- Latest-frame semantics everywhere.
- Governor sheds load to recover.
"""

# -----------------------
# Hard clamp thread pools (must be before torch/ultralytics import)
# -----------------------
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")

import time
import threading
from collections import deque

import numpy as np
import cv2
from pupil_labs.realtime_api.simple import discover_one_device


# -----------------------
# CONFIG
# -----------------------

# Suppression / utility mode
ENABLE_PRINTS = True          # suppress all prints if False
ENABLE_DISPLAY = True         # headless if False (no windows; still runs)
FIT_TO_SCREEN = False         # display-only fit (uses tkinter; one-time)

# Video overlays
SHOW_VIDEO = True             # if False but ENABLE_DISPLAY=True -> black canvas HUD
DRAW_GAZE = True              # draw gaze dot (pixel)
DRAW_DEPTH_BAR = True
DRAW_ANGLE_HUD = True         # draw yaw/pitch circles

# Window
WINDOW_RESIZABLE = True
START_FULLSCREEN = False
WINDOW_W = 1280
WINDOW_H = 720
WIN = "Neon + YOLO Governor + Depth + Head/Gaze Angles (ESC to quit)"

# Loop pacing
TARGET_LOOP_HZ = 20.0
DISPLAY_HZ = 20.0  # set <=0 to uncap

# Detector base settings (governor decides if it runs)
DET_HZ = 3.0
DET_CONF = 0.50
DET_IOU = 0.20
DET_MAX_DET = 15
MODEL_NAME = "yolo26n.pt"
DETECT_RESIZE_WIDTH = None  # None disables

# Only detect what you care about (reduces dense-scene blowups)
DET_CLASSES = [0, 63, 64, 67, 39, 41]  # person, laptop, mouse, cell phone, bottle, wine glass, cup

# Display-only resize (if not using FIT_TO_SCREEN)
ENABLE_DISPLAY_RESIZE = False
DISPLAY_RESIZE_WIDTH = 960

# Console prints
PRINT_HZ = 5
TOPK_LOG = 3                 # show top-K tracked objects in console

# Gaze-on-object label hold
GAZE_HIT_HOLD_SEC = 1.0
GAZE_RADIUS_PX = 25
NEAREST_FALLBACK_PX = 60
GAZE_RECENCY_SEC = 0.35      # ignore stale tracks for gaze-hit
GAZE_SMOOTH_ALPHA = 0.35

# -----------------------
# YOLO Governor
# -----------------------
GOV_VID_STALE_DISABLE_S = 0.20
GOV_LOOP_MIN_HZ = 14.0
GOV_SLOW_INFER_S = 0.30

GOV_COOLDOWN_S = 0.35
GOV_REENABLE_STABLE_S = 0.50

# -----------------------
# IMU gating (ENABLED)
# -----------------------
ENABLE_IMU_GATE = True
GOV_IMU_ANGVEL_DISABLE = 30.0  # rad/s
IMU_FRESH_S = 0.25

# -----------------------
# Depth + angles (from your merged util)
# -----------------------
# Vergence depth settings
MISS_MAX_MM = 30.0
DENOM_MIN = 1e-4
DEPTH_SMOOTH_ALPHA = 0.15
DEPTH_BAR_MIN_CM = 10.0
DEPTH_BAR_MAX_CM = 70.0

# "Nose forward" axis in IMU LOCAL frame
HEAD_FORWARD_LOCAL = np.array([0.0, 1.0, 0.0], dtype=float)

# Scene->IMU rotation about X (deg)
IMU_SCENE_ROT_DEG = -90.0 - 12.0
APPLY_SCENE_TO_IMU_FOR_GAZE = True

# Angle smoothing (display)
ANGLE_ALPHA = 0.25

# HUD circle layout
HUD_RADIUS = 70
HUD_MARGIN = 20
HUD_GAP_Y = 45

# Depth bar layout
DEPTH_BAR_W = 360
DEPTH_BAR_H = 22
DEPTH_BAR_X = 20
DEPTH_BAR_Y = 155

# -----------------------
# Tracking (SORT-style)
# -----------------------
USE_TRACKER = True
TRACK_MATCH_IOU = 0.22
TRACK_MAX_AGE_SEC = 1.25
TRACK_MIN_HITS = 2
TRACK_EMA_ALPHA = 0.28

KALMAN_PROCESS_NOISE_POS = 7.0
KALMAN_PROCESS_NOISE_VEL = 12.0
KALMAN_MEAS_NOISE_POS = 16.0
KALMAN_MEAS_NOISE_SIZE = 32.0

TRACK_NEARBY_REUSE_PX = 80
TRACK_NEARBY_SIZE_RATIO = 0.75
TRACK_NEARBY_ALLOW_CLASS_MISMATCH = False


# -----------------------
# Shared state between threads
# -----------------------
stop_event = threading.Event()

# Latest video frame
video_lock = threading.Lock()
video_buf_bgr = None
video_buf_shape = None
latest_video_wall_t = None

# Latest gaze datum
gaze_lock = threading.Lock()
latest_gaze = None
latest_gaze_wall_t = None

# Detector input snapshot
frame_lock = threading.Lock()
latest_frame_bgr = None
latest_frame_time = None  # unix time from gaze (best-effort)

# Detector output
det_lock = threading.Lock()
latest_dets = []
latest_det_time = None
latest_det_fps = 0.0
latest_infer_s = np.nan  # last inference duration

# UI buffers
ui_lock = threading.Lock()
ui_front_bgr = None
ui_back_bgr = None
ui_has_frame = False
ui_last_key = -1

# Display buffer (main composes overlays)
display_buf_bgr = None
display_buf_shape = None

# IMU state
imu_lock = threading.Lock()
latest_imu_angvel = None     # float magnitude (rad/s)
latest_imu_wall_t = None
latest_imu = None            # full imu datum (for quaternion)

# Reduce OpenCV oversubscription jitter
cv2.setNumThreads(1)
cv2.setUseOptimized(True)
try:
    cv2.ocl.setUseOpenCL(True)
except Exception:
    pass


# -----------------------
# Logging helper
# -----------------------
def log(*args, **kwargs):
    if ENABLE_PRINTS:
        print(*args, **kwargs)


# -----------------------
# Utilities (boxes, timing)
# -----------------------
def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def _box_area(x1, y1, x2, y2):
    return float(max(1.0, (x2 - x1) * (y2 - y1)))

def _dist_point_to_box(px, py, x1, y1, x2, y2):
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    cx = _clamp(px, x1, x2)
    cy = _clamp(py, y1, y2)
    return float(np.hypot(px - cx, py - cy))

def _box_center_xy(xyxy):
    x1, y1, x2, y2 = map(float, xyxy)
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)

def _box_wh(xyxy):
    x1, y1, x2, y2 = map(float, xyxy)
    return max(1.0, x2 - x1), max(1.0, y2 - y1)

def _size_similarity_ratio(boxA, boxB):
    w1, h1 = _box_wh(boxA)
    w2, h2 = _box_wh(boxB)
    rw = min(w1, w2) / max(w1, w2)
    rh = min(h1, h2) / max(h1, h2)
    return float(rw * rh)

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return float(inter / (area_a + area_b - inter))


def hz_from_hist(hist):
    if len(hist) < 5:
        return np.nan
    m = float(np.mean(hist))
    return (1.0 / m) if m > 1e-9 else np.nan


# -----------------------
# Depth + angles utilities
# -----------------------
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

def vergence_depth_from_eyestate(L_mm, u, R_mm, v, *, miss_max_mm=30.0, denom_min=1e-4):
    L = np.asarray(L_mm, dtype=float)
    R = np.asarray(R_mm, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return False, np.nan, np.nan

    u = u / nu
    v = v / nv

    w0 = L - R
    b = float(np.dot(u, v))
    d = float(np.dot(u, w0))
    e = float(np.dot(v, w0))

    denom = 1.0 - b * b
    if denom < denom_min:
        return False, np.nan, np.nan

    t = (b * e - d) / denom
    s = (e - b * d) / denom

    P_L = L + t * u
    P_R = R + s * v
    miss_mm_val = float(np.linalg.norm(P_L - P_R))

    C = 0.5 * (L + R)
    P = 0.5 * (P_L + P_R)
    d_mm = float(np.linalg.norm(P - C))
    d_m = d_mm / 1000.0

    valid = (miss_mm_val <= miss_max_mm) and np.isfinite(d_m) and (d_m > 0.05)
    return bool(valid), (d_m if valid else np.nan), (miss_mm_val if valid else np.nan)

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

def cartesian_to_spherical_world(world_vec: np.ndarray):
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

def gaze_ray_from_optical_axes(gaze) -> np.ndarray:
    uL = np.array([gaze.optical_axis_left_x, gaze.optical_axis_left_y, gaze.optical_axis_left_z], dtype=float)
    uR = np.array([gaze.optical_axis_right_x, gaze.optical_axis_right_y, gaze.optical_axis_right_z], dtype=float)
    return safe_unit(safe_unit(uL) + safe_unit(uR))

def draw_depth_bar(img, *, d_cm, valid, miss_mm, ipd_mm,
                   x, y, w, h, dmin_cm, dmax_cm):
    cv2.putText(img, "Depth (vergence)", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)

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
    cv2.putText(img, f"valid={valid} | miss={miss_txt} | IPD={ipd_txt} | range={dmin_cm:.0f}-{dmax_cm:.0f}cm",
                (x, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2, lineType=cv2.LINE_AA)

    cv2.putText(img, depth_txt, (x + w + 16, y + h - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, lineType=cv2.LINE_AA)

def draw_dual_angle_circle(img, center, radius, label, head_angle_deg, gaze_angle_deg):
    cx, cy = int(center[0]), int(center[1])

    cv2.circle(img, (cx, cy), radius, (255, 255, 255), 2)
    cv2.line(img, (cx - radius, cy), (cx + radius, cy), (120, 120, 120), 1)
    cv2.line(img, (cx, cy - radius), (cx, cy + radius), (120, 120, 120), 1)

    cv2.putText(img, label, (cx - radius, cy - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)

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
        cv2.putText(img, tag, (x2 + 8, y2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, lineType=cv2.LINE_AA)

    draw_ray(head_angle_deg, (255, 0, 0), "H")   # blue
    draw_ray(gaze_angle_deg, (0, 0, 255), "G")   # red


# -----------------------
# IMU angular velocity extraction (same as your stable script)
# -----------------------
def _quat_to_np_wxyz(q):
    return np.array([float(q.w), float(q.x), float(q.y), float(q.z)], dtype=np.float64)

def _quat_normalize(q):
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n

def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw
    ], dtype=np.float64)

def _omega_from_quat_delta(q_prev, q_now, dt):
    if dt <= 1e-6:
        return np.nan
    q_prev = _quat_normalize(q_prev)
    q_now = _quat_normalize(q_now)
    q_delta = _quat_mul(_quat_conj(q_prev), q_now)
    q_delta = _quat_normalize(q_delta)
    w = float(np.clip(q_delta[0], -1.0, 1.0))
    angle = 2.0 * float(np.arccos(w))
    if angle > np.pi:
        angle = 2.0 * np.pi - angle
    return float(angle / dt)

def _gyro_mag_from_imu_datum(imu):
    candidates = [
        ("angular_velocity_x", "angular_velocity_y", "angular_velocity_z"),
        ("gyro_x", "gyro_y", "gyro_z"),
        ("gyroscope_x", "gyroscope_y", "gyroscope_z"),
        ("gyr_x", "gyr_y", "gyr_z"),
    ]
    for fx, fy, fz in candidates:
        if all(hasattr(imu, f) for f in (fx, fy, fz)):
            try:
                gx = float(getattr(imu, fx))
                gy = float(getattr(imu, fy))
                gz = float(getattr(imu, fz))
                mag = float(np.sqrt(gx*gx + gy*gy + gz*gz))
                if np.isfinite(mag):
                    return mag, f"{fx},{fy},{fz}"
            except Exception:
                pass

    for subname in ("gyro", "angular_velocity", "gyroscope"):
        if hasattr(imu, subname):
            sub = getattr(imu, subname)
            for fx, fy, fz in candidates:
                if all(hasattr(sub, f) for f in (fx, fy, fz)):
                    try:
                        gx = float(getattr(sub, fx))
                        gy = float(getattr(sub, fy))
                        gz = float(getattr(sub, fz))
                        mag = float(np.sqrt(gx*gx + gy*gy + gz*gz))
                        if np.isfinite(mag):
                            return mag, f"{subname}.{fx},{subname}.{fy},{subname}.{fz}"
                    except Exception:
                        pass

    return np.nan, "no_gyro_fields"


# -----------------------
# YOLO Governor state (unchanged)
# -----------------------
gov_lock = threading.Lock()
gov_enabled = True
gov_reason = "init"
gov_cooldown_until = 0.0
gov_stable_since = None

def gov_disable(reason: str, now: float):
    global gov_enabled, gov_reason, gov_cooldown_until, gov_stable_since
    with gov_lock:
        gov_enabled = False
        gov_reason = str(reason)
        gov_cooldown_until = max(gov_cooldown_until, now + float(GOV_COOLDOWN_S))
        gov_stable_since = None

def gov_health_update(now: float, *, vid_stale_s: float, loop_hz: float, imu_angvel: float | None):
    global gov_enabled, gov_reason, gov_cooldown_until, gov_stable_since

    unhealthy = False
    why = None

    if np.isfinite(vid_stale_s) and vid_stale_s > float(GOV_VID_STALE_DISABLE_S):
        unhealthy = True
        why = f"vid_stale>{GOV_VID_STALE_DISABLE_S:.2f}s"
    elif np.isfinite(loop_hz) and loop_hz > 0 and loop_hz < float(GOV_LOOP_MIN_HZ):
        unhealthy = True
        why = f"loop<{GOV_LOOP_MIN_HZ:.1f}Hz"
    elif ENABLE_IMU_GATE and (imu_angvel is not None) and (imu_angvel > float(GOV_IMU_ANGVEL_DISABLE)):
        unhealthy = True
        why = f"imu|w|>{GOV_IMU_ANGVEL_DISABLE:.2f}"

    if unhealthy:
        gov_disable(why, now)
        return

    with gov_lock:
        if (not gov_enabled) and (now < gov_cooldown_until):
            gov_stable_since = None
            return

        if not gov_enabled:
            if gov_stable_since is None:
                gov_stable_since = now
            if (now - gov_stable_since) >= float(GOV_REENABLE_STABLE_S):
                gov_enabled = True
                gov_reason = "healthy"
                gov_stable_since = None
            return

        gov_reason = "healthy"
        gov_stable_since = None

def gov_can_run(now: float) -> tuple[bool, str, float]:
    with gov_lock:
        enabled = bool(gov_enabled)
        reason = str(gov_reason)
        cd = float(max(0.0, gov_cooldown_until - now))
    return enabled, reason, cd


# -----------------------
# Drawing + gaze hit (unchanged + minor)
# -----------------------
def draw_detections(bgr: np.ndarray, dets):
    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        name = d.get("name", "obj")
        conf = float(d.get("conf", 0.0))
        tid = int(d.get("track_id", -1))
        age = float(d.get("age", 0.0))

        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(bgr, p1, p2, (0, 255, 0), 2)

        label = f"{name}#{tid} {conf:.2f} age={age:.2f}s" if tid >= 0 else f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text = max(p1[1] - 8, th + 8)

        cv2.rectangle(bgr, (p1[0], y_text - th - 6), (p1[0] + tw + 6, y_text + 4), (0, 255, 0), -1)
        cv2.putText(bgr, label, (p1[0] + 3, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

def draw_gaze_target_panel(bgr: np.ndarray, text: str, *, x=20, y=150):
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

def gaze_object_hit(gx, gy, dets, *, gaze_radius_px=GAZE_RADIUS_PX, nearest_fallback_px=NEAREST_FALLBACK_PX):
    if not dets:
        return None, None, None

    if GAZE_RECENCY_SEC is not None:
        ages = [float(d.get("age", 0.0)) for d in dets]
        if ages and min(ages) > float(GAZE_RECENCY_SEC):
            return None, None, None

    inside_hits = []
    near_hits = []
    nearest = None

    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        area = _box_area(x1, y1, x2, y2)
        conf = float(d.get("conf", 0.0))

        if (x1 <= gx <= x2) and (y1 <= gy <= y2):
            inside_hits.append((area, -conf, d))
            continue

        dist = _dist_point_to_box(gx, gy, x1, y1, x2, y2)
        if dist <= float(gaze_radius_px):
            near_hits.append((dist, area, -conf, d))
        else:
            cand = (dist, area, -conf, d)
            if nearest is None or cand[0] < nearest[0]:
                nearest = cand

    if inside_hits:
        inside_hits.sort(key=lambda t: (t[0], t[1]))
        return inside_hits[0][2], "inside", 0.0

    if near_hits:
        near_hits.sort(key=lambda t: (t[0], t[1], t[2]))
        dist, _, _, d = near_hits[0]
        return d, "near", float(dist)

    if nearest is not None and nearest[0] <= float(nearest_fallback_px):
        dist, _, _, d = nearest
        return d, "nearest", float(dist)

    return None, None, None


# -----------------------
# Threads: Video + Gaze + IMU (IMU now publishes full datum too)
# -----------------------
def video_thread_fn(device):
    global video_buf_bgr, video_buf_shape, latest_video_wall_t, latest_frame_bgr
    while not stop_event.is_set():
        try:
            frame, _dt = device.receive_scene_video_frame()  # blocking
            if frame is None:
                continue
            now = time.time()
            with video_lock:
                if (video_buf_bgr is None) or (video_buf_shape != frame.shape):
                    video_buf_bgr = np.empty_like(frame)
                    video_buf_shape = frame.shape
                np.copyto(video_buf_bgr, frame)
                latest_video_wall_t = now
            with frame_lock:
                latest_frame_bgr = video_buf_bgr
        except Exception:
            time.sleep(0.001)

def gaze_thread_fn(device):
    global latest_gaze, latest_gaze_wall_t
    while not stop_event.is_set():
        try:
            g = device.receive_gaze_datum()  # blocking
            with gaze_lock:
                latest_gaze = g
                latest_gaze_wall_t = time.time()
        except Exception:
            time.sleep(0.01)

def imu_thread_fn(device):
    """
    Publishes:
      - latest_imu: full datum (for quaternion)
      - latest_imu_angvel: magnitude (rad/s) for gating
    """
    global latest_imu_angvel, latest_imu_wall_t, latest_imu

    q_prev = None
    t_prev = None

    while not stop_event.is_set():
        try:
            imu = device.receive_imu_datum()  # blocking
            t_now = time.time()

            mag, _used = _gyro_mag_from_imu_datum(imu)

            if not np.isfinite(mag):
                if hasattr(imu, "quaternion"):
                    try:
                        q_now = _quat_to_np_wxyz(imu.quaternion)
                        if q_prev is not None and t_prev is not None:
                            mag = _omega_from_quat_delta(q_prev, q_now, t_now - t_prev)
                        q_prev = q_now
                        t_prev = t_now
                    except Exception:
                        mag = np.nan
                else:
                    mag = np.nan

            with imu_lock:
                latest_imu = imu
                latest_imu_angvel = float(mag) if np.isfinite(mag) else None
                latest_imu_wall_t = t_now

        except Exception:
            time.sleep(0.005)


# -----------------------
# Detector thread (YOLO + Governor) — same as stable version
# -----------------------
def detector_thread_fn():
    global latest_dets, latest_det_time, latest_det_fps, latest_infer_s

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "Missing dependency 'ultralytics'. Install with:\n  pip install ultralytics\n"
        ) from e

    try:
        import torch  # type: ignore
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    model = YOLO(MODEL_NAME)
    try:
        model.to("cpu")
    except Exception:
        pass

    period = 1.0 / max(float(DET_HZ), 1e-6)
    next_run = time.perf_counter()

    fps_hist = deque(maxlen=30)
    t_prev = time.time()

    local_bgr = None
    local_shape = None

    while not stop_event.is_set():
        now_pc = time.perf_counter()
        if now_pc < next_run:
            time.sleep(min(0.002, next_run - now_pc))
            continue

        now = time.time()
        enabled, _reason, _cd_left = gov_can_run(now)

        with frame_lock:
            src = latest_frame_bgr
            ftime = latest_frame_time

        if (not enabled) or (src is None):
            next_run += period
            now_pc2 = time.perf_counter()
            if next_run < now_pc2 - 2.0 * period:
                next_run = now_pc2
            time.sleep(0.010)
            continue

        if (local_bgr is None) or (local_shape != src.shape):
            local_bgr = np.empty_like(src)
            local_shape = src.shape
        np.copyto(local_bgr, src)

        scale = 1.0
        frame_det = local_bgr

        if DETECT_RESIZE_WIDTH is not None:
            h, w = frame_det.shape[:2]
            new_w = int(DETECT_RESIZE_WIDTH)
            if w != new_w:
                new_h = int(round(h * (new_w / float(w))))
                frame_det = cv2.resize(frame_det, (new_w, new_h), interpolation=cv2.INTER_AREA)
                scale = new_w / float(w)

        t0 = time.time()
        results = model.predict(
            source=frame_det,
            verbose=False,
            conf=float(DET_CONF),
            iou=float(DET_IOU),
            device="cpu",
            max_det=int(DET_MAX_DET),
            classes=DET_CLASSES if (DET_CLASSES is not None and len(DET_CLASSES) > 0) else None,
        )
        infer_s = time.time() - t0
        latest_infer_s = float(infer_s)

        if infer_s > float(GOV_SLOW_INFER_S):
            gov_disable(f"infer>{GOV_SLOW_INFER_S:.2f}s", time.time())

        dets = []
        if results:
            r0 = results[0]
            names = r0.names if hasattr(r0, "names") else getattr(model, "names", None)
            if hasattr(r0, "boxes") and r0.boxes is not None:
                boxes = r0.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)

                if scale != 1.0:
                    xyxy = xyxy / scale

                for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
                    if isinstance(names, dict):
                        name = names.get(int(k), str(int(k)))
                    elif isinstance(names, (list, tuple)):
                        name = names[int(k)]
                    else:
                        name = str(int(k))

                    dets.append(
                        {
                            "xyxy": (float(x1), float(y1), float(x2), float(y2)),
                            "cls": int(k),
                            "conf": float(c),
                            "name": str(name),
                            "frame_time": ftime,
                        }
                    )

        with det_lock:
            latest_dets = dets
            latest_det_time = time.time()

        t_now = time.time()
        dt = t_now - t_prev
        t_prev = t_now
        if dt > 1e-6:
            fps_hist.append(1.0 / dt)
            latest_det_fps = float(np.mean(fps_hist)) if fps_hist else 0.0

        next_run += period
        now_pc2 = time.perf_counter()
        if next_run < now_pc2 - 2.0 * period:
            next_run = now_pc2


# -----------------------
# Simple Kalman + SORT-style tracker (unchanged)
# -----------------------
class KalmanBox:
    """State x = [cx, cy, vx, vy, w, h], measurement z = [cx, cy, w, h]."""
    def __init__(self, cx, cy, w, h, *, t0, cls, name, conf):
        self.x = np.array([cx, cy, 0.0, 0.0, w, h], dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 100.0
        self.cls = int(cls)
        self.name = str(name)
        self.conf = float(conf)
        self.hits = 1
        self.confirmed = False
        self.t_last = float(t0)
        self.t_last_meas = float(t0)
        self._ema_w = float(w)
        self._ema_h = float(h)

    def predict(self, t_now):
        dt = float(max(0.0, t_now - self.t_last))
        self.t_last = float(t_now)

        F = np.eye(6, dtype=np.float64)
        F[0, 2] = dt
        F[1, 3] = dt

        q_pos = float(KALMAN_PROCESS_NOISE_POS)
        q_vel = float(KALMAN_PROCESS_NOISE_VEL)
        Q = np.diag([q_pos*q_pos, q_pos*q_pos, q_vel*q_vel, q_vel*q_vel, 20.0, 20.0]).astype(np.float64)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.x[4] = max(2.0, float(self.x[4]))
        self.x[5] = max(2.0, float(self.x[5]))

    def update(self, meas, *, t_meas, conf):
        z = np.array(meas, dtype=np.float64)

        H = np.zeros((4, 6), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 4] = 1.0
        H[3, 5] = 1.0

        r_pos = float(KALMAN_MEAS_NOISE_POS)
        r_size = float(KALMAN_MEAS_NOISE_SIZE)
        R = np.diag([r_pos*r_pos, r_pos*r_pos, r_size*r_size, r_size*r_size]).astype(np.float64)

        y = z - (H @ self.x)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ H) @ self.P

        alpha = float(TRACK_EMA_ALPHA)
        self._ema_w = (1 - alpha) * self._ema_w + alpha * float(z[2])
        self._ema_h = (1 - alpha) * self._ema_h + alpha * float(z[3])
        self.x[4] = max(2.0, self._ema_w)
        self.x[5] = max(2.0, self._ema_h)

        self.conf = float(conf)
        self.hits += 1
        self.t_last_meas = float(t_meas)
        if self.hits >= int(TRACK_MIN_HITS):
            self.confirmed = True

    def age_since_meas(self, t_now):
        return float(t_now - self.t_last_meas)

    def xyxy(self):
        cx, cy, _, _, w, h = self.x.tolist()
        return (float(cx - w / 2.0), float(cy - h / 2.0), float(cx + w / 2.0), float(cy + h / 2.0))

class SimpleSORTTracker:
    def __init__(self):
        self.tracks = {}
        self._next_id = 1

    def predict(self, t_now):
        for tr in self.tracks.values():
            tr.predict(t_now)

    def update_with_dets(self, dets, *, t_now):
        if not dets:
            self._prune(t_now)
            return

        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid].xyxy() for tid in track_ids]
        det_boxes = [d["xyxy"] for d in dets]

        used_tracks = set()
        used_dets = set()

        cands = []
        for j, d in enumerate(dets):
            for i, tid in enumerate(track_ids):
                tr = self.tracks[tid]
                if int(d["cls"]) != int(tr.cls):
                    continue
                score = iou_xyxy(track_boxes[i], det_boxes[j])
                if score > 0:
                    cands.append((score, i, j))
        cands.sort(key=lambda t: t[0], reverse=True)

        for score, i, j in cands:
            if score < float(TRACK_MATCH_IOU):
                break
            tid = track_ids[i]
            if tid in used_tracks or j in used_dets:
                continue
            used_tracks.add(tid)
            used_dets.add(j)

            d = dets[j]
            x1, y1, x2, y2 = d["xyxy"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w, h = max(2.0, x2 - x1), max(2.0, y2 - y1)
            self.tracks[tid].update([cx, cy, w, h], t_meas=t_now, conf=float(d["conf"]))
            self.tracks[tid].name = str(d["name"])

        unmatched_track_idx = {i for i, tid in enumerate(track_ids) if tid not in used_tracks}

        for j, d in enumerate(dets):
            if j in used_dets:
                continue

            det_box = d["xyxy"]
            det_cx, det_cy = _box_center_xy(det_box)

            best_i = None
            best_dist = 1e18
            for i in list(unmatched_track_idx):
                tid = track_ids[i]
                tr = self.tracks[tid]
                tr_box = tr.xyxy()

                if (not TRACK_NEARBY_ALLOW_CLASS_MISMATCH) and (int(d["cls"]) != int(tr.cls)):
                    continue

                tr_cx, tr_cy = _box_center_xy(tr_box)
                dist = float(np.hypot(det_cx - tr_cx, det_cy - tr_cy))
                if dist > float(TRACK_NEARBY_REUSE_PX):
                    continue
                if _size_similarity_ratio(det_box, tr_box) < float(TRACK_NEARBY_SIZE_RATIO):
                    continue

                if dist < best_dist:
                    best_dist = dist
                    best_i = i

            if best_i is not None:
                tid = track_ids[best_i]
                x1, y1, x2, y2 = det_box
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                w, h = max(2.0, x2 - x1), max(2.0, y2 - y1)
                self.tracks[tid].update([cx, cy, w, h], t_meas=t_now, conf=float(d["conf"]))
                self.tracks[tid].cls = int(d["cls"])
                self.tracks[tid].name = str(d["name"])
                used_tracks.add(tid)
                used_dets.add(j)
                unmatched_track_idx.discard(best_i)
                continue

            x1, y1, x2, y2 = det_box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w, h = max(2.0, x2 - x1), max(2.0, y2 - y1)
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = KalmanBox(cx, cy, w, h, t0=t_now, cls=int(d["cls"]), name=str(d["name"]), conf=float(d["conf"]))

        self._prune(t_now)

    def _prune(self, t_now):
        dead = [tid for tid, tr in self.tracks.items() if tr.age_since_meas(t_now) > float(TRACK_MAX_AGE_SEC)]
        for tid in dead:
            self.tracks.pop(tid, None)

    def get_tracks_as_dets(self, t_now):
        out = []
        for tid, tr in self.tracks.items():
            if not tr.confirmed:
                continue
            age = tr.age_since_meas(t_now)
            if age > float(TRACK_MAX_AGE_SEC):
                continue
            x1, y1, x2, y2 = tr.xyxy()
            out.append(
                {
                    "xyxy": (x1, y1, x2, y2),
                    "cls": int(tr.cls),
                    "conf": float(tr.conf),
                    "name": str(tr.name),
                    "track_id": int(tid),
                    "age": float(age),
                }
            )
        return out


# -----------------------
# UI thread (owns imshow/waitKey)
# -----------------------
def ui_thread_fn():
    global ui_last_key, ui_has_frame, ui_front_bgr

    if (not ENABLE_DISPLAY) or (not SHOW_VIDEO and not ENABLE_DISPLAY):
        return

    cv2.imshow("cv/av bug", np.zeros(1, dtype=np.uint8))
    cv2.destroyAllWindows()

    flags = cv2.WINDOW_NORMAL if WINDOW_RESIZABLE else cv2.WINDOW_AUTOSIZE
    cv2.namedWindow(WIN, flags)
    if START_FULLSCREEN:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    elif WINDOW_RESIZABLE:
        cv2.resizeWindow(WIN, WINDOW_W, WINDOW_H)

    period = None if (DISPLAY_HZ is None or DISPLAY_HZ <= 0) else (1.0 / float(DISPLAY_HZ))
    next_t = time.perf_counter()

    while not stop_event.is_set():
        with ui_lock:
            frame = ui_front_bgr if ui_has_frame else None

        if frame is not None:
            cv2.imshow(WIN, frame)

        k = cv2.waitKey(1) & 0xFF
        if k != 255:
            with ui_lock:
                ui_last_key = k
            if k == 27:
                stop_event.set()
                break

        if period is not None:
            next_t += period
            now = time.perf_counter()
            sleep_s = next_t - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = now


# -----------------------
# Main
# -----------------------
def main():
    global latest_frame_time
    global display_buf_bgr, display_buf_shape
    global ui_front_bgr, ui_back_bgr, ui_has_frame, ui_last_key

    # Screen fit: one-time optional
    max_disp_w = None
    max_disp_h = None
    if ENABLE_DISPLAY and FIT_TO_SCREEN:
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            sw = int(root.winfo_screenwidth())
            sh = int(root.winfo_screenheight())
            root.destroy()
            max_disp_w = int(sw * 0.95)
            max_disp_h = int(sh * 0.90)
        except Exception:
            max_disp_w, max_disp_h = None, None

    log("Looking for the next best device...")
    device = discover_one_device(max_search_duration_seconds=10)
    if device is None:
        raise SystemExit("No device found.")

    log(f"Connected to {device}. Press ESC or Ctrl-C to quit.")
    log("Keys: ESC quit | c recenter (head+gaze angles)")

    # Threads
    if ENABLE_DISPLAY:
        threading.Thread(target=ui_thread_fn, daemon=True).start()
    if SHOW_VIDEO:
        threading.Thread(target=video_thread_fn, args=(device,), daemon=True).start()
    threading.Thread(target=gaze_thread_fn, args=(device,), daemon=True).start()
    threading.Thread(target=imu_thread_fn, args=(device,), daemon=True).start()
    threading.Thread(target=detector_thread_fn, daemon=True).start()

    tracker = SimpleSORTTracker() if USE_TRACKER else None

    # Rate instrumentation
    loop_dt_hist = deque(maxlen=120)
    video_dt_hist = deque(maxlen=120)
    t_loop_prev = time.time()
    t_video_prev = None
    next_print_t = time.time()

    # Gaze smoothing
    gaze_smooth_x = None
    gaze_smooth_y = None

    # Gaze-hit memory
    last_hit = None
    last_hit_wall_t = -1e9
    last_processed_det_time = None

    loop_period = 1.0 / float(TARGET_LOOP_HZ)

    # Depth + angle smoothers
    depth_smoother = EWMASmoother(alpha=DEPTH_SMOOTH_ALPHA)
    head_yaw_smoother = EWMASmoother(alpha=ANGLE_ALPHA)
    head_pitch_smoother = EWMASmoother(alpha=ANGLE_ALPHA)
    gaze_yaw_smoother = EWMASmoother(alpha=ANGLE_ALPHA)
    gaze_pitch_smoother = EWMASmoother(alpha=ANGLE_ALPHA)

    # Recenter offsets
    have_recenter = False
    yaw_offset = 0.0
    pitch_offset = 0.0

    S2I = scene_to_imu_matrix()

    try:
        while not stop_event.is_set():
            loop_start = time.time()
            loop_dt_hist.append(loop_start - t_loop_prev)
            t_loop_prev = loop_start

            # ---- Handle key events (UI thread writes ui_last_key) ----
            k = -1
            with ui_lock:
                if ui_last_key != -1:
                    k = ui_last_key
                    ui_last_key = -1

            # ---- Gaze snapshot ----
            with gaze_lock:
                g = latest_gaze
            if g is None:
                time.sleep(0.001)
                continue

            gx, gy = float(g.x), float(g.y)
            worn = bool(g.worn)
            t_unix = float(g.timestamp_unix_seconds)

            with frame_lock:
                latest_frame_time = t_unix

            # Smooth gaze
            if gaze_smooth_x is None:
                gaze_smooth_x, gaze_smooth_y = gx, gy
            else:
                a = float(GAZE_SMOOTH_ALPHA)
                gaze_smooth_x = (1.0 - a) * gaze_smooth_x + a * gx
                gaze_smooth_y = (1.0 - a) * gaze_smooth_y + a * gy
            gx_s, gy_s = float(gaze_smooth_x), float(gaze_smooth_y)

            # ---- Video snapshot ----
            bgr_pixels = None
            h = w = None
            vid_stale_s = np.nan

            if SHOW_VIDEO:
                with video_lock:
                    src = video_buf_bgr
                    pub_t = latest_video_wall_t
                    if src is not None:
                        if (display_buf_bgr is None) or (display_buf_shape != src.shape):
                            display_buf_bgr = np.empty_like(src)
                            display_buf_shape = src.shape
                        np.copyto(display_buf_bgr, src)
                        bgr_pixels = display_buf_bgr

                if pub_t is not None:
                    vid_stale_s = float(max(0.0, time.time() - pub_t))

                if bgr_pixels is not None:
                    now_v = time.time()
                    if t_video_prev is not None:
                        video_dt_hist.append(now_v - t_video_prev)
                    t_video_prev = now_v
                    h, w = bgr_pixels.shape[:2]

            # If headless display but still want HUD on black canvas
            if (not SHOW_VIDEO) and ENABLE_DISPLAY:
                h, w = 720, 1280
                bgr_pixels = np.zeros((h, w, 3), dtype=np.uint8)

            # ---- IMU snapshot ----
            now_wall = time.time()
            with imu_lock:
                imu_w = latest_imu_angvel
                imu_pub = latest_imu_wall_t
                imu = latest_imu

            imu_angvel = None
            imu_fresh = False
            if imu_w is not None and imu_pub is not None:
                if (now_wall - imu_pub) <= float(IMU_FRESH_S):
                    imu_angvel = float(imu_w)
                    imu_fresh = True

            # ---- Detector snapshot ----
            with det_lock:
                dets = latest_dets
                det_fps = float(latest_det_fps)
                det_age = (time.time() - latest_det_time) if latest_det_time is not None else np.nan
                infer_s = float(latest_infer_s) if np.isfinite(latest_infer_s) else np.nan
                det_pub_t = latest_det_time

            # ---- Tracking ----
            if USE_TRACKER and tracker is not None:
                tracker.predict(now_wall)
                do_update = (det_pub_t is not None) and (det_pub_t != last_processed_det_time)
                if do_update:
                    tracker.update_with_dets(dets, t_now=now_wall)
                    last_processed_det_time = det_pub_t
                view_dets = tracker.get_tracks_as_dets(now_wall)
            else:
                view_dets = dets

            loop_hz = hz_from_hist(loop_dt_hist)
            video_hz = hz_from_hist(video_dt_hist) if SHOW_VIDEO else np.nan

            # ---- Governor update ----
            gov_health_update(now_wall, vid_stale_s=vid_stale_s, loop_hz=loop_hz, imu_angvel=imu_angvel)
            enabled, reason, cd_left = gov_can_run(now_wall)

            # ---- Depth + head/gaze angles (requires IMU quaternion) ----
            d_cm = np.nan
            miss_mm = np.nan
            ipd_mm = np.nan
            valid_depth = False

            head_yaw_meas = np.nan
            head_pitch_meas = np.nan
            gaze_yaw_meas = np.nan
            gaze_pitch_meas = np.nan

            if imu is not None and hasattr(imu, "quaternion"):
                q = imu.quaternion  # w,x,y,z
                R_imu_to_world = quat_to_rotmat_wxyz(float(q.w), float(q.x), float(q.y), float(q.z))

                # Head
                head_world = safe_unit(R_imu_to_world @ HEAD_FORWARD_LOCAL)
                head_elev, head_azim = cartesian_to_spherical_world(head_world)
                head_yaw_meas = wrap_deg_180(head_azim)
                head_pitch_meas = float(np.clip(head_elev, -90.0, 90.0)) if np.isfinite(head_elev) else head_elev

                # Gaze ray -> world
                try:
                    gaze_ray = gaze_ray_from_optical_axes(g)  # scene coords
                    gaze_imu = safe_unit(S2I @ gaze_ray) if APPLY_SCENE_TO_IMU_FOR_GAZE else gaze_ray
                    gaze_world = safe_unit(R_imu_to_world @ gaze_imu)
                    g_elev, g_azim = cartesian_to_spherical_world(gaze_world)
                    gaze_yaw_meas = wrap_deg_180(g_azim)
                    gaze_pitch_meas = float(np.clip(g_elev, -90.0, 90.0)) if np.isfinite(g_elev) else g_elev
                except Exception:
                    pass

                # Vergence depth + IPD
                try:
                    L_mm = np.array([g.eyeball_center_left_x, g.eyeball_center_left_y, g.eyeball_center_left_z], dtype=float)
                    R_mm = np.array([g.eyeball_center_right_x, g.eyeball_center_right_y, g.eyeball_center_right_z], dtype=float)
                    uL = np.array([g.optical_axis_left_x, g.optical_axis_left_y, g.optical_axis_left_z], dtype=float)
                    uR = np.array([g.optical_axis_right_x, g.optical_axis_right_y, g.optical_axis_right_z], dtype=float)

                    ipd_mm = float(np.linalg.norm(L_mm - R_mm))
                    valid_depth, d_m, miss_mm_val = vergence_depth_from_eyestate(
                        L_mm, uL, R_mm, uR, miss_max_mm=MISS_MAX_MM, denom_min=DENOM_MIN
                    )
                    if worn and valid_depth:
                        d_s = depth_smoother.update(d_m)
                    else:
                        d_s = depth_smoother.y
                    d_cm = (d_s * 100.0) if np.isfinite(d_s) else np.nan
                    miss_mm = float(miss_mm_val) if np.isfinite(miss_mm_val) else np.nan
                except Exception:
                    pass

            # ---- Recenter key ----
            if k in (ord("c"), ord("C")) and np.isfinite(head_yaw_meas) and np.isfinite(head_pitch_meas):
                yaw_offset = float(head_yaw_meas)
                pitch_offset = float(head_pitch_meas)
                have_recenter = True
                head_yaw_smoother.y = 0.0
                head_pitch_smoother.y = 0.0
                gaze_yaw_smoother.y = 0.0
                gaze_pitch_smoother.y = 0.0
                log(f"[recenter] yaw_offset={yaw_offset:+.2f} deg, pitch_offset={pitch_offset:+.2f} deg")

            # Apply recenter offsets for display/logging
            if have_recenter:
                head_yaw_disp = wrap_deg_180(head_yaw_meas - yaw_offset) if np.isfinite(head_yaw_meas) else np.nan
                head_pitch_disp = float(np.clip(head_pitch_meas - pitch_offset, -90.0, 90.0)) if np.isfinite(head_pitch_meas) else np.nan
                gaze_yaw_disp = wrap_deg_180(gaze_yaw_meas - yaw_offset) if np.isfinite(gaze_yaw_meas) else np.nan
                gaze_pitch_disp = float(np.clip(gaze_pitch_meas - pitch_offset, -90.0, 90.0)) if np.isfinite(gaze_pitch_meas) else np.nan
            else:
                head_yaw_disp, head_pitch_disp = head_yaw_meas, head_pitch_meas
                gaze_yaw_disp, gaze_pitch_disp = gaze_yaw_meas, gaze_pitch_meas

            # Smooth displayed angles
            head_yaw_s = head_yaw_smoother.update(head_yaw_disp)
            head_pitch_s = head_pitch_smoother.update(head_pitch_disp)
            gaze_yaw_s = gaze_yaw_smoother.update(gaze_yaw_disp)
            gaze_pitch_s = gaze_pitch_smoother.update(gaze_pitch_disp)

            # ---- Gaze-on-object ----
            hit = hit_mode = hit_dist = None
            if (bgr_pixels is not None) and worn and (w is not None) and (h is not None) and (0 <= gx_s < w) and (0 <= gy_s < h):
                hit, hit_mode, hit_dist = gaze_object_hit(gx_s, gy_s, view_dets)

            if hit is not None:
                last_hit = hit
                last_hit_wall_t = now_wall

            show_hit = None
            if last_hit is not None and (now_wall - last_hit_wall_t) <= GAZE_HIT_HOLD_SEC:
                show_hit = last_hit

            # ---- Console print (includes tracked objects) ----
            if ENABLE_PRINTS and now_wall >= next_print_t:
                next_print_t = now_wall + (1.0 / max(PRINT_HZ, 1e-6))

                # top objects by confidence (tracks)
                topk = sorted(view_dets, key=lambda d: float(d.get("conf", 0.0)), reverse=True)[:int(TOPK_LOG)]
                objs_str = ", ".join(
                    [f"{d.get('name','?')}#{int(d.get('track_id',-1))}({float(d.get('conf',0.0)):.2f})"
                     for d in topk]
                ) if topk else "none"

                if show_hit is None:
                    hit_str = "none"
                else:
                    tid = int(show_hit.get("track_id", -1))
                    nm = show_hit.get("name", "?")
                    cf = float(show_hit.get("conf", 0.0))
                    hit_str = f"{nm}#{tid}({cf:.2f}) mode={hit_mode or '--'} d={(f'{hit_dist:.1f}px' if hit_dist is not None else '--')}"

                gov_str = f"{'ON' if enabled else 'OFF'}({reason}) cd={cd_left:.2f}s"
                imu_str = f"{imu_angvel:.2f}rad/s" if (imu_angvel is not None and imu_fresh) else "NA"
                depth_str = f"{d_cm:.1f}cm" if np.isfinite(d_cm) else "N/A"

                log(
                    f"t={t_unix:.3f} worn={worn} gaze=({gx_s:.1f},{gy_s:.1f}) | "
                    f"rates: loop={loop_hz:.1f}Hz video={(video_hz if np.isfinite(video_hz) else 0):.1f}Hz "
                    f"det~{det_fps:.1f}Hz det_age={(det_age if np.isfinite(det_age) else -1):.2f}s "
                    f"infer={(infer_s*1000.0 if np.isfinite(infer_s) else -1):.0f}ms | "
                    f"imu|w|={imu_str} | YOLO={gov_str} | tracks={len(view_dets)} | "
                    f"objs: {objs_str} | gaze_on: {hit_str} | "
                    f"depth={depth_str} miss={(miss_mm if np.isfinite(miss_mm) else -1):.1f}mm IPD={(ipd_mm if np.isfinite(ipd_mm) else -1):.1f}mm | "
                    f"HEAD(yaw,pitch)=({head_yaw_s:+.0f},{head_pitch_s:+.0f}) GAZE(yaw,pitch)=({gaze_yaw_s:+.0f},{gaze_pitch_s:+.0f})"
                )

            # ---- Overlays + UI publish ----
            if ENABLE_DISPLAY and bgr_pixels is not None:
                draw_detections(bgr_pixels, view_dets)

                if DRAW_GAZE and worn and (w is not None) and (h is not None) and (0 <= gx_s < w) and (0 <= gy_s < h):
                    cv2.circle(bgr_pixels, (int(gx_s), int(gy_s)), 10, (0, 0, 255), 2)

                # HUD text
                imu_hud = f"imu|w|={(imu_angvel if imu_angvel is not None else 0):.2f} rad/s (gate={'on' if ENABLE_IMU_GATE else 'off'})"
                gov_hud = f"YOLO={'ON' if enabled else 'OFF'} | {reason} | cd={cd_left:.2f}s"
                ang_hud = f"HEAD(yaw,pitch)=({head_yaw_s:+.0f},{head_pitch_s:+.0f})  GAZE(yaw,pitch)=({gaze_yaw_s:+.0f},{gaze_pitch_s:+.0f})"

                cv2.putText(bgr_pixels, f"loop={loop_hz:.1f}Hz video={(video_hz if np.isfinite(video_hz) else 0):.1f}Hz | vid_stale={(vid_stale_s if np.isfinite(vid_stale_s) else 0):.3f}s",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(bgr_pixels, f"det={det_fps:.1f}Hz age={(det_age if np.isfinite(det_age) else 0):.2f}s infer={(infer_s*1000.0 if np.isfinite(infer_s) else 0):.0f}ms | tracks={len(view_dets)}",
                            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(bgr_pixels, imu_hud, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(bgr_pixels, gov_hud, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(bgr_pixels, ang_hud, (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

                # Depth bar
                if DRAW_DEPTH_BAR:
                    draw_depth_bar(
                        bgr_pixels,
                        d_cm=d_cm,
                        valid=bool(worn and valid_depth),
                        miss_mm=miss_mm,
                        ipd_mm=ipd_mm,
                        x=DEPTH_BAR_X,
                        y=DEPTH_BAR_Y + 100,
                        w=DEPTH_BAR_W,
                        h=DEPTH_BAR_H,
                        dmin_cm=DEPTH_BAR_MIN_CM,
                        dmax_cm=DEPTH_BAR_MAX_CM,
                    )

                # Angle HUD circles (right side)
                if DRAW_ANGLE_HUD and (w is not None) and (h is not None):
                    cx = w - HUD_MARGIN - HUD_RADIUS
                    cy1 = HUD_MARGIN + HUD_RADIUS
                    cy2 = cy1 + 2 * HUD_RADIUS + HUD_GAP_Y
                    draw_dual_angle_circle(bgr_pixels, (cx, cy1), HUD_RADIUS, "Yaw / Azimuth", head_yaw_s, gaze_yaw_s)
                    draw_dual_angle_circle(bgr_pixels, (cx, cy2), HUD_RADIUS, "Pitch / Elev", head_pitch_s, gaze_pitch_s)
                    cv2.putText(bgr_pixels, "c=recenter | ESC=quit", (cx - HUD_RADIUS, cy2 + HUD_RADIUS + 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2, cv2.LINE_AA)

                # Gaze target panel
                if show_hit is None:
                    draw_gaze_target_panel(bgr_pixels, "none", x=20, y=190)
                else:
                    tid = int(show_hit.get("track_id", -1))
                    nm = show_hit.get("name", "?")
                    cf = float(show_hit.get("conf", 0.0))
                    txt = f"{nm}#{tid} ({cf:.2f})" if tid >= 0 else f"{nm} ({cf:.2f})"
                    draw_gaze_target_panel(bgr_pixels, txt, x=20, y=190)
                    x1, y1, x2, y2 = show_hit["xyxy"]
                    cv2.rectangle(bgr_pixels, (int(round(x1)), int(round(y1))),
                                  (int(round(x2)), int(round(y2))), (0, 0, 255), 3)

                # Display-only resize / fit-to-screen
                out = bgr_pixels
                if ENABLE_DISPLAY_RESIZE and DISPLAY_RESIZE_WIDTH is not None and out is not None:
                    src_h, src_w = out.shape[:2]
                    if src_w != int(DISPLAY_RESIZE_WIDTH):
                        new_w = int(DISPLAY_RESIZE_WIDTH)
                        new_h = int(round(src_h * (new_w / float(src_w))))
                        out = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_AREA)

                if FIT_TO_SCREEN and (max_disp_w is not None) and (max_disp_h is not None):
                    hh, ww = out.shape[:2]
                    scale = min(max_disp_w / float(ww), max_disp_h / float(hh))
                    if scale < 1.0:
                        out = cv2.resize(out, (max(1, int(ww * scale)), max(1, int(hh * scale))), interpolation=cv2.INTER_AREA)

                with ui_lock:
                    if (ui_back_bgr is None) or (ui_back_bgr.shape != out.shape):
                        ui_back_bgr = np.empty_like(out)
                    np.copyto(ui_back_bgr, out)
                    ui_front_bgr = ui_back_bgr
                    ui_has_frame = True

            # ---- Pace loop ----
            elapsed = time.time() - loop_start
            to_sleep = loop_period - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

    except KeyboardInterrupt:
        log("\nKeyboardInterrupt received. Stopping...")
    finally:
        stop_event.set()
        log("Stopping streams and closing windows...")
        try:
            device.close()
        except Exception:
            pass
        if ENABLE_DISPLAY:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        log("Done.")


if __name__ == "__main__":
    main()
