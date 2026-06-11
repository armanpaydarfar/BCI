"""
Live Neon scene-camera viewer with gaze overlay and vergence depth.

Restored from OBS/Neon_Stream_PupilRealtime.py (removed in cleanup commit
5f1c0fb). Updated to use NEON_COMPANION_HOST from config.py for direct
IP connection when mDNS multicast is blocked (university IoT VLANs).

Usage (lsl env):
    python neon_viewer.py
"""

import time
from collections import deque

import numpy as np
import cv2

try:
    import config as _cfg
    _NEON_HOST = str(getattr(_cfg, "NEON_COMPANION_HOST", ""))
except Exception:
    _NEON_HOST = ""

# -----------------------
# CONFIG
# -----------------------
SHOW_VIDEO   = True   # show scene video window
DRAW_GAZE    = True   # draw red gaze circle (only if SHOW_VIDEO)
PRINT_HZ     = 30     # console print rate

# Depth estimator settings
MISS_MAX_MM  = 30.0
DENOM_MIN    = 1e-4
SMOOTH_ALPHA = 0.15


def vergence_depth_from_eyestate(L_mm, u, R_mm, v, *, miss_max_mm=30.0, denom_min=1e-4):
    L, R = np.asarray(L_mm, float), np.asarray(R_mm, float)
    u, v = np.asarray(u, float), np.asarray(v, float)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return False, np.nan, np.nan, np.full(3, np.nan)
    u, v = u / nu, v / nv
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
    C = 0.5 * (L + R)
    d_m = float(np.linalg.norm(P - C)) / 1000.0
    valid = (miss_mm <= miss_max_mm) and np.isfinite(d_m) and (d_m > 0.05)
    return bool(valid), (d_m if valid else np.nan), miss_mm / 1000.0, (P if valid else np.full(3, np.nan))


class EWMASmoother:
    def __init__(self, alpha, init_value=np.nan):
        self.alpha = float(alpha)
        self.y = float(init_value)

    def update(self, x):
        x = float(x)
        if not np.isfinite(x):
            return self.y
        self.y = x if not np.isfinite(self.y) else self.alpha * x + (1.0 - self.alpha) * self.y
        return self.y


def _hz(hist):
    if len(hist) < 5:
        return np.nan
    m = float(np.mean(hist))
    return (1.0 / m) if m > 1e-9 else np.nan


def main():
    from pupil_labs.realtime_api.simple import discover_one_device

    if SHOW_VIDEO:
        # Workaround for https://github.com/opencv/opencv/issues/21952
        cv2.imshow("cv/av bug", np.zeros(1))
        cv2.destroyAllWindows()
        cv2.namedWindow("Neon viewer — ESC to quit", cv2.WINDOW_NORMAL)

    if _NEON_HOST:
        from pupil_labs.realtime_api.simple import Device
        print(f"Connecting directly to Neon at {_NEON_HOST}…")
        device = Device(address=_NEON_HOST, port=8080)
    else:
        print("Looking for the next best device…")
        device = discover_one_device(max_search_duration_seconds=10)
        if device is None:
            print("No device found.")
            raise SystemExit(-1)

    print(f"Connected to {device}. Press ESC or Ctrl-C to quit.")

    depth_smoother = EWMASmoother(alpha=SMOOTH_ALPHA)
    loop_dt = deque(maxlen=120)
    gaze_dt = deque(maxlen=120)
    video_dt = deque(maxlen=120)
    t_loop_prev = time.time()
    t_gaze_prev = t_video_prev = None
    next_print = time.time()

    try:
        while True:
            now = time.time()
            loop_dt.append(now - t_loop_prev)
            t_loop_prev = now

            gaze = device.receive_gaze_datum()
            t_g = time.time()
            if t_gaze_prev is not None:
                gaze_dt.append(t_g - t_gaze_prev)
            t_gaze_prev = t_g

            bgr = None
            h = w = None
            if SHOW_VIDEO:
                bgr, _ = device.receive_scene_video_frame()
                t_v = time.time()
                if t_video_prev is not None:
                    video_dt.append(t_v - t_video_prev)
                t_video_prev = t_v
                if bgr is not None:
                    h, w = bgr.shape[:2]

            gx = float(gaze.x)
            gy = float(gaze.y)
            worn = bool(gaze.worn)
            t_unix = float(gaze.timestamp_unix_seconds)
            pupil_l = float(gaze.pupil_diameter_left)
            pupil_r = float(gaze.pupil_diameter_right)

            L_mm = np.array([gaze.eyeball_center_left_x,  gaze.eyeball_center_left_y,  gaze.eyeball_center_left_z],  float)
            R_mm = np.array([gaze.eyeball_center_right_x, gaze.eyeball_center_right_y, gaze.eyeball_center_right_z], float)
            uL   = np.array([gaze.optical_axis_left_x,    gaze.optical_axis_left_y,    gaze.optical_axis_left_z],    float)
            uR   = np.array([gaze.optical_axis_right_x,   gaze.optical_axis_right_y,   gaze.optical_axis_right_z],   float)

            valid, d_m, miss_m, _ = vergence_depth_from_eyestate(L_mm, uL, R_mm, uR, miss_max_mm=MISS_MAX_MM, denom_min=DENOM_MIN)
            if worn and valid:
                d_smooth = depth_smoother.update(d_m)
            else:
                d_smooth = depth_smoother.y

            d_cm   = d_smooth * 100.0 if np.isfinite(d_smooth) else np.nan
            miss_mm = miss_m * 1000.0 if np.isfinite(miss_m) else np.nan
            ipd_mm = float(np.linalg.norm(L_mm - R_mm))

            now = time.time()
            if now >= next_print:
                next_print = now + 1.0 / max(PRINT_HZ, 1e-6)
                depth_str = f"{d_cm:.1f}cm" if np.isfinite(d_cm) else "N/A"
                print(
                    f"t={t_unix:.3f} worn={worn} gaze=({gx:.1f},{gy:.1f}) "
                    f"IPD~{ipd_mm:.1f}mm valid={valid} miss={miss_mm:.1f}mm depth={depth_str} | "
                    f"loop={_hz(loop_dt):.1f}Hz gaze={_hz(gaze_dt):.1f}Hz "
                    f"video={_hz(video_dt):.1f}Hz"
                )

            if SHOW_VIDEO and bgr is not None:
                if DRAW_GAZE and worn and h and w and (0 <= gx < w) and (0 <= gy < h):
                    cv2.circle(bgr, (int(gx), int(gy)), 15, (0, 0, 255), 2)
                cv2.putText(bgr, f"worn={worn} | pupil(L,R)=({pupil_l:.2f},{pupil_r:.2f})",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                depth_lbl = f"depth: {d_cm:.1f} cm | miss: {miss_mm:.1f} mm" if np.isfinite(d_cm) else f"depth: N/A | miss: {miss_mm:.1f} mm"
                cv2.putText(bgr, depth_lbl,
                            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(bgr, f"loop={_hz(loop_dt):.1f}Hz  gaze={_hz(gaze_dt):.1f}Hz  video={_hz(video_dt):.1f}Hz",
                            (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Neon viewer — ESC to quit", bgr)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        device.close()
        if SHOW_VIDEO:
            cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
