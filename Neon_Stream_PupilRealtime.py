import time
from collections import deque

import numpy as np

# Optional OpenCV
import cv2

from pupil_labs.realtime_api.simple import discover_one_device


# -----------------------
# CONFIG
# -----------------------
SHOW_VIDEO = True          # Toggle: show scene video window or not
DRAW_GAZE = True           # Toggle: draw the red gaze circle (only if SHOW_VIDEO)
PRINT_HZ = 30               # How often to print summary to console (Hz)

# Depth estimator settings
MISS_MAX_MM = 30.0         # Quality gate (ray miss distance at closest approach)
DENOM_MIN = 1e-4           # Near-parallel gate (depth becomes unstable)
SMOOTH_ALPHA = 0.15        # EWMA smoothing factor


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


def main():
    # Workaround for https://github.com/opencv/opencv/issues/21952
    if SHOW_VIDEO:
        cv2.imshow("cv/av bug", np.zeros(1))
        cv2.destroyAllWindows()

    print("Looking for the next best device...")
    device = discover_one_device(max_search_duration_seconds=10)
    if device is None:
        print("No device found.")
        raise SystemExit(-1)

    print(f"Connected to {device}. Press ESC or Ctrl-C to quit.")

    depth_smoother = EWMASmoother(alpha=SMOOTH_ALPHA)

    # --- rate instrumentation ---
    loop_dt_hist = deque(maxlen=120)
    gaze_dt_hist = deque(maxlen=120)
    video_dt_hist = deque(maxlen=120)

    t_loop_prev = time.time()
    t_gaze_prev = None
    t_video_prev = None

    # Print throttle
    next_print_t = time.time()

    try:
        while True:
            t_loop_now = time.time()
            loop_dt_hist.append(t_loop_now - t_loop_prev)
            t_loop_prev = t_loop_now

            # --- Get gaze (blocking) ---
            gaze = device.receive_gaze_datum()
            t_after_gaze = time.time()
            if t_gaze_prev is not None:
                gaze_dt_hist.append(t_after_gaze - t_gaze_prev)
            t_gaze_prev = t_after_gaze

            # --- Optionally get scene frame (blocking) ---
            bgr_pixels = None
            h = w = None
            if SHOW_VIDEO:
                bgr_pixels, _frame_datetime = device.receive_scene_video_frame()
                t_after_video = time.time()
                if t_video_prev is not None:
                    video_dt_hist.append(t_after_video - t_video_prev)
                t_video_prev = t_after_video
                h, w = bgr_pixels.shape[:2]

            # --- Extract gaze info ---
            gx = float(gaze.x)
            gy = float(gaze.y)
            worn = bool(gaze.worn)
            t_unix = float(gaze.timestamp_unix_seconds)

            pupil_left = float(gaze.pupil_diameter_left)
            pupil_right = float(gaze.pupil_diameter_right)

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

            valid, d_m, miss_m, _P_mm = vergence_depth_from_eyestate(
                L_mm, uL, R_mm, uR, miss_max_mm=MISS_MAX_MM, denom_min=DENOM_MIN
            )

            if worn and valid:
                d_smooth = depth_smoother.update(d_m)
            else:
                d_smooth = depth_smoother.y

            d_cm = (d_smooth * 100.0) if np.isfinite(d_smooth) else np.nan
            miss_mm = (miss_m * 1000.0) if np.isfinite(miss_m) else np.nan
            ipd_mm = float(np.linalg.norm(L_mm - R_mm))

            # --- Compute measured rates ---
            def hz_from_hist(hist):
                if len(hist) < 5:
                    return np.nan
                m = float(np.mean(hist))
                return (1.0 / m) if m > 1e-9 else np.nan

            loop_hz = hz_from_hist(loop_dt_hist)
            gaze_hz = hz_from_hist(gaze_dt_hist)
            video_hz = hz_from_hist(video_dt_hist) if SHOW_VIDEO else np.nan

            # --- Console print at PRINT_HZ ---
            now = time.time()
            if now >= next_print_t:
                next_print_t = now + (1.0 / max(PRINT_HZ, 1e-6))

                depth_str = f"{d_cm:.1f}cm" if np.isfinite(d_cm) else "N/A"
                loop_str = f"{loop_hz:.1f}Hz" if np.isfinite(loop_hz) else "N/A"
                gaze_str = f"{gaze_hz:.1f}Hz" if np.isfinite(gaze_hz) else "N/A"
                vid_str = f"{video_hz:.1f}Hz" if (SHOW_VIDEO and np.isfinite(video_hz)) else ("off" if not SHOW_VIDEO else "N/A")

                print(
                    f"t={t_unix:.3f} worn={worn} gaze=({gx:.1f},{gy:.1f}) "
                    f"IPD~{ipd_mm:.1f}mm valid={valid} miss={miss_mm:.1f}mm depth={depth_str} | "
                    f"rates: loop={loop_str} gaze={gaze_str} video={vid_str}"
                )

            # --- Video overlay (optional) ---
            if SHOW_VIDEO and bgr_pixels is not None:
                if DRAW_GAZE and worn and (0 <= gx < w) and (0 <= gy < h):
                    cv2.circle(bgr_pixels, (int(gx), int(gy)), 15, (0, 0, 255), 2)

                hud1 = f"worn={worn} | pupil(L,R)=({pupil_left:.2f},{pupil_right:.2f})"
                if np.isfinite(d_cm):
                    hud2 = f"depth: {d_cm:.1f} cm | miss: {miss_mm:.1f} mm | valid={valid}"
                else:
                    hud2 = f"depth: N/A | miss: {miss_mm:.1f} mm | valid={valid}"

                hud3 = f"rates: loop={loop_hz:.1f}Hz gaze={gaze_hz:.1f}Hz video={(video_hz if np.isfinite(video_hz) else 0):.1f}Hz"

                cv2.putText(bgr_pixels, hud1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.putText(bgr_pixels, hud2, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.putText(bgr_pixels, hud3, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, lineType=cv2.LINE_AA)

                cv2.imshow("Neon Depth (toggle SHOW_VIDEO) - Press ESC to quit", bgr_pixels)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Stopping...")
    finally:
        print("Stopping streams and closing windows...")
        device.close()
        if SHOW_VIDEO:
            cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
