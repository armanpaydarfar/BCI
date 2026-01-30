import time
import threading
from collections import deque

import numpy as np
import cv2

from pupil_labs.realtime_api.simple import discover_one_device

# YOLOv8 (Ultralytics)
try:
    from ultralytics import YOLO
except ImportError as e:
    raise SystemExit(
        "Missing dependency 'ultralytics'. Install with:\n"
        "  pip install ultralytics\n"
    ) from e


# -----------------------
# CONFIG
# -----------------------
SHOW_VIDEO = True
DRAW_GAZE = True

# ---- Window sizing / fullscreen (NO TOGGLE) ----
WINDOW_RESIZABLE = True          # allows manual resizing by dragging
START_FULLSCREEN = False          # if True, starts fullscreen (ignores WINDOW_W/H)
WINDOW_W = 1280                  # initial window size if not fullscreen
WINDOW_H = 720

# Inference rate target (CPU-friendly)
DET_HZ = 7.5               # set anywhere 5–10 Hz
DET_CONF = 0.35            # confidence threshold
DET_IOU = 0.45             # NMS IoU
MODEL_NAME = "yolov8n.pt"  # nano = fastest baseline

# Optional: shrink frames before detection to speed CPU inference
DETECT_RESIZE_WIDTH = 640  # set None to disable resize

# Console prints
PRINT_HZ = 5               # how often to print summary

# Gaze-on-object behavior
GAZE_HIT_HOLD_SEC = 1.0    # how long to keep displaying last hit if gaze leaves boxes

# Window name
WIN = "Neon + YOLO (ESC to quit)"


# -----------------------
# Shared state between threads
# -----------------------
frame_lock = threading.Lock()
latest_frame_bgr = None
latest_frame_time = None

det_lock = threading.Lock()
latest_dets = []           # list of dicts: {xyxy, cls, conf, name}
latest_det_time = None
latest_det_fps = 0.0

stop_event = threading.Event()


def _resize_keep_aspect(bgr: np.ndarray, target_w: int):
    """Resize image to target width while preserving aspect ratio."""
    h, w = bgr.shape[:2]
    if w == target_w:
        return bgr, 1.0
    scale = target_w / float(w)
    new_h = int(round(h * scale))
    resized = cv2.resize(bgr, (target_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale


def detector_thread_fn():
    """
    Runs YOLO on the latest frame at ~DET_HZ.
    Uses 'latest-frame' semantics (drops intermediate frames).
    """
    global latest_dets, latest_det_time, latest_det_fps

    model = YOLO(MODEL_NAME)
    # Force CPU
    try:
        model.to("cpu")
    except Exception:
        pass

    # Rate control
    period = 1.0 / max(DET_HZ, 1e-6)
    t_prev = time.time()
    fps_hist = deque(maxlen=30)

    while not stop_event.is_set():
        t0 = time.time()

        # Pull latest frame snapshot
        with frame_lock:
            if latest_frame_bgr is None:
                frame = None
                ftime = None
            else:
                frame = latest_frame_bgr.copy()
                ftime = latest_frame_time

        if frame is None:
            time.sleep(0.01)
            continue

        # Optional downscale for faster CPU inference
        scale = 1.0
        if DETECT_RESIZE_WIDTH is not None:
            frame_det, scale = _resize_keep_aspect(frame, DETECT_RESIZE_WIDTH)
        else:
            frame_det = frame

        # YOLO expects RGB typically; Ultralytics accepts numpy BGR but safest to convert
        rgb = cv2.cvtColor(frame_det, cv2.COLOR_BGR2RGB)

        # Inference
        try:
            results = model.predict(
                source=rgb,
                verbose=False,
                conf=DET_CONF,
                iou=DET_IOU,
                device="cpu",
            )
        except TypeError:
            results = model(rgb)

        dets = []
        if results and len(results) > 0:
            r0 = results[0]
            names = r0.names if hasattr(r0, "names") else getattr(model, "names", None)

            if hasattr(r0, "boxes") and r0.boxes is not None:
                boxes = r0.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)

                # Map boxes back to original resolution if we resized
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

        dt = time.time() - t_prev
        t_prev = time.time()
        if dt > 1e-6:
            fps_hist.append(1.0 / dt)
            latest_det_fps = float(np.mean(fps_hist)) if len(fps_hist) else 0.0

        elapsed = time.time() - t0
        to_sleep = period - elapsed
        if to_sleep > 0:
            time.sleep(to_sleep)


def draw_detections(bgr: np.ndarray, dets):
    """Draw boxes + labels onto frame."""
    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        name = d["name"]
        conf = d["conf"]

        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))

        cv2.rectangle(bgr, p1, p2, (0, 255, 0), 2)

        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(p1[1] - 8, th + 8)
        cv2.rectangle(
            bgr,
            (p1[0], y_text - th - 6),
            (p1[0] + tw + 6, y_text + 4),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            bgr,
            label,
            (p1[0] + 3, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )


def gaze_object_hit(gx, gy, dets):
    """
    Return the most 'specific' box containing gaze (smallest area), if any.
    """
    hits = []
    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        if (x1 <= gx <= x2) and (y1 <= gy <= y2):
            area = max(1.0, (x2 - x1) * (y2 - y1))
            hits.append((area, d))
    if not hits:
        return None
    hits.sort(key=lambda t: t[0])  # smallest area = likely most specific
    return hits[0][1]


def draw_gaze_target_panel(bgr: np.ndarray, text: str, *, x=20, y=110):
    """
    Draw a little panel that shows what object the gaze is currently on.
    """
    panel_text = f"GAZE ON: {text}"
    (tw, th), _ = cv2.getTextSize(panel_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

    pad = 10
    x2 = x + tw + 2 * pad
    y2 = y + th + 2 * pad

    # Semi-opaque black background: draw a filled rect then blend
    overlay = bgr.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, bgr, 0.45, 0, bgr)

    cv2.putText(
        bgr,
        panel_text,
        (x + pad, y + th + pad - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    global latest_frame_bgr, latest_frame_time

    # Workaround for https://github.com/opencv/opencv/issues/21952
    if SHOW_VIDEO:
        cv2.imshow("cv/av bug", np.zeros(1, dtype=np.uint8))
        cv2.destroyAllWindows()

    print("Looking for the next best device...")
    device = discover_one_device(max_search_duration_seconds=10)
    if device is None:
        print("No device found.")
        raise SystemExit(-1)

    print(f"Connected to {device}. Press ESC or Ctrl-C to quit.")

    # ---- Create window with resizing/fullscreen behavior (NO TOGGLE) ----
    if SHOW_VIDEO:
        flags = cv2.WINDOW_NORMAL if WINDOW_RESIZABLE else cv2.WINDOW_AUTOSIZE
        cv2.namedWindow(WIN, flags)

        if START_FULLSCREEN:
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            if WINDOW_RESIZABLE:
                cv2.resizeWindow(WIN, WINDOW_W, WINDOW_H)

    # Start detector thread
    det_thread = threading.Thread(target=detector_thread_fn, daemon=True)
    det_thread.start()

    # Rate instrumentation
    loop_dt_hist = deque(maxlen=120)
    video_dt_hist = deque(maxlen=120)
    t_loop_prev = time.time()
    t_video_prev = None

    # Print throttle
    next_print_t = time.time()

    # Gaze-hit memory
    last_hit = None
    last_hit_wall_t = -1e9

    try:
        while True:
            # --- Loop timing ---
            t_loop_now = time.time()
            loop_dt_hist.append(t_loop_now - t_loop_prev)
            t_loop_prev = t_loop_now

            # --- Get gaze (blocking) ---
            gaze = device.receive_gaze_datum()

            # --- Get scene frame (blocking) ---
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

            # --- Publish latest frame to detector (latest-frame semantics) ---
            if bgr_pixels is not None:
                with frame_lock:
                    latest_frame_bgr = bgr_pixels
                    latest_frame_time = t_unix

            # --- Pull latest detections snapshot ---
            with det_lock:
                dets = list(latest_dets)
                det_fps = float(latest_det_fps)
                det_age = (time.time() - latest_det_time) if latest_det_time is not None else np.nan

            # --- Compute rates ---
            def hz_from_hist(hist):
                if len(hist) < 5:
                    return np.nan
                m = float(np.mean(hist))
                return (1.0 / m) if m > 1e-9 else np.nan

            loop_hz = hz_from_hist(loop_dt_hist)
            video_hz = hz_from_hist(video_dt_hist) if SHOW_VIDEO else np.nan

            # --- Determine gaze-on-object (if any) ---
            hit = None
            if (bgr_pixels is not None) and worn and np.isfinite(gx) and np.isfinite(gy) and (0 <= gx < w) and (0 <= gy < h):
                hit = gaze_object_hit(gx, gy, dets)

            now_wall = time.time()
            if hit is not None:
                last_hit = hit
                last_hit_wall_t = now_wall

            # If no hit now, keep showing last hit briefly
            show_hit = None
            if last_hit is not None and (now_wall - last_hit_wall_t) <= GAZE_HIT_HOLD_SEC:
                show_hit = last_hit

            # --- Console print ---
            if now_wall >= next_print_t:
                next_print_t = now_wall + (1.0 / max(PRINT_HZ, 1e-6))

                top3 = sorted(dets, key=lambda d: d["conf"], reverse=True)[:3]
                top3_str = ", ".join([f"{d['name']}({d['conf']:.2f})" for d in top3]) if top3 else "none"

                if show_hit is None:
                    hit_str = "none"
                else:
                    hit_str = f"{show_hit['name']}({show_hit['conf']:.2f})"

                print(
                    f"t={t_unix:.3f} worn={worn} gaze=({gx:.1f},{gy:.1f}) | "
                    f"rates: loop={loop_hz:.1f}Hz video={(video_hz if np.isfinite(video_hz) else 0):.1f}Hz det~{det_fps:.1f}Hz "
                    f"det_age={(det_age if np.isfinite(det_age) else -1):.2f}s | "
                    f"top: {top3_str} | gaze_on: {hit_str}"
                )

            # --- Video overlay ---
            if SHOW_VIDEO and bgr_pixels is not None:
                draw_detections(bgr_pixels, dets)

                # draw gaze dot
                if DRAW_GAZE and worn and (0 <= gx < w) and (0 <= gy < h):
                    cv2.circle(bgr_pixels, (int(gx), int(gy)), 10, (0, 0, 255), 2)

                # HUD
                hud1 = f"worn={worn} | det={det_fps:.1f}Hz | det_age={(det_age if np.isfinite(det_age) else 0):.2f}s"
                hud2 = f"loop={loop_hz:.1f}Hz video={(video_hz if np.isfinite(video_hz) else 0):.1f}Hz | #dets={len(dets)}"
                cv2.putText(bgr_pixels, hud1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.putText(bgr_pixels, hud2, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, lineType=cv2.LINE_AA)

                # Gaze target panel
                if show_hit is None:
                    draw_gaze_target_panel(bgr_pixels, "none", x=20, y=90)
                else:
                    draw_gaze_target_panel(bgr_pixels, f"{show_hit['name']} ({show_hit['conf']:.2f})", x=20, y=90)

                    # Optional: highlight the hit box more strongly
                    x1, y1, x2, y2 = show_hit["xyxy"]
                    cv2.rectangle(
                        bgr_pixels,
                        (int(round(x1)), int(round(y1))),
                        (int(round(x2)), int(round(y2))),
                        (0, 0, 255),
                        3,
                    )

                cv2.imshow(WIN, bgr_pixels)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Stopping...")
    finally:
        stop_event.set()
        print("Stopping streams and closing windows...")
        try:
            device.close()
        except Exception:
            pass
        if SHOW_VIDEO:
            cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
