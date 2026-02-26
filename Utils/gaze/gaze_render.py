# Utils/gaze/gaze_render.py
"""
gaze_render.py

OpenCV drawing helpers for the Neon gaze pipeline.

Rules:
- NO device I/O
- NO threading
- NO Ultralytics/YOLO
- OK to import cv2 (drawing only)

If you run headless, do not import/call anything from this file.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import cv2


def draw_detections(bgr: np.ndarray, dets) -> None:
    """
    Draw bounding boxes + labels.
    det schema:
      { "xyxy": (x1,y1,x2,y2), "name": str, "conf": float, "track_id": int, "age": float }
    """
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
        cv2.putText(
            bgr,
            label,
            (p1[0] + 3, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )


def draw_gaze_target_panel(bgr: np.ndarray, text: str, *, x: int = 20, y: int = 150) -> None:
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
    cv2.putText(
        bgr,
        panel_text,
        (x1 + pad, y1 + th + pad - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


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

    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    cv2.rectangle(img, (x + 2, y + 2), (x + w - 2, y + h - 2), (25, 25, 25), -1)

    if valid and np.isfinite(d_cm):
        frac = (float(d_cm) - float(dmin_cm)) / (float(dmax_cm) - float(dmin_cm) + 1e-9)
        frac = max(0.0, min(1.0, float(frac)))
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


def draw_dual_angle_circle(
    img: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    label: str,
    head_angle_deg: float,
    gaze_angle_deg: float,
) -> None:
    cx, cy = int(center[0]), int(center[1])

    cv2.circle(img, (cx, cy), int(radius), (255, 255, 255), 2)
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

    def draw_ray(angle_deg: float, color: Tuple[int, int, int], tag: str) -> None:
        if not np.isfinite(angle_deg):
            return
        theta = np.radians(float(angle_deg))
        dx = radius * np.sin(theta)
        dy = -radius * np.cos(theta)
        x2 = int(cx + dx)
        y2 = int(cy + dy)
        cv2.line(img, (cx, cy), (x2, y2), color, 3, lineType=cv2.LINE_AA)
        cv2.circle(img, (x2, y2), 6, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(img, tag, (x2 + 8, y2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, lineType=cv2.LINE_AA)

    draw_ray(head_angle_deg, (255, 0, 0), "H")  # blue
    draw_ray(gaze_angle_deg, (0, 0, 255), "G")  # red
