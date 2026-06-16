# Vendored from harmony_vlm (https://github.com/vivianchen98/harmony_vlm) @ cfa01b6
# by Vivian Chen. Folded into the BCI repo for WS3 (2026-06-15). Edit here, not
# upstream; see Documents/SoftwareDocs/projects/harmony-bci/vlm-integration/.
"""
overlay_renderer.py — Rich segmentation + VLM state overlay for gaze intent system.

Canvas: 2060×1200 (1600×1200 video + 460-wide right panel).
"""

from __future__ import annotations

from typing import Literal, Optional

import cv2
import numpy as np

from perception.fixation_detector import FixationState
from perception.pupil_reader import GazeSample
from perception.object_detector import Detection

# ── layout constants ──────────────────────────────────────────────────────────

PANEL_W  = 464
FRAME_W  = 1600
FRAME_H  = 1200
CANVAS_W = 2064
CANVAS_H = 1200

PAD = 16

STABLE_WINDOW_SAMPLES = 16   # mirrors fixation_detector.py
MASK_ALPHA = 0.35

SPINNER_ASCII = ["|", "/", "-", "\\"]

# ── colour palette (BGR) ──────────────────────────────────────────────────────

BG_DARK     = (30, 30, 30)
BG_SECTION  = (45, 45, 45)
BG_CARD     = (55, 55, 55)

ACCENT_CYAN  = (200, 200, 50)
COL_WHITE    = (240, 240, 240)
COL_GRAY     = (140, 140, 140)
COL_GREEN    = (50, 220, 80)
COL_YELLOW   = (30, 210, 230)
COL_RED      = (50, 50, 220)
COL_HIT_BG   = (60, 100, 30)
COL_SPINNER  = (80, 200, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── VLM state ─────────────────────────────────────────────────────────────────

VlmState = Literal["IDLE", "AWAITING_SECOND", "THINKING", "DECIDED"]


# ── renderer ─────────────────────────────────────────────────────────────────

class OverlayRenderer:
    """
    Renders a 2060×1200 canvas: annotated video frame on the left (1600×1200)
    and a status panel on the right (460×1200).
    """

    CANVAS_W = CANVAS_W
    CANVAS_H = CANVAS_H

    def __init__(self) -> None:
        self._spinner_idx: int = 0
        self.caregiver_mode: bool = False   # toggled by pressing 'd'

    # ── public ────────────────────────────────────────────────────────────────

    def render(
        self,
        frame_bgr:         np.ndarray,
        detections:        list[Detection],
        hit:               Optional[Detection],
        fixation:          FixationState,
        gaze:              GazeSample,
        vlm_state:         VlmState,
        last_decision:     Optional[dict],
        api_mode:          str,
        pending_detection: Optional[Detection],
        frame_idx:         int,
        total_frames:      int,
        duration_s:        float,
        first_object:      Optional[Detection] = None,
        second_object_countdown: float = 0.0,
        hit_probability:   float = 0.0,
        calib_mode:        bool = False,
        calib_tracker=None,
        calib_tag_detection=None,
        calib_sample=None,
        calib_calibrated:  bool = False,
        hit_waypoint:      Optional[dict] = None,
    ) -> np.ndarray:
        """Return a 2060×1200 BGR canvas."""
        # Work on a copy so we don't mutate the original frame
        frame = frame_bgr.copy()

        # Resize frame to match panel height if needed (e.g. Pupil Core 720p)
        h, w = frame.shape[:2]
        scale = 1.0
        if h != CANVAS_H:
            scale = CANVAS_H / h
            new_w = int(w * scale)
            frame = cv2.resize(frame, (new_w, CANVAS_H), interpolation=cv2.INTER_LINEAR)

        # Scale detection coordinates & gaze to match resized frame
        if scale != 1.0:
            detections = [_scale_detection(d, scale) for d in detections]
            if hit is not None:
                hit = _scale_detection(hit, scale)
            if pending_detection is not None:
                pending_detection = _scale_detection(pending_detection, scale)
            gaze = GazeSample(
                timestamp_ns=gaze.timestamp_ns,
                x=gaze.x * scale,
                y=gaze.y * scale,
                eye_state=gaze.eye_state,
            )

        # Draw on the video frame portion
        self._draw_detections(frame, detections, hit)
        self._draw_gaze_fixation_ring(frame, gaze, fixation, hit, hit_probability, hit_waypoint)

        # Advance spinner when thinking
        if vlm_state == "THINKING":
            self._spinner_idx = (self._spinner_idx + 1) % len(SPINNER_ASCII)

        # Draw "1st" badge on first object in AWAITING_SECOND state
        if vlm_state == "AWAITING_SECOND" and first_object is not None:
            self._draw_first_object_badge(frame, first_object, scale)

        # Draw calibration overlay on video frame
        if calib_mode or calib_calibrated:
            self._draw_calib_on_frame(
                frame, gaze, calib_tag_detection, calib_sample,
                calib_mode, scale,
            )

        # Build right panel
        panel = np.full((CANVAS_H, PANEL_W, 3), BG_DARK, dtype=np.uint8)
        if calib_mode or calib_calibrated:
            self._draw_calib_panel(
                panel, calib_mode, calib_tracker, calib_tag_detection,
                calib_sample, calib_calibrated,
            )
        elif self.caregiver_mode:
            self._draw_panel(
                panel, detections, hit, fixation, gaze,
                vlm_state, last_decision, api_mode, pending_detection,
                frame_idx, total_frames, duration_s,
                first_object=first_object,
                second_object_countdown=second_object_countdown,
            )
        else:
            self._draw_patient_panel(panel, vlm_state, last_decision, fixation)

        # Calibration status indicator at bottom of panel (always visible)
        if calib_tracker is not None and calib_tracker.n_samples > 0 and not calib_mode:
            dx, dy = calib_tracker.pixel_offset
            status_text = f"Calib: dx={dx:.0f} dy={dy:.0f}"
            cv2.putText(
                panel, status_text,
                (PAD, CANVAS_H - 12), FONT, 0.40, ACCENT_CYAN, 1, cv2.LINE_AA,
            )

        return np.concatenate([frame, panel], axis=1)

    # ── video frame drawing ───────────────────────────────────────────────────

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        hit: Optional[Detection],
    ) -> None:
        for det in detections:
            is_hit = hit is not None and hit.label == det.label and hit.box_xyxy == det.box_xyxy

            # confidence-based border color
            if det.confidence >= 0.70:
                border_color = COL_GREEN
            elif det.confidence >= 0.50:
                border_color = COL_YELLOW
            else:
                border_color = COL_RED

            thickness = 3 if is_hit else 2

            x1, y1, x2, y2 = [int(v) for v in det.box_xyxy]

            # Segmentation mask (if available)
            if det.mask_polygon is not None:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [det.mask_polygon], border_color)
                cv2.addWeighted(overlay, MASK_ALPHA, frame, 1 - MASK_ALPHA, 0, frame)
                cv2.polylines(frame, [det.mask_polygon], True, border_color, 2, cv2.LINE_AA)
            else:
                # Bounding box fallback
                cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, thickness, cv2.LINE_AA)

            # Label background + text
            label_str = (
                f"{det.label} {det.confidence:.0%}" if self.caregiver_mode else det.label
            )
            (tw, th), baseline = cv2.getTextSize(label_str, FONT, 0.5, 1)
            bg_x1 = x1
            bg_y1 = max(y1 - th - 6, 0)
            bg_x2 = x1 + tw + 8
            bg_y2 = y1
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), border_color, -1)
            cv2.putText(
                frame, label_str,
                (x1 + 4, max(y1 - baseline - 2, th + 2)),
                FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )

    def _draw_gaze_fixation_ring(
        self,
        frame: np.ndarray,
        gaze: GazeSample,
        fixation: FixationState,
        hit: Optional[Detection],
        hit_probability: float = 0.0,
        hit_waypoint: Optional[dict] = None,
    ) -> None:
        from perception.visualize_neon import draw_gaze_cursor

        cx = int(round(gaze.x))
        cy = int(round(gaze.y))

        # Dim track ring
        cv2.ellipse(frame, (cx, cy), (32, 32), 0, 0, 360, (60, 60, 60), 2)

        if fixation.is_stable:
            # Full solid green ring
            cv2.ellipse(frame, (cx, cy), (32, 32), 0, 0, 360, COL_GREEN, 4, cv2.LINE_AA)
        elif fixation.active:
            progress = min(fixation.sample_count / STABLE_WINDOW_SAMPLES, 1.0)
            sweep = int(360 * progress)
            arc_color = COL_YELLOW if progress < 0.7 else COL_GREEN
            cv2.ellipse(frame, (cx, cy), (32, 32), 0, -90, -90 + sweep, arc_color, 4, cv2.LINE_AA)

        # Gaze cursor on top
        draw_gaze_cursor(frame, gaze.x, gaze.y)

        # HIT badge below cursor
        badge_bottom = cy + 44
        if hit is not None:
            badge_text = f"HIT: {hit.label} ({hit_probability:.0%})" if hit_probability > 0 else f"HIT: {hit.label}"
            (tw, th), _ = cv2.getTextSize(badge_text, FONT, 0.45, 1)
            bx = cx - tw // 2 - 4
            by = badge_bottom
            cv2.rectangle(frame, (bx, by - th - 4), (bx + tw + 8, by + 4), COL_HIT_BG, -1)
            cv2.putText(
                frame, badge_text,
                (bx + 4, by),
                FONT, 0.45, COL_WHITE, 1, cv2.LINE_AA,
            )
            badge_bottom = by + 4

        # 3D waypoint badge below HIT badge
        if hit_waypoint is not None:
            pos = hit_waypoint["position_cam"]
            wp_text = f"3D: ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:.2f})m"
            (tw2, th2), _ = cv2.getTextSize(wp_text, FONT, 0.42, 1)
            wx = cx - tw2 // 2 - 4
            wy = badge_bottom + th2 + 8
            cv2.rectangle(frame, (wx, wy - th2 - 4), (wx + tw2 + 8, wy + 4), (80, 50, 20), -1)
            cv2.putText(
                frame, wp_text,
                (wx + 4, wy),
                FONT, 0.42, ACCENT_CYAN, 1, cv2.LINE_AA,
            )

    def _draw_first_object_badge(
        self,
        frame: np.ndarray,
        first_obj: Detection,
        scale: float,
    ) -> None:
        """Draw a '1st' badge on the first locked object during AWAITING_SECOND."""
        x1, y1, x2, y2 = [int(v * scale) for v in first_obj.box_xyxy]
        # Highlight border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3, cv2.LINE_AA)
        # Badge
        badge = f"1st: {first_obj.label}"
        (tw, th), _ = cv2.getTextSize(badge, FONT, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), (0, 165, 255), -1)
        cv2.putText(frame, badge, (x1 + 4, y1 - 4), FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    # ── calibration overlays ─────────────────────────────────────────────────

    @staticmethod
    def _error_color(error_deg: float) -> tuple[int, int, int]:
        if error_deg < 1.0:
            return COL_GREEN
        elif error_deg < 3.0:
            return COL_YELLOW
        return COL_RED

    def _draw_calib_on_frame(
        self,
        frame: np.ndarray,
        gaze: GazeSample,
        tag_detection,
        sample,
        recording: bool,
        scale: float,
    ) -> None:
        """Draw AprilTag outline, error line, and calibration banner on video frame."""
        h, w = frame.shape[:2]

        if tag_detection is not None:
            corners = (tag_detection.corners * scale).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [corners], True, (0, 255, 100), 2, cv2.LINE_AA)
            tcx = int(round(tag_detection.center[0] * scale))
            tcy = int(round(tag_detection.center[1] * scale))
            cv2.drawMarker(frame, (tcx, tcy), ACCENT_CYAN, cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)

            if sample is not None:
                gx, gy = int(round(gaze.x)), int(round(gaze.y))
                col = self._error_color(sample.error_deg)
                cv2.line(frame, (gx, gy), (tcx, tcy), col, 2, cv2.LINE_AA)
                mx, my = (gx + tcx) // 2, (gy + tcy) // 2
                label = f"{sample.error_deg:.2f} deg"
                cv2.putText(frame, label, (mx + 8, my - 8), FONT, 0.55, col, 2, cv2.LINE_AA)
        else:
            cv2.putText(
                frame, "No AprilTag detected",
                (w // 2 - 160, h // 2), FONT, 0.8, COL_RED, 2, cv2.LINE_AA,
            )

        # Banner
        if recording:
            banner_col = COL_RED
            banner_text = "CALIBRATING — press 'e' to stop"
        else:
            banner_col = COL_GREEN
            banner_text = "CALIBRATED — press 's' to recalibrate"
        cv2.putText(frame, banner_text, (w // 2 - 250, 36), FONT, 0.7, banner_col, 2, cv2.LINE_AA)

    def _draw_calib_panel(
        self,
        panel: np.ndarray,
        recording: bool,
        tracker,
        tag_detection,
        sample,
        calibrated: bool,
    ) -> None:
        """Draw calibration stats on the right panel."""
        # Header
        cv2.putText(panel, "CALIBRATION", (PAD, 32), FONT, 0.75, ACCENT_CYAN, 2, cv2.LINE_AA)

        if recording:
            cv2.rectangle(panel, (PAD, 50), (PAD + 110, 72), COL_RED, -1)
            cv2.putText(panel, "RECORDING", (PAD + 4, 67), FONT, 0.45, COL_WHITE, 1, cv2.LINE_AA)
        elif calibrated:
            cv2.rectangle(panel, (PAD, 50), (PAD + 110, 72), COL_GREEN, -1)
            cv2.putText(panel, "CALIBRATED", (PAD + 4, 67), FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, "Press 's' to start", (PAD, 67), FONT, 0.45, COL_GRAY, 1, cv2.LINE_AA)

        cv2.line(panel, (0, 80), (PANEL_W, 80), BG_SECTION, 2)

        # Tag status
        if tag_detection is not None:
            cv2.circle(panel, (PAD + 8, 102), 6, COL_GREEN, -1)
            cv2.putText(panel, f"Tag #{tag_detection.tag_id} detected", (PAD + 22, 108), FONT, 0.44, COL_WHITE, 1, cv2.LINE_AA)
        else:
            cv2.circle(panel, (PAD + 8, 102), 6, COL_RED, -1)
            cv2.putText(panel, "No tag visible", (PAD + 22, 108), FONT, 0.44, COL_GRAY, 1, cv2.LINE_AA)

        if tracker is None or tracker.n_samples == 0:
            cv2.putText(panel, "No samples yet", (PAD, 150), FONT, 0.44, COL_GRAY, 1, cv2.LINE_AA)
            # Instructions
            cv2.putText(panel, "Instructions:", (PAD, 250), FONT, 0.48, COL_WHITE, 1, cv2.LINE_AA)
            cv2.putText(panel, "1. Look at the AprilTag", (PAD, 278), FONT, 0.42, COL_GRAY, 1, cv2.LINE_AA)
            cv2.putText(panel, "2. Press 's' to record", (PAD, 300), FONT, 0.42, COL_GRAY, 1, cv2.LINE_AA)
            cv2.putText(panel, "3. Press 'e' to finish", (PAD, 322), FONT, 0.42, COL_GRAY, 1, cv2.LINE_AA)
            return

        # Stats
        cv2.line(panel, (0, 124), (PANEL_W, 124), BG_SECTION, 2)
        cv2.putText(panel, "STATS", (PAD, 148), FONT, 0.44, ACCENT_CYAN, 1, cv2.LINE_AA)

        y = 176
        cv2.putText(panel, f"Samples: {tracker.n_samples}", (PAD, y), FONT, 0.46, COL_WHITE, 1, cv2.LINE_AA)
        y += 26
        cv2.putText(panel, f"Mean error: {tracker.mean_error:.2f} deg", (PAD, y), FONT, 0.46, COL_WHITE, 1, cv2.LINE_AA)
        y += 26
        cv2.putText(panel, f"Std: {tracker.std_error:.2f} deg", (PAD, y), FONT, 0.46, COL_GRAY, 1, cv2.LINE_AA)
        y += 26
        cv2.putText(panel, f"Min: {tracker.min_error:.2f}  Max: {tracker.max_error:.2f}", (PAD, y), FONT, 0.42, COL_GRAY, 1, cv2.LINE_AA)
        y += 26
        cv2.putText(panel, f"Median: {tracker.median_error:.2f}  P95: {tracker.p95_error:.2f}", (PAD, y), FONT, 0.42, COL_GRAY, 1, cv2.LINE_AA)

        if sample is not None:
            y += 32
            col = self._error_color(sample.error_deg)
            cv2.putText(panel, f"Current: {sample.error_deg:.2f} deg", (PAD, y), FONT, 0.50, col, 1, cv2.LINE_AA)

        # Offset
        dx, dy = tracker.pixel_offset
        y += 36
        cv2.line(panel, (0, y - 14), (PANEL_W, y - 14), BG_SECTION, 2)
        cv2.putText(panel, "OFFSET", (PAD, y + 4), FONT, 0.44, ACCENT_CYAN, 1, cv2.LINE_AA)
        y += 28
        cv2.putText(panel, f"dx = {dx:.1f} px", (PAD, y), FONT, 0.50, COL_WHITE, 1, cv2.LINE_AA)
        y += 24
        cv2.putText(panel, f"dy = {dy:.1f} px", (PAD, y), FONT, 0.50, COL_WHITE, 1, cv2.LINE_AA)

        # Corrected error estimate
        if tracker.n_samples >= 5:
            import numpy as _np
            corr_errors = [
                tracker.corrected_error(
                    s.gaze_px[0], s.gaze_px[1],
                    s.tag_center_px[0], s.tag_center_px[1],
                )
                for s in tracker.samples
            ]
            corr_mean = _np.mean(corr_errors)
            y += 28
            cv2.putText(panel, f"Corrected mean: {corr_mean:.2f} deg", (PAD, y), FONT, 0.46, COL_GREEN, 1, cv2.LINE_AA)

        # Key hints at bottom
        cv2.putText(panel, "'s' start  |  'e' stop  |  'q' quit", (PAD, CANVAS_H - 30), FONT, 0.38, COL_GRAY, 1, cv2.LINE_AA)

    # ── right panel ───────────────────────────────────────────────────────────

    def _draw_patient_panel(
        self,
        panel: np.ndarray,
        vlm_state: VlmState,
        last_decision: Optional[dict],
        fixation: FixationState,
    ) -> None:
        # ① Status dot (top-left, y=26)
        dot_color = COL_GREEN if (fixation.active or fixation.is_stable) else COL_GRAY
        cv2.circle(panel, (26, 26), 10, dot_color, -1, cv2.LINE_AA)

        # ② / ③ Decision area
        if vlm_state == "DECIDED" and last_decision is not None:
            d = last_decision
            if d.get("clarification_needed"):
                # Solid dark-orange card (non-pulsing)
                cv2.rectangle(panel, (12, 440), (444, 760), (20, 80, 180), -1)

                # Question text word-wrapped at 22 chars, scale 1.0, white, bold
                q = d.get("clarification_question") or ""
                lines = _wrap_text(q, max_chars=22)
                # Vertically center around y=580
                line_h = 36
                total_h = len(lines) * line_h
                text_y0 = 580 - total_h // 2 + line_h - 4
                for i, ln in enumerate(lines):
                    cv2.putText(
                        panel, ln,
                        (PAD + 4, text_y0 + i * line_h),
                        FONT, 1.0, COL_WHITE, 2, cv2.LINE_AA,
                    )
            else:
                # ③ Confirmation tick
                intent_str = f"\u2713 {d.get('intent', 'unknown')}"
                cv2.putText(
                    panel, intent_str,
                    (PAD, 580), FONT, 0.80, COL_GREEN, 2, cv2.LINE_AA,
                )

    def _draw_panel(
        self,
        panel: np.ndarray,
        detections: list[Detection],
        hit: Optional[Detection],
        fixation: FixationState,
        gaze: GazeSample,
        vlm_state: VlmState,
        last_decision: Optional[dict],
        api_mode: str,
        pending_detection: Optional[Detection],
        frame_idx: int,
        total_frames: int,
        duration_s: float,
        first_object: Optional[Detection] = None,
        second_object_countdown: float = 0.0,
    ) -> None:
        # ── Header (y=0–70) ──────────────────────────────────────────────
        cv2.putText(
            panel, "GAZE INTENT",
            (PAD, 32), FONT, 0.75, ACCENT_CYAN, 2, cv2.LINE_AA,
        )
        if total_frames > 0:
            elapsed_s = (frame_idx / total_frames) * duration_s
            time_str  = f"t={elapsed_s:.1f}s / {duration_s:.1f}s"
            frame_str = f"frame {frame_idx}/{total_frames}"
        else:                              # live mode: duration_s carries elapsed so far
            time_str  = f"t={duration_s:.1f}s  LIVE"
            frame_str = f"frame {frame_idx}"
        cv2.putText(panel, time_str, (PAD, 52), FONT, 0.44, COL_GRAY, 1, cv2.LINE_AA)
        cv2.putText(panel, frame_str, (PAD, 66), FONT, 0.40, COL_GRAY, 1, cv2.LINE_AA)

        # separator
        cv2.line(panel, (0, 70), (PANEL_W, 70), BG_SECTION, 2)

        # ── YOLO Detections (y=72–280) ────────────────────────────────────
        cv2.putText(panel, "YOLO DETECTIONS", (PAD, 90), FONT, 0.44, ACCENT_CYAN, 1, cv2.LINE_AA)

        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        visible = sorted_dets[:5]
        extra = len(sorted_dets) - 5

        for i, det in enumerate(visible):
            row_y = 100 + i * 30
            is_hit = (
                hit is not None
                and hit.label == det.label
                and hit.box_xyxy == det.box_xyxy
            )

            # Row background for hit
            if is_hit:
                cv2.rectangle(panel, (PAD - 2, row_y + 4), (PANEL_W - PAD, row_y + 28), COL_HIT_BG, -1)

            # Label text
            cv2.putText(
                panel, det.label,
                (PAD + 2, row_y + 20), FONT, 0.48, COL_WHITE, 1, cv2.LINE_AA,
            )

            # Confidence bar (80 px wide) at right side
            bar_x = PANEL_W - PAD - 80
            bar_y = row_y + 10
            bar_h = 12
            # background
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x + 80, bar_y + bar_h), BG_SECTION, -1)
            # fill
            if det.confidence >= 0.70:
                bar_col = COL_GREEN
            elif det.confidence >= 0.50:
                bar_col = COL_YELLOW
            else:
                bar_col = COL_RED
            fill_w = int(80 * det.confidence)
            if fill_w > 0:
                cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_col, -1)
            # percentage text
            pct_str = f"{det.confidence:.0%}"
            (pw, _), _ = cv2.getTextSize(pct_str, FONT, 0.38, 1)
            cv2.putText(
                panel, pct_str,
                (bar_x + 80 - pw - 2, bar_y + bar_h - 1), FONT, 0.38, COL_GRAY, 1, cv2.LINE_AA,
            )

        if extra > 0:
            ev_y = 100 + 5 * 30
            cv2.putText(
                panel, f"+{extra} more",
                (PAD, ev_y + 16), FONT, 0.42, COL_GRAY, 1, cv2.LINE_AA,
            )

        if not detections:
            cv2.putText(panel, "none", (PAD, 118), FONT, 0.44, COL_GRAY, 1, cv2.LINE_AA)

        # separator
        cv2.line(panel, (0, 280), (PANEL_W, 280), BG_SECTION, 2)

        # ── FIXATION (y=282–400) ─────────────────────────────────────────
        cv2.putText(panel, "FIXATION", (PAD, 302), FONT, 0.44, ACCENT_CYAN, 1, cv2.LINE_AA)

        # Mini arc icon at (55, 350) r=18
        arc_cx, arc_cy = 55, 350
        cv2.ellipse(panel, (arc_cx, arc_cy), (18, 18), 0, 0, 360, (60, 60, 60), 2)
        if fixation.is_stable:
            cv2.ellipse(panel, (arc_cx, arc_cy), (18, 18), 0, 0, 360, COL_GREEN, 3, cv2.LINE_AA)
        elif fixation.active:
            progress = min(fixation.sample_count / STABLE_WINDOW_SAMPLES, 1.0)
            sweep = int(360 * progress)
            arc_color = COL_YELLOW if progress < 0.7 else COL_GREEN
            cv2.ellipse(panel, (arc_cx, arc_cy), (18, 18), 0, -90, -90 + sweep, arc_color, 3, cv2.LINE_AA)

        # Status text
        if fixation.is_stable:
            status_str = "STABLE"
            status_col = COL_GREEN
        elif fixation.active:
            status_str = "ACTIVE"
            status_col = COL_YELLOW
        else:
            status_str = "INACTIVE"
            status_col = COL_GRAY

        cv2.putText(panel, status_str, (82, 342), FONT, 0.52, status_col, 1, cv2.LINE_AA)

        if fixation.active:
            dur_ms = fixation.duration_ns / 1_000_000
            cv2.putText(
                panel, f"dur: {dur_ms:.0f} ms",
                (82, 360), FONT, 0.42, COL_GRAY, 1, cv2.LINE_AA,
            )
            cv2.putText(
                panel, f"samples: {fixation.sample_count}",
                (82, 376), FONT, 0.42, COL_GRAY, 1, cv2.LINE_AA,
            )

        if fixation.is_stable:
            # STABLE badge
            cv2.rectangle(panel, (PAD, 384), (PAD + 72, 400), COL_GREEN, -1)
            cv2.putText(panel, "STABLE", (PAD + 4, 397), FONT, 0.38, (0, 0, 0), 1, cv2.LINE_AA)

        # separator
        cv2.line(panel, (0, 400), (PANEL_W, 400), BG_SECTION, 2)

        # ── VLM section header (y=402–432) ─────────────────────────────
        cv2.putText(panel, "VLM / GPT-4o", (PAD, 424), FONT, 0.50, ACCENT_CYAN, 1, cv2.LINE_AA)

        # separator
        cv2.line(panel, (0, 432), (PANEL_W, 432), BG_SECTION, 2)

        # ── VLM content (y=432–1190) ─────────────────────────────────────
        if vlm_state == "IDLE":
            cv2.putText(
                panel,
                "Waiting for stable fixation...",
                (PAD, 480), FONT, 0.44, COL_GRAY, 1, cv2.LINE_AA,
            )

        elif vlm_state == "AWAITING_SECOND":
            # Orange banner
            cv2.rectangle(panel, (0, 448), (PANEL_W, 540), (0, 120, 200), -1)
            cv2.putText(
                panel, "AWAITING 2ND OBJECT",
                (PAD, 476), FONT, 0.58, (255, 255, 255), 2, cv2.LINE_AA,
            )
            if first_object is not None:
                cv2.putText(
                    panel, f"1st: {first_object.label}",
                    (PAD, 504), FONT, 0.50, (255, 255, 255), 1, cv2.LINE_AA,
                )
            if second_object_countdown > 0:
                cv2.putText(
                    panel, f"Timeout: {second_object_countdown:.1f}s",
                    (PAD, 526), FONT, 0.44, (200, 200, 200), 1, cv2.LINE_AA,
                )
            cv2.putText(
                panel, "Look at another object...",
                (PAD, 560), FONT, 0.44, COL_GRAY, 1, cv2.LINE_AA,
            )

        elif vlm_state == "THINKING":
            spinner_char = SPINNER_ASCII[self._spinner_idx]
            cv2.putText(
                panel, spinner_char,
                (PAD, 480), FONT, 0.9, COL_SPINNER, 2, cv2.LINE_AA,
            )
            cv2.putText(
                panel, "Querying GPT-4o",
                (PAD + 28, 480), FONT, 0.52, COL_WHITE, 1, cv2.LINE_AA,
            )

            # Mode badge
            if api_mode == "TEXT":
                badge_fill = (30, 80, 30)
            else:
                badge_fill = (100, 60, 20)
            cv2.rectangle(panel, (PAD, 500), (PAD + 80, 524), badge_fill, -1)
            cv2.putText(panel, api_mode, (PAD + 4, 518), FONT, 0.42, COL_WHITE, 1, cv2.LINE_AA)

            # Prompt preview
            if api_mode == "TEXT" and pending_detection is not None:
                preview = f"Fixating: {pending_detection.label} ({pending_detection.confidence:.0%})"
            else:
                preview = f"Vision query @ ({gaze.x:.0f}, {gaze.y:.0f})"
            cv2.putText(
                panel, preview,
                (PAD, 540), FONT, 0.44, COL_GRAY, 1, cv2.LINE_AA,
            )

        elif vlm_state == "DECIDED" and last_decision is not None:
            d = last_decision

            # Card background
            cv2.rectangle(panel, (12, 448), (444, 780), BG_CARD, -1)

            # Clarification banner (full-width, pulsing)
            card_y = 476
            if d.get("clarification_needed"):
                CLARIFY_BG     = (30, 100, 220)    # BGR deep orange-red (bright)
                CLARIFY_BG_DIM = (20,  70, 160)    # dimmed variant
                bg = CLARIFY_BG if (frame_idx // 15) % 2 == 0 else CLARIFY_BG_DIM
                cv2.rectangle(panel, (0, 448), (PANEL_W, 540), bg, -1)

                # Large bold header
                cv2.putText(panel, "? CLARIFICATION NEEDED",
                            (PAD, 476), FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

                # Separator inside banner
                cv2.line(panel, (PAD, 484), (PANEL_W - PAD, 484), (255, 255, 255), 1)

                # Clarification question text — word-wrapped, white, scale 0.50
                q = d.get("clarification_question") or ""
                for li, ln in enumerate(_wrap_text(q, max_chars=42)[:2]):
                    cv2.putText(panel, ln,
                                (PAD, 504 + li * 20), FONT, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

                card_y = 556   # shift intent card below the banner

            # Intent text
            intent_str = str(d.get("intent", "unknown"))
            cv2.putText(
                panel, intent_str,
                (PAD + 4, card_y), FONT, 0.75, COL_WHITE, 2, cv2.LINE_AA,
            )

            # Object(s)
            obj_str = f"Object: {d.get('object') or 'n/a'}"
            cv2.putText(
                panel, obj_str,
                (PAD + 4, card_y + 28), FONT, 0.52, COL_GRAY, 1, cv2.LINE_AA,
            )
            if d.get("second_object"):
                cv2.putText(
                    panel, f"2nd: {d['second_object']}",
                    (PAD + 4, card_y + 50), FONT, 0.52, COL_GRAY, 1, cv2.LINE_AA,
                )

            # Confidence bar
            conf = float(d.get("confidence", 0.0))
            bar_y0 = card_y + 44
            cv2.rectangle(panel, (PAD + 4, bar_y0), (440, bar_y0 + 14), BG_SECTION, -1)
            fill_w = int(432 * conf)
            if fill_w > 0:
                cv2.rectangle(panel, (PAD + 4, bar_y0), (PAD + 4 + fill_w, bar_y0 + 14), COL_GREEN, -1)
            cv2.putText(
                panel, f"{conf:.0%}",
                (PAD + 4, bar_y0 + 12), FONT, 0.40, COL_WHITE, 1, cv2.LINE_AA,
            )

            # Separator
            sep_y = card_y + 70
            cv2.line(panel, (PAD + 4, sep_y), (440, sep_y), BG_SECTION, 1)

            # Reasoning header
            cv2.putText(
                panel, "REASONING",
                (PAD + 4, sep_y + 18), FONT, 0.44, ACCENT_CYAN, 1, cv2.LINE_AA,
            )

            # Word-wrapped reasoning
            reasoning = str(d.get("reasoning", ""))
            lines = _wrap_text(reasoning, max_chars=46)
            text_y = sep_y + 38
            for i, line in enumerate(lines[:18]):
                cv2.putText(
                    panel, line,
                    (PAD + 4, text_y + i * 20), FONT, 0.45, COL_GRAY, 1, cv2.LINE_AA,
                )


# ── helpers ───────────────────────────────────────────────────────────────────

def _scale_detection(det: Detection, s: float) -> Detection:
    """Return a copy of *det* with all spatial fields scaled by *s*."""
    x1, y1, x2, y2 = det.box_xyxy
    cx, cy = det.box_center
    mask = None
    if det.mask_polygon is not None:
        mask = (det.mask_polygon.astype(np.float64) * s).astype(np.int32)
    return Detection(
        label=det.label,
        confidence=det.confidence,
        box_xyxy=(x1 * s, y1 * s, x2 * s, y2 * s),
        box_center=(cx * s, cy * s),
        mask_polygon=mask,
    )

def _wrap_text(text: str, max_chars: int = 46) -> list[str]:
    """Greedy word-wrap: split text into lines of at most max_chars characters."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_chars:
            current += " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines
