"""
scene_overlay_renderer.py — Linux-side scene + overlay renderer.

Composites a Pupil Labs Neon scene frame with the JSON detection / gaze
payload pushed by the Windows-hosted vlm_service.py, so the operator panel
can paint at native frame rate without round-tripping pixels to Windows
and back. See ``Documents/SoftwareDocs/projects/harmony-bci/gpu-service/render-layer-refactor-plan.md``
§4 for design context.

**Implementation choice (§4.1).** Option B — slim reimplementation, NOT an
import of ``harmony_vlm.utils.overlay_renderer.OverlayRenderer``. Reasons:

  - The natural input contract here is the JSON push payload defined in
    §3, not native Detection/FixationState/GazeSample objects from the
    sister repo. Importing OverlayRenderer would force us to round-trip
    JSON dicts through native-type reconstruction just to be picked
    apart again inside the renderer.
  - harmony_vlm's renderer paints a 460-px-wide right panel (text,
    spinner, calibration UI). Useful when the renderer is the entire
    UI — but the BCI panel already shows status via its own Qt labels
    and tabs, so the side panel would be wasted pixels and wasted
    render budget.
  - The Linux panel needs ≥25 fps (plan §7 acceptance #3). Skipping the
    right-panel composite buys headroom.

Drift risk is bounded because ``harmony_vlm/utils/overlay_renderer.py``
is read-only reference material per CLAUDE.md sister-repo rule; new
features there can be ported here on demand if they ever matter for
production.

Inputs are explicitly JSON-shaped dicts (see ``render``'s docstring),
matching the wire payload defined in
``Documents/SoftwareDocs/projects/harmony-bci/gpu-service/render-layer-refactor-plan.md`` §3.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ── colour palette (BGR) — chosen to match harmony_vlm's overlay so the
# BCI-rendered scene is visually consistent with the Windows-side debug
# windows operators may also be looking at. ────────────────────────────────

COL_HIGH_CONF = (50, 220, 80)    # green   ≥ 0.70
COL_MED_CONF  = (30, 210, 230)   # yellow  ≥ 0.50
COL_LOW_CONF  = (50, 50, 220)    # red     <  0.50

COL_GAZE_RING_DIM     = (60, 60, 60)
COL_GAZE_RING_FIX     = (80, 200, 255)   # cyan when fixating
COL_GAZE_RING_STABLE  = (50, 220, 80)    # green when stable
COL_GAZE_DOT          = (240, 240, 240)

COL_HIT_TEXT_BG = (60, 100, 30)
COL_DECISION_BG = (40, 40, 40)
COL_DECISION_FG = (240, 240, 240)

# gaze_runner tracks (YOLO + SORT). Painted thinner / cooler-coloured
# than VLM detections so the two sets coexist without visual collisions
# when both backends run. BGR values.
COL_TRACK     = (220, 180, 60)
COL_TRACK_HIT = (240, 220, 80)

FONT = cv2.FONT_HERSHEY_SIMPLEX
MASK_ALPHA = 0.35


def _conf_color(score: float) -> Tuple[int, int, int]:
    if score >= 0.70:
        return COL_HIGH_CONF
    if score >= 0.50:
        return COL_MED_CONF
    return COL_LOW_CONF


class SceneOverlayRenderer:
    """In-place compositor: scene BGR → annotated BGR.

    No state across frames except a frame counter for diagnostic overlays.
    Cheap to construct; one instance per panel widget is enough.
    """

    def __init__(self) -> None:
        self._frame_counter: int = 0

    # ── public API ────────────────────────────────────────────────────────

    def render(
        self,
        frame_bgr: np.ndarray,
        *,
        gaze_xy: Optional[Tuple[float, float]] = None,
        detections: Optional[Sequence[Dict[str, Any]]] = None,
        hit_det_id: Optional[int] = None,
        fixation: Optional[Dict[str, Any]] = None,
        tracks: Optional[Sequence[Dict[str, Any]]] = None,
        current_hit_track_id: Optional[int] = None,
        vlm_state: str = "IDLE",
        decision_text: Optional[str] = None,
        copy: bool = True,
    ) -> np.ndarray:
        """Paint detections + gaze + state badge onto ``frame_bgr``.

        Parameters
        ----------
        frame_bgr
            Scene image. Whatever shape Neon produced (typically 1600×1200);
            we don't resize.
        gaze_xy
            Pixel-space gaze position from the *latest bundle* (not from the
            JSON push). This is the field that makes the cursor responsive
            — render every paint pass even when JSON detections are stale.
        detections
            List of dicts shaped like Render_Layer_Refactor.md §3:
            ``{label, confidence, box_xyxy=[x1,y1,x2,y2], box_center, mask_polygon?}``.
            ``mask_polygon`` is an int-quantised flat list of ``[x,y]`` pairs
            if present.
        hit_det_id
            Index into ``detections`` of the box currently under gaze. The
            JSON push delivers this as ``hit.det_id`` (-1 / None when
            nothing is hit). Drawn with a thicker border.
        fixation
            ``{active: bool, duration_ms: float, x?, y?, stable?: bool}``.
            ``None`` means no fixation; render a dim track ring.
        vlm_state
            One of ``IDLE``/``SEGMENTING``/``REASONING``/``THINKING``/
            ``AWAITING_SECOND``/``DECIDED`` — drawn as a small badge in the
            top-left corner of the frame.
        decision_text
            Optional short string from the latest decide call (e.g. the
            object label or one-line summary). Drawn under the state badge.
        copy
            If True (default), allocate a fresh BGR array and return it,
            leaving the caller's frame untouched. False means draw in-place
            and return the same array — useful when the caller already
            holds an ephemeral copy of the bundle.

        Returns
        -------
        BGR image of the same shape as ``frame_bgr`` with overlays drawn.
        """
        canvas = frame_bgr.copy() if copy else frame_bgr

        # Detections first so the gaze cursor and badges paint over them.
        if detections:
            self._draw_detections(canvas, detections, hit_det_id)

        # Gaze runner tracks (separate visual style — thin stroke + id).
        # These come from gaze_runner's `gaze_results` push and represent
        # YOLO + SORT outputs; they coexist with VLM segmentation masks
        # when both backends are running.
        if tracks:
            self._draw_tracks(canvas, tracks, current_hit_track_id)

        # Gaze cursor + fixation ring. Drawn from `gaze_xy` (which the panel
        # pulls from the freshest bundle), not from any field inside the
        # JSON push — that's the whole point of the refactor.
        if gaze_xy is not None:
            self._draw_gaze(canvas, gaze_xy, fixation)

        self._draw_state_badge(canvas, vlm_state, decision_text)
        self._frame_counter += 1
        return canvas

    # ── private drawing ───────────────────────────────────────────────────

    def _draw_detections(
        self,
        canvas: np.ndarray,
        detections: Sequence[Dict[str, Any]],
        hit_det_id: Optional[int],
    ) -> None:
        h, w = canvas.shape[:2]

        # Single-pass blend: fill all polygon masks into one shared overlay
        # buffer, blend once, then draw outlines/labels on top. The prior
        # per-detection canvas.copy()+addWeighted made paint cost O(N) in
        # full-canvas memory traffic — at N≈40 under bright lighting this
        # blew the 33 ms paint budget by 7×. Outlines stay per-detection
        # (vector-bound, cheap). Side effect at mask overlaps: last fillPoly
        # wins instead of compounding alpha — typically reads as cleaner.
        overlay: Optional[np.ndarray] = None
        poly_outlines: list = []
        rect_calls: list = []
        label_calls: list = []

        for idx, det in enumerate(detections):
            box = det.get("box_xyxy") or []
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))
            score = float(det.get("confidence", 0.0))
            color = _conf_color(score)
            is_hit = (hit_det_id is not None and hit_det_id == idx)
            thickness = 3 if is_hit else 2

            poly = det.get("mask_polygon")
            if poly:
                try:
                    pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
                    if overlay is None:
                        overlay = canvas.copy()
                    cv2.fillPoly(overlay, [pts], color)
                    poly_outlines.append((pts, color))
                except (TypeError, ValueError):
                    rect_calls.append((x1, y1, x2, y2, color, thickness))
            else:
                rect_calls.append((x1, y1, x2, y2, color, thickness))

            label = str(det.get("label") or "")
            if label:
                label_calls.append((label, (x1, y1), color))

        if overlay is not None:
            cv2.addWeighted(overlay, MASK_ALPHA, canvas, 1 - MASK_ALPHA, 0, canvas)

        for pts, color in poly_outlines:
            cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)
        for x1, y1, x2, y2, color, thickness in rect_calls:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        for label, (x1, y1), color in label_calls:
            self._draw_label(canvas, label, (x1, y1), color)

    def _draw_tracks(
        self,
        canvas: np.ndarray,
        tracks: Sequence[Dict[str, Any]],
        current_hit_track_id: Optional[int],
    ) -> None:
        """Draw gaze_runner.py tracks (YOLO + SORT). Each entry follows
        the gaze_results.tracks schema in Render_Layer_Refactor.md §3:
        ``{id, bbox=[x1,y1,x2,y2], label, score, age?, lost?}``.

        Hit track is drawn with a thicker stroke; everything else uses a
        single thin cyan stroke so VLM masks underneath remain visible.
        """
        h, w = canvas.shape[:2]
        for tr in tracks:
            box = tr.get("bbox") or []
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
            tid = int(tr.get("id", -1))
            is_hit = (current_hit_track_id is not None and tid == current_hit_track_id)
            color = COL_TRACK_HIT if is_hit else COL_TRACK
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2 if is_hit else 1, cv2.LINE_AA)
            label_str = f"{tr.get('label', '?')}#{tid}"
            score = tr.get("score")
            if isinstance(score, (int, float)):
                label_str += f" {float(score):.2f}"
            self._draw_label(canvas, label_str, (x1, y2 + 18), color)

    @staticmethod
    def _draw_label(
        canvas: np.ndarray,
        text: str,
        anchor_xy: Tuple[int, int],
        bg_color: Tuple[int, int, int],
    ) -> None:
        x, y = anchor_xy
        (tw, th), baseline = cv2.getTextSize(text, FONT, 0.5, 1)
        bg_y1 = max(y - th - 6, 0)
        cv2.rectangle(canvas, (x, bg_y1), (x + tw + 8, y), bg_color, -1)
        cv2.putText(
            canvas, text,
            (x + 4, max(y - baseline - 2, th + 2)),
            FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )

    def _draw_gaze(
        self,
        canvas: np.ndarray,
        gaze_xy: Tuple[float, float],
        fixation: Optional[Dict[str, Any]],
    ) -> None:
        h, w = canvas.shape[:2]
        gx, gy = gaze_xy
        if not (np.isfinite(gx) and np.isfinite(gy)):
            return
        cx = int(round(gx))
        cy = int(round(gy))
        if not (0 <= cx < w and 0 <= cy < h):
            return

        # Dim track ring always; bright ring + radius on fixation.
        cv2.ellipse(canvas, (cx, cy), (32, 32), 0, 0, 360, COL_GAZE_RING_DIM, 2)
        if fixation and fixation.get("active"):
            stable = bool(fixation.get("stable", False))
            ring_color = COL_GAZE_RING_STABLE if stable else COL_GAZE_RING_FIX
            cv2.ellipse(canvas, (cx, cy), (38, 38), 0, 0, 360, ring_color, 3, cv2.LINE_AA)

        # Crosshair + centre dot. Small, white, drawn over everything.
        cv2.line(canvas, (cx - 12, cy), (cx + 12, cy), COL_GAZE_DOT, 1, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy - 12), (cx, cy + 12), COL_GAZE_DOT, 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 3, COL_GAZE_DOT, -1, cv2.LINE_AA)

    def _draw_state_badge(
        self,
        canvas: np.ndarray,
        state: str,
        decision_text: Optional[str],
    ) -> None:
        # Top-left badge (stack: state on top, optional decision text below).
        # Kept compact so it doesn't compete with detection labels.
        lines: List[Tuple[str, Tuple[int, int, int]]] = [(f"VLM {state}", (240, 240, 240))]
        if decision_text:
            t = decision_text.strip().splitlines()[0] if decision_text.strip() else ""
            if t:
                lines.append((t[:80], (200, 230, 255)))
        x, y = 16, 28
        for text, fg in lines:
            (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
            cv2.rectangle(canvas, (x - 6, y - th - 6),
                          (x + tw + 6, y + 6), COL_DECISION_BG, -1)
            cv2.putText(canvas, text, (x, y), FONT, 0.55, fg, 1, cv2.LINE_AA)
            y += th + 12
