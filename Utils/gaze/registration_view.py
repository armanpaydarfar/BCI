"""
registration_view.py — live quality view for the 3-D world-map registration
(WS-4, accuracy-roadmap).

The 3-D registration capture used to run blind: a fixed-duration loop grabbed
frames silently and the operator only learned the per-tag reproducibility AFTER
the fuse (``apriltag_calibrate.py`` register-world-3d). This view closes that
loop — during capture it shows, per tag, the SAME quality signals the final
verdict gates on, so the operator can see which tags are still weak and keep
moving until they go green, then accept.

Two signals per tag, because each catches a different failure:

  - **reproducibility** — the per-tag split-half disagreement from
    :func:`Utils.gaze.apriltag_world_3d.world_map_3d_reproducibility`: the map is
    fused from two independent halves of the frames and the distance between the
    two estimates of each tag is the honest, ground-truth-free measure of map
    ACCURACY. This is what to gate on — NOT the per-frame scatter
    (``world_map_3d_geometry_report``), which measures raw AprilTag sensor noise
    (~tens of mm), floors out, and barely improves no matter how the operator
    moves. The fuse averages that scatter down; reproducibility is what survives.
  - **viewpoint diversity** — the cone half-angle of the directions the tag has
    been seen from. Averaging only decorrelates the per-view depth bias if the
    views are geometrically spread, so diversity is the prerequisite that makes
    the fuse trustworthy — and the signal that tells the operator *where to move*.

The quality math (:func:`cone_half_angle_deg`, :func:`classify_tags`,
:func:`registration_summary`) is pure and unit-tested without a display; the cv2
draw + window handling in :class:`RegistrationView` is the thin rig-only part,
mirroring ``coverage_view.CoverageBoxUI``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

# First-pass at-rig thresholds (accuracy-roadmap 2026-06-29). A tag is GOOD only
# when it has enough views AND enough angular spread AND a low split-half
# reproducibility; OK relaxes diversity/reproducibility to "usable"; anything seen
# but short of OK is WEAK ("keep working it"). The reproducibility bars are on the
# split-half disagreement, which over-estimates the full-map error ~1.3× (synthetic
# check: 15 mm disagreement ≈ 11 mm true), so ≤15 mm GOOD ≈ ≤~12 mm map error.
MIN_VIEWS = 30            # ≈1–2 s of continuous detection at relay rate
MIN_DIVERSITY_DEG = 12.0  # below this the tag was seen from ~one direction
GOOD_DIVERSITY_DEG = 25.0
OK_REPRO_MM = 30.0
GOOD_REPRO_MM = 15.0

# BGR (OpenCV) state colours, shared by the chips and the on-scene tag markers.
_COLOR_STATE = {
    "unseen": (90, 90, 90),     # grey — not detected yet
    "weak": (60, 60, 235),      # red — needs more views / angles / not yet reproducible
    "ok": (40, 140, 235),       # amber — usable, not yet good
    "good": (80, 185, 90),      # green — diverse views + reproducible position
}
_COLOR_BG = (24, 24, 24)
_COLOR_TEXT = (245, 245, 245)
_COLOR_PANEL = (38, 38, 38)


@dataclass
class TagQuality:
    """Live quality of one world tag during registration."""
    tag_id: int
    views: int
    repro_mm: Optional[float]      # split-half disagreement; None until it can be computed
    diversity_deg: float
    state: str                     # unseen | weak | ok | good


def cone_half_angle_deg(bearings: Sequence[np.ndarray]) -> float:
    """Angular spread of a tag's viewing directions as a cone half-angle (deg).

    ``bearings`` are the per-view directions from the camera to the tag (the tag's
    translation in the camera frame, any scale). Returns the largest angle between
    any bearing and their mean direction — a single number for "how much parallax
    did this tag get". 0 for <2 views; up to 180 if the views straddle the tag."""
    if len(bearings) < 2:
        return 0.0
    B = np.asarray(bearings, dtype=float)
    norms = np.linalg.norm(B, axis=1, keepdims=True)
    B = B / np.clip(norms, 1e-9, None)
    mean = B.mean(axis=0)
    n = float(np.linalg.norm(mean))
    if n < 1e-9:                       # views cancel out → maximally spread
        return 180.0
    mean = mean / n
    dots = np.clip(B @ mean, -1.0, 1.0)
    return float(np.degrees(np.arccos(dots.min())))


def classify_tags(world_ids: Sequence[int],
                  views: Mapping[int, int],
                  repro: Optional[Mapping[int, float]],
                  diversity: Mapping[int, float],
                  *,
                  min_views: int = MIN_VIEWS,
                  min_diversity_deg: float = MIN_DIVERSITY_DEG,
                  good_diversity_deg: float = GOOD_DIVERSITY_DEG,
                  ok_repro_mm: float = OK_REPRO_MM,
                  good_repro_mm: float = GOOD_REPRO_MM) -> Dict[int, TagQuality]:
    """Classify every world tag into unseen/weak/ok/good from the live counters.

    ``repro`` is the per-tag split-half disagreement from
    ``world_map_3d_reproducibility`` (or None before enough frames to compute it).
    A tag is GOOD only when it clears all three bars (views, diversity,
    reproducibility); OK when it has the views and a usable reproducibility; WEAK
    when seen but short of that; UNSEEN when never detected."""
    out: Dict[int, TagQuality] = {}
    rp = repro or {}
    for tid in world_ids:
        tid = int(tid)
        v = int(views.get(tid, 0))
        r = rp.get(tid)
        d = float(diversity.get(tid, 0.0))
        if v == 0:
            state = "unseen"
        elif (v >= min_views and d >= good_diversity_deg
              and r is not None and r <= good_repro_mm):
            state = "good"
        elif (v >= min_views and d >= min_diversity_deg
              and r is not None and r <= ok_repro_mm):
            state = "ok"
        else:
            state = "weak"
        out[tid] = TagQuality(tid, v, r, d, state)
    return out


def registration_summary(qualities: Mapping[int, TagQuality]) -> Dict:
    """Roll the per-tag states into an overall verdict + a "where to move" hint.

    ``all_good`` is the accept condition (every tag good); ``weak_ids`` is the
    operator's worklist (unseen + weak tags — point your head at these from new
    angles)."""
    rp = [q.repro_mm for q in qualities.values()
          if q.repro_mm is not None and q.state != "unseen"]
    n_good = sum(1 for q in qualities.values() if q.state == "good")
    n_total = len(qualities)
    weak = sorted(q.tag_id for q in qualities.values()
                  if q.state in ("unseen", "weak"))
    return {
        "n_good": n_good,
        "n_total": n_total,
        "mean_repro_mm": float(np.mean(rp)) if rp else float("nan"),
        "max_repro_mm": float(np.max(rp)) if rp else float("nan"),
        "weak_ids": weak,
        "all_good": n_total > 0 and n_good == n_total,
    }


class RegistrationView:
    """OpenCV live registration view. Left: the scene with each detected tag marked
    in its quality colour (so the operator sees WHICH tag in context). Right: a
    per-tag quality panel + overall meter + a "more angles" worklist.

    The operator drives termination: SPACE accepts (finish capture, build the map),
    'q' aborts. ``finished()`` / ``aborted()`` report which happened; ``prompt`` is
    the pre-capture start gate. Mirrors ``CoverageBoxUI``'s thin-wrapper pattern."""

    def __init__(self, world_ids: Sequence[int], *,
                 width: int = 1120, height: int = 620, panel_w: int = 360,
                 window: str = "AprilTag registration (3-D)") -> None:
        import cv2  # lazy: only the display path needs it
        self._cv2 = cv2
        self.world_ids = [int(i) for i in world_ids]
        self.width, self.height, self.panel_w = width, height, panel_w
        self.window = window
        self._finished = False
        self._aborted = False
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, width, height)

    def prompt(self, title: str, sublines: Sequence[str]) -> bool:
        """Pre-capture start gate: block until SPACE (True) or 'q'/closed (False)."""
        cv2 = self._cv2
        rows = [(title, 0.95, 2)] + [(s, 0.6, 1) for s in sublines]
        while True:
            canvas = np.full((self.height, self.width, 3), _COLOR_BG, dtype=np.uint8)
            for k, (text, scale, thick) in enumerate(rows):
                y = self.height // 2 - 28 + k * 36
                cv2.putText(canvas, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                            _COLOR_TEXT, thick, cv2.LINE_AA)
            cv2.imshow(self.window, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == ord(" "):
                # Fresh capture starts here — clear any verdict from a prior attempt
                # so a missing-tag retry isn't ended instantly by the last SPACE.
                self._finished = False
                self._aborted = False
                return True
            if key == ord("q"):
                self._aborted = True
                return False
            if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                self._aborted = True
                return False

    def update(self, frame_bgr: Optional[np.ndarray],
               detections: Mapping[int, Dict],
               qualities: Mapping[int, TagQuality],
               summary: Dict) -> None:
        """Redraw one frame of the live view and poll for SPACE (finish) / 'q' (abort)."""
        cv2 = self._cv2
        canvas = np.full((self.height, self.width, 3), _COLOR_BG, dtype=np.uint8)
        scene_w = self.width - self.panel_w
        self._draw_scene(canvas, frame_bgr, detections, qualities, scene_w)
        self._draw_panel(canvas, qualities, summary, scene_w)

        cv2.imshow(self.window, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            self._finished = True
        elif key == ord("q"):
            self._aborted = True
        elif cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
            self._aborted = True

    def _draw_scene(self, canvas, frame_bgr, detections, qualities, scene_w) -> None:
        cv2 = self._cv2
        if frame_bgr is None or frame_bgr.size == 0:
            cv2.putText(canvas, "waiting for frames…", (40, self.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, _COLOR_TEXT, 2, cv2.LINE_AA)
            return
        fh, fw = frame_bgr.shape[:2]
        scale = min(scene_w / fw, self.height / fh)
        dw, dh = int(fw * scale), int(fh * scale)
        ox, oy = (scene_w - dw) // 2, (self.height - dh) // 2
        canvas[oy:oy + dh, ox:ox + dw] = cv2.resize(frame_bgr, (dw, dh))
        # Mark each detected tag at its scene centre in its quality colour so the
        # operator sees, in context, which tag still needs work.
        for tid, det in detections.items():
            tid = int(tid)
            q = qualities.get(tid)
            c = det.get("center")
            if c is None or q is None:   # only mark mapped world tags (skip e.g. the EE tag)
                continue
            px, py = int(ox + float(c[0]) * scale), int(oy + float(c[1]) * scale)
            color = _COLOR_STATE[q.state]
            cv2.circle(canvas, (px, py), 11, color, 2, cv2.LINE_AA)
            cv2.putText(canvas, str(tid), (px + 13, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    def _draw_panel(self, canvas, qualities, summary, x0) -> None:
        cv2 = self._cv2
        cv2.rectangle(canvas, (x0, 0), (self.width, self.height), _COLOR_PANEL, -1)
        x = x0 + 14
        cv2.putText(canvas, "registration quality", (x, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, _COLOR_TEXT, 1, cv2.LINE_AA)
        y = 60
        for tid in self.world_ids:
            q = qualities.get(tid)
            state = q.state if q else "unseen"
            color = _COLOR_STATE[state]
            cv2.rectangle(canvas, (x, y - 12), (x + 16, y + 4), color, -1)
            r = "--" if (q is None or q.repro_mm is None) else f"{q.repro_mm:3.0f}"
            d = 0.0 if q is None else q.diversity_deg
            v = 0 if q is None else q.views
            # q = reproducibility (split-half disagreement, the gate); d = viewpoint
            # diversity (deg); v = views.
            cv2.putText(canvas, f"id {tid:>2} v{v:>4} q{r}mm d{d:3.0f}",
                        (x + 24, y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        cv2.LINE_AA)
            y += 26

        y = self.height - 118
        cv2.putText(canvas, f"good {summary['n_good']}/{summary['n_total']}   "
                    f"repro mean {summary['mean_repro_mm']:.0f}  "
                    f"max {summary['max_repro_mm']:.0f} mm",
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, _COLOR_TEXT, 1, cv2.LINE_AA)
        # Scatter is the raw per-frame sensor noise (floors out, NOT the gate) —
        # shown dim so a high value doesn't read as failure.
        scatter = summary.get("scatter_mm")
        if scatter is not None and np.isfinite(scatter):
            cv2.putText(canvas, f"(scatter ~{scatter:.0f} mm — sensor floor, not the gate)",
                        (x, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1,
                        cv2.LINE_AA)
        if summary["all_good"]:
            hint = "ALL GOOD — SPACE to accept"
        else:
            weak = " ".join(str(i) for i in summary["weak_ids"][:10])
            hint = f"more angles on: {weak}" if weak else "keep moving…"
        cv2.putText(canvas, hint, (x, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    _COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas, "[SPACE] accept & save   [q] abort", (x, y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR_TEXT, 1, cv2.LINE_AA)

    def finished(self) -> bool:
        return self._finished

    def aborted(self) -> bool:
        return self._aborted

    def close(self) -> None:
        try:
            self._cv2.destroyWindow(self.window)
        except Exception:
            # Best-effort teardown on a bench tool already exiting; the window may
            # already be gone (operator closed it).
            pass
