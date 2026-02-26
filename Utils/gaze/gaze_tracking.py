# Utils/gaze/gaze_tracking.py
"""
gaze_tracking.py

Box geometry + IoU + a lightweight SORT-style tracker (Kalman-based).

Rules:
- NO device I/O
- NO threading
- NO Ultralytics/YOLO
- NO OpenCV
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# -----------------------
# Box utilities
# -----------------------
def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _box_area(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(max(1.0, (x2 - x1) * (y2 - y1)))


def _dist_point_to_box(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    cx = _clamp(px, x1, x2)
    cy = _clamp(py, y1, y2)
    return float(np.hypot(px - cx, py - cy))


def box_center_xy(xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = map(float, xyxy)
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def box_wh(xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = map(float, xyxy)
    return max(1.0, x2 - x1), max(1.0, y2 - y1)


def size_similarity_ratio(
    boxA: Tuple[float, float, float, float],
    boxB: Tuple[float, float, float, float],
) -> float:
    w1, h1 = box_wh(boxA)
    w2, h2 = box_wh(boxB)
    rw = min(w1, w2) / max(w1, w2)
    rh = min(h1, h2) / max(h1, h2)
    return float(rw * rh)


def iou_xyxy(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return float(inter / (area_a + area_b - inter))


# -----------------------
# Kalman box tracker
# -----------------------
@dataclass
class TrackerParams:
    # Matching
    match_iou: float = 0.22
    max_age_sec: float = 1.25
    min_hits: int = 2
    ema_alpha: float = 0.28

    # Kalman noise
    process_noise_pos: float = 7.0
    process_noise_vel: float = 12.0
    meas_noise_pos: float = 16.0
    meas_noise_size: float = 32.0

    # Nearby reuse (helps ID continuity)
    nearby_reuse_px: float = 80.0
    nearby_size_ratio: float = 0.75
    nearby_allow_class_mismatch: bool = False


class KalmanBox:
    """
    State x = [cx, cy, vx, vy, w, h]
    Measurement z = [cx, cy, w, h]
    """

    def __init__(self, cx: float, cy: float, w: float, h: float, *, t0: float, cls: int, name: str, conf: float):
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

    def predict(self, t_now: float, p: TrackerParams) -> None:
        dt = float(max(0.0, t_now - self.t_last))
        self.t_last = float(t_now)

        F = np.eye(6, dtype=np.float64)
        F[0, 2] = dt
        F[1, 3] = dt

        q_pos = float(p.process_noise_pos)
        q_vel = float(p.process_noise_vel)
        Q = np.diag([q_pos * q_pos, q_pos * q_pos, q_vel * q_vel, q_vel * q_vel, 20.0, 20.0]).astype(np.float64)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        self.x[4] = max(2.0, float(self.x[4]))
        self.x[5] = max(2.0, float(self.x[5]))

    def update(self, meas: Tuple[float, float, float, float], *, t_meas: float, conf: float, p: TrackerParams) -> None:
        z = np.array(meas, dtype=np.float64)

        H = np.zeros((4, 6), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 4] = 1.0
        H[3, 5] = 1.0

        r_pos = float(p.meas_noise_pos)
        r_size = float(p.meas_noise_size)
        R = np.diag([r_pos * r_pos, r_pos * r_pos, r_size * r_size, r_size * r_size]).astype(np.float64)

        y = z - (H @ self.x)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ H) @ self.P

        # EMA smooth width/height to reduce jitter
        a = float(p.ema_alpha)
        self._ema_w = (1.0 - a) * self._ema_w + a * float(z[2])
        self._ema_h = (1.0 - a) * self._ema_h + a * float(z[3])
        self.x[4] = max(2.0, self._ema_w)
        self.x[5] = max(2.0, self._ema_h)

        self.conf = float(conf)
        self.hits += 1
        self.t_last_meas = float(t_meas)
        if self.hits >= int(p.min_hits):
            self.confirmed = True

    def age_since_meas(self, t_now: float) -> float:
        return float(t_now - self.t_last_meas)

    def xyxy(self) -> Tuple[float, float, float, float]:
        cx, cy, _, _, w, h = self.x.tolist()
        return (float(cx - w / 2.0), float(cy - h / 2.0), float(cx + w / 2.0), float(cy + h / 2.0))


class SimpleSORTTracker:
    """
    Very lightweight SORT-ish tracker:
      - IoU matching within same class
      - optional "nearby reuse" to keep ID continuity
      - KalmanBox per track
    """

    def __init__(self, params: Optional[TrackerParams] = None):
        self.p = params or TrackerParams()
        self.tracks: Dict[int, KalmanBox] = {}
        self._next_id = 1

    def predict(self, t_now: float) -> None:
        for tr in self.tracks.values():
            tr.predict(t_now, self.p)

    def update_with_dets(self, dets: List[dict], *, t_now: float) -> None:
        """
        dets expected schema (minimum):
          {
            "xyxy": (x1,y1,x2,y2),
            "cls": int,
            "conf": float,
            "name": str,
          }
        """
        if not dets:
            self._prune(t_now)
            return

        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid].xyxy() for tid in track_ids]
        det_boxes = [d["xyxy"] for d in dets]

        used_tracks = set()
        used_dets = set()

        # IoU candidates (class-consistent)
        cands = []
        for j, d in enumerate(dets):
            for i, tid in enumerate(track_ids):
                tr = self.tracks[tid]
                if int(d["cls"]) != int(tr.cls):
                    continue
                score = iou_xyxy(track_boxes[i], det_boxes[j])
                if score > 0.0:
                    cands.append((score, i, j))
        cands.sort(key=lambda t: t[0], reverse=True)

        # Greedy assignment by IoU
        for score, i, j in cands:
            if score < float(self.p.match_iou):
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
            self.tracks[tid].update((cx, cy, w, h), t_meas=t_now, conf=float(d["conf"]), p=self.p)
            self.tracks[tid].name = str(d.get("name", self.tracks[tid].name))

        unmatched_track_idx = {i for i, tid in enumerate(track_ids) if tid not in used_tracks}

        # Nearby reuse (helps maintain IDs through short matching failures)
        for j, d in enumerate(dets):
            if j in used_dets:
                continue

            det_box = d["xyxy"]
            det_cx, det_cy = box_center_xy(det_box)

            best_i = None
            best_dist = 1e18
            for i in list(unmatched_track_idx):
                tid = track_ids[i]
                tr = self.tracks[tid]
                tr_box = tr.xyxy()

                if (not self.p.nearby_allow_class_mismatch) and (int(d["cls"]) != int(tr.cls)):
                    continue

                tr_cx, tr_cy = box_center_xy(tr_box)
                dist = float(np.hypot(det_cx - tr_cx, det_cy - tr_cy))
                if dist > float(self.p.nearby_reuse_px):
                    continue
                if size_similarity_ratio(det_box, tr_box) < float(self.p.nearby_size_ratio):
                    continue

                if dist < best_dist:
                    best_dist = dist
                    best_i = i

            if best_i is not None:
                tid = track_ids[best_i]
                x1, y1, x2, y2 = det_box
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                w, h = max(2.0, x2 - x1), max(2.0, y2 - y1)

                self.tracks[tid].update((cx, cy, w, h), t_meas=t_now, conf=float(d["conf"]), p=self.p)
                self.tracks[tid].cls = int(d["cls"])
                self.tracks[tid].name = str(d.get("name", self.tracks[tid].name))

                used_tracks.add(tid)
                used_dets.add(j)
                unmatched_track_idx.discard(best_i)
                continue

            # New track
            x1, y1, x2, y2 = det_box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w, h = max(2.0, x2 - x1), max(2.0, y2 - y1)
            tid = self._next_id
            self._next_id += 1

            self.tracks[tid] = KalmanBox(
                cx, cy, w, h,
                t0=t_now,
                cls=int(d["cls"]),
                name=str(d.get("name", "obj")),
                conf=float(d.get("conf", 0.0)),
            )

        self._prune(t_now)

    def _prune(self, t_now: float) -> None:
        dead = [tid for tid, tr in self.tracks.items() if tr.age_since_meas(t_now) > float(self.p.max_age_sec)]
        for tid in dead:
            self.tracks.pop(tid, None)

    def get_tracks_as_dets(self, t_now: float) -> List[dict]:
        """
        Return track list in the same dict schema you used in your script.
        """
        out = []
        for tid, tr in self.tracks.items():
            if not tr.confirmed:
                continue
            age = tr.age_since_meas(t_now)
            if age > float(self.p.max_age_sec):
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
# Gaze-on-object hit helper (kept here because it's pure geometry)
# -----------------------
def gaze_object_hit(
    gx: float,
    gy: float,
    dets: List[dict],
    *,
    gaze_radius_px: float = 25.0,
    nearest_fallback_px: float = 60.0,
    gaze_recency_sec: Optional[float] = 0.35,
) -> Tuple[Optional[dict], Optional[str], Optional[float]]:
    """
    Decide which detection is "hit" by gaze point (gx,gy).

    Returns:
      (det, mode, dist)
    where mode in {"inside","near","nearest"} and dist is px distance to box (0 for inside).

    If gaze_recency_sec is provided, rejects if all det ages exceed that.
    """
    if not dets:
        return None, None, None

    if gaze_recency_sec is not None:
        ages = [float(d.get("age", 0.0)) for d in dets]
        if ages and min(ages) > float(gaze_recency_sec):
            return None, None, None

    inside_hits = []
    near_hits = []
    nearest = None

    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        area = _box_area(x1, y1, x2, y2)
        conf = float(d.get("conf", 0.0))

        if (x1 <= gx <= x2) and (y1 <= gy <= y2):
            # Prefer smaller boxes (more specific) then higher conf
            inside_hits.append((area, -conf, d))
            continue

        dist = _dist_point_to_box(gx, gy, x1, y1, x2, y2)
        if dist <= float(gaze_radius_px):
            # Prefer closer, then smaller, then higher conf
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
