"""
test_seg_tracking.py — WS4 F3 temporal tracking/smoothing wiring.

Exercises VLMService._apply_seg_tracking against the real (Tier-1, read-only)
SimpleSORTTracker with synthetic detections — no FastSAM/Neon. Verifies the
anti-flicker contract: min_hits-to-appear, stable track_id, mask preserved when
present, box-only coasting through max_age, then expiry. Also pins the
class-agnostic trick (constant cls=0) that lets FastSAM's unstable segment_N
labels associate by pure IoU.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from perception.object_detector import Detection  # noqa: E402
from Utils.gaze.gaze_tracking import SimpleSORTTracker  # noqa: E402
from vlm_service import VLMService  # noqa: E402


def _svc():
    # The method reads no instance state, so skip __init__.
    return VLMService.__new__(VLMService)


def _det(box=(10, 10, 50, 50), label="segment_0", conf=0.9):
    x1, y1, x2, y2 = box
    mask = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    dtype=np.int32).reshape(-1, 1, 2)
    return Detection(
        label=label, confidence=conf,
        box_xyxy=(float(x1), float(y1), float(x2), float(y2)),
        box_center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        mask_polygon=mask,
    )


def test_min_hits_hysteresis_then_stable_id_with_mask():
    svc = _svc()
    tk = SimpleSORTTracker()
    # Frame 1: first sighting — not yet confirmed (min_hits=2) → nothing shown.
    out = svc._apply_seg_tracking([_det()], tk, 0.0)
    assert out == []
    # Frame 2: second sighting → confirmed, appears with a stable id + mask.
    out = svc._apply_seg_tracking([_det()], tk, 0.1)
    assert len(out) == 1
    tid = out[0].track_id
    assert out[0].label == f"#{tid}"
    assert out[0].mask_polygon is not None  # present this frame → mask kept
    # Frame 3: still present, SAME id (identity persists).
    out = svc._apply_seg_tracking([_det()], tk, 0.2)
    assert out[0].track_id == tid


def test_coasting_then_expiry():
    svc = _svc()
    tk = SimpleSORTTracker()
    svc._apply_seg_tracking([_det()], tk, 0.0)
    out = svc._apply_seg_tracking([_det()], tk, 0.1)
    tid = out[0].track_id
    # Detection vanishes: within max_age (1.25 s) the confirmed track coasts as
    # a box-only detection (no mask) so the overlay doesn't flicker out.
    out = svc._apply_seg_tracking([], tk, 0.3)
    assert len(out) == 1
    assert out[0].track_id == tid
    assert out[0].mask_polygon is None
    # Past max_age → the track expires and nothing is shown.
    out = svc._apply_seg_tracking([], tk, 2.0)
    assert out == []


def test_unstable_labels_still_associate_by_iou():
    # FastSAM relabels the same object segment_0 → segment_3 → … frame to frame.
    # Constant cls=0 means association is pure IoU, so the id stays put.
    svc = _svc()
    tk = SimpleSORTTracker()
    svc._apply_seg_tracking([_det(label="segment_0")], tk, 0.0)
    out = svc._apply_seg_tracking([_det(label="segment_7")], tk, 0.1)
    tid = out[0].track_id
    out = svc._apply_seg_tracking([_det(label="segment_2")], tk, 0.2)
    assert out[0].track_id == tid  # same physical object, same track despite relabel
