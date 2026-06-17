"""
test_decide_waypoint_pairing.py — regression guard for the decide-path
waypoint misalignment fix.

compute_3d_waypoints drops detections with no valid mask depth, so
waypoints_out can be shorter than dets. The old code paired the gaze hit with
its waypoint via positional zip(dets, waypoints_out), which mis-attributes the
3D point to a different object once any earlier det is dropped.
VLMService._hit_det_and_waypoint pairs by label instead. These tests pin that.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from perception.object_detector import Detection  # noqa: E402
from vlm_service import VLMService  # noqa: E402

pair = VLMService._hit_det_and_waypoint


def _det(box, label):
    x1, y1, x2, y2 = box
    return Detection(
        label=label, confidence=0.9,
        box_xyxy=(float(x1), float(y1), float(x2), float(y2)),
        box_center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        mask_polygon=None,
    )


def _wp(label, z):
    return {"label": label, "position_cam": [0.0, 0.0, z],
            "pixel_center": [0.0, 0.0], "depth_median_m": z}


def test_aligned_case_picks_own_waypoint():
    dets = [_det((0, 0, 10, 10), "segment_0"), _det((40, 40, 60, 60), "segment_1")]
    wps = [_wp("segment_0", 0.5), _wp("segment_1", 1.2)]
    hit_det, hit_wp = pair(dets, wps, (50, 50))  # gaze inside segment_1
    assert hit_det.label == "segment_1"
    assert hit_wp["depth_median_m"] == 1.2


def test_dropped_earlier_det_does_not_misalign_the_hit():
    # segment_0 had no valid depth → compute_3d_waypoints dropped it, so
    # waypoints_out starts at segment_1. A positional zip would pair the hit
    # (segment_1) with segment_2's waypoint; label matching must not.
    dets = [
        _det((0, 0, 10, 10), "segment_0"),
        _det((40, 40, 60, 60), "segment_1"),
        _det((70, 70, 90, 90), "segment_2"),
    ]
    wps = [_wp("segment_1", 1.2), _wp("segment_2", 3.0)]  # segment_0 dropped
    hit_det, hit_wp = pair(dets, wps, (50, 50))  # gaze inside segment_1
    assert hit_det.label == "segment_1"
    assert hit_wp["depth_median_m"] == 1.2  # NOT 3.0 (segment_2) as a zip would give


def test_hit_det_without_waypoint_returns_none_not_misattributed():
    # The hit det itself was dropped (no valid depth) → no waypoint with its
    # label. Better to report None than a wrong object's 3D point.
    dets = [_det((40, 40, 60, 60), "segment_1")]
    wps = [_wp("segment_0", 0.5)]  # only an unrelated waypoint survived
    hit_det, hit_wp = pair(dets, wps, (50, 50))
    assert hit_det.label == "segment_1"
    assert hit_wp is None


def test_no_containing_det():
    dets = [_det((40, 40, 60, 60), "segment_1")]
    wps = [_wp("segment_1", 1.2)]
    hit_det, hit_wp = pair(dets, wps, (5, 5))  # gaze outside all boxes
    assert hit_det is None and hit_wp is None
