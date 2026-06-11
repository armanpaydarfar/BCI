"""
test_gaze_tracking_pure.py

Preventive coverage for the gaze geometry layer — no specific historical
SHA, but the gaze stack is ~3.4K LOC and growing (Plan §5.2 #4) and the
project is about to add a calibration rework (Project 2). Locking the
contract for the pure helpers (`iou_xyxy`, `size_similarity_ratio`,
`gaze_object_hit`) and the SORT tracker's ID-continuity behaviour gives
that work a safety net.

The module's own docstring labels these as pure / I/O-free; the tests
here exercise them directly without any device or display.

Citations under test (verified 2026-05-18):

  - Utils/gaze/gaze_tracking.py:53-61   `size_similarity_ratio`
  - Utils/gaze/gaze_tracking.py:64-84   `iou_xyxy`
  - Utils/gaze/gaze_tracking.py:193-322 `SimpleSORTTracker` + `update_with_dets`
  - Utils/gaze/gaze_tracking.py:357-426 `gaze_object_hit`

Plan-drift note (Plan §10):

  - Plan §5.2 #4 said: "size_similarity_ratio: identical → 1.0; 2x → 0.5".
    The implementation multiplies width-ratio by height-ratio (file:59-61),
    so a uniform 2× scale gives 0.25, while a 2× scale in *one* dimension
    gives 0.5. Both cases are locked in below to make the behaviour
    unambiguous.
  - Plan §5.2 #4 said: "gaze_object_hit: ... → 'inside'; ... → 'near'".
    The actual return is `(det, mode, dist)` — a 3-tuple. Tests assert
    on the `mode` element rather than treating the return as a bare
    string.
"""

from __future__ import annotations

import pytest

from Utils.gaze.gaze_tracking import (
    SimpleSORTTracker,
    TrackerParams,
    gaze_object_hit,
    iou_xyxy,
    size_similarity_ratio,
)


# ─── iou_xyxy ─────────────────────────────────────────────────────────────

class TestIoU:
    def test_identical_boxes_iou_is_one(self):
        a = (0.0, 0.0, 10.0, 10.0)
        assert iou_xyxy(a, a) == pytest.approx(1.0)

    def test_disjoint_boxes_iou_is_zero(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (20.0, 20.0, 30.0, 30.0)
        assert iou_xyxy(a, b) == 0.0

    def test_touching_edges_iou_is_zero(self):
        # Sharing only an edge → zero-area intersection.
        a = (0.0, 0.0, 10.0, 10.0)
        b = (10.0, 0.0, 20.0, 10.0)
        assert iou_xyxy(a, b) == 0.0

    def test_half_overlap_iou(self):
        # a = 10x10 at (0,0); b = 10x10 at (5,0). Intersection = 5x10 = 50.
        # Union = 100 + 100 - 50 = 150. IoU = 1/3.
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 0.0, 15.0, 10.0)
        assert iou_xyxy(a, b) == pytest.approx(1.0 / 3.0)

    def test_contained_box(self):
        # b fully inside a. Intersection = b's area = 16.
        # Union = 100 + 16 - 16 = 100. IoU = 16/100.
        a = (0.0, 0.0, 10.0, 10.0)
        b = (2.0, 2.0, 6.0, 6.0)
        assert iou_xyxy(a, b) == pytest.approx(0.16)


# ─── size_similarity_ratio ────────────────────────────────────────────────

class TestSizeSimilarityRatio:
    def test_identical_boxes_ratio_is_one(self):
        a = (0.0, 0.0, 10.0, 10.0)
        assert size_similarity_ratio(a, a) == pytest.approx(1.0)

    def test_2x_uniform_scale_ratio_is_0_25(self):
        # Plan §5.2 #4 said "2x → 0.5"; the actual implementation
        # multiplies w- and h-ratios, so uniform 2x → 0.5 * 0.5 = 0.25.
        a = (0.0, 0.0, 10.0, 10.0)
        b = (0.0, 0.0, 20.0, 20.0)
        assert size_similarity_ratio(a, b) == pytest.approx(0.25)

    def test_2x_single_axis_ratio_is_0_5(self):
        # 2x scale on width only — matches the plan's "2x → 0.5" reading.
        a = (0.0, 0.0, 10.0, 10.0)
        b = (0.0, 0.0, 20.0, 10.0)
        assert size_similarity_ratio(a, b) == pytest.approx(0.5)

    def test_commutative(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (0.0, 0.0, 20.0, 30.0)
        assert size_similarity_ratio(a, b) == pytest.approx(size_similarity_ratio(b, a))


# ─── gaze_object_hit ──────────────────────────────────────────────────────

def _det(xyxy, *, cls=0, conf=0.9, name="obj", age=0.0):
    return {"xyxy": xyxy, "cls": cls, "conf": conf, "name": name, "age": age}


class TestGazeObjectHit:
    def test_empty_dets_returns_none(self):
        out = gaze_object_hit(100.0, 100.0, [])
        assert out == (None, None, None)

    def test_inside_box_returns_inside_mode(self):
        dets = [_det((0.0, 0.0, 100.0, 100.0))]
        d, mode, dist = gaze_object_hit(50.0, 50.0, dets)
        assert d is dets[0]
        assert mode == "inside"
        assert dist == 0.0

    def test_near_box_returns_near_mode(self):
        # Gaze 10 px outside a box, within default gaze_radius_px=25.
        dets = [_det((0.0, 0.0, 100.0, 100.0))]
        d, mode, dist = gaze_object_hit(110.0, 50.0, dets)
        assert d is dets[0]
        assert mode == "near"
        assert dist == pytest.approx(10.0)

    def test_outside_radius_but_within_fallback_returns_nearest(self):
        # 40 px away — outside 25 (near) but inside 60 (nearest_fallback).
        dets = [_det((0.0, 0.0, 100.0, 100.0))]
        d, mode, dist = gaze_object_hit(140.0, 50.0, dets)
        assert d is dets[0]
        assert mode == "nearest"
        assert dist == pytest.approx(40.0)

    def test_far_beyond_fallback_returns_none(self):
        # 200 px away — outside even the default 60 px fallback.
        dets = [_det((0.0, 0.0, 100.0, 100.0))]
        out = gaze_object_hit(300.0, 50.0, dets)
        assert out == (None, None, None)

    def test_prefers_smaller_inside_box(self):
        # Two boxes; gaze inside both — small box wins (area, then conf).
        big = _det((0.0, 0.0, 200.0, 200.0), name="big")
        small = _det((40.0, 40.0, 60.0, 60.0), name="small")
        d, mode, _ = gaze_object_hit(50.0, 50.0, [big, small])
        assert d is small
        assert mode == "inside"

    def test_stale_dets_rejected_by_gaze_recency(self):
        # All dets older than gaze_recency_sec → return None.
        dets = [_det((0.0, 0.0, 100.0, 100.0), age=1.0)]
        out = gaze_object_hit(50.0, 50.0, dets, gaze_recency_sec=0.35)
        assert out == (None, None, None)


# ─── SimpleSORTTracker ───────────────────────────────────────────────────

class TestSimpleSORTTracker:
    def test_first_frame_creates_track(self):
        tr = SimpleSORTTracker()
        tr.update_with_dets([_det((0.0, 0.0, 50.0, 50.0))], t_now=0.0)
        assert len(tr.tracks) == 1

    def test_id_persists_across_three_consecutive_frames(self):
        """Feed the same detection at three consecutive timestamps with
        small jitter; the track ID assigned on frame 1 must survive to
        frame 3, and only one confirmed track should be visible."""
        tr = SimpleSORTTracker()
        # Frame 1: new track (id=1, hits=1, not yet confirmed)
        tr.update_with_dets([_det((0.0, 0.0, 50.0, 50.0))], t_now=0.0)
        first_id = next(iter(tr.tracks.keys()))

        # Frame 2: same box, slightly shifted — should match by IoU
        tr.update_with_dets([_det((2.0, 1.0, 52.0, 51.0))], t_now=0.05)
        assert set(tr.tracks.keys()) == {first_id}

        # Frame 3: still moving slightly
        tr.update_with_dets([_det((4.0, 2.0, 54.0, 52.0))], t_now=0.10)
        assert set(tr.tracks.keys()) == {first_id}

        confirmed = tr.get_tracks_as_dets(t_now=0.10)
        assert len(confirmed) == 1
        assert confirmed[0]["track_id"] == first_id

    def test_two_distinct_detections_get_distinct_ids(self):
        tr = SimpleSORTTracker()
        a = _det((0.0, 0.0, 50.0, 50.0), name="a")
        b = _det((500.0, 500.0, 550.0, 550.0), name="b")
        tr.update_with_dets([a, b], t_now=0.0)
        ids = list(tr.tracks.keys())
        assert len(ids) == 2
        assert len(set(ids)) == 2

    def test_class_mismatch_prevents_iou_match(self):
        """An IoU-overlapping detection of a different class must not be
        assigned to an existing track (file:236-237)."""
        tr = SimpleSORTTracker()
        tr.update_with_dets([_det((0.0, 0.0, 50.0, 50.0), cls=0, name="cup")], t_now=0.0)
        first_id = next(iter(tr.tracks.keys()))

        # Same spatial box, different class — should NOT match.
        # `nearby_allow_class_mismatch` is False by default (file:107).
        tr.update_with_dets([_det((0.0, 0.0, 50.0, 50.0), cls=1, name="plate")], t_now=0.05)
        # Two tracks now exist with distinct IDs.
        assert len(tr.tracks) == 2
        assert first_id in tr.tracks

    def test_track_pruned_after_max_age(self):
        """A track with no measurement for > max_age_sec is removed by
        `_prune` (file:324-327)."""
        params = TrackerParams(max_age_sec=0.5)
        tr = SimpleSORTTracker(params=params)
        tr.update_with_dets([_det((0.0, 0.0, 50.0, 50.0))], t_now=0.0)
        assert len(tr.tracks) == 1

        # Feed something far away so the original track is not matched.
        tr.update_with_dets([_det((1000.0, 1000.0, 1050.0, 1050.0))], t_now=1.0)
        # Original track (last measured at t=0.0) is now 1.0s stale — pruned.
        assert len(tr.tracks) == 1
        remaining = next(iter(tr.tracks.values()))
        assert remaining.t_last_meas == pytest.approx(1.0)
