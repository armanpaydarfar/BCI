"""
test_seg_constraints.py — WS4 F1/E2 segmentation-constraint filter logic.

Exercises the pure-geometry pieces of vlm_service.py's constraint pass with
synthetic Detection objects (no FastSAM, depth, or Neon). Guards the
keep-working invariant: with no thresholds set the filter is a no-op, so
segment/decide/segment_stream output is unchanged until tuned.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from perception.object_detector import Detection  # noqa: E402
from vlm_service import (  # noqa: E402
    SegConstraints,
    _apply_seg_constraints,
    _filter_overlay_dets,
    _median_mask_depth,
)

FRAME = (100, 200, 3)  # H=100, W=200 → frame area 20000


def _det(x1, y1, x2, y2, conf=0.9, mask=None, label="segment_0"):
    return Detection(
        label=label,
        confidence=conf,
        box_xyxy=(float(x1), float(y1), float(x2), float(y2)),
        box_center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        mask_polygon=mask,
    )


def test_inactive_constraints_are_a_noop():
    dets = [_det(0, 0, 200, 100), _det(10, 10, 20, 20)]
    c = SegConstraints()  # all None
    assert not c.is_active()
    assert _apply_seg_constraints(dets, FRAME, c) is dets
    assert _apply_seg_constraints(dets, FRAME, None) is dets


def test_max_area_ratio_drops_whole_scene_blob():
    big = _det(0, 0, 200, 100)      # full frame → ratio 1.0
    small = _det(10, 10, 30, 30)    # 400/20000 = 0.02
    out = _apply_seg_constraints([big, small], FRAME, SegConstraints(max_area_ratio=0.5))
    assert out == [small]


def test_min_area_ratio_drops_specks():
    speck = _det(0, 0, 2, 2)        # 4/20000 = 0.0002
    ok = _det(10, 10, 60, 60)       # 2500/20000 = 0.125
    out = _apply_seg_constraints([speck, ok], FRAME, SegConstraints(min_area_ratio=0.01))
    assert out == [ok]


def test_gaze_roi_keeps_only_near_gaze():
    near = _det(90, 40, 110, 60)    # centre (100,50) — at gaze
    far = _det(0, 0, 20, 20)        # centre (10,10)
    gaze = (100.0, 50.0)
    # roi 0.1 → half-extent 20px in x, 10px in y
    out = _apply_seg_constraints([near, far], FRAME, SegConstraints(gaze_roi=0.1), gaze_xy=gaze)
    assert out == [near]


def test_gaze_roi_skipped_when_no_gaze_supplied():
    near = _det(90, 40, 110, 60)
    far = _det(0, 0, 20, 20)
    out = _apply_seg_constraints([near, far], FRAME, SegConstraints(gaze_roi=0.1))
    assert out == [near, far]  # no gaze_xy → constraint not enforced


def test_solidity_drops_low_solidity_mask():
    # A bbox 0..40 x 0..40 (area 1600). A thin diagonal-ish polygon fills only a
    # small fraction → low solidity. A near-full square polygon → high solidity.
    thin = np.array([[0, 0], [40, 0], [40, 4]], dtype=np.int32).reshape(-1, 1, 2)
    full = np.array([[0, 0], [40, 0], [40, 40], [0, 40]], dtype=np.int32).reshape(-1, 1, 2)
    d_thin = _det(0, 0, 40, 40, mask=thin)
    d_full = _det(0, 0, 40, 40, mask=full)
    out = _apply_seg_constraints([d_thin, d_full], FRAME, SegConstraints(solidity_min=0.5))
    assert out == [d_full]


def test_depth_band_requires_depth_and_filters_out_of_band():
    # Uniform-depth frame regions: build a depth map and two masks at different
    # depths. near det at 0.8 m (in band), far det at 3.0 m (out of band).
    depth = np.full((100, 200), 3.0, dtype=np.float32)
    depth[0:40, 0:40] = 0.8
    near_mask = np.array([[0, 0], [39, 0], [39, 39], [0, 39]], dtype=np.int32).reshape(-1, 1, 2)
    far_mask = np.array([[100, 60], [139, 60], [139, 99], [100, 99]], dtype=np.int32).reshape(-1, 1, 2)
    near = _det(0, 0, 40, 40, mask=near_mask)
    far = _det(100, 60, 140, 100, mask=far_mask)
    c = SegConstraints(depth_band=(0.2, 1.5))
    # Without a depth map the band is skipped (both kept).
    assert _apply_seg_constraints([near, far], FRAME, c) == [near, far]
    # With the depth map, only the in-band det survives.
    out = _apply_seg_constraints([near, far], FRAME, c, depth_map=depth)
    assert out == [near]


def test_median_mask_depth_ignores_invalid_samples():
    depth = np.full((100, 200), -1.0, dtype=np.float32)  # all invalid
    depth[0:10, 0:10] = 0.9
    mask = np.array([[0, 0], [9, 0], [9, 9], [0, 9]], dtype=np.int32).reshape(-1, 1, 2)
    d = _det(0, 0, 10, 10, mask=mask)
    assert abs(_median_mask_depth(d, depth) - 0.9) < 1e-5
    # A mask over only-invalid depth returns None.
    bad = np.array([[100, 50], [120, 50], [120, 70], [100, 70]], dtype=np.int32).reshape(-1, 1, 2)
    assert _median_mask_depth(_det(100, 50, 120, 70, mask=bad), depth) is None


def test_overlay_top_k_is_configurable():
    dets = [_det(0, 0, 10, 10, conf=c / 10.0) for c in range(1, 11)]  # 10 dets
    kept = _filter_overlay_dets(dets, top_k=3, contain_ratio=1.1, area_ratio=0.0)
    assert len(kept) == 3
    # Highest-confidence three survive.
    assert sorted(d.confidence for d in kept) == [0.8, 0.9, 1.0]
