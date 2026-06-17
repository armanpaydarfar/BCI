"""
test_overlay_no_accumulation.py — WS4 F2 root-cause guard.

The "translucent shapes render opaque over Tailscale" bug is mask alpha
*compounding* when SceneOverlayRenderer.render(copy=False) re-composites the
same buffer across paint ticks (which happens when the frame relay stalls but
the paint timer keeps firing). This test pins the mechanism at the renderer
level so a regression (flipping the panel back to a shared-buffer in-place
blend) is caught:

  - copy=False on the SAME buffer twice → the masked region drifts toward the
    fill colour (accumulation — the bug).
  - copy=True twice → byte-identical output each call (idempotent — the fix the
    panel relies on).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from Utils.scene_overlay_renderer import SceneOverlayRenderer  # noqa: E402


def _frame():
    return np.full((60, 80, 3), 10, dtype=np.uint8)  # dark, uniform


def _det():
    # A square mask covering a known interior pixel (30, 20).
    poly = [[10, 10], [50, 10], [50, 40], [10, 40]]
    return [{
        "label": "segment_0",
        "confidence": 0.9,
        "box_xyxy": [10, 10, 50, 40],
        "box_center": [30, 25],
        "mask_polygon": poly,
    }]


def test_copy_false_on_shared_buffer_accumulates():
    r = SceneOverlayRenderer()
    buf = _frame()
    dets = _det()
    # Sample a masked pixel below the top-left VLM state badge (which is an
    # opaque draw and would clobber the alpha-blended value otherwise).
    r.render(buf, detections=dets, copy=False)
    after_one = int(buf[38, 45, 1])  # green channel inside the mask
    r.render(buf, detections=dets, copy=False)
    after_two = int(buf[38, 45, 1])
    # Second in-place blend pushes the pixel further toward the fill colour.
    assert after_two > after_one, (after_one, after_two)


def test_copy_true_is_idempotent():
    r = SceneOverlayRenderer()
    frame = _frame()
    dets = _det()
    out1 = r.render(frame, detections=dets, copy=True)
    out2 = r.render(frame, detections=dets, copy=True)
    # Fresh copy each call → the source frame is never mutated and the two
    # renders are byte-identical (no accumulation).
    assert np.array_equal(out1, out2)
    assert np.array_equal(frame, _frame())  # source untouched
