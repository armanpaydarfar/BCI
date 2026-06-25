"""
test_decide_pipeline_dedup.py — characterization guard for the shared
segment→depth→waypoints→hit pipeline behind _cmd_decide and
_process_frame_and_gaze (vlm_service.py).

Both methods run the same geometry pipeline, but they diverge in ONE way that
must be preserved if the duplication is ever factored into a shared helper:

  * _cmd_decide applies _apply_seg_constraints to the detections BEFORE
    computing 3D waypoints / hit-testing (the F1 filter).
  * _process_frame_and_gaze does NOT filter — it waypoints/hit-tests every
    detection.

This test pins that divergence by recording exactly which detections reach
compute_3d_waypoints on each path, plus the shared depth-at-gaze + hit-by-label
behaviour. It mocks the heavy deps (detector / depth / reasoner) so it runs
headless — no FastSAM, no Depth Pro, no Gemini, no Neon.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import perception.object_detector as _od  # noqa: E402
from perception.object_detector import Detection  # noqa: E402
import vlm_service  # noqa: E402
from vlm_service import VLMService  # noqa: E402


def _det(box, label):
    x1, y1, x2, y2 = box
    return Detection(
        label=label, confidence=0.9,
        box_xyxy=(float(x1), float(y1), float(x2), float(y2)),
        box_center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        mask_polygon=None,
    )


@pytest.fixture()
def svc_and_recorder(monkeypatch):
    """A VLMService with the heavy deps mocked, plus a recorder of the
    detections that reach compute_3d_waypoints on whichever path runs."""
    svc = VLMService.__new__(VLMService)

    # Three detections; gaze (50,50) lands inside 'keep_1'. 'x_drop' is what
    # _apply_seg_constraints will strip on the decide path.
    base_dets = [
        _det((0, 0, 10, 10), "keep_0"),
        _det((40, 40, 60, 60), "keep_1"),
        _det((70, 70, 90, 90), "x_drop"),
    ]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    depth_map = np.ones((100, 100), dtype=np.float32)

    svc.detector = types.SimpleNamespace(detect=lambda f: list(base_dets))
    svc.depth_estimator = types.SimpleNamespace(estimate=lambda f, **k: (depth_map, None))
    svc.reader = types.SimpleNamespace(camera_matrix=np.eye(3))
    svc._seg_constraints = object()  # opaque; the filter is mocked below
    svc._focal_px = lambda: 100.0
    svc._cache_dets = lambda *a, **k: None
    svc._set_vlm_state = lambda *a, **k: None
    svc._FixationState = lambda **kw: None
    bundle = types.SimpleNamespace(
        video=types.SimpleNamespace(bgr=frame, frame_idx=0, timestamp_ns=0),
        gaze=types.SimpleNamespace(x=50.0, y=50.0),
    )
    svc._latest = lambda: (bundle, None, None)
    svc.reasoner = types.SimpleNamespace(
        reason_async=lambda *a, **k: types.SimpleNamespace(result=lambda timeout=None: {})
    )

    # The seg-constraint filter (decide path only): drop labels ending '_drop'.
    monkeypatch.setattr(
        vlm_service, "_apply_seg_constraints",
        lambda dets, shape, constraints, **k: [d for d in dets if not d.label.endswith("_drop")],
    )

    # Record what reaches compute_3d_waypoints, and synthesise waypoints for it.
    recorded = {"dets": None}

    def _fake_waypoints(dets, depth_map_arg, K):
        recorded["dets"] = list(dets)
        return [
            types.SimpleNamespace(
                label=d.label, position_cam=[0.0, 0.0, 1.0],
                pixel_center=[d.box_center[0], d.box_center[1]], depth_median_m=1.0,
            )
            for d in dets
        ]

    monkeypatch.setattr(_od, "compute_3d_waypoints", _fake_waypoints)
    return svc, recorded, base_dets


def test_process_frame_does_not_filter(svc_and_recorder):
    svc, recorded, base_dets = svc_and_recorder
    out = svc._process_frame_and_gaze(np.zeros((100, 100, 3), np.uint8), (50.0, 50.0))
    # All 3 detections returned + all 3 reached waypoint computation (no filter).
    assert [d.label for d in out["detections"]] == ["keep_0", "keep_1", "x_drop"]
    assert [d.label for d in recorded["dets"]] == ["keep_0", "keep_1", "x_drop"]
    # Shared geometry: hit-by-label on the gaze-containing det + depth at gaze.
    assert out["hit_det"].label == "keep_1"
    assert out["hit_waypoint"]["label"] == "keep_1"
    assert out["depth_at_gaze_m"] == 1.0


def test_decide_filters_before_waypoints(svc_and_recorder):
    svc, recorded, base_dets = svc_and_recorder
    resp = svc._cmd_decide({"timeout": 1.0})
    assert resp["ok"] is True
    # 'x_drop' was stripped by _apply_seg_constraints BEFORE waypoints — only
    # the two 'keep_*' detections reached compute_3d_waypoints.
    assert [d.label for d in recorded["dets"]] == ["keep_0", "keep_1"]
    # Hit-by-label still resolves the gaze-containing det + its depth.
    assert resp["hit_waypoint"]["label"] == "keep_1"
    assert resp["depth_at_gaze_m"] == 1.0
