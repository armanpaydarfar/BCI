"""
test_waypoints_cmd.py — guard the fast 3-D-waypoints command (WS4 live control
loop) added to vlm_service.py.

`waypoints` is `decide` minus the Gemini reasoner: it runs the same shared
segment → depth → 3D-waypoints → gaze hit-test pipeline (apply_constraints=True,
same dets cached for the overlay) but must NOT call reasoner.reason_async and
must NOT enter the THINKING overlay state — Gemini's 30-40 s round-trip is
unusable per-fixation. This pins:

  * the dispatch table wires "waypoints" and it is NOT in the THINKING-state set;
  * driving _cmd_waypoints returns the 3-D keys, caches dets, and never touches
    the reasoner;
  * with depth disabled it returns ok with depth_enabled=False (empty waypoints).

Heavy deps (detector / depth / reasoner) are mocked — no FastSAM, Depth Pro,
Gemini, or Neon. The harness mirrors test_decide_pipeline_dedup.py.
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
def svc(monkeypatch):
    """A VLMService (heavy deps mocked) wired so the waypoints path runs
    headless. Records reasoner calls, cached dets, and vlm-state transitions so
    the tests can assert the fast path stays out of THINKING + reasoning."""
    s = VLMService.__new__(VLMService)

    # gaze (50,50) lands inside 'keep_1'; 'x_drop' is stripped by the F1 filter.
    base_dets = [
        _det((0, 0, 10, 10), "keep_0"),
        _det((40, 40, 60, 60), "keep_1"),
        _det((70, 70, 90, 90), "x_drop"),
    ]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    depth_map = np.ones((100, 100), dtype=np.float32)

    s.detector = types.SimpleNamespace(detect=lambda f: list(base_dets))
    s.depth_estimator = types.SimpleNamespace(estimate=lambda f, **k: (depth_map, None))
    s.reader = types.SimpleNamespace(camera_matrix=np.eye(3))
    s._seg_constraints = object()
    s._focal_px = lambda: 100.0

    s._cached = {"args": None}
    s._cache_dets = lambda *a, **k: s._cached.__setitem__("args", (a, k))

    s._states = []
    s._set_vlm_state = lambda state, **k: s._states.append(state)

    s.reasoner_calls = []
    s.reasoner = types.SimpleNamespace(
        reason_async=lambda *a, **k: s.reasoner_calls.append((a, k)),
    )

    bundle = types.SimpleNamespace(
        video=types.SimpleNamespace(bgr=frame, frame_idx=0, timestamp_ns=0),
        gaze=types.SimpleNamespace(x=50.0, y=50.0),
    )
    s._latest = lambda: (bundle, None, None)

    monkeypatch.setattr(
        vlm_service, "_apply_seg_constraints",
        lambda dets, shape, constraints, **k: [d for d in dets if not d.label.endswith("_drop")],
    )
    monkeypatch.setattr(
        _od, "compute_3d_waypoints",
        lambda dets, dm, K: [
            types.SimpleNamespace(
                label=d.label, position_cam=[0.0, 0.0, 1.0],
                pixel_center=[d.box_center[0], d.box_center[1]], depth_median_m=1.0,
            )
            for d in dets
        ],
    )
    return s


# ── dispatch wiring ──────────────────────────────────────────────────────────

def test_dispatch_wires_waypoints_and_not_thinking(svc):
    # Drive through the real _dispatch so both the handler lookup and the
    # THINKING-state gate at the dispatch level are exercised.
    resp = svc._dispatch("waypoints", {}, ("127.0.0.1", 0))
    assert resp["ok"] is True
    # The fast path must NOT enter THINKING (unlike decide/reason/etc.).
    assert "THINKING" not in svc._states


# ── pipeline behaviour: decide() minus the reasoner ──────────────────────────

def test_waypoints_returns_3d_keys_and_caches_dets(svc):
    resp = svc._cmd_waypoints({})
    assert resp["ok"] is True
    assert set(resp) >= {"waypoints", "hit_waypoint", "depth_at_gaze_m", "gaze_px"}
    # Same F1 filter as decide: 'x_drop' stripped before waypoints/hit-test.
    assert [w["label"] for w in resp["waypoints"]] == ["keep_0", "keep_1"]
    assert resp["hit_waypoint"]["label"] == "keep_1"
    assert resp["depth_at_gaze_m"] == 1.0
    assert resp["gaze_px"] == [50.0, 50.0]
    # Dets cached for the overlay (same as decide).
    assert svc._cached["args"] is not None
    # Depth on → no depth_enabled note.
    assert "depth_enabled" not in resp


def test_waypoints_never_calls_reasoner(svc):
    svc._cmd_waypoints({})
    assert svc.reasoner_calls == []


def test_waypoints_no_frame(svc):
    svc._latest = lambda: (None, None, None)
    assert svc._cmd_waypoints({}) == {"ok": False, "error": "no_frame"}


def test_waypoints_depth_disabled(svc):
    svc.depth_estimator = None
    resp = svc._cmd_waypoints({})
    assert resp["ok"] is True
    assert resp["depth_enabled"] is False
    assert resp["waypoints"] == []
    assert resp["hit_waypoint"] is None
    assert resp["depth_at_gaze_m"] is None
