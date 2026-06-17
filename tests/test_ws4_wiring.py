"""
test_ws4_wiring.py — close the WS4 review's "verified by reading, not by test"
gaps. These drive the command methods / dispatch / reasoner / widget paint tick
end-to-end (with stubs only for the heavy model + Qt leaves), so a refactor that
unwires a feature fails a test instead of silently passing.

Covers:
  - F1 constraint filter actually invoked inside _cmd_segment and _cmd_decide.
  - F5 'recognize' actually routed by _dispatch.
  - F4 thinking_budget actually consumed by IntentReasoner (incl. the gemini
    max_tokens interaction).
  - F2 widget paint tick actually renders with copy=True.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from concurrent.futures import Future
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from perception.object_detector import Detection  # noqa: E402
from perception.intent_reasoner import IntentReasoner  # noqa: E402
from vlm_service import VLMService, SegConstraints  # noqa: E402


def _det(box, label):
    x1, y1, x2, y2 = box
    return Detection(
        label=label, confidence=0.9,
        box_xyxy=(float(x1), float(y1), float(x2), float(y2)),
        box_center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        mask_polygon=None,
    )


class _StubDetector:
    def __init__(self, dets):
        self._dets = dets

    def detect(self, frame_bgr):
        # Return fresh copies so a mutation in one call can't leak across calls.
        return list(self._dets)


def _bundle():
    return types.SimpleNamespace(
        video=types.SimpleNamespace(bgr=np.zeros((100, 200, 3), dtype=np.uint8)),
        gaze=types.SimpleNamespace(x=50.0, y=50.0),
    )


def _service(detector, constraints):
    svc = VLMService.__new__(VLMService)
    svc.detector = detector
    svc._seg_constraints = constraints
    svc.depth_estimator = None
    svc._frame_lock = threading.Lock()
    svc._render_lock = threading.Lock()
    svc._latest_bundle = _bundle()
    svc._latest_fix = None
    svc._latest_bundle_t = time.time()
    svc._cached_dets = []
    svc._cached_hit_det = None
    svc._cached_hit_wp = None
    svc._vlm_state = "IDLE"
    svc._last_decision = None
    svc._first_snap_det = None
    svc._FixationState = lambda **k: None
    return svc


# A big box covering the whole frame (ratio 1.0) + a small one. With
# max_area_ratio=0.5 the big one must be dropped by F1.
_BIG = _det((0, 0, 200, 100), "segment_0")
_SMALL = _det((40, 40, 60, 60), "segment_1")


def test_f1_invoked_in_cmd_segment():
    svc = _service(_StubDetector([_BIG, _SMALL]), SegConstraints(max_area_ratio=0.5))
    out = svc._cmd_segment({})
    assert out["ok"] is True
    labels = [d["label"] for d in out["detections"]]
    assert labels == ["segment_1"]  # big whole-frame det filtered out
    assert out["n"] == 1


def test_f1_noop_in_cmd_segment_when_inactive():
    svc = _service(_StubDetector([_BIG, _SMALL]), SegConstraints())  # all off
    out = svc._cmd_segment({})
    assert out["n"] == 2  # default = unchanged


def test_f1_invoked_in_cmd_decide():
    # depth disabled → the no-depth branch still applies geometry constraints.
    svc = _service(_StubDetector([_BIG, _SMALL]), SegConstraints(max_area_ratio=0.5))

    class _StubReasoner:
        def reason_async(self, *a, **k):
            f = Future()
            f.set_result({})
            return f

    svc.reasoner = _StubReasoner()
    out = svc._cmd_decide({})
    assert out["ok"] is True
    # The filtered set is what gets cached / fed to the reasoner.
    assert [d.label for d in svc._cached_dets] == ["segment_1"]


def test_f5_dispatch_routes_recognize():
    svc = VLMService.__new__(VLMService)
    svc.recognizer = None  # → recognizer_disabled, which only _cmd_recognize returns
    resp = svc._dispatch("recognize", {}, ("127.0.0.1", 0))
    assert resp["ok"] is False
    assert resp["error"] == "recognizer_disabled"


def test_f5_dispatch_unknown_cmd_is_distinct():
    # Guard against a typo'd dispatch key silently behaving like recognize.
    svc = VLMService.__new__(VLMService)
    resp = svc._dispatch("recogni", {}, ("127.0.0.1", 0))
    assert "unknown cmd" in resp["error"]


def test_f4_reasoner_consumes_thinking_budget():
    # gemini backend: thinking_budget None (unset) leaves the default-bump path
    # (max_tokens 1024 → 2048); 0 disables it (stays 1024); explicit value kept.
    r_none = IntentReasoner(api_key="x", model="gemini-2.5-flash", thinking_budget=None)
    assert r_none.thinking_budget is None
    assert r_none.max_tokens == 2048

    r_zero = IntentReasoner(api_key="x", model="gemini-2.5-flash", thinking_budget=0)
    assert r_zero.thinking_budget == 0
    assert r_zero.max_tokens == 1024  # budget 0 suppresses the truncation bump

    r_val = IntentReasoner(api_key="x", model="gemini-2.5-flash", thinking_budget=512)
    assert r_val.thinking_budget == 512


def test_f2_widget_paints_with_copy_true():
    pytest.importorskip("PySide6")
    from Utils.vlm_scene_widget import VLMSceneWidget

    w = VLMSceneWidget.__new__(VLMSceneWidget)
    w._bundle_lock = threading.Lock()
    w._latest_frame_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    w._latest_gaze_xy = (5.0, 5.0)
    w._latest_arrival_t = 0.0
    w._latest_vlm = {}
    w._latest_gaze = {}
    w._vlm_subscriber = None
    w._paint_latency_ms = __import__("collections").deque(maxlen=8)
    w._paint_count = 0
    w._fps_window_t = time.monotonic()  # recent → skips the 1 s status branch
    w._fps = 0.0

    calls = []

    class _RecRenderer:
        def render(self, frame, **kw):
            calls.append(kw)
            return frame

    w._renderer = _RecRenderer()
    w._paint_canvas = lambda canvas: None  # stub the Qt leaf

    w._on_paint_tick()

    assert len(calls) == 1
    assert calls[0]["copy"] is True  # the WS4 F2 fix — never copy=False here
