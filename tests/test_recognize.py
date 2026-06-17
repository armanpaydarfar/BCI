"""
test_recognize.py — WS4 F5 fast COCO recognition command.

Exercises _gaze_hit_recognize and VLMService._cmd_recognize with a stub YOLO
recognizer (no Neon, no ultralytics). Guards the keep-working invariant: the
command reports recognizer_disabled when no recognizer was loaded, so a default
deployment is unaffected.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from perception.object_detector import Detection  # noqa: E402
from vlm_service import VLMService, _gaze_hit_recognize  # noqa: E402


def _det(box, label, conf=0.9):
    x1, y1, x2, y2 = box
    return Detection(
        label=label, confidence=conf,
        box_xyxy=(float(x1), float(y1), float(x2), float(y2)),
        box_center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        mask_polygon=None,
    )


def test_hit_prefers_smallest_containing_box():
    big = _det((0, 0, 100, 100), "table")
    small = _det((40, 40, 60, 60), "cup")
    # Gaze inside both → the smaller (more specific) wins.
    assert _gaze_hit_recognize([big, small], (50, 50)).label == "cup"


def test_hit_nearest_fallback_and_miss():
    d = _det((40, 40, 60, 60), "cup")
    # Just outside the box but within the 80 px fallback radius → nearest.
    assert _gaze_hit_recognize([d], (70, 50)).label == "cup"
    # Far away → no hit.
    assert _gaze_hit_recognize([d], (500, 500)) is None
    assert _gaze_hit_recognize([], (1, 1)) is None


class _StubRecognizer:
    def __init__(self, dets):
        self._dets = dets

    def detect(self, frame_bgr):
        return self._dets


def _svc_with_recognizer(recognizer):
    svc = VLMService.__new__(VLMService)
    svc.recognizer = recognizer
    svc._frame_lock = threading.Lock()
    bundle = types.SimpleNamespace(
        video=types.SimpleNamespace(bgr=np.zeros((120, 160, 3), dtype=np.uint8)),
        gaze=types.SimpleNamespace(x=50.0, y=50.0),
    )
    svc._latest_bundle = bundle
    svc._latest_fix = None
    svc._latest_bundle_t = time.time()
    return svc


def test_recognize_disabled_without_model():
    svc = VLMService.__new__(VLMService)
    svc.recognizer = None
    out = svc._cmd_recognize({})
    assert out["ok"] is False and out["error"] == "recognizer_disabled"


def test_recognize_names_gaze_object():
    dets = [_det((0, 0, 100, 100), "table"), _det((40, 40, 60, 60), "cup")]
    svc = _svc_with_recognizer(_StubRecognizer(dets))
    out = svc._cmd_recognize({})
    assert out["ok"] is True
    assert out["label"] == "cup"
    assert out["n"] == 2
    assert out["hit"]["label"] == "cup"


def test_recognize_no_frame():
    svc = _svc_with_recognizer(_StubRecognizer([]))
    svc._latest_bundle = None
    out = svc._cmd_recognize({})
    assert out["ok"] is False and out["error"] == "no_frame"
