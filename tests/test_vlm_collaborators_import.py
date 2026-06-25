#!/usr/bin/env python3
"""
test_vlm_collaborators_import.py — importability + construction smoke for the
two subsystems extracted from vlm_service.py's VLMService god class:
vlm/results_pusher.py::ResultsPusher and vlm/segment_stream.py::SegmentStream.

Confirms the modules import cleanly (no import cycle with vlm_service, which
imports them) and that each collaborator constructs against a minimal stub
service holding only the back-ref state its __init__ reads — proving the
composition wiring (service → collaborator) is intact without loading Neon /
FastSAM / Depth Pro / Gemini.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vlm.results_pusher import ResultsPusher  # noqa: E402
from vlm.segment_stream import SegmentStream  # noqa: E402


def test_results_pusher_constructs_and_owns_its_state() -> None:
    svc = types.SimpleNamespace()  # ResultsPusher.__init__ reads no svc state
    rp = ResultsPusher(svc)
    assert rp._svc is svc
    # Owns the subscriber registry + tick-thread handles (moved off VLMService).
    assert rp._subscribers == {}
    assert rp._results_tick_thread is None
    # The rate/TTL caps moved to the collaborator as class constants.
    assert ResultsPusher._RESULTS_TICK_HZ > 0.0
    assert ResultsPusher._SUBSCRIBER_TTL_S > 0.0


def test_segment_stream_constructs_and_owns_its_state() -> None:
    # SegmentStream.__init__ reads svc.args.seg_track via the back-ref.
    svc = types.SimpleNamespace(args=types.SimpleNamespace(seg_track=False))
    ss = SegmentStream(svc)
    assert ss._svc is svc
    assert ss._seg_track is False
    # Owns the stream thread + stats snapshot (moved off VLMService).
    assert ss._seg_stream_thread is None
    assert ss._seg_stream_stats["active"] is False
    assert SegmentStream._SEG_STREAM_STATS_S > 0.0
