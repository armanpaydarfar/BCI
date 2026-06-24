"""
test_apriltag_sweep.py — pure sweep-acceptance logic (REV04 §2).

Pins the hardware-free gating the 20 Hz sweep loop relies on: frame↔telemetry
time-alignment and the light per-sample quality gates. The live sweep loop itself
needs a Neon + robot and is HIL-gated.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Utils.gaze.apriltag_sweep import (  # noqa: E402
    DEFAULT_MAX_ALIGN_DT_S,
    accept_sweep_sample,
    frame_telemetry_dt,
)


def test_frame_telemetry_dt_is_absolute():
    assert frame_telemetry_dt(100.05, 100.0) == frame_telemetry_dt(100.0, 100.05)
    assert abs(frame_telemetry_dt(100.05, 100.0) - 0.05) < 1e-9


def _accept(**kw):
    base = dict(world_seen=True, ee_seen=True, margins=[40.0, 35.0], dt_s=0.01)
    base.update(kw)
    return accept_sweep_sample(**base)


def test_accept_clean_sample():
    ok, reason = _accept()
    assert ok is True
    assert reason == "ok"


def test_reject_no_world_tag():
    ok, reason = _accept(world_seen=False)
    assert ok is False
    assert "world" in reason


def test_reject_no_ee_tag():
    ok, reason = _accept(ee_seen=False)
    assert ok is False
    assert "EE" in reason


def test_reject_stale_frame():
    ok, reason = _accept(dt_s=DEFAULT_MAX_ALIGN_DT_S + 0.01)
    assert ok is False
    assert "stale" in reason


def test_reject_low_margin_uses_weakest_tag():
    # The weakest contributing tag (margin 5) drives the gate even though another
    # is strong (50) — a glancing/blurred read must not be averaged away.
    ok, reason = _accept(margins=[50.0, 5.0], min_margin=20.0)
    assert ok is False
    assert "margin" in reason


def test_empty_margins_skip_margin_gate():
    # No tag margin offered (e.g. synthetic) → the margin check is skipped, not a
    # hard reject.
    ok, _ = _accept(margins=[])
    assert ok is True


def test_world_gate_precedes_dt_gate():
    # A missing world tag is reported before a stale-frame Δt — ordering matters for
    # the operator's discard log.
    ok, reason = _accept(world_seen=False, dt_s=10.0)
    assert ok is False
    assert "world" in reason


def test_boundary_dt_accepted():
    ok, _ = _accept(dt_s=DEFAULT_MAX_ALIGN_DT_S)
    assert ok is True  # exactly at the bound is not "over" it
