"""
test_neon_lsl_bridge.py — WS4 F6 Neon→LSL bridge schema contract.

The NeonGaze channel layout is a stable contract for XDF consumers: the first 5
channels must match the pre-2026-05-27 schema by index (so old recordings load
by the same column lookup) and the synthetic channels (worn, depth_cm) must stay
at their fixed positions. This guards the schema without a phone or LSL outlet.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("pylsl")
pytest.importorskip("pupil_labs.realtime_api.simple")


def _load_bridge():
    spec = importlib.util.spec_from_file_location(
        "neon_lsl_bridge", ROOT / "tools" / "neon_lsl_bridge.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_gaze_channel_count_is_29():
    m = _load_bridge()
    assert len(m.GAZE_CHANNELS) == 29


def test_first_five_backward_compat_schema():
    m = _load_bridge()
    labels = [c[0] for c in m.GAZE_CHANNELS[:5]]
    assert labels == ["gaze_x_px", "gaze_y_px", "worn", "depth_cm", "unix_t"]


def test_synthetic_channels_are_worn_and_depth():
    # Channels with no source datum attribute (attr is None) are filled in by
    # the gaze loop directly; they must be exactly worn (idx 2) and depth_cm
    # (idx 3) — the loop writes sample[2]/sample[3] by fixed index.
    m = _load_bridge()
    synthetic = {c[0] for c in m.GAZE_CHANNELS if c[3] is None}
    assert synthetic == {"worn", "depth_cm"}
    assert m.GAZE_CHANNELS[2][0] == "worn"
    assert m.GAZE_CHANNELS[3][0] == "depth_cm"


def test_all_channel_entries_well_formed():
    m = _load_bridge()
    for label, unit, lsl_type, attr in m.GAZE_CHANNELS:
        assert isinstance(label, str) and label
        assert isinstance(unit, str) and unit
        assert isinstance(lsl_type, str) and lsl_type
        assert attr is None or isinstance(attr, str)
