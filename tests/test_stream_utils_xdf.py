"""
test_stream_utils_xdf.py

Guards the realtime-stream-selection regression class (Plan §6 #10).
Two pieces of `Utils/stream_utils.py` are covered:

1. `load_xdf` (file:80-197): when a recording contains both a hardware
   amplifier marker channel (e.g. "eegoSports-000104_markers") and the
   experiment-side `MarkerStream` produced by `UTIL_marker_stream.py`,
   the latter must always win. This is the bug class fixed in
   commit 71323ec.

2. `require_marker_stream` (file:19-52): startup guard that must call
   `pylsl.resolve_byprop` (NOT the deprecated `resolve_stream` with
   keyword args). This is the bug class fixed in commit e42cf16 —
   noted in Plan §10 as not caught by the import-smoke test.

Citations under test (verified 2026-05-18):

  - Utils/stream_utils.py:19-52    require_marker_stream
  - Utils/stream_utils.py:80-126   load_xdf marker selection branch
  - Utils/stream_utils.py:43       `resolve_byprop('name', 'MarkerStream',
                                   minimum=1, timeout=timeout)`

Synthetic XDF fixture: pyxdf is monkeypatched to return a hand-built
stream list, side-stepping the binary file format.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import Utils.stream_utils as su


# ─── synthetic stream factory ────────────────────────────────────────────

def _eeg_stream(name="eegoSports-000104", n_samples=10):
    return {
        "info": {
            "type": ["EEG"],
            "name": [name],
            "source_id": [f"{name}_src"],
            "nominal_srate": ["512"],
        },
        "time_series": np.zeros((n_samples, 32)),
        "time_stamps": np.linspace(0, 1.0, n_samples),
    }


def _marker_stream(name, *, typ="Markers"):
    return {
        "info": {
            "type": [typ],
            "name": [name],
            "source_id": [f"{name}_src"],
            "nominal_srate": ["0"],
        },
        "time_series": [[1.0]],
        "time_stamps": [0.5],
    }


# ─── load_xdf marker selection ───────────────────────────────────────────

class TestLoadXdfMarkerSelection:
    def test_prefers_marker_stream_over_hardware(self, monkeypatch):
        """When both a hardware marker channel and the experiment
        MarkerStream are present, load_xdf must return MarkerStream
        (file:115-117). This is the 71323ec regression."""
        hardware = _marker_stream("eegoSports-000104_markers")
        markerstream = _marker_stream("MarkerStream")
        streams = [_eeg_stream(), hardware, markerstream]

        with patch.object(su.pyxdf, "load_xdf", return_value=(streams, None)):
            _, marker = su.load_xdf("ignored.xdf", report=False)

        assert marker is markerstream, (
            "load_xdf returned the hardware marker stream instead of "
            "MarkerStream — see 71323ec."
        )

    def test_falls_back_to_hardware_when_no_marker_stream(self, monkeypatch):
        """If MarkerStream isn't present, the hardware channel is used as
        a fallback (file:118-123)."""
        hardware = _marker_stream("eegoSports-000104_markers")
        streams = [_eeg_stream(), hardware]

        with patch.object(su.pyxdf, "load_xdf", return_value=(streams, None)):
            _, marker = su.load_xdf("ignored.xdf", report=False)

        assert marker is hardware

    def test_prefers_marker_stream_regardless_of_order(self, monkeypatch):
        """Order independence: the MarkerStream wins even if it appears
        before the hardware marker channel."""
        markerstream = _marker_stream("MarkerStream")
        hardware = _marker_stream("eegoSports-000104_markers")
        streams = [_eeg_stream(), markerstream, hardware]

        with patch.object(su.pyxdf, "load_xdf", return_value=(streams, None)):
            _, marker = su.load_xdf("ignored.xdf", report=False)

        assert marker is markerstream

    def test_raises_when_marker_missing(self, monkeypatch):
        """No markers at all → ValueError."""
        streams = [_eeg_stream()]
        with patch.object(su.pyxdf, "load_xdf", return_value=(streams, None)):
            with pytest.raises(ValueError, match="EEG and Marker"):
                su.load_xdf("ignored.xdf", report=False)


# ─── require_marker_stream API contract ──────────────────────────────────

class TestRequireMarkerStream:
    def test_uses_resolve_byprop_not_resolve_stream(self, monkeypatch):
        """Post-e42cf16: the function must call `resolve_byprop`, NOT
        `resolve_stream(..., minimum=, timeout=)` (which rejects those
        kwargs and raised TypeError at startup)."""
        recorded = {}

        def fake_resolve_byprop(prop, value, minimum, timeout):
            recorded["call"] = (prop, value, minimum, timeout)
            # Truthy result → require_marker_stream succeeds without
            # calling sys.exit.
            return [MagicMock()]

        monkeypatch.setattr(su, "resolve_byprop", fake_resolve_byprop)
        # Trap any call to the broken API path.
        def fake_resolve_stream(*args, **kwargs):
            raise AssertionError(
                "require_marker_stream called resolve_stream — the "
                "post-e42cf16 contract requires resolve_byprop."
            )
        monkeypatch.setattr(su, "resolve_stream", fake_resolve_stream)

        su.require_marker_stream(logger=None, timeout=0.1)

        # The function must pass `prop="name", value="MarkerStream"`
        # and the timeout argument forwarded from the caller.
        assert recorded["call"] == ("name", "MarkerStream", 1, 0.1)

    def test_exits_when_no_marker_stream_found(self, monkeypatch):
        """When resolve_byprop returns an empty list, the function should
        log + sys.exit(1) — this is the hard startup gate."""
        monkeypatch.setattr(su, "resolve_byprop",
                            lambda *args, **kwargs: [])
        with pytest.raises(SystemExit) as exc_info:
            su.require_marker_stream(logger=None, timeout=0.1)
        assert exc_info.value.code == 1
