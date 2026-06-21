"""
test_fes_listener_params.py

Guards the FES stimulation-parameter selection — the highest-consequence
Tier 1 path in the repo, since the chosen value is the electrical *current*
delivered to a subject. The selection logic was extracted from
`trigger_fes` into the pure helper `select_fes_params` (mirroring the
`UTIL_marker_stream.parse_marker_message` extraction) so it can be exercised
without the Rehamove serial device or the FES UDP socket.

Citations under test (verified 2026-06-21):

  - FES_listener.py  `select_fes_params`  — mode -> (current, duration,
    pulse_width); None for unconfigured channel; ValueError on bad mode.

Skipped on Windows: importing `FES_listener` runs `from rehamove import *`
at module top, which loads the Linux-only SWIG extension
(`STM_interface/1_packages/rehamoveLibrary/_rehamovelib.so`). Realtime FES
is Linux-only anyway, so this is where the coverage belongs. Same
convention as the FES import-contract test in test_imports_smoke.py.
"""

from __future__ import annotations

import sys

import pytest

_LINUX_ONLY = pytest.mark.skipif(
    sys.platform == "win32",
    reason="FES_listener imports the Linux-only rehamove SWIG extension "
           "(_rehamovelib.so); realtime FES is Linux-only.",
)

pytestmark = _LINUX_ONLY


@pytest.fixture
def fes_config():
    """A minimal config shaped like RehamoveConfig_simple.json: distinct
    sensory vs motor current/duration so the mode branch is observable, and
    a string pulse_width to exercise the int() coercion."""
    return {
        "FES_frequency": 35,
        "channels": {
            "red": {
                "Sensory_current_mA": 4.0,
                "Motor_current_mA": 12.0,
                "duration_sense": 1.5,
                "duration_Motor": 3.0,
                "pulse_width": "200",
            },
        },
    }


def _helper():
    from FES_listener import select_fes_params
    return select_fes_params


class TestSelectFesParams:
    def test_sensory_picks_sensory_current_and_duration(self, fes_config):
        current, duration, pw = _helper()(fes_config, "red", "SENSORY")
        assert current == 4.0
        assert duration == 1.5
        assert pw == 200

    def test_motor_picks_motor_current_and_duration(self, fes_config):
        current, duration, pw = _helper()(fes_config, "red", "MOTOR")
        assert current == 12.0
        assert duration == 3.0
        assert pw == 200

    def test_pulse_width_coerced_to_int(self, fes_config):
        # Stored as the string "200" in the JSON config; must come back int.
        _, _, pw = _helper()(fes_config, "red", "SENSORY")
        assert isinstance(pw, int)

    def test_unconfigured_channel_returns_none(self, fes_config):
        # Mirrors trigger_fes's "No configuration found" early return.
        assert _helper()(fes_config, "blue", "SENSORY") is None

    def test_unknown_mode_raises_not_silent_motor(self, fes_config):
        # The safety contract: a typo'd mode must NOT silently deliver MOTOR
        # current. Before the extraction the binary if/else made any non-
        # 'SENSORY' mode select motor parameters.
        with pytest.raises(ValueError, match="Unknown FES mode"):
            _helper()(fes_config, "red", "MOTRO")

    def test_case_sensitive_mode(self, fes_config):
        # 'sensory' (lowercase) is not 'SENSORY' — must raise, not fall to motor.
        with pytest.raises(ValueError, match="Unknown FES mode"):
            _helper()(fes_config, "red", "sensory")
