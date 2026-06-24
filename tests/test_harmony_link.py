"""
test_harmony_link.py — pure helpers of the Harmony robot UDP client (WS5).

The socket methods need the robot; these pin the hardware-free wire-format
helpers: the joint-command string builder (must be exactly 7 radians + ;dur=)
and the telemetry parser (extracts X + Q for the active side).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from Utils.gaze.harmony_link import (  # noqa: E402
    HarmonyLink,
    build_joint_command_str,
    parse_telemetry,
)


def test_build_joint_command_str_format():
    s = build_joint_command_str([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], 5.0)
    assert s == "0.100000,0.200000,0.300000,0.400000,0.500000,0.600000,0.700000;dur=5.000"


def test_build_joint_command_rejects_wrong_length():
    with pytest.raises(ValueError):
        build_joint_command_str([0.1, 0.2, 0.3], 5.0)        # too few
    with pytest.raises(ValueError):
        build_joint_command_str(list(range(8)), 5.0)         # too many


def test_build_joint_command_rejects_nonfinite():
    with pytest.raises(ValueError):
        build_joint_command_str([0, 0, 0, np.nan, 0, 0, 0], 5.0)


def _telemetry(side="R"):
    return json.dumps({
        "qR": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "eeR": {"pos_mm": [100.0, 200.0, 300.0]},
        "qL": [1, 1, 1, 1, 1, 1, 1],
        "eeL": {"pos_mm": [10.0, 20.0, 30.0]},
    })


def test_parse_telemetry_right_side():
    out = parse_telemetry(_telemetry(), "R")
    assert out is not None
    np.testing.assert_allclose(out["q"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    np.testing.assert_allclose(out["ee"], [100.0, 200.0, 300.0])


def test_parse_telemetry_left_side():
    out = parse_telemetry(_telemetry(), "L")
    np.testing.assert_allclose(out["ee"], [10.0, 20.0, 30.0])


def test_parse_telemetry_rejects_non_json():
    assert parse_telemetry("ACK:CAPTURED_LOCKED", "R") is None
    assert parse_telemetry("", "R") is None


def test_parse_telemetry_rejects_missing_keys():
    assert parse_telemetry(json.dumps({"qR": [0] * 7}), "R") is None       # no ee
    assert parse_telemetry(json.dumps({"eeR": {"pos_mm": [1, 2, 3]}}), "R") is None  # no q


def test_parse_telemetry_rejects_short_vectors():
    bad = json.dumps({"qR": [0, 0, 0], "eeR": {"pos_mm": [1, 2, 3]}})
    assert parse_telemetry(bad, "R") is None


def test_query_state_stamps_arrival_time():
    """The REV04 sweep unpacks ``rstate["_t"]`` for its frame↔telemetry staleness
    gate (`stage_sweep`); ``query_state`` MUST return that host-clock arrival time
    alongside q/ee. Pins the contract so the sweep can't regress to a KeyError —
    the live sweep path needs a robot and is not otherwise covered."""
    link = HarmonyLink.__new__(HarmonyLink)  # bypass the socket bind in __init__
    link._seq = 0
    link._side = "R"
    link.send = lambda msg: None
    link.recv = lambda timeout_s: _telemetry()
    out = link.query_state()
    assert out is not None
    assert "_t" in out and out["_t"] > 0.0
    np.testing.assert_allclose(out["q"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
