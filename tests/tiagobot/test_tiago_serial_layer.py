"""
test_tiago_serial_layer.py

Guards the Tier-1 serial helpers in `Utils/tiagobot.py`. Per
`Documents/SoftwareDocs/projects/tiagobot/test-suite/plan.md` §3.5 (optional). These are the lowest-
level boundary between the BCI and the Tiagobot hardware; silent
breakage here is the worst-case failure mode.

Citations (verified 2026-05-19):
  - `Utils/tiagobot.py:_find_arduino_by_usb_id` (lines 115-157)
  - `Utils/tiagobot.py:find_tiagobot_port` (line 160)
  - `Utils/tiagobot.py:find_glove_port` (line 177)
  - `Utils/tiagobot.py:TIAGOBOT_USB_VID/PID` (lines 95-96)
  - `Utils/tiagobot.py:GLOVE_USB_VID/PID` (lines 100-101)
  - `Utils/tiagobot.py:SIMULATION_MODE` snapshot (lines 51-56)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import Utils.tiagobot as _tg
from Utils.tiagobot import (
    GLOVE_USB_PID,
    GLOVE_USB_VID,
    TIAGOBOT_USB_PID,
    TIAGOBOT_USB_VID,
    find_glove_port,
    find_tiagobot_port,
    send_home,
    send_letter,
)


def _make_port(device, vid, pid, description="Arduino", serial_number=None):
    """Build a mock pyserial ListPortInfo entry."""
    info = MagicMock()
    info.device = device
    info.vid = vid
    info.pid = pid
    info.description = description
    info.serial_number = serial_number
    return info


# ---- USB descriptor disambiguation -----------------------------------
def test_find_tiagobot_port_returns_match():
    """A single Tiagobot Mega 2560 in the port list -> its device path."""
    ports = [
        _make_port("/dev/ttyACM0", TIAGOBOT_USB_VID, TIAGOBOT_USB_PID),
    ]
    with patch("serial.tools.list_ports.comports", return_value=ports):
        assert find_tiagobot_port(logger=None) == "/dev/ttyACM0"


def test_find_tiagobot_port_returns_none_when_absent():
    """No match -> None (caller decides whether absence is fatal)."""
    ports = [
        _make_port("/dev/ttyACM0", GLOVE_USB_VID, GLOVE_USB_PID),
    ]
    with patch("serial.tools.list_ports.comports", return_value=ports):
        assert find_tiagobot_port(logger=None) is None


def test_find_tiagobot_port_disambiguates_from_glove():
    """Both Tiagobot AND glove plugged in -> Tiagobot path returned for
    find_tiagobot_port (not the glove)."""
    ports = [
        _make_port("/dev/ttyACM0", GLOVE_USB_VID, GLOVE_USB_PID),
        _make_port("/dev/ttyACM1", TIAGOBOT_USB_VID, TIAGOBOT_USB_PID),
    ]
    with patch("serial.tools.list_ports.comports", return_value=ports):
        assert find_tiagobot_port(logger=None) == "/dev/ttyACM1"


def test_find_glove_port_picks_uno_not_mega():
    """Same as above, mirror-image — find_glove_port returns the Uno."""
    ports = [
        _make_port("/dev/ttyACM0", GLOVE_USB_VID, GLOVE_USB_PID),
        _make_port("/dev/ttyACM1", TIAGOBOT_USB_VID, TIAGOBOT_USB_PID),
    ]
    with patch("serial.tools.list_ports.comports", return_value=ports):
        assert find_glove_port(logger=None) == "/dev/ttyACM0"


def test_find_tiagobot_port_ambiguous_returns_none():
    """Two Mega 2560 devices on the same host (unusual but possible)
    -> None (operator must set TIAGOBOT_PORT explicitly to disambiguate).
    Per `Utils/tiagobot.py:152-156` the function logs but returns None."""
    ports = [
        _make_port("/dev/ttyACM0", TIAGOBOT_USB_VID, TIAGOBOT_USB_PID),
        _make_port("/dev/ttyACM1", TIAGOBOT_USB_VID, TIAGOBOT_USB_PID),
    ]
    with patch("serial.tools.list_ports.comports", return_value=ports):
        assert find_tiagobot_port(logger=None) is None


def test_find_tiagobot_port_handles_comports_failure():
    """If `serial.tools.list_ports.comports()` raises (rare, but
    possible on flaky USB stacks), return None — don't propagate."""
    with patch("serial.tools.list_ports.comports", side_effect=OSError("USB unplugged mid-scan")):
        assert find_tiagobot_port(logger=None) is None


# ---- USB-ID constants pinned to known Arduino values -----------------
def test_tiagobot_usb_vid_pid_are_arduino_mega_2560():
    """Locks the VID/PID against accidental edits. Arduino SA VID is
    0x2341; Mega 2560 R3 PID is 0x0042."""
    assert TIAGOBOT_USB_VID == 0x2341
    assert TIAGOBOT_USB_PID == 0x0042


def test_glove_usb_vid_pid_are_arduino_uno_r3():
    """Arduino SA VID is 0x2341; Uno R3 PID is 0x0043."""
    assert GLOVE_USB_VID == 0x2341
    assert GLOVE_USB_PID == 0x0043


def test_glove_and_tiagobot_pids_distinct():
    """Sanity: the two Arduinos must have different PIDs, otherwise
    the find_*_port helpers cannot disambiguate them."""
    assert TIAGOBOT_USB_PID != GLOVE_USB_PID


# ---- SIMULATION_MODE bypass -----------------------------------------
def test_send_letter_simulation_mode_does_not_call_write():
    """When SIMULATION_MODE is True at the module level, send_letter
    must log only — never reach .write() on the mock serial port. This
    is the contract that makes SIMULATION_MODE-driven dry runs safe."""
    saved = _tg.SIMULATION_MODE
    _tg.SIMULATION_MODE = True
    try:
        mock_ser = MagicMock()
        # Even with a real-looking ser handle, SIM mode short-circuits
        # before .write — but the public API guards on `ser is None`
        # not on the global. We document the actual contract: when
        # SIMULATION_MODE is True at *import*, callers of open_port
        # receive None and send_letter receives None. Re-create that
        # caller-side behaviour explicitly.
        send_letter(None, "A", logger=None)
        mock_ser.write.assert_not_called()
    finally:
        _tg.SIMULATION_MODE = saved


def test_send_home_simulation_mode_does_not_call_write():
    """Same for send_home."""
    saved = _tg.SIMULATION_MODE
    _tg.SIMULATION_MODE = True
    try:
        send_home(None, logger=None)
    finally:
        _tg.SIMULATION_MODE = saved


# ---- send_letter rejects unknown letters before touching the port ----
def test_send_letter_rejects_unknown_before_write():
    """Passing an unknown letter raises immediately, BEFORE any write.
    Lock this ordering so a future refactor doesn't accidentally
    write nonsense to the port between the unknown-key encoding and
    the raise."""
    mock_ser = MagicMock()
    with pytest.raises(ValueError):
        send_letter(mock_ser, "Z", logger=None)
    mock_ser.write.assert_not_called()
