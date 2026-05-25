"""
test_arduino_serial_contract.py

Guards silent wire-format drift between the Python host
(`Utils/tiagobot.py:send_letter` / `send_home`) and the Arduino firmware
at `tools/tiago_arduino/Final_code/Final_code.ino`.

Per `Documents/SoftwareDocs/projects/tiagobot/test-suite/plan.md` §3.4.

The wire format is documented in two places:
  - `Utils/tiagobot.py:7-19` module docstring.
  - `tools/tiago_arduino/README.md` (the Arduino-side mirror).

Both must stay in sync with the sketch's `loop()` parser (verified at
`Final_code.ino:84-95` — comma-split parser reading `analog,angle,delay`).
This test pins the byte sequence per letter so a divergence in any of
those locations fails loudly.

Fixture-regeneration note: if the Arduino sketch changes its parser,
regenerate the fixture files from the new LOCATIONS dict and commit them
in the same PR as the sketch change. The fixture file is the contract;
no fixture update without a corresponding sketch update is allowed.

Citations (verified 2026-05-19):
  - `Utils/tiagobot.py:LOCATIONS` (lines 66-76) — source of the (analog,
    angle, delay) per-letter tuples.
  - `Utils/tiagobot.py:HOME_COMMAND` (line 78) — `"h\\n"`.
  - `Utils/tiagobot.py:TARGET_REACHED_MARKER` (line 84) — `"Target Location Reached."`.
  - `Utils/tiagobot.py:HOMED_MARKER` (line 85) — `"Homed."`.
"""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from Utils.tiagobot import (
    HOME_COMMAND,
    HOMED_MARKER,
    LOCATIONS,
    TARGET_REACHED_MARKER,
    send_home,
    send_letter,
    wait_for_completion,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _read_fixture(letter: str) -> bytes:
    path = FIXTURES_DIR / f"letter_{letter}_bytes.txt"
    assert path.is_file(), (
        f"Missing fixture {path}. Regenerate fixtures from "
        f"Utils.tiagobot.LOCATIONS after any sketch-side parser change."
    )
    return path.read_bytes()


# ---- send_letter wire format -----------------------------------------
@pytest.mark.parametrize("letter", list(LOCATIONS.keys()))
def test_send_letter_writes_documented_bytes(letter):
    """For each letter A-I, send_letter must write exactly the bytes
    in the golden fixture (which is generated from LOCATIONS via the
    documented `'{analog},{angle},{delay}\\n'` format)."""
    expected_bytes = _read_fixture(letter)

    # Patch the SIMULATION_MODE flag back to False so send_letter takes
    # the .write() path even when test is run on a machine where
    # config.SIMULATION_MODE may be True.
    import Utils.tiagobot as _tg
    saved = _tg.SIMULATION_MODE
    _tg.SIMULATION_MODE = False
    try:
        mock_ser = MagicMock()
        send_letter(mock_ser, letter, logger=None)
        mock_ser.write.assert_called_once()
        written = mock_ser.write.call_args[0][0]
        assert written == expected_bytes, (
            f"Letter {letter!r} wrote {written!r}, expected "
            f"{expected_bytes!r}"
        )
    finally:
        _tg.SIMULATION_MODE = saved


def test_send_letter_unknown_letter_raises():
    """Unknown letters must raise ValueError, not silently fall back
    or write a nonsense payload to the wire."""
    mock_ser = MagicMock()
    with pytest.raises(ValueError, match="unknown location letter"):
        send_letter(mock_ser, "Z", logger=None)
    mock_ser.write.assert_not_called()


def test_send_letter_with_none_serial_is_noop():
    """`ser=None` is the SIMULATION_MODE / unconfigured-port path —
    must NOT write to anywhere, just log. (Logger=None to keep the
    test free of logger state.)"""
    # Any letter; observe nothing crashes and nothing tries to .write.
    send_letter(None, "A", logger=None)


# ---- send_home wire format -------------------------------------------
def test_home_command_constant_matches_contract():
    """The HOME_COMMAND string in Utils.tiagobot is the wire payload
    the Arduino sketch parses with `if (input == "h")`. If this
    constant ever changes, the sketch must change too."""
    assert HOME_COMMAND == "h\n", (
        f"HOME_COMMAND drifted: {HOME_COMMAND!r}. The sketch parser at "
        f"Final_code.ino expects exactly 'h\\n'."
    )


def test_send_home_writes_h_newline():
    """send_home must write exactly `b\"h\\n\"`."""
    import Utils.tiagobot as _tg
    saved = _tg.SIMULATION_MODE
    _tg.SIMULATION_MODE = False
    try:
        mock_ser = MagicMock()
        send_home(mock_ser, logger=None)
        mock_ser.write.assert_called_once_with(b"h\n")
    finally:
        _tg.SIMULATION_MODE = saved


def test_send_home_with_none_serial_is_noop():
    """`ser=None` -> no write attempt."""
    send_home(None, logger=None)


# ---- Completion-marker contract --------------------------------------
def test_completion_marker_constants_match_contract():
    """The two markers the sketch prints at motion-end. The driver
    grep-matches `marker in line` against incoming serial; any change
    to these strings on either side breaks trial-loop motion sync."""
    assert TARGET_REACHED_MARKER == "Target Location Reached."
    assert HOMED_MARKER == "Homed."


class _FakeSerial:
    """Minimal Serial-like object that returns scripted readline()
    output, then empty bytes. Used to drive wait_for_completion."""

    def __init__(self, lines):
        # Each line should be a bytes object ending in b'\n'.
        self._lines = list(lines)
        self.closed = False

    def readline(self):
        if not self._lines:
            return b""
        return self._lines.pop(0)

    def close(self):
        self.closed = True


def test_wait_for_completion_returns_true_on_marker_line():
    """Feed the fake serial a stream that includes the target-reached
    marker -> returns True before timeout."""
    fake = _FakeSerial([
        b"some boot noise\n",
        b"another debug print\n",
        b"Target Location Reached.\n",
    ])
    result = wait_for_completion(
        fake, TARGET_REACHED_MARKER, timeout=2.0, logger=None,
    )
    assert result is True


def test_wait_for_completion_returns_true_on_homed_marker():
    """Same path for the HOMED_MARKER."""
    fake = _FakeSerial([
        b"some unrelated debug\n",
        b"Homed.\n",
    ])
    result = wait_for_completion(
        fake, HOMED_MARKER, timeout=2.0, logger=None,
    )
    assert result is True


def test_wait_for_completion_returns_false_on_timeout():
    """If no marker arrives within the timeout, return False (do not
    raise — caller logs and proceeds to the next phase per
    fail-safe-but-loud trial-loop contract)."""
    fake = _FakeSerial([
        b"debug line that does not contain the marker\n",
    ])
    result = wait_for_completion(
        fake, TARGET_REACHED_MARKER, timeout=0.3, logger=None,
    )
    assert result is False


def test_wait_for_completion_none_serial_returns_true():
    """`ser=None` returns True immediately — the SIMULATION_MODE path
    where there is no actual hardware to wait for."""
    assert wait_for_completion(None, TARGET_REACHED_MARKER, timeout=1.0, logger=None) is True
