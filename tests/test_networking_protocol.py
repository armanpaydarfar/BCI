"""
test_networking_protocol.py

Guards the "wire-format / ACK matching" bug class. Example commits this
file would have caught (or moved detection to pre-commit):

  - 7b20b1c  feat: ACKs feed back into marker stream; introduced
             ACK base-token matching in Utils/networking.py.
  - d268e2c  fix: networking method to correctly send/receive ACKs +
             forward to marker stream. The base-token contract
             ("h;dur=3" must ACK against "h") lives in
             `_base_token` / `_build_ack_map`.

Citations under test (verified 2026-05-18):

  - Utils/networking.py:158-169  `_to_wire`
  - Utils/networking.py:171-192  `_is_coords_string`
  - Utils/networking.py:195-206  `_base_token`
  - Utils/networking.py:209-226  `_build_ack_map`

Gotcha — SIMULATION_MODE is snapshotted at import. See
`Utils/networking.py:64-67`:

    SIMULATION_MODE = bool(getattr(_config, "SIMULATION_MODE", False)) \
                      if _config is not None else False

Any test that asserts on the flag MUST monkeypatch
`Utils.networking.SIMULATION_MODE`, NOT `config.SIMULATION_MODE`.
The pure helpers tested here do not read the flag, so this file does
not need the `sim_mode_networking` fixture — surfaced in the docstring
so the gotcha is not forgotten when extending this file.

Socket round-trip tests (using `socket.socketpair()`) are deferred
to Phase 1b `tests/test_networking_sockets.py` per Plan §6 #8.
"""

from __future__ import annotations

import pytest

from Utils.networking import (
    _base_token,
    _build_ack_map,
    _is_coords_string,
    _to_wire,
)


# ─── _to_wire ──────────────────────────────────────────────────────────────

class TestToWire:
    def test_bytes_decoded_to_utf8(self):
        assert _to_wire(b"hello") == "hello"

    def test_bytearray_decoded(self):
        assert _to_wire(bytearray(b"abc")) == "abc"

    def test_str_passthrough(self):
        assert _to_wire("h;dur=3") == "h;dur=3"

    def test_seven_numbers_formatted_with_six_decimal_places(self):
        wire = _to_wire([1.0, 2, 3.5, 4, 5, 6, 7.25])
        # Six-decimal formatting is part of the wire format (see file:166).
        assert wire == "1.000000,2.000000,3.500000,4.000000,5.000000,6.000000,7.250000"

    def test_six_or_eight_numbers_falls_through_to_str(self):
        # Only length-7 sequences get the special joint-coord encoding.
        out = _to_wire([1.0, 2, 3, 4, 5, 6])
        assert out == str([1.0, 2, 3, 4, 5, 6])

    def test_non_numeric_seq_falls_through_to_str(self):
        # Mixed-type 7-element sequence should NOT be coord-encoded.
        out = _to_wire(["a", 2, 3, 4, 5, 6, 7])
        assert out == str(["a", 2, 3, 4, 5, 6, 7])


# ─── _is_coords_string ────────────────────────────────────────────────────

class TestIsCoordsString:
    def test_plain_seven_floats_true(self):
        assert _is_coords_string("0.1,0.2,0.3,0.4,0.5,0.6,0.7") is True

    def test_bracketed_seven_floats_true(self):
        assert _is_coords_string("[0.1,0.2,0.3,0.4,0.5,0.6,0.7]") is True

    def test_with_dur_suffix_true(self):
        assert _is_coords_string("0.1,0.2,0.3,0.4,0.5,0.6,0.7;dur=2.5") is True

    def test_bracketed_with_dur_suffix_is_false(self):
        # Quirk of the current implementation (Utils/networking.py:171-192):
        # the bracket-strip only fires when the *whole* string is wrapped in
        # `[...]`. With a `;dur=...` suffix the string ends in a digit, so
        # brackets are NOT stripped, and the float parse of "[0.1" fails.
        # Locked here to surface the quirk if the implementation is reworked.
        assert _is_coords_string("[0.1,0.2,0.3,0.4,0.5,0.6,0.7];dur=2") is False

    def test_unbracketed_seven_floats_with_suffix_true(self):
        # The supported "trajectory with duration" wire form is unbracketed
        # plus suffix, e.g. as `_to_wire` emits for 7-number sequences.
        assert _is_coords_string("0.1,0.2,0.3,0.4,0.5,0.6,0.7;dur=2") is True

    def test_whitespace_tolerated(self):
        assert _is_coords_string("  0.1, 0.2,0.3 ,0.4,0.5,0.6,0.7  ") is True

    def test_six_values_false(self):
        assert _is_coords_string("0.1,0.2,0.3,0.4,0.5,0.6") is False

    def test_eight_values_false(self):
        assert _is_coords_string("0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8") is False

    def test_non_numeric_false(self):
        assert _is_coords_string("a,b,c,d,e,f,g") is False

    def test_opcode_h_false(self):
        assert _is_coords_string("h;dur=3") is False

    def test_empty_false(self):
        assert _is_coords_string("") is False


# ─── _base_token ──────────────────────────────────────────────────────────

class TestBaseToken:
    @pytest.mark.parametrize("msg,expected", [
        # The bug class fixed in 7b20b1c / d268e2c: tokens like "h;dur=3.000000"
        # must match against the bare base token "h" in the ACK map.
        ("h;dur=3.000000",        "h"),
        ("h;dur=3",               "h"),
        ("q;seq=123",             "q"),
        ("g",                     "g"),
        ("s",                     "s"),
        ("p",                     "p"),
        ("r",                     "r"),
        # Whitespace stripping is part of the contract (file:203).
        ("  h;dur=3  ",           "h"),
    ])
    def test_opcode_with_suffix_yields_bare_token(self, msg, expected):
        assert _base_token(msg) == expected

    def test_unbracketed_coords_with_suffix_kept_whole(self):
        # The supported wire form for trajectories is unbracketed-with-suffix.
        # `_is_coords_string` returns True for it, so `_base_token` must NOT
        # split on ';' — the suffix is part of the trajectory contract.
        s = "0.1,0.2,0.3,0.4,0.5,0.6,0.7;dur=2"
        assert _base_token(s) == s

    def test_bracketed_coords_with_suffix_falls_back_to_prefix(self):
        # See `test_bracketed_with_dur_suffix_is_false`: `_is_coords_string`
        # rejects the bracketed-with-suffix form, so `_base_token` falls
        # through to `split(';', 1)[0]` — the bracketed prefix.
        s = "[0.1,0.2,0.3,0.4,0.5,0.6,0.7];dur=2"
        assert _base_token(s) == "[0.1,0.2,0.3,0.4,0.5,0.6,0.7]"

    def test_bytes_coerced_via_to_wire(self):
        # `_base_token` calls `_to_wire` for non-strings (file:201-202).
        assert _base_token(b"h;dur=3") == "h"

    def test_seven_number_list_coerced_to_coords_token(self):
        out = _base_token([1.0, 2, 3, 4, 5, 6, 7])
        assert "," in out
        # Survives _is_coords_string check → returned whole.
        assert _is_coords_string(out)


# ─── _build_ack_map ───────────────────────────────────────────────────────

class _MiniConfig:
    """Minimal config-shaped object exposing only what `_build_ack_map`
    reads (ROBOT_OPCODES, TRIGGERS)."""

    def __init__(self, robot_opcodes, triggers):
        self.ROBOT_OPCODES = robot_opcodes
        self.TRIGGERS = triggers


class TestBuildAckMap:
    def test_real_config_round_trip(self):
        """The real `config` module shape must produce a map keyed on bare
        base tokens (e.g. "h" not "h;dur=3"). Otherwise an inbound ACK for
        the HOME command would fail to look up the ACK_ROBOT_HOME trigger —
        which is exactly the bug class of 7b20b1c / d268e2c."""
        import config
        m = _build_ack_map(config)
        # HOME opcode in config.py:331 is "h;dur=3" — the ACK map must
        # store it under the base token "h", and the value must be the
        # ACK_ROBOT_HOME trigger value (config.py:312 -> "385").
        assert m["h"] == config.TRIGGERS["ACK_ROBOT_HOME"]
        # GO / STOP / PAUSE / RESUME / MASTER_UNLOCK / MASTER_LOCK / QUERY
        # all map to their respective ACK triggers via the same base-token rule.
        assert m["g"] == config.TRIGGERS["ACK_ROBOT_BEGIN"]
        assert m["s"] == config.TRIGGERS["ACK_ROBOT_STOP"]
        assert m["p"] == config.TRIGGERS["ACK_ROBOT_PAUSE"]
        assert m["r"] == config.TRIGGERS["ACK_ROBOT_RESUME"]
        assert m["m"] == config.TRIGGERS["ACK_MASTER_UNLOCK"]
        assert m["c"] == config.TRIGGERS["ACK_MASTER_LOCK"]
        # QUERY's ACK is optional; the call uses `.get(..., None)`.
        assert "q" in m

    def test_none_config_returns_empty(self):
        assert _build_ack_map(None) == {}

    def test_missing_attrs_returns_empty(self):
        class _Bad:
            pass
        assert _build_ack_map(_Bad()) == {}

    def test_home_opcode_with_dur_suffix_stored_under_bare_token(self):
        cfg = _MiniConfig(
            robot_opcodes={"HOME": "h;dur=3.000000", "GO": "g", "STOP": "s",
                           "PAUSE": "p", "RESUME": "r",
                           "MASTER_UNLOCK": "m", "MASTER_LOCK": "c",
                           "QUERY": "q"},
            triggers={"ACK_ROBOT_HOME": "385", "ACK_ROBOT_BEGIN": "305",
                      "ACK_ROBOT_STOP": "345", "ACK_ROBOT_PAUSE": "365",
                      "ACK_ROBOT_RESUME": "375",
                      "ACK_MASTER_UNLOCK": "505", "ACK_MASTER_LOCK": "525"},
        )
        m = _build_ack_map(cfg)
        # The key MUST be "h" — never "h;dur=3.000000" — or runtime ACK
        # lookup fails for the HOME command.
        assert "h" in m
        assert "h;dur=3.000000" not in m
        assert m["h"] == "385"
