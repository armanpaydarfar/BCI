"""
test_marker_stream_parsing.py

Guards the marker-format drift class (Plan §6 #11). The parsing logic
was extracted from `process_udp_messages` into a pure helper
`parse_marker_message` so it can be exercised without an LSL outlet or
the UDP listener thread.

Citations under test (verified 2026-05-18):

  - UTIL_marker_stream.py — `parse_marker_message`
  - UTIL_marker_stream.py module docstring — accepted wire formats:
      "<marker_int>"
      "<marker_int>,<prob_mi>,<prob_rest>"

If a driver ever wants to send a new format, this file must change
in the same commit — surfacing what the LSL outlet downstream will
actually see.
"""

from __future__ import annotations

import pytest

from UTIL_marker_stream import parse_marker_message


class TestParseMarkerMessage:
    def test_marker_only(self):
        assert parse_marker_message("100") == (100, None, None)

    def test_marker_with_probs(self):
        assert parse_marker_message("200,0.7,0.3") == (200, 0.7, 0.3)

    def test_whitespace_tolerated(self):
        # Both outer and inner whitespace should be stripped — drivers in
        # the wild sometimes pad fields.
        assert parse_marker_message(" 200 , 0.5 , 0.5 ") == (200, 0.5, 0.5)

    def test_marker_negative_int_accepted(self):
        # int() accepts a leading "-" so negative markers parse, even
        # though the conventions in config.TRIGGERS are all non-negative.
        assert parse_marker_message("-1") == (-1, None, None)

    def test_marker_float_in_int_slot_raises(self):
        # First column is marker (int); int("2.5") rejects.
        with pytest.raises(ValueError):
            parse_marker_message("2.5")

    def test_alpha_marker_raises(self):
        with pytest.raises(ValueError):
            parse_marker_message("abc")

    def test_two_part_payload_raises(self):
        # Two columns is not a supported arity (only 1 or 3).
        with pytest.raises(ValueError, match="arity"):
            parse_marker_message("200,0.7")

    def test_four_part_payload_raises(self):
        with pytest.raises(ValueError, match="arity"):
            parse_marker_message("200,0.7,0.3,extra")

    def test_non_numeric_prob_raises(self):
        with pytest.raises(ValueError):
            parse_marker_message("200,foo,bar")

    def test_empty_message_raises(self):
        # split(',') on "" returns [""]; int("") raises ValueError.
        with pytest.raises(ValueError):
            parse_marker_message("")
