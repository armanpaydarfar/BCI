"""
test_calibration_roundtrip.py

Guards calibration save / load drift in the Tiagobot gaze layer. Per
`Tiagobot_Test_Suite_Plan.md` §3.2.

The NPZ schema and load contract are defined by:
  - `tiago_gaze_calibration_exec.py:_save_calibration` (writer)
  - `Utils/tiagobot_gaze.py:load_centroids` (reader)

A drift between the two (e.g. column order, row labels, missing-letter
encoding) would silently misclassify gaze at runtime. This test pins:
  - Write a fixture NPZ with known centroids -> load -> classify -> the
    same letters we put in come back out.
  - Loading a missing file raises FileNotFoundError (not silent
    fall-back to defaults).
  - Loading a malformed NPZ raises ValueError.
  - A NaN row in the centroids array means that letter is excluded
    (not silently coerced to nominal coords).
"""
from __future__ import annotations

import numpy as np
import pytest

from Utils.tiagobot_gaze import (
    LETTERS,
    classify_gaze_to_letter,
    load_centroids,
)


def _write_calibration_npz(
    path,
    centroids: np.ndarray,
    *,
    include_letters: bool = True,
) -> None:
    """Helper: write a calibration NPZ with the documented schema.

    `centroids` is expected to be (9, 3): [x_norm, y_norm, depth_cm].
    The depth column may contain NaN per-row; the loader treats those
    as "2D-only" centroids."""
    arrays = dict(
        T=np.array([0.0], dtype=np.float64),
        G=np.zeros((1, 4), dtype=np.float32),
        labels=np.array(["A"], dtype="S1"),
        centroids=centroids.astype(np.float32),
        meta=dict(version=1),
    )
    if include_letters:
        arrays["letters"] = np.array(list(LETTERS), dtype="S1")
    np.savez_compressed(path, **arrays)


@pytest.fixture
def fixture_centroids():
    """A deterministic set of centroids — each letter offset from its
    nominal grid position so we can verify the loader preserves the
    exact values, not the nominal fallback. Depth varies per letter so
    a row-swap bug surfaces in the depth column too."""
    base = np.array(
        [
            [0.25, 0.25, 50.0],
            [0.50, 0.25, 52.0],
            [0.75, 0.25, 54.0],
            [0.25, 0.50, 56.0],
            [0.50, 0.50, 58.0],
            [0.75, 0.50, 60.0],
            [0.25, 0.75, 62.0],
            [0.50, 0.75, 64.0],
            [0.75, 0.75, 66.0],
        ],
        dtype=np.float64,
    )
    # Add a unique per-row jitter so a row-swap bug would surface.
    jitter = np.linspace(0.001, 0.009, 9)
    base[:, 0] += jitter
    return base


def test_round_trip_preserves_values(tmp_path, fixture_centroids):
    """Write -> load -> values match within float32 precision."""
    npz_path = tmp_path / "cal.npz"
    _write_calibration_npz(npz_path, fixture_centroids)

    loaded = load_centroids(npz_path)
    assert set(loaded.keys()) == set(LETTERS)
    for i, ch in enumerate(LETTERS):
        gx, gy, gz = loaded[ch]
        # float32 round-trip tolerance.
        assert abs(gx - fixture_centroids[i, 0]) < 1e-5, (
            f"{ch}: x mismatch {gx} vs {fixture_centroids[i, 0]}"
        )
        assert abs(gy - fixture_centroids[i, 1]) < 1e-5, (
            f"{ch}: y mismatch {gy} vs {fixture_centroids[i, 1]}"
        )
        assert abs(gz - fixture_centroids[i, 2]) < 1e-4, (
            f"{ch}: depth mismatch {gz} vs {fixture_centroids[i, 2]}"
        )


def test_round_trip_bit_identical_classification(tmp_path, fixture_centroids):
    """Two calls to load_centroids on the same file followed by the
    same classify call produce the same answer (no nondeterminism in
    the loader)."""
    npz_path = tmp_path / "cal.npz"
    _write_calibration_npz(npz_path, fixture_centroids)

    loaded_a = load_centroids(npz_path)
    loaded_b = load_centroids(npz_path)
    # Same keys, same values.
    assert loaded_a == loaded_b

    # Same classification under both copies, for a fixed gaze point.
    for gx, gy in [(0.5, 0.5), (0.25, 0.75), (0.75, 0.75), (0.27, 0.27)]:
        assert classify_gaze_to_letter(gx, gy, loaded_a) == \
            classify_gaze_to_letter(gx, gy, loaded_b)


def test_missing_file_raises(tmp_path):
    """Loading a non-existent path raises FileNotFoundError (NOT silent
    fall-back to nominal centroids). Per CLAUDE.md fail-fast on Tier 2
    paths."""
    missing = tmp_path / "definitely_does_not_exist.npz"
    with pytest.raises(FileNotFoundError):
        load_centroids(missing)


def test_corrupt_npz_raises(tmp_path):
    """A file that's not a valid NPZ raises (numpy raises an
    OSError / EOFError / similar). load_centroids does not catch these
    — they propagate, surfacing the corruption."""
    bad_path = tmp_path / "corrupt.npz"
    bad_path.write_bytes(b"this is not a valid numpy archive\x00\x01\x02")
    with pytest.raises(Exception):
        # numpy raises different exception classes depending on
        # version (BadZipFile / OSError / ValueError). Any non-success
        # is acceptable; the contract is fail-loud.
        load_centroids(bad_path)


def test_missing_centroids_key_raises(tmp_path):
    """NPZ that has labels but no 'centroids' array -> ValueError."""
    npz_path = tmp_path / "no_centroids.npz"
    np.savez_compressed(
        npz_path,
        T=np.array([0.0]),
        labels=np.array(["A"], dtype="S1"),
    )
    with pytest.raises(ValueError, match="missing 'centroids'"):
        load_centroids(npz_path)


def test_wrong_shape_centroids_raises(tmp_path):
    """A (8, 3) centroids array (one letter short) -> ValueError, not
    silent truncation."""
    npz_path = tmp_path / "wrong_shape.npz"
    _write_calibration_npz(
        npz_path,
        centroids=np.zeros((8, 3), dtype=np.float64),
        include_letters=False,
    )
    with pytest.raises(ValueError, match="expected"):
        load_centroids(npz_path)


def test_old_2d_centroids_rejected(tmp_path):
    """Old (9, 2) NPZs (pre-depth, before 2026-05-20) must be rejected
    with a clear message — silently loading them with a NaN depth
    column would let a depth-aware classifier run with no depth
    information, hiding the schema mismatch."""
    npz_path = tmp_path / "legacy_2d.npz"
    _write_calibration_npz(
        npz_path,
        centroids=np.zeros((9, 2), dtype=np.float64),
        include_letters=False,
    )
    with pytest.raises(ValueError, match="expected"):
        load_centroids(npz_path)


def test_wrong_letter_order_raises(tmp_path, fixture_centroids):
    """If the 'letters' array exists and its order doesn't match
    `LETTERS`, that's a real corruption (the writer guarantees row
    ordering). The loader must fail loudly."""
    npz_path = tmp_path / "wrong_order.npz"
    bad_letters = np.array(["I", "H", "G", "F", "E", "D", "C", "B", "A"], dtype="S1")
    np.savez_compressed(
        npz_path,
        centroids=fixture_centroids.astype(np.float32),
        letters=bad_letters,
    )
    with pytest.raises(ValueError, match="letter order"):
        load_centroids(npz_path)


def test_nan_row_excludes_letter(tmp_path, fixture_centroids):
    """A NaN row in the centroids array means the calibration ran for
    that letter and got no valid samples. The loaded dict simply omits
    the letter — at runtime, classify_gaze_to_letter restricted to the
    available set will skip it."""
    centroids = fixture_centroids.copy()
    centroids[4] = np.nan  # E missing

    npz_path = tmp_path / "missing_E.npz"
    _write_calibration_npz(npz_path, centroids)

    loaded = load_centroids(npz_path)
    assert "E" not in loaded
    assert "A" in loaded and "I" in loaded
    assert len(loaded) == 8


def test_calibration_with_all_letters_missing_returns_empty(tmp_path):
    """Cold-start case: calibration NPZ exists but every row is NaN
    (e.g. operator started a session before Neon was tracking).
    Loaded dict is empty; runtime classification on empty dict returns
    None (verified in test_region_classifier.py)."""
    npz_path = tmp_path / "all_nan.npz"
    _write_calibration_npz(npz_path, np.full((9, 3), np.nan, dtype=np.float64))
    loaded = load_centroids(npz_path)
    assert loaded == {}
