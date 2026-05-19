"""
test_region_classifier.py

Guards silent miscompute in the categorical A-I gaze classifier
(`Utils.tiagobot_gaze.classify_gaze_to_letter`, added by
`feature/tiagobot-gaze-integration` Phase 2.b). The classifier is the
load-bearing piece of the rudimentary calibration — if it returns the
wrong region, the Tiagobot moves to the wrong target without any visible
failure mode.

Per `Tiagobot_Test_Suite_Plan.md` §3.1.

Citations (verified 2026-05-19):
  - `Utils/tiagobot_gaze.py:grid_centroids_norm` — nominal centroid set
  - `Utils/tiagobot_gaze.py:classify_gaze_to_letter` — function under test
  - `Documents/SoftwareDocs/Tiagobot_Gaze_AI_Layout.md` — layout decision
"""
from __future__ import annotations

import pytest

from Utils.tiagobot_gaze import (
    LETTERS,
    classify_gaze_to_letter,
    grid_centroids_norm,
)


@pytest.fixture
def nominal_centroids():
    """The 3x3 nominal grid: A..I at {0.25, 0.5, 0.75}^2."""
    return grid_centroids_norm()


# ---- Hit-at-centroid: each letter must map to itself ------------------
@pytest.mark.parametrize("letter", list(LETTERS))
def test_gaze_at_centroid_returns_that_letter(nominal_centroids, letter):
    """A gaze sample placed exactly on a centroid must map to that
    letter. Most basic correctness check."""
    cx, cy = nominal_centroids[letter]
    assert classify_gaze_to_letter(cx, cy, nominal_centroids) == letter


# ---- Quadrant nudges --------------------------------------------------
def test_gaze_near_center_resolves_to_E(nominal_centroids):
    """Gaze at the exact image center maps to E (the middle cell)."""
    assert classify_gaze_to_letter(0.5, 0.5, nominal_centroids) == "E"


def test_gaze_top_left_corner_resolves_to_A(nominal_centroids):
    """Gaze near the (0, 0) corner maps to A (top-left cell)."""
    assert classify_gaze_to_letter(0.0, 0.0, nominal_centroids) == "A"


def test_gaze_bottom_right_corner_resolves_to_I(nominal_centroids):
    """Gaze near the (1, 1) corner maps to I (bottom-right cell)."""
    assert classify_gaze_to_letter(1.0, 1.0, nominal_centroids) == "I"


# ---- Boundary cases between adjacent letters --------------------------
def test_horizontal_boundary_between_A_and_B(nominal_centroids):
    """A is at (0.25, 0.25), B is at (0.50, 0.25). Midpoint = (0.375,
    0.25) is equidistant. We document the deterministic choice: A wins
    because it iterates earlier in the centroid dict (Python 3.7+
    insertion-order guarantee). Re-evaluate if grid_centroids_norm
    changes iteration order."""
    result = classify_gaze_to_letter(0.375, 0.25, nominal_centroids)
    assert result == "A"


def test_horizontal_boundary_slightly_into_B(nominal_centroids):
    """0.5 micro past the midpoint and B wins."""
    result = classify_gaze_to_letter(0.376, 0.25, nominal_centroids)
    assert result == "B"


def test_vertical_boundary_between_A_and_D(nominal_centroids):
    """A is at (0.25, 0.25), D is at (0.25, 0.50). Midpoint = (0.25,
    0.375) is equidistant; A wins on iteration order."""
    result = classify_gaze_to_letter(0.25, 0.375, nominal_centroids)
    assert result == "A"


# ---- Out-of-range with distance ceiling -------------------------------
def test_far_gaze_returns_none_under_max_dist(nominal_centroids):
    """Gaze far outside the grid + max_dist_norm set -> None (skip
    GO). This is the plan §6.3 step 4 fallback case."""
    result = classify_gaze_to_letter(
        2.0, 2.0, nominal_centroids, max_dist_norm=0.2
    )
    assert result is None


def test_far_gaze_returns_nearest_without_max_dist(nominal_centroids):
    """No ceiling -> always return the nearest letter (I in this
    case, the bottom-right cell)."""
    result = classify_gaze_to_letter(2.0, 2.0, nominal_centroids)
    assert result == "I"


def test_just_outside_ceiling_returns_none(nominal_centroids):
    """A is at (0.25, 0.25). Gaze at (0.55, 0.55) is sqrt(0.3^2 +
    0.3^2) ~= 0.424 from A, sqrt(0.05^2 + 0.05^2) ~= 0.0707 from E.
    With max_dist_norm=0.05, E is just out of reach -> None."""
    result = classify_gaze_to_letter(
        0.55, 0.55, nominal_centroids, max_dist_norm=0.05
    )
    assert result is None


def test_within_ceiling_returns_letter(nominal_centroids):
    """Same gaze with a generous ceiling -> E."""
    result = classify_gaze_to_letter(
        0.55, 0.55, nominal_centroids, max_dist_norm=0.2
    )
    assert result == "E"


# ---- available_letters subset ----------------------------------------
def test_available_letters_restricts_choice(nominal_centroids):
    """When available_letters constrains the eligible set, even gaze
    closer to a forbidden centroid maps to the closest eligible one.
    This is the runtime path when config.TIAGOBOT_TRAJECTORY is a
    subset of A-I."""
    # Gaze at E center, but E isn't eligible.
    result = classify_gaze_to_letter(
        0.5, 0.5, nominal_centroids, available_letters=["A", "I"]
    )
    # A and I are equidistant from (0.5, 0.5); A wins on iteration.
    assert result == "A"


def test_available_letters_unknown_letter_silently_skipped(nominal_centroids):
    """Letters in `available_letters` that aren't in `centroids` are
    skipped (e.g. when the calibration is missing one letter)."""
    result = classify_gaze_to_letter(
        0.5, 0.5, nominal_centroids,
        available_letters=["B", "Z"],  # Z doesn't exist
    )
    assert result == "B"


def test_empty_available_letters_returns_none(nominal_centroids):
    """If no letters are eligible after intersection, return None."""
    assert classify_gaze_to_letter(
        0.5, 0.5, nominal_centroids, available_letters=[]
    ) is None


def test_empty_centroids_returns_none():
    """Empty centroid dict (cold-start with no valid letters) -> None.
    This is the case when calibration has no valid samples for any
    letter."""
    assert classify_gaze_to_letter(0.5, 0.5, {}) is None


# ---- Missing-letter calibration handled at classifier level ----------
def test_missing_letter_calibration_skips_silently(nominal_centroids):
    """If one calibration row is missing, the classifier just doesn't
    return that letter. Verifies the contract: missing centroids =
    ineligible letters, not silent misclassification to a neighbour.
    """
    # Drop E from the centroid dict.
    incomplete = {k: v for k, v in nominal_centroids.items() if k != "E"}
    # Gaze at exactly E's centroid (which no longer exists). The
    # nearest cell is one of B, D, F, H (all 0.25 away). Pick the
    # iteration-earliest: B.
    result = classify_gaze_to_letter(0.5, 0.5, incomplete)
    assert result == "B"


# ---- Bad input rejection ---------------------------------------------
@pytest.mark.parametrize(
    "gx, gy",
    [
        (float("nan"), 0.5),
        (0.5, float("nan")),
        (float("inf"), 0.5),
        (0.5, float("-inf")),
    ],
)
def test_non_finite_gaze_raises(nominal_centroids, gx, gy):
    """The classifier MUST reject non-finite gaze inputs loudly — the
    upstream sample filter is supposed to catch these, and silent
    coercion would mask a real bug in the gaze pipeline."""
    with pytest.raises(ValueError):
        classify_gaze_to_letter(gx, gy, nominal_centroids)
