"""
test_world_map_ba.py — WS-4 Tier-2 constrained bundle adjustment.

Fast tests pin the pure geometry helpers; the full BA convergence (which runs a
real least_squares and takes seconds) is the module's ``_self_test`` and is marked
slow so the fast pre-commit suite skips it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Analyze_world_map_ba import (  # noqa: E402
    _angles_from_normal,
    _inv_rodrigues,
    _normal_from_angles,
    _rodrigues,
    tag_object_corners,
)


def test_rodrigues_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(6):
        rv = rng.normal(0, 0.3, 3)           # |rv| < pi so the map is invertible
        np.testing.assert_allclose(_inv_rodrigues(_rodrigues(rv)), rv, atol=1e-7)


def test_normal_angles_roundtrip():
    for n in ([0, 0, 1.0], [1, 0, 0.0], [0, 1, 0.0], [0.3, 0.4, 0.866]):
        n = np.asarray(n, float); n = n / np.linalg.norm(n)
        th, ph = _angles_from_normal(n)
        np.testing.assert_allclose(_normal_from_angles(th, ph), n, atol=1e-9)


def test_tag_object_corners():
    c = tag_object_corners(0.04, 0)
    assert c.shape == (4, 3)
    assert np.allclose(c[:, 2], 0.0)
    assert np.allclose(np.abs(c[:, :2]), 20.0)     # s/2 = 20 mm for a 40 mm tag
    # a different order is a permutation of the same corner set
    assert {tuple(r) for r in tag_object_corners(0.04, 2)} == {tuple(r) for r in c}


@pytest.mark.slow
def test_ba_converges_flattens_squares():
    from Analyze_world_map_ba import _self_test
    assert _self_test() == 0
