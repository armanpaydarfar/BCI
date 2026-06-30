"""
test_calibration_mapping_3d.py — WS-4 first-pass: GazeCalibration3D lookup.

Hardware-free; the 3-D mapping is a Euclidean nearest-neighbour over a small
synthetic ``X``/``Q`` library plus the V2/V3 workspace clamp. Mirrors the V3
tests in ``test_gaze_calibration_mapping.py``.

Citations under test:
  - Utils/gaze/calibration_mapping_3d.py GazeCalibration3D.__init__
  - Utils/gaze/calibration_mapping_3d.py query_xyz
  - Utils/gaze/calibration_mapping.py WORKSPACE_BOUNDS_MARGIN (reused clamp)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from Utils.gaze.calibration_mapping import (  # noqa: E402
    GazeMappingResult,
    WORKSPACE_BOUNDS_MARGIN,
)
from Utils.gaze.calibration_mapping_3d import GazeCalibration3D  # noqa: E402


class _ShimNpz(dict):
    """dict-with-`.files`, so synthetic data flows through the same code path
    an ``np.load(...)`` NpzFile would."""
    @property
    def files(self):
        return list(self.keys())


def _make_library(N: int = 40, *, seed: int = 0) -> _ShimNpz:
    """Synthetic 3-D library: random EE positions (mm) paired with small joint
    vectors (rad). Each row is its own nearest neighbour."""
    rng = np.random.default_rng(seed)
    data = _ShimNpz()
    data["X"] = rng.uniform(-300.0, 300.0, size=(N, 3))
    data["Q"] = rng.standard_normal((N, 7)) * 0.3
    return data


def test_exact_query_returns_that_row():
    """A query exactly on a library point returns that row, dist≈0, and that
    row's (unclamped-where-interior) joint vector."""
    data = _make_library()
    m = GazeCalibration3D(data)
    target = 17
    res = m.query_xyz(data["X"][target])
    assert isinstance(res, GazeMappingResult)
    assert res.idx == target
    assert res.dist < 1e-9
    np.testing.assert_allclose(res.x_target, data["X"][target], atol=1e-9)


def test_distance_is_euclidean_3d():
    """``dist`` is the true 3-D Euclidean distance to the matched point — a
    far-fixation gate depends on this. Probe a point offset from a known row by
    a vector whose neighbours are all farther."""
    # Three widely separated points so the NN is unambiguous.
    X = np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0]])
    Q = np.zeros((3, 7))
    m = GazeCalibration3D.from_arrays(X, Q)
    offset = np.array([3.0, 4.0, 12.0])  # |offset| = 13
    res = m.query_xyz(X[0] + offset)
    assert res.idx == 0
    assert abs(res.dist - 13.0) < 1e-9


def test_nn_picks_nearest_among_many():
    res_idx = []
    data = _make_library(seed=3)
    m = GazeCalibration3D(data)
    rng = np.random.default_rng(99)
    for _ in range(50):
        p = rng.uniform(-300.0, 300.0, size=3)
        expected = int(np.argmin(np.linalg.norm(data["X"] - p[None, :], axis=1)))
        res = m.query_xyz(p)
        res_idx.append((res.idx, expected))
    assert all(a == b for a, b in res_idx)


def test_workspace_clamp_matches_v3_semantics():
    """The clamp envelope is ``[Qmin - margin*span, Qmax + margin*span]`` with
    ``WORKSPACE_BOUNDS_MARGIN`` — identical rule to V2/V3. As in V3, a ``query``
    result is always a library row and thus inside its own envelope, so the flag
    stays False; the clip/violation mechanism is exercised on the helper directly
    with a constructed out-of-envelope vector."""
    X = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    Q = np.array([[0.0] * 7, [1.0] * 7])
    m = GazeCalibration3D.from_arrays(X, Q)
    q_lo, q_hi = m.workspace_bounds

    # Bounds: min/max over rows widened by the margin (span = 1 → ±0.05).
    q_min, q_max = Q.min(axis=0), Q.max(axis=0)
    span = q_max - q_min
    np.testing.assert_allclose(q_lo, q_min - WORKSPACE_BOUNDS_MARGIN * span)
    np.testing.assert_allclose(q_hi, q_max + WORKSPACE_BOUNDS_MARGIN * span)
    np.testing.assert_allclose(q_lo, [-0.05] * 7)
    np.testing.assert_allclose(q_hi, [1.05] * 7)

    # A library row stays inside the envelope → not clamped (same as V3).
    res = m.query_xyz(X[0])
    assert res.clamped is False
    assert int(res.clamp_violations.sum()) == 0

    # Helper clips an out-of-envelope vector and flags the right joints.
    over = np.array([2.0, -1.0, 0.5, 0.5, 0.5, 0.5, 0.5])
    clipped, viol = m._apply_workspace_bounds(over)
    assert clipped[0] == pytest.approx(q_hi[0])   # 2.0 → 1.05
    assert clipped[1] == pytest.approx(q_lo[1])   # -1.0 → -0.05
    np.testing.assert_allclose(clipped[2:], over[2:])
    np.testing.assert_array_equal(viol, [1, 1, 0, 0, 0, 0, 0])


def test_from_arrays_equivalent_to_npz():
    data = _make_library(seed=7)
    a = GazeCalibration3D(data)
    b = GazeCalibration3D.from_arrays(data["X"], data["Q"])
    p = data["X"][5] + np.array([1.0, -2.0, 0.5])
    ra, rb = a.query_xyz(p), b.query_xyz(p)
    assert ra.idx == rb.idx
    np.testing.assert_allclose(ra.q_target, rb.q_target)


def test_non_finite_rows_dropped():
    """A dry-run sweep leaves Q NaN / the depth-lift leaves X NaN; those rows
    are dropped, and the surviving ``idx`` indexes back into the ORIGINAL table."""
    X = np.array([[0.0, 0.0, 0.0],
                  [np.nan, 0.0, 0.0],   # bad X
                  [50.0, 0.0, 0.0]])
    Q = np.array([[0.0] * 7,
                  [0.1] * 7,
                  [0.2] * 7])
    Q[2, 3] = np.nan                     # also make a bad-Q row
    Q = np.vstack([Q, [0.3] * 7])
    X = np.vstack([X, [99.0, 0.0, 0.0]])
    m = GazeCalibration3D.from_arrays(X, Q)
    assert m.num_valid_samples == 2      # row 0 and row 3 survive
    res = m.query_xyz([99.0, 0.0, 0.0])
    assert res.idx == 3                  # global index preserved


def test_missing_keys_and_bad_shapes_raise():
    with pytest.raises(KeyError):
        GazeCalibration3D({"UV": np.zeros((3, 2)), "Q": np.zeros((3, 7))})
    with pytest.raises(ValueError):
        GazeCalibration3D.from_arrays(np.zeros((3, 2)), np.zeros((3, 7)))
    with pytest.raises(ValueError):
        GazeCalibration3D.from_arrays(np.zeros((3, 3)), np.zeros((3, 6)))
    with pytest.raises(ValueError):
        GazeCalibration3D.from_arrays(np.zeros((3, 3)), np.zeros((4, 7)))


def test_non_finite_query_raises():
    m = GazeCalibration3D(_make_library())
    with pytest.raises(ValueError):
        m.query_xyz([0.0, np.nan, 0.0])
