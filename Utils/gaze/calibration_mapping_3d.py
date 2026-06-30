"""
calibration_mapping_3d.py — WS-4 first-pass: 3-D gaze→joint-target lookup.

This is the 3-D sibling of ``GazeCalibrationMappingV3`` in
``calibration_mapping.py``. V3 (REV04 planar) maps a table-plane ``(u,v)``
query to the recorded joint vector ``Q`` via a 2-D nearest-neighbour — it
collapses the workspace onto one plane because the REV04 calibration sweep
and the runtime gaze hit are both *on the table*. WS-4 lifts that restriction:
the query is a full 3-D point ``p_xyz`` (mm) and the library is ``(N,3)``
points → clamped ``Q`` via a plain Euclidean 3-D nearest-neighbour.

**Frame is set by the constructor, not by this class** (the NN math is
frame-agnostic — query and library must simply share a frame, mm):
  - :meth:`from_calib_npz` — the **live WS-4 path**. Loads ``P_WORLD3D`` (the
    EE point in the **WORLD** frame, un-projected) written by
    ``apriltag_calibrate.py``'s ``solve-3d`` stage. The runtime feeds a
    world-frame gaze→object point (``apriltag_control_test_3d.object_point_world_mm``),
    so library and query are both world-frame mm.
  - ``__init__`` / :meth:`from_arrays` with the sweep's ``X (N,3)`` — a
    diagnostic/secondary path. ``X`` is the robot EE position in the robot
    **BASE** frame (mm); a query against it must therefore also be base-frame.
    Do **not** mix: a world-frame query against a base-frame ``X`` library is a
    silent frame error.

Where the 3-D world query comes from (the live loop, already built): gaze-ray ∩
object — WS-1's segmented-object centroid lifted into 3-D (depth at the fixated
pixel), expressed in the WORLD frame, NOT dropped onto the table plane. The
remaining WS-4 work is the 3-D coverage UI and rig validation; this module is
the lookup/clamp building block, validated by unit tests.

Frozen-core note (WS-4 constraint): this is NEW SURFACE. It does not edit
the REV05 2-D core. It imports ``GazeMappingResult`` and
``WORKSPACE_BOUNDS_MARGIN`` from ``calibration_mapping`` so the result schema
and the clamp envelope are identical to V3 — there is exactly one workspace
clamp definition in the codebase.

Single-responsibility helper per CLAUDE.md; mirrors V3's structure and
conventions (same result dataclass, same clamp semantics, same float64
SPD-safe arithmetic).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from Utils.gaze.calibration_mapping import (
    GazeMappingResult,
    WORKSPACE_BOUNDS_MARGIN,
)


class GazeCalibration3D:
    """3-D nearest-neighbour mapping: a calibration library of 3-D points
    ``(N,3)`` (mm) → the recorded joint vectors ``Q (N,7)`` (rad),
    workspace-clamped. The point **frame is whatever the constructor loaded**
    (see the module docstring): :meth:`from_calib_npz` → WORLD frame
    (``P_WORLD3D``, the live path); ``__init__``/:meth:`from_arrays` with the
    sweep ``X`` → robot BASE frame (diagnostic). Query and library must share a
    frame — the NN itself is frame-agnostic.

    Mirrors ``GazeCalibrationMappingV3`` exactly, with the single difference
    that the index is over 3-D points (Euclidean NN) rather than the
    table-plane ``(u,v)``.

    The workspace clamp is the same envelope as V2/V3 (``Q`` min/max over the
    valid rows, widened by ``WORKSPACE_BOUNDS_MARGIN``) and is the only
    joint-safety guard, since the robot enforces none. Out-of-bounds commands
    are surfaced via the ``clamped`` flag rather than silently rewritten.
    """

    def __init__(self, npz_data: Dict[str, np.ndarray]) -> None:
        """Build the index from a sweep NPZ (``X``/``Q``).

        Accepts an ``np.load(...)`` object or a plain dict (anything with the
        keys, optionally exposing ``.files`` like an ``NpzFile``). Rows with a
        non-finite ``X`` or ``Q`` are dropped — a no-robot dry-run sweep
        leaves ``Q`` NaN, and the depth-lift may leave a sample's ``X`` NaN.

        Args:
            npz_data: dict-like with ``X (N,3)`` (mm, robot BASE frame) and
                ``Q (N,7)`` (rad). ``X`` may have >3 columns; only the first
                three (position) are indexed.

        Raises:
            KeyError: ``X`` or ``Q`` missing (this is not a sweep NPZ).
            ValueError: malformed shapes, length mismatch, or zero finite rows.
        """
        keys = npz_data.files if hasattr(npz_data, "files") else npz_data
        for key in ("X", "Q"):
            if key not in keys:
                raise KeyError(
                    f"3-D mapping requires NPZ key {key!r}; this looks like a "
                    "non-sweep calibration (expected the apriltag_sweep_*.npz "
                    "schema with X/Q columns)."
                )
        # float64 throughout (np default): the clamp envelope and the NN
        # distance feed joint commands; keep the SPD-adjacent arithmetic at
        # full precision per CLAUDE.md "Numerical code warrants extra care".
        X = np.asarray(npz_data["X"], dtype=float)
        Q = np.asarray(npz_data["Q"], dtype=float)
        if X.ndim != 2 or X.shape[1] < 3:
            raise ValueError(f"X must be (N, 3); got shape={X.shape}")
        if Q.ndim != 2 or Q.shape[1] < 7:
            raise ValueError(f"Q must be (N, 7); got shape={Q.shape}")
        if X.shape[0] != Q.shape[0]:
            raise ValueError(f"X length {X.shape[0]} != Q length {Q.shape[0]}")

        valid = np.all(np.isfinite(X[:, :3]), axis=1) & np.all(
            np.isfinite(Q[:, :7]), axis=1)
        if not np.any(valid):
            raise ValueError("3-D NPZ has zero rows with finite (X, Q).")
        self._valid_indices = np.flatnonzero(valid)
        self._X = X[valid, :3]
        self._Q = Q[valid, :7]

        # Workspace clamp envelope — identical rule to V2/V3 (same margin
        # constant, imported so there is one definition).
        q_min = np.min(self._Q, axis=0)
        q_max = np.max(self._Q, axis=0)
        span = q_max - q_min
        self._q_lo = q_min - WORKSPACE_BOUNDS_MARGIN * span
        self._q_hi = q_max + WORKSPACE_BOUNDS_MARGIN * span

    @classmethod
    def from_arrays(cls, X: np.ndarray, Q: np.ndarray) -> "GazeCalibration3D":
        """Build directly from in-memory ``X (N,3)`` / ``Q (N,7)`` arrays,
        bypassing an NPZ on disk (synthetic libraries, tests, and callers that
        already hold the matrices). Delegates to ``__init__`` so the validation
        and clamp setup stay in one place."""
        return cls({"X": np.asarray(X), "Q": np.asarray(Q)})

    @classmethod
    def from_calib_npz(cls, npz_data: Dict[str, np.ndarray]) -> "GazeCalibration3D":
        """Build the 3-D library from a WS-4 ``solve-3d`` calibration NPZ
        (``tools/apriltag_calibrate.py --stage solve-3d``).

        That calib stores the library points under ``P_WORLD3D`` — the EE origin
        in the WORLD frame in FULL 3-D (mm), i.e. the same ``p_world`` the planar
        solve computes but WITHOUT the table-plane projection, so the table-height
        ``z`` is retained. The key is deliberately distinct from a sweep's ``X``
        (robot BASE-frame telemetry): the runtime queries the gaze→object centroid
        in the SAME world frame, so there is no base-frame / hand-eye transform and
        the two point spaces must never be confused. Reading a dedicated key keeps
        that contract explicit rather than overloading ``X``.

        Args:
            npz_data: dict-like (an ``np.load(...)`` result or a plain dict) with
                ``P_WORLD3D (N,3)`` (mm, world frame) and ``Q (N,7)`` (rad).

        Returns:
            A ``GazeCalibration3D`` indexed over the world-frame points.

        Raises:
            KeyError: ``P_WORLD3D`` or ``Q`` missing (not a solve-3d calib NPZ).
        """
        keys = npz_data.files if hasattr(npz_data, "files") else npz_data
        for key in ("P_WORLD3D", "Q"):
            if key not in keys:
                raise KeyError(
                    f"3-D calib requires NPZ key {key!r}; this looks like a "
                    "non-3-D calibration (expected the apriltag_3d_*_calib.npz "
                    "schema with P_WORLD3D/Q columns)."
                )
        return cls.from_arrays(npz_data["P_WORLD3D"], npz_data["Q"])

    def query_xyz(self, p_xyz) -> GazeMappingResult:
        """Nearest calibrated point to ``p_xyz`` (mm, in the library's frame) →
        its clamped joint vector. ``p_xyz`` MUST be in the same frame the library
        was built in (WORLD for :meth:`from_calib_npz`, BASE for the ``X`` path) —
        see the class docstring; the NN is frame-agnostic and will silently match
        the wrong pose if the frames differ.

        ``dist`` is the Euclidean 3-D distance (mm) to the matched library
        point — the caller's far/implausible-target gate uses it, exactly as
        V3's in-plane distance gate does. ``x_target`` is the matched library
        point (where that pose's EE actually sat); ``features`` is the query
        ``p_xyz``.

        Args:
            p_xyz: 3-vector target point (mm, library frame). Must be finite.

        Returns:
            ``GazeMappingResult`` with ``q_target`` already workspace-clamped.
        """
        p = np.asarray(p_xyz, dtype=float).reshape(3)
        if not np.all(np.isfinite(p)):
            raise ValueError(f"query p_xyz must be finite; got {p!r}")
        d = np.linalg.norm(self._X - p[None, :], axis=1)
        best_local = int(np.argmin(d))
        q_clipped, clamp_violations = self._apply_workspace_bounds(
            self._Q[best_local])
        return GazeMappingResult(
            idx=int(self._valid_indices[best_local]),
            dist=float(d[best_local]),
            q_target=q_clipped,
            x_target=self._X[best_local].copy(),
            features=p,
            clamped=bool(np.any(clamp_violations)),
            clamp_violations=clamp_violations,
        )

    @property
    def num_valid_samples(self) -> int:
        return int(self._X.shape[0])

    @property
    def workspace_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """(q_lo, q_hi) — the clamp envelope, for diagnostic logs."""
        return self._q_lo.copy(), self._q_hi.copy()

    def _apply_workspace_bounds(
            self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Clamp each joint to ``[_q_lo, _q_hi]`` (same rule as V2/V3).
        Returns the clipped vector and a per-joint violation indicator
        (1 where the joint was clipped, 0 otherwise)."""
        clipped = np.clip(q, self._q_lo, self._q_hi)
        violations = (q != clipped).astype(np.int64)
        return clipped, violations
