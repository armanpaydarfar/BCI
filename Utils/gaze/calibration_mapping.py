"""
calibration_mapping.py — v2 gaze -> joint-target lookup
(plan §6.2 Option I, per Harmony_Gaze_Calibration_Upgrade_Plan.md).

The v1 mapping in ``ExperimentDriver_Online_GazeTracking.nearest_idx_gaze``
(file:609-624 of that driver) is a flat 2-D NN over normalised scene
pixels. It cannot disambiguate near vs far targets along the same line
of sight — the motivating bug for the upgrade per plan §1.

The v2 mapping is a Mahalanobis NN over a richer feature vector:

  Pass-1 (config.GAZE_CALIBRATION_USE_IMU=False, the default):
      features = (gaze_yaw_deg, gaze_pitch_deg, depth_cm)
  Pass-2 (config.GAZE_CALIBRATION_USE_IMU=True):
      features = (gaze_yaw_deg, gaze_pitch_deg, depth_cm,
                  head_yaw_deg, head_pitch_deg)

Per-feature scales are learned from the calibration NPZ via the
robust 1.4826*MAD estimator (a stddev surrogate that ignores
outliers from blinks / off-screen fixations).

This module also exposes the workspace-bounds clamp locked in
Gaze_Calibration_Sensor_Characterization.md §5: every commanded joint
vector is clipped to [Q.min(axis=0), Q.max(axis=0)] of the calibration
table with a 5% margin before being returned. Out-of-bounds commands
are surfaced via the ``clamped`` flag on ``GazeMappingResult`` rather
than silently rewritten — the driver decides whether to log a warning.

Single-responsibility helper module per CLAUDE.md "Software Development
Practices"; the v1 mapping stays exactly where it is at
``ExperimentDriver_Online_GazeTracking.py:609-624``, so flipping
``config.GAZE_CALIBRATION_VERSION`` between 1 and 2 changes which
function the driver dispatches to without touching the v1 code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


# Pass-1 features (legacy + Phase 1 doc §5 lock-in)
PASS1_FEATURE_KEYS: Tuple[str, ...] = (
    "Gaze_yaw_deg", "Gaze_pitch_deg", "D_cm",
)
# Pass-2 features add the IMU head-pose pair.
PASS2_FEATURE_KEYS: Tuple[str, ...] = PASS1_FEATURE_KEYS + (
    "Head_yaw_deg", "Head_pitch_deg",
)

# 1.4826 is the robust-std proportionality constant for MAD under a
# normal assumption (`1 / Phi^{-1}(0.75)`). Used in
# ``_robust_scale`` below.
MAD_TO_SIGMA = 1.4826

# Tiny epsilon so a near-constant feature column does not produce a
# divide-by-zero scale (a feature that never varies in the calibration
# data also has no information content; we float it but do not blow up).
_SCALE_EPS = 1e-6

# Workspace-bounds margin per Gaze_Calibration_Sensor_Characterization.md
# §5 row "§5.6 workspace bounds": clamp each joint to
# ``[Qmin - margin*(Qmax-Qmin), Qmax + margin*(Qmax-Qmin)]``. 5% is the
# locked-in value; expose as a module constant so the future C++-side
# bounds (plan §7.2) can read the same number if needed.
WORKSPACE_BOUNDS_MARGIN = 0.05


@dataclass
class GazeMappingResult:
    """Output of a single v2 lookup.

    Attributes:
        idx: row index into the (filtered) calibration table.
        dist: Mahalanobis distance from the query to the chosen sample.
        q_target: 7-DOF joint vector (radians) AFTER workspace clamp.
        x_target: 3-DOF EE position (mm) of the chosen sample (not
            clamped — the EE position is read-back-only).
        features: feature vector actually queried.
        clamped: True if the workspace-bounds clamp altered any joint.
        clamp_violations: per-joint count of clamp activations (length 7,
            int). All zeros when ``clamped=False``.
    """
    idx: int
    dist: float
    q_target: np.ndarray
    x_target: np.ndarray
    features: np.ndarray
    clamped: bool
    clamp_violations: np.ndarray


class GazeCalibrationMappingV2:
    """Fit-on-load mapping used by the v2 dispatch path.

    The NPZ is the only source of truth. Construction reads the
    v2 schema (``Gaze_yaw_deg``, ``Gaze_pitch_deg``, ``D_cm``, and the
    Pass-2 head-pose pair), fits per-feature scales, and pre-computes
    the bounds. Lookup is a hot path; the constructor pays the cost.

    The class accepts any NPZ that has the legacy ``Q`` / ``X`` arrays
    AND every key in the active feature set. Missing-feature NPZs raise
    in the constructor — the dispatch path is expected to fall back to
    v1 (legacy 2D NN) by reading ``config.GAZE_CALIBRATION_VERSION``
    before instantiating this class.
    """

    def __init__(self, npz_data: Dict[str, np.ndarray], *,
                 use_imu: bool = False,
                 require_depth_valid: bool = True) -> None:
        """Build the index from a loaded NPZ (an `np.load(...)` object
        works directly — `np.load` returns a dict-like with `.files`).

        Args:
            npz_data: dict-like with the v2 keys. ``np.load(path,
                allow_pickle=True)`` is the expected source.
            use_imu: if True, include the Pass-2 head-pose features.
                Driven by ``config.GAZE_CALIBRATION_USE_IMU`` at the
                call site.
            require_depth_valid: if True, drop calibration rows where
                ``D_valid`` is False before fitting. Pass-1 default; Pass-2
                should pass True so the head-pose features are not used
                with a NaN depth co-feature.
        """
        self._use_imu = bool(use_imu)
        self._feature_keys: Tuple[str, ...] = (
            PASS2_FEATURE_KEYS if self._use_imu else PASS1_FEATURE_KEYS
        )

        # Pull the matrices (will KeyError if v1 NPZ used with v2 path).
        Q = np.asarray(npz_data["Q"], dtype=float)
        X = np.asarray(npz_data["X"], dtype=float)
        if Q.ndim != 2 or Q.shape[1] < 7:
            raise ValueError(f"Q must be (N, 7); got shape={Q.shape}")
        if X.ndim != 2 or X.shape[1] < 3:
            raise ValueError(f"X must be (N, 3); got shape={X.shape}")

        feature_cols = []
        for key in self._feature_keys:
            if key not in npz_data.files if hasattr(npz_data, "files") else key not in npz_data:
                raise KeyError(
                    f"v2 mapping requires NPZ key {key!r}; falling back to v1 is "
                    "the caller's responsibility (set "
                    "config.GAZE_CALIBRATION_VERSION=1)."
                )
            col = np.asarray(npz_data[key], dtype=float).ravel()
            if col.shape[0] != Q.shape[0]:
                raise ValueError(
                    f"feature {key!r} length {col.shape[0]} does not match Q length {Q.shape[0]}"
                )
            feature_cols.append(col)
        F = np.column_stack(feature_cols)  # (N, len(feature_keys))

        # Optional row filter: drop rows where any feature is non-finite,
        # and (if requested) where D_valid is False.
        valid_mask = np.all(np.isfinite(F), axis=1)
        if require_depth_valid and "D_valid" in (npz_data.files if hasattr(npz_data, "files") else npz_data):
            d_valid = np.asarray(npz_data["D_valid"], dtype=bool).ravel()
            if d_valid.shape[0] == valid_mask.shape[0]:
                valid_mask &= d_valid

        if not np.any(valid_mask):
            raise ValueError("v2 mapping NPZ has zero rows with finite features.")

        self._valid_indices: np.ndarray = np.flatnonzero(valid_mask)
        self._F: np.ndarray = F[valid_mask]  # (M, len(feature_keys))
        self._Q: np.ndarray = Q[valid_mask, :7]
        self._X: np.ndarray = X[valid_mask, :3]

        # Per-feature robust scale (1.4826 * MAD). Floored by _SCALE_EPS
        # to avoid divide-by-zero on degenerate columns.
        self._scales: np.ndarray = _robust_scale(self._F)

        # Workspace bounds for the clamp. Computed from the FULL valid
        # row set, not just the filtered subset — the clamp's job is to
        # bound the *output* into the calibration envelope regardless of
        # which features matched.
        q_min = np.min(self._Q, axis=0)
        q_max = np.max(self._Q, axis=0)
        span = q_max - q_min
        self._q_lo: np.ndarray = q_min - WORKSPACE_BOUNDS_MARGIN * span
        self._q_hi: np.ndarray = q_max + WORKSPACE_BOUNDS_MARGIN * span

    # ------------------------------------------------------------------
    # Public lookup
    # ------------------------------------------------------------------
    def query(self, features: Dict[str, float]) -> GazeMappingResult:
        """Find the nearest calibration sample in Mahalanobis-scaled
        feature space and return the (clamped) joint target.

        Args:
            features: dict with the feature keys for the active pass.
                Missing keys raise; non-finite values raise.

        Returns:
            ``GazeMappingResult``. ``q_target`` is already workspace-clamped.
        """
        feature_vec = self._features_from_dict(features)
        diff = self._F - feature_vec[None, :]
        scaled = diff / self._scales[None, :]
        d2 = np.einsum("ij,ij->i", scaled, scaled)
        best_local = int(np.argmin(d2))
        idx_global = int(self._valid_indices[best_local])

        q_raw = self._Q[best_local].copy()
        q_clipped, clamp_violations = self._apply_workspace_bounds(q_raw)
        clamped = bool(np.any(clamp_violations))

        return GazeMappingResult(
            idx=idx_global,
            dist=float(np.sqrt(d2[best_local])),
            q_target=q_clipped,
            x_target=self._X[best_local].copy(),
            features=feature_vec,
            clamped=clamped,
            clamp_violations=clamp_violations,
        )

    # ------------------------------------------------------------------
    # Diagnostics — used by Phase 2.b validation and Phase 2.c tests
    # ------------------------------------------------------------------
    @property
    def feature_keys(self) -> Tuple[str, ...]:
        return self._feature_keys

    @property
    def feature_scales(self) -> np.ndarray:
        return self._scales.copy()

    @property
    def num_valid_samples(self) -> int:
        return int(self._F.shape[0])

    @property
    def workspace_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """(q_lo, q_hi) — the clamp envelope for diagnostic logs."""
        return self._q_lo.copy(), self._q_hi.copy()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _features_from_dict(self, features: Dict[str, float]) -> np.ndarray:
        vec = np.empty(len(self._feature_keys), dtype=float)
        for i, key in enumerate(self._feature_keys):
            if key not in features:
                raise KeyError(f"query missing feature {key!r}")
            val = float(features[key])
            if not np.isfinite(val):
                raise ValueError(f"feature {key!r}={val!r} is not finite")
            vec[i] = val
        return vec

    def _apply_workspace_bounds(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Clamp each joint to ``[_q_lo, _q_hi]``. Returns the clipped
        vector and a per-joint violation indicator (1 if the joint was
        clipped, 0 otherwise)."""
        clipped = np.clip(q, self._q_lo, self._q_hi)
        violations = (q != clipped).astype(np.int64)
        return clipped, violations


# --------------------------------------------------------------------
# Module-level helpers
# --------------------------------------------------------------------
def _robust_scale(F: np.ndarray) -> np.ndarray:
    """Return ``1.4826 * MAD`` per column of ``F`` with an ``_SCALE_EPS``
    floor. ``F`` shape: (N, k). Output shape: (k,)."""
    med = np.median(F, axis=0)
    abs_dev = np.abs(F - med[None, :])
    mad = np.median(abs_dev, axis=0)
    scale = MAD_TO_SIGMA * mad
    return np.maximum(scale, _SCALE_EPS)


def load_pose_library_v2(path: str) -> Dict[str, Any]:
    """Load an NPZ and return it as a dict so callers can introspect
    `files` cheaply without retaining the lazy ``npz`` object. Mirrors
    ``ExperimentDriver_Online_GazeTracking.load_pose_library`` (file:591-606)
    but exposes the full v2 key set rather than just X/Q/G.
    """
    z = np.load(path, allow_pickle=True)
    data = {key: z[key] for key in z.files}
    # `np.load`'s files list is bound to the NpzFile; we materialise so
    # callers can close the underlying mmap if any.
    z.close()
    # Replicate the .files attribute so the v2 class's "key in
    # npz_data.files" probe keeps working without callers caring about
    # whether they got a dict or an NpzFile.
    data["__files__"] = list(data.keys() - {"__files__"})
    return data


def detect_pose_library_version(npz_data: Any) -> int:
    """Return 1 or 2 based on the NPZ meta dict (preferred) or, as a
    fallback, the presence of v2 feature keys. v1 NPZs have no
    ``D_cm`` / ``Gaze_yaw_deg`` etc. and no ``meta['version']``
    explicitly (legacy writer at ``harmony_calibration_exec.py:404-407``
    embedded only ``side / sample_rate_hz / gaze_confidence_threshold /
    units``).
    """
    meta = _peek_meta(npz_data)
    if isinstance(meta, dict) and "version" in meta:
        try:
            return int(meta["version"])
        except (TypeError, ValueError):
            return 1
    # Fall back to feature-presence sniff.
    keys = _peek_keys(npz_data)
    if "D_cm" in keys and "Gaze_yaw_deg" in keys:
        return 2
    return 1


def _peek_meta(npz_data: Any) -> Optional[Dict[str, Any]]:
    keys = _peek_keys(npz_data)
    if "meta" not in keys:
        return None
    try:
        raw = npz_data["meta"]
    except (KeyError, AttributeError):
        return None
    # ``meta`` was written as a Python dict via ``np.savez_compressed``;
    # it round-trips as a 0-d object array.
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1:
        item = raw.item()
        if isinstance(item, dict):
            return item
    if isinstance(raw, dict):
        return raw
    return None


def _peek_keys(npz_data: Any) -> set:
    if hasattr(npz_data, "files"):
        return set(getattr(npz_data, "files"))
    if isinstance(npz_data, dict):
        return set(npz_data.keys())
    return set()
