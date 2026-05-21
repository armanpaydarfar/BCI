# Utils/tiagobot_gaze.py
"""
Gaze-to-letter classifier and calibration loader for the Tiagobot gaze
experiment.

Lives next to `Utils/tiagobot.py` (the Tier 1 serial layer) but is *not*
Tier 1 itself — these are pure functions over numeric data plus a NPZ
loader. The Tiagobot serial port is owned exclusively by
`Utils.tiagobot`.

Consumed by:
- `tiago_gaze_calibration_exec.py` — uses `grid_centroids_norm()` to
  position calibration targets on screen.
- `ExperimentDriver_Online_Tiagobot_Gaze.py` — uses `load_centroids()`
  + `classify_gaze_to_letter()` to map averaged gaze samples to the
  letter the user is looking at.
- `tools/gaze_to_tiago_test.py` — same surface, no-EEG bring-up loop.

Each per-letter centroid is a `(x_norm, y_norm, depth_cm)` triple.
The depth axis comes from Pupil Labs Neon vergence (computed by
`Utils.gaze.gaze_system.GazeSystem` via `vergence_depth_from_eyestate`)
and is essential when the user sits at an angle to the board: two
letters with similar scene-pixel positions can still be physically at
quite different distances, which the 3D classifier can separate.

Layout reference: `Documents/SoftwareDocs/Tiagobot_Gaze_AI_Layout.md`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


# A-I letter set. Matches the LOCATIONS keys in `Utils/tiagobot.py:66-76`.
LETTERS: str = "ABCDEFGHI"

# Default scale factor applied to the depth axis in the 3D classifier.
# Brings cm into the same numerical range as the [0, 1] normalized pixel
# axes: 10 cm of depth difference ≈ 0.1 in the combined distance. Pick
# a smaller value to weight depth less, larger to weight it more.
DEFAULT_DEPTH_WEIGHT_CM_INV: float = 0.01


def grid_centroids_norm() -> Dict[str, Tuple[float, float]]:
    """Return the nominal A-I centroids in normalized [0, 1] coordinates,
    laid out alphabetical row-major per
    `Documents/SoftwareDocs/Tiagobot_Gaze_AI_Layout.md`.

    Each letter is centred at one of (col, row) in {0.25, 0.5, 0.75}^2:

        A=(0.25,0.25)  B=(0.5,0.25)  C=(0.75,0.25)
        D=(0.25,0.50)  E=(0.5,0.50)  F=(0.75,0.50)
        G=(0.25,0.75)  H=(0.5,0.75)  I=(0.75,0.75)

    Used by the calibration script to position on-screen targets.
    Returns 2D coords because the on-screen grid is purely a visual
    prompt — depth comes from the live vergence measurement, not from
    the grid layout.
    """
    out: Dict[str, Tuple[float, float]] = {}
    for i, ch in enumerate(LETTERS):
        col = i % 3
        row = i // 3
        out[ch] = (0.25 + 0.25 * col, 0.25 + 0.25 * row)
    return out


def load_centroids(path: str | Path) -> Dict[str, Tuple[float, float, float]]:
    """Load per-letter centroids from a Tiagobot gaze calibration NPZ.

    Expected NPZ schema (produced by `tiago_gaze_calibration_exec.py`):
      - `centroids`: shape (9, 3) float — rows correspond to LETTERS in
        order, columns are (gaze_x_norm, gaze_y_norm, depth_cm). A row
        may have NaN x/y (letter omitted entirely) or finite x/y plus
        NaN depth (letter kept, but treated as 2D-only at classify time).
      - `letters`: shape (9,) bytes/str — the letter label for each
        centroid row, used to verify the row ordering.

    Args:
        path: Path to the calibration `.npz` file.

    Returns:
        A dict mapping `letter -> (x_norm, y_norm, depth_cm)`. Letters
        with NaN x or y are omitted (no usable position). Letters with
        finite x/y but NaN depth are kept — `classify_gaze_to_letter`
        treats those as 2D-only centroids and will not score them on
        the depth axis.

    Raises:
        FileNotFoundError: path does not exist.
        ValueError: NPZ schema mismatch (missing keys, wrong shape, label
            mismatch with `LETTERS`). Old (9, 2) NPZs raise here too —
            depth was added 2026-05-20 to support oblique seating angles
            and is required for new calibrations.

    Per CLAUDE.md fail-fast policy on Tier 2 paths: callers should let
    these propagate — silent fallback to nominal centroids would mask a
    real configuration error during a hardware session.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Tiagobot gaze calibration not found at {path}. Run "
            f"tiago_gaze_calibration_exec.py before launching the gaze "
            f"experiment driver."
        )

    z = np.load(path, allow_pickle=True)

    if "centroids" not in z.files:
        raise ValueError(
            f"Calibration NPZ at {path} missing 'centroids' array."
        )
    centroids = np.asarray(z["centroids"], dtype=float)
    if centroids.shape != (len(LETTERS), 3):
        raise ValueError(
            f"Calibration NPZ at {path} 'centroids' has shape "
            f"{centroids.shape}; expected ({len(LETTERS)}, 3) "
            f"[x_norm, y_norm, depth_cm]. Old (9, 2) NPZs are no longer "
            f"supported — rerun tiago_gaze_calibration_exec.py to "
            f"recapture with depth."
        )

    if "letters" in z.files:
        labels = z["letters"]
        # Accept either bytes or str entries; normalise to str.
        labels = [
            x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
            for x in labels
        ]
        if list(labels) != list(LETTERS):
            raise ValueError(
                f"Calibration NPZ at {path} letter order {labels} does "
                f"not match expected {list(LETTERS)}."
            )

    out: Dict[str, Tuple[float, float, float]] = {}
    for i, ch in enumerate(LETTERS):
        x = float(centroids[i, 0])
        y = float(centroids[i, 1])
        d = float(centroids[i, 2])
        if np.isfinite(x) and np.isfinite(y):
            # Depth may be NaN — caller (classify_gaze_to_letter) handles
            # this by dropping depth from the distance metric for this
            # centroid. Keep the letter usable on the 2D axes.
            out[ch] = (x, y, d)
    return out


def average_gaze_over_window(
    samples: Sequence[Tuple[float, float, float, float, float]],
    conf_threshold: float,
) -> Optional[Tuple[float, float, float]]:
    """Median of `(x_norm, y_norm, depth_cm)` over samples whose
    confidence (or worn flag, for the Neon realtime API path) is at or
    above `conf_threshold`.

    Args:
        samples: Iterable of `(t, x_norm, y_norm, conf_or_worn, depth_cm)`
            tuples. `conf_or_worn` is in `[0, 1]`: legacy LSL captures put
            real confidence here; the realtime-API path stores `1.0`
            when the glasses are worn and filters worn=False at capture
            time. `depth_cm` may be NaN for individual samples where the
            vergence estimate was invalid — those are dropped from the
            depth median only, not from x/y.
        conf_threshold: Minimum confidence to include. The same value
            used at calibration time should be used at runtime so the
            two are comparable.

    Returns:
        `(x_norm, y_norm, depth_cm_or_nan)` median triple, or `None` if
        zero samples passed the confidence threshold. `depth_cm` is NaN
        when no sample had a finite depth (e.g., depth_valid=False
        throughout the window) — the classifier then ignores depth for
        that trial and falls back to 2D matching.

    Mirrors the spirit of `harmony_online_control.py`'s gaze averaging
    but operates in normalized [0, 1] coordinates, uses median (more
    robust to single bad samples) rather than mean, and carries a depth
    channel for the 3D classifier.
    """
    if not samples:
        return None
    xs: list = []
    ys: list = []
    ds: list = []
    for entry in samples:
        if len(entry) < 4:
            continue
        x, y, c = entry[1], entry[2], entry[3]
        d = entry[4] if len(entry) >= 5 else float("nan")
        try:
            cf = float(c)
        except (TypeError, ValueError):
            continue
        if cf < conf_threshold:
            continue
        try:
            xf = float(x)
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(xf) and np.isfinite(yf)):
            continue
        xs.append(xf)
        ys.append(yf)
        # Depth is allowed to be NaN per-sample; we filter for finite
        # values only when computing the depth median.
        try:
            df = float(d)
        except (TypeError, ValueError):
            df = float("nan")
        if np.isfinite(df):
            ds.append(df)
    if not xs:
        return None
    depth_med = float(np.median(ds)) if ds else float("nan")
    return float(np.median(xs)), float(np.median(ys)), depth_med


def classify_gaze_to_letter(
    gx_norm: float,
    gy_norm: float,
    centroids: Dict[str, Tuple[float, float, float]],
    *,
    gaze_depth_cm: Optional[float] = None,
    depth_weight_cm_inv: float = DEFAULT_DEPTH_WEIGHT_CM_INV,
    available_letters: Optional[Sequence[str]] = None,
    max_dist_norm: Optional[float] = None,
) -> Optional[str]:
    """Return the letter whose centroid is closest to the gaze sample,
    or `None` if no centroid is within `max_dist_norm`.

    The distance metric is:

        d² = (cx - gx)² + (cy - gy)² + (w * (cz - gz))²

    where `w = depth_weight_cm_inv` brings depth from cm into the same
    numerical range as the normalized [0, 1] pixel axes. The depth term
    is dropped (per-centroid) when either the centroid's depth or the
    runtime `gaze_depth_cm` is missing — that lets us keep classifying
    when vergence is unreliable, gracefully degrading to a 2D match.

    Args:
        gx_norm, gy_norm: Averaged gaze position in normalized [0, 1]
            scene coordinates.
        centroids: Mapping `letter -> (x_norm, y_norm, depth_cm)`.
            Typically the return value of `load_centroids()`. A
            centroid with NaN depth is treated as 2D-only (the depth
            term contributes 0 to its distance, regardless of the
            runtime depth).
        gaze_depth_cm: Median vergence depth across the runtime gaze
            window. `None` or NaN disables the depth axis entirely
            (2D fallback for all centroids). Comes from
            `average_gaze_over_window()` third return value.
        depth_weight_cm_inv: Scaling for the depth axis in the distance
            metric. Default `DEFAULT_DEPTH_WEIGHT_CM_INV` = 0.01
            (so 10 cm ≈ 0.1 norm units). Pull from
            `config.TIAGOBOT_GAZE_DEPTH_WEIGHT_CM_INV` if set.
        available_letters: Optional iterable restricting which letters
            are eligible (e.g. `config.TIAGOBOT_TRAJECTORY`). Letters
            listed here that are not in `centroids` are silently skipped.
        max_dist_norm: Optional distance threshold. If set and no
            centroid is within this distance, returns `None`.

    Returns:
        The chosen letter (single character), or `None` per the rules
        above. Ties on distance go to the centroid earlier in the dict
        iteration order (deterministic given Python 3.7+ dict order).

    Raises:
        ValueError: `(gx_norm, gy_norm)` contains NaN/inf — caller bug.

    Used by `ExperimentDriver_Online_Tiagobot_Gaze.py` and
    `tools/gaze_to_tiago_test.py`.
    """
    if not (np.isfinite(gx_norm) and np.isfinite(gy_norm)):
        raise ValueError(
            f"classify_gaze_to_letter: non-finite gaze input "
            f"({gx_norm!r}, {gy_norm!r})"
        )

    if available_letters is not None:
        eligible = [ch for ch in available_letters if ch in centroids]
    else:
        eligible = list(centroids.keys())

    if not eligible:
        return None

    have_runtime_depth = (
        gaze_depth_cm is not None and np.isfinite(float(gaze_depth_cm))
    )
    runtime_depth = float(gaze_depth_cm) if have_runtime_depth else 0.0
    w = float(depth_weight_cm_inv)

    best_letter: Optional[str] = None
    best_dist = float("inf")
    for ch in eligible:
        cent = centroids[ch]
        cx, cy = float(cent[0]), float(cent[1])
        cz = float(cent[2]) if len(cent) >= 3 else float("nan")
        dxy2 = (cx - gx_norm) * (cx - gx_norm) + (cy - gy_norm) * (cy - gy_norm)
        if have_runtime_depth and np.isfinite(cz):
            dz = w * (cz - runtime_depth)
            d = float(np.sqrt(dxy2 + dz * dz))
        else:
            d = float(np.sqrt(dxy2))
        if d < best_dist:
            best_dist = d
            best_letter = ch

    if max_dist_norm is not None and best_dist > float(max_dist_norm):
        return None
    return best_letter


# =====================================================================
# Mahalanobis classifier (added 2026-05-20 after the centroid classifier
# topped out at ~50% on an oblique seating angle). Uses the full
# per-letter sample distribution from the calibration NPZ instead of
# collapsing each letter to a single median, so axes the calibration
# was noisy on contribute less to the per-letter distance — addresses
# the "depth_weight_cm_inv is a hand-tuned global scalar" weakness of
# the centroid classifier.
# =====================================================================

# Tikhonov-style regularization added to per-letter sample covariances
# before inversion. Prevents `np.linalg.inv` blowing up when a
# calibration target's samples all land within sub-pixel jitter (rank-
# deficient cov). 1e-5 is small relative to the typical normalized-pixel
# variance (~1e-3) and the depth variance in cm² (~10² scaled), so the
# regularization changes the Mahalanobis distance by <1% for healthy
# calibrations but keeps degenerate ones finite.
_MAHAL_REGULARIZATION: float = 1e-5

# Minimum samples per letter required to fit a covariance. With <3 the
# 3D covariance is singular under any sensible regularization; with <2
# even the 2D cov is. Letters below the threshold are omitted from the
# classifier model and silently skipped at classify time (same contract
# as missing-letter centroids in `load_centroids`).
_MAHAL_MIN_SAMPLES: int = 3


def load_calibration_samples(
    path: str | Path,
) -> Dict[str, Dict[str, Any]]:
    """Load per-letter sample distributions and fit a Mahalanobis model
    for each letter.

    Reads the full per-sample data stored in the NPZ's ``G`` and
    ``labels`` arrays (saved by
    ``tiago_gaze_calibration_exec._save_calibration``):

      - ``G``: shape ``(N, 4)`` — ``[x_norm, y_norm, worn_flag, depth_cm]``.
      - ``labels``: shape ``(N,)`` bytes — the letter each row belongs
        to (``"A"`` … ``"I"``).

    Per letter, computes:
      - ``mean``: 3-vector ``[median(x), median(y), median(depth_valid)]``
        — depth median over finite samples only (NaN if no finite depth).
      - ``cov_inv_2d``: ``(2, 2)`` precision matrix for ``(x, y)``.
      - ``cov_inv_3d``: ``(3, 3)`` precision matrix for ``(x, y, depth)``,
        or ``None`` if fewer than ``_MAHAL_MIN_SAMPLES`` samples had a
        finite depth (depth Mahalanobis is unusable for that letter, but
        the 2D Mahalanobis still works).
      - ``n_samples``, ``n_samples_with_depth``: diagnostics.

    Letters with fewer than ``_MAHAL_MIN_SAMPLES`` samples overall are
    omitted from the returned dict.

    Args:
        path: Path to the calibration `.npz` file.

    Returns:
        Mapping ``letter -> model_dict``. Pass this to
        ``classify_gaze_mahalanobis()``.

    Raises:
        FileNotFoundError: path does not exist.
        ValueError: NPZ missing the ``G`` or ``labels`` arrays (e.g., an
            older calibration NPZ — those don't store per-sample data
            and can't be used with the Mahalanobis classifier).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Tiagobot gaze calibration not found at {path}. Run "
            f"tiago_gaze_calibration_exec.py before launching the gaze "
            f"experiment driver."
        )
    z = np.load(path, allow_pickle=True)
    for required in ("G", "labels"):
        if required not in z.files:
            raise ValueError(
                f"Calibration NPZ at {path} missing '{required}' array — "
                f"the Mahalanobis classifier needs the per-sample data, "
                f"not just the centroids."
            )
    G = np.asarray(z["G"], dtype=float)
    if G.ndim != 2 or G.shape[1] < 4:
        raise ValueError(
            f"Calibration NPZ at {path} 'G' has shape {G.shape}; expected "
            f"(N, 4) [x_norm, y_norm, worn_flag, depth_cm]."
        )
    labels_arr = z["labels"]
    labels = [
        (x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x))
        for x in labels_arr
    ]
    if len(labels) != G.shape[0]:
        raise ValueError(
            f"Calibration NPZ at {path}: 'labels' length {len(labels)} "
            f"does not match 'G' row count {G.shape[0]}."
        )

    out: Dict[str, Dict[str, Any]] = {}
    for ch in LETTERS:
        mask = np.array([lab == ch for lab in labels], dtype=bool)
        n = int(mask.sum())
        if n < _MAHAL_MIN_SAMPLES:
            continue
        block = G[mask]
        xs = block[:, 0]
        ys = block[:, 1]
        ds = block[:, 3]
        finite_d = np.isfinite(ds)
        n_d = int(finite_d.sum())

        # Robust per-letter "center" — medians on each axis. We pair
        # this with the covariance computed about the *mean* (np.cov
        # uses sample mean), which slightly biases the Mahalanobis
        # distance for skewed distributions but keeps the model simple.
        # If skew turns out to matter, swap to deviation about median.
        mean_x = float(np.median(xs))
        mean_y = float(np.median(ys))
        mean_d = float(np.median(ds[finite_d])) if n_d else float("nan")

        # 2D covariance + precision.
        xy = np.column_stack([xs, ys])
        cov_2d = np.cov(xy, rowvar=False)
        cov_2d_inv = np.linalg.inv(cov_2d + _MAHAL_REGULARIZATION * np.eye(2))

        # 3D covariance + precision, only if we have enough valid-depth
        # samples to fit one. With <3 samples the 3D cov is singular
        # even after regularization; the classifier falls back to 2D
        # for this letter when at classify time.
        cov_3d_inv: Optional[np.ndarray] = None
        if n_d >= _MAHAL_MIN_SAMPLES:
            xyz = np.column_stack([xs[finite_d], ys[finite_d], ds[finite_d]])
            cov_3d = np.cov(xyz, rowvar=False)
            cov_3d_inv = np.linalg.inv(cov_3d + _MAHAL_REGULARIZATION * np.eye(3))

        out[ch] = {
            "mean": np.array([mean_x, mean_y, mean_d], dtype=float),
            "cov_inv_2d": cov_2d_inv,
            "cov_inv_3d": cov_3d_inv,
            "n_samples": n,
            "n_samples_with_depth": n_d,
        }
    return out


def classify_gaze_mahalanobis(
    gx_norm: float,
    gy_norm: float,
    letter_models: Dict[str, Dict[str, Any]],
    *,
    gaze_depth_cm: Optional[float] = None,
    available_letters: Optional[Sequence[str]] = None,
    max_mahal_dist: Optional[float] = None,
) -> Optional[str]:
    """Return the letter whose Mahalanobis distance to the runtime gaze
    sample is smallest, or ``None`` if no letter is within
    ``max_mahal_dist``.

    The Mahalanobis distance uses each letter's own sample covariance
    (fit by ``load_calibration_samples``) so axes that the calibration
    was noisy on (e.g., depth jitter for a particular letter) get
    weighted less. This is the principled replacement for the global
    ``depth_weight_cm_inv`` scalar used by ``classify_gaze_to_letter``.

    Per-letter fallback: when either the runtime depth is missing or the
    letter's 3D covariance was unfittable (too few valid-depth
    calibration samples), the classifier computes the 2D Mahalanobis
    distance for that letter only. Letters with usable depth still use
    the full 3D distance — so a single letter dropping to 2D doesn't
    degrade the others.

    Args:
        gx_norm, gy_norm: Averaged gaze position in normalized [0, 1]
            scene coordinates.
        letter_models: Output of ``load_calibration_samples()``.
        gaze_depth_cm: Median vergence depth from the runtime window.
            ``None`` or NaN forces 2D fallback for every letter.
        available_letters: Optional iterable restricting which letters
            are eligible (e.g. ``config.TIAGOBOT_TRAJECTORY``).
        max_mahal_dist: Optional ceiling on the Mahalanobis distance of
            the best match. Useful as a sanity-check ("nothing was
            plausible"); ``None`` always picks the closest letter.

            Interpretation: under Gaussian assumptions and 3 DOF,
            chi-squared critical values give ~7.81 for 95% and ~11.34
            for 99%. With 2 DOF (fallback), ~5.99 / ~9.21. Pick higher
            if you'd rather always commit to a letter.

    Returns:
        The chosen letter, or ``None`` per the rules above. Tie-break is
        dict insertion order (deterministic per Python 3.7+).

    Raises:
        ValueError: ``(gx_norm, gy_norm)`` contains NaN/inf.
    """
    if not (np.isfinite(gx_norm) and np.isfinite(gy_norm)):
        raise ValueError(
            f"classify_gaze_mahalanobis: non-finite gaze input "
            f"({gx_norm!r}, {gy_norm!r})"
        )

    if available_letters is not None:
        eligible = [ch for ch in available_letters if ch in letter_models]
    else:
        eligible = list(letter_models.keys())

    if not eligible:
        return None

    have_runtime_depth = (
        gaze_depth_cm is not None and np.isfinite(float(gaze_depth_cm))
    )
    runtime_depth = float(gaze_depth_cm) if have_runtime_depth else None

    best_letter: Optional[str] = None
    best_dist = float("inf")
    for ch in eligible:
        m = letter_models[ch]
        mu = m["mean"]
        # Use 3D iff: runtime depth present, letter has 3D model, mean
        # depth is finite (the loader stores NaN if all calibration
        # samples for this letter lacked depth).
        use_3d = (
            have_runtime_depth
            and m["cov_inv_3d"] is not None
            and np.isfinite(mu[2])
        )
        if use_3d:
            assert runtime_depth is not None  # narrowed by have_runtime_depth
            v = np.array(
                [gx_norm - mu[0], gy_norm - mu[1], runtime_depth - mu[2]],
                dtype=float,
            )
            cov_inv = m["cov_inv_3d"]
        else:
            v = np.array([gx_norm - mu[0], gy_norm - mu[1]], dtype=float)
            cov_inv = m["cov_inv_2d"]
        d2 = float(v @ cov_inv @ v)
        if d2 < 0:
            # Numerical: regularized cov_inv may give a tiny negative
            # quadratic form on near-degenerate inputs. Treat as 0.
            d2 = 0.0
        d = float(np.sqrt(d2))
        if d < best_dist:
            best_dist = d
            best_letter = ch

    if max_mahal_dist is not None and best_dist > float(max_mahal_dist):
        return None
    return best_letter


def classify_gaze_cloud_vote(
    samples: Sequence[Tuple[float, float, float, float, float]],
    letter_models: Dict[str, Dict[str, Any]],
    *,
    conf_threshold: float = 0.0,
    available_letters: Optional[Sequence[str]] = None,
    max_mahal_dist: Optional[float] = None,
) -> Tuple[Optional[str], Dict[str, int]]:
    """Classify each runtime gaze sample individually via
    `classify_gaze_mahalanobis`, then return the letter with the most
    per-sample votes — and the full vote distribution.

    Trades the single-median classifier's compactness for robustness:
    a brief gaze excursion (a few off-target samples) can flip the
    median's letter but can't flip a 120-sample vote. Also surfaces the
    runtime gaze's *spread* via the returned vote-distribution dict, so
    the caller can log "92% A, 8% B" rather than just "A".

    Args:
        samples: Iterable of `(t, x_norm, y_norm, conf_or_worn, depth_cm)`
            tuples produced by the runtime selection window. Each row
            is classified independently. Samples below `conf_threshold`
            are skipped (no vote cast). Per-sample NaN depth triggers
            this letter's 2D fallback for that sample only.
        letter_models: Output of `load_calibration_samples()`.
        conf_threshold: Per-sample minimum confidence. Realtime API
            captures store `1.0` for worn samples and discard the rest
            upstream, so 0.0 is the natural default here.
        available_letters: Optional iterable restricting which letters
            can win.
        max_mahal_dist: Forwarded to `classify_gaze_mahalanobis` for
            each per-sample classify call. A sample whose nearest
            letter exceeds this distance casts no vote.

    Returns:
        `(winning_letter, vote_counts)`:
          - `winning_letter`: the letter with the most votes, or `None`
            if no sample produced a usable classification (all below
            conf, all rejected by max_mahal_dist, or `samples` empty).
          - `vote_counts`: dict mapping letter -> integer vote count.
            Sums to ≤ len(samples). Always contains every letter that
            received ≥1 vote; letters with 0 votes are omitted (callers
            iterate the dict, not LETTERS, when logging).

        Ties on vote count go to whichever letter inserts earlier into
        `vote_counts` — which is the first sample's classification.
        Deterministic given a fixed sample order (Python 3.7+ dict).
    """
    vote_counts: Dict[str, int] = {}
    if not samples:
        return None, vote_counts

    for entry in samples:
        if len(entry) < 4:
            continue
        x, y, c = entry[1], entry[2], entry[3]
        d = entry[4] if len(entry) >= 5 else float("nan")
        try:
            cf = float(c)
        except (TypeError, ValueError):
            continue
        if cf < conf_threshold:
            continue
        try:
            xf = float(x)
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(xf) and np.isfinite(yf)):
            continue
        try:
            df = float(d)
        except (TypeError, ValueError):
            df = float("nan")
        depth_arg: Optional[float] = df if np.isfinite(df) else None

        letter = classify_gaze_mahalanobis(
            xf, yf, letter_models,
            gaze_depth_cm=depth_arg,
            available_letters=available_letters,
            max_mahal_dist=max_mahal_dist,
        )
        if letter is None:
            continue
        vote_counts[letter] = vote_counts.get(letter, 0) + 1

    if not vote_counts:
        return None, vote_counts

    # Argmax with deterministic tie-break: max() over (letter, count)
    # tuples respects iteration order on ties when count is equal —
    # but max() actually compares the second element if counts tie, so
    # we walk explicitly to keep "first to reach max wins".
    winner: Optional[str] = None
    best_count = -1
    for ch, n in vote_counts.items():
        if n > best_count:
            best_count = n
            winner = ch
    return winner, vote_counts


def per_letter_distance_breakdown(
    samples: Sequence[Tuple[float, float, float, float, float]],
    letter_models: Dict[str, Dict[str, Any]],
    *,
    conf_threshold: float = 0.0,
    available_letters: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """For each eligible letter, return the mean per-sample Mahalanobis
    distance in both 3D (uses depth) and 2D-only (ignores depth) modes,
    plus a flag for whether the 3D model exists for that letter.

    Diagnostic-only helper used by `tools/gaze_to_tiago_test.py` to
    answer the question *"is the depth axis helping or hurting on this
    trial?"* per-letter. Compare ``mean_3d`` and ``mean_2d`` for the
    winning vs runner-up letters: if ``mean_3d < mean_2d`` for the
    intended letter, depth pulled that letter closer (helped); if
    ``mean_3d > mean_2d`` for the intended letter but the opposite for
    a neighbour, depth is actively misclassifying. ``depth_delta =
    mean_3d - mean_2d`` is the convenient summary.

    Args:
        samples: Same `(t, x_norm, y_norm, conf_or_worn, depth_cm)`
            shape that `classify_gaze_cloud_vote` accepts.
        letter_models: Output of `load_calibration_samples`.
        conf_threshold: Per-sample minimum confidence. Samples below
            are skipped for every letter (consistent denominator).
        available_letters: Optional restriction on which letters to
            score. Letters not in `letter_models` are silently skipped.

    Returns:
        Mapping ``letter -> {'mean_3d', 'mean_2d', 'n_used',
        'has_3d_model'}``:
          - ``mean_3d`` (float): mean Mahalanobis distance across
            valid samples, using the 3D model when available for both
            the letter and the sample (per-sample fallback to 2D
            otherwise). If a letter has no 3D model at all, this
            equals ``mean_2d``.
          - ``mean_2d`` (float): mean 2D Mahalanobis distance across
            valid samples. Always computable.
          - ``n_used`` (int): number of samples that passed the
            confidence + finite-xy filter (denominator of the means).
          - ``has_3d_model`` (bool): whether the letter's calibration
            had enough valid-depth samples to fit a 3D covariance.
            Useful when interpreting ``depth_delta``.

        Letters with ``n_used == 0`` are omitted.
    """
    if available_letters is not None:
        eligible = [ch for ch in available_letters if ch in letter_models]
    else:
        eligible = list(letter_models.keys())
    if not eligible or not samples:
        return {}

    # Pre-filter samples once so every letter sees the same denominator.
    parsed: list = []
    for entry in samples:
        if len(entry) < 4:
            continue
        x, y, c = entry[1], entry[2], entry[3]
        d = entry[4] if len(entry) >= 5 else float("nan")
        try:
            cf = float(c)
        except (TypeError, ValueError):
            continue
        if cf < conf_threshold:
            continue
        try:
            xf = float(x)
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(xf) and np.isfinite(yf)):
            continue
        try:
            df = float(d)
        except (TypeError, ValueError):
            df = float("nan")
        parsed.append((xf, yf, df))

    if not parsed:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for ch in eligible:
        m = letter_models[ch]
        mu = m["mean"]
        cov_inv_2d = m["cov_inv_2d"]
        cov_inv_3d = m["cov_inv_3d"]
        has_3d = bool(cov_inv_3d is not None and np.isfinite(mu[2]))
        sum_2d = 0.0
        sum_3d = 0.0
        n_used = 0
        for x, y, d in parsed:
            v2 = np.array([x - mu[0], y - mu[1]], dtype=float)
            d2_sq = max(0.0, float(v2 @ cov_inv_2d @ v2))
            d2 = float(np.sqrt(d2_sq))
            sum_2d += d2
            if has_3d and np.isfinite(d):
                v3 = np.array(
                    [x - mu[0], y - mu[1], d - mu[2]], dtype=float
                )
                d3_sq = max(0.0, float(v3 @ cov_inv_3d @ v3))
                sum_3d += float(np.sqrt(d3_sq))
            else:
                # Per-sample 2D fallback — matches classify_gaze_mahalanobis.
                sum_3d += d2
            n_used += 1
        if n_used > 0:
            out[ch] = {
                "mean_3d": sum_3d / n_used,
                "mean_2d": sum_2d / n_used,
                "n_used": n_used,
                "has_3d_model": has_3d,
            }
    return out
