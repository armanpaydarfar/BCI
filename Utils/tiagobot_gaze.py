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
  + `classify_gaze_to_letter()` to map averaged gaze pixels to the
  letter the user is looking at.

Layout reference: `Documents/SoftwareDocs/Tiagobot_Gaze_AI_Layout.md`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


# A-I letter set. Matches the LOCATIONS keys in `Utils/tiagobot.py:66-76`.
LETTERS: str = "ABCDEFGHI"


def grid_centroids_norm() -> Dict[str, Tuple[float, float]]:
    """Return the nominal A-I centroids in normalized [0, 1] coordinates,
    laid out alphabetical row-major per
    `Documents/SoftwareDocs/Tiagobot_Gaze_AI_Layout.md`.

    Each letter is centred at one of (col, row) in {0.25, 0.5, 0.75}^2:

        A=(0.25,0.25)  B=(0.5,0.25)  C=(0.75,0.25)
        D=(0.25,0.50)  E=(0.5,0.50)  F=(0.75,0.50)
        G=(0.25,0.75)  H=(0.5,0.75)  I=(0.75,0.75)

    Used by the calibration script to position on-screen targets, and as
    the fallback centroid set if no calibration NPZ is supplied (degraded
    operation — accuracy will be lower than measured centroids).
    """
    out: Dict[str, Tuple[float, float]] = {}
    for i, ch in enumerate(LETTERS):
        col = i % 3
        row = i // 3
        out[ch] = (0.25 + 0.25 * col, 0.25 + 0.25 * row)
    return out


def load_centroids(path: str | Path) -> Dict[str, Tuple[float, float]]:
    """Load per-letter centroids from a Tiagobot gaze calibration NPZ.

    Expected NPZ schema (produced by `tiago_gaze_calibration_exec.py`):
      - `centroids`: shape (9, 2) float — rows correspond to LETTERS in
        order, columns are (gaze_x_norm, gaze_y_norm). May contain NaN
        rows for letters that did not collect any valid samples; those
        letters are *omitted* from the returned dict (rather than
        silently falling back to nominal positions).
      - `letters`: shape (9,) bytes/str — the letter label for each
        centroid row, used to verify the row ordering.

    Args:
        path: Path to the calibration `.npz` file.

    Returns:
        A dict mapping `letter -> (x_norm, y_norm)`, in normalized [0, 1]
        scene coordinates. Letters with no valid samples are omitted.

    Raises:
        FileNotFoundError: path does not exist.
        ValueError: NPZ schema mismatch (missing keys, wrong shape, label
            mismatch with `LETTERS`).

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
    if centroids.shape != (len(LETTERS), 2):
        raise ValueError(
            f"Calibration NPZ at {path} 'centroids' has shape "
            f"{centroids.shape}; expected ({len(LETTERS)}, 2)."
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

    out: Dict[str, Tuple[float, float]] = {}
    for i, ch in enumerate(LETTERS):
        x, y = float(centroids[i, 0]), float(centroids[i, 1])
        if np.isfinite(x) and np.isfinite(y):
            out[ch] = (x, y)
    return out


def average_gaze_over_window(
    samples: Sequence[Tuple[float, float, float, float]],
    conf_threshold: float,
) -> Optional[Tuple[float, float]]:
    """Median of `(x_norm, y_norm)` over samples whose confidence is at
    or above `conf_threshold`.

    Args:
        samples: Iterable of `(t, x_norm, y_norm, conf)` tuples (units
            match what `tiago_gaze_calibration_exec.py` captures and
            what the runtime selection window accumulates).
        conf_threshold: Minimum sample confidence to include. The same
            value used at calibration time should be used at runtime so
            the two are comparable.

    Returns:
        `(x_norm, y_norm)` median pair, or `None` if zero samples passed
        the confidence threshold. The driver translates `None` into the
        configured fallback behaviour (skip GO and log; see plan §6.3
        step 4).

    Mirrors the spirit of `harmony_online_control.py`'s gaze averaging
    but operates in normalized [0, 1] coordinates and uses median (more
    robust to single bad samples) rather than mean.
    """
    if not samples:
        return None
    xs = []
    ys = []
    for entry in samples:
        if len(entry) < 4:
            continue
        _, x, y, c = entry[0], entry[1], entry[2], entry[3]
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
    if not xs:
        return None
    return float(np.median(xs)), float(np.median(ys))


def classify_gaze_to_letter(
    gx_norm: float,
    gy_norm: float,
    centroids: Dict[str, Tuple[float, float]],
    *,
    available_letters: Optional[Sequence[str]] = None,
    max_dist_norm: Optional[float] = None,
) -> Optional[str]:
    """Return the letter whose centroid is closest to `(gx_norm, gy_norm)`
    in Euclidean distance, or `None` if no centroid is within
    `max_dist_norm`.

    Args:
        gx_norm, gy_norm: Averaged gaze position in normalized [0, 1]
            scene coordinates.
        centroids: Mapping `letter -> (x_norm, y_norm)`. Typically the
            return value of `load_centroids()`.
        available_letters: Optional iterable restricting which letters
            are eligible (e.g. `config.TIAGOBOT_TRAJECTORY` to honour a
            session-level subset). If None, all keys of `centroids` are
            eligible. Letters listed here that are not in `centroids`
            are silently skipped.
        max_dist_norm: Optional Euclidean-distance threshold in
            normalized units. If set and no centroid is within this
            distance, returns `None`. If None, the nearest letter is
            always returned (so the only way to get `None` is if
            `centroids` is empty after applying `available_letters`).

    Returns:
        The chosen letter (single character), or `None` per the rules
        above. Boundary case: when two centroids tie on distance, the
        one earlier in the iteration order of `centroids` wins
        (deterministic given a fixed dict insertion order — Python 3.7+).

    Raises:
        ValueError: `(gx_norm, gy_norm)` contains NaN/inf — calibration
            input is malformed; caller should not have called us with
            this input.

    Used by `ExperimentDriver_Online_Tiagobot_Gaze.py` to replace the
    `random.choice(config.TIAGOBOT_TRAJECTORY)` line at
    `ExperimentDriver_Online_Tiagobot.py:449`.
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

    target = np.array([gx_norm, gy_norm], dtype=float)
    best_letter: Optional[str] = None
    best_dist = float("inf")
    for ch in eligible:
        cx, cy = centroids[ch]
        d = float(np.hypot(cx - target[0], cy - target[1]))
        if d < best_dist:
            best_dist = d
            best_letter = ch

    if max_dist_norm is not None and best_dist > float(max_dist_norm):
        return None
    return best_letter
