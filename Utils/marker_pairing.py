"""
Chronological pairing of REST/MI begin→end markers for segmentation.

Markers are processed in **time order** (sorted timestamps). For each end code (120 or 220),
the matching begin is the **oldest unmatched** start of that pair type (100 or 200). That
queue discipline is standard “FIFO” per begin-code; it is the usual way to bracket intervals
in a single timeline when trials do not overlap.

It differs from the legacy pattern: k-th start with k-th end from separate ``np.where`` lists,
which is **not** guaranteed to follow wall-clock order when REST and MI trials interleave or
when start/end counts differ—then pairs can span minutes and inflate sliding-window counts.

Also drops spans outside [min_duration_sec, max_duration_sec].
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


def build_trial_windows_chronological(
    marker_timestamps: np.ndarray,
    marker_values: np.ndarray,
    epochs_start_end: dict[Any, Any],
    *,
    max_duration_sec: float = 90.0,
    min_duration_sec: float = 1.0,
) -> tuple[list[tuple[float, float, int]], dict[str, int]]:
    """
    Build (ts_start, ts_end, start_marker_code) trials from marker streams.

    Name: "chronological" = scan events in time order; FIFO = one pending deque per
    begin code so each end closes the earliest open start of that type (100/200 are independent).

    Args:
        marker_timestamps: 1D float times (seconds).
        marker_values: 1D int marker codes (same length).
        epochs_start_end: map begin_code -> end_code (e.g. 100->120, 200->220); keys/values coerced to int.

    Returns:
        trial_windows sorted by trial start time; stats dict with skip counts.
    """
    tm = np.asarray(marker_timestamps, dtype=float).ravel()
    mv = np.asarray(marker_values, dtype=int).ravel()
    if tm.shape != mv.shape:
        raise ValueError("marker_timestamps and marker_values must have the same shape")

    epochs_int = {int(sm): int(em) for sm, em in epochs_start_end.items()}
    pending: dict[int, deque[float]] = {sm: deque() for sm in epochs_int}
    trial_windows: list[tuple[float, float, int]] = []

    stats = {
        "skipped_long_duration": 0,
        "skipped_short_duration": 0,
        "orphan_end_markers": 0,
    }

    order = np.argsort(tm, kind="mergesort")
    end_codes = set(epochs_int.values())

    for idx in order:
        v = int(mv[idx])
        t = float(tm[idx])

        if v in end_codes:
            sm_for_end = None
            for sm, em in epochs_int.items():
                if em == v:
                    sm_for_end = sm
                    break
            if sm_for_end is None:
                continue
            q = pending[sm_for_end]
            if not q:
                stats["orphan_end_markers"] += 1
                continue
            ts0 = q.popleft()
            dur = t - ts0
            if dur < min_duration_sec:
                stats["skipped_short_duration"] += 1
                continue
            if dur > max_duration_sec:
                stats["skipped_long_duration"] += 1
                continue
            trial_windows.append((ts0, t, sm_for_end))
            continue

        if v in epochs_int:
            pending[v].append(t)

    trial_windows.sort(key=lambda x: x[0])
    return trial_windows, stats
