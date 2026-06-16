# Vendored from harmony_vlm (https://github.com/vivianchen98/harmony_vlm) @ cfa01b6
# by Vivian Chen. Folded into the BCI repo for WS3 (2026-06-15). Edit here, not
# upstream; see Documents/SoftwareDocs/projects/harmony-bci/vlm-integration/.
"""
fixation_detector.py — Software I-VT fixation detection for Neon gaze data.

No fixations.raw exists on disk; fixations are computed here in software
using a velocity-threshold (I-VT) algorithm with a stable-window trigger.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from perception.pupil_reader import GazeSample

# ── tuning constants ──────────────────────────────────────────────────────────

VELOCITY_THRESHOLD_PX  = 30.0   # px/frame at 33 Hz  (~100°/s)
MIN_FIXATION_SAMPLES   = 5      # 150 ms minimum
STABLE_WINDOW_SAMPLES  = 16     # 500 ms stable window before trigger
STABLE_DRIFT_PX        = 40.0   # max centroid drift allowed in stable window
BLINK_APERTURE_MM      = 1.5    # eyelid threshold for blink suppression


# ── output dataclass ──────────────────────────────────────────────────────────

@dataclass
class FixationState:
    active:       bool  = False
    is_stable:    bool  = False   # stable window full + low drift
    centroid_x:   float = 0.0
    centroid_y:   float = 0.0
    start_ts_ns:  int   = 0       # timestamp of first sample in current fixation
    duration_ns:  int   = 0       # span from start to latest sample
    sample_count: int   = 0
    max_drift_px: float = 0.0     # max sample distance from centroid in window


# ── detector ─────────────────────────────────────────────────────────────────

class FixationDetector:
    """
    Rolling I-VT fixation detector.

    Call update(gaze_sample) on every new gaze sample (in order).
    Returns a FixationState describing the current fixation status.
    """

    def __init__(self) -> None:
        self._window: deque[GazeSample] = deque(maxlen=STABLE_WINDOW_SAMPLES)
        self._prev:   Optional[GazeSample] = None
        self._start:  Optional[GazeSample] = None  # first sample of current fixation
        self._count:  int = 0                       # samples in current fixation

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, sample: GazeSample) -> FixationState:
        """Process one gaze sample and return the current FixationState."""

        # ── blink suppression ─────────────────────────────────────────────
        eye = sample.eye_state
        ea_l = float(eye["eyelid_aperture_left_mm"])
        ea_r = float(eye["eyelid_aperture_right_mm"])
        if ea_l < BLINK_APERTURE_MM or ea_r < BLINK_APERTURE_MM:
            # Pause: don't update state, don't reset
            return self._build_state()

        # ── velocity check ────────────────────────────────────────────────
        if self._prev is not None:
            dx = sample.x - self._prev.x
            dy = sample.y - self._prev.y
            velocity = math.hypot(dx, dy)

            if velocity > VELOCITY_THRESHOLD_PX:
                # Saccade detected — reset fixation
                self._reset()
                self._prev = sample
                return FixationState(active=False)

        # ── accumulate fixation ───────────────────────────────────────────
        if self._start is None:
            self._start = sample

        self._count += 1
        self._window.append(sample)
        self._prev = sample

        return self._build_state()

    def reset(self) -> None:
        """Force a full reset (e.g. when worn==False)."""
        self._reset()

    # ── private ───────────────────────────────────────────────────────────────

    def _reset(self) -> None:
        self._window.clear()
        self._prev  = None
        self._start = None
        self._count = 0

    def _build_state(self) -> FixationState:
        if self._start is None or self._count < MIN_FIXATION_SAMPLES:
            return FixationState(active=False)

        latest = self._window[-1] if self._window else self._start
        duration_ns = latest.timestamp_ns - self._start.timestamp_ns

        # ── stable window ─────────────────────────────────────────────────
        is_stable = False
        cx = cy = 0.0
        drift = STABLE_DRIFT_PX  # default when window is too small
        if len(self._window) >= STABLE_WINDOW_SAMPLES:
            xs = [s.x for s in self._window]
            ys = [s.y for s in self._window]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            drift = max(
                math.hypot(s.x - cx, s.y - cy) for s in self._window
            )
            is_stable = drift < STABLE_DRIFT_PX
        else:
            if self._window:
                xs = [s.x for s in self._window]
                ys = [s.y for s in self._window]
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
                drift = max(
                    (math.hypot(s.x - cx, s.y - cy) for s in self._window),
                    default=STABLE_DRIFT_PX,
                )

        return FixationState(
            active=True,
            is_stable=is_stable,
            centroid_x=cx,
            centroid_y=cy,
            start_ts_ns=self._start.timestamp_ns,
            duration_ns=duration_ns,
            sample_count=self._count,
            max_drift_px=drift,
        )
