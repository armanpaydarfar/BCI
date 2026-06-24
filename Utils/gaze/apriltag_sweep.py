"""
apriltag_sweep.py — pure acceptance logic for the REV04 swept calibration capture
(methodology rev04 §2).

The sweep hand-guides the end-effector across the table while a 20 Hz loop pairs
each relay frame with a robot-telemetry sample. This module holds ONLY the
hardware-free decision of whether a paired sample is calibration-quality:

  - **frame↔telemetry time-alignment** — both timestamps are ``time.time()`` on the
    control host (the relay frame's arrival time, the telemetry's ``_t``), so a
    large offset means a STALE frame (the relay stalled while the arm kept moving),
    not clock skew. Reject those.
  - **light per-sample quality gates** (rev04 §2) — require a world tag AND an EE
    tag co-visible, a fresh frame, and a decision margin above a floor (motion blur
    / glancing views fail this). The sweep trades per-sample precision for volume;
    density + the A2 planar fit average out residual noise.

Pure / deterministic (no RNG, no I/O), so the gating is unit-tested without a Neon
or robot — the sweep loop in ``tools/apriltag_calibrate.py`` just calls
``accept_sweep_sample()``.
"""

from __future__ import annotations

from typing import Sequence, Tuple

# One frame at 20 Hz = 50 ms. A paired frame/telemetry offset above this means the
# newest relay frame is older than one sweep tick — stale, drop it.
DEFAULT_MAX_ALIGN_DT_S = 0.05

# Decision-margin floor. The HIL world tags ran 34–83 (verification report §5);
# 20 is a permissive default that still rejects motion-blurred / glancing reads.
# Start loose, tighten against the control-test result (rev04 §9).
DEFAULT_MIN_MARGIN = 20.0


def frame_telemetry_dt(t_frame: float, t_robot: float) -> float:
    """Absolute host-clock offset (s) between a relay frame's arrival time and a
    telemetry sample's ``_t``. Both are ``time.time()`` on the control host."""
    return abs(float(t_frame) - float(t_robot))


def accept_sweep_sample(*, world_seen: bool, ee_seen: bool,
                        margins: Sequence[float], dt_s: float,
                        min_margin: float = DEFAULT_MIN_MARGIN,
                        max_align_dt_s: float = DEFAULT_MAX_ALIGN_DT_S
                        ) -> Tuple[bool, str]:
    """Decide whether one paired ``(frame, telemetry)`` sweep sample is
    calibration-quality (rev04 §2). Returns ``(accept, reason)`` — ``reason`` is a
    short discard tag for the operator log when rejected, ``"ok"`` when accepted.

    Rejects, in order: no world tag co-visible; no EE tag co-visible;
    frame↔telemetry Δt over bound (stale frame); the weakest contributing tag's
    decision margin below the floor. ``margins`` are the decision margins of the
    tags that fed the recovered world + EE poses — the weakest is the conservative
    gate. An empty ``margins`` skips the margin check (the caller had no tag margin
    to offer, e.g. a synthetic test)."""
    if not world_seen:
        return False, "no world tag"
    if not ee_seen:
        return False, "no EE tag"
    if dt_s > max_align_dt_s:
        return False, f"stale frame (Δt {dt_s * 1e3:.0f}ms > {max_align_dt_s * 1e3:.0f}ms)"
    if len(margins) > 0:
        worst = float(min(margins))
        if worst < min_margin:
            return False, f"low decision margin ({worst:.0f} < {min_margin:.0f})"
    return True, "ok"
