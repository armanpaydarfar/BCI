#!/usr/bin/env python3
# tools/gaze_to_tiago_test.py
"""
Non-BCI integration test for the head-fixed on-screen gaze -> tiagobot
pipeline.

Protocol (matches tiago_gaze_calibration_exec.py 2026-05-21 onward):
The 3x3 A-I letter grid plus a central fixation cross are rendered on
the screen for the entire session. The user keeps their HEAD pointed
at the cross and only their EYES dart to the chosen letter. Each trial:

    SPACE -> 4 s eyes-on-target window -> 2D centroid 1-NN classify
    -> Tiagobot GO -> wait for "Target Location Reached."
    -> HOLD_S pause -> Tiagobot HOME -> wait for "Homed."
    -> ready for next trial. ESC quits.

Why on-screen + head-fixed: scene-camera pixel coords are not
invariant to head pose, which is what made the earlier "look at the
physical board" approach fail. Fixing the head and putting the targets
on the head-mounted scene camera's known viewport restores a stable
mapping from gaze pixels to letters.

Reuses:
  - Utils.gaze.gaze_system.GazeSystem  (Pupil Labs Neon realtime API)
  - Utils.tiagobot_gaze.classify_gaze_to_letter (2D centroid 1-NN —
    no depth, no Mahalanobis, no cloud-vote; the centroids are now
    well-separated so the simplest classifier suffices)
  - Utils.tiagobot (serial port, send_letter, send_home,
    wait_for_completion)
  - config / config_local

Lives under tools/ — not Tier 1/2. Once this validates, the same
helpers fold into ExperimentDriver_Online_Tiagobot_Gaze.py's gaze
selection window.
"""
from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Make repo root importable when run as `python tools/...`. Mirrors
# tools/perception_latency_probe.py:41-42.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402  (path setup above)
import pygame  # noqa: E402

import config  # noqa: E402
from Utils.gaze.gaze_system import GazeConfig, GazeSystem  # noqa: E402
from Utils.tiagobot import (  # noqa: E402
    HOMED_MARKER,
    TARGET_REACHED_MARKER,
    close_port,
    find_tiagobot_port,
    open_port,
    send_home,
    send_letter,
    wait_for_completion,
)
from Utils.tiagobot_gaze import (  # noqa: E402
    LETTERS,
    average_gaze_over_window,
    classify_gaze_to_letter,
    grid_centroids_norm,
    load_centroids,
)


HOLD_S = 2.0          # seconds between target-reach and HOME
MOTION_TIMEOUT_S = 60.0
SNAPSHOT_POLL_HZ = 60.0
GAZE_SAMPLE_W = float(getattr(config, "GAZE_SAMPLE_WIDTH", 1600.0))
GAZE_SAMPLE_H = float(getattr(config, "GAZE_SAMPLE_HEIGHT", 1200.0))


class _Logger:
    """Tiny stdout-only logger matching the .log_event(msg, level=...)
    contract used by Utils.tiagobot helpers."""

    def log_event(self, msg: str, level: str = "info") -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        stream = sys.stderr if level == "error" else sys.stdout
        print(f"[{ts}] [{level}] {msg}", file=stream, flush=True)


def _draw_status(screen: "pygame.Surface", lines: List[str]) -> None:
    W, H = screen.get_size()
    screen.fill((0, 0, 0))
    font = pygame.font.SysFont(None, 48)
    total_h = len(lines) * 60
    y0 = H // 2 - total_h // 2
    for i, ln in enumerate(lines):
        surf = font.render(ln, True, (220, 220, 220))
        rect = surf.get_rect(center=(W // 2, y0 + i * 60))
        screen.blit(surf, rect)
    pygame.display.flip()


def _draw_fixation_cross(screen: "pygame.Surface") -> None:
    """Central yellow-green cross. Head-pose anchor — must match the
    calibration script's cross (tiago_gaze_calibration_exec._draw_fixation_cross)
    so the user's head position at runtime matches calibration."""
    W, H = screen.get_size()
    cx, cy = W // 2, H // 2
    arm = 18
    color = (200, 200, 80)
    pygame.draw.line(screen, color, (cx - arm, cy), (cx + arm, cy), 3)
    pygame.draw.line(screen, color, (cx, cy - arm), (cx, cy + arm), 3)


def _draw_grid_with_cross(
    screen: "pygame.Surface",
    centroids_norm: Dict[str, Tuple[float, float]],
    countdown_text: Optional[str] = None,
) -> None:
    """Render the 3x3 grid (no per-letter highlight at runtime — user
    chooses which letter to look at) plus the central fixation cross.

    Mirrors tiago_gaze_calibration_exec._draw_grid, minus the active-
    letter halo. Layout matches the calibration grid exactly so the
    runtime scene-pixel positions of each letter align with the
    calibrated centroids."""
    W, H = screen.get_size()
    screen.fill((0, 0, 0))

    font = pygame.font.SysFont(None, 96)
    for ch in LETTERS:
        cx_n, cy_n = centroids_norm[ch]
        cx, cy = int(cx_n * W), int(cy_n * H)
        surf = font.render(ch, True, (180, 180, 180))
        rect = surf.get_rect(center=(cx, cy))
        screen.blit(surf, rect)

    _draw_fixation_cross(screen)

    if countdown_text:
        font_cd = pygame.font.SysFont(None, 40)
        surf = font_cd.render(countdown_text, True, (200, 200, 200))
        rect = surf.get_rect(center=(W // 2, H - 40))
        screen.blit(surf, rect)

    pygame.display.flip()


def _pump_quit() -> Optional[str]:
    """Drain pygame events; return 'space', 'esc', or None."""
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            return "esc"
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                return "esc"
            if ev.key == pygame.K_SPACE:
                return "space"
    return None


def _read_latest_valid(
    gs: GazeSystem, last_unix_t: Optional[float]
) -> Tuple[Optional[Tuple[float, float, float, float, float]], Optional[float]]:
    """Return (sample_or_None, new_last_unix_t). Same dedupe/worn-filter
    + NaN-depth-when-invalid contract as
    tiago_gaze_calibration_exec._read_latest_valid."""
    snap = gs.get_snapshot(include_objects=False, include_frame=False)
    if not snap or not snap.get("ok"):
        return None, last_unix_t
    t = snap.get("unix_t")
    if t is None:
        return None, last_unix_t
    if last_unix_t is not None and t <= last_unix_t:
        return None, last_unix_t
    if not bool(snap.get("worn")):
        return None, float(t)
    px = snap.get("gaze_px_raw")
    if px is None:
        return None, float(t)
    x_px, y_px = float(px[0]), float(px[1])
    if not (np.isfinite(x_px) and np.isfinite(y_px)):
        return None, float(t)
    depth_cm = float("nan")
    if bool(snap.get("depth_valid")):
        d = snap.get("depth_cm")
        if d is not None:
            df = float(d)
            if np.isfinite(df):
                depth_cm = df
    return (
        float(t), x_px / GAZE_SAMPLE_W, y_px / GAZE_SAMPLE_H, 1.0, depth_cm,
    ), float(t)


def _collect_gaze_window(
    screen: "pygame.Surface",
    gs: GazeSystem,
    duration_s: float,
    centroids_render: Dict[str, Tuple[float, float]],
) -> Optional[List[Tuple[float, float, float, float, float]]]:
    """Accumulate gaze samples for `duration_s` while the on-screen 3x3
    grid + cross stays visible. The user keeps their head pointed at
    the cross and looks at their chosen letter with their eyes only.
    Returns None on ESC."""
    poll = 1.0 / SNAPSHOT_POLL_HZ
    samples: List[Tuple[float, float, float, float, float]] = []
    last_t: Optional[float] = None
    t_end = time.monotonic() + float(duration_s)
    while time.monotonic() < t_end:
        ev = _pump_quit()
        if ev == "esc":
            return None
        s, last_t = _read_latest_valid(gs, last_t)
        if s is not None:
            samples.append(s)
        remaining = t_end - time.monotonic()
        _draw_grid_with_cross(
            screen, centroids_render,
            countdown_text=(
                f"Eyes on chosen letter — {remaining:0.1f}s   "
                f"({len(samples)} samples)"
            ),
        )
        time.sleep(poll)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Non-BCI head-fixed on-screen gaze->tiagobot test.",
    )
    parser.add_argument(
        "--windowed", action="store_true",
        help="Use a 1280x800 windowed mode instead of fullscreen. Mirrors "
             "the calibration script's flag. NOTE: the calibration centroids "
             "are tied to the physical letter positions on the screen, so "
             "the test window MUST match the calibration window. Run "
             "calibration with --windowed too if you use this here.",
    )
    args = parser.parse_args()

    logger = _Logger()

    # ---- Calibration ----
    cal_path = str(getattr(config, "TIAGOBOT_GAZE_CALIBRATION_PATH", "") or "")
    if not cal_path:
        print("ERROR: TIAGOBOT_GAZE_CALIBRATION_PATH is empty. Set it in "
              "config_local.py to the NPZ from tiago_gaze_calibration_exec.py.",
              file=sys.stderr)
        return 1
    # Stat the NPZ + print mtime so a stale config_local.py path is
    # obvious at startup (rather than silently classifying against an
    # old calibration). Showed up 2026-05-21 when a path bump was
    # missed across multiple recalibrations.
    cal_mtime = (
        datetime.datetime.fromtimestamp(Path(cal_path).stat().st_mtime)
        .isoformat(timespec="seconds")
        if Path(cal_path).exists() else "(missing!)"
    )
    print("=" * 72, flush=True)
    print(f"CALIBRATION NPZ: {cal_path}", flush=True)
    print(f"  mtime: {cal_mtime}", flush=True)
    print("=" * 72, flush=True)
    centroids = load_centroids(cal_path)
    logger.log_event(
        f"Loaded {len(centroids)}/{len(LETTERS)} letter centroids (x, y in scene-norm coords)"
    )
    for ch, cent in centroids.items():
        cx, cy = float(cent[0]), float(cent[1])
        logger.log_event(f"  {ch}: centroid=({cx:.3f}, {cy:.3f})")

    # ---- Tiagobot ----
    port = config.TIAGOBOT_PORT or find_tiagobot_port(logger=logger)
    if not port:
        logger.log_event(
            "Tiagobot port not found (TIAGOBOT_PORT empty and USB scan "
            "returned no Mega 2560). Plug in the device or set "
            "TIAGOBOT_PORT in config_local.py.",
            level="error",
        )
        return 1
    tiago = open_port(port, config.TIAGOBOT_BAUD, logger)
    if tiago is None and not config.SIMULATION_MODE:
        logger.log_event("open_port returned None and SIMULATION_MODE=False — abort.",
                         level="error")
        return 1

    # ---- Neon (GazeSystem) ----
    host = str(getattr(config, "NEON_COMPANION_HOST", "") or "")
    logger.log_event(f"Connecting to Neon (host={host!r}, mDNS if empty)...")
    gs = GazeSystem(GazeConfig(
        enable_prints=False, enable_display=False, enable_cv=False,
        use_tracker=False, neon_host=host,
    ))
    gs.start()

    # ---- UI ----
    # Fullscreen by default so the letters land in the same scene-camera
    # pixels they did during calibration. --windowed only for off-Neon
    # development. Mirrors tiago_gaze_calibration_exec.py:407-410.
    pygame.init()
    if args.windowed:
        screen = pygame.display.set_mode((1280, 800))
    else:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.display.set_caption("Gaze -> Tiagobot test (no EEG)")

    win_s = float(getattr(config, "TIAGOBOT_GAZE_SELECTION_WINDOW", 4.0))
    conf_thr = float(getattr(config, "TIAGOBOT_GAZE_CONFIDENCE_THRESHOLD", 0.7))
    max_d = getattr(config, "TIAGOBOT_GAZE_MAX_DIST_NORM", 0.2)
    trajectory = list(getattr(config, "TIAGOBOT_TRAJECTORY", list(LETTERS)))

    # Grid layout for runtime rendering — MUST match the calibration
    # script's `centroids_norm = grid_centroids_norm()` so the on-screen
    # letter positions align with the calibrated scene-pixel centroids.
    render_centroids = grid_centroids_norm()

    # Snapshot head pose for the session reference; we log drift per
    # trial. Head drift is the main failure mode of head-fixed mode —
    # surfacing it lets the operator catch a slipping headset.
    _ref_snap = gs.get_snapshot(include_objects=False, include_frame=False) or {}
    ref_yaw_raw = _ref_snap.get("head_yaw_deg")
    ref_pitch_raw = _ref_snap.get("head_pitch_deg")
    if (
        ref_yaw_raw is not None and ref_pitch_raw is not None
        and np.isfinite(float(ref_yaw_raw)) and np.isfinite(float(ref_pitch_raw))
    ):
        ref_yaw = float(ref_yaw_raw)
        ref_pitch = float(ref_pitch_raw)
        logger.log_event(
            f"Head-pose reference: yaw={ref_yaw:+.1f}°  pitch={ref_pitch:+.1f}°"
        )
    else:
        ref_yaw = float("nan")
        ref_pitch = float("nan")
        logger.log_event(
            "Head-pose reference unavailable (IMU not warmed up?); "
            "per-trial drift will be skipped."
        )

    def _tick():
        # Pump pygame so the window stays responsive during the long
        # wait_for_completion blocks. SystemExit propagates so the
        # finally-cleanup runs on window close.
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                raise SystemExit
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                raise SystemExit

    try:
        last_letter: Optional[str] = None
        trial = 0
        while True:
            trial += 1
            # Between trials, show the grid + cross with a short prompt
            # at the bottom. User keeps head on cross even between
            # trials so the head pose carries over to the next window.
            _draw_grid_with_cross(
                screen, render_centroids,
                countdown_text=(
                    f"Trial {trial}  (last: {last_letter or '—'})  "
                    f"SPACE = start trial  /  ESC = quit"
                ),
            )
            # Wait for SPACE.
            while True:
                ev = _pump_quit()
                if ev == "esc":
                    return 0
                if ev == "space":
                    break
                time.sleep(0.02)

            # Head-pose drift log at trial start.
            snap = gs.get_snapshot(include_objects=False, include_frame=False) or {}
            yn = snap.get("head_yaw_deg")
            pn = snap.get("head_pitch_deg")
            head_drift_str = ""
            if (
                yn is not None and pn is not None
                and np.isfinite(ref_yaw) and np.isfinite(ref_pitch)
                and np.isfinite(float(yn)) and np.isfinite(float(pn))
            ):
                dy = float(yn) - ref_yaw
                dp = float(pn) - ref_pitch
                warn = "  <-- DRIFT" if (abs(dy) > 3.0 or abs(dp) > 3.0) else ""
                head_drift_str = f"head Δ=(yaw {dy:+.1f}°, pitch {dp:+.1f}°){warn}"

            # Gaze accumulation window — grid+cross stays rendered.
            samples = _collect_gaze_window(screen, gs, win_s, render_centroids)
            if samples is None:  # ESC during window
                return 0

            # Simple 2D centroid 1-NN: take the median (x, y) over
            # confidence-passing samples and classify by Euclidean
            # nearest centroid. Depth column ignored (gaze_depth_cm=None)
            # — for head-fixed on-screen mode the (x, y) signal is well
            # separated and depth adds only noise.
            avg = average_gaze_over_window(samples, conf_threshold=conf_thr)
            if avg is None:
                gx = gy = float("nan")
                letter = None
            else:
                gx, gy, _ = avg
                letter = classify_gaze_to_letter(
                    gx, gy, centroids,
                    gaze_depth_cm=None,
                    available_letters=trajectory,
                    max_dist_norm=float(max_d) if max_d is not None else None,
                )

            # ---- Per-trial diagnostic ----
            print(file=sys.stdout)
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] === Trial {trial} ===", flush=True)
            if avg is not None:
                print(
                    f"  median (x, y) = ({gx:.3f}, {gy:.3f})  samples={len(samples)}",
                    flush=True,
                )
            else:
                print(f"  median: N/A (0 valid samples)  samples={len(samples)}", flush=True)
            if head_drift_str:
                print(f"  {head_drift_str}", flush=True)
            # Top-3 letters by Euclidean distance from median (useful
            # when the picked letter is borderline).
            if avg is not None:
                dists = sorted(
                    (
                        (ch, float(np.hypot(centroids[ch][0] - gx, centroids[ch][1] - gy)))
                        for ch in trajectory if ch in centroids
                    ),
                    key=lambda kv: kv[1],
                )[:3]
                print("  top-3 letters by distance (lower = closer):", flush=True)
                for ch, d in dists:
                    marker = "  <-- picked" if ch == letter else ""
                    print(f"    {ch}: dist={d:.3f}{marker}", flush=True)
            print(f"  -> picked: {letter!r}", flush=True)

            if letter is None:
                _draw_grid_with_cross(
                    screen, render_centroids,
                    countdown_text=(
                        f"Trial {trial}: no letter (median too far). "
                        f"SPACE to retry, ESC to quit."
                    ),
                )
                continue

            # Drive tiagobot.
            _draw_grid_with_cross(
                screen, render_centroids,
                countdown_text=f"Trial {trial}: sending letter {letter}...",
            )
            send_letter(tiago, letter, logger)
            _draw_grid_with_cross(
                screen, render_centroids,
                countdown_text=f"Tiagobot moving to '{letter}' — waiting for reach...",
            )
            reached = wait_for_completion(
                tiago, TARGET_REACHED_MARKER, timeout=MOTION_TIMEOUT_S,
                logger=None, on_tick=_tick,
            )
            if not reached:
                logger.log_event(
                    f"Trial {trial}: target-reach timed out — proceeding to HOME anyway.",
                    level="error",
                )

            _draw_grid_with_cross(
                screen, render_centroids,
                countdown_text=f"Holding {HOLD_S:0.1f}s at '{letter}'...",
            )
            time.sleep(HOLD_S)

            _draw_grid_with_cross(
                screen, render_centroids,
                countdown_text="Sending HOME...",
            )
            send_home(tiago, logger)
            homed = wait_for_completion(
                tiago, HOMED_MARKER, timeout=MOTION_TIMEOUT_S,
                logger=None, on_tick=_tick,
            )
            if not homed:
                logger.log_event(
                    f"Trial {trial}: HOME timed out.", level="error"
                )

            last_letter = letter
    finally:
        try:
            gs.stop()
        except Exception:
            pass
        try:
            close_port(tiago, logger)
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
