#!/usr/bin/env python3
# tools/gaze_calibration_check.py
"""
Standalone Tiagobot gaze-calibration verification tool.

After running tiago_gaze_calibration_exec.py and pointing
config.TIAGOBOT_GAZE_CALIBRATION_PATH at the resulting NPZ, run this
to verify the calibration interactively *without* committing trials,
running EEG, or driving Tiagobot:

    python tools/gaze_calibration_check.py

Visually mirrors the driver's gaze selection screen (3x3 A-I grid +
yellow-green central fixation cross, calibration-aligned letter
positions) so the subject sees the same layout they'll see at run
time. Every frame:

  - pull the latest Neon snapshot (same dedupe / worn / finite filter
    the driver uses);
  - classify against the loaded NPZ via
    Utils.tiagobot_gaze.classify_gaze_to_letter (same call the driver
    makes — 2D centroid 1-NN with the TIAGOBOT_GAZE_MAX_DIST_NORM gate);
  - highlight the classified letter in green at 132 pt (matches the
    driver's selection screen), dim off-trajectory letters, and report
    "Looking at: X" / "—" at the bottom;
  - draw a small green dot at the gaze position so the subject can see
    how close their actual gaze lands relative to each centroid;
  - print top-3 nearest centroids (ignoring the distance gate) in the
    corner — useful when the classifier returns None and you want to
    see what it *almost* picked.

No dwell mechanic, no SPACE, no commit. Subject just looks at each
letter in turn and confirms the right one highlights. If the wrong
letter persistently highlights or the dot lands far from the intended
letter, the calibration needs to be redone — or the headset has
shifted since calibration.

ESC or window-close to quit. Lives under tools/ — not Tier 1/2.
"""
from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make repo root importable when run as `python tools/...`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pygame  # noqa: E402

import config  # noqa: E402
from Utils.gaze.gaze_system import GazeConfig, GazeSystem  # noqa: E402
from Utils.tiagobot_gaze import (  # noqa: E402
    LETTERS,
    classify_gaze_to_letter,
    grid_centroids_norm,
    load_centroids,
)


SNAPSHOT_POLL_HZ = 60.0
GAZE_SAMPLE_W = float(getattr(config, "GAZE_SAMPLE_WIDTH", 1600.0))
GAZE_SAMPLE_H = float(getattr(config, "GAZE_SAMPLE_HEIGHT", 1200.0))


def _read_latest_valid(
    gs: GazeSystem, last_unix_t: Optional[float]
) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
    """Return ((t_unix, x_norm, y_norm), new_last_t) on a fresh, worn,
    finite snapshot; (None, new_last_t) otherwise. Same contract as the
    driver's `_tiago_read_latest_valid_snapshot` (minus depth)."""
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
    return (float(t), x_px / GAZE_SAMPLE_W, y_px / GAZE_SAMPLE_H), float(t)


def _render_letter_grid(
    screen: "pygame.Surface",
    render_centroids: Dict[str, Tuple[float, float]],
    highlight_letter: Optional[str],
    available_set: set,
) -> None:
    """Mirror of the driver's _tiago_draw_letter_grid: nominal cells at
    the calibration-aligned centroids; `highlight_letter` is rendered
    larger in green (matches config.green ~ (0, 255, 0)); off-trajectory
    letters are dimmed so the operator can see which letters are
    eligible at a glance."""
    W, H = screen.get_size()
    for ch in LETTERS:
        cx_n, cy_n = render_centroids[ch]
        cx, cy = int(cx_n * W), int(cy_n * H)
        if ch == highlight_letter:
            color = (0, 255, 0)
            font_size = 132
        elif ch in available_set:
            color = (180, 180, 180)
            font_size = 96
        else:
            color = (90, 90, 90)
            font_size = 96
        font = pygame.font.SysFont(None, font_size)
        surf = font.render(ch, True, color)
        rect = surf.get_rect(center=(cx, cy))
        screen.blit(surf, rect)


def _render_fixation_cross(screen: "pygame.Surface") -> None:
    """Yellow-green head-pose anchor — same (200, 200, 80) / 18 px arm /
    3 px stroke the driver uses, so the head-fixed protocol's visual
    anchor is preserved here too."""
    W, H = screen.get_size()
    cx, cy = W // 2, H // 2
    arm = 18
    color = (200, 200, 80)
    pygame.draw.line(screen, color, (cx - arm, cy), (cx + arm, cy), 3)
    pygame.draw.line(screen, color, (cx, cy - arm), (cx, cy + arm), 3)


def _render_header(screen: "pygame.Surface") -> None:
    W, _H = screen.get_size()
    font = pygame.font.SysFont(None, 32)
    surf = font.render(
        "Calibration check — ESC to quit", True, (200, 200, 200),
    )
    rect = surf.get_rect(center=(W // 2, 40))
    screen.blit(surf, rect)


def _render_status_bottom(
    screen: "pygame.Surface", letter: Optional[str]
) -> None:
    W, H = screen.get_size()
    if letter is not None:
        text = f"Looking at: {letter}"
        color = (0, 255, 0)
    else:
        text = "Looking at: —"
        color = (220, 160, 0)
    font = pygame.font.SysFont(None, 64)
    surf = font.render(text, True, color)
    rect = surf.get_rect(center=(W // 2, H - 100))
    screen.blit(surf, rect)


def _render_gaze_dot(screen: "pygame.Surface", x_norm: float, y_norm: float) -> None:
    """Small green dot at the gaze pixel position. With a dark outline
    so it stays visible over the highlighted letter glyph."""
    W, H = screen.get_size()
    x = int(x_norm * W)
    y = int(y_norm * H)
    pygame.draw.circle(screen, (0, 255, 0), (x, y), 8)
    pygame.draw.circle(screen, (0, 0, 0), (x, y), 8, 1)


def _render_diag_corner(
    screen: "pygame.Surface",
    xy: Optional[Tuple[float, float]],
    top3: List[Tuple[str, float]],
    picked: Optional[str],
    max_d: Optional[float],
) -> None:
    """Bottom-left: gaze coords, top-3 nearest centroids (with distance),
    and the configured distance gate. Top-3 ignores the gate so the
    operator can see what the classifier almost picked when picked
    is None."""
    _W, H = screen.get_size()
    font = pygame.font.SysFont(None, 22)
    lines: List[str] = []
    if xy is None:
        lines.append("gaze: (no fresh sample)")
    else:
        lines.append(f"gaze (x_norm, y_norm) = ({xy[0]:.3f}, {xy[1]:.3f})")
    if max_d is not None:
        lines.append(f"max_dist_norm = {max_d:.3f}")
    else:
        lines.append("max_dist_norm = (disabled)")
    if top3:
        parts = []
        for ch, d in top3:
            tag = "*" if ch == picked else " "
            parts.append(f"{tag}{ch}={d:.3f}")
        lines.append("top-3:  " + "   ".join(parts))
    y0 = H - 50 - 24 * (len(lines) - 1)
    for i, ln in enumerate(lines):
        surf = font.render(ln, True, (160, 160, 160))
        screen.blit(surf, (20, y0 + i * 24))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live verification of an existing Tiagobot gaze "
                    "calibration. Renders the same grid + cross the "
                    "driver shows during gaze selection, classifies "
                    "each Neon snapshot against the calibration NPZ, "
                    "and highlights the picked letter — no EEG, no "
                    "Tiagobot, no commit.",
    )
    parser.add_argument(
        "--windowed", action="store_true",
        help="1280x800 windowed mode instead of fullscreen. Must match "
             "the mode used at calibration (centroids are tied to the "
             "physical letter positions on the active surface).",
    )
    parser.add_argument(
        "--display", type=int, default=0,
        help="Display index (0=primary). Must match the calibration's "
             "--display.",
    )
    parser.add_argument(
        "--list-displays", action="store_true",
        help="Print available pygame displays and exit.",
    )
    parser.add_argument(
        "--no-dot", action="store_true",
        help="Hide the live gaze-position dot.",
    )
    args = parser.parse_args()

    # ---- Calibration ----
    cal_path = str(getattr(config, "TIAGOBOT_GAZE_CALIBRATION_PATH", "") or "")
    if not cal_path:
        print(
            "ERROR: TIAGOBOT_GAZE_CALIBRATION_PATH is empty. Set it in "
            "config_local.py to the NPZ produced by "
            "tiago_gaze_calibration_exec.py.",
            file=sys.stderr,
        )
        return 1
    cal_mtime = (
        datetime.datetime.fromtimestamp(Path(cal_path).stat().st_mtime)
        .isoformat(timespec="seconds")
        if Path(cal_path).exists() else "(missing!)"
    )
    print("=" * 72, flush=True)
    print(f"CALIBRATION NPZ: {cal_path}", flush=True)
    print(f"  mtime: {cal_mtime}", flush=True)
    print("=" * 72, flush=True)
    try:
        centroids = load_centroids(cal_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR loading calibration: {e}", file=sys.stderr)
        return 1
    print(
        f"Loaded {len(centroids)}/{len(LETTERS)} letter centroids:",
        flush=True,
    )
    for ch in LETTERS:
        if ch in centroids:
            cx, cy = float(centroids[ch][0]), float(centroids[ch][1])
            print(f"  {ch}: ({cx:.3f}, {cy:.3f})", flush=True)
        else:
            print(f"  {ch}: (missing — will never highlight)", flush=True)

    # ---- Neon ----
    host = str(getattr(config, "NEON_COMPANION_HOST", "") or "")
    print(
        f"Connecting to Neon (host={host!r}, mDNS if empty)...", flush=True,
    )
    gs = GazeSystem(GazeConfig(
        enable_prints=False, enable_display=False, enable_cv=False,
        use_tracker=False, neon_host=host,
    ))
    gs.start()

    # ---- UI ----
    pygame.init()
    n_disp = pygame.display.get_num_displays()
    if args.list_displays:
        print(f"Available pygame displays: {n_disp}")
        try:
            sizes = pygame.display.get_desktop_sizes()
        except AttributeError:
            sizes = [None] * n_disp
        for i, sz in enumerate(sizes):
            print(f"  display {i}: size={sz}")
        try:
            gs.stop()
        except Exception:
            pass
        pygame.quit()
        return 0
    if args.display < 0 or args.display >= n_disp:
        print(
            f"ERROR: --display {args.display} out of range; "
            f"{n_disp} display(s) available. Re-run with --list-displays.",
            file=sys.stderr,
        )
        try:
            gs.stop()
        except Exception:
            pass
        pygame.quit()
        return 1
    if args.windowed:
        screen = pygame.display.set_mode((1280, 800), display=args.display)
    else:
        screen = pygame.display.set_mode(
            (0, 0), pygame.FULLSCREEN, display=args.display,
        )
    pygame.display.set_caption(
        "Tiagobot gaze calibration check (no EEG, no robot)"
    )
    W, H = screen.get_size()
    print(
        f"Display: {args.display} of {n_disp}, surface={W}x{H}",
        flush=True,
    )

    trajectory = list(getattr(config, "TIAGOBOT_TRAJECTORY", list(LETTERS)))
    available_set = set(trajectory)
    max_d_attr = getattr(config, "TIAGOBOT_GAZE_MAX_DIST_NORM", 0.2)
    max_d = float(max_d_attr) if max_d_attr is not None else None
    render_centroids = grid_centroids_norm()

    poll = 1.0 / SNAPSHOT_POLL_HZ
    last_unix_t: Optional[float] = None
    last_xy: Optional[Tuple[float, float]] = None
    last_letter: Optional[str] = None

    try:
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return 0
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    return 0

            s, last_unix_t = _read_latest_valid(gs, last_unix_t)
            if s is not None:
                _t, x_norm, y_norm = s
                last_xy = (x_norm, y_norm)
                last_letter = classify_gaze_to_letter(
                    x_norm, y_norm, centroids,
                    gaze_depth_cm=None,
                    available_letters=trajectory,
                    max_dist_norm=max_d,
                )
            # If no fresh sample, keep the previous highlight + dot
            # visible (same "sticky" feel the driver's dwell loop has).

            screen.fill((0, 0, 0))
            _render_header(screen)
            _render_letter_grid(
                screen, render_centroids, last_letter, available_set,
            )
            _render_fixation_cross(screen)
            _render_status_bottom(screen, last_letter)

            top3: List[Tuple[str, float]] = []
            if last_xy is not None:
                if not args.no_dot:
                    _render_gaze_dot(screen, last_xy[0], last_xy[1])
                top3 = sorted(
                    (
                        (ch, float(np.hypot(
                            centroids[ch][0] - last_xy[0],
                            centroids[ch][1] - last_xy[1],
                        )))
                        for ch in trajectory if ch in centroids
                    ),
                    key=lambda kv: kv[1],
                )[:3]
            _render_diag_corner(
                screen, last_xy, top3, last_letter, max_d,
            )

            pygame.display.flip()
            time.sleep(poll)
    finally:
        try:
            gs.stop()
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
