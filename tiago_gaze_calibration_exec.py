#!/usr/bin/env python3
# tiago_gaze_calibration_exec.py
"""
Tiagobot gaze calibration recorder.

Displays a 3x3 on-screen grid of A-I targets (per
`Documents/SoftwareDocs/Tiagobot_Gaze_AI_Layout.md`) and records the
user's gaze samples while they fixate each letter in turn. Writes a
calibration NPZ consumed at runtime by
`ExperimentDriver_Online_Tiagobot_Gaze.py` via `Utils.tiagobot_gaze`.

Wire format mirrors `harmony_calibration_exec.py` (pylsl `resolve_stream
('type', 'Gaze')` → Pupil Labs Neon stream). Output schema matches the
contract `Utils.tiagobot_gaze.load_centroids` expects:

    T          (n_samples,) float64   sample timestamps (unix seconds)
    G          (n_samples, 3) float32 [x_norm, y_norm, confidence]
    labels     (n_samples,)  S1       letter the sample belongs to
                                      ("A"-"I"; empty bytes for inter-
                                      target rest samples)
    centroids  (9, 2) float32         per-letter median of valid samples
                                      (NaN row if no valid samples for
                                      that letter)
    letters    (9,) S1                "A"..."I" row labels for centroids
    meta       dict                   version, capture date, subject,
                                      layout doc path, confidence
                                      threshold, samples-per-target

Per CLAUDE.md fail-fast policy: a missing Neon LSL stream aborts at
startup (no silent recovery — calibration data is the contract that the
runtime classifier depends on).

This script is **laptop-only in practice** (needs the physical Neon
tracker + a display); it is not a realtime closed-loop driver and is
not Tier 1 / Tier 2.
"""
from __future__ import annotations

import argparse
import datetime
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame
from pylsl import StreamInlet, resolve_stream

import config
from Utils.tiagobot_gaze import LETTERS, grid_centroids_norm


# ---- Calibration parameters -------------------------------------------
SAMPLES_PER_TARGET = 100            # ~1 s at ~120 Hz Neon LSL rate
SETTLE_SECONDS = 1.0                # countdown before each target
GAZE_PULL_TIMEOUT_S = 0.0           # non-blocking pull, drain everything
DEFAULT_BAUD_TIMEOUT_S = 60.0       # max wall-clock per target capture

# Pupil Labs Neon LSL channel layout (matches harmony_calibration_exec.py:194-197):
#   sample[0]  = gaze_x (pixels, 0..1600)
#   sample[1]  = gaze_y (pixels, 0..1200)
#   sample[15] = confidence
NEON_X_INDEX = 0
NEON_Y_INDEX = 1
NEON_CONF_INDEX = 15

# Default Neon scene resolution. Pulled from config so calibration and
# runtime agree on the normalization denominator.
GAZE_SAMPLE_W = float(getattr(config, "GAZE_SAMPLE_WIDTH", 1600.0))
GAZE_SAMPLE_H = float(getattr(config, "GAZE_SAMPLE_HEIGHT", 1200.0))


# ---- Helpers ----------------------------------------------------------
def _ts_print(*args, **kwargs) -> None:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}]", *args, **kwargs)


def _connect_gaze_stream() -> StreamInlet:
    """Resolve the Pupil Labs Neon LSL stream and return an inlet.

    Mirrors `harmony_calibration_exec.py:GazeStream.connect` (verified
    2026-05-19 against the branch). Raises on missing stream so the
    operator gets a loud error instead of a silently-empty NPZ.
    """
    _ts_print("[GAZE] Searching for Neon LSL stream (type='Gaze')...")
    streams = resolve_stream("type", "Gaze")
    if not streams:
        raise RuntimeError(
            "No LSL stream of type='Gaze' found. Check that Pupil Labs "
            "Neon Companion is running and the LSL relay is enabled."
        )
    inlet = StreamInlet(streams[0])
    _ts_print(f"[GAZE] Connected to: {streams[0].name()}")
    return inlet


def _drain_latest_valid(
    inlet: StreamInlet,
    conf_threshold: float,
) -> Optional[Tuple[float, float, float, float]]:
    """Pull all queued samples, return the most recent confidence-passing
    one as `(t_unix, x_norm, y_norm, conf)`, or None if none pass.

    The "drain to latest" pattern matches
    `harmony_calibration_exec.py:GazeStream.get_latest_gaze` — at 120 Hz
    Neon + ~60 Hz pygame frame rate we get a small backlog per frame,
    keeping the freshest valid sample is what the realtime decoder
    pattern uses.
    """
    latest: Optional[Tuple[float, float, float, float]] = None
    while True:
        sample, t_unix = inlet.pull_sample(timeout=GAZE_PULL_TIMEOUT_S)
        if not sample:
            break
        try:
            x_px = float(sample[NEON_X_INDEX])
            y_px = float(sample[NEON_Y_INDEX])
            conf = float(sample[NEON_CONF_INDEX])
        except (IndexError, TypeError, ValueError):
            continue
        if not np.isfinite(conf) or conf < conf_threshold:
            continue
        if not (np.isfinite(x_px) and np.isfinite(y_px)):
            continue
        x_norm = x_px / GAZE_SAMPLE_W
        y_norm = y_px / GAZE_SAMPLE_H
        latest = (float(t_unix), x_norm, y_norm, conf)
    return latest


def _pump_events_quit() -> bool:
    """Return True if a quit / ESC was requested."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return True
    return False


# ---- Drawing ----------------------------------------------------------
def _draw_grid(
    screen: "pygame.Surface",
    centroids_norm: Dict[str, Tuple[float, float]],
    active_letter: Optional[str],
    countdown_text: Optional[str] = None,
) -> None:
    """Render the 3x3 grid: highlight the active letter, dim the rest.

    Used during calibration so the subject sees where to look next.
    """
    W, H = screen.get_size()
    screen.fill((0, 0, 0))

    font_active = pygame.font.SysFont(None, 220)
    font_dim = pygame.font.SysFont(None, 96)

    for ch in LETTERS:
        cx_n, cy_n = centroids_norm[ch]
        cx, cy = int(cx_n * W), int(cy_n * H)

        if ch == active_letter:
            color = (255, 255, 255)
            surf = font_active.render(ch, True, color)
            # Bright background circle to anchor fixation.
            pygame.draw.circle(screen, (40, 80, 40), (cx, cy), 110)
        else:
            color = (60, 60, 60)
            surf = font_dim.render(ch, True, color)
        rect = surf.get_rect(center=(cx, cy))
        screen.blit(surf, rect)

    if countdown_text:
        font_cd = pygame.font.SysFont(None, 64)
        surf = font_cd.render(countdown_text, True, (200, 200, 200))
        rect = surf.get_rect(center=(W // 2, H - 60))
        screen.blit(surf, rect)

    pygame.display.flip()


def _draw_message(screen: "pygame.Surface", lines: List[str]) -> None:
    W, H = screen.get_size()
    screen.fill((0, 0, 0))
    font = pygame.font.SysFont(None, 56)
    total_h = len(lines) * 70
    y0 = H // 2 - total_h // 2
    for i, ln in enumerate(lines):
        surf = font.render(ln, True, (220, 220, 220))
        rect = surf.get_rect(center=(W // 2, y0 + i * 70))
        screen.blit(surf, rect)
    pygame.display.flip()


# ---- Capture loop -----------------------------------------------------
def _capture_target(
    screen: "pygame.Surface",
    inlet: StreamInlet,
    centroids_norm: Dict[str, Tuple[float, float]],
    letter: str,
    n_samples: int,
    conf_threshold: float,
    settle_s: float,
    timeout_s: float,
) -> List[Tuple[float, float, float, float]]:
    """Capture `n_samples` gaze samples while the subject fixates
    `letter`. Returns the list of `(t_unix, x_norm, y_norm, conf)`.

    Drains the LSL queue once per frame and keeps the latest valid
    sample, so a 60 Hz frame rate against a 120 Hz Neon stream yields
    ~60 unique samples/s — enough to hit N=100 in ~1.7 s typical.
    """
    # Settle countdown.
    settle_end = time.monotonic() + settle_s
    while time.monotonic() < settle_end:
        if _pump_events_quit():
            raise KeyboardInterrupt
        remaining = settle_end - time.monotonic()
        _draw_grid(
            screen, centroids_norm, letter,
            countdown_text=f"Look at '{letter}' — capture in {remaining:0.1f} s",
        )
        # Drain (and discard) the queue so we don't include stale data
        # from the previous target's drift.
        _drain_latest_valid(inlet, conf_threshold=0.0)
        time.sleep(0.01)

    # Capture phase.
    samples: List[Tuple[float, float, float, float]] = []
    capture_start = time.monotonic()
    capture_deadline = capture_start + timeout_s
    last_log = capture_start
    while len(samples) < n_samples and time.monotonic() < capture_deadline:
        if _pump_events_quit():
            raise KeyboardInterrupt
        s = _drain_latest_valid(inlet, conf_threshold=conf_threshold)
        if s is not None:
            samples.append(s)
        progress = len(samples) / max(1, n_samples)
        _draw_grid(
            screen, centroids_norm, letter,
            countdown_text=f"Capturing '{letter}' — {len(samples)}/{n_samples} samples",
        )
        # Throttle to ~60 Hz so we don't busy-spin between Neon samples.
        time.sleep(1.0 / 60.0)
        now = time.monotonic()
        if now - last_log > 1.0:
            _ts_print(f"[CAL] {letter}: {len(samples)}/{n_samples} samples ({progress*100:.0f}%)")
            last_log = now

    if len(samples) < n_samples:
        _ts_print(
            f"[CAL] WARN: target '{letter}' timed out at "
            f"{len(samples)}/{n_samples} samples after {timeout_s:.1f} s"
        )
    return samples


# ---- NPZ writer -------------------------------------------------------
def _save_calibration(
    out_path: Path,
    per_letter_samples: Dict[str, List[Tuple[float, float, float, float]]],
    conf_threshold: float,
    samples_per_target: int,
    subject: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten per-letter samples into NPZ arrays and write to disk.

    Returns (T, G, labels, centroids) for the caller's stat printout.
    """
    T_list: List[float] = []
    G_list: List[Tuple[float, float, float]] = []
    label_list: List[str] = []
    centroids = np.full((len(LETTERS), 2), np.nan, dtype=np.float32)

    for i, ch in enumerate(LETTERS):
        samples = per_letter_samples.get(ch, [])
        if samples:
            for (t, x, y, c) in samples:
                T_list.append(t)
                G_list.append((x, y, c))
                label_list.append(ch)
            xs = np.array([s[1] for s in samples], dtype=np.float64)
            ys = np.array([s[2] for s in samples], dtype=np.float64)
            centroids[i, 0] = float(np.median(xs))
            centroids[i, 1] = float(np.median(ys))

    T = np.asarray(T_list, dtype=np.float64)
    G = np.asarray(G_list, dtype=np.float32) if G_list else np.zeros((0, 3), dtype=np.float32)
    labels = np.asarray(label_list, dtype="S1")
    letters_arr = np.asarray(list(LETTERS), dtype="S1")

    meta = dict(
        version=1,
        capture_date=datetime.datetime.now().isoformat(timespec="seconds"),
        subject=subject,
        layout_doc="Documents/SoftwareDocs/Tiagobot_Gaze_AI_Layout.md",
        confidence_threshold=float(conf_threshold),
        samples_per_target=int(samples_per_target),
        gaze_sample_width=GAZE_SAMPLE_W,
        gaze_sample_height=GAZE_SAMPLE_H,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        T=T,
        G=G,
        labels=labels,
        centroids=centroids,
        letters=letters_arr,
        meta=meta,
    )
    return T, G, labels, centroids


# ---- Main -------------------------------------------------------------
def _resolve_output_path(subject: str, override: Optional[str]) -> Path:
    if override:
        return Path(override).expanduser()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = Path(getattr(config, "DATA_DIR", ".")).expanduser()
    return data_dir / f"sub-{subject}" / "gaze" / f"gaze_to_letter_{stamp}.npz"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Capture Tiagobot A-I gaze calibration data. Writes a NPZ "
            "consumed by ExperimentDriver_Online_Tiagobot_Gaze.py."
        )
    )
    parser.add_argument(
        "--subject",
        default=getattr(config, "TRAINING_SUBJECT", "Debug"),
        help="Subject ID for the output path (default: config.TRAINING_SUBJECT).",
    )
    parser.add_argument(
        "--samples-per-target", type=int, default=SAMPLES_PER_TARGET,
        help=f"Samples to capture per letter (default: {SAMPLES_PER_TARGET}).",
    )
    parser.add_argument(
        "--conf-threshold", type=float,
        default=float(getattr(config, "TIAGOBOT_GAZE_CONFIDENCE_THRESHOLD", 0.7)),
        help="Minimum sample confidence to include.",
    )
    parser.add_argument(
        "--settle-seconds", type=float, default=SETTLE_SECONDS,
        help=f"Settle countdown before each target (default: {SETTLE_SECONDS}).",
    )
    parser.add_argument(
        "--target-timeout-s", type=float, default=DEFAULT_BAUD_TIMEOUT_S,
        help="Per-target wall-clock timeout in seconds.",
    )
    parser.add_argument(
        "--randomize", action="store_true",
        help="Randomise letter order (default: alphabetical A->I).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output NPZ path (default: {DATA_DIR}/sub-{SUBJECT}/gaze/"
             "gaze_to_letter_<stamp>.npz).",
    )
    parser.add_argument(
        "--windowed", action="store_true",
        help="Use a 1280x800 windowed mode instead of fullscreen "
             "(handy for local validation without a head-mounted Neon).",
    )
    args = parser.parse_args()

    out_path = _resolve_output_path(args.subject, args.out)

    # Pygame setup.
    pygame.init()
    if args.windowed:
        screen = pygame.display.set_mode((1280, 800))
    else:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.display.set_caption("Tiagobot Gaze Calibration")

    centroids_norm = grid_centroids_norm()

    # Try LSL gaze stream connection up front so the failure path is
    # before any UI countdown — fail-fast per CLAUDE.md.
    try:
        inlet = _connect_gaze_stream()
    except RuntimeError as e:
        _draw_message(screen, ["Gaze stream unavailable:", str(e), "", "Press any key to exit."])
        _ts_print(f"[CAL] ERROR: {e}")
        # Wait for keypress so the operator can read the error.
        waiting = True
        while waiting:
            for ev in pygame.event.get():
                if ev.type in (pygame.QUIT, pygame.KEYDOWN):
                    waiting = False
        pygame.quit()
        return 1

    letters_in_order = list(LETTERS)
    if args.randomize:
        random.shuffle(letters_in_order)

    _draw_message(
        screen,
        [
            "Tiagobot gaze calibration",
            f"Subject: {args.subject}",
            f"Letters: {', '.join(letters_in_order)}",
            f"Samples per target: {args.samples_per_target}",
            "",
            "Press SPACE to begin, ESC to abort.",
        ],
    )
    waiting_to_start = True
    while waiting_to_start:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                pygame.quit()
                return 1
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                waiting_to_start = False
        time.sleep(0.05)

    per_letter_samples: Dict[str, List[Tuple[float, float, float, float]]] = {}
    try:
        for ch in letters_in_order:
            _ts_print(f"[CAL] Starting target '{ch}'...")
            samples = _capture_target(
                screen, inlet, centroids_norm, ch,
                n_samples=args.samples_per_target,
                conf_threshold=args.conf_threshold,
                settle_s=args.settle_seconds,
                timeout_s=args.target_timeout_s,
            )
            per_letter_samples[ch] = samples
    except KeyboardInterrupt:
        _ts_print("[CAL] Aborted by operator.")
        pygame.quit()
        return 1

    # Summary screen while saving.
    _draw_message(screen, ["Saving calibration...", str(out_path)])
    T, G, labels, centroids = _save_calibration(
        out_path,
        per_letter_samples,
        conf_threshold=args.conf_threshold,
        samples_per_target=args.samples_per_target,
        subject=args.subject,
    )

    _ts_print("[CAL] Calibration written:")
    _ts_print(f"  path: {out_path}")
    _ts_print(f"  T.shape={T.shape}, G.shape={G.shape}, labels={labels.shape}")
    for i, ch in enumerate(LETTERS):
        cx, cy = centroids[i]
        n = int(np.sum(labels == ch.encode("ascii"))) if labels.size else 0
        if np.isfinite(cx) and np.isfinite(cy):
            _ts_print(f"  {ch}: n={n:3d}  centroid=({cx:.3f}, {cy:.3f})")
        else:
            _ts_print(f"  {ch}: n={n:3d}  centroid=NaN  (no valid samples)")

    _draw_message(
        screen,
        [
            "Calibration saved.",
            str(out_path),
            "",
            "Press any key to exit.",
        ],
    )
    waiting = True
    while waiting:
        for ev in pygame.event.get():
            if ev.type in (pygame.QUIT, pygame.KEYDOWN):
                waiting = False
        time.sleep(0.05)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
