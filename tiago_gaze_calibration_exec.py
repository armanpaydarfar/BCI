#!/usr/bin/env python3
# tiago_gaze_calibration_exec.py
"""
Tiagobot gaze calibration recorder.

Displays a 3x3 on-screen grid of A-I targets (per
`Documents/SoftwareDocs/projects/tiagobot/gaze-integration/ai-layout.md`)
and records the
user's gaze samples while they fixate each letter in turn. Writes a
calibration NPZ consumed at runtime by
`ExperimentDriver_Online_Tiagobot_Gaze.py` via `Utils.tiagobot_gaze`.

Gaze source: `Utils.gaze.gaze_system.GazeSystem` configured headless
(no CV, no display, no tracker), which dials the Pupil Labs Neon
Companion app over the realtime API at `config.NEON_COMPANION_HOST`.
This matches the rest of the repo's pupil_labs API usage
(`Utils/scene_only_neon_reader.py`, `gaze_runner.py`) and avoids a
dependency on an LSL relay on the phone (Neon Companion does not
publish an LSL outlet natively).

Output schema matches the contract `Utils.tiagobot_gaze.load_centroids`
expects:

    T          (n_samples,) float64   sample timestamps (unix seconds)
    G          (n_samples, 4) float32 [x_norm, y_norm, worn_flag, depth_cm]
                                      worn_flag is 1.0 (the realtime
                                      API has no per-sample confidence;
                                      we record the `worn` boolean
                                      promoted to float). depth_cm is
                                      GazeSystem's vergence depth, NaN
                                      when depth_valid was False at the
                                      time of capture.
    labels     (n_samples,)  S1       letter the sample belongs to
                                      ("A"-"I"; empty bytes for inter-
                                      target rest samples)
    centroids  (9, 3) float32         per-letter median of valid samples
                                      [x_norm, y_norm, depth_cm]. NaN x
                                      and y mean "no valid samples";
                                      NaN depth means "no sample had a
                                      valid vergence" (letter still
                                      usable in the 2D classifier).
    letters    (9,) S1                "A"..."I" row labels for centroids
    meta       dict                   version, capture date, subject,
                                      layout doc path, samples-per-target

Per CLAUDE.md fail-fast policy: a missing Neon device aborts at startup
(no silent recovery — calibration data is the contract that the runtime
classifier depends on).

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

import config
from Utils.tiagobot_gaze import LETTERS, grid_centroids_norm
from Utils.gaze.gaze_system import GazeConfig, GazeSystem


# ---- Calibration parameters -------------------------------------------
SAMPLES_PER_TARGET = 100            # ~3 s at the realtime API's ~30 Hz gaze rate
SETTLE_SECONDS = 5.0                # countdown before each target
DEFAULT_BAUD_TIMEOUT_S = 60.0       # max wall-clock per target capture
SNAPSHOT_POLL_HZ = 60.0             # how often we poll GazeSystem for a new sample

# Default Neon scene resolution. Pulled from config so calibration and
# runtime agree on the normalization denominator.
GAZE_SAMPLE_W = float(getattr(config, "GAZE_SAMPLE_WIDTH", 1600.0))
GAZE_SAMPLE_H = float(getattr(config, "GAZE_SAMPLE_HEIGHT", 1200.0))


# ---- Helpers ----------------------------------------------------------
def _ts_print(*args, **kwargs) -> None:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}]", *args, **kwargs)


def _connect_gaze_system() -> GazeSystem:
    """Connect to the Pupil Labs Neon Companion via the realtime API and
    return a started `GazeSystem` configured for headless gaze-only use.

    Disables the YOLO/SORT/display/tracker layers (we only need raw
    gaze pixels for the calibration centroid medians). `neon_host` is
    `config.NEON_COMPANION_HOST`; when empty, GazeSystem falls back to
    mDNS auto-discovery (works on home/hotspot, blocked on enterprise
    VLANs and tailscale — set the host explicitly on those networks).

    Raises whatever GazeSystem.start raises (typically `RuntimeError`
    "No Pupil Labs Neon device found") — fail-fast per CLAUDE.md so the
    operator sees a loud error rather than a silently-empty NPZ.
    """
    host = str(getattr(config, "NEON_COMPANION_HOST", "") or "")
    _ts_print(
        f"[GAZE] Connecting to Pupil Labs Neon via realtime API "
        f"(host={host!r}, mDNS if empty)..."
    )
    gs = GazeSystem(GazeConfig(
        enable_prints=False,
        enable_display=False,
        enable_cv=False,
        use_tracker=False,
        neon_host=host,
    ))
    gs.start()
    _ts_print("[GAZE] GazeSystem started.")
    return gs


def _read_latest_valid(
    gs: GazeSystem,
    last_unix_t: Optional[float],
) -> Tuple[Optional[Tuple[float, float, float, float, float]], Optional[float]]:
    """Read the current gaze snapshot from `gs`; return the sample if
    it is fresh (newer `unix_t` than `last_unix_t`) and the user is
    wearing the glasses, else None.

    Returns ``(sample_or_None, new_last_unix_t)``. The caller threads
    `new_last_unix_t` back in on the next call so we naturally dedupe
    duplicates when polling faster than the Neon realtime API rate
    (~30 Hz delivered; we poll at SNAPSHOT_POLL_HZ).

    Sample tuple is ``(t_unix, x_norm, y_norm, worn_flag, depth_cm)``.
    `worn_flag` is always 1.0 (samples with worn=False are dropped
    here). `depth_cm` is `gs`'s vergence depth, or NaN when the
    snapshot's `depth_valid` is False — that lets us keep the sample
    for x/y averaging and let the classifier degrade to 2D for that
    trial only.
    """
    snap = gs.get_snapshot(include_objects=False, include_frame=False)
    if not snap or not snap.get("ok"):
        return None, last_unix_t
    t_unix = snap.get("unix_t")
    if t_unix is None:
        return None, last_unix_t
    if last_unix_t is not None and t_unix <= last_unix_t:
        # Same sample as last poll — Neon hasn't produced a new one yet.
        return None, last_unix_t
    if not bool(snap.get("worn")):
        return None, float(t_unix)
    px = snap.get("gaze_px_raw")
    if px is None:
        return None, float(t_unix)
    x_px, y_px = float(px[0]), float(px[1])
    if not (np.isfinite(x_px) and np.isfinite(y_px)):
        return None, float(t_unix)
    depth_cm = float("nan")
    if bool(snap.get("depth_valid")):
        d = snap.get("depth_cm")
        if d is not None:
            df = float(d)
            if np.isfinite(df):
                depth_cm = df
    return (
        float(t_unix),
        x_px / GAZE_SAMPLE_W,
        y_px / GAZE_SAMPLE_H,
        1.0,
        depth_cm,
    ), float(t_unix)


def _pump_events_quit() -> bool:
    """Return True if a quit / ESC was requested."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return True
    return False


# ---- Drawing ----------------------------------------------------------
def _draw_fixation_cross(screen: "pygame.Surface") -> None:
    """Draw a small central fixation cross. Always rendered so the
    subject has a single fixed point to anchor their HEAD pose against.
    Head-fixed / eyes-only is the whole protocol contract for the
    on-screen calibration mode (added 2026-05-21): the user keeps their
    head pointed at this cross and only their eyes move to the
    highlighted target."""
    W, H = screen.get_size()
    cx, cy = W // 2, H // 2
    arm = 18
    color = (200, 200, 80)  # yellow-green to distinguish from grid letters
    pygame.draw.line(screen, color, (cx - arm, cy), (cx + arm, cy), 3)
    pygame.draw.line(screen, color, (cx, cy - arm), (cx, cy + arm), 3)


def _draw_grid(
    screen: "pygame.Surface",
    centroids_norm: Dict[str, Tuple[float, float]],
    active_letter: Optional[str],
    countdown_text: Optional[str] = None,
) -> None:
    """Render the 3x3 grid: highlight the active letter, dim the rest,
    keep a central fixation cross visible at all times.

    Used during calibration so the subject sees where to look next while
    keeping their head pointed at the cross.
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

    # Fixation cross is the head-pose anchor — draw it last so it's on
    # top, even if a centre cell's halo would otherwise occlude it.
    _draw_fixation_cross(screen)

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
    gs: GazeSystem,
    centroids_norm: Dict[str, Tuple[float, float]],
    letter: str,
    n_samples: int,
    settle_s: float,
    timeout_s: float,
) -> List[Tuple[float, float, float, float]]:
    """Capture `n_samples` gaze samples while the subject fixates
    `letter`. Returns the list of `(t_unix, x_norm, y_norm, worn_flag)`.

    Polls GazeSystem.get_snapshot at SNAPSHOT_POLL_HZ and dedupes by
    `unix_t`. The realtime API delivers gaze at ~30 Hz, so a 60 Hz poll
    yields ~30 unique samples/s — N=100 in ~3 s typical. Samples with
    `worn=False` are discarded (no per-sample confidence in the
    realtime API; `worn` is the only validity signal).
    """
    poll_period = 1.0 / SNAPSHOT_POLL_HZ
    last_unix_t: Optional[float] = None

    # Pre-target gate: wait for SPACE. The user sees the next target
    # highlighted but is supposed to keep their head pointed at the
    # central cross. They press SPACE when ready, then dart their
    # EYES (not their head) to the highlighted letter during the
    # settle countdown. ESC aborts the run.
    _draw_grid(
        screen, centroids_norm, letter,
        countdown_text=(
            f"Head on cross — eyes-only to '{letter}'. "
            f"SPACE when ready (ESC to abort)"
        ),
    )
    waiting = True
    while waiting:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (
                ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE
            ):
                raise KeyboardInterrupt
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                waiting = False
        time.sleep(0.02)

    # Settle countdown. We still call _read_latest_valid (discarding
    # the result) so last_unix_t advances to "now" — that way the
    # capture phase only sees samples produced AFTER the subject has
    # finished moving their gaze to the new target.
    settle_end = time.monotonic() + settle_s
    while time.monotonic() < settle_end:
        if _pump_events_quit():
            raise KeyboardInterrupt
        remaining = settle_end - time.monotonic()
        _draw_grid(
            screen, centroids_norm, letter,
            countdown_text=f"Eyes on '{letter}' (head still) — capture in {remaining:0.1f} s",
        )
        _, last_unix_t = _read_latest_valid(gs, last_unix_t)
        time.sleep(poll_period)

    # Capture phase.
    samples: List[Tuple[float, float, float, float]] = []
    capture_start = time.monotonic()
    capture_deadline = capture_start + timeout_s
    last_log = capture_start
    while len(samples) < n_samples and time.monotonic() < capture_deadline:
        if _pump_events_quit():
            raise KeyboardInterrupt
        s, last_unix_t = _read_latest_valid(gs, last_unix_t)
        if s is not None:
            samples.append(s)
        progress = len(samples) / max(1, n_samples)
        _draw_grid(
            screen, centroids_norm, letter,
            countdown_text=f"Capturing '{letter}' — {len(samples)}/{n_samples} samples",
        )
        time.sleep(poll_period)
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
    samples_per_target: int,
    subject: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten per-letter samples into NPZ arrays and write to disk.

    Returns (T, G, labels, centroids) for the caller's stat printout.
    """
    T_list: List[float] = []
    G_list: List[Tuple[float, float, float, float]] = []
    label_list: List[str] = []
    centroids = np.full((len(LETTERS), 3), np.nan, dtype=np.float32)

    for i, ch in enumerate(LETTERS):
        samples = per_letter_samples.get(ch, [])
        if samples:
            for (t, x, y, c, d) in samples:
                T_list.append(t)
                G_list.append((x, y, c, d))
                label_list.append(ch)
            xs = np.array([s[1] for s in samples], dtype=np.float64)
            ys = np.array([s[2] for s in samples], dtype=np.float64)
            # Depth may have NaN entries (samples where depth_valid was
            # False at capture time). Median over finite values only;
            # leave NaN in the centroid when no sample had valid depth.
            ds = np.array(
                [s[4] for s in samples if np.isfinite(s[4])], dtype=np.float64
            )
            centroids[i, 0] = float(np.median(xs))
            centroids[i, 1] = float(np.median(ys))
            if ds.size:
                centroids[i, 2] = float(np.median(ds))

    T = np.asarray(T_list, dtype=np.float64)
    G = np.asarray(G_list, dtype=np.float32) if G_list else np.zeros((0, 4), dtype=np.float32)
    labels = np.asarray(label_list, dtype="S1")
    letters_arr = np.asarray(list(LETTERS), dtype="S1")

    meta = dict(
        version=1,
        capture_date=datetime.datetime.now().isoformat(timespec="seconds"),
        subject=subject,
        layout_doc="Documents/SoftwareDocs/projects/tiagobot/gaze-integration/ai-layout.md",
        gaze_source="pupil_labs_realtime_api",
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

    # Try the Neon connection up front so the failure path is before
    # any UI countdown — fail-fast per CLAUDE.md.
    try:
        gs = _connect_gaze_system()
    except Exception as e:
        _draw_message(screen, ["Neon device unavailable:", str(e), "", "Press any key to exit."])
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
            "Tiagobot gaze calibration  (head-fixed / eyes-only mode)",
            "",
            "Keep your HEAD pointed at the central cross at all times.",
            "Only your EYES should move to each highlighted letter.",
            "",
            f"Subject: {args.subject}    Letters: {', '.join(letters_in_order)}",
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

    # Snapshot head pose at the start of the session — we'll compare
    # every per-target head pose against this reference and warn (not
    # block) if the user has drifted. The whole on-screen calibration
    # protocol assumes head is fixed; if it isn't, the centroids learned
    # here will not match runtime.
    _ref_snap = gs.get_snapshot(include_objects=False, include_frame=False) or {}
    ref_yaw = float(_ref_snap.get("head_yaw_deg") or 0.0)
    ref_pitch = float(_ref_snap.get("head_pitch_deg") or 0.0)
    if np.isfinite(ref_yaw) and np.isfinite(ref_pitch):
        _ts_print(
            f"[CAL] Head-pose reference at session start: "
            f"yaw={ref_yaw:+.1f}°  pitch={ref_pitch:+.1f}°"
        )
    else:
        _ts_print(
            "[CAL] WARN: head-pose unavailable at session start — "
            "IMU may not have warmed up; drift checks will be skipped."
        )
        ref_yaw = float("nan")
        ref_pitch = float("nan")

    per_letter_samples: Dict[str, List[Tuple[float, float, float, float, float]]] = {}
    try:
        for ch in letters_in_order:
            # Log head-pose delta from the session reference so any
            # head drift between targets shows up in the trace.
            snap = gs.get_snapshot(include_objects=False, include_frame=False) or {}
            yaw_now = snap.get("head_yaw_deg")
            pitch_now = snap.get("head_pitch_deg")
            if (
                yaw_now is not None and pitch_now is not None
                and np.isfinite(ref_yaw) and np.isfinite(ref_pitch)
                and np.isfinite(float(yaw_now)) and np.isfinite(float(pitch_now))
            ):
                dy = float(yaw_now) - ref_yaw
                dp = float(pitch_now) - ref_pitch
                warn = "  <-- HEAD DRIFT" if (abs(dy) > 3.0 or abs(dp) > 3.0) else ""
                _ts_print(
                    f"[CAL] Starting target '{ch}'... "
                    f"head Δ=(yaw {dy:+.1f}°, pitch {dp:+.1f}°){warn}"
                )
            else:
                _ts_print(f"[CAL] Starting target '{ch}'...")
            samples = _capture_target(
                screen, gs, centroids_norm, ch,
                n_samples=args.samples_per_target,
                settle_s=args.settle_seconds,
                timeout_s=args.target_timeout_s,
            )
            per_letter_samples[ch] = samples
    except KeyboardInterrupt:
        _ts_print("[CAL] Aborted by operator.")
        gs.stop()
        pygame.quit()
        return 1

    # GazeSystem is no longer needed once capture completes; stop the
    # background threads before the (potentially long) save + summary UI.
    gs.stop()

    # Summary screen while saving.
    _draw_message(screen, ["Saving calibration...", str(out_path)])
    T, G, labels, centroids = _save_calibration(
        out_path,
        per_letter_samples,
        samples_per_target=args.samples_per_target,
        subject=args.subject,
    )

    _ts_print("[CAL] Calibration written:")
    _ts_print(f"  path: {out_path}")
    _ts_print(f"  T.shape={T.shape}, G.shape={G.shape}, labels={labels.shape}")
    for i, ch in enumerate(LETTERS):
        cx, cy, cz = centroids[i]
        n = int(np.sum(labels == ch.encode("ascii"))) if labels.size else 0
        if np.isfinite(cx) and np.isfinite(cy):
            depth_str = f"{cz:.1f}cm" if np.isfinite(cz) else "NaN"
            _ts_print(
                f"  {ch}: n={n:3d}  centroid=({cx:.3f}, {cy:.3f}, {depth_str})"
            )
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
