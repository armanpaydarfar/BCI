#!/usr/bin/env python3
# tools/neon_lsl_bridge.py
"""
Bridge Pupil Labs Neon gaze samples into an LSL outlet so LabRecorder
can capture them alongside the EEG stream into a single XDF.

Why this exists: the Companion app's on-device "LSL" toggle does not
actually publish an externally-discoverable LSL stream (verified via
discovery probe + TCP banner — the open ports speak something other
than LSL). The Pupil Labs documented path for Neon -> LSL is a
desktop-side Python relay on top of the realtime API. We already have
that realtime API wrapped in `Utils.gaze.gaze_system.GazeSystem`; this
script is the LSL outlet half.

Usage:
    /home/millanslab/opt/miniconda/envs/lsl/bin/python \
        tools/neon_lsl_bridge.py
    # LabRecorder will now see a 'NeonGaze' stream alongside the
    # eegoSports streams. Ctrl-C to stop.

Sampling rate: ~30 Hz (realtime API delivery rate, not Neon's 200 Hz
native). Good enough for trial-level sync with EEG; not fine-grained
enough for saccade-onset analysis. If you need higher-rate gaze,
record directly to the phone and post-hoc align via the unix_t column
included in the stream.

Lives under tools/ — not Tier 1/2. Doesn't touch the closed-loop
driver. Safe to start/stop independently.
"""
from __future__ import annotations

import argparse
import datetime
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pylsl  # noqa: E402

import config  # noqa: E402
from Utils.gaze.gaze_system import GazeConfig, GazeSystem  # noqa: E402


# Polling rate against GazeSystem.get_snapshot(). The realtime API
# delivers gaze at ~30 Hz; polling at 60 Hz catches each fresh sample
# once without duplicate pushes (we dedupe by unix_t).
POLL_HZ = 60.0
STATS_INTERVAL_S = 5.0

# Channel layout — keep stable so XDF consumers can rely on column
# order. Each entry is (label, unit, lsl_type).
CHANNELS = [
    ("gaze_x_px", "pixels", "PositionX"),
    ("gaze_y_px", "pixels", "PositionY"),
    ("worn",      "bool",   "Misc"),
    ("depth_cm",  "cm",     "Distance"),
    ("unix_t",    "seconds", "Misc"),
]


def _build_outlet(name: str, source_id: str) -> pylsl.StreamOutlet:
    """Construct an LSL outlet with the documented channel layout."""
    info = pylsl.StreamInfo(
        name=name,
        type="Gaze",
        channel_count=len(CHANNELS),
        nominal_srate=30.0,
        channel_format=pylsl.cf_float32,
        source_id=source_id,
    )
    chans = info.desc().append_child("channels")
    for label, unit, lsl_type in CHANNELS:
        ch = chans.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", unit)
        ch.append_child_value("type", lsl_type)
    # Acquisition metadata so XDF consumers can identify the source.
    acq = info.desc().append_child("acquisition")
    acq.append_child_value("manufacturer", "Pupil Labs")
    acq.append_child_value("model", "Neon")
    acq.append_child_value("transport", "realtime_api_via_lsl_bridge")
    return pylsl.StreamOutlet(info)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Republish Pupil Labs Neon gaze samples as an LSL outlet so "
            "LabRecorder can capture them into the EEG XDF."
        )
    )
    parser.add_argument(
        "--name", default="NeonGaze",
        help="LSL stream name (default: NeonGaze).",
    )
    parser.add_argument(
        "--source-id", default=None,
        help="LSL source_id. Default derives from NEON_COMPANION_HOST "
             "so a phone re-IPing changes the source_id and LSL won't "
             "merge old + new sessions.",
    )
    args = parser.parse_args()

    host = str(getattr(config, "NEON_COMPANION_HOST", "") or "")
    source_id = args.source_id or (
        f"neon-{host}".replace(".", "_") if host else "neon-mdns"
    )

    outlet = _build_outlet(args.name, source_id)
    print(
        f"[bridge] LSL outlet ready: name={args.name!r}  source_id={source_id!r}  "
        f"channels={[c[0] for c in CHANNELS]}",
        flush=True,
    )

    print(f"[bridge] Connecting to Neon (host={host!r}, mDNS if empty)...", flush=True)
    gs = GazeSystem(GazeConfig(
        enable_prints=False, enable_display=False, enable_cv=False,
        use_tracker=False, neon_host=host,
    ))
    gs.start()
    print("[bridge] Connected. Streaming. Ctrl-C to stop.", flush=True)

    # Graceful shutdown on SIGINT/SIGTERM so LabRecorder sees a clean
    # outlet close (not a TCP reset).
    stop = {"flag": False}
    def _on_sig(_signum, _frame):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)

    poll_period = 1.0 / POLL_HZ
    last_unix_t: float | None = None
    n_pushed = 0
    n_skipped_invalid = 0
    last_stats = time.monotonic()

    try:
        while not stop["flag"]:
            snap = gs.get_snapshot(include_objects=False, include_frame=False)
            if snap and snap.get("ok"):
                t = snap.get("unix_t")
                if t is not None and (last_unix_t is None or float(t) > last_unix_t):
                    px = snap.get("gaze_px_raw")
                    if px is not None:
                        x_px = float(px[0])
                        y_px = float(px[1])
                        if np.isfinite(x_px) and np.isfinite(y_px):
                            worn_f = 1.0 if bool(snap.get("worn")) else 0.0
                            depth_v = snap.get("depth_cm")
                            depth_cm = (
                                float(depth_v)
                                if depth_v is not None
                                and bool(snap.get("depth_valid"))
                                and np.isfinite(float(depth_v))
                                else float("nan")
                            )
                            outlet.push_sample(
                                [x_px, y_px, worn_f, depth_cm, float(t)]
                            )
                            n_pushed += 1
                            last_unix_t = float(t)
                        else:
                            n_skipped_invalid += 1
                            last_unix_t = float(t)

            now = time.monotonic()
            if now - last_stats >= STATS_INTERVAL_S:
                dt = now - last_stats
                rate = n_pushed / dt if dt > 0 else 0.0
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                print(
                    f"[bridge {ts}] pushed {n_pushed} samples in {dt:.1f}s "
                    f"(~{rate:.1f} Hz); skipped {n_skipped_invalid} invalid",
                    flush=True,
                )
                n_pushed = 0
                n_skipped_invalid = 0
                last_stats = now

            time.sleep(poll_period)
    finally:
        print("[bridge] stopping GazeSystem...", flush=True)
        try:
            gs.stop()
        except Exception as e:
            print(f"[bridge] GazeSystem.stop error: {e}", flush=True)
        print("[bridge] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
