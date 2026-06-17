#!/usr/bin/env python3
# tools/neon_lsl_bridge.py
"""
Bridge Pupil Labs Neon gaze samples into LSL outlets so LabRecorder can
capture them alongside the EEG stream into a single XDF.

Why this exists: the Companion app's on-device "LSL" toggle does not
actually publish an externally-discoverable LSL stream (verified via
discovery probe + TCP banner — the open ports speak something other
than LSL). The Pupil Labs documented path for Neon -> LSL is a
desktop-side Python relay on top of the realtime API.

The bridge publishes TWO outlets so XDF consumers can disentangle
continuous gaze samples from variable-rate eye events:

  1. `NeonGaze` — 29-channel float32, nominal 30 Hz. Per-sample gaze
     and eye-state from `device.receive_gaze_datum()`. Channel 0-4
     match the original 5-channel schema (gaze_x_px, gaze_y_px, worn,
     depth_cm, unix_t) so XDFs recorded before 2026-05-27 can be
     loaded by the same column-name lookup. Channels 5-28 are new
     additions covering pupillometry, eyelid aperture, per-eye gaze,
     eyeball center, and optical-axis vectors.

  2. `NeonEvents` — single-channel string, irregular rate. Each LSL
     sample is a JSON dump of one event from
     `device.receive_eye_events()` (fixations, blinks, saccades).
     Empty stream is normal until the Pupil Labs realtime API
     actually delivers an event.

Usage:
    conda run -n lsl python tools/neon_lsl_bridge.py
    # LabRecorder will now see two streams ('NeonGaze' + 'NeonEvents')
    # alongside the eegoSports streams. Ctrl-C to stop.

Sampling rate: ~30 Hz delivered (realtime API limit, not Neon's
200 Hz native). Good enough for trial-level sync; for higher-rate
gaze, record directly on the phone and post-hoc align via the
`unix_t` column.

depth_cm is recomputed in this process using the
`Utils.gaze.gaze_math.vergence_depth_from_eyestate` routine — keeps
column 3 semantically identical so existing XDFs read the same.

Pupillometry is a slow *state* biosignal, not a trial-level MI control
signal (WS6 found cue-locked MI-vs-Rest AUC ≈ chance). This bridge is
for *recording* gaze + pupillometry to XDF, not for closed-loop decode.

Lives under tools/ — not Tier 1/2. Doesn't touch the closed-loop
driver. Safe to start/stop independently.

CAUTION — dual subscription to one Companion phone (WS4 F6, untested):
this bridge opens its OWN realtime-api `Device` to the phone. When the
perception stack is also running, `Utils/frame_relay.py` already holds a
subscription to the same Companion phone (for scene frames + gaze). Two
concurrent subscribers to one phone is **not yet bench-verified** — run
the bridge alongside a live frame_relay once on hardware and confirm
neither drops frames / refuses the second connection before relying on
simultaneous use. If the phone can't serve both, fan out from a single
subscription instead of opening a second `Device` here. Run standalone
(no frame_relay) is unaffected.
"""
from __future__ import annotations

import argparse
import datetime
import json
import signal
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pylsl  # noqa: E402
from pupil_labs.realtime_api.simple import Device  # noqa: E402

import config  # noqa: E402
from Utils.gaze.gaze_math import vergence_depth_from_eyestate  # noqa: E402


# Polling rate against device.receive_gaze_datum(). The realtime API
# blocks until the next sample arrives (~30 Hz), so this is really just
# a max-iteration cap; in practice the thread runs at whatever Neon
# delivers.
GAZE_POLL_TIMEOUT_S = 1.0
EVENTS_POLL_TIMEOUT_S = 2.0
STATS_INTERVAL_S = 5.0

# NeonGaze channel layout — stable contract for XDF consumers. The
# first 5 entries match the 2026-05-22 schema verbatim so any code
# reading the original 5-channel stream by column index keeps working.
# Channels 5-28 are pure additions; downstream code should read by
# channel label (in the LSL desc/channels metadata) rather than by
# fixed column index.
#
# Entries are (label, unit, lsl_type, datum_attribute_or_None).
# datum_attribute_or_None=None marks the two synthetic channels (worn,
# depth_cm) that the gaze loop fills directly rather than copying from a gaze
# datum attribute. unix_t IS a direct copy (attr="timestamp_unix_seconds").
GAZE_CHANNELS = [
    # ----- Original 5 (unchanged for backward compat) -----
    ("gaze_x_px",                 "pixels",  "PositionX", "x"),
    ("gaze_y_px",                 "pixels",  "PositionY", "y"),
    ("worn",                      "bool",    "Misc",      None),  # 1/0 from .worn
    ("depth_cm",                  "cm",      "Distance",  None),  # computed
    ("unix_t",                    "seconds", "Misc",      "timestamp_unix_seconds"),
    # ----- Pupillometry -----
    ("pupil_diameter_left_mm",    "mm",      "Pupil",     "pupil_diameter_left"),
    ("pupil_diameter_right_mm",   "mm",      "Pupil",     "pupil_diameter_right"),
    # ----- Eyelid aperture (blink detection) -----
    ("eyelid_aperture_left_mm",   "mm",      "Eyelid",    "eyelid_aperture_left"),
    ("eyelid_aperture_right_mm",  "mm",      "Eyelid",    "eyelid_aperture_right"),
    # ----- Per-eye monocular gaze (binocular x/y is the average) -----
    ("gaze_mono_left_x_px",       "pixels",  "PositionX", "gaze_mono_left_x"),
    ("gaze_mono_left_y_px",       "pixels",  "PositionY", "gaze_mono_left_y"),
    ("gaze_mono_right_x_px",      "pixels",  "PositionX", "gaze_mono_right_x"),
    ("gaze_mono_right_y_px",      "pixels",  "PositionY", "gaze_mono_right_y"),
    # ----- Eyelid angles (squint / asymmetric closure) -----
    ("eyelid_angle_top_left",     "rad",     "Eyelid",    "eyelid_angle_top_left"),
    ("eyelid_angle_top_right",    "rad",     "Eyelid",    "eyelid_angle_top_right"),
    ("eyelid_angle_bottom_left",  "rad",     "Eyelid",    "eyelid_angle_bottom_left"),
    ("eyelid_angle_bottom_right", "rad",     "Eyelid",    "eyelid_angle_bottom_right"),
    # ----- 3D eye geometry: eyeball center in head frame (mm) -----
    ("eyeball_center_left_x_mm",  "mm",      "EyePosition", "eyeball_center_left_x"),
    ("eyeball_center_left_y_mm",  "mm",      "EyePosition", "eyeball_center_left_y"),
    ("eyeball_center_left_z_mm",  "mm",      "EyePosition", "eyeball_center_left_z"),
    ("eyeball_center_right_x_mm", "mm",      "EyePosition", "eyeball_center_right_x"),
    ("eyeball_center_right_y_mm", "mm",      "EyePosition", "eyeball_center_right_y"),
    ("eyeball_center_right_z_mm", "mm",      "EyePosition", "eyeball_center_right_z"),
    # ----- 3D eye geometry: optical axis unit vector per eye -----
    ("optical_axis_left_x",       "unit",    "GazeDirection", "optical_axis_left_x"),
    ("optical_axis_left_y",       "unit",    "GazeDirection", "optical_axis_left_y"),
    ("optical_axis_left_z",       "unit",    "GazeDirection", "optical_axis_left_z"),
    ("optical_axis_right_x",      "unit",    "GazeDirection", "optical_axis_right_x"),
    ("optical_axis_right_y",      "unit",    "GazeDirection", "optical_axis_right_y"),
    ("optical_axis_right_z",      "unit",    "GazeDirection", "optical_axis_right_z"),
]


def _build_gaze_outlet(name: str, source_id: str) -> pylsl.StreamOutlet:
    """Construct the multi-channel NeonGaze LSL outlet."""
    info = pylsl.StreamInfo(
        name=name,
        type="Gaze",
        channel_count=len(GAZE_CHANNELS),
        nominal_srate=30.0,
        channel_format=pylsl.cf_float32,
        source_id=source_id,
    )
    chans = info.desc().append_child("channels")
    for label, unit, lsl_type, _datum_attr in GAZE_CHANNELS:
        ch = chans.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", unit)
        ch.append_child_value("type", lsl_type)
    acq = info.desc().append_child("acquisition")
    acq.append_child_value("manufacturer", "Pupil Labs")
    acq.append_child_value("model", "Neon")
    acq.append_child_value("transport", "realtime_api_via_lsl_bridge")
    return pylsl.StreamOutlet(info)


def _build_events_outlet(name: str, source_id: str) -> pylsl.StreamOutlet:
    """Construct the variable-rate NeonEvents LSL outlet (single string
    channel, each sample is a JSON dump of a Pupil eye event)."""
    info = pylsl.StreamInfo(
        name=name,
        type="Markers",
        channel_count=1,
        nominal_srate=pylsl.IRREGULAR_RATE,
        channel_format=pylsl.cf_string,
        source_id=source_id,
    )
    chans = info.desc().append_child("channels")
    ch = chans.append_child("channel")
    ch.append_child_value("label", "event_json")
    ch.append_child_value("unit", "json")
    ch.append_child_value("type", "EyeEvent")
    acq = info.desc().append_child("acquisition")
    acq.append_child_value("manufacturer", "Pupil Labs")
    acq.append_child_value("model", "Neon")
    acq.append_child_value("transport", "realtime_api_via_lsl_bridge")
    acq.append_child_value(
        "schema",
        "Each sample is a JSON object dumped from the Pupil "
        "receive_eye_events() result. Field set is event-type-dependent "
        "(fixation / blink / saccade) and may vary across SDK versions.",
    )
    return pylsl.StreamOutlet(info)


def _gaze_loop(
    device: Device,
    outlet: pylsl.StreamOutlet,
    stop: threading.Event,
) -> None:
    """Worker thread: pull gaze datums, push 29-channel samples to LSL.

    `receive_gaze_datum(timeout_seconds=...)` blocks until a sample
    arrives or the timeout expires. Returning None on timeout lets the
    stop flag get checked between samples without burning CPU.
    """
    n_pushed = 0
    last_stats = time.monotonic()
    sample = [0.0] * len(GAZE_CHANNELS)

    while not stop.is_set():
        try:
            g = device.receive_gaze_datum(timeout_seconds=GAZE_POLL_TIMEOUT_S)
        except Exception as e:
            print(f"[gaze] receive error: {e!r}", flush=True)
            continue
        if g is None:
            continue

        # depth_cm from vergence — same computation as
        # GazeSystem (vergence_depth_from_eyestate), inlined so the
        # bridge has no GazeSystem dependency. Returns NaN when the
        # eyestate doesn't yield a valid vergence (rays don't cross,
        # eye state missing, etc.).
        try:
            L = np.array([
                float(getattr(g, "eyeball_center_left_x", 0.0)),
                float(getattr(g, "eyeball_center_left_y", 0.0)),
                float(getattr(g, "eyeball_center_left_z", 0.0)),
            ])
            u = np.array([
                float(getattr(g, "optical_axis_left_x", 0.0)),
                float(getattr(g, "optical_axis_left_y", 0.0)),
                float(getattr(g, "optical_axis_left_z", 0.0)),
            ])
            R = np.array([
                float(getattr(g, "eyeball_center_right_x", 0.0)),
                float(getattr(g, "eyeball_center_right_y", 0.0)),
                float(getattr(g, "eyeball_center_right_z", 0.0)),
            ])
            v = np.array([
                float(getattr(g, "optical_axis_right_x", 0.0)),
                float(getattr(g, "optical_axis_right_y", 0.0)),
                float(getattr(g, "optical_axis_right_z", 0.0)),
            ])
            valid, depth_m, _miss_mm = vergence_depth_from_eyestate(L, u, R, v)
            depth_cm = float(depth_m) * 100.0 if valid else float("nan")
        except Exception:
            depth_cm = float("nan")

        worn = 1.0 if bool(getattr(g, "worn", False)) else 0.0

        for i, (_label, _unit, _type, attr) in enumerate(GAZE_CHANNELS):
            if attr is None:
                # Synthetic channels handled below
                continue
            try:
                sample[i] = float(getattr(g, attr, float("nan")))
            except (TypeError, ValueError):
                sample[i] = float("nan")

        # Synthetic channels (index by position; matches GAZE_CHANNELS
        # entries whose attr=None).
        sample[2] = worn       # worn
        sample[3] = depth_cm   # depth_cm

        try:
            outlet.push_sample(sample)
            n_pushed += 1
        except Exception as e:
            print(f"[gaze] push error: {e!r}", flush=True)

        now = time.monotonic()
        if now - last_stats >= STATS_INTERVAL_S:
            dt = now - last_stats
            rate = n_pushed / dt if dt > 0 else 0.0
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            print(
                f"[gaze {ts}] pushed {n_pushed} samples in {dt:.1f}s (~{rate:.1f} Hz)",
                flush=True,
            )
            n_pushed = 0
            last_stats = now


def _events_loop(
    device: Device,
    outlet: pylsl.StreamOutlet,
    stop: threading.Event,
) -> None:
    """Worker thread: pull eye events, push as JSON markers.

    Empty stream is normal — eye events fire only when Pupil's
    realtime detector emits a fixation/blink/saccade. On a still
    operator with eyes open, expect zero events for long stretches.
    """
    n_pushed = 0
    last_stats = time.monotonic()

    while not stop.is_set():
        try:
            evt = device.receive_eye_events(timeout_seconds=EVENTS_POLL_TIMEOUT_S)
        except Exception as e:
            print(f"[events] receive error: {e!r}", flush=True)
            continue
        if evt is None:
            # No event in the timeout window — log the heartbeat so the
            # operator knows the thread is alive but the API just hasn't
            # produced anything yet.
            now = time.monotonic()
            if now - last_stats >= STATS_INTERVAL_S:
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                print(
                    f"[events {ts}] no events in last {now-last_stats:.1f}s "
                    f"(total this session: {n_pushed})",
                    flush=True,
                )
                last_stats = now
            continue

        # Serialise whatever fields the event object has into JSON. Use
        # vars(evt) when available; fall back to a getattr sweep over
        # public attributes for objects that don't have __dict__.
        try:
            payload = vars(evt)
        except TypeError:
            payload = {
                k: getattr(evt, k)
                for k in dir(evt)
                if not k.startswith("_") and not callable(getattr(evt, k))
            }
        payload["__type"] = type(evt).__name__

        def _json_default(o):
            try:
                return float(o)
            except (TypeError, ValueError):
                return str(o)

        try:
            event_json = json.dumps(payload, default=_json_default)
        except Exception as e:
            event_json = json.dumps(
                {"__type": type(evt).__name__, "__error": str(e)}
            )

        try:
            outlet.push_sample([event_json])
            n_pushed += 1
        except Exception as e:
            print(f"[events] push error: {e!r}", flush=True)
            continue

        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(
            f"[events {ts}] pushed event #{n_pushed}: {payload.get('__type')!r}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Republish Pupil Labs Neon gaze + eye events as LSL outlets "
            "so LabRecorder can capture them into the EEG XDF."
        )
    )
    parser.add_argument(
        "--gaze-name", default="NeonGaze",
        help="LSL stream name for the gaze outlet (default: NeonGaze).",
    )
    parser.add_argument(
        "--events-name", default="NeonEvents",
        help="LSL stream name for the eye-events outlet (default: NeonEvents).",
    )
    parser.add_argument(
        "--source-id", default=None,
        help="LSL source_id prefix. Default derives from "
             "NEON_COMPANION_HOST so a phone re-IPing changes the "
             "source_id and LSL won't merge old + new sessions. Final "
             "source_ids: '{prefix}-gaze' and '{prefix}-events'.",
    )
    parser.add_argument(
        "--no-events", action="store_true",
        help="Skip the NeonEvents outlet. Useful if you only want the "
             "continuous gaze stream and have no use for fixation / "
             "blink / saccade markers.",
    )
    args = parser.parse_args()

    host = str(getattr(config, "NEON_COMPANION_HOST", "") or "")
    source_prefix = args.source_id or (
        f"neon-{host}".replace(".", "_") if host else "neon-mdns"
    )

    gaze_outlet = _build_gaze_outlet(args.gaze_name, f"{source_prefix}-gaze")
    print(
        f"[bridge] gaze outlet ready: name={args.gaze_name!r}  "
        f"source_id={source_prefix}-gaze  channels={len(GAZE_CHANNELS)}",
        flush=True,
    )
    for i, (label, unit, _type, _attr) in enumerate(GAZE_CHANNELS):
        print(f"  ch {i:2d}: {label:32s} unit={unit}", flush=True)

    events_outlet = None
    if not args.no_events:
        events_outlet = _build_events_outlet(args.events_name, f"{source_prefix}-events")
        print(
            f"[bridge] events outlet ready: name={args.events_name!r}  "
            f"source_id={source_prefix}-events  format=string, IRREGULAR_RATE",
            flush=True,
        )

    print(f"[bridge] Connecting to Neon at {host!r} (mDNS if empty)...", flush=True)
    if host:
        device = Device(address=host, port=8080)
    else:
        # mDNS path — left here for portability but in practice the
        # tailnet config always sets NEON_COMPANION_HOST.
        from pupil_labs.realtime_api.simple import discover_one_device
        device = discover_one_device(max_search_duration_seconds=10)
        if device is None:
            print("[bridge] no Neon discovered via mDNS; aborting.", file=sys.stderr)
            return 1
    print(f"[bridge] Connected to {device}. Streaming. Ctrl-C to stop.", flush=True)

    stop = threading.Event()
    def _on_sig(_signum, _frame):
        stop.set()
    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)

    gaze_thread = threading.Thread(
        target=_gaze_loop, args=(device, gaze_outlet, stop),
        daemon=True, name="neon-gaze-bridge",
    )
    gaze_thread.start()

    events_thread = None
    if events_outlet is not None:
        events_thread = threading.Thread(
            target=_events_loop, args=(device, events_outlet, stop),
            daemon=True, name="neon-events-bridge",
        )
        events_thread.start()

    # Main thread parks on the stop signal so SIGINT / SIGTERM
    # propagate cleanly to the worker threads.
    try:
        while not stop.is_set():
            time.sleep(0.5)
    finally:
        print("[bridge] stopping...", flush=True)
        stop.set()
        gaze_thread.join(timeout=2.0)
        if events_thread is not None:
            events_thread.join(timeout=2.0)
        try:
            device.close()
        except Exception as e:
            print(f"[bridge] device.close error: {e!r}", flush=True)
        print("[bridge] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
