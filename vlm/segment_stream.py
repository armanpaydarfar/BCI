"""
vlm/segment_stream.py — continuous segmentation stream subsystem for vlm_service.py.

Owns the seg-stream worker thread, its stop event, the live rate/stats, the
optional F3 temporal tracker, and the F3 tracking helper. Extracted verbatim
from VLMService; method bodies are unchanged except that the shared hub state
they read (_latest(), detector, _seg_constraints, the overlay knobs, _cache_dets,
_stop_event) is reached through a back-reference to the owning VLMService
(``self._svc``). A back-ref is the accepted ceiling for this DI hub — the
subsystem state it *owns* (the stream thread + stats + tracker gate) lives here;
the shared model/frame/render state stays on the service.

Not to be confused with `perception/` (Vivian's vendored boundary) — this is OUR
code, named under `vlm/` to avoid colliding with the vlm_service.py entry module.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional


class SegmentStream:
    """Continuous segmentation stream (toggle from the panel). When on,
    _segment_stream_loop calls detector.detect at seg_stream_hz and writes into
    the service's _cached_dets so the overlay stays fresh without manual
    "Segment Now" clicks.

    Holds a back-reference to the VLMService (``_svc``) so it can run the
    detector and write the shared render cache; owns the stream thread, its
    stop event, the live rate, the stats snapshot, and the F3 tracker gate.
    """

    # Window size for seg-stream stats accumulation. 5 s is short enough
    # to feel live in the panel readout but long enough that a single slow
    # inference doesn't dominate the average.
    _SEG_STREAM_STATS_S: float = 5.0

    def __init__(self, svc) -> None:
        self._svc = svc
        # F3 gate: temporal tracking/smoothing of the seg-stream. Read off the
        # parsed args the same way VLMService.__init__ did (stub-args path stays
        # on today's behaviour: tracking off).
        self._seg_track = bool(getattr(svc.args, "seg_track", False))
        self._seg_stream_thread: Optional[threading.Thread] = None
        self._seg_stream_stop = threading.Event()
        self._seg_stream_hz: float = 10.0
        # Telemetry for the seg-stream loop. Updated under no lock from the
        # loop thread and read read-only from _cmd_status; consumers tolerate
        # a slightly stale snapshot. Each window covers _SEG_STREAM_STATS_S
        # seconds; periodic stats lines log to stdout, the latest snapshot is
        # included in status replies so the panel can surface it.
        self._seg_stream_stats: Dict[str, Any] = {
            "active": False,
            "hz_target": 0.0,
            "hz_achieved": 0.0,
            "mean_dets": 0.0,
            "mean_infer_ms": 0.0,
            "errors": 0,
            "window_s": 0.0,
            "last_emit_t": 0.0,
        }

    def cmd_segment_stream(self, req: dict) -> dict:
        enabled = bool(req.get("enabled", False))
        hz = float(req.get("hz", self._seg_stream_hz))
        if hz <= 0.0:
            return {"ok": False, "error": "hz must be > 0"}
        if enabled:
            self._start_segment_stream(hz)
        else:
            self._stop_segment_stream()
        return {"ok": True, "enabled": enabled, "hz": hz}

    def _start_segment_stream(self, hz: float) -> None:
        # Idempotent: if already running at a different rate, swap the rate
        # without restarting the thread.
        self._seg_stream_hz = hz
        if self._seg_stream_thread is not None and self._seg_stream_thread.is_alive():
            return
        self._seg_stream_stop.clear()
        self._seg_stream_thread = threading.Thread(
            target=self._segment_stream_loop, daemon=True, name="vlm-seg-stream",
        )
        self._seg_stream_thread.start()
        _log(f"segment stream started at {hz:.1f} Hz")

    def stop(self) -> None:
        self._stop_segment_stream()

    def _stop_segment_stream(self) -> None:
        self._seg_stream_stop.set()
        t = self._seg_stream_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._seg_stream_thread = None
        # Clear cached detections so the overlay doesn't keep drawing the last
        # mask set after the stream is turned off.
        self._svc._cache_dets([])
        _log("segment stream stopped")

    def _segment_stream_loop(self) -> None:
        next_run = time.perf_counter()
        last_hz = self._seg_stream_hz
        period = 1.0 / max(last_hz, 1e-6)
        # WS4 F3: optional temporal tracker. Fresh per stream session so a
        # restart doesn't inherit stale tracks. Read-only import of the Tier-1
        # gaze tracker (pure numpy, no I/O — do NOT modify it). None when the
        # --seg-track gate is off, in which case the stateless path below is
        # unchanged.
        tracker = None
        if self._seg_track:
            from Utils.gaze.gaze_tracking import SimpleSORTTracker
            tracker = SimpleSORTTracker()
        # Stats accumulators for the current window. Reset after each
        # periodic emit so each line summarises fresh activity.
        win_start = time.perf_counter()
        win_ticks = 0
        win_dets = 0
        win_infer_s = 0.0
        win_errors = 0
        self._seg_stream_stats.update({
            "active": True, "hz_target": last_hz, "last_emit_t": time.time(),
        })
        try:
            while not self._seg_stream_stop.is_set() and not self._svc._stop_event.is_set():
                # Re-read the rate each iteration so live changes take effect.
                if self._seg_stream_hz != last_hz:
                    last_hz = self._seg_stream_hz
                    period = 1.0 / max(last_hz, 1e-6)
                    self._seg_stream_stats["hz_target"] = last_hz

                now_pc = time.perf_counter()
                if now_pc < next_run:
                    time.sleep(min(0.005, next_run - now_pc))
                    continue

                bundle, _, _ = self._svc._latest()
                if bundle is None:
                    next_run = now_pc + period
                    time.sleep(0.020)
                    continue

                try:
                    t0 = time.perf_counter()
                    dets = self._svc.detector.detect(bundle.video.bgr)
                    # F1 constraints first (geometry + gaze-ROI only — per-tick
                    # Depth Pro is too costly for the stream, so depth-band is
                    # decide-only; see SegConstraints.depth_band). Then the
                    # overlay top-K / containment reduction with the configured
                    # (E2) knobs.
                    dets = _apply_seg_constraints(
                        dets, bundle.video.bgr.shape, self._svc._seg_constraints,
                        gaze_xy=(float(bundle.gaze.x), float(bundle.gaze.y)),
                    )
                    dets = _filter_overlay_dets(
                        dets, top_k=self._svc._overlay_top_k,
                        contain_ratio=self._svc._overlay_contain_ratio,
                        area_ratio=self._svc._overlay_area_ratio,
                    )
                    # F3: temporal tracking/smoothing (opt-in). Replaces the
                    # raw stateless set with confirmed tracks (min_hits to
                    # appear, max_age to disappear) carrying stable track_ids.
                    if tracker is not None:
                        dets = self._apply_seg_tracking(dets, tracker, time.monotonic())
                    self._svc._cache_dets(dets)
                    win_ticks += 1
                    win_dets += len(dets)
                    win_infer_s += (time.perf_counter() - t0)
                except Exception as e:
                    # Errors always log — silent failure here used to hide
                    # detector misconfiguration (wrong device, missing
                    # weights) until the operator wondered why the overlay
                    # was empty.
                    win_errors += 1
                    _log(f"segment stream error: {e}")

                # Periodic stats emit. Logs one line and refreshes the
                # status-reply snapshot so the panel readout stays current.
                window_s = time.perf_counter() - win_start
                if window_s >= self._SEG_STREAM_STATS_S:
                    achieved = win_ticks / max(window_s, 1e-6)
                    mean_dets = (win_dets / win_ticks) if win_ticks else 0.0
                    mean_infer_ms = (win_infer_s / win_ticks * 1000.0) if win_ticks else 0.0
                    _log(
                        f"seg-stream: target={last_hz:.1f}Hz achieved={achieved:.1f}Hz "
                        f"ticks={win_ticks} mean_dets={mean_dets:.1f} "
                        f"mean_infer={mean_infer_ms:.0f}ms errors={win_errors}"
                    )
                    self._seg_stream_stats.update({
                        "active": True,
                        "hz_target": last_hz,
                        "hz_achieved": achieved,
                        "mean_dets": mean_dets,
                        "mean_infer_ms": mean_infer_ms,
                        "errors": win_errors,
                        "window_s": window_s,
                        "last_emit_t": time.time(),
                    })
                    win_start = time.perf_counter()
                    win_ticks = 0
                    win_dets = 0
                    win_infer_s = 0.0
                    win_errors = 0

                # Schedule next tick relative to the original cadence, but if we're
                # falling behind by more than 2 periods just resync — avoids a
                # runaway catch-up burst after a slow inference.
                next_run += period
                now_pc2 = time.perf_counter()
                if next_run < now_pc2 - 2.0 * period:
                    next_run = now_pc2
        finally:
            # Mark the loop dead so a stale snapshot doesn't make the panel
            # claim the stream is still healthy after a stop.
            self._seg_stream_stats["active"] = False

    def _apply_seg_tracking(self, dets, tracker, t_now: float):
        """F3: run the SORT tracker over this tick's detections and return the
        confirmed-track view.

        FastSAM labels (``segment_N``) are index-based and **not** stable across
        frames, so feeding them as the tracker's class would break IoU
        association (SimpleSORTTracker only matches within the same class). We
        sidestep that *without editing the Tier-1 tracker* by giving every
        detection the same class (0) — the class-consistency check then always
        passes and association is pure IoU/geometry. The track_id becomes the
        stable identity.

        Output preserves masks for objects present this frame (matched to a
        confirmed track by IoU) and emits box-only detections for confirmed
        tracks that are *coasting* (no detection this frame but still within
        max_age) — that pairing is what turns frame-to-frame flicker into
        min_hits-to-appear / max_age-to-disappear behaviour. Every returned
        Detection carries a ``track_id`` attribute and a ``#<id>`` label.
        """
        from perception.object_detector import Detection
        from Utils.gaze.gaze_tracking import iou_xyxy

        tracker.predict(t_now)
        tracker.update_with_dets(
            [{"xyxy": tuple(d.box_xyxy), "cls": 0, "conf": float(d.confidence),
              "name": str(d.label)} for d in dets],
            t_now=t_now,
        )
        tracks = tracker.get_tracks_as_dets(t_now)  # confirmed, unexpired

        # Greedy 1:1 match each confirmed track to the best-IoU current det.
        det_for_track: dict = {}
        claimed: set = set()
        for tr in tracks:
            best_d = None
            best_iou = 0.10  # floor: below this it isn't really the same object
            for k, d in enumerate(dets):
                if k in claimed:
                    continue
                score = iou_xyxy(tr["xyxy"], tuple(d.box_xyxy))
                if score >= best_iou:
                    best_iou = score
                    best_d = (k, d)
            if best_d is not None:
                claimed.add(best_d[0])
                det_for_track[tr["track_id"]] = best_d[1]

        out = []
        for tr in tracks:
            tid = tr["track_id"]
            d = det_for_track.get(tid)
            if d is not None:
                # Present this frame — keep its mask, tag with the stable id.
                d.track_id = tid
                d.label = f"#{tid}"
                out.append(d)
            else:
                # Coasting (Kalman-predicted, no det this frame): emit a
                # box-only detection so it persists through max_age.
                x1, y1, x2, y2 = tr["xyxy"]
                cd = Detection(
                    label=f"#{tid}",
                    confidence=float(tr.get("conf", 0.0)),
                    box_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    box_center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    mask_polygon=None,
                )
                cd.track_id = tid
                out.append(cd)
        return out


# Seg constraint / overlay-filter helpers live in vlm/seg_ops.py (the Phase-0
# leaf modules); re-imported here so _segment_stream_loop references them as bare
# names exactly as it did on VLMService.
from vlm.seg_ops import _apply_seg_constraints, _filter_overlay_dets


# Logging goes through vlm_service._log so service + collaborator lines tee to
# the same anonymous service log. Imported lazily inside the wrapper to avoid an
# import cycle at module load (vlm_service imports this module).
def _log(msg: str) -> None:
    import vlm_service
    vlm_service._log(msg)
