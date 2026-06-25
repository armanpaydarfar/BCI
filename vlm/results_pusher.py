"""
vlm/results_pusher.py — subscribe-mode JSON push subsystem for vlm_service.py.

Owns the subscriber registry (dict + lock), the results-tick broadcast thread,
and the payload-building/wire-send logic that implements the render-on-Linux
JSON channel (Render_Layer_Refactor.md §3). Extracted verbatim from VLMService;
method bodies are unchanged except that the cached hub state they read
(_latest(), _render_lock, _cached_dets/_cached_hit_*, _vlm_state, _last_decision,
_first_snap_det, _stop_event) is reached through a back-reference to the owning
VLMService (``self._svc``). A back-ref is the accepted ceiling for this DI hub —
the subsystem state it *owns* (subscribers + tick thread) lives here; the shared
frame/render state stays on the service.

Not to be confused with `perception/` (Vivian's vendored boundary) — this is OUR
code, named under `vlm/` to avoid colliding with the vlm_service.py entry module.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from typing import Any, Dict, Optional

from vlm.wire import (
    _json_default,
    _serialize_detection_for_push,
    _serialize_decision_for_push,
)


class ResultsPusher:
    """Subscribe-mode push: panel/clients send ``cmd=subscribe`` and we
    broadcast ``vlm_results`` JSON datagrams from a tick thread. The
    render-on-Linux refactor (Render_Layer_Refactor.md §3) replaces the JPEG
    overlay round-trip with this JSON channel.

    Holds a back-reference to the VLMService (``_svc``) so it can read the
    shared frame/render state the payload is built from; owns the subscriber
    registry and the tick thread itself.
    """

    # Internal tick rate caps the per-subscriber rate. Subscribers may
    # request lower hz; higher requests are clamped at this ceiling.
    _RESULTS_TICK_HZ: float = 20.0
    # Subscribers expire after this long without a refresh. The panel is
    # expected to re-subscribe every ~10 s as a heartbeat — drops dead
    # subscribers automatically when a client crashes.
    _SUBSCRIBER_TTL_S: float = 30.0

    def __init__(self, svc) -> None:
        self._svc = svc
        # subscriber_id (uuid hex) → {addr, hz, last_sent_t, expires_at}
        self._subscribers_lock = threading.Lock()
        self._subscribers: Dict[str, Dict[str, Any]] = {}
        self._results_tick_thread: Optional[threading.Thread] = None
        self._results_tick_stop = threading.Event()

    def start(self) -> None:
        """Spawn the tick thread that broadcasts vlm_results JSON to subscribers."""
        if self._results_tick_thread is not None and self._results_tick_thread.is_alive():
            return
        self._results_tick_stop.clear()
        self._results_tick_thread = threading.Thread(
            target=self._results_tick_loop, daemon=True, name="vlm-results-push",
        )
        self._results_tick_thread.start()
        _log("results push thread started")

    def stop(self) -> None:
        self._results_tick_stop.set()

    def cmd_subscribe(self, req: dict, addr: tuple) -> dict:
        """Add (or refresh) a subscriber. Idempotent on (addr, port) so a
        client re-subscribing as a heartbeat doesn't accumulate ghosts."""
        try:
            hz = float(req.get("hz", self._RESULTS_TICK_HZ))
        except (TypeError, ValueError):
            return {"ok": False, "error": "bad_hz"}
        hz = max(0.5, min(hz, self._RESULTS_TICK_HZ))
        ttl_s = float(req.get("ttl_s", self._SUBSCRIBER_TTL_S))
        now = time.monotonic()
        with self._subscribers_lock:
            # Reuse existing id when the same (addr, port) re-subscribes;
            # this keeps the subscriber set stable across heartbeats and
            # lets the client treat subscribe as idempotent.
            existing_id: Optional[str] = None
            for sid, info in self._subscribers.items():
                if info.get("addr") == addr:
                    existing_id = sid
                    break
            sid = existing_id or uuid.uuid4().hex[:12]
            self._subscribers[sid] = {
                "addr": tuple(addr),
                "hz": hz,
                "last_sent_t": 0.0,
                "expires_at": now + ttl_s,
            }
        return {"ok": True, "stream": "results", "subscriber_id": sid, "hz": hz}

    def cmd_unsubscribe(self, req: dict) -> dict:
        sid = req.get("subscriber_id")
        if not sid:
            return {"ok": False, "error": "missing_subscriber_id"}
        with self._subscribers_lock:
            removed = self._subscribers.pop(str(sid), None)
        return {"ok": True, "removed": bool(removed)}

    def cmd_verify_chain(self, req: dict) -> dict:
        """End-to-end chain verification, used by the operator panel
        immediately after Connect to flip the Receive LED green
        without firing a real segment that would leave detections on
        the panel's overlay.

        Confirms (a) we have a frame from the upstream relay (proves
        Send + ingest), (b) the detector is loaded (proves the GPU
        side can compute), and (c) we can push to subscribers (proves
        the return path). Pushes one synthetic payload with
        ``type="chain_verify"`` to all current subscribers. The panel's
        ``_on_vlm_payload`` only paints overlays for ``type="vlm_results"``
        (Utils/vlm_scene_widget.py:413-415), so this push lights
        Receive without any visible artifact on the video tab.

        Echoes any ``token`` field from the request into the pushed
        payload so the panel can match the response to the specific
        Connect cycle that issued it. Without the token a stale push
        from a prior session's GPU cache would be indistinguishable
        from a fresh one and could trip the Receive LED prematurely.

        No state is mutated — ``_cached_dets`` is left untouched, the
        VLM state machine is not bumped, and the regular results-tick
        loop is unaffected.
        """
        bundle, _, _ = self._svc._latest()
        if bundle is None:
            return {"ok": False, "error": "no_frame"}
        if self._svc.detector is None:
            return {"ok": False, "error": "detector_not_loaded"}

        payload_dict = {
            "type": "chain_verify",
            "ok": True,
            "ts_send_ns": int(time.time_ns()),
        }
        token = req.get("token") if isinstance(req, dict) else None
        if token is not None:
            payload_dict["token"] = token
        payload = json.dumps(payload_dict).encode("utf-8")

        with self._subscribers_lock:
            addrs = [info["addr"] for info in self._subscribers.values()]
        sent = 0
        if addrs:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                for sub_addr in addrs:
                    try:
                        sock.sendto(payload, sub_addr)
                        sent += 1
                    except OSError:
                        pass
            finally:
                try:
                    sock.close()
                except OSError:
                    pass
        return {"ok": True, "subscribers_notified": sent}

    def _results_tick_loop(self) -> None:
        """Build one payload per tick, send to each due subscriber. Reads
        cached state under the existing _render_lock — no model calls in
        this thread."""
        push_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        period = 1.0 / max(self._RESULTS_TICK_HZ, 1e-6)
        # Periodic stats — emitted every _RESULTS_LOG_PERIOD_S so the
        # operator can see whether subscribers exist + whether sends are
        # actually happening. Counts reset after each emit.
        stats_period_s = 30.0
        last_emit = time.monotonic()
        ticks_run = 0
        sends_done = 0
        try:
            while not self._svc._stop_event.is_set() and not self._results_tick_stop.is_set():
                t0 = time.monotonic()
                sent_this_tick = self._tick_send_results(push_sock, t0)
                ticks_run += 1
                sends_done += sent_this_tick
                if (t0 - last_emit) >= stats_period_s:
                    with self._subscribers_lock:
                        n_subs = len(self._subscribers)
                    _log(
                        f"results push: subs={n_subs} ticks={ticks_run} "
                        f"sends={sends_done} window={t0 - last_emit:.0f}s"
                    )
                    last_emit = t0
                    ticks_run = 0
                    sends_done = 0
                slept = time.monotonic() - t0
                remaining = period - slept
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            try:
                push_sock.close()
            except OSError:
                pass

    def _tick_send_results(self, push_sock: socket.socket, now: float) -> int:
        """One pass: prune expired, build payload if any subscriber is due,
        send. Returns the number of datagrams successfully transmitted on
        this tick so _results_tick_loop can roll them up into the
        periodic stats line."""
        # Drop expired subscribers up front so we don't serialise for nobody.
        with self._subscribers_lock:
            for sid, info in list(self._subscribers.items()):
                if info["expires_at"] < now:
                    self._subscribers.pop(sid, None)
            due = []
            for sid, info in self._subscribers.items():
                period = 1.0 / max(info["hz"], 1e-6)
                if (now - info["last_sent_t"]) >= period:
                    due.append((sid, info["addr"]))
                    info["last_sent_t"] = now
        if not due:
            return 0
        try:
            payload_dict = self._build_vlm_results_payload()
        except Exception as e:
            # Always log payload build failures — silent failure here used
            # to mask broken cached_dets contents and the panel just kept
            # reporting "intake: connected" with no detections drawn.
            _log(f"results push: payload build failed: {e}")
            return 0
        if payload_dict is None:
            # No frame received from the relay yet. Pushing an empty
            # placeholder (frame_idx=0, detections=[], …) would light
            # the panel's Receive LED green before any real data has
            # flowed end-to-end, breaking the chain-of-causation
            # semantic the operator panel relies on. Skip until we
            # have a real bundle.
            return 0
        payload = json.dumps(payload_dict, default=_json_default).encode("utf-8")
        if len(payload) > 60 * 1024:
            # UDP datagrams >~64 KB get IP-fragmented. Keep payloads small;
            # the panel will fall back to the next tick.
            _log(f"results push: payload {len(payload)} B exceeds 60 KB; skipping tick")
            return 0
        sent = 0
        for _sid, addr in due:
            try:
                push_sock.sendto(payload, addr)
                sent += 1
            except OSError:
                pass
        return sent

    def _build_vlm_results_payload(self) -> Optional[dict]:
        """Snapshot the current detection / hit / fixation / decision state
        as the JSON push payload defined in Render_Layer_Refactor.md §3.

        Returns ``None`` when no frame has been received yet from the
        upstream relay (``bundle is None``). The tick-send loop skips
        the broadcast in that case so subscribers never see a placeholder
        payload before real data is flowing — the panel's "Receive" LED
        is gated on actual content under the chain-of-causation semantic.
        """
        bundle, fix, _ = self._svc._latest()
        if bundle is None:
            return None
        with self._svc._render_lock:
            dets = list(self._svc._cached_dets)
            hit_det = self._svc._cached_hit_det
            hit_wp = self._svc._cached_hit_wp
            state = self._svc._vlm_state
            decision = self._svc._last_decision
            first_det = self._svc._first_snap_det

        # Skip the push when nothing has happened yet on the GPU side
        # (no detections cached, no hit, no fixation, no decision, no
        # first-snap object, and we're still in IDLE). Without this,
        # the very first frame after Connect would trigger a fully-
        # default "vlm_results" datagram that lights the panel's
        # Receive LED before any real compute has run — breaking the
        # operator's chain-of-causation expectation. The chain_verify
        # command provides the affirmative verification path; idle
        # ticks stay silent.
        fix_active = fix is not None and getattr(fix, "active", False)
        if (not dets and hit_det is None and not fix_active
                and decision is None and first_det is None
                and state == "IDLE"):
            return None

        detections_out = [_serialize_detection_for_push(d) for d in dets]
        hit_payload = None
        if hit_det is not None:
            hit_id = next((i for i, d in enumerate(dets) if d is hit_det), -1)
            wp_pixel = (hit_wp or {}).get("pixel_center") if isinstance(hit_wp, dict) else None
            hit_payload = {
                "det_id": hit_id,
                "waypoint": list(wp_pixel) if wp_pixel is not None else None,
                "label": getattr(hit_det, "label", None),
            }

        fixation_payload = None
        if fix is not None and getattr(fix, "active", False):
            duration_ns = int(getattr(fix, "duration_ns", 0))
            fx = float(bundle.gaze.x) if bundle is not None else None
            fy = float(bundle.gaze.y) if bundle is not None else None
            fixation_payload = {
                "active": True,
                "duration_ms": duration_ns / 1_000_000.0,
                "x": fx,
                "y": fy,
                "stable": bool(getattr(fix, "is_stable", False)),
            }

        depth_at_gaze_m = None
        if isinstance(decision, dict):
            depth_at_gaze_m = decision.get("depth_at_gaze_m")

        first_object_payload = None
        if first_det is not None:
            first_object_payload = _serialize_detection_for_push(first_det)

        return {
            "type": "vlm_results",
            "frame_idx": int(getattr(getattr(bundle, "video", None), "frame_idx", 0)) if bundle is not None else 0,
            "frame_ts_ns": int(getattr(getattr(bundle, "video", None), "timestamp_ns", 0)) if bundle is not None else 0,
            "ts_send_ns": int(time.time_ns()),
            "detections": detections_out,
            "hit": hit_payload,
            "fixation": fixation_payload,
            "decision": _serialize_decision_for_push(decision) if isinstance(decision, dict) else None,
            "depth_at_gaze_m": depth_at_gaze_m,
            "vlm_state": state,
            "first_object": first_object_payload,
            "gaze_px": [float(bundle.gaze.x), float(bundle.gaze.y)] if bundle is not None else None,
        }


# Logging goes through vlm_service._log so service + collaborator lines tee to
# the same anonymous service log. Imported lazily inside the wrapper to avoid an
# import cycle at module load (vlm_service imports this module).
def _log(msg: str) -> None:
    import vlm_service
    vlm_service._log(msg)
