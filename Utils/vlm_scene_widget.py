"""
vlm_scene_widget.py — Qt widget that paints the Linux-side composited
scene-plus-overlay for the operator panel's VLM Video tab.

Replaces the legacy JPEG-over-TCP consumer (VLMVideoThread → QPixmap):
instead of receiving rendered overlay JPEGs from Windows, the widget
pulls bundles from the local frame_relay, pulls JSON detection state
from the Windows-hosted vlm_service.py over UDP 5589 subscribe, and
composites the two locally via Utils.scene_overlay_renderer.
SoftwareDocs/GPU_Service_Render_Layer_Refactor.md §4.3 / §4.4.

Threads owned by this widget:

  - One bundle-source thread (either RemoteFrameReader.__iter__ or an
    add_local_subscriber callback when a FrameRelayServer instance is
    supplied for in-process fan-out).
  - One UDP push subscriber thread per JSON channel (vlm_results,
    optional gaze_results).
  - QTimer on the main Qt thread for the paint pass.

The bundle source updates ``self._latest_frame_bgr`` (an owned copy)
plus the gaze scalars under ``self._bundle_lock``. The push subscribers emit Qt Signals; their slots
update ``self._latest_vlm`` / ``self._latest_gaze`` on the main thread.
The QTimer composites all three via SceneOverlayRenderer and paints.

Latency instrumentation (§7 acceptance #1): every received bundle is
stamped with ``time.monotonic()`` on arrival; every paint records
``time.monotonic()`` after the QPixmap swap. The widget exposes a
rolling deque of (arrival → painted) deltas via
``recent_paint_latency_ms()`` so the operator panel — or a one-shot
test harness — can read the headline metric.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from Utils.scene_overlay_renderer import SceneOverlayRenderer
from Utils.vlm_subscriber import JsonPushSubscriber


# ── bundle-source thread (RemoteFrameReader path) ──────────────────────────


class _BundleSourceThread(threading.Thread):
    """Iterates a RemoteFrameReader and forwards each bundle to a callback
    on the realtime path (the iterator handles connect / reconnect; we
    just forward). Used when the panel is not hosting a FrameRelayServer
    in-process."""

    def __init__(self, host: str, port: int, on_bundle, *, name: str = "panel-bundle-rx") -> None:
        super().__init__(daemon=True, name=name)
        self._host = host
        self._port = int(port)
        self._on_bundle = on_bundle
        self._stop = threading.Event()
        self._reader = None

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._reader is not None:
                self._reader.close()
        except Exception:
            pass

    def run(self) -> None:
        # Imported lazily so the panel still loads if Utils.remote_frame_reader
        # changes its imports. cv2 / numpy are dragged in there.
        from Utils.remote_frame_reader import RemoteFrameReader
        try:
            self._reader = RemoteFrameReader(self._host, self._port)
        except Exception:
            return
        try:
            for bundle in self._reader:
                if self._stop.is_set():
                    break
                try:
                    self._on_bundle(bundle)
                except Exception:
                    # Don't let a flaky callback take the rx loop down.
                    continue
        finally:
            try:
                self._reader.close()
            except Exception:
                pass


# ── main widget ────────────────────────────────────────────────────────────


class VLMSceneWidget(QWidget):
    """Composited scene + overlay tab. See module docstring for context.

    Bundle source priority (first that matches wins on ``start()``):
      1. ``embedded_relay`` dict — the widget HOSTS a FrameRelayServer
         in-process. Bundles flow SDK → add_local_subscriber callback,
         no JPEG encode/decode, no TCP. This is the path that matches
         neon_viewer.py quality. Windows TCP clients still connect to
         the same listening socket.
      2. ``local_relay_server`` — an externally-supplied FrameRelayServer
         instance (e.g. for tests). Same in-process fan-out hook.
      3. RemoteFrameReader on ``relay_dial_host:relay_dial_port`` —
         legacy path, JPEG round-trip on every frame; only useful when
         the relay genuinely lives on another machine or in another
         process this panel doesn't own.
    """

    PAINT_HZ = 30.0
    LATENCY_WINDOW = 240  # ~8 s of paint deltas at 30 Hz

    # Bubbled-up state from whichever JsonPushSubscriber is active
    # (vlm or gaze). The panel connects this to its Receive-LED slot.
    # Forwarded payloads: "subscribed" / "unsubscribed" / "error: ...".
    subscriber_state_changed = Signal(str)
    # Fires once per embedded-relay lifetime, on the pump thread the
    # first time a frame is broadcast to a TCP consumer. The panel
    # uses this to flip the Send LED green at the moment of the event
    # rather than waiting for the next 2 s relay-status poll. Payload
    # is the (host, port) tuple of the consumer that received the
    # first frame. See FrameRelayServer.__init__ on_first_publish.
    first_publish_observed = Signal(tuple)

    def __init__(
        self,
        *,
        vlm_host: str,
        vlm_port: int,
        gaze_host: Optional[str],
        gaze_port: Optional[int],
        relay_dial_host: str,
        relay_dial_port: int,
        local_relay_server=None,
        embedded_relay: Optional[Dict[str, Any]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._vlm_host = vlm_host
        self._vlm_port = vlm_port
        self._gaze_host = gaze_host
        self._gaze_port = gaze_port
        self._relay_dial_host = relay_dial_host
        self._relay_dial_port = relay_dial_port
        self._local_relay_server = local_relay_server
        # Config dict for an embedded relay; keys mirror FrameRelayServer
        # constructor: bind_host, bind_port, hz, neon_host, jpeg_quality,
        # repo_dir. None = caller doesn't want the panel to host the relay.
        self._embedded_relay_cfg = dict(embedded_relay) if embedded_relay else None
        self._embedded_relay = None
        self._embedded_relay_thread: Optional[threading.Thread] = None

        self._renderer = SceneOverlayRenderer()

        self._bundle_lock = threading.Lock()
        # Snapshotted bundle fields. NeonLiveReader.__iter__ reuses its
        # output buffer, so we must copy / scalar-extract on the pump
        # thread before the SDK pulls the next frame — otherwise the
        # paint pass reads a torn buffer (visible as grain/artifacts on
        # head motion). See vlm_service.py:649 for the same comment.
        self._latest_frame_bgr = None  # owned copy
        self._latest_gaze_xy: Optional[Tuple[float, float]] = None
        self._latest_frame_idx: int = 0
        self._latest_frame_ts_ns: int = 0
        self._latest_arrival_t: float = 0.0

        # Latest JSON push state — both keyed by message type.
        self._latest_vlm: Dict[str, Any] = {}
        self._latest_gaze: Dict[str, Any] = {}

        # Latency telemetry: bundle-arrival → painted deltas (ms).
        self._paint_latency_ms: Deque[float] = deque(maxlen=self.LATENCY_WINDOW)
        self._paint_count: int = 0
        self._fps_window_t: float = time.monotonic()
        self._fps: float = 0.0

        # Threads (created on start()).
        self._bundle_thread: Optional[_BundleSourceThread] = None
        self._vlm_subscriber: Optional[JsonPushSubscriber] = None
        self._gaze_subscriber: Optional[JsonPushSubscriber] = None

        # ── UI ──
        # Lifecycle (start/stop) is driven from the Main tab's VLM
        # Service row — see control_panel.py:_configure_remote_services_ui
        # for the remote-mode "Connect"/"Disconnect" rewire and
        # on_vlm_service_start/stop for the local-mode path. The
        # widget itself is a passive viewer: it shows the feed when
        # the pipeline is up and an explanation when it isn't.
        layout = QVBoxLayout(self)
        self.lbl_status = QLabel("Pipeline not started")
        self.lbl_status.setStyleSheet("color: #cccccc;")
        layout.addWidget(self.lbl_status)

        self.lbl_canvas = QLabel()
        self.lbl_canvas.setAlignment(Qt.AlignCenter)
        self.lbl_canvas.setMinimumSize(640, 360)
        self.lbl_canvas.setStyleSheet("background: #111111; color: #666666;")
        self.lbl_canvas.setText(
            "Pipeline not started.\n\n"
            "Use Connect on the Main tab's VLM Service row to open the "
            "Neon device, start the embedded frame_relay, and subscribe "
            "to vlm_service detections."
        )
        layout.addWidget(self.lbl_canvas, 1)

        self._paint_timer = QTimer(self)
        self._paint_timer.setInterval(int(1000.0 / self.PAINT_HZ))
        self._paint_timer.timeout.connect(self._on_paint_tick)

    # ── public API ────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._bundle_thread is not None or self._embedded_relay is not None:
            return  # already running

        # 1. Bundle source — prefer the in-process paths (embedded relay
        #    or externally-supplied local_relay_server) so we get raw
        #    SDK-decoded BGR with no JPEG round-trip. RemoteFrameReader
        #    is only the fallback.
        wired = False
        if self._embedded_relay_cfg is not None:
            if self._start_embedded_relay():
                wired = True
        if not wired and self._local_relay_server is not None:
            try:
                self._local_relay_server.add_local_subscriber(self._on_bundle_callback)
                wired = True
            except AttributeError:
                self._local_relay_server = None
        if not wired:
            self._bundle_thread = _BundleSourceThread(
                self._relay_dial_host,
                self._relay_dial_port,
                self._on_bundle_callback,
            )
            self._bundle_thread.start()

        # 2. VLM push subscriber (optional). The control panel can pass
        #    vlm_host=None to skip this — used when GAZE_OR_BACKEND is
        #    set to "legacy" so the panel only subscribes to gaze_runner.
        if self._vlm_host and self._vlm_port:
            self._vlm_subscriber = JsonPushSubscriber(
                self._vlm_host, self._vlm_port, hz=self.PAINT_HZ,
            )
            self._vlm_subscriber.payload_received.connect(self._on_vlm_payload)
            self._vlm_subscriber.state_changed.connect(self._on_vlm_state)
            # Forward state to the panel's Receive LED via the widget's
            # bubbled signal. Either subscriber's state is forwarded —
            # whichever backend (vlm or gaze) is active is what "Receive"
            # represents.
            self._vlm_subscriber.state_changed.connect(self.subscriber_state_changed)
            self._vlm_subscriber.start()

        # 3. Gaze push subscriber (optional).
        if self._gaze_host and self._gaze_port:
            self._gaze_subscriber = JsonPushSubscriber(
                self._gaze_host, int(self._gaze_port), hz=self.PAINT_HZ,
            )
            self._gaze_subscriber.payload_received.connect(self._on_gaze_payload)
            self._gaze_subscriber.state_changed.connect(self.subscriber_state_changed)
            self._gaze_subscriber.start()

        self._paint_timer.start()
        self.lbl_status.setText("Render path: json_local — started")

    def stop(self) -> None:
        self._paint_timer.stop()
        if self._local_relay_server is not None:
            try:
                self._local_relay_server.remove_local_subscriber(self._on_bundle_callback)
            except AttributeError:
                pass
        if self._embedded_relay is not None:
            try:
                self._embedded_relay.remove_local_subscriber(self._on_bundle_callback)
            except AttributeError:
                pass
            try:
                self._embedded_relay.stop()
            except Exception:
                pass
            if self._embedded_relay_thread is not None:
                self._embedded_relay_thread.join(timeout=3.0)
            self._embedded_relay = None
            self._embedded_relay_thread = None
        if self._bundle_thread is not None:
            self._bundle_thread.stop()
            self._bundle_thread = None
        for sub in (self._vlm_subscriber, self._gaze_subscriber):
            if sub is not None:
                sub.stop()
                sub.wait(2000)
        self._vlm_subscriber = None
        self._gaze_subscriber = None
        self.lbl_status.setText("Render path: json_local — stopped")

    # ── embedded-relay lifecycle ──────────────────────────────────────────

    def _start_embedded_relay(self) -> bool:
        """Construct + run a FrameRelayServer in-process. Returns True on
        success. Falls back to RemoteFrameReader if construction fails so
        the widget still tries to paint instead of going dark."""
        if self._embedded_relay_cfg is None:
            return False
        from Utils.frame_relay import FrameRelayServer
        from Utils.scene_only_neon_reader import SceneOnlyNeonReader
        cfg = self._embedded_relay_cfg
        try:
            # Pre-construct the SDK-direct reader (mirrors neon_viewer.py
            # exactly) and inject it into FrameRelayServer. Bypasses
            # NeonLiveReader's matched-API path which delivers visibly
            # grainier scene frames despite reading the same RTSP track.
            reader = SceneOnlyNeonReader(host=str(cfg.get("neon_host", "") or "") or None)
        except Exception as e:
            self.lbl_status.setText(f"Render path: json_local — neon reader open failed: {e}")
            self._embedded_relay = None
            return False
        try:
            self._embedded_relay = FrameRelayServer(
                bind_host=cfg.get("bind_host", "0.0.0.0"),
                bind_port=int(cfg.get("bind_port", 5591)),
                hz=float(cfg.get("hz", 30.0)),
                neon_host=str(cfg.get("neon_host", "") or ""),
                jpeg_quality=int(cfg.get("jpeg_quality", 75)),
                repo_dir=cfg.get("repo_dir"),
                reader=reader,
                on_first_publish=self._emit_first_publish,
            )
        except Exception as e:
            self.lbl_status.setText(f"Render path: json_local — embedded relay ctor failed: {e}")
            self._embedded_relay = None
            try:
                reader.close()
            except Exception:
                pass
            return False
        # Register the local subscriber BEFORE starting the pump so we
        # don't miss the first few bundles.
        self._embedded_relay.add_local_subscriber(self._on_bundle_callback)

        def _serve():
            try:
                self._embedded_relay.serve_forever()  # type: ignore[union-attr]
            except SystemExit as e:
                # _open_reader raises SystemExit("frame_relay: ...") if
                # harmony_vlm utils can't be imported. Surface that
                # message rather than crashing silently.
                msg = str(e)
                self.lbl_status.setText(
                    f"Render path: json_local — embedded relay aborted: {msg}"
                )
            except Exception as e:
                self.lbl_status.setText(
                    f"Render path: json_local — embedded relay error: {e}"
                )

        self._embedded_relay_thread = threading.Thread(
            target=_serve, daemon=True, name="panel-embedded-relay",
        )
        self._embedded_relay_thread.start()
        self.lbl_status.setText("Render path: json_local — embedded relay starting…")
        return True

    def _emit_first_publish(self, addr: Tuple[str, int]) -> None:
        """Bridge from FrameRelayServer's on_first_publish (pump thread)
        to the widget's Qt signal. Emit is thread-safe — Qt promotes it
        to a queued connection automatically when the receiver lives on
        a different thread."""
        try:
            self.first_publish_observed.emit(tuple(addr))
        except RuntimeError:
            # Widget already destroyed mid-relay-shutdown. Drop silently
            # — the panel slot can't run anyway.
            pass

    def recent_paint_latency_ms(self) -> Tuple[float, float, float]:
        """Rolling p50 / p95 / p99 of bundle-arrival → painted deltas in ms.

        Returns (NaN, NaN, NaN) until enough samples have landed. Values
        are in milliseconds. This is the headline metric for §7 #1."""
        if len(self._paint_latency_ms) < 4:
            nan = float("nan")
            return nan, nan, nan
        s = sorted(self._paint_latency_ms)
        n = len(s)

        def _q(q: float) -> float:
            return s[max(0, min(n - 1, int(round(q * (n - 1)))))]

        return _q(0.5), _q(0.95), _q(0.99)

    def current_fps(self) -> float:
        return self._fps

    # ── slots / callbacks ─────────────────────────────────────────────────

    def _on_bundle_callback(self, bundle) -> None:
        """Realtime path. Runs on the relay pump thread; do not block.

        Snapshot the BGR pixels and gaze scalars NOW because
        NeonLiveReader.__iter__ reuses its output buffer — by the next
        SDK iteration, ``bundle.video.bgr`` will be partially overwritten
        and any deferred read tears. The copy is unconditional; the
        renderer is then called with copy=False since we already own a
        clean ndarray.
        """
        try:
            frame = bundle.video.bgr
            frame_copy = frame.copy() if frame is not None else None
            gx = float(getattr(bundle.gaze, "x", float("nan")))
            gy = float(getattr(bundle.gaze, "y", float("nan")))
            ts_ns = int(getattr(bundle.video, "timestamp_ns", 0) or 0)
            idx = int(getattr(bundle.video, "frame_idx", 0) or 0)
        except AttributeError:
            return
        with self._bundle_lock:
            self._latest_frame_bgr = frame_copy
            self._latest_gaze_xy = (gx, gy)
            self._latest_frame_idx = idx
            self._latest_frame_ts_ns = ts_ns
            self._latest_arrival_t = time.monotonic()

    def _on_vlm_payload(self, payload: Dict[str, Any]) -> None:
        if payload.get("type") == "vlm_results":
            self._latest_vlm = payload

    def _on_vlm_state(self, state: str) -> None:
        # Append to the status badge but keep current path label.
        self.lbl_status.setText(f"Render path: json_local — vlm: {state}")

    def _on_gaze_payload(self, payload: Dict[str, Any]) -> None:
        if payload.get("type") == "gaze_results":
            self._latest_gaze = payload

    # ── paint pass ────────────────────────────────────────────────────────

    def _on_paint_tick(self) -> None:
        with self._bundle_lock:
            frame_bgr = self._latest_frame_bgr
            gaze_xy = self._latest_gaze_xy
            arrival_t = self._latest_arrival_t
        if frame_bgr is None or gaze_xy is None:
            return

        det_payload = self._latest_vlm or {}
        gaze_payload = self._latest_gaze or {}
        # frame_bgr is already an owned copy from _on_bundle_callback;
        # tell the renderer to draw in-place to skip a redundant copy.
        canvas = self._renderer.render(
            frame_bgr,
            copy=False,
            gaze_xy=gaze_xy,
            detections=det_payload.get("detections"),
            hit_det_id=(det_payload.get("hit") or {}).get("det_id")
                if det_payload.get("hit") else None,
            fixation=det_payload.get("fixation"),
            tracks=gaze_payload.get("tracks"),
            current_hit_track_id=(gaze_payload.get("current_hit") or {}).get("track_id")
                if gaze_payload.get("current_hit") else None,
            vlm_state=str(det_payload.get("vlm_state", "IDLE")),
            decision_text=(det_payload.get("decision") or {}).get("text"),
        )
        self._paint_canvas(canvas)

        painted_t = time.monotonic()
        if arrival_t > 0.0:
            self._paint_latency_ms.append((painted_t - arrival_t) * 1000.0)
        self._paint_count += 1
        if (painted_t - self._fps_window_t) >= 1.0:
            self._fps = self._paint_count / (painted_t - self._fps_window_t)
            self._paint_count = 0
            self._fps_window_t = painted_t
            p50, p95, p99 = self.recent_paint_latency_ms()
            if p50 == p50:  # NaN check
                self.lbl_status.setText(
                    f"Render path: json_local — {self._fps:5.1f} fps  "
                    f"paint p50/p95/p99 = {p50:5.1f}/{p95:5.1f}/{p99:5.1f} ms"
                )

    def _paint_canvas(self, canvas_bgr: np.ndarray) -> None:
        """Paint a BGR canvas into the QLabel.

        Uses ``Format_BGR888`` so we hand Qt the SDK's native BGR ndarray
        without a per-frame swap copy (Format_RGB888 + ``[:, :, ::-1]``
        forced an extra contiguous allocation every paint and adds
        rounding error on the integer slice/copy).

        Honours the screen's device-pixel ratio: the QPixmap is rendered
        at the QLabel's logical size × DPR, then ``setDevicePixelRatio``
        tells Qt the buffer is already at native pixel resolution. On
        HiDPI displays this avoids the upscale-then-downscale blur that
        looks like grain at all times.

        Skips ``scaled()`` when the source already matches the label —
        ``SmoothTransformation`` is the most expensive step in this path
        and pointless when the bitmap is already the right size.
        """
        # ``data`` must outlive the QImage. Keep a reference on self
        # until the next paint replaces it.
        self._last_canvas = canvas_bgr  # numpy refcount keeps it alive
        h, w = canvas_bgr.shape[:2]
        qimg = QImage(
            canvas_bgr.data, w, h, 3 * w, QImage.Format_BGR888,
        )
        dpr = float(self.lbl_canvas.devicePixelRatioF()) if hasattr(
            self.lbl_canvas, "devicePixelRatioF") else 1.0
        target_w = int(self.lbl_canvas.width() * dpr)
        target_h = int(self.lbl_canvas.height() * dpr)
        if w == target_w and h == target_h:
            pix = QPixmap.fromImage(qimg)
        else:
            pix = QPixmap.fromImage(qimg).scaled(
                target_w, target_h,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        if dpr != 1.0:
            pix.setDevicePixelRatio(dpr)
        self.lbl_canvas.setPixmap(pix)
