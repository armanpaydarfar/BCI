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

The bundle source updates ``self._latest_bundle`` under
``self._bundle_lock``. The push subscribers emit Qt Signals; their slots
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

import json
import socket
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from Utils.scene_overlay_renderer import SceneOverlayRenderer


# ── UDP push subscriber thread ─────────────────────────────────────────────


class _JsonPushSubscriber(QThread):
    """Subscribe-mode UDP listener. Sends ``cmd=subscribe`` to the configured
    service, then rx-loops on the same socket for pushed JSON datagrams.

    Emits one Signal per received payload. Re-subscribes on a heartbeat
    timer so the service-side TTL can prune dead clients without the
    widget having to track its own watchdog.
    """

    payload_received = Signal(dict)
    state_changed = Signal(str)  # "subscribed" / "unsubscribed" / "error: ..."

    HEARTBEAT_S = 10.0  # well below vlm_service's 30 s TTL.

    def __init__(self, host: str, port: int, *, hz: float = 20.0,
                 ttl_s: float = 30.0, parent=None) -> None:
        super().__init__(parent)
        self._host = str(host)
        self._port = int(port)
        self._hz = float(hz)
        self._ttl_s = float(ttl_s)
        self._sock: Optional[socket.socket] = None
        self._running = False
        self._subscriber_id: Optional[str] = None
        self._last_subscribe_t: float = 0.0

    def stop(self) -> None:
        self._running = False
        # Unblock the recvfrom by closing the socket; the run loop
        # tolerates the resulting OSError and exits.
        try:
            if self._sock is not None:
                self._sock.close()
        except OSError:
            pass

    # ── thread body ───────────────────────────────────────────────────────

    def run(self) -> None:
        self._running = True
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.bind(("", 0))  # ephemeral port; service replies here.
            self._sock.settimeout(0.5)
        except OSError as e:
            self.state_changed.emit(f"error: bind: {e}")
            return

        if not self._subscribe():
            self.state_changed.emit("error: initial subscribe failed")
        else:
            self.state_changed.emit("subscribed")

        while self._running:
            self._maybe_heartbeat()
            try:
                data, _addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                continue
            try:
                payload = json.loads(data.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            # Subscribe replies and push payloads share this socket. Only
            # forward push payloads (they carry a `type` field; the
            # subscribe reply has `ok`/`subscriber_id`).
            if "type" in payload:
                self.payload_received.emit(payload)

        # Best-effort unsubscribe so the service prunes immediately.
        if self._subscriber_id:
            try:
                self._send({"cmd": "unsubscribe",
                            "subscriber_id": self._subscriber_id})
            except OSError:
                pass
        try:
            if self._sock is not None:
                self._sock.close()
        except OSError:
            pass
        self.state_changed.emit("unsubscribed")

    # ── helpers ───────────────────────────────────────────────────────────

    def _send(self, payload: Dict[str, Any]) -> None:
        if self._sock is None:
            return
        self._sock.sendto(
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            (self._host, self._port),
        )

    def _subscribe(self) -> bool:
        try:
            self._send({"cmd": "subscribe", "stream": "results",
                        "hz": self._hz, "ttl_s": self._ttl_s})
            assert self._sock is not None
            self._sock.settimeout(1.5)
            data, _addr = self._sock.recvfrom(65535)
            self._sock.settimeout(0.5)
        except (OSError, AssertionError):
            return False
        try:
            resp = json.loads(data.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return False
        if not resp.get("ok"):
            return False
        self._subscriber_id = resp.get("subscriber_id")
        self._last_subscribe_t = time.monotonic()
        return True

    def _maybe_heartbeat(self) -> None:
        if time.monotonic() - self._last_subscribe_t < self.HEARTBEAT_S:
            return
        self._subscribe()  # idempotent on (addr, port) — re-uses same id


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

    Pass ``local_relay_server`` (a :class:`Utils.frame_relay.FrameRelayServer`
    instance) to enable in-process fan-out (§4.2). Otherwise the widget
    dials the configured frame_relay over TCP — works whether the relay
    runs on this machine (loopback) or another.
    """

    PAINT_HZ = 30.0
    LATENCY_WINDOW = 240  # ~8 s of paint deltas at 30 Hz

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

        self._renderer = SceneOverlayRenderer()

        self._bundle_lock = threading.Lock()
        # Latest bundle + arrival monotonic timestamp.
        self._latest_bundle = None
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
        self._vlm_subscriber: Optional[_JsonPushSubscriber] = None
        self._gaze_subscriber: Optional[_JsonPushSubscriber] = None

        # ── UI ──
        layout = QVBoxLayout(self)
        ctrl = QHBoxLayout()
        self.lbl_status = QLabel("Render path: json_local — not started")
        self.lbl_status.setStyleSheet("color: #cccccc;")
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self.start)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop)
        ctrl.addWidget(self.lbl_status, 1)
        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)
        layout.addLayout(ctrl)

        self.lbl_canvas = QLabel()
        self.lbl_canvas.setAlignment(Qt.AlignCenter)
        self.lbl_canvas.setMinimumSize(640, 360)
        self.lbl_canvas.setStyleSheet("background: #111111; color: #666666;")
        self.lbl_canvas.setText(
            "Click Start to begin streaming.\n\n"
            "Bundles come from the local frame_relay; detections come from\n"
            "vlm_service.py via UDP 5589 subscribe."
        )
        layout.addWidget(self.lbl_canvas, 1)

        self._paint_timer = QTimer(self)
        self._paint_timer.setInterval(int(1000.0 / self.PAINT_HZ))
        self._paint_timer.timeout.connect(self._on_paint_tick)

    # ── public API ────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._bundle_thread is not None:
            return  # already running

        # 1. Bundle source.
        if self._local_relay_server is not None:
            try:
                self._local_relay_server.add_local_subscriber(self._on_bundle_callback)
            except AttributeError:
                # Not a FrameRelayServer; fall through to remote.
                self._local_relay_server = None
        if self._local_relay_server is None:
            self._bundle_thread = _BundleSourceThread(
                self._relay_dial_host,
                self._relay_dial_port,
                self._on_bundle_callback,
            )
            self._bundle_thread.start()

        # 2. VLM push subscriber.
        self._vlm_subscriber = _JsonPushSubscriber(
            self._vlm_host, self._vlm_port, hz=self.PAINT_HZ,
        )
        self._vlm_subscriber.payload_received.connect(self._on_vlm_payload)
        self._vlm_subscriber.state_changed.connect(self._on_vlm_state)
        self._vlm_subscriber.start()

        # 3. Gaze push subscriber (optional).
        if self._gaze_host and self._gaze_port:
            self._gaze_subscriber = _JsonPushSubscriber(
                self._gaze_host, int(self._gaze_port), hz=self.PAINT_HZ,
            )
            self._gaze_subscriber.payload_received.connect(self._on_gaze_payload)
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
        """Realtime path. Runs on the relay/reader thread; do not block."""
        with self._bundle_lock:
            self._latest_bundle = bundle
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
            bundle = self._latest_bundle
            arrival_t = self._latest_arrival_t
        if bundle is None:
            return
        try:
            frame_bgr = bundle.video.bgr
            gx = float(getattr(bundle.gaze, "x", float("nan")))
            gy = float(getattr(bundle.gaze, "y", float("nan")))
        except AttributeError:
            return
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
            return

        det_payload = self._latest_vlm or {}
        canvas = self._renderer.render(
            frame_bgr,
            gaze_xy=(gx, gy),
            detections=det_payload.get("detections"),
            hit_det_id=(det_payload.get("hit") or {}).get("det_id")
                if det_payload.get("hit") else None,
            fixation=det_payload.get("fixation"),
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
        # OpenCV BGR → QImage RGB888. ``data`` must outlive the QImage; we
        # keep a reference on self until the next paint tick.
        h, w = canvas_bgr.shape[:2]
        self._last_rgb = np.ascontiguousarray(canvas_bgr[:, :, ::-1])
        qimg = QImage(
            self._last_rgb.data, w, h, 3 * w, QImage.Format_RGB888,
        )
        pix = QPixmap.fromImage(qimg).scaled(
            self.lbl_canvas.width(),
            self.lbl_canvas.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.lbl_canvas.setPixmap(pix)
