"""
panel/vlm_controller.py — VLM / perception-pipeline subsystem for the control panel.

The single largest collaborator extracted from ControlPanel: the whole VLM video
tab + frame-relay lifecycle, the per-Connect verify-chain state machine, the
remote-services / relay status pollers, and every VLM service command. All method
bodies were transcribed VERBATIM from ControlPanel — this is a behaviour-preserving
move. The only edits are the cross-cutting substitutions every collaborator makes
(self._append_log → self._log, self._append_log_ui → self._log_ui, the injected
set_led/timestamp callbacks, QMessageBox parent → self._parent) and repointing the
moved LEDs/widgets/state onto the controller (self.*).

Widget ownership mirrors GazeController / SerialController: the controller builds
the Perception-Pipeline row (3 LEDs + lifecycle/command buttons) into the panel's
main grid via build_pipeline_row_into(grid, row), and builds the VLM Video tab via
build_vlm_video_tab(tabs). Both transcribe the former inline _build_ui blocks
verbatim, so the layout is unchanged.

The verify-chain state machine is a QObject in its own right, so the
_remote_status_received Signal that marshals the off-thread UDP status reply back to
the GUI thread is declared HERE (class scope) and connected in __init__ — it no
longer has to live on the panel. The three periodic QTimers (_remote_status_timer /
_relay_status_timer / _seg_stream_log_timer) and all six verify-chain flags
(_connect_token, _send_observed, _compute_observed, _receive_observed,
_verify_chain_in_flight, _verify_chain_attempts) plus the inflight bookkeeping
(_vlm_inflight_lock/count, _vlm_last_snapshot_id, _last_seg_stream_emit_t,
_remote_status_in_flight) all move here too.

The QProcess lifecycle stays with the panel: self.procs (ProcessManager) and the
vlm_service Proc handle are iterated by the panel's cmd-wiring / subject-rotation /
closeEvent loops, so they're INJECTED rather than moved. The frame_relay_proc Proc
is only ever touched by the relay-subprocess methods here, so it moves into the
controller. log_files (LogFileController) is injected because the relay-log callback
was installed from the panel's __init__; the controller only reads its handles via
the panel's _append_log tee, so nothing about that wiring changes.

WARNING: the verify-chain transitions are NOT exercised by the headless tests (no
relay, no GPU service, no Neon). The construction guard proves the wiring resolves;
trusting the state machine REQUIRES a live Connect smoke at the rig.
"""

from __future__ import annotations

import json
import os
import sys
import time
import threading
from typing import Callable, Optional

from PySide6.QtCore import QObject, QProcess, QTimer, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QTabWidget, QVBoxLayout, QWidget,
)

from panel.process_manager import Proc, ProcessManager
from panel.ui_utils import _fixed_v
from panel.netutils import (
    _sleep_inhibit, _kill_orphan_vlm_service, _is_port_in_use,
)
from panel.constants import (
    ROOT, _HCFG, _IS_WINDOWS,
    GAZE_SERVICE_HOST, GAZE_SERVICE_PORT,
    VLM_SERVICE_PY, VLM_SERVICE_HOST, VLM_BIND_HOST, VLM_SERVICE_PORT,
    PERCEPTION_MODELS_DIR, VLM_MODEL, VLM_ENABLE_DEPTH, VLM_SESSION_ROOT,
    VLM_QUERY_TIMEOUT_S, VLM_DECIDE_TIMEOUT_S, GAZE_OR_BACKEND,
    NEON_COMPANION_HOST, SERVICES_HOSTED_REMOTELY, PERCEPTION_FRAME_SOURCE,
    FRAME_RELAY_DIAL_HOST, FRAME_RELAY_PORT, FRAME_RELAY_BIND_HOST,
    FRAME_RELAY_HZ, FRAME_RELAY_EMBEDDED,
)


class VlmController(QObject):
    """Owns the Perception-Pipeline row, the VLM Video tab, the verify-chain
    state machine, the status pollers, and the VLM service commands.

    Injected dependencies (behaviour-identical to the former in-class calls):
      procs              — the panel's ProcessManager (QProcess lifecycle)
      vlm_service        — the VLM Service Proc handle (stays on the panel,
                           iterated by the cmd/subject/closeEvent loops)
      log_files          — LogFileController (its handles are tee'd by the
                           panel's _append_log; relay callback installed there)
      log(title, text)   — append to the panel's log buffer
      log_ui(title, text)— same, marshaled onto the Qt main thread (worker safe)
      set_led(led, st)   — colour an LED for a state string
      timestamp()        — "HH:MM:SS" for log lines
    """

    # Emitted by the off-thread remote-status worker. Marshals the UDP status
    # reply (or a "down" sentinel) back to the GUI thread so the 0.4 s socket
    # timeout doesn't block paint. Declared on the controller (a QObject) so it
    # no longer has to live on the panel.
    _remote_status_received = Signal(dict)

    def __init__(
        self,
        parent,
        *,
        procs: ProcessManager,
        vlm_service: Proc,
        log_files,
        log: Callable[[str, str], None],
        log_ui: Callable[[str, str], None],
        set_led: Callable[[object, str], None],
        timestamp: Callable[[], str],
    ) -> None:
        super().__init__(parent)
        self._parent = parent
        self.procs = procs
        self.vlm_service = vlm_service
        self.log_files = log_files
        self._log = log
        self._log_ui = log_ui
        self._set_led = set_led
        self._ts = timestamp

        self._remote_status_in_flight = False
        # Per-Connect verification state machine. Each Connect generates
        # a fresh `_connect_token` (time.time_ns()); flags advance as
        # each phase completes:
        #   PHASE_SEND     — handshake from relay → _send_observed=True
        #   PHASE_COMPUTE  — cmd=status reply ok=True → _compute_observed=True
        #   PHASE_RECEIVE  — relay first publish + GPU has fresh bundle
        #                    → fire cmd=verify_chain with token
        #   DONE           — chain_verify push with matching token
        # Stale GPU-cache pushes from a prior session carry no token
        # (or an old one), so they cannot trip Receive on a reconnect.
        # Also tracks `_verify_chain_attempts` for the one-retry
        # policy on `no_frame` races.
        self._connect_token: Optional[int] = None
        self._send_observed: bool = False
        self._compute_observed: bool = False
        self._receive_observed: bool = False
        self._verify_chain_in_flight: bool = False
        self._verify_chain_attempts: int = 0
        self._remote_status_received.connect(self._apply_remote_status)

        self._vlm_last_snapshot_id: Optional[str] = None

        # ---- Frame relay proc (separated-decode mode) ----
        # When FRAME_RELAY_EMBEDDED is False the panel does NOT host the relay
        # in-process; it spawns Utils.frame_relay as a child process so the
        # Neon H.264 decode runs in its own GIL, isolated from the panel's
        # Qt paint + JPEG-encode load. In-process contention starves the SDK's
        # RTP-receive thread and corrupts the lower image slices ("tearing");
        # process isolation is the validated fix (rootcause record 2026-06-22).
        # cmd is built lazily on connect from the live config.
        self.frame_relay_proc = Proc("Frame Relay", None, ROOT)

        # Periodic remote-status poller (cheap UDP ping). 1 s cadence.
        self._remote_status_timer = QTimer(self)
        self._remote_status_timer.setInterval(1000)
        self._remote_status_timer.timeout.connect(self._poll_remote_status)
        if SERVICES_HOSTED_REMOTELY:
            self._remote_status_timer.start()

        # Frame relay status — surfaced in either mode. The relay is a
        # separate process from the perception services; running on Linux
        # in production but optionally on Windows for single-machine tests.
        self._relay_status_timer = QTimer(self)
        self._relay_status_timer.setInterval(2000)
        self._relay_status_timer.timeout.connect(self._poll_relay_status)
        if PERCEPTION_FRAME_SOURCE == "remote" or SERVICES_HOSTED_REMOTELY:
            self._relay_status_timer.start()

        # Seg-stream readout — only running while the operator has the
        # stream toggled on. Cadence matches the service's stats window
        # (_SEG_STREAM_STATS_S = 5s) so the same numbers refresh once per
        # log line. Emits to the VLM log buffer; a separate timer keeps
        # this off the 1 Hz remote-status path so latency on that hot
        # path stays unchanged.
        self._seg_stream_log_timer = QTimer(self)
        self._seg_stream_log_timer.setInterval(5000)
        self._seg_stream_log_timer.timeout.connect(self._poll_seg_stream_stats)
        # Suppress duplicate emissions when last_emit_t hasn't advanced.
        self._last_seg_stream_emit_t: float = 0.0
        # In-flight counter for synchronous VLM commands (decide, capture_first,
        # depth, ...). The seg-stream stats poll uses this to skip its tick
        # while a long command is occupying the server's single-threaded
        # request loop (vlm_service.py:403-437), since the resulting status
        # timeout would only mean "busy", not "unreachable".
        self._vlm_inflight_lock = threading.Lock()
        self._vlm_inflight_count: int = 0

    def build_pipeline_row_into(self, grid: QGridLayout, row: int) -> int:
        """Build the Perception-Pipeline row (3 LEDs + lifecycle/command
        buttons) into the panel's main grid starting at ``row``; return the
        next free row. Widget tree + grid placement are identical to the former
        inline _build_ui block."""
        self.lbl_send_led    = QLabel("●"); self._set_led(self.lbl_send_led,    "stopped")
        self.lbl_compute_led = QLabel("●"); self._set_led(self.lbl_compute_led, "stopped")
        self.lbl_receive_led = QLabel("●"); self._set_led(self.lbl_receive_led, "stopped")
        # Initial tooltip text — updated in place by the same drivers
        # (_apply_remote_status, _poll_relay_status, _on_subscriber_state)
        # that set the LED colour.
        self.lbl_send_led.setToolTip("send: idle")
        self.lbl_compute_led.setToolTip("compute: --")
        self.lbl_receive_led.setToolTip("receive: --")

        leds_box = QHBoxLayout()
        leds_box.setContentsMargins(0, 0, 0, 0)
        leds_box.setSpacing(2)
        for led in (self.lbl_send_led, self.lbl_compute_led, self.lbl_receive_led):
            leds_box.addWidget(led)
        leds_holder = _fixed_v(QWidget()); leds_holder.setLayout(leds_box)

        # Lifecycle + runtime actions. btn_vlm_service_start/stop are
        # constructed here but rebranded to "Connect"/"Disconnect" in
        # remote mode by _configure_remote_services_ui (which also
        # rewires their handlers to _on_vlm_video_connect/disconnect).
        self.btn_vlm_service_start  = QPushButton("Start")
        self.btn_vlm_service_stop   = QPushButton("Stop")
        self.btn_vlm_service_status = QPushButton("Status")
        self.btn_vlm_service_decide = QPushButton("Decide Now")
        self.btn_vlm_service_depth  = QPushButton("Depth Now")
        # WS4 F5: fast COCO recognition of the gaze object (no Gemini). Only
        # functional when the service was started with --recognizer-model; the
        # service returns recognizer_disabled otherwise (surfaced in the log).
        self.btn_vlm_service_recognize = QPushButton("Confirm Object")
        self.btn_vlm_service_recognize.setToolTip(
            "Fast YOLO/COCO naming of the gaze object, no VLM round-trip "
            "(needs the service started with --recognizer-model)."
        )
        self.btn_vlm_service_start.clicked.connect(self.on_vlm_service_start)
        self.btn_vlm_service_stop.clicked.connect(self.on_vlm_service_stop)
        self.btn_vlm_service_status.clicked.connect(self.on_vlm_service_status)
        self.btn_vlm_service_decide.clicked.connect(self.on_vlm_service_decide)
        self.btn_vlm_service_depth.clicked.connect(self.on_vlm_service_depth)
        self.btn_vlm_service_recognize.clicked.connect(self.on_vlm_service_recognize)

        # Continuous segmentation toggle + cadence — merged onto the
        # main perception row instead of a separate "Continuous / Pair"
        # row, since stream control belongs alongside lifecycle in the
        # operator's mental model.
        self.btn_vlm_seg_stream = QPushButton("Stream Seg: OFF")
        self.btn_vlm_seg_stream.setCheckable(True)
        self.spin_vlm_seg_hz = QDoubleSpinBox()
        self.spin_vlm_seg_hz.setRange(1.0, 30.0)
        self.spin_vlm_seg_hz.setSingleStep(1.0)
        self.spin_vlm_seg_hz.setDecimals(1)
        self.spin_vlm_seg_hz.setValue(10.0)
        self.spin_vlm_seg_hz.setSuffix(" Hz")
        self.btn_vlm_seg_stream.toggled.connect(self.on_vlm_seg_stream_toggled)
        self.spin_vlm_seg_hz.valueChanged.connect(self.on_vlm_seg_stream_hz_changed)

        # Sequential (two-object) decide controls.
        self.btn_vlm_capture_first = QPushButton("Capture First")
        self.btn_vlm_decide_pair = QPushButton("Decide Pair")
        self.lbl_vlm_pair_token = QLabel("<i>snapshot:</i> (none)")
        self.btn_vlm_capture_first.clicked.connect(self.on_vlm_capture_first)
        self.btn_vlm_decide_pair.clicked.connect(self.on_vlm_decide_pair)

        # Row 1: lifecycle (Connect/Disconnect) + stream-seg toggle + cadence.
        # Each widget gets stretch=1 with no trailing addStretch so the
        # row fills the col 2-4 span (matches the right edge set by
        # Robot's "Remove Overrides" / Marker's "Refresh") instead of
        # clustering on the left.
        actions_row1 = QHBoxLayout()
        actions_row1.setContentsMargins(0, 0, 0, 0)
        for w in (self.btn_vlm_service_start, self.btn_vlm_service_stop,
                  self.btn_vlm_seg_stream, self.spin_vlm_seg_hz):
            actions_row1.addWidget(w, 1)
        actions_row1_holder = _fixed_v(QWidget()); actions_row1_holder.setLayout(actions_row1)

        # Row 2: ad-hoc commands (status, decide-once, depth-once,
        # sequential-pair). Same col span and stretch policy as row 1
        # so Status sits directly under Connect.
        actions_row2 = QHBoxLayout()
        actions_row2.setContentsMargins(0, 0, 0, 0)
        for w in (self.btn_vlm_service_status, self.btn_vlm_service_decide,
                  self.btn_vlm_service_recognize,
                  self.btn_vlm_service_depth, self.btn_vlm_capture_first,
                  self.btn_vlm_decide_pair, self.lbl_vlm_pair_token):
            actions_row2.addWidget(w, 1)
        actions_row2_holder = _fixed_v(QWidget()); actions_row2_holder.setLayout(actions_row2)

        # Title carries the legend so dots-only LED column matches the
        # Robot row's style — see GPU_Service_Cross_Host_Hardening_Notes
        # for the design rationale.
        pipeline_title = QLabel(
            "<b>Perception Pipeline</b><br>"
            "<i>(send / compute / receive)</i>"
        )
        grid.addWidget(pipeline_title,      row, 0)
        grid.addWidget(leds_holder,         row, 1)
        grid.addWidget(actions_row1_holder, row, 2, 1, 3)
        row += 1
        grid.addWidget(actions_row2_holder, row, 2, 1, 3)
        row += 1

        # Aggregate VLM-specific widgets so backend gating (legacy mode)
        # can hide them in one shot. The pipeline-block title and the
        # Send LED stay visible because the frame_relay is shared infra
        # (gaze_runner consumes it too); only Compute/Receive LEDs and
        # the VLM-only commands are hidden.
        self.vlm_row_widgets = [
            self.lbl_compute_led, self.lbl_receive_led,
            self.btn_vlm_seg_stream, self.spin_vlm_seg_hz,
            self.btn_vlm_service_status,
            self.btn_vlm_service_decide,
            self.btn_vlm_service_recognize,
            self.btn_vlm_service_depth,
            self.btn_vlm_capture_first,
            self.btn_vlm_decide_pair,
            self.lbl_vlm_pair_token,
        ]
        return row

    def build_vlm_video_tab(self, tabs: QTabWidget) -> None:
        """Linux-side scene + JSON-overlay renderer
        (Render_Layer_Refactor.md §4). Bundles from a panel-hosted
        FrameRelayServer (or an externally-supplied one when
        FRAME_RELAY_EMBEDDED=False), detection JSON from vlm_service.py
        UDP 5589 subscribe, gaze tracks from gaze_runner.py UDP 5588,
        composited at native frame rate via Utils.scene_overlay_renderer.

        Windows TCP clients still dial the embedded relay's listening
        socket — the wire is unchanged.

        When FRAME_RELAY_EMBEDDED is True, the user must NOT also run
        `python -m Utils.frame_relay` separately on this machine: two
        Neon readers conflict at the SDK level. When False, the panel
        spawns + owns that relay process itself on Connect
        (_start_relay_subprocess) and tears it down on Disconnect, so the
        operator never launches it by hand — and the decode runs isolated
        from the panel's paint/encode load (the tearing fix).
        """
        vvt = QWidget()
        tabs.addTab(vvt, "VLM Video")
        vl = QVBoxLayout(vvt)

        from Utils.vlm_scene_widget import VLMSceneWidget
        embedded_relay = None
        if FRAME_RELAY_EMBEDDED:
            embedded_relay = {
                "bind_host": FRAME_RELAY_BIND_HOST,
                "bind_port": FRAME_RELAY_PORT,
                "hz": FRAME_RELAY_HZ,
                "neon_host": NEON_COMPANION_HOST,
            }
        # GAZE_OR_BACKEND selects which perception service the panel
        # subscribes to (same semantic as ExperimentDriver_Online_GazeTracking
        # uses to pick the active backend):
        #   "vlm"    → only subscribe to vlm_service.py (UDP 5589). The
        #              gaze_runner channel is left dark.
        #   "legacy" → only subscribe to gaze_runner.py (UDP 5588). The
        #              VLM channel is left dark.
        # Either subscriber's _JsonPushSubscriber heartbeats every ~10 s,
        # so a service that goes down then comes back is reconnected
        # automatically with no panel restart.
        if GAZE_OR_BACKEND == "vlm":
            vlm_host_arg, vlm_port_arg = VLM_SERVICE_HOST, VLM_SERVICE_PORT
            gaze_host_arg, gaze_port_arg = None, None
        else:
            vlm_host_arg, vlm_port_arg = None, None
            gaze_host_arg, gaze_port_arg = GAZE_SERVICE_HOST, GAZE_SERVICE_PORT
        self.vlm_scene_widget = VLMSceneWidget(
            vlm_host=vlm_host_arg,
            vlm_port=vlm_port_arg,
            gaze_host=gaze_host_arg,
            gaze_port=gaze_port_arg,
            relay_dial_host=FRAME_RELAY_DIAL_HOST,
            relay_dial_port=FRAME_RELAY_PORT,
            embedded_relay=embedded_relay,
        )
        # Drive the Main-tab Receive LED from whichever JsonPushSubscriber
        # the widget instantiates (vlm or gaze, by GAZE_OR_BACKEND). The
        # widget re-emits its inner subscribers' state on a single
        # bubbled signal so the panel doesn't have to know which one is
        # active.
        self.vlm_scene_widget.subscriber_state_changed.connect(
            self._on_subscriber_state
        )
        # Send LED gates on the relay's TCP handshake to its first
        # consumer — the relay has demonstrated "send is good" the
        # moment _install_client successfully delivers the handshake
        # envelope (Utils/frame_relay.py:_install_client). This is
        # independent of Pupil Labs SDK first-frame latency.
        self.vlm_scene_widget.handshake_observed.connect(
            self._on_handshake_observed
        )
        # First-publish event is repurposed: it now triggers the
        # verify_chain firing path once the GPU has confirmed a
        # fresh bundle (proving end-to-end traversal of the pipeline
        # for THIS Connect). It no longer flips the Send LED.
        self.vlm_scene_widget.first_publish_observed.connect(
            self._on_first_publish_observed
        )
        # Subscriber payloads are token-checked here so a stale
        # `chain_verify` from a prior Connect's GPU cache cannot
        # trip the Receive LED on a fresh Connect.
        self.vlm_scene_widget.vlm_payload_received.connect(
            self._on_vlm_payload_received
        )
        vl.addWidget(self.vlm_scene_widget, 1)

    def _on_vlm_video_connect(self) -> None:
        """Connect button handler. Drives the per-Connect verification
        state machine described in __init__:
          1. Generate a fresh ``_connect_token`` (time.time_ns()).
          2. Paint Send / Compute / Receive yellow ("starting") so the
             operator sees the verification is in progress.
          3. Reset the observed-flags + verify_chain attempt counter so
             a stale push from a prior session can't satisfy this
             cycle's verification.
          4. Start the widget (relay + subscriber threads), kick an
             immediate cmd=status preflight so Compute responds within
             one UDP RTT rather than waiting up to 1 s for the next
             timer tick.

        After this call returns, the state machine advances on three
        Qt-signal-driven events:
          - handshake_observed → Send green (control_panel.py:_on_handshake_observed)
          - status reply ok=True (with _send_observed) → Compute green
          - first_publish_observed + GPU has fresh bundle → fire
            cmd=verify_chain {token}; matching push → Receive green
        """
        self._connect_token = time.time_ns()
        self._send_observed = False
        self._compute_observed = False
        self._receive_observed = False
        self._verify_chain_in_flight = False
        self._verify_chain_attempts = 0
        self._set_led(self.lbl_send_led, "starting")
        self.lbl_send_led.setToolTip("send: verifying — awaiting handshake")
        self._set_led(self.lbl_compute_led, "starting")
        self.lbl_compute_led.setToolTip("compute: verifying — awaiting GPU status reply")
        self._set_led(self.lbl_receive_led, "starting")
        self.lbl_receive_led.setToolTip(
            "receive: verifying — awaiting end-to-end chain_verify response"
        )
        self._log(
            "VLM",
            f"[{self._ts()}] chain: connect armed token={self._connect_token}\n",
        )
        # Separated-decode mode: bring the relay child process up first so its
        # listening socket is ready; the widget's RemoteFrameReader auto-reconnects
        # regardless of ordering. No-op when FRAME_RELAY_EMBEDDED (widget hosts it).
        if not FRAME_RELAY_EMBEDDED:
            self._start_relay_subprocess()
        if hasattr(self, "vlm_scene_widget"):
            self.vlm_scene_widget.start()
        if getattr(self, "_relay_status_timer", None) is not None:
            self._poll_relay_status()
        if SERVICES_HOSTED_REMOTELY and getattr(self, "_remote_status_timer", None) is not None:
            self._poll_remote_status()

    def _on_vlm_video_disconnect(self) -> None:
        token = self._connect_token
        self._log("VLM", f"[{self._ts()}] chain: disconnect token={token}\n")
        if hasattr(self, "vlm_scene_widget"):
            self.vlm_scene_widget.stop()
        # Tear down the separated relay child process (if any) so its Neon
        # subscription + :5591 bind are released for the next Connect.
        if not FRAME_RELAY_EMBEDDED:
            self._stop_relay_subprocess()
        # Tear down state-machine state so a subsequent Connect starts
        # fresh. _connect_token=None disqualifies any late-arriving
        # chain_verify push from the prior session even if it slips
        # through the subscriber teardown race.
        self._connect_token = None
        self._send_observed = False
        self._compute_observed = False
        self._receive_observed = False
        self._verify_chain_in_flight = False
        self._verify_chain_attempts = 0
        # Reset all three LEDs to gray explicitly. _poll_relay_status
        # gates on _send_observed and won't repaint Send after we
        # cleared the flag above (control_panel.py:_poll_relay_status),
        # so without this Send would remain stuck on its last green
        # state. Doing the same for Compute and Receive keeps the
        # idle-state appearance consistent across all three rather
        # than relying on side-channels (next status poll for Compute,
        # the "unsubscribed" handler for Receive).
        self._set_led(self.lbl_send_led, "stopped")
        self.lbl_send_led.setToolTip("send: idle")
        self._set_led(self.lbl_compute_led, "stopped")
        self.lbl_compute_led.setToolTip("compute: idle")
        self._set_led(self.lbl_receive_led, "stopped")
        self.lbl_receive_led.setToolTip("receive: idle")

    def _start_relay_subprocess(self) -> None:
        """Spawn ``Utils.frame_relay`` as a child process (separated-decode mode,
        FRAME_RELAY_EMBEDDED=False). The relay opens Neon and runs the H.264 decode
        in its OWN process/GIL; the VLM Video widget consumes it over TCP via
        RemoteFrameReader. This isolates the decode from the panel's paint+encode
        load, which is what prevents the in-process-contention frame tearing
        (rootcause record 2026-06-22). ``--reader scene_only`` keeps the decode
        identical to the old in-process path (simple receive_scene_video_frame).

        Only spawns when the dial host is local: a remote FRAME_RELAY_DIAL_HOST
        means the relay lives on another machine (GPU-host split) and the panel is
        consume-only."""
        dial = str(FRAME_RELAY_DIAL_HOST or "127.0.0.1")
        if dial not in ("127.0.0.1", "localhost", "0.0.0.0"):
            self._log(
                "Relay",
                f"[{self._ts()}] dial host {dial} is remote — not spawning a "
                f"local relay (consume-only).\n",
            )
            return
        q = self.frame_relay_proc.q
        if q is not None and q.state() != QProcess.NotRunning:
            return  # already running
        py = sys.executable
        self.frame_relay_proc.cmd = (
            f'"{py}" -u -m Utils.frame_relay '
            f'--bind {FRAME_RELAY_BIND_HOST} --port {int(FRAME_RELAY_PORT)} '
            f'--hz {FRAME_RELAY_HZ} --neon-host "{NEON_COMPANION_HOST}" '
            f'--reader scene_only'
        )
        self._log(
            "Relay",
            f"[{self._ts()}] starting separated relay (decode isolated): "
            f"{self.frame_relay_proc.cmd}\n",
        )
        self.procs.start(self.frame_relay_proc, None, "Relay")

    def _stop_relay_subprocess(self) -> None:
        """Terminate the separated relay child process if running."""
        if self.frame_relay_proc.q is not None:
            self.procs.stop(self.frame_relay_proc, None, "Relay")

    def _on_handshake_observed(self, addr) -> None:
        """Slot for VLMSceneWidget.handshake_observed. The relay has
        successfully delivered its handshake envelope to a TCP
        consumer, so the Send utility is provably operational —
        independent of how long the SDK takes to deliver the first
        scene frame. Flip Send green and kick the status RPC so
        Compute can follow within one UDP roundtrip.
        """
        try:
            host, port = (addr[0], int(addr[1]))
        except (TypeError, ValueError, IndexError):
            host, port = "?", 0
        self._send_observed = True
        self._set_led(self.lbl_send_led, "running")
        self.lbl_send_led.setToolTip(
            f"send: handshake delivered to {host}:{port}"
        )
        self._log(
            "VLM",
            f"[{self._ts()}] chain: send handshake to {host}:{port}\n",
        )
        if SERVICES_HOSTED_REMOTELY and getattr(self, "_remote_status_timer", None) is not None:
            self._poll_remote_status()

    def _on_first_publish_observed(self, addr) -> None:
        """Slot for VLMSceneWidget.first_publish_observed. The relay
        has delivered the FIRST real frame envelope to its consumer,
        so we know the TCP send path is exercised end-to-end with
        actual data this Connect. Kick a status poll so Compute can
        confirm a fresh bundle on the GPU side; the verify_chain
        firing path runs from _apply_remote_status once that fresh
        bundle is observed.

        The Send LED itself was already flipped on the earlier
        handshake event (see _on_handshake_observed); this slot does
        not touch it.
        """
        try:
            host, port = (addr[0], int(addr[1]))
        except (TypeError, ValueError, IndexError):
            host, port = "?", 0
        self._log(
            "VLM",
            f"[{self._ts()}] chain: first frame published to {host}:{port}\n",
        )
        if SERVICES_HOSTED_REMOTELY and getattr(self, "_remote_status_timer", None) is not None:
            self._poll_remote_status()

    def _on_vlm_payload_received(self, payload: dict) -> None:
        """Slot for VLMSceneWidget.vlm_payload_received. Token-checks
        chain_verify responses against the current Connect's token to
        flip Receive green ONLY for this cycle's verification round-
        trip. Stale pushes from the GPU's prior-session cache (which
        survive across reconnects per vlm_service.py:344-353) carry
        either no token or a stale one and are ignored here for LED
        purposes — they still hit the video-tab render pipeline via
        the widget's own _on_vlm_payload handler.
        """
        if not isinstance(payload, dict):
            return
        if payload.get("type") != "chain_verify":
            return
        token = payload.get("token")
        current = self._connect_token
        if current is None or token != current:
            return
        if not self._compute_observed:
            # Out-of-order: chain_verify came back before the panel
            # observed Compute green. Defer the Receive flip — the
            # next status reply will catch up and we'll re-evaluate.
            # In practice this is a vanishingly small race window,
            # but we'd rather hold than violate the order invariant.
            return
        self._receive_observed = True
        self._set_led(self.lbl_receive_led, "running")
        self.lbl_receive_led.setToolTip(
            f"receive: end-to-end verified (token={token})"
        )
        self._log(
            "VLM",
            f"[{self._ts()}] chain: receive verified token={token}\n",
        )

    def _on_subscriber_state(self, state: str) -> None:
        """Slot for VLMSceneWidget.subscriber_state_changed. Under
        the new verification model, the Receive LED's green flip is
        owned by _on_vlm_payload_received (token-matched chain_verify),
        so this handler only manages the non-running cases:
          - "subscribed"     → keep yellow (verification in progress)
          - "receiving:<t>"  → no LED change here; token check decides
          - "error: …"       → red
          - "unsubscribed"   → gray
        """
        if state.startswith("receiving"):
            return
        if state == "subscribed":
            # Don't downgrade from a previously-green Receive (could
            # happen if the subscriber thread races a token-matched
            # push). Otherwise hold yellow during verification.
            return
        if state.startswith("error"):
            self._set_led(self.lbl_receive_led, "error")
            self.lbl_receive_led.setToolTip(f"receive: {state}")
            return
        # "unsubscribed" or anything we don't recognise — gray.
        self._set_led(self.lbl_receive_led, "stopped")
        self.lbl_receive_led.setToolTip(f"receive: {state}")

    def _fire_verify_chain(self) -> None:
        """Send `cmd=verify_chain {token}` to the GPU on a worker
        thread. The token-matched chain_verify push back to the
        subscriber is what flips Receive green via
        _on_vlm_payload_received; this method only initiates the
        round-trip and handles the failure paths.

        Retries once after 200 ms on `no_frame` (a short race window
        where the relay's first frame is still in flight to the GPU
        when verify_chain arrives). Beyond that, paints Receive red.

        Schedules a 5 s deadline check for the case where the GPU
        replies ok=True with subscribers_notified=0 (subscribe-RPC
        race) or where the chain_verify push is lost in transit. If
        no token-matched push has landed by the deadline, Receive
        goes red.
        """
        if self._connect_token is None:
            return
        self._verify_chain_in_flight = True
        self._verify_chain_attempts += 1
        token = self._connect_token
        attempt = self._verify_chain_attempts
        self._log(
            "VLM",
            f"[{self._ts()}] chain: verify_chain TX token={token} attempt={attempt}\n",
        )
        # Receive deadline: per-token guard so an old timer can't
        # nuke a current-Connect verification.
        QTimer.singleShot(
            5000, self,
            lambda t=token: self._verify_chain_deadline_check(t),
        )

        def worker():
            try:
                resp = self._vlm_udp_request(
                    {"cmd": "verify_chain", "token": token},
                    timeout_s=2.0,
                )
            except Exception as e:
                resp = {"ok": False, "error": f"udp_exception: {e}"}
            try:
                QTimer.singleShot(
                    0, self,
                    lambda r=resp, t=token: self._on_verify_chain_reply(r, t),
                )
            except RuntimeError:
                # Window closed mid-RPC.
                pass

        threading.Thread(
            target=worker, daemon=True, name="panel-verify-chain"
        ).start()

    def _on_verify_chain_reply(self, resp: dict, token: int) -> None:
        """GUI-thread slot for verify_chain RPC reply. ok=True here
        means "GPU dispatched the synthetic push" — the actual Receive
        LED flip waits for the push to land at the subscriber and
        pass the token check (_on_vlm_payload_received). This handler
        only covers failure cases.
        """
        self._verify_chain_in_flight = False
        if token != self._connect_token:
            # Connect changed under us; the response belongs to a
            # prior cycle. Drop it.
            return
        if resp.get("ok"):
            return  # Success path — push will land on subscriber.
        err = resp.get("error", "unknown")
        if err == "no_frame" and self._verify_chain_attempts < 2:
            QTimer.singleShot(200, self, self._fire_verify_chain)
            return
        self._set_led(self.lbl_receive_led, "error")
        self.lbl_receive_led.setToolTip(
            f"receive: verify_chain RPC failed: {err}"
        )
        self._log(
            "VLM",
            f"[{self._ts()}] chain: verify_chain FAILED token={token} error={err}\n",
        )

    def _verify_chain_deadline_check(self, token: int) -> None:
        """Fired 5 s after _fire_verify_chain. If no token-matched
        chain_verify push has landed by now, paint Receive red.
        Token-discriminated so a stale older-Connect timer doesn't
        revert a successful current verification."""
        if token != self._connect_token:
            return
        if self._receive_observed:
            return
        self._set_led(self.lbl_receive_led, "error")
        self.lbl_receive_led.setToolTip(
            "receive: verify_chain push did not arrive (timeout)"
        )
        self._log(
            "VLM",
            f"[{self._ts()}] chain: verify_chain TIMEOUT token={token}\n",
        )

    def configure_remote_services_ui(self) -> None:
        """When SERVICES_HOSTED_REMOTELY=True, the GPU service runs on
        another host so spawning a local copy is meaningless. We
        rebrand the VLM Start/Stop pair to "Connect"/"Disconnect" and
        rewire them to the local-pipeline lifecycle (Neon reader,
        embedded frame_relay, push subscribers via VLMSceneWidget) —
        the user's actual pre-session toggle on this box.

        Local-spawn handlers (on_vlm_service_start/stop) are
        disconnected and replaced with _on_vlm_video_connect/disconnect.
        The lbl_compute_led keeps its existing semantic — green
        when the GPU service is reachable AND has a frame source,
        which is exactly what "connected" should mean here. Status,
        Decide, Depth buttons stay as-is — they're plain UDP queries
        that travel across the LAN unchanged.

        Gaze Start variants stay visible-but-disabled so the
        surrounding grid row keeps its column alignment with the
        other module rows.
        """
        for btn, new_text, new_handler, old_handler, tip in (
            (
                getattr(self, "btn_vlm_service_start", None),
                "Connect",
                self._on_vlm_video_connect,
                self.on_vlm_service_start,
                "Open Neon over the tailnet, start the embedded "
                "frame_relay, and subscribe to vlm_service push on "
                "the GPU host. The GPU service itself runs on a "
                "different machine and is not started by this button.",
            ),
            (
                getattr(self, "btn_vlm_service_stop", None),
                "Disconnect",
                self._on_vlm_video_disconnect,
                self.on_vlm_service_stop,
                "Stop the local pipeline (paint loop, push "
                "subscribers, embedded relay, Neon reader). The GPU "
                "service is unaffected — it stays up waiting for the "
                "next client.",
            ),
        ):
            if btn is None:
                continue
            try:
                btn.clicked.disconnect(old_handler)
            except (RuntimeError, TypeError):
                pass
            btn.clicked.connect(new_handler)
            btn.setText(new_text)
            btn.setToolTip(tip)
        for btn_name in (
            "btn_gaze_service_headless", "btn_gaze_service_ui",
            "btn_gaze_service_stop",
        ):
            btn = getattr(self._parent.gaze, btn_name, None)
            if btn is not None:
                btn.setEnabled(False)
                btn.setToolTip("Disabled: SERVICES_HOSTED_REMOTELY=True. "
                               "Manage these on the GPU host.")

    def _poll_remote_status(self) -> None:
        """1 s cadence. Spawns a daemon thread to do the UDP RTT off the
        GUI thread; the result comes back via the _remote_status_received
        signal. The status request itself is cheap, but its 0.4 s timeout
        blocked the Qt event loop when the Windows host is unreachable —
        visible as a sub-second stutter in the VLM Video tab paint pass.

        Skip if a previous poll is still in flight so a flaky link can't
        accumulate worker threads.
        """
        if self._remote_status_in_flight:
            return
        self._remote_status_in_flight = True

        def _worker() -> None:
            try:
                resp = self._vlm_udp_request({"cmd": "status"}, timeout_s=0.4)
            except Exception:
                resp = {"ok": False, "_unreachable": True}
            try:
                self._remote_status_received.emit(resp or {})
            except RuntimeError:
                # Window closed while we were mid-UDP — the underlying
                # C++ ControlPanel is already gone. Drop the late status
                # reply silently; the alternative is a noisy traceback
                # at every shutdown when the GPU host is unreachable.
                # Other workers route through _append_log_ui's
                # QTimer.singleShot(self, ...) which Qt auto-cancels;
                # this one is the lone direct-emit path.
                pass

        threading.Thread(target=_worker, daemon=True,
                         name="panel-remote-status").start()

    def _apply_remote_status(self, resp: dict) -> None:
        """GUI-thread slot for _remote_status_received.

        Compute LED semantic (per the per-Connect verification model):
          - gray  (stopped)  → unreachable
          - red   (error)    → reachable but ok=False
          - green (running)  → reachable AND ok AND _send_observed
        "Compute green" means "the GPU script is alive and ready to
        accept data" — explicitly NOT gated on frames_received, since
        actual data flow is what the Receive verification step proves.

        Side-effect: when this reply confirms a fresh bundle on the
        GPU side (frame_age < 2 s) AND the verification phase is
        ready to fire (_send_observed and _compute_observed and a
        token is armed), trigger cmd=verify_chain with the current
        token. The token-matched chain_verify push back to the panel
        is what flips the Receive LED via _on_vlm_payload_received.
        """
        self._remote_status_in_flight = False
        if resp.get("_unreachable"):
            self._set_led(self.lbl_compute_led, "stopped")
            self.lbl_compute_led.setToolTip("compute: unreachable")
            return
        ok = bool(resp.get("ok"))
        connected = bool(resp.get("frame_source_connected"))
        frames = int(resp.get("frames_received") or 0)
        src = resp.get("frame_source", "?")
        age = resp.get("frame_age_s")
        age_txt = f"{float(age):.2f}s" if isinstance(age, (int, float)) else "--"
        if not ok:
            led_state = "error"
        elif self._send_observed:
            # User spec: Compute = "GPU script is alive and ready". Send
            # must already be observed for visible ordering, but we do
            # not require frames_received>0 here — that's the Receive
            # phase's job.
            led_state = "running"
        elif self._connect_token is not None:
            # Verification in progress (Connect armed) but Send hasn't
            # been observed yet — hold yellow.
            led_state = "starting"
        else:
            # Idle (no Connect armed). The 1 s status poll runs
            # continuously even before the operator clicks Connect, so
            # we must NOT paint yellow here — yellow is the "verifying"
            # state and is reserved for the active Connect cycle. Keep
            # gray to match the other LEDs in the idle state.
            led_state = "stopped"
        self._set_led(self.lbl_compute_led, led_state)
        self.lbl_compute_led.setToolTip(
            f"compute: src={src} connected={connected} frames={frames} age={age_txt}"
        )
        if led_state == "running" and not self._compute_observed:
            self._compute_observed = True
            self._log(
                "VLM",
                f"[{self._ts()}] chain: compute green ok=True frames={frames}\n",
            )
        # End-to-end Receive verification trigger. We need:
        #   - both Send and Compute observed (predecessor phases done)
        #   - GPU has a fresh bundle (connected=True, frame_age<2s) so
        #     verify_chain doesn't return no_frame
        #   - this Connect's token is armed
        #   - no in-flight verify_chain already
        # The verify_chain RPC echoes our token in its push payload;
        # _on_vlm_payload_received does the matching to flip Receive
        # green. Stale GPU-cache pushes (no token / old token) cannot
        # trip Receive on a reconnect.
        if (self._compute_observed
                and self._send_observed
                and connected
                and self._connect_token is not None
                and not self._verify_chain_in_flight
                and self._verify_chain_attempts == 0):
            self._fire_verify_chain()

    def _poll_relay_status(self) -> None:
        """2 s cadence. Reflects whether the frame relay is alive.

        Under the per-Connect verification model, the Send LED's
        green flip is owned by the handshake event
        (_on_handshake_observed). This poll therefore only owns the
        running→stopped edge — i.e. detecting that the relay thread
        has died after Send was already observed. While verification
        is in progress (_send_observed=False) this poll is a no-op
        for the LED; the state machine paints it.

        When the panel hosts the relay in-process (FRAME_RELAY_EMBEDDED),
        we ask the widget directly — TCP-pinging localhost would create
        phantom client churn (each ping does connect-then-close, the
        relay's accept loop installs the dead socket, the pump pays a
        full JPEG encode + sendall before discovering the peer is gone,
        and the SDK iterator stalls behind that work → visible stutter
        in the local subscriber path).
        """
        if not self._send_observed:
            return  # State machine owns Send before handshake.
        widget = getattr(self, "vlm_scene_widget", None)
        if widget is not None and getattr(widget, "_embedded_relay", None) is not None:
            thread = getattr(widget, "_embedded_relay_thread", None)
            alive = thread is not None and thread.is_alive()
            relay = widget._embedded_relay
            published = int(getattr(relay, "published_count", 0) or 0)
            if alive:
                # Send was observed via handshake; keep green and
                # surface the published-count in the tooltip for
                # diagnostics.
                self._set_led(self.lbl_send_led, "running")
                self.lbl_send_led.setToolTip(
                    f"send: in-process @ {FRAME_RELAY_BIND_HOST}:{FRAME_RELAY_PORT} "
                    f"(published={published})"
                )
            else:
                # Relay thread died after Send was observed.
                self._set_led(self.lbl_send_led, "stopped")
                self.lbl_send_led.setToolTip("send: in-process — thread exited")
            return

        # External relay (FRAME_RELAY_EMBEDDED=False or remote host) —
        # fall back to the TCP ping for the running→stopped edge.
        try:
            from Utils.perception_clients import FrameRelayController
        except Exception:
            return
        ctl = FrameRelayController(_HCFG) if _HCFG else None
        if ctl is None:
            return
        ping = ctl.ping(timeout_s=0.5)
        if ping.get("ok"):
            self._set_led(self.lbl_send_led, "running")
            self.lbl_send_led.setToolTip(
                f"send: reachable @ {ping['host']}:{ping['port']}"
            )
        else:
            self._set_led(self.lbl_send_led, "stopped")
            self.lbl_send_led.setToolTip(
                f"send: unreachable @ {ping['host']}:{ping['port']}"
            )

    def _vlm_udp_request(self, payload: dict, timeout_s: float = VLM_QUERY_TIMEOUT_S) -> dict:
        """One-shot JSON request against vlm_service.py.

        Delegates to ``Utils.perception_clients.udp_request`` so the wire
        format (JSON, UDP, single datagram round-trip) lives in one place;
        the panel layer here is purely UI plumbing.
        """
        from Utils.perception_clients import udp_request
        return udp_request(VLM_SERVICE_HOST, VLM_SERVICE_PORT, payload, float(timeout_s))

    def on_vlm_service_start(self):
        if not os.path.exists(VLM_SERVICE_PY):
            QMessageBox.warning(self._parent, "Missing", f"Not found:\n{VLM_SERVICE_PY}")
            return
        if not PERCEPTION_MODELS_DIR or not os.path.isdir(PERCEPTION_MODELS_DIR):
            QMessageBox.warning(self._parent, "Perception models missing",
                                f"PERCEPTION_MODELS_DIR not a dir:\n{PERCEPTION_MODELS_DIR}")
            return

        # Reap any orphaned vlm_service.py left over from a previous crash or
        # incomplete stop (a killed service can leave its python holding the port).
        _kill_orphan_vlm_service()

        if _is_port_in_use(int(VLM_SERVICE_PORT), VLM_SERVICE_HOST):
            QMessageBox.warning(
                self._parent,
                "VLM service port in use",
                f"UDP port {VLM_SERVICE_HOST}:{VLM_SERVICE_PORT} appears in use.\n"
                f"Use Stop first or change VLM_SERVICE_PORT."
            )

        session_dir = ""
        if VLM_SESSION_ROOT:
            try:
                os.makedirs(VLM_SESSION_ROOT, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                session_dir = os.path.join(VLM_SESSION_ROOT, f"session_{ts}")
                os.makedirs(session_dir, exist_ok=True)
            except OSError as e:
                self._log("VLM", f"[{self._ts()}] Failed to create session dir: {e}\n")

        depth_flag = "--enable-depth" if VLM_ENABLE_DEPTH else ""
        session_arg = f'--session-dir "{session_dir}"' if session_dir else ""
        # On the Windows dev box (RTX 4070 Ti) we want FastSAM + Depth Pro on
        # CUDA; the Linux deployment is CPU-only (no NVIDIA driver).
        device_flag = "--device cuda" if _IS_WINDOWS else "--device cpu"
        # Launch vlm_service.py with the panel's own interpreter. Since WS3
        # unified the env (perception deps now live in this env), there is no
        # separate harmony_vlm env to resolve — same env, still a separate
        # process so a blocking model call can't stall the panel.
        py = sys.executable
        self._log("VLM", f"[{self._ts()}] using python: {py}\n")
        # --neon-host "" forces discover_one_device in perception.neon's
        # NeonLiveReader (neon/reader.py), matching our gaze_system.py:250 pattern.
        # GPU-host topology: when PERCEPTION_FRAME_SOURCE=remote, vlm_service
        # consumes envelopes from the Linux frame_relay rather than opening
        # Neon itself. Dial host comes from FRAME_RELAY_DIAL_HOST.
        remote_arg = ""
        if PERCEPTION_FRAME_SOURCE == "remote":
            relay_dial = str(getattr(_HCFG, "FRAME_RELAY_DIAL_HOST", "127.0.0.1") or "127.0.0.1") if _HCFG else "127.0.0.1"
            relay_port = int(getattr(_HCFG, "FRAME_RELAY_PORT", 5591)) if _HCFG else 5591
            remote_arg = (
                f'--frame-source remote '
                f'--remote-frame-host {relay_dial} '
                f'--remote-frame-port {relay_port}'
            )
        self.vlm_service.cmd = (
            f'"{py}" -u "{VLM_SERVICE_PY}" '
            f'--host {VLM_BIND_HOST} --port {int(VLM_SERVICE_PORT)} '
            f'--neon-host "{NEON_COMPANION_HOST}" '
            f'--model {VLM_MODEL} {device_flag} '
            f'{depth_flag} {session_arg} {remote_arg}'
        )
        self.procs.start(self.vlm_service, self.lbl_compute_led, "VLM")
        self._log(
            "VLM",
            f"[{self._ts()}] Service start requested "
            f"(depth={VLM_ENABLE_DEPTH}, model={VLM_MODEL})\n",
        )
        # Block Windows from sleeping while the GPU stack is alive.
        _sleep_inhibit(True)
        self._on_vlm_video_connect()

    def on_vlm_service_stop(self):
        # Reset the streaming toggle so the panel doesn't claim "ON" while
        # the service it was driving is dead. blockSignals prevents the
        # toggle handler from firing a doomed UDP send to the dying service.
        if self.btn_vlm_seg_stream.isChecked():
            self.btn_vlm_seg_stream.blockSignals(True)
            self.btn_vlm_seg_stream.setChecked(False)
            self.btn_vlm_seg_stream.setText("Stream Seg: OFF")
            self.btn_vlm_seg_stream.blockSignals(False)
        # Stop the seg-stream readout timer too — its 5 s tick would
        # otherwise spam "unreachable" lines into the VLM log while the
        # service tears down.
        self._seg_stream_log_timer.stop()
        # Ask vlm_service to exit gracefully before killing the process. A hard
        # kill alone can leave children it spawned orphaned holding the UDP port.
        try:
            self._vlm_udp_request({"cmd": "stop"}, timeout_s=0.5)
        except Exception:
            pass
        self.procs.stop(self.vlm_service, self.lbl_compute_led, "VLM")
        # Belt-and-suspenders: reap any surviving orphan regardless.
        _kill_orphan_vlm_service()
        self._on_vlm_video_disconnect()
        _sleep_inhibit(False)

    def _vlm_command_threaded(self, payload: dict, timeout_s: float, label: str) -> None:
        import threading as _threading
        self._log("VLM", f"[{self._ts()}] {label} TX -> {VLM_SERVICE_HOST}:{VLM_SERVICE_PORT}\n")

        def worker():
            with self._vlm_inflight_lock:
                self._vlm_inflight_count += 1
            t0 = time.time()
            try:
                try:
                    resp = self._vlm_udp_request(payload, timeout_s=timeout_s)
                    dt_ms = (time.time() - t0) * 1000.0
                    pretty = json.dumps(resp, indent=2, sort_keys=True)
                    self._log_ui("VLM", f"[{self._ts()}] {label} RX OK ({dt_ms:.0f} ms)\n{pretty}\n")
                except Exception as e:
                    dt_ms = (time.time() - t0) * 1000.0
                    self._log_ui("VLM", f"[{self._ts()}] {label} RX ERROR ({dt_ms:.0f} ms): {e}\n")
            finally:
                with self._vlm_inflight_lock:
                    self._vlm_inflight_count -= 1

        _threading.Thread(target=worker, daemon=True).start()

    def on_vlm_service_status(self):
        self._vlm_command_threaded({"cmd": "status"}, VLM_QUERY_TIMEOUT_S, "status")

    def on_vlm_service_decide(self):
        self._vlm_command_threaded({"cmd": "decide"}, VLM_DECIDE_TIMEOUT_S, "decide")

    def on_vlm_service_recognize(self):
        # F5: fast COCO recognition of the gaze object (no Gemini). 5 s covers a
        # warm YOLO inference comfortably while staying well under decide's 40 s.
        self._vlm_command_threaded({"cmd": "recognize"}, 5.0, "recognize")

    def on_vlm_seg_stream_toggled(self, checked: bool) -> None:
        hz = float(self.spin_vlm_seg_hz.value())
        self.btn_vlm_seg_stream.setText(f"Stream Seg: {'ON' if checked else 'OFF'}")
        self._vlm_command_threaded(
            {"cmd": "segment_stream", "enabled": bool(checked), "hz": hz},
            VLM_QUERY_TIMEOUT_S,
            f"segment_stream({'on' if checked else 'off'}, {hz:.1f} Hz)",
        )
        # Drive the seg-stream readout off the toggle: the timer is only
        # meaningful while the stream is running. Reset the dedup marker
        # on each on-edge so the very first stats window emits.
        if checked:
            self._last_seg_stream_emit_t = 0.0
            self._seg_stream_log_timer.start()
        else:
            self._seg_stream_log_timer.stop()

    def on_vlm_seg_stream_hz_changed(self, hz: float) -> None:
        # Only push a rate change if the stream is currently on; otherwise
        # the spinner just sets the rate the next toggle-on will use.
        if not self.btn_vlm_seg_stream.isChecked():
            return
        self._vlm_command_threaded(
            {"cmd": "segment_stream", "enabled": True, "hz": float(hz)},
            VLM_QUERY_TIMEOUT_S,
            f"segment_stream(rate={hz:.1f} Hz)",
        )

    def on_vlm_service_depth(self):
        self._vlm_command_threaded({"cmd": "depth", "at_gaze": True}, 15.0, "depth")

    def _poll_seg_stream_stats(self) -> None:
        """Pull the seg-stream stats block from the VLM status reply and
        append a one-line summary to the VLM log buffer. Runs on a 5 s
        timer that's only active while the operator has Stream Seg on.

        The status request is the same UDP roundtrip the existing handlers
        use; we run it on a worker thread so a slow GPU host (or a service
        that just died) cannot stall the GUI thread."""
        def worker():
            # Skip the poll if a synchronous VLM command (decide, depth,
            # capture_first, ...) is in flight. The server's request loop
            # (vlm_service.py:403-437) is single-threaded, so a 500 ms status
            # probe issued while decide blocks on Gemini will time out — even
            # though seg-stream itself is on its own thread and still healthy.
            # Reporting that as "unreachable" was misleading.
            if self._vlm_inflight_count > 0:
                return
            try:
                resp = self._vlm_udp_request({"cmd": "status"}, timeout_s=0.5)
            except Exception as e:
                # No command in flight, so a status timeout actually does
                # suggest the service or host is gone — surface it.
                self._log_ui(
                    "VLM",
                    f"[{self._ts()}] seg-stream status: unreachable ({e})\n",
                )
                return
            stats = resp.get("seg_stream") if isinstance(resp, dict) else None
            if not isinstance(stats, dict):
                return
            # Skip ticks where the service hasn't refreshed its window yet
            # — avoids three identical lines while the first 5 s window
            # accumulates.
            emit_t = float(stats.get("last_emit_t") or 0.0)
            if emit_t and emit_t == self._last_seg_stream_emit_t:
                return
            self._last_seg_stream_emit_t = emit_t
            active = bool(stats.get("active"))
            target = float(stats.get("hz_target") or 0.0)
            achieved = float(stats.get("hz_achieved") or 0.0)
            mean_dets = float(stats.get("mean_dets") or 0.0)
            mean_infer = float(stats.get("mean_infer_ms") or 0.0)
            errors = int(stats.get("errors") or 0)
            self._log_ui(
                "VLM",
                f"[{self._ts()}] seg-stream: active={active} "
                f"target={target:.1f}Hz achieved={achieved:.1f}Hz "
                f"mean_dets={mean_dets:.1f} mean_infer={mean_infer:.0f}ms "
                f"errors={errors}\n",
            )

        threading.Thread(target=worker, daemon=True,
                         name="panel-seg-stream-stats").start()

    def on_vlm_capture_first(self):
        import threading as _threading
        self._log("VLM", f"[{self._ts()}] capture_first TX -> {VLM_SERVICE_HOST}:{VLM_SERVICE_PORT}\n")

        def worker():
            t0 = time.time()
            try:
                resp = self._vlm_udp_request({"cmd": "capture_first"}, timeout_s=12.0)
                dt_ms = (time.time() - t0) * 1000.0
                if isinstance(resp, dict) and resp.get("ok") and resp.get("snapshot_id"):
                    self._vlm_last_snapshot_id = str(resp["snapshot_id"])
                    hit = resp.get("hit_waypoint")
                    hit_lbl = hit.get("label") if isinstance(hit, dict) else "—"
                    self._log_ui(
                        "VLM",
                        f"[{self._ts()}] capture_first RX OK ({dt_ms:.0f} ms)\n"
                        f"{json.dumps(resp, indent=2, sort_keys=True)}\n",
                    )
                    # Update token label on the main thread
                    from PySide6.QtCore import QMetaObject, Qt, Q_ARG
                    QMetaObject.invokeMethod(
                        self.lbl_vlm_pair_token, "setText", Qt.QueuedConnection,
                        Q_ARG(str, f"<i>snapshot:</i> {self._vlm_last_snapshot_id} ({hit_lbl})"),
                    )
                else:
                    self._log_ui(
                        "VLM",
                        f"[{self._ts()}] capture_first RX (no snapshot_id) ({dt_ms:.0f} ms)\n"
                        f"{json.dumps(resp, indent=2, sort_keys=True)}\n",
                    )
            except Exception as e:
                dt_ms = (time.time() - t0) * 1000.0
                self._log_ui("VLM", f"[{self._ts()}] capture_first RX ERROR ({dt_ms:.0f} ms): {e}\n")

        _threading.Thread(target=worker, daemon=True).start()

    def on_vlm_decide_pair(self):
        if not self._vlm_last_snapshot_id:
            QMessageBox.information(
                self._parent, "No snapshot",
                "Click 'Capture First' on the source object before running 'Decide Pair'.",
            )
            return
        snap_id = self._vlm_last_snapshot_id
        self._vlm_command_threaded(
            {"cmd": "decide_pair", "snapshot_id": snap_id, "timeout": 45.0},
            60.0,
            f"decide_pair(snapshot_id={snap_id})",
        )
        # Cleared after use so accidental re-presses don't replay a stale snapshot.
        self._vlm_last_snapshot_id = None
        self.lbl_vlm_pair_token.setText("<i>snapshot:</i> (consumed)")

    def shutdown(self) -> None:
        """VLM-specific teardown invoked from the panel's closeEvent. Stops
        the three periodic timers up front (so no new worker threads can be
        spawned during teardown — the `panel-remote-status` worker calls
        Signal.emit() directly, which raises RuntimeError once the C++ object
        is destroyed), then disconnects the local pipeline.

        The vlm_service Proc itself is stopped by the panel's closeEvent loop
        (it owns the Proc handle + the ProcessManager); this method only owns
        the VLM-local timers + the disconnect path."""
        for t in (self._remote_status_timer, self._relay_status_timer,
                  self._seg_stream_log_timer):
            if t is not None:
                try:
                    t.stop()
                except Exception:
                    pass
        # Stop the overlay reader thread before tearing down its target service,
        # otherwise QProcess teardown logs errors that try to write to widgets
        # Qt has already destroyed.
        try:
            self._on_vlm_video_disconnect()
        except Exception:
            pass
