# Vendored from harmony_vlm (https://github.com/vivianchen98/harmony_vlm) @ cfa01b6
# by Vivian Chen. Folded into the BCI repo for WS3 (2026-06-15). Edit here, not
# upstream; see Documents/SoftwareDocs/projects/harmony-bci/vlm-integration/.
# STAGED — not import-safe in this env (deps deliberately excluded); see the
# live-vs-staged list in perception/__init__.py before importing.
"""
exo_controller.py — Main orchestration loop for the gaze-intent exoskeleton system.

Data flow (per frame):
  RecordingReader → ObjectDetector → FixationDetector →
  [trigger] → IntentReasoner (async) → stdout JSON + optional .jsonl
"""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import Future
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from perception.fixation_detector import FixationDetector, FixationState
from perception.intent_reasoner import IntentReasoner
from perception.pupil_reader import FrameBundle, GazeSample
from perception.neon import RecordingReader
from perception.object_detector import Detection, GazeHit, ObjectDetector, Waypoint3D, compute_3d_waypoints

# Overlay imports are deferred to __init__ to avoid import errors when
# overlay is not requested.

# ── tuning ────────────────────────────────────────────────────────────────────

COOLDOWN_SAME_OBJECT_NS = int(2.0 * 1e9)  # 2 s cooldown for repeated same object
COOLDOWN_NEW_OBJECT_NS = 0  # no cooldown for a new object
TRIGGER_MIN_DURATION_NS = int(
    3.0 * 1e9
)  # require 3 s of sustained gaze before triggering
SECOND_OBJECT_TIMEOUT_S = (
    20.0  # seconds to wait for second object before single-object fallback
)


# ── controller ────────────────────────────────────────────────────────────────


class ExoController:
    """
    Orchestrates the gaze-intent pipeline:
      - Runs YOLO + fixation detector on every frame
      - Triggers IntentReasoner on stable fixation (async)
      - Handles cooldown, debounce, and decision output
    """

    def __init__(
        self,
        recorder: RecordingReader,
        detector: ObjectDetector,
        fix_detector: FixationDetector,
        reasoner: IntentReasoner,
        *,
        save_path: Optional[Path] = None,
        debug: bool = False,
        show_overlay: bool = False,
        out_video_path: Optional[Path] = None,
        gaze_grounder=None,
        session_log_path: Optional[Path] = None,
        depth_estimator=None,
    ) -> None:
        self.recorder = recorder
        self.detector = detector
        self.fix_detector = fix_detector
        self.reasoner = reasoner
        self.save_path = save_path
        self.debug = debug
        self.show_overlay = show_overlay
        self.gaze_grounder = gaze_grounder
        self.depth_estimator = depth_estimator

        # Focal length from reader intrinsics (for metric depth)
        self._depth_f_px: Optional[float] = None
        if depth_estimator is not None:
            K = getattr(recorder, "camera_matrix", None)
            if K is not None:
                self._depth_f_px = float((K[0, 0] + K[1, 1]) / 2)
                print(f"[DepthEstimator] Using focal length from reader: {self._depth_f_px:.1f} px")

        # state
        self._pending_future: Optional[Future] = None
        self._last_trigger_ts_ns: int = 0
        self._last_object: Optional[str] = None
        self._say_proc = None  # track in-flight live TTS process
        self._say_save_procs: list = []  # save-to-file TTS processes
        self._audio_cues: list[tuple[float, str]] = []  # (timestamp_s, wav_path)
        self._tts_threads: list = []  # track TTS threads for join on exit
        self._cue_idx: int = 0
        self._current_frame_idx: int = 0
        self._out_video_path: Optional[Path] = out_video_path
        self._last_gaze_3d = None  # GazePoint3D from gaze_grounder

        self._pending_depth_info: Optional[dict] = None
        self._last_waypoints: list[dict] = []  # cached waypoints from last depth run

        # VLM state machine: IDLE → AWAITING_SECOND → THINKING → DECIDED → IDLE
        self._vlm_state: str = "IDLE"
        self._last_decision: Optional[dict] = None
        self._api_mode: str = "TEXT"
        self._pending_detection: Optional[Detection] = None

        # Multi-object sequence state
        self._first_object: Optional[Detection] = None
        self._first_object_frame: Optional[np.ndarray] = None
        self._first_object_time: float = 0.0
        self._first_object_fixation: Optional[FixationState] = None
        self._first_object_gaze: tuple[float, float] = (0.0, 0.0)
        self._first_object_detections: list = []
        self._first_object_hit_info: str = ""
        self._tts_done = threading.Event()
        self._tts_done.set()  # initially "done" (no TTS playing)
        self._awaiting_tts_finished: bool = False  # flag to reset timer after TTS
        self._last_object_center: tuple[float, float] | None = None  # for position-based cooldown

        # overlay renderer + video writer
        self._renderer = None
        self._video_writer = None

        if show_overlay or out_video_path:
            from perception.overlay_renderer import OverlayRenderer

            self._renderer = OverlayRenderer()

        if out_video_path:
            from perception.overlay_renderer import OverlayRenderer as OR

            out_video_path.parent.mkdir(parents=True, exist_ok=True)
            self._out_video_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._out_video_fps = recorder.fps

        # optional decisions log (legacy)
        self._log_file = open(save_path, "a") if save_path else None

        # ── session logging ──────────────────────────────────────────────
        self._session_start_time = datetime.now()
        self._session_events: list[dict] = []  # in-memory for summary
        # Capture config for session log
        gaze_off = getattr(recorder, "gaze_offset", (0.0, 0.0))
        is_live = hasattr(recorder, "_device")  # NeonLiveReader / PupilCoreReader
        reader_type = type(recorder).__name__
        self._session_config = {
            "vlm_model": reasoner.model,
            "seg_model": getattr(detector, "_is_efficientsam", False) and "EfficientSAM"
                         or getattr(detector, "_is_fastsam", False) and "FastSAM"
                         or "YOLO-seg",
            "device": detector.device,
            "mode": "live" if is_live else "recording",
            "reader": reader_type,
            "gaze_calibration": f"dx={gaze_off[0]:.1f}, dy={gaze_off[1]:.1f}" if gaze_off != (0.0, 0.0) else "none",
            "overlay": show_overlay,
            "debug": debug,
        }
        if session_log_path:
            session_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._session_vlm_log_path = session_log_path
            reasoner._session_vlm_log = session_log_path
            reasoner._session_dir = session_log_path.parent
        else:
            self._session_vlm_log_path = None

        # ── calibration state ────────────────────────────────────────────
        self._calib_mode: bool = False
        self._calib_tracker = None  # CalibrationErrorTracker, lazy-init
        self._apriltag_detector = None  # AprilTagDetector, lazy-init
        self._calib_output_path = Path("calibration/gaze_calibration.json")
        self._calib_tag_detection = None  # current frame's tag detection
        self._calib_sample = None  # current frame's CalibrationSample
        self._calib_calibrated: bool = False  # True after first successful calibration
        self._calib_last_frame: np.ndarray | None = None  # last frame with tag visible
        self._calib_last_gaze: tuple[float, float] = (0.0, 0.0)  # raw gaze on last tag frame
        self._calib_run_count: int = 0

    # ── session logging ────────────────────────────────────────────────────────

    def _log_session_event(self, event: dict) -> None:
        """Append a timestamped event to the in-memory list (for markdown summary)."""
        event.setdefault("timestamp", datetime.now().isoformat())
        event.setdefault("frame_idx", self._current_frame_idx)
        self._session_events.append(event)

    def _write_session_summary(self) -> None:
        """Prepend a summary section to the session VLM log file."""
        if not self._session_vlm_log_path:
            return
        log_path = self._session_vlm_log_path
        duration = datetime.now() - self._session_start_time
        minutes = duration.total_seconds() / 60

        calib_events = [e for e in self._session_events if e["type"].startswith("calibration")]
        vlm_events = [e for e in self._session_events if e["type"] == "vlm_decision"]
        calib_ends = [e for e in calib_events if e["type"] == "calibration_end"]

        cfg = self._session_config
        lines = [
            f"# Session Log — {self._session_start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"## Configuration",
            f"",
            f"| Setting | Value |",
            f"|---------|-------|",
            f"| VLM model | {cfg['vlm_model']} |",
            f"| Seg model | {cfg['seg_model']} |",
            f"| Device | {cfg['device']} |",
            f"| Mode | {cfg['mode']} |",
            f"| Reader | {cfg['reader']} |",
            f"| Gaze calibration | {cfg['gaze_calibration']} |",
            f"| Overlay | {cfg['overlay']} |",
            f"",
            f"## Stats",
            f"",
            f"- **Duration**: {minutes:.1f} min",
            f"- **Frames processed**: {self._current_frame_idx}",
            f"- **VLM calls**: {len(vlm_events)}",
            f"- **Calibration runs**: {len(calib_ends)}",
            f"",
        ]

        # Calibration summary table
        if calib_ends:
            lines.append("## Calibration Summary")
            lines.append("")
            for i, ce in enumerate(calib_ends, 1):
                lines.append(f"### Run {i}")
                lines.append(f"- Samples: {ce.get('n_samples', '?')}")
                offset = ce.get('offset_px', [0, 0])
                lines.append(f"- Offset: dx={offset[0]:.1f}, dy={offset[1]:.1f} px")
                lines.append(f"- Error before: {ce.get('mean_error_before', 0):.2f} deg")
                lines.append(f"- Error after: {ce.get('mean_error_after', 0):.2f} deg")
                lines.append("")

        # VLM interactions table
        if vlm_events:
            lines.append("## VLM Interactions Summary")
            lines.append("")
            lines.append("| Time | Object | Second Object | Top Intent | Confidence | Depth | Question |")
            lines.append("|------|--------|---------------|------------|------------|-------|----------|")
            for ve in vlm_events:
                ts = ve.get("timestamp", "")
                if "T" in ts:
                    ts = ts.split("T")[1][:8]
                obj = ve.get("object", "—") or "—"
                obj2 = ve.get("second_object", "") or ""
                candidates = ve.get("candidates", [])
                top_intent = candidates[0].get("intent", "?") if candidates else "?"
                top_conf = candidates[0].get("confidence", 0) if candidates else 0
                depth_info = ve.get("depth")
                if depth_info:
                    parts = []
                    for key in ("object_1", "object_2"):
                        d = depth_info.get(key)
                        if d:
                            parts.append(f"{d['depth_at_gaze_m']:.2f}m")
                    depth_str = " / ".join(parts) if parts else "—"
                else:
                    depth_str = "—"
                question = ve.get("clarification_question", "") or ""
                if len(top_intent) > 40:
                    top_intent = top_intent[:37] + "..."
                if len(question) > 40:
                    question = question[:37] + "..."
                lines.append(f"| {ts} | {obj} | {obj2} | {top_intent} | {top_conf:.0%} | {depth_str} | {question} |")
            lines.append("")

        # Embed depth images per VLM event
        depth_events = [ve for ve in vlm_events if ve.get("depth")]
        if depth_events:
            lines.append("## Depth Maps")
            lines.append("")
            session_dir = log_path.parent
            for i, ve in enumerate(depth_events, 1):
                ts = ve.get("timestamp", "")
                if "T" in ts:
                    ts = ts.split("T")[1][:8]
                depth_info = ve["depth"]
                for key, label in (("object_1", "Object 1"), ("object_2", "Object 2")):
                    d = depth_info.get(key)
                    if not d:
                        continue
                    lines.append(f"### Trigger {i} — {label} ({ts}) — {d['depth_at_gaze_m']:.2f}m at fixation")
                    lines.append("")
                    img_path = d.get("image")
                    if img_path:
                        # Make path relative to session dir
                        try:
                            rel = Path(img_path).relative_to(session_dir)
                        except ValueError:
                            rel = Path(img_path)
                        lines.append(f"![{label} depth]({rel})")
                        lines.append("")
                    # 3D waypoints table
                    wps = d.get("waypoints", [])
                    hit_label = d.get("gaze_hit_label", "")
                    if wps:
                        lines.append(f"#### 3D Waypoints")
                        lines.append("")
                        lines.append("| | Label | X (m) | Y (m) | Z (m) | Depth (m) | Pixel |")
                        lines.append("|---|-------|-------|-------|-------|-----------|-------|")
                        for wp in wps:
                            pos = wp["position_cam"]
                            px = wp["pixel_center"]
                            is_hit = wp["label"] == hit_label
                            marker = "**>>>**" if is_hit else ""
                            if is_hit:
                                lines.append(
                                    f"| {marker} | **{wp['label']}** | **{pos[0]:.3f}** | **{pos[1]:.3f}** | **{pos[2]:.3f}** "
                                    f"| **{wp['depth_median_m']:.3f}** | **({px[0]:.0f}, {px[1]:.0f})** |"
                                )
                            else:
                                lines.append(
                                    f"| {marker} | {wp['label']} | {pos[0]:.3f} | {pos[1]:.3f} | {pos[2]:.3f} "
                                    f"| {wp['depth_median_m']:.3f} | ({px[0]:.0f}, {px[1]:.0f}) |"
                                )
                        lines.append("")
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("# Detailed Log")
        lines.append("")

        summary = "\n".join(lines)

        # Prepend summary to existing log content
        existing = ""
        if log_path.exists():
            existing = log_path.read_text()
        log_path.write_text(summary + existing)
        print(f"[session] Log saved to {log_path}", file=sys.stderr)

    # ── calibration ──────────────────────────────────────────────────────────

    def _start_calibration(self) -> None:
        """Begin calibration recording (s key)."""
        from tools.calibration_check import CalibrationErrorTracker
        from perception.apriltag_detector import AprilTagDetector

        K = self.recorder.camera_matrix
        dist = getattr(self.recorder, "distortion_coeffs", None)
        self._calib_tracker = CalibrationErrorTracker(K, distortion_coeffs=dist)

        if self._apriltag_detector is None:
            camera_params = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
            self._apriltag_detector = AprilTagDetector(camera_params=camera_params)

        self._calib_mode = True
        self._calib_calibrated = False
        self._calib_last_frame = None
        self._vlm_state = "IDLE"
        self._first_object = None
        self.fix_detector.reset()
        self._calib_run_count += 1
        self._log_session_event({"type": "calibration_start"})

        # Log to session VLM markdown
        if self._session_vlm_log_path:
            try:
                with open(self._session_vlm_log_path, "a") as f:
                    f.write(f"\n\n{'='*60}\n")
                    f.write(f"# Calibration Run #{self._calib_run_count} — Started\n")
                    f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Frame: {self._current_frame_idx}\n")
            except Exception:
                pass

        print("[calibration] Recording started — look at the AprilTag", file=sys.stderr)

    def _stop_calibration(self) -> None:
        """Stop calibration recording and apply offset (e key)."""
        if self._calib_tracker is None or self._calib_tracker.n_samples == 0:
            print("[calibration] No samples recorded — calibration cancelled", file=sys.stderr)
            self._calib_mode = False
            return

        self._calib_mode = False
        self._calib_calibrated = True
        dx, dy = self._calib_tracker.pixel_offset

        # Save calibration file
        self._calib_output_path.parent.mkdir(parents=True, exist_ok=True)
        self._calib_tracker.save_calibration(self._calib_output_path)

        # Hot-swap offset into reader
        self.recorder.gaze_offset = (dx, dy)

        # Reset fixation so user starts fresh with corrected gaze
        self.fix_detector.reset()

        self._log_session_event({
            "type": "calibration_end",
            "n_samples": self._calib_tracker.n_samples,
            "offset_px": [dx, dy],
            "mean_error_before": self._calib_tracker.mean_error,
            "mean_error_after": float(np.mean([
                self._calib_tracker.corrected_error(
                    s.gaze_px[0], s.gaze_px[1],
                    s.tag_center_px[0], s.tag_center_px[1],
                )
                for s in self._calib_tracker.samples
            ])),
        })

        # Save annotated snapshot showing error on AprilTag
        snapshot_name = None
        if self._calib_last_frame is not None and self._calib_tag_detection is not None:
            snapshot_name = self._save_calib_snapshot(dx, dy)

        # Log to session VLM markdown
        if self._session_vlm_log_path:
            try:
                tracker = self._calib_tracker
                corr_mean = float(np.mean([
                    tracker.corrected_error(
                        s.gaze_px[0], s.gaze_px[1],
                        s.tag_center_px[0], s.tag_center_px[1],
                    )
                    for s in tracker.samples
                ]))
                with open(self._session_vlm_log_path, "a") as f:
                    f.write(f"\n## Calibration Run #{self._calib_run_count} — Complete\n")
                    f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(f"| Metric | Value |\n")
                    f.write(f"|--------|-------|\n")
                    f.write(f"| Samples | {tracker.n_samples} |\n")
                    f.write(f"| Raw mean error | {tracker.mean_error:.2f} deg |\n")
                    f.write(f"| Raw std | {tracker.std_error:.2f} deg |\n")
                    f.write(f"| Raw min / max | {tracker.min_error:.2f} / {tracker.max_error:.2f} deg |\n")
                    f.write(f"| Corrected mean | {corr_mean:.2f} deg |\n")
                    f.write(f"| Pixel offset | dx={dx:.1f}, dy={dy:.1f} px |\n\n")
                    if snapshot_name:
                        f.write(f"### Calibration Snapshot\n")
                        f.write(f"![calibration snapshot]({snapshot_name})\n\n")
            except Exception:
                pass

        print(
            f"[calibration] Done — offset: dx={dx:.1f} dy={dy:.1f} px, "
            f"{self._calib_tracker.n_samples} samples. "
            f"Saved to {self._calib_output_path}",
            file=sys.stderr,
        )

    def _save_calib_snapshot(self, dx: float, dy: float) -> str | None:
        """Render and save a calibration snapshot showing error angle on AprilTag.

        Returns the filename (relative to session log dir) or None on failure.
        """
        if self._calib_last_frame is None or self._calib_tag_detection is None:
            return None
        if self._session_vlm_log_path is None:
            return None

        try:
            img = self._calib_last_frame.copy()
            tag = self._calib_tag_detection
            tracker = self._calib_tracker
            raw_gx, raw_gy = self._calib_last_gaze

            # Draw AprilTag outline
            corners = tag.corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [corners], True, (0, 255, 100), 3, cv2.LINE_AA)

            # Tag center marker
            tcx, tcy = int(round(tag.center[0])), int(round(tag.center[1]))
            cv2.drawMarker(img, (tcx, tcy), (255, 220, 0), cv2.MARKER_CROSS, 30, 3, cv2.LINE_AA)
            cv2.putText(img, "TAG", (tcx + 20, tcy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2, cv2.LINE_AA)

            # Raw gaze point (red)
            rgx, rgy = int(round(raw_gx)), int(round(raw_gy))
            cv2.drawMarker(img, (rgx, rgy), (0, 0, 255), cv2.MARKER_CROSS, 30, 3, cv2.LINE_AA)
            cv2.circle(img, (rgx, rgy), 20, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(img, "RAW GAZE", (rgx + 20, rgy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            # Corrected gaze point (green)
            cgx, cgy = int(round(raw_gx - dx)), int(round(raw_gy - dy))
            cv2.drawMarker(img, (cgx, cgy), (0, 220, 80), cv2.MARKER_CROSS, 30, 3, cv2.LINE_AA)
            cv2.circle(img, (cgx, cgy), 20, (0, 220, 80), 2, cv2.LINE_AA)
            cv2.putText(img, "CORRECTED", (cgx + 20, cgy + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 80), 2, cv2.LINE_AA)

            # Error line: raw gaze → tag center (red)
            cv2.line(img, (rgx, rgy), (tcx, tcy), (0, 0, 255), 2, cv2.LINE_AA)
            mx1, my1 = (rgx + tcx) // 2, (rgy + tcy) // 2
            raw_err = tracker.mean_error
            cv2.putText(img, f"raw: {raw_err:.2f} deg", (mx1 + 10, my1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            # Error line: corrected gaze → tag center (green)
            cv2.line(img, (cgx, cgy), (tcx, tcy), (0, 220, 80), 2, cv2.LINE_AA)
            mx2, my2 = (cgx + tcx) // 2, (cgy + tcy) // 2
            corr_err = tracker.corrected_error(raw_gx, raw_gy, tag.center[0], tag.center[1])
            cv2.putText(img, f"corr: {corr_err:.2f} deg", (mx2 + 10, my2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 80), 2, cv2.LINE_AA)

            # Offset arrow from raw to corrected
            cv2.arrowedLine(img, (rgx, rgy), (cgx, cgy), (255, 255, 0), 2, cv2.LINE_AA, tipLength=0.15)

            # Stats box (top-left)
            overlay = img.copy()
            cv2.rectangle(overlay, (10, 10), (520, 160), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
            y0 = 36
            cv2.putText(img, f"Calibration Run #{self._calib_run_count}", (20, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 50), 2, cv2.LINE_AA)
            cv2.putText(img, f"Samples: {tracker.n_samples}   Raw mean: {tracker.mean_error:.2f} deg",
                        (20, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(img, f"Corrected mean: {float(np.mean([tracker.corrected_error(s.gaze_px[0], s.gaze_px[1], s.tag_center_px[0], s.tag_center_px[1]) for s in tracker.samples])):.2f} deg",
                        (20, y0 + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 80), 1, cv2.LINE_AA)
            cv2.putText(img, f"Offset: dx={dx:.1f}  dy={dy:.1f} px",
                        (20, y0 + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 1, cv2.LINE_AA)
            cv2.putText(img, f"Min: {tracker.min_error:.2f}  Max: {tracker.max_error:.2f}  P95: {tracker.p95_error:.2f} deg",
                        (20, y0 + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (140, 140, 140), 1, cv2.LINE_AA)

            # Save
            snap_dir = self._session_vlm_log_path.parent
            snap_name = f"calib_{self._calib_run_count}_{self._current_frame_idx}.jpg"
            snap_path = snap_dir / snap_name
            cv2.imwrite(str(snap_path), img)
            print(f"[calibration] Snapshot saved: {snap_path}", file=sys.stderr)
            return snap_name

        except Exception as e:
            print(f"[calibration] Failed to save snapshot: {e}", file=sys.stderr)
            return None

    def _process_calibration(self, frame: np.ndarray, gaze: GazeSample) -> None:
        """Per-frame calibration: detect tag, accumulate samples."""
        if self._apriltag_detector is None:
            return

        tags = self._apriltag_detector.detect(frame)
        self._calib_tag_detection = tags[0] if tags else None
        self._calib_sample = None

        if self._calib_tag_detection is not None and self._calib_mode:
            tag = self._calib_tag_detection
            tag_center = (float(tag.center[0]), float(tag.center[1]))
            # Use raw gaze (add back the current offset to get raw)
            offset = self.recorder.gaze_offset
            raw_gx = gaze.x + offset[0]
            raw_gy = gaze.y + offset[1]
            # Keep last frame with tag for snapshot
            self._calib_last_frame = frame.copy()
            self._calib_last_gaze = (raw_gx, raw_gy)
            self._calib_sample = self._calib_tracker.update(
                frame_idx=self._current_frame_idx,
                timestamp_ns=gaze.timestamp_ns,
                gaze_x=raw_gx,
                gaze_y=raw_gy,
                tag_center=tag_center,
                tag_id=tag.tag_id,
            )
            self._log_session_event({
                "type": "calibration_sample",
                "gaze_px": [raw_gx, raw_gy],
                "tag_center_px": list(tag_center),
                "error_deg": self._calib_sample.error_deg,
                "tag_id": tag.tag_id,
            })

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self) -> None:
        if self.show_overlay:
            cv2.namedWindow("Gaze Intent", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Gaze Intent", 1030, 600)
        try:
            for bundle in self.recorder:
                self._process(bundle)
        finally:
            if self._video_writer:
                self._video_writer.release()
                print(f"[video] saved to {self._out_video_path}", file=sys.stderr)
            if self.show_overlay or self._renderer is not None:
                cv2.destroyAllWindows()
            if self._say_proc is not None:
                self._say_proc.wait()
            if self._log_file:
                self._log_file.close()
            # Session summary
            self._log_session_event({"type": "session_end"})
            self._write_session_summary()
            self.reasoner.shutdown()
            if hasattr(self.recorder, "close"):
                self.recorder.close()
            if self._out_video_path and self._tts_threads:
                for t in self._tts_threads:
                    t.join(timeout=10)
                if self._audio_cues:
                    self._mux_audio()

    # ── per-frame ─────────────────────────────────────────────────────────────

    def _process(self, bundle: FrameBundle) -> None:
        self._current_frame_idx = bundle.video.frame_idx

        # ── check for completed async result ──────────────────────────────
        if self._pending_future is not None and self._pending_future.done():
            decision = self._pending_future.result()
            # Attach depth info if available
            if getattr(self, "_pending_depth_info", None) is not None:
                decision["depth"] = self._pending_depth_info
                self._pending_depth_info = None
            self._emit(decision)
            self._pending_future = None
            self._last_decision = decision
            self._vlm_state = "IDLE"  # cycle complete, ready for next sequence
            self.fix_detector.reset()  # reset fixation so user starts fresh

        # ── gate on worn ──────────────────────────────────────────────────
        if not bundle.worn:
            self.fix_detector.reset()
            self._vlm_state = "IDLE"
            self._first_object = None
            return

        frame = bundle.video.bgr
        gaze = bundle.gaze

        # ── calibration per-frame (if active or calibrated) ──────────────
        if self._calib_mode or self._apriltag_detector is not None:
            self._process_calibration(frame, gaze)

        # ── 3D gaze grounding (optional) ──────────────────────────────────
        if self.gaze_grounder is not None:
            self.gaze_grounder.update_imu(bundle.imu)
            g3d = self.gaze_grounder.ground(gaze.x, gaze.y, frame)
            self._last_gaze_3d = g3d
            if self.debug and g3d.valid:
                p = g3d.point_world
                print(
                    f"[DEBUG] gaze3d=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})m "
                    f"depth={g3d.depth:.3f}m rs_px=({g3d.pixel_rs[0]:.0f},{g3d.pixel_rs[1]:.0f})",
                    file=sys.stderr,
                )
            elif self.debug and g3d.error:
                print(f"[DEBUG] gaze3d: {g3d.error}", file=sys.stderr)

        # ── fixation update (<1 ms) ───────────────────────────────────────
        fixation = self.fix_detector.update(gaze)

        # ── object detection (sync, ~10-30 ms) ───────────────────────────
        detections = self.detector.detect(frame)
        _sigma = fixation.max_drift_px if fixation.active else 40.0
        gaze_hit = ObjectDetector.probabilistic_hit_test(
            detections, gaze.x, gaze.y,
            sigma=_sigma, frame_shape=frame.shape[:2],
        )
        hit = gaze_hit.detection if gaze_hit is not None else None
        _hit_prob = gaze_hit.probability if gaze_hit is not None else 0.0

        if self.debug:
            fix_ms = fixation.duration_ns / 1_000_000 if fixation.active else 0
            hit_info = f"hit={hit.label if hit else '—'} P={_hit_prob:.0%}"
            if gaze_hit is not None and len(gaze_hit.candidates) > 1:
                top3 = " ".join(
                    f"{d.label}:{p:.0%}" for d, p in gaze_hit.candidates[:3]
                )
                hit_info += f" [{top3}]"
            print(
                f"[DEBUG] frame={bundle.video.frame_idx:4d} "
                f"gaze=({gaze.x:.0f},{gaze.y:.0f}) "
                f"fix_active={fixation.active} stable={fixation.is_stable} "
                f"dur={fix_ms:.0f}ms σ={_sigma:.0f}px "
                f"{hit_info} "
                f"pending={self._pending_future is not None}",
                file=sys.stderr,
            )

        # ── optional overlay ──────────────────────────────────────────────
        if self._renderer is not None:
            countdown = 0.0
            if self._vlm_state == "AWAITING_SECOND" and self._first_object is not None:
                countdown = max(
                    0.0,
                    SECOND_OBJECT_TIMEOUT_S
                    - (time.monotonic() - self._first_object_time),
                )
            # Find matching waypoint for current hit from cached waypoints
            _hit_wp = None
            if hit is not None and self._last_waypoints:
                for wp in self._last_waypoints:
                    if wp["label"] == hit.label:
                        _hit_wp = wp
                        break
            canvas = self._renderer.render(
                frame,
                detections,
                hit,
                fixation,
                gaze,
                self._vlm_state,
                self._last_decision,
                self._api_mode,
                self._pending_detection,
                bundle.video.frame_idx,
                self.recorder.n_frames,
                (
                    self.recorder.n_frames / self.recorder.fps
                    if self.recorder.n_frames > 0
                    else bundle.video.frame_idx / self.recorder.fps
                ),
                first_object=self._first_object,
                second_object_countdown=countdown,
                hit_probability=_hit_prob,
                calib_mode=self._calib_mode,
                calib_tracker=self._calib_tracker,
                calib_tag_detection=self._calib_tag_detection,
                calib_sample=self._calib_sample,
                calib_calibrated=self._calib_calibrated,
                hit_waypoint=_hit_wp,
            )
            if self._out_video_path:
                if self._video_writer is None:
                    h, w = canvas.shape[:2]
                    # Round up to multiple of 16 for codec compatibility
                    w_aligned = (w + 15) & ~15
                    h_aligned = (h + 15) & ~15
                    self._video_writer = cv2.VideoWriter(
                        str(self._out_video_path),
                        self._out_video_fourcc,
                        self._out_video_fps,
                        (w_aligned, h_aligned),
                    )
                    self._vw_size = (w_aligned, h_aligned)
                # Pad frame if needed to match aligned size
                h, w = canvas.shape[:2]
                if (w, h) != self._vw_size:
                    padded = np.zeros(
                        (self._vw_size[1], self._vw_size[0], 3), dtype=np.uint8
                    )
                    padded[:h, :w] = canvas
                    canvas = padded
                self._video_writer.write(canvas)
            if self.show_overlay:
                display = cv2.resize(canvas, (1030, 600), interpolation=cv2.INTER_AREA)
                cv2.imshow("Gaze Intent", display)
                key = cv2.waitKey(1) & 0xFF
                try:
                    prop = cv2.getWindowProperty("Gaze Intent", cv2.WND_PROP_AUTOSIZE)
                except cv2.error:
                    prop = -1
                if key == ord("q") or prop < 0:
                    raise KeyboardInterrupt
                if key == ord("d") and self._renderer is not None:
                    self._renderer.caregiver_mode = not self._renderer.caregiver_mode
                    mode_name = (
                        "CAREGIVER" if self._renderer.caregiver_mode else "PATIENT"
                    )
                    print(f"[overlay] Switched to {mode_name} mode", file=sys.stderr)
                if key == ord("s") and not self._calib_mode:
                    self._start_calibration()
                elif key == ord("e") and self._calib_mode:
                    self._stop_calibration()

        # ── skip VLM while calibrating ───────────────────────────────────
        if self._calib_mode:
            return

        # ── state machine: gaze sequence ─────────────────────────────────
        if self._pending_future is not None and not self._pending_future.done():
            return  # debounce — still waiting for VLM

        if self._vlm_state == "IDLE":
            self._handle_idle(fixation, hit, frame, gaze, detections, gaze_hit)
        elif self._vlm_state == "AWAITING_SECOND":
            self._handle_awaiting_second(fixation, hit, frame, gaze, detections, gaze_hit)

    # ── state handlers ────────────────────────────────────────────────────────

    @staticmethod
    def _format_gaze_hit(gaze_hit: Optional[GazeHit], sigma: float) -> str:
        if gaze_hit is None:
            return ""
        lines = [f"sigma={sigma:.1f}px"]
        for det, prob in gaze_hit.candidates:
            lines.append(f"  {det.label}: P={prob:.3f} ({prob:.0%})  bbox={det.box_xyxy}")
        return "\n".join(lines)

    def _handle_idle(
        self,
        fixation: FixationState,
        hit: Optional[Detection],
        frame: np.ndarray,
        gaze: GazeSample,
        detections: list[Detection] = [],
        gaze_hit: Optional[GazeHit] = None,
    ) -> None:
        """IDLE → wait for first stable fixation on a segment."""
        if not fixation.is_stable:
            return
        if fixation.duration_ns < TRIGGER_MIN_DURATION_NS:
            return
        if hit is None:
            return  # need a segment for multi-object sequence

        now_ts = fixation.start_ts_ns
        if not self._should_trigger(hit.box_center, now_ts):
            return

        # Lock first object
        self._first_object = hit
        self._first_object_frame = frame.copy()
        self._first_object_time = time.monotonic()
        self._first_object_fixation = fixation
        self._first_object_gaze = (gaze.x, gaze.y)
        self._first_object_detections = detections
        # Run depth on first object immediately
        depth_result = self._run_depth(frame, gaze.x, gaze.y)
        if depth_result is not None:
            depth_stats, depth_map = depth_result
            K = getattr(self.recorder, "camera_matrix", None)
            if K is not None:
                wps = compute_3d_waypoints(detections, depth_map, K)
                depth_stats["waypoints"] = [
                    {"label": wp.label, "position_cam": list(wp.position_cam),
                     "depth_median_m": wp.depth_median_m,
                     "pixel_center": list(wp.pixel_center)}
                    for wp in wps
                ]
                self._last_waypoints = depth_stats["waypoints"]
                print(f"[depth] {len(wps)} 3D waypoints computed", file=sys.stderr)
            depth_stats["gaze_hit_label"] = hit.label
            self._annotate_depth_image(depth_stats, detections, hit.label)
            self._first_object_depth = depth_stats
        else:
            self._first_object_depth = None
        _sigma = fixation.max_drift_px if fixation.active else 40.0
        self._first_object_hit_info = self._format_gaze_hit(gaze_hit, _sigma)
        self._vlm_state = "AWAITING_SECOND"
        self._pending_detection = hit
        self._awaiting_tts_finished = True  # timer resets after TTS completes
        self._speak(
            "I see you're looking at something. "
            "Look at another object to tell me what to do with it."
        )
        print(
            f"[sequence] First object locked at ({gaze.x:.0f},{gaze.y:.0f}). Awaiting second...",
            file=sys.stderr,
        )

    def _handle_awaiting_second(
        self,
        fixation: FixationState,
        hit: Optional[Detection],
        frame: np.ndarray,
        gaze: GazeSample,
        detections: list[Detection] = [],
        gaze_hit: Optional[GazeHit] = None,
    ) -> None:
        """AWAITING_SECOND → wait for second object or timeout."""
        # Don't start countdown until TTS finishes speaking
        if not self._tts_done.is_set():
            return
        # Reset timer once on the first frame after TTS completes
        if self._awaiting_tts_finished:
            self._first_object_time = time.monotonic()
            self._awaiting_tts_finished = False

        elapsed = time.monotonic() - self._first_object_time

        # Check timeout → single-object fallback
        if elapsed >= SECOND_OBJECT_TIMEOUT_S:
            print(
                f"[sequence] Timeout waiting for second object. "
                f"Falling back to single-object reasoning.",
                file=sys.stderr,
            )
            self._api_mode = "VISION"
            self._vlm_state = "THINKING"
            self._last_trigger_ts_ns = self._first_object_fixation.start_ts_ns
            self._last_object_center = self._first_object.box_center
            # Use depth already computed when first object was locked
            self._pending_depth_info = {"object_1": self._first_object_depth}
            self._pending_future = self.reasoner.reason_async(
                GazeSample(
                    timestamp_ns=self._first_object_fixation.start_ts_ns,
                    x=self._first_object_gaze[0],
                    y=self._first_object_gaze[1],
                    eye_state=gaze.eye_state,
                ),
                self._first_object_fixation,
                self._first_object_frame,
                self._first_object_detections,
                gaze_hit_info=f"Object 1:\n{self._first_object_hit_info}",
            )
            self._first_object = None
            return

        # Check for second object fixation
        if not fixation.is_stable:
            return
        if fixation.duration_ns < TRIGGER_MIN_DURATION_NS:
            return
        if hit is None:
            return
        # Same segment check: if centroids are within 50px, it's the same object
        import math
        dist = math.hypot(
            hit.box_center[0] - self._first_object.box_center[0],
            hit.box_center[1] - self._first_object.box_center[1],
        )
        if dist < 50:
            return  # same region — keep waiting for a different one

        # Fire two-object reasoning
        second_gaze = (gaze.x, gaze.y)
        print(
            f"[sequence] Second object locked at ({gaze.x:.0f},{gaze.y:.0f}). "
            f"Firing VLM with pair.",
            file=sys.stderr,
        )
        self._api_mode = "PAIR"
        self._vlm_state = "THINKING"
        self._last_trigger_ts_ns = fixation.start_ts_ns
        self._last_object_center = hit.box_center
        # Depth for both objects (first was computed at lock time)
        second_depth_result = self._run_depth(frame, gaze.x, gaze.y)
        second_depth = None
        if second_depth_result is not None:
            second_depth, depth_map2 = second_depth_result
            K = getattr(self.recorder, "camera_matrix", None)
            if K is not None:
                wps = compute_3d_waypoints(detections, depth_map2, K)
                second_depth["waypoints"] = [
                    {"label": wp.label, "position_cam": list(wp.position_cam),
                     "depth_median_m": wp.depth_median_m,
                     "pixel_center": list(wp.pixel_center)}
                    for wp in wps
                ]
                self._last_waypoints = second_depth["waypoints"]
                print(f"[depth] {len(wps)} 3D waypoints computed (object 2)", file=sys.stderr)
            second_depth["gaze_hit_label"] = hit.label
            self._annotate_depth_image(second_depth, detections, hit.label)
        self._pending_depth_info = {
            "object_1": self._first_object_depth,
            "object_2": second_depth,
        }
        _sigma2 = fixation.max_drift_px if fixation.active else 40.0
        second_hit_info = self._format_gaze_hit(gaze_hit, _sigma2)
        combined_info = (
            f"Object 1:\n{self._first_object_hit_info}\n\n"
            f"Object 2:\n{second_hit_info}"
        )
        self._pending_future = self.reasoner.reason_async_pair(
            self._first_object_gaze,
            second_gaze,
            self._first_object_fixation,
            fixation,
            self._first_object_frame,
            frame,
            self._first_object_detections,
            detections,
            gaze_hit_info=combined_info,
        )
        self._first_object = None

    def _run_depth(
        self, frame: np.ndarray, gaze_x: float, gaze_y: float,
    ) -> Optional[tuple[dict, np.ndarray]]:
        """Run depth estimation on a frame at fixation trigger time.

        Returns (stats_dict, depth_map) where stats_dict has depth_at_gaze
        (metres), stats, and saved image path; depth_map is (H,W) float32 in
        metres.  Returns None if depth estimator is not configured.
        """
        if self.depth_estimator is None:
            return None
        try:
            depth_map, saved_path = self.depth_estimator.estimate(
                frame, f_px=self._depth_f_px, gaze_xy=(gaze_x, gaze_y),
            )
            h, w = depth_map.shape
            gx, gy = int(round(gaze_x)), int(round(gaze_y))
            gx = max(0, min(gx, w - 1))
            gy = max(0, min(gy, h - 1))
            depth_at_gaze = float(depth_map[gy, gx])
            result = {
                "depth_at_gaze_m": round(depth_at_gaze, 3),
                "depth_min_m": round(float(depth_map.min()), 3),
                "depth_max_m": round(float(depth_map.max()), 3),
                "depth_mean_m": round(float(depth_map.mean()), 3),
            }
            if saved_path:
                result["image"] = saved_path
            print(
                f"[depth] gaze=({gx},{gy}) → {depth_at_gaze:.3f}m "
                f"(range {result['depth_min_m']:.2f}–{result['depth_max_m']:.2f}m)",
                file=sys.stderr,
            )
            return result, depth_map
        except Exception as e:
            print(f"[depth] ERROR: {e}", file=sys.stderr)
            return None

    @staticmethod
    def _annotate_depth_image(
        depth_stats: dict,
        detections: list[Detection],
        gaze_hit_label: str,
    ) -> None:
        """Draw segment outlines and 3D waypoint labels on the saved depth PNG.

        The gaze-hit segment is highlighted with a thicker, green outline.
        All others get a thin white outline.
        """
        img_path = depth_stats.get("image")
        if not img_path:
            return
        waypoints = depth_stats.get("waypoints", [])
        if not waypoints:
            return

        img = cv2.imread(img_path)
        if img is None:
            return

        h, w = img.shape[:2]
        # Build label→waypoint lookup
        wp_by_label = {wp["label"]: wp for wp in waypoints}
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.35, h / 2400)
        thin = max(1, int(h / 1500))

        for det in detections:
            if det.mask_polygon is None or len(det.mask_polygon) < 3:
                continue
            is_hit = det.label == gaze_hit_label
            color = (0, 255, 0) if is_hit else (255, 255, 255)
            thickness = thin + 2 if is_hit else thin
            cv2.polylines(img, [det.mask_polygon], True, color, thickness, cv2.LINE_AA)

            # Only label the gaze-hit segment
            if not is_hit:
                continue
            wp = wp_by_label.get(det.label)
            if wp is None:
                continue
            pos = wp["position_cam"]
            lbl = f"{det.label}: ({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:.2f})m"
            px = int(round(wp["pixel_center"][0]))
            py = int(round(wp["pixel_center"][1]))
            lbl_scale = font_scale * 1.2
            lbl_thick = thin + 1
            (tw, th_t), baseline = cv2.getTextSize(lbl, font, lbl_scale, lbl_thick)
            lx = max(0, min(px - tw // 2, w - tw - 4))
            ly = max(th_t + 4, min(py - 6, h - 4))
            cv2.rectangle(img, (lx - 2, ly - th_t - 3), (lx + tw + 3, ly + baseline + 2),
                          (0, 0, 0), cv2.FILLED)
            cv2.putText(img, lbl, (lx, ly), font, lbl_scale, color, lbl_thick, cv2.LINE_AA)

        cv2.imwrite(img_path, img)

    def _should_trigger(self, obj_center: tuple[float, float], now_ts: int) -> bool:
        if self._last_trigger_ts_ns == 0:
            return True  # first ever trigger

        elapsed_ns = now_ts - self._last_trigger_ts_ns

        # Position-based cooldown: same region if within 50px
        if self._last_object_center is not None:
            import math
            dist = math.hypot(
                obj_center[0] - self._last_object_center[0],
                obj_center[1] - self._last_object_center[1],
            )
            if dist < 50:
                return elapsed_ns >= COOLDOWN_SAME_OBJECT_NS

        return elapsed_ns >= COOLDOWN_NEW_OBJECT_NS

    def _emit(self, decision: dict) -> None:
        line = json.dumps(decision)
        # Show top candidate as prefix
        candidates = decision.get("candidates", [])
        if candidates:
            top = candidates[0]
            print(f"[TOP] {top.get('intent', '?')} ({top.get('confidence', 0):.0%})", flush=True)
        print(line, flush=True)

        if self._log_file:
            self._log_file.write(line + "\n")
            self._log_file.flush()

        # Session log
        self._log_session_event({"type": "vlm_decision", **decision})

        q = decision.get("clarification_question")
        if q:
            self._speak(q)

    _PIPER_MODEL = Path.home() / ".local/share/piper-voices/en_US-lessac-medium.onnx"

    def _speak(self, text: str) -> None:
        """Non-blocking TTS via Piper Python API. Plays live + saves WAV for video mux."""

        self._tts_done.clear()  # mark TTS as in-progress

        def _tts_worker(msg: str) -> None:
            try:
                import subprocess
                import wave
                import tempfile
                from piper import PiperVoice

                voice = PiperVoice.load(str(self._PIPER_MODEL))
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    wav_path = f.name
                with wave.open(wav_path, "wb") as wav_file:
                    voice.synthesize_wav(msg, wav_file)

                # Save cue for video audio muxing
                if self._out_video_path:
                    ts = self._current_frame_idx / self.recorder.fps
                    self._audio_cues.append((ts, wav_path))
                    # Play live, but don't delete the file yet
                    subprocess.run(
                        ["aplay", wav_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
                else:
                    subprocess.run(
                        ["aplay", wav_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
                    Path(wav_path).unlink(missing_ok=True)
            except Exception as e:
                print(f"[TTS] error: {e}", file=sys.stderr)
            finally:
                self._tts_done.set()  # mark TTS as finished

        t = threading.Thread(target=_tts_worker, args=(text,), daemon=True)
        t.start()
        self._tts_threads.append(t)

    def _mux_audio(self) -> None:
        """Mix saved TTS cues into the output video using ffmpeg."""
        import os
        import shutil
        import subprocess

        # Augment PATH so ffmpeg is found even when conda env omits the base dir
        search_path = (
            os.environ.get("PATH", "")
            + ":/opt/miniconda3/bin:/opt/homebrew/bin:/usr/local/bin"
        )
        ffmpeg_bin = shutil.which("ffmpeg", path=search_path)
        if not ffmpeg_bin:
            print("[audio] ffmpeg not found — skipping audio mux", file=sys.stderr)
            return

        src = self._out_video_path
        tmp = src.with_suffix(".tmp.mp4")

        # Build filter: delay each cue to its video timestamp (adelay takes ms)
        filter_parts = []
        for i, (ts, _) in enumerate(self._audio_cues):
            delay_ms = int(ts * 1000)
            filter_parts.append(f"[{i + 1}:a]adelay={delay_ms}|{delay_ms}[a{i}]")
        mix_inputs = "".join(f"[a{i}]" for i in range(len(self._audio_cues)))
        filter_parts.append(
            f"{mix_inputs}amix=inputs={len(self._audio_cues)}:duration=longest[aout]"
        )

        cmd = [ffmpeg_bin, "-y", "-i", str(src)]
        for _, aiff in self._audio_cues:
            cmd += ["-i", aiff]
        cmd += [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "0:v",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-t",
            str(self._current_frame_idx / self.recorder.fps),
            str(tmp),
        ]

        result = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if result.returncode == 0:
            tmp.replace(src)
            print(
                f"[audio] muxed {len(self._audio_cues)} cue(s) into {src}",
                file=sys.stderr,
            )
        else:
            tmp.unlink(missing_ok=True)
            print(
                "[audio] ffmpeg mux failed — video saved without audio", file=sys.stderr
            )

        # Clean up temp WAV files
        for _, wav_path in self._audio_cues:
            Path(wav_path).unlink(missing_ok=True)
