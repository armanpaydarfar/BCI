# Utils/gaze/gaze_system.py
"""
gaze_system.py

Facade class that runs:
- Neon threads (video, gaze, imu)
- YOLO detector thread with governor
- Optional SORT tracker
- Computes:
    - gaze px + smoothing
    - gaze-on-object hit + hold
    - head yaw/pitch (IMU quat)
    - gaze yaw/pitch (optical axes -> scene->imu -> world)
    - vergence depth (depth cm, miss mm, ipd mm)
- Optional UI preview (can be kept OFF for experiments)

Design goals:
- Safe to run headless (no cv2 windows)
- A single import surface for the rest of your codebase:
    from Utils.gaze.gaze_system import GazeSystem, GazeConfig
- No globals: everything is instance state
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
import threading
import time
import os

import numpy as np

from pupil_labs.realtime_api.simple import discover_one_device

from Utils.gaze.gaze_math import (
    EWMASmoother,
    cartesian_to_spherical_world,
    gaze_ray_from_optical_axes,
    quat_to_rotmat_wxyz,
    safe_unit,
    scene_to_imu_matrix,
    vergence_depth_from_eyestate,
    wrap_deg_180,
)
from Utils.gaze.gaze_tracking import (
    SimpleSORTTracker,
    TrackerParams,
    gaze_object_hit,
)

# NOTE: cv2 is intentionally NOT imported here unless display is enabled.
#       This keeps headless mode lightweight and avoids GUI side-effects.


# -----------------------
# Config
# -----------------------
@dataclass
class GazeConfig:
    # Suppression / utility mode
    enable_prints: bool = True
    enable_display: bool = False     # if False -> no OpenCV windows / UI thread
    show_video: bool = False         # if False and enable_display=True -> black canvas HUD (optional later)

    # CV / Object recognition (headless-friendly)
    enable_cv: bool = True           # if True -> video thread runs (for YOLO), detector thread starts

    # Loop pacing
    target_loop_hz: float = 20.0

    # Detector base settings (governor decides if it runs)
    det_hz: float = 3.0
    det_conf: float = 0.50
    det_iou: float = 0.20
    det_max_det: int = 15
    model_name: str = "yolo26n.pt"
    detect_resize_width: Optional[int] = None
    det_classes: Optional[List[int]] = field(default_factory=lambda: [0, 63, 64, 67, 39, 41])

    # Governor
    gov_vid_stale_disable_s: float = 0.20
    gov_loop_min_hz: float = 14.0
    gov_slow_infer_s: float = 0.30
    gov_cooldown_s: float = 0.35
    gov_reenable_stable_s: float = 0.50

    # IMU gating (enabled)
    enable_imu_gate: bool = True
    gov_imu_angvel_disable: float = 20.0
    imu_fresh_s: float = 0.25

    # Depth + angles
    miss_max_mm: float = 30.0
    denom_min: float = 1e-4
    depth_smooth_alpha: float = 0.15
    angle_alpha: float = 0.25

    # "Nose forward" axis in IMU LOCAL frame
    head_forward_local: Tuple[float, float, float] = (0.0, 1.0, 0.0)

    # Scene->IMU rotation about X (deg)
    imu_scene_rot_deg: float = -90.0 - 12.0
    apply_scene_to_imu_for_gaze: bool = True

    # Gaze smoothing + gaze-on-object
    gaze_smooth_alpha: float = 0.35
    gaze_hit_hold_sec: float = 1.0
    gaze_radius_px: float = 25.0
    nearest_fallback_px: float = 60.0
    gaze_recency_sec: float = 0.35

    # Tracking
    use_tracker: bool = True
    tracker_params: TrackerParams = field(default_factory=TrackerParams)

    # Console print rate
    print_hz: float = 5.0
    topk_log: int = 3

    # Device connection — leave empty to use mDNS auto-discovery (works on home/
    # hotspot networks but blocked on enterprise/IoT VLANs).  When set, connects
    # directly via Device(address=neon_host, port=8080), bypassing mDNS entirely.
    neon_host: str = ""
    discover_timeout_s: int = 10

    # Frame source toggle for the GPU-host migration plan (see SoftwareDocs/
    # projects/harmony-bci/gpu-service/architecture-plan.md §3.4). Default `local` opens the
    # Neon device directly. `remote` consumes envelopes from a TCP relay
    # (Utils/frame_relay.py) via Utils/remote_frame_reader.RemoteNeonDevice.
    # In remote mode gaze + IMU are subsampled to the relay rate (~10 Hz vs
    # native ~200 Hz); smoothing is therefore coarser.
    frame_source: str = "local"            # "local" | "remote"
    remote_frame_host: str = ""
    remote_frame_port: int = 5591


# -----------------------
# Utility
# -----------------------
def _hz_from_hist(hist: deque) -> float:
    if len(hist) < 5:
        return float("nan")
    m = float(np.mean(hist))
    return (1.0 / m) if m > 1e-9 else float("nan")


# -----------------------
# Gaze System
# -----------------------
class GazeSystem:
    def __init__(self, cfg: Optional[GazeConfig] = None):
        self.cfg = cfg or GazeConfig()

        # Thread stop
        self._stop_event = threading.Event()

        # Device
        self._device = None

        # Shared state: video
        self._video_lock = threading.Lock()
        self._video_buf_bgr = None
        self._video_buf_shape = None
        self._latest_video_wall_t = None

        # Shared state: gaze
        self._gaze_lock = threading.Lock()
        self._latest_gaze = None
        self._latest_gaze_wall_t = None

        # Shared state: frame for detector
        self._frame_lock = threading.Lock()
        self._latest_frame_bgr = None
        self._latest_frame_time_unix = None  # from gaze best-effort

        # Shared state: IMU
        self._imu_lock = threading.Lock()
        self._latest_imu_angvel = None  # float magnitude (rad/s)
        self._latest_imu_wall_t = None
        self._latest_imu = None         # full imu datum (for quaternion)

        # Shared state: detector output
        self._det_lock = threading.Lock()
        self._latest_dets: List[dict] = []
        self._latest_det_time = None
        self._latest_det_fps = 0.0
        self._latest_infer_s = float("nan")

        # Governor state
        self._gov_lock = threading.Lock()
        self._gov_enabled = True
        self._gov_reason = "init"
        self._gov_cooldown_until = 0.0
        self._gov_stable_since = None

        # Manual realtime CV toggle (for online interfaces to save compute)
        self._cv_user_enabled = True  # can be flipped at runtime via set_cv_enabled()

        # Internals
        self._threads: List[threading.Thread] = []

        # Tracker
        self._tracker = SimpleSORTTracker(self.cfg.tracker_params) if self.cfg.use_tracker else None
        self._last_processed_det_time = None

        # Smoothers
        self._depth_smoother = EWMASmoother(alpha=self.cfg.depth_smooth_alpha)
        self._head_yaw_smoother = EWMASmoother(alpha=self.cfg.angle_alpha)
        self._head_pitch_smoother = EWMASmoother(alpha=self.cfg.angle_alpha)
        self._gaze_yaw_smoother = EWMASmoother(alpha=self.cfg.angle_alpha)
        self._gaze_pitch_smoother = EWMASmoother(alpha=self.cfg.angle_alpha)

        # Recenter offsets
        self._have_recenter = False
        self._yaw_offset = 0.0
        self._pitch_offset = 0.0

        # Gaze smoothing
        self._gaze_smooth_x = None
        self._gaze_smooth_y = None

        # Gaze-hit hold
        self._last_hit = None
        self._last_hit_wall_t = -1e9

        # Rates
        self._loop_dt_hist = deque(maxlen=120)
        self._video_dt_hist = deque(maxlen=120)
        self._t_loop_prev = time.time()
        self._t_video_prev = None
        self._next_print_t = time.time()

        # Matrices / constants
        self._S2I = scene_to_imu_matrix(self.cfg.imu_scene_rot_deg)
        self._head_forward_local = np.array(self.cfg.head_forward_local, dtype=float)

        # Thread pools clamp (helps jitter / ultralytics)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("KMP_BLOCKTIME", "0")
        os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")

    # -----------------------
    # Logging
    # -----------------------
    def _log(self, *args, **kwargs):
        if self.cfg.enable_prints:
            print(*args, **kwargs)

    # -----------------------
    # Public API
    # -----------------------
    def start(self) -> None:
        """
        Discover device and start threads.
        """
        if self.cfg.frame_source == "remote":
            if not self.cfg.remote_frame_host:
                raise RuntimeError(
                    "frame_source='remote' requires remote_frame_host to be set"
                )
            # Substitute a TCP-relay-backed Device shim. The shim implements
            # receive_scene_video_frame / receive_gaze_datum / receive_imu_datum
            # so the three Neon threads below consume it unchanged.
            from Utils.remote_frame_reader import RemoteNeonDevice
            self._log(
                f"Connecting to frame relay tcp://"
                f"{self.cfg.remote_frame_host}:{self.cfg.remote_frame_port}…"
            )
            self._device = RemoteNeonDevice(
                self.cfg.remote_frame_host, int(self.cfg.remote_frame_port)
            )
        elif self.cfg.neon_host:
            from pupil_labs.realtime_api.simple import Device
            self._log(f"Connecting directly to Neon at {self.cfg.neon_host}…")
            self._device = Device(address=self.cfg.neon_host, port=8080)
        else:
            self._log("Looking for the next best device...")
            self._device = discover_one_device(max_search_duration_seconds=int(self.cfg.discover_timeout_s))
        if self._device is None:
            raise RuntimeError("No Pupil Labs Neon device found.")

        self._log(f"Connected to {self._device}.")

        # Start threads
        # IMPORTANT FIX:
        # - If enable_cv=True, we must run the video thread even when headless,
        #   otherwise YOLO never sees frames (src=None forever) and you get no objects/hits.
        if self.cfg.enable_cv or self.cfg.show_video or self.cfg.enable_display:
            t = threading.Thread(target=self._video_thread_fn, args=(self._device,), daemon=True)
            self._threads.append(t)
            t.start()

        t = threading.Thread(target=self._gaze_thread_fn, args=(self._device,), daemon=True)
        self._threads.append(t)
        t.start()

        t = threading.Thread(target=self._imu_thread_fn, args=(self._device,), daemon=True)
        self._threads.append(t)
        t.start()

        # Detector thread (YOLO) only if CV is enabled
        if self.cfg.enable_cv:
            t = threading.Thread(target=self._detector_thread_fn, daemon=True)
            self._threads.append(t)
            t.start()

        # Display/UI thread remains OFF here; we’ll add it when we do the control panel + preview mode.

    def stop(self) -> None:
        self._stop_event.set()
        try:
            if self._device is not None:
                self._device.close()
        except Exception:
            pass

    def set_cv_enabled(self, enabled: bool) -> None:
        """
        Realtime toggle for CV/OR (YOLO + object state) without tearing down the system.
        Use this from an online interface to save compute during certain phases.

        - When False: detector loop will idle and snapshots will show gov_enabled=False with reason 'cv_user_off'.
        - When True: normal governor behavior resumes.
        """
        with self._gov_lock:
            self._cv_user_enabled = bool(enabled)

    def recenter(self) -> bool:
        """
        Recenter using the most recent head yaw/pitch measurement.
        Returns True if recenter succeeded.
        """
        snap = self.get_snapshot(include_objects=False)
        hy, hp = snap.get("head_yaw_deg"), snap.get("head_pitch_deg")
        if hy is None or hp is None:
            return False
        if (not np.isfinite(hy)) or (not np.isfinite(hp)):
            return False

        self._yaw_offset = float(hy)
        self._pitch_offset = float(hp)
        self._have_recenter = True

        # Reset smoothers to 0 for nicer behavior
        self._head_yaw_smoother.y = 0.0
        self._head_pitch_smoother.y = 0.0
        self._gaze_yaw_smoother.y = 0.0
        self._gaze_pitch_smoother.y = 0.0

        self._log(f"[recenter] yaw_offset={self._yaw_offset:+.2f} deg, pitch_offset={self._pitch_offset:+.2f} deg")
        return True

    def get_snapshot(self, *, include_objects: bool = True, include_frame: bool = True) -> Dict[str, Any]:
        """
        Main consumer API for your session/orchestrator.

        Snapshot fields form an implicit "API contract" between `gaze_system.py`,
        `gaze_runner.py`, and experiment drivers.

        include_frame:
        - True  -> include frame_bgr (COPY) for UI rendering/debug
        - False -> do NOT copy/export pixels (service/telemetry mode). CV still runs internally.

        Snapshot contract (when ok=True):
          - timing:
              - unix_t / wall_t: timestamps (Unix seconds, and wall-clock seconds)
              - loop_hz: measured consumer loop rate (approx)
              - video_hz / vid_stale_s: video publish health metrics
              - det_hz / det_age_s: detector publish rate/age
              - infer_ms: most recent inference duration in milliseconds
          - gaze & head/eye geometry:
              - gaze_px: (x_px, y_px) in Neon scene pixel coordinates (smoothed)
              - gaze_px_raw: (x_px, y_px) raw/unsmoothed samples
              - head_yaw_deg / head_pitch_deg: head orientation
              - gaze_yaw_deg / gaze_pitch_deg: gaze direction (with recenter applied)
          - depth & quality (when available):
              - depth_cm: estimated vergence depth in centimeters
              - depth_valid: depth validity flag
              - miss_mm: miss distance in millimeters (when computed)
              - ipd_mm: interpupillary distance (millimeters)
          - IMU:
              - imu_angvel: angular velocity (rad/s) or None
              - imu_fresh: whether IMU is recent enough to trust
          - object tracking (optional):
              - objects: list of tracked detections (only if include_objects=True)
              - gaze_hit: None or a dict describing the current gaze-selected hit:
                  { track_id, name, conf, xyxy, mode, dist_px }
        """
        now_wall = time.time()

        # ---- Loop rate accounting ----
        self._loop_dt_hist.append(now_wall - self._t_loop_prev)
        self._t_loop_prev = now_wall
        loop_hz = _hz_from_hist(self._loop_dt_hist)

        # ---- Gaze snapshot ----
        with self._gaze_lock:
            g = self._latest_gaze

        if g is None:
            return {
                "ok": False,
                "reason": "no_gaze_yet",
                "wall_t": now_wall,
                "loop_hz": loop_hz,
            }

        worn = bool(getattr(g, "worn", False))
        gx, gy = float(getattr(g, "x", np.nan)), float(getattr(g, "y", np.nan))
        t_unix = float(getattr(g, "timestamp_unix_seconds", np.nan))

        # Smooth gaze
        if self._gaze_smooth_x is None:
            self._gaze_smooth_x, self._gaze_smooth_y = gx, gy
        else:
            a = float(self.cfg.gaze_smooth_alpha)
            self._gaze_smooth_x = (1.0 - a) * self._gaze_smooth_x + a * gx
            self._gaze_smooth_y = (1.0 - a) * self._gaze_smooth_y + a * gy
        gx_s, gy_s = float(self._gaze_smooth_x), float(self._gaze_smooth_y)

        # ---- Video staleness + rate + optional frame for UI ----
        vid_stale_s = float("nan")
        video_hz = float("nan")
        frame_shape = None
        frame_bgr = None

        with self._video_lock:
            pub_t = self._latest_video_wall_t
            buf = self._video_buf_bgr
            if buf is not None:
                frame_shape = buf.shape
                # Only copy/export pixels if requested (UI/debug)
                if include_frame:
                    frame_bgr = buf.copy()

        if pub_t is not None:
            vid_stale_s = float(max(0.0, now_wall - pub_t))

            if self._t_video_prev is not None and pub_t != self._t_video_prev:
                self._video_dt_hist.append(pub_t - self._t_video_prev)
            self._t_video_prev = pub_t
            video_hz = _hz_from_hist(self._video_dt_hist)

        # ---- IMU snapshot ----
        with self._imu_lock:
            imu_w = self._latest_imu_angvel
            imu_pub = self._latest_imu_wall_t
            imu = self._latest_imu

        imu_angvel = None
        imu_fresh = False
        if imu_w is not None and imu_pub is not None:
            if (now_wall - imu_pub) <= float(self.cfg.imu_fresh_s):
                imu_angvel = float(imu_w)
                imu_fresh = True

        # ---- Detector snapshot ----
        with self._det_lock:
            dets = list(self._latest_dets)
            det_fps = float(self._latest_det_fps)
            det_pub = self._latest_det_time
            infer_s = float(self._latest_infer_s) if np.isfinite(self._latest_infer_s) else float("nan")

        det_age_s = (now_wall - det_pub) if det_pub is not None else float("nan")
        infer_ms = (infer_s * 1000.0) if np.isfinite(infer_s) else float("nan")

        # ---- Tracking ----
        view_dets = dets
        if include_objects and self.cfg.use_tracker and self._tracker is not None:
            self._tracker.predict(now_wall)
            do_update = (det_pub is not None) and (det_pub != self._last_processed_det_time)
            if do_update:
                self._tracker.update_with_dets(dets, t_now=now_wall)
                self._last_processed_det_time = det_pub
            view_dets = self._tracker.get_tracks_as_dets(now_wall)

        # ---- Governor update ----
        self._gov_health_update(
            now_wall,
            vid_stale_s=vid_stale_s,
            loop_hz=loop_hz,
            imu_angvel=imu_angvel,
        )
        gov_enabled, gov_reason, gov_cd_left = self._gov_can_run(now_wall)

        # ---- Depth + angles ----
        d_cm = float("nan")
        miss_mm = float("nan")
        ipd_mm = float("nan")
        depth_valid = False

        head_yaw_meas = float("nan")
        head_pitch_meas = float("nan")
        gaze_yaw_meas = float("nan")
        gaze_pitch_meas = float("nan")

        if imu is not None and hasattr(imu, "quaternion"):
            q = imu.quaternion  # w,x,y,z expected
            R_imu_to_world = quat_to_rotmat_wxyz(float(q.w), float(q.x), float(q.y), float(q.z))

            # Head angles
            head_world = safe_unit(R_imu_to_world @ self._head_forward_local)
            head_elev, head_azim = cartesian_to_spherical_world(head_world)
            head_yaw_meas = wrap_deg_180(head_azim)
            head_pitch_meas = float(np.clip(head_elev, -90.0, 90.0)) if np.isfinite(head_elev) else head_elev

            # Gaze ray -> world
            try:
                gaze_ray = gaze_ray_from_optical_axes(g)  # scene coords
                gaze_imu = safe_unit(self._S2I @ gaze_ray) if self.cfg.apply_scene_to_imu_for_gaze else gaze_ray
                gaze_world = safe_unit(R_imu_to_world @ gaze_imu)
                g_elev, g_azim = cartesian_to_spherical_world(gaze_world)
                gaze_yaw_meas = wrap_deg_180(g_azim)
                gaze_pitch_meas = float(np.clip(g_elev, -90.0, 90.0)) if np.isfinite(g_elev) else g_elev
            except Exception:
                pass

            # Vergence depth + IPD
            try:
                L_mm = np.array([g.eyeball_center_left_x, g.eyeball_center_left_y, g.eyeball_center_left_z], dtype=float)
                R_mm = np.array([g.eyeball_center_right_x, g.eyeball_center_right_y, g.eyeball_center_right_z], dtype=float)
                uL = np.array([g.optical_axis_left_x, g.optical_axis_left_y, g.optical_axis_left_z], dtype=float)
                uR = np.array([g.optical_axis_right_x, g.optical_axis_right_y, g.optical_axis_right_z], dtype=float)

                ipd_mm = float(np.linalg.norm(L_mm - R_mm))
                depth_valid, d_m, miss_mm_val = vergence_depth_from_eyestate(
                    L_mm, uL, R_mm, uR,
                    miss_max_mm=self.cfg.miss_max_mm,
                    denom_min=self.cfg.denom_min
                )
                if worn and depth_valid:
                    d_s = self._depth_smoother.update(d_m)
                else:
                    d_s = self._depth_smoother.y
                d_cm = (d_s * 100.0) if np.isfinite(d_s) else float("nan")
                miss_mm = float(miss_mm_val) if np.isfinite(miss_mm_val) else float("nan")
            except Exception:
                pass

        # Apply recenter offsets
        if self._have_recenter:
            head_yaw_disp = wrap_deg_180(head_yaw_meas - self._yaw_offset) if np.isfinite(head_yaw_meas) else float("nan")
            head_pitch_disp = float(np.clip(head_pitch_meas - self._pitch_offset, -90.0, 90.0)) if np.isfinite(head_pitch_meas) else float("nan")
            gaze_yaw_disp = wrap_deg_180(gaze_yaw_meas - self._yaw_offset) if np.isfinite(gaze_yaw_meas) else float("nan")
            gaze_pitch_disp = float(np.clip(gaze_pitch_meas - self._pitch_offset, -90.0, 90.0)) if np.isfinite(gaze_pitch_meas) else float("nan")
        else:
            head_yaw_disp, head_pitch_disp = head_yaw_meas, head_pitch_meas
            gaze_yaw_disp, gaze_pitch_disp = gaze_yaw_meas, gaze_pitch_meas

        # Smooth displayed angles
        head_yaw_s = self._head_yaw_smoother.update(head_yaw_disp)
        head_pitch_s = self._head_pitch_smoother.update(head_pitch_disp)
        gaze_yaw_s = self._gaze_yaw_smoother.update(gaze_yaw_disp)
        gaze_pitch_s = self._gaze_pitch_smoother.update(gaze_pitch_disp)

        # ---- Gaze-on-object ----
        hit = hit_mode = hit_dist = None
        if include_objects and worn and frame_shape is not None and view_dets:
            h, w = int(frame_shape[0]), int(frame_shape[1])
            if (0 <= gx_s < w) and (0 <= gy_s < h):
                hit, hit_mode, hit_dist = gaze_object_hit(
                    gx_s, gy_s, view_dets,
                    gaze_radius_px=self.cfg.gaze_radius_px,
                    nearest_fallback_px=self.cfg.nearest_fallback_px,
                    gaze_recency_sec=self.cfg.gaze_recency_sec,
                )

        if hit is not None:
            self._last_hit = hit
            self._last_hit_wall_t = now_wall

        show_hit = None
        if self._last_hit is not None and (now_wall - self._last_hit_wall_t) <= float(self.cfg.gaze_hit_hold_sec):
            show_hit = self._last_hit

        # ---- Optional console print ----
        if self.cfg.enable_prints and now_wall >= self._next_print_t:
            self._next_print_t = now_wall + (1.0 / max(float(self.cfg.print_hz), 1e-6))

            topk = []
            if include_objects and view_dets:
                topk = sorted(
                    view_dets,
                    key=lambda d: float(d.get("conf", 0.0)),
                    reverse=True
                )[: int(self.cfg.topk_log)]

            objs_str = ", ".join(
                [f"{d.get('name','?')}#{int(d.get('track_id',-1))}({float(d.get('conf',0.0)):.2f})" for d in topk]
            ) if topk else "none"

            if show_hit is None:
                hit_str = "none"
            else:
                tid = int(show_hit.get("track_id", -1))
                nm = str(show_hit.get("name", "?"))
                cf = float(show_hit.get("conf", 0.0))
                hit_str = f"{nm}#{tid}({cf:.2f}) mode={hit_mode or '--'} d={(f'{hit_dist:.1f}px' if hit_dist is not None else '--')}"

            imu_str = f"{imu_angvel:.2f}rad/s" if (imu_angvel is not None and imu_fresh) else "NA"
            depth_str = f"{d_cm:.1f}cm" if np.isfinite(d_cm) else "N/A"
            gov_str = f"{'ON' if gov_enabled else 'OFF'}({gov_reason}) cd={gov_cd_left:.2f}s"

            self._log(
                f"t={t_unix:.3f} worn={worn} gaze=({gx_s:.1f},{gy_s:.1f}) | "
                f"rates: loop={loop_hz:.1f}Hz video={(video_hz if np.isfinite(video_hz) else 0):.1f}Hz "
                f"det~{det_fps:.1f}Hz det_age={(det_age_s if np.isfinite(det_age_s) else -1):.2f}s "
                f"infer={(infer_ms if np.isfinite(infer_ms) else -1):.0f}ms | "
                f"imu|w|={imu_str} | YOLO={gov_str} | tracks={(len(view_dets) if include_objects else 0)} | "
                f"objs: {objs_str} | gaze_on: {hit_str} | "
                f"depth={depth_str} miss={(miss_mm if np.isfinite(miss_mm) else -1):.1f}mm "
                f"IPD={(ipd_mm if np.isfinite(ipd_mm) else -1):.1f}mm | "
                f"HEAD(yaw,pitch)=({head_yaw_s:+.0f},{head_pitch_s:+.0f}) "
                f"GAZE(yaw,pitch)=({gaze_yaw_s:+.0f},{gaze_pitch_s:+.0f})"
            )

        # ---- Package snapshot ----
        snap: Dict[str, Any] = {
            "ok": True,
            "unix_t": t_unix,
            "wall_t": now_wall,
            "worn": worn,

            "frame_bgr": frame_bgr,          # None if include_frame=False
            "frame_shape": frame_shape,

            "gaze_px": (gx_s, gy_s),
            "gaze_px_raw": (gx, gy),

            "head_yaw_deg": float(head_yaw_s) if np.isfinite(head_yaw_s) else float("nan"),
            "head_pitch_deg": float(head_pitch_s) if np.isfinite(head_pitch_s) else float("nan"),
            "gaze_yaw_deg": float(gaze_yaw_s) if np.isfinite(gaze_yaw_s) else float("nan"),
            "gaze_pitch_deg": float(gaze_pitch_s) if np.isfinite(gaze_pitch_s) else float("nan"),

            "depth_cm": float(d_cm) if np.isfinite(d_cm) else float("nan"),
            "depth_valid": bool(worn and depth_valid),
            "miss_mm": float(miss_mm) if np.isfinite(miss_mm) else float("nan"),
            "ipd_mm": float(ipd_mm) if np.isfinite(ipd_mm) else float("nan"),

            "imu_angvel": float(imu_angvel) if imu_angvel is not None else None,
            "imu_fresh": bool(imu_fresh),

            "gov_enabled": bool(gov_enabled),
            "gov_reason": str(gov_reason),
            "gov_cd_left": float(gov_cd_left),

            "loop_hz": float(loop_hz) if np.isfinite(loop_hz) else float("nan"),
            "video_hz": float(video_hz) if np.isfinite(video_hz) else float("nan"),
            "vid_stale_s": float(vid_stale_s) if np.isfinite(vid_stale_s) else float("nan"),
            "det_hz": float(det_fps),
            "det_age_s": float(det_age_s) if np.isfinite(det_age_s) else float("nan"),
            "infer_ms": float(infer_ms) if np.isfinite(infer_ms) else float("nan"),
        }

        if include_objects:
            snap["objects"] = view_dets
            if show_hit is None:
                snap["gaze_hit"] = None
            else:
                snap["gaze_hit"] = {
                    "track_id": int(show_hit.get("track_id", -1)),
                    "name": str(show_hit.get("name", "?")),
                    "conf": float(show_hit.get("conf", 0.0)),
                    "xyxy": tuple(show_hit.get("xyxy")),
                    "mode": hit_mode,
                    "dist_px": float(hit_dist) if hit_dist is not None else None,
                }

        return snap


    # -----------------------
    # Threads: Video/Gaze/IMU
    # -----------------------
    def _video_thread_fn(self, device):
        """
        Blocking receive loop for scene video frames.

        Publish strategy:
        - Copy-on-publish (immutable snapshot) each frame.
        - Single canonical buffer under _video_lock.
        - Detector/UI read from _video_buf_bgr under _video_lock.
        """
        while not self._stop_event.is_set():
            try:
                frame, _dt = device.receive_scene_video_frame()  # blocking
                if frame is None:
                    continue

                now = time.time()

                # Copy-on-publish: prevents SDK buffer reuse / tearing
                frame_pub = frame.copy()

                with self._video_lock:
                    self._video_buf_bgr = frame_pub
                    self._video_buf_shape = frame_pub.shape
                    self._latest_video_wall_t = now

            except Exception:
                time.sleep(0.001)

    def _gaze_thread_fn(self, device):
        while not self._stop_event.is_set():
            try:
                g = device.receive_gaze_datum()  # blocking
                with self._gaze_lock:
                    self._latest_gaze = g
                    self._latest_gaze_wall_t = time.time()
            except Exception:
                time.sleep(0.01)

    def _imu_thread_fn(self, device):
        """
        Publishes:
          - latest_imu: full datum (for quaternion)
          - latest_imu_angvel: magnitude (rad/s) for gating
        """
        q_prev = None
        t_prev = None

        while not self._stop_event.is_set():
            try:
                imu = device.receive_imu_datum()  # blocking
                t_now = time.time()

                mag = self._gyro_mag_from_imu_datum(imu)

                if not np.isfinite(mag):
                    # fallback omega from quaternion delta
                    if hasattr(imu, "quaternion"):
                        try:
                            q_now = self._quat_to_np_wxyz(imu.quaternion)
                            if q_prev is not None and t_prev is not None:
                                mag = self._omega_from_quat_delta(q_prev, q_now, t_now - t_prev)
                            q_prev = q_now
                            t_prev = t_now
                        except Exception:
                            mag = float("nan")

                with self._imu_lock:
                    self._latest_imu = imu
                    self._latest_imu_angvel = float(mag) if np.isfinite(mag) else None
                    self._latest_imu_wall_t = t_now

            except Exception:
                time.sleep(0.005)

    # -----------------------
    # Detector thread (YOLO + Governor)
    # -----------------------
    def _detector_thread_fn(self):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:
            raise SystemExit(
                "Missing dependency 'ultralytics'. Install with:\n  pip install ultralytics\n"
            ) from e

        det_device = "cpu"
        try:
            import torch  # type: ignore
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            # Use CUDA when available — typically the Windows dev box (RTX 4070
            # Ti). Linux deployment has no NVIDIA driver, so this falls back to
            # CPU automatically per the CLAUDE.md platform-gating policy.
            if torch.cuda.is_available():
                det_device = "cuda"
        except Exception:
            pass

        model = YOLO(self.cfg.model_name)
        try:
            model.to(det_device)
        except Exception:
            pass

        period = 1.0 / max(float(self.cfg.det_hz), 1e-6)
        next_run = time.perf_counter()

        fps_hist = deque(maxlen=30)
        t_prev = time.time()

        local_bgr = None
        local_shape = None

        while not self._stop_event.is_set():
            now_pc = time.perf_counter()
            if now_pc < next_run:
                time.sleep(min(0.002, next_run - now_pc))
                continue

            now = time.time()
            enabled, _reason, _cd_left = self._gov_can_run(now)

            # Read latest published frame (canonical buffer)
            with self._video_lock:
                src = self._video_buf_bgr
                pub_wall = self._latest_video_wall_t

            if (not enabled) or (src is None):
                next_run += period
                now_pc2 = time.perf_counter()
                if next_run < now_pc2 - 2.0 * period:
                    next_run = now_pc2
                time.sleep(0.010)
                continue

            # Local copy for detector safety + resize/infer
            if (local_bgr is None) or (local_shape != src.shape):
                local_bgr = np.empty_like(src)
                local_shape = src.shape
            np.copyto(local_bgr, src)

            scale = 1.0
            frame_det = local_bgr

            if self.cfg.detect_resize_width is not None:
                h, w = frame_det.shape[:2]
                new_w = int(self.cfg.detect_resize_width)
                if w != new_w:
                    new_h = int(round(h * (new_w / float(w))))
                    import cv2
                    frame_det = cv2.resize(frame_det, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    scale = new_w / float(w)

            t0 = time.time()
            results = model.predict(
                source=frame_det,
                verbose=False,
                conf=float(self.cfg.det_conf),
                iou=float(self.cfg.det_iou),
                device=det_device,
                max_det=int(self.cfg.det_max_det),
                classes=self.cfg.det_classes if (self.cfg.det_classes is not None and len(self.cfg.det_classes) > 0) else None,
            )
            infer_s = time.time() - t0

            with self._det_lock:
                self._latest_infer_s = float(infer_s)

            if infer_s > float(self.cfg.gov_slow_infer_s):
                self._gov_disable(f"infer>{self.cfg.gov_slow_infer_s:.2f}s", time.time())

            dets: List[dict] = []
            if results:
                r0 = results[0]
                names = r0.names if hasattr(r0, "names") else getattr(model, "names", None)
                if hasattr(r0, "boxes") and r0.boxes is not None:
                    boxes = r0.boxes
                    xyxy = boxes.xyxy.cpu().numpy()
                    conf = boxes.conf.cpu().numpy()
                    cls = boxes.cls.cpu().numpy().astype(int)

                    if scale != 1.0:
                        xyxy = xyxy / scale

                    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
                        if isinstance(names, dict):
                            name = names.get(int(k), str(int(k)))
                        elif isinstance(names, (list, tuple)):
                            name = names[int(k)]
                        else:
                            name = str(int(k))

                        dets.append(
                            {
                                "xyxy": (float(x1), float(y1), float(x2), float(y2)),
                                "cls": int(k),
                                "conf": float(c),
                                "name": str(name),
                                # best-effort "frame time" for correlation (use wall publish time)
                                "frame_time": float(pub_wall) if pub_wall is not None else float("nan"),
                            }
                        )

            with self._det_lock:
                self._latest_dets = dets
                self._latest_det_time = time.time()

            # detector fps (wall)
            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 1e-6:
                fps_hist.append(1.0 / dt)
                with self._det_lock:
                    self._latest_det_fps = float(np.mean(fps_hist)) if fps_hist else 0.0

            next_run += period
            now_pc2 = time.perf_counter()
            if next_run < now_pc2 - 2.0 * period:
                next_run = now_pc2

    # -----------------------
    # Governor
    # -----------------------
    def _gov_disable(self, reason: str, now: float):
        with self._gov_lock:
            self._gov_enabled = False
            self._gov_reason = str(reason)
            self._gov_cooldown_until = max(self._gov_cooldown_until, now + float(self.cfg.gov_cooldown_s))
            self._gov_stable_since = None

    def _gov_health_update(self, now: float, *, vid_stale_s: float, loop_hz: float, imu_angvel: Optional[float]):
        unhealthy = False
        why = None

        if np.isfinite(vid_stale_s) and vid_stale_s > float(self.cfg.gov_vid_stale_disable_s):
            unhealthy = True
            why = f"vid_stale>{self.cfg.gov_vid_stale_disable_s:.2f}s"
        elif np.isfinite(loop_hz) and loop_hz > 0 and loop_hz < float(self.cfg.gov_loop_min_hz):
            unhealthy = True
            why = f"loop<{self.cfg.gov_loop_min_hz:.1f}Hz"
        elif self.cfg.enable_imu_gate and (imu_angvel is not None) and (imu_angvel > float(self.cfg.gov_imu_angvel_disable)):
            unhealthy = True
            why = f"imu|w|>{self.cfg.gov_imu_angvel_disable:.2f}"

        if unhealthy:
            self._gov_disable(str(why), now)
            return

        with self._gov_lock:
            if (not self._gov_enabled) and (now < self._gov_cooldown_until):
                self._gov_stable_since = None
                return

            if not self._gov_enabled:
                if self._gov_stable_since is None:
                    self._gov_stable_since = now
                if (now - self._gov_stable_since) >= float(self.cfg.gov_reenable_stable_s):
                    self._gov_enabled = True
                    self._gov_reason = "healthy"
                    self._gov_stable_since = None
                return

            self._gov_reason = "healthy"
            self._gov_stable_since = None

    def _gov_can_run(self, now: float) -> Tuple[bool, str, float]:
        # Manual CV toggle takes precedence but does NOT mutate governor internals.
        if not bool(getattr(self.cfg, "enable_cv", True)):
            return False, "cv_disabled", 0.0

        with self._gov_lock:
            user_on = bool(self._cv_user_enabled)
            enabled = bool(self._gov_enabled) and user_on
            reason = "cv_user_off" if (not user_on) else str(self._gov_reason)
            cd = float(max(0.0, self._gov_cooldown_until - now)) if user_on else 0.0

        return enabled, reason, cd

    # -----------------------
    # IMU angular velocity helpers (copied from your stable version)
    # -----------------------
    @staticmethod
    def _quat_to_np_wxyz(q) -> np.ndarray:
        return np.array([float(q.w), float(q.x), float(q.y), float(q.z)], dtype=np.float64)

    @staticmethod
    def _quat_normalize(q: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(q))
        if n < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return q / n

    @staticmethod
    def _quat_conj(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

    @staticmethod
    def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw
        ], dtype=np.float64)

    def _omega_from_quat_delta(self, q_prev: np.ndarray, q_now: np.ndarray, dt: float) -> float:
        if dt <= 1e-6:
            return float("nan")
        q_prev = self._quat_normalize(q_prev)
        q_now = self._quat_normalize(q_now)
        q_delta = self._quat_mul(self._quat_conj(q_prev), q_now)
        q_delta = self._quat_normalize(q_delta)
        w = float(np.clip(q_delta[0], -1.0, 1.0))
        angle = 2.0 * float(np.arccos(w))
        if angle > np.pi:
            angle = 2.0 * np.pi - angle
        return float(angle / dt)

    @staticmethod
    def _gyro_mag_from_imu_datum(imu) -> float:
        candidates = [
            ("angular_velocity_x", "angular_velocity_y", "angular_velocity_z"),
            ("gyro_x", "gyro_y", "gyro_z"),
            ("gyroscope_x", "gyroscope_y", "gyroscope_z"),
            ("gyr_x", "gyr_y", "gyr_z"),
        ]
        for fx, fy, fz in candidates:
            if all(hasattr(imu, f) for f in (fx, fy, fz)):
                try:
                    gx = float(getattr(imu, fx))
                    gy = float(getattr(imu, fy))
                    gz = float(getattr(imu, fz))
                    mag = float(np.sqrt(gx*gx + gy*gy + gz*gz))
                    if np.isfinite(mag):
                        return mag
                except Exception:
                    pass

        for subname in ("gyro", "angular_velocity", "gyroscope"):
            if hasattr(imu, subname):
                sub = getattr(imu, subname)
                for fx, fy, fz in candidates:
                    if all(hasattr(sub, f) for f in (fx, fy, fz)):
                        try:
                            gx = float(getattr(sub, fx))
                            gy = float(getattr(sub, fy))
                            gz = float(getattr(sub, fz))
                            mag = float(np.sqrt(gx*gx + gy*gy + gz*gz))
                            if np.isfinite(mag):
                                return mag
                        except Exception:
                            pass

        return float("nan")
