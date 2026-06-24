"""
apriltag_detect.py — AprilTag detection + frame-relay consumption, shared by
the AprilTag calibration + control-test tools (WS5).

Keeps the perception I/O (pupil-apriltags detection, the frame-relay latest-bundle
thread) in one place so both `tools/apriltag_calibrate.py` and
`tools/apriltag_control_test.py` use the same code. The geometry math is in
`apriltag_calib.py`; the robot link is in `harmony_link.py`.

`pupil-apriltags` is imported lazily (it is intentionally not yet in
environment.yml — added only once the AprilTag calibration is adopted). Tag-pose
translations are converted metres→mm so they share units with robot telemetry
and the X calibration column.
"""

from __future__ import annotations

import threading
from typing import Dict, Tuple

import numpy as np

from Utils.gaze.apriltag_calib import make_transform

_M_TO_MM = 1000.0


def _log(msg: str) -> None:
    print(f"[apriltag_detect] {msg}", flush=True)


def load_detector(families: str = "tag36h11"):
    """Build a pupil-apriltags Detector, failing fast with a remediation message
    if the (intentionally not-yet-vendored) dep is missing."""
    try:
        from pupil_apriltags import Detector
    except ImportError as exc:
        raise SystemExit(
            "pupil-apriltags is required for AprilTag detection but is not installed.\n"
            "  pip install pupil-apriltags\n"
            "(It is deliberately not in environment.yml until the AprilTag "
            "calibration is adopted — see rev03-apriltag-methodology.md §6.4.)"
        ) from exc
    return Detector(families=families)


def camera_params(K: np.ndarray) -> Tuple[float, float, float, float]:
    """(fx, fy, cx, cy) for pupil-apriltags from the 3×3 intrinsics."""
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def rescale_pose_t_mm(pose_t_m, true_size_m: float, ref_size_m: float) -> np.ndarray:
    """Translation (mm) of a tag of physical edge ``true_size_m`` from a pose the
    detector computed at ``ref_size_m``. A tag's recovered translation scales
    **linearly** with its physical edge (the rotation is size-independent), so
    detecting once at a reference size and rescaling per-tag lets the world and
    EE tags be DIFFERENT sizes — e.g. a large world tag + a small EE tag."""
    return np.asarray(pose_t_m, dtype=float).ravel() * (_M_TO_MM * true_size_m / ref_size_m)


def detect_tags(detector, bgr: np.ndarray, K: np.ndarray, tag_size_m: float,
                tag_sizes: Dict[int, float] = None) -> Dict[int, Dict]:
    """Detect tags → ``{tag_id: {T (4×4, mm), margin, hamming, center}}``. Pose
    translation is metres→mm so it matches robot telemetry / the X column.

    ``tag_size_m`` is the default physical edge (m); ``tag_sizes`` optionally
    overrides it per tag id (e.g. ``{ee_tag_id: 0.04}`` for a small EE tag while
    the world tag stays large). Detection runs once at ``tag_size_m`` and each
    tag's translation is rescaled to its true size (see ``rescale_pose_t_mm``)."""
    import cv2  # lazy: only the camera path needs it
    sizes = tag_sizes or {}
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    results = detector.detect(
        gray, estimate_tag_pose=True,
        camera_params=camera_params(K), tag_size=tag_size_m,
    )
    out: Dict[int, Dict] = {}
    for r in results:
        if r.pose_R is None or r.pose_t is None:
            continue
        true_size = sizes.get(int(r.tag_id), tag_size_m)
        t_mm = rescale_pose_t_mm(r.pose_t, true_size, tag_size_m)
        T = make_transform(np.asarray(r.pose_R, dtype=float), t_mm)
        out[int(r.tag_id)] = {
            "T": T,
            "margin": float(r.decision_margin),
            "hamming": int(r.hamming),
            "center": np.asarray(r.center, dtype=float),
        }
    return out


class RelayConsumer:
    """Owns a RemoteFrameReader and keeps the latest bundle available on demand.
    The reader's ``__iter__`` blocks, so a daemon thread drains it; callers poll
    ``latest()`` and dedup on ``frame_idx``. Exposes the handshake ``camera_matrix``."""

    def __init__(self, host: str, port: int, handshake_s: float = 5.0):
        import time
        from Utils.remote_frame_reader import RemoteFrameReader
        self._time = time.time
        self._reader = RemoteFrameReader(
            host, port, wait_for_handshake_s=handshake_s, auto_reconnect=False)
        self._latest = None
        self._latest_t = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="relay-consumer",
                                        daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            for bundle in self._reader:
                if self._stop.is_set():
                    break
                t_recv = self._time()
                with self._lock:
                    self._latest = bundle
                    self._latest_t = t_recv
        except Exception as exc:  # surface, don't swallow: the relay died
            _log(f"relay consumer stopped: {exc!r}")

    def latest(self):
        with self._lock:
            return self._latest

    def latest_with_time(self):
        """The latest bundle paired with its host-clock arrival time (``time.time()``
        at the moment this consumer received it). The sweep stage time-aligns this
        against the robot telemetry timestamp to reject stale frames (rev04 §2)."""
        with self._lock:
            return self._latest, self._latest_t

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.asarray(self._reader.camera_matrix, dtype=float)

    def close(self) -> None:
        self._stop.set()
        try:
            self._reader.close()
        except Exception:
            # Best-effort teardown: the reader/socket may already be closed by
            # the daemon thread's iterator exit; a failure here is benign on a
            # bench tool already shutting down.
            pass
