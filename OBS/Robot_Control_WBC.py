#!/usr/bin/env python3
# harmony_wbc_mm_calib_exec.py
#
# Phase 1 (Calibration): Accepts two comma-separated lists:
#   Trajectories: a,x,y,z,...
#   Durations(s): 2,3,4,1.5,...
# For each (traj_i, dur_i): preset; dur=dur_i -> g -> log telemetry for dur_i seconds.
# Builds ΔQ/ΔX and fits J (mm/rad).
#
# Phase 2 (Control): Go-to XYZ (mm) via duration-controlled joint waypoints with a nullspace bubble.
#  - NEW: Optionally updates the Jacobian online from recent (q,x) samples (sliding window)
#
# Safety: Type 'p' + Enter at ANY time -> pause, home, exit.
#
# Networking: Binds to CONTROL_IP:CONTROL_PORT and talks to ROBOT_IP:ROBOT_PORT.
# Telemetry via 'q;seq=...'. ACKs printed with timestamps.

import json, time, socket, threading, sys
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from datetime import datetime
from collections import deque

# ======================
# Network config
# ======================
ROBOT_IP   = "192.168.2.1"
ROBOT_PORT = 8080

CONTROL_IP   = "0.0.0.0"   # bind all interfaces; change to "192.168.2.2" if you prefer explicit binding
CONTROL_PORT = 8080

ACK_TIMEOUT_S = 0.35
TEL_TIMEOUT_S = 0.60

# ======================
# Control / calibration config
# ======================

# Duration format the robot expects (note the spaces around '=')
DUR_FMT = ";dur={dur:.3f}"

ACTIVE_SIDE = "R"                     # "R" or "L"
SAMPLE_RATE_HZ = 10.0                 # telemetry rate during calibration
SUFF_MIN_SAMPLES = 40
SUFF_MIN_SINGVAL = 1e-3

WAYPOINT_DUR_S = 1.0               # sec per waypoint in Phase 2
PREEMPT_RATE_HZ = 4.0                 # Hz
MAX_DQ_STEP = 0.05                    # rad ∞-norm per update clamp
DAMPING = 1e-3

# Bubble (participant keepout), mm
BUBBLE_CENTER_MM = np.array([0.0, 250.0, 900.0], dtype=float)
BUBBLE_RADIUS_MM = 250.0
BUBBLE_GAIN = 0.25

GOAL_TOL_MM = 5.0
PRINT_J = True

# Motion completion / home settle logic
MIN_ACTIVE_TIME_S   = 0.75   # must see motion for at least this long before considering stillness
STILL_DQ_THRESH     = 0.01   # rad/s per joint to consider "still"
STILL_CONSEC_N      = 5      # need N consecutive still samples
STILL_POLL_HZ       = 10.0   # Hz poll rate for stillness checks
HOME_SLEEP_S        = 3.0    # coarse wait after 'h' (matches your log timing)
HOME_LOG_EXTRA_S    = 1.0    # keep logging a bit into home
# How long to stay quiet after ANY ACKed command so the controller can process it
CMD_PROCESS_GRACE_S = 0.40  # seconds (tune 0.3–0.7 if needed)

# Motion-state guards
ARMING_DELAY_S    = 1.00   # wait after preset ACK before sending 'g'
POST_GO_QUIET_S   = 0.80   # don't send any 'q' for this long after 'g'

# Telemetry print throttling (0 = never print telemetry lines, otherwise print every N)
TELEMETRY_PRINT_EVERY = 1

# ======================
# NEW: Online Jacobian update settings (Phase 2)
# ======================
ENABLE_ONLINE_J = True            # toggle online adaptation
ONLINE_WINDOW_SEC = 6.0           # sliding window length of recent samples
ONLINE_UPDATE_EVERY_STEPS = 4     # how often (in Phase 2 loop iterations) to attempt a re-fit
ONLINE_MIN_PAIRS = 20             # minimum Δ pairs needed to consider an update
ONLINE_MIN_SINGVAL = 5e-4         # smaller than offline to allow smaller windows
ONLINE_BLEND_ALPHA = 0.25         # blend factor: J <- (1-α)J + α J_new
ONLINE_PRINT_EVERY_UPD = 2        # print every N successful online J updates (set 1 to print every update)


# --- NEW: Time-boxed target move (total duration to reach target) ---
MIN_TOTAL_DUR_S   = 1.0
MAX_TOTAL_DUR_S   = 15.0

# Robot-side duration constraints (from robot app)
MIN_ROBOT_DUR_S = 1.00
MAX_ROBOT_DUR_S = 10.00

# --- NEW: Gate the emergency watcher so it doesn't steal prompts ---
_watcher_enabled = threading.Event()



# ======================
# Globals
# ======================
_emergency_evt = threading.Event()
_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
_sock.bind((CONTROL_IP, CONTROL_PORT))
_sock.settimeout(0.5)

_tel_counter = 0

def _ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def _send(msg: str):
    _sock.sendto(msg.encode("utf-8"), (ROBOT_IP, ROBOT_PORT))

def _recv(timeout_s: float) -> Optional[str]:
    _sock.settimeout(timeout_s)
    try:
        data, _ = _sock.recvfrom(65535)
        return data.decode("utf-8", errors="ignore")
    except socket.timeout:
        return None

# ======================
# Packet slimming (minimize memory / terminal load)
# ======================
def _slim_packet(d: Dict[str, Any], side: str = ACTIVE_SIDE) -> Optional[Dict[str, Any]]:
    """
    Keep only what we need for ΔQ/ΔX and motion checks.
    Returns a dict with keys:
      side=='R' -> {'qR': [...], 'eeR': {'pos_mm': [...]}, 'dqR': [...]}
      side=='L' -> {'qL': [...], 'eeL': {'pos_mm': [...]}, 'dqL': [...]}
    """
    try:
        if side.upper() == "R":
            return {
                "qR":  d["qR"],
                "dqR": d.get("dqR", []),
                "eeR": {"pos_mm": d["eeR"]["pos_mm"]},
            }
        else:
            return {
                "qL":  d["qL"],
                "dqL": d.get("dqL", []),
                "eeL": {"pos_mm": d["eeL"]["pos_mm"]},
            }
    except Exception:
        return None

# ======================
# Protocol helpers
# ======================
def send_and_wait_ack(msg: str, expect_prefix: Optional[str] = None, timeout: float = ACK_TIMEOUT_S) -> bool:
    """
    Send and wait for ACK. If expect_prefix is None, any 'ACK:' is accepted.
    Prints ACK with timestamps. After a successful ACK, pauses for controller processing.
    """
    print(f"[{_ts()}] TX: {msg}")
    _send(msg)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _emergency_evt.is_set():
            return False
        r = _recv(timeout)
        if not r:
            continue
        if r.startswith("ACK:"):
            print(f"[{_ts()}] {r.strip()}")
            if expect_prefix is None or r.strip().startswith(f"ACK:{expect_prefix}"):
                # --- NEW: post-ACK processing grace ---
                if CMD_PROCESS_GRACE_S > 0:
                    time.sleep(CMD_PROCESS_GRACE_S)
                return True
        # ignore non-ACK chatter here
    print(f"[{_ts()}] ACK timeout waiting for: {msg}")
    return False

def query_state() -> Optional[dict]:
    """Send 'q;seq=...' and receive JSON (slimmed). Timestamp host-side events."""
    global _tel_counter
    if _emergency_evt.is_set():
        return None
    seq = int(time.time() * 1000) & 0xFFFFFFFF
    msg = f"q;seq={seq}"
    # Keep TX/ACK printed; throttle RX prints below
    print(f"[{_ts()}] TX: {msg}")
    _send(msg)

    # Try to get ACK:q (optional)
    t0 = time.time()
    while time.time() - t0 < ACK_TIMEOUT_S:
        r = _recv(ACK_TIMEOUT_S)
        if not r:
            break
        if r.startswith("ACK:q"):
            print(f"[{_ts()}] {r.strip()}")
            break
        if r.startswith("{"):
            try:
                pkt = json.loads(r)
                slim = _slim_packet(pkt, ACTIVE_SIDE)
                if slim is None:
                    return None
                _tel_counter += 1
                if TELEMETRY_PRINT_EVERY and (_tel_counter % TELEMETRY_PRINT_EVERY == 0):
                    print(f"[{_ts()}] RX: telemetry (seq={seq})")
                return slim
            except json.JSONDecodeError:
                pass

    # Then read JSON if we didn't already
    t1 = time.time()
    while time.time() - t1 < TEL_TIMEOUT_S:
        r = _recv(TEL_TIMEOUT_S)
        if not r:
            break
        if r.startswith("{"):
            try:
                pkt = json.loads(r)
                slim = _slim_packet(pkt, ACTIVE_SIDE)
                if slim is None:
                    return None
                _tel_counter += 1
                if TELEMETRY_PRINT_EVERY and (_tel_counter % TELEMETRY_PRINT_EVERY == 0):
                    print(f"[{_ts()}] RX: telemetry (seq={seq})")
                return slim
            except json.JSONDecodeError:
                continue
        elif r.startswith("ACK:"):
            print(f"[{_ts()}] {r.strip()}")
    print(f"[{_ts()}] WARN: telemetry timeout (seq={seq})")
    return None

def parse_arm(d: dict, side="R") -> Tuple[np.ndarray, np.ndarray]:
    if side.upper() == "R":
        q  = np.array(d["qR"], dtype=float)
        ee = np.array(d["eeR"]["pos_mm"], dtype=float)
    else:
        q  = np.array(d["qL"], dtype=float)
        ee = np.array(d["eeL"]["pos_mm"], dtype=float)
    return q, ee

def _max_abs_dq(d: dict, side: str) -> float:
    if side.upper() == "R":
        dq = np.array(d.get("dqR", []), dtype=float)
    else:
        dq = np.array(d.get("dqL", []), dtype=float)
    if dq.size != 7:
        return float("inf")
    return float(np.max(np.abs(dq)))

def send_coords_and_go(q_target: np.ndarray, dur_s: float) -> bool:
    if _emergency_evt.is_set():
        return False
    # clamp once, centrally
    dur_s = float(max(MIN_ROBOT_DUR_S, min(MAX_ROBOT_DUR_S, dur_s)))

    coords = ",".join(f"{v:.6f}" for v in q_target.tolist()) + DUR_FMT.format(dur=dur_s)
    if not send_and_wait_ack(coords, expect_prefix=None):
        return False
    time.sleep(0.01)
    ok = send_and_wait_ack("g", expect_prefix="g")
    if ok:
        time.sleep(0.02)
    return ok

def wait_until_motion_complete(min_active_s: float,
                               still_thresh: float,
                               still_consec: int,
                               poll_hz: float) -> None:
    """
    Poll telemetry until:
      1) we've observed at least `min_active_s` of motion time since 'g', and
      2) then we see `still_consec` consecutive samples with max|dq| <= still_thresh.
    Hard-cap the wait so we don't get stuck if telemetry hiccups.
    """
    t_start = time.time()
    period = 1.0 / max(1e-6, poll_hz)
    consec = 0
    deadline = t_start + 10.0

    # Phase A: require some active time
    while time.time() < deadline:
        d = query_state()
        if d and (time.time() - t_start) >= min_active_s:
            break
        time.sleep(period)

    # Phase B: require consecutive still samples
    while time.time() < deadline and consec < still_consec:
        d = query_state()
        if d and _max_abs_dq(d, ACTIVE_SIDE) <= still_thresh:
            consec += 1
        else:
            consec = 0
        time.sleep(period)

def wait_for_home_complete() -> None:
    """Coarse sleep + stillness check to ensure home finished."""
    time.sleep(HOME_SLEEP_S)
    wait_until_motion_complete(0.0, STILL_DQ_THRESH, STILL_CONSEC_N, STILL_POLL_HZ)

# ======================
# Emergency watcher (type 'p' + Enter any time)
# ======================
def emergency_watcher():
    """
    Only reads stdin when _watcher_enabled is set. This prevents it from
    stealing lines during interactive prompts. Use Ctrl+C for safety while
    prompts are shown (KeyboardInterrupt handler already pauses/homes/exits).
    """
    try:
        while True:
            if not _watcher_enabled.is_set():
                time.sleep(0.05)
                continue
            s = sys.stdin.readline()
            if not s:
                time.sleep(0.02)
                continue
            if s.strip().lower() == "p":
                print("[EMERGENCY] Operator requested PAUSE.")
                _emergency_evt.set()
                try: send_and_wait_ack("p", expect_prefix="p")
                except Exception: pass
                print("[EMERGENCY] Homing...")
                try: send_and_wait_ack("h", expect_prefix="h")
                except Exception: pass
            # If not 'p', ignore (do NOT consume user prompts anymore)
    except Exception:
        _emergency_evt.set()
        try:
            send_and_wait_ack("p", expect_prefix="p")
            send_and_wait_ack("h", expect_prefix="h")
        except Exception:
            pass
        import os; os._exit(1)

# ======================
# Calibration routines
# ======================
def wait_until_motion_starts(thresh_rad_s: float = 0.02, timeout_s: float = 1.2) -> bool:
    """Return True once max|dq| exceeds threshold within timeout, else False."""
    t0 = time.time()
    while time.time() - t0 < timeout_s and not _emergency_evt.is_set():
        d = query_state()
        if d and _max_abs_dq(d, ACTIVE_SIDE) >= thresh_rad_s:
            return True
        time.sleep(0.05)
    return False

def run_preset_with_duration_and_log(opcode: str, dur_s: float, samplerate_hz: float) -> List[dict]:
    """
    Exact shot flow (with arm/quiet guards):
      (preset + dur) -> ACK           e.g., "a; dur = 4.000"
      [ARMING_DELAY_S]
      g -> ACK
      [POST_GO_QUIET_S]   <-- no 'q' during this time
      query loop for dur_s seconds (slimmed)
      h -> ACK
      short query during home; wait_for_home_complete()
    """
    logs: List[dict] = []
    if _emergency_evt.is_set():
        return logs

    op = opcode.strip()
    # --- send preset WITH duration ---
    preset_with_dur = f"{op}{DUR_FMT.format(dur=max(0.05, float(dur_s)))}"
    print(f"[{_ts()}] [CAL] Stage preset '{op}' with duration {float(dur_s):.2f}s.")
    if not send_and_wait_ack(preset_with_dur, expect_prefix=None, timeout=ACK_TIMEOUT_S):
        print(f"[{_ts()}] [CAL] ERROR: preset '{op}' not ACKed; skipping shot.")
        return logs

    # Let the robot actually arm the preset before 'g'
    print(f"[{_ts()}] [CAL] Arming delay {ARMING_DELAY_S:.2f}s before 'g'...")
    time.sleep(ARMING_DELAY_S)

    # Send GO
    if not send_and_wait_ack("g", expect_prefix="g", timeout=ACK_TIMEOUT_S):
        print(f"[{_ts()}] [CAL] WARN: 'g' not ACKed; continuing anyway.")

    # Quiet window after GO — DO NOT send 'q' here
    print(f"[{_ts()}] [CAL] Quiet {POST_GO_QUIET_S:.2f}s after GO (no telemetry requests).")
    time.sleep(POST_GO_QUIET_S)

    # Verify motion started (by looking at dq) with a few light queries
    if not wait_until_motion_starts(thresh_rad_s=0.02, timeout_s=1.2):
        print(f"[{_ts()}] [CAL] WARN: motion not detected; retry 'g' once.")
        time.sleep(0.2)
        send_and_wait_ack("g", expect_prefix="g", timeout=ACK_TIMEOUT_S)
        print(f"[{_ts()}] [CAL] Quiet {POST_GO_QUIET_S:.2f}s after retry-GO.")
        time.sleep(POST_GO_QUIET_S)
        if not wait_until_motion_starts(thresh_rad_s=0.02, timeout_s=1.2):
            print(f"[{_ts()}] [CAL] ERROR: motion failed to start; proceeding to home.")
            send_and_wait_ack("h", expect_prefix="h", timeout=ACK_TIMEOUT_S)
            wait_for_home_complete()
            return logs

    # Main logging loop (slim packets only)
    period = 1.0 / max(1e-6, samplerate_hz)
    t_end = time.time() + float(dur_s)
    print(f"[{_ts()}] [CAL] Logging telemetry for ~{dur_s:.2f}s at {samplerate_hz:.1f} Hz...")
    while time.time() < t_end and not _emergency_evt.is_set():
        d = query_state()
        if d:
            logs.append(d)  # already slimmed
        time.sleep(period)

    # Home afterwards
    print(f"[{_ts()}] [CAL] Homing...")
    if not send_and_wait_ack("h", expect_prefix="h", timeout=ACK_TIMEOUT_S):
        print(f"[{_ts()}] [CAL] WARN: 'h' not ACKed; waiting out home anyway.")

    # light logging into home
    t_home_log_end = time.time() + HOME_LOG_EXTRA_S
    while time.time() < t_home_log_end and not _emergency_evt.is_set():
        d = query_state()
        if d:
            logs.append(d)
        time.sleep(period)

    wait_for_home_complete()
    d = query_state()
    if d:
        logs.append(d)
    print(f"[{_ts()}] [CAL] Shot complete.")
    return logs

# ======================
# Linear-algebra helpers
# ======================
def build_delta_mats(samples: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    if len(samples) < 2:
        return np.zeros((7,0)), np.zeros((3,0))
    dq_cols, dx_cols = [], []
    for k in range(len(samples) - 1):
        q0, x0 = samples[k]
        q1, x1 = samples[k+1]
        dq_cols.append((q1 - q0).reshape(-1,1))  # rad
        dx_cols.append((x1 - x0).reshape(-1,1))  # mm
    ΔQ = np.hstack(dq_cols) if dq_cols else np.zeros((7,0))
    ΔX = np.hstack(dx_cols) if dx_cols else np.zeros((3,0))
    return ΔQ, ΔX

def fit_global_J(ΔQ: np.ndarray, ΔX: np.ndarray, lam: float = DAMPING) -> np.ndarray:
    if ΔQ.shape[1] == 0:
        return np.zeros((3,7))
    G = ΔQ @ ΔQ.T
    H = G + lam * np.eye(ΔQ.shape[0])
    J = (ΔX @ ΔQ.T) @ np.linalg.inv(H)
    return J

def sufficiency_check(ΔQ: np.ndarray) -> Tuple[bool, dict]:
    info = {"n_pairs": ΔQ.shape[1], "singvals": [], "condition": None}
    if ΔQ.shape[1] < SUFF_MIN_SAMPLES:
        return False, info
    u, s, vh = np.linalg.svd(ΔQ, full_matrices=False)
    info["singvals"] = s.tolist()
    cond = (s[0] / (s[-1] + 1e-12)) if s[-1] > 0 else np.inf
    info["condition"] = cond
    ok = (s[-1] >= SUFF_MIN_SINGVAL)
    return ok, info

def pinv_damped(J: np.ndarray, lam: float = DAMPING) -> np.ndarray:
    JTJ = J.T @ J
    H = JTJ + lam * np.eye(JTJ.shape[0])
    return np.linalg.solve(H, J.T)  # (7x3)

# ======================
# Bubble nullspace term
# ======================
def bubble_grad_task_space(x_mm: np.ndarray, c_mm: np.ndarray, R_mm: float, gain: float) -> np.ndarray:
    v = x_mm - c_mm
    d = float(np.linalg.norm(v)) + 1e-9
    if d >= R_mm:
        return np.zeros(3)
    return gain * (R_mm - d) * (v / d)

# ======================
# NEW: Online Jacobian Estimator (sliding window refit + blending)
# ======================
class OnlineJEstimator:
    """
    Maintains a sliding window of recent (q, x_mm) samples, periodically
    re-fits J with damped least-squares, and blends it with the current J.
    """
    def __init__(self,
                 J_init: np.ndarray,
                 lam: float = DAMPING,
                 window_sec: float = ONLINE_WINDOW_SEC,
                 loop_rate_hz: float = PREEMPT_RATE_HZ,
                 min_pairs: int = ONLINE_MIN_PAIRS,
                 min_singval: float = ONLINE_MIN_SINGVAL,
                 blend_alpha: float = ONLINE_BLEND_ALPHA):
        self.J = np.array(J_init, dtype=float).reshape(3,7)
        self.lam = float(lam)
        self.window_len = max(2, int(round(window_sec * max(1e-3, loop_rate_hz))))
        self.samples: deque[Tuple[np.ndarray, np.ndarray]] = deque(maxlen=self.window_len)
        self.min_pairs = int(min_pairs)
        self.min_singval = float(min_singval)
        self.alpha = float(blend_alpha)
        self.update_count = 0

    def add_sample(self, q: np.ndarray, x_mm: np.ndarray):
        self.samples.append((q.copy(), x_mm.copy()))

    def maybe_update(self) -> bool:
        """Try to refit J from the current window. Returns True if updated."""
        if len(self.samples) < (self.min_pairs + 1):
            return False
        ΔQ, ΔX = build_delta_mats(list(self.samples))
        if ΔQ.shape[1] < self.min_pairs:
            return False
        # local sufficiency check with relaxed thresholds
        try:
            u, s, vh = np.linalg.svd(ΔQ, full_matrices=False)
        except np.linalg.LinAlgError:
            return False
        if s[-1] < self.min_singval:
            return False
        # fit and blend
        J_fit = fit_global_J(ΔQ, ΔX, lam=self.lam)
        self.J = (1.0 - self.alpha) * self.J + self.alpha * J_fit
        self.update_count += 1
        return True

# ======================
# Phase 2 control
# ======================
def control_to_target_mm_timeboxed(J_init: np.ndarray, target_mm: np.ndarray, total_time_s: float):
    """
    Drive toward target_mm and arrive in approximately total_time_s by chunking
    the motion into N evenly spaced steps. Each step is executed with '; dur = step_dur'
    and the Cartesian step is scaled ~1/N_remaining to finish on time.
    """

    # total time must at least allow a single legal chunk
    total_time_s = float(max(MIN_TOTAL_DUR_S, total_time_s))

    # Use WAYPOINT_DUR_S as the TARGET step size
    target_step = float(WAYPOINT_DUR_S)
    # bound the target step to robot-legal range
    target_step = max(MIN_ROBOT_DUR_S, min(MAX_ROBOT_DUR_S, target_step))

    # Pick number of chunks so that each chunk is close to target_step
    # N >= 1 and per-chunk duration stays within [MIN_ROBOT_DUR_S, MAX_ROBOT_DUR_S]
    N = max(1, int(np.ceil(total_time_s / target_step)))

    # Compute the actual per-chunk duration we’ll use
    step_dur = total_time_s / N
    step_dur = max(MIN_ROBOT_DUR_S, min(MAX_ROBOT_DUR_S, step_dur))

    # Pace the loop by the chunk duration so we never queue early
    period = max(1.0 / PREEMPT_RATE_HZ, step_dur)

    # Enable 'p' watcher ONLY during motion
    _watcher_enabled.set()
    try:
        estimator = OnlineJEstimator(
            J_init=J_init,
            lam=DAMPING,
            window_sec=ONLINE_WINDOW_SEC,
            loop_rate_hz=PREEMPT_RATE_HZ,
            min_pairs=ONLINE_MIN_PAIRS,
            min_singval=ONLINE_MIN_SINGVAL,
            blend_alpha=ONLINE_BLEND_ALPHA
        )
        J  = estimator.J
        Jp = pinv_damped(J, DAMPING)

        prev_q: Optional[np.ndarray] = None
        prev_x: Optional[np.ndarray] = None

        t_start = time.time()
        for k in range(1, N + 1):
            if _emergency_evt.is_set():
                break
            t0 = time.time()

            d = query_state()
            if not d:
                print("[CTRL] Telemetry timeout; retrying...")
                time.sleep(period)
                continue

            q, x_mm = parse_arm(d, ACTIVE_SIDE)

            # collect for online J
            if prev_q is not None and prev_x is not None:
                estimator.add_sample(q, x_mm)
            prev_q, prev_x = q, x_mm

            dx   = (target_mm - x_mm)
            dist = float(np.linalg.norm(dx))
            steps_left = max(1, N - k + 1)

            print(f"[CTRL] T_rem≈{max(0.0, total_time_s - (time.time() - t_start)):.2f}s | "
                  f"step {k}/{N} | dist={dist:.1f} mm")

            if dist <= GOAL_TOL_MM:
                print("[CTRL] Target reached within tolerance; holding.")
                break

            # Periodic online-J update (same cadence as before)
            if ENABLE_ONLINE_J and (k % ONLINE_UPDATE_EVERY_STEPS == 0):
                if estimator.maybe_update():
                    J  = estimator.J
                    Jp = pinv_damped(J, DAMPING)
                    if (estimator.update_count % ONLINE_PRINT_EVERY_UPD) == 0:
                        with np.printoptions(precision=3, suppress=True):
                            print(f"[CTRL] Online J updated (#{estimator.update_count}).\nJ ≈\n{J}")

            # --- Time-box scaling in task space ---
            # Aim to reduce the remaining dx by ~1/steps_left this iteration
            alpha = 1.0 / steps_left
            # Keep alpha conservative if we’re close, but allow catching up near the end
            if steps_left <= 2 and dist > GOAL_TOL_MM:
                alpha = min(1.0, max(alpha, 0.5))  # nudge to finish on time
            dx_step = dx * alpha

            # Primary step in joint space
            dq = (Jp @ dx_step).reshape(-1)

            # Nullspace bubble
            gradU = bubble_grad_task_space(x_mm, BUBBLE_CENTER_MM, BUBBLE_RADIUS_MM, BUBBLE_GAIN)
            Nmat = np.eye(7) - (Jp @ J)
            dq  += Nmat @ (J.T @ gradU)

            # Clamp joint change
            m = float(np.linalg.norm(dq, ord=np.inf))
            if m > MAX_DQ_STEP:
                dq *= (MAX_DQ_STEP / (m + 1e-12))

            q_next = q + dq
            if not send_coords_and_go(q_next, step_dur):
                print("[CTRL] Command failed; attempting 'p' then retry.")
                send_and_wait_ack("p", expect_prefix="p")
                time.sleep(0.05)
                if _emergency_evt.is_set():
                    break
                send_coords_and_go(q_next, step_dur)

            # Pace the loop to hit the overall deadline
            # Target wall-clock for this step:
            target_k_time = t_start + k * (total_time_s / N)
            # Also respect the control period
            min_next_time = t0 + period
            sleep_until = max(target_k_time, min_next_time)
            time.sleep(max(0.0, sleep_until - time.time()))

        # Final touch if we’re just outside tolerance after N steps
        if not _emergency_evt.is_set():
            d = query_state()
            if d:
                _, x_mm = parse_arm(d, ACTIVE_SIDE)
                if np.linalg.norm(target_mm - x_mm) > GOAL_TOL_MM:
                    print("[CTRL] Final settle step.")
                    dq = (Jp @ (target_mm - x_mm)).reshape(-1)
                    m  = float(np.linalg.norm(dq, ord=np.inf))
                    if m > MAX_DQ_STEP:
                        dq *= (MAX_DQ_STEP / (m + 1e-12))
                    send_coords_and_go((parse_arm(d, ACTIVE_SIDE)[0] + dq), min(step_dur, 0.15))

    finally:
        _watcher_enabled.clear()  # ensure watcher OFF at prompts/idle

# ======================
# Input parsing helpers (CSV)
# ======================
def parse_csv_ops(s: str):
    toks = [t.strip() for t in s.split(",") if t.strip()]
    return toks

def parse_csv_durs(s: str):
    out = []
    for t in s.split(","):
        t = t.strip()
        if not t:
            continue
        try:
            out.append(float(t))
        except:
            print(f"[CAL] Invalid duration '{t}', defaulting to 1.0s")
            out.append(1.0)
        if out[-1] < 0.05:
            out[-1] = 0.05
    return out

# ======================
# Main
# ======================
def main():
    print(f"[{_ts()}] Bound local UDP socket to {_sock.getsockname()} — sending to {(ROBOT_IP, ROBOT_PORT)}")
    print(f"[{_ts()}] === Phase 1: Calibration (mm) ===")
    print(f"[{_ts()}] Provide TWO comma-separated lists (same length). Example:")
    print(f"[{_ts()}]   Trajectories: a,x,y,z")
    print(f"[{_ts()}]   Durations(s): 2,3,4,1.5")
    print(f"[{_ts()}] Type 'p' + Enter at ANY time during motion phases to PAUSE -> HOME -> EXIT.\n")

    # --- collect lists BEFORE starting the emergency watcher ---
    ops_in  = input(f"[{_ts()}] Trajectories list: ").strip()
    durs_in = input(f"[{_ts()}] Durations list:    ").strip()

    ops  = parse_csv_ops(ops_in)
    durs = parse_csv_durs(durs_in)

    # sanity check & echo back
    print(f"[{_ts()}] [CAL] Parsed trajectories: {ops}")
    print(f"[{_ts()}] [CAL] Parsed durations:    {durs}")

    if len(ops) == 0 or len(ops) != len(durs):
        print(f"[{_ts()}] [CAL] Mismatch or empty lists. Got {len(ops)} ops and {len(durs)} durations.")
        return

    # --- now start the emergency watcher (so it won’t steal the prompts) ---
    threading.Thread(target=emergency_watcher, daemon=True).start()
    print(f"[{_ts()}] [SAFE] Emergency watcher armed. Type 'p' to pause/home/exit.\n")

    # Execute calibration shots
    all_samples: List[Tuple[np.ndarray, np.ndarray]] = []
    for op, dur in zip(ops, durs):
        if _emergency_evt.is_set():
            return
        op_clean = op.strip()
        if not op_clean:
            continue
        print(f"[{_ts()}] [CAL] Executing '{op_clean}' for {dur:.2f}s → g → query → h.")
        logs = run_preset_with_duration_and_log(op_clean, dur, SAMPLE_RATE_HZ)
        for d in logs:
            try:
                q, x = parse_arm(d, ACTIVE_SIDE)
                all_samples.append((q, x))
            except Exception:
                continue

    if _emergency_evt.is_set():
        return

    # Build Δ and fit J
    ΔQ, ΔX = build_delta_mats(all_samples)
    enough, info = sufficiency_check(ΔQ)
    print(f"[{_ts()}] [CAL] Δ pairs: {info.get('n_pairs', 0)}")
    if info.get("singvals"):
        sv = np.array(info["singvals"][:7])
        print(f"[{_ts()}] [CAL] ΔQ singular values (first 7): {np.round(sv, 4)}")
        print(f"[{_ts()}] [CAL] ΔQ condition ~ {info.get('condition'):.2e}")

    if not enough:
        print(f"[{_ts()}] [CAL] Coverage not sufficient. Please add more/longer shots and re-run.")
        return

    J = fit_global_J(ΔQ, ΔX, lam=DAMPING)
    if PRINT_J:
        print(f"[{_ts()}] [CAL] Fitted J (mm per rad):")
        with np.printoptions(precision=3, suppress=True):
            print(J)

    # Phase 2: control
    while not _emergency_evt.is_set():
        # Watchdog OFF during input
        _watcher_enabled.clear()
        targ = input(f"[{_ts()}] Target mm (x,y,z): ").strip()
        if _emergency_evt.is_set():
            break
        try:
            xd = np.array([float(s) for s in targ.replace(' ', '').split(',')], dtype=float)
            if xd.shape[0] != 3:
                raise ValueError
        except Exception:
            print(f"[{_ts()}] Parse error. Try like: 350,200,820")
            continue

        # Ask for total desired time to reach target
        dur_in = input(f"[{_ts()}] Total duration to reach target (s): ").strip()
        try:
            total_dur = float(dur_in)
        except Exception:
            print(f"[{_ts()}] Invalid time. Try a number like 4.0")
            continue
        total_dur = float(max(MIN_TOTAL_DUR_S, min(MAX_TOTAL_DUR_S, total_dur)))

        print(f"[{_ts()}] [CTRL] Moving toward {xd.tolist()} mm with total_time={total_dur:.3f}s...")
        control_to_target_mm_timeboxed(J, xd, total_dur)
        print(f"[{_ts()}] [CTRL] Holding at target. Enter another target, or type 'p' then Enter to exit safely.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[SAFE] KeyboardInterrupt: sending pause, home, exit.")
        try:
            send_and_wait_ack("p", expect_prefix="p")
            send_and_wait_ack("h", expect_prefix="h")
            send_and_wait_ack("e", expect_prefix="e")
        except Exception:
            pass
        sys.exit(0)
