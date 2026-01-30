#!/usr/bin/env python3
# simple_posemap_exec.py
#
# Very simple "go to nearest pose" controller + telemetry logging & plots.
# Saved plots per move (when SAVE_LOGS=True):
#   1) EE position x,y,z (with dotted x_d,y_d,z_d)
#   2) EE velocity x_dot,y_dot,z_dot (from smoothed EE positions)
#   3) Joint velocities dq1..dq7 (from telemetry, with 0-line reference)
#   4) Joint positions q1..q7 (with dotted target q_d lines)
#
# Now also supports "vision" mode:
#   • User types 'v' → wait 2 s → collect 1 s gaze → median gaze (x,y) in [0,1]
#   • Find nearest library sample in gaze space G (ignoring NaNs)
#   • Move to that sample's X/Q.

import sys, socket, json, time, os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from pylsl import StreamInlet, resolve_stream  # For gaze LSL

ROBOT_IP, ROBOT_PORT = "192.168.2.1", 8080
CONTROL_IP, CONTROL_PORT = "0.0.0.0", 8080

ACK_TIMEOUT_S = 0.15
CMD_PROCESS_GRACE_S = 0.30
ACTIVE_SIDE = "R"   # "R" → use eeR, dqR, qR;  "L" → eeL, dqL, qL

# Telemetry capture parameters
TELEMETRY_POLL_HZ = 40.0
POST_MOVE_GRACE_S = 0.15   # existing grace
CAPTURE_AFTER_S    = 1.00  # extra time beyond duration to see settling

# Logging / plotting toggle
LOG_DIR = "telemetry_logs"
SAVE_LOGS = True  # <-- Toggle this ON/OFF to save plots & files

# Smoothing cutoff for EE velocity (EMA on positions)
EE_VEL_EMA_FC_HZ = 1.0

# How to compute EE velocity: "slope" (local linear fit) or "gradient"
EE_VEL_METHOD = "gradient"

# Window (in seconds) for local slope fitting
EE_VEL_WIN_SEC = 0.5

# Gaze configuration (normalize pixels to [0,1] like calibration)
GAZE_CONFIDENCE_THRESHOLD = 0.7
GAZE_TIMEOUT_S = 0.1  # seconds
GAZE_SAMPLE_WIDTH  = 1600.0
GAZE_SAMPLE_HEIGHT = 1200.0

# ---------------- UDP ----------------

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((CONTROL_IP, CONTROL_PORT))
sock.settimeout(0.5)

def ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def utc_stamp_filename():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]

def udp_send(msg):
    print(f"[{ts()}] TX: {msg}")
    sock.sendto(msg.encode(), (ROBOT_IP, ROBOT_PORT))

def udp_recv(timeout):
    sock.settimeout(timeout)
    try:
        data,_ = sock.recvfrom(65535)
        return data.decode(errors="ignore")
    except socket.timeout:
        return None

def send_and_wait_ack(msg, expect=None, timeout=ACK_TIMEOUT_S):
    udp_send(msg)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        r = udp_recv(timeout)
        if r and r.startswith("ACK:"):
            print(f"[{ts()}] {r.strip()}")
            time.sleep(CMD_PROCESS_GRACE_S)
            if expect is None or r.startswith(f"ACK:{expect}"):
                return True
    print(f"[{ts()}] ACK timeout: {msg}")
    return False

# ---------- Telemetry parsers ----------

def extract_pose_dq_q(pkt, side="R"):
    """
    Return:
      pos_mm: np.array([3]) or None
      dq_rad_s: np.array([7]) or None
      q_rad: np.array([7]) or None
    Uses eeR/eeL for position, dqR/dqL and qR/qL for joint data.
    """
    side = side.upper()
    ee_key = "eeR" if side == "R" else "eeL"
    dq_key = "dqR" if side == "R" else "dqL"
    q_key  = "qR"  if side == "R" else "qL"

    pos_mm, dq, q = None, None, None

    try:
        if ee_key in pkt and isinstance(pkt[ee_key], dict) and "pos_mm" in pkt[ee_key]:
            pos_mm = np.array(pkt[ee_key]["pos_mm"], dtype=float)
    except Exception:
        pos_mm = None

    try:
        if dq_key in pkt:
            arr = np.array(pkt[dq_key], dtype=float).ravel()
            if arr.size >= 7:
                dq = arr[:7]
    except Exception:
        dq = None

    try:
        if q_key in pkt:
            arr = np.array(pkt[q_key], dtype=float).ravel()
            if arr.size >= 7:
                q = arr[:7]
    except Exception:
        q = None

    # Fallbacks for dq only (rare)
    if dq is None:
        for k in ("dq", "jvel", "qdot", "dq_rad_s"):
            if k in pkt:
                try:
                    arr = np.array(pkt[k], dtype=float).ravel()
                    if arr.size >= 7:
                        dq = arr[:7]
                        break
                except Exception:
                    pass

    return pos_mm, dq, q

def query_state_full():
    """Query one telemetry packet; return (pos_mm, dq_rad_s, q_rad) or (None, None, None)."""
    seq = int(time.time()*1000) & 0xFFFFFFFF
    udp_send(f"q;seq={seq}")
    t0 = time.time()
    while time.time() - t0 < 0.6:
        r = udp_recv(0.6)
        if not r or not r.startswith("{"):
            continue
        try:
            pkt = json.loads(r)
            pos_mm, dq, q = extract_pose_dq_q(pkt, side=ACTIVE_SIDE)
            if pos_mm is not None:
                return pos_mm, dq, q
        except Exception:
            continue
    print(f"[{ts()}] Telemetry error (no pose).")
    return None, None, None

def query_position_only():
    pos, _, _ = query_state_full()
    return pos

def go_home():
    print(f"[{ts()}] HOME...")
    send_and_wait_ack("h", expect="h")
    print(f"[{ts()}] HOME done.")

# ---------------- Library ----------------

def load_library(path):
    z = np.load(path, allow_pickle=True)
    X = z["X"]    # positions mm
    Q = z["Q"]    # joint rad
    G = z["G"] if "G" in z.files else None  # gaze (normalized 0–1), shape (N,>=2) or None
    return X, Q, G

def nearest_idx(X, x):
    d = np.linalg.norm(X - x[None, :], axis=1)
    return int(np.argmin(d)), float(np.min(d))

# ---------------- Gaze (LSL) helpers ----------------

class GazeStream:
    def __init__(self, confidence_threshold=GAZE_CONFIDENCE_THRESHOLD):
        self.inlet = None
        self.confidence_threshold = confidence_threshold

    def connect(self):
        print(f"[{ts()}] [GAZE] Searching for Neon LSL stream...")
        try:
            streams = resolve_stream('type', 'Gaze')
            if not streams:
                print(f"[{ts()}] [GAZE] ERROR: No gaze stream found.")
                return False
            self.inlet = StreamInlet(streams[0])
            print(f"[{ts()}] [GAZE] Connected to: {streams[0].name()}")
            return True
        except Exception as e:
            print(f"[{ts()}] [GAZE] ERROR: {e}")
            return False

    def average_gaze_over_window(self, dur_s=1.0):
        """
        For dur_s seconds, accumulate valid gaze samples and return
        (x_norm_median, y_norm_median) in [0,1] if any valid samples exist,
        otherwise return None.

        Normalization matches harmony_gaze_calibration.py:
          x_norm = sample[0] / 1600.0
          y_norm = sample[1] / 1200.0

        Uses median instead of mean for robustness to outliers.
        """
        if self.inlet is None:
            return None

        t0 = time.time()
        xs, ys = [], []
        while time.time() - t0 < dur_s:
            try:
                sample, _ = self.inlet.pull_sample(timeout=GAZE_TIMEOUT_S)
            except Exception:
                sample = None
            if not sample:
                continue

            try:
                x_raw = float(sample[0])
                y_raw = float(sample[1])
                conf  = float(sample[15])

                x_norm = x_raw / GAZE_SAMPLE_WIDTH
                y_norm = y_raw / GAZE_SAMPLE_HEIGHT
            except Exception:
                continue

            if conf >= self.confidence_threshold:
                xs.append(x_norm)
                ys.append(y_norm)

        if len(xs) == 0:
            return None
        return float(np.median(xs)), float(np.median(ys))


def flush_gaze_buffer(inlet):
    """Empty all pending samples from an LSL inlet."""
    if inlet is None:
        return
    while True:
        try:
            sample, _ = inlet.pull_sample(timeout=0.0)
            if not sample:
                break
        except Exception:
            break


def nearest_idx_gaze(G, gaze_xy):
    """
    G: array of shape (N, >=2), gaze entries [x_norm, y_norm, gaze_conf?]
       Some rows may have NaNs (invalid gaze during calibration).
    gaze_xy: (x_norm, y_norm)
    Returns: (idx_global, dist) where idx_global indexes into X/Q/G.
    """
    if G is None or G.size == 0:
        return 0, float("nan")

    gx = G[:, 0]
    gy = G[:, 1]
    valid_mask = np.isfinite(gx) & np.isfinite(gy)
    if not np.any(valid_mask):
        print(f"[{ts()}] [GAZE] No valid rows in gaze library (all NaNs).")
        return 0, float("nan")

    g2 = np.column_stack([gx[valid_mask], gy[valid_mask]])  # (M,2)
    target = np.asarray(gaze_xy, dtype=float).ravel()
    d = np.linalg.norm(g2 - target[None, :], axis=1)

    best_local = int(np.argmin(d))
    valid_indices = np.flatnonzero(valid_mask)
    idx_global = int(valid_indices[best_local])
    return idx_global, float(d[best_local])

# ---------------- Telemetry capture & math ----------------

def collect_telemetry(duration_s):
    """
    Poll telemetry for (duration_s + POST_MOVE_GRACE_S + CAPTURE_AFTER_S)
    so we can see settling after the commanded duration.
    """
    if duration_s <= 0:
        return np.array([]), np.zeros((0, 3)), np.zeros((0, 7)), np.zeros((0, 7))

    dt = 1.0 / TELEMETRY_POLL_HZ
    t0 = time.perf_counter()
    next_t = t0
    ts_list, pos_list, dq_list, q_list = [], [], [], []

    # Initial sample
    pos0, dq0, q0 = query_state_full()
    t_now = time.perf_counter()
    if pos0 is not None:
        ts_list.append(t_now - t0)
        pos_list.append(pos0)
        dq_list.append(dq0 if dq0 is not None else np.full(7, np.nan))
        q_list.append(q0 if q0 is not None else np.full(7, np.nan))

    # Loop
    total_window = duration_s + POST_MOVE_GRACE_S + CAPTURE_AFTER_S
    while (time.perf_counter() - t0) < total_window:
        next_t += dt
        sleep_amt = next_t - time.perf_counter()
        if sleep_amt > 0:
            time.sleep(sleep_amt)
        pos, dq, q = query_state_full()
        t_rel = time.perf_counter() - t0
        if pos is not None:
            ts_list.append(t_rel)
            pos_list.append(pos)
            dq_list.append(dq if dq is not None else np.full(7, np.nan))
            q_list.append(q if q is not None else np.full(7, np.nan))

    if len(ts_list) == 0:
        return np.array([]), np.zeros((0, 3)), np.zeros((0, 7)), np.zeros((0, 7))
    return np.asarray(ts_list), np.vstack(pos_list), np.vstack(dq_list), np.vstack(q_list)

def ema_smooth_positions(pos_mm, t_rel, fc_hz=EE_VEL_EMA_FC_HZ):
    """Exponential moving average smoothing (component-wise) with variable dt."""
    if len(t_rel) == 0:
        return pos_mm
    pos_s = pos_mm.copy()
    pos_s[0] = pos_mm[0]
    tau = 1.0 / (2*np.pi*fc_hz)
    for i in range(1, len(t_rel)):
        dt = max(1e-9, t_rel[i] - t_rel[i-1])
        alpha = dt / (tau + dt)
        pos_s[i] = alpha * pos_mm[i] + (1.0 - alpha) * pos_s[i-1]
    return pos_s

def finite_difference_vel(t_rel, XYZ):
    """Compute per-axis velocities via gradient; t_rel in seconds, XYZ in mm."""
    if len(t_rel) < 2:
        return np.zeros_like(XYZ)
    vx = np.gradient(XYZ[:, 0], t_rel)
    vy = np.gradient(XYZ[:, 1], t_rel)
    vz = np.gradient(XYZ[:, 2], t_rel)
    return np.column_stack([vx, vy, vz])

def vel_local_linear(t_rel, XYZ, win_sec=0.25):
    """Local linear slope fit for velocity (robust to jitter)."""
    N = len(t_rel)
    if N < 2:
        return np.zeros_like(XYZ)
    half = 0.5 * max(1e-6, win_sec)
    V = np.zeros_like(XYZ)
    for i in range(N):
        t0 = t_rel[i] - half
        t1 = t_rel[i] + half
        mask = (t_rel >= t0) & (t_rel <= t1)
        idx = np.nonzero(mask)[0]
        if idx.size < 3:
            if 0 < i < N-1:
                dt = max(1e-9, t_rel[i+1] - t_rel[i-1])
                V[i] = (XYZ[i+1] - XYZ[i-1]) / dt
            else:
                V[i] = 0.0
            continue
        t_seg = t_rel[idx]
        t_c = t_seg - t_rel[i]
        w = np.clip(1.0 - np.abs(t_c)/half, 0.0, 1.0)
        A = np.column_stack([t_c, np.ones_like(t_c)])
        W = np.diag(w)
        AtW = A.T @ W
        H = AtW @ A
        try:
            H_inv = np.linalg.inv(H)
            for k in range(3):
                y = XYZ[idx, k]
                beta = H_inv @ (AtW @ y)
                V[i, k] = beta[0]
        except np.linalg.LinAlgError:
            for k in range(3):
                m, _ = np.linalg.lstsq(A, XYZ[idx, k], rcond=None)[0]
                V[i, k] = m
    if N >= 2:
        V[0] = V[1]; V[-1] = V[-2]
    return V

def ensure_log_dir():
    if not SAVE_LOGS:
        return
    os.makedirs(LOG_DIR, exist_ok=True)

# ---------------- Plotting ----------------

def plot_positions(t_rel, pos_mm, goal_xyz_mm, duration_s, out_path):
    if len(t_rel) == 0:
        print(f"[{ts()}] No telemetry to plot (positions).")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_rel, pos_mm[:, 0], label="x (mm)")
    ax.plot(t_rel, pos_mm[:, 1], label="y (mm)")
    ax.plot(t_rel, pos_mm[:, 2], label="z (mm)")
    ax.hlines(goal_xyz_mm[0], t_rel[0], t_rel[-1], linestyles="dotted", label="x_d (mm)")
    ax.hlines(goal_xyz_mm[1], t_rel[0], t_rel[-1], linestyles="dotted", label="y_d (mm)")
    ax.hlines(goal_xyz_mm[2], t_rel[0], t_rel[-1], linestyles="dotted", label="z_d (mm)")
    ax.axvline(duration_s, linestyle="--", alpha=0.4)
    ax.set_title("End Effector Position vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (mm)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{ts()}] Saved position plot → {out_path}")

def plot_ee_velocity(t_rel, vel_mm_s, duration_s, out_path):
    if len(t_rel) == 0:
        print(f"[{ts()}] No telemetry to plot (EE velocity).")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_rel, vel_mm_s[:, 0], label="x_dot (mm/s)")
    ax.plot(t_rel, vel_mm_s[:, 1], label="y_dot (mm/s)")
    ax.plot(t_rel, vel_mm_s[:, 2], label="z_dot (mm/s)")
    ax.hlines(0.0, t_rel[0], t_rel[-1], linestyles="dashed", alpha=0.5)  # zero reference
    ax.axvline(duration_s, linestyle="--", alpha=0.4)
    ax.set_title("End Effector Velocity vs Time (smoothed positions)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (mm/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{ts()}] Saved EE velocity plot → {out_path}")

def plot_joint_velocities(t_rel, dq_rad_s, duration_s, out_path):
    if len(t_rel) == 0 or dq_rad_s.size == 0:
        print(f"[{ts()}] No telemetry to plot (joint velocities).")
        return
    dq_valid = np.copy(dq_rad_s)
    dq_valid[np.isnan(dq_valid)] = 0.0
    fig, ax = plt.subplots(figsize=(10, 5))
    for j in range(min(7, dq_valid.shape[1])):
        ax.plot(t_rel, dq_valid[:, j], label=f"dq{j+1} (rad/s)")
    ax.hlines(0.0, t_rel[0], t_rel[-1], linestyles="dashed", alpha=0.5)  # zero reference
    ax.axvline(duration_s, linestyle="--", alpha=0.4)
    ax.set_title("Joint Velocities vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint velocity (rad/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=9)
    # Expand y if too tiny
    ymin, ymax = np.nanmin(dq_valid), np.nanmax(dq_valid)
    if np.isfinite(ymin) and np.isfinite(ymax):
        yr = ymax - ymin
        if yr < 0.1:
            mean = 0.5*(ymax+ymin)
            pad = max(0.05, 0.5*max(abs(mean), 0.1))
            ax.set_ylim(mean-pad, mean+pad)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{ts()}] Saved joint velocity plot → {out_path}")

def plot_joint_positions(t_rel, q_rad, q_target_rad, duration_s, out_path):
    if len(t_rel) == 0 or q_rad.size == 0:
        print(f"[{ts()}] No telemetry to plot (joint positions).")
        return
    q_valid = np.copy(q_rad)
    fig, ax = plt.subplots(figsize=(10, 5))
    for j in range(min(7, q_valid.shape[1])):
        ax.plot(t_rel, q_valid[:, j], label=f"q{j+1} (rad)")
        ax.hlines(q_target_rad[j], t_rel[0], t_rel[-1], linestyles="dotted", alpha=0.6)
    ax.hlines(0.0, t_rel[0], t_rel[-1], linestyles="dashed", alpha=0.4)  # zero reference
    ax.axvline(duration_s, linestyle="--", alpha=0.4)
    ax.set_title("Joint Positions vs Time (targets dotted)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint position (rad)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{ts()}] Saved joint position plot → {out_path}")

# ---------------- User interface ----------------

def ask_goal_and_duration():
    """
    Returns:
      mode, goal_xyz, dur

    mode ∈ {"cartesian", "home", "vision", "quit"}
    - "cartesian": goal_xyz is np.array([x,y,z]), dur is float
    - "home":      goal_xyz is NaN array, dur is 0.0
    - "vision":    goal_xyz is None (to be determined from gaze), dur is float
    - "quit":      goal_xyz, dur are None
    """
    s = input("Enter goal 'x,y,z' (mm), or 'v' for gaze, or 'home', or 'quit': ").strip().lower()
    if s in ("quit", "q", "exit"):
        return "quit", None, None
    if s in ("home", "h"):
        return "home", np.array([np.nan]*3), 0.0
    if s in ("v", "vision"):
        try:
            dur = float(input("Duration (s): ").strip())
        except Exception:
            print("Bad duration, try again.")
            return ask_goal_and_duration()
        return "vision", None, dur

    # Cartesian x,y,z
    try:
        xyz = np.array([float(v) for v in s.split(",")], dtype=float)
        dur = float(input("Duration (s): ").strip())
        return "cartesian", xyz, dur
    except Exception:
        print("Bad input, try again.")
        return ask_goal_and_duration()

# ---------------- Main ----------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_posemap_exec.py poses.npz")
        sys.exit(1)

    ensure_log_dir()
    X, Q, G = load_library(sys.argv[1])
    has_gaze = (G is not None) and (G.shape[1] >= 2)
    print(f"[{ts()}] Loaded library with {len(X)} samples.")
    if has_gaze:
        print(f"[{ts()}] Library has gaze entries: G shape = {G.shape}")
    else:
        print(f"[{ts()}] Library has no gaze entries (G). Vision mode 'v' will be disabled.")

    # Initialize gaze stream if we have gaze-enabled library
    gaze_stream = None
    if has_gaze:
        gaze_stream = GazeStream()
        if not gaze_stream.connect():
            print(f"[{ts()}] WARNING: Could not connect to gaze stream. Vision mode 'v' disabled.")
            gaze_stream = None

    while True:
        mode, goal, dur = ask_goal_and_duration()
        if mode == "quit":
            print("Exiting.")
            break

        if mode == "home":
            go_home()
            continue

        # -------- CARTESIAN MODE: x,y,z input --------
        if mode == "cartesian":
            print(f"[{ts()}] Querying current EE...")
            x_now = query_position_only()
            if x_now is None:
                print("No telemetry, skip.")
                continue

            idx, d = nearest_idx(X, goal)
            q_target = Q[idx]
            print(f"[{ts()}] Nearest node {idx} (dist {d:.1f} mm)")
            print(f"q_target: {np.round(q_target, 4).tolist()}")

            coords = ",".join(f"{v:.6f}" for v in q_target) + f";dur={dur:.3f}"
            ok1 = send_and_wait_ack(coords)
            ok2 = send_and_wait_ack("g", expect="g")
            print(f"[{ts()}] Command {'OK' if (ok1 and ok2) else 'ERR'}")

            print(f"[{ts()}] Capturing telemetry for "
                  f"{dur:.2f}s + {POST_MOVE_GRACE_S:.2f}s + {CAPTURE_AFTER_S:.2f}s …")
            t_rel, pos_mm, dq_rad_s, q_rad = collect_telemetry(dur)

            pos_s = ema_smooth_positions(pos_mm, t_rel, fc_hz=EE_VEL_EMA_FC_HZ)

            if EE_VEL_METHOD.lower() == "slope":
                vel_mm_s = vel_local_linear(t_rel, pos_s, win_sec=EE_VEL_WIN_SEC)
            else:
                vel_mm_s = finite_difference_vel(t_rel, pos_s)

            if SAVE_LOGS and len(t_rel) > 0:
                stamp = utc_stamp_filename()
                base = os.path.join(LOG_DIR, f"telemetry_{stamp}")
                plot_positions(t_rel, pos_mm, goal, dur, base + "_pos.png")
                plot_ee_velocity(t_rel, vel_mm_s, dur, base + "_eevel.png")
                plot_joint_velocities(t_rel, dq_rad_s, dur, base + "_dq.png")
                plot_joint_positions(t_rel, q_rad, q_target, dur, base + "_qpos.png")
            continue

        # -------- VISION MODE: 'v' opcode --------
        if mode == "vision":
            if (not has_gaze) or (gaze_stream is None):
                print(f"[{ts()}] ERROR: Vision mode requested but gaze is unavailable.")
                continue

            print(f"[{ts()}] Vision mode: hold gaze on desired target.")
            print(f"[{ts()}] Flushing gaze buffer...")
            flush_gaze_buffer(gaze_stream.inlet)

            print(f"[{ts()}] Waiting 2.0s before sampling gaze...")
            time.sleep(2.0)

            print(f"[{ts()}] Flushing gaze buffer before averaging...")
            flush_gaze_buffer(gaze_stream.inlet)

            print(f"[{ts()}] Averaging gaze for 1.0s...")
            g_avg = gaze_stream.average_gaze_over_window(dur_s=1.0)
            if g_avg is None:
                print(f"[{ts()}] No valid gaze samples collected. Try again.")
                continue

            print(f"[{ts()}] Gaze median (x_norm, y_norm) = ({g_avg[0]:.3f}, {g_avg[1]:.3f})")
            idx_g, d_g = nearest_idx_gaze(G, g_avg)
            q_target = Q[idx_g]
            goal = X[idx_g]  # EE position (mm) for logging/plots

            print(f"[{ts()}] Nearest library entry in gaze space: idx {idx_g}, dist {d_g:.4f}")
            print(f"[{ts()}] Target EE (mm): {np.round(goal, 1).tolist()}")
            print(f"[{ts()}] q_target: {np.round(q_target, 4).tolist()}")

            coords = ",".join(f"{v:.6f}" for v in q_target) + f";dur={dur:.3f}"
            ok1 = send_and_wait_ack(coords)
            ok2 = send_and_wait_ack("g", expect="g")
            print(f"[{ts()}] Command {'OK' if (ok1 and ok2) else 'ERR'}")

            print(f"[{ts()}] Capturing telemetry for "
                  f"{dur:.2f}s + {POST_MOVE_GRACE_S:.2f}s + {CAPTURE_AFTER_S:.2f}s …")
            t_rel, pos_mm, dq_rad_s, q_rad = collect_telemetry(dur)

            pos_s = ema_smooth_positions(pos_mm, t_rel, fc_hz=EE_VEL_EMA_FC_HZ)
            if EE_VEL_METHOD.lower() == "slope":
                vel_mm_s = vel_local_linear(t_rel, pos_s, win_sec=EE_VEL_WIN_SEC)
            else:
                vel_mm_s = finite_difference_vel(t_rel, pos_s)

            if SAVE_LOGS and len(t_rel) > 0:
                stamp = utc_stamp_filename()
                base = os.path.join(LOG_DIR, f"telemetry_{stamp}")
                plot_positions(t_rel, pos_mm, goal, dur, base + "_pos.png")
                plot_ee_velocity(t_rel, vel_mm_s, dur, base + "_eevel.png")
                plot_joint_velocities(t_rel, dq_rad_s, dur, base + "_dq.png")
                plot_joint_positions(t_rel, q_rad, q_target, dur, base + "_qpos.png")
            continue

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[SAFE EXIT] Sending pause+home")
        send_and_wait_ack("p", expect="p")
        send_and_wait_ack("h", expect="h")
