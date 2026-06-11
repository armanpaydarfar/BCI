#!/usr/bin/env python3
# harmony_gaze_calibration.py
#
# Calibration with gaze tracking integration.
# Moves through predefined positions while capturing:
#   - Joint angles (q)
#   - End-effector positions (ee_pos_mm)
#   - Gaze coordinates (gaze_x_norm, gaze_y_norm)
#
# Saves enhanced library: poses_with_gaze.npz

import json, time, socket, threading, sys, os
import numpy as np
from datetime import datetime
from typing import Any, List, Dict, Optional, Tuple
from pylsl import StreamInlet, resolve_stream

# ============================================================
# Network configuration
# ============================================================
ROBOT_IP   = "192.168.2.1"
ROBOT_PORT = 8080
CONTROL_IP = "0.0.0.0"
CONTROL_PORT = 8080
ACK_TIMEOUT_S = 0.35
TEL_TIMEOUT_S = 0.60
CMD_PROCESS_GRACE_S = 0.40

# ============================================================
# Calibration configuration
# ============================================================
ACTIVE_SIDE = "R"
SAMPLE_RATE_HZ = 25.0
HOME_SLEEP_S = 3.0
HOME_LOG_EXTRA_S = 1.0
ARMING_DELAY_S = 0.5
POST_GO_QUIET_S = 0.05
MIN_ROBOT_DUR_S = 1.0
MAX_ROBOT_DUR_S = 10.0
DUR_FMT = ";dur={dur:.3f}"

# Gaze configuration
GAZE_CONFIDENCE_THRESHOLD = 0.7
GAZE_TIMEOUT_S = 0.1  # Quick timeout for non-blocking reads

# ============================================================
# Built-in protocol (same presets)
# ============================================================
PRESET = {
    "p1":   np.array([0.125189, -0.055846, -0.240755, -0.116407, -0.171039, -1.781620, 0.277774]),
    "p2":   np.array([0.127052, -0.060886, -0.304217, -0.094218,  0.128845, -1.627144, 0.277898]),
    "p3":   np.array([0.182464, -0.073947, -0.370300, -0.043051,  0.439869, -1.455687, 0.284454]),
    "a":    np.array([0.218842, -0.0877821, -0.370182,  0.0941577,  1.02211,  -1.04011,  0.315629]),
    "x":    np.array([0.200571, -0.0687415, -0.570200,  0.3901040,  0.95567,  -0.736053,  0.166412]),
    "y":    np.array([0.242212, -0.1226770, -0.350630, -0.4304190,  1.17505,  -0.510246,  0.0386217]),
    "p4": np.array([0.374528, -0.116106, -0.454026, -0.719217, 1.4378, -0.603143, -0.168045]),
    "p5": np.array([0.374426, -0.112933, -0.560625, -1.163042, 0.966008, -0.347849, 0.090975]),
    "p6": np.array([-0.035173, -0.175749, 0.216441, -0.117557, 0.698894, -0.639738, 0.388295]),
    "z":    np.array([0.383150, -0.1416660, -0.132891,  0.2596360,  1.60026,  -0.769595, -0.0552481]),
    "home": np.array([0.136380, -0.054498,  -0.216416, -0.157495,  -0.557142, -1.977339,  0.292421]),
}

PROTOCOL_BLOCKS = os.getenv("CAL_BLOCKS", "P1,P2,P3,A,X,Y,P4,P5,P6,Z")

def build_protocol(blocks: str) -> Tuple[List[np.ndarray], List[float], List[str]]:
    """Builds a simple ordered list of positions, durations, and names."""
    blocks = blocks.upper().split(",")
    tokens, durs, names = [], [], []
    for b in blocks:
        key = b.strip().lower()
        if key not in PRESET:
            print(f"[CAL] WARN: Unknown preset '{b}', skipping.")
            continue
        tokens.append(PRESET[key])
        durs.append(3.0)  # 3s per pose for stable gaze capture
        names.append(key.upper())
    tokens.append(PRESET["home"])
    durs.append(2.0)
    names.append("HOME")
    return tokens, durs, names

# ============================================================
# Networking helpers
# ============================================================
_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
_sock.bind((CONTROL_IP, CONTROL_PORT))
_sock.settimeout(0.5)

_emergency_evt = threading.Event()
_watcher_enabled = threading.Event()

def _ts(): return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def _send(msg: str):
    print(f"[{_ts()}] TX: {msg}")
    _sock.sendto(msg.encode("utf-8"), (ROBOT_IP, ROBOT_PORT))

def _recv(timeout_s: float) -> Optional[str]:
    _sock.settimeout(timeout_s)
    try:
        data, _ = _sock.recvfrom(65535)
        return data.decode("utf-8", errors="ignore")
    except socket.timeout:
        return None

def send_and_wait_ack(msg: str, expect_prefix: Optional[str] = None, timeout: float = ACK_TIMEOUT_S) -> bool:
    _send(msg)
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = _recv(timeout)
        if not r: continue
        if r.startswith("ACK:"):
            print(f"[{_ts()}] {r.strip()}")
            if expect_prefix is None or r.strip().startswith(f"ACK:{expect_prefix}"):
                time.sleep(CMD_PROCESS_GRACE_S)
                return True
    print(f"[{_ts()}] ACK timeout for {msg}")
    return False

def query_state() -> Optional[dict]:
    seq = int(time.time() * 1000) & 0xFFFFFFFF
    _send(f"q;seq={seq}")
    t0 = time.time()
    while time.time() - t0 < TEL_TIMEOUT_S:
        r = _recv(TEL_TIMEOUT_S)
        if not r: continue
        if r.startswith("{"):
            try:
                pkt = json.loads(r)
                if ACTIVE_SIDE == "R":
                    q = np.array(pkt["qR"], dtype=float)
                    ee = np.array(pkt["eeR"]["pos_mm"], dtype=float)
                else:
                    q = np.array(pkt["qL"], dtype=float)
                    ee = np.array(pkt["eeL"]["pos_mm"], dtype=float)
                return {"_t": time.time(), "q": q, "ee": ee}
            except Exception:
                continue
    return None

def send_coords_and_go(q_target: np.ndarray, dur_s: float) -> bool:
    dur_s = float(max(MIN_ROBOT_DUR_S, min(MAX_ROBOT_DUR_S, dur_s)))
    coords = ",".join(f"{v:.6f}" for v in q_target.tolist()) + DUR_FMT.format(dur=dur_s)
    if not send_and_wait_ack(coords): return False
    time.sleep(0.01)
    ok = send_and_wait_ack("g", expect_prefix="g")
    if ok: time.sleep(0.02)
    return ok

def wait_for_home(): time.sleep(HOME_SLEEP_S)

# ============================================================
# Gaze stream management
# ============================================================
class GazeStream:
    def __init__(self, confidence_threshold=GAZE_CONFIDENCE_THRESHOLD):
        self.inlet = None
        self.confidence_threshold = confidence_threshold
        
    def connect(self):
        """Find and connect to Neon gaze stream"""
        print(f"[{_ts()}] [GAZE] Searching for Neon LSL stream...")
        try:
            streams = resolve_stream('type', 'Gaze')
            if not streams:
                print(f"[{_ts()}] [GAZE] ERROR: No gaze stream found!")
                return False
            self.inlet = StreamInlet(streams[0])
            print(f"[{_ts()}] [GAZE] Connected to: {streams[0].name()}")
            return True
        except Exception as e:
            print(f"[{_ts()}] [GAZE] ERROR: {e}")
            return False
    
    def get_latest_gaze(self) -> Optional[Tuple[float, float, float]]:
        """
        Pull the most recent gaze sample(s) from the stream and return the latest valid one.
        Since gaze streams at ~33Hz and we query at 25Hz, there should always be fresh data.
        
        Returns: (x_norm, y_norm, confidence) or None
        """
        if not self.inlet:
            return None
        
        try:
            # Pull all available samples (non-blocking), keep the last valid one
            latest_valid = None
            
            while True:
                sample, _ = self.inlet.pull_sample(timeout=0.0)  # Non-blocking
                if not sample:
                    break  # No more samples available
                
                x_norm = sample[0] / 1600.0
                y_norm = sample[1] / 1200.0
                confidence = sample[15]
                
                if confidence >= self.confidence_threshold:
                    latest_valid = (x_norm, y_norm, confidence)
            
            return latest_valid
            
        except Exception:
            return None

# ============================================================
# Emergency watcher
# ============================================================
def emergency_watcher():
    try:
        while True:
            if not _watcher_enabled.is_set():
                time.sleep(0.05); continue
            s = sys.stdin.readline()
            if s.strip().lower() == "p":
                print("[EMERGENCY] Operator requested PAUSE.")
                _emergency_evt.set()
                try: send_and_wait_ack("p", expect_prefix="p")
                except Exception: pass
                try: send_and_wait_ack("h", expect_prefix="h")
                except Exception: pass
    except Exception:
        _emergency_evt.set()
        try: send_and_wait_ack("p", expect_prefix="p"); send_and_wait_ack("h", expect_prefix="h")
        except Exception: pass
        os._exit(1)

# ============================================================
# Core routine with gaze capture
# ============================================================
def run_sequence_with_gaze(tokens: List[np.ndarray], durs: List[float], 
                           names: List[str], samplerate_hz: float, 
                           gaze_stream: GazeStream) -> List[dict]:
    """
    Capture robot telemetry + synchronized gaze data.
    Returns: logs (list of dicts with q, ee, gaze_x, gaze_y, gaze_conf)
    """
    logs: List[dict] = []
    period = 1.0 / max(1e-6, samplerate_hz)

    for i, (q_target, dur, name) in enumerate(zip(tokens, durs, names)):
        if _emergency_evt.is_set(): break
        
        print(f"\n{'='*70}")
        print(f"[{_ts()}] WAYPOINT {i+1}/{len(tokens)}: {name}")
        print(f"[{_ts()}] Duration: {dur:.1f}s")
        print(f"[{_ts()}] >>> LOOK AT THE END EFFECTOR <<<")
        print(f"{'='*70}\n")
        
        ok = send_coords_and_go(q_target, dur)
        if not ok:
            print(f"[{_ts()}] WARN: Failed to stage waypoint {name}")
            continue
        
        time.sleep(POST_GO_QUIET_S)
        t_start = time.time()
        t_end = t_start + dur
        
        waypoint_samples = 0
        waypoint_gaze_valid = 0
        
        # Capture robot + gaze synchronously
        while time.time() < t_end and not _emergency_evt.is_set():
            # Query robot state
            d_robot = query_state()
            if not d_robot:
                time.sleep(period)
                continue
            
            # Get latest gaze (pulls most recent sample from stream)
            gaze = gaze_stream.get_latest_gaze()
            
            if gaze:
                d_robot["gaze_x"] = gaze[0]
                d_robot["gaze_y"] = gaze[1]
                d_robot["gaze_conf"] = gaze[2]
                waypoint_gaze_valid += 1
            else:
                d_robot["gaze_x"] = np.nan
                d_robot["gaze_y"] = np.nan
                d_robot["gaze_conf"] = np.nan
            
            logs.append(d_robot)
            waypoint_samples += 1
            time.sleep(period)
        
        print(f"[{_ts()}] Waypoint {name} complete: {waypoint_samples} samples, "
              f"{waypoint_gaze_valid} with valid gaze "
              f"({100*waypoint_gaze_valid/max(1,waypoint_samples):.1f}%)\n")

    # Final home
    print(f"\n{'='*70}")
    print(f"[{_ts()}] RETURNING HOME...")
    print(f"{'='*70}\n")
    
    send_and_wait_ack("h", expect_prefix="h")
    t_end = time.time() + HOME_LOG_EXTRA_S
    while time.time() < t_end:
        d_robot = query_state()
        if d_robot:
            gaze = gaze_stream.get_latest_gaze()
            if gaze:
                d_robot["gaze_x"] = gaze[0]
                d_robot["gaze_y"] = gaze[1]
                d_robot["gaze_conf"] = gaze[2]
            else:
                d_robot["gaze_x"] = np.nan
                d_robot["gaze_y"] = np.nan
                d_robot["gaze_conf"] = np.nan
            logs.append(d_robot)
        time.sleep(period)
    
    wait_for_home()
    
    total_valid_gaze = sum(1 for log in logs if not np.isnan(log.get("gaze_x", np.nan)))
    print(f"\n{'='*70}")
    print(f"[{_ts()}] CALIBRATION COMPLETE")
    print(f"[{_ts()}] Total samples: {len(logs)}")
    print(f"[{_ts()}] Valid gaze: {total_valid_gaze} ({100*total_valid_gaze/max(1,len(logs)):.1f}%)")
    print(f"{'='*70}\n")
    
    return logs

# ============================================================
# Main
# ============================================================
def main():
    print(f"\n{'='*70}")
    print(f"  GAZE + ROBOT CALIBRATION")
    print(f"{'='*70}\n")
    print(f"[{_ts()}] Bound UDP socket {_sock.getsockname()} → {(ROBOT_IP, ROBOT_PORT)}")
    print(f"[{_ts()}] Type 'p' + Enter at ANY time to PAUSE + HOME.\n")
    
    # Connect to gaze stream
    gaze_stream = GazeStream()
    if not gaze_stream.connect():
        print(f"\n[{_ts()}] [ERROR] Cannot proceed without gaze stream.")
        print(f"[{_ts()}] Ensure:")
        print(f"  1. Neon Companion app is running")
        print(f"  2. LSL streaming is enabled")
        print(f"  3. Both devices on same network\n")
        return
    
    tokens, durs, names = build_protocol(PROTOCOL_BLOCKS)
    
    print(f"\n{'='*70}")
    print(f"  CALIBRATION PROTOCOL")
    print(f"{'='*70}")
    print(f"[{_ts()}] Waypoints: {len(tokens)}")
    print(f"[{_ts()}] Total duration: ~{sum(durs):.1f}s")
    print(f"[{_ts()}] Sequence: {' → '.join(names)}")
    print(f"{'='*70}\n")
    
    print("INSTRUCTIONS:")
    print("  • Put on your Neon eye tracking glasses")
    print("  • Position yourself comfortably near the robot")
    print("  • The robot will move to multiple waypoints")
    print("  • KEEP YOUR EYES ON THE END EFFECTOR at all times")
    print("  • Try not to blink excessively during movements")
    print("  • Type 'p' + Enter to emergency stop\n")
    
    input("Press Enter when ready to start calibration...")
    
    print(f"\n[{_ts()}] Starting in 3 seconds...")
    time.sleep(1)
    print(f"[{_ts()}] 2...")
    time.sleep(1)
    print(f"[{_ts()}] 1...")
    time.sleep(1)
    print(f"[{_ts()}] GO!\n")
    
    threading.Thread(target=emergency_watcher, daemon=True).start()
    _watcher_enabled.set()

    logs = run_sequence_with_gaze(tokens, durs, names, SAMPLE_RATE_HZ, gaze_stream)
    
    if not logs:
        print(f"[{_ts()}] [ERROR] No data collected, aborting.")
        return
    
    # Extract arrays
    T = np.array([d["_t"] for d in logs])
    Q = np.vstack([d["q"] for d in logs])
    X = np.vstack([d["ee"] for d in logs])
    G = np.column_stack([
        [d["gaze_x"] for d in logs],
        [d["gaze_y"] for d in logs],
        [d["gaze_conf"] for d in logs]
    ])
    
    # Save enhanced library
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"poses_with_gaze_{stamp}.npz"
    # Output contract (consumed by `harmony_online_control.py::load_library`):
    #   - T: (num_samples,) timestamps (seconds)
    #   - Q: (num_samples, ...) joint angles (radians)
    #   - X: (num_samples, 3) end-effector positions (mm)
    #   - G: (num_samples, 3) gaze coords + confidence
    #        * gaze_x, gaze_y are normalized to [0,1]
    #        * third column is gaze_conf (used as validity gating later)
    np.savez_compressed(out_path, 
                        T=T, Q=Q, X=X, G=G,
                        meta=dict(side=ACTIVE_SIDE,
                                  sample_rate_hz=SAMPLE_RATE_HZ,
                                  gaze_confidence_threshold=GAZE_CONFIDENCE_THRESHOLD,
                                  units=dict(X="mm", Q="rad", G="normalized_0_to_1")))
    
    print(f"\n{'='*70}")
    print(f"  CALIBRATION SAVED")
    print(f"{'='*70}")
    print(f"[{_ts()}] File: {out_path}")
    print(f"[{_ts()}] Contents:")
    print(f"  - T: Timestamps ({T.shape})")
    print(f"  - Q: Joint angles ({Q.shape}) [rad]")
    print(f"  - X: EE positions ({X.shape}) [mm]")
    print(f"  - G: Gaze coords ({G.shape}) [normalized 0-1]")
    
    # Quick stats
    valid_gaze = np.sum(~np.isnan(G[:, 0]))
    print(f"\n[{_ts()}] Quality: {valid_gaze}/{len(G)} samples with valid gaze "
          f"({100*valid_gaze/len(G):.1f}%)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[SAFE] KeyboardInterrupt — homing.")
        try:
            send_and_wait_ack("p", expect_prefix="p")
            send_and_wait_ack("h", expect_prefix="h")
        except Exception:
            pass
        sys.exit(0)