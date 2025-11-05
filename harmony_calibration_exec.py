#!/usr/bin/env python3
# harmony_calibration_posemap.py
#
# Simple calibration executable.
# Moves through predefined joint-space positions, logs telemetry (q, ee_pos_mm),
# and saves a library mapping joint angles ↔ end-effector positions.
#
# This replaces harmony_calibration_exec.py (no Jacobians, no velocity regression).

import json, time, socket, threading, sys, os
import numpy as np
from datetime import datetime
from typing import Any, List, Dict, Optional, Tuple

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
TELEMETRY_PRINT_EVERY = 1
MIN_ROBOT_DUR_S = 1.0
MAX_ROBOT_DUR_S = 10.0
DUR_FMT = ";dur={dur:.3f}"

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
    "z":    np.array([0.383150, -0.1416660, -0.132891,  0.2596360,  1.60026,  -0.769595, -0.0552481]),
    "home": np.array([0.136380, -0.054498,  -0.216416, -0.157495,  -0.557142, -1.977339,  0.292421]),
}

PROTOCOL_BLOCKS = os.getenv("CAL_BLOCKS", "NEAR,A,X,Y,Z")

def build_protocol(blocks: str) -> Tuple[List[np.ndarray], List[float]]:
    """Builds a simple ordered list of positions and durations."""
    blocks = blocks.upper().split(",")
    tokens, durs = [], []
    for b in blocks:
        key = b.strip().lower()
        if key not in PRESET:
            print(f"[CAL] WARN: Unknown preset '{b}', skipping.")
            continue
        tokens.append(PRESET[key])
        durs.append(2.0)  # default 2 s per pose (adjust later if needed)
    tokens.append(PRESET["home"])
    durs.append(2.0)
    return tokens, durs

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
# Core routine
# ============================================================
def run_sequence(tokens: List[np.ndarray], durs: List[float], samplerate_hz: float) -> List[dict]:
    logs: List[dict] = []
    period = 1.0 / max(1e-6, samplerate_hz)

    for i, (q_target, dur) in enumerate(zip(tokens, durs)):
        if _emergency_evt.is_set(): break
        print(f"[{_ts()}] [CAL] Step {i+1}/{len(tokens)}: move for {dur:.2f}s")
        ok = send_coords_and_go(q_target, dur)
        if not ok:
            print(f"[{_ts()}] WARN: failed to stage step {i+1}")
            continue
        time.sleep(POST_GO_QUIET_S)
        t_end = time.time() + dur
        while time.time() < t_end and not _emergency_evt.is_set():
            d = query_state()
            if d: logs.append(d)
            time.sleep(period)

    # Final home
    print(f"[{_ts()}] [CAL] Homing...")
    send_and_wait_ack("h", expect_prefix="h")
    t_end = time.time() + HOME_LOG_EXTRA_S
    while time.time() < t_end:
        d = query_state()
        if d: logs.append(d)
        time.sleep(period)
    wait_for_home()
    print(f"[{_ts()}] [CAL] Complete. Collected {len(logs)} samples.")
    return logs

# ============================================================
# Main
# ============================================================
def main():
    print(f"[{_ts()}] Bound UDP socket {_sock.getsockname()} → {(ROBOT_IP, ROBOT_PORT)}")
    print(f"[{_ts()}] === SIMPLE CALIBRATION ===")
    print(f"[{_ts()}] Type 'p' + Enter at ANY time to PAUSE + HOME.\n")

    tokens, durs = build_protocol(PROTOCOL_BLOCKS)
    print(f"[{_ts()}] Prepared {len(tokens)} waypoints, total time ≈ {sum(durs):.1f}s.")

    threading.Thread(target=emergency_watcher, daemon=True).start()
    _watcher_enabled.set()
    print(f"[{_ts()}] [SAFE] Emergency watcher armed.\n")

    logs = run_sequence(tokens, durs, SAMPLE_RATE_HZ)
    if not logs:
        print(f"[{_ts()}] [CAL] No data collected, aborting.")
        return

    # Flatten logs
    T = np.array([d["_t"] for d in logs])
    Q = np.vstack([d["q"] for d in logs])
    X = np.vstack([d["ee"] for d in logs])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"calib_pose_library_{stamp}.npz"
    np.savez_compressed(out_path, T=T, Q=Q, X=X,
                        meta=dict(side=ACTIVE_SIDE,
                                  sample_rate_hz=SAMPLE_RATE_HZ,
                                  units=dict(X="mm", Q="rad")))
    print(f"[{_ts()}] [CAL] Saved pose library → {out_path}")

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
