#!/usr/bin/env python3
# simple_posemap_exec.py
#
# Very simple "go to nearest pose" controller.
# - User enters EE goal (mm) + duration (s)
# - Find nearest pose in library
# - Send q1...q7;dur= and then "g"
# - Option to home
#
# NO streaming, NO lookahead, NO planner ticks.
# Just point → lookup → go.

import sys, socket, json, time
import numpy as np
from datetime import datetime

ROBOT_IP, ROBOT_PORT = "192.168.2.1", 8080
CONTROL_IP, CONTROL_PORT = "0.0.0.0", 8080

ACK_TIMEOUT_S = 0.35
CMD_PROCESS_GRACE_S = 0.40
ACTIVE_SIDE = "R"

# ---------------- UDP ----------------

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((CONTROL_IP, CONTROL_PORT))
sock.settimeout(0.5)

def ts(): return datetime.now().strftime("%H:%M:%S.%f")[:-3]

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

def query_state():
    seq = int(time.time()*1000) & 0xFFFFFFFF
    udp_send(f"q;seq={seq}")
    t0 = time.time()
    while time.time() - t0 < 0.6:
        r = udp_recv(0.6)
        if not r: continue
        if not r.startswith("{"): continue
        try:
            pkt = json.loads(r)
            if ACTIVE_SIDE == "R":
                return np.array(pkt["eeR"]["pos_mm"])
            else:
                return np.array(pkt["eeL"]["pos_mm"])
        except:
            continue
    print(f"[{ts()}] Telemetry error.")
    return None

def go_home():
    print(f"[{ts()}] HOME...")
    send_and_wait_ack("h", expect="h")
    print(f"[{ts()}] HOME done.")

# ---------------- Library ----------------

def load_library(path):
    z = np.load(path, allow_pickle=True)
    X = z["X"]    # positions mm
    Q = z["Q"]    # joint rad
    return X, Q

def nearest_idx(X, x):
    d = np.linalg.norm(X - x[None,:], axis=1)
    return int(np.argmin(d)), float(np.min(d))

# ---------------- User interface ----------------

def ask_goal_and_duration():
    s = input("Enter goal 'x,y,z' (mm), or 'home', or 'quit': ").strip().lower()
    if s in ("quit","q","exit"): return None, None
    if s in ("home","h"): return np.array([np.nan]*3), 0.0

    try:
        xyz = np.array([float(v) for v in s.split(",")], dtype=float)
        dur = float(input("Duration (s): ").strip())
        return xyz, dur
    except:
        print("Bad input, try again.")
        return ask_goal_and_duration()

# ---------------- Main ----------------

def main():
    if len(sys.argv)<2:
        print("Usage: python simple_posemap_exec.py poses.npz")
        sys.exit(1)

    X, Q = load_library(sys.argv[1])
    print(f"[{ts()}] Loaded library with {len(X)} samples.")

    while True:
        goal, dur = ask_goal_and_duration()
        if goal is None:
            print("Exiting.")
            break
        if np.isnan(goal).any():
            go_home()
            continue

        print(f"[{ts()}] Querying current EE...")
        x_now = query_state()
        if x_now is None:
            print("No telemetry, skip.")
            continue

        idx, d = nearest_idx(X, goal)
        q = Q[idx]
        print(f"[{ts()}] Nearest node {idx} (dist {d:.1f} mm)")
        print(f"q: {np.round(q,4).tolist()}")

        coords = ",".join(f"{v:.6f}" for v in q) + f";dur={dur:.3f}"

        ok1 = send_and_wait_ack(coords)
        ok2 = send_and_wait_ack("g", expect="g")

        print(f"[{ts()}] Command {'OK' if (ok1 and ok2) else 'ERR'}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[SAFE EXIT] Sending pause+home")
        send_and_wait_ack("p", expect="p")
        send_and_wait_ack("h", expect="h")
