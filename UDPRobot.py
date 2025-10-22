import socket, json, time, threading
from datetime import datetime

# Robot's IP and port
ROBOT_IP   = "192.168.2.1"
ROBOT_PORT = 8080

# Control-side bind (must match what robot sends back to!)
CONTROL_IP   = "0.0.0.0"   # listen on all interfaces
CONTROL_PORT = 8080

# Create socket and bind to control port
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((CONTROL_IP, CONTROL_PORT))
sock.settimeout(0.5)  # poll interval for recv

def timestamp():
    """Return local time as HH:MM:SS.mmm"""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

# Background receiver prints anything the robot sends
def rx_loop():
    while True:
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except OSError:
            break  # socket closed

        text = data.decode("utf-8", errors="replace")
        n_bytes = len(data)
        print(f"\n[{timestamp()}] FROM {addr} ({n_bytes} bytes): {text}")

        if text.startswith("{"):
            try:
                pkt = json.loads(text)
                print(json.dumps(pkt, indent=2))
            except Exception as e:
                print(f"[{timestamp()}] JSON parse error: {e}")
        print("> ", end="", flush=True)  # re-show prompt


threading.Thread(target=rx_loop, daemon=True).start()

def send(cmd: str):
    # attach seq for queries
    if cmd.strip() == "q":
        seq = int(time.time() * 1000) & 0xFFFFFFFF
        cmd = f"q;seq={seq}"
    sock.sendto(cmd.encode(), (ROBOT_IP, ROBOT_PORT))
    print(f"[{timestamp()}] Sent: {cmd}")

print("Commands: q (telemetry), h (home), x/y/z/a (presets), 7-coordinate joint angles, "
      "g (go), p (pause), r (resume), s (stop), e (exit robot). ;dur=x supported.")

try:
    while True:
        cmd = input("> ").strip()
        if not cmd:
            continue
        if cmd.lower() == "exit":
            break
        send(cmd)
except KeyboardInterrupt:
    pass
finally:
    print(f"[{timestamp()}] Closing socket.")
    sock.close()
