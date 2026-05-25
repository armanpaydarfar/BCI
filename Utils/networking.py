# utils/networking.py
"""
UDP/ACK protocol helpers for Harmony experiments.

This module is performance- and safety-critical: it implements the "wire format"
between experiment drivers and the robot/FES/marker systems.

Key concepts:
1) Message payloads are typically short strings or opcodes defined in `config.py`.
2) Some commands expect an ACK from the robot (see `send_udp_message(..., expect_ack=True)`).
3) ACK matching uses "base tokens": e.g. robot ACK can be like `ACK:h;dur=3.000000`,
   which should still match the base token `h`.

Common protocol fields (by convention):
- Robot opcodes / trajectories:
  - Symbolic tokens in `config.ROBOT_OPCODES` (e.g. `h;dur=3`, `g`, `a/x/y/z`).
  - Or direct 7-DOF joint coordinate trajectories as a single string in the
    form `[x1,x2,x3,x4,x5,x6,x7];dur=<seconds>` (square brackets optional).
    The validator expects exactly 7 numeric values separated by commas; any
    optional `;<suffix>` part (commonly `;dur=...`) is allowed.
- Robot triggers/ACKs: symbolic tokens in `config.TRIGGERS` (e.g. `ACK_ROBOT_HOME`).
"""
import sys
import importlib
import socket
import time
from pathlib import Path
import pygame
import select

# =========================================================
# Config loader (one level up, no deeper to avoid collisions)
# =========================================================
def _load_config():
    try:
        return importlib.import_module("config")
    except Exception:
        pass

    here = Path(__file__).resolve()
    parent = here.parent.parent
    cfg_path = parent / "config.py"
    if cfg_path.exists():
        parent_str = str(parent)
        if parent_str not in sys.path:
            sys.path.insert(0, parent_str)
        try:
            return importlib.import_module("config")
        except Exception:
            return None
    return None

_config = _load_config()

# =========================================================
# Constants
# =========================================================
ACK_PREFIX   = "ACK:"
ACK_TIMEOUT  = 0.5      # seconds per wait window
MAX_RETRIES  = 1        # resend attempts when gating
QUERY_OPCODE = "q"
STAGE_TO_GO_DELAY_S = 0.10  # 100 ms

# Simulation mode: suppress robot I/O, allow marker I/O.
# NOTE: this flag is computed at import time, so any behavior gated by it is
# decided when the module is imported (before drivers wire runtime sockets).
SIMULATION_MODE = bool(getattr(_config, "SIMULATION_MODE", False)) if _config is not None else False
print("SIM MODE:", SIMULATION_MODE)
# --- minimal state for standalone 'g' ---
_pending_target_ready = False
_pending_target_ctx   = None

# =========================================================
# Endpoints (locked)
# =========================================================
_ROBOT_IP   = None
_ROBOT_PORT = None
_BIND_IP    = None
_BIND_PORT  = None

# Marker endpoint (optional)
_MARKER_IP = None
_MARKER_PORT = None

if _config is not None:
    try:
        _ROBOT_IP   = _config.UDP_ROBOT["IP"]
        _ROBOT_PORT = int(_config.UDP_ROBOT["PORT"])
        _BIND_IP    = _config.UDP_CONTROL_BIND["IP"]
        _BIND_PORT  = int(_config.UDP_CONTROL_BIND["PORT"])
    except Exception:
        pass
    try:
        _MARKER_IP = _config.UDP_MARKER["IP"]
        _MARKER_PORT = int(_config.UDP_MARKER["PORT"])
    except Exception:
        _MARKER_IP, _MARKER_PORT = None, None

# Sensible fallback if config missing
_ROBOT_IP   = _ROBOT_IP   or "192.168.2.1"
_ROBOT_PORT = _ROBOT_PORT or 8080
_BIND_IP    = _BIND_IP    or "0.0.0.0"
_BIND_PORT  = _BIND_PORT  or 8080

# =========================================================
# Derived symbols (AFTER config is loaded)
# =========================================================
try:
    _RO = getattr(_config, "ROBOT_OPCODES", {}) or {}
    _SYMBOLIC_TRAJ = {
        _RO.get("TRAJECTORY_A", "a"),
        _RO.get("TRAJECTORY_X", "x"),
        _RO.get("TRAJECTORY_Y", "y"),
        _RO.get("TRAJECTORY_Z", "z"),
    }
except Exception:
    _SYMBOLIC_TRAJ = {"a", "x", "y", "z"}

# =========================================================
# Sockets
# =========================================================
_marker_sock = None
_ROBOT_SOCK = None
_generic_sock = None
# Once the control socket bind has failed, suppress repeated log lines.
# Reset back to False on any successful bind so future re-attempts log
# their first failure again.
_ROBOT_BIND_FAILED_LOGGED = False

# Gate the Harmony UDP control-socket bind. On Tiagobot-only rigs the
# Harmony bind address (`config.UDP_CONTROL_BIND["IP"]`, typically
# 192.168.2.2) is not assigned to any local interface, so the bind
# fails with EADDRNOTAVAIL on every send attempt and floods the log.
# Setting `BIND_ROBOT_CONTROL_SOCKET = False` in config_local.py on
# those rigs makes import-time + late-bind silently skip; Harmony rigs
# keep the default True. Tiagobot drivers never send to UDP_ROBOT
# anyway (their actuator is serial), so skipping the socket has zero
# functional impact for them.
_BIND_ROBOT_CONTROL_SOCKET = bool(
    getattr(_config, "BIND_ROBOT_CONTROL_SOCKET", True)
)

# Bind robot control socket at import-time (best effort)
try:
    if SIMULATION_MODE or not _BIND_ROBOT_CONTROL_SOCKET:
        _ROBOT_SOCK = None
    else:
        _ROBOT_SOCK = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            _ROBOT_SOCK.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
        _ROBOT_SOCK.bind((_BIND_IP, _BIND_PORT))
        _ROBOT_SOCK.setblocking(False)
except Exception as e:
    print(f"[ERROR] Could not bind control socket to {_BIND_IP}:{_BIND_PORT}: {e}")
    _ROBOT_SOCK = None


# =========================================================
# Logging
# =========================================================
def _udp_log(logger, msg: str):
    if logger is not None:
        try:
            logger.log_event(msg)
            return
        except Exception:
            pass
    print(msg)


# =========================================================
# Helpers
# =========================================================
def _to_wire(op):
    if isinstance(op, (bytes, bytearray)):
        return op.decode("utf-8", errors="ignore")
    if isinstance(op, str):
        return op
    try:
        it = list(op)
        if len(it) == 7 and all(isinstance(x, (int, float)) for x in it):
            return ",".join(f"{float(x):.6f}" for x in it)
    except Exception:
        pass
    return str(op)

def _is_coords_string(s: str) -> bool:
    t = s.strip()

    # Strip optional list brackets
    if t.startswith("[") and t.endswith("]"):
        t = t[1:-1].strip()

    # Strip optional suffixes like ;dur=...
    t = t.split(";", 1)[0].strip()

    if t.count(",") != 6:
        return False

    parts = [p.strip() for p in t.split(",")]
    if len(parts) != 7:
        return False

    try:
        _ = [float(p) for p in parts]
        return True
    except Exception:
        return False


def _base_token(msg: str) -> str:
    """
    'h;dur=3' -> 'h'
    'q;seq=123' -> 'q'
    'g' -> 'g'
    """
    if not isinstance(msg, str):
        msg = _to_wire(msg)
    s = msg.strip()
    if _is_coords_string(s):
        return s
    return s.split(";", 1)[0]


def _build_ack_map(config):
    if config is None:
        return {}
    try:
        ro = config.ROBOT_OPCODES
        tr = config.TRIGGERS
        return {
            _base_token(ro.get("GO", "g")):            tr.get("ACK_ROBOT_BEGIN"),
            _base_token(ro.get("STOP", "s")):          tr.get("ACK_ROBOT_STOP"),
            _base_token(ro.get("HOME", "h")):          tr.get("ACK_ROBOT_HOME"),
            _base_token(ro.get("PAUSE", "p")):         tr.get("ACK_ROBOT_PAUSE"),
            _base_token(ro.get("RESUME", "r")):        tr.get("ACK_ROBOT_RESUME"),
            _base_token(ro.get("MASTER_UNLOCK", "m")): tr.get("ACK_MASTER_UNLOCK"),
            _base_token(ro.get("MASTER_LOCK", "c")):   tr.get("ACK_MASTER_LOCK"),
            _base_token(ro.get("QUERY", "q")):         tr.get("ACK_ROBOT_QUERY", None),
        }
    except Exception:
        return {}

def _ensure_marker_socket(logger=None):
    """Dedicated socket for marker stream; never depends on robot bind."""
    global _marker_sock
    if _marker_sock is not None:
        return _marker_sock
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # bind to any local interface with ephemeral port
        s.bind(("0.0.0.0", 0))
        s.setblocking(False)
        _marker_sock = s
        return _marker_sock
    except Exception as e:
        _udp_log(logger, f"[ERROR] Marker socket unavailable: {e}")
        _marker_sock = None
        return None

def _ensure_control_socket(logger=None):
    """Ensure we have a bound control socket; bind now if import-time failed."""
    global _ROBOT_SOCK, _ROBOT_BIND_FAILED_LOGGED

    if SIMULATION_MODE:
        # Absolutely do not touch/bind robot sockets in sim mode.
        return None

    if not _BIND_ROBOT_CONTROL_SOCKET:
        # Tiagobot-only rig: Harmony control IP isn't on a local
        # interface, retrying the bind is guaranteed to fail. Skip
        # silently — Tiagobot drivers never send to UDP_ROBOT so the
        # missing socket has no functional impact.
        return None

    if _ROBOT_SOCK is not None:
        return _ROBOT_SOCK

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
        s.bind((_BIND_IP, _BIND_PORT))
        s.setblocking(False)
        _ROBOT_SOCK = s
        _ROBOT_BIND_FAILED_LOGGED = False
        _udp_log(logger, f"[UDP] Control socket bound at {_BIND_IP}:{_BIND_PORT} (late bind).")
    except Exception as e:
        # On a machine that doesn't have the Harmony control IP locally
        # (e.g. Tiagobot-only rig), this bind fails on every send attempt
        # and floods the log. Log the failure once per session, then
        # suppress until a successful bind clears the flag.
        if not _ROBOT_BIND_FAILED_LOGGED:
            _udp_log(logger, f"[ERROR] Late bind failed at {_BIND_IP}:{_BIND_PORT}: {e}")
            _udp_log(logger, "[INFO] Suppressing further 'Late bind failed' messages this session.")
            _ROBOT_BIND_FAILED_LOGGED = True
        _ROBOT_SOCK = None
    return _ROBOT_SOCK


def _ensure_generic_socket(logger=None):
    """Ephemeral socket for 'other' UDP messages (never binds to control IP/port)."""
    global _generic_sock
    if _generic_sock is not None:
        return _generic_sock
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("0.0.0.0", 0))
        s.setblocking(False)
        _generic_sock = s
        return _generic_sock
    except Exception as e:
        _udp_log(logger, f"[ERROR] Generic UDP socket unavailable: {e}")
        _generic_sock = None
        return None


def _drain_socket(sock, max_ms: int = 50):
    if sock is None:
        return
    end = time.time() + (max_ms / 1000.0)
    while time.time() < end:
        r, _, _ = select.select([sock], [], [], 0.0)
        if not r:
            break
        try:
            sock.recvfrom(65535)
        except BlockingIOError:
            break
        except Exception:
            break

def _drain_control_socket(max_ms: int = 50, logger=None):
    s = _ensure_control_socket(logger)
    if s is None:
        return
    _drain_socket(s, max_ms=max_ms)

def _send_marker_trigger(config, logger, trigger_value: str):
    """Fire a software trigger to the marker stream (marker-only socket)."""
    if not trigger_value:
        return
    if config is None:
        _udp_log(logger, f"[WARN] No config; cannot send marker trigger {trigger_value}.")
        return
    try:
        ip = config.UDP_MARKER["IP"]
        port = int(config.UDP_MARKER["PORT"])
        ms = _ensure_marker_socket(logger)
        if ms is None:
            _udp_log(logger, f"[ERROR] Marker socket unavailable; cannot send trigger {trigger_value}.")
            return
        ms.sendto(str(trigger_value).encode("utf-8"), (ip, port))
        _udp_log(logger, f"[TRIGGER] Sent marker trigger={trigger_value}")
    except Exception as e:
        _udp_log(logger, f"[ERROR] Failed to send marker trigger {trigger_value}: {e}")

def _sendto_robot(payload: bytes, logger=None):
    """Always send FROM the bound control socket TO the robot endpoint."""
    if SIMULATION_MODE:
        raise RuntimeError("SIMULATION_MODE enabled; robot send suppressed.")
    s = _ensure_control_socket(logger)
    if s is None:
        raise RuntimeError("Control socket not available; cannot send to robot.")
    s.sendto(payload, (_ROBOT_IP, _ROBOT_PORT))

def _await_ack_blocking(expected_token: str, logger=None) -> bool:
    """
    Wait up to ACK_TIMEOUT for an ACK corresponding to `expected_token` on the control socket.

    Robustness:
      - Accepts duration-bearing tokens by matching on `_base_token()`:
          expected_token: "h;dur=3"   will accept ACK:"h;dur=3.000000"
          expected_token: "h"         will accept ACK:"h;dur=3.000000" (base token "h")
      - Preserves special-case exact match for "COORDS_STAGED_RAD"
    """
    if SIMULATION_MODE:
        return True  # in simulation we don't wait for robot ACKs

    s = _ensure_control_socket(logger)
    if s is None:
        return False

    expected_base = _base_token(expected_token)

    end = time.time() + ACK_TIMEOUT
    while time.time() < end:
        r, _, _ = select.select([s], [], [], max(0.0, end - time.time()))
        if not r:
            continue

        try:
            data, _ = s.recvfrom(65535)
        except BlockingIOError:
            continue
        except Exception as e:
            _udp_log(logger, f"[ERROR] recv failed while waiting for ACK:{expected_token}: {e}")
            return False

        txt = data.decode("utf-8", errors="ignore").strip()
        if not txt.startswith(ACK_PREFIX):
            # Ignore non-ACK chatter while we're waiting
            continue

        token = txt[len(ACK_PREFIX):].strip()

        # Preserve the staged-coords special-case as an exact token.
        if token == "COORDS_STAGED_RAD" and expected_token == "COORDS_STAGED_RAD":
            _udp_log(logger, f"[ROBOT->UDP] {txt}")
            return True

        token_base = _base_token(token)

        if token_base == expected_base:
            _udp_log(logger, f"[ROBOT->UDP] {txt}")
            return True

        _udp_log(logger, f"[ROBOT->UDP][IGNORED DURING WAIT] {txt}")

    return False

_fes_sock = None
def _ensure_fes_socket(logger=None):
    """Dedicated socket for FES to avoid binding conflicts."""
    global _fes_sock
    if _fes_sock is not None:
        return _fes_sock
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("0.0.0.0", 0)) # Ephemeral port
        s.setblocking(False)
        _fes_sock = s
        return _fes_sock
    except Exception as e:
        _udp_log(logger, f"[ERROR] FES socket unavailable: {e}")
        return None
    
# =========================================================
# Public API
# =========================================================
def display_multiple_messages_with_udp(
    messages, colors, offsets, duration=13,
    udp_messages=None, udp_socket=None, udp_ip=None, udp_port=None,
    logger=None, eeg_state=None
):
    """
    Simulation mode support:
      - If SIMULATION_MODE=True: suppress robot sends + ACK waits, but keep marker triggers working.

    Robustness:
      - All opcode branching uses `_base_token()` so tokens like "h;dur=3" behave like "h".
      - Passive ACK->trigger mapping uses `_base_token()` so ACKs like "ACK:h;dur=3.000000" map to "h".
    """
    ack_to_trigger = _build_ack_map(_config)  # already base-tokenized in your current version
    fired_triggers = set()

    # UI prep
    font = pygame.font.SysFont(None, 96)
    end_time = pygame.time.get_ticks() + int(duration * 1000)
    query_payload = None

    # ---------------- SEND PHASE ----------------
    if udp_messages:
        global _pending_target_ready, _pending_target_ctx
        i = 0
        while i < len(udp_messages):
            op = _to_wire(udp_messages[i])
            op_base = _base_token(op)

            # Defer sending trajectory if immediately followed by 'g' (base-token aware)
            if (
                op_base != "g"
                and (_is_coords_string(op) or op_base in _SYMBOLIC_TRAJ or op in _SYMBOLIC_TRAJ)
                and (i + 1) < len(udp_messages)
                and _base_token(_to_wire(udp_messages[i + 1])) == "g"
            ):
                i += 1
                continue

            # --- 'g' gating (base-token aware) ---
            if op_base == "g":
                if SIMULATION_MODE:
                    _udp_log(logger, "[SIM] Suppressed robot GO sequence ('g').")
                    # optional: still fire the "robot begin" trigger so the experiment pipeline proceeds
                    trig = ack_to_trigger.get("g")
                    if trig and trig not in fired_triggers:
                        _send_marker_trigger(_config, logger, trig)
                        fired_triggers.add(trig)
                    _pending_target_ready = False
                    _pending_target_ctx = None
                    i += 1
                    continue

                if i == 0 and _pending_target_ready:
                    _drain_control_socket(max_ms=30, logger=logger)
                    try:
                        _sendto_robot(b"g", logger=logger)
                        _udp_log(logger, "[UDP->ROBOT] Sent opcode: g (standalone)")
                    except Exception as e:
                        _udp_log(logger, f"[ERROR] Failed to send 'g': {e}")
                        i += 1
                        continue

                    ack_ok = False
                    attempts = 0
                    while not ack_ok and attempts <= MAX_RETRIES:
                        attempts += 1
                        ack_ok = _await_ack_blocking("g", logger=logger)
                        if not ack_ok and attempts <= MAX_RETRIES:
                            try:
                                _sendto_robot(b"g", logger=logger)
                                _udp_log(logger, "[RETRY] Resent opcode: g")
                            except Exception as e:
                                _udp_log(logger, f"[ERROR] Retry send failed for 'g': {e}")
                                break

                    if not ack_ok:
                        _udp_log(logger, "[ERROR] No ACK for standalone 'g'.")
                    else:
                        trig = ack_to_trigger.get("g")
                        if trig and trig not in fired_triggers:
                            _send_marker_trigger(_config, logger, trig)
                            fired_triggers.add(trig)

                    _pending_target_ready = False
                    _pending_target_ctx = None
                    i += 1
                    continue

                if i == 0:
                    _udp_log(logger, "[ERROR] 'g' cannot be first; skipping.")
                    i += 1
                    continue

                traj = _to_wire(udp_messages[i - 1])

                _drain_control_socket(max_ms=50, logger=logger)

                try:
                    _sendto_robot(traj.encode("utf-8"), logger=logger)
                    _udp_log(logger, f"[UDP->ROBOT] Sent trajectory: {traj}")
                except Exception as e:
                    _udp_log(logger, f"[ERROR] Failed to send trajectory '{traj}': {e}")
                    i += 1
                    continue

                expected = "COORDS_STAGED_RAD" if _is_coords_string(traj) else traj
                ack_ok = False
                attempts = 0
                while not ack_ok and attempts <= MAX_RETRIES:
                    attempts += 1
                    ack_ok = _await_ack_blocking(expected, logger=logger)
                    if not ack_ok and attempts <= MAX_RETRIES:
                        try:
                            _sendto_robot(traj.encode("utf-8"), logger=logger)
                            _udp_log(logger, f"[RETRY] Resent trajectory: {traj}")
                        except Exception as e:
                            _udp_log(logger, f"[ERROR] Retry send failed: {e}")
                            break

                if not ack_ok:
                    _udp_log(logger, f"[ERROR] No ACK for trajectory '{traj}' (expected ACK:{expected}). Skipping 'g'.")
                    i += 1
                    continue

                time.sleep(STAGE_TO_GO_DELAY_S)
                _drain_control_socket(max_ms=20, logger=logger)

                try:
                    _sendto_robot(b"g", logger=logger)
                    _udp_log(logger, "[UDP->ROBOT] Sent opcode: g")
                except Exception as e:
                    _udp_log(logger, f"[ERROR] Failed to send 'g': {e}")
                    i += 1
                    continue

                ack_ok = False
                attempts = 0
                while not ack_ok and attempts <= MAX_RETRIES:
                    attempts += 1
                    ack_ok = _await_ack_blocking("g", logger=logger)
                    if not ack_ok and attempts <= MAX_RETRIES:
                        try:
                            _sendto_robot(b"g", logger=logger)
                            _udp_log(logger, "[RETRY] Resent opcode: g")
                        except Exception as e:
                            _udp_log(logger, f"[ERROR] Retry send failed for 'g': {e}")
                            break

                if not ack_ok:
                    _udp_log(logger, "[ERROR] No ACK for 'g' after trajectory.")
                else:
                    trig = ack_to_trigger.get("g")
                    if trig and trig not in fired_triggers:
                        _send_marker_trigger(_config, logger, trig)
                        fired_triggers.add(trig)

                _pending_target_ready = False
                _pending_target_ctx = None
                i += 1
                continue

            # --- 'q' special (base-token aware) ---
            if op_base == QUERY_OPCODE:
                if SIMULATION_MODE:
                    _udp_log(logger, "[SIM] Suppressed robot query ('q').")
                    i += 1
                    continue
                try:
                    seq = (int(time.time() * 1000) & 0xFFFFFFFF)
                    qmsg = f"q;seq={seq}"
                    _sendto_robot(qmsg.encode("utf-8"), logger=logger)
                    _udp_log(logger, f"[UDP->ROBOT] Sent opcode: {qmsg}")
                except Exception as e:
                    _udp_log(logger, f"[ERROR] Failed to send 'q': {e}")
                i += 1
                continue

            # --- all other opcodes ---
            if SIMULATION_MODE:
                _udp_log(logger, f"[SIM] Suppressed robot opcode: {op}")
            else:
                try:
                    _sendto_robot(op.encode("utf-8"), logger=logger)
                    _udp_log(logger, f"[UDP->ROBOT] Sent opcode: {op}")
                except Exception as e:
                    _udp_log(logger, f"[ERROR] Failed to send opcode '{op}': {e}")

            # Track internal state based on base tokens (durations won't break this)
            if op_base == "c":
                _pending_target_ready = True
                _pending_target_ctx = None
            elif op_base in ("m", "h"):
                _pending_target_ready = False
                _pending_target_ctx = None

            i += 1

    # ---------------- UI + PASSIVE RECV ----------------
    clock = pygame.time.Clock()
    while pygame.time.get_ticks() < end_time:
        surface = pygame.display.get_surface()
        if surface is not None:
            surface.fill((0, 0, 0))
            for i, text in enumerate(messages):
                img = font.render(text, True, colors[i])
                surface.blit(
                    img,
                    (surface.get_width() // 2 - img.get_width() // 2,
                     surface.get_height() // 2 - img.get_height() // 2 + offsets[i])
                )
            pygame.display.flip()

        if eeg_state is not None:
            try:
                eeg_state.update()
            except Exception as e:
                _udp_log(logger, f"[WARN] eeg_state.update() failed: {e}")

        # In simulation mode we still allow marker triggers (outgoing),
        # but we don't try to read robot UDP.
        if not SIMULATION_MODE:
            s = _ensure_control_socket(logger)
            if s is not None:
                while True:
                    try:
                        data, _ = s.recvfrom(65535)
                    except BlockingIOError:
                        break
                    except Exception as e:
                        _udp_log(logger, f"[ERROR] recvfrom failed: {e}")
                        break

                    txt = data.decode("utf-8", errors="ignore").strip()

                    if txt.startswith(ACK_PREFIX):
                        token = txt[len(ACK_PREFIX):].strip()

                        # Keep special-case as-is
                        if token == "COORDS_STAGED_RAD":
                            _udp_log(logger, "[ROBOT->UDP] ACK:COORDS_STAGED_RAD (staged)")
                            continue

                        # Base-token mapping so ACK:h;dur=3.000000 maps to 'h'
                        token_base = _base_token(token)

                        trig = (
                            ack_to_trigger.get(token_base)  # preferred
                            or ack_to_trigger.get(token)    # fallback (if any legacy full-token keys exist)
                        )
                        if trig and trig not in fired_triggers:
                            _send_marker_trigger(_config, logger, trig)
                            fired_triggers.add(trig)

                        _udp_log(logger, f"[ROBOT->UDP] {txt}")
                        continue

                    if query_payload is None and txt.startswith("{") and "\"op\":\"Q\"" in txt:
                        query_payload = txt
                    _udp_log(logger, f"[ROBOT->UDP] {txt}")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        clock.tick(60)

    return query_payload

def send_udp_message(
    sock, ip, port, message,
    logger=None, quiet=False,
    *,
    expect_ack: bool = False,
    ack_prefix: str = "ACK:",
    ack_timeout: float = 0.8,
    max_retries: int = 0,
    capture_query: bool = False
):
    """
    Simulation mode support: True
      - Robot: Suppress send + ACK wait (SAFETY).
      - FES: ALWAYS SEND (Active in simulation).
      - Marker: ALWAYS SEND.

    Bug fix:
      - In SIMULATION_MODE, never touch/bind/control-check the robot control socket.
      - If destination is not robot/marker/FES, use a generic ephemeral socket (never the control socket).
      - If a FES message is sent but caller passed wrong endpoint, optionally re-route based on message prefix.
    """
    if not quiet:
        _log = lambda m: _udp_log(logger, m)
    else:
        _log = lambda m: None  # noqa: E731

    # 1) LOAD CONFIGURATIONS
    ro = getattr(_config, "UDP_ROBOT", {}) or {}
    mk = getattr(_config, "UDP_MARKER", {}) or {}
    fs = getattr(_config, "UDP_FES", {}) or {}

    # Robot Endpoints
    robot_ip = ro.get("IP", _ROBOT_IP)
    robot_port = int(ro.get("PORT", _ROBOT_PORT))

    # Marker Endpoints
    marker_ip = mk.get("IP", _MARKER_IP)
    marker_port = int(mk.get("PORT", _MARKER_PORT)) if mk.get("PORT", _MARKER_PORT) else None

    # FES Endpoints
    fes_ip = fs.get("IP", None)
    fes_port = int(fs.get("PORT", 0)) if fs.get("PORT", None) else None

    # Normalize inputs
    try:
        port_i = int(port)
    except Exception:
        port_i = port

    msg_str = message if isinstance(message, str) else str(message)

    # 2) IDENTIFY DESTINATION
    is_robot  = (ip == robot_ip and port_i == int(robot_port))
    is_marker = (marker_ip is not None and marker_port is not None and ip == marker_ip and port_i == int(marker_port))
    is_fes    = (fes_ip is not None and fes_port is not None and ip == fes_ip and port_i == int(fes_port))
    is_traj   = _is_coords_string(msg_str)

    # 2b) SAFETY: if it looks like a FES command but caller passed wrong endpoint,
    # re-route to UDP_FES endpoint if configured.
    # (This prevents accidental fallback to control socket which was causing your bind errors.)
    if (not is_fes) and (fes_ip is not None and fes_port is not None):
        if isinstance(msg_str, str) and msg_str.startswith("FES_"):
            _log(f"[INFO] Detected FES message '{msg_str}' but destination was {ip}:{port}. "
                 f"Routing to UDP_FES {fes_ip}:{fes_port}.")
            ip = fes_ip
            port_i = int(fes_port)
            is_fes = True
            # re-evaluate "is_robot/is_marker" after rewrite
            is_robot = (ip == robot_ip and port_i == int(robot_port))
            is_marker = (marker_ip is not None and marker_port is not None and ip == marker_ip and port_i == int(marker_port))

    # 3) HANDLE SIMULATION MODE
    # STRICTLY suppress Robot only. FES/Marker/Other should still send.
    if SIMULATION_MODE and is_robot:
        _log(f"[SIM] Robot destination passed ({ip}:{port_i}) but suppressed. message='{msg_str}'")
        if expect_ack:
            return (True, None)
        return None

    # 4) CHOOSE THE RIGHT SOCKET
    # IMPORTANT: in SIMULATION_MODE we must NEVER attempt to bind/control-check the robot socket.
    if is_marker:
        s = _ensure_marker_socket(logger)
        if s is None:
            _log("[ERROR] Marker socket unavailable; cannot send.")
            return (False, None) if expect_ack else None

    elif is_fes:
        s = _ensure_fes_socket(logger)
        if s is None:
            _log("[ERROR] FES socket unavailable; cannot send.")
            return (False, None) if expect_ack else None

    elif is_robot:
        # Only robot traffic uses the control socket
        s = _ensure_control_socket(logger)
        if s is None:
            _log("[ERROR] Control socket unavailable; cannot send to robot.")
            return (False, None) if expect_ack else None

    else:
        # Unknown/other endpoint: never default to the control socket.
        # Use a generic ephemeral socket so non-robot features keep working.
        try:
            s = _ensure_generic_socket(logger)  # you said you already implemented this
        except NameError:
            s = None
        if s is None:
            _log("[ERROR] Generic socket unavailable; cannot send.")
            return (False, None) if expect_ack else None

    # ACK->trigger map if needed
    if expect_ack:
        try:
            _ack_to_trigger_map = _build_ack_map(_config)
        except Exception:
            _ack_to_trigger_map = {}
    else:
        _ack_to_trigger_map = None

    attempts = 0
    while True:
        # Only drain control socket when we expect robot ACKs
        if is_robot:
            _drain_control_socket(max_ms=30, logger=logger)

        try:
            s.sendto(msg_str.encode("utf-8"), (ip, int(port_i)))
        except Exception as e:
            _log(f"UDP send failed to {ip}:{port_i} — {e}")
            return (False, None) if expect_ack else None

        # 5) DETERMINE LOG PREFIX
        if is_robot:
            prefix = "[UDP->ROBOT]"
        elif is_fes:
            prefix = "[UDP->FES]"
        elif is_marker:
            prefix = "[TRIGGER]"
        else:
            prefix = f"[UDP->{ip}:{port_i}]"

        kind = "Sent trajectory" if is_traj else "Sent opcode"
        _log(f"{prefix} {kind}: {msg_str}")

        # --- INTERNAL STATE TRACKING (ROBOT ONLY) ---
        global _pending_target_ready, _pending_target_ctx
        try:
            rop = getattr(_config, "ROBOT_OPCODES", {}) or {}
            tok_c = rop.get("MASTER_LOCK",   "c")
            tok_m = rop.get("MASTER_UNLOCK", "m")
            tok_h = rop.get("HOME",          "h")
        except Exception:
            tok_c, tok_m, tok_h = "c", "m", "h"

        if is_robot:
            msg_base = _base_token(msg_str)
            if msg_base == _base_token(tok_c):
                _pending_target_ready = True
                _pending_target_ctx   = None
            elif msg_base in (_base_token(tok_m), _base_token(tok_h)):
                _pending_target_ready = False
                _pending_target_ctx   = None

        # Fast exit (FES and Marker don't wait for ACKs)
        if not expect_ack and not capture_query:
            return None

        # If this is NOT robot traffic, we generally don't wait for ACKs
        if not is_robot:
            return (True, None) if expect_ack else None

        # --- ROBOT ACK WAIT LOOP ---
        end = time.time() + float(ack_timeout)
        acked = False
        query_payload = None
        ack_token_matched = None

        while time.time() < end:
            r, _, _ = select.select([s], [], [], max(0.0, end - time.time()))
            if not r:
                continue
            try:
                data, addr = s.recvfrom(65535)
            except Exception:
                break

            src_ip, src_port = addr[0], int(addr[1])
            if src_ip != robot_ip or src_port != int(robot_port):
                continue

            txt = data.decode("utf-8", errors="ignore").strip()

            if expect_ack and txt.startswith(ack_prefix):
                token = txt[len(ack_prefix):].strip()

                # Query ACK: accept ACK:q* for any q;seq=...
                if msg_str.startswith("q"):
                    if token.startswith("q"):
                        _log(f"[ROBOT->UDP] {txt}")
                        acked = True
                        ack_token_matched = "q"
                        if not capture_query:
                            if _ack_to_trigger_map:
                                trig = _ack_to_trigger_map.get(ack_token_matched)
                                if trig:
                                    _send_marker_trigger(_config, logger, trig)
                            return (True, None)
                    continue  # <--- important: always continue for q-case

                # Non-query ACK: base-token match (durations etc.)
                token_base = _base_token(token)
                msg_base   = _base_token(msg_str)

                if token_base == msg_base:
                    _log(f"[ROBOT->UDP] {txt}")
                    acked = True
                    ack_token_matched = token_base  # store base token for trigger lookup
                    if not capture_query:
                        if _ack_to_trigger_map:
                            trig = (_ack_to_trigger_map.get(ack_token_matched)
                                    or _ack_to_trigger_map.get(msg_base))
                            if trig:
                                _send_marker_trigger(_config, logger, trig)
                        return (True, None)

                continue

            if capture_query and msg_str.startswith("q") and not txt.startswith(ack_prefix) and query_payload is None:
                query_payload = txt
                if not expect_ack:
                    return query_payload

        if expect_ack and acked:
            if _ack_to_trigger_map and ack_token_matched is not None:
                trig = _ack_to_trigger_map.get(ack_token_matched)
                if trig:
                    _send_marker_trigger(_config, logger, trig)
            return (True, query_payload)

        if not expect_ack:
            return query_payload

        if attempts < int(max_retries):
            attempts += 1
            _log(f"[RETRY] No expected ACK for '{msg_str}' after {ack_timeout:.2f}s. "
                 f"Retrying ({attempts}/{max_retries})…")
            continue

        return (False, query_payload)
