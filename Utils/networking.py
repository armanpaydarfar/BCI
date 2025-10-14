# utils/networking.py
import sys
import importlib
import socket
import time
from pathlib import Path
import pygame
import select



# =========================================================
# Constants 
# =========================================================
ACK_PREFIX   = "ACK:"
ACK_TIMEOUT  = 0.5      # seconds per wait window
MAX_RETRIES  = 1        # resend attempts when gating
QUERY_OPCODE = "q"
# How long to wait after traj ACK before sending 'g' (race guard)
STAGE_TO_GO_DELAY_S = 0.10   # 100 ms; bump to 0.12–0.15 if you still see "nothing armed"


# --- minimal state for standalone 'g' ---
_pending_target_ready = False
_pending_target_ctx   = None


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

# Endpoints (locked)
_ROBOT_IP   = None
_ROBOT_PORT = None
_BIND_IP    = None
_BIND_PORT  = None

if _config is not None:
    try:
        _ROBOT_IP   = _config.UDP_ROBOT["IP"]
        _ROBOT_PORT = int(_config.UDP_ROBOT["PORT"])
        _BIND_IP    = _config.UDP_CONTROL_BIND["IP"]
        _BIND_PORT  = int(_config.UDP_CONTROL_BIND["PORT"])
    except Exception:
        pass

# Sensible fallback if config missing (keeps UI alive, but robot I/O won’t work)
_ROBOT_IP   = _ROBOT_IP   or "192.168.2.1"
_ROBOT_PORT = _ROBOT_PORT or 8080
_BIND_IP    = _BIND_IP    or "0.0.0.0"
_BIND_PORT  = _BIND_PORT  or 8080

# The one socket we’ll use for ALL robot TX/RX to avoid ephemeral ports
_ROBOT_SOCK = None
try:
    _ROBOT_SOCK = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        _ROBOT_SOCK.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except Exception:
        pass
    _ROBOT_SOCK.bind((_BIND_IP, _BIND_PORT))
    _ROBOT_SOCK.setblocking(False)
    # NOTE: We intentionally DO NOT connect(); we always use sendto() to (_ROBOT_IP,_ROBOT_PORT).
    # This guarantees we keep the same local port (CONTROL_BIND) and never leak ephemeral ports.
except Exception as e:
    # If binding fails (e.g., port in use), we’ll defer to best-effort in functions.
    print(f"[ERROR] Could not bind control socket to {_BIND_IP}:{_BIND_PORT}: {e}")
    _ROBOT_SOCK = None




# =========================================================
# Core helpers using the bound socket
# =========================================================
def _drain_control_socket(max_ms: int = 50, logger=None):
    """
    Non-blocking drain of the control socket for up to max_ms to clear stale datagrams
    (e.g., leftover ACKs from a previous command).
    """
    s = _ensure_control_socket(logger)
    if s is None:
        return
    end = time.time() + (max_ms / 1000.0)
    while time.time() < end:
        r, _, _ = select.select([s], [], [], 0.0)
        if not r:
            break
        try:
            s.recvfrom(65535)  # discard
        except BlockingIOError:
            break
        except Exception:
            break

def _udp_log(logger, msg: str):
    if logger is not None:
        try:
            logger.log_event(msg); return
        except Exception:
            pass
    print(msg)

def _to_wire(op):
    """Stringify opcodes or 7-element coordinate vectors into wire format."""
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
    if t.startswith("[") and t.endswith("]"):
        t = t[1:-1].strip()
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

def _build_ack_map(config):
    """
    Map opcode tokens (what appears after ACK:) -> marker trigger.
    Only map motion/control (300-series). No trigger for COORDS_STAGED_RAD.
    """
    if config is None:
        return {}
    try:
        ro = config.ROBOT_OPCODES
        tr = config.TRIGGERS
        return {
            ro.get("GO", "g"):               tr.get("ACK_ROBOT_BEGIN"),
            ro.get("STOP", "s"):             tr.get("ACK_ROBOT_STOP"),
            ro.get("HOME", "h"):             tr.get("ACK_ROBOT_HOME"),
            ro.get("Pause", "p"):            tr.get("ACK_ROBOT_PAUSE"),
            ro.get("RESUME", "r"):           tr.get("ACK_ROBOT_RESUME"),
            ro.get("MASTER_UNLOCK", "m"):    tr.get("ACK_MASTER_UNLOCK"),
            ro.get("MASTER_LOCK", "c"):      tr.get("ACK_MASTER_LOCK"),
        }
    except Exception:
        return {}

def _send_marker_trigger(config, logger, trigger_value: str):
    """Fire a software trigger to the marker stream."""
    if not trigger_value:
        return
    if config is None:
        _udp_log(logger, f"[WARN] No config; cannot send marker trigger {trigger_value}.")
        return
    try:
        ip = config.UDP_MARKER["IP"]; port = int(config.UDP_MARKER["PORT"])
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as ms:
            ms.sendto(str(trigger_value).encode("utf-8"), (ip, port))
        _udp_log(logger, f"[TRIGGER] Sent marker trigger={trigger_value}")
    except Exception as e:
        _udp_log(logger, f"[ERROR] Failed to send marker trigger {trigger_value}: {e}")


# =========================================================
# Import-time binding: ONE control socket on CONTROL_BIND
# =========================================================
def _ensure_control_socket(logger=None):
    """Ensure we have a bound control socket; bind now if import-time failed."""
    global _ROBOT_SOCK
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
        _udp_log(logger, f"[UDP] Control socket bound at {_BIND_IP}:{_BIND_PORT} (late bind).")
    except Exception as e:
        _udp_log(logger, f"[ERROR] Late bind failed at {_BIND_IP}:{_BIND_PORT}: {e}")
        _ROBOT_SOCK = None
    return _ROBOT_SOCK

def _sendto_robot(payload: bytes, logger=None):
    """Always send FROM the bound control socket TO the robot endpoint."""
    s = _ensure_control_socket(logger)
    if s is None:
        raise RuntimeError("Control socket not available; cannot send to robot.")
    s.sendto(payload, (_ROBOT_IP, _ROBOT_PORT))

def _await_ack_blocking(expected_token: str, logger=None) -> bool:
    """
    Wait up to ACK_TIMEOUT for f"ACK:{expected_token}" on the bound control socket.
    Retries handled by caller.
    """
    s = _ensure_control_socket(logger)
    if s is None:
        return False
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
        if txt.startswith(ACK_PREFIX):
            token = txt[len(ACK_PREFIX):].strip()
            # Log ALL ACKs, including ACK:q;seq=...
            _udp_log(None if logger is None else logger, f"[ROBOT->UDP] {txt}")
            if token == expected_token:
                return True
        # Non-ACKs are ignored during the blocking wait to avoid consuming telemetry.
    return False


def _await_ack_blocking(expected_token: str, logger=None) -> bool:
    s = _ensure_control_socket(logger)
    if s is None:
        return False
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
        if txt.startswith(ACK_PREFIX):
            token = txt[len(ACK_PREFIX):].strip()
            if token == expected_token:
                _udp_log(logger, f"[ROBOT->UDP] {txt}")  # expected ACK
                return True
            else:
                # Unrelated/stale ACK — note it but mark as ignored during this wait
                _udp_log(logger, f"[ROBOT->UDP][IGNORED DURING WAIT] {txt}")
            continue
        # Non-ACKs ignored here (we don’t consume telemetry during the blocking wait)
    return False


# =========================================================
# Public API
# =========================================================
def display_multiple_messages_with_udp(
    messages, colors, offsets, duration=13,
    udp_messages=None, udp_socket=None, udp_ip=None, udp_port=None,
    logger=None, eeg_state=None):
    """
    One-socket model locked by config:
      TX → UDP_ROBOT.IP:UDP_ROBOT.PORT (e.g., 192.168.2.1:8080)
      RX ← UDP_CONTROL_BIND.IP:UDP_CONTROL_BIND.PORT (e.g., 192.168.2.2:8080)

    Changes in this version:
      • Race guard: wait STAGE_TO_GO_DELAY_S after traj ACK before sending 'g'.
      • Drain stale ACKs: clear the control socket before sending traj, and again before 'g'.
      • De-dup trajectories: if NEXT op is 'g', defer sending and let the 'g' branch send it once.
      • Logs ALL ACKs (incl. ACK:q), but triggers only for mapped control ACKs; never for COORDS_STAGED_RAD.
    """
    # Resolve / ensure control socket
    s = _ensure_control_socket(logger)
    if s is None:
        _udp_log(logger, "[ERROR] Control socket unavailable; robot I/O disabled.")

    ack_to_trigger = _build_ack_map(_config)
    fired_triggers = set()

    # UI prep
    font = pygame.font.SysFont(None, 96)
    end_time = pygame.time.get_ticks() + int(duration * 1000)
    query_payload = None

    # ---------------- SEND PHASE ----------------
    if udp_messages:
        # <<< ADDED: use minimal pending-goal state
        global _pending_target_ready, _pending_target_ctx

        i = 0
        while i < len(udp_messages):
            op = _to_wire(udp_messages[i])

            # Defer send ONLY if this op is a staged trajectory immediately followed by 'g'
            # (so we don't accidentally skip control ops like 'c')
            # <<< CHANGED: gate the deferral on _is_coords_string(op)
            if op != "g" and _is_coords_string(op) and (i + 1) < len(udp_messages) and _to_wire(udp_messages[i + 1]) == "g":  # <<< CHANGED
                i += 1
                continue

            # --- 'g' gating ---
            if op == "g":
                # <<< ADDED: allow 'g' as first iff a pending target exists (set by 'c')
                if i == 0 and _pending_target_ready:  # <<< ADDED
                    # Drain before sending 'g' so we don't eat the fresh ACK
                    _drain_control_socket(max_ms=30, logger=logger)
                    # Send 'g' directly (no preceding traj in this packet)
                    try:
                        _sendto_robot(b"g", logger=logger)
                        _udp_log(logger, "[UDP->ROBOT] Sent opcode: g (standalone)")
                    except Exception as e:
                        _udp_log(logger, f"[ERROR] Failed to send 'g': {e}")
                        i += 1
                        continue

                    # Wait for ACK:g (with retry), reusing your existing pattern
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
                        # motion-begin trigger (same as your existing path)
                        trig = ack_to_trigger.get("g") or ack_to_trigger.get(
                            getattr(_config, "ROBOT_OPCODES", {}).get("GO", "g") if _config else "g"
                        )
                        if trig and trig not in fired_triggers:
                            _send_marker_trigger(_config, logger, trig)
                            fired_triggers.add(trig)

                    # Clear pending target after attempting to go
                    _pending_target_ready = False        # <<< ADDED
                    _pending_target_ctx   = None         # <<< ADDED

                    i += 1
                    continue

                # Fall back to original behavior (needs a preceding traj in this packet)
                if i == 0:
                    _udp_log(logger, "[ERROR] 'g' cannot be first; skipping.")
                    i += 1
                    continue

                traj = _to_wire(udp_messages[i - 1])

                # 0) Drain stale ACKs before starting a gated transaction
                _drain_control_socket(max_ms=50, logger=logger)

                # 1) send trajectory
                try:
                    _sendto_robot(traj.encode("utf-8"), logger=logger)
                    _udp_log(logger, f"[UDP->ROBOT] Sent trajectory: {traj}")
                except Exception as e:
                    _udp_log(logger, f"[ERROR] Failed to send trajectory '{traj}': {e}")
                    i += 1
                    continue

                # 1b) wait for ACK:<traj> or ACK:COORDS_STAGED_RAD (with retry)
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

                # 1c) Race guard: give the robot time to flip "armed=true"
                time.sleep(STAGE_TO_GO_DELAY_S)

                # 1d) Optional: small drain again so a late ACK doesn't precede 'g' handling
                _drain_control_socket(max_ms=20, logger=logger)

                # 2) send 'g'
                try:
                    _sendto_robot(b"g", logger=logger)
                    _udp_log(logger, "[UDP->ROBOT] Sent opcode: g")
                except Exception as e:
                    _udp_log(logger, f"[ERROR] Failed to send 'g': {e}")
                    i += 1
                    continue

                # 2b) wait for ACK:g (with retry)
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
                    # motion-begin trigger
                    trig = ack_to_trigger.get("g") or ack_to_trigger.get(
                        getattr(_config, "ROBOT_OPCODES", {}).get("GO", "g") if _config else "g"
                    )
                    if trig and trig not in fired_triggers:
                        _send_marker_trigger(_config, logger, trig)
                        fired_triggers.add(trig)

                # Clear any stale pending state
                _pending_target_ready = False          # <<< ADDED
                _pending_target_ctx   = None           # <<< ADDED

                i += 1
                continue

            # --- 'q' special ---
            if op == QUERY_OPCODE:
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
            try:
                _sendto_robot(op.encode("utf-8"), logger=logger)
                _udp_log(logger, f"[UDP->ROBOT] Sent opcode: {op}")
            except Exception as e:
                _udp_log(logger, f"[ERROR] Failed to send opcode '{op}': {e}")

            # <<< ADDED: flip/clear pending-target flag for the narrow ops we care about
            if op == "c":
                _pending_target_ready = True
                _pending_target_ctx   = None  # control side doesn't need a ctx object
            elif op in ("m", "h"):
                _pending_target_ready = False
                _pending_target_ctx   = None

            i += 1

    # ---------------- UI + PASSIVE RECV ----------------
    clock = pygame.time.Clock()
    while pygame.time.get_ticks() < end_time:
        # draw
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

        # EEG
        if eeg_state is not None:
            try:
                eeg_state.update()
            except Exception as e:
                _udp_log(logger, f"[WARN] eeg_state.update() failed: {e}")

        # drain all inbound datagrams from the same bound socket
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
                    # Never trigger for coordinate staging
                    if token == "COORDS_STAGED_RAD":
                        _udp_log(logger, "[ROBOT->UDP] ACK:COORDS_STAGED_RAD (staged)")
                        continue
                    # Map control ACKs to triggers (GO/HOME/PAUSE/RESUME/STOP/MASTER_*)
                    trig = ack_to_trigger.get(token)
                    if trig and trig not in fired_triggers:
                        _send_marker_trigger(_config, logger, trig)
                        fired_triggers.add(trig)
                    # Always log ACKs (incl. ACK:q;seq=...)
                    _udp_log(logger, f"[ROBOT->UDP] {txt}")
                    continue

                # Non-ACK: capture first Q JSON
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
    ack_timeout: float = 0.8,   # a touch friendlier than 0.5s; tune as needed
    max_retries: int = 0,
    capture_query: bool = False
):
    """
    One-off sender that ALWAYS uses the import-time bound control socket.
    Destination (ip,port) is honored; source is fixed CONTROL_BIND.

    - expect_ack=True: wait for ACK:<token>.
      * For 'q', accept 'ACK:q;seq=...' (prefix match).
      * For others, exact match to the message ('h','g','p','r','s', etc.)
    - capture_query=True & message=='q': capture first non-ACK telemetry after send.

    Returns:
      None (legacy) or (acked: bool, query_payload|None) depending on flags.
    """
    # unified logger
    def _log(msg: str):
        if not quiet:
            _udp_log(logger, msg)

    s = _ensure_control_socket(logger)
    if s is None:
        _log("[ERROR] Control socket unavailable; cannot send.")
        return (False, None) if expect_ack else None

    # Robot endpoint for reply filtering
    robot_ip   = getattr(_config, "UDP_ROBOT", {}).get("IP", _ROBOT_IP)
    robot_port = int(getattr(_config, "UDP_ROBOT", {}).get("PORT", _ROBOT_PORT))

    attempts = 0
    while True:
        # IMPORTANT: drain BEFORE sending to avoid eating the fresh ACK
        try:
            _drain_control_socket(max_ms=30, logger=logger)
        except NameError:
            pass

        # Send
        try:
            s.sendto(message.encode("utf-8"), (ip, port))
        except Exception as e:
            _log(f"UDP send failed to {ip}:{port} — {e}")
            return (False, None) if expect_ack else None

        # Consistent TX label + wording
        try:
            ro = getattr(_config, "UDP_ROBOT", {}) or {}
            mk = getattr(_config, "UDP_MARKER", {}) or {}
            robot_ip, robot_port = ro.get("IP", _ROBOT_IP), int(ro.get("PORT", _ROBOT_PORT))
            marker_ip, marker_port = mk.get("IP"), int(mk.get("PORT")) if mk.get("PORT") else None
        except Exception:
            robot_ip, robot_port = _ROBOT_IP, _ROBOT_PORT
            marker_ip, marker_port = None, None

        is_robot  = (ip == robot_ip and int(port) == int(robot_port))
        is_marker = (marker_ip is not None and ip == marker_ip and int(port) == int(marker_port))
        is_traj   = _is_coords_string(message)

        prefix = "[UDP->ROBOT]" if is_robot else ("[TRIGGER]" if is_marker else f"[UDP->{ip}:{port}]")
        kind   = "Sent trajectory" if is_traj else "Sent opcode"

        _log(f"{prefix} {kind}: {message}")

        # -------------------------  <<< ADD: pending-target bookkeeping >>>  -------------------------
        # Make standalone 'g' legal after a 'c' sent via this helper, and clear on 'm'/'h'.
        # Uses config tokens if present so it stays correct even if you remap opcodes.
        try:
            ro = getattr(_config, "ROBOT_OPCODES", {}) or {}
            tok_c = ro.get("MASTER_LOCK",   "c")
            tok_m = ro.get("MASTER_UNLOCK", "m")
            tok_h = ro.get("HOME",          "h")
        except Exception:
            tok_c, tok_m, tok_h = "c", "m", "h"

        # keep state in sync
        global _pending_target_ready, _pending_target_ctx
        if message == tok_c:
            _pending_target_ready = True
            _pending_target_ctx   = None
        elif message in (tok_m, tok_h):
            _pending_target_ready = False
            _pending_target_ctx   = None
        # --------------------------------------------------------------------------------------------

        # Fast exit
        if not expect_ack and not capture_query:
            return None

        # Wait window
        end = time.time() + ack_timeout
        acked = False
        query_payload = None

        while time.time() < end:
            r, _, _ = select.select([s], [], [], max(0.0, end - time.time()))
            if not r:
                continue
            try:
                data, addr = s.recvfrom(65535)
            except Exception:
                break

            # Accept only from robot
            src_ip, src_port = addr[0], int(addr[1])
            if src_ip != robot_ip or src_port != robot_port:
                continue

            txt = data.decode("utf-8", errors="ignore").strip()

            # ACK handling
            if expect_ack and txt.startswith(ack_prefix):
                token = txt[len(ack_prefix):].strip()
                if message.startswith("q"):
                    # accept ACK:q;seq=...
                    if token.startswith("q"):
                        _log(f"[ROBOT->UDP] {txt}")
                        acked = True
                        if not capture_query: return (True, None)
                else:
                    if token == message:
                        _log(f"[ROBOT->UDP] {txt}")
                        acked = True
                        if not capture_query: return (True, None)
                # unrelated ACKs ignored during this wait
                continue

            # Telemetry path for q
            if capture_query and message.startswith("q") and not txt.startswith(ack_prefix) and query_payload is None:
                query_payload = txt
                if not expect_ack:
                    return query_payload
                # else keep waiting until ACK or timeout

        # Window ended
        if expect_ack and acked:
            return (True, query_payload)
        if not expect_ack:
            return query_payload

        # Retry?
        if attempts < max_retries:
            attempts += 1
            _log(f"[RETRY] No expected ACK for '{message}' after {ack_timeout:.2f}s. Retrying ({attempts}/{max_retries})…")
            continue

        return (False, query_payload)
