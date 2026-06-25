"""
panel/netutils.py — network / process helpers for the control panel.

Behaviour-preserving extraction of the module-level network-discovery and
process-management helpers that used to live at the top of control_panel.py:
sleep inhibition, LAN / Tailscale IP discovery, orphan-service cleanup, the
marker UDP port lookup, and the UDP port-in-use probe. No Qt, no panel state.
Lives in a leaf module so panel collaborators import these by name instead of
the former bare module-level references.

Imports _HCFG / _IS_WINDOWS from panel.constants (DAG: constants → netutils;
netutils never imports control_panel — no cycle).

The module-level LAN / Tailscale IP discovery + the report prints run at import
time, exactly as they did at the top of control_panel.py: control_panel imports
this module, so the discovery and prints fire on panel startup unchanged.
"""

from __future__ import annotations

import socket
import subprocess
from typing import Optional

from panel.constants import _HCFG, _IS_WINDOWS


def _sleep_inhibit(enable: bool) -> None:
    """Prevent / allow Windows from sleeping while perception services run.

    Windows-only — uses kernel32.SetThreadExecutionState with
    ES_CONTINUOUS | ES_SYSTEM_REQUIRED. Per
    SoftwareDocs/projects/harmony-bci/gpu-service/architecture-plan.md §4.10 the GPU host
    must not sleep mid-session. POSIX is a no-op (Linux deployment uses
    systemd-inhibit / external power management).
    """
    if not _IS_WINDOWS:
        return
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enable else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))
    except Exception:
        # Best-effort — sleep inhibit is a hardening, not a correctness gate.
        pass


def _detect_lan_ip() -> Optional[str]:
    """Best-effort LAN IP discovery. Opens a UDP socket and asks the
    kernel which interface it would use to reach a routable address —
    no packet is actually sent. Returns None if no route is available
    (offline, captive portal, etc.)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        try:
            s.close()
        except OSError:
            pass


def _detect_tailscale_ip() -> Optional[str]:
    """Return this host's Tailscale IPv4 if the daemon is up + signed in,
    else None. The tailnet IP is the address remote peers (Windows GPU
    host, lab box) should dial — it's stable across physical networks,
    unlike the LAN IP from _detect_lan_ip()."""
    try:
        out = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True, text=True, timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    ip = (out.stdout or "").strip().splitlines()
    return ip[0].strip() if ip and ip[0].strip() else None


_LAN_IP = _detect_lan_ip()
_TS_IP = _detect_tailscale_ip()
if _LAN_IP:
    print(f"[control_panel] Linux LAN IP = {_LAN_IP} (physical network)", flush=True)
else:
    print("[control_panel] Linux LAN IP = <unavailable> (no route)", flush=True)
if _TS_IP:
    print(
        f"[control_panel] Linux Tailscale IP = {_TS_IP}  "
        f"— remote peers should use this in FRAME_RELAY_DIAL_HOST "
        f"(stable across networks)",
        flush=True,
    )
else:
    print(
        "[control_panel] Tailscale not active — peers reach this box via the LAN IP above. "
        "Run `sudo tailscale up` to get a stable tailnet IP for cross-network use.",
        flush=True,
    )


def _kill_orphan_vlm_service() -> None:
    """Kill any leftover vlm_service.py processes from a prior crash.

    A killed launch can leave the service's python alive (mid-startup, or
    children it spawned) holding the UDP port. POSIX uses `pkill -f`; Windows
    walks tasklist for python.exe with the script in its command line and
    kills with taskkill.
    """
    if _IS_WINDOWS:
        try:
            out = subprocess.check_output(
                ["wmic", "process", "where",
                 "name='python.exe' and CommandLine like '%vlm_service.py%'",
                 "get", "ProcessId"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return
        for line in out.splitlines():
            line = line.strip()
            if line.isdigit():
                subprocess.run(
                    ["taskkill", "/F", "/PID", line],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
    else:
        subprocess.run(["pkill", "-f", "vlm_service.py"], capture_output=True)


def _marker_udp_port() -> int:
    if _HCFG is not None:
        return int(_HCFG.UDP_MARKER["PORT"])
    return 12345


# ----------------- UDP readiness probe -----------------
def _is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """
    UDP bind probe:
      - If we can bind, nobody is bound -> NOT in use
      - If bind fails, in use
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind((host, port))
        s.close()
        return False
    except OSError:
        s.close()
        return True
