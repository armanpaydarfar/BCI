#!/usr/bin/env bash
# setup_linux_ufw.sh — open ufw for the frame-relay server on the Linux
# device host.
#
# Per SoftwareDocs/GPU_Service_Host_Architecture_Plan.md §4.9:
#   Inbound on Linux:
#     TCP 5591 — frame_relay server (Windows GPU host dials in)
#
# All other perception sockets bind on Windows; Linux is purely a client
# for those, so no further inbound rules are needed.
#
# Run as root or with sudo. Idempotent — re-running is harmless.

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "[setup_linux_ufw] must run as root (sudo)." >&2
    exit 1
fi

if ! command -v ufw >/dev/null 2>&1; then
    echo "[setup_linux_ufw] ufw not installed. Install with: apt install ufw" >&2
    exit 2
fi

ufw allow 5591/tcp comment 'BCI frame_relay (Windows GPU host)'

# Optional: restrict to a specific source LAN. Uncomment + set
# WINDOWS_HOST_IP to harden against rogue LAN clients.
# WINDOWS_HOST_IP="192.168.1.50"
# ufw allow from "$WINDOWS_HOST_IP" to any port 5591 proto tcp \
#     comment 'BCI frame_relay (Windows GPU host, restricted)'

ufw status verbose
echo "[setup_linux_ufw] done."
