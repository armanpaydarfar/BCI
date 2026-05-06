# config_local.example.py
#
# Copy to config_local.py on each machine and edit in place:
#   cp config_local.example.py config_local.py
#
# config_local.py is gitignored. It supplies machine-local overrides on
# top of the committed config.py (paths, network endpoints, USB device
# paths). Imported via `from config_local import *` at the bottom of
# config.py, so anything defined here shadows the default in config.py.
#
# You only need to set the keys that actually differ from config.py
# defaults — unset keys keep the committed default.

# =============================================================================
# Paths
# =============================================================================
# Repo root on this machine. Used by drivers to resolve sibling scripts.
WORKING_DIR = "/path/to/Harmony/"
# Where XDFs, training data, vlm session artifacts live.
DATA_DIR    = "/path/to/CurrentStudy"

# =============================================================================
# Gaze service (gaze_runner.py UDP 5588)
# =============================================================================
# GAZE_UDP_IP is the dial host that the panel + experiment driver hit.
# GAZE_BIND_HOST is what gaze_runner.py binds on. In production on
# Windows GPU host, set BIND="0.0.0.0" and on the Linux operator box
# set GAZE_UDP_IP to the Windows LAN/Tailscale IP. For single-machine
# dev both stay 127.0.0.1.
GAZE_UDP_IP    = "127.0.0.1"
GAZE_BIND_HOST = "127.0.0.1"

# =============================================================================
# Pupil Labs Neon Companion phone
# =============================================================================
# IP from the Companion app's streaming icon, e.g. "10.42.0.100". Leave
# empty to use mDNS auto-discovery (works on home/hotspot, blocked on
# enterprise IoT VLANs).
NEON_COMPANION_HOST = ""

# =============================================================================
# Perception frame source / service hosting
# =============================================================================
# "local" → service opens Neon directly (single-machine).
# "remote" → service consumes envelopes from the frame_relay TCP server.
PERCEPTION_FRAME_SOURCE = "local"

# True on the Linux operator panel when services run on the Windows GPU
# host; False on Windows or single-machine dev. Drives panel UX (disables
# local Start/Stop, enables remote-status badge polling).
SERVICES_HOSTED_REMOTELY = False

# =============================================================================
# Frame relay (Utils/frame_relay.py)
# =============================================================================
# Bind host on the relay server (the box that owns Neon). 0.0.0.0 to
# accept LAN peers, 127.0.0.1 for loopback-only.
FRAME_RELAY_HOST      = "127.0.0.1"
# Dial host on the consumer side. Set to the relay's LAN/Tailscale IP
# in production; loopback for single-machine.
FRAME_RELAY_DIAL_HOST = "127.0.0.1"

# =============================================================================
# VLM service
# =============================================================================
# Sibling clone of the harmony_vlm repo on this machine.
VLM_REPO_DIR     = "/path/to/harmony_vlm"
# Dial host (panel + driver) and bind host (vlm_service.py). Same
# convention as GAZE_UDP_IP / GAZE_BIND_HOST.
VLM_SERVICE_HOST = "127.0.0.1"
VLM_BIND_HOST    = "127.0.0.1"

# =============================================================================
# Arduino USB device path
# =============================================================================
# Linux: "/dev/ttyACM0". Windows: "COM3" etc.
ARDUINO_PORT = ""
