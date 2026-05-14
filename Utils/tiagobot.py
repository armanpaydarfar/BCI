# utils/tiagobot.py
"""
Serial helpers for the Tiagobot Arduino actuator.

This module owns the Tiagobot serial port (Tier 1 hardware I/O). It is
imported by ExperimentDriver_Online_Tiagobot.py; nothing else should open
the Tiagobot port.

Wire protocol (matches tools/tiago_arduino/Final_code.ino):
- GO  : `"{analog},{angle},{delay}\n"`  — go to position; the actuator
        remains at the target until the next command.
- HOME: `"h\n"`                          — center servo + retract actuator.

No ACK is exchanged; the sketch prints status to serial but the driver does
not parse it. Realtime hardware errors (port closed mid-experiment, etc.)
propagate to the caller per the project's fail-fast policy.
"""
import importlib
import sys
import time
from pathlib import Path

import serial


# =========================================================
# Config loader (mirrors Utils/networking.py:_load_config)
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

# Snapshot at import (mirrors Utils/networking.py:67). When True, send_letter
# and send_home log the intended write but do not touch the serial port; the
# driver still receives a real Serial object from open_port for parity.
SIMULATION_MODE = (
    bool(getattr(_config, "SIMULATION_MODE", False)) if _config is not None else False
)


# =========================================================
# Constants
# =========================================================
# Calibrated locations for the Tiagobot servo (9°–69°, center 39°) plus the
# linear actuator (analog 350–900). Tuple = (analog, servo_angle, servo_step_delay_ms).
# The third value is the per-degree servo step delay (rotation speed), not a
# wait-time. Source: manipulandum_udp_code.py (Tiago, 2026-01-07).
LOCATIONS = {
    "A": (620, 14, 220),
    "B": (720, 24, 600),
    "C": (650, 32, 1150),
    "D": (890, 33, 1150),
    "E": (890, 39, 80),
    "F": (890, 45, 1150),
    "G": (650, 46, 1150),
    "H": (670, 53, 600),
    "I": (620, 64, 220),
}

HOME_COMMAND = "h\n"

# Banner printed by the sketch after calibrateServo() + calibrateLinAct().
# We wait for it on port open so the first letter is not sent into a
# mid-calibration Arduino.
CALIBRATION_READY_MARKER = "Calibration complete."
CALIBRATION_TIMEOUT_S = 30.0


# =========================================================
# Port lifecycle
# =========================================================
def open_port(port, baud, logger):
    """Open the Tiagobot Arduino serial port and wait for the sketch's
    `Calibration complete.` banner.

    Returns the serial.Serial handle, or None if `port` is empty or
    SIMULATION_MODE is set (the driver still runs in either case; the
    send_* helpers no-op when given None).

    Raises serial.SerialException on real open failures (port specified
    but unavailable) — caller decides whether to abort.
    """
    if SIMULATION_MODE:
        if logger is not None:
            logger.log_event("Tiagobot: SIMULATION_MODE — skipping serial open.")
        return None
    if not port:
        if logger is not None:
            logger.log_event("Tiagobot: no port configured — skipping serial open.")
        return None

    if logger is not None:
        logger.log_event(f"Tiagobot: opening {port} @ {baud} baud...")

    ser = serial.Serial(port, baud, timeout=0.5)

    # The sketch's calibrateServo() + calibrateLinAct() runs full-stroke
    # on every port open. Read lines until the ready banner appears or
    # the timeout fires. 2s (the glove driver's wait) is not enough.
    deadline = time.monotonic() + CALIBRATION_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            line = ser.readline().decode("utf-8", errors="replace").strip()
        except Exception:
            line = ""
        if not line:
            continue
        if logger is not None:
            logger.log_event(f"Tiagobot[boot]: {line}")
        if CALIBRATION_READY_MARKER in line:
            if logger is not None:
                logger.log_event("Tiagobot: calibration complete.")
            return ser

    ser.close()
    raise TimeoutError(
        f"Tiagobot did not emit '{CALIBRATION_READY_MARKER}' within "
        f"{CALIBRATION_TIMEOUT_S:.0f}s on {port}"
    )


def close_port(ser, logger):
    """Best-effort close. Safe to call with None."""
    if ser is None:
        return
    try:
        ser.close()
    except Exception as e:
        if logger is not None:
            logger.log_event(f"Tiagobot: error closing port: {e}", level="error")


# =========================================================
# Wire-level sends
# =========================================================
def send_letter(ser, letter, logger):
    """Look up `letter` in LOCATIONS and write the CSV command.

    No-op (log only) if `ser` is None (SIMULATION_MODE or unconfigured port).
    """
    if letter not in LOCATIONS:
        raise ValueError(f"Tiagobot: unknown location letter {letter!r}")
    analog, angle, delay_ms = LOCATIONS[letter]
    payload = f"{analog},{angle},{delay_ms}\n"

    if ser is None:
        if logger is not None:
            logger.log_event(
                f"Tiagobot[SIM]: would send letter {letter} -> {payload.strip()}"
            )
        return

    ser.write(payload.encode("utf-8"))
    if logger is not None:
        logger.log_event(f"Tiagobot: sent {letter} -> {payload.strip()}")


def send_home(ser, logger):
    """Write the HOME command. No-op (log only) if `ser` is None."""
    if ser is None:
        if logger is not None:
            logger.log_event("Tiagobot[SIM]: would send HOME (h\\n)")
        return
    ser.write(HOME_COMMAND.encode("utf-8"))
    if logger is not None:
        logger.log_event("Tiagobot: sent HOME (h\\n)")
