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
import subprocess
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

# Completion markers printed by the sketch after each command. Used by
# wait_for_completion() so the driver can sync to actual motion end
# instead of guessing with a fixed timeout — A-I moves vary from ~5 s
# (shortest, e.g. 'E' at delayTime=80) to ~30+ s (e.g. 'C' at 1150).
TARGET_REACHED_MARKER = "Target Location Reached."
HOMED_MARKER = "Homed."

# Default per-motion wait ceiling. Larger than any plausible single move,
# but bounded so the trial loop doesn't hang forever on a stuck actuator.
DEFAULT_MOTION_TIMEOUT_S = 60.0

# USB descriptor of the Arduino on the Tiagobot. Used by find_tiagobot_port
# to disambiguate Tiagobot from the glove Arduino when both are plugged
# into the same Linux host (where /dev/ttyACM* enumeration is otherwise
# non-deterministic across reboots).
TIAGOBOT_USB_VID = 0x2341  # Arduino SA
TIAGOBOT_USB_PID = 0x0042  # Mega 2560 R3

# USB descriptor of the glove Arduino (used by find_glove_port for the
# same reason as TIAGOBOT_USB_*). The glove is a stock Arduino Uno R3.
GLOVE_USB_VID = 0x2341  # Arduino SA
GLOVE_USB_PID = 0x0043  # Uno R3

# Banner printed by the sketch after calibrateServo() + calibrateLinAct().
# Latched by `calibrate()` (the explicit operator-triggered calibration
# command). Servo sweep alone takes ~10 s; the linear actuator
# extend+retract adds another 10-30 s depending on stroke length and
# friction. 60 s leaves comfortable headroom.
CALIBRATION_READY_MARKER = "Calibration complete."
CALIBRATION_TIMEOUT_S = 60.0

# Banner printed by the sketch on boot (after the Serial.begin and
# before the main loop). Latched by `open_port` as proof the device is
# alive. Substring match — the actual line includes a usage hint.
BOOT_READY_MARKER = "Tiagobot ready."

# Handshake / ping. open_port writes this and waits for HANDSHAKE_REPLY
# on the WARM-reopen path (Arduino already past boot, silent), or as a
# fallback after the cold-boot drain finishes/times-out. Returns as soon
# as the OK arrives, so warm reopens still feel instant — the ceiling
# only matters when the Arduino is mid-bootloader.
#
# 10 s ceiling accommodates the Mega 2560's STK500v2 bootloader delay
# (~1-2 s before the sketch's setup() runs) plus the rare case where
# the panel's first port open races against the bootloader and the `?`
# byte sits in the UART buffer until loop() starts.
HANDSHAKE_REQUEST = b"?\n"
HANDSHAKE_REPLY = "OK"
HANDSHAKE_TIMEOUT_S = 10.0

# Cold-boot path: on USB plug-in (or any DTR-reset), the sketch's
# setup() prints chatter while parking the actuator at home position.
# The wait is adaptive: as long as the Arduino is producing output,
# we keep listening (resets the deadline on every fresh line). Only
# falls through to the `?` handshake after BOOT_SILENT_TIMEOUT_S of
# silence without seeing BOOT_READY_MARKER, OR after the absolute
# ceiling BOOT_DRAIN_TIMEOUT_S. The ceiling covers worst-case
# retract from a fully-extended actuator (~20-30 s) plus headroom.
BOOT_DRAIN_TIMEOUT_S = 60.0
BOOT_SILENT_TIMEOUT_S = 5.0

# Explicit calibration trigger sent by `calibrate()`.
CALIBRATE_REQUEST = b"t\n"


# =========================================================
# Port lifecycle
# =========================================================
def _find_arduino_by_usb_id(vid, pid, role_name, logger=None):
    """Internal: scan pyserial.tools.list_ports.comports for a device
    matching `(vid, pid)`. Returns the device path on exactly-one match,
    None on zero / >1 matches. Shared between Tiagobot (Mega 2560 R3)
    and glove (Uno R3) auto-detect."""
    try:
        import serial.tools.list_ports
    except Exception as e:
        if logger is not None:
            logger.log_event(f"{role_name} auto-detect: pyserial unavailable ({e})")
        return None

    try:
        ports = list(serial.tools.list_ports.comports())
    except Exception as e:
        if logger is not None:
            logger.log_event(f"{role_name} auto-detect: comports() failed ({e})")
        return None

    matches = [p for p in ports if p.vid == vid and p.pid == pid]

    if logger is not None:
        for p in matches:
            logger.log_event(
                f"{role_name} auto-detect: candidate {p.device} "
                f"({p.description or 'n/a'}, serial={p.serial_number or 'n/a'})"
            )

    if len(matches) == 1:
        return matches[0].device
    if len(matches) == 0:
        if logger is not None:
            logger.log_event(
                f"{role_name} auto-detect: no device with USB ID "
                f"{vid:04x}:{pid:04x} found."
            )
        return None
    if logger is not None:
        logger.log_event(
            f"{role_name} auto-detect: {len(matches)} devices with USB ID "
            f"{vid:04x}:{pid:04x} found — set the port explicitly to disambiguate."
        )
    return None


def find_tiagobot_port(logger=None):
    """Scan available serial ports for an Arduino Mega 2560 R3 and return
    its device path, or None if none found / multiple ambiguous.

    Used as a fallback when TIAGOBOT_PORT is empty: lets the panel + driver
    auto-pick the Tiagobot Arduino out of a multi-Arduino setup without
    the operator having to read `ls /dev/serial/by-id/` and edit
    config_local.py.

    Returns None (not raises) on any failure — caller decides whether the
    absence is fatal.
    """
    return _find_arduino_by_usb_id(
        TIAGOBOT_USB_VID, TIAGOBOT_USB_PID, "Tiagobot", logger=logger,
    )


def find_glove_port(logger=None):
    """Scan available serial ports for an Arduino Uno R3 (the glove
    board) and return its device path, or None if none found / multiple
    ambiguous. Used by the Tiagobot driver and the panel when
    ARDUINO_PORT is empty and TIAGOBOT_USE_GLOVE is True.
    """
    return _find_arduino_by_usb_id(
        GLOVE_USB_VID, GLOVE_USB_PID, "Glove", logger=logger,
    )


def open_port(port, baud, logger, yield_callback=None):
    """Open the Tiagobot Arduino serial port and verify the device is
    alive via the lightweight `?`->`OK` handshake.

    This is a FAST operation (sub-second on a responsive Arduino). It
    does NOT trigger the actuator calibration sweep — that lives in
    `calibrate()` and is now an explicit, operator-triggered step
    (panel "Test" button, or driver if you decide to call it).

    Returns the serial.Serial handle, or None if `port` is empty or
    SIMULATION_MODE is set (the driver still runs in either case; the
    send_* helpers no-op when given None).

    `yield_callback` is an optional zero-arg callable invoked once per
    serial read iteration during the handshake wait — same Qt-event-
    pump pattern as `calibrate()`. Largely vestigial here since the
    handshake completes well within HANDSHAKE_TIMEOUT_S, but kept for
    signature symmetry.

    Raises serial.SerialException on real open failures, or TimeoutError
    if the handshake never returns within HANDSHAKE_TIMEOUT_S (device
    plugged in but sketch is wedged / not flashed with the new
    handshake-aware firmware).
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

    # Disable HUPCL on this tty so closing the port doesn't drop DTR.
    # Without this, every close-then-reopen across the session triggers
    # an Arduino auto-reset (DTR low->high edge on next open). With the
    # post-2026-05-27 handshake-only boot the reset only costs ~100 ms
    # of banner-print (no actuator sweep), but we still want to avoid
    # spurious sketch restarts — they reset the BOOT_READY_MARKER state
    # and chew through the serial buffer needlessly.
    try:
        subprocess.run(["stty", "-F", port, "-hupcl"],
                       check=False, timeout=2)
    except Exception as e:
        if logger is not None:
            logger.log_event(f"Tiagobot: could not clear HUPCL on {port}: {e}")

    ser = serial.Serial(port, baud, timeout=0.2)

    # Branch on cold-boot vs warm-reopen:
    #
    # COLD BOOT (USB plug-in, DTR-reset): the sketch is mid-setup(),
    # parking the actuator at home. setup() prints chatter via
    # displayOutput() during park, then BOOT_READY_MARKER, then enters
    # loop(). A `?` sent now would block in the Arduino's UART buffer
    # until loop() drains it (5-10 s later). Instead, wait for the
    # boot banner and treat that as proof of life.
    #
    # WARM REOPEN (close-then-reopen with HUPCL cleared): the sketch
    # is silent in loop(). Send `?` and expect an immediate OK.
    #
    # Discriminate by checking whether anything arrives in the first
    # 500 ms after open. Silence -> warm reopen. Chatter -> cold boot.
    quick_probe_deadline = time.monotonic() + 0.5
    cold_boot = False
    while time.monotonic() < quick_probe_deadline:
        if ser.in_waiting:
            cold_boot = True
            break
        time.sleep(0.02)

    if cold_boot:
        if logger is not None:
            logger.log_event(
                "Tiagobot: cold-boot chatter detected; waiting for boot banner..."
            )
        # Adaptive wait: reset last_chatter on every fresh line. As
        # long as the Arduino keeps printing (e.g., displayOutput()
        # during a long retract from a fully-extended actuator), we
        # keep listening. Only give up after BOOT_SILENT_TIMEOUT_S of
        # silence without seeing the marker, OR after the absolute
        # BOOT_DRAIN_TIMEOUT_S ceiling.
        boot_start = time.monotonic()
        last_chatter = boot_start
        while True:
            now = time.monotonic()
            if now - boot_start > BOOT_DRAIN_TIMEOUT_S:
                if logger is not None:
                    logger.log_event(
                        f"Tiagobot: BOOT_READY_MARKER not seen in "
                        f"{BOOT_DRAIN_TIMEOUT_S:.0f}s ceiling; falling "
                        f"back to ? handshake."
                    )
                break
            if now - last_chatter > BOOT_SILENT_TIMEOUT_S:
                if logger is not None:
                    logger.log_event(
                        f"Tiagobot: silent for {BOOT_SILENT_TIMEOUT_S:.0f}s "
                        f"without BOOT_READY_MARKER; falling back to ? "
                        f"handshake."
                    )
                break
            try:
                line = ser.readline().decode("utf-8", errors="replace").strip()
            except Exception:
                line = ""
            if yield_callback is not None:
                try:
                    yield_callback()
                except Exception:
                    pass
            if not line:
                continue
            last_chatter = now
            if logger is not None:
                logger.log_event(f"Tiagobot[boot]: {line}")
            if BOOT_READY_MARKER in line:
                if logger is not None:
                    logger.log_event(
                        "Tiagobot: boot complete; device alive (banner seen)."
                    )
                return ser

    # Warm-reopen path (or cold-boot fallback): explicit ?->OK probe.
    try:
        ser.write(HANDSHAKE_REQUEST)
    except Exception as e:
        ser.close()
        raise serial.SerialException(
            f"Tiagobot: write to {port} failed during handshake: {e}"
        ) from e

    deadline = time.monotonic() + HANDSHAKE_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            line = ser.readline().decode("utf-8", errors="replace").strip()
        except Exception:
            line = ""
        if yield_callback is not None:
            try:
                yield_callback()
            except Exception:
                pass
        if not line:
            continue
        if logger is not None:
            logger.log_event(f"Tiagobot[hs]: {line}")
        if HANDSHAKE_REPLY in line:
            if logger is not None:
                logger.log_event("Tiagobot: handshake OK; device alive.")
            return ser

    ser.close()
    raise TimeoutError(
        f"Tiagobot did not reply '{HANDSHAKE_REPLY}' to handshake within "
        f"{HANDSHAKE_TIMEOUT_S:.1f}s on {port}. Is the sketch flashed "
        f"with handshake support (post-2026-05-27 firmware)?"
    )


def calibrate(ser, logger, yield_callback=None):
    """Send the explicit calibration command and wait for the
    `Calibration complete.` banner. Triggers the ~15-30 s actuator
    sweep on the Arduino (calibrateServo + calibrateLinAct).

    Called explicitly by the panel "Test" button at the start of a
    session. The driver does NOT call this — operator decides when to
    calibrate. Motion commands work without prior calibration using
    the sketch's hardcoded servo / linear-actuator defaults; accuracy
    may be lower, but the device won't refuse to move.

    Returns True on success, False if `ser` is None (SIMULATION_MODE).
    Raises TimeoutError if the banner doesn't arrive within
    CALIBRATION_TIMEOUT_S.

    `yield_callback` pumps Qt events from a long-running GUI thread,
    same pattern as the old open_port. Pass
    `QApplication.processEvents` from the panel.
    """
    if ser is None:
        if logger is not None:
            logger.log_event("Tiagobot[cal]: SIMULATION_MODE — skipping.")
        return False

    if logger is not None:
        logger.log_event("Tiagobot[cal]: sending calibrate (sweep starts now)...")

    try:
        ser.write(CALIBRATE_REQUEST)
    except Exception as e:
        raise serial.SerialException(
            f"Tiagobot: write of calibrate command failed: {e}"
        ) from e

    deadline = time.monotonic() + CALIBRATION_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            line = ser.readline().decode("utf-8", errors="replace").strip()
        except Exception:
            line = ""
        if yield_callback is not None:
            try:
                yield_callback()
            except Exception:
                # Callback errors are non-fatal — keep waiting for the
                # banner. A flaky GUI shouldn't break hardware setup.
                pass
        if not line:
            continue
        if logger is not None:
            logger.log_event(f"Tiagobot[cal]: {line}")
        if CALIBRATION_READY_MARKER in line:
            if logger is not None:
                logger.log_event("Tiagobot: calibration complete.")
            return True

    raise TimeoutError(
        f"Tiagobot did not emit '{CALIBRATION_READY_MARKER}' within "
        f"{CALIBRATION_TIMEOUT_S:.0f}s after calibrate request"
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


def wait_for_completion(ser, marker, timeout=DEFAULT_MOTION_TIMEOUT_S,
                        logger=None, on_tick=None):
    """Block until the sketch prints `marker` on the serial port, or
    `timeout` seconds elapse. Returns True on completion, False on timeout.

    Used by the driver to sync to actual motion-end: after sending a
    letter, wait for ``TARGET_REACHED_MARKER``; after HOME, wait for
    ``HOMED_MARKER``. Variable-time moves (5-30+ s depending on the
    letter's analog/angle/delay triple) are handled correctly — there's
    no fixed sleep that's either too short (next trial starts mid-motion)
    or too long (wasted dead time on fast moves).

    ``on_tick`` is an optional zero-arg callable invoked once per
    readline iteration. Use it from the driver to pump pygame events,
    update EEG state, refresh the display, etc. — keeps the realtime
    loop alive during a long wait. Callback errors are silently swallowed.

    No-op (returns True immediately) if `ser` is None (SIMULATION_MODE
    or unconfigured port) — the SIM-mode driver doesn't have an Arduino
    to wait for.
    """
    if ser is None:
        if logger is not None:
            logger.log_event(f"Tiagobot[SIM]: would wait for {marker!r}")
        return True

    deadline = time.monotonic() + timeout
    # The Arduino sketch emits a per-tick position line on every step of
    # the actuator move ("Linear Analog Reading: NNN || Angle Reading: NN").
    # We need to drain those from the serial buffer so it doesn't back up,
    # but the per-tick log lines flood the experiment log (~50 lines per
    # move × 2 moves per trial × 20 trials = ~2000 noise lines per
    # session) and obscure the real events. Track the latest tick line
    # silently and emit ONE summary line ("final position — ...") on
    # marker hit so the log shows where the actuator stopped without the
    # intermediate spam.
    last_tick_line = None
    while time.monotonic() < deadline:
        try:
            line = ser.readline().decode("utf-8", errors="replace").strip()
        except Exception:
            line = ""
        if on_tick is not None:
            try:
                on_tick()
            except SystemExit:
                # Propagate Ctrl+C / window-close cleanly so the driver's
                # try/finally hardware cleanup can run.
                raise
            except Exception:
                pass
        if not line:
            continue
        is_position_tick = (
            "Linear Analog Reading" in line and "Angle Reading" in line
        )
        if is_position_tick:
            last_tick_line = line
        elif logger is not None:
            logger.log_event(f"Tiagobot: {line}")
        if marker in line:
            if logger is not None and last_tick_line is not None:
                logger.log_event(f"Tiagobot: final position — {last_tick_line}")
            return True

    if logger is not None:
        logger.log_event(
            f"Tiagobot: timeout after {timeout:.0f}s waiting for {marker!r}",
            level="error",
        )
    return False
