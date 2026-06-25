"""
panel/serial_controller.py — Arduino / serial-port controls for the control panel.

First "owns-its-UI-section" collaborator: SerialController builds its own widgets
(port combo, Refresh/Test/Save/Send buttons, status LED) into the panel's main
grid via build_into(), owns the serial state (selected port + baud), and holds
the handler logic — all extracted verbatim from ControlPanel. Cross-cutting
concerns are injected as callbacks (log / set_led / timestamp / refresh_cmds) so
the controller has no back-reference into the panel beyond a QMessageBox parent.

Baud is read from / written to config via panel.config_io; the editable baud
spinbox itself lives in the panel's Runtime-config tab (rarely changed) and is
intentionally NOT owned here.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import serial
import serial.tools.list_ports

from PySide6.QtCore import QObject
from PySide6.QtWidgets import (
    QApplication, QComboBox, QGridLayout, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QWidget,
)

# ARDUINO_PORT is the committed default port; fall back if config is unavailable.
try:
    from config import ARDUINO_PORT
except ImportError:
    ARDUINO_PORT = ""

from panel.config_io import (
    read_arduino_baud_from_config,
    write_arduino_baud_to_config,
    write_arduino_port_to_config,
)
from panel.ui_utils import _fixed_v


class SerialController(QObject):
    """Owns the Arduino/serial row and its handlers.

    Injected callbacks (behaviour-identical to the former in-class calls):
      log(title, text)   — append to the panel's log buffer
      set_led(led, state)— colour the status LED for a state string
      timestamp()        — "HH:MM:SS" for log lines
      refresh_cmds()     — re-derive subprocess command env after a port change
    """

    def __init__(
        self,
        parent: QWidget,
        *,
        log: Callable[[str, str], None],
        set_led: Callable[[object, str], None],
        timestamp: Callable[[], str],
        refresh_cmds: Callable[[], None],
    ) -> None:
        super().__init__(parent)
        self._parent = parent
        self._log = log
        self._set_led = set_led
        self._ts = timestamp
        self._refresh_cmds = refresh_cmds

        # Arduino / BCI online config
        self.serial_port_name = ""
        try:
            self.serial_baudrate = str(read_arduino_baud_from_config(9600))
        except Exception:
            self.serial_baudrate = "9600"

    def build_into(self, grid: QGridLayout, row: int) -> int:
        """Build the Arduino row into the panel's main grid at ``row``; return
        the next free row. Widget tree + grid placement are identical to the
        former inline _build_ui block."""
        # ===== Arduino =====
        # Single-line layout matching the other module rows. Baud lives in
        # the Runtime config tab (rarely changed); per-test status updates
        # land in the Panel log buffer rather than a dedicated label, and
        # the LED reflects the last connection-test / send result.
        self.lbl_arduino = QLabel("●"); self._set_led(self.lbl_arduino, "stopped")
        self.cmb_serial_port = QComboBox()
        self.cmb_serial_port.currentIndexChanged.connect(self.on_serial_port_changed)
        self.btn_serial_refresh = QPushButton("Refresh")
        self.btn_serial_refresh.clicked.connect(self.on_serial_refresh)
        self.btn_serial_test = QPushButton("Test")
        self.btn_serial_test.clicked.connect(self.on_serial_test)
        self.btn_save_serial_to_config = QPushButton("Save → config")
        self.btn_save_serial_to_config.setToolTip(
            "Writes ARDUINO_PORT to config_local.py (machine-local) and "
            "ARDUINO_BAUD to config.py."
        )
        self.btn_save_serial_to_config.clicked.connect(self.on_save_serial_to_config)
        self.btn_send_1 = QPushButton("Send 1 (close)")
        self.btn_send_1.clicked.connect(self.on_send_arduino_one)
        self.btn_send_0 = QPushButton("Send 0 (open)")
        self.btn_send_0.clicked.connect(self.on_send_arduino_zero)

        arduino_row = QHBoxLayout()
        arduino_row.setContentsMargins(0, 0, 0, 0)
        arduino_row.addWidget(self.cmb_serial_port, 1)
        for w in (self.btn_serial_refresh, self.btn_serial_test,
                  self.btn_save_serial_to_config,
                  self.btn_send_1, self.btn_send_0):
            arduino_row.addWidget(w)
        arduino_row_holder = _fixed_v(QWidget()); arduino_row_holder.setLayout(arduino_row)
        grid.addWidget(QLabel("<b>Arduino</b>"), row, 0)
        grid.addWidget(self.lbl_arduino, row, 1)
        grid.addWidget(arduino_row_holder, row, 2, 1, 3)
        return row + 1

    # ----- handlers -----
    def on_save_serial_to_config(self):
        port = (self.serial_port_name or self.cmb_serial_port.currentData() or "").strip()
        if not port:
            QMessageBox.warning(self._parent, "Serial", "Select a serial port first.")
            return
        try:
            baud = int(str(self.serial_baudrate).strip())
        except ValueError:
            QMessageBox.warning(self._parent, "Serial", "Baud must be an integer (set it in Runtime config).")
            return
        try:
            write_arduino_port_to_config(port)
            write_arduino_baud_to_config(baud)
        except Exception as e:
            QMessageBox.warning(self._parent, "config.py", f"Failed to write Arduino settings:\n{e}")
            return
        self._log("Panel", f"[{self._ts()}] Saved ARDUINO_PORT={port} ARDUINO_BAUD={baud} to config.py\n")
        QMessageBox.information(self._parent, "Serial", "ARDUINO_PORT and ARDUINO_BAUD saved to config.py.")

    def on_serial_refresh(self):
        self.cmb_serial_port.blockSignals(True)
        self.cmb_serial_port.clear()

        try:
            ports = list(serial.tools.list_ports.comports())
        except Exception as e:
            self.cmb_serial_port.blockSignals(False)
            self._log("Panel", f"[{self._ts()}] Error listing serial ports: {e}\n")
            self._set_led(self.lbl_arduino, "error")
            return

        if not ports:
            self.cmb_serial_port.addItem("No ports found", "")
            self.serial_port_name = ""
            self.cmb_serial_port.blockSignals(False)
            self._log("Panel", f"[{self._ts()}] No serial ports found\n")
            self._set_led(self.lbl_arduino, "stopped")
            return

        for p in ports:
            desc = p.description or "n/a"
            text = f"{p.device} ({desc})"
            self.cmb_serial_port.addItem(text, p.device)

        idx = -1
        if self.serial_port_name:
            idx = self.cmb_serial_port.findData(self.serial_port_name)
        if idx < 0:
            idx = self.cmb_serial_port.findData(ARDUINO_PORT)
        if idx < 0:
            idx = 0

        self.cmb_serial_port.setCurrentIndex(idx)
        self.serial_port_name = self.cmb_serial_port.currentData() or ""
        self.cmb_serial_port.blockSignals(False)

        self._log("Panel", f"[{self._ts()}] Serial ports refreshed. Selected: {self.serial_port_name or 'None'}\n")

        self._refresh_cmds()

    def on_serial_port_changed(self, index: int):
        device = self.cmb_serial_port.itemData(index)
        self.serial_port_name = device or ""
        self._log("Panel", f"[{self._ts()}] Serial port set to: {self.serial_port_name}\n")
        # New port not yet validated — clear any prior pass/fail signal.
        self._set_led(self.lbl_arduino, "stopped")
        self._refresh_cmds()

    def _serial_baud_int(self) -> Optional[int]:
        """Return the configured baud rate as int, or None on parse failure.
        Source of truth is ``self.serial_baudrate`` (loaded from config and
        editable from the Runtime config tab)."""
        try:
            return int(str(self.serial_baudrate).strip())
        except (TypeError, ValueError):
            return None

    def on_serial_test(self):
        port = self.serial_port_name or self.cmb_serial_port.currentData()
        if not port:
            self._log("Panel", f"[{self._ts()}] Serial test: no port selected\n")
            self._set_led(self.lbl_arduino, "error")
            QMessageBox.information(self._parent, "Serial test", "No serial port selected.")
            return

        baud = self._serial_baud_int()
        if baud is None:
            self._log("Panel", f"[{self._ts()}] Serial test: invalid baudrate {self.serial_baudrate!r}\n")
            self._set_led(self.lbl_arduino, "error")
            QMessageBox.warning(self._parent, "Serial test", "Invalid baudrate (set ARDUINO_BAUD in Runtime config).")
            return

        self._set_led(self.lbl_arduino, "starting")
        try:
            ser = serial.Serial(port, baudrate=baud, timeout=1)
            time.sleep(2)
            if ser.is_open:
                self.serial_port_name = port
                self._log("Panel", f"[{self._ts()}] Serial test OK on {port} @ {baud}\n")
                ser.close()
                self._set_led(self.lbl_arduino, "running")
                self._refresh_cmds()
            else:
                self._log("Panel", f"[{self._ts()}] Serial test FAILED (not open)\n")
                self._set_led(self.lbl_arduino, "error")
        except Exception as e:
            self._log("Panel", f"[{self._ts()}] Serial test ERROR: {e}\n")
            self._set_led(self.lbl_arduino, "error")
            QMessageBox.warning(self._parent, "Serial test", f"Error opening {port}:\n{e}")

    def _send_arduino_manual_value(self, value: str):
        port = self.serial_port_name or self.cmb_serial_port.currentData()
        if not port:
            self._log("Panel", f"[{self._ts()}] Arduino send: no port selected\n")
            self._set_led(self.lbl_arduino, "error")
            QMessageBox.information(self._parent, "Arduino manual test", "No serial port selected.")
            return

        baud = self._serial_baud_int()
        if baud is None:
            self._log("Panel", f"[{self._ts()}] Arduino send: invalid baudrate {self.serial_baudrate!r}\n")
            self._set_led(self.lbl_arduino, "error")
            QMessageBox.warning(self._parent, "Arduino manual test", "Invalid baudrate (set ARDUINO_BAUD in Runtime config).")
            return

        self._set_led(self.lbl_arduino, "starting")
        try:
            ser = serial.Serial(port, baudrate=baud, timeout=1)
            self._log("Panel", f"[{self._ts()}] Waiting for Arduino reset (2s)...\n")
            QApplication.processEvents()
            time.sleep(2)

            if not ser.is_open:
                self._log("Panel", f"[{self._ts()}] Arduino manual: failed to open {port}\n")
                self._set_led(self.lbl_arduino, "error")
                return

            ser.write(value.encode("ascii"))
            ser.flush()
            self._log("Panel", f"[{self._ts()}] Arduino manual: sent '{value}' on {port}\n")
            self._set_led(self.lbl_arduino, "running")
            ser.close()

        except Exception as e:
            self._log("Panel", f"[{self._ts()}] Arduino manual ERROR: {e}\n")
            self._set_led(self.lbl_arduino, "error")
            QMessageBox.warning(self._parent, "Arduino manual test", f"Error sending '{value}' on {port}:\n{e}")

    def on_send_arduino_one(self):
        self._send_arduino_manual_value("1")

    def on_send_arduino_zero(self):
        self._send_arduino_manual_value("0")
