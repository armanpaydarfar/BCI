# Tiagobot Arduino Sketch

`Final_code.ino` is the firmware for the Tiagobot mobile-arm device (servo +
linear actuator). Flash from the Arduino IDE (`/usr/local/bin/arduino` on this
machine; sketchbook at `~/Arduino`).

## Wire protocol

Serial 9600 baud. The BCI driver writes one of two newline-terminated commands:

- `"{analog_value},{angle},{delay}\n"` — GO. Drives the actuator + servo
  simultaneously to the target. Leaves the actuator at the target until the
  next command. `delay` is the per-degree servo step time (servo rotation
  speed), not a wait.
- `"h\n"` — HOME. Centers the servo and retracts the actuator to its lower
  limit.

No ACK is emitted; the sketch prints human-readable status to serial but the
Python driver does not parse it.

## Differences vs. the original sketch (2026-01-07)

The original at `/home/millanslab/Downloads/Final_code.ino` auto-retracted at
the end of every GO (a 7-second hold then a retract-to-home loop). To match
Harmony's GO-then-later-HOME pattern in concept, this version:

- Drops the post-target hold + auto-retract from the GO path.
- Adds a discrete `h\n` HOME command that performs the retract + center.

All hardware-facing routines (`driveServo`, `driveActuator`, `moveToLimit`,
`calibrateServo`, `calibrateLinAct`) are untouched.

## Flashing

1. Open `Final_code.ino` in the Arduino IDE.
2. Select board (Arduino Uno or equivalent — match the Tiagobot hardware).
3. Select the correct serial port. With both Tiagobot and the glove plugged
   in, identify the Tiagobot Arduino via `ls -l /dev/serial/by-id/`.
4. Upload.
5. Open the Serial Monitor at 9600 baud and confirm the `Calibration complete.`
   banner appears after roughly 10–15 seconds of calibration motion.
