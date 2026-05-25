# tests/tiagobot/

Tiagobot-specific tests for the gaze integration. Canonical feature
reference: `Documents/SoftwareDocs/projects/tiagobot/gaze-integration/reference.md`.
Test-suite plan: `Documents/SoftwareDocs/projects/tiagobot/test-suite/plan.md`.

## Running

```bash
# Run only the tiagobot subdirectory
pytest tests/tiagobot/ -q

# Run the full suite (Harmony base + tiagobot)
pytest tests/ -m "not slow" -q
```

All tests are hardware-free — no Mega 2560, no Neon, no LSL streams
needed.

## File map

| File | Purpose | Tier |
|------|---------|------|
| `test_region_classifier.py` | `Utils.tiagobot_gaze.classify_gaze_to_letter` boundary cases, missing-letter calibration, out-of-range gaze. | Plan §3.1 |
| `test_calibration_roundtrip.py` | Write fixture NPZ -> `Utils.tiagobot_gaze.load_centroids` -> classify -> bit-identical result. Loud failure on missing / corrupt / wrong-shape NPZ. | Plan §3.2 |
| `test_tiago_driver_letter_dispatch.py` | AST-only contract checks on `ExperimentDriver_Online_Tiagobot*.py`. Verifies the gaze driver replaces `random.choice` with classification and preserves the marker stream + glove writes. | Plan §3.3 |
| `test_arduino_serial_contract.py` | Pins the byte sequence per letter A-I against `fixtures/letter_<X>_bytes.txt`. `send_home` writes `b"h\n"`. `wait_for_completion` returns True on the marker line, False on timeout. | Plan §3.4 |
| `test_tiago_serial_layer.py` | `find_tiagobot_port` / `find_glove_port` USB-ID disambiguation, SIMULATION_MODE bypass, unknown-letter rejection. | Plan §3.5 (optional) |

## Fixtures

`fixtures/letter_A_bytes.txt` through `letter_I_bytes.txt` contain the
exact bytes that `Utils.tiagobot.send_letter(mock_ser, letter, logger)`
must write to the serial port. They are generated from
`Utils/tiagobot.py:LOCATIONS` via the documented
`"{analog},{angle},{delay}\n"` format string.

### Regenerating fixtures

If the Arduino sketch (`tools/tiago_arduino/Final_code/Final_code.ino`)
changes its parser, the LOCATIONS dict in `Utils/tiagobot.py` must
match, and the fixtures must be regenerated:

```bash
python -c "
import os
from Utils.tiagobot import LOCATIONS
out_dir = 'tests/tiagobot/fixtures'
os.makedirs(out_dir, exist_ok=True)
for letter, (analog, angle, delay) in LOCATIONS.items():
    with open(os.path.join(out_dir, f'letter_{letter}_bytes.txt'), 'w') as f:
        f.write(f'{analog},{angle},{delay}\n')
"
```

Commit the regenerated fixtures in the SAME commit as the sketch and
LOCATIONS change. The fixture is the wire-format contract — no
fixture-only update is allowed.

## Plan citations

Per `Documents/SoftwareDocs/projects/tiagobot/test-suite/plan.md` §3,
each test docstring cites the file:line for the bug class it guards.
See the plan for the full list and verification protocol.
