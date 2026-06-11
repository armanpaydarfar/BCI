# Harmony test suite

Hardware-free pytest suite that runs on the Linux primary BCI machine
(and on Windows when the lsl conda env is available). No Neon, no
eegoSports, no Rehamove, no robot, no network peers.

Originating plan: `Documents/SoftwareDocs/Harmony_Test_Suite_Plan.md`.

## Running

The default fast subset (excludes anything marked `@pytest.mark.slow`):

```bash
pytest tests/ -m "not slow" -q
```

The full suite (includes loopback / network-adjacent tests in
`test_relay_loopback.py` etc.):

```bash
pytest tests/ -q
```

A single file:

```bash
pytest tests/test_networking_protocol.py -v
```

## Pre-commit gate

The `.githooks/pre-commit` hook (enabled per clone with
`git config core.hooksPath .githooks`, see `.githooks/README.md`)
runs the fast subset before every commit and aborts the commit on
failure. The pytest invocation lives in `tools/pre-commit-pytest.sh`
so it can be run standalone too.

To bypass the gate in an emergency (NEVER on `main` / `master`):

```bash
git commit --no-verify
```

Acceptable bypass reasons (rare): an in-progress feature branch where
the failing test is exactly what the next commit is about to fix.
Unacceptable: pushing onto a shared branch with red tests.

## Adding a new test

Per the plan's §3 verification protocol — *before* writing any test
against a citation in the plan or in this directory:

1. Read the cited file:line range with the `Read` tool (or open it).
   Confirm the function name, signature, and behavior are as described.
2. If a citation no longer matches, add a dated note to §10
   ("Plan drift log") of `Harmony_Test_Suite_Plan.md` before writing
   the test. Don't write a test against stale facts.
3. Cite file:line in the docstring of every test function so the next
   reviewer can audit.
4. New file goes under `tests/` with a name beginning `test_`. Pytest
   discovers it automatically (see `pytest.ini`).

If your test is slow (network-adjacent loopback, large fixture load,
> ~1 s) mark the slow tests with `@pytest.mark.slow` so the
pre-commit fast subset stays under ~10 seconds.

## Fixtures

`tests/conftest.py` provides:

- Headless pygame: sets `SDL_VIDEODRIVER=dummy` before any pygame
  import. Required because several modules call `pygame.display.set_mode`
  at import time.
- `sys.path` includes the repo root and the vendored Rehamove
  library, so `import Utils.foo` and `from rehamove import *` work
  without per-file path dancing.
- `config.SIMULATION_MODE = True` is set at session start so any
  module that snapshots it at import reads the simulation value.
- The `sim_mode_networking` pytest fixture monkeypatches the
  `Utils.networking.SIMULATION_MODE` snapshot (which is taken at
  import time per `Utils/networking.py:64-67`) for the duration of
  a single test.

## Layout (Phase 1a)

| File | Guards | Plan |
|---|---|---|
| `test_imports_smoke.py` | Startup SyntaxError / wrong-import-API class | §5.2 #1 |
| `test_networking_protocol.py` | Wire-format / ACK base-token matching | §5.2 #2 |
| `test_config_contract.py` | Config drift / wrong key name | §5.2 #3 |
| `test_gaze_tracking_pure.py` | Gaze geometry regressions (preventive) | §5.2 #4 |
| `test_relay_loopback.py` | Frame-relay envelope (promoted) | §5.1.c |
| `test_vlm_results_push.py` | VLM result push (promoted) | §5.1.c |
| `test_gaze_results_push.py` | Gaze result push (promoted) | §5.1.c |
| `test_relay_local_subscriber.py` | Local subscriber (promoted) | §5.1.c |

Phase 1b additions land per `Harmony_Test_Suite_Plan.md` §6.

## Cross-platform parity

Cross-platform (Linux ↔ Windows) decoder equivalence — Plan §6.1 —
is **deferred** until a Windows CI runner exists. Both machines run
the same suite manually for now.
