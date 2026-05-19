"""
Shared pytest setup for the Harmony test suite.

Responsibilities:
1. Add the repo root and the vendored Rehamove library to `sys.path` so test
   files can `import Utils.foo` and `from rehamove import *` without each
   test file repeating the path dance.
2. Force pygame into headless mode by setting `SDL_VIDEODRIVER=dummy` *before*
   any pygame import happens. Several modules call `pygame.display.set_mode`
   at import time; without the dummy driver they fail on headless CI and
   spawn a real window on a desktop session.
3. Force `config.SIMULATION_MODE = True` early so any module that snapshots
   the flag at import (e.g. `Utils.networking`, see file:line 64-67) reads
   the simulation value rather than the live default.

Per `Harmony_Test_Suite_Plan.md` §4: `Utils.networking.SIMULATION_MODE` is
snapshotted at import — tests that need it must monkeypatch the *module*
attribute, not `config.SIMULATION_MODE`. Tests can use the
`sim_mode_networking` fixture below to do this safely.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Headless pygame must be set BEFORE any test imports a pygame-using module.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_REHAMOVE = ROOT / "STM_interface" / "1_packages" / "rehamoveLibrary"
if _REHAMOVE.is_dir() and str(_REHAMOVE) not in sys.path:
    sys.path.insert(0, str(_REHAMOVE))

# Switch on simulation mode at the config level before any test triggers an
# import of a module that snapshots it. Modules that already cached the value
# need their own monkeypatch via `sim_mode_networking` fixture below.
import config as _config  # noqa: E402

_config.SIMULATION_MODE = True


import pytest  # noqa: E402


@pytest.fixture
def sim_mode_networking(monkeypatch):
    """Set `Utils.networking.SIMULATION_MODE = True` for the duration of a
    single test. The flag is snapshotted at import time (see
    `Utils/networking.py:64-67`) so any test that reaches networking helpers
    must flip the module attribute explicitly."""
    import Utils.networking as _net
    monkeypatch.setattr(_net, "SIMULATION_MODE", True)
    return _net
