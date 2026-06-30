"""
test_no_undefined_names.py — static guard against the bug class that 579 runtime
tests could not catch: a NameError on an un-imported symbol inside a conditional /
hardware-only branch (e.g. ``invert_transform`` in stage_sweep's stabilizer-wobble
diagnostic, 2026-06-30, which crashed AFTER a full sweep but BEFORE the save).

Such names resolve at CALL time, not import time, so ``import`` smokes miss them and
the branch only runs with a live relay+robot. pyflakes flags them statically, across
every code path, without executing anything. This test runs pyflakes over the code
we actually develop and fails on any ``undefined name`` (only that category — unused
imports / f-string nits are intentionally NOT gated here to keep the signal clean).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Directories/globs we actively develop — realtime drivers, calibration tools, gaze
# utils, and the offline analysis scripts. pyflakes parses statically (no import), so
# files with unmet optional deps are fine.
GLOBS = [
    "*.py",                 # realtime drivers + Generate_*/Analyze_*/explore_* at root
    "tools/*.py",
    "Utils/**/*.py",
    "perception/*.py",
]


def _has_pyflakes() -> bool:
    try:
        subprocess.run([sys.executable, "-m", "pyflakes", "--version"],
                       capture_output=True, check=False)
        import pyflakes  # noqa: F401
        return True
    except Exception:
        return False


def test_no_undefined_names():
    if not _has_pyflakes():
        pytest.skip("pyflakes not installed (it is in environment.yml; install to gate)")
    files = sorted({str(p) for g in GLOBS for p in ROOT.glob(g) if p.is_file()})
    assert files, "no python files matched — glob/layout changed"
    res = subprocess.run([sys.executable, "-m", "pyflakes", *files],
                         capture_output=True, text=True)
    # pyflakes prints one finding per line; we gate ONLY on "undefined name" (real
    # NameErrors). Star-import files report "unable to detect undefined names" — a
    # superstring we must exclude so the `import *` pattern (config.py etc.) doesn't
    # trip the gate.
    offenders = [ln for ln in res.stdout.splitlines()
                 if "undefined name" in ln and "unable to detect" not in ln]
    assert not offenders, (
        "pyflakes found undefined name(s) — a latent NameError that would crash at "
        "runtime in some branch:\n  " + "\n  ".join(offenders))
