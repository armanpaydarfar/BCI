"""
test_env_drift_slow.py

P2 thin mirror of the doctor's env-vs-spec drift check, for a FUTURE CI runner
(per the proposal + control-host response: env-drift lives in tools/preflight.py;
this is only a slow-marked echo so a clean-built CI env stays honest). It is
@pytest.mark.slow on purpose — EXCLUDED from the fast pre-commit gate — because it
asserts the *live* env matches environment.yml's pinned numerical core, which is a
property of the installed env, not of the code.

Expected to FAIL on a host whose env drifted off the pins (e.g. numpy upgraded
past 1.26.4) — that failure is the signal, which is why it is not in the fast gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from preflight import Context, check_env_drift, FAIL  # noqa: E402


@pytest.mark.slow
def test_pinned_core_matches_environment_yml():
    ctx = Context(role="server", strict=False)
    fails = [r for r in check_env_drift(ctx) if r.status == FAIL]
    assert not fails, (
        "env drifted off environment.yml pins:\n  "
        + "\n  ".join(f"{r.name}: {r.detail}" for r in fails)
    )
