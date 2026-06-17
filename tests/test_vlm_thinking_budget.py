"""
test_vlm_thinking_budget.py — guard WS4 F4's --vlm-thinking-budget plumbing.

F4 added the `--vlm-thinking-budget` flag + `VLM_THINKING_BUDGET` config key to
vlm_service.py and passes the value through to IntentReasoner. After the
on-hardware latency benchmark (2026-06-17), the committed default was flipped to
**0** (thinking disabled — ~2.9× faster decide). The contract these tests pin:
the default resolves to the committed config value (0), an explicit `0` is
honoured, and a positive int is coerced and passes through unchanged so thinking
can still be re-enabled per-run / per-machine.

Hardware-free: only exercises argparse (no Gemini client, no network).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import vlm_service  # noqa: E402


def _parse(argv):
    saved = sys.argv
    try:
        sys.argv = ["vlm_service.py", *argv]
        return vlm_service.parse_args()
    finally:
        sys.argv = saved


def test_default_is_committed_config_value():
    # No flag → the committed config default (VLM_THINKING_BUDGET = 0, the
    # benchmark-chosen fast path). _cfg_default reads it from config.py.
    import config
    assert _parse([]).vlm_thinking_budget == config.VLM_THINKING_BUDGET == 0


def test_zero_disables_thinking():
    assert _parse(["--vlm-thinking-budget", "0"]).vlm_thinking_budget == 0


def test_explicit_budget_passes_through_as_int():
    # A positive int re-enables (caps) thinking — the escape hatch from the
    # disabled-by-default fast path. type=int coerces the supplied value.
    a = _parse(["--vlm-thinking-budget", "2048"])
    assert a.vlm_thinking_budget == 2048
    assert isinstance(a.vlm_thinking_budget, int)
