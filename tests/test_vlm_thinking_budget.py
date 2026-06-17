"""
test_vlm_thinking_budget.py — guard WS4 F4's keep-working invariant.

F4 added the `--vlm-thinking-budget` flag + `VLM_THINKING_BUDGET` config key to
vlm_service.py and passes the value through to IntentReasoner. The non-negotiable
constraint is that the default preserves today's behaviour: when neither the flag
nor the config key is set, the reasoner must receive `thinking_budget=None`, which
IntentReasoner reads as "emit no ThinkingConfig" (i.e. Gemini's own default
budget). A supplied value (including 0, which disables thinking for low latency)
must be coerced to int and pass through unchanged.

Hardware-free: only exercises argparse and IntentReasoner.__init__'s pure-Python
budget bookkeeping (no Gemini client, no network).
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


def test_default_preserves_current_behaviour():
    # No flag → None sentinel → IntentReasoner emits no ThinkingConfig.
    assert _parse([]).vlm_thinking_budget is None


def test_zero_disables_thinking():
    assert _parse(["--vlm-thinking-budget", "0"]).vlm_thinking_budget == 0


def test_explicit_budget_passes_through_as_int():
    a = _parse(["--vlm-thinking-budget", "2048"])
    assert a.vlm_thinking_budget == 2048
    assert isinstance(a.vlm_thinking_budget, int)
