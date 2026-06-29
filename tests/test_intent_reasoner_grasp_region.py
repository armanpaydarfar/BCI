"""
test_intent_reasoner_grasp_region.py — WS3 first-pass semantic grasp region.

Pins the contract that the intent reasoner ALSO emits an optional `grasp_region`
for the chosen object, additively and backward-compatibly:

  * a well-formed grasp_region survives parsing intact,
  * an absent grasp_region degrades to None (existing prompts/responses are
    unaffected — no consumer reads the field),
  * a malformed grasp_region degrades to None rather than raising.

Both the module-level coercion helper and the full `_call_api` parse path are
exercised. Hardware-free: no Gemini/OpenAI client, no network — the backend call
is stubbed and logs are redirected to a tmp dir.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import perception.intent_reasoner as ir  # noqa: E402
from perception.intent_reasoner import IntentReasoner, _coerce_grasp_region  # noqa: E402


# ── module-level coercion helper ───────────────────────────────────────────────

def test_coerce_wellformed_grasp_region():
    out = _coerce_grasp_region(
        {"label": "mug", "region": "handle", "grasp_pixel": [120, 88]}
    )
    assert out == {"label": "mug", "region": "handle", "grasp_pixel": [120.0, 88.0]}


def test_coerce_absent_is_none():
    assert _coerce_grasp_region(None) is None


def test_coerce_non_dict_is_none():
    # The VLM occasionally returns a bare string / list for an optional field.
    assert _coerce_grasp_region("handle") is None
    assert _coerce_grasp_region([1, 2]) is None


def test_coerce_malformed_pixel_drops_pixel_keeps_rest():
    out = _coerce_grasp_region(
        {"label": "mug", "region": "handle", "grasp_pixel": ["x", "y"]}
    )
    assert out == {"label": "mug", "region": "handle", "grasp_pixel": None}


def test_coerce_pixel_wrong_arity_drops_pixel():
    out = _coerce_grasp_region({"region": "rim", "grasp_pixel": [1, 2, 3]})
    assert out == {"label": None, "region": "rim", "grasp_pixel": None}


def test_coerce_empty_dict_is_none():
    assert _coerce_grasp_region({}) is None


# ── full parse path through _call_api ──────────────────────────────────────────

def _reasoner_with_stub_response(raw: str, tmp_path: Path) -> IntentReasoner:
    """Build an IntentReasoner WITHOUT __init__ (no API client) whose backend
    returns ``raw``, and whose prompt logs land in ``tmp_path``."""
    r = object.__new__(IntentReasoner)
    r.model = "stub-model"
    r._backend = "openai"
    r._session_dir = tmp_path
    r._session_vlm_log = None
    r._vlm_call_count = 0
    r._call_openai = lambda messages, system_prompt: raw  # type: ignore[method-assign]
    return r


def _decide(raw: str, tmp_path: Path) -> dict:
    r = _reasoner_with_stub_response(raw, tmp_path)
    return r._call_api([{"role": "user", "content": "hi"}])


def test_call_api_parses_with_grasp_region(tmp_path):
    raw = json.dumps(
        {
            "object": "mug",
            "candidates": [{"intent": "drink", "confidence": 1.0}],
            "grasp_region": {"label": "mug", "region": "handle", "grasp_pixel": [10, 20]},
        }
    )
    out = _decide(raw, tmp_path)
    assert out["object"] == "mug"
    assert out["grasp_region"] == {"label": "mug", "region": "handle", "grasp_pixel": [10.0, 20.0]}


def test_call_api_without_grasp_region_yields_none(tmp_path):
    raw = json.dumps(
        {"object": "mug", "candidates": [{"intent": "drink", "confidence": 1.0}]}
    )
    out = _decide(raw, tmp_path)
    # Backward-compatible: existing responses parse unchanged, field is None.
    assert out["object"] == "mug"
    assert out["grasp_region"] is None


def test_call_api_malformed_grasp_region_degrades_to_none(tmp_path):
    raw = json.dumps(
        {
            "object": "mug",
            "candidates": [{"intent": "drink", "confidence": 1.0}],
            "grasp_region": "handle",  # wrong type, must not raise
        }
    )
    out = _decide(raw, tmp_path)
    assert out["grasp_region"] is None


# ── prompt/schema actually asks for it ─────────────────────────────────────────

def test_schema_and_prompts_request_grasp_region():
    assert "grasp_region" in ir.JSON_SCHEMA_DESCRIPTION
    assert "grasp_region" in ir.SYSTEM_PROMPT
    assert "grasp_region" in ir.PAIR_SYSTEM_PROMPT
