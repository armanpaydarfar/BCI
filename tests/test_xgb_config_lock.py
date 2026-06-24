"""
test_xgb_config_lock.py

Guards the "silent XGB hyperparameter regression" class (Plan §6 #9).

These four values define the XGBoost decoder's training and runtime
behaviour. They were tuned in commit fe1814e and should not drift
silently — a bump should be a deliberate, reviewed change that
includes updating this test.

Citations under test (verified 2026-05-18, all in config.py):
  - SHRINKAGE_PARAM_XGB   (line 76)
  - XGB_MAX_DEPTH         (line 112)
  - XGB_LEARNING_RATE     (line 114)
  - XGB_TUNE_CRITERION    (line 124)

The test reads the values via AST so it does not depend on a fully
importable `config_local.py`.
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PY = REPO_ROOT / "config.py"

# Post-fe1814e values. To change one of these intentionally, bump the
# config AND update this dict in the same commit so the bump is
# explicit in code review.
_EXPECTED = {
    "SHRINKAGE_PARAM_XGB": 0.02,
    "XGB_MAX_DEPTH": 6,
    "XGB_LEARNING_RATE": 0.05,
    "XGB_TUNE_CRITERION": "auc",
}


def _config_literal_values():
    tree = ast.parse(CONFIG_PY.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id in _EXPECTED:
                try:
                    out[tgt.id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    out[tgt.id] = "<non-literal>"
    return out


def test_xgb_hyperparameters_match_expected():
    actual = _config_literal_values()
    missing = [k for k in _EXPECTED if k not in actual]
    assert not missing, (
        f"config.py is missing required XGB hyperparameter assignments: "
        f"{missing}"
    )

    mismatches = [(k, actual[k], _EXPECTED[k]) for k in _EXPECTED
                  if actual[k] != _EXPECTED[k]]
    assert not mismatches, (
        "XGB hyperparameters drifted from the post-fe1814e values.\n"
        + "\n".join(f"  {k}: now {a!r}, expected {e!r}" for k, a, e in mismatches)
        + "\n\nIf this change is intentional, update _EXPECTED in "
          "tests/test_xgb_config_lock.py in the same commit."
    )
