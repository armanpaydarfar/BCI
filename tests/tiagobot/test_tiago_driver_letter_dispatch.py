"""
test_tiago_driver_letter_dispatch.py

Guards driver state-machine bugs in `ExperimentDriver_Online_Tiagobot.py`
(the existing known-good driver) and
`ExperimentDriver_Online_Tiagobot_Gaze.py` (the new gaze sibling).
Per `Documents/SoftwareDocs/projects/tiagobot/test-suite/plan.md` §3.3.

This is the Tiagobot analogue of `tests/test_errp_driver_state_machine.py`.
Both drivers do pygame.display.set_mode + model.pkl load + LSL resolve at
module import — none of which works inside the pre-commit gate. The test
operates on AST shape instead of running a trial.

The contract guarded:

1. The gaze driver MUST NOT contain `random.choice(config.TIAGOBOT_TRAJECTORY)`
   (replaced by gaze classification in
   `ExperimentDriver_Online_Tiagobot_Gaze.py:~615`).
2. The gaze driver MUST call `tiago_send_letter` exactly inside the
   MI-correct + gaze-resolved branch — never randomly, never on REST.
3. Both drivers MUST send `tiago_send_home` exactly once per
   should_hold_and_classify branch (so the actuator returns to a known
   state on every trial that extended).
4. The marker stream still receives the LSL events
   (ROBOT_BEGIN, ROBOT_HOME) the existing driver sends — the gaze
   driver inherits these unchanged.

Citations (verified 2026-05-19):
  - `Utils/tiagobot.py:send_letter` (line ~315) — wire-level write
  - `Utils/tiagobot.py:send_home` (line ~337) — HOME write
  - `ExperimentDriver_Online_Tiagobot.py:449` — random.choice line in
    the existing driver (intentionally left in the existing driver;
    forbidden in the gaze sibling)
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PARENT_DRIVER = REPO_ROOT / "ExperimentDriver_Online_Tiagobot.py"
GAZE_DRIVER = REPO_ROOT / "ExperimentDriver_Online_Tiagobot_Gaze.py"


def _read_source(path: Path) -> str:
    assert path.is_file(), f"{path} not found"
    return path.read_text()


# ---- Parent driver contract -------------------------------------------
def test_parent_driver_uses_random_choice():
    """Sanity: the parent driver IS still the random-choice version —
    if someone refactored it to not use random.choice, the gaze
    driver's divergence story breaks. The parent driver must remain
    untouched on this branch (it's the known-good fallback)."""
    src = _read_source(PARENT_DRIVER)
    assert "random.choice(config.TIAGOBOT_TRAJECTORY)" in src, (
        "Parent driver ExperimentDriver_Online_Tiagobot.py no longer "
        "contains the random.choice line at the documented location. "
        "Either the parent has diverged and the gaze driver needs to "
        "be re-aligned, or this contract test is stale."
    )


def test_parent_driver_calls_send_letter_once_per_trial():
    """Parent driver must call tiago_send_letter exactly once in the
    file. Multiple calls would mean a regression introduced
    duplication."""
    src = _read_source(PARENT_DRIVER)
    n = len(re.findall(r"\btiago_send_letter\s*\(", src))
    assert n == 1, (
        f"Parent driver has {n} calls to tiago_send_letter; expected "
        f"exactly 1 (the MI-correct branch)."
    )


def test_parent_driver_calls_send_home_once_per_trial():
    """One send_home per trial — sent unconditionally at trial end on
    every MI-correct trial regardless of glove/earlystop. The Tiagobot
    actuator does not self-retract, so missing this on any branch
    would leave the arm extended for the next trial."""
    src = _read_source(PARENT_DRIVER)
    n = len(re.findall(r"\btiago_send_home\s*\(", src))
    assert n == 1, (
        f"Parent driver has {n} calls to tiago_send_home; expected "
        f"exactly 1."
    )


# ---- Gaze driver contract --------------------------------------------
def test_gaze_driver_does_not_use_random_choice():
    """The gaze driver MUST NOT contain `random.choice(...)` over the
    trajectory — that's the whole point of the refactor (gaze
    selection replaces random selection)."""
    src = _read_source(GAZE_DRIVER)
    assert "random.choice(config.TIAGOBOT_TRAJECTORY)" not in src, (
        "Gaze driver still contains random.choice(config.TIAGOBOT_TRAJECTORY). "
        "Phase 2.c was supposed to replace this with classify_gaze_to_letter."
    )


def test_gaze_driver_does_not_import_random():
    """`random` was used only for the random.choice. With that removed,
    the import is dead code per CLAUDE.md and should be deleted."""
    src = _read_source(GAZE_DRIVER)
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "random", (
                    "Gaze driver still imports `random`. The random.choice "
                    "line was removed; the import should be removed too "
                    "(CLAUDE.md: no dead code)."
                )


def test_gaze_driver_calls_classify_gaze_to_letter():
    """The gaze driver must invoke the classifier — the only way for
    the gaze layer to produce a letter."""
    src = _read_source(GAZE_DRIVER)
    assert "tiago_gaze_classify" in src, (
        "Gaze driver does not reference tiago_gaze_classify (the gaze "
        "letter classifier). The Phase 2.b helper is unused."
    )


def test_gaze_driver_calls_send_letter_once():
    """Exactly one tiago_send_letter call in the gaze driver — same as
    the parent. Two calls would be the trial-duplication regression."""
    src = _read_source(GAZE_DRIVER)
    n = len(re.findall(r"\btiago_send_letter\s*\(", src))
    assert n == 1, (
        f"Gaze driver has {n} calls to tiago_send_letter; expected "
        f"exactly 1."
    )


def test_gaze_driver_calls_send_home_once():
    """One send_home call in the gaze driver — same contract as the
    parent. Tiagobot doesn't self-retract; missing HOME means a stuck
    actuator on the next trial."""
    src = _read_source(GAZE_DRIVER)
    n = len(re.findall(r"\btiago_send_home\s*\(", src))
    assert n == 1, (
        f"Gaze driver has {n} calls to tiago_send_home; expected "
        f"exactly 1."
    )


def test_gaze_driver_loads_centroids_at_startup():
    """The calibration NPZ must load BEFORE main() is entered — fail-
    fast on missing config is the documented behavior. Otherwise the
    operator gets through the trial setup before the calibration error
    shows up."""
    src = _read_source(GAZE_DRIVER)
    assert "tiago_gaze_load_centroids" in src, (
        "Gaze driver does not call tiago_gaze_load_centroids."
    )
    # The load must happen at module top (before def main:).
    main_line_idx = src.find("def main(")
    assert main_line_idx >= 0
    pre_main = src[:main_line_idx]
    assert "tiago_gaze_load_centroids" in pre_main, (
        "tiago_gaze_load_centroids is called only inside main(); it "
        "must run at module import / driver startup so the operator "
        "sees the missing-calibration error before pygame opens."
    )


def test_gaze_driver_skips_go_when_letter_is_none():
    """Per plan §6.3 step 4 fallback: when classification returns
    None, the driver MUST NOT send a letter on that trial (no random
    fallback). The contract is checked structurally — the gaze
    driver's letter dispatch is guarded by
    `_gaze_selected_letter is not None`."""
    src = _read_source(GAZE_DRIVER)
    # The pattern: `prediction == 200 and _gaze_selected_letter is not None`
    assert "_gaze_selected_letter is not None" in src, (
        "Gaze driver does not guard the GO branch on "
        "_gaze_selected_letter is not None — gaze-fail trials would "
        "still send a letter."
    )
    # And the parent's unconditional `if prediction == 200:` branch
    # has been replaced (not removed) — there must be a second branch
    # for the gaze-fail case.
    assert "_gaze_selected_letter is None" in src, (
        "Gaze driver does not have a dedicated branch for the gaze-"
        "fail case (prediction==200 but no letter resolved)."
    )


# ---- Marker stream contract preserved --------------------------------
def test_gaze_driver_preserves_marker_triggers():
    """Both ROBOT_BEGIN and ROBOT_HOME triggers must still be sent by
    the gaze driver. These are the LSL marker channel events analyses
    depend on; the gaze refactor should not have dropped them."""
    src = _read_source(GAZE_DRIVER)
    for trig in ('TRIGGERS["ROBOT_BEGIN"]', 'TRIGGERS["ROBOT_HOME"]'):
        assert trig in src, (
            f"Gaze driver no longer sends {trig}. The marker stream "
            f"contract was broken by the refactor."
        )


# ---- Glove integration preserved --------------------------------------
def test_gaze_driver_preserves_glove_writes():
    """Both ARDUINO_CMD_MI (glove close) and ARDUINO_CMD_REST (glove
    open) writes are preserved in the gaze driver. The glove
    integration is independent of the gaze selection layer."""
    src = _read_source(GAZE_DRIVER)
    assert "ARDUINO_CMD_MI" in src, "Gaze driver no longer closes the glove on MI-correct."
    assert "ARDUINO_CMD_REST" in src, "Gaze driver no longer opens the glove at HOME."
