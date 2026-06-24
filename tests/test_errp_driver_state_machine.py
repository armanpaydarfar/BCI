"""
test_errp_driver_state_machine.py

Guards the marker / trigger duplication-or-omission bug class in the
online ErrP driver (Plan §6 #7). Example commits:

  - cbfa120  remove duplicate MI_BEGIN/REST_BEGIN triggers in ErrP
             online driver. The driver was sending the begin triggers
             immediately before calling show_feedback(), which sends
             them itself via runtime_common — every trial had two of
             each marker. After this fix show_feedback is the sole
             owner of MI_BEGIN / REST_BEGIN.
  - abbe928  always send ROBOT_HOME after a no-ErrP hold expiry. The
             home command was previously gated on
             `not move_result["robot_earlystop"]` and the robot got
             stuck mid-trajectory on the earlystop path.

Because `ExperimentDriver_ErrP_Online.py`'s `main()` has heavy
import-time and runtime side effects (pygame display, model loading,
LSL stream resolution), this file does NOT instantiate the driver. It
asserts on the static structure of the driver source instead:

  - Zero `config.TRIGGERS["MI_BEGIN" | "REST_BEGIN"]` references in
    the driver — they live only inside `runtime_common.show_feedback`
    (file:704, 706).
  - Every `config.TRIGGERS["ROBOT_HOME"]` send in the driver is NOT
    nested inside an `if not move_result["robot_earlystop"]:` block,
    so the homing command always fires.

Citations under test (verified 2026-05-18):

  - ExperimentDriver_ErrP_Online.py:582  driver no longer sends
    MI_BEGIN / REST_BEGIN here.
  - ExperimentDriver_ErrP_Online.py:670-684  ROBOT_HOME send sits at
    the same indent as the earlystop guard, not inside it.
  - Utils/runtime_common.py:702-706  show_feedback owns MI_BEGIN
    and REST_BEGIN.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
ERRP_DRIVER = REPO_ROOT / "ExperimentDriver_ErrP_Online.py"
RUNTIME_COMMON = REPO_ROOT / "Utils" / "runtime_common.py"


# ─── helpers ──────────────────────────────────────────────────────────────

def _parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"))


def _is_trigger_ref(node: ast.AST, key: str) -> bool:
    """Match a node literally shaped as `config.TRIGGERS["<key>"]`."""
    if not isinstance(node, ast.Subscript):
        return False
    val = node.value
    if not (isinstance(val, ast.Attribute) and val.attr == "TRIGGERS"
            and isinstance(val.value, ast.Name) and val.value.id == "config"):
        return False
    sl = node.slice
    if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
        return sl.value == key
    return False


def _walk_triggers(tree: ast.AST, key: str):
    """Yield every `config.TRIGGERS["<key>"]` reference in tree."""
    for node in ast.walk(tree):
        if _is_trigger_ref(node, key):
            yield node


def _find_send_udp_calls_with_trigger(tree: ast.AST, key: str):
    """Yield every `send_udp_message(...)` Call whose argument list contains
    a `config.TRIGGERS["<key>"]` reference."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Match both `send_udp_message(...)` and `<module>.send_udp_message(...)`.
        name = (func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute)
                else None)
        if name != "send_udp_message":
            continue
        for arg in node.args:
            if _is_trigger_ref(arg, key):
                yield node
                break


def _set_parents(tree: ast.AST):
    """Annotate every node with a `.parent` attribute for ancestor walking."""
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent  # type: ignore[attr-defined]


def _ancestors(node):
    cur = getattr(node, "parent", None)
    while cur is not None:
        yield cur
        cur = getattr(cur, "parent", None)


def _is_earlystop_guarded(node: ast.AST) -> bool:
    """True if `node` lives inside the body of an `if not move_result["robot_earlystop"]:`
    (or equivalently-shaped) conditional, including via the `else` branch."""
    for anc in _ancestors(node):
        if not isinstance(anc, ast.If):
            continue
        test = anc.test
        # Match `not move_result["robot_earlystop"]`.
        if (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)):
            inner = test.operand
            if (isinstance(inner, ast.Subscript)
                    and isinstance(inner.value, ast.Name)
                    and inner.value.id == "move_result"
                    and isinstance(inner.slice, ast.Constant)
                    and inner.slice.value == "robot_earlystop"):
                # Only the body, not the else (else means earlystop=True path).
                # Walk back from `node` up to `anc` and check which branch
                # we descended through.
                cur = node
                while cur is not None and cur is not anc:
                    parent = getattr(cur, "parent", None)
                    if parent is anc and cur in anc.body:
                        return True
                    cur = parent
    return False


# ─── tests ────────────────────────────────────────────────────────────────

class TestNoDuplicateBeginTriggers:
    def test_driver_does_not_reference_MI_BEGIN(self):
        """Post-cbfa120: the ErrP online driver must not emit MI_BEGIN —
        runtime_common.show_feedback is the sole owner."""
        tree = _parse(ERRP_DRIVER)
        offenders = list(_walk_triggers(tree, "MI_BEGIN"))
        assert not offenders, (
            f"ExperimentDriver_ErrP_Online.py still references "
            f"config.TRIGGERS['MI_BEGIN'] at line(s) "
            f"{[n.lineno for n in offenders]} — duplicates the marker that "
            f"show_feedback sends (Utils/runtime_common.py:704)."
        )

    def test_driver_does_not_reference_REST_BEGIN(self):
        """Same for REST_BEGIN."""
        tree = _parse(ERRP_DRIVER)
        offenders = list(_walk_triggers(tree, "REST_BEGIN"))
        assert not offenders, (
            f"ExperimentDriver_ErrP_Online.py still references "
            f"config.TRIGGERS['REST_BEGIN'] at line(s) "
            f"{[n.lineno for n in offenders]} — duplicates the marker that "
            f"show_feedback sends (Utils/runtime_common.py:706)."
        )

    def test_show_feedback_sends_MI_BEGIN_exactly_once(self):
        """The sole-owner contract: `show_feedback` in runtime_common
        must contain exactly one `send_udp_message(...MI_BEGIN...)` call
        and one for REST_BEGIN."""
        tree = _parse(RUNTIME_COMMON)
        # Find the FunctionDef node for show_feedback.
        sf = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "show_feedback"),
                  None)
        assert sf is not None, "show_feedback not found in Utils/runtime_common.py"

        mi_sends = list(_find_send_udp_calls_with_trigger(sf, "MI_BEGIN"))
        rest_sends = list(_find_send_udp_calls_with_trigger(sf, "REST_BEGIN"))
        assert len(mi_sends) == 1, (
            f"show_feedback should send MI_BEGIN exactly once; "
            f"found {len(mi_sends)}"
        )
        assert len(rest_sends) == 1, (
            f"show_feedback should send REST_BEGIN exactly once; "
            f"found {len(rest_sends)}"
        )


class TestRobotHomeAlwaysSent:
    def test_robot_home_send_is_not_earlystop_guarded(self):
        """Post-abbe928: every `send_udp_message(...ROBOT_HOME...)` call in
        the ErrP online driver must live OUTSIDE the
        `if not move_result["robot_earlystop"]:` block, so the robot is
        always commanded home (even when the trial ended on earlystop)."""
        tree = _parse(ERRP_DRIVER)
        _set_parents(tree)
        sends = list(_find_send_udp_calls_with_trigger(tree, "ROBOT_HOME"))
        assert sends, (
            "Driver no longer sends ROBOT_HOME at all — the post-abbe928 "
            "contract requires at least one send."
        )
        guarded = [s for s in sends if _is_earlystop_guarded(s)]
        assert not guarded, (
            f"ROBOT_HOME send at line(s) {[s.lineno for s in guarded]} is "
            f"nested inside `if not move_result['robot_earlystop']:`. "
            f"That gating was the abbe928 bug — earlystop trials left the "
            f"robot stuck mid-trajectory."
        )
