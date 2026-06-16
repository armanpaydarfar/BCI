"""
test_config_contract.py

Guards the "config drift / wrong key name" bug class. Example commits this
file would have caught (or moved detection to pre-commit):

  - abbe928  fix: control panel ERRP_P_STOP → ERRP_ONLINE_P_STOP. The panel
             read/wrote a key that does not exist in config.py, silently
             returning the getattr default and the slider's UI value never
             reached the driver.
  - 0334fe3  fix: stale VISUALIZE_MAX_ABS_UV reference (renamed to
             EPOCH_MAX_ABS_UV). Wrong-key class — runtime AttributeError.
  - 5b04a29  fix: rbnnet dispatch with wrong filename. Same class — code
             referenced a config-derived name that didn't match disk.

Citations under test (verified 2026-05-18):

  - config.py:295 (TRIGGERS dict) / config.py:325 (ROBOT_OPCODES dict)
  - control_panel.py:46  `from config import ARDUINO_PORT`
  - control_panel.py:316 `_HCFG.UDP_MARKER["PORT"]`
  - ExperimentDriver_*.py + Utils/runtime_common.py — `config.TRIGGERS["..."]`
    and `config.ROBOT_OPCODES["..."]` references.

Machine-local key list duplicated from `~/.claude/hooks/config-py-guard.sh`
(the pre-commit hook). If that hook adds/removes a key, update _LOCAL_KEYS
here to match. Safe-default literals come from the same source.

The whole file is AST-only — `config.py` is NEVER imported here, because
importing it has side effects (machine-local override pulls in
`config_local`, which is gitignored and not always present on every
machine).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PY = REPO_ROOT / "config.py"
CONTROL_PANEL_PY = REPO_ROOT / "control_panel.py"
DRIVER_FILES = sorted(REPO_ROOT.glob("ExperimentDriver_*.py")) + [
    REPO_ROOT / "Utils" / "runtime_common.py",
]

# Mirror of ~/.claude/hooks/config-py-guard.sh LOCAL_KEYS.
_LOCAL_KEYS = {
    "WORKING_DIR", "DATA_DIR",
    "GAZE_UDP_IP", "GAZE_BIND_HOST",
    "NEON_COMPANION_HOST",
    "PERCEPTION_FRAME_SOURCE", "SERVICES_HOSTED_REMOTELY",
    "FRAME_RELAY_HOST", "FRAME_RELAY_DIAL_HOST",
    "PERCEPTION_MODELS_DIR", "GOOGLE_API_KEY",
    "VLM_SERVICE_HOST", "VLM_BIND_HOST",
    "ARDUINO_PORT",
}

# Mirror of ~/.claude/hooks/config-py-guard.sh SAFE_RHS, expressed as
# AST-evaluable literal values (the hook compares string-equality of the
# source slice; here we compare evaluated values).
_SAFE_DEFAULT_VALUES = {
    "",            # '""', "''"
    "127.0.0.1",
    "0.0.0.0",
    "local",
    False,
}


# ─── parse config.py once ──────────────────────────────────────────────────

def _parse_config():
    """Return (config_keys, triggers_keys, robot_opcodes_keys, assignments)."""
    src = CONFIG_PY.read_text()
    tree = ast.parse(src)

    config_keys = set()
    assignments = {}  # name -> ast.expr (value node)
    triggers_keys = set()
    robot_opcodes_keys = set()

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    config_keys.add(tgt.id)
                    assignments[tgt.id] = node.value

    # Pull TRIGGERS dict keys and ROBOT_OPCODES dict keys from their literal
    # dict assignments. Both are documented as plain dict literals at module
    # top (config.py:295 and :325).
    for name in ("TRIGGERS", "ROBOT_OPCODES"):
        node = assignments.get(name)
        if not isinstance(node, ast.Dict):
            continue
        target_set = triggers_keys if name == "TRIGGERS" else robot_opcodes_keys
        for k in node.keys:
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                target_set.add(k.value)

    return config_keys, triggers_keys, robot_opcodes_keys, assignments


@pytest.fixture(scope="module")
def config_meta():
    return _parse_config()


# ─── tests ────────────────────────────────────────────────────────────────

def test_machine_local_keys_have_safe_defaults(config_meta):
    """For each machine-local key per CLAUDE.md + config-py-guard.sh, the
    top-level assignment in config.py must be a safe literal default. This
    is the static analogue of the pre-commit hook — anything that would
    trip the hook also trips this test (which runs in pytest, so CI catches
    it too)."""
    _, _, _, assignments = config_meta
    bad = []
    for key in sorted(_LOCAL_KEYS):
        if key not in assignments:
            bad.append(f"{key}: missing top-level assignment in config.py")
            continue
        value_node = assignments[key]
        try:
            value = ast.literal_eval(value_node)
        except (ValueError, SyntaxError):
            bad.append(f"{key}: assigned a non-literal expression in config.py")
            continue
        if value not in _SAFE_DEFAULT_VALUES:
            bad.append(f"{key}: assigned {value!r}, expected one of "
                       f"{sorted(_SAFE_DEFAULT_VALUES, key=str)}")
    assert not bad, (
        "Machine-local keys assigned non-default values in config.py "
        "(these belong in config_local.py):\n  " + "\n  ".join(bad)
    )


# ─── control_panel key references ─────────────────────────────────────────

# Regex-match the panel's bespoke key accessors. The accessors are private
# functions defined inside control_panel.py; rather than execute the panel
# (which imports PySide6 and instantiates QWidgets), we lift the key names
# from the source.
_PANEL_KEY_PATTERNS = [
    re.compile(r'_read_float_key\(\s*"([A-Z_][A-Z0-9_]*)"'),
    re.compile(r'_read_int_key\(\s*"([A-Z_][A-Z0-9_]*)"'),
    re.compile(r'_read_str_key\(\s*"([A-Z_][A-Z0-9_]*)"'),
    re.compile(r'_read_bool_key\(\s*"([A-Z_][A-Z0-9_]*)"'),
    re.compile(r'_write_assign_rhs\(\s*"([A-Z_][A-Z0-9_]*)"'),
    re.compile(r'_write_assign_rhs_local\(\s*"([A-Z_][A-Z0-9_]*)"'),
    re.compile(r'getattr\(\s*_HCFG\s*,\s*"([A-Z_][A-Z0-9_]*)"'),
]


def _panel_key_references():
    src = CONTROL_PANEL_PY.read_text()
    refs = set()
    for pat in _PANEL_KEY_PATTERNS:
        refs.update(pat.findall(src))
    return refs


def test_control_panel_keys_exist_in_config(config_meta):
    """Every config key the panel reads or writes must exist in config.py.
    Otherwise getattr returns the (likely wrong) UI default and writes
    create a brand-new key that no driver ever reads — exactly the
    `abbe928` ERRP_P_STOP vs ERRP_ONLINE_P_STOP bug."""
    config_keys, _, _, _ = config_meta
    referenced = _panel_key_references()
    missing = sorted(referenced - config_keys)
    assert not missing, (
        f"control_panel.py references config keys not defined in config.py: "
        f"{missing}. Either add the key to config.py with a safe default, or "
        f"fix the panel reference."
    )


def test_control_panel_hard_imports_exist(config_meta):
    """The panel's hard `from config import KEY` reads must succeed. Today
    that's ARDUINO_PORT (control_panel.py:46) and UDP_MARKER
    (control_panel.py:316)."""
    config_keys, _, _, _ = config_meta
    required = {"ARDUINO_PORT", "UDP_MARKER"}
    missing = sorted(required - config_keys)
    assert not missing, (
        f"control_panel.py's hard config reads need these keys defined "
        f"in config.py: {missing}."
    )


# ─── driver-level TRIGGERS / ROBOT_OPCODES references ─────────────────────

_TRIGGERS_REF = re.compile(r'config\.TRIGGERS\[\s*"([A-Z_][A-Z0-9_]*)"\s*\]')
_OPCODES_REF = re.compile(r'config\.ROBOT_OPCODES\[\s*"([A-Z_][A-Z0-9_]*)"\s*\]')


def _collect_keyed_references(pattern):
    refs = set()
    for path in DRIVER_FILES:
        if not path.is_file():
            continue
        refs.update(pattern.findall(path.read_text()))
    return refs


def test_driver_trigger_references_exist_in_config(config_meta):
    """Every `config.TRIGGERS["..."]` reference in ExperimentDriver_*.py and
    Utils/runtime_common.py must be a key of the TRIGGERS dict literal in
    config.py. Catches the rename-without-update class."""
    _, triggers_keys, _, _ = config_meta
    referenced = _collect_keyed_references(_TRIGGERS_REF)
    missing = sorted(referenced - triggers_keys)
    assert not missing, (
        f"Drivers reference TRIGGERS keys not present in config.py "
        f"TRIGGERS dict: {missing}"
    )


def test_driver_robot_opcode_references_exist_in_config(config_meta):
    """Every `config.ROBOT_OPCODES["..."]` reference in driver files must
    be a key of the ROBOT_OPCODES dict literal in config.py."""
    _, _, robot_opcodes_keys, _ = config_meta
    referenced = _collect_keyed_references(_OPCODES_REF)
    missing = sorted(referenced - robot_opcodes_keys)
    assert not missing, (
        f"Drivers reference ROBOT_OPCODES keys not present in config.py "
        f"ROBOT_OPCODES dict: {missing}"
    )


def test_local_keys_list_matches_hook():
    """Self-consistency: the _LOCAL_KEYS list above must stay in lockstep
    with the pre-commit hook's LOCAL_KEYS set, which is the source of truth
    referenced by CLAUDE.md. If the hook changes, this test reminds us to
    update the duplicate here.

    The hook path may not exist on every machine (CI, etc.); the test is
    a skip in that case so test runs don't depend on user-config presence.
    """
    hook_path = Path.home() / ".claude" / "hooks" / "config-py-guard.sh"
    if not hook_path.is_file():
        pytest.skip(f"{hook_path} not present on this machine")
    src = hook_path.read_text()
    # The hook embeds its key list as `LOCAL_KEYS = { ... }` inside a python
    # heredoc. Parse it with a forgiving regex rather than re-running bash.
    hook_keys = set(re.findall(r'"([A-Z_][A-Z0-9_]*)"', src))
    # The hook also references SAFE_RHS values which match the pattern, so
    # filter to known machine-local-shape names.
    hook_keys = {k for k in hook_keys if k in _LOCAL_KEYS or k.endswith("_DIR")
                 or k.endswith("_HOST") or k.endswith("_IP") or k.endswith("_PORT")
                 or k in {"PERCEPTION_FRAME_SOURCE", "SERVICES_HOSTED_REMOTELY",
                          "ARDUINO_PORT"}}
    missing_in_hook = _LOCAL_KEYS - hook_keys
    extra_in_hook = hook_keys - _LOCAL_KEYS
    assert not missing_in_hook and not extra_in_hook, (
        f"_LOCAL_KEYS in this test must match the hook. "
        f"Only in this test: {sorted(missing_in_hook)}. "
        f"Only in hook: {sorted(extra_in_hook)}."
    )
