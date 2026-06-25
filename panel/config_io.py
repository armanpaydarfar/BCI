"""
panel/config_io.py — config.py / config_local.py read-write layer.

Behaviour-preserving extraction of the module-level config-file helpers that
used to live at the top of control_panel.py. Pure functions over the two config
files (regex read + atomic write); no Qt, no panel state. Lives in a leaf module
so panel collaborators and the panel itself import these by name instead of the
former bare module-level references — and so the machinery is unit-testable
without standing up the panel.

The guarded machine-local key list (LOCAL_CONFIG_KEYS) is mirrored in
~/.claude/hooks/config-py-guard.sh — keep both in sync.
"""

from __future__ import annotations

import os
import re
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PY = os.path.join(ROOT, "config.py")
CONFIG_LOCAL_PY = os.path.join(ROOT, "config_local.py")

SUBJECT_RE = re.compile(r'^(TRAINING_SUBJECT\s*=\s*)([\'"])([^\'"]+)\2\s*$', re.M)
FES_RE     = re.compile(r'^(FES_toggle\s*=\s*)([01])\s*$', re.M)
SIM_RE     = re.compile(r'^(SIMULATION_MODE\s*=\s*)(True|False)(\s*(#.*)?)\s*$', re.M)


def read_text(path: str) -> str:
    """
    Read a UTF-8 text file.

    Returns an empty string if the file does not exist.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def write_atomic(path: str, text: str):
    """
    Write text to `path` using an atomic replace.

    This reduces the chance of producing a partially-written `config.py`
    if the application crashes mid-write.
    """
    tmp = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
    try:
        tmp.write(text)
        tmp.flush(); os.fsync(tmp.fileno()); tmp.close()
        os.replace(tmp.name, path)
    except Exception:
        try: os.unlink(tmp.name)
        except Exception: pass
        raise


def read_simulation_mode(default=False) -> bool:
    """
    Read `SIMULATION_MODE` from `config.py` using a regex match.
    """
    txt = read_text(CONFIG_PY)
    m = SIM_RE.search(txt)
    if not m:
        return bool(default)
    return (m.group(2) == "True")


def write_simulation_mode(val: bool):
    """
    Update `SIMULATION_MODE = True/False` inside `config.py` in-place.

    Implementation detail: uses regex substitution so we don't need to import
    or rewrite unrelated config lines.
    """
    val_txt = "True" if val else "False"
    txt = read_text(CONFIG_PY)
    if SIM_RE.search(txt):
        new = SIM_RE.sub(rf'\g<1>{val_txt}', txt)
    else:
        sep = "" if (txt.endswith("\n") or txt == "") else "\n"
        new = txt + f"{sep}SIMULATION_MODE = {val_txt}\n"
    write_atomic(CONFIG_PY, new)


def read_training_subject(default="PILOT007"):
    """
    Read `TRAINING_SUBJECT` from `config.py`.
    """
    txt = read_text(CONFIG_PY)
    m = SUBJECT_RE.search(txt)
    return m.group(3) if m else default


def write_training_subject(val: str):
    """
    Update `TRAINING_SUBJECT = "<val>"` inside `config.py`.
    """
    txt = read_text(CONFIG_PY)
    if SUBJECT_RE.search(txt):
        new = SUBJECT_RE.sub(rf'\g<1>"{val}"', txt)
    else:
        sep = "" if (txt.endswith("\n") or txt == "") else "\n"
        new = txt + f"{sep}TRAINING_SUBJECT = \"{val}\"\n"
    write_atomic(CONFIG_PY, new)


def read_fes_toggle(default=0):
    """
    Read `FES_toggle` from `config.py` as an integer 0/1.
    """
    txt = read_text(CONFIG_PY)
    m = FES_RE.search(txt)
    try:
        return int(m.group(2)) if m else default
    except Exception:
        return default


def write_fes_toggle(val: int):
    """
    Update `FES_toggle = 0/1` inside `config.py`.

    Any truthy value is normalized to 1, otherwise 0.
    """
    val = 1 if val else 0
    txt = read_text(CONFIG_PY)
    if FES_RE.search(txt):
        new = FES_RE.sub(rf'\g<1>{val}', txt)
    else:
        sep = "" if (txt.endswith("\n") or txt == "") else "\n"
        new = txt + f"{sep}FES_toggle = {val}\n"
    write_atomic(CONFIG_PY, new)


# Keys that live in config_local.py (machine-local). config.py holds safe
# defaults; the bottom of config.py does `from config_local import *` so a
# value set here shadows the default at import time. Mirrored in
# ~/.claude/hooks/config-py-guard.sh — keep both lists in sync.
LOCAL_CONFIG_KEYS = frozenset({
    "WORKING_DIR", "DATA_DIR",
    "GAZE_UDP_IP", "GAZE_BIND_HOST",
    "NEON_COMPANION_HOST",
    "PERCEPTION_FRAME_SOURCE", "SERVICES_HOSTED_REMOTELY",
    "FRAME_RELAY_HOST", "FRAME_RELAY_DIAL_HOST",
    "PERCEPTION_MODELS_DIR", "GOOGLE_API_KEY",
    "VLM_SERVICE_HOST", "VLM_BIND_HOST",
    "ARDUINO_PORT",
})


def _assign_line_re(name: str) -> re.Pattern:
    return re.compile(rf"^(\s*{re.escape(name)}\s*=\s*)([^#\n]+?)(\s*(#.*)?)\s*$", re.M)


def _find_assignment(name: str):
    """Return (file_path, match_object) for the first config file that
    contains an assignment to ``name``. config_local.py is checked first
    so local overrides take precedence — same precedence the live import
    uses at runtime. Returns (None, None) if neither file has the key."""
    pat = _assign_line_re(name)
    if os.path.isfile(CONFIG_LOCAL_PY):
        m = pat.search(read_text(CONFIG_LOCAL_PY))
        if m:
            return CONFIG_LOCAL_PY, m
    m = pat.search(read_text(CONFIG_PY))
    if m:
        return CONFIG_PY, m
    return None, None


def _read_float_key(name: str, default: float) -> float:
    _, m = _find_assignment(name)
    if not m:
        return default
    try:
        return float(m.group(2).strip())
    except ValueError:
        return default


def _read_int_key(name: str, default: int) -> int:
    _, m = _find_assignment(name)
    if not m:
        return default
    try:
        return int(float(m.group(2).strip()))
    except ValueError:
        return default


def _read_bool_key(name: str, default: bool) -> bool:
    _, m = _find_assignment(name)
    if not m:
        return default
    v = m.group(2).strip()
    if v == "True":
        return True
    if v == "False":
        return False
    return default


def _read_quoted_str_key(name: str, default: str) -> str:
    _, m = _find_assignment(name)
    if not m:
        return default
    raw = m.group(2).strip()
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    return raw


def _write_assign_rhs_local(name: str, rhs: str) -> None:
    """Write/update an assignment in config_local.py.

    config_local.py is gitignored and may not exist on a fresh checkout —
    if missing, we create it with a single-line header. If the key is
    already assigned, the line is rewritten in place; otherwise a new
    line is appended.
    """
    header = (
        "# Machine-local config overrides. Gitignored. Loaded by config.py\n"
        "# via `from config_local import *`. Schema in config_local.example.py.\n"
    )
    if not os.path.isfile(CONFIG_LOCAL_PY):
        write_atomic(CONFIG_LOCAL_PY, header)
    txt = read_text(CONFIG_LOCAL_PY)
    pat = _assign_line_re(name)
    if pat.search(txt):
        new = pat.sub(rf"\g<1>{rhs}\3", txt, count=1)
    else:
        sep = "" if (txt.endswith("\n") or txt == "") else "\n"
        new = txt + f"{sep}{name} = {rhs}\n"
    write_atomic(CONFIG_LOCAL_PY, new)


def _write_assign_rhs(name: str, rhs: str):
    """Route writes to the right file. Machine-local keys land in
    config_local.py; everything else stays in config.py (where the
    runtime-config tab's algorithm settings live)."""
    if name in LOCAL_CONFIG_KEYS:
        _write_assign_rhs_local(name, rhs)
        return
    txt = read_text(CONFIG_PY)
    pat = _assign_line_re(name)
    if not pat.search(txt):
        raise ValueError(f"config.py: assignment for {name} not found")
    new = pat.sub(rf"\g<1>{rhs}\3", txt, count=1)
    write_atomic(CONFIG_PY, new)


def _read_01_key(name: str, default: int) -> int:
    v = _read_int_key(name, default)
    return 1 if v else 0


ARDUINO_PORT_RE = re.compile(
    r"^(\s*ARDUINO_PORT\s*=\s*)([\"'])([^\"']*)\2(\s*(#.*)?)\s*$", re.M
)
ARDUINO_BAUD_RE = re.compile(
    r"^(\s*ARDUINO_BAUD\s*=\s*)(\d+)(\s*(#.*)?)\s*$", re.M
)


def read_arduino_baud_from_config(default: int = 9600) -> int:
    txt = read_text(CONFIG_PY)
    m = ARDUINO_BAUD_RE.search(txt)
    try:
        return int(m.group(2)) if m else default
    except Exception:
        return default


def write_arduino_port_to_config(port: str):
    """ARDUINO_PORT is machine-local — route through the local writer so
    the value lands in config_local.py rather than the committed
    config.py default."""
    _write_assign_rhs_local("ARDUINO_PORT", f'"{port}"')


def write_arduino_baud_to_config(baud: int):
    txt = read_text(CONFIG_PY)
    b = int(baud)
    if ARDUINO_BAUD_RE.search(txt):
        new = ARDUINO_BAUD_RE.sub(rf"\g<1>{b}\3", txt, count=1)
    else:
        sep = "" if (txt.endswith("\n") or txt == "") else "\n"
        new = txt + f"{sep}ARDUINO_BAUD = {b}\n"
    write_atomic(CONFIG_PY, new)
