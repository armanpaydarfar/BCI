"""
sleep_inhibit.py — keep the host awake while a perception service runs.

Windows-only effect; no-op on POSIX. Mirrors the helper in
``control_panel.py:_sleep_inhibit`` so the perception services
(``vlm_service.py``, ``gaze_runner.py``) can prevent sleep themselves
when launched directly on Windows — independent of whether any control
panel is up. Without this, leaving the GPU host unattended for an hour
puts it to sleep and the Linux operator has to physically wake it.

Uses ``kernel32.SetThreadExecutionState`` with
``ES_CONTINUOUS | ES_SYSTEM_REQUIRED`` per Microsoft's documentation
of unattended-service hosts. Best-effort: failures are swallowed
because sleep inhibit is hardening, not a correctness gate.
"""

from __future__ import annotations

import sys


_IS_WINDOWS = sys.platform == "win32"

# Windows execution-state flags. Documented at
# https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate
_ES_CONTINUOUS       = 0x80000000
_ES_SYSTEM_REQUIRED  = 0x00000001


def inhibit() -> None:
    """Prevent the host from sleeping. Safe to call multiple times."""
    if not _IS_WINDOWS:
        return
    try:
        import ctypes
        flags = _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))
    except Exception:
        pass


def release() -> None:
    """Clear the inhibit so the host can sleep again. Safe at process exit
    (Windows clears the flag automatically when the calling thread dies,
    so this is mostly cosmetic — but explicit cleanup is cheap)."""
    if not _IS_WINDOWS:
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(_ES_CONTINUOUS))
    except Exception:
        pass
