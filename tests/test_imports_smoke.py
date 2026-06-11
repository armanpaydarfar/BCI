"""
test_imports_smoke.py

Guards the "startup SyntaxError / wrong-API-call at import time" bug class.
Example commits that this layer of testing would have caught (or moved
detection from a session-start traceback to pre-commit):

  - 88ee6e7  fix: remove double `global` SyntaxError in `update_xgb_cov_features`
  - 95a5f97  fix: remove invalid `xdawn_weights` kwarg
  - 131e0db  fix: numpy structured-array access
  - e42cf16  fix: use `resolve_byprop` for MarkerStream LSL check
             (caught at compile-only level; the bad call itself fires at
             runtime — see drift note below)

This file has two tiers:

1. `test_all_files_compile` — `py_compile.compile()` every .py under repo
   root + `Utils/` + `Utils/gaze/` + `STM_interface/`, skipping vendor
   `rehamoveLibrary/` and the Tk-GUI `STMsetup.py`. Catches SyntaxError class
   (`88ee6e7`, etc.) and any other parser-level breakage in any committed file.

2. `test_safe_modules_import` — `importlib.import_module` for the curated
   "should-be-library" subset under `Utils/` and the post-refactor
   `UTIL_marker_stream` / `FES_listener`. Catches import-time API-shape
   mismatches and verifies the §5.1.a/b refactors stuck.

NOT covered here: importing top-level driver scripts (ExperimentDriver_*.py,
control_panel.py, harmony_*.py, generate_*.py, *_viewer.py, etc.). These
files do non-trivial work at module top — pygame display init, model file
loads, XDF directory scans, logger directory creation — which makes
"importlib.import_module" unsafe inside a pre-commit hook. They are still
covered by the compile pass.

Tier 1 file citations this test depends on:
  - Utils/networking.py:64-67 — SIMULATION_MODE snapshot at import; see
    conftest.py and the §4 gotcha in Harmony_Test_Suite_Plan.md.
  - UTIL_marker_stream.py — post-refactor module-level state is None
    until main() runs (see refactor 5.1.a).
  - FES_listener.py — post-refactor module-level state is None until
    main() runs (see refactor 5.1.b).

Plan-drift note (see Harmony_Test_Suite_Plan.md §10):
  - The plan claimed `e42cf16` would be caught by this test. That commit
    fixed a bad `resolve_stream(..., minimum=1, timeout=...)` call inside
    `Utils.stream_utils.require_marker_stream`, which only fires when that
    function is called — importing the module does not exercise the bad
    call site. Compile/import catches the *SyntaxError* and
    *import-time-API* bug classes; runtime-call signature bugs need a
    functional test of the call site. Listed here for transparency.
"""

from __future__ import annotations

import importlib
import io
import contextlib
import py_compile
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories the smoke test walks for both compile and (filtered) import.
_SCAN_DIRS = [
    REPO_ROOT,                          # top-level scripts
    REPO_ROOT / "Utils",
    REPO_ROOT / "Utils" / "gaze",
    REPO_ROOT / "STM_interface",
]

# Vendor / GUI files we deliberately do not touch (per plan §1 non-goals
# and §4 STMsetup note).
_COMPILE_EXCLUDE_DIRS = {
    REPO_ROOT / "STM_interface" / "1_packages",  # rehamoveLibrary vendor tree
    REPO_ROOT / "tests",
    REPO_ROOT / "tools",
    REPO_ROOT / ".git",
}
_COMPILE_EXCLUDE_FILES = {
    REPO_ROOT / "STMsetup.py",         # Tk GUI + serial at module top
    REPO_ROOT / "config_local.example.py",  # dotted filename — not a module
}

# Modules safe to fully import inside a pre-commit hook. Each entry must
# import without spawning a window, opening a port, writing files, or
# requiring a device. The list intentionally omits anything driver-style.
#
# FES_listener is wrapped in a pytest.param so it can be marked skipif on
# Windows — its rehamove SWIG dependency is a Linux-only `.so`
# (`STM_interface/1_packages/rehamoveLibrary/_rehamovelib.so`). Per
# Plan §6.1, cross-platform parity is deferred until a Windows CI runner
# exists; until then the test is a no-op on win32.
_LINUX_ONLY = pytest.mark.skipif(
    sys.platform == "win32",
    reason="rehamove SWIG extension (_rehamovelib.so) is Linux-only; "
           "Plan §6.1 cross-platform parity is deferred."
)
_SAFE_IMPORT_MODULES = [
    # Configuration
    "config",
    "config_local",
    # Post-refactor Tier 1 entrypoints (Plan §5.1.a, §5.1.b)
    "UTIL_marker_stream",
    pytest.param("FES_listener", marks=_LINUX_ONLY),
    # Utils library modules
    "Utils.artifact_rejection",
    "Utils.EEGStreamState",
    "Utils.errp_alignment",
    "Utils.errp_feature_pipeline",
    "Utils.errp_liu_pipeline",
    "Utils.experiment_utils",
    "Utils.frame_relay",
    "Utils.liu_data_loader",
    "Utils.logging_manager",
    "Utils.marker_pairing",
    "Utils.networking",
    "Utils.perception_clients",
    "Utils.preprocessing",
    "Utils.remote_frame_reader",
    "Utils.runtime_common",
    "Utils.scene_only_neon_reader",
    "Utils.scene_overlay_renderer",
    "Utils.sleep_inhibit",
    "Utils.stream_utils",
    "Utils.tangent_feature_labels",
    "Utils.transfer_benchmark_core",
    "Utils.visualization",
    "Utils.vlm_scene_widget",
    "Utils.vlm_subscriber",
    "Utils.xgb_feature_pipeline",
    "Utils.xgb_train_eval",
    # Gaze submodules (pure per their docstrings)
    "Utils.gaze.gaze_math",
    "Utils.gaze.gaze_render",
    "Utils.gaze.gaze_system",
    "Utils.gaze.gaze_tracking",
    "Utils.gaze.gaze_ui",
    # STM_interface (non-vendor)
    "STM_interface.RehamoveConfig",
]

# Library modules excluded from the import tier with reasons. Compile tier
# still covers them.
_IMPORT_EXCLUSIONS = {
    "Utils.Montage_creator": (
        "Script-style: saves CA-209-dig.fif and renders a matplotlib plot at "
        "import time. Compile tier still catches SyntaxError."
    ),
}


def _iter_py_files():
    for d in _SCAN_DIRS:
        if not d.is_dir():
            continue
        for path in d.glob("*.py"):
            if path in _COMPILE_EXCLUDE_FILES:
                continue
            yield path


def test_all_files_compile():
    """Every .py in the scanned dirs must parse. Catches SyntaxError class
    such as `88ee6e7` (double `global`)."""
    failures = []
    for path in _iter_py_files():
        # Skip vendor sub-trees that happen to live under STM_interface
        if any(str(path).startswith(str(d)) for d in _COMPILE_EXCLUDE_DIRS):
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as e:
            failures.append(f"{path.relative_to(REPO_ROOT)}: {e.msg.strip()}")
    assert not failures, (
        "py_compile failed for:\n  " + "\n  ".join(failures)
    )


@pytest.mark.parametrize("module_name", _SAFE_IMPORT_MODULES)
def test_safe_modules_import(module_name):
    """Curated library modules must `importlib.import_module` cleanly with
    SIMULATION_MODE on and headless pygame. Catches import-time API or
    dependency shape regressions."""
    # Some modules print noisy banners (pygame, montages, etc.). Suppress
    # stdout/stderr so test output stays readable; an exception still raises.
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        importlib.import_module(module_name)


def test_util_marker_stream_imports_without_hardware():
    """Refactor 5.1.a contract: importing UTIL_marker_stream must not open
    an LSL outlet or create the subject log directory. The module-level
    `outlet` and `subject_log_dir` should be None until `main()` runs."""
    # Already imported by the parametrized test above, but re-import here
    # to make the contract explicit and surface a clear failure message.
    mod = importlib.import_module("UTIL_marker_stream")
    assert mod.outlet is None, (
        "UTIL_marker_stream.outlet should be None at import; "
        "refactor 5.1.a regressed."
    )
    assert mod.subject_log_dir is None, (
        "UTIL_marker_stream.subject_log_dir should be None at import; "
        "refactor 5.1.a regressed."
    )


@_LINUX_ONLY
def test_fes_listener_imports_without_hardware():
    """Refactor 5.1.b contract: importing FES_listener must not open the
    Rehamove serial port or bind the FES UDP socket. Module-level
    `FES_device` and `sock` should be None until `main()` runs.

    Skipped on Windows: depends on `_rehamovelib.so` (Linux SWIG ext).
    Plan §6.1 defers cross-platform parity until a Windows CI runner exists."""
    mod = importlib.import_module("FES_listener")
    assert mod.FES_device is None, (
        "FES_listener.FES_device should be None at import; "
        "refactor 5.1.b regressed."
    )
    assert mod.sock is None, (
        "FES_listener.sock should be None at import; refactor 5.1.b regressed."
    )


def test_import_exclusions_are_documented():
    """Self-test: every excluded module has a non-empty rationale. Forces
    future authors to justify additions to the exclusion list."""
    for mod, reason in _IMPORT_EXCLUSIONS.items():
        assert reason and len(reason) > 20, (
            f"Exclusion for {mod} needs a documented rationale."
        )
