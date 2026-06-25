"""
test_config_io.py — unit tests for panel.config_io.

The config-file read/write layer was extracted from control_panel.py into a leaf
module; this exercises it directly against temp config files (no panel, no Qt) —
the testability payoff of the extraction. Covers the round-trip writers/readers,
the typed key readers, and the local-vs-committed write routing that protects
machine-local keys.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from panel import config_io  # noqa: E402


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    """Point config_io at throwaway config.py / config_local.py files."""
    cfg_py = tmp_path / "config.py"
    cfg_local = tmp_path / "config_local.py"
    cfg_py.write_text(
        "SIMULATION_MODE = True\n"
        'TRAINING_SUBJECT = "PILOT007"\n'
        "FES_toggle = 0\n"
        "THRESHOLD_MI = 0.65\n"
        "N_ITERS = 12\n"
        "USE_IMU = False\n"
        'SOME_NAME = "hello"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(config_io, "CONFIG_PY", str(cfg_py))
    monkeypatch.setattr(config_io, "CONFIG_LOCAL_PY", str(cfg_local))
    return cfg_py, cfg_local


def test_training_subject_round_trip(cfg):
    assert config_io.read_training_subject() == "PILOT007"
    config_io.write_training_subject("SUBJ042")
    assert config_io.read_training_subject() == "SUBJ042"


def test_simulation_mode_round_trip(cfg):
    assert config_io.read_simulation_mode() is True
    config_io.write_simulation_mode(False)
    assert config_io.read_simulation_mode() is False


def test_fes_toggle_round_trip(cfg):
    assert config_io.read_fes_toggle() == 0
    config_io.write_fes_toggle(1)
    assert config_io.read_fes_toggle() == 1
    config_io.write_fes_toggle(0)  # truthy normalisation
    assert config_io.read_fes_toggle() == 0


def test_typed_key_readers(cfg):
    assert config_io._read_float_key("THRESHOLD_MI", 0.0) == 0.65
    assert config_io._read_int_key("N_ITERS", 0) == 12
    assert config_io._read_bool_key("USE_IMU", True) is False
    assert config_io._read_quoted_str_key("SOME_NAME", "") == "hello"
    # Missing key → default.
    assert config_io._read_float_key("NOPE", 9.9) == 9.9


def test_write_assign_rhs_updates_committed_key(cfg):
    cfg_py, _ = cfg
    config_io._write_assign_rhs("THRESHOLD_MI", "0.80")
    assert config_io._read_float_key("THRESHOLD_MI", 0.0) == 0.80
    assert "0.80" in cfg_py.read_text(encoding="utf-8")


def test_write_assign_rhs_missing_committed_key_raises(cfg):
    with pytest.raises(ValueError):
        config_io._write_assign_rhs("DOES_NOT_EXIST", "1")


def test_local_key_routes_to_config_local(cfg):
    """A machine-local key (in LOCAL_CONFIG_KEYS) must be written to
    config_local.py, never the committed config.py."""
    cfg_py, cfg_local = cfg
    config_io.write_arduino_port_to_config("/dev/ttyUSB9")
    assert cfg_local.exists()
    assert "/dev/ttyUSB9" in cfg_local.read_text(encoding="utf-8")
    assert "/dev/ttyUSB9" not in cfg_py.read_text(encoding="utf-8")
