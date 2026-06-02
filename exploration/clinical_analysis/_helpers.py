"""Shared helpers for the CLIN_* pass-1 analysis suite.

All helpers below are analysis-only. They wrap (without modifying)
existing analysis-side functions from `exploration/preprocessing_sweep/`
and replicate small format-parsing rules where no existing helper
exists (e.g. `training data:` line in `event_log.txt`).

Imports of Tier 1 / Tier 2 files are forbidden per CLAUDE.md. See
`Documents/SoftwareDocs/projects/harmony-bci/clinical-analysis/pass1-reuse-map.md`.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path
from typing import Iterable

# Make the preprocessing_sweep helpers importable when this package is
# loaded from the repo root or any other working directory. The sweep
# scripts import `Utils.stream_utils` (read-only Tier 1, freely
# importable per CLAUDE.md analysis-side policy).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SWEEP_DIR = _REPO_ROOT / "exploration" / "preprocessing_sweep"
for _p in (str(_REPO_ROOT), str(_SWEEP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ----------------------------------------------------------------------
# Cohort enumeration
# ----------------------------------------------------------------------

def enumerate_clin_subjects() -> list[str]:
    """Return the 7 CLIN_SUBJ_002..008 cohort IDs.

    CLIN_PILOT_001 is excluded per the cohort definition
    (`sweep_phase3_validation.py:17,99`).
    """
    return [f"CLIN_SUBJ_{i:03d}" for i in range(2, 9)]


def enumerate_online_sessions_for_subject(subject: str) -> list[str]:
    """Sorted ONLINE session labels for one CLIN subject.

    Wraps `exploration/preprocessing_sweep/sweep_phase3_validation.py:133-140`
    so all pass-1 scripts share one enumeration path.
    """
    from sweep_phase3_validation import enumerate_online_sessions  # noqa: WPS433
    return enumerate_online_sessions(subject)


# ----------------------------------------------------------------------
# event_log.txt training-pool snapshot
# ----------------------------------------------------------------------

# The driver writes the active expert pool to event_log.txt at session
# start; the line format (verified on
# `sub-CLIN_SUBJ_004/ses-S001ONLINE/.../event_log.txt:11`) is:
#   "<ts> [INFO] training data: ['<abs_path>.xdf', '<abs_path>.xdf', ...]"
# Reference: `ExperimentDriver_Online.py:173-184` (and the other
# driver variants per `rev01-paper-angle.md` §1.1).
_TRAINING_DATA_RE = re.compile(r"training data:\s*(\[.*?\])\s*$")


def parse_training_pool_from_event_log(event_log_path: str | Path) -> list[str]:
    """Extract the per-run training-pool XDF paths from event_log.txt.

    Returns the list of absolute XDF paths the driver enumerated at
    session start (the expert pool that fed the deployed MDM). Returns
    an empty list if no `training data: [...]` line is found.
    """
    path = Path(event_log_path)
    if not path.exists():
        return []
    with open(path, "r") as f:
        for line in f:
            m = _TRAINING_DATA_RE.search(line)
            if not m:
                continue
            try:
                files = ast.literal_eval(m.group(1))
            except (SyntaxError, ValueError):
                continue
            if isinstance(files, list):
                return [str(p) for p in files]
    return []


# ----------------------------------------------------------------------
# Motor cluster resolution (per rev01-erd-refinement-plan.md §3, §4)
# ----------------------------------------------------------------------

# Constants per `rev01-erd-refinement-plan.md`:
#   §4.1 CONTRA_MOTOR_CLUSTER (right-arm MI: left hemisphere)
#   §4.2 BILATERAL_MOTOR_CLUSTER (both hemispheres)
#   §4.3 IPSI_MOTOR_CLUSTER (right hemisphere — the midline-mirror of
#       CONTRA; added 2026-06-01 to replace the redundant motor-focal
#       panel row, since the focal pool overlapped almost entirely with
#       BILATERAL and offered little marginal information)
#   §3.1 MOTOR_CLUSTER (intersection of config.MOTOR_CHANNEL_NAMES with
#       ZONES L-/R-motor + Cz — used as the focal-electrode search pool)
CONTRA_MOTOR_CLUSTER = ["C3", "FC1", "CP1", "CP5"]
IPSI_MOTOR_CLUSTER = ["C4", "FC2", "CP2", "CP6"]
BILATERAL_MOTOR_CLUSTER = [
    "FC1", "FC2", "C3", "C4", "CP1", "CP2", "CP5", "CP6",
]
MOTOR_FOCAL_POOL = [
    "FC1", "FC2", "C3", "Cz", "C4", "CP1", "CP2", "CP5", "CP6",
]


def resolve_motor_cluster(ch_names: Iterable[str]) -> dict[str, list[str]]:
    """Resolve the motor-cluster definitions against a session's
    surviving channel list.

    `ch_names` is the channel list present after Config A preprocessing
    (drop_fp removes Fp1/Fp2/Fpz, auto-drop may remove up to 4 more
    channels per `sweep_phase3_validation.py:123-126`). The returned
    lists preserve cluster-definition order and contain only channels
    that survived.
    """
    present = set(ch_names)
    return {
        "contralateral":  [c for c in CONTRA_MOTOR_CLUSTER if c in present],
        "ipsilateral":    [c for c in IPSI_MOTOR_CLUSTER if c in present],
        "bilateral":      [c for c in BILATERAL_MOTOR_CLUSTER if c in present],
        "motor_focal_pool": [c for c in MOTOR_FOCAL_POOL if c in present],
    }


# ----------------------------------------------------------------------
# Config A preprocessing pipeline wrapper
# ----------------------------------------------------------------------

# Config A constants per `sweep_phase3_validation.py:101-107`.
CONFIG_A = {
    "spatial_filter":    "car",
    "blink_removal":     "drop_fp",
    "baseline_mode":     "logratio",
    "spectral_baseline": (-1.5, -0.25),
}


def config_a_pipeline(subject: str, session: str) -> dict:
    """Run Config A preprocess + TFR for one (subject, session).

    Thin wrapper around
    `exploration/preprocessing_sweep/generate_plots_config_a.py:88-161`
    (`preprocess_and_tfr`). Returns the same dict:
        {tfr_avg, tfr_trials, dropped_channels, n_kept, n_attempted}

    `tfr_avg[marker]` is an `mne.time_frequency.AverageTFR`.
    `tfr_trials[marker]` is an `EpochsTFR` (per-trial, baseline-corrected,
    cropped to TRIAL_WIN = (-1, 4) s).
    """
    from generate_plots_config_a import preprocess_and_tfr  # noqa: WPS433
    return preprocess_and_tfr(subject, session, CONFIG_A)


# ----------------------------------------------------------------------
# Output paths
# ----------------------------------------------------------------------

_PICTURES_ROOT = Path.home() / "Pictures" / "clin_analysis"


def clin_pictures_root() -> Path:
    """Return the `~/Pictures/clin_analysis/` root directory.

    Creates it (and parents) on demand. Subdirectories (e.g.
    `eds/`, `erd_refined/`) are created lazily by callers via
    `(clin_pictures_root() / "eds").mkdir(parents=True, exist_ok=True)`.
    """
    _PICTURES_ROOT.mkdir(parents=True, exist_ok=True)
    return _PICTURES_ROOT


# ----------------------------------------------------------------------
# Misc utilities used by multiple scripts
# ----------------------------------------------------------------------

def session_idx_from_label(session_label: str) -> int:
    """Extract the integer N from a session label like 'S003ONLINE' → 3.

    Returns 999 if no match (mirrors
    `Analyze_experiment_logs_cross_subject.py:231-234`).
    """
    m = re.search(r"S(\d+)", session_label)
    return int(m.group(1)) if m else 999
