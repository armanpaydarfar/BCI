"""
Session-level aggregation of logs/*/config_snapshot.json for exploration filters.

Harmony layout (see Utils/logging_manager.LoggerManager):
  sub-<subject>/ses-<session>/eeg/<file>.xdf   ← typically **one** continuous recording per session
  sub-<subject>/ses-<session>/logs/ONLINE_* / OFFLINE_*/config_snapshot.json  ← **many** runs per session

The EEG filename may contain cosmetic tokens like run1; it does **not** pair 1:1 with log run folders.
All snapshots under that session’s logs/ are read together: ERRP if **any** run enables ERRP channels;
arm filter fails if **any** snapshot’s ARM_SIDE conflicts with the requested arm.

When no snapshots exist or ARM_SIDE is never set, exploration **keeps** the file (grandfathered).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def session_dir_from_xdf(xdf_path: str | Path) -> Path | None:
    """…/ses-XXX/eeg/file.xdf → …/ses-XXX"""
    p = Path(xdf_path).resolve()
    if p.parent.name.lower() != "eeg":
        return None
    return p.parents[1]


def _iter_log_run_dirs(session_dir: Path) -> list[Path]:
    logs = session_dir / "logs"
    if not logs.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(logs.iterdir()):
        if not child.is_dir():
            continue
        if not (
            child.name.startswith("ONLINE_")
            or child.name.startswith("OFFLINE_")
        ):
            continue
        if (child / "config_snapshot.json").is_file():
            out.append(child)
    return out


def load_config_snapshot(run_dir: Path) -> dict[str, Any] | None:
    snap = run_dir / "config_snapshot.json"
    if not snap.is_file():
        return None
    try:
        with open(snap, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def load_session_log_snapshots(session_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    """(log_run_folder_name, snapshot) for every ONLINE_/OFFLINE_* run with a readable JSON."""
    out: list[tuple[str, dict[str, Any]]] = []
    for run_dir in _iter_log_run_dirs(session_dir):
        snap = load_config_snapshot(run_dir)
        if snap is not None:
            out.append((run_dir.name, snap))
    return out


def _truthy(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return x != 0
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "on")
    return False


def path_suggests_left_mi_data_root(p: str) -> bool:
    """Heuristic: PILOT007-style tests under training_data/left/…"""
    return bool(re.search(r"training_data[/\\][^/\\]*left", p, re.IGNORECASE))


def path_suggests_errp_data_root(p: str) -> bool:
    """Heuristic: folder under training_data named like *errp* or ERRP in path."""
    if "errp" in p.lower():
        return True
    return bool(re.search(r"training_data[/\\][^/\\]*errp", p, re.IGNORECASE))


def exploration_filter_reason(
    xdf_path: str | Path,
    *,
    mi_arm: str | None,
    skip_errp: bool,
) -> tuple[str | None, dict[str, Any]]:
    """
    Returns (eval_error_or_None, extra_row_fields).

    mi_arm: 'right' | 'left' — if any session snapshot sets ARM_SIDE to something else, skip.
            None — do not filter on arm.
    skip_errp: if True, skip when any snapshot has SELECT_ERRP_CHANNELS truthy (or path hint).
    """
    p = Path(xdf_path)
    ps = str(p.resolve())
    extra: dict[str, Any] = {
        "n_session_log_snapshots": 0,
        "session_logs_note": "",
        "logged_arm_sides": "",
        "logged_errp_any_run": "",
    }

    if skip_errp and path_suggests_errp_data_root(ps):
        return "skipped_errp_path_hint", extra

    if mi_arm == "right" and path_suggests_left_mi_data_root(ps):
        return "skipped_left_folder_under_training_data", extra

    if mi_arm == "left" and re.search(r"training_data[/\\][^/\\]*right", ps, re.I):
        return "skipped_right_folder_under_training_data", extra

    session = session_dir_from_xdf(p)
    if session is None:
        extra["session_logs_note"] = "not_under_ses_eeg"
        return None, extra

    pairs = load_session_log_snapshots(session)
    extra["n_session_log_snapshots"] = int(len(pairs))

    if not pairs:
        if (session / "logs").is_dir():
            extra["session_logs_note"] = "logs_dir_but_no_ONLINE_OFFLINE_snapshots"
        else:
            extra["session_logs_note"] = "no_logs_dir"
        return None, extra

    extra["session_logs_note"] = f"aggregated_{len(pairs)}_log_runs"

    arms_set: set[str] = set()
    any_errp = False
    for _folder, snap in pairs:
        arm = snap.get("ARM_SIDE")
        if arm is not None and str(arm).strip() != "":
            arms_set.add(str(arm).strip().lower())
        if _truthy(snap.get("SELECT_ERRP_CHANNELS")):
            any_errp = True

    if arms_set:
        extra["logged_arm_sides"] = ",".join(sorted(arms_set))
    extra["logged_errp_any_run"] = "1" if any_errp else ""

    if skip_errp and any_errp:
        return "skipped_errp_config_snapshot_any_run", extra

    if mi_arm is not None and arms_set:
        want = mi_arm.strip().lower()
        conflicting = {a for a in arms_set if a != want}
        if conflicting:
            return (
                f"arm_mismatch:logged_arms_{'-'.join(sorted(arms_set))}_want_{want}",
                extra,
            )

    return None, extra
