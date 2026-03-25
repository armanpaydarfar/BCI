"""
Discover and index .xdf files under a study root with subject/session/modality heuristics.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


_SUBJECT_RE = re.compile(r"sub-([^/\\]+)", re.IGNORECASE)
_SESSION_RE = re.compile(r"(ses-[^/\\]+)", re.IGNORECASE)


def path_matches_exclude(path: str | Path, exclude_substrings: tuple[str, ...]) -> bool:
    """True if resolved path string or basename contains any exclude token (case-insensitive)."""
    if not exclude_substrings:
        return False
    p = Path(path)
    rel_l = str(p.resolve()).lower().replace("\\", "/")
    name_l = p.name.lower()
    for excl in exclude_substrings:
        s = str(excl).lower()
        if s and (s in rel_l or s in name_l):
            return True
    return False


@dataclass
class XdfIndexRow:
    path: str
    filename: str
    subject: str | None
    session_folder: str | None
    modality: str  # ONLINE, OFFLINE, UNKNOWN
    is_symlink: bool
    size_bytes: int
    in_baseline_pool: bool
    baseline_subject: str


def _infer_modality(path_str: str, filename: str) -> str:
    u = f"{path_str}/{filename}".upper()
    if "ONLINE" in u:
        return "ONLINE"
    if "OFFLINE" in u:
        return "OFFLINE"
    return "UNKNOWN"


def index_xdf_path(
    path: str | Path,
    *,
    data_dir: str | Path,
    baseline_subject: str,
) -> XdfIndexRow:
    p = Path(path).resolve()
    data_dir = Path(data_dir).resolve()
    rel = str(p)
    try:
        rel = str(p.relative_to(data_dir))
    except ValueError:
        pass

    sub_m = _SUBJECT_RE.search(rel.replace("\\", "/"))
    ses_m = _SESSION_RE.search(rel.replace("\\", "/"))
    subject = sub_m.group(1) if sub_m else None
    session_folder = ses_m.group(1) if ses_m else None

    modality = _infer_modality(rel, p.name)
    is_sym = p.is_symlink()
    size = p.stat().st_size if p.exists() else 0

    baseline_root = (data_dir / f"sub-{baseline_subject}" / "training_data").resolve()
    try:
        p_res = p.resolve()
        br = baseline_root
        in_pool = p_res == br or br in p_res.parents
    except OSError:
        in_pool = False

    return XdfIndexRow(
        path=str(p),
        filename=p.name,
        subject=subject,
        session_folder=session_folder,
        modality=modality,
        is_symlink=is_sym,
        size_bytes=size,
        in_baseline_pool=in_pool,
        baseline_subject=baseline_subject,
    )


def discover_xdf_files(
    root: str | Path,
    *,
    exclude_substrings: tuple[str, ...] = ("OBS", "old"),
    follow_symlinks: bool = False,
) -> list[str]:
    """
    Recursively list .xdf files under root.
    Skips files whose full path or basename contains any exclude token (case-insensitive).
    """
    root = Path(root)
    if not root.is_dir():
        return []

    out: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        for name in filenames:
            if not name.lower().endswith(".xdf"):
                continue
            full = Path(dirpath) / name
            if path_matches_exclude(full, exclude_substrings):
                continue
            out.append(str(full.resolve()))
    out.sort()
    return out


def iter_index_rows(
    paths: list[str],
    *,
    data_dir: str | Path,
    baseline_subject: str,
) -> Iterator[XdfIndexRow]:
    for p in paths:
        yield index_xdf_path(p, data_dir=data_dir, baseline_subject=baseline_subject)


def duplicate_groups_by_size(paths: list[str]) -> dict[int, list[str]]:
    """Group paths by file size (cheap duplicate hint; not cryptographic)."""
    from collections import defaultdict

    by_size: dict[int, list[str]] = defaultdict(list)
    for p in paths:
        try:
            sz = Path(p).stat().st_size
        except OSError:
            continue
        by_size[sz].append(p)
    return {k: v for k, v in by_size.items() if len(v) > 1}
