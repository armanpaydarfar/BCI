"""
test_yolo_weight_tracked.py

P5-inverse (control agent's refinement in the preflight proposal): the gaze
recognizer weight `yolo26n.pt` must stay TRACKED at the repo root. It was
silently untracked once (commit 6054e07), which made a fresh clone's
`YOLO("yolo26n.pt")` fall back to an ultralytics auto-download instead of the
intended weight — re-tracked in 732144c. This contract test makes a recurrence a
red test instead of a silent field surprise.

Pure / hardware-free; fast suite. Skips if not in a git checkout.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")
def test_yolo26n_is_git_tracked_at_repo_root():
    # The recognizer (vlm_service --recognizer-model) and the gaze tracker both
    # expect this file at the repo root; YOLO("yolo26n.pt") resolves relative to
    # cwd, so it must travel with the repo, not be auto-downloaded.
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "yolo26n.pt"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert out.returncode == 0 and "yolo26n.pt" in out.stdout, (
        "yolo26n.pt is not git-tracked at the repo root — a 6054e07-style silent "
        "untrack would make YOLO auto-download a different weight. Re-add it "
        "(see the !/yolo26n.pt exception in .gitignore)."
    )
