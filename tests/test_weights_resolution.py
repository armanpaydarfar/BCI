"""
test_weights_resolution.py

P5 (proposal: test-suite/preflight-and-env-drift-proposal.md): lock the weights
resolution contract that `vlm_service.py:1943-1949` implements — a bare filename
joins onto PERCEPTION_MODELS_DIR; an absolute path is taken as-is. The doctor's
`tools/preflight.py:resolve_weight` mirrors that logic; this test pins both so a
silent change to either side (which would make the recognizer/seg/depth weight
resolve to the wrong place) shows up as a red test.

Pure / hardware-free; fast suite.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from preflight import resolve_weight  # noqa: E402


class TestResolveWeight:
    def test_bare_name_joins_models_dir(self):
        out = resolve_weight(os.path.join("models", "dir"), "depth_pro.pt")
        assert out == os.path.join("models", "dir", "depth_pro.pt")

    def test_absolute_path_taken_as_is(self):
        # The repo-root yolo26n.pt is passed as an absolute path; it must NOT be
        # re-joined under the models dir.
        abs_path = os.path.abspath(os.path.join("anywhere", "yolo26n.pt"))
        assert resolve_weight(os.path.join("any", "models"), abs_path) == abs_path

    def test_empty_name_returns_empty(self):
        assert resolve_weight("/models", "") == ""

    def test_empty_models_dir_with_bare_name(self):
        # No models dir + bare name → join onto "" (caller then stats and fails).
        assert resolve_weight("", "FastSAM-s.pt") == os.path.join("", "FastSAM-s.pt")
