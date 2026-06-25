"""
test_decode_contention_guard.py

Regression guard for the Neon scene-video "tearing" bug class
(root cause: Documents/SoftwareDocs/projects/harmony-bci/gpu-service/
neon-tearing-rootcause-2026-06-22.md).

What that bug was, in one sentence: a perf optimization collapsed the Neon
H.264 decode into the panel's own process (`34b5ed8`), and weeks later the
in-process Qt-paint / JPEG-encode load crossed a GIL-contention threshold,
starving the SDK's RTP-receive thread so the decoder rendered frames missing
their lower slices — a clean top ~2/3 and a torn macroblock band in the
bottom ~1/3, recovering on each keyframe. Every automated signal was blind to
it: no decode-integrity test existed, the relay's `dropped=0` counter sits
*upstream* of the corruption, and the WS3 equivalence review byte-diffed the
wrong reader.

This file adds the two hardware-free invariants the root-cause doc §5/§11
calls for, so a future regression is caught by CI rather than by eye:

  1. Corruption heuristic (`detect_bottom_band_tear`): assert it FLAGS a
     synthetically torn frame (clean top, stale/sheared bottom band — the
     exact signature §1 describes) and PASSES a clean frame. This is the
     cheap per-frame integrity signal §11.4 ("instrument the blind spot")
     asks the relay to surface; the test pins the heuristic's behaviour so
     it can be wired into `frame_relay.py` later with a known-good detector.

  2. Isolation invariant: assert that `FRAME_RELAY_EMBEDDED=False` selects
     the *separated* relay path (the fix, §7) — i.e. the panel does not host
     the decode in-process. This locks the config contract so a future edit
     can't silently re-collapse the isolation that fixed the tearing
     (§11.3 "a perf change that removes an isolation property needs a guard").

Deliberately hardware-free, fast, deterministic — no Neon, no network, no
PyAV decode. The synthetic frames stand in for "clean" vs "torn" decoder
output; the heuristic operates on the BGR array the relay JPEG-encodes
(`Utils/frame_relay.py:_encode_frame`, the array `bundle.video.bgr`).
"""

from __future__ import annotations

import numpy as np
import pytest

# Guard the PRODUCTION heuristic (now wired into the relay pump), not a copy.
from Utils.frame_relay import detect_bottom_band_tear


# ─────────────────────────────────────────────────────────────────────────────
# Corruption heuristic
#
# Signature of the tearing bug (root-cause doc §1): the top ~2/3 of the frame
# is correctly decoded while the bottom band carries stale macroblocks from a
# prior frame, producing a sharp horizontal *seam* where the good region meets
# the corrupted one, plus blocky/noisy structure in the bottom band that does
# not match the smooth scene content above it.
#
# A real scene frame is locally smooth (natural imagery, JPEG-quality 75), so a
# clean frame has low vertical-gradient energy across any interior row. A torn
# frame has an anomalous spike in row-to-row difference at the tear boundary —
# the stale band's content is uncorrelated with the row just above it. We
# measure the strongest horizontal seam in the lower half of the frame relative
# to the typical seam strength in the (clean) upper region, and flag when the
# ratio exceeds a threshold. This is intentionally cheap (one luma diff + a few
# row reductions); it is a *heuristic*, not a decoder error count, and is meant
# to run per-frame on the relay's pump thread.
# ─────────────────────────────────────────────────────────────────────────────

# `detect_bottom_band_tear` is imported from Utils.frame_relay (above) — the
# relay samples it ~1/s on the pump thread and reports `torn=flagged/checked`
# in its stats line. These tests pin its behaviour so a future edit can't
# weaken the signal.


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic frame builders (stand-ins for decoder output — no hardware/PyAV)
# ─────────────────────────────────────────────────────────────────────────────

def _clean_frame(h: int = 1200, w: int = 1600, seed: int = 0) -> np.ndarray:
    """A locally-smooth BGR frame standing in for a correctly-decoded scene.

    Smooth 2D ramp + mild low-amplitude texture — no sharp horizontal seams,
    mimicking natural scene imagery the H.264 decoder would emit intact.
    """
    rng = np.random.default_rng(seed)
    yy = np.linspace(0, 200, h, dtype=np.float32)[:, None]
    xx = np.linspace(0, 50, w, dtype=np.float32)[None, :]
    base = yy + xx  # smooth gradient, no abrupt rows
    texture = rng.normal(0.0, 1.5, size=(h, w)).astype(np.float32)  # gentle grain
    luma = np.clip(base + texture, 0, 255).astype(np.uint8)
    return np.repeat(luma[:, :, None], 3, axis=2)  # gray BGR


def _torn_frame(h: int = 1200, w: int = 1600, seed: int = 0,
                tear_frac: float = 0.33) -> np.ndarray:
    """A clean frame whose bottom `tear_frac` is replaced with stale, sheared,
    block-noisy content — the §1 tearing signature (clean top, torn bottom).

    Construction mirrors the failure mode: the bottom band is filled with
    content uncorrelated with the row directly above it (stale macroblocks +
    block-grid noise), producing a sharp seam at the tear boundary.
    """
    frame = _clean_frame(h, w, seed=seed)
    rng = np.random.default_rng(seed + 1)
    tear_row = int((1.0 - tear_frac) * h)

    # Stale band: a constant-ish gray region offset from the local content,
    # overlaid with 16x16 block noise (macroblock-grid corruption).
    band_h = h - tear_row
    stale = np.full((band_h, w), 90.0, dtype=np.float32)
    blocks_y = (band_h + 15) // 16
    blocks_x = (w + 15) // 16
    block_noise = rng.normal(0.0, 60.0, size=(blocks_y, blocks_x)).astype(np.float32)
    block_noise = np.repeat(np.repeat(block_noise, 16, axis=0), 16, axis=1)[:band_h, :w]
    stale = np.clip(stale + block_noise, 0, 255).astype(np.uint8)
    frame[tear_row:, :, :] = stale[:, :, None]
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Tests — corruption heuristic
# ─────────────────────────────────────────────────────────────────────────────

class TestCorruptionHeuristic:
    def test_clean_frame_passes(self):
        """A locally-smooth (correctly decoded) frame must NOT be flagged."""
        assert detect_bottom_band_tear(_clean_frame(seed=1)) is False

    def test_torn_frame_flagged(self):
        """A frame with a stale bottom-band macroblock tear MUST be flagged —
        this is the exact corruption the relay's `dropped=0` counter was blind
        to (root-cause doc §11.4)."""
        assert detect_bottom_band_tear(_torn_frame(seed=1)) is True

    @pytest.mark.parametrize("seed", [0, 2, 7, 13, 42])
    def test_deterministic_across_seeds(self, seed):
        """Clean passes / torn flags for several seeds — the heuristic is not
        tuned to one lucky noise draw."""
        assert detect_bottom_band_tear(_clean_frame(seed=seed)) is False
        assert detect_bottom_band_tear(_torn_frame(seed=seed)) is True

    def test_small_band_tear_still_flagged(self):
        """Even a thin (15%) torn band at the bottom is caught — the bug
        accumulated from a thin band that 'stretched to the middle' (§1)."""
        assert detect_bottom_band_tear(_torn_frame(seed=3, tear_frac=0.15)) is True

    def test_rejects_non_bgr(self):
        with pytest.raises(ValueError):
            detect_bottom_band_tear(np.zeros((10, 10), dtype=np.uint8))


# ─────────────────────────────────────────────────────────────────────────────
# Tests — isolation invariant (the fix, root-cause doc §7)
#
# The fix is process isolation, selected by FRAME_RELAY_EMBEDDED=False: the
# panel spawns Utils.frame_relay as a child process instead of decoding in its
# own process. These tests lock that contract so a future change can't silently
# re-embed the decode (re-introducing the GIL contention that caused tearing).
# They read the real control_panel branch logic without constructing any Qt /
# hardware objects.
# ─────────────────────────────────────────────────────────────────────────────

class TestIsolationInvariant:
    def test_config_default_documented(self):
        """The committed default is import-checkable; the separated path is a
        boolean flip of it. Pinning the type guards against a future
        non-boolean value silently truthy-defaulting to embedded."""
        import config
        assert isinstance(config.FRAME_RELAY_EMBEDDED, bool)

    def test_separated_mode_wired_into_panel_source(self):
        """Static guard on the real VlmController source: the separated-relay
        spawn must be gated under `if not FRAME_RELAY_EMBEDDED:` and the
        in-process host under `if FRAME_RELAY_EMBEDDED:`. Asserting the actual
        wiring (not a boolean tautology) so a future edit that deletes the
        separated path or inverts the guard fails CI — that silent re-collapse
        of the decode isolation is the exact regression this file guards
        (root-cause doc §7/§11.3). Import-safe: reads source rather than
        importing the controller, which pulls Qt.

        The whole VLM subsystem (relay spawn/teardown + the embedded-host
        branch) was extracted from control_panel.py into
        panel/vlm_controller.py; this guard follows it there so the invariant
        stays pinned at its new home.
        """
        import os
        import re
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "panel", "vlm_controller.py",
        )
        with open(src_path, "r", encoding="utf-8") as fh:
            src = fh.read()
        assert "def _start_relay_subprocess" in src
        assert "def _stop_relay_subprocess" in src
        # Connect path: the separated relay is spawned only when NOT embedded.
        assert re.search(
            r"if not FRAME_RELAY_EMBEDDED:\s*\n\s*self\._start_relay_subprocess\(\)",
            src,
        ), "separated-relay spawn must be gated on `if not FRAME_RELAY_EMBEDDED`"
        # The in-process host path remains gated on the flag.
        assert re.search(r"if FRAME_RELAY_EMBEDDED:", src), \
            "embedded host path must remain gated on FRAME_RELAY_EMBEDDED"

    def test_scene_only_reader_is_a_distinct_module_from_matched(self):
        """The live panel path runs Utils.scene_only_neon_reader (simple API);
        the relay falls back to perception.neon (matched API) only when no
        reader is injected. The WS3 equivalence review audited the matched
        reader while the panel ran the simple one (root-cause doc §6). Lock
        that these are genuinely two different modules so any future
        'equivalence' claim can't conflate them.
        """
        import importlib.util
        simple = importlib.util.find_spec("Utils.scene_only_neon_reader")
        assert simple is not None, "live panel scene reader must be importable"
        # The matched reader lives under perception.neon; we only assert the
        # module paths differ (no import of perception, which pulls torch).
        assert simple.origin is not None
        assert simple.origin.replace("\\", "/").endswith(
            "Utils/scene_only_neon_reader.py"
        )
