"""
test_snapshot_cache.py — unit tests for vlm.snapshot_cache.SnapshotCache.

The bounded TTL snapshot cache (the capture_first → decide_pair two-object flow)
was extracted from vlm_service.py into a leaf module; this exercises it directly
— put/get/pop round-trips, TTL expiry, and max-size eviction of the oldest entry
— with no Neon, no UDP, no VLMService. Deterministic: TTL is driven by a
monkeypatched monotonic clock, never wall-clock sleeps.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from vlm.snapshot_cache import SnapshotCache  # noqa: E402


def test_put_get_round_trip():
    cache = SnapshotCache()
    sid = cache.put({"gaze": (10, 20)})
    got = cache.get(sid)
    assert got is not None
    assert got["gaze"] == (10, 20)
    # The stored entry carries an internal timestamp key alongside the payload.
    assert "_t" in got


def test_get_unknown_id_returns_none():
    cache = SnapshotCache()
    assert cache.get("deadbeef") is None


def test_pop_removes_entry():
    cache = SnapshotCache()
    sid = cache.put({"a": 1})
    assert cache.pop(sid) is not None
    assert cache.get(sid) is None
    # Second pop on an already-removed id is a clean None.
    assert cache.pop(sid) is None


def test_ttl_expiry(monkeypatch):
    """An entry older than ttl_s is pruned on the next access."""
    clock = {"now": 1000.0}
    monkeypatch.setattr(
        "vlm.snapshot_cache.time.monotonic", lambda: clock["now"]
    )
    cache = SnapshotCache(ttl_s=60.0)
    sid = cache.put({"a": 1})
    clock["now"] = 1059.0  # within TTL
    assert cache.get(sid) is not None
    clock["now"] = 1061.0  # past TTL
    assert cache.get(sid) is None


def test_max_size_evicts_oldest(monkeypatch):
    """Once at capacity, put() evicts the oldest entry to make room."""
    clock = {"now": 0.0}
    monkeypatch.setattr(
        "vlm.snapshot_cache.time.monotonic", lambda: clock["now"]
    )
    cache = SnapshotCache(ttl_s=10_000.0, max_size=2)
    a = cache.put({"k": "a"})
    clock["now"] = 1.0
    b = cache.put({"k": "b"})
    clock["now"] = 2.0
    c = cache.put({"k": "c"})  # at capacity → evicts the oldest (a)
    assert cache.get(a) is None
    assert cache.get(b) is not None
    assert cache.get(c) is not None
