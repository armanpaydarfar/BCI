"""
vlm/snapshot_cache.py — bounded TTL snapshot cache for the two-object flow.

Behaviour-preserving extraction of the SnapshotCache class from vlm_service.py.
Self-contained (stdlib only: threading/uuid/time) — no VLMService state, no UDP,
no _log — so it lives in a leaf module the service re-imports by name (call sites
unchanged) and can be unit-tested in isolation (tests/test_snapshot_cache.py).
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Dict, Optional


class SnapshotCache:
    """
    Bounded TTL cache of captured frame+gaze+detections+waypoints snapshots.

    Used by the two-object (sequential-decide) flow: capture_first stores a
    snapshot under a random id; decide_pair retrieves it to pair with the
    current frame and run the VLM's reason_async_pair. Frames are large
    (~5 MB each) so we keep the cache small and short-lived.
    """

    def __init__(self, ttl_s: float = 60.0, max_size: int = 4) -> None:
        self._items: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._ttl = float(ttl_s)
        self._max = int(max_size)

    def put(self, data: dict) -> str:
        snap_id = uuid.uuid4().hex[:8]
        now = time.monotonic()
        with self._lock:
            self._prune_locked(now)
            if len(self._items) >= self._max:
                oldest = min(self._items.items(), key=lambda kv: kv[1]["_t"])[0]
                self._items.pop(oldest, None)
            self._items[snap_id] = {**data, "_t": now}
        return snap_id

    def get(self, snap_id: str) -> Optional[dict]:
        with self._lock:
            self._prune_locked(time.monotonic())
            return self._items.get(snap_id)

    def pop(self, snap_id: str) -> Optional[dict]:
        with self._lock:
            return self._items.pop(snap_id, None)

    def _prune_locked(self, now: float) -> None:
        dead = [k for k, v in self._items.items() if now - v["_t"] > self._ttl]
        for k in dead:
            self._items.pop(k, None)
