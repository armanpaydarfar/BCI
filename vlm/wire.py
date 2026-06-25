"""
vlm/wire.py — JSON serialization helpers for the VLM service wire protocol.

Behaviour-preserving extraction of the module-level serialization helpers from
vlm_service.py: the numpy-aware json.dumps default, and the compact JSON-safe
views of a Detection / decision dict pushed over the UDP results channel. Pure
functions (stdlib + numpy + duck-typed Detection access) — no VLMService state,
no _log — so the service re-imports them by name (call sites unchanged) and they
stay unit-testable in isolation (tests/test_vlm_results_push.py).
"""

from __future__ import annotations

import numpy as np


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"unsupported type for JSON: {type(o)}")


def _serialize_detection_for_push(d) -> dict:
    """Compact JSON-safe view of a harmony_vlm Detection for the UDP push.

    Mask polygons are int-quantised vertex lists (already int via .astype(int)
    in _cmd_segment) — small enough that 5-10 detections at 20 Hz stay
    comfortably under the 60 KB datagram budget."""
    box_xyxy = [float(v) for v in getattr(d, "box_xyxy", (0.0, 0.0, 0.0, 0.0))]
    out = {
        "label": getattr(d, "label", None),
        "confidence": float(getattr(d, "confidence", 0.0)),
        "box_xyxy": box_xyxy,
        "box_center": [float(v) for v in getattr(d, "box_center", (0.0, 0.0))],
    }
    poly = getattr(d, "mask_polygon", None)
    if poly is not None:
        try:
            out["mask_polygon"] = poly.reshape(-1, 2).astype(int).tolist()
        except Exception:
            pass
    # WS4 F3: stable tracker identity, present only when --seg-track is on
    # (the stateless path's Detections have no track_id attribute).
    tid = getattr(d, "track_id", None)
    if tid is not None:
        out["track_id"] = int(tid)
    return out


def _serialize_decision_for_push(decision: dict) -> dict:
    """Trim a decision dict to fields the renderer actually paints.

    The full _last_decision can hold large nested fields (waypoints lists,
    paired-object metadata). The push payload only carries what the
    Linux-side renderer needs to draw the decision badge."""
    keep = ("text", "object_label", "object", "second_object",
            "ts_ns", "model", "elapsed_s", "summary")
    return {k: decision[k] for k in keep if k in decision}
