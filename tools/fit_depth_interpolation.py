#!/usr/bin/env python3
"""
tools/fit_depth_interpolation.py — Fill transit-row depths from the
bracketing anchor depths in a vlm-only calibration NPZ.

Usage:

    python tools/fit_depth_interpolation.py poses_with_gaze_<stamp>_v2_freearm.npz \\
        [--out poses_with_gaze_<stamp>_v2_freearm_interp.npz]

The recorder emits anchor rows tagged ``Depth_source == "vlm_depth_pro"``
(with the VLM Depth Pro reading at the gaze pixel) and transit rows
tagged ``Depth_source == "pending_interpolation"`` (with ``D_cm = NaN``).
This script:

1. Loads the NPZ.
2. Splits the fit-block rows by ``Depth_source`` into anchors vs transit.
3. For each transit row, identifies the bracketing anchors from its
   ``Leg_label`` (format ``"transit_<from_label>_to_<to_label>"``) and
   linearly interpolates ``D_cm`` by the row's relative EE distance
   between the two anchor EE positions:

       t = d_from / (d_from + d_to)
       D_cm = (1 - t) * D_cm_from + t * D_cm_to

   This is "bracketed" interpolation — sequential waypoints are
   typically close together in space, and constraining the interp to
   the two anchors that physically span the leg avoids pulling depth
   from a far-away anchor that happens to be the global NN.

4. Falls back to global KDTree NN over anchor EE positions when leg
   info is missing (legacy NPZ) or when one of the bracketing anchors
   has no anchor row (e.g., the recorder dropped that anchor capture).
5. Rewrites transit rows with the interpolated ``D_cm`` and tags
   ``Depth_source = "vlm_interpolated_bracketed"`` (or
   ``"vlm_interpolated_nearest_anchor"`` for fallback rows).
6. Pins ``meta["depth_source"] = "vlm_depth_pro"`` so the v2 runtime
   dispatches to its per-query VLM Depth Pro path.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


# Leg labels have the shape "transit_<from>_to_<to>" where <from> /
# <to> can be anything but the literal "_to_" infix. Anchor at WP01 is
# tagged Target_label="wp01"; the matching leg is
# "transit_wp01_to_wp02". The "start" sentinel is used by the recorder
# when the first transit leg begins before any anchor (should not
# happen in WP1-skip mode, but tolerated for robustness).
_LEG_PATTERN = re.compile(r"^transit_(?P<from>.+?)_to_(?P<to>.+)$")


def _split_anchors_and_transit(
    depth_source: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (anchor_mask, transit_mask) over the fit-block rows,
    discriminating by the per-row ``Depth_source`` column. Anchors are
    rows where Depth_source == "vlm_depth_pro"; transit rows are
    everything else (typically "pending_interpolation").
    """
    ds = depth_source.astype(str)
    anchor_mask = ds == "vlm_depth_pro"
    transit_mask = ~anchor_mask
    if not np.any(anchor_mask):
        raise RuntimeError(
            "No anchor rows in the input NPZ (no row with "
            "Depth_source == 'vlm_depth_pro'). Re-record with "
            "GAZE_CALIBRATION_DEPTH_SOURCE='vlm_depth_pro' before "
            "running this tool."
        )
    if not np.any(transit_mask):
        raise RuntimeError(
            "No transit rows in the input NPZ — nothing to interpolate. "
            "(All rows are tagged 'vlm_depth_pro'; this only happens for "
            "anchor-only NPZs.)"
        )
    return anchor_mask, transit_mask


def _parse_leg_label(leg_label: str) -> Optional[Tuple[str, str]]:
    """Return (from_label, to_label) parsed from a transit leg label,
    or None if the label is not in the expected shape.
    """
    if not leg_label:
        return None
    m = _LEG_PATTERN.match(leg_label)
    if m is None:
        return None
    return m.group("from"), m.group("to")


def _build_anchor_lookup(
    target_label: np.ndarray, anchor_mask: np.ndarray
) -> Dict[str, int]:
    """Map ``Target_label`` → fit-block row index, restricted to anchor
    rows. Used to look up the from/to anchors for each transit leg.
    """
    tl = target_label.astype(str)
    lookup: Dict[str, int] = {}
    for i in np.flatnonzero(anchor_mask):
        label = tl[i]
        if label and label not in lookup:
            lookup[label] = int(i)
    return lookup


def _interpolate_transit_depths(
    X: np.ndarray,
    d_cm: np.ndarray,
    target_label: np.ndarray,
    leg_label: np.ndarray,
    anchor_mask: np.ndarray,
    transit_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill transit-row depths and return (d_cm_out, per_row_method).

    For each transit row:

    - If the row's leg parses to (from_label, to_label) and BOTH anchor
      labels exist in the anchor row set: linear interp by EE-distance
      ratio between the two anchors.
    - If only one of the brackets exists: use that anchor's depth
      (constant across the leg).
    - Otherwise: fall back to global KDTree NN over the anchor EE
      positions.

    ``per_row_method`` mirrors the shape of ``d_cm_out`` and carries the
    tag ("bracketed" | "single_bracket" | "nearest_anchor") used to
    drive Depth_source rewriting.
    """
    anchor_indices = np.flatnonzero(anchor_mask)
    anchor_X = X[anchor_indices]
    anchor_d = d_cm[anchor_indices]
    label_to_anchor_idx = _build_anchor_lookup(target_label, anchor_mask)

    # KDTree built once over anchor EE positions for the fallback path.
    tree = cKDTree(anchor_X)

    d_cm_out = d_cm.copy()
    methods = np.full(len(d_cm), "", dtype="<U24")

    transit_idx = np.flatnonzero(transit_mask)
    leg_str = leg_label.astype(str)

    for i in transit_idx:
        leg = _parse_leg_label(leg_str[i])
        x_i = X[i]
        d_from: Optional[float] = None
        d_to: Optional[float] = None
        x_from: Optional[np.ndarray] = None
        x_to: Optional[np.ndarray] = None
        if leg is not None:
            from_label, to_label = leg
            if from_label in label_to_anchor_idx:
                fi = label_to_anchor_idx[from_label]
                d_from = float(d_cm[fi])
                x_from = X[fi]
            if to_label in label_to_anchor_idx:
                ti = label_to_anchor_idx[to_label]
                d_to = float(d_cm[ti])
                x_to = X[ti]

        if d_from is not None and d_to is not None:
            assert x_from is not None and x_to is not None  # for mypy
            d_a = float(np.linalg.norm(x_i - x_from))
            d_b = float(np.linalg.norm(x_i - x_to))
            denom = d_a + d_b
            if denom <= 0.0 or not np.isfinite(denom):
                d_cm_out[i] = 0.5 * (d_from + d_to)
            else:
                t = d_a / denom
                d_cm_out[i] = (1.0 - t) * d_from + t * d_to
            methods[i] = "bracketed"
        elif d_from is not None:
            d_cm_out[i] = d_from
            methods[i] = "single_bracket"
        elif d_to is not None:
            d_cm_out[i] = d_to
            methods[i] = "single_bracket"
        else:
            _, nearest = tree.query(x_i, k=1)
            d_cm_out[i] = float(anchor_d[int(nearest)])
            methods[i] = "nearest_anchor"

    return d_cm_out, methods


# Per-method Depth_source tag written into the rewritten NPZ.
_METHOD_TO_TAG = {
    "bracketed": "vlm_interpolated_bracketed",
    "single_bracket": "vlm_interpolated_single_bracket",
    "nearest_anchor": "vlm_interpolated_nearest_anchor",
}


def _rewrite_npz(in_path: str, out_path: str) -> None:
    z = np.load(in_path, allow_pickle=True)
    try:
        if "Depth_source" not in z.files:
            raise KeyError(
                "Input NPZ has no Depth_source column. Re-record with "
                "the current harmony_free_arm_calibration.py — older "
                "REV01 NPZs are not compatible with this tool."
            )
        depth_source_in = np.asarray(z["Depth_source"])
        anchor_mask, transit_mask = _split_anchors_and_transit(depth_source_in)

        X = np.asarray(z["X"], dtype=float)
        d_cm = np.asarray(z["D_cm"], dtype=float)
        target_label = np.asarray(z["Target_label"])
        leg_label_key = "Leg_label" if "Leg_label" in z.files else None
        if leg_label_key is None:
            # Legacy NPZs without per-fit-row leg labels fall back to
            # nearest-anchor exclusively (the bracket parser cannot run).
            leg_label = np.full(len(X), "", dtype="<U64")
        else:
            leg_label = np.asarray(z[leg_label_key])

        d_cm_out, methods = _interpolate_transit_depths(
            X, d_cm, target_label, leg_label, anchor_mask, transit_mask
        )

        fields: Dict[str, Any] = {k: z[k] for k in z.files if k != "meta"}
        fields["D_cm"] = d_cm_out

        # Per-row Depth_source: anchor rows keep "vlm_depth_pro";
        # transit rows get the per-method interp tag.
        ds = depth_source_in.astype("<U64").copy()
        for i in np.flatnonzero(transit_mask):
            tag = _METHOD_TO_TAG.get(methods[i], "vlm_interpolated_nearest_anchor")
            ds[i] = tag
        fields["Depth_source"] = ds

        # Mirror the rewrite into the *_all block so downstream
        # diagnostics that index by Phase_all keep agreeing with
        # the fit block.
        if "Depth_source_all" in fields and "Phase_all" in fields:
            ds_all = np.asarray(fields["Depth_source_all"]).astype("<U64").copy()
            phase_all = np.asarray(fields["Phase_all"]).astype(str)
            # The fit-block rows are exactly the captured + transit
            # rows of the *_all block, in the same order. Build a
            # mapping by walking phase_all and copying the per-row
            # tag from the fit block.
            fit_idx = 0
            for j, ph in enumerate(phase_all):
                if ph in ("captured", "transit"):
                    if fit_idx < len(ds):
                        ds_all[j] = ds[fit_idx]
                    fit_idx += 1
            fields["Depth_source_all"] = ds_all

        meta_raw = z["meta"]
        if isinstance(meta_raw, np.ndarray) and meta_raw.dtype == object:
            meta = meta_raw.item()
        else:
            meta = dict(meta_raw)
        if not isinstance(meta, dict):
            raise RuntimeError(
                f"Input NPZ meta is not a dict (got {type(meta).__name__}); "
                f"refusing to rewrite."
            )
        meta = dict(meta)
        # Pin depth_source so the v2 runtime dispatches to vlm_depth_pro
        # (the runtime calls VLM per query at command time; the
        # interpolated transit depths are only used by the calibration
        # fit, not at runtime).
        meta["depth_source"] = "vlm_depth_pro"
        meta["affine_map"] = None
        # Record interp-method counts so the operator can audit how
        # many transit rows used each path.
        counts: Dict[str, int] = {}
        for i in np.flatnonzero(transit_mask):
            counts[methods[i]] = counts.get(methods[i], 0) + 1
        meta["transit_interp_method_counts"] = counts
        fields["meta"] = meta

        np.savez_compressed(out_path, **fields)

        # Operator summary
        n_transit = int(transit_mask.sum())
        n_anchor = int(anchor_mask.sum())
        print(f"  anchors:           {n_anchor}")
        print(f"  transit rows:      {n_transit}")
        for m_name in ("bracketed", "single_bracket", "nearest_anchor"):
            print(f"    via {m_name:<18s} {counts.get(m_name, 0)}")
        print(f"  wrote: {out_path}")
        print(f"  meta['depth_source'] = 'vlm_depth_pro'")
    finally:
        z.close()


def _derive_out_path(in_path: str) -> str:
    """``foo.npz`` → ``foo_interp.npz``."""
    if in_path.lower().endswith(".npz"):
        return in_path[:-4] + "_interp.npz"
    return in_path + "_interp.npz"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fill transit-row depths from bracketing anchor depths "
                    "(vlm-only calibration NPZ)."
    )
    parser.add_argument("input", help="Path to the vlm-only calibration NPZ.")
    parser.add_argument("--out", default=None,
                        help="Output NPZ path (default: <input>_interp.npz).")
    args = parser.parse_args(argv)

    out_path = (args.out if args.out is not None
                else _derive_out_path(args.input))
    _rewrite_npz(args.input, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
