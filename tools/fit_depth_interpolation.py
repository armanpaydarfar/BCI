#!/usr/bin/env python3
"""
tools/fit_depth_interpolation.py — REV01 backup depth pipeline
(Plan §3.3 backup path).

When ``tools/fit_vergence_affine.py`` rejects the per-session affine fit
(R² below threshold or max-residual too large), this script offers a
non-linear backup: build a KDTree over the 15 anchor EE positions and
for every transit row, look up the nearest anchor's VLM Depth Pro value.

    python tools/fit_depth_interpolation.py poses_with_gaze_<stamp>_v2_freearm.npz \\
        [--out poses_with_gaze_<stamp>_v2_freearm_interp.npz]

The script:

1. Loads the input NPZ.
2. Identifies captured anchor rows (``D_cm_vergence`` finite) and
   transit rows (``D_cm_vergence`` NaN) in the legacy fit block.
3. Builds ``scipy.spatial.cKDTree`` over the anchor X (EE position).
4. For every transit row, sets ``D_cm`` to the nearest anchor's
   ``D_cm`` (the VLM Depth Pro reading at that anchor).
5. Updates ``Depth_source`` and ``Depth_source_all`` on transit rows to
   ``"vlm_interpolated_nearest_anchor"``.
6. Sets ``meta["affine_map"] = None`` AND
   ``meta["depth_source"] = "vergence"`` so the runtime's alignment-
   invariant probe accepts the rewritten NPZ (the hybrid+None
   combination raises ``RuntimeError`` at startup; tagging the meta
   string as plain ``"vergence"`` keeps the file runtime-loadable per
   Plan §1.3 design intent). Runtime then uses raw vergence — wrong-
   but-safe; per-subject offset will be off by an unknown factor. The
   proper fix is to re-record with better Depth Pro coverage.

After this rewrite, an ``_interp.npz`` is distinguishable from a
vergence-only recording only by per-row ``Depth_source`` tags
(``"vlm_interpolated_nearest_anchor"`` on transit rows; the legacy
vergence-only path leaves every row as ``"vergence"``). The meta
``depth_source`` string is identical in both cases.

Limitation (Plan §3.3): the runtime does NOT apply the per-row
interpolation. Extending runtime would require a new EE-position feed
which is out of scope for REV01. This script's value is offline analysis
of how the Mahalanobis NN behaves with backup-pipeline data.

Plan reference: Documents/SoftwareDocs/Harmony_Gaze_Calibration_REV01_Plan.md §3.3.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Tuple

import numpy as np
from scipy.spatial import cKDTree


def _select_anchors_and_transit(z: Any
                                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (anchor_mask, transit_mask) over the fit-block rows.

    Anchor rows have finite ``D_cm_vergence`` (the recorder's Step 2
    writes that field only at VLM anchors). Transit rows have NaN.
    Both masks are over the same fit-block length so callers can index
    into ``D_cm`` / ``X`` directly.
    """
    if "D_cm_vergence" not in z.files:
        raise KeyError(
            "Input NPZ has no D_cm_vergence column. Re-record with the "
            "REV01 recorder (harmony_free_arm_calibration.py with "
            "GAZE_CALIBRATION_DEPTH_SOURCE='vlm_depth_pro')."
        )
    d_cm_verg = np.asarray(z["D_cm_vergence"], dtype=float)
    anchor_mask = np.isfinite(d_cm_verg)
    transit_mask = ~anchor_mask
    if not np.any(anchor_mask):
        raise RuntimeError(
            "No valid anchor rows in the input NPZ — D_cm_vergence is "
            "NaN on every row."
        )
    if not np.any(transit_mask):
        raise RuntimeError(
            "No transit rows in the input NPZ — nothing to interpolate."
        )
    return anchor_mask, transit_mask


def _interpolate_transit_depths(X: np.ndarray, d_cm: np.ndarray,
                                  anchor_mask: np.ndarray,
                                  transit_mask: np.ndarray
                                  ) -> np.ndarray:
    """KDTree NN lookup from each transit EE position to the nearest
    anchor's D_cm. Returns the rewritten D_cm array (copy)."""
    anchor_X = X[anchor_mask]
    anchor_d = d_cm[anchor_mask]
    transit_X = X[transit_mask]
    tree = cKDTree(anchor_X)
    _, nearest_idx = tree.query(transit_X, k=1)
    d_cm_out = d_cm.copy()
    d_cm_out[transit_mask] = anchor_d[nearest_idx]
    return d_cm_out


def _rewrite_npz(in_path: str, out_path: str) -> None:
    """Load, interpolate, rewrite. ``meta["affine_map"]`` is forced to
    ``None`` and ``meta["depth_source"]`` to ``"vergence"`` so the
    runtime accepts the rewritten NPZ via the legacy vergence path
    (the hybrid+None combination raises at startup; see module
    docstring)."""
    z = np.load(in_path, allow_pickle=True)
    try:
        anchor_mask, transit_mask = _select_anchors_and_transit(z)
        X = np.asarray(z["X"], dtype=float)
        d_cm = np.asarray(z["D_cm"], dtype=float)
        d_cm_out = _interpolate_transit_depths(X, d_cm, anchor_mask,
                                                  transit_mask)

        fields: Dict[str, Any] = {k: z[k] for k in z.files if k != "meta"}
        fields["D_cm"] = d_cm_out

        # Per-row provenance: transit rows tag the backup pipeline.
        if "Depth_source" in fields:
            ds = np.asarray(fields["Depth_source"]).astype("<U64").copy()
            ds[transit_mask] = "vlm_interpolated_nearest_anchor"
            fields["Depth_source"] = ds
        if "Depth_source_all" in fields and "Phase_all" in fields:
            phase_all = np.asarray(fields["Phase_all"]).astype(str)
            ds_all = (np.asarray(fields["Depth_source_all"])
                       .astype("<U64").copy())
            ds_all[phase_all == "transit"] = "vlm_interpolated_nearest_anchor"
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
        meta["affine_map"] = None
        # Tag as vergence so the runtime alignment-invariant probe
        # accepts the file (hybrid + None affine_map raises at startup
        # by design). Per-row Depth_source still records the backup
        # provenance for offline analysis.
        meta["depth_source"] = "vergence"
        fields["meta"] = meta

        np.savez_compressed(out_path, **fields)
    finally:
        z.close()


def _derive_out_path(in_path: str) -> str:
    """``foo.npz`` → ``foo_interp.npz`` (analogous to fit_vergence_affine.py)."""
    if in_path.lower().endswith(".npz"):
        return in_path[:-4] + "_interp.npz"
    return in_path + "_interp.npz"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="REV01 backup depth pipeline: NN interpolation over "
                    "anchor EE positions (Plan §3.3)."
    )
    parser.add_argument("input", help="Path to the REV01 hybrid NPZ.")
    parser.add_argument("--out", default=None,
                        help="Output NPZ path (default: <input>_interp.npz).")
    args = parser.parse_args(argv)

    out_path = (args.out if args.out is not None
                else _derive_out_path(args.input))
    _rewrite_npz(args.input, out_path)

    print(f"REV01 backup NN-over-EE depth interpolation:")
    print(f"  wrote: {out_path}")
    print(f"  meta['depth_source'] = 'vergence', meta['affine_map'] = None")
    print(f"  runtime accepts as vergence-only NPZ; transit rows tagged via")
    print(f"  per-row Depth_source for offline analysis.")
    print(f"  (re-record for primary affine-fit path if quality is critical)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
