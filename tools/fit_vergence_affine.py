#!/usr/bin/env python3
"""
tools/fit_vergence_affine.py — Per-session affine fit from vergence to
Depth Pro at the 15 calibration anchors (REV01 Plan §3.3 primary path).

Operator workflow (after recording a REV01 hybrid NPZ via
``harmony_free_arm_calibration.py`` with
``GAZE_CALIBRATION_DEPTH_SOURCE="vlm_depth_pro"``):

    python tools/fit_vergence_affine.py poses_with_gaze_<stamp>_v2_freearm.npz \\
        [--out poses_with_gaze_<stamp>_v2_freearm_affine.npz] \\
        [--plot fit.png] [--min-r2 0.85] [--max-residual-cm 8.0]

The script:

1. Loads the input NPZ.
2. Selects the captured anchor rows (``Phase_all == "captured"``,
   ``D_valid_all == True``).
3. Solves ``D_vlm = a · D_vergence + b`` via least squares using the
   anchor pairs (D_cm at anchors is the VLM Depth Pro reading from
   ``settle_and_snapshot``; D_cm_vergence carries the parallel vergence
   reading preserved at recorder time).
4. Computes R² and max-abs residual in cm.
5. Applies the affine map to every transit row's D_cm (the legacy
   block's transit rows; anchor rows are unchanged). Updates
   ``Depth_source`` for those rows to ``"vergence_affine"``.
6. Updates ``meta["affine_map"]`` to a dict ``{a, b, R2,
   max_abs_residual_cm}`` and writes a new NPZ alongside the input.
7. Exits non-zero if R² < ``--min-r2`` or
   max_abs_residual_cm > ``--max-residual-cm`` — the operator's signal
   to either tighten the inputs or fall through to the backup pipeline
   at ``tools/fit_depth_interpolation.py``.

Operator-run only; the runtime path is untouched. The runtime reads
the rewritten NPZ via ``config.POSE_LIBRARY_PATH`` (set in
``config_local.py``) and dispatches on ``runtime_depth_pipeline``
(``Utils/gaze/calibration_mapping.py``).

Plan reference: Documents/SoftwareDocs/Harmony_Gaze_Calibration_REV01_Plan.md §3.3.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Tuple

import numpy as np


# Defaults locked in plan §6.4 #2.
DEFAULT_MIN_R2: float = 0.85
DEFAULT_MAX_RESIDUAL_CM: float = 8.0


def _fit_affine(d_vergence: np.ndarray, d_vlm: np.ndarray
                ) -> Tuple[float, float, float, float]:
    """Solve D_vlm = a * D_vergence + b via least squares.

    Returns ``(a, b, R2, max_abs_residual_cm)``. R² uses the standard
    1 - SS_res / SS_tot definition. When SS_tot == 0 (degenerate input:
    all D_vlm values equal) R² is reported as ``1.0`` if SS_res is
    also 0, otherwise ``0.0`` — the operator's --min-r2 still gates
    pass/fail at the CLI layer.
    """
    if d_vergence.shape != d_vlm.shape:
        raise ValueError(
            f"d_vergence ({d_vergence.shape}) and d_vlm ({d_vlm.shape}) "
            f"must have matching shapes."
        )
    if d_vergence.size < 2:
        raise ValueError(
            f"Need >= 2 anchor pairs to fit; got {d_vergence.size}."
        )
    A = np.column_stack([d_vergence, np.ones_like(d_vergence)])
    coeffs, *_ = np.linalg.lstsq(A, d_vlm, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])
    pred = a * d_vergence + b
    residuals = d_vlm - pred
    ss_res = float(np.sum(residuals * residuals))
    ss_tot = float(np.sum((d_vlm - np.mean(d_vlm)) ** 2))
    if ss_tot == 0.0:
        r2 = 1.0 if ss_res == 0.0 else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot
    max_abs_residual_cm = float(np.max(np.abs(residuals)))
    return a, b, r2, max_abs_residual_cm


def _select_anchor_pairs(z: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Pull (D_cm_vergence, D_cm) anchor pairs out of the NPZ.

    Reads the captured-anchor rows directly from the fit block (legacy
    keys), filtered to D_valid True and finite D_cm_vergence (the
    vergence-anchor parallel reading is NaN for non-anchor rows and
    for any anchor recorded without VLM substitution). Raises
    ``KeyError`` on the missing-D_cm_vergence column case so the
    operator gets a clear error instead of an empty fit.
    """
    if "D_cm_vergence" not in z.files:
        raise KeyError(
            "Input NPZ has no D_cm_vergence column. Re-record with the "
            "REV01 recorder (harmony_free_arm_calibration.py with "
            "GAZE_CALIBRATION_DEPTH_SOURCE='vlm_depth_pro')."
        )
    d_cm = np.asarray(z["D_cm"], dtype=float)
    d_cm_verg = np.asarray(z["D_cm_vergence"], dtype=float)
    d_valid = np.asarray(z["D_valid"], dtype=bool)
    # Anchor rows are exactly those where D_cm_vergence is finite —
    # transit rows have NaN there by construction.
    anchor_mask = np.isfinite(d_cm_verg) & np.isfinite(d_cm) & d_valid
    if not np.any(anchor_mask):
        raise RuntimeError(
            "No valid anchor pairs in the input NPZ — D_cm_vergence is "
            "NaN on every row. Was the session recorded with "
            "GAZE_CALIBRATION_DEPTH_SOURCE='vlm_depth_pro'?"
        )
    return d_cm_verg[anchor_mask], d_cm[anchor_mask]


def _rewrite_npz(in_path: str, out_path: str, *, a: float, b: float,
                 r2: float, max_abs_residual_cm: float) -> None:
    """Load the input NPZ, apply the affine map to transit rows, update
    meta, and write to out_path. Preserves every other key unchanged.
    """
    z = np.load(in_path, allow_pickle=True)
    fields: Dict[str, Any] = {k: z[k] for k in z.files if k != "meta"}
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

    # Apply affine map to transit rows of D_cm (legacy block). Anchor
    # rows have finite D_cm_vergence; transit rows have NaN there —
    # the same mask we used to identify anchors above flips for
    # transit.
    d_cm = np.asarray(fields["D_cm"], dtype=float).copy()
    d_cm_verg = np.asarray(fields["D_cm_vergence"], dtype=float)
    transit_mask = ~np.isfinite(d_cm_verg)
    d_cm[transit_mask] = a * d_cm[transit_mask] + b
    fields["D_cm"] = d_cm

    # Update per-row Depth_source: transit rows get "vergence_affine".
    if "Depth_source" in fields:
        depth_source = np.asarray(fields["Depth_source"]).astype("<U64").copy()
        depth_source[transit_mask] = "vergence_affine"
        fields["Depth_source"] = depth_source
    # Mirror the rewrite into Depth_source_all: locate rows where
    # Phase_all == "transit" and tag them too.
    if "Depth_source_all" in fields and "Phase_all" in fields:
        phase_all = np.asarray(fields["Phase_all"]).astype(str)
        depth_source_all = (
            np.asarray(fields["Depth_source_all"]).astype("<U64").copy()
        )
        depth_source_all[phase_all == "transit"] = "vergence_affine"
        fields["Depth_source_all"] = depth_source_all

    meta = dict(meta)
    meta["affine_map"] = {
        "a": float(a), "b": float(b),
        "R2": float(r2),
        "max_abs_residual_cm": float(max_abs_residual_cm),
    }
    fields["meta"] = meta

    np.savez_compressed(out_path, **fields)
    z.close()


def _maybe_plot(d_vergence: np.ndarray, d_vlm: np.ndarray, *,
                a: float, b: float, r2: float, max_abs_residual_cm: float,
                out_path: str) -> None:
    """Scatter + fit-line + residual histogram. Defers matplotlib import
    so the script's no-plot path stays headless-friendly."""
    import matplotlib  # noqa: WPS433 — local import is intentional
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    fig, (ax_scatter, ax_resid) = plt.subplots(1, 2, figsize=(10, 4))

    ax_scatter.scatter(d_vergence, d_vlm, label="anchor pairs", alpha=0.7)
    x_line = np.linspace(d_vergence.min(), d_vergence.max(), 50)
    ax_scatter.plot(x_line, a * x_line + b, "r-",
                    label=f"fit: D_vlm = {a:.4f} · D_vergence + {b:.4f}")
    ax_scatter.set_xlabel("D_vergence (cm)")
    ax_scatter.set_ylabel("D_vlm (cm)")
    ax_scatter.set_title(f"REV01 affine fit (R² = {r2:.3f})")
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.legend()

    residuals = d_vlm - (a * d_vergence + b)
    ax_resid.hist(residuals, bins=max(5, len(residuals) // 2))
    ax_resid.axvline(0.0, color="k", linestyle="--", alpha=0.5)
    ax_resid.set_xlabel("residual (cm)")
    ax_resid.set_ylabel("count")
    ax_resid.set_title(
        f"residuals (max |Δ| = {max_abs_residual_cm:.2f} cm)"
    )
    ax_resid.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _derive_out_path(in_path: str) -> str:
    """``foo.npz`` → ``foo_affine.npz`` per plan §3.3."""
    if in_path.lower().endswith(".npz"):
        return in_path[:-4] + "_affine.npz"
    return in_path + "_affine.npz"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit per-session vergence -> Depth Pro affine map "
                    "(REV01 Plan §3.3)."
    )
    parser.add_argument("input", help="Path to the REV01 hybrid NPZ.")
    parser.add_argument("--out", default=None,
                        help="Output NPZ path (default: <input>_affine.npz).")
    parser.add_argument("--plot", default=None,
                        help="Optional diagnostic plot path (PNG).")
    parser.add_argument("--min-r2", type=float, default=DEFAULT_MIN_R2,
                        help=f"Reject fit if R² < this (default: {DEFAULT_MIN_R2}).")
    parser.add_argument("--max-residual-cm", type=float,
                        default=DEFAULT_MAX_RESIDUAL_CM,
                        help=f"Reject fit if max |Δ| > this cm "
                              f"(default: {DEFAULT_MAX_RESIDUAL_CM}).")
    args = parser.parse_args(argv)

    z = np.load(args.input, allow_pickle=True)
    try:
        d_vergence, d_vlm = _select_anchor_pairs(z)
    finally:
        z.close()

    a, b, r2, max_abs_residual_cm = _fit_affine(d_vergence, d_vlm)
    print(f"REV01 affine fit:")
    print(f"  anchor pairs:        {d_vergence.size}")
    print(f"  D_vlm = {a:.6f} * D_vergence + {b:.6f}")
    print(f"  R²:                  {r2:.4f}")
    print(f"  max |Δ| (residual):  {max_abs_residual_cm:.4f} cm")

    out_path = args.out if args.out is not None else _derive_out_path(args.input)
    _rewrite_npz(args.input, out_path, a=a, b=b, r2=r2,
                  max_abs_residual_cm=max_abs_residual_cm)
    print(f"  wrote: {out_path}")

    if args.plot is not None:
        _maybe_plot(d_vergence, d_vlm, a=a, b=b, r2=r2,
                     max_abs_residual_cm=max_abs_residual_cm,
                     out_path=args.plot)
        print(f"  plot:  {args.plot}")

    # Gate on the accept thresholds (Plan §6.4 #2). Exits non-zero so
    # the operator's calling script can branch on this.
    if r2 < args.min_r2:
        print(f"REJECT: R² {r2:.4f} < --min-r2 {args.min_r2}", file=sys.stderr)
        return 2
    if max_abs_residual_cm > args.max_residual_cm:
        print(f"REJECT: max |Δ| {max_abs_residual_cm:.4f} cm > "
              f"--max-residual-cm {args.max_residual_cm}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
