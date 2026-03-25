"""
Compare candidates to baseline reference; composite score and categorical flags.

Current ERD scoring uses physiology-aware topography metrics:
  - erd_strength_total
  - erd_contra_minus_ipsi
and no longer uses legacy erd_mean_abs / erd_cohen_d_max in composite_score.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd


def _z_higher_better(x: float, ref: dict[str, float]) -> float:
    if math.isnan(x):
        return 0.0
    iqr = ref.get("iqr", 1.0) or 1e-9
    return (x - ref["median"]) / iqr


def _z_lower_better(x: float, ref: dict[str, float]) -> float:
    if math.isnan(x):
        return 0.0
    iqr = ref.get("iqr", 1.0) or 1e-9
    return (ref["median"] - x) / iqr


def add_scores_and_flags(
    df: pd.DataFrame,
    ref: dict[str, dict[str, float]],
    *,
    primary_backend: str,
    baseline_channel_counts: list[int],
    pool_subject: str,
    skew_penalty_lambda: float = 0.5,
    frac_thresh_col_suffix: str = "0p5",
    erd_rest_leak_threshold: float = 0.03,
    erd_rest_leak_penalty: float = 2.0,
) -> pd.DataFrame:
    """
    Mutates a copy of df with composite_score, dist_shape_score, z_* columns, and boolean flags.
    """
    out = df.copy()

    pb = primary_backend.strip().lower()
    frac_col = f"{pb}__frac_correct_prob_above_{frac_thresh_col_suffix}"
    skew_col = f"{pb}__skew_p_mi"

    ref_skew_p95 = ref.get("_ref_abs_skew_p_mi_p95", {}).get("p95", 2.0)

    z_cols: list[tuple[str, str]] = []  # (new_col, direction h|l)

    metric_directions: list[tuple[str, str]] = [
        ("fisher_tangent_mu", "h"),
        ("tangent_mean_l2_distance", "h"),
        ("erd_strength_total", "h"),
        ("erd_contra_minus_ipsi", "h"),
        (f"{pb}__auc", "h"),
        (f"{pb}__balanced_accuracy_argmax", "h"),
        (f"{pb}__accuracy_argmax", "h"),
        (f"{pb}__brier_pmi", "l"),
    ]
    if f"{pb}__decided_balanced_accuracy_mean" in out.columns:
        metric_directions.append((f"{pb}__decided_balanced_accuracy_mean", "h"))
    if f"{pb}__ambiguous_rate_mean" in out.columns:
        metric_directions.append((f"{pb}__ambiguous_rate_mean", "l"))

    for col, direction in metric_directions:
        if col not in out.columns or col not in ref:
            continue
        zname = f"z__{col}"
        r = ref[col]
        if direction == "h":
            out[zname] = out[col].apply(lambda v, rr=r: _z_higher_better(float(v), rr))
        else:
            out[zname] = out[col].apply(lambda v, rr=r: _z_lower_better(float(v), rr))
        z_cols.append((zname, direction))

    # Distribution shape score
    def _dist_shape_row(row: pd.Series) -> float:
        fc = float(row[frac_col]) if frac_col in row.index and pd.notna(row[frac_col]) else float("nan")
        sk = float(row[skew_col]) if skew_col in row.index and pd.notna(row[skew_col]) else float("nan")
        z_fc = _z_higher_better(fc, ref.get(frac_col, {"median": 0.5, "iqr": 0.2, "q1": 0.4, "q3": 0.6}))
        skew_excess = max(0.0, abs(sk) - ref_skew_p95) if not math.isnan(sk) else 0.0
        return z_fc - skew_penalty_lambda * skew_excess

    if frac_col in out.columns:
        out["dist_shape_score"] = out.apply(_dist_shape_row, axis=1)
    else:
        out["dist_shape_score"] = np.nan

    # Composite (unweighted mean of block means of z-scores):
    # quality + geometry/separability + ERD(topography/strength).
    z_quality = [c for c, _ in z_cols if c.startswith("z__") and pb in c]
    z_geom = [c for c, _ in z_cols if "fisher" in c or "tangent_mean" in c]
    z_erd = [c for c, _ in z_cols if "erd_" in c]

    def _row_block_mean(row: pd.Series, cols: list[str]) -> float:
        vals = [float(row[c]) for c in cols if c in row.index and pd.notna(row[c])]
        return float(np.mean(vals)) if vals else float("nan")

    parts: list[pd.Series] = []
    if z_quality:
        s = out.apply(lambda r, cols=z_quality: _row_block_mean(r, cols), axis=1)
        out["_z_quality_block"] = s
        parts.append(s)
    if z_geom:
        s = out.apply(lambda r, cols=z_geom: _row_block_mean(r, cols), axis=1)
        out["_z_separability_block"] = s
        parts.append(s)
    if z_erd:
        s = out.apply(lambda r, cols=z_erd: _row_block_mean(r, cols), axis=1)
        out["_z_erd_block"] = s
        parts.append(s)

    if parts:
        mat = np.column_stack([p.values for p in parts])
        out["composite_score"] = np.nanmean(mat, axis=1)
    elif "dist_shape_score" in out.columns:
        out["composite_score"] = out["dist_shape_score"]
    else:
        out["composite_score"] = np.nan

    # Hard penalty: ERD leakage during REST class should strongly down-rank a file.
    if "erd_rest_leak_strength" in out.columns:
        leak = pd.to_numeric(out["erd_rest_leak_strength"], errors="coerce")
        leak_flag = leak >= float(erd_rest_leak_threshold)
        out["erd_rest_leak_penalty"] = leak_flag.fillna(False).astype(float) * float(erd_rest_leak_penalty)
        out["composite_score"] = out["composite_score"] - out["erd_rest_leak_penalty"]
    else:
        out["erd_rest_leak_penalty"] = 0.0

    # Channel compatibility
    if baseline_channel_counts and "n_channels" in out.columns:
        ref_set = set(baseline_channel_counts)
        out["channel_match_baseline"] = out["n_channels"].apply(lambda n: int(n) in ref_set if pd.notna(n) else False)
    else:
        out["channel_match_baseline"] = True

    # Flags
    def _flags(row: pd.Series) -> dict[str, bool]:
        subj = row.get("subject")
        cov_ok = (
            isinstance(subj, str)
            and subj
            and subj.upper() != str(pool_subject).upper()
        )

        auc = float(row.get(f"{pb}__auc", np.nan))
        fisher = float(row.get("fisher_tangent_mu", np.nan))
        dist_s = float(row.get("dist_shape_score", np.nan))

        strong_erd = float(row.get("erd_strength_total", 0.0) or 0) >= float(
            ref.get("erd_strength_total", {}).get("median", 0.0)
        )
        leak_v = row.get("erd_rest_leak_strength", np.nan)
        rest_leak_flag = (
            bool(float(leak_v) >= float(erd_rest_leak_threshold))
            if pd.notna(leak_v)
            else False
        )

        ev = row.get("eval_error", np.nan)
        ev_s = "" if pd.isna(ev) else str(ev).strip()
        insufficient = ev_s.startswith("below_min_windows")
        parse_failed = bool(ev_s) and not insufficient
        harmful = parse_failed or (
            not math.isnan(auc) and auc < 0.52 and float(row.get("n_windows", 0) or 0) > 80
        )

        surface_sep = (not math.isnan(dist_s) and dist_s > 0) and (
            not math.isnan(fisher) and fisher < ref.get("fisher_tangent_mu", {}).get("q1", -np.inf)
        )

        comp = float(row.get("composite_score", np.nan))
        individual = (not math.isnan(comp) and comp > 0) and not harmful

        expensive = float(row.get("n_windows", 0) or 0) > 5000

        return {
            "flag_strength_individual": bool(individual),
            "flag_coverage_complement": bool(cov_ok),
            "flag_risk_harmful": bool(harmful),
            "flag_insufficient_windows": bool(insufficient),
            "flag_good_surface_poor_separability": bool(surface_sep),
            "flag_strong_erd": bool(strong_erd),
            "flag_erd_rest_leak": bool(rest_leak_flag),
            "flag_expensive_full_eval": bool(expensive),
        }

    flag_df = out.apply(_flags, axis=1, result_type="expand")
    out = pd.concat([out, flag_df], axis=1)

    return out


def rank_proposals(
    df: pd.DataFrame,
    *,
    exclude_in_pool: bool = True,
    min_composite: float | None = None,
    composite_column: str = "composite_score",
    rank_column: str = "proposal_rank",
) -> pd.DataFrame:
    """Sort by composite column descending; exclude files already in baseline pool."""
    d = df.copy()
    if exclude_in_pool and "in_baseline_pool" in d.columns:
        d = d[~d["in_baseline_pool"].astype(bool)]
    if "eval_error" in d.columns:
        ev = d["eval_error"]
        d = d[ev.isna() | (ev.astype(str).str.strip() == "")]
    if composite_column not in d.columns:
        raise KeyError(f"rank_proposals: missing column {composite_column!r}")
    d = d.sort_values(composite_column, ascending=False, na_position="last")
    if min_composite is not None:
        d = d[d[composite_column] >= min_composite]
    d = d.copy()
    d[rank_column] = np.arange(1, len(d) + 1)
    return d.reset_index(drop=True)


def merge_multi_primary_scored(
    base: pd.DataFrame,
    scored_by_primary: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge per-primary scoring into one wide table for candidates.csv.

    Per-primary columns (composite, dist_shape, block z's, flags) get __{primary} suffix.
    z__ columns that are shared across primaries (e.g. z__fisher_tangent_mu) are kept once.
    """
    out = base.copy()
    suffix_always = {
        "composite_score",
        "dist_shape_score",
        "_z_quality_block",
        "_z_separability_block",
        "_z_erd_block",
    }
    for primary, sp in scored_by_primary.items():
        for c in sp.columns:
            if c in base.columns:
                continue
            if c in suffix_always:
                out[f"{c}__{primary}"] = sp[c]
            elif c.startswith("flag_"):
                out[f"{c}__{primary}"] = sp[c]
            elif c.startswith("z__"):
                if c not in out.columns:
                    out[c] = sp[c]
            elif c not in out.columns:
                out[c] = sp[c]
    first = next(iter(scored_by_primary.values()))
    if "channel_match_baseline" in first.columns:
        out["channel_match_baseline"] = first["channel_match_baseline"]
    return out


def report_primary_backends(backends: list[str]) -> list[str]:
    """
    One entry per decoder in `--backends`, in the order given (deduped).

    Each backend gets its own scoring, ranking, and proposal files — no merged ranking
    across models. With a single backend, that list has length 1.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for x in backends:
        b = x.strip().lower()
        if not b or b in seen:
            continue
        seen.add(b)
        ordered.append(b)
    if not ordered:
        raise ValueError("backends list is empty after parsing")
    return ordered
