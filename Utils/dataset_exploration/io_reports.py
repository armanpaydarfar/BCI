"""
Write exploration outputs: parquet (optional), CSV, text proposals, JSON config snapshot.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def write_df(df: pd.DataFrame, path_base: Path, *, basename: str) -> tuple[str, str | None]:
    """
    Write CSV and optional Parquet. Returns (csv_path, parquet_path_or_none).
    """
    path_base.mkdir(parents=True, exist_ok=True)
    csv_path = path_base / f"{basename}.csv"
    df.to_csv(csv_path, index=False)
    pq_path = path_base / f"{basename}.parquet"
    pq_written: str | None = None
    try:
        df.to_parquet(pq_path, index=False)
        pq_written = str(pq_path)
    except Exception:
        pass
    return str(csv_path), pq_written


def _fmt_num(x: Any, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def write_proposed_additions(
    ranked: pd.DataFrame,
    path: Path,
    *,
    max_lines: int = 200,
    require_positive_composite: bool = True,
    primary_backend: str = "mdm",
    run_meta: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append("# Proposed .xdf files to add to expert decoder training pool")
    lines.append(f"# Generated: {ts}")
    if run_meta:
        lines.append(f"# data_dir={run_meta.get('data_dir', '')}")
        lines.append(f"# baseline_subject={run_meta.get('baseline_subject', '')}  backends={run_meta.get('backends', '')}")
        lines.append(f"# primary_backend={run_meta.get('primary_backend', primary_backend)}  score_threshold={run_meta.get('score_threshold', '')}")
    lines.append("#")
    lines.append("# Section 1: filenames only (positive composite_score, capped; deduped by filename)")
    lines.append("")

    sub_filtered = ranked
    if require_positive_composite and "composite_score" in sub_filtered.columns:
        sub_filtered = sub_filtered[sub_filtered["composite_score"] > 0]

    # Section 1 is "filenames only"; dedupe by filename to avoid confusing repeats.
    # (Section 2 still includes `path` so you can disambiguate if needed.)
    seen_filenames: set[str] = set()
    selected_rows: list[pd.Series] = []
    for _, row in sub_filtered.iterrows():
        fn = row.get("filename", Path(str(row.get("path", ""))).name)
        fn_s = str(fn)
        if fn_s in seen_filenames:
            continue
        seen_filenames.add(fn_s)
        selected_rows.append(row)
        if len(selected_rows) >= max_lines:
            break

    sub = pd.DataFrame(selected_rows)
    for _, row in sub.iterrows():
        fn = row.get("filename", Path(str(row.get("path", ""))).name)
        lines.append(str(fn))

    auc_col = f"{primary_backend}__auc"
    lines.append("")
    lines.append("# Section 2: ranked detail (tab-separated; ERD columns reflect strength + contra-vs-ipsi)")
    hdr = (
        "rank\tfilename\tsubject\tmodality\tcomposite\t"
        f"{primary_backend}_auc\tbal_acc\tfisher_mu\terd_strength\terd_contra_minus_ipsi\terd_rest_leak\terd_pattern\trest_leak_penalty\tchannel_ok\tflags\tpath"
    )
    lines.append(hdr)
    bal_col = f"{primary_backend}__balanced_accuracy_argmax"
    for _, row in sub.iterrows():
        pr = int(row["proposal_rank"]) if "proposal_rank" in row and pd.notna(row["proposal_rank"]) else 0
        fn = str(row.get("filename", ""))
        fl = []
        for k in (
            "flag_strength_individual",
            "flag_coverage_complement",
            "flag_risk_harmful",
            "flag_insufficient_windows",
            "flag_strong_erd",
            "flag_erd_rest_leak",
            "flag_good_surface_poor_separability",
            "flag_expensive_full_eval",
        ):
            if k in row and bool(row[k]):
                fl.append(k.replace("flag_", ""))
        ch_ok = row.get("channel_match_baseline", "")
        lines.append(
            f"{pr}\t{fn}\t{row.get('subject', '')}\t{row.get('modality', '')}\t"
            f"{_fmt_num(row.get('composite_score'))}\t{_fmt_num(row.get(auc_col))}\t"
            f"{_fmt_num(row.get(bal_col))}\t{_fmt_num(row.get('fisher_tangent_mu'))}\t"
            f"{_fmt_num(row.get('erd_strength_total'))}\t{_fmt_num(row.get('erd_contra_minus_ipsi'))}\t"
            f"{_fmt_num(row.get('erd_rest_leak_strength'))}\t{row.get('erd_pattern', '—')}\t"
            f"{_fmt_num(row.get('erd_rest_leak_penalty'))}\t{ch_ok}\t"
            f"{','.join(fl) if fl else '—'}\t{row.get('path', '')}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_summary(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
